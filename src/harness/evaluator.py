"""Quality evaluator for root report versions produced by the harness."""

from __future__ import annotations

from collections.abc import Callable
import contextlib
from datetime import datetime, timezone
import difflib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from src.harness.artifacts import (
    agent_workspace_paths,
    create_agent_workspace,
    initialize_run_root,
    latest_agent_records,
    load_request,
    read_jsonl,
    registry_paths,
    write_json_atomic,
    write_text_atomic,
)
from src.harness.presets import default_tool_allowlist, visible_skills_for_preset
from src.harness.prompt_builder import render_tools_markdown
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.types import HarnessRequest, HarnessResponse
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model


AREA_WEIGHTS: dict[str, float] = {
    "factual_correctness": 0.22,
    "freshness": 0.16,
    "evidence_grounding": 0.18,
    "logical_soundness": 0.14,
    "completeness": 0.12,
    "numerical_discipline": 0.10,
    "decision_usefulness": 0.08,
}
AREA_REQUIREMENTS: dict[str, list[str]] = {
    "factual_correctness": [
        "Material factual claims are true relative to cited or available run evidence.",
        "Contradicted market, company, filing, or macro facts are treated as critical failures.",
    ],
    "freshness": [
        "Time-sensitive market, filing, commodity, and macro data include clear as-of dates.",
        "The report distinguishes current data from stale historical context.",
    ],
    "evidence_grounding": [
        "Important claims that affect the investment view are backed by sources, artifacts, or child outputs.",
        "Unsupported material claims are called out rather than rewarded for confident prose.",
    ],
    "logical_soundness": [
        "Investment conclusions follow from valuation, cash-flow, commodity, macro, and risk evidence.",
        "The report does not use true but insufficient facts as a valuation bridge.",
    ],
    "completeness": [
        "The answer covers the user's requested valuation, risks, bull/bear case, macro, and commodity dimensions.",
        "Missing requested topics materially reduce the score.",
    ],
    "numerical_discipline": [
        "Ratios, percentages, prices, dates, and units are internally consistent and plausible.",
        "Visible math uses coherent numerators, denominators, and units.",
    ],
    "decision_usefulness": [
        "The report gives a clear view, key uncertainties, and what would change the conclusion.",
        "The risk/reward framing is actionable rather than only descriptive.",
    ],
}
CLAIM_STATUSES = {
    "supported",
    "partially_supported",
    "unsupported",
    "contradicted",
    "stale",
    "not_checkable",
}
IMPORTANCE_LEVELS = {"critical", "high", "medium", "low"}
DEFAULT_ACCEPTABLE_SCORE = 7.0
DEFAULT_MAX_REPORT_CHARS = 70_000
DEFAULT_MAX_CONTEXT_CHARS = 70_000
DEFAULT_PER_FILE_CHARS = 5_000
DEFAULT_MAX_CONTEXT_FILES = 60
DEFAULT_EVAL_AGENT_BUDGET_SECONDS = 600

JudgeFn = Callable[[str, str], str]


class EvalArea(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    weight: float
    requirements: list[str] = Field(default_factory=list)


class EvalContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    prompt: str
    areas: list[EvalArea]
    prompt_specific_requirements: list[str] = Field(default_factory=list)
    freshness_requirements: list[str] = Field(default_factory=list)
    critical_fail_rules: list[str] = Field(default_factory=list)
    generated_at: str = ""
    model: str = ""


class ReportVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    source_snapshot_path: str
    eval_snapshot_path: str
    sha256: str
    chars: int
    created_at: str
    trigger_tool: str = ""
    trigger_operation: str = ""


class EvidenceContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_root: str
    rendered: str
    included_files: list[str] = Field(default_factory=list)
    omitted_file_count: int = 0
    chars: int = 0


class AreaScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float
    rationale: str = ""
    critical_issues: list[str] = Field(default_factory=list)


class ClaimCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    category: str = ""
    status: Literal[
        "supported",
        "partially_supported",
        "unsupported",
        "contradicted",
        "stale",
        "not_checkable",
    ] = "not_checkable"
    importance: Literal["critical", "high", "medium", "low"] = "medium"
    evidence: str = ""
    notes: str = ""


class MajorConclusion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conclusion: str
    supporting_claims: list[str] = Field(default_factory=list)
    counterarguments_considered: list[str] = Field(default_factory=list)
    missing_links: list[str] = Field(default_factory=list)
    logic_score: float = 0.0


class VersionEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    eval_id: str
    version: int
    snapshot_path: str
    source_snapshot_path: str
    model: str
    raw_model_score: float
    weighted_score: float
    overall_score: float
    area_scores: dict[str, AreaScore]
    claim_checks: list[ClaimCheck] = Field(default_factory=list)
    major_conclusions: list[MajorConclusion] = Field(default_factory=list)
    critical_fail_flags: list[str] = Field(default_factory=list)
    decision_usefulness_notes: list[str] = Field(default_factory=list)
    summary: str = ""
    generated_at: str = Field(default_factory=lambda: _utc_now_iso())


class PairwiseRegression(BaseModel):
    model_config = ConfigDict(extra="forbid")

    area: str = ""
    severity: Literal["critical", "material", "minor"] = "material"
    reason: str


class PairwiseEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    eval_id: str
    older_version: int
    newer_version: int
    winner: Literal["older", "newer", "tie"]
    margin: float = 0.0
    score_delta: float = 0.0
    is_regression: bool = False
    regressions: list[PairwiseRegression] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    introduced_errors: list[str] = Field(default_factory=list)
    removed_evidence: list[str] = Field(default_factory=list)
    summary: str = ""
    generated_at: str = Field(default_factory=lambda: _utc_now_iso())
    model: str = ""


class TrajectorySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_id: str
    case_id: str
    run_root: str
    output_root: str
    prompt: str
    model: str
    versions_evaluated: int
    best_score_seen: float
    final_score: float
    best_version: int | None = None
    final_version: int | None = None
    time_to_first_acceptable_seconds: float | None = None
    time_to_best_seconds: float | None = None
    regressions: int = 0
    critical_errors_remaining: int = 0
    pairwise_win_rate_vs_previous: float | None = None
    version_scores: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    generated_at: str = Field(default_factory=lambda: _utc_now_iso())


class EvaluationRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_id: str
    case_id: str
    output_root: str
    contract: EvalContract
    versions: list[ReportVersion]
    version_evaluations: list[VersionEvaluation]
    pairwise_evaluations: list[PairwiseEvaluation]
    trajectory: TrajectorySummary


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _iso_to_epoch(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _clamp_score(value: Any, *, default: float = 0.0, scale: float = 1.0) -> float:
    try:
        score = float(value) * scale
    except (TypeError, ValueError):
        score = default
    return max(0.0, min(10.0, score))


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_text(path: str | Path) -> str:
    target = Path(path)
    if not target.exists() or not target.is_file():
        return ""
    return target.read_text(encoding="utf-8", errors="replace")


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[: max(0, limit - 80)].rstrip() + f"\n\n[truncated {omitted} chars]"


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = _strip_provider_thinking(text).strip()
    if not stripped:
        raise ValueError("judge returned empty content")
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("judge response did not contain a JSON object")
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("judge JSON payload must be an object")
    return payload


def _judge_json(judge: JudgeFn, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    first_response = judge(system_prompt, user_prompt)
    try:
        return _extract_json_object(first_response)
    except Exception as exc:
        repair_prompt = (
            f"{user_prompt}\n\n"
            "Your previous response was not valid JSON for the requested schema. "
            f"Parse error: {type(exc).__name__}: {exc}\n\n"
            f"Previous response:\n{_truncate_text(first_response, 4000)}\n\n"
            "Return only one valid JSON object. Do not include markdown fences or commentary."
        )
        return _extract_json_object(judge(system_prompt, repair_prompt))


def _strip_provider_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned


def _llm_response_text(response: Any) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def default_judge_fn(model_name: str | None = None) -> JudgeFn:
    resolved_model = model_name or get_model("harness", "agent")
    llm = get_llm(resolved_model)

    def _judge(system_prompt: str, user_prompt: str) -> str:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return _llm_response_text(response)

    return _judge


def default_eval_contract(case_id: str, prompt: str, *, model: str = "") -> EvalContract:
    requirements = _prompt_requirements(prompt)
    return EvalContract(
        case_id=case_id,
        prompt=prompt,
        areas=[
            EvalArea(name=name, weight=weight, requirements=AREA_REQUIREMENTS[name])
            for name, weight in AREA_WEIGHTS.items()
        ],
        prompt_specific_requirements=requirements,
        freshness_requirements=[
            "Market prices, multiples, commodity data, macro releases, and filings should have explicit as-of dates.",
            "Latest-period claims should identify the period or release date when available.",
        ],
        critical_fail_rules=[
            "A major contradicted number or date should cap the version below 6 even if prose quality is high.",
            "A stale time-sensitive claim presented as current should count as a critical failure when it affects the investment conclusion.",
            "A report that misses the user's central requested topic should not score above 7.",
        ],
        generated_at=_utc_now_iso(),
        model=model,
    )


def _prompt_requirements(prompt: str) -> list[str]:
    lower = prompt.lower()
    checks = [
        ("valuation", "Address valuation with visible market multiples, cash-flow framing, scenario math, or peer/history comparison."),
        ("balance", "Assess balance-sheet quality, leverage, liquidity, and shareholder returns when relevant."),
        ("shareholder", "Assess dividends, buybacks, and capital-allocation sustainability when requested."),
        ("crude", "Address crude-oil supply, demand, inventory, OPEC, futures-curve, or price-driver evidence when requested."),
        ("oil", "Address oil-market drivers when the prompt asks for energy exposure."),
        ("macro", "Connect macro indicators to the investment thesis when requested."),
        ("bull", "Provide a bull case with catalysts or upside drivers."),
        ("bear", "Provide a bear case with downside risks and what would make it wrong."),
        ("risk/reward", "State a clear 12-month risk/reward or decision frame when requested."),
    ]
    requirements = [text for needle, text in checks if needle in lower]
    return list(dict.fromkeys(requirements))


def generate_eval_contract(
    *,
    case_id: str,
    prompt: str,
    judge_fn: JudgeFn | None = None,
    model_name: str | None = None,
) -> EvalContract:
    model = model_name or get_model("harness", "agent")
    fallback = default_eval_contract(case_id, prompt, model=model)
    judge = judge_fn or default_judge_fn(model)
    system_prompt = (
        "You create compact evaluation contracts for AlphaSeeker investment research reports. "
        "Return JSON only. Do not include prose outside the JSON object."
    )
    user_prompt = (
        "Generate an evaluation contract for this prompt. Use the fixed areas and weights unless the "
        "prompt clearly requires an added prompt-specific requirement. Do not add new areas.\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Fixed areas and weights:\n{json.dumps(AREA_WEIGHTS, indent=2, ensure_ascii=True)}\n\n"
        "Return this JSON shape:\n"
        "{\n"
        '  "prompt_specific_requirements": ["..."],\n'
        '  "freshness_requirements": ["..."],\n'
        '  "critical_fail_rules": ["..."]\n'
        "}"
    )
    try:
        payload = _judge_json(judge, system_prompt, user_prompt)
    except Exception:
        return fallback

    return fallback.model_copy(
        update={
            "prompt_specific_requirements": _string_list(
                payload.get("prompt_specific_requirements"),
                fallback.prompt_specific_requirements,
            ),
            "freshness_requirements": _string_list(
                payload.get("freshness_requirements"),
                fallback.freshness_requirements,
            ),
            "critical_fail_rules": _string_list(
                payload.get("critical_fail_rules"),
                fallback.critical_fail_rules,
            ),
            "generated_at": _utc_now_iso(),
            "model": model,
        }
    )


def _string_list(value: Any, fallback: list[str] | None = None) -> list[str]:
    if isinstance(value, list):
        result = [str(item).strip() for item in value if str(item).strip()]
        if result:
            return result
    return list(fallback or [])


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return label.strip("_") or "eval"


def default_eval_id(case_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{_safe_label(case_id)}-{timestamp}"


def _default_output_root(eval_id: str, case_id: str) -> Path:
    return Path.cwd() / "data" / "eval_runs" / _safe_label(eval_id) / _safe_label(case_id)


def collect_report_versions(run_root: str | Path, output_root: str | Path) -> list[ReportVersion]:
    root = Path(run_root)
    versions_root = Path(output_root) / "versions"
    versions_root.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in read_jsonl(registry_paths(root)["final_report_versions"])
        if str(row.get("agent_id") or "") == "agent_root"
    ]

    versions: list[ReportVersion] = []
    seen_hashes: set[str] = set()
    for row in rows:
        version = _int_or_default(row.get("version"), len(versions) + 1)
        source_path = Path(str(row.get("snapshot_path") or row.get("source_path") or ""))
        if not source_path.exists():
            continue
        text = _read_text(source_path)
        digest = str(row.get("sha256") or _sha256(text))
        seen_hashes.add(digest)
        eval_path = versions_root / f"v{version:04d}.md"
        write_text_atomic(eval_path, text)
        versions.append(
            ReportVersion(
                version=version,
                source_snapshot_path=str(source_path),
                eval_snapshot_path=str(eval_path),
                sha256=digest,
                chars=len(text),
                created_at=str(row.get("created_at") or _utc_now_iso()),
                trigger_tool=str(row.get("trigger_tool") or ""),
                trigger_operation=str(row.get("trigger_operation") or ""),
            )
        )

    final_path = agent_workspace_paths(root, "agent_root")["publish_final"]
    final_text = _read_text(final_path)
    if final_text:
        digest = _sha256(final_text)
        if digest not in seen_hashes:
            version = (max((item.version for item in versions), default=0) + 1)
            eval_path = versions_root / f"v{version:04d}.md"
            write_text_atomic(eval_path, final_text)
            versions.append(
                ReportVersion(
                    version=version,
                    source_snapshot_path=str(final_path),
                    eval_snapshot_path=str(eval_path),
                    sha256=digest,
                    chars=len(final_text),
                    created_at=_mtime_iso(final_path),
                    trigger_tool="current_final",
                    trigger_operation="fallback_snapshot",
                )
            )

    if not versions:
        raise ValueError(f"No root publish/final.md versions found for run_root={root}")
    return sorted(versions, key=lambda item: item.version)


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mtime_iso(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except OSError:
        return _utc_now_iso()


def collect_evidence_context(
    run_root: str | Path,
    *,
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    per_file_chars: int = DEFAULT_PER_FILE_CHARS,
    max_files: int = DEFAULT_MAX_CONTEXT_FILES,
) -> EvidenceContext:
    root = Path(run_root)
    records = latest_agent_records(root)
    ordered_records = sorted(
        records.values(),
        key=lambda record: (0 if not record.parent_id else 1, record.agent_id),
    )
    sections: list[str] = []
    included_files: list[str] = []
    omitted = 0

    def include(path: Path, label: str, limit: int = per_file_chars) -> None:
        nonlocal omitted
        if len(included_files) >= max_files:
            omitted += 1
            return
        if not path.exists() or not path.is_file():
            return
        text = _compact_file_for_context(path, limit=limit)
        if not text.strip():
            return
        candidate = f"## {label}\nPath: {path}\n\n{text.strip()}\n"
        current = sum(len(item) for item in sections)
        if current + len(candidate) > max_chars:
            remaining = max_chars - current
            if remaining <= 200:
                omitted += 1
                return
            candidate = _truncate_text(candidate, remaining)
        sections.append(candidate)
        included_files.append(str(path))

    for record in ordered_records:
        paths = agent_workspace_paths(root, record.agent_id)
        prefix = f"{record.agent_id} ({record.preset})"
        include(paths["publish_summary"], f"{prefix} publish/summary.md")
        include(paths["publish_index"], f"{prefix} publish/artifact_index.md")
        if record.parent_id:
            include(paths["publish_final"], f"{prefix} publish/final.md")

        scratch_root = paths["scratch_root"]
        if scratch_root.exists():
            for path in sorted(scratch_root.rglob("*")):
                if path.is_file() and path.suffix.lower() in {".md", ".txt", ".json"}:
                    include(path, f"{prefix} scratch/{path.relative_to(scratch_root)}")

        artifacts_root = paths["artifacts_root"]
        if artifacts_root.exists():
            candidates = [
                path
                for path in sorted(artifacts_root.rglob("*"))
                if path.is_file()
                and path.suffix.lower() in {".md", ".txt", ".json", ".jsonl"}
                and "_harness" not in path.parts
            ]
            priority = sorted(candidates, key=_artifact_priority)
            for path in priority:
                include(path, f"{prefix} artifacts/{path.relative_to(artifacts_root)}")

    rendered = "\n".join(sections).strip()
    if omitted:
        rendered += f"\n\n[omitted {omitted} additional evidence file(s) due to context budget]"
    return EvidenceContext(
        run_root=str(root),
        rendered=rendered,
        included_files=included_files,
        omitted_file_count=omitted,
        chars=len(rendered),
    )


def _artifact_priority(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    if name in {"summary.md", "output.md", "evidence.json", "details.json"}:
        return (0, str(path))
    if "search" in path.parts:
        return (2, str(path))
    return (3, str(path))


def _compact_file_for_context(path: Path, *, limit: int) -> str:
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(_read_text(path))
        except json.JSONDecodeError:
            return _truncate_text(_read_text(path), limit)
        compact = _compact_json_payload(payload)
        return _truncate_text(compact, limit)
    if path.suffix.lower() == ".jsonl":
        lines = [line for line in _read_text(path).splitlines() if line.strip()]
        return _truncate_text("\n".join(lines[:20]), limit)
    return _truncate_text(_read_text(path), limit)


def _compact_json_payload(payload: Any) -> str:
    if isinstance(payload, list):
        if payload and all(isinstance(item, dict) for item in payload):
            rows = []
            for item in payload[:8]:
                assert isinstance(item, dict)
                title = str(item.get("title") or item.get("summary") or item.get("name") or "")[:180]
                href = str(item.get("href") or item.get("url") or item.get("source") or "")[:240]
                body = str(item.get("body") or item.get("snippet") or item.get("content") or "")[:500]
                rows.append(json.dumps({"title": title, "url": href, "body": body}, ensure_ascii=True))
            return "\n".join(rows)
        return json.dumps(payload[:20], indent=2, ensure_ascii=True)
    if isinstance(payload, dict):
        compact = dict(payload)
        for key in list(compact):
            value = compact[key]
            if isinstance(value, str) and len(value) > 1200:
                compact[key] = _truncate_text(value, 1200)
            elif isinstance(value, list) and len(value) > 12:
                compact[key] = value[:12]
        return json.dumps(compact, indent=2, ensure_ascii=True)
    return json.dumps(payload, indent=2, ensure_ascii=True)


def _markdown_section_index(path: str | Path) -> list[dict[str, Any]]:
    text = _read_text(path)
    lines = text.splitlines()
    headings: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            continue
        headings.append(
            {
                "level": len(match.group(1)),
                "title": match.group(2).strip(),
                "start_line": index,
                "end_line": len(lines),
            }
        )
    if not headings:
        return [
            {
                "level": 1,
                "title": "Document",
                "start_line": 1,
                "end_line": len(lines),
            }
        ]
    for pos, heading in enumerate(headings):
        current_level = int(heading["level"])
        end_line = len(lines)
        for later in headings[pos + 1 :]:
            if int(later["level"]) <= current_level:
                end_line = int(later["start_line"]) - 1
                break
        heading["end_line"] = end_line
    return headings


def _write_markdown_section_index(markdown_path: str | Path) -> Path:
    path = Path(markdown_path)
    section_path = path.with_suffix(path.suffix + ".sections.json")
    write_json_atomic(section_path, {"file": str(path), "sections": _markdown_section_index(path)})
    return section_path


def _format_section_listing(section_path: str | Path) -> str:
    try:
        payload = json.loads(_read_text(section_path))
    except json.JSONDecodeError:
        return ""
    sections = payload.get("sections")
    if not isinstance(sections, list):
        return ""
    lines: list[str] = []
    for section in sections[:80]:
        if not isinstance(section, dict):
            continue
        title = str(section.get("title") or "").strip()
        if not title:
            continue
        level = int(section.get("level") or 1)
        start = int(section.get("start_line") or 1)
        end = int(section.get("end_line") or start)
        lines.append(f"- {'#' * max(1, min(level, 6))} {title} (lines {start}-{end})")
    return "\n".join(lines)


def _write_pairwise_diff(older_path: str | Path, newer_path: str | Path, diff_path: str | Path) -> Path:
    older = _read_text(older_path).splitlines(keepends=True)
    newer = _read_text(newer_path).splitlines(keepends=True)
    diff = difflib.unified_diff(
        older,
        newer,
        fromfile=Path(older_path).name,
        tofile=Path(newer_path).name,
    )
    target = Path(diff_path)
    write_text_atomic(target, "".join(diff))
    return target


@contextlib.contextmanager
def _temporary_harness_agent_model(model_name: str | None):
    env_key = "ALPHASEEKER_MODEL_HARNESS_AGENT"
    if not model_name:
        yield
        return
    previous = os.environ.get(env_key)
    os.environ[env_key] = model_name
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous


def _unique_eval_agent_run_root(output_root: Path, label: str) -> Path:
    base = output_root / "_agent_runs" / _safe_label(label)
    candidate = base
    counter = 1
    while candidate.exists():
        counter += 1
        candidate = base.with_name(f"{base.name}_{counter}")
    return candidate


def _source_skill_packs(source_run_root: str | Path) -> list[str]:
    try:
        source_request = load_request(source_run_root)
    except Exception:
        return ["core", "equity", "macro", "commodity"]
    return source_request.available_skill_packs or ["core", "equity", "macro", "commodity"]


def _infer_source_run_root(source_path: str | Path) -> Path | None:
    path = Path(source_path).resolve(strict=False)
    for parent in [path, *path.parents]:
        if parent.name == "harness_runs":
            return None
        if parent.parent.name == "harness_runs":
            return parent
    return None


def _run_tool_driven_evaluator_agent(
    *,
    source_run_root: str | Path,
    output_root: str | Path,
    eval_id: str,
    case_id: str,
    label: str,
    task_prompt: str,
    context_files: list[str | Path],
    evaluator_model: str | None,
) -> dict[str, Any]:
    from src.harness import run_harness

    resolved_output = Path(output_root)
    eval_run_root = _unique_eval_agent_run_root(resolved_output, label)
    registry = build_skill_registry()
    request = HarnessRequest(
        user_prompt=task_prompt,
        run_id=f"{_safe_label(eval_id)}-{_safe_label(case_id)}-{_safe_label(label)}",
        root_preset="evaluator",
        wall_clock_budget_seconds=DEFAULT_EVAL_AGENT_BUDGET_SECONDS,
        root_wall_clock_seconds=DEFAULT_EVAL_AGENT_BUDGET_SECONDS,
        per_agent_wall_clock_seconds=DEFAULT_EVAL_AGENT_BUDGET_SECONDS,
        max_agents_per_run=8,
        max_live_agents=4,
        max_live_children_per_parent=2,
        available_skill_packs=_source_skill_packs(source_run_root),
        commenter_interval_seconds=0.0,
        resume_from_run_root=str(eval_run_root),
    )
    run_root, root_agent_id = initialize_run_root(request)
    available_skills = get_skills_for_packs(registry, request.available_skill_packs or ["core"])
    visible_skills = visible_skills_for_preset(preset="evaluator", available_skills=available_skills)
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="evaluator",
        task_name=f"Evaluate {case_id} {label}",
        description=task_prompt[:160],
        task_markdown=task_prompt,
        tools_markdown=render_tools_markdown(
            preset="evaluator",
            available_tools=default_tool_allowlist("evaluator"),
            available_skills=visible_skills,
        ),
        context_files=[str(Path(path)) for path in context_files],
    )
    with _temporary_harness_agent_model(evaluator_model):
        response = run_harness(HarnessRequest(user_prompt=task_prompt, resume_from_run_root=str(eval_run_root)))
    final_path = Path(response.final_report_path or agent_workspace_paths(run_root, root_agent_id)["publish_final"])
    payload = _extract_json_object(_read_text(final_path))
    payload["_tool_driven_eval_run_root"] = str(run_root)
    payload["_tool_driven_eval_status"] = response.status
    if response.error:
        payload["_tool_driven_eval_error"] = response.error
    return payload


def _version_evaluation_from_payload(
    *,
    payload: dict[str, Any],
    case_id: str,
    eval_id: str,
    contract: EvalContract,
    version: ReportVersion,
    model: str,
) -> VersionEvaluation:
    score_scale = _score_scale_from_payload(payload)
    area_scores = _normalize_area_scores(
        payload.get("area_scores"),
        contract,
        payload.get("overall_score"),
        score_scale=score_scale,
    )
    raw_score = _clamp_score(
        payload.get("overall_score"),
        default=_weighted_score(area_scores, contract),
        scale=score_scale,
    )
    weighted = _weighted_score(area_scores, contract)
    claim_checks = _normalize_claim_checks(payload.get("claim_checks"))
    critical_flags = _string_list(payload.get("critical_fail_flags"))
    capped = _apply_critical_caps(
        min(raw_score, weighted),
        critical_fail_flags=critical_flags,
        claim_checks=claim_checks,
    )
    notes = _string_list(payload.get("decision_usefulness_notes"))
    eval_run_root = str(payload.get("_tool_driven_eval_run_root") or "")
    if eval_run_root:
        notes.append(f"tool_driven_eval_run_root={eval_run_root}")
    return VersionEvaluation(
        case_id=case_id,
        eval_id=eval_id,
        version=version.version,
        snapshot_path=version.eval_snapshot_path,
        source_snapshot_path=version.source_snapshot_path,
        model=model,
        raw_model_score=raw_score,
        weighted_score=weighted,
        overall_score=capped,
        area_scores=area_scores,
        claim_checks=claim_checks,
        major_conclusions=_normalize_major_conclusions(payload.get("major_conclusions"), score_scale=score_scale),
        critical_fail_flags=critical_flags,
        decision_usefulness_notes=notes,
        summary=str(payload.get("summary") or ""),
    )


def _score_report_version_with_agent(
    *,
    case_id: str,
    eval_id: str,
    prompt: str,
    contract: EvalContract,
    version: ReportVersion,
    evidence_context: EvidenceContext,
    model_name: str,
) -> VersionEvaluation:
    output_root = Path(version.eval_snapshot_path).parent.parent
    evidence_path = output_root / "evidence_context.md"
    contract_path = output_root / "eval_contract.json"
    section_path = _write_markdown_section_index(version.eval_snapshot_path)
    section_listing = _format_section_listing(section_path)
    task_prompt = (
        "# Task Assignment\n\n"
        "Evaluate one AlphaSeeker report version. You are a full evaluator agent with the same tool style "
        "as harness agents; use tools to inspect files and retrieve/check evidence as needed. Do not assume "
        "the report body from this prompt. Read the relevant file sections yourself.\n\n"
        f"Case id: {case_id}\n"
        f"Eval id: {eval_id}\n"
        f"Report file: context/{Path(version.eval_snapshot_path).name}\n"
        f"Section index file: context/{section_path.name}\n"
        f"Evidence context file: context/{evidence_path.name}\n"
        f"Eval contract file: context/{contract_path.name}\n\n"
        f"Original user prompt:\n{prompt}\n\n"
        "Available report sections:\n"
        f"{section_listing or '[no section headings found]'}\n\n"
        "Use `read` and `grep` first. You may use the same research/data skills available to harness agents "
        "when freshness or factuality needs checking. Score on a 0 to 10 scale.\n\n"
        "Write exactly one JSON object to `publish/final.md`, then write a short `publish/summary.md`, "
        "`publish/artifact_index.md`, and call status(\"done\"). The JSON object must have this shape:\n"
        "{\n"
        '  "overall_score": 0.0,\n'
        '  "area_scores": {\n'
        '    "factual_correctness": {"score": 0.0, "rationale": "...", "critical_issues": []},\n'
        '    "freshness": {"score": 0.0, "rationale": "...", "critical_issues": []},\n'
        '    "evidence_grounding": {"score": 0.0, "rationale": "...", "critical_issues": []},\n'
        '    "logical_soundness": {"score": 0.0, "rationale": "...", "critical_issues": []},\n'
        '    "completeness": {"score": 0.0, "rationale": "...", "critical_issues": []},\n'
        '    "numerical_discipline": {"score": 0.0, "rationale": "...", "critical_issues": []},\n'
        '    "decision_usefulness": {"score": 0.0, "rationale": "...", "critical_issues": []}\n'
        "  },\n"
        '  "claim_checks": [{"claim": "...", "category": "market_data", "status": "supported", "importance": "high", "evidence": "...", "notes": "..."}],\n'
        '  "major_conclusions": [{"conclusion": "...", "supporting_claims": [], "counterarguments_considered": [], "missing_links": [], "logic_score": 0.0}],\n'
        '  "critical_fail_flags": [],\n'
        '  "decision_usefulness_notes": [],\n'
        '  "summary": "..."\n'
        "}"
    )
    payload = _run_tool_driven_evaluator_agent(
        source_run_root=evidence_context.run_root,
        output_root=output_root,
        eval_id=eval_id,
        case_id=case_id,
        label=f"score_v{version.version:04d}",
        task_prompt=task_prompt,
        context_files=[version.eval_snapshot_path, evidence_path, contract_path, section_path],
        evaluator_model=model_name,
    )
    return _version_evaluation_from_payload(
        payload=payload,
        case_id=case_id,
        eval_id=eval_id,
        contract=contract,
        version=version,
        model=model_name,
    )


def score_report_version(
    *,
    case_id: str,
    eval_id: str,
    prompt: str,
    contract: EvalContract,
    version: ReportVersion,
    evidence_context: EvidenceContext,
    judge_fn: JudgeFn | None = None,
    model_name: str | None = None,
    max_report_chars: int = DEFAULT_MAX_REPORT_CHARS,
) -> VersionEvaluation:
    model = model_name or get_model("harness", "agent")
    if judge_fn is None:
        return _score_report_version_with_agent(
            case_id=case_id,
            eval_id=eval_id,
            prompt=prompt,
            contract=contract,
            version=version,
            evidence_context=evidence_context,
            model_name=model,
        )

    judge = judge_fn
    report_text = _truncate_text(_read_text(version.eval_snapshot_path), max_report_chars)
    system_prompt = (
        "You are a strict evaluator of AlphaSeeker investment research reports. "
        "Use only the report, eval contract, and run evidence context supplied by the user. "
        "If a material claim is not supported by that evidence, mark it unsupported or not_checkable. "
        "Return JSON only."
    )
    user_prompt = (
        f"Evaluate report version {version.version} for case {case_id}.\n\n"
        f"Original user prompt:\n{prompt}\n\n"
        f"Evaluation contract JSON:\n{contract.model_dump_json(indent=2)}\n\n"
        f"Run evidence context:\n{evidence_context.rendered or '[no evidence context available]'}\n\n"
        f"Report version {version.version}:\n{report_text}\n\n"
        "All scores must be on a 0 to 10 scale, where 10 is excellent and 0 is unusable.\n"
        "Return this JSON shape exactly:\n"
        "{\n"
        '  "overall_score": 0.0,\n'
        '  "area_scores": {\n'
        '    "factual_correctness": {"score": 0.0, "rationale": "...", "critical_issues": []},\n'
        '    "freshness": {"score": 0.0, "rationale": "...", "critical_issues": []},\n'
        '    "evidence_grounding": {"score": 0.0, "rationale": "...", "critical_issues": []},\n'
        '    "logical_soundness": {"score": 0.0, "rationale": "...", "critical_issues": []},\n'
        '    "completeness": {"score": 0.0, "rationale": "...", "critical_issues": []},\n'
        '    "numerical_discipline": {"score": 0.0, "rationale": "...", "critical_issues": []},\n'
        '    "decision_usefulness": {"score": 0.0, "rationale": "...", "critical_issues": []}\n'
        "  },\n"
        '  "claim_checks": [{"claim": "...", "category": "market_data", "status": "supported", "importance": "high", "evidence": "...", "notes": "..."}],\n'
        '  "major_conclusions": [{"conclusion": "...", "supporting_claims": [], "counterarguments_considered": [], "missing_links": [], "logic_score": 0.0}],\n'
        '  "critical_fail_flags": [],\n'
        '  "decision_usefulness_notes": [],\n'
        '  "summary": "..."\n'
        "}"
    )
    payload = _judge_json(judge, system_prompt, user_prompt)
    return _version_evaluation_from_payload(
        payload=payload,
        case_id=case_id,
        eval_id=eval_id,
        contract=contract,
        version=version,
        model=model,
    )


def _score_scale_from_payload(payload: dict[str, Any]) -> float:
    values: list[float] = []
    raw_overall = payload.get("overall_score")
    with_context = [raw_overall]
    area_scores = payload.get("area_scores")
    if isinstance(area_scores, dict):
        for raw in area_scores.values():
            if isinstance(raw, dict):
                with_context.append(raw.get("score"))
    for value in with_context:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if values and max(values) <= 1.0 and any(value > 0 for value in values):
        return 10.0
    return 1.0


def _normalize_area_scores(
    value: Any,
    contract: EvalContract,
    raw_overall: Any,
    *,
    score_scale: float = 1.0,
) -> dict[str, AreaScore]:
    default_score = _clamp_score(raw_overall, default=0.0, scale=score_scale)
    payload = value if isinstance(value, dict) else {}
    scores: dict[str, AreaScore] = {}
    for area in contract.areas:
        raw = payload.get(area.name)
        if isinstance(raw, dict):
            scores[area.name] = AreaScore(
                score=_clamp_score(raw.get("score"), default=default_score, scale=score_scale),
                rationale=str(raw.get("rationale") or ""),
                critical_issues=_string_list(raw.get("critical_issues")),
            )
        else:
            scores[area.name] = AreaScore(score=default_score)
    return scores


def _weighted_score(area_scores: dict[str, AreaScore], contract: EvalContract) -> float:
    numerator = 0.0
    denominator = 0.0
    for area in contract.areas:
        score = area_scores.get(area.name)
        if score is None:
            continue
        weight = max(0.0, float(area.weight))
        numerator += score.score * weight
        denominator += weight
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 3)


def _normalize_claim_checks(value: Any) -> list[ClaimCheck]:
    if not isinstance(value, list):
        return []
    checks: list[ClaimCheck] = []
    for item in value[:40]:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "not_checkable")
        importance = str(item.get("importance") or "medium")
        checks.append(
            ClaimCheck(
                claim=str(item.get("claim") or ""),
                category=str(item.get("category") or ""),
                status=status if status in CLAIM_STATUSES else "not_checkable",  # type: ignore[arg-type]
                importance=importance if importance in IMPORTANCE_LEVELS else "medium",  # type: ignore[arg-type]
                evidence=str(item.get("evidence") or ""),
                notes=str(item.get("notes") or ""),
            )
        )
    return [check for check in checks if check.claim.strip()]


def _normalize_major_conclusions(value: Any, *, score_scale: float = 1.0) -> list[MajorConclusion]:
    if not isinstance(value, list):
        return []
    conclusions: list[MajorConclusion] = []
    for item in value[:12]:
        if not isinstance(item, dict):
            continue
        conclusions.append(
            MajorConclusion(
                conclusion=str(item.get("conclusion") or ""),
                supporting_claims=_string_list(item.get("supporting_claims")),
                counterarguments_considered=_string_list(item.get("counterarguments_considered")),
                missing_links=_string_list(item.get("missing_links")),
                logic_score=_clamp_score(item.get("logic_score"), default=0.0, scale=score_scale),
            )
        )
    return [item for item in conclusions if item.conclusion.strip()]


def _apply_critical_caps(
    score: float,
    *,
    critical_fail_flags: list[str],
    claim_checks: list[ClaimCheck],
) -> float:
    capped = score
    if critical_fail_flags:
        capped = min(capped, 6.0)
    for check in claim_checks:
        if check.importance not in {"critical", "high"}:
            continue
        if check.status == "contradicted":
            capped = min(capped, 5.5)
        elif check.status == "stale":
            capped = min(capped, 6.0)
        elif check.status == "unsupported" and check.importance == "critical":
            capped = min(capped, 6.5)
    return round(capped, 3)


def _pairwise_evaluation_from_payload(
    *,
    payload: dict[str, Any],
    case_id: str,
    eval_id: str,
    older_version: ReportVersion,
    newer_version: ReportVersion,
    older_eval: VersionEvaluation,
    newer_eval: VersionEvaluation,
    model: str,
) -> PairwiseEvaluation:
    winner = str(payload.get("winner") or "tie")
    if winner not in {"older", "newer", "tie"}:
        winner = "tie"
    regressions = _normalize_pairwise_regressions(payload.get("regressions"))
    score_delta = round(newer_eval.overall_score - older_eval.overall_score, 3)
    deterministic_regressions = _deterministic_regressions(older_eval, newer_eval, score_delta)
    all_regressions = [*regressions, *deterministic_regressions]
    material_regression = any(item.severity in {"critical", "material"} for item in all_regressions)
    is_regression = (
        material_regression
        or score_delta < -0.5
        or (winner == "older" and _clamp_score(payload.get("margin"), default=0.0) >= 0.5)
    )
    summary = str(payload.get("summary") or "")
    eval_run_root = str(payload.get("_tool_driven_eval_run_root") or "")
    if eval_run_root:
        summary = (summary + f"\n\ntool_driven_eval_run_root={eval_run_root}").strip()
    return PairwiseEvaluation(
        case_id=case_id,
        eval_id=eval_id,
        older_version=older_version.version,
        newer_version=newer_version.version,
        winner=winner,  # type: ignore[arg-type]
        margin=_clamp_score(payload.get("margin"), default=0.0),
        score_delta=score_delta,
        is_regression=is_regression,
        regressions=all_regressions,
        improvements=_string_list(payload.get("improvements")),
        introduced_errors=_string_list(payload.get("introduced_errors")),
        removed_evidence=_string_list(payload.get("removed_evidence")),
        summary=summary,
        model=model,
    )


def _compare_report_versions_with_agent(
    *,
    case_id: str,
    eval_id: str,
    prompt: str,
    older_version: ReportVersion,
    newer_version: ReportVersion,
    older_eval: VersionEvaluation,
    newer_eval: VersionEvaluation,
    model_name: str,
) -> PairwiseEvaluation:
    output_root = Path(newer_version.eval_snapshot_path).parent.parent
    older_sections = _write_markdown_section_index(older_version.eval_snapshot_path)
    newer_sections = _write_markdown_section_index(newer_version.eval_snapshot_path)
    diff_path = _write_pairwise_diff(
        older_version.eval_snapshot_path,
        newer_version.eval_snapshot_path,
        output_root / f"pairwise_v{older_version.version:04d}_v{newer_version.version:04d}.diff",
    )
    older_eval_path = output_root / "versions" / f"v{older_version.version:04d}.eval.json"
    newer_eval_path = output_root / "versions" / f"v{newer_version.version:04d}.eval.json"
    task_prompt = (
        "# Task Assignment\n\n"
        "Compare two adjacent AlphaSeeker report versions for material quality regressions. You are a full "
        "evaluator agent with the same tool style as harness agents. Use tools to inspect the diff, section "
        "indexes, evaluation JSON, and any report sections needed. Do not assume the full report body from "
        "this prompt.\n\n"
        f"Case id: {case_id}\n"
        f"Eval id: {eval_id}\n"
        f"Older report file: context/{Path(older_version.eval_snapshot_path).name}\n"
        f"Newer report file: context/{Path(newer_version.eval_snapshot_path).name}\n"
        f"Diff file: context/{diff_path.name}\n"
        f"Older evaluation file: context/{older_eval_path.name}\n"
        f"Newer evaluation file: context/{newer_eval_path.name}\n\n"
        f"Original user prompt:\n{prompt}\n\n"
        "Older report sections:\n"
        f"{_format_section_listing(older_sections) or '[no section headings found]'}\n\n"
        "Newer report sections:\n"
        f"{_format_section_listing(newer_sections) or '[no section headings found]'}\n\n"
        "A newer version is a regression if it introduces a new critical factual error, removes important "
        "correct evidence, becomes less complete against the prompt, the older version is preferred by a "
        "meaningful margin, or an area score drops more than 0.5 on a 10-point scale.\n\n"
        "Write exactly one JSON object to `publish/final.md`, then write a short `publish/summary.md`, "
        "`publish/artifact_index.md`, and call status(\"done\"). JSON shape:\n"
        "{\n"
        '  "winner": "older|newer|tie",\n'
        '  "margin": 0.0,\n'
        '  "regressions": [{"area": "...", "severity": "critical|material|minor", "reason": "..."}],\n'
        '  "improvements": ["..."],\n'
        '  "introduced_errors": ["..."],\n'
        '  "removed_evidence": ["..."],\n'
        '  "summary": "..."\n'
        "}"
    )
    payload = _run_tool_driven_evaluator_agent(
        source_run_root=_infer_source_run_root(older_version.source_snapshot_path) or output_root,
        output_root=output_root,
        eval_id=eval_id,
        case_id=case_id,
        label=f"pairwise_v{older_version.version:04d}_v{newer_version.version:04d}",
        task_prompt=task_prompt,
        context_files=[
            older_version.eval_snapshot_path,
            newer_version.eval_snapshot_path,
            older_sections,
            newer_sections,
            diff_path,
            older_eval_path,
            newer_eval_path,
        ],
        evaluator_model=model_name,
    )
    return _pairwise_evaluation_from_payload(
        payload=payload,
        case_id=case_id,
        eval_id=eval_id,
        older_version=older_version,
        newer_version=newer_version,
        older_eval=older_eval,
        newer_eval=newer_eval,
        model=model_name,
    )


def compare_report_versions(
    *,
    case_id: str,
    eval_id: str,
    prompt: str,
    older_version: ReportVersion,
    newer_version: ReportVersion,
    older_eval: VersionEvaluation,
    newer_eval: VersionEvaluation,
    judge_fn: JudgeFn | None = None,
    model_name: str | None = None,
    max_report_chars: int = DEFAULT_MAX_REPORT_CHARS // 2,
) -> PairwiseEvaluation:
    model = model_name or get_model("harness", "agent")
    if judge_fn is None:
        return _compare_report_versions_with_agent(
            case_id=case_id,
            eval_id=eval_id,
            prompt=prompt,
            older_version=older_version,
            newer_version=newer_version,
            older_eval=older_eval,
            newer_eval=newer_eval,
            model_name=model,
        )

    judge = judge_fn
    older_text = _truncate_text(_read_text(older_version.eval_snapshot_path), max_report_chars)
    newer_text = _truncate_text(_read_text(newer_version.eval_snapshot_path), max_report_chars)
    system_prompt = (
        "You compare adjacent AlphaSeeker report versions for quality regressions. "
        "A regression is material: new critical error, removed important correct evidence, worse completeness, "
        "older version clearly preferred, or area-score drop above threshold. Return JSON only."
    )
    user_prompt = (
        f"Compare adjacent report versions {older_version.version} and {newer_version.version} for case {case_id}.\n\n"
        f"Original user prompt:\n{prompt}\n\n"
        f"Older evaluation summary:\n{older_eval.model_dump_json(indent=2)}\n\n"
        f"Newer evaluation summary:\n{newer_eval.model_dump_json(indent=2)}\n\n"
        f"Older report:\n{older_text}\n\n"
        f"Newer report:\n{newer_text}\n\n"
        "Return this JSON shape:\n"
        "{\n"
        '  "winner": "older|newer|tie",\n'
        '  "margin": 0.0,\n'
        '  "regressions": [{"area": "...", "severity": "critical|material|minor", "reason": "..."}],\n'
        '  "improvements": ["..."],\n'
        '  "introduced_errors": ["..."],\n'
        '  "removed_evidence": ["..."],\n'
        '  "summary": "..."\n'
        "}"
    )
    payload = _judge_json(judge, system_prompt, user_prompt)
    return _pairwise_evaluation_from_payload(
        payload=payload,
        case_id=case_id,
        eval_id=eval_id,
        older_version=older_version,
        newer_version=newer_version,
        older_eval=older_eval,
        newer_eval=newer_eval,
        model=model,
    )


def _normalize_pairwise_regressions(value: Any) -> list[PairwiseRegression]:
    if not isinstance(value, list):
        return []
    regressions: list[PairwiseRegression] = []
    for item in value[:20]:
        if isinstance(item, str):
            regressions.append(PairwiseRegression(reason=item))
            continue
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity") or "material")
        regressions.append(
            PairwiseRegression(
                area=str(item.get("area") or ""),
                severity=severity if severity in {"critical", "material", "minor"} else "material",  # type: ignore[arg-type]
                reason=str(item.get("reason") or ""),
            )
        )
    return [item for item in regressions if item.reason.strip()]


def _deterministic_regressions(
    older_eval: VersionEvaluation,
    newer_eval: VersionEvaluation,
    score_delta: float,
) -> list[PairwiseRegression]:
    regressions: list[PairwiseRegression] = []
    if score_delta < -0.5:
        regressions.append(
            PairwiseRegression(
                area="overall",
                severity="material",
                reason=f"Overall score dropped by {abs(score_delta):.2f} points.",
            )
        )
    old_flags = set(older_eval.critical_fail_flags)
    for flag in newer_eval.critical_fail_flags:
        if flag not in old_flags:
            regressions.append(
                PairwiseRegression(
                    area="critical_fail_flags",
                    severity="critical",
                    reason=f"New critical fail flag: {flag}",
                )
            )
    for area, new_score in newer_eval.area_scores.items():
        old_score = older_eval.area_scores.get(area)
        if old_score is None:
            continue
        drop = old_score.score - new_score.score
        if drop > 0.5:
            severity = "material" if drop < 1.5 else "critical"
            regressions.append(
                PairwiseRegression(
                    area=area,
                    severity=severity,  # type: ignore[arg-type]
                    reason=f"{area} score dropped by {drop:.2f} points.",
                )
            )
    return regressions


def build_trajectory_summary(
    *,
    eval_id: str,
    case_id: str,
    run_root: str | Path,
    output_root: str | Path,
    prompt: str,
    model: str,
    versions: list[ReportVersion],
    evaluations: list[VersionEvaluation],
    pairwise: list[PairwiseEvaluation],
    acceptable_threshold: float = DEFAULT_ACCEPTABLE_SCORE,
    artifacts: dict[str, str] | None = None,
) -> TrajectorySummary:
    if not evaluations:
        raise ValueError("Cannot build trajectory without evaluations.")
    best_eval = max(evaluations, key=lambda item: item.overall_score)
    final_eval = evaluations[-1]
    start_epoch = _run_start_epoch(run_root, versions)
    first_acceptable = next(
        (item for item in evaluations if item.overall_score >= acceptable_threshold),
        None,
    )
    newer_wins = sum(1 for item in pairwise if item.winner == "newer")
    pairwise_rate = (newer_wins / len(pairwise)) if pairwise else None
    version_by_number = {item.version: item for item in versions}
    return TrajectorySummary(
        eval_id=eval_id,
        case_id=case_id,
        run_root=str(run_root),
        output_root=str(output_root),
        prompt=prompt,
        model=model,
        versions_evaluated=len(evaluations),
        best_score_seen=best_eval.overall_score,
        final_score=final_eval.overall_score,
        best_version=best_eval.version,
        final_version=final_eval.version,
        time_to_first_acceptable_seconds=_elapsed_from_start(
            start_epoch,
            version_by_number.get(first_acceptable.version).created_at if first_acceptable and version_by_number.get(first_acceptable.version) else None,
        ),
        time_to_best_seconds=_elapsed_from_start(
            start_epoch,
            version_by_number.get(best_eval.version).created_at if version_by_number.get(best_eval.version) else None,
        ),
        regressions=sum(1 for item in pairwise if item.is_regression),
        critical_errors_remaining=len(final_eval.critical_fail_flags),
        pairwise_win_rate_vs_previous=pairwise_rate,
        version_scores=[
            {
                "version": item.version,
                "overall_score": item.overall_score,
                "weighted_score": item.weighted_score,
                "critical_fail_flags": item.critical_fail_flags,
            }
            for item in evaluations
        ],
        artifacts=artifacts or {},
    )


def _run_start_epoch(run_root: str | Path, versions: list[ReportVersion]) -> float | None:
    root_record = latest_agent_records(run_root).get("agent_root")
    if root_record is not None:
        started = _iso_to_epoch(root_record.started_at) or _iso_to_epoch(root_record.created_at)
        if started is not None:
            return started
    if versions:
        return _iso_to_epoch(versions[0].created_at)
    return None


def _elapsed_from_start(start_epoch: float | None, timestamp: str | None) -> float | None:
    if start_epoch is None:
        return None
    epoch = _iso_to_epoch(timestamp)
    if epoch is None:
        return None
    return round(max(0.0, epoch - start_epoch), 3)


def evaluate_harness_run(
    run_root: str | Path,
    *,
    case_id: str | None = None,
    eval_id: str | None = None,
    output_root: str | Path | None = None,
    judge_fn: JudgeFn | None = None,
    evaluator_model: str | None = None,
    acceptable_threshold: float = DEFAULT_ACCEPTABLE_SCORE,
) -> EvaluationRunResult:
    request = load_request(run_root)
    resolved_case_id = case_id or Path(run_root).name
    resolved_eval_id = eval_id or default_eval_id(resolved_case_id)
    resolved_output_root = Path(output_root) if output_root else _default_output_root(resolved_eval_id, resolved_case_id)
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    model = evaluator_model or get_model("harness", "agent")
    contract_judge = judge_fn or default_judge_fn(model)

    versions = collect_report_versions(run_root, resolved_output_root)
    evidence_context = collect_evidence_context(run_root)
    write_text_atomic(resolved_output_root / "evidence_context.md", evidence_context.rendered + "\n")
    write_json_atomic(resolved_output_root / "evidence_context.json", evidence_context.model_dump(mode="json"))

    contract = generate_eval_contract(
        case_id=resolved_case_id,
        prompt=request.user_prompt,
        judge_fn=contract_judge,
        model_name=model,
    )
    write_json_atomic(resolved_output_root / "eval_contract.json", contract.model_dump(mode="json"))

    evaluations: list[VersionEvaluation] = []
    for version in versions:
        evaluation = score_report_version(
            case_id=resolved_case_id,
            eval_id=resolved_eval_id,
            prompt=request.user_prompt,
            contract=contract,
            version=version,
            evidence_context=evidence_context,
            judge_fn=judge_fn,
            model_name=model,
        )
        evaluations.append(evaluation)
        write_json_atomic(
            resolved_output_root / "versions" / f"v{version.version:04d}.eval.json",
            evaluation.model_dump(mode="json"),
        )

    pairwise: list[PairwiseEvaluation] = []
    for older_version, newer_version, older_eval, newer_eval in zip(
        versions,
        versions[1:],
        evaluations,
        evaluations[1:],
        strict=False,
    ):
        comparison = compare_report_versions(
            case_id=resolved_case_id,
            eval_id=resolved_eval_id,
            prompt=request.user_prompt,
            older_version=older_version,
            newer_version=newer_version,
            older_eval=older_eval,
            newer_eval=newer_eval,
            judge_fn=judge_fn,
            model_name=model,
        )
        pairwise.append(comparison)
        write_json_atomic(
            resolved_output_root / f"pairwise_v{older_version.version:04d}_v{newer_version.version:04d}.json",
            comparison.model_dump(mode="json"),
        )

    artifacts = {
        "contract": str(resolved_output_root / "eval_contract.json"),
        "evidence_context": str(resolved_output_root / "evidence_context.md"),
        "versions_dir": str(resolved_output_root / "versions"),
    }
    trajectory = build_trajectory_summary(
        eval_id=resolved_eval_id,
        case_id=resolved_case_id,
        run_root=run_root,
        output_root=resolved_output_root,
        prompt=request.user_prompt,
        model=model,
        versions=versions,
        evaluations=evaluations,
        pairwise=pairwise,
        acceptable_threshold=acceptable_threshold,
        artifacts=artifacts,
    )
    trajectory_path = resolved_output_root / "trajectory.json"
    write_json_atomic(trajectory_path, trajectory.model_dump(mode="json"))
    artifacts["trajectory"] = str(trajectory_path)
    trajectory = trajectory.model_copy(update={"artifacts": artifacts})
    write_json_atomic(trajectory_path, trajectory.model_dump(mode="json"))

    return EvaluationRunResult(
        eval_id=resolved_eval_id,
        case_id=resolved_case_id,
        output_root=str(resolved_output_root),
        contract=contract,
        versions=versions,
        version_evaluations=evaluations,
        pairwise_evaluations=pairwise,
        trajectory=trajectory,
    )


def run_eval_case(
    *,
    prompt: str,
    case_id: str,
    run_id: str,
    eval_id: str | None = None,
    request_overrides: dict[str, Any] | None = None,
    judge_fn: JudgeFn | None = None,
    evaluator_model: str | None = None,
) -> tuple[HarnessResponse, EvaluationRunResult]:
    from src.harness import run_harness

    request = HarnessRequest(
        user_prompt=prompt,
        run_id=run_id,
        **(request_overrides or {}),
    )
    response = run_harness(request)
    run_root = response.run_root or str(Path.cwd() / "data" / "harness_runs" / run_id)
    result = evaluate_harness_run(
        run_root,
        case_id=case_id,
        eval_id=eval_id,
        judge_fn=judge_fn,
        evaluator_model=evaluator_model,
    )
    return response, result


def copy_eval_outputs(result: EvaluationRunResult, destination: str | Path) -> None:
    target = Path(destination)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(result.output_root, target)
