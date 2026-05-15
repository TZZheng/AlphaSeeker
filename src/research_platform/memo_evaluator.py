"""Post-harness LLM evaluator for investment memo deliverables."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from src.research_platform.memo_artifacts import MemoArtifactSet
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model

JudgeFn = Callable[[str, str], str]
CRITICAL_INPUTS = {"final_md", "source_index_md", "source_bodies", "source_use_table_md"}
DEFAULT_SOURCE_BODY_CHAR_LIMIT = 8_192
DEFAULT_TOTAL_SOURCE_BODY_CHAR_LIMIT = 65_536


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_text(path: str | Path | None) -> str:
    if not path:
        return ""
    target = Path(path)
    if not target.exists() or not target.is_file():
        return ""
    return target.read_text(encoding="utf-8", errors="replace")


def _truncate_text(text: str, limit: int) -> tuple[str, bool]:
    if limit < 0 or len(text) <= limit:
        return text, False
    omitted = len(text) - limit
    return text[: max(0, limit - 80)].rstrip() + f"\n\n[truncated {omitted} chars]", True


def _strip_provider_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned


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
            f"Previous response:\n{_truncate_text(first_response, 4000)[0]}\n\n"
            "Return only one valid JSON object. Do not include markdown fences or commentary."
        )
        return _extract_json_object(judge(system_prompt, repair_prompt))


def _llm_response_text(response: Any) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(getattr(item, "text", None), str):
                parts.append(item.text)
        return "\n".join(parts)
    return str(content)


def resolve_evaluator_model(model_name: str | None = None) -> str:
    if model_name:
        return model_name
    env_model = os.environ.get("ALPHASEEKER_MODEL_MEMO_EVALUATOR")
    if env_model:
        return env_model
    return get_model("harness", "agent")


def default_judge_fn(model_name: str | None = None) -> JudgeFn:
    resolved_model = resolve_evaluator_model(model_name)
    llm = get_llm(resolved_model)

    def _judge(system_prompt: str, user_prompt: str) -> str:
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        return _llm_response_text(response)

    return _judge


class MemoEvaluatorInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    ticker: str
    final_md: str = ""
    source_index_md: str = ""
    source_bodies: dict[str, str] = Field(default_factory=dict)
    source_use_table_md: str = ""
    research_state_summary: str | None = None
    optional_traces: dict[str, str] = Field(default_factory=dict)
    harness_tool_calls_jsonl: str | None = None
    inputs_received: list[str] = Field(default_factory=list)
    inputs_missing: list[str] = Field(default_factory=list)
    truncation_warnings: list[str] = Field(default_factory=list)


class MemoEvaluatorVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["pass", "warn", "fail"]
    rationale: str
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source_review: str
    logic_review: str
    trace_review: str
    link_review: str
    state_linkage_review: str | None = None
    inputs_received: list[str] = Field(default_factory=list)
    inputs_missing: list[str] = Field(default_factory=list)
    evaluator_model: str
    prompt_sha256: str
    inputs_sha256: str
    created_at: str


class EvaluatorEvaluationRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "missing", "timeout", "error", "unparseable"]
    path: str | None = None
    reason: list[str] = Field(default_factory=list)
    verdict: MemoEvaluatorVerdict | None = None


def build_evaluator_inputs(
    *,
    run_id: str,
    ticker: str,
    memo_dir: str | Path,
    source_index_path: str | Path | None,
    source_body_paths: dict[str, str | Path] | None,
    artifacts: MemoArtifactSet,
    research_state_summary: str | None = None,
    optional_trace_paths: dict[str, str | Path] | None = None,
    max_source_body_chars: int = DEFAULT_SOURCE_BODY_CHAR_LIMIT,
    max_total_source_body_chars: int = DEFAULT_TOTAL_SOURCE_BODY_CHAR_LIMIT,
) -> MemoEvaluatorInputs:
    del memo_dir  # currently only part of public builder signature
    inputs_received: list[str] = []
    inputs_missing: list[str] = []
    truncation_warnings: list[str] = []

    final_md = _read_text(artifacts.paths.product_final_path or artifacts.paths.root_final_path)
    if final_md:
        inputs_received.append("final_md")
    else:
        inputs_missing.append("final_md")

    source_index_md = _read_text(source_index_path)
    if source_index_md:
        inputs_received.append("source_index_md")
    else:
        inputs_missing.append("source_index_md")

    source_use_table_md = _read_text(artifacts.paths.product_source_use_table_path or artifacts.paths.root_source_use_table_path)
    if source_use_table_md:
        inputs_received.append("source_use_table_md")
    else:
        inputs_missing.append("source_use_table_md")

    source_bodies: dict[str, str] = {}
    total = 0
    for key, path in (source_body_paths or {}).items():
        text = _read_text(path)
        if not text:
            continue
        remaining = max(0, max_total_source_body_chars - total)
        limit = min(max_source_body_chars, remaining)
        truncated, did_truncate = _truncate_text(text, limit)
        source_bodies[key] = truncated
        total += len(truncated)
        if did_truncate:
            truncation_warnings.append(f"source body {key} truncated to {limit} chars")
        if total >= max_total_source_body_chars:
            break
    if source_bodies:
        inputs_received.append("source_bodies")
    else:
        inputs_missing.append("source_bodies")

    tool_log = _read_text(artifacts.paths.harness_tool_calls_path)
    if tool_log:
        inputs_received.append("harness_tool_calls_jsonl")
    else:
        inputs_missing.append("harness_tool_calls_jsonl")

    optional_traces: dict[str, str] = {}
    combined_optional = dict(artifacts.paths.optional_root_publish_paths)
    if optional_trace_paths:
        combined_optional.update({k: str(v) for k, v in optional_trace_paths.items()})
    for name, path in combined_optional.items():
        text = _read_text(path)
        if text:
            optional_traces[name] = text
    if optional_traces:
        inputs_received.append("optional_traces")

    return MemoEvaluatorInputs(
        run_id=run_id,
        ticker=ticker,
        final_md=final_md,
        source_index_md=source_index_md,
        source_bodies=source_bodies,
        source_use_table_md=source_use_table_md,
        research_state_summary=research_state_summary,
        optional_traces=optional_traces,
        harness_tool_calls_jsonl=tool_log or None,
        inputs_received=inputs_received,
        inputs_missing=inputs_missing,
        truncation_warnings=truncation_warnings,
    )


def _system_prompt() -> str:
    return """You are AlphaSeeker's post-harness investment memo evaluator. Judge the memo against the provided evidence binder. Return only JSON matching the requested schema. The system/Python owns provenance and status; your job is semantic judgment of source support, logic, traceability, links, and state linkage."""


def _user_prompt(inputs: MemoEvaluatorInputs) -> str:
    schema = {
        "status": "pass|warn|fail",
        "rationale": "string",
        "blockers": ["string"],
        "warnings": ["string"],
        "source_review": "string",
        "logic_review": "string",
        "trace_review": "string",
        "link_review": "string",
        "state_linkage_review": "string|null",
        "inputs_received": ["string"],
        "inputs_missing": ["string"],
    }
    return "\n\n".join(
        [
            f"Run ID: {inputs.run_id}\nTicker: {inputs.ticker}",
            f"Required output JSON schema:\n{json.dumps(schema, indent=2)}",
            f"Python-authoritative inputs_received: {inputs.inputs_received}",
            f"Python-authoritative inputs_missing: {inputs.inputs_missing}",
            f"Truncation warnings: {inputs.truncation_warnings}",
            f"# final.md\n{inputs.final_md}",
            f"# source_use_table.md\n{inputs.source_use_table_md}",
            f"# source_index.md\n{inputs.source_index_md}",
            f"# source bodies\n{json.dumps(inputs.source_bodies, ensure_ascii=False, indent=2)}",
            f"# optional traces\n{json.dumps(inputs.optional_traces, ensure_ascii=False, indent=2)}",
            f"# harness tool calls\n{inputs.harness_tool_calls_jsonl or ''}",
            f"# research state summary\n{inputs.research_state_summary or ''}",
        ]
    )


def _model_same_as_root(model_name: str) -> bool:
    try:
        return model_name == get_model("harness", "agent")
    except Exception:
        return False


def _finalize_verdict(payload: dict[str, Any], *, inputs: MemoEvaluatorInputs, system_prompt: str, user_prompt: str, model_name: str) -> MemoEvaluatorVerdict:
    payload = dict(payload)
    payload["evaluator_model"] = model_name
    payload["prompt_sha256"] = _sha256(system_prompt + "\n" + user_prompt)
    payload["inputs_sha256"] = _sha256(inputs.model_dump_json())
    payload["created_at"] = _utc_now_iso()
    payload.setdefault("inputs_received", [])
    payload.setdefault("inputs_missing", [])
    verdict = MemoEvaluatorVerdict.model_validate(payload)

    missing_critical = sorted(CRITICAL_INPUTS.intersection(inputs.inputs_missing))
    if verdict.status == "pass" and missing_critical:
        warnings = list(verdict.warnings)
        warnings.append(f"Downgraded: missing critical evaluator input(s): {', '.join(missing_critical)}")
        verdict = verdict.model_copy(update={"status": "warn", "warnings": warnings})
    return verdict


def run_memo_evaluator(inputs: MemoEvaluatorInputs, *, judge_fn: JudgeFn | None = None, model_name: str | None = None) -> MemoEvaluatorVerdict:
    resolved_model = resolve_evaluator_model(model_name)
    judge = judge_fn or default_judge_fn(resolved_model)
    system_prompt = _system_prompt()
    user_prompt = _user_prompt(inputs)
    payload = _judge_json(judge, system_prompt, user_prompt)
    return _finalize_verdict(payload, inputs=inputs, system_prompt=system_prompt, user_prompt=user_prompt, model_name=resolved_model)


def _review_markdown(verdict: MemoEvaluatorVerdict | None, ref: EvaluatorEvaluationRef) -> str:
    if verdict is None:
        return f"# Deliverable Review\n\nEvaluator status: `{ref.status}`\n\nReasons:\n" + "\n".join(f"- {r}" for r in ref.reason) + "\n"
    return (
        f"# Deliverable Review\n\nStatus: `{verdict.status}`\n\n{verdict.rationale}\n\n"
        f"## Blockers\n" + "\n".join(f"- {b}" for b in verdict.blockers or ["None"]) + "\n\n"
        f"## Warnings\n" + "\n".join(f"- {w}" for w in verdict.warnings or ["None"]) + "\n"
    )


def write_evaluator_evaluation(*, memo_dir: str | Path, ref: EvaluatorEvaluationRef) -> EvaluatorEvaluationRef:
    target_dir = Path(memo_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "deliverable_evaluation.json"
    review_path = target_dir / "deliverable_review.md"
    ref = ref.model_copy(update={"path": str(json_path)})
    json_path.write_text(ref.model_dump_json(indent=2), encoding="utf-8")
    review_path.write_text(_review_markdown(ref.verdict, ref), encoding="utf-8")
    return ref


def run_and_write_memo_evaluation(
    inputs: MemoEvaluatorInputs,
    *,
    memo_dir: str | Path,
    judge_fn: JudgeFn | None = None,
    model_name: str | None = None,
) -> EvaluatorEvaluationRef:
    reasons: list[str] = []
    resolved_model = resolve_evaluator_model(model_name)
    if _model_same_as_root(resolved_model):
        reasons.append("evaluator_model_same_as_root")
    try:
        verdict = run_memo_evaluator(inputs, judge_fn=judge_fn, model_name=resolved_model)
        ref = EvaluatorEvaluationRef(status="ok", reason=reasons, verdict=verdict)
    except TimeoutError as exc:
        ref = EvaluatorEvaluationRef(status="timeout", reason=[*reasons, f"{type(exc).__name__}: {exc}"], verdict=None)
    except (ValueError, json.JSONDecodeError) as exc:
        ref = EvaluatorEvaluationRef(status="unparseable", reason=[*reasons, f"{type(exc).__name__}: {exc}"], verdict=None)
    except Exception as exc:
        ref = EvaluatorEvaluationRef(status="error", reason=[*reasons, f"{type(exc).__name__}: {exc}"], verdict=None)
    return write_evaluator_evaluation(memo_dir=memo_dir, ref=ref)
