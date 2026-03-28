"""Persistent dossier, checkpoint, and trace helpers for the harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.harness.types import (
    ClaimRecord,
    CoverageMatrix,
    DiscoveredSource,
    FactIndexRecord,
    HarnessState,
    ReadQueueEntry,
    ReadResultRecord,
    RetrievalQueryBucket,
    SectionBrief,
    SourceCard,
    VerificationReport,
)
from src.shared.report_filename import build_prompt_report_filename


def build_run_paths(state: HarnessState) -> dict[str, str]:
    """Return deterministic filesystem paths for the current run."""

    if state.run_id:
        run_id = state.run_id
    else:
        report_name = build_prompt_report_filename(
            prompt_text=state.request.user_prompt,
            fallback_stem="harness_report",
            suffix="harness",
        )
        run_id = report_name.removesuffix(".md")
        state.run_id = run_id

    root = Path.cwd()
    run_root = root / "data" / "harness_runs" / run_id
    checkpoints_dir = run_root / "checkpoints"
    report_path = root / "reports" / f"{run_id}.md"
    trace_path = run_root / "trace.json"
    return {
        "run_root": str(run_root),
        "mission": str(run_root / "mission.md"),
        "progress": str(run_root / "progress.md"),
        "research_plan": str(run_root / "research_plan.json"),
        "research_contract": str(run_root / "research_contract.json"),
        "qa_report": str(run_root / "qa_report.json"),
        "findings": str(run_root / "findings.json"),
        "claim_map": str(run_root / "claim_map.json"),
        "discovered_sources": str(run_root / "discovered_sources.json"),
        "read_queue": str(run_root / "read_queue.json"),
        "read_results": str(run_root / "read_results.json"),
        "source_cards": str(run_root / "source_cards.jsonl"),
        "fact_index": str(run_root / "fact_index.json"),
        "section_briefs": str(run_root / "section_briefs.json"),
        "coverage_matrix": str(run_root / "coverage_matrix.json"),
        "checkpoints_dir": str(checkpoints_dir),
        "report": str(report_path),
        "trace": str(trace_path),
    }


def ensure_run_directories(paths: dict[str, str]) -> None:
    Path(paths["run_root"]).mkdir(parents=True, exist_ok=True)
    Path(paths["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["report"]).parent.mkdir(parents=True, exist_ok=True)


def _write_text(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def _write_json(path: str, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
    if text:
        text += "\n"
    Path(path).write_text(text, encoding="utf-8")


def _read_text(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8")


def _read_json(path: str) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _render_mission(state: HarnessState) -> str:
    brief = state.research_brief
    lines = [
        "# Mission",
        "",
        "## Original User Question",
        state.request.user_prompt.strip(),
        "",
        "## Current Objective",
        (brief.primary_question if brief else state.request.user_prompt).strip(),
        "",
    ]

    user_constraints = brief.user_constraints if brief else []
    lines.extend(["## User Constraints"])
    if user_constraints:
        lines.extend(f"- {item}" for item in user_constraints[:10])
    else:
        lines.append("- No explicit user constraints were extracted.")
    lines.append("")

    lines.extend(["## Important Definitions"])
    if brief and brief.sub_questions:
        lines.extend(f"- {item}" for item in brief.sub_questions[:8])
    else:
        lines.append("- Definitions are still being refined from the prompt and evidence.")
    lines.append("")

    lines.extend(["## Success Criteria"])
    if state.research_contract:
        clauses = state.research_contract.global_clauses[:4]
        if clauses:
            lines.extend(f"- {clause.text}" for clause in clauses)
        else:
            lines.append("- Produce a grounded, well-cited report.")
    else:
        lines.append("- Produce a grounded, well-cited report.")

    return "\n".join(lines).strip() + "\n"


def _compact_items(items: list[str], limit: int) -> list[str]:
    return [item for item in items if item.strip()][-limit:]


def _render_progress(state: HarnessState) -> str:
    plan = state.research_plan
    completed_steps = [
        step_id
        for step_id, status in state.step_statuses.items()
        if status in {"completed", "partial"}
    ]
    open_steps: list[str] = []
    if plan:
        for step in plan.steps:
            status = state.step_statuses.get(step.id, "pending")
            if status not in {"completed", "partial", "skipped"}:
                open_steps.append(f"{step.id}: {step.objective} [{status}]")

    next_actions: list[str] = []
    if plan:
        for step in plan.steps:
            if state.step_statuses.get(step.id, "pending") == "pending":
                dependencies_met = all(
                    state.step_statuses.get(dep) in {"completed", "partial", "skipped"}
                    for dep in step.depends_on
                )
                if dependencies_met:
                    next_actions.append(f"Run step {step.id}: {step.objective}")
        next_actions = next_actions[:6]

    unresolved = []
    for result in state.step_results[-10:]:
        unresolved.extend(result.open_questions)

    key_findings = _compact_items(state.findings, 12)
    phase_updates = _compact_items(
        [f"{update.phase}: {update.summary}" for update in state.phase_history],
        20,
    )

    lines = [
        "# Progress",
        "",
        "## Major Phase Log",
    ]
    if phase_updates:
        lines.extend(f"- {item}" for item in phase_updates)
    else:
        lines.append("- Run initialized.")
    lines.append("")

    lines.extend(["## Completed Steps"])
    if completed_steps:
        lines.extend(f"- {step_id}" for step_id in completed_steps[-12:])
    else:
        lines.append("- No completed steps yet.")
    lines.append("")

    lines.extend(["## Open Steps"])
    if open_steps:
        lines.extend(f"- {item}" for item in open_steps[:12])
    else:
        lines.append("- No open steps.")
    lines.append("")

    lines.extend(["## Key Findings"])
    if key_findings:
        lines.extend(f"- {item}" for item in key_findings)
    else:
        lines.append("- No key findings recorded yet.")
    lines.append("")

    lines.extend(["## Unresolved Questions"])
    unresolved_items = _compact_items(unresolved, 12)
    if unresolved_items:
        lines.extend(f"- {item}" for item in unresolved_items)
    else:
        lines.append("- No unresolved questions.")
    lines.append("")

    lines.extend(["## Next Scheduled Actions"])
    if next_actions:
        lines.extend(f"- {item}" for item in next_actions)
    else:
        lines.append("- No further actions scheduled.")

    return "\n".join(lines).strip() + "\n"


def _qa_payload(report: VerificationReport | None) -> dict[str, Any]:
    if report is None:
        return {
            "decision": "revise",
            "summary": "QA has not run yet.",
            "blocking_issues": [],
            "missing_sections": [],
            "report_section_feedback": [],
        }
    return report.model_dump(mode="json")


def sync_reduction_artifacts(state: HarnessState) -> None:
    """Persist deep-research artifacts independently of the full dossier refresh."""

    if not state.dossier_paths:
        state.dossier_paths = build_run_paths(state)
    ensure_run_directories(state.dossier_paths)
    _write_json(
        state.dossier_paths["discovered_sources"],
        {"query_buckets": [bucket.model_dump(mode="json") for bucket in state.deep_query_buckets],
         "sources": [source.model_dump(mode="json") for source in state.discovered_sources]},
    )
    _write_json(
        state.dossier_paths["read_queue"],
        {"queue": [entry.model_dump(mode="json") for entry in state.read_queue]},
    )
    _write_json(
        state.dossier_paths["read_results"],
        {"results": [entry.model_dump(mode="json") for entry in state.read_results]},
    )
    _write_jsonl(
        state.dossier_paths["source_cards"],
        [card.model_dump(mode="json") for card in state.source_cards],
    )
    _write_json(
        state.dossier_paths["fact_index"],
        {"facts": [record.model_dump(mode="json") for record in state.fact_index]},
    )
    _write_json(
        state.dossier_paths["section_briefs"],
        {"sections": [brief.model_dump(mode="json") for brief in state.section_briefs]},
    )
    _write_json(
        state.dossier_paths["coverage_matrix"],
        state.coverage_matrix.model_dump(mode="json") if state.coverage_matrix else {},
    )


def sync_dossier(state: HarnessState) -> None:
    """Persist the mission/progress dossier and structured artifacts."""

    if not state.dossier_paths:
        state.dossier_paths = build_run_paths(state)
    ensure_run_directories(state.dossier_paths)
    state.run_root = state.dossier_paths["run_root"]

    mission_text = _render_mission(state)
    progress_text = _render_progress(state)
    state.mission_text = mission_text
    state.progress_text = progress_text

    _write_text(state.dossier_paths["mission"], mission_text)
    _write_text(state.dossier_paths["progress"], progress_text)
    _write_json(
        state.dossier_paths["research_plan"],
        state.research_plan.model_dump(mode="json") if state.research_plan else {},
    )
    _write_json(
        state.dossier_paths["research_contract"],
        state.research_contract.model_dump(mode="json") if state.research_contract else {},
    )
    _write_json(state.dossier_paths["qa_report"], _qa_payload(state.verification_reports[-1] if state.verification_reports else None))
    _write_json(
        state.dossier_paths["findings"],
        {
            "findings": state.findings[-50:],
            "step_results": [result.model_dump(mode="json") for result in state.step_results[-50:]],
        },
    )
    _write_json(
        state.dossier_paths["claim_map"],
        {"claims": [claim.model_dump(mode="json") for claim in state.claim_map]},
    )
    sync_reduction_artifacts(state)


def refresh_dossier(state: HarnessState) -> None:
    """Reread dossier files into memory before a major model phase."""

    if not state.dossier_paths:
        state.dossier_paths = build_run_paths(state)
    state.run_root = state.dossier_paths["run_root"]
    state.mission_text = _read_text(state.dossier_paths["mission"])
    state.progress_text = _read_text(state.dossier_paths["progress"])

    plan_payload = _read_json(state.dossier_paths["research_plan"])
    if isinstance(plan_payload, dict) and plan_payload:
        from src.harness.types import ResearchPlan

        state.research_plan = ResearchPlan.model_validate(plan_payload)

    contract_payload = _read_json(state.dossier_paths["research_contract"])
    if isinstance(contract_payload, dict) and contract_payload:
        from src.harness.types import ResearchContract

        state.research_contract = ResearchContract.model_validate(contract_payload)

    claim_payload = _read_json(state.dossier_paths["claim_map"])
    if isinstance(claim_payload, dict):
        claims = claim_payload.get("claims")
        if isinstance(claims, list):
            state.claim_map = [ClaimRecord.model_validate(item) for item in claims]

    discovered_payload = _read_json(state.dossier_paths["discovered_sources"])
    if isinstance(discovered_payload, dict):
        query_buckets = discovered_payload.get("query_buckets")
        if isinstance(query_buckets, list):
            state.deep_query_buckets = [
                RetrievalQueryBucket.model_validate(item) for item in query_buckets
            ]
        sources = discovered_payload.get("sources")
        if isinstance(sources, list):
            state.discovered_sources = [
                DiscoveredSource.model_validate(item) for item in sources
            ]

    read_queue_payload = _read_json(state.dossier_paths["read_queue"])
    if isinstance(read_queue_payload, dict):
        queue_items = read_queue_payload.get("queue")
        if isinstance(queue_items, list):
            state.read_queue = [ReadQueueEntry.model_validate(item) for item in queue_items]

    read_results_payload = _read_json(state.dossier_paths["read_results"])
    if isinstance(read_results_payload, dict):
        read_items = read_results_payload.get("results")
        if isinstance(read_items, list):
            state.read_results = [
                ReadResultRecord.model_validate(item) for item in read_items
            ]

    source_card_rows = _read_jsonl(state.dossier_paths["source_cards"])
    if source_card_rows:
        state.source_cards = [SourceCard.model_validate(item) for item in source_card_rows]

    fact_payload = _read_json(state.dossier_paths["fact_index"])
    if isinstance(fact_payload, dict):
        facts = fact_payload.get("facts")
        if isinstance(facts, list):
            state.fact_index = [FactIndexRecord.model_validate(item) for item in facts]

    section_payload = _read_json(state.dossier_paths["section_briefs"])
    if isinstance(section_payload, dict):
        sections = section_payload.get("sections")
        if isinstance(sections, list):
            state.section_briefs = [SectionBrief.model_validate(item) for item in sections]

    coverage_payload = _read_json(state.dossier_paths["coverage_matrix"])
    if isinstance(coverage_payload, dict) and coverage_payload:
        state.coverage_matrix = CoverageMatrix.model_validate(coverage_payload)


def initialize_dossier(state: HarnessState) -> None:
    """Create the initial dossier files on disk."""

    state.dossier_paths = build_run_paths(state)
    ensure_run_directories(state.dossier_paths)
    sync_dossier(state)
    refresh_dossier(state)


def build_state_snapshot(state: HarnessState) -> dict[str, Any]:
    """Serialize the current harness state for checkpoints and trace output."""

    return {
        "state": state.model_dump(mode="json"),
        "dossier_paths": state.dossier_paths,
    }


def write_checkpoint(state: HarnessState, label: str) -> str:
    """Persist a stepwise checkpoint in run order."""

    if not state.dossier_paths:
        initialize_dossier(state)
    checkpoint_index = len(state.checkpoint_paths) + 1
    checkpoint_path = (
        Path(state.dossier_paths["checkpoints_dir"])
        / f"checkpoint_step_{checkpoint_index:03d}_{label}.json"
    )
    _write_json(str(checkpoint_path), build_state_snapshot(state))
    state.checkpoint_paths.append(str(checkpoint_path))
    return str(checkpoint_path)


def persist_report(state: HarnessState) -> None:
    if not state.dossier_paths:
        initialize_dossier(state)
    report_path = state.dossier_paths["report"]
    _write_text(report_path, state.final_response or "")
    state.report_path = report_path


def persist_trace(state: HarnessState, payload: dict[str, Any]) -> None:
    if not state.dossier_paths:
        initialize_dossier(state)
    trace_path = state.dossier_paths["trace"]
    _write_json(trace_path, payload)
    state.trace_path = trace_path
