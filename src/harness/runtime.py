"""Bounded control-loop runtime for the side-by-side harness."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from src.harness.controller import decide_next_step
from src.harness.planner import plan_research, plan_research_fallback, validate_research_plan
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.selector import select_packs
from src.harness.types import (
    ControllerDecision,
    HarnessRequest,
    HarnessResponse,
    HarnessState,
    ResearchBrief,
    ResearchContract,
    ResearchPlan,
    ResearchStep,
    SkillResult,
    SkillSpec,
    StepExecutionResult,
    VerificationReport,
)
from src.harness.verifier import verify_draft
from src.harness.worker import execute_research_step
from src.harness.writer import build_claim_map, write_draft
from src.shared.report_filename import build_prompt_report_filename
from src.shared.text_utils import condense_context


ControllerFn = Callable[[HarnessState], ControllerDecision]
WriterFn = Callable[[HarnessState], str]
VerifierFn = Callable[[HarnessState, str], VerificationReport]
SelectorFn = Callable[[str], list[str]]
PlannerFn = Callable[[HarnessState], tuple[ResearchBrief, ResearchPlan, ResearchContract]]
WorkerFn = Callable[[ResearchStep, HarnessState, dict[str, SkillSpec]], tuple[list[SkillResult], StepExecutionResult]]


def _normalize_domain_packs(selected: list[str] | None) -> list[str]:
    packs = [pack for pack in (selected or []) if pack and pack != "core"]
    seen: set[str] = set()
    normalized: list[str] = []
    for pack in packs:
        if pack not in seen:
            normalized.append(pack)
            seen.add(pack)
    return normalized


def _enabled_packs_from_domain_packs(domain_packs: list[str]) -> list[str]:
    return ["core", *_normalize_domain_packs(domain_packs)]


def initialize_state(
    request: HarnessRequest,
    *,
    selector_fn: SelectorFn | None = None,
    registry: dict[str, SkillSpec] | None = None,
) -> HarnessState:
    """Construct the initial harness state with selected packs and skills."""

    registry_map = registry or build_skill_registry()
    raw_selected = request.selected_packs or (selector_fn or select_packs)(request.user_prompt)
    domain_packs = _normalize_domain_packs(raw_selected)
    enabled_packs = _enabled_packs_from_domain_packs(domain_packs)
    available_skills = get_skills_for_packs(registry_map, enabled_packs)
    return HarnessState(
        request=request,
        enabled_packs=enabled_packs,
        available_skills=available_skills,
    )


def _run_stem(state: HarnessState) -> str:
    report_name = build_prompt_report_filename(
        prompt_text=state.request.user_prompt,
        fallback_stem="harness_report",
        suffix="harness",
    )
    return report_name.replace(".md", "")


def _setup_run_paths(state: HarnessState) -> None:
    reports_dir = os.path.join(os.getcwd(), "reports")
    runs_root = os.path.join(os.getcwd(), "data", "harness_runs")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(runs_root, exist_ok=True)

    run_dir = os.path.join(runs_root, _run_stem(state))
    dossier_dir = os.path.join(run_dir, "dossier")
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(dossier_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    state.run_dir = run_dir
    state.dossier_dir = dossier_dir
    state.mission_path = os.path.join(dossier_dir, "mission.md")
    state.progress_path = os.path.join(dossier_dir, "progress.md")
    state.plan_path = os.path.join(dossier_dir, "research_plan.json")
    state.contract_path = os.path.join(dossier_dir, "research_contract.json")
    state.qa_report_path = os.path.join(dossier_dir, "qa_report.json")
    state.trace_path = os.path.join(run_dir, "trace.json")
    state.report_path = os.path.join(reports_dir, f"{_run_stem(state)}.md")


def _build_trace_payload(state: HarnessState) -> dict[str, object]:
    return {
        "request": state.request.model_dump(mode="json"),
        "enabled_packs": state.enabled_packs,
        "available_skills": [spec.model_dump(mode="json") for spec in state.available_skills],
        "research_brief": state.research_brief.model_dump(mode="json") if state.research_brief else None,
        "research_plan": state.research_plan.model_dump(mode="json") if state.research_plan else None,
        "research_contract": state.research_contract.model_dump(mode="json") if state.research_contract else None,
        "controller_log": [decision.model_dump(mode="json") for decision in state.controller_log],
        "skill_history": [result.model_dump(mode="json") for result in state.skill_history],
        "step_results": [result.model_dump(mode="json") for result in state.step_results],
        "evidence_ledger": [item.model_dump(mode="json") for item in state.evidence_ledger],
        "verification_reports": [report.model_dump(mode="json") for report in state.verification_reports],
        "claim_map": [claim.model_dump(mode="json") for claim in state.claim_map],
        "working_memory": state.working_memory,
        "revision_notes": state.revision_notes,
        "artifacts": state.artifacts,
        "progress_updates": state.progress_updates,
        "completed_step_ids": state.completed_step_ids,
        "failed_step_ids": state.failed_step_ids,
        "pending_follow_up_calls": [call.model_dump(mode="json") for call in state.pending_follow_up_calls],
        "latest_draft": state.latest_draft,
        "final_response": state.final_response,
        "step_count": state.step_count,
        "revision_count": state.revision_count,
        "status": state.status,
        "error": state.error,
        "run_dir": state.run_dir,
        "dossier_dir": state.dossier_dir,
        "mission_path": state.mission_path,
        "progress_path": state.progress_path,
        "plan_path": state.plan_path,
        "contract_path": state.contract_path,
        "qa_report_path": state.qa_report_path,
        "report_path": state.report_path,
        "trace_path": state.trace_path,
        "checkpoint_paths": state.checkpoint_paths,
    }


def _write_json(path: str | None, payload: dict[str, object]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True)


def _write_text(path: str | None, text: str) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _render_mission(state: HarnessState) -> str:
    lines = [
        f"# Mission",
        "",
        f"## User Request",
        state.request.user_prompt,
    ]
    if state.research_brief:
        lines.extend(
            [
                "",
                "## Primary Question",
                state.research_brief.primary_question,
                "",
                "## Domain Packs",
                ", ".join(state.research_brief.domain_packs) or "None",
            ]
        )
    if state.research_contract:
        lines.extend(["", "## Contract Highlights"])
        for clause in state.research_contract.global_clauses[:6]:
            lines.append(f"- {clause.text}")
    return "\n".join(lines).strip() + "\n"


def _render_progress(state: HarnessState) -> str:
    step_status = []
    if state.research_plan:
        result_by_step = {result.step_id: result.status for result in state.step_results}
        for step in state.research_plan.steps:
            status = result_by_step.get(step.id, step.status)
            step_status.append(f"- `{step.id}`: {status} :: {step.objective}")
    updates = [f"- {update}" for update in state.progress_updates[-20:]]
    if not updates:
        updates = ["- No progress updates yet."]

    lines = [
        "# Progress",
        "",
        "## Status",
        f"- runtime status: {state.status}",
        f"- steps executed: {state.step_count}",
        f"- revisions used: {state.revision_count}",
        "",
        "## Step Status",
        *(step_status or ["- No planned steps yet."]),
        "",
        "## Updates",
        *updates,
    ]
    return "\n".join(lines).strip() + "\n"


def _persist_dossier(state: HarnessState) -> None:
    _write_text(state.mission_path, _render_mission(state))
    _write_text(state.progress_path, _render_progress(state))
    if state.research_plan and state.plan_path:
        _write_json(state.plan_path, state.research_plan.model_dump(mode="json"))
    if state.research_contract and state.contract_path:
        _write_json(state.contract_path, state.research_contract.model_dump(mode="json"))
    if state.verification_reports and state.qa_report_path:
        _write_json(state.qa_report_path, state.verification_reports[-1].model_dump(mode="json"))


def _persist_checkpoint(state: HarnessState, label: str) -> None:
    if not state.run_dir:
        return
    checkpoints_dir = os.path.join(state.run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoints_dir,
        f"checkpoint_{len(state.checkpoint_paths) + 1:03d}_{label}.json",
    )
    payload = _build_trace_payload(state)
    payload["checkpoint_label"] = label
    _write_json(checkpoint_path, payload)
    state.checkpoint_paths.append(checkpoint_path)


def _persist_outputs(state: HarnessState) -> None:
    _persist_dossier(state)
    if state.report_path:
        _write_text(state.report_path, state.final_response or "")
    if state.trace_path:
        _write_json(state.trace_path, _build_trace_payload(state))


def _restore_state_from_checkpoint(
    request: HarnessRequest,
    checkpoint_path: str,
    registry_map: dict[str, SkillSpec],
) -> HarnessState:
    payload = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    payload.pop("checkpoint_label", None)
    state = HarnessState.model_validate(payload)
    state.request = request
    state.available_skills = get_skills_for_packs(registry_map, state.enabled_packs)
    return state


def _record_progress(state: HarnessState, message: str) -> None:
    state.progress_updates.append(message)
    if len(state.progress_updates) > 60:
        head = "\n".join(state.progress_updates[:-30])
        condensed = condense_context(
            text=head,
            max_chars=min(state.request.max_chars_before_condense, 3000),
            agent="harness",
            purpose="progress log compaction",
        )
        state.progress_updates = [f"[compacted] {condensed}", *state.progress_updates[-30:]]
    _persist_dossier(state)


def _assign_evidence_ids(state: HarnessState, result: SkillResult) -> None:
    start_index = len(state.evidence_ledger) + 1
    for offset, item in enumerate(result.evidence):
        if not item.id:
            item.id = f"E{start_index + offset}"


def _maybe_condense_output(state: HarnessState, result: SkillResult) -> None:
    output_text = result.output_text or ""
    if len(output_text) <= state.request.max_chars_before_condense:
        return
    condensed = condense_context(
        text=output_text,
        max_chars=state.request.max_chars_before_condense,
        agent="harness",
        purpose=f"{result.skill_name} output review",
    )
    result.output_text = condensed
    result.summary = (
        f"{result.summary} Output was condensed from {len(output_text)} to {len(condensed)} characters."
    )
    result.structured_data["condensed"] = True
    result.structured_data["output_chars_before_condense"] = len(output_text)
    result.structured_data["output_chars_after_condense"] = len(condensed)


def _record_skill_result(state: HarnessState, result: SkillResult) -> None:
    _assign_evidence_ids(state, result)
    _maybe_condense_output(state, result)

    state.skill_history.append(result)
    state.evidence_ledger.extend(result.evidence)
    for path in result.artifacts:
        if path not in state.artifacts:
            state.artifacts.append(path)

    memory_entry = f"[{result.skill_name}] {result.summary}"
    if result.output_text:
        memory_entry += f"\n{result.output_text[:1200]}"
    state.working_memory.append(memory_entry)
    if result.status == "failed" and result.error:
        state.error = result.error


def _record_step_result(state: HarnessState, step_result: StepExecutionResult) -> None:
    state.step_results.append(step_result)
    if step_result.step_id not in state.completed_step_ids and step_result.status in {"completed", "partial"}:
        state.completed_step_ids.append(step_result.step_id)
    if step_result.step_id not in state.failed_step_ids and step_result.status in {"failed", "blocked"}:
        state.failed_step_ids.append(step_result.step_id)
    if state.research_plan:
        for step in state.research_plan.steps:
            if step.id == step_result.step_id:
                step.status = "completed" if step_result.status in {"completed", "partial"} else "failed"
                break
    _record_progress(state, step_result.summary)


def _verify_latest_draft(state: HarnessState, verifier_fn: VerifierFn) -> VerificationReport:
    draft = state.latest_draft or ""
    report = verifier_fn(state, draft)
    state.verification_reports.append(report)
    state.pending_follow_up_calls = report.required_follow_up_calls
    _persist_dossier(state)
    return report


def _finalize_from_draft(state: HarnessState) -> None:
    if state.latest_draft:
        state.final_response = state.latest_draft
        state.status = "completed"
        return
    state.final_response = "Harness failed to generate a draft."
    state.status = "failed"


def _ensure_final_draft(state: HarnessState, writer_fn: WriterFn) -> None:
    if not state.latest_draft:
        state.latest_draft = writer_fn(state)
        state.claim_map = build_claim_map(state, state.latest_draft)


def _all_planned_steps_completed(state: HarnessState) -> bool:
    if not state.research_plan or not state.research_plan.steps:
        return False
    completed = set(state.completed_step_ids)
    return all(step.id in completed for step in state.research_plan.steps)


def _step_map(state: HarnessState) -> dict[str, ResearchStep]:
    if not state.research_plan:
        return {}
    return {step.id: step for step in state.research_plan.steps}


def _execute_skill_call(
    state: HarnessState,
    call_name: str,
    arguments: dict[str, object],
    registry_map: dict[str, SkillSpec],
) -> None:
    spec = registry_map.get(call_name)
    if spec is None or spec.executor is None:
        failed = SkillResult(
            skill_name=call_name,
            arguments=arguments,
            status="failed",
            summary=f"Unknown skill '{call_name}'.",
            error=f"Unknown skill '{call_name}'.",
        )
        _record_skill_result(state, failed)
        _record_progress(state, failed.summary)
        return
    try:
        result = spec.executor(arguments, state)
    except Exception as exc:
        result = SkillResult(
            skill_name=call_name,
            arguments=arguments,
            status="failed",
            summary=f"Skill '{call_name}' raised an exception.",
            error=f"{type(exc).__name__}: {exc}",
        )
    _record_skill_result(state, result)
    _record_progress(state, result.summary)


def _execute_step_batch(
    state: HarnessState,
    step_ids: list[str],
    registry_map: dict[str, SkillSpec],
    worker_fn: WorkerFn,
) -> None:
    step_lookup = _step_map(state)
    steps = [step_lookup[step_id] for step_id in step_ids if step_id in step_lookup]
    if not steps:
        return
    if len(steps) == 1 or not state.request.enable_parallel_steps:
        for step in steps:
            skill_results, step_result = worker_fn(step, state, registry_map)
            for result in skill_results:
                _record_skill_result(state, result)
            _record_step_result(state, step_result)
        return

    with ThreadPoolExecutor(max_workers=len(steps)) as executor:
        futures = {executor.submit(worker_fn, step, state, registry_map): step.id for step in steps}
        for future in as_completed(futures):
            skill_results, step_result = future.result()
            for result in skill_results:
                _record_skill_result(state, result)
            _record_step_result(state, step_result)


def run_harness(
    request: HarnessRequest,
    *,
    controller_fn: ControllerFn | None = None,
    writer_fn: WriterFn | None = None,
    verifier_fn: VerifierFn | None = None,
    selector_fn: SelectorFn | None = None,
    planner_fn: PlannerFn | None = None,
    worker_fn: WorkerFn | None = None,
    registry: dict[str, SkillSpec] | None = None,
) -> HarnessResponse:
    """Execute the harness end to end and persist the report plus trace."""

    controller = controller_fn or decide_next_step
    writer = writer_fn or write_draft
    verifier = verifier_fn or verify_draft
    if planner_fn is not None:
        planner = planner_fn
    elif controller_fn is not None or writer_fn is not None or verifier_fn is not None or worker_fn is not None:
        planner = plan_research_fallback
    else:
        planner = plan_research
    worker = worker_fn or execute_research_step
    registry_map = registry or build_skill_registry()

    if request.resume_from_checkpoint:
        state = _restore_state_from_checkpoint(request, request.resume_from_checkpoint, registry_map)
    else:
        state = initialize_state(request, selector_fn=selector_fn, registry=registry_map)
        _setup_run_paths(state)

    if not state.run_dir:
        _setup_run_paths(state)

    if not state.available_skills:
        state.status = "failed"
        state.final_response = "Harness has no available skills for this request."
        _persist_outputs(state)
        return HarnessResponse(
            final_response=state.final_response,
            status="failed",
            report_path=state.report_path,
            trace_path=state.trace_path,
            enabled_packs=state.enabled_packs,
        )

    if state.research_plan is None:
        brief, plan, contract = planner(state)
        validate_research_plan(plan, {spec.name for spec in state.available_skills})
        state.research_brief = brief
        state.research_plan = plan
        state.research_contract = contract
        state.enabled_packs = _enabled_packs_from_domain_packs(plan.domain_packs)
        state.available_skills = get_skills_for_packs(registry_map, state.enabled_packs)
        _record_progress(state, "Planner produced research brief, plan, and contract.")
        _persist_checkpoint(state, "planned")

    for _ in range(request.max_steps):
        state.step_count += 1
        decision = controller(state)
        state.controller_log.append(decision)

        if decision.action == "call_skill":
            call = decision.skill_call
            if call is None:
                state.error = "Controller returned call_skill without a skill_call payload."
                _record_progress(state, state.error)
                continue
            _execute_skill_call(state, call.name, call.arguments, registry_map)
            _persist_checkpoint(state, "skill")
            continue

        if decision.action == "run_step":
            step_lookup = _step_map(state)
            if decision.step_id and decision.step_id in step_lookup:
                _execute_step_batch(state, [decision.step_id], registry_map, worker)
                _persist_checkpoint(state, "step")
            continue

        if decision.action == "run_steps":
            _execute_step_batch(state, decision.step_ids, registry_map, worker)
            _persist_checkpoint(state, "step_batch")
            continue

        if decision.action == "draft":
            state.latest_draft = writer(state)
            state.claim_map = build_claim_map(state, state.latest_draft)
            _record_progress(state, "Writer produced a draft.")
            report = _verify_latest_draft(state, verifier)
            _persist_checkpoint(state, "evaluated")
            if report.decision == "pass":
                _finalize_from_draft(state)
                break
            if report.decision == "revise" and state.revision_count < request.max_revision_rounds:
                state.revision_count += 1
                state.revision_notes.append(report.summary)
                state.revision_notes.extend(report.improvement_instructions)
                _record_progress(state, "Evaluator requested revision.")
                continue
            _finalize_from_draft(state)
            break

        if decision.action == "finalize":
            _ensure_final_draft(state, writer)
            _record_progress(state, "Controller requested finalization.")
            if not state.verification_reports or state.verification_reports[-1].decision == "revise":
                report = _verify_latest_draft(state, verifier)
                _persist_checkpoint(state, "final_verify")
                if report.decision == "revise" and state.revision_count < request.max_revision_rounds:
                    state.revision_count += 1
                    state.revision_notes.append(report.summary)
                    state.revision_notes.extend(report.improvement_instructions)
                    _record_progress(state, "Evaluator requested revision during finalize.")
                    continue
            _finalize_from_draft(state)
            break

    else:
        _ensure_final_draft(state, writer)
        if not state.verification_reports:
            _verify_latest_draft(state, verifier)
        _finalize_from_draft(state)

    if state.status == "running":
        if _all_planned_steps_completed(state):
            _ensure_final_draft(state, writer)
        _finalize_from_draft(state)

    _persist_outputs(state)

    return HarnessResponse(
        final_response=state.final_response or "",
        status="failed" if state.status == "failed" else "completed",
        report_path=state.report_path,
        trace_path=state.trace_path,
        verification=state.verification_reports[-1] if state.verification_reports else None,
        enabled_packs=state.enabled_packs,
        skills_used=[result.skill_name for result in state.skill_history],
        artifacts=state.artifacts,
    )
