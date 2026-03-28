"""Dependency-aware harness runtime with persistent dossier artifacts."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from src.harness.artifacts import (
    build_state_snapshot,
    initialize_dossier,
    persist_report,
    persist_trace,
    refresh_dossier,
    sync_dossier,
    write_checkpoint,
)
from src.harness.claims import build_claim_map
from src.harness.controller import decide_next_step
from src.harness.planner import plan_research
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.types import (
    ControllerDecision,
    DOMAIN_PACKS,
    HarnessRequest,
    HarnessResponse,
    HarnessState,
    PhaseUpdate,
    ResearchBrief,
    ResearchContract,
    ResearchPlan,
    ResearchStep,
    SkillResult,
    StepExecutionResult,
    VerificationReport,
)
from src.harness.worker import execute_research_step, execute_skill_call
from src.harness.writer import write_draft
from src.harness.verifier import verify_draft
from src.shared.text_utils import condense_context


ControllerFn = Callable[[HarnessState], ControllerDecision]
WriterFn = Callable[[HarnessState], str]
VerifierFn = Callable[[HarnessState, str], VerificationReport]
PlannerFn = Callable[[HarnessState, dict[str, object]], tuple[ResearchBrief, ResearchPlan, ResearchContract]]
WorkerFn = Callable[[HarnessState, ResearchStep, dict[str, object]], StepExecutionResult]


def _domain_override(request: HarnessRequest) -> list[str] | None:
    if request.selected_packs is None:
        return None
    selected = []
    for pack in request.selected_packs:
        pack_name = pack.strip().lower()
        if pack_name in DOMAIN_PACKS and pack_name not in selected:
            selected.append(pack_name)
    return selected


def initialize_state(
    request: HarnessRequest,
    *,
    registry: dict[str, object] | None = None,
) -> HarnessState:
    """Construct the initial harness state before planning."""

    registry_map = registry or build_skill_registry()
    state = HarnessState(
        request=request,
        available_skills=list(registry_map.values()),
    )
    initialize_dossier(state)
    return state


def _restore_state_from_checkpoint(
    request: HarnessRequest,
    *,
    registry: dict[str, object] | None = None,
) -> HarnessState:
    """Rehydrate runtime state from a previously written checkpoint."""

    checkpoint_path = request.resume_from
    if not checkpoint_path:
        raise ValueError("resume_from is required to restore a checkpoint.")
    payload = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    state_payload = payload.get("state")
    if not isinstance(state_payload, dict):
        raise ValueError("Checkpoint payload is missing a serialized state.")

    state = HarnessState.model_validate(state_payload)
    state.request = request.model_copy(update={"user_prompt": state.request.user_prompt})
    state.dossier_paths = payload.get("dossier_paths") or state.dossier_paths
    registry_map = registry or build_skill_registry()
    state.available_skills = get_skills_for_packs(registry_map, state.enabled_packs or ["core"])
    refresh_dossier(state)
    return state


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


def _record_phase(state: HarnessState, phase: str, summary: str) -> None:
    state.phase_history.append(PhaseUpdate(phase=phase, summary=summary))
    sync_dossier(state)
    write_checkpoint(state, phase)


def _refresh_context(state: HarnessState) -> None:
    refresh_dossier(state)
    if state.request.reset_context_each_phase:
        state.working_memory = state.working_memory[-4:]
        state.revision_notes = state.revision_notes[-8:]


def _apply_plan_to_state(
    state: HarnessState,
    brief: ResearchBrief,
    plan: ResearchPlan,
    contract: ResearchContract,
    registry_map: dict[str, object],
) -> None:
    state.research_brief = brief
    state.research_plan = plan
    state.research_contract = contract
    state.enabled_packs = ["core", *plan.domain_packs]
    state.available_skills = get_skills_for_packs(registry_map, state.enabled_packs)
    for step in plan.steps:
        state.step_statuses.setdefault(step.id, "pending")


def _run_planner(
    state: HarnessState,
    registry_map: dict[str, object],
    planner_fn: PlannerFn | None,
) -> None:
    _refresh_context(state)
    override = _domain_override(state.request)
    if planner_fn is not None:
        brief, plan, contract = planner_fn(state, registry_map)
    else:
        brief, plan, contract = plan_research(
            state.request.user_prompt,
            registry=registry_map,
            selected_packs=override,
        )
    _apply_plan_to_state(state, brief, plan, contract, registry_map)
    _record_phase(
        state,
        "planner",
        f"Planned {len(plan.steps)} steps across packs: {', '.join(state.enabled_packs)}.",
    )


def _step_by_id(state: HarnessState, step_id: str) -> ResearchStep | None:
    if not state.research_plan:
        return None
    for step in state.research_plan.steps:
        if step.id == step_id:
            return step
    return None


def _record_step_execution(state: HarnessState, result: StepExecutionResult) -> None:
    state.step_results.append(result)
    state.step_statuses[result.step_id] = result.status
    for finding in result.findings:
        if finding not in state.findings:
            state.findings.append(finding)
    for path in result.artifact_paths:
        if path not in state.artifacts:
            state.artifacts.append(path)


def _run_step_worker(
    state: HarnessState,
    step: ResearchStep,
    registry_map: dict[str, object],
    worker_fn: WorkerFn | None,
) -> StepExecutionResult:
    if worker_fn is not None:
        return worker_fn(state, step, registry_map)
    return execute_research_step(
        state,
        step,
        registry_map,
        record_skill_result=_record_skill_result,
        max_iterations=state.request.max_worker_iterations,
    )


def _run_parallel_steps(
    state: HarnessState,
    step_ids: list[str],
    registry_map: dict[str, object],
    worker_fn: WorkerFn | None,
) -> list[StepExecutionResult]:
    if worker_fn is not None:
        return [
            _run_step_worker(state, step, registry_map, worker_fn)
            for step in (_step_by_id(state, step_id) for step_id in step_ids)
            if step is not None
        ]

    def _worker_job(step: ResearchStep) -> tuple[str, StepExecutionResult, list[SkillResult]]:
        local_state = state.model_copy(deep=True)
        local_results: list[SkillResult] = []

        def _local_record(_local_state: HarnessState, result: SkillResult) -> None:
            _assign_evidence_ids(local_state, result)
            _maybe_condense_output(local_state, result)
            local_results.append(result)
            local_state.evidence_ledger.extend(result.evidence)

        step_result = execute_research_step(
            local_state,
            step,
            registry_map,
            record_skill_result=_local_record,
            max_iterations=state.request.max_worker_iterations,
        )
        return step.id, step_result, local_results

    ordered_steps = [
        step
        for step in (_step_by_id(state, step_id) for step_id in step_ids)
        if step is not None
    ]
    results_by_step: dict[str, tuple[StepExecutionResult, list[SkillResult]]] = {}
    with ThreadPoolExecutor(max_workers=min(len(ordered_steps), state.request.max_parallel_steps)) as pool:
        future_map = {pool.submit(_worker_job, step): step.id for step in ordered_steps}
        for future in as_completed(future_map):
            step_id, step_result, skill_results = future.result()
            results_by_step[step_id] = (step_result, skill_results)

    ordered_results: list[StepExecutionResult] = []
    for step in ordered_steps:
        step_result, skill_results = results_by_step[step.id]
        for result in skill_results:
            _record_skill_result(state, result)
        ordered_results.append(step_result)
    return ordered_results


def _verify_latest_draft(state: HarnessState, verifier_fn: VerifierFn) -> VerificationReport:
    draft = state.latest_draft or ""
    report = verifier_fn(state, draft)
    state.verification_reports.append(report)
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
        _refresh_context(state)
        state.latest_draft = writer_fn(state)
        state.claim_map = build_claim_map(state, state.latest_draft)
        _record_phase(state, "writer", "Generated a final fallback draft.")


def _build_trace_payload(state: HarnessState) -> dict[str, object]:
    payload = build_state_snapshot(state)
    payload.update(
        {
            "request": state.request.model_dump(mode="json"),
            "enabled_packs": state.enabled_packs,
            "available_skills": [spec.model_dump(mode="json") for spec in state.available_skills],
            "controller_log": [decision.model_dump(mode="json") for decision in state.controller_log],
            "skill_history": [result.model_dump(mode="json") for result in state.skill_history],
            "step_results": [result.model_dump(mode="json") for result in state.step_results],
            "evidence_ledger": [item.model_dump(mode="json") for item in state.evidence_ledger],
            "verification_reports": [report.model_dump(mode="json") for report in state.verification_reports],
            "phase_history": [update.model_dump(mode="json") for update in state.phase_history],
            "claim_map": [claim.model_dump(mode="json") for claim in state.claim_map],
            "working_memory": state.working_memory,
            "revision_notes": state.revision_notes,
            "artifacts": state.artifacts,
            "latest_draft": state.latest_draft,
            "final_response": state.final_response,
            "step_count": state.step_count,
            "revision_count": state.revision_count,
            "status": state.status,
            "error": state.error,
            "report_path": state.report_path,
            "trace_path": state.trace_path,
            "checkpoint_paths": state.checkpoint_paths,
            "dossier_paths": state.dossier_paths,
        }
    )
    return payload


def _persist_outputs(state: HarnessState) -> None:
    persist_report(state)
    payload = _build_trace_payload(state)
    payload["trace_path"] = state.dossier_paths.get("trace")
    persist_trace(state, payload)


def run_harness(
    request: HarnessRequest,
    *,
    controller_fn: ControllerFn | None = None,
    writer_fn: WriterFn | None = None,
    verifier_fn: VerifierFn | None = None,
    planner_fn: PlannerFn | None = None,
    worker_fn: WorkerFn | None = None,
    registry: dict[str, object] | None = None,
) -> HarnessResponse:
    """Execute the harness end to end and persist the dossier, report, and trace."""

    controller = controller_fn or decide_next_step
    writer = writer_fn or write_draft
    verifier = verifier_fn or verify_draft
    registry_map = registry or build_skill_registry()

    if request.resume_from:
        state = _restore_state_from_checkpoint(request, registry=registry_map)
    else:
        state = initialize_state(request, registry=registry_map)

    if not state.research_plan:
        _run_planner(state, registry_map, planner_fn)

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
            run_root=state.run_root,
            dossier_paths=state.dossier_paths,
        )

    for _ in range(request.max_steps):
        state.step_count += 1
        _refresh_context(state)
        decision = controller(state)
        state.controller_log.append(decision)
        _record_phase(state, "controller", decision.rationale or decision.action)

        if decision.action == "call_skill":
            call = decision.skill_call
            if call is None:
                state.error = "Controller returned call_skill without a skill_call payload."
                continue
            result = execute_skill_call(state, call, registry_map)
            _record_skill_result(state, result)
            _record_phase(state, "worker", f"Executed direct skill call: {call.name}.")
            continue

        if decision.action == "execute_step":
            step = _step_by_id(state, decision.step_id or "")
            if step is None:
                state.error = f"Controller referenced unknown step '{decision.step_id}'."
                continue
            step_result = _run_step_worker(state, step, registry_map, worker_fn)
            _record_step_execution(state, step_result)
            _record_phase(state, "worker", step_result.summary)
            continue

        if decision.action == "execute_parallel_steps":
            step_results = _run_parallel_steps(state, decision.step_ids, registry_map, worker_fn)
            for step_result in step_results:
                _record_step_execution(state, step_result)
            if step_results:
                joined = ", ".join(result.step_id for result in step_results)
                _record_phase(state, "worker", f"Executed parallel steps: {joined}.")
            continue

        if decision.action == "draft":
            _refresh_context(state)
            state.latest_draft = writer(state)
            state.claim_map = build_claim_map(state, state.latest_draft)
            _record_phase(state, "writer", "Generated draft.")

            _refresh_context(state)
            report = _verify_latest_draft(state, verifier)
            _record_phase(state, "evaluator", report.summary)
            if report.decision == "pass":
                _finalize_from_draft(state)
                break
            if report.decision == "revise" and state.revision_count < request.max_revision_rounds:
                state.revision_count += 1
                state.revision_notes.append(report.summary)
                state.revision_notes.extend(report.improvement_instructions)
                continue
            _finalize_from_draft(state)
            break

        if decision.action == "finalize":
            _ensure_final_draft(state, writer)
            if not state.verification_reports or state.verification_reports[-1].decision == "revise":
                _refresh_context(state)
                report = _verify_latest_draft(state, verifier)
                _record_phase(state, "evaluator", report.summary)
                if report.decision == "revise" and state.revision_count < request.max_revision_rounds:
                    state.revision_count += 1
                    state.revision_notes.append(report.summary)
                    state.revision_notes.extend(report.improvement_instructions)
                    continue
            _finalize_from_draft(state)
            break

    else:
        _ensure_final_draft(state, writer)
        if not state.verification_reports:
            _refresh_context(state)
            report = _verify_latest_draft(state, verifier)
            _record_phase(state, "evaluator", report.summary)
        _finalize_from_draft(state)

    _record_phase(state, "finalize", f"Run finished with status: {state.status}.")
    _persist_outputs(state)
    return HarnessResponse(
        final_response=state.final_response or "",
        status=state.status if state.status != "running" else "failed",
        report_path=state.report_path,
        trace_path=state.trace_path,
        verification=state.verification_reports[-1] if state.verification_reports else None,
        enabled_packs=state.enabled_packs,
        skills_used=[result.skill_name for result in state.skill_history],
        artifacts=state.artifacts,
        run_root=state.run_root,
        dossier_paths=state.dossier_paths,
    )
