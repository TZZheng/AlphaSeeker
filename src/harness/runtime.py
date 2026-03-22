"""Bounded control-loop runtime for the side-by-side harness."""

from __future__ import annotations

import json
import os
from typing import Callable

from src.harness.controller import decide_next_step
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.selector import select_packs
from src.harness.types import (
    ControllerDecision,
    HarnessRequest,
    HarnessResponse,
    HarnessState,
    SkillResult,
    VerificationReport,
)
from src.harness.writer import write_draft
from src.harness.verifier import verify_draft
from src.shared.report_filename import build_prompt_report_filename
from src.shared.text_utils import condense_context


ControllerFn = Callable[[HarnessState], ControllerDecision]
WriterFn = Callable[[HarnessState], str]
VerifierFn = Callable[[HarnessState, str], VerificationReport]
SelectorFn = Callable[[str], list[str]]


def initialize_state(
    request: HarnessRequest,
    *,
    selector_fn: SelectorFn | None = None,
    registry: dict[str, object] | None = None,
) -> HarnessState:
    """Construct the initial harness state with selected packs and skills."""

    registry_map = registry or build_skill_registry()
    selected = request.selected_packs or (selector_fn or select_packs)(request.user_prompt)
    enabled_packs = []
    seen = set()
    for pack in ["core", *selected]:
        if pack not in seen:
            enabled_packs.append(pack)
            seen.add(pack)

    available_skills = get_skills_for_packs(registry_map, enabled_packs)
    return HarnessState(
        request=request,
        enabled_packs=enabled_packs,
        available_skills=available_skills,
    )


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
        state.latest_draft = writer_fn(state)


def _build_trace_payload(state: HarnessState) -> dict[str, object]:
    return {
        "request": state.request.model_dump(mode="json"),
        "enabled_packs": state.enabled_packs,
        "available_skills": [spec.model_dump(mode="json") for spec in state.available_skills],
        "controller_log": [decision.model_dump(mode="json") for decision in state.controller_log],
        "skill_history": [result.model_dump(mode="json") for result in state.skill_history],
        "evidence_ledger": [item.model_dump(mode="json") for item in state.evidence_ledger],
        "verification_reports": [report.model_dump(mode="json") for report in state.verification_reports],
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
    }


def _persist_outputs(state: HarnessState) -> None:
    os.makedirs(os.path.join(os.getcwd(), "reports"), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), "data", "harness_runs"), exist_ok=True)

    report_name = build_prompt_report_filename(
        prompt_text=state.request.user_prompt,
        fallback_stem="harness_report",
        suffix="harness",
    )
    report_path = os.path.join(os.getcwd(), "reports", report_name)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(state.final_response or "")
    state.report_path = report_path

    trace_name = report_name.replace(".md", ".json")
    trace_path = os.path.join(os.getcwd(), "data", "harness_runs", trace_name)
    payload = _build_trace_payload(state)
    payload["trace_path"] = trace_path
    with open(trace_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True)
    state.trace_path = trace_path


def run_harness(
    request: HarnessRequest,
    *,
    controller_fn: ControllerFn | None = None,
    writer_fn: WriterFn | None = None,
    verifier_fn: VerifierFn | None = None,
    selector_fn: SelectorFn | None = None,
    registry: dict[str, object] | None = None,
) -> HarnessResponse:
    """Execute the harness end to end and persist the report plus trace."""

    controller = controller_fn or decide_next_step
    writer = writer_fn or write_draft
    verifier = verifier_fn or verify_draft
    state = initialize_state(request, selector_fn=selector_fn, registry=registry)

    registry_map = registry or build_skill_registry()
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

    for _ in range(request.max_steps):
        state.step_count += 1
        decision = controller(state)
        state.controller_log.append(decision)

        if decision.action == "call_skill":
            call = decision.skill_call
            if call is None:
                state.error = "Controller returned call_skill without a skill_call payload."
                continue
            spec = registry_map.get(call.name)
            if spec is None or getattr(spec, "executor", None) is None:
                failed = SkillResult(
                    skill_name=call.name,
                    arguments=call.arguments,
                    status="failed",
                    summary=f"Unknown skill '{call.name}'.",
                    error=f"Unknown skill '{call.name}'.",
                )
                _record_skill_result(state, failed)
                continue
            try:
                result = spec.executor(call.arguments, state)
            except Exception as exc:
                result = SkillResult(
                    skill_name=call.name,
                    arguments=call.arguments,
                    status="failed",
                    summary=f"Skill '{call.name}' raised an exception.",
                    error=f"{type(exc).__name__}: {exc}",
                )
            _record_skill_result(state, result)
            continue

        if decision.action == "draft":
            state.latest_draft = writer(state)
            report = _verify_latest_draft(state, verifier)
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
                report = _verify_latest_draft(state, verifier)
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
            _verify_latest_draft(state, verifier)
        _finalize_from_draft(state)

    if state.status == "running":
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
