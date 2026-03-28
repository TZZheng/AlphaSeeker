"""Plan-aware controller for deciding the next harness action."""

from __future__ import annotations

from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage

from src.harness.types import ControllerDecision, HarnessState, ResearchStep
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model


def _format_skills(state: HarnessState) -> str:
    lines = []
    for spec in state.available_skills:
        lines.append(f"- {spec.name} ({spec.pack}): {spec.description}")
    return "\n".join(lines) or "- No skills loaded."


def _format_steps(state: HarnessState) -> str:
    if not state.research_plan or not state.research_plan.steps:
        return "No research plan loaded."
    lines = []
    for step in state.research_plan.steps:
        status = state.step_statuses.get(step.id, "pending")
        lines.append(
            f"- {step.id} [{status}] depends_on={step.depends_on} "
            f"parallel_safe={step.parallel_safe} objective={step.objective}"
        )
    return "\n".join(lines)


def _format_follow_up_calls(state: HarnessState) -> str:
    if state.pending_follow_up_calls:
        return "\n".join(
            f"- {call.name} {call.arguments}" for call in state.pending_follow_up_calls
        )
    if not state.verification_reports:
        return "None"
    last_report = state.verification_reports[-1]
    if not last_report.required_follow_up_calls:
        return "None"
    return "\n".join(
        f"- {call.name} {call.arguments}" for call in last_report.required_follow_up_calls
    )


def _format_coverage_matrix(state: HarnessState) -> str:
    if not state.coverage_matrix:
        return "None"
    lines = []
    for entry in [
        *state.coverage_matrix.sections,
        *state.coverage_matrix.evidence_types,
        *state.coverage_matrix.counterevidence_requirements,
    ][:18]:
        lines.append(f"- {entry.coverage_type}:{entry.label} [{entry.status}] count={entry.evidence_count}")
    stats = state.coverage_matrix.stats
    if stats:
        lines.append(
            f"- stats discovered={stats.get('discovered_count', 0)} full_reads={stats.get('full_read_count', 0)} "
            f"qa_iterations={stats.get('qa_iteration_count', 0)}"
        )
    return "\n".join(lines) or "None"


def get_unblocked_steps(state: HarnessState) -> list[ResearchStep]:
    """Return pending steps whose dependencies are already satisfied."""

    if not state.research_plan:
        return []

    ready: list[ResearchStep] = []
    for step in state.research_plan.steps:
        status = state.step_statuses.get(step.id, "pending")
        if status in {"completed", "partial", "skipped"}:
            continue
        if status == "failed":
            continue
        dependencies_met = all(
            state.step_statuses.get(dep) in {"completed", "partial", "skipped"}
            for dep in step.depends_on
        )
        if dependencies_met:
            ready.append(step)
    return ready


def _fallback_decision(state: HarnessState) -> ControllerDecision:
    if state.latest_draft and state.verification_reports:
        last_report = state.verification_reports[-1]
        if last_report.decision == "pass":
            return ControllerDecision(
                action="finalize",
                rationale="The latest evaluated draft already passed.",
            )

    if state.verification_reports:
        if state.pending_follow_up_calls:
            call = state.pending_follow_up_calls[0]
            return ControllerDecision(
                action="call_skill",
                rationale="Evaluator requested a concrete follow-up call.",
                skill_call=call,
            )
        last_report = state.verification_reports[-1]
        if (
            last_report.decision == "revise"
            and state.revision_count < state.request.max_revision_rounds
            and state.latest_draft
        ):
            return ControllerDecision(
                action="draft",
                rationale="Revise the draft against evaluator section feedback.",
            )

    ready_steps = get_unblocked_steps(state)
    if ready_steps:
        parallel_ready = [
            step.id for step in ready_steps if step.parallel_safe
        ][: state.request.max_parallel_steps]
        if (
            state.request.allow_parallel_steps
            and len(parallel_ready) > 1
        ):
            return ControllerDecision(
                action="execute_parallel_steps",
                rationale="Multiple independent steps are ready and parallel-safe.",
                step_ids=parallel_ready,
            )
        return ControllerDecision(
            action="execute_step",
            rationale=f"Run the next unblocked step: {ready_steps[0].id}.",
            step_id=ready_steps[0].id,
        )

    if (
        state.request.research_profile == "deep"
        and state.coverage_matrix is not None
        and state.coverage_matrix.needs_more_retrieval
    ):
        return ControllerDecision(
            action="call_skill",
            rationale="Coverage matrix still shows weak sections or evidence types.",
            skill_call={
                "name": "deep_retrieval",
                "arguments": {
                    "stage": "run_wave",
                    "prompt": state.request.user_prompt,
                    "ingest_batch_size": state.request.deep_read_batch_size,
                },
            },
        )

    if not state.latest_draft:
        return ControllerDecision(
            action="draft",
            rationale="All planned steps are complete enough for a draft.",
        )

    return ControllerDecision(
        action="finalize",
        rationale="No remaining unblocked steps and a draft already exists.",
    )


def decide_next_step(state: HarnessState) -> ControllerDecision:
    """Pick the next action for the harness, with a scheduler-oriented fallback."""

    system_prompt = """
You are the controller for a research harness.
You are scheduling work, not inventing a new plan.

You may choose exactly one next action:
1. call_skill
2. execute_step
3. execute_parallel_steps
4. draft
5. finalize

Rules:
- Prefer executing unblocked planned steps before drafting.
- Use required_follow_up_calls when evaluator feedback provides them.
- Only schedule steps whose dependencies are already satisfied.
- Use execute_parallel_steps only for independent, parallel-safe steps.
- Draft only when the remaining gaps are best solved by writing or revising.
"""

    user_prompt = f"""Mission:
{state.mission_text or state.request.user_prompt}

Progress:
{state.progress_text or "No progress file loaded."}

Enabled packs:
{", ".join(state.enabled_packs) or "None"}

Available skills:
{_format_skills(state)}

Research plan:
{_format_steps(state)}

Evaluator follow-up calls:
{_format_follow_up_calls(state)}

Coverage matrix:
{_format_coverage_matrix(state)}

Revision notes:
{chr(10).join(state.revision_notes[-8:]) or "None"}
"""

    try:
        model_name = get_model("harness", "controller")
        llm = get_llm(model_name).with_structured_output(ControllerDecision, method="json_mode")
        decision = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return cast(ControllerDecision, decision)
    except Exception as exc:
        print(f"Harness controller fallback triggered: {exc}")
        return _fallback_decision(state)
