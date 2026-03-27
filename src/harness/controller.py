"""LLM controller for deciding the next harness action."""

from __future__ import annotations

from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage

from src.harness.types import ControllerDecision, HarnessState
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model


def _format_skills(state: HarnessState) -> str:
    lines = []
    for spec in state.available_skills:
        lines.append(
            f"- {spec.name} ({spec.pack}): {spec.description} | input_schema={spec.input_schema}"
        )
    return "\n".join(lines)


def _format_evidence(state: HarnessState, limit: int = 10) -> str:
    if not state.evidence_ledger:
        return "No evidence collected yet."

    lines = []
    for item in state.evidence_ledger[-limit:]:
        source_text = ", ".join(item.sources or item.artifact_paths) or item.source_type
        content = (item.content or "").strip()
        if len(content) > 320:
            content = content[:320] + "..."
        lines.append(f"- [{item.id}] {item.summary} | {source_text}\n{content}")
    return "\n".join(lines)


def _step_status_lines(state: HarnessState) -> list[str]:
    if not state.research_plan:
        return ["- No research plan yet."]
    completed = set(state.completed_step_ids)
    failed = set(state.failed_step_ids)
    lines = []
    for step in state.research_plan.steps:
        if step.id in completed:
            status = "completed"
        elif step.id in failed:
            status = "failed"
        else:
            status = "pending"
        lines.append(f"- {step.id}: {status} | depends_on={step.depends_on} | {step.objective}")
    return lines


def _unblocked_step_ids(state: HarnessState) -> list[str]:
    if not state.research_plan:
        return []
    completed = set(state.completed_step_ids)
    failed = set(state.failed_step_ids)
    ready: list[str] = []
    for step in state.research_plan.steps:
        if step.id in completed or step.id in failed:
            continue
        if all(dep in completed for dep in step.depends_on):
            ready.append(step.id)
    return ready


def _parallelizable_unblocked_step_ids(state: HarnessState) -> list[str]:
    if not state.research_plan:
        return []
    step_map = {step.id: step for step in state.research_plan.steps}
    return [
        step_id
        for step_id in _unblocked_step_ids(state)
        if step_map.get(step_id) is not None and step_map[step_id].can_run_parallel
    ]


def _fallback_decision(state: HarnessState) -> ControllerDecision:
    if state.pending_follow_up_calls:
        call = state.pending_follow_up_calls.pop(0)
        return ControllerDecision(
            action="call_skill",
            rationale="Run evaluator-requested follow-up skill call.",
            skill_call=call,
        )

    if state.latest_draft and state.verification_reports:
        last_report = state.verification_reports[-1]
        if last_report.decision == "pass":
            return ControllerDecision(
                action="finalize",
                rationale="The latest verified draft already passed the evaluator.",
            )

    parallel_ids = _parallelizable_unblocked_step_ids(state)
    if len(parallel_ids) > 1 and state.request.enable_parallel_steps:
        return ControllerDecision(
            action="run_steps",
            rationale="Multiple independent planned steps are ready and can run in parallel.",
            step_ids=parallel_ids,
        )

    ready_steps = _unblocked_step_ids(state)
    if ready_steps:
        return ControllerDecision(
            action="run_step",
            rationale="Run the next unblocked planned research step.",
            step_id=ready_steps[0],
        )

    if not state.latest_draft:
        return ControllerDecision(
            action="draft",
            rationale="No planned steps remain unblocked, so draft the report from the collected evidence.",
        )

    if state.verification_reports and state.verification_reports[-1].decision == "revise":
        return ControllerDecision(
            action="draft",
            rationale="Revise the draft using evaluator feedback and the collected evidence.",
        )

    return ControllerDecision(
        action="finalize",
        rationale="No more planned work is ready, and a draft already exists.",
    )


def decide_next_step(state: HarnessState) -> ControllerDecision:
    """Pick the next action for the harness, with a heuristic fallback."""

    system_prompt = """
You are the controller for a financial research harness.
Choose exactly one next action:
1. call_skill
2. run_step
3. run_steps
4. draft
5. finalize

Rules:
- Prefer planned steps over ad hoc tool calls.
- Use call_skill only for evaluator-requested follow-up calls or exceptional gaps.
- Use run_steps only when multiple independent steps are ready.
- Do not finalize an unverified draft unless revision budget is exhausted.
- Keep the response minimal and structured.
"""

    user_prompt = f"""User request:
{state.request.user_prompt}

Enabled packs:
{", ".join(state.enabled_packs)}

Available skills:
{_format_skills(state)}

Plan status:
{chr(10).join(_step_status_lines(state))}

Pending follow-up calls:
{[call.model_dump(mode="json") for call in state.pending_follow_up_calls] or "None"}

Recent progress:
{chr(10).join(state.progress_updates[-10:]) or "No progress yet."}

Recent evidence:
{_format_evidence(state)}
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
