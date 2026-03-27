"""Step worker subagent for harness research steps."""

from __future__ import annotations

from src.harness.types import HarnessState, ResearchStep, SkillResult, SkillSpec, StepExecutionResult


def execute_research_step(
    step: ResearchStep,
    state: HarnessState,
    registry_map: dict[str, SkillSpec],
) -> tuple[list[SkillResult], StepExecutionResult]:
    """Execute one planned research step with a bounded inner loop."""

    skill_results: list[SkillResult] = []
    findings: list[str] = []
    open_questions: list[str] = []
    artifact_paths: list[str] = []
    evidence_ids: list[str] = []

    for call in step.recommended_skill_calls[: state.request.max_step_worker_iterations]:
        spec = registry_map.get(call.name)
        if spec is None or spec.executor is None:
            skill_results.append(
                SkillResult(
                    skill_name=call.name,
                    arguments=call.arguments,
                    status="failed",
                    summary=f"Unknown skill '{call.name}' requested by step '{step.id}'.",
                    error=f"Unknown skill '{call.name}'.",
                )
            )
            open_questions.append(f"Skill '{call.name}' is unavailable for step '{step.id}'.")
            continue
        try:
            result = spec.executor(call.arguments, state)
        except Exception as exc:
            result = SkillResult(
                skill_name=call.name,
                arguments=call.arguments,
                status="failed",
                summary=f"Skill '{call.name}' raised an exception while executing step '{step.id}'.",
                error=f"{type(exc).__name__}: {exc}",
            )
        skill_results.append(result)
        findings.append(result.summary)
        artifact_paths.extend(result.artifacts)
        evidence_ids.extend(item.id for item in result.evidence if item.id)
        if result.status == "failed":
            open_questions.append(result.error or f"Skill '{call.name}' failed.")

    statuses = {result.status for result in skill_results}
    if not skill_results:
        step_status = "blocked"
        summary = f"Step '{step.id}' had no executable skill calls."
        open_questions.append("Planner produced no executable skill calls.")
    elif statuses == {"ok"} or statuses == {"ok", "partial"} and any(
        result.evidence or result.artifacts for result in skill_results
    ):
        step_status = "completed"
        summary = f"Completed step '{step.id}' with {len(skill_results)} skill call(s)."
    elif "failed" in statuses and any(result.evidence or result.artifacts for result in skill_results):
        step_status = "partial"
        summary = f"Step '{step.id}' produced partial results with {len(skill_results)} skill call(s)."
    elif "failed" in statuses:
        step_status = "failed"
        summary = f"Step '{step.id}' failed."
    else:
        step_status = "blocked"
        summary = f"Step '{step.id}' is blocked."

    return skill_results, StepExecutionResult(
        step_id=step.id,
        status=step_status,
        summary=summary,
        evidence_ids=list(dict.fromkeys(evidence_ids)),
        artifact_paths=list(dict.fromkeys(artifact_paths)),
        findings=findings,
        open_questions=open_questions,
        suggested_next_steps=step.depends_on,
    )
