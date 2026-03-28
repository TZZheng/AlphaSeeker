"""Step-worker execution for dependency-aware research steps."""

from __future__ import annotations

from typing import Callable

from src.harness.types import (
    HarnessState,
    ResearchStep,
    SkillCall,
    SkillResult,
    StepExecutionResult,
)


SkillRecorder = Callable[[HarnessState, SkillResult], None]


def execute_skill_call(
    state: HarnessState,
    call: SkillCall,
    registry_map: dict[str, object],
) -> SkillResult:
    """Execute one normalized skill call against the registry."""

    spec = registry_map.get(call.name)
    if spec is None or getattr(spec, "executor", None) is None:
        return SkillResult(
            skill_name=call.name,
            arguments=call.arguments,
            status="failed",
            summary=f"Unknown skill '{call.name}'.",
            error=f"Unknown skill '{call.name}'.",
        )

    try:
        return spec.executor(call.arguments, state)
    except Exception as exc:
        return SkillResult(
            skill_name=call.name,
            arguments=call.arguments,
            status="failed",
            summary=f"Skill '{call.name}' raised an exception.",
            error=f"{type(exc).__name__}: {exc}",
        )


def execute_research_step(
    state: HarnessState,
    step: ResearchStep,
    registry_map: dict[str, object],
    *,
    record_skill_result: SkillRecorder,
    max_iterations: int = 3,
) -> StepExecutionResult:
    """Execute a structured research step with a bounded inner loop."""

    executed: list[SkillCall] = []
    evidence_ids: list[str] = []
    artifact_paths: list[str] = []
    findings: list[str] = []
    open_questions: list[str] = []
    failures = 0

    if not step.recommended_skill_calls:
        return StepExecutionResult(
            step_id=step.id,
            status="blocked",
            summary=f"Step '{step.id}' has no skill calls to execute.",
            open_questions=["Planner must provide at least one skill call."],
            suggested_next_steps=["Replan this step."],
            iteration_count=0,
        )

    iteration_count = 0
    for iteration_count in range(1, max_iterations + 1):
        progress_made = False
        for call in step.recommended_skill_calls:
            if any(existing.name == call.name and existing.arguments == call.arguments for existing in executed):
                continue
            result = execute_skill_call(state, call, registry_map)
            record_skill_result(state, result)
            executed.append(call)
            progress_made = True
            evidence_ids.extend(item.id for item in result.evidence if item.id)
            artifact_paths.extend(path for path in result.artifacts if path)
            findings.append(result.summary)
            if result.status == "failed":
                failures += 1
                if result.error:
                    open_questions.append(result.error)
            elif result.status == "partial" and result.error:
                open_questions.append(result.error)

        if len(executed) == len(step.recommended_skill_calls) or not progress_made:
            break

    unique_evidence = list(dict.fromkeys(evidence_ids))
    unique_artifacts = list(dict.fromkeys(artifact_paths))
    findings = list(dict.fromkeys(findings))
    open_questions = list(dict.fromkeys(open_questions))

    if len(executed) == 0:
        status = "blocked"
        summary = f"Step '{step.id}' could not start."
    elif failures == 0:
        status = "completed"
        summary = f"Completed step '{step.id}' with {len(executed)} skill call(s)."
    elif failures < len(executed):
        status = "partial"
        summary = f"Step '{step.id}' completed partially after {len(executed)} skill call(s)."
    else:
        status = "failed"
        summary = f"Step '{step.id}' failed across all attempted skill call(s)."

    suggested_next_steps = []
    if status in {"partial", "failed"}:
        suggested_next_steps.append("Review missing evidence and rerun the step if needed.")

    return StepExecutionResult(
        step_id=step.id,
        status=status,
        summary=summary,
        evidence_ids=unique_evidence,
        artifact_paths=unique_artifacts,
        findings=findings,
        open_questions=open_questions,
        suggested_next_steps=suggested_next_steps,
        executed_skill_calls=executed,
        iteration_count=iteration_count,
    )
