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
        if len(content) > 400:
            content = content[:400] + "..."
        lines.append(f"- [{item.id}] {item.summary} | {source_text}\n{content}")
    return "\n".join(lines)


def _fallback_decision(state: HarnessState) -> ControllerDecision:
    skill_names = {spec.name for spec in state.available_skills}

    if state.latest_draft and state.verification_reports:
        last_report = state.verification_reports[-1]
        if last_report.decision == "pass":
            return ControllerDecision(
                action="finalize",
                rationale="The latest verified draft already passed the judge.",
            )

    if not state.evidence_ledger:
        if "search_and_read" in skill_names:
            return ControllerDecision(
                action="call_skill",
                rationale="Start by collecting full-text evidence for the user prompt.",
                skill_call={
                    "name": "search_and_read",
                    "arguments": {"queries": [state.request.user_prompt], "urls_per_query": 2},
                },
            )
        if "search_web" in skill_names:
            return ControllerDecision(
                action="call_skill",
                rationale="Start by collecting discovery results for the user prompt.",
                skill_call={"name": "search_web", "arguments": {"query": state.request.user_prompt}},
            )

    if state.verification_reports and state.verification_reports[-1].decision == "revise":
        if state.revision_count < state.request.max_revision_rounds and "search_and_read" in skill_names:
            critique = " ".join(state.verification_reports[-1].improvement_instructions[:2]).strip()
            query = state.request.user_prompt
            if critique:
                query = f"{query} {critique}"
            return ControllerDecision(
                action="call_skill",
                rationale="Gather another round of evidence to address the verifier critique.",
                skill_call={
                    "name": "search_and_read",
                    "arguments": {"queries": [query], "urls_per_query": 2},
                },
            )
        return ControllerDecision(
            action="draft",
            rationale="Revise the draft using the collected evidence and verifier feedback.",
        )

    if not state.latest_draft:
        return ControllerDecision(
            action="draft",
            rationale="Sufficient evidence exists to attempt a draft.",
        )

    return ControllerDecision(
        action="finalize",
        rationale="A draft already exists and no better next skill is obvious.",
    )


def decide_next_step(state: HarnessState) -> ControllerDecision:
    """Pick the next action for the harness, with a heuristic fallback."""

    system_prompt = """
You are the controller for a research harness.
You may choose exactly one next action:
1. call_skill
2. draft
3. finalize

Rules:
- Prefer gathering evidence before drafting.
- Use only one skill call at a time.
- Do not finalize an unverified draft unless the revision budget is exhausted.
- When calling a skill, populate both skill name and arguments.
- Keep arguments minimal and concrete.
"""

    user_prompt = f"""User request:
{state.request.user_prompt}

Enabled packs:
{", ".join(state.enabled_packs)}

Available skills:
{_format_skills(state)}

Working memory:
{chr(10).join(state.working_memory[-6:]) or "No memory yet."}

Revision notes:
{chr(10).join(state.revision_notes[-6:]) or "No revision notes yet."}

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
