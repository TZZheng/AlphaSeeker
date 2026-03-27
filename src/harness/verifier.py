"""LLM-as-a-judge verifier for harness drafts."""

from __future__ import annotations

from typing import Callable, cast

from langchain_core.messages import HumanMessage, SystemMessage

from src.harness.types import HarnessState, VerificationReport
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model


def _format_evidence(state: HarnessState, limit: int = 12) -> str:
    if not state.evidence_ledger:
        return "No evidence collected."

    lines = []
    for item in state.evidence_ledger[-limit:]:
        source_text = ", ".join(item.sources or item.artifact_paths) or item.source_type
        lines.append(f"[{item.id}] {item.summary} | {source_text}")
    return "\n".join(lines)


def _fallback_verification(state: HarnessState, draft: str) -> VerificationReport:
    if not draft.strip():
        return VerificationReport(
            decision="fail",
            summary="The draft is empty.",
            grounding="fail",
            completeness="fail",
            numeric_consistency="fail",
            citation_coverage="fail",
            formatting="fail",
            improvement_instructions=["Generate a non-empty draft before finalizing."],
            raw_feedback="Empty draft.",
        )

    if not state.evidence_ledger:
        return VerificationReport(
            decision="revise",
            summary="The draft has no collected evidence behind it.",
            grounding="fail",
            completeness="revise",
            numeric_consistency="revise",
            citation_coverage="fail",
            formatting="pass",
            improvement_instructions=[
                "Collect at least one grounded source before finalizing.",
                "Add evidence citations to the answer.",
            ],
            raw_feedback="No evidence ledger entries were available.",
        )

    cited_ids = [item.id for item in state.evidence_ledger if item.id and f"[{item.id}]" in draft]
    if not cited_ids:
        return VerificationReport(
            decision="revise",
            summary="The draft does not cite any collected evidence.",
            grounding="revise",
            completeness="pass",
            numeric_consistency="pass",
            citation_coverage="fail",
            formatting="pass",
            improvement_instructions=[
                "Add inline evidence citations such as [E1].",
                "Tie each major claim to at least one evidence item.",
            ],
            raw_feedback="Evidence existed, but no evidence IDs appeared in the draft.",
        )

    return VerificationReport(
        decision="pass",
        summary="The draft is grounded and cites collected evidence.",
        grounding="pass",
        completeness="pass",
        numeric_consistency="pass",
        citation_coverage="pass",
        formatting="pass",
        raw_feedback="Fallback verifier found evidence citations in the draft.",
    )


def verify_draft(
    state: HarnessState,
    draft: str,
    judge_fn: Callable[[HarnessState, str], VerificationReport] | None = None,
) -> VerificationReport:
    """Run the verifier, with optional test injection and a rule-based fallback."""

    if judge_fn is not None:
        return judge_fn(state, draft)

    system_prompt = """
You are the verifier for a research harness draft.

Score these categories with only: pass, revise, or fail
- grounding
- completeness
- numeric_consistency
- citation_coverage
- formatting

Rules:
- If a material claim lacks support from the provided evidence list, do not pass grounding.
- If the draft does not cite evidence IDs, do not pass citation_coverage.
- Provide concrete improvement_instructions when decision is revise or fail.
- Keep summary short and factual.
"""

    user_prompt = f"""User request:
{state.request.user_prompt}

Evidence ledger:
{_format_evidence(state)}

Draft to judge:
{draft}
"""

    try:
        model_name = get_model("harness", "verify")
        llm = get_llm(model_name).with_structured_output(VerificationReport, method="json_mode")
        report = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return cast(VerificationReport, report)
    except Exception as exc:
        print(f"Harness verifier fallback triggered: {exc}")
        return _fallback_verification(state, draft)
