"""Evaluator for harness drafts."""

from __future__ import annotations

from typing import Callable, cast

from langchain_core.messages import HumanMessage, SystemMessage

from src.harness.types import HarnessState, ReportSectionFeedback, SkillCall, VerificationReport
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


def _missing_sections(state: HarnessState, draft: str) -> list[str]:
    if not state.research_plan:
        return []
    lowered = draft.lower()
    missing: list[str] = []
    for section in state.research_plan.required_sections:
        if section.lower() not in lowered:
            missing.append(section)
    return missing


def _missing_counterevidence_topics(state: HarnessState, draft: str) -> list[str]:
    if not state.research_plan:
        return []
    lowered = draft.lower()
    gaps = []
    for topic in state.research_plan.counterevidence_topics:
        tokens = topic.lower().split()
        if not any(token in lowered for token in tokens):
            gaps.append(topic)
    return gaps


def _draft_has_citations(draft: str) -> bool:
    return "[E" in draft


def _collect_missing_citations(state: HarnessState, draft: str) -> list[str]:
    if _draft_has_citations(draft):
        return []
    return [item.id for item in state.evidence_ledger if item.id]


def _numeric_inconsistencies(state: HarnessState, draft: str) -> list[str]:
    issues: list[str] = []
    if any(char.isdigit() for char in draft) and not _draft_has_citations(draft):
        issues.append("Numeric claims appear without evidence citations.")
    if state.claim_map:
        for claim in state.claim_map:
            if any(char.isdigit() for char in claim.text) and not claim.supporting_evidence_ids:
                issues.append(f"Claim {claim.id} contains numeric content without supporting evidence IDs.")
    return issues


def _freshness_warnings(state: HarnessState, draft: str) -> list[str]:
    warnings: list[str] = []
    if not state.research_plan:
        return warnings
    if state.research_plan.freshness_requirements and not any(char.isdigit() for char in draft):
        warnings.append("Freshness-sensitive prompt lacks explicit dates in the draft.")
    return warnings


def _section_feedback(state: HarnessState, draft: str) -> list[ReportSectionFeedback]:
    feedback: list[ReportSectionFeedback] = []
    for section in _missing_sections(state, draft):
        feedback.append(
            ReportSectionFeedback(
                section_label=section,
                issue="Missing required section",
                why_it_fails=f"The research plan requires the section '{section}', but it is absent from the draft.",
                suggested_fix=f"Add a section titled '{section}' and support it with evidence citations.",
            )
        )
    if not _draft_has_citations(draft):
        feedback.append(
            ReportSectionFeedback(
                section_label="Global",
                issue="Missing citations",
                why_it_fails="The draft does not cite evidence IDs.",
                suggested_fix="Add inline evidence citations such as [E1] to each major claim.",
                missing_evidence_ids=[item.id for item in state.evidence_ledger if item.id],
            )
        )
    counter_gaps = _missing_counterevidence_topics(state, draft)
    if counter_gaps:
        feedback.append(
            ReportSectionFeedback(
                section_label="Risks / Counter-Evidence",
                issue="Counter-evidence coverage is incomplete",
                why_it_fails=f"The draft does not address required counter-evidence topics: {', '.join(counter_gaps)}.",
                suggested_fix="Add a risk or counter-evidence section that addresses the missing topics with citations.",
            )
        )
    return feedback


def _follow_up_calls(state: HarnessState, draft: str) -> list[SkillCall]:
    calls: list[SkillCall] = []
    skill_names = {spec.name for spec in state.available_skills}
    if not state.evidence_ledger and "search_and_read" in skill_names:
        calls.append(
            SkillCall(
                name="search_and_read",
                arguments={"queries": [state.request.user_prompt], "urls_per_query": 2},
            )
        )
    if "equity" in state.enabled_packs and "fetch_financials" in skill_names:
        if not any(result.skill_name == "fetch_financials" for result in state.skill_history):
            calls.append(SkillCall(name="fetch_financials", arguments=_infer_ticker_args(state)))
    if _missing_counterevidence_topics(state, draft) and "search_and_read" in skill_names:
        calls.append(
            SkillCall(
                name="search_and_read",
                arguments={
                    "queries": [f"{state.request.user_prompt} risks competition downside case"],
                    "urls_per_query": 2,
                    "use_news": True,
                },
            )
        )
    deduped: list[SkillCall] = []
    seen: set[tuple[str, str]] = set()
    for call in calls:
        key = (call.name, str(sorted(call.arguments.items())))
        if key not in seen:
            deduped.append(call)
            seen.add(key)
    return deduped


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
            blocking_issues=["Draft is empty."],
            raw_feedback="Empty draft.",
        )

    missing_sections = _missing_sections(state, draft)
    missing_citations = _collect_missing_citations(state, draft)
    numeric_issues = _numeric_inconsistencies(state, draft)
    freshness_warnings = _freshness_warnings(state, draft)
    counter_gaps = _missing_counterevidence_topics(state, draft)
    section_feedback = _section_feedback(state, draft)

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
            missing_evidence=["No evidence ledger entries were available."],
            missing_citations=missing_citations,
            required_follow_up_calls=_follow_up_calls(state, draft),
            report_section_feedback=section_feedback,
            raw_feedback="No evidence ledger entries were available.",
        )

    decision = "pass"
    grounding = "pass"
    completeness = "pass"
    numeric_consistency = "pass"
    citation_coverage = "pass"
    if missing_citations:
        decision = "revise"
        grounding = "revise"
        citation_coverage = "fail"
    if missing_sections or counter_gaps:
        decision = "revise"
        completeness = "revise"
    if numeric_issues:
        decision = "revise"
        numeric_consistency = "revise"
    if freshness_warnings:
        decision = "revise"
        grounding = "revise"

    return VerificationReport(
        decision=cast(str, decision),  # type: ignore[arg-type]
        summary="Draft evaluation completed.",
        grounding=cast(str, grounding),  # type: ignore[arg-type]
        completeness=cast(str, completeness),  # type: ignore[arg-type]
        numeric_consistency=cast(str, numeric_consistency),  # type: ignore[arg-type]
        citation_coverage=cast(str, citation_coverage),  # type: ignore[arg-type]
        formatting="pass",
        improvement_instructions=[
            *([f"Add section: {section}." for section in missing_sections]),
            *["Add inline evidence citations to major claims." for _ in missing_citations[:1]],
            *numeric_issues[:2],
            *[f"Address counter-evidence topic: {topic}." for topic in counter_gaps[:2]],
        ],
        missing_evidence=[],
        blocking_issues=[],
        missing_sections=missing_sections,
        missing_citations=missing_citations,
        freshness_warnings=freshness_warnings,
        numeric_inconsistencies=numeric_issues,
        required_follow_up_calls=_follow_up_calls(state, draft),
        counterevidence_gaps=counter_gaps,
        report_section_feedback=section_feedback,
        raw_feedback="Fallback evaluator used structured heuristic checks.",
    )


def verify_draft(
    state: HarnessState,
    draft: str,
    judge_fn: Callable[[HarnessState, str], VerificationReport] | None = None,
) -> VerificationReport:
    """Run the evaluator, with optional test injection and a rule-based fallback."""

    if judge_fn is not None:
        return judge_fn(state, draft)

    system_prompt = """
You are the evaluator for a financial research harness draft.

Score these categories with only: pass, revise, or fail
- grounding
- completeness
- numeric_consistency
- citation_coverage
- formatting

Rules:
- If a material claim lacks support from the provided evidence list, do not pass grounding.
- If the draft does not cite evidence IDs, do not pass citation_coverage.
- Inspect required sections, contract clauses, and counter-evidence coverage.
- Return targeted section feedback and executable follow-up calls when revision is needed.
- Keep summary short and factual.
"""

    contract_json = (
        state.research_contract.model_dump_json(indent=2)
        if state.research_contract is not None
        else "null"
    )
    claim_map_json = jsonable_claim_map(state)
    user_prompt = f"""User request:
{state.request.user_prompt}

Research contract:
{contract_json}

Claim map:
{claim_map_json}

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


def jsonable_claim_map(state: HarnessState) -> str:
    if not state.claim_map:
        return "[]"
    return "[\n" + ",\n".join(claim.model_dump_json() for claim in state.claim_map[:20]) + "\n]"


def _infer_ticker_args(state: HarnessState) -> dict[str, str]:
    prompt = state.request.user_prompt.strip()
    tokens = [token.strip(",.()").upper() for token in prompt.split()]
    ticker = next((token for token in tokens if 1 <= len(token) <= 5 and token.isalpha() and token == token.upper()), "")
    return {"ticker": ticker} if ticker else {}
