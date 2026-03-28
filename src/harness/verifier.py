"""Contract-aware evaluator for harness drafts."""

from __future__ import annotations

import re
from typing import Callable, cast

from langchain_core.messages import HumanMessage, SystemMessage

from src.harness.controller import get_unblocked_steps
from src.harness.types import HarnessState, ReportSectionFeedback, SkillCall, VerificationReport
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model


def _parse_sections(draft: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current = "__preamble__"
    buffer: list[str] = []
    for line in draft.splitlines():
        if line.startswith("## "):
            sections[current] = "\n".join(buffer).strip()
            current = line[3:].strip()
            buffer = []
            continue
        buffer.append(line)
    sections[current] = "\n".join(buffer).strip()
    return sections


def _format_evidence(state: HarnessState, limit: int = 12) -> str:
    if not state.evidence_ledger:
        return "No evidence collected."

    lines = []
    for item in state.evidence_ledger[-limit:]:
        source_text = ", ".join(item.sources or item.artifact_paths) or item.source_type
        lines.append(f"[{item.id}] {item.summary} | {source_text}")
    return "\n".join(lines)


def _required_sections(state: HarnessState) -> list[str]:
    if state.research_plan and state.research_plan.required_sections:
        return state.research_plan.required_sections
    return []


def _rule_based_evaluation(state: HarnessState, draft: str) -> VerificationReport:
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
            blocking_issues=["The draft is empty."],
            raw_feedback="Empty draft.",
        )

    sections = _parse_sections(draft)
    required_sections = _required_sections(state)
    missing_sections = [section for section in required_sections if section not in sections]

    report_section_feedback: list[ReportSectionFeedback] = []
    missing_citations: list[str] = []
    for section in required_sections:
        if section not in sections:
            report_section_feedback.append(
                ReportSectionFeedback(
                    section_label=section,
                    issue="Missing required section",
                    why_it_fails="The contract requires this section but it does not appear in the draft.",
                    suggested_fix=f"Add a '{section}' section grounded in collected evidence.",
                )
            )
            continue
        section_text = sections[section]
        cited_ids = re.findall(r"\[(E\d+)\]", section_text)
        if section != "Sources" and not cited_ids:
            missing_citations.append(section)
            report_section_feedback.append(
                ReportSectionFeedback(
                    section_label=section,
                    quoted_text=section_text[:200],
                    issue="Missing citations",
                    why_it_fails="The section makes claims without inline evidence ids.",
                    suggested_fix="Tie each material claim in this section to one or more evidence ids.",
                )
            )

    if not state.evidence_ledger:
        report_section_feedback.append(
            ReportSectionFeedback(
                section_label="Executive Summary",
                issue="No evidence ledger",
                why_it_fails="The report has no collected evidence behind it.",
                suggested_fix="Collect grounded evidence before finalizing.",
            )
        )

    missing_evidence_types: list[str] = []
    counterevidence_gaps: list[str] = []
    freshness_warnings: list[str] = []
    numeric_inconsistencies: list[str] = []
    blocking_issues: list[str] = []

    if state.research_plan and state.research_plan.counterevidence_topics:
        risk_text = sections.get("Risks and Counterevidence", "")
        if not risk_text or "risk" not in risk_text.lower():
            counterevidence_gaps.append("Risks and Counterevidence section is weak or absent.")
            report_section_feedback.append(
                ReportSectionFeedback(
                    section_label="Risks and Counterevidence",
                    quoted_text=risk_text[:200],
                    issue="Weak counterevidence coverage",
                    why_it_fails="The report does not clearly discuss the bear case or conflicting evidence.",
                    suggested_fix="Add explicit counterevidence, downside scenarios, and peer challenge evidence.",
                )
            )

    if state.research_plan and state.research_plan.freshness_requirements:
        if not re.search(r"\b(?:20\d{2}|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", draft):
            freshness_warnings.append("Time-sensitive report lacks an explicit date.")

    if "equity" in state.enabled_packs:
        peer_evidence = [
            item.id
            for item in state.evidence_ledger
            if item.id and ("peer" in item.summary.lower() or "competitor" in item.summary.lower())
        ]
        if not peer_evidence:
            missing_evidence_types.append("peer_or_competitor_evidence")
            counterevidence_gaps.append("No peer or competitor evidence was collected.")

    if "commodity" in state.enabled_packs:
        if "Curve and Positioning" in required_sections and "Curve and Positioning" in sections:
            if not re.findall(r"\[(E\d+)\]", sections["Curve and Positioning"]):
                missing_evidence_types.append("curve_or_positioning_evidence")

    if "macro" in state.enabled_packs:
        if "Scenarios" in required_sections and "Scenarios" in sections:
            if "scenario" not in sections["Scenarios"].lower():
                report_section_feedback.append(
                    ReportSectionFeedback(
                        section_label="Scenarios",
                        quoted_text=sections["Scenarios"][:200],
                        issue="Weak scenario analysis",
                        why_it_fails="The macro section does not clearly discuss alternative scenarios.",
                        suggested_fix="Add at least one alternative scenario or policy failure mode.",
                    )
                )

    if missing_sections:
        blocking_issues.extend(f"Missing section: {section}" for section in missing_sections)
    if missing_citations:
        blocking_issues.extend(f"Missing citations in: {section}" for section in missing_citations)
    if not state.evidence_ledger:
        blocking_issues.append("No evidence ledger entries were available.")

    follow_up_calls: list[SkillCall] = []
    ready_steps = get_unblocked_steps(state)
    if missing_sections or missing_citations or counterevidence_gaps:
        for step in ready_steps[:2]:
            follow_up_calls.extend(step.recommended_skill_calls[:1])
    if not follow_up_calls and counterevidence_gaps:
        follow_up_calls.append(
            SkillCall(
                name="search_and_read",
                arguments={
                    "queries": [f"{state.request.user_prompt} risks counterevidence bearish case"],
                    "urls_per_query": 2,
                    "use_news": True,
                },
            )
        )

    if blocking_issues:
        decision = "revise"
    else:
        decision = "pass"

    if not state.evidence_ledger:
        grounding = "fail"
    else:
        grounding = "pass" if not missing_sections else "revise"
    completeness = "pass" if not missing_sections else "revise"
    if not state.evidence_ledger:
        citation_coverage = "fail"
    else:
        citation_coverage = "pass" if not missing_citations else "revise"
    formatting = "pass" if draft.startswith("# ") else "revise"
    numeric_consistency = "pass" if not numeric_inconsistencies else "revise"

    return VerificationReport(
        decision=decision,
        summary="Structured evaluator completed contract checks.",
        grounding=grounding,
        completeness=completeness,
        numeric_consistency=numeric_consistency,
        citation_coverage=citation_coverage,
        formatting=formatting,
        improvement_instructions=[
            feedback.suggested_fix for feedback in report_section_feedback[:8]
        ],
        missing_evidence=[item for item in missing_evidence_types],
        blocking_issues=blocking_issues,
        missing_sections=missing_sections,
        missing_evidence_types=missing_evidence_types,
        missing_citations=missing_citations,
        freshness_warnings=freshness_warnings,
        numeric_inconsistencies=numeric_inconsistencies,
        required_follow_up_calls=follow_up_calls,
        counterevidence_gaps=counterevidence_gaps,
        report_section_feedback=report_section_feedback,
        raw_feedback="Fallback evaluator applied structural checks.",
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
You are the evaluator for a research harness draft.

Score these categories with only: pass, revise, or fail
- grounding
- completeness
- numeric_consistency
- citation_coverage
- formatting

Rules:
- Inspect the supplied plan, contract, and evidence rather than only the prose.
- If a required section is missing, do not pass completeness.
- If a material claim lacks support from the provided evidence list, do not pass grounding.
- If the draft does not cite evidence IDs, do not pass citation_coverage.
- Include report_section_feedback for weak or missing sections.
- Include required_follow_up_calls when concrete next actions are obvious.
- Keep summary short and factual.
"""

    user_prompt = f"""Mission:
{state.mission_text or state.request.user_prompt}

Progress:
{state.progress_text or "None"}

Evidence ledger:
{_format_evidence(state)}

Research plan:
{state.research_plan.model_dump_json(indent=2) if state.research_plan else "None"}

Research contract:
{state.research_contract.model_dump_json(indent=2) if state.research_contract else "None"}

Draft to judge:
{draft}
"""

    try:
        model_name = get_model("harness", "evaluator")
        llm = get_llm(model_name).with_structured_output(VerificationReport, method="json_mode")
        report = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return cast(VerificationReport, report)
    except Exception as exc:
        print(f"Harness verifier fallback triggered: {exc}")
        return _rule_based_evaluation(state, draft)
