"""Contract-aware draft writer for harness responses."""

from __future__ import annotations

import re
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage

from src.harness.types import HarnessState
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model


COMMON_SECTION_TITLES = {
    "Executive Summary",
    "Key Findings",
    "Equity Overview",
    "Peer and Competitive Pressure",
    "Valuation and Scenarios",
    "Macro Transmission",
    "Scenarios",
    "Commodity Balance",
    "Curve and Positioning",
    "Risks and Counterevidence",
    "Sources",
}


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_val = item.get("text")
                if isinstance(text_val, str):
                    parts.append(text_val)
        return "\n".join(parts)
    return str(content)


def _ordered_sections(state: HarnessState) -> list[str]:
    if state.research_plan and state.research_plan.required_sections:
        return state.research_plan.required_sections
    return ["Executive Summary", "Key Findings", "Risks and Counterevidence", "Sources"]


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
        bold_match = re.fullmatch(r"\*\*([^*]+)\*\*:?", line.strip())
        if bold_match:
            sections[current] = "\n".join(buffer).strip()
            current = bold_match.group(1).strip()
            buffer = []
            continue
        buffer.append(line)
    sections[current] = "\n".join(buffer).strip()
    return sections


def _format_evidence(state: HarnessState, limit: int = 28) -> str:
    if not state.evidence_ledger:
        return "No evidence collected."

    chunks = []
    for item in state.evidence_ledger[-limit:]:
        source_text = ", ".join(item.sources or item.artifact_paths) or item.source_type
        content = (item.content or "").strip()
        if len(content) > 1000:
            content = content[:1000] + "..."
        chunks.append(f"[{item.id}] {item.summary}\nSource: {source_text}\n{content}")
    return "\n\n".join(chunks)


def _format_contract(state: HarnessState) -> str:
    if not state.research_contract:
        return "No contract loaded."
    clauses = []
    for group in (
        state.research_contract.global_clauses,
        state.research_contract.section_clauses,
        state.research_contract.numeric_clauses,
        state.research_contract.counterevidence_clauses,
    ):
        clauses.extend(group)
    return "\n".join(f"- ({clause.severity}) {clause.text}" for clause in clauses[:20])


def _format_step_results(state: HarnessState) -> str:
    if not state.step_results:
        return "No step results yet."
    lines = []
    for result in state.step_results[-12:]:
        lines.append(f"- {result.step_id} [{result.status}]: {result.summary}")
        for finding in result.findings[:6]:
            lines.append(f"  finding: {finding}")
    return "\n".join(lines)


def _format_section_briefs(state: HarnessState) -> str:
    if not state.section_briefs:
        return "No section briefs."
    lines = []
    for brief in state.section_briefs[:16]:
        lines.append(f"## {brief.section_label} [{brief.coverage_status}]")
        lines.append(brief.summary)
        for fact in brief.key_facts[:4]:
            lines.append(f"- fact: {fact}")
        for point in brief.counterpoints[:2]:
            lines.append(f"- counterpoint: {point}")
    return "\n".join(lines)


def _format_fact_index(state: HarnessState) -> str:
    if not state.fact_index:
        return "No fact index."
    lines = []
    for record in state.fact_index[:24]:
        lines.append(
            f"- [{record.fact_id}] ({record.stance}) {record.fact} | sections={record.section_labels}"
        )
    return "\n".join(lines)


def _format_coverage_matrix(state: HarnessState) -> str:
    if not state.coverage_matrix:
        return "No coverage matrix."
    lines = []
    for entry in [
        *state.coverage_matrix.sections,
        *state.coverage_matrix.evidence_types,
        *state.coverage_matrix.counterevidence_requirements,
    ][:18]:
        lines.append(f"- {entry.coverage_type}:{entry.label} [{entry.status}]")
    return "\n".join(lines)


def _title_from_prompt(state: HarnessState) -> str:
    text = state.request.user_prompt.strip().rstrip(".?")
    if not text:
        return "Research Report"
    return text


def _normalize_markdown_report(state: HarnessState, draft: str) -> str:
    """Force a stable title plus markdown section headings."""

    if not draft.strip():
        return _fallback_draft(state)

    required_sections = set(_ordered_sections(state)) | COMMON_SECTION_TITLES
    lines = draft.splitlines()
    normalized_lines: list[str] = []
    has_title = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not has_title and stripped.startswith("# "):
            has_title = True
            if stripped.lower().startswith("# response to:"):
                normalized_lines.append(f"# {_title_from_prompt(state)}")
            else:
                normalized_lines.append(line)
            continue

        bold_match = re.fullmatch(r"\*\*([^*]+)\*\*:?", stripped)
        if bold_match:
            heading = bold_match.group(1).strip()
            if heading in required_sections:
                normalized_lines.append(f"## {heading}")
                continue

        if index == 0 and stripped and not stripped.startswith("# "):
            normalized_lines.append(f"# {_title_from_prompt(state)}")
            normalized_lines.append("")
            has_title = True

        normalized_lines.append(line)

    if not has_title:
        normalized_lines.insert(0, "")
        normalized_lines.insert(0, f"# {_title_from_prompt(state)}")

    return "\n".join(normalized_lines).strip()


def _section_evidence(state: HarnessState, section_label: str) -> list[str]:
    relevant_ids: list[str] = []
    lowered_section = section_label.lower()
    for item in state.evidence_ledger:
        lowered_summary = item.summary.lower()
        if lowered_section.startswith("source"):
            continue
        if (
            lowered_section in lowered_summary
            or any(token in lowered_summary for token in lowered_section.split())
            or lowered_section in {"executive summary", "key findings", "risks and counterevidence"}
        ):
            if item.id:
                relevant_ids.append(item.id)
    if not relevant_ids:
        relevant_ids = [item.id for item in state.evidence_ledger[-4:] if item.id]
    return relevant_ids[:4]


def _fallback_section_text(state: HarnessState, section_label: str) -> str:
    if state.request.research_profile == "deep":
        for brief in state.section_briefs:
            if brief.section_label == section_label and brief.coverage_status != "missing":
                facts = "\n".join(f"- {fact}" for fact in brief.key_facts[:5])
                counterpoints = "\n".join(f"- {point}" for point in brief.counterpoints[:3])
                body = brief.summary
                if facts:
                    body += f"\n\n{facts}"
                if counterpoints:
                    body += f"\n\nCounterevidence:\n{counterpoints}"
                return body

    evidence_ids = _section_evidence(state, section_label)
    citations = " ".join(f"[{evidence_id}]" for evidence_id in evidence_ids) or "[E1]"

    if section_label == "Executive Summary":
        return (
            "The current evidence set points to the main drivers of the question, "
            "the most important scenario split, and the highest-confidence facts. "
            "This summary should be read alongside the explicit risks and counterevidence below. "
            f"{citations}"
        )
    if section_label == "Key Findings":
        bullets = []
        for item in state.evidence_ledger[-8:]:
            if item.id:
                bullets.append(f"- {item.summary} [{item.id}]")
        return "\n".join(bullets) or f"- No grounded findings were collected yet. {citations}"
    if section_label == "Macro Transmission":
        return (
            "The macro transmission runs through rates, inflation expectations, credit conditions, "
            "and asset-pricing effects. Each channel should be tied back to the collected evidence. "
            f"{citations}"
        )
    if section_label == "Scenarios":
        return (
            "Lay out a base case, upside case, and downside case with the main trigger conditions, "
            "rather than giving only one linear forecast. "
            f"{citations}"
        )
    if section_label in {"Peer and Competitive Pressure", "Risks and Counterevidence"}:
        return (
            "Counterevidence matters here: peer pressure, adverse scenarios, or conflicting "
            f"data should temper the base case. {citations}"
        )
    if section_label == "Sources":
        lines = []
        for item in state.evidence_ledger[-10:]:
            source_text = ", ".join(item.sources or item.artifact_paths) or item.source_type
            lines.append(f"- [{item.id}] {source_text}")
        return "\n".join(lines) or "- No sources collected."
    return (
        f"This section is grounded in the collected evidence and current step results. {citations}"
    )


def _fallback_draft(state: HarnessState) -> str:
    existing_sections = _parse_sections(state.latest_draft or "")
    targeted = set()
    if state.verification_reports:
        last_report = state.verification_reports[-1]
        targeted.update(item.section_label for item in last_report.report_section_feedback)
        targeted.update(last_report.missing_sections)

    lines = [f"# {_title_from_prompt(state)}", ""]
    for section in _ordered_sections(state):
        lines.append(f"## {section}")
        if (
            section in existing_sections
            and section not in targeted
            and existing_sections[section].strip()
        ):
            lines.append(existing_sections[section])
        else:
            lines.append(_fallback_section_text(state, section))
        lines.append("")
    return "\n".join(lines).strip()


def _targeted_revision_needed(state: HarnessState) -> bool:
    if not state.latest_draft or not state.verification_reports:
        return False
    last_report = state.verification_reports[-1]
    return bool(last_report.report_section_feedback or last_report.missing_sections)


def write_draft(state: HarnessState) -> str:
    """Write a markdown draft grounded in the plan, contract, and evidence."""

    targeted_revision = _targeted_revision_needed(state)
    system_prompt = """
You are writing the user-facing answer for a research harness.

Rules:
- Answer the user's actual question directly.
- Use only the supplied evidence.
- If a statement is an inference, label it clearly as an inference.
- Cite evidence IDs inline like [E1], [E2].
- Satisfy the supplied contract and required sections.
- Start with exactly one H1 title line.
- Use markdown section headings like '## Executive Summary', never bold-only pseudo-headings.
- Write a substantive professional report, not a short memo.
- For a single-domain prompt, aim for roughly 1,200 to 2,200 words when the evidence supports it.
- End with a Sources section listing the evidence IDs you relied on.
"""

    if targeted_revision:
        system_prompt += """
- Revise only the flagged or missing sections by default.
- Preserve unflagged sections if they are already adequate.
"""

    if state.request.research_profile == "deep":
        system_prompt += """
- Draft from the supplied section briefs, fact index, and coverage matrix, not from raw working memory alone.
- For deep single-name equity, aim for roughly 4,000 to 8,000 words when the corpus depth supports it.
- Integrate counterevidence and peer pressure as first-class parts of the report.
"""

    user_prompt = f"""Mission:
{state.mission_text or state.request.user_prompt}

Progress:
{state.progress_text or "None"}

Contract:
{_format_contract(state)}

Step results:
{_format_step_results(state)}

Section briefs:
{_format_section_briefs(state)}

Fact index:
{_format_fact_index(state)}

Coverage matrix:
{_format_coverage_matrix(state)}

Revision notes:
{chr(10).join(state.revision_notes[-10:]) or "None"}

Existing draft:
{state.latest_draft or "None"}

Evidence:
{_format_evidence(state)}
"""

    try:
        model_name = get_model("harness", "writer")
        llm = get_llm(model_name)
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        text = _content_to_text(cast(object, response.content)).strip()
        if text:
            return _normalize_markdown_report(state, text)
    except Exception as exc:
        print(f"Harness writer fallback triggered: {exc}")

    return _normalize_markdown_report(state, _fallback_draft(state))
