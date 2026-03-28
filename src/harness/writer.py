"""Contract-aware draft writer for harness responses."""

from __future__ import annotations

import re
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage

from src.harness.types import HarnessState
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model


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
        buffer.append(line)
    sections[current] = "\n".join(buffer).strip()
    return sections


def _format_evidence(state: HarnessState, limit: int = 16) -> str:
    if not state.evidence_ledger:
        return "No evidence collected."

    chunks = []
    for item in state.evidence_ledger[-limit:]:
        source_text = ", ".join(item.sources or item.artifact_paths) or item.source_type
        content = (item.content or "").strip()
        if len(content) > 700:
            content = content[:700] + "..."
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
        for finding in result.findings[:3]:
            lines.append(f"  finding: {finding}")
    return "\n".join(lines)


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
    evidence_ids = _section_evidence(state, section_label)
    citations = " ".join(f"[{evidence_id}]" for evidence_id in evidence_ids) or "[E1]"

    if section_label == "Executive Summary":
        return (
            "The current evidence set points to the main drivers of the question, "
            "but the conclusion should still be read alongside the explicit risks below. "
            f"{citations}"
        )
    if section_label == "Key Findings":
        bullets = []
        for item in state.evidence_ledger[-4:]:
            if item.id:
                bullets.append(f"- {item.summary} [{item.id}]")
        return "\n".join(bullets) or f"- No grounded findings were collected yet. {citations}"
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

    lines = [f"# Response to: {state.request.user_prompt}", ""]
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
- End with a Sources section listing the evidence IDs you relied on.
"""

    if targeted_revision:
        system_prompt += """
- Revise only the flagged or missing sections by default.
- Preserve unflagged sections if they are already adequate.
"""

    user_prompt = f"""Mission:
{state.mission_text or state.request.user_prompt}

Progress:
{state.progress_text or "None"}

Contract:
{_format_contract(state)}

Step results:
{_format_step_results(state)}

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
            return text
    except Exception as exc:
        print(f"Harness writer fallback triggered: {exc}")

    return _fallback_draft(state)
