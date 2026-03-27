"""Draft writer and claim-map builder for harness responses."""

from __future__ import annotations

import re
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage

from src.harness.types import ClaimRecord, HarnessState
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model


_EVIDENCE_ID_RE = re.compile(r"\[(E\d+)\]")


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


def _format_evidence(state: HarnessState, limit: int = 14) -> str:
    if not state.evidence_ledger:
        return "No evidence collected."

    chunks = []
    for item in state.evidence_ledger[-limit:]:
        source_text = ", ".join(item.sources or item.artifact_paths) or item.source_type
        content = (item.content or "").strip()
        if len(content) > 900:
            content = content[:900] + "..."
        chunks.append(f"[{item.id}] {item.summary}\nSource: {source_text}\n{content}")
    return "\n\n".join(chunks)


def _format_contract(state: HarnessState) -> str:
    if not state.research_contract:
        return "No research contract available."
    lines = []
    for group in [
        state.research_contract.global_clauses,
        state.research_contract.section_clauses,
        state.research_contract.freshness_clauses,
        state.research_contract.numeric_clauses,
        state.research_contract.counterevidence_clauses,
    ]:
        for clause in group[:20]:
            lines.append(f"- [{clause.category}] {clause.text}")
    return "\n".join(lines) or "No contract clauses."


def _format_step_results(state: HarnessState) -> str:
    if not state.step_results:
        return "No step results yet."
    return "\n".join(
        f"- {result.step_id}: {result.status} | {result.summary}"
        for result in state.step_results[-20:]
    )


def _format_section_feedback(state: HarnessState) -> str:
    if not state.verification_reports:
        return "No evaluator feedback."
    feedback = state.verification_reports[-1].report_section_feedback
    if not feedback:
        return "No section-specific evaluator feedback."
    return "\n".join(
        f"- Section: {item.section_label} | Issue: {item.issue} | Fix: {item.suggested_fix}"
        for item in feedback
    )


def _fallback_draft(state: HarnessState) -> str:
    lines = [
        f"# Response to: {state.request.user_prompt}",
        "",
        "## Answer",
    ]

    if state.revision_notes and state.latest_draft:
        lines.append("Revised draft based on evaluator feedback:")
        lines.append(state.latest_draft)
    elif not state.evidence_ledger:
        lines.append("Insufficient evidence was collected to produce a grounded answer.")
    else:
        lines.append("Grounded summary based on the evidence collected so far:")
        for item in state.evidence_ledger[-5:]:
            lines.append(f"- [{item.id}] {item.summary}")

    lines.extend(["", "## Sources"])
    if state.evidence_ledger:
        for item in state.evidence_ledger[-8:]:
            source_text = ", ".join(item.sources or item.artifact_paths) or item.source_type
            lines.append(f"- [{item.id}] {source_text}")
    else:
        lines.append("- No sources collected.")
    return "\n".join(lines)


def write_draft(state: HarnessState) -> str:
    """Write a markdown draft grounded in the current evidence ledger and dossier."""

    has_targeted_feedback = bool(
        state.verification_reports and state.verification_reports[-1].report_section_feedback
    )

    system_prompt = """
You are writing the final user-facing answer for a financial research harness.

Rules:
- Answer the user's actual question directly.
- Use only the supplied evidence.
- Follow the research contract.
- If a statement is an inference, label it clearly as an inference.
- Cite evidence IDs inline like [E1], [E2].
- Produce clean Markdown with concise sections.
- End with a short Sources section listing the evidence IDs you relied on.
- If evaluator feedback exists, revise only the weak sections when possible and preserve good sections.
"""

    current_draft = state.latest_draft or "No prior draft."
    user_prompt = f"""User request:
{state.request.user_prompt}

Mission:
{state.research_brief.primary_question if state.research_brief else state.request.user_prompt}

Research contract:
{_format_contract(state)}

Completed steps:
{_format_step_results(state)}

Recent progress:
{chr(10).join(state.progress_updates[-12:]) or "No progress yet."}

Evaluator section feedback:
{_format_section_feedback(state)}

Current draft:
{current_draft if has_targeted_feedback else "No targeted revision requested."}

Evidence:
{_format_evidence(state)}
"""

    try:
        model_name = get_model("harness", "writer")
        llm = get_llm(model_name)
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return _content_to_text(cast(object, response.content)).strip()
    except Exception as exc:
        print(f"Harness writer fallback triggered: {exc}")
        return _fallback_draft(state)


def build_claim_map(state: HarnessState, draft: str) -> list[ClaimRecord]:
    """Build a lightweight normalized claim map from the current draft."""

    if not draft.strip():
        return []

    section_label = "Answer"
    records: list[ClaimRecord] = []
    buffer: list[str] = []
    claim_index = 1

    def flush_buffer(current_section: str, lines: list[str], next_index: int) -> int:
        text = " ".join(line.strip() for line in lines if line.strip()).strip()
        if not text or text.startswith("#"):
            return next_index
        citations = _EVIDENCE_ID_RE.findall(text)
        lowered = text.lower()
        claim_type = "inference" if any(token in lowered for token in ("inference", "likely", "may", "could", "suggests")) else "fact"
        complicating = citations if any(token in lowered for token in ("however", "risk", "downside", "counter", "bear")) else []
        records.append(
            ClaimRecord(
                id=f"C{next_index}",
                text=text,
                section_label=current_section,
                claim_type=claim_type,
                supporting_evidence_ids=list(dict.fromkeys(citations)),
                complicating_evidence_ids=list(dict.fromkeys(complicating)),
                appears_in_report=True,
            )
        )
        return next_index + 1

    for raw_line in draft.splitlines():
        line = raw_line.rstrip()
        if line.startswith("#"):
            claim_index = flush_buffer(section_label, buffer, claim_index)
            buffer = []
            section_label = line.lstrip("#").strip() or section_label
            continue
        if not line.strip():
            claim_index = flush_buffer(section_label, buffer, claim_index)
            buffer = []
            continue
        buffer.append(line)
    flush_buffer(section_label, buffer, claim_index)
    return records
