"""Apply StateOwner decisions to markdown-first research state files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

from src.research_platform.state.contracts import (
    ConflictEntry,
    OpenQuestion,
    ProposalRecord,
    StateUpdateDecision,
    ValuationSnapshot,
    utc_now_iso,
)
from src.research_platform.state.policy import decide_proposal
from src.research_platform.state.render import render_conflicts, render_open_questions
from src.research_platform.state.storage import (
    MarkdownSection,
    append_jsonl,
    initialize_state_folder,
    parse_markdown_sections,
    read_proposals,
    read_state_sidecars,
    state_paths,
    write_json_model,
    write_jsonl,
    write_research_state,
)


@dataclass
class ApplyResult:
    decisions: list[StateUpdateDecision] = field(default_factory=list)
    applied_count: int = 0
    rejected_count: int = 0
    revised_count: int = 0


def _heading_from_key(section_key: str) -> str:
    return section_key.replace("_", " ").title()


def _section_block(heading: str, section_key: str, body: str) -> str:
    body = body.strip("\n")
    return f"## {heading}\n<!-- key: {section_key} -->\n\n{body}\n"


def _find_section(sections: list[MarkdownSection], section_key: str) -> MarkdownSection | None:
    return next((section for section in sections if section.section_key == section_key), None)


def _replace_section(markdown: str, section: MarkdownSection, new_body: str) -> str:
    block = _section_block(section.heading, section.section_key, new_body)
    return markdown[: section.start] + block + "\n" + markdown[section.end :].lstrip("\n")


def _append_section(markdown: str, section: MarkdownSection, body: str) -> str:
    existing = section.body.strip("\n")
    combined = (existing + "\n\n" + body.strip("\n")).strip("\n")
    return _replace_section(markdown, section, combined)


def _create_section(markdown: str, section_key: str, body: str) -> str:
    suffix = "" if markdown.endswith("\n") else "\n"
    return markdown + suffix + "\n" + _section_block(_heading_from_key(section_key), section_key, body)


def _remove_section(markdown: str, section: MarkdownSection) -> str:
    return markdown[: section.start].rstrip("\n") + "\n\n" + markdown[section.end :].lstrip("\n")


def _apply_section_update(markdown: str, proposal: ProposalRecord) -> str:
    assert proposal.section_key is not None
    assert proposal.action is not None
    sections = parse_markdown_sections(markdown)
    section = _find_section(sections, proposal.section_key)
    body = proposal.body_markdown or ""

    if proposal.action == "create":
        if section:
            return _append_section(markdown, section, body)
        return _create_section(markdown, proposal.section_key, body)
    if proposal.action == "append":
        if section:
            return _append_section(markdown, section, body)
        return _create_section(markdown, proposal.section_key, body)
    if proposal.action == "replace":
        if section:
            return _replace_section(markdown, section, body)
        return _create_section(markdown, proposal.section_key, body)
    if proposal.action == "remove":
        return _remove_section(markdown, section) if section else markdown
    return markdown


def _next_question_id(existing: set[str]) -> str:
    index = 1
    today = utc_now_iso()[:10]
    while True:
        candidate = f"q-{today}-{index:03d}"
        if candidate not in existing:
            return candidate
        index += 1


def _next_conflict_id(existing: set[str]) -> str:
    index = 1
    today = utc_now_iso()[:10]
    while True:
        candidate = f"conf-{today}-{index:03d}"
        if candidate not in existing:
            return candidate
        index += 1


def _replace_derived_section(markdown: str, section_key: str, body: str) -> str:
    sections = parse_markdown_sections(markdown)
    section = _find_section(sections, section_key)
    if not section:
        return _create_section(markdown, section_key, body)
    return _replace_section(markdown, section, body)


def _apply_sidecar_update(paths, proposal: ProposalRecord, *, ticker: str) -> str | None:
    _, _, open_questions, conflicts, valuation = read_state_sidecars(paths, ticker)
    if proposal.type == "propose_question":
        existing_ids = {question.question_id for question in open_questions.questions}
        question_id = proposal.question_id or _next_question_id(existing_ids)
        existing = next((q for q in open_questions.questions if q.question_id == question_id), None)
        if existing:
            existing.text = proposal.text or existing.text
            existing.priority = proposal.priority
            existing.related_section_key = proposal.related_section_key
            existing.updated_at = utc_now_iso()
        else:
            open_questions.questions.append(
                OpenQuestion(
                    question_id=question_id,
                    text=proposal.text or "",
                    priority=proposal.priority,
                    related_section_key=proposal.related_section_key,
                )
            )
        write_json_model(paths.open_questions, open_questions)
        return "open_questions"

    if proposal.type == "propose_close_question":
        for question in open_questions.questions:
            if question.question_id == proposal.question_id:
                question.status = "proposed_close"
                question.evidence_keys = proposal.evidence_keys
                question.proposed_answer = proposal.proposed_answer
                question.rationale = proposal.rationale
                question.updated_at = utc_now_iso()
                break
        write_json_model(paths.open_questions, open_questions)
        return "open_questions"

    if proposal.type == "propose_conflict":
        existing_ids = {conflict.conflict_id for conflict in conflicts.conflicts}
        conflicts.conflicts.append(
            ConflictEntry(
                conflict_id=_next_conflict_id(existing_ids),
                summary=proposal.summary or "",
                left_evidence_key=proposal.left_evidence_key or "S1",
                right_evidence_key=proposal.right_evidence_key or "S1",
                severity=proposal.severity,
                rationale=proposal.rationale or None,
            )
        )
        write_json_model(paths.conflicts, conflicts)
        return "conflicts_uncertainty"

    if proposal.type == "propose_valuation_snapshot":
        valuation = ValuationSnapshot(
            ticker=ticker,
            as_of=proposal.as_of,
            fields=proposal.fields,
            source_keys=proposal.source_keys,
            assumptions_markdown=proposal.assumptions_markdown,
        )
        write_json_model(paths.valuation_snapshot, valuation)
        return None

    return None


def _refresh_derived_sections(markdown: str, paths, ticker: str) -> str:
    _, _, open_questions, conflicts, _ = read_state_sidecars(paths, ticker)
    markdown = _replace_derived_section(markdown, "open_questions", render_open_questions(open_questions))
    markdown = _replace_derived_section(markdown, "conflicts_uncertainty", render_conflicts(conflicts))
    return markdown


def apply_proposals(
    *,
    vault_root: str | Path,
    ticker: str,
    run_id: str,
    proposals_path: str | Path,
    accepted_changes_path: str | Path | None = None,
) -> ApplyResult:
    """Apply a run's proposals to durable state and append audit artifacts."""

    paths = initialize_state_folder(vault_root, ticker)
    proposals = read_proposals(proposals_path)
    _, evidence_index, _, _, _ = read_state_sidecars(paths, ticker)
    result = ApplyResult()
    accepted_records: list[dict[str, object]] = []
    markdown = paths.research_state.read_text(encoding="utf-8")

    for index, proposal in enumerate(proposals, start=1):
        decision = decide_proposal(proposal, evidence_index=evidence_index, proposal_number=index)
        result.decisions.append(decision)
        if decision.decision == "rejected":
            result.rejected_count += 1
            append_jsonl(paths.diff_log, {"run_id": run_id, "proposal": proposal.model_dump(mode="json"), "decision": decision.model_dump(mode="json")})
            continue
        if decision.decision == "revised":
            result.revised_count += 1
            result.applied_count += 1
        else:
            result.applied_count += 1

        if proposal.type == "propose_section_update":
            markdown = _apply_section_update(markdown, proposal)
        elif proposal.type != "propose_no_op":
            _apply_sidecar_update(paths, proposal, ticker=ticker)
            markdown = _refresh_derived_sections(markdown, paths, ticker)

        record = {"run_id": run_id, "proposal": proposal.model_dump(mode="json"), "decision": decision.model_dump(mode="json")}
        accepted_records.append(record)
        append_jsonl(paths.diff_log, record)

    write_research_state(paths, markdown, ticker=ticker, run_id=run_id)
    if accepted_changes_path is not None:
        write_jsonl(accepted_changes_path, accepted_records)
    return result
