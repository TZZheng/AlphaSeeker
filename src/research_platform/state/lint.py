"""Integrity checks for markdown-first research state."""

from __future__ import annotations

import re
from pathlib import Path

from src.research_platform.state.render import render_conflicts, render_open_questions, render_valuation_snapshot
from src.research_platform.state.storage import CITE_TAG_RE, parse_markdown_sections, read_state_sidecars, state_paths


NUMBERISH_RE = re.compile(r"(?<![A-Za-z])(?:\$\s*)?\d+(?:\.\d+)?\s*(?:%|x|billion|million|bn|mm|mb/d|boe/d)?", re.IGNORECASE)
CITED_SENTENCE_RE = re.compile(r"\[(S[1-9][0-9]*)\]")


def number_without_cite_warnings(markdown: str) -> list[str]:
    warnings: list[str] = []
    for sentence in re.split(r"(?<=[.!?。！？])\s+", markdown):
        if NUMBERISH_RE.search(sentence) and not CITED_SENTENCE_RE.search(sentence):
            warnings.append(sentence.strip()[:160])
    return [warning for warning in warnings if warning]


def _body_without_managed_comments(body: str) -> str:
    lines = [line for line in body.splitlines() if not line.strip().startswith("<!-- derived:")]
    return "\n".join(lines).strip()


def validate_state_integrity(vault_root: str | Path, ticker: str) -> list[str]:
    """Return integrity errors for the current state folder."""

    paths = state_paths(vault_root, ticker)
    errors: list[str] = []
    if not paths.research_state.exists():
        return [f"missing {paths.research_state}"]

    markdown = paths.research_state.read_text(encoding="utf-8")
    sections = parse_markdown_sections(markdown)
    section_keys = {section.section_key for section in sections}
    state_index, evidence_index, open_questions, conflicts, valuation = read_state_sidecars(paths, ticker)

    for cite_key in sorted(set(CITE_TAG_RE.findall(markdown))):
        if cite_key not in evidence_index.entries:
            errors.append(f"unresolved cite key: {cite_key}")

    indexed_keys = {section.section_key for section in state_index.sections}
    for key in indexed_keys - section_keys:
        errors.append(f"indexed section missing from markdown: {key}")
    for key in section_keys - indexed_keys:
        errors.append(f"markdown section missing from state_index: {key}")

    by_key = {section.section_key: section for section in sections}
    if "open_questions" in by_key:
        rendered = render_open_questions(open_questions).strip()
        if _body_without_managed_comments(by_key["open_questions"].body) != rendered:
            errors.append("open_questions section does not match rendered sidecar")
    if "valuation_snapshot" in by_key:
        rendered = render_valuation_snapshot(valuation).strip()
        if _body_without_managed_comments(by_key["valuation_snapshot"].body) != rendered:
            errors.append("valuation_snapshot section does not match rendered sidecar")
    if "conflicts_uncertainty" in by_key:
        rendered = render_conflicts(conflicts).strip()
        if _body_without_managed_comments(by_key["conflicts_uncertainty"].body) != rendered:
            errors.append("conflicts_uncertainty section does not match rendered sidecar")
    return errors
