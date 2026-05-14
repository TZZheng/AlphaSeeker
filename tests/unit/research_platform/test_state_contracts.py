from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.research_platform.state.contracts import EvidenceIndex, ProposalRecord, StateIndex
from src.research_platform.state.storage import build_state_index, parse_markdown_sections
from src.vault.contracts import EvidenceRef


def _evidence(evidence_id: str = "ev_xom_1") -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence_id,
        document_id="doc_xom_10k",
        vault_relative_path="companies/XOM/sources/sec_filings/10k.md",
        display_title="XOM 10-K",
        quoted_snippet="Management reported production growth.",
        source_grade="A",
    )


def test_state_index_parses_anchored_markdown_sections() -> None:
    markdown = """# XOM Research State

## Guyana growth engine
<!-- key: guyana_growth_engine -->

Production growth is cited [S1].

## Risks / bear cases
<!-- key: risks_bear_cases -->

Commodity exposure remains material [S2].
"""

    sections = parse_markdown_sections(markdown)
    index = build_state_index(markdown, ticker="xom", run_id="run-1")

    assert [section.section_key for section in sections] == ["guyana_growth_engine", "risks_bear_cases"]
    assert index.ticker == "XOM"
    assert index.sections[0].heading == "Guyana growth engine"
    assert index.sections[0].cite_keys == ["S1"]
    assert index.sections[0].byte_range is not None


def test_state_index_rejects_duplicate_section_keys() -> None:
    with pytest.raises(ValidationError):
        StateIndex(
            ticker="XOM",
            sections=[
                {"section_key": "risks", "heading": "Risks"},
                {"section_key": "risks", "heading": "Risks duplicate"},
            ],
        )


def test_evidence_index_enforces_stable_unique_cite_keys() -> None:
    index = EvidenceIndex(ticker="xom", entries={"S1": _evidence("ev1")})
    assert index.ticker == "XOM"
    assert index.entries["S1"].evidence_id == "ev1"

    with pytest.raises(ValidationError):
        EvidenceIndex(ticker="XOM", entries={"bad": _evidence("ev2")})

    with pytest.raises(ValidationError):
        EvidenceIndex(ticker="XOM", entries={"S1": _evidence("same"), "S2": _evidence("same")})


def test_proposal_record_validates_by_type() -> None:
    proposal = ProposalRecord(
        proposal_id="p1",
        type="propose_section_update",
        section_key="guyana_growth_engine",
        action="append",
        body_markdown="New cited insight [S1].",
        evidence_keys=["S1"],
        rationale="New evidence updates the section.",
    )
    assert proposal.section_key == "guyana_growth_engine"

    with pytest.raises(ValidationError):
        ProposalRecord(
            proposal_id="p2",
            type="propose_section_update",
            section_key="bad-key",
            action="append",
            body_markdown="Missing valid section key [S1].",
            evidence_keys=["S1"],
        )

    with pytest.raises(ValidationError):
        ProposalRecord(
            proposal_id="p3",
            type="propose_close_question",
            question_id="q1",
            proposed_answer="Answered but uncited.",
        )
