from __future__ import annotations

import json
from pathlib import Path

from src.research_platform.state.apply import apply_proposals
from src.research_platform.state.contracts import EvidenceIndex
from src.research_platform.state.lint import validate_state_integrity
from src.research_platform.state.storage import initialize_state_folder, read_json_model, write_json_model, write_jsonl
from src.vault.contracts import EvidenceRef


def _evidence(evidence_id: str = "ev_xom_1", snippet: str = "Production growth was reported.") -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence_id,
        document_id="doc_xom_10k",
        vault_relative_path="companies/XOM/sources/sec_filings/10k.md",
        display_title="XOM 10-K",
        quoted_snippet=snippet,
        source_grade="A",
    )


def test_apply_section_proposal_updates_markdown_index_and_audit(tmp_path: Path) -> None:
    vault_root = tmp_path / "vault"
    paths = initialize_state_folder(vault_root, "XOM")
    write_json_model(paths.evidence_index, EvidenceIndex(ticker="XOM", entries={"S1": _evidence()}))
    proposals_path = tmp_path / "proposals.jsonl"
    accepted_path = tmp_path / "accepted_changes.jsonl"
    write_jsonl(
        proposals_path,
        [
            {
                "proposal_id": "p1",
                "type": "propose_section_update",
                "section_key": "guyana_growth_engine",
                "action": "create",
                "body_markdown": "Guyana production is an important growth driver [S1].",
                "evidence_keys": ["S1"],
                "rationale": "First durable Guyana section.",
            }
        ],
    )

    result = apply_proposals(
        vault_root=vault_root,
        ticker="XOM",
        run_id="run-1",
        proposals_path=proposals_path,
        accepted_changes_path=accepted_path,
    )

    markdown = paths.research_state.read_text(encoding="utf-8")
    state_index = json.loads(paths.state_index.read_text(encoding="utf-8"))
    diff_log = paths.diff_log.read_text(encoding="utf-8")
    accepted = accepted_path.read_text(encoding="utf-8")

    assert result.applied_count == 1
    assert "<!-- key: guyana_growth_engine -->" in markdown
    assert "Guyana production is an important growth driver [S1]." in markdown
    assert any(section["section_key"] == "guyana_growth_engine" for section in state_index["sections"])
    assert "p1" in diff_log
    assert "p1" in accepted
    assert validate_state_integrity(vault_root, "XOM") == []


def test_apply_rejects_unresolved_evidence_without_mutating_state(tmp_path: Path) -> None:
    vault_root = tmp_path / "vault"
    paths = initialize_state_folder(vault_root, "XOM")
    original = paths.research_state.read_text(encoding="utf-8")
    proposals_path = tmp_path / "proposals.jsonl"
    write_jsonl(
        proposals_path,
        [
            {
                "proposal_id": "p_bad",
                "type": "propose_section_update",
                "section_key": "risks_bear_cases",
                "action": "append",
                "body_markdown": "Unresolved citation [S9].",
                "evidence_keys": ["S9"],
                "rationale": "Should fail.",
            }
        ],
    )

    result = apply_proposals(vault_root=vault_root, ticker="XOM", run_id="run-1", proposals_path=proposals_path)

    assert result.rejected_count == 1
    assert paths.research_state.read_text(encoding="utf-8") == original
    assert "unresolved evidence keys" in paths.diff_log.read_text(encoding="utf-8")


def test_apply_question_and_close_updates_sidecar_and_rendered_section(tmp_path: Path) -> None:
    vault_root = tmp_path / "vault"
    paths = initialize_state_folder(vault_root, "XOM")
    write_json_model(paths.evidence_index, EvidenceIndex(ticker="XOM", entries={"S1": _evidence()}))
    proposals_path = tmp_path / "proposals.jsonl"
    write_jsonl(
        proposals_path,
        [
            {
                "proposal_id": "p_q1",
                "type": "propose_question",
                "question_id": "q-capex",
                "text": "What is the latest capex guidance?",
                "priority": "high",
                "related_section_key": "key_guidance_metrics",
            },
            {
                "proposal_id": "p_q2",
                "type": "propose_close_question",
                "question_id": "q-capex",
                "evidence_keys": ["S1"],
                "proposed_answer": "The filing provides the updated range.",
                "rationale": "Directly answered by filing evidence.",
            },
        ],
    )

    result = apply_proposals(vault_root=vault_root, ticker="XOM", run_id="run-1", proposals_path=proposals_path)

    questions = json.loads(paths.open_questions.read_text(encoding="utf-8"))["questions"]
    markdown = paths.research_state.read_text(encoding="utf-8")

    assert result.applied_count == 2
    assert questions[0]["question_id"] == "q-capex"
    assert questions[0]["status"] == "proposed_close"
    assert "q-capex" in markdown
    assert "Proposed answer" in markdown
    assert validate_state_integrity(vault_root, "XOM") == []


def test_apply_valuation_snapshot_writes_sidecar_and_rendered_markdown_pointer(tmp_path: Path) -> None:
    vault_root = tmp_path / "vault"
    paths = initialize_state_folder(vault_root, "XOM")
    write_json_model(paths.evidence_index, EvidenceIndex(ticker="XOM", entries={"S1": _evidence()}))
    proposals_path = tmp_path / "proposals.jsonl"
    write_jsonl(
        proposals_path,
        [
            {
                "proposal_id": "p_val",
                "type": "propose_valuation_snapshot",
                "as_of": "2026-05-14",
                "fields": {"share_price": 100.0, "ev_ebitda": 7.5},
                "source_keys": ["S1"],
                "assumptions_markdown": "Point-in-time valuation snapshot [S1].",
            }
        ],
    )

    result = apply_proposals(vault_root=vault_root, ticker="XOM", run_id="run-1", proposals_path=proposals_path)
    valuation = json.loads(paths.valuation_snapshot.read_text(encoding="utf-8"))

    markdown = paths.research_state.read_text(encoding="utf-8")

    assert result.applied_count == 1
    assert valuation["as_of"] == "2026-05-14"
    assert valuation["fields"]["ev_ebitda"] == 7.5
    assert "<!-- key: valuation_snapshot -->" in markdown
    assert "As of: 2026-05-14" in markdown
    assert "| ev_ebitda | 7.5 |" in markdown
    assert "Point-in-time valuation snapshot [S1]." in markdown
    assert validate_state_integrity(vault_root, "XOM") == []
