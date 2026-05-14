from __future__ import annotations

import json
from pathlib import Path

from src.research_platform.state.contracts import EvidenceIndex, OpenQuestion, OpenQuestionsFile
from src.research_platform.state.lint import validate_state_integrity
from src.research_platform.state.staging import commit_stage, prepare_state_stage
from src.research_platform.state.storage import initialize_state_folder, read_json_model, write_json_model
from src.research_platform.state.validator import validate_stage
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


def _prepare(tmp_path: Path):
    vault_root = tmp_path / "vault"
    live = initialize_state_folder(vault_root, "XOM")
    write_json_model(live.evidence_index, EvidenceIndex(ticker="XOM", entries={"S1": _evidence()}))
    stage_paths = prepare_state_stage(
        live_paths=live,
        baseline_root=tmp_path / "baseline",
        stage_root=tmp_path / "stage",
    )
    return vault_root, live, stage_paths


def test_validate_and_commit_stage_direct_markdown_edit(tmp_path: Path) -> None:
    vault_root, live, stage_paths = _prepare(tmp_path)
    staged_markdown = stage_paths.stage.research_state.read_text(encoding="utf-8")
    staged_markdown += "\n## Guyana growth engine\n<!-- key: guyana_growth_engine -->\n\nGuyana production is durable growth evidence [S1].\n"
    stage_paths.stage.research_state.write_text(staged_markdown, encoding="utf-8")

    report = validate_stage(stage_paths, ticker="XOM", run_id="run-v33")

    assert report.ok, report.errors
    result = commit_stage(stage_paths=stage_paths, vault_root=vault_root, ticker="XOM", run_id="run-v33", diff=report.diff)
    assert result.committed is True
    assert "Guyana production is durable growth evidence [S1]." in live.research_state.read_text(encoding="utf-8")
    assert validate_state_integrity(vault_root, "XOM") == []
    assert "v33_stage_commit" in live.diff_log.read_text(encoding="utf-8")


def test_validate_stage_rerenders_hand_edited_derived_section_from_sidecar(tmp_path: Path) -> None:
    _, _, stage_paths = _prepare(tmp_path)
    open_questions = OpenQuestionsFile(
        ticker="XOM",
        questions=[OpenQuestion(question_id="q-cycle", text="What deck is required through the cycle?", priority="high")],
    )
    write_json_model(stage_paths.stage.open_questions, open_questions)
    markdown = stage_paths.stage.research_state.read_text(encoding="utf-8").replace(
        "No open questions recorded.",
        "LLM hand-edited derived text that should be overwritten.",
    )
    stage_paths.stage.research_state.write_text(markdown, encoding="utf-8")

    report = validate_stage(stage_paths, ticker="XOM", run_id="run-v33")

    assert report.ok, report.errors
    rerendered = stage_paths.stage.research_state.read_text(encoding="utf-8")
    assert "LLM hand-edited derived text" not in rerendered
    assert "q-cycle" in rerendered
    assert "What deck is required through the cycle?" in rerendered


def test_validate_stage_rejects_malformed_sidecar_without_live_mutation(tmp_path: Path) -> None:
    _, live, stage_paths = _prepare(tmp_path)
    original_live = live.research_state.read_text(encoding="utf-8")
    stage_paths.stage.open_questions.write_text("not json\n", encoding="utf-8")

    report = validate_stage(stage_paths, ticker="XOM", run_id="run-v33")

    assert report.ok is False
    assert any("invalid open_questions.json" in error for error in report.errors)
    assert live.research_state.read_text(encoding="utf-8") == original_live



def test_validate_stage_ignores_uncited_numbers_in_rerendered_derived_questions(tmp_path: Path) -> None:
    _, _, stage_paths = _prepare(tmp_path)
    open_questions = OpenQuestionsFile(
        ticker="XOM",
        questions=[
            OpenQuestion(
                question_id="q-v33-plumbing-cycle-deck",
                text="What commodity price deck and downstream margin assumptions are required for through-cycle return underwriting?",
            )
        ],
    )
    write_json_model(stage_paths.stage.open_questions, open_questions)

    report = validate_stage(stage_paths, ticker="XOM", run_id="run-v33")

    assert report.ok, report.errors
    assert "q-v33-plumbing-cycle-deck" in stage_paths.stage.research_state.read_text(encoding="utf-8")


def test_validate_stage_rejects_unresolved_cite_and_uncited_number(tmp_path: Path) -> None:
    _, _, stage_paths = _prepare(tmp_path)
    staged_markdown = stage_paths.stage.research_state.read_text(encoding="utf-8")
    staged_markdown += "\n## Bad section\n<!-- key: bad_section -->\n\nUnresolved claim [S9]. Revenue was 123 billion.\n"
    stage_paths.stage.research_state.write_text(staged_markdown, encoding="utf-8")

    report = validate_stage(stage_paths, ticker="XOM", run_id="run-v33")

    assert report.ok is False
    assert "unresolved cite key: S9" in report.errors
    assert any("quantitative-looking sentence" in error for error in report.errors)


def test_validate_stage_rejects_readonly_evidence_or_index_edit(tmp_path: Path) -> None:
    _, _, stage_paths = _prepare(tmp_path)
    stage_paths.stage.evidence_index.write_text(json.dumps({"schema_version": 1, "ticker": "XOM", "entries": {}}), encoding="utf-8")
    stage_paths.stage.state_index.write_text(json.dumps({"schema_version": 1, "ticker": "XOM", "sections": []}), encoding="utf-8")

    report = validate_stage(stage_paths, ticker="XOM", run_id="run-v33")

    assert report.ok is False
    assert any("read-only state file was modified: evidence_index.json" in error for error in report.errors)
    assert any("read-only state file was modified: state_index.json" in error for error in report.errors)


def test_validate_stage_rejects_excessive_section_deletion(tmp_path: Path) -> None:
    _, _, stage_paths = _prepare(tmp_path)
    stage_paths.stage.research_state.write_text("# XOM Research State\n\n", encoding="utf-8")

    report = validate_stage(stage_paths, ticker="XOM", run_id="run-v33", max_removed_sections=0)

    assert report.ok is False
    assert any("stage removed too many sections" in error for error in report.errors)
