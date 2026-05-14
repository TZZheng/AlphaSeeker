from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.harness.types import HarnessRequest, HarnessResponse
from src.research_platform.memo_run import SourceAdapters, run_research_memo, strict_render_template
from src.research_platform.state.storage import state_paths


def test_strict_render_template_raises_on_unresolved_placeholder():
    with pytest.raises(ValueError, match="missing"):
        strict_render_template("Hello {{name}} {{missing}}", {"name": "Ted"})


def test_run_research_memo_with_mocked_sources_writes_artifacts_and_context(tmp_path):
    manual = tmp_path / "manual.md"
    manual.write_text("# Manual packet\n\nCustomer concentration notes.", encoding="utf-8")
    final_source = tmp_path / "harness_final.md"
    final_source.write_text("# Final memo\n\nUse S1 and S2.", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_sec(*args, **kwargs):
        return [
            {"form_type": "10-Q", "filing_date": "2026-05-01", "url": "https://sec/q", "text": "Quarterly filing text"},
            {"form_type": "10-K", "filing_date": "2026-02-15", "url": "https://sec/k", "text": "Annual filing text"},
        ]

    def fake_profile(ticker: str, output_dir: str | Path | None = None):
        path = Path(output_dir) / "profile.md"
        path.write_text("# Profile", encoding="utf-8")
        return str(path), {"company_name": "Exxon Mobil"}

    def fake_financials(ticker: str, output_dir: str | Path | None = None):
        path = Path(output_dir) / "financials.md"
        path.write_text("# Financials", encoding="utf-8")
        return str(path), {"source": "test"}

    def fake_market(ticker: str, period: str = "1y", output_dir: str | Path | None = None):
        path = Path(output_dir) / "market.csv"
        path.write_text("Date,Close\n2026-01-01,100\n", encoding="utf-8")
        return str(path)

    def fake_harness(request: HarnessRequest) -> HarnessResponse:
        captured["request"] = request
        return HarnessResponse(
            status="completed",
            stop_reason="done",
            run_root=str(tmp_path / "run"),
            root_agent_path=str(tmp_path / "run" / "agents" / "root"),
            final_report_path=str(final_source),
        )

    result = run_research_memo(
        user_prompt="Build an investment memo for XOM.",
        ticker="xom",
        company_name="Exxon Mobil",
        manual_files=[str(manual)],
        vault_root=tmp_path / "vault",
        run_id="unit-run",
        adapters=SourceAdapters(fake_sec, fake_profile, fake_financials, fake_market),
        run_harness_fn=fake_harness,
    )

    assert result.status == "succeeded"
    assert Path(result.final_path).read_text(encoding="utf-8").startswith("# Final memo")
    request = captured["request"]
    assert isinstance(request, HarnessRequest)
    assert request.available_skill_packs == ["core", "equity"]
    assert request.context_files[:3] == [result.context_package_path, result.source_index_path, result.question_list_path]
    attached_context_files = [Path(path) for path in request.context_files[3:]]
    assert len(attached_context_files) == 6
    assert len({path.name for path in attached_context_files}) == 6
    assert all(path.exists() for path in attached_context_files)
    assert any(path.name.startswith("S1_manual_file_doc_") for path in attached_context_files)
    assert any(path.name.startswith("S2_sec_filing_doc_") for path in attached_context_files)
    assert "Prefer the attached context files" in request.user_prompt
    assert "Build an investment memo for XOM." in request.user_prompt

    package = json.loads(Path(result.context_package_path).read_text(encoding="utf-8"))
    assert package["ticker"] == "XOM"
    docs = package["documents"]
    assert any(doc["source_type"] == "manual_file" and doc["source_grade"] == "B" for doc in docs)
    assert any(doc["source_type"] == "sec_filing" and doc["source_grade"] == "A" for doc in docs)
    assert any(doc["source_type"] == "company_profile" and doc["source_grade"] == "B" for doc in docs)
    manual_doc = next(doc for doc in docs if doc["source_type"] == "manual_file")
    assert manual_doc["source_grade_rationale"] == "manual file, provenance not machine-verified"
    assert manual_doc["metadata"]["harness_context_file"].startswith("S1_manual_file_doc_")
    assert package["citations"][0]["metadata"]["harness_context_file"].startswith("S1_manual_file_doc_")

    source_index = Path(result.source_index_path).read_text(encoding="utf-8")
    assert "[[documents/" in source_index
    assert "Attached context file: `S1_manual_file_doc_" in source_index
    assert "Attached context file: `S2_sec_filing_doc_" in source_index
    assert "S1" in source_index
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["final_path"] == result.final_path


def test_run_research_memo_aborts_before_harness_without_required_source(tmp_path):
    harness_called = False

    def fake_sec(*args, **kwargs):
        return []

    def failing_provider(*args, **kwargs):
        raise RuntimeError("offline")

    def fake_harness(request: HarnessRequest) -> HarnessResponse:
        nonlocal harness_called
        harness_called = True
        return HarnessResponse(status="completed")

    result = run_research_memo(
        user_prompt="Build memo.",
        ticker="XOM",
        vault_root=tmp_path / "vault",
        run_id="no-required",
        adapters=SourceAdapters(fake_sec, failing_provider, failing_provider, failing_provider),
        run_harness_fn=fake_harness,
    )

    assert result.status == "failed"
    assert harness_called is False
    assert any("No required source" in error for error in result.errors)
    assert Path(result.manifest_path).exists()
    assert Path(result.status_path).exists()


def test_run_research_memo_with_research_state_applies_proposals(tmp_path):
    manual = tmp_path / "manual.md"
    manual.write_text("# Manual packet\n\nGuyana production growth notes.", encoding="utf-8")
    final_source = tmp_path / "harness_final.md"
    final_source.write_text("# Final memo\n\nUse S1.", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_sec(*args, **kwargs):
        return []

    def failing_provider(*args, **kwargs):
        raise RuntimeError("offline")

    def fake_harness(request: HarnessRequest) -> HarnessResponse:
        captured["request"] = request
        proposal_path = tmp_path / "vault" / "companies" / "XOM" / "research" / "memos" / "state-run" / "proposals.jsonl"
        proposal_path.write_text(
            json.dumps(
                {
                    "proposal_id": "p-state-1",
                    "type": "propose_section_update",
                    "section_key": "guyana_growth_engine",
                    "action": "create",
                    "body_markdown": "Guyana is now tracked as a durable growth section [S1].",
                    "evidence_keys": ["S1"],
                    "rationale": "Manual packet introduced a durable Guyana growth topic.",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return HarnessResponse(
            status="completed",
            stop_reason="done",
            run_root=str(tmp_path / "run"),
            root_agent_path=str(tmp_path / "run" / "agents" / "root"),
            final_report_path=str(final_source),
        )

    result = run_research_memo(
        user_prompt="Build an investment memo for XOM.",
        ticker="xom",
        company_name="Exxon Mobil",
        manual_files=[str(manual)],
        vault_root=tmp_path / "vault",
        run_id="state-run",
        adapters=SourceAdapters(fake_sec, failing_provider, failing_provider, failing_provider),
        run_harness_fn=fake_harness,
        enable_research_state=True,
    )

    request = captured["request"]
    assert isinstance(request, HarnessRequest)
    assert Path(request.context_files[0]).name == "research_state.md"
    assert Path(request.context_files[1]).name == "state_index.json"
    assert Path(request.context_files[2]).name == "evidence_index.json"
    assert Path(request.context_files[6]).name == "proposal_protocol.md"
    assert "proposals.jsonl" in request.user_prompt
    assert "proposal file must be valid JSONL" in request.user_prompt

    assert result.status == "succeeded"
    assert result.research_state_path is not None
    assert result.state_snapshot_path is not None
    assert result.proposals_path is not None
    assert result.accepted_changes_path is not None
    state_markdown = Path(result.research_state_path).read_text(encoding="utf-8")
    assert "<!-- key: guyana_growth_engine -->" in state_markdown
    assert "Guyana is now tracked as a durable growth section [S1]." in state_markdown
    assert Path(result.accepted_changes_path).exists()

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["research_state"]["enabled"] is True
    assert manifest["research_state"]["rolled_back"] is False
    assert Path(manifest["research_state"]["proposal_protocol_path"]).name == "proposal_protocol.md"
    protocol_text = Path(manifest["research_state"]["proposal_protocol_path"]).read_text(encoding="utf-8")
    expected_proposals_path = tmp_path / "vault" / "companies" / "XOM" / "research" / "memos" / "state-run" / "proposals.jsonl"
    assert str(expected_proposals_path) in protocol_text
    assert request.external_writable_files == [str(expected_proposals_path)]
    assert manifest["research_state"]["apply_counts"]["applied"] == 1
    assert manifest["research_state"]["integrity_errors"] == []


def test_run_research_memo_accepts_direct_external_proposal_path(tmp_path):
    manual = tmp_path / "manual.md"
    manual.write_text("# Manual packet\n\nGuyana production growth notes.", encoding="utf-8")
    final_source = tmp_path / "harness_final.md"
    final_source.write_text("# Final memo\n\nUse S1.", encoding="utf-8")

    def fake_sec(*args, **kwargs):
        return []

    def failing_provider(*args, **kwargs):
        raise RuntimeError("offline")

    def fake_harness(request: HarnessRequest) -> HarnessResponse:
        assert len(request.external_writable_files) == 1
        proposal_path = Path(request.external_writable_files[0])
        assert proposal_path.name == "proposals.jsonl"
        assert proposal_path.parent.name == "direct-proposals"
        proposal_path.write_text(
            json.dumps(
                {
                    "proposal_id": "p-direct",
                    "type": "propose_section_update",
                    "section_key": "guyana_growth_engine",
                    "action": "create",
                    "body_markdown": "Direct external proposal path updated durable state [S1].",
                    "evidence_keys": ["S1"],
                    "rationale": "Harness agents can write the approved durable proposal file directly.",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return HarnessResponse(
            status="completed",
            stop_reason="done",
            run_root=str(tmp_path / "run"),
            root_agent_path=str(tmp_path / "run" / "agents" / "root"),
            final_report_path=str(final_source),
        )

    result = run_research_memo(
        user_prompt="Build an investment memo for XOM.",
        ticker="xom",
        company_name="Exxon Mobil",
        manual_files=[str(manual)],
        vault_root=tmp_path / "vault",
        run_id="direct-proposals",
        adapters=SourceAdapters(fake_sec, failing_provider, failing_provider, failing_provider),
        run_harness_fn=fake_harness,
        enable_research_state=True,
    )

    assert result.status == "succeeded"
    assert Path(result.proposals_path).read_text(encoding="utf-8").strip()
    assert "Direct external proposal path updated" in Path(result.research_state_path).read_text(encoding="utf-8")


def test_run_research_memo_rolls_back_research_state_on_malformed_proposals(tmp_path):
    manual = tmp_path / "manual.md"
    manual.write_text("# Manual packet\n\nGuyana production growth notes.", encoding="utf-8")
    final_source = tmp_path / "harness_final.md"
    final_source.write_text("# Final memo\n\nUse S1.", encoding="utf-8")

    def fake_sec(*args, **kwargs):
        return []

    def failing_provider(*args, **kwargs):
        raise RuntimeError("offline")

    def fake_harness(request: HarnessRequest) -> HarnessResponse:
        proposal_path = tmp_path / "vault" / "companies" / "XOM" / "research" / "memos" / "bad-proposals" / "proposals.jsonl"
        proposal_path.write_text("not json\n", encoding="utf-8")
        return HarnessResponse(
            status="completed",
            stop_reason="done",
            run_root=str(tmp_path / "run"),
            root_agent_path=str(tmp_path / "run" / "agents" / "root"),
            final_report_path=str(final_source),
        )

    result = run_research_memo(
        user_prompt="Build an investment memo for XOM.",
        ticker="xom",
        company_name="Exxon Mobil",
        manual_files=[str(manual)],
        vault_root=tmp_path / "vault",
        run_id="bad-proposals",
        adapters=SourceAdapters(fake_sec, failing_provider, failing_provider, failing_provider),
        run_harness_fn=fake_harness,
        enable_research_state=True,
    )

    assert result.status == "failed"
    assert any("proposal apply failed" in error for error in result.errors)
    assert any("rolled back" in error for error in result.errors)
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["research_state"]["rolled_back"] is True
    assert manifest["research_state"]["apply_counts"] == {"applied": 0, "rejected": 0, "revised": 0}


def test_run_research_memo_rolls_back_research_state_on_integrity_failure(tmp_path, monkeypatch):
    manual = tmp_path / "manual.md"
    manual.write_text("# Manual packet\n\nGuyana production growth notes.", encoding="utf-8")
    final_source = tmp_path / "harness_final.md"
    final_source.write_text("# Final memo\n\nUse S1.", encoding="utf-8")

    def fake_sec(*args, **kwargs):
        return []

    def failing_provider(*args, **kwargs):
        raise RuntimeError("offline")

    def fake_harness(request: HarnessRequest) -> HarnessResponse:
        proposal_path = tmp_path / "vault" / "companies" / "XOM" / "research" / "memos" / "rollback-run" / "proposals.jsonl"
        proposal_path.write_text(
            json.dumps(
                {
                    "proposal_id": "p-state-rollback",
                    "type": "propose_section_update",
                    "section_key": "guyana_growth_engine",
                    "action": "create",
                    "body_markdown": "This update should be rolled back after integrity failure [S1].",
                    "evidence_keys": ["S1"],
                    "rationale": "Manual packet introduced a durable Guyana growth topic.",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return HarnessResponse(
            status="completed",
            stop_reason="done",
            run_root=str(tmp_path / "run"),
            root_agent_path=str(tmp_path / "run" / "agents" / "root"),
            final_report_path=str(final_source),
        )

    monkeypatch.setattr("src.research_platform.memo_run.validate_state_integrity", lambda *args, **kwargs: ["forced integrity failure"])

    result = run_research_memo(
        user_prompt="Build an investment memo for XOM.",
        ticker="xom",
        company_name="Exxon Mobil",
        manual_files=[str(manual)],
        vault_root=tmp_path / "vault",
        run_id="rollback-run",
        adapters=SourceAdapters(fake_sec, failing_provider, failing_provider, failing_provider),
        run_harness_fn=fake_harness,
        enable_research_state=True,
    )

    assert result.status == "failed"
    assert any("rolled back" in error for error in result.errors)
    paths = state_paths(tmp_path / "vault", "XOM")
    state_markdown = paths.research_state.read_text(encoding="utf-8")
    snapshot_markdown = (Path(result.state_snapshot_path) / "research_state.md").read_text(encoding="utf-8")
    assert state_markdown == snapshot_markdown
    assert "This update should be rolled back" not in state_markdown

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["research_state"]["rolled_back"] is True
    assert manifest["research_state"]["integrity_errors"] == ["forced integrity failure"]



def test_run_research_memo_v33_commits_staged_direct_edits(tmp_path):
    manual = tmp_path / "manual.md"
    manual.write_text("# Manual packet\n\nGuyana production growth notes.", encoding="utf-8")
    final_source = tmp_path / "harness_final.md"
    final_source.write_text("# Final memo\n\nUse S1.", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_sec(*args, **kwargs):
        return []

    def failing_provider(*args, **kwargs):
        raise RuntimeError("offline")

    def fake_harness(request: HarnessRequest) -> HarnessResponse:
        captured["request"] = request
        writable = {Path(path).name: Path(path) for path in request.external_writable_files}
        assert set(writable) == {"research_state.md", "open_questions.json", "conflicts.json", "valuation_snapshot.json"}
        assert all("state_stage" in str(path) for path in writable.values())
        assert "direct-edit protocol (v3.3)" in request.user_prompt
        markdown = writable["research_state.md"].read_text(encoding="utf-8")
        markdown += "\n## Guyana growth engine\n<!-- key: guyana_growth_engine -->\n\nGuyana is now tracked directly in staged durable state [S1].\n"
        writable["research_state.md"].write_text(markdown, encoding="utf-8")
        return HarnessResponse(
            status="completed",
            stop_reason="done",
            run_root=str(tmp_path / "run"),
            root_agent_path=str(tmp_path / "run" / "agents" / "root"),
            final_report_path=str(final_source),
        )

    result = run_research_memo(
        user_prompt="Build an investment memo for XOM.",
        ticker="xom",
        company_name="Exxon Mobil",
        manual_files=[str(manual)],
        vault_root=tmp_path / "vault",
        run_id="v33-direct",
        adapters=SourceAdapters(fake_sec, failing_provider, failing_provider, failing_provider),
        run_harness_fn=fake_harness,
        enable_research_state_v33=True,
    )

    assert result.status == "succeeded"
    request = captured["request"]
    assert isinstance(request, HarnessRequest)
    assert Path(request.context_files[0]).name == "research_state.md"
    assert "state_stage" in request.context_files[0]
    assert result.research_state_path is not None
    state_markdown = Path(result.research_state_path).read_text(encoding="utf-8")
    assert "Guyana is now tracked directly in staged durable state [S1]." in state_markdown
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    research_state = manifest["research_state"]
    assert research_state["mode"] == "v33_staged_direct_edit"
    assert research_state["rolled_back"] is False
    assert research_state["validation_report"]["ok"] is True
    assert research_state["commit"]["committed"] is True
    assert Path(research_state["state_validation_report_path"]).exists()


def test_run_research_memo_reuses_state_and_appends_evidence_across_runs(tmp_path):
    manual_one = tmp_path / "manual-one.md"
    manual_one.write_text("# Manual packet one\n\nGuyana growth engine.", encoding="utf-8")
    manual_two = tmp_path / "manual-two.md"
    manual_two.write_text("# Manual packet two\n\nPermian decline risk.", encoding="utf-8")
    final_source = tmp_path / "harness_final.md"
    final_source.write_text("# Final memo\n\nUse durable state.", encoding="utf-8")
    def fake_sec(*args, **kwargs):
        return []

    def failing_provider(*args, **kwargs):
        raise RuntimeError("offline")

    def fake_harness(request: HarnessRequest) -> HarnessResponse:
        proposal_path = tmp_path / "vault" / "companies" / "XOM" / "research" / "memos" / request.run_id / "proposals.jsonl"
        if request.run_id == "carryover-one":
            proposal = {
                "proposal_id": "p-carryover-1",
                "type": "propose_section_update",
                "section_key": "guyana_growth_engine",
                "action": "create",
                "body_markdown": "Guyana is tracked as the first durable growth topic [S1].",
                "evidence_keys": ["S1"],
                "rationale": "Run one introduced the Guyana growth topic.",
            }
        else:
            assert "Guyana is tracked as the first durable growth topic [S1]." in Path(request.context_files[0]).read_text(encoding="utf-8")
            proposal = {
                "proposal_id": "p-carryover-2",
                "type": "propose_section_update",
                "section_key": "permian_decline_risk",
                "action": "create",
                "body_markdown": "Permian decline risk is tracked as a second durable topic [S2].",
                "evidence_keys": ["S2"],
                "rationale": "Run two introduced a distinct durable Permian risk topic.",
            }
        proposal_path.write_text(json.dumps(proposal, ensure_ascii=False) + "\n", encoding="utf-8")
        return HarnessResponse(
            status="completed",
            stop_reason="done",
            run_root=str(tmp_path / "run" / request.run_id),
            root_agent_path=str(tmp_path / "run" / request.run_id / "agents" / "root"),
            final_report_path=str(final_source),
        )

    adapters = SourceAdapters(fake_sec, failing_provider, failing_provider, failing_provider)
    result_one = run_research_memo(
        user_prompt="Build an investment memo for XOM.",
        ticker="xom",
        company_name="Exxon Mobil",
        manual_files=[str(manual_one)],
        vault_root=tmp_path / "vault",
        run_id="carryover-one",
        adapters=adapters,
        run_harness_fn=fake_harness,
        enable_research_state=True,
    )
    result_two = run_research_memo(
        user_prompt="Build an updated investment memo for XOM.",
        ticker="xom",
        company_name="Exxon Mobil",
        manual_files=[str(manual_two)],
        vault_root=tmp_path / "vault",
        run_id="carryover-two",
        adapters=adapters,
        run_harness_fn=fake_harness,
        enable_research_state=True,
    )

    assert result_one.status == "succeeded"
    assert result_two.status == "succeeded"
    paths = state_paths(tmp_path / "vault", "XOM")
    state_markdown = paths.research_state.read_text(encoding="utf-8")
    assert "Guyana is tracked as the first durable growth topic [S1]." in state_markdown
    assert "Permian decline risk is tracked as a second durable topic [S2]." in state_markdown

    evidence_index = json.loads(paths.evidence_index.read_text(encoding="utf-8"))
    assert sorted(evidence_index["entries"]) == ["S1", "S2"]
    assert evidence_index["entries"]["S1"]["display_title"] == "manual-one"
    assert evidence_index["entries"]["S2"]["display_title"] == "manual-two"

    diff_records = [json.loads(line) for line in paths.diff_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    accepted_records = [record for record in diff_records if record["decision"]["decision"] == "accepted"]
    assert [record["run_id"] for record in accepted_records] == ["carryover-one", "carryover-two"]

    snapshot_one = Path(result_one.state_snapshot_path) / "research_state.md"
    snapshot_two = Path(result_two.state_snapshot_path) / "research_state.md"
    assert "Guyana is tracked" not in snapshot_one.read_text(encoding="utf-8")
    snapshot_two_text = snapshot_two.read_text(encoding="utf-8")
    assert "Guyana is tracked as the first durable growth topic [S1]." in snapshot_two_text
    assert "Permian decline risk" not in snapshot_two_text
