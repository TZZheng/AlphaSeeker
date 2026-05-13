from __future__ import annotations

from pathlib import Path

from src.harness.types import HarnessResponse, HarnessRequest
from src.vault.memo_flow import build_vault_backed_prompt, run_vault_backed_memo


def test_build_vault_backed_prompt_references_context_file_without_inlining():
    prompt = build_vault_backed_prompt(
        "Build an investment memo for XOM.",
        ticker="xom",
        research_state_path="/tmp/vault/companies/XOM/llm_research_state.md",
    )

    assert prompt.startswith("Build an investment memo for XOM.")
    assert "llm_research_state.md" in prompt
    assert "publish/final.md" in prompt
    assert "/tmp/vault/companies/XOM" not in prompt


def test_run_vault_backed_memo_plumbs_research_state_as_context_file(monkeypatch, tmp_path):
    source = tmp_path / "xom_packet.md"
    source.write_text("XOM source packet", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_ingest(path, **kwargs):
        captured["ingest_path"] = path
        captured["ingest_kwargs"] = kwargs
        return {"doc_id": "doc_test"}

    def fake_synthesis(*args, **kwargs):
        wiki_path = tmp_path / "llm_research_state.md"
        wiki_path.write_text("# XOM research state", encoding="utf-8")
        captured["synthesis_args"] = args
        captured["synthesis_kwargs"] = kwargs
        return {"wiki_path": str(wiki_path)}

    def fake_run_harness(request: HarnessRequest):
        captured["request"] = request
        return HarnessResponse(status="completed", stop_reason="done", run_root="/tmp/run", root_agent_path="/tmp/run/agents/agent_root", final_report_path="/tmp/run/agents/agent_root/publish/final.md")

    monkeypatch.setattr("src.vault.memo_flow.ingest_file", fake_ingest)
    monkeypatch.setattr("src.vault.memo_flow.synthesize_company_research_state", fake_synthesis)
    monkeypatch.setattr("src.vault.memo_flow.run_harness", fake_run_harness)

    response = run_vault_backed_memo(
        user_prompt="Build an investment memo for XOM.",
        ticker="xom",
        company_name="Exxon Mobil",
        source_paths=[str(source)],
        root=tmp_path / "vault",
        synthesis_model="fake-model",
        run_id="memo-test",
        wall_clock_budget_seconds=123,
    )

    assert response.status == "completed"
    assert captured["ingest_path"] == str(source)
    request = captured["request"]
    assert isinstance(request, HarnessRequest)
    assert request.run_id == "memo-test"
    assert request.wall_clock_budget_seconds == 123
    assert request.context_files == [str(tmp_path / "llm_research_state.md")]
    assert request.available_skill_packs == ["core", "equity", "macro", "commodity", "vault"]
    assert "Build an investment memo for XOM." in request.user_prompt
    assert "llm_research_state.md" in request.user_prompt
    assert "# XOM research state" not in request.user_prompt
    assert captured["synthesis_args"] == ("XOM",)
    assert captured["synthesis_kwargs"]["model_name"] == "fake-model"


def test_harness_request_root_context_files_are_copied_and_listed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    context_file = tmp_path / "llm_research_state.md"
    context_file.write_text("# XOM research state", encoding="utf-8")

    from src.harness.artifacts import agent_workspace_paths, initialize_run_root
    from src.harness.prompt_builder import _render_runtime_snapshot
    from src.harness.registry import build_skill_registry
    from src.harness.runtime import _ensure_root_workspace

    request = HarnessRequest(user_prompt="Analyze XOM", run_id="context-root", context_files=[str(context_file)])
    run_root, root_agent_id = initialize_run_root(request)
    _ensure_root_workspace(request, run_root=str(run_root), root_agent_id=root_agent_id, registry_map=build_skill_registry())

    paths = agent_workspace_paths(run_root, root_agent_id)
    copied = paths["context_root"] / "llm_research_state.md"
    assert copied.exists()
    assert copied.read_text(encoding="utf-8") == "# XOM research state"
    snapshot = _render_runtime_snapshot(request=request, run_root=str(run_root), agent_id=root_agent_id)
    assert "llm_research_state.md" in snapshot
