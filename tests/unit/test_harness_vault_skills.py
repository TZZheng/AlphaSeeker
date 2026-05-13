from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.skills.vault import (
    vault_extract_company_records_skill,
    vault_ingest_document_skill,
    vault_synthesize_research_state_skill,
    vault_update_company_wiki_skill,
)
from src.harness.types import HarnessRequest, HarnessState


def test_vault_pack_is_registered_and_validated():
    request = HarnessRequest(user_prompt="test", available_skill_packs=["vault"])
    registry = build_skill_registry()
    skills = get_skills_for_packs(registry, request.available_skill_packs or [])

    names = {skill.name for skill in skills}
    assert "vault_ingest_document" in names
    assert "vault_get_company_context" in names
    assert "vault_extract_company_records" in names
    assert "vault_update_company_wiki" in names
    assert "vault_synthesize_research_state" in names
    assert "vault_onboard_company" in names


class _FakeSynthesisLLM:
    def invoke(self, _messages):
        return SimpleNamespace(
            content='{"business_summary":"Business from source.","key_metrics":"Metric from source.","guidance":"Not found in provided sources.","risks":"Risk from source.","thesis":"Takeaway from source.","open_questions":["What should we verify next?"],"citations":[{"claim":"Business from source","doc_id":"doc_missing","quote":"Official company note"}]}'
        )


def test_vault_ingest_and_wiki_skills_use_default_vault(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "xom_note.md"
    source.write_text("# XOM\n\nOfficial company note.", encoding="utf-8")
    state = HarnessState(request=HarnessRequest(user_prompt="test"), run_id="run-test")

    ingest_result = vault_ingest_document_skill(
        {"path": str(source), "ticker": "XOM", "source_type": "sec", "source_grade": "A"},
        state,
    )
    extract_result = vault_extract_company_records_skill({"ticker": "XOM"}, state)
    wiki_result = vault_update_company_wiki_skill({"ticker": "XOM"}, state)

    assert ingest_result.status == "ok"
    assert extract_result.status == "ok"
    assert extract_result.details["counts"]["facts"] == 1
    assert wiki_result.status == "ok"
    assert Path("data/research_vault/vault.sqlite").exists()
    assert Path("data/research_vault/companies/XOM/wiki.md").exists()
    assert "[[source_index]]" in (wiki_result.output_text or "")


def test_vault_synthesis_skill_uses_default_vault(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.vault.synthesis.get_llm", lambda _model: _FakeSynthesisLLM())
    source = tmp_path / "xom_note.md"
    source.write_text("# XOM\n\nOfficial company note.", encoding="utf-8")
    state = HarnessState(request=HarnessRequest(user_prompt="test"), run_id="run-test")

    ingest_result = vault_ingest_document_skill({"path": str(source), "ticker": "XOM", "source_type": "manual_note", "source_grade": "A"}, state)
    synthesis_result = vault_synthesize_research_state_skill({"ticker": "XOM", "model_name": "fake-model"}, state)

    assert ingest_result.status == "ok"
    assert synthesis_result.status == "ok"
    assert synthesis_result.details["question_count"] == 1
    assert Path("data/research_vault/companies/XOM/llm_research_state.md").exists()
    assert "## Business summary" in (synthesis_result.output_text or "")
