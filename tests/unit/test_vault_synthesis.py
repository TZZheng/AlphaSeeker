from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from src.vault.ingest import ingest_text
from src.vault.store import VaultStore
from src.vault.synthesis import build_source_bundle, parse_synthesis_json, synthesize_company_research_state


class _FakeLLM:
    def invoke(self, messages):
        prompt = messages[-1].content
        assert "### SOURCE doc_" in prompt
        assert "Exxon Mobil explores for and produces crude oil" in prompt
        return SimpleNamespace(
            content="""
{
  "business_summary": "Exxon Mobil explores for and produces crude oil.",
  "key_metrics": "2025 revenue was $10 billion in the provided source.",
  "guidance": "Management plans disciplined capex.",
  "risks": "Commodity price volatility is a key risk.",
  "thesis": "The starting thesis is integrated energy exposure with capital discipline.",
  "open_questions": ["How durable is capex discipline through the cycle?"],
  "citations": [
    {"claim": "Exxon Mobil explores for and produces crude oil", "doc_id": "DOC_ID_PLACEHOLDER", "quote": "Exxon Mobil explores for and produces crude oil"}
  ]
}
""".replace("DOC_ID_PLACEHOLDER", prompt.split("### SOURCE ", 1)[1].split("\n", 1)[0])
        )


def test_build_source_bundle_reads_vault_documents(tmp_path):
    root = tmp_path / "research_vault"
    result = ingest_text(
        "Exxon Mobil explores for and produces crude oil. " * 20,
        ticker="XOM",
        title="XOM source note",
        source_type="manual_note",
        source_grade="A",
        root=root,
    )

    bundle = build_source_bundle("xom", root=root, doc_char_limit=80, bundle_char_limit=120)

    assert len(bundle) == 1
    assert bundle[0].doc_id == result["doc_id"]
    assert bundle[0].title == "XOM source note"
    assert "Exxon Mobil explores" in bundle[0].excerpt
    assert "truncated" in bundle[0].excerpt


def test_parse_synthesis_json_strips_fences_and_normalizes_questions():
    parsed = parse_synthesis_json(
        """```json
{
  "business_summary": "Business",
  "key_metrics": "Metrics",
  "guidance": "Guidance",
  "risks": "Risks",
  "thesis": "Thesis",
  "open_questions": [" Q1 ", ""],
  "citations": [{"claim": "C", "doc_id": "doc_1", "quote": "Q"}]
}
```"""
    )

    assert parsed["open_questions"] == ["Q1"]
    assert parsed["citations"] == [{"claim": "C", "doc_id": "doc_1", "quote": "Q"}]


def test_synthesize_company_research_state_writes_wiki_and_questions(monkeypatch, tmp_path):
    root = tmp_path / "research_vault"
    ingest_text(
        "Exxon Mobil explores for and produces crude oil. 2025 revenue was $10 billion. Commodity price volatility is a key risk.",
        ticker="XOM",
        title="XOM analyst packet",
        source_type="manual_note",
        source_grade="A",
        root=root,
    )
    monkeypatch.setattr("src.vault.synthesis.get_llm", lambda _model: _FakeLLM())

    result = synthesize_company_research_state("xom", root=root, model_name="fake-model")

    wiki_path = Path(result["wiki_path"])
    assert wiki_path.name == "llm_research_state.md"
    text = wiki_path.read_text(encoding="utf-8")
    assert "## Business summary" in text
    assert "Exxon Mobil explores for and produces crude oil" in text
    assert "## Citations" in text
    assert "XOM analyst packet" in text
    assert Path(result["raw_response_path"]).exists()

    questions = VaultStore(root).company_context("XOM")["questions"]
    assert [question["question"] for question in questions] == ["How durable is capex discipline through the cycle?"]
    assert (root / "companies" / "XOM" / "question_list.md").exists()
