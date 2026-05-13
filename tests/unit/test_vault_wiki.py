from __future__ import annotations

from pathlib import Path

from src.vault.ingest import ingest_text
from src.vault.store import VaultStore
from src.vault.wiki import render_company_wiki


def test_render_company_wiki_creates_obsidian_pages(tmp_path):
    root = tmp_path / "research_vault"
    store = VaultStore(root)
    store.upsert_company("XOM", name="Exxon Mobil")
    doc = ingest_text("Official filing text", ticker="XOM", source_type="sec", source_grade="A", title="XOM 10-K", root=root)
    store.add_fact("XOM", "XOM operates integrated upstream and downstream assets.", section="Business overview", source_doc_id=doc["doc_id"], source_grade="A")
    store.add_fact("XOM", "Management discusses business results in MD&A.", section="Management commentary / official commentary", source_doc_id=doc["doc_id"], source_grade="A")
    store.add_fact("XOM", "XOM is exposed to commodity price risk.", section="Key risks from official filings", source_doc_id=doc["doc_id"], source_grade="A")
    store.add_metric("XOM", "Free cash flow", "11630499840", period="TTM", unit="USD", source_doc_id=doc["doc_id"], source_grade="A")
    store.add_metric("XOM", "Capital Expenditures", "-28358000000", period="FY2025", unit="USD", source_doc_id=doc["doc_id"], source_grade="B")
    store.add_question("XOM", "How durable is Guyana growth?", priority="high")
    store.add_conflict("XOM", "metric_mismatch", "FCF differs between market data and filing-derived estimate.")

    wiki_path = render_company_wiki("xom", root=root)

    wiki = wiki_path.read_text(encoding="utf-8")
    assert "# XOM — Exxon Mobil" in wiki
    assert "[[source_index]]" in wiki
    assert "[[question_list]]" in wiki
    assert "Free cash flow" in wiki
    assert "Capital Expenditures" in wiki
    assert "Management discusses business results" in wiki
    assert "commodity price risk" in wiki
    assert "Valuation-relevant metrics" in wiki
    assert "How durable is Guyana growth?" in wiki
    assert "FCF differs" in wiki
    assert (root / "companies" / "XOM" / "source_index.md").exists()
    assert (root / "companies" / "XOM" / "question_list.md").exists()
    assert (root / "companies" / "XOM" / "conflicts.md").exists()

    second_path = render_company_wiki("XOM", root=root)
    assert second_path == wiki_path
    assert list((root / "companies" / "XOM" / "versions").glob("wiki_*.md"))
