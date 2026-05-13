from __future__ import annotations

from pathlib import Path

from src.vault.sec_import import import_sec_filings
from src.vault.store import VaultStore


def test_import_sec_filings_mocks_fetch_and_stores_a_grade_documents(monkeypatch, tmp_path):
    def fake_search_and_read_filings(**kwargs):
        assert kwargs["ticker"] == "XOM"
        assert kwargs["form_types"] == ["10-K", "10-Q"]
        return [
            {
                "form_type": "10-K",
                "filing_date": "2026-02-01",
                "company": "Exxon Mobil Corp.",
                "url": "https://www.sec.gov/xom-10k",
                "text": "Official 10-K text",
            },
            {
                "form_type": "10-Q",
                "filing_date": "2026-05-01",
                "company": "Exxon Mobil Corp.",
                "url": "https://www.sec.gov/xom-10q",
                "text": "Official 10-Q text",
            },
        ]

    monkeypatch.setattr("src.vault.sec_import.search_and_read_filings", fake_search_and_read_filings)
    root = tmp_path / "research_vault"

    imported = import_sec_filings(
        ticker="xom",
        company_name="Exxon Mobil Corp.",
        form_types=["10-K", "10-Q"],
        max_filings=2,
        root=root,
    )

    assert len(imported) == 2
    context = VaultStore(root).company_context("XOM")
    assert context["company"]["name"] == "Exxon Mobil Corp."
    assert [doc["source_type"] for doc in context["documents"]] == ["sec", "sec"]
    assert {doc["source_grade"] for doc in context["documents"]} == {"A"}
    assert Path(imported[0]["extracted_path"]).read_text(encoding="utf-8") == "Official 10-K text"
