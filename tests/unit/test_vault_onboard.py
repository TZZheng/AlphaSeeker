from __future__ import annotations

from pathlib import Path

from src.vault.onboard import onboard_company
from src.vault.store import VaultStore


def test_onboard_company_ties_sec_import_and_wiki_render(monkeypatch, tmp_path):
    def fake_import_sec_filings(**kwargs):
        from src.vault.ingest import ingest_text

        return [
            ingest_text(
                "Official 10-K text",
                ticker=kwargs["ticker"],
                source_type="sec",
                title="XOM 10-K",
                source_grade="A",
                url="https://www.sec.gov/xom-10k",
                published_at="2026-02-01",
                root=kwargs["root"],
                store=kwargs["store"],
            )
        ]

    monkeypatch.setattr("src.vault.onboard.import_sec_filings", fake_import_sec_filings)
    root = tmp_path / "research_vault"

    result = onboard_company(
        ticker="xom",
        company_name="Exxon Mobil",
        forms=["10-K"],
        max_filings=1,
        include_market_support=False,
        root=root,
    )

    assert result["ticker"] == "XOM"
    assert result["company"]["name"] == "Exxon Mobil"
    assert len(result["sec_documents"]) == 1
    assert Path(result["wiki_path"]).exists()
    assert Path(result["database_path"]).exists()
    context = VaultStore(root).company_context("XOM")
    assert context["documents"][0]["source_grade"] == "A"
