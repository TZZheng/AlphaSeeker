from __future__ import annotations

from pathlib import Path

from src.vault.onboard import onboard_company
from src.vault.store import VaultStore


FINANCIALS_MD = """# Financial Analysis for XOM

## Key Ratios
- **Current Price**: 150.63
- **Market Cap**: 624353411072
- **EV/EBITDA**: 11.966
- **Total Revenue (TTM)**: 326008012800
- **Free Cash Flow**: 11630499840

## ESG Data
ESG data not available.
"""


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
    assert result["extracted_records"]["counts"]["facts"] == 1
    assert result["extracted_records"]["counts"]["questions"] == 1
    context = VaultStore(root).company_context("XOM")
    assert context["documents"][0]["source_grade"] == "A"
    assert context["facts"][0]["section"] == "SEC source registry"


def test_onboard_company_extracts_derived_financial_metrics(monkeypatch, tmp_path):
    def fake_import_sec_filings(**kwargs):
        return []

    def fake_fetch_company_profile(ticker, output_dir=None):
        path = Path(output_dir) / f"{ticker}_profile.md"
        path.write_text("# Company Profile\n\nName: Exxon Mobil\n", encoding="utf-8")
        return str(path), {"company_name": "Exxon Mobil"}

    def fake_fetch_financial_metrics(ticker, output_dir=None):
        path = Path(output_dir) / f"{ticker}_financials.md"
        path.write_text(FINANCIALS_MD, encoding="utf-8")
        return str(path), {"source": "test"}

    monkeypatch.setattr("src.vault.onboard.import_sec_filings", fake_import_sec_filings)
    monkeypatch.setattr("src.vault.onboard.fetch_company_profile", fake_fetch_company_profile)
    monkeypatch.setattr("src.vault.onboard.fetch_financial_metrics", fake_fetch_financial_metrics)
    root = tmp_path / "research_vault"

    result = onboard_company(ticker="xom", company_name="Exxon Mobil", include_market_support=True, root=root)

    assert result["extracted_records"]["counts"]["metrics"] == 5
    assert result["extracted_records"]["counts"]["questions"] == 2
    context = VaultStore(root).company_context("XOM", limit=20)
    assert {metric["metric_name"] for metric in context["metrics"]} >= {"Current Price", "Free Cash Flow"}
    assert any("valuation support metrics" in question["question"] for question in context["questions"])
    wiki = Path(result["wiki_path"]).read_text(encoding="utf-8")
    assert "Current Price" in wiki
    assert "Free Cash Flow" in wiki
