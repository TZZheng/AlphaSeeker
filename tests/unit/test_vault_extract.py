from __future__ import annotations

from src.vault.extract import extract_company_records, parse_key_ratio_metrics
from src.vault.ingest import ingest_text
from src.vault.store import VaultStore


FINANCIALS_MD = """# Financial Analysis for XOM

## Key Ratios
- **Current Price**: 150.63
- **Market Cap**: 624353411072
- **Enterprise Value**: 670194401280
- **Trailing P/E**: 25.40135
- **Forward P/E**: 14.445208
- **EV/EBITDA**: 11.966
- **Debt/Equity**: 18.261
- **ROE**: 0.09873
- **Revenue Growth (YoY)**: N/A
- **Total Revenue (TTM)**: 326008012800
- **Operating Cash Flow**: 47722000384
- **Free Cash Flow**: 11630499840
- **Fiscal Year End**: 1767139200

## ESG Data
ESG data not available.
"""


def test_parse_key_ratio_metrics_keeps_supported_non_missing_values():
    metrics = parse_key_ratio_metrics(FINANCIALS_MD)

    names = [metric["metric_name"] for metric in metrics]
    assert "Current Price" in names
    assert "Market Cap" in names
    assert "Revenue Growth (YoY)" not in names
    assert "Fiscal Year End" not in names
    assert {metric["metric_name"]: metric["unit"] for metric in metrics}["EV/EBITDA"] == "x"
    assert {metric["metric_name"]: metric["period"] for metric in metrics}["Free Cash Flow"] == "TTM"


def test_extract_company_records_populates_metrics_facts_questions_idempotently(tmp_path):
    root = tmp_path / "research_vault"
    store = VaultStore(root)
    store.upsert_company("XOM", name="Exxon Mobil")
    sec_doc = ingest_text(
        """FORM 10-K official filing text
Exxon Mobil Corporation's principal business involves exploration for, and production of, crude oil and natural gas; manufacture, trade, transport and sale of crude oil, natural gas, petroleum products, petrochemicals, and a wide variety of specialty products.
Operating data and industry segment information for the Corporation are contained in the Financial Section of this report under the following: “Management's Discussion and Analysis of Financial Condition and Results of Operations: Business Results” and Note 3.
The oil, gas, and petrochemical businesses are fundamentally commodity businesses. This means ExxonMobil’s operations and earnings may be significantly affected by changes in oil, gas, and petrochemical prices and by changes in margins on refined products.
""",
        ticker="XOM",
        source_type="sec",
        title="XOM 10-K 2026-02-25",
        source_grade="A",
        url="https://www.sec.gov/xom-10k",
        published_at="2026-02-25",
        metadata={"form_type": "10-K", "filing_date": "2026-02-25"},
        root=root,
        store=store,
    )
    financials_doc = ingest_text(
        FINANCIALS_MD,
        ticker="XOM",
        source_type="derived_financials",
        title="XOM derived financial metrics",
        source_grade="B",
        root=root,
        store=store,
    )

    first = extract_company_records("xom", root=root, store=store)
    second = extract_company_records("XOM", root=root, store=store)
    context = store.company_context("XOM", limit=50)

    assert first["counts"] == second["counts"]
    assert first["counts"]["metrics"] == 11
    assert first["counts"]["facts"] == 4
    assert first["counts"]["questions"] == 2
    assert len(context["metrics"]) == 11
    assert len(context["facts"]) == 4
    assert len(context["questions"]) == 2
    assert {metric["metric_name"] for metric in context["metrics"]} >= {"Current Price", "Free Cash Flow"}
    assert {metric["source_doc_id"] for metric in context["metrics"]} == {financials_doc["doc_id"]}
    assert {fact["source_doc_id"] for fact in context["facts"]} == {sec_doc["doc_id"]}
    sections = {fact["section"] for fact in context["facts"]}
    assert {"SEC source registry", "Business overview", "Management commentary / official commentary", "Key risks from official filings"}.issubset(sections)
    assert any("A-grade SEC 10-K filing" in fact["statement"] for fact in context["facts"])
    assert any("principal business involves" in fact["statement"] for fact in context["facts"])
