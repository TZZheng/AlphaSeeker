from __future__ import annotations

from src.vault.extract import compare_a_b_metrics, extract_company_records, parse_capital_return_metrics, parse_key_ratio_metrics, parse_sec_companyfacts_capital_return_metrics
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

### Annual Cash Flow Statement
|                                                |   2025-12-31 00:00:00 |   2024-12-31 00:00:00 |
|:-----------------------------------------------|----------------------:|----------------------:|
| Free Cash Flow                                 |            2.3612e+10 |            3.0716e+10 |
| Repurchase Of Capital Stock                    |           -2.0273e+10 |           -1.9629e+10 |
| Capital Expenditure                            |           -2.8358e+10 |           -2.4306e+10 |
| Cash Dividends Paid                            |           -1.7231e+10 |           -1.6704e+10 |
| Operating Cash Flow                            |            5.197e+10  |            5.5022e+10 |

### Quarterly Financial Statements
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


def test_parse_capital_return_metrics_reads_latest_annual_cash_flow_rows():
    metrics = parse_capital_return_metrics(FINANCIALS_MD)

    by_name = {metric["metric_name"]: metric for metric in metrics}
    assert by_name["Capital Expenditures"]["period"] == "FY2025"
    assert by_name["Capital Expenditures"]["value"] == "-28358000000"
    assert by_name["Share Repurchases"]["value"] == "-20273000000"
    assert by_name["Cash Dividends Paid"]["value"] == "-17231000000"
    assert by_name["Annual Operating Cash Flow"]["value"] == "51970000000"


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
    assert first["counts"]["metrics"] == 16
    assert first["counts"]["facts"] == 4
    assert first["counts"]["questions"] == 4
    assert len(context["metrics"]) == 16
    assert len(context["facts"]) == 4
    assert len(context["questions"]) == 4
    assert {metric["metric_name"] for metric in context["metrics"]} >= {"Current Price", "Free Cash Flow", "Capital Expenditures", "Share Repurchases"}
    assert {metric["source_doc_id"] for metric in context["metrics"]} == {financials_doc["doc_id"]}
    assert {fact["source_doc_id"] for fact in context["facts"]} == {sec_doc["doc_id"]}
    sections = {fact["section"] for fact in context["facts"]}
    assert {"SEC source registry", "Business overview", "Management commentary / official commentary", "Key risks from official filings"}.issubset(sections)
    assert any("A-grade SEC 10-K filing" in fact["statement"] for fact in context["facts"])
    assert any("principal business involves" in fact["statement"] for fact in context["facts"])
    questions = {question["question"]: question for question in context["questions"]}
    assert any("latest annual capital-return metrics" in question for question in questions)
    assert any("Capital Expenditures" in question and "Share Repurchases" in question for question in questions)
    assert any("valuation support metrics" in question for question in questions)
    assert any("Enterprise Value" in question and "EV/EBITDA" in question for question in questions)


def test_parse_sec_companyfacts_capital_return_metrics_derives_a_grade_fcf():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": [{"val": 51970000000, "fy": 2025, "fp": "FY", "form": "10-K", "end": "2025-12-31", "filed": "2026-02-18", "accn": "0000034088-26-000045"}]}
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {"USD": [{"val": 28358000000, "fy": 2025, "fp": "FY", "form": "10-K", "end": "2025-12-31", "filed": "2026-02-18", "accn": "0000034088-26-000045"}]}
                },
                "PaymentsForRepurchaseOfCommonStock": {
                    "units": {"USD": [{"val": 20273000000, "fy": 2025, "fp": "FY", "form": "10-K", "end": "2025-12-31", "filed": "2026-02-18", "accn": "0000034088-26-000045"}]}
                },
                "PaymentsOfDividendsCommonStock": {
                    "units": {"USD": [{"val": 17231000000, "fy": 2025, "fp": "FY", "form": "10-K", "end": "2025-12-31", "filed": "2026-02-18", "accn": "0000034088-26-000045"}]}
                },
            }
        }
    }

    metrics = parse_sec_companyfacts_capital_return_metrics(companyfacts)

    by_name = {metric["metric_name"]: metric for metric in metrics}
    assert by_name["Annual Operating Cash Flow"]["value"] == "51970000000"
    assert by_name["Capital Expenditures"]["value"] == "-28358000000"
    assert by_name["Share Repurchases"]["value"] == "-20273000000"
    assert by_name["Cash Dividends Paid"]["value"] == "-17231000000"
    assert by_name["Annual Free Cash Flow"]["value"] == "23612000000"
    assert by_name["Annual Free Cash Flow"]["period"] == "FY2025"


def test_extract_company_records_adds_matching_a_grade_companyfacts_metrics(monkeypatch, tmp_path):
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
        title="XOM 10-K 2026-02-18",
        source_grade="A",
        url="https://www.sec.gov/Archives/edgar/data/34088/000003408826000045/xom-20251231.htm",
        published_at="2026-02-18",
        metadata={"form_type": "10-K", "filing_date": "2026-02-18", "url": "https://www.sec.gov/Archives/edgar/data/34088/000003408826000045/xom-20251231.htm"},
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
    fake_companyfacts = {
        "facts": {
            "us-gaap": {
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [{"val": 51970000000, "fy": 2025, "fp": "FY", "form": "10-K", "end": "2025-12-31", "filed": "2026-02-18", "accn": "0000034088-26-000045"}]}},
                "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [{"val": 28358000000, "fy": 2025, "fp": "FY", "form": "10-K", "end": "2025-12-31", "filed": "2026-02-18", "accn": "0000034088-26-000045"}]}},
                "PaymentsForRepurchaseOfCommonStock": {"units": {"USD": [{"val": 20273000000, "fy": 2025, "fp": "FY", "form": "10-K", "end": "2025-12-31", "filed": "2026-02-18", "accn": "0000034088-26-000045"}]}},
                "PaymentsOfDividendsCommonStock": {"units": {"USD": [{"val": 17231000000, "fy": 2025, "fp": "FY", "form": "10-K", "end": "2025-12-31", "filed": "2026-02-18", "accn": "0000034088-26-000045"}]}},
            }
        }
    }
    monkeypatch.setattr("src.vault.extract.fetch_sec_companyfacts", lambda cik: fake_companyfacts)

    result = extract_company_records("XOM", root=root, store=store)
    context = store.company_context("XOM", limit=100)

    assert result["counts"]["metrics"] == 21
    assert result["counts"]["conflicts"] == 0
    a_metrics = [metric for metric in context["metrics"] if metric["source_grade"] == "A" and metric["metric_name"] in {"Annual Free Cash Flow", "Capital Expenditures", "Share Repurchases"}]
    assert len(a_metrics) == 3
    assert {metric["source_doc_id"] for metric in a_metrics} == {sec_doc["doc_id"]}
    assert any(metric["source_grade"] == "B" and metric["source_doc_id"] == financials_doc["doc_id"] for metric in context["metrics"])
    assert context["conflicts"] == []


def test_compare_a_b_metrics_creates_idempotent_conflict_for_mismatch(tmp_path):
    store = VaultStore(tmp_path / "research_vault")
    metrics = [
        {"metric_id": "metric_a", "metric_name": "Capital Expenditures", "period": "FY2025", "value": "-28000000000", "source_grade": "A"},
        {"metric_id": "metric_b", "metric_name": "Capital Expenditures", "period": "FY2025", "value": "-28358000000", "source_grade": "B"},
    ]

    first = compare_a_b_metrics("XOM", metrics, store=store)
    second = compare_a_b_metrics("XOM", metrics, store=store)
    context = store.company_context("XOM")

    assert len(first) == 1
    assert len(second) == 1
    assert len(context["conflicts"]) == 1
    assert "Capital Expenditures differs" in context["conflicts"][0]["summary"]
    assert context["conflicts"][0]["left_ref"] == "metric_a"
    assert context["conflicts"][0]["right_ref"] == "metric_b"
