from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.research_platform.lingtai_tools import source_data


def test_yfinance_snapshot_uses_fast_info_and_optional_profile(monkeypatch):
    monkeypatch.setattr(
        source_data,
        "_ticker_fast_info",
        lambda ticker: {"lastPrice": 123.45, "marketCap": 1000},
    )
    monkeypatch.setattr(source_data, "_ticker_info_subset", lambda ticker: {"longName": "Example Inc."})

    payload = source_data.yfinance_snapshot("tsla")

    assert payload["status"] == "ok"
    assert payload["ticker"] == "TSLA"
    assert payload["fast_info"]["lastPrice"] == 123.45
    assert payload["info_subset"]["longName"] == "Example Inc."


def test_yfinance_history_saves_csv(monkeypatch, tmp_path):
    idx = pd.to_datetime(["2026-01-02", "2026-01-03"])
    hist = pd.DataFrame({"Open": [1.0, 2.0], "Close": [1.5, 2.5], "Volume": [10, 20]}, index=idx)
    hist.index.name = "Date"
    monkeypatch.setattr(source_data, "cached_retry_call", lambda _name, _key, loader, **_kwargs: loader())

    class FakeTicker:
        def __init__(self, ticker: str) -> None:
            self.ticker = ticker

        def history(self, **kwargs):
            assert kwargs["period"] == "5d"
            assert kwargs["interval"] == "1d"
            return hist

    monkeypatch.setattr(source_data.yf, "Ticker", FakeTicker)

    payload = source_data.yfinance_history("tsla", period="5d", output_dir=str(tmp_path))

    assert payload["rows"] == 2
    csv_path = Path(payload["csv_path"])
    assert csv_path.exists()
    assert csv_path.name == "yfinance_TSLA_5d_1d.csv"
    assert "Close" in csv_path.read_text()


def test_sec_find_company_resolves_ticker(monkeypatch):
    monkeypatch.setattr(
        source_data,
        "_company_tickers",
        lambda: [{"ticker": "TSLA", "cik_str": 1318605, "title": "Tesla, Inc."}],
    )

    payload = source_data.sec_find_company("tsla")

    assert payload["cik"] == "0001318605"
    assert payload["title"] == "Tesla, Inc."


def test_sec_recent_filings_builds_archive_urls(monkeypatch):
    monkeypatch.setattr(
        source_data,
        "sec_find_company",
        lambda ticker: {"ticker": "TSLA", "cik": "0001318605", "title": "Tesla, Inc."},
    )
    monkeypatch.setattr(
        source_data,
        "_sec_submissions",
        lambda cik: {
            "cik": "1318605",
            "filings": {
                "recent": {
                    "form": ["10-Q", "8-K", "4"],
                    "accessionNumber": ["0001628280-26-026673", "0001628280-26-026551", "x"],
                    "filingDate": ["2026-04-23", "2026-04-22", "2026-04-01"],
                    "reportDate": ["2026-03-31", "2026-04-22", ""],
                    "primaryDocument": ["tsla-20260331.htm", "tsla-20260422.htm", "x.htm"],
                    "primaryDocDescription": ["10-Q", "8-K", "FORM 4"],
                }
            },
        },
    )

    payload = source_data.sec_recent_filings("TSLA", form_types=["10-Q", "8-K"], limit=10)

    assert [f["form"] for f in payload["filings"]] == ["10-Q", "8-K"]
    assert payload["filings"][0]["url"] == (
        "https://www.sec.gov/Archives/edgar/data/1318605/000162828026026673/tsla-20260331.htm"
    )


def test_sec_companyfacts_snapshot_saves_full_and_compact(monkeypatch, tmp_path):
    monkeypatch.setattr(
        source_data,
        "sec_find_company",
        lambda ticker: {"ticker": "TSLA", "cik": "0001318605", "title": "Tesla, Inc."},
    )
    monkeypatch.setattr(
        source_data,
        "_sec_companyfacts",
        lambda cik: {
            "entityName": "Tesla, Inc.",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "label": "Revenue",
                        "description": "Revenue desc",
                        "units": {"USD": [{"fy": 2025, "val": 94827000000}]},
                    }
                }
            },
        },
    )

    payload = source_data.sec_companyfacts_snapshot("TSLA", output_dir=str(tmp_path))

    assert Path(payload["companyfacts_path"]).exists()
    assert Path(payload["snapshot_path"]).exists()
    assert payload["selected_metrics"]["Revenues"]["recent"][0]["val"] == 94827000000


def test_invalid_ticker_rejected():
    with pytest.raises(source_data.SourceDataError):
        source_data.yfinance_snapshot("../TSLA")
