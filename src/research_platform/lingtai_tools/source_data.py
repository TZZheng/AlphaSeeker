"""Minimal market-data and SEC helpers for LingTai-native ticker teams."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from src.shared.reliability import cached_retry_call, request_bytes, request_json, request_text
from src.tools.equity.sec_filings import SEC_HEADERS, resolve_sec_primary_document_url


class SourceDataError(Exception):
    """Raised when a source-data helper cannot produce a usable result."""


def _normalize_ticker(ticker: str) -> str:
    symbol = ticker.strip().upper()
    if not symbol or not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,15}", symbol):
        raise SourceDataError(f"invalid ticker: {ticker!r}")
    return symbol


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "item"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _ticker_fast_info(ticker: str) -> dict[str, Any]:
    symbol = _normalize_ticker(ticker)

    def _load() -> dict[str, Any]:
        fast = yf.Ticker(symbol).fast_info
        keys = [
            "currency",
            "exchange",
            "quoteType",
            "timezone",
            "lastPrice",
            "previousClose",
            "regularMarketPreviousClose",
            "open",
            "dayHigh",
            "dayLow",
            "lastVolume",
            "tenDayAverageVolume",
            "threeMonthAverageVolume",
            "marketCap",
            "shares",
            "fiftyDayAverage",
            "twoHundredDayAverage",
            "yearHigh",
            "yearLow",
            "yearChange",
        ]
        out: dict[str, Any] = {}
        for key in keys:
            try:
                out[key] = _to_jsonable(fast.get(key))
            except Exception:
                out[key] = None
        return out

    return cached_retry_call(
        "lingtai_yfinance_fast_info",
        {"ticker": symbol},
        _load,
        ttl_seconds=300,
        attempts=3,
    )


def _ticker_info_subset(ticker: str) -> dict[str, Any]:
    symbol = _normalize_ticker(ticker)

    def _load() -> dict[str, Any]:
        info = yf.Ticker(symbol).info or {}
        keys = [
            "shortName",
            "longName",
            "symbol",
            "quoteType",
            "exchange",
            "currency",
            "sector",
            "industry",
            "website",
            "marketCap",
            "sharesOutstanding",
            "floatShares",
            "enterpriseValue",
            "trailingPE",
            "forwardPE",
            "priceToSalesTrailing12Months",
            "enterpriseToRevenue",
            "enterpriseToEbitda",
            "profitMargins",
            "grossMargins",
            "operatingMargins",
            "revenueGrowth",
            "earningsGrowth",
            "totalRevenue",
            "ebitda",
            "freeCashflow",
            "operatingCashflow",
            "totalCash",
            "totalDebt",
            "currentPrice",
            "targetMeanPrice",
            "recommendationMean",
            "recommendationKey",
            "numberOfAnalystOpinions",
            "longBusinessSummary",
        ]
        return {key: _to_jsonable(info.get(key)) for key in keys if key in info}

    return cached_retry_call(
        "lingtai_yfinance_info_subset",
        {"ticker": symbol},
        _load,
        ttl_seconds=1800,
        attempts=2,
    )


def yfinance_snapshot(ticker: str, *, include_profile: bool = True) -> dict[str, Any]:
    """Return a compact quote/valuation/profile snapshot from yfinance."""

    symbol = _normalize_ticker(ticker)
    result: dict[str, Any] = {
        "status": "ok",
        "source": "yfinance",
        "ticker": symbol,
        "fetched_at": _utc_now_iso(),
        "fast_info": _ticker_fast_info(symbol),
    }
    if include_profile:
        try:
            result["info_subset"] = _ticker_info_subset(symbol)
        except Exception as exc:
            result["info_subset_error"] = f"{type(exc).__name__}: {exc}"
    return result


def yfinance_history(
    ticker: str,
    *,
    period: str = "1y",
    interval: str = "1d",
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Fetch OHLCV history and optionally persist it as CSV."""

    symbol = _normalize_ticker(ticker)
    period = period.strip() or "1y"
    interval = interval.strip() or "1d"

    def _load() -> pd.DataFrame:
        return yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)

    hist = cached_retry_call(
        "lingtai_yfinance_history",
        {"ticker": symbol, "period": period, "interval": interval},
        _load,
        ttl_seconds=900,
        attempts=3,
    )
    if hist.empty:
        raise SourceDataError(f"no yfinance history for {symbol} period={period} interval={interval}")

    # Normalize index for stable JSON/CSV output.
    out = hist.reset_index()
    out.columns = [str(col).replace(" ", "_") for col in out.columns]
    if out.columns[0] in {"Date", "Datetime"}:
        out[out.columns[0]] = out[out.columns[0]].astype(str)

    payload: dict[str, Any] = {
        "status": "ok",
        "source": "yfinance",
        "ticker": symbol,
        "period": period,
        "interval": interval,
        "fetched_at": _utc_now_iso(),
        "rows": int(len(out)),
        "first_row": _to_jsonable(out.iloc[0].to_dict()),
        "last_row": _to_jsonable(out.iloc[-1].to_dict()),
    }
    if output_dir:
        dest_dir = Path(output_dir).expanduser()
        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = f"yfinance_{_safe_filename(symbol)}_{_safe_filename(period)}_{_safe_filename(interval)}.csv"
        path = dest_dir / filename
        out.to_csv(path, index=False)
        payload["csv_path"] = str(path)
    else:
        payload["records"] = _to_jsonable(out.tail(10).to_dict(orient="records"))
        payload["records_note"] = "No output_dir supplied; returning only the last 10 rows."
    return payload


@dataclass(frozen=True)
class SecCompany:
    ticker: str
    cik: str
    title: str


def _company_tickers() -> list[dict[str, Any]]:
    data = request_json(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS,
        timeout=20,
        ttl_seconds=86400,
        attempts=3,
    )
    if isinstance(data, dict):
        return list(data.values())
    raise SourceDataError("unexpected SEC company_tickers response")


def sec_find_company(ticker: str) -> dict[str, Any]:
    """Resolve ticker to SEC CIK/title using SEC's company_tickers mapping."""

    symbol = _normalize_ticker(ticker)
    for row in _company_tickers():
        if str(row.get("ticker", "")).upper() == symbol:
            cik_int = int(row["cik_str"])
            return {
                "status": "ok",
                "source": "sec_company_tickers",
                "ticker": symbol,
                "cik": f"{cik_int:010d}",
                "title": str(row.get("title", "")),
                "fetched_at": _utc_now_iso(),
            }
    raise SourceDataError(f"ticker not found in SEC company_tickers: {symbol}")


def _sec_submissions(cik: str) -> dict[str, Any]:
    cik10 = str(cik).strip().zfill(10)
    return request_json(
        f"https://data.sec.gov/submissions/CIK{cik10}.json",
        headers=SEC_HEADERS,
        timeout=20,
        ttl_seconds=1800,
        attempts=3,
    )


def _sec_companyfacts(cik: str) -> dict[str, Any]:
    cik10 = str(cik).strip().zfill(10)
    return request_json(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json",
        headers=SEC_HEADERS,
        timeout=30,
        ttl_seconds=21600,
        attempts=3,
    )


def _recent_filings_from_submissions(submissions: dict[str, Any], form_types: list[str], limit: int) -> list[dict[str, Any]]:
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    primary_docs = recent.get("primaryDocument", [])
    descriptions = recent.get("primaryDocDescription", [])
    cik_raw = str(submissions.get("cik", "")).lstrip("0") or str(submissions.get("cik", ""))

    wanted = {form.upper() for form in form_types}
    out: list[dict[str, Any]] = []
    for i, form in enumerate(forms):
        form_u = str(form).upper()
        if wanted and form_u not in wanted:
            continue
        accession = accession_numbers[i]
        accession_nodash = str(accession).replace("-", "")
        primary_doc = primary_docs[i]
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_raw}/{accession_nodash}/{primary_doc}"
        out.append({
            "form": form_u,
            "filing_date": filing_dates[i] if i < len(filing_dates) else "",
            "report_date": report_dates[i] if i < len(report_dates) else "",
            "accession_number": accession,
            "primary_document": primary_doc,
            "description": descriptions[i] if i < len(descriptions) else "",
            "url": url,
        })
        if len(out) >= limit:
            break
    return out


def sec_recent_filings(ticker: str, *, form_types: list[str] | None = None, limit: int = 10) -> dict[str, Any]:
    """Return recent SEC filings for a ticker from SEC submissions JSON."""

    company = sec_find_company(ticker)
    forms = form_types or ["10-K", "10-Q", "8-K", "DEF 14A"]
    limit = max(1, min(int(limit), 50))
    submissions = _sec_submissions(company["cik"])
    filings = _recent_filings_from_submissions(submissions, forms, limit)
    return {
        "status": "ok",
        "source": "sec_submissions",
        "ticker": company["ticker"],
        "cik": company["cik"],
        "company_title": company["title"],
        "form_types": forms,
        "limit": limit,
        "fetched_at": _utc_now_iso(),
        "filings": filings,
    }


def sec_companyfacts_snapshot(ticker: str, *, output_dir: str | None = None) -> dict[str, Any]:
    """Fetch SEC companyfacts JSON and return/save a compact metadata snapshot."""

    company = sec_find_company(ticker)
    facts = _sec_companyfacts(company["cik"])
    payload: dict[str, Any] = {
        "status": "ok",
        "source": "sec_companyfacts",
        "ticker": company["ticker"],
        "cik": company["cik"],
        "company_title": company["title"],
        "entity_name": facts.get("entityName"),
        "fetched_at": _utc_now_iso(),
        "taxonomy_namespaces": sorted((facts.get("facts") or {}).keys()),
    }

    us_gaap = (facts.get("facts") or {}).get("us-gaap", {})
    selected = [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "NetIncomeLoss",
        "OperatingIncomeLoss",
        "Assets",
        "Liabilities",
        "StockholdersEquity",
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "NetCashProvidedByUsedInOperatingActivities",
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "EarningsPerShareDiluted",
    ]
    metrics: dict[str, Any] = {}
    for name in selected:
        item = us_gaap.get(name)
        if not item:
            continue
        units = item.get("units", {})
        first_unit = next(iter(units.keys()), None)
        unit_rows = units.get(first_unit, []) if first_unit else []
        metrics[name] = {
            "label": item.get("label"),
            "description": item.get("description"),
            "unit": first_unit,
            "recent": _to_jsonable(unit_rows[-5:]),
        }
    payload["selected_metrics"] = metrics

    if output_dir:
        dest_dir = Path(output_dir).expanduser()
        dest_dir.mkdir(parents=True, exist_ok=True)
        full_path = dest_dir / f"companyfacts_CIK{company['cik']}.json"
        full_path.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
        snapshot_path = dest_dir / "companyfacts_key_metrics_snapshot.json"
        snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["companyfacts_path"] = str(full_path)
        payload["snapshot_path"] = str(snapshot_path)
    return payload


def sec_fetch_filing(url: str, *, output_dir: str | None = None, max_bytes: int = 20_000_000) -> dict[str, Any]:
    """Fetch a SEC filing URL and optionally persist the raw document."""

    if not url.startswith("https://www.sec.gov/") and not url.startswith("https://sec.gov/"):
        raise SourceDataError("sec_fetch_filing only accepts SEC filing URLs")
    resolved = resolve_sec_primary_document_url(url)
    content = request_bytes(
        resolved,
        headers=SEC_HEADERS,
        timeout=30,
        ttl_seconds=21600,
        attempts=3,
    )
    if len(content) > max_bytes:
        raise SourceDataError(f"SEC filing exceeds max_bytes={max_bytes}: {len(content)} bytes")
    payload: dict[str, Any] = {
        "status": "ok",
        "source": "sec_filing_document",
        "url": url,
        "resolved_url": resolved,
        "bytes": len(content),
        "fetched_at": _utc_now_iso(),
    }
    if output_dir:
        dest_dir = Path(output_dir).expanduser()
        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = _safe_filename(Path(resolved).name or "sec_filing.htm")
        path = dest_dir / filename
        path.write_bytes(content)
        payload["path"] = str(path)
    else:
        text = content[:2000].decode("utf-8", errors="replace")
        payload["preview"] = text
        payload["preview_note"] = "No output_dir supplied; returning only first 2000 decoded bytes."
    return payload


def sec_save_source_pack(
    ticker: str,
    *,
    output_dir: str,
    form_types: list[str] | None = None,
    limit: int = 6,
    include_companyfacts: bool = True,
) -> dict[str, Any]:
    """Save a compact SEC source pack for a ticker into output_dir."""

    dest = Path(output_dir).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    recent = sec_recent_filings(ticker, form_types=form_types, limit=limit)
    summary_path = dest / "sec_recent_filings_summary.json"
    summary_path.write_text(json.dumps(recent, ensure_ascii=False, indent=2), encoding="utf-8")

    saved: list[dict[str, Any]] = []
    for filing in recent["filings"]:
        try:
            saved.append(sec_fetch_filing(filing["url"], output_dir=str(dest)))
            time.sleep(0.2)
        except Exception as exc:
            saved.append({"status": "error", "url": filing.get("url"), "error": f"{type(exc).__name__}: {exc}"})

    facts_payload: dict[str, Any] | None = None
    if include_companyfacts:
        try:
            facts_payload = sec_companyfacts_snapshot(ticker, output_dir=str(dest))
        except Exception as exc:
            facts_payload = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    return {
        "status": "ok",
        "source": "sec_source_pack",
        "ticker": recent["ticker"],
        "cik": recent["cik"],
        "company_title": recent["company_title"],
        "output_dir": str(dest),
        "summary_path": str(summary_path),
        "filing_count": len(recent["filings"]),
        "saved_filings": saved,
        "companyfacts": facts_payload,
        "fetched_at": _utc_now_iso(),
    }
