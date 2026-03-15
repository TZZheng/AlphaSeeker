"""
Commodity Tool — CFTC COT Reports: speculative long/short positioning.

Fetches the Commitments of Traders (COT) report data from the CFTC,
showing how speculators, hedgers, and dealers are positioned in futures markets.

Data source: CFTC public CSV releases (no API key required).
URL: https://www.cftc.gov/dea/newcot/deafut.txt (futures-only report)
"""

from __future__ import annotations

import csv
import datetime
import io
import os
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.shared.reliability import request_bytes


# ---------------------------------------------------------------------------
# Commodity → CFTC Market Code mapping
# ---------------------------------------------------------------------------
# CFTC uses specific market/exchange codes. These map common asset names.

CFTC_MARKET_CODES: Dict[str, Dict[str, str]] = {
    "crude oil": {
        "market_name": "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
        "contract_code": "067651",
    },
    "gold": {
        "market_name": "GOLD - COMMODITY EXCHANGE INC.",
        "contract_code": "088691",
    },
    "silver": {
        "market_name": "SILVER - COMMODITY EXCHANGE INC.",
        "contract_code": "084691",
    },
    "copper": {
        "market_name": "COPPER-GRADE #1 - COMMODITY EXCHANGE INC.",
        "contract_code": "085692",
    },
    "natural gas": {
        "market_name": "NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE",
        "contract_code": "023651",
    },
    "corn": {
        "market_name": "CORN - CHICAGO BOARD OF TRADE",
        "contract_code": "002602",
    },
    "soybeans": {
        "market_name": "SOYBEANS - CHICAGO BOARD OF TRADE",
        "contract_code": "005602",
    },
    "wheat": {
        "market_name": "WHEAT-SRW - CHICAGO BOARD OF TRADE",
        "contract_code": "001602",
    },
}

COT_YEARLY_URL = "https://www.cftc.gov/files/dea/history/deacot{year}.zip"

_COL_MARKET = "Market and Exchange Names"
_COL_DATE = "As of Date in Form YYYY-MM-DD"
_COL_CONTRACT_CODE = "CFTC Contract Market Code"
_COL_OPEN_INTEREST = "Open Interest (All)"
_COL_NONCOMM_LONG = "Noncommercial Positions-Long (All)"
_COL_NONCOMM_SHORT = "Noncommercial Positions-Short (All)"
_COL_COMM_LONG = "Commercial Positions-Long (All)"
_COL_COMM_SHORT = "Commercial Positions-Short (All)"


def _match_market(asset: str) -> Optional[Dict[str, str]]:
    asset_normalized = asset.lower().strip()
    matched_key = next(
        (k for k in CFTC_MARKET_CODES if k in asset_normalized or asset_normalized in k),
        None,
    )
    if not matched_key:
        return None
    return CFTC_MARKET_CODES[matched_key]


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip().replace(",", "")
    if not text:
        return 0
    return int(float(text))


def _download_year_rows(year: int) -> List[Dict[str, str]]:
    url = COT_YEARLY_URL.format(year=year)
    try:
        payload = request_bytes(
            url,
            timeout=30,
            ttl_seconds=21600,
            attempts=3,
        )
    except Exception as exc:
        print(f"CFTC: failed to download {url} ({exc})")
        return []

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            member_name = next((n for n in zf.namelist() if n.lower().endswith(".txt")), None)
            if not member_name:
                print(f"CFTC: no TXT member in yearly archive {year}")
                return []
            with zf.open(member_name, "r") as raw_file:
                text_file = io.TextIOWrapper(raw_file, encoding="utf-8-sig", newline="")
                return list(csv.DictReader(text_file))
    except Exception as exc:
        print(f"CFTC: failed to parse archive for {year} ({exc})")
        return []


def _load_recent_positions(contract_code: str, weeks: int) -> List[Dict[str, Any]]:
    """Load most recent weekly observations for a specific CFTC contract code."""
    today = datetime.date.today()
    years = [today.year, today.year - 1]

    parsed_rows: List[Dict[str, Any]] = []
    for year in years:
        for row in _download_year_rows(year):
            if row.get(_COL_CONTRACT_CODE, "").strip() != contract_code:
                continue
            try:
                report_date = datetime.date.fromisoformat(row.get(_COL_DATE, "").strip())
            except Exception:
                continue

            noncomm_long = _to_int(row.get(_COL_NONCOMM_LONG))
            noncomm_short = _to_int(row.get(_COL_NONCOMM_SHORT))
            comm_long = _to_int(row.get(_COL_COMM_LONG))
            comm_short = _to_int(row.get(_COL_COMM_SHORT))
            open_interest = _to_int(row.get(_COL_OPEN_INTEREST))

            parsed_rows.append(
                {
                    "date": report_date,
                    "market_name": row.get(_COL_MARKET, "").strip(),
                    "open_interest": open_interest,
                    "noncomm_long": noncomm_long,
                    "noncomm_short": noncomm_short,
                    "comm_long": comm_long,
                    "comm_short": comm_short,
                    "spec_net": noncomm_long - noncomm_short,
                    "commercial_net": comm_long - comm_short,
                }
            )

    parsed_rows.sort(key=lambda r: r["date"], reverse=True)

    # Deduplicate report dates in case a file source appears twice.
    unique_rows: List[Dict[str, Any]] = []
    seen_dates = set()
    for row in parsed_rows:
        if row["date"] in seen_dates:
            continue
        seen_dates.add(row["date"])
        unique_rows.append(row)
        if len(unique_rows) >= weeks:
            break

    return unique_rows


def get_positioning_summary(asset: str) -> str:
    """
    Returns a one-paragraph natural language summary of current COT positioning.
    Useful for the synthesize_research node to incorporate into the final brief.

    Args:
        asset: Commodity name.

    Returns:
        A short summary string, e.g.:
        "Speculative net longs in crude oil are at 180K contracts, down 12K w/w.
         This is the lowest positioning since March 2024, suggesting bearish sentiment."
        Returns "No COT data available for {asset}." if asset not in mapping.
    """
    market = _match_market(asset)
    if not market:
        return f"No COT data available for {asset}."

    rows = _load_recent_positions(market["contract_code"], weeks=2)
    if not rows:
        return f"No recent COT rows found for {asset}."

    latest = rows[0]
    if len(rows) > 1:
        prev = rows[1]
        spec_delta = latest["spec_net"] - prev["spec_net"]
        commercial_delta = latest["commercial_net"] - prev["commercial_net"]
    else:
        spec_delta = 0
        commercial_delta = 0

    return (
        f"As of {latest['date']}, speculative net positioning is {latest['spec_net']:,} contracts "
        f"(w/w change {spec_delta:+,}); commercial net positioning is {latest['commercial_net']:,} "
        f"(w/w change {commercial_delta:+,})."
    )


def fetch_cot_report(
    asset: str,
    num_weeks: int = 12,
) -> Tuple[Optional[str], Dict]:
    """
    Fetches the latest COT positioning data for the given commodity.

    Downloads and parses the CFTC COT futures-only report, extracts the
    rows matching the commodity's contract code, and formats as Markdown:
    - Commercial (hedger) net position
    - Non-commercial (speculative) net position
    - Weekly change in each
    - Historical positioning over num_weeks

    Args:
        asset: Commodity name, e.g. "crude oil", "gold", "corn".
        num_weeks: Number of recent weekly reports to include.

    Returns:
        Tuple of (file_path_or_None, metadata_dict).
    """
    market = _match_market(asset)
    if not market:
        return None, {}

    rows = _load_recent_positions(market["contract_code"], weeks=max(2, num_weeks))
    if not rows:
        print(f"CFTC: no parsed rows found for {asset} ({market['contract_code']})")
        return None, {}

    latest = rows[0]
    prev = rows[1] if len(rows) > 1 else rows[0]
    spec_change = latest["spec_net"] - prev["spec_net"]
    commercial_change = latest["commercial_net"] - prev["commercial_net"]

    markdown_content = f"# COT Positioning: {latest['market_name']}\n\n"
    markdown_content += f"**Contract Code:** `{market['contract_code']}`\n"
    markdown_content += f"**Latest Report Date:** {latest['date']}\n"
    markdown_content += f"**Speculative Net (Non-commercial):** {latest['spec_net']:,} ({spec_change:+,} w/w)\n"
    markdown_content += f"**Commercial Net:** {latest['commercial_net']:,} ({commercial_change:+,} w/w)\n"
    markdown_content += f"**Open Interest:** {latest['open_interest']:,}\n\n"

    markdown_content += (
        "| Date | Non-Comm Long | Non-Comm Short | Spec Net | Comm Long | Comm Short | "
        "Commercial Net | Open Interest |\n"
    )
    markdown_content += "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for row in rows[:num_weeks]:
        markdown_content += (
            f"| {row['date']} | {row['noncomm_long']:,} | {row['noncomm_short']:,} | "
            f"{row['spec_net']:,} | {row['comm_long']:,} | {row['comm_short']:,} | "
            f"{row['commercial_net']:,} | {row['open_interest']:,} |\n"
        )

    metadata_dict = {
        "asset": latest["market_name"],
        "contract_code": market["contract_code"],
        "latest_date": latest["date"].isoformat(),
        "spec_net_long": latest["spec_net"],
        "spec_net_change_1w": spec_change,
        "commercial_net": latest["commercial_net"],
        "commercial_net_change_1w": commercial_change,
        "open_interest": latest["open_interest"],
    }

    save_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(save_dir, f"cot_data_{timestamp}.md")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return file_path, metadata_dict
