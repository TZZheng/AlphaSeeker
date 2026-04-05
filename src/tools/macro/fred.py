"""
Macro Tool — FRED API: interest rates, CPI, GDP, employment data.

Fetches time-series data from the Federal Reserve Economic Data (FRED) API
and saves it as a structured Markdown file for LLM consumption.

Requires: FRED_API_KEY environment variable (free from https://fred.stlouisfed.org/docs/api/fred/v2/api_key.html).
"""

import os
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Common FRED Series IDs grouped by category
# ---------------------------------------------------------------------------

FRED_SERIES: Dict[str, List[str]] = {
    "interest_rates": [
        "FEDFUNDS",     # Federal Funds Effective Rate
        "DGS10",        # 10-Year Treasury Constant Maturity Rate
        "DGS2",         # 2-Year Treasury Constant Maturity Rate
        "T10Y2Y",       # 10-Year minus 2-Year spread (yield curve)
    ],
    "inflation": [
        "CPIAUCSL",     # CPI for All Urban Consumers
        "CPILFESL",     # Core CPI (less food & energy)
        "PCEPI",        # PCE Price Index (Fed's preferred)
        "PCEPILFE",     # Core PCE
    ],
    "employment": [
        "UNRATE",       # Unemployment Rate
        "PAYEMS",       # Total Nonfarm Payrolls
        "ICSA",         # Initial Jobless Claims (weekly)
    ],
    "gdp_output": [
        "GDP",          # Gross Domestic Product
        "GDPC1",        # Real GDP
        "INDPRO",       # Industrial Production Index
    ],
    "money_supply": [
        "M2SL",         # M2 Money Stock
        "WALCL",        # Fed Balance Sheet (Total Assets)
    ],
}


import datetime
import requests
import pandas as pd

from src.shared.reliability import request_json

def get_series_for_topic(topic: str) -> List[str]:
    """
    Maps a macro topic string to the relevant FRED series IDs.
    Uses keyword matching against the topic to select appropriate series.

    Args:
        topic: Natural language macro topic, e.g. "US interest rate outlook".

    Returns:
        List of FRED series ID strings to fetch.
    """
    topic_lower = topic.lower()
    series_ids = []
    
    if any(kw in topic_lower for kw in ["interest", "rate", "fed funds", "yield"]):
        series_ids.extend(FRED_SERIES["interest_rates"])
    if any(kw in topic_lower for kw in ["inflation", "cpi", "pce", "price"]):
        series_ids.extend(FRED_SERIES["inflation"])
    if any(kw in topic_lower for kw in ["employment", "job", "unemployment", "payroll"]):
        series_ids.extend(FRED_SERIES["employment"])
    if any(kw in topic_lower for kw in ["gdp", "growth", "output", "production", "economy"]):
        series_ids.extend(FRED_SERIES["gdp_output"])
    if any(kw in topic_lower for kw in ["money", "supply", "balance sheet", "m2", "liquidity"]):
        series_ids.extend(FRED_SERIES["money_supply"])
        
    # Default to interest rates and inflation if nothing matched
    if not series_ids:
        series_ids.extend(FRED_SERIES["interest_rates"])
        series_ids.extend(FRED_SERIES["inflation"])
        
    return list(set(series_ids))


def fetch_fred_series(
    series_ids: List[str],
    observation_start: Optional[str] = None,
    observation_end: Optional[str] = None,
    limit: int = 60,
) -> Tuple[str, Dict]:
    """
    Fetches one or more FRED time series and saves as a structured Markdown file.

    Each series is formatted as a table with Date | Value columns, plus metadata
    (title, units, frequency, last updated). The Markdown file is optimized for
    LLM consumption — clear headers, no extraneous formatting.

    Args:
        series_ids: List of FRED series IDs to fetch.
        observation_start: Start date (YYYY-MM-DD). Defaults to 5 years ago.
        observation_end: End date (YYYY-MM-DD). Defaults to today.
        limit: Max observations per series (most recent N).

    Returns:
        Tuple of (file_path, metadata_dict).
        - file_path: Path to the saved Markdown file under data/.
        - metadata_dict: {series_id: {title, units, frequency, last_updated}}.

    Raises:
        ValueError: If FRED_API_KEY is not set.
        requests.HTTPError: If the FRED API returns an error.
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY environment variable is not set")
        
    if not observation_end:
        observation_end = datetime.datetime.now().strftime("%Y-%m-%d")
    if not observation_start:
        observation_start = (datetime.datetime.now() - datetime.timedelta(days=5*365)).strftime("%Y-%m-%d")
        
    metadata_dict = {}
    markdown_content = "# FRED Macroeconomic Data\n\n"
    
    for series_id in series_ids:
        print(f"Fetching FRED series: {series_id}")
        # Fetch metadata
        meta_url = "https://api.stlouisfed.org/fred/series"
        meta_params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json"
        }
        meta_json = request_json(
            meta_url,
            params=meta_params,
            timeout=15,
            ttl_seconds=21600,
            attempts=3,
        )
        
        if not meta_json.get("seriess"):
            continue
            
        series_meta = meta_json["seriess"][0]
        title = series_meta.get("title", series_id)
        units = series_meta.get("units", "")
        freq = series_meta.get("frequency", "")
        last_updated = series_meta.get("last_updated", "")
        
        metadata_dict[series_id] = {
            "title": title,
            "units": units,
            "frequency": freq,
            "last_updated": last_updated
        }
        
        markdown_content += f"## {title} ({series_id})\n"
        markdown_content += f"- **Units**: {units}\n"
        markdown_content += f"- **Frequency**: {freq}\n"
        markdown_content += f"- **Last Updated**: {last_updated}\n\n"
        
        # Fetch observations
        obs_url = "https://api.stlouisfed.org/fred/series/observations"
        obs_params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": observation_start,
            "observation_end": observation_end,
            "sort_order": "desc",
            "limit": limit
        }
        obs_json = request_json(
            obs_url,
            params=obs_params,
            timeout=15,
            ttl_seconds=21600,
            attempts=3,
        )
        
        observations = obs_json.get("observations", [])
        if not observations:
            markdown_content += "_No data available for this range._\n\n"
            continue
            
        # Add to markdown as table
        markdown_content += "| Date | Value |\n|---|---|\n"
        for obs in observations:
            date = obs.get("date", "")
            value = obs.get("value", "")
            if value != ".": # FRED uses "." for missing values
                markdown_content += f"| {date} | {value} |\n"
        
        markdown_content += "\n---\n\n"

    # Save to file
    save_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(save_dir, f"fred_data_{timestamp}.md")
    
    with open(file_path, "w") as f:
        f.write(markdown_content)
        
    return file_path, metadata_dict


def fetch_macro_indicators(
    topic: str,
    countries: List[str] = [],
) -> Tuple[str, Dict]:
    """
    High-level convenience function called by the macro planner node.
    Maps the topic to FRED series, fetches them, and returns the results.

    This is the primary interface that macro/nodes.py calls.

    Args:
        topic: Natural language macro topic.
        countries: List of country codes (currently only "US" is supported via FRED;
                   other countries fall through to world_bank.py).

    Returns:
        Tuple of (file_path, metadata_dict) — same as fetch_fred_series.
    """
    series_ids = get_series_for_topic(topic)
    
    try:
        file_path, metadata = fetch_fred_series(series_ids=series_ids, limit=24)
        return file_path, metadata
    except Exception as e:
        print(f"Error fetching FRED data: {e}")
        return "", {}
