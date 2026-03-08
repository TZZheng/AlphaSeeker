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


def get_series_for_topic(topic: str) -> List[str]:
    """
    Maps a macro topic string to the relevant FRED series IDs.
    Uses keyword matching against the topic to select appropriate series.

    Args:
        topic: Natural language macro topic, e.g. "US interest rate outlook".

    Returns:
        List of FRED series ID strings to fetch.
    """
    ...


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
    ...


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
    ...
