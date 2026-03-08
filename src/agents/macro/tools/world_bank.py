"""
Macro Tool — World Bank API: cross-country economic indicators.

Fetches comparative economic data across countries from the World Bank Open Data API.
Used when the macro topic involves non-US economies or cross-country comparisons.

API Docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
No API key required.
"""

from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Common World Bank Indicator Codes
# ---------------------------------------------------------------------------

WB_INDICATORS: Dict[str, str] = {
    "NY.GDP.MKTP.CD":     "GDP (current US$)",
    "NY.GDP.MKTP.KD.ZG":  "GDP growth (annual %)",
    "FP.CPI.TOTL.ZG":     "Inflation, consumer prices (annual %)",
    "SL.UEM.TOTL.ZS":     "Unemployment, total (% of labor force)",
    "BN.CAB.XOKA.GD.ZS":  "Current account balance (% of GDP)",
    "GC.DOD.TOTL.GD.ZS":  "Central government debt (% of GDP)",
    "FR.INR.RINR":         "Real interest rate (%)",
    "NE.EXP.GNFS.ZS":     "Exports of goods and services (% of GDP)",
    "NE.IMP.GNFS.ZS":     "Imports of goods and services (% of GDP)",
}

# ISO 3166-1 alpha-3 codes for common economies
COUNTRY_CODES: Dict[str, str] = {
    "US": "USA",
    "China": "CHN",
    "EU": "EUU",
    "Japan": "JPN",
    "UK": "GBR",
    "Germany": "DEU",
    "India": "IND",
    "Brazil": "BRA",
}


def resolve_country_codes(countries: List[str]) -> List[str]:
    """
    Maps natural language country names to World Bank ISO codes.
    Falls back to the input string if no mapping is found.

    Args:
        countries: List of country names or codes, e.g. ["US", "China", "EU"].

    Returns:
        List of ISO alpha-3 codes, e.g. ["USA", "CHN", "EUU"].
    """
    ...


def fetch_world_bank_indicators(
    countries: List[str],
    indicator_codes: Optional[List[str]] = None,
    date_range: str = "2019:2025",
) -> Tuple[str, Dict]:
    """
    Fetches World Bank indicator data for the given countries and saves as Markdown.

    If indicator_codes is not specified, fetches a default set of key macro indicators
    (GDP growth, inflation, unemployment, current account, government debt).

    The output Markdown contains one section per country, each with a table of
    indicators across years.

    Args:
        countries: List of country names (mapped via resolve_country_codes).
        indicator_codes: Optional list of WB indicator codes. Defaults to core set.
        date_range: Year range string, e.g. "2019:2025".

    Returns:
        Tuple of (file_path, metadata_dict).
        - file_path: Path to the saved Markdown file under data/.
        - metadata_dict: {country: {indicator: latest_value}}.

    Raises:
        requests.HTTPError: If the World Bank API returns an error.
    """
    ...
