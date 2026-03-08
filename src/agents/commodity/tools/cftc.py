"""
Commodity Tool — CFTC COT Reports: speculative long/short positioning.

Fetches the Commitments of Traders (COT) report data from the CFTC,
showing how speculators, hedgers, and dealers are positioned in futures markets.

Data source: CFTC public CSV releases (no API key required).
URL: https://www.cftc.gov/dea/newcot/deafut.txt (futures-only report)
"""

from typing import Tuple, Dict, Optional


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
        - file_path: Path to saved Markdown under data/, or None if asset not in CFTC_MARKET_CODES.
        - metadata_dict: {
              "asset": str,
              "latest_date": str,
              "spec_net_long": int,
              "spec_net_change_1w": int,
              "commercial_net": int,
          }
    """
    ...


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
    ...
