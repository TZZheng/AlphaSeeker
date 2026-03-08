"""
Commodity Tool — Futures Curve Data: contango / backwardation analysis.

Fetches futures contract prices across expiration months to build the
forward curve. Determines market structure (contango vs backwardation)
and calculates annualised roll yield.

Data source: yfinance continuous futures contracts.
"""

from typing import Tuple, Dict, Optional, List


# ---------------------------------------------------------------------------
# Commodity → yfinance futures ticker mapping
# ---------------------------------------------------------------------------
# yfinance uses specific ticker suffixes for commodity futures.

FUTURES_TICKERS: Dict[str, Dict[str, str]] = {
    "crude oil": {
        "base_ticker": "CL",
        "exchange": "NYMEX",
        "contract_months": ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"],
        "name": "WTI Crude Oil",
    },
    "gold": {
        "base_ticker": "GC",
        "exchange": "COMEX",
        "contract_months": ["G", "J", "M", "Q", "V", "Z"],
        "name": "Gold",
    },
    "silver": {
        "base_ticker": "SI",
        "exchange": "COMEX",
        "contract_months": ["H", "K", "N", "U", "Z"],
        "name": "Silver",
    },
    "copper": {
        "base_ticker": "HG",
        "exchange": "COMEX",
        "contract_months": ["H", "K", "N", "U", "Z"],
        "name": "Copper",
    },
    "natural gas": {
        "base_ticker": "NG",
        "exchange": "NYMEX",
        "contract_months": ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"],
        "name": "Henry Hub Natural Gas",
    },
    "corn": {
        "base_ticker": "ZC",
        "exchange": "CBOT",
        "contract_months": ["H", "K", "N", "U", "Z"],
        "name": "Corn",
    },
    "soybeans": {
        "base_ticker": "ZS",
        "exchange": "CBOT",
        "contract_months": ["F", "H", "K", "N", "Q", "U", "X"],
        "name": "Soybeans",
    },
}


def fetch_futures_curve(
    asset: str,
    num_contracts: int = 12,
) -> Tuple[Optional[str], Dict]:
    """
    Fetches the current futures curve for the given commodity and saves as Markdown.

    For each active contract month, fetches the last settlement price.
    Outputs a table with columns: Contract Month | Price | Days to Expiry.
    Also computes:
    - Market structure: "contango" or "backwardation"
    - Front-month vs 12-month spread (absolute and %)
    - Annualised roll yield

    Args:
        asset: Commodity name, e.g. "crude oil", "gold".
        num_contracts: Number of forward months to fetch.

    Returns:
        Tuple of (file_path_or_None, metadata_dict).
        - file_path: Path to saved Markdown under data/.
        - metadata_dict: {
              "asset": str,
              "spot_price": float,
              "12m_price": float,
              "structure": "contango" | "backwardation",
              "annualised_roll_yield_pct": float,
          }
        Returns (None, {}) if the asset is not in FUTURES_TICKERS.
    """
    ...


def classify_curve_structure(
    spot_price: float,
    forward_price: float,
) -> str:
    """
    Determines whether the futures curve is in contango or backwardation.

    Args:
        spot_price: Front-month (nearest) contract price.
        forward_price: Deferred (e.g. 12-month) contract price.

    Returns:
        "contango" if forward > spot, "backwardation" if forward < spot,
        "flat" if within 0.5% of each other.
    """
    ...


def calculate_roll_yield(
    spot_price: float,
    forward_price: float,
    days_between: int = 365,
) -> float:
    """
    Calculates the annualised roll yield from contango/backwardation.

    Roll yield = (spot - forward) / forward * (365 / days_between) * 100

    Positive roll yield = backwardation (favorable for long holders).
    Negative roll yield = contango (cost of carry for long holders).

    Args:
        spot_price: Front-month contract price.
        forward_price: Deferred contract price.
        days_between: Calendar days between the two contracts.

    Returns:
        Annualised roll yield as a percentage.
    """
    ...
