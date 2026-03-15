"""
Commodity Tool — Futures Curve Data: contango / backwardation analysis.

Fetches futures contract prices across expiration months to build the
forward curve. Determines market structure (contango vs backwardation)
and calculates annualised roll yield.

Data source: yfinance continuous futures contracts.
"""

from __future__ import annotations

import datetime
import os
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import yfinance as yf

from src.shared.reliability import cached_retry_call


# ---------------------------------------------------------------------------
# Commodity → yfinance futures ticker mapping
# ---------------------------------------------------------------------------
# yfinance uses specific ticker suffixes for commodity futures.

class FuturesTickerConfig(TypedDict):
    base_ticker: str
    exchange: str
    contract_months: List[str]
    name: str


FUTURES_TICKERS: Dict[str, FuturesTickerConfig] = {
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

EXCHANGE_SUFFIX = {
    "NYMEX": "NYM",
    "COMEX": "CMX",
    "CBOT": "CBT",
}

MONTH_CODE_BY_INT = {
    1: "F",
    2: "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z",
}


def _match_asset(asset: str) -> Optional[FuturesTickerConfig]:
    asset_normalized = asset.lower().strip()
    if asset_normalized in FUTURES_TICKERS:
        return FUTURES_TICKERS[asset_normalized]
    matched_key = next(
        (k for k in FUTURES_TICKERS if k in asset_normalized or asset_normalized in k),
        None,
    )
    if not matched_key:
        return None
    return FUTURES_TICKERS[matched_key]


def _iter_candidate_contracts(
    config: FuturesTickerConfig,
    months_ahead: int = 48,
) -> List[Dict[str, Any]]:
    """Generate candidate Yahoo futures symbols based on month code schedules."""
    base_ticker = config["base_ticker"]
    allowed_month_codes = set(config["contract_months"])
    suffix = EXCHANGE_SUFFIX.get(config["exchange"])
    if not suffix:
        return []

    today = datetime.date.today()
    candidates: List[Dict[str, Any]] = []

    for offset in range(months_ahead):
        month = ((today.month - 1 + offset) % 12) + 1
        year = today.year + ((today.month - 1 + offset) // 12)
        month_code = MONTH_CODE_BY_INT[month]
        if month_code not in allowed_month_codes:
            continue

        symbol = f"{base_ticker}{month_code}{str(year)[-2:]}.{suffix}"
        contract_month_date = datetime.date(year, month, 1)
        candidates.append(
            {
                "symbol": symbol,
                "contract_month": f"{year}-{month:02d}",
                "contract_date": contract_month_date,
            }
        )

    return candidates


def _fetch_last_close(symbol: str) -> Optional[float]:
    try:
        hist = cached_retry_call(
            "yfinance_futures_history",
            {"symbol": symbol, "period": "1mo"},
            lambda: yf.Ticker(symbol).history(period="1mo"),
            ttl_seconds=900,
            attempts=3,
        )
        if hist.empty or "Close" not in hist.columns:
            return None
        close_series = hist["Close"].dropna()
        if close_series.empty:
            return None
        return float(close_series.iloc[-1])
    except Exception:
        return None


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
    if not spot_price or not forward_price:
        return "unknown"
        
    spread_pct = (forward_price - spot_price) / spot_price
    
    if abs(spread_pct) <= 0.005:
        return "flat"
    elif spread_pct > 0:
        return "contango"
    else:
        return "backwardation"


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
    if not forward_price or not days_between:
        return 0.0
    return ((spot_price - forward_price) / forward_price) * (365.0 / days_between) * 100.0


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
    """
    config = _match_asset(asset)
    if not config:
        return None, {}

    print(f"Fetching futures curve for {config['name']} via yfinance contracts...")

    curve_points: List[Dict[str, Any]] = []
    today = datetime.date.today()
    for candidate in _iter_candidate_contracts(config, months_ahead=max(36, num_contracts * 6)):
        price = _fetch_last_close(candidate["symbol"])
        if price is None:
            continue
        days_to_contract = max((candidate["contract_date"] - today).days, 0)
        curve_points.append(
            {
                "symbol": candidate["symbol"],
                "contract_month": candidate["contract_month"],
                "contract_date": candidate["contract_date"],
                "days_to_contract": days_to_contract,
                "price": price,
            }
        )
        if len(curve_points) >= num_contracts:
            break

    if not curve_points:
        print(f"Futures: no contract prices available for {asset}")
        return None, {}

    front = curve_points[0]
    far = curve_points[-1]

    if len(curve_points) >= 2:
        days_between = max((far["contract_date"] - front["contract_date"]).days, 1)
        structure = classify_curve_structure(front["price"], far["price"])
        roll_yield = calculate_roll_yield(front["price"], far["price"], days_between=days_between)
        spread_abs = far["price"] - front["price"]
        spread_pct = ((far["price"] - front["price"]) / front["price"]) * 100 if front["price"] else 0.0
    else:
        days_between = 0
        structure = "unknown"
        roll_yield = 0.0
        spread_abs = 0.0
        spread_pct = 0.0

    markdown_content = f"# Futures Curve: {config['name']}\n\n"
    markdown_content += f"**Contracts Loaded:** {len(curve_points)}\n"
    markdown_content += f"**Front Contract:** {front['symbol']} ({front['contract_month']}) @ {front['price']:.4f}\n"
    if len(curve_points) >= 2:
        markdown_content += f"**Far Contract:** {far['symbol']} ({far['contract_month']}) @ {far['price']:.4f}\n"
    markdown_content += f"**Market Structure:** {structure.title()}\n"
    markdown_content += f"**Front-to-Far Spread:** {spread_abs:+.4f} ({spread_pct:+.2f}%)\n"
    markdown_content += f"**Annualised Roll Yield:** {roll_yield:.2f}%\n\n"

    markdown_content += "| Contract Month | Symbol | Last Price | Days to Contract Month |\n|---|---|---:|---:|\n"
    for point in curve_points:
        markdown_content += (
            f"| {point['contract_month']} | {point['symbol']} | {point['price']:.4f} | "
            f"{point['days_to_contract']} |\n"
        )

    metadata_dict = {
        "asset": config["name"],
        "front_symbol": front["symbol"],
        "front_price": front["price"],
        "far_symbol": far["symbol"],
        "far_price": far["price"],
        "days_between": days_between,
        "structure": structure,
        "annualised_roll_yield_pct": roll_yield,
        "curve_points": len(curve_points),
    }

    save_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(save_dir, f"futures_curve_{timestamp}.md")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return file_path, metadata_dict
