"""
Peer Analysis Tool — data-driven peer discovery and comparison.

Uses yfinance sector/industry data and market-cap filtering to find
real comparable companies, rather than relying on LLM hallucination.
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional


class PeerAnalysisError(Exception):
    """Custom exception for peer analysis errors."""
    pass


# --- Well-known sector peer pools ---
# These are curated lists of liquid, well-known companies per sector/industry.
# yfinance does not provide a "find peers" API, so we maintain a lookup table
# of candidates that can be filtered by market-cap proximity.
INDUSTRY_PEER_CANDIDATES: Dict[str, List[str]] = {
    # AI / Semiconductors / Data Infrastructure
    "Semiconductors": [
        "NVDA", "AMD", "INTC", "AVGO", "QCOM", "TSM", "MRVL", "MU",
        "LRCX", "KLAC", "AMAT", "ON", "NXPI", "TXN", "ADI",
    ],
    "Software—Infrastructure": [
        "PLTR", "SNOW", "DDOG", "NET", "MDB", "CRWD", "ZS", "PANW",
        "SPLK", "ESTC", "CFLT", "S", "GTLB", "PATH", "AI",
    ],
    "Software—Application": [
        "CRM", "NOW", "WDAY", "VEEV", "HUBS", "TEAM", "ADSK", "INTU",
        "CDNS", "ANSS", "DOCU", "ZM", "BILL", "DDOG", "MNDY",
    ],
    "Information Technology Services": [
        "IBM", "ACN", "CTSH", "INFY", "WIT", "EPAM", "GLOB", "GDYN",
    ],
    "Internet Content & Information": [
        "GOOGL", "META", "SNAP", "PINS", "RDDT", "YELP",
    ],
    "Electronic Components": [
        "APH", "TEL", "GLW", "JBL", "FLEX", "CLS",
    ],
    "Computer Hardware": [
        "AAPL", "DELL", "HPQ", "HPE", "SMCI", "LNVGY",
    ],
    # Cloud / Hyperscaler-adjacent
    "Cloud Infrastructure": [
        "AMZN", "MSFT", "GOOGL", "ORCL", "IBM", "SNOW", "NET",
    ],
}

# Fallback: broad Technology sector peers
TECHNOLOGY_FALLBACK = [
    "NVDA", "AMD", "PLTR", "SNOW", "DDOG", "CRWD", "NET", "CRM",
    "NOW", "PANW", "SMCI", "AI", "PATH", "MDB", "CFLT",
]


def discover_peers(
    ticker: str,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    market_cap: Optional[float] = None,
    max_peers: int = 5,
) -> List[str]:
    """
    Discovers comparable peers using sector/industry matching and market-cap filtering.

    Strategy:
      1. Look up the target's industry in INDUSTRY_PEER_CANDIDATES.
      2. If no industry match, fall back to TECHNOLOGY_FALLBACK.
      3. Filter by market-cap proximity (within 0.1× to 10× of target).
      4. Sort by market-cap closeness and return top N.

    Args:
        ticker: Target stock ticker.
        sector: Target company's sector (from yfinance .info).
        industry: Target company's industry (from yfinance .info).
        market_cap: Target company's market cap.
        max_peers: Maximum number of peers to return.

    Returns:
        List of peer ticker symbols.
    """
    # Determine candidate pool
    candidates: List[str] = []

    if industry and industry in INDUSTRY_PEER_CANDIDATES:
        candidates = INDUSTRY_PEER_CANDIDATES[industry]
    else:
        # Try partial matching on industry keywords
        if industry:
            industry_lower = industry.lower()
            for key, tickers in INDUSTRY_PEER_CANDIDATES.items():
                if any(word in key.lower() for word in industry_lower.split()):
                    candidates.extend(tickers)
        # Fallback
        if not candidates:
            candidates = TECHNOLOGY_FALLBACK

    # Remove the target itself
    candidates = [c for c in candidates if c.upper() != ticker.upper()]

    # Remove duplicates
    candidates = list(dict.fromkeys(candidates))

    if not market_cap or market_cap == "N/A":
        # Can't filter by market cap; return first N candidates
        return candidates[:max_peers]

    # Filter and rank by market-cap proximity
    scored: List[Tuple[str, float, float]] = []  # (ticker, mcap, distance)

    for candidate in candidates:
        try:
            info = yf.Ticker(candidate).info
            c_mcap = info.get("marketCap")
            if c_mcap and c_mcap > 0:
                ratio = c_mcap / market_cap
                # Accept peers within 0.05x to 20x market cap
                if 0.05 <= ratio <= 20:
                    distance = abs(1.0 - ratio)
                    scored.append((candidate, c_mcap, distance))
        except Exception:
            continue

    # Sort by closeness to target market cap
    scored.sort(key=lambda x: x[2])

    return [t[0] for t in scored[:max_peers]]


def fetch_peer_metrics(
    tickers: List[str],
    target_ticker: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Fetches comparative metrics for a list of tickers.

    Collects more data points than before: revenue, margins, EV/EBITDA,
    revenue growth, FCF — not just price and P/E.

    Args:
        tickers: List of stock symbols (e.g., ['AAPL', 'MSFT', 'GOOGL']).
        target_ticker: The primary ticker being analyzed (highlighted in output).

    Returns:
        Tuple of (absolute path to saved Markdown file, metadata dict).

    Raises:
        PeerAnalysisError: If no data could be fetched.
    """
    if not tickers:
        raise PeerAnalysisError("No tickers provided for peer analysis.")

    stats: List[Dict[str, Any]] = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # 1-year performance
            hist = stock.history(period="1y")
            if not hist.empty:
                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                perf_1y = ((end_price - start_price) / start_price) * 100
            else:
                perf_1y = None

            mcap = info.get('marketCap')
            revenue = info.get('totalRevenue')
            ev = info.get('enterpriseValue')
            ev_ebitda = info.get('enterpriseToEbitda')

            stats.append({
                "Ticker": ticker,
                "Company": info.get("shortName", ticker),
                "Sector": info.get('sector', 'N/A'),
                "Industry": info.get('industry', 'N/A'),
                "Price": info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
                "Market Cap": _fmt(mcap),
                "EV": _fmt(ev),
                "1y Change (%)": f"{perf_1y:.1f}" if perf_1y is not None else "N/A",
                "P/E (Trailing)": _round(info.get('trailingPE')),
                "P/E (Forward)": _round(info.get('forwardPE')),
                "EV/EBITDA": _round(ev_ebitda),
                "Rev Growth": _pct(info.get('revenueGrowth')),
                "Gross Margin": _pct(info.get('grossMargins')),
                "Op Margin": _pct(info.get('operatingMargins')),
                "Revenue (TTM)": _fmt(revenue),
                "FCF": _fmt(info.get('freeCashflow')),
                "D/E": _round(info.get('debtToEquity')),
            })

        except Exception as e:
            print(f"Warning: Failed to fetch data for {ticker}: {e}")
            continue

    if not stats:
        raise PeerAnalysisError("Failed to fetch data for any of the provided tickers.")

    df = pd.DataFrame(stats)

    # Build markdown
    md = "# Peer Comparison Analysis\n\n"
    if target_ticker:
        md += f"**Target Company**: {target_ticker}\n\n"
    md += df.to_markdown(index=False)
    md += "\n"

    # Save
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    filename = f"peer_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    file_path = os.path.join(data_dir, filename)

    with open(file_path, "w") as f:
        f.write(md)

    metadata: Dict[str, Any] = {
        "source": "Yahoo Finance (Market Data + Info)",
        "peers_analyzed": [s["Ticker"] for s in stats],
        "methodology": {
            "1y Change (%)": "((Last Close - First Close) / First Close) * 100 over 1y period",
            "Peer Selection": "Data-driven: sector/industry match + market-cap proximity filter",
        },
    }

    return file_path, metadata


def _fmt(value: Any) -> str:
    """Format large numbers for readability."""
    if value is None or value == "N/A":
        return "N/A"
    try:
        num = float(value)
        if abs(num) >= 1e12:
            return f"${num / 1e12:.1f}T"
        elif abs(num) >= 1e9:
            return f"${num / 1e9:.1f}B"
        elif abs(num) >= 1e6:
            return f"${num / 1e6:.0f}M"
        else:
            return f"${num:,.0f}"
    except (ValueError, TypeError):
        return str(value)


def _round(value: Any) -> str:
    """Round a float to 1 decimal or return N/A."""
    if value is None or value == "N/A":
        return "N/A"
    try:
        return f"{float(value):.1f}"
    except (ValueError, TypeError):
        return str(value)


def _pct(value: Any) -> str:
    """Format a decimal as percentage."""
    if value is None or value == "N/A":
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%"
    except (ValueError, TypeError):
        return str(value)

