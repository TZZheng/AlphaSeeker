"""
Peer Analysis Tool — data-driven peer discovery and comparison.

Uses web search and LLM extraction to identify competitors, then validates them
via yfinance to categorize them into Giants, Peers, and Disruptors.
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.messages import HumanMessage

from src.agents.equity.tools.web_search import search_web
from src.shared.llm_manager import get_llm

# Use a cheap/fast model for extraction
MODEL_PEER_EXTRACTION = "sf/Qwen/Qwen3-14B"  # or similar fast model


class PeerAnalysisError(Exception):
    """Custom exception for peer analysis errors."""
    pass


# --- Well-known sector peer pools (Fallback only) ---
INDUSTRY_PEER_CANDIDATES: Dict[str, List[str]] = {
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
    "Cloud Infrastructure": [
        "AMZN", "MSFT", "GOOGL", "ORCL", "IBM", "SNOW", "NET",
    ],
}

# Fallback: broad Technology sector peers
TECHNOLOGY_FALLBACK = [
    "NVDA", "AMD", "PLTR", "SNOW", "DDOG", "CRWD", "NET", "CRM",
    "NOW", "PANW", "SMCI", "AI", "PATH", "MDB", "CFLT",
]


def extract_peers_from_text(text: str) -> List[str]:
    """
    Extracts potential competitor names/tickers from raw research text using an LLM.

    Args:
        text: Raw text containing company mentions.

    Returns:
        List of competitor names/tickers (strings).
    """
    if not text:
        return []

    prompt = f"""
You are an expert financial analyst. 
Identify the KEY PUBLIC AND PRIVATE COMPETITORS mentioned in the following text.
Focus on companies that are direct rivals, larger incumbents, or emerging disruptors.

TEXT:
{text[:15000]}  # Truncate to avoid context limit if necessary

Return ONLY a comma-separated list.
Rules:
1. If the company is PUBLIC, provide the VALID TICKER (e.g. MSFT, GOOGL, AMD).
2. If the company is PRIVATE, provide the full name (e.g. OpenAI, Anthropic).
3. Do NOT include the target company itself if mentioned.

Example: "MSFT, GOOGL, OpenAI, Anthropic"
"""
    try:
        response = get_llm(MODEL_PEER_EXTRACTION).invoke([HumanMessage(content=prompt)])
        content = response.content.replace("\n", "").strip()
        # Clean up list
        peers = [p.strip() for p in content.split(",") if p.strip()]
        # Remove empty strings and artifacts
        cleaned = []
        for p in peers:
            clean_p = p.strip(" .-_*")
            if len(clean_p) > 1:
                cleaned.append(clean_p)
        return list(set(cleaned))  # Deduplicate
    except Exception as e:
        print(f"Warning: Peer extraction failed: {e}")
        return []


def _find_ticker_from_web(company_name: str) -> Optional[str]:
    """Attempts to find a ticker for a company name via web search."""
    try:
        print(f"  Searching for ticker for '{company_name}'...")
        # Simple search
        results = search_web(f"stock ticker symbol for {company_name}", max_results=1)
        if not results:
            return None
        
        snippet = (results[0].get("body", "") + " " + results[0].get("title", "")).upper()
        
        # Heuristic: look for (NASDAQ: XXX) or (NYSE: XXX) or "Ticker: XXX"
        # Since we don't want to use another LLM call to save time/cost, 
        # let's ask the cheap LLM to extract it quickly.
        
        prompt = f"""
        Extract the stock ticker for {company_name} from this text, or return "None".
        Text: {snippet[:500]}
        Return ONLY the ticker (e.g. MSFT).
        """
        response = get_llm(MODEL_PEER_EXTRACTION).invoke([HumanMessage(content=prompt)])
        ticker = response.content.strip().strip(".").split()[0] # basic cleanup
        
        if len(ticker) <= 5 and ticker.isalpha() and ticker != "NONE":
            print(f"  Found ticker: {ticker}")
            return ticker
        return None
    except Exception as e:
        print(f"  Ticker search failed: {e}")
        return None


def evaluate_candidates(
    candidates: List[str],
    target_ticker: str,
) -> Dict[str, List[str]]:
    """
    Validates candidates via yfinance and categorizes them into Giants, Peers, and Disruptors.

    Args:
        candidates: List of company names or tickers.
        target_ticker: The ticker of the subject company.

    Returns:
        Dict with keys "Giants", "Peers", "Disruptors".
    """
    # 1. Get Target Market Cap
    target_mcap = None
    try:
        t_info = yf.Ticker(target_ticker).info
        target_mcap = t_info.get("marketCap")
    except Exception:
        pass

    if not target_mcap:
        # If we can't get target market cap, just return flat list as "Peers"
        print(f"Warning: Could not fetch market cap for {target_ticker}")
        valid_peers = []
        for c in candidates:
             if _is_valid_ticker(c):
                 valid_peers.append(c)
        return {"Peers": valid_peers[:10], "Giants": [], "Disruptors": []}

    # 2. Categorize
    giants = []
    peers = []
    disruptors = []

    # If no candidates provided, try fallback based on target's industry
    if not candidates:
        print("No candidates provided, attempting industry fallback...")
        industry = t_info.get("industry")
        if industry and industry in INDUSTRY_PEER_CANDIDATES:
            candidates = INDUSTRY_PEER_CANDIDATES[industry]
        else:
            candidates = TECHNOLOGY_FALLBACK

    # Deduplicate and remove target
    unique_candidates = list(set([c.upper() for c in candidates]))
    
    # Remove target name variations
    target_ticker_upper = target_ticker.upper()
    target_name_parts = t_info.get("shortName", "").upper().split()
    
    filtered_candidates = []
    for c in unique_candidates:
        if c == target_ticker_upper: continue
        # Simple name check
        if any(part in c for part in target_name_parts if len(part) > 3): continue
        filtered_candidates.append(c)

    print(f"Evaluating {len(filtered_candidates)} peer candidates...")

    for cand in filtered_candidates:
        try:
            # Check if it's a known private co or just a weird name
            if "PRIVATE" in cand:
                 disruptors.append(cand)
                 continue

            # Try as ticker
            stock = yf.Ticker(cand)
            info = stock.info
            mcap = info.get("marketCap")
            
            # If valid ticker found
            if mcap and mcap > 0:
                ratio = mcap / target_mcap
                if ratio > 5.0:
                    giants.append(cand)
                elif ratio < 0.1:
                    disruptors.append(cand) 
                else:
                    peers.append(cand)
                continue

            # If failed, it might be a name ("Microsoft")
            # Try to resolve to ticker
            resolved_ticker = None
            if len(cand) > 5 or " " in cand:
                resolved_ticker = _find_ticker_from_web(cand)
            
            if resolved_ticker:
                # Retry with resolved ticker
                stock = yf.Ticker(resolved_ticker)
                info = stock.info
                mcap = info.get("marketCap")
                if mcap and mcap > 0:
                    cand = resolved_ticker # Use the ticker
                    ratio = mcap / target_mcap
                    if ratio > 5.0:
                        giants.append(cand)
                    elif ratio < 0.1:
                        disruptors.append(cand) 
                    else:
                        peers.append(cand)
                    continue

            # If still no valid ticker, assume private disruptor
            if len(cand) > 3: # Ignore very short garbage
                disruptors.append(f"{cand.title()} (Private)")

        except Exception as e:
             # print(f"  Error checking {cand}: {e}")
             if len(cand) > 3:
                disruptors.append(f"{cand.title()} (Private)")

    return {
        "Giants": giants[:3],      # Top 3 Giants
        "Peers": peers[:5],        # Top 5 Direct Peers
        "Disruptors": disruptors[:5] # Top 5 Disruptors/Startups
    }


def _is_valid_ticker(ticker: str) -> bool:
    try:
        info = yf.Ticker(ticker).info
        return "marketCap" in info
    except:
        return False


def fetch_peer_metrics(
    categorized_peers: Dict[str, List[str]],
    target_ticker: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Fetches comparative metrics for categorized peers.

    Args:
        categorized_peers: Dict with "Giants", "Peers", "Disruptors".
        target_ticker: The primary ticker being analyzed.

    Returns:
        Tuple of (absolute path to saved Markdown file, metadata dict).
    """
    stats: List[Dict[str, Any]] = []
    
    # Flatten list but keep track of category
    all_items = []
    for cat, ticker_list in categorized_peers.items():
        for t in ticker_list:
            all_items.append((t, cat))
    
    if target_ticker:
         # Prepend target
         all_items.insert(0, (target_ticker, "Target"))

    if not all_items:
         raise PeerAnalysisError("No peers to fetch metrics for.")

    for ticker_raw, category in all_items:
        ticker = ticker_raw.split(" (Private)")[0] # Clean up private names for display
        is_private = "(Private)" in ticker_raw

        try:
            if is_private:
                # Private company -> Minimal info
                stats.append({
                    "Ticker": "Private",
                    "Company": ticker,
                    "Type": category,
                    "Price": "N/A",
                    "Market Cap": "N/A",
                    "P/E (Fwd)": "N/A",
                    "Rev Growth": "N/A",
                    "Ev/EBITDA": "N/A",
                    "Gross Margin": "N/A",
                })
                continue

            # Public company
            stock = yf.Ticker(ticker)
            info = stock.info

            mcap = info.get('marketCap')
            # Skip if really broken
            if not mcap and category != "Target": 
                 continue

            stats.append({
                "Ticker": ticker,
                "Company": info.get("shortName", ticker),
                "Type": category,
                "Price": info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
                "Market Cap": _fmt(mcap),
                "P/E (Fwd)": _round(info.get('forwardPE')),
                "Rev Growth": _pct(info.get('revenueGrowth')),
                "Ev/EBITDA": _round(info.get('enterpriseToEbitda')),
                "Gross Margin": _pct(info.get('grossMargins')),
            })

        except Exception as e:
            print(f"Warning: Failed to fetch data for {ticker}: {e}")
            continue

    if not stats:
        raise PeerAnalysisError("Failed to fetch data for any tickers.")

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
        "peers_analyzed": [s["Ticker"] for s in stats if s["Ticker"] != "Private"],
        "private_competitors": [s["Company"] for s in stats if s["Ticker"] == "Private"],
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
