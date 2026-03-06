"""
Company Profile Tool — fetches identity, ownership, and institutional holder data.

Uses yfinance to pull:
  - longBusinessSummary, sector, industry, employees, website
  - Major holders (insider %, institution %)
  - Top institutional holders (name, shares, %, value)
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from typing import Tuple, Dict, Any


class CompanyProfileError(Exception):
    """Custom exception for company profile errors."""
    pass


def fetch_company_profile(ticker: str) -> Tuple[str, Dict[str, Any]]:
    """
    Fetches company identity, ownership structure, and institutional holders.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL').

    Returns:
        Tuple of (absolute path to saved Markdown file, metadata dict).

    Raises:
        CompanyProfileError: If the ticker is invalid or data fetch fails.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or info.get("regularMarketPrice") is None:
            # Fallback check — some tickers have sparse info
            if not info.get("longBusinessSummary") and not info.get("shortName"):
                raise CompanyProfileError(
                    f"No profile data found for ticker '{ticker}'."
                )

        # --- Company Identity ---
        company_name = info.get("longName") or info.get("shortName", ticker)
        summary = info.get("longBusinessSummary", "Not available.")
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        employees = info.get("fullTimeEmployees", "N/A")
        website = info.get("website", "N/A")
        market_cap = info.get("marketCap", "N/A")
        country = info.get("country", "N/A")
        exchange = info.get("exchange", "N/A")

        md = f"# Company Profile: {company_name} ({ticker})\n\n"
        md += "## Company Overview\n"
        md += f"- **Name**: {company_name}\n"
        md += f"- **Sector**: {sector}\n"
        md += f"- **Industry**: {industry}\n"
        md += f"- **Country**: {country}\n"
        md += f"- **Exchange**: {exchange}\n"
        md += f"- **Employees**: {employees}\n"
        md += f"- **Website**: {website}\n"
        md += f"- **Market Cap**: {_format_number(market_cap)}\n\n"
        md += "## Business Description\n"
        md += f"{summary}\n\n"

        # --- Major Holders ---
        md += "## Ownership Structure\n"
        try:
            major_holders = stock.major_holders
            if major_holders is not None and not major_holders.empty:
                md += "### Major Holders\n"
                md += major_holders.to_markdown(index=False) + "\n\n"
            else:
                md += "Major holders data not available.\n\n"
        except Exception:
            md += "Major holders data fetch failed.\n\n"

        # --- Institutional Holders ---
        try:
            inst_holders = stock.institutional_holders
            if inst_holders is not None and not inst_holders.empty:
                md += "### Top Institutional Holders\n"
                # Format for readability
                display_df = inst_holders.head(15).copy()
                if "Date Reported" in display_df.columns:
                    display_df["Date Reported"] = pd.to_datetime(
                        display_df["Date Reported"], errors="coerce"
                    ).dt.strftime("%Y-%m-%d")
                if "Value" in display_df.columns:
                    display_df["Value"] = display_df["Value"].apply(_format_number)
                if "Shares" in display_df.columns:
                    display_df["Shares"] = display_df["Shares"].apply(_format_number)
                if "pctHeld" in display_df.columns:
                    display_df["pctHeld"] = display_df["pctHeld"].apply(
                        lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
                    )
                md += display_df.to_markdown(index=False) + "\n\n"
            else:
                md += "Institutional holders data not available.\n\n"
        except Exception:
            md += "Institutional holders data fetch failed.\n\n"

        # --- Mutual Fund Holders (bonus) ---
        try:
            mf_holders = stock.mutualfund_holders
            if mf_holders is not None and not mf_holders.empty:
                md += "### Top Mutual Fund Holders\n"
                display_mf = mf_holders.head(10).copy()
                if "Date Reported" in display_mf.columns:
                    display_mf["Date Reported"] = pd.to_datetime(
                        display_mf["Date Reported"], errors="coerce"
                    ).dt.strftime("%Y-%m-%d")
                if "Value" in display_mf.columns:
                    display_mf["Value"] = display_mf["Value"].apply(_format_number)
                if "Shares" in display_mf.columns:
                    display_mf["Shares"] = display_mf["Shares"].apply(_format_number)
                md += display_mf.to_markdown(index=False) + "\n\n"
        except Exception:
            pass  # Not critical

        # --- Save ---
        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)
        filename = f"{ticker}_profile_{datetime.now().strftime('%Y%m%d')}.md"
        file_path = os.path.join(data_dir, filename)

        with open(file_path, "w") as f:
            f.write(md)

        metadata: Dict[str, Any] = {
            "source": "Yahoo Finance API (Ticker.info, major_holders, institutional_holders)",
            "company_name": company_name,
            "sector": sector,
            "industry": industry,
            "market_cap": market_cap,
        }

        return file_path, metadata

    except CompanyProfileError:
        raise
    except Exception as e:
        raise CompanyProfileError(
            f"Failed to fetch company profile for {ticker}: {str(e)}"
        )


def _format_number(value: Any) -> str:
    """Formats large numbers for readability (e.g., 49.9B, 1.2M)."""
    if value is None or value == "N/A" or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        num = float(value)
        if abs(num) >= 1e12:
            return f"${num / 1e12:.2f}T"
        elif abs(num) >= 1e9:
            return f"${num / 1e9:.2f}B"
        elif abs(num) >= 1e6:
            return f"${num / 1e6:.2f}M"
        elif abs(num) >= 1e3:
            return f"${num / 1e3:.1f}K"
        else:
            return f"{num:,.0f}"
    except (ValueError, TypeError):
        return str(value)
