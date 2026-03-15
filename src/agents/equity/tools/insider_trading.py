"""
Insider Trading Tool (Form 4) — fetches and summarizes the latest insider transactions.

Uses the free FMP API endpoint to aggregate recent executive buys and sells.
"""

import os
import requests
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any, cast

class InsiderTradingError(Exception):
    """Custom exception for insider trading fetch errors."""
    pass

def fetch_insider_activity(ticker: str) -> Tuple[str, Dict[str, Any]]:
    """
    Fetches the latest insider trading activity from FMP, parses it, and
    compiles a summary table.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL').

    Returns:
        Tuple of (absolute path to saved Markdown file, metadata dict).

    Raises:
        InsiderTradingError: If the API call fails.
    """
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        print("Warning: FMP_API_KEY not found in environment. Skipping Insider Trading.")
        return "", {}

    url = f"https://financialmodelingprep.com/api/v4/insider-trading?symbol={ticker}&page=0&apikey={api_key}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data or not isinstance(data, list):
            md = f"# Insider Trading Activity: {ticker}\n\nNo recent insider trading data found for {ticker}."
            file_path = _save_md(ticker, md)
            return file_path, {"source": "FMP (Empty)"}

        # Convert to DataFrame for easy aggregation
        df = pd.DataFrame(data)
        
        # We need specific columns. If they are missing, handle gracefully.
        required_cols = ["transactionDate", "reportingName", "typeOfOwner", "transactionType", "securitiesTransacted", "price"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = "Unknown" if col in ["reportingName", "typeOfOwner", "transactionType"] else 0

        # Filter out rows where transactionType is missing or not a standard Buy/Sell
        # P-Purchase, S-Sale
        df = df[df["transactionType"].isin(["P-Purchase", "S-Sale"])]
        
        if df.empty:
            md = f"# Insider Trading Activity: {ticker}\n\nNo recent open-market buys/sells found for {ticker}."
            file_path = _save_md(ticker, md)
            return file_path, {"source": "FMP (No relevant trades)"}

        # Ensure numeric columns
        securities_series = cast(pd.Series, pd.to_numeric(df["securitiesTransacted"], errors="coerce"))
        price_series = cast(pd.Series, pd.to_numeric(df["price"], errors="coerce"))
        df["securitiesTransacted"] = securities_series.fillna(0)
        df["price"] = price_series.fillna(0)
        df["transactionValue"] = df["securitiesTransacted"] * df["price"]

        # Aggregate by reporting person
        summary_rows = []
        for name, group in df.groupby("reportingName"):
            role = cast(pd.Series, group["typeOfOwner"]).iloc[0]
            
            buys = group[group["transactionType"] == "P-Purchase"]
            sells = group[group["transactionType"] == "S-Sale"]
            
            buy_vol = buys["transactionValue"].sum()
            sell_vol = sells["transactionValue"].sum()
            net_vol = buy_vol - sell_vol
            
            if buy_vol > 0 or sell_vol > 0:
                summary_rows.append({
                    "Insider Name": name,
                    "Title/Role": role,
                    "Total Bought": f"${buy_vol:,.0f}",
                    "Total Sold": f"${sell_vol:,.0f}",
                    "Net Value": f"${net_vol:,.0f}",
                    "Latest Trade": group["transactionDate"].max()
                })

        summary_df = pd.DataFrame(summary_rows)
        
        if summary_df.empty:
            md = f"# Insider Trading Activity: {ticker}\n\nNo significant buy/sell volume found in the latest batch."
            file_path = _save_md(ticker, md)
            return file_path, {"source": "FMP (No volume)"}

        # Sort by Net Value absolute to see largest movers
        summary_df["_net_abs"] = summary_df["Net Value"].apply(lambda x: abs(float(str(x).replace("$", "").replace(",", ""))))
        summary_df = summary_df.sort_values(by="_net_abs", ascending=False).drop(columns=["_net_abs"]).head(15)

        total_bought = summary_df["Total Bought"].apply(lambda x: float(str(x).replace("$", "").replace(",", ""))).sum()
        total_sold = summary_df["Total Sold"].apply(lambda x: float(str(x).replace("$", "").replace(",", ""))).sum()

        md = f"# Insider Trading Activity: {ticker}\n\n"
        md += "## Executive Summary (Latest Reported Batch)\n"
        md += f"- **Total Aggregate Buying**: ${total_bought:,.0f}\n"
        md += f"- **Total Aggregate Selling**: ${total_sold:,.0f}\n"
        md += f"- **Net Insider Flow**: ${total_bought - total_sold:,.0f}\n\n"
        
        md += "## Top Insider Transactions\n"
        md += summary_df.to_markdown(index=False) + "\n"

        file_path = _save_md(ticker, md)
        
        metadata = {
            "source": "Financial Modeling Prep Free API (Form 4)",
            "total_bought": total_bought,
            "total_sold": total_sold,
        }
        return file_path, metadata

    except Exception as e:
        raise InsiderTradingError(f"Failed to fetch insider trading for {ticker}: {e}")


def _save_md(ticker: str, content: str) -> str:
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    filename = f"{ticker}_insider_trading_{datetime.now().strftime('%Y%m%d')}.md"
    file_path = os.path.join(data_dir, filename)

    with open(file_path, "w") as f:
        f.write(content)
        
    return file_path
