import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from typing import Tuple, Dict, Any, List
from src.shared.web_search import search_web, deep_search
from src.shared.reliability import cached_retry_call


class FinancialsError(Exception):
    """Custom exception for financial data errors."""
    pass


def _load_financial_snapshot(ticker: str) -> Dict[str, Any]:
    """Fetch and cache the raw Yahoo Finance payload used by the financials tool."""
    def _load() -> Dict[str, Any]:
        stock = yf.Ticker(ticker)
        return {
            "info": stock.info,
            "income_stmt": stock.income_stmt,
            "balance_sheet": stock.balance_sheet,
            "cashflow": stock.cashflow,
            "quarterly_income_stmt": stock.quarterly_income_stmt,
            "quarterly_balance_sheet": stock.quarterly_balance_sheet,
            "quarterly_cashflow": stock.quarterly_cashflow,
            "sustainability": stock.sustainability,
        }

    return cached_retry_call(
        "yfinance_financials",
        {"ticker": ticker},
        _load,
        ttl_seconds=1800,
        attempts=3,
    )


def fetch_financial_metrics(ticker: str) -> Tuple[str, Dict[str, Any]]:
    """
    Fetches key financial metrics, annual + quarterly statements, and cash flow.

    Includes:
      - Key ratios from yfinance .info
      - Annual income statement & balance sheet (last 2 years)
      - Quarterly income statement (last 4 quarters) + TTM approximation
      - Annual & quarterly cash flow statements
      - ESG data (if available)

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL').

    Returns:
        Tuple of (absolute path to saved Markdown file, metadata dict).

    Raises:
        FinancialsError: If the ticker is invalid or data fetch fails.
    """
    try:
        snapshot = _load_financial_snapshot(ticker)
        info = snapshot["info"]

        # --- Key Ratios ---
        # yfinance 404s often result in empty info or missing keys, but no exception.
        
        # DEBUG: Print info to see what's actually returned on 404
        print(f"DEBUG info keys: {list(info.keys())}")
        print(f"DEBUG marketCap: {info.get('marketCap')}")
        print(f"DEBUG totalRevenue: {info.get('totalRevenue')}")

        pe_ratio = info.get('trailingPE', 'N/A')
        forward_pe = info.get('forwardPE', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        debt_to_equity = info.get('debtToEquity', 'N/A')
        roe = info.get('returnOnEquity', 'N/A')
        revenue_growth = info.get('revenueGrowth', 'N/A')
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
        enterprise_value = info.get('enterpriseValue', 'N/A')
        ev_to_ebitda = info.get('enterpriseToEbitda', 'N/A')
        gross_margins = info.get('grossMargins', 'N/A')
        operating_margins = info.get('operatingMargins', 'N/A')
        profit_margins = info.get('profitMargins', 'N/A')
        free_cashflow = info.get('freeCashflow', 'N/A')
        operating_cashflow = info.get('operatingCashflow', 'N/A')
        total_revenue = info.get('totalRevenue', 'N/A')
        fiscal_year_end = info.get('lastFiscalYearEnd', 'N/A')
        # --- Annual Statements (last 2 years) ---
        income_stmt_src = snapshot["income_stmt"]
        balance_sheet_src = snapshot["balance_sheet"]
        cash_flow_src = snapshot["cashflow"]
        income_stmt = income_stmt_src.iloc[:, :2] if not income_stmt_src.empty else pd.DataFrame()
        balance_sheet = balance_sheet_src.iloc[:, :2] if not balance_sheet_src.empty else pd.DataFrame()
        cash_flow = cash_flow_src.iloc[:, :2] if not cash_flow_src.empty else pd.DataFrame()
        

        
        # CRITICAL CHECK:
        # If we have market data (info) but NO financial statements, it's likely a data provider issue 
        # (common for recent IPOs). In this case, we MUST trigger the fallback to get actual reports.
        if income_stmt.empty and balance_sheet.empty:
            raise FinancialsError("Financial statements are empty (recent IPO?).")

        # --- Quarterly Statements (last 4 quarters) ---
        q_income_stmt_src = snapshot["quarterly_income_stmt"]
        q_balance_sheet_src = snapshot["quarterly_balance_sheet"]
        q_cash_flow_src = snapshot["quarterly_cashflow"]
        q_income_stmt = q_income_stmt_src.iloc[:, :4] if not q_income_stmt_src.empty else pd.DataFrame()
        q_balance_sheet = q_balance_sheet_src.iloc[:, :1] if not q_balance_sheet_src.empty else pd.DataFrame()
        q_cash_flow = q_cash_flow_src.iloc[:, :4] if not q_cash_flow_src.empty else pd.DataFrame()

        # --- TTM Approximation (sum of last 4 quarterly income periods) ---
        ttm_income = pd.DataFrame()
        if not q_income_stmt.empty and q_income_stmt.shape[1] >= 4:
            numeric_cols = q_income_stmt.select_dtypes(include="number")
            if not numeric_cols.empty:
                ttm_series = numeric_cols.sum(axis=1)
                ttm_income = ttm_series.to_frame(name="TTM (sum of last 4Q)")

        ttm_cashflow = pd.DataFrame()
        if not q_cash_flow.empty and q_cash_flow.shape[1] >= 4:
            numeric_cols = q_cash_flow.select_dtypes(include="number")
            if not numeric_cols.empty:
                ttm_cf_series = numeric_cols.sum(axis=1)
                ttm_cashflow = ttm_cf_series.to_frame(name="TTM (sum of last 4Q)")
        
        # --- ESG Data ---
        try:
            esg_df = snapshot["sustainability"]
            if esg_df is not None and not esg_df.empty:
                esg_data = esg_df.to_markdown()
            else:
                esg_data = "ESG data not available."
        except Exception:
            esg_data = "ESG data fetch failed."
        
        # --- Format as Markdown ---
        md = f"# Financial Analysis for {ticker}\n\n"

        md += "## Key Ratios\n"
        md += f"- **Current Price**: {current_price}\n"
        md += f"- **Market Cap**: {market_cap}\n"
        md += f"- **Enterprise Value**: {enterprise_value}\n"
        md += f"- **Trailing P/E**: {pe_ratio}\n"
        md += f"- **Forward P/E**: {forward_pe}\n"
        md += f"- **EV/EBITDA**: {ev_to_ebitda}\n"
        md += f"- **Debt/Equity**: {debt_to_equity}\n"
        md += f"- **ROE**: {roe}\n"
        md += f"- **Revenue Growth (YoY)**: {revenue_growth}\n"
        md += f"- **Gross Margin**: {gross_margins}\n"
        md += f"- **Operating Margin**: {operating_margins}\n"
        md += f"- **Profit Margin**: {profit_margins}\n"
        md += f"- **Total Revenue (TTM)**: {total_revenue}\n"
        md += f"- **Operating Cash Flow**: {operating_cashflow}\n"
        md += f"- **Free Cash Flow**: {free_cashflow}\n"
        md += f"- **Fiscal Year End**: {fiscal_year_end}\n\n"

        md += "## ESG Data\n"
        md += f"{esg_data}\n\n"
        
        # --- Annual Statements ---
        md += "## Annual Financial Statements\n"

        if not income_stmt.empty:
            md += "### Annual Income Statement\n"
            md += income_stmt.to_markdown() + "\n\n"

        if not balance_sheet.empty:
            md += "### Annual Balance Sheet\n"
            md += balance_sheet.to_markdown() + "\n\n"

        if not cash_flow.empty:
            md += "### Annual Cash Flow Statement\n"
            md += cash_flow.to_markdown() + "\n\n"

        # --- Quarterly Statements ---
        md += "## Quarterly Financial Statements (Last 4 Quarters)\n"

        if not q_income_stmt.empty:
            md += "### Quarterly Income Statement\n"
            md += q_income_stmt.to_markdown() + "\n\n"

        if not ttm_income.empty:
            md += "### TTM Income Statement (Sum of Last 4 Quarters)\n"
            md += ttm_income.to_markdown() + "\n\n"

        if not q_balance_sheet.empty:
            md += "### Latest Quarterly Balance Sheet\n"
            md += q_balance_sheet.to_markdown() + "\n\n"

        if not q_cash_flow.empty:
            md += "### Quarterly Cash Flow Statement\n"
            md += q_cash_flow.to_markdown() + "\n\n"

        if not ttm_cashflow.empty:
            md += "### TTM Cash Flow (Sum of Last 4 Quarters)\n"
            md += ttm_cashflow.to_markdown() + "\n\n"
        
        # --- Save ---
        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)
        filename = f"{ticker}_financials_{datetime.now().strftime('%Y%m%d')}.md"
        file_path = os.path.join(data_dir, filename)

        with open(file_path, "w") as f:
            f.write(md)

        metadata: Dict[str, Any] = {
            "source": "Yahoo Finance API",
            "data_date": datetime.now().strftime('%Y-%m-%d'),
            "methodology": {
                "Trailing P/E": "Current Price / Trailing 12-month EPS",
                "Forward P/E": "Current Price / Estimated Future EPS",
                "Debt/Equity": "Total Debt / Total Shareholder Equity",
                "ROE": "Net Income / Shareholder Equity",
                "Revenue Growth": "Year-over-Year Revenue Growth Rate",
                "TTM": "Sum of last 4 reported quarterly periods",
                "ESG": "Sustainalytics ESG Risk Ratings (if available)",
            },
        }

        return file_path, metadata

    except FinancialsError as e:
        print(f"Financials check failed for {ticker} ({e}). Attempting fallback...")
        return fetch_financials_fallback(ticker)
    except Exception as e:
        print(f"yfinance failed for {ticker} ({e}). Attempting fallback to web search...")
        return fetch_financials_fallback(ticker)


def fetch_financials_fallback(ticker: str) -> Tuple[str, Dict[str, Any]]:
    """
    Fallback method to find financial data via web search when yfinance fails.
    Prioritizes "Investor Relations" pages for public companies, then generic estimates.
    """
    print(f"Fallback: Searching for financial data for {ticker}...")
    
    # Strategy 1: Investor Relations / Official Reports (for public companies like CRWV)
    ir_queries = [
        f"{ticker} Investor Relations financial results 2024 2025",
        f"{ticker} quarterly earnings report 10-Q 10-K",
        f"{ticker} financial statements revenue net income",
    ]
    
    # Strategy 2: Generic Estimates (for private companies or missing data)
    generic_queries = [
        f"{ticker} revenue valuation last funding round",
        f"{ticker} annual revenue growth rate",
    ]
    
    # Execute IR search first
    ir_results = deep_search(
        queries=ir_queries,
        urls_per_query=2,
        max_workers=5,
        max_chars_per_url=6000,
        search_delay=0.5,
    )
    
    # Check if IR search yielded good results (look for keywords in titles)
    good_ir_hits = 0
    for r in ir_results:
        title_lower = r.get("title", "").lower()
        if any(x in title_lower for x in ["investor", "report", "earnings", "results", "quarter", "10-k", "10-q"]):
            good_ir_hits += 1
            
    print(f"Fallback: Found {good_ir_hits} high-quality IR results.")
    
    # Execute generic search if IR results are weak, or just add them for breadth
    generic_results = []
    if good_ir_hits < 3:
         generic_results = deep_search(
            queries=generic_queries,
            urls_per_query=2,
            max_workers=5,
            max_chars_per_url=4000,
            search_delay=0.5,
        )
         
    all_results = ir_results + generic_results
    
    if not all_results:
        raise FinancialsError(f"Fallback search failed to find any financial data for {ticker}")

    # Compile into Markdown
    md = f"# Financial Analysis for {ticker} (Web Search Fallback)\n\n"
    md += "> **Note**: Official data via API was unavailable. These are search results from Investor Relations and news sources.\n\n"
    
    dt = datetime.now().strftime('%Y-%m-%d')
    
    md += "## Financial Search Results\n\n"
    
    for r in all_results:
        title = r.get("title", "No Title")
        url = r.get("url", "#")
        # Use full text if available and reasonable length, else snippet
        content = r.get("full_text")
        if not content:
            content = r.get("snippet", "No content available.")
        
        # Truncate very long content for the report to avoid massive file
        if len(content) > 3000:
            content = content[:3000] + "\n... [truncated]"
            
        md += f"### [{title}]({url})\n\n"
        md += f"{content}\n\n---\n\n"

    # Save
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    filename = f"{ticker}_financials_fallback_{datetime.now().strftime('%Y%m%d')}.md"
    file_path = os.path.join(data_dir, filename)

    with open(file_path, "w") as f:
        f.write(md)
        
    metadata = {
        "source": "Web Search (Fallback)",
        "data_date": dt,
        "methodology": "Aggregated search results from Investor Relations and News",
        "is_fallback": True
    }
    
    print(f"Fallback successful: Saved to {file_path}")
    return file_path, metadata
