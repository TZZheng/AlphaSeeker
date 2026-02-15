import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from typing import Tuple, Dict, Any


class FinancialsError(Exception):
    """Custom exception for financial data errors."""
    pass


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
        stock = yf.Ticker(ticker)
        info = stock.info

        # --- Key Ratios ---
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
        income_stmt = stock.income_stmt.iloc[:, :2] if not stock.income_stmt.empty else pd.DataFrame()
        balance_sheet = stock.balance_sheet.iloc[:, :2] if not stock.balance_sheet.empty else pd.DataFrame()
        cash_flow = stock.cashflow.iloc[:, :2] if not stock.cashflow.empty else pd.DataFrame()

        # --- Quarterly Statements (last 4 quarters) ---
        q_income_stmt = stock.quarterly_income_stmt.iloc[:, :4] if not stock.quarterly_income_stmt.empty else pd.DataFrame()
        q_balance_sheet = stock.quarterly_balance_sheet.iloc[:, :1] if not stock.quarterly_balance_sheet.empty else pd.DataFrame()
        q_cash_flow = stock.quarterly_cashflow.iloc[:, :4] if not stock.quarterly_cashflow.empty else pd.DataFrame()

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
            esg_df = stock.sustainability
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

    except FinancialsError:
        raise
    except Exception as e:
        raise FinancialsError(f"Failed to fetch financials for {ticker}. Error: {str(e)}")
