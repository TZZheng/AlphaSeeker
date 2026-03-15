import yfinance as yf
import pandas as pd
import os
from datetime import datetime

from src.shared.reliability import cached_retry_call

class MarketDataError(Exception):
    """Custom exception for market data errors."""
    pass

def fetch_historical_data(ticker: str, period: str = "1y") -> str:
    """
    Fetches historical stock data for the given ticker and period.
    
    Args:
        ticker (str): The stock symbol (e.g., 'AAPL').
        period (str): The period to download (e.g., '1y', '5d', 'max').
        
    Returns:
        str: Absolute path to the saved CSV file.
        
    Raises:
        MarketDataError: If the ticker is invalid or data download fails.
    """
    try:
        hist = cached_retry_call(
            "yfinance_history",
            {"ticker": ticker, "period": period},
            lambda: yf.Ticker(ticker).history(period=period),
            ttl_seconds=900,
            attempts=3,
        )
        
        if hist.empty:
            raise MarketDataError(f"No data found for ticker '{ticker}' with period '{period}'.")
        
        # Ensure data directory exists
        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save to CSV
        filename = f"{ticker}_{period}_{datetime.now().strftime('%Y%m%d')}.csv"
        file_path = os.path.join(data_dir, filename)
        hist.to_csv(file_path)
        
        return file_path
        
    except Exception as e:
        if isinstance(e, MarketDataError):
            raise
        raise MarketDataError(f"Failed to fetch data for {ticker}: {str(e)}")
