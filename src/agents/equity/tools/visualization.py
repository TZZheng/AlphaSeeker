import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

class VisualizationError(Exception):
    """Custom exception for visualization errors."""
    pass

def plot_price_history(data_path: str, ticker: str) -> str:
    """
    Plots the closing price history from a CSV file.
    
    Args:
        data_path (str): The absolute path to the CSV file.
        ticker (str): The stock ticker (for title).
        
    Returns:
        str: Absolute path to the saved chart image (PNG).
        
    Raises:
        VisualizationError: If plotting fails or data is invalid.
    """
    try:
        if not os.path.exists(data_path):
            raise VisualizationError(f"Data file not found at {data_path}")
            
        df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
        
        if 'Close' not in df.columns:
             # yfinance might save as 'Adj Close' or 'Close' depending on version/settings, handle both
             if 'Adj Close' in df.columns:
                 plot_col = 'Adj Close'
             else:
                 raise VisualizationError("CSV does not contain 'Close' or 'Adj Close' column.")
        else:
            plot_col = 'Close'

        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[plot_col], label=f'{ticker} Price')
        plt.title(f"{ticker} Price History")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        
        # Ensure charts directory exists
        charts_dir = os.path.join(os.getcwd(), "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        filename = f"{ticker}_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        chart_path = os.path.join(charts_dir, filename)
        
        plt.savefig(chart_path)
        plt.close()
        
        return chart_path

    except Exception as e:
        if isinstance(e, VisualizationError):
            raise
        raise VisualizationError(f"Failed to generate chart: {str(e)}")
