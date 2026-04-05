import pandas as pd
import numpy as np
from typing import cast

def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """Calculates Simple Moving Average."""
    return cast(pd.Series, data.rolling(window=window).mean())

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """Calculates Relative Strength Index."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    return cast(pd.Series, 100 - (100 / (1 + rs)))

def calculate_macd(data: pd.Series, slow: int = 26, fast: int = 12, signal: int = 9):
    """Calculates MACD, Signal line, and Histogram."""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram
