import pandas as pd
import numpy as np


def validate_market_data(df: pd.DataFrame, period: str = "5M") -> pd.DataFrame:
    """
    Validates and cleans time-series OHLCV market data.
    
    Checks:
    1. Not empty
    2. Has required columns: Open, High, Low, Close, Volume
    3. Chronological sorting
    4. Minimum data length (15+ days for 1M, 40+ days for 5M)
    5. Clean missing / invalid numbers without lookahead imputation
    """
    if df is None or df.empty:
        raise ValueError("Market data is empty or unavailable.")

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required OHLCV column '{col}' is missing from market data.")

    clean_df = df[required_cols].copy()

    # Convert to numeric
    for col in required_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # Drop NaNs or infinities
    clean_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    clean_df.dropna(inplace=True)

    # Chronological sort
    clean_df.sort_index(ascending=True, inplace=True)

    # Check minimum trading days
    min_days = 15 if period.upper() == "1M" else 40
    if len(clean_df) < min_days:
        raise ValueError(
            f"Insufficient trading data: received {len(clean_df)} trading days, "
            f"minimum required for {period} analysis is {min_days} days."
        )

    # Validate price positive
    if (clean_df["Close"] <= 0).any():
        raise ValueError("Invalid non-positive price values found in stock data.")

    return clean_df
