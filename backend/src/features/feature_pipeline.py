import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler


@dataclass
class PreparedData:
    feature_df: pd.DataFrame
    feature_names: List[str]
    X_tabular: np.ndarray          # 2D array for XGBoost: (samples, n_features)
    y_tabular: np.ndarray          # 1D array of next-day returns: (samples,)
    X_sequences: np.ndarray        # 3D array for LSTM/Transformer: (samples, seq_len, n_features)
    y_sequences: np.ndarray        # 1D array of next-day returns for sequence targets
    latest_tabular_input: np.ndarray # Latest feature vector for 1-step forecast: (1, n_features)
    latest_sequence_input: np.ndarray # Latest sequence for 1-step forecast: (1, seq_len, n_features)
    last_close_price: float
    last_date: str
    scaler: MinMaxScaler


class FeaturePipeline:
    """
    Feature Engineering Pipeline for Stock Forecasting.
    
    Generates compact time-series features strictly without lookahead bias:
    - OHLCV basic features
    - Daily Returns
    - Short-term moving averages (SMA 5, EMA 5)
    - Short-term Volatility (5-day rolling std of returns)
    - Compact RSI (relative strength index)
    - Target: Next-day return = (Close_{t+1} - Close_t) / Close_t
    """

    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length

    def _sanitize(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """Replace inf/-inf with NaN then forward/backward fill. Clip extreme outliers."""
        df = df.copy()
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0.0)
        # Clip return/volume-change columns at ±5 (500%) to guard against data errors
        for col in ["Return", "Volume_Change"]:
            if col in df.columns:
                df[col] = df[col].clip(-5.0, 5.0)
        return df

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        # Sanitize raw OHLCV first (NSE/BSE data can have zero-volume rows)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)
            data[col] = data[col].ffill().bfill()
        # Remove rows where Close is still NaN or zero
        data = data[(data["Close"].notna()) & (data["Close"] > 0)]

        # 1. Daily return (safe: previous close guaranteed > 0 after filter above)
        data["Return"] = data["Close"].pct_change()

        # 2. Moving Averages
        data["SMA_5"] = data["Close"].rolling(window=5, min_periods=1).mean()
        data["EMA_5"] = data["Close"].ewm(span=5, adjust=False).mean()

        # 3. Rolling Volatility
        data["Volatility_5"] = data["Return"].rolling(window=5, min_periods=1).std().fillna(0.0)

        # 4. RSI (10-period)
        delta = data["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=10, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=10, min_periods=1).mean()
        rs = gain / (loss + 1e-9)
        data["RSI_10"] = 100 - (100 / (1 + rs))

        # 5. Volume Change — guarded against zero-volume sessions
        safe_vol = data["Volume"].replace(0, np.nan).ffill().bfill().fillna(1.0)
        data["Volume_Change"] = safe_vol.pct_change().fillna(0.0)

        # Drop initial NaN row from pct_change
        data.dropna(subset=["Return"], inplace=True)
        return data

    def prepare_data(self, df: pd.DataFrame, test_ratio: float = 0.2) -> PreparedData:
        feat_df = self.calculate_features(df)
        
        feature_cols = [
            "Open", "High", "Low", "Close", "Volume",
            "Return", "SMA_5", "EMA_5", "Volatility_5", "RSI_10", "Volume_Change"
        ]
        
        # Target: Next-day Return
        # Shift -1 means target at row t is the return of t+1
        feat_df["Target_Return"] = feat_df["Return"].shift(-1)

        # Scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # We need historical samples that have known target for training/validation
        known_df = feat_df.dropna(subset=["Target_Return"]).copy()

        # Final sanitization pass — catches any residual inf/nan in derived features
        known_df = self._sanitize(known_df, feature_cols)
        feat_df = self._sanitize(feat_df, feature_cols)

        # Fit scaler on features
        scaled_features = scaler.fit_transform(known_df[feature_cols])
        targets = known_df["Target_Return"].values

        # Build tabular datasets (XGBoost)
        X_tabular = scaled_features
        y_tabular = targets

        # Build sequence datasets (LSTM / Transformer)
        # Adapt sequence length if data is short (e.g. 1 Month ~20 days -> seq_len 5)
        seq_len = self.sequence_length
        if len(scaled_features) < seq_len + 3:
            seq_len = max(3, len(scaled_features) // 3)

        X_seq, y_seq = [], []
        for i in range(seq_len, len(scaled_features)):
            X_seq.append(scaled_features[i - seq_len:i])
            y_seq.append(targets[i])

        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32)

        # Latest feature vector for out-of-sample forward prediction (latest row in feat_df)
        all_scaled = scaler.transform(feat_df[feature_cols])
        latest_tabular = all_scaled[-1:].astype(np.float32)
        latest_seq = all_scaled[-seq_len:][np.newaxis, :, :].astype(np.float32)

        last_close = float(feat_df["Close"].iloc[-1])
        last_date = feat_df.index[-1].strftime("%Y-%m-%d")

        return PreparedData(
            feature_df=feat_df,
            feature_names=feature_cols,
            X_tabular=X_tabular,
            y_tabular=y_tabular,
            X_sequences=X_seq,
            y_sequences=y_seq,
            latest_tabular_input=latest_tabular,
            latest_sequence_input=latest_seq,
            last_close_price=last_close,
            last_date=last_date,
            scaler=scaler
        )
