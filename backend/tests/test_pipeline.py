import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pytest
import numpy as np
import pandas as pd
import datetime as dt

from src.data.market_provider import YahooFinanceProvider, QuoteData, HistoricalData
from src.data.data_validator import validate_market_data
from src.features.feature_pipeline import FeaturePipeline
from src.models.lstm_model import LSTMForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.transformer_model import TransformerForecaster
from src.models.validation import perform_walk_forward_validation, calculate_metrics
from src.models.ensemble import EnsembleForecaster
from src.prediction.predictor import StockPredictor
from src.api.app import create_app


@pytest.fixture
def sample_market_df():
    """Generates synthetic valid OHLCV dataframe for testing."""
    dates = pd.date_range(start="2025-01-01", periods=60, freq="B")
    np.random.seed(42)
    prices = 150.0 + np.cumsum(np.random.randn(len(dates)) * 2)
    
    df = pd.DataFrame({
        "Open": prices - 0.5,
        "High": prices + 1.5,
        "Low": prices - 1.5,
        "Close": prices,
        "Volume": np.random.randint(1000000, 5000000, size=len(dates))
    }, index=dates)
    return df


def test_data_validation(sample_market_df):
    clean_df = validate_market_data(sample_market_df, period="5M")
    assert len(clean_df) == 60
    assert list(clean_df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    # Test rejection of insufficient records
    short_df = sample_market_df.iloc[:5]
    with pytest.raises(ValueError):
        validate_market_data(short_df, period="5M")


def test_feature_pipeline(sample_market_df):
    pipeline = FeaturePipeline(sequence_length=10)
    prepared = pipeline.prepare_data(sample_market_df)
    
    assert prepared.X_tabular.ndim == 2
    assert prepared.y_tabular.ndim == 1
    assert prepared.X_sequences.ndim == 3
    assert prepared.X_sequences.shape[1] == 10  # seq_len
    assert len(prepared.feature_names) == 11
    assert prepared.latest_sequence_input.shape == (1, 10, len(prepared.feature_names))


def test_three_models_and_ensemble(sample_market_df):
    pipeline = FeaturePipeline(sequence_length=10)
    prepared = pipeline.prepare_data(sample_market_df)

    # 1. LSTM
    lstm = LSTMForecaster(input_dim=prepared.X_sequences.shape[2], epochs=10)
    lstm.train(prepared.X_sequences, prepared.y_sequences)
    lstm_pred = lstm.predict(prepared.latest_sequence_input)
    assert lstm_pred.shape == (1,)

    # 2. XGBoost
    xgb = XGBoostForecaster(n_estimators=10)
    xgb.train(prepared.X_tabular, prepared.y_tabular)
    xgb_pred = xgb.predict(prepared.latest_tabular_input)
    assert xgb_pred.shape == (1,)

    # 3. Transformer
    transformer = TransformerForecaster(input_dim=prepared.X_sequences.shape[2], epochs=10)
    transformer.train(prepared.X_sequences, prepared.y_sequences)
    trans_pred = transformer.predict(prepared.latest_sequence_input)
    assert trans_pred.shape == (1,)

    # Walk-forward validation
    metrics = perform_walk_forward_validation(
        X_seq=prepared.X_sequences,
        y_seq=prepared.y_sequences,
        X_tab=prepared.X_tabular,
        y_tab=prepared.y_tabular
    )
    assert "LSTM" in metrics and "XGBoost" in metrics and "Transformer" in metrics

    # Ensemble
    ensemble = EnsembleForecaster()
    result = ensemble.combine_predictions(
        current_price=200.0,
        lstm_return=float(lstm_pred[0]),
        xgb_return=float(xgb_pred[0]),
        transformer_return=float(trans_pred[0]),
        historical_volatility=0.015,
        metrics=metrics
    )
    assert result.current_price == 200.0
    assert result.model_agreement in ("HIGH", "MODERATE", "LOW")
    assert result.forecast_range_min < result.forecast_range_max
    assert "LSTM" in result.models
    assert "XGBoost" in result.models
    assert "Transformer" in result.models


def test_api_routes():
    app = create_app()
    client = app.test_client()

    # Test index
    res_index = client.get("/")
    assert res_index.status_code == 200

    # Test search
    res_search = client.get("/api/search?q=AAPL")
    assert res_search.status_code == 200
    data_search = res_search.get_json()
    assert data_search["success"] is True
    assert len(data_search["results"]) > 0

    # Test quote
    res_quote = client.get("/api/quote?symbol=AAPL")
    assert res_quote.status_code == 200
    data_quote = res_quote.get_json()
    assert data_quote["success"] is True
    assert data_quote["quote"]["symbol"] == "AAPL"
    assert data_quote["quote"]["price"] > 0
