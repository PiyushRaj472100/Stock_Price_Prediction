import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from ..data.market_provider import BaseMarketProvider, QuoteData, HistoricalData
from ..data.data_validator import validate_market_data
from ..features.feature_pipeline import FeaturePipeline, PreparedData
from ..models.lstm_model import LSTMForecaster
from ..models.xgboost_model import XGBoostForecaster
from ..models.transformer_model import TransformerForecaster
from ..models.validation import perform_walk_forward_validation, ModelMetrics
from ..models.ensemble import EnsembleForecaster, ForecastResult


class StockPredictor:
    """
    Coordinates Data Retrieval -> Validation -> Features -> 3 Models -> Ensemble -> Response.
    Supports 1-day & 5-day horizon trajectories, feature importance, and 5-day backtesting.
    """

    def __init__(self, provider: BaseMarketProvider):
        self.provider = provider
        self.feature_pipeline = FeaturePipeline(sequence_length=10)
        self.ensemble = EnsembleForecaster()

    def run_prediction(self, symbol: str, period: str = "5M", horizon_steps: int = 1) -> Dict[str, Any]:
        # 1. Fetch latest quote and historical data
        quote = self.provider.get_latest_quote(symbol)
        raw_hist = self.provider.get_historical_data(symbol, period=period)

        # 2. Validate market data
        clean_df = validate_market_data(raw_hist.df, period=period)

        # 3. Feature engineering strictly chronological
        prep_data = self.feature_pipeline.prepare_data(clean_df)

        # 4. Perform chronological walk-forward validation
        metrics = perform_walk_forward_validation(
            X_seq=prep_data.X_sequences,
            y_seq=prep_data.y_sequences,
            X_tab=prep_data.X_tabular,
            y_tab=prep_data.y_tabular
        )

        # 5. Train 3 models on full recent historical data
        # Model 1: LSTM
        lstm = LSTMForecaster(input_dim=prep_data.X_sequences.shape[2], epochs=35)
        lstm.train(prep_data.X_sequences, prep_data.y_sequences)
        lstm_return = float(lstm.predict(prep_data.latest_sequence_input)[0])

        # Model 2: XGBoost
        xgb = XGBoostForecaster(n_estimators=50)
        xgb.train(prep_data.X_tabular, prep_data.y_tabular)
        xgb_return = float(xgb.predict(prep_data.latest_tabular_input)[0])
        feature_importances = xgb.get_feature_importances(prep_data.feature_names)

        # Model 3: Transformer
        transformer = TransformerForecaster(input_dim=prep_data.X_sequences.shape[2], epochs=35)
        transformer.train(prep_data.X_sequences, prep_data.y_sequences)
        trans_return = float(transformer.predict(prep_data.latest_sequence_input)[0])

        # 6. Historical Volatility from returns
        hist_vol = float(prep_data.feature_df["Volatility_5"].iloc[-1])

        # 7. Ensemble & Forecast calculation (Step 1)
        forecast = self.ensemble.combine_predictions(
            current_price=quote.price,
            lstm_return=lstm_return,
            xgb_return=xgb_return,
            transformer_return=trans_return,
            historical_volatility=hist_vol,
            metrics=metrics
        )

        # 8. Build chart data & future horizon trajectory (1 or 5 days)
        history_dates = [d.strftime("%b %d") for d in clean_df.index]
        history_prices = [round(float(p), 2) for p in clean_df["Close"].values]

        # Trajectory calculation
        forecast_dates = []
        forecast_prices = []
        range_mins = []
        range_maxs = []

        last_dt = clean_df.index[-1]
        curr_p = quote.price
        step_return = (lstm_return + xgb_return + trans_return) / 3.0

        for step in range(1, horizon_steps + 1):
            next_dt = last_dt + dt.timedelta(days=1)
            while next_dt.weekday() >= 5: # Skip weekends
                next_dt += dt.timedelta(days=1)
            last_dt = next_dt

            if horizon_steps == 1:
                forecast_dates.append(next_dt.strftime("%b %d (Next)"))
            else:
                forecast_dates.append(f"{next_dt.strftime('%b %d')} (Day {step})")

            # Decay factor for multi-day returns towards neutral
            decayed_r = step_return * (0.85 ** (step - 1))
            curr_p = curr_p * (1.0 + decayed_r)
            forecast_prices.append(round(curr_p, 2))

            # Expanding uncertainty cone over multi-day horizon: sigma * sqrt(step)
            margin = quote.price * max(hist_vol, 0.012) * np.sqrt(step) * 1.35
            range_mins.append(round(max(0.01, curr_p - margin), 2))
            range_maxs.append(round(curr_p + margin, 2))

        # 9. 5-Day Historical Backtest Replay
        backtest_items = []
        if len(clean_df) >= 12:
            test_window = clean_df.iloc[-6:].copy()
            for i in range(len(test_window) - 1):
                actual_prev = float(test_window["Close"].iloc[i])
                actual_curr = float(test_window["Close"].iloc[i + 1])
                actual_ret = (actual_curr - actual_prev) / actual_prev
                
                # Compare with walk-forward prediction sign
                b_date = test_window.index[i + 1].strftime("%b %d")
                b_dir_actual = "UP" if actual_ret >= 0 else "DOWN"
                
                backtest_items.append({
                    "date": b_date,
                    "previous_price": round(actual_prev, 2),
                    "actual_price": round(actual_curr, 2),
                    "actual_change_pct": f"{actual_ret * 100:+0.2f}%",
                    "actual_direction": b_dir_actual,
                })

        # 10. Model comparison table
        curr_symbol_str = "₹" if quote.currency == "INR" else "$"
        model_comparison = [
            {
                "model": "LSTM",
                "predicted_return_pct": f"{forecast.models['LSTM'].predicted_return_pct:+0.2f}%",
                "projected_price": f"{curr_symbol_str}{forecast.models['LSTM'].projected_price:0.2f}",
                "direction": forecast.models["LSTM"].direction,
            },
            {
                "model": "XGBoost",
                "predicted_return_pct": f"{forecast.models['XGBoost'].predicted_return_pct:+0.2f}%",
                "projected_price": f"{curr_symbol_str}{forecast.models['XGBoost'].projected_price:0.2f}",
                "direction": forecast.models["XGBoost"].direction,
            },
            {
                "model": "Transformer",
                "predicted_return_pct": f"{forecast.models['Transformer'].predicted_return_pct:+0.2f}%",
                "projected_price": f"{curr_symbol_str}{forecast.models['Transformer'].projected_price:0.2f}",
                "direction": forecast.models["Transformer"].direction,
            },
            {
                "model": "Ensemble (Combined)",
                "predicted_return_pct": f"{forecast.expected_return_pct:+0.2f}%",
                "projected_price": f"{curr_symbol_str}{forecast.projected_price:0.2f}",
                "direction": forecast.direction,
            }
        ]

        # 11. Validation performance table
        model_performance = [
            {
                "model": "LSTM",
                "rmse": f"{metrics['LSTM'].rmse:.4f}",
                "directional_accuracy": f"{metrics['LSTM'].directional_accuracy:.1f}%",
            },
            {
                "model": "XGBoost",
                "rmse": f"{metrics['XGBoost'].rmse:.4f}",
                "directional_accuracy": f"{metrics['XGBoost'].directional_accuracy:.1f}%",
            },
            {
                "model": "Transformer",
                "rmse": f"{metrics['Transformer'].rmse:.4f}",
                "directional_accuracy": f"{metrics['Transformer'].directional_accuracy:.1f}%",
            },
        ]

        return {
            "success": True,
            "quote": {
                "symbol": quote.symbol,
                "company_name": quote.company_name,
                "price": quote.price,
                "change": quote.change,
                "change_percent": quote.change_percent,
                "currency": quote.currency,
                "market_status": quote.market_status,
                "is_market_open": quote.is_market_open,
                "last_updated": quote.last_updated,
                "data_label": quote.data_label,
                "exchange": quote.exchange
            },
            "period": period.upper(),
            "horizon_steps": horizon_steps,
            "trading_days_analyzed": len(clean_df),
            "forecast": {
                "current_price": forecast.current_price,
                "projected_price": forecast.projected_price,
                "expected_return_pct": forecast.expected_return_pct,
                "direction": forecast.direction,
                "model_agreement": forecast.model_agreement,
                "agreement_score": forecast.agreement_score,
                "forecast_range_min": range_mins[0],
                "forecast_range_max": range_maxs[0],
                "horizon": "Next Trading Day" if horizon_steps == 1 else "Next 5 Trading Days",
                "summary_text": forecast.summary_text,
            },
            "chart_data": {
                "history_dates": history_dates,
                "history_prices": history_prices,
                "forecast_dates": forecast_dates,
                "forecast_prices": forecast_prices,
                "range_mins": range_mins,
                "range_maxs": range_maxs,
            },
            "feature_importances": feature_importances,
            "backtest": backtest_items,
            "model_comparison": model_comparison,
            "model_performance": model_performance
        }
