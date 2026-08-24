import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any
from .validation import ModelMetrics


@dataclass
class ModelPrediction:
    name: str
    predicted_return: float        # e.g. +0.028 (+2.8%)
    predicted_return_pct: float    # e.g. +2.8
    projected_price: float         # e.g. $205.60
    direction: str                 # "UP" or "DOWN" or "NEUTRAL"
    metrics: ModelMetrics


@dataclass
class ForecastResult:
    current_price: float
    projected_price: float
    expected_return_pct: float
    direction: str                 # "UP" or "DOWN" or "NEUTRAL"
    model_agreement: str           # "HIGH", "MODERATE", "LOW"
    agreement_score: float         # 0.0 to 1.0
    forecast_range_min: float
    forecast_range_max: float
    models: Dict[str, ModelPrediction]
    validation_metrics: Dict[str, ModelMetrics]
    horizon: str                   # "Next Trading Day"
    summary_text: str


class EnsembleForecaster:
    """
    Ensemble combination of LSTM, XGBoost, and Transformer.
    Computes weighted return, price projection, model agreement, and uncertainty range.
    """

    def combine_predictions(
        self,
        current_price: float,
        lstm_return: float,
        xgb_return: float,
        transformer_return: float,
        historical_volatility: float,
        metrics: Dict[str, ModelMetrics]
    ) -> ForecastResult:
        
        # Individual model returns
        r_lstm = float(lstm_return)
        r_xgb = float(xgb_return)
        r_trans = float(transformer_return)

        # Equal-weighted ensemble (or inverse-error weighted)
        combined_return = (r_lstm + r_xgb + r_trans) / 3.0
        combined_return_pct = round(combined_return * 100.0, 2)
        projected_price = round(current_price * (1.0 + combined_return), 2)

        # Direction helper
        def get_dir(r: float) -> str:
            if abs(r) < 0.0005:
                return "NEUTRAL"
            return "UP" if r > 0 else "DOWN"

        # Model Agreement Logic
        # 1. Sign alignment: do all three have the same directional sign?
        signs = [1 if r > 0 else -1 for r in [r_lstm, r_xgb, r_trans]]
        sign_agreement = (abs(sum(signs)) == 3) # 3 if all same sign, 1 if 2 vs 1
        
        # 2. Spread (variance of predictions)
        spread = float(np.std([r_lstm, r_xgb, r_trans]))
        
        if sign_agreement and spread < 0.015:
            model_agreement = "HIGH"
            agreement_score = 0.9
        elif abs(sum(signs)) == 1:
            model_agreement = "LOW"
            agreement_score = 0.35
        else:
            model_agreement = "MODERATE"
            agreement_score = 0.65

        # Defined Uncertainty Methodology for Forecast Range:
        # Range = Central ± (1.645 * combined_uncertainty)
        # where combined_uncertainty = sqrt(volatility^2 + model_spread^2)
        daily_vol = max(float(historical_volatility), 0.01)
        sigma = np.sqrt((daily_vol ** 2) + (spread ** 2))
        margin = current_price * float(sigma) * 1.25

        range_min = round(max(0.01, projected_price - margin), 2)
        range_max = round(projected_price + margin, 2)

        # Build Individual model predictions
        models_dict = {
            "LSTM": ModelPrediction(
                name="LSTM",
                predicted_return=round(r_lstm, 4),
                predicted_return_pct=round(r_lstm * 100.0, 2),
                projected_price=round(current_price * (1.0 + r_lstm), 2),
                direction=get_dir(r_lstm),
                metrics=metrics.get("LSTM", ModelMetrics(0.0, 0.0, 50.0))
            ),
            "XGBoost": ModelPrediction(
                name="XGBoost",
                predicted_return=round(r_xgb, 4),
                predicted_return_pct=round(r_xgb * 100.0, 2),
                projected_price=round(current_price * (1.0 + r_xgb), 2),
                direction=get_dir(r_xgb),
                metrics=metrics.get("XGBoost", ModelMetrics(0.0, 0.0, 50.0))
            ),
            "Transformer": ModelPrediction(
                name="Transformer",
                predicted_return=round(r_trans, 4),
                predicted_return_pct=round(r_trans * 100.0, 2),
                projected_price=round(current_price * (1.0 + r_trans), 2),
                direction=get_dir(r_trans),
                metrics=metrics.get("Transformer", ModelMetrics(0.0, 0.0, 50.0))
            )
        }

        direction_label = get_dir(combined_return)
        if direction_label == "UP":
            summary_text = "The ensemble model forecasts an upward movement based on recent historical data."
        elif direction_label == "DOWN":
            summary_text = "The ensemble model forecasts a downward movement based on recent historical data."
        else:
            summary_text = "The ensemble model forecasts a neutral/stable trend based on recent historical data."

        return ForecastResult(
            current_price=current_price,
            projected_price=projected_price,
            expected_return_pct=combined_return_pct,
            direction=direction_label,
            model_agreement=model_agreement,
            agreement_score=agreement_score,
            forecast_range_min=range_min,
            forecast_range_max=range_max,
            models=models_dict,
            validation_metrics=metrics,
            horizon="Next Trading Day",
            summary_text=summary_text
        )
