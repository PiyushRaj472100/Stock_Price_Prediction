import numpy as np
import xgboost as xgb


class XGBoostForecaster:
    """XGBoost Tabular Time-Series Forecasting Model."""

    def __init__(self, n_estimators: int = 50, max_depth: int = 3, learning_rate: float = 0.05):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1
        )
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        if len(X_train) == 0:
            return
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X_input: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return np.zeros((len(X_input),))
        if X_input.ndim == 1:
            X_input = X_input.reshape(1, -1)
        preds = self.model.predict(X_input)
        return np.array(preds)

    def get_feature_importances(self, feature_names: list) -> list:
        if not self.is_trained or not hasattr(self.model, "feature_importances_"):
            return []
        importances = self.model.feature_importances_
        total = float(np.sum(importances)) or 1.0
        results = []
        for name, val in zip(feature_names, importances):
            pct = round((float(val) / total) * 100.0, 1)
            results.append({"feature": name, "importance_pct": pct})
        results.sort(key=lambda x: x["importance_pct"], reverse=True)
        return results[:5]
