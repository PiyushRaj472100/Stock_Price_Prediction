# Real-Time Multi-Model Stock Forecasting System

A focused, real-time stock forecasting web application powered by **three distinct Machine Learning models** (LSTM, XGBoost, and Time-Series Transformer) with an ensemble aggregator, live market quotes, and chronological walk-forward validation.

---

## 🎯 Core Concept & Workflow

```text
SELECT REAL STOCK (US / India)
          ↓
FETCH CURRENT / LATEST MARKET PRICE & STATUS
          ↓
SELECT RECENT HISTORY (1-Month / 5-Months)
          ↓
CLEAN DATA & EXTRACT COMPACT TIME-SERIES FEATURES
          ↓
RUN 3 MODELS (LSTM + XGBoost + Transformer)
          ↓
CHRONOLOGICAL WALK-FORWARD VALIDATION (No Data Leakage)
          ↓
ENSEMBLE FORECAST + RANGE + MODEL AGREEMENT
          ↓
INTERACTIVE VISUALIZATION (Actual vs Forecast)
```

---

## 🧠 The Three ML Models

1. **LSTM (Long Short-Term Memory Neural Network)**:
   - Captures non-linear temporal sequence patterns across price & volume momentum.
   - Built with PyTorch with temporal sequence sliding windows.

2. **XGBoost (Extreme Gradient Boosting Regressor)**:
   - Robust tabular decision-tree ensemble learning from multi-feature snapshots (Returns, Moving Averages, Volatility, RSI, Volume Changes).

3. **Time-Series Transformer**:
   - Multi-Head Self-Attention architecture with Positional Encoding tailored for short-to-medium sequence dependency modeling.

### ⚖️ Ensemble Combination
- **Equal-Weighted Return Forecast**: Averages predicted next-step returns from all 3 models.
- **Model Agreement Index**:
  - `HIGH`: All 3 models agree on directional sign with tight dispersion.
  - `MODERATE`: 2 of 3 models align directionally.
  - `LOW`: Models exhibit significant directional disagreement or wide spread.
- **Uncertainty Forecast Range**: Derived from historical daily price volatility combined with model prediction dispersion.

---

## 🛡️ Anti-Leakage & Proper Validation
- **Chronological Sequence Splitting**: Strict time-ordered evaluation ($t_{train} < t_{test}$). No random shuffling or lookahead bias.
- **Real Performance Metrics**: Calculates out-of-sample RMSE, MAE, and Directional Accuracy (%) on real historical market data.

---

## 🚀 Quick Start & Installation

### 1. Clone & Setup Environment
```bash
git clone https://github.com/AdilShamim8/Stock_Price_Prediction.git
cd Stock_Price_Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables (Optional)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Default provider is `yfinance` (works out of the box with zero API keys required). Alternatively, set `MARKET_DATA_PROVIDER=alphavantage` and your `MARKET_DATA_API_KEY`.

### 4. Run the Application
```bash
python main.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

## 🧪 Running Tests
```bash
pytest tests/test_pipeline.py
```

---

## 📁 Project Structure

```
Stock_Price_Prediction/
├── src/
│   ├── data/
│   │   ├── market_provider.py    # Pluggable data layer (Yahoo Finance & Alpha Vantage)
│   │   └── data_validator.py     # Time-series data integrity validation
│   ├── features/
│   │   └── feature_pipeline.py   # Compact technical indicators & sequence prep
│   ├── models/
│   │   ├── lstm_model.py         # PyTorch LSTM Forecaster
│   │   ├── xgboost_model.py      # XGBoost Tabular Forecaster
│   │   ├── transformer_model.py  # PyTorch Time-Series Transformer Forecaster
│   │   ├── validation.py         # Chronological walk-forward validation (RMSE, Acc %)
│   │   └── ensemble.py           # 3-Model ensemble, agreement & uncertainty range
│   ├── prediction/
│   │   └── predictor.py          # Pipeline orchestration
│   └── api/
│       └── app.py                # Flask REST API endpoints
├── static/
│   ├── css/style.css             # Modern dark financial styling
│   └── js/app.js                 # Interactive Chart.js and live search
├── templates/
│   └── index.html                # Responsive web dashboard
├── tests/
│   └── test_pipeline.py          # Unit & integration tests
├── .env.example
├── requirements.txt
├── main.py
└── README.md
```

---

## ⚠️ Financial Prediction Disclaimer
This software is intended strictly for educational, academic, and research purposes. Stock markets involve substantial risk. Model forecasts are statistical estimates and **do NOT constitute financial advice or investment recommendations**.
