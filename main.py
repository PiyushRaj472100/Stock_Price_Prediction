import os
from src.api.app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    print(f"\n=======================================================")
    print(f"  REAL-TIME MULTI-MODEL STOCK FORECASTING SYSTEM")
    print(f"  Models: LSTM + XGBoost + Transformer")
    print(f"  Running on: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
