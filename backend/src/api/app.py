import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# backend/ is the root for Python imports; frontend/ is a sibling of backend/
backend_dir = Path(__file__).resolve().parent.parent.parent   # .../backend
root_dir = backend_dir  # keep root_dir alias for sys.path
frontend_dir = backend_dir.parent / "frontend"                # .../frontend

if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from src.data.market_provider import get_market_provider
from src.prediction.predictor import StockPredictor


def create_app() -> Flask:
    template_dir = str(frontend_dir / "templates")
    static_dir   = str(frontend_dir / "static")

    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "stock-forecasting-secret-key")

    provider = get_market_provider()
    predictor = StockPredictor(provider=provider)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/healthz")
    def healthz():
        return jsonify({"status": "ok"}), 200

    @app.route("/favicon.ico")
    def favicon():
        return "", 204   # No Content — silences the browser 404

    @app.route("/api/search", methods=["GET"])
    def search():
        query = request.args.get("q", "").strip()
        try:
            results = provider.search_symbols(query)
            return jsonify({"success": True, "results": results})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/quote", methods=["GET"])
    def quote():
        symbol = request.args.get("symbol", "").strip()
        if not symbol:
            return jsonify({"success": False, "error": "Symbol parameter is required."}), 400
        try:
            quote_data = provider.get_latest_quote(symbol)
            return jsonify({
                "success": True,
                "quote": {
                    "symbol": quote_data.symbol,
                    "company_name": quote_data.company_name,
                    "price": quote_data.price,
                    "change": quote_data.change,
                    "change_percent": quote_data.change_percent,
                    "currency": quote_data.currency,
                    "market_status": quote_data.market_status,
                    "is_market_open": quote_data.is_market_open,
                    "last_updated": quote_data.last_updated,
                    "data_label": quote_data.data_label,
                    "exchange": quote_data.exchange
                }
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400

    @app.route("/api/forecast", methods=["POST"])
    def forecast():
        data = request.get_json(silent=True) or {}
        symbol = data.get("symbol", "").strip()
        period = data.get("period", "5M").strip().upper()

        if not symbol:
            return jsonify({"success": False, "error": "Stock symbol is required."}), 400

        if period not in ("1M", "5M"):
            period = "5M"

        horizon_steps = int(data.get("horizon_steps", 1))
        if horizon_steps not in (1, 5):
            horizon_steps = 1

        try:
            result = predictor.run_prediction(symbol=symbol, period=period, horizon_steps=horizon_steps)
            return jsonify(result)
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Forecast error for '{symbol}': {str(e)}"
            }), 400

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
