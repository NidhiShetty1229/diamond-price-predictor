"""Flask backend for the Diamond Price Predictor.

Run with:  python app.py   ->  http://localhost:5000
"""
import threading

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from model import DEFAULT_CSV, DEFAULT_MODEL, DiamondPricePredictor

app = Flask(__name__)
CORS(app)

predictor = DiamondPricePredictor()
train_lock = threading.Lock()

CUT_OPTIONS = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_OPTIONS = ["D", "E", "F", "G", "H", "I", "J"]
CLARITY_OPTIONS = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

# Load an existing model at startup (if one was trained earlier)
predictor.load_model(DEFAULT_MODEL)


def price_category(price):
    if price < 2000:
        return "Budget"
    if price < 6000:
        return "Mid-range"
    if price < 12000:
        return "Premium"
    return "Luxury"


def carat_category(carat):
    if carat < 0.5:
        return f"Small (<0.5 ct)"
    if carat < 0.9:
        return f"Medium ({carat:.1f} ct)"
    if carat <= 1.1:
        return "Standard (≈1.0 ct)"
    if carat < 2.0:
        return f"Large ({carat:.1f} ct)"
    return f"Exceptional ({carat:.1f} ct)"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "healthy", "message": "API is running"})


@app.route("/api/status")
def status():
    return jsonify({
        "model_loaded": bool(predictor.is_trained),
        "metrics": predictor.metrics if predictor.is_trained else {},
        "options": {"cut": CUT_OPTIONS, "color": COLOR_OPTIONS, "clarity": CLARITY_OPTIONS},
    })


@app.route("/api/train", methods=["POST"])
def train():
    if not train_lock.acquire(blocking=False):
        return jsonify({"success": False, "error": "Training is already in progress"}), 409
    try:
        if not predictor.train_model(DEFAULT_CSV):
            return jsonify({"success": False, "error": "Training failed. Is diamonds.csv present?"}), 500
        predictor.save_model(DEFAULT_MODEL)
        return jsonify({"success": True, "message": "Model trained successfully", "metrics": predictor.metrics})
    except Exception as e:  # pragma: no cover
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        train_lock.release()


@app.route("/api/predict", methods=["POST"])
def predict():
    if not predictor.is_trained:
        return jsonify({"success": False, "error": "Model is not trained yet. Click 'Train Model' first."}), 400

    data = request.get_json(silent=True) or {}
    try:
        features = {
            "carat": float(data["carat"]),
            "cut": data["cut"],
            "color": data["color"],
            "clarity": data["clarity"],
            "depth": float(data["depth"]),
            "table": float(data["table"]),
            "x": float(data["x"]),
            "y": float(data["y"]),
            "z": float(data["z"]),
        }
    except (KeyError, TypeError, ValueError):
        return jsonify({"success": False, "error": "Please fill in all fields with valid values."}), 400

    try:
        price = predictor.predict_price(features)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    volume = features["x"] * features["y"] * features["z"]
    return jsonify({
        "success": True,
        "predicted_price": round(price, 2),
        "price_per_carat": round(price / features["carat"], 2),
        "volume": round(volume, 1),
        "price_category": price_category(price),
        "carat_category": carat_category(features["carat"]),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
