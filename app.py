"""
Flight Delay Prediction — Flask Backend
=========================================
Loads the flight_delay (1).csv dataset, cleans it,
trains a RandomForestClassifier, and exposes a /predict API.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Run on import / startup
print("\n" + "=" * 60)
print("  Flight Delay Predictor - Initializing...")
print("=" * 60 + "\n")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
print(f"Loading pre-trained model from {MODEL_PATH} ...")
rf_model = joblib.load(MODEL_PATH)
feature_names = [
    "departure_hour",
    "scheduled_duration_min",
    "distance_miles",
    "num_connections",
    "wind_speed_kmh",
    "visibility_km",
    "prev_flight_delay_min"
]

print("=" * 60)
print("  Model ready.  Starting Flask server...")
print("=" * 60 + "\n")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept JSON with 7 features, return prediction + probability."""
    data = request.get_json(force=True)

    try:
        input_values = [
            float(data["departure_hour"]),
            float(data["scheduled_duration_min"]),
            float(data["distance_miles"]),
            float(data["num_connections"]),
            float(data["wind_speed_kmh"]),
            float(data["visibility_km"]),
            float(data["prev_flight_delay_min"]),
        ]
    except (KeyError, ValueError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    input_df = pd.DataFrame([input_values], columns=feature_names)

    prediction = int(rf_model.predict(input_df)[0])
    probabilities = rf_model.predict_proba(input_df)[0]
    delay_probability = float(probabilities[1])

    return jsonify({
        "prediction": prediction,
        "probability": round(delay_probability, 2),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
