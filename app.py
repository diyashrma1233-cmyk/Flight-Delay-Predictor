"""
Flight Delay Prediction — Flask Backend
=========================================
Loads the flight_delay (1).csv dataset, cleans it,
trains a RandomForestClassifier, and exposes a /predict API.
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Data Loading & Model Training  (runs once on startup)
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "flight_delay (1).csv")

def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load the CSV and perform data cleaning."""
    df = pd.read_csv(path)
    print(f"Dataset Shape: {df.shape}")
    print(f"\n--- Initial Missing Values ---\n{df.isnull().sum()}")

    # 1. Fix Data Types: Convert non-numeric columns to numeric (strings → NaN)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Impute missing numerical data with median strategy
    imputer = SimpleImputer(strategy="median")
    df_cleaned = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    print(f"\n--- Missing Values After Imputation ---\n{df_cleaned.isnull().sum()}")
    print(f"\nShape after cleaning: {df_cleaned.shape}")
    return df_cleaned


def train_model(df: pd.DataFrame):
    """Split data 80/20, train RandomForestClassifier, return model + feature names."""
    target_col = "is_delayed"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 80 % training · 20 % testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Testing data shape:  {X_test.shape}")

    # RandomForest — 100 trees, balanced class weights
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nRandom Forest Accuracy: {accuracy:.4f}\n")
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    return model, list(X_train.columns)


# Run on import / startup
print("\n" + "=" * 60)
print("  Flight Delay Predictor - Initializing...")
print("=" * 60 + "\n")

df_clean = load_and_clean_data(CSV_PATH)
rf_model, feature_names = train_model(df_clean)

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
    app.run(debug=False, port = 5000)
