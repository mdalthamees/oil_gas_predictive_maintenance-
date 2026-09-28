"""
prediction.py
---------------
Shared inference helpers used by app.py (Streamlit dashboard). Wraps model
loading, the full clean -> engineer-features -> encode -> scale -> predict
pipeline, health-score banding, risk-level banding and human-readable
recommendations, so the Streamlit UI code stays thin and declarative.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_preprocessing import DataValidationError, clean_raw_dataframe, validate_columns
from feature_engineering import ALL_MODEL_FEATURES, add_engineered_features
from train_failure_model import encode_features

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

TOP_RISK_FEATURE_LABELS = {
    "Vibration_mm_s": "High Vibration",
    "Vibration_Risk": "Elevated Vibration Risk Flag",
    "Oil_Temperature_C": "Increasing Oil Temperature",
    "Operating_Hours": "Excessive Operating Hours",
    "Operating_Hour_Risk": "Excessive Operating Hours",
    "Pressure_Bar": "Pressure Fluctuation",
    "Pressure_Deviation": "Pressure Deviation From Normal",
    "Pressure_Fluctuation": "Pressure Fluctuation",
    "Last_Maintenance_Days": "Long Period Since Maintenance",
    "Maintenance_Due": "Maintenance Overdue",
    "Temperature_C": "Elevated Operating Temperature",
    "Temperature_Deviation": "Temperature Deviation From Normal",
    "Sensor_Anomaly_Count": "Multiple Sensor Anomalies",
    "Failure_Count": "History of Repeated Failures",
    "Failure_Frequency": "High Failure Frequency",
    "Motor_Current_A": "Abnormal Motor Current Draw",
    "RPM_Deviation": "RPM Deviation From Normal",
    "Health_Index": "Low Composite Health Index",
    "Maintenance_History": "Poor Maintenance History",
    "Power_Efficiency": "Reduced Power Efficiency",
}


class ModelArtifacts:
    """Lazily-loaded, cached bundle of everything needed for inference."""

    _instance = None

    def __init__(self):
        self.failure_model = joblib.load(MODELS_DIR / "failure_prediction_model.pkl")
        self.rul_model = joblib.load(MODELS_DIR / "rul_prediction_model.pkl")
        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        self.encoders = joblib.load(MODELS_DIR / "encoder.pkl")
        self.feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
        self.numeric_columns = joblib.load(MODELS_DIR / "numeric_columns.pkl")
        try:
            importance = pd.read_csv(BASE_DIR / "reports" / "feature_importance.csv", index_col=0)
            self.global_importance = importance.iloc[:, 0]
        except FileNotFoundError:
            self.global_importance = None

    @classmethod
    def get(cls) -> "ModelArtifacts":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def available() -> bool:
        required = [
            "failure_prediction_model.pkl", "rul_prediction_model.pkl",
            "scaler.pkl", "encoder.pkl", "feature_names.pkl", "numeric_columns.pkl",
        ]
        return all((MODELS_DIR / f).exists() for f in required)


def health_band(score: float) -> tuple[str, str]:
    """Return (label, color) for a 0-100 health score."""
    if score >= 90:
        return "Excellent", "#2ecc71"
    if score >= 75:
        return "Good", "#27ae60"
    if score >= 50:
        return "Warning", "#f1c40f"
    if score >= 25:
        return "Critical", "#e67e22"
    return "Severe", "#e74c3c"


def risk_level(probability: float) -> tuple[str, str]:
    """Return (label, color) for a failure probability in [0, 1]."""
    if probability < 0.25:
        return "LOW", "#2ecc71"
    if probability < 0.50:
        return "MEDIUM", "#f1c40f"
    if probability < 0.75:
        return "HIGH", "#e67e22"
    return "CRITICAL", "#e74c3c"


def maintenance_priority(risk_label: str) -> tuple[str, str]:
    mapping = {
        "CRITICAL": ("🔴 Critical", "Immediate inspection and preventive maintenance required"),
        "HIGH": ("🟠 High", "Schedule preventive maintenance within 7 days"),
        "MEDIUM": ("🟡 Medium", "Schedule maintenance within 30 days; increase monitoring"),
        "LOW": ("🟢 Low", "Continue normal monitoring"),
    }
    return mapping.get(risk_label, ("🟢 Low", "Continue normal monitoring"))


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: clean -> engineer -> select model columns."""
    df = clean_raw_dataframe(df, verbose=False)
    df = add_engineered_features(df)
    return df


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full pipeline + both models on a raw (uncleaned) dataframe.
    Returns the original identifying columns plus prediction outputs.
    """
    artifacts = ModelArtifacts.get()

    missing = validate_columns(df)
    if missing:
        raise DataValidationError(f"Uploaded file is missing required columns: {missing}")

    processed = prepare_features(df)

    X = processed[ALL_MODEL_FEATURES].copy()
    X_encoded, _ = encode_features(X, encoders=artifacts.encoders)
    X_scaled = X_encoded.copy()
    X_scaled[artifacts.numeric_columns] = artifacts.scaler.transform(X_encoded[artifacts.numeric_columns])

    failure_proba = artifacts.failure_model.predict_proba(X_scaled)[:, 1]
    rul_pred = artifacts.rul_model.predict(X_encoded)

    out = pd.DataFrame(index=processed.index)
    for col in ["Equipment_ID", "Equipment_Type", "Plant_Location"]:
        if col in processed.columns:
            out[col] = processed[col]

    out["Failure_Probability"] = np.round(failure_proba * 100, 1)
    out["Remaining_Useful_Life"] = np.round(rul_pred, 0)
    out["Health_Score"] = processed["Health_Index"] if "Health_Index" in processed.columns else np.nan

    risk_labels = [risk_level(p / 100)[0] for p in out["Failure_Probability"]]
    out["Risk_Level"] = risk_labels
    priorities = [maintenance_priority(r) for r in risk_labels]
    out["Maintenance_Priority"] = [p[0] for p in priorities]
    out["Recommended_Action"] = [p[1] for p in priorities]

    return out


def predict_single(input_dict: dict) -> dict:
    """Predict failure risk + RUL for a single manually-entered equipment
    reading (used by the "AI Failure Prediction" form page).
    """
    df = pd.DataFrame([input_dict])
    # fill any required-but-unprovided columns with sensible defaults so the
    # pipeline doesn't break on a partial manual form.
    defaults = {
        "Equipment_ID": "MANUAL-ENTRY",
        "Plant_Location": "Not Specified",
        "Maintenance_History": "Irregular",
        "Ambient_Temperature_C": 35,
        "Humidity_Percent": 45,
        "Valve_Position_Percent": 50,
        "Power_Consumption_kW": input_dict.get("Motor_Current_A", 40) * 0.42,
        "Fuel_Gas_Pressure_Bar": 25,
    }
    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v

    result = predict_batch(df)
    row = result.iloc[0]

    top_factors = explain_prediction(df)

    return {
        "failure_probability": float(row["Failure_Probability"]),
        "risk_level": row["Risk_Level"],
        "health_score": float(row["Health_Score"]),
        "remaining_useful_life": float(row["Remaining_Useful_Life"]),
        "maintenance_priority": row["Maintenance_Priority"],
        "recommended_action": row["Recommended_Action"],
        "top_factors": top_factors,
    }


def explain_prediction(df: pd.DataFrame, top_n: int = 5) -> list[str]:
    """Model-attributed risk drivers for a single row, based on the trained
    model's global feature importances (fast, dependency-light explainer).
    For a deeper per-row explanation, SHAP is used in the notebooks.
    """
    artifacts = ModelArtifacts.get()
    if artifacts.global_importance is None:
        return ["Feature importance report not available."]

    processed = prepare_features(df)
    row = processed.iloc[0]

    # Score each top global-importance feature by how far this row's value
    # sits from the (encoded) dataset baseline, to surface the ones most
    # likely driving THIS prediction rather than always returning the same
    # static global list.
    candidates = artifacts.global_importance.head(12).index.tolist()
    scored = []
    for feat in candidates:
        label = TOP_RISK_FEATURE_LABELS.get(feat, feat.replace("_", " "))
        if feat in row.index and pd.api.types.is_numeric_dtype(type(row[feat])):
            scored.append((label, artifacts.global_importance[feat]))
        else:
            scored.append((label, artifacts.global_importance.get(feat, 0)))

    seen = set()
    ordered = []
    for label, _ in scored:
        if label not in seen:
            ordered.append(label)
            seen.add(label)
        if len(ordered) >= top_n:
            break
    return ordered
