"""
train_rul_model.py
--------------------
Trains and compares regression models to estimate Remaining Useful Life
(RUL, in days) for each piece of equipment, and saves the best model.

Run:
    python src/train_rul_model.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from data_preprocessing import clean_raw_dataframe
from feature_engineering import ALL_MODEL_FEATURES, add_engineered_features
from train_failure_model import encode_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
TARGET_COLUMN = "Remaining_Useful_Life"

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading + cleaning + feature-engineering data...")
    raw_path = BASE_DIR / "data" / "raw" / "oil_gas_equipment_sensor_data.csv"
    df = pd.read_csv(raw_path)
    df = clean_raw_dataframe(df)
    df = add_engineered_features(df)

    X = df[ALL_MODEL_FEATURES].copy()
    y = df[TARGET_COLUMN].astype(float)

    # Reuse the SAME label encoders as the classification model so both
    # models agree on the categorical encoding scheme.
    encoders_path = MODELS_DIR / "encoder.pkl"
    if encoders_path.exists():
        encoders = joblib.load(encoders_path)
        X_encoded, _ = encode_features(X, encoders=encoders)
    else:
        X_encoded, encoders = encode_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=RANDOM_SEED
    )

    logger.info("Training RUL regression candidates...")
    models = {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300, max_depth=14, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=250, max_depth=4, learning_rate=0.06, random_state=RANDOM_SEED
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost Regressor"] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.06, random_state=RANDOM_SEED, n_jobs=-1
        )
    else:
        logger.warning("xgboost not available - skipping XGBoost Regressor")

    results = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results.append({"Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)})
        logger.info("  %-28s MAE=%.2f RMSE=%.2f R2=%.4f", name, mae, rmse, r2)

    results_df = pd.DataFrame(results).sort_values(by="RMSE")
    results_df.to_csv(REPORTS_DIR / "rul_model_performance.csv", index=False)
    print("\n" + results_df.to_string(index=False) + "\n")

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted[best_name]
    logger.info("Selected final RUL model: %s", best_name)

    joblib.dump(best_model, MODELS_DIR / "rul_prediction_model.pkl")

    with open(MODELS_DIR / "rul_model_metadata.json", "w") as f:
        json.dump({
            "selected_model": best_name,
            "target": TARGET_COLUMN,
            "metrics": results_df.iloc[0].to_dict(),
        }, f, indent=2, default=str)

    logger.info("Saved RUL model -> %s", MODELS_DIR / "rul_prediction_model.pkl")


if __name__ == "__main__":
    main()
