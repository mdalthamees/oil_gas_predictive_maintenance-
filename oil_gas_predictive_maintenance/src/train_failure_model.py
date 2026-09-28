"""
train_failure_model.py
------------------------
Trains and compares multiple classifiers to predict Failure_Within_30_Days,
selects a final model (favoring Recall, since missing a real failure is more
costly than a false alarm in predictive maintenance), and saves the model +
preprocessing artifacts (scaler, encoder, feature name list) with Joblib.

Run:
    python src/train_failure_model.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from data_preprocessing import clean_raw_dataframe
from feature_engineering import (
    ALL_MODEL_FEATURES,
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURE_COLUMNS,
    add_engineered_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
TARGET_COLUMN = "Failure_Within_30_Days"

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


def build_training_frame() -> pd.DataFrame:
    raw_path = BASE_DIR / "data" / "raw" / "oil_gas_equipment_sensor_data.csv"
    df = pd.read_csv(raw_path)
    df = clean_raw_dataframe(df)
    df = add_engineered_features(df)
    return df


def encode_features(df: pd.DataFrame, encoders: dict[str, LabelEncoder] | None = None):
    """Label-encode categorical columns. If `encoders` is None, fit new
    encoders (training time); otherwise reuse them (inference time).
    """
    df = df.copy()
    fitted = encoders is None
    encoders = encoders or {}

    for col in CATEGORICAL_FEATURES:
        if fitted:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # handle unseen categories gracefully at inference time
            df[col] = df[col].astype(str).map(
                lambda v: v if v in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])
    return df, encoders


def evaluate_model(name, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1_Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_test, y_proba), 4),
    }
    cm = confusion_matrix(y_test, y_pred)
    metrics["False_Negatives"] = int(cm[1][0]) if cm.shape == (2, 2) else None
    metrics["False_Positives"] = int(cm[0][1]) if cm.shape == (2, 2) else None
    logger.info("  %-22s Acc=%.3f Prec=%.3f Recall=%.3f F1=%.3f ROC-AUC=%.3f  FN=%s",
                name, metrics["Accuracy"], metrics["Precision"], metrics["Recall"],
                metrics["F1_Score"], metrics["ROC_AUC"], metrics["False_Negatives"])
    return metrics


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Building training dataframe (clean + feature engineer)...")
    df = build_training_frame()

    X = df[ALL_MODEL_FEATURES].copy()
    y = df[TARGET_COLUMN].astype(int)

    logger.info("Encoding categorical features...")
    X_encoded, encoders = encode_features(X)

    logger.info("Splitting train/test (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    logger.info("Scaling numeric features...")
    numeric_cols = BASE_NUMERIC_FEATURES + ENGINEERED_FEATURE_COLUMNS
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    logger.info("Training candidate models...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08, random_state=RANDOM_SEED
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=14, random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1
        ),
    }
    if XGBOOST_AVAILABLE:
        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.08, random_state=RANDOM_SEED,
            eval_metric="logloss", scale_pos_weight=neg / max(pos, 1), n_jobs=-1,
        )
    else:
        logger.warning("xgboost not available - skipping XGBoost model")

    results = []
    fitted_models = {}
    for name, model in models.items():
        # Tree ensembles don't need scaling, but using scaled data doesn't
        # hurt them either and keeps the pipeline simple/uniform.
        model.fit(X_train_scaled, y_train)
        fitted_models[name] = model
        results.append(evaluate_model(name, model, X_test_scaled, y_test))

    results_df = pd.DataFrame(results).sort_values(by=["Recall", "F1_Score"], ascending=False)
    results_path = REPORTS_DIR / "model_performance.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved model comparison -> %s", results_path)
    print("\n" + results_df.to_string(index=False) + "\n")

    # ------------------------------------------------------------------
    # Model selection: predictive maintenance cares most about NOT missing
    # a real failure (Recall / False Negatives), while still keeping
    # reasonable precision (too many false alarms erode trust in the
    # system). We rank primarily by Recall, using F1 as a tiebreaker,
    # among models with acceptable ROC-AUC.
    # ------------------------------------------------------------------
    candidates = results_df[results_df["ROC_AUC"] >= results_df["ROC_AUC"].median()]
    best_name = candidates.sort_values(by=["Recall", "F1_Score"], ascending=False).iloc[0]["Model"]
    best_model = fitted_models[best_name]
    logger.info("Selected final model: %s", best_name)

    joblib.dump(best_model, MODELS_DIR / "failure_prediction_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(encoders, MODELS_DIR / "encoder.pkl")
    joblib.dump(ALL_MODEL_FEATURES, MODELS_DIR / "feature_names.pkl")
    joblib.dump(numeric_cols, MODELS_DIR / "numeric_columns.pkl")

    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump({
            "selected_model": best_name,
            "target": TARGET_COLUMN,
            "features": ALL_MODEL_FEATURES,
            "numeric_columns": numeric_cols,
            "categorical_columns": CATEGORICAL_FEATURES,
            "metrics": candidates[candidates["Model"] == best_name].iloc[0].to_dict(),
        }, f, indent=2, default=str)

    logger.info("Saved model + scaler + encoder + feature_names to %s", MODELS_DIR)

    # ---- Feature importance (explainability) ---------------------------
    if hasattr(best_model, "feature_importances_"):
        importance = pd.Series(best_model.feature_importances_, index=X_encoded.columns)
        importance = importance.sort_values(ascending=False)
        importance.to_csv(REPORTS_DIR / "feature_importance.csv", header=["Importance"])
        logger.info("Top 10 risk drivers:\n%s", importance.head(10).to_string())
    elif hasattr(best_model, "coef_"):
        importance = pd.Series(np.abs(best_model.coef_[0]), index=X_encoded.columns)
        importance = importance.sort_values(ascending=False)
        importance.to_csv(REPORTS_DIR / "feature_importance.csv", header=["Importance"])


if __name__ == "__main__":
    main()
