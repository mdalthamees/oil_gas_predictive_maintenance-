"""
data_preprocessing.py
----------------------
Reusable cleaning pipeline for the Oil & Gas equipment sensor dataset.

This module is imported by:
    * notebooks (for EDA / model training)
    * src/train_failure_model.py, src/train_rul_model.py
    * app.py (to clean any CSV a user uploads, using the SAME rules
      that were used at training time)

Design goal: every cleaning decision lives in ONE place so training and
inference can never drift apart.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Column groups
# ----------------------------------------------------------------------
NUMERIC_SENSOR_COLUMNS = [
    "Operating_Hours", "Temperature_C", "Pressure_Bar", "Vibration_mm_s",
    "Flow_Rate_m3_h", "RPM", "Motor_Current_A", "Oil_Temperature_C",
    "Oil_Pressure_Bar", "Fuel_Gas_Pressure_Bar", "Ambient_Temperature_C",
    "Humidity_Percent", "Valve_Position_Percent", "Power_Consumption_kW",
    "Last_Maintenance_Days", "Failure_Count",
]

CATEGORICAL_COLUMNS = ["Equipment_Type", "Plant_Location", "Maintenance_History"]

# Physically-plausible bounds used to flag/clip impossible sensor readings.
# (These are deliberately generous synthetic-industry bounds, not real
# engineering specs, since this is a simulated dataset.)
VALID_RANGES = {
    "Temperature_C": (-20, 220),
    "Pressure_Bar": (0, 200),
    "Vibration_mm_s": (0, 20),
    "Flow_Rate_m3_h": (0, 600),
    "RPM": (0, 15000),
    "Motor_Current_A": (0, 250),
    "Oil_Temperature_C": (0, 180),
    "Oil_Pressure_Bar": (0, 15),
    "Fuel_Gas_Pressure_Bar": (0, 60),
    "Ambient_Temperature_C": (-10, 60),
    "Humidity_Percent": (0, 100),
    "Valve_Position_Percent": (0, 100),
    "Power_Consumption_kW": (0, 200),
    "Last_Maintenance_Days": (0, 1500),
    "Failure_Count": (0, 50),
}

REQUIRED_RAW_COLUMNS = [
    "Equipment_ID", "Equipment_Type", "Plant_Location", "Temperature_C",
    "Pressure_Bar", "Vibration_mm_s", "Flow_Rate_m3_h", "RPM",
    "Motor_Current_A", "Oil_Temperature_C", "Oil_Pressure_Bar",
    "Operating_Hours", "Last_Maintenance_Days", "Failure_Count",
]


class DataValidationError(Exception):
    """Raised when an uploaded / input CSV is missing required columns."""


def validate_columns(df: pd.DataFrame, required: list[str] | None = None) -> list[str]:
    """Return list of missing required columns (empty list = OK)."""
    required = required or REQUIRED_RAW_COLUMNS
    return [c for c in required if c not in df.columns]


def _clip_out_of_range(df: pd.DataFrame) -> pd.DataFrame:
    """Treat physically-impossible sensor spikes as missing (NaN) rather
    than silently clipping them, so later imputation handles them properly.
    """
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            n_bad = int(mask.sum())
            if n_bad:
                logger.info("  %s: flagging %d out-of-range readings as missing", col, n_bad)
                df.loc[mask, col] = np.nan
    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    logger.info("  Removed %d exact duplicate rows", before - len(df))
    return df


def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    for col in NUMERIC_SENSOR_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _impute_missing(df: pd.DataFrame, group_col: str = "Equipment_Type") -> pd.DataFrame:
    """Impute numeric sensor columns with the median *within equipment type*
    (different equipment types have very different normal operating ranges),
    falling back to the global median if a whole group is missing.
    """
    for col in NUMERIC_SENSOR_COLUMNS:
        if col not in df.columns:
            continue
        if df[col].isna().any():
            if group_col in df.columns:
                df[col] = df.groupby(group_col)[col].transform(lambda s: s.fillna(s.median()))
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].isna().any():
            mode = df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)

    return df


def _winsorize_outliers(df: pd.DataFrame, cols: list[str] | None = None, lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    """Cap extreme statistical outliers (post physically-impossible-value
    removal) using percentile winsorization, so a handful of extreme but
    *physically possible* readings don't destabilize model training.
    """
    cols = cols or [c for c in NUMERIC_SENSOR_COLUMNS if c in df.columns]
    for col in cols:
        lo, hi = df[col].quantile(lower_q), df[col].quantile(upper_q)
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def clean_raw_dataframe(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Full cleaning pipeline: dtype fixes -> invalid value flagging ->
    duplicate removal -> imputation -> outlier winsorization.

    This is safe to call on both the original training data AND on any
    new CSV a user uploads through the Streamlit app.
    """
    if verbose:
        logger.info("Starting data cleaning: %d rows, %d columns", len(df), df.shape[1])

    df = df.copy()
    df = _fix_dtypes(df)
    df = _remove_duplicates(df)
    df = _clip_out_of_range(df)
    df = _impute_missing(df)
    df = _winsorize_outliers(df)

    # Drop rows that still have no Equipment_ID / Equipment_Type -- these
    # cannot be meaningfully used downstream.
    essential = [c for c in ["Equipment_ID", "Equipment_Type"] if c in df.columns]
    if essential:
        before = len(df)
        df = df.dropna(subset=essential)
        if verbose and before != len(df):
            logger.info("  Dropped %d rows missing essential identifiers", before - len(df))

    df = df.reset_index(drop=True)

    if verbose:
        logger.info("Finished cleaning: %d rows, %d columns, %d remaining NaNs",
                     len(df), df.shape[1], int(df.isna().sum().sum()))
    return df


def load_and_clean(raw_path: str | Path, processed_path: str | Path | None = None) -> pd.DataFrame:
    """Convenience entry point used by notebooks/scripts: read the raw CSV,
    clean it, optionally persist the cleaned version, and return it.
    """
    raw_path = Path(raw_path)
    df = pd.read_csv(raw_path)
    df = clean_raw_dataframe(df)
    if processed_path is not None:
        processed_path = Path(processed_path)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)
        logger.info("Saved cleaned dataset -> %s", processed_path)
    return df


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    load_and_clean(
        raw_path=base / "data" / "raw" / "oil_gas_equipment_sensor_data.csv",
        processed_path=base / "data" / "processed" / "oil_gas_equipment_cleaned.csv",
    )
