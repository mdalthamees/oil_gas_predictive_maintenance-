"""
feature_engineering.py
------------------------
Builds industry-oriented engineered features on top of the cleaned sensor
data. Used identically at training time and prediction time so the feature
set the model sees is always consistent.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# "Normal" baseline sensor values per equipment type, used to compute
# deviation-based features. Derived from typical simulated operating
# profiles (see src/generate_data.py EQUIPMENT_PROFILES).
BASELINES = {
    "Oil Pump":            dict(temp=68,  pressure=45, vibration=2.4, rpm=1750),
    "Gas Compressor":      dict(temp=95,  pressure=85, vibration=3.1, rpm=6200),
    "Turbine":             dict(temp=140, pressure=65, vibration=3.6, rpm=9500),
    "Centrifugal Pump":    dict(temp=62,  pressure=38, vibration=2.1, rpm=2950),
    "Heat Exchanger":      dict(temp=110, pressure=30, vibration=1.2, rpm=0),
    "Drilling Equipment":  dict(temp=80,  pressure=120, vibration=4.2, rpm=120),
    "Valve":               dict(temp=55,  pressure=55, vibration=0.9, rpm=0),
}
DEFAULT_BASELINE = dict(temp=80, pressure=60, vibration=2.5, rpm=2000)


def _baseline_series(equipment_types: pd.Series, key: str) -> pd.Series:
    return equipment_types.map(lambda t: BASELINES.get(t, DEFAULT_BASELINE)[key])


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features required by the ML models. Assumes df has
    already been through `data_preprocessing.clean_raw_dataframe`.
    """
    df = df.copy()
    eq_type = df["Equipment_Type"] if "Equipment_Type" in df.columns else pd.Series(["Oil Pump"] * len(df))

    # ---- Deviation-from-normal features --------------------------------
    df["Temperature_Deviation"] = df["Temperature_C"] - _baseline_series(eq_type, "temp")
    df["Pressure_Deviation"] = df["Pressure_Bar"] - _baseline_series(eq_type, "pressure")
    df["RPM_Deviation"] = df["RPM"] - _baseline_series(eq_type, "rpm")

    # ---- Risk flags ------------------------------------------------------
    vib_baseline = _baseline_series(eq_type, "vibration")
    df["Vibration_Risk"] = (df["Vibration_mm_s"] > vib_baseline * 1.5).astype(int)
    df["Operating_Hour_Risk"] = (df["Operating_Hours"] > 5000).astype(int)
    df["Maintenance_Due"] = (df["Last_Maintenance_Days"] > 120).astype(int)

    # ---- Anomaly count: how many sensors are "out of normal band" ------
    anomaly_flags = pd.DataFrame({
        "temp": (df["Temperature_Deviation"].abs() > 20),
        "pressure": (df["Pressure_Deviation"].abs() > 15),
        "vibration": df["Vibration_Risk"].astype(bool),
        "oil_temp": (df["Oil_Temperature_C"] > 100),
        "current": (df["Motor_Current_A"] > df["Motor_Current_A"].median() * 1.6),
    })
    df["Sensor_Anomaly_Count"] = anomaly_flags.sum(axis=1)

    # ---- History / frequency features ------------------------------------
    df["Failure_Frequency"] = df["Failure_Count"] / (df["Operating_Hours"].clip(lower=1) / 1000)

    if "Equipment_ID" in df.columns:
        df["Average_Temperature"] = df.groupby("Equipment_ID")["Temperature_C"].transform("mean")
        df["Average_Vibration"] = df.groupby("Equipment_ID")["Vibration_mm_s"].transform("mean")
        df["Pressure_Fluctuation"] = df.groupby("Equipment_ID")["Pressure_Bar"].transform("std").fillna(0)
        df["Maintenance_Frequency"] = df.groupby("Equipment_ID")["Failure_Count"].transform("mean")
    else:
        df["Average_Temperature"] = df["Temperature_C"]
        df["Average_Vibration"] = df["Vibration_mm_s"]
        df["Pressure_Fluctuation"] = 0.0
        df["Maintenance_Frequency"] = df["Failure_Count"]

    # ---- Efficiency / age --------------------------------------------------
    df["Power_Efficiency"] = df["Flow_Rate_m3_h"] / df["Power_Consumption_kW"].clip(lower=0.1)
    df["Equipment_Age"] = df["Operating_Hours"] / 24.0  # approx age in "days of runtime"

    maint_map = {"Regular": 0, "Irregular": 1, "Poor": 2, "No_Records": 3, "Unknown": 2}
    if "Maintenance_History" in df.columns:
        maint_score = df["Maintenance_History"].map(maint_map).fillna(2)
    else:
        maint_score = pd.Series(1, index=df.index)

    # ---- Composite Health Index (0-100, higher = healthier) -------------
    norm_vib = (df["Vibration_mm_s"] / (vib_baseline.replace(0, 1) * 3)).clip(0, 1)
    norm_temp_dev = (df["Temperature_Deviation"].abs() / 60).clip(0, 1)
    norm_maint = (df["Last_Maintenance_Days"] / 300).clip(0, 1)
    norm_hist = (maint_score / 3).clip(0, 1)
    norm_anom = (df["Sensor_Anomaly_Count"] / 5).clip(0, 1)

    df["Health_Index"] = 100 * (1 - (
        0.30 * norm_vib + 0.20 * norm_temp_dev + 0.20 * norm_maint + 0.15 * norm_hist + 0.15 * norm_anom
    ))
    df["Health_Index"] = df["Health_Index"].clip(0, 100).round(1)

    return df


ENGINEERED_FEATURE_COLUMNS = [
    "Temperature_Deviation", "Pressure_Deviation", "RPM_Deviation",
    "Vibration_Risk", "Operating_Hour_Risk", "Maintenance_Due",
    "Sensor_Anomaly_Count", "Failure_Frequency", "Average_Temperature",
    "Average_Vibration", "Pressure_Fluctuation", "Power_Efficiency",
    "Equipment_Age", "Maintenance_Frequency", "Health_Index",
]

BASE_NUMERIC_FEATURES = [
    "Operating_Hours", "Temperature_C", "Pressure_Bar", "Vibration_mm_s",
    "Flow_Rate_m3_h", "RPM", "Motor_Current_A", "Oil_Temperature_C",
    "Oil_Pressure_Bar", "Fuel_Gas_Pressure_Bar", "Ambient_Temperature_C",
    "Humidity_Percent", "Valve_Position_Percent", "Power_Consumption_kW",
    "Last_Maintenance_Days", "Failure_Count",
]

CATEGORICAL_FEATURES = ["Equipment_Type", "Plant_Location", "Maintenance_History"]

ALL_MODEL_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_FEATURE_COLUMNS + CATEGORICAL_FEATURES


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    processed_path = base / "data" / "processed" / "oil_gas_equipment_cleaned.csv"
    df = pd.read_csv(processed_path)
    df_fe = add_engineered_features(df)
    out_path = base / "data" / "processed" / "oil_gas_equipment_features.csv"
    df_fe.to_csv(out_path, index=False)
    logger.info("Saved feature-engineered dataset -> %s (%d cols)", out_path, df_fe.shape[1])
