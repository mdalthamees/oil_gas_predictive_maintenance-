"""
generate_data.py
-----------------
Generates a realistic SYNTHETIC Oil & Gas equipment sensor dataset for the
Predictive Maintenance project.

IMPORTANT: This dataset is entirely simulated. It does not represent any
real company, facility, or equipment. It is designed to *feel* realistic
(noisy sensors, missing values, duplicates, outliers, mixed operating
conditions) so that the downstream cleaning / feature engineering / ML
pipeline has genuine work to do, just like a real industrial dataset would.

Run:
    python src/generate_data.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
N_ROWS = 22000

EQUIPMENT_TYPES = [
    "Oil Pump",
    "Gas Compressor",
    "Turbine",
    "Centrifugal Pump",
    "Heat Exchanger",
    "Drilling Equipment",
    "Valve",
]

PLANT_LOCATIONS = [
    "Abu Dhabi Onshore Field",
    "Dubai Processing Plant",
    "Ruwais Refinery",
    "Ghasha Offshore Complex",
    "Habshan Gas Plant",
    "Shah Gas Facility",
    "Jebel Ali Terminal",
]

FAILURE_TYPES = [
    "No_Failure",
    "Bearing Failure",
    "Seal Leakage",
    "Overheating",
    "Vibration Fault",
    "Corrosion",
    "Motor Failure",
    "Blockage",
    "Pressure Loss",
]

# Typical "normal operating" ranges per equipment type -> (mean, std)
# These are illustrative, not real engineering specs.
EQUIPMENT_PROFILES = {
    "Oil Pump": dict(temp=(68, 8), pressure=(45, 7), vibration=(2.4, 0.8), flow=(220, 35),
                      rpm=(1750, 150), current=(38, 6), oil_temp=(60, 7), oil_pressure=(4.2, 0.7)),
    "Gas Compressor": dict(temp=(95, 12), pressure=(85, 10), vibration=(3.1, 1.0), flow=(310, 45),
                            rpm=(6200, 400), current=(72, 10), oil_temp=(78, 9), oil_pressure=(5.8, 0.9)),
    "Turbine": dict(temp=(140, 18), pressure=(65, 9), vibration=(3.6, 1.2), flow=(180, 30),
                     rpm=(9500, 600), current=(95, 14), oil_temp=(85, 10), oil_pressure=(6.5, 1.0)),
    "Centrifugal Pump": dict(temp=(62, 7), pressure=(38, 6), vibration=(2.1, 0.7), flow=(260, 40),
                               rpm=(2950, 200), current=(42, 7), oil_temp=(55, 6), oil_pressure=(3.8, 0.6)),
    "Heat Exchanger": dict(temp=(110, 15), pressure=(30, 5), vibration=(1.2, 0.5), flow=(150, 25),
                            rpm=(0, 0), current=(18, 4), oil_temp=(50, 6), oil_pressure=(2.5, 0.5)),
    "Drilling Equipment": dict(temp=(80, 11), pressure=(120, 18), vibration=(4.2, 1.4), flow=(90, 20),
                                 rpm=(120, 25), current=(110, 16), oil_temp=(72, 9), oil_pressure=(7.2, 1.1)),
    "Valve": dict(temp=(55, 6), pressure=(55, 8), vibration=(0.9, 0.4), flow=(200, 30),
                   rpm=(0, 0), current=(8, 2), oil_temp=(40, 5), oil_pressure=(2.0, 0.4)),
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_PATH = OUTPUT_DIR / "oil_gas_equipment_sensor_data.csv"


def make_equipment_ids(n_equipment: int) -> pd.DataFrame:
    """Create a roster of unique equipment assets across types and plants."""
    prefixes = {
        "Oil Pump": "PUMP", "Gas Compressor": "COMP", "Turbine": "TURB",
        "Centrifugal Pump": "CFPUMP", "Heat Exchanger": "HEX",
        "Drilling Equipment": "DRILL", "Valve": "VALVE",
    }
    rows = []
    for i in range(n_equipment):
        eq_type = np.random.choice(EQUIPMENT_TYPES)
        plant = np.random.choice(PLANT_LOCATIONS)
        eq_id = f"{prefixes[eq_type]}-{100 + i}"
        install_age_days = np.random.randint(60, 4000)  # equipment age
        rows.append((eq_id, eq_type, plant, install_age_days))
    return pd.DataFrame(rows, columns=["Equipment_ID", "Equipment_Type", "Plant_Location", "Equipment_Age_Days"])


def generate_dataset(n_rows: int = N_ROWS) -> pd.DataFrame:
    logger.info("Generating equipment roster...")
    n_equipment = 550
    roster = make_equipment_ids(n_equipment)

    # Give each equipment asset a baseline "degradation trajectory" so that
    # failures are not purely random -- older / already-degraded assets are
    # more likely to fail, which is what makes RUL and health-score modeling
    # meaningful.
    roster["Base_Degradation"] = np.clip(
        np.random.beta(2, 5, size=len(roster)) + roster["Equipment_Age_Days"] / 8000, 0, 1
    )
    roster["Maintenance_History"] = np.random.choice(
        ["Regular", "Irregular", "Poor", "No_Records"], size=len(roster), p=[0.45, 0.30, 0.18, 0.07]
    )

    logger.info("Simulating %d sensor readings across %d assets...", n_rows, n_equipment)

    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2025-12-31")
    timestamps = pd.to_datetime(
        np.random.randint(start_date.value // 10**9, end_date.value // 10**9, size=n_rows), unit="s"
    )

    records = []
    equipment_running_hours = {eid: np.random.randint(500, 6000) for eid in roster["Equipment_ID"]}
    equipment_last_maint = {eid: np.random.randint(0, 220) for eid in roster["Equipment_ID"]}
    equipment_failure_count = {eid: np.random.poisson(1.1) for eid in roster["Equipment_ID"]}

    roster_indexed = roster.set_index("Equipment_ID")

    for i in range(n_rows):
        eq_id = np.random.choice(roster["Equipment_ID"].values)
        info = roster_indexed.loc[eq_id]
        eq_type = info["Equipment_Type"]
        plant = info["Plant_Location"]
        degradation = info["Base_Degradation"]
        maint_hist = info["Maintenance_History"]

        profile = EQUIPMENT_PROFILES[eq_type]

        # Degradation increases sensor drift: higher temp, pressure swings,
        # more vibration, lower efficiency.
        drift = degradation * np.random.uniform(0.6, 1.6)

        temperature = np.random.normal(profile["temp"][0] + drift * 18, profile["temp"][1])
        pressure = np.random.normal(profile["pressure"][0] - drift * 6, profile["pressure"][1])
        vibration = np.abs(np.random.normal(profile["vibration"][0] + drift * 3.2, profile["vibration"][1]))
        flow_rate = np.random.normal(profile["flow"][0] - drift * 25, profile["flow"][1])
        rpm = max(0, np.random.normal(profile["rpm"][0] - drift * profile["rpm"][0] * 0.08, profile["rpm"][1]))
        motor_current = np.random.normal(profile["current"][0] + drift * 9, profile["current"][1])
        oil_temp = np.random.normal(profile["oil_temp"][0] + drift * 14, profile["oil_temp"][1])
        oil_pressure = np.random.normal(profile["oil_pressure"][0] - drift * 1.4, profile["oil_pressure"][1])
        fuel_gas_pressure = np.random.normal(28 - drift * 4, 4.5)
        ambient_temp = np.random.normal(34, 6)  # Gulf-region ambient climate
        humidity = np.clip(np.random.normal(45, 15), 5, 95)
        valve_position = np.clip(np.random.normal(55 - drift * 10, 18), 0, 100)
        power_consumption = np.random.normal(motor_current * 0.42 + drift * 12, 8)

        operating_hours = equipment_running_hours[eq_id] + np.random.randint(0, 12)
        equipment_running_hours[eq_id] = operating_hours

        last_maint_days = equipment_last_maint[eq_id] + np.random.randint(0, 3)
        # occasional maintenance event resets the clock
        if np.random.rand() < 0.01:
            last_maint_days = np.random.randint(0, 5)
        equipment_last_maint[eq_id] = last_maint_days

        failure_count = equipment_failure_count[eq_id]

        maint_penalty = {"Regular": 0.0, "Irregular": 0.12, "Poor": 0.25, "No_Records": 0.35}[maint_hist]

        # ---- Failure risk score (latent) drives labels -----------------
        risk_score = (
            0.30 * drift
            + 0.15 * np.clip(vibration / 6.0, 0, 1)
            + 0.15 * np.clip((last_maint_days) / 250, 0, 1)
            + 0.10 * np.clip(operating_hours / 8000, 0, 1)
            + 0.10 * maint_penalty
            + 0.10 * np.clip((oil_temp - profile["oil_temp"][0]) / 40, 0, 1)
            + 0.10 * np.clip(failure_count / 5, 0, 1)
        )
        risk_score = np.clip(risk_score + np.random.normal(0, 0.06), 0, 1)

        failure_within_30 = 1 if np.random.rand() < risk_score * 0.9 else 0
        failure_within_7 = 1 if (failure_within_30 and np.random.rand() < risk_score * 0.55) else 0

        if failure_within_30:
            failure_type = np.random.choice(
                FAILURE_TYPES[1:], p=[0.20, 0.14, 0.16, 0.15, 0.10, 0.13, 0.07, 0.05]
            )
        else:
            failure_type = "No_Failure"

        health_score = np.clip(100 - risk_score * 100 + np.random.normal(0, 4), 0, 100)
        remaining_useful_life = np.clip(
            (1 - risk_score) * np.random.uniform(180, 420) - operating_hours * 0.01, 1, 500
        )

        records.append(
            dict(
                Equipment_ID=eq_id,
                Equipment_Type=eq_type,
                Plant_Location=plant,
                Timestamp=timestamps[i],
                Operating_Hours=operating_hours,
                Temperature_C=round(temperature, 2),
                Pressure_Bar=round(pressure, 2),
                Vibration_mm_s=round(vibration, 3),
                Flow_Rate_m3_h=round(flow_rate, 2),
                RPM=round(rpm, 1),
                Motor_Current_A=round(motor_current, 2),
                Oil_Temperature_C=round(oil_temp, 2),
                Oil_Pressure_Bar=round(oil_pressure, 2),
                Fuel_Gas_Pressure_Bar=round(fuel_gas_pressure, 2),
                Ambient_Temperature_C=round(ambient_temp, 2),
                Humidity_Percent=round(humidity, 1),
                Valve_Position_Percent=round(valve_position, 1),
                Power_Consumption_kW=round(power_consumption, 2),
                Maintenance_History=maint_hist,
                Last_Maintenance_Days=last_maint_days,
                Failure_Count=failure_count,
                Failure_Type=failure_type,
                Equipment_Health_Score=round(health_score, 1),
                Remaining_Useful_Life=round(remaining_useful_life, 1),
                Failure_Within_7_Days=failure_within_7,
                Failure_Within_30_Days=failure_within_30,
            )
        )

        if (i + 1) % 5000 == 0:
            logger.info("  ...%d / %d rows generated", i + 1, n_rows)

    df = pd.DataFrame(records)
    return df


def inject_data_quality_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Inject realistic messiness: missing values, duplicates, outliers,
    invalid entries -- exactly what a real SCADA/historian export would have.
    """
    df = df.copy()
    n = len(df)
    rng = np.random.default_rng(RANDOM_SEED)

    logger.info("Injecting missing values...")
    missing_cols = [
        "Temperature_C", "Pressure_Bar", "Vibration_mm_s", "Flow_Rate_m3_h",
        "RPM", "Motor_Current_A", "Oil_Temperature_C", "Oil_Pressure_Bar",
        "Fuel_Gas_Pressure_Bar", "Humidity_Percent", "Last_Maintenance_Days",
    ]
    for col in missing_cols:
        frac = rng.uniform(0.01, 0.045)
        idx = rng.choice(n, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    logger.info("Injecting sensor outliers/spikes...")
    outlier_specs = {
        "Temperature_C": (250, 400),
        "Pressure_Bar": (300, 500),
        "Vibration_mm_s": (25, 60),
        "Motor_Current_A": (300, 600),
        "RPM": (20000, 40000),
    }
    for col, (lo, hi) in outlier_specs.items():
        idx = rng.choice(n, size=int(n * 0.006), replace=False)
        df.loc[idx, col] = rng.uniform(lo, hi, size=len(idx))

    logger.info("Injecting a handful of negative/invalid sensor values...")
    for col in ["Pressure_Bar", "Flow_Rate_m3_h", "Vibration_mm_s", "Oil_Pressure_Bar"]:
        idx = rng.choice(n, size=int(n * 0.003), replace=False)
        df.loc[idx, col] = -np.abs(df.loc[idx, col])

    logger.info("Injecting duplicate records...")
    dup_idx = rng.choice(n, size=int(n * 0.02), replace=False)
    dup_rows = df.loc[dup_idx]
    df = pd.concat([df, dup_rows], ignore_index=True)

    logger.info("Shuffling row order...")
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(N_ROWS)
    df = inject_data_quality_issues(df)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved raw synthetic dataset -> %s  (%d rows, %d cols)", OUTPUT_PATH, len(df), df.shape[1])
    logger.info("Failure_Within_30_Days positive rate: %.2f%%", df["Failure_Within_30_Days"].mean() * 100)


if __name__ == "__main__":
    main()
