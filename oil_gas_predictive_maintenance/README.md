# 🛢️ AI-Powered Oil & Gas Predictive Maintenance & Equipment Failure Detection

An end-to-end data science / AI portfolio project that simulates a **predictive maintenance platform** for Oil & Gas industrial equipment — oil pumps, gas compressors, turbines, centrifugal pumps, heat exchangers, drilling equipment, and valves.

> ⚠️ **All data in this project is synthetic / simulated.** It was generated specifically for this project and does not represent any real company, facility, equipment, or sensor feed. The models here have **not** been deployed at any real oil & gas facility — this is a demonstration / portfolio system.

---

## 1. Project Overview

Unplanned equipment downtime is one of the most expensive problems in Oil & Gas operations. This project builds a full machine-learning pipeline — from raw, messy sensor data to a production-style Streamlit dashboard — that:

1. Predicts whether a piece of equipment is likely to fail within 30 days.
2. Estimates the probability of failure.
3. Scores each asset's overall health (0–100).
4. Estimates Remaining Useful Life (RUL) in days.
5. Surfaces the model-attributed factors driving each risk prediction.
6. Ranks equipment by maintenance priority and recommends an action.

## 2. Business Problem

Maintenance teams typically operate reactively (fix after failure) or on fixed schedules (time-based maintenance), both of which are costly: reactive maintenance causes unplanned downtime and safety risk, while fixed-schedule maintenance wastes resources on healthy equipment. **Predictive maintenance** uses sensor data and machine learning to intervene only when a failure is actually likely — reducing downtime, safety incidents, and unnecessary maintenance spend.

## 3. Objectives

- Simulate a realistic (i.e., imperfect and noisy) Oil & Gas sensor dataset.
- Build a reusable, production-style data cleaning and feature engineering pipeline.
- Train and fairly compare multiple classification models for failure prediction, prioritizing **Recall** (missed failures are costly).
- Train a separate regression model to estimate Remaining Useful Life.
- Provide model explainability (feature importance / risk drivers) rather than a black box.
- Package everything into an interactive, professional Streamlit dashboard suitable for a Data Scientist / AI Engineer interview, including the ability to upload new data and download reports.

## 4. Architecture

```
oil_gas_predictive_maintenance/
│
├── data/
│   ├── raw/oil_gas_equipment_sensor_data.csv        # synthetic raw sensor data (~22,000 rows)
│   └── processed/oil_gas_equipment_cleaned.csv       # cleaned dataset
│
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_failure_prediction.ipynb
│   └── 06_rul_prediction.ipynb
│
├── src/
│   ├── generate_data.py            # synthetic dataset generator
│   ├── data_preprocessing.py       # reusable cleaning pipeline
│   ├── feature_engineering.py      # engineered feature definitions
│   ├── train_failure_model.py      # classification model training + selection
│   ├── train_rul_model.py          # RUL regression model training
│   └── prediction.py               # shared inference helpers used by app.py
│
├── models/
│   ├── failure_prediction_model.pkl
│   ├── rul_prediction_model.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   └── feature_names.pkl
│
├── reports/
│   ├── model_performance.csv
│   ├── rul_model_performance.csv
│   ├── feature_importance.csv
│   └── maintenance_predictions.csv
│
├── app.py                # Streamlit dashboard
├── requirements.txt
├── README.md
└── .gitignore
```

## 5. Dataset

A synthetic dataset of **22,000+ sensor readings** (after de-duplication: ~22,000 clean rows) across 550 simulated equipment assets, 7 equipment types, and 7 Gulf-region plant locations. Each asset has an underlying, hidden "degradation trajectory" so failures correlate realistically with vibration, temperature drift, maintenance history, and operating hours — rather than being purely random.

Columns include: `Equipment_ID`, `Equipment_Type`, `Plant_Location`, `Timestamp`, `Operating_Hours`, `Temperature_C`, `Pressure_Bar`, `Vibration_mm_s`, `Flow_Rate_m3_h`, `RPM`, `Motor_Current_A`, `Oil_Temperature_C`, `Oil_Pressure_Bar`, `Fuel_Gas_Pressure_Bar`, `Ambient_Temperature_C`, `Humidity_Percent`, `Valve_Position_Percent`, `Power_Consumption_kW`, `Maintenance_History`, `Last_Maintenance_Days`, `Failure_Count`, `Failure_Type`, `Equipment_Health_Score`, `Remaining_Useful_Life`, `Failure_Within_7_Days`, `Failure_Within_30_Days`.

The raw data intentionally includes: missing values (1–4.5% per sensor column), duplicate rows, physically-impossible sensor spikes/outliers, and negative/invalid readings — so the cleaning pipeline has genuine work to do, just like a real SCADA/historian export.

Regenerate it any time with:
```bash
python src/generate_data.py
```

## 6. Data Cleaning

`src/data_preprocessing.py` implements a single, reusable pipeline used identically at training time and at inference time (including for user-uploaded CSVs):

1. Fix data types (timestamps, numeric coercion).
2. Remove exact duplicate rows.
3. Flag physically-impossible readings (e.g., 400°C temperature, negative pressure) as missing.
4. Impute missing numeric values with the **median within equipment type** (falls back to global median).
5. Impute missing categorical values with the mode.
6. Winsorize remaining statistical outliers at the 1st/99th percentile.

## 7. Feature Engineering

`src/feature_engineering.py` adds 15 industry-oriented features on top of the cleaned data, including: `Temperature_Deviation`, `Pressure_Deviation`, `RPM_Deviation`, `Vibration_Risk`, `Operating_Hour_Risk`, `Maintenance_Due`, `Sensor_Anomaly_Count`, `Failure_Frequency`, `Average_Temperature`, `Average_Vibration`, `Pressure_Fluctuation`, `Power_Efficiency`, `Equipment_Age`, `Maintenance_Frequency`, and a composite `Health_Index` (0–100).

## 8. Machine Learning — Failure Prediction

`src/train_failure_model.py` trains and compares:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Extra Trees
- XGBoost

on `Failure_Within_30_Days`, evaluated with Accuracy, Precision, **Recall**, F1, ROC-AUC, and confusion-matrix-derived false negatives/positives. The final model is selected by ranking primarily on **Recall** (then F1, among models with above-median ROC-AUC) — because in predictive maintenance, a **missed failure (false negative)** is typically far more costly than a false alarm. See `reports/model_performance.csv` for the full comparison table produced by the latest training run.

The selected model, scaler, label encoders, and feature name list are saved with Joblib to `models/`.

## 9. Machine Learning — Remaining Useful Life (RUL)

`src/train_rul_model.py` trains and compares Random Forest Regressor, Gradient Boosting Regressor, and XGBoost Regressor on `Remaining_Useful_Life`, evaluated with MAE, RMSE, and R². The best model (by RMSE) is saved to `models/rul_prediction_model.pkl`. See `reports/rul_model_performance.csv`.

## 10. Explainable AI

Feature importances from the selected classification model are saved to `reports/feature_importance.csv` and surfaced in the app as **model-attributed risk drivers** (e.g., "High Vibration", "Increasing Oil Temperature", "Excessive Operating Hours"). These are explicitly presented as risk drivers the model has learned to weight heavily — **not** as proof of causation.

## 11. Equipment Health Score

A composite 0–100 `Health_Index` is engineered from vibration, temperature deviation, maintenance recency, maintenance history quality, and sensor anomaly count:

| Score Range | Band |
|---|---|
| 90–100 | Excellent |
| 75–89 | Good |
| 50–74 | Warning |
| 25–49 | Critical |
| 0–24 | Severe |

## 12. Streamlit Dashboard

Run with:
```bash
streamlit run app.py
```

The dashboard uses a dark navy industrial theme with orange/red risk indicators and includes 7 pages:

1. **Executive Dashboard** — fleet-wide KPIs, risk distribution, health distribution, failure trend, model performance summary.
2. **Equipment Monitoring** — select an Equipment ID, view its current readings, health/risk/RUL, and sensor trend history.
3. **AI Failure Prediction** — manual input form → PREDICT FAILURE → probability, risk level, top risk factors, recommendation.
4. **Predictive Maintenance** — filterable, sortable maintenance priority table with 🔴🟠🟡🟢 indicators and a CSV download.
5. **Sensor Analytics** — filterable interactive Plotly trend charts (temperature, pressure, vibration, RPM, flow, power) and a correlation heatmap.
6. **AI Insights** — global risk-factor chart, automated critical-equipment narratives, equipment due for maintenance, sensor anomaly summary.
7. **Upload & Predict** — upload your own equipment sensor CSV; the app validates columns, applies the same cleaning + feature pipeline used at training time, scores it with the saved models, and lets you download the results.

## 13. Installation & How to Run

```bash
# 1. Create environment and install dependencies
pip install -r requirements.txt

# 2. (Optional — pre-trained models are already included) Regenerate data & retrain models
python src/generate_data.py
python src/data_preprocessing.py
python src/feature_engineering.py
python src/train_failure_model.py
python src/train_rul_model.py

# 3. Launch the dashboard
streamlit run app.py
```

## 14. Screenshots

*(Add screenshots of the Executive Dashboard, AI Failure Prediction page, and Predictive Maintenance table here after running the app locally.)*

## 15. Future Improvements

- Integrate SHAP for true per-prediction (local) explanations instead of importance-based approximations.
- Add time-series/sequence models (e.g., LSTM, Temporal Convolutional Networks) to exploit the sensor history directly rather than snapshot features.
- Add automated model retraining / drift monitoring as new data arrives.
- Connect to a real historian/SCADA data source via a proper ETL layer.
- Add authentication and role-based access for a multi-plant deployment.

## 16. Business Impact (Illustrative)

In a real deployment, a system like this could help reduce unplanned downtime by flagging at-risk equipment before failure, better prioritize a maintenance team's limited time toward the highest-risk assets, and reduce unnecessary time-based maintenance on healthy equipment. These are illustrative potential benefits of the approach, not measured outcomes — this project has not been deployed or validated against real facility data.

## 17. Technologies Used

Python · Pandas · NumPy · Scikit-learn · XGBoost · Plotly · Matplotlib · Joblib · Streamlit

---

*Built as a portfolio / demonstration project. Not affiliated with, deployed at, or representative of any real oil & gas operator or facility.*
