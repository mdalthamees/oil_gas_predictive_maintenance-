"""
app.py
-------
AI-Powered Oil & Gas Predictive Maintenance & Equipment Failure Detection
Streamlit dashboard.

Run:
    streamlit run app.py

NOTE: All data in this application is SYNTHETIC / SIMULATED and generated
for demonstration purposes. It does not represent any real company,
facility, or equipment, and this model has not been deployed at any real
oil & gas facility.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# Path setup so we can import from src/
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from data_preprocessing import DataValidationError, validate_columns  # noqa: E402
from prediction import (  # noqa: E402
    ModelArtifacts,
    health_band,
    maintenance_priority,
    predict_batch,
    predict_single,
    risk_level,
)

DATA_PROCESSED = BASE_DIR / "data" / "processed" / "oil_gas_equipment_cleaned.csv"
DATA_RAW = BASE_DIR / "data" / "raw" / "oil_gas_equipment_sensor_data.csv"
REPORTS_DIR = BASE_DIR / "reports"

# ----------------------------------------------------------------------
# Page config + theme
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Oil & Gas Predictive Maintenance",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --navy-bg: #0b1220;
    --panel-bg: #111a2b;
    --panel-bg-2: #16223a;
    --border-col: #223353;
    --accent-blue: #2f9bf0;
    --accent-orange: #ff8a3d;
    --accent-red: #ff4d4d;
    --accent-green: #35d07f;
    --text-main: #f2f5fa;
    --text-dim: #9fb0c9;
}

html, body, [class*="css"]  {
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

.stApp {
    background: radial-gradient(circle at top left, #0f1b30 0%, #070c16 65%);
    color: var(--text-main);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1122 0%, #060a14 100%);
    border-right: 1px solid var(--border-col);
}

.app-header {
    padding: 1.1rem 1.6rem;
    border-radius: 14px;
    background: linear-gradient(120deg, #0d1a30 0%, #13233f 60%, #0d1a30 100%);
    border: 1px solid var(--border-col);
    margin-bottom: 1.2rem;
}
.app-header h1 {
    margin: 0;
    font-size: 1.65rem;
    color: var(--text-main);
    letter-spacing: 0.3px;
}
.app-header p {
    margin: 0.2rem 0 0 0;
    color: var(--accent-blue);
    font-size: 0.95rem;
    font-weight: 500;
}
.synthetic-badge {
    display: inline-block;
    margin-top: 0.5rem;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    background: rgba(255, 138, 61, 0.15);
    border: 1px solid var(--accent-orange);
    color: var(--accent-orange);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.4px;
}

.kpi-card {
    background: linear-gradient(160deg, var(--panel-bg) 0%, var(--panel-bg-2) 100%);
    border: 1px solid var(--border-col);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    height: 100%;
}
.kpi-label {
    color: var(--text-dim);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 0.35rem;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text-main);
}
.kpi-sub {
    font-size: 0.78rem;
    color: var(--text-dim);
    margin-top: 0.2rem;
}

.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-main);
    border-left: 4px solid var(--accent-blue);
    padding-left: 0.6rem;
    margin: 1.4rem 0 0.7rem 0;
}

.risk-pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.4px;
}

.info-panel {
    background: var(--panel-bg);
    border: 1px solid var(--border-col);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}

.insight-card {
    background: var(--panel-bg);
    border-left: 4px solid var(--accent-orange);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    color: var(--text-main);
    font-size: 0.92rem;
}

hr { border-color: var(--border-col); }

.stButton>button {
    background: linear-gradient(120deg, var(--accent-blue), #1d6fc4);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
}
.stButton>button:hover {
    background: linear-gradient(120deg, #47aefc, var(--accent-blue));
    color: white;
}

[data-testid="stMetricValue"] {
    color: var(--text-main);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
COLOR_SEQUENCE = ["#2f9bf0", "#ff8a3d", "#35d07f", "#ff4d4d", "#a78bfa", "#f1c40f", "#1abc9c"]
RISK_COLOR_MAP = {"LOW": "#35d07f", "MEDIUM": "#f1c40f", "HIGH": "#ff8a3d", "CRITICAL": "#ff4d4d"}


# ----------------------------------------------------------------------
# Cached data / model loading
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_processed_data() -> pd.DataFrame:
    if DATA_PROCESSED.exists():
        df = pd.read_csv(DATA_PROCESSED)
    else:
        df = pd.read_csv(DATA_RAW)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_maintenance_report() -> pd.DataFrame | None:
    path = REPORTS_DIR / "maintenance_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_model_performance() -> pd.DataFrame | None:
    path = REPORTS_DIR / "model_performance.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_resource(show_spinner=False)
def get_model_artifacts():
    if not ModelArtifacts.available():
        return None
    try:
        return ModelArtifacts.get()
    except Exception as exc:  # pragma: no cover
        st.error(f"Could not load model artifacts: {exc}")
        return None


@st.cache_data(show_spinner=False)
def score_fleet(df: pd.DataFrame) -> pd.DataFrame:
    """Run the prediction pipeline over the full (or sampled) fleet so the
    dashboard pages have live model outputs even without the pre-baked
    maintenance report.
    """
    try:
        return predict_batch(df)
    except DataValidationError:
        return pd.DataFrame()


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🛢️ AI-Powered Oil &amp; Gas Predictive Maintenance</h1>
        <p>Intelligent Equipment Failure Detection &amp; Asset Health Monitoring</p>
        <span class="synthetic-badge">SYNTHETIC / SIMULATED DATA — DEMO SYSTEM</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Go to",
        [
            "📊 Executive Dashboard",
            "🔧 Equipment Monitoring",
            "🤖 AI Failure Prediction",
            "🛠️ Predictive Maintenance",
            "📈 Sensor Analytics",
            "💡 AI Insights",
            "📤 Upload & Predict",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "This platform runs on a synthetic Oil & Gas equipment sensor "
        "dataset and machine-learning models trained for demonstration "
        "purposes. It is not connected to any real facility."
    )
    artifacts_ready = ModelArtifacts.available()
    if artifacts_ready:
        st.success("Models loaded ✅", icon="✅")
    else:
        st.warning("Model files not found — run the training scripts in src/ first.", icon="⚠️")

# ----------------------------------------------------------------------
# Shared data load
# ----------------------------------------------------------------------
try:
    fleet_df = load_processed_data()
except FileNotFoundError:
    st.error(
        "No dataset found. Please run `python src/generate_data.py` and "
        "`python src/data_preprocessing.py` first, or upload a CSV on the "
        "Upload & Predict page."
    )
    st.stop()

maintenance_report = load_maintenance_report()
model_perf = load_model_performance()
artifacts = get_model_artifacts()


def kpi_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_pill(label: str):
    color = RISK_COLOR_MAP.get(label, "#9fb0c9")
    st.markdown(
        f'<span class="risk-pill" style="background:{color}22;color:{color};'
        f'border:1px solid {color};">{label}</span>',
        unsafe_allow_html=True,
    )


# ========================================================================
# PAGE 1 — EXECUTIVE DASHBOARD
# ========================================================================
if page == "📊 Executive Dashboard":
    st.markdown('<div class="section-title">Fleet Overview</div>', unsafe_allow_html=True)

    if maintenance_report is not None and not maintenance_report.empty:
        mreport = maintenance_report
    else:
        latest = fleet_df.sort_values("Timestamp").groupby("Equipment_ID").tail(1) if "Timestamp" in fleet_df.columns else fleet_df.drop_duplicates("Equipment_ID")
        mreport = score_fleet(latest)

    if mreport.empty or artifacts is None:
        st.info("Model artifacts not available yet. Train the models (see README) to populate live KPIs.")
    else:
        total_equipment = mreport["Equipment_ID"].nunique()
        healthy = (mreport["Risk_Level"] == "LOW").sum()
        at_risk = mreport["Risk_Level"].isin(["MEDIUM", "HIGH"]).sum()
        critical = (mreport["Risk_Level"] == "CRITICAL").sum()
        predicted_failures = (mreport["Failure_Probability"] >= 50).sum()
        avg_health = mreport["Health_Score"].mean()

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            kpi_card("Total Equipment", f"{total_equipment:,}")
        with c2:
            kpi_card("Healthy Equipment", f"{healthy:,}", "Risk Level: LOW")
        with c3:
            kpi_card("At-Risk Equipment", f"{at_risk:,}", "Risk Level: MEDIUM / HIGH")
        with c4:
            kpi_card("Critical Equipment", f"{critical:,}", "Risk Level: CRITICAL")
        with c5:
            kpi_card("Predicted Failures", f"{predicted_failures:,}", "Failure Probability ≥ 50%")
        with c6:
            kpi_card("Avg Health Score", f"{avg_health:.1f}", "out of 100")

        st.markdown('<div class="section-title">Fleet Risk &amp; Health Analytics</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            risk_counts = mreport["Risk_Level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0)
            fig = go.Figure(
                data=[go.Pie(
                    labels=risk_counts.index, values=risk_counts.values, hole=0.55,
                    marker=dict(colors=[RISK_COLOR_MAP[k] for k in risk_counts.index]),
                )]
            )
            fig.update_layout(template=PLOTLY_TEMPLATE, title="Equipment Risk Distribution",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                mreport, x="Health_Score", nbins=25, color_discrete_sequence=[COLOR_SEQUENCE[0]],
                title="Equipment Health Score Distribution",
            )
            fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", height=380, xaxis_title="Health Score", yaxis_title="Equipment Count")
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            by_type = mreport.groupby("Equipment_Type")["Failure_Probability"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(
                by_type, x="Equipment_Type", y="Failure_Probability", color="Failure_Probability",
                color_continuous_scale=["#35d07f", "#f1c40f", "#ff8a3d", "#ff4d4d"],
                title="Average Failure Probability by Equipment Type",
            )
            fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", height=380, yaxis_title="Avg Failure Probability (%)")
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            if "Timestamp" in fleet_df.columns:
                trend = fleet_df.copy()
                trend["Month"] = trend["Timestamp"].dt.to_period("M").astype(str)
                trend_agg = trend.groupby("Month")["Failure_Within_30_Days"].mean().reset_index()
                trend_agg["Failure_Within_30_Days"] *= 100
                fig = px.line(
                    trend_agg, x="Month", y="Failure_Within_30_Days", markers=True,
                    color_discrete_sequence=[COLOR_SEQUENCE[1]], title="Failure Rate Trend Over Time",
                )
                fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)", height=380, yaxis_title="Failure-Within-30-Days Rate (%)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No timestamp column available for trend analysis.")

        st.markdown('<div class="section-title">Model Performance Summary</div>', unsafe_allow_html=True)
        if model_perf is not None:
            st.dataframe(model_perf, use_container_width=True, hide_index=True)
        else:
            st.info("Run `python src/train_failure_model.py` to generate the model comparison report.")


# ========================================================================
# PAGE 2 — EQUIPMENT MONITORING
# ========================================================================
elif page == "🔧 Equipment Monitoring":
    st.markdown('<div class="section-title">Select Equipment</div>', unsafe_allow_html=True)

    equipment_ids = sorted(fleet_df["Equipment_ID"].unique())
    selected_id = st.selectbox("Equipment ID", equipment_ids)

    eq_history = fleet_df[fleet_df["Equipment_ID"] == selected_id].sort_values("Timestamp")
    if eq_history.empty:
        st.warning("No records found for this equipment.")
        st.stop()

    latest_row = eq_history.iloc[-1]

    if artifacts is not None:
        pred = score_fleet(pd.DataFrame([latest_row]))
    else:
        pred = pd.DataFrame()

    st.markdown('<div class="section-title">Equipment Summary</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Equipment Type", str(latest_row["Equipment_Type"]))
        kpi_card("Plant Location", str(latest_row["Plant_Location"]))
    with c2:
        if not pred.empty:
            kpi_card("Health Score", f"{pred.iloc[0]['Health_Score']:.1f}", health_band(pred.iloc[0]['Health_Score'])[0])
            kpi_card("Failure Probability", f"{pred.iloc[0]['Failure_Probability']:.1f}%")
        else:
            kpi_card("Health Score", "N/A")
            kpi_card("Failure Probability", "N/A")
    with c3:
        if not pred.empty:
            kpi_card("Remaining Useful Life", f"{pred.iloc[0]['Remaining_Useful_Life']:.0f} days")
            risk_pill(pred.iloc[0]["Risk_Level"])
        else:
            kpi_card("Remaining Useful Life", "N/A")

    st.markdown('<div class="section-title">Current Sensor Readings</div>', unsafe_allow_html=True)
    sensor_cols = st.columns(4)
    sensor_fields = [
        ("Temperature", "Temperature_C", "°C"), ("Pressure", "Pressure_Bar", "bar"),
        ("Vibration", "Vibration_mm_s", "mm/s"), ("RPM", "RPM", "rpm"),
        ("Operating Hours", "Operating_Hours", "hrs"), ("Oil Temperature", "Oil_Temperature_C", "°C"),
        ("Motor Current", "Motor_Current_A", "A"), ("Maintenance Status", "Maintenance_History", ""),
    ]
    for i, (label, col, unit) in enumerate(sensor_fields):
        with sensor_cols[i % 4]:
            val = latest_row[col]
            display = f"{val:.1f} {unit}".strip() if isinstance(val, (int, float, np.floating)) else str(val)
            kpi_card(label, display)

    st.markdown('<div class="section-title">Sensor Trend History</div>', unsafe_allow_html=True)
    trend_metric = st.selectbox(
        "Select sensor to plot over time",
        ["Temperature_C", "Pressure_Bar", "Vibration_mm_s", "RPM", "Motor_Current_A",
         "Oil_Temperature_C", "Power_Consumption_kW"],
    )
    fig = px.line(
        eq_history, x="Timestamp", y=trend_metric, markers=True,
        color_discrete_sequence=[COLOR_SEQUENCE[0]], title=f"{trend_metric} Trend — {selected_id}",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", height=420)
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.line(eq_history, x="Timestamp", y="Vibration_mm_s", markers=True,
                       color_discrete_sequence=[COLOR_SEQUENCE[3]], title="Vibration Over Time")
        fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=340)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.line(eq_history, x="Timestamp", y="Oil_Temperature_C", markers=True,
                       color_discrete_sequence=[COLOR_SEQUENCE[1]], title="Oil Temperature Over Time")
        fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=340)
        st.plotly_chart(fig, use_container_width=True)


# ========================================================================
# PAGE 3 — AI FAILURE PREDICTION
# ========================================================================
elif page == "🤖 AI Failure Prediction":
    st.markdown('<div class="section-title">Enter Equipment Sensor Readings</div>', unsafe_allow_html=True)

    if artifacts is None:
        st.error("Model artifacts not found. Please train the models first (see README).")
        st.stop()

    equipment_types = ["Oil Pump", "Gas Compressor", "Turbine", "Centrifugal Pump",
                        "Heat Exchanger", "Drilling Equipment", "Valve"]

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            equipment_type = st.selectbox("Equipment Type", equipment_types)
            temperature = st.number_input("Temperature (°C)", value=85.0, step=1.0)
            pressure = st.number_input("Pressure (Bar)", value=55.0, step=1.0)
            vibration = st.number_input("Vibration (mm/s)", value=2.5, step=0.1)
        with col2:
            flow_rate = st.number_input("Flow Rate (m³/h)", value=220.0, step=5.0)
            rpm = st.number_input("RPM", value=3000.0, step=50.0)
            motor_current = st.number_input("Motor Current (A)", value=45.0, step=1.0)
            oil_temperature = st.number_input("Oil Temperature (°C)", value=65.0, step=1.0)
        with col3:
            oil_pressure = st.number_input("Oil Pressure (Bar)", value=4.5, step=0.1)
            operating_hours = st.number_input("Operating Hours", value=3500, step=100)
            last_maintenance_days = st.number_input("Last Maintenance (days ago)", value=60, step=5)
            failure_count = st.number_input("Historical Failure Count", value=0, step=1, min_value=0)

        submitted = st.form_submit_button("⚡ PREDICT FAILURE", use_container_width=True)

    if submitted:
        input_dict = {
            "Equipment_Type": equipment_type,
            "Temperature_C": temperature,
            "Pressure_Bar": pressure,
            "Vibration_mm_s": vibration,
            "Flow_Rate_m3_h": flow_rate,
            "RPM": rpm,
            "Motor_Current_A": motor_current,
            "Oil_Temperature_C": oil_temperature,
            "Oil_Pressure_Bar": oil_pressure,
            "Operating_Hours": operating_hours,
            "Last_Maintenance_Days": last_maintenance_days,
            "Failure_Count": failure_count,
        }
        with st.spinner("Running AI prediction..."):
            try:
                result = predict_single(input_dict)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

        st.markdown("---")
        verdict = "FAILURE RISK" if result["failure_probability"] >= 50 else "NORMAL"
        verdict_color = "#ff4d4d" if verdict == "FAILURE RISK" else "#35d07f"

        c1, c2, c3 = st.columns([1.3, 1, 1])
        with c1:
            st.markdown(
                f"""
                <div class="kpi-card" style="border-color:{verdict_color};">
                    <div class="kpi-label">Prediction</div>
                    <div class="kpi-value" style="color:{verdict_color};">{verdict}</div>
                    <div class="kpi-sub">Failure Probability: {result['failure_probability']:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            kpi_card("Risk Level", result["risk_level"])
        with c3:
            kpi_card("Estimated RUL", f"{result['remaining_useful_life']:.0f} days")

        c4, c5 = st.columns(2)
        with c4:
            kpi_card("Health Score", f"{result['health_score']:.1f} / 100", health_band(result["health_score"])[0])
        with c5:
            kpi_card("Maintenance Priority", result["maintenance_priority"])

        st.markdown('<div class="section-title">Top Risk Factors (Model-Attributed)</div>', unsafe_allow_html=True)
        st.caption("These are the model's most influential features for this type of prediction, ranked by "
                   "global importance and adjusted for this reading — not a proven causal explanation.")
        for i, factor in enumerate(result["top_factors"], start=1):
            st.markdown(f'<div class="insight-card">{i}. {factor}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Recommendation</div>', unsafe_allow_html=True)
        if verdict == "FAILURE RISK":
            st.error(f"⚠️ {result['recommended_action']}")
        else:
            st.success(f"✅ {result['recommended_action']}")


# ========================================================================
# PAGE 4 — PREDICTIVE MAINTENANCE
# ========================================================================
elif page == "🛠️ Predictive Maintenance":
    st.markdown('<div class="section-title">Maintenance Priority Table</div>', unsafe_allow_html=True)

    if maintenance_report is not None and not maintenance_report.empty:
        mreport = maintenance_report.copy()
    elif artifacts is not None:
        latest = fleet_df.sort_values("Timestamp").groupby("Equipment_ID").tail(1) if "Timestamp" in fleet_df.columns else fleet_df.drop_duplicates("Equipment_ID")
        mreport = score_fleet(latest)
    else:
        mreport = pd.DataFrame()

    if mreport.empty:
        st.info("Model artifacts not available yet. Train the models to populate this table.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            type_filter = st.multiselect("Filter by Equipment Type", sorted(mreport["Equipment_Type"].unique()))
        with col2:
            risk_filter = st.multiselect("Filter by Risk Level", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        with col3:
            plant_filter = st.multiselect("Filter by Plant", sorted(mreport["Plant_Location"].unique()))

        filtered = mreport.copy()
        if type_filter:
            filtered = filtered[filtered["Equipment_Type"].isin(type_filter)]
        if risk_filter:
            filtered = filtered[filtered["Risk_Level"].isin(risk_filter)]
        if plant_filter:
            filtered = filtered[filtered["Plant_Location"].isin(plant_filter)]

        risk_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        filtered["_sort"] = filtered["Risk_Level"].map(risk_order)
        filtered = filtered.sort_values(["_sort", "Failure_Probability"], ascending=[True, False]).drop(columns="_sort")

        display_cols = ["Equipment_ID", "Equipment_Type", "Health_Score", "Failure_Probability",
                         "Remaining_Useful_Life", "Risk_Level", "Maintenance_Priority", "Recommended_Action"]
        st.dataframe(
            filtered[display_cols].rename(columns={
                "Equipment_ID": "Equipment ID", "Equipment_Type": "Equipment Type",
                "Health_Score": "Health Score", "Failure_Probability": "Failure Probability (%)",
                "Remaining_Useful_Life": "RUL (days)", "Risk_Level": "Risk Level",
                "Maintenance_Priority": "Priority", "Recommended_Action": "Recommended Action",
            }),
            use_container_width=True, hide_index=True, height=520,
        )

        st.download_button(
            "⬇️ Download Maintenance Report",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="maintenance_report.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ========================================================================
# PAGE 5 — SENSOR ANALYTICS
# ========================================================================
elif page == "📈 Sensor Analytics":
    st.markdown('<div class="section-title">Filters</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        eq_filter = st.multiselect("Equipment", sorted(fleet_df["Equipment_ID"].unique()))
    with c2:
        type_filter = st.multiselect("Equipment Type", sorted(fleet_df["Equipment_Type"].unique()))
    with c3:
        plant_filter = st.multiselect("Plant", sorted(fleet_df["Plant_Location"].unique()))
    with c4:
        if "Timestamp" in fleet_df.columns and fleet_df["Timestamp"].notna().any():
            min_d, max_d = fleet_df["Timestamp"].min(), fleet_df["Timestamp"].max()
            date_range = st.date_input("Date Range", value=(min_d.date(), max_d.date()))
        else:
            date_range = None

    filtered = fleet_df.copy()
    if eq_filter:
        filtered = filtered[filtered["Equipment_ID"].isin(eq_filter)]
    if type_filter:
        filtered = filtered[filtered["Equipment_Type"].isin(type_filter)]
    if plant_filter:
        filtered = filtered[filtered["Plant_Location"].isin(plant_filter)]
    if date_range and isinstance(date_range, tuple) and len(date_range) == 2 and "Timestamp" in filtered.columns:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        filtered = filtered[(filtered["Timestamp"] >= start) & (filtered["Timestamp"] <= end)]

    if filtered.empty:
        st.warning("No data matches the selected filters.")
    else:
        filtered = filtered.sort_values("Timestamp") if "Timestamp" in filtered.columns else filtered

        st.markdown('<div class="section-title">Sensor Trends</div>', unsafe_allow_html=True)
        metrics = [
            ("Temperature Trend", "Temperature_C"), ("Pressure Trend", "Pressure_Bar"),
            ("Vibration Trend", "Vibration_mm_s"), ("RPM Trend", "RPM"),
            ("Flow Rate Trend", "Flow_Rate_m3_h"), ("Power Consumption Trend", "Power_Consumption_kW"),
        ]
        for i in range(0, len(metrics), 2):
            cols = st.columns(2)
            for j, (title, metric) in enumerate(metrics[i:i + 2]):
                with cols[j]:
                    agg = filtered.groupby(filtered["Timestamp"].dt.to_period("M").astype(str))[metric].mean().reset_index()
                    fig = px.line(agg, x="Timestamp", y=metric, markers=True,
                                   color_discrete_sequence=[COLOR_SEQUENCE[(i + j) % len(COLOR_SEQUENCE)]], title=title)
                    fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                                       plot_bgcolor="rgba(0,0,0,0)", height=340, xaxis_title="Month")
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Sensor Correlation Matrix</div>', unsafe_allow_html=True)
        corr_cols = ["Temperature_C", "Pressure_Bar", "Vibration_mm_s", "Flow_Rate_m3_h", "RPM",
                     "Motor_Current_A", "Oil_Temperature_C", "Power_Consumption_kW"]
        corr = filtered[corr_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Sensor Correlation Heatmap")
        fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", height=480)
        st.plotly_chart(fig, use_container_width=True)


# ========================================================================
# PAGE 6 — AI INSIGHTS
# ========================================================================
elif page == "💡 AI Insights":
    st.markdown('<div class="section-title">Automated Fleet Insights</div>', unsafe_allow_html=True)

    if maintenance_report is not None and not maintenance_report.empty:
        mreport = maintenance_report.copy()
    elif artifacts is not None:
        latest = fleet_df.sort_values("Timestamp").groupby("Equipment_ID").tail(1) if "Timestamp" in fleet_df.columns else fleet_df.drop_duplicates("Equipment_ID")
        mreport = score_fleet(latest)
    else:
        mreport = pd.DataFrame()

    if mreport.empty:
        st.info("Model artifacts not available yet. Train the models to populate insights.")
    else:
        st.markdown('<div class="section-title">Top Global Risk Factors</div>', unsafe_allow_html=True)
        if artifacts is not None and artifacts.global_importance is not None:
            top_global = artifacts.global_importance.head(8)
            fig = px.bar(
                x=top_global.values, y=[n.replace("_", " ") for n in top_global.index], orientation="h",
                color=top_global.values, color_continuous_scale=["#2f9bf0", "#ff8a3d", "#ff4d4d"],
                title="Model Feature Importance (Global)",
            )
            fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", height=420, xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance report not available.")

        st.markdown('<div class="section-title">Automated Recommendations</div>', unsafe_allow_html=True)
        critical_eq = mreport[mreport["Risk_Level"] == "CRITICAL"].sort_values("Failure_Probability", ascending=False).head(8)

        if critical_eq.empty:
            st.markdown('<div class="insight-card">✅ No equipment currently at CRITICAL risk level.</div>', unsafe_allow_html=True)
        else:
            for _, row in critical_eq.iterrows():
                msg = (
                    f"<b>{row['Equipment_ID']}</b> ({row['Equipment_Type']}, {row['Plant_Location']}) has an elevated "
                    f"failure probability of <b>{row['Failure_Probability']:.0f}%</b> with an estimated remaining useful "
                    f"life of <b>{row['Remaining_Useful_Life']:.0f} days</b>. Health score: {row['Health_Score']:.0f}/100. "
                    f"{row['Recommended_Action']}."
                )
                st.markdown(f'<div class="insight-card">⚠️ {msg}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Equipment Requiring Maintenance Soon</div>', unsafe_allow_html=True)
        soon = mreport[mreport["Remaining_Useful_Life"] <= 30].sort_values("Remaining_Useful_Life").head(10)
        if soon.empty:
            st.markdown('<div class="insight-card">✅ No equipment projected to require maintenance within 30 days.</div>', unsafe_allow_html=True)
        else:
            st.dataframe(
                soon[["Equipment_ID", "Equipment_Type", "Plant_Location", "Remaining_Useful_Life", "Risk_Level"]]
                .rename(columns={"Equipment_ID": "Equipment ID", "Equipment_Type": "Type",
                                  "Plant_Location": "Plant", "Remaining_Useful_Life": "RUL (days)",
                                  "Risk_Level": "Risk"}),
                use_container_width=True, hide_index=True,
            )

        st.markdown('<div class="section-title">Sensor Anomaly Summary</div>', unsafe_allow_html=True)
        if "Sensor_Anomaly_Count" in fleet_df.columns:
            anomaly_by_type = fleet_df.groupby("Equipment_Type")["Vibration_mm_s"].apply(
                lambda s: (s > s.quantile(0.95)).sum()
            ).sort_values(ascending=False).reset_index(name="High_Vibration_Events")
            fig = px.bar(anomaly_by_type, x="Equipment_Type", y="High_Vibration_Events",
                          color_discrete_sequence=[COLOR_SEQUENCE[3]], title="High-Vibration Anomaly Events by Equipment Type")
            fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", height=380)
            st.plotly_chart(fig, use_container_width=True)


# ========================================================================
# PAGE 7 — UPLOAD YOUR OWN DATASET
# ========================================================================
elif page == "📤 Upload & Predict":
    st.markdown('<div class="section-title">Upload Equipment Sensor CSV</div>', unsafe_allow_html=True)
    st.caption(
        "Upload a CSV with equipment sensor readings to generate AI failure risk predictions. "
        "Required columns: Equipment_ID, Equipment_Type, Plant_Location, Temperature_C, Pressure_Bar, "
        "Vibration_mm_s, Flow_Rate_m3_h, RPM, Motor_Current_A, Oil_Temperature_C, Oil_Pressure_Bar, "
        "Operating_Hours, Last_Maintenance_Days, Failure_Count."
    )

    if artifacts is None:
        st.error("Model artifacts not found. Please train the models first (see README) before using this page.")
        st.stop()

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the uploaded file as CSV: {exc}")
            st.stop()

        st.markdown('<div class="section-title">Validation</div>', unsafe_allow_html=True)
        missing_cols = validate_columns(user_df)
        if missing_cols:
            st.error(f"❌ The uploaded file is missing required columns: {', '.join(missing_cols)}")
        else:
            st.success(f"✅ File validated: {len(user_df):,} rows, {user_df.shape[1]} columns.")

            with st.spinner("Cleaning data, applying the trained pipeline, and generating predictions..."):
                try:
                    predictions = predict_batch(user_df)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    predictions = pd.DataFrame()

            if not predictions.empty:
                st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    kpi_card("Rows Scored", f"{len(predictions):,}")
                with c2:
                    kpi_card("Critical Risk", f"{(predictions['Risk_Level'] == 'CRITICAL').sum():,}")
                with c3:
                    kpi_card("High Risk", f"{(predictions['Risk_Level'] == 'HIGH').sum():,}")
                with c4:
                    kpi_card("Avg Failure Probability", f"{predictions['Failure_Probability'].mean():.1f}%")

                risk_counts = predictions["Risk_Level"].value_counts().reindex(
                    ["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0)
                fig = go.Figure(data=[go.Pie(
                    labels=risk_counts.index, values=risk_counts.values, hole=0.55,
                    marker=dict(colors=[RISK_COLOR_MAP[k] for k in risk_counts.index]),
                )])
                fig.update_layout(template=PLOTLY_TEMPLATE, title="Uploaded Fleet — Risk Distribution",
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(predictions, use_container_width=True, hide_index=True, height=420)

                st.download_button(
                    "⬇️ Download Prediction Results",
                    data=predictions.to_csv(index=False).encode("utf-8"),
                    file_name="uploaded_equipment_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    else:
        st.markdown('<div class="info-panel">Upload a CSV file above to get started, or use the sample raw '
                     'dataset at <code>data/raw/oil_gas_equipment_sensor_data.csv</code> as a template.</div>',
                     unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown("---")
st.caption(
    "AI-Powered Oil & Gas Predictive Maintenance — demonstration project built with synthetic data. "
    "Not affiliated with, deployed at, or representative of any real oil & gas operator or facility."
)
