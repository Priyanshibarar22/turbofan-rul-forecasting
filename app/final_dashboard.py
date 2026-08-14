import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import shap
import time
from pathlib import Path

# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Turbofan RUL Dashboard",
    layout="wide"
)

st.title("Turbofan Engine Health Monitor")
st.caption(
    "Real-time RUL prediction, explainability, and drift monitoring "
    "for a fleet of turbofan engines."
)


# ================================================================
# PROJECT PATH
# ================================================================

# dashboard.py is inside /app
# parent.parent points to the project root

BASE_DIR = Path(__file__).resolve().parent.parent


# ================================================================
# MODEL & DATA LOADING
# ================================================================

@st.cache_resource
def load_model():
    model_path = BASE_DIR / "models" / "xgb_fd001.pkl"

    if not model_path.exists():
        st.error(
            f"Model file not found: {model_path}"
        )
        st.stop()

    return joblib.load(model_path)


@st.cache_data
def load_data():

    val_path = BASE_DIR / "data" / "processed" / "val_FD001.csv"
    train_path = BASE_DIR / "data" / "processed" / "train_FD001.csv"

    if not val_path.exists():
        st.error(
            f"Validation data not found: {val_path}"
        )
        st.stop()

    if not train_path.exists():
        st.error(
            f"Training data not found: {train_path}"
        )
        st.stop()

    val = pd.read_csv(val_path)
    train = pd.read_csv(train_path)

    return val, train


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


# Load everything once

model = load_model()
val_data, train_data = load_data()
explainer = get_explainer(model)


# ================================================================
# FEATURE SELECTION
# ================================================================

feature_cols = [
    c for c in val_data.columns
    if c not in [
        "unit",
        "RUL",
        "dataset",
        "op_condition",
        "cycle_original"
    ]
]

engine_ids = sorted(
    val_data["unit"].unique()
)


# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.header("Engine Selection")

selected_engine = st.sidebar.selectbox(
    "Select Engine",
    engine_ids
)


# ================================================================
# FLEET OVERVIEW
# ================================================================

st.divider()

st.header("Fleet Overview")

st.caption(
    "Current health status and predicted Remaining Useful Life "
    "across the turbofan engine fleet."
)


# One batched prediction for every engine's latest cycle

latest_per_engine = (
    val_data
    .sort_values("cycle")
    .groupby("unit")
    .tail(1)
    .copy()
)

latest_preds = model.predict(
    latest_per_engine[feature_cols]
)

latest_per_engine["Predicted RUL"] = np.round(
    latest_preds,
    0
)


# Health status

latest_per_engine["Status"] = np.select(
    [
        latest_per_engine["Predicted RUL"] < 30,
        latest_per_engine["Predicted RUL"] < 60
    ],
    [
        "🔴 Critical",
        "🟡 Warning"
    ],
    default="🟢 Healthy"
)


# Fleet dataframe

fleet_df = (
    latest_per_engine[
        [
            "unit",
            "Predicted RUL",
            "Status"
        ]
    ]
    .rename(
        columns={
            "unit": "Engine"
        }
    )
    .sort_values(
        "Predicted RUL"
    )
    .reset_index(drop=True)
)


# Fleet table

st.subheader("Current Fleet Health")

st.dataframe(
    fleet_df,
    width="stretch",
    hide_index=True
)


# ================================================================
# FLEET RISK DISTRIBUTION
# ================================================================

st.subheader("Fleet Risk Distribution")


status_order = [
    "🟢 Healthy",
    "🟡 Warning",
    "🔴 Critical"
]

status_counts = (
    fleet_df["Status"]
    .value_counts()
    .reindex(
        status_order,
        fill_value=0
    )
)


fig_fleet = go.Figure(
    data=[
        go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            marker=dict(
                colors=[
                    "#2ecc71",
                    "#f39c12",
                    "#e74c3c"
                ]
            ),
            textinfo="label+percent",
            hole=0.35
        )
    ]
)

fig_fleet.update_layout(
    title="Fleet Risk Distribution",
    height=420
)

st.plotly_chart(
    fig_fleet,
    width="stretch"
)


# ================================================================
# SINGLE ENGINE DETAIL VIEW
# ================================================================

st.divider()

st.header(
    f"Engine {selected_engine} — Detail View"
)

st.caption(
    "Detailed RUL prediction and health trajectory "
    "for the selected engine."
)


# Selected engine data

engine_data = (
    val_data[
        val_data["unit"] == selected_engine
    ]
    .sort_values("cycle")
    .copy()
)


# Predictions

engine_data["predicted_RUL"] = model.predict(
    engine_data[feature_cols]
)


latest_pred = (
    engine_data["predicted_RUL"]
    .iloc[-1]
)


# ================================================================
# RUL OVER TIME
# ================================================================

col1, col2 = st.columns(
    [2, 1]
)


with col1:

    fig = go.Figure()


    # Actual RUL

    fig.add_trace(
        go.Scatter(
            x=engine_data["cycle_original"],
            y=engine_data["RUL"],
            name="Actual RUL",
            line=dict(
                color="slategrey"
            )
        )
    )


    # Predicted RUL

    fig.add_trace(
        go.Scatter(
            x=engine_data["cycle_original"],
            y=engine_data["predicted_RUL"],
            name="Predicted RUL",
            line=dict(
                color="steelblue"
            )
        )
    )


    fig.update_layout(
        title="RUL Over Time",
        xaxis_title="Cycle",
        yaxis_title="RUL",
        height=450
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


# ================================================================
# CURRENT ENGINE STATUS
# ================================================================

with col2:

    st.metric(
        "Current Predicted RUL",
        f"{latest_pred:.0f} cycles"
    )


    if latest_pred < 30:

        st.error(
            "🔴 Critical — maintenance recommended soon"
        )

    elif latest_pred < 60:

        st.warning(
            "🟡 Warning — monitor closely"
        )

    else:

        st.success(
            "🟢 Healthy"
        )


# ================================================================
# 3. WHY THIS PREDICTION?
# ================================================================

st.subheader(
    "Why this prediction?"
)

st.caption(
    "SHAP-based explanation of the features contributing "
    "most strongly to the current RUL prediction."
)


latest_row = (
    engine_data[
        feature_cols
    ]
    .iloc[[-1]]
)


shap_values = explainer.shap_values(
    latest_row
)


# Handle SHAP output

if isinstance(shap_values, list):

    shap_values_current = shap_values[0]

else:

    shap_values_current = shap_values


shap_df = pd.DataFrame(
    {
        "feature": feature_cols,
        "shap_value": shap_values_current[0]
    }
)


shap_df = (
    shap_df
    .sort_values(
        "shap_value",
        key=abs,
        ascending=False
    )
    .head(8)
)


st.bar_chart(
    shap_df.set_index(
        "feature"
    )["shap_value"]
)


# ================================================================
# LIVE MONITORING SIMULATION
# ================================================================

st.divider()

st.header(
    "Live Monitoring Simulation"
)

st.caption(
    "Simulate the selected engine's RUL predictions "
    "as new operating cycles arrive."
)


if st.button(
    "▶ Start Simulation"
):

    demo_engine = selected_engine


    demo_data = (
        val_data[
            val_data["unit"] == demo_engine
        ]
        .sort_values("cycle")
        .copy()
    )


    chart_placeholder = st.empty()


    status_placeholder = st.empty()


    # Step through in chunks of 5 cycles,
    # while always landing on the final cycle.

    steps = list(
        range(
            10,
            len(demo_data) + 1,
            5
        )
    )


    if (
        not steps
        or steps[-1] != len(demo_data)
    ):

        steps.append(
            len(demo_data)
        )


    for i in steps:

        partial_data = (
            demo_data
            .iloc[:i]
            .copy()
        )


        partial_data[
            "predicted_RUL"
        ] = model.predict(
            partial_data[feature_cols]
        )


        # Live chart

        fig = go.Figure()


        fig.add_trace(
            go.Scatter(
                x=partial_data[
                    "cycle_original"
                ],
                y=partial_data[
                    "predicted_RUL"
                ],
                name="Predicted RUL",
                line=dict(
                    color="steelblue"
                )
            )
        )


        fig.update_layout(
            title=(
                f"Live Feed — Engine "
                f"{demo_engine}"
            ),
            xaxis_title="Cycle",
            yaxis_title="Predicted RUL",
            height=450
        )


        chart_placeholder.plotly_chart(
            fig,
            width="stretch"
        )


        # Latest prediction

        latest = float(
            partial_data[
                "predicted_RUL"
            ].iloc[-1]
        )


        current_cycle = (
            partial_data[
                "cycle_original"
            ].iloc[-1]
        )


        # Status

        if latest < 30:

            status_placeholder.error(
                f"🔴 Engine {demo_engine} | "
                f"Cycle {current_cycle}: "
                f"Critical — RUL {latest:.0f}"
            )

        elif latest < 60:

            status_placeholder.warning(
                f"🟡 Engine {demo_engine} | "
                f"Cycle {current_cycle}: "
                f"Warning — RUL {latest:.0f}"
            )

        else:

            status_placeholder.success(
                f"🟢 Engine {demo_engine} | "
                f"Cycle {current_cycle}: "
                f"Healthy — RUL {latest:.0f}"
            )


        time.sleep(0.3)


# ================================================================
# 5. MODEL ROBUSTNESS & DRIFT SIMULATOR
# ================================================================

st.divider()

st.header(
    "Model Robustness & Drift Simulator"
)

st.caption(
    "Simulate fleet-wide sensor drift and observe its impact "
    "on model performance in real time."
)


# Drift slider

drift_pct = st.slider(
    "Simulated Sensor Drift",
    0,
    100,
    0,
    help=(
        "0% = current validation data, "
        "100% = heavily shifted operating conditions"
    )
)


# Reference and validation data

reference_data = train_data[feature_cols]

base_data = val_data[feature_cols].copy()


# Simulated drift

shift_amount = (
    drift_pct / 100
) * 2.5

drifted_data = (
    base_data
    + shift_amount * reference_data.std()
)


# Standardized feature shift

drift_scores = (
    (
        drifted_data.mean()
        - reference_data.mean()
    )
    / reference_data.std()
).abs().sort_values(
    ascending=False
)


# Overall fleet-wide drift score

overall_drift = drift_scores.mean()


# ================================================================
# DRIFT METRICS + TOP FEATURES
# ================================================================

col1, col2 = st.columns(
    [1, 2]
)


# ------------------------------------------------
# Drift Metrics
# ------------------------------------------------

with col1:

    if overall_drift < 0.5:

        status = "🟢 Stable"

    elif overall_drift < 1.5:

        status = "🟡 Moderate"

    else:

        status = "🔴 Severe"


    st.metric(
        "Overall Drift Score",
        f"{overall_drift:.2f}",
        status
    )


    # Baseline RMSE

    baseline_predictions = model.predict(
        base_data
    )


    baseline_rmse = np.sqrt(
        np.mean(
            (
                baseline_predictions
                - val_data["RUL"].values
            ) ** 2
        )
    )


    # Drifted RMSE

    drifted_predictions = model.predict(
        drifted_data
    )


    drifted_rmse = np.sqrt(
        np.mean(
            (
                drifted_predictions
                - val_data["RUL"].values
            ) ** 2
        )
    )


    st.metric(
        "Baseline RMSE",
        f"{baseline_rmse:.2f}"
    )


    st.metric(
        "Model RMSE Under Drift",
        f"{drifted_rmse:.2f}",
        delta=(
            f"{drifted_rmse - baseline_rmse:+.2f} "
            "vs. baseline"
        ),
        delta_color="inverse"
    )


# ------------------------------------------------
# Top 10 Drifted Features
# ------------------------------------------------

with col2:

    top_drifted_features = (
        drift_scores
        .head(10)
        .sort_values()
    )


    fig_drift = go.Figure(
        go.Bar(
            x=top_drifted_features.values,
            y=top_drifted_features.index,
            orientation="h"
        )
    )


    fig_drift.update_layout(
        title="Top 10 Drifted Features",
        xaxis_title="Standardized Shift",
        yaxis_title="Feature",
        height=420
    )


    st.plotly_chart(
        fig_drift,
        width="stretch"
    )


# ================================================================
# DETAILED DRIFT REPORTS
# ================================================================

st.divider()

st.header(
    "Detailed Drift Reports"
)

st.caption(
    "Detailed statistical analysis of data distributions "
    "and feature drift, generated with Evidently AI."
)


tab1, tab2 = st.tabs(
    [
        "Validation vs. Training",
        "Simulated Drift Scenario"
    ]
)


# ================================================================
# VALIDATION VS TRAINING REPORT
# ================================================================

with tab1:

    drift_report_path = (
        BASE_DIR
        / "app"
        / "drift_report.html"
    )


    if drift_report_path.exists():

        with open(
            drift_report_path,
            "r",
            encoding="utf-8"
        ) as f:

            st.components.v1.html(
                f.read(),
                height=800,
                scrolling=True
            )

    else:

        st.warning(
            "drift_report.html was not found."
        )


# ================================================================
# SIMULATED DRIFT REPORT
# ================================================================

with tab2:

    simulated_report_path = (
        BASE_DIR
        / "app"
        / "simulated_drift_report.html"
    )


    if simulated_report_path.exists():

        with open(
            simulated_report_path,
            "r",
            encoding="utf-8"
        ) as f:

            st.components.v1.html(
                f.read(),
                height=800,
                scrolling=True
            )

    else:

        st.warning(
            "simulated_drift_report.html was not found."
        )