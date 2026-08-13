import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import shap
import time

st.set_page_config(page_title="Turbofan RUL Dashboard", layout="wide")
st.title("Turbofan Engine Health Monitor")
st.caption("Real-time RUL prediction, explainability, and drift monitoring for a fleet of turbofan engines.")

# ---- Cached loading: runs once, not on every click/slider drag ----
@st.cache_resource
def load_model():
    return joblib.load("../models/xgb_fd001.pkl")

@st.cache_data
def load_data():
    val = pd.read_csv("../data/processed/val_FD001.csv")
    train = pd.read_csv("../data/processed/train_FD001.csv")
    return val, train

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

model = load_model()
val_data, train_data = load_data()
explainer = get_explainer(model)

feature_cols = [c for c in val_data.columns if c not in ['unit', 'RUL', 'dataset', 'op_condition', 'cycle_original']]
engine_ids = sorted(val_data['unit'].unique())

selected_engine = st.sidebar.selectbox("Select Engine", engine_ids)

# ================================================================
# 1. FLEET OVERVIEW — leads with the "this is a product" view
# ================================================================
st.divider()
st.header("Fleet Overview")

# One batched prediction for every engine's latest cycle, instead of
# calling model.predict() separately inside a loop for each engine.
latest_per_engine = val_data.sort_values('cycle').groupby('unit').tail(1).copy()
latest_preds = model.predict(latest_per_engine[feature_cols])
latest_per_engine['Predicted RUL'] = np.round(latest_preds, 0)
latest_per_engine['Status'] = np.select(
    [latest_per_engine['Predicted RUL'] < 30, latest_per_engine['Predicted RUL'] < 60],
    ['🔴 Critical', '🟡 Warning'],
    default='🟢 Healthy'
)

fleet_df = (
    latest_per_engine[['unit', 'Predicted RUL', 'Status']]
    .rename(columns={'unit': 'Engine'})
    .sort_values('Predicted RUL')
    .reset_index(drop=True)
)
st.dataframe(fleet_df, use_container_width=True, hide_index=True)

# Fixed order so colors always map to the same status, regardless of
# which status happens to be most common (value_counts() sorts by
# frequency, not by a fixed Healthy/Warning/Critical order).
status_order = ["🟢 Healthy", "🟡 Warning", "🔴 Critical"]
status_counts = fleet_df["Status"].value_counts().reindex(status_order, fill_value=0)

fig_fleet = go.Figure(data=[go.Pie(
    labels=status_counts.index,
    values=status_counts.values,
    marker=dict(colors=["#2ecc71", "#f39c12", "#e74c3c"]),
    textinfo="label+percent",
    hole=0.35
)])
fig_fleet.update_layout(title="Fleet Risk Distribution")
st.plotly_chart(fig_fleet, use_container_width=True)



# ================================================================
# 3. SINGLE ENGINE DRILL-DOWN — detail view for the picked engine
# ================================================================
st.divider()
st.header(f"Engine {selected_engine} — Detail View")

engine_data = val_data[val_data['unit'] == selected_engine].sort_values('cycle').copy()
engine_data['predicted_RUL'] = model.predict(engine_data[feature_cols])
latest_pred = engine_data['predicted_RUL'].iloc[-1]

col1, col2 = st.columns([2, 1])

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=engine_data['cycle_original'], y=engine_data['RUL'],
                              name='Actual RUL', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=engine_data['cycle_original'], y=engine_data['predicted_RUL'],
                              name='Predicted RUL', line=dict(color='steelblue')))
    fig.update_layout(title='RUL Over Time', xaxis_title='Cycle', yaxis_title='RUL')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric("Current Predicted RUL", f"{latest_pred:.0f} cycles")
    if latest_pred < 30:
        st.error("🔴 Critical — maintenance recommended soon")
    elif latest_pred < 60:
        st.warning("🟡 Warning — monitor closely")
    else:
        st.success("🟢 Healthy")

st.subheader("Why this prediction?")
latest_row = engine_data[feature_cols].iloc[[-1]]
shap_values = explainer.shap_values(latest_row)

shap_df = pd.DataFrame({
    'feature': feature_cols,
    'shap_value': shap_values[0]
}).sort_values('shap_value', key=abs, ascending=False).head(8)

st.bar_chart(shap_df.set_index('feature')['shap_value'])

# ================================================================
# 4. LIVE MONITORING SIMULATION 
# ================================================================
st.divider()
st.header("Live Monitoring Simulation")

if st.button("▶  Start Simulation"):
    demo_engine = selected_engine  # follows the sidebar, not always Engine 1
    demo_data = val_data[val_data['unit'] == demo_engine].sort_values('cycle').copy()

    chart_placeholder = st.empty()
    status_placeholder = st.empty()

    # Step through in chunks of 5 cycles, but always land on the true final cycle
    steps = list(range(10, len(demo_data) + 1, 5))
    if not steps or steps[-1] != len(demo_data):
        steps.append(len(demo_data))

    for i in steps:
        partial_data = demo_data.iloc[:i].copy()
        partial_data['predicted_RUL'] = model.predict(partial_data[feature_cols])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=partial_data['cycle_original'], y=partial_data['predicted_RUL'],
                                  name='Predicted RUL', line=dict(color='steelblue')))
        fig.update_layout(title=f'Live Feed — Engine {demo_engine}',
                           xaxis_title='Cycle', yaxis_title='Predicted RUL')
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        latest = float(partial_data['predicted_RUL'].iloc[-1])
        current_cycle = partial_data['cycle_original'].iloc[-1]

        if latest < 30:
            status_placeholder.error(f"🔴 Engine {demo_engine} | Cycle {current_cycle}: Critical — RUL {latest:.0f}")
        elif latest < 60:
            status_placeholder.warning(f"🟡 Engine {demo_engine} | Cycle {current_cycle}: Warning — RUL {latest:.0f}")
        else:
            status_placeholder.success(f"🟢 Engine {demo_engine} | Cycle {current_cycle}: Healthy — RUL {latest:.0f}")

        time.sleep(0.3)
        
# ================================================================
# 2. LIVE DRIFT & MODEL DEGRADATION SIMULATOR 
# ================================================================
st.divider()
st.header("Model Robustness & Drift Simulator")
st.caption("Simulate fleet-wide sensor drift and observe its impact on model performance in real time")

drift_pct = st.slider("Simulated Sensor Drift", 0, 100, 0,
                       help="0% = current validation data, 100% = heavily shifted operating conditions")

reference_data = train_data[feature_cols]
base_data = val_data[feature_cols].copy()
shift_amount = (drift_pct / 100) * 2.5
drifted_data = base_data + shift_amount * reference_data.std()

drift_scores = ((drifted_data.mean() - reference_data.mean()) / reference_data.std()).abs().sort_values(ascending=False)
overall_drift = drift_scores.mean()

col1, col2 = st.columns([1, 2])
with col1:
    status = "🟢 Stable" if overall_drift < 0.5 else "🟡 Moderate" if overall_drift < 1.5 else "🔴 Severe"
    st.metric("Overall Drift Score", f"{overall_drift:.2f}", status)

    baseline_rmse = np.sqrt(np.mean((model.predict(base_data) - val_data['RUL'].values) ** 2))
    drifted_rmse = np.sqrt(np.mean((model.predict(drifted_data) - val_data['RUL'].values) ** 2))
    st.metric("Model RMSE Under Drift", f"{drifted_rmse:.2f}",
              delta=f"{drifted_rmse - baseline_rmse:+.2f} vs. baseline", delta_color="inverse")

with col2:
    fig = go.Figure(go.Bar(x=drift_scores.head(10).values, y=drift_scores.head(10).index, orientation='h'))
    fig.update_layout(title="Top 10 Drifted Features", xaxis_title="Standardized Shift", height=350)
    st.plotly_chart(fig, use_container_width=True)        

# ================================================================
# 5. DETAILED DRIFT REPORTS 
# ================================================================
st.divider()
st.header("Detailed Drift Reports")
st.caption("Detailed statistical analysis of data distributions and feature drift, generated with Evidently AI")

tab1, tab2 = st.tabs(["Validation vs. Training", "Simulated Drift Scenario"])

with tab1:
    with open("drift_report.html", "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=800, scrolling=True)

with tab2:
    with open("simulated_drift_report.html", "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=800, scrolling=True)