import streamlit as st
import pandas as pd
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.components.metrics import render_decision_card, render_key_signal, format_number, format_currency, format_pct
from app.components.charts import trajectory_chart

st.markdown("<h1>Global Immunization Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:var(--giip-text-muted); font-size:1rem; margin-bottom:2rem;'>Forecast vaccine demand, assess uncertainty, and explore procurement scenarios across countries.</p>", unsafe_allow_html=True)

artifacts = st.session_state.get('artifacts', {})
country = st.session_state.get('selected_country')
cost_config = artifacts.get('cost_config', {'price_per_dose': 0.318, 'baked_wastage': 0.25})

# --- DECISION SUMMARY ---
st.markdown("### Decision Summary")

# Compute metrics
forecast_data = artifacts.get('forecast', {})
mc_data = artifacts.get('monte_carlo', {})

demand_2030 = 0
if forecast_data and country in forecast_data:
    df_fc = pd.DataFrame(forecast_data[country])
    demand_2030 = df_fc[df_fc['Year'] == 2030]['Predicted'].values[0]

children_covered = demand_2030 * 0.75
required_procurement = children_covered / (1 - cost_config['baked_wastage'])
estimated_budget = required_procurement * 1000 * cost_config['price_per_dose']

# Compute uncertainty spread for 2030
uncertainty_pct = 0
if mc_data and country in mc_data:
    df_mc = pd.DataFrame(mc_data[country])
    mc_2030 = df_mc[df_mc['year'] == 2030].iloc[0]
    p50 = mc_2030['p50']
    spread = (mc_2030['p95'] - mc_2030['p5']) / 2
    if p50 > 0:
        uncertainty_pct = (spread / p50) * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    render_decision_card("Forecast Demand (2030)", f"{format_number(demand_2030)}k", "Estimated doses required")
with col2:
    render_decision_card("Procurement Target", f"{format_number(required_procurement)}k", "At 25% wastage")
with col3:
    render_decision_card("Estimated Budget", format_currency(estimated_budget), "Based on $0.318 per dose")
with col4:
    render_decision_card("Forecast Uncertainty", f"±{format_pct(uncertainty_pct)}", "90% confidence interval")

st.divider()

# --- KEY SIGNALS ---
st.markdown("### Key Signals")
signal_col1, signal_col2 = st.columns(2)

# Compute demand increase signal
latest_historical = 0
hist_data = artifacts.get('backtest', {})
if hist_data and country in hist_data:
    df_hist = pd.DataFrame(hist_data[country])
    latest_historical = df_hist.iloc[-1]['Actual']

demand_growth = ((demand_2030 - latest_historical) / latest_historical * 100) if latest_historical > 0 else 0
direction = "increase" if demand_growth > 0 else "decrease"
sign = "+" if demand_growth > 0 else ""

with signal_col1:
    render_key_signal(
        f"Demand is projected to {direction}",
        f"{sign}{format_pct(demand_growth)} through 2030"
    )

with signal_col2:
    render_key_signal(
        f"Model validation error (MAPE)",
        f"±{format_pct(artifacts.get('backtest_metrics', [{}])[0].get('MAPE', 0))}"
    )

st.divider()

# --- MAIN FORECAST VISUALIZATION ---
st.markdown("### Projected Demand Trajectory")
st.markdown("<p style='color:var(--giip-text-muted); font-size:0.85rem;'>Historical actuals against 2030 forecast with 50% and 90% confidence intervals.</p>", unsafe_allow_html=True)
fig = trajectory_chart(
    artifacts.get('backtest', {}), 
    artifacts.get('forecast', {}), 
    artifacts.get('monte_carlo', {}), 
    country
)
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.divider()

# --- COUNTRY COMPARISON ---
st.markdown("### Cross-Country Procurement Outlook (2030)")
comparison_rows = []
for c in artifacts.get('countries', []):
    if c in forecast_data:
        c_df = pd.DataFrame(forecast_data[c])
        c_demand = c_df[c_df['Year'] == 2030]['Predicted'].values[0]
        c_req = (c_demand * 0.75) / (1 - cost_config['baked_wastage'])
        c_cost = c_req * 1000 * cost_config['price_per_dose']
        
        comparison_rows.append({
            'Country': c,
            'Projected Demand (thousands)': f"{c_demand:,.0f}",
            'Procurement Requirement': f"{c_req:,.0f}",
            'Estimated Budget (USD)': format_currency(c_cost)
        })

comp_df = pd.DataFrame(comparison_rows)
st.dataframe(comp_df, hide_index=True, use_container_width=True)

