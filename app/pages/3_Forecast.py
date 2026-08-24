import streamlit as st
import pandas as pd
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.components.metrics import format_number, format_currency

st.markdown("<h2>Detailed Forecast & Budget</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:var(--giip-text-muted); font-size:0.9rem; margin-bottom:2rem;'>Year-by-year breakdown of predicted demand, required procurement, and estimated costs.</p>", unsafe_allow_html=True)

artifacts = st.session_state.get('artifacts', {})
country = st.session_state.get('selected_country')
cost_config = artifacts.get('cost_config', {'price_per_dose': 0.318, 'baked_wastage': 0.25})

if not country or country not in artifacts.get('forecast', {}):
    st.warning("Forecast data not available.")
    st.stop()

# Settings
st.markdown("### Procurement Assumptions")
col1, col2 = st.columns(2)
with col1:
    wastage = st.slider("Wastage Allowance (%)", min_value=0, max_value=50, value=int(cost_config['baked_wastage']*100)) / 100.0
with col2:
    price = st.number_input("Cost per dose (USD)", min_value=0.01, max_value=10.0, value=cost_config['price_per_dose'])

st.divider()

# Table
fc_df = pd.DataFrame(artifacts['forecast'][country])
mc_df = pd.DataFrame(artifacts.get('monte_carlo', {}).get(country, []))

rows = []
for _, row in fc_df.iterrows():
    year = int(row['Year'])
    demand = row['Predicted']
    children = demand * 0.75
    procurement = children / (1 - wastage)
    cost = procurement * 1000 * price
    
    # Get P95 for contingency
    p95 = demand
    if not mc_df.empty:
        mc_row = mc_df[mc_df['year'] == year]
        if not mc_row.empty:
            p95 = mc_row.iloc[0]['p95']
            
    p95_procurement = (p95 * 0.75) / (1 - wastage)
    contingency = (p95_procurement - procurement) * 1000 * price
    
    rows.append({
        'Year': year,
        'Base Demand (k)': f"{demand:,.1f}",
        'Required Procurement (k)': f"{procurement:,.1f}",
        'Base Budget': format_currency(cost),
        'P95 Contingency Fund': format_currency(contingency),
        'Max Budget Exposure': format_currency(cost + contingency)
    })

table_df = pd.DataFrame(rows)
st.dataframe(table_df, hide_index=True, use_container_width=True)

st.markdown("<p style='font-size:0.8rem; color:var(--giip-text-muted); margin-top:1rem;'>* P95 Contingency represents the additional funds required if actual demand falls in the 95th percentile of the Monte Carlo simulation.</p>", unsafe_allow_html=True)

