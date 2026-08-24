import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.forecaster import recursive_forecast
from app.components.metrics import format_number, format_currency, format_pct

st.markdown("<h2>Scenario Workspace</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:var(--giip-text-muted); font-size:0.9rem; margin-bottom:2rem;'>Adjust demographic assumptions to see the real-time impact on forecasted vaccine demand and procurement budget.</p>", unsafe_allow_html=True)

artifacts = st.session_state.get('artifacts', {})
country = st.session_state.get('selected_country')
cost_config = artifacts.get('cost_config', {'price_per_dose': 0.318, 'baked_wastage': 0.25})

@st.cache_resource
def load_model_pipeline():
    model_path = os.path.join(PROJECT_ROOT, 'models', 'final_model.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model_pipeline()
if model is None:
    st.error("Model file not found. Please run training pipeline.")
    st.stop()

# Layout: Workspace Controls (Left) | Impact Summary (Right)
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("<div class='scenario-box'>", unsafe_allow_html=True)
    st.markdown("<div class='scenario-header'>Demographic Assumptions</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:0.85rem; color:var(--giip-text-muted);'>Adjust the baseline UN demographic projections for the forecast horizon (2025-2030).</p>", unsafe_allow_html=True)
    
    birth_adj = st.slider("Births (%)", min_value=-30, max_value=30, value=0, step=1)
    imr_adj = st.slider("Infant Mortality Rate (%)", min_value=-30, max_value=30, value=0, step=1)
    pop_adj = st.slider("Population Age 0 (%)", min_value=-30, max_value=30, value=0, step=1)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='scenario-box'>", unsafe_allow_html=True)
    st.markdown("<div class='scenario-header'>Scenario Impact (2030)</div>", unsafe_allow_html=True)
    
    # Calculate Impact
    # Load raw data to feed into forecaster
    raw_data_path = os.path.join(PROJECT_ROOT, 'data', 'vaccine_data.csv')
    fut_data_path = os.path.join(PROJECT_ROOT, 'data', 'future_demographics.csv')
    
    if os.path.exists(raw_data_path) and os.path.exists(fut_data_path):
        df_raw = pd.read_csv(raw_data_path)
        df_fut = pd.read_csv(fut_data_path)
        
        c_raw = df_raw[df_raw['Country'] == country].copy()
        c_fut = df_fut[df_fut['Country'] == country].copy()
        
        feature_cols = artifacts.get('feature_importance', [])
        feature_names = [f['Feature'] for f in feature_cols] if feature_cols else []
        
        if not feature_names:
            st.warning("Feature names missing from artifacts.")
        else:
            # Base forecast
            base_preds = recursive_forecast(c_raw, model, feature_names, split_year=2025, future_demo_df=c_fut)
            base_2030 = base_preds[base_preds['Year'] == 2030]['Predicted'].values[0]
            
            # Scenario forecast
            if 'Births' in c_fut.columns:
                c_fut['Births'] = c_fut['Births'] * (1 + birth_adj/100.0)
            if 'IMR' in c_fut.columns:
                c_fut['IMR'] = c_fut['IMR'] * (1 + imr_adj/100.0)
            if 'Pop_Age_0(In Thousands)' in c_fut.columns:
                c_fut['Pop_Age_0(In Thousands)'] = c_fut['Pop_Age_0(In Thousands)'] * (1 + pop_adj/100.0)
                
            scen_preds = recursive_forecast(c_raw, model, feature_names, split_year=2025, future_demo_df=c_fut)
            scen_2030 = scen_preds[scen_preds['Year'] == 2030]['Predicted'].values[0]
            
            delta_demand = scen_2030 - base_2030
            delta_pct = (delta_demand / base_2030 * 100) if base_2030 != 0 else 0
            
            base_cost = (base_2030 * 0.75 / (1 - cost_config['baked_wastage'])) * 1000 * cost_config['price_per_dose']
            scen_cost = (scen_2030 * 0.75 / (1 - cost_config['baked_wastage'])) * 1000 * cost_config['price_per_dose']
            delta_cost = scen_cost - base_cost
            
            st.markdown(f"**Baseline Demand:** {format_number(base_2030)}k doses")
            st.markdown(f"**Scenario Demand:** {format_number(scen_2030)}k doses")
            
            st.markdown("### Variance")
            st.markdown(f"<span style='font-size:1.5rem; font-weight:600; color:{'#0d9488' if delta_demand > 0 else '#dc2626'};'>{format_pct(delta_pct, explicit_sign=True)}</span>", unsafe_allow_html=True)
            sign_prefix = '+' if delta_demand > 0 else ''
            st.markdown(f"<span style='color:var(--giip-text-muted);'>({format_number(delta_demand, prefix=sign_prefix)}k doses)</span>", unsafe_allow_html=True)
            
            st.markdown("### Budget Impact")
            st.markdown(f"<span style='font-size:1.2rem; font-weight:600;'>{format_currency(delta_cost)}</span>", unsafe_allow_html=True)
            
    st.markdown("</div>", unsafe_allow_html=True)

