"""GIIP - Global Immunization Intelligence Platform"""
import streamlit as st
import json
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.utils.style import get_custom_css

st.set_page_config(
    page_title="GIIP Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(get_custom_css(), unsafe_allow_html=True)

@st.cache_data
def load_artifacts():
    artifacts_path = os.path.join(PROJECT_ROOT, 'models', 'artifacts.json')
    if os.path.exists(artifacts_path):
        with open(artifacts_path, 'r') as f:
            return json.load(f)
    return {}

artifacts = load_artifacts()
st.session_state['artifacts'] = artifacts

# Ensure artifacts exist
if not artifacts:
    st.error("Model artifacts not found. Please run 'python train.py' first.")
    st.stop()

# Sidebar context selector
st.sidebar.markdown("<h2 style='font-size:1.2rem; color:var(--giip-primary);'>GIIP</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='color:var(--giip-text-muted); font-size:0.8rem; margin-bottom:1rem;'>Global Immunization Intelligence Platform</div>", unsafe_allow_html=True)

st.session_state['selected_country'] = st.sidebar.selectbox(
    "Context",
    options=artifacts.get('countries', []),
    help="Select the primary country context for the dashboard."
)
st.session_state['selected_vaccine'] = st.sidebar.selectbox(
    "Vaccine Program",
    options=["MCV1 (Measles Dose 1)"],
    disabled=True
)
st.session_state['forecast_horizon'] = st.sidebar.selectbox(
    "Forecast Horizon",
    options=["2025-2030"],
    disabled=True
)

st.sidebar.divider()
st.sidebar.markdown(f"<div style='font-size:0.8rem; color:var(--giip-text-muted);'>Data Scope: {len(artifacts.get('countries', []))} countries, 1980-2030</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"<div style='font-size:0.8rem; color:var(--giip-text-muted);'>Model Engine: Huber Regressor</div>", unsafe_allow_html=True)

# Define the multipage navigation
pages = {
    "Analytics & Planning": [
        st.Page("pages/1_Overview.py", title="Overview"),
        st.Page("pages/2_Explore.py", title="Explore"),
        st.Page("pages/3_Forecast.py", title="Forecast"),
        st.Page("pages/4_Scenarios.py", title="Scenarios"),
    ],
    "Technical Details": [
        st.Page("pages/5_Model.py", title="Model"),
        st.Page("pages/6_Insights.py", title="Insights"),
    ]
}

pg = st.navigation(pages)
pg.run()

