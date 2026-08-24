import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.utils.style import COLORS
from app.components.charts import tornado_chart

st.markdown("<h2>Sensitivity & Risk Analysis</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:var(--giip-text-muted); font-size:0.9rem; margin-bottom:2rem;'>Understand which demographic variables create the highest exposure to forecast variance.</p>", unsafe_allow_html=True)

artifacts = st.session_state.get('artifacts', {})
country = st.session_state.get('selected_country')

col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("### Variance Exposure (Tornado Analysis)")
    st.markdown("<p style='font-size:0.85rem; color:var(--giip-text-muted);'>Shows the percentage impact on 2030 demand when a variable shifts by ±5%.</p>", unsafe_allow_html=True)
    
    tornado_data = artifacts.get('tornado', {})
    if tornado_data and country in tornado_data:
        fig_tornado = tornado_chart(tornado_data, country)
        st.plotly_chart(fig_tornado, use_container_width=True, config={'displayModeBar': False})
    else:
        st.warning("Tornado analysis not available.")

with col2:
    st.markdown("### Strategic Takeaways")
    st.markdown("""
    <div style='background-color:#f8fafc; border-left:3px solid #d97706; padding:1.5rem; margin-top:1rem;'>
        <h4 style='font-size:1rem; margin-bottom:1rem; color:var(--giip-text);'>Demographic Drivers</h4>
        <ul style='font-size:0.9rem; color:var(--giip-text-muted); padding-left:1rem; line-height:1.6;'>
            <li><b>Birth Cohort Size</b> consistently dominates demand variance across all regions.</li>
            <li><b>Infant Mortality</b> exerts a secondary but significant dampening effect on final doses required.</li>
            <li>Migration flows remain largely immaterial for the 0-1 age bracket targeted by MCV1.</li>
        </ul>
        <h4 style='font-size:1rem; margin-top:1.5rem; margin-bottom:1rem; color:var(--giip-text);'>System Limitations</h4>
        <ul style='font-size:0.9rem; color:var(--giip-text-muted); padding-left:1rem; line-height:1.6;'>
            <li>The linear nature of the Huber Regressor may underestimate compounding demographic collapses.</li>
            <li>UN future projections are deterministic; the Monte Carlo layer here injects synthetic variance but cannot foresee true black-swan events.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

