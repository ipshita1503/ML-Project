import streamlit as st
import pandas as pd
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.components.charts import feature_importance_chart, comparison_bar_chart

st.markdown("<h2>Model Diagnostics & Validation</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:var(--giip-text-muted); font-size:0.9rem; margin-bottom:2rem;'>Evaluate forecasting model reliability and examine driving features.</p>", unsafe_allow_html=True)

artifacts = st.session_state.get('artifacts', {})

st.markdown("### Selected Model Architecture")
st.markdown("<p style='font-size:0.9rem; color:var(--giip-text-muted);'>The system utilizes a <b>Huber Regressor</b>, selected for its robust loss function which mitigates the impact of historical outliers (such as pandemic-era disruptions) without losing linear interpretability.</p>", unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Feature Importance")
    feat_data = artifacts.get('feature_importance', [])
    if feat_data:
        fig_feat = feature_importance_chart(feat_data)
        st.plotly_chart(fig_feat, use_container_width=True, config={'displayModeBar': False})
    else:
        st.warning("No feature importance data available.")

with col2:
    st.markdown("### Algorithm Benchmark (Walk-Forward CV)")
    comp_data = artifacts.get('model_comparison', [])
    if comp_data:
        df_comp = pd.DataFrame(comp_data)
        fig_comp = comparison_bar_chart(df_comp, 'Model', 'MAPE', 'Mean Absolute Percentage Error (MAPE)')
        st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': False})
    else:
        st.warning("No model comparison data available.")

st.divider()

st.markdown("### Validation Backtest Metrics (2020-2024)")
backtest_metrics = artifacts.get('backtest_metrics', [])
if backtest_metrics:
    df_metrics = pd.DataFrame(backtest_metrics)
    st.dataframe(df_metrics, hide_index=True, use_container_width=True)

