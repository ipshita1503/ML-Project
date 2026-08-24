import streamlit as st
import pandas as pd
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.database import get_analytical_queries, run_query

st.markdown("<h2>Explore Historical Data</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:var(--giip-text-muted); font-size:0.9rem; margin-bottom:2rem;'>Analyze historical demographic trends and execute pre-built SQL intelligence queries against the database.</p>", unsafe_allow_html=True)

db_path = os.path.join(PROJECT_ROOT, 'data', 'giip.db')

tab1, tab2 = st.tabs(["SQL Analytics", "Data Preview"])

with tab1:
    queries = get_analytical_queries()
    query_names = [q['name'] for q in queries]
    
    selected_name = st.selectbox("Select Intelligence Query", options=query_names)
    selected_query = next(q for q in queries if q['name'] == selected_name)
    
    st.markdown(f"<div style='color:var(--giip-text); font-weight:600; margin-top:1rem;'>{selected_query['description']}</div>", unsafe_allow_html=True)
    
    st.code(selected_query['sql'], language='sql')
    
    if os.path.exists(db_path):
        try:
            results = run_query(selected_query['sql'], db_path)
            st.dataframe(results, hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"Error executing query: {e}")
    else:
        st.warning("Database not found. Please run the training pipeline.")

with tab2:
    data_path = os.path.join(PROJECT_ROOT, 'data', 'vaccine_data.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        country = st.session_state.get('selected_country')
        if country:
            st.markdown(f"**Raw Data Preview: {country}**")
            st.dataframe(df[df['Country'] == country].sort_values('Year', ascending=False), hide_index=True)
    else:
        st.warning("Data file not found.")

