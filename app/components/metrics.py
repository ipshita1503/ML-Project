"""Metric display components."""
import streamlit as st

def format_number(value, prefix='', suffix='', decimals=0):
    """Format numbers with commas."""
    if value is None:
        return "-"
    if decimals == 0:
        formatted = f"{value:,.0f}"
    else:
        formatted = f"{value:,.{decimals}f}"
    return f"{prefix}{formatted}{suffix}"

def format_currency(value):
    """Format as USD with commas and M/K suffixes where appropriate."""
    if value is None:
        return "-"
    if value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value/1_000:.1f}k"
    return f"${value:,.0f}"

def format_pct(value, decimals=1, explicit_sign=False):
    """Format as percentage."""
    if value is None:
        return "-"
    sign = "+" if explicit_sign and value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"

def render_decision_card(label, value, subtext):
    """Render a clean decision metric card using custom HTML."""
    html = f"""
    <div class="decision-card">
        <div class="decision-label">{label}</div>
        <div class="decision-value">{value}</div>
        <div class="decision-subtext">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_key_signal(title, value):
    """Render a key signal callout using custom HTML."""
    html = f"""
    <div class="signal-box">
        <div class="signal-title">{title}</div>
        <div class="signal-value">{value}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
