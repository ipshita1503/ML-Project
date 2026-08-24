"""Reusable chart components for the dashboard."""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from app.utils.style import COLORS, CHART_COLORS

def _apply_clean_layout(fig):
    """Apply a sparse, professional layout to any Plotly figure."""
    fig.update_layout(
        margin=dict(t=40, l=40, r=20, b=40),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title=None,
            font=dict(size=11)
        )
    )
    fig.update_xaxes(
        showgrid=False,
        showline=True
    )
    fig.update_yaxes(
        showgrid=True,
        showline=False,
        zeroline=True
    )
    return fig

def trajectory_chart(historical, forecast, mc_data, country, title=None):
    """Clean line chart showing history + forecast + uncertainty."""
    fig = go.Figure()
    
    # Historical
    if historical and country in historical:
        hist_df = pd.DataFrame(historical[country])
        fig.add_trace(go.Scatter(
            x=hist_df['Year'], 
            y=hist_df['Actual'],
            mode='lines',
            name='Historical',
            line=dict(color=COLORS['primary'], width=2)
        ))
        
    # Uncertainty Bands
    if mc_data and country in mc_data:
        mc_df = pd.DataFrame(mc_data[country])
        # P5 to P95
        fig.add_trace(go.Scatter(
            x=list(mc_df['year']) + list(mc_df['year'])[::-1],
            y=list(mc_df['p95']) + list(mc_df['p5'])[::-1],
            fill='toself',
            fillcolor='rgba(13, 148, 136, 0.1)', # Teal muted
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='90% Interval'
        ))
        # P25 to P75
        fig.add_trace(go.Scatter(
            x=list(mc_df['year']) + list(mc_df['year'])[::-1],
            y=list(mc_df['p75']) + list(mc_df['p25'])[::-1],
            fill='toself',
            fillcolor='rgba(13, 148, 136, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='50% Interval'
        ))

    # Forecast
    if forecast and country in forecast:
        fc_df = pd.DataFrame(forecast[country])
        fig.add_trace(go.Scatter(
            x=fc_df['Year'],
            y=fc_df['Predicted'],
            mode='lines',
            name='Forecast',
            line=dict(color=COLORS['secondary'], width=2, dash='dash')
        ))
        
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14, color=COLORS['text'])))
        
    fig.update_yaxes(title="Doses (thousands)")
    return _apply_clean_layout(fig)

def tornado_chart(tornado_data, country):
    """Bidirectional sensitivity bar chart."""
    fig = go.Figure()
    
    if tornado_data and country in tornado_data:
        df = pd.DataFrame(tornado_data[country])
        df = df.sort_values('abs_impact', ascending=True) # Ascending for horizontal bar
        
        fig.add_trace(go.Bar(
            y=df['label'],
            x=df['positive_impact'],
            name='Increase Driver',
            orientation='h',
            marker_color=COLORS['secondary']
        ))
        fig.add_trace(go.Bar(
            y=df['label'],
            x=df['negative_impact'],
            name='Decrease Driver',
            orientation='h',
            marker_color=COLORS['primary']
        ))
        
    fig.update_layout(barmode='relative', title="Feature Sensitivity Impact (%)")
    fig.update_xaxes(title="Impact on Forecast (%)")
    return _apply_clean_layout(fig)

def feature_importance_chart(importance_data, top_n=10):
    """Horizontal bar chart for feature importance."""
    df = pd.DataFrame(importance_data)
    df = df.head(top_n).sort_values('Abs_Importance', ascending=True)
    
    fig = px.bar(
        df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        color='Importance',
        color_continuous_scale=[COLORS['primary'], COLORS['surface'], COLORS['secondary']]
    )
    
    fig.update_layout(coloraxis_showscale=False, title="Model Feature Drivers")
    return _apply_clean_layout(fig)

def comparison_bar_chart(df, category_col, value_col, title):
    """Clean vertical bar chart for comparison."""
    fig = px.bar(df, x=category_col, y=value_col)
    fig.update_traces(marker_color=COLORS['primary'])
    fig.update_layout(title=dict(text=title, font=dict(size=14)))
    return _apply_clean_layout(fig)
