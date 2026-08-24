"""Sensitivity analysis: tornado charts and elasticity curves."""
import numpy as np
import pandas as pd

from src.data.loader import TARGET, COUNTRIES
from src.features.engineer import engineer_features
from src.models.forecaster import recursive_forecast

SENS_FEATURES = [
    'Births (thousands)',
    'Crude Birth Rate (births per 1,000 population)',
    'Infant Mortality Rate (infant deaths per 1,000 live births)',
    'Infant Deaths, under age 1 (thousands)',
    'Under-Five Deaths, under age 5 (thousands)',
    'Net Migration Rate (per 1,000 population)',
    'Pop_Age_0(In Thousands)',
]

FEATURE_LABELS = {
    'Births (thousands)': 'Births',
    'Crude Birth Rate (births per 1,000 population)': 'Birth Rate',
    'Infant Mortality Rate (infant deaths per 1,000 live births)': 'Infant Mortality Rate',
    'Infant Deaths, under age 1 (thousands)': 'Infant Deaths',
    'Under-Five Deaths, under age 5 (thousands)': 'Under-5 Deaths',
    'Net Migration Rate (per 1,000 population)': 'Migration Rate',
    'Pop_Age_0(In Thousands)': 'Population Age 0',
}

def run_tornado_analysis(df_raw, model, feature_cols, future_demo_df, pct=5) -> dict:
    """For each country and each SENS_FEATURE: perturb +pct% and -pct% in future_demo_df, run recursive_forecast, compute average % change vs baseline forecast."""
    results = {}
    
    for country in COUNTRIES:
        country_df = df_raw[df_raw['Country'] == country].copy()
        country_future = future_demo_df[future_demo_df['Country'] == country].copy()
        
        # Baseline
        base_forecast = recursive_forecast(country_df, model, feature_cols, split_year=2025, future_demo_df=country_future)
        base_avg = np.mean(base_forecast['Predicted'])
        
        country_results = []
        for feature in SENS_FEATURES:
            if feature not in country_future.columns:
                continue
                
            # Positive perturbation
            fut_pos = country_future.copy()
            fut_pos[feature] = fut_pos[feature] * (1 + pct/100.0)
            pos_forecast = recursive_forecast(country_df, model, feature_cols, split_year=2025, future_demo_df=fut_pos)
            pos_avg = np.mean(pos_forecast['Predicted'])
            pos_impact = ((pos_avg - base_avg) / base_avg) * 100 if base_avg != 0 else 0
            
            # Negative perturbation
            fut_neg = country_future.copy()
            fut_neg[feature] = fut_neg[feature] * (1 - pct/100.0)
            neg_forecast = recursive_forecast(country_df, model, feature_cols, split_year=2025, future_demo_df=fut_neg)
            neg_avg = np.mean(neg_forecast['Predicted'])
            neg_impact = ((neg_avg - base_avg) / base_avg) * 100 if base_avg != 0 else 0
            
            country_results.append({
                'feature': feature,
                'label': FEATURE_LABELS.get(feature, feature),
                'positive_impact': pos_impact,
                'negative_impact': neg_impact,
                'abs_impact': max(abs(pos_impact), abs(neg_impact))
            })
            
        country_results.sort(key=lambda x: x['abs_impact'], reverse=True)
        results[country] = country_results
        
    return results

def classify_feature_importance(tornado_results) -> dict:
    """Classify each feature as HIGH (abs > 0.5), MEDIUM (abs > 0.2), LOW."""
    classified = {}
    for country, results in tornado_results.items():
        country_class = []
        for res in results:
            score = res['abs_impact']
            if score > 0.5:
                level = 'HIGH'
            elif score > 0.2:
                level = 'MEDIUM'
            else:
                level = 'LOW'
            country_class.append({
                'name': res['label'],
                'impact_level': level,
                'score': score
            })
        classified[country] = country_class
    return classified

def compute_elasticity(df_raw, model, feature_cols, future_demo_df, target_year=2030) -> dict:
    """For top 3 features per country (from tornado), sweep from -20% to +20% in 21 steps."""
    tornado = run_tornado_analysis(df_raw, model, feature_cols, future_demo_df, pct=5)
    
    elasticity_results = {}
    sweep_pcts = np.linspace(-20, 20, 21)
    
    for country in COUNTRIES:
        elasticity_results[country] = {}
        top3_features = [res['feature'] for res in tornado[country][:3]]
        
        country_df = df_raw[df_raw['Country'] == country].copy()
        country_future = future_demo_df[future_demo_df['Country'] == country].copy()
        
        # Baseline
        base_forecast = recursive_forecast(country_df, model, feature_cols, split_year=2025, future_demo_df=country_future)
        base_avg = np.mean(base_forecast['Predicted'])
        
        for feature in top3_features:
            curve = []
            for pct in sweep_pcts:
                fut_pert = country_future.copy()
                fut_pert[feature] = fut_pert[feature] * (1 + pct/100.0)
                pert_forecast = recursive_forecast(country_df, model, feature_cols, split_year=2025, future_demo_df=fut_pert)
                pert_avg = np.mean(pert_forecast['Predicted'])
                impact = ((pert_avg - base_avg) / base_avg) * 100 if base_avg != 0 else 0
                
                curve.append({'x': pct, 'y': impact})
                
            label = FEATURE_LABELS.get(feature, feature)
            elasticity_results[country][label] = curve
            
    return elasticity_results
