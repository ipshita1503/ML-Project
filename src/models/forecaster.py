"""Recursive multi-step forecasting and Monte Carlo simulation."""
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation, t as t_dist
from copy import deepcopy

from src.data.loader import TARGET, COUNTRIES
from src.features.engineer import engineer_features
from src.models.trainer import get_models

def recursive_forecast(df_raw, model, feature_cols, split_year, future_demo_df=None):
    """
    Perform recursive forecasting or backtesting.
    If future_demo_df is provided, forecasts into the future using the provided model.
    If future_demo_df is None, performs walk-forward backtesting.
    """
    df_combined = df_raw.copy()
    is_backtest = future_demo_df is None
    
    if not is_backtest:
        future_demo = future_demo_df.copy()
        future_demo[TARGET] = np.nan
        df_combined = pd.concat([df_combined, future_demo], ignore_index=True)
    
    df_combined = df_combined.sort_values(['Country', 'Year']).reset_index(drop=True)
    
    max_year = df_combined['Year'].max()
    years_to_forecast = range(split_year, max_year + 1)
    
    results = []
    
    # For backtesting, cache retrained models per year
    yearly_models = {}
    if is_backtest:
        base_model_name = 'Huber Regressor' # Defaulting or could extract from model
        if hasattr(model, 'steps'):
            base_model_name = type(model.steps[-1][1]).__name__
        models_dict = get_models()
        # Find matching pipeline
        pipeline = None
        for name, pl in models_dict.items():
            if type(pl.steps[-1][1]).__name__ == base_model_name:
                pipeline = pl
                break
        if pipeline is None:
            pipeline = list(models_dict.values())[0]
            
    for year in years_to_forecast:
        # If backtesting, retrain model using data < year
        if is_backtest:
            if year not in yearly_models:
                train_data, _ = engineer_features(df_combined[df_combined['Year'] < year], TARGET)
                train_data = train_data.dropna(subset=feature_cols + [TARGET])
                
                current_model = deepcopy(pipeline)
                current_model.fit(train_data[feature_cols], train_data[TARGET])
                yearly_models[year] = current_model
            
            active_model = yearly_models[year]
        else:
            active_model = model
            
        # Engineer features up to current year
        current_data, _ = engineer_features(df_combined[df_combined['Year'] <= year], TARGET)
        
        for country in COUNTRIES:
            country_mask = (current_data['Country'] == country) & (current_data['Year'] == year)
            if not country_mask.any():
                continue
                
            X_current = current_data.loc[country_mask, feature_cols]
            
            # Predict
            if not X_current.isna().any().any():
                pred = active_model.predict(X_current)[0]
                
                # Store prediction
                actual = df_combined.loc[(df_combined['Country'] == country) & (df_combined['Year'] == year), TARGET].values[0]
                results.append({
                    'Country': country,
                    'Year': year,
                    'Predicted': pred,
                    'Actual': actual
                })
                
                # Update df_combined with prediction for future lags
                if pd.isna(actual) or not is_backtest:
                    df_combined.loc[(df_combined['Country'] == country) & (df_combined['Year'] == year), TARGET] = pred
                    
    return pd.DataFrame(results)

def run_monte_carlo(df_raw, model, feature_cols, future_demo_df, n_sim=500, seed=42):
    """Run Monte Carlo simulations with demographic and residual noise."""
    np.random.seed(seed)
    
    # 1. Get backtest residuals for t-distribution fitting
    # Use split_year=2010 to get residuals
    backtest_results = recursive_forecast(df_raw, model, feature_cols, split_year=2010, future_demo_df=None)
    residuals = (backtest_results['Actual'] - backtest_results['Predicted']).dropna()
    
    if len(residuals) > 0:
        t_params = t_dist.fit(residuals)
    else:
        t_params = (3, 0, 1) # fallback
        
    latest_actuals = df_raw.groupby('Country')[TARGET].last().to_dict()
    
    simulations = {country: [] for country in COUNTRIES}
    base_year = df_raw['Year'].max()
    
    # Run simulations
    for i in range(n_sim):
        # 2. Add demographic noise
        sim_future_demo = future_demo_df.copy()
        
        for year in sim_future_demo['Year'].unique():
            h = 1 + 0.15 * (year - base_year)
            year_mask = sim_future_demo['Year'] == year
            
            # N(1, 0.05*h) for births and Pop_Age_0
            birth_noise = np.random.normal(1, 0.05 * h, size=year_mask.sum())
            if 'Births' in sim_future_demo.columns:
                sim_future_demo.loc[year_mask, 'Births'] *= birth_noise
            if 'Pop_Age_0(In Thousands)' in sim_future_demo.columns:
                sim_future_demo.loc[year_mask, 'Pop_Age_0(In Thousands)'] *= birth_noise
                
            # N(1, 0.10*h) for IMR and U5 mortality
            mortality_noise = np.random.normal(1, 0.10 * h, size=year_mask.sum())
            if 'IMR' in sim_future_demo.columns:
                sim_future_demo.loc[year_mask, 'IMR'] *= mortality_noise
            if 'Under_5_Mortality' in sim_future_demo.columns:
                sim_future_demo.loc[year_mask, 'Under_5_Mortality'] *= mortality_noise
                
        # Run forecast
        forecast = recursive_forecast(df_raw, model, feature_cols, split_year=base_year + 1, future_demo_df=sim_future_demo)
        
        for country in COUNTRIES:
            c_forecast = forecast[forecast['Country'] == country].copy()
            
            # Add residual noise
            noise = t_dist.rvs(*t_params, size=len(c_forecast))
            c_forecast['Sim_Predicted'] = c_forecast['Predicted'] + noise
            
            # Clip predictions
            clip_val = latest_actuals.get(country, 100) * 2
            c_forecast['Sim_Predicted'] = c_forecast['Sim_Predicted'].clip(0, clip_val)
            
            for _, row in c_forecast.iterrows():
                simulations[country].append({
                    'Year': row['Year'],
                    'Predicted': row['Sim_Predicted'],
                    'Sim_ID': i
                })
                
    # Calculate percentiles
    results = {country: [] for country in COUNTRIES}
    for country in COUNTRIES:
        df_sims = pd.DataFrame(simulations[country])
        if not df_sims.empty:
            grouped = df_sims.groupby('Year')['Predicted']
            for year, group in grouped:
                results[country].append({
                    'year': year,
                    'p5': np.percentile(group, 5),
                    'p25': np.percentile(group, 25),
                    'p50': np.percentile(group, 50),
                    'p75': np.percentile(group, 75),
                    'p95': np.percentile(group, 95)
                })
                
    return results

def run_scenarios(df_raw, model, feature_cols, future_demo_df):
    """Run baseline, optimistic, pessimistic, and pandemic scenarios."""
    scenarios = {
        'baseline': lambda df: df,
        'optimistic': lambda df: modify_scenario(df, {'IMR': 0.85, 'Births': 1.02}),
        'pessimistic': lambda df: modify_scenario(df, {'IMR': 1.15, 'Births': 0.95}),
        'pandemic': lambda df: modify_scenario(df, {'Net_Migration': 0.3, 'Births': 0.97})
    }
    
    scenario_meta = {
        'baseline': {'label': 'Baseline', 'color': 'blue', 'description': 'UN projected demographics'},
        'optimistic': {'label': 'Optimistic', 'color': 'green', 'description': '15% lower IMR, 2% higher births'},
        'pessimistic': {'label': 'Pessimistic', 'color': 'orange', 'description': '15% higher IMR, 5% lower births'},
        'pandemic': {'label': 'Pandemic', 'color': 'red', 'description': '70% drop in migration, 3% lower births'}
    }
    
    def modify_scenario(df, multipliers):
        df_mod = df.copy()
        for col, mult in multipliers.items():
            if col in df_mod.columns:
                df_mod[col] *= mult
        return df_mod
        
    base_year = df_raw['Year'].max()
    scenario_data = {country: {} for country in COUNTRIES}
    
    for name, modifier in scenarios.items():
        scenario_demo = modifier(future_demo_df)
        forecast = recursive_forecast(df_raw, model, feature_cols, split_year=base_year + 1, future_demo_df=scenario_demo)
        
        for country in COUNTRIES:
            country_preds = forecast[forecast['Country'] == country]
            scenario_data[country][name] = country_preds[['Year', 'Predicted']].to_dict('records')
            
    return scenario_data, scenario_meta
