"""Evaluation metrics and error analysis."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.loader import TARGET, COUNTRIES

def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error, mask zeros. Return as percentage."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_metrics(y_true, y_pred) -> dict:
    """Return dict with MAE, RMSE, MAPE, R2."""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mape(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

def per_country_metrics(results_df) -> pd.DataFrame:
    """Compute MAE, RMSE, MAPE per country + overall. Return formatted DataFrame."""
    metrics = []
    
    # Per country
    for country in results_df['Country'].unique():
        country_df = results_df[results_df['Country'] == country]
        m = compute_metrics(country_df['Actual'], country_df['Predicted'])
        m['Country'] = country
        metrics.append(m)
        
    # Overall
    overall_m = compute_metrics(results_df['Actual'], results_df['Predicted'])
    overall_m['Country'] = 'Overall'
    metrics.append(overall_m)
    
    df = pd.DataFrame(metrics)
    return df[['Country', 'MAE', 'RMSE', 'MAPE', 'R2']]

def error_analysis(results_df) -> dict:
    """Identify worst predictions, compute error by year, error by country."""
    df = results_df.copy()
    df['Absolute Error'] = np.abs(df['Actual'] - df['Predicted'])
    
    # Worst predictions (top 5)
    worst = df.nlargest(5, 'Absolute Error').to_dict(orient='records')
    
    # Error by year
    if 'Year' in df.columns:
        error_by_year = df.groupby('Year')['Absolute Error'].mean().to_dict()
    else:
        error_by_year = {}
        
    # Error by country
    error_by_country = df.groupby('Country')['Absolute Error'].mean().to_dict()
    
    return {
        'worst_predictions': worst,
        'error_by_year': error_by_year,
        'error_by_country': error_by_country
    }
