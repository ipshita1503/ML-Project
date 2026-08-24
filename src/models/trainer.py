"""Model training, comparison, and selection."""
import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.loader import TARGET, COUNTRIES
from src.features.engineer import engineer_features

def mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error, ignoring zeros in y_true."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def get_models():
    """Return dictionary of models wrapped in pipelines with StandardScaler."""
    return {
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        'Ridge Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ]),
        'Huber Regressor': Pipeline([
            ('scaler', StandardScaler()),
            ('model', HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=2000))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42))
        ])
    }

def compare_models(df_raw, split_year=2020):
    """Compare multiple models using train/test split and cross-validation."""
    df_engineered, feature_cols = engineer_features(df_raw, TARGET)
    
    # Drop rows with NaN in features or target
    df_clean = df_engineered.dropna(subset=feature_cols + [TARGET])
    
    train = df_clean[df_clean['Year'] < split_year]
    test = df_clean[df_clean['Year'] >= split_year]
    
    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_test = test[feature_cols]
    y_test = test[TARGET]
    
    models = get_models()
    results = []
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    print(f"{'Model':<20} | {'MAE':<8} | {'RMSE':<8} | {'MAPE':<8} | {'R2':<8} | {'CV_MAE_Mean':<12} | {'CV_MAE_Std':<10}")
    print("-" * 90)
    
    for name, model in models.items():
        # Cross validation
        cv_scores = -cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
        cv_mae_mean = np.mean(cv_scores)
        cv_mae_std = np.std(cv_scores)
        
        # Train and test
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mae_val = mean_absolute_error(y_test, preds)
        rmse_val = np.sqrt(mean_squared_error(y_test, preds))
        mape_val = mape(y_test, preds)
        r2_val = r2_score(y_test, preds)
        
        results.append({
            'Model': name,
            'MAE': mae_val,
            'RMSE': rmse_val,
            'MAPE': mape_val,
            'R2': r2_val,
            'CV_MAE_Mean': cv_mae_mean,
            'CV_MAE_Std': cv_mae_std
        })
        
        print(f"{name:<20} | {mae_val:<8.2f} | {rmse_val:<8.2f} | {mape_val:<8.2f} | {r2_val:<8.3f} | {cv_mae_mean:<12.2f} | {cv_mae_std:<10.2f}")
        
    return pd.DataFrame(results)

def train_final_model(df_raw, model_name='Huber Regressor', split_year=2025):
    """Train the final model on all data before split_year."""
    df_engineered, feature_cols = engineer_features(df_raw, TARGET)
    df_clean = df_engineered.dropna(subset=feature_cols + [TARGET])
    
    train = df_clean[df_clean['Year'] < split_year]
    X_train = train[feature_cols]
    y_train = train[TARGET]
    
    models = get_models()
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
        
    model = models[model_name]
    model.fit(X_train, y_train)
    
    preds = model.predict(X_train)
    metrics = {
        'Train_MAE': float(mean_absolute_error(y_train, preds)),
        'Train_RMSE': float(np.sqrt(mean_squared_error(y_train, preds))),
        'Train_R2': float(r2_score(y_train, preds)),
        'Train_MAPE': float(mape(y_train, preds))
    }
    
    return model, feature_cols, metrics

def save_model(model, feature_cols, metrics, model_dir=None):
    """Save the model and metadata to disk."""
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
    
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'final_model.joblib')
    joblib.dump(model, model_path)
    
    metadata = {
        'model_name': model.steps[-1][0] if isinstance(model, Pipeline) else type(model).__name__,
        'feature_cols': feature_cols,
        'metrics': metrics,
        'training_date': datetime.now().isoformat(),
        'split_year': metrics.get('split_year', None),
        'n_training_samples': metrics.get('n_training_samples', None)
    }
    
    meta_path = os.path.join(model_dir, 'model_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def load_model(model_dir=None):
    """Load model and metadata from disk."""
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        
    model_path = os.path.join(model_dir, 'final_model.joblib')
    meta_path = os.path.join(model_dir, 'model_metadata.json')
    
    model = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
        
    return model, metadata
