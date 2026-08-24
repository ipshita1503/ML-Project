"""Model explainability: feature importance and SHAP analysis."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import warnings

from src.data.loader import TARGET, COUNTRIES
from src.features.engineer import engineer_features

def get_feature_importance(model, feature_cols):
    """Extract coefficients or feature importances from the model."""
    if hasattr(model, 'steps'):
        estimator = model.steps[-1][1]
    else:
        estimator = model
        
    if hasattr(estimator, 'coef_'):
        importances = estimator.coef_
    elif hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
    else:
        return pd.DataFrame({'Feature': feature_cols, 'Importance': 0, 'Abs_Importance': 0})
        
    df_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances,
        'Abs_Importance': np.abs(importances)
    })
    
    return df_imp.sort_values('Abs_Importance', ascending=False).reset_index(drop=True)

def get_permutation_importance(model, X_test, y_test, feature_cols):
    """Calculate permutation importance."""
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    
    df_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance_Mean': result.importances_mean,
        'Importance_Std': result.importances_std
    })
    
    return df_imp.sort_values('Importance_Mean', ascending=False).reset_index(drop=True)

def compute_shap_values(model, X_train, X_explain, feature_cols):
    """Compute SHAP values, gracefully failing if not supported."""
    try:
        import shap
        
        if hasattr(model, 'steps'):
            estimator = model.steps[-1][1]
            # Transform data through the pipeline steps except the final estimator
            X_transformed = X_train
            X_explain_transformed = X_explain
            for name, step in model.steps[:-1]:
                X_transformed = step.transform(X_transformed)
                X_explain_transformed = step.transform(X_explain_transformed)
        else:
            estimator = model
            X_transformed = X_train
            X_explain_transformed = X_explain
            
        explainer = shap.Explainer(estimator, X_transformed)
        shap_values = explainer(X_explain_transformed)
        
        return {
            'shap_values': shap_values.values,
            'expected_value': shap_values.base_values,
            'feature_names': feature_cols
        }
    except Exception as e:
        warnings.warn(f"SHAP computation failed: {str(e)}")
        return None

def get_country_explanations(model, df_engineered, feature_cols, target_col='MCV1_TARGET'):
    """Get feature importance explanations for each country's latest prediction."""
    explanations = {}
    
    # Get base feature importances to guide the explanation
    feat_imp = get_feature_importance(model, feature_cols)
    imp_dict = dict(zip(feat_imp['Feature'], feat_imp['Importance']))
    
    # Get latest data per country
    latest_data = df_engineered.dropna(subset=feature_cols).sort_values('Year').groupby('Country').last().reset_index()
    
    for _, row in latest_data.iterrows():
        country = row['Country']
        country_exp = []
        
        for feat in feature_cols:
            val = row[feat]
            imp = imp_dict.get(feat, 0)
            
            # Determine direction of effect
            # Positive importance * Positive value (assuming scaled/centered or simple linear interpretation)
            # This is a simplification; for tree models direction is not strictly linear
            direction = "Increases" if imp > 0 else "Decreases" if imp < 0 else "Neutral"
            
            country_exp.append({
                'feature': feat,
                'value': val,
                'importance': imp,
                'direction': direction,
                'abs_importance': abs(imp)
            })
            
        # Sort by absolute importance
        country_exp.sort(key=lambda x: x['abs_importance'], reverse=True)
        # Clean up internal key
        for exp in country_exp:
            del exp['abs_importance']
            
        explanations[country] = country_exp
        
    return explanations
