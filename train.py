"""
GIIP Training Pipeline
======================
Train the vaccine demand forecasting model, run analytical engines,
and save all artifacts for the dashboard.

Usage:
    python train.py
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import load_vaccine_data, load_future_demographics, validate_data
from src.data.database import create_database
from src.features.engineer import engineer_features
from src.models.trainer import compare_models, train_final_model, save_model
from src.models.forecaster import recursive_forecast, run_monte_carlo, run_scenarios
from src.models.explainer import (
    get_feature_importance, get_permutation_importance,
    compute_shap_values, get_country_explanations,
)
from src.evaluation.metrics import per_country_metrics, error_analysis
from src.evaluation.sensitivity import (
    run_tornado_analysis, classify_feature_importance, compute_elasticity,
)

FORECAST_YEARS = list(range(2025, 2031))
COST_PER_DOSE = 0.318
BAKED_WASTAGE = 0.25


def main():
    print("=" * 60)
    print("GIIP - Global Immunization Intelligence Platform")
    print("Training Pipeline")
    print("=" * 60)

    # --- Step 1: Load and validate data ---
    print("\n[1/8] Loading and validating data...")
    df_raw = load_vaccine_data()
    future_demo = load_future_demographics()
    validation = validate_data(df_raw)
    print(f"  Historical data: {validation['row_count']} rows, "
          f"{len(validation['countries'])} countries, "
          f"years {validation['year_range'][0]}-{validation['year_range'][1]}")

    # --- Step 2: Create SQLite database ---
    print("\n[2/8] Creating SQLite database...")
    db_path = create_database(df_raw, future_demo)
    print(f"  Database created: {db_path}")

    # --- Step 3: Model comparison ---
    print("\n[3/8] Comparing models...")
    comparison_df = compare_models(df_raw, split_year=2020)
    print(comparison_df.to_string(index=False))

    # --- Step 4: Train final model ---
    print("\n[4/8] Training final model on all data < 2025...")
    model, feature_cols, train_metrics = train_final_model(
        df_raw, model_name='Huber Regressor', split_year=2025
    )

    # --- Step 5: Backtest and evaluate ---
    print("\n[5/8] Running walk-forward backtest (2020-2024)...")
    backtest_results = recursive_forecast(
        df_raw, model, feature_cols, split_year=2020
    )
    backtest_results = backtest_results.dropna(subset=['Actual', 'Predicted'])
    backtest_metrics = per_country_metrics(backtest_results)
    print(backtest_metrics.to_string(index=False))

    errors = error_analysis(backtest_results)

    # --- Step 6: Forecast, Monte Carlo, Scenarios ---
    print("\n[6/8] Running forecast, Monte Carlo (500 sims), and scenarios...")
    forecast_results = recursive_forecast(
        df_raw, model, feature_cols, split_year=2025,
        future_demo_df=future_demo,
    )

    mc_data = run_monte_carlo(
        df_raw, model, feature_cols, future_demo, n_sim=500, seed=42
    )

    scenarios_data, scenario_meta = run_scenarios(
        df_raw, model, feature_cols, future_demo
    )

    # --- Step 7: Explainability and sensitivity ---
    print("\n[7/8] Computing explainability and sensitivity...")
    df_engineered, _ = engineer_features(df_raw)
    train_eng = df_engineered[df_engineered['Year'] < 2025].copy()
    X_train = train_eng[feature_cols]
    y_train = train_eng['MCV1_TARGET']

    feature_imp = get_feature_importance(model, feature_cols)

    # Permutation importance on backtest set
    test_eng = df_engineered[df_engineered['Year'] >= 2020].copy()
    X_test = test_eng[feature_cols]
    y_test = test_eng['MCV1_TARGET']
    perm_imp = get_permutation_importance(model, X_test, y_test, feature_cols)

    # SHAP
    shap_results = compute_shap_values(model, X_train, X_test, feature_cols)

    # Country explanations
    country_explanations = get_country_explanations(
        model, df_engineered, feature_cols
    )

    # Sensitivity
    tornado = run_tornado_analysis(df_raw, model, feature_cols, future_demo)
    feat_impact = classify_feature_importance(tornado)
    elasticity = compute_elasticity(df_raw, model, feature_cols, future_demo)

    # --- Step 8: Save all artifacts ---
    print("\n[8/8] Saving model and artifacts...")
    save_model(model, feature_cols, train_metrics)

    # Save comprehensive results for dashboard
    artifacts = {
        'model_comparison': comparison_df.to_dict('records'),
        'backtest': {
            country: backtest_results[backtest_results['Country'] == country][
                ['Year', 'Actual', 'Predicted']
            ].to_dict('records')
            for country in backtest_results['Country'].unique()
        },
        'backtest_metrics': backtest_metrics.to_dict('records'),
        'error_analysis': {
            'worst_predictions': errors['worst_predictions'],
            'error_by_country': errors['error_by_country'],
        },
        'forecast': {
            country: forecast_results[forecast_results['Country'] == country][
                ['Year', 'Predicted']
            ].to_dict('records')
            for country in forecast_results['Country'].unique()
        },
        'monte_carlo': mc_data,
        'scenarios': scenarios_data,
        'scenario_meta': scenario_meta,
        'feature_importance': feature_imp.to_dict('records'),
        'permutation_importance': perm_imp.to_dict('records'),
        'country_explanations': country_explanations,
        'tornado': tornado,
        'feature_impact': feat_impact,
        'elasticity': elasticity,
        'cost_config': {
            'price_per_dose': COST_PER_DOSE,
            'baked_wastage': BAKED_WASTAGE,
            'default_wastage': BAKED_WASTAGE,
        },
        'forecast_years': FORECAST_YEARS,
        'countries': list(backtest_results['Country'].unique()),
    }

    if shap_results is not None:
        # Convert SHAP values to serializable format
        artifacts['shap'] = {
            'expected_value': float(shap_results['expected_value'])
                if np.isscalar(shap_results['expected_value'])
                else float(shap_results['expected_value'][0]),
            'feature_names': shap_results['feature_names'],
            'mean_abs_shap': [
                float(v) for v in
                np.abs(shap_results['shap_values']).mean(axis=0)
            ],
        }

    artifacts_path = os.path.join(PROJECT_ROOT, 'models', 'artifacts.json')
    with open(artifacts_path, 'w', encoding='utf-8') as f:
        json.dump(artifacts, f, indent=2, default=str)

    print(f"\n  Model saved to: models/final_model.joblib")
    print(f"  Metadata saved to: models/model_metadata.json")
    print(f"  Artifacts saved to: models/artifacts.json")

    print("\n" + "=" * 60)
    print("Training complete.")
    print("=" * 60)
    print(f"\nTo launch the dashboard:")
    print(f"  streamlit run app/app.py")


if __name__ == '__main__':
    main()
