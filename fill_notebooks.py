import json
import os

def create_notebook(cells, filename):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.10"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text + "\n"]}

def code(text):
    # Split text into lines but keep the newline character for valid Jupyter format
    lines = [line + "\n" for line in text.split('\n')]
    if lines:
        lines[-1] = lines[-1].strip('\n') # Remove trailing newline from last line
    return {"cell_type": "code", "metadata": {}, "source": lines, "execution_count": None, "outputs": []}

# --- EDA NOTEBOOK ---
eda_cells = [
    md("# GIIP: Exploratory Data Analysis"),
    md("This notebook performs data quality checks, temporal trend analysis, and correlation mapping for global vaccine demand. As an analyst, validating these distributions is critical before feeding them into our predictive models."),
    code("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Configure professional styling\nplt.style.use('seaborn-v0_8-whitegrid')\nsns.set_palette('muted')"),
    md("## 1. Data Loading & Quality Checks"),
    code("df = pd.read_csv('../data/vaccine_data.csv')\ndisplay(df.head())\nprint(f'\\nDataset shape: {df.shape}')"),
    code("# Check for missing values to ensure data integrity\nmissing = df.isnull().sum()\ndisplay(missing[missing > 0])"),
    md("## 2. Target Variable Analysis (Vaccine Demand)"),
    md("Understanding the distribution of our target variable (`MCV1_TARGET`) helps us identify outliers and determine if normal-distribution assumptions hold true."),
    code("plt.figure(figsize=(10, 5))\nsns.histplot(df['MCV1_TARGET'], bins=20, kde=True, color='#1a365d')\nplt.title('Distribution of MCV1 Vaccine Target (Doses)', fontsize=14, pad=15)\nplt.xlabel('Required Doses (Thousands)')\nplt.ylabel('Frequency')\nplt.tight_layout()\nplt.show()"),
    md("## 3. Time Series Trends by Country"),
    code("plt.figure(figsize=(12, 6))\nsns.lineplot(data=df, x='Year', y='MCV1_TARGET', hue='Country', marker='o', linewidth=2)\nplt.title('Vaccine Demand Trends Over Time (1980-2024)', fontsize=14, pad=15)\nplt.ylabel('MCV1 Target Doses (Thousands)')\nplt.grid(True, alpha=0.3)\nplt.tight_layout()\nplt.show()"),
    md("## 4. Demographic Correlation Analysis"),
    md("We need to understand which demographic factors drive vaccine demand. High correlation with Births is expected, but measuring the exact linear relationship is key for our regression baseline."),
    code("cols = ['MCV1_TARGET', 'Births (thousands)', 'Infant Deaths, under age 1 (thousands)', 'Total Population, as of 1 January (thousands)']\ncorr = df[cols].corr()\n\nplt.figure(figsize=(8, 6))\nsns.heatmap(corr, annot=True, cmap='Blues', vmin=0.5, vmax=1, fmt='.3f', \n            cbar_kws={'label': 'Pearson Correlation'})\nplt.title('Correlation Matrix: Demographics vs Target', pad=15)\nplt.tight_layout()\nplt.show()")
]

# --- MODELING NOTEBOOK ---
mod_cells = [
    md("# GIIP: Predictive Modeling & Pipeline"),
    md("This notebook demonstrates the machine learning pipeline, strictly avoiding temporal data leakage through walk-forward validation and careful feature engineering."),
    code("import pandas as pd\nimport numpy as np\nfrom sklearn.linear_model import HuberRegressor\nfrom sklearn.metrics import mean_absolute_percentage_error\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')"),
    md("## 1. Feature Engineering & Leakage Prevention"),
    md("When forecasting 2025 demand, we cannot use 2025 demographic actuals (as they haven't happened yet). We must strictly `.shift()` our features by 1 year to simulate real-world prediction conditions."),
    code("df = pd.read_csv('../data/vaccine_data.csv')\n\n# Sort chronologically to prevent temporal mixing\ndf = df.sort_values(['Country', 'Year'])\n\n# Create lagged features\nfeatures = ['Births (thousands)', 'Infant Deaths, under age 1 (thousands)']\nfor f in features:\n    df[f'Lag_{f}'] = df.groupby('Country')[f].shift(1)\n\n# Drop NaN rows created by the shift\ndf_clean = df.dropna(subset=[f'Lag_{f}' for f in features]).copy()\ndisplay(df_clean[['Country', 'Year', 'MCV1_TARGET'] + [f'Lag_{f}' for f in features]].head())"),
    md("## 2. Walk-Forward Validation (Time-based Split)"),
    md("Random train/test splits (like K-Fold) are fundamentally flawed for time-series data. We must train on the past (e.g., < 2020) and test on the future (>= 2020) to prove the model actually generalizes to unseen future trends."),
    code("train = df_clean[df_clean['Year'] < 2020]\ntest = df_clean[df_clean['Year'] >= 2020]\n\nX_cols = [f'Lag_{f}' for f in features]\n\nX_train, y_train = train[X_cols], train['MCV1_TARGET']\nX_test, y_test = test[X_cols], test['MCV1_TARGET']\n\nprint(f'Training records: {len(train)}\\nTesting records: {len(test)}')"),
    md("## 3. Model Training & Evaluation"),
    md("We select **Huber Regressor** over standard OLS (Linear Regression) because Huber's loss function is mathematically robust to outliers. It prevents massive one-off anomalies (like pandemic drops) from skewing the regression line."),
    code("model = HuberRegressor(epsilon=1.35)\nmodel.fit(X_train, y_train)\n\npreds = model.predict(X_test)\nmape = mean_absolute_percentage_error(y_test, preds)\nprint(f'Model Test MAPE: {mape:.2%}')"),
    md("## 4. Feature Importance Insights"),
    code("importance = pd.DataFrame({'Feature': ['Lagged Births', 'Lagged Infant Mortality'], 'Coefficient': model.coef_})\nimportance = importance.sort_values('Coefficient', key=abs, ascending=False)\n\nplt.figure(figsize=(8, 4))\nsns.barplot(data=importance, x='Coefficient', y='Feature', color='#0d9488')\nplt.title('Huber Regressor: Feature Drivers', pad=15)\nplt.xlabel('Impact on Forecasted Doses')\nplt.ylabel('')\nplt.tight_layout()\nplt.show()")
]

create_notebook(eda_cells, 'notebooks/01_eda.ipynb')
create_notebook(mod_cells, 'notebooks/02_modeling.ipynb')
print("Notebooks generated successfully.")
