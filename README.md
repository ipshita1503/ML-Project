# GIIP - Global Immunization Intelligence Platform

An end-to-end Machine Learning and Analytics platform for forecasting vaccine demand, quantifying procurement risk, and guiding budget allocation across countries and vaccine programs.

## Overview

International organizations like Gavi and UNICEF need accurate vaccine demand forecasts to plan procurement budgets, allocate resources, and prepare for demographic shifts. This project provides a complete ML pipeline and analytical dashboard to forecast MCV1 (Measles) vaccine doses required for Kyrgyzstan, Lesotho, and Uzbekistan using UN demographic data.

### Key Features

1. **Robust ML Forecasting**: Tests 5 algorithms (Linear, Ridge, Huber, Gradient Boosting, Random Forest) with walk-forward time-series validation.
2. **Interactive What-If Analysis**: Instantly recalculate predictions and budgets based on custom demographic shifts.
3. **Uncertainty Quantification**: 500-simulation Monte Carlo engine with Student-t residual modeling to determine budget contingency reserves.
4. **SHAP Explainability**: Understand exactly which demographic features drive each country's vaccine demand.
5. **SQL Analytics Layer**: SQLite database enabling deep multi-country analytical queries.
6. **Professional Streamlit Dashboard**: Clean, analytical interface without unnecessary visual clutter.

## Repository Structure

* `app/`: Streamlit dashboard and UI components
* `data/`: Raw and processed data (CSV, SQLite)
* `models/`: Serialized models and generated artifacts (JSON)
* `notebooks/`: Exploratory Data Analysis and Modeling narrative
* `src/`: Core Python modules (data, features, models, evaluation, analytics)
* `train.py`: Main orchestration script
* `tests/`: Basic pipeline verification tests

## Installation and Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate
   # Linux/Mac: source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training pipeline (generates models and data artifacts):
   ```bash
   python train.py
   ```

4. Launch the dashboard:
   ```bash
   streamlit run app/app.py
   ```

## Analytical Methodology

* **Target Variable**: Annual vaccine doses required (in thousands), baking in a 25% wastage allowance.
* **Features**: 19 demographic features including time-lagged target variables (lag 1, 2, 3), rolling means, year-over-year demographic changes, and country indicators.
* **Validation**: Walk-forward validation prevents temporal data leakage, simulating how the model would perform in real-world sequential years.
* **Final Model**: Huber Regressor, chosen for its robustness to historical outliers (e.g., pandemic disruptions) while maintaining strong linear interpretability and superior MAPE (2.87% on backtest).

## Future Improvements

* Integrate actual procurement cost curves (variable pricing based on volume).
* Add integration with live UN demographic APIs instead of static CSVs.
* Expand to cover other vaccine types (DTP, Polio) for full portfolio modeling.
