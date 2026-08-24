# Global Immunization Intelligence Platform (GIIP)
## Implementation Plan

### Project Objective

Build an original, placement-ready end-to-end ML + Analytics platform for vaccine procurement intelligence.

The two GitHub repositories supplied by the user are reference implementations only. They must be studied, critiqued, and used for ideas—not copied, merged, or presented as the user's original work.

The final product should combine:
- Machine Learning
- Time-series forecasting
- Data analytics
- SQL
- Explainability
- What-if analysis
- Procurement/budget intelligence
- Professional dashboard
- Clean software engineering

The final application should feel like one deliberately designed product.

---

# Phase 0 — Reference Repository Audit

## 0.1 Inspect both repositories

For each repository, document:

- Problem statement
- Dataset and data sources
- Dataset size and granularity
- Target variables
- Preprocessing
- Feature engineering
- ML models
- Validation methodology
- Evaluation metrics
- Forecasting approach
- Uncertainty methodology
- Sensitivity analysis
- Dashboard/UI
- Architecture
- Dependencies
- Strengths
- Weaknesses
- Potential data leakage
- Potential overfitting
- Reusable ideas

## 0.2 Preserve useful ideas

The following ideas from the references are candidates for retention:

- Vaccine demand forecasting
- MCV1/MCV2 procurement targets
- Recursive multi-step forecasting
- Walk-forward validation
- Monte Carlo uncertainty
- Student-t residual simulation
- Sensitivity analysis
- Wastage calculations
- Procurement cost modelling

These must be independently implemented and improved.

## 0.3 Explicitly improve weak areas

The reference implementations should be improved in:

- Model comparison
- Data validation
- Explainability
- SQL analytics
- Cross-country analytics
- Error analysis
- Dashboard interaction
- Software architecture
- Documentation

## 0.4 Originality requirement

Do not copy:

- Source code
- README text
- UI layouts
- Branding
- Exact charts
- Exact architecture
- Exact written explanations

The raw public data source may remain shared where legally appropriate, but all project implementation, analysis, UI, architecture, and documentation should be independently created.

---

# Phase 1 — Final Project Definition

## Project Name

**GIIP — Global Immunization Intelligence Platform**

## One-line description

Forecast vaccine procurement demand across countries using demographic-driven ML, uncertainty quantification, what-if analysis, and actionable budget insights.

## Problem Statement

International organizations and health-program planners need reliable vaccine demand estimates to support procurement budgets and resource allocation.

The system will use demographic and immunization data to forecast future vaccine procurement requirements and help users understand:

- Expected demand
- Uncertainty around forecasts
- Demographic drivers
- Procurement costs
- What happens under different demographic scenarios
- Which countries may require greater attention

## Target Users

- Health program planners
- Procurement analysts
- Data teams
- International health organizations

## Core Story

```text
Public demographic + immunization data
        ↓
Data validation and cleaning
        ↓
Exploratory analysis
        ↓
Feature engineering
        ↓
Time-series ML
        ↓
Model comparison
        ↓
Final model selection
        ↓
2025–2030 forecasting
        ↓
Uncertainty quantification
        ↓
What-if analysis
        ↓
Explainability
        ↓
Procurement and budget insights
        ↓
Actionable recommendations
```

---

# Phase 2 — Data Layer

## 2.1 Data sources

Use the public UN/WHO/UNICEF data already identified by the reference projects where appropriate.

Current intended scope:

- MCV1
- MCV2
- Countries represented in the reference datasets
- Historical demographic data
- Future demographic projections for 2025–2030

The final README must clearly identify the actual data sources and licensing/usage information.

## 2.2 Data validation

Create:

`src/data/loader.py`

Responsibilities:

- Load source files
- Validate required columns
- Validate data types
- Check missing values
- Check duplicate records
- Check ranges
- Check chronological ordering
- Produce a data-quality summary

Do not silently repair suspicious data without recording what was changed.

## 2.3 Database layer

Create:

`src/data/database.py`

Use SQLite unless the final architecture has a strong reason to use another database.

Suggested tables:

- `demographics`
- `vaccine_targets`
- `forecasts`
- `model_metrics`

Provide reusable SQL queries for analytical use cases.

---

# Phase 3 — Exploratory Data Analysis

Create:

`notebooks/01_eda.ipynb`

The notebook should contain a clear narrative, not a collection of disconnected charts.

Include:

- Dataset overview
- Missing-value analysis
- Data-quality analysis
- Country comparison
- Vaccine-type comparison
- Historical trends
- Distribution analysis
- Correlations
- Demographic trends
- Demand trends
- Outlier/anomaly investigation
- Important findings

Each major chart should answer a question.

Do not create visualizations just to increase chart count.

---

# Phase 4 — Feature Engineering

Create:

`src/features/engineer.py`

Potential features:

### Demographic ratios

- births_per_1000pop
- infant_mortality_ratio
- under_5_mortality_ratio

### Lagged variables

- lag_1
- lag_2
- lag_3

### Rolling statistics

- rolling_3_mean
- rolling_5_mean
- rolling_3_std

All rolling features must be shifted appropriately to avoid leakage.

### Growth features

- births_yoy
- population_yoy
- birthrate_change
- mortality_change

### Time features

- years_since_2000
- relevant temporal indicators

### Country encoding

- appropriate categorical encoding

Do not automatically use every proposed feature.

Feature selection must consider the small dataset size.

---

# Phase 5 — ML Pipeline

Create:

`src/models/trainer.py`

## 5.1 Temporal splitting

Do not use random train/test splitting for the forecasting problem.

Use chronological evaluation.

Initial proposed split:

- Training: before 2020
- Test: 2020–2024

Adjust only if inspection of the actual data requires it.

## 5.2 Models

Start with an appropriate baseline.

Potential comparison:

- Linear Regression
- Ridge
- Huber Regression
- Gradient Boosting Regressor
- Random Forest Regressor

Do not use every model automatically.

The final selection must be justified.

## 5.3 Cross-validation

Use `TimeSeriesSplit` where appropriate.

Do not use standard random K-fold validation for temporal forecasting.

## 5.4 Metrics

Evaluate with relevant regression metrics:

- MAE
- RMSE
- MAPE where appropriate
- R²

Do not overinterpret R² on a very small time-series dataset.

## 5.5 Walk-forward validation

Implement walk-forward/backtesting to evaluate how the model behaves over time.

---

# Important Statistical Constraint

The reference data is small, with only a few countries and yearly observations.

Do NOT pretend this is a large-scale ML dataset.

The project should explicitly acknowledge:

- Small sample size
- Limited country coverage
- Potential instability in model comparison
- Forecast uncertainty
- Dependence on demographic projections

The goal is decision-support demonstration, not claiming production-grade global forecasting accuracy.

This limitation should appear in the README and dashboard caveats.

---

# Phase 6 — Forecasting

Create:

`src/models/forecaster.py`

Implement recursive multi-step forecasting for 2025–2030.

For each future step:

1. Use available demographic inputs
2. Construct features
3. Generate prediction
4. Feed appropriate lagged prediction into the next step
5. Continue through the forecast horizon

Avoid leakage from future target values.

---

# Phase 7 — Uncertainty Quantification

Retain the useful Monte Carlo idea from the references, but implement it carefully.

Potential approach:

- Estimate residual distribution
- Use Student-t residuals where justified
- Run approximately 500 simulations initially
- Generate prediction intervals

Report:

- P5
- P25
- P50
- P75
- P95

Clearly distinguish model uncertainty from uncertainty in external demographic projections.

Do not imply that Monte Carlo simulation magically guarantees statistically calibrated confidence intervals.

---

# Phase 8 — Scenario Analysis

Implement scenarios such as:

- Baseline
- Optimistic
- Pessimistic

A pandemic/disruption scenario may be included only if it can be justified using actual assumptions.

Scenario assumptions must be transparent.

Avoid arbitrary manipulations that produce dramatic-looking results.

---

# Phase 9 — Explainability

Create:

`src/models/explainer.py`

Use appropriate explainability methods.

Potential components:

- Model coefficients for linear models
- Permutation importance
- SHAP for the final model if computationally and statistically appropriate

The dashboard should answer:

> Why did the model produce this forecast?

Use plain-language explanations.

Do not present SHAP values as causal effects.

---

# Phase 10 — Error Analysis

Create:

`src/evaluation/metrics.py`

Include:

- Overall metrics
- Per-country performance
- Per-vaccine performance
- Worst prediction years
- Largest absolute errors
- Error distribution
- Backtest results

Investigate potential reasons for large errors.

Do not invent explanations when the data cannot support them.

---

# Phase 11 — Sensitivity Analysis

Create:

`src/evaluation/sensitivity.py`

Include:

- One-at-a-time sensitivity analysis
- Tornado chart
- Elasticity-style curves where mathematically meaningful
- Driver ranking

Clearly distinguish:

- Model sensitivity
- Correlation
- Causal impact

Do not describe model sensitivity as causal evidence.

---

# Phase 12 — SQL Analytics

Create:

`src/analytics/sql_queries.py`

Implement meaningful analytical SQL queries.

Examples:

1. Countries with highest vaccine demand growth
2. Year-over-year demographic changes
3. Countries with largest mortality-rate changes
4. Cross-country demand comparison
5. Procurement cost projections
6. Years with unusual demand changes
7. MCV1 vs MCV2 demand comparison
8. Forecast vs historical demand comparison

The SQL should be real SQL executed against the SQLite database, not hardcoded outputs.

---

# Phase 13 — Procurement and Budget Intelligence

Create a procurement calculation layer.

Inputs may include:

- Predicted vaccine demand
- Wastage rate
- Cost per dose
- Procurement quantity

Outputs:

- Required doses
- Wastage-adjusted quantity
- Estimated procurement cost
- Scenario cost
- Country-level budget comparison

Clearly label assumptions.

Do not present assumed prices as real procurement prices unless sourced.

---

# Phase 14 — KEY DIFFERENTIATOR: INTERACTIVE WHAT-IF ANALYSIS

This should be the standout feature.

Create an interactive what-if workflow.

User selects:

- Country
- Forecast year

User can modify appropriate demographic inputs within realistic bounds.

Potential controls:

- Birth rate: ±20%
- Infant mortality: ±30%
- Population under age 1: ±20%
- Net migration: ±50%

The exact variables and ranges must be determined from the actual dataset.

When the user changes an input:

1. Recalculate features
2. Re-run the model
3. Update forecast
4. Show baseline vs scenario
5. Show absolute and percentage change
6. Show relevant feature importance/explanation
7. Update procurement quantity
8. Update estimated budget

Example explanation:

> Increasing the projected birth rate increases the estimated eligible cohort, which raises the predicted vaccine requirement.

Do not generate unsupported causal claims.

Phrase explanations as model-based interpretations.

---

# Phase 15 — Dashboard Architecture

The dashboard can use Streamlit unless Antigravity determines that another architecture is clearly justified.

The UI should prioritize analytical usability.

Suggested pages:

## 1. Overview

Show:

- 2030 demand
- cumulative forecast demand
- estimated procurement budget
- forecast uncertainty
- country comparison
- historical vs forecast trends

## 2. Analysis

Show:

- demographic trends
- vaccine trends
- country comparisons
- SQL analytics
- anomalies
- distributions

## 3. Prediction

Show:

- country selector
- forecast year
- baseline forecast
- uncertainty interval
- What-If Analysis
- procurement calculator
- explanation

## 4. Model Performance

Show:

- model comparison
- evaluation metrics
- walk-forward results
- actual vs predicted
- feature importance
- error analysis

## 5. Insights

Show:

- major findings
- sensitivity analysis
- country-specific insights
- procurement implications
- recommendations
- limitations

Do not create pages if the actual content does not justify them.

---

# Phase 16 — UI/UX

The dashboard must NOT look AI-generated.

Avoid:

- emojis
- excessive gradients
- glassmorphism
- excessive rounded cards
- giant headings
- random icons
- decorative illustrations
- fake AI branding
- unnecessary animations
- excessive badges
- marketing language

Use:

- restrained color palette
- consistent typography
- consistent spacing
- professional charts
- readable tables
- sensible filters
- clear labels
- strong visual hierarchy

The interface should feel like a serious analytics/internal decision-support application.

---

# Kibo UI

Kibo UI may be used if it fits naturally into the chosen frontend architecture.

Do not force Kibo UI into the project.

If using Kibo UI requires moving from Streamlit to React/Next.js + API infrastructure, first evaluate whether the added engineering complexity is genuinely worthwhile.

Prioritize:

1. ML correctness
2. Analytics quality
3. Product usability
4. Maintainability
5. UI polish

Do not rewrite the entire architecture just for visual components.

If Kibo UI is used, use it selectively and do not make the application look like a Kibo UI component showcase.

---

# Phase 17 — Code Architecture

Suggested structure:

```text
ML Project/
│
├── app/
│   ├── app.py
│   ├── pages/
│   │   ├── 1_Overview.py
│   │   ├── 2_Analysis.py
│   │   ├── 3_Prediction.py
│   │   ├── 4_Model_Performance.py
│   │   └── 5_Insights.py
│   ├── components/
│   │   ├── charts.py
│   │   └── metrics.py
│   └── utils/
│       └── style.py
│
├── data/
│   ├── vaccine_data.csv
│   └── future_demographics.csv
│
├── models/
│   ├── final_model.joblib
│   └── model_metadata.json
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── database.py
│   ├── features/
│   │   └── engineer.py
│   ├── models/
│   │   ├── trainer.py
│   │   ├── forecaster.py
│   │   └── explainer.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── sensitivity.py
│   └── analytics/
│       └── sql_queries.py
│
├── tests/
│   └── test_pipeline.py
│
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

Modify the structure if the final architecture requires it.

Do not create unnecessary abstractions.

---

# Phase 18 — Testing

Create basic tests for:

- Data loading
- Data validation
- Feature engineering
- Model training
- Model persistence
- Inference
- Invalid inputs
- Missing values
- What-if calculations

Run:

```bash
python -m pytest tests/ -v
```

Also verify:

```bash
python train.py
```

and:

```bash
streamlit run app/app.py
```

if Streamlit is the selected architecture.

---

# Phase 19 — Documentation

Create a strong README with:

1. Project overview
2. Problem statement
3. Why this problem matters
4. Data sources
5. Dataset limitations
6. Architecture
7. Data pipeline
8. EDA
9. Feature engineering
10. ML models
11. Validation strategy
12. Forecasting
13. Uncertainty
14. Explainability
15. What-if analysis
16. SQL analytics
17. Procurement calculations
18. Dashboard
19. Results
20. Limitations
21. Future improvements
22. Installation
23. Running instructions

Do not exaggerate performance.

Do not claim production readiness.

---

# Phase 20 — Placement Readiness

The final project should allow me to explain:

- Why this problem?
- Why vaccine procurement?
- Why this dataset?
- Why forecasting?
- Why these features?
- Why time-series validation?
- Why these models?
- Why the final model?
- How did you avoid leakage?
- How does recursive forecasting work?
- How does uncertainty estimation work?
- What does SHAP tell you?
- What is the purpose of the What-If Analysis?
- How does SQL contribute?
- How are procurement costs calculated?
- What are the limitations?
- How would you improve it with more countries/data?
- How would you deploy it?
- How would you monitor it?

Build the project so every major technical decision has a defensible reason.

---

# Phase 21 — Final Validation

Before declaring the project complete:

### ML
- Verify no leakage
- Verify temporal validation
- Verify model metrics
- Verify saved model
- Verify inference

### Data
- Verify source data
- Verify validation
- Verify missing-value handling
- Verify assumptions

### Dashboard
- Every page loads
- Filters work
- Charts work
- What-if updates correctly
- Metrics are consistent with model outputs
- No broken components

### Engineering
- Clean installation
- No hardcoded paths
- No secrets
- Requirements work
- Tests pass

### UX
- No emojis
- No excessive visual decoration
- No fake data
- No AI marketing copy
- Consistent typography and spacing
- Professional analytics aesthetic

---

# Phase 22 — Final Review

Before finishing, critically evaluate the project against these dimensions:

| Dimension | Question |
|---|---|
| Problem | Is the problem meaningful? |
| Data | Is the data appropriate and honestly represented? |
| ML | Is the methodology correct? |
| Validation | Is temporal leakage avoided? |
| Analytics | Are insights meaningful? |
| Explainability | Can predictions be interpreted? |
| Product | Does the system support a decision? |
| Dashboard | Is the UI useful and professional? |
| Engineering | Is the project maintainable? |
| Originality | Is it independently implemented? |
| Interviewability | Can the user explain the whole project? |

If any major dimension is weak, improve it before completion.

---

# Important Constraints

## Do not overengineer

Do not add technologies simply to increase the technology stack.

Do not add:

- Kubernetes
- microservices
- vector databases
- LangChain
- unnecessary cloud infrastructure
- unnecessary APIs

unless there is a genuine requirement.

A technically correct, well-explained project is better than a complex project that cannot be defended in an interview.

## Be honest about the dataset

The reference datasets are small.

Do not make unsupported claims such as:

- "production-grade forecasting"
- "highly accurate global vaccine prediction"
- "causal impact"
- "guaranteed confidence intervals"

Frame GIIP as an ML-powered analytical decision-support prototype.

---

# Final Deliverables

At completion, provide:

## Project

- Final title
- One-line description
- Problem statement
- Target user
- Key features

## ML

- ML task
- Features
- Models tested
- Final model
- Metrics
- Validation methodology
- Explainability

## Analytics

- SQL analyses
- KPIs
- Key findings
- Sensitivity analysis

## Product

- What-if analysis
- Procurement calculator
- Recommendations

## Dashboard

- Page structure
- Main interactions
- UI architecture

## Engineering

- Architecture
- Folder structure
- Tech stack
- Dependencies

## Execution

- Installation commands
- Training command
- Dashboard startup command

## Placement

Provide 5–8 resume-ready achievement bullets and 10 strong interview talking points.

## Limitations

List realistic limitations.

## Future Improvements

List realistic improvements that would matter with larger/more comprehensive data.

---

# Guiding Principle

Do not optimize for the number of technologies.

Optimize for:

**Strong problem**
+
**Correct ML**
+
**Meaningful analytics**
+
**Explainability**
+
**Decision support**
+
**Professional UI**
+
**Clean engineering**
+
**Interviewability**

The final product should be substantially better than both reference repositories while remaining understandable and defensible for the user's current skill level.
