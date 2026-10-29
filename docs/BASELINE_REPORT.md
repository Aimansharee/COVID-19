# 📊 COVID-19 AI — Baseline Assessment & Project Understanding (Phase 0)

**Date**: October 29, 2026  
**Auditor**: Senior Python Engineer, ML Engineer & QA Specialist  
**Phase**: Phase 0 — Baseline Assessment & Architectural Inspection  
**Historical Test Status at Phase 0**: **45 / 45 Passed (100%)** | **Coverage: 71%**  

> [!NOTE]
> **Historical Baseline Document**: This report records the historical Phase 0 baseline assessment.
> For the remediated, non-leaky forecasting architecture, baselines, and current verification (60 passed, 94% coverage),
> please consult [PHASE_1A_REPORT.md](file:///f:/covid-19/docs/PHASE_1A_REPORT.md) and [PHASE_1A_CLOSURE.md](file:///f:/covid-19/docs/PHASE_1A_CLOSURE.md).

---

## 1. Existing Project Architecture

The repository is structured as a data science and machine learning dashboard built with **Streamlit** (v1.40.0) running on **Python 3.11.5**. 

### High-Level Architecture Diagram
```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (main.py)                          │
├───────────────────────────────────┬────────────────────────────────────┤
│    Mode 1: Model Prediction       │  Mode 2: Exploratory Data Analysis │
│  - Parameter tuning sliders       │  - Descriptive statistics & shapes │
│  - Sequential Train/Test Split    │  - Time-series weekly aggregations │
│  - GradientBoostingRegressor fit  │  - Choropleth global spread map    │
│  - Error & Importance charts      │  - Fatality rate over time         │
│  - 5-Country Baseline Comparison  │  - Global vaccination progress     │
└─────────────────▲─────────────────┴──────────────────▲─────────────────┘
                  │                                    │
                  └──────────────────┬─────────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     │ Preprocessing & Pipeline      │
                     │ - preprocess_data()           │
                     │ - create_features()           │
                     └───────────────▲───────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     │ Data Loading (st.cache_data)  │
                     │ - cleaned_covid_data_last.csv │
                     │ - country_vaccinations.csv    │
                     └───────────────────────────────┘
```

### Module Organization
- **`main.py`**: Monolithic core module combining data loading, preprocessing, feature engineering, model training, evaluation metrics, and the full interactive Streamlit interface.
- **`tests/`**: Comprehensive automated test suite comprising 12 test modules covering imports, validation, loading, preprocessing, modeling, predictions, evaluation, visualizations, integration, regression, and security.
- **`docs/`**: Technical reports including `BUG_REPORT.md`, `TEST_REPORT.md`, and `TESTING_GUIDE.md`.

---

## 2. Dataset Overview

The project relies on two primary CSV datasets located in the workspace root:

| Dataset | File Size | Records | Features | Date Range | Missing Values | Integrity Notes |
| :--- | :---:| :---:| :---:| :---:| :---:| :--- |
| **`cleaned_covid_data_last.csv`** | 18.45 MB | 166,326 | 15 | 2020-01-01 to 2022-03-05 | 0 (0.0%) | 238 countries/territories; 0 duplicate (location, date) pairs; 0 negative cases. |
| **`country_vaccinations.csv`** | 4.98 MB | 25,862 | 15 | 2020-12-02 to 2021-06-20 | Varies by country | Filtered to >= 2021 in loader; 0 duplicate (country, date) pairs. |

---

## 3. Actual Machine Learning Models

The repository contains exactly **one** implemented machine learning model architecture:

- **Model Class**: `sklearn.ensemble.GradientBoostingRegressor`
- **Objective Loss**: Huber loss (`loss='huber'`, `alpha=0.95`) to provide robustness against extreme pandemic outlier spikes.
- **Key Hyperparameters (configurable via UI)**:
  - `n_estimators`: 100 to 1,000 (default: 500)
  - `learning_rate`: 0.01 to 0.20 (default: 0.05)
  - `max_depth`: 3 to 10 (default: 7)
  - `subsample`: 0.8
  - `min_samples_split`: 10
  - `min_samples_leaf`: 5
  - `validation_fraction`: 0.1
  - `n_iter_no_change`: 5 (early stopping)
  - `random_state`: 42
- **Model Scope**: Trained on-demand for a single user-selected country (e.g., United States, Egypt, Germany) rather than as a global unified multi-country model.
- **Artifact Persistence**: Models are fitted strictly in-memory during session state; **no serialized model files (.pkl, .joblib, or ONNX)** are saved to disk.

---

## 4. Forecasting Capability Assessment

A rigorous examination of the forecasting mechanics reveals the following:

1. **Prediction Target**:
   - `target = np.log1p(new_cases)` (log-transformed 1-step ahead daily new cases).
   - Predictions are inverted to natural case scale using `np.expm1(y_pred)`.
2. **Forecast Horizon**:
   - The model is strictly a **1-day ahead step-by-step regressor** ($t-1 \to t$).
   - It does **not** perform true multi-step iterative forecasting (e.g., 7-day, 14-day, or 30-day recursive trajectory forecasting into the future).
3. **Input Features**:
   - `lag_1`: 1-day lagged new cases.
   - `lag_7`: 7-day lagged new cases.
   - `rolling_avg_1`: 1-day rolling average (equivalent to current observation).
   - `rolling_avg_7`: 7-day rolling average of new cases.
   - `total_cases`: Cumulative total cases at time $t$.
   - `population`: Country population denominator.
   - `stringency_index`: Oxford Government Response Tracker index at time $t$.
4. **Hypothetical Comparisons (Not Real AI Predictions)**:
   - In the "Model Prediction" view, the section titled *Actual vs Baseline COVID-19 Cases Over Time by Country* displays a line `Predicted (Baseline) = Actual * 1.2` across the top 5 countries. This is an exploratory +20% benchmark, not an actual machine learning forecast.

---

## 5. Current Evaluation Methodology

The current pipeline evaluates predictions on a chronological test split (`X_test`, `y_test`) using:
- **MAE** (Mean Absolute Error): Measures average magnitude of absolute errors in raw case counts.
- **RMSE** (Root Mean Squared Error): Penalizes larger prediction errors heavily.
- **R² Score** (Coefficient of Determination): Proportion of variance explained by the model.
- **Adjusted SMAPE** (Symmetric Mean Absolute Percentage Error): Custom symmetric percentage metric bounded by a threshold (1.0) to prevent zero-case division blowups.
- **Accuracy Estimate**: Defined as `100 - SMAPE`.
- **Accuracy for Cases > 10**: Filtered MAPE accuracy specifically for significant infection days.

### Evaluation Limitations:
- No **Time Series Cross-Validation** (e.g., `TimeSeriesSplit` or rolling walk-forward validation).
- No comparison against a **Naive Persistence Baseline** ($y_t = y_{t-1}$) or **Moving Average Baseline**.
- No confidence intervals or uncertainty bounds around predictions.

---

## 6. Data Leakage Risks & Temporal Integrity

An audit of temporal dependencies revealed three critical nuances:

1. **Global Outlier Filtering**:
   - In `preprocess_data()`, quantiles (`q_low = 0.05` and `q_high = 0.95`) are computed on the entire dataset across all years and countries simultaneously. Trimming the 5%–95% quantiles before the train/test chronological split introduces minor global distribution leakage across time.
2. **Contemporaneous Features at Step $t$**:
   - `total_cases` and `stringency_index` are extracted at time $t$. In a real multi-step future forecasting scenario, these values are unknown at forecast time unless modeled or estimated independently.
3. **Chronological Splitting (Preserved)**:
   - The train/test split correctly utilizes sequential slicing (`iloc[:split_idx]` and `iloc[split_idx:]`) rather than random k-fold shuffling, successfully avoiding future-to-past temporal leakage.

---

## 7. Existing Dashboard Capabilities

The Streamlit dashboard (`main.py`) provides two operational modes:

### Mode 1: Model Prediction
- Country selection dropdown (238 locations).
- Interactive hyperparameter sliders (Estimators, Learning Rate, Tree Depth, Split Ratio).
- On-click model training button with progress spinner.
- Metric display cards (`MAE`, `RMSE`, `R² Score`, `Adjusted SMAPE`, `Accuracy Estimate`).
- Sample prediction inspection table.
- Interactive Plotly visualizations:
  - Actual vs. Predicted scatter plot with perfect prediction diagonal.
  - Prediction error (residuals) histogram.
  - Horizontal feature importance bar chart.
  - Interactive time-series line comparison for the selected country.
  - Top 5 countries baseline comparison line chart.
  - Matplotlib relative error vs. actual scatter plot.

### Mode 2: Exploratory Data Analysis (EDA)
- High-level dataset shape and data type summaries.
- Descriptive statistical table (mean, std, min, percentiles, max) for numeric columns.
- Missing values summary.
- Global weekly trend line chart by country (Dark Plotly theme).
- Heatmap correlation matrix of key indicators.
- Top 20 countries by average case fatality rate horizontal bar chart.
- Global choropleth world map of COVID-19 spread (Cases per 100k).
- Log-scale scatter plot: Total Cases vs. Total Deaths by population.
- Monthly evolution of average fatality rate.
- Vaccination analytics: Global cumulative doses over time and Top 10 vaccinated countries.

---

## 8. Automated Test Results

The automated test suite was executed in the workspace environment on October 29, 2026:

- **Command**: `python -m pytest tests/ -v`
- **Platform**: Windows 11 (AMD64)
- **Python**: 3.11.5
- **Pytest**: 9.1.1
- **Plugins**: `anyio-3.7.1`, `dash-2.18.2`, `Faker-4.14.0`, `langsmith-0.8.9`, `asyncio-1.4.0`, `cov-7.1.0`

### Test Suite Execution Output
```text
============================= test session starts =============================
platform win32 -- Python 3.11.5, pytest-9.1.1, pluggy-1.6.0
rootdir: F:\covid-19
collected 45 items

tests/test_application.py::test_app_startup PASSED                       [  2%]
tests/test_application.py::test_app_sidebar_controls PASSED              [  4%]
tests/test_application.py::test_app_switch_to_eda_mode PASSED            [  6%]
tests/test_data_loading.py::test_load_data_default_file PASSED           [  8%]
tests/test_data_loading.py::test_load_vaccine_data_default_file PASSED   [ 11%]
tests/test_data_loading.py::test_load_vaccine_data_missing_file PASSED   [ 13%]
tests/test_data_loading.py::test_load_data_custom_path PASSED            [ 15%]
tests/test_data_validation.py::test_covid_data_schema PASSED             [ 17%]
tests/test_data_validation.py::test_covid_no_negative_cumulative_totals PASSED [ 20%]
tests/test_data_validation.py::test_covid_population_positive PASSED     [ 22%]
tests/test_data_validation.py::test_covid_uniqueness_per_location_date PASSED [ 24%]
tests/test_data_validation.py::test_vaccine_data_schema PASSED           [ 26%]
tests/test_data_validation.py::test_vaccine_date_order PASSED            [ 28%]
tests/test_imports.py::test_python_version PASSED                        [ 31%]
tests/test_imports.py::test_required_packages_installed[streamlit] PASSED [ 33%]
tests/test_imports.py::test_required_packages_installed[pandas] PASSED   [ 35%]
tests/test_imports.py::test_required_packages_installed[numpy] PASSED    [ 37%]
tests/test_imports.py::test_required_packages_installed[sklearn] PASSED  [ 40%]
tests/test_imports.py::test_required_packages_installed[scipy] PASSED    [ 42%]
tests/test_imports.py::test_required_packages_installed[plotly] PASSED   [ 44%]
tests/test_imports.py::test_required_packages_installed[matplotlib] PASSED [ 46%]
tests/test_imports.py::test_main_module_import PASSED                    [ 48%]
tests/test_model_evaluation.py::test_adjusted_smape_perfect_prediction PASSED [ 51%]
tests/test_model_evaluation.py::test_adjusted_smape_empty_mask_threshold PASSED [ 53%]
tests/test_model_evaluation.py::test_adjusted_smape_known_calculation PASSED [ 55%]
tests/test_model_evaluation.py::test_evaluate_predictions_metrics PASSED [ 57%]
tests/test_model_prediction.py::test_model_prediction_single_and_batch PASSED [ 60%]
tests/test_models.py::test_build_model_default_hyperparameters PASSED    [ 62%]
tests/test_models.py::test_model_training_smoke PASSED                   [ 64%]
tests/test_preprocessing.py::test_preprocess_data_missing_values PASSED  [ 66%]
tests/test_preprocessing.py::test_preprocess_data_does_not_mutate_original PASSED [ 68%]
tests/test_preprocessing.py::test_preprocess_data_outlier_filtering PASSED [ 71%]
tests/test_preprocessing.py::test_create_features_output_columns PASSED  [ 73%]
tests/test_preprocessing.py::test_create_features_lag_correctness PASSED [ 75%]
tests/test_regression.py::test_regression_bug_001_division_by_zero_fatality_rate PASSED [ 77%]
tests/test_regression.py::test_regression_bug_002_pandas_frequency_me PASSED [ 80%]
tests/test_regression.py::test_regression_bug_003_vectorized_lag_features_no_deprecation PASSED [ 82%]
tests/test_regression.py::test_regression_bug_005_preprocess_data_copy_integrity PASSED [ 84%]
tests/test_regression.py::test_regression_bug_007_robust_path_resolution PASSED [ 86%]
tests/test_regression.py::test_regression_bug_009_matplotlib_figure_closed PASSED [ 88%]
tests/test_security.py::test_no_hardcoded_secrets_in_source_code PASSED  [ 91%]
tests/test_security.py::test_no_patient_identifiable_information PASSED  [ 93%]
tests/test_visualization.py::test_plotly_scatter_actual_vs_predicted PASSED [ 95%]
tests/test_visualization.py::test_plotly_choropleth_generation PASSED    [ 97%]
tests/test_visualization.py::test_matplotlib_relative_error_figure_cleanup PASSED [100%]

============================= 45 passed in 18.16s =============================
```

- **Tests Collected**: 45
- **Tests Passed**: 45
- **Tests Failed**: 0
- **Tests Skipped**: 0

---

## 9. Current Code Coverage

- **Command**: `python -m pytest tests/ --cov=main --cov-report=term-missing`
- **Coverage Tool**: `coverage 7.17.0` + `pytest-cov 7.1.0`

### Coverage Summary Table
| Module | Statements | Missing Lines | Coverage Percentage | Missing Details |
| :--- | :---:| :---:| :---:| :--- |
| **`main.py`** | 219 | 64 | **71%** | Lines 54–55 (corrupted vaccine branch), Lines 211–362 (interactive button clicks & detailed EDA layout blocks) |
| **TOTAL** | **219** | **64** | **71%** | |

---

## 10. Application Startup Verification

The application startup was independently verified using Streamlit's official headless test harness:

```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('main.py', default_timeout=30)
at.run()
assert not at.exception
```

### Verification Findings:
- ✅ Application boots in **< 4.5 seconds**.
- ✅ Page title renders correctly: `🦠 COVID-19 Cases Prediction Model`.
- ✅ Sidebar loads model parameter sliders and operational mode radio buttons.
- ✅ Mode switching between *Model Prediction* and *Exploratory Data Analysis* runs without exceptions.
- ✅ Data caching (`@st.cache_data`) successfully accelerates consecutive executions.

---

## 11. Identified Technical Gaps

Before elevating this project into an enterprise-grade GitHub portfolio showcase, the following technical gaps have been identified:

1. **Absence of True Multi-Day Trajectory Forecasting**: The model predicts 1-step test cases rather than projecting 7-day, 14-day, or 30-day future horizons.
2. **Single-Model Dependency**: Only `GradientBoostingRegressor` is supported; no comparative benchmarks against **Random Forest**, **XGBoost / LightGBM**, or **Prophet / ARIMA / Baseline Persistence**.
3. **Lack of Model Serialization & Persistence**: Models cannot be exported or imported (`.joblib` / `.onnx`), requiring retrain on every slider interaction.
4. **Monolithic Single-Page Architecture**: All UI code resides in `main.py` (450+ lines) instead of modern Streamlit multipage structure (`pages/` or `st.navigation`).
5. **No Confidence Intervals / Uncertainty Estimation**: Predictions are purely point estimates with no quantile regression or probabilistic error bands.
6. **No Prediction Export Utility**: Users cannot download predicted time-series or evaluation reports as CSV/Excel.
7. **Global Outlier Leakage**: Outlier filtering is applied across the entire dataset rather than within training folds.

---

## 12. Recommended Improvements by Priority

| Priority | Category | Proposed Enhancement | Technical & Portfolio Value |
| :---: | :--- | :--- | :--- |
| **P1** | **ML Forecasting** | **Multi-Horizon Forecasting Engine**: Implement iterative multi-step ahead rollouts (7-day, 14-day, 30-day). | Transforms 1-step evaluation into true future epidemiological forecasting. |
| **P1** | **ML Architecture** | **Model Comparison Hub**: Add comparative models (Random Forest, XGBoost, Naive Baseline) side-by-side. | Demonstrates rigorous machine learning benchmarking. |
| **P2** | **Evaluation & Validation** | **Time-Series Walk-Forward Validation**: Implement chronological rolling-origin evaluation. | Eliminates data leakage and provides publication-grade cross-validation. |
| **P2** | **Model Persistence** | **Model Export / Loading**: Save trained models with metadata to disk (`artifacts/models/`). | Enables real-world ML deployment workflows. |
| **P3** | **UI / UX Architecture** | **Modular Multipage Redesign**: Decompose `main.py` into clean modular components and dedicated pages. | Follows modern software engineering best practices. |
| **P3** | **Interactive Analytics** | **Uncertainty Bounds & CSV Export**: Render 90% confidence bands and download buttons for forecasts. | Enhances stakeholder usability and analytical depth. |

---

## 13. Risks and Constraints for Phase 1

1. **Computational Overhead on Streamlit**: Training multiple complex ensemble models (e.g., XGBoost, Gradient Boosting, Random Forest) on large country histories could cause UI latency if not optimized with caching or precomputed weights.
2. **Missing Future Exogenous Variables**: Features like `stringency_index` and `total_cases` are not known 14 days into the future; a recursive forecasting engine must rely on autoregressive lags (`lag_1`..`lag_7`) and rolling statistics.
3. **Zero Breaking Changes to Existing Tests**: All Phase 1 enhancements must maintain 100% pass rates across existing automated tests without degrading current functionality.

---

**PHASE 0 COMPLETE — READY FOR REVIEW**
