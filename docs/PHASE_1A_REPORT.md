# 🛡️ COVID-19 AI — Phase 1A: Data Leakage Remediation & Forecasting Foundation Report

**Date**: October 29, 2026  
**Auditor & Specialist**: Senior Machine Learning Engineer, Time-Series Specialist & QA Automation Engineer  
**Phase**: Phase 1A — Data Leakage Remediation & Forecasting Foundation  
**Status**: **56 / 56 Automated Tests Passed (100%)** | **Code Coverage: 96%**  

---

## Executive Summary

Phase 1A rigorously audited the forecasting architecture in [main.py](file:///f:/covid-19/main.py), confirmed severe target and temporal data leakage, and successfully remediated all confirmed defects. A defensible one-step-ahead ($t-1 \to t$) time-series forecasting pipeline was established alongside empirical baseline models (**Naive Persistence** and **7-Day Trailing Moving Average**).

Key accomplishments:
1. **Target Leakage Eliminated**: Discovered and proved that `rolling_avg_1` was mathematically identical to the unshifted target $y_t = \text{new\_cases}_t$, giving the ML model the exact label during training. All rolling statistics were converted to trailing windows ending strictly at forecast origin $t-1$.
2. **Contemporaneous Feature Leakage Remedied**: `total_cases` (cumulative cases) and `stringency_index` were shifted by 1 day so that predictions for day $t$ strictly utilize historical metrics available by the end of day $t-1$.
3. **Preprocessing Temporal Leakage Fixed**: Replaced the global quantile filtering (which trimmed the 5%–95% quantiles across the entire multi-year global dataset before splitting, wiping out 70% of peak infection days) with training-only parameter learning (`fit_outlier_bounds` / `apply_outlier_bounds`) and preserved authentic epidemiological curve peaks in forecasting.
4. **Baselines Implemented & Evaluated**: Created `NaivePersistenceModel` and `MovingAverageModel` and evaluated them against `GradientBoostingRegressor` over identical chronological evaluation windows.
5. **Rigorous Empirical Findings Documented**: As required by scientific integrity, we report that the simple **7-Day Moving Average Baseline** ($R^2 = 0.7078$, $\text{MAE} = 85,157.50$ in the US) significantly outperformed the Gradient Boosting Regressor ($R^2 = 0.0522$, $\text{MAE} = 130,179.59$) during the late-2021 Omicron regime shift, as decision trees cannot extrapolate beyond the training target range.
6. **Testing & Coverage Expansion**: Preserved all 45 original tests and added 11 new tests (including future target mutation invariance tests and Streamlit app flow integration), bringing total tests to **56 passing** and boosting core coverage from **71% to 96%**.

---

## 1. Confirmed Leakage Findings & Diagnostic Evidence

### Leakage 1: Exact Contemporaneous Target Leakage via `rolling_avg_1`
- **Location**: [main.py:create_features()](file:///f:/covid-19/main.py#L78-L100)
- **Defect Mechanism**:
  ```python
  # Original code:
  df[f'rolling_avg_{lag}'] = (
      df.groupby('location')['new_cases']
      .rolling(lag, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
  )
  ```
  For `lag = 1`, Pandas computes a rolling mean with a window of 1 on the contemporaneous series:
  $$\text{rolling\_avg\_1}_t = \frac{1}{1} \sum_{k=0}^0 \text{new\_cases}_{t-k} = \text{new\_cases}_t$$
  Since the target was defined as `target = np.log1p(new_cases)`, `rolling_avg_1` was literally the raw target label placed directly into the feature matrix $X$ for day $t$. The model trivially learned `np.log1p(rolling_avg_1)`, yielding artificial, fraudulent $R^2 \approx 0.999$.

### Leakage 2: Unshifted Rolling Window Leakage in `rolling_avg_7`
- **Location**: [main.py:create_features()](file:///f:/covid-19/main.py#L78-L100)
- **Defect Mechanism**:
  For `lag = 7`, the rolling window ended at day $t$:
  $$\text{rolling\_avg\_7}_t = \frac{1}{7} (\text{new\_cases}_t + \text{new\_cases}_{t-1} + \dots + \text{new\_cases}_{t-6})$$
  The feature contained the target day's value $\text{new\_cases}_t$, leaking 14.3% of the target into the regressor at inference time.

### Leakage 3: Contemporaneous Cumulative Totals and Policy Indices
- **Location**: [main.py:create_features()](file:///f:/covid-19/main.py#L78-L100) and `features` list
- **Defect Mechanism**:
  `total_cases` at date $t$ is defined epidemiologically as $\text{total\_cases}_t = \text{total\_cases}_{t-1} + \text{new\_cases}_t$. In the previous code, `total_cases` was supplied unshifted. Similarly, Oxford `stringency_index` was supplied at day $t$. At forecast origin $t-1$, cumulative numbers and same-day policy scores for day $t$ have not yet been reported.

### Leakage 4: Global Preprocessing Quantile Leakage & Severe Epidemic Curve Truncation
- **Location**: [main.py:preprocess_data()](file:///f:/covid-19/main.py#L58-L76)
- **Defect Mechanism**:
  ```python
  q_low = df['new_cases'].quantile(0.05)
  q_high = df['new_cases'].quantile(0.95)
  df = df[(df['new_cases'] >= q_low) & (df['new_cases'] <= q_high)]
  ```
  1. **Temporal Leakage**: Quantiles were computed across the entire global multi-year dataset prior to chronological train/test splitting, leaking future distribution statistics into training folds.
  2. **Severe Time-Series Discontinuity**: The global 95th percentile across 238 territories was $32,969$ cases. For major nations (such as the United States), 543 out of 774 days (70.1% of the entire pandemic history!) were deleted. This fractured the time axis, creating synthetic gaps of up to 223 days across which `shift(1)` operated as if consecutive. Real outbreak peaks were erroneously expunged.

---

## 2. Exact Affected Functions, Files, and Applied Fixes

| Component / Function | File | Defect Addressed | Applied Fix Summary |
| :--- | :--- | :--- | :--- |
| `create_features(df)` | [main.py](file:///f:/covid-19/main.py) | Target leakage in `rolling_avg_1`, `rolling_avg_7`, contemporaneous `total_cases`, `stringency_index` | Shifted all historical calculations to end at $t-1$: `new_cases.shift(1).rolling(lag, min_periods=1).mean()`, and shifted `total_cases` and `stringency_index` by 1 day. |
| `preprocess_data(df, filter_outliers, q_low, q_high)` | [main.py](file:///f:/covid-19/main.py) | Global preprocessing leakage & epidemic peak truncation | Added `filter_outliers=False` default for continuous time-series forecasting. Added support for passing externally fitted bounds `(q_low, q_high)`. Added zero-clipping and country-grouped forward filling. |
| `fit_outlier_bounds(train_df)` | [main.py](file:///f:/covid-19/main.py) | Lack of training-only parameter fitting | New function learning lower and upper quantiles exclusively from training fold observations. |
| `apply_outlier_bounds(df, q_low, q_high)` | [main.py](file:///f:/covid-19/main.py) | Outlier evaluation leakage | New function applying training-learned thresholds to validation/test partitions without refitting. |
| `NaivePersistenceModel` | [main.py](file:///f:/covid-19/main.py) | Absence of persistence baseline | New scikit-learn compatible class predicting $y_t = y_{t-1}$ from `lag_1`. |
| `MovingAverageModel` | [main.py](file:///f:/covid-19/main.py) | Absence of moving average baseline | New scikit-learn compatible class predicting $y_t = \frac{1}{W} \sum_{i=1}^W y_{t-i}$ using trailing history only. |
| `main()` UI Workflow | [main.py](file:///f:/covid-19/main.py) | Artificial baseline (`Actual * 1.2`) & lack of comparative benchmarking | Replaced artificial benchmark with true 7-Day Moving Average Baseline. Added live comparative metrics table comparing Gradient Boosting against both baselines. |

---

## 3. Feature Availability & Information Origin Matrix

All features are now formally tied to the **Forecast Origin ($T = t-1$)** for target date $t$:

| Feature Name | Temporal Formula | Information Source | Available at Origin $t-1$? | Leakage Status |
| :--- | :--- | :--- | :---: | :--- |
| `target` | $\log(1 + \text{new\_cases}_t)$ | Ground truth at step $t$ | **NO** (Target variable) | Isolated as prediction objective |
| `lag_1` | $\text{new\_cases}_{t-1}$ | Observed cases on day $t-1$ | **YES** | **Clean / Zero Leakage** |
| `lag_7` | $\text{new\_cases}_{t-7}$ | Observed cases on day $t-7$ | **YES** | **Clean / Zero Leakage** |
| `rolling_avg_1` | $\text{new\_cases}_{t-1}$ | Trailing 1-day average at $t-1$ | **YES** | **Clean (remediated from leaky $t$)** |
| `rolling_avg_7` | $\frac{1}{7} \sum_{i=1}^7 \text{new\_cases}_{t-i}$ | Trailing 7-day average ($t-7 \dots t-1$) | **YES** | **Clean (remediated from leaky $t$)** |
| `total_cases` | $\text{total\_cases}_{t-1}$ | Cumulative cases reported up to $t-1$ | **YES** | **Clean (remediated from leaky $t$)** |
| `stringency_index` | $\text{stringency\_index}_{t-1}$ | Policy index active at $t-1$ | **YES** | **Clean (remediated from leaky $t$)** |
| `population` | $\text{population}$ | Static demographic census statistic | **YES** | **Clean** |

---

## 4. Corrected Forecasting Pipeline Methodology

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                    Input Data (cleaned_covid_data_last.csv)            │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 1. Preprocessing (preprocess_data(df, filter_outliers=False))          │
 │    - Per-country forward filling & zero imputations                    │
 │    - Negative case clipping (lower bound = 0.0)                        │
 │    - Chronological date sorting (retains all epidemic peaks)           │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 2. Feature Engineering (create_features(df))                           │
 │    - Lags: lag_1, lag_7                                                │
 │    - Shifted Trailing Rolling Means: rolling_avg_1, rolling_avg_7      │
 │    - Shifted Cumulative Totals & Policy: total_cases, stringency_index │
 │    - Target Transformation: target = log1p(new_cases_t)                │
 │    - NaN Dropping for initial warm-up period (first 7 days)            │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 3. Chronological Sequential Split (80% Train / 20% Test)              │
 │    - Train Period: [T_start, T_split]                                  │
 │    - Forecast Origin: T_split                                          │
 │    - Test Period: [T_split + 1 day, T_end]                             │
 │    - Zero temporal overlap or shuffling                                │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
             ┌───────────────────────┼───────────────────────┐
             │                       │                       │
             ▼                       ▼                       ▼
 ┌───────────────────────┐ ┌───────────────────┐ ┌────────────────────────┐
 │ Model 1: Persistence  │ │ Model 2: 7-Day MA │ │ Model 3: Gradient      │
 │ y_pred = lag_1        │ │ y_pred = roll_7   │ │ Boosting Regressor     │
 └───────────┬───────────┘ └─────────┬─────────┘ └───────────┬────────────┘
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 4. Multi-Metric Rigorous Evaluation (evaluate_predictions)             │
 │    - MAE, RMSE, R² Score, Adjusted SMAPE, Accuracy                     │
 │    - Exact identical test observations and dates                       │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Real Empirical Evaluation Metrics

Below are the actual, non-invented evaluation results measured on the cleaned dataset under identical chronological test periods (Train: first 80%, Test: final 20%):

### Country 1: United States
- **Total Historical Days**: 767 continuous observations (2020-01-29 to 2022-03-05).
- **Training Period**: 2020-01-29 to 2021-10-02 ($N = 613$).
- **Test Period**: 2021-10-03 to 2022-03-05 ($N = 154$).
- **Context**: The test period encompasses the massive Winter 2021–2022 Omicron surge (where daily cases rose from ~100k to over 800k daily).

| Model | MAE | RMSE | $R^2$ Score | Adjusted SMAPE | Status |
| :--- | :---:| :---:| :---:| :---:| :--- |
| **7-Day Moving Average Baseline** | **85,157.50** | **143,771.58** | **0.7078** | **23.95%** | **Best Performance** |
| **Naive Persistence Baseline (Lag 1)** | 100,823.62 | 176,886.25 | 0.5576 | 28.36% | Strong Baseline |
| **Gradient Boosting Regressor (Huber)** | 130,179.59 | 258,924.25 | 0.0522 | 34.22% | Underperformed Baselines |

### Country 2: Germany
- **Total Historical Days**: 765 continuous observations (2020-01-30 to 2022-03-05).
- **Training Period**: 2020-01-30 to 2021-10-03 ($N = 612$).
- **Test Period**: 2021-10-04 to 2022-03-05 ($N = 153$).

| Model | MAE | RMSE | $R^2$ Score | Adjusted SMAPE | Status |
| :--- | :---:| :---:| :---:| :---:| :--- |
| **7-Day Moving Average Baseline** | **31,055.83** | **50,430.35** | **0.5739** | **28.71%** | **Best Performance** |
| **Naive Persistence Baseline (Lag 1)** | 33,645.42 | 59,569.55 | 0.4055 | 32.14% | Strong Baseline |
| **Gradient Boosting Regressor (Huber)** | 62,701.81 | 97,650.03 | -0.5976 | 43.18% | Negative $R^2$ (Regime Shift) |

### Country 3: United Kingdom
- **Total Historical Days**: 760 continuous observations (2020-02-04 to 2022-03-05).
- **Training Period**: 2020-02-04 to 2021-10-04 ($N = 608$).
- **Test Period**: 2021-10-05 to 2022-03-05 ($N = 152$).

| Model | MAE | RMSE | $R^2$ Score | Adjusted SMAPE | Status |
| :--- | :---:| :---:| :---:| :---:| :--- |
| **Naive Persistence Baseline (Lag 1)** | **9,956.93** | **18,351.90** | **0.8145** | **12.45%** | **Best Performance** |
| **7-Day Moving Average Baseline** | 13,076.91 | 20,738.55 | 0.7631 | 15.62% | Strong Baseline |
| **Gradient Boosting Regressor (Huber)** | 29,139.37 | 48,922.85 | -0.3184 | 29.80% | Underperformed Baselines |

### Why Did the ML Model Underperform the Baselines?
1. **Decision Tree Extrapolation Limits**: Gradient Boosted Decision Trees partition feature space with orthogonal hyperplanes. Consequently, trees **cannot extrapolate** values higher than the maximum target observed in the training distribution ($\approx 250,000$ in US 2020–2021). When the Omicron variant drove actual cases to $800,000+$ per day, the GBR topped out at its historical bound, producing large positive errors.
2. **Weekly Seasonality**: Daily reported cases exhibit strong weekend drops and midweek catch-up surges. A 7-day moving average acts as an optimal filter for weekly administrative reporting cycles, while persistence preserves the immediate trajectory without overfitting to historical levels.
3. **Scientific Value**: Reporting these results establishes credibility. In genuine time-series forecasting, complex ML models without differencing or trend-stationarity often trail moving averages during non-stationary regime shifts.

---

## 6. Regression and Leakage Test Suite

A dedicated automated test module [tests/test_leakage_and_baselines.py](file:///f:/covid-19/tests/test_leakage_and_baselines.py) and Streamlit integration test was created:

1. `test_no_current_target_leakage_rolling_avg_1`:
   Confirms mathematically that for all observations, `rolling_avg_1` does not equal `new_cases[t]`, is decoupled from the current target, and equals `lag_1`.
2. `test_shifted_rolling_averages_use_only_past_observations`:
   Validates that `rolling_avg_7` is the exact mean of days $t-7$ through $t-1$ and excludes day $t$.
3. `test_contemporaneous_cumulative_and_stringency_lagged`:
   Verifies that `total_cases` and `stringency_index` are shifted by 1 day, preventing same-day cumulative or policy leakage.
4. `test_future_target_mutation_leakage_invariance`:
   **Strongest Leakage Test**: Modifies future test cases by $+50,000$ in a synthetic series and verifies that historical features up to day $t$ remain 100% bit-for-bit identical.
5. `test_training_only_preprocessing_outlier_bounds`:
   Tests that `fit_outlier_bounds` learns quantiles exclusively from the training partition and that test spikes do not influence training statistics.
6. `test_chronological_split_isolation`:
   Confirms strict time ordering: $\max(\text{train\_dates}) < \min(\text{test\_dates})$.
7. `test_naive_persistence_baseline_behavior`:
   Tests `NaivePersistenceModel` prediction logic on both DataFrames and arrays.
8. `test_moving_average_baseline_behavior`:
   Tests `MovingAverageModel` trailing window calculations.
9. `test_comparable_model_evaluation_periods`:
   Asserts that all three candidate models evaluate over identical observation counts, timestamps, and target vectors.
10. `test_safe_handling_of_zero_and_missing_cases`:
    Tests robustness to missing values, leading NaNs, and consecutive zero-case periods.
11. `test_app_model_training_flow` in [tests/test_application.py](file:///f:/covid-19/tests/test_application.py):
    Streamlit native `AppTest` verifying interactive button click, model fitting, baseline generation, and comparison table rendering without UI exceptions.

---

## 7. Complete Verification Results

### Test Suite Execution
```text
============================= test session starts =============================
platform win32 -- Python 3.11.5, pytest-9.1.1, pluggy-1.6.0
rootdir: F:\covid-19
collected 56 items

tests/test_application.py::test_app_startup PASSED                       [  1%]
tests/test_application.py::test_app_sidebar_controls PASSED              [  3%]
tests/test_application.py::test_app_switch_to_eda_mode PASSED            [  5%]
tests/test_application.py::test_app_model_training_flow PASSED           [  7%]
tests/test_data_loading.py::test_load_data_default_file PASSED           [  8%]
tests/test_data_loading.py::test_load_vaccine_data_default_file PASSED   [ 10%]
tests/test_data_loading.py::test_load_vaccine_data_missing_file PASSED   [ 12%]
tests/test_data_loading.py::test_load_data_custom_path PASSED            [ 14%]
tests/test_data_validation.py::test_covid_data_schema PASSED             [ 16%]
tests/test_data_validation.py::test_covid_no_negative_cumulative_totals PASSED [ 17%]
tests/test_data_validation.py::test_covid_population_positive PASSED     [ 19%]
tests/test_data_validation.py::test_covid_uniqueness_per_location_date PASSED [ 21%]
tests/test_data_validation.py::test_vaccine_data_schema PASSED           [ 23%]
tests/test_data_validation.py::test_vaccine_date_order PASSED            [ 25%]
tests/test_imports.py::test_python_version PASSED                        [ 26%]
tests/test_imports.py::test_required_packages_installed[streamlit] PASSED [ 28%]
tests/test_imports.py::test_required_packages_installed[pandas] PASSED   [ 30%]
tests/test_imports.py::test_required_packages_installed[numpy] PASSED    [ 32%]
tests/test_imports.py::test_required_packages_installed[sklearn] PASSED  [ 33%]
tests/test_imports.py::test_required_packages_installed[scipy] PASSED    [ 35%]
tests/test_imports.py::test_required_packages_installed[plotly] PASSED   [ 37%]
tests/test_imports.py::test_required_packages_installed[matplotlib] PASSED [ 39%]
tests/test_imports.py::test_main_module_import PASSED                    [ 41%]
tests/test_leakage_and_baselines.py::test_no_current_target_leakage_rolling_avg_1 PASSED [ 42%]
tests/test_leakage_and_baselines.py::test_shifted_rolling_averages_use_only_past_observations PASSED [ 44%]
tests/test_leakage_and_baselines.py::test_contemporaneous_cumulative_and_stringency_lagged PASSED [ 46%]
tests/test_leakage_and_baselines.py::test_future_target_mutation_leakage_invariance PASSED [ 48%]
tests/test_leakage_and_baselines.py::test_training_only_preprocessing_outlier_bounds PASSED [ 50%]
tests/test_leakage_and_baselines.py::test_chronological_split_isolation PASSED [ 51%]
tests/test_leakage_and_baselines.py::test_naive_persistence_baseline_behavior PASSED [ 53%]
tests/test_leakage_and_baselines.py::test_moving_average_baseline_behavior PASSED [ 55%]
tests/test_leakage_and_baselines.py::test_comparable_model_evaluation_periods PASSED [ 57%]
tests/test_leakage_and_baselines.py::test_safe_handling_of_zero_and_missing_cases PASSED [ 58%]
tests/test_model_evaluation.py::test_adjusted_smape_perfect_prediction PASSED [ 60%]
tests/test_model_evaluation.py::test_adjusted_smape_empty_mask_threshold PASSED [ 62%]
tests/test_model_evaluation.py::test_adjusted_smape_known_calculation PASSED [ 64%]
tests/test_model_evaluation.py::test_evaluate_predictions_metrics PASSED [ 66%]
tests/test_model_prediction.py::test_model_prediction_single_and_batch PASSED [ 67%]
tests/test_models.py::test_build_model_default_hyperparameters PASSED    [ 69%]
tests/test_models.py::test_model_training_smoke PASSED                   [ 71%]
tests/test_preprocessing.py::test_preprocess_data_missing_values PASSED  [ 73%]
tests/test_preprocessing.py::test_preprocess_data_does_not_mutate_original PASSED [ 75%]
tests/test_preprocessing.py::test_preprocess_data_outlier_filtering PASSED [ 76%]
tests/test_preprocessing.py::test_create_features_output_columns PASSED  [ 78%]
tests/test_preprocessing.py::test_create_features_lag_correctness PASSED [ 80%]
tests/test_regression.py::test_regression_bug_001_division_by_zero_fatality_rate PASSED [ 82%]
tests/test_regression.py::test_regression_bug_002_pandas_frequency_me PASSED [ 83%]
tests/test_regression.py::test_regression_bug_003_vectorized_lag_features_no_deprecation PASSED [ 85%]
tests/test_regression.py::test_regression_bug_005_preprocess_data_copy_integrity PASSED [ 87%]
tests/test_regression.py::test_regression_bug_007_robust_path_resolution PASSED [ 89%]
tests/test_regression.py::test_regression_bug_009_matplotlib_figure_closed PASSED [ 91%]
tests/test_security.py::test_no_hardcoded_secrets_in_source_code PASSED  [ 92%]
tests/test_security.py::test_no_patient_identifiable_information PASSED  [ 94%]
tests/test_visualization.py::test_plotly_scatter_actual_vs_predicted PASSED [ 96%]
tests/test_visualization.py::test_plotly_choropleth_generation PASSED    [ 98%]
tests/test_visualization.py::test_matplotlib_relative_error_figure_cleanup PASSED [100%]

============================= 56 passed in 18.60s =============================
```

### Coverage Report
```text
Name      Stmts   Miss  Cover   Missing
---------------------------------------
main.py     292     13    96%   54-55, 88, 118, 124, 133-134, 160, 183-185, 194, 204
---------------------------------------
TOTAL       292     13    96%
```
*(Missing lines are defensive exception blocks and rare fallback handlers).*

---

## 8. Remaining Technical Risks & Architecture Observations

1. **Non-Stationarity & Tree Model Under-extrapolation**: Standard regression trees cannot extrapolate above $\max(y_{\text{train}})$. During unprecedented epidemic surges, raw-level predictions will consistently underestimate reality unless modeled in terms of first-differences ($\Delta y_t$) or ratio multipliers ($y_t / y_{t-1}$).
2. **Weekly Seasonality**: Daily pandemic metrics exhibit strong 7-day cyclicality due to weekend reporting lags. Incorporating day-of-week one-hot encoding or 7-day seasonal differencing will stabilize predictions.
3. **Multi-Step Horizons**: The current pipeline is strictly 1-step ahead ($t \to t+1$). Predicting $t+7$ or $t+14$ recursively requires careful chained predictions without compounding errors.

---

## 9. Recommended Phase 1B Requirements

1. **Stationary Target Engineering**:
   - Model logarithmic rate of change or differences ($\Delta \log(1 + \text{new\_cases}_t) = \log(1 + y_t) - \log(1 + y_{t-1})$) to enable extrapolation during outbreak waves.
2. **Calendar & Seasonality Features**:
   - Introduce cyclical encoding for `day_of_week` ($\sin/\cos$ transformations) to capture weekend reporting dips without overfitting.
3. **Expanded Model Zoo**:
   - Implement Random Forest Regressor and XGBoost Regressor with hyperparameter tuning via TimeSeriesSplit walk-forward validation.
4. **Multi-Horizon Support**:
   - Design direct multi-horizon forecasters for 7-day and 14-day cumulative projections.
5. **Confidence & Prediction Intervals**:
   - Implement quantile regression (e.g. 10th and 90th percentiles using GradientBoostingRegressor `loss='quantile'`) to provide uncertainty bounds.

---

## 10. Acceptance Verification Checklist

- [x] Confirmed target leakage in `rolling_avg_1` and `rolling_avg_7` investigated and eliminated.
- [x] Information availability at forecast origin ($t-1$) explicitly established and verified.
- [x] Global outlier preprocessing leakage fixed; training-only parameter learning functions added.
- [x] Naive Persistence and 7-Day Trailing Moving Average baseline models implemented.
- [x] Empirical evaluation performed and honest comparative metrics documented.
- [x] 11 new automated regression, leakage, and AppTest integration tests added (total 56 passed).
- [x] Test coverage improved to 96%.
- [x] Streamlit app verified running cleanly with comparison table and non-leaky baselines.
- [x] Comprehensive documentation provided in `docs/PHASE_1A_REPORT.md`.

---

**PHASE 1A COMPLETE — READY FOR REVIEW**
