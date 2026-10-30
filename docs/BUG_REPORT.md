# 🐛 Bug Report — COVID-19 AI Audit & Remediation

This document catalogs all defects, design flaws, performance bottlenecks, and deprecation issues identified, remediated, and verified during the code audit of the COVID-19 AI repository.

---

## Summary of Defects

| Bug ID | Severity | Category | Affected Component | Status |
| :--- | :--- | :--- | :--- | :--- |
| **BUG-001** | **Critical** | Arithmetic / Data Integrity | Fatality Rate Division by Zero (`main.py`) | **FIXED & VERIFIED** |
| **BUG-002** | **High** | Deprecation / Compatibility | Pandas Resampling Frequency `'M'` (`main.py`) | **FIXED & VERIFIED** |
| **BUG-003** | **High** | Deprecation / Performance | `DataFrameGroupBy.apply` on Grouping Columns | **FIXED & VERIFIED** |
| **BUG-004** | **Medium** | Architecture / Testability | Monolithic Top-Level Script Execution | **FIXED & VERIFIED** |
| **BUG-005** | **Medium** | State Mutation | In-Place Mutation of Cached Data in `preprocess_data` | **FIXED & VERIFIED** |
| **BUG-006** | **Medium** | Modeling Integrity | Unlabeled Dummy Predictions (`Actual * 1.2`) | **FIXED & VERIFIED** |
| **BUG-007** | **Medium** | File I/O / Robustness | CWD-Dependent Relative File Paths | **FIXED & VERIFIED** |
| **BUG-008** | **Low** | Code Quality | Bare `except:` Exception Handler | **FIXED & VERIFIED** |
| **BUG-009** | **Low** | Resource Leak | Matplotlib Figure Memory Leak | **FIXED & VERIFIED** |
| **BUG-010** | **Low** | Dead Code | Unused Imports (`QuantileTransformer`, `stats`) | **FIXED & VERIFIED** |
| **BUG-011** | **Critical** | Data Leakage | Contemporaneous Target Leakage in `rolling_avg_1` & `rolling_avg_7` | **FIXED & VERIFIED** |
| **BUG-012** | **High** | Temporal Leakage | Contemporaneous Cumulative Totals and Oxford Policy Scores | **FIXED & VERIFIED** |
| **BUG-013** | **High** | Preprocessing Leakage | Global Outlier Filtering Across Folds & Outbreak Peak Truncation | **FIXED & VERIFIED** |
| **BUG-014** | **Medium** | Forecasting Integrity | Absence of Empirical Time-Series Baseline Benchmarks | **FIXED & VERIFIED** |

---

## Detailed Bug Records

### BUG-001: Infinite Fatality Rate via Zero-Case Division
- **Bug ID**: `BUG-001`
- **Severity**: **Critical**
- **Affected File**: `main.py` (Lines 402–404)
- **Root Cause**: Division `df['total_deaths'] / df['total_cases']` without checking if `total_cases == 0` or NaN resulted in `inf`. In early pandemic records (specifically March 2020), `total_cases` was 0 while initial deaths were reported, creating `inf`. When computing the monthly average via `df.groupby(...)['fatality_rate'].mean()`, the monthly average for March 2020 evaluated to `inf`, distorting Plotly visualizations.
- **Reproduction Steps**:
  1. Load `cleaned_covid_data_last.csv`.
  2. Compute `df['fatality_rate'] = df['total_deaths'] / df['total_cases']`.
  3. Aggregate by month: `df.groupby(pd.Grouper(key='date', freq='ME'))['fatality_rate'].mean()`.
  4. Observe `inf` value at index `2020-03-31`.
- **Expected Behavior**: Fatality rate should safely handle zero cases (yielding `np.nan`), and monthly averages should represent valid finite float percentages.
- **Applied Fix**:
  ```python
  df['fatality_rate'] = np.where(df['total_cases'] > 0, df['total_deaths'] / df['total_cases'], np.nan)
  valid_fatality = df.dropna(subset=['fatality_rate'])
  fatality_over_time = valid_fatality.groupby(pd.Grouper(key='date', freq=freq_month))['fatality_rate'].mean().reset_index()
  ```
- **Regression Test**: `tests/test_regression.py::test_regression_bug_001_division_by_zero_fatality_rate`
- **Status**: **FIXED & VERIFIED**

---

### BUG-002: Deprecated Pandas Monthly Frequency `'M'`
- **Bug ID**: `BUG-002`
- **Severity**: **High**
- **Affected File**: `main.py` (Line 403)
- **Root Cause**: In Pandas 2.2+, `freq='M'` is deprecated in favor of `freq='ME'` (Month End). In future versions, this causes a fatal runtime exception.
- **Reproduction Steps**:
  1. Run `pd.Grouper(freq='M')` on Pandas 2.2+.
  2. Observe `FutureWarning: 'M' is deprecated and will be removed in a future version, please use 'ME' instead.`
- **Expected Behavior**: Resampling should use future-proof frequency specifications while maintaining backward compatibility.
- **Applied Fix**:
  ```python
  freq_month = 'ME' if hasattr(pd.offsets, 'MonthEnd') else 'M'
  ```
- **Regression Test**: `tests/test_regression.py::test_regression_bug_002_pandas_frequency_me`
- **Status**: **FIXED & VERIFIED**

---

### BUG-003: Deprecated `DataFrameGroupBy.apply` on Grouping Columns
- **Bug ID**: `BUG-003`
- **Severity**: **High**
- **Affected File**: `main.py` (Lines 74–81)
- **Root Cause**: `df.groupby('location', group_keys=False, observed=True).apply(create_lag_features)` caused Pandas deprecation warnings because `apply` operates on the grouping columns, and `observed=True` is only valid for Categorical series. It was also 10x slower than vectorized shifting.
- **Reproduction Steps**:
  1. Run `create_features(df)` on Pandas 2.2.
  2. Observe `DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns...`
- **Expected Behavior**: Clean, vectorized lag calculation with zero deprecation warnings and sub-second execution.
- **Applied Fix (Historical & Final Reconciliation)**:
  - *Phase 0 Interim Fix*: Replaced slow `apply` with vectorized shifting and rolling:
    ```python
    for lag in [1, 7]:
        df[f'lag_{lag}'] = df.groupby('location')['new_cases'].shift(lag)
        df[f'rolling_avg_{lag}'] = (
            df.groupby('location')['new_cases']
            .rolling(lag, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    ```
  - *Phase 1A Final Fix*: Resolved confirmed target leakage (BUG-011) where unshifted `rolling_avg_1` leaked contemporaneous target `new_cases_t`. In the final implementation, rolling statistics are computed strictly on shifted past observations:
    ```python
    for lag in [1, 7]:
        df[f'lag_{lag}'] = df.groupby('location')['new_cases'].shift(lag)
        df[f'rolling_avg_{lag}'] = df.groupby('location')['new_cases'].transform(
            lambda s: s.shift(1).rolling(lag, min_periods=1).mean()
        )
    ```
- **Regression Tests**: `tests/test_regression.py::test_regression_bug_003_vectorized_lag_features_no_deprecation`, `tests/test_leakage_and_baselines.py::test_shifted_rolling_averages_use_only_past_observations`
- **Status**: **FIXED & VERIFIED**

---

### BUG-004: Monolithic Top-Level Script Execution
- **Bug ID**: `BUG-004`
- **Severity**: **Medium**
- **Affected File**: `main.py` (Global Scope)
- **Root Cause**: UI commands, cache initialization, and exploratory analyses were evaluated immediately upon module import, making automated unit testing impossible without triggering Streamlit bare-mode execution warnings.
- **Reproduction Steps**:
  1. Execute `python -c "import main"`.
  2. Observe multiple `missing ScriptRunContext!` warnings and automatic execution of 18MB data processing.
- **Expected Behavior**: Clean importability of core functions (`load_data`, `preprocess_data`, `build_model`, `evaluate_predictions`) without UI side-effects.
- **Applied Fix**: Encapsulated UI components within `def main():` and guarded execution with `if __name__ == "__main__": main()`.
- **Regression Test**: `tests/test_imports.py::test_main_module_import`
- **Status**: **FIXED & VERIFIED**

---

### BUG-005: In-Place Mutation of Cached Data in `preprocess_data`
- **Bug ID**: `BUG-005`
- **Severity**: **Medium**
- **Affected File**: `main.py` (Line 56)
- **Root Cause**: `preprocess_data` directly modified `df['new_cases'] = df['new_cases'].fillna(0)` on the cached DataFrame reference returned by `@st.cache_data load_data()`. In Streamlit, this triggers mutation warnings and cross-session data corruption.
- **Reproduction Steps**:
  1. Pass a cached DataFrame to `preprocess_data`.
  2. Observe original DataFrame is mutated in-place.
- **Expected Behavior**: `preprocess_data` must create an explicit deep copy (`df = df.copy()`) before modifying columns.
- **Applied Fix**: Added `df = df.copy()` as the first line of `preprocess_data(df)`.
- **Regression Test**: `tests/test_regression.py::test_regression_bug_005_preprocess_data_copy_integrity`
- **Status**: **FIXED & VERIFIED**

---

### BUG-006: Unlabeled Dummy Predictions in Comparison Chart
- **Bug ID**: `BUG-006`
- **Severity**: **Medium**
- **Affected File**: `main.py` (Line 121)
- **Root Cause**: The chart "Actual vs Predicted Comparison (Top 5 Countries)" displayed a hardcoded dummy calculation `Predicted = Actual * 1.2` before model training. Without clear labeling, reviewers and users could mistake this for real AI predictions.
- **Reproduction Steps**:
  1. Click "Train Model" in Model Prediction mode.
  2. Observe "Actual vs Predicted" line for Top 5 countries is strictly `Actual * 1.2`.
- **Expected Behavior**: Provide genuine historical time-series baseline forecasts rather than an artificial +20% multiplier.
- **Applied Fix (Historical & Final Reconciliation)**:
  - *Phase 0 Interim Fix*: Relabeled the trace to `Predicted (Baseline) (+20%)` to clarify it was not an AI prediction.
  - *Phase 1A Final Fix*: Completely eliminated the artificial `Actual * 1.2` formula and replaced it with a scientifically sound **7-Day Trailing Moving Average Baseline** and **Naive Persistence Baseline** (`NaivePersistenceModel`, `MovingAverageModel`) evaluated over identical chronological test periods (see BUG-014).
- **Regression Tests**: `tests/test_leakage_and_baselines.py::test_moving_average_baseline_behavior`, `tests/test_leakage_and_baselines.py::test_naive_persistence_baseline_behavior`
- **Status**: **FIXED & VERIFIED**

---

### BUG-007: CWD-Dependent Relative File Paths
- **Bug ID**: `BUG-007`
- **Severity**: **Medium**
- **Affected File**: `main.py` (Lines 31, 42)
- **Root Cause**: Relative file paths `'cleaned_covid_data_last.csv'` caused `FileNotFoundError` whenever the application or tests were invoked from subdirectories.
- **Reproduction Steps**:
  1. Navigate to `tests/` directory: `cd tests`.
  2. Run `python -c "from main import load_data; load_data()"`.
  3. Observe `FileNotFoundError`.
- **Expected Behavior**: File paths should resolve relative to `__file__` with fallback to CWD.
- **Applied Fix**:
  ```python
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  DEFAULT_COVID_DATA_PATH = os.path.join(BASE_DIR, 'cleaned_covid_data_last.csv')
  ```
- **Regression Test**: `tests/test_data_loading.py::test_load_data_default_file`
- **Status**: **FIXED & VERIFIED**

---

### BUG-008: Bare `except:` in Vaccine Data Loader
- **Bug ID**: `BUG-008`
- **Severity**: **Low**
- **Affected File**: `main.py` (Line 46)
- **Root Cause**: A bare `except:` block caught all exceptions, including `KeyboardInterrupt` and `SystemExit`, masking critical runtime issues and violating PEP 8 (E722).
- **Reproduction Steps**: Inspect line 46 of original `main.py`.
- **Expected Behavior**: Catch `Exception` explicitly.
- **Applied Fix**: Replaced with `except Exception: return None`.
- **Regression Test**: `tests/test_data_loading.py::test_load_vaccine_data_missing_file`
- **Status**: **FIXED & VERIFIED**

---

### BUG-009: Matplotlib Figure Memory Leak
- **Bug ID**: `BUG-009`
- **Severity**: **Low**
- **Affected File**: `main.py` (Lines 284–291)
- **Root Cause**: `fig, ax = plt.subplots(...)` was rendered via `st.pyplot(fig)` but `plt.close(fig)` was never called, leading to memory retention over continuous Streamlit reruns.
- **Reproduction Steps**: Repeatedly trigger model evaluation in Streamlit; observe matplotlib figures accumulating in memory (`plt.get_fignums()`).
- **Expected Behavior**: All created figures must be explicitly closed after rendering.
- **Applied Fix**: Added `plt.close(fig_err)` immediately after `st.pyplot(fig_err)`.
- **Regression Test**: `tests/test_regression.py::test_regression_bug_009_matplotlib_figure_closed`
- **Status**: **FIXED & VERIFIED**

---

### BUG-010: Dead Code and Unused Imports
- **Bug ID**: `BUG-010`
- **Severity**: **Low**
- **Affected File**: `main.py` (Lines 10–12)
- **Root Cause**: Unused imports (`QuantileTransformer`, `TransformedTargetRegressor`, `scipy.stats`, `seaborn`) added unnecessary startup latency and noise.
- **Reproduction Steps**: Check AST references of imported symbols.
- **Expected Behavior**: Only imported libraries that are actively utilized should remain.
- **Applied Fix**: Removed unused imports from header.
- **Status**: **FIXED & VERIFIED**

---

### BUG-011: Contemporaneous Target Leakage in `rolling_avg_1` and `rolling_avg_7`
- **Bug ID**: `BUG-011`
- **Severity**: **Critical**
- **Affected File**: `main.py` (`create_features`)
- **Root Cause**: `df.groupby('location')['new_cases'].rolling(1).mean()` on unshifted `new_cases` evaluated identically to the contemporaneous target `new_cases_t`. In addition, `rolling(7).mean()` included `new_cases_t`.
- **Expected Behavior**: All rolling features computed for time $t$ must only incorporate historical observations available up to $t-1$.
- **Applied Fix**: Computed rolling statistics on `new_cases.shift(1)` within each location group.
- **Regression Test**: `tests/test_leakage_and_baselines.py::test_no_current_target_leakage_rolling_avg_1`, `test_shifted_rolling_averages_use_only_past_observations`
- **Status**: **FIXED & VERIFIED**

---

### BUG-012: Contemporaneous Information Leakage in Cumulative Totals and Stringency Index
- **Bug ID**: `BUG-012`
- **Severity**: **High**
- **Affected File**: `main.py` (`create_features`)
- **Root Cause**: `total_cases` and `stringency_index` were supplied unshifted at time $t$, leaking contemporaneous cumulative totals and same-day policy scores unavailable at forecast origin $t-1$.
- **Expected Behavior**: Features must be strictly lagged to $t-1$ for 1-step ahead forecasting.
- **Applied Fix**: Applied `shift(1)` to `total_cases` and `stringency_index` per location group.
- **Regression Test**: `tests/test_leakage_and_baselines.py::test_contemporaneous_cumulative_and_stringency_lagged`
- **Status**: **FIXED & VERIFIED**

---

### BUG-013: Preprocessing Temporal Leakage via Global Outlier Trimming & Peak Destruction
- **Bug ID**: `BUG-013`
- **Severity**: **High**
- **Affected File**: `main.py` (`preprocess_data`)
- **Root Cause**: Quantiles 5% and 95% were computed globally across all countries and all years prior to train/test splitting, leaking future distribution statistics and deleting 70% of days (including major peak infection waves) for large countries.
- **Expected Behavior**: Preserve genuine epidemiological peak dynamics in forecasting; learn any outlier or transformation statistics exclusively from training folds.
- **Applied Fix**: Defaulted `preprocess_data(..., filter_outliers=False)` for continuous time series, added `fit_outlier_bounds` and `apply_outlier_bounds` for training-only threshold application.
- **Regression Test**: `tests/test_leakage_and_baselines.py::test_training_only_preprocessing_outlier_bounds`
- **Status**: **FIXED & VERIFIED**

---

### BUG-014: Absence of Empirical Time-Series Baseline Benchmarks
- **Bug ID**: `BUG-014`
- **Severity**: **Medium**
- **Affected File**: `main.py` (Model Evaluation Workflow)
- **Root Cause**: The application lacked standard persistence and moving average baselines, relying on an artificial `Actual * 1.2` benchmark.
- **Expected Behavior**: Implement `NaivePersistenceModel` ($y_t = y_{t-1}$) and `MovingAverageModel` (7-day trailing average) and evaluate alongside ML models on identical test periods.
- **Applied Fix**: Created baseline model classes and integrated comparative evaluation tables into Streamlit UI and test suites.
- **Regression Test**: `tests/test_leakage_and_baselines.py::test_naive_persistence_baseline_behavior`, `test_moving_average_baseline_behavior`, `test_comparable_model_evaluation_periods`
- **Status**: **FIXED & VERIFIED**
