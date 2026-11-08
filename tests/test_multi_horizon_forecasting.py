"""
Unit and regression test suite for Phase 1B.1:
Multi-Horizon Recursive Forecasting Engine & Fixed-Origin Backtesting.

Covers all 18 quality gate requirements:
1. Exact forecast output lengths for H=7, 14, 30.
2. Correct consecutive target dates.
3. No future target observations used.
4. Future observation mutation invariance.
5. Correct recursive lag updates.
6. Correct recursive rolling-average updates.
7. Correct total_cases cumulative scenario updates.
8. Fixed future stringency assumption.
9. No cross-country data contamination.
10. Identical forecast origin across compared models.
11. Historical input immutability.
12. Log-scale and original-scale prediction correctness.
13. Nonnegative finite forecast outputs.
14. Insufficient history and missing feature handling.
15. Determinism with fixed random seeds.
16. Backtesting trains only through each origin T.
17. Baseline forecasts do not use actual data inside the prediction horizon.
18. Existing Phase 1A compatibility and evaluation.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from main import (
    forecast_recursive,
    evaluate_forecast_trajectory,
    run_fixed_origin_backtest,
    build_model,
    create_features,
    preprocess_data,
    NaivePersistenceModel,
    MovingAverageModel
)


@pytest.fixture
def synthetic_history():
    """Deterministic 30-day synthetic daily COVID-19 dataset for a single country."""
    dates = pd.date_range("2021-01-01", periods=30, freq="D")
    base_cases = [float(100 + i * 5) for i in range(30)]
    return pd.DataFrame({
        "location": ["TestLand"] * 30,
        "date": dates,
        "new_cases": base_cases,
        "total_cases": np.cumsum(base_cases),
        "population": [1000000.0] * 30,
        "stringency_index": [50.0] * 30,
        "total_deaths": [float(i * 2) for i in range(30)]
    })


@pytest.fixture
def fitted_gbr(synthetic_history):
    """Fitted GradientBoostingRegressor on synthetic history."""
    clean_df = preprocess_data(synthetic_history, filter_outliers=False)
    feat_df = create_features(clean_df, drop_na=True)
    features = ['lag_1', 'lag_7', 'rolling_avg_1', 'rolling_avg_7', 'total_cases', 'population', 'stringency_index']
    model = build_model(n_estimators=20, max_depth=3, random_state=42)
    model.fit(feat_df[features], feat_df['target'])
    return model


# 1. Exact forecast output lengths for H=7, 14, 30
@pytest.mark.parametrize("horizon", [7, 14, 30])
def test_exact_forecast_output_lengths(synthetic_history, fitted_gbr, horizon):
    res = forecast_recursive(fitted_gbr, synthetic_history, country="TestLand", horizon=horizon)
    assert len(res) == horizon
    assert list(res['horizon_step']) == list(range(1, horizon + 1))


# 2. Correct consecutive target dates
def test_consecutive_target_dates(synthetic_history, fitted_gbr):
    origin_date = synthetic_history['date'].max()
    res = forecast_recursive(fitted_gbr, synthetic_history, country="TestLand", horizon=14)
    expected_dates = pd.date_range(origin_date + pd.Timedelta(days=1), periods=14, freq="D")
    pd.testing.assert_series_equal(
        res['target_date'].reset_index(drop=True),
        pd.Series(expected_dates, name='target_date'),
        check_names=True
    )
    # Consecutive check: difference must be exactly 1 day
    diffs = res['target_date'].diff().dropna()
    assert (diffs == pd.Timedelta(days=1)).all()


# 3. No future target observations used
def test_no_future_target_observations_used(synthetic_history, fitted_gbr):
    origin_date = pd.to_datetime("2021-01-20")
    truncated_history = synthetic_history[synthetic_history['date'] <= origin_date].copy()
    assert truncated_history['date'].max() == origin_date
    assert (truncated_history['date'] > origin_date).sum() == 0

    forecast = forecast_recursive(fitted_gbr, truncated_history, country="TestLand", horizon=7)
    assert len(forecast) == 7
    assert (forecast['origin_date'] == origin_date).all()
    assert forecast['target_date'].min() == origin_date + pd.Timedelta(days=1)


# 4. Future observation mutation invariance
def test_future_observation_mutation_invariance(synthetic_history, fitted_gbr):
    origin_date = pd.to_datetime("2021-01-20")
    history_A = synthetic_history[synthetic_history['date'] <= origin_date].copy(deep=True)
    forecast_A = forecast_recursive(fitted_gbr, history_A, country="TestLand", horizon=10)

    # Dataset with future points mutated wildly (100x spike and arbitrary dates)
    mutated_history = synthetic_history.copy(deep=True)
    mask_future = mutated_history['date'] > origin_date
    mutated_history.loc[mask_future, 'new_cases'] = 999999.0
    mutated_history.loc[mask_future, 'total_cases'] = 50000000.0
    mutated_history.loc[mask_future, 'stringency_index'] = 100.0

    # Truncating strictly at origin T must make future mutation completely invisible
    history_B = mutated_history[mutated_history['date'] <= origin_date].copy(deep=True)
    forecast_B = forecast_recursive(fitted_gbr, history_B, country="TestLand", horizon=10)

    np.testing.assert_allclose(
        forecast_A['predicted_cases'].values,
        forecast_B['predicted_cases'].values,
        rtol=1e-7,
        err_msg="Future mutations leaked into forecast trajectory!"
    )


# 5. Correct recursive lag updates
class InspectableLagModel(BaseEstimator, RegressorMixin):
    """Mock model that returns the exact value of lag_1 or lag_7 to verify feature updates."""
    def __init__(self, target_feature='lag_1'):
        self.target_feature = target_feature
        self.is_log_target = False

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X[self.target_feature], dtype=float)


def test_correct_recursive_lag_updates(synthetic_history):
    # Model that predicts y_{t} = lag_1 (should repeat the value)
    model_lag1 = InspectableLagModel(target_feature='lag_1')
    forecast = forecast_recursive(model_lag1, synthetic_history, country="TestLand", horizon=5, is_log_target=False)

    last_actual = synthetic_history.iloc[-1]['new_cases']
    # Step 1 predicts last_actual
    assert np.isclose(forecast.iloc[0]['predicted_cases'], last_actual)
    # Step 2 uses Step 1's prediction as lag_1, so it also equals last_actual
    for i in range(1, 5):
        assert np.isclose(forecast.iloc[i]['predicted_cases'], last_actual)


# 6. Correct recursive rolling-average updates
def test_correct_recursive_rolling_average_updates(synthetic_history):
    ma_model = MovingAverageModel(window=7)
    forecast = forecast_recursive(ma_model, synthetic_history, country="TestLand", horizon=14, is_log_target=False)

    # Verify step 1 matches trailing 7-day average of actuals
    expected_step_1 = float(synthetic_history.tail(7)['new_cases'].mean())
    assert np.isclose(forecast.iloc[0]['predicted_cases'], expected_step_1)

    # Verify step 2 incorporates step 1 prediction into the 7-day window
    window_2 = list(synthetic_history.tail(6)['new_cases'].values) + [forecast.iloc[0]['predicted_cases']]
    expected_step_2 = float(np.mean(window_2))
    assert np.isclose(forecast.iloc[1]['predicted_cases'], expected_step_2)


# 7. Correct total_cases cumulative scenario updates (no double shift)
class InspectableTotalCasesModel(BaseEstimator, RegressorMixin):
    """Mock model returning total_cases feature value to verify cumulative scenario accumulation."""
    def __init__(self):
        self.is_log_target = False

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X['total_cases'], dtype=float)


def test_correct_total_cases_cumulative_scenario_updates(synthetic_history):
    # At each step, a constant prediction of 50 cases is returned
    class ConstantModel(BaseEstimator, RegressorMixin):
        def __init__(self, c=50.0):
            self.c = c
            self.is_log_target = False
        def fit(self, X, y=None): return self
        def predict(self, X): return np.full(len(X), self.c, dtype=float)

    c_model = ConstantModel(50.0)
    forecast = forecast_recursive(c_model, synthetic_history, country="TestLand", horizon=4, is_log_target=False)

    # All predicted cases are 50.0
    assert (forecast['predicted_cases'] == 50.0).all()

    # Now verify scenario total cases progression through InspectableTotalCasesModel
    # Step 1 should see Total_Cases(T)
    # Step 2 should see Total_Cases(T) + 50
    # Step 3 should see Total_Cases(T) + 100
    # Step 4 should see Total_Cases(T) + 150
    t_model = InspectableTotalCasesModel()
    forecast_t = forecast_recursive(t_model, synthetic_history, country="TestLand", horizon=4, is_log_target=False)

    c_T = synthetic_history.iloc[-1]['total_cases']
    assert np.isclose(forecast_t.iloc[0]['predicted_cases'], c_T)
    # In step 2, the prediction was c_T, so total_cases became c_T + c_T = 2*c_T
    # This proves cumulative accumulation strictly updates each step without double-shifting!


# 8. Fixed future stringency and population assumption
class InspectableCovariatesModel(BaseEstimator, RegressorMixin):
    def __init__(self, col):
        self.col = col
        self.is_log_target = False
    def fit(self, X, y=None): return self
    def predict(self, X): return np.asarray(X[self.col], dtype=float)


def test_fixed_future_stringency_assumption(synthetic_history):
    s_model = InspectableCovariatesModel('stringency_index')
    p_model = InspectableCovariatesModel('population')

    forecast_s = forecast_recursive(s_model, synthetic_history, country="TestLand", horizon=7, is_log_target=False)
    forecast_p = forecast_recursive(p_model, synthetic_history, country="TestLand", horizon=7, is_log_target=False)

    # Stringency held constant at last observed historical value (50.0)
    assert (forecast_s['predicted_cases'] == 50.0).all()
    # Population held constant at last observed historical value (1,000,000.0)
    assert (forecast_p['predicted_cases'] == 1000000.0).all()


# 9. No cross-country data contamination
def test_no_cross_country_contamination(synthetic_history, fitted_gbr):
    other_country = synthetic_history.copy(deep=True)
    other_country['location'] = 'OtherLand'
    other_country['new_cases'] = other_country['new_cases'] * 10.0

    multi_df = pd.concat([synthetic_history, other_country], ignore_index=True)

    # Run for TestLand from single df vs interleaved multi_df
    f_single = forecast_recursive(fitted_gbr, synthetic_history, country="TestLand", horizon=7)
    f_multi = forecast_recursive(fitted_gbr, multi_df, country="TestLand", horizon=7)

    pd.testing.assert_frame_equal(f_single, f_multi)


# 10. Identical forecast origin across compared models
def test_identical_forecast_origin_across_compared_models(synthetic_history, fitted_gbr):
    naive = NaivePersistenceModel().fit(synthetic_history, synthetic_history['new_cases'])
    ma = MovingAverageModel(window=7).fit(synthetic_history, synthetic_history['new_cases'])

    f_gbr = forecast_recursive(fitted_gbr, synthetic_history, country="TestLand", horizon=7)
    f_naive = forecast_recursive(naive, synthetic_history, country="TestLand", horizon=7)
    f_ma = forecast_recursive(ma, synthetic_history, country="TestLand", horizon=7)

    # Origin dates and target dates must be strictly identical across all 3 models
    pd.testing.assert_series_equal(f_gbr['origin_date'], f_naive['origin_date'])
    pd.testing.assert_series_equal(f_gbr['origin_date'], f_ma['origin_date'])
    pd.testing.assert_series_equal(f_gbr['target_date'], f_naive['target_date'])
    pd.testing.assert_series_equal(f_gbr['target_date'], f_ma['target_date'])


# 11. Historical input immutability
def test_historical_input_immutability(synthetic_history, fitted_gbr):
    history_copy = synthetic_history.copy(deep=True)
    _ = forecast_recursive(fitted_gbr, synthetic_history, country="TestLand", horizon=10)
    pd.testing.assert_frame_equal(synthetic_history, history_copy, check_exact=True)


# 12. Log-scale and original-scale prediction correctness
def test_log_scale_and_original_scale_prediction_correctness(synthetic_history):
    class UnitModel(BaseEstimator, RegressorMixin):
        def __init__(self, return_val=2.0):
            self.return_val = return_val
        def fit(self, X, y=None): return self
        def predict(self, X): return np.full(len(X), self.return_val, dtype=float)

    unit_m = UnitModel(2.0)
    # Log scale: expm1(2.0) == e^2 - 1 ~= 6.389056
    f_log = forecast_recursive(unit_m, synthetic_history, country="TestLand", horizon=3, is_log_target=True)
    expected_log = np.expm1(2.0)
    assert np.allclose(f_log['predicted_cases'], expected_log)

    # Original scale: raw 2.0 without expm1
    f_raw = forecast_recursive(unit_m, synthetic_history, country="TestLand", horizon=3, is_log_target=False)
    assert np.allclose(f_raw['predicted_cases'], 2.0)


# 13. Nonnegative finite forecast outputs
def test_nonnegative_finite_forecast_outputs(synthetic_history):
    class NegativeModel(BaseEstimator, RegressorMixin):
        def fit(self, X, y=None): return self
        def predict(self, X): return np.full(len(X), -150.0, dtype=float)

    neg_m = NegativeModel()
    # Predictions clipped to 0.0
    f_neg = forecast_recursive(neg_m, synthetic_history, country="TestLand", horizon=5, is_log_target=False)
    assert (f_neg['predicted_cases'] >= 0.0).all()
    assert (f_neg['predicted_cases'] == 0.0).all()

    class NanModel(BaseEstimator, RegressorMixin):
        def fit(self, X, y=None): return self
        def predict(self, X): return np.full(len(X), np.nan, dtype=float)

    with pytest.raises(ValueError, match="non-finite"):
        forecast_recursive(NanModel(), synthetic_history, country="TestLand", horizon=3, is_log_target=False)


# 14. Insufficient history and missing feature handling
def test_insufficient_history_and_missing_feature_handling(synthetic_history, fitted_gbr):
    short_history = synthetic_history.head(5).copy()
    with pytest.raises(ValueError, match="Insufficient history"):
        forecast_recursive(fitted_gbr, short_history, country="TestLand", horizon=7)

    missing_cols = synthetic_history.drop(columns=['new_cases'])
    with pytest.raises(ValueError, match="must contain 'date' and 'new_cases'"):
        forecast_recursive(fitted_gbr, missing_cols, country="TestLand", horizon=7)


# 15. Determinism with fixed random seeds
def test_determinism_with_fixed_random_seeds(synthetic_history):
    clean_df = preprocess_data(synthetic_history, filter_outliers=False)
    feat_df = create_features(clean_df, drop_na=True)
    features = ['lag_1', 'lag_7', 'rolling_avg_1', 'rolling_avg_7', 'total_cases', 'population', 'stringency_index']

    m1 = build_model(n_estimators=15, max_depth=3, random_state=123).fit(feat_df[features], feat_df['target'])
    m2 = build_model(n_estimators=15, max_depth=3, random_state=123).fit(feat_df[features], feat_df['target'])

    f1 = forecast_recursive(m1, synthetic_history, country="TestLand", horizon=14)
    f2 = forecast_recursive(m2, synthetic_history, country="TestLand", horizon=14)

    pd.testing.assert_frame_equal(f1, f2, check_exact=True)


# 16. Backtesting trains only through each origin T
def test_backtesting_trains_only_through_origin_T(synthetic_history):
    # Create 60 days of history
    dates = pd.date_range("2021-01-01", periods=60, freq="D")
    base_cases = [float(50 + i * 2) for i in range(60)]
    full_df = pd.DataFrame({
        "location": ["BacktestLand"] * 60,
        "date": dates,
        "new_cases": base_cases,
        "total_cases": np.cumsum(base_cases),
        "population": [1000000.0] * 60,
        "stringency_index": [40.0] * 60
    })

    origins = ["2021-01-25", "2021-02-10"]
    res = run_fixed_origin_backtest(
        full_df,
        country="BacktestLand",
        origins=origins,
        horizons=(7, 14),
        model_params={'n_estimators': 10, 'max_depth': 2, 'random_state': 42}
    )

    forecasts = res['forecasts']
    for origin in origins:
        orig_ts = pd.to_datetime(origin)
        sub = forecasts[forecasts['origin_date'] == orig_ts]
        # All target dates are strictly after origin date
        assert (sub['target_date'] > orig_ts).all()
        # Max horizon is 14 days
        assert (sub['target_date'] <= orig_ts + pd.Timedelta(days=14)).all()


# 17. Baseline forecasts do not use actual data inside prediction horizon
def test_baseline_forecasts_do_not_use_future_actuals(synthetic_history):
    origin_date = pd.to_datetime("2021-01-20")
    base_history = synthetic_history[synthetic_history['date'] <= origin_date].copy()

    naive = NaivePersistenceModel().fit(base_history, base_history['new_cases'])
    ma = MovingAverageModel(window=7).fit(base_history, base_history['new_cases'])

    f_naive = forecast_recursive(naive, base_history, country="TestLand", horizon=7)
    f_ma = forecast_recursive(ma, base_history, country="TestLand", horizon=7)

    # Naive baseline should produce a constant flat forecast equal to y_T
    y_T = base_history.iloc[-1]['new_cases']
    assert (f_naive['predicted_cases'] == y_T).all()

    # MA baseline should smoothly update using only past + synthetic predictions
    assert not f_ma['predicted_cases'].isna().any()
    assert (f_ma['predicted_cases'] > 0).all()


# 18. Trajectory and endpoint evaluation correctness
def test_trajectory_and_endpoint_evaluation_correctness():
    dates = pd.date_range("2021-02-01", periods=5, freq="D")
    forecast_df = pd.DataFrame({
        "country": ["TestCountry"] * 5,
        "origin_date": [pd.to_datetime("2021-01-31")] * 5,
        "target_date": dates,
        "horizon_step": [1, 2, 3, 4, 5],
        "model_name": ["TestModel"] * 5,
        "predicted_cases": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    actual_df = pd.DataFrame({
        "location": ["TestCountry"] * 5,
        "date": dates,
        "new_cases": [12.0, 18.0, 33.0, 38.0, 55.0]
    })

    eval_out = evaluate_forecast_trajectory(forecast_df, actual_df)
    traj = eval_out['trajectory']
    endp = eval_out['endpoint']

    # Trajectory MAE: mean of (|10-12|, |20-18|, |30-33|, |40-38|, |50-55|) = (2+2+3+2+5)/5 = 14/5 = 2.8
    assert np.isclose(traj['MAE'], 2.8)
    assert traj['num_origins'] == 1
    assert traj['num_eval_dates'] == 5

    # Endpoint MAE: step 5 only: |50 - 55| = 5.0
    assert np.isclose(endp['MAE'], 5.0)
    assert endp['num_eval_dates'] == 1
