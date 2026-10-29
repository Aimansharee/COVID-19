"""
Automated tests for data leakage prevention, temporal forecasting integrity,
training-only preprocessing isolation, and baseline forecasting models.
"""

import numpy as np
import pandas as pd
import pytest
from main import (
    preprocess_data,
    create_features,
    build_model,
    fit_outlier_bounds,
    apply_outlier_bounds,
    evaluate_predictions,
    NaivePersistenceModel,
    MovingAverageModel,
    predict_naive_persistence,
    predict_moving_average,
    check_date_continuity,
    reindex_daily_calendar
)


def test_no_current_target_leakage_rolling_avg_1():
    """
    Verify that rolling_avg_1 does NOT contain the current-day target new_cases[t].
    Previously, rolling(1).mean() on unshifted new_cases evaluated exactly to new_cases[t].
    In the corrected pipeline, rolling_avg_1 must equal new_cases[t-1] (i.e., lag_1).
    """
    df = pd.DataFrame({
        "location": ["TestCountry"] * 10,
        "date": pd.date_range("2021-01-01", periods=10, freq="D"),
        "new_cases": [10.0, 25.0, 40.0, 70.0, 110.0, 160.0, 220.0, 290.0, 370.0, 460.0],
        "total_cases": [10.0, 35.0, 75.0, 145.0, 255.0, 415.0, 635.0, 925.0, 1295.0, 1755.0],
        "population": [1000000.0] * 10,
        "stringency_index": [50.0] * 10
    })
    feat_df = create_features(df)
    
    for _, row in feat_df.iterrows():
        # rolling_avg_1 must NOT equal the current-day new_cases
        assert row["rolling_avg_1"] != row["new_cases"], (
            f"Target leakage detected! rolling_avg_1 ({row['rolling_avg_1']}) "
            f"equals contemporaneous new_cases ({row['new_cases']})"
        )
        # rolling_avg_1 must equal lag_1
        assert np.isclose(row["rolling_avg_1"], row["lag_1"])
        # target is log1p(new_cases[t]), which must not equal log1p(rolling_avg_1)
        assert not np.isclose(row["target"], np.log1p(row["rolling_avg_1"]))


def test_shifted_rolling_averages_use_only_past_observations():
    """
    Verify that rolling_avg_7 at date t is computed strictly from days t-7 to t-1,
    and does NOT incorporate the target day t.
    """
    new_cases_series = [float(i * 10) for i in range(15)]
    df = pd.DataFrame({
        "location": ["CountryA"] * 15,
        "date": pd.date_range("2021-01-01", periods=15, freq="D"),
        "new_cases": new_cases_series,
        "total_cases": np.cumsum(new_cases_series),
        "population": [500000.0] * 15,
        "stringency_index": [40.0] * 15
    })
    feat_df = create_features(df)
    
    # Inspect row at index 8 (corresponding to original day index 8, value 80.0)
    # History for 7 days strictly before day 8 are days 1..7: values 10, 20, 30, 40, 50, 60, 70
    expected_7day_mean = np.mean([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]) # 40.0
    row_8 = feat_df[feat_df["new_cases"] == 80.0].iloc[0]
    
    assert np.isclose(row_8["rolling_avg_7"], expected_7day_mean), (
        f"Expected rolling_avg_7 to be {expected_7day_mean}, got {row_8['rolling_avg_7']}"
    )
    # The contemporaneous value 80.0 was NOT included in rolling_avg_7
    unfiltered_leaky_mean = np.mean([20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]) # 50.0
    assert not np.isclose(row_8["rolling_avg_7"], unfiltered_leaky_mean)


def test_contemporaneous_cumulative_and_stringency_lagged():
    """
    Verify that total_cases and stringency_index at time t are lagged to t-1
    so forecast origin t-1 cannot see contemporaneous cumulative totals or policies.
    """
    dates = pd.date_range("2021-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "location": ["CountryX"] * 10,
        "date": dates,
        "new_cases": [10.0 * (i + 1) for i in range(10)],
        "total_cases": [100.0 + i * 50.0 for i in range(10)],
        "population": [1000000.0] * 10,
        "stringency_index": [20.0 + i * 2.0 for i in range(10)]
    })
    feat_df = create_features(df)
    
    # At row date 2021-01-08 (index 7), total_cases must equal index 6's original total_cases (400.0)
    target_row = feat_df[feat_df["date"] == pd.Timestamp("2021-01-08")].iloc[0]
    assert np.isclose(target_row["total_cases"], 100.0 + 6 * 50.0)
    assert np.isclose(target_row["stringency_index"], 20.0 + 6 * 2.0)


def test_future_target_mutation_leakage_invariance():
    """
    Strongest leakage invariance test:
    Modifying future target values in a time series MUST NOT alter historical
    feature representations or historical predictions in any way.
    """
    dates = pd.date_range("2021-01-01", periods=25, freq="D")
    base_cases = [float(50 + i * 5) for i in range(25)]
    
    df_clean = pd.DataFrame({
        "location": ["InvariantLand"] * 25,
        "date": dates,
        "new_cases": base_cases.copy(),
        "total_cases": np.cumsum(base_cases),
        "population": [2000000.0] * 25,
        "stringency_index": [55.0] * 25
    })
    
    feat_clean = create_features(df_clean)
    
    # Historical cutoff at index 15
    hist_cutoff_date = dates[15]
    hist_features_clean = feat_clean[feat_clean["date"] <= hist_cutoff_date][
        ['lag_1', 'lag_7', 'rolling_avg_1', 'rolling_avg_7', 'total_cases', 'stringency_index']
    ].copy()
    
    # Mutate future targets drastically (from day 18 onwards)
    df_mutated = df_clean.copy(deep=True)
    df_mutated.loc[18:, "new_cases"] = df_mutated.loc[18:, "new_cases"] + 50000.0
    df_mutated.loc[18:, "total_cases"] = df_mutated.loc[18:, "total_cases"] + 500000.0
    
    feat_mutated = create_features(df_mutated)
    hist_features_mutated = feat_mutated[feat_mutated["date"] <= hist_cutoff_date][
        ['lag_1', 'lag_7', 'rolling_avg_1', 'rolling_avg_7', 'total_cases', 'stringency_index']
    ].copy()
    
    # Assert historical feature representations are exactly identical
    pd.testing.assert_frame_equal(hist_features_clean, hist_features_mutated)


def test_training_only_preprocessing_outlier_bounds():
    """
    Verify that outlier preprocessing bounds (q_low, q_high) are fitted exclusively
    on training data and applied to test partitions without recalculation on test data.
    """
    train_dates = pd.date_range("2021-01-01", periods=40, freq="D")
    test_dates = pd.date_range("2021-02-10", periods=20, freq="D")
    
    # Training set has moderate variation: 100 to 200
    train_df = pd.DataFrame({
        "location": ["TestLand"] * 40,
        "date": train_dates,
        "new_cases": np.linspace(100.0, 200.0, 40)
    })
    
    # Test set contains a massive surge: 500 to 2000
    test_df = pd.DataFrame({
        "location": ["TestLand"] * 20,
        "date": test_dates,
        "new_cases": np.linspace(500.0, 2000.0, 20)
    })
    
    # Fit bounds strictly on train
    q_low, q_high = fit_outlier_bounds(train_df, lower_quantile=0.05, upper_quantile=0.95)
    assert 100.0 <= q_low <= 110.0
    assert 190.0 <= q_high <= 200.0
    
    # Applying training bounds to test data uses the learned thresholds
    filtered_test = apply_outlier_bounds(test_df, q_low, q_high)
    assert len(filtered_test) == 0  # All test values exceed train q_high
    
    # Verify preprocess_data respects passed bounds without refitting
    preprocessed_test = preprocess_data(test_df, filter_outliers=True, q_low=q_low, q_high=q_high)
    assert len(preprocessed_test) == 0


def test_chronological_split_isolation():
    """
    Verify that train/test split maintains strict temporal order with no future leakage.
    """
    dates = pd.date_range("2021-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "location": ["CountryA"] * 30,
        "date": dates,
        "new_cases": [float(i + 10) for i in range(30)],
        "total_cases": [float(i * 100) for i in range(30)],
        "population": [100000.0] * 30,
        "stringency_index": [50.0] * 30
    })
    feat_df = create_features(df)
    
    split_idx = int(len(feat_df) * 0.75)
    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx:]
    
    assert train_df["date"].max() < test_df["date"].min(), (
        f"Temporal overlap! Train max date ({train_df['date'].max()}) "
        f"is not strictly before test min date ({test_df['date'].min()})"
    )


def test_naive_persistence_baseline_behavior():
    """
    Verify Naive Persistence Baseline predicts next-day cases using y_t = y_{t-1}.
    """
    model = NaivePersistenceModel()
    
    # Test with lag_1 DataFrame feature
    X_test = pd.DataFrame({"lag_1": [150.0, 220.0, 310.0, 280.0]})
    preds = model.predict(X_test)
    np.testing.assert_array_equal(preds, [150.0, 220.0, 310.0, 280.0])
    
    # Functional helper
    preds_func = predict_naive_persistence(X_test)
    np.testing.assert_array_equal(preds_func, [150.0, 220.0, 310.0, 280.0])
    
    # Fit fallback
    model_fallback = NaivePersistenceModel().fit(y=[100.0, 110.0, 125.0])
    dummy_X = pd.DataFrame({"other_col": [1, 2, 3]})
    np.testing.assert_array_equal(model_fallback.predict(dummy_X), [125.0, 125.0, 125.0])


def test_moving_average_baseline_behavior():
    """
    Verify Moving Average Baseline predicts using trailing past rolling average.
    """
    model = MovingAverageModel(window=7)
    
    X_test = pd.DataFrame({"rolling_avg_7": [105.0, 112.5, 120.0]})
    preds = model.predict(X_test)
    np.testing.assert_array_equal(preds, [105.0, 112.5, 120.0])
    
    # Functional helper
    preds_func = predict_moving_average(X_test, window=7)
    np.testing.assert_array_equal(preds_func, [105.0, 112.5, 120.0])


def test_comparable_model_evaluation_periods():
    """
    Verify that Gradient Boosting, Naive Persistence, and 7-Day Moving Average
    are evaluated on identical test observations, length, and date indices.
    """
    dates = pd.date_range("2021-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "location": ["TestLand"] * 30,
        "date": dates,
        "new_cases": [float(50 + i * 2) for i in range(30)],
        "total_cases": [float(i * 100) for i in range(30)],
        "population": [100000.0] * 30,
        "stringency_index": [50.0] * 30
    })
    feat_df = create_features(df)
    features = ['lag_1', 'lag_7', 'rolling_avg_1', 'rolling_avg_7',
                'total_cases', 'population', 'stringency_index']
    
    split_idx = int(len(feat_df) * 0.7)
    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx:]
    
    X_train, y_train = train_df[features], train_df['target']
    X_test, y_test = test_df[features], test_df['target']
    y_test_actual = np.expm1(y_test)
    
    # Train GBR
    gbr = build_model(n_estimators=10, max_depth=3, random_state=42)
    gbr.fit(X_train, y_train)
    y_pred_gbr = np.expm1(gbr.predict(X_test))
    
    # Baselines
    y_pred_naive = predict_naive_persistence(X_test)
    y_pred_ma = predict_moving_average(X_test, window=7)
    
    # Verify same lengths
    assert len(y_pred_gbr) == len(test_df)
    assert len(y_pred_naive) == len(test_df)
    assert len(y_pred_ma) == len(test_df)
    
    # Verify evaluate_predictions produces valid metrics for all three
    metrics_gbr = evaluate_predictions(y_test_actual, y_pred_gbr)
    metrics_naive = evaluate_predictions(y_test_actual, y_pred_naive)
    metrics_ma = evaluate_predictions(y_test_actual, y_pred_ma)
    
    for m in [metrics_gbr, metrics_naive, metrics_ma]:
        assert not np.isnan(m["MAE"])
        assert not np.isnan(m["RMSE"])
        assert m["MAE"] >= 0
        assert m["RMSE"] >= m["MAE"] - 1e-9


def test_safe_handling_of_zero_and_missing_cases():
    """
    Verify robust handling of missing dates/cases and zeroes without NaN propagation.
    """
    df = pd.DataFrame({
        "location": ["ZeroLand"] * 15,
        "date": pd.date_range("2021-01-01", periods=15, freq="D"),
        "new_cases": [None, 0.0, 0.0, 0.0, None, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 5.0, 0.0, 10.0, 0.0],
        "total_cases": [0.0] * 5 + [1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 8.0, 8.0, 18.0, 18.0],
        "population": [50000.0] * 15,
        "stringency_index": [30.0] * 15
    })
    
    clean_df = preprocess_data(df, filter_outliers=False)
    feat_df = create_features(clean_df)
    
    assert feat_df["new_cases"].isnull().sum() == 0
    assert (feat_df["new_cases"] >= 0).all()
    assert (feat_df["target"] >= 0).all()
    assert not np.isinf(feat_df["target"]).any()


def test_chronological_sorting_on_unsorted_and_interleaved_records():
    """
    Phase C: Verify that records shuffled in arbitrary chronological order and
    interleaved across multiple countries are cleanly sorted chronologically
    per country before lag and rolling calculations, with zero cross-country leakage.
    """
    # Create interleaved and shuffled records for CountryA and CountryB (15 days each)
    dates_A = pd.date_range("2021-01-01", periods=15, freq="D")
    dates_B = pd.date_range("2021-01-01", periods=15, freq="D")
    
    rows = []
    for i in range(15):
        rows.append({"location": "CountryA", "date": dates_A[i], "new_cases": float((i + 1) * 10), "total_cases": 100.0 + i * 10.0})
        rows.append({"location": "CountryB", "date": dates_B[i], "new_cases": float((i + 1) * 100), "total_cases": 1000.0 + i * 100.0})
        
    df = pd.DataFrame(rows)
    # Shuffle randomly
    shuffled_df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    feat_df = create_features(shuffled_df)
    
    # Check CountryA
    a_feats = feat_df[feat_df["location"] == "CountryA"].sort_values("date")
    assert a_feats["date"].is_monotonic_increasing
    # At day 9 of CountryA (2021-01-09, new_cases = 90), lag_1 must be 80.0 (day 8), NOT anything from CountryB
    row_a_day9 = a_feats[a_feats["date"] == pd.Timestamp("2021-01-09")].iloc[0]
    assert row_a_day9["lag_1"] == 80.0
    assert row_a_day9["rolling_avg_1"] == 80.0
    assert row_a_day9["lag_7"] == 20.0
    
    # Check CountryB
    b_feats = feat_df[feat_df["location"] == "CountryB"].sort_values("date")
    assert b_feats["date"].is_monotonic_increasing
    row_b_day9 = b_feats[b_feats["date"] == pd.Timestamp("2021-01-09")].iloc[0]
    assert row_b_day9["lag_1"] == 800.0
    assert row_b_day9["rolling_avg_1"] == 800.0
    assert row_b_day9["lag_7"] == 200.0


def test_missing_dates_detection_and_reindexing():
    """
    Phase C: Verify that missing calendar dates within a country time-series
    are detected by check_date_continuity and successfully reindexed to daily continuity
    by reindex_daily_calendar so that shift(1) strictly represents day t-1.
    """
    # Country with missing dates: 2021-01-01, 2021-01-02, GAP (Jan 3-4), 2021-01-05 .. 2021-01-15
    dates = [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-02")] + list(pd.date_range("2021-01-05", periods=10, freq="D"))
    cases = [50.0, 60.0] + [float(100 + i * 10) for i in range(10)]
    df = pd.DataFrame({
        "location": ["GapLand"] * len(dates),
        "date": dates,
        "new_cases": cases,
        "total_cases": np.cumsum(cases),
        "population": [500000.0] * len(dates),
        "stringency_index": [40.0] * len(dates)
    })
    
    # Continuity check flags gap
    is_continuous, gap_count = check_date_continuity(df)
    assert not is_continuous
    assert gap_count == 1
    
    # Reindex fills missing calendar days (Jan 3 and Jan 4) with zero cases
    reindexed_df = reindex_daily_calendar(df)
    is_cont_after, gap_after = check_date_continuity(reindexed_df)
    assert is_cont_after
    assert gap_after == 0
    assert len(reindexed_df) == 14  # Jan 1 through Jan 14
    
    # Jan 3 and Jan 4 have 0 new cases and forward filled cumulative
    jan_3 = reindexed_df[reindexed_df["date"] == pd.Timestamp("2021-01-03")].iloc[0]
    assert jan_3["new_cases"] == 0.0
    assert jan_3["total_cases"] == 110.0
    
    # In feature generation, verify date sequence is preserved
    feat_df = create_features(reindexed_df)
    assert len(feat_df) > 0
    # On day Jan 9, lag_1 is Jan 8 cases, lag_7 is Jan 2 cases (60.0)
    jan_9_feat = feat_df[feat_df["date"] == pd.Timestamp("2021-01-09")].iloc[0]
    assert jan_9_feat["lag_1"] == 130.0
    assert jan_9_feat["lag_7"] == 60.0


def test_forecast_origin_isolation_procedure():
    """
    Phase D: Realistic Forecast-Origin Isolation Test.
    For a historical cutoff date T:
    1. Fit preprocessing and train model using only historical data through T.
    2. Generate 1-step ahead forecast for T+1.
    3. Take an alternative run where all future data (after T+1) is removed or mutated drastically.
    4. Repeat the procedure.
    5. Verify that forecast for T+1 remains 100% identical within numerical tolerance (1e-6).
    """
    dates = pd.date_range("2021-01-01", periods=30, freq="D")
    base_cases = [float(20 + i * 3) for i in range(30)]
    
    full_df = pd.DataFrame({
        "location": ["IsolationLand"] * 30,
        "date": dates,
        "new_cases": base_cases,
        "total_cases": np.cumsum(base_cases),
        "population": [1000000.0] * 30,
        "stringency_index": [50.0] * 30
    })
    
    # Cutoff date T is day 20 (2021-01-21)
    cutoff_T = dates[20]
    target_date = dates[21]  # T+1
    
    # --- RUN 1: Full dataset with future points existing through day 30 ---
    feat_df_1 = create_features(preprocess_data(full_df, filter_outliers=False))
    train_1 = feat_df_1[feat_df_1["date"] <= cutoff_T]
    test_1 = feat_df_1[feat_df_1["date"] == target_date]
    
    features = ['lag_1', 'lag_7', 'rolling_avg_1', 'rolling_avg_7', 'total_cases', 'population', 'stringency_index']
    
    model_1 = build_model(n_estimators=15, max_depth=3, random_state=42)
    model_1.fit(train_1[features], train_1['target'])
    pred_1 = np.expm1(model_1.predict(test_1[features]))[0]
    
    # --- RUN 2: Dataset physically truncated to only data available at forecast time (T+1) ---
    # In addition, mutate any future values if present to simulate extreme shock
    truncated_df = full_df[full_df["date"] <= target_date].copy(deep=True)
    feat_df_2 = create_features(preprocess_data(truncated_df, filter_outliers=False))
    train_2 = feat_df_2[feat_df_2["date"] <= cutoff_T]
    test_2 = feat_df_2[feat_df_2["date"] == target_date]
    
    model_2 = build_model(n_estimators=15, max_depth=3, random_state=42)
    model_2.fit(train_2[features], train_2['target'])
    pred_2 = np.expm1(model_2.predict(test_2[features]))[0]
    
    # Numerical tolerance check
    assert np.isclose(pred_1, pred_2, atol=1e-6, rtol=1e-6), (
        f"Forecast origin isolation violation! Run 1 pred: {pred_1}, Run 2 pred: {pred_2}"
    )


def test_outbreak_peak_preservation():
    """
    Phase C: Verify that authentic epidemiological peak waves (outbreak surges)
    are preserved and NOT truncated or flattened when filter_outliers=False is used.
    """
    dates = pd.date_range("2021-01-01", periods=20, freq="D")
    # Base cases around 100 with an authentic explosive surge up to 5,000
    cases = [100.0] * 10 + [500.0, 1500.0, 3000.0, 5000.0, 4000.0, 2000.0, 800.0, 300.0, 150.0, 100.0]
    df = pd.DataFrame({
        "location": ["SurgeLand"] * 20,
        "date": dates,
        "new_cases": cases,
        "total_cases": np.cumsum(cases),
        "population": [1000000.0] * 20,
        "stringency_index": [50.0] * 20
    })
    
    clean_df = preprocess_data(df, filter_outliers=False)
    feat_df = create_features(clean_df)
    
    # The true maximum peak (5000.0) must be fully retained in the series
    assert feat_df["new_cases"].max() == 5000.0
    assert len(clean_df) == 20

