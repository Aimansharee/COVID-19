"""
Tests for preprocessing and feature engineering pipelines.
"""

import numpy as np
import pandas as pd
from main import preprocess_data, create_features


def test_preprocess_data_missing_values():
    """Verify that preprocess_data fills missing values without crashing."""
    df = pd.DataFrame({
        "location": ["TestLand"] * 10,
        "date": pd.date_range("2021-01-01", periods=10),
        "new_cases": [None, 10.0, None, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    })
    res = preprocess_data(df)
    assert res["new_cases"].isnull().sum() == 0


def test_preprocess_data_does_not_mutate_original(sample_covid_df):
    """Verify that preprocess_data works on a copy and does not mutate input df."""
    original_copy = sample_covid_df.copy(deep=True)
    _ = preprocess_data(sample_covid_df)
    pd.testing.assert_frame_equal(sample_covid_df, original_copy)


def test_preprocess_data_outlier_filtering():
    """Verify that values outside 5% and 95% quantiles are filtered."""
    # Create 100 values with extreme outliers
    values = [10.0] * 90 + [0.0] * 5 + [100000.0] * 5
    df = pd.DataFrame({
        "location": ["TestLand"] * 100,
        "date": pd.date_range("2021-01-01", periods=100),
        "new_cases": values
    })
    res = preprocess_data(df)
    # The extreme upper values (100000.0) should be excluded
    assert res["new_cases"].max() < 100000.0
    assert len(res) <= len(df)


def test_create_features_output_columns(sample_covid_df):
    """Verify that create_features adds lags, rolling averages, and log1p target."""
    processed = preprocess_data(sample_covid_df)
    features_df = create_features(processed)
    
    expected_cols = ["lag_1", "lag_7", "rolling_avg_1", "rolling_avg_7", "target"]
    for col in expected_cols:
        assert col in features_df.columns, f"Missing engineered feature: {col}"
    
    assert features_df["lag_1"].isnull().sum() == 0
    assert features_df["lag_7"].isnull().sum() == 0
    assert (features_df["target"] >= 0).all()


def test_create_features_lag_correctness():
    """Verify lag_1 and lag_7 values accurately match previous records."""
    df = pd.DataFrame({
        "location": ["A"] * 10,
        "date": pd.date_range("2021-01-01", periods=10),
        "new_cases": [float(i) for i in range(10)]
    })
    res = create_features(df)
    # After dropping NaNs for lag_7, row at index 7 (new_cases = 7.0) has lag_1=6.0, lag_7=0.0
    row_7 = res[res["new_cases"] == 7.0].iloc[0]
    assert row_7["lag_1"] == 6.0
    assert row_7["lag_7"] == 0.0
    assert np.isclose(row_7["target"], np.log1p(7.0))
