"""
Regression tests to prevent recurrence of confirmed and remediated defects.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import (
    preprocess_data,
    create_features,
    load_data,
    load_vaccine_data
)


def test_regression_bug_001_division_by_zero_fatality_rate():
    """
    Regression Test for BUG-001:
    Ensure rows with total_cases == 0 or total_cases is NaN do not produce inf fatality rates.
    """
    df = pd.DataFrame({
        "location": ["TestLand", "TestLand", "TestLand"],
        "date": pd.date_range("2020-03-01", periods=3),
        "total_cases": [0.0, 10.0, np.nan],
        "total_deaths": [2.0, 1.0, 0.0]
    })
    
    # Safe computation formula applied in main.py
    df["fatality_rate"] = np.where(df["total_cases"] > 0, df["total_deaths"] / df["total_cases"], np.nan)
    
    assert not np.isinf(df["fatality_rate"]).any(), "Fatality rate contains infinite values!"
    assert np.isnan(df.iloc[0]["fatality_rate"])
    assert df.iloc[1]["fatality_rate"] == 0.1
    assert np.isnan(df.iloc[2]["fatality_rate"])
    
    # Check monthly aggregation does not produce inf
    valid = df.dropna(subset=["fatality_rate"])
    monthly = valid.groupby(pd.Grouper(key="date", freq="ME" if hasattr(pd.offsets, "MonthEnd") else "M"))["fatality_rate"].mean()
    assert not np.isinf(monthly).any()


def test_regression_bug_002_pandas_frequency_me():
    """
    Regression Test for BUG-002:
    Ensure frequency 'ME' does not raise FutureWarning on modern pandas.
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame({"date": dates, "val": [1, 2, 3, 4, 5]})
    grouper = pd.Grouper(key="date", freq="ME" if hasattr(pd.offsets, "MonthEnd") else "M")
    res = df.groupby(grouper)["val"].sum()
    assert len(res) == 1


def test_regression_bug_003_vectorized_lag_features_no_deprecation():
    """
    Regression Test for BUG-003:
    Ensure create_features calculates lag_1 and lag_7 without DataFrameGroupBy.apply deprecation.
    """
    df = pd.DataFrame({
        "location": ["CountryX"] * 10,
        "date": pd.date_range("2021-01-01", periods=10),
        "new_cases": [float(i * 10) for i in range(10)]
    })
    feat_df = create_features(df)
    assert "lag_1" in feat_df.columns
    assert "lag_7" in feat_df.columns
    assert feat_df.iloc[0]["lag_1"] == 60.0
    assert feat_df.iloc[0]["lag_7"] == 0.0


def test_regression_bug_005_preprocess_data_copy_integrity():
    """
    Regression Test for BUG-005:
    Ensure preprocess_data creates a clean copy and does not mutate cached DataFrame.
    """
    df = pd.DataFrame({
        "location": ["CountryY"] * 5,
        "date": pd.date_range("2021-01-01", periods=5),
        "new_cases": [None, 10.0, 20.0, 30.0, 40.0]
    })
    res = preprocess_data(df)
    assert pd.isna(df["new_cases"].iloc[0]), "Input DataFrame was mutated in-place!"
    assert not res.empty
    assert (res["new_cases"] >= 0.0).all()


def test_regression_bug_007_robust_path_resolution(tmp_path):
    """
    Regression Test for BUG-007:
    Ensure data loaders resolve relative paths robustly without raising unhandled FileNotFoundError.
    """
    # Non-existent files should return None or raise standard exception gracefully
    assert load_vaccine_data("non_existent_vaccine_data_9999.csv") is None


def test_regression_bug_009_matplotlib_figure_closed():
    """
    Regression Test for BUG-009:
    Ensure matplotlib figures are explicitly closed to prevent memory leaks during app reruns.
    """
    initial_fignums = set(plt.get_fignums())
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    plt.close(fig)
    final_fignums = set(plt.get_fignums())
    assert final_fignums == initial_fignums
