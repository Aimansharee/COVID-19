"""
Tests for epidemiological dataset integrity, consistency, and sanity checks.
"""

import pandas as pd
import numpy as np


def test_covid_data_schema(sample_covid_df):
    """Verify expected columns exist in COVID dataset."""
    required_cols = [
        "location", "date", "total_cases", "new_cases",
        "total_deaths", "new_deaths", "population", "stringency_index"
    ]
    for col in required_cols:
        assert col in sample_covid_df.columns, f"Missing required column: {col}"


def test_covid_no_negative_cumulative_totals(sample_covid_df):
    """Verify cumulative totals like total_cases and total_deaths are non-negative."""
    assert (sample_covid_df["total_cases"] >= 0).all()
    assert (sample_covid_df["total_deaths"] >= 0).all()


def test_covid_population_positive(sample_covid_df):
    """Verify population values are strictly positive."""
    assert (sample_covid_df["population"] > 0).all()


def test_covid_uniqueness_per_location_date(sample_covid_df):
    """Verify there are no duplicate date entries for the same location."""
    duplicates = sample_covid_df.duplicated(subset=["location", "date"]).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate location-date pairs"


def test_vaccine_data_schema(sample_vaccine_df):
    """Verify expected columns exist in vaccine dataset."""
    required_cols = ["country", "date", "total_vaccinations"]
    for col in required_cols:
        assert col in sample_vaccine_df.columns, f"Missing required vaccine column: {col}"


def test_vaccine_date_order(sample_vaccine_df):
    """Verify dates are parseable and chronologically valid."""
    assert pd.api.types.is_datetime64_any_dtype(sample_vaccine_df["date"])
    for country, group in sample_vaccine_df.groupby("country"):
        assert group["date"].is_monotonic_increasing
