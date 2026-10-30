"""
Pytest configuration and shared fixtures for COVID-19 AI test suite.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure workspace root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


@pytest.fixture(scope="session")
def workspace_root():
    return BASE_DIR


@pytest.fixture(scope="session")
def covid_csv_path():
    return os.path.join(BASE_DIR, "cleaned_covid_data_last.csv")


@pytest.fixture(scope="session")
def vaccine_csv_path():
    return os.path.join(BASE_DIR, "country_vaccinations.csv")


@pytest.fixture
def sample_covid_df():
    """
    Creates a small, realistic synthetic COVID-19 dataframe for fast unit tests.
    """
    dates = pd.date_range("2021-01-01", periods=20, freq="D")
    data = []
    for loc in ["CountryA", "CountryB"]:
        base_cases = 100.0 if loc == "CountryA" else 200.0
        pop = 1000000.0 if loc == "CountryA" else 5000000.0
        for i, dt in enumerate(dates):
            new_cases = base_cases + i * 5.0 + (10.0 if i % 2 == 0 else -5.0)
            data.append({
                "location": loc,
                "date": dt,
                "total_cases": base_cases * 10 + i * 50,
                "new_cases": new_cases,
                "new_cases_smoothed": new_cases,
                "total_deaths": 10.0 + i,
                "new_deaths": 1.0,
                "population": pop,
                "case_fatality_rate": (10.0 + i) / (base_cases * 10 + i * 50),
                "death_rolling_avg": 1.0,
                "year": dt.year,
                "month": dt.month,
                "week": dt.isocalendar().week,
                "day_of_week": dt.dayofweek,
                "stringency_index": 50.0 + (i % 5)
            })
    return pd.DataFrame(data)


@pytest.fixture
def sample_vaccine_df():
    """
    Creates a small, realistic synthetic vaccination dataframe.
    """
    dates = pd.date_range("2021-01-01", periods=10, freq="D")
    data = []
    for c in ["CountryA", "CountryB"]:
        for i, dt in enumerate(dates):
            data.append({
                "country": c,
                "iso_code": "CTA" if c == "CountryA" else "CTB",
                "date": dt,
                "total_vaccinations": 10000.0 + i * 1000,
                "people_vaccinated": 8000.0 + i * 800,
                "people_fully_vaccinated": 2000.0 + i * 500,
                "daily_vaccinations_raw": 1000.0,
                "daily_vaccinations": 1000.0,
                "total_vaccinations_per_hundred": 1.0 + i * 0.1,
                "people_vaccinated_per_hundred": 0.8 + i * 0.08,
                "people_fully_vaccinated_per_hundred": 0.2 + i * 0.05,
                "daily_vaccinations_per_million": 100.0,
                "vaccines": "Oxford/AstraZeneca, Pfizer/BioNTech",
                "source_name": "Ministry of Health",
                "source_website": "https://example.com"
            })
    return pd.DataFrame(data)
