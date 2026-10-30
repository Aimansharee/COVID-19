"""
Tests for data loading functions and error handling.
"""

import os
import pytest
import pandas as pd
from main import load_data, load_vaccine_data


def test_load_data_default_file(covid_csv_path):
    """Verify that default COVID dataset loads with correct columns and dtypes."""
    assert os.path.exists(covid_csv_path), "COVID CSV file does not exist!"
    df = load_data(covid_csv_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "date" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert "new_cases" in df.columns
    assert "location" in df.columns
    assert df["new_cases"].dtype == "float32"


def test_load_vaccine_data_default_file(vaccine_csv_path):
    """Verify that default vaccine dataset loads with 2021+ dates and cleaned countries."""
    assert os.path.exists(vaccine_csv_path), "Vaccine CSV file does not exist!"
    vdf = load_vaccine_data(vaccine_csv_path)
    assert isinstance(vdf, pd.DataFrame)
    assert not vdf.empty
    assert "country" in vdf.columns
    assert "total_vaccinations" in vdf.columns
    assert (vdf["date"].dt.year >= 2021).all()


def test_load_vaccine_data_missing_file():
    """Verify that load_vaccine_data gracefully returns None on missing file."""
    result = load_vaccine_data("non_existent_vaccine_file_12345.csv")
    assert result is None


def test_load_data_custom_path(tmp_path):
    """Verify load_data works on custom temporary CSV file."""
    temp_csv = tmp_path / "mini_covid.csv"
    temp_csv.write_text(
        "location,date,new_cases,total_cases,population,stringency_index\n"
        "TestLand,2021-01-01,10.0,100.0,1000.0,20.0\n"
    )
    df = load_data(str(temp_csv))
    assert len(df) == 1
    assert df.iloc[0]["location"] == "TestLand"
    assert df.iloc[0]["new_cases"] == 10.0
