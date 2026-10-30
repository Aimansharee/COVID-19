"""
Integration smoke tests for the Streamlit application using native AppTest.
"""

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(scope="module")
def app_runner():
    """Initializes AppTest from main.py."""
    at = AppTest.from_file("main.py", default_timeout=30)
    at.run()
    return at


def test_app_startup(app_runner):
    """Verify application boots, sets title, and initializes sidebar controls."""
    assert not app_runner.exception, f"App crashed on startup with: {app_runner.exception}"
    assert len(app_runner.title) > 0
    assert "COVID-19" in app_runner.title[0].value


def test_app_sidebar_controls(app_runner):
    """Verify radio selector exists with expected operational modes."""
    assert len(app_runner.sidebar.radio) > 0
    modes = app_runner.sidebar.radio[0].options
    assert "Model Prediction" in modes
    assert "Exploratory Data Analysis" in modes


def test_app_switch_to_eda_mode():
    """Verify switching to EDA mode runs successfully without exceptions."""
    at = AppTest.from_file("main.py", default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value("Exploratory Data Analysis").run()
    assert not at.exception, f"EDA mode raised exception: {at.exception}"
    # Verify EDA header is rendered
    header_values = [h.value for h in at.header]
    assert "Exploratory Data Analysis (EDA)" in header_values


def test_app_model_training_flow():
    """Verify clicking Train Model in Streamlit trains model, evaluates baselines, and displays results without errors."""
    at = AppTest.from_file("main.py", default_timeout=60)
    at.run()
    # Set n_estimators slider to 100 for fast test execution
    at.sidebar.slider[1].set_value(100).run()
    at.sidebar.button[0].click().run()
    assert not at.exception, f"Train Model triggered exception: {at.exception}"
    assert len(at.success) > 0
    assert "Model training completed!" in at.success[0].value
