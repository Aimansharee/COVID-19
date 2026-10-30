"""
Tests for syntax, module imports, and required dependencies.
"""

import sys
import importlib
import pytest


def test_python_version():
    """Verify that runtime Python version is at least 3.9."""
    assert sys.version_info >= (3, 9), f"Python version too old: {sys.version}"


@pytest.mark.parametrize("package_name", [
    "streamlit",
    "pandas",
    "numpy",
    "sklearn",
    "scipy",
    "plotly",
    "matplotlib",
])
def test_required_packages_installed(package_name):
    """Verify each core package can be imported without error."""
    mod = importlib.import_module(package_name)
    assert mod is not None


def test_main_module_import():
    """Verify main module imports cleanly and exposes public API."""
    import main
    expected_attrs = [
        "load_data",
        "load_vaccine_data",
        "preprocess_data",
        "create_features",
        "build_model",
        "adjusted_smape",
        "evaluate_predictions",
        "main",
    ]
    for attr in expected_attrs:
        assert hasattr(main, attr), f"main module is missing expected attribute: {attr}"
