"""
Tests for model evaluation metrics: MAE, RMSE, R2, Adjusted SMAPE, and accuracy.
"""

import numpy as np
import pytest
from main import adjusted_smape, evaluate_predictions


def test_adjusted_smape_perfect_prediction():
    """Verify that identical actual and predicted yield SMAPE of 0.0."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])
    score = adjusted_smape(y_true, y_pred)
    assert np.isclose(score, 0.0)


def test_adjusted_smape_empty_mask_threshold():
    """Verify threshold mask returns NaN when all values are beneath threshold."""
    y_true = np.array([0.1, 0.2])
    y_pred = np.array([0.1, 0.2])
    score = adjusted_smape(y_true, y_pred, threshold=1.0)
    assert np.isnan(score)


def test_adjusted_smape_known_calculation():
    """Verify calculation against known manual formula."""
    y_true = np.array([10.0])
    y_pred = np.array([20.0])
    # denom = 30.0 > 1.0; 2 * |20-10| / 30 = 20/30 = 2/3 = 0.66667; 100 * 2/3 = 66.6667
    score = adjusted_smape(y_true, y_pred)
    assert np.isclose(score, 200.0 / 3.0)


def test_evaluate_predictions_metrics():
    """Verify evaluate_predictions returns complete dictionary with valid numerical scores."""
    y_true = np.array([50.0, 100.0, 150.0, 200.0])
    y_pred = np.array([55.0, 95.0, 160.0, 190.0])
    
    metrics = evaluate_predictions(y_true, y_pred)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert "SMAPE" in metrics
    assert "Accuracy" in metrics
    
    assert metrics["MAE"] == pytest.approx(7.5, 0.1)
    assert metrics["RMSE"] >= metrics["MAE"]
    assert 0.0 <= metrics["R2"] <= 1.0
    assert metrics["Accuracy"] == pytest.approx(100.0 - metrics["SMAPE"])
