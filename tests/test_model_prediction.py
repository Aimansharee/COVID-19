"""
Tests for model prediction, inference output dimensions, and stability.
"""

import numpy as np
from main import build_model, preprocess_data, create_features


def test_model_prediction_single_and_batch(sample_covid_df):
    """Verify inference works on single sample and batch without NaNs."""
    processed = preprocess_data(sample_covid_df)
    feat_df = create_features(processed)
    
    features = ['lag_1', 'lag_7', 'rolling_avg_1', 'rolling_avg_7',
                'total_cases', 'population', 'stringency_index']
    
    country_df = feat_df[feat_df['location'] == 'CountryA']
    X = country_df[features]
    y = country_df['target']
    
    model = build_model(n_estimators=10, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    
    # Batch prediction
    y_pred_log = model.predict(X)
    assert len(y_pred_log) == len(X)
    assert not np.isnan(y_pred_log).any()
    assert not np.isinf(y_pred_log).any()
    
    # Expm1 inverse transformation
    y_pred_cases = np.expm1(y_pred_log)
    assert (y_pred_cases >= 0).all(), "Predictions on case counts must be non-negative!"
    
    # Single sample prediction
    single_x = X.iloc[[0]]
    single_pred = model.predict(single_x)
    assert single_pred.shape == (1,)
    assert np.isclose(single_pred[0], y_pred_log[0])
