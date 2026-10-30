"""
Tests for AI/ML Model instantiation and training pipeline.
"""

from sklearn.ensemble import GradientBoostingRegressor
from main import build_model, preprocess_data, create_features


def test_build_model_default_hyperparameters():
    """Verify build_model initializes GradientBoostingRegressor with expected params."""
    model = build_model(n_estimators=150, learning_rate=0.03, max_depth=4, random_state=42)
    assert isinstance(model, GradientBoostingRegressor)
    assert model.n_estimators == 150
    assert model.learning_rate == 0.03
    assert model.max_depth == 4
    assert model.loss == 'huber'
    assert model.random_state == 42


def test_model_training_smoke(sample_covid_df):
    """Smoke test for model training on sample synthetic data."""
    processed = preprocess_data(sample_covid_df)
    feat_df = create_features(processed)
    
    features = ['lag_1', 'lag_7', 'rolling_avg_1', 'rolling_avg_7',
                'total_cases', 'population', 'stringency_index']
    
    country_df = feat_df[feat_df['location'] == 'CountryA']
    X = country_df[features]
    y = country_df['target']
    
    # Train lightweight model
    model = build_model(n_estimators=10, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    
    assert hasattr(model, 'estimators_')
    assert len(model.estimators_) == 10
    assert hasattr(model, 'feature_importances_')
    assert len(model.feature_importances_) == len(features)
    assert (model.feature_importances_ >= 0).all()
