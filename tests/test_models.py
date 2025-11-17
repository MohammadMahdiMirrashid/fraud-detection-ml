"""
Tests for model modules.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.isolation_forest import train_isolation_forest, predict_fraud_probability
from src.models.gradient_boosting import train_lightgbm


@pytest.fixture
def sample_data():
    """Create sample training data."""
    n = 1000
    X = pd.DataFrame({
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.randn(n)
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.99, 0.01]))
    return X, y


def test_isolation_forest_train(sample_data):
    """Test Isolation Forest training."""
    X, y = sample_data
    model = train_isolation_forest(X, contamination=0.01, random_state=42)
    
    assert model is not None
    assert hasattr(model, 'scaler_')


def test_isolation_forest_predict(sample_data):
    """Test Isolation Forest prediction."""
    X, y = sample_data
    model = train_isolation_forest(X, contamination=0.01, random_state=42)
    proba = predict_fraud_probability(model, X)
    
    assert len(proba) == len(X)
    assert all(0 <= p <= 1 for p in proba)


@pytest.mark.skipif(
    not pytest.importorskip("lightgbm", reason="LightGBM not installed"),
    reason="LightGBM not available"
)
def test_lightgbm_train(sample_data):
    """Test LightGBM training."""
    X, y = sample_data
    try:
        model = train_lightgbm(
            X, y,
            num_boost_round=10,
            verbose_eval=0
        )
        assert model is not None
    except ImportError:
        pytest.skip("LightGBM not installed")

