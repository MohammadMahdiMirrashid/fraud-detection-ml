"""
Tests for thresholding module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.thresholding import (
    calculate_cost_matrix,
    find_optimal_threshold,
    precision_at_k
)


@pytest.fixture
def sample_predictions():
    """Create sample predictions."""
    np.random.seed(42)
    y_true = np.random.choice([0, 1], 1000, p=[0.99, 0.01])
    y_proba = np.random.beta(2, 5, 1000)
    return y_true, y_proba


def test_calculate_cost_matrix(sample_predictions):
    """Test cost matrix calculation."""
    y_true, y_proba = sample_predictions
    y_pred = (y_proba >= 0.5).astype(int)
    
    cost_metrics = calculate_cost_matrix(
        y_true, y_pred,
        fraud_loss_cost=1000.0,
        false_positive_cost=10.0
    )
    
    assert 'total_cost' in cost_metrics
    assert 'tp' in cost_metrics
    assert 'fp' in cost_metrics
    assert 'fn' in cost_metrics
    assert 'tn' in cost_metrics


def test_find_optimal_threshold(sample_predictions):
    """Test optimal threshold finding."""
    y_true, y_proba = sample_predictions
    
    threshold, metrics = find_optimal_threshold(
        y_true, y_proba,
        fraud_loss_cost=1000.0,
        false_positive_cost=10.0
    )
    
    assert 0 <= threshold <= 1
    assert 'total_cost' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics


def test_precision_at_k(sample_predictions):
    """Test precision at K calculation."""
    y_true, y_proba = sample_predictions
    
    precision_k = precision_at_k(y_true, y_proba, k=100)
    
    assert 0 <= precision_k <= 1

