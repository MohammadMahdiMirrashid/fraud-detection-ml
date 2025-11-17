"""
Tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample transaction data."""
    n = 1000
    dates = pd.date_range('2023-01-01', periods=n, freq='1H')
    return pd.DataFrame({
        'transaction_id': range(1, n+1),
        'customer_id': np.random.randint(1, 100, n),
        'timestamp': dates,
        'amount': np.random.lognormal(3, 1, n),
        'transaction_type': np.random.choice(['purchase', 'transfer'], n),
        'merchant_category': np.random.choice(['retail', 'groceries'], n),
        'country': np.random.choice(['US', 'CA'], n),
        'fraud': np.random.choice([0, 1], n, p=[0.99, 0.01])
    })


def test_feature_engineer_init():
    """Test FeatureEngineer initialization."""
    engineer = FeatureEngineer()
    assert engineer.feature_names == []


def test_create_time_features(sample_data):
    """Test time feature creation."""
    engineer = FeatureEngineer()
    df = engineer.create_time_features(sample_data)
    
    assert 'hour' in df.columns
    assert 'day_of_week' in df.columns
    assert 'month' in df.columns


def test_create_customer_aggregates(sample_data):
    """Test customer aggregate features."""
    engineer = FeatureEngineer()
    df = engineer.create_customer_aggregates(sample_data)
    
    assert 'customer_txn_count' in df.columns
    assert 'customer_total_amount' in df.columns
    assert 'customer_avg_amount' in df.columns


def test_engineer_features(sample_data):
    """Test complete feature engineering pipeline."""
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(sample_data)
    
    assert len(df_features.columns) > len(sample_data.columns)
    assert len(engineer.feature_names) > 0
    assert 'fraud' in df_features.columns

