"""
Tests for data simulation module.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.simulate_data import TransactionSimulator


def test_transaction_simulator_init():
    """Test TransactionSimulator initialization."""
    simulator = TransactionSimulator(
        n_customers=100,
        n_transactions=1000,
        fraud_rate=0.01,
        random_state=42
    )
    assert simulator.n_customers == 100
    assert simulator.n_transactions == 1000
    assert simulator.fraud_rate == 0.01


def test_generate_base_transactions():
    """Test base transaction generation."""
    simulator = TransactionSimulator(
        n_customers=100,
        n_transactions=1000,
        random_state=42
    )
    df = simulator.generate_base_transactions()
    
    assert len(df) == 1000
    assert 'transaction_id' in df.columns
    assert 'customer_id' in df.columns
    assert 'amount' in df.columns
    assert 'fraud' in df.columns
    assert df['fraud'].sum() == 0  # No fraud in base transactions


def test_inject_fraud_patterns():
    """Test fraud pattern injection."""
    simulator = TransactionSimulator(
        n_customers=100,
        n_transactions=1000,
        fraud_rate=0.01,
        random_state=42
    )
    df_base = simulator.generate_base_transactions()
    df = simulator.inject_fraud_patterns(df_base)
    
    assert len(df) == 1000
    assert df['fraud'].sum() > 0
    fraud_rate = df['fraud'].mean()
    assert 0.005 <= fraud_rate <= 0.02  # Allow some variance


def test_generate_complete():
    """Test complete generation pipeline."""
    simulator = TransactionSimulator(
        n_customers=100,
        n_transactions=1000,
        fraud_rate=0.01,
        random_state=42
    )
    df = simulator.generate()
    
    assert len(df) == 1000
    assert 'fraud' in df.columns
    assert df['fraud'].dtype in [int, bool]

