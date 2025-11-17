"""
Isolation Forest for unsupervised fraud detection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Tuple, Optional


def train_isolation_forest(
    X_train: pd.DataFrame,
    contamination: float = 0.01,
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1
) -> IsolationForest:
    """
    Train Isolation Forest model for anomaly detection.
    
    Args:
        X_train: Training features
        contamination: Expected proportion of outliers (default 1%)
        n_estimators: Number of trees
        random_state: Random seed
        n_jobs: Number of parallel jobs
    
    Returns:
        Trained IsolationForest model
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    model.fit(X_train_scaled)
    
    # Store scaler in model for prediction
    model.scaler_ = scaler
    
    return model


def predict_anomaly_scores(
    model: IsolationForest,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Predict anomaly scores (lower = more anomalous).
    
    Args:
        model: Trained IsolationForest model
        X: Features
    
    Returns:
        Anomaly scores (negative = anomaly, positive = normal)
    """
    X_scaled = model.scaler_.transform(X)
    scores = model.score_samples(X_scaled)
    return scores


def predict_fraud_probability(
    model: IsolationForest,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Convert anomaly scores to fraud probabilities.
    
    Args:
        model: Trained IsolationForest model
        X: Features
    
    Returns:
        Fraud probabilities (0-1, higher = more likely fraud)
    """
    scores = predict_anomaly_scores(model, X)
    
    # Convert scores to probabilities using min-max scaling and inversion
    # Lower scores (more anomalous) -> higher fraud probability
    min_score = scores.min()
    max_score = scores.max()
    
    if max_score == min_score:
        return np.ones(len(scores)) * 0.5
    
    # Normalize and invert
    normalized = (scores - min_score) / (max_score - min_score)
    probabilities = 1 - normalized  # Invert so lower scores = higher fraud prob
    
    return probabilities


def save_model(model: IsolationForest, path: Path):
    """Save trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: Path) -> IsolationForest:
    """Load trained model from disk."""
    model = joblib.load(path)
    return model

