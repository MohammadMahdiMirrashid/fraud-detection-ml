"""
Unified training API for all fraud detection models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

from src.models.isolation_forest import train_isolation_forest, save_model as save_if_model
from src.models.autoencoder import train_autoencoder, save_model as save_ae_model
from src.models.gradient_boosting import (
    train_lightgbm, train_xgboost, train_catboost,
    save_model as save_gbm_model
)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    model_type: str = 'lightgbm',
    model_params: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None,
    **kwargs
):
    """
    Train a fraud detection model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Optional validation features
        y_val: Optional validation labels
        model_type: 'isolation_forest', 'autoencoder', 'lightgbm', 'xgboost', or 'catboost'
        model_params: Model-specific parameters
        output_path: Path to save trained model
        **kwargs: Additional model-specific arguments
    
    Returns:
        Trained model
    """
    print(f"Training {model_type} model...")
    
    if model_type == 'isolation_forest':
        model = train_isolation_forest(
            X_train,
            contamination=kwargs.get('contamination', 0.01),
            n_estimators=kwargs.get('n_estimators', 100),
            random_state=kwargs.get('random_state', 42)
        )
        if output_path:
            save_if_model(model, output_path)
    
    elif model_type == 'autoencoder':
        autoencoder, encoder, scaler = train_autoencoder(
            X_train,
            X_val=X_val,
            encoding_dim=kwargs.get('encoding_dim', 32),
            hidden_layers=kwargs.get('hidden_layers', [64, 32]),
            epochs=kwargs.get('epochs', 50),
            batch_size=kwargs.get('batch_size', 256),
            random_state=kwargs.get('random_state', 42)
        )
        model = autoencoder  # Return autoencoder for predictions
        if output_path:
            save_ae_model(model, output_path)
    
    elif model_type == 'lightgbm':
        model = train_lightgbm(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            params=model_params,
            num_boost_round=kwargs.get('num_boost_round', 1000),
            early_stopping_rounds=kwargs.get('early_stopping_rounds', 50),
            random_state=kwargs.get('random_state', 42)
        )
        if output_path:
            save_gbm_model(model, output_path, model_type='lightgbm')
    
    elif model_type == 'xgboost':
        model = train_xgboost(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            params=model_params,
            num_boost_round=kwargs.get('num_boost_round', 1000),
            early_stopping_rounds=kwargs.get('early_stopping_rounds', 50),
            random_state=kwargs.get('random_state', 42)
        )
        if output_path:
            save_gbm_model(model, output_path, model_type='xgboost')
    
    elif model_type == 'catboost':
        model = train_catboost(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            params=model_params,
            iterations=kwargs.get('iterations', 1000),
            early_stopping_rounds=kwargs.get('early_stopping_rounds', 50),
            random_state=kwargs.get('random_state', 42)
        )
        if output_path:
            save_gbm_model(model, output_path, model_type='catboost')
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    print(f"✓ {model_type} model trained successfully")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--data", type=str, default="data/processed/features.csv",
                       help="Path to feature matrix")
    parser.add_argument("--model", type=str, default="lightgbm",
                       choices=["isolation_forest", "autoencoder", "lightgbm", "xgboost", "catboost"],
                       help="Model type")
    parser.add_argument("--output", type=str, default="models/best_model.pkl",
                       help="Output model path")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Test set size")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    # Separate features and target
    X = df.drop('fraud', axis=1)
    y = df['fraud']
    
    # Remove metadata columns if present
    metadata_cols = ['transaction_id', 'customer_id', 'timestamp']
    X = X.drop(columns=[col for col in metadata_cols if col in X.columns])
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    
    # Further split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=args.seed
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Fraud rate - Train: {y_train.mean():.2%}, Val: {y_val.mean():.2%}, Test: {y_test.mean():.2%}")
    
    # Train model
    model = train_model(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        model_type=args.model,
        output_path=Path(args.output),
        random_state=args.seed
    )
    
    print(f"\nModel saved to {args.output}")

