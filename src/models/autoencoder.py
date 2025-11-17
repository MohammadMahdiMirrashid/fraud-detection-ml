"""
Autoencoder for unsupervised fraud detection.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from pathlib import Path
from typing import Tuple, Optional


def build_autoencoder(
    input_dim: int,
    encoding_dim: int = 32,
    hidden_layers: list = [64, 32],
    activation: str = 'relu',
    dropout_rate: float = 0.2
) -> keras.Model:
    """
    Build autoencoder model.
    
    Args:
        input_dim: Input feature dimension
        encoding_dim: Dimension of encoding layer
        hidden_layers: List of hidden layer sizes
        activation: Activation function
        dropout_rate: Dropout rate
    
    Returns:
        Compiled autoencoder model
    """
    # Input
    input_layer = layers.Input(shape=(input_dim,))
    
    # Encoder
    x = input_layer
    for hidden_size in hidden_layers:
        x = layers.Dense(hidden_size, activation=activation)(x)
        x = layers.Dropout(dropout_rate)(x)
    
    encoded = layers.Dense(encoding_dim, activation=activation, name='encoding')(x)
    
    # Decoder
    x = encoded
    for hidden_size in reversed(hidden_layers):
        x = layers.Dense(hidden_size, activation=activation)(x)
        x = layers.Dropout(dropout_rate)(x)
    
    decoded = layers.Dense(input_dim, activation='linear', name='decoded')(x)
    
    # Autoencoder
    autoencoder = keras.Model(input_layer, decoded)
    
    # Encoder (for feature extraction)
    encoder = keras.Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return autoencoder, encoder


def train_autoencoder(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
    encoding_dim: int = 32,
    hidden_layers: list = [64, 32],
    epochs: int = 50,
    batch_size: int = 256,
    validation_split: float = 0.2,
    verbose: int = 1,
    random_state: int = 42
) -> Tuple[keras.Model, keras.Model, StandardScaler]:
    """
    Train autoencoder model.
    
    Args:
        X_train: Training features
        X_val: Optional validation features
        encoding_dim: Encoding dimension
        hidden_layers: Hidden layer sizes
        epochs: Training epochs
        batch_size: Batch size
        validation_split: Validation split if X_val not provided
        verbose: Verbosity level
        random_state: Random seed
    
    Returns:
        Tuple of (autoencoder, encoder, scaler)
    """
    # Set random seeds
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Build model
    input_dim = X_train_scaled.shape[1]
    autoencoder, encoder = build_autoencoder(
        input_dim=input_dim,
        encoding_dim=encoding_dim,
        hidden_layers=hidden_layers
    )
    
    # Prepare validation data
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        validation_data = (X_val_scaled, X_val_scaled)
        validation_split = None
    else:
        validation_data = None
    
    # Train
    history = autoencoder.fit(
        X_train_scaled,
        X_train_scaled,  # Autoencoder reconstructs input
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=True
    )
    
    # Store scaler in model
    autoencoder.scaler_ = scaler
    encoder.scaler_ = scaler
    
    return autoencoder, encoder, scaler


def predict_reconstruction_error(
    model: keras.Model,
    X: pd.DataFrame,
    scaler: Optional[StandardScaler] = None
) -> np.ndarray:
    """
    Predict reconstruction error (MSE) for each sample.
    
    Args:
        model: Trained autoencoder
        X: Features
        scaler: Optional scaler (if not stored in model)
    
    Returns:
        Reconstruction errors (higher = more anomalous)
    """
    if scaler is None:
        scaler = model.scaler_
    
    X_scaled = scaler.transform(X)
    X_reconstructed = model.predict(X_scaled, verbose=0)
    
    # Calculate MSE per sample
    mse = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
    
    return mse


def predict_fraud_probability(
    model: keras.Model,
    X: pd.DataFrame,
    scaler: Optional[StandardScaler] = None
) -> np.ndarray:
    """
    Convert reconstruction error to fraud probabilities.
    
    Args:
        model: Trained autoencoder
        X: Features
        scaler: Optional scaler
    
    Returns:
        Fraud probabilities (0-1, higher = more likely fraud)
    """
    errors = predict_reconstruction_error(model, X, scaler)
    
    # Convert errors to probabilities using min-max scaling
    min_error = errors.min()
    max_error = errors.max()
    
    if max_error == min_error:
        return np.ones(len(errors)) * 0.5
    
    probabilities = (errors - min_error) / (max_error - min_error)
    
    return probabilities


def save_model(model: keras.Model, path: Path):
    """Save trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    print(f"Model saved to {path}")


def load_model(path: Path) -> keras.Model:
    """Load trained model from disk."""
    model = keras.models.load_model(str(path))
    return model

