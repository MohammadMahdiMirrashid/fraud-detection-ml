"""
Gradient Boosting models (LightGBM, XGBoost, CatBoost) for fraud detection.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import joblib
from pathlib import Path

# Try importing each library
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 100,
    random_state: int = 42
) -> lgb.Booster:
    """
    Train LightGBM model for fraud detection.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Optional validation features
        y_val: Optional validation labels
        params: Model parameters
        num_boost_round: Number of boosting rounds
        early_stopping_rounds: Early stopping rounds
        verbose_eval: Verbosity
        random_state: Random seed
    
    Returns:
        Trained LightGBM model
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
    
    # Default parameters optimized for imbalanced classification
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': random_state,
            'is_unbalance': True,  # Handle class imbalance
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum()
        }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    
    valid_sets = [train_data]
    valid_names = ['train']
    
    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        valid_sets.append(val_data)
        valid_names.append('valid')
    
    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(verbose_eval) if verbose_eval > 0 else lambda x: None
        ]
    )
    
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 100,
    random_state: int = 42
) -> xgb.Booster:
    """
    Train XGBoost model for fraud detection.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Optional validation features
        y_val: Optional validation labels
        params: Model parameters
        num_boost_round: Number of boosting rounds
        early_stopping_rounds: Early stopping rounds
        verbose_eval: Verbosity
        random_state: Random seed
    
    Returns:
        Trained XGBoost model
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    # Default parameters
    if params is None:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state,
            'scale_pos_weight': scale_pos_weight
        }
    
    # Create datasets
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    evals = [(dtrain, 'train')]
    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        evals.append((dval, 'val'))
    
    # Train
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval
    )
    
    return model


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
    iterations: int = 1000,
    early_stopping_rounds: int = 50,
    verbose: int = 100,
    random_state: int = 42
) -> cb.CatBoostClassifier:
    """
    Train CatBoost model for fraud detection.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Optional validation features
        y_val: Optional validation labels
        params: Model parameters
        iterations: Number of iterations
        early_stopping_rounds: Early stopping rounds
        verbose: Verbosity
        random_state: Random seed
    
    Returns:
        Trained CatBoost model
    """
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost not installed. Install with: pip install catboost")
    
    # Default parameters
    if params is None:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        params = {
            'iterations': iterations,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'Logloss',
            'random_seed': random_state,
            'scale_pos_weight': scale_pos_weight,
            'verbose': verbose,
            'early_stopping_rounds': early_stopping_rounds
        }
    
    # Create model
    model = cb.CatBoostClassifier(**params)
    
    # Train
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=verbose
        )
    else:
        model.fit(X_train, y_train, verbose=verbose)
    
    return model


def predict_proba_gbm(model, X: pd.DataFrame, model_type: str = 'lightgbm') -> np.ndarray:
    """
    Predict fraud probabilities from gradient boosting model.
    
    Args:
        model: Trained GBM model
        X: Features
        model_type: 'lightgbm', 'xgboost', or 'catboost'
    
    Returns:
        Fraud probabilities (shape: [n_samples, 2])
    """
    if model_type == 'lightgbm':
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")
        probs = model.predict(X, num_iteration=model.best_iteration)
        # LightGBM returns probabilities for class 1
        return np.column_stack([1 - probs, probs])
    
    elif model_type == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")
        dtest = xgb.DMatrix(X)
        probs = model.predict(dtest, ntree_limit=model.best_ntree_limit)
        return np.column_stack([1 - probs, probs])
    
    elif model_type == 'catboost':
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed")
        probs = model.predict_proba(X)
        return probs
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def save_model(model, path: Path, model_type: str = 'lightgbm'):
    """Save trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if model_type == 'lightgbm':
        model.save_model(str(path))
    elif model_type == 'xgboost':
        model.save_model(str(path))
    elif model_type == 'catboost':
        model.save_model(str(path))
    else:
        joblib.dump(model, path)
    
    print(f"Model saved to {path}")


def load_model(path: Path, model_type: str = 'lightgbm'):
    """Load trained model from disk."""
    if model_type == 'lightgbm':
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")
        model = lgb.Booster(model_file=str(path))
    elif model_type == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")
        model = xgb.Booster()
        model.load_model(str(path))
    elif model_type == 'catboost':
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed")
        model = cb.CatBoostClassifier()
        model.load_model(str(path))
    else:
        model = joblib.load(path)
    
    return model

