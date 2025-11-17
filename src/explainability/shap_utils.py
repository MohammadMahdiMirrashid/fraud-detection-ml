"""
SHAP utilities for model explainability.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import Optional, List
from pathlib import Path


def explain_model_shap(
    model,
    X: pd.DataFrame,
    model_type: str = 'lightgbm',
    sample_size: Optional[int] = 1000,
    background_size: Optional[int] = 100
) -> Tuple[shap.Explainer, np.ndarray]:
    """
    Generate SHAP explanations for a model.
    
    Args:
        model: Trained model
        X: Feature matrix
        model_type: 'lightgbm', 'xgboost', 'catboost', or 'tree'
        sample_size: Number of samples to explain (None = all)
        background_size: Size of background dataset for TreeExplainer
    
    Returns:
        Tuple of (explainer, shap_values)
    """
    # Sample data if needed
    if sample_size is not None and len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X
    
    # Create background dataset
    if background_size is not None and len(X) > background_size:
        X_background = X.sample(n=background_size, random_state=42)
    else:
        X_background = X
    
    # Create explainer based on model type
    if model_type in ['lightgbm', 'xgboost', 'catboost']:
        explainer = shap.TreeExplainer(model, X_background)
    else:
        # For other models, use KernelExplainer
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1],
            X_background
        )
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # Handle binary classification (SHAP returns list for binary)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class
    
    return explainer, shap_values


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    max_display: int = 20,
    save_path: Optional[str] = None
):
    """
    Plot SHAP summary plot.
    
    Args:
        shap_values: SHAP values array
        X: Feature matrix
        feature_names: Optional feature names
        max_display: Maximum number of features to display
        save_path: Optional path to save plot
    """
    if feature_names is None:
        feature_names = X.columns.tolist()
    
    # Sample if too large
    if len(X) > 1000:
        sample_idx = np.random.choice(len(X), 1000, replace=False)
        shap_values_sample = shap_values[sample_idx]
        X_sample = X.iloc[sample_idx]
    else:
        shap_values_sample = shap_values
        X_sample = X
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_sample,
        X_sample,
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP summary plot saved to {save_path}")
    else:
        plt.show()


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_display: int = 20,
    save_path: Optional[str] = None
):
    """
    Plot SHAP bar plot (mean absolute SHAP values).
    
    Args:
        shap_values: SHAP values array
        feature_names: Optional feature names
        max_display: Maximum number of features to display
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(10, 8))
    shap.plots.bar(
        shap_values,
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP bar plot saved to {save_path}")
    else:
        plt.show()


def explain_single_prediction(
    explainer: shap.Explainer,
    X_instance: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Generate SHAP explanation for a single prediction.
    
    Args:
        explainer: SHAP explainer
        X_instance: Single instance to explain
        feature_names: Optional feature names
        save_path: Optional path to save plot
    """
    shap_values = explainer.shap_values(X_instance)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
            data=X_instance.iloc[0].values,
            feature_names=feature_names if feature_names else X_instance.columns.tolist()
        ),
        show=False
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP explanation saved to {save_path}")
    else:
        plt.show()


def get_feature_importance_shap(
    shap_values: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Get feature importance from SHAP values.
    
    Args:
        shap_values: SHAP values array
        feature_names: Feature names
    
    Returns:
        DataFrame with feature importance
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    return importance

