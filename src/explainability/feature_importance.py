"""
Feature importance utilities for fraud detection models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


def get_feature_importance_gbm(model, model_type: str = 'lightgbm') -> pd.DataFrame:
    """
    Extract feature importance from gradient boosting model.
    
    Args:
        model: Trained GBM model
        model_type: 'lightgbm', 'xgboost', or 'catboost'
    
    Returns:
        DataFrame with feature importance
    """
    if model_type == 'lightgbm':
        importance = model.feature_importance(importance_type='gain')
        feature_names = model.feature_name()
    elif model_type == 'xgboost':
        importance = model.get_score(importance_type='gain')
        # Convert to array
        feature_names = list(importance.keys())
        importance = [importance[f] for f in feature_names]
    elif model_type == 'catboost':
        importance = model.get_feature_importance()
        feature_names = model.feature_names_
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return df_importance


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[str] = None
):
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        save_path: Optional path to save plot
    """
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()

