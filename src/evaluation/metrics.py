"""
Evaluation metrics for fraud detection models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from typing import Dict, Tuple, Optional


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['tn'] = int(cm[0, 0])
    metrics['fp'] = int(cm[0, 1])
    metrics['fn'] = int(cm[1, 0])
    metrics['tp'] = int(cm[1, 1])
    
    return metrics


def precision_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: int = 100) -> float:
    """
    Calculate precision at top K predictions.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        k: Number of top predictions to consider
    
    Returns:
        Precision at K
    """
    # Get top K indices
    top_k_indices = np.argsort(y_proba)[-k:][::-1]
    
    # Calculate precision
    precision_k = y_true[top_k_indices].mean()
    
    return precision_k


def recall_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: int = 100) -> float:
    """
    Calculate recall at top K predictions.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        k: Number of top predictions to consider
    
    Returns:
        Recall at K
    """
    n_fraud = y_true.sum()
    if n_fraud == 0:
        return 0.0
    
    # Get top K indices
    top_k_indices = np.argsort(y_proba)[-k:][::-1]
    
    # Calculate recall
    recall_k = y_true[top_k_indices].sum() / n_fraud
    
    return recall_k


def print_metrics_report(metrics: Dict[str, float], title: str = "Model Metrics"):
    """Print formatted metrics report."""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0):.4f}")
    print(f"F1 Score:  {metrics.get('f1', 0):.4f}")
    
    if 'roc_auc' in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    if 'pr_auc' in metrics:
        print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics.get('tn', 0):6d}  FP: {metrics.get('fp', 0):6d}")
    print(f"  FN: {metrics.get('fn', 0):6d}  TP: {metrics.get('tp', 0):6d}")
    print(f"{'='*50}\n")

