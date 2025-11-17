"""
Threshold optimization for fraud detection based on cost matrices and business constraints.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt


def calculate_cost_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fraud_loss_cost: float = 1000.0,
    false_positive_cost: float = 10.0,
    true_positive_reward: float = 0.0
) -> Dict[str, float]:
    """
    Calculate total cost based on confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        fraud_loss_cost: Cost of missing a fraud (FN)
        false_positive_cost: Cost of false alarm (FP)
        true_positive_reward: Reward for catching fraud (TP, usually 0 or negative)
    
    Returns:
        Dictionary with cost breakdown
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    total_cost = (
        fn * fraud_loss_cost +      # Missed fraud
        fp * false_positive_cost -  # False alarms
        tp * true_positive_reward   # Caught fraud (reward)
    )
    
    return {
        'total_cost': total_cost,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'fraud_loss': fn * fraud_loss_cost,
        'false_positive_cost': fp * false_positive_cost,
        'true_positive_reward': tp * true_positive_reward
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fraud_loss_cost: float = 1000.0,
    false_positive_cost: float = 10.0,
    true_positive_reward: float = 0.0,
    max_investigations: Optional[int] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal threshold that minimizes total cost.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        fraud_loss_cost: Cost of missing fraud
        false_positive_cost: Cost of false alarm
        true_positive_reward: Reward for catching fraud
        max_investigations: Maximum number of investigations allowed (constraint)
    
    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    # Get precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    best_threshold = 0.5
    best_cost = float('inf')
    best_metrics = {}
    
    # Test different thresholds
    for threshold in np.arange(0.01, 1.0, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        
        # Check investigation constraint
        if max_investigations is not None:
            if y_pred.sum() > max_investigations:
                continue
        
        cost_metrics = calculate_cost_matrix(
            y_true, y_pred,
            fraud_loss_cost=fraud_loss_cost,
            false_positive_cost=false_positive_cost,
            true_positive_reward=true_positive_reward
        )
        
        if cost_metrics['total_cost'] < best_cost:
            best_cost = cost_metrics['total_cost']
            best_threshold = threshold
            best_metrics = cost_metrics.copy()
            best_metrics['threshold'] = threshold
            best_metrics['precision'] = precision[np.argmax(thresholds >= threshold)] if np.any(thresholds >= threshold) else 0
            best_metrics['recall'] = recall[np.argmax(thresholds >= threshold)] if np.any(thresholds >= threshold) else 0
    
    return best_threshold, best_metrics


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


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    optimal_threshold: Optional[float] = None,
    save_path: Optional[str] = None
):
    """Plot precision-recall curve with optional optimal threshold marker."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    if optimal_threshold is not None:
        # Find point on curve for optimal threshold
        idx = np.argmin(np.abs(thresholds - optimal_threshold))
        plt.plot(recall[idx], precision[idx], 'ro', markersize=10, label=f'Optimal Threshold ({optimal_threshold:.3f})')
    
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to {save_path}")
    else:
        plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    optimal_threshold: Optional[float] = None,
    save_path: Optional[str] = None
):
    """Plot ROC curve with optional optimal threshold marker."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = np.trapz(tpr, fpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if optimal_threshold is not None:
        # Find point on curve for optimal threshold
        idx = np.argmin(np.abs(thresholds - optimal_threshold))
        plt.plot(fpr[idx], tpr[idx], 'ro', markersize=10, label=f'Optimal Threshold ({optimal_threshold:.3f})')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()


def plot_cost_vs_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fraud_loss_cost: float = 1000.0,
    false_positive_cost: float = 10.0,
    save_path: Optional[str] = None
):
    """Plot total cost vs threshold."""
    thresholds = np.arange(0.01, 1.0, 0.01)
    costs = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cost_metrics = calculate_cost_matrix(
            y_true, y_pred,
            fraud_loss_cost=fraud_loss_cost,
            false_positive_cost=false_positive_cost
        )
        costs.append(cost_metrics['total_cost'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs)
    plt.xlabel('Threshold')
    plt.ylabel('Total Cost')
    plt.title('Total Cost vs Threshold')
    plt.grid(True, alpha=0.3)
    
    # Mark minimum
    min_idx = np.argmin(costs)
    plt.plot(thresholds[min_idx], costs[min_idx], 'ro', markersize=10,
             label=f'Minimum Cost at Threshold {thresholds[min_idx]:.3f}')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cost curve saved to {save_path}")
    else:
        plt.show()

