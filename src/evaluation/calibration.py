"""
Model calibration utilities for fraud detection.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def calibrate_probabilities(
    y_proba: np.ndarray,
    y_true: np.ndarray,
    method: str = 'isotonic'
) -> np.ndarray:
    """
    Calibrate predicted probabilities using Platt scaling or isotonic regression.
    
    Args:
        y_proba: Uncalibrated probabilities
        y_true: True labels
        method: 'platt' (logistic regression) or 'isotonic'
    
    Returns:
        Calibrated probabilities
    """
    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_proba, y_true)
        y_proba_calibrated = calibrator.predict(y_proba)
    elif method == 'platt':
        # Platt scaling using logistic regression
        calibrator = LogisticRegression()
        calibrator.fit(y_proba.reshape(-1, 1), y_true)
        y_proba_calibrated = calibrator.predict_proba(y_proba.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    return y_proba_calibrated


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba_uncalibrated: np.ndarray,
    y_proba_calibrated: Optional[np.ndarray] = None,
    n_bins: int = 10,
    save_path: Optional[str] = None
):
    """
    Plot calibration curve comparing uncalibrated and calibrated probabilities.
    
    Args:
        y_true: True labels
        y_proba_uncalibrated: Uncalibrated probabilities
        y_proba_calibrated: Optional calibrated probabilities
        n_bins: Number of bins for calibration curve
        save_path: Optional path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Uncalibrated
    fraction_of_positives_uncal, mean_predicted_value_uncal = calibration_curve(
        y_true, y_proba_uncalibrated, n_bins=n_bins, strategy='uniform'
    )
    brier_uncal = brier_score_loss(y_true, y_proba_uncalibrated)
    
    ax.plot(mean_predicted_value_uncal, fraction_of_positives_uncal,
            's-', label=f'Uncalibrated (Brier={brier_uncal:.4f})', color='blue')
    
    # Calibrated
    if y_proba_calibrated is not None:
        fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
            y_true, y_proba_calibrated, n_bins=n_bins, strategy='uniform'
        )
        brier_cal = brier_score_loss(y_true, y_proba_calibrated)
        
        ax.plot(mean_predicted_value_cal, fraction_of_positives_cal,
                's-', label=f'Calibrated (Brier={brier_cal:.4f})', color='red')
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.5)
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curve', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Calibration curve saved to {save_path}")
    else:
        plt.show()


def calculate_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calculate Brier score (lower is better)."""
    return brier_score_loss(y_true, y_proba)


def calculate_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
    
    Returns:
        ECE score (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy in this bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_proba[in_bin].mean()
            
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

