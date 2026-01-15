"""
Recovery analysis utilities for checkerboard tests.

This module provides functions for:
- Computing fair recovery masks (where inverted model recovers the pattern well)
- Creating contour data for visualization
"""

import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def compute_fair_recovery_mask(
    checkerboard_pattern: np.ndarray,
    inverted_pattern: np.ndarray,
    recovery_threshold: float,
) -> np.ndarray:
    """
    Compute mask for regions with fair recovery.

    Fair recovery is defined as regions where:
    1. The inverted pattern has the SAME SIGN as the checkerboard pattern
    2. The magnitude of the inverted pattern is above the threshold

    Args:
        checkerboard_pattern: Input checkerboard pattern (% perturbation)
        inverted_pattern: Recovered pattern from inversion (% perturbation)
        recovery_threshold: Minimum magnitude threshold for fair recovery (%)

    Returns:
        Boolean mask where True indicates fair recovery
    """
    # Compute sign agreement
    same_sign = np.sign(checkerboard_pattern) == np.sign(inverted_pattern)

    # Compute magnitude threshold
    above_threshold = np.abs(inverted_pattern) >= recovery_threshold

    # Fair recovery: both conditions must be true
    fair_recovery = same_sign & above_threshold

    # Log statistics
    total_points = fair_recovery.size
    recovery_points = np.sum(fair_recovery)
    recovery_percentage = (recovery_points / total_points) * 100 if total_points > 0 else 0

    logger.debug(
        f"Fair recovery: {recovery_points}/{total_points} points "
        f"({recovery_percentage:.2f}%) with threshold {recovery_threshold}%"
    )

    return fair_recovery


def compute_recovery_metrics(
    checkerboard_pattern: np.ndarray,
    inverted_pattern: np.ndarray,
) -> dict:
    """
    Compute various recovery metrics for analysis.

    Args:
        checkerboard_pattern: Input checkerboard pattern (% perturbation)
        inverted_pattern: Recovered pattern from inversion (% perturbation)

    Returns:
        Dictionary of metrics
    """
    # Mask out nan values
    valid_mask = ~(np.isnan(checkerboard_pattern) | np.isnan(inverted_pattern))

    if not np.any(valid_mask):
        return {
            "correlation": 0.0,
            "rmse": 0.0,
            "mean_absolute_error": 0.0,
            "sign_agreement": 0.0,
        }

    check_valid = checkerboard_pattern[valid_mask]
    inv_valid = inverted_pattern[valid_mask]

    # Correlation coefficient
    correlation = np.corrcoef(check_valid.flatten(), inv_valid.flatten())[0, 1]

    # RMSE
    rmse = np.sqrt(np.mean((check_valid - inv_valid) ** 2))

    # Mean absolute error
    mae = np.mean(np.abs(check_valid - inv_valid))

    # Sign agreement percentage
    sign_agreement = np.mean(np.sign(check_valid) == np.sign(inv_valid)) * 100

    return {
        "correlation": correlation,
        "rmse": rmse,
        "mean_absolute_error": mae,
        "sign_agreement": sign_agreement,
    }
