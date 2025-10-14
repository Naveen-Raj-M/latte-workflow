"""
Distance analysis statistics computation module.

This module computes comprehensive distance-dependent statistics.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

from utils.distance_utils import get_distance_column_name

logger = logging.getLogger(__name__)


def compute_distance_statistics(df: pd.DataFrame, config: Dict[str, Any],
                               distance_type: str) -> Dict[str, Any]:
    """
    Compute comprehensive distance-dependent statistics.
    
    Args:
        df: Merged dataframe with distances
        config: Configuration dictionary
        distance_type: "hypocentral" or "epicentral"
        
    Returns:
        Dictionary of statistics
    """
    coordinate_type = config['distance_analysis']['coordinate_type']
    distance_col = get_distance_column_name(coordinate_type, distance_type)
    comparison_mode = config['distance_analysis'].get('comparison_mode', 'ground_truth')
    
    stats = {}
    stats['distance_type'] = distance_type
    stats['coordinate_type'] = coordinate_type
    stats['comparison_mode'] = comparison_mode
    
    if comparison_mode == 'method_comparison':
        stats['method1_name'] = config['distance_analysis'].get('method1_name', 'Method 1')
        stats['method2_name'] = config['distance_analysis'].get('method2_name', 'Method 2')
    
    # Overall distance statistics
    distances = df[distance_col]
    residuals = df['residual_s']
    
    stats['overall'] = {
        'count': len(df),
        'distance_min': float(distances.min()),
        'distance_max': float(distances.max()),
        'distance_mean': float(distances.mean()),
        'distance_median': float(distances.median()),
        'distance_std': float(distances.std()),
        'residual_distance_correlation': float(distances.corr(residuals)),
        'residual_abs_distance_correlation': float(distances.corr(np.abs(residuals)))
    }
    
    # Phase-specific statistics
    for phase in ['P', 'S']:
        phase_data = df[df['phase'] == phase]
        if not phase_data.empty:
            phase_stats = _compute_phase_statistics(
                phase_data, distance_col, config, phase
            )
            stats[f'{phase.lower()}_wave'] = phase_stats
    
    # Binned analysis
    if config['distance_analysis'].get('enable_binned_analysis', True):
        stats['binned'] = compute_binned_distance_statistics(df, config, distance_type)
    
    # Ratio analysis
    if config['distance_analysis'].get('enable_ratio_analysis', True):
        stats['ratio'] = compute_ratio_distance_statistics(df, config, distance_type)
    
    return stats


def _compute_phase_statistics(phase_data: pd.DataFrame, distance_col: str,
                              config: Dict[str, Any], phase: str) -> Dict[str, Any]:
    """Compute statistics for a specific phase."""
    phase_distances = phase_data[distance_col]
    phase_residuals = phase_data['residual_s']
    
    phase_stats = {
        'count': len(phase_data),
        'distance_mean': float(phase_distances.mean()),
        'residual_distance_correlation': float(phase_distances.corr(phase_residuals)),
        'residual_abs_distance_correlation': float(phase_distances.corr(np.abs(phase_residuals))),
        'mean_residual': float(phase_residuals.mean()),
        'rms_residual': float(np.sqrt(np.mean(phase_residuals**2)))
    }
    
    # Fit polynomial if enabled
    if config['distance_analysis'].get('fit_polynomial_degree', 1) > 0:
        degree = config['distance_analysis']['fit_polynomial_degree']
        coeffs = np.polyfit(phase_distances, phase_residuals, degree)
        phase_stats['polynomial_coeffs'] = [float(c) for c in coeffs]
    
    return phase_stats


def compute_binned_distance_statistics(df: pd.DataFrame, config: Dict[str, Any],
                                      distance_type: str) -> Dict[str, Any]:
    """Compute statistics for distance bins."""
    coordinate_type = config['distance_analysis']['coordinate_type']
    distance_col = get_distance_column_name(coordinate_type, distance_type)
    n_bins = config['distance_analysis'].get('distance_bins', 20)
    
    # Create distance bins
    df_copy = df.copy()
    df_copy['distance_bin'] = pd.cut(df_copy[distance_col], bins=n_bins)
    
    bin_stats = {}
    
    # Overall binned statistics
    overall_bins = df_copy.groupby('distance_bin', observed=True).agg({
        'residual_s': ['count', 'mean', 'std', lambda x: np.sqrt(np.mean(x**2))],
        distance_col: 'mean'
    }).round(4)
    
    overall_bins.columns = ['count', 'mean_residual', 'std_residual', 'rms_residual', 'mean_distance']
    bin_stats['overall'] = overall_bins.to_dict('index')
    
    # Phase-specific binned statistics
    for phase in ['P', 'S']:
        phase_data = df_copy[df_copy['phase'] == phase]
        if not phase_data.empty:
            phase_bins = phase_data.groupby('distance_bin', observed=True).agg({
                'residual_s': ['count', 'mean', 'std', lambda x: np.sqrt(np.mean(x**2))],
                distance_col: 'mean'
            }).round(4)
            
            phase_bins.columns = ['count', 'mean_residual', 'std_residual', 'rms_residual', 'mean_distance']
            bin_stats[f'{phase.lower()}_wave'] = phase_bins.to_dict('index')
    
    return bin_stats


def compute_ratio_distance_statistics(df: pd.DataFrame, config: Dict[str, Any],
                                     distance_type: str) -> Dict[str, Any]:
    """Compute ratio analysis statistics."""
    # Filter positive travel times for ratio analysis
    ratio_data = df[(df['observed_tt'] > 0) & (df['synthetic_tt'] > 0)].copy()
    
    if ratio_data.empty:
        logger.warning("No positive travel times available for ratio analysis")
        return {}
    
    coordinate_type = config['distance_analysis']['coordinate_type']
    distance_col = get_distance_column_name(coordinate_type, distance_type)
    
    ratio_data['ratio'] = ratio_data['synthetic_tt'] / ratio_data['observed_tt']
    ratio_data['ln_ratio'] = np.log(ratio_data['ratio'])
    
    ratio_stats = {}
    
    # Overall ratio statistics
    distances = ratio_data[distance_col]
    ratios = ratio_data['ratio']
    ln_ratios = ratio_data['ln_ratio']
    
    ratio_stats['overall'] = {
        'count': len(ratio_data),
        'mean_ratio': float(ratios.mean()),
        'mean_ln_ratio': float(ln_ratios.mean()),
        'ratio_distance_correlation': float(distances.corr(ratios)),
        'ln_ratio_distance_correlation': float(distances.corr(ln_ratios))
    }
    
    # Phase-specific ratio statistics
    for phase in ['P', 'S']:
        phase_data = ratio_data[ratio_data['phase'] == phase]
        if not phase_data.empty:
            phase_distances = phase_data[distance_col]
            phase_ratios = phase_data['ratio']
            phase_ln_ratios = phase_data['ln_ratio']
            
            ratio_stats[f'{phase.lower()}_wave'] = {
                'count': len(phase_data),
                'mean_ratio': float(phase_ratios.mean()),
                'mean_ln_ratio': float(phase_ln_ratios.mean()),
                'ratio_distance_correlation': float(phase_distances.corr(phase_ratios)),
                'ln_ratio_distance_correlation': float(phase_distances.corr(phase_ln_ratios))
            }
    
    return ratio_stats