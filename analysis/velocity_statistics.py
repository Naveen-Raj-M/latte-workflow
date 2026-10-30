"""
Velocity model statistics computation module.

This module handles all statistical calculations for velocity model comparisons.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def compute_velocity_differences(df1: pd.DataFrame, df2: pd.DataFrame, 
                                config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Compute velocity differences between models.
    
    Args:
        df1: First velocity model dataframe
        df2: Second velocity model dataframe
        config: Configuration dictionary
        
    Returns:
        Dictionary containing difference arrays
    """
    logger.debug("Computing velocity differences")
    
    # Basic differences
    vp_diff = df2['vp'].values - df1['vp'].values
    vs_diff = df2['vs'].values - df1['vs'].values
    
    # Get velocities
    vp1, vs1 = df1['vp'].values, df1['vs'].values
    vp2, vs2 = df2['vp'].values, df2['vs'].values
    
    # Ratios
    ratio1 = vp1 / vs1
    ratio2 = vp2 / vs2
    ratio_diff = ratio2 - ratio1
    
    # Percent changes
    vp_percent = (vp_diff / vp1) * 100
    vs_percent = (vs_diff / vs1) * 100
    
    # Total difference magnitude
    total_diff = np.sqrt(vp_diff**2 + vs_diff**2)
    
    differences = {
        'vp_diff': vp_diff,
        'vs_diff': vs_diff,
        'ratio_diff': ratio_diff,
        'vp_percent': vp_percent,
        'vs_percent': vs_percent,
        'vp1': vp1, 'vp2': vp2,
        'vs1': vs1, 'vs2': vs2,
        'ratio1': ratio1, 'ratio2': ratio2,
        'total_diff': total_diff
    }
    
    logger.debug(f"Vp differences: {vp_diff.min():.3f} to {vp_diff.max():.3f} km/s")
    logger.debug(f"Vs differences: {vs_diff.min():.3f} to {vs_diff.max():.3f} km/s")
    
    return differences


def compute_log_ratios(df1: pd.DataFrame, df2: pd.DataFrame,
                      config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Compute logarithmic ratios between models.
    
    Args:
        df1: First velocity model dataframe
        df2: Second velocity model dataframe
        config: Configuration dictionary
        
    Returns:
        Dictionary containing log ratio arrays
    """
    logger.debug("Computing logarithmic ratios")
    
    vp1, vs1 = df1['vp'].values, df1['vs'].values
    vp2, vs2 = df2['vp'].values, df2['vs'].values
    
    # Compute log ratios: ln(model2/model1)
    vp_log_ratio = np.log(vp2 / vp1)
    vs_log_ratio = np.log(vs2 / vs1)
    
    return {
        'vp_log_ratio': vp_log_ratio,
        'vs_log_ratio': vs_log_ratio
    }


def compute_velocity_statistics(df1: pd.DataFrame, df2: pd.DataFrame,
                               differences: Dict[str, np.ndarray], 
                               config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute comprehensive velocity comparison statistics.
    
    Args:
        df1: First velocity model dataframe
        df2: Second velocity model dataframe
        differences: Differences dictionary
        config: Configuration dictionary
        
    Returns:
        Dictionary of statistics
    """
    logger.debug("Computing velocity statistics")
    
    stats = {}
    total_points = len(differences['vp_diff'])
    
    # Vp comparison statistics
    vp_diff = differences['vp_diff']
    vp_percent = differences['vp_percent']
    
    stats['vp_comparison'] = {
        'model1_mean': float(differences['vp1'].mean()),
        'model1_std': float(differences['vp1'].std()),
        'model1_min': float(differences['vp1'].min()),
        'model1_max': float(differences['vp1'].max()),
        'model2_mean': float(differences['vp2'].mean()),
        'model2_std': float(differences['vp2'].std()),
        'model2_min': float(differences['vp2'].min()),
        'model2_max': float(differences['vp2'].max()),
        'diff_mean': float(vp_diff.mean()),
        'diff_std': float(vp_diff.std()),
        'diff_min': float(vp_diff.min()),
        'diff_max': float(vp_diff.max()),
        'diff_abs_mean': float(np.abs(vp_diff).mean()),
        'diff_rms': float(np.sqrt(np.mean(vp_diff**2))),
        'percent_mean': float(vp_percent.mean()),
        'percent_std': float(vp_percent.std()),
        'points_increased': int(np.sum(vp_diff > 0)),
        'points_decreased': int(np.sum(vp_diff < 0)),
        'points_unchanged': int(np.sum(np.abs(vp_diff) < 1e-6)),
        'correlation': float(np.corrcoef(differences['vp1'], differences['vp2'])[0,1]),
        'total_points': total_points
    }
    
    # Vs comparison statistics
    vs_diff = differences['vs_diff']
    vs_percent = differences['vs_percent']
    
    stats['vs_comparison'] = {
        'model1_mean': float(differences['vs1'].mean()),
        'model1_std': float(differences['vs1'].std()),
        'model1_min': float(differences['vs1'].min()),
        'model1_max': float(differences['vs1'].max()),
        'model2_mean': float(differences['vs2'].mean()),
        'model2_std': float(differences['vs2'].std()),
        'model2_min': float(differences['vs2'].min()),
        'model2_max': float(differences['vs2'].max()),
        'diff_mean': float(vs_diff.mean()),
        'diff_std': float(vs_diff.std()),
        'diff_min': float(vs_diff.min()),
        'diff_max': float(vs_diff.max()),
        'diff_abs_mean': float(np.abs(vs_diff).mean()),
        'diff_rms': float(np.sqrt(np.mean(vs_diff**2))),
        'percent_mean': float(vs_percent.mean()),
        'percent_std': float(vs_percent.std()),
        'points_increased': int(np.sum(vs_diff > 0)),
        'points_decreased': int(np.sum(vs_diff < 0)),
        'points_unchanged': int(np.sum(np.abs(vs_diff) < 1e-6)),
        'correlation': float(np.corrcoef(differences['vs1'], differences['vs2'])[0,1]),
        'total_points': total_points
    }
    
    # Vp/Vs ratio comparison
    ratio_diff = differences['ratio_diff']
    stats['ratio_comparison'] = {
        'model1_mean': float(differences['ratio1'].mean()),
        'model1_std': float(differences['ratio1'].std()),
        'model1_min': float(differences['ratio1'].min()),
        'model1_max': float(differences['ratio1'].max()),
        'model2_mean': float(differences['ratio2'].mean()),
        'model2_std': float(differences['ratio2'].std()),
        'model2_min': float(differences['ratio2'].min()),
        'model2_max': float(differences['ratio2'].max()),
        'diff_mean': float(ratio_diff.mean()),
        'diff_std': float(ratio_diff.std()),
        'diff_min': float(ratio_diff.min()),
        'diff_max': float(ratio_diff.max()),
        'correlation': float(np.corrcoef(differences['ratio1'], differences['ratio2'])[0,1])
    }
    
    return stats


def compute_binned_statistics(values: np.ndarray, bins: np.ndarray,
                             data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute binned statistics for data.
    
    Args:
        values: Values to bin by (e.g., distances)
        bins: Bin edges
        data: Data to compute statistics for
        
    Returns:
        Dictionary of binned statistics
    """
    bin_indices = np.digitize(values, bins)
    
    bin_means = []
    bin_stds = []
    bin_counts = []
    bin_centers = []
    
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_means.append(np.mean(data[mask]))
            bin_stds.append(np.std(data[mask]))
            bin_counts.append(np.sum(mask))
            bin_centers.append((bins[i-1] + bins[i]) / 2)
    
    return {
        'centers': np.array(bin_centers),
        'means': np.array(bin_means),
        'stds': np.array(bin_stds),
        'counts': np.array(bin_counts)
    }


def compute_depth_statistics(df: pd.DataFrame, differences: Dict[str, np.ndarray],
                            config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute depth-dependent statistics.
    
    Args:
        df: Velocity dataframe
        differences: Differences dictionary
        config: Configuration dictionary
        
    Returns:
        DataFrame with depth statistics
    """
    df_temp = df.copy()
    df_temp['vp_diff'] = differences['vp_diff']
    df_temp['vs_diff'] = differences['vs_diff']
    df_temp['total_diff'] = differences['total_diff']
    
    depth_stats = df_temp.groupby('z').agg({
        'vp_diff': ['mean', 'std', 'min', 'max', lambda x: np.sqrt(np.mean(x**2))],
        'vs_diff': ['mean', 'std', 'min', 'max', lambda x: np.sqrt(np.mean(x**2))],
        'total_diff': ['mean', 'max', 'std'],
        'x': 'count'
    }).round(4)
    
    # Flatten column names
    depth_stats.columns = ['_'.join(col).strip() for col in depth_stats.columns]
    depth_stats = depth_stats.rename(columns={'x_count': 'point_count'})
    
    return depth_stats


def compute_spatial_rms(df: pd.DataFrame, diff_col: str, 
                       groupby_cols: list) -> pd.DataFrame:
    """
    Compute RMS in spatial groupings.
    
    Args:
        df: Dataframe with difference column
        diff_col: Name of difference column
        groupby_cols: Columns to group by
        
    Returns:
        DataFrame with RMS values
    """
    return df.groupby(groupby_cols).agg({
        diff_col: lambda x: np.sqrt(np.mean(x**2))
    }).reset_index()


def compute_geometric_mean_ratio(df: pd.DataFrame, ratio_col: str,
                                groupby_cols: list) -> pd.DataFrame:
    """
    Compute geometric mean of ratios in spatial groupings.
    
    Args:
        df: Dataframe with ratio column
        ratio_col: Name of ratio column
        groupby_cols: Columns to group by
        
    Returns:
        DataFrame with geometric mean ratios
    """
    return df.groupby(groupby_cols).agg({
        ratio_col: lambda x: np.exp(np.mean(np.log(x)))
    }).reset_index()