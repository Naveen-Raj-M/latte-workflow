"""
Velocity model spatial analysis module.

This module handles spatial analysis of velocity differences including
finding extreme profiles and depth slices. Grid nodes are (lon, lat, depth): longitude and WGS84
geodetic latitude in degrees, depth below mean sea level in km. A vertical profile is the set of
nodes at one (lon, lat); every statistic weighs the nodes equally.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def find_extreme_velocity_profiles(df: pd.DataFrame, differences: Dict[str, np.ndarray],
                                  config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find locations with extreme velocity differences.
    
    Args:
        df: Velocity dataframe (either model)
        differences: Differences dictionary
        config: Configuration dictionary
        
    Returns:
        Dictionary with extreme profile information
    """
    logger.debug("Finding extreme velocity difference profiles")
    
    total_diff = differences['total_diff']
    
    # Add total difference to dataframe for grouping
    df_with_diff = df.copy()
    df_with_diff['total_diff'] = total_diff
    
    # Group by (lon, lat) to get vertical profiles
    profile_stats = df_with_diff.groupby(['lon', 'lat']).agg({
        'total_diff': ['mean', 'max', 'sum', 'std'],
        'depth': ['min', 'max', 'count']
    }).round(3)
    
    # Flatten column names
    profile_stats.columns = ['_'.join(col).strip() for col in profile_stats.columns]
    
    # Find profiles with maximum and minimum average differences
    max_profile_coords = profile_stats['total_diff_mean'].idxmax()
    min_profile_coords = profile_stats[profile_stats['total_diff_mean'] > 1e-6]['total_diff_mean'].idxmin()
    
    # Get profile masks
    max_profile_mask = ((df['lon'] == max_profile_coords[0]) &
                        (df['lat'] == max_profile_coords[1]))
    min_profile_mask = ((df['lon'] == min_profile_coords[0]) &
                        (df['lat'] == min_profile_coords[1]))

    # Get profile data
    max_profile_data = df[max_profile_mask].copy().sort_values('depth')
    min_profile_data = df[min_profile_mask].copy().sort_values('depth')

    # Find depth slices with extreme differences
    depth_diff_stats = df_with_diff.groupby('depth')['total_diff'].mean()
    max_diff_depth = depth_diff_stats.idxmax()
    min_diff_depth = depth_diff_stats[depth_diff_stats > 1e-6].idxmin()
    
    logger.debug(f"Max difference profile at lon {max_profile_coords[0]:.4f}, lat {max_profile_coords[1]:.4f} deg")
    logger.debug(f"Min difference profile at lon {min_profile_coords[0]:.4f}, lat {min_profile_coords[1]:.4f} deg")
    logger.debug(f"Max difference depth: {max_diff_depth:.1f} km")
    logger.debug(f"Min difference depth: {min_diff_depth:.1f} km")
    
    return {
        'max_profile_coords': max_profile_coords,
        'min_profile_coords': min_profile_coords,
        'max_profile_mask': max_profile_mask,
        'min_profile_mask': min_profile_mask,
        'max_profile_data': max_profile_data,
        'min_profile_data': min_profile_data,
        'profile_stats': profile_stats,
        'max_diff_depth': max_diff_depth,
        'min_diff_depth': min_diff_depth,
        'total_diff': total_diff
    }


def find_max_min_ratio_change_depths(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
    """
    Find the depth slices with maximum and minimum average ratio changes.
    
    Args:
        df1: First velocity model dataframe
        df2: Second velocity model dataframe
        
    Returns:
        Dictionary with depth information
    """
    logger.debug("Finding depths with maximum and minimum average ratio changes")
    
    # Get all unique depths
    depths = sorted(df1['depth'].unique())
    
    depth_avg_changes = []
    
    for depth in depths:
        depth_data1 = df1[df1['depth'] == depth]
        depth_data2 = df2[df2['depth'] == depth]
        
        if len(depth_data1) > 0 and len(depth_data2) > 0:
            # Calculate ratios
            vp_ratios = depth_data2['vp'].values / depth_data1['vp'].values
            vs_ratios = depth_data2['vs'].values / depth_data1['vs'].values
            
            # Calculate average absolute deviation from 1.0 for the entire slice
            vp_avg_change = np.mean(np.abs(vp_ratios - 1.0))
            vs_avg_change = np.mean(np.abs(vs_ratios - 1.0))
            
            # Combined average change
            combined_avg_change = (vp_avg_change + vs_avg_change) / 2.0
            
            depth_avg_changes.append({
                'depth': depth,
                'vp_avg_change': vp_avg_change,
                'vs_avg_change': vs_avg_change,
                'combined_avg_change': combined_avg_change
            })
    
    # Sort by combined average change
    depth_avg_changes.sort(key=lambda x: x['combined_avg_change'])
    
    # Get depths with minimum and maximum average changes
    min_change_info = depth_avg_changes[0]
    max_change_info = depth_avg_changes[-1]
    
    logger.debug(f"Depth with minimum average ratio change: {min_change_info['depth']:.1f} km")
    logger.debug(f"Depth with maximum average ratio change: {max_change_info['depth']:.1f} km")
    
    return {
        'max_change_depth': max_change_info['depth'],
        'max_vp_avg_change': max_change_info['vp_avg_change'],
        'max_vs_avg_change': max_change_info['vs_avg_change'],
        'max_combined_avg_change': max_change_info['combined_avg_change'],
        'min_change_depth': min_change_info['depth'],
        'min_vp_avg_change': min_change_info['vp_avg_change'],
        'min_vs_avg_change': min_change_info['vs_avg_change'],
        'min_combined_avg_change': min_change_info['combined_avg_change']
    }


def find_extreme_ratio_change_profiles(df1: pd.DataFrame, df2: pd.DataFrame,
                                      config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find vertical profiles with maximum positive ratio change and minimum ratio change.

    Args:
        df1: First velocity model dataframe
        df2: Second velocity model dataframe
        config: Configuration dictionary

    Returns:
        Dictionary with extreme ratio profile information
    """
    logger.debug("Finding extreme ratio change profiles")

    # Calculate ratios
    vp_ratios = df2['vp'].values / df1['vp'].values
    vs_ratios = df2['vs'].values / df1['vs'].values

    # Add ratios to dataframe for grouping
    df_with_ratios = df1.copy()
    df_with_ratios['vp_ratio'] = vp_ratios
    df_with_ratios['vs_ratio'] = vs_ratios
    df_with_ratios['vp_ratio_change'] = vp_ratios - 1.0
    df_with_ratios['vs_ratio_change'] = vs_ratios - 1.0
    df_with_ratios['combined_ratio_change'] = (df_with_ratios['vp_ratio_change'] +
                                                df_with_ratios['vs_ratio_change']) / 2.0

    # Group by (lon, lat) to get vertical profiles
    profile_stats = df_with_ratios.groupby(['lon', 'lat']).agg({
        'combined_ratio_change': ['mean', 'max', 'min', 'std'],
        'vp_ratio_change': ['mean'],
        'vs_ratio_change': ['mean'],
        'depth': ['min', 'max', 'count']
    }).round(6)

    # Flatten column names
    profile_stats.columns = ['_'.join(col).strip() for col in profile_stats.columns]

    # Find profile with maximum positive ratio change (most positive mean)
    max_positive_profile_coords = profile_stats['combined_ratio_change_mean'].idxmax()

    # Find profile with minimum ratio change (most negative mean)
    min_ratio_profile_coords = profile_stats['combined_ratio_change_mean'].idxmin()

    # Get profile masks
    max_positive_mask = ((df1['lon'] == max_positive_profile_coords[0]) &
                         (df1['lat'] == max_positive_profile_coords[1]))
    min_ratio_mask = ((df1['lon'] == min_ratio_profile_coords[0]) &
                      (df1['lat'] == min_ratio_profile_coords[1]))

    # Get profile data
    max_positive_profile_data = df1[max_positive_mask].copy().sort_values('depth')
    min_ratio_profile_data = df1[min_ratio_mask].copy().sort_values('depth')

    max_positive_avg_change = profile_stats.loc[max_positive_profile_coords, 'combined_ratio_change_mean']
    min_ratio_avg_change = profile_stats.loc[min_ratio_profile_coords, 'combined_ratio_change_mean']

    logger.debug(f"Max positive ratio change profile at lon {max_positive_profile_coords[0]:.4f}, "
                 f"lat {max_positive_profile_coords[1]:.4f} deg, avg change: {max_positive_avg_change:.4f}")
    logger.debug(f"Min ratio change profile at lon {min_ratio_profile_coords[0]:.4f}, "
                 f"lat {min_ratio_profile_coords[1]:.4f} deg, avg change: {min_ratio_avg_change:.4f}")

    return {
        'max_positive_coords': max_positive_profile_coords,
        'min_ratio_coords': min_ratio_profile_coords,
        'max_positive_mask': max_positive_mask,
        'min_ratio_mask': min_ratio_mask,
        'max_positive_profile_data': max_positive_profile_data,
        'min_ratio_profile_data': min_ratio_profile_data,
        'max_positive_avg_change': max_positive_avg_change,
        'min_ratio_avg_change': min_ratio_avg_change,
        'profile_stats': profile_stats
    }


def get_depth_slice_data(df1: pd.DataFrame, df2: pd.DataFrame,
                        depth: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract data for a specific depth slice from both models.

    Args:
        df1: First velocity model dataframe
        df2: Second velocity model dataframe
        depth: Depth value to extract

    Returns:
        Tuple of (slice_data1, slice_data2)
    """
    slice_data1 = df1[df1['depth'] == depth].copy()
    slice_data2 = df2[df2['depth'] == depth].copy()

    return slice_data1, slice_data2


def compute_spatial_extent(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Compute spatial extent of the velocity model domain.
    
    Args:
        df: Velocity dataframe
        
    Returns:
        Dictionary with min/max for each dimension
    """
    return {axis: (df[axis].min(), df[axis].max()) for axis in ('lon', 'lat', 'depth')}


def get_unique_coordinates(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Get unique coordinate values from velocity model.
    
    Args:
        df: Velocity dataframe
        
    Returns:
        Dictionary with unique values for each coordinate
    """
    return {axis: np.sort(df[axis].unique()) for axis in ('lon', 'lat', 'depth')}