"""
Short distance analysis statistics computation module.

This module computes comprehensive statistics for short distance analysis.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

from utils.distance_utils import get_distance_column_name

logger = logging.getLogger(__name__)


def compute_short_distance_statistics(df: pd.DataFrame, config: Dict[str, Any],
                                     distance_type: str) -> Dict[str, Any]:
    """
    Compute comprehensive short distance statistics.
    
    Args:
        df: Filtered dataframe with short distances
        config: Configuration dictionary
        distance_type: "hypocentral" or "epicentral"
        
    Returns:
        Dictionary of statistics
    """
    coordinate_type = config['short_distance_analysis']['coordinate_type']
    distance_col = get_distance_column_name(coordinate_type, distance_type)
    comparison_mode = config['short_distance_analysis'].get('comparison_mode', 'ground_truth')
    distance_cutoff = config['short_distance_analysis']['distance_cutoff_km']
    
    stats = {}
    stats['distance_type'] = distance_type
    stats['coordinate_type'] = coordinate_type
    stats['comparison_mode'] = comparison_mode
    stats['distance_cutoff_km'] = distance_cutoff
    
    if comparison_mode == 'method_comparison':
        stats['method1_name'] = config['short_distance_analysis'].get('method1_name', 'Method 1')
        stats['method2_name'] = config['short_distance_analysis'].get('method2_name', 'Method 2')
    
    # Overall statistics
    stats['overall'] = _compute_overall_statistics(df, distance_col)
    
    # Phase-specific statistics
    for phase in ['P', 'S']:
        phase_data = df[df['phase'] == phase]
        if not phase_data.empty:
            stats[f'{phase.lower()}_wave'] = _compute_phase_statistics(
                phase_data, distance_col, distance_type
            )
    
    # Event and station statistics
    stats['events'] = _compute_event_statistics(df)
    stats['stations'] = _compute_station_statistics(df)
    
    # Cluster analysis if enabled
    if config['short_distance_analysis'].get('enable_zoom_clusters', False):
        stats['clusters'] = _compute_cluster_statistics(df, config, distance_col)
    
    return stats


def _compute_overall_statistics(df: pd.DataFrame, distance_col: str) -> Dict[str, Any]:
    """Compute overall statistics."""
    distances = df[distance_col]
    residuals = df['residual_s']
    
    return {
        'count': len(df),
        'unique_events': df['event_id'].nunique(),
        'unique_stations': df['station'].nunique(),
        'distance_min': float(distances.min()),
        'distance_max': float(distances.max()),
        'distance_mean': float(distances.mean()),
        'distance_median': float(distances.median()),
        'residual_mean': float(residuals.mean()),
        'residual_std': float(residuals.std()),
        'residual_rms': float(np.sqrt(np.mean(residuals**2))),
        'residual_min': float(residuals.min()),
        'residual_max': float(residuals.max()),
        'residual_median': float(residuals.median())
    }


def _compute_phase_statistics(phase_data: pd.DataFrame, distance_col: str,
                              distance_type: str) -> Dict[str, Any]:
    """Compute statistics for a specific phase."""
    distances = phase_data[distance_col]
    residuals = phase_data['residual_s']
    
    return {
        'count': len(phase_data),
        'distance_mean': float(distances.mean()),
        'distance_std': float(distances.std()),
        'residual_mean': float(residuals.mean()),
        'residual_std': float(residuals.std()),
        'residual_rms': float(np.sqrt(np.mean(residuals**2))),
        'residual_min': float(residuals.min()),
        'residual_max': float(residuals.max()),
        'distance_residual_corr': float(distances.corr(residuals))
    }


def _compute_event_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute event-related statistics."""
    event_counts = df.groupby('event_id').size()
    
    return {
        'unique_events': len(event_counts),
        'mean_picks_per_event': float(event_counts.mean()),
        'median_picks_per_event': float(event_counts.median()),
        'min_picks_per_event': int(event_counts.min()),
        'max_picks_per_event': int(event_counts.max())
    }


def _compute_station_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute station-related statistics."""
    station_counts = df.groupby('station').size()
    
    return {
        'unique_stations': len(station_counts),
        'mean_picks_per_station': float(station_counts.mean()),
        'median_picks_per_station': float(station_counts.median()),
        'min_picks_per_station': int(station_counts.min()),
        'max_picks_per_station': int(station_counts.max())
    }


def _compute_cluster_statistics(df: pd.DataFrame, config: Dict[str, Any],
                                distance_col: str) -> Dict[str, Any]:
    """Compute statistics for defined zoom clusters."""
    clusters = config['short_distance_analysis'].get('zoom_clusters', [])
    
    if not clusters:
        return {}
    
    coordinate_type = config['short_distance_analysis']['coordinate_type']
    
    # Determine coordinate column names
    if coordinate_type == 'transformed':
        x_col_event = 'X_grid_km'
        y_col_event = 'Y_grid_km'
        x_col_station = 'X_grid_km'
        y_col_station = 'Y_grid_km'
    else:
        x_col_event = 'x_km'
        y_col_event = 'y_km'
        x_col_station = 'x_km'
        y_col_station = 'y_km'
    
    # Check for suffixes from merge
    if x_col_event + '_y' in df.columns:
        x_col_event += '_y'
        y_col_event += '_y'
    if x_col_station + '_x' in df.columns:
        x_col_station += '_x'
        y_col_station += '_x'
    
    cluster_stats = {}
    
    for cluster in clusters:
        cluster_name = cluster['name']
        x_min, x_max = cluster['x_min'], cluster['x_max']
        y_min, y_max = cluster['y_min'], cluster['y_max']
        
        # Filter data within cluster boundaries (either event or station in cluster)
        cluster_data = df[
            ((df[x_col_event] >= x_min) & (df[x_col_event] <= x_max) &
             (df[y_col_event] >= y_min) & (df[y_col_event] <= y_max)) |
            ((df[x_col_station] >= x_min) & (df[x_col_station] <= x_max) &
             (df[y_col_station] >= y_min) & (df[y_col_station] <= y_max))
        ]
        
        if not cluster_data.empty:
            residuals = cluster_data['residual_s']
            distances = cluster_data[distance_col]
            
            cluster_stats[cluster_name] = {
                'count': len(cluster_data),
                'description': cluster.get('description', ''),
                'residual_mean': float(residuals.mean()),
                'residual_std': float(residuals.std()),
                'residual_rms': float(np.sqrt(np.mean(residuals**2))),
                'distance_mean': float(distances.mean()),
                'unique_events': cluster_data['event_id'].nunique(),
                'unique_stations': cluster_data['station'].nunique()
            }
        else:
            cluster_stats[cluster_name] = {
                'count': 0,
                'description': cluster.get('description', ''),
                'message': 'No data in this cluster'
            }
    
    return cluster_stats