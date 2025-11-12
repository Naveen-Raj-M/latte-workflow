"""
Distance analysis data loading and preparation module.

This module handles data loading for both ground_truth and method_comparison modes.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

from utils.io_utils import read_csv_file
from utils.distance_utils import (
    calculate_all_distances, validate_distance_types,
    get_distance_column_name
)
from .filters import apply_analysis_filters

logger = logging.getLogger(__name__)


def load_and_merge_distance_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load and merge travel time, station, and event data with distance calculations.
    For ground_truth mode.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Merged dataframe with distance calculations
    """
    travel_time_csv = config['distance_analysis']['travel_time_csv']
    station_csv = config['distance_analysis']['station_csv']
    events_csv = config['distance_analysis']['events_csv']
    
    logger.info(f"Loading data files:")
    logger.info(f"  Travel times: {travel_time_csv}")
    logger.info(f"  Stations: {station_csv}")
    logger.info(f"  Events: {events_csv}")
    
    # Load data
    df_tt = read_csv_file(travel_time_csv)
    df_stations = read_csv_file(station_csv)
    df_events = read_csv_file(events_csv)
    
    logger.info(f"Loaded {len(df_tt)} travel time records")
    logger.info(f"Loaded {len(df_stations)} stations")
    logger.info(f"Loaded {len(df_events)} events")
    
    # Merge with station data
    df_merged = df_tt.merge(df_stations, on='station', how='left', suffixes=('', '_x'))
    logger.info(f"After station merge: {len(df_merged)} records")

    # Merge with event data
    df_merged = df_merged.merge(df_events, on='event_id', how='left', suffixes=('', '_y'))
    logger.info(f"After event merge: {len(df_merged)} records")
    
    # Calculate all requested distance types
    coordinate_type = config['distance_analysis']['coordinate_type']
    distance_types = validate_distance_types(
        config['distance_analysis'].get('distance_types', ['hypocentral'])
    )
    
    df_merged = calculate_all_distances(df_merged, coordinate_type, distance_types)
    
    # Remove records with missing distances
    df_merged = _remove_missing_distances(df_merged, coordinate_type, distance_types)

    # Apply analysis filters (depth and phase type)
    df_merged = apply_analysis_filters(df_merged, config, data_type="distance analysis data")

    logger.info(f"Final merged dataset: {len(df_merged)} records")
    return df_merged


def prepare_method_comparison_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Prepare data for method comparison mode.
    Loads two separate travel time files and computes residuals between them.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Merged dataframe with computed residuals and distance calculations
    """
    method1_csv = config['distance_analysis']['method1_csv']
    method2_csv = config['distance_analysis']['method2_csv']
    station_csv = config['distance_analysis']['station_csv']
    events_csv = config['distance_analysis']['events_csv']
    
    method1_name = config['distance_analysis'].get('method1_name', 'Method 1')
    method2_name = config['distance_analysis'].get('method2_name', 'Method 2')
    
    logger.info(f"Loading method comparison data:")
    logger.info(f"  Method 1 ({method1_name}): {method1_csv}")
    logger.info(f"  Method 2 ({method2_name}): {method2_csv}")
    logger.info(f"  Stations: {station_csv}")
    logger.info(f"  Events: {events_csv}")
    
    # Load travel time data
    df_method1 = read_csv_file(method1_csv)
    df_method2 = read_csv_file(method2_csv)
    df_stations = read_csv_file(station_csv)
    df_events = read_csv_file(events_csv)
    
    logger.info(f"Loaded {len(df_method1)} records for {method1_name}")
    logger.info(f"Loaded {len(df_method2)} records for {method2_name}")
    logger.info(f"Loaded {len(df_stations)} stations")
    logger.info(f"Loaded {len(df_events)} events")
    
    # Merge the two methods
    df_merged = _merge_method_data(df_method1, df_method2, method1_name, method2_name)
    
    # Merge with station data
    df_merged = df_merged.merge(df_stations, on='station', how='left', suffixes=('', '_x'))
    logger.info(f"After station merge: {len(df_merged)} records")
    
    # Merge with event data
    df_merged = df_merged.merge(df_events, on='event_id', how='left', suffixes=('', '_y'))
    logger.info(f"After event merge: {len(df_merged)} records")
    
    # Calculate all requested distance types
    coordinate_type = config['distance_analysis']['coordinate_type']
    distance_types = validate_distance_types(
        config['distance_analysis'].get('distance_types', ['hypocentral'])
    )
    
    df_merged = calculate_all_distances(df_merged, coordinate_type, distance_types)

    # Remove records with missing distances
    df_merged = _remove_missing_distances(df_merged, coordinate_type, distance_types)

    # Apply analysis filters (depth and phase type)
    df_merged = apply_analysis_filters(df_merged, config, data_type="distance method comparison data")

    logger.info(f"Final merged dataset: {len(df_merged)} records")
    logger.info(f"Method comparison: {method1_name} vs {method2_name}")

    return df_merged


def _merge_method_data(df_method1: pd.DataFrame, df_method2: pd.DataFrame,
                       method1_name: str, method2_name: str) -> pd.DataFrame:
    """
    Merge two method datasets and compute residuals.
    
    Args:
        df_method1: First method dataframe
        df_method2: Second method dataframe
        method1_name: Name of first method
        method2_name: Name of second method
        
    Returns:
        Merged dataframe with residuals
    """
    # Identify common columns for merging (event_id, station, phase)
    merge_keys = ['event_id', 'station', 'phase']
    
    # Check if merge keys exist
    for key in merge_keys:
        if key not in df_method1.columns or key not in df_method2.columns:
            raise ValueError(f"Required merge key '{key}' not found in both method datasets")
    
    # Merge on event_id, station, and phase
    df_merged = df_method1.merge(
        df_method2,
        on=merge_keys,
        how='inner',
        suffixes=(f'_{method1_name}', f'_{method2_name}')
    )
    
    logger.info(f"Matched {len(df_merged)} common records between methods")
    
    # Determine column names for travel times
    # Try common column name patterns
    tt_patterns = ['synthetic_tt', 'observed_tt', 'travel_time', 'travel_time_s', 'traveltime', 'tt']
    
    method1_tt_col = None
    method2_tt_col = None
    
    for pattern in tt_patterns:
        col1 = f'{pattern}_{method1_name}'
        col2 = f'{pattern}_{method2_name}'
        if col1 in df_merged.columns and col2 in df_merged.columns:
            method1_tt_col = col1
            method2_tt_col = col2
            break
    
    if method1_tt_col is None or method2_tt_col is None:
        raise ValueError(
            f"Could not identify travel time columns. Expected patterns: {tt_patterns}"
        )
    
    logger.info(f"Using travel time columns: {method1_tt_col} and {method2_tt_col}")
    
    # Compute residuals (Method1 - Method2)
    # Use new column names to avoid conflicts with existing columns
    df_merged['observed_tt'] = df_merged[method1_tt_col].values
    df_merged['synthetic_tt'] = df_merged[method2_tt_col].values
    df_merged['residual_s'] = df_merged['observed_tt'] - df_merged['synthetic_tt']
    
    # Log statistics
    logger.info(f"Residual statistics ({method1_name} - {method2_name}):")
    logger.info(f"  Mean: {df_merged['residual_s'].mean():.4f} s")
    logger.info(f"  Std: {df_merged['residual_s'].std():.4f} s")
    logger.info(f"  RMS: {np.sqrt(np.mean(df_merged['residual_s']**2)):.4f} s")
    
    return df_merged


def _remove_missing_distances(df: pd.DataFrame, coordinate_type: str,
                              distance_types: list) -> pd.DataFrame:
    """
    Remove records with missing distance values.
    
    Args:
        df: Input dataframe
        coordinate_type: Coordinate type
        distance_types: List of distance types
        
    Returns:
        Filtered dataframe
    """
    initial_count = len(df)
    
    for distance_type in distance_types:
        col_name = get_distance_column_name(coordinate_type, distance_type)
        df = df.dropna(subset=[col_name])
    
    final_count = len(df)
    
    if final_count < initial_count:
        logger.warning(f"Removed {initial_count - final_count} records with missing distances")
    
    return df