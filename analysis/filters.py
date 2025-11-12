"""
Common filtering utilities for all analysis modules.

This module provides reusable filters for depth and phase type filtering
to avoid circular import issues.
"""

import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def apply_depth_filter(df: pd.DataFrame, max_depth_km: Optional[float],
                       depth_column: str = 'event_z_transformed',
                       data_type: str = "data") -> pd.DataFrame:
    """
    Apply depth filtering to analysis data.

    Args:
        df: DataFrame to filter
        max_depth_km: Maximum depth in km (None = no filtering)
        depth_column: Name of depth column to filter on
        data_type: Description for logging

    Returns:
        Filtered DataFrame
    """
    if max_depth_km is None:
        return df

    if depth_column not in df.columns:
        logger.warning(f"Depth column '{depth_column}' not found in {data_type}. Skipping depth filtering.")
        return df

    initial_count = len(df)
    filtered_df = df[df[depth_column] <= max_depth_km].copy()
    removed_count = initial_count - len(filtered_df)

    if removed_count > 0:
        logger.info(f"Depth filtering for {data_type}:")
        logger.info(f"  Maximum depth limit: {max_depth_km} km")
        logger.info(f"  Points before filtering: {initial_count}")
        logger.info(f"  Points removed (depth > {max_depth_km} km): {removed_count}")
        logger.info(f"  Points remaining: {len(filtered_df)}")
    else:
        logger.info(f"Depth filtering for {data_type}: No points exceeded {max_depth_km} km limit")

    return filtered_df


def apply_phase_filter(df: pd.DataFrame, phase_type: str = "both",
                       phase_column: str = 'phase',
                       data_type: str = "data") -> pd.DataFrame:
    """
    Apply phase type filtering to analysis data.

    Args:
        df: DataFrame to filter
        phase_type: "both", "p", or "s"
        phase_column: Name of phase column
        data_type: Description for logging

    Returns:
        Filtered DataFrame
    """
    phase_type = phase_type.lower()

    if phase_type == "both":
        return df

    if phase_column not in df.columns:
        logger.warning(f"Phase column '{phase_column}' not found in {data_type}. Skipping phase filtering.")
        return df

    initial_count = len(df)

    if phase_type == "p":
        filtered_df = df[df[phase_column].str.upper() == 'P'].copy()
        phase_label = "P-wave"
    elif phase_type == "s":
        filtered_df = df[df[phase_column].str.upper() == 'S'].copy()
        phase_label = "S-wave"
    else:
        logger.warning(f"Invalid phase_type: {phase_type}. Using 'both'.")
        return df

    removed_count = initial_count - len(filtered_df)

    logger.info(f"Phase filtering for {data_type}:")
    logger.info(f"  Phase type: {phase_label} only")
    logger.info(f"  Points before filtering: {initial_count}")
    logger.info(f"  Points removed: {removed_count}")
    logger.info(f"  Points remaining: {len(filtered_df)}")

    return filtered_df


def apply_analysis_filters(df: pd.DataFrame, config: dict,
                           depth_column: str = 'event_z_transformed',
                           phase_column: str = 'phase',
                           data_type: str = "data") -> pd.DataFrame:
    """
    Apply both depth and phase filtering to analysis data.

    Args:
        df: DataFrame to filter
        config: Configuration dictionary with 'analysis' section
        depth_column: Name of depth column
        phase_column: Name of phase column
        data_type: Description for logging

    Returns:
        Filtered DataFrame
    """
    logger.info(f"Applying analysis filters to {data_type}...")

    # Get filter parameters from config
    phase_type = config.get('analysis', {}).get('phase_type', 'both')
    max_depth_km = config.get('analysis', {}).get('max_depth_km', None)

    # Apply filters
    df_filtered = apply_phase_filter(df, phase_type, phase_column, data_type)
    df_filtered = apply_depth_filter(df_filtered, max_depth_km, depth_column, data_type)

    return df_filtered
