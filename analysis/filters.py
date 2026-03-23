"""
Common filtering utilities for all analysis modules.

This module provides reusable filters for phase type filtering
to avoid circular import issues.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


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
                           phase_column: str = 'phase',
                           data_type: str = "data") -> pd.DataFrame:
    """
    Apply phase filtering to analysis data.

    Args:
        df: DataFrame to filter
        config: Configuration dictionary with 'analysis' section
        phase_column: Name of phase column
        data_type: Description for logging

    Returns:
        Filtered DataFrame
    """
    logger.info(f"Applying analysis filters to {data_type}...")

    phase_type = config.get('analysis', {}).get('phase_type', 'both')
    return apply_phase_filter(df, phase_type, phase_column, data_type)
