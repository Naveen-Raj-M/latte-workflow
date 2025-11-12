"""
Analysis module for LATTE tomography results.
This module provides comprehensive analysis tools for travel time residuals,
velocity model comparisons, distance-dependent analysis, source corrections,
and gradient/search direction analysis.

Updated to support method comparison mode for distance and short distance analysis.
"""

__version__ = "1.1.0"
__author__ = "Naveen Raj Manoharan"

__all__ = [
    # Travel time analysis
    'analyze_travel_time_residuals',
    'create_travel_time_plots',
    'compute_travel_time_statistics',

    # Velocity analysis
    'analyze_velocity_models',
    'compute_velocity_differences',
    'compute_velocity_statistics',
    'create_velocity_comparison_plots',
    'find_extreme_velocity_profiles',
    
    # Distance analysis
    'analyze_distance_dependent_residuals',
    'create_distance_analysis_plots',
    'compute_distance_statistics',
    'load_and_merge_distance_data',
    'prepare_method_comparison_data',
    
    # Short distance analysis
    'analyze_short_distance_residuals',
    'create_short_distance_plots',
    'compute_short_distance_statistics',
    'load_and_filter_short_distance_data',
    'prepare_short_distance_comparison_data',
    
    # Source correction analysis
    'analyze_source_corrections',
    'compute_source_corrections',
    'compute_source_statistics',
    'create_source_plots',
    
    # Gradient and search direction analysis
    'analyze_gradients_and_search',
    'analyze_velocity_gradients',
    'analyze_source_gradients',
]

# Import main analysis functions
from .traveltime_analysis import analyze_travel_time_residuals

try:
    from .traveltime_plots import create_travel_time_plots
    from .traveltime_statistics import compute_travel_time_statistics
    from .traveltime_comparison import prepare_comparison_data
except ImportError:
    pass

# Velocity analysis imports
try:
    from .velocity_analysis import analyze_velocity_models
    from .velocity_statistics import (
        compute_velocity_differences,
        compute_velocity_statistics,
        compute_log_ratios
    )
    from .velocity_spatial import find_extreme_velocity_profiles
    from .velocity_plots import (
        create_basic_comparison_plots,
        create_spatial_projection_plots,
        create_profile_comparison_plots,
        create_slice_heatmap_plots,
        create_ratio_analysis_plots,
        create_domain_overview_plot
    )
    from .velocity_reports import generate_velocity_comparison_report
except ImportError as e:
    import logging
    logging.debug(f"Velocity analysis module import warning: {e}")

# Distance analysis imports (modular structure)
try:
    from .distance_analysis import analyze_distance_dependent_residuals
    from .distance_data import (
        load_and_merge_distance_data,
        prepare_method_comparison_data
    )
    from .distance_statistics import compute_distance_statistics
    from .distance_plots import create_distance_analysis_plots
    from .distance_reports import generate_distance_analysis_report
except ImportError as e:
    import logging
    logging.debug(f"Distance analysis module import warning: {e}")

# Short distance analysis imports (modular structure)
try:
    from .short_distance_analysis import analyze_short_distance_residuals
    from .short_distance_data import (
        load_and_filter_short_distance_data,
        prepare_short_distance_comparison_data
    )
    from .short_distance_statistics import compute_short_distance_statistics
    from .short_distance_plots import create_short_distance_plots
    from .short_distance_reports import generate_short_distance_report
except ImportError as e:
    import logging
    logging.debug(f"Short distance analysis module import warning: {e}")

# Source correction analysis imports
try:
    from .source_analysis import (
        analyze_source_corrections,
        compute_source_corrections,
        compute_source_statistics
    )
    from .source_plots import (
        create_correction_histograms,
        create_scatter_relationships,
        create_geographic_map,
        create_depth_analysis,
        create_correction_summary,
        create_temporal_corrections
    )
except ImportError as e:
    import logging
    logging.debug(f"Source correction analysis module import warning: {e}")

# Gradient and search direction analysis imports
try:
    from .gradient_analysis import (
        analyze_gradients_and_search,
        analyze_velocity_gradients,
        analyze_source_gradients
    )
    from .gradient_plots import (
        create_velocity_gradient_plots,
        create_source_gradient_plots,
        create_gradient_histograms,
        create_search_histograms,
        create_alignment_plots
    )
    from .gradient_reports import (
        generate_velocity_gradient_report,
        generate_source_gradient_report
    )
except ImportError as e:
    import logging
    logging.debug(f"Gradient analysis module import warning: {e}")


# Common filtering utilities for all analysis modules
import pandas as pd
from typing import Optional
import logging

analysis_logger = logging.getLogger(__name__)


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
        analysis_logger.warning(f"Depth column '{depth_column}' not found in {data_type}. Skipping depth filtering.")
        return df

    initial_count = len(df)
    filtered_df = df[df[depth_column] <= max_depth_km].copy()
    removed_count = initial_count - len(filtered_df)

    if removed_count > 0:
        analysis_logger.info(f"Depth filtering for {data_type}:")
        analysis_logger.info(f"  Maximum depth limit: {max_depth_km} km")
        analysis_logger.info(f"  Points before filtering: {initial_count}")
        analysis_logger.info(f"  Points removed (depth > {max_depth_km} km): {removed_count}")
        analysis_logger.info(f"  Points remaining: {len(filtered_df)}")
    else:
        analysis_logger.info(f"Depth filtering for {data_type}: No points exceeded {max_depth_km} km limit")

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
        analysis_logger.warning(f"Phase column '{phase_column}' not found in {data_type}. Skipping phase filtering.")
        return df

    initial_count = len(df)

    if phase_type == "p":
        filtered_df = df[df[phase_column].str.upper() == 'P'].copy()
        phase_label = "P-wave"
    elif phase_type == "s":
        filtered_df = df[df[phase_column].str.upper() == 'S'].copy()
        phase_label = "S-wave"
    else:
        analysis_logger.warning(f"Invalid phase_type: {phase_type}. Using 'both'.")
        return df

    removed_count = initial_count - len(filtered_df)

    analysis_logger.info(f"Phase filtering for {data_type}:")
    analysis_logger.info(f"  Phase type: {phase_label} only")
    analysis_logger.info(f"  Points before filtering: {initial_count}")
    analysis_logger.info(f"  Points removed: {removed_count}")
    analysis_logger.info(f"  Points remaining: {len(filtered_df)}")

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
    analysis_logger.info(f"Applying analysis filters to {data_type}...")

    # Get filter parameters from config
    phase_type = config.get('analysis', {}).get('phase_type', 'both')
    max_depth_km = config.get('analysis', {}).get('max_depth_km', None)

    # Apply filters
    df_filtered = apply_phase_filter(df, phase_type, phase_column, data_type)
    df_filtered = apply_depth_filter(df_filtered, max_depth_km, depth_column, data_type)

    return df_filtered


# Export filter functions
__all__ += [
    'apply_depth_filter',
    'apply_phase_filter',
    'apply_analysis_filters',
]