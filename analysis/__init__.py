"""
Analysis module for LATTE tomography results.
This module provides comprehensive analysis tools for travel time residuals,
velocity model comparisons, distance-dependent analysis, and source corrections.
"""

__version__ = "1.0.0"

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
    
    # Short distance analysis
    'analyze_short_distance_residuals',
    'create_short_distance_plots',
    'compute_short_distance_statistics',
    
    # Source correction analysis
    'analyze_source_corrections',
    'compute_source_corrections',
    'compute_source_statistics',
    'create_source_plots',
]

# Import main analysis functions
from .traveltime_analysis import (
    analyze_travel_time_residuals,
    compute_travel_time_statistics
)

try:
    from .traveltime_analysis import create_travel_time_plots
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

# Distance analysis imports
try:
    from .distance_analysis import (
        analyze_distance_dependent_residuals,
        create_distance_analysis_plots,
        compute_distance_statistics
    )
except ImportError:
    pass

# Short distance analysis imports
try:
    from .short_distance_analysis import (
        analyze_short_distance_residuals,
        create_short_distance_plots,
        compute_short_distance_statistics
    )
except ImportError:
    pass

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