"""
Analysis module for LATTE tomography results.
This module provides comprehensive analysis tools for travel time residuals,
velocity model comparisons, distance-dependent analysis, source corrections,
and gradient/search direction analysis.

Updated to support method comparison mode for distance and short distance analysis.
"""

__version__ = "2.0.0"
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

    # Ray path analysis
    'analyze_ray_paths',
    'load_ray_paths_from_cache',
    'get_ray_path_metadata',
    'get_ray_path_statistics',
    'plot_ray_paths_projection',

    # Ray density analysis
    'analyze_ray_density',

    # Tradeoff analysis
    'analyze_tradeoff',
    'assemble_tradeoff_table',
    'create_all_tradeoff_plots',

    # Basin boundaries (for XY plane plots)
    'load_basin_boundaries',
    'plot_basin_boundaries',
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

# Ray path analysis imports
try:
    from .raypath_analysis import analyze_ray_paths
    from .raypath_data import (
        load_ray_paths_from_cache,
        get_ray_path_metadata,
        get_ray_path_statistics
    )
    from .raypath_plots import plot_ray_paths_projection
except ImportError as e:
    import logging
    logging.debug(f"Ray path analysis module import warning: {e}")

# Ray density analysis imports
try:
    from .ray_density_analysis import analyze_ray_density
except ImportError as e:
    import logging
    logging.debug(f"Ray density analysis module import warning: {e}")

# Basin boundaries imports (for XY plane plots)
try:
    from .basin_boundaries import load_basin_boundaries, plot_basin_boundaries
except ImportError as e:
    import logging
    logging.debug(f"Basin boundaries module import warning: {e}")

# Tradeoff analysis imports
try:
    from .tradeoff_analysis import analyze_tradeoff
    from .tradeoff_statistics import assemble_tradeoff_table
    from .tradeoff_plots import create_all_tradeoff_plots
except ImportError as e:
    import logging
    logging.debug(f"Tradeoff analysis module import warning: {e}")

# Import common filtering utilities
# Moved to separate module to avoid circular imports
from .filters import (
    apply_depth_filter,
    apply_phase_filter,
    apply_analysis_filters
)

# Export filter functions
__all__ += [
    'apply_depth_filter',
    'apply_phase_filter',
    'apply_analysis_filters',
]