"""
Analysis module for LATTE tomography results.

This module provides comprehensive analysis tools for travel time residuals,
velocity model comparisons, and distance-dependent analysis.
"""

__version__ = "1.0.0"

__all__ = [
    # Travel time analysis
    'analyze_travel_time_residuals',
    'create_travel_time_plots',
    'compute_travel_time_statistics',
    
    # Velocity analysis
    'analyze_velocity_models',
    'create_velocity_comparison_plots',
    'compute_velocity_statistics',
    
    # Distance analysis
    'analyze_distance_dependent_residuals',
    'create_distance_analysis_plots',
    'compute_distance_statistics',
    
    # Short distance analysis
    'analyze_short_distance_residuals',
    'create_short_distance_plots',
    'compute_short_distance_statistics',
]

# Import main analysis functions
from .travel_time_analysis import (
    analyze_travel_time_residuals,
    create_travel_time_plots,
    compute_travel_time_statistics
)

from .velocity_analysis import (
    analyze_velocity_models,
    create_velocity_comparison_plots,
    compute_velocity_statistics
)

from .distance_analysis import (
    analyze_distance_dependent_residuals,
    create_distance_analysis_plots,
    compute_distance_statistics
)

from .short_distance_analysis import (
    analyze_short_distance_residuals,
    create_short_distance_plots,
    compute_short_distance_statistics
)