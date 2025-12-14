"""
LATTE - Los Alamos Traveltime Tomography based on Eikonal Equation
A comprehensive package for seismic tomography preprocessing, post-processing, and analysis.
"""

__version__ = "2.0.0"
__author__ = "Naveen Raj Manoharan"
__email__ = "nm35335@my.utexas.edu"

# Import main modules
from . import preprocess
from . import postprocess
from . import analysis
from . import utils

# Version info
version_info = tuple(map(int, __version__.split('.')))

# Main functionality
__all__ = [
    'preprocess',
    'postprocess',
    'analysis',
    'utils',
    'version_info'
]

# Optional: Add convenience imports for common functions
try:
    # Preprocessing functions
    from .preprocess import (
        run_single_function as preprocess_single_function,
        run_preprocessing_pipeline,
        transform_coordinates,
        create_initial_3d_model,
        create_geometry_and_traveltime_files,
        create_velocity_binary_files
    )
    
    # Post-processing functions
    from .postprocess import (
        create_velocity_csv_from_binaries,
        create_synthetic_travel_time_csv,
        create_synthetic_travel_time_csv_with_exchange,
        process_source_updates
    )
    
    # Analysis functions
    from .analysis import (
        analyze_gradients_and_search,
        analyze_travel_time_residuals,
        analyze_velocity_models,
        analyze_distance_dependent_residuals,
        analyze_short_distance_residuals,
        analyze_source_corrections
    )

    # Add to __all__ if successfully imported
    __all__.extend([
        # Preprocessing
        'preprocess_single_function',
        'run_preprocessing_pipeline',
        'transform_coordinates',
        'create_initial_3d_model',
        'create_geometry_and_traveltime_files',
        'create_velocity_binary_files',
        # Post-processing
        'create_velocity_csv_from_binaries',
        'create_synthetic_travel_time_csv',
        'create_synthetic_travel_time_csv_with_exchange',
        'process_source_updates',
        # Analysis
        'analyze_gradients_and_search',
        'analyze_travel_time_residuals',
        'analyze_velocity_models',
        'analyze_distance_dependent_residuals',
        'analyze_short_distance_residuals',
        'analyze_source_corrections'
    ])

except ImportError as e:
    # If imports fail, just provide basic module access
    import warnings
    warnings.warn(f"Some convenience imports failed: {e}. Use module-level imports instead.")
    pass