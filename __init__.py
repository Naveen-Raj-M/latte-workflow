"""
LATTE Tomography Toolkit
End-to-end workflow for seismic tomography preprocessing, inversion post-processing,
and results analysis using the LATTE (Los Alamos Travel Time Estimation) framework.

Modules:
    preprocess  — Load seismic data, transform coordinates, and generate LATTE inputs
    postprocess — Convert LATTE binary outputs to CSV/HDF5; compute ray paths and DWS
    analysis    — Post-inversion diagnostics: residuals, velocity comparison, coverage
    utils       — Shared I/O, logging, distance, and plot utilities
"""

__version__ = "3.0.0"
__author__ = "Naveen Raj Manoharan"
__email__ = "nm35335@my.utexas.edu"

# Import main modules
from . import preprocess
from . import postprocess
from . import analysis
from . import utils

# Version info tuple
version_info = tuple(map(int, __version__.split('.')))

__all__ = [
    'preprocess',
    'postprocess',
    'analysis',
    'utils',
    'version_info',
]

# Convenience imports — fall back gracefully if optional dependencies are absent
try:
    # Preprocessing
    from .preprocess import (
        load_seismic_data,
        transform_coordinates,
        create_geometry_and_traveltime_files,
        create_velocity_model,
    )

    # Post-processing
    from .postprocess import (
        create_velocity_csv_from_binaries,
        create_synthetic_travel_time_csv,
        create_synthetic_travel_time_csv_with_exchange,
        process_source_updates,
        compute_all_ray_paths,
        load_ray_paths_from_cache,
        create_dws_from_ray_paths,
    )

    # Analysis
    from .analysis import (
        analyze_travel_time_residuals,
        analyze_velocity_models,
        analyze_distance_dependent_residuals,
        analyze_short_distance_residuals,
        analyze_source_corrections,
        analyze_gradients_and_search,
        analyze_ray_paths,
        analyze_ray_density,
        analyze_tradeoff,
    )

    __all__.extend([
        # Preprocessing
        'load_seismic_data',
        'transform_coordinates',
        'create_geometry_and_traveltime_files',
        'create_velocity_model',
        # Post-processing
        'create_velocity_csv_from_binaries',
        'create_synthetic_travel_time_csv',
        'create_synthetic_travel_time_csv_with_exchange',
        'process_source_updates',
        'compute_all_ray_paths',
        'load_ray_paths_from_cache',
        'create_dws_from_ray_paths',
        # Analysis
        'analyze_travel_time_residuals',
        'analyze_velocity_models',
        'analyze_distance_dependent_residuals',
        'analyze_short_distance_residuals',
        'analyze_source_corrections',
        'analyze_gradients_and_search',
        'analyze_ray_paths',
        'analyze_ray_density',
        'analyze_tradeoff',
    ])

except ImportError as e:
    import warnings
    warnings.warn(
        f"Some convenience imports failed: {e}. Use module-level imports instead."
    )
