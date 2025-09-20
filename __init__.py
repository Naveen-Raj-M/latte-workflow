"""
LATTE - Lightning-fast Adaptive Tomographic Toolkit for Earth
A comprehensive package for seismic tomography preprocessing, inversion, and analysis.
"""

__version__ = "0.1.0"
__author__ = "Naveen Raj Manoharan"
__email__ = "nm35335@my.utexas.edu"

# Import main modules
from . import preprocess
from . import utils

# Version info
version_info = tuple(map(int, __version__.split('.')))

# Main functionality
__all__ = [
    'preprocess',
    'utils',
    'version_info'
]

# Optional: Add convenience imports for common functions
try:
    from .preprocess import (
        run_single_function,
        run_preprocessing_pipeline,
        transform_coordinates,
        create_initial_3d_model,
        create_geometry_and_traveltime_files,
        create_velocity_binary_files
    )
    
    # Add to __all__ if successfully imported
    __all__.extend([
        'run_single_function',
        'run_preprocessing_pipeline', 
        'transform_coordinates',
        'create_initial_3d_model',
        'create_geometry_and_traveltime_files',
        'create_velocity_binary_files'
    ])
    
except ImportError:
    # If imports fail, just provide basic module access
    pass