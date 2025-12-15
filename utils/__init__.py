"""
LATTE Utilities Module
Common utilities used across preprocessing, postprocessing, and analysis.
"""

from .io_utils import *
from .logging_utils import setup_logging, ProcessTimer, ProgressLogger
from .distance_utils import (
    validate_distance_types,
    calculate_hypocentral_distance,
    calculate_epicentral_distance,
)
from .origin_time_utils import (
    load_origin_time_metadata,
    save_origin_time_metadata
)
from .plot_utils import (
    setup_plot_style,
    save_figure,
)

__all__ = [
    # I/O utilities
    'check_file_exists',
    'ensure_directory_exists',
    'read_text_file',
    'write_text_file',
    'read_csv_file',
    'write_csv_file',
    'save_json',
    'load_json',
    'save_pickle',
    'load_pickle',
    'save_numpy_array',
    'load_numpy_array',
    'get_file_info',
    'list_files_by_extension',
    'FileLogger',

    # Logging utilities
    'setup_logging',
    'ProcessTimer',
    'ProgressLogger',

    # Distance utilities
    'validate_distance_types',
    'calculate_hypocentral_distance',
    'calculate_epicentral_distance',

    # Origin time utilities
    'load_origin_time_metadata',
    'save_origin_time_metadata',

    # Plot utilities
    'setup_plot_style',
    'save_figure'
]