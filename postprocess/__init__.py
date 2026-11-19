"""LATTE post-processing module.

Converts the outputs of the spherical LATTE branch into analysis-ready CSV and HDF5 files.
Positions are longitude and WGS84 geodetic latitude (degrees) and depth below mean sea level
(km). All three pipeline stages are independent and disabled by default.

Workflow:
1. Velocity processing — convert updated Vp/Vs binaries to CSV/HDF5
2. Travel time processing — merge ground truth and synthetic travel times
3. Source processing — extract updated hypocenter locations (slon, slat, sdepth) and origin times

Main components:
    - postprocess_velocity: Velocity binary to CSV/HDF5 conversion
    - postprocess_traveltime: Synthetic travel time extraction and residual computation
    - postprocess_source: Source parameter (location + origin time) extraction
    - main: Pipeline orchestration

Usage:
    From command line:
        python -m postprocess.main --config-name=example

    With Hydra overrides:
        python -m postprocess.main --config-name=example \\
            velocity_processing.enabled=true \\
            travel_time_processing.enabled=true
"""

__version__ = "4.0.0"
__author__ = "Naveen Raj Manoharan"

from postprocess.postprocess_velocity import create_velocity_csv_from_binaries
from postprocess.postprocess_traveltime import (
    create_synthetic_travel_time_csv,
    create_synthetic_travel_time_csv_with_exchange
)
from postprocess.postprocess_source import process_source_updates

__all__ = [
    # Velocity processing
    "create_velocity_csv_from_binaries",
    # Travel time processing
    "create_synthetic_travel_time_csv",
    "create_synthetic_travel_time_csv_with_exchange",
    # Source processing
    "process_source_updates",
]
