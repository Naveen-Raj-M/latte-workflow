"""LATTE post-processing module.

Converts LATTE inversion binary outputs into analysis-ready CSV and HDF5 files.
All five pipeline stages are independent and disabled by default.

Workflow:
1. Velocity processing — convert updated Vp/Vs binaries to CSV/HDF5
2. Travel time processing — merge ground truth and synthetic travel times
3. Source processing — extract updated hypocenter locations and origin times
4. Ray path diagnosis — build HDF5 ray path cache and compute DWS sensitivity
5. Ray path simplification — reduce ray path footprint via RDP algorithm

Main components:
    - postprocess_velocity: Velocity binary to CSV/HDF5 conversion
    - postprocess_traveltime: Synthetic travel time extraction and residual computation
    - postprocess_source: Source parameter (location + origin time) extraction
    - postprocess_raypath: Ray path tracing and HDF5 cache construction
    - postprocess_dws: Derivative Weight Sum computation from ray path cache
    - compute_ray_paths: Ray path cache management and batch computation
    - simplify_ray_paths: RDP-based ray path memory reduction
    - main: Pipeline orchestration

Usage:
    From command line:
        python -m postprocess.main --config-name=example

    With Hydra overrides:
        python -m postprocess.main --config-name=example \\
            velocity_processing.enabled=true \\
            travel_time_processing.enabled=true
"""

__version__ = "3.0.0"
__author__ = "Naveen Raj Manoharan"

from postprocess.postprocess_velocity import create_velocity_csv_from_binaries
from postprocess.postprocess_traveltime import (
    create_synthetic_travel_time_csv,
    create_synthetic_travel_time_csv_with_exchange
)
from postprocess.postprocess_source import process_source_updates
from postprocess.compute_ray_paths import (
    compute_all_ray_paths,
    load_ray_paths_from_cache,
)
from postprocess.postprocess_dws import (
    create_dws_from_ray_paths,
    compute_dws_from_precomputed_rays,
)
from postprocess.simplify_ray_paths import simplify_ray_paths_main

__all__ = [
    # Velocity processing
    "create_velocity_csv_from_binaries",
    # Travel time processing
    "create_synthetic_travel_time_csv",
    "create_synthetic_travel_time_csv_with_exchange",
    # Source processing
    "process_source_updates",
    # Ray path processing
    "compute_all_ray_paths",
    "load_ray_paths_from_cache",
    # DWS processing
    "create_dws_from_ray_paths",
    "compute_dws_from_precomputed_rays",
    # Ray path simplification
    "simplify_ray_paths_main",
]
