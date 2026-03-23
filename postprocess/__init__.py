"""LATTE postprocessing module for seismic tomography outputs.

This module provides postprocessing functionality for seismic tomography,
converting LATTE binary outputs to CSV format and handling source parameter updates.

Workflow:
1. Load binary velocity model outputs and convert to CSV
2. Process synthetic travel times from binary outputs
3. Update source parameters from LATTE inversion results
4. Compute and analyze ray paths
5. Generate derivative weight sum (DWS) analysis

Main components:
    - postprocess_velocity: Convert velocity binary outputs to CSV
    - postprocess_traveltime: Convert synthetic travel time binaries to CSV
    - postprocess_source: Process source parameter updates
    - postprocess_raypath: Analyze ray paths from binary outputs
    - postprocess_dws: Compute derivative weight sum
    - main: Orchestration and pipeline execution

Usage:
    From command line:
        python -m postprocess.main

    With custom config:
        python -m postprocess.main \\
            output.iteration=5 \\
            output.format=csv
"""

__version__ = "2.0.0"
__author__ = "Naveen Raj Manoharan"

from postprocess.postprocess_velocity import create_velocity_csv_from_binaries
from postprocess.postprocess_traveltime import (
    create_synthetic_travel_time_csv,
    create_synthetic_travel_time_csv_with_exchange
)
from postprocess.postprocess_source import process_source_updates

__all__ = [
    "create_velocity_csv_from_binaries",
    "create_synthetic_travel_time_csv",
    "create_synthetic_travel_time_csv_with_exchange",
    "process_source_updates",
]
