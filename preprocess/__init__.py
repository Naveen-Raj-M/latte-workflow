"""LATTE preprocessing module for QuakeML seismic data.

This module provides preprocessing functionality for seismic tomography,
specifically designed to work with QuakeML format data and HDF5 files.

Workflow:
1. Load data from QuakeML or HDF5
2. Transform coordinates (flattening transformation)
3. Create geometry and travel time files
4. Create initial velocity model and binaries (IASP91)
5. Generate visualizations

Main components:
    - data_loader: Load data from QuakeML files or HDF5
    - coordinate_transformation: Flattening transformation to Cartesian
    - geometry_traveltime: Generate geometry and travel time files
    - initial_velocity_model: Create velocity model and binaries (merged)
    - visualization: Visualize events, stations, and velocity models
    - main: Orchestration and pipeline execution

Usage:
    From command line:
        python -m preprocess.main

    With custom config:
        python -m preprocess.main \
            coordinate_transform.origin_lon=-108.0 \
            coordinate_transform.origin_lat=34.5 \
            filters.magnitude_min=2.0
"""

__version__ = "2.0.0"
__author__ = "LATTE Workflow Team"

from preprocess.data_loader import load_seismic_data
from preprocess.coordinate_transformation import (
    transform_coordinates,
    flattening_transformation
)
from preprocess.geometry_traveltime import create_geometry_and_traveltime_files
from preprocess.initial_velocity_model import create_velocity_model
from preprocess.visualization import (
    create_station_event_visualization,
    create_velocity_model_visualization
)

__all__ = [
    'load_seismic_data',
    'transform_coordinates',
    'flattening_transformation',
    'create_geometry_and_traveltime_files',
    'create_velocity_model',
    'create_station_event_visualization',
    'create_velocity_model_visualization',
]
