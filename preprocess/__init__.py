"""LATTE preprocessing module.

Transforms raw seismic data (QuakeML or HDF5 catalog) into binary and tabular
inputs required by the LATTE tomography inversion kernel.

Workflow:
1. Load data from QuakeML or HDF5 and apply filters
2. Transform coordinates (geodetic → local Cartesian via flattening)
3. Create geometry and travel time files
4. Create initial 3D velocity model and Fortran-compatible binaries
5. Generate visualizations

Main components:
    - data_loader: Load events/stations/picks from QuakeML or HDF5; apply filters
    - coordinate_transformation: Geodetic to Cartesian coordinate transformation
    - geometry_traveltime: Source-receiver geometry and travel time table generation
    - initial_velocity_model: 3D velocity model creation and binary output
    - visualization: Station/event maps and velocity model plots
    - main: Pipeline orchestration

Usage:
    From command line:
        python -m preprocess.main --config-name=example

    With Hydra overrides:
        python -m preprocess.main --config-name=example \\
            coordinate_transform.origin_lon=-108.0 \\
            coordinate_transform.origin_lat=34.5 \\
            filters.magnitude_min=2.0
"""

__version__ = "3.0.0"
__author__ = "Naveen Raj Manoharan"

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
