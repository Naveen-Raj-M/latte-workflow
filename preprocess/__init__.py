"""LATTE preprocessing module.

Transforms raw seismic data (QuakeML or HDF5 catalog) into binary and tabular
inputs required by the LATTE tomography inversion kernel.

Workflow:
0. Parse QuakeML once (if any enabled step needs it)
1. Optional raw data visualization and QuakeML-to-HDF5 conversion (own filters each)
2. Load data from QuakeML or HDF5 and apply the global filters
3. Transform coordinates (geodetic → local Cartesian via flattening)
4. Create geometry and travel time files
5. Create initial 3D velocity model and Fortran-compatible binaries
6. Generate final data visualizations

Main components:
    - quakeml_loader: One-time, unfiltered QuakeML parsing (optionally two catalogs)
    - catalog_filters: Apply a ``filters`` node to a parsed catalog
    - hdf5_conversion: Write the filtered catalog to HDF5
    - raw_data_visualization: Plots of the raw catalog
    - plot_settings: Unified ``plots`` settings shared by all visualizations
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
from preprocess.quakeml_loader import load_quakeml_catalog
from preprocess.hdf5_conversion import convert_to_hdf5
from preprocess.raw_data_visualization import create_raw_data_visualization
from preprocess.visualization import (
    create_station_event_visualization,
    create_velocity_model_visualization
)

__all__ = [
    'load_seismic_data',
    'load_quakeml_catalog',
    'convert_to_hdf5',
    'create_raw_data_visualization',
    'transform_coordinates',
    'flattening_transformation',
    'create_geometry_and_traveltime_files',
    'create_velocity_model',
    'create_station_event_visualization',
    'create_velocity_model_visualization',
]
