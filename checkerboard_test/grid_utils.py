"""
Grid utility functions for velocity model processing.

This module provides functions for:
- Preparing 3D grids from DataFrames
- Converting between physical coordinates (km) and grid indices
- Parsing slice specifications
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def prepare_3d_grid(
    df: pd.DataFrame,
    velocity_field: str = "vp",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare 3D grid from DataFrame.

    Args:
        df: DataFrame with x, y, z and velocity field
        velocity_field: "vp" or "vs"

    Returns:
        (x_coords, y_coords, z_coords, velocity_3d)
    """
    # Get unique coordinates (sorted)
    x_coords = np.sort(df["x"].unique())
    y_coords = np.sort(df["y"].unique())
    z_coords = np.sort(df["z"].unique())

    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)

    logger.debug(f"Grid dimensions: {nx} × {ny} × {nz}")

    # Create 3D velocity array
    velocity_3d = np.full((nz, ny, nx), np.nan)

    # Fill the array using vectorized operations (much faster than iterrows)
    # Compute all indices at once using numpy searchsorted
    x_indices = np.searchsorted(x_coords, df["x"].values)
    y_indices = np.searchsorted(y_coords, df["y"].values)
    z_indices = np.searchsorted(z_coords, df["z"].values)

    # Get velocity values
    velocity_values = df[velocity_field].values

    # Fill the grid using advanced indexing (vectorized assignment)
    velocity_3d[z_indices, y_indices, x_indices] = velocity_values

    logger.debug(f"Filled {len(df)} grid points")

    return x_coords, y_coords, z_coords, velocity_3d


def km_to_grid_index(
    km_value: float,
    coords: np.ndarray,
    tolerance: float = 1e-6,
) -> Optional[int]:
    """
    Convert a physical coordinate (km) to grid index.

    Args:
        km_value: Physical coordinate value in km
        coords: Array of coordinate values (km) for the axis
        tolerance: Tolerance for floating point comparison (km)

    Returns:
        Grid index if km_value falls exactly on a grid point, None otherwise
    """
    # Find the closest coordinate
    distances = np.abs(coords - km_value)
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]

    # Check if it falls exactly on a grid point (within tolerance)
    if min_distance <= tolerance:
        return int(min_idx)
    else:
        logger.debug(
            f"Value {km_value:.3f} km does not fall on grid points. "
            f"Closest grid point is {coords[min_idx]:.3f} km "
            f"(distance: {min_distance:.6f} km)"
        )
        return None


def convert_km_spacing_to_indices(
    spacing_km: float,
    coords: np.ndarray,
    tolerance: float = 1e-6,
) -> List[int]:
    """
    Convert a spacing in km to a list of valid grid indices.

    Args:
        spacing_km: Spacing between slices in km
        coords: Array of coordinate values (km) for the axis
        tolerance: Tolerance for floating point comparison (km)

    Returns:
        List of valid grid indices that fall on grid points
    """
    if spacing_km <= 0:
        logger.warning(f"Invalid spacing {spacing_km} km, using all grid points")
        return list(range(len(coords)))

    # Start from the first coordinate and increment by spacing_km
    valid_indices = []
    current_km = coords[0]

    while current_km <= coords[-1]:
        idx = km_to_grid_index(current_km, coords, tolerance)
        if idx is not None:
            valid_indices.append(idx)
        current_km += spacing_km

    return valid_indices


def parse_slice_specification(
    slice_spec: Optional[List],
    coords: np.ndarray,
    unit: str = "indices",
    tolerance: float = 1e-6,
) -> List[int]:
    """
    Parse slice specification and return list of valid grid indices.

    Args:
        slice_spec: List of slice positions (either indices or km values) or None
        coords: Array of coordinate values (km) for the axis
        unit: "indices" or "km" to specify the unit of slice_spec
        tolerance: Tolerance for floating point comparison (km)

    Returns:
        List of valid grid indices
    """
    if slice_spec is None:
        return None

    if unit == "indices":
        # Already in grid indices, return as-is (filter out invalid indices)
        return [int(i) for i in slice_spec if 0 <= i < len(coords)]
    elif unit == "km":
        # Convert km values to grid indices
        valid_indices = []
        for km_value in slice_spec:
            idx = km_to_grid_index(km_value, coords, tolerance)
            if idx is not None:
                valid_indices.append(idx)
            else:
                logger.warning(
                    f"Skipping slice at {km_value:.3f} km as it does not fall on a grid point"
                )
        return valid_indices
    else:
        logger.error(f"Invalid unit '{unit}', expected 'indices' or 'km'")
        return []
