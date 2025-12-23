"""Coordinate transformation functions for LATTE preprocessing.

This module provides flattening transformation to convert geodetic coordinates
(latitude, longitude, depth) to Cartesian coordinates (x, y, z) in km.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from utils.io_utils import ensure_directory_exists

# Constants
EARTH_RADIUS_KM = 6371.0


def flattening_transformation(df: pd.DataFrame, origin_lon: float,
                              origin_lat: float, origin_depth: float,
                              invert_y_axis: bool = True) -> pd.DataFrame:
    """Apply flattening transformation from geodetic to Cartesian coordinates.

    Transforms (longitude, latitude, depth) to (x, y, z) in km using simple
    geometric conversion relative to specified origin.

    Args:
        df: DataFrame with columns ['longitude', 'latitude', 'depth'].
            Can also accept ['lon', 'lat', 'depth'].
        origin_lon: Origin longitude in degrees.
        origin_lat: Origin latitude in degrees.
        origin_depth: Origin depth in km (positive down).
        invert_y_axis: If True, positive y when latitude decreases (going south).
                       If False, positive y when latitude increases (going north).

    Returns:
        DataFrame with added columns ['x', 'y', 'z'] in km.

    Notes:
        - x points East (positive)
        - y direction controlled by invert_y_axis parameter
        - z points Down (positive, relative to origin depth)
    """
    logger = logging.getLogger(__name__)
    df_out = df.copy()

    # Handle both naming conventions
    lon_col = 'longitude' if 'longitude' in df.columns else 'lon'
    lat_col = 'latitude' if 'latitude' in df.columns else 'lat'

    # Convert to radians
    lon_rad = np.radians(df_out[lon_col])
    lat_rad = np.radians(df_out[lat_col])
    origin_lon_rad = np.radians(origin_lon)
    origin_lat_rad = np.radians(origin_lat)

    # Apply flattening transformation
    # x: East direction (positive)
    x_km = (EARTH_RADIUS_KM * (lon_rad - origin_lon_rad) *
            np.cos(origin_lat_rad))

    # y: Direction controlled by invert_y_axis
    # If invert_y_axis=True: positive y as lat decreases (origin at top, y increases going south)
    # If invert_y_axis=False: positive y as lat increases (origin at bottom, y increases going north)
    y_multiplier = -1 if invert_y_axis else 1
    y_km = y_multiplier * (EARTH_RADIUS_KM * (lat_rad - origin_lat_rad))

    # z: Depth direction (positive down, relative to origin)
    z_km = df_out['depth'] - origin_depth

    df_out['X_grid_km'] = x_km
    df_out['Y_grid_km'] = y_km
    df_out['Z_grid_km'] = z_km

    # Also keep lowercase versions for backward compatibility
    df_out['x'] = x_km
    df_out['y'] = y_km
    df_out['z'] = z_km

    logger.debug(f"Transformed {len(df_out)} points to Cartesian coordinates")
    logger.debug(f"  x range: [{df_out['X_grid_km'].min():.2f}, {df_out['X_grid_km'].max():.2f}] km")
    logger.debug(f"  y range: [{df_out['Y_grid_km'].min():.2f}, {df_out['Y_grid_km'].max():.2f}] km")
    logger.debug(f"  z range: [{df_out['Z_grid_km'].min():.2f}, {df_out['Z_grid_km'].max():.2f}] km")

    return df_out


def compute_grid_indices(df: pd.DataFrame, grid_spacing: Tuple[float, float, float],
                         min_coords: Tuple[float, float, float]) -> pd.DataFrame:
    """Compute grid indices from Cartesian coordinates.

    Args:
        df: DataFrame with columns ['X_grid_km', 'Y_grid_km', 'Z_grid_km'] or ['x', 'y', 'z'] in km.
        grid_spacing: Tuple of (dx, dy, dz) spacing in km.
        min_coords: Tuple of (min_x, min_y, min_z) to define grid origin.

    Returns:
        DataFrame with added columns ['i', 'j', 'k'] (1-based indices).
    """
    logger = logging.getLogger(__name__)
    df_out = df.copy()

    dx, dy, dz = grid_spacing
    min_x, min_y, min_z = min_coords

    # Use capital case columns if available, otherwise lowercase
    x_col = 'X_grid_km' if 'X_grid_km' in df_out.columns else 'x'
    y_col = 'Y_grid_km' if 'Y_grid_km' in df_out.columns else 'y'
    z_col = 'Z_grid_km' if 'Z_grid_km' in df_out.columns else 'z'

    # Shift coordinates to grid origin
    x_grid = df_out[x_col] - min_x
    y_grid = df_out[y_col] - min_y
    z_grid = df_out[z_col] - min_z

    # Compute indices (1-based for Fortran compatibility)
    df_out['i'] = np.round(x_grid / dx).astype(int) + 1
    df_out['j'] = np.round(y_grid / dy).astype(int) + 1
    df_out['k'] = np.round(z_grid / dz).astype(int) + 1

    logger.debug(f"Computed grid indices:")
    logger.debug(f"  i range: [{df_out['i'].min()}, {df_out['i'].max()}]")
    logger.debug(f"  j range: [{df_out['j'].min()}, {df_out['j'].max()}]")
    logger.debug(f"  k range: [{df_out['k'].min()}, {df_out['k'].max()}]")

    return df_out


def compute_domain_bounds(events_df: pd.DataFrame, stations_df: pd.DataFrame,
                         origin_lon: float, origin_lat: float,
                         origin_depth: float) -> Tuple[Tuple[float, float, float],
                                                       Tuple[float, float, float]]:
    """Compute Cartesian domain bounds from events and stations.

    Args:
        events_df: DataFrame with event geodetic coordinates.
        stations_df: DataFrame with station geodetic coordinates.
        origin_lon: Origin longitude in degrees.
        origin_lat: Origin latitude in degrees.
        origin_depth: Origin depth in km.

    Returns:
        Tuple of (min_coords, max_coords) where each is (x, y, z) in km.
    """
    logger = logging.getLogger(__name__)

    # Transform events and stations
    events_cart = flattening_transformation(events_df, origin_lon,
                                           origin_lat, origin_depth)
    stations_cart = flattening_transformation(stations_df, origin_lon,
                                             origin_lat, origin_depth)

    # Combine for bounds calculation
    combined = pd.concat([events_cart, stations_cart], ignore_index=True)

    x_col = 'X_grid_km' if 'X_grid_km' in combined.columns else 'x'
    y_col = 'Y_grid_km' if 'Y_grid_km' in combined.columns else 'y'
    z_col = 'Z_grid_km' if 'Z_grid_km' in combined.columns else 'z'

    min_x = combined[x_col].min()
    max_x = combined[x_col].max()
    min_y = combined[y_col].min()
    max_y = combined[y_col].max()
    min_z = combined[z_col].min()
    max_z = combined[z_col].max()

    logger.info(f"Computed domain bounds (Cartesian):")
    logger.info(f"  x: [{min_x:.2f}, {max_x:.2f}] km")
    logger.info(f"  y: [{min_y:.2f}, {max_y:.2f}] km")
    logger.info(f"  z: [{min_z:.2f}, {max_z:.2f}] km")

    return (min_x, min_y, min_z), (max_x, max_y, max_z)


def transform_coordinates(config: DictConfig, events_df: pd.DataFrame,
                          stations_df: pd.DataFrame) -> Tuple[pd.DataFrame,
                                                              pd.DataFrame]:
    """Transform events and stations to Cartesian coordinates.

    Args:
        config: Full configuration.
        events_df: DataFrame with event geodetic coordinates.
        stations_df: DataFrame with station geodetic coordinates.

    Returns:
        Tuple of (events_cart_df, stations_cart_df) with Cartesian coordinates.
    """
    logger = logging.getLogger(__name__)
    coord_config = config.coordinate_transform

    if not coord_config.enabled:
        logger.info("Coordinate transformation disabled")
        return events_df, stations_df

    logger.info("Transforming coordinates to Cartesian (flattening)...")

    # Get origin
    origin_lon = coord_config.origin_lon
    origin_lat = coord_config.origin_lat
    origin_depth = coord_config.origin_depth_km
    invert_y_axis = coord_config.get('invert_y_axis', True)  # Default to True for backward compatibility

    logger.info(f"Origin: ({origin_lon}°, {origin_lat}°, {origin_depth} km)")
    logger.info(f"Y-axis mode: {'inverted (positive south)' if invert_y_axis else 'standard (positive north)'}")

    # Transform events
    logger.info(f"Transforming {len(events_df)} events...")
    events_cart = flattening_transformation(events_df, origin_lon,
                                           origin_lat, origin_depth, invert_y_axis)

    # Transform stations
    logger.info(f"Transforming {len(stations_df)} stations...")
    stations_cart = flattening_transformation(stations_df, origin_lon,
                                             origin_lat, origin_depth, invert_y_axis)

    # Save to CSV
    output_dir = Path(coord_config.output_dir)
    ensure_directory_exists(output_dir)

    events_csv = output_dir / 'events_cartesian.csv'
    stations_csv = output_dir / 'stations_cartesian.csv'

    events_cart.to_csv(events_csv, index=False)
    stations_cart.to_csv(stations_csv, index=False)

    logger.info(f"Saved transformed events: {events_csv}")
    logger.info(f"Saved transformed stations: {stations_csv}")

    return events_cart, stations_cart
