"""
Distance calculation utilities for travel time analysis.

This module provides functions to calculate both hypocentral (3D) and
epicentral (2D horizontal) distances between events and stations.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List

from utils.traveltime_io import km_per_unit

logger = logging.getLogger(__name__)

# Earth's mean radius in kilometers
EARTH_RADIUS_KM = 6371.0


def get_distance_label(distance_type: str) -> str:
    """
    Get the formatted label for distance type.
    
    Args:
        distance_type: "hypocentral" or "epicentral"
        
    Returns:
        Formatted label string
    """
    if distance_type == "hypocentral":
        return "Hypocentral"
    elif distance_type == "epicentral":
        return "Epicentral"
    else:
        return distance_type.title()


def get_distance_column_name(coordinate_type: str, distance_type: str) -> str:
    """
    Get the column name for a specific distance type.
    
    Args:
        coordinate_type: "transformed" or "original"
        distance_type: "hypocentral" or "epicentral"
        
    Returns:
        Column name string
    """
    return f'distance_{coordinate_type}_{distance_type}_km'


def get_output_suffix(distance_type: str) -> str:
    """
    Get the output filename suffix for a distance type.
    
    Args:
        distance_type: "hypocentral" or "epicentral"
        
    Returns:
        Suffix string for filenames
    """
    if distance_type == "hypocentral":
        return "hypo"
    elif distance_type == "epicentral":
        return "epi"
    else:
        return distance_type


def calculate_hypocentral_distance(event_x: pd.Series, event_y: pd.Series, event_z: pd.Series,
                                   station_x: pd.Series, station_y: pd.Series, station_z: pd.Series) -> pd.Series:
    """
    Calculate 3D hypocentral distance between events and stations.
    
    Args:
        event_x, event_y, event_z: Event coordinates
        station_x, station_y, station_z: Station coordinates
        
    Returns:
        Series of hypocentral distances in km
    """
    distance = np.sqrt((event_x - station_x)**2 + 
                      (event_y - station_y)**2 + 
                      (event_z - station_z)**2)
    return distance


def calculate_epicentral_distance(event_x: pd.Series, event_y: pd.Series,
                                  station_x: pd.Series, station_y: pd.Series) -> pd.Series:
    """
    Calculate 2D epicentral distance (horizontal only) between events and stations.

    Args:
        event_x, event_y: Event horizontal coordinates
        station_x, station_y: Station horizontal coordinates

    Returns:
        Series of epicentral distances in km
    """
    distance = np.sqrt((event_x - station_x)**2 +
                      (event_y - station_y)**2)
    return distance


def haversine_distance(lat1: pd.Series, lon1: pd.Series,
                      lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """
    Calculate great circle distance between two points on Earth using the haversine formula.

    This is the shortest distance between two points on a sphere (epicentral distance).

    Args:
        lat1, lon1: Latitude and longitude of first point (event) in degrees
        lat2, lon2: Latitude and longitude of second point (station) in degrees

    Returns:
        Series of great circle distances in km
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    distance = EARTH_RADIUS_KM * c
    return distance


def spherical_hypocentral_distance(lat1: pd.Series, lon1: pd.Series, depth1: pd.Series,
                                   lat2: pd.Series, lon2: pd.Series, depth2: pd.Series) -> pd.Series:
    """
    Calculate 3D hypocentral distance using spherical earth geometry.

    Combines great circle surface distance with depth difference.

    Args:
        lat1, lon1: Event latitude and longitude in degrees
        depth1: Event depth in km (positive down)
        lat2, lon2: Station latitude and longitude in degrees
        depth2: Station depth/elevation in km (positive down)

    Returns:
        Series of 3D hypocentral distances in km
    """
    # Calculate surface distance using haversine
    surface_distance = haversine_distance(lat1, lon1, lat2, lon2)

    # Calculate depth difference
    depth_diff = depth1 - depth2

    # 3D distance using Pythagorean theorem
    # (approximation assuming small distances compared to Earth's radius)
    distance = np.sqrt(surface_distance**2 + depth_diff**2)

    return distance


def calculate_all_distances(df: pd.DataFrame, coordinate_type: str, 
                           distance_types: List[str]) -> pd.DataFrame:
    """
    Calculate all requested distance types for a dataframe.
    
    Args:
        df: Merged dataframe with event and station coordinates
        coordinate_type: "transformed" or "original"
        distance_types: List of distance types to calculate ["hypocentral", "epicentral"]
        
    Returns:
        Dataframe with distance columns added
    """
    logger.info(f"Calculating distances using {coordinate_type} coordinates")
    logger.info(f"Distance types: {distance_types}") 
    
    # Get coordinate column names based on coordinate type and what's available
    use_spherical = False

    if coordinate_type == "transformed":
        if 'event_x' in df.columns and 'station_x' in df.columns:
            # Cartesian coordinates are stored in metres (units metadata); distances are in km
            event_x, event_y, event_z, station_x, station_y, station_z = (
                df[col] * km_per_unit(df, col)
                for col in ('event_x', 'event_y', 'event_z', 'station_x', 'station_y', 'station_z')
            )
        else:
            raise KeyError("Could not find transformed coordinate columns in dataframe")
    else:  # original
        # First check for lat/lon columns (spherical coordinates)
        if 'event_longitude' in df.columns and 'station_longitude' in df.columns:
            logger.info("Using spherical earth coordinates (lat/lon) for distance calculation")
            use_spherical = True
            event_lat = df['event_latitude']
            event_lon = df['event_longitude']
            event_depth = df['event_depth_km']
            station_lat = df['station_latitude']
            station_lon = df['station_longitude']
            # Station depth/elevation - default to 0 if not present
            if 'station_depth_km' in df.columns:
                station_depth = df['station_depth_km']
            elif 'station_elevation_km' in df.columns:
                station_depth = -df['station_elevation_km']  # Convert elevation to depth
            else:
                station_depth = pd.Series(0.0, index=df.index)
                logger.info("Station depth not found, assuming 0 km")
        else:
            raise KeyError("Could not find original coordinate columns in dataframe")
    
    # Calculate each requested distance type
    for distance_type in distance_types:
        if distance_type == "hypocentral":
            if use_spherical:
                distance = spherical_hypocentral_distance(
                    event_lat, event_lon, event_depth,
                    station_lat, station_lon, station_depth
                )
            else:
                distance = calculate_hypocentral_distance(
                    event_x, event_y, event_z,
                    station_x, station_y, station_z
                )
            col_name = get_distance_column_name(coordinate_type, "hypocentral")
            df[col_name] = distance

            logger.info(f"Hypocentral distance range: {distance.min():.3f} - {distance.max():.3f} km")
            logger.info(f"Mean hypocentral distance: {distance.mean():.3f} km")

        elif distance_type == "epicentral":
            if use_spherical:
                distance = haversine_distance(
                    event_lat, event_lon,
                    station_lat, station_lon
                )
            else:
                distance = calculate_epicentral_distance(
                    event_x, event_y,
                    station_x, station_y
                )
            col_name = get_distance_column_name(coordinate_type, "epicentral")
            df[col_name] = distance

            logger.info(f"Epicentral distance range: {distance.min():.3f} - {distance.max():.3f} km")
            logger.info(f"Mean epicentral distance: {distance.mean():.3f} km")

        else:
            logger.warning(f"Unknown distance type: {distance_type}, skipping")

    return df


def validate_distance_types(distance_types: List[str]) -> List[str]:
    """
    Validate and filter distance types.
    
    Args:
        distance_types: List of requested distance types
        
    Returns:
        List of valid distance types
    """
    valid_types = []
    for dt in distance_types:
        if dt in ["hypocentral", "epicentral"]:
            valid_types.append(dt)
        else:
            logger.warning(f"Invalid distance type '{dt}' ignored. Valid options: hypocentral, epicentral")
    
    if not valid_types:
        logger.warning("No valid distance types specified, defaulting to hypocentral")
        valid_types = ["hypocentral"]
    
    return valid_types