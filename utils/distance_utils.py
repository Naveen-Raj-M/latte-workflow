"""
Distance calculation utilities for travel time analysis.

This module provides functions to calculate both hypocentral (3D) and 
epicentral (2D horizontal) distances between events and stations.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


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
    if coordinate_type == "transformed":
        if 'event_x_transformed' in df.columns and 'station_x_transformed' in df.columns:
            # Direct columns without suffixes
            event_x = df['event_x_transformed'] / 1000
            event_y = df['event_y_transformed'] / 1000
            event_z = df['event_z_transformed'] / 1000
            station_x = df['station_x_transformed'] / 1000
            station_y = df['station_y_transformed'] / 1000
            station_z = df['station_z_transformed'] / 1000
        else:
            raise KeyError("Could not find transformed coordinate columns in dataframe")
    else:  # original
        # Check if columns already exist without suffixes
        if 'event_x_km' in df.columns and 'station_x_km' in df.columns:
            event_x = df['event_x_km']
            event_y = df['event_y_km']
            event_z = df['event_z_km']
            station_x = df['station_x_km']
            station_y = df['station_y_km']
            station_z = df['station_z_km']
        # Otherwise use suffixed columns from merge
        elif 'event_x_km_y' in df.columns:
            event_x = df['event_x_km_y']
            event_y = df['event_y_km_y']
            event_z = df['event_z_km_y']
            station_x = df['station_x_km_x']
            station_y = df['station_y_km_x']
            station_z = df['station_z_km_x']
        else:
            raise KeyError("Could not find original coordinate columns in dataframe")
    
    # Calculate each requested distance type
    for distance_type in distance_types:
        if distance_type == "hypocentral":
            distance = calculate_hypocentral_distance(event_x, event_y, event_z,
                                                     station_x, station_y, station_z)
            col_name = get_distance_column_name(coordinate_type, "hypocentral")
            df[col_name] = distance
            
            logger.info(f"Hypocentral distance range: {distance.min():.3f} - {distance.max():.3f} km")
            logger.info(f"Mean hypocentral distance: {distance.mean():.3f} km")
            
        elif distance_type == "epicentral":
            distance = calculate_epicentral_distance(event_x, event_y,
                                                    station_x, station_y)
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