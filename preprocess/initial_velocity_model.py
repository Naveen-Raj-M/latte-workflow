"""
Velocity model processing functions for LATTE preprocessing.
Functions for creating 3D initial velocity models from 1D models.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from utils.io_utils import check_file_exists, ensure_directory_exists


def load_1d_model(filepath: str) -> List[Tuple[float, float, float]]:
    """
    Loads the 1D velocity model from a text file into memory.
    
    The function reads a file where each line contains 'depth vp vs',
    parses these values, and returns them as a list of tuples, sorted by depth.

    Args:
        filepath (str): The path to the 1D model file.

    Returns:
        List[Tuple[float, float, float]]: A list of tuples, where each
            tuple contains (depth, vp, vs). The list is sorted by depth
            in ascending order.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If a line in the file does not contain three valid numbers.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: The file '{filepath}' was not found.")
    
    model_1d = []
    print(f"Reading 1D model from: {filepath}")
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty or commented lines
                continue
            try:
                parts = line.split()
                depth = float(parts[0])
                vp = float(parts[1])
                vs = float(parts[2])
                model_1d.append((depth, vp, vs))
            except (ValueError, IndexError):
                raise ValueError(
                    f"Error parsing line {i+1} in '{filepath}'. "
                    "Expected format: depth vp vs"
                )
    
    # Sort the model by depth to ensure correct lookups
    model_1d.sort(key=lambda x: x[0])
    print("Successfully loaded and sorted 1D model.")
    return model_1d


def get_velocity_for_depth(
    target_depth: float, 
    model_1d: List[Tuple[float, float, float]]
) -> Tuple[float, float]:
    """
    Finds the appropriate Vp and Vs from the 1D model for a given depth.

    This function implements the logic to find the correct velocity layer. It
    selects the layer with the largest depth that is less than or equal to the
    target_depth. It also handles extrapolation for depths outside the model's
    range as per the specified rules.

    Args:
        target_depth (float): The depth from the 3D model grid point.
        model_1d (List[Tuple[float, float, float]]): The sorted 1D velocity
            model data.

    Returns:
        Tuple[float, float]: A tuple containing the corresponding (vp, vs).
    """
    if not model_1d:
        raise ValueError("The 1D model data is empty.")

    # Rule 2: If target_depth is shallower than the first layer, use the first layer's values.
    if target_depth < model_1d[0][0]:
        return model_1d[0][1], model_1d[0][2]

    # Rule 1: Find the best matching layer.
    # Initialize with the first layer as a fallback.
    best_vp, best_vs = model_1d[0][1], model_1d[0][2]
    for depth, vp, vs in model_1d:
        if depth <= target_depth:
            best_vp = vp
            best_vs = vs
        else:
            # Since the list is sorted, we can stop once we pass the target_depth.
            break
            
    # Rule 3 (extrapolation for deeper points) is implicitly handled by the loop.
    # The last successfully assigned (best_vp, best_vs) will be from the deepest
    # layer that is shallower than or equal to target_depth.

    return best_vp, best_vs


def create_initial_3d_model(config: DictConfig) -> None:
    """
    Creates the 3D initial model file using configuration.

    This function orchestrates the process: it loads the 1D model, then iterates
    through the 3D template file line by line. For each line, it extracts the
    spatial coordinates, finds the corresponding 1D velocity, and writes the
    new data point to the output file.

    Args:
        config: Configuration containing file paths and formatting options
    """
    try:
        # Load the 1D model once
        model_1d = load_1d_model(config.model_1d_file)

        print(f"Reading 3D template grid from: {config.template_3d_file}")
        print(f"Writing new 3D initial model to: {config.output_file}")

        # Ensure output directory exists
        ensure_directory_exists(Path(config.output_file).parent)

        with open(config.template_3d_file, 'r') as f_in, open(config.output_file, 'w') as f_out:
            for line in f_in:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                lon, lat, depth = float(parts[0]), float(parts[1]), float(parts[2])

                # Get the new Vp and Vs from the 1D model based on depth
                new_vp, new_vs = get_velocity_for_depth(depth, model_1d)

                # Write the new line using configured formatting
                precision = config.precision
                width = config.field_width
                f_out.write(
                    f"{lon:<{width}.{precision.longitude}f} "
                    f"{lat:<{width}.{precision.latitude}f} "
                    f"{depth:<{width}.{precision.depth}f} "
                    f"{new_vp:<{width}.{precision.vp}f} "
                    f"{new_vs:<{width}.{precision.vs}f}\n"
                )
        
        print("\nSuccessfully created the 3D initial velocity model.")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nOperation failed: {e}")
        raise


def validate_velocity_model(filepath: str) -> bool:
    """
    Validate a 1D velocity model file.
    
    Args:
        filepath: Path to velocity model file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        model_1d = load_1d_model(filepath)
        
        # Extract arrays for validation
        depths = np.array([d[0] for d in model_1d])
        vp = np.array([d[1] for d in model_1d])
        vs = np.array([d[2] for d in model_1d])
        
        # Check for reasonable velocity values
        if np.any(vp <= 0) or np.any(vs <= 0):
            print("Warning: Found non-positive velocity values")
            return False
            
        # Check Vp > Vs
        if np.any(vp <= vs):
            print("Warning: Found Vp <= Vs, which is geophysically unrealistic")
            return False
            
        # Check for monotonic depth increase
        if not np.all(np.diff(depths) > 0):
            print("Warning: Depths are not monotonically increasing")
            return False
            
        return True
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return False