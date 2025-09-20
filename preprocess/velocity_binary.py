"""
Velocity binary processing functions for LATTE preprocessing.
Functions for creating binary velocity model files from CSV data.
"""

import numpy as np
import pandas as pd
import os
from scipy.ndimage import convolve
from scipy.interpolate import griddata
from typing import Tuple
from omegaconf import DictConfig
from pathlib import Path

from utils.io_utils import ensure_directory_exists


def get_grid_dimensions_from_csv(df: pd.DataFrame) -> Tuple[int, int, int]:
    """Auto-detect grid dimensions from CSV data and print warnings if necessary."""
    nx = int(df['i'].max())
    ny = int(df['j'].max()) 
    nz = int(df['k'].max())
    
    # Print warnings if dimensions differ from common expectations
    common_dims = (41, 37, 41)
    if (nx, ny, nz) != common_dims:
        print(f"\n*** GRID DIMENSION WARNING ***")
        print(f"Auto-detected dimensions: NX={nx}, NY={ny}, NZ={nz}")
        print(f"Common/expected dimensions: NX={common_dims[0]}, NY={common_dims[1]}, NZ={common_dims[2]}")
        print(f"Using auto-detected dimensions to avoid data loss.")
        print(f"*** END WARNING ***\n")
    else:
        print(f"Grid dimensions: NX={nx}, NY={ny}, NZ={nz} (matches expected)")
    
    return nx, ny, nz


def load_transformed_data(config: DictConfig) -> pd.DataFrame:
    """Load and validate CSV velocity data."""
    file_path = config.input_csv
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input CSV file not found at: {file_path}")
    
    print(f"Reading data from '{file_path}'...")
    df = pd.read_csv(file_path)
    
    # Check for Chapman-transformed velocity columns
    has_chapman_cols = 'vp_flat' in df.columns and 'vs_flat' in df.columns
    has_original_cols = 'vp_original' in df.columns and 'vs_original' in df.columns
    
    if config.use_chapman_velocities:
        if has_chapman_cols:
            print("Using Chapman-transformed velocities (vp_flat, vs_flat)")
            required_columns = {'i', 'j', 'k', 'vp', 'vs'}
        else:
            print("Warning: Chapman velocities requested but vp_flat/vs_flat columns not found.")
            print("Falling back to original velocities.")
            required_columns = {'i', 'j', 'k', 'vp', 'vs'}
    else:
        if has_original_cols:
            print("Using original (non-Chapman) velocities")
            # Replace vp/vs with original values
            df['vp'] = df['vp_original']
            df['vs'] = df['vs_original']
            required_columns = {'i', 'j', 'k', 'vp', 'vs'}
        else:
            print("Using velocities from vp/vs columns (may be Chapman-transformed)")
            required_columns = {'i', 'j', 'k', 'vp', 'vs'}
    
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV file must contain the following columns: {list(required_columns)}")
    
    # Print velocity statistics for verification
    print(f"Velocity statistics:")
    print(f"  Vp range: {df['vp'].min():.2f} - {df['vp'].max():.2f} km/s")
    print(f"  Vs range: {df['vs'].min():.2f} - {df['vs'].max():.2f} km/s")
    
    if has_chapman_cols and has_original_cols:
        print(f"Chapman velocity statistics:")
        print(f"  Vp_flat range: {df['vp_flat'].min():.2f} - {df['vp_flat'].max():.2f} km/s")
        print(f"  Vs_flat range: {df['vs_flat'].min():.2f} - {df['vs_flat'].max():.2f} km/s")
        print(f"  Vp_original range: {df['vp_original'].min():.2f} - {df['vp_original'].max():.2f} km/s")
        print(f"  Vs_original range: {df['vs_original'].min():.2f} - {df['vs_original'].max():.2f} km/s")
    
    print("Successfully loaded CSV data.")
    return df


def interpolate_internal_gaps(model: np.ndarray, neighbor_threshold: int) -> np.ndarray:
    """Interpolate internal gaps in the velocity model."""
    known_indices = np.array(np.nonzero(model)).T
    if known_indices.shape[0] < 4:
        print("Warning: Not enough data points to perform interpolation. Skipping.")
        return model
    
    known_values = model[tuple(known_indices.T)]
    binary_mask = model > 0
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    neighbor_counts = convolve(binary_mask.astype(np.uint8), kernel, mode='constant', cval=0)
    is_target_for_interpolation = (model == 0) & (neighbor_counts >= neighbor_threshold)
    target_indices = np.array(np.nonzero(is_target_for_interpolation)).T
    
    if target_indices.shape[0] == 0:
        print("No internal gaps found to interpolate. Model is already dense.")
        return model
    
    print(f"Found {target_indices.shape[0]} internal grid points to interpolate.")
    interpolated_values = griddata(known_indices, known_values, target_indices, method='nearest')
    filled_model = model.copy()
    valid_interpolations = ~np.isnan(interpolated_values)
    fill_indices = target_indices[valid_interpolations]
    fill_values = interpolated_values[valid_interpolations]
    filled_model[tuple(fill_indices.T)] = fill_values
    print(f"Successfully filled {len(fill_values)} points.")
    return filled_model


def populate_grid_and_write_bins(df: pd.DataFrame, config: DictConfig, nx: int, ny: int, nz: int) -> None:
    """Populate 3D model grid from CSV data and write binary files."""
    print("\nPopulating 3D model grid from CSV data...")
    
    # Initialize models with auto-detected dimensions
    vp_model = np.zeros((nx, ny, nz), dtype=np.float32, order='F')
    vs_model = np.zeros((nx, ny, nz), dtype=np.float32, order='F')
    
    points_processed = 0
    points_outside_grid = 0
    
    for _, row in df.iterrows():
        i = int(row["i"]) - 1
        j = int(row["j"]) - 1
        k = int(row["k"]) - 1
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
            vp_model[i, j, k] = row['vp'] * 1000  # Convert km/s to m/s
            vs_model[i, j, k] = row['vs'] * 1000  # Convert km/s to m/s
            points_processed += 1
        else:
            points_outside_grid += 1
    
    print(f"Successfully placed {points_processed} data points into the grid.")
    if points_outside_grid > 0:
        print(f"Warning: {points_outside_grid} points from the CSV were outside the defined grid boundaries and were ignored.")

    # Apply interpolation if enabled
    if config.interpolate:
        vp_model_filled = interpolate_internal_gaps(vp_model, config.neighbor_threshold)
        vs_model_filled = interpolate_internal_gaps(vs_model, config.neighbor_threshold)
    else:
        vp_model_filled = vp_model
        vs_model_filled = vs_model

    # Ensure output directories exist
    ensure_directory_exists(Path(config.vp_output).parent)
    ensure_directory_exists(Path(config.vs_output).parent)

    try:
        # Write Vp model (Fortran stream access format)
        print(f"\nWriting final Vp model to '{config.vp_output}'...")
        vp_model_filled.astype(np.float32).tofile(config.vp_output)
        print("Vp model written successfully.")

        # Write Vs model (Fortran stream access format)
        print(f"Writing final Vs model to '{config.vs_output}'...")
        vs_model_filled.astype(np.float32).tofile(config.vs_output)
        print("Vs model written successfully.")

        # Print verification information
        print(f"\nFile verification:")
        print(f"Vp file size: {os.path.getsize(config.vp_output)} bytes")
        print(f"Vs file size: {os.path.getsize(config.vs_output)} bytes")
        print(f"Expected size: {nx * ny * nz * 4} bytes (for float32)")
        
        # Verify the files can be read back correctly
        print(f"\nVerification test:")
        vp_check = np.fromfile(config.vp_output, dtype=np.float32).reshape((nx, ny, nz), order='F')
        vs_check = np.fromfile(config.vs_output, dtype=np.float32).reshape((nx, ny, nz), order='F')
        print(f"Vp range: {vp_check.min():.1f} - {vp_check.max():.1f} m/s")
        print(f"Vs range: {vs_check.min():.1f} - {vs_check.max():.1f} m/s")
        print(f"Arrays match original: Vp={np.allclose(vp_model_filled, vp_check)}, Vs={np.allclose(vs_model_filled, vs_check)}")

    except IOError as e:
        print(f"ERROR: Could not write to output file. {e}")
        raise

    print("\n--- Conversion Complete ---")


def create_velocity_binary_files(config: DictConfig) -> None:
    """
    Main function to create velocity binary files from CSV data.
    """
    print("--- Starting CSV to 3D Binary Velocity Model Conversion (Chapman Workflow) ---")
    print("Binary format: Fortran stream access (no record markers)")
    print("Data type: 32-bit float")
    print("Array order: Fortran column-major")
    
    velocity_type = "Chapman-transformed" if config.use_chapman_velocities else "Original"
    print(f"Velocity type: {velocity_type}")
    
    try:
        transformed_df = load_transformed_data(config)
        nx, ny, nz = get_grid_dimensions_from_csv(transformed_df)
        print(f"Final grid dimensions: NX={nx}, NY={ny}, NZ={nz}")
        populate_grid_and_write_bins(df=transformed_df, config=config, nx=nx, ny=ny, nz=nz)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        raise