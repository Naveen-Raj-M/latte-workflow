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

    # Get phase_type setting
    phase_type = config.get('phase_type', 'both').lower()
    print(f"Phase type: {phase_type}")

    # Initialize models with shape (nz, ny, nx) in Fortran-order to match LATTE
    # Using natural shape that matches Fortran's real :: vp(nz, ny, nx)
    # Only initialize the models we need based on phase_type
    vp_model = None
    vs_model = None

    if phase_type in ['both', 'p']:
        vp_model = np.zeros((nz, ny, nx), dtype=np.float32, order='F')
    if phase_type in ['both', 's']:
        vs_model = np.zeros((nz, ny, nx), dtype=np.float32, order='F')

    points_processed = 0
    points_outside_grid = 0

    for _, row in df.iterrows():
        i = int(row["i"]) - 1
        j = int(row["j"]) - 1
        k = int(row["k"]) - 1
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
            if vp_model is not None:
                vp_model[k, j, i] = row['vp'] * 1000  # Convert km/s to m/s, index as [depth, y, x]
            if vs_model is not None:
                vs_model[k, j, i] = row['vs'] * 1000  # Convert km/s to m/s, index as [depth, y, x]
            points_processed += 1
        else:
            points_outside_grid += 1

    print(f"Successfully placed {points_processed} data points into the grid.")
    if points_outside_grid > 0:
        print(f"Warning: {points_outside_grid} points from the CSV were outside the defined grid boundaries and were ignored.")

    # Apply interpolation if enabled
    vp_model_filled = None
    vs_model_filled = None

    if config.interpolate:
        if vp_model is not None:
            vp_model_filled = interpolate_internal_gaps(vp_model, config.neighbor_threshold)
        if vs_model is not None:
            vs_model_filled = interpolate_internal_gaps(vs_model, config.neighbor_threshold)
    else:
        vp_model_filled = vp_model
        vs_model_filled = vs_model

    # Ensure output directories exist
    if vp_model_filled is not None:
        ensure_directory_exists(Path(config.vp_output).parent)
    if vs_model_filled is not None:
        ensure_directory_exists(Path(config.vs_output).parent)

    try:
        # Write Vp model if processing P-waves
        if phase_type in ['both', 'p'] and vp_model_filled is not None:
            print(f"\nWriting Vp model to '{config.vp_output}'...")
            vp_model_filled.astype(np.float32).flatten('F').tofile(config.vp_output)
            print("Vp model written successfully.")
            print(f"  File size: {os.path.getsize(config.vp_output)} bytes")

            # Verify the file
            vp_check = np.fromfile(config.vp_output, dtype=np.float32).reshape((nz, ny, nx), order='F')
            print(f"  Vp range: {vp_check.min():.1f} - {vp_check.max():.1f} m/s")
            print(f"  Verification: {np.allclose(vp_model_filled, vp_check)}")

        # Write Vs model if processing S-waves
        if phase_type in ['both', 's'] and vs_model_filled is not None:
            print(f"\nWriting Vs model to '{config.vs_output}'...")
            vs_model_filled.astype(np.float32).flatten('F').tofile(config.vs_output)
            print("Vs model written successfully.")
            print(f"  File size: {os.path.getsize(config.vs_output)} bytes")

            # Verify the file
            vs_check = np.fromfile(config.vs_output, dtype=np.float32).reshape((nz, ny, nx), order='F')
            print(f"  Vs range: {vs_check.min():.1f} - {vs_check.max():.1f} m/s")
            print(f"  Verification: {np.allclose(vs_model_filled, vs_check)}")

        # Print summary
        print(f"\nExpected size per file: {nx * ny * nz * 4} bytes (for float32)")
        if phase_type == 'p':
            print("NOTE: Only Vp binary created (P-wave mode)")
        elif phase_type == 's':
            print("NOTE: Only Vs binary created (S-wave mode)")

    except IOError as e:
        print(f"ERROR: Could not write to output file. {e}")
        raise

    print("\n--- Conversion Complete ---")


def create_velocity_binary_files(full_config: DictConfig) -> None:
    """
    Main function to create velocity binary files from CSV data.
    Can process both initial and final velocity models.
    Accepts full config and extracts velocity_binary section.
    """
    # Extract velocity_binary section
    config = full_config.velocity_binary

    print("--- Starting CSV to 3D Binary Velocity Model Conversion ---")
    print("Binary format: Raw binary (no record markers)")
    print("Data type: 32-bit float (float32)")
    print("Array order: C-order (row-major)")

    velocity_type = "Chapman-transformed" if config.use_chapman_velocities else "Original"
    print(f"Velocity type: {velocity_type}")
    
    try:
        # Process initial velocity model
        print("\n=== Processing Initial Velocity Model ===")
        transformed_df = load_transformed_data(config)
        nx, ny, nz = get_grid_dimensions_from_csv(transformed_df)
        print(f"Initial model grid dimensions: NX={nx}, NY={ny}, NZ={nz}")
        
        # Create temporary config for initial model
        initial_config = config.copy()
        initial_config.vp_output = config.vp_output
        initial_config.vs_output = config.vs_output
        populate_grid_and_write_bins(df=transformed_df, config=initial_config, nx=nx, ny=ny, nz=nz)
        
        # Process final velocity model if enabled
        if config.create_final_binaries:
            print("\n=== Processing Final Velocity Model ===")
            if not os.path.exists(config.final_input_csv):
                raise FileNotFoundError(f"Final velocity CSV not found: {config.final_input_csv}")
            
            # Create temporary config for final model
            final_config = config.copy()
            final_config.input_csv = config.final_input_csv
            final_config.vp_output = config.vp_final_output
            final_config.vs_output = config.vs_final_output
            
            final_df = load_transformed_data(final_config)
            final_nx, final_ny, final_nz = get_grid_dimensions_from_csv(final_df)
            print(f"Final model grid dimensions: NX={final_nx}, NY={final_ny}, NZ={final_nz}")
            populate_grid_and_write_bins(df=final_df, config=final_config, nx=final_nx, ny=final_ny, nz=final_nz)
            
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        raise