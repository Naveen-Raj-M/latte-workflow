"""
Post-processing functions for LATTE velocity model outputs.
Converts velocity binary files to CSV format for analysis.
Supports Chapman inverse transformation for spherical coordinates.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from omegaconf import DictConfig

from utils.io_utils import ensure_directory_exists


def get_grid_dimensions_from_csv(df: pd.DataFrame) -> Tuple[int, int, int]:
    """Auto-detect grid dimensions from CSV data."""
    nx = int(df['i'].max())
    ny = int(df['j'].max()) 
    nz = int(df['k'].max())
    print(f"Grid dimensions detected: NX={nx}, NY={ny}, NZ={nz}")
    return nx, ny, nz


def load_velocity_binary(bin_file: str, shape: Tuple[int, int, int], velocity_type: str) -> np.ndarray:
    """Load velocity model from binary file."""
    print(f"Loading {velocity_type} from: {bin_file}")
    
    if not os.path.exists(bin_file):
        raise FileNotFoundError(f"Binary file not found: {bin_file}")
    
    # Read binary file
    data = np.fromfile(bin_file, dtype=np.float32)
    expected_size = np.prod(shape)
    
    print(f"  Read {data.size} floats, expected {expected_size}")
    
    if data.size != expected_size:
        raise ValueError(f"Binary size mismatch: got {data.size}, expected {expected_size}")
    
    # Reshape using C order
    model = data.reshape(shape, order='C')
    
    # Convert from m/s to km/s
    model = model / 1000.0
    
    print(f"  {velocity_type} range: {model[model>0].min():.3f} - {model[model>0].max():.3f} km/s")
    print(f"  Non-zero points: {np.sum(model > 0)}")
    
    return model


def apply_chapman_inverse_transform(flattened_velocity: float, depth_km: float, earth_radius_km: float = 6371.0) -> float:
    """
    Apply Chapman (1973) inverse transformation to convert flattened velocities
    back to spherical coordinates.
    
    v_spherical = exp(-z/r) * v_flattened
    """
    # Handle the case where depth might be negative (above sea level)
    depth_positive = max(depth_km, 0.001)  # Avoid numerical issues
    
    # Chapman inverse transformation
    spherical_velocity = np.exp(-depth_positive / earth_radius_km) * flattened_velocity
    
    return spherical_velocity


def create_velocity_csv_from_binaries(config: DictConfig) -> None:
    """Create velocity CSV from binary files matching the original format."""
    print("\n=== CREATING VELOCITY CSV FROM BINARIES ===")
    
    # Load original CSV for structure
    print(f"Loading original velocity CSV: {config.original_velocity_csv}")
    df_original = pd.read_csv(config.original_velocity_csv)

    if config.grid_dimensions is None:
        print("WARNING: Auto detecting grid dimensions from input velocity CSV")
        nx, ny, nz = get_grid_dimensions_from_csv(df_original)
    else:
        nx, ny, nz = config.grid_dimensions
    
    # Load binary files
    vp_model = load_velocity_binary(config.vp_binary, (nx, ny, nz), "Vp")
    vs_model = load_velocity_binary(config.vs_binary, (nx, ny, nz), "Vs")
    
    # Create output DataFrame as copy of original
    df_output = df_original.copy()
    
    # Update velocities
    vp_updated_count = 0
    vs_updated_count = 0
    
    print(f"Updating velocities...")
    transformation_method = "Chapman inverse" if config.use_chapman_inverse else "Direct"
    print(f"Using {transformation_method} transformation")
    
    for idx, row in df_output.iterrows():
        i1, j1, k1 = int(row['i']), int(row['j']), int(row['k'])
        i0, j0, k0 = i1-1, j1-1, k1-1  # Convert to 0-based indexing
        
        if 0 <= i0 < nx and 0 <= j0 < ny and 0 <= k0 < nz:
            # Get flattened velocities from binary
            vp_flattened = vp_model[i0, j0, k0]
            vs_flattened = vs_model[i0, j0, k0]
            
            if vp_flattened > 0:
                if config.use_chapman_inverse:
                    depth_km = row['depth']
                    vp_final = apply_chapman_inverse_transform(vp_flattened, depth_km, config.earth_radius_km)
                else:
                    vp_final = vp_flattened
                
                df_output.at[idx, 'vp'] = vp_final
                vp_updated_count += 1
            
            if vs_flattened > 0:
                if config.use_chapman_inverse:
                    depth_km = row['depth']
                    vs_final = apply_chapman_inverse_transform(vs_flattened, depth_km, config.earth_radius_km)
                else:
                    vs_final = vs_flattened
                
                df_output.at[idx, 'vs'] = vs_final
                vs_updated_count += 1
    
    # Write output CSV
    ensure_directory_exists(os.path.dirname(config.velocity_output_csv))
    df_output.to_csv(config.velocity_output_csv, index=False)
    
    print(f"Velocity CSV written: {config.velocity_output_csv}")
    print(f"  Updated Vp: {vp_updated_count} points")
    print(f"  Updated Vs: {vs_updated_count} points")
    print(f"  Transformation: {transformation_method}")