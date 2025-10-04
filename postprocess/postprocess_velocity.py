"""
Post-processing functions for LATTE velocity model outputs.
Converts velocity binary files to CSV format for analysis.
Supports Chapman inverse transformation for spherical coordinates.
Optionally includes gradient and search direction fields.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict
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


def load_auxiliary_binary(bin_file: str, shape: Tuple[int, int, int], field_name: str, 
                         convert_to_kms: bool = False) -> Optional[np.ndarray]:
    """
    Load auxiliary binary file (gradient or search direction).
    
    Args:
        bin_file: Path to binary file
        shape: Grid dimensions (nx, ny, nz)
        field_name: Name of field for logging
        convert_to_kms: Whether to convert from m/s to km/s
        
    Returns:
        Reshaped array or None if file doesn't exist
    """
    if not os.path.exists(bin_file):
        print(f"  {field_name} file not found: {bin_file} (skipping)")
        return None
    
    print(f"  Loading {field_name} from: {bin_file}")
    
    try:
        data = np.fromfile(bin_file, dtype=np.float32)
        expected_size = np.prod(shape)
        
        if data.size != expected_size:
            print(f"    Warning: Size mismatch for {field_name}: got {data.size}, expected {expected_size} (skipping)")
            return None
        
        # Reshape using C order
        model = data.reshape(shape, order='C')
        
        # Convert units if needed
        if convert_to_kms:
            model = model / 1000.0
        
        print(f"    {field_name} range: {model.min():.6e} - {model.max():.6e}")
        print(f"    Non-zero points: {np.sum(np.abs(model) > 1e-10)}")
        
        return model
        
    except Exception as e:
        print(f"    Error loading {field_name}: {e} (skipping)")
        return None


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
    
    shape = (nx, ny, nz)
    
    # Load binary files for updated velocities (REQUIRED)
    vp_model = load_velocity_binary(config.vp_binary, shape, "Vp")
    vs_model = load_velocity_binary(config.vs_binary, shape, "Vs")
    
    # Determine which parameters to process for gradients/search
    params_to_process = config.get('parameters_to_process', [])
    print(f"\nOptional fields to process: {params_to_process if params_to_process else 'None'}")
    
    # Dictionary to store optional fields
    auxiliary_fields: Dict[str, Dict[str, Optional[np.ndarray]]] = {
        'vp': {'gradient': None, 'search': None},
        'vs': {'gradient': None, 'search': None}
    }
    
    # Load optional gradient and search direction files
    if params_to_process:
        print("\nLoading optional gradient and search direction files...")
        
        for param in params_to_process:
            if param not in ['vp', 'vs']:
                print(f"  Warning: Unknown parameter '{param}' in parameters_to_process (skipping)")
                continue
            
            print(f"\n  Processing optional fields for {param.upper()}:")
            
            # Load gradient file if specified
            grad_key = f'grad_{param}_binary'
            if hasattr(config, grad_key):
                grad_file = getattr(config, grad_key)
                auxiliary_fields[param]['gradient'] = load_auxiliary_binary(
                    grad_file, shape, f"{param.upper()} gradient", convert_to_kms=True
                )
            
            # Load search direction file if specified
            srch_key = f'srch_{param}_binary'
            if hasattr(config, srch_key):
                srch_file = getattr(config, srch_key)
                auxiliary_fields[param]['search'] = load_auxiliary_binary(
                    srch_file, shape, f"{param.upper()} search direction", convert_to_kms=True
                )
    
    # Create output DataFrame as copy of original
    df_output = df_original.copy()
    
    # Add new columns for optional fields (initialize with NaN)
    new_columns = {}
    for param in ['vp', 'vs']:
        if auxiliary_fields[param]['gradient'] is not None:
            new_columns[f'grad_{param}'] = np.nan
        if auxiliary_fields[param]['search'] is not None:
            new_columns[f'srch_{param}'] = np.nan
    
    if new_columns:
        for col_name in new_columns:
            df_output[col_name] = new_columns[col_name]
        print(f"\nAdded new columns: {list(new_columns.keys())}")
    
    # Update velocities and optional fields
    vp_updated_count = 0
    vs_updated_count = 0
    
    print(f"\nUpdating velocities and optional fields...")
    transformation_method = "Chapman inverse" if config.use_chapman_inverse else "Direct"
    print(f"Using {transformation_method} transformation")
    
    for idx, row in df_output.iterrows():
        i1, j1, k1 = int(row['i']), int(row['j']), int(row['k'])
        i0, j0, k0 = i1-1, j1-1, k1-1  # Convert to 0-based indexing
        
        if 0 <= i0 < nx and 0 <= j0 < ny and 0 <= k0 < nz:
            # Get flattened velocities from binary
            vp_flattened = vp_model[i0, j0, k0]
            vs_flattened = vs_model[i0, j0, k0]
            
            # Update Vp
            if vp_flattened > 0:
                if config.use_chapman_inverse:
                    depth_km = row['depth']
                    vp_final = apply_chapman_inverse_transform(vp_flattened, depth_km, config.earth_radius_km)
                else:
                    vp_final = vp_flattened
                
                df_output.at[idx, 'vp'] = vp_final
                vp_updated_count += 1
                
                # Update Vp gradient if available
                if auxiliary_fields['vp']['gradient'] is not None:
                    grad_vp = auxiliary_fields['vp']['gradient'][i0, j0, k0]
                    df_output.at[idx, 'grad_vp'] = grad_vp
                
                # Update Vp search direction if available
                if auxiliary_fields['vp']['search'] is not None:
                    srch_vp = auxiliary_fields['vp']['search'][i0, j0, k0]
                    df_output.at[idx, 'srch_vp'] = srch_vp
            
            # Update Vs
            if vs_flattened > 0:
                if config.use_chapman_inverse:
                    depth_km = row['depth']
                    vs_final = apply_chapman_inverse_transform(vs_flattened, depth_km, config.earth_radius_km)
                else:
                    vs_final = vs_flattened
                
                df_output.at[idx, 'vs'] = vs_final
                vs_updated_count += 1
                
                # Update Vs gradient if available
                if auxiliary_fields['vs']['gradient'] is not None:
                    grad_vs = auxiliary_fields['vs']['gradient'][i0, j0, k0]
                    df_output.at[idx, 'grad_vs'] = grad_vs
                
                # Update Vs search direction if available
                if auxiliary_fields['vs']['search'] is not None:
                    srch_vs = auxiliary_fields['vs']['search'][i0, j0, k0]
                    df_output.at[idx, 'srch_vs'] = srch_vs
    
    # Write output CSV
    ensure_directory_exists(os.path.dirname(config.velocity_output_csv))
    df_output.to_csv(config.velocity_output_csv, index=False)
    
    print(f"\n=== VELOCITY OUTPUT SUMMARY ===")
    print(f"Velocity CSV written: {config.velocity_output_csv}")
    print(f"  Updated Vp: {vp_updated_count} points")
    print(f"  Updated Vs: {vs_updated_count} points")
    print(f"  Transformation: {transformation_method}")
    
    if new_columns:
        print(f"  Additional fields added: {', '.join(new_columns.keys())}")