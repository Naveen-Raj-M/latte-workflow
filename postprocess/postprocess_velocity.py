"""
Post-processing functions for LATTE velocity model outputs.
Converts velocity binary files to CSV format for analysis.
Fresh implementation with proper grid handling and coordinate-based sampling.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from omegaconf import DictConfig
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from utils.io_utils import ensure_directory_exists
from utils.velocity_io import read_velocity_file, write_velocity_file, subsample_to_coarse_grid


class GridParameters:
    """Container for grid parameters."""
    def __init__(self, nx: int, ny: int, nz: int, 
                 dx: float, dy: float, dz: float,
                 origin_x: float, origin_y: float, origin_z: float):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.origin_z = origin_z
        
        # Build coordinate arrays (1-based indexing)
        self.x_coords = origin_x + np.arange(nx) * dx
        self.y_coords = origin_y + np.arange(ny) * dy
        self.z_coords = origin_z + np.arange(nz) * dz
    
    def __repr__(self):
        return (f"Grid({self.nx}x{self.ny}x{self.nz}, "
                f"dx={self.dx:.3f}, dy={self.dy:.3f}, dz={self.dz:.3f}, "
                f"origin=({self.origin_x:.3f}, {self.origin_y:.3f}, {self.origin_z:.3f}))")


def detect_coarse_grid_from_csv(df: pd.DataFrame) -> Tuple[int, int, int]:
    """Detect grid dimensions from CSV."""
    nx = int(df['i'].max())
    ny = int(df['j'].max())
    nz = int(df['k'].max())
    print(f"Detected coarse grid dimensions: {nx} x {ny} x {nz}")
    return nx, ny, nz


def infer_coarse_grid_parameters(df: pd.DataFrame, 
                                  nx: Optional[int] = None,
                                  ny: Optional[int] = None,
                                  nz: Optional[int] = None) -> GridParameters:
    """
    Infer coarse grid parameters from CSV data.
    Uses provided dimensions if available, otherwise auto-detects.
    """
    # Get dimensions
    if nx is None or ny is None or nz is None:
        nx_detected, ny_detected, nz_detected = detect_coarse_grid_from_csv(df)
        nx = nx if nx is not None else nx_detected
        ny = ny if ny is not None else ny_detected
        nz = nz if nz is not None else nz_detected
    
    # Get unique sorted coordinates
    x_unique = np.sort(df['X_grid_km'].unique())
    y_unique = np.sort(df['Y_grid_km'].unique())
    z_unique = np.sort(df['Z_grid_km'].unique())
    
    # Compute spacing
    dx = np.median(np.diff(x_unique)) if len(x_unique) > 1 else 0.0
    dy = np.median(np.diff(y_unique)) if len(y_unique) > 1 else 0.0
    dz = np.median(np.diff(z_unique)) if len(z_unique) > 1 else 0.0
    
    # Get origin
    origin_x = x_unique[0]
    origin_y = y_unique[0]
    origin_z = z_unique[0]
    
    grid = GridParameters(nx, ny, nz, dx, dy, dz, origin_x, origin_y, origin_z)
    print(f"Coarse grid: {grid}")
    
    return grid


def load_binary_file(bin_file: str, nx: int, ny: int, nz: int, 
                     field_name: str, convert_to_kms: bool = True) -> np.ndarray:
    """
    Load binary file and reshape to 3D array.
    
    Args:
        bin_file: Path to binary file
        nx, ny, nz: Grid dimensions
        field_name: Name for logging
        convert_to_kms: Convert from m/s to km/s
    
    Returns:
        3D array with shape (nz, ny, nx) in C-order
    """
    if not os.path.exists(bin_file):
        raise FileNotFoundError(f"{field_name} file not found: {bin_file}")
    
    print(f"Loading {field_name}: {bin_file}")
    
    # Read binary
    data = np.fromfile(bin_file, dtype=np.float32)
    expected_size = nx * ny * nz
    
    if data.size != expected_size:
        raise ValueError(f"{field_name} size mismatch: got {data.size}, expected {expected_size}")
    
    # Reshape to (nz, ny, nx) C-order
    array_3d = data.reshape((nz, ny, nx), order='F')
    
    # Convert units if needed
    if convert_to_kms:
        array_3d = array_3d / 1000.0
        print(f"  Range: {array_3d[array_3d>0].min():.3f} - {array_3d[array_3d>0].max():.3f} km/s")
    else:
        print(f"  Range: {array_3d.min():.6e} - {array_3d.max():.6e}")
    
    print(f"  Shape: {array_3d.shape}, Non-zero: {np.sum(array_3d > 0 if convert_to_kms else np.abs(array_3d) > 1e-10)}")
    
    return array_3d


def find_nearest_grid_indices(query_coords: np.ndarray,
                               grid_coords: np.ndarray,
                               tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find nearest grid indices for query coordinates (VECTORIZED).

    Args:
        query_coords: Nx1 array of query coordinates
        grid_coords: Mx1 array of grid coordinates
        tolerance: Tolerance for exact match

    Returns:
        indices: Integer indices (or -1 if no close match)
        is_exact: Boolean array indicating exact matches
    """
    # Vectorized computation using broadcasting
    # Shape: (N, M) where N = len(query_coords), M = len(grid_coords)
    diffs = np.abs(query_coords[:, np.newaxis] - grid_coords[np.newaxis, :])

    # Find minimum distance indices for each query point
    min_indices = np.argmin(diffs, axis=1)
    min_diffs = diffs[np.arange(len(query_coords)), min_indices]

    # Check if within tolerance
    is_exact = min_diffs < tolerance

    # Set indices to -1 where not within tolerance
    indices = np.where(is_exact, min_indices, -1).astype(np.int32)

    return indices, is_exact


def sample_fine_grid_at_coarse_points(df_coarse: pd.DataFrame,
                                       fine_grid: GridParameters,
                                       vp_fine: np.ndarray,
                                       vs_fine: np.ndarray,
                                       auxiliary_fields: Dict,
                                       use_chapman: bool,
                                       earth_radius: float) -> pd.DataFrame:
    """
    Sample fine grid at coarse grid points using coordinate matching.
    
    Args:
        df_coarse: Coarse grid CSV DataFrame
        fine_grid: Fine grid parameters
        vp_fine, vs_fine: Fine grid velocity arrays (nz, ny, nx)
        auxiliary_fields: Dict of optional gradient/search arrays
        use_chapman: Apply Chapman inverse transform
        earth_radius: Earth radius for Chapman transform
    
    Returns:
        Updated DataFrame with sampled values
    """
    print("\n=== SAMPLING FINE GRID AT COARSE POINTS ===")
    
    df_output = df_coarse.copy()
    
    # Add columns for optional fields
    for param in ['vp', 'vs']:
        if auxiliary_fields[param]['gradient'] is not None:
            df_output[f'grad_{param}'] = np.nan
        if auxiliary_fields[param]['search'] is not None:
            df_output[f'srch_{param}'] = np.nan
    
    # Extract coarse coordinates
    x_coarse = df_output['X_grid_km'].values
    y_coarse = df_output['Y_grid_km'].values
    z_coarse = df_output['Z_grid_km'].values
    
    # Find grid indices for each dimension
    print("Finding grid point matches...")
    ix, exact_x = find_nearest_grid_indices(x_coarse, fine_grid.x_coords)
    iy, exact_y = find_nearest_grid_indices(y_coarse, fine_grid.y_coords)
    iz, exact_z = find_nearest_grid_indices(z_coarse, fine_grid.z_coords)
    
    # Check which points need interpolation
    exact_match = exact_x & exact_y & exact_z
    need_interp = ~exact_match
    
    n_exact = np.sum(exact_match)
    n_interp = np.sum(need_interp)
    
    print(f"  Exact matches: {n_exact}")
    print(f"  Need interpolation: {n_interp}")
    
    # Counters
    vp_updated = 0
    vs_updated = 0
    
    # Process exact matches with direct indexing (VECTORIZED)
    if n_exact > 0:
        print("Processing exact matches (vectorized)...")
        if vp_fine is None and vs_fine is None:
            print("  Warning: Both Vp and Vs binaries are missing")

        exact_indices = np.where(exact_match)[0]

        # Get grid indices for all exact matches
        i_exact = ix[exact_indices]
        j_exact = iy[exact_indices]
        k_exact = iz[exact_indices]

        # Get velocities for all exact matches (vectorized indexing)
        if vp_fine is not None:
            vp_vals = vp_fine[k_exact, j_exact, i_exact]
            valid_vp = vp_vals > 0

            if use_chapman:
                depths = df_output.loc[exact_indices, 'depth'].values
                chapman_factors = np.exp(-np.maximum(depths, 0.001) / earth_radius)
                vp_vals = vp_vals * chapman_factors

            # Update DataFrame using vectorized assignment
            df_output.loc[exact_indices[valid_vp], 'vp'] = vp_vals[valid_vp]
            vp_updated += np.sum(valid_vp)

            # Update auxiliary fields
            if auxiliary_fields['vp']['gradient'] is not None:
                grad_vals = auxiliary_fields['vp']['gradient'][k_exact, j_exact, i_exact]
                df_output.loc[exact_indices[valid_vp], 'grad_vp'] = grad_vals[valid_vp]
            if auxiliary_fields['vp']['search'] is not None:
                srch_vals = auxiliary_fields['vp']['search'][k_exact, j_exact, i_exact]
                df_output.loc[exact_indices[valid_vp], 'srch_vp'] = srch_vals[valid_vp]

        if vs_fine is not None:
            vs_vals = vs_fine[k_exact, j_exact, i_exact]
            valid_vs = vs_vals > 0

            if use_chapman:
                depths = df_output.loc[exact_indices, 'depth'].values
                chapman_factors = np.exp(-np.maximum(depths, 0.001) / earth_radius)
                vs_vals = vs_vals * chapman_factors

            # Update DataFrame using vectorized assignment
            df_output.loc[exact_indices[valid_vs], 'vs'] = vs_vals[valid_vs]
            vs_updated += np.sum(valid_vs)

            # Update auxiliary fields
            if auxiliary_fields['vs']['gradient'] is not None:
                grad_vals = auxiliary_fields['vs']['gradient'][k_exact, j_exact, i_exact]
                df_output.loc[exact_indices[valid_vs], 'grad_vs'] = grad_vals[valid_vs]
            if auxiliary_fields['vs']['search'] is not None:
                srch_vals = auxiliary_fields['vs']['search'][k_exact, j_exact, i_exact]
                df_output.loc[exact_indices[valid_vs], 'srch_vs'] = srch_vals[valid_vs]
    
    # Process points needing interpolation
    if n_interp > 0:
        print("Processing interpolated points...")
        interp_indices = np.where(need_interp)[0]

        # Create interpolators
        interp_vp = None
        interp_vs = None

        if vp_fine is not None:
            interp_vp = RegularGridInterpolator(
                (fine_grid.z_coords, fine_grid.y_coords, fine_grid.x_coords),
                vp_fine, method='linear', bounds_error=False, fill_value=0.0
            )
        if vs_fine is not None:
            interp_vs = RegularGridInterpolator(
                (fine_grid.z_coords, fine_grid.y_coords, fine_grid.x_coords),
                vs_fine, method='linear', bounds_error=False, fill_value=0.0
            )
        
        # Create interpolators for auxiliary fields
        interp_aux = {}
        for param in ['vp', 'vs']:
            interp_aux[param] = {}
            if auxiliary_fields[param]['gradient'] is not None:
                interp_aux[param]['gradient'] = RegularGridInterpolator(
                    (fine_grid.z_coords, fine_grid.y_coords, fine_grid.x_coords),
                    auxiliary_fields[param]['gradient'], method='linear', 
                    bounds_error=False, fill_value=0.0
                )
            if auxiliary_fields[param]['search'] is not None:
                interp_aux[param]['search'] = RegularGridInterpolator(
                    (fine_grid.z_coords, fine_grid.y_coords, fine_grid.x_coords),
                    auxiliary_fields[param]['search'], method='linear',
                    bounds_error=False, fill_value=0.0
                )
        
        # Query points (z, y, x order)
        query_pts = np.column_stack([
            z_coarse[interp_indices],
            y_coarse[interp_indices],
            x_coarse[interp_indices]
        ])
        
        # Interpolate (VECTORIZED)
        vp_interp = interp_vp(query_pts) if interp_vp is not None else np.zeros(len(query_pts))
        vs_interp = interp_vs(query_pts) if interp_vs is not None else np.zeros(len(query_pts))

        # Interpolate auxiliary fields
        vp_grad_interp = None
        vp_srch_interp = None
        vs_grad_interp = None
        vs_srch_interp = None

        if 'gradient' in interp_aux['vp']:
            vp_grad_interp = interp_aux['vp']['gradient'](query_pts)
        if 'search' in interp_aux['vp']:
            vp_srch_interp = interp_aux['vp']['search'](query_pts)
        if 'gradient' in interp_aux['vs']:
            vs_grad_interp = interp_aux['vs']['gradient'](query_pts)
        if 'search' in interp_aux['vs']:
            vs_srch_interp = interp_aux['vs']['search'](query_pts)

        # Update Vp (vectorized)
        if interp_vp is not None:
            valid_vp = vp_interp > 0

            if use_chapman:
                depths = df_output.loc[interp_indices, 'depth'].values
                chapman_factors = np.exp(-np.maximum(depths, 0.001) / earth_radius)
                vp_interp = vp_interp * chapman_factors

            df_output.loc[interp_indices[valid_vp], 'vp'] = vp_interp[valid_vp]
            vp_updated += np.sum(valid_vp)

            if vp_grad_interp is not None:
                df_output.loc[interp_indices[valid_vp], 'grad_vp'] = vp_grad_interp[valid_vp]
            if vp_srch_interp is not None:
                df_output.loc[interp_indices[valid_vp], 'srch_vp'] = vp_srch_interp[valid_vp]

        # Update Vs (vectorized)
        if interp_vs is not None:
            valid_vs = vs_interp > 0

            if use_chapman:
                depths = df_output.loc[interp_indices, 'depth'].values
                chapman_factors = np.exp(-np.maximum(depths, 0.001) / earth_radius)
                vs_interp = vs_interp * chapman_factors

            df_output.loc[interp_indices[valid_vs], 'vs'] = vs_interp[valid_vs]
            vs_updated += np.sum(valid_vs)

            if vs_grad_interp is not None:
                df_output.loc[interp_indices[valid_vs], 'grad_vs'] = vs_grad_interp[valid_vs]
            if vs_srch_interp is not None:
                df_output.loc[interp_indices[valid_vs], 'srch_vs'] = vs_srch_interp[valid_vs]
    
    print(f"\nUpdated Vp: {vp_updated} points")
    print(f"Updated Vs: {vs_updated} points")
    
    return df_output


def expand_to_fine_grid(df_coarse: pd.DataFrame,
                        coarse_grid: GridParameters,
                        fine_grid: GridParameters,
                        vp_fine: np.ndarray,
                        vs_fine: np.ndarray,
                        auxiliary_fields: Dict,
                        use_chapman: bool,
                        earth_radius: float) -> pd.DataFrame:
    """
    Expand coarse CSV to fine grid with interpolation.
    
    Creates all fine grid points, interpolates coarse CSV fields,
    then directly assigns velocity values from binary files.
    """
    print("\n=== EXPANDING TO FINE GRID ===")
    print(f"Creating {fine_grid.nx * fine_grid.ny * fine_grid.nz:,} fine grid points...")
    
    # Create meshgrid for fine grid
    X, Y, Z = np.meshgrid(fine_grid.x_coords, fine_grid.y_coords, fine_grid.z_coords, indexing='ij')
    
    # Create base DataFrame
    total_points = fine_grid.nx * fine_grid.ny * fine_grid.nz
    
    # Generate i, j, k indices (1-based)
    i_indices = np.tile(np.arange(1, fine_grid.nx + 1), fine_grid.ny * fine_grid.nz)
    j_indices = np.repeat(np.tile(np.arange(1, fine_grid.ny + 1), fine_grid.nz), fine_grid.nx)
    k_indices = np.repeat(np.arange(1, fine_grid.nz + 1), fine_grid.nx * fine_grid.ny)
    
    df_fine = pd.DataFrame({
        'i': i_indices,
        'j': j_indices,
        'k': k_indices,
        'X_grid_km': X.flatten(order='C'),
        'Y_grid_km': Y.flatten(order='C'),
        'Z_grid_km': Z.flatten(order='C')
    })
    
    # Interpolate other fields from coarse CSV
    print("Interpolating coarse CSV fields to fine grid...")
    
    # Fields to interpolate
    fields_to_interpolate = [col for col in df_coarse.columns 
                            if col not in ['i', 'j', 'k', 'X_grid_km', 'Y_grid_km', 'Z_grid_km']]
    
    for field in fields_to_interpolate:
        print(f"  Interpolating {field}...")

        # Build 3D array from coarse data (VECTORIZED)
        field_3d = np.zeros((coarse_grid.nz, coarse_grid.ny, coarse_grid.nx))

        # Extract indices and values as arrays
        i_coarse = df_coarse['i'].values - 1
        j_coarse = df_coarse['j'].values - 1
        k_coarse = df_coarse['k'].values - 1
        field_vals = df_coarse[field].values

        # Create mask for valid indices
        valid_mask = (
            (i_coarse >= 0) & (i_coarse < coarse_grid.nx) &
            (j_coarse >= 0) & (j_coarse < coarse_grid.ny) &
            (k_coarse >= 0) & (k_coarse < coarse_grid.nz)
        )

        # Vectorized assignment
        field_3d[k_coarse[valid_mask], j_coarse[valid_mask], i_coarse[valid_mask]] = field_vals[valid_mask]

        # Interpolate
        interpolator = RegularGridInterpolator(
            (coarse_grid.z_coords, coarse_grid.y_coords, coarse_grid.x_coords),
            field_3d, method='linear', bounds_error=False, fill_value=0.0
        )

        query_pts = np.column_stack([
            df_fine['Z_grid_km'].values,
            df_fine['Y_grid_km'].values,
            df_fine['X_grid_km'].values
        ])

        df_fine[field] = interpolator(query_pts)
    
    # Add columns for auxiliary fields
    for param in ['vp', 'vs']:
        if auxiliary_fields[param]['gradient'] is not None:
            df_fine[f'grad_{param}'] = np.nan
        if auxiliary_fields[param]['search'] is not None:
            df_fine[f'srch_{param}'] = np.nan
    
    # Now assign velocities from fine grid binaries (VECTORIZED)
    print("Assigning velocities from fine grid binaries (vectorized)...")

    vp_updated = 0
    vs_updated = 0

    # Extract indices as arrays (convert to 0-based)
    i_fine = df_fine['i'].values - 1
    j_fine = df_fine['j'].values - 1
    k_fine = df_fine['k'].values - 1

    # Get all velocity values at once (vectorized fancy indexing)
    vp_vals = vp_fine[k_fine, j_fine, i_fine]
    vs_vals = vs_fine[k_fine, j_fine, i_fine]

    # Update Vp (vectorized)
    valid_vp = vp_vals > 0

    if use_chapman:
        depths = df_fine['depth'].values
        chapman_factors = np.exp(-np.maximum(depths, 0.001) / earth_radius)
        vp_vals = vp_vals * chapman_factors

    df_fine.loc[valid_vp, 'vp'] = vp_vals[valid_vp]
    vp_updated = np.sum(valid_vp)

    # Update Vp auxiliary fields
    if auxiliary_fields['vp']['gradient'] is not None:
        grad_vp_vals = auxiliary_fields['vp']['gradient'][k_fine, j_fine, i_fine]
        df_fine.loc[valid_vp, 'grad_vp'] = grad_vp_vals[valid_vp]
    if auxiliary_fields['vp']['search'] is not None:
        srch_vp_vals = auxiliary_fields['vp']['search'][k_fine, j_fine, i_fine]
        df_fine.loc[valid_vp, 'srch_vp'] = srch_vp_vals[valid_vp]

    # Update Vs (vectorized)
    valid_vs = vs_vals > 0

    if use_chapman:
        depths = df_fine['depth'].values
        chapman_factors = np.exp(-np.maximum(depths, 0.001) / earth_radius)
        vs_vals = vs_vals * chapman_factors

    df_fine.loc[valid_vs, 'vs'] = vs_vals[valid_vs]
    vs_updated = np.sum(valid_vs)

    # Update Vs auxiliary fields
    if auxiliary_fields['vs']['gradient'] is not None:
        grad_vs_vals = auxiliary_fields['vs']['gradient'][k_fine, j_fine, i_fine]
        df_fine.loc[valid_vs, 'grad_vs'] = grad_vs_vals[valid_vs]
    if auxiliary_fields['vs']['search'] is not None:
        srch_vs_vals = auxiliary_fields['vs']['search'][k_fine, j_fine, i_fine]
        df_fine.loc[valid_vs, 'srch_vs'] = srch_vs_vals[valid_vs]

    print(f"\nUpdated Vp: {vp_updated} points")
    print(f"Updated Vs: {vs_updated} points")
    
    return df_fine


def get_performance_setting(config: DictConfig, key: str, default):
    """
    Get performance setting from global performance config or local section.

    Args:
        config: Full configuration object (parent config, not local section)
        key: Setting key (e.g., 'n_jobs', 'show_progress')
        default: Default value if not found

    Returns:
        Setting value with proper fallback hierarchy
    """
    # Check global performance section first
    if hasattr(config, '_parent') and hasattr(config._parent, 'performance'):
        perf_config = config._parent.performance
        if hasattr(perf_config, key):
            return getattr(perf_config, key)
    elif hasattr(config, 'performance'):
        perf_config = config.performance
        if hasattr(perf_config, key):
            return getattr(perf_config, key)

    # Fallback to default
    return default


def create_velocity_csv_from_binaries(config: DictConfig, phase_type: str = "both", max_depth_km: Optional[float] = None) -> None:
    """Main function to create velocity CSV from binary files."""
    print("\n=== CREATING VELOCITY CSV FROM BINARIES ===")
    print(f"Phase type: {phase_type}")
    if max_depth_km is not None:
        print(f"Max depth filter: {max_depth_km} km")

    # Get performance settings from global config
    show_progress = get_performance_setting(config, 'show_progress', True)

    # Load original velocity file (CSV or HDF5)
    # Check for HDF5 input first, fallback to CSV
    if hasattr(config, 'original_velocity_hdf5') and config.original_velocity_hdf5:
        input_path = config.original_velocity_hdf5
        use_coarse = config.get('hdf5_options', {}).get('use_coarse_dataset', False)
        dataset = 'velocity_coarse' if use_coarse else 'velocity'
        print(f"\nLoading original velocity from HDF5: {input_path}")
        if use_coarse:
            print("  Using coarse dataset for faster processing")
        df_original = read_velocity_file(input_path, dataset=dataset, show_progress=show_progress)
    else:
        input_path = config.original_velocity_csv
        print(f"\nLoading original velocity from CSV: {input_path}")
        df_original = read_velocity_file(input_path, show_progress=show_progress)

    print(f"  Loaded from: {input_path}")
    print(f"  Rows: {len(df_original):,}")
    
    # Get coarse grid parameters
    coarse_nx = config.get('coarse_grid', {}).get('nx', None) if hasattr(config, 'coarse_grid') else None
    coarse_ny = config.get('coarse_grid', {}).get('ny', None) if hasattr(config, 'coarse_grid') else None
    coarse_nz = config.get('coarse_grid', {}).get('nz', None) if hasattr(config, 'coarse_grid') else None
    
    coarse_grid = infer_coarse_grid_parameters(df_original, coarse_nx, coarse_ny, coarse_nz)
    
    # Get fine grid parameters from config
    print(f"\nFine grid configuration:")
    fine_grid = GridParameters(
        nx=config.fine_grid.nx,
        ny=config.fine_grid.ny,
        nz=config.fine_grid.nz,
        dx=config.fine_grid.dx,
        dy=config.fine_grid.dy,
        dz=config.fine_grid.dz,
        origin_x=config.fine_grid.origin_x,
        origin_y=config.fine_grid.origin_y,
        origin_z=config.fine_grid.origin_z
    )
    print(f"Fine grid: {fine_grid}")
    
    # Load binary files based on phase_type
    print(f"\nLoading binary files...")
    vp_fine = None
    vs_fine = None

    if phase_type in ['both', 'p']:
        if os.path.exists(config.vp_binary):
            vp_fine = load_binary_file(config.vp_binary, fine_grid.nx, fine_grid.ny, fine_grid.nz,
                                        "Vp", convert_to_kms=True)
        else:
            print(f"  Warning: Vp binary not found: {config.vp_binary}")
            if phase_type == 'p':
                raise FileNotFoundError(f"Vp binary required for phase_type='p' but not found: {config.vp_binary}")

    if phase_type in ['both', 's']:
        if os.path.exists(config.vs_binary):
            vs_fine = load_binary_file(config.vs_binary, fine_grid.nx, fine_grid.ny, fine_grid.nz,
                                        "Vs", convert_to_kms=True)
        else:
            print(f"  Warning: Vs binary not found: {config.vs_binary}")
            if phase_type == 's':
                raise FileNotFoundError(f"Vs binary required for phase_type='s' but not found: {config.vs_binary}")
    
    # Load auxiliary fields
    params_to_process = config.get('parameters_to_process', [])
    print(f"\nOptional fields: {params_to_process if params_to_process else 'None'}")
    
    auxiliary_fields = {
        'vp': {'gradient': None, 'search': None},
        'vs': {'gradient': None, 'search': None}
    }
    
    if params_to_process:
        print("Loading auxiliary fields...")
        for param in params_to_process:
            if param not in ['vp', 'vs']:
                continue
            
            grad_key = f'grad_{param}_binary'
            if hasattr(config, grad_key):
                grad_file = getattr(config, grad_key)
                if os.path.exists(grad_file):
                    auxiliary_fields[param]['gradient'] = load_binary_file(
                        grad_file, fine_grid.nx, fine_grid.ny, fine_grid.nz,
                        f"{param.upper()} gradient", convert_to_kms=True
                    )
            
            srch_key = f'srch_{param}_binary'
            if hasattr(config, srch_key):
                srch_file = getattr(config, srch_key)
                if os.path.exists(srch_file):
                    auxiliary_fields[param]['search'] = load_binary_file(
                        srch_file, fine_grid.nx, fine_grid.ny, fine_grid.nz,
                        f"{param.upper()} search", convert_to_kms=True
                    )
    
    # Get processing mode
    mode = config.get('interpolation_mode', 'sample_at_coarse')
    print(f"\nProcessing mode: {mode}")
    
    use_chapman = config.get('use_chapman_inverse', False)
    earth_radius = config.get('earth_radius_km', 6371.0)
    
    # Process based on mode
    if mode == 'expand_to_fine':
        df_output = expand_to_fine_grid(
            df_original, coarse_grid, fine_grid,
            vp_fine, vs_fine, auxiliary_fields,
            use_chapman, earth_radius
        )
    else:  # sample_at_coarse
        df_output = sample_fine_grid_at_coarse_points(
            df_original, fine_grid,
            vp_fine, vs_fine, auxiliary_fields,
            use_chapman, earth_radius
        )
    
    # Apply depth filtering if specified
    if max_depth_km is not None and 'Z_grid_km' in df_output.columns:
        print(f"\nApplying depth filter: max_depth_km = {max_depth_km} km")
        initial_count = len(df_output)
        df_output = df_output[df_output['Z_grid_km'] <= max_depth_km].copy()
        removed_count = initial_count - len(df_output)
        print(f"  Points before filtering: {initial_count}")
        print(f"  Points removed (depth > {max_depth_km} km): {removed_count}")
        print(f"  Points remaining: {len(df_output)}")

    # Write output (CSV and/or HDF5)
    output_format = config.get('velocity_output_format', 'csv')
    print(f"\nWriting output in format: {output_format}")

    # Prepare metadata for HDF5
    metadata = {
        'grid_dimensions': {'nx': fine_grid.nx, 'ny': fine_grid.ny, 'nz': fine_grid.nz},
        'grid_spacing': {'dx': fine_grid.dx, 'dy': fine_grid.dy, 'dz': fine_grid.dz},
        'grid_origin': {'x': fine_grid.origin_x, 'y': fine_grid.origin_y, 'z': fine_grid.origin_z},
        'use_chapman_inverse': use_chapman,
        'earth_radius_km': earth_radius,
        'interpolation_mode': mode,
        'phase_type': phase_type,
    }

    # Prepare coarse grid config if enabled
    coarse_config = None
    hdf5_opts = config.get('hdf5_options', {})
    if hdf5_opts.get('create_coarse_dataset', False):
        coarse_config = {
            'dx_coarse': hdf5_opts.coarse_grid.dx,
            'dy_coarse': hdf5_opts.coarse_grid.dy,
            'dz_coarse': hdf5_opts.coarse_grid.dz,
        }

    # Get performance settings from global config
    show_progress = get_performance_setting(config, 'show_progress', True)
    n_jobs = get_performance_setting(config, 'n_jobs', 1)

    # Write CSV if requested
    if output_format in ['csv', 'both']:
        ensure_directory_exists(os.path.dirname(config.velocity_output_csv))
        write_velocity_file(df_output, config.velocity_output_csv, format='csv',
                           show_progress=show_progress)
        print(f"CSV written: {config.velocity_output_csv}")

    # Write HDF5 if requested
    if output_format in ['hdf5', 'both']:
        ensure_directory_exists(os.path.dirname(config.velocity_output_hdf5))
        write_velocity_file(
            df_output, config.velocity_output_hdf5, format='hdf5',
            coarse_grid_config=coarse_config,
            compression=hdf5_opts.get('compression', 'gzip'),
            compression_opts=hdf5_opts.get('compression_level', 4),
            metadata=metadata,
            show_progress=show_progress,
            n_jobs=n_jobs
        )
        print(f"HDF5 written: {config.velocity_output_hdf5}")

    # Write separate coarse CSV if requested (for quick visualization)
    if hdf5_opts.get('write_coarse_csv', False) and coarse_config:
        print("\nGenerating coarse-sampled CSV for quick visualization...")
        output_coarse_csv = hdf5_opts.get('output_coarse_csv')
        if output_coarse_csv is None:
            print("Warning: write_coarse_csv is True but output_coarse_csv path not specified. Skipping.")
        else:
            try:
                # Subsample to coarse grid
                df_output_coarse = subsample_to_coarse_grid(
                    df_output,
                    dx_coarse=coarse_config['dx_coarse'],
                    dy_coarse=coarse_config['dy_coarse'],
                    dz_coarse=coarse_config['dz_coarse'],
                    fine_dx=fine_grid.dx,
                    fine_dy=fine_grid.dy,
                    fine_dz=fine_grid.dz,
                    n_jobs=n_jobs,
                    show_progress=show_progress
                )

                # Write coarse CSV
                ensure_directory_exists(os.path.dirname(output_coarse_csv))
                write_velocity_file(
                    df_output_coarse, output_coarse_csv, format='csv',
                    show_progress=show_progress
                )
                print(f"Coarse CSV written: {output_coarse_csv}")
                print(f"  Original rows: {len(df_output):,}")
                print(f"  Coarse rows: {len(df_output_coarse):,}")
                print(f"  Reduction factor: {len(df_output) / len(df_output_coarse):.1f}x")
            except Exception as e:
                print(f"Warning: Failed to generate coarse CSV: {e}")

    print(f"\n=== OUTPUT SUMMARY ===")
    print(f"  Rows: {len(df_output):,}")
    print(f"  Mode: {mode}")
    print(f"  Phase type: {phase_type}")
    print(f"  Chapman transform: {use_chapman}")