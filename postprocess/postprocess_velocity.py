"""
Post-processing functions for LATTE velocity model outputs.
Converts velocity binary files to CSV format for analysis.

The grid is regular in longitude, WGS84 geodetic latitude (degrees) and depth below mean sea
level (km), as in the spherical LATTE branch; the binaries are float32 in Fortran order
(ndepth, nlat, nlon), depth fastest. Values between grid nodes are interpolated linearly in
(depth, lat, lon).
"""

import os
from types import SimpleNamespace
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from utils.io_utils import ensure_directory_exists
from utils.velocity_io import (
    read_velocity_file, write_velocity_file, subsample_to_coarse_grid,
    resolve_grid_metadata, resolve_binary_grid, build_coarse_metadata,
    normalize_metadata, GRID_KEYS, FINE_GRID_KEYS,
)


def make_grid(nlon: int, nlat: int, ndepth: int,
              dlon: float, dlat: float, ddepth: float,
              olon: float, olat: float, odepth: float) -> SimpleNamespace:
    """Build grid parameters and the node coordinates of each axis (degrees, degrees, km)."""
    return SimpleNamespace(
        nlon=nlon, nlat=nlat, ndepth=ndepth,
        dlon=dlon, dlat=dlat, ddepth=ddepth,
        olon=olon, olat=olat, odepth=odepth,
        lon_coords=olon + np.arange(nlon) * dlon,
        lat_coords=olat + np.arange(nlat) * dlat,
        depth_coords=odepth + np.arange(ndepth) * ddepth,
    )


def format_grid(grid: SimpleNamespace) -> str:
    """Format grid parameters as a one-line summary."""
    return (f"Grid({grid.nlon}x{grid.nlat}x{grid.ndepth}, "
            f"dlon={grid.dlon:.6g} deg, dlat={grid.dlat:.6g} deg, ddepth={grid.ddepth:.6g} km, "
            f"origin=({grid.olon:.6f} deg, {grid.olat:.6f} deg, {grid.odepth:.3f} km))")


def grid_from_metadata(grid_meta: Dict) -> SimpleNamespace:
    """Build grid parameters from grid metadata (nlon..odepth).

    An axis with a single point has no spacing; it gets 0.0.

    Raises:
        ValueError: If the dimensions or origin are missing.
    """
    spacing_keys = ('dlon', 'dlat', 'ddepth')
    missing = [k for k in GRID_KEYS if k not in grid_meta and k not in spacing_keys]
    if missing:
        raise ValueError(f"Grid metadata is incomplete, missing: {missing}")
    return make_grid(
        grid_meta['nlon'], grid_meta['nlat'], grid_meta['ndepth'],
        grid_meta.get('dlon', 0.0), grid_meta.get('dlat', 0.0), grid_meta.get('ddepth', 0.0),
        grid_meta['olon'], grid_meta['olat'], grid_meta['odepth'],
    )


def get_coarse_grid(df: pd.DataFrame, source: str) -> SimpleNamespace:
    """Grid of the reference velocity model, from its metadata checked against its lon/lat/depth columns."""
    grid = grid_from_metadata(resolve_grid_metadata(df, source=source))
    print(f"Reference grid: {format_grid(grid)}")
    return grid


def coordinate_indices(values: np.ndarray, origin: float, spacing: float) -> np.ndarray:
    """0-based grid indices of coordinates on a regular grid (all zero for a single-point axis)."""
    if not spacing:
        return np.zeros(len(values), dtype=int)
    return np.round((values - origin) / spacing).astype(int)


def load_binary_file(bin_file: str, nlon: int, nlat: int, ndepth: int,
                     field_name: str, convert_to_kms: bool = True) -> np.ndarray:
    """
    Load binary file and reshape to 3D array.

    Args:
        bin_file: Path to binary file
        nlon, nlat, ndepth: Grid dimensions
        field_name: Name for logging
        convert_to_kms: Convert from m/s to km/s

    Returns:
        3D array indexed [depth, lat, lon]
    """
    if not os.path.exists(bin_file):
        raise FileNotFoundError(f"{field_name} file not found: {bin_file}")
    
    print(f"Loading {field_name}: {bin_file}")
    
    # Read binary
    data = np.fromfile(bin_file, dtype=np.float32)
    expected_size = nlon * nlat * ndepth
    
    if data.size != expected_size:
        raise ValueError(f"{field_name} size mismatch: got {data.size}, expected {expected_size}")
    
    # LATTE writes Fortran order (ndepth, nlat, nlon), depth fastest
    array_3d = data.reshape((ndepth, nlat, nlon), order='F')
    
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
    # Memory-efficient approach using searchsorted (O(N log M), no NxM matrix)
    sorted_grid = np.sort(grid_coords)
    sort_order = np.argsort(grid_coords)

    # Find insertion points in sorted grid
    insert_pos = np.searchsorted(sorted_grid, query_coords)
    insert_pos = np.clip(insert_pos, 0, len(sorted_grid) - 1)

    # Check both neighbors (left and right of insertion point)
    left_pos = np.maximum(insert_pos - 1, 0)
    right_pos = insert_pos

    left_diff = np.abs(query_coords - sorted_grid[left_pos])
    right_diff = np.abs(query_coords - sorted_grid[right_pos])

    # Pick the closer neighbor
    use_right = right_diff <= left_diff
    best_sorted_idx = np.where(use_right, right_pos, left_pos)
    min_diffs = np.where(use_right, right_diff, left_diff)

    # Map back to original (unsorted) grid indices
    min_indices = sort_order[best_sorted_idx]

    # Check if within tolerance
    is_exact = min_diffs < tolerance

    # Set indices to -1 where not within tolerance
    indices = np.where(is_exact, min_indices, -1).astype(np.int32)

    return indices, is_exact


def sample_fine_grid_at_coarse_points(df_coarse: pd.DataFrame,
                                       fine_grid: SimpleNamespace,
                                       vp_fine: np.ndarray,
                                       vs_fine: np.ndarray,
                                       auxiliary_fields: Dict) -> pd.DataFrame:
    """
    Sample fine grid at coarse grid points using coordinate matching.

    Args:
        df_coarse: Coarse grid CSV DataFrame
        fine_grid: Fine grid parameters
        vp_fine, vs_fine: Fine grid velocity arrays indexed [depth, lat, lon]
        auxiliary_fields: Dict of optional gradient/search arrays

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
    lon_coarse = df_output['lon'].values
    lat_coarse = df_output['lat'].values
    depth_coarse = df_output['depth'].values

    # Find grid indices for each dimension
    print("Finding grid point matches...")
    ilon, exact_lon = find_nearest_grid_indices(lon_coarse, fine_grid.lon_coords)
    ilat, exact_lat = find_nearest_grid_indices(lat_coarse, fine_grid.lat_coords)
    idepth, exact_depth = find_nearest_grid_indices(depth_coarse, fine_grid.depth_coords)

    # Check which points need interpolation
    exact_match = exact_lon & exact_lat & exact_depth
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
        i_exact = ilon[exact_indices]
        j_exact = ilat[exact_indices]
        k_exact = idepth[exact_indices]

        # Get velocities for all exact matches (vectorized indexing)
        if vp_fine is not None:
            vp_vals = vp_fine[k_exact, j_exact, i_exact]
            valid_vp = vp_vals > 0

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
                (fine_grid.depth_coords, fine_grid.lat_coords, fine_grid.lon_coords),
                vp_fine, method='linear', bounds_error=False, fill_value=0.0
            )
        if vs_fine is not None:
            interp_vs = RegularGridInterpolator(
                (fine_grid.depth_coords, fine_grid.lat_coords, fine_grid.lon_coords),
                vs_fine, method='linear', bounds_error=False, fill_value=0.0
            )
        
        # Create interpolators for auxiliary fields
        interp_aux = {}
        for param in ['vp', 'vs']:
            interp_aux[param] = {}
            if auxiliary_fields[param]['gradient'] is not None:
                interp_aux[param]['gradient'] = RegularGridInterpolator(
                    (fine_grid.depth_coords, fine_grid.lat_coords, fine_grid.lon_coords),
                    auxiliary_fields[param]['gradient'], method='linear', 
                    bounds_error=False, fill_value=0.0
                )
            if auxiliary_fields[param]['search'] is not None:
                interp_aux[param]['search'] = RegularGridInterpolator(
                    (fine_grid.depth_coords, fine_grid.lat_coords, fine_grid.lon_coords),
                    auxiliary_fields[param]['search'], method='linear',
                    bounds_error=False, fill_value=0.0
                )
        
        # Query points (depth, lat, lon order)
        query_pts = np.column_stack([
            depth_coarse[interp_indices],
            lat_coarse[interp_indices],
            lon_coarse[interp_indices]
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

            df_output.loc[interp_indices[valid_vp], 'vp'] = vp_interp[valid_vp]
            vp_updated += np.sum(valid_vp)

            if vp_grad_interp is not None:
                df_output.loc[interp_indices[valid_vp], 'grad_vp'] = vp_grad_interp[valid_vp]
            if vp_srch_interp is not None:
                df_output.loc[interp_indices[valid_vp], 'srch_vp'] = vp_srch_interp[valid_vp]

        # Update Vs (vectorized)
        if interp_vs is not None:
            valid_vs = vs_interp > 0

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
                        coarse_grid: SimpleNamespace,
                        fine_grid: SimpleNamespace,
                        vp_fine: np.ndarray,
                        vs_fine: np.ndarray,
                        auxiliary_fields: Dict) -> pd.DataFrame:
    """
    Expand coarse CSV to fine grid with interpolation.
    
    Creates all fine grid points, interpolates coarse CSV fields,
    then directly assigns velocity values from binary files.
    """
    print("\n=== EXPANDING TO FINE GRID ===")
    print(f"Creating {fine_grid.nlon * fine_grid.nlat * fine_grid.ndepth:,} fine grid points...")

    # Create meshgrid for fine grid
    LON, LAT, DEPTH = np.meshgrid(fine_grid.lon_coords, fine_grid.lat_coords, fine_grid.depth_coords,
                                  indexing='ij')

    # 0-based grid indices of each fine point, consistent with the lon, lat, depth flattening order
    i_fine, j_fine, k_fine = (
        index.flatten(order='C') for index in
        np.meshgrid(np.arange(fine_grid.nlon), np.arange(fine_grid.nlat), np.arange(fine_grid.ndepth),
                    indexing='ij')
    )

    df_fine = pd.DataFrame({
        'lon': LON.flatten(order='C'),
        'lat': LAT.flatten(order='C'),
        'depth': DEPTH.flatten(order='C')
    })

    # Interpolate other fields from coarse CSV
    print("Interpolating coarse CSV fields to fine grid...")

    # Fields to interpolate
    fields_to_interpolate = [col for col in df_coarse.columns if col not in ['lon', 'lat', 'depth']]

    # 0-based coarse grid indices from the coordinates
    i_coarse = coordinate_indices(df_coarse['lon'].values, coarse_grid.olon, coarse_grid.dlon)
    j_coarse = coordinate_indices(df_coarse['lat'].values, coarse_grid.olat, coarse_grid.dlat)
    k_coarse = coordinate_indices(df_coarse['depth'].values, coarse_grid.odepth, coarse_grid.ddepth)

    # Create mask for valid indices
    valid_mask = (
        (i_coarse >= 0) & (i_coarse < coarse_grid.nlon) &
        (j_coarse >= 0) & (j_coarse < coarse_grid.nlat) &
        (k_coarse >= 0) & (k_coarse < coarse_grid.ndepth)
    )

    for field in fields_to_interpolate:
        print(f"  Interpolating {field}...")

        # Build 3D array from coarse data (VECTORIZED)
        field_3d = np.zeros((coarse_grid.ndepth, coarse_grid.nlat, coarse_grid.nlon))
        field_vals = df_coarse[field].values

        # Vectorized assignment
        field_3d[k_coarse[valid_mask], j_coarse[valid_mask], i_coarse[valid_mask]] = field_vals[valid_mask]

        # Interpolate
        interpolator = RegularGridInterpolator(
            (coarse_grid.depth_coords, coarse_grid.lat_coords, coarse_grid.lon_coords),
            field_3d, method='linear', bounds_error=False, fill_value=0.0
        )

        query_pts = np.column_stack([
            df_fine['depth'].values,
            df_fine['lat'].values,
            df_fine['lon'].values
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

    # Get all velocity values at once (vectorized fancy indexing)
    vp_vals = vp_fine[k_fine, j_fine, i_fine]
    vs_vals = vs_fine[k_fine, j_fine, i_fine]

    # Update Vp (vectorized)
    valid_vp = vp_vals > 0

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


def create_velocity_csv_from_binaries(config: DictConfig) -> None:
    """Main function to create velocity CSV from binary files."""
    phase_type = config.get('phase_type', 'both')
    print("\n=== CREATING VELOCITY CSV FROM BINARIES ===")
    print(f"Phase type: {phase_type}")

    # Get performance settings from global config
    show_progress = get_performance_setting(config, 'show_progress', True)

    # Load reference velocity file (auto-detects CSV or HDF5 from extension)
    input_path = str(config.reference_velocity_file)
    use_coarse = False
    if input_path.endswith(('.h5', '.hdf5')):
        use_coarse = config.get('hdf5_options', {}).get('use_coarse_dataset', False)
    dataset = 'velocity_coarse' if use_coarse else 'velocity'
    print(f"\nLoading reference velocity: {input_path}")
    if use_coarse:
        print("  Using coarse dataset for faster processing")
    df_original = read_velocity_file(input_path, dataset=dataset, show_progress=show_progress)

    print(f"  Loaded from: {input_path}")
    print(f"  Rows: {len(df_original):,}")
    
    # Grid of the reference model: from its metadata, checked against its lon/lat/depth columns
    ref_metadata = normalize_metadata(df_original.attrs.get('metadata'))
    ref_grid_meta = resolve_grid_metadata(df_original, ref_metadata, source=Path(input_path).name)
    coarse_grid = grid_from_metadata(ref_grid_meta)
    print(f"Reference grid: {format_grid(coarse_grid)}")

    # Grid of the LATTE binaries: the metadata is the reference, the optional model_grid
    # configuration is an additional check; both are tested against the binary sizes
    config_fine_grid = config.get('model_grid', None)
    fine_grid_meta = resolve_binary_grid(
        metadata=ref_metadata, df=df_original,
        config_grid=OmegaConf.to_container(config_fine_grid) if config_fine_grid else None,
        binary_files=[config.vp_binary, config.vs_binary],
        source=Path(input_path).name
    )
    fine_grid = grid_from_metadata(fine_grid_meta)
    print(f"Fine grid: {format_grid(fine_grid)}")

    # Load binary files based on phase_type
    print(f"\nLoading binary files...")
    vp_fine = None
    vs_fine = None

    if phase_type in ['both', 'p']:
        if os.path.exists(config.vp_binary):
            vp_fine = load_binary_file(config.vp_binary, fine_grid.nlon, fine_grid.nlat, fine_grid.ndepth,
                                        "Vp", convert_to_kms=True)
        else:
            print(f"  Warning: Vp binary not found: {config.vp_binary}")
            if phase_type == 'p':
                raise FileNotFoundError(f"Vp binary required for phase_type='p' but not found: {config.vp_binary}")

    if phase_type in ['both', 's']:
        if os.path.exists(config.vs_binary):
            vs_fine = load_binary_file(config.vs_binary, fine_grid.nlon, fine_grid.nlat, fine_grid.ndepth,
                                        "Vs", convert_to_kms=True)
        else:
            print(f"  Warning: Vs binary not found: {config.vs_binary}")
            if phase_type == 's':
                raise FileNotFoundError(f"Vs binary required for phase_type='s' but not found: {config.vs_binary}")
    
    # Load auxiliary fields
    params_to_process = config.get('parameters_to_process', [])
    process_gradient = config.get('process_gradient', False)
    process_search = config.get('process_search', False)
    print(f"\nOptional fields: {params_to_process if params_to_process else 'None'}")
    print(f"  process_gradient: {process_gradient}, process_search: {process_search}")

    auxiliary_fields = {
        'vp': {'gradient': None, 'search': None},
        'vs': {'gradient': None, 'search': None}
    }

    if params_to_process and (process_gradient or process_search):
        print("Loading auxiliary fields...")
        for param in params_to_process:
            if param not in ['vp', 'vs']:
                continue

            if process_gradient:
                grad_key = f'grad_{param}_binary'
                if hasattr(config, grad_key):
                    grad_file = getattr(config, grad_key)
                    if os.path.exists(grad_file):
                        auxiliary_fields[param]['gradient'] = load_binary_file(
                            grad_file, fine_grid.nlon, fine_grid.nlat, fine_grid.ndepth,
                            f"{param.upper()} gradient", convert_to_kms=True
                        )

            if process_search:
                srch_key = f'srch_{param}_binary'
                if hasattr(config, srch_key):
                    srch_file = getattr(config, srch_key)
                    if os.path.exists(srch_file):
                        auxiliary_fields[param]['search'] = load_binary_file(
                            srch_file, fine_grid.nlon, fine_grid.nlat, fine_grid.ndepth,
                            f"{param.upper()} search", convert_to_kms=True
                        )
    
    # Get processing mode
    mode = config.get('interpolation_mode', 'sample_at_coarse')
    print(f"\nProcessing mode: {mode}")
    
    # Process based on mode
    if mode == 'expand_to_fine':
        df_output = expand_to_fine_grid(
            df_original, coarse_grid, fine_grid,
            vp_fine, vs_fine, auxiliary_fields,
        )
    else:  # sample_at_coarse
        df_output = sample_fine_grid_at_coarse_points(
            df_original, fine_grid,
            vp_fine, vs_fine, auxiliary_fields,
        )
    
    # Write output (CSV and/or HDF5)
    output_format = config.get('velocity_output_format', 'csv')
    print(f"\nWriting output in format: {output_format}")

    # Metadata written to every output (HDF5 attributes / CSV header): geography and
    # provenance carried over from the reference, plus the grid of the output
    metadata = {k: v for k, v in ref_metadata.items()
                if k not in GRID_KEYS and k not in FINE_GRID_KEYS and not k.startswith('stride_')}
    if mode == 'expand_to_fine':
        metadata.update(fine_grid_meta)
    else:
        metadata.update(ref_grid_meta)
        metadata.update({f'fine_{k}': v for k, v in fine_grid_meta.items()})
    metadata.update({
        'data_type': 'velocity',
        'interpolation_mode': mode,
        'phase_type': phase_type,
    })

    # Prepare coarse grid config if enabled
    coarse_config = None
    hdf5_opts = config.get('hdf5_options', {})
    if hdf5_opts.get('create_coarse_dataset', False):
        coarse_config = {
            'dlon_coarse': hdf5_opts.coarse_grid.dlon,
            'dlat_coarse': hdf5_opts.coarse_grid.dlat,
            'ddepth_coarse': hdf5_opts.coarse_grid.ddepth,
        }

    # Get performance settings from global config
    show_progress = get_performance_setting(config, 'show_progress', True)
    n_jobs = get_performance_setting(config, 'n_jobs', 1)

    # Write CSV if requested
    if output_format in ['csv', 'both']:
        ensure_directory_exists(os.path.dirname(config.velocity_output_csv))
        write_velocity_file(df_output, config.velocity_output_csv, format='csv',
                           metadata=metadata, show_progress=show_progress)
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
                    dlon_coarse=coarse_config['dlon_coarse'],
                    dlat_coarse=coarse_config['dlat_coarse'],
                    ddepth_coarse=coarse_config['ddepth_coarse'],
                    fine_dlon=fine_grid.dlon,
                    fine_dlat=fine_grid.dlat,
                    fine_ddepth=fine_grid.ddepth,
                    n_jobs=n_jobs,
                    show_progress=show_progress
                )

                # Write coarse CSV
                ensure_directory_exists(os.path.dirname(output_coarse_csv))
                write_velocity_file(
                    df_output_coarse, output_coarse_csv, format='csv',
                    metadata=build_coarse_metadata(df_output_coarse, metadata, coarse_config),
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

    return df_output
