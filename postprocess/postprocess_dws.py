"""
Post-processing functions for computing Derivative Weight Sum (DWS) from ray paths.

DWS is computed by summing the squared ray path lengths in each cell, weighted by
observation weights from the ground truth traveltime data. DWS provides a measure
of ray path coverage and data sensitivity in the velocity model.

This module:
1. Traces ray paths from receivers to sources using gradient fields
2. Computes path length contributions in each grid cell
3. Weights contributions by observation weights
4. Saves DWS values in HDF5 format matching velocity model structure
"""

import os
import numpy as np
import pandas as pd
import h5py
from typing import Dict, Tuple, Optional
from omegaconf import DictConfig
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings

from utils.io_utils import ensure_directory_exists
from utils.velocity_io import write_velocity_file, VELOCITY_COLUMNS
from .postprocess_raypath import (
    read_gradient_metadata,
    read_gradient_binaries,
    reshape_gradients,
    trace_ray_path
)


def compute_ray_segment_contributions(
    ray_path: np.ndarray,
    grid_dims: Dict[str, float],
    is_3d: bool
) -> Dict[Tuple[int, int, int], float]:
    """
    Compute the contribution of a ray path to each grid cell it passes through.

    A cell is defined as a cube of dx/2, dy/2, dz/2 around each grid point.
    For each segment of the ray, we compute which grid cells it intersects
    and the length of the segment within each cell.

    Args:
        ray_path: Array of ray coordinates (n_points, 3) in km
        grid_dims: Grid dimensions dictionary (nx, ny, nz, dx, dy, dz, origin_x, origin_y, origin_z)
        is_3d: Whether this is a 3D problem

    Returns:
        Dictionary mapping (i, j, k) grid indices to total ray length in that cell (km)
    """
    nx, ny, nz = grid_dims['nx'], grid_dims['ny'], grid_dims['nz']
    dx, dy, dz = grid_dims['dx'], grid_dims['dy'], grid_dims['dz']
    ox, oy, oz = grid_dims['origin_x'], grid_dims['origin_y'], grid_dims['origin_z']

    # Cell half-widths
    dx_half = dx / 2.0
    dy_half = dy / 2.0
    dz_half = dz / 2.0

    cell_contributions = {}

    # Process ray segments
    for i in range(len(ray_path) - 1):
        p1 = ray_path[i]
        p2 = ray_path[i + 1]

        # Segment length
        segment_length = np.linalg.norm(p2 - p1)

        if segment_length < 1e-10:
            continue

        # Midpoint of segment
        midpoint = (p1 + p2) / 2.0

        # Find which grid cell contains the midpoint
        # Grid cells are centered at grid points with half-width cells
        if is_3d:
            i_cell = int(np.round((midpoint[0] - ox) / dx))
            j_cell = int(np.round((midpoint[1] - oy) / dy))
            k_cell = int(np.round((midpoint[2] - oz) / dz))

            # Check bounds
            if 0 <= i_cell < nx and 0 <= j_cell < ny and 0 <= k_cell < nz:
                cell_key = (i_cell, j_cell, k_cell)
                cell_contributions[cell_key] = cell_contributions.get(cell_key, 0.0) + segment_length
        else:
            # 2D case
            i_cell = int(np.round((midpoint[0] - ox) / dx))
            k_cell = int(np.round((midpoint[2] - oz) / dz))

            # In 2D, j is fixed at 0
            j_cell = 0

            # Check bounds
            if 0 <= i_cell < nx and 0 <= k_cell < nz:
                cell_key = (i_cell, j_cell, k_cell)
                cell_contributions[cell_key] = cell_contributions.get(cell_key, 0.0) + segment_length

    return cell_contributions


def process_single_ray_dws(
    shot_id: int,
    source_pos: Tuple[float, float, float],
    receiver_pos: Tuple[float, float, float],
    weight: float,
    gradient_dir: str,
    phase: str,
    precision: str,
    tracing_params: Dict,
    interpolation_method: str,
    grid_dims: Dict[str, int]
) -> Optional[Dict[Tuple[int, int, int], float]]:
    """
    Process a single shot-receiver pair and compute its DWS contribution.

    Args:
        shot_id: Shot ID
        source_pos: Source position (x, y, z) in km
        receiver_pos: Receiver position (x, y, z) in km
        weight: Observation weight from traveltime data
        gradient_dir: Directory containing gradient files
        phase: Wave phase ('p' or 's')
        precision: Numerical precision ('single' or 'double')
        tracing_params: Ray tracing parameters dictionary
        interpolation_method: 'nearest' or 'trilinear' interpolation method
        grid_dims: Global grid dimensions (for validation)

    Returns:
        Dictionary mapping (i, j, k) to weighted squared path length contribution
    """
    try:
        # Read gradient metadata to get the local grid parameters for this shot
        metadata = read_gradient_metadata(gradient_dir, shot_id)

        # Create local grid_dims from metadata
        local_grid_dims = {
            'nx': metadata['nx'],
            'ny': metadata['ny'],
            'nz': metadata['nz'],
            'dx': metadata['dx'],
            'dy': metadata['dy'],
            'dz': metadata['dz'],
            'origin_x': metadata['origin_x'],
            'origin_y': metadata['origin_y'],
            'origin_z': metadata['origin_z']
        }

        # Read gradient files
        grad_x, grad_y, grad_z = read_gradient_binaries(
            gradient_dir, shot_id, phase, precision
        )

        # Reshape gradients using the local grid dimensions
        grad_x, grad_y, grad_z = reshape_gradients(
            grad_x, grad_y, grad_z, local_grid_dims
        )

        # Determine if 3D
        is_3d = grad_y is not None

        # Trace ray path
        ray_path = trace_ray_path(
            source_pos,
            receiver_pos,
            grad_x, grad_y, grad_z,
            local_grid_dims,
            max_steps=tracing_params['max_steps'],
            step_size=tracing_params['step_size'],
            tolerance=tracing_params['tolerance'],
            interpolation_method=interpolation_method
        )

        if len(ray_path) <= 1:
            return None

        # Compute segment contributions for this ray
        cell_contributions = compute_ray_segment_contributions(
            ray_path, local_grid_dims, is_3d
        )

        # Weight contributions by observation weight and square the lengths
        # DWS = sum(weight * length^2) for each cell
        weighted_contributions = {
            cell_idx: weight * (length ** 2)
            for cell_idx, length in cell_contributions.items()
        }

        return weighted_contributions

    except Exception:
        return None


def compute_dws_from_ray_paths(config: DictConfig) -> np.ndarray:
    """
    Compute DWS (Derivative Weight Sum) from ray paths.

    Args:
        config: Configuration object containing all parameters

    Returns:
        3D numpy array of DWS values (nz, ny, nx)
    """
    print("\n" + "="*80)
    print("COMPUTING DWS (DERIVATIVE WEIGHT SUM) FROM RAY PATHS")
    print("="*80)

    # Load ground truth data
    print(f"\nLoading ground truth data: {config.ground_truth_csv}")
    df_gt = pd.read_csv(config.ground_truth_csv)

    # Convert coordinates from meters to kilometers
    coordinate_columns = [
        'event_x_transformed', 'event_y_transformed', 'event_z_transformed',
        'station_x_transformed', 'station_y_transformed', 'station_z_transformed'
    ]
    for col in coordinate_columns:
        if col in df_gt.columns:
            df_gt[col] = df_gt[col] / 1000.0
    print("  Converted coordinates from meters to kilometers")

    # Check if weight column exists
    weight_column = config.get('weight_column', 'weight')
    if weight_column not in df_gt.columns:
        print(f"  WARNING: Weight column '{weight_column}' not found, using uniform weights (1.0)")
        df_gt[weight_column] = 1.0

    print(f"Found {len(df_gt)} source-receiver pairs")

    # Initialize DWS grid
    nx, ny, nz = config.grid.nx, config.grid.ny, config.grid.nz
    dws_grid = np.zeros((nz, ny, nx), dtype=np.float64)

    # Process by phase type
    phase_types = []
    if config.phase_type in ['both', 'p']:
        phase_types.append('p')
    if config.phase_type in ['both', 's']:
        phase_types.append('s')

    for phase in phase_types:
        print(f"\n{'='*80}")
        print(f"Processing {phase.upper()}-wave DWS")
        print(f"{'='*80}")

        # Filter by phase if column exists
        if 'phase' in df_gt.columns:
            df_phase = df_gt[df_gt['phase'].str.upper() == phase.upper()].copy()
        else:
            df_phase = df_gt.copy()

        # Prepare task list: one task per shot-receiver pair
        tasks = []

        # Get unique shots
        unique_shots = df_phase['shot_id'].unique()

        # Limit number of shots if specified
        if config.max_shots > 0:
            unique_shots = unique_shots[:config.max_shots]

        for shot_id in unique_shots:
            shot_data = df_phase[df_phase['shot_id'] == shot_id]
            if len(shot_data) == 0:
                continue

            # Get source position (same for all receivers in this shot)
            source_row = shot_data.iloc[0]
            source_pos = (
                source_row['event_x_transformed'],
                source_row['event_y_transformed'],
                source_row['event_z_transformed']
            )

            # Create task for each receiver
            for _, row in shot_data.iterrows():
                receiver_pos = (
                    row['station_x_transformed'],
                    row['station_y_transformed'],
                    row['station_z_transformed']
                )
                weight = row[weight_column]
                tasks.append((shot_id, source_pos, receiver_pos, weight))

        print(f"Created {len(tasks)} ray tracing tasks ({len(unique_shots)} shots)")

        # Get parallelization parameters
        use_parallel = hasattr(config, 'parallel') and config.parallel.enabled
        n_jobs = config.parallel.n_jobs if use_parallel else 1
        backend = getattr(config.parallel, 'backend', 'loky') if use_parallel else 'loky'
        batch_size = getattr(config.parallel, 'batch_size', 'auto') if use_parallel else 'auto'
        pre_dispatch = getattr(config.parallel, 'pre_dispatch', '2*n_jobs') if use_parallel else '2*n_jobs'

        if use_parallel and n_jobs != 1:
            print(f"Processing tasks using {n_jobs if n_jobs > 0 else 'all'} CPU cores...")
            print(f"  Backend: {backend}, Batch size: {batch_size}, Pre-dispatch: {pre_dispatch}")
        else:
            print(f"Processing tasks sequentially...")

        # Prepare tracing parameters
        tracing_params = {
            'max_steps': config.tracing.max_steps,
            'step_size': config.tracing.step_size,
            'tolerance': config.tracing.tolerance
        }

        # Get interpolation method
        interpolation_method = getattr(config.tracing, 'interpolation_method', 'nearest')
        print(f"Using '{interpolation_method}' interpolation for ray tracing")

        # Grid dimensions
        grid_dims = dict(config.grid)

        # Process all shot-receiver pairs in parallel
        contributions_all = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            batch_size=batch_size,
            pre_dispatch=pre_dispatch,
            verbose=0
        )(
            delayed(process_single_ray_dws)(
                shot_id,
                source_pos,
                receiver_pos,
                weight,
                config.gradient_dir,
                phase,
                config.precision,
                tracing_params,
                interpolation_method,
                grid_dims
            )
            for shot_id, source_pos, receiver_pos, weight in tqdm(tasks, desc=f"Computing {phase}-wave DWS")
        )

        # Accumulate contributions into DWS grid
        n_valid = 0
        print(f"\nAccumulating DWS contributions into grid...")
        for contributions in tqdm(contributions_all, desc="Accumulating"):
            if contributions is not None:
                n_valid += 1
                for (i, j, k), dws_value in contributions.items():
                    if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                        dws_grid[k, j, i] += dws_value

        print(f"Successfully computed DWS from {n_valid}/{len(tasks)} ray paths")

    print("\n" + "="*80)
    print("DWS COMPUTATION COMPLETE")
    print("="*80)

    return dws_grid


def save_dws_to_hdf5(
    dws_grid: np.ndarray,
    config: DictConfig,
    output_file: str
) -> None:
    """
    Save DWS grid to HDF5 file in the same format as velocity model.

    Args:
        dws_grid: 3D DWS array (nz, ny, nx)
        config: Configuration object
        output_file: Output HDF5 file path
    """
    print(f"\nSaving DWS to HDF5: {output_file}")

    nx, ny, nz = config.grid.nx, config.grid.ny, config.grid.nz
    dx, dy, dz = config.grid.dx, config.grid.dy, config.grid.dz
    ox, oy, oz = config.grid.origin_x, config.grid.origin_y, config.grid.origin_z

    # Create coordinate grids
    x = np.arange(nx) * dx + ox
    y = np.arange(ny) * dy + oy
    z = np.arange(nz) * dz + oz

    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Flatten grids
    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_flat = Z.ravel()
    dws_flat = dws_grid.transpose(2, 1, 0).ravel()  # Transpose to (nx, ny, nz) then flatten

    # Create indices
    i_vals = np.arange(nx).reshape(-1, 1, 1) * np.ones((nx, ny, nz), dtype=int)
    j_vals = np.arange(ny).reshape(1, -1, 1) * np.ones((nx, ny, nz), dtype=int)
    k_vals = np.arange(nz).reshape(1, 1, -1) * np.ones((nx, ny, nz), dtype=int)

    i_flat = i_vals.ravel()
    j_flat = j_vals.ravel()
    k_flat = k_vals.ravel()

    # Create DataFrame in velocity format
    # We'll set vp and vs to NaN since this is a DWS file, not velocity
    # Store DWS value in a custom column
    data = {
        'X_grid_km': x_flat,
        'Y_grid_km': y_flat,
        'Z_grid_km': z_flat,
        'x': x_flat,
        'y': y_flat,
        'z': z_flat,
        'i_raw': i_flat.astype(float),
        'j_raw': j_flat.astype(float),
        'k_raw': k_flat.astype(float),
        'i': i_flat,
        'j': j_flat,
        'k': k_flat,
        'vp': dws_flat,  # Store DWS in vp column
        'vs': np.zeros_like(dws_flat),  # Placeholder
        'lon': np.zeros_like(dws_flat),  # Placeholder
        'lat': np.zeros_like(dws_flat),  # Placeholder
        'depth': z_flat
    }

    df = pd.DataFrame(data)

    # Metadata
    metadata = {
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'grid_spacing': {'dx': dx, 'dy': dy, 'dz': dz},
        'grid_origin': {'x': ox, 'y': oy, 'z': oz},
        'data_type': 'dws',
        'description': 'Derivative Weight Sum computed from ray paths',
        'units': 'km^2 (weighted squared path length)'
    }

    # Write using velocity_io utility
    write_velocity_file(
        df,
        output_file,
        format='hdf5',
        metadata=metadata,
        compression='gzip',
        compression_opts=4,
        show_progress=True
    )

    # Print statistics
    print(f"\nDWS Statistics:")
    print(f"  Min: {dws_grid.min():.6f}")
    print(f"  Max: {dws_grid.max():.6f}")
    print(f"  Mean: {dws_grid.mean():.6f}")
    print(f"  Median: {np.median(dws_grid):.6f}")
    print(f"  Non-zero cells: {np.count_nonzero(dws_grid)} / {dws_grid.size} ({100*np.count_nonzero(dws_grid)/dws_grid.size:.2f}%)")


def create_dws_from_ray_paths(config: DictConfig) -> None:
    """
    Main function to compute and save DWS from ray paths.

    Args:
        config: Configuration object containing all parameters
    """
    print("\n" + "="*80)
    print("DWS (DERIVATIVE WEIGHT SUM) COMPUTATION FROM RAY PATHS")
    print("="*80)

    # Create output directory
    ensure_directory_exists(os.path.dirname(config.output_file))

    # Compute DWS
    dws_grid = compute_dws_from_ray_paths(config)

    # Save to HDF5
    save_dws_to_hdf5(dws_grid, config, config.output_file)

    print("\n" + "="*80)
    print("DWS COMPUTATION AND SAVING COMPLETE")
    print(f"Output file: {config.output_file}")
    print("="*80)
