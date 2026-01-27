"""
Ray density data loading and processing module.

This module loads ray paths from HDF5 cache and computes 3D ray density grids
using actual traced ray paths (not straight lines).
"""

import os
import logging
import numpy as np
import h5py
from joblib import Parallel, delayed
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm

logger = logging.getLogger(__name__)


def find_ray_grid_intersections(
    ray_path: np.ndarray,
    grid_dims: Dict,
    num_samples: Optional[int] = None
) -> np.ndarray:
    """
    Find which grid cells a ray path passes through.

    Args:
        ray_path: Array of (x, y, z) positions along the ray in km
        grid_dims: Grid dimension dictionary
        num_samples: If provided, resample ray to this many points for more uniform sampling

    Returns:
        Array of unique grid indices (i, j, k) that the ray passes through
    """
    if len(ray_path) < 2:
        return np.array([], dtype=int).reshape(0, 3)

    # Optionally resample ray for more uniform grid cell coverage
    if num_samples is not None and len(ray_path) > 2:
        # Create parametric representation of ray path
        distances = np.cumsum(np.sqrt(np.sum(np.diff(ray_path, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0)

        # Interpolate to get uniform sampling
        t_uniform = np.linspace(0, distances[-1], num_samples)
        ray_x = np.interp(t_uniform, distances, ray_path[:, 0])
        ray_y = np.interp(t_uniform, distances, ray_path[:, 1])
        ray_z = np.interp(t_uniform, distances, ray_path[:, 2])

        sample_points = np.column_stack([ray_x, ray_y, ray_z])
    else:
        sample_points = ray_path

    # Convert to grid indices (0-based)
    i = np.floor((sample_points[:, 0] - grid_dims['x_min']) / grid_dims['dx']).astype(int)
    j = np.floor((sample_points[:, 1] - grid_dims['y_min']) / grid_dims['dy']).astype(int)
    k = np.floor((sample_points[:, 2] - grid_dims['z_min']) / grid_dims['dz']).astype(int)

    # Filter out points outside the grid
    valid = (
        (i >= 0) & (i < grid_dims['nx']) &
        (j >= 0) & (j < grid_dims['ny']) &
        (k >= 0) & (k < grid_dims['nz'])
    )

    # Get unique grid cells
    grid_cells = np.column_stack([i[valid], j[valid], k[valid]])
    unique_cells = np.unique(grid_cells, axis=0)

    return unique_cells


def _process_ray_batch_traced(
    ray_batch: List[np.ndarray],
    grid_dims: Dict,
    num_samples: int
) -> np.ndarray:
    """
    Process a batch of traced ray paths and return partial density grid.

    This function is designed to be called in parallel for efficient processing.

    Args:
        ray_batch: List of ray path arrays (each shape N x 3)
        grid_dims: Grid dimension dictionary
        num_samples: Number of samples along each ray path

    Returns:
        3D array of ray counts (partial density) with shape (nx, ny, nz)
    """
    # Initialize partial density grid
    partial_density = np.zeros((grid_dims['nx'], grid_dims['ny'], grid_dims['nz']), dtype=np.int32)

    for ray_path in ray_batch:
        if len(ray_path) < 2:
            continue

        # Sample points along the ray path (vectorized)
        n_points = len(ray_path)
        if n_points <= num_samples:
            sampled_points = ray_path
        else:
            sample_indices = np.linspace(0, n_points - 1, num_samples, dtype=int)
            sampled_points = ray_path[sample_indices]

        # Convert to grid indices (vectorized)
        i = np.floor((sampled_points[:, 0] - grid_dims['x_min']) / grid_dims['dx']).astype(np.int32)
        j = np.floor((sampled_points[:, 1] - grid_dims['y_min']) / grid_dims['dy']).astype(np.int32)
        k = np.floor((sampled_points[:, 2] - grid_dims['z_min']) / grid_dims['dz']).astype(np.int32)

        # Filter valid points (vectorized)
        valid = (
            (i >= 0) & (i < grid_dims['nx']) &
            (j >= 0) & (j < grid_dims['ny']) &
            (k >= 0) & (k < grid_dims['nz'])
        )

        if np.any(valid):
            # Get unique cells for this ray
            grid_cells = np.column_stack([i[valid], j[valid], k[valid]])
            unique_cells = np.unique(grid_cells, axis=0)

            # Increment density using np.add.at for efficiency
            np.add.at(partial_density, (unique_cells[:, 0], unique_cells[:, 1], unique_cells[:, 2]), 1)

    return partial_density


def compute_ray_density_from_cache(
    cache_path: str,
    grid_dims: Dict,
    phase: str = 'p',
    max_rays: Optional[int] = None,
    shot_ids: Optional[list] = None,
    num_samples_per_ray: int = 1000,
    show_progress: bool = True,
    n_jobs: int = 1,
    batch_size: int = 1000
) -> np.ndarray:
    """
    Compute 3D ray density grid from ray path cache using parallel processing.

    Uses actual traced ray paths from the cache (not straight lines).

    Args:
        cache_path: Path to ray path HDF5 cache
        grid_dims: Grid dimension dictionary with nx, ny, nz, dx, dy, dz, x_min, y_min, z_min
        phase: Phase type ('p' or 's')
        max_rays: Maximum number of rays to load (None for all)
        shot_ids: Optional list of shot IDs to filter (None for all)
        num_samples_per_ray: Number of sample points per ray for grid intersection
        show_progress: Whether to show progress bar
        n_jobs: Number of parallel jobs (-1 = all cores, 1 = sequential)
        batch_size: Number of rays per batch for parallel processing

    Returns:
        3D array of ray counts with shape (nx, ny, nz)
    """
    logger.info(f"Computing ray density from cache: {cache_path}")
    logger.info(f"  Phase: {phase.upper()}-wave")
    logger.info(f"  Grid: {grid_dims['nx']} x {grid_dims['ny']} x {grid_dims['nz']}")

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Ray path cache not found: {cache_path}")

    # Initialize density grid
    density = np.zeros((grid_dims['nx'], grid_dims['ny'], grid_dims['nz']), dtype=np.int32)

    phase_key = f'{phase}_wave'

    # Step 1: Load all ray paths into memory
    logger.info(f"  Loading ray paths from HDF5 cache...")
    ray_paths = []
    corrupted_shots = []

    with h5py.File(cache_path, 'r') as f:
        if phase_key not in f or 'shots' not in f[phase_key]:
            logger.warning(f"No {phase.upper()}-wave data in cache")
            return density

        shots_group = f[phase_key]['shots']

        # Get shot IDs to process
        if shot_ids is not None:
            # Filter to only requested shot IDs
            available_shots = set(shots_group.keys())
            shot_keys = [str(sid) for sid in shot_ids if str(sid) in available_shots]
        else:
            shot_keys = list(shots_group.keys())

        # Load ray paths
        iterator = tqdm(shot_keys, desc=f"Loading {phase}-wave rays") if show_progress else shot_keys

        for shot_key in iterator:
            try:
                shot_grp = shots_group[shot_key]

                # Load data (may fail if compressed data is corrupted)
                lengths = shot_grp['ray_lengths'][:]
                data = shot_grp['ray_data'][:]
            except (OSError, RuntimeError) as e:
                # Skip corrupted shots (gzip decompression failure or other HDF5 errors)
                error_msg = str(e).lower()
                if "filter" in error_msg or "read" in error_msg or "decompress" in error_msg:
                    corrupted_shots.append(int(shot_key))
                    logger.debug(f"Skipping corrupted shot {shot_key}: {e}")
                    continue
                else:
                    logger.error(f"Unexpected error reading shot {shot_key}: {e}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected exception for shot {shot_key}: {type(e).__name__}: {e}")
                raise

            # Extract individual ray paths
            offset = 0
            for length in lengths:
                if max_rays is not None and len(ray_paths) >= max_rays:
                    break

                if length > 0:
                    ray_path = data[offset:offset + length]
                    ray_paths.append(ray_path)

                offset += length

            if max_rays is not None and len(ray_paths) >= max_rays:
                break

        # Warn if corrupted shots were found
        if corrupted_shots:
            logger.warning(
                f"Skipped {len(corrupted_shots)} corrupted shots in {phase.upper()}-wave cache: {corrupted_shots}. "
                f"Run postprocess with fix_corrupted_shots=true to repair cache."
            )

    n_rays = len(ray_paths)
    logger.info(f"  Loaded {n_rays:,} ray paths")

    if n_rays == 0:
        logger.warning("  No ray paths loaded!")
        return density

    # Step 2: Split ray paths into batches
    n_batches = (n_rays + batch_size - 1) // batch_size
    batches = [ray_paths[i * batch_size:min((i + 1) * batch_size, n_rays)] for i in range(n_batches)]

    if show_progress:
        logger.info(f"  Processing {n_rays:,} rays in {n_batches} batches (batch_size={batch_size}, n_jobs={n_jobs})")

    # Step 3: Process batches in parallel
    if n_jobs == 1:
        # Sequential processing with progress bar
        partial_densities = []
        iterator = tqdm(batches, desc=f"Computing {phase}-wave ray density") if show_progress else batches
        for batch in iterator:
            partial_densities.append(_process_ray_batch_traced(batch, grid_dims, num_samples_per_ray))
    else:
        # Parallel processing with joblib
        partial_densities = Parallel(n_jobs=n_jobs, verbose=10 if show_progress else 0)(
            delayed(_process_ray_batch_traced)(batch, grid_dims, num_samples_per_ray)
            for batch in batches
        )

    # Step 4: Merge partial densities
    density = np.zeros((grid_dims['nx'], grid_dims['ny'], grid_dims['nz']), dtype=np.int32)
    for partial in partial_densities:
        density += partial

    logger.info(f"  Processed {n_rays:,} valid rays")
    logger.info(f"  Max ray density: {density.max()} rays per cell")

    nonzero_cells = density > 0
    if nonzero_cells.any():
        logger.info(f"  Mean ray density: {density[nonzero_cells].mean():.2f} rays per cell (non-zero cells)")
        logger.info(f"  Coverage: {nonzero_cells.sum():,} / {density.size:,} cells ({100*nonzero_cells.sum()/density.size:.1f}%)")
    else:
        logger.warning("  No grid cells have ray coverage!")

    return density
