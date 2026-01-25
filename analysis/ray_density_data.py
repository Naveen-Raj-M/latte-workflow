"""
Ray density data loading and processing module.

This module loads ray paths from HDF5 cache and computes 3D ray density grids
using actual traced ray paths (not straight lines).
"""

import os
import logging
import numpy as np
import h5py
from typing import Dict, Tuple, Optional
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


def compute_ray_density_from_cache(
    cache_path: str,
    grid_dims: Dict,
    phase: str = 'p',
    max_rays: Optional[int] = None,
    shot_ids: Optional[list] = None,
    num_samples_per_ray: int = 1000,
    show_progress: bool = True
) -> np.ndarray:
    """
    Compute 3D ray density grid from ray path cache.

    Uses actual traced ray paths from the cache (not straight lines).

    Args:
        cache_path: Path to ray path HDF5 cache
        grid_dims: Grid dimension dictionary with nx, ny, nz, dx, dy, dz, x_min, y_min, z_min
        phase: Phase type ('p' or 's')
        max_rays: Maximum number of rays to load (None for all)
        shot_ids: Optional list of shot IDs to filter (None for all)
        num_samples_per_ray: Number of sample points per ray for grid intersection
        show_progress: Whether to show progress bar

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

        # Count total rays for progress tracking (skip corrupted shots)
        total_rays = 0
        for sid in shot_keys:
            if 'ray_lengths' in shots_group[sid]:
                try:
                    total_rays += len(shots_group[sid]['ray_lengths'][:])
                except (OSError, RuntimeError) as e:
                    # Skip corrupted shots during counting
                    error_msg = str(e).lower()
                    if "filter" not in error_msg and "read" not in error_msg and "decompress" not in error_msg:
                        raise
                    # else: silently skip corrupted shot

        logger.info(f"  Total rays in cache: {total_rays:,}")

        if max_rays is not None and max_rays < total_rays:
            logger.info(f"  Limiting to {max_rays:,} rays")
            total_rays = max_rays

        # Process rays from cache
        n_valid_rays = 0
        loaded_rays = 0
        corrupted_shots = []

        iterator = tqdm(shot_keys, desc=f"Computing {phase}-wave density") if show_progress else shot_keys

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

            offset = 0
            for length in lengths:
                if max_rays is not None and loaded_rays >= max_rays:
                    break

                if length > 0:
                    # Extract this ray path
                    ray_path = data[offset:offset + length]

                    # Find grid cells this ray passes through
                    grid_cells = find_ray_grid_intersections(
                        ray_path, grid_dims, num_samples_per_ray
                    )

                    # Increment density for each cell
                    for i, j, k in grid_cells:
                        density[i, j, k] += 1

                    n_valid_rays += 1

                offset += length
                loaded_rays += 1

            if max_rays is not None and loaded_rays >= max_rays:
                break

        # Warn if corrupted shots were found
        if corrupted_shots:
            logger.warning(
                f"Skipped {len(corrupted_shots)} corrupted shots in {phase.upper()}-wave cache: {corrupted_shots}. "
                f"Run postprocess with fix_corrupted_shots=true to repair cache."
            )

    logger.info(f"  Processed {n_valid_rays:,} valid rays")
    logger.info(f"  Max ray density: {density.max()} rays per cell")

    nonzero_cells = density > 0
    if nonzero_cells.any():
        logger.info(f"  Mean ray density: {density[nonzero_cells].mean():.2f} rays per cell (non-zero cells)")
        logger.info(f"  Coverage: {nonzero_cells.sum():,} / {density.size:,} cells ({100*nonzero_cells.sum()/density.size:.1f}%)")
    else:
        logger.warning("  No grid cells have ray coverage!")

    return density
