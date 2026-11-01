"""
Ray path simplification module for LATTE post-processing.

This module provides functionality to simplify ray paths using the Ramer-Douglas-Peucker (RDP)
algorithm to reduce memory usage and improve analysis performance. The simplified ray paths
are stored in a new HDF5 file with the same format as the original.

Typical use case:
- Original: 300k rays with 20k points each = massive memory footprint
- Simplified: 300k rays with 100-500 points each = manageable memory footprint
"""

import os
import gc
import time
from datetime import datetime
from typing import Dict, Optional

import h5py
import numba as nb
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig

from utils.io_utils import ensure_directory_exists


@nb.njit
def rdp_3d(points: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Ramer-Douglas-Peucker algorithm for 3D line simplification using Numba JIT.

    This algorithm recursively finds the point with maximum distance from the line
    segment and splits if the distance exceeds epsilon.

    Args:
        points: Array of shape (N, 3) containing (x, y, z) coordinates
        epsilon: Distance threshold in km - points closer than this to the line
                 segment will be removed

    Returns:
        Boolean array of shape (N,) indicating which points to keep
    """
    n = len(points)
    if n < 2:
        return np.ones(n, dtype=np.bool_)

    keep = np.zeros(n, dtype=np.bool_)
    keep[0] = True
    keep[-1] = True

    # Use a manual stack instead of recursion for better performance
    stack = [(0, n - 1)]

    while stack:
        start, end = stack.pop()

        if end - start <= 1:
            continue

        max_dist = 0.0
        max_index = -1

        # Get start and end points
        x1, y1, z1 = points[start]
        x2, y2, z2 = points[end]

        # Vector from start to end
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1

        # Length squared of the segment
        denom = dx*dx + dy*dy + dz*dz

        # Find point with maximum perpendicular distance to line segment
        for i in range(start + 1, end):
            px, py, pz = points[i]

            if denom > 0:
                # Project point onto line segment (clamped to [0, 1])
                t = ((px - x1)*dx + (py - y1)*dy + (pz - z1)*dz) / denom
                t = min(1.0, max(0.0, t))

                # Find closest point on segment
                projx = x1 + t*dx
                projy = y1 + t*dy
                projz = z1 + t*dz

                # Distance squared from point to projection
                dist = (px - projx)**2 + (py - projy)**2 + (pz - projz)**2
            else:
                # Degenerate case: start and end are the same point
                dist = (px - x1)**2 + (py - y1)**2 + (pz - z1)**2

            if dist > max_dist:
                max_dist = dist
                max_index = i

        # If max distance exceeds threshold, keep that point and recurse
        if max_dist > epsilon*epsilon:
            keep[max_index] = True
            stack.append((start, max_index))
            stack.append((max_index, end))

    return keep


def simplify_ray_path(ray_path: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Simplify a single ray path using the RDP algorithm.

    Args:
        ray_path: Array of shape (N, 3) containing ray path coordinates in km
        epsilon: Distance threshold in km

    Returns:
        Simplified ray path array of shape (M, 3) where M <= N
    """
    if len(ray_path) < 3:
        # Too short to simplify
        return ray_path.copy()

    keep_mask = rdp_3d(ray_path, epsilon)
    return ray_path[keep_mask]


def simplify_ray_paths_in_cache(
    input_cache_path: str,
    output_cache_path: str,
    epsilon: float,
    config: DictConfig,
    phases: Optional[list] = None
) -> Dict[str, int]:
    """
    Read ray paths from an existing cache, simplify them, and write to a new cache.

    The output cache will have the identical HDF5 structure as the input cache,
    allowing it to be used as a drop-in replacement in the analysis module.

    Args:
        input_cache_path: Path to the existing ray_paths.h5 file
        output_cache_path: Path to the new simplified ray_paths.h5 file
        epsilon: RDP distance threshold in km (typical: 0.01 to 0.1)
        config: Configuration object (for metadata)
        phases: List of phases to process (default: ['p', 's'])

    Returns:
        Dictionary with statistics:
            - 'p_wave_rays': Number of P-wave rays processed
            - 's_wave_rays': Number of S-wave rays processed
            - 'p_wave_shots': Number of P-wave shots processed
            - 's_wave_shots': Number of S-wave shots processed
            - 'p_wave_points_original': Original point count for P-wave
            - 'p_wave_points_simplified': Simplified point count for P-wave
            - 's_wave_points_original': Original point count for S-wave
            - 's_wave_points_simplified': Simplified point count for S-wave
            - 'processing_time_seconds': Total processing time
    """
    if not os.path.exists(input_cache_path):
        raise FileNotFoundError(f"Input cache not found: {input_cache_path}")

    if phases is None:
        phases = ['p', 's']

    # Ensure output directory exists
    ensure_directory_exists(os.path.dirname(output_cache_path))

    stats = {
        'p_wave_rays': 0,
        's_wave_rays': 0,
        'p_wave_shots': 0,
        's_wave_shots': 0,
        'p_wave_points_original': 0,
        'p_wave_points_simplified': 0,
        's_wave_points_original': 0,
        's_wave_points_simplified': 0,
        'processing_time_seconds': 0
    }

    print("\n" + "="*80)
    print("RAY PATH SIMPLIFICATION")
    print("="*80)
    print(f"Input:   {input_cache_path}")
    print(f"Output:  {output_cache_path}")
    print(f"Epsilon: {epsilon} km (RDP distance threshold)")
    print(f"Phases:  {', '.join([p.upper() for p in phases])}")

    start_time = time.time()

    # Open input and create output
    with h5py.File(input_cache_path, 'r', locking=False) as f_in:
        with h5py.File(output_cache_path, 'w') as f_out:
            # Copy metadata group with modifications
            meta_out = f_out.create_group('metadata')

            if 'metadata' in f_in:
                meta_in = f_in['metadata']

                # Copy basic metadata
                if 'config_hash' in meta_in.attrs:
                    meta_out.attrs['config_hash'] = meta_in.attrs['config_hash']
                if 'ground_truth_csv' in meta_in.attrs:
                    meta_out.attrs['ground_truth_csv'] = meta_in.attrs['ground_truth_csv']

                # Update timestamps
                meta_out.attrs['created'] = datetime.now().isoformat()
                meta_out.attrs['source_file'] = input_cache_path
                meta_out.attrs['simplified'] = True
                meta_out.attrs['rdp_epsilon'] = epsilon

                # Copy tracing parameters
                if 'tracing' in meta_in:
                    tracing_out = meta_out.create_group('tracing')
                    tracing_in = meta_in['tracing']
                    for key in tracing_in.attrs.keys():
                        tracing_out.attrs[key] = tracing_in.attrs[key]

                # Copy grid parameters
                if 'grid' in meta_in:
                    grid_out = meta_out.create_group('grid')
                    grid_in = meta_in['grid']
                    for key in grid_in.attrs.keys():
                        grid_out.attrs[key] = grid_in.attrs[key]

            # Process each phase
            for phase in phases:
                phase_group = f'{phase}_wave'

                if phase_group not in f_in:
                    print(f"\nSkipping {phase.upper()}-wave (not in input cache)")
                    continue

                if 'shots' not in f_in[phase_group]:
                    print(f"\nSkipping {phase.upper()}-wave (no shots in cache)")
                    continue

                print(f"\n{'='*80}")
                print(f"Processing {phase.upper()}-wave rays")
                print(f"{'='*80}")

                shots_in = f_in[phase_group]['shots']
                shot_keys = list(shots_in.keys())

                print(f"  Found {len(shot_keys)} shots")

                # Create output phase group
                phase_out = f_out.create_group(phase_group)
                shots_out = phase_out.create_group('shots')

                total_rays = 0
                total_points_orig = 0
                total_points_simp = 0

                # Process each shot
                for shot_key in tqdm(shot_keys, desc=f"Simplifying {phase.upper()}-wave"):
                    shot_in = shots_in[shot_key]

                    try:
                        # Load shot data
                        source = shot_in['source'][:]
                        receivers = shot_in['receivers'][:]
                        weights = shot_in['weights'][:]
                        ray_lengths = shot_in['ray_lengths'][:]
                        ray_data = shot_in['ray_data'][:]
                    except Exception as e:
                        print(f"\n  WARNING: Failed to read shot {shot_key}: {e}")
                        continue

                    # Simplify each ray
                    simplified_rays = []
                    simplified_lengths = []

                    offset = 0
                    for length in ray_lengths:
                        if length > 0:
                            # Extract ray path
                            ray_path = ray_data[offset:offset + length]
                            total_points_orig += length

                            # Simplify
                            simplified_ray = simplify_ray_path(ray_path, epsilon)
                            simplified_rays.append(simplified_ray)
                            simplified_lengths.append(len(simplified_ray))
                            total_points_simp += len(simplified_ray)
                            total_rays += 1
                        else:
                            # Empty ray (shouldn't happen, but handle it)
                            simplified_lengths.append(0)

                        offset += length

                    # Create output shot group
                    shot_out = shots_out.create_group(shot_key)

                    # Store metadata (unchanged)
                    shot_out.create_dataset('source', data=source)
                    shot_out.create_dataset('receivers', data=receivers,
                                          compression='gzip', compression_opts=4)
                    shot_out.create_dataset('weights', data=weights)

                    # Store simplified ray paths
                    if simplified_rays:
                        simplified_lengths_arr = np.array(simplified_lengths, dtype=np.int32)
                        simplified_data = np.vstack(simplified_rays).astype(np.float32)

                        shot_out.create_dataset('ray_lengths', data=simplified_lengths_arr)
                        shot_out.create_dataset('ray_data', data=simplified_data,
                                              compression='gzip', compression_opts=4)
                    else:
                        # Empty shot
                        shot_out.create_dataset('ray_lengths',
                                              data=np.array([], dtype=np.int32))
                        shot_out.create_dataset('ray_data',
                                              data=np.array([]).reshape(0, 3).astype(np.float32))

                    # Periodic garbage collection
                    if int(shot_key) % 100 == 0:
                        gc.collect()

                # Update statistics
                stats[f'{phase}_wave_rays'] = total_rays
                stats[f'{phase}_wave_shots'] = len(shot_keys)
                stats[f'{phase}_wave_points_original'] = total_points_orig
                stats[f'{phase}_wave_points_simplified'] = total_points_simp

                # Calculate compression ratio
                if total_points_orig > 0:
                    compression_ratio = total_points_simp / total_points_orig
                    reduction_pct = (1 - compression_ratio) * 100

                    print(f"\n  {phase.upper()}-wave Summary:")
                    print(f"    Rays processed:     {total_rays:,}")
                    print(f"    Points (original):  {total_points_orig:,}")
                    print(f"    Points (simplified):{total_points_simp:,}")
                    print(f"    Compression ratio:  {compression_ratio:.3f}")
                    print(f"    Point reduction:    {reduction_pct:.1f}%")

    elapsed_time = time.time() - start_time
    stats['processing_time_seconds'] = elapsed_time

    # Get file sizes
    input_size_mb = os.path.getsize(input_cache_path) / (1024 * 1024)
    output_size_mb = os.path.getsize(output_cache_path) / (1024 * 1024)

    print(f"\n{'='*80}")
    print("SIMPLIFICATION COMPLETE")
    print(f"{'='*80}")
    print(f"Processing time: {elapsed_time:.1f} seconds")
    print(f"Input file size:  {input_size_mb:.1f} MB")
    print(f"Output file size: {output_size_mb:.1f} MB")
    print(f"File size ratio:  {output_size_mb/input_size_mb:.3f}")

    # Calculate overall statistics
    total_rays = stats['p_wave_rays'] + stats['s_wave_rays']
    total_orig = stats['p_wave_points_original'] + stats['s_wave_points_original']
    total_simp = stats['p_wave_points_simplified'] + stats['s_wave_points_simplified']

    if total_orig > 0:
        overall_compression = total_simp / total_orig
        overall_reduction = (1 - overall_compression) * 100
        print(f"\nOverall compression: {overall_compression:.3f} ({overall_reduction:.1f}% reduction)")
        print(f"Average points per ray: {total_orig/total_rays:.0f} -> {total_simp/total_rays:.0f}")

    return stats


def simplify_ray_paths_main(config: DictConfig) -> None:
    """
    Main entry point for ray path simplification.

    Args:
        config: ray_path_processing configuration containing the ``simplification`` block
    """
    if config.get('simplification', None) is None:
        print("No ray_path_processing.simplification configuration found")
        return

    simp_config = config.simplification

    if not getattr(simp_config, 'enabled', False):
        print("Ray path simplification is disabled")
        return

    input_path = simp_config.input_cache_path
    output_path = simp_config.output_cache_path
    epsilon = simp_config.epsilon

    # Determine phases to process
    phase_type = getattr(simp_config, 'phase_type', 'both')
    phases = []
    if phase_type in ['both', 'p']:
        phases.append('p')
    if phase_type in ['both', 's']:
        phases.append('s')

    # Run simplification
    stats = simplify_ray_paths_in_cache(
        input_cache_path=input_path,
        output_cache_path=output_path,
        epsilon=epsilon,
        config=config,
        phases=phases
    )

    print("\nSimplified ray paths are ready for use in analysis!")
    print(f"To use them, update your analysis config to point to: {output_path}")
