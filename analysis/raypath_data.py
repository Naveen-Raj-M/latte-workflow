"""
Ray path data loading utilities for LATTE analysis.

This module provides functions for loading ray path data from HDF5 cache files
created by the postprocess module.
"""

import os
import gc
import logging
from typing import Dict, Any, List, Optional, Set, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _resample_ray_path(ray_path: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Resample a ray path to have a fixed number of points using linear interpolation.

    Args:
        ray_path: Array of (x, y, z) positions along the ray (shape: N x 3)
        num_samples: Target number of samples

    Returns:
        Resampled ray path with shape (num_samples, 3)
    """
    if len(ray_path) < 2:
        # For degenerate rays, just repeat the point
        return np.tile(ray_path[0], (num_samples, 1)) if len(ray_path) > 0 else np.zeros((num_samples, 3))

    if len(ray_path) == num_samples:
        return ray_path.copy()

    # Compute cumulative distance along the ray
    distances = np.cumsum(np.sqrt(np.sum(np.diff(ray_path, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)  # Add starting point at distance 0

    # Create uniform sampling points
    total_distance = distances[-1]
    if total_distance == 0:
        # All points are the same, just repeat
        return np.tile(ray_path[0], (num_samples, 1))

    t_uniform = np.linspace(0, total_distance, num_samples)

    # Interpolate x, y, z coordinates
    x_interp = np.interp(t_uniform, distances, ray_path[:, 0])
    y_interp = np.interp(t_uniform, distances, ray_path[:, 1])
    z_interp = np.interp(t_uniform, distances, ray_path[:, 2])

    return np.column_stack([x_interp, y_interp, z_interp])


def load_ray_paths_from_cache_vectorized(
    cache_path: str,
    phase: str = None,
    max_rays: Optional[int] = None,
    shot_ids: Optional[List[int]] = None,
    num_samples_per_ray: int = 500,
    receiver_filter_by_shot: Optional[Dict[str, Any]] = None,
    receiver_filter_tolerance: float = 0.5,
) -> Dict[str, Any]:
    """
    Load ray paths from HDF5 cache file with vectorized resampling for optimized processing.

    All rays are resampled to have the same number of points, enabling vectorized operations
    on the resulting numpy arrays.

    Args:
        cache_path: Path to the HDF5 cache file
        phase: Phase type to load ('p', 's', or None for all)
        max_rays: Optional maximum number of rays to load per phase
        shot_ids: Optional list of shot IDs to filter
        num_samples_per_ray: Number of points to resample each ray to (default: 500)

    Returns:
        Dictionary with phase keys ('p', 's') containing:
            - 'ray_paths': numpy array of shape (n_rays, num_samples_per_ray, 3)
            - 'sources': numpy array of shape (n_rays, 3)
            - 'receivers': numpy array of shape (n_rays, 3)
            - 'shot_ids': numpy array of shape (n_rays,)
            - 'weights': numpy array of shape (n_rays,)
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    # Determine phases to load
    if phase is not None:
        phases = [phase.lower()]
    else:
        phases = ['p', 's']

    result = {}

    with h5py.File(cache_path, 'r', locking=False) as f:
        for ph in phases:
            phase_group = f'{ph}_wave'

            if phase_group not in f or 'shots' not in f[phase_group]:
                logger.debug(f"No {ph.upper()}-wave data in cache")
                continue

            # Lists to collect data before converting to arrays
            ray_paths_list = []
            sources_list = []
            receivers_list = []
            shot_ids_list = []
            weights_list = []

            shots_group = f[phase_group]['shots']

            # Get shot IDs to load
            if shot_ids is not None:
                load_shot_ids = [str(sid) for sid in shot_ids if str(sid) in shots_group]
            else:
                load_shot_ids = list(shots_group.keys())

            total_loaded = 0
            corrupted_shots = []
            memory_error_shots = []

            logger.info(f"Loading {ph.upper()}-wave rays with vectorized resampling to {num_samples_per_ray} points...")

            for shot_key in tqdm(load_shot_ids):
                if shot_key not in shots_group:
                    continue

                # Resolve the per-shot receiver KDTree (if filter is active).
                # Skip this shot entirely if it has no allowed receivers.
                _shot_recv_tree = None
                if receiver_filter_by_shot is not None:
                    _shot_recv_tree = receiver_filter_by_shot.get(shot_key)
                    if _shot_recv_tree is None:
                        continue  # shot not in filter — skip without reading HDF5 data

                try:
                    shot_grp = shots_group[shot_key]

                    # Load data (may fail if compressed data is corrupted or memory is insufficient)
                    source = shot_grp['source'][:]
                    recv_array = shot_grp['receivers'][:]
                    wts_array = shot_grp['weights'][:]
                    lengths = shot_grp['ray_lengths'][:]
                    data = shot_grp['ray_data'][:]
                except MemoryError as e:
                    # Handle memory allocation failures gracefully
                    memory_error_shots.append(int(shot_key))
                    logger.warning(f"Skipping shot {shot_key} due to memory error: {e}")
                    gc.collect()  # Force garbage collection to free memory
                    continue
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

                # Reconstruct and resample individual ray paths
                offset = 0
                for i, length in enumerate(lengths):
                    if max_rays is not None and total_loaded >= max_rays:
                        break

                    if length > 0:
                        # Per-shot receiver-coordinate filter: only include rays
                        # whose receiver matches an allowed position for this shot.
                        _include = True
                        if _shot_recv_tree is not None:
                            _dist, _ = _shot_recv_tree.query(recv_array[i])
                            if _dist > receiver_filter_tolerance:
                                _include = False

                        if _include:
                            # Extract ray path
                            ray_path = data[offset:offset + length]

                            # Resample to fixed size
                            resampled_ray = _resample_ray_path(ray_path, num_samples_per_ray)

                            ray_paths_list.append(resampled_ray)
                            sources_list.append(source)
                            receivers_list.append(recv_array[i])
                            shot_ids_list.append(int(shot_key))
                            weights_list.append(wts_array[i])
                            total_loaded += 1

                    offset += length

                # Clear temporary arrays to free memory
                del data, recv_array, wts_array, lengths

                if max_rays is not None and total_loaded >= max_rays:
                    break

                # Periodic garbage collection to manage memory
                if total_loaded > 0 and total_loaded % 100 == 0:
                    gc.collect()

            # Warn if corrupted shots were found
            if corrupted_shots:
                logger.warning(
                    f"Skipped {len(corrupted_shots)} corrupted shots in {ph.upper()}-wave cache: {corrupted_shots}. "
                    f"Run postprocess with fix_corrupted_shots=true to repair cache."
                )

            # Warn if memory error shots were found
            if memory_error_shots:
                logger.warning(
                    f"Skipped {len(memory_error_shots)} shots in {ph.upper()}-wave cache due to memory errors: {memory_error_shots}. "
                    f"Consider increasing available memory or reducing max_rays parameter."
                )

            # Convert lists to numpy arrays for vectorized operations
            if len(ray_paths_list) > 0:
                result[ph] = {
                    'ray_paths': np.array(ray_paths_list),  # Shape: (n_rays, num_samples_per_ray, 3)
                    'sources': np.array(sources_list),      # Shape: (n_rays, 3)
                    'receivers': np.array(receivers_list),  # Shape: (n_rays, 3)
                    'shot_ids': np.array(shot_ids_list),    # Shape: (n_rays,)
                    'weights': np.array(weights_list)       # Shape: (n_rays,)
                }
                logger.info(f"Loaded {len(ray_paths_list)} {ph.upper()}-wave ray paths (vectorized: {result[ph]['ray_paths'].shape})")
            else:
                result[ph] = {
                    'ray_paths': np.empty((0, num_samples_per_ray, 3)),
                    'sources': np.empty((0, 3)),
                    'receivers': np.empty((0, 3)),
                    'shot_ids': np.empty((0,), dtype=int),
                    'weights': np.empty((0,))
                }
                logger.warning(f"No {ph.upper()}-wave ray paths loaded")

    # Final garbage collection to clean up
    gc.collect()

    return result


def load_event_ids_from_file(event_id_file: str) -> List[str]:
    """
    Load event IDs from a text file.

    Args:
        event_id_file: Path to file with event IDs (one per line)

    Returns:
        List of event IDs as strings
    """
    if not os.path.exists(event_id_file):
        raise FileNotFoundError(f"Event ID file not found: {event_id_file}")

    event_ids = []
    with open(event_id_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                event_ids.append(line)

    logger.info(f"Loaded {len(event_ids)} event IDs from {event_id_file}")
    return event_ids


def convert_event_ids_to_shot_ids(event_ids: List[str], shot_map_file: str) -> List[int]:
    """
    Convert event IDs to shot IDs using shot map file.

    Args:
        event_ids: List of event IDs
        shot_map_file: Path to shot map TSV file

    Returns:
        List of shot IDs as integers
    """
    if not os.path.exists(shot_map_file):
        raise FileNotFoundError(f"Shot map file not found: {shot_map_file}")

    # Load shot map
    df = pd.read_csv(shot_map_file, sep='\t')
    event_to_shot = {}
    for _, row in df.iterrows():
        event_to_shot[str(row['Original_Event_ID'])] = int(row['Shot_ID'])

    # Convert event IDs to shot IDs
    shot_ids = []
    for event_id in event_ids:
        if event_id in event_to_shot:
            shot_ids.append(event_to_shot[event_id])
        else:
            logger.warning(f"Event ID {event_id} not found in shot map")

    logger.info(f"Converted {len(shot_ids)} event IDs to shot IDs")
    return shot_ids


def load_ray_paths_from_cache(
    cache_path: str,
    phase: str = None,
    max_rays: Optional[int] = None,
    shot_ids: Optional[List[int]] = None,
    receiver_filter_by_shot: Optional[Dict[str, Any]] = None,
    receiver_filter_tolerance: float = 0.5,
) -> Dict[str, Any]:
    """
    Load ray paths from HDF5 cache file.

    Args:
        cache_path: Path to the HDF5 cache file
        phase: Phase type to load ('p', 's', or None for all)
        max_rays: Optional maximum number of rays to load per phase
        shot_ids: Optional list of shot IDs to filter

    Returns:
        Dictionary with phase keys ('p', 's') containing:
            - 'ray_paths': List of ray path arrays
            - 'sources': List of source positions
            - 'receivers': List of receiver positions
            - 'shot_ids': List of shot IDs
            - 'weights': List of weights
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    # Determine phases to load
    if phase is not None:
        phases = [phase.lower()]
    else:
        phases = ['p', 's']

    result = {}

    with h5py.File(cache_path, 'r', locking=False) as f:
        for ph in phases:
            phase_group = f'{ph}_wave'

            if phase_group not in f or 'shots' not in f[phase_group]:
                logger.debug(f"No {ph.upper()}-wave data in cache")
                continue

            ray_paths = []
            sources = []
            receivers = []
            shot_ids_list = []
            weights = []

            shots_group = f[phase_group]['shots']

            # Get shot IDs to load
            if shot_ids is not None:
                load_shot_ids = [str(sid) for sid in shot_ids if str(sid) in shots_group]
            else:
                load_shot_ids = list(shots_group.keys())

            total_loaded = 0
            corrupted_shots = []
            memory_error_shots = []

            for shot_key in load_shot_ids:
                if shot_key not in shots_group:
                    continue

                # Resolve the per-shot receiver KDTree (if filter is active).
                # Skip this shot entirely if it has no allowed receivers.
                _shot_recv_tree = None
                if receiver_filter_by_shot is not None:
                    _shot_recv_tree = receiver_filter_by_shot.get(shot_key)
                    if _shot_recv_tree is None:
                        continue  # shot not in filter — skip without reading HDF5 data

                try:
                    shot_grp = shots_group[shot_key]

                    # Load data (may fail if compressed data is corrupted or memory is insufficient)
                    source = tuple(shot_grp['source'][:])
                    recv_array = shot_grp['receivers'][:]
                    wts_array = shot_grp['weights'][:]
                    lengths = shot_grp['ray_lengths'][:]
                    data = shot_grp['ray_data'][:]
                except MemoryError as e:
                    # Handle memory allocation failures gracefully
                    memory_error_shots.append(int(shot_key))
                    logger.warning(f"Skipping shot {shot_key} due to memory error: {e}")
                    gc.collect()  # Force garbage collection to free memory
                    continue
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

                # Reconstruct individual ray paths
                offset = 0
                for i, length in enumerate(lengths):
                    if max_rays is not None and total_loaded >= max_rays:
                        break

                    if length > 0:
                        # Per-shot receiver-coordinate filter: only include rays
                        # whose receiver matches an allowed position for this shot.
                        _include = True
                        if _shot_recv_tree is not None:
                            _dist, _ = _shot_recv_tree.query(recv_array[i])
                            if _dist > receiver_filter_tolerance:
                                _include = False

                        if _include:
                            ray_path = data[offset:offset + length].copy()
                            ray_paths.append(ray_path)
                            sources.append(source)
                            receivers.append(tuple(recv_array[i]))
                            shot_ids_list.append(int(shot_key))
                            weights.append(wts_array[i])
                            total_loaded += 1

                    offset += length

                # Clear temporary arrays to free memory
                del data, recv_array, wts_array, lengths

                if max_rays is not None and total_loaded >= max_rays:
                    break

                # Periodic garbage collection to manage memory
                if total_loaded > 0 and total_loaded % 100 == 0:
                    gc.collect()

            # Warn if corrupted shots were found
            if corrupted_shots:
                logger.warning(
                    f"Skipped {len(corrupted_shots)} corrupted shots in {ph.upper()}-wave cache: {corrupted_shots}. "
                    f"Run postprocess with fix_corrupted_shots=true to repair cache."
                )

            # Warn if memory error shots were found
            if memory_error_shots:
                logger.warning(
                    f"Skipped {len(memory_error_shots)} shots in {ph.upper()}-wave cache due to memory errors: {memory_error_shots}. "
                    f"Consider increasing available memory or reducing max_rays parameter."
                )

            result[ph] = {
                'ray_paths': ray_paths,
                'sources': sources,
                'receivers': receivers,
                'shot_ids': shot_ids_list,
                'weights': weights
            }

            logger.info(f"Loaded {len(ray_paths)} {ph.upper()}-wave ray paths")

    # Final garbage collection to clean up
    gc.collect()

    return result


def get_ray_path_metadata(cache_path: str) -> Dict[str, Any]:
    """
    Get metadata from ray path cache file.

    Args:
        cache_path: Path to the HDF5 cache file

    Returns:
        Dictionary with metadata including:
            - 'config_hash': Configuration hash
            - 'created': Creation timestamp
            - 'last_updated': Last update timestamp
            - 'tracing': Tracing parameters
            - 'grid': Grid parameters
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    metadata = {}

    with h5py.File(cache_path, 'r', locking=False) as f:
        if 'metadata' not in f:
            return metadata

        meta = f['metadata']
        metadata['config_hash'] = meta.attrs.get('config_hash', '')
        metadata['created'] = meta.attrs.get('created', None)
        metadata['last_updated'] = meta.attrs.get('last_updated', None)
        metadata['ground_truth_csv'] = meta.attrs.get('ground_truth_csv', '')

        # Tracing parameters
        if 'tracing' in meta:
            tracing = meta['tracing']
            metadata['tracing'] = {
                'max_steps': tracing.attrs.get('max_steps', None),
                'step_size': tracing.attrs.get('step_size', None),
                'tolerance': tracing.attrs.get('tolerance', None),
                'interpolation': tracing.attrs.get('interpolation', None)
            }

        # Grid parameters
        if 'grid' in meta:
            grid = meta['grid']
            metadata['grid'] = {
                'nx': grid.attrs.get('nx', None),
                'ny': grid.attrs.get('ny', None),
                'nz': grid.attrs.get('nz', None),
                'dx': grid.attrs.get('dx', None),
                'dy': grid.attrs.get('dy', None),
                'dz': grid.attrs.get('dz', None),
                'origin_x': grid.attrs.get('origin_x', None),
                'origin_y': grid.attrs.get('origin_y', None),
                'origin_z': grid.attrs.get('origin_z', None)
            }

    return metadata


def get_ray_path_statistics(cache_path: str) -> Dict[str, Any]:
    """
    Get statistics about the ray path cache.

    Args:
        cache_path: Path to the HDF5 cache file

    Returns:
        Dictionary with statistics including:
            - 'exists': Whether file exists
            - 'file_size_mb': File size in MB
            - 'p_wave_shots': Number of P-wave shots
            - 's_wave_shots': Number of S-wave shots
            - 'p_wave_rays': Number of P-wave rays
            - 's_wave_rays': Number of S-wave rays
            - 'created': Creation timestamp
            - 'last_updated': Last update timestamp
            - 'p_wave_corrupted': Number of corrupted P-wave shots
            - 's_wave_corrupted': Number of corrupted S-wave shots
    """
    stats = {
        'exists': os.path.exists(cache_path),
        'p_wave_shots': 0,
        's_wave_shots': 0,
        'p_wave_rays': 0,
        's_wave_rays': 0,
        'p_wave_corrupted': 0,
        's_wave_corrupted': 0,
        'file_size_mb': 0,
        'created': None,
        'last_updated': None
    }

    if not stats['exists']:
        return stats

    stats['file_size_mb'] = os.path.getsize(cache_path) / (1024 * 1024)

    try:
        with h5py.File(cache_path, 'r', locking=False) as f:
            if 'metadata' in f:
                stats['created'] = f['metadata'].attrs.get('created', None)
                stats['last_updated'] = f['metadata'].attrs.get('last_updated', None)

            for phase in ['p', 's']:
                phase_group = f'{phase}_wave'
                if phase_group in f and 'shots' in f[phase_group]:
                    shots = f[phase_group]['shots']
                    stats[f'{phase}_wave_shots'] = len(shots)

                    # Count total rays and check for corruption
                    total_rays = 0
                    corrupted_count = 0
                    for shot_key in shots.keys():
                        if 'ray_lengths' in shots[shot_key]:
                            try:
                                # Try to read ray_data to check for corruption
                                _ = shots[shot_key]['ray_data'][:]
                                total_rays += len(shots[shot_key]['ray_lengths'][:])
                            except MemoryError:
                                # Skip shots that cause memory errors
                                corrupted_count += 1
                                logger.debug(f"Found shot {shot_key} with memory allocation issue in statistics")
                                gc.collect()
                            except (OSError, RuntimeError) as e:
                                error_msg = str(e).lower()
                                if "filter" in error_msg or "read" in error_msg or "decompress" in error_msg:
                                    corrupted_count += 1
                                    logger.debug(f"Found corrupted shot {shot_key} in statistics")
                                else:
                                    raise

                    stats[f'{phase}_wave_rays'] = total_rays
                    stats[f'{phase}_wave_corrupted'] = corrupted_count

                    if corrupted_count > 0:
                        logger.warning(
                            f"Found {corrupted_count} corrupted or memory-problematic shots in {phase.upper()}-wave cache. "
                            f"Run postprocess with fix_corrupted_shots=true to repair or increase available memory."
                        )
    except Exception as e:
        logger.error(f"Error reading cache stats: {e}")

    return stats


def get_unique_sources(cache_path: str, phase: str = 'p') -> List[Tuple[float, float, float]]:
    """
    Get unique source positions from cache.

    Args:
        cache_path: Path to the HDF5 cache file
        phase: Phase type ('p' or 's')

    Returns:
        List of unique source positions as (x, y, z) tuples
    """
    if not os.path.exists(cache_path):
        return []

    sources = set()

    with h5py.File(cache_path, 'r', locking=False) as f:
        phase_group = f'{phase}_wave'
        if phase_group not in f or 'shots' not in f[phase_group]:
            return []

        shots_group = f[phase_group]['shots']
        for shot_key in shots_group.keys():
            source = tuple(shots_group[shot_key]['source'][:])
            sources.add(source)

    return list(sources)


def get_unique_receivers(cache_path: str, phase: str = 'p') -> List[Tuple[float, float, float]]:
    """
    Get unique receiver positions from cache.

    Args:
        cache_path: Path to the HDF5 cache file
        phase: Phase type ('p' or 's')

    Returns:
        List of unique receiver positions as (x, y, z) tuples
    """
    if not os.path.exists(cache_path):
        return []

    receivers = set()
    corrupted_count = 0

    with h5py.File(cache_path, 'r', locking=False) as f:
        phase_group = f'{phase}_wave'
        if phase_group not in f or 'shots' not in f[phase_group]:
            return []

        shots_group = f[phase_group]['shots']
        for shot_key in shots_group.keys():
            try:
                recv_array = shots_group[shot_key]['receivers'][:]
                for recv in recv_array:
                    receivers.add(tuple(recv))
            except MemoryError:
                # Skip shots that cause memory errors
                corrupted_count += 1
                logger.debug(f"Skipping shot {shot_key} with memory allocation issue in get_unique_receivers")
                gc.collect()
                continue
            except (OSError, RuntimeError) as e:
                error_msg = str(e).lower()
                if "filter" in error_msg or "read" in error_msg or "decompress" in error_msg:
                    corrupted_count += 1
                    logger.debug(f"Skipping corrupted shot {shot_key} in get_unique_receivers")
                    continue
                else:
                    raise

    if corrupted_count > 0:
        logger.warning(
            f"Skipped {corrupted_count} corrupted or memory-problematic shots while reading receivers. "
            f"Run postprocess with fix_corrupted_shots=true to repair cache or increase available memory."
        )

    return list(receivers)
