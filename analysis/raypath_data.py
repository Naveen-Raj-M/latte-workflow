"""
Ray path data loading utilities for LATTE analysis.

This module provides functions for loading ray path data from HDF5 cache files
created by the postprocess module.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Set, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def load_ray_paths_from_cache(
    cache_path: str,
    phase: str = None,
    max_rays: Optional[int] = None,
    shot_ids: Optional[List[int]] = None
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

    with h5py.File(cache_path, 'r') as f:
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

            for shot_key in load_shot_ids:
                if shot_key not in shots_group:
                    continue

                shot_grp = shots_group[shot_key]

                # Load data
                source = tuple(shot_grp['source'][:])
                recv_array = shot_grp['receivers'][:]
                wts_array = shot_grp['weights'][:]
                lengths = shot_grp['ray_lengths'][:]
                data = shot_grp['ray_data'][:]

                # Reconstruct individual ray paths
                offset = 0
                for i, length in enumerate(lengths):
                    if max_rays is not None and total_loaded >= max_rays:
                        break

                    if length > 0:
                        ray_path = data[offset:offset + length]
                        ray_paths.append(ray_path)
                        sources.append(source)
                        receivers.append(tuple(recv_array[i]))
                        shot_ids_list.append(int(shot_key))
                        weights.append(wts_array[i])
                        total_loaded += 1

                    offset += length

                if max_rays is not None and total_loaded >= max_rays:
                    break

            result[ph] = {
                'ray_paths': ray_paths,
                'sources': sources,
                'receivers': receivers,
                'shot_ids': shot_ids_list,
                'weights': weights
            }

            logger.info(f"Loaded {len(ray_paths)} {ph.upper()}-wave ray paths")

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

    with h5py.File(cache_path, 'r') as f:
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
                'dz': grid.attrs.get('dz', None)
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
    """
    stats = {
        'exists': os.path.exists(cache_path),
        'p_wave_shots': 0,
        's_wave_shots': 0,
        'p_wave_rays': 0,
        's_wave_rays': 0,
        'file_size_mb': 0,
        'created': None,
        'last_updated': None
    }

    if not stats['exists']:
        return stats

    stats['file_size_mb'] = os.path.getsize(cache_path) / (1024 * 1024)

    try:
        with h5py.File(cache_path, 'r') as f:
            if 'metadata' in f:
                stats['created'] = f['metadata'].attrs.get('created', None)
                stats['last_updated'] = f['metadata'].attrs.get('last_updated', None)

            for phase in ['p', 's']:
                phase_group = f'{phase}_wave'
                if phase_group in f and 'shots' in f[phase_group]:
                    shots = f[phase_group]['shots']
                    stats[f'{phase}_wave_shots'] = len(shots)

                    # Count total rays
                    total_rays = 0
                    for shot_key in shots.keys():
                        if 'ray_lengths' in shots[shot_key]:
                            total_rays += len(shots[shot_key]['ray_lengths'][:])
                    stats[f'{phase}_wave_rays'] = total_rays
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

    with h5py.File(cache_path, 'r') as f:
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

    with h5py.File(cache_path, 'r') as f:
        phase_group = f'{phase}_wave'
        if phase_group not in f or 'shots' not in f[phase_group]:
            return []

        shots_group = f[phase_group]['shots']
        for shot_key in shots_group.keys():
            recv_array = shots_group[shot_key]['receivers'][:]
            for recv in recv_array:
                receivers.add(tuple(recv))

    return list(receivers)
