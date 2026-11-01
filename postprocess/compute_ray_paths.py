"""
Shared ray path cache management module for LATTE post-processing.

Provides HDF5 cache management for ray paths read from pre-computed binary files.
The cache supports:
- Incremental computation (resume from where you left off)
- Config validation (recompute if grid parameters change)
- Batch-wise saving with immediate memory cleanup
"""

import os
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set

import h5py
import numpy as np
import pandas as pd
from omegaconf import DictConfig


# =============================================================================
# CACHE MANAGEMENT FUNCTIONS
# =============================================================================

def compute_config_hash(config: DictConfig) -> str:
    """
    Compute a hash of critical parameters that affect ray tracing results.

    If any of these parameters change, the cache must be invalidated.

    Args:
        config: Ray path processing configuration

    Returns:
        MD5 hash string of the critical parameters
    """
    grid = config.grid
    params = {
        'grid': {
            'nx': grid.nx, 'ny': grid.ny, 'nz': grid.nz,
            'dx': grid.dx, 'dy': grid.dy, 'dz': grid.dz,
            'origin_x': grid.origin_x, 'origin_y': grid.origin_y, 'origin_z': grid.origin_z
        },
        'raypath_binary_dir': getattr(config, 'raypath_binary_dir', None)
    }
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()


def get_cache_path(config: DictConfig) -> str:
    """
    Get the cache file path from config.

    Args:
        config: Ray path processing configuration

    Returns:
        Path to the cache HDF5 file
    """
    return config.ray_path_output_file


def is_cache_valid(cache_path: str, config: DictConfig) -> bool:
    """
    Check if the cache file exists and has matching configuration.

    Args:
        cache_path: Path to the cache file
        config: Current configuration

    Returns:
        True if cache is valid and can be used
    """
    if not os.path.exists(cache_path):
        return False

    try:
        with h5py.File(cache_path, 'r') as f:
            if 'metadata' not in f:
                return False

            stored_hash = f['metadata'].attrs.get('config_hash', '')
            current_hash = compute_config_hash(config)

            if stored_hash != current_hash:
                print(f"  Cache config mismatch (stored: {stored_hash[:8]}..., current: {current_hash[:8]}...)")
                return False

            return True
    except Exception as e:
        print(f"  Cache validation error: {e}")
        return False


def initialize_cache(cache_path: str, config: DictConfig) -> None:
    """
    Initialize a new cache file with metadata.

    Args:
        cache_path: Path to the cache file
        config: Configuration to store
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    with h5py.File(cache_path, 'w') as f:
        # Create metadata group
        meta = f.create_group('metadata')
        meta.attrs['config_hash'] = compute_config_hash(config)
        meta.attrs['created'] = datetime.now().isoformat()
        meta.attrs['ground_truth_csv'] = config.ground_truth_csv

        # Store grid parameters
        grid_grp = meta.create_group('grid')
        grid_grp.attrs['nx'] = config.grid.nx
        grid_grp.attrs['ny'] = config.grid.ny
        grid_grp.attrs['nz'] = config.grid.nz
        grid_grp.attrs['dx'] = config.grid.dx
        grid_grp.attrs['dy'] = config.grid.dy
        grid_grp.attrs['dz'] = config.grid.dz

        # Create phase groups
        f.create_group('p_wave')
        f.create_group('s_wave')

    print(f"  Initialized new cache: {cache_path}")


def get_computed_shots(cache_path: str, phase: str) -> Set[int]:
    """
    Get the set of shot IDs that have already been computed.

    Args:
        cache_path: Path to the cache file
        phase: 'p' or 's'

    Returns:
        Set of shot IDs that are in the cache
    """
    if not os.path.exists(cache_path):
        return set()

    try:
        with h5py.File(cache_path, 'r') as f:
            phase_group = f'{phase}_wave'
            if phase_group not in f:
                return set()

            if 'shots' not in f[phase_group]:
                return set()

            return set(int(sid) for sid in f[phase_group]['shots'].keys())
    except Exception:
        return set()


def get_computed_receivers_for_shot(cache_path: str, phase: str, shot_id: int) -> Set[Tuple[float, float, float]]:
    """
    Get the set of receiver positions that have been computed for a specific shot.

    Args:
        cache_path: Path to the cache file
        phase: 'p' or 's'
        shot_id: Shot ID

    Returns:
        Set of receiver positions (as tuples) that are in the cache for this shot
    """
    if not os.path.exists(cache_path):
        return set()

    try:
        with h5py.File(cache_path, 'r') as f:
            shot_path = f'{phase}_wave/shots/{shot_id}'
            if shot_path not in f:
                return set()

            receivers = f[shot_path]['receivers'][:]
            return set(tuple(r) for r in receivers)
    except Exception:
        return set()


def save_batch_to_cache(
    cache_path: str,
    phase: str,
    batch_results: Dict[int, Dict]
) -> int:
    """
    Save a batch of computed ray paths to the HDF5 cache.

    Args:
        cache_path: Path to the cache file
        phase: 'p' or 's'
        batch_results: Dictionary mapping shot_id to:
            - 'source': (x, y, z) tuple
            - 'receivers': list of (x, y, z) tuples
            - 'weights': list of weights
            - 'ray_paths': list of numpy arrays (each shape (N, 3))

    Returns:
        Number of rays saved
    """
    total_rays = 0

    with h5py.File(cache_path, 'a') as f:
        phase_group = f'{phase}_wave'

        # Ensure shots group exists
        if 'shots' not in f[phase_group]:
            f[phase_group].create_group('shots')

        shots_group = f[phase_group]['shots']

        for shot_id, data in batch_results.items():
            shot_key = str(shot_id)

            # If shot already exists, we need to handle merging
            # For simplicity, we'll skip already computed shots
            # (incremental computation handles this at the task level)
            if shot_key in shots_group:
                continue

            shot_grp = shots_group.create_group(shot_key)

            # Store source position
            source = np.array(data['source'], dtype=np.float64)
            shot_grp.create_dataset('source', data=source)

            # Store receivers
            receivers = np.array(data['receivers'], dtype=np.float64)
            shot_grp.create_dataset('receivers', data=receivers, compression='gzip', compression_opts=4)

            # Store weights
            weights = np.array(data['weights'], dtype=np.float64)
            shot_grp.create_dataset('weights', data=weights)

            # Store ray paths: concatenate all rays and store lengths
            # Support two formats:
            # 1. Legacy: data['ray_paths'] = list of arrays (requires vstack)
            # 2. Optimized: data['ray_lengths'] + data['ray_data'] (pre-concatenated)

            if 'ray_data' in data and 'ray_lengths' in data:
                # Format 2: Pre-concatenated (binary reader optimization)
                ray_lengths = data['ray_lengths']
                ray_data = data['ray_data']

                shot_grp.create_dataset('ray_lengths', data=ray_lengths)
                shot_grp.create_dataset('ray_data', data=ray_data, compression='gzip', compression_opts=4)

                total_rays += len(ray_lengths)
            elif 'ray_paths' in data:
                # Format 1: List of arrays (gradient-based, requires vstack)
                ray_paths = data['ray_paths']
                if ray_paths:
                    ray_lengths = np.array([len(rp) for rp in ray_paths], dtype=np.int32)
                    ray_data = np.vstack(ray_paths).astype(np.float32)

                    shot_grp.create_dataset('ray_lengths', data=ray_lengths)
                    shot_grp.create_dataset('ray_data', data=ray_data, compression='gzip', compression_opts=4)

                    total_rays += len(ray_paths)
                else:
                    # No valid rays for this shot
                    shot_grp.create_dataset('ray_lengths', data=np.array([], dtype=np.int32))
                    shot_grp.create_dataset('ray_data', data=np.array([]).reshape(0, 3).astype(np.float32))
            else:
                # Empty shot
                shot_grp.create_dataset('ray_lengths', data=np.array([], dtype=np.int32))
                shot_grp.create_dataset('ray_data', data=np.array([]).reshape(0, 3).astype(np.float32))

        # Update metadata with last update time
        f['metadata'].attrs['last_updated'] = datetime.now().isoformat()

    return total_rays


def load_rays_from_cache(
    cache_path: str,
    phase: str,
    shot_ids: Optional[List[int]] = None,
    max_rays: Optional[int] = None
) -> Dict:
    """
    Load ray paths from cache.

    Args:
        cache_path: Path to the cache file
        phase: 'p' or 's'
        shot_ids: Optional list of shot IDs to load. If None, loads all.
        max_rays: Optional maximum number of rays to load (for memory efficiency)

    Returns:
        Dictionary with:
            - 'ray_paths': List of ray path arrays
            - 'sources': List of source positions
            - 'receivers': List of receiver positions
            - 'shot_ids': List of shot IDs
            - 'weights': List of weights
    """
    ray_paths = []
    sources = []
    receivers = []
    shot_ids_list = []
    weights = []

    with h5py.File(cache_path, 'r') as f:
        phase_group = f'{phase}_wave'

        if 'shots' not in f[phase_group]:
            return {
                'ray_paths': ray_paths,
                'sources': sources,
                'receivers': receivers,
                'shot_ids': shot_ids_list,
                'weights': weights
            }

        shots_group = f[phase_group]['shots']

        # Get all shot IDs if not specified
        if shot_ids is None:
            shot_ids = [int(sid) for sid in shots_group.keys()]

        total_loaded = 0
        corrupted_shots = []
        for shot_id in shot_ids:
            shot_key = str(shot_id)
            if shot_key not in shots_group:
                continue

            shot_grp = shots_group[shot_key]

            try:
                # Load data (may fail if compressed data is corrupted)
                source = tuple(shot_grp['source'][:])
                recv_array = shot_grp['receivers'][:]
                wts_array = shot_grp['weights'][:]
                lengths = shot_grp['ray_lengths'][:]
                data = shot_grp['ray_data'][:]
            except OSError as e:
                # Skip corrupted shots (gzip decompression failure)
                if "filter returned failure during read" in str(e):
                    corrupted_shots.append(shot_id)
                    continue
                else:
                    raise

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
                    shot_ids_list.append(shot_id)
                    weights.append(wts_array[i])
                    total_loaded += 1

                offset += length

            if max_rays is not None and total_loaded >= max_rays:
                break

        # Warn if corrupted shots were found
        if corrupted_shots:
            import warnings
            warnings.warn(
                f"Skipped {len(corrupted_shots)} corrupted shots in cache: {corrupted_shots}. "
                f"Enable fix_corrupted_shots=true to automatically recompute them, or use force_recompute=true to regenerate entire cache.",
                UserWarning
            )

    return {
        'ray_paths': ray_paths,
        'sources': sources,
        'receivers': receivers,
        'shot_ids': shot_ids_list,
        'weights': weights
    }


def find_corrupted_shots(cache_path: str, phase: str) -> List[int]:
    """
    Scan cache to find corrupted shots that cannot be read.

    Args:
        cache_path: Path to the cache file
        phase: 'p' or 's'

    Returns:
        List of corrupted shot IDs
    """
    corrupted_shots = []

    if not os.path.exists(cache_path):
        return corrupted_shots

    try:
        with h5py.File(cache_path, 'r') as f:
            phase_group = f'{phase}_wave'

            if phase_group not in f or 'shots' not in f[phase_group]:
                return corrupted_shots

            shots_group = f[phase_group]['shots']
            shot_ids = [int(sid) for sid in shots_group.keys()]

            for shot_id in shot_ids:
                shot_key = str(shot_id)
                try:
                    # Try to read the compressed ray_data
                    _ = shots_group[shot_key]['ray_data'][:]
                except OSError as e:
                    if "filter returned failure during read" in str(e):
                        corrupted_shots.append(shot_id)

    except Exception as e:
        print(f"  Warning: Error while scanning for corrupted shots: {e}")

    return corrupted_shots


def get_cache_stats(cache_path: str) -> Dict:
    """
    Get statistics about the cache.

    Args:
        cache_path: Path to the cache file

    Returns:
        Dictionary with cache statistics
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
        print(f"  Error reading cache stats: {e}")

    return stats


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_performance_setting(config: DictConfig, key: str, default):
    """Get performance setting from global performance config."""
    if hasattr(config, '_parent') and hasattr(config._parent, 'performance'):
        perf_config = config._parent.performance
        if hasattr(perf_config, key):
            return getattr(perf_config, key)
    elif hasattr(config, 'performance'):
        perf_config = config.performance
        if hasattr(perf_config, key):
            return getattr(perf_config, key)
    return default


def load_shot_mapping(shot_map_file: str) -> Dict[int, str]:
    """Load shot ID to event ID mapping."""
    shot_map = {}
    df = pd.read_csv(shot_map_file, sep='\t')
    for _, row in df.iterrows():
        shot_map[int(row['Shot_ID'])] = str(row['Original_Event_ID'])
    return shot_map




def load_event_id_list_and_convert_to_shot_ids(
    event_id_list_file: str,
    shot_map_file: str
) -> Optional[set]:
    """
    Load event IDs from a file and convert them to shot IDs using shot mapping.

    Args:
        event_id_list_file: Path to text file with event IDs (one per line)
        shot_map_file: Path to shot mapping file (Shot_ID -> Original_Event_ID)

    Returns:
        Set of shot IDs corresponding to the event IDs in the list, or None if files not found
    """
    if not event_id_list_file or event_id_list_file == "null":
        return None

    try:
        # Read event ID list
        print(f"\nLoading event ID list: {event_id_list_file}")
        with open(event_id_list_file, 'r') as f:
            event_ids = set(line.strip() for line in f if line.strip())
        print(f"  Found {len(event_ids)} event IDs in list")

        # Load shot mapping (Shot_ID -> Event_ID)
        print(f"Loading shot mapping: {shot_map_file}")
        shot_map = load_shot_mapping(shot_map_file)

        # Create reverse mapping (Event_ID -> Shot_ID)
        event_to_shot = {event_id: shot_id for shot_id, event_id in shot_map.items()}

        # Convert event IDs to shot IDs
        shot_ids = set()
        missing_events = []
        for event_id in event_ids:
            if event_id in event_to_shot:
                shot_ids.add(event_to_shot[event_id])
            else:
                missing_events.append(event_id)

        if missing_events:
            print(f"  WARNING: {len(missing_events)} event IDs not found in shot mapping:")
            for event_id in missing_events[:5]:  # Show first 5
                print(f"    - {event_id}")
            if len(missing_events) > 5:
                print(f"    ... and {len(missing_events) - 5} more")

        print(f"  Converted to {len(shot_ids)} shot IDs")
        return shot_ids

    except FileNotFoundError as e:
        print(f"  ERROR: File not found: {e}")
        return None
    except Exception as e:
        print(f"  ERROR: Failed to load event ID list: {e}")
        return None


# =============================================================================
# MAIN COMPUTATION FUNCTION (WITH CACHING)
# =============================================================================

def compute_all_ray_paths(
    config: DictConfig,
    subsample_during_computation: bool = False,
    filter_shot_ids: Optional[set] = None
) -> Dict[str, Dict]:
    """
    Read ray paths from pre-computed binary files and cache to HDF5.

    Args:
        config: Ray path processing configuration object
        subsample_during_computation: If True, apply subsampling.
        filter_shot_ids: Optional set of shot IDs to filter by.

    Returns:
        Dictionary with phase as keys ('p', 's') and values containing:
            {'ray_paths', 'sources', 'receivers', 'shot_ids', 'weights'}
    """
    from postprocess.read_raypath_binary import compute_all_ray_paths_from_binary
    return compute_all_ray_paths_from_binary(
        config, subsample_during_computation, filter_shot_ids
    )

# =============================================================================
# CACHE-ONLY LOADING (FOR VISUALIZATION)
# =============================================================================

def load_ray_paths_from_cache(
    config: DictConfig,
    filter_shot_ids: Optional[Set[int]] = None,
    subsample_fraction: float = 1.0
) -> Dict[str, Dict]:
    """
    Load ray paths from cache without computing anything.

    This function is useful for visualization when ray paths have already
    been computed and cached.

    Args:
        config: Ray path processing configuration
        filter_shot_ids: Optional set of shot IDs to filter
        subsample_fraction: Fraction of rays to load (for memory efficiency)

    Returns:
        Dictionary with phase keys ('p', 's') containing ray path data
    """
    cache_path = get_cache_path(config)

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    if not is_cache_valid(cache_path, config):
        raise ValueError("Cache is invalid (config mismatch). Run computation first.")

    print(f"\nLoading ray paths from cache: {cache_path}")

    # Determine phase types
    phase_type = getattr(config, 'phase_type', 'both')
    phase_types = []
    if phase_type in ['both', 'p']:
        phase_types.append('p')
    if phase_type in ['both', 's']:
        phase_types.append('s')

    result = {}

    for phase in phase_types:
        print(f"  Loading {phase.upper()}-wave rays...")

        # Get total rays in cache
        stats = get_cache_stats(cache_path)
        total_rays = stats[f'{phase}_wave_rays']

        # Calculate max rays to load
        max_rays = None
        if subsample_fraction < 1.0:
            max_rays = max(1, int(total_rays * subsample_fraction))
            print(f"    Subsampling: loading {max_rays}/{total_rays} rays ({subsample_fraction*100:.1f}%)")

        # Load from cache
        shot_ids = list(filter_shot_ids) if filter_shot_ids else None
        phase_result = load_rays_from_cache(cache_path, phase, shot_ids, max_rays)

        print(f"    Loaded {len(phase_result['ray_paths'])} rays")
        result[phase] = phase_result

    return result
