"""
Shared ray path computation module for LATTE post-processing.

This module computes ray paths from gradient fields and caches them to HDF5 files
for efficient storage and retrieval. The cache supports:
- Incremental computation (resume from where you left off)
- Config validation (recompute if tracing parameters change)
- Batch-wise saving with immediate memory cleanup
- Subsample fraction handling (add more rays to existing cache)
"""

import gc
import os
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set

import h5py
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from joblib import Parallel, delayed

from postprocess.postprocess_raypath import (
    process_single_ray,
    get_performance_setting,
    get_tracing_setting,
    load_shot_mapping,
    read_gradient_metadata,
    read_gradient_binaries,
    reshape_gradients,
    trace_ray_path
)


# =============================================================================
# CACHE MANAGEMENT FUNCTIONS
# =============================================================================

def compute_config_hash(config: DictConfig) -> str:
    """
    Compute a hash of critical parameters that affect ray tracing results.

    If any of these parameters change, the cache must be invalidated.

    Args:
        config: Ray path diagnosis configuration

    Returns:
        MD5 hash string of the critical parameters
    """
    # Get tracing parameters
    max_steps = get_tracing_setting(config, 'max_steps', 10000)
    step_size = get_tracing_setting(config, 'step_size', 0.1)
    tolerance = get_tracing_setting(config, 'tolerance', 0.5)
    interpolation = get_tracing_setting(config, 'interpolation_method', 'nearest')

    # Get precision
    precision = getattr(config, 'precision', 'single')

    # Get grid parameters
    grid = config.grid

    params = {
        'max_steps': max_steps,
        'step_size': step_size,
        'tolerance': tolerance,
        'interpolation': interpolation,
        'precision': precision,
        'grid': {
            'nx': grid.nx, 'ny': grid.ny, 'nz': grid.nz,
            'dx': grid.dx, 'dy': grid.dy, 'dz': grid.dz,
            'origin_x': grid.origin_x, 'origin_y': grid.origin_y, 'origin_z': grid.origin_z
        }
    }

    # Create a deterministic string representation
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()


def get_cache_path(config: DictConfig) -> str:
    """
    Get the cache file path from config.

    Args:
        config: Ray path diagnosis configuration

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

        # Store tracing parameters
        tracing = meta.create_group('tracing')
        tracing.attrs['max_steps'] = get_tracing_setting(config, 'max_steps', 10000)
        tracing.attrs['step_size'] = get_tracing_setting(config, 'step_size', 0.1)
        tracing.attrs['tolerance'] = get_tracing_setting(config, 'tolerance', 0.5)
        tracing.attrs['interpolation'] = get_tracing_setting(config, 'interpolation_method', 'nearest')

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
# RAY TRACING HELPER FUNCTIONS
# =============================================================================

def process_shot_with_preloaded_gradients(
    shot_id: int,
    shot_receivers: List[Tuple[Tuple[float, float, float], float]],
    source_pos: Tuple[float, float, float],
    grad_x: np.ndarray,
    grad_y: Optional[np.ndarray],
    grad_z: np.ndarray,
    local_grid_dims: Dict,
    tracing_params: Dict,
    interpolation_method: str = 'nearest'
) -> List[Tuple[Optional[np.ndarray], Tuple[float, float, float], int, float]]:
    """
    Process all receivers for a single shot using pre-loaded gradient data.

    This function is designed for batch processing where gradients are
    pre-loaded into memory before parallel processing begins.

    Args:
        shot_id: Shot ID
        shot_receivers: List of (receiver_pos, weight) tuples for this shot
        source_pos: Source position (x, y, z) in km
        grad_x, grad_y, grad_z: Pre-loaded and reshaped gradient arrays
        local_grid_dims: Grid dimensions dictionary
        tracing_params: Ray tracing parameters dictionary
        interpolation_method: 'nearest' or 'trilinear' interpolation method

    Returns:
        List of (ray_path, receiver_pos, shot_id, weight) tuples for each receiver
    """
    results = []
    for receiver_pos, weight in shot_receivers:
        try:
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

            if len(ray_path) > 1:
                results.append((ray_path, receiver_pos, shot_id, weight))
            else:
                results.append((None, receiver_pos, shot_id, weight))
        except Exception:
            results.append((None, receiver_pos, shot_id, weight))

    return results


# Global metadata cache to avoid redundant reads
# All shots typically have the same grid dimensions
_METADATA_CACHE: Dict[str, Dict] = {}


def _get_cached_metadata(gradient_dir: str, shot_id: int) -> Dict:
    """
    Get metadata from cache or load it once.

    Since all shots typically share the same grid parameters,
    we cache the metadata after the first read.
    """
    cache_key = gradient_dir

    if cache_key not in _METADATA_CACHE:
        # First time: read metadata from this shot
        metadata = read_gradient_metadata(gradient_dir, shot_id)
        _METADATA_CACHE[cache_key] = {
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

    return _METADATA_CACHE[cache_key]


def _load_single_shot_gradients_fast(
    shot_id: int,
    gradient_dir: str,
    phase: str,
    precision: str,
    grid_dims: Dict
) -> Optional[Tuple[int, Dict]]:
    """
    Load gradient files for a single shot using pre-cached metadata.

    Optimizations:
    - Uses pre-cached grid dimensions (no metadata file read)
    - No os.path.exists() checks (uses try/except)
    - Direct file reads with minimal overhead

    Returns:
        Tuple of (shot_id, gradient_dict) or None if loading failed
    """
    dtype = np.float64 if precision == 'double' else np.float32
    nx, ny, nz = grid_dims['nx'], grid_dims['ny'], grid_dims['nz']

    # Build file paths
    grad_px_file = f"{gradient_dir}/shot_{shot_id}_gradient_{phase}x.bin"
    grad_py_file = f"{gradient_dir}/shot_{shot_id}_gradient_{phase}y.bin"
    grad_pz_file = f"{gradient_dir}/shot_{shot_id}_gradient_{phase}z.bin"

    try:
        # Read all gradient files (no existence checks - faster on network FS)
        grad_x = np.fromfile(grad_px_file, dtype=dtype)
        grad_z = np.fromfile(grad_pz_file, dtype=dtype)

        # Try Y component (may not exist for 2D)
        try:
            grad_y = np.fromfile(grad_py_file, dtype=dtype)
        except FileNotFoundError:
            grad_y = None

        # Reshape using cached dimensions
        if grad_y is not None:
            grad_x = grad_x.reshape((nz, ny, nx), order='F')
            grad_y = grad_y.reshape((nz, ny, nx), order='F')
            grad_z = grad_z.reshape((nz, ny, nx), order='F')
        else:
            grad_x = grad_x.reshape((nz, nx), order='F')
            grad_z = grad_z.reshape((nz, nx), order='F')

        return (shot_id, {
            'grad_x': grad_x,
            'grad_y': grad_y,
            'grad_z': grad_z,
            'grid_dims': grid_dims
        })
    except Exception:
        return None


def _load_single_shot_gradients(
    shot_id: int,
    gradient_dir: str,
    phase: str,
    precision: str
) -> Optional[Tuple[int, Dict]]:
    """
    Load gradient files for a single shot. Helper for parallel loading.

    NOTE: This is the legacy function. Use _load_single_shot_gradients_fast
    with preload_gradients_for_batch_optimized for better performance.

    Returns:
        Tuple of (shot_id, gradient_dict) or None if loading failed
    """
    try:
        metadata = read_gradient_metadata(gradient_dir, shot_id)

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

        grad_x, grad_y, grad_z = read_gradient_binaries(
            gradient_dir, shot_id, phase, precision
        )

        grad_x, grad_y, grad_z = reshape_gradients(
            grad_x, grad_y, grad_z, local_grid_dims
        )

        return (shot_id, {
            'grad_x': grad_x,
            'grad_y': grad_y,
            'grad_z': grad_z,
            'grid_dims': local_grid_dims
        })
    except Exception:
        return None


def preload_gradients_for_batch(
    shot_ids: List[int],
    gradient_dir: str,
    phase: str,
    precision: str,
    n_jobs: int = -1,
    use_optimized: bool = True
) -> Dict[int, Dict]:
    """
    Pre-load gradient files for a batch of shots into memory using parallel I/O.

    Args:
        shot_ids: List of shot IDs to load
        gradient_dir: Directory containing gradient files
        phase: Wave phase ('p' or 's')
        precision: Numerical precision ('single' or 'double')
        n_jobs: Number of parallel loaders (-1 = all cores)
        use_optimized: Use optimized loading (cached metadata, no existence checks)

    Returns:
        Dictionary mapping shot_id to {'grad_x', 'grad_y', 'grad_z', 'grid_dims'}
    """
    if not shot_ids:
        return {}

    # For parallel filesystems (Lustre, GPFS), higher concurrency is beneficial
    # Cap at 48 to avoid overwhelming the metadata server
    io_jobs = min(48, n_jobs if n_jobs > 0 else 48)

    if use_optimized:
        # Get cached metadata (read once, reuse for all shots)
        try:
            grid_dims = _get_cached_metadata(gradient_dir, shot_ids[0])
        except Exception:
            # Fallback to non-optimized if metadata read fails
            use_optimized = False

    if use_optimized:
        # Optimized path: use cached metadata, no existence checks
        results = Parallel(
            n_jobs=io_jobs,
            backend='threading',
            verbose=0
        )(
            delayed(_load_single_shot_gradients_fast)(
                shot_id, gradient_dir, phase, precision, grid_dims
            )
            for shot_id in shot_ids
        )
    else:
        # Legacy path: read metadata per shot
        results = Parallel(
            n_jobs=io_jobs,
            backend='threading',
            verbose=0
        )(
            delayed(_load_single_shot_gradients)(
                shot_id, gradient_dir, phase, precision
            )
            for shot_id in shot_ids
        )

    # Build cache from successful loads
    gradient_cache = {}
    for result in results:
        if result is not None:
            shot_id, grad_data = result
            gradient_cache[shot_id] = grad_data

    return gradient_cache


def preload_gradients_for_batch_async(
    shot_ids: List[int],
    gradient_dir: str,
    phase: str,
    precision: str,
    max_workers: int = 48
) -> Dict[int, Dict]:
    """
    Pre-load gradient files using concurrent.futures ThreadPoolExecutor.

    This can be faster than joblib for pure I/O workloads as it has
    lower overhead and better handles the GIL for I/O-bound tasks.

    Args:
        shot_ids: List of shot IDs to load
        gradient_dir: Directory containing gradient files
        phase: Wave phase ('p' or 's')
        precision: Numerical precision ('single' or 'double')
        max_workers: Maximum number of concurrent I/O threads

    Returns:
        Dictionary mapping shot_id to {'grad_x', 'grad_y', 'grad_z', 'grid_dims'}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not shot_ids:
        return {}

    # Get cached metadata
    try:
        grid_dims = _get_cached_metadata(gradient_dir, shot_ids[0])
    except Exception as e:
        print(f"    Warning: Failed to read metadata: {e}")
        return {}

    gradient_cache = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_shot = {
            executor.submit(
                _load_single_shot_gradients_fast,
                shot_id, gradient_dir, phase, precision, grid_dims
            ): shot_id
            for shot_id in shot_ids
        }

        # Collect results as they complete
        for future in as_completed(future_to_shot):
            result = future.result()
            if result is not None:
                shot_id, grad_data = result
                gradient_cache[shot_id] = grad_data

    return gradient_cache


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
    Compute all ray paths for configured phase types (P and/or S waves).

    This function computes ray paths in batches, saves each batch to an HDF5 cache,
    and clears memory after each batch. This allows processing large datasets
    without running out of memory.

    The function supports incremental computation - if the cache already contains
    some shots, only the missing shots will be computed.

    Args:
        config: Ray path diagnosis configuration object
        subsample_during_computation: If True, apply subsampling before ray tracing.
        filter_shot_ids: Optional set of shot IDs to filter by.

    Returns:
        Dictionary with phase as keys ('p', 's') and values containing:
            {
                'ray_paths': List of ray path arrays
                'sources': List of source positions
                'receivers': List of receiver positions
                'shot_ids': List of shot IDs
                'weights': List of observation weights
            }
    """
    print("\n" + "="*80)
    print("COMPUTING RAY PATHS (WITH CACHE)")
    print("="*80)

    # Get cache settings (caching is always enabled)
    force_recompute = getattr(config, 'force_recompute', False)
    cache_path = get_cache_path(config)

    print(f"\nCache settings:")
    print(f"  Cache file: {cache_path}")
    print(f"  Force recompute: {force_recompute}")

    # Check cache validity
    if not force_recompute and is_cache_valid(cache_path, config):
        print(f"  Cache is VALID (config hash matches)")
        cache_stats = get_cache_stats(cache_path)
        print(f"  Existing cache: {cache_stats['p_wave_rays']} P-wave rays, "
              f"{cache_stats['s_wave_rays']} S-wave rays ({cache_stats['file_size_mb']:.1f} MB)")
    else:
        if force_recompute:
            print(f"  Force recompute requested - will create new cache")
        else:
            print(f"  Cache invalid or doesn't exist - will create new cache")

        # Initialize new cache
        initialize_cache(cache_path, config)

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
    print(f"  Total source-receiver pairs: {len(df_gt)}")

    # Determine phase types to process
    phase_type = getattr(config, 'phase_type', 'both')
    phase_types = []
    if phase_type in ['both', 'p']:
        phase_types.append('p')
    if phase_type in ['both', 's']:
        phase_types.append('s')

    result = {}

    # Get performance settings from global config
    parallel_processing = get_performance_setting(config, 'parallel_processing', True)
    n_jobs = get_performance_setting(config, 'n_jobs', 1) if parallel_processing else 1
    backend = get_performance_setting(config, 'backend', 'loky')
    batch_size = get_performance_setting(config, 'batch_size', 'auto')
    pre_dispatch = get_performance_setting(config, 'pre_dispatch', '2*n_jobs')

    # Get tracing parameters from global config
    tracing_params = {
        'max_steps': get_tracing_setting(config, 'max_steps', 10000),
        'step_size': get_tracing_setting(config, 'step_size', 0.1),
        'tolerance': get_tracing_setting(config, 'tolerance', 0.5)
    }
    interpolation_method = get_tracing_setting(config, 'interpolation_method', 'nearest')

    # Get common parameters
    gradient_dir = config.gradient_dir
    precision = getattr(config, 'precision', 'single')
    max_shots = getattr(config, 'max_shots', 0)

    # Get weight column (for DWS)
    weight_column = None
    if hasattr(config, 'dws') and hasattr(config.dws, 'weight_column'):
        weight_column = config.dws.weight_column

    # Get parallel mode setting
    parallel_mode = getattr(config, 'parallel_mode', 'by_pairs')

    print(f"\nGlobal settings:")
    print(f"  Parallel processing: {parallel_processing} (n_jobs={n_jobs}, backend={backend})")
    print(f"  Parallel mode: {parallel_mode}")
    print(f"  Ray tracing: max_steps={tracing_params['max_steps']}, step_size={tracing_params['step_size']}, interpolation={interpolation_method}")
    print(f"  Grid: {config.grid.nx}x{config.grid.ny}x{config.grid.nz}, spacing={config.grid.dx}x{config.grid.dy}x{config.grid.dz} km")

    for phase in phase_types:
        print(f"\n{'='*80}")
        print(f"Computing {phase.upper()}-wave ray paths")
        print(f"{'='*80}")

        # Get already computed shots from cache
        computed_shots = get_computed_shots(cache_path, phase) if not force_recompute else set()
        if computed_shots:
            print(f"  Found {len(computed_shots)} shots already in cache")

        # Check for corrupted shots if requested
        corrupted_shots = set()
        fix_corrupted = getattr(config, 'fix_corrupted_shots', False)
        if fix_corrupted and not force_recompute and computed_shots:
            print(f"  Scanning for corrupted shots...")
            corrupted_shots = set(find_corrupted_shots(cache_path, phase))
            if corrupted_shots:
                print(f"  Found {len(corrupted_shots)} corrupted shots: {sorted(list(corrupted_shots))}")
                print(f"  These will be recomputed")
            else:
                print(f"  No corrupted shots found")

        # Filter by phase if column exists
        if 'phase' in df_gt.columns:
            df_phase = df_gt[df_gt['phase'].str.upper() == phase.upper()].copy()
        else:
            df_phase = df_gt.copy()

        # Get unique shots
        unique_shots = df_phase['shot_id'].unique()

        # Apply event filtering if provided and we're in visualization-only mode
        if filter_shot_ids is not None and subsample_during_computation:
            n_shots_before = len(unique_shots)
            unique_shots = np.array([s for s in unique_shots if s in filter_shot_ids])
            print(f"  EVENT FILTER: Selected {len(unique_shots)}/{n_shots_before} shots from event ID list")

        # Limit number of shots if specified (applied after event filtering)
        if max_shots > 0:
            unique_shots = unique_shots[:max_shots]
            print(f"  Limited to {max_shots} shots")

        # Filter out already computed shots (but include corrupted ones)
        shots_to_compute = [s for s in unique_shots if (s not in computed_shots) or (s in corrupted_shots)]

        if len(shots_to_compute) < len(unique_shots):
            n_skipped = len(unique_shots) - len(shots_to_compute)
            n_corrupted = len([s for s in shots_to_compute if s in corrupted_shots])
            print(f"  Skipping {n_skipped} already computed shots")
            if n_corrupted > 0:
                print(f"  Will recompute {n_corrupted} corrupted shots")
            print(f"  Will compute {len(shots_to_compute)} total shots ({len(shots_to_compute) - n_corrupted} new + {n_corrupted} recomputed)")

        # Get weight column if it exists
        has_weights = weight_column is not None and weight_column in df_phase.columns
        if has_weights:
            print(f"  Using weight column: '{weight_column}'")
        else:
            print(f"  No weights found, using uniform weights (1.0)")

        # Build shot tasks for shots to compute
        shot_tasks = {}
        for shot_id in shots_to_compute:
            shot_data = df_phase[df_phase['shot_id'] == shot_id]
            if len(shot_data) == 0:
                continue

            source_row = shot_data.iloc[0]
            source_pos = (
                source_row['event_x_transformed'],
                source_row['event_y_transformed'],
                source_row['event_z_transformed']
            )

            receivers = []
            for _, row in shot_data.iterrows():
                receiver_pos = (
                    row['station_x_transformed'],
                    row['station_y_transformed'],
                    row['station_z_transformed']
                )
                weight = row[weight_column] if has_weights else 1.0
                receivers.append((receiver_pos, weight))

            shot_tasks[shot_id] = {
                'source_pos': source_pos,
                'receivers': receivers
            }

        total_receivers = sum(len(st['receivers']) for st in shot_tasks.values())
        print(f"  Total ray tracing tasks: {total_receivers} receivers across {len(shot_tasks)} shots")

        # Apply subsampling if requested
        if subsample_during_computation and shot_tasks:
            subsample_fraction = getattr(config, 'subsample_fraction', 1.0)
            if subsample_fraction < 1.0:
                n_shots_original = len(shot_tasks)
                n_shots_to_compute = max(1, int(n_shots_original * subsample_fraction))

                # Random subsample of shots
                shot_ids_sampled = np.random.choice(
                    list(shot_tasks.keys()),
                    n_shots_to_compute,
                    replace=False
                )
                shot_tasks = {sid: shot_tasks[sid] for sid in shot_ids_sampled}

                print(f"  SUBSAMPLE: Selected {len(shot_tasks)}/{n_shots_original} shots ({subsample_fraction*100:.1f}%)")

        # Process in batches
        if shot_tasks:
            # Calculate batch size
            if batch_size == 'auto' or batch_size is None:
                try:
                    import psutil
                    available_mem_gb = psutil.virtual_memory().available / (1024**3)
                    shots_per_batch = max(100, int((available_mem_gb * 0.5 * 1024) / 160))
                except ImportError:
                    shots_per_batch = 500
            else:
                shots_per_batch = int(batch_size)

            # Sort shot IDs for better disk locality on parallel filesystems
            shot_ids_to_process = sorted(shot_tasks.keys())
            n_shots = len(shot_ids_to_process)
            n_batches = (n_shots + shots_per_batch - 1) // shots_per_batch

            print(f"\n  Processing {n_shots} shots in {n_batches} batches ({shots_per_batch} shots/batch)")
            print(f"  Each batch will be saved to cache and memory cleared")

            total_rays_computed = 0
            total_failed = 0

            # Get I/O optimization settings
            io_max_threads = 48
            io_use_async = True
            io_cache_metadata = True

            if hasattr(config, '_parent') and hasattr(config._parent, 'performance'):
                perf = config._parent.performance
                if hasattr(perf, 'io'):
                    io_max_threads = getattr(perf.io, 'max_threads', 48)
                    io_use_async = getattr(perf.io, 'use_async', True)
                    io_cache_metadata = getattr(perf.io, 'cache_metadata', True)
            elif hasattr(config, 'performance') and hasattr(config.performance, 'io'):
                perf_io = config.performance.io
                io_max_threads = getattr(perf_io, 'max_threads', 48)
                io_use_async = getattr(perf_io, 'use_async', True)
                io_cache_metadata = getattr(perf_io, 'cache_metadata', True)

            print(f"  I/O settings: max_threads={io_max_threads}, async={io_use_async}, cache_metadata={io_cache_metadata}")

            for batch_idx in range(n_batches):
                batch_start = batch_idx * shots_per_batch
                batch_end = min(batch_start + shots_per_batch, n_shots)
                batch_shot_ids = shot_ids_to_process[batch_start:batch_end]

                print(f"\n  Batch {batch_idx + 1}/{n_batches}: Processing {len(batch_shot_ids)} shots...")

                # Pre-load gradient files for this batch (parallel I/O)
                import time
                t_io_start = time.time()

                if io_use_async:
                    # Use concurrent.futures for async I/O (lower overhead)
                    gradient_cache = preload_gradients_for_batch_async(
                        batch_shot_ids,
                        gradient_dir,
                        phase,
                        precision,
                        max_workers=io_max_threads
                    )
                else:
                    # Use joblib threading backend
                    gradient_cache = preload_gradients_for_batch(
                        batch_shot_ids,
                        gradient_dir,
                        phase,
                        precision,
                        n_jobs=io_max_threads,
                        use_optimized=io_cache_metadata
                    )

                t_io_elapsed = time.time() - t_io_start
                data_loaded_mb = len(gradient_cache) * 81  # ~81 MB per shot (3 x 27 MB)
                throughput = data_loaded_mb / t_io_elapsed if t_io_elapsed > 0 else 0

                print(f"    Loaded gradients for {len(gradient_cache)}/{len(batch_shot_ids)} shots "
                      f"in {t_io_elapsed:.1f}s ({throughput:.0f} MB/s)")

                # Filter to only shots that loaded successfully
                valid_batch_shots = [sid for sid in batch_shot_ids if sid in gradient_cache]

                if not valid_batch_shots:
                    print(f"    WARNING: No valid gradient files in this batch, skipping...")
                    del gradient_cache
                    gc.collect()
                    continue

                # Process this batch in parallel
                batch_results_raw = Parallel(
                    n_jobs=n_jobs,
                    backend=backend,
                    pre_dispatch=pre_dispatch,
                    verbose=0
                )(
                    delayed(process_shot_with_preloaded_gradients)(
                        shot_id,
                        shot_tasks[shot_id]['receivers'],
                        shot_tasks[shot_id]['source_pos'],
                        gradient_cache[shot_id]['grad_x'],
                        gradient_cache[shot_id]['grad_y'],
                        gradient_cache[shot_id]['grad_z'],
                        gradient_cache[shot_id]['grid_dims'],
                        tracing_params,
                        interpolation_method
                    )
                    for shot_id in tqdm(valid_batch_shots, desc=f"Batch {batch_idx + 1}/{n_batches}")
                )

                # Clear gradient cache immediately
                del gradient_cache
                gc.collect()

                # Format results for cache storage
                batch_results_for_cache = {}
                batch_rays = 0
                batch_failed = 0

                for shot_idx, shot_id in enumerate(valid_batch_shots):
                    shot_result = batch_results_raw[shot_idx]

                    valid_rays = []
                    valid_receivers = []
                    valid_weights = []

                    for ray_path, receiver_pos, _, weight in shot_result:
                        if ray_path is not None:
                            valid_rays.append(ray_path)
                            valid_receivers.append(receiver_pos)
                            valid_weights.append(weight)
                            batch_rays += 1
                        else:
                            batch_failed += 1

                    if valid_rays:  # Only save if we have valid rays
                        batch_results_for_cache[shot_id] = {
                            'source': shot_tasks[shot_id]['source_pos'],
                            'receivers': valid_receivers,
                            'weights': valid_weights,
                            'ray_paths': valid_rays
                        }

                # Clear raw results
                del batch_results_raw
                gc.collect()

                # Save batch to cache
                if batch_results_for_cache:
                    saved = save_batch_to_cache(cache_path, phase, batch_results_for_cache)
                    print(f"    Saved {saved} rays to cache")

                total_rays_computed += batch_rays
                total_failed += batch_failed

                # Clear batch results
                del batch_results_for_cache
                gc.collect()

                print(f"    Batch complete: {batch_rays} rays traced, {batch_failed} failed")
                print(f"    Total so far: {total_rays_computed} rays")

            print(f"\n  Computation complete: {total_rays_computed} rays traced, {total_failed} failed")

        # Only load results from cache if needed for downstream use
        load_for_downstream = getattr(config, 'load_cache_for_downstream', False)

        if load_for_downstream:
            print(f"\n  Loading all {phase.upper()}-wave rays from cache for downstream use...")
            phase_result = load_rays_from_cache(cache_path, phase)
            print(f"    Loaded {len(phase_result['ray_paths'])} rays")

            # Handle event filtering for DWS mode
            if filter_shot_ids is not None and not subsample_during_computation:
                # Filter rays for visualization
                viz_indices = [i for i, sid in enumerate(phase_result['shot_ids']) if sid in filter_shot_ids]
                print(f"  EVENT FILTER: Will plot {len(viz_indices)}/{len(phase_result['ray_paths'])} rays")

                phase_result['viz_ray_paths'] = [phase_result['ray_paths'][i] for i in viz_indices]
                phase_result['viz_sources'] = [phase_result['sources'][i] for i in viz_indices]
                phase_result['viz_receivers'] = [phase_result['receivers'][i] for i in viz_indices]
                phase_result['viz_shot_ids'] = [phase_result['shot_ids'][i] for i in viz_indices]

            result[phase] = phase_result
        else:
            print(f"\n  Skipping cache load (load_cache_for_downstream=false)")
            print(f"  Ray paths are saved in cache: {cache_path}")
            # Return empty result for this phase
            result[phase] = {
                'ray_paths': [],
                'sources': [],
                'receivers': [],
                'shot_ids': [],
                'weights': []
            }

    # Print cache stats
    print(f"\n" + "="*80)
    print("CACHE SUMMARY")
    print("="*80)
    cache_stats = get_cache_stats(cache_path)
    print(f"  Cache file: {cache_path}")
    print(f"  File size: {cache_stats['file_size_mb']:.2f} MB")
    print(f"  P-wave: {cache_stats['p_wave_shots']} shots, {cache_stats['p_wave_rays']} rays")
    print(f"  S-wave: {cache_stats['s_wave_shots']} shots, {cache_stats['s_wave_rays']} rays")

    print("\n" + "="*80)
    print("RAY PATH COMPUTATION COMPLETE")
    print("="*80)

    return result


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
        config: Ray path diagnosis configuration
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
