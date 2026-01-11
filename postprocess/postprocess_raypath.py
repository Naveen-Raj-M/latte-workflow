"""
Post-processing functions for LATTE ray path visualization.
Traces and visualizes ray paths using traveltime gradient fields.

This module reads traveltime gradient binary files (gradient_px.bin, gradient_py.bin, gradient_pz.bin)
and traces ray paths from receivers back to sources using the steepest descent algorithm.
"""

import os
import numpy as np
import pandas as pd

# Use non-interactive backend for better performance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from typing import Dict, Tuple, List, Optional
from omegaconf import DictConfig
from tqdm import tqdm
import warnings
from joblib import Parallel, delayed

from utils.io_utils import ensure_directory_exists, check_file_exists


def get_performance_setting(config: DictConfig, key: str, default):
    """
    Get performance setting from global performance config.

    Args:
        config: Configuration object
        key: Setting key (e.g., 'n_jobs', 'backend', 'batch_size', 'pre_dispatch')
        default: Default value if not found

    Returns:
        Setting value with proper fallback
    """
    # Check global performance section
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


def get_tracing_setting(config: DictConfig, key: str, default):
    """
    Get ray tracing setting from global tracing config.

    Args:
        config: Configuration object
        key: Setting key (e.g., 'max_steps', 'step_size', 'tolerance', 'interpolation_method')
        default: Default value if not found

    Returns:
        Setting value with proper fallback
    """
    # Check global tracing section
    if hasattr(config, '_parent') and hasattr(config._parent, 'tracing'):
        tracing_config = config._parent.tracing
        if hasattr(tracing_config, key):
            return getattr(tracing_config, key)
    elif hasattr(config, 'tracing'):
        tracing_config = config.tracing
        if hasattr(tracing_config, key):
            return getattr(tracing_config, key)

    # Fallback to default
    return default


def read_gradient_metadata(gradient_dir: str, shot_id: int) -> Dict[str, float]:
    """
    Read gradient metadata file for a specific shot.

    Args:
        gradient_dir: Directory containing gradient files
        shot_id: Shot ID

    Returns:
        Dictionary containing local grid parameters
    """
    metadata_file = f"{gradient_dir}/shot_{shot_id}_gradient_metadata.txt"

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Gradient metadata file not found: {metadata_file}")

    metadata = {}
    with open(metadata_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            # Parse key = value format
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Convert to appropriate type
                if key in ['nx', 'ny', 'nz']:
                    metadata[key] = int(value)
                else:
                    metadata[key] = float(value)

    # CRITICAL FIX: LATTE writes grid spacing in METERS, but we need KILOMETERS
    # Convert dx, dy, dz from meters to kilometers
    if 'dx' in metadata:
        metadata['dx'] = metadata['dx'] / 1000.0
    if 'dy' in metadata:
        metadata['dy'] = metadata['dy'] / 1000.0
    if 'dz' in metadata:
        metadata['dz'] = metadata['dz'] / 1000.0

    # Convert origin coordinates from meters to kilometers
    if 'origin_x' in metadata:
        metadata['origin_x'] = metadata['origin_x'] / 1000.0
    if 'origin_y' in metadata:
        metadata['origin_y'] = metadata['origin_y'] / 1000.0
    if 'origin_z' in metadata:
        metadata['origin_z'] = metadata['origin_z'] / 1000.0

    return metadata


def read_gradient_binaries(gradient_dir: str, shot_id: int, phase: str = 'p',
                           precision: str = 'single') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read gradient binary files for a specific shot.

    Args:
        gradient_dir: Directory containing gradient files
        shot_id: Shot ID
        phase: 'p' or 's' wave
        precision: 'single' or 'double'

    Returns:
        Tuple of (grad_x, grad_y, grad_z) arrays
    """
    dtype = np.float64 if precision == 'double' else np.float32

    # File paths
    grad_px_file = f"{gradient_dir}/shot_{shot_id}_gradient_{phase}x.bin"
    grad_py_file = f"{gradient_dir}/shot_{shot_id}_gradient_{phase}y.bin"
    grad_pz_file = f"{gradient_dir}/shot_{shot_id}_gradient_{phase}z.bin"

    # Check if files exist
    if not os.path.exists(grad_px_file):
        raise FileNotFoundError(f"Gradient file not found: {grad_px_file}")

    # Read gradient files
    grad_x = np.fromfile(grad_px_file, dtype=dtype)
    grad_z = np.fromfile(grad_pz_file, dtype=dtype)

    # Check if 3D (has Y component)
    if os.path.exists(grad_py_file):
        grad_y = np.fromfile(grad_py_file, dtype=dtype)
    else:
        grad_y = None

    return grad_x, grad_y, grad_z


def reshape_gradients(grad_x: np.ndarray, grad_y: Optional[np.ndarray], grad_z: np.ndarray,
                     grid_dims: Dict[str, int]) -> Tuple[np.ndarray, ...]:
    """
    Reshape gradient arrays to 3D grid.

    Args:
        grad_x, grad_y, grad_z: Flattened gradient arrays
        grid_dims: Dictionary with nx, ny, nz, dx, dy, dz, origin_x, origin_y, origin_z

    Returns:
        Reshaped gradient arrays
    """
    nx, ny, nz = grid_dims['nx'], grid_dims['ny'], grid_dims['nz']

    if grad_y is not None:
        # 3D case - use Fortran order since LATTE writes in column-major format
        grad_x = grad_x.reshape((nz, ny, nx), order='F')
        grad_y = grad_y.reshape((nz, ny, nx), order='F')
        grad_z = grad_z.reshape((nz, ny, nx), order='F')
        return grad_x, grad_y, grad_z
    else:
        # 2D case - use Fortran order since LATTE writes in column-major format
        grad_x = grad_x.reshape((nz, nx), order='F')
        grad_z = grad_z.reshape((nz, nx), order='F')
        return grad_x, None, grad_z


def interpolate_gradient_trilinear(grad_x: np.ndarray, grad_y: Optional[np.ndarray], grad_z: np.ndarray,
                                   pos: np.ndarray, grid_dims: Dict[str, int], is_3d: bool) -> np.ndarray:
    """
    Interpolate gradient values using trilinear interpolation.

    Args:
        grad_x, grad_y, grad_z: Gradient field arrays
        pos: Current position [x, y, z] in km
        grid_dims: Grid dimensions dictionary
        is_3d: Whether this is a 3D problem

    Returns:
        Interpolated gradient vector [gx, gy, gz]
    """
    nx, ny, nz = grid_dims['nx'], grid_dims['ny'], grid_dims['nz']
    dx, dy, dz = grid_dims['dx'], grid_dims['dy'], grid_dims['dz']
    ox, oy, oz = grid_dims['origin_x'], grid_dims['origin_y'], grid_dims['origin_z']

    if is_3d:
        # Compute continuous grid coordinates
        x_coord = (pos[0] - ox) / dx
        y_coord = (pos[1] - oy) / dy
        z_coord = (pos[2] - oz) / dz

        # Get integer indices (floor)
        i0 = int(np.floor(x_coord))
        j0 = int(np.floor(y_coord))
        k0 = int(np.floor(z_coord))

        # Get upper indices (clamp to grid bounds)
        i1 = min(i0 + 1, nx - 1)
        j1 = min(j0 + 1, ny - 1)
        k1 = min(k0 + 1, nz - 1)

        # Clamp lower indices too
        i0 = max(0, i0)
        j0 = max(0, j0)
        k0 = max(0, k0)

        # Compute interpolation weights
        wx = x_coord - i0
        wy = y_coord - j0
        wz = z_coord - k0

        # Trilinear interpolation for each gradient component
        # Sample at 8 corners of the cube
        gx = (1-wx)*(1-wy)*(1-wz)*grad_x[k0, j0, i0] + \
             (  wx)*(1-wy)*(1-wz)*grad_x[k0, j0, i1] + \
             (1-wx)*(  wy)*(1-wz)*grad_x[k0, j1, i0] + \
             (  wx)*(  wy)*(1-wz)*grad_x[k0, j1, i1] + \
             (1-wx)*(1-wy)*(  wz)*grad_x[k1, j0, i0] + \
             (  wx)*(1-wy)*(  wz)*grad_x[k1, j0, i1] + \
             (1-wx)*(  wy)*(  wz)*grad_x[k1, j1, i0] + \
             (  wx)*(  wy)*(  wz)*grad_x[k1, j1, i1]

        gy = (1-wx)*(1-wy)*(1-wz)*grad_y[k0, j0, i0] + \
             (  wx)*(1-wy)*(1-wz)*grad_y[k0, j0, i1] + \
             (1-wx)*(  wy)*(1-wz)*grad_y[k0, j1, i0] + \
             (  wx)*(  wy)*(1-wz)*grad_y[k0, j1, i1] + \
             (1-wx)*(1-wy)*(  wz)*grad_y[k1, j0, i0] + \
             (  wx)*(1-wy)*(  wz)*grad_y[k1, j0, i1] + \
             (1-wx)*(  wy)*(  wz)*grad_y[k1, j1, i0] + \
             (  wx)*(  wy)*(  wz)*grad_y[k1, j1, i1]

        gz = (1-wx)*(1-wy)*(1-wz)*grad_z[k0, j0, i0] + \
             (  wx)*(1-wy)*(1-wz)*grad_z[k0, j0, i1] + \
             (1-wx)*(  wy)*(1-wz)*grad_z[k0, j1, i0] + \
             (  wx)*(  wy)*(1-wz)*grad_z[k0, j1, i1] + \
             (1-wx)*(1-wy)*(  wz)*grad_z[k1, j0, i0] + \
             (  wx)*(1-wy)*(  wz)*grad_z[k1, j0, i1] + \
             (1-wx)*(  wy)*(  wz)*grad_z[k1, j1, i0] + \
             (  wx)*(  wy)*(  wz)*grad_z[k1, j1, i1]

        return np.array([gx, gy, gz])
    else:
        # 2D case: bilinear interpolation
        x_coord = (pos[0] - ox) / dx
        z_coord = (pos[2] - oz) / dz

        # Get integer indices (floor)
        i0 = int(np.floor(x_coord))
        k0 = int(np.floor(z_coord))

        # Get upper indices (clamp to grid bounds)
        i1 = min(i0 + 1, nx - 1)
        k1 = min(k0 + 1, nz - 1)

        # Clamp lower indices too
        i0 = max(0, i0)
        k0 = max(0, k0)

        # Compute interpolation weights
        wx = x_coord - i0
        wz = z_coord - k0

        # Bilinear interpolation
        gx = (1-wx)*(1-wz)*grad_x[k0, i0] + \
             (  wx)*(1-wz)*grad_x[k0, i1] + \
             (1-wx)*(  wz)*grad_x[k1, i0] + \
             (  wx)*(  wz)*grad_x[k1, i1]

        gz = (1-wx)*(1-wz)*grad_z[k0, i0] + \
             (  wx)*(1-wz)*grad_z[k0, i1] + \
             (1-wx)*(  wz)*grad_z[k1, i0] + \
             (  wx)*(  wz)*grad_z[k1, i1]

        return np.array([gx, 0.0, gz])


def trace_ray_path(source_pos: Tuple[float, float, float],
                   receiver_pos: Tuple[float, float, float],
                   grad_x: np.ndarray, grad_y: Optional[np.ndarray], grad_z: np.ndarray,
                   grid_dims: Dict[str, int],
                   max_steps: int = 10000,
                   step_size: float = 0.1,
                   tolerance: float = 0.5,
                   interpolation_method: str = 'nearest') -> np.ndarray:
    """
    Trace ray path from receiver to source using gradient descent.

    The ray follows the negative gradient of the traveltime field:
    dx/ds = -∇T / |∇T|

    Args:
        source_pos: (x, y, z) source coordinates in km
        receiver_pos: (x, y, z) receiver coordinates in km
        grad_x, grad_y, grad_z: Gradient fields (∂T/∂x, ∂T/∂y, ∂T/∂z)
        grid_dims: Grid dimensions and spacing
        max_steps: Maximum number of integration steps
        step_size: Integration step size in km
        tolerance: Distance tolerance to source in km
        interpolation_method: 'nearest' or 'trilinear' (default: 'nearest')
                            - 'nearest': Floor-based nearest neighbor sampling (faster, less accurate)
                            - 'trilinear': Trilinear interpolation (slower, more accurate, smoother rays)

    Returns:
        Array of ray path coordinates (n_points, 3)
    """
    is_3d = grad_y is not None

    # Extract grid parameters
    nx, ny, nz = grid_dims['nx'], grid_dims['ny'], grid_dims['nz']
    dx, dy, dz = grid_dims['dx'], grid_dims['dy'], grid_dims['dz']
    ox, oy, oz = grid_dims['origin_x'], grid_dims['origin_y'], grid_dims['origin_z']

    # Initialize ray path
    ray_path = [receiver_pos]
    current_pos = np.array(receiver_pos)

    for step in range(max_steps):
        # Check if we've reached the source
        dist_to_source = np.linalg.norm(current_pos - np.array(source_pos))
        if dist_to_source < tolerance:
            ray_path.append(source_pos)
            break

        # Get gradient at current position using selected interpolation method
        if interpolation_method == 'trilinear':
            # Use trilinear/bilinear interpolation
            gradient = interpolate_gradient_trilinear(
                grad_x, grad_y, grad_z, current_pos, grid_dims, is_3d
            )

            # Check if we're still within reasonable bounds
            # (The interpolation function handles clamping, but we should still check)
            if is_3d:
                x_coord = (current_pos[0] - ox) / dx
                y_coord = (current_pos[1] - oy) / dy
                z_coord = (current_pos[2] - oz) / dz
                if not (-0.5 <= x_coord <= nx - 0.5 and -0.5 <= y_coord <= ny - 0.5 and -0.5 <= z_coord <= nz - 0.5):
                    warnings.warn(f"Ray path left grid bounds at step {step}")
                    break
            else:
                x_coord = (current_pos[0] - ox) / dx
                z_coord = (current_pos[2] - oz) / dz
                if not (-0.5 <= x_coord <= nx - 0.5 and -0.5 <= z_coord <= nz - 0.5):
                    warnings.warn(f"Ray path left grid bounds at step {step}")
                    break
        else:
            # Use nearest neighbor (floor-based) interpolation - original method
            if is_3d:
                i = int((current_pos[0] - ox) / dx)
                j = int((current_pos[1] - oy) / dy)
                k = int((current_pos[2] - oz) / dz)

                # Check bounds
                if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
                    warnings.warn(f"Ray path left grid bounds at step {step}")
                    break

                # Get gradient at current position
                gx = grad_x[k, j, i]
                gy = grad_y[k, j, i]
                gz = grad_z[k, j, i]
                gradient = np.array([gx, gy, gz])
            else:
                # 2D case
                i = int((current_pos[0] - ox) / dx)
                k = int((current_pos[2] - oz) / dz)

                # Check bounds
                if not (0 <= i < nx and 0 <= k < nz):
                    warnings.warn(f"Ray path left grid bounds at step {step}")
                    break

                # Get gradient at current position
                gx = grad_x[k, i]
                gz = grad_z[k, i]
                gradient = np.array([gx, 0.0, gz])  # Y-component is zero in 2D

        # Normalize gradient
        grad_magnitude = np.linalg.norm(gradient)
        if grad_magnitude < 1e-10:
            warnings.warn(f"Gradient too small at step {step}, stopping")
            break

        gradient_normalized = gradient / grad_magnitude

        # Move in negative gradient direction (toward source)
        new_pos = current_pos - step_size * gradient_normalized
        # In 2D case, keep Y coordinate fixed
        if not is_3d:
            new_pos[1] = current_pos[1]
        current_pos = new_pos
        ray_path.append(tuple(current_pos))

    return np.array(ray_path)


def plot_ray_paths_projection(ray_paths: List[np.ndarray],
                              sources: List[Tuple[float, float, float]],
                              receivers: List[Tuple[float, float, float]],
                              plane: str,
                              output_file: str,
                              config: DictConfig) -> None:
    """
    Plot ray paths in a specific 2D projection plane.

    Args:
        ray_paths: List of ray path arrays
        sources: List of source coordinates
        receivers: List of receiver coordinates
        plane: 'xy', 'xz', or 'yz'
        output_file: Output file path
        config: Configuration object
    """
    import time
    t_start = time.time()

    # Subsample ray paths if requested for faster plotting
    # Check for subsample_fraction at root level first (new location), then in plot config (backward compat)
    subsample_fraction = getattr(config, 'subsample_fraction', None)
    if subsample_fraction is None:
        subsample_fraction = getattr(config.plot, 'subsample_fraction', 1.0)

    if subsample_fraction < 1.0 and len(ray_paths) > 0:
        n_rays_original = len(ray_paths)
        n_rays_to_plot = max(1, int(n_rays_original * subsample_fraction))
        # Random subsample for representative visualization
        indices = np.random.choice(n_rays_original, n_rays_to_plot, replace=False)
        ray_paths = [ray_paths[i] for i in sorted(indices)]
        print(f"    Subsampled {n_rays_to_plot}/{n_rays_original} ray paths for plotting ({subsample_fraction*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(config.plot.figure_width, config.plot.figure_height))

    # Determine axes for projection
    if plane == 'xy':
        idx1, idx2 = 0, 1
        xlabel, ylabel = 'X (km)', 'Y (km)'
    elif plane == 'xz':
        idx1, idx2 = 0, 2
        xlabel, ylabel = 'X (km)', 'Z (km)'
    elif plane == 'yz':
        idx1, idx2 = 1, 2
        xlabel, ylabel = 'Y (km)', 'Z (km)'
    else:
        raise ValueError(f"Invalid plane: {plane}")

    # Plot ray paths using LineCollection for much better performance
    if ray_paths:
        # Prepare line segments for LineCollection
        segments = []
        for ray in tqdm(ray_paths, desc=f"Preparing {plane.upper()} plot segments", leave=False):
            if len(ray) > 1:
                # Extract the coordinates for this projection plane
                ray_2d = ray[:, [idx1, idx2]]
                segments.append(ray_2d)

        # Create LineCollection with all ray paths at once
        if segments:
            print(f"    Creating LineCollection with {len(segments)} ray paths...")
            lc = LineCollection(
                segments,
                colors=config.plot.ray_color,
                alpha=config.plot.ray_alpha,
                linewidths=config.plot.ray_linewidth,
                label='Ray paths',
                rasterized=True  # CRITICAL: Rasterize for huge performance gain
            )
            ax.add_collection(lc)
            # LineCollection doesn't auto-scale, so we need to update limits
            ax.autoscale()
            print(f"    LineCollection added ({time.time() - t_start:.1f}s)")

    # Plot sources
    if sources:
        sources_array = np.array(sources)
        ax.scatter(sources_array[:, idx1], sources_array[:, idx2],
                  c=config.plot.source_color,
                  marker=config.plot.source_marker,
                  s=config.plot.source_size,
                  label='Sources',
                  edgecolors='black',
                  linewidth=0.5,
                  zorder=10)

    # Plot receivers
    if receivers:
        receivers_array = np.array(receivers)
        ax.scatter(receivers_array[:, idx1], receivers_array[:, idx2],
                  c=config.plot.receiver_color,
                  marker=config.plot.receiver_marker,
                  s=config.plot.receiver_size,
                  label='Receivers',
                  edgecolors='black',
                  linewidth=0.5,
                  zorder=10)

    ax.set_xlabel(xlabel, fontsize=config.plot.label_fontsize)
    ax.set_ylabel(ylabel, fontsize=config.plot.label_fontsize)
    ax.set_title(f'Ray Paths - {plane.upper()} Plane', fontsize=config.plot.title_fontsize)
    ax.legend(fontsize=config.plot.legend_fontsize)
    ax.grid(config.plot.grid, alpha=0.3)
    ax.set_aspect('equal' if config.plot.equal_aspect else 'auto')

    # Invert z-axis to have depth increase downwards
    if plane in ['xz', 'yz']:
        ax.invert_yaxis()

    print(f"    Saving figure at {config.plot.dpi} DPI...")
    t_save = time.time()
    plt.tight_layout()
    plt.savefig(output_file, dpi=config.plot.dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved {plane.upper()} plot: {output_file} (save took {time.time() - t_save:.1f}s, total {time.time() - t_start:.1f}s)")


def load_shot_mapping(shot_map_file: str) -> Dict[int, str]:
    """Load shot ID to event ID mapping."""
    shot_map = {}
    df = pd.read_csv(shot_map_file, sep='\t')
    for _, row in df.iterrows():
        shot_map[int(row['Shot_ID'])] = str(row['Original_Event_ID'])
    return shot_map


def process_single_ray(shot_id: int,
                       source_pos: Tuple[float, float, float],
                       receiver_pos: Tuple[float, float, float],
                       gradient_dir: str,
                       phase: str,
                       precision: str,
                       tracing_params: Dict,
                       interpolation_method: str = 'nearest') -> Optional[np.ndarray]:
    """
    Process a single shot-receiver pair and trace its ray path.

    This function is designed for fine-grained parallelization over individual rays.

    Args:
        shot_id: Shot ID
        source_pos: Source position (x, y, z) in km
        receiver_pos: Receiver position (x, y, z) in km
        gradient_dir: Directory containing gradient files
        phase: Wave phase ('p' or 's')
        precision: Numerical precision ('single' or 'double')
        tracing_params: Ray tracing parameters dictionary
        interpolation_method: 'nearest' or 'trilinear' interpolation method

    Returns:
        Ray path array or None if failed
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

        return ray_path if len(ray_path) > 1 else None

    except Exception:
        return None


def process_single_shot(shot_id: int,
                        df_gt: pd.DataFrame,
                        gradient_dir: str,
                        phase: str,
                        precision: str,
                        grid_dims: Dict,  # Not used - kept for backward compatibility
                        tracing_params: Dict,
                        verbose: bool,
                        interpolation_method: str = 'nearest') -> Tuple[List[np.ndarray], Tuple, List[Tuple]]:
    """
    Process a single shot and trace all its ray paths.

    Args:
        shot_id: Shot ID to process
        df_gt: Ground truth dataframe
        gradient_dir: Directory containing gradient files
        phase: Wave phase ('p' or 's')
        precision: Numerical precision ('single' or 'double')
        grid_dims: Grid dimension dictionary (will be overridden by metadata)
        tracing_params: Ray tracing parameters dictionary
        verbose: Whether to print warnings
        interpolation_method: 'nearest' or 'trilinear' interpolation method

    Returns:
        Tuple of (ray_paths, source_position, receiver_positions)
    """
    import time
    t_start = time.time()

    try:
        # Read gradient metadata to get the local grid parameters for this shot
        metadata = read_gradient_metadata(gradient_dir, shot_id)

        # Create local grid_dims from metadata (overrides the global grid_dims)
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

        # Get source-receiver pairs for this shot
        shot_data = df_gt[df_gt['shot_id'] == shot_id]

        if len(shot_data) == 0:
            return [], None, []

        # Get source position
        source_row = shot_data.iloc[0]
        source_pos = (
            source_row['event_x_transformed'],
            source_row['event_y_transformed'],
            source_row['event_z_transformed']
        )

        ray_paths = []
        receiver_positions = []

        # Trace rays for each receiver
        slow_rays_count = 0
        for _, row in shot_data.iterrows():
            receiver_pos = (
                row['station_x_transformed'],
                row['station_y_transformed'],
                row['station_z_transformed']
            )
            receiver_positions.append(receiver_pos)

            # Trace ray path using the local grid dimensions for this shot
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
                ray_paths.append(ray_path)
                # Track rays that hit max_steps (potentially problematic)
                if len(ray_path) >= tracing_params['max_steps'] * 0.9:
                    slow_rays_count += 1

        # Warn if shot took too long to process
        elapsed = time.time() - t_start
        if verbose and elapsed > 5.0:
            print(f"  WARNING: Shot {shot_id} took {elapsed:.1f}s ({len(ray_paths)} rays, {slow_rays_count} hit max_steps)")

        return ray_paths, source_pos, receiver_positions

    except FileNotFoundError:
        if verbose:
            warnings.warn(f"Gradient files not found for shot {shot_id}")
        return [], None, []
    except Exception as e:
        if verbose:
            warnings.warn(f"Error processing shot {shot_id}: {e}")
        return [], None, []


def create_ray_path_visualizations_by_pairs(config: DictConfig) -> None:
    """
    Create ray path visualizations by parallelizing over shot-receiver pairs.

    This provides better load balancing than parallelizing over shots,
    especially when individual rays have varying computational costs.

    Args:
        config: Configuration object containing all parameters
    """
    print("\n" + "="*80)
    print("RAY PATH VISUALIZATION FROM GRADIENT FIELDS (PAIR-LEVEL PARALLELIZATION)")
    print("="*80)

    # Create output directory
    ensure_directory_exists(config.output_dir)

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

    print(f"Found {len(df_gt)} source-receiver pairs")

    # Process by phase type
    phase_types = []
    if config.phase_type in ['both', 'p']:
        phase_types.append('p')
    if config.phase_type in ['both', 's']:
        phase_types.append('s')

    for phase in phase_types:
        print(f"\n{'='*80}")
        print(f"Processing {phase.upper()}-wave ray paths")
        print(f"{'='*80}")

        # Prepare task list: one task per shot-receiver pair
        tasks = []
        sources_list = []
        receivers_list = []

        # Get unique shots
        unique_shots = df_gt['shot_id'].unique()

        # Limit number of shots if specified
        if config.max_shots > 0:
            unique_shots = unique_shots[:config.max_shots]

        for shot_id in unique_shots:
            shot_data = df_gt[df_gt['shot_id'] == shot_id]
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
                tasks.append((shot_id, source_pos, receiver_pos))
                sources_list.append(source_pos)
                receivers_list.append(receiver_pos)

        print(f"Created {len(tasks)} ray tracing tasks ({len(unique_shots)} shots)")

        # Get parallelization parameters from global config
        parallel_processing = get_performance_setting(config, 'parallel_processing', True)
        n_jobs = get_performance_setting(config, 'n_jobs', 1) if parallel_processing else 1
        backend = get_performance_setting(config, 'backend', 'loky')
        batch_size = get_performance_setting(config, 'batch_size', 'auto')
        pre_dispatch = get_performance_setting(config, 'pre_dispatch', '2*n_jobs')

        if parallel_processing and n_jobs != 1:
            print(f"Processing tasks using {n_jobs if n_jobs > 0 else 'all'} CPU cores...")
            print(f"  Backend: {backend}, Batch size: {batch_size}, Pre-dispatch: {pre_dispatch}")
        else:
            print(f"Processing tasks sequentially...")

        # Prepare tracing parameters from global config
        tracing_params = {
            'max_steps': get_tracing_setting(config, 'max_steps', 10000),
            'step_size': get_tracing_setting(config, 'step_size', 0.1),
            'tolerance': get_tracing_setting(config, 'tolerance', 0.5)
        }

        # Get interpolation method from global config
        interpolation_method = get_tracing_setting(config, 'interpolation_method', 'nearest')
        print(f"Using '{interpolation_method}' interpolation for ray tracing")

        # Process all shot-receiver pairs in parallel
        ray_paths_all = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            batch_size=batch_size,
            pre_dispatch=pre_dispatch,
            verbose=0
        )(
            delayed(process_single_ray)(
                shot_id,
                source_pos,
                receiver_pos,
                config.gradient_dir,
                phase,
                config.precision,
                tracing_params,
                interpolation_method
            )
            for shot_id, source_pos, receiver_pos in tqdm(tasks, desc=f"Tracing {phase}-wave rays")
        )

        # Filter out None values (failed rays)
        ray_paths_valid = [rp for rp in ray_paths_all if rp is not None]
        sources_valid = [sources_list[i] for i, rp in enumerate(ray_paths_all) if rp is not None]
        receivers_valid = [receivers_list[i] for i, rp in enumerate(ray_paths_all) if rp is not None]

        print(f"\nSuccessfully traced {len(ray_paths_valid)}/{len(tasks)} ray paths")

        # Create plots for each projection
        for plane in config.plot.planes:
            output_file = f"{config.output_dir}/ray_paths_{phase}_wave_{plane}.{config.plot.format}"
            plot_ray_paths_projection(
                ray_paths_valid,
                sources_valid,
                receivers_valid,
                plane,
                output_file,
                config
            )

    print("\n" + "="*80)
    print("RAY PATH VISUALIZATION COMPLETE")
    print("="*80)


def create_ray_path_visualizations(config: DictConfig) -> None:
    """
    Main function to create ray path visualizations from gradient files.

    Args:
        config: Configuration object containing all parameters
    """
    print("\n" + "="*80)
    print("RAY PATH VISUALIZATION FROM GRADIENT FIELDS")
    print("="*80)

    # Create output directory
    ensure_directory_exists(config.output_dir)

    # Load ground truth data
    print(f"\nLoading ground truth data: {config.ground_truth_csv}")
    df_gt = pd.read_csv(config.ground_truth_csv)

    # CRITICAL FIX: Convert coordinates from meters to kilometers
    # The ground truth CSV has coordinates in meters, but we work in km
    coordinate_columns = [
        'event_x_transformed', 'event_y_transformed', 'event_z_transformed',
        'station_x_transformed', 'station_y_transformed', 'station_z_transformed'
    ]
    for col in coordinate_columns:
        if col in df_gt.columns:
            df_gt[col] = df_gt[col] / 1000.0
    print("  Converted coordinates from meters to kilometers")

    # Get unique source-receiver pairs
    print(f"Found {len(df_gt)} source-receiver pairs")

    # Load shot mapping if provided
    shot_mapping = {}
    if hasattr(config, 'shot_map_file') and config.shot_map_file:
        print(f"Loading shot mapping: {config.shot_map_file}")
        shot_mapping = load_shot_mapping(config.shot_map_file)

    # Process by phase type
    phase_types = []
    if config.phase_type in ['both', 'p']:
        phase_types.append('p')
    if config.phase_type in ['both', 's']:
        phase_types.append('s')

    for phase in phase_types:
        print(f"\n{'='*80}")
        print(f"Processing {phase.upper()}-wave ray paths")
        print(f"{'='*80}")

        ray_paths_all = []
        sources_all = []
        receivers_all = []

        # Get unique shots
        unique_shots = df_gt['shot_id'].unique()

        # Limit number of shots if specified
        if config.max_shots > 0:
            unique_shots = unique_shots[:config.max_shots]

        # Get parallelization parameters from global config
        parallel_processing = get_performance_setting(config, 'parallel_processing', True)
        n_jobs = get_performance_setting(config, 'n_jobs', 1) if parallel_processing else 1
        backend = get_performance_setting(config, 'backend', 'loky')
        batch_size = get_performance_setting(config, 'batch_size', 'auto')
        pre_dispatch = get_performance_setting(config, 'pre_dispatch', '2*n_jobs')

        if parallel_processing and n_jobs != 1:
            print(f"Processing {len(unique_shots)} shots using {n_jobs if n_jobs > 0 else 'all'} CPU cores...")
            print(f"  Backend: {backend}, Batch size: {batch_size}, Pre-dispatch: {pre_dispatch}")
        else:
            print(f"Processing {len(unique_shots)} shots sequentially...")

        # Prepare tracing parameters from global config
        tracing_params = {
            'max_steps': get_tracing_setting(config, 'max_steps', 10000),
            'step_size': get_tracing_setting(config, 'step_size', 0.1),
            'tolerance': get_tracing_setting(config, 'tolerance', 0.5)
        }

        # Get interpolation method from global config
        interpolation_method = get_tracing_setting(config, 'interpolation_method', 'nearest')
        print(f"Using '{interpolation_method}' interpolation for ray tracing")

        # Process shots in parallel or sequential
        # Convert to list to avoid generator issues with tqdm and allow proper progress tracking
        shot_list = list(unique_shots)

        results = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            batch_size=batch_size,
            pre_dispatch=pre_dispatch,
            verbose=0  # Suppress joblib verbose output
        )(
            delayed(process_single_shot)(
                shot_id,
                df_gt,
                config.gradient_dir,
                phase,
                config.precision,
                dict(config.grid),  # Convert DictConfig to dict
                tracing_params,
                config.verbose,
                interpolation_method
            )
            for shot_id in tqdm(shot_list, desc=f"Tracing {phase}-wave rays")
        )

        # Aggregate results from all shots
        for ray_paths, source_pos, receiver_positions in results:
            if source_pos is not None:
                sources_all.append(source_pos)
                ray_paths_all.extend(ray_paths)
                receivers_all.extend(receiver_positions)

        print(f"\nSuccessfully traced {len(ray_paths_all)} ray paths")

        # Create plots for each projection
        for plane in config.plot.planes:
            output_file = f"{config.output_dir}/ray_paths_{phase}_wave_{plane}.{config.plot.format}"
            plot_ray_paths_projection(
                ray_paths_all,
                sources_all,
                receivers_all,
                plane,
                output_file,
                config
            )

    print("\n" + "="*80)
    print("RAY PATH VISUALIZATION COMPLETE")
    print("="*80)


def create_visualizations_from_precomputed_rays(config: DictConfig, precomputed_rays: Dict) -> None:
    """
    Create ray path visualizations from pre-computed ray paths.

    This function takes pre-computed ray paths (from compute_all_ray_paths)
    and generates visualization plots without re-tracing rays.

    Args:
        config: Visualization configuration object
        precomputed_rays: Dictionary with phase keys ('p', 's') containing:
            - 'ray_paths': List of ray path arrays
            - 'sources': List of source positions
            - 'receivers': List of receiver positions
    """
    print("\n" + "="*80)
    print("RAY PATH VISUALIZATION (FROM PRE-COMPUTED RAYS)")
    print("="*80)

    # Create output directory
    ensure_directory_exists(config.output_dir)

    # Normalize config structure for compatibility with plot_ray_paths_projection
    # The plotting function expects config.plot, but we have config.visualization.plot
    if hasattr(config, 'visualization') and hasattr(config.visualization, 'plot'):
        # Create a temporary config object with plot at the root level
        from omegaconf import OmegaConf
        plot_config = config.visualization.plot
        # Create a shallow wrapper that has 'plot' at the root and subsample_fraction
        config_dict = {'plot': OmegaConf.to_container(plot_config)}
        # Pass subsample_fraction from parent config if it exists
        if hasattr(config, 'subsample_fraction'):
            config_dict['subsample_fraction'] = config.subsample_fraction
        config_for_plotting = OmegaConf.create(config_dict)
    else:
        config_for_plotting = config

    # Get plotting config
    plot_cfg = config.visualization.plot if hasattr(config, 'visualization') else config.plot

    # Create plots for each phase
    for phase, phase_data in precomputed_rays.items():
        print(f"\n{'='*80}")
        print(f"Creating {phase.upper()}-wave visualization")
        print(f"{'='*80}")

        ray_paths = phase_data['ray_paths']
        sources = phase_data['sources']
        receivers = phase_data['receivers']

        print(f"  Ray paths to plot: {len(ray_paths)}")

        # Create plots for each projection plane
        for plane in plot_cfg.planes:
            output_file = f"{config.output_dir}/ray_paths_{phase}_wave_{plane}.{plot_cfg.format}"
            plot_ray_paths_projection(
                ray_paths,
                sources,
                receivers,
                plane,
                output_file,
                config_for_plotting  # Use normalized config with plot at root level
            )

    print("\n" + "="*80)
    print("RAY PATH VISUALIZATION COMPLETE")
    print("="*80)
