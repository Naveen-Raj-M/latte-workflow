"""
Post-processing functions for LATTE ray path visualization.
Traces and visualizes ray paths using traveltime gradient fields.

This module reads traveltime gradient binary files (gradient_px.bin, gradient_py.bin, gradient_pz.bin)
and traces ray paths from receivers back to sources using the steepest descent algorithm.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Dict, Tuple, List, Optional
from omegaconf import DictConfig
from tqdm import tqdm
import warnings

from utils.io_utils import ensure_directory_exists, check_file_exists


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
        # 3D case
        grad_x = grad_x.reshape((nz, ny, nx))
        grad_y = grad_y.reshape((nz, ny, nx))
        grad_z = grad_z.reshape((nz, ny, nx))
        return grad_x, grad_y, grad_z
    else:
        # 2D case
        grad_x = grad_x.reshape((nz, nx))
        grad_z = grad_z.reshape((nz, nx))
        return grad_x, None, grad_z


def trace_ray_path(source_pos: Tuple[float, float, float],
                   receiver_pos: Tuple[float, float, float],
                   grad_x: np.ndarray, grad_y: Optional[np.ndarray], grad_z: np.ndarray,
                   grid_dims: Dict[str, int],
                   max_steps: int = 10000,
                   step_size: float = 0.1,
                   tolerance: float = 0.5) -> np.ndarray:
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

        # Convert current position to grid indices
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
            gradient = np.array([gx, current_pos[1], gz])  # Keep Y fixed in 2D

        # Normalize gradient
        grad_magnitude = np.linalg.norm(gradient)
        if grad_magnitude < 1e-10:
            warnings.warn(f"Gradient too small at step {step}, stopping")
            break

        gradient_normalized = gradient / grad_magnitude

        # Move in negative gradient direction (toward source)
        current_pos = current_pos - step_size * gradient_normalized
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

    # Plot ray paths
    for ray in ray_paths:
        if len(ray) > 1:
            ax.plot(ray[:, idx1], ray[:, idx2],
                   color=config.plot.ray_color,
                   alpha=config.plot.ray_alpha,
                   linewidth=config.plot.ray_linewidth)

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

    plt.tight_layout()
    plt.savefig(output_file, dpi=config.plot.dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved {plane.upper()} plot: {output_file}")


def load_shot_mapping(shot_map_file: str) -> Dict[int, str]:
    """Load shot ID to event ID mapping."""
    shot_map = {}
    df = pd.read_csv(shot_map_file, sep='\t')
    for _, row in df.iterrows():
        shot_map[int(row['Shot_ID'])] = str(row['Original_Event_ID'])
    return shot_map


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
        unique_shots = df_gt['event_id'].unique() if 'event_id' in df_gt.columns else df_gt['shot_id'].unique()

        # Limit number of shots if specified
        if config.max_shots > 0:
            unique_shots = unique_shots[:config.max_shots]

        print(f"Processing {len(unique_shots)} shots...")

        for shot_id in tqdm(unique_shots, desc=f"Tracing {phase}-wave rays"):
            try:
                # Read gradient files
                grad_x, grad_y, grad_z = read_gradient_binaries(
                    config.gradient_dir, shot_id, phase, config.precision
                )

                # Reshape gradients
                grad_x, grad_y, grad_z = reshape_gradients(
                    grad_x, grad_y, grad_z, config.grid
                )

                # Get source-receiver pairs for this shot
                shot_data = df_gt[df_gt['event_id'] == shot_id] if 'event_id' in df_gt.columns \
                           else df_gt[df_gt['shot_id'] == shot_id]

                if len(shot_data) == 0:
                    continue

                # Get source position
                source_row = shot_data.iloc[0]
                source_pos = (
                    source_row['event_x_transformed'],
                    source_row['event_y_transformed'],
                    source_row['event_z_transformed']
                )
                sources_all.append(source_pos)

                # Trace rays for each receiver
                for _, row in shot_data.iterrows():
                    receiver_pos = (
                        row['station_x_transformed'],
                        row['station_y_transformed'],
                        row['station_z_transformed']
                    )
                    receivers_all.append(receiver_pos)

                    # Trace ray path
                    ray_path = trace_ray_path(
                        source_pos,
                        receiver_pos,
                        grad_x, grad_y, grad_z,
                        config.grid,
                        max_steps=config.tracing.max_steps,
                        step_size=config.tracing.step_size,
                        tolerance=config.tracing.tolerance
                    )

                    if len(ray_path) > 1:
                        ray_paths_all.append(ray_path)

            except FileNotFoundError:
                if config.verbose:
                    print(f"  Warning: Gradient files not found for shot {shot_id}")
                continue
            except Exception as e:
                if config.verbose:
                    print(f"  Error processing shot {shot_id}: {e}")
                continue

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
