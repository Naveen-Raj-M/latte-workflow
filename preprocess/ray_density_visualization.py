"""Ray density visualization module.

This module computes and visualizes ray density (coverage) with respect to the 3D velocity model grid.
Uses straight-line rays connecting sources and receivers.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from utils.io_utils import ensure_directory_exists


def compute_grid_dimensions(events_df: pd.DataFrame,
                            stations_df: pd.DataFrame,
                            grid_spacing: Tuple[float, float, float],
                            config: DictConfig) -> Dict:
    """Compute grid dimensions from events and stations.

    Args:
        events_df: DataFrame with event data including x, y, z columns
        stations_df: DataFrame with station data including x, y, z columns
        grid_spacing: (dx, dy, dz) in km
        config: Configuration object

    Returns:
        Dictionary with grid parameters: nx, ny, nz, dx, dy, dz, x_min, y_min, z_min, x_max, y_max, z_max
    """
    logger = logging.getLogger(__name__)

    # Get coordinate columns (try lowercase first, then uppercase)
    x_col = 'x' if 'x' in events_df.columns else 'X_grid_km'
    y_col = 'y' if 'y' in events_df.columns else 'Y_grid_km'
    z_col = 'z' if 'z' in events_df.columns else 'Z_grid_km'

    # Collect all coordinates
    all_x = np.concatenate([events_df[x_col].values, stations_df[x_col].values])
    all_y = np.concatenate([events_df[y_col].values, stations_df[y_col].values])
    all_z = np.concatenate([events_df[z_col].values, stations_df[z_col].values])

    dx, dy, dz = grid_spacing

    # Check if manual bounds are specified
    if hasattr(config, 'initial_velocity_model') and not config.initial_velocity_model.use_auto_bounds:
        x_min = config.initial_velocity_model.x_min
        x_max = config.initial_velocity_model.x_max
        y_min = config.initial_velocity_model.y_min
        y_max = config.initial_velocity_model.y_max
        z_min = config.initial_velocity_model.z_min
        z_max = config.initial_velocity_model.z_max
    else:
        # Auto-compute bounds
        x_min = 0.0
        x_max = np.ceil(all_x.max() / dx) * dx
        y_min = 0.0
        y_max = np.ceil(all_y.max() / dy) * dy
        z_min = np.floor(all_z.min() / dz) * dz
        z_max = np.ceil(all_z.max() / dz) * dz

    # Compute number of grid points
    nx = int(np.round((x_max - x_min) / dx)) + 1
    ny = int(np.round((y_max - y_min) / dy)) + 1
    nz = int(np.round((z_max - z_min) / dz)) + 1

    grid_dims = {
        'nx': nx, 'ny': ny, 'nz': nz,
        'dx': dx, 'dy': dy, 'dz': dz,
        'x_min': x_min, 'y_min': y_min, 'z_min': z_min,
        'x_max': x_max, 'y_max': y_max, 'z_max': z_max
    }

    logger.info(f"Grid dimensions: {nx} x {ny} x {nz} = {nx*ny*nz} cells")
    logger.info(f"  X: [{x_min:.2f}, {x_max:.2f}] km, dx={dx} km")
    logger.info(f"  Y: [{y_min:.2f}, {y_max:.2f}] km, dy={dy} km")
    logger.info(f"  Z: [{z_min:.2f}, {z_max:.2f}] km, dz={dz} km")

    return grid_dims


def find_ray_grid_intersections(source: Tuple[float, float, float],
                                receiver: Tuple[float, float, float],
                                grid_dims: Dict,
                                num_samples: int = 1000) -> np.ndarray:
    """Find which grid cells a straight-line ray passes through.

    Uses simple sampling along the ray to determine grid cell intersections.

    Args:
        source: (x, y, z) source position in km
        receiver: (x, y, z) receiver position in km
        grid_dims: Grid dimension dictionary
        num_samples: Number of sample points along the ray

    Returns:
        Array of unique grid indices (i, j, k) that the ray passes through
    """
    # Create sample points along the ray
    t = np.linspace(0, 1, num_samples)
    ray_x = source[0] + t * (receiver[0] - source[0])
    ray_y = source[1] + t * (receiver[1] - source[1])
    ray_z = source[2] + t * (receiver[2] - source[2])

    # Convert to grid indices (0-based)
    i = np.floor((ray_x - grid_dims['x_min']) / grid_dims['dx']).astype(int)
    j = np.floor((ray_y - grid_dims['y_min']) / grid_dims['dy']).astype(int)
    k = np.floor((ray_z - grid_dims['z_min']) / grid_dims['dz']).astype(int)

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


def compute_ray_density_3d(events_df: pd.DataFrame,
                           stations_df: pd.DataFrame,
                           picks_df: pd.DataFrame,
                           grid_dims: Dict,
                           use_parallel: bool = True,
                           show_progress: bool = True) -> np.ndarray:
    """Compute 3D ray density grid.

    Args:
        events_df: DataFrame with event data
        stations_df: DataFrame with station data
        picks_df: DataFrame with pick data (event_id, station_id pairs)
        grid_dims: Grid dimension dictionary
        use_parallel: Whether to use parallel processing (currently unused, for future optimization)
        show_progress: Whether to show progress bar

    Returns:
        3D array of ray counts with shape (nx, ny, nz)
    """
    logger = logging.getLogger(__name__)
    logger.info("Computing ray density grid...")

    # Initialize density grid
    density = np.zeros((grid_dims['nx'], grid_dims['ny'], grid_dims['nz']), dtype=np.int32)

    # Get coordinate columns
    x_col = 'x' if 'x' in events_df.columns else 'X_grid_km'
    y_col = 'y' if 'y' in events_df.columns else 'Y_grid_km'
    z_col = 'z' if 'z' in events_df.columns else 'Z_grid_km'

    # Create lookup dictionaries for event and station coordinates
    event_coords = {
        row['event_id']: (row[x_col], row[y_col], row[z_col])
        for _, row in events_df.iterrows()
    }

    station_coords = {
        row['station_id']: (row[x_col], row[y_col], row[z_col])
        for _, row in stations_df.iterrows()
    }

    # Process each pick (source-receiver pair)
    n_valid_rays = 0
    iterator = tqdm(picks_df.iterrows(), total=len(picks_df), desc="Computing ray density") if show_progress else picks_df.iterrows()

    for _, pick in iterator:
        event_id = pick['event_id']
        station_id = pick['station_id']

        # Get coordinates
        if event_id not in event_coords or station_id not in station_coords:
            continue

        source = event_coords[event_id]
        receiver = station_coords[station_id]

        # Find grid cells this ray passes through
        grid_cells = find_ray_grid_intersections(source, receiver, grid_dims)

        # Increment density for each cell
        for i, j, k in grid_cells:
            density[i, j, k] += 1

        n_valid_rays += 1

    logger.info(f"Processed {n_valid_rays} valid rays")
    logger.info(f"Max ray density: {density.max()} rays per cell")
    logger.info(f"Mean ray density: {density[density > 0].mean():.2f} rays per cell (non-zero cells)")
    logger.info(f"Grid cells with coverage: {(density > 0).sum()} / {density.size} ({100*(density > 0).sum()/density.size:.1f}%)")

    return density


def plot_ray_density_xy(density_3d: np.ndarray,
                        grid_dims: Dict,
                        output_file: Path,
                        threshold: float,
                        vis_config: DictConfig) -> None:
    """Plot ray density in XY plane (summed over depth).

    Args:
        density_3d: 3D ray density array
        grid_dims: Grid dimension dictionary
        output_file: Output file path
        threshold: Ray count threshold (values below are white)
        vis_config: Visualization configuration
    """
    logger = logging.getLogger(__name__)

    # Sum along Z axis (depth)
    density_xy = np.sum(density_3d, axis=2)  # Shape: (nx, ny)

    # Create coordinate arrays for plotting
    x_coords = np.arange(grid_dims['x_min'], grid_dims['x_max'] + grid_dims['dx']/2, grid_dims['dx'])
    y_coords = np.arange(grid_dims['y_min'], grid_dims['y_max'] + grid_dims['dy']/2, grid_dims['dy'])

    # Create figure
    fig, ax = plt.subplots(figsize=tuple(vis_config.get('figure_size', [12, 10])),
                          dpi=vis_config.get('figure_dpi', 150))

    # Mask values below threshold (make them transparent/white)
    density_masked = np.ma.masked_where(density_xy.T < threshold, density_xy.T)

    # Plot
    cmap = plt.get_cmap(vis_config.get('density_colormap', 'hot'))
    cmap.set_bad(color='white')  # Set masked values to white

    im = ax.pcolormesh(x_coords, y_coords, density_masked,
                       cmap=cmap, shading='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Ray Count (summed over depth)')
    fontsize = vis_config.get('fontsize', {})
    cbar.set_label('Ray Count (summed over depth)',
                   fontsize=fontsize.get('colorbar_label', 14))
    cbar.ax.tick_params(labelsize=fontsize.get('colorbar_ticks', 12))

    # Labels and title
    ax.set_xlabel('X (km)', fontsize=fontsize.get('axis_label', 16))
    ax.set_ylabel('Y (km)', fontsize=fontsize.get('axis_label', 16))
    ax.set_title('Ray Density - XY Plane (Map View)',
                fontsize=fontsize.get('title', 18), fontweight='bold')
    ax.tick_params(axis='both', labelsize=fontsize.get('axis_ticks', 14))
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(output_file, dpi=vis_config.get('figure_dpi', 150), bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved XY plane density plot: {output_file}")


def plot_ray_density_xz(density_3d: np.ndarray,
                        grid_dims: Dict,
                        output_file: Path,
                        threshold: float,
                        vis_config: DictConfig) -> None:
    """Plot ray density in XZ plane (summed over Y).

    Args:
        density_3d: 3D ray density array
        grid_dims: Grid dimension dictionary
        output_file: Output file path
        threshold: Ray count threshold (values below are white)
        vis_config: Visualization configuration
    """
    logger = logging.getLogger(__name__)

    # Sum along Y axis
    density_xz = np.sum(density_3d, axis=1)  # Shape: (nx, nz)

    # Create coordinate arrays for plotting
    x_coords = np.arange(grid_dims['x_min'], grid_dims['x_max'] + grid_dims['dx']/2, grid_dims['dx'])
    z_coords = np.arange(grid_dims['z_min'], grid_dims['z_max'] + grid_dims['dz']/2, grid_dims['dz'])

    # Create figure
    fig, ax = plt.subplots(figsize=tuple(vis_config.get('figure_size', [12, 10])),
                          dpi=vis_config.get('figure_dpi', 150))

    # Mask values below threshold
    density_masked = np.ma.masked_where(density_xz.T < threshold, density_xz.T)

    # Plot
    cmap = plt.get_cmap(vis_config.get('density_colormap', 'hot'))
    cmap.set_bad(color='white')

    im = ax.pcolormesh(x_coords, z_coords, density_masked,
                       cmap=cmap, shading='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Ray Count (summed over Y)')
    fontsize = vis_config.get('fontsize', {})
    cbar.set_label('Ray Count (summed over Y)',
                   fontsize=fontsize.get('colorbar_label', 14))
    cbar.ax.tick_params(labelsize=fontsize.get('colorbar_ticks', 12))

    # Labels and title
    ax.set_xlabel('X (km)', fontsize=fontsize.get('axis_label', 16))
    ax.set_ylabel('Depth (km)', fontsize=fontsize.get('axis_label', 16))
    ax.set_title('Ray Density - XZ Plane (Vertical Cross-Section)',
                fontsize=fontsize.get('title', 18), fontweight='bold')
    ax.tick_params(axis='both', labelsize=fontsize.get('axis_ticks', 14))
    ax.invert_yaxis()  # Depth increases downward
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_file, dpi=vis_config.get('figure_dpi', 150), bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved XZ plane density plot: {output_file}")


def plot_ray_density_yz(density_3d: np.ndarray,
                        grid_dims: Dict,
                        output_file: Path,
                        threshold: float,
                        vis_config: DictConfig) -> None:
    """Plot ray density in YZ plane (summed over X).

    Args:
        density_3d: 3D ray density array
        grid_dims: Grid dimension dictionary
        output_file: Output file path
        threshold: Ray count threshold (values below are white)
        vis_config: Visualization configuration
    """
    logger = logging.getLogger(__name__)

    # Sum along X axis
    density_yz = np.sum(density_3d, axis=0)  # Shape: (ny, nz)

    # Create coordinate arrays for plotting
    y_coords = np.arange(grid_dims['y_min'], grid_dims['y_max'] + grid_dims['dy']/2, grid_dims['dy'])
    z_coords = np.arange(grid_dims['z_min'], grid_dims['z_max'] + grid_dims['dz']/2, grid_dims['dz'])

    # Create figure
    fig, ax = plt.subplots(figsize=tuple(vis_config.get('figure_size', [12, 10])),
                          dpi=vis_config.get('figure_dpi', 150))

    # Mask values below threshold
    density_masked = np.ma.masked_where(density_yz.T < threshold, density_yz.T)

    # Plot
    cmap = plt.get_cmap(vis_config.get('density_colormap', 'hot'))
    cmap.set_bad(color='white')

    im = ax.pcolormesh(y_coords, z_coords, density_masked,
                       cmap=cmap, shading='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Ray Count (summed over X)')
    fontsize = vis_config.get('fontsize', {})
    cbar.set_label('Ray Count (summed over X)',
                   fontsize=fontsize.get('colorbar_label', 14))
    cbar.ax.tick_params(labelsize=fontsize.get('colorbar_ticks', 12))

    # Labels and title
    ax.set_xlabel('Y (km)', fontsize=fontsize.get('axis_label', 16))
    ax.set_ylabel('Depth (km)', fontsize=fontsize.get('axis_label', 16))
    ax.set_title('Ray Density - YZ Plane (Vertical Cross-Section)',
                fontsize=fontsize.get('title', 18), fontweight='bold')
    ax.tick_params(axis='both', labelsize=fontsize.get('axis_ticks', 14))
    ax.invert_yaxis()  # Depth increases downward
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_file, dpi=vis_config.get('figure_dpi', 150), bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved YZ plane density plot: {output_file}")


def plot_ray_density_xy_threshold_count(density_3d: np.ndarray,
                                         grid_dims: Dict,
                                         output_file: Path,
                                         threshold: float,
                                         vis_config: DictConfig) -> None:
    """Plot map view showing count of depth layers with rays >= threshold.

    Each grid cell (x, y) is colored by the number of depth layers below it
    that contain at least 'threshold' rays.

    Args:
        density_3d: 3D ray density array
        grid_dims: Grid dimension dictionary
        output_file: Output file path
        threshold: Ray count threshold
        vis_config: Visualization configuration
    """
    logger = logging.getLogger(__name__)

    # For each (x, y) position, count how many z layers have >= threshold rays
    # density_3d has shape (nx, ny, nz)
    threshold_mask = density_3d >= threshold  # Boolean array
    depth_layer_count = np.sum(threshold_mask, axis=2)  # Sum over Z axis, shape: (nx, ny)

    # Create coordinate arrays for plotting
    x_coords = np.arange(grid_dims['x_min'], grid_dims['x_max'] + grid_dims['dx']/2, grid_dims['dx'])
    y_coords = np.arange(grid_dims['y_min'], grid_dims['y_max'] + grid_dims['dy']/2, grid_dims['dy'])

    # Create figure
    fig, ax = plt.subplots(figsize=tuple(vis_config.get('figure_size', [12, 10])),
                          dpi=vis_config.get('figure_dpi', 150))

    # Mask cells with zero depth layers above threshold
    depth_count_masked = np.ma.masked_where(depth_layer_count.T == 0, depth_layer_count.T)

    # Plot
    cmap = plt.get_cmap(vis_config.get('density_colormap', 'hot'))
    cmap.set_bad(color='white')  # Set masked values to white

    im = ax.pcolormesh(x_coords, y_coords, depth_count_masked,
                       cmap=cmap, shading='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    fontsize = vis_config.get('fontsize', {})
    cbar.set_label(f'Number of depth layers with >= {threshold} rays',
                   fontsize=fontsize.get('colorbar_label', 14))
    cbar.ax.tick_params(labelsize=fontsize.get('colorbar_ticks', 12))

    # Labels and title
    ax.set_xlabel('X (km)', fontsize=fontsize.get('axis_label', 16))
    ax.set_ylabel('Y (km)', fontsize=fontsize.get('axis_label', 16))
    ax.set_title(f'Ray Coverage >= {threshold} Rays (Map View)',
                fontsize=fontsize.get('title', 18), fontweight='bold')
    ax.tick_params(axis='both', labelsize=fontsize.get('axis_ticks', 14))
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(output_file, dpi=vis_config.get('figure_dpi', 150), bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved XY threshold count plot: {output_file}")


def plot_ray_density_depth_slices(density_3d: np.ndarray,
                                   grid_dims: Dict,
                                   output_dir: Path,
                                   depth_values: list,
                                   threshold: float,
                                   vis_config: DictConfig) -> None:
    """Plot horizontal slices at specified depths showing ray density.

    Each grid cell in a slice is colored by the number of rays passing through
    that particular grid cell at that depth.

    Args:
        density_3d: 3D ray density array
        grid_dims: Grid dimension dictionary
        output_dir: Output directory for slice plots
        depth_values: List of depths (in km) to create slices for
        threshold: Ray count threshold (values below are white)
        vis_config: Visualization configuration
    """
    logger = logging.getLogger(__name__)

    # Create coordinate arrays for plotting
    x_coords = np.arange(grid_dims['x_min'], grid_dims['x_max'] + grid_dims['dx']/2, grid_dims['dx'])
    y_coords = np.arange(grid_dims['y_min'], grid_dims['y_max'] + grid_dims['dy']/2, grid_dims['dy'])

    for depth_km in depth_values:
        # Find the closest depth index
        depth_index = int(np.round((depth_km - grid_dims['z_min']) / grid_dims['dz']))

        # Check if depth is within grid bounds
        if depth_index < 0 or depth_index >= grid_dims['nz']:
            logger.warning(f"Depth {depth_km} km is outside grid bounds "
                          f"[{grid_dims['z_min']}, {grid_dims['z_max']}] km, skipping")
            continue

        # Extract the depth slice (XY plane at this Z level)
        actual_depth = grid_dims['z_min'] + depth_index * grid_dims['dz']
        depth_slice = density_3d[:, :, depth_index]  # Shape: (nx, ny)

        # Create figure
        fig, ax = plt.subplots(figsize=tuple(vis_config.get('figure_size', [12, 10])),
                              dpi=vis_config.get('figure_dpi', 150))

        # Mask values below threshold
        slice_masked = np.ma.masked_where(depth_slice.T < threshold, depth_slice.T)

        # Plot
        cmap = plt.get_cmap(vis_config.get('density_colormap', 'hot'))
        cmap.set_bad(color='white')  # Set masked values to white

        im = ax.pcolormesh(x_coords, y_coords, slice_masked,
                          cmap=cmap, shading='auto')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        fontsize = vis_config.get('fontsize', {})
        cbar.set_label('Ray Count',
                      fontsize=fontsize.get('colorbar_label', 14))
        cbar.ax.tick_params(labelsize=fontsize.get('colorbar_ticks', 12))

        # Labels and title
        ax.set_xlabel('X (km)', fontsize=fontsize.get('axis_label', 16))
        ax.set_ylabel('Y (km)', fontsize=fontsize.get('axis_label', 16))
        ax.set_title(f'Ray Density at Depth = {actual_depth:.2f} km',
                    fontsize=fontsize.get('title', 18), fontweight='bold')
        ax.tick_params(axis='both', labelsize=fontsize.get('axis_ticks', 14))
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()

        # Save with depth in filename
        output_file = output_dir / f'ray_density_depth_{actual_depth:.1f}km.png'
        fig.savefig(output_file, dpi=vis_config.get('figure_dpi', 150), bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved depth slice at {actual_depth:.2f} km: {output_file}")


def generate_ray_density_report(density_3d: np.ndarray,
                                 grid_dims: Dict,
                                 threshold: float,
                                 output_file: Path,
                                 config: DictConfig) -> None:
    """Generate a text report with ray density statistics.

    Args:
        density_3d: 3D ray density array
        grid_dims: Grid dimension dictionary
        threshold: Ray count threshold
        output_file: Output file path for the report
        config: Configuration object
    """
    logger = logging.getLogger(__name__)

    with open(output_file, 'w') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("RAY PATH DENSITY COVERAGE REPORT\n")
        f.write("="*80 + "\n\n")

        # Grid information
        f.write("GRID CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Grid dimensions: {grid_dims['nx']} x {grid_dims['ny']} x {grid_dims['nz']}\n")
        f.write(f"Grid spacing: dx={grid_dims['dx']:.3f} km, dy={grid_dims['dy']:.3f} km, dz={grid_dims['dz']:.3f} km\n")
        f.write(f"X range: [{grid_dims['x_min']:.3f}, {grid_dims['x_max']:.3f}] km\n")
        f.write(f"Y range: [{grid_dims['y_min']:.3f}, {grid_dims['y_max']:.3f}] km\n")
        f.write(f"Z range: [{grid_dims['z_min']:.3f}, {grid_dims['z_max']:.3f}] km\n")
        f.write(f"Total 3D cells: {density_3d.size:,}\n")
        f.write(f"Threshold: {threshold} rays\n\n")

        # 3D Statistics
        f.write("3D GRID STATISTICS\n")
        f.write("-"*80 + "\n")
        total_cells = density_3d.size
        zero_cells = np.sum(density_3d == 0)
        below_threshold = np.sum(density_3d < threshold)
        at_or_above_threshold = np.sum(density_3d >= threshold)
        nonzero_cells = total_cells - zero_cells

        f.write(f"Cells with zero rays:      {zero_cells:10,} ({100*zero_cells/total_cells:6.2f}%)\n")
        f.write(f"Cells below threshold:     {below_threshold:10,} ({100*below_threshold/total_cells:6.2f}%)\n")
        f.write(f"Cells at/above threshold:  {at_or_above_threshold:10,} ({100*at_or_above_threshold/total_cells:6.2f}%)\n\n")

        if nonzero_cells > 0:
            f.write("Ray count statistics (non-zero cells only):\n")
            nonzero_values = density_3d[density_3d > 0]
            f.write(f"  Min:        {nonzero_values.min():10,} rays\n")
            f.write(f"  Max:        {nonzero_values.max():10,} rays\n")
            f.write(f"  Mean:       {nonzero_values.mean():10.2f} rays\n")
            f.write(f"  Median:     {np.median(nonzero_values):10.2f} rays\n")
            f.write(f"  Std Dev:    {nonzero_values.std():10.2f} rays\n")
            f.write(f"  25th percentile: {np.percentile(nonzero_values, 25):10.2f} rays\n")
            f.write(f"  75th percentile: {np.percentile(nonzero_values, 75):10.2f} rays\n")
            f.write(f"  95th percentile: {np.percentile(nonzero_values, 95):10.2f} rays\n\n")

        # XY Plane (Map View)
        f.write("="*80 + "\n")
        f.write("XY PLANE (MAP VIEW - SUMMED OVER DEPTH)\n")
        f.write("="*80 + "\n")
        density_xy = np.sum(density_3d, axis=2)  # Sum over Z
        total_cells_xy = density_xy.size
        zero_cells_xy = np.sum(density_xy == 0)
        below_threshold_xy = np.sum(density_xy < threshold)
        at_or_above_threshold_xy = np.sum(density_xy >= threshold)

        f.write(f"Total XY cells: {total_cells_xy:,} ({grid_dims['nx']} x {grid_dims['ny']})\n")
        f.write(f"Cells with zero rays:      {zero_cells_xy:10,} ({100*zero_cells_xy/total_cells_xy:6.2f}%)\n")
        f.write(f"Cells below threshold:     {below_threshold_xy:10,} ({100*below_threshold_xy/total_cells_xy:6.2f}%)\n")
        f.write(f"Cells at/above threshold:  {at_or_above_threshold_xy:10,} ({100*at_or_above_threshold_xy/total_cells_xy:6.2f}%)\n\n")

        if zero_cells_xy < total_cells_xy:
            nonzero_xy = density_xy[density_xy > 0]
            f.write("Ray count statistics (non-zero cells only):\n")
            f.write(f"  Min:        {nonzero_xy.min():10,} rays\n")
            f.write(f"  Max:        {nonzero_xy.max():10,} rays\n")
            f.write(f"  Mean:       {nonzero_xy.mean():10.2f} rays\n")
            f.write(f"  Median:     {np.median(nonzero_xy):10.2f} rays\n")
            f.write(f"  Std Dev:    {nonzero_xy.std():10.2f} rays\n\n")

        # XZ Plane (Vertical Cross-Section)
        f.write("="*80 + "\n")
        f.write("XZ PLANE (VERTICAL CROSS-SECTION - SUMMED OVER Y)\n")
        f.write("="*80 + "\n")
        density_xz = np.sum(density_3d, axis=1)  # Sum over Y
        total_cells_xz = density_xz.size
        zero_cells_xz = np.sum(density_xz == 0)
        below_threshold_xz = np.sum(density_xz < threshold)
        at_or_above_threshold_xz = np.sum(density_xz >= threshold)

        f.write(f"Total XZ cells: {total_cells_xz:,} ({grid_dims['nx']} x {grid_dims['nz']})\n")
        f.write(f"Cells with zero rays:      {zero_cells_xz:10,} ({100*zero_cells_xz/total_cells_xz:6.2f}%)\n")
        f.write(f"Cells below threshold:     {below_threshold_xz:10,} ({100*below_threshold_xz/total_cells_xz:6.2f}%)\n")
        f.write(f"Cells at/above threshold:  {at_or_above_threshold_xz:10,} ({100*at_or_above_threshold_xz/total_cells_xz:6.2f}%)\n\n")

        if zero_cells_xz < total_cells_xz:
            nonzero_xz = density_xz[density_xz > 0]
            f.write("Ray count statistics (non-zero cells only):\n")
            f.write(f"  Min:        {nonzero_xz.min():10,} rays\n")
            f.write(f"  Max:        {nonzero_xz.max():10,} rays\n")
            f.write(f"  Mean:       {nonzero_xz.mean():10.2f} rays\n")
            f.write(f"  Median:     {np.median(nonzero_xz):10.2f} rays\n")
            f.write(f"  Std Dev:    {nonzero_xz.std():10.2f} rays\n\n")

        # YZ Plane (Vertical Cross-Section)
        f.write("="*80 + "\n")
        f.write("YZ PLANE (VERTICAL CROSS-SECTION - SUMMED OVER X)\n")
        f.write("="*80 + "\n")
        density_yz = np.sum(density_3d, axis=0)  # Sum over X
        total_cells_yz = density_yz.size
        zero_cells_yz = np.sum(density_yz == 0)
        below_threshold_yz = np.sum(density_yz < threshold)
        at_or_above_threshold_yz = np.sum(density_yz >= threshold)

        f.write(f"Total YZ cells: {total_cells_yz:,} ({grid_dims['ny']} x {grid_dims['nz']})\n")
        f.write(f"Cells with zero rays:      {zero_cells_yz:10,} ({100*zero_cells_yz/total_cells_yz:6.2f}%)\n")
        f.write(f"Cells below threshold:     {below_threshold_yz:10,} ({100*below_threshold_yz/total_cells_yz:6.2f}%)\n")
        f.write(f"Cells at/above threshold:  {at_or_above_threshold_yz:10,} ({100*at_or_above_threshold_yz/total_cells_yz:6.2f}%)\n\n")

        if zero_cells_yz < total_cells_yz:
            nonzero_yz = density_yz[density_yz > 0]
            f.write("Ray count statistics (non-zero cells only):\n")
            f.write(f"  Min:        {nonzero_yz.min():10,} rays\n")
            f.write(f"  Max:        {nonzero_yz.max():10,} rays\n")
            f.write(f"  Mean:       {nonzero_yz.mean():10.2f} rays\n")
            f.write(f"  Median:     {np.median(nonzero_yz):10.2f} rays\n")
            f.write(f"  Std Dev:    {nonzero_yz.std():10.2f} rays\n\n")

        # Depth-wise coverage analysis
        f.write("="*80 + "\n")
        f.write("DEPTH-WISE COVERAGE ANALYSIS\n")
        f.write("="*80 + "\n")
        f.write("Coverage percentage at each depth level:\n")
        f.write(f"{'Depth (km)':>12}  {'Cells with rays':>15}  {'Total cells':>12}  {'Coverage %':>12}\n")
        f.write("-"*80 + "\n")

        for k in range(grid_dims['nz']):
            depth = grid_dims['z_min'] + k * grid_dims['dz']
            depth_slice = density_3d[:, :, k]
            total_at_depth = depth_slice.size
            covered_at_depth = np.sum(depth_slice > 0)
            coverage_pct = 100 * covered_at_depth / total_at_depth if total_at_depth > 0 else 0

            f.write(f"{depth:12.3f}  {covered_at_depth:15,}  {total_at_depth:12,}  {coverage_pct:11.2f}%\n")

        f.write("\n")

        # Threshold-based depth layer count analysis
        f.write("="*80 + "\n")
        f.write("THRESHOLD-BASED DEPTH LAYER COUNT (XY PLANE)\n")
        f.write("="*80 + "\n")
        threshold_mask = density_3d >= threshold
        depth_layer_count = np.sum(threshold_mask, axis=2)  # Count of z-layers >= threshold per (x,y)

        total_xy_cells = depth_layer_count.size
        cells_with_no_layers = np.sum(depth_layer_count == 0)
        cells_with_some_layers = total_xy_cells - cells_with_no_layers

        f.write(f"Analyzing each (x,y) position for number of depth layers with >= {threshold} rays:\n")
        f.write(f"Total XY cells: {total_xy_cells:,}\n")
        f.write(f"Cells with no depth layers >= threshold: {cells_with_no_layers:10,} ({100*cells_with_no_layers/total_xy_cells:6.2f}%)\n")
        f.write(f"Cells with some depth layers >= threshold: {cells_with_some_layers:10,} ({100*cells_with_some_layers/total_xy_cells:6.2f}%)\n\n")

        if cells_with_some_layers > 0:
            nonzero_counts = depth_layer_count[depth_layer_count > 0]
            f.write("Depth layer count statistics (for XY cells with at least one layer >= threshold):\n")
            f.write(f"  Min layers:     {nonzero_counts.min():10,}\n")
            f.write(f"  Max layers:     {nonzero_counts.max():10,} (out of {grid_dims['nz']} total depth layers)\n")
            f.write(f"  Mean layers:    {nonzero_counts.mean():10.2f}\n")
            f.write(f"  Median layers:  {np.median(nonzero_counts):10.2f}\n\n")

        # Summary and recommendations
        f.write("="*80 + "\n")
        f.write("SUMMARY AND RECOMMENDATIONS\n")
        f.write("="*80 + "\n")

        coverage_3d = 100 * (total_cells - zero_cells) / total_cells
        coverage_xy = 100 * (total_cells_xy - zero_cells_xy) / total_cells_xy
        coverage_xz = 100 * (total_cells_xz - zero_cells_xz) / total_cells_xz
        coverage_yz = 100 * (total_cells_yz - zero_cells_yz) / total_cells_yz

        f.write(f"Overall 3D coverage: {coverage_3d:.2f}%\n")
        f.write(f"XY plane coverage:   {coverage_xy:.2f}%\n")
        f.write(f"XZ plane coverage:   {coverage_xz:.2f}%\n")
        f.write(f"YZ plane coverage:   {coverage_yz:.2f}%\n\n")

        if coverage_3d < 10:
            f.write("WARNING: Very low 3D coverage (<10%). Consider:\n")
            f.write("  - Using a coarser grid spacing\n")
            f.write("  - Adding more source-receiver pairs\n")
            f.write("  - Reducing the spatial extent of the model\n\n")
        elif coverage_3d < 30:
            f.write("NOTE: Low 3D coverage (<30%). Model regions with no rays will not be constrained.\n\n")

        if at_or_above_threshold < 0.05 * total_cells:
            f.write(f"NOTE: Only {100*at_or_above_threshold/total_cells:.2f}% of cells meet or exceed threshold.\n")
            f.write(f"      Consider lowering threshold from {threshold} for better visualization.\n\n")

        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    logger.info(f"Ray path density report saved to: {output_file}")


def create_ray_density_visualization(config: DictConfig,
                                     events_df: pd.DataFrame,
                                     stations_df: pd.DataFrame,
                                     picks_df: pd.DataFrame) -> None:
    """Create ray density visualizations.

    Computes ray density on a 3D grid and creates plots for XY, XZ, and YZ planes.

    Args:
        config: Full configuration
        events_df: DataFrame with event data
        stations_df: DataFrame with station data
        picks_df: DataFrame with pick data
    """
    logger = logging.getLogger(__name__)

    # Check if ray density visualization is enabled
    if not hasattr(config.visualization, 'ray_density'):
        logger.info("Ray density visualization not configured, skipping")
        return

    ray_density_config = config.visualization.ray_density

    if not ray_density_config.get('enabled', False):
        logger.info("Ray density visualization disabled")
        return

    logger.info("Creating ray density visualizations...")

    # Create output directory
    output_dir = Path(ray_density_config.output_dir)
    ensure_directory_exists(output_dir)

    # Get grid spacing
    grid_spacing = tuple(config.coordinate_transform.grid_spacing_km)

    # Compute grid dimensions
    grid_dims = compute_grid_dimensions(events_df, stations_df, grid_spacing, config)

    # Compute 3D ray density
    density_3d = compute_ray_density_3d(
        events_df, stations_df, picks_df, grid_dims,
        use_parallel=ray_density_config.get('use_parallel', True),
        show_progress=ray_density_config.get('show_progress', True)
    )

    # Get threshold
    threshold = ray_density_config.get('threshold', 0)
    logger.info(f"Using density threshold: {threshold}")

    # Generate text report
    if ray_density_config.get('generate_report', True):
        report_file = output_dir / 'ray_path_density_report.txt'
        generate_ray_density_report(density_3d, grid_dims, threshold, report_file, config)

    # Get visualization config (inherit from station_event_vis if not specified)
    vis_config = ray_density_config
    if not vis_config.get('figure_size'):
        vis_config.figure_size = config.visualization.station_event_vis.figure_size
    if not vis_config.get('figure_dpi'):
        vis_config.figure_dpi = config.visualization.station_event_vis.figure_dpi
    if not vis_config.get('fontsize'):
        vis_config.fontsize = config.visualization.station_event_vis.fontsize

    # Create plots for each plane (summed projections)
    # Only create these plots if plot_overall_ray_density is enabled
    plot_density = ray_density_config.get('plot_overall_ray_density', True)

    if plot_density:
        if ray_density_config.get('plot_xy', True):
            plot_ray_density_xy(density_3d, grid_dims,
                               output_dir / 'ray_density_xy.png',
                               threshold, vis_config)

        if ray_density_config.get('plot_xz', True):
            plot_ray_density_xz(density_3d, grid_dims,
                               output_dir / 'ray_density_xz.png',
                               threshold, vis_config)

        if ray_density_config.get('plot_yz', True):
            plot_ray_density_yz(density_3d, grid_dims,
                               output_dir / 'ray_density_yz.png',
                               threshold, vis_config)

    # Create new visualization: Map view with threshold-based depth layer counting
    if ray_density_config.get('plot_xy_threshold_count', False):
        plot_ray_density_xy_threshold_count(density_3d, grid_dims,
                                            output_dir / 'ray_density_xy_threshold_count.png',
                                            threshold, vis_config)

    # Create new visualization: Depth slices at specified depths
    if ray_density_config.get('plot_depth_slices', False):
        depth_slices = ray_density_config.get('depth_slices', [])
        if depth_slices:
            plot_ray_density_depth_slices(density_3d, grid_dims, output_dir,
                                         depth_slices, threshold, vis_config)
        else:
            logger.warning("plot_depth_slices is enabled but no depth_slices specified")

    logger.info(f"Ray density visualizations saved to: {output_dir}")
