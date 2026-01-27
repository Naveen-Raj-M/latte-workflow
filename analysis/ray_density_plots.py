"""
Ray density visualization plotting functions.

Creates various plots of ray density using actual traced ray paths from cache.
"""

import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional

from .basin_boundaries import plot_basin_boundaries

logger = logging.getLogger(__name__)


def plot_ray_density_xy(
    density_3d: np.ndarray,
    grid_dims: Dict,
    output_file: str,
    threshold: float,
    plot_config: Dict,
    basin_data: Optional[Dict] = None,
    basin_config: Optional[Dict] = None
) -> None:
    """
    Plot ray density in XY plane (summed over depth).

    Args:
        density_3d: 3D ray density array
        grid_dims: Grid dimension dictionary
        output_file: Output file path
        threshold: Ray count threshold (values below are masked)
        plot_config: Plot configuration dictionary
        basin_data: Optional basin boundary data from load_basin_boundaries()
        basin_config: Optional basin boundary styling configuration
    """
    logger.info(f"Creating XY plane density plot...")

    # Sum along Z axis (depth)
    density_xy = np.sum(density_3d, axis=2)  # Shape: (nx, ny)

    # Create coordinate arrays for plotting
    x_coords = np.arange(grid_dims['x_min'], grid_dims['x_max'] + grid_dims['dx']/2, grid_dims['dx'])
    y_coords = np.arange(grid_dims['y_min'], grid_dims['y_max'] + grid_dims['dy']/2, grid_dims['dy'])

    # Create figure
    fig_width = plot_config.get('figure_width', 12)
    fig_height = plot_config.get('figure_height', 10)
    dpi = plot_config.get('dpi', 300)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    # Mask values below or equal to threshold (no coloring for zero)
    density_masked = np.ma.masked_where(density_xy.T <= threshold, density_xy.T)

    # Plot
    cmap = plt.get_cmap(plot_config.get('colormap', 'hot'))
    cmap.set_bad(color='white')

    im = ax.pcolormesh(x_coords, y_coords, density_masked, cmap=cmap, shading='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    label_fontsize = plot_config.get('label_fontsize', 14)
    tick_fontsize = plot_config.get('tick_fontsize', 12)
    title_fontsize = plot_config.get('title_fontsize', 16)

    cbar.set_label('Ray Count (summed over depth)', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    # Labels and title
    ax.set_xlabel('X (km)', fontsize=label_fontsize)
    ax.set_ylabel('Y (km)', fontsize=label_fontsize)
    ax.set_title('Ray Density - XY Plane (Map View)', fontsize=title_fontsize, fontweight='bold')
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot basin boundaries if provided
    if basin_data and basin_config:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        plot_basin_boundaries(ax, basin_data, basin_config, xlim=xlim, ylim=ylim)

    plt.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"  Saved: {output_file}")


def plot_ray_density_xz(
    density_3d: np.ndarray,
    grid_dims: Dict,
    output_file: str,
    threshold: float,
    plot_config: Dict
) -> None:
    """
    Plot ray density in XZ plane (summed over Y).

    Args:
        density_3d: 3D ray density array
        grid_dims: Grid dimension dictionary
        output_file: Output file path
        threshold: Ray count threshold
        plot_config: Plot configuration dictionary
    """
    logger.info(f"Creating XZ plane density plot...")

    # Sum along Y axis
    density_xz = np.sum(density_3d, axis=1)  # Shape: (nx, nz)

    # Create coordinate arrays
    x_coords = np.arange(grid_dims['x_min'], grid_dims['x_max'] + grid_dims['dx']/2, grid_dims['dx'])
    z_coords = np.arange(grid_dims['z_min'], grid_dims['z_max'] + grid_dims['dz']/2, grid_dims['dz'])

    # Create figure
    fig_width = plot_config.get('figure_width', 12)
    fig_height = plot_config.get('figure_height', 10)
    dpi = plot_config.get('dpi', 300)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    # Mask values below or equal to threshold (no coloring for zero)
    density_masked = np.ma.masked_where(density_xz.T <= threshold, density_xz.T)

    # Plot
    cmap = plt.get_cmap(plot_config.get('colormap', 'hot'))
    cmap.set_bad(color='white')

    im = ax.pcolormesh(x_coords, z_coords, density_masked, cmap=cmap, shading='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    label_fontsize = plot_config.get('label_fontsize', 14)
    tick_fontsize = plot_config.get('tick_fontsize', 12)
    title_fontsize = plot_config.get('title_fontsize', 16)

    cbar.set_label('Ray Count (summed over Y)', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    # Labels and title
    ax.set_xlabel('X (km)', fontsize=label_fontsize)
    ax.set_ylabel('Depth (km)', fontsize=label_fontsize)
    ax.set_title('Ray Density - XZ Plane (Vertical Cross-Section)', fontsize=title_fontsize, fontweight='bold')
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.invert_yaxis()  # Depth increases downward
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"  Saved: {output_file}")


def plot_ray_density_yz(
    density_3d: np.ndarray,
    grid_dims: Dict,
    output_file: str,
    threshold: float,
    plot_config: Dict
) -> None:
    """
    Plot ray density in YZ plane (summed over X).

    Args:
        density_3d: 3D ray density array
        grid_dims: Grid dimension dictionary
        output_file: Output file path
        threshold: Ray count threshold
        plot_config: Plot configuration dictionary
    """
    logger.info(f"Creating YZ plane density plot...")

    # Sum along X axis
    density_yz = np.sum(density_3d, axis=0)  # Shape: (ny, nz)

    # Create coordinate arrays
    y_coords = np.arange(grid_dims['y_min'], grid_dims['y_max'] + grid_dims['dy']/2, grid_dims['dy'])
    z_coords = np.arange(grid_dims['z_min'], grid_dims['z_max'] + grid_dims['dz']/2, grid_dims['dz'])

    # Create figure
    fig_width = plot_config.get('figure_width', 12)
    fig_height = plot_config.get('figure_height', 10)
    dpi = plot_config.get('dpi', 300)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    # Mask values below or equal to threshold (no coloring for zero)
    density_masked = np.ma.masked_where(density_yz.T <= threshold, density_yz.T)

    # Plot
    cmap = plt.get_cmap(plot_config.get('colormap', 'hot'))
    cmap.set_bad(color='white')

    im = ax.pcolormesh(y_coords, z_coords, density_masked, cmap=cmap, shading='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    label_fontsize = plot_config.get('label_fontsize', 14)
    tick_fontsize = plot_config.get('tick_fontsize', 12)
    title_fontsize = plot_config.get('title_fontsize', 16)

    cbar.set_label('Ray Count (summed over X)', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    # Labels and title
    ax.set_xlabel('Y (km)', fontsize=label_fontsize)
    ax.set_ylabel('Depth (km)', fontsize=label_fontsize)
    ax.set_title('Ray Density - YZ Plane (Vertical Cross-Section)', fontsize=title_fontsize, fontweight='bold')
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.invert_yaxis()  # Depth increases downward
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"  Saved: {output_file}")


def plot_ray_density_xy_threshold_count(
    density_3d: np.ndarray,
    grid_dims: Dict,
    output_file: str,
    threshold: float,
    plot_config: Dict,
    basin_data: Optional[Dict] = None,
    basin_config: Optional[Dict] = None
) -> None:
    """
    Plot map view showing count of depth layers with rays >= threshold.

    Each grid cell (x, y) is colored by the number of depth layers below it
    that contain at least 'threshold' rays.

    Args:
        density_3d: 3D ray density array
        grid_dims: Grid dimension dictionary
        output_file: Output file path
        threshold: Ray count threshold
        plot_config: Plot configuration dictionary
        basin_data: Optional basin boundary data from load_basin_boundaries()
        basin_config: Optional basin boundary styling configuration
    """
    logger.info(f"Creating XY threshold count plot...")

    # Count depth layers >= threshold for each (x, y)
    threshold_mask = density_3d >= threshold
    depth_layer_count = np.sum(threshold_mask, axis=2)  # Shape: (nx, ny)

    # Create coordinate arrays
    x_coords = np.arange(grid_dims['x_min'], grid_dims['x_max'] + grid_dims['dx']/2, grid_dims['dx'])
    y_coords = np.arange(grid_dims['y_min'], grid_dims['y_max'] + grid_dims['dy']/2, grid_dims['dy'])

    # Create figure
    fig_width = plot_config.get('figure_width', 12)
    fig_height = plot_config.get('figure_height', 10)
    dpi = plot_config.get('dpi', 300)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    # Mask cells with zero depth layers above threshold
    depth_count_masked = np.ma.masked_where(depth_layer_count.T == 0, depth_layer_count.T)

    # Plot
    cmap = plt.get_cmap(plot_config.get('colormap', 'hot'))
    cmap.set_bad(color='white')

    im = ax.pcolormesh(x_coords, y_coords, depth_count_masked, cmap=cmap, shading='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    label_fontsize = plot_config.get('label_fontsize', 14)
    tick_fontsize = plot_config.get('tick_fontsize', 12)
    title_fontsize = plot_config.get('title_fontsize', 16)

    cbar.set_label(f'Number of depth layers with >= {threshold} rays', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    # Labels and title
    ax.set_xlabel('X (km)', fontsize=label_fontsize)
    ax.set_ylabel('Y (km)', fontsize=label_fontsize)
    ax.set_title(f'Ray Coverage >= {threshold} Rays (Map View)', fontsize=title_fontsize, fontweight='bold')
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot basin boundaries if provided
    if basin_data and basin_config:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        plot_basin_boundaries(ax, basin_data, basin_config, xlim=xlim, ylim=ylim)

    plt.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"  Saved: {output_file}")


def plot_ray_density_depth_slices(
    density_3d: np.ndarray,
    grid_dims: Dict,
    output_dir: Path,
    depth_values: list,
    threshold: float,
    plot_config: Dict,
    basin_data: Optional[Dict] = None,
    basin_config: Optional[Dict] = None
) -> None:
    """
    Plot horizontal slices at specified depths showing ray density.

    Args:
        density_3d: 3D ray density array
        grid_dims: Grid dimension dictionary
        output_dir: Output directory for slice plots
        depth_values: List of depths (in km) to create slices for
        threshold: Ray count threshold
        plot_config: Plot configuration dictionary
        basin_data: Optional basin boundary data from load_basin_boundaries()
        basin_config: Optional basin boundary styling configuration
    """
    logger.info(f"Creating depth slice plots for {len(depth_values)} depths...")

    # Create coordinate arrays
    x_coords = np.arange(grid_dims['x_min'], grid_dims['x_max'] + grid_dims['dx']/2, grid_dims['dx'])
    y_coords = np.arange(grid_dims['y_min'], grid_dims['y_max'] + grid_dims['dy']/2, grid_dims['dy'])

    for depth_km in depth_values:
        # Find the closest depth index
        depth_index = int(np.round((depth_km - grid_dims['z_min']) / grid_dims['dz']))

        # Check bounds
        if depth_index < 0 or depth_index >= grid_dims['nz']:
            logger.warning(f"  Depth {depth_km} km is outside grid bounds, skipping")
            continue

        # Extract depth slice
        actual_depth = grid_dims['z_min'] + depth_index * grid_dims['dz']
        depth_slice = density_3d[:, :, depth_index]  # Shape: (nx, ny)

        # Create figure
        fig_width = plot_config.get('figure_width', 12)
        fig_height = plot_config.get('figure_height', 10)
        dpi = plot_config.get('dpi', 300)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

        # Mask values below or equal to threshold (no coloring for zero)
        slice_masked = np.ma.masked_where(depth_slice.T <= threshold, depth_slice.T)

        # Plot
        cmap = plt.get_cmap(plot_config.get('colormap', 'hot'))
        cmap.set_bad(color='white')

        im = ax.pcolormesh(x_coords, y_coords, slice_masked, cmap=cmap, shading='auto')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        label_fontsize = plot_config.get('label_fontsize', 14)
        tick_fontsize = plot_config.get('tick_fontsize', 12)
        title_fontsize = plot_config.get('title_fontsize', 16)

        cbar.set_label('Ray Count', fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)

        # Labels and title
        ax.set_xlabel('X (km)', fontsize=label_fontsize)
        ax.set_ylabel('Y (km)', fontsize=label_fontsize)
        ax.set_title(f'Ray Density at Depth = {actual_depth:.2f} km', fontsize=title_fontsize, fontweight='bold')
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Plot basin boundaries if provided
        if basin_data and basin_config:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            plot_basin_boundaries(ax, basin_data, basin_config, xlim=xlim, ylim=ylim)

        plt.tight_layout()

        # Save with depth in filename
        output_file = output_dir / f'ray_density_depth_{actual_depth:.1f}km.png'
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"  Saved depth slice: {output_file}")
