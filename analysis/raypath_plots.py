"""
Ray path visualization plotting functions for LATTE analysis.

This module provides plotting functions for ray path visualization,
migrated from postprocess_raypath.py with an interface adapted for
the analysis module.
"""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tqdm import tqdm

from utils.io_utils import ensure_directory_exists
from .basin_boundaries import plot_basin_boundaries

logger = logging.getLogger(__name__)


def plot_ray_paths_projection(
    ray_paths,  # Union[List[np.ndarray], np.ndarray]
    sources: List[Tuple[float, float, float]],
    receivers: List[Tuple[float, float, float]],
    plane: str,
    output_file: str,
    config: Dict[str, Any],
    basin_data: Optional[Dict] = None,
    basin_config: Optional[Dict] = None
) -> None:
    """
    Plot ray paths in a specific 2D projection plane.

    Args:
        ray_paths: Either:
            - List of ray path arrays (each shape N x 3) [standard format]
            - Single numpy array of shape (n_rays, num_samples, 3) [vectorized format]
        sources: List of source coordinates (x, y, z)
        receivers: List of receiver coordinates (x, y, z)
        plane: Projection plane ('xy', 'xz', or 'yz')
        output_file: Output file path
        config: Plot configuration dictionary with keys:
            - figure_width: Figure width in inches
            - figure_height: Figure height in inches
            - dpi: Output DPI
            - ray_color: Ray path color
            - ray_alpha: Ray path transparency
            - ray_linewidth: Ray path line width
            - source_color: Source marker color
            - source_marker: Source marker style
            - source_size: Source marker size
            - receiver_color: Receiver marker color
            - receiver_marker: Receiver marker style
            - receiver_size: Receiver marker size
            - grid: Show grid
            - equal_aspect: Use equal aspect ratio
            - label_fontsize: Axis label font size
            - title_fontsize: Title font size
            - legend_fontsize: Legend font size
            - format: Output format (png, pdf, svg)
        basin_data: Optional basin boundary data from load_basin_boundaries()
                    (only used for 'xy' plane)
        basin_config: Optional basin boundary styling configuration
    """
    import time
    t_start = time.time()

    # Get configuration values with defaults
    figure_width = config.get('figure_width', 12)
    figure_height = config.get('figure_height', 10)
    dpi = config.get('dpi', 300)
    ray_color = config.get('ray_color', 'blue')
    ray_alpha = config.get('ray_alpha', 0.3)
    ray_linewidth = config.get('ray_linewidth', 0.5)
    source_color = config.get('source_color', 'red')
    source_marker = config.get('source_marker', '*')
    source_size = config.get('source_size', 100)
    receiver_color = config.get('receiver_color', 'green')
    receiver_marker = config.get('receiver_marker', '^')
    receiver_size = config.get('receiver_size', 50)
    grid = config.get('grid', True)
    equal_aspect = config.get('equal_aspect', True)
    label_fontsize = config.get('label_fontsize', 12)
    title_fontsize = config.get('title_fontsize', 14)
    legend_fontsize = config.get('legend_fontsize', 10)

    # Detect format: vectorized (numpy array) vs standard (list)
    is_vectorized = isinstance(ray_paths, np.ndarray)

    if is_vectorized:
        logger.debug(f"Using vectorized ray paths: shape {ray_paths.shape}")

    # Subsampling
    subsample_fraction = config.get('subsample_fraction', 1.0)
    max_rays = config.get('max_rays', None)

    # Apply subsampling
    if max_rays is not None and len(ray_paths) > max_rays:
        indices = np.random.choice(len(ray_paths), max_rays, replace=False)
        if is_vectorized:
            ray_paths = ray_paths[sorted(indices)]
        else:
            ray_paths = [ray_paths[i] for i in sorted(indices)]
        logger.info(f"Subsampled to {max_rays} ray paths for plotting")
    elif subsample_fraction < 1.0 and len(ray_paths) > 0:
        n_rays_original = len(ray_paths)
        n_rays_to_plot = max(1, int(n_rays_original * subsample_fraction))
        indices = np.random.choice(n_rays_original, n_rays_to_plot, replace=False)
        if is_vectorized:
            ray_paths = ray_paths[sorted(indices)]
        else:
            ray_paths = [ray_paths[i] for i in sorted(indices)]
        logger.info(f"Subsampled {n_rays_to_plot}/{n_rays_original} ray paths ({subsample_fraction*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

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

    # Plot ray paths using LineCollection for performance
    if len(ray_paths) > 0:
        if is_vectorized:
            # Vectorized format: extract 2D projections using numpy slicing (MUCH faster!)
            # Shape: (n_rays, num_samples, 3) -> (n_rays, num_samples, 2)
            ray_2d = ray_paths[:, :, [idx1, idx2]]
            # Convert to list of segments for LineCollection
            segments = [ray_2d[i] for i in range(len(ray_2d))]
        else:
            # Standard format: iterate through list
            segments = []
            for ray in ray_paths:
                if len(ray) > 1:
                    ray_2d = ray[:, [idx1, idx2]]
                    segments.append(ray_2d)

        if segments:
            logger.debug(f"Creating LineCollection with {len(segments)} ray paths...")
            lc = LineCollection(
                segments,
                colors=ray_color,
                alpha=ray_alpha,
                linewidths=ray_linewidth,
                label='Ray paths',
                rasterized=True  # Rasterize for performance
            )
            ax.add_collection(lc)
            ax.autoscale()

    # Plot sources
    if sources:
        sources_array = np.array(sources)
        ax.scatter(sources_array[:, idx1], sources_array[:, idx2],
                   c=source_color,
                   marker=source_marker,
                   s=source_size,
                   label='Sources',
                   edgecolors='black',
                   linewidth=0.5,
                   zorder=10)

    # Plot receivers
    if receivers:
        receivers_array = np.array(receivers)
        ax.scatter(receivers_array[:, idx1], receivers_array[:, idx2],
                   c=receiver_color,
                   marker=receiver_marker,
                   s=receiver_size,
                   label='Receivers',
                   edgecolors='black',
                   linewidth=0.5,
                   zorder=10)

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_title(f'Ray Paths - {plane.upper()} Plane', fontsize=title_fontsize)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(grid, alpha=0.3)
    ax.set_aspect('equal' if equal_aspect else 'auto')

    # Plot basin boundaries for XY plane only
    if plane == 'xy' and basin_data and basin_config:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        plot_basin_boundaries(ax, basin_data, basin_config, xlim=xlim, ylim=ylim)

    # Invert z-axis to have depth increase downwards
    if plane in ['xz', 'yz']:
        ax.invert_yaxis()

    # Ensure output directory exists
    ensure_directory_exists(os.path.dirname(output_file))

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - t_start
    logger.info(f"Saved {plane.upper()} plot: {output_file} ({elapsed:.1f}s)")


def plot_ray_paths_3d(
    ray_paths,  # Union[List[np.ndarray], np.ndarray]
    sources: List[Tuple[float, float, float]],
    receivers: List[Tuple[float, float, float]],
    output_file: str,
    config: Dict[str, Any]
) -> None:
    """
    Plot ray paths in 3D.

    Args:
        ray_paths: Either:
            - List of ray path arrays (each shape N x 3) [standard format]
            - Single numpy array of shape (n_rays, num_samples, 3) [vectorized format]
        sources: List of source coordinates (x, y, z)
        receivers: List of receiver coordinates (x, y, z)
        output_file: Output file path
        config: Plot configuration dictionary
    """
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # Detect format: vectorized (numpy array) vs standard (list)
    is_vectorized = isinstance(ray_paths, np.ndarray)

    # Get configuration
    figure_width = config.get('figure_width', 12)
    figure_height = config.get('figure_height', 10)
    dpi = config.get('dpi', 300)
    ray_color = config.get('ray_color', 'blue')
    ray_alpha = config.get('ray_alpha', 0.3)
    ray_linewidth = config.get('ray_linewidth', 0.5)
    source_color = config.get('source_color', 'red')
    source_marker = config.get('source_marker', '*')
    source_size = config.get('source_size', 100)
    receiver_color = config.get('receiver_color', 'green')
    receiver_marker = config.get('receiver_marker', '^')
    receiver_size = config.get('receiver_size', 50)
    max_rays = config.get('max_rays', 1000)  # Limit for 3D performance

    # Subsample for 3D performance
    if len(ray_paths) > max_rays:
        indices = np.random.choice(len(ray_paths), max_rays, replace=False)
        if is_vectorized:
            ray_paths = ray_paths[sorted(indices)]
        else:
            ray_paths = [ray_paths[i] for i in sorted(indices)]
        logger.info(f"Subsampled to {max_rays} ray paths for 3D plotting")

    fig = plt.figure(figsize=(figure_width, figure_height))
    ax = fig.add_subplot(111, projection='3d')

    # Plot ray paths
    if is_vectorized:
        # Vectorized format: iterate through array (still need loop for 3D plotting)
        for i in range(len(ray_paths)):
            ax.plot(ray_paths[i, :, 0], ray_paths[i, :, 1], ray_paths[i, :, 2],
                    color=ray_color, alpha=ray_alpha, linewidth=ray_linewidth)
    else:
        # Standard format: iterate through list
        for ray in ray_paths:
            if len(ray) > 1:
                ax.plot(ray[:, 0], ray[:, 1], ray[:, 2],
                        color=ray_color, alpha=ray_alpha, linewidth=ray_linewidth)

    # Plot sources
    if sources:
        sources_array = np.array(sources)
        ax.scatter(sources_array[:, 0], sources_array[:, 1], sources_array[:, 2],
                   c=source_color, marker=source_marker, s=source_size,
                   label='Sources', edgecolors='black', linewidth=0.5)

    # Plot receivers
    if receivers:
        receivers_array = np.array(receivers)
        ax.scatter(receivers_array[:, 0], receivers_array[:, 1], receivers_array[:, 2],
                   c=receiver_color, marker=receiver_marker, s=receiver_size,
                   label='Receivers', edgecolors='black', linewidth=0.5)

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Ray Paths - 3D View')
    ax.legend()

    # Invert z-axis
    ax.invert_zaxis()

    ensure_directory_exists(os.path.dirname(output_file))
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved 3D plot: {output_file}")


def create_ray_path_summary_plot(
    stats: Dict[str, Any],
    output_file: str,
    config: Dict[str, Any]
) -> None:
    """
    Create a summary plot showing ray path statistics.

    Args:
        stats: Ray path statistics dictionary
        output_file: Output file path
        config: Plot configuration dictionary
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart of ray counts by phase
    phases = []
    ray_counts = []
    shot_counts = []

    for phase in ['p', 's']:
        ray_key = f'{phase}_wave_rays'
        shot_key = f'{phase}_wave_shots'
        if ray_key in stats:
            phases.append(f'{phase.upper()}-wave')
            ray_counts.append(stats[ray_key])
            shot_counts.append(stats.get(shot_key, 0))

    if phases:
        x = np.arange(len(phases))
        width = 0.35

        ax1 = axes[0]
        ax1.bar(x - width/2, ray_counts, width, label='Rays', color='steelblue')
        ax1.bar(x + width/2, shot_counts, width, label='Shots', color='coral')
        ax1.set_ylabel('Count')
        ax1.set_title('Ray Path Counts by Phase')
        ax1.set_xticks(x)
        ax1.set_xticklabels(phases)
        ax1.legend()

        # Add value labels
        for i, (rc, sc) in enumerate(zip(ray_counts, shot_counts)):
            ax1.text(i - width/2, rc, f'{rc:,}', ha='center', va='bottom', fontsize=9)
            ax1.text(i + width/2, sc, f'{sc:,}', ha='center', va='bottom', fontsize=9)

    # Text summary
    ax2 = axes[1]
    ax2.axis('off')

    summary_text = "Ray Path Cache Summary\n" + "="*30 + "\n\n"
    summary_text += f"File size: {stats.get('file_size_mb', 0):.2f} MB\n"
    summary_text += f"Created: {stats.get('created', 'N/A')}\n"
    summary_text += f"Last updated: {stats.get('last_updated', 'N/A')}\n\n"

    for phase in ['p', 's']:
        summary_text += f"{phase.upper()}-wave:\n"
        summary_text += f"  Shots: {stats.get(f'{phase}_wave_shots', 0):,}\n"
        summary_text += f"  Rays: {stats.get(f'{phase}_wave_rays', 0):,}\n\n"

    ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()

    ensure_directory_exists(os.path.dirname(output_file))
    plt.savefig(output_file, dpi=config.get('dpi', 300), bbox_inches='tight')
    plt.close()

    logger.info(f"Saved summary plot: {output_file}")
