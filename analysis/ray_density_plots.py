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


# ============================================================================
# OBLIQUE SECTION HELPERS
# ============================================================================

def _compute_cell_edges(centers: np.ndarray) -> np.ndarray:
    """Return cell edge array (len+1) from cell center array."""
    centers = np.sort(centers)
    mids  = 0.5 * (centers[:-1] + centers[1:])
    left  = centers[0]  - 0.5 * (centers[1]  - centers[0])
    right = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate([[left], mids, [right]])


def _extract_oblique_section(
    density_3d: np.ndarray,
    grid_dims: Dict,
    x1: float, y1: float,
    x2: float, y2: float,
):
    """Extract a 2D ray density section along a vertical oblique plane.

    The plane is defined by two (x, y) endpoints and spans all depths.
    Uses the same Minkowski projection test as model_visualization_section.py:
        perp_tolerance = 0.5·dx·|n_x| + 0.5·dy·|n_y|

    Returns:
        section_2d   : np.ndarray (nz, n_cols) — density at each (along, depth) cell
        along_coords : np.ndarray (n_cols,)    — along-section km from start
        z_coords     : np.ndarray (nz,)        — depth values (km)
        section_length : float                 — total length (km)
    """
    dx   = grid_dims['dx']
    dy   = grid_dims['dy']
    dz   = grid_dims['dz']
    x_min = grid_dims['x_min']
    y_min = grid_dims['y_min']
    z_min = grid_dims['z_min']
    nx   = grid_dims['nx']
    ny   = grid_dims['ny']
    nz   = grid_dims['nz']

    vec = np.array([x2 - x1, y2 - y1], dtype=float)
    section_length = float(np.linalg.norm(vec))
    if section_length < 1e-9:
        raise ValueError(
            f"Section start and end are identical: ({x1}, {y1}) == ({x2}, {y2})"
        )

    d_norm = vec / section_length
    n_norm = np.array([-d_norm[1], d_norm[0]])

    tol_perp  = 0.5 * dx * abs(n_norm[0]) + 0.5 * dy * abs(n_norm[1])
    tol_along = 0.5 * dx * abs(d_norm[0]) + 0.5 * dy * abs(d_norm[1])

    # Grid coordinates — broadcast over (nx, ny)
    xg = x_min + np.arange(nx) * dx   # (nx,)
    yg = y_min + np.arange(ny) * dy   # (ny,)
    xx = xg[:, None]                   # (nx, 1)
    yy = yg[None, :]                   # (1, ny)

    rel_x = xx - x1                    # (nx, 1)
    rel_y = yy - y1                    # (1, ny)
    along = rel_x * d_norm[0] + rel_y * d_norm[1]   # (nx, ny)
    perp  = rel_x * n_norm[0] + rel_y * n_norm[1]   # (nx, ny)

    mask = (
        (np.abs(perp) <= tol_perp) &
        (along >= -tol_along) &
        (along <= section_length + tol_along)
    )

    if not mask.any():
        raise ValueError(
            "No grid columns found within the oblique section plane tolerance. "
            "Verify that section coordinates lie inside the model domain."
        )

    i_idx, j_idx = np.where(mask)
    along_vals = along[i_idx, j_idx]
    perp_abs   = np.abs(perp[i_idx, j_idx])

    # Deduplicate: for identical along positions keep the column closest to section
    sort_order = np.lexsort((perp_abs, along_vals))
    i_idx      = i_idx[sort_order]
    j_idx      = j_idx[sort_order]
    along_vals = along_vals[sort_order]
    perp_abs   = perp_abs[sort_order]

    seen_along: dict = {}
    for idx in range(len(along_vals)):
        a = round(float(along_vals[idx]), 4)
        if a not in seen_along:
            seen_along[a] = idx

    unique_indices = sorted(seen_along.values(), key=lambda idx: along_vals[idx])
    n_cols = len(unique_indices)

    along_out = np.array([along_vals[k] for k in unique_indices])
    z_coords  = z_min + np.arange(nz) * dz

    section_2d = np.zeros((nz, n_cols), dtype=float)
    for col, orig_idx in enumerate(unique_indices):
        ii, jj = int(i_idx[orig_idx]), int(j_idx[orig_idx])
        section_2d[:, col] = density_3d[ii, jj, :]

    return section_2d, along_out, z_coords, section_length


def _find_basin_intersections_density(
    basin_data: Dict,
    x1: float, y1: float,
    x2: float, y2: float,
) -> list:
    """Return sorted along-section distances (km from start) where basin boundaries
    cross the section line segment (x1,y1)→(x2,y2).

    Solves the 2-D line intersection parametrically and deduplicates crossings
    within 0.01 km of each other.
    """
    dx = x2 - x1
    dy = y2 - y1
    section_length = np.hypot(dx, dy)
    if section_length < 1e-9:
        return []

    hits = []
    for data in basin_data.values():
        bx = data['x']
        by = data['y']
        n  = len(bx)
        for i in range(n - 1):
            bx0, by0 = float(bx[i]),     float(by[i])
            bx1, by1 = float(bx[i + 1]), float(by[i + 1])
            dbx = bx1 - bx0
            dby = by1 - by0
            det = -dx * dby + dbx * dy
            if abs(det) < 1e-12:
                continue
            t = (-(bx0 - x1) * dby + dbx * (by0 - y1)) / det
            s = (-dx * (by0 - y1) + dy * (bx0 - x1)) / det
            if 0.0 <= t <= 1.0 and 0.0 <= s <= 1.0:
                hits.append(round(t * section_length, 4))

    hits.sort()
    deduped: list = []
    for h in hits:
        if not deduped or abs(h - deduped[-1]) > 0.01:
            deduped.append(h)
    return deduped


# ============================================================================
# OBLIQUE SECTION PLOT
# ============================================================================

def plot_ray_density_oblique_section(
    density_3d: np.ndarray,
    grid_dims: Dict,
    output_file: str,
    section: Dict,
    threshold: float,
    plot_config: Dict,
    basin_data: Optional[Dict] = None,
    basin_config: Optional[Dict] = None,
) -> None:
    """Plot ray density along a single oblique vertical plane section.

    The section is defined by (x1,y1)→(x2,y2); the plane spans all depths.
    Basin boundary crossings are drawn as vertical dashed lines.

    Args:
        density_3d  : 3D ray density array (nx, ny, nz)
        grid_dims   : Grid dimension dictionary
        output_file : Output file path
        section     : Section dict with keys label, x1, y1, x2, y2
        threshold   : Ray count threshold (values at or below are masked)
        plot_config : Plot configuration dictionary
        basin_data  : Optional basin boundary data from load_basin_boundaries()
        basin_config: Optional basin boundary styling configuration
    """
    label = section.get('label', '')
    x1 = float(section['x1'])
    y1 = float(section['y1'])
    x2 = float(section['x2'])
    y2 = float(section['y2'])

    logger.info(f"Creating oblique section '{label}' "
                f"({x1:.1f},{y1:.1f}) → ({x2:.1f},{y2:.1f}) ...")

    section_2d, along_coords, z_coords, section_length = _extract_oblique_section(
        density_3d, grid_dims, x1, y1, x2, y2
    )

    # Mask values at or below threshold
    section_masked = np.ma.masked_where(
        section_2d <= threshold, section_2d.astype(float)
    )

    # Cell edges for pcolormesh (shading='flat')
    if len(along_coords) > 1:
        along_edges = _compute_cell_edges(along_coords)
    else:
        along_edges = np.array([along_coords[0] - 0.5, along_coords[0] + 0.5])

    dz = grid_dims['dz']
    z_edges = np.concatenate([z_coords - 0.5 * dz, [z_coords[-1] + 0.5 * dz]])

    # Create figure (inherits figure_width / figure_height from ray_density_analysis config)
    fig_width  = plot_config.get('figure_width', 12)
    fig_height = plot_config.get('figure_height', 10)
    dpi = plot_config.get('dpi', 300)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    cmap = plt.get_cmap(plot_config.get('colormap', 'hot'))
    cmap.set_bad(color='white')

    im = ax.pcolormesh(along_edges, z_edges, section_masked,
                       cmap=cmap, shading='flat')
    ax.invert_yaxis()   # depth increases downward

    # Basin boundary intersections as vertical lines
    if basin_data:
        basin_hits = _find_basin_intersections_density(basin_data, x1, y1, x2, y2)
        if basin_hits:
            logger.info(f"  Basin intersections at: {[f'{v:.1f}' for v in basin_hits]} km")
        # Reuse styling from top-level basin_boundaries config section
        bb_color  = basin_config.get('line_color', 'black') if basin_config else 'black'
        bb_lw     = basin_config.get('line_width', 1.0)     if basin_config else 1.0
        bb_ls     = basin_config.get('line_style', '-')     if basin_config else '-'
        bb_alpha  = basin_config.get('line_alpha', 0.8)     if basin_config else 0.8
        for along_km in basin_hits:
            ax.axvline(x=along_km, color=bb_color, linewidth=bb_lw,
                       linestyle=bb_ls, alpha=bb_alpha, zorder=5)

    # Section endpoint labels (A / A')
    ax.text(0.0, 1.0, f' {label}',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=11, fontweight='bold', clip_on=False)
    ax.text(1.0, 1.0, f"{label}' ",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=11, fontweight='bold', clip_on=False)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    label_fontsize = plot_config.get('label_fontsize', 14)
    tick_fontsize  = plot_config.get('tick_fontsize', 12)
    title_fontsize = plot_config.get('title_fontsize', 16)

    cbar.set_label('Ray Count', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    ax.set_xlabel('Distance along section (km)', fontsize=label_fontsize)
    ax.set_ylabel('Depth (km)', fontsize=label_fontsize)
    ax.set_title(
        f"Ray Density — Oblique Section {label}-{label}'  "
        f"({x1:.1f}, {y1:.1f}) → ({x2:.1f}, {y2:.1f})",
        fontsize=title_fontsize, fontweight='bold'
    )
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"  Saved oblique section: {output_file}")


# ============================================================================
# OBLIQUE SECTION PLAN VIEW
# ============================================================================

def plot_ray_density_section_plan_view(
    sections: list,
    grid_dims: Dict,
    output_file: str,
    plot_config: Dict,
    basin_data: Optional[Dict] = None,
    basin_config: Optional[Dict] = None,
) -> None:
    """Plot a plan view (XY map) showing all oblique section traces.

    Basin boundaries are shown as polygons and section lines are labelled
    A/A', B/B', etc.

    Args:
        sections    : List of section dicts (label, x1, y1, x2, y2)
        grid_dims   : Grid dimension dictionary (used for axis limits)
        output_file : Output file path
        plot_config : Plot configuration dictionary
        basin_data  : Optional basin boundary data from load_basin_boundaries()
        basin_config: Optional basin boundary styling configuration
    """
    logger.info("Creating oblique section plan view ...")

    fig_width  = plot_config.get('figure_width', 12)
    fig_height = plot_config.get('figure_height', 10)
    dpi = plot_config.get('dpi', 300)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    # Basin boundaries
    if basin_data and basin_config:
        xlim_est = (grid_dims['x_min'], grid_dims['x_max'])
        ylim_est = (grid_dims['y_min'], grid_dims['y_max'])
        plot_basin_boundaries(ax, basin_data, basin_config,
                              xlim=xlim_est, ylim=ylim_est)

    # Section lines
    color_cycle    = plt.rcParams['axes.prop_cycle'].by_key()['color']
    section_lw     = 2.0
    section_lbl_fs = plot_config.get('label_fontsize', 14)
    offset_frac    = 0.02

    for idx, sec in enumerate(sections):
        lbl   = sec.get('label', str(idx))
        x1, y1 = float(sec['x1']), float(sec['y1'])
        x2, y2 = float(sec['x2']), float(sec['y2'])
        color = color_cycle[idx % len(color_cycle)]

        ax.plot([x1, x2], [y1, y2], color=color, linewidth=section_lw, zorder=10)

        seg_len = max(np.hypot(x2 - x1, y2 - y1), 1e-9)
        off     = offset_frac * seg_len
        nx_off  = -(y2 - y1) / seg_len * off
        ny_off  =  (x2 - x1) / seg_len * off

        ax.text(x1 + nx_off, y1 + ny_off, lbl,
                color=color, fontsize=section_lbl_fs, fontweight='bold',
                ha='center', va='center', zorder=11)
        ax.text(x2 + nx_off, y2 + ny_off, f"{lbl}'",
                color=color, fontsize=section_lbl_fs, fontweight='bold',
                ha='center', va='center', zorder=11)

    label_fontsize = plot_config.get('label_fontsize', 14)
    title_fontsize = plot_config.get('title_fontsize', 16)
    tick_fontsize  = plot_config.get('tick_fontsize', 12)

    ax.set_xlabel('X (km)', fontsize=label_fontsize)
    ax.set_ylabel('Y (km)', fontsize=label_fontsize)
    ax.set_title('Oblique Section Locations', fontsize=title_fontsize, fontweight='bold')
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"  Saved section plan view: {output_file}")
