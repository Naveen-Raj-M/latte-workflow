"""
Plotting utility functions for slice visualization.

This module provides functions for plotting XY, XZ, and YZ slices with:
- Custom colormaps (including three-color scheme for threshold visualization)
- Fair recovery contour overlays
- Flexible configuration options
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional

from checkerboard_test.colormap_utils import create_centered_norm

logger = logging.getLogger(__name__)


def plot_xy_slice(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    velocity_slice: np.ndarray,
    z_value: float,
    output_path: Path,
    config: Dict,
    title_prefix: str = "",
    recovery_mask: Optional[np.ndarray] = None,
) -> None:
    """
    Plot XY (horizontal/depth) slice.

    Args:
        x_coords: X coordinates (km)
        y_coords: Y coordinates (km)
        velocity_slice: 2D velocity array (ny, nx)
        z_value: Depth value (km)
        output_path: Output file path
        config: Visualization configuration
        title_prefix: Prefix for title
        recovery_mask: Optional 2D boolean mask for fair recovery contours
    """
    fig, ax = plt.subplots(
        figsize=config["figure"]["figsize"],
        dpi=config["figure"]["dpi"],
    )

    # Create meshgrid for pcolormesh
    X, Y = np.meshgrid(x_coords, y_coords)

    # Get configuration options
    threshold = config.get("amplitude_threshold_viz")
    is_pattern_plot = "pattern" in title_prefix.lower() or "pct" in title_prefix.lower()
    use_uniform = config.get("use_uniform_colorbar", False)
    symmetric = config.get("symmetric_colorbar", False)

    # Get pre-computed vmin/vmax from config (global if uniform, None otherwise)
    config_vmin = config.get("vmin")
    config_vmax = config.get("vmax")

    # Compute local slice limits
    local_vmin = np.nanmin(velocity_slice)
    local_vmax = np.nanmax(velocity_slice)

    # Determine the data range to use
    if use_uniform and config_vmin is not None and config_vmax is not None:
        # Use global limits from config
        data_vmin, data_vmax = config_vmin, config_vmax
    else:
        # Use local slice limits
        data_vmin, data_vmax = local_vmin, local_vmax

    # Handle threshold and colorbar limits for pattern plots
    norm = None
    extend = 'neither'
    plot_data = velocity_slice

    if is_pattern_plot and threshold is not None:
        # Get base colormap and set bad color for masked values
        cmap = plt.get_cmap(config.get("colormap", "seismic")).copy()
        cmap.set_bad('black')

        # Determine colorbar limits
        if use_uniform:
            # Uniform colorbar: show global range, mask values outside ±threshold as black
            vmin, vmax = data_vmin, data_vmax
            # Mask values that exceed the threshold (strictly greater than)
            mask = (np.abs(velocity_slice) > threshold)
            if np.any(mask):
                plot_data = np.ma.masked_where(mask, velocity_slice)
        else:
            # Local colorbar with threshold: use ±threshold as limits
            vmin, vmax = -threshold, threshold
            # Use set_over/set_under for values exceeding limits
            cmap.set_over('black')
            cmap.set_under('black')
            # Check if local data exceeds threshold
            if local_vmin < -threshold or local_vmax > threshold:
                extend = 'both'

        # For centered symmetric colorbar (without forcing equal magnitude)
        if symmetric:
            # Use TwoSlopeNorm to center at zero
            norm = create_centered_norm(vmin, vmax, vcenter=0.0)

    else:
        # Standard colormap (no threshold)
        cmap = config.get("colormap", "seismic")
        vmin, vmax = data_vmin, data_vmax

        # Apply symmetric colorbar if needed
        if symmetric:
            # Center at zero using TwoSlopeNorm (asymmetric range allowed)
            norm = create_centered_norm(vmin, vmax, vcenter=0.0)

    # Plot
    plot_kwargs = {
        'cmap': cmap,
        'shading': 'auto',
    }

    if norm is not None:
        plot_kwargs['norm'] = norm
    else:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax

    im = ax.pcolormesh(X, Y, plot_data, **plot_kwargs)

    # Add fair recovery contours if provided
    if recovery_mask is not None and config.get("show_recovery_contours", False):
        figure_config = config.get("figure", {})
        contour_color = figure_config.get("recovery_contour_color", "green")
        contour_linestyle = figure_config.get("recovery_contour_linestyle", "-")
        contour_linewidth = figure_config.get("recovery_contour_linewidth", 2)

        # Plot contours around fair recovery regions
        ax.contour(
            X, Y, recovery_mask.astype(float),
            levels=[0.5],  # Boundary between True (1) and False (0)
            colors=contour_color,
            linestyles=contour_linestyle,
            linewidths=contour_linewidth,
        )

    # Colorbar
    if config["show_colorbar"]:
        cbar = plt.colorbar(im, ax=ax, extend=extend)
        # Determine label based on plot type
        if is_pattern_plot:
            label = "Velocity Perturbation (%)"
        elif "diff" in title_prefix.lower():
            label = "Velocity Difference (km/s)"
        else:
            label = "Velocity (km/s)"
        cbar.set_label(label, fontsize=config["font_size"]["colorbar"])

    # Labels and title
    ax.set_xlabel("X (km)", fontsize=config["font_size"]["labels"])
    ax.set_ylabel("Y (km)", fontsize=config["font_size"]["labels"])

    if config["show_title"]:
        title = f"{title_prefix}Depth Slice at Z = {z_value:.1f} km"
        ax.set_title(title, fontsize=config["font_size"]["title"])

    # Grid
    if config["show_grid"]:
        ax.grid(True, alpha=0.3)

    # Aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Save
    plt.tight_layout()
    plt.savefig(output_path, format=config["figure"]["format"], dpi=config["figure"]["dpi"])
    plt.close()


def plot_xz_slice(
    x_coords: np.ndarray,
    z_coords: np.ndarray,
    velocity_slice: np.ndarray,
    y_value: float,
    output_path: Path,
    config: Dict,
    title_prefix: str = "",
    recovery_mask: Optional[np.ndarray] = None,
) -> None:
    """
    Plot XZ (N-S vertical) slice.

    Args:
        x_coords: X coordinates (km)
        z_coords: Z coordinates (km)
        velocity_slice: 2D velocity array (nz, nx)
        y_value: Y coordinate value (km)
        output_path: Output file path
        config: Visualization configuration
        title_prefix: Prefix for title
        recovery_mask: Optional 2D boolean mask for fair recovery contours
    """
    fig, ax = plt.subplots(
        figsize=config["figure"]["figsize"],
        dpi=config["figure"]["dpi"],
    )

    # Create meshgrid for pcolormesh
    X, Z = np.meshgrid(x_coords, z_coords)

    # Get configuration options
    threshold = config.get("amplitude_threshold_viz")
    is_pattern_plot = "pattern" in title_prefix.lower() or "pct" in title_prefix.lower()
    use_uniform = config.get("use_uniform_colorbar", False)
    symmetric = config.get("symmetric_colorbar", False)

    # Get pre-computed vmin/vmax from config (global if uniform, None otherwise)
    config_vmin = config.get("vmin")
    config_vmax = config.get("vmax")

    # Compute local slice limits
    local_vmin = np.nanmin(velocity_slice)
    local_vmax = np.nanmax(velocity_slice)

    # Determine the data range to use
    if use_uniform and config_vmin is not None and config_vmax is not None:
        # Use global limits from config
        data_vmin, data_vmax = config_vmin, config_vmax
    else:
        # Use local slice limits
        data_vmin, data_vmax = local_vmin, local_vmax

    # Handle threshold and colorbar limits for pattern plots
    norm = None
    extend = 'neither'
    plot_data = velocity_slice

    if is_pattern_plot and threshold is not None:
        # Get base colormap and set bad color for masked values
        cmap = plt.get_cmap(config.get("colormap", "seismic")).copy()
        cmap.set_bad('black')

        # Determine colorbar limits
        if use_uniform:
            # Uniform colorbar: show global range, mask values outside ±threshold as black
            vmin, vmax = data_vmin, data_vmax
            # Mask values that exceed the threshold (strictly greater than)
            mask = (np.abs(velocity_slice) > threshold)
            if np.any(mask):
                plot_data = np.ma.masked_where(mask, velocity_slice)
        else:
            # Local colorbar with threshold: use ±threshold as limits
            vmin, vmax = -threshold, threshold
            # Use set_over/set_under for values exceeding limits
            cmap.set_over('black')
            cmap.set_under('black')
            # Check if local data exceeds threshold
            if local_vmin < -threshold or local_vmax > threshold:
                extend = 'both'

        # For centered symmetric colorbar (without forcing equal magnitude)
        if symmetric:
            # Use TwoSlopeNorm to center at zero
            norm = create_centered_norm(vmin, vmax, vcenter=0.0)

    else:
        # Standard colormap (no threshold)
        cmap = config.get("colormap", "seismic")
        vmin, vmax = data_vmin, data_vmax

        # Apply symmetric colorbar if needed
        if symmetric:
            # Center at zero using TwoSlopeNorm (asymmetric range allowed)
            norm = create_centered_norm(vmin, vmax, vcenter=0.0)

    # Plot
    plot_kwargs = {
        'cmap': cmap,
        'shading': 'auto',
    }

    if norm is not None:
        plot_kwargs['norm'] = norm
    else:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax

    im = ax.pcolormesh(X, Z, plot_data, **plot_kwargs)

    # Add fair recovery contours if provided
    if recovery_mask is not None and config.get("show_recovery_contours", False):
        figure_config = config.get("figure", {})
        contour_color = figure_config.get("recovery_contour_color", "green")
        contour_linestyle = figure_config.get("recovery_contour_linestyle", "-")
        contour_linewidth = figure_config.get("recovery_contour_linewidth", 2)

        # Plot contours around fair recovery regions
        ax.contour(
            X, Z, recovery_mask.astype(float),
            levels=[0.5],
            colors=contour_color,
            linestyles=contour_linestyle,
            linewidths=contour_linewidth,
        )

    # Colorbar
    if config["show_colorbar"]:
        cbar = plt.colorbar(im, ax=ax, extend=extend)
        # Determine label based on plot type
        if is_pattern_plot:
            label = "Velocity Perturbation (%)"
        elif "diff" in title_prefix.lower():
            label = "Velocity Difference (km/s)"
        else:
            label = "Velocity (km/s)"
        cbar.set_label(label, fontsize=config["font_size"]["colorbar"])

    # Labels and title
    ax.set_xlabel("X (km)", fontsize=config["font_size"]["labels"])
    ax.set_ylabel("Depth (km)", fontsize=config["font_size"]["labels"])

    if config["show_title"]:
        title = f"{title_prefix}N-S Vertical Slice at Y = {y_value:.1f} km"
        ax.set_title(title, fontsize=config["font_size"]["title"])

    # Grid
    if config["show_grid"]:
        ax.grid(True, alpha=0.3)

    # Invert y-axis (depth increases downward)
    ax.invert_yaxis()

    # Save
    plt.tight_layout()
    plt.savefig(output_path, format=config["figure"]["format"], dpi=config["figure"]["dpi"])
    plt.close()


def plot_yz_slice(
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    velocity_slice: np.ndarray,
    x_value: float,
    output_path: Path,
    config: Dict,
    title_prefix: str = "",
    recovery_mask: Optional[np.ndarray] = None,
) -> None:
    """
    Plot YZ (E-W vertical) slice.

    Args:
        y_coords: Y coordinates (km)
        z_coords: Z coordinates (km)
        velocity_slice: 2D velocity array (nz, ny)
        x_value: X coordinate value (km)
        output_path: Output file path
        config: Visualization configuration
        title_prefix: Prefix for title
        recovery_mask: Optional 2D boolean mask for fair recovery contours
    """
    fig, ax = plt.subplots(
        figsize=config["figure"]["figsize"],
        dpi=config["figure"]["dpi"],
    )

    # Create meshgrid for pcolormesh
    Y, Z = np.meshgrid(y_coords, z_coords)

    # Get configuration options
    threshold = config.get("amplitude_threshold_viz")
    is_pattern_plot = "pattern" in title_prefix.lower() or "pct" in title_prefix.lower()
    use_uniform = config.get("use_uniform_colorbar", False)
    symmetric = config.get("symmetric_colorbar", False)

    # Get pre-computed vmin/vmax from config (global if uniform, None otherwise)
    config_vmin = config.get("vmin")
    config_vmax = config.get("vmax")

    # Compute local slice limits
    local_vmin = np.nanmin(velocity_slice)
    local_vmax = np.nanmax(velocity_slice)

    # Determine the data range to use
    if use_uniform and config_vmin is not None and config_vmax is not None:
        # Use global limits from config
        data_vmin, data_vmax = config_vmin, config_vmax
    else:
        # Use local slice limits
        data_vmin, data_vmax = local_vmin, local_vmax

    # Handle threshold and colorbar limits for pattern plots
    norm = None
    extend = 'neither'
    plot_data = velocity_slice

    if is_pattern_plot and threshold is not None:
        # Get base colormap and set bad color for masked values
        cmap = plt.get_cmap(config.get("colormap", "seismic")).copy()
        cmap.set_bad('black')

        # Determine colorbar limits
        if use_uniform:
            # Uniform colorbar: show global range, mask values outside ±threshold as black
            vmin, vmax = data_vmin, data_vmax
            # Mask values that exceed the threshold (strictly greater than)
            mask = (np.abs(velocity_slice) > threshold)
            if np.any(mask):
                plot_data = np.ma.masked_where(mask, velocity_slice)
        else:
            # Local colorbar with threshold: use ±threshold as limits
            vmin, vmax = -threshold, threshold
            # Use set_over/set_under for values exceeding limits
            cmap.set_over('black')
            cmap.set_under('black')
            # Check if local data exceeds threshold
            if local_vmin < -threshold or local_vmax > threshold:
                extend = 'both'

        # For centered symmetric colorbar (without forcing equal magnitude)
        if symmetric:
            # Use TwoSlopeNorm to center at zero
            norm = create_centered_norm(vmin, vmax, vcenter=0.0)

    else:
        # Standard colormap (no threshold)
        cmap = config.get("colormap", "seismic")
        vmin, vmax = data_vmin, data_vmax

        # Apply symmetric colorbar if needed
        if symmetric:
            # Center at zero using TwoSlopeNorm (asymmetric range allowed)
            norm = create_centered_norm(vmin, vmax, vcenter=0.0)

    # Plot
    plot_kwargs = {
        'cmap': cmap,
        'shading': 'auto',
    }

    if norm is not None:
        plot_kwargs['norm'] = norm
    else:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax

    im = ax.pcolormesh(Y, Z, plot_data, **plot_kwargs)

    # Add fair recovery contours if provided
    if recovery_mask is not None and config.get("show_recovery_contours", False):
        figure_config = config.get("figure", {})
        contour_color = figure_config.get("recovery_contour_color", "green")
        contour_linestyle = figure_config.get("recovery_contour_linestyle", "-")
        contour_linewidth = figure_config.get("recovery_contour_linewidth", 2)

        # Plot contours around fair recovery regions
        ax.contour(
            Y, Z, recovery_mask.astype(float),
            levels=[0.5],
            colors=contour_color,
            linestyles=contour_linestyle,
            linewidths=contour_linewidth,
        )

    # Colorbar
    if config["show_colorbar"]:
        cbar = plt.colorbar(im, ax=ax, extend=extend)
        # Determine label based on plot type
        if is_pattern_plot:
            label = "Velocity Perturbation (%)"
        elif "diff" in title_prefix.lower():
            label = "Velocity Difference (km/s)"
        else:
            label = "Velocity (km/s)"
        cbar.set_label(label, fontsize=config["font_size"]["colorbar"])

    # Labels and title
    ax.set_xlabel("Y (km)", fontsize=config["font_size"]["labels"])
    ax.set_ylabel("Depth (km)", fontsize=config["font_size"]["labels"])

    if config["show_title"]:
        title = f"{title_prefix}E-W Vertical Slice at X = {x_value:.1f} km"
        ax.set_title(title, fontsize=config["font_size"]["title"])

    # Grid
    if config["show_grid"]:
        ax.grid(True, alpha=0.3)

    # Invert y-axis (depth increases downward)
    ax.invert_yaxis()

    # Save
    plt.tight_layout()
    plt.savefig(output_path, format=config["figure"]["format"], dpi=config["figure"]["dpi"])
    plt.close()
