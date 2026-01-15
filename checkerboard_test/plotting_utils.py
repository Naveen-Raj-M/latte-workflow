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

from checkerboard_test.colormap_utils import (
    create_threshold_colormap,
    apply_threshold_to_data,
)

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

    # Determine color limits (already computed for uniform colorbar)
    vmin = config.get("vmin")
    vmax = config.get("vmax")

    # Handle threshold-based coloring for pattern plots
    threshold = config.get("amplitude_threshold_viz")
    is_pattern_plot = "pattern" in title_prefix.lower() or "pct" in title_prefix.lower()

    if is_pattern_plot and threshold is not None:
        # Apply three-color scheme
        cmap, norm = create_threshold_colormap(
            config.get("colormap", "seismic"),
            threshold=threshold,
        )

        # Extend vmin/vmax to show black regions
        velocity_slice_plot, vmin, vmax = apply_threshold_to_data(
            velocity_slice, threshold, vmin if vmin is not None else np.nanmin(velocity_slice),
            vmax if vmax is not None else np.nanmax(velocity_slice)
        )
    else:
        # Standard colormap
        cmap = config.get("colormap", "seismic")
        norm = None
        velocity_slice_plot = velocity_slice

        # Apply symmetric colorbar if needed
        if config.get("symmetric_colorbar", False):
            if vmin is None or vmax is None:
                vmax_abs = max(abs(np.nanmin(velocity_slice)), abs(np.nanmax(velocity_slice)))
                vmin = -vmax_abs
                vmax = vmax_abs

    # Use provided vmin/vmax or auto-scale
    if vmin is None:
        vmin = np.nanmin(velocity_slice_plot)
    if vmax is None:
        vmax = np.nanmax(velocity_slice_plot)

    # Plot
    im = ax.pcolormesh(
        X, Y, velocity_slice_plot,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        shading='auto',
    )

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
        cbar = plt.colorbar(im, ax=ax)
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

    # Determine color limits (already computed for uniform colorbar)
    vmin = config.get("vmin")
    vmax = config.get("vmax")

    # Handle threshold-based coloring for pattern plots
    threshold = config.get("amplitude_threshold_viz")
    is_pattern_plot = "pattern" in title_prefix.lower() or "pct" in title_prefix.lower()

    if is_pattern_plot and threshold is not None:
        # Apply three-color scheme
        cmap, norm = create_threshold_colormap(
            config.get("colormap", "seismic"),
            threshold=threshold,
        )

        # Extend vmin/vmax to show black regions
        velocity_slice_plot, vmin, vmax = apply_threshold_to_data(
            velocity_slice, threshold, vmin if vmin is not None else np.nanmin(velocity_slice),
            vmax if vmax is not None else np.nanmax(velocity_slice)
        )
    else:
        # Standard colormap
        cmap = config.get("colormap", "seismic")
        norm = None
        velocity_slice_plot = velocity_slice

        # Apply symmetric colorbar if needed
        if config.get("symmetric_colorbar", False):
            if vmin is None or vmax is None:
                vmax_abs = max(abs(np.nanmin(velocity_slice)), abs(np.nanmax(velocity_slice)))
                vmin = -vmax_abs
                vmax = vmax_abs

    # Use provided vmin/vmax or auto-scale
    if vmin is None:
        vmin = np.nanmin(velocity_slice_plot)
    if vmax is None:
        vmax = np.nanmax(velocity_slice_plot)

    # Plot
    im = ax.pcolormesh(
        X, Z, velocity_slice_plot,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        shading='auto',
    )

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
        cbar = plt.colorbar(im, ax=ax)
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

    # Determine color limits (already computed for uniform colorbar)
    vmin = config.get("vmin")
    vmax = config.get("vmax")

    # Handle threshold-based coloring for pattern plots
    threshold = config.get("amplitude_threshold_viz")
    is_pattern_plot = "pattern" in title_prefix.lower() or "pct" in title_prefix.lower()

    if is_pattern_plot and threshold is not None:
        # Apply three-color scheme
        cmap, norm = create_threshold_colormap(
            config.get("colormap", "seismic"),
            threshold=threshold,
        )

        # Extend vmin/vmax to show black regions
        velocity_slice_plot, vmin, vmax = apply_threshold_to_data(
            velocity_slice, threshold, vmin if vmin is not None else np.nanmin(velocity_slice),
            vmax if vmax is not None else np.nanmax(velocity_slice)
        )
    else:
        # Standard colormap
        cmap = config.get("colormap", "seismic")
        norm = None
        velocity_slice_plot = velocity_slice

        # Apply symmetric colorbar if needed
        if config.get("symmetric_colorbar", False):
            if vmin is None or vmax is None:
                vmax_abs = max(abs(np.nanmin(velocity_slice)), abs(np.nanmax(velocity_slice)))
                vmin = -vmax_abs
                vmax = vmax_abs

    # Use provided vmin/vmax or auto-scale
    if vmin is None:
        vmin = np.nanmin(velocity_slice_plot)
    if vmax is None:
        vmax = np.nanmax(velocity_slice_plot)

    # Plot
    im = ax.pcolormesh(
        Y, Z, velocity_slice_plot,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        shading='auto',
    )

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
        cbar = plt.colorbar(im, ax=ax)
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
