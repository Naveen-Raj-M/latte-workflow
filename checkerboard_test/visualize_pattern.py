"""
Visualization module for pattern plots (checkerboard_pattern, inverted_pattern).

These show percentage perturbation from reference model.
Supports: centered_at_zero, uniform_cbar, threshold with black regions, recovery contours.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from checkerboard_test.grid_utils import (
    prepare_3d_grid,
    parse_slice_specification,
    convert_km_spacing_to_indices,
)
from checkerboard_test.recovery_analysis import compute_fair_recovery_mask, compute_recovery_metrics
from analysis.basin_boundaries import plot_basin_boundaries

logger = logging.getLogger(__name__)


def compute_global_limits(velocity_3d: np.ndarray) -> Tuple[float, float]:
    """Compute global min/max across entire 3D volume."""
    return float(np.nanmin(velocity_3d)), float(np.nanmax(velocity_3d))


def create_centered_norm(vmin: float, vmax: float) -> mcolors.Normalize:
    """
    Create a symmetric Normalize centered at zero.

    Uses abs_max = max(|vmin|, |vmax|) so the colorbar spans [-abs_max, +abs_max].
    Zero maps to the physical midpoint of the colorbar, with ticks on both sides.

    Args:
        vmin: Minimum value
        vmax: Maximum value

    Returns:
        Normalize with symmetric range around zero
    """
    abs_max = max(abs(vmin), abs(vmax))
    if abs_max == 0:
        abs_max = 1e-6
    return mcolors.Normalize(vmin=-abs_max, vmax=abs_max)


def create_threshold_colormap(
    base_cmap_name: str,
    vmin: float,
    vmax: float,
    threshold: float,
    n_samples: int = 512,
) -> mcolors.ListedColormap:
    """
    Create a colormap with black regions beyond ±threshold.

    Maps:
    - [vmin, -threshold] → black
    - [-threshold, +threshold] → base colormap
    - [+threshold, vmax]  → black

    This makes both the plot and the colorbar show black beyond the threshold.

    Args:
        base_cmap_name: Name of the base matplotlib colormap
        vmin: Norm minimum (norm.vmin)
        vmax: Norm maximum (norm.vmax)
        threshold: Values with |value| > threshold are coloured black
        n_samples: Number of discrete colour samples for smoothness

    Returns:
        ListedColormap with black beyond ±threshold
    """
    base_cmap = plt.get_cmap(base_cmap_name)
    total_range = vmax - vmin
    if total_range <= 0:
        total_range = 1e-6

    # Positions of ±threshold in [0, 1] normalised space
    pos_neg = max(0.0, (-threshold - vmin) / total_range)
    pos_pos = min(1.0, ( threshold - vmin) / total_range)

    colors = []
    for i in range(n_samples):
        t = i / (n_samples - 1)
        if t <= pos_neg or t >= pos_pos:
            colors.append((0.0, 0.0, 0.0, 1.0))
        else:
            t_base = (t - pos_neg) / (pos_pos - pos_neg)
            colors.append(base_cmap(t_base))

    return mcolors.ListedColormap(colors)



def plot_pattern_xy_slice(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    velocity_slice: np.ndarray,
    z_value: float,
    output_path: Path,
    config: Dict,
    title_prefix: str = "",
    recovery_mask: Optional[np.ndarray] = None,
    global_vmin: Optional[float] = None,
    global_vmax: Optional[float] = None,
    basin_data: Optional[Dict] = None,
    basin_config: Optional[Dict] = None,
) -> None:
    """
    Plot XY slice for pattern visualization.

    Args:
        x_coords: X coordinates (km)
        y_coords: Y coordinates (km)
        velocity_slice: 2D velocity perturbation array (ny, nx) in %
        z_value: Depth value (km)
        output_path: Output file path
        config: Visualization configuration with:
            - centered_at_zero: bool
            - uniform_cbar: bool
            - threshold: Optional[float]
            - show_recovery_contours: bool
        title_prefix: Prefix for title
        recovery_mask: Optional 2D boolean mask for fair recovery contours
        global_vmin/vmax: Pre-computed global limits (used if uniform_cbar=True)
        basin_data: Optional basin boundary data from load_basin_boundaries()
        basin_config: Optional basin boundaries configuration for styling
    """
    fig, ax = plt.subplots(
        figsize=config["figure"]["figsize"],
        dpi=config["figure"]["dpi"],
    )

    X, Y = np.meshgrid(x_coords, y_coords)

    # Determine vmin/vmax based on uniform_cbar setting
    if config.get("uniform_cbar", False) and global_vmin is not None:
        vmin, vmax = global_vmin, global_vmax
    else:
        vmin, vmax = float(np.nanmin(velocity_slice)), float(np.nanmax(velocity_slice))

    threshold = config.get("threshold")
    centered = config.get("centered_at_zero", True)
    cmap_name = config.get("colormap", "seismic")

    # Step 1: Build norm — centered_at_zero is independent of threshold
    if centered:
        norm = create_centered_norm(vmin, vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Step 2: Build colormap — threshold bakes black into both the plot and the colorbar
    if threshold is not None:
        cmap = create_threshold_colormap(cmap_name, norm.vmin, norm.vmax, threshold)
    else:
        cmap = plt.get_cmap(cmap_name).copy()
    data_to_plot = velocity_slice

    im = ax.pcolormesh(X, Y, data_to_plot, cmap=cmap, norm=norm, shading='auto')

    # Recovery contours (only for inverted_pattern)
    if recovery_mask is not None and config.get("show_recovery_contours", False):
        figure_config = config.get("figure", {})
        ax.contour(
            X, Y, recovery_mask.astype(float),
            levels=[0.5],
            colors=figure_config.get("recovery_contour_color", "green"),
            linestyles=figure_config.get("recovery_contour_linestyle", "-"),
            linewidths=figure_config.get("recovery_contour_linewidth", 2),
        )

    tick_size = config["font_size"]["ticks"]

    if config.get("show_colorbar", True):
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Velocity Perturbation (%)", fontsize=config["font_size"]["colorbar"])
        cbar.ax.tick_params(labelsize=tick_size)

    ax.set_xlabel("X (km)", fontsize=config["font_size"]["labels"])
    ax.set_ylabel("Y (km)", fontsize=config["font_size"]["labels"])
    ax.tick_params(labelsize=tick_size)

    if config.get("show_title", True):
        ax.set_title(f"{title_prefix}Depth Slice at Z = {z_value:.1f} km",
                     fontsize=config["font_size"]["title"])

    if config.get("show_grid", True):
        ax.grid(True, alpha=0.3)

    if basin_data and basin_config:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        plot_basin_boundaries(ax, basin_data, basin_config, xlim=xlim, ylim=ylim)

    plt.tight_layout()
    plt.savefig(output_path, format=config["figure"]["format"], dpi=config["figure"]["dpi"])
    plt.close()


def plot_pattern_xz_slice(
    x_coords: np.ndarray,
    z_coords: np.ndarray,
    velocity_slice: np.ndarray,
    y_value: float,
    output_path: Path,
    config: Dict,
    title_prefix: str = "",
    recovery_mask: Optional[np.ndarray] = None,
    global_vmin: Optional[float] = None,
    global_vmax: Optional[float] = None,
) -> None:
    """
    Plot XZ slice for pattern visualization.

    Args:
        x_coords: X coordinates (km)
        z_coords: Z coordinates (km)
        velocity_slice: 2D velocity perturbation array (nz, nx) in %
        y_value: Y coordinate value (km)
        output_path: Output file path
        config: Visualization configuration
        title_prefix: Prefix for title
        recovery_mask: Optional 2D boolean mask
        global_vmin/vmax: Pre-computed global limits
    """
    fig, ax = plt.subplots(
        figsize=config["figure"]["figsize"],
        dpi=config["figure"]["dpi"],
    )

    X, Z = np.meshgrid(x_coords, z_coords)

    # Determine vmin/vmax
    if config.get("uniform_cbar", False) and global_vmin is not None:
        vmin, vmax = global_vmin, global_vmax
    else:
        vmin, vmax = float(np.nanmin(velocity_slice)), float(np.nanmax(velocity_slice))

    threshold = config.get("threshold")
    centered = config.get("centered_at_zero", True)
    cmap_name = config.get("colormap", "seismic")

    # Step 1: Build norm — centered_at_zero is independent of threshold
    if centered:
        norm = create_centered_norm(vmin, vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Step 2: Build colormap — threshold bakes black into both the plot and the colorbar
    if threshold is not None:
        cmap = create_threshold_colormap(cmap_name, norm.vmin, norm.vmax, threshold)
    else:
        cmap = plt.get_cmap(cmap_name).copy()
    data_to_plot = velocity_slice

    im = ax.pcolormesh(X, Z, data_to_plot, cmap=cmap, norm=norm, shading='auto')

    # Invert y-axis so depth increases downward
    ax.invert_yaxis()

    # Recovery contours
    if recovery_mask is not None and config.get("show_recovery_contours", False):
        figure_config = config.get("figure", {})
        ax.contour(
            X, Z, recovery_mask.astype(float),
            levels=[0.5],
            colors=figure_config.get("recovery_contour_color", "green"),
            linestyles=figure_config.get("recovery_contour_linestyle", "-"),
            linewidths=figure_config.get("recovery_contour_linewidth", 2),
        )

    tick_size = config["font_size"]["ticks"]

    if config.get("show_colorbar", True):
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Velocity Perturbation (%)", fontsize=config["font_size"]["colorbar"])
        cbar.ax.tick_params(labelsize=tick_size)

    ax.set_xlabel("X (km)", fontsize=config["font_size"]["labels"])
    ax.set_ylabel("Z (km)", fontsize=config["font_size"]["labels"])
    ax.tick_params(labelsize=tick_size)

    if config.get("show_title", True):
        ax.set_title(f"{title_prefix}N-S Slice at Y = {y_value:.1f} km",
                     fontsize=config["font_size"]["title"])

    if config.get("show_grid", True):
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format=config["figure"]["format"], dpi=config["figure"]["dpi"])
    plt.close()


def plot_pattern_yz_slice(
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    velocity_slice: np.ndarray,
    x_value: float,
    output_path: Path,
    config: Dict,
    title_prefix: str = "",
    recovery_mask: Optional[np.ndarray] = None,
    global_vmin: Optional[float] = None,
    global_vmax: Optional[float] = None,
) -> None:
    """
    Plot YZ slice for pattern visualization.

    Args:
        y_coords: Y coordinates (km)
        z_coords: Z coordinates (km)
        velocity_slice: 2D velocity perturbation array (nz, ny) in %
        x_value: X coordinate value (km)
        output_path: Output file path
        config: Visualization configuration
        title_prefix: Prefix for title
        recovery_mask: Optional 2D boolean mask
        global_vmin/vmax: Pre-computed global limits
    """
    fig, ax = plt.subplots(
        figsize=config["figure"]["figsize"],
        dpi=config["figure"]["dpi"],
    )

    Y, Z = np.meshgrid(y_coords, z_coords)

    # Determine vmin/vmax
    if config.get("uniform_cbar", False) and global_vmin is not None:
        vmin, vmax = global_vmin, global_vmax
    else:
        vmin, vmax = float(np.nanmin(velocity_slice)), float(np.nanmax(velocity_slice))

    threshold = config.get("threshold")
    centered = config.get("centered_at_zero", True)
    cmap_name = config.get("colormap", "seismic")

    # Step 1: Build norm — centered_at_zero is independent of threshold
    if centered:
        norm = create_centered_norm(vmin, vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Step 2: Build colormap — threshold bakes black into both the plot and the colorbar
    if threshold is not None:
        cmap = create_threshold_colormap(cmap_name, norm.vmin, norm.vmax, threshold)
    else:
        cmap = plt.get_cmap(cmap_name).copy()
    data_to_plot = velocity_slice

    im = ax.pcolormesh(Y, Z, data_to_plot, cmap=cmap, norm=norm, shading='auto')

    # Invert y-axis so depth increases downward
    ax.invert_yaxis()

    # Recovery contours
    if recovery_mask is not None and config.get("show_recovery_contours", False):
        figure_config = config.get("figure", {})
        ax.contour(
            Y, Z, recovery_mask.astype(float),
            levels=[0.5],
            colors=figure_config.get("recovery_contour_color", "green"),
            linestyles=figure_config.get("recovery_contour_linestyle", "-"),
            linewidths=figure_config.get("recovery_contour_linewidth", 2),
        )

    tick_size = config["font_size"]["ticks"]

    if config.get("show_colorbar", True):
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Velocity Perturbation (%)", fontsize=config["font_size"]["colorbar"])
        cbar.ax.tick_params(labelsize=tick_size)

    ax.set_xlabel("Y (km)", fontsize=config["font_size"]["labels"])
    ax.set_ylabel("Z (km)", fontsize=config["font_size"]["labels"])
    ax.tick_params(labelsize=tick_size)

    if config.get("show_title", True):
        ax.set_title(f"{title_prefix}E-W Slice at X = {x_value:.1f} km",
                     fontsize=config["font_size"]["title"])

    if config.get("show_grid", True):
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format=config["figure"]["format"], dpi=config["figure"]["dpi"])
    plt.close()


def visualize_pattern(
    df_pattern: pd.DataFrame,
    output_dir: Path,
    config: Dict,
    pattern_name: str,
    velocity_field: str,
    checkerboard_pattern_3d: Optional[np.ndarray] = None,
    basin_data: Optional[Dict] = None,
    basin_config: Optional[Dict] = None,
) -> None:
    """
    Main function to visualize a pattern.

    Args:
        df_pattern: DataFrame with perturbation values (% in velocity_field column)
        output_dir: Output directory
        config: From visualize_checkerboard_pattern or visualize_inverted_pattern section
        pattern_name: "checkerboard_pattern" or "inverted_pattern"
        velocity_field: "vp" or "vs"
        checkerboard_pattern_3d: For recovery contour computation (inverted_pattern only)
        basin_data: Optional basin boundary data from load_basin_boundaries()
        basin_config: Optional basin boundaries configuration for styling
    """
    logger.info(f"Visualizing {pattern_name} for {velocity_field}")

    # Create output subdirectory
    model_name = f"{pattern_name}_pct_{velocity_field}" if config.get("velocity_field") == "both" else f"{pattern_name}_pct"
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Prepare 3D grid
    x_coords, y_coords, z_coords, velocity_3d = prepare_3d_grid(df_pattern, velocity_field)
    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)

    # Pre-compute global limits if uniform_cbar enabled
    global_vmin, global_vmax = None, None
    if config.get("uniform_cbar", False):
        global_vmin, global_vmax = compute_global_limits(velocity_3d)
        logger.info(f"  Using uniform colorbar: [{global_vmin:.4f}, {global_vmax:.4f}]")

    # Compute recovery mask if needed (inverted_pattern only)
    recovery_3d = None
    is_inverted_pattern = "inverted" in pattern_name
    if (is_inverted_pattern and
        checkerboard_pattern_3d is not None and
        config.get("show_recovery_contours", False)):

        recovery_threshold = config.get("recovery_threshold_pct", 0.0)
        logger.info(f"  Computing fair recovery mask with threshold {recovery_threshold}%")

        recovery_3d = compute_fair_recovery_mask(
            checkerboard_pattern_3d,
            velocity_3d,
            recovery_threshold,
        )

        # Compute and log recovery metrics
        metrics = compute_recovery_metrics(checkerboard_pattern_3d, velocity_3d)
        logger.info(f"  Recovery metrics:")
        logger.info(f"    Correlation: {metrics['correlation']:.4f}")
        logger.info(f"    RMSE: {metrics['rmse']:.4f}%")
        logger.info(f"    Mean Absolute Error: {metrics['mean_absolute_error']:.4f}%")
        logger.info(f"    Sign Agreement: {metrics['sign_agreement']:.2f}%")

    # Get slice configuration
    planes = config.get("planes", ["xy", "xz", "yz"])
    xy_slices = config.get("xy_slices")
    xz_slices = config.get("xz_slices")
    yz_slices = config.get("yz_slices")
    slice_spacing = config.get("slice_spacing", {})

    slice_unit = config.get("slice_unit", "indices")
    spacing_unit = config.get("spacing_unit", "indices")
    tolerance = config.get("tolerance_km", 1e-6)

    title_prefix = f"{pattern_name.replace('_', ' ').title()} {velocity_field.upper()} - "

    # Plot XY slices (depth slices)
    if "xy" in planes:
        logger.info(f"    Creating XY (depth) slices for {velocity_field}")

        z_indices = None
        if xy_slices is not None:
            z_indices = parse_slice_specification(xy_slices, z_coords, slice_unit, tolerance)
            if not z_indices:
                logger.warning("    No valid XY slices found, skipping")
        else:
            if spacing_unit == "km":
                spacing_km = slice_spacing.get("xy", 1.0)
                z_indices = convert_km_spacing_to_indices(spacing_km, z_coords, tolerance)
            else:
                spacing = slice_spacing.get("xy", 1)
                z_indices = range(0, nz, spacing)

        if z_indices:
            for k in tqdm(z_indices, desc=f"    {pattern_name} {velocity_field} XY slices"):
                if k >= nz:
                    continue

                velocity_slice = velocity_3d[k, :, :]
                z_value = z_coords[k]
                recovery_mask = recovery_3d[k, :, :] if recovery_3d is not None else None

                output_path = model_dir / f"{velocity_field}_xy_z{k:03d}.{config['figure']['format']}"

                plot_pattern_xy_slice(
                    x_coords, y_coords, velocity_slice, z_value,
                    output_path, config,
                    title_prefix=title_prefix,
                    recovery_mask=recovery_mask,
                    global_vmin=global_vmin,
                    global_vmax=global_vmax,
                    basin_data=basin_data,
                    basin_config=basin_config,
                )

    # Plot XZ slices (N-S vertical)
    if "xz" in planes:
        logger.info(f"    Creating XZ (N-S vertical) slices for {velocity_field}")

        y_indices = None
        if xz_slices is not None:
            y_indices = parse_slice_specification(xz_slices, y_coords, slice_unit, tolerance)
            if not y_indices:
                logger.warning("    No valid XZ slices found, skipping")
        else:
            if spacing_unit == "km":
                spacing_km = slice_spacing.get("xz", 1.0)
                y_indices = convert_km_spacing_to_indices(spacing_km, y_coords, tolerance)
            else:
                spacing = slice_spacing.get("xz", 1)
                y_indices = range(0, ny, spacing)

        if y_indices:
            for j in tqdm(y_indices, desc=f"    {pattern_name} {velocity_field} XZ slices"):
                if j >= ny:
                    continue

                velocity_slice = velocity_3d[:, j, :]
                y_value = y_coords[j]
                recovery_mask = recovery_3d[:, j, :] if recovery_3d is not None else None

                output_path = model_dir / f"{velocity_field}_xz_y{j:03d}.{config['figure']['format']}"

                plot_pattern_xz_slice(
                    x_coords, z_coords, velocity_slice, y_value,
                    output_path, config,
                    title_prefix=title_prefix,
                    recovery_mask=recovery_mask,
                    global_vmin=global_vmin,
                    global_vmax=global_vmax,
                )

    # Plot YZ slices (E-W vertical)
    if "yz" in planes:
        logger.info(f"    Creating YZ (E-W vertical) slices for {velocity_field}")

        x_indices = None
        if yz_slices is not None:
            x_indices = parse_slice_specification(yz_slices, x_coords, slice_unit, tolerance)
            if not x_indices:
                logger.warning("    No valid YZ slices found, skipping")
        else:
            if spacing_unit == "km":
                spacing_km = slice_spacing.get("yz", 1.0)
                x_indices = convert_km_spacing_to_indices(spacing_km, x_coords, tolerance)
            else:
                spacing = slice_spacing.get("yz", 1)
                x_indices = range(0, nx, spacing)

        if x_indices:
            for i in tqdm(x_indices, desc=f"    {pattern_name} {velocity_field} YZ slices"):
                if i >= nx:
                    continue

                velocity_slice = velocity_3d[:, :, i]
                x_value = x_coords[i]
                recovery_mask = recovery_3d[:, :, i] if recovery_3d is not None else None

                output_path = model_dir / f"{velocity_field}_yz_x{i:03d}.{config['figure']['format']}"

                plot_pattern_yz_slice(
                    y_coords, z_coords, velocity_slice, x_value,
                    output_path, config,
                    title_prefix=title_prefix,
                    recovery_mask=recovery_mask,
                    global_vmin=global_vmin,
                    global_vmax=global_vmax,
                )

    logger.info(f"  Visualization complete: {model_dir}")
