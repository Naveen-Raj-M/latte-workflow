"""
Visualization module for difference plots (inverted - checkerboard).

Supports: centered_at_zero, uniform_cbar
Does NOT support: threshold, recovery contours
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

logger = logging.getLogger(__name__)


def compute_global_limits(velocity_3d: np.ndarray) -> Tuple[float, float]:
    """Compute global min/max across entire 3D volume."""
    return float(np.nanmin(velocity_3d)), float(np.nanmax(velocity_3d))


def create_centered_norm(vmin: float, vmax: float) -> mcolors.TwoSlopeNorm:
    """
    Create a TwoSlopeNorm centered at zero.

    This centers the colormap at zero WITHOUT forcing symmetric +/- values.
    Example: data range [-5, +10] will have white/neutral at 0.

    Args:
        vmin: Minimum value
        vmax: Maximum value

    Returns:
        TwoSlopeNorm for centered colormap
    """
    # Handle edge cases
    if vmin >= 0:
        vmin = -1e-6
    if vmax <= 0:
        vmax = 1e-6
    return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)


def plot_difference_xy_slice(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    velocity_slice: np.ndarray,
    z_value: float,
    output_path: Path,
    config: Dict,
    title_prefix: str = "",
    global_vmin: Optional[float] = None,
    global_vmax: Optional[float] = None,
) -> None:
    """
    Plot XY slice for difference visualization.

    NO threshold support.
    NO recovery contour support.

    Args:
        x_coords: X coordinates (km)
        y_coords: Y coordinates (km)
        velocity_slice: 2D velocity difference array (ny, nx)
        z_value: Depth value (km)
        output_path: Output file path
        config: Visualization configuration
        title_prefix: Prefix for title
        global_vmin: Pre-computed global minimum (used if uniform_cbar=True)
        global_vmax: Pre-computed global maximum (used if uniform_cbar=True)
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

    cmap_name = config.get("colormap", "seismic")
    plot_kwargs = {'cmap': cmap_name, 'shading': 'auto'}

    if config.get("centered_at_zero", True):
        norm = create_centered_norm(vmin, vmax)
        plot_kwargs['norm'] = norm
    else:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax

    im = ax.pcolormesh(X, Y, velocity_slice, **plot_kwargs)

    if config.get("show_colorbar", True):
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Velocity Difference (km/s)", fontsize=config["font_size"]["colorbar"])

    ax.set_xlabel("X (km)", fontsize=config["font_size"]["labels"])
    ax.set_ylabel("Y (km)", fontsize=config["font_size"]["labels"])

    if config.get("show_title", True):
        ax.set_title(f"{title_prefix}Depth Slice at Z = {z_value:.1f} km",
                     fontsize=config["font_size"]["title"])

    if config.get("show_grid", True):
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format=config["figure"]["format"], dpi=config["figure"]["dpi"])
    plt.close()


def plot_difference_xz_slice(
    x_coords: np.ndarray,
    z_coords: np.ndarray,
    velocity_slice: np.ndarray,
    y_value: float,
    output_path: Path,
    config: Dict,
    title_prefix: str = "",
    global_vmin: Optional[float] = None,
    global_vmax: Optional[float] = None,
) -> None:
    """
    Plot XZ slice for difference visualization.

    Args:
        x_coords: X coordinates (km)
        z_coords: Z coordinates (km)
        velocity_slice: 2D velocity difference array (nz, nx)
        y_value: Y coordinate value (km)
        output_path: Output file path
        config: Visualization configuration
        title_prefix: Prefix for title
        global_vmin: Pre-computed global minimum
        global_vmax: Pre-computed global maximum
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

    cmap_name = config.get("colormap", "seismic")
    plot_kwargs = {'cmap': cmap_name, 'shading': 'auto'}

    if config.get("centered_at_zero", True):
        norm = create_centered_norm(vmin, vmax)
        plot_kwargs['norm'] = norm
    else:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax

    im = ax.pcolormesh(X, Z, velocity_slice, **plot_kwargs)

    # Invert y-axis so depth increases downward
    ax.invert_yaxis()

    if config.get("show_colorbar", True):
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Velocity Difference (km/s)", fontsize=config["font_size"]["colorbar"])

    ax.set_xlabel("X (km)", fontsize=config["font_size"]["labels"])
    ax.set_ylabel("Z (km)", fontsize=config["font_size"]["labels"])

    if config.get("show_title", True):
        ax.set_title(f"{title_prefix}N-S Slice at Y = {y_value:.1f} km",
                     fontsize=config["font_size"]["title"])

    if config.get("show_grid", True):
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format=config["figure"]["format"], dpi=config["figure"]["dpi"])
    plt.close()


def plot_difference_yz_slice(
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    velocity_slice: np.ndarray,
    x_value: float,
    output_path: Path,
    config: Dict,
    title_prefix: str = "",
    global_vmin: Optional[float] = None,
    global_vmax: Optional[float] = None,
) -> None:
    """
    Plot YZ slice for difference visualization.

    Args:
        y_coords: Y coordinates (km)
        z_coords: Z coordinates (km)
        velocity_slice: 2D velocity difference array (nz, ny)
        x_value: X coordinate value (km)
        output_path: Output file path
        config: Visualization configuration
        title_prefix: Prefix for title
        global_vmin: Pre-computed global minimum
        global_vmax: Pre-computed global maximum
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

    cmap_name = config.get("colormap", "seismic")
    plot_kwargs = {'cmap': cmap_name, 'shading': 'auto'}

    if config.get("centered_at_zero", True):
        norm = create_centered_norm(vmin, vmax)
        plot_kwargs['norm'] = norm
    else:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax

    im = ax.pcolormesh(Y, Z, velocity_slice, **plot_kwargs)

    # Invert y-axis so depth increases downward
    ax.invert_yaxis()

    if config.get("show_colorbar", True):
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Velocity Difference (km/s)", fontsize=config["font_size"]["colorbar"])

    ax.set_xlabel("Y (km)", fontsize=config["font_size"]["labels"])
    ax.set_ylabel("Z (km)", fontsize=config["font_size"]["labels"])

    if config.get("show_title", True):
        ax.set_title(f"{title_prefix}E-W Slice at X = {x_value:.1f} km",
                     fontsize=config["font_size"]["title"])

    if config.get("show_grid", True):
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format=config["figure"]["format"], dpi=config["figure"]["dpi"])
    plt.close()


def visualize_difference(
    df_inverted: pd.DataFrame,
    df_checkerboard: pd.DataFrame,
    output_dir: Path,
    config: Dict,
    velocity_field: str,
) -> None:
    """
    Visualize difference between inverted and checkerboard models.

    Args:
        df_inverted: Inverted model DataFrame
        df_checkerboard: Checkerboard model DataFrame
        output_dir: Output directory
        config: From visualize_difference section
        velocity_field: "vp" or "vs"
    """
    logger.info(f"Visualizing difference for {velocity_field}")

    # Create output subdirectory
    model_name = f"difference_{velocity_field}" if config.get("velocity_field") == "both" else "difference"
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Compute difference
    df_diff = pd.DataFrame({
        'X_grid_km': df_inverted['X_grid_km'].values,
        'Y_grid_km': df_inverted['Y_grid_km'].values,
        'Z_grid_km': df_inverted['Z_grid_km'].values,
        velocity_field: df_inverted[velocity_field].values - df_checkerboard[velocity_field].values
    })

    # Prepare 3D grid
    x_coords, y_coords, z_coords, diff_3d = prepare_3d_grid(df_diff, velocity_field)
    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)

    # Pre-compute global limits if uniform_cbar enabled
    global_vmin, global_vmax = None, None
    if config.get("uniform_cbar", False):
        global_vmin, global_vmax = compute_global_limits(diff_3d)
        logger.info(f"  Using uniform colorbar: [{global_vmin:.4f}, {global_vmax:.4f}]")

    # Get slice configuration
    planes = config.get("planes", ["xy", "xz", "yz"])
    xy_slices = config.get("xy_slices")
    xz_slices = config.get("xz_slices")
    yz_slices = config.get("yz_slices")
    slice_spacing = config.get("slice_spacing", {})

    slice_unit = config.get("slice_unit", "indices")
    spacing_unit = config.get("spacing_unit", "indices")
    tolerance = config.get("tolerance_km", 1e-6)

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
            for k in tqdm(z_indices, desc=f"    difference {velocity_field} XY slices"):
                if k >= nz:
                    continue

                velocity_slice = diff_3d[k, :, :]
                z_value = z_coords[k]

                output_path = model_dir / f"{velocity_field}_xy_z{k:03d}.{config['figure']['format']}"

                plot_difference_xy_slice(
                    x_coords, y_coords, velocity_slice, z_value,
                    output_path, config,
                    title_prefix=f"Difference {velocity_field.upper()} - ",
                    global_vmin=global_vmin,
                    global_vmax=global_vmax,
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
            for j in tqdm(y_indices, desc=f"    difference {velocity_field} XZ slices"):
                if j >= ny:
                    continue

                velocity_slice = diff_3d[:, j, :]
                y_value = y_coords[j]

                output_path = model_dir / f"{velocity_field}_xz_y{j:03d}.{config['figure']['format']}"

                plot_difference_xz_slice(
                    x_coords, z_coords, velocity_slice, y_value,
                    output_path, config,
                    title_prefix=f"Difference {velocity_field.upper()} - ",
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
            for i in tqdm(x_indices, desc=f"    difference {velocity_field} YZ slices"):
                if i >= nx:
                    continue

                velocity_slice = diff_3d[:, :, i]
                x_value = x_coords[i]

                output_path = model_dir / f"{velocity_field}_yz_x{i:03d}.{config['figure']['format']}"

                plot_difference_yz_slice(
                    y_coords, z_coords, velocity_slice, x_value,
                    output_path, config,
                    title_prefix=f"Difference {velocity_field.upper()} - ",
                    global_vmin=global_vmin,
                    global_vmax=global_vmax,
                )

    logger.info(f"  Visualization complete: {model_dir}")
