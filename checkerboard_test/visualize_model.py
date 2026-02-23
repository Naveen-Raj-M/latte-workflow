"""
Visualization module for velocity models (input, inverted, checkerboard).

These plots show absolute velocity values without centering or thresholds.
Always uses autoscaling - no manual vmin/vmax.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from checkerboard_test.grid_utils import (
    prepare_3d_grid,
    parse_slice_specification,
    convert_km_spacing_to_indices,
)
from analysis.basin_boundaries import plot_basin_boundaries

logger = logging.getLogger(__name__)


def plot_model_xy_slice(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    velocity_slice: np.ndarray,
    z_value: float,
    output_path: Path,
    config: Dict,
    title_prefix: str = "",
    basin_data: Optional[Dict] = None,
    basin_config: Optional[Dict] = None,
) -> None:
    """
    Plot XY (horizontal/depth) slice for model visualization.

    Always autoscales - no centering, no threshold.

    Args:
        x_coords: X coordinates (km)
        y_coords: Y coordinates (km)
        velocity_slice: 2D velocity array (ny, nx)
        z_value: Depth value (km)
        output_path: Output file path
        config: Visualization configuration
        title_prefix: Prefix for title
        basin_data: Optional basin boundary data from load_basin_boundaries()
        basin_config: Optional basin boundaries configuration for styling
    """
    fig, ax = plt.subplots(
        figsize=config["figure"]["figsize"],
        dpi=config["figure"]["dpi"],
    )

    X, Y = np.meshgrid(x_coords, y_coords)

    # Always autoscale from local slice data
    vmin = np.nanmin(velocity_slice)
    vmax = np.nanmax(velocity_slice)

    cmap = config.get("colormap", "seismic")

    im = ax.pcolormesh(X, Y, velocity_slice, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')

    if config.get("show_colorbar", True):
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Velocity (km/s)", fontsize=config["font_size"]["colorbar"])

    ax.set_xlabel("X (km)", fontsize=config["font_size"]["labels"])
    ax.set_ylabel("Y (km)", fontsize=config["font_size"]["labels"])

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


def plot_model_xz_slice(
    x_coords: np.ndarray,
    z_coords: np.ndarray,
    velocity_slice: np.ndarray,
    y_value: float,
    output_path: Path,
    config: Dict,
    title_prefix: str = "",
) -> None:
    """
    Plot XZ (N-S vertical) slice for model visualization.

    Always autoscales - no centering, no threshold.

    Args:
        x_coords: X coordinates (km)
        z_coords: Z coordinates (km)
        velocity_slice: 2D velocity array (nz, nx)
        y_value: Y coordinate value (km)
        output_path: Output file path
        config: Visualization configuration
        title_prefix: Prefix for title
    """
    fig, ax = plt.subplots(
        figsize=config["figure"]["figsize"],
        dpi=config["figure"]["dpi"],
    )

    X, Z = np.meshgrid(x_coords, z_coords)

    # Always autoscale from local slice data
    vmin = np.nanmin(velocity_slice)
    vmax = np.nanmax(velocity_slice)

    cmap = config.get("colormap", "seismic")

    im = ax.pcolormesh(X, Z, velocity_slice, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')

    # Invert y-axis so depth increases downward
    ax.invert_yaxis()

    if config.get("show_colorbar", True):
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Velocity (km/s)", fontsize=config["font_size"]["colorbar"])

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


def plot_model_yz_slice(
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    velocity_slice: np.ndarray,
    x_value: float,
    output_path: Path,
    config: Dict,
    title_prefix: str = "",
) -> None:
    """
    Plot YZ (E-W vertical) slice for model visualization.

    Always autoscales - no centering, no threshold.

    Args:
        y_coords: Y coordinates (km)
        z_coords: Z coordinates (km)
        velocity_slice: 2D velocity array (nz, ny)
        x_value: X coordinate value (km)
        output_path: Output file path
        config: Visualization configuration
        title_prefix: Prefix for title
    """
    fig, ax = plt.subplots(
        figsize=config["figure"]["figsize"],
        dpi=config["figure"]["dpi"],
    )

    Y, Z = np.meshgrid(y_coords, z_coords)

    # Always autoscale from local slice data
    vmin = np.nanmin(velocity_slice)
    vmax = np.nanmax(velocity_slice)

    cmap = config.get("colormap", "seismic")

    im = ax.pcolormesh(Y, Z, velocity_slice, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')

    # Invert y-axis so depth increases downward
    ax.invert_yaxis()

    if config.get("show_colorbar", True):
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Velocity (km/s)", fontsize=config["font_size"]["colorbar"])

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


def visualize_velocity_model(
    df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
    model_name: str,
    basin_data: Optional[Dict] = None,
    basin_config: Optional[Dict] = None,
) -> None:
    """
    Main function to visualize a velocity model.

    Args:
        df: DataFrame with X_grid_km, Y_grid_km, Z_grid_km, vp/vs
        output_dir: Output directory
        config: Visualization config (from visualize_input/checkerboard/inverted section)
        model_name: Name for output subdirectory
        basin_data: Optional basin boundary data from load_basin_boundaries()
        basin_config: Optional basin boundaries configuration for styling
    """
    logger.info(f"Visualizing {model_name}")

    # Create output subdirectory
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Handle "both" option
    velocity_field = config.get("velocity_field", "vp")
    if velocity_field == "both":
        velocity_fields = ["vp", "vs"]
    else:
        velocity_fields = [velocity_field]

    for vel_field in velocity_fields:
        logger.info(f"  Processing velocity field: {vel_field}")

        # Prepare 3D grid
        x_coords, y_coords, z_coords, velocity_3d = prepare_3d_grid(df, vel_field)
        nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)

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
            logger.info(f"    Creating XY (depth) slices for {vel_field}")

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
                for k in tqdm(z_indices, desc=f"    {model_name} {vel_field} XY slices"):
                    if k >= nz:
                        continue

                    velocity_slice = velocity_3d[k, :, :]
                    z_value = z_coords[k]

                    output_path = model_dir / f"{vel_field}_xy_z{k:03d}.{config['figure']['format']}"

                    plot_model_xy_slice(
                        x_coords, y_coords, velocity_slice, z_value,
                        output_path, config,
                        title_prefix=f"{model_name.replace('_', ' ').title()} {vel_field.upper()} - ",
                        basin_data=basin_data,
                        basin_config=basin_config,
                    )

        # Plot XZ slices (N-S vertical)
        if "xz" in planes:
            logger.info(f"    Creating XZ (N-S vertical) slices for {vel_field}")

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
                for j in tqdm(y_indices, desc=f"    {model_name} {vel_field} XZ slices"):
                    if j >= ny:
                        continue

                    velocity_slice = velocity_3d[:, j, :]
                    y_value = y_coords[j]

                    output_path = model_dir / f"{vel_field}_xz_y{j:03d}.{config['figure']['format']}"

                    plot_model_xz_slice(
                        x_coords, z_coords, velocity_slice, y_value,
                        output_path, config,
                        title_prefix=f"{model_name.replace('_', ' ').title()} {vel_field.upper()} - ",
                    )

        # Plot YZ slices (E-W vertical)
        if "yz" in planes:
            logger.info(f"    Creating YZ (E-W vertical) slices for {vel_field}")

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
                for i in tqdm(x_indices, desc=f"    {model_name} {vel_field} YZ slices"):
                    if i >= nx:
                        continue

                    velocity_slice = velocity_3d[:, :, i]
                    x_value = x_coords[i]

                    output_path = model_dir / f"{vel_field}_yz_x{i:03d}.{config['figure']['format']}"

                    plot_model_yz_slice(
                        y_coords, z_coords, velocity_slice, x_value,
                        output_path, config,
                        title_prefix=f"{model_name.replace('_', ' ').title()} {vel_field.upper()} - ",
                    )

    logger.info(f"  Visualization complete: {model_dir}")
