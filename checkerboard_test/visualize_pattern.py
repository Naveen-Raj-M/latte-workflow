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

logger = logging.getLogger(__name__)


def compute_global_limits(velocity_3d: np.ndarray) -> Tuple[float, float]:
    """Compute global min/max across entire 3D volume."""
    return float(np.nanmin(velocity_3d)), float(np.nanmax(velocity_3d))


def create_centered_norm(vmin: float, vmax: float) -> mcolors.TwoSlopeNorm:
    """
    Create a TwoSlopeNorm centered at zero.

    This centers the colormap at zero WITHOUT forcing symmetric +/- values.
    Example: data range [-5%, +12%] will show full range with neutral color at 0.

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


def create_threshold_colormap(
    base_cmap_name: str,
    vmin: float,
    vmax: float,
    threshold: float,
    n_samples: int = 50,
) -> Tuple[mcolors.LinearSegmentedColormap, mcolors.Normalize]:
    """
    Create colormap with black regions for values exceeding threshold.

    The colormap maps:
    - [-threshold, +threshold] -> base colormap (centered at 0)
    - Values outside this range -> black

    The colorbar shows actual vmin to vmax with black in regions beyond threshold.

    Args:
        base_cmap_name: Name of the base matplotlib colormap
        vmin: Global minimum value
        vmax: Global maximum value
        threshold: Threshold value for black regions
        n_samples: Number of color samples for smooth gradient

    Returns:
        (colormap, norm) tuple
    """
    base_cmap = plt.get_cmap(base_cmap_name)

    # Calculate normalized positions for the boundaries
    total_range = vmax - vmin
    if total_range <= 0:
        total_range = 1e-6

    # Position in [0, 1] normalized space
    pos_neg_thresh = max(0.0, (-threshold - vmin) / total_range)
    pos_pos_thresh = min(1.0, (threshold - vmin) / total_range)

    black = (0.0, 0.0, 0.0)

    # Build the colormap segments
    cdict = {'red': [], 'green': [], 'blue': []}

    for channel_idx, channel in enumerate(['red', 'green', 'blue']):
        segments = []

        # Black region from vmin to -threshold (if applicable)
        if vmin < -threshold:
            segments.append((0.0, black[channel_idx], black[channel_idx]))
            segments.append((pos_neg_thresh, black[channel_idx], black[channel_idx]))

        # Gradient region from -threshold to +threshold
        # Sample the base colormap and map to the threshold region
        for i in range(n_samples):
            # Position in base colormap [0, 1]
            t = i / (n_samples - 1) if n_samples > 1 else 0.5
            color = base_cmap(t)

            # Position in final colormap
            if vmin < -threshold and vmax > threshold:
                # Both black regions exist
                pos = pos_neg_thresh + t * (pos_pos_thresh - pos_neg_thresh)
            elif vmin < -threshold:
                # Only lower black region
                pos = pos_neg_thresh + t * (1.0 - pos_neg_thresh)
            elif vmax > threshold:
                # Only upper black region
                pos = t * pos_pos_thresh
            else:
                # No black regions
                pos = t

            segments.append((pos, color[channel_idx], color[channel_idx]))

        # Black region from +threshold to vmax (if applicable)
        if vmax > threshold:
            segments.append((pos_pos_thresh, black[channel_idx], black[channel_idx]))
            segments.append((1.0, black[channel_idx], black[channel_idx]))

        cdict[channel] = segments

    # Create the colormap
    custom_cmap = mcolors.LinearSegmentedColormap(f'{base_cmap_name}_threshold', cdict)

    # Use standard normalization with vmin/vmax
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    return custom_cmap, norm


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

    # Build plot kwargs
    plot_kwargs = {'shading': 'auto'}

    if threshold is not None:
        # Threshold mode: cmap scaled to +/-threshold, black beyond
        cmap, norm = create_threshold_colormap(cmap_name, vmin, vmax, threshold)
        plot_kwargs['cmap'] = cmap
        plot_kwargs['norm'] = norm
    elif centered:
        # Centered at zero mode (no threshold)
        cmap = plt.get_cmap(cmap_name)
        norm = create_centered_norm(vmin, vmax)
        plot_kwargs['cmap'] = cmap
        plot_kwargs['norm'] = norm
    else:
        # Simple autoscale
        plot_kwargs['cmap'] = cmap_name
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax

    im = ax.pcolormesh(X, Y, velocity_slice, **plot_kwargs)

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

    if config.get("show_colorbar", True):
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Velocity Perturbation (%)", fontsize=config["font_size"]["colorbar"])

    ax.set_xlabel("X (km)", fontsize=config["font_size"]["labels"])
    ax.set_ylabel("Y (km)", fontsize=config["font_size"]["labels"])

    if config.get("show_title", True):
        ax.set_title(f"{title_prefix}Depth Slice at Z = {z_value:.1f} km",
                     fontsize=config["font_size"]["title"])

    if config.get("show_grid", True):
        ax.grid(True, alpha=0.3)

    ax.set_aspect('equal', adjustable='box')
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

    plot_kwargs = {'shading': 'auto'}

    if threshold is not None:
        cmap, norm = create_threshold_colormap(cmap_name, vmin, vmax, threshold)
        plot_kwargs['cmap'] = cmap
        plot_kwargs['norm'] = norm
    elif centered:
        cmap = plt.get_cmap(cmap_name)
        norm = create_centered_norm(vmin, vmax)
        plot_kwargs['cmap'] = cmap
        plot_kwargs['norm'] = norm
    else:
        plot_kwargs['cmap'] = cmap_name
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax

    im = ax.pcolormesh(X, Z, velocity_slice, **plot_kwargs)

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

    if config.get("show_colorbar", True):
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Velocity Perturbation (%)", fontsize=config["font_size"]["colorbar"])

    ax.set_xlabel("X (km)", fontsize=config["font_size"]["labels"])
    ax.set_ylabel("Z (km)", fontsize=config["font_size"]["labels"])

    if config.get("show_title", True):
        ax.set_title(f"{title_prefix}N-S Slice at Y = {y_value:.1f} km",
                     fontsize=config["font_size"]["title"])

    if config.get("show_grid", True):
        ax.grid(True, alpha=0.3)

    ax.set_aspect('equal', adjustable='box')
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

    plot_kwargs = {'shading': 'auto'}

    if threshold is not None:
        cmap, norm = create_threshold_colormap(cmap_name, vmin, vmax, threshold)
        plot_kwargs['cmap'] = cmap
        plot_kwargs['norm'] = norm
    elif centered:
        cmap = plt.get_cmap(cmap_name)
        norm = create_centered_norm(vmin, vmax)
        plot_kwargs['cmap'] = cmap
        plot_kwargs['norm'] = norm
    else:
        plot_kwargs['cmap'] = cmap_name
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax

    im = ax.pcolormesh(Y, Z, velocity_slice, **plot_kwargs)

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

    if config.get("show_colorbar", True):
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Velocity Perturbation (%)", fontsize=config["font_size"]["colorbar"])

    ax.set_xlabel("Y (km)", fontsize=config["font_size"]["labels"])
    ax.set_ylabel("Z (km)", fontsize=config["font_size"]["labels"])

    if config.get("show_title", True):
        ax.set_title(f"{title_prefix}E-W Slice at X = {x_value:.1f} km",
                     fontsize=config["font_size"]["title"])

    if config.get("show_grid", True):
        ax.grid(True, alpha=0.3)

    ax.set_aspect('equal', adjustable='box')
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
