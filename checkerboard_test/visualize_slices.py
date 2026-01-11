"""
Visualize velocity model slices in different planes.

This module creates comprehensive visualizations of velocity models,
showing slices in XY (depth), XZ (N-S vertical), and YZ (E-W vertical) planes.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from utils.velocity_io import read_velocity_file

logger = logging.getLogger(__name__)


def prepare_3d_grid(
    df: pd.DataFrame,
    velocity_field: str = "vp",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare 3D grid from DataFrame.

    Args:
        df: DataFrame with X_grid_km, Y_grid_km, Z_grid_km and velocity field
        velocity_field: "vp" or "vs"

    Returns:
        (x_coords, y_coords, z_coords, velocity_3d)
    """
    # Get unique coordinates (sorted)
    x_coords = np.sort(df["X_grid_km"].unique())
    y_coords = np.sort(df["Y_grid_km"].unique())
    z_coords = np.sort(df["Z_grid_km"].unique())

    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)

    logger.debug(f"Grid dimensions: {nx} × {ny} × {nz}")

    # Create 3D velocity array
    velocity_3d = np.full((nz, ny, nx), np.nan)

    # Fill the array using vectorized operations (much faster than iterrows)
    # Compute all indices at once using numpy searchsorted
    x_indices = np.searchsorted(x_coords, df["X_grid_km"].values)
    y_indices = np.searchsorted(y_coords, df["Y_grid_km"].values)
    z_indices = np.searchsorted(z_coords, df["Z_grid_km"].values)

    # Get velocity values
    velocity_values = df[velocity_field].values

    # Fill the grid using advanced indexing (vectorized assignment)
    velocity_3d[z_indices, y_indices, x_indices] = velocity_values

    logger.debug(f"Filled {len(df)} grid points")

    return x_coords, y_coords, z_coords, velocity_3d


def km_to_grid_index(
    km_value: float,
    coords: np.ndarray,
    tolerance: float = 1e-6,
) -> Optional[int]:
    """
    Convert a physical coordinate (km) to grid index.

    Args:
        km_value: Physical coordinate value in km
        coords: Array of coordinate values (km) for the axis
        tolerance: Tolerance for floating point comparison (km)

    Returns:
        Grid index if km_value falls exactly on a grid point, None otherwise
    """
    # Find the closest coordinate
    distances = np.abs(coords - km_value)
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]

    # Check if it falls exactly on a grid point (within tolerance)
    if min_distance <= tolerance:
        return int(min_idx)
    else:
        logger.debug(
            f"Value {km_value:.3f} km does not fall on grid points. "
            f"Closest grid point is {coords[min_idx]:.3f} km "
            f"(distance: {min_distance:.6f} km)"
        )
        return None


def convert_km_spacing_to_indices(
    spacing_km: float,
    coords: np.ndarray,
    tolerance: float = 1e-6,
) -> List[int]:
    """
    Convert a spacing in km to a list of valid grid indices.

    Args:
        spacing_km: Spacing between slices in km
        coords: Array of coordinate values (km) for the axis
        tolerance: Tolerance for floating point comparison (km)

    Returns:
        List of valid grid indices that fall on grid points
    """
    if spacing_km <= 0:
        logger.warning(f"Invalid spacing {spacing_km} km, using all grid points")
        return list(range(len(coords)))

    # Start from the first coordinate and increment by spacing_km
    valid_indices = []
    current_km = coords[0]

    while current_km <= coords[-1]:
        idx = km_to_grid_index(current_km, coords, tolerance)
        if idx is not None:
            valid_indices.append(idx)
        current_km += spacing_km

    return valid_indices


def parse_slice_specification(
    slice_spec: Optional[List],
    coords: np.ndarray,
    unit: str = "indices",
    tolerance: float = 1e-6,
) -> List[int]:
    """
    Parse slice specification and return list of valid grid indices.

    Args:
        slice_spec: List of slice positions (either indices or km values) or None
        coords: Array of coordinate values (km) for the axis
        unit: "indices" or "km" to specify the unit of slice_spec
        tolerance: Tolerance for floating point comparison (km)

    Returns:
        List of valid grid indices
    """
    if slice_spec is None:
        return None

    if unit == "indices":
        # Already in grid indices, return as-is (filter out invalid indices)
        return [int(i) for i in slice_spec if 0 <= i < len(coords)]
    elif unit == "km":
        # Convert km values to grid indices
        valid_indices = []
        for km_value in slice_spec:
            idx = km_to_grid_index(km_value, coords, tolerance)
            if idx is not None:
                valid_indices.append(idx)
            else:
                logger.warning(
                    f"Skipping slice at {km_value:.3f} km as it does not fall on a grid point"
                )
        return valid_indices
    else:
        logger.error(f"Invalid unit '{unit}', expected 'indices' or 'km'")
        return []


def plot_xy_slice(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    velocity_slice: np.ndarray,
    z_value: float,
    output_path: Path,
    config: Dict,
    title_prefix: str = "",
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
    """
    fig, ax = plt.subplots(
        figsize=config["figure"]["figsize"],
        dpi=config["figure"]["dpi"],
    )

    # Create meshgrid for pcolormesh
    X, Y = np.meshgrid(x_coords, y_coords)

    # Determine color limits
    if config.get("symmetric_colorbar", False):
        vmax = max(abs(np.nanmin(velocity_slice)), abs(np.nanmax(velocity_slice)))
        vmin = -vmax
    else:
        vmin = config.get("vmin", np.nanmin(velocity_slice))
        vmax = config.get("vmax", np.nanmax(velocity_slice))

    # Plot
    im = ax.pcolormesh(
        X, Y, velocity_slice,
        cmap=config["colormap"],
        vmin=vmin,
        vmax=vmax,
        shading='auto',
    )

    # Colorbar
    if config["show_colorbar"]:
        cbar = plt.colorbar(im, ax=ax)
        # Determine label based on plot type
        if "pattern" in title_prefix.lower() or "pct" in title_prefix.lower():
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
    """
    fig, ax = plt.subplots(
        figsize=config["figure"]["figsize"],
        dpi=config["figure"]["dpi"],
    )

    # Create meshgrid for pcolormesh
    X, Z = np.meshgrid(x_coords, z_coords)

    # Determine color limits
    if config.get("symmetric_colorbar", False):
        vmax = max(abs(np.nanmin(velocity_slice)), abs(np.nanmax(velocity_slice)))
        vmin = -vmax
    else:
        vmin = config.get("vmin", np.nanmin(velocity_slice))
        vmax = config.get("vmax", np.nanmax(velocity_slice))

    # Plot
    im = ax.pcolormesh(
        X, Z, velocity_slice,
        cmap=config["colormap"],
        vmin=vmin,
        vmax=vmax,
        shading='auto',
    )

    # Colorbar
    if config["show_colorbar"]:
        cbar = plt.colorbar(im, ax=ax)
        # Determine label based on plot type
        if "pattern" in title_prefix.lower() or "pct" in title_prefix.lower():
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
    """
    fig, ax = plt.subplots(
        figsize=config["figure"]["figsize"],
        dpi=config["figure"]["dpi"],
    )

    # Create meshgrid for pcolormesh
    Y, Z = np.meshgrid(y_coords, z_coords)

    # Determine color limits
    if config.get("symmetric_colorbar", False):
        vmax = max(abs(np.nanmin(velocity_slice)), abs(np.nanmax(velocity_slice)))
        vmin = -vmax
    else:
        vmin = config.get("vmin", np.nanmin(velocity_slice))
        vmax = config.get("vmax", np.nanmax(velocity_slice))

    # Plot
    im = ax.pcolormesh(
        Y, Z, velocity_slice,
        cmap=config["colormap"],
        vmin=vmin,
        vmax=vmax,
        shading='auto',
    )

    # Colorbar
    if config["show_colorbar"]:
        cbar = plt.colorbar(im, ax=ax)
        # Determine label based on plot type
        if "pattern" in title_prefix.lower() or "pct" in title_prefix.lower():
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


def visualize_model(
    df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
    model_name: str = "model",
) -> None:
    """
    Visualize velocity model with all configured slices.

    Args:
        df: DataFrame with velocity model
        output_dir: Output directory for plots
        config: Visualization configuration
        model_name: Name for output files
    """
    logger.info(f"Visualizing {model_name}")

    # Create output subdirectory
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Handle "both" option - visualize both vp and vs
    velocity_field = config["velocity_field"]
    if velocity_field == "both":
        velocity_fields = ["vp", "vs"]
    else:
        velocity_fields = [velocity_field]

    for vel_field in velocity_fields:
        logger.info(f"  Processing velocity field: {vel_field}")

        # Prepare 3D grid
        x_coords, y_coords, z_coords, velocity_3d_kms = prepare_3d_grid(df, vel_field)

        nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)

        # Get slice configuration
        planes = config["planes"]
        xy_slices = config.get("xy_slices")
        xz_slices = config.get("xz_slices")
        yz_slices = config.get("yz_slices")
        slice_spacing = config.get("slice_spacing", {})

        # Get unit specifications (default to "indices" for backward compatibility)
        slice_unit = config.get("slice_unit", "indices")  # "indices" or "km"
        spacing_unit = config.get("spacing_unit", "indices")  # "indices" or "km"
        tolerance = config.get("tolerance_km", 1e-6)  # Tolerance for km-to-index conversion

        # Plot XY slices (depth slices)
        if "xy" in planes:
            logger.info(f"    Creating XY (depth) slices for {vel_field}")

            # Determine which slices to plot
            z_indices = None
            if xy_slices is not None:
                # Parse user-specified slices (could be indices or km)
                z_indices = parse_slice_specification(xy_slices, z_coords, slice_unit, tolerance)
                if not z_indices:
                    logger.warning("    No valid XY slices found, skipping")
            else:
                # Use spacing to determine slices
                if spacing_unit == "km":
                    spacing_km = slice_spacing.get("xy", 1.0)
                    z_indices = convert_km_spacing_to_indices(spacing_km, z_coords, tolerance)
                    if not z_indices:
                        logger.warning(f"    No valid grid points found for spacing {spacing_km} km, skipping XY slices")
                else:
                    spacing = slice_spacing.get("xy", 1)
                    z_indices = range(0, nz, spacing)

            if z_indices:
                for k in tqdm(z_indices, desc=f"    {model_name} {vel_field} XY slices"):
                    if k >= nz:
                        continue

                    velocity_slice = velocity_3d_kms[k, :, :]
                    z_value = z_coords[k]

                    output_path = model_dir / f"{vel_field}_xy_z{k:03d}.{config['figure']['format']}"

                    plot_xy_slice(
                        x_coords, y_coords, velocity_slice, z_value,
                        output_path, config, title_prefix=f"{model_name.title()} {vel_field.upper()} - "
                    )

        # Plot XZ slices (N-S vertical)
        if "xz" in planes:
            logger.info(f"    Creating XZ (N-S vertical) slices for {vel_field}")

            # Determine which slices to plot
            y_indices = None
            if xz_slices is not None:
                # Parse user-specified slices (could be indices or km)
                y_indices = parse_slice_specification(xz_slices, y_coords, slice_unit, tolerance)
                if not y_indices:
                    logger.warning("    No valid XZ slices found, skipping")
            else:
                # Use spacing to determine slices
                if spacing_unit == "km":
                    spacing_km = slice_spacing.get("xz", 1.0)
                    y_indices = convert_km_spacing_to_indices(spacing_km, y_coords, tolerance)
                    if not y_indices:
                        logger.warning(f"    No valid grid points found for spacing {spacing_km} km, skipping XZ slices")
                else:
                    spacing = slice_spacing.get("xz", 1)
                    y_indices = range(0, ny, spacing)

            if y_indices:
                for j in tqdm(y_indices, desc=f"    {model_name} {vel_field} XZ slices"):
                    if j >= ny:
                        continue

                    velocity_slice = velocity_3d_kms[:, j, :]
                    y_value = y_coords[j]

                    output_path = model_dir / f"{vel_field}_xz_y{j:03d}.{config['figure']['format']}"

                    plot_xz_slice(
                        x_coords, z_coords, velocity_slice, y_value,
                        output_path, config, title_prefix=f"{model_name.title()} {vel_field.upper()} - "
                    )

        # Plot YZ slices (E-W vertical)
        if "yz" in planes:
            logger.info(f"    Creating YZ (E-W vertical) slices for {vel_field}")

            # Determine which slices to plot
            x_indices = None
            if yz_slices is not None:
                # Parse user-specified slices (could be indices or km)
                x_indices = parse_slice_specification(yz_slices, x_coords, slice_unit, tolerance)
                if not x_indices:
                    logger.warning("    No valid YZ slices found, skipping")
            else:
                # Use spacing to determine slices
                if spacing_unit == "km":
                    spacing_km = slice_spacing.get("yz", 1.0)
                    x_indices = convert_km_spacing_to_indices(spacing_km, x_coords, tolerance)
                    if not x_indices:
                        logger.warning(f"    No valid grid points found for spacing {spacing_km} km, skipping YZ slices")
                else:
                    spacing = slice_spacing.get("yz", 1)
                    x_indices = range(0, nx, spacing)

            if x_indices:
                for i in tqdm(x_indices, desc=f"    {model_name} {vel_field} YZ slices"):
                    if i >= nx:
                        continue

                    velocity_slice = velocity_3d_kms[:, :, i]
                    x_value = x_coords[i]

                    output_path = model_dir / f"{vel_field}_yz_x{i:03d}.{config['figure']['format']}"

                    plot_yz_slice(
                        y_coords, z_coords, velocity_slice, x_value,
                        output_path, config, title_prefix=f"{model_name.title()} {vel_field.upper()} - "
                    )

    logger.info(f"  Visualization complete: {model_dir}")


def visualize_checkerboard_test(
    input_csv: Optional[Path],
    checkerboard_csv: Optional[Path],
    inverted_csv: Optional[Path],
    output_dir: Path,
    config: Dict,
) -> None:
    """
    Main function to create all checkerboard test visualizations.

    Args:
        input_csv: Input (original) velocity model CSV
        checkerboard_csv: Checkerboard perturbed model CSV
        inverted_csv: Inverted model CSV
        output_dir: Output directory
        config: Visualization configuration
    """
    logger.info("=" * 80)
    logger.info("STEP 4: Creating Visualizations")
    logger.info("=" * 80)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize input model
    if config.get("visualize_input", False) and input_csv is not None:
        logger.info("Visualizing input model")
        df_input = read_velocity_file(input_csv, show_progress=False)
        visualize_model(df_input, output_dir, config, "input")
        del df_input  # Free memory

    # Visualize checkerboard model
    if config.get("visualize_checkerboard", False) and checkerboard_csv is not None:
        logger.info("Visualizing checkerboard model")
        df_checkerboard = read_velocity_file(checkerboard_csv, show_progress=False)
        visualize_model(df_checkerboard, output_dir, config, "checkerboard")
        del df_checkerboard  # Free memory

    # Visualize inverted model
    if config.get("visualize_inverted", False) and inverted_csv is not None:
        logger.info("Visualizing inverted model")
        df_inverted = read_velocity_file(inverted_csv, show_progress=False)
        visualize_model(df_inverted, output_dir, config, "inverted")
        del df_inverted  # Free memory

    # Visualize difference (inverted - checkerboard)
    if config.get("visualize_difference", False) and inverted_csv is not None and checkerboard_csv is not None:
        logger.info("Visualizing difference (inverted - checkerboard)")

        df_inverted = read_velocity_file(inverted_csv, show_progress=False)
        df_checkerboard = read_velocity_file(checkerboard_csv, show_progress=False)

        # Handle "both" option - process both vp and vs
        velocity_field = config["velocity_field"]
        if velocity_field == "both":
            velocity_fields = ["vp", "vs"]
        else:
            velocity_fields = [velocity_field]

        for vel_field in velocity_fields:
            logger.info(f"  Processing difference for {vel_field}")

            # Calculate difference - avoid copy() for memory efficiency
            # Create new DataFrame with only necessary columns
            df_diff = pd.DataFrame({
                'X_grid_km': df_inverted['X_grid_km'].values,
                'Y_grid_km': df_inverted['Y_grid_km'].values,
                'Z_grid_km': df_inverted['Z_grid_km'].values,
                vel_field: df_inverted[vel_field].values - df_checkerboard[vel_field].values
            })

            # Use difference-specific color settings
            diff_config = config.copy()
            diff_config["velocity_field"] = vel_field  # Override to process single field
            diff_config["vmin"] = config.get("diff_vmin", -0.5)  # km/s
            diff_config["vmax"] = config.get("diff_vmax", 0.5)  # km/s
            diff_config["symmetric_colorbar"] = config.get("symmetric_colorbar", True)

            visualize_model(df_diff, output_dir, diff_config, f"difference_{vel_field}" if velocity_field == "both" else "difference")

            # Free memory
            del df_diff

        # Free memory
        del df_inverted, df_checkerboard

    # Visualize checkerboard pattern only (checkerboard - input) as percentage perturbation
    if config.get("visualize_checkerboard_pattern", False) and checkerboard_csv is not None and input_csv is not None:
        logger.info("Visualizing checkerboard pattern as percentage perturbation")

        df_checkerboard = read_velocity_file(checkerboard_csv, show_progress=False)
        df_input = read_velocity_file(input_csv, show_progress=False)

        # Handle "both" option - process both vp and vs
        velocity_field = config["velocity_field"]
        if velocity_field == "both":
            velocity_fields = ["vp", "vs"]
        else:
            velocity_fields = [velocity_field]

        for vel_field in velocity_fields:
            logger.info(f"  Processing checkerboard pattern for {vel_field}")

            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                perturbation_pct = ((df_checkerboard[vel_field].values - df_input[vel_field].values)
                                   / df_input[vel_field].values) * 100.0
                # Replace inf/nan with 0
                perturbation_pct = np.where(np.isfinite(perturbation_pct), perturbation_pct, 0.0)

            # Create new DataFrame with only necessary columns - avoid copy()
            df_pattern = pd.DataFrame({
                'X_grid_km': df_checkerboard['X_grid_km'].values,
                'Y_grid_km': df_checkerboard['Y_grid_km'].values,
                'Z_grid_km': df_checkerboard['Z_grid_km'].values,
                vel_field: perturbation_pct
            })

            # Use pattern-specific color settings (now in percentage)
            pattern_config = config.copy()
            pattern_config["velocity_field"] = vel_field  # Override to process single field
            pattern_config["vmin"] = config.get("pattern_vmin", -10.0)  # percentage
            pattern_config["vmax"] = config.get("pattern_vmax", 10.0)   # percentage
            pattern_config["colormap"] = config.get("pattern_colormap", "seismic")
            pattern_config["symmetric_colorbar"] = config.get("pattern_symmetric_colorbar", True)

            visualize_model(df_pattern, output_dir, pattern_config, f"checkerboard_pattern_pct_{vel_field}" if velocity_field == "both" else "checkerboard_pattern_pct")

            # Free memory
            del df_pattern

        # Free memory
        del df_checkerboard, df_input

    # Visualize inverted pattern only (inverted - input) as percentage perturbation
    if config.get("visualize_inverted_pattern", False) and inverted_csv is not None and input_csv is not None:
        logger.info("Visualizing inverted pattern as percentage perturbation")

        df_inverted = read_velocity_file(inverted_csv, show_progress=False)
        df_input = read_velocity_file(input_csv, show_progress=False)

        # Handle "both" option - process both vp and vs
        velocity_field = config["velocity_field"]
        if velocity_field == "both":
            velocity_fields = ["vp", "vs"]
        else:
            velocity_fields = [velocity_field]

        for vel_field in velocity_fields:
            logger.info(f"  Processing inverted pattern for {vel_field}")

            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                perturbation_pct = ((df_inverted[vel_field].values - df_input[vel_field].values)
                                   / df_input[vel_field].values) * 100.0
                # Replace inf/nan with 0
                perturbation_pct = np.where(np.isfinite(perturbation_pct), perturbation_pct, 0.0)

            # Create new DataFrame with only necessary columns - avoid copy()
            df_pattern = pd.DataFrame({
                'X_grid_km': df_inverted['X_grid_km'].values,
                'Y_grid_km': df_inverted['Y_grid_km'].values,
                'Z_grid_km': df_inverted['Z_grid_km'].values,
                vel_field: perturbation_pct
            })

            # Use pattern-specific color settings (now in percentage)
            pattern_config = config.copy()
            pattern_config["velocity_field"] = vel_field  # Override to process single field
            pattern_config["vmin"] = config.get("pattern_vmin", -10.0)  # percentage
            pattern_config["vmax"] = config.get("pattern_vmax", 10.0)   # percentage
            pattern_config["colormap"] = config.get("pattern_colormap", "seismic")
            pattern_config["symmetric_colorbar"] = config.get("pattern_symmetric_colorbar", True)

            visualize_model(df_pattern, output_dir, pattern_config, f"inverted_pattern_pct_{vel_field}" if velocity_field == "both" else "inverted_pattern_pct")

            # Free memory
            del df_pattern

        # Free memory
        del df_inverted, df_input

    logger.info("All visualizations completed successfully!")


if __name__ == "__main__":
    # Example standalone usage
    import yaml

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        full_config = yaml.safe_load(f)

    # Extract paths
    base_dir = Path(full_config["general"]["output_dir"])
    input_csv = Path(full_config["input"]["velocity_model_csv"])
    checkerboard_csv = base_dir / "checkerboard_model" / "velocity_model_checkerboard.csv"
    inverted_csv = base_dir / "inverted_model" / "velocity_model_inverted.csv"
    output_dir = base_dir / "visualizations"

    # Run
    visualize_checkerboard_test(
        input_csv,
        checkerboard_csv,
        inverted_csv,
        output_dir,
        full_config["visualization"],
    )
