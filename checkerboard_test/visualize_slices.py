"""
Visualize velocity model slices in different planes.

This module creates comprehensive visualizations of velocity models,
showing slices in XY (depth), XZ (N-S vertical), and YZ (E-W vertical) planes.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from utils.velocity_io import read_velocity_file
from checkerboard_test.grid_utils import (
    prepare_3d_grid,
    parse_slice_specification,
    convert_km_spacing_to_indices,
)
from checkerboard_test.colormap_utils import compute_uniform_colorbar_limits
from checkerboard_test.plotting_utils import plot_xy_slice, plot_xz_slice, plot_yz_slice
from checkerboard_test.recovery_analysis import compute_fair_recovery_mask, compute_recovery_metrics

logger = logging.getLogger(__name__)


def visualize_model(
    df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
    model_name: str = "model",
    checkerboard_pattern_3d: Optional[np.ndarray] = None,
    coords_tuple: Optional[Tuple] = None,
) -> None:
    """
    Visualize velocity model with all configured slices.

    Args:
        df: DataFrame with velocity model
        output_dir: Output directory for plots
        config: Visualization configuration
        model_name: Name for output files
        checkerboard_pattern_3d: Optional 3D checkerboard pattern for recovery analysis (% perturbation)
        coords_tuple: Optional (x_coords, y_coords, z_coords) if already computed
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

        # Compute uniform colorbar limits if enabled
        if config.get("use_uniform_colorbar", False):
            vmin, vmax = compute_uniform_colorbar_limits(
                velocity_3d_kms,
                symmetric=config.get("symmetric_colorbar", False),
                vmin=config.get("vmin"),
                vmax=config.get("vmax"),
            )
            logger.info(f"    Using uniform colorbar: [{vmin:.4f}, {vmax:.4f}]")
        else:
            # Use per-slice auto-scaling or config values
            vmin = config.get("vmin")
            vmax = config.get("vmax")

        # Update config with computed limits
        plot_config = config.copy()
        plot_config["vmin"] = vmin
        plot_config["vmax"] = vmax

        # Compute fair recovery mask if applicable
        recovery_3d = None
        is_inverted_pattern = "inverted_pattern" in model_name
        if (is_inverted_pattern and
            checkerboard_pattern_3d is not None and
            config.get("show_recovery_contours", False)):

            recovery_threshold = config.get("recovery_threshold_pct", 0.0)
            logger.info(f"    Computing fair recovery mask with threshold {recovery_threshold}%")

            # The velocity_3d_kms for inverted pattern is in % perturbation
            # checkerboard_pattern_3d is also in % perturbation
            recovery_3d = compute_fair_recovery_mask(
                checkerboard_pattern_3d,
                velocity_3d_kms,
                recovery_threshold,
            )

            # Compute and log recovery metrics
            metrics = compute_recovery_metrics(checkerboard_pattern_3d, velocity_3d_kms)
            logger.info(f"    Recovery metrics:")
            logger.info(f"      Correlation: {metrics['correlation']:.4f}")
            logger.info(f"      RMSE: {metrics['rmse']:.4f}%")
            logger.info(f"      Mean Absolute Error: {metrics['mean_absolute_error']:.4f}%")
            logger.info(f"      Sign Agreement: {metrics['sign_agreement']:.2f}%")

        # Get slice configuration
        planes = config["planes"]
        xy_slices = config.get("xy_slices")
        xz_slices = config.get("xz_slices")
        yz_slices = config.get("yz_slices")
        slice_spacing = config.get("slice_spacing", {})

        # Get unit specifications (default to "indices" for backward compatibility)
        slice_unit = config.get("slice_unit", "indices")
        spacing_unit = config.get("spacing_unit", "indices")
        tolerance = config.get("tolerance_km", 1e-6)

        # Plot XY slices (depth slices)
        if "xy" in planes:
            logger.info(f"    Creating XY (depth) slices for {vel_field}")

            # Determine which slices to plot
            z_indices = None
            if xy_slices is not None:
                z_indices = parse_slice_specification(xy_slices, z_coords, slice_unit, tolerance)
                if not z_indices:
                    logger.warning("    No valid XY slices found, skipping")
            else:
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
                    recovery_mask = recovery_3d[k, :, :] if recovery_3d is not None else None

                    output_path = model_dir / f"{vel_field}_xy_z{k:03d}.{plot_config['figure']['format']}"

                    plot_xy_slice(
                        x_coords, y_coords, velocity_slice, z_value,
                        output_path, plot_config,
                        title_prefix=f"{model_name.title()} {vel_field.upper()} - ",
                        recovery_mask=recovery_mask,
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
                    recovery_mask = recovery_3d[:, j, :] if recovery_3d is not None else None

                    output_path = model_dir / f"{vel_field}_xz_y{j:03d}.{plot_config['figure']['format']}"

                    plot_xz_slice(
                        x_coords, z_coords, velocity_slice, y_value,
                        output_path, plot_config,
                        title_prefix=f"{model_name.title()} {vel_field.upper()} - ",
                        recovery_mask=recovery_mask,
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
                    recovery_mask = recovery_3d[:, :, i] if recovery_3d is not None else None

                    output_path = model_dir / f"{vel_field}_yz_x{i:03d}.{plot_config['figure']['format']}"

                    plot_yz_slice(
                        y_coords, z_coords, velocity_slice, x_value,
                        output_path, plot_config,
                        title_prefix=f"{model_name.title()} {vel_field.upper()} - ",
                        recovery_mask=recovery_mask,
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
        del df_input

    # Visualize checkerboard model
    if config.get("visualize_checkerboard", False) and checkerboard_csv is not None:
        logger.info("Visualizing checkerboard model")
        df_checkerboard = read_velocity_file(checkerboard_csv, show_progress=False)
        visualize_model(df_checkerboard, output_dir, config, "checkerboard")
        del df_checkerboard

    # Visualize inverted model
    if config.get("visualize_inverted", False) and inverted_csv is not None:
        logger.info("Visualizing inverted model")
        df_inverted = read_velocity_file(inverted_csv, show_progress=False)
        visualize_model(df_inverted, output_dir, config, "inverted")
        del df_inverted

    # Visualize difference (inverted - checkerboard)
    if config.get("visualize_difference", False) and inverted_csv is not None and checkerboard_csv is not None:
        logger.info("Visualizing difference (inverted - checkerboard)")

        df_inverted = read_velocity_file(inverted_csv, show_progress=False)
        df_checkerboard = read_velocity_file(checkerboard_csv, show_progress=False)

        velocity_field = config["velocity_field"]
        if velocity_field == "both":
            velocity_fields = ["vp", "vs"]
        else:
            velocity_fields = [velocity_field]

        for vel_field in velocity_fields:
            logger.info(f"  Processing difference for {vel_field}")

            df_diff = pd.DataFrame({
                'X_grid_km': df_inverted['X_grid_km'].values,
                'Y_grid_km': df_inverted['Y_grid_km'].values,
                'Z_grid_km': df_inverted['Z_grid_km'].values,
                vel_field: df_inverted[vel_field].values - df_checkerboard[vel_field].values
            })

            diff_config = config.copy()
            diff_config["velocity_field"] = vel_field
            diff_config["vmin"] = config.get("diff_vmin", -0.5)
            diff_config["vmax"] = config.get("diff_vmax", 0.5)
            diff_config["symmetric_colorbar"] = config.get("symmetric_colorbar", True)

            visualize_model(df_diff, output_dir, diff_config,
                          f"difference_{vel_field}" if velocity_field == "both" else "difference")
            del df_diff

        del df_inverted, df_checkerboard

    # Visualize checkerboard pattern only (checkerboard - input) as percentage perturbation
    if config.get("visualize_checkerboard_pattern", False) and checkerboard_csv is not None and input_csv is not None:
        logger.info("Visualizing checkerboard pattern as percentage perturbation")

        df_checkerboard = read_velocity_file(checkerboard_csv, show_progress=False)
        df_input = read_velocity_file(input_csv, show_progress=False)

        velocity_field = config["velocity_field"]
        if velocity_field == "both":
            velocity_fields = ["vp", "vs"]
        else:
            velocity_fields = [velocity_field]

        for vel_field in velocity_fields:
            logger.info(f"  Processing checkerboard pattern for {vel_field}")

            with np.errstate(divide='ignore', invalid='ignore'):
                perturbation_pct = ((df_checkerboard[vel_field].values - df_input[vel_field].values)
                                   / df_input[vel_field].values) * 100.0
                perturbation_pct = np.where(np.isfinite(perturbation_pct), perturbation_pct, 0.0)

            df_pattern = pd.DataFrame({
                'X_grid_km': df_checkerboard['X_grid_km'].values,
                'Y_grid_km': df_checkerboard['Y_grid_km'].values,
                'Z_grid_km': df_checkerboard['Z_grid_km'].values,
                vel_field: perturbation_pct
            })

            pattern_config = config.copy()
            pattern_config["velocity_field"] = vel_field
            pattern_config["vmin"] = config.get("pattern_vmin", -10.0)
            pattern_config["vmax"] = config.get("pattern_vmax", 10.0)
            pattern_config["colormap"] = config.get("pattern_colormap", "seismic")
            pattern_config["symmetric_colorbar"] = config.get("pattern_symmetric_colorbar", True)

            visualize_model(df_pattern, output_dir, pattern_config,
                          f"checkerboard_pattern_pct_{vel_field}" if velocity_field == "both" else "checkerboard_pattern_pct")
            del df_pattern

        del df_checkerboard, df_input

    # Visualize inverted pattern only (inverted - input) as percentage perturbation with recovery analysis
    if config.get("visualize_inverted_pattern", False) and inverted_csv is not None and input_csv is not None:
        logger.info("Visualizing inverted pattern as percentage perturbation")

        df_inverted = read_velocity_file(inverted_csv, show_progress=False)
        df_input = read_velocity_file(input_csv, show_progress=False)

        # Also load checkerboard pattern for recovery analysis if needed
        checkerboard_pattern_dict = {}
        if config.get("show_recovery_contours", False) and checkerboard_csv is not None:
            df_checkerboard = read_velocity_file(checkerboard_csv, show_progress=False)

        velocity_field = config["velocity_field"]
        if velocity_field == "both":
            velocity_fields = ["vp", "vs"]
        else:
            velocity_fields = [velocity_field]

        for vel_field in velocity_fields:
            logger.info(f"  Processing inverted pattern for {vel_field}")

            with np.errstate(divide='ignore', invalid='ignore'):
                perturbation_pct = ((df_inverted[vel_field].values - df_input[vel_field].values)
                                   / df_input[vel_field].values) * 100.0
                perturbation_pct = np.where(np.isfinite(perturbation_pct), perturbation_pct, 0.0)

            df_pattern = pd.DataFrame({
                'X_grid_km': df_inverted['X_grid_km'].values,
                'Y_grid_km': df_inverted['Y_grid_km'].values,
                'Z_grid_km': df_inverted['Z_grid_km'].values,
                vel_field: perturbation_pct
            })

            pattern_config = config.copy()
            pattern_config["velocity_field"] = vel_field
            pattern_config["vmin"] = config.get("pattern_vmin", -10.0)
            pattern_config["vmax"] = config.get("pattern_vmax", 10.0)
            pattern_config["colormap"] = config.get("pattern_colormap", "seismic")
            pattern_config["symmetric_colorbar"] = config.get("pattern_symmetric_colorbar", True)

            # Prepare checkerboard pattern for recovery analysis
            checkerboard_pattern_3d = None
            if config.get("show_recovery_contours", False) and checkerboard_csv is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    check_perturbation_pct = ((df_checkerboard[vel_field].values - df_input[vel_field].values)
                                             / df_input[vel_field].values) * 100.0
                    check_perturbation_pct = np.where(np.isfinite(check_perturbation_pct), check_perturbation_pct, 0.0)

                # Convert to 3D grid
                x_coords, y_coords, z_coords, _ = prepare_3d_grid(df_pattern, vel_field)
                checkerboard_pattern_3d = np.full((len(z_coords), len(y_coords), len(x_coords)), np.nan)

                x_indices = np.searchsorted(x_coords, df_checkerboard["X_grid_km"].values)
                y_indices = np.searchsorted(y_coords, df_checkerboard["Y_grid_km"].values)
                z_indices = np.searchsorted(z_coords, df_checkerboard["Z_grid_km"].values)
                checkerboard_pattern_3d[z_indices, y_indices, x_indices] = check_perturbation_pct

            visualize_model(
                df_pattern, output_dir, pattern_config,
                f"inverted_pattern_pct_{vel_field}" if velocity_field == "both" else "inverted_pattern_pct",
                checkerboard_pattern_3d=checkerboard_pattern_3d,
            )
            del df_pattern

        del df_inverted, df_input
        if config.get("show_recovery_contours", False) and checkerboard_csv is not None:
            del df_checkerboard

    logger.info("All visualizations completed successfully!")


if __name__ == "__main__":
    # Example standalone usage
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        full_config = yaml.safe_load(f)

    base_dir = Path(full_config["general"]["output_dir"])
    input_csv = Path(full_config["input"]["velocity_model_csv"])
    checkerboard_csv = base_dir / "checkerboard_model" / "velocity_model_checkerboard.csv"
    inverted_csv = base_dir / "inverted_model" / "velocity_model_inverted.csv"

    # Get output directory from config, default to visualizations/
    output_dir = Path(full_config["visualization"].get("output_dir", base_dir / "visualizations"))

    visualize_checkerboard_test(
        input_csv,
        checkerboard_csv,
        inverted_csv,
        output_dir,
        full_config["visualization"],
    )
