"""
Visualize velocity model slices in different planes.

This module creates comprehensive visualizations of velocity models,
showing slices in XY (depth), XZ (N-S vertical), and YZ (E-W vertical) planes.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from utils.velocity_io import read_velocity_file
from checkerboard_test.grid_utils import prepare_3d_grid
from checkerboard_test.visualize_model import visualize_velocity_model
from checkerboard_test.visualize_difference import visualize_difference
from checkerboard_test.visualize_pattern import visualize_pattern
from analysis.basin_boundaries import load_basin_boundaries

logger = logging.getLogger(__name__)


def _merge_config(base_config: Dict, viz_config: Dict) -> Dict:
    """
    Merge base visualization config with per-visualization config.

    The per-visualization config takes precedence over base config.

    Args:
        base_config: Base visualization configuration (shared settings)
        viz_config: Per-visualization configuration

    Returns:
        Merged configuration dictionary
    """
    merged = dict(base_config)

    # Copy over per-visualization settings
    for key, value in viz_config.items():
        if key != "enabled":  # Don't copy 'enabled' to merged config
            merged[key] = value

    return merged


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
        config: Visualization configuration with nested per-visualization settings
    """
    logger.info("=" * 80)
    logger.info("STEP 4: Creating Visualizations")
    logger.info("=" * 80)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract shared configuration
    shared_config = {
        "planes": config.get("planes", ["xy", "xz", "yz"]),
        "velocity_field": config.get("velocity_field", "vp"),
        "slice_unit": config.get("slice_unit", "indices"),
        "spacing_unit": config.get("spacing_unit", "indices"),
        "tolerance_km": config.get("tolerance_km", 1e-6),
        "xy_slices": config.get("xy_slices"),
        "xz_slices": config.get("xz_slices"),
        "yz_slices": config.get("yz_slices"),
        "slice_spacing": config.get("slice_spacing", {"xy": 5, "xz": 20, "yz": 20}),
        "figure": config.get("figure", {"dpi": 150, "figsize": [10, 8], "format": "png"}),
        "show_colorbar": config.get("show_colorbar", True),
        "show_grid": config.get("show_grid", True),
        "show_title": config.get("show_title", True),
        "font_size": config.get("font_size", {"title": 14, "labels": 12, "colorbar": 10}),
    }

    # Load basin boundaries once (shared across all visualizations, XY plane only)
    basin_config = config.get("basin_boundaries", {})
    basin_data = load_basin_boundaries(basin_config)

    # Get velocity fields to process
    velocity_field = shared_config["velocity_field"]
    if velocity_field == "both":
        velocity_fields = ["vp", "vs"]
    else:
        velocity_fields = [velocity_field]

    # =========== INPUT MODEL ===========
    input_config = config.get("visualize_input", {})
    if isinstance(input_config, bool):
        # Handle legacy format where visualize_input is just a boolean
        input_config = {"enabled": input_config}

    if input_config.get("enabled", False) and input_csv is not None:
        logger.info("Visualizing input model")
        df_input = read_velocity_file(input_csv, show_progress=False)
        model_config = _merge_config(shared_config, input_config)
        visualize_velocity_model(df_input, output_dir, model_config, "input",
                                 basin_data=basin_data, basin_config=basin_config)
        del df_input

    # =========== CHECKERBOARD MODEL ===========
    checkerboard_config = config.get("visualize_checkerboard", {})
    if isinstance(checkerboard_config, bool):
        checkerboard_config = {"enabled": checkerboard_config}

    if checkerboard_config.get("enabled", False) and checkerboard_csv is not None:
        logger.info("Visualizing checkerboard model")
        df_checkerboard = read_velocity_file(checkerboard_csv, show_progress=False)
        model_config = _merge_config(shared_config, checkerboard_config)
        visualize_velocity_model(df_checkerboard, output_dir, model_config, "checkerboard",
                                 basin_data=basin_data, basin_config=basin_config)
        del df_checkerboard

    # =========== INVERTED MODEL ===========
    inverted_config = config.get("visualize_inverted", {})
    if isinstance(inverted_config, bool):
        inverted_config = {"enabled": inverted_config}

    if inverted_config.get("enabled", False) and inverted_csv is not None:
        logger.info("Visualizing inverted model")
        df_inverted = read_velocity_file(inverted_csv, show_progress=False)
        model_config = _merge_config(shared_config, inverted_config)
        visualize_velocity_model(df_inverted, output_dir, model_config, "inverted",
                                 basin_data=basin_data, basin_config=basin_config)
        del df_inverted

    # =========== DIFFERENCE PLOT ===========
    diff_config = config.get("visualize_difference", {})
    if isinstance(diff_config, bool):
        diff_config = {"enabled": diff_config}

    if diff_config.get("enabled", False) and inverted_csv is not None and checkerboard_csv is not None:
        logger.info("Visualizing difference (inverted - checkerboard)")

        df_inverted = read_velocity_file(inverted_csv, show_progress=False)
        df_checkerboard = read_velocity_file(checkerboard_csv, show_progress=False)

        difference_config = _merge_config(shared_config, diff_config)

        for vel_field in velocity_fields:
            visualize_difference(
                df_inverted, df_checkerboard, output_dir,
                difference_config, vel_field,
                basin_data=basin_data, basin_config=basin_config,
            )

        del df_inverted, df_checkerboard

    # =========== CHECKERBOARD PATTERN ===========
    pattern_config = config.get("visualize_checkerboard_pattern", {})
    if isinstance(pattern_config, bool):
        pattern_config = {"enabled": pattern_config}

    if pattern_config.get("enabled", False) and checkerboard_csv is not None and input_csv is not None:
        logger.info("Visualizing checkerboard pattern as percentage perturbation")

        df_checkerboard = read_velocity_file(checkerboard_csv, show_progress=False)
        df_input = read_velocity_file(input_csv, show_progress=False)

        checkerboard_pattern_config = _merge_config(shared_config, pattern_config)

        for vel_field in velocity_fields:
            logger.info(f"  Processing checkerboard pattern for {vel_field}")

            # Compute pattern as percentage perturbation
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

            visualize_pattern(
                df_pattern, output_dir, checkerboard_pattern_config,
                "checkerboard_pattern", vel_field,
                checkerboard_pattern_3d=None,  # No recovery for checkerboard pattern
                basin_data=basin_data, basin_config=basin_config,
            )
            del df_pattern

        del df_checkerboard, df_input

    # =========== INVERTED PATTERN ===========
    inv_pattern_config = config.get("visualize_inverted_pattern", {})
    if isinstance(inv_pattern_config, bool):
        inv_pattern_config = {"enabled": inv_pattern_config}

    if inv_pattern_config.get("enabled", False) and inverted_csv is not None and input_csv is not None:
        logger.info("Visualizing inverted pattern as percentage perturbation")

        df_inverted = read_velocity_file(inverted_csv, show_progress=False)
        df_input = read_velocity_file(input_csv, show_progress=False)

        # Load checkerboard for recovery analysis if needed
        df_checkerboard = None
        if inv_pattern_config.get("show_recovery_contours", False) and checkerboard_csv is not None:
            df_checkerboard = read_velocity_file(checkerboard_csv, show_progress=False)

        inverted_pattern_config = _merge_config(shared_config, inv_pattern_config)

        for vel_field in velocity_fields:
            logger.info(f"  Processing inverted pattern for {vel_field}")

            # Compute inverted pattern as percentage perturbation
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

            # Prepare checkerboard pattern for recovery analysis
            checkerboard_pattern_3d = None
            if df_checkerboard is not None and inv_pattern_config.get("show_recovery_contours", False):
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

            visualize_pattern(
                df_pattern, output_dir, inverted_pattern_config,
                "inverted_pattern", vel_field,
                checkerboard_pattern_3d=checkerboard_pattern_3d,
                basin_data=basin_data, basin_config=basin_config,
            )
            del df_pattern

        del df_inverted, df_input
        if df_checkerboard is not None:
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
