"""
Visualize spike-test velocity model slices in different planes.

This module creates comprehensive visualizations of velocity models for the spike
test, showing slices in XY (depth), XZ (N-S vertical), and YZ (E-W vertical) planes.

The per-slice plotting primitives are perturbation-agnostic and are reused from the
checkerboard module; only the orchestration and spike-specific naming live here.
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


def visualize_spike_test(
    input_csv: Optional[Path],
    spike_csv: Optional[Path],
    inverted_csv: Optional[Path],
    output_dir: Path,
    config: Dict,
) -> None:
    """
    Main function to create all spike test visualizations.

    Args:
        input_csv: Input (original) velocity model CSV
        spike_csv: Spike perturbed model CSV
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

    # =========== SPIKE MODEL ===========
    spike_config = config.get("visualize_spike", {})
    if isinstance(spike_config, bool):
        spike_config = {"enabled": spike_config}

    if spike_config.get("enabled", False) and spike_csv is not None:
        logger.info("Visualizing spike model")
        df_spike = read_velocity_file(spike_csv, show_progress=False)
        model_config = _merge_config(shared_config, spike_config)
        visualize_velocity_model(df_spike, output_dir, model_config, "spike",
                                 basin_data=basin_data, basin_config=basin_config)
        del df_spike

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

    if diff_config.get("enabled", False) and inverted_csv is not None and spike_csv is not None:
        logger.info("Visualizing difference (inverted - spike)")

        df_inverted = read_velocity_file(inverted_csv, show_progress=False)
        df_spike = read_velocity_file(spike_csv, show_progress=False)

        difference_config = _merge_config(shared_config, diff_config)

        for vel_field in velocity_fields:
            visualize_difference(
                df_inverted, df_spike, output_dir,
                difference_config, vel_field,
                basin_data=basin_data, basin_config=basin_config,
            )

        del df_inverted, df_spike

    # =========== SPIKE PATTERN ===========
    pattern_config = config.get("visualize_spike_pattern", {})
    if isinstance(pattern_config, bool):
        pattern_config = {"enabled": pattern_config}

    if pattern_config.get("enabled", False) and spike_csv is not None and input_csv is not None:
        logger.info("Visualizing spike pattern as percentage perturbation")

        df_spike = read_velocity_file(spike_csv, show_progress=False)
        df_input = read_velocity_file(input_csv, show_progress=False)

        spike_pattern_config = _merge_config(shared_config, pattern_config)

        for vel_field in velocity_fields:
            logger.info(f"  Processing spike pattern for {vel_field}")

            # Compute pattern as percentage perturbation
            with np.errstate(divide='ignore', invalid='ignore'):
                perturbation_pct = ((df_spike[vel_field].values - df_input[vel_field].values)
                                   / df_input[vel_field].values) * 100.0
                perturbation_pct = np.where(np.isfinite(perturbation_pct), perturbation_pct, 0.0)

            df_pattern = pd.DataFrame({
                'X_grid_km': df_spike['X_grid_km'].values,
                'Y_grid_km': df_spike['Y_grid_km'].values,
                'Z_grid_km': df_spike['Z_grid_km'].values,
                vel_field: perturbation_pct
            })

            visualize_pattern(
                df_pattern, output_dir, spike_pattern_config,
                "spike_pattern", vel_field,
                checkerboard_pattern_3d=None,  # No recovery for spike pattern
                basin_data=basin_data, basin_config=basin_config,
            )
            del df_pattern

        del df_spike, df_input

    # =========== INVERTED PATTERN ===========
    inv_pattern_config = config.get("visualize_inverted_pattern", {})
    if isinstance(inv_pattern_config, bool):
        inv_pattern_config = {"enabled": inv_pattern_config}

    if inv_pattern_config.get("enabled", False) and inverted_csv is not None and input_csv is not None:
        logger.info("Visualizing inverted pattern as percentage perturbation")

        df_inverted = read_velocity_file(inverted_csv, show_progress=False)
        df_input = read_velocity_file(input_csv, show_progress=False)

        # Load spike model for recovery analysis if needed
        df_spike = None
        if inv_pattern_config.get("show_recovery_contours", False) and spike_csv is not None:
            df_spike = read_velocity_file(spike_csv, show_progress=False)

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

            # Prepare spike pattern for recovery analysis
            spike_pattern_3d = None
            if df_spike is not None and inv_pattern_config.get("show_recovery_contours", False):
                with np.errstate(divide='ignore', invalid='ignore'):
                    spike_perturbation_pct = ((df_spike[vel_field].values - df_input[vel_field].values)
                                             / df_input[vel_field].values) * 100.0
                    spike_perturbation_pct = np.where(np.isfinite(spike_perturbation_pct), spike_perturbation_pct, 0.0)

                # Convert to 3D grid
                x_coords, y_coords, z_coords, _ = prepare_3d_grid(df_pattern, vel_field)
                spike_pattern_3d = np.full((len(z_coords), len(y_coords), len(x_coords)), np.nan)

                x_indices = np.searchsorted(x_coords, df_spike["X_grid_km"].values)
                y_indices = np.searchsorted(y_coords, df_spike["Y_grid_km"].values)
                z_indices = np.searchsorted(z_coords, df_spike["Z_grid_km"].values)
                spike_pattern_3d[z_indices, y_indices, x_indices] = spike_perturbation_pct

            visualize_pattern(
                df_pattern, output_dir, inverted_pattern_config,
                "inverted_pattern", vel_field,
                checkerboard_pattern_3d=spike_pattern_3d,
                basin_data=basin_data, basin_config=basin_config,
            )
            del df_pattern

        del df_inverted, df_input
        if df_spike is not None:
            del df_spike

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
    spike_csv = base_dir / "spike_model" / "velocity_model_spike.csv"
    inverted_csv = base_dir / "inverted_model" / "velocity_model_inverted.csv"

    # Get output directory from config, default to visualizations/
    output_dir = Path(full_config["visualization"].get("output_dir", base_dir / "visualizations"))

    visualize_spike_test(
        input_csv,
        spike_csv,
        inverted_csv,
        output_dir,
        full_config["visualization"],
    )
