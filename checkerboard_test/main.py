"""
Main workflow script for LATTE checkerboard resolution testing.

This module orchestrates the entire checkerboard test workflow:
1. Create checkerboard velocity model
2. Add noise to synthetic traveltimes (after LATTE forward modeling)
3. Convert inverted model to CSV (after LATTE inversion)
4. Visualize results

Usage:
    python main.py [--config config.yaml]
"""

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Dict, Union, Optional

from checkerboard_test.create_checkerboard_model import create_checkerboard_model
from checkerboard_test.add_traveltime_noise import process_traveltime_directory
from checkerboard_test.convert_inverted_model import convert_inverted_model
from checkerboard_test.visualize_slices import visualize_checkerboard_test


def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def run_checkerboard_workflow(config: Union[Dict, DictConfig]) -> None:
    """
    Run the complete checkerboard test workflow.

    Args:
        config: Configuration dictionary
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("LATTE CHECKERBOARD RESOLUTION TEST WORKFLOW")
    logger.info("=" * 80)

    # Extract configuration sections
    # Support both Dict and DictConfig access patterns
    general_config = config.general if isinstance(config, DictConfig) else config["general"]
    input_config = config.input if isinstance(config, DictConfig) else config["input"]
    workflow_config = config.workflow if isinstance(config, DictConfig) else config["workflow"]
    checkerboard_config = config.checkerboard if isinstance(config, DictConfig) else config["checkerboard"]
    model_conversion_config = config.model_conversion if isinstance(config, DictConfig) else config["model_conversion"]
    visualization_config = config.visualization if isinstance(config, DictConfig) else config["visualization"]

    # Get base paths
    output_dir = Path(general_config.output_dir if isinstance(general_config, DictConfig) else general_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define subdirectories
    checkerboard_dir = output_dir / "checkerboard_model"
    noisy_traveltime_dir = output_dir / "noisy_traveltimes"
    inverted_dir = output_dir / "inverted_model"
    visualization_dir = output_dir / "visualizations"

    # Step 1: Create checkerboard model
    step1_enabled = workflow_config.step1_create_checkerboard if isinstance(workflow_config, DictConfig) else workflow_config.get("step1_create_checkerboard", True)
    if step1_enabled:
        checkerboard_enabled = checkerboard_config.enabled if isinstance(checkerboard_config, DictConfig) else checkerboard_config["enabled"]
        if not checkerboard_enabled:
            logger.info("Step 1: SKIPPED (checkerboard.enabled = false)")
        else:
            try:
                logger.info("\nExecuting Step 1: Create Checkerboard Model")
                # Get input path, preferring H5 if available
                input_csv_path = input_config.velocity_model_hdf5 if (isinstance(input_config, DictConfig) and input_config.get('velocity_model_hdf5')) else (input_config.get('velocity_model_hdf5') if isinstance(input_config, dict) else None)
                if not input_csv_path or input_csv_path == "null":
                    input_csv_path = input_config.velocity_model_csv if isinstance(input_config, DictConfig) else input_config["velocity_model_csv"]

                create_checkerboard_model(
                    input_csv=Path(input_csv_path),
                    output_dir=checkerboard_dir,
                    config=checkerboard_config,
                )
                logger.info("✓ Step 1 completed successfully")
            except Exception as e:
                logger.error(f"✗ Step 1 failed: {e}", exc_info=True)
                raise
    else:
        logger.info("Step 1: SKIPPED (workflow.step1_create_checkerboard = false)")

    logger.info("\n" + "=" * 80)
    logger.info("PAUSE: Run LATTE forward modeling with checkerboard model")
    logger.info("=" * 80)
    logger.info("Next steps to perform manually:")
    logger.info(f"1. Copy {checkerboard_dir}/vp_checkerboard.bin to LATTE model directory")
    logger.info(f"2. Copy {checkerboard_dir}/vs_checkerboard.bin to LATTE model directory")
    logger.info(f"3. Copy original geometry files to LATTE")
    logger.info(f"4. Run LATTE forward modeling to generate synthetic traveltimes")
    logger.info(f"5. Update config.yaml with path to synthetic traveltime directory")
    logger.info(f"6. Re-run this script to continue with Step 2")
    logger.info("=" * 80)

    # Step 2: Add noise to synthetic traveltimes
    step2_enabled = workflow_config.step2_add_noise if isinstance(workflow_config, DictConfig) else workflow_config.get("step2_add_noise", True)
    if step2_enabled:
        traveltime_noise_config = config.traveltime_noise if isinstance(config, DictConfig) else config["traveltime_noise"]
        noise_enabled = traveltime_noise_config.enabled if isinstance(traveltime_noise_config, DictConfig) else traveltime_noise_config["enabled"]
        if not noise_enabled:
            logger.info("\nStep 2: SKIPPED (traveltime_noise.enabled = false)")
        else:
            # Check if synthetic traveltime directory exists
            synthetic_dir_path = input_config.synthetic_traveltime_dir if isinstance(input_config, DictConfig) else input_config["synthetic_traveltime_dir"]
            synthetic_dir = Path(synthetic_dir_path)
            if not synthetic_dir.exists() or not any(synthetic_dir.glob("shot_*_traveltime_*.bin")):
                logger.warning("\nStep 2: SKIPPED (synthetic traveltime files not found)")
                logger.warning(f"Expected location: {synthetic_dir}")
                logger.warning("Please run LATTE forward modeling first, then re-run this script")
            else:
                try:
                    logger.info("\nExecuting Step 2: Add Noise to Synthetic Traveltimes")
                    ground_truth_path = input_config.ground_truth_csv if isinstance(input_config, DictConfig) else input_config["ground_truth_csv"]
                    random_seed = general_config.random_seed if isinstance(general_config, DictConfig) else general_config.get("random_seed")
                    process_traveltime_directory(
                        input_dir=synthetic_dir,
                        output_dir=noisy_traveltime_dir,
                        ground_truth_csv=Path(ground_truth_path),
                        noise_config=traveltime_noise_config,
                        random_seed=random_seed,
                    )
                    logger.info("✓ Step 2 completed successfully")
                except Exception as e:
                    logger.error(f"✗ Step 2 failed: {e}", exc_info=True)
                    raise
    else:
        logger.info("\nStep 2: SKIPPED (workflow.step2_add_noise = false)")

    logger.info("\n" + "=" * 80)
    logger.info("PAUSE: Run LATTE inversion with noisy traveltimes")
    logger.info("=" * 80)
    logger.info("Next steps to perform manually:")
    logger.info(f"1. Copy noisy traveltime files from {noisy_traveltime_dir}")
    logger.info(f"2. Run LATTE inversion")
    logger.info(f"3. Update config.yaml with path to inverted model directory")
    logger.info(f"4. Re-run this script to continue with Step 3")
    logger.info("=" * 80)

    # Step 3: Convert inverted model
    step3_enabled = workflow_config.step3_convert_inverted if isinstance(workflow_config, DictConfig) else workflow_config.get("step3_convert_inverted", True)
    if step3_enabled:
        conversion_enabled = model_conversion_config.enabled if isinstance(model_conversion_config, DictConfig) else model_conversion_config["enabled"]
        if not conversion_enabled:
            logger.info("\nStep 3: SKIPPED (model_conversion.enabled = false)")
        else:
            # Check if inverted model exists
            inverted_model_dir_path = input_config.inverted_model_dir if isinstance(input_config, DictConfig) else input_config["inverted_model_dir"]
            inverted_model_dir = Path(inverted_model_dir_path)
            vp_bin = inverted_model_dir / "updated_vp.bin"
            vs_bin = inverted_model_dir / "updated_vs.bin"

            if not vp_bin.exists() or not vs_bin.exists():
                logger.warning("\nStep 3: SKIPPED (inverted model files not found)")
                logger.warning(f"Expected location: {inverted_model_dir}")
                logger.warning("Please run LATTE inversion first, then re-run this script")
            else:
                try:
                    logger.info("\nExecuting Step 3: Convert Inverted Model to CSV/H5")
                    # Get input path for sampling, preferring H5 if available
                    input_vel_path = input_config.velocity_model_hdf5 if (isinstance(input_config, DictConfig) and input_config.get('velocity_model_hdf5')) else (input_config.get('velocity_model_hdf5') if isinstance(input_config, dict) else None)
                    if not input_vel_path or input_vel_path == "null":
                        input_vel_path = input_config.velocity_model_csv if isinstance(input_config, DictConfig) else input_config["velocity_model_csv"]

                    convert_inverted_model(
                        input_model_dir=inverted_model_dir,
                        output_dir=inverted_dir,
                        input_csv=Path(input_vel_path),
                        config=model_conversion_config,
                    )
                    logger.info("✓ Step 3 completed successfully")
                except Exception as e:
                    logger.error(f"✗ Step 3 failed: {e}", exc_info=True)
                    raise
    else:
        logger.info("\nStep 3: SKIPPED (workflow.step3_convert_inverted = false)")

    # Step 4: Visualize results
    step4_enabled = workflow_config.step4_visualize if isinstance(workflow_config, DictConfig) else workflow_config.get("step4_visualize", True)
    if step4_enabled:
        viz_enabled = visualization_config.enabled if isinstance(visualization_config, DictConfig) else visualization_config["enabled"]
        if not viz_enabled:
            logger.info("\nStep 4: SKIPPED (visualization.enabled = false)")
        else:
            try:
                logger.info("\nExecuting Step 4: Create Visualizations")

                # Helper function to get model path based on output format
                def get_model_path(base_dir: Path, base_name: str, format_config: str) -> Optional[Path]:
                    """Get model path based on output format, preferring H5 if both exist."""
                    if format_config in ["hdf5", "both"]:
                        h5_path = base_dir / f"{base_name}.h5"
                        if h5_path.exists():
                            return h5_path
                    if format_config in ["csv", "both"]:
                        csv_path = base_dir / f"{base_name}.csv"
                        if csv_path.exists():
                            return csv_path
                    return None

                # Get input model path (prefer H5 if available)
                input_model_path = input_config.velocity_model_hdf5 if (isinstance(input_config, DictConfig) and input_config.get('velocity_model_hdf5')) else (input_config.get('velocity_model_hdf5') if isinstance(input_config, dict) else None)
                if not input_model_path or input_model_path == "null":
                    input_model_path = input_config.velocity_model_csv if isinstance(input_config, DictConfig) else input_config["velocity_model_csv"]
                input_csv = Path(input_model_path) if Path(input_model_path).exists() else None

                # Get checkerboard model path based on output format
                checkerboard_format = checkerboard_config.output_format if isinstance(checkerboard_config, DictConfig) else checkerboard_config.get("output_format", "csv")
                checkerboard_csv = get_model_path(checkerboard_dir, "velocity_model_checkerboard", checkerboard_format)

                # Get inverted model path based on output format
                inverted_format = model_conversion_config.output_format if isinstance(model_conversion_config, DictConfig) else model_conversion_config.get("output_format", "csv")
                inverted_csv = get_model_path(inverted_dir, "velocity_model_inverted", inverted_format)

                if checkerboard_csv is None and inverted_csv is None:
                    logger.warning("No model files found for visualization")
                else:
                    visualize_checkerboard_test(
                        input_csv=input_csv,
                        checkerboard_csv=checkerboard_csv,
                        inverted_csv=inverted_csv,
                        output_dir=visualization_dir,
                        config=visualization_config,
                    )
                    logger.info("✓ Step 4 completed successfully")
            except Exception as e:
                logger.error(f"✗ Step 4 failed: {e}", exc_info=True)
                raise
    else:
        logger.info("\nStep 4: SKIPPED (workflow.step4_visualize = false)")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("WORKFLOW SUMMARY")
    logger.info("=" * 80)
    logger.info("Output directory: " + str(output_dir))
    logger.info("\nGenerated files:")
    if checkerboard_dir.exists():
        logger.info(f"  - Checkerboard model: {checkerboard_dir}")
    if noisy_traveltime_dir.exists():
        logger.info(f"  - Noisy traveltimes: {noisy_traveltime_dir}")
    if inverted_dir.exists():
        logger.info(f"  - Inverted model: {inverted_dir}")
    if visualization_dir.exists():
        logger.info(f"  - Visualizations: {visualization_dir}")
    logger.info("\n" + "=" * 80)
    logger.info("Checkerboard test workflow completed!")
    logger.info("=" * 80)


@hydra.main(version_base=None, config_path="../configs/checkerboard_test", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Main entry point with Hydra configuration.

    Args:
        cfg: Hydra configuration object loaded from configs/checkerboard_test/default.yaml

    Usage:
        # Use default config
        python -m checkerboard_test.main

        # Use custom config
        python -m checkerboard_test.main --config-name=my_config

        # Override specific values
        python -m checkerboard_test.main checkerboard.amplitude=0.10

        # Override output format
        python -m checkerboard_test.main checkerboard.output_format=hdf5
    """
    # Setup logging from config
    log_level = cfg.general.get('log_level', 'INFO')
    setup_logging(log_level)

    logger = logging.getLogger(__name__)

    # Log configuration in debug mode
    if log_level.upper() == "DEBUG":
        logger.debug("Full configuration:")
        logger.debug(f"\n{OmegaConf.to_yaml(cfg)}")

    # Run workflow
    try:
        run_checkerboard_workflow(cfg)
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
