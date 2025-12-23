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
import argparse
import yaml
from pathlib import Path
from typing import Dict

from create_checkerboard_model import create_checkerboard_model
from add_traveltime_noise import process_traveltime_directory
from convert_inverted_model import convert_inverted_model
from visualize_slices import visualize_checkerboard_test


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


def load_config(config_path: Path) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Configuration dictionary
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def run_checkerboard_workflow(config: Dict) -> None:
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
    general_config = config["general"]
    input_config = config["input"]
    workflow_config = config["workflow"]

    # Get base paths
    output_dir = Path(general_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define subdirectories
    checkerboard_dir = output_dir / "checkerboard_model"
    noisy_traveltime_dir = output_dir / "noisy_traveltimes"
    inverted_dir = output_dir / "inverted_model"
    visualization_dir = output_dir / "visualizations"

    # Step 1: Create checkerboard model
    if workflow_config.get("step1_create_checkerboard", True):
        if not config["checkerboard"]["enabled"]:
            logger.info("Step 1: SKIPPED (checkerboard.enabled = false)")
        else:
            try:
                logger.info("\nExecuting Step 1: Create Checkerboard Model")
                create_checkerboard_model(
                    input_csv=Path(input_config["velocity_model_csv"]),
                    output_dir=checkerboard_dir,
                    config=config["checkerboard"],
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
    if workflow_config.get("step2_add_noise", True):
        if not config["traveltime_noise"]["enabled"]:
            logger.info("\nStep 2: SKIPPED (traveltime_noise.enabled = false)")
        else:
            # Check if synthetic traveltime directory exists
            synthetic_dir = Path(input_config["synthetic_traveltime_dir"])
            if not synthetic_dir.exists() or not any(synthetic_dir.glob("shot_*_traveltime_*.bin")):
                logger.warning("\nStep 2: SKIPPED (synthetic traveltime files not found)")
                logger.warning(f"Expected location: {synthetic_dir}")
                logger.warning("Please run LATTE forward modeling first, then re-run this script")
            else:
                try:
                    logger.info("\nExecuting Step 2: Add Noise to Synthetic Traveltimes")
                    process_traveltime_directory(
                        input_dir=synthetic_dir,
                        output_dir=noisy_traveltime_dir,
                        ground_truth_csv=Path(input_config["ground_truth_csv"]),
                        noise_config=config["traveltime_noise"],
                        random_seed=general_config.get("random_seed"),
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
    if workflow_config.get("step3_convert_inverted", True):
        if not config["model_conversion"]["enabled"]:
            logger.info("\nStep 3: SKIPPED (model_conversion.enabled = false)")
        else:
            # Check if inverted model exists
            inverted_model_dir = Path(input_config["inverted_model_dir"])
            vp_bin = inverted_model_dir / "updated_vp.bin"
            vs_bin = inverted_model_dir / "updated_vs.bin"

            if not vp_bin.exists() or not vs_bin.exists():
                logger.warning("\nStep 3: SKIPPED (inverted model files not found)")
                logger.warning(f"Expected location: {inverted_model_dir}")
                logger.warning("Please run LATTE inversion first, then re-run this script")
            else:
                try:
                    logger.info("\nExecuting Step 3: Convert Inverted Model to CSV")
                    convert_inverted_model(
                        input_model_dir=inverted_model_dir,
                        output_dir=inverted_dir,
                        input_csv=Path(input_config["velocity_model_csv"]),
                        config=config["model_conversion"],
                    )
                    logger.info("✓ Step 3 completed successfully")
                except Exception as e:
                    logger.error(f"✗ Step 3 failed: {e}", exc_info=True)
                    raise
    else:
        logger.info("\nStep 3: SKIPPED (workflow.step3_convert_inverted = false)")

    # Step 4: Visualize results
    if workflow_config.get("step4_visualize", True):
        if not config["visualization"]["enabled"]:
            logger.info("\nStep 4: SKIPPED (visualization.enabled = false)")
        else:
            try:
                logger.info("\nExecuting Step 4: Create Visualizations")

                # Determine which CSVs exist
                input_csv = Path(input_config["velocity_model_csv"])
                checkerboard_csv = checkerboard_dir / "velocity_model_checkerboard.csv"
                inverted_csv = inverted_dir / "velocity_model_inverted.csv"

                # Check existence
                input_csv = input_csv if input_csv.exists() else None
                checkerboard_csv = checkerboard_csv if checkerboard_csv.exists() else None
                inverted_csv = inverted_csv if inverted_csv.exists() else None

                if checkerboard_csv is None and inverted_csv is None:
                    logger.warning("No model CSVs found for visualization")
                else:
                    visualize_checkerboard_test(
                        input_csv=input_csv,
                        checkerboard_csv=checkerboard_csv,
                        inverted_csv=inverted_csv,
                        output_dir=visualization_dir,
                        config=config["visualization"],
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LATTE Checkerboard Resolution Test Workflow"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Assume relative to script directory
        config_path = Path(__file__).parent / config_path

    config = load_config(config_path)

    # Setup logging
    setup_logging(config["general"].get("log_level", "INFO"))

    # Run workflow
    try:
        run_checkerboard_workflow(config)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Workflow failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
