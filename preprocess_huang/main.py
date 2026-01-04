"""
Main preprocessing module for LATTE tomography package.
This module orchestrates all preprocessing tasks using Hydra configuration.
"""

import logging
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

# Import preprocessing functions
from preprocess_huang.coordinate_transformation import transform_coordinates
from preprocess_huang.initial_velocity_model import create_initial_3d_model, validate_velocity_model
from preprocess_huang.geometry_traveltime import create_geometry_and_traveltime_files
from preprocess_huang.velocity_binary import create_velocity_binary_files
from preprocess_huang.visualization import (
    create_station_event_visualization, 
    create_velocity_comparison_plots,
    create_velocity_animation
)

# Import utilities
from utils.logging_utils import setup_logging, ProcessTimer
from utils.io_utils import ensure_directory_exists


def run_coordinate_transform(cfg: DictConfig) -> None:
    """Run coordinate transformation."""
    if not cfg.coordinate_transform.enabled:
        return

    print("\n" + "="*60)
    print("COORDINATE TRANSFORMATION")
    print("="*60)

    with ProcessTimer("Coordinate Transformation"):
        # Pass full config so submodule can access parent-level configs
        transform_coordinates(cfg)

        print(f"✓ Coordinate transformation completed: {cfg.coordinate_transform.output_dir}")


def run_initial_3d_model(cfg: DictConfig) -> None:
    """Run initial 3D velocity model creation."""
    if not cfg.initial_3d_model.enabled:
        return

    print("\n" + "="*60)
    print("CREATING INITIAL 3D VELOCITY MODEL")
    print("="*60)

    with ProcessTimer("Initial 3D Model Creation"):
        # Validate input file first
        print("Validating 1D velocity model...")
        if not validate_velocity_model(cfg.initial_3d_model.model_1d_file):
            print("Warning: 1D velocity model validation failed, but proceeding anyway.")

        # Create the 3D model - pass full config
        create_initial_3d_model(cfg)

        print(f"✓ Initial 3D velocity model created: {cfg.initial_3d_model.output_file}")


def run_geometry_traveltime(cfg: DictConfig) -> None:
    """Run geometry and travel time file creation."""
    if not cfg.geometry_traveltime.enabled:
        return

    print("\n" + "="*60)
    print("CREATING GEOMETRY AND TRAVEL TIME FILES")
    print("="*60)

    with ProcessTimer("Geometry and Travel Time Processing"):
        # Pass full config so submodule can access origin_time and other parent-level configs
        create_geometry_and_traveltime_files(cfg)

        print(f"✓ Geometry files created in: {cfg.geometry_traveltime.geometry_dir}")
        print(f"✓ Travel time files created in: {cfg.geometry_traveltime.traveltime_dir}")
        if cfg.geometry_traveltime.generate_source_binaries:
            print(f"✓ Source binary files created in: {cfg.geometry_traveltime.source_dir}")


def run_velocity_binary(cfg: DictConfig) -> None:
    """Run velocity binary file creation."""
    if not cfg.velocity_binary.enabled:
        return

    print("\n" + "="*60)
    print("CREATING VELOCITY BINARY FILES")
    print("="*60)

    with ProcessTimer("Velocity Binary Creation"):
        # Pass full config
        create_velocity_binary_files(cfg)

        phase_type = cfg.velocity_binary.get('phase_type', 'both')
        if phase_type in ['both', 'p']:
            print(f"✓ Vp binary file created: {cfg.velocity_binary.vp_output}")
        if phase_type in ['both', 's']:
            print(f"✓ Vs binary file created: {cfg.velocity_binary.vs_output}")


def run_visualization(cfg: DictConfig) -> None:
    """Run visualization tasks."""
    if not cfg.visualization.enabled:
        return

    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    # Station and event visualization
    if 'station_event_vis' in cfg.visualization:
        print("Creating station and event visualization...")
        with ProcessTimer("Station/Event Visualization"):
            create_station_event_visualization(cfg.visualization.station_event_vis)
            print(f"✓ Station/event visualization: {cfg.visualization.station_event_vis.output_image}")

    # Velocity comparison plots
    if 'velocity_vis' in cfg.visualization:
        print("Creating velocity comparison plots...")
        with ProcessTimer("Velocity Visualization"):
            create_velocity_comparison_plots(cfg.visualization.velocity_vis)
            print("✓ Velocity comparison plots created")

    # Animated velocity visualization
    if 'velocity_animation' in cfg.visualization:
        print("Creating animated velocity visualization...")
        with ProcessTimer("Velocity Animation"):
            create_velocity_animation(cfg.visualization.velocity_animation)
            print(f"✓ Velocity animation: {cfg.visualization.velocity_animation.output_gif}")


def setup_output_directories(config: DictConfig) -> None:
    """Create necessary output directories."""
    output_dir = Path(config.preprocessing.output_dir)
    ensure_directory_exists(output_dir)
    
    # Create subdirectories for different processing types
    subdirs = ['coordinate_transform', 'velocity_models', 'geometry', 'traveltime', 'model', 'visualization']
    for subdir in subdirs:
        ensure_directory_exists(output_dir / subdir)


def print_configuration_summary(config: DictConfig) -> None:
    """Print a summary of what will be processed."""
    print("\n" + "="*60)
    print("LATTE PREPROCESSING CONFIGURATION SUMMARY")
    print("="*60)

    print(f"Output directory: {config.preprocessing.output_dir}")
    print(f"Data directory: {config.data.base_dir}")

    # Origin time configuration
    origin_mode = config.origin_time.get('mode', 'fixed')
    print(f"\nOrigin time mode: {origin_mode}")
    if origin_mode == 'fixed':
        fixed_value = config.origin_time.get('fixed_value', 0.0)
        print(f"  Fixed origin time value: {fixed_value} seconds")
    elif origin_mode == 'unix_epoch':
        print(f"  Using Unix epoch timestamps")

    print("\nEnabled processing steps:")
    if config.coordinate_transform.enabled:
        print(f"  ✓ Coordinate transformation ({config.coordinate_transform.transformation_method})")
        if config.coordinate_transform.get('max_depth_km'):
            print(f"    + Depth filtering: max_depth_km = {config.coordinate_transform.max_depth_km} km")
        if config.coordinate_transform.process_final_velocity:
            print(f"    + Final velocity text to CSV conversion")
    if config.initial_3d_model.enabled:
        print("  ✓ Initial 3D velocity model creation")
    if config.geometry_traveltime.enabled:
        phase_type = config.geometry_traveltime.get('phase_type', 'both')
        print(f"  ✓ Geometry and travel time file creation (phase_type: {phase_type})")
    if config.velocity_binary.enabled:
        phase_type = config.velocity_binary.get('phase_type', 'both')
        print(f"  ✓ Velocity binary file creation (phase_type: {phase_type})")
        if config.velocity_binary.create_final_binaries:
            print("    + Final velocity binary files (vp_final.bin, vs_final.bin)")
    if config.visualization.enabled:
        print("  ✓ Visualization generation")
    
    print("\nKey input files:")
    if config.coordinate_transform.enabled:
        print(f"  - Stations: {config.coordinate_transform.station_file}")
        event_type = "corrected" if config.coordinate_transform.use_corrected_phase_pickings else "original"
        event_file = config.coordinate_transform.corrected_event_file if config.coordinate_transform.use_corrected_phase_pickings else config.coordinate_transform.event_file
        print(f"  - Events ({event_type}): {event_file}")
        print(f"  - Initial velocity model: {config.coordinate_transform.initial_velocity_file}")
    if config.initial_3d_model.enabled:
        print(f"  - 1D model: {config.initial_3d_model.model_1d_file}")
        print(f"  - 3D template: {config.initial_3d_model.template_3d_file}")
    if config.geometry_traveltime.enabled:
        file_type = "corrected" if config.geometry_traveltime.use_corrected_phase_pickings else "original"
        phase_file = config.geometry_traveltime.corrected_phase_file if config.geometry_traveltime.use_corrected_phase_pickings else config.geometry_traveltime.phase_file
        print(f"  - Phase picks ({file_type}): {phase_file}")
        print(f"  - Stations CSV: {config.geometry_traveltime.station_csv}")
        print(f"  - Events CSV: {config.geometry_traveltime.event_csv}")
    if config.velocity_binary.enabled:
        print(f"  - Initial velocity CSV: {config.velocity_binary.input_csv}")
    if config.coordinate_transform.enabled and config.coordinate_transform.process_final_velocity:
        print(f"  - Final velocity model: {config.coordinate_transform.final_velocity_file}")
    if config.velocity_binary.enabled and config.velocity_binary.create_final_binaries:
        print(f"  - Final velocity CSV: {config.velocity_binary.final_input_csv}")


def run_preprocessing_pipeline(config: DictConfig) -> None:
    """
    Run the complete preprocessing pipeline.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting LATTE preprocessing pipeline")

    # Print configuration summary
    print_configuration_summary(config)

    # Setup output directories
    setup_output_directories(config)

    # Run processing steps in order
    try:

        # Step 1: Initial 3D velocity model
        run_initial_3d_model(config)

        # Step 2: Coordinate transformation
        run_coordinate_transform(config)

        # Step 3: Geometry and travel time files
        run_geometry_traveltime(config)

        # Step 4: Velocity binary files
        run_velocity_binary(config)

        # Step 5: Visualizations
        run_visualization(config)
        
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"All outputs saved to: {config.preprocessing.output_dir}")
        
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {e}")
        print(f"\n❌ ERROR: {e}")
        print("Preprocessing pipeline failed. Check the logs for details.")
        raise


def run_single_function(config: DictConfig, function_name: str) -> None:
    """
    Run a single preprocessing function.

    Args:
        config: Full configuration
        function_name: Name of function to run
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running single function: {function_name}")

    # Map function names to their corresponding functions
    function_map = {
        'coordinate_transform': run_coordinate_transform,
        'initial_3d_model': run_initial_3d_model,
        'geometry_traveltime': run_geometry_traveltime,
        'velocity_binary': run_velocity_binary,
        'visualization': run_visualization
    }

    if function_name not in function_map:
        available_functions = ', '.join(function_map.keys())
        raise ValueError(f"Unknown function: {function_name}. Available: {available_functions}")

    # Enable only the requested function
    for key in function_map.keys():
        if hasattr(config, key):
            config[key].enabled = (key == function_name)

    # Setup minimal output directories
    setup_output_directories(config)

    print(f"\n" + "="*60)
    print(f"RUNNING SINGLE FUNCTION: {function_name.upper()}")
    print("="*60)

    try:
        # Run the specific function - pass full config so submodules can access parent-level configs
        function_map[function_name](config)
        
        print(f"\n✓ {function_name} completed successfully!")
        
    except Exception as e:
        logger.error(f"Function {function_name} failed: {e}")
        print(f"\n❌ ERROR in {function_name}: {e}")
        raise


def validate_preprocessing_config(cfg: DictConfig) -> None:
    """Validate preprocessing configuration."""
    logger = logging.getLogger(__name__)

    # Validate origin_time configuration
    origin_mode = cfg.origin_time.get('mode', 'fixed')
    if origin_mode not in ['fixed', 'unix_epoch']:
        logger.error(f"Invalid origin_time.mode: {origin_mode}. Must be 'fixed' or 'unix_epoch'")
        raise ValueError(f"Invalid origin_time.mode: {origin_mode}")

    if origin_mode == 'fixed':
        fixed_value = cfg.origin_time.get('fixed_value')
        if fixed_value is None:
            logger.error("origin_time.fixed_value is required when mode is 'fixed'")
            raise ValueError("origin_time.fixed_value is required when mode is 'fixed'")

    # Validate phase_type in geometry_traveltime
    if cfg.geometry_traveltime.enabled:
        phase_type = cfg.geometry_traveltime.get('phase_type', 'both')
        if phase_type not in ['both', 'p', 's']:
            logger.error(f"Invalid geometry_traveltime.phase_type: {phase_type}. Must be 'both', 'p', or 's'")
            raise ValueError(f"Invalid phase_type: {phase_type}")

    # Validate phase_type in velocity_binary
    if cfg.velocity_binary.enabled:
        phase_type = cfg.velocity_binary.get('phase_type', 'both')
        if phase_type not in ['both', 'p', 's']:
            logger.error(f"Invalid velocity_binary.phase_type: {phase_type}. Must be 'both', 'p', or 's'")
            raise ValueError(f"Invalid phase_type: {phase_type}")

    logger.info("Configuration validation passed")


@hydra.main(version_base=None, config_path="../configs/preprocess_huang", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for preprocessing with Hydra.

    Args:
        cfg: Hydra configuration
    """
    # Setup logging
    setup_logging(level=cfg.preprocessing.log_level)
    logger = logging.getLogger(__name__)

    # Log configuration (only in debug mode)
    if cfg.preprocessing.log_level.upper() == "DEBUG":
        logger.debug("Full configuration:")
        logger.debug(f"\n{OmegaConf.to_yaml(cfg)}")

    # Validate configuration
    validate_preprocessing_config(cfg)

    # Check if we should run a single function
    function_name = cfg.get('single_function', None)
    
    if function_name:
        # Run single function
        run_single_function(cfg, function_name)
    else:
        # Run full pipeline
        run_preprocessing_pipeline(cfg)


if __name__ == "__main__":
    main()