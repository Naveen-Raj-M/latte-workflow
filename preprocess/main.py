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
from preprocess.coordinate_transformation import transform_coordinates
from preprocess.initial_velocity_model import create_initial_3d_model, validate_velocity_model
from preprocess.geometry_traveltime import create_geometry_and_traveltime_files
from preprocess.velocity_binary import create_velocity_binary_files
from preprocess.visualization import (
    create_station_event_visualization, 
    create_velocity_comparison_plots,
    create_velocity_animation
)

# Import utilities
from utils.logging_utils import setup_logging, ProcessTimer
from utils.io_utils import ensure_directory_exists


def run_coordinate_transform(config: DictConfig) -> None:
    """Run coordinate transformation."""
    if not config.enabled:
        return
        
    print("\n" + "="*60)
    print("COORDINATE TRANSFORMATION")
    print("="*60)
    
    with ProcessTimer("Coordinate Transformation"):
        transform_coordinates(config)
        
        print(f"✓ Coordinate transformation completed: {config.output_dir}")


def run_initial_3d_model(config: DictConfig) -> None:
    """Run initial 3D velocity model creation."""
    if not config.enabled:
        return
        
    print("\n" + "="*60)
    print("CREATING INITIAL 3D VELOCITY MODEL")
    print("="*60)
    
    with ProcessTimer("Initial 3D Model Creation"):
        # Validate input file first
        print("Validating 1D velocity model...")
        if not validate_velocity_model(config.model_1d_file):
            print("Warning: 1D velocity model validation failed, but proceeding anyway.")
        
        # Create the 3D model
        create_initial_3d_model(config)
        
        print(f"✓ Initial 3D velocity model created: {config.output_file}")


def run_geometry_traveltime(config: DictConfig) -> None:
    """Run geometry and travel time file creation."""
    if not config.enabled:
        return
        
    print("\n" + "="*60)
    print("CREATING GEOMETRY AND TRAVEL TIME FILES")
    print("="*60)
    
    with ProcessTimer("Geometry and Travel Time Processing"):
        create_geometry_and_traveltime_files(config)
        
        print(f"✓ Geometry files created in: {config.geometry_dir}")
        print(f"✓ Travel time files created in: {config.traveltime_dir}")
        if config.generate_source_binaries:
            print(f"✓ Source binary files created in: {config.source_dir}")


def run_velocity_binary(config: DictConfig) -> None:
    """Run velocity binary file creation."""
    if not config.enabled:
        return
        
    print("\n" + "="*60)
    print("CREATING VELOCITY BINARY FILES")
    print("="*60)
    
    with ProcessTimer("Velocity Binary Creation"):
        create_velocity_binary_files(config)
        
        print(f"✓ Vp binary file created: {config.vp_output}")
        print(f"✓ Vs binary file created: {config.vs_output}")


def run_visualization(config: DictConfig) -> None:
    """Run visualization tasks."""
    if not config.enabled:
        return
        
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Station and event visualization
    if 'station_event_vis' in config:
        print("Creating station and event visualization...")
        with ProcessTimer("Station/Event Visualization"):
            create_station_event_visualization(config.station_event_vis)
            print(f"✓ Station/event visualization: {config.station_event_vis.output_image}")
    
    # Velocity comparison plots
    if 'velocity_vis' in config:
        print("Creating velocity comparison plots...")
        with ProcessTimer("Velocity Visualization"):
            create_velocity_comparison_plots(config.velocity_vis)
            print("✓ Velocity comparison plots created")
    
    # Animated velocity visualization
    if 'velocity_animation' in config:
        print("Creating animated velocity visualization...")
        with ProcessTimer("Velocity Animation"):
            create_velocity_animation(config.velocity_animation)
            print(f"✓ Velocity animation: {config.velocity_animation.output_gif}")


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
    
    print("\nEnabled processing steps:")
    if config.coordinate_transform.enabled:
        print(f"  ✓ Coordinate transformation ({config.coordinate_transform.transformation_method})")
    if config.initial_3d_model.enabled:
        print("  ✓ Initial 3D velocity model creation")
    if config.geometry_traveltime.enabled:
        print("  ✓ Geometry and travel time file creation")
    if config.velocity_binary.enabled:
        print("  ✓ Velocity binary file creation")
    if config.visualization.enabled:
        print("  ✓ Visualization generation")
    
    print("\nKey input files:")
    if config.coordinate_transform.enabled:
        print(f"  - Stations: {config.coordinate_transform.station_file}")
        event_type = "corrected" if config.coordinate_transform.use_corrected_phase_pickings else "original"
        event_file = config.coordinate_transform.corrected_event_file if config.coordinate_transform.use_corrected_phase_pickings else config.coordinate_transform.event_file
        print(f"  - Events ({event_type}): {event_file}")
        print(f"  - Velocity model: {config.coordinate_transform.velocity_file}")
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
        print(f"  - Velocity CSV: {config.velocity_binary.input_csv}")


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
        # Step 1: Coordinate transformation
        run_coordinate_transform(config.coordinate_transform)
        
        # Step 2: Initial 3D velocity model
        run_initial_3d_model(config.initial_3d_model)
        
        # Step 3: Geometry and travel time files
        run_geometry_traveltime(config.geometry_traveltime)
        
        # Step 4: Velocity binary files
        run_velocity_binary(config.velocity_binary)
        
        # Step 5: Visualizations
        run_visualization(config.visualization)
        
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
        # Run the specific function
        function_map[function_name](config[function_name])
        
        print(f"\n✓ {function_name} completed successfully!")
        
    except Exception as e:
        logger.error(f"Function {function_name} failed: {e}")
        print(f"\n❌ ERROR in {function_name}: {e}")
        raise


@hydra.main(version_base=None, config_path="../configs", config_name="preprocess/default")
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