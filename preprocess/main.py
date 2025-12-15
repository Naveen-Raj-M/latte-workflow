"""Main preprocessing module for LATTE tomography with QuakeML data.

This module orchestrates all preprocessing tasks for seismic data from
QuakeML files or HDF5 format, using Hydra configuration.

Workflow order:
1. Load data (QuakeML or HDF5)
2. Transform coordinates (flattening)
3. Create geometry and travel time files
4. Create velocity model and binaries
5. Create visualizations
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Import preprocessing functions
from preprocess.data_loader import load_seismic_data
from preprocess.coordinate_transformation import transform_coordinates
from preprocess.geometry_traveltime import create_geometry_and_traveltime_files
from preprocess.initial_velocity_model import create_velocity_model
from preprocess.visualization import (
    create_station_event_visualization,
    create_velocity_model_visualization
)

# Import utilities
from utils.logging_utils import setup_logging, ProcessTimer
from utils.io_utils import ensure_directory_exists


def run_data_loading(cfg: DictConfig):
    """Load seismic data from QuakeML or HDF5.

    Args:
        cfg: Configuration settings.

    Returns:
        Tuple of (events_df, stations_df, arrivals_df, picks_df).
    """
    print("\n" + "="*60)
    print("DATA LOADING")
    print("="*60)

    with ProcessTimer("Data Loading"):
        events_df, stations_df, arrivals_df, picks_df = load_seismic_data(cfg)

        print(f"\nLoaded data summary:")
        print(f"  Events: {len(events_df)}")
        print(f"  Stations: {len(stations_df)}")
        print(f"  Arrivals: {len(arrivals_df)}")
        print(f"  Picks: {len(picks_df)}")

    return events_df, stations_df, arrivals_df, picks_df


def run_coordinate_transform(cfg: DictConfig, events_df, stations_df):
    """Transform coordinates from geodetic to Cartesian.

    Args:
        cfg: Configuration settings.
        events_df: DataFrame with events in geodetic coordinates.
        stations_df: DataFrame with stations in geodetic coordinates.

    Returns:
        Tuple of (events_cart_df, stations_cart_df).
    """
    if not cfg.coordinate_transform.enabled:
        return events_df, stations_df

    print("\n" + "="*60)
    print("COORDINATE TRANSFORMATION (FLATTENING)")
    print("="*60)

    with ProcessTimer("Coordinate Transformation"):
        events_cart, stations_cart = transform_coordinates(
            cfg, events_df, stations_df
        )

        print(f"\nTransformed coordinates:")
        print(f"  Events: {len(events_cart)} to Cartesian")
        print(f"  Stations: {len(stations_cart)} to Cartesian")

    return events_cart, stations_cart


def run_geometry_traveltime(cfg: DictConfig, events_cart, stations_cart,
                            picks_df) -> None:
    """Create geometry and travel time files.

    Args:
        cfg: Configuration settings.
        events_cart: DataFrame with events in Cartesian coordinates.
        stations_cart: DataFrame with stations in Cartesian coordinates.
        picks_df: DataFrame with picks.
    """
    if not cfg.geometry_traveltime.enabled:
        return

    print("\n" + "="*60)
    print("CREATING GEOMETRY AND TRAVEL TIME FILES")
    print("="*60)

    with ProcessTimer("Geometry and Travel Time Processing"):
        create_geometry_and_traveltime_files(
            cfg, events_cart, stations_cart, picks_df
        )

        print(f"\nGeometry files created in: "
              f"{cfg.geometry_traveltime.geometry_dir}")
        print(f"Travel time files created in: "
              f"{cfg.geometry_traveltime.traveltime_dir}")
        if cfg.geometry_traveltime.generate_source_binaries:
            print(f"Source binary files created in: "
                  f"{cfg.geometry_traveltime.source_dir}")


def run_velocity_model(cfg: DictConfig, events_cart, stations_cart) -> None:
    """Create velocity model and binary files.

    Args:
        cfg: Configuration settings.
        events_cart: DataFrame with events in Cartesian coordinates.
        stations_cart: DataFrame with stations in Cartesian coordinates.
    """
    if not cfg.initial_velocity_model.enabled:
        return

    print("\n" + "="*60)
    print("CREATING INITIAL VELOCITY MODEL (IASP91)")
    print("="*60)

    with ProcessTimer("Velocity Model Creation"):
        create_velocity_model(cfg, events_cart, stations_cart)

        print(f"\nVelocity model CSV: {cfg.initial_velocity_model.output_csv}")

        phase_type = cfg.initial_velocity_model.get('phase_type', 'both')
        if phase_type in ['both', 'p']:
            print(f"Vp binary file: {cfg.initial_velocity_model.vp_binary}")
        if phase_type in ['both', 's']:
            print(f"Vs binary file: {cfg.initial_velocity_model.vs_binary}")


def run_visualization(cfg: DictConfig, events_df, stations_df,
                     picks_df) -> None:
    """Create visualizations.

    Args:
        cfg: Configuration settings.
        events_df: DataFrame with events (geodetic coordinates).
        stations_df: DataFrame with stations (geodetic coordinates).
        picks_df: DataFrame with picks.
    """
    if not cfg.visualization.enabled:
        return

    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    # Station and event visualization
    if 'station_event_vis' in cfg.visualization:
        print("\nCreating station and event visualization...")
        with ProcessTimer("Station/Event Visualization"):
            create_station_event_visualization(
                cfg, events_df, stations_df, picks_df
            )
            print(f"  Station/event visualizations saved to: "
                  f"{cfg.visualization.station_event_vis.output_dir}")

    # Velocity model visualization
    if ('velocity_model_vis' in cfg.visualization and
        cfg.visualization.velocity_model_vis.enabled and
        cfg.initial_velocity_model.enabled):
        print("\nCreating velocity model visualization...")
        with ProcessTimer("Velocity Model Visualization"):
            create_velocity_model_visualization(
                cfg,
                cfg.initial_velocity_model.output_csv
            )
            print(f"  Velocity model visualizations saved to: "
                  f"{cfg.visualization.velocity_model_vis.output_dir}")


def setup_output_directories(config: DictConfig) -> None:
    """Create necessary output directories.

    Args:
        config: Configuration settings.
    """
    output_dir = Path(config.preprocessing.output_dir)
    ensure_directory_exists(output_dir)

    # Create subdirectories for different processing types
    subdirs = ['coordinate_transform', 'velocity_models', 'geometry',
              'traveltime', 'model', 'visualizations']
    for subdir in subdirs:
        ensure_directory_exists(output_dir / subdir)


def print_configuration_summary(config: DictConfig) -> None:
    """Print a summary of what will be processed.

    Args:
        config: Configuration settings.
    """
    print("\n" + "="*60)
    print("LATTE PREPROCESSING CONFIGURATION SUMMARY")
    print("="*60)

    print(f"\nOutput directory: {config.preprocessing.output_dir}")

    # Data source
    if config.data.use_hdf5:
        print(f"Data source: HDF5 file")
        print(f"  HDF5 file: {config.data.hdf5_file}")
    else:
        print(f"Data source: QuakeML files")
        print(f"  QuakeML directory: {config.data.quakeml_dir}")
        print(f"  Station file: {config.data.station_file}")

    # Origin time configuration
    origin_mode = config.origin_time.get('mode', 'fixed')
    print(f"\nOrigin time mode: {origin_mode}")
    if origin_mode == 'fixed':
        fixed_value = config.origin_time.get('fixed_value', 0.0)
        print(f"  Fixed origin time value: {fixed_value} seconds")
    elif origin_mode == 'unix_epoch':
        print(f"  Using Unix epoch timestamps")

    # Filters
    print("\nApplied filters:")
    filters = config.filters
    if filters.magnitude_min is not None or filters.magnitude_max is not None:
        print(f"  Magnitude: [{filters.magnitude_min}, "
              f"{filters.magnitude_max}]")
    if (filters.lat_min is not None or filters.lat_max is not None or
        filters.lon_min is not None or filters.lon_max is not None):
        print(f"  Geographic: lat[{filters.lat_min}, {filters.lat_max}], "
              f"lon[{filters.lon_min}, {filters.lon_max}]")
    if filters.depth_min is not None or filters.depth_max is not None:
        print(f"  Depth: [{filters.depth_min}, {filters.depth_max}] km")
    if (filters.epicentral_distance_min is not None or
        filters.epicentral_distance_max is not None):
        print(f"  Epicentral distance: [{filters.epicentral_distance_min}, "
              f"{filters.epicentral_distance_max}] km")
    if (filters.hypocentral_distance_min is not None or
        filters.hypocentral_distance_max is not None):
        print(f"  Hypocentral distance: [{filters.hypocentral_distance_min}, "
              f"{filters.hypocentral_distance_max}] km")

    # Enabled processing steps
    print("\nEnabled processing steps:")
    if config.coordinate_transform.enabled:
        print(f"  1. Coordinate transformation (flattening)")
        print(f"     Origin: ({config.coordinate_transform.origin_lon}°, "
              f"{config.coordinate_transform.origin_lat}°, "
              f"{config.coordinate_transform.origin_depth_km} km)")
        print(f"     Grid spacing: {config.coordinate_transform.grid_spacing_km} km")
    if config.geometry_traveltime.enabled:
        phase_type = config.geometry_traveltime.get('phase_type', 'both')
        print(f"  2. Geometry and travel time file creation "
              f"(phase_type: {phase_type})")
    if config.initial_velocity_model.enabled:
        print(f"  3. Initial velocity model (IASP91)")
        print(f"     Max depth: {config.initial_velocity_model.max_depth_km} km")
        phase_type = config.initial_velocity_model.get('phase_type', 'both')
        print(f"     Phase type: {phase_type}")
    if config.visualization.enabled:
        print(f"  4. Visualization generation")

    # Parallel processing
    num_workers = (config.parallel.num_workers if config.parallel.num_workers
                  else "all CPU cores")
    print(f"\nParallel processing: {num_workers}")


def run_preprocessing_pipeline(config: DictConfig) -> None:
    """Run the complete preprocessing pipeline.

    Args:
        config: Configuration settings.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting LATTE preprocessing pipeline")

    # Print configuration summary
    print_configuration_summary(config)

    # Setup output directories
    setup_output_directories(config)

    # Run processing steps in order
    try:
        # Step 1: Load data
        events_df, stations_df, arrivals_df, picks_df = run_data_loading(
            config
        )

        if len(events_df) == 0:
            logger.error("No events loaded. Cannot proceed.")
            print("\nNo events loaded. Please check your data source and "
                  "filters.")
            return

        # Step 2: Transform coordinates (flattening)
        events_cart, stations_cart = run_coordinate_transform(
            config, events_df, stations_df
        )

        # Step 3: Geometry and travel time files
        run_geometry_traveltime(config, events_cart, stations_cart, picks_df)

        # Step 4: Velocity model and binaries
        run_velocity_model(config, events_cart, stations_cart)

        # Step 5: Visualizations
        run_visualization(config, events_df, stations_df, picks_df)

        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"\nAll outputs saved to: {config.preprocessing.output_dir}")

    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        print("Preprocessing pipeline failed. Check the logs for details.")
        raise


def validate_preprocessing_config(cfg: DictConfig) -> None:
    """Validate preprocessing configuration.

    Args:
        cfg: Configuration settings.

    Raises:
        ValueError: If configuration is invalid.
    """
    logger = logging.getLogger(__name__)

    # Validate origin_time configuration
    origin_mode = cfg.origin_time.get('mode', 'fixed')
    if origin_mode not in ['fixed', 'unix_epoch']:
        logger.error(f"Invalid origin_time.mode: {origin_mode}. "
                    f"Must be 'fixed' or 'unix_epoch'")
        raise ValueError(f"Invalid origin_time.mode: {origin_mode}")

    if origin_mode == 'fixed':
        fixed_value = cfg.origin_time.get('fixed_value')
        if fixed_value is None:
            logger.error("origin_time.fixed_value is required when "
                        "mode is 'fixed'")
            raise ValueError("origin_time.fixed_value is required when "
                           "mode is 'fixed'")

    # Validate phase_type in geometry_traveltime
    if cfg.geometry_traveltime.enabled:
        phase_type = cfg.geometry_traveltime.get('phase_type', 'both')
        if phase_type not in ['both', 'p', 's']:
            logger.error(f"Invalid geometry_traveltime.phase_type: "
                        f"{phase_type}. Must be 'both', 'p', or 's'")
            raise ValueError(f"Invalid phase_type: {phase_type}")

    # Validate phase_type in velocity model
    if cfg.initial_velocity_model.enabled:
        phase_type = cfg.initial_velocity_model.get('phase_type', 'both')
        if phase_type not in ['both', 'p', 's']:
            logger.error(f"Invalid initial_velocity_model.phase_type: "
                        f"{phase_type}. Must be 'both', 'p', or 's'")
            raise ValueError(f"Invalid phase_type: {phase_type}")

    logger.info("Configuration validation passed")


@hydra.main(version_base=None, config_path="../configs/preprocess",
           config_name="default")
def main(cfg: DictConfig) -> None:
    """Main entry point for preprocessing with Hydra.

    Args:
        cfg: Hydra configuration.
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

    # Run full pipeline
    run_preprocessing_pipeline(cfg)


if __name__ == "__main__":
    main()
