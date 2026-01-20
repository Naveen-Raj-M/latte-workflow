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
from typing import Tuple, Optional

import hydra
import pandas as pd
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
from preprocess.ray_density_visualization import create_ray_density_visualization

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


def run_coordinate_transform(cfg: DictConfig, events_df, stations_df, picks_df):
    """Transform coordinates from geodetic to Cartesian.

    Args:
        cfg: Configuration settings.
        events_df: DataFrame with events in geodetic coordinates.
        stations_df: DataFrame with stations in geodetic coordinates.
        picks_df: DataFrame with picks.

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

        # Also save picks to coordinate_transform output directory for independent visualization
        if picks_df is not None and not picks_df.empty:
            output_dir = Path(cfg.coordinate_transform.output_dir)
            ensure_directory_exists(output_dir)
            picks_csv = output_dir / 'picks_cartesian.csv'
            picks_df.to_csv(picks_csv, index=False)
            print(f"  Picks: {len(picks_df)} saved to {picks_csv}")

        print(f"\nTransformed coordinates:")
        print(f"  Events: {len(events_cart)} to Cartesian")
        print(f"  Stations: {len(stations_cart)} to Cartesian")

    return events_cart, stations_cart


def run_geometry_traveltime(cfg: DictConfig, events_cart, stations_cart,
                            picks_df) -> Tuple[Optional[pd.DataFrame],
                                               Optional[pd.DataFrame],
                                               Optional[pd.DataFrame]]:
    """Create geometry and travel time files.

    Args:
        cfg: Configuration settings.
        events_cart: DataFrame with events in Cartesian coordinates.
        stations_cart: DataFrame with stations in Cartesian coordinates.
        picks_df: DataFrame with picks.

    Returns:
        Tuple of (filtered_events_df, filtered_stations_df, filtered_picks_df)
        containing only the data that passed all filtering. Returns (None, None, None)
        if geometry_traveltime is disabled.
    """
    if not cfg.geometry_traveltime.enabled:
        return None, None, None

    print("\n" + "="*60)
    print("CREATING GEOMETRY AND TRAVEL TIME FILES")
    print("="*60)

    with ProcessTimer("Geometry and Travel Time Processing"):
        filtered_events, filtered_stations, filtered_picks = create_geometry_and_traveltime_files(
            cfg, events_cart, stations_cart, picks_df
        )

        print(f"\nGeometry files created in: "
              f"{cfg.geometry_traveltime.geometry_dir}")
        print(f"Travel time files created in: "
              f"{cfg.geometry_traveltime.traveltime_dir}")
        if cfg.geometry_traveltime.generate_source_binaries:
            print(f"Source binary files created in: "
                  f"{cfg.geometry_traveltime.source_dir}")

    return filtered_events, filtered_stations, filtered_picks


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


def load_existing_data_for_visualization(cfg: DictConfig) -> Tuple[Optional[pd.DataFrame],
                                                                   Optional[pd.DataFrame],
                                                                   Optional[pd.DataFrame],
                                                                   Optional[pd.DataFrame],
                                                                   Optional[pd.DataFrame]]:
    """Load data from existing CSV files for visualization.

    This function attempts to load previously saved data files to enable
    visualization without running the entire preprocessing pipeline.

    IMPORTANT: This function prefers FILTERED data files if they exist.
    Filtered data only includes picks that passed all geometry processing
    filters (velocity validation, P&S pair requirements, etc.) and were
    written to the output geometry/traveltime files.

    Args:
        cfg: Configuration settings.

    Returns:
        Tuple of (events_df, stations_df, events_cart, stations_cart, picks_df).
        Any or all can be None if files don't exist.
    """
    logger = logging.getLogger(__name__)

    # Define paths - prefer filtered data if available
    # Filtered geodetic coordinates
    events_geo_filtered_path = Path(cfg.data.output_base_dir) / "events_geodetic_filtered.csv"
    stations_geo_filtered_path = Path(cfg.data.output_base_dir) / "stations_geodetic_filtered.csv"
    # Original geodetic coordinates (fallback)
    events_geo_path = Path(cfg.data.output_base_dir) / "events_geodetic.csv"
    stations_geo_path = Path(cfg.data.output_base_dir) / "stations_geodetic.csv"

    # Filtered Cartesian coordinates
    events_cart_filtered_path = Path(cfg.coordinate_transform.output_dir) / "events_cartesian_filtered.csv"
    stations_cart_filtered_path = Path(cfg.coordinate_transform.output_dir) / "stations_cartesian_filtered.csv"
    picks_filtered_path = Path(cfg.coordinate_transform.output_dir) / "picks_filtered.csv"
    # Original Cartesian coordinates (fallback)
    events_cart_path = Path(cfg.coordinate_transform.output_dir) / "events_cartesian.csv"
    stations_cart_path = Path(cfg.coordinate_transform.output_dir) / "stations_cartesian.csv"
    picks_path = Path(cfg.coordinate_transform.output_dir) / "picks_cartesian.csv"

    events_df = None
    stations_df = None
    events_cart = None
    stations_cart = None
    picks_df = None

    # Load geodetic - prefer filtered data
    if events_geo_filtered_path.exists() and stations_geo_filtered_path.exists():
        try:
            events_df = pd.read_csv(events_geo_filtered_path)
            stations_df = pd.read_csv(stations_geo_filtered_path)
            logger.info(f"Loaded FILTERED geodetic data: {len(events_df)} events, {len(stations_df)} stations")
        except Exception as e:
            logger.warning(f"Failed to load filtered geodetic CSVs: {e}")
            events_df = None
            stations_df = None

    # Fallback to original geodetic if filtered not available
    if events_df is None and events_geo_path.exists() and stations_geo_path.exists():
        try:
            events_df = pd.read_csv(events_geo_path)
            stations_df = pd.read_csv(stations_geo_path)
            logger.info(f"Loaded original geodetic data: {len(events_df)} events, {len(stations_df)} stations")
            logger.warning("Using UNFILTERED geodetic data - visualization may show more data than output files")
        except Exception as e:
            logger.warning(f"Failed to load geodetic CSVs: {e}")
            events_df = None
            stations_df = None

    # Load Cartesian - prefer filtered data
    if events_cart_filtered_path.exists() and stations_cart_filtered_path.exists():
        try:
            events_cart = pd.read_csv(events_cart_filtered_path)
            stations_cart = pd.read_csv(stations_cart_filtered_path)
            logger.info(f"Loaded FILTERED Cartesian data: {len(events_cart)} events, {len(stations_cart)} stations")
        except Exception as e:
            logger.warning(f"Failed to load filtered Cartesian CSVs: {e}")
            events_cart = None
            stations_cart = None

    # Fallback to original Cartesian if filtered not available
    if events_cart is None and events_cart_path.exists() and stations_cart_path.exists():
        try:
            events_cart = pd.read_csv(events_cart_path)
            stations_cart = pd.read_csv(stations_cart_path)
            logger.info(f"Loaded original Cartesian data: {len(events_cart)} events, {len(stations_cart)} stations")
            logger.warning("Using UNFILTERED Cartesian data - visualization may show more data than output files")
        except Exception as e:
            logger.warning(f"Failed to load Cartesian CSVs: {e}")
            events_cart = None
            stations_cart = None

    # Load picks - prefer filtered data
    if picks_filtered_path.exists():
        try:
            picks_df = pd.read_csv(picks_filtered_path)
            logger.info(f"Loaded FILTERED picks: {len(picks_df)} picks")
        except Exception as e:
            logger.warning(f"Failed to load filtered picks CSV: {e}")
            picks_df = None

    # Fallback to original picks if filtered not available
    if picks_df is None and picks_path.exists():
        try:
            picks_df = pd.read_csv(picks_path)
            logger.info(f"Loaded original picks: {len(picks_df)} picks")
            logger.warning("Using UNFILTERED picks - visualization may show more data than output files")
        except Exception as e:
            logger.warning(f"Failed to load picks CSV: {e}")
            picks_df = None

    return events_df, stations_df, events_cart, stations_cart, picks_df


def run_visualization(cfg: DictConfig, events_df=None, stations_df=None,
                     picks_df=None, events_cart=None, stations_cart=None,
                     load_from_files=False) -> None:
    """Create visualizations.

    Args:
        cfg: Configuration settings.
        events_df: DataFrame with events (geodetic coordinates). If None and load_from_files=True, will try to load.
        stations_df: DataFrame with stations (geodetic coordinates). If None and load_from_files=True, will try to load.
        picks_df: DataFrame with picks. If None and load_from_files=True, will try to load.
        events_cart: DataFrame with events (Cartesian coordinates). If None and load_from_files=True, will try to load.
        stations_cart: DataFrame with stations (Cartesian coordinates). If None and load_from_files=True, will try to load.
        load_from_files: If True, attempt to load data from existing CSV files when data is None.
    """
    if not cfg.visualization.enabled:
        return

    logger = logging.getLogger(__name__)

    # If load_from_files is True and data is None, try loading from CSVs
    if load_from_files and (events_df is None or stations_df is None):
        logger.info("Attempting to load existing data for visualization...")
        events_df, stations_df, events_cart, stations_cart, picks_df = load_existing_data_for_visualization(cfg)

        if events_df is None and stations_df is None and events_cart is None:
            logger.warning("No existing data files found for visualization")
            print("\nWARNING: No existing data files found. Run the full workflow first to generate data.")
            return

    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    # Station and event visualization
    if ('station_event_vis' in cfg.visualization and
        cfg.visualization.station_event_vis.get('enabled', True)):
        if events_df is not None and stations_df is not None:
            print("\nCreating station and event visualization...")
            with ProcessTimer("Station/Event Visualization"):
                create_station_event_visualization(
                    cfg, events_df, stations_df, picks_df
                )
                print(f"  Station/event visualizations saved to: "
                      f"{cfg.visualization.station_event_vis.output_dir}")
        else:
            logger.warning("Station/event visualization enabled but geodetic coordinate data not available")

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

    # Ray density visualization
    if ('ray_density' in cfg.visualization and
        cfg.visualization.ray_density.enabled):
        # Check if we have Cartesian coordinates (requires coordinate_transform to be enabled)
        has_cartesian = (events_cart is not None and stations_cart is not None and
                        picks_df is not None and
                        'x' in events_cart.columns and 'y' in events_cart.columns and 'z' in events_cart.columns)

        if has_cartesian:
            print("\nCreating ray density visualization...")
            with ProcessTimer("Ray Density Visualization"):
                create_ray_density_visualization(
                    cfg, events_cart, stations_cart, picks_df
                )
                print(f"  Ray density visualizations saved to: "
                      f"{cfg.visualization.ray_density.output_dir}")
        else:
            if not cfg.coordinate_transform.enabled:
                logger.warning("Ray density visualization requires coordinate_transform to be enabled")
                print("  WARNING: Ray density visualization skipped - requires coordinate_transform.enabled: true")
            else:
                logger.warning("Ray density visualization enabled but Cartesian coordinate data or picks not available")


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
            config, events_df, stations_df, picks_df
        )

        # Step 3: Geometry and travel time files
        filtered_events_cart, filtered_stations_cart, filtered_picks = run_geometry_traveltime(
            config, events_cart, stations_cart, picks_df
        )

        # Step 4: Velocity model and binaries
        run_velocity_model(config, events_cart, stations_cart)

        # Save filtered data to CSV for standalone visualization runs
        if filtered_events_cart is not None and len(filtered_events_cart) > 0:
            output_dir = Path(config.coordinate_transform.output_dir)
            ensure_directory_exists(output_dir)

            # Save filtered Cartesian data
            filtered_events_cart.to_csv(output_dir / 'events_cartesian_filtered.csv', index=False)
            filtered_stations_cart.to_csv(output_dir / 'stations_cartesian_filtered.csv', index=False)
            filtered_picks.to_csv(output_dir / 'picks_filtered.csv', index=False)

            # Also save filtered geodetic data
            filtered_event_ids = set(filtered_events_cart['event_id'].unique())
            filtered_station_ids = set(filtered_stations_cart['station_id'].unique())
            events_df_filtered = events_df[events_df['event_id'].isin(filtered_event_ids)]
            stations_df_filtered = stations_df[stations_df['station_id'].isin(filtered_station_ids)]

            geo_output_dir = Path(config.data.output_base_dir)
            ensure_directory_exists(geo_output_dir)
            events_df_filtered.to_csv(geo_output_dir / 'events_geodetic_filtered.csv', index=False)
            stations_df_filtered.to_csv(geo_output_dir / 'stations_geodetic_filtered.csv', index=False)

            print(f"\nFiltered data saved:")
            print(f"  {output_dir / 'events_cartesian_filtered.csv'}")
            print(f"  {output_dir / 'stations_cartesian_filtered.csv'}")
            print(f"  {output_dir / 'picks_filtered.csv'}")
            print(f"  {geo_output_dir / 'events_geodetic_filtered.csv'}")
            print(f"  {geo_output_dir / 'stations_geodetic_filtered.csv'}")

        # Step 5: Visualizations
        # Use filtered data if available (from geometry processing), otherwise use original data
        # This ensures visualizations show only the data that was written to output files
        if filtered_events_cart is not None and len(filtered_events_cart) > 0:
            # Filter geodetic events/stations to match filtered Cartesian data
            filtered_event_ids = set(filtered_events_cart['event_id'].unique())
            filtered_station_ids = set(filtered_stations_cart['station_id'].unique())
            events_df_for_vis = events_df[events_df['event_id'].isin(filtered_event_ids)]
            stations_df_for_vis = stations_df[stations_df['station_id'].isin(filtered_station_ids)]
            picks_df_for_vis = filtered_picks
            events_cart_for_vis = filtered_events_cart
            stations_cart_for_vis = filtered_stations_cart

            print(f"\nVisualization will use FILTERED data:")
            print(f"  Events: {len(events_df_for_vis)} (filtered from {len(events_df)})")
            print(f"  Stations: {len(stations_df_for_vis)} (filtered from {len(stations_df)})")
            print(f"  Picks: {len(picks_df_for_vis)} (filtered from {len(picks_df)})")
        else:
            # Fallback to original data if geometry processing was disabled
            events_df_for_vis = events_df
            stations_df_for_vis = stations_df
            picks_df_for_vis = picks_df
            events_cart_for_vis = events_cart
            stations_cart_for_vis = stations_cart

        run_visualization(config, events_df_for_vis, stations_df_for_vis, picks_df_for_vis,
                         events_cart_for_vis, stations_cart_for_vis)

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

        # Validate mutual exclusivity of weight settings
        apply_default_weights = cfg.geometry_traveltime.get(
            'apply_default_weights', False
        )
        apply_azimuthal_weighting = cfg.geometry_traveltime.get(
            'apply_azimuthal_weighting', False
        )
        if apply_default_weights and apply_azimuthal_weighting:
            logger.error(
                "Configuration error: 'apply_default_weights' and "
                "'apply_azimuthal_weighting' cannot both be set to true. "
                "Please enable only one weighting method."
            )
            raise ValueError(
                "apply_default_weights and apply_azimuthal_weighting are "
                "mutually exclusive. Set only one to true."
            )

        # Validate azimuthal weighting settings if enabled
        if apply_azimuthal_weighting:
            bin_size = cfg.geometry_traveltime.get('azimuthal_bin_size_deg', 30)
            if bin_size <= 0 or bin_size > 360:
                logger.error(
                    f"Invalid azimuthal_bin_size_deg: {bin_size}. "
                    f"Must be > 0 and <= 360."
                )
                raise ValueError(
                    f"Invalid azimuthal_bin_size_deg: {bin_size}. "
                    f"Must be > 0 and <= 360."
                )
            if 360 % bin_size != 0:
                logger.warning(
                    f"azimuthal_bin_size_deg ({bin_size}) does not evenly "
                    f"divide 360. This may result in uneven bin coverage."
                )

            # Validate weight formula
            weight_formula = cfg.geometry_traveltime.get(
                'azimuthal_weight_formula', 'inverse_sqrt'
            )
            valid_formulas = ['inverse_sqrt', 'inverse']
            if weight_formula not in valid_formulas:
                logger.error(
                    f"Invalid azimuthal_weight_formula: '{weight_formula}'. "
                    f"Must be one of: {valid_formulas}"
                )
                raise ValueError(
                    f"Invalid azimuthal_weight_formula: '{weight_formula}'. "
                    f"Must be one of: {valid_formulas}"
                )

            # Validate weighting scale
            weighting_scale = cfg.geometry_traveltime.get(
                'azimuthal_weighting_scale', 'global'
            )
            valid_scales = ['global', 'per_event']
            if weighting_scale not in valid_scales:
                logger.error(
                    f"Invalid azimuthal_weighting_scale: '{weighting_scale}'. "
                    f"Must be one of: {valid_scales}"
                )
                raise ValueError(
                    f"Invalid azimuthal_weighting_scale: '{weighting_scale}'. "
                    f"Must be one of: {valid_scales}"
                )

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
