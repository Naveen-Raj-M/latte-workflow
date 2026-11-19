"""Main preprocessing module for LATTE tomography with QuakeML data.

This module orchestrates all preprocessing tasks for seismic data from
QuakeML files or HDF5 format, using Hydra configuration.

Workflow order:
0. Parse the QuakeML files ONCE when any enabled step needs them
   (raw data visualization, HDF5 conversion, or preprocessing from QuakeML)
1. Raw data visualization (optional, own filters)
2. QuakeML to HDF5 conversion (optional, own filters)
3. Preprocessing pipeline (global filters), if preprocessing.run_pipeline:
   load data (QuakeML or HDF5), check coordinates (longitude, WGS84 geodetic latitude,
   depth below MSL on a spherical Earth), create geometry and travel time files,
   create velocity model and binaries, final data visualization
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Import preprocessing functions
from preprocess.data_loader import load_seismic_data
from preprocess.coordinate_transformation import transform_coordinates, check_grid_configuration
from preprocess.geometry_traveltime import create_geometry_and_traveltime_files
from preprocess.initial_velocity_model import create_velocity_model
from preprocess.hdf5_conversion import convert_to_hdf5
from preprocess.quakeml_loader import CatalogData, load_quakeml_catalog
from preprocess.raw_data_visualization import PLOT_FUNCTIONS, create_raw_data_visualization
from preprocess.visualization import (
    create_station_event_visualization,
    create_velocity_model_visualization
)

# Import utilities
from utils.logging_utils import setup_logging, ProcessTimer
from utils.io_utils import ensure_directory_exists
from utils.column_utils import normalize_coordinate_columns


def run_data_loading(cfg: DictConfig, raw: Optional[CatalogData] = None):
    """Load seismic data from QuakeML or HDF5.

    Args:
        cfg: Configuration settings.
        raw: Already parsed QuakeML catalog, reused when data.use_hdf5 is false.

    Returns:
        Tuple of (events_df, stations_df, arrivals_df, picks_df).
    """
    print("\n" + "="*60)
    print("DATA LOADING")
    print("="*60)

    with ProcessTimer("Data Loading"):
        events_df, stations_df, arrivals_df, picks_df = load_seismic_data(cfg, raw)

        print(f"\nLoaded data summary:")
        print(f"  Events: {len(events_df)}")
        print(f"  Stations: {_unique_positions(stations_df)} (unique positions)")
        print(f"  Arrivals: {len(arrivals_df)}")
        print(f"  Picks: {len(picks_df)}")

    return events_df, stations_df, arrivals_df, picks_df


def _unique_positions(df: Optional[pd.DataFrame]) -> int:
    """Number of distinct (longitude, latitude, depth) positions in a table."""
    if df is None or df.empty:
        return 0
    return len(df.drop_duplicates(subset=['longitude', 'latitude', 'depth']))


def run_coordinate_transform(cfg: DictConfig, events_df, stations_df, picks_df):
    """Check the coordinates of events and stations (longitude, geodetic latitude, depth).

    Args:
        cfg: Configuration settings.
        events_df: DataFrame with events in geodetic coordinates.
        stations_df: DataFrame with stations in geodetic coordinates.
        picks_df: DataFrame with picks.

    Returns:
        Tuple of (events_df, stations_df) with the columns longitude, latitude, depth.
    """
    if not cfg.coordinate_transform.enabled:
        return events_df, stations_df

    print("\n" + "="*60)
    print("COORDINATES (SPHERICAL EARTH)")
    print("="*60)

    with ProcessTimer("Coordinate Check"):
        events_out, stations_out = transform_coordinates(
            cfg, events_df, stations_df
        )

        # Also save picks to coordinate_transform output directory for independent visualization
        if picks_df is not None and not picks_df.empty:
            picks_csv = Path(cfg.coordinate_transform.picks_csv)
            ensure_directory_exists(picks_csv.parent)
            picks_df.to_csv(picks_csv, index=False)
            print(f"  Picks: {len(picks_df)} saved to {picks_csv}")

        print(f"\nChecked coordinates:")
        print(f"  Events: {len(events_out)}")
        print(f"  Stations: {_unique_positions(stations_out)} (unique positions)")

    return events_out, stations_out


def run_geometry_traveltime(cfg: DictConfig, events_df, stations_df,
                            picks_df) -> Tuple[Optional[pd.DataFrame],
                                               Optional[pd.DataFrame],
                                               Optional[pd.DataFrame]]:
    """Create geometry and travel time files.

    Args:
        cfg: Configuration settings.
        events_df: DataFrame with events (longitude, latitude, depth).
        stations_df: DataFrame with stations (longitude, latitude, depth).
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
            cfg, events_df, stations_df, picks_df
        )

        if filtered_events is not None and len(filtered_events) > 0:
            print(f"\nAfter geometry/traveltime filtering:")
            print(f"  Sources: {len(filtered_events)}")
            print(f"  Unique receivers (unique positions): {_unique_positions(filtered_stations)}")
            print(f"  Picks: {len(filtered_picks)}")

        print(f"\nGeometry files created in: "
              f"{cfg.geometry_traveltime.geometry_dir}")
        print(f"Travel time files created in: "
              f"{cfg.geometry_traveltime.traveltime_dir}")
        if cfg.geometry_traveltime.generate_source_binaries:
            print(f"Source binary files created in: "
                  f"{cfg.geometry_traveltime.source_dir}")

    return filtered_events, filtered_stations, filtered_picks


def run_velocity_model(cfg: DictConfig, events_df, stations_df) -> None:
    """Create velocity model and binary files.

    Args:
        cfg: Configuration settings.
        events_df: DataFrame with events (longitude, latitude, depth).
        stations_df: DataFrame with stations (longitude, latitude, depth).
    """
    if not cfg.initial_velocity_model.enabled:
        return

    print("\n" + "="*60)
    print("CREATING INITIAL VELOCITY MODEL")
    print("="*60)

    with ProcessTimer("Velocity Model Creation"):
        create_velocity_model(cfg, events_df, stations_df)

        print(f"\nVelocity model CSV: {cfg.initial_velocity_model.output_csv}")

        phase_type = cfg.initial_velocity_model.get('phase_type', 'both')
        if phase_type in ['both', 'p']:
            print(f"Vp binary file: {cfg.initial_velocity_model.vp_binary}")
        if phase_type in ['both', 's']:
            print(f"Vs binary file: {cfg.initial_velocity_model.vs_binary}")
        latte_grid_file = cfg.coordinate_transform.get('latte_grid_file', None)
        if latte_grid_file:
            print(f"LATTE grid parameters: {latte_grid_file}")


def load_existing_data_for_visualization(cfg: DictConfig) -> Tuple[Optional[pd.DataFrame],
                                                                   Optional[pd.DataFrame],
                                                                   Optional[pd.DataFrame]]:
    """Load data from existing CSV files for visualization.

    This function attempts to load previously saved data files to enable
    visualization without running the entire preprocessing pipeline.

    The files hold only the data that passed all geometry processing filters
    (velocity validation, P&S pair requirements, etc.) and were written to the
    output geometry/traveltime files.

    Args:
        cfg: Configuration settings.

    Returns:
        Tuple of (events_df, stations_df, picks_df). Any or all can be None if
        files don't exist.
    """
    logger = logging.getLogger(__name__)

    coord_cfg = cfg.coordinate_transform
    events_path = Path(coord_cfg.events_csv)
    stations_path = Path(coord_cfg.stations_csv)
    picks_path = Path(coord_cfg.picks_csv)

    events_df = None
    stations_df = None
    picks_df = None

    if events_path.exists() and stations_path.exists():
        try:
            events_df = normalize_coordinate_columns(pd.read_csv(events_path), context=events_path.name)
            stations_df = normalize_coordinate_columns(pd.read_csv(stations_path), context=stations_path.name)
            logger.info(f"Loaded data: {len(events_df)} events, {_unique_positions(stations_df)} stations (unique positions)")
        except Exception as e:
            logger.warning(f"Failed to load event/station CSVs: {e}")

    if picks_path.exists():
        try:
            picks_df = pd.read_csv(picks_path)
            logger.info(f"Loaded picks: {len(picks_df)} picks")
        except Exception as e:
            logger.warning(f"Failed to load picks CSV: {e}")

    return events_df, stations_df, picks_df


def run_visualization(cfg: DictConfig, events_df=None, stations_df=None,
                     picks_df=None, load_from_files=False) -> None:
    """Create visualizations.

    Args:
        cfg: Configuration settings.
        events_df: DataFrame with events (geodetic coordinates). If None and load_from_files=True, will try to load.
        stations_df: DataFrame with stations (geodetic coordinates). If None and load_from_files=True, will try to load.
        picks_df: DataFrame with picks. If None and load_from_files=True, will try to load.
        load_from_files: If True, attempt to load data from existing CSV files when data is None.
    """
    if not cfg.final_data_visualization.enabled:
        return

    logger = logging.getLogger(__name__)

    # If load_from_files is True and data is None, try loading from CSVs
    if load_from_files and (events_df is None or stations_df is None):
        logger.info("Attempting to load existing data for visualization...")
        events_df, stations_df, picks_df = load_existing_data_for_visualization(cfg)

        if events_df is None and stations_df is None:
            logger.warning("No existing data files found for visualization")
            print("\nWARNING: No existing data files found. Run the full workflow first to generate data.")
            return

    print("\n" + "="*60)
    print("CREATING FINAL DATA VISUALIZATIONS")
    print("="*60)

    # Station and event visualization
    final_vis = cfg.final_data_visualization
    if ('station_event_vis' in final_vis and
        final_vis.station_event_vis.get('enabled', True)):
        if events_df is not None and stations_df is not None:
            print("\nCreating station and event visualization...")
            with ProcessTimer("Station/Event Visualization"):
                create_station_event_visualization(
                    cfg, events_df, stations_df, picks_df
                )
                print(f"  Station/event visualizations saved to: "
                      f"{final_vis.station_event_vis.output_dir}")
        else:
            logger.warning("Station/event visualization enabled but coordinate data not available")

    # Velocity model visualization
    if ('velocity_model_vis' in final_vis and
        final_vis.velocity_model_vis.enabled and
        cfg.initial_velocity_model.enabled):
        print("\nCreating velocity model visualization...")
        with ProcessTimer("Velocity Model Visualization"):
            # Determine which velocity file to use based on output_format
            output_format = cfg.initial_velocity_model.get('output_format', 'csv')
            if output_format in ['hdf5', 'both']:
                # Prefer HDF5 format (more efficient for large grids)
                velocity_file = cfg.initial_velocity_model.output_hdf5
            else:
                # Use CSV format
                velocity_file = cfg.initial_velocity_model.output_csv

            create_velocity_model_visualization(cfg, velocity_file)
            print(f"  Velocity model visualizations saved to: "
                  f"{final_vis.velocity_model_vis.output_dir}")


def setup_output_directories(config: DictConfig) -> None:
    """Create necessary output directories.

    Args:
        config: Configuration settings.
    """
    output_dir = Path(config.preprocessing.output_dir)
    ensure_directory_exists(output_dir)

    # Create subdirectories for different processing types
    subdirs = ['coordinate_transform', 'geometry', 'model', 'visualizations']
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
    needs_quakeml = quakeml_required(config)
    if config.raw_data_visualization.enabled:
        print(f"Raw data visualization: enabled -> {config.raw_data_visualization.output_dir}")
    if config.hdf5_conversion.enabled:
        print(f"HDF5 conversion: enabled -> {config.hdf5_conversion.output_file}")
    if not config.preprocessing.run_pipeline:
        print("Preprocessing pipeline: skipped (preprocessing.run_pipeline=false)")
    if needs_quakeml:
        print("QuakeML files are parsed once and shared by all enabled steps")
    if not config.preprocessing.run_pipeline:
        return
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
    coord_cfg = config.coordinate_transform
    print(f"\nSpherical Earth: radius {coord_cfg.get('earth_radius_km', 6371.0)} km")
    print(f"  Grid origin (west, south, top): ({coord_cfg.origin_lon}°, "
          f"{coord_cfg.origin_lat}°, {coord_cfg.origin_depth_km} km)")
    dlon, dlat, ddepth = coord_cfg.grid_spacing
    print(f"  Grid spacing: {dlon}° lon, {dlat}° lat, {ddepth} km depth")
    if coord_cfg.enabled:
        print(f"  1. Coordinate check")
    if config.geometry_traveltime.enabled:
        phase_type = config.geometry_traveltime.get('phase_type', 'both')
        print(f"  2. Geometry and travel time file creation "
              f"(phase_type: {phase_type})")
    if config.initial_velocity_model.enabled:
        print(f"  3. Initial velocity model")
        print(f"     Max depth: {config.initial_velocity_model.max_depth_km} km")
        phase_type = config.initial_velocity_model.get('phase_type', 'both')
        print(f"     Phase type: {phase_type}")
    if config.final_data_visualization.enabled:
        print(f"  4. Final data visualization")

    # Parallel processing
    num_workers = (config.parallel.num_workers if config.parallel.num_workers
                  else "all CPU cores")
    print(f"\nParallel processing: {num_workers}")


def quakeml_required(config: DictConfig) -> bool:
    """True when any enabled step reads the QuakeML files."""
    return bool(config.raw_data_visualization.enabled or config.hdf5_conversion.enabled or
                (config.preprocessing.run_pipeline and not config.data.use_hdf5))


def run_preprocessing_pipeline(config: DictConfig) -> None:
    """Run the enabled workflow steps.

    Args:
        config: Configuration settings.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting LATTE preprocessing pipeline")

    # Print configuration summary
    print_configuration_summary(config)

    # Setup output directories
    setup_output_directories(config)

    try:
        # Parse the QuakeML files once for every step that needs them
        raw = None
        if quakeml_required(config):
            print("\n" + "="*60)
            print("QUAKEML LOADING")
            print("="*60)
            with ProcessTimer("QuakeML Loading"):
                raw = load_quakeml_catalog(config)
            if raw.events.empty:
                logger.error("No events parsed from QuakeML. Cannot proceed.")
                print("\nNo events parsed. Please check data.quakeml_dir and data.years.")
                return

        if config.raw_data_visualization.enabled:
            print("\n" + "="*60)
            print("RAW DATA VISUALIZATION")
            print("="*60)
            with ProcessTimer("Raw Data Visualization"):
                out_dir = create_raw_data_visualization(config, raw)
            if out_dir is not None:
                print(f"\nRaw data visualizations saved to: {out_dir}")

        if config.hdf5_conversion.enabled:
            print("\n" + "="*60)
            print("QUAKEML TO HDF5 CONVERSION")
            print("="*60)
            with ProcessTimer("HDF5 Conversion"):
                convert_to_hdf5(config, raw)

        if not config.preprocessing.run_pipeline:
            print("\nPreprocessing pipeline skipped (preprocessing.run_pipeline=false).")
            return

        # Step 1: Load data
        events_df, stations_df, arrivals_df, picks_df = run_data_loading(
            config, raw
        )

        if len(events_df) == 0:
            logger.error("No events loaded. Cannot proceed.")
            print("\nNo events loaded. Please check your data source and "
                  "filters.")
            return

        # Step 2: Check coordinates (spherical Earth)
        events_df, stations_df = run_coordinate_transform(
            config, events_df, stations_df, picks_df
        )

        # Step 3: Geometry and travel time files
        filtered_events, filtered_stations, filtered_picks = run_geometry_traveltime(
            config, events_df, stations_df, picks_df
        )

        # Step 4: Velocity model and binaries
        run_velocity_model(config, events_df, stations_df)

        # Save fully-filtered data to CSV
        has_filtered = filtered_events is not None and len(filtered_events) > 0
        if has_filtered:
            coord_cfg = config.coordinate_transform
            outputs = ((filtered_events, Path(coord_cfg.events_csv)),
                       (filtered_stations, Path(coord_cfg.stations_csv)),
                       (filtered_picks, Path(coord_cfg.picks_csv)))
            print(f"\nFiltered data saved:")
            for table, path in outputs:
                ensure_directory_exists(path.parent)
                table.to_csv(path, index=False)
                print(f"  {path}")

        # Step 5: Visualizations
        # Use filtered data if available (from geometry processing), otherwise use original data
        # This ensures visualizations show only the data that was written to output files
        if has_filtered:
            events_df_for_vis = filtered_events
            stations_df_for_vis = filtered_stations
            picks_df_for_vis = filtered_picks

            print(f"\nVisualization will use FILTERED data:")
            print(f"  Events: {len(events_df_for_vis)} (filtered from {len(events_df)})")
            print(f"  Stations: {_unique_positions(stations_df_for_vis)} (unique positions, filtered from {_unique_positions(stations_df)})")
            print(f"  Picks: {len(picks_df_for_vis)} (filtered from {len(picks_df)})")
        else:
            # Fallback to original data if geometry processing was disabled
            events_df_for_vis = events_df
            stations_df_for_vis = stations_df
            picks_df_for_vis = picks_df

        run_visualization(config, events_df_for_vis, stations_df_for_vis, picks_df_for_vis)

        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)

        print(f"\nFinal dataset summary:")
        if has_filtered:
            print(f"  Sources (events after all filtering): {len(filtered_events)}")
            print(f"  Unique receivers (unique positions): {_unique_positions(filtered_stations)}")
        else:
            print(f"  Sources: {len(events_df)} (geometry filtering disabled)")
            print(f"  Unique receivers (unique positions): {_unique_positions(stations_df)}")

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

    # Validate the grid origin and spacing (spherical Earth)
    check_grid_configuration(cfg.coordinate_transform)

    # Validate raw data visualization plot selection
    valid_plots = set(PLOT_FUNCTIONS)
    unknown = set(cfg.raw_data_visualization.plots_to_generate) - valid_plots
    if unknown:
        raise ValueError(f"Unknown raw_data_visualization.plots_to_generate entries: "
                         f"{sorted(unknown)}. Valid: {sorted(valid_plots)}")

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
