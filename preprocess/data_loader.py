"""Data loading functions for QuakeML and HDF5 seismic data.

This module provides functions to load seismic data from either QuakeML files
or pre-processed HDF5 files, with support for parallel processing and filtering.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from obspy.geodetics import gps2dist_azimuth
from omegaconf import DictConfig


def load_station_inventory(station_file: str) -> Dict[str, Dict[str, float]]:
    """Load station inventory from custom format file.

    Format: ID lat lon elevation
    - Space-separated values
    - elevation in meters, positive = above MSL

    Args:
        station_file: Path to station inventory file.

    Returns:
        Dictionary mapping station_id to {lat, lon, elevation}.
    """
    stations = {}

    if station_file is None or not os.path.exists(station_file):
        return stations

    with open(station_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 4:
                station_id = parts[0]
                lat = float(parts[1])
                lon = float(parts[2])
                elevation = float(parts[3])  # meters, positive = above MSL

                stations[station_id] = {
                    'lat': lat,
                    'lon': lon,
                    'elevation': elevation
                }

    return stations


def apply_filters_to_dataframes(events_df: pd.DataFrame, stations_df: pd.DataFrame,
                                  arrivals_df: pd.DataFrame, picks_df: pd.DataFrame,
                                  config: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                                pd.DataFrame, pd.DataFrame]:
    """Apply filters to loaded dataframes.

    Args:
        events_df: DataFrame with events.
        stations_df: DataFrame with stations.
        arrivals_df: DataFrame with arrivals.
        picks_df: DataFrame with picks.
        config: Configuration settings.

    Returns:
        Tuple of (filtered_events_df, filtered_stations_df,
                  filtered_arrivals_df, filtered_picks_df).
    """
    # Check if we have any filters to apply
    filters = config.filters
    has_any_filter = any([
        filters.magnitude_min is not None,
        filters.magnitude_max is not None,
        filters.lat_min is not None,
        filters.lat_max is not None,
        filters.lon_min is not None,
        filters.lon_max is not None,
        filters.depth_min is not None,
        filters.depth_max is not None,
        filters.hypocentral_distance_min is not None,
        filters.hypocentral_distance_max is not None,
        filters.epicentral_distance_min is not None,
        filters.epicentral_distance_max is not None,
    ])

    if not has_any_filter:
        return events_df, stations_df, arrivals_df, picks_df

    # 1. Filter events based on geographic, magnitude, and depth filters
    if not events_df.empty:
        mask = events_df.apply(
            lambda row: passes_filters(
                row['latitude'], row['longitude'], row['depth'],
                row.get('magnitude'), config
            ),
            axis=1
        )
        events_df = events_df[mask].copy()

    # Get valid event IDs
    valid_event_ids = set(events_df['event_id']) if not events_df.empty else set()

    # 2. Filter stations based on geographic filters (lat/lon only, no depth/magnitude)
    if not stations_df.empty:
        mask = stations_df.apply(
            lambda row: passes_filters(
                row['latitude'], row['longitude'], 0.0,  # depth=0 for stations
                None,  # no magnitude for stations
                config
            ),
            axis=1
        )
        stations_df = stations_df[mask].copy()

    # Get valid station IDs
    valid_station_ids = set(stations_df['station_id']) if not stations_df.empty else set()

    # 3. Filter arrivals based on valid events AND valid stations
    if not arrivals_df.empty:
        # First filter by valid events
        if valid_event_ids:
            arrivals_df = arrivals_df[arrivals_df['event_id'].isin(valid_event_ids)].copy()

        # Then filter by valid stations
        if valid_station_ids and 'station_id' in arrivals_df.columns:
            arrivals_df = arrivals_df[arrivals_df['station_id'].isin(valid_station_ids)].copy()

        # Apply distance filters to arrivals if we have the distance columns
        has_distance_filter = any([
            filters.hypocentral_distance_min is not None,
            filters.hypocentral_distance_max is not None,
            filters.epicentral_distance_min is not None,
            filters.epicentral_distance_max is not None,
        ])

        if has_distance_filter:
            # Check if distance columns exist
            has_epi = 'epicentral_distance' in arrivals_df.columns
            has_hypo = 'hypocentral_distance' in arrivals_df.columns

            if has_epi and has_hypo:
                # Filter arrivals by distance
                mask = arrivals_df.apply(
                    lambda row: passes_distance_filters(
                        row.get('epicentral_distance', 0.0) or 0.0,
                        row.get('hypocentral_distance', 0.0) or 0.0,
                        config
                    ) if pd.notna(row.get('epicentral_distance')) and
                         pd.notna(row.get('hypocentral_distance')) else False,
                    axis=1
                )
                arrivals_df = arrivals_df[mask].copy()
            else:
                print("  Warning: Distance filters specified but distance columns "
                      "not found in arrivals. Skipping distance filtering.")

    # 4. Filter picks based on valid events and stations (not by arrivals)
    # This preserves all picks from filtered events/stations, not just those with arrivals
    if not picks_df.empty:
        picks_df = picks_df[
            picks_df['event_id'].isin(valid_event_ids) &
            picks_df['station_id'].isin(valid_station_ids)
        ].copy()

    # 5. Apply manual_picks_only filter if enabled
    manual_picks_only = getattr(filters, 'manual_picks_only', False)
    if not picks_df.empty:
        has_eval_mode = 'evaluation_mode' in picks_df.columns
        if manual_picks_only:
            if not has_eval_mode:
                raise ValueError(
                    "manual_picks_only=True but 'evaluation_mode' column not found in picks. "
                    "Please regenerate the HDF5 file with evaluation_mode column enabled."
                )
            original_count = len(picks_df)
            picks_df = picks_df[picks_df['evaluation_mode'] == 'manual'].copy()
            print(f"  Manual picks filter: {original_count} -> {len(picks_df)} picks")

            # Filter arrivals to only include those referencing manual picks
            if not arrivals_df.empty:
                valid_pick_ids = set(picks_df['pick_id'])
                original_arrivals = len(arrivals_df)
                arrivals_df = arrivals_df[arrivals_df['pick_id'].isin(valid_pick_ids)].copy()
                print(f"  Manual picks filter: {original_arrivals} -> {len(arrivals_df)} arrivals")

    return events_df, stations_df, arrivals_df, picks_df


def load_from_hdf5(h5_file: str, config: DictConfig = None) -> Tuple[pd.DataFrame, pd.DataFrame,
                                          pd.DataFrame, pd.DataFrame]:
    """Load seismic data from HDF5 file.

    Args:
        h5_file: Path to HDF5 file.
        config: Configuration settings (optional, for filtering).

    Returns:
        Tuple of (events_df, stations_df, arrivals_df, picks_df).
    """
    print(f"Loading data from HDF5: {h5_file}")

    events_df = pd.read_hdf(h5_file, 'events')

    try:
        stations_df = pd.read_hdf(h5_file, 'stations')
    except KeyError:
        stations_df = pd.DataFrame()

    try:
        arrivals_df = pd.read_hdf(h5_file, 'arrivals')
    except KeyError:
        arrivals_df = pd.DataFrame()

    try:
        picks_df = pd.read_hdf(h5_file, 'picks')
    except KeyError:
        picks_df = pd.DataFrame()

    print(f"  Loaded {len(events_df)} events")
    print(f"  Loaded {len(stations_df.drop_duplicates(subset=['longitude', 'latitude', 'depth']))} stations (unique positions)")
    if not arrivals_df.empty:
        print(f"  Loaded {len(arrivals_df)} arrivals")
    if not picks_df.empty:
        print(f"  Loaded {len(picks_df)} picks")

    # Apply filters if config is provided
    if config is not None:
        print("\nApplying filters to HDF5 data...")
        events_df, stations_df, arrivals_df, picks_df = apply_filters_to_dataframes(
            events_df, stations_df, arrivals_df, picks_df, config
        )
        print(f"  After filtering: {len(events_df)} events")
        print(f"  After filtering: {len(stations_df.drop_duplicates(subset=['longitude', 'latitude', 'depth']))} stations (unique positions)")
        if not arrivals_df.empty:
            print(f"  After filtering: {len(arrivals_df)} arrivals")
        if not picks_df.empty:
            print(f"  After filtering: {len(picks_df)} picks")

    return events_df, stations_df, arrivals_df, picks_df


def get_qml_files(data_dir: str, years: Optional[List[int]] = None,
                  max_files: Optional[int] = None) -> Tuple[List[str], Dict]:
    """Get list of QuakeML files, optionally filtered by year.

    When *years* is None the function auto-detects:
      1. Year-named subdirectories (4-digit numeric names, e.g. 2020/) and
         collects all .qml files inside them.
      2. .qml files sitting directly in *data_dir* (not inside any subdirectory).

    Args:
        data_dir: Base directory containing QuakeML files.
        years: Optional list of years to filter.
        max_files: Optional maximum number of files to process.

    Returns:
        Tuple of (sorted list of QuakeML file paths, detection_info dict).
        detection_info keys:
          - 'years_found'       : list of detected year directories (auto-detect only)
          - 'files_per_year'    : dict mapping year -> file count (auto-detect only)
          - 'direct_files_count': number of .qml files found directly in data_dir
          - 'mode'              : 'auto' or 'specified'
    """
    logger = logging.getLogger(__name__)
    qml_files = []
    detection_info: Dict = {
        'years_found': [],
        'files_per_year': {},
        'direct_files_count': 0,
        'mode': 'auto' if years is None else 'specified',
    }

    data_path = Path(data_dir)

    if years is None:
        # --- 1. Detect year-named subdirectories ---
        year_dirs = sorted([
            d for d in data_path.iterdir()
            if d.is_dir() and d.name.isdigit() and len(d.name) == 4
        ], key=lambda d: d.name)

        for year_dir in year_dirs:
            year_files = sorted(year_dir.rglob('*.qml'))
            year_files_str = [str(f) for f in year_files]
            qml_files.extend(year_files_str)
            detection_info['years_found'].append(year_dir.name)
            detection_info['files_per_year'][year_dir.name] = len(year_files_str)
            logger.info(f"Year {year_dir.name}: found {len(year_files_str)} QML file(s)")

        # --- 2. Detect .qml files directly in data_dir ---
        direct_files = sorted([
            str(f) for f in data_path.glob('*.qml')
        ])
        if direct_files:
            qml_files.extend(direct_files)
            detection_info['direct_files_count'] = len(direct_files)
            logger.info(f"Direct files in quakeml_dir: found {len(direct_files)} QML file(s)")

    else:
        # Get QML files only from specified years
        for year in years:
            year_dir = data_path / str(year)
            if year_dir.exists():
                year_files = sorted([str(f) for f in year_dir.rglob('*.qml')])
                qml_files.extend(year_files)
                detection_info['files_per_year'][str(year)] = len(year_files)
                logger.info(f"Year {year}: found {len(year_files)} QML file(s)")
            else:
                logger.warning(f"Year directory not found: {year_dir}")

    qml_files = sorted(qml_files)

    if max_files is not None:
        qml_files = qml_files[:max_files]

    return qml_files, detection_info


def calculate_distances(source_lat: float, source_lon: float,
                        source_depth: float, receiver_lat: float,
                        receiver_lon: float,
                        receiver_elevation: float = 0.0) -> Tuple[float, float, float]:
    """Calculate epicentral and hypocentral distances.

    Args:
        source_lat: Source latitude in degrees.
        source_lon: Source longitude in degrees.
        source_depth: Source depth in km (positive down).
        receiver_lat: Receiver latitude in degrees.
        receiver_lon: Receiver longitude in degrees.
        receiver_elevation: Receiver elevation in meters (positive up).

    Returns:
        Tuple of (epicentral_dist_km, hypocentral_dist_km, azimuth_deg).
    """
    # Epicentral distance (surface distance)
    dist_m, az, baz = gps2dist_azimuth(
        source_lat, source_lon,
        receiver_lat, receiver_lon
    )
    epicentral_dist = dist_m / 1000.0  # Convert to km

    # Hypocentral distance (3D distance)
    # Convert receiver elevation to depth (negative of elevation, in km)
    receiver_depth = -receiver_elevation / 1000.0
    depth_diff = source_depth - receiver_depth
    hypocentral_dist = np.sqrt(epicentral_dist**2 + depth_diff**2)

    return epicentral_dist, hypocentral_dist, az


def passes_filters(lat: float, lon: float, depth: float,
                   magnitude: Optional[float], config: DictConfig) -> bool:
    """Check if event/station passes all configured filters.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        depth: Depth in km (positive down).
        magnitude: Magnitude value (can be None).
        config: Configuration with filter settings.

    Returns:
        True if passes all filters, False otherwise.
    """
    filters = config.filters

    # Geographic filters
    if filters.lat_min is not None and lat < filters.lat_min:
        return False
    if filters.lat_max is not None and lat > filters.lat_max:
        return False
    if filters.lon_min is not None and lon < filters.lon_min:
        return False
    if filters.lon_max is not None and lon > filters.lon_max:
        return False

    # Depth filters
    if filters.depth_min is not None and depth < filters.depth_min:
        return False
    if filters.depth_max is not None and depth > filters.depth_max:
        return False

    # Magnitude filters
    if magnitude is not None:
        if (filters.magnitude_min is not None and
            magnitude < filters.magnitude_min):
            return False
        if (filters.magnitude_max is not None and
            magnitude > filters.magnitude_max):
            return False

    return True


def passes_distance_filters(epi_dist: float, hypo_dist: float,
                            config: DictConfig) -> bool:
    """Check if distances pass configured filters.

    Args:
        epi_dist: Epicentral distance in km.
        hypo_dist: Hypocentral distance in km.
        config: Configuration with filter settings.

    Returns:
        True if passes all distance filters, False otherwise.
    """
    filters = config.filters

    if (filters.epicentral_distance_min is not None and
        epi_dist < filters.epicentral_distance_min):
        return False
    if (filters.epicentral_distance_max is not None and
        epi_dist > filters.epicentral_distance_max):
        return False
    if (filters.hypocentral_distance_min is not None and
        hypo_dist < filters.hypocentral_distance_min):
        return False
    if (filters.hypocentral_distance_max is not None and
        hypo_dist > filters.hypocentral_distance_max):
        return False

    return True


def load_from_quakeml(config: DictConfig, raw=None) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                             pd.DataFrame, pd.DataFrame]:
    """Load seismic data from QuakeML files and apply the global filters.

    The QuakeML files are parsed once, unfiltered (quakeml_loader); pass the
    already parsed catalog as ``raw`` to reuse it for several consumers.

    Args:
        config: Configuration settings.
        raw: Optional CatalogData from load_quakeml_catalog().

    Returns:
        Tuple of (events_df, stations_df, arrivals_df, picks_df).
    """
    from preprocess.catalog_filters import apply_catalog_filters, restrict_picks_to_arrivals
    from preprocess.quakeml_loader import load_quakeml_catalog

    if raw is None:
        raw = load_quakeml_catalog(config)
    if raw.events.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    data = raw if config.data.save_all_picks else restrict_picks_to_arrivals(raw)
    data = apply_catalog_filters(data, config.filters, label='preprocessing')

    # Pick-level distances are only used for filtering
    picks_df = data.picks.drop(columns=['epicentral_distance', 'hypocentral_distance'],
                               errors='ignore')
    print(f"\nData Summary:")
    print(f"  Total events: {len(data.events)}")
    print(f"  Total picks: {len(picks_df)}")
    print(f"  Total arrivals: {len(data.arrivals)}")
    print(f"  Total unique station positions: "
          f"{len(data.stations.drop_duplicates(subset=['longitude', 'latitude', 'depth']))}")
    return data.events, data.stations, data.arrivals, picks_df


def load_seismic_data(config: DictConfig, raw=None) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                             pd.DataFrame, pd.DataFrame]:
    """Load seismic data from either HDF5 or QuakeML based on config.

    Args:
        config: Configuration settings.
        raw: Optional already parsed QuakeML catalog (used when data.use_hdf5 is false).

    Returns:
        Tuple of (events_df, stations_df, arrivals_df, picks_df).
    """
    if config.data.use_hdf5:
        return load_from_hdf5(config.data.hdf5_file, config)
    return load_from_quakeml(config, raw)
