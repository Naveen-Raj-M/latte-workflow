"""Data loading functions for QuakeML and HDF5 seismic data.

This module provides functions to load seismic data from either QuakeML files
or pre-processed HDF5 files, with support for parallel processing and filtering.
"""

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
from obspy import read_events
from obspy.geodetics import gps2dist_azimuth
from omegaconf import DictConfig
from tqdm import tqdm


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
    print(f"  Loaded {len(stations_df)} stations")
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
        print(f"  After filtering: {len(stations_df)} stations")
        if not arrivals_df.empty:
            print(f"  After filtering: {len(arrivals_df)} arrivals")
        if not picks_df.empty:
            print(f"  After filtering: {len(picks_df)} picks")

    return events_df, stations_df, arrivals_df, picks_df


def get_qml_files(data_dir: str, years: Optional[List[int]] = None,
                  max_files: Optional[int] = None) -> List[str]:
    """Get list of QuakeML files, optionally filtered by year.

    Args:
        data_dir: Base directory containing QuakeML files.
        years: Optional list of years to filter.
        max_files: Optional maximum number of files to process.

    Returns:
        Sorted list of QuakeML file paths.
    """
    qml_files = []

    if years is None:
        # Get all QML files recursively
        qml_files = glob.glob(
            os.path.join(data_dir, '**', '*.qml'),
            recursive=True
        )
    else:
        # Get QML files only from specified years
        for year in years:
            year_dir = os.path.join(data_dir, str(year))
            if os.path.exists(year_dir):
                year_files = glob.glob(
                    os.path.join(year_dir, '**', '*.qml'),
                    recursive=True
                )
                qml_files.extend(year_files)

    qml_files = sorted(qml_files)

    if max_files is not None:
        qml_files = qml_files[:max_files]

    return qml_files


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


def process_single_qml_file(qml_file: str, station_inventory: Dict,
                            config: DictConfig) -> Tuple[List, List, List, Set]:
    """Process a single QuakeML file and extract data.

    This function is called by multiprocessing workers.

    Args:
        qml_file: Path to QuakeML file.
        station_inventory: Dictionary of station information.
        config: Configuration settings.

    Returns:
        Tuple of (events, picks, arrivals, stations_used).
    """
    try:
        catalog = read_events(qml_file)
        return extract_event_data(catalog, station_inventory, config)
    except Exception as e:
        # Return empty results on error
        return [], [], [], set()


def extract_event_data(catalog, station_inventory: Dict,
                      config: DictConfig) -> Tuple[List, List, List, Set]:
    """Extract event data from catalog with filtering.

    Args:
        catalog: ObsPy catalog object.
        station_inventory: Dictionary of station information.
        config: Configuration settings.

    Returns:
        Tuple of (events, picks, arrivals, stations_used).
        - events: list of dicts with event info
        - picks: list of dicts with pick info
        - arrivals: list of dicts with arrival info
        - stations_used: set of station IDs used
    """
    events = []
    picks = []
    arrivals = []
    stations_used = set()

    # Check if we need distance filtering
    has_distance_filter = any([
        config.filters.hypocentral_distance_min is not None,
        config.filters.hypocentral_distance_max is not None,
        config.filters.epicentral_distance_min is not None,
        config.filters.epicentral_distance_max is not None,
    ])

    has_station_coords = len(station_inventory) > 0

    for event in catalog:
        # Select origin
        if config.data.use_preferred_origin:
            origin = (event.preferred_origin() or
                     (event.origins[0] if event.origins else None))
        else:
            origin = event.origins[0] if event.origins else None

        if origin is None:
            continue

        # Select magnitude
        if config.data.use_preferred_magnitude:
            magnitude = (event.preferred_magnitude() or
                        (event.magnitudes[0] if event.magnitudes else None))
        else:
            magnitude = (event.magnitudes[0] if event.magnitudes else None)

        source_lat = origin.latitude
        source_lon = origin.longitude
        source_depth = (origin.depth / 1000.0 if origin.depth is not None
                       else 0.0)  # Convert to km
        source_time = origin.time
        mag_value = magnitude.mag if magnitude else None

        # Apply filters to source
        if not passes_filters(source_lat, source_lon, source_depth,
                             mag_value, config):
            continue

        # Build pick dictionary
        pick_dict = {pick.resource_id.id: pick for pick in event.picks}

        # Determine which picks to save
        if config.data.save_all_picks:
            picks_to_process = list(pick_dict.values())
            pick_ids_in_origin = set()
        else:
            # Only picks associated with selected origin
            arrivals_list = (origin.arrivals if hasattr(origin, 'arrivals')
                           else [])
            pick_ids_in_origin = {arr.pick_id.id for arr in arrivals_list}
            picks_to_process = [pick_dict[pid] for pid in pick_ids_in_origin
                              if pid in pick_dict]

        # Get arrivals for this origin
        arrivals_list = origin.arrivals if hasattr(origin, 'arrivals') else []

        # Track if event has valid data after filtering
        event_picks = []
        event_arrivals = []
        event_stations = set()

        # Process arrivals (which reference picks)
        for arrival in arrivals_list:
            pick_id = arrival.pick_id.id
            if pick_id not in pick_dict:
                continue

            pick = pick_dict[pick_id]

            # Extract station ID - prioritize waveform_id.station_code
            station_id = None
            if (hasattr(pick, 'waveform_id') and
                hasattr(pick.waveform_id, 'station_code')):
                station_id = pick.waveform_id.station_code

            if station_id is None:
                continue

            # Check if we have station coordinates
            if has_station_coords and station_id in station_inventory:
                sta_info = station_inventory[station_id]
                sta_lat = sta_info['lat']
                sta_lon = sta_info['lon']
                sta_elev = sta_info['elevation']  # meters, positive = above MSL

                # Apply geographic filter to receiver
                if not passes_filters(sta_lat, sta_lon, 0.0, None, config):
                    continue

                # Calculate distances
                epi_dist, hypo_dist, azimuth = calculate_distances(
                    source_lat, source_lon, source_depth,
                    sta_lat, sta_lon, sta_elev
                )

                # Apply distance filters
                if has_distance_filter:
                    if not passes_distance_filters(epi_dist, hypo_dist, config):
                        continue
            else:
                # No coordinates available
                if has_distance_filter:
                    # Can't apply distance filter without coordinates, skip
                    continue
                epi_dist = None
                hypo_dist = None
                azimuth = None

            # Arrival passed all filters
            event_stations.add(station_id)

            # Extract arrival data
            arrival_data = {
                'arrival_id': (str(arrival.resource_id.id) if
                             hasattr(arrival, 'resource_id') else
                             f"{event.resource_id.id}_{pick_id}"),
                'event_id': str(event.resource_id.id),
                'pick_id': pick_id,
                'station_id': station_id,
                'phase': arrival.phase if hasattr(arrival, 'phase') else '',
                'azimuth': (arrival.azimuth if hasattr(arrival, 'azimuth')
                          else azimuth),
                'distance': (arrival.distance if hasattr(arrival, 'distance')
                           else None),
                'time_residual': (arrival.time_residual if
                                hasattr(arrival, 'time_residual') else None),
                'time_weight': (arrival.time_weight if
                              hasattr(arrival, 'time_weight') else None),
                'epicentral_distance': epi_dist,
                'hypocentral_distance': hypo_dist,
            }
            event_arrivals.append(arrival_data)

            # Extract pick data
            if pick_id in pick_ids_in_origin or config.data.save_all_picks:
                pick_data = {
                    'pick_id': pick_id,
                    'event_id': str(event.resource_id.id),
                    'station_id': station_id,
                    'pick_time': str(pick.time),
                    'phase_hint': (pick.phase_hint if
                                 hasattr(pick, 'phase_hint') else ''),
                }
                # Only add if not already added
                if pick_id not in {p['pick_id'] for p in event_picks}:
                    event_picks.append(pick_data)

        # If saving all picks, process picks not associated with arrivals
        if config.data.save_all_picks:
            for pick in picks_to_process:
                pick_id = str(pick.resource_id.id)
                if pick_id not in {p['pick_id'] for p in event_picks}:
                    # Extract station ID
                    station_id = None
                    if (hasattr(pick, 'waveform_id') and
                        hasattr(pick.waveform_id, 'station_code')):
                        station_id = pick.waveform_id.station_code

                    if station_id:
                        pick_data = {
                            'pick_id': pick_id,
                            'event_id': str(event.resource_id.id),
                            'station_id': station_id,
                            'pick_time': str(pick.time),
                            'phase_hint': (pick.phase_hint if
                                         hasattr(pick, 'phase_hint') else ''),
                        }
                        event_picks.append(pick_data)
                        event_stations.add(station_id)

        # Only add event if it has arrivals (or picks if save_all_picks)
        if (len(event_arrivals) > 0 or
            (config.data.save_all_picks and len(event_picks) > 0)):
            # Extract event data
            event_data = {
                'event_id': str(event.resource_id.id),
                'origin_time': str(source_time),
                'latitude': source_lat,
                'longitude': source_lon,
                'depth': source_depth,  # km, positive = below MSL
                'magnitude': mag_value,
                'magnitude_type': (magnitude.magnitude_type if magnitude
                                 else ''),
                'num_picks': len(event_picks),
                'num_arrivals': len(event_arrivals),
            }

            events.append(event_data)
            picks.extend(event_picks)
            arrivals.extend(event_arrivals)
            stations_used.update(event_stations)

    return events, picks, arrivals, stations_used


def load_from_quakeml(config: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                   pd.DataFrame, pd.DataFrame]:
    """Load seismic data from QuakeML files with parallel processing.

    Args:
        config: Configuration settings.

    Returns:
        Tuple of (events_df, stations_df, arrivals_df, picks_df).
    """
    print("\nLoading data from QuakeML files...")

    # Load station inventory
    print(f"Loading station inventory from: {config.data.station_file}")
    station_inventory = load_station_inventory(config.data.station_file)
    print(f"Loaded {len(station_inventory)} stations")

    # Get QML files
    print(f"\nSearching for QML files in: {config.data.quakeml_dir}")
    if config.data.years:
        print(f"Filtering for years: {config.data.years}")
    if config.data.max_files:
        print(f"Processing max {config.data.max_files} files")

    qml_files = get_qml_files(
        config.data.quakeml_dir,
        config.data.years,
        config.data.max_files
    )
    print(f"Found {len(qml_files)} QML files to process")

    if len(qml_files) == 0:
        print("No QML files found.")
        return (pd.DataFrame(), pd.DataFrame(),
                pd.DataFrame(), pd.DataFrame())

    # Process all QML files in parallel
    all_events = []
    all_picks = []
    all_arrivals = []
    all_stations = set()

    # Determine number of workers
    num_workers = (config.parallel.num_workers if
                  config.parallel.num_workers else cpu_count())
    print(f"\nProcessing QML files using {num_workers} workers...")

    # Create partial function with fixed arguments
    worker_func = partial(
        process_single_qml_file,
        station_inventory=station_inventory,
        config=config
    )

    # Process files in parallel with progress bar
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(worker_func, qml_files),
            total=len(qml_files),
            desc="Converting"
        ))

    # Combine results from all workers
    print("\nCombining results...")
    for events, picks, arrivals, stations in results:
        all_events.extend(events)
        all_picks.extend(picks)
        all_arrivals.extend(arrivals)
        all_stations.update(stations)

    print(f"\nData Summary:")
    print(f"  Total events: {len(all_events)}")
    print(f"  Total picks: {len(all_picks)}")
    print(f"  Total arrivals: {len(all_arrivals)}")
    print(f"  Total unique stations: {len(all_stations)}")

    # Convert to DataFrames
    events_df = pd.DataFrame(all_events)
    picks_df = pd.DataFrame(all_picks) if all_picks else pd.DataFrame()
    arrivals_df = (pd.DataFrame(all_arrivals) if all_arrivals
                  else pd.DataFrame())

    # Create stations DataFrame
    stations_data = []
    for station_id in all_stations:
        if station_id in station_inventory:
            sta_info = station_inventory[station_id]
            stations_data.append({
                'station_id': station_id,
                'latitude': sta_info['lat'],
                'longitude': sta_info['lon'],
                'depth': -sta_info['elevation'] / 1000.0,  # km, negative = above MSL
            })
    stations_df = pd.DataFrame(stations_data)

    return events_df, stations_df, arrivals_df, picks_df


def load_seismic_data(config: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                   pd.DataFrame, pd.DataFrame]:
    """Load seismic data from either HDF5 or QuakeML based on config.

    Args:
        config: Configuration settings.

    Returns:
        Tuple of (events_df, stations_df, arrivals_df, picks_df).
    """
    if config.data.use_hdf5:
        return load_from_hdf5(config.data.hdf5_file, config)
    else:
        return load_from_quakeml(config)
