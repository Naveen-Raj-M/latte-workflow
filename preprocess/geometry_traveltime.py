"""Geometry and travel time processing for LATTE preprocessing.

This module provides functions for creating geometry files and travel time
binary files from Cartesian coordinates (after coordinate transformation).
"""

import logging
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dateutil import parser as date_parser
from omegaconf import DictConfig
from joblib import Parallel, delayed
from tqdm import tqdm

from utils.io_utils import ensure_directory_exists
from utils.origin_time_utils import (
    get_origin_time_for_event,
    save_origin_time_metadata
)


def call_get_origin_time_from_datetime(
    dt: datetime,
    mode: str,
    fixed_value: float = 0.0
) -> Tuple[float, float]:
    """
    Helper to call get_origin_time_for_event using a datetime object.

    Returns:
        Tuple of (configured_origin_time, original_origin_time_s)
    """
    if dt.tzinfo is None:
        raise ValueError("dt must be timezone-aware")

    year = dt.year
    jday = dt.timetuple().tm_yday
    hour = dt.hour
    minute = dt.minute
    second = dt.second + dt.microsecond * 1e-6

    # Calculate original origin time (seconds from start of year)
    original_origin_time_s = (jday - 1) * 86400 + hour * 3600 + minute * 60 + second

    # Get configured origin time
    configured_origin_time = get_origin_time_for_event(
        original_time_s=original_origin_time_s,
        mode=mode,
        fixed_value=fixed_value,
        year=year,
        jday=jday,
        hour=hour,
        minute=minute,
        second=second
    )

    return configured_origin_time, original_origin_time_s


def _process_single_event(event_id, event_picks, event_lookup, station_lookup,
                          geom_config, origin_time_config):
    """Process picks for a single event.

    This function is designed to be called in parallel for each event.

    Returns:
        Tuple of (event_id, event_data_dict, unrealistic_count) or None if event invalid
    """
    logger = logging.getLogger(__name__)

    if event_id not in event_lookup:
        return None

    event_info = event_lookup[event_id]
    event_data = {
        'origin_time_s': 0.0,
        'original_origin_time_s': 0.0,
        'year': None,
        'jday': None,
        'hour': None,
        'minute': None,
        'second': None,
        'p_picks': {},
        's_picks': {},
        'coords': None,
        'original_coords': None
    }

    unrealistic_velocity_count = 0

    # Parse origin time
    try:
        origin_dt = date_parser.parse(event_info['origin_time'])
        if origin_dt.tzinfo is None:
            origin_dt = origin_dt.replace(tzinfo=timezone.utc)

        configured_origin_time, original_origin_time_s = call_get_origin_time_from_datetime(
            origin_dt,
            origin_time_config.mode,
            origin_time_config.fixed_value
        )

        # Store time components
        event_data['origin_time_s'] = configured_origin_time
        event_data['original_origin_time_s'] = original_origin_time_s
        event_data['year'] = origin_dt.year
        event_data['jday'] = origin_dt.timetuple().tm_yday
        event_data['hour'] = origin_dt.hour
        event_data['minute'] = origin_dt.minute
        event_data['second'] = origin_dt.second + origin_dt.microsecond * 1e-6
        event_data['coords'] = (event_info['x'], event_info['y'], event_info['z'])
        event_data['original_coords'] = (event_info['lon'], event_info['lat'], event_info['depth'])

    except Exception as e:
        logger.warning(f"Could not parse origin time for {event_id}: {e}")
        return None

    # Separate P and S picks
    for _, pick in event_picks.iterrows():
        station_id = pick['station_id']
        phase_hint = pick.get('phase_hint', '').upper()

        if station_id not in station_lookup:
            logger.debug(f"Station {station_id} not found in station lookup")
            continue

        station_info = station_lookup[station_id]

        # Parse pick time
        try:
            pick_dt = date_parser.parse(pick['pick_time'])
            if pick_dt.tzinfo is None:
                pick_dt = pick_dt.replace(tzinfo=timezone.utc)

            # Calculate physical travel time
            physical_travel_time = (pick_dt - origin_dt).total_seconds()

            if physical_travel_time < 0:
                logger.debug(f"Negative travel time for {event_id}, {station_id}: {physical_travel_time:.2f}s")
                continue

            # Velocity check
            distance = np.sqrt((event_info['x'] - station_info['x'])**2 +
                             (event_info['y'] - station_info['y'])**2 +
                             (event_info['z'] - station_info['z'])**2)

            if physical_travel_time < 1e-6:
                unrealistic_velocity_count += 1
                logger.debug(f"Travel time is zero for {event_id}, {station_id}")
                continue

            apparent_velocity = distance / physical_travel_time

            is_realistic = False
            if 'P' in phase_hint:
                if geom_config.p_velocity_range_m_s[0] <= apparent_velocity <= geom_config.p_velocity_range_m_s[1]:
                    is_realistic = True
            elif 'S' in phase_hint:
                if geom_config.s_velocity_range_m_s[0] <= apparent_velocity <= geom_config.s_velocity_range_m_s[1]:
                    is_realistic = True

            if not is_realistic:
                unrealistic_velocity_count += 1
                logger.debug(f"Unrealistic velocity for {event_id}, {station_id} ({phase_hint}): {apparent_velocity:.1f} m/s")
                continue

            # Calculate observed travel time (physical + configured origin time)
            observed_travel_time = physical_travel_time + configured_origin_time

            # Get weight (use default if configured)
            if geom_config.get('apply_default_weights', False):
                weight = geom_config.get('default_weight', 1.0)
            else:
                weight = pick.get('weight', 1.0)

            pick_data = {
                'station': station_id,
                'coords': (station_info['x'], station_info['y'], station_info['z']),
                'original_coords': (station_info['lon'], station_info['lat'], station_info['depth']),
                'physical_travel_time': physical_travel_time,
                'observed_travel_time': observed_travel_time,
                'weight': weight,
                'pick_time': pick_dt
            }

            # Classify phase
            if 'P' in phase_hint:
                event_data['p_picks'][station_id] = pick_data
            elif 'S' in phase_hint:
                event_data['s_picks'][station_id] = pick_data

        except Exception as e:
            logger.warning(f"Could not parse pick time: {e}")
            continue

    return (event_id, event_data, unrealistic_velocity_count)


def parse_picks_with_cartesian(events_df: pd.DataFrame,
                               stations_df: pd.DataFrame,
                               picks_df: pd.DataFrame,
                               config: DictConfig) -> Tuple[Dict, Dict]:
    """Parse picks and match with Cartesian coordinates.

    Args:
        events_df: DataFrame with event Cartesian coordinates.
        stations_df: DataFrame with station Cartesian coordinates.
        picks_df: DataFrame with pick data.
        config: Full configuration.

    Returns:
        Tuple of (event_data dict, station_lookup dict)
    """
    logger = logging.getLogger(__name__)
    geom_config = config.geometry_traveltime
    origin_time_config = config.origin_time

    # Create lookup dictionaries for event coordinates
    event_lookup = {}
    for _, row in events_df.iterrows():
        event_lookup[row['event_id']] = {
            'x': row['X_grid_km'] * 1000,  # Convert to meters
            'y': row['Y_grid_km'] * 1000,
            'z': row['Z_grid_km'] * 1000,
            'origin_time': row['origin_time'],
            'lon': row.get('longitude', row.get('lon', 0.0)),
            'lat': row.get('latitude', row.get('lat', 0.0)),
            'depth': row.get('depth', 0.0)
        }

    # Create lookup dictionaries for station coordinates
    station_lookup = {}
    for _, row in stations_df.iterrows():
        station_lookup[row['station_id']] = {
            'x': row['X_grid_km'] * 1000,  # Convert to meters
            'y': row['Y_grid_km'] * 1000,
            'z': row['Z_grid_km'] * 1000,
            'lon': row.get('longitude', row.get('lon', 0.0)),
            'lat': row.get('latitude', row.get('lat', 0.0)),
            'depth': row.get('depth', 0.0)
        }

    # Get parallel processing configuration
    parallel_config = config.get('parallel', {})
    n_jobs = parallel_config.get('num_workers', None)
    if n_jobs is None:
        n_jobs = -1  # Use all CPU cores

    logger.info(f"Using parallel processing with n_jobs={n_jobs if n_jobs != -1 else 'all cores'}")

    # Process picks grouped by event
    picks_by_event = list(picks_df.groupby('event_id'))

    logger.info(f"Processing {len(picks_by_event)} events in parallel...")

    # Process events in parallel with progress bar
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_process_single_event)(
            event_id, event_picks, event_lookup, station_lookup,
            geom_config, origin_time_config
        )
        for event_id, event_picks in tqdm(picks_by_event, desc="Processing events", unit="event")
    )

    # Collect results
    temp_events = {}
    total_picks = 0
    unrealistic_velocity_count = 0

    for result in results:
        if result is not None:
            event_id, event_data, unrealistic_count = result
            temp_events[event_id] = event_data
            # Count total picks from this event
            total_picks += len(event_data['p_picks']) + len(event_data['s_picks'])
            unrealistic_velocity_count += unrealistic_count

    # Filter based on phase_type configuration
    final_events = {}
    phase_type = geom_config.get('phase_type', 'both').lower()

    if phase_type == 'both':
        logger.info("Filtering for stations with both valid P and S picks...")
        for eid, data in temp_events.items():
            valid_p_stations = set(data['p_picks'].keys())
            valid_s_stations = set(data['s_picks'].keys())
            common_stations = sorted(list(valid_p_stations.intersection(valid_s_stations)))

            if common_stations:
                final_events[eid] = {
                    'origin_time_s': data['origin_time_s'],
                    'original_origin_time_s': data['original_origin_time_s'],
                    'year': data['year'],
                    'jday': data['jday'],
                    'hour': data['hour'],
                    'minute': data['minute'],
                    'second': data['second'],
                    'coords': data['coords'],
                    'original_coords': data['original_coords'],
                    'p_picks': [data['p_picks'][stn] for stn in common_stations],
                    's_picks': [data['s_picks'][stn] for stn in common_stations]
                }
    elif phase_type == 'p':
        logger.info("Processing P-wave picks only (no S-wave requirement)...")
        for eid, data in temp_events.items():
            if data['p_picks']:
                p_stations = sorted(list(data['p_picks'].keys()))
                final_events[eid] = {
                    'origin_time_s': data['origin_time_s'],
                    'original_origin_time_s': data['original_origin_time_s'],
                    'year': data['year'],
                    'jday': data['jday'],
                    'hour': data['hour'],
                    'minute': data['minute'],
                    'second': data['second'],
                    'coords': data['coords'],
                    'original_coords': data['original_coords'],
                    'p_picks': [data['p_picks'][stn] for stn in p_stations],
                    's_picks': []
                }
    elif phase_type == 's':
        logger.info("Processing S-wave picks only (no P-wave requirement)...")
        for eid, data in temp_events.items():
            if data['s_picks']:
                s_stations = sorted(list(data['s_picks'].keys()))
                final_events[eid] = {
                    'origin_time_s': data['origin_time_s'],
                    'original_origin_time_s': data['original_origin_time_s'],
                    'year': data['year'],
                    'jday': data['jday'],
                    'hour': data['hour'],
                    'minute': data['minute'],
                    'second': data['second'],
                    'coords': data['coords'],
                    'original_coords': data['original_coords'],
                    'p_picks': [],
                    's_picks': [data['s_picks'][stn] for stn in s_stations]
                }

    logger.info(f"Parsed {len(final_events)} events with picks")
    logger.info(f"Total picks processed: {total_picks}")
    logger.info(f"Picks rejected due to unrealistic velocity: {unrealistic_velocity_count}")

    return final_events, station_lookup


def _write_single_event_files(event_item, geom_config, phase_type):
    """Write geometry and travel time files for a single event.

    This function is designed to be called in parallel for each event.

    Args:
        event_item: Tuple of (idx, eid, data) where idx is 1-based shot ID
        geom_config: Geometry configuration
        phase_type: Phase type filter ('both', 'p', or 's')

    Returns:
        Tuple of (idx, eid, geom_fname) for the mapping file
    """
    idx, eid, data = event_item

    # Use P picks for geometry if available, otherwise S picks
    receivers_for_geometry = data['p_picks'] if data['p_picks'] else data['s_picks']

    geom_fname = f"shot_{idx}_geometry.txt"
    ex, ey, ez = data['coords']

    # Prepare origin time formatting
    origin_time = data.get('origin_time_s', 0.0)
    origin_time_precision = geom_config.get('origin_time_precision', 'single')
    if origin_time_precision == 'double':
        origin_time_fmt = f"{origin_time:.12f}"
    else:
        origin_time_fmt = f"{origin_time:.6f}"

    # Write Geometry File
    geom_path = Path(geom_config.geometry_dir) / geom_fname
    with open(geom_path, "w") as out:
        out.write(f"{idx}\n\n1\n")
        out.write(f"{ex:.6f} {ey:.6f} {ez:.6f} {origin_time_fmt}\n\n")
        out.write(f"{len(receivers_for_geometry)}\n")
        for pick in receivers_for_geometry:
            sx, sy, sz = pick['coords']
            w = pick['weight']
            out.write(f"{sx:.6f} {sy:.6f} {sz:.6f} {w:.6f}\n")

    # Write P-wave traveltime binary (if processing P-waves)
    if phase_type in ['both', 'p'] and data['p_picks']:
        tt_p_fname = Path(geom_config.traveltime_dir) / f"shot_{idx}_traveltime_p.bin"
        p_travel_times = [p['observed_travel_time'] for p in data['p_picks']]
        p_array = np.array(p_travel_times, dtype='<f4')
        with open(tt_p_fname, 'wb') as f:
            f.write(p_array.tobytes())

    # Write S-wave traveltime binary (if processing S-waves)
    if phase_type in ['both', 's'] and data['s_picks']:
        tt_s_fname = Path(geom_config.traveltime_dir) / f"shot_{idx}_traveltime_s.bin"
        s_travel_times = [s['observed_travel_time'] for s in data['s_picks']]
        s_array = np.array(s_travel_times, dtype='<f4')
        with open(tt_s_fname, 'wb') as f:
            f.write(s_array.tobytes())

    return (idx, eid, geom_fname)


def write_source_binary_files(events: Dict, config: DictConfig, origin_time_config: DictConfig) -> None:
    """
    Write source binary files (sx.bin, sy.bin, sz.bin, st0.bin) for the events.
    These files contain the source coordinates and origin times for each event.
    """
    logger = logging.getLogger(__name__)
    geom_config = config.geometry_traveltime

    if not geom_config.generate_source_binaries:
        return

    ensure_directory_exists(Path(geom_config.source_dir))

    logger.info(f"Writing source binary files to: {geom_config.source_dir}")

    # Sort events to maintain consistent ordering (same as in geometry files)
    sorted_events = sorted(events.items(), key=lambda item: item[0])

    # Extract source coordinates and origin times
    sx_values = []
    sy_values = []
    sz_values = []
    st0_values = []
    event_original_times = {}

    for eid, data in sorted_events:
        ex, ey, ez = data['coords']
        sx_values.append(ex)
        sy_values.append(ey)
        sz_values.append(ez)
        st0_values.append(data['origin_time_s'])
        event_original_times[eid] = data.get('original_origin_time_s', 0.0)

    # Convert to numpy arrays
    sx_array = np.array(sx_values, dtype='<f4')  # Little-endian 32-bit float
    sy_array = np.array(sy_values, dtype='<f4')
    sz_array = np.array(sz_values, dtype='<f4')

    # Use configured precision for origin times
    origin_time_precision = geom_config.get('origin_time_precision', 'single')
    if origin_time_precision == 'double':
        st0_array = np.array(st0_values, dtype='<f8')  # Little-endian 64-bit double
    else:
        st0_array = np.array(st0_values, dtype='<f4')  # Little-endian 32-bit float

    # Write binary files
    source_dir = Path(geom_config.source_dir)
    with open(source_dir / 'sx.bin', 'wb') as f:
        f.write(sx_array.tobytes())
    with open(source_dir / 'sy.bin', 'wb') as f:
        f.write(sy_array.tobytes())
    with open(source_dir / 'sz.bin', 'wb') as f:
        f.write(sz_array.tobytes())
    with open(source_dir / 'st0.bin', 'wb') as f:
        f.write(st0_array.tobytes())

    precision_desc = "double precision (8 bytes)" if origin_time_precision == 'double' else "single precision (4 bytes)"
    logger.info(f"Source binary files written:")
    logger.info(f"  sx.bin: {len(sx_values)} values (x-coordinates, single precision)")
    logger.info(f"  sy.bin: {len(sy_values)} values (y-coordinates, single precision)")
    logger.info(f"  sz.bin: {len(sz_values)} values (z-coordinates, single precision)")
    logger.info(f"  st0.bin: {len(st0_values)} values (origin times in {origin_time_config.mode} mode, {precision_desc})")

    # Save origin time metadata
    save_origin_time_metadata(
        output_dir=geom_config.source_dir,
        mode=origin_time_config.mode,
        fixed_value=origin_time_config.get('fixed_value', 0.0),
        event_original_times=event_original_times
    )

    # Print summary statistics
    if sx_values:
        logger.info(f"Source coordinate ranges:")
        logger.info(f"  X: {min(sx_values):.1f} to {max(sx_values):.1f} m")
        logger.info(f"  Y: {min(sy_values):.1f} to {max(sy_values):.1f} m")
        logger.info(f"  Z: {min(sz_values):.1f} to {max(sz_values):.1f} m")
        logger.info(f"  Origin times ({origin_time_config.mode} mode): {min(st0_values):.1f} to {max(st0_values):.1f} s")


def write_output_files(events: Dict, config: DictConfig, origin_time_config: DictConfig) -> None:
    """
    Writes geometry files and travel time binary files.
    Geometry files are text files, binary files are raw floats.
    """
    logger = logging.getLogger(__name__)
    geom_config = config.geometry_traveltime

    ensure_directory_exists(Path(geom_config.geometry_dir))
    ensure_directory_exists(Path(geom_config.traveltime_dir))

    id_map_path = Path(geom_config.geometry_dir) / "shot_to_source_id_map.txt"
    phase_type = geom_config.get('phase_type', 'both').lower()

    logger.info(f"Writing output files (phase_type: {phase_type})...")

    # Get parallel processing configuration
    parallel_config = config.get('parallel', {})
    n_jobs = parallel_config.get('num_workers', None)
    if n_jobs is None:
        n_jobs = -1  # Use all CPU cores

    # Sort events to maintain consistent ordering
    sorted_events = sorted(events.items(), key=lambda item: item[0])

    # Prepare event items with indices for parallel processing
    event_items = [(idx, eid, data) for idx, (eid, data) in enumerate(sorted_events, start=1)]

    logger.info(f"Writing {len(event_items)} event files in parallel...")
    logger.info(f"Using parallel processing with n_jobs={n_jobs if n_jobs != -1 else 'all cores'}")

    # Process events in parallel with progress bar
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_write_single_event_files)(event_item, geom_config, phase_type)
        for event_item in tqdm(event_items, desc="Writing event files", unit="event")
    )

    # Sort results by shot ID to maintain order
    results = sorted(results, key=lambda x: x[0])

    # Write the mapping file and collect shot filenames
    shot_files = []
    with open(id_map_path, "w") as map_f:
        map_f.write("Shot_ID\tOriginal_Event_ID\n")
        for idx, eid, geom_fname in results:
            map_f.write(f"{idx}\t{eid}\n")
            shot_files.append(geom_fname)

    # Write geometry index file
    geom_index_path = Path(geom_config.geometry_dir) / "geometry.txt"
    with open(geom_index_path, "w") as idxf:
        for shot in shot_files:
            idxf.write(shot + "\n")

    origin_time_precision = geom_config.get('origin_time_precision', 'single')
    precision_desc = "double precision (12 decimal places)" if origin_time_precision == 'double' else "single precision (6 decimal places)"

    logger.info(f"File Generation Summary:")
    logger.info(f"Successfully wrote {len(shot_files)} shots in total.")
    logger.info(f"Geometry index: {geom_index_path}")
    logger.info(f"Source ID map: {id_map_path}")
    logger.info(f"Origin time mode: {origin_time_config.mode}")
    logger.info(f"Origin time precision: {precision_desc}")
    logger.info(f"IMPORTANT: Update 'ns' parameter in Fortran input to {len(shot_files)}")


def write_travel_time_csv(events: Dict, config: DictConfig) -> None:
    """
    Write comprehensive travel time data to CSV file with all metadata.
    Creates ground_truth_traveltimes.csv.
    """
    logger = logging.getLogger(__name__)
    geom_config = config.geometry_traveltime

    if not geom_config.generate_travel_time_csv:
        return

    ensure_directory_exists(Path(geom_config.csv_output_dir))
    phase_type = geom_config.get('phase_type', 'both').lower()

    logger.info(f"Writing comprehensive travel time CSV to: {geom_config.csv_output_dir}")
    logger.info(f"Phase type filter: {phase_type}")

    # Prepare data for CSV export
    csv_data = []

    # Sort events to maintain consistent ordering (same as in geometry files)
    sorted_events = sorted(events.items(), key=lambda item: item[0])

    for idx, (eid, data) in enumerate(sorted_events, start=1):
        shot_id = idx

        # Get event coordinates
        ex_transformed, ey_transformed, ez_transformed = data['coords']
        event_lon, event_lat, event_depth_km = data['original_coords']

        # Get event metadata
        origin_time_s = data.get('origin_time_s', 0.0)

        # Get P-wave and S-wave picks
        p_picks = data['p_picks']
        s_picks = data['s_picks']

        # Determine which phases to process based on phase_type
        phases_to_process = []
        if phase_type in ['both', 'p'] and p_picks:
            phases_to_process.append(('P', p_picks))
        if phase_type in ['both', 's'] and s_picks:
            phases_to_process.append(('S', s_picks))

        # Process selected phases
        for phase, picks in phases_to_process:
            for pick in picks:
                station = pick['station']
                physical_travel_time = pick['physical_travel_time']
                weight = pick['weight']
                sx_transformed, sy_transformed, sz_transformed = pick['coords']
                station_lon, station_lat, station_depth_km = pick['original_coords']

                # Calculate distance using transformed coordinates
                distance_m = np.sqrt((ex_transformed - sx_transformed)**2 +
                                   (ey_transformed - sy_transformed)**2 +
                                   (ez_transformed - sz_transformed)**2)

                csv_data.append({
                    # Identifiers
                    'shot_id': shot_id,
                    'event_id': eid,
                    'station': station,
                    'phase': phase,

                    # Travel time data (CSV uses physical travel time only)
                    'observed_tt': physical_travel_time,
                    'weight': weight,
                    'residual': 0.0,  # Not available in new format

                    # Original coordinates (lat/lon/depth)
                    'event_longitude': event_lon,
                    'event_latitude': event_lat,
                    'event_depth_km': event_depth_km,
                    'station_longitude': station_lon,
                    'station_latitude': station_lat,
                    'station_depth_km': station_depth_km,

                    # Transformed coordinates (cartesian)
                    'event_x_transformed': ex_transformed,
                    'event_y_transformed': ey_transformed,
                    'event_z_transformed': ez_transformed,
                    'station_x_transformed': sx_transformed,
                    'station_y_transformed': sy_transformed,
                    'station_z_transformed': sz_transformed,

                    # Derived quantities
                    'distance_m': distance_m,
                    'origin_time_s': origin_time_s,
                    'weighted_rms': 0.0  # Not available in new format
                })

    # Convert to DataFrame and save
    if csv_data:
        df = pd.DataFrame(csv_data)
        
        csv_filename = "ground_truth_traveltimes.csv"
        csv_path = Path(geom_config.csv_output_dir) / csv_filename
        df.to_csv(csv_path, index=False)

        logger.info(f"Comprehensive travel time CSV written: {csv_filename}")
        logger.info(f"  Phase type: {phase_type}")
        logger.info(f"  Total records: {len(csv_data)}")
        logger.info(f"  Unique events: {df['event_id'].nunique()}")
        logger.info(f"  Unique stations: {df['station'].nunique()}")

        # Print phase-specific statistics
        if phase_type in ['both', 'p']:
            p_data = df[df['phase'] == 'P']
            if not p_data.empty:
                logger.info(f"  P-wave picks: {len(p_data)}")
                logger.info(f"  P-wave travel times: {p_data['observed_tt'].min():.3f} to {p_data['observed_tt'].max():.3f} s (mean: {p_data['observed_tt'].mean():.3f} s)")

        if phase_type in ['both', 's']:
            s_data = df[df['phase'] == 'S']
            if not s_data.empty:
                logger.info(f"  S-wave picks: {len(s_data)}")
                logger.info(f"  S-wave travel times: {s_data['observed_tt'].min():.3f} to {s_data['observed_tt'].max():.3f} s (mean: {s_data['observed_tt'].mean():.3f} s)")

        # Print coordinate range information
        if not df['distance_m'].isna().all():
            logger.info(f"  Distance range: {df['distance_m'].min():.1f} to {df['distance_m'].max():.1f} m")

        logger.info(f"  CSV columns: {', '.join(df.columns)}")
        logger.info("  Note: Shot ID matches binary file numbering - no separate mapping file needed")
        logger.info("  IMPORTANT: CSV 'observed_tt' contains PHYSICAL travel times only")
        logger.info("             Binary files contain: physical_tt + configured_origin_time")

    else:
        logger.warning("No travel time data to write to CSV")


def create_geometry_and_traveltime_files(config: DictConfig,
                                        events_df: pd.DataFrame,
                                        stations_df: pd.DataFrame,
                                        picks_df: pd.DataFrame) -> None:
    """Create geometry and travel time files from Cartesian coordinates.

    Args:
        config: Full configuration.
        events_df: DataFrame with event Cartesian coordinates.
        stations_df: DataFrame with station Cartesian coordinates.
        picks_df: DataFrame with pick data.
    """
    logger = logging.getLogger(__name__)
    geom_config = config.geometry_traveltime
    origin_time_config = config.origin_time

    if not geom_config.enabled:
        logger.info("Geometry/traveltime generation disabled")
        return

    logger.info("--- Starting Geometry and Travel Time File Generation ---")

    # Create output directories
    ensure_directory_exists(Path(geom_config.geometry_dir))
    ensure_directory_exists(Path(geom_config.traveltime_dir))
    if geom_config.generate_source_binaries:
        ensure_directory_exists(Path(geom_config.source_dir))
    if geom_config.generate_travel_time_csv:
        ensure_directory_exists(Path(geom_config.csv_output_dir))

    # Parse picks with Cartesian coordinates
    event_data, station_lookup = parse_picks_with_cartesian(
        events_df, stations_df, picks_df, config
    )

    if len(event_data) == 0:
        logger.warning("No valid events found after filtering. No files were written.")
        return

    # Write source binary files
    write_source_binary_files(event_data, config, origin_time_config)

    # Write geometry and traveltime files
    write_output_files(event_data, config, origin_time_config)

    # Write comprehensive travel time CSV
    write_travel_time_csv(event_data, config)

    # Print data processing summary
    phase_type = geom_config.get('phase_type', 'both').lower()
    num_sources = len(event_data)

    logger.info("")
    logger.info("=" * 60)
    logger.info("GEOMETRY & TRAVELTIME GENERATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Phase type: {phase_type}")
    logger.info(f"Number of sources (filtered events): {num_sources}")

    if phase_type == 'both':
        total_pairs = sum(len(data['p_picks']) for data in event_data.values())
        logger.info(f"Total consistent P/S station pairs: {total_pairs}")
    elif phase_type == 'p':
        total_picks = sum(len(data['p_picks']) for data in event_data.values())
        logger.info(f"Total P-wave picks: {total_picks}")
    elif phase_type == 's':
        total_picks = sum(len(data['s_picks']) for data in event_data.values())
        logger.info(f"Total S-wave picks: {total_picks}")

    if geom_config.generate_source_binaries:
        logger.info(f"Source binary files generated in: {geom_config.source_dir}")
        logger.info(f"  - Format: travel_time")
        origin_time_desc = f"{origin_time_config.mode}"
        if origin_time_config.mode == "fixed":
            origin_time_desc += f" (value={origin_time_config.get('fixed_value', 0.0)}s)"
        logger.info(f"  - Origin times: {origin_time_desc}")
        logger.info(f"  - Files: sx.bin, sy.bin, sz.bin, st0.bin")

    logger.info("=" * 60)
    logger.info("Geometry and traveltime generation completed successfully")
    logger.info("=" * 60)
