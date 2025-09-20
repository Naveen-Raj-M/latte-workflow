"""
Geometry and travel time processing functions for LATTE preprocessing.
Functions for creating geometry files and travel time binary files from phase picks.
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Any
from omegaconf import DictConfig
from pathlib import Path

from utils.io_utils import ensure_directory_exists


def _calculate_time_in_seconds(jday_str: str, hour_str: str, min_str: str, sec_str: str) -> float:
    """Helper function to calculate absolute time in seconds from start of year."""
    return (int(jday_str) - 1) * 86400 + int(hour_str) * 3600 + int(min_str) * 60 + float(sec_str)


def _convert_seconds_to_time_components(time_in_seconds: float) -> Tuple[int, int, int, float]:
    """Convert seconds from start of year back to jday, hour, min, sec format."""
    jday = int(time_in_seconds // 86400) + 1
    remaining_seconds = time_in_seconds % 86400
    hour = int(remaining_seconds // 3600)
    remaining_seconds = remaining_seconds % 3600
    min_val = int(remaining_seconds // 60)
    sec = remaining_seconds % 60
    return jday, hour, min_val, sec


def load_events_to_discard(file_path: str) -> Set[str]:
    """
    Load event IDs to discard from a text file in Shot_ID\tOriginal_Event_ID format.
    Returns a set of event IDs for fast lookup.
    """
    events_to_discard = set()
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            line = line.strip()
            if i == 0 and ('Shot_ID' in line or 'Original_Event_ID' in line):
                continue  # Skip header
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    event_id = parts[1].strip()
                    events_to_discard.add(event_id)
        
        print(f"Loaded {len(events_to_discard)} event IDs to discard from {file_path}")
        return events_to_discard
    
    except FileNotFoundError:
        print(f"Warning: Events discard file not found at {file_path}. No events will be discarded.")
        return set()
    except Exception as e:
        print(f"Error reading events discard file {file_path}: {e}")
        return set()


def load_stations_to_discard(file_path: str) -> Set[str]:
    """
    Load station IDs to discard from a text file (one station ID per line).
    Returns a set of station IDs for fast lookup.
    """
    stations_to_discard = set()
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                stations_to_discard.add(line)
        
        print(f"Loaded {len(stations_to_discard)} station IDs to discard from {file_path}")
        return stations_to_discard
    
    except FileNotFoundError:
        print(f"Warning: Stations discard file not found at {file_path}. No stations will be discarded.")
        return set()
    except Exception as e:
        print(f"Error reading stations discard file {file_path}: {e}")
        return set()


def parse_and_filter_data_original(config: DictConfig, station_map: Dict, event_map: Dict, 
                                 events_to_discard_set: Set[str], stations_to_discard_set: Set[str]) -> Tuple:
    """
    Parses phase picks from original phasePickings.txt format and filters for stations 
    that have BOTH a valid P and a valid S pick.
    """
    temp_events = defaultdict(lambda: {'origin_time_s': 0.0, 'p_picks': {}, 's_picks': {}, 'header_info': None})
    total_picks_in_file = 0
    unrealistic_velocity_count = 0
    discarded_events_count = 0
    discarded_stations_count = 0
    
    try:
        with open(config.phase_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Phase file not found at {config.phase_file}")
        return None, 0, 0, 0, 0, 0

    current_event_id = None
    for line in lines:
        line = line.strip()
        parts = line.split()
        if not line or not parts:
            current_event_id = None
            continue

        if len(parts) == 9:  # Event header line
            try:
                current_event_id = parts[8]
                
                # Check if event should be discarded
                if config.filter_events and current_event_id in events_to_discard_set:
                    print(f"# Warning: Event {current_event_id} discarded (reason: in events discard file).")
                    discarded_events_count += 1
                    current_event_id = None
                    continue
                
                origin_time_s = _calculate_time_in_seconds(parts[1], parts[2], parts[3], parts[4])
                temp_events[current_event_id]['origin_time_s'] = origin_time_s
                temp_events[current_event_id]['header_info'] = parts
            except (IndexError, ValueError):
                current_event_id = None
            continue

        if current_event_id and len(parts) >= 7:  # Pick line
            total_picks_in_file += 1
            try:
                station, _, jday, hour, min_time, sec, phase = parts[:7]
                weight = float(parts[7]) if len(parts) > 7 else 0.0

                # Check if station should be discarded
                if config.filter_stations and station in stations_to_discard_set:
                    print(f"# Warning: Event {current_event_id}, station {station} skipped (reason: in stations discard file).")
                    discarded_stations_count += 1
                    continue

                # Filtering logic
                if station not in station_map:
                    print(f"# Warning: Event {current_event_id}, station {station} skipped (reason: not found in station_csv).")
                    continue
                
                sx, sy, sz = station_map[station]
                if sx < 0 or sy < 0 or sz < 0:
                    print(f"# Warning: Event {current_event_id}, station {station} skipped (reason: invalid negative coordinates).")
                    continue

                pick_time_s = _calculate_time_in_seconds(jday, hour, min_time, sec)
                travel_time = pick_time_s - temp_events[current_event_id]['origin_time_s']
                if travel_time < -43200: 
                    travel_time += 86400.0
                
                if travel_time < 0:
                    print(f"# Warning: Event {current_event_id}, station {station} skipped (reason: negative travel time: {travel_time:.2f}s).")
                    continue
                
                # Velocity Check
                ex, ey, ez = event_map.get(current_event_id, (None, None, None))
                if ex is None:
                    print(f"# Warning: Event {current_event_id}, station {station} skipped (reason: event coordinates not found).")
                    continue
                
                if travel_time < 1e-6:  # Avoid division by zero
                    unrealistic_velocity_count += 1
                    print(f"# Warning: Event {current_event_id}, station {station} skipped (reason: travel time is zero).")
                    continue

                distance = np.linalg.norm(np.array([ex, ey, ez]) - np.array([sx, sy, sz]))
                apparent_velocity = distance / travel_time
                
                is_realistic = False
                if phase.upper() == 'P':
                    if config.p_velocity_range_m_s[0] <= apparent_velocity <= config.p_velocity_range_m_s[1]:
                        is_realistic = True
                elif phase.upper() == 'S':
                    if config.s_velocity_range_m_s[0] <= apparent_velocity <= config.s_velocity_range_m_s[1]:
                        is_realistic = True

                if not is_realistic:
                    unrealistic_velocity_count += 1
                    print(f"# Warning: Event {current_event_id}, station {station} ({phase}-phase) skipped (reason: unrealistic velocity: {apparent_velocity:.1f} m/s).")
                    continue

                pick_data = {
                    'station': station, 
                    'coords': (sx, sy, sz), 
                    'travel_time': travel_time, 
                    'weight': weight,
                    'original_parts': parts,
                    'pick_time_s': pick_time_s
                }
                
                if phase.upper() == 'P':
                    temp_events[current_event_id]['p_picks'][station] = pick_data
                elif phase.upper() == 'S':
                    temp_events[current_event_id]['s_picks'][station] = pick_data
            except (IndexError, ValueError):
                pass

    # Final Filtering: Keep only events and stations with BOTH P and S picks
    final_events = {}
    print("\nFiltering for stations with both valid P and S picks...")
    for eid, data in temp_events.items():
        valid_p_stations = set(data['p_picks'].keys())
        valid_s_stations = set(data['s_picks'].keys())
        
        common_stations = sorted(list(valid_p_stations.intersection(valid_s_stations)))

        if common_stations:
            final_events[eid] = {
                'header_info': data['header_info'],
                'origin_time_s': data['origin_time_s'],
                'p_picks': [data['p_picks'][stn] for stn in common_stations],
                's_picks': [data['s_picks'][stn] for stn in common_stations]
            }
    
    final_written_pairs = sum(len(data['p_picks']) for data in final_events.values())
    return final_events, total_picks_in_file, final_written_pairs, unrealistic_velocity_count, discarded_events_count, discarded_stations_count


def parse_and_filter_data_corrected(config: DictConfig, station_map: Dict, event_map: Dict,
                                  events_to_discard_set: Set[str], stations_to_discard_set: Set[str]) -> Tuple:
    """
    Parses phase picks from corrected .dat format and filters for stations 
    that have BOTH a valid P and a valid S pick.
    
    Format:
    - First line: year julian_day hour minute second lat lon depth event_id weighted_rms
    - Following lines: station year julian_day hour minute second phase_type uncertainty residual
    """
    temp_events = defaultdict(lambda: {'origin_time_s': 0.0, 'p_picks': {}, 's_picks': {}, 'header_info': None, 'weighted_rms': 0.0})
    total_picks_in_file = 0
    unrealistic_velocity_count = 0
    discarded_events_count = 0
    discarded_stations_count = 0
    
    try:
        with open(config.corrected_phase_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Phase file not found at {config.corrected_phase_file}")
        return None, 0, 0, 0, 0, 0

    current_event_id = None
    current_event_origin_time = None
    
    for line in lines:
        line = line.strip()
        parts = line.split()
        if not line or not parts:
            current_event_id = None
            continue

        # Check if this is an event line (10 columns)
        if len(parts) == 10:
            try:
                year = int(parts[0])
                julian_day = int(parts[1])
                hour = int(parts[2])
                minute = int(parts[3])
                second = float(parts[4])
                lat = float(parts[5])
                lon = float(parts[6])
                depth = float(parts[7])
                current_event_id = parts[8]
                weighted_rms = float(parts[9])
                
                # Check if event should be discarded
                if config.filter_events and current_event_id in events_to_discard_set:
                    print(f"# Warning: Event {current_event_id} discarded (reason: in events discard file).")
                    discarded_events_count += 1
                    current_event_id = None
                    continue
                
                current_event_origin_time = _calculate_time_in_seconds(julian_day, hour, minute, second)
                temp_events[current_event_id]['origin_time_s'] = current_event_origin_time
                temp_events[current_event_id]['weighted_rms'] = weighted_rms
                
                # Create header info in original format for consistency
                temp_events[current_event_id]['header_info'] = [
                    str(year), str(julian_day), str(hour), str(minute), f"{second:.4f}",
                    f"{lat:.5f}", f"{lon:.5f}", f"{depth:.4f}", current_event_id
                ]
                
            except (IndexError, ValueError) as e:
                print(f"Error parsing event line: {line}, error: {e}")
                current_event_id = None
            continue

        # Check if this is a station pick line (9 columns)
        elif len(parts) == 9 and current_event_id:
            total_picks_in_file += 1
            try:
                station = parts[0]
                pick_year = int(parts[1])
                pick_jday = int(parts[2])
                pick_hour = int(parts[3])
                pick_minute = int(parts[4])
                pick_second = float(parts[5])
                phase = parts[6]
                uncertainty = float(parts[7])
                residual = float(parts[8])

                # Check if station should be discarded
                if config.filter_stations and station in stations_to_discard_set:
                    print(f"# Warning: Event {current_event_id}, station {station} skipped (reason: in stations discard file).")
                    discarded_stations_count += 1
                    continue

                # Filtering logic (same as original)
                if station not in station_map:
                    print(f"# Warning: Event {current_event_id}, station {station} skipped (reason: not found in station_csv).")
                    continue
                
                sx, sy, sz = station_map[station]
                if sx < 0 or sy < 0 or sz < 0:
                    print(f"# Warning: Event {current_event_id}, station {station} skipped (reason: invalid negative coordinates).")
                    continue

                pick_time_s = _calculate_time_in_seconds(pick_jday, pick_hour, pick_minute, pick_second)
                travel_time = pick_time_s - current_event_origin_time
                if travel_time < -43200: 
                    travel_time += 86400.0
                
                if travel_time < 0:
                    print(f"# Warning: Event {current_event_id}, station {station} skipped (reason: negative travel time: {travel_time:.2f}s).")
                    continue
                
                # Velocity Check
                ex, ey, ez = event_map.get(current_event_id, (None, None, None))
                if ex is None:
                    print(f"# Warning: Event {current_event_id}, station {station} skipped (reason: event coordinates not found).")
                    continue
                
                if travel_time < 1e-6:  # Avoid division by zero
                    unrealistic_velocity_count += 1
                    print(f"# Warning: Event {current_event_id}, station {station} skipped (reason: travel time is zero).")
                    continue

                distance = np.linalg.norm(np.array([ex, ey, ez]) - np.array([sx, sy, sz]))
                apparent_velocity = distance / travel_time
                
                is_realistic = False
                if phase.upper() == 'P':
                    if config.p_velocity_range_m_s[0] <= apparent_velocity <= config.p_velocity_range_m_s[1]:
                        is_realistic = True
                elif phase.upper() == 'S':
                    if config.s_velocity_range_m_s[0] <= apparent_velocity <= config.s_velocity_range_m_s[1]:
                        is_realistic = True

                if not is_realistic:
                    unrealistic_velocity_count += 1
                    print(f"# Warning: Event {current_event_id}, station {station} ({phase}-phase) skipped (reason: unrealistic velocity: {apparent_velocity:.1f} m/s).")
                    continue

                pick_data = {
                    'station': station, 
                    'coords': (sx, sy, sz), 
                    'travel_time': travel_time, 
                    'weight': uncertainty,  # Use uncertainty as weight
                    'residual': residual,
                    'original_parts': parts,
                    'pick_time_s': pick_time_s
                }
                
                if phase.upper() == 'P':
                    temp_events[current_event_id]['p_picks'][station] = pick_data
                elif phase.upper() == 'S':
                    temp_events[current_event_id]['s_picks'][station] = pick_data
                    
            except (IndexError, ValueError) as e:
                print(f"Error parsing pick line: {line}, error: {e}")
                pass

    # Final Filtering: Keep only events and stations with BOTH P and S picks
    final_events = {}
    print("\nFiltering for stations with both valid P and S picks...")
    for eid, data in temp_events.items():
        valid_p_stations = set(data['p_picks'].keys())
        valid_s_stations = set(data['s_picks'].keys())
        
        common_stations = sorted(list(valid_p_stations.intersection(valid_s_stations)))

        if common_stations:
            final_events[eid] = {
                'header_info': data['header_info'],
                'origin_time_s': data['origin_time_s'],
                'weighted_rms': data.get('weighted_rms', 0.0),
                'p_picks': [data['p_picks'][stn] for stn in common_stations],
                's_picks': [data['s_picks'][stn] for stn in common_stations]
            }
    
    final_written_pairs = sum(len(data['p_picks']) for data in final_events.values())
    return final_events, total_picks_in_file, final_written_pairs, unrealistic_velocity_count, discarded_events_count, discarded_stations_count


def write_source_binary_files(events: Dict, event_map: Dict, config: DictConfig) -> None:
    """
    Write source binary files (sx.bin, sy.bin, sz.bin, st0.bin) for the events.
    These files contain the source coordinates and origin times for each event.
    """
    if not config.generate_source_binaries:
        return
        
    ensure_directory_exists(config.source_dir)
    
    print(f"\nWriting source binary files to: {config.source_dir}")
    
    # Sort events to maintain consistent ordering (same as in geometry files)
    sorted_events = sorted(events.items(), key=lambda item: item[0])
    
    # Extract source coordinates and origin times
    sx_values = []
    sy_values = []
    sz_values = []
    st0_values = []
    
    for eid, data in sorted_events:
        ex, ey, ez = event_map.get(eid, (None, None, None))
        if ex is None:
            print(f"Warning: Event {eid} coordinates not found in event_map, skipping from source binaries")
            continue
            
        sx_values.append(ex)
        sy_values.append(ey)
        sz_values.append(ez)
        
        # Handle origin time based on zero_origin_time flag
        if config.zero_origin_time:
            st0_values.append(0.0)
        else:
            st0_values.append(data['origin_time_s'])
    
    # Convert to numpy arrays
    sx_array = np.array(sx_values, dtype='<f4')  # Little-endian 32-bit float
    sy_array = np.array(sy_values, dtype='<f4')
    sz_array = np.array(sz_values, dtype='<f4')
    st0_array = np.array(st0_values, dtype='<f4')
    
    # Write binary files based on format compatibility setting
    if config.source_binary_format == "travel_time":
        # Use same format as traveltime binary files
        print("Using travel_time binary format for source files")
        with open(os.path.join(config.source_dir, 'sx.bin'), 'wb') as f:
            f.write(sx_array.tobytes())
        with open(os.path.join(config.source_dir, 'sy.bin'), 'wb') as f:
            f.write(sy_array.tobytes())
        with open(os.path.join(config.source_dir, 'sz.bin'), 'wb') as f:
            f.write(sz_array.tobytes())
        with open(os.path.join(config.source_dir, 'st0.bin'), 'wb') as f:
            f.write(st0_array.tobytes())
    
    elif config.source_binary_format == "fortran":
        # Alternative format (if needed for Fortran compatibility)
        print("Using fortran binary format for source files")
        # Write as big-endian format (typical for Fortran)
        sx_array_be = sx_array.astype('>f4')
        sy_array_be = sy_array.astype('>f4')
        sz_array_be = sz_array.astype('>f4')
        st0_array_be = st0_array.astype('>f4')
        
        with open(os.path.join(config.source_dir, 'sx.bin'), 'wb') as f:
            f.write(sx_array_be.tobytes())
        with open(os.path.join(config.source_dir, 'sy.bin'), 'wb') as f:
            f.write(sy_array_be.tobytes())
        with open(os.path.join(config.source_dir, 'sz.bin'), 'wb') as f:
            f.write(sz_array_be.tobytes())
        with open(os.path.join(config.source_dir, 'st0.bin'), 'wb') as f:
            f.write(st0_array_be.tobytes())
    
    else:
        print(f"Warning: Unknown source_binary_format: {config.source_binary_format}")
        return
    
    print(f"Source binary files written:")
    print(f"  sx.bin: {len(sx_values)} values (x-coordinates)")
    print(f"  sy.bin: {len(sy_values)} values (y-coordinates)")  
    print(f"  sz.bin: {len(sz_values)} values (z-coordinates)")
    print(f"  st0.bin: {len(st0_values)} values (origin times{'=0' if config.zero_origin_time else ''})")
    
    # Print summary statistics
    if sx_values:  # Only print if we have values
        print(f"\nSource coordinate ranges:")
        print(f"  X: {min(sx_values):.1f} to {max(sx_values):.1f} m")
        print(f"  Y: {min(sy_values):.1f} to {max(sy_values):.1f} m")
        print(f"  Z: {min(sz_values):.1f} to {max(sz_values):.1f} m")
        if not config.zero_origin_time:
            print(f"  Origin times: {min(st0_values):.1f} to {max(st0_values):.1f} s")


def write_output_files(events: Dict, event_map: Dict, config: DictConfig) -> None:
    """
    Writes geometry files and travel time binary files.
    The binary files are written as a raw stream of little-endian floats to be
    compatible with np.fromfile.
    """
    ensure_directory_exists(config.geometry_dir)
    ensure_directory_exists(config.traveltime_dir)

    shot_files = []
    id_map_path = os.path.join(config.geometry_dir, "shot_to_source_id_map.txt")
    
    print(f"\nWriting consistent output files...")
    with open(id_map_path, "w") as map_f:
        map_f.write("Shot_ID\tOriginal_Event_ID\n")
        
        sorted_events = sorted(events.items(), key=lambda item: item[0])

        for idx, (eid, data) in enumerate(sorted_events, start=1):
            receivers_for_geometry = data['p_picks']
            
            geom_fname = f"shot_{idx}_geometry.txt"
            shot_files.append(geom_fname)
            map_f.write(f"{idx}\t{eid}\n")

            ex, ey, ez = event_map.get(eid, (None, None, None))
            if ex is None: 
                continue

            # Write Geometry File
            with open(os.path.join(config.geometry_dir, geom_fname), "w") as out:
                out.write(f"{idx}\n\n1\n")
                out.write(f"{ex:.6f} {ey:.6f} {ez:.6f} {0.0:.6f}\n\n")
                out.write(f"{len(receivers_for_geometry)}\n")
                for pick in receivers_for_geometry:
                    sx, sy, sz = pick['coords']
                    w = pick['weight']
                    out.write(f"{sx:.6f} {sy:.6f} {sz:.6f} {w:.6f}\n")

            # Write P-wave traveltime binary
            tt_p_fname = os.path.join(config.traveltime_dir, f"shot_{idx}_traveltime_p.bin")
            p_travel_times = [p['travel_time'] for p in data['p_picks']]
            p_array = np.array(p_travel_times, dtype='<f4')
            with open(tt_p_fname, 'wb') as f:
                f.write(p_array.tobytes())

            # Write S-wave traveltime binary
            tt_s_fname = os.path.join(config.traveltime_dir, f"shot_{idx}_traveltime_s.bin")
            s_travel_times = [s['travel_time'] for s in data['s_picks']]
            s_array = np.array(s_travel_times, dtype='<f4')
            with open(tt_s_fname, 'wb') as f:
                f.write(s_array.tobytes())

    with open(os.path.join(config.geometry_dir, "geometry.txt"), "w") as idxf:
        for shot in shot_files:
            idxf.write(shot + "\n")
    
    print("\n--- File Generation Summary ---")
    print(f"Successfully wrote {len(shot_files)} consistent shots in total.")
    print(f"Geometry index: {os.path.join(config.geometry_dir, 'geometry.txt')}")
    print(f"Source ID map: {id_map_path}")
    print(f"\nIMPORTANT: You must update the 'ns' parameter in your Fortran input to {len(shot_files)}")


def create_geometry_and_traveltime_files(config: DictConfig) -> None:
    """
    Main function to create geometry and travel time files from phase picks.
    """
    print("--- Starting Geometry and Travel Time File Generation ---")
    
    # Display configuration
    print(f"\n--- Input File Configuration ---")
    if config.use_corrected_phase_pickings:
        input_file = config.corrected_phase_file
        print(f"Using corrected phase pickings: {input_file}")
    else:
        input_file = config.phase_file
        print(f"Using original phase pickings: {input_file}")
    
    print(f"\n--- Source Binary File Configuration ---")
    print(f"Generate source binaries: {config.generate_source_binaries}")
    if config.generate_source_binaries:
        print(f"Source directory: {config.source_dir}")
        print(f"Zero origin time: {config.zero_origin_time}")
        print(f"Binary format compatibility: {config.source_binary_format}")
    
    print(f"\n--- Filtering Configuration ---")
    print(f"Event filtering enabled: {config.filter_events}")
    events_to_discard_set = set()
    if config.filter_events:
        print(f"Events discard file: {config.events_to_discard_file}")
        events_to_discard_set = load_events_to_discard(config.events_to_discard_file)
    
    print(f"Station filtering enabled: {config.filter_stations}")
    stations_to_discard_set = set()
    if config.filter_stations:
        print(f"Stations discard file: {config.stations_to_discard_file}")
        stations_to_discard_set = load_stations_to_discard(config.stations_to_discard_file)
    print("---")
    
    # Load station and event data
    station_df = pd.read_csv(config.station_csv)
    event_df = pd.read_csv(config.event_csv)
    station_map = {row.station: (row.X_grid_km * 1000, row.Y_grid_km * 1000, row.Z_grid_km * 1000) 
                   for row in station_df.itertuples()}
    event_map = {str(row.event_id): (row.X_grid_km * 1000, row.Y_grid_km * 1000, row.Z_grid_km * 1000) 
                 for row in event_df.itertuples()}

    # Parse and filter data
    if config.use_corrected_phase_pickings:
        result = parse_and_filter_data_corrected(config, station_map, event_map, 
                                               events_to_discard_set, stations_to_discard_set)
    else:
        result = parse_and_filter_data_original(config, station_map, event_map, 
                                              events_to_discard_set, stations_to_discard_set)
    
    valid_events, total_picks, written_pairs, unrealistic_picks, discarded_events, discarded_stations = result

    if valid_events:
        # Write source binary files
        write_source_binary_files(valid_events, event_map, config)
        
        # Write other output files
        write_output_files(valid_events, event_map, config)
        
        print("\n--- Data Processing Summary ---")
        print(f"Input file format: {'Corrected (.dat)' if config.use_corrected_phase_pickings else 'Original (.txt)'}")
        print(f"Total P/S picks found in phase file:       {total_picks}")
        print(f"Picks rejected due to unrealistic velocity: {unrealistic_picks}")
        if config.filter_events:
            print(f"Events discarded by filter:                 {discarded_events}")
        if config.filter_stations:
            print(f"Station picks discarded by filter:         {discarded_stations}")
        print(f"Total consistent P/S pairs written:          {written_pairs}")
        if total_picks > 0:
            final_picks_written = written_pairs * 2
            retention_rate = (final_picks_written / total_picks) * 100
            print(f"Data retention rate:                         {retention_rate:.1f}%")

        # Show additional info for corrected data
        if config.use_corrected_phase_pickings:
            print("\nNOTE: Processing corrected phase pickings with:")
            print("  - Corrected event locations and origin times")
            print("  - Travel time residuals preserved in data structure")
            print("  - Uncertainty values used as pick weights")
            # Show some sample weighted RMS values if available
            sample_events = list(valid_events.items())[:3]
            if sample_events and 'weighted_rms' in sample_events[0][1]:
                print("  - Sample weighted RMS values:")
                for eid, data in sample_events:
                    print(f"    Event {eid}: {data['weighted_rms']:.3f}")
        
        # Show source binary info if generated
        if config.generate_source_binaries:
            print(f"\nSource binary files generated in: {config.source_dir}")
            print(f"  - Format: {config.source_binary_format}")
            print(f"  - Origin times: {'Zero' if config.zero_origin_time else 'Actual'}")
            print(f"  - Files: sx.bin, sy.bin, sz.bin, st0.bin")
        
        print("\n--- Script finished successfully ---")
    else:
        print("\n--- No valid events found after filtering. No files were written. ---")