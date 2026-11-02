"""
Post-processing functions for LATTE travel time outputs - OPTIMIZED VERSION.
Converts travel time binary files to CSV format for analysis.
Supports source-receiver exchange for reciprocity-based inversions.
Supports source correction for corrected origin times.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from omegaconf import DictConfig
from tqdm import tqdm

from utils.io_utils import ensure_directory_exists, resolve_csv_path
from utils.traveltime_io import read_traveltime_csv, write_traveltime_csv
from utils.exchange import resolve_layout, match_stations, station_coordinates, check_binary_counts, check_array_size


def load_shot_mapping(shot_map_file: str) -> Dict[int, str]:
    """Load shot ID to event ID mapping."""
    print(f"Loading shot mapping: {shot_map_file}")
    
    shot_map = {}
    try:
        df = pd.read_csv(shot_map_file, sep='\t')
        for _, row in df.iterrows():
            shot_map[int(row['Shot_ID'])] = str(row['Original_Event_ID'])
        print(f"  Loaded {len(shot_map)} shot mappings")
    except Exception as e:
        print(f"Error loading shot mapping: {e}")
        raise
    
    return shot_map


def read_travel_time_binary(bin_path: str) -> np.ndarray:
    """Read travel time binary file with automatic format detection."""
    if not os.path.exists(bin_path):
        return None
    
    if os.path.getsize(bin_path) == 0:
        return None
    
    try:
        # LATTE writes float32 only
        return np.fromfile(bin_path, dtype=np.float32)
    except Exception as e:
        print(f"Error reading {bin_path}: {e}")
        return None


def load_source_correction_binaries(source_correction_file: str, mode: str = "standard") -> Dict[int, float]:
    """
    Load corrected source origin times from binary files.

    Args:
        source_correction_file: Path to source origin time bin file
        mode: "standard" (indices = shots) or "exchange" (indices = shots, applied per shot_id)

    Returns:
        Dictionary mapping shot_id (1-based) to corrected origin time.
    """
    print(f"\nLoading source correction binaries from: {source_correction_file}")

    st0_file = source_correction_file

    if not os.path.exists(st0_file):
        raise FileNotFoundError(f"Source correction file not found: {st0_file}")

    try:
        # Read binary file as single precision (float32)
        st0_data = np.fromfile(st0_file, dtype=np.float32)
        if len(st0_data) == 0 or np.any(np.isnan(st0_data)):
            raise ValueError(f"Invalid data in {st0_file}")
        
        # Create mapping from shot_id (1-indexed) to corrected origin time
        corrected_origin_times = {}
        for idx, origin_time in enumerate(st0_data, start=1):
            corrected_origin_times[idx] = origin_time
        
        print(f"  Loaded {len(corrected_origin_times)} corrected origin times")
        print(f"  Origin time range: {min(st0_data):.3f} to {max(st0_data):.3f} s")
        print(f"  Mode: {mode} (corrections indexed by shot_id)")
        
        # Check if all origin times are zero (no correction applied)
        if np.allclose(st0_data, 0.0):
            print("  Warning: All corrected origin times are zero. Source correction may not have been applied.")
        
        return corrected_origin_times
        
    except Exception as e:
        print(f"Error loading source correction file {st0_file}: {e}")
        raise


def create_ground_truth_lookup(df_ground_truth: pd.DataFrame) -> Dict[Tuple[int, str, str], pd.Series]:
    """
    Create optimized lookup dictionary for ground truth data.
    
    Args:
        df_ground_truth: Ground truth DataFrame
        
    Returns:
        Dictionary mapping (shot_id, station, phase) -> row data
    """
    print("\nCreating ground truth lookup table...")
    gt_lookup = {}
    
    for _, row in df_ground_truth.iterrows():
        key = (row['shot_id'], row['station'], row['phase'])
        gt_lookup[key] = row
    
    print(f"  Created lookup table with {len(gt_lookup)} entries")
    return gt_lookup



def create_synthetic_travel_time_csv(config: DictConfig) -> None:
    """
    Create synthetic travel time CSV from binary files (standard mode - no exchange).

    Processing steps:
    1. Read synthetic travel times from LATTE binary outputs (contain: physical_tt + configured_origin_time)
    2. Subtract configured origin time to get physical travel time
    3. Optionally apply source correction if enabled
    4. Compare with observed travel times (which are already physical travel times from CSV)
    """
    phase_type = config.get('phase_type', 'both')
    print("\n=== CREATING SYNTHETIC TRAVEL TIME CSV FROM BINARIES (STANDARD MODE) ===")
    print(f"Phase type: {phase_type}")

    # Load ground truth CSV for structure and metadata
    print(f"Loading ground truth travel time CSV: {config.ground_truth_csv}")
    df_ground_truth = read_traveltime_csv(config.ground_truth_csv)

    # Load shot mapping
    shot_map = load_shot_mapping(config.shot_map_file)

    # Load corrected origin times if source correction is enabled
    corrected_origin_times = None
    if config.get('use_source_correction', False):
        corrected_origin_times = load_source_correction_binaries(config.source_correction_file, mode="standard")
        print("Source correction enabled: Will use corrected origin times")

    # Prepare data for synthetic CSV
    csv_data = []
    missing_shots = []

    print("\nProcessing synthetic travel time binaries...")

    # Get unique shots from ground truth data
    unique_shots = sorted(df_ground_truth['shot_id'].unique())

    for shot_id in tqdm(unique_shots, desc="Processing shots"):
        # Get ground truth data for this shot
        shot_gt_data = df_ground_truth[df_ground_truth['shot_id'] == shot_id]
        
        if shot_gt_data.empty:
            continue
            
        # Get event ID for this shot
        event_id = shot_map.get(shot_id, str(shot_id))
        
        # Get configured origin time from ground truth (used to adjust synthetic times)
        configured_origin_time = shot_gt_data.iloc[0]['origin_time_s'] if not shot_gt_data.empty else 0.0

        # Get corrected origin time if source correction is enabled
        origin_time_correction = configured_origin_time
        if corrected_origin_times is not None:
            origin_time_correction = corrected_origin_times.get(shot_id, configured_origin_time)

        # Process phases based on phase_type
        phases_to_process = []
        if phase_type in ['both', 'p']:
            phases_to_process.append('P')
        if phase_type in ['both', 's']:
            phases_to_process.append('S')

        for phase in phases_to_process:
            phase_lower = phase.lower()
            
            # Get ground truth data for this phase
            phase_gt_data = shot_gt_data[shot_gt_data['phase'] == phase]
            
            if phase_gt_data.empty:
                continue
            
            # Load synthetic travel time binary
            bin_file = os.path.join(config.synthetic_tt_dir, f"shot_{shot_id}_traveltime_{phase_lower}.bin")
            synthetic_times = read_travel_time_binary(bin_file)

            if synthetic_times is None:
                print(f"  Warning: No synthetic data for shot {shot_id}, phase {phase}")
                if shot_id not in missing_shots:
                    missing_shots.append(shot_id)
                continue
            
            # Sort ground truth data by station to ensure consistent ordering
            phase_gt_data_sorted = phase_gt_data.sort_values('station')
            
            # Check if counts match
            num_synthetic = len(synthetic_times)
            num_observed = len(phase_gt_data_sorted)
            
            if num_synthetic != num_observed:
                print(f"  Warning: Count mismatch for shot {shot_id}, phase {phase}: "
                      f"synthetic={num_synthetic}, observed={num_observed}")
                continue
            
            # Create records for each station
            for i, (_, gt_row) in enumerate(phase_gt_data_sorted.iterrows()):
                synthetic_tt = synthetic_times[i]
                observed_tt = gt_row["observed_tt"]

                if corrected_origin_times is not None:
                    # Source correction: binary st0 is the absolute corrected origin time.
                    # del_t shifts the observed TT from the old origin time to the corrected one.
                    corrected_ot = origin_time_correction
                    del_t = configured_origin_time - corrected_ot
                    synthetic_tt = synthetic_tt - corrected_ot
                    observed_tt = observed_tt + del_t
                else:
                    # No source correction: subtract configured origin time.
                    # Binary contains physical_tt + configured_origin_time.
                    synthetic_tt = synthetic_tt - configured_origin_time
         
                # Skip unrealistic synthetic values
                if synthetic_tt > 1e10:
                    print(f"  Warning: Unrealistic synthetic time {synthetic_tt:.2e}s for shot {shot_id}, station {gt_row['station']}")
                    continue
                
                csv_record = {
                    # Identifiers (from ground truth)
                    'shot_id': shot_id,
                    'event_id': event_id,
                    'station': gt_row['station'],
                    'phase': phase,
                    
                    # Travel times (residual = observed_tt - synthetic_tt is computed where needed)
                    'observed_tt': observed_tt,
                    'synthetic_tt': synthetic_tt,
                    
                    # Metadata (from ground truth)
                    'weight': gt_row['weight'],
                    
                    # Coordinates (from ground truth; Cartesian coordinates in metres)
                    'event_longitude': gt_row.get('event_longitude', np.nan),
                    'event_latitude': gt_row.get('event_latitude', np.nan),
                    'event_depth_km': gt_row.get('event_depth_km', np.nan),
                    'station_longitude': gt_row.get('station_longitude', np.nan),
                    'station_latitude': gt_row.get('station_latitude', np.nan),
                    'station_depth_km': gt_row.get('station_depth_km', np.nan),
                    'event_x': gt_row.get('event_x', np.nan),
                    'event_y': gt_row.get('event_y', np.nan),
                    'event_z': gt_row.get('event_z', np.nan),
                    'station_x': gt_row.get('station_x', np.nan),
                    'station_y': gt_row.get('station_y', np.nan),
                    'station_z': gt_row.get('station_z', np.nan),
                    'origin_time_s': gt_row.get('origin_time_s', np.nan),
                }
                
                # Add source correction metadata
                if corrected_origin_times is not None:
                    csv_record['corrected_origin_time_s'] = origin_time_correction
                
                # Add any additional columns that might exist in ground truth
                for col in gt_row.index:
                    if col not in csv_record:
                        csv_record[col] = gt_row[col]
                
                csv_data.append(csv_record)
    
    # Convert to DataFrame and save
    if csv_data:
        df_synthetic = pd.DataFrame(csv_data)

        ensure_directory_exists(os.path.dirname(config.synthetic_tt_output_csv))
        write_traveltime_csv(df_synthetic, config.synthetic_tt_output_csv, data_type='synthetic_traveltime')

        print(f"\nSynthetic travel time CSV written: {config.synthetic_tt_output_csv}")
        print(f"  Total records: {len(df_synthetic)}")
        print(f"  Unique events: {df_synthetic['event_id'].nunique()}")
        print(f"  Phase type: {phase_type}")
        print(f"  Unique stations: {df_synthetic['station'].nunique()}")
        print(f"  P-wave picks: {len(df_synthetic[df_synthetic['phase'] == 'P'])}")
        print(f"  S-wave picks: {len(df_synthetic[df_synthetic['phase'] == 'S'])}")
        
        # Calculate basic statistics
        weights = df_synthetic['weight']
        residual_s = df_synthetic['observed_tt'] - df_synthetic['synthetic_tt']
        rms_error = np.sqrt(np.sum(weights * residual_s**2) / np.sum(weights))
        print(f"  Weighted RMS residual: {rms_error:.4f} s")
        
        if corrected_origin_times is not None:
            print(f"  Source correction applied: YES")
            if 'corrected_origin_time_s' in df_synthetic.columns:
                print(f"  Origin time correction range: {df_synthetic['corrected_origin_time_s'].min():.3f} to {df_synthetic['corrected_origin_time_s'].max():.3f} s")
        else:
            print(f"  Source correction applied: NO")

        # Report missing shots
        if missing_shots:
            missing_shots_sorted = sorted(set(missing_shots))
            print(f"\n  WARNING: {len(missing_shots_sorted)} shots had missing binary files:")
            print(f"    Missing shot IDs: {missing_shots_sorted}")

    else:
        print("Warning: No synthetic travel time data was processed")


def create_synthetic_travel_time_csv_with_exchange(config: DictConfig) -> None:
    """
    Create synthetic travel time CSV using source-receiver exchange - OPTIMIZED.

    Processing steps:
    1. Read synthetic travel times from LATTE binary outputs (contain: physical_tt + configured_origin_time)
    2. Subtract configured origin time to get physical travel time
    3. Optionally apply source correction if enabled
    4. Compare with observed travel times (which are already physical travel times from CSV)
    """
    phase_type = config.get('phase_type', 'both')
    print("\n=== CREATING SYNTHETIC TRAVEL TIME CSV WITH SOURCE-RECEIVER EXCHANGE ===")
    print(f"Phase type: {phase_type}")

    # Load ground truth CSV
    print(f"Loading ground truth travel time CSV: {config.ground_truth_csv}")
    df_ground_truth = read_traveltime_csv(config.ground_truth_csv)

    print(f"  Ground truth: {len(df_ground_truth)} records")
    print(f"  Unique shots: {df_ground_truth['shot_id'].nunique()}")
    print(f"  Unique stations: {df_ground_truth['station'].nunique()}")

    # OPTIMIZATION: Create lookup dictionary for O(1) access
    gt_lookup = create_ground_truth_lookup(df_ground_truth)

    # Rebuild LATTE's source-receiver exchange (optionally checked against exchange_geometry.txt)
    tolerance = config.get('coordinate_tolerance', 1.0)
    layout = resolve_layout(
        df_ground_truth,
        geometry_dir=config.get('geometry_dir', None),
        geometry_file=config.get('geometry_file', 'geometry.txt'),
        exchange_geometry_file=config.get('exchange_geometry_file', None),
        filtered_shot_ids=config.get('filtered_shot_ids', None) or [],
        filtered_receiver_ids=config.get('filtered_receiver_ids', None) or [],
        tolerance=tolerance,
    )
    print(f"  Exchange layout from {layout.source}: {layout.n_virtual} virtual sources, {layout.n_shots} shots")
    virtual_to_station = match_stations(layout.virtual_coords, station_coordinates(df_ground_truth), layout.matching_tolerance(tolerance))
    unmatched = layout.n_virtual - len(virtual_to_station)
    if unmatched:
        print(f"  Warning: {unmatched} virtual sources could not be matched to a station within {tolerance} m")

    # Load source correction if enabled (pre-load all corrections)
    corrected_origin_times = None
    if config.get('use_source_correction', False):
        corrected_origin_times = load_source_correction_binaries(
            config.source_correction_file,
            mode="exchange",
        )
        # st0 holds one value per kept shot (array index), not per original shot id
        check_array_size(layout, "source correction file", len(corrected_origin_times))
        corrected_origin_times = {layout.shot_ids[k - 1]: v for k, v in corrected_origin_times.items()}
        print("Source correction enabled")

    # Array index (0-based) -> shot ID, in LATTE's (compacted) shot order
    array_index_to_shot_id = dict(enumerate(layout.shot_ids))

    # Count binary files to verify
    bin_files = sorted([f for f in os.listdir(config.synthetic_tt_dir)
                       if f.startswith('shot_') and f.endswith('.bin')])
    print(f"\nFound {len(bin_files)} binary files in {config.synthetic_tt_dir}")
    print(f"Expected {layout.n_virtual} files per phase (one per virtual source)")
    
    # Create output records matching ground truth
    output_records = []
    missing_pairs = []
    
    # Determine phases to process based on phase_type
    phases_to_process = []
    if phase_type in ['both', 'p']:
        phases_to_process.append('P')
    if phase_type in ['both', 's']:
        phases_to_process.append('S')

    # Number of binaries must equal the number of virtual sources
    check_binary_counts(layout, config.synthetic_tt_dir, phases_to_process)

    print("\nProcessing binary files...")

    # Process each binary file (one per virtual station per phase)
    for phase in phases_to_process:
        phase_lower = phase.lower()
        phase_bin_files = [f for f in bin_files if f.endswith(f'_traveltime_{phase_lower}.bin')]
        
        print(f"\nProcessing {phase}-wave files: {len(phase_bin_files)} files")

        print(f"No of bin files: {len(phase_bin_files)}")
        
        for bin_file in tqdm(phase_bin_files, desc=f"Processing {phase}-wave"):
            # Extract virtual shot ID from filename: shot_{ID}_traveltime_p.bin
            virtual_shot_id = int(bin_file.split('_')[1])
            
            # Get actual station name(s) for this virtual shot (may be multiple co-located stations)
            station_names = virtual_to_station.get(virtual_shot_id)
            if station_names is None:
                print(f"station_names could not be found for virtual shot id: {virtual_shot_id}")
                continue
            
            # Read binary file
            bin_path = os.path.join(config.synthetic_tt_dir, bin_file)
            travel_times = read_travel_time_binary(bin_path)

            if travel_times is None:
                continue

            # CRITICAL: one value per kept shot
            check_array_size(layout, bin_file, len(travel_times))

            # Process each array element using proper shot ID mapping
            for array_idx in range(len(travel_times)):
                # Map array index (0-based) to actual shot ID (1-based)
                shot_id = array_index_to_shot_id[array_idx]
                synthetic_tt_raw = travel_times[array_idx]

                # Skip unrealistic values
                if synthetic_tt_raw > 1e10 or synthetic_tt_raw < -1e10:
                    continue

                # Create a record for each co-located station (usually just one,
                # but multiple stations may share the same geographic location)
                for station_name in station_names:
                    # OPTIMIZATION: O(1) lookup instead of DataFrame filtering
                    key = (shot_id, station_name, phase)
                    gt_row = gt_lookup.get(key)

                    if gt_row is None:
                        missing_pairs.append(key)
                        continue

                    # Get configured origin time for this shot
                    configured_origin_time = gt_row['origin_time_s']

                    observed_tt = gt_row['observed_tt']
                    if corrected_origin_times is not None:
                        # Source correction: binary st0 is the absolute corrected origin time.
                        # del_t shifts the observed TT from the old origin time to the corrected one.
                        origin_correction = corrected_origin_times.get(shot_id, configured_origin_time)
                        del_t = configured_origin_time - origin_correction
                        synthetic_tt = synthetic_tt_raw - origin_correction
                        observed_tt = observed_tt + del_t
                    else:
                        # No source correction: subtract configured origin time.
                        # Binary contains physical_tt + configured_origin_time.
                        synthetic_tt = synthetic_tt_raw - configured_origin_time
                        origin_correction = 0.0

                    # Create output record matching ground truth format
                    # (residual = observed_tt - synthetic_tt is computed where needed)
                    record = {
                        'shot_id': shot_id,
                        'event_id': gt_row['event_id'],
                        'station': station_name,
                        'phase': phase,
                        'observed_tt': observed_tt,
                        'synthetic_tt': synthetic_tt,
                        'weight': gt_row['weight'],
                        'event_longitude': gt_row['event_longitude'],
                        'event_latitude': gt_row['event_latitude'],
                        'event_depth_km': gt_row['event_depth_km'],
                        'station_longitude': gt_row['station_longitude'],
                        'station_latitude': gt_row['station_latitude'],
                        'station_depth_km': gt_row['station_depth_km'],
                        'event_x': gt_row['event_x'],
                        'event_y': gt_row['event_y'],
                        'event_z': gt_row['event_z'],
                        'station_x': gt_row['station_x'],
                        'station_y': gt_row['station_y'],
                        'station_z': gt_row['station_z'],
                        'origin_time_s': gt_row['origin_time_s'],
                    }

                    # Carry over any further ground-truth columns (e.g. magnitude), as in standard mode
                    for col in gt_row.index:
                        if col not in record:
                            record[col] = gt_row[col]

                    # Add source correction if applied
                    if corrected_origin_times is not None:
                        record['corrected_origin_time_s'] = origin_correction

                    output_records.append(record)
    
    # Convert to DataFrame
    if output_records:
        df_output = pd.DataFrame(output_records)

        # Sort to match ground truth order
        df_output = df_output.sort_values(['shot_id', 'station', 'phase']).reset_index(drop=True)

        # Save output
        ensure_directory_exists(os.path.dirname(config.synthetic_tt_output_csv))
        write_traveltime_csv(df_output, config.synthetic_tt_output_csv, data_type='synthetic_traveltime')
        
        print(f"\n=== OUTPUT SUMMARY ===")
        print(f"Synthetic travel time CSV written: {config.synthetic_tt_output_csv}")
        print(f"  Total records: {len(output_records)}")
        print(f"  Unique shots: {df_output['shot_id'].nunique()}")
        print(f"  Unique stations: {df_output['station'].nunique()}")
        print(f"  P-wave picks: {len(df_output[df_output['phase'] == 'P'])}")
        print(f"  S-wave picks: {len(df_output[df_output['phase'] == 'S'])}")
        
        # Statistics
        weights = df_output['weight']
        residual_s = df_output['observed_tt'] - df_output['synthetic_tt']
        rms_error = np.sqrt(np.sum(weights * residual_s**2) / np.sum(weights))
        mae = np.mean(np.abs(residual_s))
        print(f"  Weighted RMS residual: {rms_error:.4f} s")
        print(f"  MAE residual: {mae:.4f} s")
        
        if corrected_origin_times is not None:
            print(f"  Source correction applied: YES")
        
        if missing_pairs:
            print(f"\n  Warning: {len(missing_pairs)} source-receiver pairs in binaries not found in ground truth")
        
        # Compare coverage with ground truth
        gt_pairs = set(zip(df_ground_truth['shot_id'], df_ground_truth['station'], df_ground_truth['phase']))
        output_pairs = set(zip(df_output['shot_id'], df_output['station'], df_output['phase']))
        coverage = len(output_pairs) / len(gt_pairs) * 100
        print(f"  Coverage: {coverage:.1f}% of ground truth pairs")
        
    else:
        print("Error: No output records generated")