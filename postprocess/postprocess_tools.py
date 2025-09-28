"""
Post-processing functions for LATTE tomography outputs.
Converts binary files to CSV format for analysis.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from omegaconf import DictConfig, OmegaConf

def ensure_directory_exists(directory_path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def get_grid_dimensions_from_csv(df: pd.DataFrame) -> Tuple[int, int, int]:
    """Auto-detect grid dimensions from CSV data."""
    nx = int(df['i'].max())
    ny = int(df['j'].max()) 
    nz = int(df['k'].max())
    print(f"Grid dimensions detected: NX={nx}, NY={ny}, NZ={nz}")
    return nx, ny, nz

def load_velocity_binary(bin_file: str, shape: Tuple[int, int, int], velocity_type: str) -> np.ndarray:
    """Load velocity model from binary file."""
    print(f"Loading {velocity_type} from: {bin_file}")
    
    if not os.path.exists(bin_file):
        raise FileNotFoundError(f"Binary file not found: {bin_file}")
    
    # Read binary file
    data = np.fromfile(bin_file, dtype=np.float32)
    expected_size = np.prod(shape)
    
    print(f"  Read {data.size} floats, expected {expected_size}")
    
    if data.size != expected_size:
        raise ValueError(f"Binary size mismatch: got {data.size}, expected {expected_size}")
    
    # Reshape using C order
    model = data.reshape(shape, order='C')
    
    # Convert from m/s to km/s
    model = model / 1000.0
    
    print(f"  {velocity_type} range: {model[model>0].min():.3f} - {model[model>0].max():.3f} km/s")
    print(f"  Non-zero points: {np.sum(model > 0)}")
    
    return model

def apply_chapman_inverse_transform(flattened_velocity: float, depth_km: float, earth_radius_km: float = 6371.0) -> float:
    """
    Apply Chapman (1973) inverse transformation to convert flattened velocities
    back to spherical coordinates.
    
    v_spherical = exp(-z/r) * v_flattened
    """
    # Handle the case where depth might be negative (above sea level)
    depth_positive = max(depth_km, 0.001)  # Avoid numerical issues
    
    # Chapman inverse transformation
    spherical_velocity = np.exp(-depth_positive / earth_radius_km) * flattened_velocity
    
    return spherical_velocity

def create_velocity_csv_from_binaries(config: DictConfig) -> None:
    """Create velocity CSV from binary files matching the original format."""
    print("\n=== CREATING VELOCITY CSV FROM BINARIES ===")
    
    # Load original CSV for structure
    print(f"Loading original velocity CSV: {config.original_velocity_csv}")
    df_original = pd.read_csv(config.original_velocity_csv)
    nx, ny, nz = get_grid_dimensions_from_csv(df_original)
    
    # Load binary files
    vp_model = load_velocity_binary(config.vp_binary, (nx, ny, nz), "Vp")
    vs_model = load_velocity_binary(config.vs_binary, (nx, ny, nz), "Vs")
    
    # Create output DataFrame as copy of original
    df_output = df_original.copy()
    
    # Update velocities
    vp_updated_count = 0
    vs_updated_count = 0
    
    print(f"Updating velocities...")
    transformation_method = "Chapman inverse" if config.use_chapman_inverse else "Direct"
    print(f"Using {transformation_method} transformation")
    
    for idx, row in df_output.iterrows():
        i1, j1, k1 = int(row['i']), int(row['j']), int(row['k'])
        i0, j0, k0 = i1-1, j1-1, k1-1  # Convert to 0-based indexing
        
        if 0 <= i0 < nx and 0 <= j0 < ny and 0 <= k0 < nz:
            # Get flattened velocities from binary
            vp_flattened = vp_model[i0, j0, k0]
            vs_flattened = vs_model[i0, j0, k0]
            
            if vp_flattened > 0:
                if config.use_chapman_inverse:
                    depth_km = row['depth']
                    vp_final = apply_chapman_inverse_transform(vp_flattened, depth_km, config.earth_radius_km)
                else:
                    vp_final = vp_flattened
                
                df_output.at[idx, 'vp'] = vp_final
                vp_updated_count += 1
            
            if vs_flattened > 0:
                if config.use_chapman_inverse:
                    depth_km = row['depth']
                    vs_final = apply_chapman_inverse_transform(vs_flattened, depth_km, config.earth_radius_km)
                else:
                    vs_final = vs_flattened
                
                df_output.at[idx, 'vs'] = vs_final
                vs_updated_count += 1
    
    # Write output CSV
    ensure_directory_exists(os.path.dirname(config.velocity_output_csv))
    df_output.to_csv(config.velocity_output_csv, index=False)
    
    print(f"Velocity CSV written: {config.velocity_output_csv}")
    print(f"  Updated Vp: {vp_updated_count} points")
    print(f"  Updated Vs: {vs_updated_count} points")
    print(f"  Transformation: {transformation_method}")

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
    """Read travel time binary file."""
    if not os.path.exists(bin_path):
        return None
    
    if os.path.getsize(bin_path) == 0:
        return None
    
    try:
        travel_times = np.fromfile(bin_path, dtype='<f4')  # Little-endian 32-bit float
        return travel_times
    except Exception as e:
        print(f"Error reading {bin_path}: {e}")
        return None

def create_synthetic_travel_time_csv(config: DictConfig) -> None:
    """Create synthetic travel time CSV from binary files."""
    print("\n=== CREATING SYNTHETIC TRAVEL TIME CSV FROM BINARIES ===")
    
    # Load ground truth CSV for structure and metadata
    print(f"Loading ground truth travel time CSV: {config.ground_truth_csv}")
    df_ground_truth = pd.read_csv(config.ground_truth_csv)
    
    # Load shot mapping
    shot_map = load_shot_mapping(config.shot_map_file)
    
    # Prepare data for synthetic CSV
    csv_data = []
    
    print("Processing synthetic travel time binaries...")
    
    # Get unique shots from ground truth data
    unique_shots = sorted(df_ground_truth['shot_id'].unique())
    
    for shot_id in unique_shots:
        print(f"Processing shot {shot_id}...")
        
        # Get ground truth data for this shot
        shot_gt_data = df_ground_truth[df_ground_truth['shot_id'] == shot_id]
        
        if shot_gt_data.empty:
            continue
            
        # Get event ID for this shot
        event_id = shot_map.get(shot_id, str(shot_id))
        
        # Process P and S phases
        for phase in ['P', 'S']:
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
                    
                    # Travel times
                    'observed_tt': gt_row['observed_tt'],
                    'synthetic_tt': synthetic_tt,
                    'residual_s': gt_row['observed_tt'] - synthetic_tt,  # observed - synthetic
                    
                    # Metadata (from ground truth)
                    'weight': gt_row['weight'],
                    
                    # Coordinates (from ground truth)
                    'event_longitude': gt_row.get('event_longitude', np.nan),
                    'event_latitude': gt_row.get('event_latitude', np.nan),
                    'event_depth_km': gt_row.get('event_depth_km', np.nan),
                    'station_longitude': gt_row.get('station_longitude', np.nan),
                    'station_latitude': gt_row.get('station_latitude', np.nan),
                    'station_depth_km': gt_row.get('station_depth_km', np.nan),
                    'event_x_transformed': gt_row.get('event_x_transformed', np.nan),
                    'event_y_transformed': gt_row.get('event_y_transformed', np.nan),
                    'event_z_transformed': gt_row.get('event_z_transformed', np.nan),
                    'station_x_transformed': gt_row.get('station_x_transformed', np.nan),
                    'station_y_transformed': gt_row.get('station_y_transformed', np.nan),
                    'station_z_transformed': gt_row.get('station_z_transformed', np.nan),
                    'distance_m': gt_row.get('distance_m', np.nan),
                    'origin_time_s': gt_row.get('origin_time_s', np.nan),
                    'weighted_rms': gt_row.get('weighted_rms', np.nan)
                }
                
                # Add any additional columns that might exist in ground truth
                for col in gt_row.index:
                    if col not in csv_record:
                        csv_record[col] = gt_row[col]
                
                csv_data.append(csv_record)
    
    # Convert to DataFrame and save
    if csv_data:
        df_synthetic = pd.DataFrame(csv_data)
        
        ensure_directory_exists(os.path.dirname(config.synthetic_tt_output_csv))
        df_synthetic.to_csv(config.synthetic_tt_output_csv, index=False)
        
        print(f"Synthetic travel time CSV written: {config.synthetic_tt_output_csv}")
        print(f"  Total records: {len(csv_data)}")
        print(f"  Unique events: {df_synthetic['event_id'].nunique()}")
        print(f"  Unique stations: {df_synthetic['station'].nunique()}")
        print(f"  P-wave picks: {len(df_synthetic[df_synthetic['phase'] == 'P'])}")
        print(f"  S-wave picks: {len(df_synthetic[df_synthetic['phase'] == 'S'])}")
        
        # Calculate basic statistics
        rms_error = np.sqrt(np.mean(df_synthetic['residual_s']**2))
        print(f"  RMS residual: {rms_error:.4f} s")
        
    else:
        print("Warning: No synthetic travel time data was processed")

def postprocess_latte_outputs(config: DictConfig) -> None:
    """Main post-processing function."""
    print("=== LATTE POST-PROCESSING: BINARY TO CSV CONVERSION ===")
    print(f"Configuration loaded from: {config.config_file if hasattr(config, 'config_file') else 'default'}")
    
    # Create output directories
    if config.velocity_processing.enabled:
        ensure_directory_exists(os.path.dirname(config.velocity_processing.velocity_output_csv))
    
    if config.travel_time_processing.enabled:
        ensure_directory_exists(os.path.dirname(config.travel_time_processing.synthetic_tt_output_csv))
    
    # Process velocity binaries
    if config.velocity_processing.enabled:
        print(f"\nVelocity processing: ENABLED")
        try:
            # Create temporary config for velocity processing
            velocity_config = OmegaConf.create({
                'original_velocity_csv': config.velocity_processing.original_velocity_csv,
                'vp_binary': config.velocity_processing.vp_binary,
                'vs_binary': config.velocity_processing.vs_binary,
                'velocity_output_csv': config.velocity_processing.velocity_output_csv,
                'use_chapman_inverse': config.velocity_processing.use_chapman_inverse,
                'earth_radius_km': config.velocity_processing.earth_radius_km
            })
            
            create_velocity_csv_from_binaries(velocity_config)
            
        except Exception as e:
            print(f"Error in velocity processing: {e}")
            if config.strict_mode:
                raise
    else:
        print(f"\nVelocity processing: DISABLED")
    
    # Process travel time binaries
    if config.travel_time_processing.enabled:
        print(f"\nTravel time processing: ENABLED")
        try:
            # Create temporary config for travel time processing
            tt_config = OmegaConf.create({
                'ground_truth_csv': config.travel_time_processing.ground_truth_csv,
                'synthetic_tt_dir': config.travel_time_processing.synthetic_tt_dir,
                'shot_map_file': config.travel_time_processing.shot_map_file,
                'synthetic_tt_output_csv': config.travel_time_processing.synthetic_tt_output_csv
            })
            
            create_synthetic_travel_time_csv(tt_config)
            
        except Exception as e:
            print(f"Error in travel time processing: {e}")
            if config.strict_mode:
                raise
    else:
        print(f"\nTravel time processing: DISABLED")
    
    print("\n=== POST-PROCESSING COMPLETE ===")