"""
Post-processing functions for LATTE source parameter updates.
Reads updated source coordinates (sx, sy, sz) and origin times (st0) from binary files
and merges them with travel time CSV output to create corrected event parameters.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def read_binary_file(path: str, count: int, dtype=np.float32) -> np.ndarray:
    """Read a binary file into a numpy array with validation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Binary file not found: {path}")
    
    logger.info(f"Reading binary file: {path}")
    data = np.fromfile(path, dtype=dtype, count=count)
    
    if data.size != count:
        raise ValueError(f"Expected {count} values in {path}, got {data.size}")
    
    logger.info(f"  Successfully read {data.size} values")
    logger.info(f"  Value range: [{data.min():.6f}, {data.max():.6f}]")
    
    return data


def validate_shot_ids(df: pd.DataFrame, ns: int) -> None:
    """Validate that shot_ids are within expected range."""
    unique_shots = df['shot_id'].unique()
    min_shot = unique_shots.min()
    max_shot = unique_shots.max()
    
    logger.info(f"Shot ID validation:")
    logger.info(f"  Unique shots in CSV: {len(unique_shots)}")
    logger.info(f"  Shot ID range: [{min_shot}, {max_shot}]")
    logger.info(f"  Expected range: [1, {ns}]")
    
    if min_shot < 1:
        raise ValueError(f"Shot IDs must be >= 1, found {min_shot}")
    if max_shot > ns:
        raise ValueError(f"Shot IDs must be <= {ns}, found {max_shot}")
    
    # Check for any shots without corrections
    missing_shots = set(range(1, ns + 1)) - set(unique_shots)
    if missing_shots:
        logger.warning(f"  {len(missing_shots)} shots in binary have no corresponding travel time records")


def process_source_updates(cfg: DictConfig) -> pd.DataFrame:
    """
    Process updated source parameters and merge with travel time CSV.
    
    Reads binary files containing updated source coordinates (sx, sy, sz) and 
    origin times (st0), then merges them with the travel time CSV output.
    Uses 1-based indexing: shot_id N corresponds to binary array index N-1.
    
    Args:
        cfg: Configuration dictionary with paths and parameters
        
    Returns:
        DataFrame with original travel time data plus corrected event parameters
    """
    if not cfg.get("enabled", False):
        logger.info("Source update processing is disabled in config.")
        return pd.DataFrame()

    logger.info("=== PROCESSING SOURCE PARAMETER UPDATES ===")
    
    # Get number of sources
    ns = cfg["grid_dimensions"][0]
    logger.info(f"Number of sources: {ns}")

    # Read binary files with source corrections
    logger.info("\nReading updated source parameter binaries...")
    sx = read_binary_file(str(cfg["sx_file"]), ns)  # in meters
    sy = read_binary_file(str(cfg["sy_file"]), ns)  # in meters
    sz = read_binary_file(str(cfg["sz_file"]), ns)  # in meters
    st0 = read_binary_file(str(cfg["st0_file"]), ns)  # in seconds

    # Create source corrections lookup (1-based shot_id -> corrections)
    source_corrections = {}
    for shot_id in range(1, ns + 1):
        idx = shot_id - 1  # Convert to 0-based array index
        source_corrections[shot_id] = {
            'corrected_event_x_transformed': sx[idx],
            'corrected_event_y_transformed': sy[idx],
            'corrected_event_z_transformed': sz[idx],
            'corrected_origin_time_s': st0[idx]
        }
    
    logger.info(f"Created corrections for {len(source_corrections)} shots")

    # Load travel time CSV
    tt_csv_path = Path(str(cfg["travel_time_csv"]))
    if not tt_csv_path.exists():
        raise FileNotFoundError(f"Travel time CSV not found: {tt_csv_path}")
    
    logger.info(f"\nLoading travel time CSV: {tt_csv_path}")
    df = pd.read_csv(tt_csv_path)
    logger.info(f"  Loaded {len(df)} travel time records")
    logger.info(f"  Columns: {', '.join(df.columns.tolist())}")
    
    # Validate shot IDs
    validate_shot_ids(df, ns)
    
    # Verify required columns exist
    required_cols = ['shot_id', 'event_id', 'station', 'phase']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in travel time CSV: {missing_cols}")

    # Apply corrections to each row
    logger.info("\nApplying source corrections to travel time records...")
    
    corrected_x = []
    corrected_y = []
    corrected_z = []
    corrected_t0 = []
    
    shots_with_corrections = set()
    shots_without_corrections = set()
    
    for _, row in df.iterrows():
        shot_id = int(row['shot_id'])
        
        if shot_id in source_corrections:
            corr = source_corrections[shot_id]
            corrected_x.append(corr['corrected_event_x_transformed'])
            corrected_y.append(corr['corrected_event_y_transformed'])
            corrected_z.append(corr['corrected_event_z_transformed'])
            corrected_t0.append(corr['corrected_origin_time_s'])
            shots_with_corrections.add(shot_id)
        else:
            # Should not happen if validation passed, but handle gracefully
            corrected_x.append(np.nan)
            corrected_y.append(np.nan)
            corrected_z.append(np.nan)
            corrected_t0.append(np.nan)
            shots_without_corrections.add(shot_id)
    
    # Add corrected columns to DataFrame
    df['corrected_event_x_transformed'] = corrected_x
    df['corrected_event_y_transformed'] = corrected_y
    df['corrected_event_z_transformed'] = corrected_z
    df['corrected_origin_time_s'] = corrected_t0
    
    # Report statistics
    logger.info(f"\nCorrection statistics:")
    logger.info(f"  Unique shots with corrections: {len(shots_with_corrections)}")
    if shots_without_corrections:
        logger.warning(f"  Shots without corrections: {shots_without_corrections}")
    
    # Calculate correction magnitudes
    if 'event_x_transformed' in df.columns:
        dx = df['corrected_event_x_transformed'] - df['event_x_transformed']
        dy = df['corrected_event_y_transformed'] - df['event_y_transformed']
        dz = df['corrected_event_z_transformed'] - df['event_z_transformed']
        dt = df['corrected_origin_time_s'] - df['origin_time_s']
        
        spatial_change = np.sqrt(dx**2 + dy**2 + dz**2)
        
        logger.info(f"\nCorrection magnitudes:")
        logger.info(f"  X shift: mean={dx.mean():.3f} km, std={dx.std():.3f} km, range=[{dx.min():.3f}, {dx.max():.3f}]")
        logger.info(f"  Y shift: mean={dy.mean():.3f} km, std={dy.std():.3f} km, range=[{dy.min():.3f}, {dy.max():.3f}]")
        logger.info(f"  Z shift: mean={dz.mean():.3f} km, std={dz.std():.3f} km, range=[{dz.min():.3f}, {dz.max():.3f}]")
        logger.info(f"  3D spatial shift: mean={spatial_change.mean():.3f} km, max={spatial_change.max():.3f} km")
        logger.info(f"  Origin time shift: mean={dt.mean():.6f} s, std={dt.std():.6f} s, range=[{dt.min():.6f}, {dt.max():.6f}]")
    
    # Save output CSV
    out_path = Path(str(cfg["output_csv"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    logger.info(f"\n=== OUTPUT SUMMARY ===")
    logger.info(f"Source-corrected travel time CSV written: {out_path}")
    logger.info(f"  Total records: {len(df)}")
    logger.info(f"  Added columns: corrected_event_x_transformed, corrected_event_y_transformed, "
                f"corrected_event_z_transformed, corrected_origin_time_s")
    
    return df