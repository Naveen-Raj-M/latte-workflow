"""
Post-processing functions for LATTE source parameter updates.
Reads updated source coordinates (sx, sy, sz) and origin times (st0) from binary files
and merges them with travel time CSV output to create corrected event parameters.
Optionally includes gradient and search direction fields.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

_PARAM_FILE_KEY = {
    'sx': 'sx_file',
    'sy': 'sy_file',
    'sz': 'sz_file',
    'st0': 'st0_file',
}

_PARAM_COLUMN = {
    'sx': 'corrected_event_x_transformed',
    'sy': 'corrected_event_y_transformed',
    'sz': 'corrected_event_z_transformed',
    'st0': 'corrected_origin_time_s',
}


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


def read_optional_binary_file(path: str, count: int, field_name: str, dtype=np.float32) -> Optional[np.ndarray]:
    """
    Read an optional binary file (gradient or search direction).
    
    Args:
        path: Path to binary file
        count: Expected number of values
        field_name: Name of field for logging
        dtype: Data type to read
        
    Returns:
        Numpy array or None if file doesn't exist or has errors
    """
    path = Path(path)
    
    if not path.exists():
        logger.info(f"  {field_name} file not found: {path} (skipping)")
        return None
    
    logger.info(f"  Loading {field_name} from: {path}")
    
    try:
        data = np.fromfile(path, dtype=dtype, count=count)
        
        if data.size != count:
            logger.warning(f"    Size mismatch for {field_name}: expected {count}, got {data.size} (skipping)")
            return None
        
        logger.info(f"    {field_name} range: [{data.min():.6e}, {data.max():.6e}]")
        logger.info(f"    Non-zero values: {np.sum(np.abs(data) > 1e-10)}")
        
        return data
        
    except Exception as e:
        logger.warning(f"    Error reading {field_name}: {e} (skipping)")
        return None


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
    Optionally includes gradient and search direction fields.
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

    ns = cfg["n_sources"]
    logger.info(f"Number of sources: {ns}")

    # Determine which parameters to process (controls base reads, path validation, and optional fields)
    params_to_process = list(cfg.get('parameters_to_process', list(_PARAM_FILE_KEY.keys())))
    process_gradient = cfg.get('process_gradient', False)
    process_search = cfg.get('process_search', False)
    logger.info(f"Parameters to process: {params_to_process}")
    logger.info(f"  process_gradient: {process_gradient}, process_search: {process_search}")

    # Read base binary files only for parameters in parameters_to_process
    logger.info("\nReading updated source parameter binaries...")
    base_data: Dict[str, np.ndarray] = {}
    for param in params_to_process:
        if param not in _PARAM_FILE_KEY:
            logger.warning(f"  Unknown parameter '{param}' in parameters_to_process (skipping)")
            continue
        base_data[param] = read_binary_file(str(cfg[_PARAM_FILE_KEY[param]]), ns)

    # Dictionary to store optional gradient/search fields
    auxiliary_fields: Dict[str, Dict[str, Optional[np.ndarray]]] = {
        p: {'gradient': None, 'search': None} for p in base_data
    }

    # Load optional gradient and search direction files
    if base_data and (process_gradient or process_search):
        logger.info("\nLoading optional gradient and search direction files...")
        for param in base_data:
            logger.info(f"\n  Processing optional fields for {param.upper()}:")
            if process_gradient:
                grad_key = f'grad_{param}_file'
                if hasattr(cfg, grad_key):
                    auxiliary_fields[param]['gradient'] = read_optional_binary_file(
                        str(getattr(cfg, grad_key)), ns, f"{param.upper()} gradient"
                    )
            if process_search:
                srch_key = f'srch_{param}_file'
                if hasattr(cfg, srch_key):
                    auxiliary_fields[param]['search'] = read_optional_binary_file(
                        str(getattr(cfg, srch_key)), ns, f"{param.upper()} search direction"
                    )

    # Create source corrections lookup (1-based shot_id -> corrections)
    source_corrections = {}
    for shot_id in range(1, ns + 1):
        idx = shot_id - 1
        correction_dict = {_PARAM_COLUMN[p]: base_data[p][idx] for p in base_data}
        for param in base_data:
            if auxiliary_fields[param]['gradient'] is not None:
                correction_dict[f'grad_{param}'] = auxiliary_fields[param]['gradient'][idx]
            if auxiliary_fields[param]['search'] is not None:
                correction_dict[f'srch_{param}'] = auxiliary_fields[param]['search'][idx]
        source_corrections[shot_id] = correction_dict

    logger.info(f"\nCreated corrections for {len(source_corrections)} shots")

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

    # Determine all output column names (base params first, then gradient/search)
    all_output_cols = [_PARAM_COLUMN[p] for p in base_data]
    for param in base_data:
        if auxiliary_fields[param]['gradient'] is not None:
            all_output_cols.append(f'grad_{param}')
        if auxiliary_fields[param]['search'] is not None:
            all_output_cols.append(f'srch_{param}')

    # Apply corrections to each row
    logger.info("\nApplying source corrections to travel time records...")
    output_columns: Dict[str, list] = {col: [] for col in all_output_cols}
    shots_with_corrections = set()
    shots_without_corrections = set()

    for _, row in df.iterrows():
        shot_id = int(row['shot_id'])
        if shot_id in source_corrections:
            corr = source_corrections[shot_id]
            for col in output_columns:
                output_columns[col].append(corr.get(col, np.nan))
            shots_with_corrections.add(shot_id)
        else:
            for col in output_columns:
                output_columns[col].append(np.nan)
            shots_without_corrections.add(shot_id)

    for col, values in output_columns.items():
        df[col] = values

    # Report statistics
    logger.info(f"\nCorrection statistics:")
    logger.info(f"  Unique shots with corrections: {len(shots_with_corrections)}")
    if shots_without_corrections:
        logger.warning(f"  Shots without corrections: {shots_without_corrections}")

    optional_cols = [c for c in all_output_cols if c not in _PARAM_COLUMN.values()]
    if optional_cols:
        logger.info(f"  Optional fields added: {', '.join(optional_cols)}")

    # Calculate correction magnitudes (only when all spatial and temporal params were processed)
    spatial_out = ['corrected_event_x_transformed', 'corrected_event_y_transformed', 'corrected_event_z_transformed']
    spatial_orig = ['event_x_transformed', 'event_y_transformed', 'event_z_transformed']
    if (all(c in df.columns for c in spatial_out + spatial_orig) and
            'corrected_origin_time_s' in df.columns and 'origin_time_s' in df.columns):
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
    logger.info(f"  Added columns: {', '.join(all_output_cols)}")

    return df