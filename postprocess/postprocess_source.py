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

    # Get number of sources
    ns = cfg["grid_dimensions"][0]
    logger.info(f"Number of sources: {ns}")

    # Determine precision for st0 (origin time) from config
    # Access parent config through OmegaConf structure
    st0_precision = 'single'  # default
    if hasattr(cfg, '_parent') and cfg._parent is not None:
        # Try to get precision from general.precision.st0
        parent_cfg = cfg._parent
        if hasattr(parent_cfg, 'general') and hasattr(parent_cfg.general, 'precision'):
            st0_precision = parent_cfg.general.precision.get('st0', 'single')

    # Determine dtype for st0
    st0_dtype = np.float64 if st0_precision == 'double' else np.float32
    logger.info(f"Using {st0_precision} precision for st0 (origin time) - dtype: {st0_dtype.__name__}")

    # Read binary files with source corrections (REQUIRED)
    logger.info("\nReading updated source parameter binaries...")
    sx = read_binary_file(str(cfg["sx_file"]), ns)  # in meters, single precision
    sy = read_binary_file(str(cfg["sy_file"]), ns)  # in meters, single precision
    sz = read_binary_file(str(cfg["sz_file"]), ns)  # in meters, single precision
    st0 = read_binary_file(str(cfg["st0_file"]), ns, dtype=st0_dtype)  # in seconds, configurable precision

    # Determine which parameters to process for gradients/search
    params_to_process = cfg.get('parameters_to_process', [])
    logger.info(f"\nOptional fields to process: {params_to_process if params_to_process else 'None'}")
    
    # Dictionary to store optional fields
    auxiliary_fields: Dict[str, Dict[str, Optional[np.ndarray]]] = {
        'sx': {'gradient': None, 'search': None},
        'sy': {'gradient': None, 'search': None},
        'sz': {'gradient': None, 'search': None},
        'st0': {'gradient': None, 'search': None}
    }
    
    # Load optional gradient and search direction files
    if params_to_process:
        logger.info("\nLoading optional gradient and search direction files...")
        
        for param in params_to_process:
            if param not in ['sx', 'sy', 'sz', 'st0']:
                logger.warning(f"  Unknown parameter '{param}' in parameters_to_process (skipping)")
                continue

            logger.info(f"\n  Processing optional fields for {param.upper()}:")

            # Use st0 precision for st0 gradient/search, single precision for others
            param_dtype = st0_dtype if param == 'st0' else np.float32

            # Load gradient file if specified
            grad_key = f'grad_{param}_file'
            if hasattr(cfg, grad_key):
                grad_file = getattr(cfg, grad_key)
                auxiliary_fields[param]['gradient'] = read_optional_binary_file(
                    str(grad_file), ns, f"{param.upper()} gradient", dtype=param_dtype
                )

            # Load search direction file if specified
            srch_key = f'srch_{param}_file'
            if hasattr(cfg, srch_key):
                srch_file = getattr(cfg, srch_key)
                auxiliary_fields[param]['search'] = read_optional_binary_file(
                    str(srch_file), ns, f"{param.upper()} search direction", dtype=param_dtype
                )

    # Create source corrections lookup (1-based shot_id -> corrections)
    source_corrections = {}
    for shot_id in range(1, ns + 1):
        idx = shot_id - 1  # Convert to 0-based array index
        
        correction_dict = {
            'corrected_event_x_transformed': sx[idx],
            'corrected_event_y_transformed': sy[idx],
            'corrected_event_z_transformed': sz[idx],
            'corrected_origin_time_s': st0[idx]
        }
        
        # Add optional gradient fields
        for param in ['sx', 'sy', 'sz', 'st0']:
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

    # Apply corrections to each row
    logger.info("\nApplying source corrections to travel time records...")
    
    # Initialize lists for required columns
    corrected_x = []
    corrected_y = []
    corrected_z = []
    corrected_t0 = []
    
    # Initialize dictionaries for optional columns
    optional_columns = {}
    for param in ['sx', 'sy', 'sz', 'st0']:
        if auxiliary_fields[param]['gradient'] is not None:
            optional_columns[f'grad_{param}'] = []
        if auxiliary_fields[param]['search'] is not None:
            optional_columns[f'srch_{param}'] = []
    
    shots_with_corrections = set()
    shots_without_corrections = set()
    
    for _, row in df.iterrows():
        shot_id = int(row['shot_id'])
        
        if shot_id in source_corrections:
            corr = source_corrections[shot_id]
            
            # Add required fields
            corrected_x.append(corr['corrected_event_x_transformed'])
            corrected_y.append(corr['corrected_event_y_transformed'])
            corrected_z.append(corr['corrected_event_z_transformed'])
            corrected_t0.append(corr['corrected_origin_time_s'])
            
            # Add optional fields
            for col_name in optional_columns.keys():
                optional_columns[col_name].append(corr.get(col_name, np.nan))
            
            shots_with_corrections.add(shot_id)
        else:
            # Should not happen if validation passed, but handle gracefully
            corrected_x.append(np.nan)
            corrected_y.append(np.nan)
            corrected_z.append(np.nan)
            corrected_t0.append(np.nan)
            
            # Add NaN for optional fields
            for col_name in optional_columns.keys():
                optional_columns[col_name].append(np.nan)
            
            shots_without_corrections.add(shot_id)
    
    # Add corrected columns to DataFrame (required fields)
    df['corrected_event_x_transformed'] = corrected_x
    df['corrected_event_y_transformed'] = corrected_y
    df['corrected_event_z_transformed'] = corrected_z
    df['corrected_origin_time_s'] = corrected_t0
    
    # Add optional columns to DataFrame
    for col_name, values in optional_columns.items():
        df[col_name] = values
    
    # Report statistics
    logger.info(f"\nCorrection statistics:")
    logger.info(f"  Unique shots with corrections: {len(shots_with_corrections)}")
    if shots_without_corrections:
        logger.warning(f"  Shots without corrections: {shots_without_corrections}")
    
    if optional_columns:
        logger.info(f"  Optional fields added: {', '.join(optional_columns.keys())}")
    
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
    
    if optional_columns:
        logger.info(f"  Optional columns added: {', '.join(optional_columns.keys())}")
    
    return df