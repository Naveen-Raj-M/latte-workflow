"""
Data loading and preparation for tradeoff analysis.

Handles: discovering iteration directories, loading data_misfit.txt,
loading per-iteration traveltime residual CSVs, and loading velocity models.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.io_utils import read_csv_file
from utils.velocity_io import read_velocity_file

logger = logging.getLogger(__name__)

# Required columns for traveltime CSV
_TT_REQUIRED_COLS = {'residual_s', 'phase'}

# Required columns for method-comparison CSVs
_MC_REQUIRED_COLS = {'synthetic_tt', 'phase', 'event_id', 'station'}
_MC_MERGE_KEYS = ['event_id', 'station', 'phase']

# Required columns for velocity CSV
_VEL_REQUIRED_COLS = {'X_grid_km', 'Y_grid_km', 'Z_grid_km', 'vp', 'vs'}

# Regex to extract iteration index from directory names
_ITER_DIR_PATTERN = re.compile(r'^iteration_(\d+)$')


def discover_iteration_dirs(base_path: str) -> List[Tuple[int, Path]]:
    """
    Discover and sort iteration_* subdirectories under base_path.

    Scans base_path for directories whose names match ``iteration_<N>``
    where N is a non-negative integer. Returns them sorted ascending by N.

    Args:
        base_path: Filesystem path to the parent directory.

    Returns:
        List of (iteration_index, path) sorted by index ascending.
        Empty list if no matching directories found.

    Raises:
        FileNotFoundError: If base_path does not exist.
        NotADirectoryError: If base_path is not a directory.
    """
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"Directory not found: {base_path}")
    if not base.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {base_path}")

    results = []
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        match = _ITER_DIR_PATTERN.match(entry.name)
        if match:
            idx = int(match.group(1))
            results.append((idx, entry))

    results.sort(key=lambda t: t[0])

    if not results:
        logger.warning(f"No iteration_* directories found under: {base_path}")
    else:
        logger.info(
            f"Detected {len(results)} iteration directories in {base_path}: "
            f"iterations {[i for i, _ in results]}"
        )

    return results


def load_data_misfit(
    filepath: str,
    use_normalized: bool = False,
) -> pd.DataFrame:
    """
    Load data_misfit.txt into a DataFrame.

    The file has 3 whitespace-separated columns (no header):
        col 0: iteration (int, 0-indexed)
        col 1: data_misfit (float64, absolute)
        col 2: normalized_data_misfit (float64, normalized to iter 0)

    Args:
        filepath: Path to data_misfit.txt.
        use_normalized: If True use column 2 (normalized); else column 1 (absolute).

    Returns:
        DataFrame with columns [iteration, y_value].

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If the file cannot be parsed or has < 3 columns.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"data_misfit file not found: {filepath}")

    try:
        df = pd.read_csv(
            path,
            sep=r'\s+',
            header=None,
            names=['iteration', 'data_misfit', 'normalized_data_misfit'],
            comment='#',
        )
    except Exception as e:
        raise ValueError(f"Failed to parse data_misfit file {filepath}: {e}") from e

    if df.shape[1] < 3:
        raise ValueError(
            f"data_misfit file must have at least 3 columns, got {df.shape[1]}: {filepath}"
        )

    df['iteration'] = df['iteration'].astype(int)

    if use_normalized:
        df['y_value'] = df['normalized_data_misfit'].astype(float)
        y_label = "normalized data misfit"
    else:
        df['y_value'] = df['data_misfit'].astype(float)
        y_label = "absolute data misfit"

    df = df[['iteration', 'y_value']].copy()

    iter_range = (df['iteration'].min(), df['iteration'].max())
    logger.info(
        f"Loaded data_misfit: {len(df)} rows, iterations {iter_range[0]}-{iter_range[1]}, "
        f"using {y_label}"
    )

    return df


def load_traveltime_residuals_for_iteration(
    iteration_dir: Path,
    traveltime_csv_pattern: str,
) -> Optional[pd.DataFrame]:
    """
    Find and load the traveltime residual CSV for one iteration directory.

    Args:
        iteration_dir: Path object pointing to a single iteration_* dir.
        traveltime_csv_pattern: Glob pattern (e.g., "travel_time_residuals*.csv").

    Returns:
        DataFrame with at least columns [residual_s, phase], or None if not found.
    """
    candidates = sorted(iteration_dir.glob(traveltime_csv_pattern))

    if not candidates:
        logger.warning(
            f"No traveltime CSV matching '{traveltime_csv_pattern}' "
            f"found in {iteration_dir}"
        )
        return None

    if len(candidates) > 1:
        logger.warning(
            f"Multiple traveltime CSVs found in {iteration_dir}, "
            f"using: {candidates[0].name}. Others: {[c.name for c in candidates[1:]]}"
        )

    chosen = candidates[0]
    try:
        df = read_csv_file(str(chosen))
    except Exception as e:
        logger.error(f"Failed to load traveltime CSV {chosen}: {e}")
        return None

    missing = _TT_REQUIRED_COLS - set(df.columns)
    if missing:
        logger.error(
            f"Traveltime CSV {chosen} missing required columns: {missing}"
        )
        return None

    logger.debug(
        f"Loaded traveltime CSV: {chosen.name} ({len(df)} rows)"
    )
    return df


def compute_rms_for_iteration(
    df: pd.DataFrame,
    phase_mode: str,
    min_residual_threshold: float = -50.0,
    max_residual_threshold: float = 50.0,
) -> Dict[str, Optional[float]]:
    """
    Compute RMS traveltime residual(s) for one iteration.

    Formula: rms = sqrt(mean(residual_s**2))

    Args:
        df: Traveltime residuals DataFrame with columns [residual_s, phase].
        phase_mode: One of "both", "p", "s", "separate".
        min_residual_threshold: Minimum residual_s to include (seconds).
        max_residual_threshold: Maximum residual_s to include (seconds).

    Returns:
        Dict with RMS values per curve key:
            "both"/"p"/"s" → {'combined'/'p'/'s': float_or_None}
            "separate"     → {'p': float_or_None, 's': float_or_None}
    """
    # Apply thresholds
    df_filtered = df[
        (df['residual_s'] >= min_residual_threshold)
        & (df['residual_s'] <= max_residual_threshold)
    ]

    def _rms(subset: pd.DataFrame) -> Optional[float]:
        residuals = subset['residual_s'].dropna().values
        if len(residuals) == 0:
            return None
        return float(np.sqrt(np.mean(residuals ** 2)))

    if phase_mode == 'both':
        rms_val = _rms(df_filtered)
        if rms_val is None:
            logger.warning("No valid residuals found for RMS computation (all phases)")
        return {'combined': rms_val}

    elif phase_mode == 'p':
        subset = df_filtered[df_filtered['phase'].str.upper() == 'P']
        rms_val = _rms(subset)
        if rms_val is None:
            logger.warning("No valid P-wave residuals found for RMS computation")
        return {'p': rms_val}

    elif phase_mode == 's':
        subset = df_filtered[df_filtered['phase'].str.upper() == 'S']
        rms_val = _rms(subset)
        if rms_val is None:
            logger.warning("No valid S-wave residuals found for RMS computation")
        return {'s': rms_val}

    elif phase_mode == 'separate':
        p_subset = df_filtered[df_filtered['phase'].str.upper() == 'P']
        s_subset = df_filtered[df_filtered['phase'].str.upper() == 'S']
        rms_p = _rms(p_subset)
        rms_s = _rms(s_subset)
        if rms_p is None:
            logger.warning("No valid P-wave residuals found for RMS computation")
        if rms_s is None:
            logger.warning("No valid S-wave residuals found for RMS computation")
        return {'p': rms_p, 's': rms_s}

    else:
        raise ValueError(
            f"Invalid phase_mode: '{phase_mode}'. "
            f"Valid options: 'both', 'p', 's', 'separate'"
        )


def load_static_method_csv(filepath: str) -> pd.DataFrame:
    """
    Load and validate a static Method 1 CSV for method-comparison RMS mode.

    Required columns: event_id, station, phase, synthetic_tt.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Validated DataFrame.

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If the file cannot be loaded or required columns are missing.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Method 1 CSV not found: {filepath}")

    try:
        df = read_csv_file(str(path))
    except Exception as e:
        raise ValueError(f"Failed to load Method 1 CSV {filepath}: {e}") from e

    missing = _MC_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Method 1 CSV missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    logger.info(f"Loaded Method 1 CSV (static): {path.name} ({len(df)} rows)")
    return df


def compute_method_comparison_rms_for_iteration(
    method1_df: pd.DataFrame,
    iteration_dir: Path,
    method2_csv_pattern: str,
    phase_mode: str,
    min_residual_threshold: float = -50.0,
    max_residual_threshold: float = 50.0,
) -> Optional[Dict[str, Optional[float]]]:
    """
    Compute RMS for method comparison for a single iteration.

    Loads the Method 2 CSV from the iteration directory, merges with the
    static Method 1 DataFrame on (event_id, station, phase), computes
    residuals (Method2.synthetic_tt - Method1.synthetic_tt), and returns RMS.

    Args:
        method1_df: Static Method 1 DataFrame with columns
            [event_id, station, phase, synthetic_tt].
        iteration_dir: Path to one iteration_* subdirectory.
        method2_csv_pattern: Glob pattern for Method 2 CSV within the iteration dir.
        phase_mode: One of "both", "p", "s", "separate".
        min_residual_threshold: Minimum residual to include (seconds).
        max_residual_threshold: Maximum residual to include (seconds).

    Returns:
        Dict with RMS values per curve key (same format as compute_rms_for_iteration),
        or None if loading or merging fails.
    """
    candidates = sorted(iteration_dir.glob(method2_csv_pattern))

    if not candidates:
        logger.warning(
            f"No Method 2 CSV matching '{method2_csv_pattern}' "
            f"found in {iteration_dir}"
        )
        return None

    if len(candidates) > 1:
        logger.warning(
            f"Multiple Method 2 CSVs found in {iteration_dir}, "
            f"using: {candidates[0].name}. Others: {[c.name for c in candidates[1:]]}"
        )

    chosen = candidates[0]
    try:
        df2 = read_csv_file(str(chosen))
    except Exception as e:
        logger.error(f"Failed to load Method 2 CSV {chosen}: {e}")
        return None

    missing2 = _MC_REQUIRED_COLS - set(df2.columns)
    if missing2:
        logger.error(
            f"Method 2 CSV {chosen.name} missing required columns: {missing2}"
        )
        return None

    df_merged = pd.merge(
        method1_df[_MC_MERGE_KEYS + ['synthetic_tt']],
        df2[_MC_MERGE_KEYS + ['synthetic_tt']],
        on=_MC_MERGE_KEYS,
        suffixes=('_m1', '_m2'),
        how='inner',
    )

    if df_merged.empty:
        logger.warning(
            f"No matching records between Method 1 and Method 2 in {iteration_dir}"
        )
        return None

    df_merged['residual_s'] = (
        df_merged['synthetic_tt_m2'] - df_merged['synthetic_tt_m1']
    )

    logger.debug(
        f"Method comparison {iteration_dir.name}: {len(df_merged)} matched pairs, "
        f"residual mean={df_merged['residual_s'].mean():.4f} s, "
        f"RMS={np.sqrt(np.mean(df_merged['residual_s'] ** 2)):.4f} s"
    )

    return compute_rms_for_iteration(
        df_merged[['residual_s', 'phase']],
        phase_mode,
        min_residual_threshold,
        max_residual_threshold,
    )


def load_velocity_model_for_iteration(
    iteration_dir: Path,
    velocity_dataset: str = 'velocity',
) -> Optional[pd.DataFrame]:
    """
    Load postprocessed velocity model from an iteration directory.

    Prefers HDF5 over CSV. If multiple candidates exist, logs a WARNING
    and uses the first alphabetically.

    Args:
        iteration_dir: Path to one iteration_* subdirectory.
        velocity_dataset: HDF5 dataset name ('velocity' or 'velocity_coarse').

    Returns:
        DataFrame with columns [X_grid_km, Y_grid_km, Z_grid_km, vp, vs],
        or None on failure.
    """
    hdf5_candidates = sorted(
        list(iteration_dir.rglob('*.h5')) + list(iteration_dir.rglob('*.hdf5'))
    )
    csv_candidates = sorted(iteration_dir.rglob('*.csv'))

    if hdf5_candidates:
        if len(hdf5_candidates) > 1:
            logger.warning(
                f"Multiple HDF5 files found under {iteration_dir}, "
                f"using: {hdf5_candidates[0].relative_to(iteration_dir)}"
            )
        chosen = hdf5_candidates[0]
    elif csv_candidates:
        if len(csv_candidates) > 1:
            logger.warning(
                f"Multiple CSV files found under {iteration_dir}, "
                f"using: {csv_candidates[0].relative_to(iteration_dir)}"
            )
        chosen = csv_candidates[0]
    else:
        logger.error(
            f"No velocity model file (HDF5 or CSV) found in {iteration_dir} "
            "or any of its subdirectories"
        )
        return None

    try:
        df = read_velocity_file(str(chosen), dataset=velocity_dataset, show_progress=False)
    except Exception as e:
        logger.error(f"Failed to load velocity model {chosen}: {e}")
        return None

    missing = _VEL_REQUIRED_COLS - set(df.columns)
    if missing:
        logger.error(
            f"Velocity model {chosen} missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )
        return None

    logger.debug(
        f"Loaded velocity model: {chosen.relative_to(iteration_dir)} "
        f"({len(df)} grid points)"
    )
    return df


def load_initial_velocity_model(
    csv_path: Optional[str],
    hdf5_path: Optional[str],
    velocity_dataset: str = 'velocity',
) -> pd.DataFrame:
    """
    Load the initial (reference) velocity model.

    HDF5 takes precedence over CSV if both are provided.

    Args:
        csv_path: Path to CSV file (or None).
        hdf5_path: Path to HDF5 file (or None).
        velocity_dataset: Dataset name for HDF5.

    Returns:
        DataFrame with columns [X_grid_km, Y_grid_km, Z_grid_km, vp, vs, ...].

    Raises:
        ValueError: If both paths are None.
        FileNotFoundError: If the chosen file does not exist.
    """
    if not csv_path and not hdf5_path:
        raise ValueError(
            "Initial velocity model not specified. "
            "Provide initial_model_csv or initial_model_hdf5."
        )

    if hdf5_path:
        filepath = Path(hdf5_path)
        if not filepath.exists():
            raise FileNotFoundError(f"Initial model HDF5 not found: {hdf5_path}")
        logger.info(f"Loading initial velocity model from HDF5: {hdf5_path}")
    else:
        filepath = Path(csv_path)
        if not filepath.exists():
            raise FileNotFoundError(f"Initial model CSV not found: {csv_path}")
        logger.info(f"Loading initial velocity model from CSV: {csv_path}")

    df = read_velocity_file(str(filepath), dataset=velocity_dataset, show_progress=False)

    missing = _VEL_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Initial velocity model missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    logger.info(
        f"Initial model loaded: {len(df)} grid points, "
        f"X=[{df['X_grid_km'].min():.2f}, {df['X_grid_km'].max():.2f}] km, "
        f"Y=[{df['Y_grid_km'].min():.2f}, {df['Y_grid_km'].max():.2f}] km, "
        f"Z=[{df['Z_grid_km'].min():.2f}, {df['Z_grid_km'].max():.2f}] km"
    )

    return df
