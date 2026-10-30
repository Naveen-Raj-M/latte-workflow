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

from utils.io_utils import read_csv_file, glob_csv, resolve_velocity_path
from utils.velocity_io import read_velocity_file

logger = logging.getLogger(__name__)

# Required columns for traveltime CSV
_TT_REQUIRED_COLS = {'residual_s', 'phase'}

# Required columns for method-comparison CSVs
_MC_REQUIRED_COLS = {'synthetic_tt', 'phase', 'event_id', 'station'}
_MC_MERGE_KEYS = ['event_id', 'station', 'phase']

# Required columns for velocity CSV
_VEL_REQUIRED_COLS = {'x', 'y', 'z', 'vp', 'vs'}

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
        traveltime_csv_pattern: Glob pattern (e.g., "synthetic_traveltimes.csv").

    Returns:
        DataFrame with at least columns [residual_s, phase], or None if not found.
    """
    candidates = glob_csv(iteration_dir, traveltime_csv_pattern)

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
    stat_mode: str = 'rms',
    use_log_residuals: bool = False,
    log_base: str = 'natural',
    log_residual_epsilon: float = 1e-10,
    use_weighted: bool = False,
) -> Dict[str, Optional[float]]:
    """
    Compute traveltime statistic(s) for one iteration.

    stat_mode='rms':          sqrt(mean(residual_s²))
    stat_mode='data_variance': mean(residual_s²)   [MSE, units s²]

    When use_log_residuals=True, the per-row metric becomes the absolute
    log-ratio of the observed and synthetic travel times:

        t = |log(max(observed_tt, ε)) − log(max(synthetic_tt, ε))|
          = |log(max(observed_tt, ε) / max(synthetic_tt, ε))|

    where ε = log_residual_epsilon.  Any travel time below ε is clamped
    to ε before the log is applied.  The log base is selected by log_base
    ('natural' → ln, 'log10' → log₁₀).

    Thresholding (min/max_residual_threshold) still applies to residual_s
    before the statistic is computed, so extreme outliers are excluded.

    Args:
        df: Traveltime residuals DataFrame with columns [residual_s, phase]
            and, when use_log_residuals=True, also [observed_tt, synthetic_tt].
        phase_mode: One of "both", "p", "s", "separate".
        min_residual_threshold: Minimum residual_s to include (seconds).
        max_residual_threshold: Maximum residual_s to include (seconds).
        stat_mode: 'rms' (default) or 'data_variance'.
        use_log_residuals: If True, compute |log(observed_tt/synthetic_tt)|
            instead of residual_s before squaring.
        log_base: 'natural' (ln) or 'log10' (log₁₀). Used only when
            use_log_residuals=True.
        log_residual_epsilon: Travel times below this value are clamped to
            this value before taking the log.  Used only when
            use_log_residuals=True.
        use_weighted: If True, use per-row values from the 'weight' column
            for weighted RMS/variance. If False (default), all weights are
            treated as 1 regardless of whether the column exists.

    Returns:
        Dict with values per curve key:
            "both"/"p"/"s" → {'combined'/'p'/'s': float_or_None}
            "separate"     → {'p': float_or_None, 's': float_or_None}
    """
    if stat_mode not in ('rms', 'data_variance'):
        raise ValueError(
            f"Invalid stat_mode: '{stat_mode}'. Valid options: 'rms', 'data_variance'"
        )
    if use_log_residuals and log_base not in ('natural', 'log10'):
        raise ValueError(
            f"Invalid log_base: '{log_base}'. Valid options: 'natural', 'log10'"
        )
    if use_log_residuals:
        for col in ('observed_tt', 'synthetic_tt'):
            if col not in df.columns:
                raise ValueError(
                    f"use_log_residuals=True requires column '{col}' in the "
                    "traveltime CSV, but it was not found. "
                    "Ensure the CSV contains both 'observed_tt' and 'synthetic_tt' columns."
                )

    # Apply residual thresholds before computing the statistic
    df_filtered = df[
        (df['residual_s'] >= min_residual_threshold)
        & (df['residual_s'] <= max_residual_threshold)
    ]

    def _weighted_mse(values: np.ndarray, weights: Optional[np.ndarray]) -> float:
        if weights is not None:
            w_sum = weights.sum()
            if w_sum > 0:
                return float(np.sum(weights * values**2) / w_sum)
        return float(np.mean(values**2))

    def _rms(subset: pd.DataFrame) -> Optional[float]:
        has_weights = use_weighted and 'weight' in subset.columns
        if use_log_residuals:
            cols = ['observed_tt', 'synthetic_tt']
            if has_weights:
                cols.append('weight')
            valid = subset[cols].dropna()
            if len(valid) == 0:
                return None
            obs = np.maximum(valid['observed_tt'].values, log_residual_epsilon)
            syn = np.maximum(valid['synthetic_tt'].values, log_residual_epsilon)
            if log_base == 'log10':
                log_res = np.abs(np.log10(obs) - np.log10(syn))
            else:
                log_res = np.abs(np.log(obs) - np.log(syn))
            w = valid['weight'].values if has_weights else None
            mse = _weighted_mse(log_res, w)
        else:
            valid = subset[['residual_s'] + (['weight'] if has_weights else [])].dropna()
            if len(valid) == 0:
                return None
            r = valid['residual_s'].values
            w = valid['weight'].values if has_weights else None
            mse = _weighted_mse(r, w)
        return mse if stat_mode == 'data_variance' else float(np.sqrt(mse))

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
    stat_mode: str = 'rms',
    use_log_residuals: bool = False,
    log_base: str = 'natural',
    log_residual_epsilon: float = 1e-10,
    use_weighted: bool = False,
) -> Optional[Dict[str, Optional[float]]]:
    """
    Compute traveltime statistic for method comparison for a single iteration.

    Loads the Method 2 CSV from the iteration directory, merges with the
    static Method 1 DataFrame on (event_id, station, phase), computes
    residuals (Method2.synthetic_tt - Method1.synthetic_tt), and returns
    the requested statistic.

    Args:
        method1_df: Static Method 1 DataFrame with columns
            [event_id, station, phase, synthetic_tt].
        iteration_dir: Path to one iteration_* subdirectory.
        method2_csv_pattern: Glob pattern for Method 2 CSV within the iteration dir.
        phase_mode: One of "both", "p", "s", "separate".
        min_residual_threshold: Minimum residual to include (seconds).
        max_residual_threshold: Maximum residual to include (seconds).
        stat_mode: 'rms' or 'data_variance'.
        use_log_residuals: If True, transform residuals via log before squaring.
        log_base: 'natural' or 'log10'. Used only when use_log_residuals=True.
        log_residual_epsilon: Small value added to |residual| before log.
        use_weighted: If True, use per-row 'weight' column for weighted computation.

    Returns:
        Dict with values per curve key (same format as compute_rms_for_iteration),
        or None if loading or merging fails.
    """
    candidates = glob_csv(iteration_dir, method2_csv_pattern)

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

    # For log-residual mode, expose Method 1 as 'observed_tt' and Method 2 as
    # 'synthetic_tt' so that compute_rms_for_iteration can compute
    # |log(observed_tt / synthetic_tt)| per matched pair.
    if use_log_residuals:
        df_for_rms = df_merged[['residual_s', 'phase']].copy()
        df_for_rms['observed_tt'] = df_merged['synthetic_tt_m1'].values
        df_for_rms['synthetic_tt'] = df_merged['synthetic_tt_m2'].values
    else:
        df_for_rms = df_merged[['residual_s', 'phase']]

    return compute_rms_for_iteration(
        df_for_rms,
        phase_mode,
        min_residual_threshold,
        max_residual_threshold,
        stat_mode=stat_mode,
        use_log_residuals=use_log_residuals,
        log_base=log_base,
        log_residual_epsilon=log_residual_epsilon,
        use_weighted=use_weighted,
    )


def load_velocity_model_for_iteration(
    iteration_dir: Path,
    velocity_dataset: str = 'velocity',
    velocity_csv_pattern: str = '*.csv',
) -> Optional[pd.DataFrame]:
    """
    Load postprocessed velocity model from an iteration directory.

    Prefers HDF5 over CSV. If multiple candidates exist, logs a WARNING
    and uses the first alphabetically.

    Args:
        iteration_dir: Path to one iteration_* subdirectory.
        velocity_dataset: HDF5 dataset name ('velocity' or 'velocity_coarse').
        velocity_csv_pattern: Glob pattern for CSV velocity model files.
            Narrow this (e.g. 'velocity*.csv') when traveltime CSVs or other
            non-model CSVs co-exist in the same iteration directory.

    Returns:
        DataFrame with columns [x, y, z, vp, vs],
        or None on failure.
    """
    hdf5_candidates = sorted(
        list(iteration_dir.rglob('*.h5')) + list(iteration_dir.rglob('*.hdf5'))
    )
    csv_candidates = sorted(iteration_dir.rglob(velocity_csv_pattern))

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


def build_parameter_entry_path(
    base_path: str,
    directory_name: str,
    iteration: int,
) -> Path:
    """
    Build and validate the directory path for a parameter sweep entry.

    Path construction: <base_path>/<directory_name>/iteration_<iteration>/

    Args:
        base_path: Root directory (inversion_base_path or inversion_analysis_dir).
        directory_name: Subdirectory name for this parameter run.
        iteration: Iteration number.

    Returns:
        Path object pointing to the constructed iteration directory.

    Raises:
        FileNotFoundError: If the constructed path does not exist.
        NotADirectoryError: If the constructed path is not a directory.
    """
    path = Path(base_path) / directory_name / f"iteration_{iteration}"
    if not path.exists():
        raise FileNotFoundError(
            f"Parameter entry directory not found: {path} "
            f"(base_path={base_path}, directory_name={directory_name}, "
            f"iteration={iteration})"
        )
    if not path.is_dir():
        raise NotADirectoryError(f"Parameter entry path is not a directory: {path}")
    return path


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
        DataFrame with columns [x, y, z, vp, vs, ...].

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
        filepath = Path(resolve_velocity_path(hdf5_path))
        if not filepath.exists():
            raise FileNotFoundError(f"Initial model HDF5 not found: {hdf5_path}")
        logger.info(f"Loading initial velocity model from HDF5: {hdf5_path}")
    else:
        filepath = Path(resolve_velocity_path(csv_path))
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
        f"X=[{df['x'].min():.2f}, {df['x'].max():.2f}] km, "
        f"Y=[{df['y'].min():.2f}, {df['y'].max():.2f}] km, "
        f"Z=[{df['z'].min():.2f}, {df['z'].max():.2f}] km"
    )

    return df
