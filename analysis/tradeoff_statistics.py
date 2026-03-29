"""
Statistical computations for tradeoff analysis.

Handles: spatial domain validation, model variance computation per iteration,
elbow/optimal-point detection, and summary statistics assembly.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Type alias for an optional float bound pair
BoundPair = Tuple[Optional[float], Optional[float]]


def validate_spatial_bounds(
    df_initial: pd.DataFrame,
    x_bounds: BoundPair,
    y_bounds: BoundPair,
    z_bounds: Optional[BoundPair],
) -> None:
    """
    Validate user-specified spatial bounds against the actual model domain.

    For each axis, checks for overlap with the model extent. Partial overlap
    raises a WARNING; no overlap at all raises a ValueError.

    Args:
        df_initial: Initial velocity model DataFrame with X/Y/Z_grid_km columns.
        x_bounds: (x_min, x_max) – either element may be None (no bound).
        y_bounds: (y_min, y_max) – same convention.
        z_bounds: (z_min, z_max) or None (no Z filter).

    Raises:
        ValueError: If any bound has zero overlap with the model domain.
    """
    axes = [
        ('X', 'X_grid_km', x_bounds),
        ('Y', 'Y_grid_km', y_bounds),
    ]
    if z_bounds is not None:
        axes.append(('Z', 'Z_grid_km', z_bounds))

    for axis_name, col, bounds in axes:
        lo, hi = bounds
        if lo is None and hi is None:
            continue  # No bound on this axis

        model_min = float(df_initial[col].min())
        model_max = float(df_initial[col].max())

        effective_lo = lo if lo is not None else model_min
        effective_hi = hi if hi is not None else model_max

        # Check for zero overlap
        if effective_lo > model_max or effective_hi < model_min:
            raise ValueError(
                f"{axis_name}_bounds [{lo}, {hi}] has no overlap with the model "
                f"domain [{model_min:.3f}, {model_max:.3f}] km. "
                f"No grid points would be selected."
            )

        # Warn if bounds extend beyond model domain
        if lo is not None and lo < model_min:
            logger.warning(
                f"{axis_name}_bounds lower limit {lo:.3f} km is below the model "
                f"minimum {model_min:.3f} km – it will be clipped to the model domain."
            )
        if hi is not None and hi > model_max:
            logger.warning(
                f"{axis_name}_bounds upper limit {hi:.3f} km exceeds the model "
                f"maximum {model_max:.3f} km – it will be clipped to the model domain."
            )

        logger.info(
            f"{axis_name}_bounds [{lo}, {hi}] km valid against model "
            f"domain [{model_min:.3f}, {model_max:.3f}] km"
        )


def build_spatial_mask(
    df: pd.DataFrame,
    x_bounds: BoundPair,
    y_bounds: BoundPair,
    z_bounds: Optional[BoundPair],
) -> np.ndarray:
    """
    Build a boolean mask selecting grid points within all spatial bounds.

    A None bound value means "no restriction" on that side.

    Args:
        df: Velocity model DataFrame with X_grid_km, Y_grid_km, Z_grid_km.
        x_bounds: (x_min, x_max).
        y_bounds: (y_min, y_max).
        z_bounds: (z_min, z_max) or None.

    Returns:
        Boolean numpy array of length len(df), True where point is inside bounds.
    """
    mask = np.ones(len(df), dtype=bool)

    def _apply_bound(col: str, lo: Optional[float], hi: Optional[float]) -> None:
        nonlocal mask
        if lo is not None:
            mask &= (df[col].values >= lo)
        if hi is not None:
            mask &= (df[col].values <= hi)

    _apply_bound('X_grid_km', *x_bounds)
    _apply_bound('Y_grid_km', *y_bounds)
    if z_bounds is not None:
        _apply_bound('Z_grid_km', *z_bounds)

    n_selected = int(mask.sum())
    logger.debug(f"Spatial mask selects {n_selected} / {len(df)} grid points")
    return mask


def build_depth_layer_mask(
    df: pd.DataFrame,
    depth_km: float,
    depth_tolerance_km: float,
    x_bounds: BoundPair,
    y_bounds: BoundPair,
) -> np.ndarray:
    """
    Build mask for points within tolerance of a specific depth layer.

    Args:
        df: Velocity model DataFrame.
        depth_km: Target depth in km.
        depth_tolerance_km: Half-window tolerance around target depth.
        x_bounds: (x_min, x_max) optional horizontal bounds.
        y_bounds: (y_min, y_max) optional horizontal bounds.

    Returns:
        Boolean numpy array of length len(df).
    """
    depth_mask = np.abs(df['Z_grid_km'].values - depth_km) <= depth_tolerance_km

    # Apply horizontal bounds
    xy_mask = np.ones(len(df), dtype=bool)
    x_lo, x_hi = x_bounds
    y_lo, y_hi = y_bounds
    if x_lo is not None:
        xy_mask &= (df['X_grid_km'].values >= x_lo)
    if x_hi is not None:
        xy_mask &= (df['X_grid_km'].values <= x_hi)
    if y_lo is not None:
        xy_mask &= (df['Y_grid_km'].values >= y_lo)
    if y_hi is not None:
        xy_mask &= (df['Y_grid_km'].values <= y_hi)

    combined = depth_mask & xy_mask
    n_selected = int(combined.sum())

    if n_selected == 0:
        available_z = np.sort(df['Z_grid_km'].unique())
        nearest = available_z[np.argmin(np.abs(available_z - depth_km))]
        logger.warning(
            f"Depth layer mask at Z={depth_km:.2f} km (±{depth_tolerance_km:.2f} km) "
            f"selected 0 points. Nearest available Z: {nearest:.3f} km. "
            f"Available Z values: {available_z[:10].tolist()}{'...' if len(available_z) > 10 else ''}"
        )
    else:
        logger.debug(
            f"Depth layer mask at Z={depth_km:.2f} km: {n_selected} grid points selected"
        )

    return combined


def compute_model_variance_for_mask(
    df_initial: pd.DataFrame,
    df_inverted: pd.DataFrame,
    mask: np.ndarray,
    use_percentage_change: bool = False,
    aggregation: str = 'mean',
) -> Dict[str, float]:
    """
    Compute model variance for vp and vs at selected grid points.

    Two value formulas:

    Absolute (use_percentage_change=False):
        values = |v_initial - v_inverted|   [km/s]

    Percentage (use_percentage_change=True):
        values = |v_initial - v_inverted| / |v_initial| * 100   [%]

        Grid points where |v_initial| < 1e-6 km/s are excluded from the
        percentage computation. A WARNING is logged if any are found.

    Two aggregation modes (aggregation):
        "mean"  variance = mean(values)
        "max"   variance = max(values)

    Args:
        df_initial: Initial velocity model DataFrame.
        df_inverted: Inverted velocity model DataFrame for one iteration.
        mask: Boolean selection mask for grid points to include.
        use_percentage_change: If True, compute percentage change.
        aggregation: "mean" or "max".

    Returns:
        Dict with keys: 'vp', 'vs', 'max'.
        Units are km/s when use_percentage_change=False, % otherwise.

    Raises:
        ValueError: If mask selects zero points or aggregation is invalid.
    """
    if aggregation not in ('mean', 'max'):
        raise ValueError(
            f"Invalid aggregation: '{aggregation}'. Valid options: 'mean', 'max'"
        )
    n_selected = int(mask.sum())
    if n_selected == 0:
        raise ValueError(
            "Model variance mask selects 0 grid points. "
            "Check your bounds configuration."
        )

    vp_init = df_initial['vp'].values[mask]
    vs_init = df_initial['vs'].values[mask]
    vp_inv = df_inverted['vp'].values[mask]
    vs_inv = df_inverted['vs'].values[mask]

    vp_abs_diff = np.abs(vp_init - vp_inv)
    vs_abs_diff = np.abs(vs_init - vs_inv)

    if use_percentage_change:
        # Exclude near-zero initial velocities to avoid division by ~0
        vp_valid = np.abs(vp_init) >= 1e-6
        vs_valid = np.abs(vs_init) >= 1e-6

        n_vp_excluded = int((~vp_valid).sum())
        n_vs_excluded = int((~vs_valid).sum())

        if n_vp_excluded:
            logger.warning(
                f"{n_vp_excluded} grid points have |vp_initial| < 1e-6 km/s "
                "and are excluded from percentage variance computation."
            )
        if n_vs_excluded:
            logger.warning(
                f"{n_vs_excluded} grid points have |vs_initial| < 1e-6 km/s "
                "and are excluded from percentage variance computation."
            )

        if not vp_valid.any():
            raise ValueError(
                "All selected grid points have near-zero initial Vp – "
                "cannot compute percentage model variance."
            )
        if not vs_valid.any():
            raise ValueError(
                "All selected grid points have near-zero initial Vs – "
                "cannot compute percentage model variance."
            )

        agg = np.max if aggregation == 'max' else np.mean
        variance_vp = float(
            agg(vp_abs_diff[vp_valid] / np.abs(vp_init[vp_valid])) * 100.0
        )
        variance_vs = float(
            agg(vs_abs_diff[vs_valid] / np.abs(vs_init[vs_valid])) * 100.0
        )
    else:
        agg = np.max if aggregation == 'max' else np.mean
        variance_vp = float(agg(vp_abs_diff))
        variance_vs = float(agg(vs_abs_diff))

    variance_max = max(variance_vp, variance_vs)
    return {'vp': variance_vp, 'vs': variance_vs, 'max': variance_max}


def assemble_tradeoff_table(
    iteration_indices: List[int],
    y_values: Dict[int, Any],
    vp_variances: Dict[int, float],
    vs_variances: Dict[int, float],
    phase_mode: str,
) -> pd.DataFrame:
    """
    Assemble per-iteration statistics into a single tradeoff DataFrame.

    For all modes except phase_mode=="separate":
        Columns: [iteration, y_value, model_variance_vp, model_variance_vs,
                  model_variance_max]

    For phase_mode=="separate":
        Columns: [iteration, rms_p, rms_s, model_variance_vp, model_variance_vs,
                  model_variance_max]

    Args:
        iteration_indices: Ordered list of iteration integers to include.
        y_values: Mapping from iteration → y value(s).
                  For non-separate: {iter: float}
                  For separate:     {iter: {'p': float, 's': float}}
        vp_variances: Mapping from iteration → vp variance.
        vs_variances: Mapping from iteration → vs variance.
        phase_mode: Determines column naming for separate phase mode.

    Returns:
        DataFrame sorted ascending by iteration. Rows with missing data excluded.
    """
    rows = []
    skipped = []

    for idx in iteration_indices:
        vp_var = vp_variances.get(idx)
        vs_var = vs_variances.get(idx)
        y_val = y_values.get(idx)

        if vp_var is None or vs_var is None:
            skipped.append(idx)
            continue

        if phase_mode == 'separate':
            if not isinstance(y_val, dict) or y_val.get('p') is None or y_val.get('s') is None:
                skipped.append(idx)
                continue
            rows.append({
                'iteration': idx,
                'rms_p': y_val['p'],
                'rms_s': y_val['s'],
                'model_variance_vp': vp_var,
                'model_variance_vs': vs_var,
                'model_variance_max': max(vp_var, vs_var),
            })
        else:
            # For non-separate modes, y_val may be a dict with one key or a scalar
            if isinstance(y_val, dict):
                key = next(iter(y_val))  # 'combined', 'p', or 's'
                scalar_y = y_val.get(key)
            else:
                scalar_y = y_val

            if scalar_y is None:
                skipped.append(idx)
                continue

            rows.append({
                'iteration': idx,
                'y_value': scalar_y,
                'model_variance_vp': vp_var,
                'model_variance_vs': vs_var,
                'model_variance_max': max(vp_var, vs_var),
            })

    if skipped:
        logger.warning(
            f"Skipped {len(skipped)} iterations due to missing data: {skipped}"
        )

    if not rows:
        logger.warning("Tradeoff table is empty – no complete iteration data found")
        # Return empty DataFrame with correct columns
        if phase_mode == 'separate':
            cols = ['iteration', 'rms_p', 'rms_s', 'model_variance_vp',
                    'model_variance_vs', 'model_variance_max']
        else:
            cols = ['iteration', 'y_value', 'model_variance_vp',
                    'model_variance_vs', 'model_variance_max']
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows).sort_values('iteration').reset_index(drop=True)
    logger.info(
        f"Tradeoff table assembled: {len(df)} iterations "
        f"[{df['iteration'].min()}–{df['iteration'].max()}]"
    )
    return df


def assemble_parameter_tradeoff_table(
    processed_entries: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Assemble processed parameter sweep entries into a tradeoff DataFrame.

    Args:
        processed_entries: List of dicts with keys:
            parameter_value, directory_name, iteration,
            model_variance_vp, model_variance_vs, model_variance_max, rms.

    Returns:
        DataFrame sorted by parameter_value. Empty DataFrame if no valid entries.
    """
    _cols = [
        'parameter_value', 'directory_name', 'iteration',
        'model_variance_vp', 'model_variance_vs', 'model_variance_max', 'rms',
    ]
    if not processed_entries:
        logger.warning("No parameter entries to assemble into tradeoff table")
        return pd.DataFrame(columns=_cols)

    rows = []
    skipped = []
    for e in processed_entries:
        if e.get('model_variance_vp') is None or e.get('model_variance_vs') is None:
            skipped.append(e.get('parameter_value'))
            continue
        rows.append({
            'parameter_value':    e['parameter_value'],
            'directory_name':     e['directory_name'],
            'iteration':          e['iteration'],
            'model_variance_vp':  e['model_variance_vp'],
            'model_variance_vs':  e['model_variance_vs'],
            'model_variance_max': e['model_variance_max'],
            'rms':                e.get('rms'),
        })

    if skipped:
        logger.warning(
            f"Skipped {len(skipped)} parameter entries due to missing model variance: "
            f"{skipped}"
        )

    if not rows:
        logger.warning("Parameter tradeoff table is empty – no complete entries found")
        return pd.DataFrame(columns=_cols)

    df = pd.DataFrame(rows).sort_values('parameter_value').reset_index(drop=True)
    logger.info(
        f"Parameter tradeoff table: {len(df)} entries, "
        f"parameter range [{df['parameter_value'].min()}, {df['parameter_value'].max()}]"
    )
    return df


def write_parameter_report(
    parameter_name: str,
    p_table: Optional[pd.DataFrame],
    s_table: Optional[pd.DataFrame],
    report_path: "Path",
) -> None:
    """
    Write a plain text report mapping parameter values to model variance.

    Args:
        parameter_name: Name of the swept parameter (e.g., "step_max").
        p_table: P-wave parameter tradeoff DataFrame (or None/empty).
        s_table: S-wave parameter tradeoff DataFrame (or None/empty).
        report_path: Full path for the output .txt file.
    """
    from pathlib import Path as _Path
    from datetime import datetime

    report_path = _Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    col_w = max(len(parameter_name), 14)
    sep = "=" * 72

    lines = [
        "Parameter Tradeoff Report",
        sep,
        f"Parameter : {parameter_name}",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    def _write_section(df: pd.DataFrame, phase_label: str, variance_col: str) -> None:
        lines.append(f"{phase_label} RESULTS")
        lines.append("-" * 72)
        hdr = (
            f"  {parameter_name:<{col_w}}"
            f"  {'directory_name':<28}"
            f"  {'iter':>6}"
            f"  {variance_col:>20}"
            f"  {'rms':>14}"
        )
        lines.append(hdr)
        lines.append("-" * 72)
        for _, row in df.iterrows():
            mv = row.get(variance_col)
            rms = row.get('rms')
            mv_str = f"{mv:.6f}" if mv is not None else "n/a"
            rms_str = f"{rms:.6f}" if rms is not None else "n/a"
            lines.append(
                f"  {row['parameter_value']:<{col_w}.6g}"
                f"  {str(row['directory_name']):<28}"
                f"  {int(row['iteration']):>6}"
                f"  {mv_str:>20}"
                f"  {rms_str:>14}"
            )
        lines.append("")

    if p_table is not None and not p_table.empty:
        _write_section(p_table, "P-WAVE", "model_variance_vp")

    if s_table is not None and not s_table.empty:
        _write_section(s_table, "S-WAVE", "model_variance_vs")

    report_path.write_text('\n'.join(lines) + '\n')
    logger.info(f"Parameter tradeoff report written: {report_path}")


def detect_elbow(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> int:
    """
    Detect the 'elbow' index on a tradeoff curve using the perpendicular
    distance method (L-curve corner criterion).

    Algorithm:
        1. Normalize both axes to [0, 1] via min-max scaling.
        2. Draw a straight line from (x[0], y[0]) to (x[-1], y[-1]).
        3. Compute the perpendicular distance from each point to this line.
        4. Return the index of the maximum distance point.

    Args:
        x_values: X-axis values (model variance).
        y_values: Y-axis values (data misfit or RMS).

    Returns:
        Integer index into the arrays (not iteration number).
    """
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    if len(x) < 3:
        logger.warning(
            f"Only {len(x)} points available for elbow detection – "
            "returning midpoint index"
        )
        return len(x) // 2

    # Normalize to [0, 1]
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    if x_range == 0 or y_range == 0:
        logger.warning(
            "Constant axis in tradeoff curve – elbow detection unreliable, "
            "returning midpoint index"
        )
        return len(x) // 2

    x_norm = (x - x.min()) / x_range
    y_norm = (y - y.min()) / y_range

    # Line from first to last point
    dx = x_norm[-1] - x_norm[0]
    dy = y_norm[-1] - y_norm[0]
    line_len = np.sqrt(dx ** 2 + dy ** 2)

    if line_len < 1e-10:
        logger.warning(
            "First and last tradeoff points are identical – "
            "elbow detection unreliable, returning midpoint index"
        )
        return len(x) // 2

    # Perpendicular distance of each point from the line
    # Distance = |dy*(x_norm[i]-x_norm[0]) - dx*(y_norm[i]-y_norm[0])| / line_len
    distances = np.abs(
        dy * (x_norm - x_norm[0]) - dx * (y_norm - y_norm[0])
    ) / line_len

    elbow_idx = int(np.argmax(distances))
    logger.info(
        f"Elbow detected at array index {elbow_idx} "
        f"(max perpendicular distance: {distances[elbow_idx]:.4f})"
    )
    return elbow_idx
