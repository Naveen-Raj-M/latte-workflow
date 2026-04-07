"""
Statistical computations for tradeoff analysis.

Handles: spatial domain validation, model variance computation per iteration,
elbow/optimal-point detection, and summary statistics assembly.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Type alias for an optional float bound pair
BoundPair = Tuple[Optional[float], Optional[float]]


def _get_x_label(aggregation: str, use_pct: bool) -> str:
    """Return the X-axis label string matching the plot."""
    if use_pct:
        units, units_sq = "%", "%\u00b2"
    else:
        units, units_sq = "km/s", "(km/s)\u00b2"

    if aggregation == 'max':
        return f"Max Model Variation ({units})"
    elif aggregation == 'mean':
        return f"Average Model Variation ({units})"
    elif aggregation == 'model_variance':
        return f"Model Variance ({units_sq})"
    elif aggregation == 'rms_variation':
        return f"RMS Model Variation ({units})"
    else:
        return f"Model Variation ({units})"


def _get_x_formula(aggregation: str, use_pct: bool) -> str:
    """Return a human-readable formula string for the X-axis metric."""
    if use_pct:
        formulas = {
            'mean':           "mean(|\u0394v| / |v\u2080|) \u00d7 100   [%]",
            'max':            "max(|\u0394v| / |v\u2080|) \u00d7 100    [%]",
            'model_variance': "mean((\u0394v / v\u2080)\u00b2) \u00d7 10000  [%\u00b2]",
            'rms_variation':  "100 \u00d7 sqrt(mean((\u0394v / v\u2080)\u00b2))  [%]",
        }
    else:
        formulas = {
            'mean':           "mean(|\u0394v|)              [km/s]",
            'max':            "max(|\u0394v|)               [km/s]",
            'model_variance': "mean((\u0394v)\u00b2)             [(km/s)\u00b2]",
            'rms_variation':  "sqrt(mean((\u0394v)\u00b2))        [km/s]",
        }
    return formulas.get(aggregation, f"unknown aggregation: {aggregation}")


def _get_y_label(y_axis_mode: str, phase_mode: str, use_normalized: bool) -> str:
    """Return the Y-axis label string matching the plot."""
    if y_axis_mode == 'data_misfit':
        return "Normalized Data Misfit" if use_normalized else "Data Misfit"
    if y_axis_mode == 'data_variance':
        labels = {
            'both':     "Combined Data Variance (s\u00b2)",
            'p':        "P-wave Data Variance (s\u00b2)",
            's':        "S-wave Data Variance (s\u00b2)",
            'separate': "Data Variance (s\u00b2)",
        }
        return labels.get(phase_mode, "Data Variance (s\u00b2)")
    # rms mode
    labels = {
        'both':     "Combined RMS Residual (s)",
        'p':        "P-wave RMS Residual (s)",
        's':        "S-wave RMS Residual (s)",
        'separate': "RMS Residual (s)",
    }
    return labels.get(phase_mode, "RMS Residual (s)")


def _get_y_formula(y_axis_mode: str, phase_mode: str, use_normalized: bool) -> str:
    """Return a human-readable formula string for the Y-axis metric."""
    if y_axis_mode == 'data_misfit':
        if use_normalized:
            return "normalized_data_misfit column from data_misfit.txt (normalized to iteration 0)"
        return "data_misfit column from data_misfit.txt  [absolute]"
    if y_axis_mode == 'data_variance':
        phase_desc = {
            'both': 'all phases', 'p': 'P-wave only', 's': 'S-wave only', 'separate': 'P and S separately',
        }.get(phase_mode, phase_mode)
        return f"mean(residual_s\u00b2)  [s\u00b2]  ({phase_desc})"
    # rms
    phase_desc = {
        'both': 'all phases', 'p': 'P-wave only', 's': 'S-wave only', 'separate': 'P and S separately',
    }.get(phase_mode, phase_mode)
    return f"sqrt(mean(residual_s\u00b2))  [s]  ({phase_desc})"


def _get_column_descriptions(
    phase_mode: str,
    aggregation: str,
    use_pct: bool,
    y_axis_mode: str,
    use_normalized: bool,
) -> Dict[str, str]:
    """
    Return a dict mapping each statistics column name to a human-readable description.
    Descriptions match what is plotted on each axis.
    """
    units = "%" if use_pct else "km/s"
    units_sq = "%\u00b2" if use_pct else "(km/s)\u00b2"
    x_formula = _get_x_formula(aggregation, use_pct)

    agg_label = {
        'mean':           f"mean(|\u0394vp|)  [{units}]",
        'max':            f"max(|\u0394vp|)   [{units}]",
        'model_variance': f"mean((\u0394vp)\u00b2)  [{units_sq}]",
        'rms_variation':  f"sqrt(mean((\u0394vp)\u00b2))  [{units}]",
    }
    agg_label_vs = {k: v.replace('vp', 'vs') for k, v in agg_label.items()}

    col_vp = agg_label.get(aggregation, f"Vp model variation [{units}]")
    col_vs = agg_label_vs.get(aggregation, f"Vs model variation [{units}]")

    y_formula = _get_y_formula(y_axis_mode, phase_mode, use_normalized)

    descriptions: Dict[str, str] = {
        'iteration':          "Inversion iteration index",
        'model_variance_vp':  f"Vp X-axis metric: {col_vp}  |  formula: {x_formula}",
        'model_variance_vs':  f"Vs X-axis metric: {col_vs}  |  formula: {x_formula.replace('vp', 'vs')}",
        'model_variance_max': "max(model_variance_vp, model_variance_vs)  — used as X-axis when phase_mode != 'separate'",
    }

    if phase_mode == 'separate':
        descriptions['rms_p'] = f"P-wave Y-axis metric: {y_formula}"
        descriptions['rms_s'] = f"S-wave Y-axis metric: {y_formula}"
    else:
        descriptions['y_value'] = f"Y-axis metric: {y_formula}"

    return descriptions


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

    Value formulas (use_percentage_change=False):
        mean:           mean(|Δv|)              [km/s]
        max:            max(|Δv|)               [km/s]
        model_variance: mean((Δv)²)             [(km/s)²]
        rms_variation:  sqrt(mean((Δv)²))       [km/s]

    Value formulas (use_percentage_change=True):
        mean:           mean(|Δv| / |v_init|) × 100                    [%]
        max:            max(|Δv| / |v_init|)  × 100                    [%]
        model_variance: mean((Δv / v_init)²)  × 10000                  [%²]
        rms_variation:  100 × sqrt(mean((Δv / v_init)²))               [%]

        Grid points where |v_initial| < 1e-6 km/s are excluded from
        percentage computations. A WARNING is logged if any are found.

    Args:
        df_initial: Initial velocity model DataFrame.
        df_inverted: Inverted velocity model DataFrame for one iteration.
        mask: Boolean selection mask for grid points to include.
        use_percentage_change: If True, compute percentage-based metrics.
        aggregation: "mean", "max", or "model_variance".

    Returns:
        Dict with keys: 'vp', 'vs', 'max'.

    Raises:
        ValueError: If mask selects zero points or aggregation is invalid.
    """
    if aggregation not in ('mean', 'max', 'model_variance', 'rms_variation'):
        raise ValueError(
            f"Invalid aggregation: '{aggregation}'. "
            "Valid options: 'mean', 'max', 'model_variance', 'rms_variation'"
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

    vp_diff = vp_init - vp_inv
    vs_diff = vs_init - vs_inv
    vp_abs_diff = np.abs(vp_diff)
    vs_abs_diff = np.abs(vs_diff)

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

        vp_rel = vp_abs_diff[vp_valid] / np.abs(vp_init[vp_valid])
        vs_rel = vs_abs_diff[vs_valid] / np.abs(vs_init[vs_valid])

        if aggregation == 'max':
            variance_vp = float(np.max(vp_rel) * 100.0)
            variance_vs = float(np.max(vs_rel) * 100.0)
        elif aggregation == 'mean':
            variance_vp = float(np.mean(vp_rel) * 100.0)
            variance_vs = float(np.mean(vs_rel) * 100.0)
        elif aggregation == 'rms_variation':  # 100 × sqrt(mean((Δv/v_init)²))
            variance_vp = float(np.sqrt(np.mean(vp_rel ** 2)) * 100.0)
            variance_vs = float(np.sqrt(np.mean(vs_rel ** 2)) * 100.0)
        else:  # model_variance: mean((Δv/v_init)²) × 100²
            variance_vp = float(np.mean(vp_rel ** 2) * 10000.0)
            variance_vs = float(np.mean(vs_rel ** 2) * 10000.0)
    else:
        if aggregation == 'max':
            variance_vp = float(np.max(vp_abs_diff))
            variance_vs = float(np.max(vs_abs_diff))
        elif aggregation == 'mean':
            variance_vp = float(np.mean(vp_abs_diff))
            variance_vs = float(np.mean(vs_abs_diff))
        elif aggregation == 'rms_variation':  # sqrt(mean((Δv)²))
            variance_vp = float(np.sqrt(np.mean(vp_diff ** 2)))
            variance_vs = float(np.sqrt(np.mean(vs_diff ** 2)))
        else:  # model_variance: mean((Δv)²)
            variance_vp = float(np.mean(vp_diff ** 2))
            variance_vs = float(np.mean(vs_diff ** 2))

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


def write_statistics_csv(
    df: pd.DataFrame,
    csv_path: Path,
    config: Dict[str, Any],
    depth_km: Optional[float] = None,
) -> None:
    """
    Write the tradeoff statistics DataFrame to CSV with a metadata comment header.

    The header lines (prefixed with ``#``) document the plot title, axis labels,
    formulas, phase mode, and column descriptions so the CSV is self-contained.
    Re-read with ``pd.read_csv(path, comment='#')``.

    Args:
        df: Tradeoff statistics DataFrame.
        csv_path: Destination file path.
        config: Full configuration dictionary.
        depth_km: If set, include depth-layer context in the header.
    """
    from datetime import datetime

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    tcfg = config.get('tradeoff_analysis', {})
    mv_cfg = tcfg.get('model_variance', {})

    y_axis_mode  = tcfg.get('y_axis_mode', 'data_misfit')
    phase_mode   = tcfg.get('phase_mode', 'both')
    aggregation  = mv_cfg.get('aggregation', 'mean')
    use_pct      = mv_cfg.get('use_percentage_change', False)
    use_normalized = tcfg.get('use_normalized_misfit', False)
    region       = mv_cfg.get('region', 'full')

    x_label   = tcfg.get('plot', {}).get('x_label') or _get_x_label(aggregation, use_pct)
    x_formula = _get_x_formula(aggregation, use_pct)
    y_label   = tcfg.get('plot', {}).get('y_label') or _get_y_label(y_axis_mode, phase_mode, use_normalized)
    y_formula = _get_y_formula(y_axis_mode, phase_mode, use_normalized)
    col_desc  = _get_column_descriptions(phase_mode, aggregation, use_pct, y_axis_mode, use_normalized)

    region_label = {'full': 'Full Model', 'z_bounded': 'Z-bounded Region'}.get(region, 'Depth Layer')
    if depth_km is not None:
        region_label = f"Depth Layer Z = {depth_km:.2f} km"

    if phase_mode == 'separate':
        plot_title = (
            f"Data Misfit vs Model Variance Tradeoff ({region_label}, P-wave)  /  "
            f"Data Misfit vs Model Variance Tradeoff ({region_label}, S-wave)"
        )
    else:
        plot_title = f"Data Misfit vs Model Variance Tradeoff ({region_label})"

    _phase_desc = {
        'both':     "both — P and S residuals pooled; single combined y-axis value",
        'p':        "p — P-wave residuals only",
        's':        "s — S-wave residuals only",
        'separate': "separate — rms_p and rms_s columns hold independent P/S values",
    }.get(phase_mode, phase_mode)

    meta = [
        f"# TRADEOFF STATISTICS",
        f"# Generated     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Plot title    : {plot_title}",
        f"#",
        f"# X-axis label  : {x_label}",
        f"# X-axis formula: Δv = v_initial − v_inverted",
        f"#                 {x_formula}",
        f"#",
        f"# Y-axis label  : {y_label}",
        f"# Y-axis formula: {y_formula}",
        f"#",
        f"# Phase mode    : {_phase_desc}",
        f"# Y-axis mode   : {y_axis_mode}",
        f"# Aggregation   : {aggregation}",
        f"# Percentage    : {use_pct}",
        f"#",
        f"# COLUMN DESCRIPTIONS",
    ]
    for col, desc in col_desc.items():
        meta.append(f"#   {col:<28}: {desc}")
    meta.append(f"#")
    meta.append(f"# NOTE: re-read with pd.read_csv(path, comment='#')")
    meta.append(f"#")

    with open(csv_path, 'w') as fh:
        fh.write('\n'.join(meta) + '\n')
        fh.write(df.to_csv(index=False))

    logger.info(f"Saved statistics CSV with metadata header: {csv_path}")


def write_parameter_report(
    parameter_name: str,
    p_table: Optional[pd.DataFrame],
    s_table: Optional[pd.DataFrame],
    report_path: Path,
) -> None:
    """
    Write a plain text report mapping parameter values to model variance.

    Args:
        parameter_name: Name of the swept parameter (e.g., "step_max").
        p_table: P-wave parameter tradeoff DataFrame (or None/empty).
        s_table: S-wave parameter tradeoff DataFrame (or None/empty).
        report_path: Full path for the output .txt file.
    """
    from datetime import datetime

    report_path = Path(report_path)
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


def write_general_report(
    config: Dict[str, Any],
    tradeoff_results: Dict[str, Any],
    report_path: Path,
) -> None:
    """
    Write a comprehensive plain-text report for the tradeoff analysis run.

    Includes evaluation mode, all configuration parameters, paths, and a
    full table of computed values per iteration (or per depth layer).

    Args:
        config: Full configuration dictionary.
        tradeoff_results: Dict returned by the main orchestration loop.
        report_path: Full path for the output .txt file.
    """
    from datetime import datetime

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    tcfg = config.get('tradeoff_analysis', {})
    mv_cfg = tcfg.get('model_variance', {})

    x_axis_mode        = tcfg.get('x_axis_mode', 'iteration')
    y_axis_mode        = tcfg.get('y_axis_mode', 'data_misfit')
    phase_mode         = tcfg.get('phase_mode', 'both')
    rms_comparison     = tcfg.get('rms_comparison_mode', 'ground_truth')
    region             = mv_cfg.get('region', 'full')
    aggregation        = mv_cfg.get('aggregation', 'mean')
    use_pct            = mv_cfg.get('use_percentage_change', False)
    velocity_dataset   = tcfg.get('velocity_dataset', 'velocity')
    use_normalized     = tcfg.get('use_normalized_misfit', False)

    # Derive axis labels and formulas (mirrors what is rendered on the plot)
    x_label   = tcfg.get('plot', {}).get('x_label') or _get_x_label(aggregation, use_pct)
    x_formula = _get_x_formula(aggregation, use_pct)
    y_label   = tcfg.get('plot', {}).get('y_label') or _get_y_label(y_axis_mode, phase_mode, use_normalized)
    y_formula = _get_y_formula(y_axis_mode, phase_mode, use_normalized)

    # Derive plot title (mirrors create_all_tradeoff_plots logic)
    if region == 'full':
        region_label = "Full Model"
    elif region == 'z_bounded':
        region_label = "Z-bounded Region"
    else:
        region_label = "Depth Layer"

    if phase_mode == 'separate':
        plot_title_p = f"Data Misfit vs Model Variance Tradeoff ({region_label}, P-wave)"
        plot_title_s = f"Data Misfit vs Model Variance Tradeoff ({region_label}, S-wave)"
        plot_title   = f"{plot_title_p}  /  {plot_title_s}"
    else:
        plot_title = f"Data Misfit vs Model Variance Tradeoff ({region_label})"

    # Column descriptions keyed to the statistics CSV
    col_desc = _get_column_descriptions(
        phase_mode, aggregation, use_pct, y_axis_mode, use_normalized
    )

    sep  = "=" * 72
    sep2 = "-" * 72

    _phase_mode_desc = {
        'both':     "both  (P and S residuals combined into a single y-axis value)",
        'p':        "p     (P-wave residuals only)",
        's':        "s     (S-wave residuals only)",
        'separate': "separate  (P and S plotted on separate axes; rms_p / rms_s columns)",
    }.get(phase_mode, phase_mode)

    lines = [
        "Tradeoff Analysis Report",
        sep,
        f"Generated              : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sep,
        "",
        "CONFIGURATION",
        sep2,
        f"  X-axis mode          : {x_axis_mode}",
        f"  Y-axis mode          : {y_axis_mode}",
        f"  Phase mode           : {_phase_mode_desc}",
    ]

    if y_axis_mode == 'data_misfit':
        lines.append(f"  Use normalized misfit : {use_normalized}")
    elif y_axis_mode in ('rms', 'data_variance'):
        lines.append(f"  RMS comparison mode  : {rms_comparison}")
        thresh_lo = tcfg.get('min_residual_threshold', -50.0)
        thresh_hi = tcfg.get('max_residual_threshold',  50.0)
        lines.append(f"  Residual thresholds  : [{thresh_lo}, {thresh_hi}] s")

    lines += [
        f"  Region               : {region}",
        f"  Aggregation          : {aggregation}",
        f"  Use percentage change: {use_pct}",
        f"  Velocity dataset     : {velocity_dataset}",
    ]

    if region == 'z_bounded':
        lines.append(f"  Z bounds             : {mv_cfg.get('z_bounds')}")
    elif region == 'depth_layer':
        lines.append(f"  Depth layers (km)    : {mv_cfg.get('depth_layers')}")
        lines.append(f"  Depth tolerance (km) : {mv_cfg.get('depth_tolerance_km')}")

    x_lo = mv_cfg.get('x_bounds', [None, None])
    y_lo = mv_cfg.get('y_bounds', [None, None])
    if any(b is not None for b in x_lo):
        lines.append(f"  X bounds             : {x_lo}")
    if any(b is not None for b in y_lo):
        lines.append(f"  Y bounds             : {y_lo}")

    # Phase interpretation note for PLOT DETAILS
    _phase_plot_note = {
        'both':     ("Both P and S residuals are pooled together and a single combined "
                     "RMS/variance is computed.  One curve is plotted; the X-axis uses "
                     "max(model_variance_vp, model_variance_vs)."),
        'p':        ("Only P-wave residuals are used.  One curve is plotted; the X-axis "
                     "uses model_variance_vp."),
        's':        ("Only S-wave residuals are used.  One curve is plotted; the X-axis "
                     "uses model_variance_vs."),
        'separate': ("P and S residuals are computed independently.  Two curves are "
                     "plotted (one per wave type); columns rms_p / rms_s in the CSV "
                     "hold the respective Y-axis values."),
    }.get(phase_mode, phase_mode)

    lines += [
        "",
        "PLOT DETAILS",
        sep2,
        f"  Plot title           : {plot_title}",
        "",
        f"  X-axis label         : {x_label}",
        f"  X-axis formula       : \u0394v = v_initial \u2212 v_inverted",
        f"                         {x_formula}",
        "",
        f"  Y-axis label         : {y_label}",
        f"  Y-axis formula       : {y_formula}",
        "",
        f"  Phase mode           : {_phase_mode_desc}",
        f"  Phase interpretation : {_phase_plot_note}",
    ]

    if region == 'depth_layer':
        depth_layers = mv_cfg.get('depth_layers', [])
        tol = mv_cfg.get('depth_tolerance_km', 0.5)
        lines.append("")
        for d in depth_layers:
            lines.append(f"    Depth layer plot   : Z = {d} km (±{tol} km)  →  "
                         f"tradeoff_curve_depth_{float(d):.2f}km.*")

    lines += [
        "",
        "COLUMN DESCRIPTIONS  (statistics CSV)",
        sep2,
    ]
    for col, desc in col_desc.items():
        lines.append(f"  {col:<28} : {desc}")

    lines += [
        "",
        "PATHS",
        sep2,
    ]

    path_fields = [
        ('initial_model_hdf5',      'Initial model HDF5      '),
        ('initial_model_csv',       'Initial model CSV       '),
        ('inversion_base_path',     'Inversion base path     '),
        ('inversion_analysis_dir',  'Inversion analysis dir  '),
        ('data_misfit_file_path',   'Data misfit file        '),
        ('method1_csv',             'Method 1 CSV            '),
        ('output_dir',              'Output directory        '),
    ]
    for key, label in path_fields:
        val = tcfg.get(key)
        if val:
            lines.append(f"  {label}: {val}")

    lines.append("")

    # Build a short column legend line (matches the CSV header)
    def _column_legend_lines(col_desc_map: Dict[str, str]) -> List[str]:
        out = ["  Column legend:"]
        for col, desc in col_desc_map.items():
            out.append(f"    {col:<28} : {desc}")
        return out

    # --- Computed values ---
    if 'tradeoff_table' in tradeoff_results:
        df = tradeoff_results['tradeoff_table']
        if not df.empty:
            lines += [
                "COMPUTED VALUES",
                sep2,
                f"  Iterations processed : {len(df)}",
                f"  Iteration range      : [{int(df['iteration'].min())}, "
                f"{int(df['iteration'].max())}]",
                "",
            ]
            lines += _column_legend_lines(col_desc)
            lines.append("")
            lines.append(df.to_string(index=False))
        else:
            lines += ["COMPUTED VALUES", sep2, "  (no data)"]

    elif 'by_depth' in tradeoff_results:
        by_depth = tradeoff_results['by_depth']
        lines += ["COMPUTED VALUES (per depth layer)", sep2]
        for depth_km in sorted(by_depth.keys()):
            df = by_depth[depth_km]
            lines.append(f"\n  Z = {depth_km:.2f} km")
            lines.append("  " + "-" * 40)
            if not df.empty:
                lines.append(f"  Iterations processed : {len(df)}")
                lines.append(f"  Iteration range      : [{int(df['iteration'].min())}, "
                             f"{int(df['iteration'].max())}]")
                lines.append("")
                lines += _column_legend_lines(col_desc)
                lines.append("")
                lines.append(df.to_string(index=False))
            else:
                lines.append("  (no data)")

    report_path.write_text('\n'.join(lines) + '\n')
    logger.info(f"Tradeoff analysis report written: {report_path}")


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
