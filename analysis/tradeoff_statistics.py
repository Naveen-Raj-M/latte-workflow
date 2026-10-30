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


def _get_y_label(
    y_axis_mode: str,
    phase_mode: str,
    use_normalized: bool,
    use_log_residuals: bool = False,
    log_base: str = 'natural',
) -> str:
    """Return the Y-axis label string matching the plot."""
    if y_axis_mode == 'data_misfit':
        return "Normalized Data Misfit" if use_normalized else "Data Misfit"
    log_tag = ""
    if use_log_residuals:
        log_tag = " [ln]" if log_base == 'natural' else " [log₁₀]"
    if y_axis_mode == 'data_variance':
        labels = {
            'both':     f"Combined Log-Data Variance{log_tag}" if use_log_residuals else "Combined Data Variance (s\u00b2)",
            'p':        f"P-wave Log-Data Variance{log_tag}" if use_log_residuals else "P-wave Data Variance (s\u00b2)",
            's':        f"S-wave Log-Data Variance{log_tag}" if use_log_residuals else "S-wave Data Variance (s\u00b2)",
            'separate': f"Log-Data Variance{log_tag}" if use_log_residuals else "Data Variance (s\u00b2)",
        }
        return labels.get(phase_mode, f"Log-Data Variance{log_tag}" if use_log_residuals else "Data Variance (s\u00b2)")
    # rms mode
    if use_log_residuals:
        labels = {
            'both':     f"Combined Log-RMS Residual{log_tag}",
            'p':        f"P-wave Log-RMS Residual{log_tag}",
            's':        f"S-wave Log-RMS Residual{log_tag}",
            'separate': f"Log-RMS Residual{log_tag}",
        }
        return labels.get(phase_mode, f"Log-RMS Residual{log_tag}")
    labels = {
        'both':     "Combined RMS Residual (s)",
        'p':        "P-wave RMS Residual (s)",
        's':        "S-wave RMS Residual (s)",
        'separate': "RMS Residual (s)",
    }
    return labels.get(phase_mode, "RMS Residual (s)")


def _get_y_formula(
    y_axis_mode: str,
    phase_mode: str,
    use_normalized: bool,
    use_log_residuals: bool = False,
    log_base: str = 'natural',
    log_residual_epsilon: float = 1e-10,
) -> str:
    """Return a human-readable formula string for the Y-axis metric."""
    if y_axis_mode == 'data_misfit':
        if use_normalized:
            return "normalized_data_misfit column from data_misfit.txt (normalized to iteration 0)"
        return "data_misfit column from data_misfit.txt  [absolute]"
    phase_desc = {
        'both': 'all phases', 'p': 'P-wave only', 's': 'S-wave only', 'separate': 'P and S separately',
    }.get(phase_mode, phase_mode)
    if use_log_residuals:
        log_fn = "ln" if log_base == 'natural' else "log\u2081\u2080"
        r_expr = f"|{log_fn}(max(obs_tt, \u03b5)) \u2212 {log_fn}(max(syn_tt, \u03b5))|  [\u03b5={log_residual_epsilon}]"
        if y_axis_mode == 'data_variance':
            return f"mean(t\u00b2)  where t = {r_expr}  ({phase_desc})"
        return f"sqrt(mean(t\u00b2))  where t = {r_expr}  ({phase_desc})"
    if y_axis_mode == 'data_variance':
        return f"mean(residual_s\u00b2)  [s\u00b2]  ({phase_desc})"
    return f"sqrt(mean(residual_s\u00b2))  [s]  ({phase_desc})"


def _build_col_names(
    aggregation: str,
    use_pct: bool,
    phase_mode: str,
    y_axis_mode: str,
    use_normalized: bool,
) -> Dict[str, Optional[str]]:
    """
    Return the exact column names used in the statistics DataFrame / CSV.

    The names encode aggregation, units, and phase so the CSV is self-describing.

    Keys always present:
        'x'    — primary X column (combined for 'both', vp for 'p'/'separate', vs for 's')
        'x_vs' — secondary X column for Vs (only set when phase_mode='separate')
        'y'    — Y column for non-separate modes (None when phase_mode='separate')
        'y_p'  — P-wave Y column (only set when phase_mode='separate')
        'y_s'  — S-wave Y column (only set when phase_mode='separate')
    """
    # Aggregation prefix
    agg_prefix = {
        'mean':           'mean_var',
        'max':            'max_var',
        'model_variance': 'model_var',
        'rms_variation':  'rms_var',
    }.get(aggregation, aggregation)

    # Units suffix
    if aggregation == 'model_variance':
        units = 'pct_sq' if use_pct else 'km_s_sq'
    else:
        units = 'pct' if use_pct else 'km_s'

    # X column names
    if phase_mode == 'both':
        x     = f"{agg_prefix}_combined_{units}"
        x_vs  = None
    elif phase_mode == 'p':
        x     = f"{agg_prefix}_vp_{units}"
        x_vs  = None
    elif phase_mode == 's':
        x     = f"{agg_prefix}_vs_{units}"
        x_vs  = None
    else:  # separate
        x     = f"{agg_prefix}_vp_{units}"
        x_vs  = f"{agg_prefix}_vs_{units}"

    # Y column names
    if y_axis_mode == 'data_misfit':
        y     = 'data_misfit_norm' if use_normalized else 'data_misfit'
        y_p   = None
        y_s   = None
    elif y_axis_mode == 'data_variance':
        if phase_mode == 'separate':
            y, y_p, y_s = None, 'data_var_p_s2', 'data_var_s_s2'
        elif phase_mode == 'p':
            y, y_p, y_s = 'data_var_p_s2', None, None
        elif phase_mode == 's':
            y, y_p, y_s = 'data_var_s_s2', None, None
        else:  # both
            y, y_p, y_s = 'data_var_combined_s2', None, None
    else:  # rms
        if phase_mode == 'separate':
            y, y_p, y_s = None, 'rms_p_sec', 'rms_s_sec'
        elif phase_mode == 'p':
            y, y_p, y_s = 'rms_p_sec', None, None
        elif phase_mode == 's':
            y, y_p, y_s = 'rms_s_sec', None, None
        else:  # both
            y, y_p, y_s = 'rms_combined_sec', None, None

    return {'x': x, 'x_vs': x_vs, 'y': y, 'y_p': y_p, 'y_s': y_s}


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
        df_initial: Initial velocity model DataFrame with X/Y/z columns.
        x_bounds: (x_min, x_max) – either element may be None (no bound).
        y_bounds: (y_min, y_max) – same convention.
        z_bounds: (z_min, z_max) or None (no Z filter).

    Raises:
        ValueError: If any bound has zero overlap with the model domain.
    """
    axes = [
        ('X', 'x', x_bounds),
        ('Y', 'y', y_bounds),
    ]
    if z_bounds is not None:
        axes.append(('Z', 'z', z_bounds))

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
        df: Velocity model DataFrame with x, y, z.
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

    _apply_bound('x', *x_bounds)
    _apply_bound('y', *y_bounds)
    if z_bounds is not None:
        _apply_bound('z', *z_bounds)

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
    depth_mask = np.abs(df['z'].values - depth_km) <= depth_tolerance_km

    # Apply horizontal bounds
    xy_mask = np.ones(len(df), dtype=bool)
    x_lo, x_hi = x_bounds
    y_lo, y_hi = y_bounds
    if x_lo is not None:
        xy_mask &= (df['x'].values >= x_lo)
    if x_hi is not None:
        xy_mask &= (df['x'].values <= x_hi)
    if y_lo is not None:
        xy_mask &= (df['y'].values >= y_lo)
    if y_hi is not None:
        xy_mask &= (df['y'].values <= y_hi)

    combined = depth_mask & xy_mask
    n_selected = int(combined.sum())

    if n_selected == 0:
        available_z = np.sort(df['z'].unique())
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
    exclude_zero_change: bool = False,
    zero_change_tolerance: float = 1e-6,
    phase_mode: str = 'both',
) -> Dict[str, Any]:
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

    When aggregation='rms_variation' and exclude_zero_change=True, grid
    points whose |Δv| <= zero_change_tolerance (km/s) are excluded from
    the RMS computation at each call.  The number of points actually used
    is returned as 'n_used_vp' and 'n_used_vs' in the result dict so it
    can be tracked per iteration.

    Args:
        df_initial: Initial velocity model DataFrame.
        df_inverted: Inverted velocity model DataFrame for one iteration.
        mask: Boolean selection mask for grid points to include.
        use_percentage_change: If True, compute percentage-based metrics.
        aggregation: "mean", "max", "model_variance", or "rms_variation".
        exclude_zero_change: If True and aggregation='rms_variation', omit
            grid points where |Δv| <= zero_change_tolerance from the RMS.
        zero_change_tolerance: Absolute velocity-change threshold (km/s)
            below which a node is considered unchanged.  Used only when
            exclude_zero_change=True and aggregation='rms_variation'.
        phase_mode: One of 'both', 'p', 's', 'separate'. When 'p', only Vp
            variance is computed and 'vs'/'n_used_vs' are returned as None.
            When 's', only Vs variance is computed and 'vp'/'n_used_vp' are
            returned as None. 'both' and 'separate' compute both phases.

    Returns:
        Dict with keys: 'vp', 'vs', 'combined', 'n_used_vp', 'n_used_vs'.
        'vp' and 'n_used_vp' are None when phase_mode='s'; 'vs' and
        'n_used_vs' are None when phase_mode='p'.
        n_used_* reflect the number of grid points included in the
        computation (after any zero-change filtering).

    Raises:
        ValueError: If mask selects zero points, aggregation is invalid,
            or exclude_zero_change filters out all points.
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

    compute_vp = phase_mode in ('p', 'both', 'separate')
    compute_vs = phase_mode in ('s', 'both', 'separate')

    # Load only the arrays needed for the requested phase(s)
    if compute_vp:
        vp_init = df_initial['vp'].values[mask]
        vp_inv  = df_inverted['vp'].values[mask]
        vp_diff = vp_init - vp_inv
        vp_abs_diff = np.abs(vp_diff)

    if compute_vs:
        vs_init = df_initial['vs'].values[mask]
        vs_inv  = df_inverted['vs'].values[mask]
        vs_diff = vs_init - vs_inv
        vs_abs_diff = np.abs(vs_diff)

    variance_vp: Optional[float] = None
    variance_vs: Optional[float] = None
    n_used_vp:   Optional[int]   = None
    n_used_vs:   Optional[int]   = None

    if use_percentage_change:
        if compute_vp:
            vp_valid = np.abs(vp_init) >= 1e-6
            n_vp_excluded = int((~vp_valid).sum())
            if n_vp_excluded:
                logger.warning(
                    f"{n_vp_excluded} grid points have |vp_initial| < 1e-6 km/s "
                    "and are excluded from percentage variance computation."
                )
            if not vp_valid.any():
                raise ValueError(
                    "All selected grid points have near-zero initial Vp – "
                    "cannot compute percentage model variance."
                )
            vp_rel = vp_abs_diff[vp_valid] / np.abs(vp_init[vp_valid])

        if compute_vs:
            vs_valid = np.abs(vs_init) >= 1e-6
            n_vs_excluded = int((~vs_valid).sum())
            if n_vs_excluded:
                logger.warning(
                    f"{n_vs_excluded} grid points have |vs_initial| < 1e-6 km/s "
                    "and are excluded from percentage variance computation."
                )
            if not vs_valid.any():
                raise ValueError(
                    "All selected grid points have near-zero initial Vs – "
                    "cannot compute percentage model variance."
                )
            vs_rel = vs_abs_diff[vs_valid] / np.abs(vs_init[vs_valid])

        if aggregation == 'max':
            if compute_vp:
                n_used_vp   = int(vp_valid.sum())
                variance_vp = float(np.max(vp_rel) * 100.0)
            if compute_vs:
                n_used_vs   = int(vs_valid.sum())
                variance_vs = float(np.max(vs_rel) * 100.0)
        elif aggregation == 'mean':
            if compute_vp:
                n_used_vp   = int(vp_valid.sum())
                variance_vp = float(np.mean(vp_rel) * 100.0)
            if compute_vs:
                n_used_vs   = int(vs_valid.sum())
                variance_vs = float(np.mean(vs_rel) * 100.0)
        elif aggregation == 'rms_variation':  # 100 × sqrt(mean((Δv/v_init)²))
            if exclude_zero_change:
                if compute_vp:
                    vp_nonzero = vp_abs_diff[vp_valid] > zero_change_tolerance
                    n_used_vp  = int(vp_nonzero.sum())
                    if n_used_vp == 0:
                        raise ValueError(
                            f"All valid Vp grid points have |Δvp| <= {zero_change_tolerance} km/s "
                            "(zero-change tolerance); cannot compute rms_variation."
                        )
                    vp_rel_used = vp_rel[vp_nonzero]
                if compute_vs:
                    vs_nonzero = vs_abs_diff[vs_valid] > zero_change_tolerance
                    n_used_vs  = int(vs_nonzero.sum())
                    if n_used_vs == 0:
                        raise ValueError(
                            f"All valid Vs grid points have |Δvs| <= {zero_change_tolerance} km/s "
                            "(zero-change tolerance); cannot compute rms_variation."
                        )
                    vs_rel_used = vs_rel[vs_nonzero]
                dbg_vp = f"Vp {n_used_vp}/{int(vp_valid.sum())} pts" if compute_vp else "Vp skipped"
                dbg_vs = f"Vs {n_used_vs}/{int(vs_valid.sum())} pts" if compute_vs else "Vs skipped"
                logger.debug(
                    f"rms_variation (pct, exclude_zero): {dbg_vp}, {dbg_vs} "
                    f"(tol={zero_change_tolerance} km/s)"
                )
            else:
                if compute_vp:
                    n_used_vp   = int(vp_valid.sum())
                    vp_rel_used = vp_rel
                if compute_vs:
                    n_used_vs   = int(vs_valid.sum())
                    vs_rel_used = vs_rel
            if compute_vp:
                variance_vp = float(np.sqrt(np.mean(vp_rel_used ** 2)) * 100.0)
            if compute_vs:
                variance_vs = float(np.sqrt(np.mean(vs_rel_used ** 2)) * 100.0)
        else:  # model_variance: mean((Δv/v_init)²) × 100²
            if compute_vp:
                n_used_vp   = int(vp_valid.sum())
                variance_vp = float(np.mean(vp_rel ** 2) * 10000.0)
            if compute_vs:
                n_used_vs   = int(vs_valid.sum())
                variance_vs = float(np.mean(vs_rel ** 2) * 10000.0)
    else:
        if aggregation == 'max':
            if compute_vp:
                n_used_vp   = n_selected
                variance_vp = float(np.max(vp_abs_diff))
            if compute_vs:
                n_used_vs   = n_selected
                variance_vs = float(np.max(vs_abs_diff))
        elif aggregation == 'mean':
            if compute_vp:
                n_used_vp   = n_selected
                variance_vp = float(np.mean(vp_abs_diff))
            if compute_vs:
                n_used_vs   = n_selected
                variance_vs = float(np.mean(vs_abs_diff))
        elif aggregation == 'rms_variation':  # sqrt(mean((Δv)²))
            if exclude_zero_change:
                if compute_vp:
                    vp_nonzero = vp_abs_diff > zero_change_tolerance
                    n_used_vp  = int(vp_nonzero.sum())
                    if n_used_vp == 0:
                        raise ValueError(
                            f"All Vp grid points have |Δvp| <= {zero_change_tolerance} km/s "
                            "(zero-change tolerance); cannot compute rms_variation."
                        )
                if compute_vs:
                    vs_nonzero = vs_abs_diff > zero_change_tolerance
                    n_used_vs  = int(vs_nonzero.sum())
                    if n_used_vs == 0:
                        raise ValueError(
                            f"All Vs grid points have |Δvs| <= {zero_change_tolerance} km/s "
                            "(zero-change tolerance); cannot compute rms_variation."
                        )
                dbg_vp = f"Vp {n_used_vp}/{n_selected} pts" if compute_vp else "Vp skipped"
                dbg_vs = f"Vs {n_used_vs}/{n_selected} pts" if compute_vs else "Vs skipped"
                logger.debug(
                    f"rms_variation (exclude_zero): {dbg_vp}, {dbg_vs} "
                    f"(tol={zero_change_tolerance} km/s)"
                )
                if compute_vp:
                    variance_vp = float(np.sqrt(np.mean(vp_diff[vp_nonzero] ** 2)))
                if compute_vs:
                    variance_vs = float(np.sqrt(np.mean(vs_diff[vs_nonzero] ** 2)))
            else:
                if compute_vp:
                    n_used_vp   = n_selected
                    variance_vp = float(np.sqrt(np.mean(vp_diff ** 2)))
                if compute_vs:
                    n_used_vs   = n_selected
                    variance_vs = float(np.sqrt(np.mean(vs_diff ** 2)))
        else:  # model_variance: mean((Δv)²)
            if compute_vp:
                n_used_vp   = n_selected
                variance_vp = float(np.mean(vp_diff ** 2))
            if compute_vs:
                n_used_vs   = n_selected
                variance_vs = float(np.mean(vs_diff ** 2))

    if phase_mode == 'p':
        return {
            'vp': variance_vp, 'vs': None,
            'combined': variance_vp,
            'n_used_vp': n_used_vp, 'n_used_vs': None,
        }
    if phase_mode == 's':
        return {
            'vp': None, 'vs': variance_vs,
            'combined': variance_vs,
            'n_used_vp': None, 'n_used_vs': n_used_vs,
        }
    variance_combined = (variance_vp + variance_vs) / 2.0
    return {
        'vp': variance_vp,
        'vs': variance_vs,
        'combined': variance_combined,
        'n_used_vp': n_used_vp,
        'n_used_vs': n_used_vs,
    }


def assemble_tradeoff_table(
    iteration_indices: List[int],
    y_values: Dict[int, Any],
    vp_variances: Dict[int, float],
    vs_variances: Dict[int, float],
    combined_variances: Dict[int, float],
    phase_mode: str,
    aggregation: str = 'mean',
    use_pct: bool = False,
    y_axis_mode: str = 'data_misfit',
    use_normalized: bool = False,
    n_used_vp_values: Optional[Dict[int, int]] = None,
    n_used_vs_values: Optional[Dict[int, int]] = None,
) -> pd.DataFrame:
    """
    Assemble per-iteration statistics into a tradeoff DataFrame.

    Column names are derived from config so the CSV is self-describing:
        phase_mode='both'     → [iteration, <x_combined>, <y_combined>]
        phase_mode='p'        → [iteration, <x_vp>, <y_p>]
        phase_mode='s'        → [iteration, <x_vs>, <y_s>]
        phase_mode='separate' → [iteration, <x_vp>, <x_vs>, <y_p>, <y_s>]

    The X-axis value for phase_mode='both' is the mean of Vp and Vs metrics;
    for 'p'/'s'/'separate' it is the per-wave metric directly.

    Args:
        iteration_indices:  Ordered list of iteration integers to include.
        y_values:           {iter: float} or {iter: {'p': float, 's': float}}.
        vp_variances:       {iter: vp metric}.
        vs_variances:       {iter: vs metric}.
        combined_variances: {iter: mean(vp, vs) metric} — used for phase_mode='both'.
        phase_mode:         'both' | 'p' | 's' | 'separate'.
        aggregation:        'mean' | 'max' | 'model_variance' | 'rms_variation'.
        use_pct:            Whether metrics are in %.
        y_axis_mode:        'data_misfit' | 'rms' | 'data_variance'.
        use_normalized:     Whether data_misfit is normalized.
        n_used_vp_values:   Optional {iter: int} of Vp grid points used in
                            the variance computation (after zero-change filtering).
                            When provided, a 'n_used_vp' column is added.
        n_used_vs_values:   Optional {iter: int} of Vs grid points used.
                            When provided, a 'n_used_vs' column is added.

    Returns:
        DataFrame sorted ascending by iteration. Rows with missing data excluded.
    """
    cols = _build_col_names(aggregation, use_pct, phase_mode, y_axis_mode, use_normalized)
    x_col   = cols['x']
    x_vs_col = cols['x_vs']
    y_col   = cols['y']
    y_p_col = cols['y_p']
    y_s_col = cols['y_s']

    rows = []
    skipped = []

    for idx in iteration_indices:
        vp_var   = vp_variances.get(idx)
        vs_var   = vs_variances.get(idx)
        comb_var = combined_variances.get(idx)
        y_val    = y_values.get(idx)

        need_vp = phase_mode in ('p', 'both', 'separate')
        need_vs = phase_mode in ('s', 'both', 'separate')
        if (need_vp and vp_var is None) or (need_vs and vs_var is None):
            skipped.append(idx)
            continue

        if phase_mode == 'separate':
            if not isinstance(y_val, dict) or y_val.get('p') is None or y_val.get('s') is None:
                skipped.append(idx)
                continue
            row: Dict[str, Any] = {
                'iteration': idx,
                x_col:    vp_var,
                x_vs_col: vs_var,
                y_p_col:  y_val['p'],
                y_s_col:  y_val['s'],
            }
            if n_used_vp_values is not None:
                row['n_used_vp'] = n_used_vp_values.get(idx)
            if n_used_vs_values is not None:
                row['n_used_vs'] = n_used_vs_values.get(idx)
            rows.append(row)
        else:
            # Resolve scalar y
            if isinstance(y_val, dict):
                key = next(iter(y_val))
                scalar_y = y_val.get(key)
            else:
                scalar_y = y_val

            if scalar_y is None:
                skipped.append(idx)
                continue

            # X value: combined mean for 'both', per-wave for 'p'/'s'
            if phase_mode == 'both':
                x_val = comb_var if comb_var is not None else (vp_var + vs_var) / 2.0
            elif phase_mode == 's':
                x_val = vs_var
            else:  # 'p'
                x_val = vp_var

            row = {
                'iteration': idx,
                x_col:  x_val,
                y_col:  scalar_y,
            }
            if n_used_vp_values is not None:
                row['n_used_vp'] = n_used_vp_values.get(idx)
            if n_used_vs_values is not None:
                row['n_used_vs'] = n_used_vs_values.get(idx)
            rows.append(row)

    if skipped:
        logger.warning(
            f"Skipped {len(skipped)} iterations due to missing data: {skipped}"
        )

    if not rows:
        logger.warning("Tradeoff table is empty – no complete iteration data found")
        if phase_mode == 'separate':
            empty_cols = ['iteration', x_col, x_vs_col, y_p_col, y_s_col]
        else:
            empty_cols = ['iteration', x_col, y_col]
        return pd.DataFrame(columns=empty_cols)

    df = pd.DataFrame(rows).sort_values('iteration').reset_index(drop=True)
    logger.info(
        f"Tradeoff table assembled: {len(df)} iterations "
        f"[{df['iteration'].min()}–{df['iteration'].max()}]"
    )
    return df


def assemble_parameter_tradeoff_table(
    processed_entries: List[Dict[str, Any]],
    phase_mode: str = 'both',
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
    need_vp = phase_mode in ('p', 'both')
    need_vs = phase_mode in ('s', 'both')
    for e in processed_entries:
        if (need_vp and e.get('model_variance_vp') is None) or \
                (need_vs and e.get('model_variance_vs') is None):
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


def write_statistics_csv(df: pd.DataFrame, csv_path: Path) -> None:
    """Write the tradeoff statistics DataFrame to CSV. Column names are self-describing."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved tradeoff statistics: {csv_path}")


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

    x_axis_mode           = tcfg.get('x_axis_mode', 'iteration')
    y_axis_mode           = tcfg.get('y_axis_mode', 'data_misfit')
    phase_mode            = tcfg.get('phase_mode', 'both')
    rms_comparison        = tcfg.get('rms_comparison_mode', 'ground_truth')
    region                = mv_cfg.get('region', 'full')
    aggregation           = mv_cfg.get('aggregation', 'mean')
    use_pct               = mv_cfg.get('use_percentage_change', False)
    velocity_dataset      = tcfg.get('velocity_dataset', 'velocity')
    use_normalized        = tcfg.get('use_normalized_misfit', False)
    use_log_residuals     = bool(tcfg.get('use_log_residuals', False))
    log_base              = str(tcfg.get('log_residual_base', 'natural'))
    log_residual_epsilon  = float(tcfg.get('log_residual_epsilon', 1e-10))

    # Derive axis labels and formulas (mirrors what is rendered on the plot)
    x_label   = tcfg.get('plot', {}).get('x_label') or _get_x_label(aggregation, use_pct)
    x_formula = _get_x_formula(aggregation, use_pct)
    y_label   = tcfg.get('plot', {}).get('y_label') or _get_y_label(
        y_axis_mode, phase_mode, use_normalized, use_log_residuals, log_base
    )
    y_formula = _get_y_formula(
        y_axis_mode, phase_mode, use_normalized,
        use_log_residuals, log_base, log_residual_epsilon,
    )

    # Column names that appear in the CSV (derived from config)
    cols = _build_col_names(aggregation, use_pct, phase_mode, y_axis_mode, use_normalized)

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

    sep  = "=" * 72
    sep2 = "-" * 72

    _phase_mode_desc = {
        'both':     "both  (P and S residuals combined into a single y-axis value)",
        'p':        "p     (P-wave residuals only)",
        's':        "s     (S-wave residuals only)",
        'separate': "separate  (P and S plotted on separate axes)",
    }.get(phase_mode, phase_mode)

    _phase_plot_note = {
        'both':     ("P and S residuals pooled; X-axis is mean(Vp metric, Vs metric); "
                     "one curve plotted."),
        'p':        "P-wave residuals only; X-axis is Vp metric; one curve plotted.",
        's':        "S-wave residuals only; X-axis is Vs metric; one curve plotted.",
        'separate': ("P and S residuals computed independently; two curves plotted, "
                     "one per wave type."),
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
        lines.append(f"  Use log residuals    : {use_log_residuals}")
        if use_log_residuals:
            lines.append(f"  Log base             : {log_base}")
            lines.append(f"  Log epsilon          : {log_residual_epsilon}")

    exclude_zero_change    = mv_cfg.get('exclude_zero_change', False)
    zero_change_tolerance  = mv_cfg.get('zero_change_tolerance', 1e-6)

    lines += [
        f"  Region               : {region}",
        f"  Aggregation          : {aggregation}",
        f"  Use percentage change: {use_pct}",
        f"  Velocity dataset     : {velocity_dataset}",
    ]

    if aggregation == 'rms_variation':
        lines.append(f"  Exclude zero change  : {exclude_zero_change}")
        if exclude_zero_change:
            lines.append(f"  Zero change tol (km/s): {zero_change_tolerance}")
            lines.append(
                "  Note: n_used_vp / n_used_vs in the data table show how many "
                "grid points were included in each RMS computation after removing "
                "nodes with |Δv| <= tolerance."
            )

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
            lines.append(f"    Depth layer plot   : Z = {d} km (\u00b1{tol} km)  \u2192  "
                         f"tradeoff_curve_depth_{float(d):.2f}km.*")

    lines += [
        "",
        "CSV COLUMNS  (statistics CSV)",
        sep2,
    ]
    if phase_mode == 'separate':
        lines += [
            f"  iteration            : inversion iteration index",
            f"  {cols['x']:<28} : X-axis Vp metric  ({x_formula})",
            f"  {cols['x_vs']:<28} : X-axis Vs metric  ({x_formula})",
            f"  {cols['y_p']:<28} : Y-axis P-wave metric  ({y_formula})",
            f"  {cols['y_s']:<28} : Y-axis S-wave metric  ({y_formula})",
        ]
    else:
        x_phase = {'both': 'combined mean(Vp, Vs)', 'p': 'Vp', 's': 'Vs'}.get(phase_mode, 'X')
        lines += [
            f"  iteration            : inversion iteration index",
            f"  {cols['x']:<28} : X-axis metric  {x_phase}  ({x_formula})",
            f"  {cols['y']:<28} : Y-axis metric  ({y_formula})",
        ]

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
                df.to_string(index=False),
            ]
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
