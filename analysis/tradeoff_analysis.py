"""
Tradeoff analysis orchestration module.

Computes and visualizes the tradeoff between data fit (data misfit or RMS
traveltime residuals) and model variance (mean absolute velocity deviation
from the initial model), tracked across inversion iterations.

The tradeoff curve is useful in seismic tomography for understanding how
much the inverted model has departed from the starting model relative to
how well the data are fit (analogous to the L-curve in regularization).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.io_utils import ensure_directory_exists
from utils.plot_utils import setup_plot_style
from utils.logging_utils import ProcessTimer

from .tradeoff_data import (
    discover_iteration_dirs,
    load_data_misfit,
    load_initial_velocity_model,
    load_traveltime_residuals_for_iteration,
    compute_rms_for_iteration,
    load_velocity_model_for_iteration,
)
from .tradeoff_statistics import (
    validate_spatial_bounds,
    build_spatial_mask,
    build_depth_layer_mask,
    compute_model_variance_for_mask,
    assemble_tradeoff_table,
    detect_elbow,
)
from .tradeoff_plots import create_all_tradeoff_plots

logger = logging.getLogger(__name__)


def _parse_bound_pair(
    raw: Any,
) -> Tuple[Optional[float], Optional[float]]:
    """Parse a [lo, hi] config value, converting None/null to Python None."""
    if raw is None:
        return (None, None)
    lo = raw[0] if raw[0] is not None else None
    hi = raw[1] if raw[1] is not None else None
    return (lo, hi)


def _validate_grid_alignment(
    df_initial: pd.DataFrame,
    df_inverted: pd.DataFrame,
    iteration_idx: int,
) -> bool:
    """
    Check that the inverted model grid matches the initial model.

    Returns True if validation passes (possibly with warnings).
    Raises ValueError if row counts differ (hard failure).
    """
    if len(df_inverted) != len(df_initial):
        raise ValueError(
            f"Iteration {iteration_idx}: inverted model has {len(df_inverted)} rows "
            f"but initial model has {len(df_initial)} rows. "
            f"Grid mismatch – cannot compute variance."
        )

    # Soft coordinate drift check
    for col in ('X_grid_km', 'Y_grid_km', 'Z_grid_km'):
        if col not in df_inverted.columns:
            continue
        if not np.allclose(df_initial[col].values, df_inverted[col].values, atol=1e-4):
            logger.warning(
                f"Iteration {iteration_idx}: coordinate drift detected in {col} "
                f"between initial and inverted models."
            )

    return True


def analyze_tradeoff(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main tradeoff analysis function.

    Computes the data fit vs model variance tradeoff across inversion iterations
    and generates plots and statistics files.

    Args:
        config: Full configuration dictionary (from Hydra, converted to plain dict).

    Returns:
        Results dictionary with keys:
            'tradeoff_table': DataFrame (for region in {full, z_bounded})
            'by_depth': Dict[float, DataFrame] (for region == depth_layer)
            'config': module config sub-dict
    """
    tcfg = config['tradeoff_analysis']
    analysis_cfg = config.get('analysis', {})

    # ------------------------------------------------------------------ #
    # PHASE 0 – SETUP                                                     #
    # ------------------------------------------------------------------ #
    setup_plot_style(analysis_cfg)

    output_dir = Path(tcfg['output_dir'])
    plots_dir = output_dir / config.get('paths', {}).get('plots_subdir', 'plots')
    stats_dir = output_dir / config.get('paths', {}).get('stats_subdir', 'statistics')
    reports_dir = output_dir / config.get('paths', {}).get('reports_subdir', 'reports')

    for d in (plots_dir, stats_dir, reports_dir):
        ensure_directory_exists(d)

    y_axis_mode = tcfg.get('y_axis_mode', 'data_misfit')
    phase_mode = tcfg.get('phase_mode', 'both')
    mv_cfg = tcfg.get('model_variance', {})
    region = mv_cfg.get('region', 'full')
    velocity_dataset = tcfg.get('velocity_dataset', 'velocity')

    min_residual_threshold = tcfg.get('min_residual_threshold', -50.0)
    max_residual_threshold = tcfg.get('max_residual_threshold', 50.0)

    logger.info("=" * 60)
    logger.info("TRADEOFF ANALYSIS CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Y-axis mode:      {y_axis_mode}")
    if y_axis_mode == 'rms':
        logger.info(f"Phase mode:       {phase_mode}")
    logger.info(f"Region mode:      {region}")
    logger.info(f"Velocity dataset: {velocity_dataset}")
    logger.info(f"Output directory: {output_dir}")

    # ------------------------------------------------------------------ #
    # PHASE 1 – LOAD INITIAL MODEL                                        #
    # ------------------------------------------------------------------ #
    with ProcessTimer("Loading initial velocity model"):
        df_initial = load_initial_velocity_model(
            csv_path=tcfg.get('initial_model_csv'),
            hdf5_path=tcfg.get('initial_model_hdf5'),
            velocity_dataset=velocity_dataset,
        )

    # ------------------------------------------------------------------ #
    # PHASE 2 – VALIDATE BOUNDS                                           #
    # ------------------------------------------------------------------ #
    x_bounds = _parse_bound_pair(mv_cfg.get('x_bounds', [None, None]))
    y_bounds = _parse_bound_pair(mv_cfg.get('y_bounds', [None, None]))

    z_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    if region == 'z_bounded':
        z_bounds = _parse_bound_pair(mv_cfg.get('z_bounds', [None, None]))

    # validate_spatial_bounds raises ValueError on no-overlap
    validate_spatial_bounds(df_initial, x_bounds, y_bounds, z_bounds)

    # ------------------------------------------------------------------ #
    # PHASE 3 – PRECOMPUTE MASKS                                          #
    # ------------------------------------------------------------------ #
    depth_layers: List[float] = []
    masks: Dict[float, np.ndarray] = {}

    if region == 'full':
        mask_main = np.ones(len(df_initial), dtype=bool)
        # Still apply optional x/y bounds if specified
        if any(b is not None for b in (*x_bounds, *y_bounds)):
            mask_main = build_spatial_mask(df_initial, x_bounds, y_bounds, z_bounds=None)
        logger.info(f"Full-model mask: {mask_main.sum()} / {len(df_initial)} grid points")

    elif region == 'z_bounded':
        mask_main = build_spatial_mask(df_initial, x_bounds, y_bounds, z_bounds)
        logger.info(f"Z-bounded mask: {mask_main.sum()} / {len(df_initial)} grid points")

    elif region == 'depth_layer':
        raw_layers = mv_cfg.get('depth_layers', [])
        if not raw_layers:
            raise ValueError(
                "model_variance.region = 'depth_layer' requires at least one entry "
                "in depth_layers"
            )
        depth_layers = [float(d) for d in raw_layers]
        depth_tolerance = float(mv_cfg.get('depth_tolerance_km', 0.5))

        logger.info(f"Depth layers to analyze: {depth_layers} km "
                    f"(tolerance ±{depth_tolerance} km)")

        for depth_km in depth_layers:
            m = build_depth_layer_mask(
                df_initial, depth_km, depth_tolerance, x_bounds, y_bounds
            )
            if m.sum() == 0:
                logger.warning(
                    f"Depth layer Z={depth_km:.2f} km mask selected 0 points. "
                    "This depth will be skipped."
                )
            else:
                masks[depth_km] = m
                logger.info(
                    f"Depth layer Z={depth_km:.2f} km mask: {m.sum()} grid points"
                )

        if not masks:
            raise ValueError(
                "All specified depth layers selected 0 grid points. "
                "Check depth_layers values and depth_tolerance_km against the model."
            )
        mask_main = None  # not used in depth_layer mode
    else:
        raise ValueError(
            f"Unknown model_variance.region: '{region}'. "
            "Valid options: 'full', 'z_bounded', 'depth_layer'"
        )

    # ------------------------------------------------------------------ #
    # PHASE 4 – LOAD Y-AXIS DATA                                          #
    # ------------------------------------------------------------------ #
    y_dict: Dict[int, Any] = {}

    if y_axis_mode == 'data_misfit':
        with ProcessTimer("Loading data_misfit.txt"):
            df_misfit = load_data_misfit(
                filepath=tcfg['data_misfit_file_path'],
                use_normalized=tcfg.get('use_normalized_misfit', False),
            )
        for _, row in df_misfit.iterrows():
            y_dict[int(row['iteration'])] = float(row['y_value'])

    elif y_axis_mode == 'rms':
        with ProcessTimer("Loading traveltime residuals per iteration"):
            inversion_base = tcfg.get('inversion_base_path')
            if not inversion_base:
                raise ValueError(
                    "inversion_base_path must be set when y_axis_mode = 'rms'"
                )
            tt_iter_dirs = discover_iteration_dirs(inversion_base)

            if not tt_iter_dirs:
                raise ValueError(
                    f"No iteration_* directories found in inversion_base_path: "
                    f"{inversion_base}"
                )

            traveltime_pattern = tcfg.get(
                'traveltime_csv_pattern', 'travel_time_residuals*.csv'
            )

            for idx, iter_dir in tt_iter_dirs:
                df_tt = load_traveltime_residuals_for_iteration(
                    iter_dir, traveltime_pattern
                )
                if df_tt is None:
                    logger.warning(
                        f"Iteration {idx}: skipping – no traveltime CSV found"
                    )
                    continue
                rms_dict = compute_rms_for_iteration(
                    df_tt, phase_mode,
                    min_residual_threshold, max_residual_threshold,
                )
                y_dict[idx] = rms_dict
    else:
        raise ValueError(
            f"Unknown y_axis_mode: '{y_axis_mode}'. "
            "Valid options: 'data_misfit', 'rms'"
        )

    logger.info(f"Y-axis data loaded for {len(y_dict)} iterations")

    # ------------------------------------------------------------------ #
    # PHASE 5 – COMPUTE MODEL VARIANCE PER ITERATION                     #
    # ------------------------------------------------------------------ #
    inversion_analysis_dir = tcfg.get('inversion_analysis_dir')
    if not inversion_analysis_dir:
        raise ValueError("inversion_analysis_dir must be specified")

    with ProcessTimer("Discovering velocity model iteration directories"):
        vel_iter_dirs = discover_iteration_dirs(inversion_analysis_dir)

    if not vel_iter_dirs:
        raise ValueError(
            f"No iteration_* directories found in inversion_analysis_dir: "
            f"{inversion_analysis_dir}"
        )

    # Cross-check iteration sets
    y_iter_set = set(y_dict.keys())
    vel_iter_set = {i for i, _ in vel_iter_dirs}

    in_y_not_vel = y_iter_set - vel_iter_set
    in_vel_not_y = vel_iter_set - y_iter_set

    if in_y_not_vel:
        logger.warning(
            f"Iterations with Y-axis data but no velocity model directory: "
            f"{sorted(in_y_not_vel)} – excluded from tradeoff curve"
        )
    if in_vel_not_y:
        logger.warning(
            f"Iterations with velocity model directory but no Y-axis data: "
            f"{sorted(in_vel_not_y)} – excluded from tradeoff curve"
        )

    common_iterations = sorted(y_iter_set & vel_iter_set)
    logger.info(
        f"Common iterations to process: {len(common_iterations)} "
        f"({common_iterations[:5]}{'...' if len(common_iterations) > 5 else ''})"
    )

    # Build a lookup for velocity dirs
    vel_dir_map = {i: p for i, p in vel_iter_dirs}

    vp_variances: Dict[int, float] = {}
    vs_variances: Dict[int, float] = {}
    # Per-depth variant
    vp_variances_by_depth: Dict[float, Dict[int, float]] = {d: {} for d in depth_layers}
    vs_variances_by_depth: Dict[float, Dict[int, float]] = {d: {} for d in depth_layers}

    with ProcessTimer("Computing model variance per iteration"):
        for idx in common_iterations:
            iter_dir = vel_dir_map[idx]
            df_inverted = load_velocity_model_for_iteration(iter_dir, velocity_dataset)

            if df_inverted is None:
                logger.error(
                    f"Iteration {idx}: failed to load velocity model, skipping"
                )
                continue

            try:
                _validate_grid_alignment(df_initial, df_inverted, idx)
            except ValueError as e:
                logger.error(str(e))
                continue

            if region in ('full', 'z_bounded'):
                try:
                    var = compute_model_variance_for_mask(
                        df_initial, df_inverted, mask_main
                    )
                    vp_variances[idx] = var['vp']
                    vs_variances[idx] = var['vs']
                    logger.debug(
                        f"Iteration {idx}: vp_var={var['vp']:.6f}, "
                        f"vs_var={var['vs']:.6f} km/s"
                    )
                except ValueError as e:
                    logger.error(f"Iteration {idx}: {e}")
                    continue

            elif region == 'depth_layer':
                any_succeeded = False
                for depth_km, mask_d in masks.items():
                    try:
                        var_d = compute_model_variance_for_mask(
                            df_initial, df_inverted, mask_d
                        )
                        vp_variances_by_depth[depth_km][idx] = var_d['vp']
                        vs_variances_by_depth[depth_km][idx] = var_d['vs']
                        any_succeeded = True
                    except ValueError as e:
                        logger.error(
                            f"Iteration {idx}, depth {depth_km:.2f} km: {e}"
                        )
                if any_succeeded:
                    # Use max across depths for the combined variance store
                    # (needed for assembling the combined table if depth_layer)
                    vp_variances[idx] = max(
                        vp_variances_by_depth[d].get(idx, 0.0)
                        for d in depth_layers if d in masks
                    )
                    vs_variances[idx] = max(
                        vs_variances_by_depth[d].get(idx, 0.0)
                        for d in depth_layers if d in masks
                    )

    # ------------------------------------------------------------------ #
    # PHASE 6 – ASSEMBLE TRADEOFF TABLES                                  #
    # ------------------------------------------------------------------ #
    results: Dict[str, Any] = {'config': tcfg}

    all_indices = sorted(set(vp_variances.keys()) & set(vs_variances.keys()))

    if region in ('full', 'z_bounded'):
        df_tradeoff = assemble_tradeoff_table(
            iteration_indices=all_indices,
            y_values=y_dict,
            vp_variances=vp_variances,
            vs_variances=vs_variances,
            phase_mode=phase_mode,
            y_axis_mode=y_axis_mode,
        )
        results['tradeoff_table'] = df_tradeoff

    elif region == 'depth_layer':
        by_depth: Dict[float, pd.DataFrame] = {}
        for depth_km in sorted(masks.keys()):
            df_d = assemble_tradeoff_table(
                iteration_indices=sorted(vp_variances_by_depth[depth_km].keys()),
                y_values=y_dict,
                vp_variances=vp_variances_by_depth[depth_km],
                vs_variances=vs_variances_by_depth[depth_km],
                phase_mode=phase_mode,
                y_axis_mode=y_axis_mode,
            )
            by_depth[depth_km] = df_d
        results['by_depth'] = by_depth

    # ------------------------------------------------------------------ #
    # PHASE 7 – SAVE STATISTICS CSV                                       #
    # ------------------------------------------------------------------ #
    if region in ('full', 'z_bounded'):
        stats_file = stats_dir / "tradeoff_statistics.csv"
        results['tradeoff_table'].to_csv(stats_file, index=False)
        logger.info(f"Saved tradeoff statistics: {stats_file}")

    elif region == 'depth_layer':
        for depth_km, df_d in results.get('by_depth', {}).items():
            stats_file = stats_dir / f"tradeoff_statistics_depth_{depth_km:.2f}km.csv"
            df_d.to_csv(stats_file, index=False)
            logger.info(
                f"Saved depth-layer tradeoff statistics (Z={depth_km:.2f} km): "
                f"{stats_file}"
            )

    # ------------------------------------------------------------------ #
    # PHASE 8 – CREATE PLOTS                                              #
    # ------------------------------------------------------------------ #
    with ProcessTimer("Creating tradeoff plots"):
        create_all_tradeoff_plots(results, config, plots_dir)

    # ------------------------------------------------------------------ #
    # PHASE 9 – LOG SUMMARY                                               #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("TRADEOFF ANALYSIS SUMMARY")
    logger.info("=" * 60)

    if region in ('full', 'z_bounded'):
        df_out = results.get('tradeoff_table', pd.DataFrame())
        if not df_out.empty:
            logger.info(f"Iterations processed: {len(df_out)}")
            logger.info(
                f"Iteration range: [{df_out['iteration'].min()}, "
                f"{df_out['iteration'].max()}]"
            )
            logger.info(
                f"Model variance (max) range: "
                f"[{df_out['model_variance_max'].min():.6f}, "
                f"{df_out['model_variance_max'].max():.6f}] km/s"
            )

            if 'y_value' in df_out.columns:
                logger.info(
                    f"Y-axis ({y_axis_mode}) range: "
                    f"[{df_out['y_value'].min():.6e}, {df_out['y_value'].max():.6e}]"
                )

            if tcfg.get('plot', {}).get('highlight_optimal', False):
                x_col = 'model_variance_max'
                y_col = 'y_value' if 'y_value' in df_out.columns else 'rms_p'
                if x_col in df_out.columns and y_col in df_out.columns:
                    elbow_idx = detect_elbow(
                        df_out[x_col].values, df_out[y_col].values
                    )
                    elbow_iter = df_out['iteration'].iloc[elbow_idx]
                    logger.info(f"Optimal (elbow) iteration: {elbow_iter}")

    elif region == 'depth_layer':
        for depth_km, df_d in results.get('by_depth', {}).items():
            if not df_d.empty:
                logger.info(
                    f"Z={depth_km:.2f} km: {len(df_d)} iterations, "
                    f"model_variance_max range "
                    f"[{df_d['model_variance_max'].min():.6f}, "
                    f"{df_d['model_variance_max'].max():.6f}] km/s"
                )

    return results
