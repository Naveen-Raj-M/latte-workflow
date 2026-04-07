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

from tqdm import tqdm

from utils.io_utils import ensure_directory_exists
from utils.plot_utils import setup_plot_style
from utils.logging_utils import ProcessTimer

from .tradeoff_data import (
    discover_iteration_dirs,
    build_parameter_entry_path,
    load_data_misfit,
    load_initial_velocity_model,
    load_traveltime_residuals_for_iteration,
    compute_rms_for_iteration,
    load_velocity_model_for_iteration,
    load_static_method_csv,
    compute_method_comparison_rms_for_iteration,
)
from .tradeoff_statistics import (
    validate_spatial_bounds,
    build_spatial_mask,
    build_depth_layer_mask,
    compute_model_variance_for_mask,
    assemble_tradeoff_table,
    assemble_parameter_tradeoff_table,
    write_parameter_report,
    write_general_report,
    detect_elbow,
)
from .tradeoff_plots import create_all_tradeoff_plots, create_parameter_tradeoff_plot

logger = logging.getLogger(__name__)


def _process_parameter_entries(
    entries_cfg: List[Dict[str, Any]],
    phase: str,
    inversion_base_path: str,
    inversion_analysis_dir: str,
    traveltime_csv_pattern: str,
    df_initial: pd.DataFrame,
    mask_main: np.ndarray,
    use_percentage_change: bool,
    aggregation: str,
    velocity_dataset: str,
    min_residual_threshold: float,
    max_residual_threshold: float,
    stat_mode: str = 'rms',
) -> List[Dict[str, Any]]:
    """
    Process parameter sweep entries: load traveltime CSVs and velocity models,
    compute y-axis statistic and model variance per entry.

    Args:
        entries_cfg: List of dicts from config, each with keys
            parameter_value, directory_name, iteration.
        phase: Phase mode for y-axis computation ('p' or 's').
        inversion_base_path: Root dir for traveltime CSV subdirs.
        inversion_analysis_dir: Root dir for velocity model subdirs.
        traveltime_csv_pattern: Glob pattern for traveltime CSV files.
        df_initial: Initial velocity model DataFrame.
        mask_main: Spatial mask for model variance computation.
        use_percentage_change: Compute variance as percentage if True.
        aggregation: 'mean', 'max', or 'model_variance' aggregation for variance.
        velocity_dataset: HDF5 dataset name for velocity models.
        min_residual_threshold: Lower bound for y-axis computation (s).
        max_residual_threshold: Upper bound for y-axis computation (s).
        stat_mode: 'rms' or 'data_variance' — controls y-axis statistic.

    Returns:
        List of processed result dicts with keys:
            parameter_value, directory_name, iteration,
            model_variance_vp, model_variance_vs, model_variance_max, rms.
    """
    processed = []

    for entry_cfg in entries_cfg:
        param_val = float(entry_cfg['parameter_value'])
        dir_name = str(entry_cfg['directory_name'])
        iteration = int(entry_cfg['iteration'])

        result: Dict[str, Any] = {
            'parameter_value':    param_val,
            'directory_name':     dir_name,
            'iteration':          iteration,
            'model_variance_vp':  None,
            'model_variance_vs':  None,
            'model_variance_max': None,
            'rms':                None,
        }

        # Load traveltime CSV and compute RMS
        try:
            tt_dir = build_parameter_entry_path(inversion_base_path, dir_name, iteration)
            df_tt = load_traveltime_residuals_for_iteration(tt_dir, traveltime_csv_pattern)
            if df_tt is not None:
                rms_dict = compute_rms_for_iteration(
                    df_tt, phase,
                    min_residual_threshold, max_residual_threshold,
                    stat_mode=stat_mode,
                )
                key = next(iter(rms_dict))
                result['rms'] = rms_dict.get(key)
        except (FileNotFoundError, NotADirectoryError) as e:
            logger.warning(
                f"Parameter entry (val={param_val}, dir={dir_name}, iter={iteration}): "
                f"traveltime dir – {e}"
            )

        # Load velocity model and compute model variance
        try:
            vel_dir = build_parameter_entry_path(inversion_analysis_dir, dir_name, iteration)
            df_inverted = load_velocity_model_for_iteration(vel_dir, velocity_dataset)
            if df_inverted is not None:
                try:
                    _validate_grid_alignment(df_initial, df_inverted, iteration)
                    var = compute_model_variance_for_mask(
                        df_initial, df_inverted, mask_main,
                        use_percentage_change=use_percentage_change,
                        aggregation=aggregation,
                    )
                    result['model_variance_vp'] = var['vp']
                    result['model_variance_vs'] = var['vs']
                    result['model_variance_max'] = var['max']
                    units = "%" if use_percentage_change else "km/s"
                    logger.debug(
                        f"Entry (val={param_val}): "
                        f"vp_var={var['vp']:.6f}, vs_var={var['vs']:.6f} {units}"
                    )
                except ValueError as e:
                    logger.error(
                        f"Parameter entry (val={param_val}, dir={dir_name}): {e}"
                    )
        except (FileNotFoundError, NotADirectoryError) as e:
            logger.warning(
                f"Parameter entry (val={param_val}, dir={dir_name}, iter={iteration}): "
                f"velocity dir – {e}"
            )

        processed.append(result)

    return processed


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

    write_report = bool(tcfg.get('write_report', True))
    y_axis_mode = tcfg.get('y_axis_mode', 'data_misfit')
    phase_mode = tcfg.get('phase_mode', 'both')
    rms_comparison_mode = tcfg.get('rms_comparison_mode', 'ground_truth')
    mv_cfg = tcfg.get('model_variance', {})
    region = mv_cfg.get('region', 'full')
    velocity_dataset = tcfg.get('velocity_dataset', 'velocity')

    min_residual_threshold = tcfg.get('min_residual_threshold', -50.0)
    max_residual_threshold = tcfg.get('max_residual_threshold', 50.0)
    use_percentage_change = bool(mv_cfg.get('use_percentage_change', False))
    aggregation = mv_cfg.get('aggregation', 'mean')
    if aggregation not in ('mean', 'max', 'model_variance', 'rms_variation'):
        raise ValueError(
            f"Invalid model_variance.aggregation: '{aggregation}'. "
            "Valid options: 'mean', 'max', 'model_variance', 'rms_variation'"
        )

    logger.info("=" * 60)
    logger.info("TRADEOFF ANALYSIS CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Y-axis mode:      {y_axis_mode}")
    if y_axis_mode == 'rms':
        logger.info(f"Phase mode:       {phase_mode}")
        logger.info(f"RMS comparison:   {rms_comparison_mode}")
    logger.info(f"Region mode:      {region}")
    logger.info(f"Aggregation:      {aggregation}")
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
    # PARAMETER MODE – process parameter sweep and return early           #
    # ------------------------------------------------------------------ #
    x_axis_mode = tcfg.get('x_axis_mode', 'iteration')

    if x_axis_mode not in ('iteration', 'parameter'):
        raise ValueError(
            f"Unknown x_axis_mode: '{x_axis_mode}'. "
            "Valid options: 'iteration', 'parameter'"
        )

    if x_axis_mode == 'parameter':
        if region == 'depth_layer':
            raise ValueError(
                "x_axis_mode='parameter' does not support "
                "model_variance.region='depth_layer'. Use 'full' or 'z_bounded'."
            )

        pm_cfg = tcfg.get('parameter_mode', {})
        parameter_name = pm_cfg.get('parameter_name', 'parameter')
        p_entries_cfg = pm_cfg.get('p_wave_entries') or []
        s_entries_cfg = pm_cfg.get('s_wave_entries') or []

        if not p_entries_cfg and not s_entries_cfg:
            raise ValueError(
                "x_axis_mode='parameter' requires at least one entry in "
                "parameter_mode.p_wave_entries or parameter_mode.s_wave_entries"
            )

        inversion_base_path = tcfg.get('inversion_base_path')
        if not inversion_base_path:
            raise ValueError(
                "inversion_base_path must be set when x_axis_mode='parameter'"
            )

        inversion_analysis_dir = tcfg.get('inversion_analysis_dir')
        if not inversion_analysis_dir:
            raise ValueError(
                "inversion_analysis_dir must be set when x_axis_mode='parameter'"
            )

        traveltime_pattern = tcfg.get(
            'traveltime_csv_pattern', 'travel_time_residuals*.csv'
        )

        logger.info("=" * 60)
        logger.info("PARAMETER SWEEP MODE")
        logger.info("=" * 60)
        logger.info(f"Parameter:        {parameter_name}")
        logger.info(f"P-wave entries:   {len(p_entries_cfg)}")
        logger.info(f"S-wave entries:   {len(s_entries_cfg)}")

        param_stat_mode = 'data_variance' if y_axis_mode == 'data_variance' else 'rms'

        df_p: Optional[pd.DataFrame] = None
        df_s: Optional[pd.DataFrame] = None

        if p_entries_cfg:
            with ProcessTimer("Processing P-wave parameter entries"):
                p_processed = _process_parameter_entries(
                    entries_cfg=p_entries_cfg,
                    phase='p',
                    inversion_base_path=inversion_base_path,
                    inversion_analysis_dir=inversion_analysis_dir,
                    traveltime_csv_pattern=traveltime_pattern,
                    df_initial=df_initial,
                    mask_main=mask_main,
                    use_percentage_change=use_percentage_change,
                    aggregation=aggregation,
                    velocity_dataset=velocity_dataset,
                    min_residual_threshold=min_residual_threshold,
                    max_residual_threshold=max_residual_threshold,
                    stat_mode=param_stat_mode,
                )
            df_p = assemble_parameter_tradeoff_table(p_processed)

        if s_entries_cfg:
            with ProcessTimer("Processing S-wave parameter entries"):
                s_processed = _process_parameter_entries(
                    entries_cfg=s_entries_cfg,
                    phase='s',
                    inversion_base_path=inversion_base_path,
                    inversion_analysis_dir=inversion_analysis_dir,
                    traveltime_csv_pattern=traveltime_pattern,
                    df_initial=df_initial,
                    mask_main=mask_main,
                    use_percentage_change=use_percentage_change,
                    aggregation=aggregation,
                    velocity_dataset=velocity_dataset,
                    min_residual_threshold=min_residual_threshold,
                    max_residual_threshold=max_residual_threshold,
                    stat_mode=param_stat_mode,
                )
            df_s = assemble_parameter_tradeoff_table(s_processed)

        # Save statistics CSV
        if df_p is not None and not df_p.empty:
            p_stats_file = stats_dir / f"parameter_tradeoff_{parameter_name}_p.csv"
            df_p.to_csv(p_stats_file, index=False)
            logger.info(f"Saved P-wave parameter statistics: {p_stats_file}")

        if df_s is not None and not df_s.empty:
            s_stats_file = stats_dir / f"parameter_tradeoff_{parameter_name}_s.csv"
            df_s.to_csv(s_stats_file, index=False)
            logger.info(f"Saved S-wave parameter statistics: {s_stats_file}")

        # Write text report
        if write_report:
            report_file = reports_dir / f"parameter_tradeoff_{parameter_name}.txt"
            write_parameter_report(
                parameter_name=parameter_name,
                p_table=df_p,
                s_table=df_s,
                report_path=report_file,
            )

        # Create plots
        with ProcessTimer("Creating parameter tradeoff plot"):
            output_format = analysis_cfg.get('output_format', 'png')
            plot_path = plots_dir / f"parameter_tradeoff_{parameter_name}.{output_format}"
            create_parameter_tradeoff_plot(
                df_p=df_p,
                df_s=df_s,
                parameter_name=parameter_name,
                config=config,
                output_path=plot_path,
            )

        return {
            'config': tcfg,
            'parameter_table_p': df_p,
            'parameter_table_s': df_s,
        }

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

    elif y_axis_mode in ('rms', 'data_variance'):
        stat_mode = 'data_variance' if y_axis_mode == 'data_variance' else 'rms'
        timer_label = (
            "Loading traveltime residuals per iteration (data variance)"
            if y_axis_mode == 'data_variance'
            else "Loading traveltime residuals per iteration"
        )
        with ProcessTimer(timer_label):
            inversion_base = tcfg.get('inversion_base_path')
            if not inversion_base:
                raise ValueError(
                    f"inversion_base_path must be set when y_axis_mode = '{y_axis_mode}'"
                )
            tt_iter_dirs = discover_iteration_dirs(inversion_base)

            if not tt_iter_dirs:
                raise ValueError(
                    f"No iteration_* directories found in inversion_base_path: "
                    f"{inversion_base}"
                )

            if rms_comparison_mode == 'method_comparison':
                method1_csv = tcfg.get('method1_csv')
                if not method1_csv:
                    raise ValueError(
                        "method1_csv must be specified when "
                        "rms_comparison_mode = 'method_comparison'"
                    )
                df_method1 = load_static_method_csv(method1_csv)
                method2_pattern = tcfg.get(
                    'method2_csv_pattern', 'travel_time_residuals*.csv'
                )
                logger.info(f"Method 2 pattern (per iteration): {method2_pattern}")

                for idx, iter_dir in tt_iter_dirs:
                    rms_dict = compute_method_comparison_rms_for_iteration(
                        df_method1, iter_dir, method2_pattern,
                        phase_mode, min_residual_threshold, max_residual_threshold,
                        stat_mode=stat_mode,
                    )
                    if rms_dict is None:
                        logger.warning(
                            f"Iteration {idx}: skipping – method comparison failed"
                        )
                        continue
                    y_dict[idx] = rms_dict

            else:
                # ground_truth: residuals are pre-computed in the per-iteration CSV
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
                        stat_mode=stat_mode,
                    )
                    y_dict[idx] = rms_dict
    else:
        raise ValueError(
            f"Unknown y_axis_mode: '{y_axis_mode}'. "
            "Valid options: 'data_misfit', 'rms', 'data_variance'"
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
        for idx in tqdm(common_iterations, desc="Model variance", unit="iter"):
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
                        df_initial, df_inverted, mask_main,
                        use_percentage_change=use_percentage_change,
                        aggregation=aggregation,
                    )
                    vp_variances[idx] = var['vp']
                    vs_variances[idx] = var['vs']
                    units = "%" if use_percentage_change else "km/s"
                    logger.debug(
                        f"Iteration {idx}: vp_var={var['vp']:.6f}, "
                        f"vs_var={var['vs']:.6f} {units}"
                    )
                except ValueError as e:
                    logger.error(f"Iteration {idx}: {e}")
                    continue

            elif region == 'depth_layer':
                any_succeeded = False
                for depth_km, mask_d in masks.items():
                    try:
                        var_d = compute_model_variance_for_mask(
                            df_initial, df_inverted, mask_d,
                            use_percentage_change=use_percentage_change,
                            aggregation=aggregation,
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
                    f"{df_d['model_variance_max'].max():.6f}]"
                )

    # ------------------------------------------------------------------ #
    # PHASE 10 – WRITE REPORT                                             #
    # ------------------------------------------------------------------ #
    if write_report:
        report_file = reports_dir / "tradeoff_analysis_report.txt"
        write_general_report(config, results, report_file)

    return results
