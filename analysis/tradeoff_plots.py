"""
Plotting functions for tradeoff analysis.

Creates tradeoff curve plots (single or per-depth-layer), with support
for iteration labels, colormap coloring, log scales, and elbow annotation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from utils.io_utils import ensure_directory_exists
from utils.plot_utils import save_figure, get_color_scheme

from .tradeoff_statistics import detect_elbow

logger = logging.getLogger(__name__)


def _resolve_colors(
    plot_cfg: Dict[str, Any],
    global_colors: Dict[str, str],
) -> Dict[str, str]:
    """
    Resolve vp_color and vs_color with fallback to global plot_style colors.

    Args:
        plot_cfg: tradeoff_analysis.plot sub-config dict.
        global_colors: Colors from analysis.plot_style.colors.

    Returns:
        Dict with keys 'vp' and 'vs' (hex color strings).
    """
    vp_color = plot_cfg.get('vp_color') or global_colors.get('p_wave_dark', '#1f4e79')
    vs_color = plot_cfg.get('vs_color') or global_colors.get('s_wave_dark', '#2e7d32')
    return {'vp': vp_color, 'vs': vs_color}


def _auto_y_label(y_axis_mode: str, phase_mode: str, use_normalized: bool) -> str:
    """Derive automatic Y-axis label from mode settings."""
    if y_axis_mode == 'data_misfit':
        return "Normalized Data Misfit" if use_normalized else "Data Misfit"
    # rms mode
    labels = {
        'both': "Combined RMS Residual (s)",
        'p': "P-wave RMS Residual (s)",
        's': "S-wave RMS Residual (s)",
        'separate': "RMS Residual (s)",
    }
    return labels.get(phase_mode, "RMS Residual (s)")


def create_tradeoff_plot(
    df: pd.DataFrame,
    config: Dict[str, Any],
    output_path: Path,
    title: str = "Tradeoff Curve",
    depth_label: Optional[str] = None,
) -> None:
    """
    Create and save a single tradeoff plot.

    Args:
        df: Tradeoff statistics DataFrame (from assemble_tradeoff_table).
        config: Full configuration dictionary.
        output_path: Full path including filename (extension added if missing).
        title: Plot title.
        depth_label: Optional string like "Z = 15.0 km" for depth_layer mode.
    """
    if df.empty:
        logger.warning(f"Tradeoff table is empty – skipping plot: {output_path}")
        return

    tcfg = config.get('tradeoff_analysis', {})
    plot_cfg = tcfg.get('plot', {})
    analysis_cfg = config.get('analysis', {})

    output_format = analysis_cfg.get('output_format', 'png')
    dpi = analysis_cfg.get('dpi', 300)
    plot_style = analysis_cfg.get('plot_style', {})

    y_axis_mode = tcfg.get('y_axis_mode', 'data_misfit')
    phase_mode = tcfg.get('phase_mode', 'both')
    use_normalized = tcfg.get('use_normalized_misfit', False)

    global_colors = get_color_scheme(analysis_cfg)
    colors = _resolve_colors(plot_cfg, global_colors)

    fig_width = plot_cfg.get('figure_width', 10)
    fig_height = plot_cfg.get('figure_height', 8)
    marker_style = plot_cfg.get('marker_style', 'o')
    marker_size = plot_style.get('marker_size', 6)
    line_width = plot_style.get('line_width', 2)
    show_grid = plot_cfg.get('show_grid', True)
    show_iter_labels = plot_cfg.get('show_iteration_labels', True)
    use_log_x = plot_cfg.get('use_log_scale_x', False)
    use_log_y = plot_cfg.get('use_log_scale_y', False)
    color_by_iter = plot_cfg.get('color_by_iteration', False)
    colormap_name = plot_cfg.get('colormap', 'viridis')
    highlight_optimal = plot_cfg.get('highlight_optimal', False)

    use_pct = bool(tcfg.get('model_variance', {}).get('use_percentage_change', False))
    default_x_label = "Model Variance (%)" if use_pct else "Model Variance (km/s)"
    x_label = plot_cfg.get('x_label') or default_x_label
    y_label = plot_cfg.get('y_label') or _auto_y_label(y_axis_mode, phase_mode, use_normalized)

    iterations = df['iteration'].values

    if phase_mode == 'separate' and 'rms_p' in df.columns and 'rms_s' in df.columns:
        # ── Separate subplots: P (left) and S (right) ──────────────────────
        fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

        for ax, x_col, y_col, color, phase_label in [
            (ax_p, 'model_variance_vp', 'rms_p', colors['vp'], 'P-wave'),
            (ax_s, 'model_variance_vs', 'rms_s', colors['vs'], 'S-wave'),
        ]:
            x_vals = df[x_col].values
            y_vals = df[y_col].values

            if color_by_iter:
                _plot_colormap_curve(
                    ax, x_vals, y_vals, iterations, marker_style, marker_size,
                    line_width, colormap_name, plot_style,
                )
            else:
                ax.plot(
                    x_vals, y_vals,
                    marker=marker_style,
                    markersize=marker_size,
                    linewidth=line_width,
                    color=color,
                    alpha=plot_style.get('alpha', 0.85),
                    label=phase_label,
                )

            if show_iter_labels:
                _annotate_iterations(ax, x_vals, y_vals, iterations, plot_style)

            if highlight_optimal and len(x_vals) >= 3:
                elbow_idx = detect_elbow(x_vals, y_vals)
                ax.plot(
                    x_vals[elbow_idx], y_vals[elbow_idx],
                    marker='*',
                    markersize=marker_size * 3,
                    color='gold',
                    markeredgecolor='black',
                    markeredgewidth=0.8,
                    zorder=5,
                    label=f"Optimal (iter {iterations[elbow_idx]})",
                )

            ax.legend(fontsize=plot_style.get('legend_size', 12))

            if use_log_x:
                ax.set_xscale('log')
            if use_log_y:
                ax.set_yscale('log')

            ax.set_xlabel(x_label, fontsize=plot_style.get('label_size', 14))
            ax.set_ylabel(
                f"{phase_label} RMS Residual (s)",
                fontsize=plot_style.get('label_size', 14),
            )

            if show_grid:
                ax.grid(
                    True,
                    alpha=plot_style.get('grid_alpha', 0.3),
                    color=global_colors.get('grid_color', '#cccccc'),
                )

            if depth_label:
                ax.text(
                    0.98, 0.98, depth_label,
                    transform=ax.transAxes,
                    ha='right', va='top',
                    fontsize=plot_style.get('legend_size', 12),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                )

        fig.suptitle(title, fontsize=plot_style.get('title_size', 16))

    else:
        # ── Single axes ─────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Single curve
        x_col = 'model_variance_max'
        y_col = 'y_value'
        x_vals = df[x_col].values
        y_vals = df[y_col].values

        if color_by_iter:
            _plot_colormap_curve(
                ax, x_vals, y_vals, iterations, marker_style, marker_size,
                line_width, colormap_name, plot_style,
            )
        else:
            ax.plot(
                x_vals, y_vals,
                marker=marker_style,
                markersize=marker_size,
                linewidth=line_width,
                color=colors['vp'],
                alpha=plot_style.get('alpha', 0.85),
            )

        if show_iter_labels:
            _annotate_iterations(ax, x_vals, y_vals, iterations, plot_style)

        if highlight_optimal and len(x_vals) >= 3:
            elbow_idx = detect_elbow(x_vals, y_vals)
            ax.plot(
                x_vals[elbow_idx], y_vals[elbow_idx],
                marker='*',
                markersize=marker_size * 3,
                color='gold',
                markeredgecolor='black',
                markeredgewidth=0.8,
                zorder=5,
                label=f"Optimal (iter {iterations[elbow_idx]})",
            )
            ax.legend(fontsize=plot_style.get('legend_size', 12))

        # Axis configuration
        if use_log_x:
            ax.set_xscale('log')
        if use_log_y:
            ax.set_yscale('log')

        ax.set_xlabel(x_label, fontsize=plot_style.get('label_size', 14))
        ax.set_ylabel(y_label, fontsize=plot_style.get('label_size', 14))
        ax.set_title(title, fontsize=plot_style.get('title_size', 16))

        if show_grid:
            ax.grid(True, alpha=plot_style.get('grid_alpha', 0.3),
                    color=global_colors.get('grid_color', '#cccccc'))

        # Depth label text box
        if depth_label:
            ax.text(
                0.98, 0.98, depth_label,
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=plot_style.get('legend_size', 12),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            )

    fig.tight_layout()

    # Ensure output path has extension
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(f'.{output_format}')

    save_figure(fig, output_path, output_format=output_format, dpi=dpi)
    logger.info(f"Saved tradeoff plot: {output_path}")



def _plot_colormap_curve(
    ax: plt.Axes,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    iterations: np.ndarray,
    marker_style: str,
    marker_size: float,
    line_width: float,
    colormap_name: str,
    plot_style: Dict[str, Any],
) -> None:
    """Plot a curve colored by iteration number using a colormap."""
    cmap = cm.get_cmap(colormap_name)
    norm = mcolors.Normalize(vmin=iterations.min(), vmax=iterations.max())

    # Draw connecting lines in gray first
    ax.plot(x_vals, y_vals, color='lightgray', linewidth=line_width * 0.8, zorder=1)

    # Then colored scatter points on top
    sc = ax.scatter(
        x_vals, y_vals,
        c=iterations, cmap=cmap, norm=norm,
        s=marker_size ** 2 * 4,
        marker=marker_style,
        zorder=2,
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Iteration', fontsize=plot_style.get('legend_size', 12))


def _annotate_iterations(
    ax: plt.Axes,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    iterations: np.ndarray,
    plot_style: Dict[str, Any],
    color: str = 'black',
) -> None:
    """Annotate each data point with its iteration number."""
    fontsize = max(6, plot_style.get('tick_size', 10) - 2)
    for x, y, it in zip(x_vals, y_vals, iterations):
        ax.annotate(
            str(int(it)),
            (x, y),
            textcoords='offset points',
            xytext=(4, 4),
            fontsize=fontsize,
            color=color,
            ha='left',
        )


def _annotate_parameter_values(
    ax: plt.Axes,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    param_vals: np.ndarray,
    plot_style: Dict[str, Any],
    color: str = 'black',
) -> None:
    """Annotate each data point with its parameter value."""
    fontsize = max(6, plot_style.get('tick_size', 10) - 2)
    for x, y, pv in zip(x_vals, y_vals, param_vals):
        label = f"{pv:g}" if isinstance(pv, (int, float)) else str(pv)
        ax.annotate(
            label,
            (x, y),
            textcoords='offset points',
            xytext=(4, 4),
            fontsize=fontsize,
            color=color,
            ha='left',
        )


def create_parameter_tradeoff_plot(
    df_p: Optional[pd.DataFrame],
    df_s: Optional[pd.DataFrame],
    parameter_name: str,
    config: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Create and save a parameter sweep tradeoff plot.

    X-axis: model variance (Vp for P-wave, Vs for S-wave).
    Y-axis: RMS traveltime residual.
    Points are annotated with their parameter value.

    Args:
        df_p: P-wave parameter tradeoff table (or None / empty).
        df_s: S-wave parameter tradeoff table (or None / empty).
        parameter_name: Name of the swept parameter used in labels.
        config: Full configuration dictionary.
        output_path: Full path including filename.
    """
    has_p = df_p is not None and not df_p.empty and df_p['rms'].notna().any()
    has_s = df_s is not None and not df_s.empty and df_s['rms'].notna().any()

    if not has_p and not has_s:
        logger.warning("No parameter tradeoff data available for plotting")
        return

    tcfg = config.get('tradeoff_analysis', {})
    plot_cfg = tcfg.get('plot', {})
    analysis_cfg = config.get('analysis', {})

    output_format = analysis_cfg.get('output_format', 'png')
    dpi = analysis_cfg.get('dpi', 300)
    plot_style = analysis_cfg.get('plot_style', {})

    global_colors = get_color_scheme(analysis_cfg)
    colors = _resolve_colors(plot_cfg, global_colors)

    fig_width = plot_cfg.get('figure_width', 10)
    fig_height = plot_cfg.get('figure_height', 8)
    marker_style = plot_cfg.get('marker_style', 'o')
    marker_size = plot_style.get('marker_size', 6)
    line_width = plot_style.get('line_width', 2)
    show_grid = plot_cfg.get('show_grid', True)
    show_param_labels = plot_cfg.get('show_iteration_labels', True)
    use_log_x = plot_cfg.get('use_log_scale_x', False)
    use_log_y = plot_cfg.get('use_log_scale_y', False)

    use_pct = bool(tcfg.get('model_variance', {}).get('use_percentage_change', False))
    units = "%" if use_pct else "km/s"

    def _draw_curve(
        ax: plt.Axes,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        param_vals: np.ndarray,
        color: str,
        phase_label: str,
        x_label_str: str,
        y_label_str: str,
    ) -> None:
        ax.plot(
            x_vals, y_vals,
            marker=marker_style,
            markersize=marker_size,
            linewidth=line_width,
            color=color,
            alpha=plot_style.get('alpha', 0.85),
            label=phase_label,
        )
        if show_param_labels:
            _annotate_parameter_values(ax, x_vals, y_vals, param_vals, plot_style)
        if use_log_x:
            ax.set_xscale('log')
        if use_log_y:
            ax.set_yscale('log')
        ax.set_xlabel(x_label_str, fontsize=plot_style.get('label_size', 14))
        ax.set_ylabel(y_label_str, fontsize=plot_style.get('label_size', 14))
        if show_grid:
            ax.grid(
                True,
                alpha=plot_style.get('grid_alpha', 0.3),
                color=global_colors.get('grid_color', '#cccccc'),
            )
        ax.legend(fontsize=plot_style.get('legend_size', 12))

    title = f"Parameter Tradeoff: {parameter_name}"

    if has_p and has_s:
        fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        _draw_curve(
            ax_p,
            df_p['model_variance_vp'].values,
            df_p['rms'].values,
            df_p['parameter_value'].values,
            colors['vp'], 'P-wave',
            f"Vp Model Variance ({units})",
            "P-wave RMS Residual (s)",
        )
        _draw_curve(
            ax_s,
            df_s['model_variance_vs'].values,
            df_s['rms'].values,
            df_s['parameter_value'].values,
            colors['vs'], 'S-wave',
            f"Vs Model Variance ({units})",
            "S-wave RMS Residual (s)",
        )
        fig.suptitle(title, fontsize=plot_style.get('title_size', 16))
    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        if has_p:
            _draw_curve(
                ax,
                df_p['model_variance_vp'].values,
                df_p['rms'].values,
                df_p['parameter_value'].values,
                colors['vp'], 'P-wave',
                f"Vp Model Variance ({units})",
                "P-wave RMS Residual (s)",
            )
        else:
            _draw_curve(
                ax,
                df_s['model_variance_vs'].values,
                df_s['rms'].values,
                df_s['parameter_value'].values,
                colors['vs'], 'S-wave',
                f"Vs Model Variance ({units})",
                "S-wave RMS Residual (s)",
            )
        ax.set_title(title, fontsize=plot_style.get('title_size', 16))

    fig.tight_layout()

    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(f'.{output_format}')

    save_figure(fig, output_path, output_format=output_format, dpi=dpi)
    logger.info(f"Saved parameter tradeoff plot: {output_path}")


def create_all_tradeoff_plots(
    tradeoff_results: Dict[str, Any],
    config: Dict[str, Any],
    plots_dir: Path,
) -> List[Path]:
    """
    Create all tradeoff plots based on region mode and depth layers.

    For region in {full, z_bounded}: one plot saved as tradeoff_curve.{fmt}
    For region == depth_layer: one plot per depth saved as
        tradeoff_curve_depth_{d:.2f}km.{fmt}

    Args:
        tradeoff_results: Dict returned by the main orchestration loop.
            Keys: 'tradeoff_table' (for full/z_bounded) OR
                  'by_depth' (dict of depth_km → DataFrame)
            Plus 'config' (the module sub-config).
        config: Full configuration dictionary.
        plots_dir: Directory where plots are saved.

    Returns:
        List of Path objects for all saved plots.
    """
    ensure_directory_exists(plots_dir)

    tcfg = config.get('tradeoff_analysis', {})
    analysis_cfg = config.get('analysis', {})
    output_format = analysis_cfg.get('output_format', 'png')
    region = tcfg.get('model_variance', {}).get('region', 'full')

    saved_paths = []

    if region in ('full', 'z_bounded'):
        df = tradeoff_results.get('tradeoff_table')
        if df is None or df.empty:
            logger.warning("No tradeoff data available for plotting")
            return saved_paths

        region_label = "Full Model" if region == 'full' else "Z-bounded Region"
        title = f"Data Misfit vs Model Variance Tradeoff\n({region_label})"
        output_path = plots_dir / f"tradeoff_curve.{output_format}"
        create_tradeoff_plot(df, config, output_path, title=title)
        saved_paths.append(output_path)

    elif region == 'depth_layer':
        by_depth = tradeoff_results.get('by_depth', {})
        if not by_depth:
            logger.warning("No per-depth tradeoff data available for plotting")
            return saved_paths

        for depth_km, df in sorted(by_depth.items()):
            if df is None or df.empty:
                logger.warning(f"No data for depth layer Z={depth_km:.2f} km, skipping plot")
                continue

            depth_label = f"Z = {depth_km:.2f} km"
            title = f"Data Misfit vs Model Variance Tradeoff\n({depth_label})"
            fname = f"tradeoff_curve_depth_{depth_km:.2f}km.{output_format}"
            output_path = plots_dir / fname
            create_tradeoff_plot(
                df, config, output_path,
                title=title, depth_label=depth_label,
            )
            saved_paths.append(output_path)

    logger.info(f"Created {len(saved_paths)} tradeoff plot(s) in {plots_dir}")
    return saved_paths
