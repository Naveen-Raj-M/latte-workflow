"""
Distance analysis plotting module.

This module creates all distance analysis visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, Any

from utils.io_utils import ensure_directory_exists
from utils.plot_utils import (
    get_color_scheme, save_figure, create_multi_panel_figure,
    subsample_data_for_plotting
)
from utils.distance_utils import (
    get_distance_column_name, get_distance_label, get_output_suffix
)

logger = logging.getLogger(__name__)


def create_distance_analysis_plots(df: pd.DataFrame, stats: Dict[str, Any], 
                                  config: Dict[str, Any], distance_type: str) -> None:
    """
    Create comprehensive distance analysis plots.
    
    Args:
        df: Merged dataframe with distances
        stats: Statistics dictionary
        config: Configuration dictionary
        distance_type: "hypocentral" or "epicentral"
    """
    output_dir = Path(config['distance_analysis']['output_dir'])
    coordinate_type = config['distance_analysis']['coordinate_type']
    comparison_mode = config['distance_analysis'].get('comparison_mode', 'ground_truth')
    
    # Create subdirectory for distance type
    suffix = get_output_suffix(distance_type)
    mode_suffix = 'comparison' if comparison_mode == 'method_comparison' else 'ground_truth'
    plots_dir = output_dir / 'plots' / f'{suffix}_{coordinate_type}_{mode_suffix}'
    ensure_directory_exists(plots_dir)
    
    colors = get_color_scheme(config['analysis'])
    output_format = config['analysis'].get('output_format', 'png')
    
    # Subsample data for plotting if needed
    max_points = config['analysis']['plot_style'].get('max_plot_points', 50000)
    df_plot = subsample_data_for_plotting(df, max_points)
    
    logger.info(f"Creating distance analysis plots for {distance_type} distance")
    
    # Basic residual vs distance plots
    create_residual_distance_plots(df_plot, colors, plots_dir, output_format, config, distance_type)
    
    # Correlation analysis plots
    if config['distance_analysis'].get('enable_correlation_analysis', True):
        create_correlation_analysis_plots(df_plot, stats, colors, plots_dir, output_format, config, distance_type)
    
    # Binned analysis plots
    if config['distance_analysis'].get('enable_binned_analysis', True):
        create_binned_analysis_plots(df_plot, stats, colors, plots_dir, output_format, config, distance_type)
    
    # Ratio analysis plots
    if config['distance_analysis'].get('enable_ratio_analysis', True):
        create_ratio_distance_plots(df_plot, colors, plots_dir, output_format, config, distance_type)
    
    logger.info(f"Distance analysis plots saved to: {plots_dir}")


def create_residual_distance_plots(df: pd.DataFrame, colors: Dict[str, str],
                                  plots_dir: Path, output_format: str,
                                  config: Dict[str, Any], distance_type: str) -> None:
    """Create basic residual vs distance plots."""
    coordinate_type = config['distance_analysis']['coordinate_type']
    distance_col = get_distance_column_name(coordinate_type, distance_type)
    distance_label = get_distance_label(distance_type)
    comparison_mode = config['distance_analysis'].get('comparison_mode', 'ground_truth')
    
    # Generate title suffix and y-label based on comparison mode
    if comparison_mode == 'method_comparison':
        method1_name = config['distance_analysis'].get('method1_name', 'Method 1')
        method2_name = config['distance_analysis'].get('method2_name', 'Method 2')
        title_suffix = f'\n({method1_name} - {method2_name})'
        ylabel_residual = f'Residual ({method1_name} - {method2_name}) [s]'
    else:
        title_suffix = ''
        ylabel_residual = 'Residual (Est - Syn) [s]'
    
    p_data = df[df['phase'] == 'P']
    s_data = df[df['phase'] == 'S']
    
    fig, axes = create_multi_panel_figure(
        2, 3, figsize=(20, 14),
        title=f'Residuals vs {distance_label} Distance ({coordinate_type.title()} Coordinates){title_suffix}'
    )
    
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # P-wave linear residuals vs distance
    _plot_phase_residuals(axes_flat[0], p_data, distance_col, colors['p_wave_dark'],
                         'P-wave: Linear Residuals vs Distance', distance_label,
                         coordinate_type, colors['neutral_color'], ylabel_residual)
    
    # P-wave absolute residuals vs distance
    _plot_absolute_residuals(axes_flat[1], p_data, distance_col, colors['p_wave_light'],
                            'P-wave: Absolute Residuals vs Distance', distance_label,
                            coordinate_type, ylabel_residual)
    
    # P-wave binned residual statistics
    _plot_binned_statistics(axes_flat[2], p_data, distance_col, colors['p_wave_dark'],
                           'P-wave: Binned Statistics', distance_label, coordinate_type, ylabel_residual)
    
    # S-wave linear residuals vs distance
    _plot_phase_residuals(axes_flat[3], s_data, distance_col, colors['s_wave_dark'],
                         'S-wave: Linear Residuals vs Distance', distance_label,
                         coordinate_type, colors['neutral_color'], ylabel_residual)
    
    # S-wave absolute residuals vs distance
    _plot_absolute_residuals(axes_flat[4], s_data, distance_col, colors['s_wave_light'],
                            'S-wave: Absolute Residuals vs Distance', distance_label,
                            coordinate_type, ylabel_residual)
    
    # S-wave binned residual statistics
    _plot_binned_statistics(axes_flat[5], s_data, distance_col, colors['s_wave_dark'],
                           'S-wave: Binned Statistics', distance_label, coordinate_type, ylabel_residual)
    
    plt.tight_layout()
    save_figure(fig, plots_dir / 'residuals_vs_distance', output_format)


def _plot_phase_residuals(ax, data, distance_col, color, title, distance_label,
                          coordinate_type, neutral_color, ylabel_residual):
    """Plot phase residuals with trend line."""
    if data.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        return
    
    ax.scatter(data[distance_col], data['residual_s'], alpha=0.6, s=15, color=color)
    ax.axhline(0, color=neutral_color, linestyle='--', linewidth=2)
    
    # Add trend line
    if len(data) > 1:
        z = np.polyfit(data[distance_col], data['residual_s'], 1)
        p_trend = np.poly1d(z)
        x_trend = np.linspace(data[distance_col].min(), data[distance_col].max(), 100)
        ax.plot(x_trend, p_trend(x_trend), "r--", alpha=0.8, linewidth=2)
    
    ax.set_xlabel(f'{distance_label} Distance ({coordinate_type}) [km]')
    ax.set_ylabel(ylabel_residual)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _plot_absolute_residuals(ax, data, distance_col, color, title, distance_label,
                             coordinate_type, ylabel_residual):
    """Plot absolute residuals."""
    if data.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        return
    
    ax.scatter(data[distance_col], np.abs(data['residual_s']), alpha=0.6, s=15, color=color)
    ax.set_xlabel(f'{distance_label} Distance ({coordinate_type}) [km]')
    ax.set_ylabel('|Residual| [s]')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _plot_binned_statistics(ax, data, distance_col, color, title, distance_label,
                            coordinate_type, ylabel_residual):
    """Plot binned statistics."""
    if data.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        return
    
    # Create distance bins
    n_bins = 8
    data_copy = data.copy()
    data_copy['distance_bin'] = pd.cut(data_copy[distance_col], bins=n_bins)
    
    # Calculate bin statistics
    bin_stats = data_copy.groupby('distance_bin', observed=True).agg({
        'residual_s': ['count', 'mean', 'std'],
        distance_col: 'mean'
    }).reset_index()
    
    bin_stats.columns = ['distance_bin', 'count', 'mean_residual', 'std_residual', 'mean_distance']
    bin_stats = bin_stats.dropna()
    
    if not bin_stats.empty:
        ax.errorbar(bin_stats['mean_distance'], bin_stats['mean_residual'],
                   yerr=bin_stats['std_residual'],
                   fmt='o-', color=color, capsize=5, linewidth=2, markersize=6)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
    
    ax.set_xlabel(f'{distance_label} Distance ({coordinate_type}) [km]')
    ax.set_ylabel('Mean Residual ± Std [s]')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def create_correlation_analysis_plots(df: pd.DataFrame, stats: Dict[str, Any],
                                     colors: Dict[str, str], plots_dir: Path,
                                     output_format: str, config: Dict[str, Any],
                                     distance_type: str) -> None:
    """Create correlation analysis plots."""
    coordinate_type = config['distance_analysis']['coordinate_type']
    distance_col = get_distance_column_name(coordinate_type, distance_type)
    distance_label = get_distance_label(distance_type)
    comparison_mode = config['distance_analysis'].get('comparison_mode', 'ground_truth')
    
    if comparison_mode == 'method_comparison':
        method1_name = config['distance_analysis'].get('method1_name', 'Method 1')
        method2_name = config['distance_analysis'].get('method2_name', 'Method 2')
        title_suffix = f'\n({method1_name} - {method2_name})'
        ylabel_residual = f'Residual ({method1_name} - {method2_name}) [s]'
    else:
        title_suffix = ''
        ylabel_residual = 'Residual (Est - Syn) [s]'
    
    p_data = df[df['phase'] == 'P']
    s_data = df[df['phase'] == 'S']
    
    fig, axes = create_multi_panel_figure(
        2, 2, figsize=(16, 12),
        title=f'{distance_label} Distance Correlation Analysis ({coordinate_type.title()}){title_suffix}'
    )
    
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Overall correlation plot
    if not df.empty:
        axes_flat[0].scatter(df[distance_col], df['residual_s'], alpha=0.6, s=15,
                           color=colors['neutral_color'])
        axes_flat[0].axhline(0, color='black', linestyle='--', linewidth=2)
        
        if len(df) > 1:
            z = np.polyfit(df[distance_col], df['residual_s'], 1)
            p_trend = np.poly1d(z)
            x_trend = np.linspace(df[distance_col].min(), df[distance_col].max(), 100)
            axes_flat[0].plot(x_trend, p_trend(x_trend), "r--", alpha=0.8, linewidth=2)
        
        corr = df[distance_col].corr(df['residual_s'])
        axes_flat[0].set_xlabel(f'{distance_label} Distance ({coordinate_type}) [km]')
        axes_flat[0].set_ylabel(ylabel_residual)
        axes_flat[0].set_title(f'Overall Correlation (r = {corr:.4f})')
        axes_flat[0].grid(True, alpha=0.3)
    
    # Phase-specific correlations
    if not p_data.empty:
        axes_flat[1].scatter(p_data[distance_col], p_data['residual_s'],
                           alpha=0.6, s=15, color=colors['p_wave_dark'], label='P-wave')
        p_corr = p_data[distance_col].corr(p_data['residual_s'])
        
        if len(p_data) > 1:
            z_p = np.polyfit(p_data[distance_col], p_data['residual_s'], 1)
            p_p = np.poly1d(z_p)
            x_trend_p = np.linspace(p_data[distance_col].min(), p_data[distance_col].max(), 100)
            axes_flat[1].plot(x_trend_p, p_p(x_trend_p), color=colors['p_wave_light'],
                            linestyle='--', alpha=0.8, linewidth=2)
    
    if not s_data.empty:
        axes_flat[1].scatter(s_data[distance_col], s_data['residual_s'],
                           alpha=0.6, s=15, color=colors['s_wave_dark'], label='S-wave')
        s_corr = s_data[distance_col].corr(s_data['residual_s'])
        
        if len(s_data) > 1:
            z_s = np.polyfit(s_data[distance_col], s_data['residual_s'], 1)
            p_s = np.poly1d(z_s)
            x_trend_s = np.linspace(s_data[distance_col].min(), s_data[distance_col].max(), 100)
            axes_flat[1].plot(x_trend_s, p_s(x_trend_s), color=colors['s_wave_light'],
                            linestyle='--', alpha=0.8, linewidth=2)
    
    axes_flat[1].axhline(0, color='black', linestyle='--', linewidth=2)
    axes_flat[1].set_xlabel(f'{distance_label} Distance ({coordinate_type}) [km]')
    axes_flat[1].set_ylabel(ylabel_residual)
    
    title_text = 'Phase-specific Correlations'
    if not p_data.empty and not s_data.empty:
        title_text += f'\nP: r={p_corr:.4f}, S: r={s_corr:.4f}'
    elif not p_data.empty:
        title_text += f'\nP: r={p_corr:.4f}'
    elif not s_data.empty:
        title_text += f'\nS: r={s_corr:.4f}'
    
    axes_flat[1].set_title(title_text)
    axes_flat[1].grid(True, alpha=0.3)
    axes_flat[1].legend()
    
    # Absolute residuals vs distance
    if not df.empty:
        axes_flat[2].scatter(df[distance_col], np.abs(df['residual_s']),
                           alpha=0.6, s=15, color=colors['neutral_color'])
        
        abs_corr = df[distance_col].corr(np.abs(df['residual_s']))
        axes_flat[2].set_xlabel(f'{distance_label} Distance ({coordinate_type}) [km]')
        axes_flat[2].set_ylabel('|Residual| [s]')
        axes_flat[2].set_title(f'Distance vs |Residual| (r = {abs_corr:.4f})')
        axes_flat[2].grid(True, alpha=0.3)
    
    # Distance distribution
    if not df.empty:
        axes_flat[3].hist(df[distance_col], bins=30, alpha=0.7, color=colors['neutral_color'],
                        edgecolor='black')
        axes_flat[3].axvline(df[distance_col].mean(), color='red', linestyle='-',
                           linewidth=2, label='Mean')
        axes_flat[3].axvline(df[distance_col].median(), color='orange', linestyle='--',
                           linewidth=2, label='Median')
        axes_flat[3].set_xlabel(f'{distance_label} Distance ({coordinate_type}) [km]')
        axes_flat[3].set_ylabel('Frequency')
        axes_flat[3].set_title('Distance Distribution')
        axes_flat[3].grid(True, alpha=0.3)
        axes_flat[3].legend()
    
    plt.tight_layout()
    save_figure(fig, plots_dir / 'correlation_analysis', output_format)


def create_binned_analysis_plots(df: pd.DataFrame, stats: Dict[str, Any], colors: Dict[str, str],
                                plots_dir: Path, output_format: str, config: Dict[str, Any],
                                distance_type: str) -> None:
    """Create binned analysis plots."""
    coordinate_type = config['distance_analysis']['coordinate_type']
    distance_col = get_distance_column_name(coordinate_type, distance_type)
    distance_label = get_distance_label(distance_type)
    comparison_mode = config['distance_analysis'].get('comparison_mode', 'ground_truth')
    
    if comparison_mode == 'method_comparison':
        method1_name = config['distance_analysis'].get('method1_name', 'Method 1')
        method2_name = config['distance_analysis'].get('method2_name', 'Method 2')
        title_suffix = f'\n({method1_name} - {method2_name})'
    else:
        title_suffix = ''
    
    p_data = df[df['phase'] == 'P']
    s_data = df[df['phase'] == 'S']
    
    fig, axes = create_multi_panel_figure(
        2, 2, figsize=(16, 12),
        title=f'Binned {distance_label} Distance Analysis ({coordinate_type.title()}){title_suffix}'
    )
    
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # P-wave mean residuals by distance bin
    if not p_data.empty:
        _plot_binned_statistics(axes_flat[0], p_data, distance_col, colors['p_wave_dark'],
                               'P-wave: Mean Residuals by Distance', distance_label, coordinate_type, '')
    
    # P-wave RMS residuals by distance bin
    if not p_data.empty:
        _plot_rms_binned(axes_flat[1], p_data, distance_col, colors['p_wave_dark'],
                        'P-wave: RMS Residuals by Distance', distance_label, coordinate_type)
    
    # S-wave mean residuals by distance bin
    if not s_data.empty:
        _plot_binned_statistics(axes_flat[2], s_data, distance_col, colors['s_wave_dark'],
                               'S-wave: Mean Residuals by Distance', distance_label, coordinate_type, '')
    
    # S-wave RMS residuals by distance bin
    if not s_data.empty:
        _plot_rms_binned(axes_flat[3], s_data, distance_col, colors['s_wave_dark'],
                        'S-wave: RMS Residuals by Distance', distance_label, coordinate_type)
    
    plt.tight_layout()
    save_figure(fig, plots_dir / 'binned_distance_analysis', output_format)


def _plot_rms_binned(ax, data, distance_col, color, title, distance_label, coordinate_type):
    """Plot RMS binned statistics."""
    if data.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        return
    
    n_bins = 8
    data_copy = data.copy()
    data_copy['distance_bin'] = pd.cut(data_copy[distance_col], bins=n_bins)
    
    def rms(x):
        return np.sqrt(np.mean(x**2))
    
    bin_stats = data_copy.groupby('distance_bin', observed=True).agg({
        'residual_s': ['count', rms],
        distance_col: 'mean'
    }).reset_index()
    
    bin_stats.columns = ['distance_bin', 'count', 'rms_residual', 'mean_distance']
    bin_stats = bin_stats.dropna()
    
    if not bin_stats.empty:
        ax.plot(bin_stats['mean_distance'], bin_stats['rms_residual'],
               'o-', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel(f'{distance_label} Distance ({coordinate_type}) [km]')
    ax.set_ylabel('RMS Residual [s]')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def create_ratio_distance_plots(df: pd.DataFrame, colors: Dict[str, str],
                               plots_dir: Path, output_format: str,
                               config: Dict[str, Any], distance_type: str) -> None:
    """Create ratio vs distance plots."""
    ratio_data = df[(df['observed_tt'] > 0) & (df['synthetic_tt'] > 0)].copy()
    
    if ratio_data.empty:
        logger.warning("No positive travel times for ratio analysis")
        return
    
    coordinate_type = config['distance_analysis']['coordinate_type']
    distance_col = get_distance_column_name(coordinate_type, distance_type)
    distance_label = get_distance_label(distance_type)
    comparison_mode = config['distance_analysis'].get('comparison_mode', 'ground_truth')
    
    if comparison_mode == 'method_comparison':
        method1_name = config['distance_analysis'].get('method1_name', 'Method 1')
        method2_name = config['distance_analysis'].get('method2_name', 'Method 2')
        title_suffix = f'\n({method2_name} / {method1_name})'
        ylabel_ratio = f'Ratio ({method2_name}/{method1_name})'
    else:
        title_suffix = ''
        ylabel_ratio = 'Ratio (Syn/Est)'
    
    ratio_data['ratio'] = ratio_data['synthetic_tt'] / ratio_data['observed_tt']
    
    p_ratio = ratio_data[ratio_data['phase'] == 'P']
    s_ratio = ratio_data[ratio_data['phase'] == 'S']
    
    fig, axes = create_multi_panel_figure(
        2, 2, figsize=(16, 12),
        title=f'Travel Time Ratios vs {distance_label} Distance ({coordinate_type.title()}){title_suffix}'
    )
    
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # P-wave ratio vs distance
    if not p_ratio.empty:
        axes_flat[0].scatter(p_ratio[distance_col], p_ratio['ratio'],
                           alpha=0.6, s=15, color=colors['p_wave_dark'])
        axes_flat[0].axhline(1, color=colors['neutral_color'], linestyle='--', linewidth=2)
        axes_flat[0].set_xlabel(f'{distance_label} Distance ({coordinate_type}) [km]')
        axes_flat[0].set_ylabel(ylabel_ratio)
        axes_flat[0].set_title('P-wave: Travel Time Ratio vs Distance')
        axes_flat[0].grid(True, alpha=0.3)
    
    # P-wave ratio vs distance (log scale)
    if not p_ratio.empty:
        axes_flat[1].scatter(p_ratio[distance_col], p_ratio['ratio'],
                           alpha=0.6, s=15, color=colors['p_wave_light'])
        axes_flat[1].axhline(1, color=colors['neutral_color'], linestyle='--', linewidth=2)
        axes_flat[1].set_xscale('log')
        axes_flat[1].set_xlabel(f'{distance_label} Distance ({coordinate_type}) [km]')
        axes_flat[1].set_ylabel(ylabel_ratio)
        axes_flat[1].set_title('P-wave: Ratio vs Distance (log scale)')
        axes_flat[1].grid(True, which='both', alpha=0.3)
    
    # S-wave ratio vs distance
    if not s_ratio.empty:
        axes_flat[2].scatter(s_ratio[distance_col], s_ratio['ratio'],
                           alpha=0.6, s=15, color=colors['s_wave_dark'])
        axes_flat[2].axhline(1, color=colors['neutral_color'], linestyle='--', linewidth=2)
        axes_flat[2].set_xlabel(f'{distance_label} Distance ({coordinate_type}) [km]')
        axes_flat[2].set_ylabel(ylabel_ratio)
        axes_flat[2].set_title('S-wave: Travel Time Ratio vs Distance')
        axes_flat[2].grid(True, alpha=0.3)
    
    # S-wave ratio vs distance (log scale)
    if not s_ratio.empty:
        axes_flat[3].scatter(s_ratio[distance_col], s_ratio['ratio'],
                           alpha=0.6, s=15, color=colors['s_wave_light'])
        axes_flat[3].axhline(1, color=colors['neutral_color'], linestyle='--', linewidth=2)
        axes_flat[3].set_xscale('log')
        axes_flat[3].set_xlabel(f'{distance_label} Distance ({coordinate_type}) [km]')
        axes_flat[3].set_ylabel(ylabel_ratio)
        axes_flat[3].set_title('S-wave: Ratio vs Distance (log scale)')
        axes_flat[3].grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, plots_dir / 'ratios_vs_distance', output_format) 