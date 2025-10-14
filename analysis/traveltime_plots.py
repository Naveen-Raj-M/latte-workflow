"""
Travel time plotting functions.

This module contains all plotting functions for travel time analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
import logging

from utils.plot_utils import (
    get_color_scheme, save_figure, create_multi_panel_figure,
    add_statistics_text
)
from utils.io_utils import ensure_directory_exists

logger = logging.getLogger(__name__)


def create_travel_time_plots(df: pd.DataFrame, stats: Dict[str, Any], 
                            config: Dict[str, Any],
                            comparison_mode: str = 'ground_truth') -> None:
    """
    Create comprehensive travel time analysis plots.
    
    Args:
        df: Travel time dataframe
        stats: Statistics dictionary
        config: Configuration dictionary
        comparison_mode: 'ground_truth' or 'method_comparison'
    """
    output_dir = Path(config['travel_time_analysis']['output_dir'])
    plots_dir = output_dir / 'plots'
    ensure_directory_exists(plots_dir)
    
    colors = get_color_scheme(config['analysis'])
    output_format = config['analysis'].get('output_format', 'png')
    
    # Get titles based on comparison mode
    if comparison_mode == 'method_comparison':
        method1_name = config['travel_time_analysis'].get('method1_name', 'Method 1')
        method2_name = config['travel_time_analysis'].get('method2_name', 'Method 2')
        title_suffix = f"{method1_name} vs {method2_name}"
        xlabel_obs = f'{method1_name} Travel Time (s)'
        xlabel_syn = f'{method2_name} Travel Time (s)'
    else:
        title_suffix = "Observed vs Synthetic"
        xlabel_obs = 'Observed Travel Time (s)'
        xlabel_syn = 'Synthetic Travel Time (s)'
    
    # Basic comparison plots
    create_basic_travel_time_plots(df, colors, plots_dir, output_format, config,
                                   title_suffix, xlabel_obs, xlabel_syn)
    
    # Residual histograms
    create_residual_histograms(df, colors, plots_dir, output_format, config,
                              comparison_mode)
    
    # Log-log plots if enabled
    if config['travel_time_analysis'].get('enable_log_log_plots', True):
        create_log_log_plots(df, colors, plots_dir, output_format, config,
                            title_suffix, xlabel_obs, xlabel_syn)
    
    # Logarithmic ratio analysis if enabled
    if config['travel_time_analysis'].get('enable_ln_ratio_analysis', True):
        create_ln_ratio_histograms(df, colors, plots_dir, output_format, config,
                                   comparison_mode)
        create_ln_ratio_plots(df, colors, plots_dir, output_format, config,
                             xlabel_obs)


def create_basic_travel_time_plots(df: pd.DataFrame, colors: Dict[str, str],
                                  plots_dir: Path, output_format: str,
                                  config: Dict[str, Any],
                                  title_suffix: str,
                                  xlabel_obs: str,
                                  xlabel_syn: str) -> None:
    """Create basic travel time comparison plots."""
    p_data = df[df['phase'] == 'P']
    s_data = df[df['phase'] == 'S']
    
    fig, axes = create_multi_panel_figure(1, 2, figsize=(16, 8),
                                        title=f'Travel Time Comparison: {title_suffix}')
    
    # P-wave plot
    if not p_data.empty:
        axes[0].scatter(p_data['observed_tt'], p_data['synthetic_tt'], 
                       alpha=0.6, s=15, color=colors['p_wave_dark'])
        
        # Add 1:1 line
        obs_range = [p_data['observed_tt'].min(), p_data['observed_tt'].max()]
        axes[0].plot(obs_range, obs_range, 'r--', lw=2, label='1:1 line')
        
        axes[0].set_xlabel(xlabel_obs)
        axes[0].set_ylabel(xlabel_syn)
        axes[0].set_title('P-wave')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal', adjustable='box')
        
        # Add statistics
        add_statistics_text(axes[0], p_data['residual_s'], 'upper left')
    
    # S-wave plot
    if not s_data.empty:
        axes[1].scatter(s_data['observed_tt'], s_data['synthetic_tt'], 
                       alpha=0.6, s=15, color=colors['s_wave_dark'])
        
        # Add 1:1 line
        obs_range = [s_data['observed_tt'].min(), s_data['observed_tt'].max()]
        axes[1].plot(obs_range, obs_range, 'r--', lw=2, label='1:1 line')
        
        axes[1].set_xlabel(xlabel_obs)
        axes[1].set_ylabel(xlabel_syn)
        axes[1].set_title('S-wave')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect('equal', adjustable='box')
        
        # Add statistics
        add_statistics_text(axes[1], s_data['residual_s'], 'upper left')
    
    save_figure(fig, plots_dir / 'synthetic_vs_observed', output_format)


def create_residual_histograms(df: pd.DataFrame, colors: Dict[str, str],
                              plots_dir: Path, output_format: str,
                              config: Dict[str, Any],
                              comparison_mode: str) -> None:
    """Create residual distribution histograms."""
    p_data = df[df['phase'] == 'P']
    s_data = df[df['phase'] == 'S']
    
    bins = config['travel_time_analysis'].get('histogram_bins', 50)
    
    if comparison_mode == 'method_comparison':
        method1_name = config['travel_time_analysis'].get('method1_name', 'Method 1')
        method2_name = config['travel_time_analysis'].get('method2_name', 'Method 2')
        title = f'Travel Time Difference Distributions\n({method2_name} - {method1_name})'
        xlabel = f'Difference ({method2_name} - {method1_name}) [s]'
    else:
        title = 'Travel Time Residual Distributions'
        xlabel = 'Residual (Observed - Synthetic) [s]'
    
    fig, axes = create_multi_panel_figure(1, 2, figsize=(16, 8), title=title)
    
    # P-wave histogram
    if not p_data.empty:
        axes[0].hist(p_data['residual_s'], bins=bins, alpha=0.7, 
                    color=colors['p_wave_light'], edgecolor='black', linewidth=0.5)
        axes[0].axvline(p_data['residual_s'].mean(), color='red', linestyle='--',
                       linewidth=2, label=f"Mean: {p_data['residual_s'].mean():.3f}s")
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('P-wave')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # S-wave histogram
    if not s_data.empty:
        axes[1].hist(s_data['residual_s'], bins=bins, alpha=0.7, 
                    color=colors['s_wave_light'], edgecolor='black', linewidth=0.5)
        axes[1].axvline(s_data['residual_s'].mean(), color='red', linestyle='--',
                       linewidth=2, label=f"Mean: {s_data['residual_s'].mean():.3f}s")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('S-wave')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    save_figure(fig, plots_dir / 'residual_histograms', output_format)


def create_log_log_plots(df: pd.DataFrame, colors: Dict[str, str],
                        plots_dir: Path, output_format: str,
                        config: Dict[str, Any],
                        title_suffix: str,
                        xlabel_obs: str,
                        xlabel_syn: str) -> None:
    """Create log-log scale comparison plots."""
    # Filter positive values for log plots
    log_data = df[(df['observed_tt'] > 0) & (df['synthetic_tt'] > 0)].copy()
    
    if log_data.empty:
        logger.warning("No positive travel times found for log-log plots")
        return
    
    p_data = log_data[log_data['phase'] == 'P']
    s_data = log_data[log_data['phase'] == 'S']
    
    fig, axes = create_multi_panel_figure(1, 2, figsize=(16, 8),
                                        title=f'Travel Time Comparison: Log-Log Scale\n{title_suffix}')
    
    # P-wave log-log plot
    if not p_data.empty:
        axes[0].loglog(p_data['observed_tt'], p_data['synthetic_tt'], 'o', 
                      alpha=0.6, markersize=4, color=colors['p_wave_dark'])
        
        # Add 1:1 line
        obs_range = [p_data['observed_tt'].min(), p_data['observed_tt'].max()]
        axes[0].loglog(obs_range, obs_range, 'r--', lw=2, label='1:1 line')
        
        axes[0].set_xlabel(xlabel_obs)
        axes[0].set_ylabel(xlabel_syn)
        axes[0].set_title('P-wave: Log-Log Scale')
        axes[0].legend()
        axes[0].grid(True, which="both", ls='--', alpha=0.3)
        axes[0].tick_params(which='both', direction='in')
        axes[0].minorticks_on()
    
    # S-wave log-log plot
    if not s_data.empty:
        axes[1].loglog(s_data['observed_tt'], s_data['synthetic_tt'], 'o', 
                      alpha=0.6, markersize=4, color=colors['s_wave_dark'])
        
        # Add 1:1 line
        obs_range = [s_data['observed_tt'].min(), s_data['observed_tt'].max()]
        axes[1].loglog(obs_range, obs_range, 'r--', lw=2, label='1:1 line')
        
        axes[1].set_xlabel(xlabel_obs)
        axes[1].set_ylabel(xlabel_syn)
        axes[1].set_title('S-wave: Log-Log Scale')
        axes[1].legend()
        axes[1].grid(True, which="both", ls='--', alpha=0.3)
        axes[1].tick_params(which='both', direction='in')
        axes[1].minorticks_on()
    
    save_figure(fig, plots_dir / 'synthetic_vs_observed_loglog', output_format)


def create_ln_ratio_histograms(df: pd.DataFrame, colors: Dict[str, str],
                              plots_dir: Path, output_format: str,
                              config: Dict[str, Any],
                              comparison_mode: str) -> None:
    """Create separate logarithmic ratio histogram plots."""
    # Filter positive values for ratio analysis
    ratio_data = df[(df['observed_tt'] > 0) & (df['synthetic_tt'] > 0)].copy()
    
    if ratio_data.empty:
        logger.warning("No positive travel times found for ratio analysis")
        return
    
    ratio_data['ln_ratio'] = np.log(ratio_data['synthetic_tt'] / ratio_data['observed_tt'])
    
    # Remove infinite values
    ratio_data = ratio_data[np.isfinite(ratio_data['ln_ratio'])]
    
    if ratio_data.empty:
        logger.warning("No finite ln(ratio) values found")
        return
    
    p_data = ratio_data[ratio_data['phase'] == 'P']
    s_data = ratio_data[ratio_data['phase'] == 'S']
    
    if comparison_mode == 'method_comparison':
        method1_name = config['travel_time_analysis'].get('method1_name', 'Method 1')
        method2_name = config['travel_time_analysis'].get('method2_name', 'Method 2')
        title = f'Logarithmic Ratio Histograms: ln({method2_name}/{method1_name})'
        xlabel = f'ln({method2_name}/{method1_name})'
    else:
        title = 'Logarithmic Ratio Histograms: ln(Synthetic/Observed)'
        xlabel = 'ln(Synthetic/Observed)'
    
    # Create histogram plots
    fig, axes = create_multi_panel_figure(1, 2, figsize=(16, 8), title=title)
    
    bins = config['travel_time_analysis'].get('histogram_bins', 50)
    
    # P-wave ln(ratio) histogram
    if not p_data.empty:
        axes[0].hist(p_data['ln_ratio'], bins=bins, alpha=0.7, 
                    color=colors['p_wave_light'], edgecolor='black', linewidth=0.5)
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='ln(ratio)=0')
        axes[0].axvline(p_data['ln_ratio'].mean(), color='orange', linestyle='-', linewidth=2,
                       label=f"Mean: {p_data['ln_ratio'].mean():.4f}")
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('P-wave: ln(Ratio) Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # S-wave ln(ratio) histogram
    if not s_data.empty:
        axes[1].hist(s_data['ln_ratio'], bins=bins, alpha=0.7, 
                    color=colors['s_wave_light'], edgecolor='black', linewidth=0.5)
        axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='ln(ratio)=0')
        axes[1].axvline(s_data['ln_ratio'].mean(), color='orange', linestyle='-', linewidth=2,
                       label=f"Mean: {s_data['ln_ratio'].mean():.4f}")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('S-wave: ln(Ratio) Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    save_figure(fig, plots_dir / 'ln_ratio_histograms', output_format)


def create_ln_ratio_plots(df: pd.DataFrame, colors: Dict[str, str],
                         plots_dir: Path, output_format: str,
                         config: Dict[str, Any],
                         xlabel_obs: str) -> None:
    """Create logarithmic ratio vs travel time plots."""
    # Filter positive values for ratio analysis
    ratio_data = df[(df['observed_tt'] > 0) & (df['synthetic_tt'] > 0)].copy()
    
    if ratio_data.empty:
        logger.warning("No positive travel times found for ratio analysis")
        return
    
    ratio_data['ln_ratio'] = np.log(ratio_data['synthetic_tt'] / ratio_data['observed_tt'])
    
    # Remove infinite values
    ratio_data = ratio_data[np.isfinite(ratio_data['ln_ratio'])]
    
    if ratio_data.empty:
        logger.warning("No finite ln(ratio) values found")
        return
    
    p_data = ratio_data[ratio_data['phase'] == 'P']
    s_data = ratio_data[ratio_data['phase'] == 'S']
    
    # Create ln(ratio) vs travel time plots
    fig, axes = create_multi_panel_figure(2, 2, figsize=(16, 12),
                                        title='Logarithmic Ratio vs Travel Time Analysis')
    
    # Ensure axes is iterable - handle both single axis and array cases
    if not hasattr(axes, '__len__'):
        axes = [axes]
    elif hasattr(axes, 'flatten'):
        axes = axes.flatten()
    
    # First row: ln(ratio) vs observed time (linear scale)
    # P-wave ln(ratio) vs observed time
    if not p_data.empty and len(axes) > 0:
        axes[0].scatter(p_data['observed_tt'], p_data['ln_ratio'], 
                       alpha=0.6, s=15, color=colors['p_wave_dark'])
        axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel(xlabel_obs)
        axes[0].set_ylabel('ln(Synthetic/Observed)')
        axes[0].set_title('P-wave: ln(Ratio) vs Time')
        axes[0].grid(True, alpha=0.3)
    
    # S-wave ln(ratio) vs observed time
    if not s_data.empty and len(axes) > 1:
        axes[1].scatter(s_data['observed_tt'], s_data['ln_ratio'], 
                       alpha=0.6, s=15, color=colors['s_wave_dark'])
        axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel(xlabel_obs)
        axes[1].set_ylabel('ln(Synthetic/Observed)')
        axes[1].set_title('S-wave: ln(Ratio) vs Time')
        axes[1].grid(True, alpha=0.3)
    
    # Second row: ln(ratio) vs observed time (log scale)
    # P-wave ln(ratio) vs observed time (log scale)
    if not p_data.empty and len(axes) > 2:
        axes[2].semilogx(p_data['observed_tt'], p_data['ln_ratio'], 'o',
                        alpha=0.6, markersize=4, color=colors['p_wave_dark'])
        axes[2].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[2].set_xlabel(f'{xlabel_obs} [log scale]')
        axes[2].set_ylabel('ln(Synthetic/Observed)')
        axes[2].set_title('P-wave: ln(Ratio) vs Time (Log Scale)')
        axes[2].grid(True, which="both", ls='--', alpha=0.3)
        axes[2].tick_params(which='both', direction='in')
        axes[2].minorticks_on()
    
    # S-wave ln(ratio) vs observed time (log scale)
    if not s_data.empty and len(axes) > 3:
        axes[3].semilogx(s_data['observed_tt'], s_data['ln_ratio'], 'o',
                        alpha=0.6, markersize=4, color=colors['s_wave_dark'])
        axes[3].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[3].set_xlabel(f'{xlabel_obs} [log scale]')
        axes[3].set_ylabel('ln(Synthetic/Observed)')
        axes[3].set_title('S-wave: ln(Ratio) vs Time (Log Scale)')
        axes[3].grid(True, which="both", ls='--', alpha=0.3)
        axes[3].tick_params(which='both', direction='in')
        axes[3].minorticks_on()
    
    save_figure(fig, plots_dir / 'ln_ratio_analysis', output_format)