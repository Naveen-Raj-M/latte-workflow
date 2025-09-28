"""
Travel time residual analysis module.

This module provides comprehensive analysis of travel time residuals
including basic statistics, ratio analysis, and comparison plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
from datetime import datetime

from utils.io_utils import read_csv_file, write_text_file, ensure_directory_exists
from utils.plot_utils import (
    setup_plot_style, get_color_scheme, save_figure,
    create_histogram_with_stats, create_scatter_plot,
    create_multi_panel_figure, add_statistics_text
)

logger = logging.getLogger(__name__)


def analyze_travel_time_residuals(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function for travel time residual analysis.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    logger.info("Starting travel time residual analysis")
    
    # Setup plot style
    setup_plot_style(config['analysis'])
    
    # Load data
    input_csv = config['travel_time_analysis']['input_csv']
    output_dir = Path(config['travel_time_analysis']['output_dir'])
    
    logger.info(f"Loading travel time data from: {input_csv}")
    df = read_csv_file(input_csv)
    
    # Validate and clean data
    df_clean = validate_and_clean_travel_time_data(df, config)
    
    # Compute statistics
    logger.info("Computing travel time statistics")
    stats = compute_travel_time_statistics(df_clean, config)
    
    # Create plots
    logger.info("Creating travel time plots")
    create_travel_time_plots(df_clean, stats, config)
    
    # Generate summary report
    if config['travel_time_analysis'].get('generate_summary_report', True):
        logger.info("Generating summary report")
        generate_travel_time_report(df_clean, stats, config)
    
    logger.info("Travel time residual analysis completed")
    return stats


def validate_and_clean_travel_time_data(df: pd.DataFrame, 
                                       config: Dict[str, Any]) -> pd.DataFrame:
    """
    Validate and clean travel time data.
    
    Args:
        df: Input dataframe
        config: Configuration dictionary
        
    Returns:
        Cleaned dataframe
    """
    logger.info(f"Validating travel time data: {len(df)} initial records")
    
    # Remove infinite and NaN values
    initial_count = len(df)
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['observed_tt', 'synthetic_tt', 'residual_s']
    )
    
    if len(df_clean) < initial_count:
        logger.warning(f"Removed {initial_count - len(df_clean)} rows with non-finite values")
    
    # Apply residual thresholds
    min_res = config['travel_time_analysis'].get('min_residual_threshold', -10.0)
    max_res = config['travel_time_analysis'].get('max_residual_threshold', 10.0)
    
    mask = (df_clean['residual_s'] >= min_res) & (df_clean['residual_s'] <= max_res)
    df_clean = df_clean[mask]
    
    logger.info(f"Final clean dataset: {len(df_clean)} records")
    return df_clean


def compute_travel_time_statistics(df: pd.DataFrame, 
                                 config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute comprehensive travel time statistics.
    
    Args:
        df: Travel time dataframe
        config: Configuration dictionary
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    # Overall statistics
    residuals = df['residual_s']
    stats['overall'] = {
        'count': len(df),
        'mean_residual': residuals.mean(),
        'std_residual': residuals.std(),
        'median_residual': residuals.median(),
        'min_residual': residuals.min(),
        'max_residual': residuals.max(),
        'rms_residual': np.sqrt(np.mean(residuals**2)),
        'mae_residual': np.mean(np.abs(residuals)),
    }
    
    # Phase-specific statistics
    for phase in ['P', 'S']:
        phase_data = df[df['phase'] == phase]
        if not phase_data.empty:
            phase_residuals = phase_data['residual_s']
            obs_tt = phase_data['observed_tt']
            syn_tt = phase_data['synthetic_tt']
            
            stats[f'{phase.lower()}_wave'] = {
                'count': len(phase_data),
                'mean_residual': phase_residuals.mean(),
                'std_residual': phase_residuals.std(),
                'rms_residual': np.sqrt(np.mean(phase_residuals**2)),
                'mae_residual': np.mean(np.abs(phase_residuals)),
                'mean_observed_tt': obs_tt.mean(),
                'mean_synthetic_tt': syn_tt.mean(),
                'correlation': obs_tt.corr(syn_tt),
            }
            
            # Logarithmic ratio analysis if enabled
            if config['travel_time_analysis'].get('enable_ln_ratio_analysis', True):
                valid_mask = (obs_tt > 0) & (syn_tt > 0)
                if valid_mask.sum() > 0:
                    ln_ratios = np.log(syn_tt[valid_mask] / obs_tt[valid_mask])
                    stats[f'{phase.lower()}_wave']['ln_ratio'] = {
                        'mean': ln_ratios.mean(),
                        'std': ln_ratios.std(),
                        'median': ln_ratios.median(),
                        'equivalent_percent_change': (np.exp(ln_ratios.mean()) - 1) * 100
                    }
    
    # Station statistics
    station_stats = df.groupby('station')['residual_s'].agg([
        'count', 'mean', 'std'
    ]).round(4)
    stats['station_summary'] = {
        'unique_stations': df['station'].nunique(),
        'top_5_by_count': station_stats.nlargest(5, 'count').to_dict('index')
    }
    
    # Event statistics  
    event_stats = df.groupby('event_id')['residual_s'].agg([
        'count', 'mean', 'std'
    ]).round(4)
    stats['event_summary'] = {
        'unique_events': df['event_id'].nunique(),
        'top_5_by_count': event_stats.nlargest(5, 'count').to_dict('index')
    }
    
    return stats


def create_travel_time_plots(df: pd.DataFrame, stats: Dict[str, Any], 
                           config: Dict[str, Any]) -> None:
    """
    Create comprehensive travel time analysis plots.
    
    Args:
        df: Travel time dataframe
        stats: Statistics dictionary
        config: Configuration dictionary
    """
    output_dir = Path(config['travel_time_analysis']['output_dir'])
    plots_dir = output_dir / 'plots'
    ensure_directory_exists(plots_dir)
    
    colors = get_color_scheme(config['analysis'])
    output_format = config['analysis'].get('output_format', 'png')
    
    # Basic comparison plots
    create_basic_travel_time_plots(df, colors, plots_dir, output_format, config)
    
    # Residual histograms
    create_residual_histograms(df, colors, plots_dir, output_format, config)
    
    # Log-log plots if enabled
    if config['travel_time_analysis'].get('enable_log_log_plots', True):
        create_log_log_plots(df, colors, plots_dir, output_format, config)
    
    # Logarithmic ratio analysis if enabled
    if config['travel_time_analysis'].get('enable_ln_ratio_analysis', True):
        create_ln_ratio_plots(df, colors, plots_dir, output_format, config)


def create_basic_travel_time_plots(df: pd.DataFrame, colors: Dict[str, str],
                                 plots_dir: Path, output_format: str,
                                 config: Dict[str, Any]) -> None:
    """Create basic travel time comparison plots."""
    p_data = df[df['phase'] == 'P']
    s_data = df[df['phase'] == 'S']
    
    fig, axes = create_multi_panel_figure(1, 2, figsize=(16, 8),
                                        title='Travel Time Comparison: Observed vs Synthetic')
    
    # P-wave plot
    if not p_data.empty:
        axes[0].scatter(p_data['observed_tt'], p_data['synthetic_tt'], 
                       alpha=0.6, s=15, color=colors['p_wave_dark'])
        
        # Add 1:1 line
        obs_range = [p_data['observed_tt'].min(), p_data['observed_tt'].max()]
        axes[0].plot(obs_range, obs_range, 'r--', lw=2, label='1:1 line')
        
        axes[0].set_xlabel('Observed Travel Time (s)')
        axes[0].set_ylabel('Synthetic Travel Time (s)')
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
        
        axes[1].set_xlabel('Observed Travel Time (s)')
        axes[1].set_ylabel('Synthetic Travel Time (s)')
        axes[1].set_title('S-wave')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect('equal', adjustable='box')
        
        # Add statistics
        add_statistics_text(axes[1], s_data['residual_s'], 'upper left')
    
    save_figure(fig, plots_dir / 'synthetic_vs_observed', output_format)


def create_residual_histograms(df: pd.DataFrame, colors: Dict[str, str],
                             plots_dir: Path, output_format: str,
                             config: Dict[str, Any]) -> None:
    """Create residual distribution histograms."""
    p_data = df[df['phase'] == 'P']
    s_data = df[df['phase'] == 'S']
    
    bins = config['travel_time_analysis'].get('histogram_bins', 50)
    
    fig, axes = create_multi_panel_figure(1, 2, figsize=(16, 8),
                                        title='Travel Time Residual Distributions')
    
    # P-wave histogram
    if not p_data.empty:
        axes[0].hist(p_data['residual_s'], bins=bins, alpha=0.7, 
                    color=colors['p_wave_light'], edgecolor='black', linewidth=0.5)
        axes[0].axvline(p_data['residual_s'].mean(), color='red', linestyle='--',
                       linewidth=2, label=f"Mean: {p_data['residual_s'].mean():.3f}s")
        axes[0].set_xlabel('Residual (Observed - Synthetic) [s]')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('P-wave Residuals')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # S-wave histogram
    if not s_data.empty:
        axes[1].hist(s_data['residual_s'], bins=bins, alpha=0.7, 
                    color=colors['s_wave_light'], edgecolor='black', linewidth=0.5)
        axes[1].axvline(s_data['residual_s'].mean(), color='red', linestyle='--',
                       linewidth=2, label=f"Mean: {s_data['residual_s'].mean():.3f}s")
        axes[1].set_xlabel('Residual (Observed - Synthetic) [s]')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('S-wave Residuals')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    save_figure(fig, plots_dir / 'residual_histograms', output_format)


def create_log_log_plots(df: pd.DataFrame, colors: Dict[str, str],
                        plots_dir: Path, output_format: str,
                        config: Dict[str, Any]) -> None:
    """Create log-log scale comparison plots."""
    # Filter positive values for log plots
    log_data = df[(df['observed_tt'] > 0) & (df['synthetic_tt'] > 0)].copy()
    
    if log_data.empty:
        logger.warning("No positive travel times found for log-log plots")
        return
    
    p_data = log_data[log_data['phase'] == 'P']
    s_data = log_data[log_data['phase'] == 'S']
    
    fig, axes = create_multi_panel_figure(1, 2, figsize=(16, 8),
                                        title='Travel Time Comparison: Log-Log Scale')
    
    # P-wave log-log plot
    if not p_data.empty:
        axes[0].loglog(p_data['observed_tt'], p_data['synthetic_tt'], 'o', 
                      alpha=0.6, markersize=4, color=colors['p_wave_dark'])
        
        # Add 1:1 line
        obs_range = [p_data['observed_tt'].min(), p_data['observed_tt'].max()]
        axes[0].loglog(obs_range, obs_range, 'r--', lw=2, label='1:1 line')
        
        axes[0].set_xlabel('Observed Travel Time (s)')
        axes[0].set_ylabel('Synthetic Travel Time (s)')
        axes[0].set_title('P-wave: Log-Log Scale')
        axes[0].legend()
        axes[0].grid(True, which="both", ls='--', alpha=0.3)
    
    # S-wave log-log plot
    if not s_data.empty:
        axes[1].loglog(s_data['observed_tt'], s_data['synthetic_tt'], 'o', 
                      alpha=0.6, markersize=4, color=colors['s_wave_dark'])
        
        # Add 1:1 line
        obs_range = [s_data['observed_tt'].min(), s_data['observed_tt'].max()]
        axes[1].loglog(obs_range, obs_range, 'r--', lw=2, label='1:1 line')
        
        axes[1].set_xlabel('Observed Travel Time (s)')
        axes[1].set_ylabel('Synthetic Travel Time (s)')
        axes[1].set_title('S-wave: Log-Log Scale')
        axes[1].legend()
        axes[1].grid(True, which="both", ls='--', alpha=0.3)
    
    save_figure(fig, plots_dir / 'synthetic_vs_observed_loglog', output_format)


def create_ln_ratio_plots(df: pd.DataFrame, colors: Dict[str, str],
                         plots_dir: Path, output_format: str,
                         config: Dict[str, Any]) -> None:
    """Create logarithmic ratio analysis plots."""
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
    
    # Create histogram plots
    fig, axes = create_multi_panel_figure(2, 2, figsize=(16, 12),
                                        title='Logarithmic Ratio Analysis: ln(Synthetic/Observed)')
    
    bins = config['travel_time_analysis'].get('histogram_bins', 50)
    
    # P-wave ln(ratio) histogram
    if not p_data.empty:
        axes[0].hist(p_data['ln_ratio'], bins=bins, alpha=0.7, 
                    color=colors['p_wave_light'], edgecolor='black', linewidth=0.5)
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='ln(ratio)=0')
        axes[0].axvline(p_data['ln_ratio'].mean(), color='orange', linestyle='-', linewidth=2,
                       label=f"Mean: {p_data['ln_ratio'].mean():.4f}")
        axes[0].set_xlabel('ln(Synthetic/Observed)')
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
        axes[1].set_xlabel('ln(Synthetic/Observed)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('S-wave: ln(Ratio) Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # P-wave ln(ratio) vs observed time
    if not p_data.empty:
        axes[2].scatter(p_data['observed_tt'], p_data['ln_ratio'], 
                       alpha=0.6, s=15, color=colors['p_wave_dark'])
        axes[2].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[2].set_xlabel('Observed Travel Time (s)')
        axes[2].set_ylabel('ln(Synthetic/Observed)')
        axes[2].set_title('P-wave: ln(Ratio) vs Observed Time')
        axes[2].grid(True, alpha=0.3)
    
    # S-wave ln(ratio) vs observed time
    if not s_data.empty:
        axes[3].scatter(s_data['observed_tt'], s_data['ln_ratio'], 
                       alpha=0.6, s=15, color=colors['s_wave_dark'])
        axes[3].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[3].set_xlabel('Observed Travel Time (s)')
        axes[3].set_ylabel('ln(Synthetic/Observed)')
        axes[3].set_title('S-wave: ln(Ratio) vs Observed Time')
        axes[3].grid(True, alpha=0.3)
    
    save_figure(fig, plots_dir / 'ln_ratio_analysis', output_format)


def generate_travel_time_report(df: pd.DataFrame, stats: Dict[str, Any], 
                              config: Dict[str, Any]) -> None:
    """
    Generate comprehensive travel time analysis report.
    
    Args:
        df: Travel time dataframe
        stats: Statistics dictionary
        config: Configuration dictionary
    """
    output_dir = Path(config['travel_time_analysis']['output_dir'])
    reports_dir = output_dir / 'reports'
    ensure_directory_exists(reports_dir)
    
    report_file = reports_dir / 'travel_time_analysis_report.txt'
    
    report_lines = [
        "TRAVEL TIME RESIDUAL ANALYSIS REPORT",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Input file: {config['travel_time_analysis']['input_csv']}",
        "",
        "SUMMARY",
        "-" * 20,
        f"Total data points: {stats['overall']['count']}",
        f"Mean residual: {stats['overall']['mean_residual']:.4f} s",
        f"RMS residual: {stats['overall']['rms_residual']:.4f} s",
        f"Standard deviation: {stats['overall']['std_residual']:.4f} s",
        f"Mean absolute error: {stats['overall']['mae_residual']:.4f} s",
        ""
    ]
    
    # Phase-specific statistics
    for phase in ['P', 'S']:
        phase_key = f'{phase.lower()}_wave'
        if phase_key in stats:
            phase_stats = stats[phase_key]
            report_lines.extend([
                f"{phase}-WAVE STATISTICS",
                "-" * 30,
                f"Data points: {phase_stats['count']}",
                f"Mean residual: {phase_stats['mean_residual']:.4f} s",
                f"RMS residual: {phase_stats['rms_residual']:.4f} s",
                f"Mean absolute error: {phase_stats['mae_residual']:.4f} s",
                f"Travel time correlation: {phase_stats['correlation']:.4f}",
                ""
            ])
            
            # Add ln ratio statistics if available
            if 'ln_ratio' in phase_stats:
                ln_stats = phase_stats['ln_ratio']
                report_lines.extend([
                    f"Logarithmic ratio analysis:",
                    f"  Mean ln(ratio): {ln_stats['mean']:.4f}",
                    f"  Std ln(ratio): {ln_stats['std']:.4f}",
                    f"  Equivalent % change: {ln_stats['equivalent_percent_change']:.2f}%",
                    ""
                ])
    
    # Station and event summaries
    report_lines.extend([
        "STATION SUMMARY",
        "-" * 20,
        f"Unique stations: {stats['station_summary']['unique_stations']}",
        "",
        "EVENT SUMMARY", 
        "-" * 20,
        f"Unique events: {stats['event_summary']['unique_events']}",
        ""
    ])
    
    # Write report
    write_text_file(report_file, '\n'.join(report_lines))
    logger.info(f"Analysis report saved to: {report_file}")