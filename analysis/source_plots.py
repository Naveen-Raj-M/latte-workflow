"""
Plotting functions for source correction analysis.

This module contains all plotting functions for visualizing
hypocenter corrections and method comparisons. Locations are longitude and latitude (degrees)
and depth (km); maps are drawn at true distance. Shifts are east, north and depth components in km.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from utils.plot_utils import save_figure, create_multi_panel_figure
from .basin_boundaries import plot_basin_boundaries
from .velocity_plots import map_aspect, LON_LABEL, LAT_LABEL, DEPTH_LABEL

logger = logging.getLogger(__name__)


def _get_labels(comparison_mode: str, config: Dict[str, Any]) -> Dict[str, str]:
    """
    Get appropriate labels based on comparison mode.
    
    Args:
        comparison_mode: 'ground_truth' or 'method_comparison'
        config: Configuration dictionary
        
    Returns:
        Dictionary of labels
    """
    if comparison_mode == 'method_comparison':
        method1_name = config['source_analysis'].get('method1_name', 'Method 1')
        method2_name = config['source_analysis'].get('method2_name', 'Method 2')
        return {
            'difference': 'Difference',
            'shift': 'Difference',
            'title_suffix': f'{method2_name} - {method1_name}',
            'original': method1_name,
            'corrected': method2_name
        }
    else:
        return {
            'difference': 'Correction',
            'shift': 'Shift',
            'title_suffix': 'Corrections',
            'original': 'Original',
            'corrected': 'Corrected'
        }


def create_correction_histograms(df: pd.DataFrame, colors: Dict[str, str],
                                plots_dir: Path, output_format: str,
                                use_log: bool, config: Dict[str, Any],
                                comparison_mode: str = 'ground_truth') -> None:
    """
    Create enhanced histograms for all correction types.
    
    Args:
        df: Dataframe with corrections
        colors: Color scheme dictionary
        plots_dir: Output directory for plots
        output_format: Output format (png, pdf, svg)
        use_log: Whether to use log scale
        config: Configuration dictionary
        comparison_mode: 'ground_truth' or 'method_comparison'
    """
    labels = _get_labels(comparison_mode, config)
    params_to_process = config['source_analysis'].get('parameters_to_process', ['slon', 'slat', 'sdepth', 'st0'])

    corrections = []
    if 'st0' in params_to_process:
        corrections.append(('delta_time_s', f'Origin Time {labels["shift"]} (s)', '#2E86AB'))
    corrections.extend([
        ('delta_depth_km', f'Depth {labels["difference"]} (km)', '#A23B72'),
        ('delta_east_km', f'East {labels["difference"]} (km)', '#F18F01'),
        ('delta_north_km', f'North {labels["difference"]} (km)', '#C73E1D'),
        ('delta_horiz_km', f'Horizontal {labels["shift"]} (km)', '#6A994E'),
    ])
    
    bins = config['source_analysis'].get('histogram_bins', 50)
    suffix = "_log" if use_log else ""
    scale_note = " (Log Scale)" if use_log else ""
    
    for col, xlabel, color in corrections:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = df[col].dropna()
        
        ax.hist(data, bins=bins, alpha=0.8, color=color, 
               edgecolor='black', linewidth=0.8)
        
        # Add vertical lines for mean and median
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='darkred', linestyle='--', 
                  linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='darkblue', linestyle='--', 
                  linewidth=2, label=f'Median: {median_val:.3f}')
        
        # Add statistics text box
        stats_text = (f'N: {len(data)}\n'
                     f'Std: {data.std():.3f}\n'
                     f'Mean: {mean_val:.3f}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor='gray', alpha=0.9))
        
        ax.set_title(f'Distribution of {xlabel}{scale_note}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=11)
        
        if use_log:
            ax.set_yscale('log')
            ax.set_ylabel('Frequency (log scale)', fontsize=11)
        else:
            ax.set_ylabel('Frequency', fontsize=11)
        
        ax.legend(frameon=True, fancybox=False, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        fig.tight_layout()
        save_figure(fig, plots_dir / f"hist_{col}{suffix}", output_format)


def create_scatter_relationships(df: pd.DataFrame, colors: Dict[str, str],
                                plots_dir: Path, output_format: str,
                                use_log: bool, config: Dict[str, Any],
                                comparison_mode: str = 'ground_truth') -> None:
    """
    Create scatter plots showing correction relationships.
    
    Args:
        df: Dataframe with corrections
        colors: Color scheme dictionary
        plots_dir: Output directory for plots
        output_format: Output format (png, pdf, svg)
        use_log: Whether to use log scale
        config: Configuration dictionary
        comparison_mode: 'ground_truth' or 'method_comparison'
    """
    labels = _get_labels(comparison_mode, config)
    suffix = "_log" if use_log else ""
    scale_note = " (Log Scale)" if use_log else ""
    params_to_process = config['source_analysis'].get('parameters_to_process', ['slon', 'slat', 'sdepth', 'st0'])
    has_st0 = 'st0' in params_to_process

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Depth vs horizontal shift
    axes[0, 0].scatter(df['event_depth_km'], df['delta_horiz_km'],
                      alpha=0.6, s=20, color='#2E86AB',
                      edgecolors='black', linewidth=0.3)
    axes[0, 0].set_xlabel('Original Depth (km)', fontsize=10)
    axes[0, 0].set_ylabel(f'Horizontal {labels["shift"]} (km)', fontsize=10)
    axes[0, 0].set_title(f'Horizontal {labels["difference"]} vs Depth{scale_note}',
                        fontsize=11, fontweight='bold')
    if use_log:
        axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Depth vs depth correction
    axes[0, 1].scatter(df['event_depth_km'], df['delta_depth_km'],
                      alpha=0.6, s=20, color='#A23B72',
                      edgecolors='black', linewidth=0.3)
    axes[0, 1].set_xlabel('Original Depth (km)', fontsize=10)
    axes[0, 1].set_ylabel(f'Depth {labels["difference"]} (km)', fontsize=10)
    axes[0, 1].set_title(f'Depth {labels["difference"]} vs Original Depth{scale_note}',
                        fontsize=11, fontweight='bold')
    if use_log:
        axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Horizontal vs time shift (if st0 processed) or east vs depth (otherwise)
    if has_st0:
        axes[1, 0].scatter(df['delta_horiz_km'], df['delta_time_s'].abs(),
                          alpha=0.6, s=20, color='#6A994E',
                          edgecolors='black', linewidth=0.3)
        axes[1, 0].set_xlabel(f'Horizontal {labels["shift"]} (km)', fontsize=10)
        axes[1, 0].set_ylabel(f'|Time {labels["shift"]}| (s)', fontsize=10)
        axes[1, 0].set_title(f'Spatial vs Temporal {labels["difference"]}s{scale_note}',
                            fontsize=11, fontweight='bold')
        if use_log:
            axes[1, 0].set_xscale('log')
            axes[1, 0].set_yscale('log')
    else:
        axes[1, 0].scatter(df['delta_east_km'], df['delta_depth_km'],
                          alpha=0.6, s=20, color='#6A994E',
                          edgecolors='black', linewidth=0.3)
        axes[1, 0].set_xlabel(f'East {labels["difference"]} (km)', fontsize=10)
        axes[1, 0].set_ylabel(f'Depth {labels["difference"]} (km)', fontsize=10)
        axes[1, 0].set_title(f'East vs Depth {labels["difference"]}s{scale_note}',
                            fontsize=11, fontweight='bold')
        axes[1, 0].axhline(0, color='gray', linestyle='--', linewidth=1)
        axes[1, 0].axvline(0, color='gray', linestyle='--', linewidth=1)
        if use_log:
            axes[1, 0].set_xscale('log')
            axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # East vs north corrections
    axes[1, 1].scatter(df['delta_east_km'], df['delta_north_km'],
                      alpha=0.6, s=20, color='#C73E1D',
                      edgecolors='black', linewidth=0.3)
    axes[1, 1].set_xlabel(f'East {labels["difference"]} (km)', fontsize=10)
    axes[1, 1].set_ylabel(f'North {labels["difference"]} (km)', fontsize=10)
    axes[1, 1].set_title(f'East vs North {labels["difference"]}s{scale_note}',
                        fontsize=11, fontweight='bold')
    if use_log:
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axes[1, 1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[1, 1].axvline(0, color='gray', linestyle='--', linewidth=1)
    
    fig.tight_layout()
    save_figure(fig, plots_dir / f"scatter_relationships{suffix}", output_format)


# Planes of the geographic plots: (name, horizontal column, vertical column, axis labels)
GEOGRAPHIC_PLANES = {
    'map': ('longitude', 'latitude', LON_LABEL, LAT_LABEL),
    'lon_depth': ('longitude', 'depth_km', LON_LABEL, DEPTH_LABEL),
    'lat_depth': ('latitude', 'depth_km', LAT_LABEL, DEPTH_LABEL),
}


def _single_geographic_plot(df: pd.DataFrame, colors: Dict[str, str],
                         plots_dir: Path, output_format: str,
                         plane: str,
                         config: Dict[str, Any],
                         comparison_mode: str = 'ground_truth',
                         basin_data: Optional[Dict] = None,
                         basin_config: Optional[Dict] = None) -> None:
    """
    Create a plot of original and corrected locations joined by correction vectors.

    Args:
        df: Dataframe with corrections
        colors: Color scheme dictionary
        plots_dir: Output directory for plots
        plane: 'map' (longitude-latitude, at true distance), 'lon_depth' or 'lat_depth'
        output_format: Output format (png, pdf, svg)
        config: Configuration dictionary
        comparison_mode: 'ground_truth' or 'method_comparison'
    """
    labels = _get_labels(comparison_mode, config)
    col_h, col_v, xlabel, ylabel = GEOGRAPHIC_PLANES[plane]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot original locations
    ax.scatter(df[f'event_{col_h}'], df[f'event_{col_v}'],
              c='#3498db', s=40, alpha=0.5, label=f'{labels["original"]} Locations',
              zorder=2, edgecolors='black', linewidth=0.5)

    # Plot corrected locations
    ax.scatter(df[f'corrected_event_{col_h}'], df[f'corrected_event_{col_v}'],
              c='#e74c3c', s=40, alpha=0.5, label=f'{labels["corrected"]} Locations',
              zorder=3, edgecolors='black', linewidth=0.5)

    # Draw correction vectors (subsample if too many)
    max_arrows = config['source_analysis'].get('max_arrows_map', 5000)
    sample_df = df.sample(min(max_arrows, len(df))) if len(df) > max_arrows else df

    for _, row in sample_df.iterrows():
        ax.arrow(row[f'event_{col_h}'], row[f'event_{col_v}'],
                row[f'corrected_event_{col_h}'] - row[f'event_{col_h}'],
                row[f'corrected_event_{col_v}'] - row[f'event_{col_v}'],
                head_width=0, head_length=0, fc='#2c3e50',
                ec='#2c3e50', alpha=0.4,
                zorder=1, linewidth=0.8)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'Geographic Distribution of Source {labels["difference"]}s\n{labels["title_suffix"]}',
                fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    if plane == 'map':
        ax.set_aspect(map_aspect(df['event_latitude'].to_numpy(dtype=float)), adjustable='box')
        if basin_data:
            plot_basin_boundaries(ax, basin_data, basin_config or {})
    else:
        ax.invert_yaxis()

    fig.tight_layout()
    save_figure(fig, plots_dir / f'geographic_corrections_{plane}', output_format)


def create_geographic_map(df: pd.DataFrame, colors: Dict[str, str],
                         plots_dir: Path, output_format: str,
                         config: Dict[str, Any],
                         comparison_mode: str = 'ground_truth',
                         basin_data: Optional[Dict] = None,
                         basin_config: Optional[Dict] = None) -> None:
    """
    Create a map and two depth sections showing correction vectors.

    Args:
        df: Dataframe with corrections
        colors: Color scheme dictionary
        plots_dir: Output directory for plots
        output_format: Output format (png, pdf, svg)
        config: Configuration dictionary
        comparison_mode: 'ground_truth' or 'method_comparison'
        basin_data: Basin boundary data loaded from load_basin_boundaries()
        basin_config: Basin boundary configuration for styling
    """
    _single_geographic_plot(df, colors, plots_dir, output_format, 'map', config, comparison_mode,
                            basin_data=basin_data, basin_config=basin_config)
    _single_geographic_plot(df, colors, plots_dir, output_format, 'lat_depth', config, comparison_mode)
    _single_geographic_plot(df, colors, plots_dir, output_format, 'lon_depth', config, comparison_mode)


def create_depth_analysis(df: pd.DataFrame, colors: Dict[str, str],
                         plots_dir: Path, output_format: str,
                         use_log: bool, config: Dict[str, Any],
                         comparison_mode: str = 'ground_truth') -> None:
    """
    Create comprehensive depth analysis plots.
    
    Args:
        df: Dataframe with corrections
        colors: Color scheme dictionary
        plots_dir: Output directory for plots
        output_format: Output format (png, pdf, svg)
        use_log: Whether to use log scale
        config: Configuration dictionary
        comparison_mode: 'ground_truth' or 'method_comparison'
    """
    labels = _get_labels(comparison_mode, config)
    suffix = "_log" if use_log else ""
    scale_note = " (Log Scale)" if use_log else ""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    original_depth = df['event_depth_km']
    corrected_depth = df['corrected_event_depth_km']
    
    # Original vs corrected depth
    axes[0].scatter(original_depth, 
                   corrected_depth,
                   alpha=0.6, s=20, color='#2E86AB',
                   edgecolors='black', linewidth=0.3)
    
    if use_log:
        min_depth = max(0.01, min(original_depth.min(),
                                 corrected_depth.min()))
        max_depth = max(original_depth.max(),
                       corrected_depth.max())
        lims = [min_depth, max_depth]
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
    else:
        lims = [min(original_depth.min(),
                   corrected_depth.min()),
               max(original_depth.max(),
                   corrected_depth.max())]
    
    axes[0].plot(lims, lims, 'r--', alpha=0.7, linewidth=2, label='1:1 line')
    axes[0].set_xlabel(f'{labels["original"]} Depth (km)', fontsize=11)
    axes[0].set_ylabel(f'{labels["corrected"]} Depth (km)', fontsize=11)
    axes[0].set_title(f'{labels["original"]} vs {labels["corrected"]} Depths{scale_note}', 
                     fontsize=12, fontweight='bold')
    axes[0].legend(frameon=True, fancybox=False, edgecolor='black')
    axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Depth correction by depth bins
    df_copy = df.copy()
    n_bins = config['source_analysis'].get('depth_bins', 10)
    df_copy['depth_bin'] = pd.cut(df_copy['event_depth_km'], bins=n_bins)
    
    depth_stats = df_copy.groupby('depth_bin')['delta_depth_km'].agg(
        ['mean', 'std', 'count']
    )
    
    min_events = config['source_analysis'].get('min_events_per_bin', 5)
    depth_stats = depth_stats[depth_stats['count'] >= min_events]
    
    if not depth_stats.empty:
        x_pos = range(len(depth_stats))
        axes[1].bar(x_pos, depth_stats['mean'], yerr=depth_stats['std'],
                   capsize=5, alpha=0.8, color='#A23B72',
                   edgecolor='black', linewidth=0.8)
        axes[1].set_xlabel('Depth Bin', fontsize=11)
        axes[1].set_ylabel(f'Mean Depth {labels["difference"]} (km)', fontsize=11)
        axes[1].set_title(f'Depth {labels["difference"]} by Depth Range{scale_note}', 
                         fontsize=12, fontweight='bold')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(
            [f'{int(interval.left)}-{int(interval.right)}' 
             for interval in depth_stats.index],
            rotation=45, ha='right'
        )
        if use_log:
            axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
        axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
    
    fig.tight_layout()
    save_figure(fig, plots_dir / f"depth_analysis{suffix}", output_format)


def create_correction_summary(df: pd.DataFrame, colors: Dict[str, str],
                             plots_dir: Path, output_format: str,
                             use_log: bool, config: Dict[str, Any],
                             comparison_mode: str = 'ground_truth') -> None:
    """
    Create summary box plot of all corrections.
    
    Args:
        df: Dataframe with corrections
        colors: Color scheme dictionary
        plots_dir: Output directory for plots
        output_format: Output format (png, pdf, svg)
        use_log: Whether to use log scale
        config: Configuration dictionary
        comparison_mode: 'ground_truth' or 'method_comparison'
    """
    labels = _get_labels(comparison_mode, config)
    suffix = "_log" if use_log else ""
    scale_note = " (Log Scale)" if use_log else ""
    params_to_process = config['source_analysis'].get('parameters_to_process', ['slon', 'slat', 'sdepth', 'st0'])

    fig, ax = plt.subplots(figsize=(12, 6))

    correction_data = []
    plot_labels = []
    if 'st0' in params_to_process:
        correction_data.append(df['delta_time_s'].abs().dropna())
        plot_labels.append('|Time| (s)')
    correction_data.extend([
        df['delta_depth_km'].abs().dropna(),
        df['delta_east_km'].abs().dropna(),
        df['delta_north_km'].abs().dropna(),
        df['delta_horiz_km'].dropna(),
    ])
    plot_labels.extend(['|Depth| (km)', '|East| (km)', '|North| (km)', 'Horiz (km)'])
    
    bp = ax.boxplot(correction_data, labels=plot_labels, patch_artist=True,
                   showmeans=True, meanline=True,
                   boxprops=dict(linewidth=1.5, edgecolor='black'),
                   whiskerprops=dict(linewidth=1.5, color='black'),
                   capprops=dict(linewidth=1.5, color='black'),
                   medianprops=dict(linewidth=2, color='darkred'),
                   meanprops=dict(linewidth=2, color='darkblue', linestyle='--'))
    
    all_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    plot_colors = all_colors if 'st0' in params_to_process else all_colors[1:]
    for patch, color in zip(bp['boxes'], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel(f'{labels["difference"]} Magnitude', fontsize=11)
    ax.set_title(f'Summary of Source {labels["difference"]}s\n{labels["title_suffix"]}{scale_note}', 
                fontsize=12, fontweight='bold')
    
    if use_log:
        ax.set_yscale('log')
    
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    fig.tight_layout()
    save_figure(fig, plots_dir / f"correction_summary{suffix}", output_format)


def create_temporal_corrections(df: pd.DataFrame, colors: Dict[str, str],
                               plots_dir: Path, output_format: str,
                               use_log: bool, config: Dict[str, Any],
                               comparison_mode: str = 'ground_truth') -> None:
    """
    Create time series of correction magnitudes.
    
    Args:
        df: Dataframe with corrections
        colors: Color scheme dictionary
        plots_dir: Output directory for plots
        output_format: Output format (png, pdf, svg)
        use_log: Whether to use log scale
        config: Configuration dictionary
        comparison_mode: 'ground_truth' or 'method_comparison'
    """
    labels = _get_labels(comparison_mode, config)
    suffix = "_log" if use_log else ""
    scale_note = " (Log Scale)" if use_log else ""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    df_sorted = df.sort_values('origin_time_s')
    
    # Convert origin time to hours for better readability
    time_hours = (df_sorted['origin_time_s'] - df_sorted['origin_time_s'].min()) / 3600
    
    # Spatial corrections over time
    axes[0].plot(time_hours, df_sorted['delta_horiz_km'],
                'o-', alpha=0.6, markersize=3, label='Horizontal',
                color='#2E86AB', linewidth=1)
    axes[0].plot(time_hours, df_sorted['delta_depth_km'].abs(),
                'o-', alpha=0.6, markersize=3, label='|Depth|',
                color='#A23B72', linewidth=1)
    axes[0].set_ylabel(f'Spatial {labels["difference"]} (km)', fontsize=11)
    axes[0].set_title(f'Spatial {labels["difference"]}s Over Time\n{labels["title_suffix"]}{scale_note}', 
                     fontsize=12, fontweight='bold')
    if use_log:
        axes[0].set_yscale('log')
    axes[0].legend(frameon=True, fancybox=False, edgecolor='black')
    axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Time corrections over time
    axes[1].plot(time_hours, df_sorted['delta_time_s'].abs(),
                'o-', alpha=0.6, markersize=3, color='#C73E1D', linewidth=1)
    axes[1].set_ylabel(f'|Time {labels["difference"]}| (s)', fontsize=11)
    axes[1].set_xlabel('Time Since First Event (hours)', fontsize=11)
    axes[1].set_title(f'Temporal {labels["difference"]}s Over Time{scale_note}', 
                     fontsize=12, fontweight='bold')
    if use_log:
        axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    fig.tight_layout()
    save_figure(fig, plots_dir / f"temporal_corrections{suffix}", output_format)