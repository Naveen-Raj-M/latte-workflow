"""
Short distance analysis plotting module.

This module creates all short distance analysis visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, Any

from utils.io_utils import ensure_directory_exists
from utils.plot_utils import get_color_scheme, save_figure, create_multi_panel_figure
from utils.distance_utils import get_distance_column_name, get_distance_label, get_output_suffix

logger = logging.getLogger(__name__)


def create_short_distance_plots(df: pd.DataFrame, stats: Dict[str, Any],
                               config: Dict[str, Any], distance_type: str) -> None:
    """
    Create comprehensive short distance analysis plots.
    
    Args:
        df: Filtered dataframe with short distances
        stats: Statistics dictionary
        config: Configuration dictionary
        distance_type: "hypocentral" or "epicentral"
    """
    output_dir = Path(config['short_distance_analysis']['output_dir'])
    coordinate_type = config['short_distance_analysis']['coordinate_type']
    comparison_mode = config['short_distance_analysis'].get('comparison_mode', 'ground_truth')
    
    # Create subdirectory
    suffix = get_output_suffix(distance_type)
    mode_suffix = 'comparison' if comparison_mode == 'method_comparison' else 'ground_truth'
    plots_dir = output_dir / 'plots' / f'{suffix}_{coordinate_type}_{mode_suffix}'
    ensure_directory_exists(plots_dir)
    
    colors = get_color_scheme(config['analysis'])
    output_format = config['analysis'].get('output_format', 'png')
    
    logger.info(f"Creating short distance analysis plots for {distance_type} distance")
    
    # Overview map with residuals
    create_overview_map(df, colors, plots_dir, output_format, config, distance_type)
    
    # Residual histograms
    if config['short_distance_analysis'].get('enable_residual_histograms', True):
        create_residual_histograms(df, stats, colors, plots_dir, output_format, config, distance_type)
    
    # Distance vs residual scatter plots
    create_distance_residual_plots(df, colors, plots_dir, output_format, config, distance_type)
    
    # Cluster zoom plots
    if config['short_distance_analysis'].get('enable_zoom_clusters', False):
        create_cluster_zoom_plots(df, stats, colors, plots_dir, output_format, config, distance_type)
    
    logger.info(f"Short distance analysis plots saved to: {plots_dir}")


def create_overview_map(df: pd.DataFrame, colors: Dict[str, str],
                       plots_dir: Path, output_format: str,
                       config: Dict[str, Any], distance_type: str) -> None:
    """Create overview map showing events, stations, and residuals."""
    coordinate_type = config['short_distance_analysis']['coordinate_type']
    comparison_mode = config['short_distance_analysis'].get('comparison_mode', 'ground_truth')
    distance_cutoff = config['short_distance_analysis']['distance_cutoff_km']
    
    # Title suffix
    if comparison_mode == 'method_comparison':
        method1_name = config['short_distance_analysis'].get('method1_name', 'Method 1')
        method2_name = config['short_distance_analysis'].get('method2_name', 'Method 2')
        title_suffix = f'\n({method1_name} - {method2_name})'
    else:
        title_suffix = ''
    
    distance_label = get_distance_label(distance_type)
    
    # Determine coordinate columns
    if coordinate_type == 'transformed':
        x_event = 'X_grid_km_y' if 'X_grid_km_y' in df.columns else 'X_grid_km'
        y_event = 'Y_grid_km_y' if 'Y_grid_km_y' in df.columns else 'Y_grid_km'
        x_station = 'X_grid_km_x' if 'X_grid_km_x' in df.columns else 'X_grid_km'
        y_station = 'Y_grid_km_x' if 'Y_grid_km_x' in df.columns else 'Y_grid_km'
    else:
        x_event = 'x_km_y' if 'x_km_y' in df.columns else 'x_km'
        y_event = 'y_km_y' if 'y_km_y' in df.columns else 'y_km'
        x_station = 'x_km_x' if 'x_km_x' in df.columns else 'x_km'
        y_station = 'y_km_x' if 'y_km_x' in df.columns else 'y_km'
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot travel paths colored by residuals
    if config['short_distance_analysis'].get('show_travel_paths', True):
        max_paths = 5000
        df_sample = df.sample(n=min(len(df), max_paths), random_state=42)
        
        for _, row in df_sample.iterrows():
            color = colors['positive_residual'] if row['residual_s'] > 0 else colors['negative_residual']
            alpha = min(0.3, abs(row['residual_s']) / 2.0)
            ax.plot([row[x_event], row[x_station]], 
                   [row[y_event], row[y_station]],
                   color=color, alpha=alpha, linewidth=0.5)
    
    # Plot events
    if config['short_distance_analysis'].get('show_events', True):
        events_unique = df.groupby('event_id').first()
        ax.scatter(events_unique[x_event], events_unique[y_event],
                  c=colors['event_color'], marker='*', s=100,
                  edgecolors='black', linewidths=0.5, label='Events', zorder=5)
    
    # Plot stations
    if config['short_distance_analysis'].get('show_stations', True):
        stations_unique = df.groupby('station').first()
        ax.scatter(stations_unique[x_station], stations_unique[y_station],
                  c=colors['station_color'], marker='^', s=100,
                  edgecolors='black', linewidths=0.5, label='Stations', zorder=5)
    
    ax.set_xlabel(f'X ({coordinate_type}) [km]')
    ax.set_ylabel(f'Y ({coordinate_type}) [km]')
    ax.set_title(f'Short {distance_label} Distance Analysis Overview (≤{distance_cutoff} km){title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    save_figure(fig, plots_dir / 'overview_map', output_format)


def create_residual_histograms(df: pd.DataFrame, stats: Dict[str, Any],
                              colors: Dict[str, str], plots_dir: Path,
                              output_format: str, config: Dict[str, Any],
                              distance_type: str) -> None:
    """Create residual distribution histograms."""
    comparison_mode = config['short_distance_analysis'].get('comparison_mode', 'ground_truth')
    distance_cutoff = config['short_distance_analysis']['distance_cutoff_km']
    
    if comparison_mode == 'method_comparison':
        method1_name = config['short_distance_analysis'].get('method1_name', 'Method 1')
        method2_name = config['short_distance_analysis'].get('method2_name', 'Method 2')
        title_suffix = f'\n({method1_name} - {method2_name})'
    else:
        title_suffix = ''
    
    distance_label = get_distance_label(distance_type)
    
    p_data = df[df['phase'] == 'P']
    s_data = df[df['phase'] == 'S']
    
    fig, axes = create_multi_panel_figure(2, 2, figsize=(16, 12),
                                        title=f'Residual Histograms ({distance_label} ≤{distance_cutoff} km){title_suffix}')
    
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    bins = config['short_distance_analysis'].get('histogram_bins', 50)
    
    # Overall histogram
    if not df.empty:
        axes_flat[0].hist(df['residual_s'], bins=bins, alpha=0.7,
                        color=colors['neutral_color'], edgecolor='black')
        axes_flat[0].axvline(df['residual_s'].mean(), color='red', linestyle='--',
                           linewidth=2, label=f"Mean: {df['residual_s'].mean():.3f} s")
        axes_flat[0].axvline(0, color='black', linestyle='-', linewidth=2)
        axes_flat[0].set_xlabel('Residual [s]')
        axes_flat[0].set_ylabel('Frequency')
        axes_flat[0].set_title(f"Overall (N={len(df)})")
        axes_flat[0].legend()
        axes_flat[0].grid(True, alpha=0.3)
    
    # P-wave histogram
    if not p_data.empty:
        axes_flat[1].hist(p_data['residual_s'], bins=bins, alpha=0.7,
                        color=colors['p_wave_dark'], edgecolor='black')
        axes_flat[1].axvline(p_data['residual_s'].mean(), color='red', linestyle='--',
                           linewidth=2, label=f"Mean: {p_data['residual_s'].mean():.3f} s")
        axes_flat[1].axvline(0, color='black', linestyle='-', linewidth=2)
        axes_flat[1].set_xlabel('Residual [s]')
        axes_flat[1].set_ylabel('Frequency')
        axes_flat[1].set_title(f"P-wave (N={len(p_data)})")
        axes_flat[1].legend()
        axes_flat[1].grid(True, alpha=0.3)
    
    # S-wave histogram
    if not s_data.empty:
        axes_flat[2].hist(s_data['residual_s'], bins=bins, alpha=0.7,
                        color=colors['s_wave_dark'], edgecolor='black')
        axes_flat[2].axvline(s_data['residual_s'].mean(), color='red', linestyle='--',
                           linewidth=2, label=f"Mean: {s_data['residual_s'].mean():.3f} s")
        axes_flat[2].axvline(0, color='black', linestyle='-', linewidth=2)
        axes_flat[2].set_xlabel('Residual [s]')
        axes_flat[2].set_ylabel('Frequency')
        axes_flat[2].set_title(f"S-wave (N={len(s_data)})")
        axes_flat[2].legend()
        axes_flat[2].grid(True, alpha=0.3)
    
    # Comparison P vs S
    if not p_data.empty or not s_data.empty:
        if not p_data.empty:
            axes_flat[3].hist(p_data['residual_s'], bins=bins, alpha=0.6,
                            color=colors['p_wave_dark'], edgecolor='black', label='P-wave')
        if not s_data.empty:
            axes_flat[3].hist(s_data['residual_s'], bins=bins, alpha=0.6,
                            color=colors['s_wave_dark'], edgecolor='black', label='S-wave')
        axes_flat[3].axvline(0, color='black', linestyle='-', linewidth=2)
        axes_flat[3].set_xlabel('Residual [s]')
        axes_flat[3].set_ylabel('Frequency')
        axes_flat[3].set_title('P-wave vs S-wave Comparison')
        axes_flat[3].legend()
        axes_flat[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, plots_dir / 'residual_histograms', output_format)


def create_distance_residual_plots(df: pd.DataFrame, colors: Dict[str, str],
                                  plots_dir: Path, output_format: str,
                                  config: Dict[str, Any], distance_type: str) -> None:
    """Create distance vs residual scatter plots."""
    coordinate_type = config['short_distance_analysis']['coordinate_type']
    distance_col = get_distance_column_name(coordinate_type, distance_type)
    distance_label = get_distance_label(distance_type)
    comparison_mode = config['short_distance_analysis'].get('comparison_mode', 'ground_truth')
    distance_cutoff = config['short_distance_analysis']['distance_cutoff_km']
    
    if comparison_mode == 'method_comparison':
        method1_name = config['short_distance_analysis'].get('method1_name', 'Method 1')
        method2_name = config['short_distance_analysis'].get('method2_name', 'Method 2')
        title_suffix = f'\n({method1_name} - {method2_name})'
    else:
        title_suffix = ''
    
    p_data = df[df['phase'] == 'P']
    s_data = df[df['phase'] == 'S']
    
    fig, axes = create_multi_panel_figure(1, 2, figsize=(16, 6),
                                        title=f'Residuals vs {distance_label} Distance (≤{distance_cutoff} km){title_suffix}')
    
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # P-wave
    if not p_data.empty:
        axes_flat[0].scatter(p_data[distance_col], p_data['residual_s'],
                           alpha=0.6, s=20, color=colors['p_wave_dark'])
        axes_flat[0].axhline(0, color=colors['neutral_color'], linestyle='--', linewidth=2)
        axes_flat[0].set_xlabel(f'{distance_label} Distance [km]')
        axes_flat[0].set_ylabel('Residual [s]')
        axes_flat[0].set_title(f'P-wave (N={len(p_data)})')
        axes_flat[0].grid(True, alpha=0.3)
    
    # S-wave
    if not s_data.empty:
        axes_flat[1].scatter(s_data[distance_col], s_data['residual_s'],
                           alpha=0.6, s=20, color=colors['s_wave_dark'])
        axes_flat[1].axhline(0, color=colors['neutral_color'], linestyle='--', linewidth=2)
        axes_flat[1].set_xlabel(f'{distance_label} Distance [km]')
        axes_flat[1].set_ylabel('Residual [s]')
        axes_flat[1].set_title(f'S-wave (N={len(s_data)})')
        axes_flat[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, plots_dir / 'distance_vs_residual', output_format)


def create_cluster_zoom_plots(df: pd.DataFrame, stats: Dict[str, Any],
                             colors: Dict[str, str], plots_dir: Path,
                             output_format: str, config: Dict[str, Any],
                             distance_type: str) -> None:
    """Create zoom plots for defined clusters."""
    clusters = config['short_distance_analysis'].get('zoom_clusters', [])
    
    if not clusters or 'clusters' not in stats:
        return
    
    coordinate_type = config['short_distance_analysis']['coordinate_type']
    
    # Determine coordinate columns
    if coordinate_type == 'transformed':
        x_event = 'X_grid_km_y' if 'X_grid_km_y' in df.columns else 'X_grid_km'
        y_event = 'Y_grid_km_y' if 'Y_grid_km_y' in df.columns else 'Y_grid_km'
        x_station = 'X_grid_km_x' if 'X_grid_km_x' in df.columns else 'X_grid_km'
        y_station = 'Y_grid_km_x' if 'Y_grid_km_x' in df.columns else 'Y_grid_km'
    else:
        x_event = 'x_km_y' if 'x_km_y' in df.columns else 'x_km'
        y_event = 'y_km_y' if 'y_km_y' in df.columns else 'y_km'
        x_station = 'x_km_x' if 'x_km_x' in df.columns else 'x_km'
        y_station = 'y_km_x' if 'y_km_x' in df.columns else 'y_km'
    
    for cluster in clusters:
        cluster_name = cluster['name']
        x_min, x_max = cluster['x_min'], cluster['x_max']
        y_min, y_max = cluster['y_min'], cluster['y_max']
        description = cluster.get('description', '')
        
        # Filter data
        cluster_data = df[
            ((df[x_event] >= x_min) & (df[x_event] <= x_max) &
             (df[y_event] >= y_min) & (df[y_event] <= y_max)) |
            ((df[x_station] >= x_min) & (df[x_station] <= x_max) &
             (df[y_station] >= y_min) & (df[y_station] <= y_max))
        ]
        
        if cluster_data.empty:
            logger.warning(f"No data in cluster {cluster_name}, skipping plot")
            continue
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot travel paths
        for _, row in cluster_data.iterrows():
            color = colors['positive_residual'] if row['residual_s'] > 0 else colors['negative_residual']
            alpha = min(0.5, abs(row['residual_s']) / 2.0)
            ax.plot([row[x_event], row[x_station]],
                   [row[y_event], row[y_station]],
                   color=color, alpha=alpha, linewidth=1.0)
        
        # Plot events
        events_unique = cluster_data.groupby('event_id').first()
        ax.scatter(events_unique[x_event], events_unique[y_event],
                  c=colors['event_color'], marker='*', s=150,
                  edgecolors='black', linewidths=0.5, label='Events', zorder=5)
        
        # Plot stations
        stations_unique = cluster_data.groupby('station').first()
        ax.scatter(stations_unique[x_station], stations_unique[y_station],
                  c=colors['station_color'], marker='^', s=150,
                  edgecolors='black', linewidths=0.5, label='Stations', zorder=5)
        
        # Cluster boundary
        ax.plot([x_min, x_max, x_max, x_min, x_min],
               [y_min, y_min, y_max, y_max, y_min],
               'k--', linewidth=2, label='Cluster Boundary')
        
        cluster_stats = stats['clusters'].get(cluster_name, {})
        stats_text = f"N = {cluster_stats.get('count', 0)}\n"
        stats_text += f"RMS = {cluster_stats.get('residual_rms', 0):.3f} s"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(f'X ({coordinate_type}) [km]')
        ax.set_ylabel(f'Y ({coordinate_type}) [km]')
        ax.set_title(f'Cluster: {cluster_name}\n{description}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(x_min - 10, x_max + 10)
        ax.set_ylim(y_min - 10, y_max + 10)
        
        plt.tight_layout()
        save_figure(fig, plots_dir / f'cluster_{cluster_name}', output_format)