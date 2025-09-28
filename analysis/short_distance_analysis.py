"""
Short distance travel time analysis module.

This module focuses on analysis of travel time residuals for short source-receiver distances,
including spatial clustering, travel path visualization, and zoom region analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime

from utils.io_utils import read_csv_file, write_text_file, ensure_directory_exists
from utils.plot_utils import (
    setup_plot_style, get_color_scheme, save_figure,
    create_multi_panel_figure, add_statistics_text
)

logger = logging.getLogger(__name__)


def analyze_short_distance_residuals(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function for short distance travel time analysis.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    logger.info("Starting short distance travel time analysis")
    
    # Setup plot style
    setup_plot_style(config['analysis'])
    
    # Load and filter data
    df_short = load_and_filter_short_distance_data(config)
    
    if df_short.empty:
        logger.warning("No data found within distance cutoff")
        return {}
    
    # Load supporting data
    station_df = read_csv_file(config['short_distance_analysis']['station_csv'])
    events_df = read_csv_file(config['short_distance_analysis']['events_csv'])
    
    # Compute statistics
    logger.info("Computing short distance statistics")
    stats = compute_short_distance_statistics(df_short, config)
    
    # Create plots
    logger.info("Creating short distance plots")
    create_short_distance_plots(df_short, station_df, events_df, stats, config)
    
    # Generate report
    logger.info("Generating short distance report")
    generate_short_distance_report(df_short, stats, config)
    
    logger.info("Short distance analysis completed")
    return stats


def load_and_filter_short_distance_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load travel time data and filter for short distances.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Filtered dataframe with short distance data
    """
    # Load travel time data
    travel_time_csv = config['short_distance_analysis']['travel_time_csv']
    station_csv = config['short_distance_analysis']['station_csv']
    events_csv = config['short_distance_analysis']['events_csv']
    
    df_tt = read_csv_file(travel_time_csv)
    df_stations = read_csv_file(station_csv)
    df_events = read_csv_file(events_csv)
    
    logger.info(f"Initial data: {len(df_tt)} travel time records")
    
    # Merge data to get coordinates
    df_merged = df_tt.merge(df_stations, on='station', how='left')
    df_merged = df_merged.merge(df_events, on='event_id', how='left')
    
    # Calculate distances
    coordinate_type = config['short_distance_analysis']['coordinate_type']
    distance_cutoff = config['short_distance_analysis']['distance_cutoff_km']
    
    if coordinate_type == "transformed":
        event_x = df_merged['X_grid_km_y']
        event_y = df_merged['Y_grid_km_y']
        event_z = df_merged['Z_grid_km_y']
        station_x = df_merged['X_grid_km_x']
        station_y = df_merged['Y_grid_km_x']
        station_z = df_merged['Z_grid_km_x']
    else:
        event_x = df_merged['x_km_y']
        event_y = df_merged['y_km_y']
        event_z = df_merged['z_km_y']
        station_x = df_merged['x_km_x']
        station_y = df_merged['y_km_x']
        station_z = df_merged['z_km_x']
    
    # Calculate 3D distance
    distance_km = np.sqrt((event_x - station_x)**2 + 
                         (event_y - station_y)**2 + 
                         (event_z - station_z)**2)
    
    df_merged[f'distance_{coordinate_type}_km'] = distance_km
    
    # Filter by distance cutoff
    df_short = df_merged[distance_km <= distance_cutoff].copy()
    
    # Apply residual filters
    min_res = config['short_distance_analysis'].get('min_residual_threshold', -5.0)
    max_res = config['short_distance_analysis'].get('max_residual_threshold', 5.0)
    
    df_short = df_short[
        (df_short['residual_s'] >= min_res) & 
        (df_short['residual_s'] <= max_res)
    ]
    
    # Apply phase filter if specified
    phase_filter = config['short_distance_analysis'].get('phase_filter')
    if phase_filter:
        df_short = df_short[df_short['phase'] == phase_filter]
    
    logger.info(f"Short distance data: {len(df_short)} records within {distance_cutoff} km")
    logger.info(f"Distance range: {df_short[f'distance_{coordinate_type}_km'].min():.3f} - {df_short[f'distance_{coordinate_type}_km'].max():.3f} km")
    
    return df_short


def compute_short_distance_statistics(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute comprehensive short distance statistics.
    
    Args:
        df: Short distance dataframe
        config: Configuration dictionary
        
    Returns:
        Dictionary of statistics
    """
    coordinate_type = config['short_distance_analysis']['coordinate_type']
    distance_col = f'distance_{coordinate_type}_km'
    
    stats = {}
    
    # Overall statistics
    residuals = df['residual_s']
    distances = df[distance_col]
    
    stats['overall'] = {
        'count': len(df),
        'unique_events': df['event_id'].nunique(),
        'unique_stations': df['station'].nunique(),
        'distance_mean': distances.mean(),
        'distance_median': distances.median(),
        'distance_std': distances.std(),
        'distance_min': distances.min(),
        'distance_max': distances.max(),
        'residual_mean': residuals.mean(),
        'residual_std': residuals.std(),
        'residual_rms': np.sqrt(np.mean(residuals**2)),
        'residual_mae': np.mean(np.abs(residuals)),
        'distance_residual_correlation': distances.corr(residuals)
    }
    
    # Phase-specific statistics
    for phase in ['P', 'S']:
        phase_data = df[df['phase'] == phase]
        if not phase_data.empty:
            phase_residuals = phase_data['residual_s']
            phase_distances = phase_data[distance_col]
            
            stats[f'{phase.lower()}_wave'] = {
                'count': len(phase_data),
                'residual_mean': phase_residuals.mean(),
                'residual_std': phase_residuals.std(),
                'residual_rms': np.sqrt(np.mean(phase_residuals**2)),
                'residual_mae': np.mean(np.abs(phase_residuals)),
                'distance_mean': phase_distances.mean(),
                'distance_residual_correlation': phase_distances.corr(phase_residuals)
            }
    
    # Station statistics
    station_stats = df.groupby('station')['residual_s'].agg([
        'count', 'mean', 'std'
    ]).round(4)
    
    stats['station_summary'] = {
        'unique_stations': len(station_stats),
        'top_5_by_count': station_stats.nlargest(5, 'count').to_dict('index'),
        'mean_measurements_per_station': station_stats['count'].mean()
    }
    
    # Event statistics
    event_stats = df.groupby('event_id')['residual_s'].agg([
        'count', 'mean', 'std'
    ]).round(4)
    
    stats['event_summary'] = {
        'unique_events': len(event_stats),
        'top_5_by_count': event_stats.nlargest(5, 'count').to_dict('index'),
        'mean_measurements_per_event': event_stats['count'].mean()
    }
    
    return stats


def create_short_distance_plots(df: pd.DataFrame, station_df: pd.DataFrame,
                               events_df: pd.DataFrame, stats: Dict[str, Any],
                               config: Dict[str, Any]) -> None:
    """
    Create comprehensive short distance analysis plots.
    
    Args:
        df: Short distance dataframe
        station_df: Station dataframe
        events_df: Events dataframe
        stats: Statistics dictionary
        config: Configuration dictionary
    """
    output_dir = Path(config['short_distance_analysis']['output_dir'])
    plots_dir = output_dir / 'plots'
    ensure_directory_exists(plots_dir)
    
    colors = get_color_scheme(config['analysis'])
    output_format = config['analysis'].get('output_format', 'png')
    
    # Travel path maps
    create_travel_path_maps(df, station_df, events_df, colors, plots_dir, output_format, config)
    
    # Residual histograms
    if config['short_distance_analysis'].get('enable_residual_histograms', True):
        create_short_distance_histograms(df, colors, plots_dir, output_format, config)
    
    # Distance correlation plots
    create_short_distance_correlation_plots(df, colors, plots_dir, output_format, config)
    
    # Zoom cluster analysis
    if config['short_distance_analysis'].get('enable_zoom_clusters', True):
        create_zoom_cluster_plots(df, station_df, events_df, colors, plots_dir, output_format, config)


def create_travel_path_maps(df: pd.DataFrame, station_df: pd.DataFrame,
                           events_df: pd.DataFrame, colors: Dict[str, str],
                           plots_dir: Path, output_format: str,
                           config: Dict[str, Any]) -> None:
    """Create travel path visualization maps."""
    coordinate_type = config['short_distance_analysis']['coordinate_type']
    
    # Get coordinate column names
    if coordinate_type == "transformed":
        x_col_station = 'X_grid_km'
        y_col_station = 'Y_grid_km'
        x_col_event = 'X_grid_km'
        y_col_event = 'Y_grid_km'
        coord_label = "Transformed"
    else:
        x_col_station = 'x_km'
        y_col_station = 'y_km'
        x_col_event = 'x_km'
        y_col_event = 'y_km'
        coord_label = "Original"
    
    # Separate P and S data
    p_data = df[df['phase'] == 'P']
    s_data = df[df['phase'] == 'S']
    
    phase_filter = config['short_distance_analysis'].get('phase_filter')
    if phase_filter:
        if phase_filter == 'P':
            fig, axes = create_multi_panel_figure(1, 2, figsize=(20, 8))
            plot_configs = [('P', 'paths'), ('P', 'residuals')]
        else:
            fig, axes = create_multi_panel_figure(1, 2, figsize=(20, 8))
            plot_configs = [('S', 'paths'), ('S', 'residuals')]
    else:
        fig, axes = create_multi_panel_figure(2, 2, figsize=(20, 16))
        plot_configs = [('P', 'paths'), ('S', 'paths'), ('P', 'residuals'), ('S', 'residuals')]
    
    fig.suptitle(f'Short Distance Travel Paths ({coord_label} Coordinates)', fontsize=18)
    
    # Create individual plots
    for idx, (phase, plot_type) in enumerate(plot_configs):
        if idx >= len(axes):
            break
            
        ax = axes[idx] if isinstance(axes, np.ndarray) else axes
        
        # Get phase data
        if phase == 'P':
            phase_data = p_data
            phase_color = colors['p_wave_dark']
        else:
            phase_data = s_data
            phase_color = colors['s_wave_dark']
        
        if phase_data.empty:
            ax.text(0.5, 0.5, f'No {phase}-wave data\nwithin distance cutoff', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{phase}-wave: {plot_type.title()}')
            continue
        
        if plot_type == 'paths':
            # Show travel paths
            show_travel_paths = config['short_distance_analysis'].get('show_travel_paths', True)
            if show_travel_paths:
                for _, row in phase_data.iterrows():
                    # Get station coordinates
                    station_info = station_df[station_df['station'] == row['station']]
                    if station_info.empty:
                        continue
                    station_info = station_info.iloc[0]
                    
                    # Get event coordinates
                    event_info = events_df[events_df['event_id'] == row['event_id']]
                    if event_info.empty:
                        continue
                    event_info = event_info.iloc[0]
                    
                    # Color by residual
                    if row['residual_s'] > 0.1:
                        line_color = colors['positive_residual']
                        alpha = 0.6
                    elif row['residual_s'] < -0.1:
                        line_color = colors['negative_residual']
                        alpha = 0.6
                    else:
                        line_color = colors['zero_line']
                        alpha = 0.8
                    
                    # Line width proportional to absolute residual
                    line_width = min(3.0, max(0.5, abs(row['residual_s']) * 1.0))
                    
                    # Draw path
                    ax.plot([event_info[x_col_event], station_info[x_col_station]], 
                           [event_info[y_col_event], station_info[y_col_station]], 
                           color=line_color, alpha=alpha, linewidth=line_width)
            
            # Show events and stations
            show_events = config['short_distance_analysis'].get('show_events', True)
            show_stations = config['short_distance_analysis'].get('show_stations', True)
            
            unique_events = phase_data['event_id'].unique()
            unique_stations = phase_data['station'].unique()
            
            if show_events:
                event_coords = []
                for event_id in unique_events:
                    event_info = events_df[events_df['event_id'] == event_id]
                    if not event_info.empty:
                        event_info = event_info.iloc[0]
                        event_coords.append([event_info[x_col_event], event_info[y_col_event]])
                
                if event_coords:
                    event_coords = np.array(event_coords)
                    ax.scatter(event_coords[:, 0], event_coords[:, 1], 
                              marker='*', s=100, color=colors['event_color'], 
                              edgecolors='black', linewidth=1, alpha=0.8, 
                              label=f'Events (N={len(unique_events)})', zorder=5)
            
            if show_stations:
                station_coords = []
                for station in unique_stations:
                    station_info = station_df[station_df['station'] == station]
                    if not station_info.empty:
                        station_info = station_info.iloc[0]
                        station_coords.append([station_info[x_col_station], station_info[y_col_station]])
                
                if station_coords:
                    station_coords = np.array(station_coords)
                    ax.scatter(station_coords[:, 0], station_coords[:, 1], 
                              marker='^', s=60, color=colors['station_color'], 
                              edgecolors='white', linewidth=1, alpha=0.8, 
                              label=f'Stations (N={len(unique_stations)})', zorder=5)
            
            if show_events or show_stations:
                ax.legend(loc='upper right')
            
        elif plot_type == 'residuals':
            # Create residual intensity map
            station_coords = []
            residual_values = []
            
            # Get mean residuals by station
            station_residuals = phase_data.groupby('station')['residual_s'].mean()
            
            for station, mean_residual in station_residuals.items():
                station_info = station_df[station_df['station'] == station]
                if not station_info.empty:
                    station_info = station_info.iloc[0]
                    station_coords.append([station_info[x_col_station], station_info[y_col_station]])
                    residual_values.append(mean_residual)
            
            if station_coords:
                station_coords = np.array(station_coords)
                scatter = ax.scatter(station_coords[:, 0], station_coords[:, 1], 
                                   c=residual_values, cmap='RdBu_r', 
                                   s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(f'{phase}-wave Mean Residual [s]', rotation=270, labelpad=20)
        
        # Common formatting
        ax.set_xlabel(f'X ({coord_label}) [km]')
        ax.set_ylabel(f'Y ({coord_label}) [km]')
        ax.set_title(f'{phase}-wave: {plot_type.title()}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Add statistics text
        if plot_type == 'paths':
            stats_text = f'N = {len(phase_data)}\nMean Res: {phase_data["residual_s"].mean():.3f}s\nRMS: {np.sqrt(np.mean(phase_data["residual_s"]**2)):.3f}s'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    save_figure(fig, plots_dir / f'short_distance_travel_paths_{coordinate_type}', output_format)


def create_short_distance_histograms(df: pd.DataFrame, colors: Dict[str, str],
                                    plots_dir: Path, output_format: str,
                                    config: Dict[str, Any]) -> None:
    """Create residual distribution histograms for short distances."""
    p_data = df[df['phase'] == 'P']
    s_data = df[df['phase'] == 'S']
    
    fig, axes = create_multi_panel_figure(2, 3, figsize=(18, 12),
                                        title='Short Distance Residual Analysis')
    
    # Overall distribution
    axes[0].hist(df['residual_s'], bins=30, alpha=0.7, 
                color=colors['neutral_color'], edgecolor='black')
    axes[0].axvline(0, color='black', linestyle='--', linewidth=2)
    axes[0].axvline(df['residual_s'].mean(), color='red', linestyle='-', linewidth=2, label='Mean')
    axes[0].set_xlabel('Residual [s]')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Overall Distribution (N={len(df)})')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # P-wave vs S-wave comparison
    if not p_data.empty and not s_data.empty:
        axes[1].hist(p_data['residual_s'], bins=20, alpha=0.6, 
                    color=colors['p_wave_light'], label=f'P-wave (N={len(p_data)})', edgecolor='black')
        axes[1].hist(s_data['residual_s'], bins=20, alpha=0.6, 
                    color=colors['s_wave_light'], label=f'S-wave (N={len(s_data)})', edgecolor='black')
        axes[1].axvline(0, color='black', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Residual [s]')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('P-wave vs S-wave Residuals')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    # Distance vs residual scatter
    coordinate_type = config['short_distance_analysis']['coordinate_type']
    distance_col = f'distance_{coordinate_type}_km'
    
    axes[2].scatter(df[distance_col], df['residual_s'], alpha=0.6, s=20, color=colors['neutral_color'])
    axes[2].axhline(0, color='black', linestyle='--', linewidth=2)
    axes[2].set_xlabel('Distance [km]')
    axes[2].set_ylabel('Residual [s]')
    axes[2].set_title('Residuals vs Distance')
    axes[2].grid(True, alpha=0.3)
    
    # Cumulative distributions
    if not p_data.empty:
        p_sorted = np.sort(p_data['residual_s'])
        p_cumulative = np.arange(1, len(p_sorted) + 1) / len(p_sorted)
        axes[3].plot(p_sorted, p_cumulative, color=colors['p_wave_dark'], linewidth=2, label='P-wave')
    
    if not s_data.empty:
        s_sorted = np.sort(s_data['residual_s'])
        s_cumulative = np.arange(1, len(s_sorted) + 1) / len(s_sorted)
        axes[3].plot(s_sorted, s_cumulative, color=colors['s_wave_dark'], linewidth=2, label='S-wave')
    
    axes[3].axvline(0, color='black', linestyle='--', linewidth=2)
    axes[3].set_xlabel('Residual [s]')
    axes[3].set_ylabel('Cumulative Probability')
    axes[3].set_title('Cumulative Distributions')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    # Q-Q plot for normality
    try:
        from scipy import stats
        stats.probplot(df['residual_s'], dist="norm", plot=axes[4])
        axes[4].set_title('Q-Q Plot (Normal Distribution)')
        axes[4].grid(True, alpha=0.3)
    except ImportError:
        axes[4].text(0.5, 0.5, 'scipy not available\nfor Q-Q plot', 
                    ha='center', va='center', transform=axes[4].transAxes)
    
    # Residual vs travel time
    axes[5].scatter(df['observed_tt'], df['residual_s'], alpha=0.6, s=20, color=colors['neutral_color'])
    axes[5].axhline(0, color='black', linestyle='--', linewidth=2)
    axes[5].set_xlabel('Observed Travel Time [s]')
    axes[5].set_ylabel('Residual [s]')
    axes[5].set_title('Residual vs Travel Time')
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, plots_dir / 'short_distance_histograms', output_format)


def create_short_distance_correlation_plots(df: pd.DataFrame, colors: Dict[str, str],
                                           plots_dir: Path, output_format: str,
                                           config: Dict[str, Any]) -> None:
    """Create correlation analysis plots for short distances."""
    coordinate_type = config['short_distance_analysis']['coordinate_type']
    distance_col = f'distance_{coordinate_type}_km'
    
    p_data = df[df['phase'] == 'P']
    s_data = df[df['phase'] == 'S']
    
    fig, axes = create_multi_panel_figure(2, 2, figsize=(16, 12),
                                        title='Short Distance Correlation Analysis')
    
    # Overall correlation
    axes[0].scatter(df[distance_col], df['residual_s'], alpha=0.6, s=20, color=colors['neutral_color'])
    axes[0].axhline(0, color='black', linestyle='--', linewidth=2)
    
    # Add trend line
    if len(df) > 1:
        z = np.polyfit(df[distance_col], df['residual_s'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df[distance_col].min(), df[distance_col].max(), 100)
        axes[0].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    
    corr = df[distance_col].corr(df['residual_s'])
    axes[0].set_xlabel(f'Distance ({coordinate_type}) [km]')
    axes[0].set_ylabel('Residual [s]')
    axes[0].set_title(f'Distance vs Residual (r = {corr:.4f})')
    axes[0].grid(True, alpha=0.3)
    
    # Phase-specific correlations
    if not p_data.empty:
        axes[1].scatter(p_data[distance_col], p_data['residual_s'], 
                       alpha=0.6, s=20, color=colors['p_wave_dark'], label='P-wave')
        p_corr = p_data[distance_col].corr(p_data['residual_s'])
        
        # P-wave trend line
        if len(p_data) > 1:
            z_p = np.polyfit(p_data[distance_col], p_data['residual_s'], 1)
            p_p = np.poly1d(z_p)
            x_trend_p = np.linspace(p_data[distance_col].min(), p_data[distance_col].max(), 100)
            axes[1].plot(x_trend_p, p_p(x_trend_p), color=colors['p_wave_light'], 
                        linestyle='--', alpha=0.8, linewidth=2)
    
    if not s_data.empty:
        axes[1].scatter(s_data[distance_col], s_data['residual_s'], 
                       alpha=0.6, s=20, color=colors['s_wave_dark'], label='S-wave')
        s_corr = s_data[distance_col].corr(s_data['residual_s'])
        
        # S-wave trend line
        if len(s_data) > 1:
            z_s = np.polyfit(s_data[distance_col], s_data['residual_s'], 1)
            p_s = np.poly1d(z_s)
            x_trend_s = np.linspace(s_data[distance_col].min(), s_data[distance_col].max(), 100)
            axes[1].plot(x_trend_s, p_s(x_trend_s), color=colors['s_wave_light'], 
                        linestyle='--', alpha=0.8, linewidth=2)
    
    axes[1].axhline(0, color='black', linestyle='--', linewidth=2)
    axes[1].set_xlabel(f'Distance ({coordinate_type}) [km]')
    axes[1].set_ylabel('Residual [s]')
    
    title_text = 'Phase-specific Correlations'
    if not p_data.empty and not s_data.empty:
        title_text += f'\nP: r={p_corr:.4f}, S: r={s_corr:.4f}'
    elif not p_data.empty:
        title_text += f'\nP: r={p_corr:.4f}'
    elif not s_data.empty:
        title_text += f'\nS: r={s_corr:.4f}'
    
    axes[1].set_title(title_text)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Absolute residuals vs distance
    axes[2].scatter(df[distance_col], np.abs(df['residual_s']), 
                   alpha=0.6, s=20, color=colors['neutral_color'])
    
    abs_corr = df[distance_col].corr(np.abs(df['residual_s']))
    axes[2].set_xlabel(f'Distance ({coordinate_type}) [km]')
    axes[2].set_ylabel('|Residual| [s]')
    axes[2].set_title(f'Distance vs |Residual| (r = {abs_corr:.4f})')
    axes[2].grid(True, alpha=0.3)
    
    # Binned analysis
    df_copy = df.copy()
    df_copy['distance_bin'] = pd.cut(df_copy[distance_col], bins=8)
    bin_stats = df_copy.groupby('distance_bin').agg({
        'residual_s': ['mean', 'std', 'count'],
        distance_col: 'mean'
    }).reset_index()
    
    bin_stats['bin_center'] = bin_stats['distance_bin'].apply(lambda x: x.mid if pd.notna(x) else 0)
    bin_stats = bin_stats.dropna()
    
    if not bin_stats.empty:
        axes[3].errorbar(bin_stats['bin_center'], bin_stats[('residual_s', 'mean')], 
                        yerr=bin_stats[('residual_s', 'std')], 
                        fmt='o-', capsize=5, linewidth=2, markersize=8, color=colors['neutral_color'])
        axes[3].axhline(0, color='black', linestyle='--', linewidth=2)
    
    axes[3].set_xlabel(f'Distance ({coordinate_type}) [km]')
    axes[3].set_ylabel('Mean Residual ± Std [s]')
    axes[3].set_title('Binned Residual Statistics')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, plots_dir / f'short_distance_correlation_{coordinate_type}', output_format)


def create_zoom_cluster_plots(df: pd.DataFrame, station_df: pd.DataFrame,
                             events_df: pd.DataFrame, colors: Dict[str, str],
                             plots_dir: Path, output_format: str,
                             config: Dict[str, Any]) -> None:
    """Create zoomed-in cluster analysis plots."""
    zoom_clusters = config['short_distance_analysis'].get('zoom_clusters', [])
    if not zoom_clusters:
        logger.info("No zoom clusters defined, skipping cluster analysis")
        return
    
    coordinate_type = config['short_distance_analysis']['coordinate_type']
    
    if coordinate_type == "transformed":
        x_col_station = 'X_grid_km'
        y_col_station = 'Y_grid_km'
        x_col_event = 'X_grid_km'
        y_col_event = 'Y_grid_km'
        coord_label = "Transformed"
    else:
        x_col_station = 'x_km'
        y_col_station = 'y_km'
        x_col_event = 'x_km'
        y_col_event = 'y_km'
        coord_label = "Original"
    
    for cluster in zoom_clusters:
        logger.info(f"Creating zoom plot for {cluster['name']}")
        
        # Filter stations within cluster bounds
        cluster_stations = station_df[
            (station_df[x_col_station] >= cluster['x_min']) & 
            (station_df[x_col_station] <= cluster['x_max']) &
            (station_df[y_col_station] >= cluster['y_min']) & 
            (station_df[y_col_station] <= cluster['y_max'])
        ]
        
        if cluster_stations.empty:
            logger.warning(f"No stations found in {cluster['name']} bounds")
            continue
        
        cluster_station_names = cluster_stations['station'].tolist()
        
        # Filter data for stations in this cluster
        cluster_data = df[df['station'].isin(cluster_station_names)].copy()
        
        if cluster_data.empty:
            logger.warning(f"No data found for {cluster['name']} stations")
            continue
        
        # Get events with data in this cluster
        cluster_events = cluster_data['event_id'].unique()
        cluster_events_df = events_df[events_df['event_id'].isin(cluster_events)]
        
        # Create subplot based on phase filter
        phase_filter = config['short_distance_analysis'].get('phase_filter')
        if phase_filter:
            fig, axes = create_multi_panel_figure(1, 2, figsize=(16, 6))
            plot_configs = [(phase_filter, 'paths'), (phase_filter, 'residuals')]
        else:
            fig, axes = create_multi_panel_figure(2, 2, figsize=(16, 12))
            plot_configs = [('P', 'paths'), ('S', 'paths'), ('P', 'residuals'), ('S', 'residuals')]
        
        # Create plots
        for plot_idx, (phase, plot_type) in enumerate(plot_configs):
            if plot_idx >= len(axes):
                break
                
            ax = axes[plot_idx] if isinstance(axes, np.ndarray) else axes
            
            # Filter data for this phase
            phase_data = cluster_data[cluster_data['phase'] == phase]
            
            if phase_data.empty:
                ax.text(0.5, 0.5, f'No {phase}-wave data\nin {cluster["name"]}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{cluster["name"]}: {phase}-wave {plot_type.title()}')
                continue
            
            if plot_type == 'paths':
                # Draw travel paths with residual color coding
                for _, row in phase_data.iterrows():
                    station_info = station_df[station_df['station'] == row['station']]
                    event_info = events_df[events_df['event_id'] == row['event_id']]
                    
                    if station_info.empty or event_info.empty:
                        continue
                        
                    station_info = station_info.iloc[0]
                    event_info = event_info.iloc[0]
                    
                    # Color and thickness based on residual
                    residual = row['residual_s']
                    if residual > 0.5:
                        color = colors['positive_residual']
                        alpha = 0.8
                    elif residual < -0.5:
                        color = colors['negative_residual'] 
                        alpha = 0.8
                    else:
                        color = colors['neutral_color']
                        alpha = 0.6
                    
                    line_width = min(4.0, max(0.8, abs(residual) * 1.0))
                    
                    # Draw path
                    ax.plot([event_info[x_col_event], station_info[x_col_station]], 
                           [event_info[y_col_event], station_info[y_col_station]], 
                           color=color, alpha=alpha, linewidth=line_width)
                
                # Plot events and stations
                if not cluster_events_df.empty:
                    ax.scatter(cluster_events_df[x_col_event], cluster_events_df[y_col_event], 
                              marker='*', s=120, color=colors['event_color'], 
                              edgecolors='black', linewidth=1, alpha=0.9, 
                              label=f'Events (N={len(cluster_events_df)})', zorder=5)
                
                ax.scatter(cluster_stations[x_col_station], cluster_stations[y_col_station], 
                          marker='^', s=80, color=colors['station_color'], 
                          edgecolors='white', linewidth=1, alpha=0.9, 
                          label=f'Stations (N={len(cluster_stations)})', zorder=5)
            
            elif plot_type == 'residuals':
                # Create residual intensity map
                station_residuals = phase_data.groupby('station')['residual_s'].mean()
                
                station_coords = []
                residual_values = []
                
                for station, mean_residual in station_residuals.items():
                    station_info = station_df[station_df['station'] == station]
                    if not station_info.empty:
                        station_info = station_info.iloc[0]
                        station_coords.append([station_info[x_col_station], station_info[y_col_station]])
                        residual_values.append(mean_residual)
                
                if station_coords:
                    station_coords = np.array(station_coords)
                    scatter = ax.scatter(station_coords[:, 0], station_coords[:, 1], 
                                       c=residual_values, cmap='RdBu_r', 
                                       s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
                    cbar.set_label(f'{phase}-wave Mean Residual [s]', rotation=270, labelpad=15)
                
                # Show events for reference
                if not cluster_events_df.empty:
                    ax.scatter(cluster_events_df[x_col_event], cluster_events_df[y_col_event], 
                              marker='x', s=60, color='black', alpha=0.4, 
                              label=f'Events (N={len(cluster_events_df)})')
            
            # Set plot bounds to cluster area with padding
            x_range = cluster['x_max'] - cluster['x_min']
            y_range = cluster['y_max'] - cluster['y_min']
            padding_x = x_range * 0.05
            padding_y = y_range * 0.05
            
            ax.set_xlim(cluster['x_min'] - padding_x, cluster['x_max'] + padding_x)
            ax.set_ylim(cluster['y_min'] - padding_y, cluster['y_max'] + padding_y)
            
            # Formatting
            ax.set_xlabel(f'X ({coord_label}) [km]')
            ax.set_ylabel(f'Y ({coord_label}) [km]')
            ax.set_title(f'{cluster["name"]}: {phase}-wave {plot_type.title()}\n'
                        f'({len(phase_data)} measurements)')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc='upper right', fontsize=9)
            
            # Add statistics box for paths plot
            if plot_type == 'paths' and not phase_data.empty:
                stats_text = (f'Mean Res: {phase_data["residual_s"].mean():.3f}s\n'
                             f'Std Res: {phase_data["residual_s"].std():.3f}s\n'
                             f'RMS: {np.sqrt(np.mean(phase_data["residual_s"]**2)):.3f}s')
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add cluster description as overall title
        fig.suptitle(f'Zoom Analysis: {cluster["name"]}\n{cluster["description"]}', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save plot
        filename = f"zoom_cluster_{cluster['name'].lower()}_{coordinate_type}"
        save_figure(fig, plots_dir / filename, output_format)


def generate_short_distance_report(df: pd.DataFrame, stats: Dict[str, Any], 
                                  config: Dict[str, Any]) -> None:
    """Generate comprehensive short distance analysis report."""
    output_dir = Path(config['short_distance_analysis']['output_dir'])
    reports_dir = output_dir / 'reports'
    ensure_directory_exists(reports_dir)
    
    report_file = reports_dir / 'short_distance_analysis_report.txt'
    coordinate_type = config['short_distance_analysis']['coordinate_type']
    distance_cutoff = config['short_distance_analysis']['distance_cutoff_km']
    
    report_lines = [
        "SHORT DISTANCE TRAVEL TIME ANALYSIS REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Distance cutoff: {distance_cutoff} km",
        f"Coordinate type: {coordinate_type}",
        f"Travel time file: {config['short_distance_analysis']['travel_time_csv']}",
        "",
        "SUMMARY",
        "-" * 20,
        f"Total data points: {stats['overall']['count']}",
        f"Unique events: {stats['overall']['unique_events']}",
        f"Unique stations: {stats['overall']['unique_stations']}",
        f"Distance range: {stats['overall']['distance_min']:.3f} - {stats['overall']['distance_max']:.3f} km",
        f"Mean distance: {stats['overall']['distance_mean']:.3f} km",
        f"Mean residual: {stats['overall']['residual_mean']:.4f} s",
        f"RMS residual: {stats['overall']['residual_rms']:.4f} s",
        f"Distance-residual correlation: {stats['overall']['distance_residual_correlation']:.4f}",
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
                f"Mean distance: {phase_stats['distance_mean']:.3f} km",
                f"Mean residual: {phase_stats['residual_mean']:.4f} s",
                f"RMS residual: {phase_stats['residual_rms']:.4f} s",
                f"Distance-residual correlation: {phase_stats['distance_residual_correlation']:.4f}",
                ""
            ])
    
    # Station and event summaries
    report_lines.extend([
        "STATION SUMMARY",
        "-" * 20,
        f"Unique stations: {stats['station_summary']['unique_stations']}",
        f"Mean measurements per station: {stats['station_summary']['mean_measurements_per_station']:.1f}",
        "",
        "EVENT SUMMARY",
        "-" * 20,
        f"Unique events: {stats['event_summary']['unique_events']}",
        f"Mean measurements per event: {stats['event_summary']['mean_measurements_per_event']:.1f}",
        ""
    ])
    
    write_text_file(report_file, '\n'.join(report_lines))
    logger.info(f"Short distance analysis report saved to: {report_file}")