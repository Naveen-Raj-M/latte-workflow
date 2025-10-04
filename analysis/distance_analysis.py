"""
Distance-dependent travel time analysis module.

This module analyzes how travel time residuals vary with distance,
including correlation analysis, binned statistics, and ratio analysis.
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
    create_multi_panel_figure, create_residual_vs_distance_plot,
    add_statistics_text, subsample_data_for_plotting
)

logger = logging.getLogger(__name__)


def analyze_distance_dependent_residuals(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function for distance-dependent travel time analysis.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    logger.info("Starting distance-dependent travel time analysis")
    
    # Setup plot style
    setup_plot_style(config['analysis'])
    
    # Load data
    travel_time_csv = config['distance_analysis']['travel_time_csv']
    station_csv = config['distance_analysis']['station_csv']
    events_csv = config['distance_analysis']['events_csv']
    
    logger.info(f"Loading data files:")
    logger.info(f"  Travel times: {travel_time_csv}")
    logger.info(f"  Stations: {station_csv}")
    logger.info(f"  Events: {events_csv}")
    
    # Load and merge data
    df_merged = load_and_merge_distance_data(travel_time_csv, station_csv, events_csv, config)
    
    # Compute distance statistics
    logger.info("Computing distance-dependent statistics")
    stats = compute_distance_statistics(df_merged, config)
    
    # Create plots
    logger.info("Creating distance analysis plots")
    create_distance_analysis_plots(df_merged, stats, config)
    
    # Generate report
    logger.info("Generating distance analysis report")
    generate_distance_analysis_report(df_merged, stats, config)
    
    logger.info("Distance-dependent analysis completed")
    return stats


def load_and_merge_distance_data(travel_time_csv: str, station_csv: str, 
                                events_csv: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load and merge travel time, station, and event data with distance calculations.
    
    Args:
        travel_time_csv: Path to travel time CSV
        station_csv: Path to station CSV
        events_csv: Path to events CSV
        config: Configuration dictionary
        
    Returns:
        Merged dataframe with distance calculations
    """
    # Load data
    df_tt = read_csv_file(travel_time_csv)
    df_stations = read_csv_file(station_csv)
    df_events = read_csv_file(events_csv)
    
    logger.info(f"Loaded {len(df_tt)} travel time records")
    logger.info(f"Loaded {len(df_stations)} stations")
    logger.info(f"Loaded {len(df_events)} events")
    
    # Merge with station data
    df_merged = df_tt.merge(df_stations, on='station', how='left')
    logger.info(f"After station merge: {len(df_merged)} records")
    
    # Merge with event data
    df_merged = df_merged.merge(df_events, on='event_id', how='left')
    logger.info(f"After event merge: {len(df_merged)} records")
    
    # Calculate distances
    coordinate_type = config['distance_analysis']['coordinate_type']
    df_merged = calculate_distances(df_merged, coordinate_type)
    
    # Remove records with missing distances
    initial_count = len(df_merged)
    df_merged = df_merged.dropna(subset=[f'distance_{coordinate_type}_km'])
    final_count = len(df_merged)
    
    if final_count < initial_count:
        logger.warning(f"Removed {initial_count - final_count} records with missing distances")
    
    logger.info(f"Final merged dataset: {len(df_merged)} records")
    return df_merged


def calculate_distances(df: pd.DataFrame, coordinate_type: str) -> pd.DataFrame:
    """
    Calculate 3D Euclidean distances between events and stations.
    
    Args:
        df: Merged dataframe
        coordinate_type: "transformed" or "original"
        
    Returns:
        Dataframe with distance calculations
    """
    logger.info(f"Calculating distances using {coordinate_type} coordinates")
    
    if coordinate_type == "transformed":
        # Use grid coordinates (assumed to be in km)
        event_x = df['X_grid_km_y']  # _y suffix from merge
        event_y = df['Y_grid_km_y']
        event_z = df['Z_grid_km_y']
        station_x = df['X_grid_km_x']  # _x suffix from merge
        station_y = df['Y_grid_km_x']
        station_z = df['Z_grid_km_x']
    else:  # original
        # Use original coordinates (convert to km if needed)
        event_x = df['x_km_y']
        event_y = df['y_km_y']
        event_z = df['z_km_y']
        station_x = df['x_km_x']
        station_y = df['y_km_x']
        station_z = df['z_km_x']
    
    # Calculate 3D Euclidean distance
    distance_km = np.sqrt((event_x - station_x)**2 + 
                         (event_y - station_y)**2 + 
                         (event_z - station_z)**2)
    
    df[f'distance_{coordinate_type}_km'] = distance_km
    
    logger.info(f"Distance range: {distance_km.min():.3f} - {distance_km.max():.3f} km")
    logger.info(f"Mean distance: {distance_km.mean():.3f} km")
    
    return df


def compute_distance_statistics(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute comprehensive distance-dependent statistics.
    
    Args:
        df: Merged dataframe with distances
        config: Configuration dictionary
        
    Returns:
        Dictionary of statistics
    """
    coordinate_type = config['distance_analysis']['coordinate_type']
    distance_col = f'distance_{coordinate_type}_km'
    
    stats = {}
    
    # Overall distance statistics
    distances = df[distance_col]
    residuals = df['residual_s']
    
    stats['overall'] = {
        'count': len(df),
        'distance_min': distances.min(),
        'distance_max': distances.max(),
        'distance_mean': distances.mean(),
        'distance_median': distances.median(),
        'distance_std': distances.std(),
        'residual_distance_correlation': distances.corr(residuals),
        'residual_abs_distance_correlation': distances.corr(np.abs(residuals))
    }
    
    # Phase-specific statistics
    for phase in ['P', 'S']:
        phase_data = df[df['phase'] == phase]
        if not phase_data.empty:
            phase_distances = phase_data[distance_col]
            phase_residuals = phase_data['residual_s']
            
            stats[f'{phase.lower()}_wave'] = {
                'count': len(phase_data),
                'distance_mean': phase_distances.mean(),
                'residual_distance_correlation': phase_distances.corr(phase_residuals),
                'residual_abs_distance_correlation': phase_distances.corr(np.abs(phase_residuals)),
                'mean_residual': phase_residuals.mean(),
                'rms_residual': np.sqrt(np.mean(phase_residuals**2))
            }
            
            # Fit polynomial if enabled
            if config['distance_analysis'].get('fit_polynomial_degree', 1) > 0:
                degree = config['distance_analysis']['fit_polynomial_degree']
                coeffs = np.polyfit(phase_distances, phase_residuals, degree)
                stats[f'{phase.lower()}_wave']['polynomial_coeffs'] = coeffs.tolist()
    
    # Binned analysis
    if config['distance_analysis'].get('enable_binned_analysis', True):
        stats['binned'] = compute_binned_distance_statistics(df, config)
    
    # Ratio analysis
    if config['distance_analysis'].get('enable_ratio_analysis', True):
        stats['ratio'] = compute_ratio_distance_statistics(df, config)
    
    return stats


def compute_binned_distance_statistics(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Compute statistics for distance bins."""
    coordinate_type = config['distance_analysis']['coordinate_type']
    distance_col = f'distance_{coordinate_type}_km'
    n_bins = config['distance_analysis'].get('distance_bins', 20)
    
    # Create distance bins
    df_copy = df.copy()
    df_copy['distance_bin'] = pd.cut(df_copy[distance_col], bins=n_bins)
    
    bin_stats = {}
    
    # Overall binned statistics
    overall_bins = df_copy.groupby('distance_bin').agg({
        'residual_s': ['count', 'mean', 'std', lambda x: np.sqrt(np.mean(x**2))],
        distance_col: 'mean'
    }).round(4)
    
    overall_bins.columns = ['count', 'mean_residual', 'std_residual', 'rms_residual', 'mean_distance']
    bin_stats['overall'] = overall_bins.to_dict('index')
    
    # Phase-specific binned statistics
    for phase in ['P', 'S']:
        phase_data = df_copy[df_copy['phase'] == phase]
        if not phase_data.empty:
            phase_bins = phase_data.groupby('distance_bin').agg({
                'residual_s': ['count', 'mean', 'std', lambda x: np.sqrt(np.mean(x**2))],
                distance_col: 'mean'
            }).round(4)
            
            phase_bins.columns = ['count', 'mean_residual', 'std_residual', 'rms_residual', 'mean_distance']
            bin_stats[f'{phase.lower()}_wave'] = phase_bins.to_dict('index')
    
    return bin_stats


def compute_ratio_distance_statistics(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Compute ratio analysis statistics."""
    # Filter positive travel times for ratio analysis
    ratio_data = df[(df['observed_tt'] > 0) & (df['synthetic_tt'] > 0)].copy()
    
    if ratio_data.empty:
        return {}
    
    coordinate_type = config['distance_analysis']['coordinate_type']
    distance_col = f'distance_{coordinate_type}_km'
    
    ratio_data['ratio'] = ratio_data['synthetic_tt'] / ratio_data['observed_tt']
    ratio_data['ln_ratio'] = np.log(ratio_data['ratio'])
    
    ratio_stats = {}
    
    # Overall ratio statistics
    distances = ratio_data[distance_col]
    ratios = ratio_data['ratio']
    ln_ratios = ratio_data['ln_ratio']
    
    ratio_stats['overall'] = {
        'count': len(ratio_data),
        'mean_ratio': ratios.mean(),
        'mean_ln_ratio': ln_ratios.mean(),
        'ratio_distance_correlation': distances.corr(ratios),
        'ln_ratio_distance_correlation': distances.corr(ln_ratios)
    }
    
    # Phase-specific ratio statistics
    for phase in ['P', 'S']:
        phase_data = ratio_data[ratio_data['phase'] == phase]
        if not phase_data.empty:
            phase_distances = phase_data[distance_col]
            phase_ratios = phase_data['ratio']
            phase_ln_ratios = phase_data['ln_ratio']
            
            ratio_stats[f'{phase.lower()}_wave'] = {
                'count': len(phase_data),
                'mean_ratio': phase_ratios.mean(),
                'mean_ln_ratio': phase_ln_ratios.mean(),
                'ratio_distance_correlation': phase_distances.corr(phase_ratios),
                'ln_ratio_distance_correlation': phase_distances.corr(phase_ln_ratios)
            }
    
    return ratio_stats

def create_simple_binned_plot(ax: plt.Axes, data: pd.DataFrame, distance_col: str, 
                             residual_col: str, color: str, title: str, coordinate_type: str) -> None:
    """Create simple binned plot when external stats are not available."""
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
    bin_stats = data_copy.groupby('distance_bin').agg({
        residual_col: ['count', 'mean', 'std'],
        distance_col: 'mean'
    }).reset_index()
    
    # Flatten column names
    bin_stats.columns = ['distance_bin', 'count', 'mean_residual', 'std_residual', 'mean_distance']
    bin_stats = bin_stats.dropna()
    
    if not bin_stats.empty:
        ax.errorbar(bin_stats['mean_distance'], bin_stats['mean_residual'], 
                   yerr=bin_stats['std_residual'], 
                   fmt='o-', color=color, capsize=5, linewidth=2, markersize=6)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
    
    ax.set_xlabel(f'Distance ({coordinate_type}) [km]')
    ax.set_ylabel('Mean Residual ± Std [s]')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def create_rms_binned_plot(ax: plt.Axes, data: pd.DataFrame, distance_col: str, 
                          residual_col: str, color: str, title: str, coordinate_type: str) -> None:
    """Create RMS binned plot."""
    if data.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        return
    
    # Create distance bins
    n_bins = 8
    data_copy = data.copy()
    data_copy['distance_bin'] = pd.cut(data_copy[distance_col], bins=n_bins)
    
    # Calculate RMS by bin
    def rms(x):
        return np.sqrt(np.mean(x**2))
    
    bin_stats = data_copy.groupby('distance_bin').agg({
        residual_col: ['count', rms],
        distance_col: 'mean'
    }).reset_index()
    
    # Flatten column names
    bin_stats.columns = ['distance_bin', 'count', 'rms_residual', 'mean_distance']
    bin_stats = bin_stats.dropna()
    
    if not bin_stats.empty:
        ax.plot(bin_stats['mean_distance'], bin_stats['rms_residual'], 
               'o-', color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel(f'Distance ({coordinate_type}) [km]')
    ax.set_ylabel('RMS Residual [s]')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

def create_binned_errorbar_plot(ax: plt.Axes, bin_data: Dict, color: str, 
                               title: str, coordinate_type: str) -> None:
    """Create errorbar plot for binned data."""
    # Extract data from binned statistics
    bin_centers = []
    mean_residuals = []
    std_residuals = []
    
    for bin_interval, data in bin_data.items():
        if pd.notna(bin_interval) and data['count'] > 0:
            bin_centers.append(data['mean_distance'])
            mean_residuals.append(data['mean_residual'])
            std_residuals.append(data['std_residual'])
    
    if bin_centers:
        ax.errorbar(bin_centers, mean_residuals, yerr=std_residuals, 
                   fmt='o-', color=color, capsize=5, linewidth=2, markersize=6)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel(f'Distance ({coordinate_type}) [km]')
        ax.set_ylabel('Mean Residual ± Std [s]')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)


def create_binned_analysis_plots(df: pd.DataFrame, stats: Dict[str, Any], colors: Dict[str, str],
                                plots_dir: Path, output_format: str, config: Dict[str, Any]) -> None:
    """Create binned analysis plots."""
    coordinate_type = config['distance_analysis']['coordinate_type']
    distance_col = f'distance_{coordinate_type}_km'
    
    p_data = df[df['phase'] == 'P']
    s_data = df[df['phase'] == 'S']
    
    fig, axes = create_multi_panel_figure(2, 2, figsize=(16, 12),
                                        title=f'Binned Distance Analysis ({coordinate_type.title()} Coordinates)')
    
    # Ensure axes is always a flat array
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else axes
    
    # P-wave mean residuals by distance bin
    if not p_data.empty:
        create_simple_binned_plot(axes_flat[0], p_data, distance_col, 'residual_s',
                                colors['p_wave_dark'], 'P-wave: Mean Residuals by Distance', coordinate_type)
    
    # P-wave RMS residuals by distance bin
    if not p_data.empty:
        create_rms_binned_plot(axes_flat[1], p_data, distance_col, 'residual_s',
                             colors['p_wave_dark'], 'P-wave: RMS Residuals by Distance', coordinate_type)
    
    # S-wave mean residuals by distance bin
    if not s_data.empty:
        create_simple_binned_plot(axes_flat[2], s_data, distance_col, 'residual_s',
                                colors['s_wave_dark'], 'S-wave: Mean Residuals by Distance', coordinate_type)
    
    # S-wave RMS residuals by distance bin
    if not s_data.empty:
        create_rms_binned_plot(axes_flat[3], s_data, distance_col, 'residual_s',
                             colors['s_wave_dark'], 'S-wave: RMS Residuals by Distance', coordinate_type)
    
    plt.tight_layout()
    save_figure(fig, plots_dir / 'binned_distance_analysis', output_format)


def create_distance_analysis_plots(df: pd.DataFrame, stats: Dict[str, Any], 
                                 config: Dict[str, Any]) -> None:
    """
    Create comprehensive distance analysis plots.
    
    Args:
        df: Merged dataframe with distances
        stats: Statistics dictionary
        config: Configuration dictionary
    """
    output_dir = Path(config['distance_analysis']['output_dir'])
    plots_dir = output_dir / 'plots'
    ensure_directory_exists(plots_dir)
    
    colors = get_color_scheme(config['analysis'])
    output_format = config['analysis'].get('output_format', 'png')
    coordinate_type = config['distance_analysis']['coordinate_type']
    
    # Subsample data for plotting if needed
    max_points = config['analysis']['plot_style'].get('max_plot_points', 50000)
    df_plot = subsample_data_for_plotting(df, max_points)
    
    # Basic residual vs distance plots
    create_residual_distance_plots(df_plot, colors, plots_dir, output_format, config)
    
    # Correlation analysis plots
    if config['distance_analysis'].get('enable_correlation_analysis', True):
        create_correlation_analysis_plots(df_plot, stats, colors, plots_dir, output_format, config)
    
    # Binned analysis plots
    if config['distance_analysis'].get('enable_binned_analysis', True):
        create_binned_analysis_plots(df_plot, stats, colors, plots_dir, output_format, config)
    
    # Ratio analysis plots
    if config['distance_analysis'].get('enable_ratio_analysis', True):
        create_ratio_distance_plots(df_plot, colors, plots_dir, output_format, config)


def create_residual_distance_plots(df: pd.DataFrame, colors: Dict[str, str],
                                  plots_dir: Path, output_format: str,
                                  config: Dict[str, Any]) -> None:
    """Create basic residual vs distance plots."""
    coordinate_type = config['distance_analysis']['coordinate_type']
    distance_col = f'distance_{coordinate_type}_km'
    
    p_data = df[df['phase'] == 'P']
    s_data = df[df['phase'] == 'S']
    
    fig, axes = create_multi_panel_figure(2, 3, figsize=(20, 14),
                                        title=f'Residuals vs Distance ({coordinate_type.title()} Coordinates)')
    
    # Ensure axes is always a flat array
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else axes
    
    # P-wave linear residuals vs distance
    if not p_data.empty:
        axes_flat[0].scatter(p_data[distance_col], p_data['residual_s'], 
                       alpha=0.6, s=15, color=colors['p_wave_dark'])
        axes_flat[0].axhline(0, color=colors['neutral_color'], linestyle='--', linewidth=2)
        axes_flat[0].set_xlabel(f'Distance ({coordinate_type}) [km]')
        axes_flat[0].set_ylabel('Residual (Obs - Syn) [s]')
        axes_flat[0].set_title('P-wave: Linear Residuals vs Distance')
        axes_flat[0].grid(True, alpha=0.3)
        
        # Add trend line
        if len(p_data) > 1:
            z = np.polyfit(p_data[distance_col], p_data['residual_s'], 1)
            p_trend = np.poly1d(z)
            x_trend = np.linspace(p_data[distance_col].min(), p_data[distance_col].max(), 100)
            axes_flat[0].plot(x_trend, p_trend(x_trend), "r--", alpha=0.8, linewidth=2)
    
    # P-wave absolute residuals vs distance
    if not p_data.empty:
        axes_flat[1].scatter(p_data[distance_col], np.abs(p_data['residual_s']), 
                       alpha=0.6, s=15, color=colors['p_wave_light'])
        axes_flat[1].set_xlabel(f'Distance ({coordinate_type}) [km]')
        axes_flat[1].set_ylabel('|Residual| [s]')
        axes_flat[1].set_title('P-wave: Absolute Residuals vs Distance')
        axes_flat[1].grid(True, alpha=0.3)
    
    # P-wave binned residual statistics
    if not p_data.empty:
        create_simple_binned_plot(axes_flat[2], p_data, distance_col, 'residual_s',
                                colors['p_wave_dark'], 'P-wave: Binned Statistics', coordinate_type)
    
    # S-wave linear residuals vs distance
    if not s_data.empty:
        axes_flat[3].scatter(s_data[distance_col], s_data['residual_s'], 
                       alpha=0.6, s=15, color=colors['s_wave_dark'])
        axes_flat[3].axhline(0, color=colors['neutral_color'], linestyle='--', linewidth=2)
        axes_flat[3].set_xlabel(f'Distance ({coordinate_type}) [km]')
        axes_flat[3].set_ylabel('Residual (Obs - Syn) [s]')
        axes_flat[3].set_title('S-wave: Linear Residuals vs Distance')
        axes_flat[3].grid(True, alpha=0.3)
        
        # Add trend line
        if len(s_data) > 1:
            z = np.polyfit(s_data[distance_col], s_data['residual_s'], 1)
            p_trend = np.poly1d(z)
            x_trend = np.linspace(s_data[distance_col].min(), s_data[distance_col].max(), 100)
            axes_flat[3].plot(x_trend, p_trend(x_trend), "r--", alpha=0.8, linewidth=2)
    
    # S-wave absolute residuals vs distance
    if not s_data.empty:
        axes_flat[4].scatter(s_data[distance_col], np.abs(s_data['residual_s']), 
                       alpha=0.6, s=15, color=colors['s_wave_light'])
        axes_flat[4].set_xlabel(f'Distance ({coordinate_type}) [km]')
        axes_flat[4].set_ylabel('|Residual| [s]')
        axes_flat[4].set_title('S-wave: Absolute Residuals vs Distance')
        axes_flat[4].grid(True, alpha=0.3)
    
    # S-wave binned residual statistics
    if not s_data.empty:
        create_simple_binned_plot(axes_flat[5], s_data, distance_col, 'residual_s',
                                colors['s_wave_dark'], 'S-wave: Binned Statistics', coordinate_type)
    
    plt.tight_layout()
    save_figure(fig, plots_dir / 'residuals_vs_distance', output_format)


def create_binned_errorbar_plot(ax: plt.Axes, bin_data: Dict, color: str, 
                               title: str, coordinate_type: str) -> None:
    """Create errorbar plot for binned data."""
    # Extract data from binned statistics
    bin_centers = []
    mean_residuals = []
    std_residuals = []
    
    for bin_interval, data in bin_data.items():
        if pd.notna(bin_interval) and data['count'] > 0:
            bin_centers.append(data['mean_distance'])
            mean_residuals.append(data['mean_residual'])
            std_residuals.append(data['std_residual'])
    
    if bin_centers:
        ax.errorbar(bin_centers, mean_residuals, yerr=std_residuals, 
                   fmt='o-', color=color, capsize=5, linewidth=2, markersize=6)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel(f'Distance ({coordinate_type}) [km]')
        ax.set_ylabel('Mean Residual ± Std [s]')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)


def create_correlation_analysis_plots(df: pd.DataFrame, stats: Dict[str, Any],
                                     colors: Dict[str, str], plots_dir: Path,
                                     output_format: str, config: Dict[str, Any]) -> None:
    """Create correlation analysis plots."""
    coordinate_type = config['distance_analysis']['coordinate_type']
    distance_col = f'distance_{coordinate_type}_km'
    
    p_data = df[df['phase'] == 'P']
    s_data = df[df['phase'] == 'S']
    
    fig, axes = create_multi_panel_figure(2, 2, figsize=(16, 12),
                                        title=f'Correlation Analysis ({coordinate_type.title()} Coordinates)')
    
    # Ensure axes is always a flat array
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else axes
    
    # Overall correlation plot
    if not df.empty:
        axes_flat[0].scatter(df[distance_col], df['residual_s'], alpha=0.6, s=15, 
                           color=colors['neutral_color'])
        axes_flat[0].axhline(0, color='black', linestyle='--', linewidth=2)
        
        # Add trend line
        if len(df) > 1:
            z = np.polyfit(df[distance_col], df['residual_s'], 1)
            p_trend = np.poly1d(z)
            x_trend = np.linspace(df[distance_col].min(), df[distance_col].max(), 100)
            axes_flat[0].plot(x_trend, p_trend(x_trend), "r--", alpha=0.8, linewidth=2)
        
        corr = df[distance_col].corr(df['residual_s'])
        axes_flat[0].set_xlabel(f'Distance ({coordinate_type}) [km]')
        axes_flat[0].set_ylabel('Residual [s]')
        axes_flat[0].set_title(f'Overall Correlation (r = {corr:.4f})')
        axes_flat[0].grid(True, alpha=0.3)
    
    # Phase-specific correlations
    if not p_data.empty:
        axes_flat[1].scatter(p_data[distance_col], p_data['residual_s'], 
                           alpha=0.6, s=15, color=colors['p_wave_dark'], label='P-wave')
        p_corr = p_data[distance_col].corr(p_data['residual_s'])
        
        # P-wave trend line
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
        
        # S-wave trend line
        if len(s_data) > 1:
            z_s = np.polyfit(s_data[distance_col], s_data['residual_s'], 1)
            p_s = np.poly1d(z_s)
            x_trend_s = np.linspace(s_data[distance_col].min(), s_data[distance_col].max(), 100)
            axes_flat[1].plot(x_trend_s, p_s(x_trend_s), color=colors['s_wave_light'], 
                            linestyle='--', alpha=0.8, linewidth=2)
    
    axes_flat[1].axhline(0, color='black', linestyle='--', linewidth=2)
    axes_flat[1].set_xlabel(f'Distance ({coordinate_type}) [km]')
    axes_flat[1].set_ylabel('Residual [s]')
    
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
        axes_flat[2].set_xlabel(f'Distance ({coordinate_type}) [km]')
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
        axes_flat[3].set_xlabel(f'Distance ({coordinate_type}) [km]')
        axes_flat[3].set_ylabel('Frequency')
        axes_flat[3].set_title('Distance Distribution')
        axes_flat[3].grid(True, alpha=0.3)
        axes_flat[3].legend()
    
    plt.tight_layout()
    save_figure(fig, plots_dir / 'correlation_analysis', output_format)


def create_ratio_distance_plots(df: pd.DataFrame, colors: Dict[str, str],
                               plots_dir: Path, output_format: str,
                               config: Dict[str, Any]) -> None:
    """Create ratio vs distance plots."""
    # Filter positive travel times for ratio analysis
    ratio_data = df[(df['observed_tt'] > 0) & (df['synthetic_tt'] > 0)].copy()
    
    if ratio_data.empty:
        logger.warning("No positive travel times for ratio analysis")
        return
    
    coordinate_type = config['distance_analysis']['coordinate_type']
    distance_col = f'distance_{coordinate_type}_km'
    
    ratio_data['ratio'] = ratio_data['synthetic_tt'] / ratio_data['observed_tt']
    
    p_ratio = ratio_data[ratio_data['phase'] == 'P']
    s_ratio = ratio_data[ratio_data['phase'] == 'S']
    
    fig, axes = create_multi_panel_figure(2, 2, figsize=(16, 12),
                                        title=f'Travel Time Ratios vs Distance ({coordinate_type.title()})')
    
    # Ensure axes is always a flat array
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else axes
    
    # P-wave ratio vs distance
    if not p_ratio.empty:
        axes_flat[0].scatter(p_ratio[distance_col], p_ratio['ratio'], 
                           alpha=0.6, s=15, color=colors['p_wave_dark'])
        axes_flat[0].axhline(1, color=colors['neutral_color'], linestyle='--', linewidth=2)
        axes_flat[0].set_xlabel(f'Distance ({coordinate_type}) [km]')
        axes_flat[0].set_ylabel('Ratio (Syn/Obs)')
        axes_flat[0].set_title('P-wave: Travel Time Ratio vs Distance')
        axes_flat[0].grid(True, alpha=0.3)
    
    # P-wave ratio vs distance (log scale)
    if not p_ratio.empty:
        axes_flat[1].scatter(p_ratio[distance_col], p_ratio['ratio'], 
                           alpha=0.6, s=15, color=colors['p_wave_light'])
        axes_flat[1].axhline(1, color=colors['neutral_color'], linestyle='--', linewidth=2)
        axes_flat[1].set_xscale('log')
        axes_flat[1].set_xlabel(f'Distance ({coordinate_type}) [km]')
        axes_flat[1].set_ylabel('Ratio (Syn/Obs)')
        axes_flat[1].set_title('P-wave: Ratio vs Distance (log scale)')
        axes_flat[1].grid(True, which='both', alpha=0.3)
    
    # S-wave ratio vs distance
    if not s_ratio.empty:
        axes_flat[2].scatter(s_ratio[distance_col], s_ratio['ratio'], 
                           alpha=0.6, s=15, color=colors['s_wave_dark'])
        axes_flat[2].axhline(1, color=colors['neutral_color'], linestyle='--', linewidth=2)
        axes_flat[2].set_xlabel(f'Distance ({coordinate_type}) [km]')
        axes_flat[2].set_ylabel('Ratio (Syn/Obs)')
        axes_flat[2].set_title('S-wave: Travel Time Ratio vs Distance')
        axes_flat[2].grid(True, alpha=0.3)
    
    # S-wave ratio vs distance (log scale)
    if not s_ratio.empty:
        axes_flat[3].scatter(s_ratio[distance_col], s_ratio['ratio'], 
                           alpha=0.6, s=15, color=colors['s_wave_light'])
        axes_flat[3].axhline(1, color=colors['neutral_color'], linestyle='--', linewidth=2)
        axes_flat[3].set_xscale('log')
        axes_flat[3].set_xlabel(f'Distance ({coordinate_type}) [km]')
        axes_flat[3].set_ylabel('Ratio (Syn/Obs)')
        axes_flat[3].set_title('S-wave: Ratio vs Distance (log scale)')
        axes_flat[3].grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, plots_dir / 'ratios_vs_distance', output_format)


def generate_distance_analysis_report(df: pd.DataFrame, stats: Dict[str, Any], 
                                     config: Dict[str, Any]) -> None:
    """Generate comprehensive distance analysis report."""
    output_dir = Path(config['distance_analysis']['output_dir'])
    reports_dir = output_dir / 'reports'
    ensure_directory_exists(reports_dir)
    
    report_file = reports_dir / 'distance_analysis_report.txt'
    coordinate_type = config['distance_analysis']['coordinate_type']
    
    report_lines = [
        "DISTANCE-DEPENDENT TRAVEL TIME ANALYSIS REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Coordinate type: {coordinate_type}",
        f"Travel time file: {config['distance_analysis']['travel_time_csv']}",
        "",
        "SUMMARY",
        "-" * 20,
        f"Total data points: {stats['overall']['count']}",
        f"Distance range: {stats['overall']['distance_min']:.3f} - {stats['overall']['distance_max']:.3f} km",
        f"Mean distance: {stats['overall']['distance_mean']:.3f} km",
        f"Distance-residual correlation: {stats['overall']['residual_distance_correlation']:.4f}",
        f"Distance-|residual| correlation: {stats['overall']['residual_abs_distance_correlation']:.4f}",
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
                f"Distance-residual correlation: {phase_stats['residual_distance_correlation']:.4f}",
                f"Distance-|residual| correlation: {phase_stats['residual_abs_distance_correlation']:.4f}",
                f"Mean residual: {phase_stats['mean_residual']:.4f} s",
                f"RMS residual: {phase_stats['rms_residual']:.4f} s",
                ""
            ])
            
            # Add polynomial fit if available
            if 'polynomial_coeffs' in phase_stats:
                coeffs = phase_stats['polynomial_coeffs']
                report_lines.append(f"Linear fit coefficients: slope={coeffs[0]:.6f}, intercept={coeffs[1]:.6f}")
                report_lines.append("")
    
    # Ratio analysis if available
    if 'ratio' in stats:
        ratio_stats = stats['ratio']
        report_lines.extend([
            "RATIO ANALYSIS",
            "-" * 20,
            f"Total ratio points: {ratio_stats['overall']['count']}",
            f"Mean ratio (Syn/Obs): {ratio_stats['overall']['mean_ratio']:.4f}",
            f"Mean ln(ratio): {ratio_stats['overall']['mean_ln_ratio']:.4f}",
            f"Ratio-distance correlation: {ratio_stats['overall']['ratio_distance_correlation']:.4f}",
            f"ln(ratio)-distance correlation: {ratio_stats['overall']['ln_ratio_distance_correlation']:.4f}",
            ""
        ])
        
        for phase in ['P', 'S']:
            phase_key = f'{phase.lower()}_wave'
            if phase_key in ratio_stats:
                phase_ratio = ratio_stats[phase_key]
                report_lines.extend([
                    f"{phase}-wave ratio statistics:",
                    f"  Points: {phase_ratio['count']}",
                    f"  Mean ratio: {phase_ratio['mean_ratio']:.4f}",
                    f"  Ratio-distance correlation: {phase_ratio['ratio_distance_correlation']:.4f}",
                    ""
                ])
    
    write_text_file(report_file, report_lines)
    logger.info(f"Distance analysis report saved to: {report_file}")