"""
Velocity model comparison analysis module.

This module provides comprehensive comparison analysis between velocity models
including spatial projections, profile analysis, and statistical comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
from datetime import datetime
from scipy.interpolate import griddata

from utils.io_utils import read_csv_file, write_text_file, ensure_directory_exists
from utils.plot_utils import (
    setup_plot_style, get_color_scheme, save_figure,
    create_multi_panel_figure, create_spatial_plot, add_statistics_text
)

logger = logging.getLogger(__name__)


def analyze_velocity_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function for velocity model comparison analysis.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    logger.info("Starting velocity model comparison analysis")
    
    # Setup plot style
    setup_plot_style(config['analysis'])
    
    # Load data
    model1_csv = config['velocity_analysis']['model1_csv']
    model2_csv = config['velocity_analysis']['model2_csv']
    
    logger.info(f"Loading velocity models:")
    logger.info(f"  Model 1: {model1_csv}")
    logger.info(f"  Model 2: {model2_csv}")
    
    df1, df2 = load_and_validate_velocity_models(model1_csv, model2_csv, config)
    
    # Compute differences and statistics
    logger.info("Computing velocity differences and statistics")
    differences = compute_velocity_differences(df1, df2, config)
    stats = compute_velocity_statistics(df1, df2, differences, config)
    
    # Find extreme profiles and locations
    extremes = find_extreme_velocity_profiles(df1, differences, config)
    
    # Create plots
    logger.info("Creating velocity comparison plots")
    create_velocity_comparison_plots(df1, df2, differences, extremes, config)
    
    # Generate summary report
    logger.info("Generating velocity comparison report")
    generate_velocity_comparison_report(df1, df2, differences, stats, extremes, config)
    
    logger.info("Velocity model comparison analysis completed")
    return stats


def load_and_validate_velocity_models(model1_csv: str, model2_csv: str, 
                                     config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and validate velocity model CSV files.
    
    Args:
        model1_csv: Path to first model CSV
        model2_csv: Path to second model CSV
        config: Configuration dictionary
        
    Returns:
        Tuple of (model1_df, model2_df)
    """
    logger.info("Loading and validating velocity models")
    
    df1 = read_csv_file(model1_csv)
    df2 = read_csv_file(model2_csv)
    
    logger.info(f"Model 1: {len(df1)} rows, {len(df1.columns)} columns")
    logger.info(f"Model 2: {len(df2)} rows, {len(df2.columns)} columns")
    
    # Validate structure
    if len(df1) != len(df2):
        raise ValueError(f"Row count mismatch: Model 1 has {len(df1)}, Model 2 has {len(df2)}")
    
    required_cols = ['vp', 'vs', 'X_grid_km', 'Y_grid_km', 'Z_grid_km', 'i', 'j', 'k']
    for col in required_cols:
        if col not in df1.columns:
            raise ValueError(f"Required column '{col}' missing from Model 1")
        if col not in df2.columns:
            raise ValueError(f"Required column '{col}' missing from Model 2")
    
    # Check coordinate consistency
    if config['quality_control'].get('check_coordinate_consistency', True):
        coord_cols = ['X_grid_km', 'Y_grid_km', 'Z_grid_km', 'i', 'j', 'k']
        for col in coord_cols:
            if not np.allclose(df1[col].values, df2[col].values, rtol=1e-6):
                logger.warning(f"Coordinate column '{col}' differs between models")
    
    logger.info("Velocity model validation completed")
    return df1, df2


def compute_velocity_differences(df1: pd.DataFrame, df2: pd.DataFrame, 
                               config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Compute velocity differences between models.
    
    Args:
        df1: First velocity model dataframe
        df2: Second velocity model dataframe
        config: Configuration dictionary
        
    Returns:
        Dictionary containing difference arrays
    """
    logger.info("Computing velocity differences")
    
    # Basic differences
    vp_diff = df2['vp'].values - df1['vp'].values
    vs_diff = df2['vs'].values - df1['vs'].values
    
    # Ratios and ratio differences
    vp1, vs1 = df1['vp'].values, df1['vs'].values
    vp2, vs2 = df2['vp'].values, df2['vs'].values
    
    ratio1 = vp1 / vs1
    ratio2 = vp2 / vs2
    ratio_diff = ratio2 - ratio1
    
    # Percent changes
    vp_percent = (vp_diff / vp1) * 100
    vs_percent = (vs_diff / vs1) * 100
    
    # Logarithmic ratios if enabled
    if config['velocity_analysis'].get('compute_log_ratios', True):
        vp_log_ratio = np.log(vp2 / vp1)
        vs_log_ratio = np.log(vs2 / vs1)
    else:
        vp_log_ratio = None
        vs_log_ratio = None
    
    differences = {
        'vp_diff': vp_diff,
        'vs_diff': vs_diff,
        'ratio_diff': ratio_diff,
        'vp_percent': vp_percent,
        'vs_percent': vs_percent,
        'vp1': vp1, 'vp2': vp2,
        'vs1': vs1, 'vs2': vs2,
        'ratio1': ratio1, 'ratio2': ratio2,
        'vp_log_ratio': vp_log_ratio,
        'vs_log_ratio': vs_log_ratio
    }
    
    logger.info(f"Vp differences: {vp_diff.min():.3f} to {vp_diff.max():.3f} km/s")
    logger.info(f"Vs differences: {vs_diff.min():.3f} to {vs_diff.max():.3f} km/s")
    
    return differences


def compute_velocity_statistics(df1: pd.DataFrame, df2: pd.DataFrame,
                              differences: Dict[str, np.ndarray], 
                              config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute comprehensive velocity comparison statistics.
    
    Args:
        df1: First velocity model dataframe
        df2: Second velocity model dataframe
        differences: Differences dictionary
        config: Configuration dictionary
        
    Returns:
        Dictionary of statistics
    """
    logger.info("Computing velocity statistics")
    
    stats = {}
    total_points = len(differences['vp_diff'])
    
    # Vp comparison statistics
    vp_diff = differences['vp_diff']
    vp_percent = differences['vp_percent']
    
    stats['vp_comparison'] = {
        'model1_mean': differences['vp1'].mean(),
        'model1_std': differences['vp1'].std(),
        'model1_min': differences['vp1'].min(),
        'model1_max': differences['vp1'].max(),
        'model2_mean': differences['vp2'].mean(),
        'model2_std': differences['vp2'].std(),
        'model2_min': differences['vp2'].min(),
        'model2_max': differences['vp2'].max(),
        'diff_mean': vp_diff.mean(),
        'diff_std': vp_diff.std(),
        'diff_min': vp_diff.min(),
        'diff_max': vp_diff.max(),
        'diff_abs_mean': np.abs(vp_diff).mean(),
        'diff_rms': np.sqrt(np.mean(vp_diff**2)),
        'percent_mean': vp_percent.mean(),
        'percent_std': vp_percent.std(),
        'points_increased': np.sum(vp_diff > 0),
        'points_decreased': np.sum(vp_diff < 0),
        'points_unchanged': np.sum(np.abs(vp_diff) < 1e-6),
        'correlation': np.corrcoef(differences['vp1'], differences['vp2'])[0,1]
    }
    
    # Vs comparison statistics
    vs_diff = differences['vs_diff']
    vs_percent = differences['vs_percent']
    
    stats['vs_comparison'] = {
        'model1_mean': differences['vs1'].mean(),
        'model1_std': differences['vs1'].std(),
        'model1_min': differences['vs1'].min(),
        'model1_max': differences['vs1'].max(),
        'model2_mean': differences['vs2'].mean(),
        'model2_std': differences['vs2'].std(),
        'model2_min': differences['vs2'].min(),
        'model2_max': differences['vs2'].max(),
        'diff_mean': vs_diff.mean(),
        'diff_std': vs_diff.std(),
        'diff_min': vs_diff.min(),
        'diff_max': vs_diff.max(),
        'diff_abs_mean': np.abs(vs_diff).mean(),
        'diff_rms': np.sqrt(np.mean(vs_diff**2)),
        'percent_mean': vs_percent.mean(),
        'percent_std': vs_percent.std(),
        'points_increased': np.sum(vs_diff > 0),
        'points_decreased': np.sum(vs_diff < 0),
        'points_unchanged': np.sum(np.abs(vs_diff) < 1e-6),
        'correlation': np.corrcoef(differences['vs1'], differences['vs2'])[0,1]
    }
    
    # Vp/Vs ratio comparison
    ratio_diff = differences['ratio_diff']
    stats['ratio_comparison'] = {
        'model1_mean': differences['ratio1'].mean(),
        'model1_std': differences['ratio1'].std(),
        'model2_mean': differences['ratio2'].mean(),
        'model2_std': differences['ratio2'].std(),
        'diff_mean': ratio_diff.mean(),
        'diff_std': ratio_diff.std(),
        'diff_min': ratio_diff.min(),
        'diff_max': ratio_diff.max(),
        'correlation': np.corrcoef(differences['ratio1'], differences['ratio2'])[0,1]
    }
    
    # Log ratio statistics if computed
    if differences['vp_log_ratio'] is not None:
        vp_log = differences['vp_log_ratio']
        vs_log = differences['vs_log_ratio']
        
        stats['vp_log_ratio'] = {
            'mean': vp_log.mean(),
            'std': vp_log.std(),
            'min': vp_log.min(),
            'max': vp_log.max(),
            'equivalent_percent_change': (np.exp(vp_log.mean()) - 1) * 100
        }
        
        stats['vs_log_ratio'] = {
            'mean': vs_log.mean(),
            'std': vs_log.std(),
            'min': vs_log.min(),
            'max': vs_log.max(),
            'equivalent_percent_change': (np.exp(vs_log.mean()) - 1) * 100
        }
    
    return stats


def find_extreme_velocity_profiles(df: pd.DataFrame, differences: Dict[str, np.ndarray],
                                  config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find locations with extreme velocity differences.
    
    Args:
        df: Velocity dataframe (either model)
        differences: Differences dictionary
        config: Configuration dictionary
        
    Returns:
        Dictionary with extreme profile information
    """
    logger.info("Finding extreme velocity difference profiles")
    
    vp_diff = differences['vp_diff']
    vs_diff = differences['vs_diff']
    total_diff = np.sqrt(vp_diff**2 + vs_diff**2)
    
    # Add total difference to dataframe for grouping
    df_with_diff = df.copy()
    df_with_diff['total_diff'] = total_diff
    
    # Group by X,Y coordinates to get vertical profiles
    profile_stats = df_with_diff.groupby(['X_grid_km', 'Y_grid_km']).agg({
        'total_diff': ['mean', 'max', 'sum', 'std'],
        'Z_grid_km': ['min', 'max', 'count']
    }).round(3)
    
    # Flatten column names
    profile_stats.columns = ['_'.join(col).strip() for col in profile_stats.columns]
    
    # Find profiles with maximum and minimum average differences
    max_profile_coords = profile_stats['total_diff_mean'].idxmax()
    min_profile_coords = profile_stats[profile_stats['total_diff_mean'] > 1e-6]['total_diff_mean'].idxmin()
    
    # Get profile masks
    max_profile_mask = ((df['X_grid_km'] == max_profile_coords[0]) & 
                       (df['Y_grid_km'] == max_profile_coords[1]))
    min_profile_mask = ((df['X_grid_km'] == min_profile_coords[0]) & 
                       (df['Y_grid_km'] == min_profile_coords[1]))
    
    # Find depth slices with extreme differences
    depth_diff_stats = df_with_diff.groupby('Z_grid_km')['total_diff'].mean()
    max_diff_depth = depth_diff_stats.idxmax()
    min_diff_depth = depth_diff_stats[depth_diff_stats > 1e-6].idxmin()
    
    logger.info(f"Max difference profile at ({max_profile_coords[0]:.1f}, {max_profile_coords[1]:.1f}) km")
    logger.info(f"Min difference profile at ({min_profile_coords[0]:.1f}, {min_profile_coords[1]:.1f}) km")
    logger.info(f"Max difference depth: {max_diff_depth:.1f} km")
    logger.info(f"Min difference depth: {min_diff_depth:.1f} km")
    
    return {
        'max_profile_coords': max_profile_coords,
        'min_profile_coords': min_profile_coords,
        'max_profile_mask': max_profile_mask,
        'min_profile_mask': min_profile_mask,
        'profile_stats': profile_stats,
        'max_diff_depth': max_diff_depth,
        'min_diff_depth': min_diff_depth,
        'total_diff': total_diff
    }


def create_velocity_comparison_plots(df1: pd.DataFrame, df2: pd.DataFrame,
                                   differences: Dict[str, np.ndarray],
                                   extremes: Dict[str, Any],
                                   config: Dict[str, Any]) -> None:
    """
    Create comprehensive velocity comparison plots.
    
    Args:
        df1: First velocity model dataframe
        df2: Second velocity model dataframe  
        differences: Differences dictionary
        extremes: Extreme profiles dictionary
        config: Configuration dictionary
    """
    output_dir = Path(config['velocity_analysis']['output_dir'])
    plots_dir = output_dir / 'plots'
    ensure_directory_exists(plots_dir)
    
    output_format = config['analysis'].get('output_format', 'png')
    
    # Basic comparison plots
    create_basic_velocity_comparison_plots(differences, plots_dir, output_format, config)
    
    # Spatial distribution plots
    if config['velocity_analysis'].get('enable_spatial_projections', True):
        create_spatial_velocity_plots(df1, differences, plots_dir, output_format, config)
    
    # Vertical profile plots
    if config['velocity_analysis'].get('enable_profile_analysis', True):
        create_velocity_profile_plots(df1, df2, differences, extremes, plots_dir, output_format, config)
    
    # Horizontal slice analysis
    if config['velocity_analysis'].get('enable_slice_analysis', True):
        create_velocity_slice_plots(df1, df2, differences, extremes, plots_dir, output_format, config)


def create_basic_velocity_comparison_plots(differences: Dict[str, np.ndarray],
                                         plots_dir: Path, output_format: str,
                                         config: Dict[str, Any]) -> None:
    """Create basic velocity comparison scatter plots and histograms."""
    comparison_mode = config['velocity_analysis']['comparison_mode']
    model1_name = get_model_name(config, 1)
    model2_name = get_model_name(config, 2)
    difference_label = f"{model2_name} - {model1_name}"
    
    fig, axes = create_multi_panel_figure(2, 3, figsize=(20, 14),
                                        title=f'Velocity Model Comparison: {model1_name} vs {model2_name}')
    
    # Vp comparison scatter
    axes[0].scatter(differences['vp1'], differences['vp2'], alpha=0.6, s=2)
    min_vp = min(differences['vp1'].min(), differences['vp2'].min())
    max_vp = max(differences['vp1'].max(), differences['vp2'].max())
    axes[0].plot([min_vp, max_vp], [min_vp, max_vp], 'r--', linewidth=2, label='1:1 line')
    axes[0].set_xlabel(f'{model1_name} Vp (km/s)')
    axes[0].set_ylabel(f'{model2_name} Vp (km/s)')
    axes[0].set_title('Vp Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Vs comparison scatter
    axes[1].scatter(differences['vs1'], differences['vs2'], alpha=0.6, s=2)
    min_vs = min(differences['vs1'].min(), differences['vs2'].min())
    max_vs = max(differences['vs1'].max(), differences['vs2'].max())
    axes[1].plot([min_vs, max_vs], [min_vs, max_vs], 'r--', linewidth=2, label='1:1 line')
    axes[1].set_xlabel(f'{model1_name} Vs (km/s)')
    axes[1].set_ylabel(f'{model2_name} Vs (km/s)')
    axes[1].set_title('Vs Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Vp/Vs ratio comparison
    axes[2].scatter(differences['ratio1'], differences['ratio2'], alpha=0.6, s=2)
    min_ratio = min(differences['ratio1'].min(), differences['ratio2'].min())
    max_ratio = max(differences['ratio1'].max(), differences['ratio2'].max())
    axes[2].plot([min_ratio, max_ratio], [min_ratio, max_ratio], 'r--', linewidth=2, label='1:1 line')
    axes[2].set_xlabel(f'{model1_name} Vp/Vs Ratio')
    axes[2].set_ylabel(f'{model2_name} Vp/Vs Ratio')
    axes[2].set_title('Vp/Vs Ratio Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Difference histograms
    axes[3].hist(differences['vp_diff'], bins=50, alpha=0.7, color='blue', 
                edgecolor='black', linewidth=0.5)
    axes[3].axvline(differences['vp_diff'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {differences["vp_diff"].mean():.3f}')
    axes[3].set_xlabel(f'Vp Difference (km/s)\n{difference_label}')
    axes[3].set_ylabel('Frequency')
    axes[3].set_title('Vp Differences Distribution')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    axes[4].hist(differences['vs_diff'], bins=50, alpha=0.7, color='green', 
                edgecolor='black', linewidth=0.5)
    axes[4].axvline(differences['vs_diff'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {differences["vs_diff"].mean():.3f}')
    axes[4].set_xlabel(f'Vs Difference (km/s)\n{difference_label}')
    axes[4].set_ylabel('Frequency')
    axes[4].set_title('Vs Differences Distribution')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    
    axes[5].hist(differences['ratio_diff'], bins=50, alpha=0.7, color='purple', 
                edgecolor='black', linewidth=0.5)
    axes[5].axvline(differences['ratio_diff'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {differences["ratio_diff"].mean():.3f}')
    axes[5].set_xlabel(f'Vp/Vs Ratio Difference\n{difference_label}')
    axes[5].set_ylabel('Frequency')
    axes[5].set_title('Ratio Differences Distribution')
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)
    
    save_figure(fig, plots_dir / 'basic_velocity_comparison', output_format)


def create_spatial_velocity_plots(df: pd.DataFrame, differences: Dict[str, np.ndarray],
                                 plots_dir: Path, output_format: str,
                                 config: Dict[str, Any]) -> None:
    """Create spatial distribution plots of velocity differences."""
    x_coords = df['X_grid_km'].values
    y_coords = df['Y_grid_km'].values
    z_coords = df['Z_grid_km'].values
    
    # Create RMS spatial projections
    df_temp = df.copy()
    df_temp['vp_diff'] = differences['vp_diff']
    df_temp['vs_diff'] = differences['vs_diff']
    
    fig, axes = create_multi_panel_figure(2, 4, figsize=(25, 14),
                                        title='Velocity Changes: RMS Spatial Projections')
    
    # Vp projections
    create_velocity_spatial_projection(df_temp, 'vp_diff', 'Vp', axes[0:4], config)
    
    # Vs projections  
    create_velocity_spatial_projection(df_temp, 'vs_diff', 'Vs', axes[4:8], config)
    
    save_figure(fig, plots_dir / 'spatial_velocity_projections', output_format)


def create_velocity_spatial_projection(df_temp: pd.DataFrame, diff_col: str, 
                                     velocity_type: str, axes: list,
                                     config: Dict[str, Any]) -> None:
    """Create spatial projection for a single velocity type."""
    grid_resolution = config['velocity_analysis'].get('grid_resolution', 100)
    interpolation_method = config['velocity_analysis'].get('interpolation_method', 'cubic')
    
    # XY plane: RMS along Z (depth)
    xy_rms = df_temp.groupby(['X_grid_km', 'Y_grid_km']).agg({
        diff_col: lambda x: np.sqrt(np.mean(x**2))
    }).reset_index()
    
    # XZ plane: RMS along Y  
    xz_rms = df_temp.groupby(['X_grid_km', 'Z_grid_km']).agg({
        diff_col: lambda x: np.sqrt(np.mean(x**2))
    }).reset_index()
    
    # YZ plane: RMS along X
    yz_rms = df_temp.groupby(['Y_grid_km', 'Z_grid_km']).agg({
        diff_col: lambda x: np.sqrt(np.mean(x**2))
    }).reset_index()
    
    # Create grids and interpolate
    x_coords = df_temp['X_grid_km'].values
    y_coords = df_temp['Y_grid_km'].values
    z_coords = df_temp['Z_grid_km'].values
    
    xi = np.linspace(x_coords.min(), x_coords.max(), grid_resolution)
    yi = np.linspace(y_coords.min(), y_coords.max(), grid_resolution)
    zi = np.linspace(z_coords.min(), z_coords.max(), grid_resolution)
    
    # XY projection
    Xi, Yi = np.meshgrid(xi, yi)
    xy_interp = griddata((xy_rms['X_grid_km'], xy_rms['Y_grid_km']), 
                        xy_rms[diff_col], (Xi, Yi), method=interpolation_method)
    
    vmax = max(xy_rms[diff_col].max(), xz_rms[diff_col].max(), yz_rms[diff_col].max())
    
    im1 = axes[0].imshow(xy_interp, extent=[xi.min(), xi.max(), yi.min(), yi.max()], 
                        cmap='Reds', vmin=0, vmax=vmax, origin='lower', aspect='auto')
    axes[0].set_xlabel('X (km)')
    axes[0].set_ylabel('Y (km)')
    axes[0].set_title(f'{velocity_type} RMS Changes - XY Projection')
    plt.colorbar(im1, ax=axes[0], label=f'{velocity_type} RMS (km/s)')
    
    # XZ projection
    Xi_xz, Zi = np.meshgrid(xi, zi)
    xz_interp = griddata((xz_rms['X_grid_km'], xz_rms['Z_grid_km']), 
                        xz_rms[diff_col], (Xi_xz, Zi), method=interpolation_method)
    
    im2 = axes[1].imshow(xz_interp, extent=[xi.min(), xi.max(), zi.max(), zi.min()], 
                        cmap='Reds', vmin=0, vmax=vmax, origin='upper', aspect='auto')
    axes[1].set_xlabel('X (km)')
    axes[1].set_ylabel('Z (km)')
    axes[1].set_title(f'{velocity_type} RMS Changes - XZ Projection')
    plt.colorbar(im2, ax=axes[1], label=f'{velocity_type} RMS (km/s)')
    
    # YZ projection
    Yi_yz, Zi_yz = np.meshgrid(yi, zi)
    yz_interp = griddata((yz_rms['Y_grid_km'], yz_rms['Z_grid_km']), 
                        yz_rms[diff_col], (Yi_yz, Zi_yz), method=interpolation_method)
    
    im3 = axes[2].imshow(yz_interp, extent=[yi.min(), yi.max(), zi.max(), zi.min()], 
                        cmap='Reds', vmin=0, vmax=vmax, origin='upper', aspect='auto')
    axes[2].set_xlabel('Y (km)')
    axes[2].set_ylabel('Z (km)')
    axes[2].set_title(f'{velocity_type} RMS Changes - YZ Projection')
    plt.colorbar(im3, ax=axes[2], label=f'{velocity_type} RMS (km/s)')
    
    # Depth profile
    axes[3].scatter(df_temp[diff_col], z_coords, alpha=0.6, s=2)
    axes[3].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[3].set_xlabel(f'{velocity_type} Difference (km/s)')
    axes[3].set_ylabel('Depth (km)')
    axes[3].set_title(f'{velocity_type} Differences vs Depth')
    axes[3].invert_yaxis()
    axes[3].grid(True, alpha=0.3)


def create_velocity_profile_plots(df1: pd.DataFrame, df2: pd.DataFrame,
                                differences: Dict[str, np.ndarray],
                                extremes: Dict[str, Any],
                                plots_dir: Path, output_format: str,
                                config: Dict[str, Any]) -> None:
    """Create vertical profile plots at extreme difference locations."""
    model1_name = get_model_name(config, 1)
    model2_name = get_model_name(config, 2)
    
    fig, axes = create_multi_panel_figure(2, 2, figsize=(16, 14),
                                        title=f'Vertical Velocity Profiles: {model1_name} vs {model2_name}')
    
    # Get profile data for max difference location
    max_profile_data1 = df1[extremes['max_profile_mask']].sort_values('Z_grid_km')
    max_profile_data2 = df2[extremes['max_profile_mask']].sort_values('Z_grid_km')
    
    # Get profile data for min difference location  
    min_profile_data1 = df1[extremes['min_profile_mask']].sort_values('Z_grid_km')
    min_profile_data2 = df2[extremes['min_profile_mask']].sort_values('Z_grid_km')
    
    # Max difference location - Vp profile
    axes[0].plot(max_profile_data1['vp'], max_profile_data1['Z_grid_km'], 'b-',
                label=model1_name, linewidth=3, alpha=0.7)
    axes[0].plot(max_profile_data2['vp'], max_profile_data2['Z_grid_km'], 'r-',
                label=model2_name, linewidth=3, alpha=0.7)
    axes[0].set_xlabel('Vp (km/s)')
    axes[0].set_ylabel('Depth (km)')
    axes[0].set_title(f'Vp Profile at Max Diff Location\n({extremes["max_profile_coords"][0]:.1f}, {extremes["max_profile_coords"][1]:.1f}) km')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_yaxis()
    
    # Max difference location - Vs profile
    axes[1].plot(max_profile_data1['vs'], max_profile_data1['Z_grid_km'], 'b-',
                label=model1_name, linewidth=3, alpha=0.7)
    axes[1].plot(max_profile_data2['vs'], max_profile_data2['Z_grid_km'], 'r-',
                label=model2_name, linewidth=3, alpha=0.7)
    axes[1].set_xlabel('Vs (km/s)')
    axes[1].set_ylabel('Depth (km)')
    axes[1].set_title(f'Vs Profile at Max Diff Location\n({extremes["max_profile_coords"][0]:.1f}, {extremes["max_profile_coords"][1]:.1f}) km')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_yaxis()
    
    # Min difference location - Vp profile
    axes[2].plot(min_profile_data1['vp'], min_profile_data1['Z_grid_km'], 'b-',
                label=model1_name, linewidth=3, alpha=0.7)
    axes[2].plot(min_profile_data2['vp'], min_profile_data2['Z_grid_km'], 'r-',
                label=model2_name, linewidth=3, alpha=0.7)
    axes[2].set_xlabel('Vp (km/s)')
    axes[2].set_ylabel('Depth (km)')
    axes[2].set_title(f'Vp Profile at Min Diff Location\n({extremes["min_profile_coords"][0]:.1f}, {extremes["min_profile_coords"][1]:.1f}) km')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].invert_yaxis()
    
    # Min difference location - Vs profile
    axes[3].plot(min_profile_data1['vs'], min_profile_data1['Z_grid_km'], 'b-',
                label=model1_name, linewidth=3, alpha=0.7)
    axes[3].plot(min_profile_data2['vs'], min_profile_data2['Z_grid_km'], 'r-',
                label=model2_name, linewidth=3, alpha=0.7)
    axes[3].set_xlabel('Vs (km/s)')
    axes[3].set_ylabel('Depth (km)')
    axes[3].set_title(f'Vs Profile at Min Diff Location\n({extremes["min_profile_coords"][0]:.1f}, {extremes["min_profile_coords"][1]:.1f}) km')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].invert_yaxis()
    
    save_figure(fig, plots_dir / 'velocity_vertical_profiles', output_format)


def create_velocity_slice_plots(df1: pd.DataFrame, df2: pd.DataFrame,
                               differences: Dict[str, np.ndarray],
                               extremes: Dict[str, Any],
                               plots_dir: Path, output_format: str,
                               config: Dict[str, Any]) -> None:
    """Create horizontal slice comparison plots."""
    # Implementation would create depth slice heatmaps similar to the original code
    # This is a simplified version focusing on structure
    pass


def get_model_name(config: Dict[str, Any], model_num: int) -> str:
    """Get model name based on configuration."""
    comparison_mode = config['velocity_analysis']['comparison_mode']
    
    if comparison_mode == "initial_vs_final":
        return "Initial Model" if model_num == 1 else "Final Model"
    elif comparison_mode == "method_comparison":
        return "Method 1" if model_num == 1 else "Method 2"
    elif comparison_mode == "custom":
        return config['velocity_analysis']['custom_title1'] if model_num == 1 else config['velocity_analysis']['custom_title2']
    else:
        return config['velocity_analysis']['model1_name'] if model_num == 1 else config['velocity_analysis']['model2_name']


def generate_velocity_comparison_report(df1: pd.DataFrame, df2: pd.DataFrame,
                                       differences: Dict[str, np.ndarray],
                                       stats: Dict[str, Any],
                                       extremes: Dict[str, Any],
                                       config: Dict[str, Any]) -> None:
    """Generate comprehensive velocity comparison report."""
    output_dir = Path(config['velocity_analysis']['output_dir'])
    reports_dir = output_dir / 'reports'
    ensure_directory_exists(reports_dir)
    
    report_file = reports_dir / 'velocity_comparison_report.txt'
    
    model1_name = get_model_name(config, 1)
    model2_name = get_model_name(config, 2)
    comparison_mode = config['velocity_analysis']['comparison_mode']
    
    report_lines = [
        "VELOCITY MODEL COMPARISON ANALYSIS REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Comparison Mode: {comparison_mode}",
        f"Model 1: {model1_name}",
        f"Model 2: {model2_name}",
        "",
        "SUMMARY",
        "-" * 20,
        f"Total comparison points: {len(differences['vp_diff'])}",
        f"Vp correlation: {stats['vp_comparison']['correlation']:.4f}",
        f"Vs correlation: {stats['vs_comparison']['correlation']:.4f}",
        f"Vp/Vs ratio correlation: {stats['ratio_comparison']['correlation']:.4f}",
        ""
    ]
    
    # Add detailed statistics for each velocity type
    for vel_type in ['vp', 'vs']:
        vel_stats = stats[f'{vel_type}_comparison']
        report_lines.extend([
            f"{vel_type.upper()} VELOCITY COMPARISON",
            "-" * 30,
            f"{model1_name}:",
            f"  Mean: {vel_stats['model1_mean']:.3f} ± {vel_stats['model1_std']:.3f} km/s",
            f"  Range: {vel_stats['model1_min']:.3f} - {vel_stats['model1_max']:.3f} km/s",
            "",
            f"{model2_name}:",
            f"  Mean: {vel_stats['model2_mean']:.3f} ± {vel_stats['model2_std']:.3f} km/s",
            f"  Range: {vel_stats['model2_min']:.3f} - {vel_stats['model2_max']:.3f} km/s",
            "",
            f"Differences ({model2_name} - {model1_name}):",
            f"  Mean: {vel_stats['diff_mean']:.3f} ± {vel_stats['diff_std']:.3f} km/s",
            f"  Range: {vel_stats['diff_min']:.3f} to {vel_stats['diff_max']:.3f} km/s",
            f"  Mean absolute: {vel_stats['diff_abs_mean']:.3f} km/s",
            f"  RMS: {vel_stats['diff_rms']:.3f} km/s",
            f"  Mean percent change: {vel_stats['percent_mean']:.2f}% ± {vel_stats['percent_std']:.2f}%",
            f"  Points increased: {vel_stats['points_increased']} ({100*vel_stats['points_increased']/len(differences['vp_diff']):.1f}%)",
            f"  Points decreased: {vel_stats['points_decreased']} ({100*vel_stats['points_decreased']/len(differences['vp_diff']):.1f}%)",
            f"  Correlation: {vel_stats['correlation']:.4f}",
            ""
        ])
    
    # Add log ratio statistics if available
    if f'{vel_type}_log_ratio' in stats:
        log_stats = stats[f'{vel_type}_log_ratio']
        report_lines.extend([
            f"Logarithmic ratio analysis:",
            f"  Mean ln(ratio): {log_stats['mean']:.4f}",
            f"  Std ln(ratio): {log_stats['std']:.4f}",
            f"  Equivalent % change: {log_stats['equivalent_percent_change']:.2f}%",
            ""
        ])
    
    # Ratio comparison
    ratio_stats = stats['ratio_comparison']
    report_lines.extend([
        "VP/VS RATIO COMPARISON",
        "-" * 30,
        f"{model1_name} ratio: {ratio_stats['model1_mean']:.3f} ± {ratio_stats['model1_std']:.3f}",
        f"{model2_name} ratio: {ratio_stats['model2_mean']:.3f} ± {ratio_stats['model2_std']:.3f}",
        f"Ratio difference: {ratio_stats['diff_mean']:.3f} ± {ratio_stats['diff_std']:.3f}",
        f"Ratio difference range: {ratio_stats['diff_min']:.3f} to {ratio_stats['diff_max']:.3f}",
        f"Ratio correlation: {ratio_stats['correlation']:.4f}",
        ""
    ])
    
    # Extreme locations
    max_profile_stats = extremes['profile_stats'].loc[extremes['max_profile_coords']]
    min_profile_stats = extremes['profile_stats'].loc[extremes['min_profile_coords']]
    
    report_lines.extend([
        "EXTREME DIFFERENCE LOCATIONS",
        "-" * 40,
        f"Maximum difference vertical profile:",
        f"  Location (X,Y): ({extremes['max_profile_coords'][0]:.1f}, {extremes['max_profile_coords'][1]:.1f}) km",
        f"  Depth range: {max_profile_stats['Z_grid_km_min']:.1f} to {max_profile_stats['Z_grid_km_max']:.1f} km",
        f"  Average total difference: {max_profile_stats['total_diff_mean']:.3f} km/s",
        "",
        f"Minimum difference vertical profile:",
        f"  Location (X,Y): ({extremes['min_profile_coords'][0]:.1f}, {extremes['min_profile_coords'][1]:.1f}) km",
        f"  Depth range: {min_profile_stats['Z_grid_km_min']:.1f} to {min_profile_stats['Z_grid_km_max']:.1f} km",
        f"  Average total difference: {min_profile_stats['total_diff_mean']:.3f} km/s",
        "",
        f"Depth slices:",
        f"  Maximum average difference depth: {extremes['max_diff_depth']:.1f} km",
        f"  Minimum average difference depth: {extremes['min_diff_depth']:.1f} km",
    ])
    
    write_text_file(report_file, '\n'.join(report_lines))
    logger.info(f"Velocity comparison report saved to: {report_file}")  