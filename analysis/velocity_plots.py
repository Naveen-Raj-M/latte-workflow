"""
Velocity model comparison plotting module.

This module creates all visualization plots for velocity model comparisons including:
- Basic scatter and histogram comparisons
- Spatial RMS projections (XY, XZ, YZ)
- Vertical profile comparisons
- Horizontal slice heatmaps
- Ratio analysis and geometric mean projections
- Domain overview plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Dict, Any, Tuple
from scipy.interpolate import griddata
import logging

from utils.plot_utils import save_figure, get_color_scheme
from .velocity_spatial import find_max_min_ratio_change_depths

logger = logging.getLogger(__name__)


def get_model_names(config: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Get model names based on configuration.
    
    Returns:
        Tuple of (model1_name, model2_name, difference_label)
    """
    comparison_mode = config['velocity_analysis']['comparison_mode']
    
    if comparison_mode == "initial_vs_final":
        return "Initial Model", "Final Model", "Final - Initial"
    elif comparison_mode == "method_comparison":
        return "Method 1", "Method 2", "Method 2 - Method 1"
    elif comparison_mode == "custom":
        model1 = config['velocity_analysis']['custom_title1']
        model2 = config['velocity_analysis']['custom_title2']
        return model1, model2, f"{model2} - {model1}"
    else:
        model1 = config['velocity_analysis']['model1_name']
        model2 = config['velocity_analysis']['model2_name']
        return model1, model2, f"{model2} - {model1}"


def create_basic_comparison_plots(differences: Dict[str, np.ndarray],
                                 plots_dir: Path, output_format: str,
                                 config: Dict[str, Any]) -> None:
    """
    Create basic velocity comparison scatter plots and histograms.
    
    Args:
        differences: Differences dictionary
        plots_dir: Directory to save plots
        output_format: Output format (png, pdf, svg)
        config: Configuration dictionary
    """
    model1_name, model2_name, difference_label = get_model_names(config)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()

    # Vp comparison scatter
    axes[0].scatter(differences['vp1'], differences['vp2'], alpha=0.6, s=2)
    max_vp = max(differences['vp1'].max(), differences['vp2'].max())
    axes[0].plot([0, max_vp], [0, max_vp], 'r--', linewidth=2, label='1:1 line')
    axes[0].set_xlabel(f'{model1_name} Vp (km/s)')
    axes[0].set_ylabel(f'{model2_name} Vp (km/s)')
    axes[0].set_title('Vp Comparison')
    # origin at (0,0)
    axes[0].set_xlim(left=0)
    axes[0].set_ylim(bottom=0)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Vs comparison scatter
    axes[1].scatter(differences['vs1'], differences['vs2'], alpha=0.6, s=2)
    max_vs = max(differences['vs1'].max(), differences['vs2'].max())
    axes[1].plot([0, max_vs], [0, max_vs], 'r--', linewidth=2, label='1:1 line')
    axes[1].set_xlabel(f'{model1_name} Vs (km/s)')
    axes[1].set_ylabel(f'{model2_name} Vs (km/s)')
    axes[1].set_title('Vs Comparison')
    # origin at (0,0)
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0)
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

    # Get combined limits from both histograms
    x_min = min(axes[3].get_xlim()[0], axes[4].get_xlim()[0])
    x_max = max(axes[3].get_xlim()[1], axes[4].get_xlim()[1])
    y_min = min(axes[3].get_ylim()[0], axes[4].get_ylim()[0])
    y_max = max(axes[3].get_ylim()[1], axes[4].get_ylim()[1])

    # Apply equal limits
    axes[3].set_xlim(x_min, x_max)
    axes[4].set_xlim(x_min, x_max)
    axes[3].set_ylim(y_min, y_max)
    axes[4].set_ylim(y_min, y_max)
    
    axes[5].hist(differences['ratio_diff'], bins=50, alpha=0.7, color='purple', 
                edgecolor='black', linewidth=0.5)
    axes[5].axvline(differences['ratio_diff'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {differences["ratio_diff"].mean():.3f}')
    axes[5].set_xlabel(f'Vp/Vs Ratio Difference\n{difference_label}')
    axes[5].set_ylabel('Frequency')
    axes[5].set_title('Ratio Differences Distribution')
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)
    
    plt.suptitle(f'Velocity Model Comparison: {model1_name} vs {model2_name}', fontsize=20)
    plt.tight_layout()
    save_figure(fig, plots_dir / 'basic_velocity_comparison', output_format)


def create_spatial_projection_plots(df: pd.DataFrame, differences: Dict[str, np.ndarray],
                                   plots_dir: Path, output_format: str,
                                   config: Dict[str, Any]) -> None:
    """
    Create spatial distribution plots of velocity differences with XY, XZ, YZ projections.
    
    Args:
        df: Velocity dataframe
        differences: Differences dictionary
        plots_dir: Directory to save plots
        output_format: Output format
        config: Configuration dictionary
    """
    model1_name, model2_name, difference_label = get_model_names(config)
    grid_resolution = config['velocity_analysis'].get('grid_resolution', 100)
    interpolation_method = config['velocity_analysis'].get('interpolation_method', 'cubic')
    
    x_coords = df['X_grid_km'].values
    y_coords = df['Y_grid_km'].values
    z_coords = df['Z_grid_km'].values
    
    # Create RMS spatial projections
    df_temp = df.copy()
    df_temp['vp_diff'] = differences['vp_diff']
    df_temp['vs_diff'] = differences['vs_diff']
    
    fig, axes = plt.subplots(2, 4, figsize=(25, 14))
    axes = axes.flatten()

    # Vp projections
    _create_velocity_spatial_projection(df_temp, 'vp_diff', 'Vp', axes[0:4], 
                                       grid_resolution, interpolation_method, difference_label)
    
    # Vs projections  
    _create_velocity_spatial_projection(df_temp, 'vs_diff', 'Vs', axes[4:8], 
                                       grid_resolution, interpolation_method, difference_label)
    
    plt.suptitle(f'Velocity Changes: RMS Spatial Projections\n{difference_label}', fontsize=20)
    plt.tight_layout()
    save_figure(fig, plots_dir / 'spatial_velocity_projections', output_format)


def _create_velocity_spatial_projection(df_temp: pd.DataFrame, diff_col: str, 
                                       velocity_type: str, axes: list,
                                       grid_resolution: int, interpolation_method: str,
                                       difference_label: str) -> None:
    """
    Create spatial projection for a single velocity type.
    
    Args:
        df_temp: Dataframe with difference columns
        diff_col: Difference column name
        velocity_type: Type of velocity (Vp or Vs)
        axes: List of 4 axes to plot on
        grid_resolution: Grid resolution for interpolation
        interpolation_method: Interpolation method
        difference_label: Label for difference
    """
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

def create_profile_comparison_plots(df1: pd.DataFrame, df2: pd.DataFrame,
                                   differences: Dict[str, np.ndarray],
                                   extremes: Dict[str, Any],
                                   plots_dir: Path, output_format: str,
                                   config: Dict[str, Any]) -> None:
    """
    Create vertical profile plots at extreme difference locations.
    
    Args:
        df1: First velocity model dataframe
        df2: Second velocity model dataframe
        differences: Differences dictionary
        extremes: Extreme profiles dictionary
        plots_dir: Directory to save plots
        output_format: Output format
        config: Configuration dictionary
    """
    from .velocity_plots import get_model_names
    model1_name, model2_name, _ = get_model_names(config)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
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
    
    plt.suptitle(f'Vertical Velocity Profiles: {model1_name} vs {model2_name}', fontsize=20)
    plt.tight_layout()
    save_figure(fig, plots_dir / 'velocity_vertical_profiles', output_format)


def create_slice_heatmap_plots(df1: pd.DataFrame, df2: pd.DataFrame,
                               differences: Dict[str, np.ndarray],
                               extremes: Dict[str, Any],
                               plots_dir: Path, output_format: str,
                               config: Dict[str, Any]) -> None:
    """
    Create horizontal slice comparison plots at extreme difference depths.
    
    Args:
        df1: First velocity model dataframe
        df2: Second velocity model dataframe
        differences: Differences dictionary
        extremes: Extreme profiles dictionary
        plots_dir: Directory to save plots
        output_format: Output format
        config: Configuration dictionary
    """
    from .velocity_plots import get_model_names
    model1_name, model2_name, difference_label = get_model_names(config)
    
    # Get data for max and min difference depth slices
    max_depth_data1 = df1[df1['Z_grid_km'] == extremes['max_diff_depth']]
    max_depth_data2 = df2[df2['Z_grid_km'] == extremes['max_diff_depth']]
    min_depth_data1 = df1[df1['Z_grid_km'] == extremes['min_diff_depth']]
    min_depth_data2 = df2[df2['Z_grid_km'] == extremes['min_diff_depth']]
    
    # Create grids for interpolation
    x_unique = np.sort(df1['X_grid_km'].unique())
    y_unique = np.sort(df1['Y_grid_km'].unique())
    X, Y = np.meshgrid(x_unique, y_unique)
    
    # Create difference plot
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    _plot_depth_slice_row(axes[0, :], max_depth_data1, max_depth_data2, X, Y,
                         extremes['max_diff_depth'], model1_name, model2_name,
                         difference_label, "Max Diff")
    
    _plot_depth_slice_row(axes[1, :], min_depth_data1, min_depth_data2, X, Y,
                         extremes['min_diff_depth'], model1_name, model2_name,
                         difference_label, "Min Diff")
    
    plt.suptitle(f'Depth Slice Heatmaps: {difference_label}', fontsize=20)
    plt.subplots_adjust(left=0.05, right=0.92, top=0.85, bottom=0.08, wspace=0.3, hspace=0.35)
    save_figure(fig, plots_dir / 'depth_slice_heatmaps', output_format)


def _plot_depth_slice_row(axes: np.ndarray, depth_data1: pd.DataFrame, 
                          depth_data2: pd.DataFrame, X: np.ndarray, Y: np.ndarray,
                          depth: float, model1_name: str, model2_name: str,
                          difference_label: str, slice_label: str) -> None:
    """
    Plot a single row of depth slice comparisons.
    
    Args:
        axes: Array of 4 axes
        depth_data1: First model depth slice data
        depth_data2: Second model depth slice data
        X, Y: Meshgrid coordinates
        depth: Depth value
        model1_name: First model name
        model2_name: Second model name
        difference_label: Label for differences
        slice_label: Label for this slice (Max/Min Diff)
    """
    if len(depth_data1) == 0 or len(depth_data2) == 0:
        return
    
    # Interpolate Vp
    Z1_vp = griddata((depth_data1['X_grid_km'], depth_data1['Y_grid_km']), 
                    depth_data1['vp'], (X, Y), method='linear')
    Z2_vp = griddata((depth_data2['X_grid_km'], depth_data2['Y_grid_km']), 
                    depth_data2['vp'], (X, Y), method='linear')
    
    # Interpolate Vs
    Z1_vs = griddata((depth_data1['X_grid_km'], depth_data1['Y_grid_km']), 
                    depth_data1['vs'], (X, Y), method='linear')
    Z2_vs = griddata((depth_data2['X_grid_km'], depth_data2['Y_grid_km']), 
                    depth_data2['vs'], (X, Y), method='linear')
    
    # Find common velocity ranges
    vp_min = min(np.nanmin(Z1_vp), np.nanmin(Z2_vp))
    vp_max = max(np.nanmax(Z1_vp), np.nanmax(Z2_vp))
    if vp_max - vp_min < 1e-10:
        vp_min, vp_max = vp_min - 0.1, vp_max + 0.1
    norm_vp = Normalize(vmin=vp_min, vmax=vp_max)
    
    # Plot Vp Model 1
    im1 = axes[0].contourf(X, Y, Z1_vp, levels=20, cmap='viridis', vmin=vp_min, vmax=vp_max)
    axes[0].set_title(f'{model1_name} Vp at {depth:.1f} km\n({slice_label} Slice)')
    axes[0].set_xlabel('X (km)')
    axes[0].set_ylabel('Y (km)')
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_vp, cmap='viridis'), ax=axes[0], label='Vp (km/s)')
    
    # Plot Vp Model 2
    im2 = axes[1].contourf(X, Y, Z2_vp, levels=20, cmap='viridis', vmin=vp_min, vmax=vp_max)
    axes[1].set_title(f'{model2_name} Vp at {depth:.1f} km\n({slice_label} Slice)')
    axes[1].set_xlabel('X (km)')
    axes[1].set_ylabel('Y (km)')
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_vp, cmap='viridis'), ax=axes[1], label='Vp (km/s)')
    
    # Plot Vp Difference
    Z_diff_vp = Z2_vp - Z1_vp
    vmax_vp = np.nanmax(np.abs(Z_diff_vp))
    if vmax_vp < 1e-10:
        vmax_vp = 0.1
    im3 = axes[2].contourf(X, Y, Z_diff_vp, levels=20, cmap='RdBu_r', vmin=-vmax_vp, vmax=vmax_vp)
    axes[2].set_title(f'Vp Difference at {depth:.1f} km\n{difference_label}')
    axes[2].set_xlabel('X (km)')
    axes[2].set_ylabel('Y (km)')
    plt.colorbar(im3, ax=axes[2], label='Vp Diff (km/s)')
    
    # Plot Vs Difference
    Z_diff_vs = Z2_vs - Z1_vs
    vmax_vs = np.nanmax(np.abs(Z_diff_vs))
    if vmax_vs < 1e-10:
        vmax_vs = 0.1
    im4 = axes[3].contourf(X, Y, Z_diff_vs, levels=20, cmap='RdBu_r', vmin=-vmax_vs, vmax=vmax_vs)
    axes[3].set_title(f'Vs Difference at {depth:.1f} km\n{difference_label}')
    axes[3].set_xlabel('X (km)')
    axes[3].set_ylabel('Y (km)')
    plt.colorbar(im4, ax=axes[3], label='Vs Diff (km/s)')


def create_domain_overview_plot(df: pd.DataFrame, extremes: Dict[str, Any],
                               plots_dir: Path, output_format: str,
                               config: Dict[str, Any]) -> None:
    """
    Create domain overview showing locations of extreme differences.
    
    Args:
        df: Velocity dataframe
        extremes: Extreme profiles dictionary
        plots_dir: Directory to save plots
        output_format: Output format
        config: Configuration dictionary
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    x_coords = df['X_grid_km'].values
    y_coords = df['Y_grid_km'].values
    z_coords = df['Z_grid_km'].values
    
    # Map view
    axes[0].scatter(x_coords, y_coords, c='lightgray', s=2, alpha=0.5, label='All points')
    axes[0].scatter(extremes['max_profile_coords'][0], extremes['max_profile_coords'][1],
                   c='red', s=150, marker='*', label='Max difference profile', 
                   edgecolor='black', linewidth=2)
    axes[0].scatter(extremes['min_profile_coords'][0], extremes['min_profile_coords'][1],
                   c='blue', s=150, marker='*', label='Min difference profile', 
                   edgecolor='black', linewidth=2)
    
    axes[0].set_xlabel('X (km)')
    axes[0].set_ylabel('Y (km)')
    axes[0].set_title('Domain Overview - Map View')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Cross section view (X-Z)
    axes[1].scatter(x_coords, z_coords, c='lightgray', s=2, alpha=0.5)
    
    # Show all points from the max and min profiles as vertical lines
    max_profile_data = extremes['max_profile_data']
    min_profile_data = extremes['min_profile_data']
    
    axes[1].scatter(max_profile_data['X_grid_km'], max_profile_data['Z_grid_km'],
                   c='red', s=30, marker='|', 
                   label=f'Max diff profile ({extremes["max_profile_coords"][0]:.1f}, {extremes["max_profile_coords"][1]:.1f})')
    axes[1].scatter(min_profile_data['X_grid_km'], min_profile_data['Z_grid_km'],
                   c='blue', s=30, marker='|', 
                   label=f'Min diff profile ({extremes["min_profile_coords"][0]:.1f}, {extremes["min_profile_coords"][1]:.1f})')
    
    # Mark slice planes
    axes[1].axhline(extremes['max_diff_depth'], color='red', linestyle='--', 
                   alpha=0.7, linewidth=2, label=f'Max diff slice ({extremes["max_diff_depth"]:.1f} km)')
    axes[1].axhline(extremes['min_diff_depth'], color='blue', linestyle='--', 
                   alpha=0.7, linewidth=2, label=f'Min diff slice ({extremes["min_diff_depth"]:.1f} km)')
    
    axes[1].set_xlabel('X (km)')
    axes[1].set_ylabel('Z (km)')
    axes[1].set_title('Domain Overview - Cross Section')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_yaxis()
    
    from .velocity_plots import get_model_names
    model1_name, model2_name, _ = get_model_names(config)
    plt.suptitle(f'Domain Overview: {model1_name} vs {model2_name}', fontsize=20)
    plt.tight_layout()
    save_figure(fig, plots_dir / 'domain_overview', output_format)

def create_ratio_analysis_plots(df1: pd.DataFrame, df2: pd.DataFrame,
                                differences: Dict[str, np.ndarray],
                                extremes: Dict[str, Any],
                                plots_dir: Path, output_format: str,
                                config: Dict[str, Any]) -> None:
    """
    Create ratio analysis plots including geometric mean spatial projections.
    
    Args:
        df1: First velocity model dataframe
        df2: Second velocity model dataframe
        differences: Differences dictionary
        extremes: Extreme profiles dictionary
        plots_dir: Directory to save plots
        output_format: Output format
        config: Configuration dictionary
    """
    # Create geometric mean ratio projections
    _create_geometric_ratio_projections(df1, df2, plots_dir, output_format, config)
    
    # Create ratio heatmaps at extreme depths
    _create_ratio_heatmaps(df1, df2, plots_dir, output_format, config)


def _create_geometric_ratio_projections(df1: pd.DataFrame, df2: pd.DataFrame,
                                       plots_dir: Path, output_format: str,
                                       config: Dict[str, Any]) -> None:
    """
    Create spatial distribution plots of geometric mean ratios with XY, XZ, YZ projections.
    
    Args:
        df1: First velocity model dataframe
        df2: Second velocity model dataframe
        plots_dir: Directory to save plots
        output_format: Output format
        config: Configuration dictionary
    """
    from .velocity_plots import get_model_names
    model1_name, model2_name, difference_label = get_model_names(config)
    
    grid_resolution = config['velocity_analysis'].get('grid_resolution', 100)
    interpolation_method = config['velocity_analysis'].get('interpolation_method', 'cubic')
    
    x_coords = df1['X_grid_km'].values
    y_coords = df1['Y_grid_km'].values
    z_coords = df1['Z_grid_km'].values
    
    # Calculate ratios
    vp_ratios = df2['vp'].values / df1['vp'].values
    vs_ratios = df2['vs'].values / df1['vs'].values
    
    # Add ratios to dataframe temporarily
    df_temp = df1.copy()
    df_temp['vp_ratio'] = vp_ratios
    df_temp['vs_ratio'] = vs_ratios
    
    fig, axes = plt.subplots(2, 4, figsize=(25, 14))
    axes = axes.flatten()
    
    # Vp ratio projections
    _create_ratio_spatial_projection(df_temp, 'vp_ratio', 'Vp', axes[0:4], 
                                    x_coords, y_coords, z_coords,
                                    grid_resolution, interpolation_method)
    
    # Vs ratio projections
    _create_ratio_spatial_projection(df_temp, 'vs_ratio', 'Vs', axes[4:8], 
                                    x_coords, y_coords, z_coords,
                                    grid_resolution, interpolation_method)
    
    plt.suptitle(f'Geometric Mean Ratio Spatial Projections\n{difference_label}', fontsize=20)
    plt.tight_layout()
    save_figure(fig, plots_dir / 'spatial_geometric_ratio_projections', output_format)


def _create_ratio_spatial_projection(df_temp: pd.DataFrame, ratio_col: str,
                                    velocity_type: str, axes: list,
                                    x_coords: np.ndarray, y_coords: np.ndarray, 
                                    z_coords: np.ndarray,
                                    grid_resolution: int, interpolation_method: str) -> None:
    """
    Create spatial projection for velocity ratios.
    
    Args:
        df_temp: Dataframe with ratio columns
        ratio_col: Ratio column name
        velocity_type: Type of velocity (Vp or Vs)
        axes: List of 4 axes to plot on
        x_coords, y_coords, z_coords: Coordinate arrays
        grid_resolution: Grid resolution for interpolation
        interpolation_method: Interpolation method
    """
    # XY plane: Geometric mean along Z (depth)
    xy_geomean = df_temp.groupby(['X_grid_km', 'Y_grid_km']).agg({
        ratio_col: lambda x: np.exp(np.mean(np.log(x)))
    }).reset_index()
    
    # XZ plane: Geometric mean along Y
    xz_geomean = df_temp.groupby(['X_grid_km', 'Z_grid_km']).agg({
        ratio_col: lambda x: np.exp(np.mean(np.log(x)))
    }).reset_index()
    
    # YZ plane: Geometric mean along X
    yz_geomean = df_temp.groupby(['Y_grid_km', 'Z_grid_km']).agg({
        ratio_col: lambda x: np.exp(np.mean(np.log(x)))
    }).reset_index()
    
    # Find min/max for consistent color scaling (centered around 1.0)
    all_ratios = np.concatenate([xy_geomean[ratio_col].values, 
                                 xz_geomean[ratio_col].values, 
                                 yz_geomean[ratio_col].values])
    ratio_range = max(abs(all_ratios.max() - 1), abs(all_ratios.min() - 1))
    vmin = 1 - ratio_range
    vmax = 1 + ratio_range
    
    # Create grids
    xi = np.linspace(x_coords.min(), x_coords.max(), grid_resolution)
    yi = np.linspace(y_coords.min(), y_coords.max(), grid_resolution)
    zi = np.linspace(z_coords.min(), z_coords.max(), grid_resolution)
    
    # XY projection
    Xi, Yi = np.meshgrid(xi, yi)
    geomean_xy_interp = griddata((xy_geomean['X_grid_km'], xy_geomean['Y_grid_km']), 
                                 xy_geomean[ratio_col], (Xi, Yi), method=interpolation_method)
    
    im1 = axes[0].imshow(geomean_xy_interp, extent=[xi.min(), xi.max(), yi.min(), yi.max()], 
                        cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
    axes[0].set_xlabel('X (km)')
    axes[0].set_ylabel('Y (km)')
    axes[0].set_title(f'{velocity_type} Geometric Mean Ratio - XY Projection')
    cbar1 = plt.colorbar(im1, ax=axes[0], label=f'{velocity_type} Ratio')
    cbar1.ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1)
    
    # XZ projection
    Xi_xz, Zi = np.meshgrid(xi, zi)
    geomean_xz_interp = griddata((xz_geomean['X_grid_km'], xz_geomean['Z_grid_km']), 
                                 xz_geomean[ratio_col], (Xi_xz, Zi), method=interpolation_method)
    
    im2 = axes[1].imshow(geomean_xz_interp, extent=[xi.min(), xi.max(), zi.max(), zi.min()], 
                        cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='upper', aspect='auto')
    axes[1].set_xlabel('X (km)')
    axes[1].set_ylabel('Z (km)')
    axes[1].set_title(f'{velocity_type} Geometric Mean Ratio - XZ Projection')
    cbar2 = plt.colorbar(im2, ax=axes[1], label=f'{velocity_type} Ratio')
    cbar2.ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1)
    
    # YZ projection
    Yi_yz, Zi_yz = np.meshgrid(yi, zi)
    geomean_yz_interp = griddata((yz_geomean['Y_grid_km'], yz_geomean['Z_grid_km']), 
                                 yz_geomean[ratio_col], (Yi_yz, Zi_yz), method=interpolation_method)
    
    im3 = axes[2].imshow(geomean_yz_interp, extent=[yi.min(), yi.max(), zi.max(), zi.min()], 
                        cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='upper', aspect='auto')
    axes[2].set_xlabel('Y (km)')
    axes[2].set_ylabel('Z (km)')
    axes[2].set_title(f'{velocity_type} Geometric Mean Ratio - YZ Projection')
    cbar3 = plt.colorbar(im3, ax=axes[2], label=f'{velocity_type} Ratio')
    cbar3.ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1)
    
    # Depth profile
    axes[3].scatter(df_temp[ratio_col], z_coords, alpha=0.6, s=2)
    axes[3].axvline(1.0, color='red', linestyle='--', linewidth=2)
    axes[3].set_xlabel(f'{velocity_type} Ratio')
    axes[3].set_ylabel('Depth (km)')
    axes[3].set_title(f'{velocity_type} Ratios vs Depth')
    axes[3].invert_yaxis()
    axes[3].grid(True, alpha=0.3)


def _create_ratio_heatmaps(df1: pd.DataFrame, df2: pd.DataFrame,
                          plots_dir: Path, output_format: str,
                          config: Dict[str, Any]) -> None:
    """
    Create ratio heatmap plots at extreme ratio change depths.
    
    Args:
        df1: First velocity model dataframe
        df2: Second velocity model dataframe
        plots_dir: Directory to save plots
        output_format: Output format
        config: Configuration dictionary
    """
    from .velocity_plots import get_model_names
    model1_name, model2_name, difference_label = get_model_names(config)
    
    # Find depth slices with maximum and minimum average ratio changes
    ratio_info = find_max_min_ratio_change_depths(df1, df2)
    
    # Get data for maximum and minimum average ratio change depths
    max_ratio_depth_data1 = df1[df1['Z_grid_km'] == ratio_info['max_change_depth']]
    max_ratio_depth_data2 = df2[df2['Z_grid_km'] == ratio_info['max_change_depth']]
    min_ratio_depth_data1 = df1[df1['Z_grid_km'] == ratio_info['min_change_depth']]
    min_ratio_depth_data2 = df2[df2['Z_grid_km'] == ratio_info['min_change_depth']]
    
    # Create grids
    x_unique = np.sort(df1['X_grid_km'].unique())
    y_unique = np.sort(df1['Y_grid_km'].unique())
    X, Y = np.meshgrid(x_unique, y_unique)
    
    # Create Vp ratio analysis figure
    _create_single_velocity_ratio_figure(max_ratio_depth_data1, max_ratio_depth_data2,
                                        min_ratio_depth_data1, min_ratio_depth_data2,
                                        X, Y, ratio_info, 'vp', 'Vp', 'viridis',
                                        model1_name, model2_name, difference_label,
                                        plots_dir, output_format)
    
    # Create Vs ratio analysis figure
    _create_single_velocity_ratio_figure(max_ratio_depth_data1, max_ratio_depth_data2,
                                        min_ratio_depth_data1, min_ratio_depth_data2,
                                        X, Y, ratio_info, 'vs', 'Vs', 'plasma',
                                        model1_name, model2_name, difference_label,
                                        plots_dir, output_format)


def _create_single_velocity_ratio_figure(max_data1: pd.DataFrame, max_data2: pd.DataFrame,
                                         min_data1: pd.DataFrame, min_data2: pd.DataFrame,
                                         X: np.ndarray, Y: np.ndarray, ratio_info: Dict[str, Any],
                                         vel_col: str, vel_label: str, vel_cmap: str,
                                         model1_name: str, model2_name: str, difference_label: str,
                                         plots_dir: Path, output_format: str) -> None:
    """
    Create a single velocity ratio analysis figure (Vp or Vs).
    
    Args:
        max_data1, max_data2: Max ratio depth data for both models
        min_data1, min_data2: Min ratio depth data for both models
        X, Y: Meshgrid coordinates
        ratio_info: Dictionary with ratio information
        vel_col: Velocity column name ('vp' or 'vs')
        vel_label: Velocity label ('Vp' or 'Vs')
        vel_cmap: Colormap for velocity
        model1_name, model2_name: Model names
        difference_label: Difference label
        plots_dir: Directory to save plots
        output_format: Output format
    """
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    # TOP ROW: Maximum average ratio change depth slice
    _plot_ratio_row(axes[0, :], max_data1, max_data2, X, Y, ratio_info['max_change_depth'],
                   vel_col, vel_label, vel_cmap, model1_name, model2_name, difference_label,
                   f"Avg change: {ratio_info[f'max_{vel_col}_avg_change']:.4f}", "Max Avg Ratio Change")
    
    # BOTTOM ROW: Minimum average ratio change depth slice
    _plot_ratio_row(axes[1, :], min_data1, min_data2, X, Y, ratio_info['min_change_depth'],
                   vel_col, vel_label, vel_cmap, model1_name, model2_name, difference_label,
                   f"Avg change: {ratio_info[f'min_{vel_col}_avg_change']:.4f}", "Min Avg Ratio Change")
    
    plt.suptitle(f'{vel_label} Ratio Analysis: {difference_label}', fontsize=20)
    plt.subplots_adjust(left=0.05, right=0.92, top=0.85, bottom=0.08, wspace=0.3, hspace=0.35)
    save_figure(fig, plots_dir / f'{vel_col}_ratio_analysis', output_format)


def _plot_ratio_row(axes: np.ndarray, depth_data1: pd.DataFrame, depth_data2: pd.DataFrame,
                   X: np.ndarray, Y: np.ndarray, depth: float,
                   vel_col: str, vel_label: str, vel_cmap: str,
                   model1_name: str, model2_name: str, difference_label: str,
                   avg_change_text: str, slice_label: str) -> None:
    """
    Plot a single row of ratio analysis.

    Args:
        axes: Array of 4 axes
        depth_data1, depth_data2: Depth slice data for both models
        X, Y: Meshgrid coordinates
        depth: Depth value
        vel_col: Velocity column name
        vel_label: Velocity label
        vel_cmap: Colormap for velocity
        model1_name, model2_name: Model names
        difference_label: Difference label
        avg_change_text: Text showing average change
        slice_label: Label for this slice
    """
    if len(depth_data1) == 0 or len(depth_data2) == 0:
        return

    # Interpolate velocities
    Z1 = griddata((depth_data1['X_grid_km'], depth_data1['Y_grid_km']),
                 depth_data1[vel_col], (X, Y), method='linear')
    Z2 = griddata((depth_data2['X_grid_km'], depth_data2['Y_grid_km']),
                 depth_data2[vel_col], (X, Y), method='linear')

    # Find common velocity ranges
    v_min = min(np.nanmin(Z1), np.nanmin(Z2))
    v_max = max(np.nanmax(Z1), np.nanmax(Z2))
    if v_max - v_min < 1e-10:
        v_min, v_max = v_min - 0.1, v_max + 0.1
    norm_v = Normalize(vmin=v_min, vmax=v_max)

    # Model 1
    im1 = axes[0].contourf(X, Y, Z1, levels=20, cmap=vel_cmap, vmin=v_min, vmax=v_max)
    axes[0].set_title(f'{model1_name} {vel_label}\nat {depth:.1f} km ({slice_label})')
    axes[0].set_xlabel('X (km)')
    axes[0].set_ylabel('Y (km)')
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_v, cmap=vel_cmap), ax=axes[0], label=f'{vel_label} (km/s)')

    # Model 2
    im2 = axes[1].contourf(X, Y, Z2, levels=20, cmap=vel_cmap, vmin=v_min, vmax=v_max)
    axes[1].set_title(f'{model2_name} {vel_label}\nat {depth:.1f} km ({slice_label})')
    axes[1].set_xlabel('X (km)')
    axes[1].set_ylabel('Y (km)')
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_v, cmap=vel_cmap), ax=axes[1], label=f'{vel_label} (km/s)')

    # Difference
    Z_diff = Z2 - Z1
    vmax_diff = np.nanmax(np.abs(Z_diff))
    if vmax_diff < 1e-10:
        vmax_diff = 0.1
    im3 = axes[2].contourf(X, Y, Z_diff, levels=20, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
    axes[2].set_title(f'{vel_label} Difference\nat {depth:.1f} km\n{difference_label}')
    axes[2].set_xlabel('X (km)')
    axes[2].set_ylabel('Y (km)')
    plt.colorbar(im3, ax=axes[2], label=f'{vel_label} Diff (km/s)')

    # Ratio - centered at 1.0
    Z_ratio = np.divide(Z2, Z1, out=np.ones_like(Z2), where=Z1!=0)
    ratio_deviations = np.abs(Z_ratio - 1.0)
    max_deviation = np.nanmax(ratio_deviations)
    if max_deviation < 1e-10:
        max_deviation = 0.1
    vmin_ratio = 1.0 - max_deviation
    vmax_ratio = 1.0 + max_deviation

    im4 = axes[3].contourf(X, Y, Z_ratio, levels=20, cmap='RdBu_r', vmin=vmin_ratio, vmax=vmax_ratio)
    axes[3].set_title(f'{vel_label} Ratio ({model2_name}/{model1_name})\nat {depth:.1f} km\n{avg_change_text}')
    axes[3].set_xlabel('X (km)')
    axes[3].set_ylabel('Y (km)')
    plt.colorbar(im4, ax=axes[3], label=f'{vel_label} Ratio')
    axes[3].contour(X, Y, Z_ratio, levels=[1.0], colors='black', linewidths=2, linestyles='--')


def create_plane_specific_ratio_plots(df1: pd.DataFrame, df2: pd.DataFrame,
                                      plots_dir: Path, output_format: str,
                                      config: Dict[str, Any]) -> None:
    """
    Create plane-specific ratio plots for user-specified XY, XZ, or YZ planes.
    Similar to depth_slice_heatmaps but uses ratios instead of differences.

    Args:
        df1: First velocity model dataframe
        df2: Second velocity model dataframe
        plots_dir: Directory to save plots
        output_format: Output format
        config: Configuration dictionary
    """
    from .velocity_plots import get_model_names
    model1_name, model2_name, _ = get_model_names(config)

    plane_config = config['velocity_analysis'].get('plane_ratio_plots', {})
    planes = plane_config.get('planes', [])

    # Get unique coordinates for validation
    x_unique = np.sort(df1['X_grid_km'].unique())
    y_unique = np.sort(df1['Y_grid_km'].unique())
    z_unique = np.sort(df1['Z_grid_km'].unique())

    for plane_spec in planes:
        plane_type = plane_spec.get('plane_type')
        coordinates = plane_spec.get('coordinates')
        label = plane_spec.get('label', '')

        # Skip if coordinates is null, empty, or not specified
        if not coordinates:
            logger.debug(f"Skipping {plane_type} plane (no coordinates specified)")
            continue

        # Ensure coordinates is a list
        if not isinstance(coordinates, list):
            coordinates = [coordinates]

        logger.info(f"Creating {len(coordinates)} {plane_type} plane ratio plot(s)")

        for coord in coordinates:
            if coord is None:
                continue

            logger.info(f"  - {plane_type} plane at coordinate {coord} km")

            if plane_type == 'xy':
                # Horizontal slice at specified Z depth
                _create_xy_plane_ratio_plot(df1, df2, coord, z_unique,
                                           model1_name, model2_name, label,
                                           plots_dir, output_format)
            elif plane_type == 'xz':
                # Vertical slice at specified Y coordinate
                _create_xz_plane_ratio_plot(df1, df2, coord, y_unique,
                                           model1_name, model2_name, label,
                                           plots_dir, output_format)
            elif plane_type == 'yz':
                # Vertical slice at specified X coordinate
                _create_yz_plane_ratio_plot(df1, df2, coord, x_unique,
                                           model1_name, model2_name, label,
                                           plots_dir, output_format)
            else:
                logger.warning(f"Unknown plane type: {plane_type}")


def _create_xy_plane_ratio_plot(df1: pd.DataFrame, df2: pd.DataFrame,
                                z_coord: float, z_unique: np.ndarray,
                                model1_name: str, model2_name: str, label: str,
                                plots_dir: Path, output_format: str) -> None:
    """Create XY plane (horizontal slice) ratio plot at specified Z coordinate."""
    # Find nearest Z coordinate
    nearest_z = z_unique[np.argmin(np.abs(z_unique - z_coord))]

    if abs(nearest_z - z_coord) > 0.1:
        logger.warning(f"Requested Z={z_coord} km, using nearest available Z={nearest_z} km")

    # Get data at this depth
    data1 = df1[df1['Z_grid_km'] == nearest_z]
    data2 = df2[df2['Z_grid_km'] == nearest_z]

    if len(data1) == 0 or len(data2) == 0:
        logger.warning(f"No data found at Z={nearest_z} km")
        return

    # Create meshgrid
    x_unique = np.sort(data1['X_grid_km'].unique())
    y_unique = np.sort(data1['Y_grid_km'].unique())
    X, Y = np.meshgrid(x_unique, y_unique)

    # Create 2x4 figure
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # TOP ROW: Vp
    _plot_plane_full_comparison_row(axes[0, :], data1, data2, X, Y, 'vp', 'Vp', 'viridis',
                                    model1_name, model2_name, nearest_z, label,
                                    'X (km)', 'Y (km)')

    # BOTTOM ROW: Vs
    _plot_plane_full_comparison_row(axes[1, :], data1, data2, X, Y, 'vs', 'Vs', 'plasma',
                                    model1_name, model2_name, nearest_z, label,
                                    'X (km)', 'Y (km)')

    plt.suptitle(f'XY Plane Analysis at Z={int(nearest_z)} km: {label}', fontsize=20)
    plt.subplots_adjust(left=0.05, right=0.92, top=0.85, bottom=0.08, wspace=0.3, hspace=0.35)
    save_figure(fig, plots_dir / f'xy_plane_ratio_z{int(nearest_z)}_km', output_format)


def _create_xz_plane_ratio_plot(df1: pd.DataFrame, df2: pd.DataFrame,
                                y_coord: float, y_unique: np.ndarray,
                                model1_name: str, model2_name: str, label: str,
                                plots_dir: Path, output_format: str) -> None:
    """Create XZ plane (vertical slice) ratio plot at specified Y coordinate."""
    # Find nearest Y coordinate
    nearest_y = y_unique[np.argmin(np.abs(y_unique - y_coord))]

    if abs(nearest_y - y_coord) > 0.1:
        logger.warning(f"Requested Y={y_coord} km, using nearest available Y={nearest_y} km")

    # Get data at this Y
    data1 = df1[df1['Y_grid_km'] == nearest_y]
    data2 = df2[df2['Y_grid_km'] == nearest_y]

    if len(data1) == 0 or len(data2) == 0:
        logger.warning(f"No data found at Y={nearest_y} km")
        return

    # Create meshgrid
    x_unique = np.sort(data1['X_grid_km'].unique())
    z_unique = np.sort(data1['Z_grid_km'].unique())
    X, Z = np.meshgrid(x_unique, z_unique)

    # Create 2x4 figure
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # TOP ROW: Vp
    _plot_plane_full_comparison_row(axes[0, :], data1, data2, X, Z, 'vp', 'Vp', 'viridis',
                                    model1_name, model2_name, nearest_y, label,
                                    'X (km)', 'Z (km)', coord1='X_grid_km',
                                    coord2='Z_grid_km', invert_y=True)

    # BOTTOM ROW: Vs
    _plot_plane_full_comparison_row(axes[1, :], data1, data2, X, Z, 'vs', 'Vs', 'plasma',
                                    model1_name, model2_name, nearest_y, label,
                                    'X (km)', 'Z (km)', coord1='X_grid_km',
                                    coord2='Z_grid_km', invert_y=True)

    plt.suptitle(f'XZ Plane Analysis at Y={int(nearest_y)} km: {label}', fontsize=20)
    plt.subplots_adjust(left=0.05, right=0.92, top=0.85, bottom=0.08, wspace=0.3, hspace=0.35)
    save_figure(fig, plots_dir / f'xz_plane_ratio_y{int(nearest_y)}_km', output_format)


def _create_yz_plane_ratio_plot(df1: pd.DataFrame, df2: pd.DataFrame,
                                x_coord: float, x_unique: np.ndarray,
                                model1_name: str, model2_name: str, label: str,
                                plots_dir: Path, output_format: str) -> None:
    """Create YZ plane (vertical slice) ratio plot at specified X coordinate."""
    # Find nearest X coordinate
    nearest_x = x_unique[np.argmin(np.abs(x_unique - x_coord))]

    if abs(nearest_x - x_coord) > 0.1:
        logger.warning(f"Requested X={x_coord} km, using nearest available X={nearest_x} km")

    # Get data at this X
    data1 = df1[df1['X_grid_km'] == nearest_x]
    data2 = df2[df2['X_grid_km'] == nearest_x]

    if len(data1) == 0 or len(data2) == 0:
        logger.warning(f"No data found at X={nearest_x} km")
        return

    # Create meshgrid
    y_unique = np.sort(data1['Y_grid_km'].unique())
    z_unique = np.sort(data1['Z_grid_km'].unique())
    Y, Z = np.meshgrid(y_unique, z_unique)

    # Create 2x4 figure
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # TOP ROW: Vp
    _plot_plane_full_comparison_row(axes[0, :], data1, data2, Y, Z, 'vp', 'Vp', 'viridis',
                                    model1_name, model2_name, nearest_x, label,
                                    'Y (km)', 'Z (km)', coord1='Y_grid_km',
                                    coord2='Z_grid_km', invert_y=True)

    # BOTTOM ROW: Vs
    _plot_plane_full_comparison_row(axes[1, :], data1, data2, Y, Z, 'vs', 'Vs', 'plasma',
                                    model1_name, model2_name, nearest_x, label,
                                    'Y (km)', 'Z (km)', coord1='Y_grid_km',
                                    coord2='Z_grid_km', invert_y=True)

    plt.suptitle(f'YZ Plane Analysis at X={int(nearest_x)} km: {label}', fontsize=20)
    plt.subplots_adjust(left=0.05, right=0.92, top=0.85, bottom=0.08, wspace=0.3, hspace=0.35)
    save_figure(fig, plots_dir / f'yz_plane_ratio_x{int(nearest_x)}_km', output_format)


def _plot_plane_full_comparison_row(axes: np.ndarray, data1: pd.DataFrame, data2: pd.DataFrame,
                                   X: np.ndarray, Y: np.ndarray, vel_col: str, vel_label: str,
                                   vel_cmap: str, model1_name: str, model2_name: str,
                                   coord_value: float, label: str, xlabel: str, ylabel: str,
                                   coord1: str = 'X_grid_km', coord2: str = 'Y_grid_km',
                                   invert_y: bool = False) -> None:
    """
    Plot a full comparison row (Model 1, Model 2, Difference, Ratio) for a single velocity type.
    Matches the style of _plot_ratio_row from the existing ratio analysis plots.

    Args:
        axes: Array of 4 axes for the row
        data1, data2: Data for both models at the plane
        X, Y: Meshgrid coordinates
        vel_col: Velocity column ('vp' or 'vs')
        vel_label: Velocity label for display
        vel_cmap: Colormap for velocity
        model1_name, model2_name: Model names
        coord_value: Coordinate value for the plane
        label: Plane label
        xlabel, ylabel: Axis labels
        coord1, coord2: Column names for coordinates
        invert_y: Whether to invert Y axis (for depth)
    """
    if len(data1) == 0 or len(data2) == 0:
        return

    # Interpolate velocities
    Z1 = griddata((data1[coord1], data1[coord2]),
                 data1[vel_col], (X, Y), method='linear')
    Z2 = griddata((data2[coord1], data2[coord2]),
                 data2[vel_col], (X, Y), method='linear')

    # Find common velocity ranges
    v_min = min(np.nanmin(Z1), np.nanmin(Z2))
    v_max = max(np.nanmax(Z1), np.nanmax(Z2))
    if v_max - v_min < 1e-10:
        v_min, v_max = v_min - 0.1, v_max + 0.1
    norm_v = Normalize(vmin=v_min, vmax=v_max)

    # Column 1: Model 1
    im1 = axes[0].contourf(X, Y, Z1, levels=20, cmap=vel_cmap, vmin=v_min, vmax=v_max)
    axes[0].set_title(f'{model1_name} {vel_label}\n{label}')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_v, cmap=vel_cmap), ax=axes[0], label=f'{vel_label} (km/s)')
    if invert_y:
        axes[0].invert_yaxis()

    # Column 2: Model 2
    im2 = axes[1].contourf(X, Y, Z2, levels=20, cmap=vel_cmap, vmin=v_min, vmax=v_max)
    axes[1].set_title(f'{model2_name} {vel_label}\n{label}')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_v, cmap=vel_cmap), ax=axes[1], label=f'{vel_label} (km/s)')
    if invert_y:
        axes[1].invert_yaxis()

    # Column 3: Difference
    Z_diff = Z2 - Z1
    vmax_diff = np.nanmax(np.abs(Z_diff))
    if vmax_diff < 1e-10:
        vmax_diff = 0.1
    im3 = axes[2].contourf(X, Y, Z_diff, levels=20, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
    axes[2].set_title(f'{vel_label} Difference\n({model2_name} - {model1_name})')
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    plt.colorbar(im3, ax=axes[2], label=f'{vel_label} Diff (km/s)')
    if invert_y:
        axes[2].invert_yaxis()

    # Column 4: Ratio - centered at 1.0
    Z_ratio = np.divide(Z2, Z1, out=np.ones_like(Z2), where=Z1!=0)
    ratio_deviations = np.abs(Z_ratio - 1.0)
    max_deviation = np.nanmax(ratio_deviations)
    if max_deviation < 1e-10:
        max_deviation = 0.1
    vmin_ratio = 1.0 - max_deviation
    vmax_ratio = 1.0 + max_deviation

    # Calculate average ratio change for annotation
    avg_ratio_change = np.nanmean(np.abs(Z_ratio - 1.0))

    im4 = axes[3].contourf(X, Y, Z_ratio, levels=20, cmap='RdBu_r', vmin=vmin_ratio, vmax=vmax_ratio)
    axes[3].set_title(f'{vel_label} Ratio ({model2_name}/{model1_name})\nAvg change: {avg_ratio_change:.4f}')
    axes[3].set_xlabel(xlabel)
    axes[3].set_ylabel(ylabel)
    plt.colorbar(im4, ax=axes[3], label=f'{vel_label} Ratio')
    axes[3].contour(X, Y, Z_ratio, levels=[1.0], colors='black', linewidths=2, linestyles='--')
    if invert_y:
        axes[3].invert_yaxis()