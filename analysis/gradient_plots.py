"""
Plotting functions for gradient and search direction analysis.

This module creates visualizations for gradients and search directions including
histograms, spatial distributions, depth profiles, and alignment plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List
import logging

from utils.plot_utils import save_figure, create_multi_panel_figure, get_color_scheme
from utils.traveltime_io import km_per_unit
from .basin_boundaries import load_basin_boundaries, plot_basin_boundaries

logger = logging.getLogger(__name__)


def create_velocity_gradient_plots(df: pd.DataFrame, 
                                  available_fields: Dict[str, List[str]],
                                  stats: Dict[str, Any],
                                  config: Dict[str, Any]) -> None:
    """
    Create comprehensive plots for velocity gradients and search directions.
    
    Args:
        df: Velocity dataframe with gradient/search fields
        available_fields: Dictionary of available fields
        stats: Statistics dictionary
        config: Configuration dictionary
    """
    output_dir = Path(config['gradient_analysis']['velocity_output_dir'])
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    colors = get_color_scheme(config['analysis'])
    output_format = config['analysis'].get('output_format', 'png')
    
    # Gradient histograms
    if available_fields['gradients']:
        logger.info("Creating gradient distribution plots")
        create_gradient_histograms(df, available_fields['gradients'], colors, 
                                  plots_dir, output_format, config)
    
    # Search direction histograms
    if available_fields['search']:
        logger.info("Creating search direction distribution plots")
        create_search_histograms(df, available_fields['search'], colors,
                                plots_dir, output_format, config)
    
    # Gradient vs search alignment plots
    if available_fields['gradients'] and available_fields['search']:
        logger.info("Creating gradient-search alignment plots")
        create_alignment_plots(df, available_fields, colors,
                             plots_dir, output_format, config)
    
    # Spatial distribution plots
    logger.info("Creating spatial distribution plots")
    create_velocity_spatial_gradient_plots(df, available_fields, colors,
                                          plots_dir, output_format, config)
    
    # Depth profile plots
    if 'z' in df.columns:
        logger.info("Creating depth profile plots")
        create_velocity_depth_profile_plots(df, available_fields, stats, colors,
                                           plots_dir, output_format, config)


def create_source_gradient_plots(df: pd.DataFrame,
                                available_fields: Dict[str, List[str]],
                                stats: Dict[str, Any],
                                config: Dict[str, Any]) -> None:
    """
    Create comprehensive plots for source parameter gradients.
    
    Args:
        df: Source dataframe with gradient/search fields
        available_fields: Dictionary of available fields
        stats: Statistics dictionary
        config: Configuration dictionary
    """
    output_dir = Path(config['gradient_analysis']['source_output_dir'])
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    colors = get_color_scheme(config['analysis'])
    output_format = config['analysis'].get('output_format', 'png')
    
    # Gradient histograms
    if available_fields['gradients']:
        logger.info("Creating source gradient distribution plots")
        create_gradient_histograms(df, available_fields['gradients'], colors,
                                  plots_dir, output_format, config, prefix='source')
    
    # Search direction histograms
    if available_fields['search']:
        logger.info("Creating source search direction plots")
        create_search_histograms(df, available_fields['search'], colors,
                                plots_dir, output_format, config, prefix='source')
    
    # Alignment plots
    if available_fields['gradients'] and available_fields['search']:
        logger.info("Creating source gradient-search alignment plots")
        create_alignment_plots(df, available_fields, colors,
                             plots_dir, output_format, config, prefix='source')
    
    # Geographic distribution for spatial parameters
    if any(f'grad_{p}' in available_fields['gradients'] for p in ['sx', 'sy', 'sz']):
        logger.info("Creating source spatial gradient plots")
        create_source_spatial_gradient_plots(df, available_fields, colors,
                                            plots_dir, output_format, config)


def create_gradient_histograms(df: pd.DataFrame, gradient_fields: List[str],
                              colors: Dict[str, str], plots_dir: Path,
                              output_format: str, config: Dict[str, Any],
                              prefix: str = 'velocity') -> None:
    """Create histogram plots for gradients."""
    n_fields = len(gradient_fields)
    if n_fields == 0:
        return
    
    # Determine grid layout
    ncols = min(2, n_fields)
    nrows = (n_fields + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 6*nrows))
    if n_fields == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    bins = config['gradient_analysis'].get('histogram_bins', 50)
    
    for idx, field in enumerate(gradient_fields):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        param = field.replace('grad_', '')
        data = df[field].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, f'No data for {param}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Gradient: {param}')
            continue
        
        # Create histogram
        ax.hist(data, bins=bins, alpha=0.7, color='#2E86AB',
               edgecolor='black', linewidth=0.5)
        
        # Add mean and median lines
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.3e}')
        ax.axvline(median_val, color='orange', linestyle='-', linewidth=2,
                  label=f'Median: {median_val:.3e}')
        ax.axvline(0, color='black', linestyle=':', linewidth=1.5)
        
        # Labels and title
        unit = get_gradient_unit(param, prefix)
        ax.set_xlabel(f'Gradient: {param} {unit}')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{param.upper()} Gradient Distribution')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = (f'N: {len(data)}\n'
                     f'Std: {data.std():.3e}\n'
                     f'RMS: {np.sqrt(np.mean(data**2)):.3e}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for idx in range(n_fields, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'{prefix.title()} Gradient Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, plots_dir / f'{prefix}_gradient_histograms', output_format)


def create_search_histograms(df: pd.DataFrame, search_fields: List[str],
                            colors: Dict[str, str], plots_dir: Path,
                            output_format: str, config: Dict[str, Any],
                            prefix: str = 'velocity') -> None:
    """Create histogram plots for search directions."""
    n_fields = len(search_fields)
    if n_fields == 0:
        return
    
    # Determine grid layout
    ncols = min(2, n_fields)
    nrows = (n_fields + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 6*nrows))
    if n_fields == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    bins = config['gradient_analysis'].get('histogram_bins', 50)
    
    for idx, field in enumerate(search_fields):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        param = field.replace('srch_', '')
        data = df[field].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, f'No data for {param}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Search Direction: {param}')
            continue
        
        # Create histogram
        ax.hist(data, bins=bins, alpha=0.7, color='#6A994E',
               edgecolor='black', linewidth=0.5)
        
        # Add mean and median lines
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.3e}')
        ax.axvline(median_val, color='orange', linestyle='-', linewidth=2,
                  label=f'Median: {median_val:.3e}')
        ax.axvline(0, color='black', linestyle=':', linewidth=1.5)
        
        # Labels and title
        unit = get_search_unit(param, prefix)
        ax.set_xlabel(f'Search Direction: {param} {unit}')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{param.upper()} Search Direction Distribution')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = (f'N: {len(data)}\n'
                     f'Std: {data.std():.3e}\n'
                     f'RMS: {np.sqrt(np.mean(data**2)):.3e}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for idx in range(n_fields, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'{prefix.title()} Search Direction Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, plots_dir / f'{prefix}_search_histograms', output_format)


def create_alignment_plots(df: pd.DataFrame, available_fields: Dict[str, List[str]],
                          colors: Dict[str, str], plots_dir: Path,
                          output_format: str, config: Dict[str, Any],
                          prefix: str = 'velocity') -> None:
    """Create gradient vs search direction alignment plots."""
    # Find matching gradient-search pairs
    pairs = []
    for grad_field in available_fields['gradients']:
        param = grad_field.replace('grad_', '')
        search_field = f'srch_{param}'
        if search_field in available_fields['search']:
            pairs.append((param, grad_field, search_field))
    
    if not pairs:
        return
    
    n_pairs = len(pairs)
    ncols = min(2, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 6*nrows))
    if n_pairs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for idx, (param, grad_field, search_field) in enumerate(pairs):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Get valid data
        valid_mask = df[grad_field].notna() & df[search_field].notna()
        grad_data = df.loc[valid_mask, grad_field]
        search_data = df.loc[valid_mask, search_field]
        
        if len(grad_data) == 0:
            ax.text(0.5, 0.5, f'No data for {param}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Alignment: {param}')
            continue
        
        # Create scatter plot
        ax.scatter(grad_data, search_data, alpha=0.6, s=20, color='#A23B72')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        
        # Add quadrant lines
        ax.plot([grad_data.min(), grad_data.max()],
               [-grad_data.min(), -grad_data.max()],
               'r--', alpha=0.5, linewidth=1.5, label='Perfect alignment')
        
        # Labels
        grad_unit = get_gradient_unit(param, prefix)
        search_unit = get_search_unit(param, prefix)
        ax.set_xlabel(f'Gradient {grad_unit}')
        ax.set_ylabel(f'Search Direction {search_unit}')
        ax.set_title(f'{param.upper()}: Gradient vs Search Direction')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
        
        # Compute and display alignment statistics
        same_sign = ((grad_data > 0) & (search_data > 0)) | ((grad_data < 0) & (search_data < 0))
        percent_aligned = same_sign.sum() / len(same_sign) * 100
        
        stats_text = (f'Same sign: {percent_aligned:.1f}%\n'
                     f'N: {len(grad_data)}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'{prefix.title()} Gradient-Search Alignment', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, plots_dir / f'{prefix}_gradient_search_alignment', output_format)


def create_velocity_spatial_gradient_plots(df: pd.DataFrame,
                                          available_fields: Dict[str, List[str]],
                                          colors: Dict[str, str], plots_dir: Path,
                                          output_format: str, config: Dict[str, Any]) -> None:
    """Create spatial distribution plots for velocity gradients."""
    if 'x' not in df.columns or 'y' not in df.columns:
        return

    basin_config = config.get('basin_boundaries', {})
    basin_data = load_basin_boundaries(basin_config) if config['gradient_analysis'].get('plot_basin_boundaries', False) else {}

    # Create XY projection plots for each gradient
    for grad_field in available_fields['gradients']:
        param = grad_field.replace('grad_', '')
        
        valid_mask = df[grad_field].notna()
        if valid_mask.sum() == 0:
            continue
        
        plot_data = df.loc[valid_mask].copy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(plot_data['x'], plot_data['y'],
                           c=plot_data[grad_field], cmap='RdBu_r',
                           s=5, alpha=0.6, edgecolors='none')
        
        cbar = plt.colorbar(scatter, ax=ax)
        grad_unit = get_gradient_unit(param, 'velocity')
        cbar.set_label(f'Gradient {grad_unit}', rotation=270, labelpad=20)
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title(f'{param.upper()} Gradient Spatial Distribution (XY Projection)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

        if basin_data:
            plot_basin_boundaries(ax, basin_data, basin_config)

        plt.tight_layout()
        save_figure(fig, plots_dir / f'velocity_gradient_{param}_xy_projection', output_format)


def create_source_spatial_gradient_plots(df: pd.DataFrame,
                                        available_fields: Dict[str, List[str]],
                                        colors: Dict[str, str], plots_dir: Path,
                                        output_format: str, config: Dict[str, Any]) -> None:
    """Create spatial/geographic plots for source parameter gradients."""
    basin_config = config.get('basin_boundaries', {})
    basin_data = load_basin_boundaries(basin_config) if config['gradient_analysis'].get('plot_basin_boundaries', False) else {}

    # Plot horizontal gradient magnitude if available
    if 'grad_sx' in available_fields['gradients'] and 'grad_sy' in available_fields['gradients']:
        valid_mask = df['grad_sx'].notna() & df['grad_sy'].notna()
        if valid_mask.sum() > 0:
            plot_data = df.loc[valid_mask].copy()
            plot_data['horiz_grad_mag'] = np.sqrt(plot_data['grad_sx']**2 + plot_data['grad_sy']**2)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            scatter = ax.scatter(plot_data['corrected_event_x'] * km_per_unit(plot_data, 'corrected_event_x'),
                               plot_data['corrected_event_y'] * km_per_unit(plot_data, 'corrected_event_y'),
                               c=plot_data['horiz_grad_mag'], cmap='viridis',
                               s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Horizontal Gradient Magnitude (m)', rotation=270, labelpad=20)
            
            ax.set_xlabel('X (km)')
            ax.set_ylabel('Y (km)')
            ax.set_title('Source Horizontal Gradient Magnitude')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)

            if basin_data:
                plot_basin_boundaries(ax, basin_data, basin_config)

            plt.tight_layout()
            save_figure(fig, plots_dir / 'source_horizontal_gradient_magnitude', output_format)


def create_velocity_depth_profile_plots(df: pd.DataFrame,
                                       available_fields: Dict[str, List[str]],
                                       stats: Dict[str, Any],
                                       colors: Dict[str, str], plots_dir: Path,
                                       output_format: str, config: Dict[str, Any]) -> None:
    """Create depth profile plots for velocity gradients."""
    if 'depth_dependent' not in stats or not stats['depth_dependent']:
        return
    
    n_params = len(stats['depth_dependent'])
    if n_params == 0:
        return
    
    ncols = min(2, n_params)
    nrows = (n_params + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 6*nrows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for idx, (param, bin_stats) in enumerate(stats['depth_dependent'].items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Extract data from nested dict structure
        depths = []
        means = []
        stds = []
        
        for bin_interval, data in bin_stats.items():
            # Get bin center
            if hasattr(bin_interval, 'mid'):
                depths.append(bin_interval.mid)
            else:
                continue
            means.append(data['mean'])
            stds.append(data['std'])
        
        if not depths:
            ax.text(0.5, 0.5, f'No depth data for {param}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Depth Profile: {param}')
            continue
        
        # Plot with error bars
        ax.errorbar(means, depths, xerr=stds, fmt='o-', 
                   capsize=5, linewidth=2, markersize=6, color='#2E86AB')
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
        
        grad_unit = get_gradient_unit(param, 'velocity')
        ax.set_xlabel(f'Mean Gradient {grad_unit}')
        ax.set_ylabel('Depth (km)')
        ax.set_title(f'{param.upper()} Gradient Depth Profile')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Velocity Gradient Depth Profiles', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, plots_dir / 'velocity_gradient_depth_profiles', output_format)


def get_gradient_unit(param: str, analysis_type: str) -> str:
    """Get appropriate unit string for gradient."""
    if analysis_type == 'velocity':
        if param in ['vp', 'vs']:
            return '[(km/s) per unit]'
    else:  # source
        if param in ['sx', 'sy', 'sz']:
            return '[m per unit]'
        elif param == 'st0':
            return '[s per unit]'
    return '[per unit]'


def get_search_unit(param: str, analysis_type: str) -> str:
    """Get appropriate unit string for search direction."""
    if analysis_type == 'velocity':
        if param in ['vp', 'vs']:
            return '[km/s]'
    else:  # source
        if param in ['sx', 'sy', 'sz']:
            return '[m]'
        elif param == 'st0':
            return '[s]'
    return '[]'