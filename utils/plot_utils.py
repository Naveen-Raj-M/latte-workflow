"""
Plot utilities for LATTE analysis module.

This module provides common plotting functions and styling utilities
used across different analysis modules.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import seaborn as sns


def setup_plot_style(config: Dict[str, Any]) -> None:
    """
    Setup global matplotlib styling based on configuration.
    
    Args:
        config: Plot style configuration dictionary
    """
    plot_style = config.get('plot_style', {})
    
    # Set matplotlib parameters
    plt.rcParams.update({
        'font.family': plot_style.get('font_family', 'sans-serif'),
        'font.size': plot_style.get('font_size', 14),
        'axes.titlesize': plot_style.get('title_size', 16),
        'axes.labelsize': plot_style.get('label_size', 14),
        'xtick.labelsize': plot_style.get('tick_size', 12),
        'ytick.labelsize': plot_style.get('tick_size', 12),
        'legend.fontsize': plot_style.get('legend_size', 12),
        'figure.titlesize': plot_style.get('figure_title_size', 18),
        'lines.linewidth': plot_style.get('line_width', 2),
        'lines.markersize': plot_style.get('marker_size', 4),
        'grid.alpha': plot_style.get('grid_alpha', 0.3),
        'figure.dpi': config.get('dpi', 300),
    })


def get_color_scheme(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Get color scheme from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of color mappings
    """
    return config.get('plot_style', {}).get('colors', {
        'p_wave_dark': '#1f4e79',
        'p_wave_light': '#5b9bd5',
        's_wave_dark': '#2e7d32',
        's_wave_light': '#66bb6a',
        'event_color': '#ff6b35',
        'station_color': '#4a4a4a',
        'neutral_color': '#d32f2f',
        'grid_color': '#cccccc',
        'positive_residual': '#ff4444',
        'negative_residual': '#4444ff',
        'zero_line': '#000000',
    })


def save_figure(fig: plt.Figure, output_path: Path, 
                output_format: str = 'png', dpi: int = 300,
                bbox_inches: str = 'tight') -> None:
    """
    Save figure with consistent formatting.
    
    Args:
        fig: Matplotlib figure object
        output_path: Path to save the figure
        output_format: Output format (png, pdf, svg)
        dpi: Resolution for raster formats
        bbox_inches: Bounding box for saved figure
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add extension if not present
    if not output_path.suffix:
        output_path = output_path.with_suffix(f'.{output_format}')
    
    fig.savefig(output_path, format=output_format, dpi=dpi, 
                bbox_inches=bbox_inches)
    plt.close(fig)


def create_histogram_with_stats(data: np.ndarray, bins: int = 50, 
                               color: str = 'blue', alpha: float = 0.7,
                               title: str = '', xlabel: str = '',
                               ylabel: str = 'Frequency',
                               show_mean: bool = True,
                               show_median: bool = False) -> plt.Figure:
    """
    Create histogram with statistical overlays.
    
    Args:
        data: Data array to plot
        bins: Number of histogram bins
        color: Histogram color
        alpha: Transparency
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        show_mean: Whether to show mean line
        show_median: Whether to show median line
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    n, bins_edges, patches = ax.hist(data, bins=bins, alpha=alpha, 
                                   color=color, edgecolor='black', linewidth=0.5)
    
    # Add statistical lines
    if show_mean:
        mean_val = np.mean(data)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.3f}')
    
    if show_median:
        median_val = np.median(data)
        ax.axvline(median_val, color='orange', linestyle='-', linewidth=2,
                  label=f'Median: {median_val:.3f}')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    if show_mean or show_median:
        ax.legend(loc='upper right')
    
    return fig


def create_scatter_plot(x: np.ndarray, y: np.ndarray, 
                       c: Optional[np.ndarray] = None,
                       title: str = '', xlabel: str = '', ylabel: str = '',
                       colormap: str = 'viridis', alpha: float = 0.6,
                       s: float = 20, add_diagonal: bool = False,
                       colorbar_label: str = '') -> plt.Figure:
    """
    Create scatter plot with optional color mapping.
    
    Args:
        x: X data
        y: Y data
        c: Color values (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        colormap: Colormap name
        alpha: Point transparency
        s: Point size
        add_diagonal: Whether to add 1:1 diagonal line
        colorbar_label: Label for colorbar
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if c is not None:
        scatter = ax.scatter(x, y, c=c, cmap=colormap, alpha=alpha, s=s)
        if colorbar_label:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(colorbar_label)
    else:
        ax.scatter(x, y, alpha=alpha, s=s)
    
    if add_diagonal:
        min_val = min(np.min(x), np.min(y))
        max_val = max(np.max(x), np.max(y))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--',
               linewidth=2, label='1:1 line')
        ax.legend(loc='upper right')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    return fig


def create_spatial_plot(x: np.ndarray, y: np.ndarray, values: np.ndarray,
                       title: str = '', xlabel: str = 'X (km)', 
                       ylabel: str = 'Y (km)', colormap: str = 'RdBu_r',
                       interpolation_method: str = 'cubic',
                       grid_resolution: int = 100,
                       colorbar_label: str = '') -> plt.Figure:
    """
    Create spatial interpolated plot.
    
    Args:
        x: X coordinates
        y: Y coordinates  
        values: Values to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        colormap: Colormap name
        interpolation_method: Interpolation method
        grid_resolution: Grid resolution for interpolation
        colorbar_label: Colorbar label
        
    Returns:
        Matplotlib figure object
    """
    from scipy.interpolate import griddata
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create regular grid
    xi = np.linspace(x.min(), x.max(), grid_resolution)
    yi = np.linspace(y.min(), y.max(), grid_resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate values
    Zi = griddata((x, y), values, (Xi, Yi), method=interpolation_method)
    
    # Create plot
    im = ax.imshow(Zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                  cmap=colormap, origin='lower', aspect='auto')
    
    if colorbar_label:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    return fig


def create_residual_vs_distance_plot(distances: np.ndarray, residuals: np.ndarray,
                                   phase: str = '', colors: Dict[str, str] = None,
                                   add_trend_line: bool = True,
                                   log_scale: bool = False) -> plt.Figure:
    """
    Create residual vs distance plot with trend analysis.
    
    Args:
        distances: Distance values
        residuals: Residual values
        phase: Phase name (P or S)
        colors: Color scheme dictionary
        add_trend_line: Whether to add polynomial trend line
        log_scale: Whether to use log scale for distance
        
    Returns:
        Matplotlib figure object
    """
    if colors is None:
        colors = get_color_scheme({})
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Choose color based on phase
    if phase.upper() == 'P':
        color = colors.get('p_wave_dark', 'blue')
    elif phase.upper() == 'S':
        color = colors.get('s_wave_dark', 'green')
    else:
        color = colors.get('neutral_color', 'gray')
    
    # Create scatter plot
    ax.scatter(distances, residuals, alpha=0.6, s=15, color=color)
    
    # Add zero line
    ax.axhline(0, color=colors.get('zero_line', 'black'), 
              linestyle='--', linewidth=2)
    
    # Add trend line
    if add_trend_line:
        z = np.polyfit(distances, residuals, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(distances.min(), distances.max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr = np.corrcoef(distances, residuals)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.4f}', 
               transform=ax.transAxes, bbox=dict(boxstyle='round', 
               facecolor='white', alpha=0.8))
    
    if log_scale:
        ax.set_xscale('log')
    
    phase_label = f"{phase}-wave " if phase else ""
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Residual (s)')
    ax.set_title(f'{phase_label}Residuals vs Distance')
    ax.grid(True, alpha=0.3)
    
    return fig


def create_multi_panel_figure(nrows: int, ncols: int, figsize: Tuple[int, int] = None,
                             title: str = '') -> Tuple[plt.Figure, np.ndarray]:
    """
    Create multi-panel figure with consistent styling.
    
    Args:
        nrows: Number of rows
        ncols: Number of columns
        figsize: Figure size tuple
        title: Overall figure title
        
    Returns:
        Tuple of (figure, axes array)
    """
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Ensure axes is always an array
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    
    if title:
        fig.suptitle(title, fontsize=18)
    
    return fig, axes


def add_statistics_text(ax: plt.Axes, data: np.ndarray, 
                       position: str = 'upper right',
                       include_count: bool = True,
                       include_mean: bool = True,
                       include_std: bool = True,
                       include_rms: bool = False) -> None:
    """
    Add statistics text box to plot.
    
    Args:
        ax: Matplotlib axes object
        data: Data array for statistics
        position: Text box position
        include_count: Include data count
        include_mean: Include mean value
        include_std: Include standard deviation
        include_rms: Include RMS value
    """
    stats_text = []
    
    if include_count:
        stats_text.append(f'N = {len(data)}')
    if include_mean:
        stats_text.append(f'Mean: {np.mean(data):.3f}')
    if include_std:
        stats_text.append(f'Std: {np.std(data):.3f}')
    if include_rms:
        stats_text.append(f'RMS: {np.sqrt(np.mean(data**2)):.3f}')
    
    text = '\n'.join(stats_text)
    
    # Convert position string to coordinates
    position_map = {
        'upper right': (0.95, 0.95),
        'upper left': (0.05, 0.95),
        'lower right': (0.95, 0.05),
        'lower left': (0.05, 0.05),
    }
    
    x, y = position_map.get(position, (0.95, 0.95))
    ha = 'right' if 'right' in position else 'left'
    va = 'top' if 'upper' in position else 'bottom'
    
    ax.text(x, y, text, transform=ax.transAxes, 
           verticalalignment=va, horizontalalignment=ha,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def subsample_data_for_plotting(data: pd.DataFrame, max_points: int = 50000,
                              random_state: int = 42) -> pd.DataFrame:
    """
    Subsample data for plotting to avoid overcrowding.
    
    Args:
        data: Input dataframe
        max_points: Maximum number of points to plot
        random_state: Random seed for reproducibility
        
    Returns:
        Subsampled dataframe
    """
    if len(data) <= max_points:
        return data
    
    return data.sample(n=max_points, random_state=random_state)


def format_scientific_notation(value: float, precision: int = 2) -> str:
    """
    Format value in scientific notation if needed.
    
    Args:
        value: Value to format
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    if abs(value) >= 1000 or (abs(value) < 0.01 and value != 0):
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"