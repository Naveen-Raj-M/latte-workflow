"""
Visualization functions for DWS (Derivative Weight Sum) analysis.

This module provides functions to create horizontal and vertical slice plots
of DWS values with support for both linear and logarithmic color scales.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

from .dws_data import extract_horizontal_slice, extract_vertical_slice

logger = logging.getLogger(__name__)


def plot_horizontal_slice(
    df: pd.DataFrame,
    depth_km: float,
    output_file: str,
    config: Dict[str, Any],
    log_scale: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot horizontal slice of DWS at specified depth.

    Args:
        df: DWS DataFrame
        depth_km: Depth in km
        output_file: Output file path
        config: Plot configuration dictionary
        log_scale: Use logarithmic color scale
        vmin: Minimum value for colorbar (None for auto)
        vmax: Maximum value for colorbar (None for auto)
        title: Custom title (None for auto)
    """
    logger.info(f"Creating horizontal slice plot at depth {depth_km} km")

    # Extract slice
    df_slice = extract_horizontal_slice(df, depth_km, tolerance_km=config.get('depth_tolerance', 0.01))

    if len(df_slice) == 0:
        logger.warning(f"No data for horizontal slice at {depth_km} km, skipping plot")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(config.get('figure_width', 12), config.get('figure_height', 10)))

    # Prepare data for pcolormesh
    x = df_slice['X_grid_km'].values
    y = df_slice['Y_grid_km'].values
    dws = df_slice['dws'].values

    # Unit conversion: check if we should use m² instead of km²
    use_m2_units = config.get('use_m2_units', False)
    unit_label = 'm²' if use_m2_units else 'km²'
    unit_conversion = 1e6 if use_m2_units else 1.0  # 1 km² = 1,000,000 m²

    # Create grid
    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))

    # Create 2D grid for pcolormesh
    dws_grid = np.full((len(y_unique), len(x_unique)), np.nan)

    for _, row in df_slice.iterrows():
        i_x = np.searchsorted(x_unique, row['X_grid_km'])
        i_y = np.searchsorted(y_unique, row['Y_grid_km'])
        if i_x < len(x_unique) and i_y < len(y_unique):
            dws_grid[i_y, i_x] = row['dws'] * unit_conversion

    # Handle log scale
    if log_scale:
        # Replace zeros and negative values with small positive value for log scale
        dws_grid_plot = dws_grid.copy()
        min_positive = np.nanmin(dws_grid[dws_grid > 0]) if np.any(dws_grid > 0) else 1e-10
        dws_grid_plot[dws_grid_plot <= 0] = min_positive / 10.0

        # Set vmin/vmax for log scale
        if vmin is None:
            vmin = min_positive / 10.0
        if vmax is None:
            vmax = np.nanmax(dws_grid_plot)

        norm = LogNorm(vmin=vmin, vmax=vmax)
        cmap = config.get('colormap_log', 'viridis')
    else:
        dws_grid_plot = dws_grid

        # Set vmin/vmax for linear scale
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = np.nanmax(dws_grid_plot)

        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = config.get('colormap', 'viridis')

    # Plot
    im = ax.pcolormesh(
        x_unique, y_unique, dws_grid_plot,
        cmap=cmap,
        norm=norm,
        shading='auto'
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label=f'DWS ({unit_label})')
    cbar.ax.tick_params(labelsize=config.get('tick_fontsize', 10))

    # Labels and title
    ax.set_xlabel('X (km)', fontsize=config.get('label_fontsize', 12))
    ax.set_ylabel('Y (km)', fontsize=config.get('label_fontsize', 12))

    if title is None:
        scale_str = "(Log Scale)" if log_scale else "(Linear Scale)"
        title = f'DWS Horizontal Slice at Depth = {depth_km:.2f} km {scale_str}'
    ax.set_title(title, fontsize=config.get('title_fontsize', 14))

    ax.set_aspect('equal' if config.get('equal_aspect', True) else 'auto')
    ax.grid(config.get('grid', True), alpha=0.3)

    # Save
    plt.tight_layout()
    plt.savefig(output_file, dpi=config.get('dpi', 300), bbox_inches='tight')
    plt.close()

    logger.info(f"Saved horizontal slice plot: {output_file}")


def plot_vertical_slice(
    df: pd.DataFrame,
    slice_type: str,
    position: float,
    output_file: str,
    config: Dict[str, Any],
    log_scale: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot vertical slice of DWS.

    Args:
        df: DWS DataFrame
        slice_type: 'xz' (constant Y) or 'yz' (constant X)
        position: Position value in km
        output_file: Output file path
        config: Plot configuration dictionary
        log_scale: Use logarithmic color scale
        vmin: Minimum value for colorbar (None for auto)
        vmax: Maximum value for colorbar (None for auto)
        title: Custom title (None for auto)
    """
    logger.info(f"Creating {slice_type.upper()} vertical slice plot at position {position} km")

    # Extract slice
    df_slice = extract_vertical_slice(df, slice_type, position, tolerance_km=config.get('position_tolerance', 0.01))

    if len(df_slice) == 0:
        logger.warning(f"No data for {slice_type} slice at {position} km, skipping plot")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(config.get('figure_width', 12), config.get('figure_height', 8)))

    # Prepare data
    if slice_type == 'xz':
        x = df_slice['X_grid_km'].values
        xlabel = 'X (km)'
    elif slice_type == 'yz':
        x = df_slice['Y_grid_km'].values
        xlabel = 'Y (km)'
    else:
        raise ValueError(f"Invalid slice_type: {slice_type}")

    z = df_slice['Z_grid_km'].values
    dws = df_slice['dws'].values

    # Unit conversion: check if we should use m² instead of km²
    use_m2_units = config.get('use_m2_units', False)
    unit_label = 'm²' if use_m2_units else 'km²'
    unit_conversion = 1e6 if use_m2_units else 1.0  # 1 km² = 1,000,000 m²

    # Create grid
    x_unique = np.sort(np.unique(x))
    z_unique = np.sort(np.unique(z))

    # Create 2D grid for pcolormesh
    dws_grid = np.full((len(z_unique), len(x_unique)), np.nan)

    for _, row in df_slice.iterrows():
        if slice_type == 'xz':
            x_val = row['X_grid_km']
        else:
            x_val = row['Y_grid_km']

        i_x = np.searchsorted(x_unique, x_val)
        i_z = np.searchsorted(z_unique, row['Z_grid_km'])

        if i_x < len(x_unique) and i_z < len(z_unique):
            dws_grid[i_z, i_x] = row['dws'] * unit_conversion

    # Handle log scale
    if log_scale:
        # Replace zeros and negative values with small positive value for log scale
        dws_grid_plot = dws_grid.copy()
        min_positive = np.nanmin(dws_grid[dws_grid > 0]) if np.any(dws_grid > 0) else 1e-10
        dws_grid_plot[dws_grid_plot <= 0] = min_positive / 10.0

        # Set vmin/vmax for log scale
        if vmin is None:
            vmin = min_positive / 10.0
        if vmax is None:
            vmax = np.nanmax(dws_grid_plot)

        norm = LogNorm(vmin=vmin, vmax=vmax)
        cmap = config.get('colormap_log', 'viridis')
    else:
        dws_grid_plot = dws_grid

        # Set vmin/vmax for linear scale
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = np.nanmax(dws_grid_plot)

        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = config.get('colormap', 'viridis')

    # Plot
    im = ax.pcolormesh(
        x_unique, z_unique, dws_grid_plot,
        cmap=cmap,
        norm=norm,
        shading='auto'
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label=f'DWS ({unit_label})')
    cbar.ax.tick_params(labelsize=config.get('tick_fontsize', 10))

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=config.get('label_fontsize', 12))
    ax.set_ylabel('Depth (km)', fontsize=config.get('label_fontsize', 12))

    if title is None:
        scale_str = "(Log Scale)" if log_scale else "(Linear Scale)"
        if slice_type == 'xz':
            title = f'DWS Vertical Slice (X-Z) at Y = {position:.2f} km {scale_str}'
        else:
            title = f'DWS Vertical Slice (Y-Z) at X = {position:.2f} km {scale_str}'
    ax.set_title(title, fontsize=config.get('title_fontsize', 14))

    ax.set_aspect('equal' if config.get('equal_aspect', False) else 'auto')
    ax.grid(config.get('grid', True), alpha=0.3)

    # Invert z-axis to have depth increase downwards
    ax.invert_yaxis()

    # Save
    plt.tight_layout()
    plt.savefig(output_file, dpi=config.get('dpi', 300), bbox_inches='tight')
    plt.close()

    logger.info(f"Saved {slice_type.upper()} slice plot: {output_file}")


def plot_dws_histogram(
    df: pd.DataFrame,
    output_file: str,
    config: Dict[str, Any],
    log_scale_x: bool = False,
    bins: int = 50
) -> None:
    """
    Plot histogram of DWS values.

    Args:
        df: DWS DataFrame
        output_file: Output file path
        config: Plot configuration dictionary
        log_scale_x: Use logarithmic scale for x-axis
        bins: Number of bins
    """
    logger.info(f"Creating DWS histogram")

    # Unit conversion: check if we should use m² instead of km²
    use_m2_units = config.get('use_m2_units', False)
    unit_label = 'm²' if use_m2_units else 'km²'
    unit_conversion = 1e6 if use_m2_units else 1.0  # 1 km² = 1,000,000 m²

    # Filter out zeros for log scale and apply unit conversion
    dws_values = df['dws'].values * unit_conversion
    if log_scale_x:
        dws_values = dws_values[dws_values > 0]

    fig, ax = plt.subplots(figsize=(config.get('figure_width', 10), config.get('figure_height', 6)))

    # Plot histogram
    if log_scale_x:
        ax.hist(dws_values, bins=np.logspace(np.log10(dws_values.min()), np.log10(dws_values.max()), bins),
                edgecolor='black', alpha=0.7)
        ax.set_xscale('log')
        scale_str = "(Log Scale)"
    else:
        ax.hist(dws_values, bins=bins, edgecolor='black', alpha=0.7)
        scale_str = "(Linear Scale)"

    ax.set_xlabel(f'DWS ({unit_label}) {scale_str}', fontsize=config.get('label_fontsize', 12))
    ax.set_ylabel('Frequency', fontsize=config.get('label_fontsize', 12))
    ax.set_title('DWS Distribution', fontsize=config.get('title_fontsize', 14))
    ax.grid(config.get('grid', True), alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=config.get('dpi', 300), bbox_inches='tight')
    plt.close()

    logger.info(f"Saved DWS histogram: {output_file}")


def plot_coverage_map(
    df: pd.DataFrame,
    depth_km: float,
    output_file: str,
    config: Dict[str, Any],
    threshold: float = 0.0,
    title: Optional[str] = None
) -> None:
    """
    Plot binary coverage map (covered vs uncovered cells).

    Args:
        df: DWS DataFrame
        depth_km: Depth in km
        output_file: Output file path
        config: Plot configuration dictionary
        threshold: DWS threshold for considering a cell as "covered"
        title: Custom title (None for auto)
    """
    logger.info(f"Creating coverage map at depth {depth_km} km (threshold: {threshold})")

    # Extract slice
    df_slice = extract_horizontal_slice(df, depth_km, tolerance_km=config.get('depth_tolerance', 0.01))

    if len(df_slice) == 0:
        logger.warning(f"No data for coverage map at {depth_km} km, skipping plot")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(config.get('figure_width', 12), config.get('figure_height', 10)))

    # Prepare data
    x = df_slice['X_grid_km'].values
    y = df_slice['Y_grid_km'].values
    covered = (df_slice['dws'].values > threshold).astype(int)

    # Create grid
    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))

    # Create 2D grid
    coverage_grid = np.full((len(y_unique), len(x_unique)), np.nan)

    for _, row in df_slice.iterrows():
        i_x = np.searchsorted(x_unique, row['X_grid_km'])
        i_y = np.searchsorted(y_unique, row['Y_grid_km'])
        if i_x < len(x_unique) and i_y < len(y_unique):
            coverage_grid[i_y, i_x] = 1 if row['dws'] > threshold else 0

    # Plot
    im = ax.pcolormesh(
        x_unique, y_unique, coverage_grid,
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        shading='auto'
    )

    # Colorbar with discrete labels
    cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Not Covered', 'Covered'])

    # Labels and title
    ax.set_xlabel('X (km)', fontsize=config.get('label_fontsize', 12))
    ax.set_ylabel('Y (km)', fontsize=config.get('label_fontsize', 12))

    if title is None:
        title = f'DWS Coverage Map at Depth = {depth_km:.2f} km (Threshold: {threshold:.2e} km²)'
    ax.set_title(title, fontsize=config.get('title_fontsize', 14))

    ax.set_aspect('equal' if config.get('equal_aspect', True) else 'auto')
    ax.grid(config.get('grid', True), alpha=0.3)

    # Compute and display coverage percentage
    coverage_pct = 100 * np.nansum(coverage_grid) / np.sum(~np.isnan(coverage_grid))
    ax.text(
        0.02, 0.98, f'Coverage: {coverage_pct:.1f}%',
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Save
    plt.tight_layout()
    plt.savefig(output_file, dpi=config.get('dpi', 300), bbox_inches='tight')
    plt.close()

    logger.info(f"Saved coverage map: {output_file} (coverage: {coverage_pct:.1f}%)")
