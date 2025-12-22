#!/usr/bin/env python3
"""
Analyze DWS (Derivative Weighted Sum) from LATTE Traveltime Inversion

This script reads DWS binary output files from latte_traveltime and generates
visualizations and statistical reports to assess data coverage and sensitivity.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION DICTIONARY
# ============================================================================

CONFIG = {
    # ==== PATH SETTINGS ====
    'paths': {
        # Directory containing DWS output files
        'dws_directory': '/path/to/iteration_directory',
        # Output directory for plots and reports
        'output_directory': '/path/to/dws_analysis',
        # DWS file names (without directory path)
        'dws_vp_file': 'dws_vp.bin',
        'dws_vs_file': 'dws_vs.bin',
    },

    # ==== GRID DIMENSIONS ====
    'grid': {
        # Grid dimensions (must match the model grid)
        'nz': 100,  # number of points in z (depth)
        'nx': 200,  # number of points in x
        'ny': 150,  # number of points in y (set to 1 for 2D)
        'is_3d': True,  # Set to False for 2D data

        # Physical dimensions (for axis labels)
        'z_min': 0.0,   # km
        'z_max': 10.0,  # km
        'x_min': 0.0,   # km
        'x_max': 20.0,  # km
        'y_min': 0.0,   # km (ignored for 2D)
        'y_max': 15.0,  # km (ignored for 2D)
    },

    # ==== ANALYSIS SETTINGS ====
    'analysis': {
        # Parameters to analyze
        'analyze_vp': True,
        'analyze_vs': True,

        # Statistical percentiles to report
        'percentiles': [10, 25, 50, 75, 90, 95, 99],

        # Threshold for "well-constrained" regions (fraction of max DWS)
        'constraint_threshold': 0.1,  # 10% of max

        # Normalization method: 'global' or 'per_parameter'
        # 'global': normalize both vp and vs by max(vp_max, vs_max)
        # 'per_parameter': normalize vp and vs independently
        'normalization': 'per_parameter',
    },

    # ==== PLOTTING SETTINGS ====
    'plotting': {
        # Figure settings
        'dpi': 150,
        'figsize': (10, 8),  # for individual slice plots

        # Colormap settings
        'cmap': 'viridis',  # Options: 'viridis', 'plasma', 'inferno', 'hot', 'jet'
        'cmap_diverging': 'RdBu_r',  # for difference/ratio plots
        'vmin_percentile': 0,   # percentile for color scale minimum
        'vmax_percentile': 98,  # percentile for color scale maximum

        # Slice positions (as fractions of total dimension, 0-1)
        'xy_slices': [0.1, 0.3, 0.5, 0.7, 0.9],  # depth slices (z positions)
        'xz_slices': [0.3, 0.5, 0.7],             # y slices
        'yz_slices': [0.3, 0.5, 0.7],             # x slices

        # Contour settings
        'show_contours': True,
        'contour_levels': 10,
        'contour_alpha': 0.3,
        'contour_colors': 'black',
        'contour_linewidth': 0.5,

        # Colorbar settings
        'colorbar_label': 'DWS (Normalized)',
        'colorbar_orientation': 'vertical',  # 'vertical' or 'horizontal'
        'colorbar_shrink': 0.8,
        'colorbar_pad': 0.02,

        # Grid and labels
        'show_grid': True,
        'grid_alpha': 0.3,
        'grid_linestyle': '--',
        'xlabel_fontsize': 12,
        'ylabel_fontsize': 12,
        'title_fontsize': 14,
        'colorbar_fontsize': 11,

        # Aspect ratio
        'aspect': 'auto',  # 'auto' or 'equal'

        # Save format
        'save_format': 'png',  # Options: 'png', 'pdf', 'svg', 'jpg'
        'save_dpi': 300,
    },

    # ==== REPORT SETTINGS ====
    'report': {
        'generate_markdown': True,
        'generate_json': True,
        'report_filename': 'dws_analysis_report',
        'include_plots': True,
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def read_dws_binary(filepath, nz, nx, ny, is_3d=True):
    """
    Read DWS binary file (Fortran binary format).

    Parameters:
    -----------
    filepath : str or Path
        Path to the binary file
    nz, nx, ny : int
        Grid dimensions
    is_3d : bool
        If True, read as 3D array (nz, ny, nx)
        If False, read as 2D array (nz, nx)

    Returns:
    --------
    dws : np.ndarray
        DWS array
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"DWS file not found: {filepath}")

    # Read binary data (assuming float32 - standard Fortran real)
    data = np.fromfile(filepath, dtype=np.float32)

    if is_3d:
        expected_size = nz * ny * nx
        if data.size != expected_size:
            raise ValueError(
                f"Data size mismatch: expected {expected_size} "
                f"(nz={nz}, ny={ny}, nx={nx}), got {data.size}"
            )
        # Fortran order: reshape to (nz, ny, nx)
        dws = data.reshape((nz, ny, nx), order='F')
    else:
        expected_size = nz * nx
        if data.size != expected_size:
            raise ValueError(
                f"Data size mismatch: expected {expected_size} "
                f"(nz={nz}, nx={nx}), got {data.size}"
            )
        # Fortran order: reshape to (nz, nx)
        dws = data.reshape((nz, nx), order='F')

    return dws


def create_coordinate_arrays(config):
    """Create coordinate arrays for plotting."""
    grid = config['grid']

    z = np.linspace(grid['z_min'], grid['z_max'], grid['nz'])
    x = np.linspace(grid['x_min'], grid['x_max'], grid['nx'])

    if grid['is_3d']:
        y = np.linspace(grid['y_min'], grid['y_max'], grid['ny'])
    else:
        y = None

    return x, y, z


def compute_statistics(dws, config):
    """
    Compute statistical measures of DWS.

    Returns:
    --------
    stats : dict
        Dictionary containing statistics
    """
    stats = {}

    # Basic statistics
    stats['min'] = float(np.min(dws))
    stats['max'] = float(np.max(dws))
    stats['mean'] = float(np.mean(dws))
    stats['median'] = float(np.median(dws))
    stats['std'] = float(np.std(dws))

    # Percentiles
    percentiles = config['analysis']['percentiles']
    stats['percentiles'] = {}
    for p in percentiles:
        stats['percentiles'][f'p{p}'] = float(np.percentile(dws, p))

    # Coverage analysis
    threshold = config['analysis']['constraint_threshold'] * stats['max']
    well_constrained = np.sum(dws > threshold)
    total_points = dws.size
    stats['well_constrained_fraction'] = float(well_constrained / total_points)
    stats['constraint_threshold'] = float(threshold)

    # Zero or near-zero values
    near_zero = np.sum(dws < 1e-10)
    stats['near_zero_fraction'] = float(near_zero / total_points)

    return stats


def normalize_dws(dws_vp, dws_vs, method='per_parameter'):
    """
    Normalize DWS arrays.

    Parameters:
    -----------
    dws_vp, dws_vs : np.ndarray or None
        DWS arrays
    method : str
        'global' or 'per_parameter'

    Returns:
    --------
    normalized_vp, normalized_vs : np.ndarray or None
    """
    if method == 'global':
        max_val = 0.0
        if dws_vp is not None:
            max_val = max(max_val, np.max(dws_vp))
        if dws_vs is not None:
            max_val = max(max_val, np.max(dws_vs))

        if max_val > 0:
            norm_vp = dws_vp / max_val if dws_vp is not None else None
            norm_vs = dws_vs / max_val if dws_vs is not None else None
        else:
            norm_vp = dws_vp
            norm_vs = dws_vs

    else:  # per_parameter
        if dws_vp is not None:
            max_vp = np.max(dws_vp)
            norm_vp = dws_vp / max_vp if max_vp > 0 else dws_vp
        else:
            norm_vp = None

        if dws_vs is not None:
            max_vs = np.max(dws_vs)
            norm_vs = dws_vs / max_vs if max_vs > 0 else dws_vs
        else:
            norm_vs = None

    return norm_vp, norm_vs


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def setup_plot_style(config):
    """Set up matplotlib style."""
    plt.rcParams['font.size'] = config['plotting']['xlabel_fontsize']
    plt.rcParams['axes.labelsize'] = config['plotting']['xlabel_fontsize']
    plt.rcParams['axes.titlesize'] = config['plotting']['title_fontsize']
    plt.rcParams['xtick.labelsize'] = config['plotting']['xlabel_fontsize'] - 2
    plt.rcParams['ytick.labelsize'] = config['plotting']['ylabel_fontsize'] - 2
    plt.rcParams['figure.dpi'] = config['plotting']['dpi']


def get_color_limits(dws, config):
    """Calculate color limits based on percentiles."""
    vmin = np.percentile(dws, config['plotting']['vmin_percentile'])
    vmax = np.percentile(dws, config['plotting']['vmax_percentile'])
    return vmin, vmax


def plot_single_xy_slice(dws, x, y, z, iz, param_name, config, output_dir, vmin, vmax):
    """Plot a single XY (horizontal) slice."""
    plot_cfg = config['plotting']

    fig, ax = plt.subplots(figsize=plot_cfg['figsize'])

    # Extract slice
    slice_data = dws[iz, :, :]

    # Plot
    im = ax.imshow(
        slice_data,
        extent=[x[0], x[-1], y[0], y[-1]],
        origin='lower',
        aspect=plot_cfg['aspect'],
        cmap=plot_cfg['cmap'],
        vmin=vmin,
        vmax=vmax
    )

    # Contours
    if plot_cfg['show_contours']:
        X, Y = np.meshgrid(x, y)
        ax.contour(
            X, Y, slice_data,
            levels=plot_cfg['contour_levels'],
            colors=plot_cfg['contour_colors'],
            alpha=plot_cfg['contour_alpha'],
            linewidths=plot_cfg['contour_linewidth']
        )

    ax.set_xlabel('X (km)', fontsize=plot_cfg['xlabel_fontsize'])
    ax.set_ylabel('Y (km)', fontsize=plot_cfg['ylabel_fontsize'])
    ax.set_title(f'DWS {param_name.upper()} - XY Slice at Depth = {z[iz]:.2f} km',
                 fontsize=plot_cfg['title_fontsize'])

    if plot_cfg['show_grid']:
        ax.grid(True, alpha=plot_cfg['grid_alpha'], linestyle=plot_cfg['grid_linestyle'])

    # Colorbar
    cbar = plt.colorbar(
        im, ax=ax,
        orientation=plot_cfg['colorbar_orientation'],
        label=plot_cfg['colorbar_label'],
        shrink=plot_cfg['colorbar_shrink'],
        pad=plot_cfg['colorbar_pad']
    )
    cbar.ax.tick_params(labelsize=plot_cfg['colorbar_fontsize'])

    plt.tight_layout()

    # Save
    filename = f'dws_{param_name}_xy_z{iz:03d}_depth{z[iz]:.2f}km.{plot_cfg["save_format"]}'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=plot_cfg['save_dpi'], bbox_inches='tight')
    plt.close()

    return filepath


def plot_single_xz_slice(dws, x, y, z, iy, param_name, config, output_dir, vmin, vmax, is_2d=False):
    """Plot a single XZ (vertical) slice."""
    plot_cfg = config['plotting']

    fig, ax = plt.subplots(figsize=plot_cfg['figsize'])

    # Extract slice
    if is_2d:
        slice_data = dws
        title_suffix = 'XZ Section'
    else:
        slice_data = dws[:, iy, :]
        title_suffix = f'XZ Slice at Y = {y[iy]:.2f} km'

    # Plot
    im = ax.imshow(
        slice_data,
        extent=[x[0], x[-1], z[-1], z[0]],
        aspect=plot_cfg['aspect'],
        cmap=plot_cfg['cmap'],
        vmin=vmin,
        vmax=vmax
    )

    # Contours
    if plot_cfg['show_contours']:
        X, Z = np.meshgrid(x, z)
        ax.contour(
            X, Z, slice_data,
            levels=plot_cfg['contour_levels'],
            colors=plot_cfg['contour_colors'],
            alpha=plot_cfg['contour_alpha'],
            linewidths=plot_cfg['contour_linewidth']
        )

    ax.set_xlabel('X (km)', fontsize=plot_cfg['xlabel_fontsize'])
    ax.set_ylabel('Depth (km)', fontsize=plot_cfg['ylabel_fontsize'])
    ax.set_title(f'DWS {param_name.upper()} - {title_suffix}',
                 fontsize=plot_cfg['title_fontsize'])
    ax.invert_yaxis()

    if plot_cfg['show_grid']:
        ax.grid(True, alpha=plot_cfg['grid_alpha'], linestyle=plot_cfg['grid_linestyle'])

    # Colorbar
    cbar = plt.colorbar(
        im, ax=ax,
        orientation=plot_cfg['colorbar_orientation'],
        label=plot_cfg['colorbar_label'],
        shrink=plot_cfg['colorbar_shrink'],
        pad=plot_cfg['colorbar_pad']
    )
    cbar.ax.tick_params(labelsize=plot_cfg['colorbar_fontsize'])

    plt.tight_layout()

    # Save
    if is_2d:
        filename = f'dws_{param_name}_xz.{plot_cfg["save_format"]}'
    else:
        filename = f'dws_{param_name}_xz_y{iy:03d}_y{y[iy]:.2f}km.{plot_cfg["save_format"]}'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=plot_cfg['save_dpi'], bbox_inches='tight')
    plt.close()

    return filepath


def plot_single_yz_slice(dws, x, y, z, ix, param_name, config, output_dir, vmin, vmax):
    """Plot a single YZ (vertical) slice."""
    plot_cfg = config['plotting']

    fig, ax = plt.subplots(figsize=plot_cfg['figsize'])

    # Extract slice
    slice_data = dws[:, :, ix]

    # Plot
    im = ax.imshow(
        slice_data,
        extent=[y[0], y[-1], z[-1], z[0]],
        aspect=plot_cfg['aspect'],
        cmap=plot_cfg['cmap'],
        vmin=vmin,
        vmax=vmax
    )

    # Contours
    if plot_cfg['show_contours']:
        Y, Z = np.meshgrid(y, z)
        ax.contour(
            Y, Z, slice_data,
            levels=plot_cfg['contour_levels'],
            colors=plot_cfg['contour_colors'],
            alpha=plot_cfg['contour_alpha'],
            linewidths=plot_cfg['contour_linewidth']
        )

    ax.set_xlabel('Y (km)', fontsize=plot_cfg['xlabel_fontsize'])
    ax.set_ylabel('Depth (km)', fontsize=plot_cfg['ylabel_fontsize'])
    ax.set_title(f'DWS {param_name.upper()} - YZ Slice at X = {x[ix]:.2f} km',
                 fontsize=plot_cfg['title_fontsize'])
    ax.invert_yaxis()

    if plot_cfg['show_grid']:
        ax.grid(True, alpha=plot_cfg['grid_alpha'], linestyle=plot_cfg['grid_linestyle'])

    # Colorbar
    cbar = plt.colorbar(
        im, ax=ax,
        orientation=plot_cfg['colorbar_orientation'],
        label=plot_cfg['colorbar_label'],
        shrink=plot_cfg['colorbar_shrink'],
        pad=plot_cfg['colorbar_pad']
    )
    cbar.ax.tick_params(labelsize=plot_cfg['colorbar_fontsize'])

    plt.tight_layout()

    # Save
    filename = f'dws_{param_name}_yz_x{ix:03d}_x{x[ix]:.2f}km.{plot_cfg["save_format"]}'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=plot_cfg['save_dpi'], bbox_inches='tight')
    plt.close()

    return filepath


def plot_xy_slices(dws, x, y, z, param_name, config, output_dir):
    """Plot all XY slices (horizontal slices at different depths)."""
    plot_cfg = config['plotting']
    grid_cfg = config['grid']

    if not grid_cfg['is_3d']:
        print(f"  Skipping XY slices for {param_name} (2D data)")
        return []

    slice_fractions = plot_cfg['xy_slices']
    vmin, vmax = get_color_limits(dws, config)

    plot_files = []
    for frac in slice_fractions:
        iz = int(frac * (grid_cfg['nz'] - 1))
        iz = min(iz, grid_cfg['nz'] - 1)

        filepath = plot_single_xy_slice(dws, x, y, z, iz, param_name, config, output_dir, vmin, vmax)
        plot_files.append(filepath)
        print(f"    Saved: {filepath.name}")

    return plot_files


def plot_xz_slices(dws, x, y, z, param_name, config, output_dir):
    """Plot all XZ slices (vertical slices along Y)."""
    plot_cfg = config['plotting']
    grid_cfg = config['grid']

    vmin, vmax = get_color_limits(dws, config)
    plot_files = []

    if not grid_cfg['is_3d']:
        # For 2D, plot the single XZ slice
        filepath = plot_single_xz_slice(dws, x, y, z, None, param_name, config, output_dir, vmin, vmax, is_2d=True)
        plot_files.append(filepath)
        print(f"    Saved: {filepath.name}")
        return plot_files

    # 3D case
    slice_fractions = plot_cfg['xz_slices']

    for frac in slice_fractions:
        iy = int(frac * (grid_cfg['ny'] - 1))
        iy = min(iy, grid_cfg['ny'] - 1)

        filepath = plot_single_xz_slice(dws, x, y, z, iy, param_name, config, output_dir, vmin, vmax, is_2d=False)
        plot_files.append(filepath)
        print(f"    Saved: {filepath.name}")

    return plot_files


def plot_yz_slices(dws, x, y, z, param_name, config, output_dir):
    """Plot all YZ slices (vertical slices along X)."""
    plot_cfg = config['plotting']
    grid_cfg = config['grid']

    if not grid_cfg['is_3d']:
        print(f"  Skipping YZ slices for {param_name} (2D data)")
        return []

    slice_fractions = plot_cfg['yz_slices']
    vmin, vmax = get_color_limits(dws, config)

    plot_files = []
    for frac in slice_fractions:
        ix = int(frac * (grid_cfg['nx'] - 1))
        ix = min(ix, grid_cfg['nx'] - 1)

        filepath = plot_single_yz_slice(dws, x, y, z, ix, param_name, config, output_dir, vmin, vmax)
        plot_files.append(filepath)
        print(f"    Saved: {filepath.name}")

    return plot_files


def plot_vp_vs_comparison(dws_vp, dws_vs, x, y, z, config, output_dir):
    """Plot side-by-side comparison of VP and VS DWS."""
    plot_cfg = config['plotting']
    grid_cfg = config['grid']

    if dws_vp is None or dws_vs is None:
        print("  Skipping VP/VS comparison (one or both parameters missing)")
        return []

    plot_files = []

    # For 2D
    if not grid_cfg['is_3d']:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Use same color scale for both
        vmin = min(np.min(dws_vp), np.min(dws_vs))
        vmax = max(np.max(dws_vp), np.max(dws_vs))

        # VP
        im1 = axes[0].imshow(
            dws_vp,
            extent=[x[0], x[-1], z[-1], z[0]],
            aspect=plot_cfg['aspect'],
            cmap=plot_cfg['cmap'],
            vmin=vmin,
            vmax=vmax
        )
        axes[0].set_xlabel('X (km)', fontsize=plot_cfg['xlabel_fontsize'])
        axes[0].set_ylabel('Depth (km)', fontsize=plot_cfg['ylabel_fontsize'])
        axes[0].set_title('DWS VP', fontsize=plot_cfg['title_fontsize'])
        axes[0].invert_yaxis()
        plt.colorbar(im1, ax=axes[0], label=plot_cfg['colorbar_label'])

        # VS
        im2 = axes[1].imshow(
            dws_vs,
            extent=[x[0], x[-1], z[-1], z[0]],
            aspect=plot_cfg['aspect'],
            cmap=plot_cfg['cmap'],
            vmin=vmin,
            vmax=vmax
        )
        axes[1].set_xlabel('X (km)', fontsize=plot_cfg['xlabel_fontsize'])
        axes[1].set_ylabel('Depth (km)', fontsize=plot_cfg['ylabel_fontsize'])
        axes[1].set_title('DWS VS', fontsize=plot_cfg['title_fontsize'])
        axes[1].invert_yaxis()
        plt.colorbar(im2, ax=axes[1], label=plot_cfg['colorbar_label'])

        # Ratio VP/VS
        ratio = np.divide(dws_vp, dws_vs, where=dws_vs > 1e-10, out=np.ones_like(dws_vp))
        im3 = axes[2].imshow(
            ratio,
            extent=[x[0], x[-1], z[-1], z[0]],
            aspect=plot_cfg['aspect'],
            cmap=plot_cfg['cmap_diverging'],
            vmin=0.5,
            vmax=1.5
        )
        axes[2].set_xlabel('X (km)', fontsize=plot_cfg['xlabel_fontsize'])
        axes[2].set_ylabel('Depth (km)', fontsize=plot_cfg['ylabel_fontsize'])
        axes[2].set_title('DWS Ratio (VP/VS)', fontsize=plot_cfg['title_fontsize'])
        axes[2].invert_yaxis()
        plt.colorbar(im3, ax=axes[2], label='Ratio')

        plt.tight_layout()

        filename = f'dws_vp_vs_comparison_2d.{plot_cfg["save_format"]}'
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=plot_cfg['save_dpi'], bbox_inches='tight')
        plt.close()
        plot_files.append(filepath)

        print(f"    Saved: {filepath.name}")
        return plot_files

    # For 3D - create comparisons at multiple depths
    slice_fractions = plot_cfg['xy_slices']

    for frac in slice_fractions:
        iz = int(frac * (grid_cfg['nz'] - 1))
        iz = min(iz, grid_cfg['nz'] - 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        vmin = min(np.min(dws_vp[iz, :, :]), np.min(dws_vs[iz, :, :]))
        vmax = max(np.max(dws_vp[iz, :, :]), np.max(dws_vs[iz, :, :]))

        # VP
        im1 = axes[0].imshow(
            dws_vp[iz, :, :],
            extent=[x[0], x[-1], y[0], y[-1]],
            origin='lower',
            aspect=plot_cfg['aspect'],
            cmap=plot_cfg['cmap'],
            vmin=vmin,
            vmax=vmax
        )
        axes[0].set_xlabel('X (km)', fontsize=plot_cfg['xlabel_fontsize'])
        axes[0].set_ylabel('Y (km)', fontsize=plot_cfg['ylabel_fontsize'])
        axes[0].set_title(f'DWS VP', fontsize=plot_cfg['title_fontsize'])
        plt.colorbar(im1, ax=axes[0], label=plot_cfg['colorbar_label'])

        # VS
        im2 = axes[1].imshow(
            dws_vs[iz, :, :],
            extent=[x[0], x[-1], y[0], y[-1]],
            origin='lower',
            aspect=plot_cfg['aspect'],
            cmap=plot_cfg['cmap'],
            vmin=vmin,
            vmax=vmax
        )
        axes[1].set_xlabel('X (km)', fontsize=plot_cfg['xlabel_fontsize'])
        axes[1].set_ylabel('Y (km)', fontsize=plot_cfg['ylabel_fontsize'])
        axes[1].set_title(f'DWS VS', fontsize=plot_cfg['title_fontsize'])
        plt.colorbar(im2, ax=axes[1], label=plot_cfg['colorbar_label'])

        # Ratio
        vp_slice = dws_vp[iz, :, :]
        vs_slice = dws_vs[iz, :, :]
        ratio = np.divide(vp_slice, vs_slice, where=vs_slice > 1e-10, out=np.ones_like(vp_slice))

        im3 = axes[2].imshow(
            ratio,
            extent=[x[0], x[-1], y[0], y[-1]],
            origin='lower',
            aspect=plot_cfg['aspect'],
            cmap=plot_cfg['cmap_diverging'],
            vmin=0.5,
            vmax=1.5
        )
        axes[2].set_xlabel('X (km)', fontsize=plot_cfg['xlabel_fontsize'])
        axes[2].set_ylabel('Y (km)', fontsize=plot_cfg['ylabel_fontsize'])
        axes[2].set_title(f'DWS Ratio VP/VS', fontsize=plot_cfg['title_fontsize'])
        plt.colorbar(im3, ax=axes[2], label='Ratio')

        plt.suptitle(f'VP/VS Comparison at Depth = {z[iz]:.2f} km',
                     fontsize=plot_cfg['title_fontsize'] + 2)
        plt.tight_layout()

        filename = f'dws_vp_vs_comparison_z{iz:03d}_depth{z[iz]:.2f}km.{plot_cfg["save_format"]}'
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=plot_cfg['save_dpi'], bbox_inches='tight')
        plt.close()
        plot_files.append(filepath)

        print(f"    Saved: {filepath.name}")

    return plot_files


def plot_histograms(dws_vp, dws_vs, config, output_dir):
    """Plot histograms of DWS values."""
    plot_cfg = config['plotting']
    plot_files = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if dws_vp is not None:
        axes[0].hist(dws_vp.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_xlabel('DWS Value', fontsize=plot_cfg['xlabel_fontsize'])
        axes[0].set_ylabel('Frequency', fontsize=plot_cfg['ylabel_fontsize'])
        axes[0].set_title('DWS VP Distribution', fontsize=plot_cfg['title_fontsize'])
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=plot_cfg['grid_alpha'])

    if dws_vs is not None:
        axes[1].hist(dws_vs.flatten(), bins=100, alpha=0.7, color='red', edgecolor='black')
        axes[1].set_xlabel('DWS Value', fontsize=plot_cfg['xlabel_fontsize'])
        axes[1].set_ylabel('Frequency', fontsize=plot_cfg['ylabel_fontsize'])
        axes[1].set_title('DWS VS Distribution', fontsize=plot_cfg['title_fontsize'])
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=plot_cfg['grid_alpha'])

    plt.tight_layout()

    filename = f'dws_histograms.{plot_cfg["save_format"]}'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=plot_cfg['save_dpi'], bbox_inches='tight')
    plt.close()
    plot_files.append(filepath)

    print(f"    Saved: {filepath.name}")
    return plot_files


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_markdown_report(stats_vp, stats_vs, config, output_dir, plot_files):
    """Generate a markdown report."""
    report_path = output_dir / f"{config['report']['report_filename']}.md"

    # Organize plot files by type
    xy_plots = [f for f in plot_files if '_xy_' in f.name]
    xz_plots = [f for f in plot_files if '_xz_' in f.name or f.name.endswith('_xz.png')]
    yz_plots = [f for f in plot_files if '_yz_' in f.name]
    comparison_plots = [f for f in plot_files if 'comparison' in f.name]
    histogram_plots = [f for f in plot_files if 'histogram' in f.name]

    with open(report_path, 'w') as f:
        f.write("# DWS Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- **DWS Directory:** `{config['paths']['dws_directory']}`\n")
        f.write(f"- **Grid Dimensions:** {config['grid']['nz']} × {config['grid']['nx']}")
        if config['grid']['is_3d']:
            f.write(f" × {config['grid']['ny']}")
        f.write(f" ({'3D' if config['grid']['is_3d'] else '2D'})\n")
        f.write(f"- **Normalization:** {config['analysis']['normalization']}\n")
        f.write(f"- **Constraint Threshold:** {config['analysis']['constraint_threshold'] * 100:.1f}%\n\n")

        # VP Statistics
        if stats_vp is not None:
            f.write("## DWS VP Statistics\n\n")
            f.write(f"- **Min:** {stats_vp['min']:.6e}\n")
            f.write(f"- **Max:** {stats_vp['max']:.6e}\n")
            f.write(f"- **Mean:** {stats_vp['mean']:.6e}\n")
            f.write(f"- **Median:** {stats_vp['median']:.6e}\n")
            f.write(f"- **Std Dev:** {stats_vp['std']:.6e}\n")
            f.write(f"- **Well-constrained fraction:** {stats_vp['well_constrained_fraction']*100:.2f}%\n")
            f.write(f"- **Near-zero fraction:** {stats_vp['near_zero_fraction']*100:.2f}%\n\n")

            f.write("### Percentiles (VP)\n\n")
            f.write("| Percentile | Value |\n")
            f.write("|------------|-------|\n")
            for key, val in stats_vp['percentiles'].items():
                f.write(f"| {key} | {val:.6e} |\n")
            f.write("\n")

        # VS Statistics
        if stats_vs is not None:
            f.write("## DWS VS Statistics\n\n")
            f.write(f"- **Min:** {stats_vs['min']:.6e}\n")
            f.write(f"- **Max:** {stats_vs['max']:.6e}\n")
            f.write(f"- **Mean:** {stats_vs['mean']:.6e}\n")
            f.write(f"- **Median:** {stats_vs['median']:.6e}\n")
            f.write(f"- **Std Dev:** {stats_vs['std']:.6e}\n")
            f.write(f"- **Well-constrained fraction:** {stats_vs['well_constrained_fraction']*100:.2f}%\n")
            f.write(f"- **Near-zero fraction:** {stats_vs['near_zero_fraction']*100:.2f}%\n\n")

            f.write("### Percentiles (VS)\n\n")
            f.write("| Percentile | Value |\n")
            f.write("|------------|-------|\n")
            for key, val in stats_vs['percentiles'].items():
                f.write(f"| {key} | {val:.6e} |\n")
            f.write("\n")

        # Interpretation
        f.write("## Interpretation\n\n")
        f.write("### DWS Values\n")
        f.write("- **High DWS:** Indicates regions well-constrained by data (good ray coverage)\n")
        f.write("- **Low DWS:** Indicates poorly constrained regions (sparse coverage)\n")
        f.write("- **Near-zero DWS:** Indicates no sensitivity (no rays passing through)\n\n")

        f.write("### Coverage Analysis\n")
        if stats_vp is not None:
            f.write(f"- **VP:** {stats_vp['well_constrained_fraction']*100:.1f}% of the model is well-constrained ")
            f.write(f"(DWS > {stats_vp['constraint_threshold']:.6e})\n")
        if stats_vs is not None:
            f.write(f"- **VS:** {stats_vs['well_constrained_fraction']*100:.1f}% of the model is well-constrained ")
            f.write(f"(DWS > {stats_vs['constraint_threshold']:.6e})\n")
        f.write("\n")

        # Plots
        if config['report']['include_plots'] and plot_files:
            f.write("## Visualizations\n\n")

            if histogram_plots:
                f.write("### Histograms\n\n")
                for plot_file in sorted(histogram_plots):
                    f.write(f"![{plot_file.stem}]({plot_file.name})\n\n")

            if comparison_plots:
                f.write("### VP/VS Comparisons\n\n")
                for plot_file in sorted(comparison_plots):
                    f.write(f"![{plot_file.stem}]({plot_file.name})\n\n")

            if xy_plots:
                f.write("### XY Slices (Horizontal)\n\n")
                for plot_file in sorted(xy_plots):
                    f.write(f"![{plot_file.stem}]({plot_file.name})\n\n")

            if xz_plots:
                f.write("### XZ Slices (Vertical)\n\n")
                for plot_file in sorted(xz_plots):
                    f.write(f"![{plot_file.stem}]({plot_file.name})\n\n")

            if yz_plots:
                f.write("### YZ Slices (Vertical)\n\n")
                for plot_file in sorted(yz_plots):
                    f.write(f"![{plot_file.stem}]({plot_file.name})\n\n")

    print(f"    Saved: {report_path.name}")
    return report_path


def generate_json_report(stats_vp, stats_vs, config, output_dir):
    """Generate a JSON report with statistics."""
    report_path = output_dir / f"{config['report']['report_filename']}.json"

    report_data = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'dws_directory': str(config['paths']['dws_directory']),
            'grid_dimensions': {
                'nz': config['grid']['nz'],
                'nx': config['grid']['nx'],
                'ny': config['grid']['ny'],
                'is_3d': config['grid']['is_3d'],
            },
            'normalization': config['analysis']['normalization'],
            'constraint_threshold': config['analysis']['constraint_threshold'],
        },
        'statistics': {
            'vp': stats_vp,
            'vs': stats_vs,
        }
    }

    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)

    print(f"    Saved: {report_path.name}")
    return report_path


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def main():
    """Main analysis function."""
    print("\n" + "="*70)
    print("DWS Analysis for LATTE Traveltime Inversion")
    print("="*70 + "\n")

    # Setup
    setup_plot_style(CONFIG)

    # Create output directory
    output_dir = Path(CONFIG['paths']['output_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Read DWS files
    dws_dir = Path(CONFIG['paths']['dws_directory'])
    grid_cfg = CONFIG['grid']

    dws_vp = None
    dws_vs = None
    dws_vp_norm = None
    dws_vs_norm = None

    if CONFIG['analysis']['analyze_vp']:
        vp_file = dws_dir / CONFIG['paths']['dws_vp_file']
        print(f"Reading VP DWS from: {vp_file}")
        try:
            dws_vp = read_dws_binary(
                vp_file,
                grid_cfg['nz'],
                grid_cfg['nx'],
                grid_cfg['ny'],
                grid_cfg['is_3d']
            )
            print(f"  Shape: {dws_vp.shape}")
            print(f"  Range: [{np.min(dws_vp):.6e}, {np.max(dws_vp):.6e}]\n")
        except Exception as e:
            print(f"  Error reading VP: {e}\n")

    if CONFIG['analysis']['analyze_vs']:
        vs_file = dws_dir / CONFIG['paths']['dws_vs_file']
        print(f"Reading VS DWS from: {vs_file}")
        try:
            dws_vs = read_dws_binary(
                vs_file,
                grid_cfg['nz'],
                grid_cfg['nx'],
                grid_cfg['ny'],
                grid_cfg['is_3d']
            )
            print(f"  Shape: {dws_vs.shape}")
            print(f"  Range: [{np.min(dws_vs):.6e}, {np.max(dws_vs):.6e}]\n")
        except Exception as e:
            print(f"  Error reading VS: {e}\n")

    if dws_vp is None and dws_vs is None:
        print("ERROR: No DWS data could be loaded. Exiting.")
        return

    # Normalize
    print("Normalizing DWS arrays...")
    dws_vp_norm, dws_vs_norm = normalize_dws(
        dws_vp, dws_vs,
        method=CONFIG['analysis']['normalization']
    )
    print("  Done.\n")

    # Compute statistics
    stats_vp = None
    stats_vs = None

    if dws_vp_norm is not None:
        print("Computing VP statistics...")
        stats_vp = compute_statistics(dws_vp_norm, CONFIG)
        print("  Done.\n")

    if dws_vs_norm is not None:
        print("Computing VS statistics...")
        stats_vs = compute_statistics(dws_vs_norm, CONFIG)
        print("  Done.\n")

    # Create coordinate arrays
    x, y, z = create_coordinate_arrays(CONFIG)

    # Generate plots
    print("Generating plots...\n")
    all_plot_files = []

    if dws_vp_norm is not None:
        print("  VP XY slices (horizontal):")
        plot_files = plot_xy_slices(dws_vp_norm, x, y, z, 'vp', CONFIG, output_dir)
        all_plot_files.extend(plot_files)

        print("  VP XZ slices (vertical):")
        plot_files = plot_xz_slices(dws_vp_norm, x, y, z, 'vp', CONFIG, output_dir)
        all_plot_files.extend(plot_files)

        print("  VP YZ slices (vertical):")
        plot_files = plot_yz_slices(dws_vp_norm, x, y, z, 'vp', CONFIG, output_dir)
        all_plot_files.extend(plot_files)
        print()

    if dws_vs_norm is not None:
        print("  VS XY slices (horizontal):")
        plot_files = plot_xy_slices(dws_vs_norm, x, y, z, 'vs', CONFIG, output_dir)
        all_plot_files.extend(plot_files)

        print("  VS XZ slices (vertical):")
        plot_files = plot_xz_slices(dws_vs_norm, x, y, z, 'vs', CONFIG, output_dir)
        all_plot_files.extend(plot_files)

        print("  VS YZ slices (vertical):")
        plot_files = plot_yz_slices(dws_vs_norm, x, y, z, 'vs', CONFIG, output_dir)
        all_plot_files.extend(plot_files)
        print()

    if dws_vp_norm is not None and dws_vs_norm is not None:
        print("  VP/VS comparison plots:")
        plot_files = plot_vp_vs_comparison(dws_vp_norm, dws_vs_norm, x, y, z, CONFIG, output_dir)
        all_plot_files.extend(plot_files)
        print()

    print("  Histogram plots:")
    plot_files = plot_histograms(dws_vp_norm, dws_vs_norm, CONFIG, output_dir)
    all_plot_files.extend(plot_files)
    print()

    # Generate reports
    print("Generating reports:\n")

    if CONFIG['report']['generate_markdown']:
        generate_markdown_report(stats_vp, stats_vs, CONFIG, output_dir, all_plot_files)

    if CONFIG['report']['generate_json']:
        generate_json_report(stats_vp, stats_vs, CONFIG, output_dir)

    print("\n" + "="*70)
    print(f"Analysis complete! {len(all_plot_files)} plots generated.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
