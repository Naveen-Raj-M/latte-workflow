"""
Main ray path analysis module for LATTE.

This module orchestrates ray path visualization analysis including:
- Loading ray paths from HDF5 cache
- Creating projection plots (XY, XZ, YZ planes)
- Generating statistics and summary reports

This module is designed to work with pre-computed ray path cache files
created by the postprocess module.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from tqdm import tqdm

from .raypath_data import (
    load_ray_paths_from_cache,
    load_ray_paths_from_cache_vectorized,
    get_ray_path_metadata,
    get_ray_path_statistics,
    get_unique_sources,
    get_unique_receivers,
    load_event_ids_from_file,
    convert_event_ids_to_shot_ids
)
from .raypath_plots import (
    plot_ray_paths_projection,
    plot_ray_paths_3d,
    create_ray_path_summary_plot
)
from .basin_boundaries import load_basin_boundaries
from utils.io_utils import ensure_directory_exists

logger = logging.getLogger(__name__)


def analyze_ray_paths(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main ray path visualization analysis function.

    This function loads pre-computed ray paths from a cache file and
    generates visualization plots for different projection planes.

    Args:
        config_dict: Configuration dictionary with 'raypath_analysis' key

    Returns:
        Dictionary with analysis results including:
            - 'statistics': Ray path statistics
            - 'metadata': Cache metadata
            - 'plots': Generated plot files
    """
    config = config_dict['raypath_analysis']

    logger.info("="*80)
    logger.info("STARTING RAY PATH VISUALIZATION ANALYSIS")
    logger.info("="*80)

    # Create output directories
    output_dir = Path(config['output_dir'])
    ensure_directory_exists(output_dir)
    ensure_directory_exists(output_dir / 'plots')
    ensure_directory_exists(output_dir / 'statistics')

    # Get input cache path
    cache_path = config['input_cache']

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Ray path cache not found: {cache_path}")

    logger.info(f"Loading ray paths from: {cache_path}")

    # Get metadata and statistics
    metadata = get_ray_path_metadata(cache_path)
    stats = get_ray_path_statistics(cache_path)

    logger.info(f"Cache contains:")
    logger.info(f"  P-wave: {stats['p_wave_shots']} shots, {stats['p_wave_rays']} rays")
    logger.info(f"  S-wave: {stats['s_wave_shots']} shots, {stats['s_wave_rays']} rays")
    logger.info(f"  File size: {stats['file_size_mb']:.2f} MB")

    # Save statistics
    stats_file = output_dir / 'statistics' / 'raypath_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics: {stats_file}")

    # Build plot configuration
    plot_style = config_dict.get('analysis', {}).get('plot_style', {})
    plot_config_section = config.get('plot', {})

    plot_config = {
        # Base settings from global plot_style
        'dpi': config_dict.get('analysis', {}).get('dpi', 300),
        'label_fontsize': plot_style.get('label_size', 12),
        'title_fontsize': plot_style.get('title_size', 14),
        'legend_fontsize': plot_style.get('legend_size', 10),
        # Defaults
        'figure_width': 12,
        'figure_height': 10,
        'ray_color': 'blue',
        'ray_alpha': 0.3,
        'ray_linewidth': 0.5,
        'source_color': 'red',
        'source_marker': '*',
        'source_size': 100,
        'receiver_color': 'green',
        'receiver_marker': '^',
        'receiver_size': 50,
        'grid': True,
        'equal_aspect': True,
    }
    # Override with raypath-specific settings
    plot_config.update(plot_config_section)

    # Load basin boundaries if configured
    basin_data = {}
    basin_config = config.get('basin_boundaries', {})
    if basin_config.get('enabled', False):
        basin_data = load_basin_boundaries(basin_config)
        if basin_data:
            logger.info(f"Loaded {len(basin_data)} basin boundaries for XY plane plots")

    # Get visualization settings
    phase_type = config.get('phase_type', 'both')
    planes = config.get('plot', {}).get('planes', ['xy', 'xz', 'yz'])
    output_format = config.get('plot', {}).get('format', 'png')

    # Vectorization settings
    use_vectorized = config.get('use_vectorized_loading', True)
    num_samples_per_ray = config.get('num_samples_per_ray', 500)

    # Event filtering/subsampling (priority: event_id_list_file > max_rays > subsample_fraction)
    event_id_list_file = config.get('event_id_list_file', None)
    max_rays = config.get('max_rays', None)
    subsample_fraction = config.get('subsample_fraction', 1.0)
    filter_shot_ids = None

    if event_id_list_file and event_id_list_file != "null":
        # Load event IDs and convert to shot IDs
        try:
            event_ids = load_event_ids_from_file(event_id_list_file)
            # We need shot_map_file - look for it in paths or use common location
            shot_map_file = config_dict.get('paths', {}).get('shot_map_file', None)
            if not shot_map_file:
                # Try to find it from the postprocess config location
                shot_map_file = config.get('shot_map_file', None)

            if shot_map_file:
                filter_shot_ids = convert_event_ids_to_shot_ids(event_ids, shot_map_file)
                logger.info(f"Filtering to {len(filter_shot_ids)} specific events")
            else:
                logger.warning("event_id_list_file provided but shot_map_file not found, ignoring filter")
        except Exception as e:
            logger.warning(f"Failed to load event ID list: {e}, ignoring filter")

    # Add subsampling to plot config
    plot_config['subsample_fraction'] = subsample_fraction
    plot_config['max_rays'] = max_rays
    plot_config['filter_shot_ids'] = filter_shot_ids

    # Determine phases to process
    phases = []
    if phase_type in ['both', 'p']:
        phases.append('p')
    if phase_type in ['both', 's']:
        phases.append('s')

    results = {
        'statistics': stats,
        'metadata': metadata,
        'plots': {
            'projections': [],
            '3d': [],
            'summary': []
        }
    }

    # Process each phase
    for phase in phases:
        phase_key = f'{phase}_wave_rays'
        if stats.get(phase_key, 0) == 0:
            logger.info(f"No {phase.upper()}-wave data in cache, skipping")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {phase.upper()}-wave ray paths")
        logger.info(f"{'='*60}")

        # Load ray paths (vectorized or standard)
        if use_vectorized:
            logger.info(f"Loading ray paths from cache (vectorized, {num_samples_per_ray} samples/ray)...")
            ray_data = load_ray_paths_from_cache_vectorized(
                cache_path,
                phase=phase,
                max_rays=max_rays,
                shot_ids=filter_shot_ids,
                num_samples_per_ray=num_samples_per_ray
            )
        else:
            logger.info("Loading ray paths from cache (standard)...")
            ray_data = load_ray_paths_from_cache(
                cache_path,
                phase=phase,
                max_rays=max_rays,
                shot_ids=filter_shot_ids
            )

        if phase not in ray_data or len(ray_data[phase]['ray_paths']) == 0:
            logger.warning(f"No {phase.upper()}-wave ray paths loaded")
            continue

        phase_data = ray_data[phase]
        ray_paths = phase_data['ray_paths']
        sources = phase_data['sources']
        receivers = phase_data['receivers']

        logger.info(f"Loaded {len(ray_paths)} ray paths")

        # Get unique sources and receivers for plotting
        if use_vectorized:
            # For vectorized data, sources and receivers are already numpy arrays
            # Convert to tuples for set operations
            unique_sources = [tuple(s) for s in np.unique(sources, axis=0)]
            unique_receivers = [tuple(r) for r in np.unique(receivers, axis=0)]
        else:
            # For standard data, sources and receivers are lists of tuples
            unique_sources = list(set(sources))
            unique_receivers = list(set(receivers))

        logger.info(f"Unique sources: {len(unique_sources)}")
        logger.info(f"Unique receivers: {len(unique_receivers)}")

        # Create projection plots
        logger.info("Creating projection plots...")
        for plane in tqdm(planes, desc=f"{phase.upper()}-wave projections"):
            output_file = str(output_dir / 'plots' / f'ray_paths_{phase}_wave_{plane}.{output_format}')

            try:
                plot_ray_paths_projection(
                    ray_paths,
                    unique_sources,
                    unique_receivers,
                    plane,
                    output_file,
                    plot_config,
                    basin_data=basin_data,
                    basin_config=basin_config
                )
                results['plots']['projections'].append({
                    'phase': phase,
                    'plane': plane,
                    'output_file': output_file,
                    'status': 'success'
                })
            except Exception as e:
                logger.error(f"Error creating {plane} plot: {e}")
                results['plots']['projections'].append({
                    'phase': phase,
                    'plane': plane,
                    'status': 'failed',
                    'error': str(e)
                })

        # Create 3D plot if enabled
        if config.get('enable_3d_plot', False):
            logger.info("Creating 3D plot...")
            output_file_3d = str(output_dir / 'plots' / f'ray_paths_{phase}_wave_3d.{output_format}')

            try:
                plot_ray_paths_3d(
                    ray_paths,
                    unique_sources,
                    unique_receivers,
                    output_file_3d,
                    plot_config
                )
                results['plots']['3d'].append({
                    'phase': phase,
                    'output_file': output_file_3d,
                    'status': 'success'
                })
            except Exception as e:
                logger.error(f"Error creating 3D plot: {e}")
                results['plots']['3d'].append({
                    'phase': phase,
                    'status': 'failed',
                    'error': str(e)
                })

    # Create summary plot
    logger.info("Creating summary plot...")
    summary_file = str(output_dir / 'plots' / f'raypath_summary.{output_format}')
    try:
        create_ray_path_summary_plot(stats, summary_file, plot_config)
        results['plots']['summary'].append({
            'output_file': summary_file,
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error creating summary plot: {e}")
        results['plots']['summary'].append({
            'status': 'failed',
            'error': str(e)
        })

    # Generate text report
    if config.get('generate_report', True):
        logger.info("Generating analysis report...")
        report_file = output_dir / 'raypath_analysis_report.txt'
        _generate_report(stats, metadata, results, str(report_file))

    logger.info("="*80)
    logger.info("RAY PATH VISUALIZATION ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)

    return results


def _generate_report(
    stats: Dict[str, Any],
    metadata: Dict[str, Any],
    results: Dict[str, Any],
    output_file: str
) -> None:
    """
    Generate a text report of the ray path analysis.

    Args:
        stats: Ray path statistics
        metadata: Cache metadata
        results: Analysis results
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RAY PATH VISUALIZATION ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        # Cache information
        f.write("CACHE INFORMATION\n")
        f.write("-"*40 + "\n")
        f.write(f"File size: {stats.get('file_size_mb', 0):.2f} MB\n")
        f.write(f"Created: {stats.get('created', 'N/A')}\n")
        f.write(f"Last updated: {stats.get('last_updated', 'N/A')}\n\n")

        # Ray path counts
        f.write("RAY PATH COUNTS\n")
        f.write("-"*40 + "\n")
        f.write(f"P-wave shots: {stats.get('p_wave_shots', 0):,}\n")
        f.write(f"P-wave rays: {stats.get('p_wave_rays', 0):,}\n")
        f.write(f"S-wave shots: {stats.get('s_wave_shots', 0):,}\n")
        f.write(f"S-wave rays: {stats.get('s_wave_rays', 0):,}\n\n")

        # Tracing parameters
        if 'tracing' in metadata:
            f.write("TRACING PARAMETERS\n")
            f.write("-"*40 + "\n")
            tracing = metadata['tracing']
            f.write(f"Max steps: {tracing.get('max_steps', 'N/A')}\n")
            f.write(f"Step size: {tracing.get('step_size', 'N/A')} km\n")
            f.write(f"Tolerance: {tracing.get('tolerance', 'N/A')} km\n")
            f.write(f"Interpolation: {tracing.get('interpolation', 'N/A')}\n\n")

        # Grid parameters
        if 'grid' in metadata:
            f.write("GRID PARAMETERS\n")
            f.write("-"*40 + "\n")
            grid = metadata['grid']
            f.write(f"Dimensions: {grid.get('nx', 'N/A')} x {grid.get('ny', 'N/A')} x {grid.get('nz', 'N/A')}\n")
            f.write(f"Spacing: {grid.get('dx', 'N/A')} x {grid.get('dy', 'N/A')} x {grid.get('dz', 'N/A')} km\n\n")

        # Generated plots
        f.write("GENERATED PLOTS\n")
        f.write("-"*40 + "\n")
        for plot_type, plots in results.get('plots', {}).items():
            f.write(f"\n{plot_type.upper()}:\n")
            for plot in plots:
                status = plot.get('status', 'unknown')
                if status == 'success':
                    f.write(f"  [OK] {plot.get('output_file', 'N/A')}\n")
                else:
                    f.write(f"  [FAILED] {plot.get('error', 'Unknown error')}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    logger.info(f"Saved report: {output_file}")
