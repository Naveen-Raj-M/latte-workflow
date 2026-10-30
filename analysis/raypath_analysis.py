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
from typing import Dict, Any

import pandas as pd

from scipy.spatial import KDTree
from tqdm import tqdm
from utils.io_utils import read_csv_file
from utils.traveltime_io import km_per_unit

from .raypath_data import (
    load_ray_paths_from_cache,
    get_ray_path_metadata,
    get_ray_path_statistics,
    load_event_ids_from_file,
    convert_event_ids_to_shot_ids
)
from .raypath_plots import (
    plot_ray_paths_projection,
    create_ray_path_summary_plot
)
from .basin_boundaries import load_basin_boundaries
from utils.io_utils import ensure_directory_exists
from utils.plot_utils import setup_plot_style

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
    setup_plot_style(config_dict.get('analysis', {}))

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

    # Load basin boundaries if configured (global basin_boundaries section)
    basin_data = {}
    basin_config = config_dict.get('basin_boundaries', {})
    if config.get('plot_basin_boundaries', False):
        basin_data = load_basin_boundaries(basin_config)
        if basin_data:
            logger.info(f"Loaded {len(basin_data)} basin boundaries for XY plane plots")

    # Get visualization settings
    phase_type = config.get('phase_type', 'both')
    planes = config.get('plot', {}).get('planes', ['xy', 'xz', 'yz'])
    output_format = config_dict.get('analysis', {}).get('output_format', 'png')

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

    # Receiver-coordinate filter from an explicit CSV
    # (produced e.g. by checkerboard_perturbation_analysis.py export_anomalous_cat1)
    # A per-shot KDTree is built so that only the specific (shot, receiver) pairs
    # listed in the CSV are loaded, preventing cross-shot station leakage.
    receiver_filter_by_shot = None
    filter_match_tolerance_km = float(config.get('filter_match_tolerance_km', 0.5))
    filter_csv = config.get('filter_csv', None)
    if filter_csv and filter_csv not in ('null', '~', ''):
        try:
            fdf = read_csv_file(filter_csv)
            logger.info(f"Loaded filter CSV: {filter_csv}  ({len(fdf):,} rows)")

            # Pre-filter shots if shot_id column is present
            if 'shot_id' in fdf.columns and filter_shot_ids is None:
                csv_shot_ids = fdf['shot_id'].dropna().astype(str).unique().tolist()
                try:
                    filter_shot_ids = [int(s) for s in csv_shot_ids if s.isdigit()]
                    logger.info(f"Pre-filtering to {len(filter_shot_ids)} shots from filter CSV")
                except ValueError:
                    logger.warning("shot_id column in filter_csv contains non-integer values; "
                                   "shot-level pre-filtering skipped")

            # Build per-shot KDTrees over allowed receiver coordinates.
            # Using per-shot trees prevents cross-shot station leakage: a station
            # that appears in shot A's allowed set must not accidentally match a
            # ray in shot B that travels to the same station for a different event.
            recv_coord_cols = [
                'station_x', 'station_y', 'station_z'
            ]
            if 'shot_id' in fdf.columns and all(c in fdf.columns for c in recv_coord_cols):
                receiver_filter_by_shot = {}
                for shot_id_str, grp in fdf.groupby(fdf['shot_id'].astype(str)):
                    # ground_truth_csv stores coords in metres (units metadata) → km
                    coords_km = grp[recv_coord_cols].dropna().values * km_per_unit(fdf, 'station_x')
                    if len(coords_km) > 0:
                        receiver_filter_by_shot[shot_id_str] = KDTree(coords_km)
                n_shots_with_filter = len(receiver_filter_by_shot)
                n_recv_total = sum(t.data.shape[0] for t in receiver_filter_by_shot.values())
                logger.info(
                    f"Built per-shot receiver KDTrees: {n_shots_with_filter} shots, "
                    f"{n_recv_total:,} allowed receiver positions "
                    f"(tolerance {filter_match_tolerance_km} km)"
                )
            else:
                logger.warning(
                    "filter_csv missing shot_id or station_x/y/z columns; "
                    "only shot-level pre-filtering will be applied."
                )
        except Exception as e:
            logger.warning(f"Failed to load filter_csv '{filter_csv}': {e} — filter ignored")

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

        # Load ray paths
        logger.info("Loading ray paths from cache...")
        ray_data = load_ray_paths_from_cache(
            cache_path,
            phase=phase,
            max_rays=max_rays,
            shot_ids=filter_shot_ids,
            receiver_filter_by_shot=receiver_filter_by_shot,
            receiver_filter_tolerance=filter_match_tolerance_km,
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
