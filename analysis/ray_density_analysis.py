"""
Main ray density analysis module for LATTE.

This module orchestrates ray density analysis using actual traced ray paths from cache.
Unlike the preprocessing module which uses straight lines, this uses the actual curved
ray paths from the postprocessing module.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm

from .ray_density_data import compute_ray_density_from_cache
from .ray_density_plots import (
    plot_ray_density_xy,
    plot_ray_density_xz,
    plot_ray_density_yz,
    plot_ray_density_xy_threshold_count,
    plot_ray_density_depth_slices,
    plot_ray_density_oblique_section,
    plot_ray_density_section_plan_view,
)
from .raypath_data import (
    load_event_ids_from_file,
    convert_event_ids_to_shot_ids
)
from .basin_boundaries import load_basin_boundaries
from utils.io_utils import ensure_directory_exists
from utils.plot_utils import setup_plot_style

logger = logging.getLogger(__name__)


def analyze_ray_density(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main ray density analysis function.

    Computes and visualizes ray density using actual traced ray paths from cache.

    Args:
        config_dict: Configuration dictionary with 'ray_density_analysis' key

    Returns:
        Dictionary with analysis results including statistics and plot paths
    """
    config = config_dict['ray_density_analysis']
    setup_plot_style(config_dict.get('analysis', {}))

    logger.info("="*80)
    logger.info("STARTING RAY DENSITY ANALYSIS")
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

    # Get grid dimensions. There is no velocity model among the inputs, so the grid is
    # defined in the configuration; the origin keys are origin_x/y/z (legacy x_min/y_min/z_min
    # are still accepted).
    grid_config = config['grid']
    origin = {}
    for axis in ('x', 'y', 'z'):
        if grid_config.get(f'origin_{axis}') is not None:
            origin[axis] = grid_config[f'origin_{axis}']
        elif grid_config.get(f'{axis}_min') is not None:
            logger.warning(f"ray_density.grid.{axis}_min is a legacy key; use origin_{axis}")
            origin[axis] = grid_config[f'{axis}_min']
        else:
            raise KeyError(f"ray_density.grid.origin_{axis} is required")
    grid_dims = {
        'nx': grid_config['nx'],
        'ny': grid_config['ny'],
        'nz': grid_config['nz'],
        'dx': grid_config['dx'],
        'dy': grid_config['dy'],
        'dz': grid_config['dz'],
        'x_min': origin['x'],
        'y_min': origin['y'],
        'z_min': origin['z'],
        'x_max': origin['x'] + (grid_config['nx'] - 1) * grid_config['dx'],
        'y_max': origin['y'] + (grid_config['ny'] - 1) * grid_config['dy'],
        'z_max': origin['z'] + (grid_config['nz'] - 1) * grid_config['dz'],
    }

    logger.info(f"Grid dimensions: {grid_dims['nx']} x {grid_dims['ny']} x {grid_dims['nz']}")
    logger.info(f"  X: [{grid_dims['x_min']:.2f}, {grid_dims['x_max']:.2f}] km")
    logger.info(f"  Y: [{grid_dims['y_min']:.2f}, {grid_dims['y_max']:.2f}] km")
    logger.info(f"  Z: [{grid_dims['z_min']:.2f}, {grid_dims['z_max']:.2f}] km")

    # Build plot configuration
    plot_style = config_dict.get('analysis', {}).get('plot_style', {})
    plot_config = {
        'dpi': config_dict.get('analysis', {}).get('dpi', 300),
        'label_fontsize': plot_style.get('label_size', 14),
        'title_fontsize': plot_style.get('title_size', 16),
        'tick_fontsize': plot_style.get('tick_size', 12),
        'figure_width': config.get('figure_width', 12),
        'figure_height': config.get('figure_height', 10),
        'colormap': config.get('colormap', 'hot'),
    }

    # Load basin boundaries if configured (global basin_boundaries section)
    basin_data = {}
    basin_config = config_dict.get('basin_boundaries', {})
    if config.get('plot_basin_boundaries', False):
        basin_data = load_basin_boundaries(basin_config)
        if basin_data:
            logger.info(f"Loaded {len(basin_data)} basin boundaries for XY plane plots")

    # Get processing settings
    phase_type = config.get('phase_type', 'both')
    threshold = config.get('threshold', 0)
    num_samples_per_ray = config.get('num_samples_per_ray', 1000)
    n_jobs = config.get('n_jobs', 1)
    batch_size = config.get('batch_size', 1000)
    planes = config.get('planes', ['xy', 'xz', 'yz'])
    enable_xy_threshold_count = config.get('enable_xy_threshold_count', True)
    enable_depth_slices = config.get('enable_depth_slices', True)
    depth_slices = config.get('depth_slices', [])

    # Oblique section settings
    enable_oblique_sections = config.get('enable_oblique_sections', False)
    enable_oblique_section_plan_view = config.get('enable_oblique_section_plan_view', False)
    oblique_sections = config.get('oblique_sections', [])

    # Event filtering/subsampling (priority: event_id_list_file > max_rays > subsample_fraction)
    event_id_list_file = config.get('event_id_list_file', None)
    max_rays = config.get('max_rays', None)
    subsample_fraction = config.get('subsample_fraction', 1.0)
    filter_shot_ids = None

    if event_id_list_file and event_id_list_file != "null":
        # Load event IDs and convert to shot IDs
        try:
            event_ids = load_event_ids_from_file(event_id_list_file)
            shot_map_file = config_dict.get('paths', {}).get('shot_map_file', None)
            if not shot_map_file:
                shot_map_file = config.get('shot_map_file', None)

            if shot_map_file:
                filter_shot_ids = convert_event_ids_to_shot_ids(event_ids, shot_map_file)
                logger.info(f"Filtering to {len(filter_shot_ids)} specific events")
            else:
                logger.warning("event_id_list_file provided but shot_map_file not found, ignoring filter")
        except Exception as e:
            logger.warning(f"Failed to load event ID list: {e}, ignoring filter")

    logger.info(f"Processing settings:")
    logger.info(f"  Phase type: {phase_type}")
    logger.info(f"  Threshold: {threshold} rays")
    logger.info(f"  Samples per ray: {num_samples_per_ray}")
    logger.info(f"  Parallel jobs: {n_jobs} (-1 = all cores)")
    logger.info(f"  Batch size: {batch_size}")
    if filter_shot_ids:
        logger.info(f"  Filtering to {len(filter_shot_ids)} events")
    elif max_rays:
        logger.info(f"  Max rays to load: {max_rays:,}")
    elif subsample_fraction < 1.0:
        logger.info(f"  Subsampling: {subsample_fraction*100:.1f}%")

    # Determine phases to process
    phases = []
    if phase_type in ['both', 'p']:
        phases.append('p')
    if phase_type in ['both', 's']:
        phases.append('s')

    results = {
        'statistics': {},
        'plots': {
            'projections': [],
            'threshold_count': [],
            'depth_slices': [],
            'oblique_sections': [],
            'plan_view': [],
        }
    }

    # ── Oblique section plan view (geometry only; generated once before phase loop) ──
    if (enable_oblique_section_plan_view or enable_oblique_sections) and oblique_sections:
        if enable_oblique_section_plan_view:
            plan_view_file = str(output_dir / 'plots' / 'ray_density_oblique_section_plan_view.png')
            plot_ray_density_section_plan_view(
                oblique_sections, grid_dims, plan_view_file, plot_config,
                basin_data=basin_data, basin_config=basin_config,
            )
            results['plots']['plan_view'].append({'output_file': plan_view_file})
    elif enable_oblique_section_plan_view and not oblique_sections:
        logger.warning("enable_oblique_section_plan_view is True but no oblique_sections defined")

    # Process each phase
    for phase in phases:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {phase.upper()}-wave ray density")
        logger.info(f"{'='*60}")

        # Compute ray density from cache
        density_3d = compute_ray_density_from_cache(
            cache_path,
            grid_dims,
            phase=phase,
            max_rays=max_rays,
            shot_ids=filter_shot_ids,
            num_samples_per_ray=num_samples_per_ray,
            show_progress=True,
            n_jobs=n_jobs,
            batch_size=batch_size
        )

        # Compute statistics
        stats = {
            'phase': phase,
            'total_cells': int(density_3d.size),
            'nonzero_cells': int((density_3d > 0).sum()),
            'coverage_percentage': float(100 * (density_3d > 0).sum() / density_3d.size),
            'min_density': int(density_3d[density_3d > 0].min()) if (density_3d > 0).any() else 0,
            'max_density': int(density_3d.max()),
            'mean_density': float(density_3d[density_3d > 0].mean()) if (density_3d > 0).any() else 0.0,
            'threshold': threshold,
            'cells_above_threshold': int((density_3d >= threshold).sum()),
        }
        results['statistics'][phase] = stats

        logger.info(f"\n{phase.upper()}-wave statistics:")
        logger.info(f"  Coverage: {stats['coverage_percentage']:.1f}%")
        logger.info(f"  Density range: [{stats['min_density']}, {stats['max_density']}]")
        logger.info(f"  Mean density: {stats['mean_density']:.2f}")

        # Save statistics
        stats_file = output_dir / 'statistics' / f'ray_density_{phase}_wave_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"  Saved statistics: {stats_file}")

        # Create projection plots
        logger.info(f"\nCreating projection plots...")
        for plane in planes:
            output_file = str(output_dir / 'plots' / f'ray_density_{phase}_wave_{plane}.png')

            if plane == 'xy':
                plot_ray_density_xy(density_3d, grid_dims, output_file, threshold, plot_config,
                                    basin_data=basin_data, basin_config=basin_config)
            elif plane == 'xz':
                plot_ray_density_xz(density_3d, grid_dims, output_file, threshold, plot_config)
            elif plane == 'yz':
                plot_ray_density_yz(density_3d, grid_dims, output_file, threshold, plot_config)

            results['plots']['projections'].append({
                'phase': phase,
                'plane': plane,
                'output_file': output_file
            })

        # Create XY threshold count plot
        if enable_xy_threshold_count:
            output_file = str(output_dir / 'plots' / f'ray_density_{phase}_wave_xy_threshold_count.png')
            plot_ray_density_xy_threshold_count(density_3d, grid_dims, output_file, threshold, plot_config,
                                                 basin_data=basin_data, basin_config=basin_config)
            results['plots']['threshold_count'].append({
                'phase': phase,
                'output_file': output_file
            })

        # Create depth slice plots
        if enable_depth_slices and depth_slices:
            plot_ray_density_depth_slices(
                density_3d, grid_dims, output_dir / 'plots',
                depth_slices, threshold, plot_config,
                basin_data=basin_data, basin_config=basin_config
            )
            for depth in depth_slices:
                results['plots']['depth_slices'].append({
                    'phase': phase,
                    'depth_km': depth
                })

        # Create oblique section plots
        if enable_oblique_sections and oblique_sections:
            logger.info(f"\nCreating oblique section plots ({len(oblique_sections)} sections)...")
            for sec in oblique_sections:
                lbl = sec.get('label', '')
                output_file = str(
                    output_dir / 'plots' /
                    f'ray_density_{phase}_wave_oblique_{lbl}.png'
                )
                try:
                    plot_ray_density_oblique_section(
                        density_3d, grid_dims, output_file, sec,
                        threshold, plot_config,
                        basin_data=basin_data, basin_config=basin_config,
                    )
                    results['plots']['oblique_sections'].append({
                        'phase': phase,
                        'label': lbl,
                        'output_file': output_file,
                    })
                except ValueError as exc:
                    logger.warning(f"  Skipping oblique section '{lbl}': {exc}")
        elif enable_oblique_sections and not oblique_sections:
            logger.warning("enable_oblique_sections is True but no oblique_sections defined")

    logger.info("="*80)
    logger.info("RAY DENSITY ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)

    return results
