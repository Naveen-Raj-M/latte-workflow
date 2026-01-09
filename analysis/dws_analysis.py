"""
Main DWS (Derivative Weight Sum) analysis module.

This module orchestrates DWS analysis including:
- Loading DWS data
- Computing statistics
- Creating visualizations (horizontal/vertical slices, histograms, coverage maps)
- Parallel processing of multiple depth slices
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from joblib import Parallel, delayed
from tqdm import tqdm

from .dws_data import (
    load_dws_data,
    get_dws_metadata,
    compute_coverage_statistics,
    filter_dws_by_threshold
)
from .dws_plots import (
    plot_horizontal_slice,
    plot_vertical_slice,
    plot_dws_histogram,
    plot_coverage_map
)
from .dws_reports import generate_dws_report
from utils.io_utils import ensure_directory_exists

logger = logging.getLogger(__name__)


def process_single_horizontal_slice(
    df: pd.DataFrame,
    depth_km: float,
    output_dir: str,
    plot_config: Dict[str, Any],
    log_scale: bool,
    prefix: str = ""
) -> Dict[str, Any]:
    """
    Process a single horizontal slice (wrapper for parallel processing).

    Args:
        df: DWS DataFrame
        depth_km: Depth in km
        output_dir: Output directory
        plot_config: Plot configuration
        log_scale: Use log scale colorbar
        prefix: Filename prefix

    Returns:
        Dictionary with slice information
    """
    try:
        # Linear scale plot
        output_file_linear = os.path.join(
            output_dir,
            f"{prefix}dws_horizontal_depth_{depth_km:.2f}km_linear.png"
        )
        plot_horizontal_slice(
            df, depth_km, output_file_linear, plot_config,
            log_scale=False
        )

        # Log scale plot (if requested)
        if log_scale:
            output_file_log = os.path.join(
                output_dir,
                f"{prefix}dws_horizontal_depth_{depth_km:.2f}km_log.png"
            )
            plot_horizontal_slice(
                df, depth_km, output_file_log, plot_config,
                log_scale=True
            )

        return {
            'depth_km': depth_km,
            'status': 'success',
            'output_files': [output_file_linear] + ([output_file_log] if log_scale else [])
        }

    except Exception as e:
        logger.error(f"Error processing horizontal slice at {depth_km} km: {e}")
        return {
            'depth_km': depth_km,
            'status': 'failed',
            'error': str(e)
        }


def process_single_vertical_slice(
    df: pd.DataFrame,
    slice_type: str,
    position: float,
    output_dir: str,
    plot_config: Dict[str, Any],
    log_scale: bool,
    prefix: str = ""
) -> Dict[str, Any]:
    """
    Process a single vertical slice (wrapper for parallel processing).

    Args:
        df: DWS DataFrame
        slice_type: 'xz' or 'yz'
        position: Position in km
        output_dir: Output directory
        plot_config: Plot configuration
        log_scale: Use log scale colorbar
        prefix: Filename prefix

    Returns:
        Dictionary with slice information
    """
    try:
        # Linear scale plot
        output_file_linear = os.path.join(
            output_dir,
            f"{prefix}dws_vertical_{slice_type}_pos_{position:.2f}km_linear.png"
        )
        plot_vertical_slice(
            df, slice_type, position, output_file_linear, plot_config,
            log_scale=False
        )

        # Log scale plot (if requested)
        if log_scale:
            output_file_log = os.path.join(
                output_dir,
                f"{prefix}dws_vertical_{slice_type}_pos_{position:.2f}km_log.png"
            )
            plot_vertical_slice(
                df, slice_type, position, output_file_log, plot_config,
                log_scale=True
            )

        return {
            'slice_type': slice_type,
            'position': position,
            'status': 'success',
            'output_files': [output_file_linear] + ([output_file_log] if log_scale else [])
        }

    except Exception as e:
        logger.error(f"Error processing {slice_type} vertical slice at {position} km: {e}")
        return {
            'slice_type': slice_type,
            'position': position,
            'status': 'failed',
            'error': str(e)
        }


def analyze_dws(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main DWS analysis function.

    Args:
        config_dict: Configuration dictionary

    Returns:
        Dictionary with analysis results
    """
    config = config_dict['dws_analysis']

    logger.info("="*80)
    logger.info("STARTING DWS ANALYSIS")
    logger.info("="*80)

    # Create output directories
    output_dir = Path(config['output_dir'])
    ensure_directory_exists(output_dir)
    ensure_directory_exists(output_dir / 'plots')
    ensure_directory_exists(output_dir / 'statistics')
    ensure_directory_exists(output_dir / 'reports')

    # Load DWS data
    logger.info(f"Loading DWS data from: {config['input_hdf5']}")
    df = load_dws_data(config['input_hdf5'], show_progress=True)

    # Get metadata
    metadata = get_dws_metadata(config['input_hdf5'])

    # Compute overall statistics
    logger.info("Computing coverage statistics...")
    stats = compute_coverage_statistics(df)

    # Save statistics
    stats_file = output_dir / 'statistics' / 'dws_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics: {stats_file}")

    # Merge plot configuration from analysis.plot_style and dws_analysis.plot_config
    # Start with global plot_style settings
    plot_style = config_dict.get('analysis', {}).get('plot_style', {})
    plot_config = config.get('plot_config', {})

    # Build merged config with proper precedence (dws_analysis.plot_config overrides plot_style)
    merged_config = {
        # Base settings from plot_style
        'dpi': config_dict.get('analysis', {}).get('dpi', 300),
        'label_fontsize': plot_style.get('label_size', 12),
        'title_fontsize': plot_style.get('title_size', 14),
        'tick_fontsize': plot_style.get('tick_size', 10),
        # DWS-specific defaults
        'figure_width': 12,
        'figure_height': 10,
        'colormap': 'viridis',
        'colormap_log': 'viridis',
        'equal_aspect': True,
        'grid': True,
    }
    # Override with DWS-specific settings from plot_config
    merged_config.update(plot_config)
    plot_config = merged_config

    # Get visualization settings
    enable_log_scale = config.get('enable_log_scale', True)
    enable_coverage_maps = config.get('enable_coverage_maps', True)
    enable_histograms = config.get('enable_histograms', True)
    enable_vertical_slices = config.get('enable_vertical_slices', False)
    use_m2_units = config.get('use_m2_units', False)

    # Add unit settings to plot_config
    plot_config['use_m2_units'] = use_m2_units

    # Parallel processing settings from performance config
    performance_config = config_dict.get('performance', {})
    use_parallel = performance_config.get('parallel_processing', False)
    n_jobs = performance_config.get('n_jobs', 1)

    results = {
        'statistics': stats,
        'metadata': metadata,
        'plots': {
            'horizontal_slices': [],
            'vertical_slices': [],
            'histograms': [],
            'coverage_maps': []
        }
    }

    # ===== HORIZONTAL SLICES =====
    depths_km = config.get('horizontal_slice_depths', [])
    if depths_km:
        logger.info(f"Creating horizontal slices at {len(depths_km)} depths...")

        if use_parallel and len(depths_km) > 1:
            logger.info(f"Using parallel processing with {n_jobs if n_jobs > 0 else 'all'} jobs")
            slice_results = Parallel(n_jobs=n_jobs)(
                delayed(process_single_horizontal_slice)(
                    df, depth, str(output_dir / 'plots'), plot_config, enable_log_scale
                )
                for depth in tqdm(depths_km, desc="Processing horizontal slices")
            )
        else:
            logger.info("Processing horizontal slices sequentially")
            slice_results = []
            for depth in tqdm(depths_km, desc="Processing horizontal slices"):
                result = process_single_horizontal_slice(
                    df, depth, str(output_dir / 'plots'), plot_config, enable_log_scale
                )
                slice_results.append(result)

        results['plots']['horizontal_slices'] = slice_results

        # Log summary
        n_success = sum(1 for r in slice_results if r['status'] == 'success')
        logger.info(f"Horizontal slices: {n_success}/{len(depths_km)} successful")

    # ===== COVERAGE MAPS =====
    if enable_coverage_maps and depths_km:
        logger.info(f"Creating coverage maps...")
        coverage_threshold = config.get('coverage_threshold', 0.0)

        for depth in tqdm(depths_km, desc="Creating coverage maps"):
            try:
                output_file = output_dir / 'plots' / f'dws_coverage_depth_{depth:.2f}km.png'
                plot_coverage_map(
                    df, depth, str(output_file), plot_config,
                    threshold=coverage_threshold
                )
                results['plots']['coverage_maps'].append({
                    'depth_km': depth,
                    'output_file': str(output_file),
                    'status': 'success'
                })
            except Exception as e:
                logger.error(f"Error creating coverage map at {depth} km: {e}")
                results['plots']['coverage_maps'].append({
                    'depth_km': depth,
                    'status': 'failed',
                    'error': str(e)
                })

    # ===== VERTICAL SLICES =====
    if enable_vertical_slices:
        vertical_slices = config.get('vertical_slices', {})

        # XZ slices (constant Y)
        xz_positions = vertical_slices.get('xz_positions', [])
        if xz_positions:
            logger.info(f"Creating XZ vertical slices at {len(xz_positions)} positions...")

            if use_parallel and len(xz_positions) > 1:
                slice_results = Parallel(n_jobs=n_jobs)(
                    delayed(process_single_vertical_slice)(
                        df, 'xz', pos, str(output_dir / 'plots'), plot_config, enable_log_scale
                    )
                    for pos in tqdm(xz_positions, desc="Processing XZ slices")
                )
            else:
                slice_results = []
                for pos in tqdm(xz_positions, desc="Processing XZ slices"):
                    result = process_single_vertical_slice(
                        df, 'xz', pos, str(output_dir / 'plots'), plot_config, enable_log_scale
                    )
                    slice_results.append(result)

            results['plots']['vertical_slices'].extend(slice_results)

        # YZ slices (constant X)
        yz_positions = vertical_slices.get('yz_positions', [])
        if yz_positions:
            logger.info(f"Creating YZ vertical slices at {len(yz_positions)} positions...")

            if use_parallel and len(yz_positions) > 1:
                slice_results = Parallel(n_jobs=n_jobs)(
                    delayed(process_single_vertical_slice)(
                        df, 'yz', pos, str(output_dir / 'plots'), plot_config, enable_log_scale
                    )
                    for pos in tqdm(yz_positions, desc="Processing YZ slices")
                )
            else:
                slice_results = []
                for pos in tqdm(yz_positions, desc="Processing YZ slices"):
                    result = process_single_vertical_slice(
                        df, 'yz', pos, str(output_dir / 'plots'), plot_config, enable_log_scale
                    )
                    slice_results.append(result)

            results['plots']['vertical_slices'].extend(slice_results)

    # ===== HISTOGRAMS =====
    if enable_histograms:
        logger.info("Creating DWS histograms...")

        # Linear scale histogram
        output_file_linear = output_dir / 'plots' / 'dws_histogram_linear.png'
        plot_dws_histogram(df, str(output_file_linear), plot_config, log_scale_x=False)
        results['plots']['histograms'].append({
            'type': 'linear',
            'output_file': str(output_file_linear),
            'status': 'success'
        })

        # Log scale histogram (if requested)
        if enable_log_scale:
            output_file_log = output_dir / 'plots' / 'dws_histogram_log.png'
            plot_dws_histogram(df, str(output_file_log), plot_config, log_scale_x=True)
            results['plots']['histograms'].append({
                'type': 'log',
                'output_file': str(output_file_log),
                'status': 'success'
            })

    # ===== GENERATE REPORT =====
    if config.get('generate_report', True):
        logger.info("Generating DWS analysis report...")
        report_file = output_dir / 'reports' / 'dws_analysis_report.txt'
        generate_dws_report(stats, metadata, results, str(report_file), use_m2_units=use_m2_units)

    logger.info("="*80)
    logger.info("DWS ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)

    return results


# Import pandas at module level
import pandas as pd
