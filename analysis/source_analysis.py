"""
Source correction analysis module.

This module provides comprehensive analysis of hypocenter corrections
including spatial, temporal, and depth corrections.
Supports both ground truth comparison and method-to-method comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime

from utils.io_utils import read_csv_file, write_text_file, ensure_directory_exists
from utils.plot_utils import setup_plot_style, get_color_scheme, save_figure
from utils.origin_time_utils import load_origin_time_metadata
from .basin_boundaries import load_basin_boundaries

logger = logging.getLogger(__name__)


def analyze_source_corrections(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function for source correction analysis.
    
    Supports two modes:
    1. Ground truth comparison: Compare corrected vs original source parameters
    2. Method comparison: Compare source parameters from two methods
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    logger.info("Starting source correction analysis")
    
    # Setup plot style
    setup_plot_style(config['analysis'])
    
    # Determine comparison mode
    comparison_mode = config['source_analysis'].get('comparison_mode', 'ground_truth')
    
    if comparison_mode == 'method_comparison':
        logger.info("Running in METHOD COMPARISON mode")
        results = _analyze_method_comparison(config)
    else:
        logger.info("Running in GROUND TRUTH COMPARISON mode")
        results = _analyze_ground_truth_comparison(config)
    
    logger.info("Source correction analysis completed")
    return results


def _analyze_ground_truth_comparison(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze source corrections against ground truth (original parameters).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    input_csv = config['source_analysis']['input_csv']
    output_dir = Path(config['source_analysis']['output_dir'])
    
    logger.info(f"Loading source correction data from: {input_csv}")
    df = read_csv_file(input_csv)

    # Validate and compute corrections
    df_corrected = compute_source_corrections(df, config, comparison_mode='ground_truth')
    
    # Compute statistics
    logger.info("Computing source correction statistics")
    stats = compute_source_statistics(df_corrected, config, comparison_mode='ground_truth')
    
    # Create plots
    logger.info("Creating source correction plots")
    create_source_plots(df_corrected, stats, config, comparison_mode='ground_truth')
    
    # Generate summary report
    if config['source_analysis'].get('generate_summary_report', True):
        logger.info("Generating summary report")
        generate_source_report(df_corrected, stats, config, comparison_mode='ground_truth')
    
    # Save corrected data with deltas
    if config['source_analysis'].get('save_corrected_data', True):
        output_csv = output_dir / 'source_corrections_with_deltas.csv'
        df_corrected.to_csv(output_csv, index=False)
        logger.info(f"Saved corrected data to: {output_csv}")
    
    return stats


def _analyze_method_comparison(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze source parameter comparison between two methods.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    method1_csv = config['source_analysis']['method1_csv']
    method2_csv = config['source_analysis']['method2_csv']
    output_dir = Path(config['source_analysis']['output_dir'])
    
    logger.info(f"Loading Method 1 data from: {method1_csv}")
    logger.info(f"Loading Method 2 data from: {method2_csv}")
    
    df1 = read_csv_file(method1_csv)
    df2 = read_csv_file(method2_csv)

    # Prepare comparison data
    logger.info("Preparing method comparison data")
    df_comparison = prepare_method_comparison_data(df1, df2, config)

    # Compute corrections (differences between methods)
    df_corrected = compute_source_corrections(df_comparison, config,
                                             comparison_mode='method_comparison')
    
    # Compute statistics
    logger.info("Computing method comparison statistics")
    stats = compute_source_statistics(df_corrected, config, 
                                     comparison_mode='method_comparison')
    
    # Create plots
    logger.info("Creating method comparison plots")
    create_source_plots(df_corrected, stats, config, 
                       comparison_mode='method_comparison')
    
    # Generate summary report
    if config['source_analysis'].get('generate_summary_report', True):
        logger.info("Generating summary report")
        generate_source_report(df_corrected, stats, config, 
                             comparison_mode='method_comparison')
    
    # Save comparison data with deltas
    if config['source_analysis'].get('save_corrected_data', True):
        output_csv = output_dir / 'source_method_comparison_with_deltas.csv'
        df_corrected.to_csv(output_csv, index=False)
        logger.info(f"Saved comparison data to: {output_csv}")
    
    return stats


def prepare_method_comparison_data(df1: pd.DataFrame, df2: pd.DataFrame,
                                  config: Dict[str, Any]) -> pd.DataFrame:
    """
    Prepare data for method-to-method comparison.
    
    Args:
        df1: Method 1 dataframe
        df2: Method 2 dataframe
        config: Configuration dictionary
        
    Returns:
        Merged dataframe for comparison
    """
    logger.info(f"Method 1 events: {len(df1)}")
    logger.info(f"Method 2 events: {len(df2)}")

    # Merge on event_id
    merge_key = 'event_id'

    if merge_key not in df1.columns:
        raise ValueError(f"Missing '{merge_key}' column in Method 1 data")
    if merge_key not in df2.columns:
        raise ValueError(f"Missing '{merge_key}' column in Method 2 data")

    # Deduplicate on merge key to prevent cartesian product explosion
    dup1 = df1[merge_key].duplicated().sum()
    dup2 = df2[merge_key].duplicated().sum()
    if dup1 > 0:
        logger.warning(f"Method 1 has {dup1} duplicate event_ids — keeping first occurrence")
        df1 = df1.drop_duplicates(subset=[merge_key], keep='first')
    if dup2 > 0:
        logger.warning(f"Method 2 has {dup2} duplicate event_ids — keeping first occurrence")
        df2 = df2.drop_duplicates(subset=[merge_key], keep='first')

    logger.info(f"Merging datasets on key: {merge_key}")
    df_merged = pd.merge(
        df1, df2,
        on=merge_key,
        suffixes=('_method1', '_method2'),
        how='inner'
    )
    
    logger.info(f"Merged events: {len(df_merged)}")
    
    if len(df_merged) == 0:
        raise ValueError("No matching events found between Method 1 and Method 2 data")
    
    # Rename for consistency with ground truth comparison
    # Method 1 becomes "original", Method 2 becomes "corrected"
    rename_map = {}

    # Handle both possible column name patterns.
    # Prefer corrected_event_*_transformed (post-inversion LATTE locations) over
    # event_*_transformed (initial catalog, identical across all runs) so that
    # method_comparison reflects actual inversion differences.
    for coord in ['x', 'y', 'z']:
        # Pattern 1: corrected_event_x_transformed (LATTE output with both initial/corrected)
        if f'corrected_event_{coord}_transformed_method1' in df_merged.columns:
            rename_map[f'corrected_event_{coord}_transformed_method1'] = f'event_{coord}_transformed'
            rename_map[f'corrected_event_{coord}_transformed_method2'] = f'corrected_event_{coord}_transformed'
        # Pattern 2: event_x_transformed only (single-location output)
        elif f'event_{coord}_transformed_method1' in df_merged.columns:
            rename_map[f'event_{coord}_transformed_method1'] = f'event_{coord}_transformed'
            rename_map[f'event_{coord}_transformed_method2'] = f'corrected_event_{coord}_transformed'
        # Pattern 3: x, y, z (velocity-format coordinate columns)
        elif f'{coord}_method1' in df_merged.columns:
            rename_map[f'{coord}_method1'] = f'event_{coord}_transformed'
            rename_map[f'{coord}_method2'] = f'corrected_event_{coord}_transformed'

    # Handle origin time: prefer corrected times (post-inversion) for method comparison
    if 'corrected_origin_time_s_method1' in df_merged.columns:
        rename_map['corrected_origin_time_s_method1'] = 'origin_time_s'
        rename_map['corrected_origin_time_s_method2'] = 'corrected_origin_time_s'
    elif 'origin_time_s_method1' in df_merged.columns:
        rename_map['origin_time_s_method1'] = 'origin_time_s'
        rename_map['origin_time_s_method2'] = 'corrected_origin_time_s'
    
    df_merged = df_merged.rename(columns=rename_map)
    
    # Drop unnecessary columns
    drop_cols = [col for col in df_merged.columns 
                 if col.endswith('_method1') or col.endswith('_method2')]
    df_merged = df_merged.drop(columns=drop_cols, errors='ignore')
    
    return df_merged


def compute_source_corrections(df: pd.DataFrame, 
                               config: Dict[str, Any],
                               comparison_mode: str = 'ground_truth') -> pd.DataFrame:
    """
    Compute correction deltas and derived quantities.
    
    Args:
        df: Input dataframe with original and corrected columns
        config: Configuration dictionary
        comparison_mode: 'ground_truth' or 'method_comparison'
        
    Returns:
        Dataframe with added correction columns
    """
    logger.info(f"Computing corrections for {len(df)} events")
    logger.info(f"Comparison mode: {comparison_mode}")

    params_to_process = config['source_analysis'].get('parameters_to_process', ['sx', 'sy', 'sz', 'st0'])
    has_st0 = 'st0' in params_to_process

    # Temporal correction (only when st0 is in parameters_to_process)
    if has_st0:
        try:
            metadata_dir = config['source_analysis'].get('origin_time_metadata_dir',
                                                         config.get('model_dir', ''))

            if metadata_dir:
                origin_time_metadata = load_origin_time_metadata(metadata_dir)
                origin_time_mode = origin_time_metadata['mode']
                logger.info(f"Origin time mode from preprocessing: {origin_time_mode}")

            df['delta_time_s'] = df['corrected_origin_time_s'] - df['origin_time_s']

        except FileNotFoundError as e:
            logger.warning(f"Origin time metadata not found: {e}")
            logger.warning("Falling back to 'zero_initial_origin_time' config flag")

            if config['source_analysis'].get('zero_initial_origin_time', False):
                df['delta_time_s'] = df['corrected_origin_time_s']
            else:
                df['delta_time_s'] = df['corrected_origin_time_s'] - df['origin_time_s']
    else:
        logger.info("Skipping origin time (st0 not in parameters_to_process)")

    # Spatial corrections (in transformed coordinates, assumed to be in meters)
    df['delta_x_km'] = (df['corrected_event_x_transformed'] - df['event_x_transformed']) / 1000
    df['delta_y_km'] = (df['corrected_event_y_transformed'] - df['event_y_transformed']) / 1000
    df['delta_z_km'] = (df['corrected_event_z_transformed'] - df['event_z_transformed']) / 1000

    # Horizontal shift magnitude
    df['delta_horiz_km'] = np.sqrt(df['delta_x_km']**2 + df['delta_y_km']**2)

    # Total 3D shift magnitude
    df['delta_3d_km'] = np.sqrt(
        df['delta_x_km']**2 +
        df['delta_y_km']**2 +
        df['delta_z_km']**2
    )

    # Depth correction (z is depth)
    df['delta_depth_km'] = df['delta_z_km']

    # Remove any infinite or NaN values
    initial_count = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    required_cols = ['delta_x_km', 'delta_y_km', 'delta_z_km',
                     'delta_horiz_km', 'delta_3d_km', 'delta_depth_km']
    if has_st0:
        required_cols.append('delta_time_s')

    df = df.dropna(subset=required_cols)
    
    if len(df) < initial_count:
        logger.warning(f"Removed {initial_count - len(df)} rows with non-finite values")
    
    logger.info(f"Successfully computed corrections for {len(df)} events")
    
    return df


def compute_source_statistics(df: pd.DataFrame, 
                              config: Dict[str, Any],
                              comparison_mode: str = 'ground_truth') -> Dict[str, Any]:
    """
    Compute comprehensive source correction statistics.
    
    Args:
        df: Dataframe with corrections
        config: Configuration dictionary
        comparison_mode: 'ground_truth' or 'method_comparison'
        
    Returns:
        Dictionary of statistics
    """
    stats = {'comparison_mode': comparison_mode}

    params_to_process = config['source_analysis'].get('parameters_to_process', ['sx', 'sy', 'sz', 'st0'])
    has_st0 = 'st0' in params_to_process

    # Overall statistics for each correction type
    correction_columns = {
        'x': 'delta_x_km',
        'y': 'delta_y_km',
        'depth': 'delta_depth_km',
        'horizontal': 'delta_horiz_km',
        '3d': 'delta_3d_km',
    }
    if has_st0:
        correction_columns['time'] = 'delta_time_s'
    
    for name, col in correction_columns.items():
        values = df[col]
        stats[name] = {
            'count': int(values.count()),
            'mean': float(values.mean()),
            'std': float(values.std()),
            'median': float(values.median()),
            'min': float(values.min()),
            'max': float(values.max()),
            'mean_abs': float(values.abs().mean()),
            'max_abs': float(values.abs().max()),
            'p25': float(values.quantile(0.25)),
            'p75': float(values.quantile(0.75)),
        }
    
    # Depth-dependent statistics
    if config['source_analysis'].get('enable_depth_analysis', True):
        stats['depth_bins'] = compute_depth_dependent_stats(df, config)
    
    # Temporal statistics if timestamps available
    if 'origin_time_s' in df.columns:
        stats['temporal'] = {
            'time_span_hours': float((df['origin_time_s'].max() - 
                                    df['origin_time_s'].min()) / 3600),
            'mean_origin_time': float(df['origin_time_s'].mean()),
        }
    
    return stats


def compute_depth_dependent_stats(df: pd.DataFrame, 
                                  config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute statistics binned by depth.
    
    Args:
        df: Dataframe with corrections
        config: Configuration dictionary
        
    Returns:
        Dictionary of depth-binned statistics
    """
    n_bins = config['source_analysis'].get('depth_bins', 10)
    
    df_copy = df.copy()
    df_copy['depth_bin'] = pd.cut(df_copy['event_z_transformed'], bins=n_bins)

    params_to_process = config['source_analysis'].get('parameters_to_process', ['sx', 'sy', 'sz', 'st0'])
    agg_dict = {
        'delta_horiz_km': ['mean', 'std', 'count'],
        'delta_depth_km': ['mean', 'std'],
    }
    if 'st0' in params_to_process:
        agg_dict['delta_time_s'] = ['mean', 'std']

    depth_stats = df_copy.groupby('depth_bin').agg(agg_dict).round(4)
    
    # Filter bins with sufficient data
    min_events = config['source_analysis'].get('min_events_per_bin', 5)
    depth_stats = depth_stats[depth_stats[('delta_horiz_km', 'count')] >= min_events]
    
    return depth_stats.to_dict()


def create_source_plots(df: pd.DataFrame, stats: Dict[str, Any],
                       config: Dict[str, Any],
                       comparison_mode: str = 'ground_truth') -> None:
    """
    Create comprehensive source correction plots.
    
    Args:
        df: Dataframe with corrections
        stats: Statistics dictionary
        config: Configuration dictionary
        comparison_mode: 'ground_truth' or 'method_comparison'
    """
    output_dir = Path(config['source_analysis']['output_dir'])
    plots_dir = output_dir / 'plots'
    ensure_directory_exists(plots_dir)
    
    colors = get_color_scheme(config['analysis'])
    output_format = config['analysis'].get('output_format', 'png')
    use_log = config['source_analysis'].get('use_log_scale', False)

    # Load basin boundaries if configured (global basin_boundaries section)
    basin_config = config.get('basin_boundaries', {})
    basin_data = load_basin_boundaries(basin_config) if config['source_analysis'].get('plot_basin_boundaries', False) else {}

    # Import plotting functions
    from .source_plots import (
        create_correction_histograms,
        create_scatter_relationships,
        create_geographic_map,
        create_depth_analysis,
        create_correction_summary,
        create_temporal_corrections
    )

    # Create all plots
    create_correction_histograms(df, colors, plots_dir, output_format, use_log, config, comparison_mode)
    create_scatter_relationships(df, colors, plots_dir, output_format, use_log, config, comparison_mode)
    create_geographic_map(df, colors, plots_dir, output_format, config, comparison_mode,
                          basin_data=basin_data, basin_config=basin_config)
    create_depth_analysis(df, colors, plots_dir, output_format, use_log, config, comparison_mode)
    create_correction_summary(df, colors, plots_dir, output_format, use_log, config, comparison_mode)
    
    # Temporal plots only when st0 was processed and original origin times are present
    _params = config['source_analysis'].get('parameters_to_process', ['sx', 'sy', 'sz', 'st0'])
    if 'origin_time_s' in df.columns and 'st0' in _params:
        create_temporal_corrections(df, colors, plots_dir, output_format, use_log, config, comparison_mode)


def generate_source_report(df: pd.DataFrame, stats: Dict[str, Any],
                          config: Dict[str, Any],
                          comparison_mode: str = 'ground_truth') -> None:
    """
    Generate comprehensive source correction analysis report.
    
    Args:
        df: Dataframe with corrections
        stats: Statistics dictionary
        config: Configuration dictionary
        comparison_mode: 'ground_truth' or 'method_comparison'
    """
    output_dir = Path(config['source_analysis']['output_dir'])
    reports_dir = output_dir / 'reports'
    ensure_directory_exists(reports_dir)
    
    report_file = reports_dir / 'source_correction_report.txt'
    
    scale_type = "Log Scale" if config['source_analysis'].get('use_log_scale', False) else "Linear Scale"
    
    # Get titles based on comparison mode
    if comparison_mode == 'method_comparison':
        method1_name = config['source_analysis'].get('method1_name', 'Method 1')
        method2_name = config['source_analysis'].get('method2_name', 'Method 2')
        title = f"SOURCE PARAMETER METHOD COMPARISON REPORT\n{method1_name} vs {method2_name}"
        mode_description = f"Comparing: {method2_name} - {method1_name}"
    else:
        title = "SOURCE CORRECTION ANALYSIS REPORT"
        mode_description = "Comparing: Corrected vs Original"
    
    report_lines = [
        title,
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Comparison mode: {comparison_mode}",
        mode_description,
        f"Plot scale: {scale_type}",
        "",
    ]
    
    if comparison_mode == 'method_comparison':
        report_lines.extend([
            f"Method 1 file: {config['source_analysis']['method1_csv']}",
            f"Method 2 file: {config['source_analysis']['method2_csv']}",
            "",
        ])
    else:
        report_lines.extend([
            f"Input file: {config['source_analysis']['input_csv']}",
            "",
        ])
    
    event_count = stats.get('time', stats.get('horizontal', {})).get('count', 0)
    report_lines.extend([
        "SUMMARY",
        "-" * 60,
        f"Total events analyzed: {event_count}",
        "",
    ])

    # Temporal corrections/differences (only when st0 was processed)
    if 'time' in stats:
        if comparison_mode == 'method_comparison':
            report_lines.extend([
                "TEMPORAL DIFFERENCES",
                "-" * 60,
                f"Mean time difference: {stats['time']['mean']:.4f} s",
                f"Std time difference: {stats['time']['std']:.4f} s",
                f"Median time difference: {stats['time']['median']:.4f} s",
                f"Mean absolute time difference: {stats['time']['mean_abs']:.4f} s",
                f"Max absolute time difference: {stats['time']['max_abs']:.4f} s",
                f"Range: [{stats['time']['min']:.4f}, {stats['time']['max']:.4f}] s",
                "",
            ])
        else:
            report_lines.extend([
                "TEMPORAL CORRECTIONS",
                "-" * 60,
                f"Mean time shift: {stats['time']['mean']:.4f} s",
                f"Std time shift: {stats['time']['std']:.4f} s",
                f"Median time shift: {stats['time']['median']:.4f} s",
                f"Mean absolute time shift: {stats['time']['mean_abs']:.4f} s",
                f"Max absolute time shift: {stats['time']['max_abs']:.4f} s",
                f"Range: [{stats['time']['min']:.4f}, {stats['time']['max']:.4f}] s",
                "",
            ])
    
    # Horizontal corrections/differences
    label_type = "difference" if comparison_mode == 'method_comparison' else "shift"
    report_lines.extend([
        f"HORIZONTAL {label_type.upper()}S",
        "-" * 60,
        f"Mean horizontal {label_type}: {stats['horizontal']['mean']:.4f} km",
        f"Std horizontal {label_type}: {stats['horizontal']['std']:.4f} km",
        f"Median horizontal {label_type}: {stats['horizontal']['median']:.4f} km",
        f"Mean absolute horizontal {label_type}: {stats['horizontal']['mean_abs']:.4f} km",
        f"Max horizontal {label_type}: {stats['horizontal']['max']:.4f} km",
        "",
    ])
    
    # Depth corrections/differences
    report_lines.extend([
        f"DEPTH {label_type.upper()}S",
        "-" * 60,
        f"Mean depth {label_type}: {stats['depth']['mean']:.4f} km",
        f"Std depth {label_type}: {stats['depth']['std']:.4f} km",
        f"Median depth {label_type}: {stats['depth']['median']:.4f} km",
        f"Mean absolute depth {label_type}: {stats['depth']['mean_abs']:.4f} km",
        f"Max absolute depth {label_type}: {stats['depth']['max_abs']:.4f} km",
        f"Range: [{stats['depth']['min']:.4f}, {stats['depth']['max']:.4f}] km",
        "",
    ])
    
    # 3D corrections/differences
    report_lines.extend([
        f"TOTAL 3D {label_type.upper()}S",
        "-" * 60,
        f"Mean 3D {label_type}: {stats['3d']['mean']:.4f} km",
        f"Std 3D {label_type}: {stats['3d']['std']:.4f} km",
        f"Median 3D {label_type}: {stats['3d']['median']:.4f} km",
        f"Max 3D {label_type}: {stats['3d']['max']:.4f} km",
        "",
    ])
    
    # X and Y components
    report_lines.extend([
        f"COORDINATE-WISE {label_type.upper()}S",
        "-" * 60,
        "X-direction:",
        f"  Mean: {stats['x']['mean']:.4f} km",
        f"  Std: {stats['x']['std']:.4f} km",
        f"  Mean absolute: {stats['x']['mean_abs']:.4f} km",
        "",
        "Y-direction:",
        f"  Mean: {stats['y']['mean']:.4f} km",
        f"  Std: {stats['y']['std']:.4f} km",
        f"  Mean absolute: {stats['y']['mean_abs']:.4f} km",
        "",
    ])
    
    # Temporal info if available
    if 'temporal' in stats:
        report_lines.extend([
            "TEMPORAL COVERAGE",
            "-" * 60,
            f"Time span: {stats['temporal']['time_span_hours']:.2f} hours",
            "",
        ])
    
    # Percentile information
    report_lines.extend([
        "PERCENTILE ANALYSIS",
        "-" * 60,
        f"Horizontal {label_type}s:",
        f"  25th percentile: {stats['horizontal']['p25']:.4f} km",
        f"  50th percentile: {stats['horizontal']['median']:.4f} km",
        f"  75th percentile: {stats['horizontal']['p75']:.4f} km",
        "",
        f"Depth {label_type}s:",
        f"  25th percentile: {stats['depth']['p25']:.4f} km",
        f"  50th percentile: {stats['depth']['median']:.4f} km",
        f"  75th percentile: {stats['depth']['p75']:.4f} km",
        "",
    ])
    
    # Write report
    write_text_file(report_file, report_lines)
    logger.info(f"Analysis report saved to: {report_file}")