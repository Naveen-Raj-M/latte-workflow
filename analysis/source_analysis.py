"""
Source correction analysis module.

This module provides comprehensive analysis of hypocenter corrections
including spatial, temporal, and depth corrections.
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

logger = logging.getLogger(__name__)


def analyze_source_corrections(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function for source correction analysis.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    logger.info("Starting source correction analysis")
    
    # Setup plot style
    setup_plot_style(config['analysis'])
    
    # Load data
    input_csv = config['source_analysis']['input_csv']
    output_dir = Path(config['source_analysis']['output_dir'])
    
    logger.info(f"Loading source correction data from: {input_csv}")
    df = read_csv_file(input_csv)
    
    # Validate and compute corrections
    df_corrected = compute_source_corrections(df, config)
    
    # Compute statistics
    logger.info("Computing source correction statistics")
    stats = compute_source_statistics(df_corrected, config)
    
    # Create plots
    logger.info("Creating source correction plots")
    create_source_plots(df_corrected, stats, config)
    
    # Generate summary report
    if config['source_analysis'].get('generate_summary_report', True):
        logger.info("Generating summary report")
        generate_source_report(df_corrected, stats, config)
    
    # Save corrected data with deltas
    if config['source_analysis'].get('save_corrected_data', True):
        output_csv = output_dir / 'source_corrections_with_deltas.csv'
        df_corrected.to_csv(output_csv, index=False)
        logger.info(f"Saved corrected data to: {output_csv}")
    
    logger.info("Source correction analysis completed")
    return stats


def compute_source_corrections(df: pd.DataFrame, 
                               config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute correction deltas and derived quantities.
    
    Args:
        df: Input dataframe with raw and corrected columns
        config: Configuration dictionary
        
    Returns:
        Dataframe with added correction columns
    """
    logger.info(f"Computing corrections for {len(df)} events")
    
    df = df.copy()
        
    # Temporal correction
    if config['source_analysis'].get('zero_initial_origin_time', False):
        df['delta_time_s'] = df['corrected_origin_time_s']

    else:
        df['delta_time_s'] = df['origin_time_s'] - df['corrected_origin_time_s'] 
    
    # Spatial corrections (in transformed coordinates, assumed to be in km)
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
    
    # Check for required columns
    required_cols = [
        'delta_time_s', 'delta_x_km', 'delta_y_km', 'delta_z_km',
        'delta_horiz_km', 'delta_3d_km', 'delta_depth_km'
    ]
    
    df = df.dropna(subset=required_cols)
    
    if len(df) < initial_count:
        logger.warning(f"Removed {initial_count - len(df)} rows with non-finite values")
    
    logger.info(f"Successfully computed corrections for {len(df)} events")
    
    return df


def compute_source_statistics(df: pd.DataFrame, 
                              config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute comprehensive source correction statistics.
    
    Args:
        df: Dataframe with corrections
        config: Configuration dictionary
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    # Overall statistics for each correction type
    correction_columns = {
        'time': 'delta_time_s',
        'x': 'delta_x_km',
        'y': 'delta_y_km',
        'depth': 'delta_depth_km',
        'horizontal': 'delta_horiz_km',
        '3d': 'delta_3d_km'
    }
    
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
    
    depth_stats = df_copy.groupby('depth_bin').agg({
        'delta_horiz_km': ['mean', 'std', 'count'],
        'delta_depth_km': ['mean', 'std'],
        'delta_time_s': ['mean', 'std']
    }).round(4)
    
    # Filter bins with sufficient data
    min_events = config['source_analysis'].get('min_events_per_bin', 5)
    depth_stats = depth_stats[depth_stats[('delta_horiz_km', 'count')] >= min_events]
    
    return depth_stats.to_dict()


def create_source_plots(df: pd.DataFrame, stats: Dict[str, Any],
                       config: Dict[str, Any]) -> None:
    """
    Create comprehensive source correction plots.
    
    Args:
        df: Dataframe with corrections
        stats: Statistics dictionary
        config: Configuration dictionary
    """
    output_dir = Path(config['source_analysis']['output_dir'])
    plots_dir = output_dir / 'plots'
    ensure_directory_exists(plots_dir)
    
    colors = get_color_scheme(config['analysis'])
    output_format = config['analysis'].get('output_format', 'png')
    use_log = config['source_analysis'].get('use_log_scale', False)
    
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
    create_correction_histograms(df, colors, plots_dir, output_format, use_log, config)
    create_scatter_relationships(df, colors, plots_dir, output_format, use_log, config)
    create_geographic_map(df, colors, plots_dir, output_format, config)
    create_depth_analysis(df, colors, plots_dir, output_format, use_log, config)
    create_correction_summary(df, colors, plots_dir, output_format, use_log, config)
    
    # Temporal plots if time data available
    if 'origin_time_s' in df.columns:
        create_temporal_corrections(df, colors, plots_dir, output_format, use_log, config)


def generate_source_report(df: pd.DataFrame, stats: Dict[str, Any],
                          config: Dict[str, Any]) -> None:
    """
    Generate comprehensive source correction analysis report.
    
    Args:
        df: Dataframe with corrections
        stats: Statistics dictionary
        config: Configuration dictionary
    """
    output_dir = Path(config['source_analysis']['output_dir'])
    reports_dir = output_dir / 'reports'
    ensure_directory_exists(reports_dir)
    
    report_file = reports_dir / 'source_correction_report.txt'
    
    scale_type = "Log Scale" if config['source_analysis'].get('use_log_scale', False) else "Linear Scale"
    
    report_lines = [
        "SOURCE CORRECTION ANALYSIS REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Input file: {config['source_analysis']['input_csv']}",
        f"Plot scale: {scale_type}",
        "",
        "SUMMARY",
        "-" * 60,
        f"Total events analyzed: {stats['time']['count']}",
        "",
    ]
    
    # Temporal corrections
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
    
    # Horizontal corrections
    report_lines.extend([
        "HORIZONTAL CORRECTIONS",
        "-" * 60,
        f"Mean horizontal shift: {stats['horizontal']['mean']:.4f} km",
        f"Std horizontal shift: {stats['horizontal']['std']:.4f} km",
        f"Median horizontal shift: {stats['horizontal']['median']:.4f} km",
        f"Mean absolute horizontal shift: {stats['horizontal']['mean_abs']:.4f} km",
        f"Max horizontal shift: {stats['horizontal']['max']:.4f} km",
        "",
    ])
    
    # Depth corrections
    report_lines.extend([
        "DEPTH CORRECTIONS",
        "-" * 60,
        f"Mean depth shift: {stats['depth']['mean']:.4f} km",
        f"Std depth shift: {stats['depth']['std']:.4f} km",
        f"Median depth shift: {stats['depth']['median']:.4f} km",
        f"Mean absolute depth shift: {stats['depth']['mean_abs']:.4f} km",
        f"Max absolute depth shift: {stats['depth']['max_abs']:.4f} km",
        f"Range: [{stats['depth']['min']:.4f}, {stats['depth']['max']:.4f}] km",
        "",
    ])
    
    # 3D corrections
    report_lines.extend([
        "TOTAL 3D CORRECTIONS",
        "-" * 60,
        f"Mean 3D shift: {stats['3d']['mean']:.4f} km",
        f"Std 3D shift: {stats['3d']['std']:.4f} km",
        f"Median 3D shift: {stats['3d']['median']:.4f} km",
        f"Max 3D shift: {stats['3d']['max']:.4f} km",
        "",
    ])
    
    # X and Y components
    report_lines.extend([
        "COORDINATE-WISE CORRECTIONS",
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
        "Horizontal shifts:",
        f"  25th percentile: {stats['horizontal']['p25']:.4f} km",
        f"  50th percentile: {stats['horizontal']['median']:.4f} km",
        f"  75th percentile: {stats['horizontal']['p75']:.4f} km",
        "",
        "Depth shifts:",
        f"  25th percentile: {stats['depth']['p25']:.4f} km",
        f"  50th percentile: {stats['depth']['median']:.4f} km",
        f"  75th percentile: {stats['depth']['p75']:.4f} km",
        "",
    ])
    
    # Write report
    write_text_file(report_file, report_lines)
    logger.info(f"Analysis report saved to: {report_file}")