"""
Travel time residual analysis module.

This module provides comprehensive analysis of travel time residuals
including basic statistics, ratio analysis, and comparison plots.
Supports both ground truth comparison and method-to-method comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
from datetime import datetime

from utils.io_utils import read_csv_file, write_text_file, ensure_directory_exists
from utils.plot_utils import setup_plot_style, get_color_scheme

from .traveltime_plots import create_travel_time_plots
from .traveltime_statistics import compute_travel_time_statistics
from .traveltime_comparison import prepare_comparison_data
from .filters import apply_analysis_filters

logger = logging.getLogger(__name__)


def analyze_travel_time_residuals(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function for travel time residual analysis.
    
    Supports two modes:
    1. Ground truth comparison: Compare synthetic travel times with observed
    2. Method comparison: Compare synthetic travel times from two methods
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    logger.info("Starting travel time residual analysis")
    
    # Setup plot style
    setup_plot_style(config['analysis'])
    
    # Determine comparison mode
    comparison_mode = config['travel_time_analysis'].get('comparison_mode', 'ground_truth')
    
    if comparison_mode == 'method_comparison':
        logger.info("Running in METHOD COMPARISON mode")
        results = _analyze_method_comparison(config)
    else:
        logger.info("Running in GROUND TRUTH COMPARISON mode")
        results = _analyze_ground_truth_comparison(config)
    
    logger.info("Travel time residual analysis completed")
    return results


def _analyze_ground_truth_comparison(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze travel time residuals against ground truth (observed data).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    input_csv = config['travel_time_analysis']['input_csv']
    output_dir = Path(config['travel_time_analysis']['output_dir'])
    
    logger.info(f"Loading travel time data from: {input_csv}")
    df = read_csv_file(input_csv)

    # Apply analysis filters (depth and phase type)
    df = apply_analysis_filters(df, config, data_type="travel time data")

    # Validate and clean data
    df_clean = validate_and_clean_travel_time_data(df, config)
    
    # Compute statistics
    logger.info("Computing travel time statistics")
    stats = compute_travel_time_statistics(df_clean, config, comparison_mode='ground_truth')
    
    # Create plots
    logger.info("Creating travel time plots")
    create_travel_time_plots(df_clean, stats, config, comparison_mode='ground_truth')
    
    # Generate summary report
    if config['travel_time_analysis'].get('generate_summary_report', True):
        logger.info("Generating summary report")
        generate_travel_time_report(df_clean, stats, config, comparison_mode='ground_truth')
    
    return stats


def _analyze_method_comparison(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze travel time comparison between two methods.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    method1_csv = config['travel_time_analysis']['method1_csv']
    method2_csv = config['travel_time_analysis']['method2_csv']
    output_dir = Path(config['travel_time_analysis']['output_dir'])
    
    logger.info(f"Loading Method 1 data from: {method1_csv}")
    logger.info(f"Loading Method 2 data from: {method2_csv}")

    df1 = read_csv_file(method1_csv)
    df2 = read_csv_file(method2_csv)

    # Apply analysis filters to both datasets
    df1 = apply_analysis_filters(df1, config, data_type="Method 1 travel time data")
    df2 = apply_analysis_filters(df2, config, data_type="Method 2 travel time data")

    # Prepare comparison data
    logger.info("Preparing method comparison data")
    df_comparison = prepare_comparison_data(df1, df2, config)
    
    # Compute statistics
    logger.info("Computing method comparison statistics")
    stats = compute_travel_time_statistics(df_comparison, config, comparison_mode='method_comparison')
    
    # Create plots
    logger.info("Creating method comparison plots")
    create_travel_time_plots(df_comparison, stats, config, comparison_mode='method_comparison')
    
    # Generate summary report
    if config['travel_time_analysis'].get('generate_summary_report', True):
        logger.info("Generating summary report")
        generate_travel_time_report(df_comparison, stats, config, comparison_mode='method_comparison')
    
    return stats


def validate_and_clean_travel_time_data(df: pd.DataFrame, 
                                       config: Dict[str, Any]) -> pd.DataFrame:
    """
    Validate and clean travel time data.
    
    Args:
        df: Input dataframe
        config: Configuration dictionary
        
    Returns:
        Cleaned dataframe
    """
    logger.info(f"Validating travel time data: {len(df)} initial records")
    
    # Remove infinite and NaN values
    initial_count = len(df)
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['observed_tt', 'synthetic_tt', 'residual_s']
    )
    
    if len(df_clean) < initial_count:
        logger.warning(f"Removed {initial_count - len(df_clean)} rows with non-finite values")
    
    # Apply residual thresholds
    min_res = config['travel_time_analysis'].get('min_residual_threshold', -10.0)
    max_res = config['travel_time_analysis'].get('max_residual_threshold', 10.0)
    
    mask = (df_clean['residual_s'] >= min_res) & (df_clean['residual_s'] <= max_res)
    df_clean = df_clean[mask]
    
    logger.info(f"Final clean dataset: {len(df_clean)} records")
    return df_clean


def generate_travel_time_report(df: pd.DataFrame, stats: Dict[str, Any], 
                              config: Dict[str, Any],
                              comparison_mode: str = 'ground_truth') -> None:
    """
    Generate comprehensive travel time analysis report.
    
    Args:
        df: Travel time dataframe
        stats: Statistics dictionary
        config: Configuration dictionary
        comparison_mode: 'ground_truth' or 'method_comparison'
    """
    output_dir = Path(config['travel_time_analysis']['output_dir'])
    reports_dir = output_dir / 'reports'
    ensure_directory_exists(reports_dir)
    
    report_file = reports_dir / 'travel_time_analysis_report.txt'
    
    # Get custom titles if in method comparison mode
    if comparison_mode == 'method_comparison':
        method1_name = config['travel_time_analysis'].get('method1_name', 'Method 1')
        method2_name = config['travel_time_analysis'].get('method2_name', 'Method 2')
        title = f"TRAVEL TIME METHOD COMPARISON REPORT\n{method1_name} vs {method2_name}"
        mode_description = f"Comparing: {method1_name} vs {method2_name}"
    else:
        title = "TRAVEL TIME RESIDUAL ANALYSIS REPORT"
        mode_description = "Comparing: Synthetic vs Observed (Ground Truth)"
    
    report_lines = [
        title,
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Comparison mode: {comparison_mode}",
        mode_description,
        "",
    ]
    
    if comparison_mode == 'method_comparison':
        report_lines.extend([
            f"Method 1 file: {config['travel_time_analysis']['method1_csv']}",
            f"Method 2 file: {config['travel_time_analysis']['method2_csv']}",
            "",
        ])
    else:
        report_lines.extend([
            f"Input file: {config['travel_time_analysis']['input_csv']}",
            "",
        ])
    
    report_lines.extend([
        "SUMMARY",
        "-" * 60,
        f"Total data points: {stats['overall']['count']}",
        f"Mean residual: {stats['overall']['mean_residual']:.4f} s",
        f"RMS residual: {stats['overall']['rms_residual']:.4f} s",
        f"Standard deviation: {stats['overall']['std_residual']:.4f} s",
        f"Mean absolute error: {stats['overall']['mae_residual']:.4f} s",
        "",
    ])
    
    # Phase-specific statistics
    for phase in ['P', 'S']:
        phase_key = f'{phase.lower()}_wave'
        if phase_key in stats:
            phase_stats = stats[phase_key]
            report_lines.extend([
                f"{phase}-WAVE STATISTICS",
                "-" * 60,
                f"Data points: {phase_stats['count']}",
                f"Mean residual: {phase_stats['mean_residual']:.4f} s",
                f"RMS residual: {phase_stats['rms_residual']:.4f} s",
                f"Mean absolute error: {phase_stats['mae_residual']:.4f} s",
                f"Travel time correlation: {phase_stats['correlation']:.4f}",
                "",
            ])
            
            # Add ln ratio statistics if available
            if 'ln_ratio' in phase_stats:
                ln_stats = phase_stats['ln_ratio']
                report_lines.extend([
                    f"Logarithmic ratio analysis:",
                    f"  Mean ln(ratio): {ln_stats['mean']:.4f}",
                    f"  Std ln(ratio): {ln_stats['std']:.4f}",
                    f"  Equivalent % change: {ln_stats['equivalent_percent_change']:.2f}%",
                    "",
                ])
    
    # Station and event summaries
    report_lines.extend([
        "STATION SUMMARY",
        "-" * 60,
        f"Unique stations: {stats['station_summary']['unique_stations']}",
        "",
        "EVENT SUMMARY", 
        "-" * 60,
        f"Unique events: {stats['event_summary']['unique_events']}",
        "",
    ])
    
    # Write report
    write_text_file(report_file, report_lines)
    logger.info(f"Analysis report saved to: {report_file}")