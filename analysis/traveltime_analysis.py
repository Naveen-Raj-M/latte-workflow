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
    df_clean, filter_summary = validate_and_clean_travel_time_data(df, config)

    # Compute statistics
    logger.info("Computing travel time statistics")
    stats = compute_travel_time_statistics(df_clean, config, comparison_mode='ground_truth')

    # Create plots
    logger.info("Creating travel time plots")
    create_travel_time_plots(df_clean, stats, config, comparison_mode='ground_truth')

    # Generate summary report
    if config['travel_time_analysis'].get('generate_summary_report', True):
        logger.info("Generating summary report")
        generate_travel_time_report(df_clean, stats, config,
                                    comparison_mode='ground_truth',
                                    filter_summary=filter_summary)

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

    # Validate and clean data (identical pipeline to ground truth mode)
    df_comparison, filter_summary = validate_and_clean_travel_time_data(df_comparison, config)

    # Compute statistics
    logger.info("Computing method comparison statistics")
    stats = compute_travel_time_statistics(df_comparison, config, comparison_mode='method_comparison')

    # Create plots
    logger.info("Creating method comparison plots")
    create_travel_time_plots(df_comparison, stats, config, comparison_mode='method_comparison')

    # Generate summary report
    if config['travel_time_analysis'].get('generate_summary_report', True):
        logger.info("Generating summary report")
        generate_travel_time_report(df_comparison, stats, config,
                                    comparison_mode='method_comparison',
                                    filter_summary=filter_summary)

    return stats


def validate_and_clean_travel_time_data(df: pd.DataFrame,
                                       config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate and clean travel time data.

    Applies three sequential filters:
      1. Remove rows with non-finite (NaN / ±inf) values in observed_tt,
         synthetic_tt, or residual_s.
      2. Remove rows where either observed_tt or synthetic_tt is negative
         (physically meaningless travel times).
      3. Remove rows whose residual falls outside the configured threshold
         window [min_residual_threshold, max_residual_threshold].

    Args:
        df: Input dataframe
        config: Configuration dictionary

    Returns:
        Tuple of (cleaned dataframe, filter_summary dict)
    """
    initial_count = len(df)
    logger.info(f"Validating travel time data: {initial_count} initial records")

    # --- Step 1: Remove non-finite values ---
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['observed_tt', 'synthetic_tt', 'residual_s']
    )
    non_finite_removed = initial_count - len(df_clean)
    if non_finite_removed > 0:
        logger.warning(
            f"Removed {non_finite_removed} rows with non-finite values "
            f"(NaN or ±inf) in observed_tt / synthetic_tt / residual_s"
        )

    # --- Step 2: Remove negative travel times ---
    neg_obs  = int((df_clean['observed_tt']  < 0).sum())
    neg_syn  = int((df_clean['synthetic_tt'] < 0).sum())
    neg_mask = (df_clean['observed_tt'] < 0) | (df_clean['synthetic_tt'] < 0)
    neg_either = int(neg_mask.sum())
    if neg_either > 0:
        logger.warning(
            f"Removing {neg_either} rows with negative travel times "
            f"(observed_tt < 0: {neg_obs}, synthetic_tt < 0: {neg_syn})"
        )
        df_clean = df_clean[~neg_mask]

    # --- Step 3: Apply residual thresholds ---
    min_res = config['travel_time_analysis'].get('min_residual_threshold', -50.0)
    max_res = config['travel_time_analysis'].get('max_residual_threshold',  50.0)
    before_thresh = len(df_clean)
    res_mask = (df_clean['residual_s'] >= min_res) & (df_clean['residual_s'] <= max_res)
    df_clean = df_clean[res_mask]
    residual_threshold_removed = before_thresh - len(df_clean)
    if residual_threshold_removed > 0:
        logger.warning(
            f"Removed {residual_threshold_removed} rows with residuals outside "
            f"[{min_res}, {max_res}] s"
        )

    logger.info(f"Final clean dataset: {len(df_clean)} records "
                f"({initial_count - len(df_clean)} total removed)")

    filter_summary = {
        'initial_count':               initial_count,
        'non_finite_removed':          non_finite_removed,
        'negative_tt_removed':         neg_either,
        'negative_observed_tt_count':  neg_obs,
        'negative_synthetic_tt_count': neg_syn,
        'residual_threshold_removed':  residual_threshold_removed,
        'min_residual_threshold':      min_res,
        'max_residual_threshold':      max_res,
        'final_count':                 len(df_clean),
    }
    return df_clean, filter_summary


def generate_travel_time_report(df: pd.DataFrame, stats: Dict[str, Any],
                              config: Dict[str, Any],
                              comparison_mode: str = 'ground_truth',
                              filter_summary: Optional[Dict[str, Any]] = None) -> None:
    """
    Generate comprehensive travel time analysis report.

    Args:
        df: Travel time dataframe (after all cleaning)
        stats: Statistics dictionary
        config: Configuration dictionary
        comparison_mode: 'ground_truth' or 'method_comparison'
        filter_summary: Dict returned by validate_and_clean_travel_time_data,
                        documents how many records were removed at each stage.
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
    
    # Data filtering section
    if filter_summary is not None:
        fs = filter_summary
        total_removed = fs['initial_count'] - fs['final_count']
        report_lines.extend([
            "DATA FILTERING",
            "-" * 60,
            f"Initial records:                    {fs['initial_count']}",
            f"  Removed (non-finite values):      {fs['non_finite_removed']}",
            f"  Removed (negative travel times):  {fs['negative_tt_removed']}",
        ])
        if fs['negative_tt_removed'] > 0:
            report_lines.extend([
                f"    - negative observed_tt:         {fs['negative_observed_tt_count']}",
                f"    - negative synthetic_tt:        {fs['negative_synthetic_tt_count']}",
            ])
        report_lines.extend([
            f"  Removed (residual threshold):     {fs['residual_threshold_removed']}",
            f"    Threshold window:               [{fs['min_residual_threshold']}, "
            f"{fs['max_residual_threshold']}] s",
            f"Total records removed:              {total_removed}",
            f"Final records used:                 {fs['final_count']}",
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