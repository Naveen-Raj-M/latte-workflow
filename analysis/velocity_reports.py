"""
Velocity model comparison report generation module.

This module generates comprehensive text reports for velocity model comparisons.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from utils.io_utils import write_text_file, ensure_directory_exists

logger = logging.getLogger(__name__)


def generate_velocity_comparison_report(df1: pd.DataFrame, df2: pd.DataFrame,
                                       differences: Dict[str, np.ndarray],
                                       stats: Dict[str, Any],
                                       extremes: Dict[str, Any],
                                       config: Dict[str, Any]) -> None:
    """
    Generate comprehensive velocity comparison report.
    
    Args:
        df1: First velocity model dataframe
        df2: Second velocity model dataframe
        differences: Differences dictionary
        stats: Statistics dictionary
        extremes: Extreme profiles dictionary
        config: Configuration dictionary
    """
    output_dir = Path(config['velocity_analysis']['output_dir'])
    reports_dir = output_dir / 'reports'
    ensure_directory_exists(reports_dir)
    
    report_file = reports_dir / 'velocity_comparison_report.txt'
    
    # Get model names
    comparison_mode = config['velocity_analysis']['comparison_mode']
    if comparison_mode == "initial_vs_final":
        model1_name, model2_name = "Initial Model", "Final Model"
    elif comparison_mode == "method_comparison":
        model1_name, model2_name = "Method 1", "Method 2"
    elif comparison_mode == "custom":
        model1_name = config['velocity_analysis']['custom_title1']
        model2_name = config['velocity_analysis']['custom_title2']
    else:
        model1_name = config['velocity_analysis']['model1_name']
        model2_name = config['velocity_analysis']['model2_name']
    
    report_lines = []
    
    # Header
    report_lines.extend([
        "=" * 80,
        "VELOCITY MODEL COMPARISON ANALYSIS REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Comparison Mode: {comparison_mode}",
        f"Model 1: {model1_name}",
        f"Model 2: {model2_name}",
        "",
    ])
    
    # Summary
    total_points = len(differences['vp_diff'])
    report_lines.extend([
        "SUMMARY",
        "-" * 40,
        f"Total comparison points: {total_points}",
        f"Vp correlation: {stats['vp_comparison']['correlation']:.4f}",
        f"Vs correlation: {stats['vs_comparison']['correlation']:.4f}",
        f"Vp/Vs ratio correlation: {stats['ratio_comparison']['correlation']:.4f}",
        ""
    ])
    
    # Vp Statistics
    report_lines.extend(_format_velocity_statistics('VP', stats['vp_comparison'], 
                                                    model1_name, model2_name, total_points))
    
    # Vs Statistics
    report_lines.extend(_format_velocity_statistics('VS', stats['vs_comparison'], 
                                                    model1_name, model2_name, total_points))
    
    # Log ratio statistics if available
    if 'log_ratios' in stats:
        report_lines.extend(_format_log_ratio_statistics('VP', stats['log_ratios']['vp']))
        report_lines.extend(_format_log_ratio_statistics('VS', stats['log_ratios']['vs']))
    
    # Ratio statistics
    report_lines.extend(_format_ratio_statistics(stats['ratio_comparison'], 
                                                 model1_name, model2_name))
    
    # Extreme locations
    report_lines.extend(_format_extreme_locations(extremes))
    
    # Configuration summary
    report_lines.extend(_format_configuration_summary(config))
    
    # Write report
    write_text_file(report_file, report_lines)
    logger.info(f"Velocity comparison report saved to: {report_file}")


def _format_velocity_statistics(vel_type: str, vel_stats: Dict[str, Any],
                                model1_name: str, model2_name: str,
                                total_points: int) -> list:
    """
    Format velocity statistics section.
    
    Args:
        vel_type: Type of velocity (VP or VS)
        vel_stats: Velocity statistics dictionary
        model1_name: First model name
        model2_name: Second model name
        total_points: Total number of comparison points
        
    Returns:
        List of formatted lines
    """
    lines = [
        f"{vel_type} VELOCITY COMPARISON",
        "-" * 40,
        f"{model1_name}:",
        f"  Mean: {vel_stats['model1_mean']:.3f} ± {vel_stats['model1_std']:.3f} km/s",
        f"  Range: {vel_stats['model1_min']:.3f} - {vel_stats['model1_max']:.3f} km/s",
        "",
        f"{model2_name}:",
        f"  Mean: {vel_stats['model2_mean']:.3f} ± {vel_stats['model2_std']:.3f} km/s",
        f"  Range: {vel_stats['model2_min']:.3f} - {vel_stats['model2_max']:.3f} km/s",
        "",
        f"Differences ({model2_name} - {model1_name}):",
        f"  Mean: {vel_stats['diff_mean']:.3f} ± {vel_stats['diff_std']:.3f} km/s",
        f"  Range: {vel_stats['diff_min']:.3f} to {vel_stats['diff_max']:.3f} km/s",
        f"  Mean absolute: {vel_stats['diff_abs_mean']:.3f} km/s",
        f"  RMS: {vel_stats['diff_rms']:.3f} km/s",
        f"  Mean percent change: {vel_stats['percent_mean']:.2f}% ± {vel_stats['percent_std']:.2f}%",
        f"  Points increased: {vel_stats['points_increased']} ({100*vel_stats['points_increased']/total_points:.1f}%)",
        f"  Points decreased: {vel_stats['points_decreased']} ({100*vel_stats['points_decreased']/total_points:.1f}%)",
        f"  Points unchanged: {vel_stats['points_unchanged']} ({100*vel_stats['points_unchanged']/total_points:.1f}%)",
        f"  Correlation: {vel_stats['correlation']:.4f}",
        ""
    ]
    return lines


def _format_log_ratio_statistics(vel_type: str, log_stats: Dict[str, Any]) -> list:
    """
    Format logarithmic ratio statistics section.
    
    Args:
        vel_type: Type of velocity (VP or VS)
        log_stats: Log ratio statistics dictionary
        
    Returns:
        List of formatted lines
    """
    lines = [
        f"{vel_type} LOGARITHMIC RATIO ANALYSIS",
        "-" * 40,
        f"  Mean ln(ratio): {log_stats['mean']:.4f} ± {log_stats['std']:.4f}",
        f"  Range: {log_stats['min']:.4f} to {log_stats['max']:.4f}",
        f"  Mean absolute ln(ratio): {log_stats['abs_mean']:.4f}",
        f"  RMS ln(ratio): {log_stats['rms']:.4f}",
        f"  Equivalent % change: {log_stats['equivalent_percent_change']:.2f}%",
        f"  Points increased: {log_stats['points_increased']}",
        f"  Points decreased: {log_stats['points_decreased']}",
        ""
    ]
    return lines


def _format_ratio_statistics(ratio_stats: Dict[str, Any],
                             model1_name: str, model2_name: str) -> list:
    """
    Format Vp/Vs ratio statistics section.
    
    Args:
        ratio_stats: Ratio statistics dictionary
        model1_name: First model name
        model2_name: Second model name
        
    Returns:
        List of formatted lines
    """
    lines = [
        "VP/VS RATIO COMPARISON",
        "-" * 40,
        f"{model1_name} ratio:",
        f"  Mean: {ratio_stats['model1_mean']:.3f} ± {ratio_stats['model1_std']:.3f}",
        f"  Range: {ratio_stats['model1_min']:.3f} - {ratio_stats['model1_max']:.3f}",
        "",
        f"{model2_name} ratio:",
        f"  Mean: {ratio_stats['model2_mean']:.3f} ± {ratio_stats['model2_std']:.3f}",
        f"  Range: {ratio_stats['model2_min']:.3f} - {ratio_stats['model2_max']:.3f}",
        "",
        f"Ratio difference:",
        f"  Mean: {ratio_stats['diff_mean']:.3f} ± {ratio_stats['diff_std']:.3f}",
        f"  Range: {ratio_stats['diff_min']:.3f} to {ratio_stats['diff_max']:.3f}",
        f"  Correlation: {ratio_stats['correlation']:.4f}",
        ""
    ]
    return lines


def _format_extreme_locations(extremes: Dict[str, Any]) -> list:
    """
    Format extreme difference locations section.
    
    Args:
        extremes: Extreme profiles dictionary
        
    Returns:
        List of formatted lines
    """
    max_profile_stats = extremes['profile_stats'].loc[extremes['max_profile_coords']]
    min_profile_stats = extremes['profile_stats'].loc[extremes['min_profile_coords']]
    
    lines = [
        "EXTREME DIFFERENCE LOCATIONS",
        "-" * 40,
        "Maximum difference vertical profile:",
        f"  Location (X,Y): ({extremes['max_profile_coords'][0]:.1f}, {extremes['max_profile_coords'][1]:.1f}) km",
        f"  Depth range: {max_profile_stats['z_min']:.1f} to {max_profile_stats['z_max']:.1f} km",
        f"  Number of points: {int(max_profile_stats['z_count'])}",
        f"  Average total difference: {max_profile_stats['total_diff_mean']:.3f} km/s",
        f"  Maximum total difference: {max_profile_stats['total_diff_max']:.3f} km/s",
        "",
        "Minimum difference vertical profile:",
        f"  Location (X,Y): ({extremes['min_profile_coords'][0]:.1f}, {extremes['min_profile_coords'][1]:.1f}) km",
        f"  Depth range: {min_profile_stats['z_min']:.1f} to {min_profile_stats['z_max']:.1f} km",
        f"  Number of points: {int(min_profile_stats['z_count'])}",
        f"  Average total difference: {min_profile_stats['total_diff_mean']:.3f} km/s",
        f"  Maximum total difference: {min_profile_stats['total_diff_max']:.3f} km/s",
        "",
        "Depth slices:",
        f"  Maximum average difference depth: {extremes['max_diff_depth']:.1f} km",
        f"  Minimum average difference depth: {extremes['min_diff_depth']:.1f} km",
        ""
    ]
    return lines


def _format_configuration_summary(config: Dict[str, Any]) -> list:
    """
    Format configuration summary section.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of formatted lines
    """
    lines = [
        "ANALYSIS CONFIGURATION",
        "-" * 40,
        f"Grid resolution: {config['velocity_analysis'].get('grid_resolution', 100)}",
        f"Interpolation method: {config['velocity_analysis'].get('interpolation_method', 'cubic')}",
        f"Log ratios computed: {config['velocity_analysis'].get('compute_log_ratios', True)}",
        f"Geometric mean ratios: {config['velocity_analysis'].get('compute_geometric_mean_ratios', True)}",
        f"Spatial projections: {config['velocity_analysis'].get('enable_spatial_projections', True)}",
        f"Profile analysis: {config['velocity_analysis'].get('enable_profile_analysis', True)}",
        f"Slice analysis: {config['velocity_analysis'].get('enable_slice_analysis', True)}",
        "",
        "=" * 80
    ]
    return lines