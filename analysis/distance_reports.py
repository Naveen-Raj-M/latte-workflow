"""
Distance analysis report generation module.

This module generates comprehensive text reports for distance analysis.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from utils.io_utils import write_text_file, ensure_directory_exists
from utils.distance_utils import get_distance_label, get_output_suffix

logger = logging.getLogger(__name__)


def generate_distance_analysis_report(df, stats: Dict[str, Any],
                                     config: Dict[str, Any], distance_type: str) -> None:
    """Generate comprehensive distance analysis report."""
    output_dir = Path(config['distance_analysis']['output_dir'])
    coordinate_type = config['distance_analysis']['coordinate_type']
    comparison_mode = config['distance_analysis'].get('comparison_mode', 'ground_truth')
    
    # Create subdirectory for distance type
    suffix = get_output_suffix(distance_type)
    mode_suffix = 'comparison' if comparison_mode == 'method_comparison' else 'ground_truth'
    reports_dir = output_dir / 'reports' / f'{suffix}_{coordinate_type}_{mode_suffix}'
    ensure_directory_exists(reports_dir)
    
    report_file = reports_dir / f'distance_analysis_report_{suffix}.txt'
    distance_label = get_distance_label(distance_type)
    
    report_lines = _build_report_header(distance_label, coordinate_type, comparison_mode, config, stats)
    report_lines.extend(_build_overall_statistics(stats))
    report_lines.extend(_build_phase_statistics(stats))
    
    if 'ratio' in stats:
        report_lines.extend(_build_ratio_statistics(stats))
    
    if comparison_mode == 'method_comparison':
        report_lines.extend(_build_comparison_summary(stats, config))
    
    write_text_file(report_file, report_lines)
    logger.info(f"{distance_label} distance analysis report saved to: {report_file}")


def _build_report_header(distance_label: str, coordinate_type: str,
                        comparison_mode: str, config: Dict[str, Any],
                        stats: Dict[str, Any]) -> list:
    """Build report header section."""
    lines = [
        f"{distance_label.upper()} DISTANCE-DEPENDENT TRAVEL TIME ANALYSIS REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Distance type: {distance_label}",
        f"Coordinate type: {coordinate_type}",
        f"Comparison mode: {comparison_mode}",
        ""
    ]
    
    if comparison_mode == 'method_comparison':
        method1_name = config['distance_analysis'].get('method1_name', 'Method 1')
        method2_name = config['distance_analysis'].get('method2_name', 'Method 2')
        method1_csv = config['distance_analysis'].get('method1_csv', 'N/A')
        method2_csv = config['distance_analysis'].get('method2_csv', 'N/A')
        
        lines.extend([
            "METHOD COMPARISON",
            "-" * 40,
            f"Method 1: {method1_name}",
            f"  File: {method1_csv}",
            f"Method 2: {method2_name}",
            f"  File: {method2_csv}",
            f"Residuals computed as: {method1_name} - {method2_name}",
            ""
        ])
    else:
        travel_time_csv = config['distance_analysis'].get('travel_time_csv', 'N/A')
        lines.extend([
            f"Travel time file: {travel_time_csv}",
            ""
        ])
    
    return lines


def _build_overall_statistics(stats: Dict[str, Any]) -> list:
    """Build overall statistics section."""
    overall = stats['overall']
    
    lines = [
        "OVERALL STATISTICS",
        "-" * 40,
        f"Total data points: {overall['count']}",
        f"Distance range: {overall['distance_min']:.3f} - {overall['distance_max']:.3f} km",
        f"Mean distance: {overall['distance_mean']:.3f} km",
        f"Median distance: {overall['distance_median']:.3f} km",
        f"Distance std dev: {overall['distance_std']:.3f} km",
        f"Distance-residual correlation: {overall['residual_distance_correlation']:.4f}",
        f"Distance-|residual| correlation: {overall['residual_abs_distance_correlation']:.4f}",
        ""
    ]
    
    return lines


def _build_phase_statistics(stats: Dict[str, Any]) -> list:
    """Build phase-specific statistics sections."""
    lines = []
    
    for phase in ['P', 'S']:
        phase_key = f'{phase.lower()}_wave'
        if phase_key in stats:
            phase_stats = stats[phase_key]
            lines.extend([
                f"{phase}-WAVE STATISTICS",
                "-" * 40,
                f"Data points: {phase_stats['count']}",
                f"Mean distance: {phase_stats['distance_mean']:.3f} km",
                f"Distance-residual correlation: {phase_stats['residual_distance_correlation']:.4f}",
                f"Distance-|residual| correlation: {phase_stats['residual_abs_distance_correlation']:.4f}",
                f"Mean residual: {phase_stats['mean_residual']:.4f} s",
                f"RMS residual: {phase_stats['rms_residual']:.4f} s",
            ])
            
            # Add polynomial fit if available
            if 'polynomial_coeffs' in phase_stats:
                coeffs = phase_stats['polynomial_coeffs']
                if len(coeffs) == 2:
                    lines.append(f"Linear fit: y = {coeffs[0]:.6f}x + {coeffs[1]:.6f}")
                else:
                    lines.append(f"Polynomial coefficients: {coeffs}")
            
            lines.append("")
    
    return lines


def _build_ratio_statistics(stats: Dict[str, Any]) -> list:
    """Build ratio analysis statistics section."""
    ratio_stats = stats['ratio']
    
    if not ratio_stats:
        return []
    
    lines = [
        "RATIO ANALYSIS",
        "-" * 40,
        f"Total ratio points: {ratio_stats['overall']['count']}",
        f"Mean ratio (Syn/Obs): {ratio_stats['overall']['mean_ratio']:.4f}",
        f"Mean ln(ratio): {ratio_stats['overall']['mean_ln_ratio']:.4f}",
        f"Ratio-distance correlation: {ratio_stats['overall']['ratio_distance_correlation']:.4f}",
        f"ln(ratio)-distance correlation: {ratio_stats['overall']['ln_ratio_distance_correlation']:.4f}",
        ""
    ]
    
    for phase in ['P', 'S']:
        phase_key = f'{phase.lower()}_wave'
        if phase_key in ratio_stats:
            phase_ratio = ratio_stats[phase_key]
            lines.extend([
                f"{phase}-wave ratio statistics:",
                f"  Points: {phase_ratio['count']}",
                f"  Mean ratio: {phase_ratio['mean_ratio']:.4f}",
                f"  Mean ln(ratio): {phase_ratio['mean_ln_ratio']:.4f}",
                f"  Ratio-distance correlation: {phase_ratio['ratio_distance_correlation']:.4f}",
                f"  ln(ratio)-distance correlation: {phase_ratio['ln_ratio_distance_correlation']:.4f}",
                ""
            ])
    
    return lines


def _build_comparison_summary(stats: Dict[str, Any], config: Dict[str, Any]) -> list:
    """Build method comparison summary section."""
    method1_name = stats.get('method1_name', 'Method 1')
    method2_name = stats.get('method2_name', 'Method 2')
    
    lines = [
        "METHOD COMPARISON SUMMARY",
        "-" * 40,
        f"Comparing: {method1_name} vs {method2_name}",
        f"Residuals: {method1_name} - {method2_name}",
        ""
    ]
    
    # Add interpretation
    overall_mean = stats['overall'].get('residual_distance_correlation', 0)
    
    lines.append("INTERPRETATION:")
    if abs(overall_mean) < 0.1:
        lines.append(f"  Weak correlation between residuals and distance (r={overall_mean:.4f})")
        lines.append(f"  Methods show similar performance across distance ranges")
    elif overall_mean > 0.1:
        lines.append(f"  Positive correlation between residuals and distance (r={overall_mean:.4f})")
        lines.append(f"  {method1_name} tends to overpredict at larger distances")
    else:
        lines.append(f"  Negative correlation between residuals and distance (r={overall_mean:.4f})")
        lines.append(f"  {method1_name} tends to underpredict at larger distances")
    
    lines.append("")
    
    return lines