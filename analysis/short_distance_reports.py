"""
Short distance analysis report generation module.

This module generates comprehensive text reports for short distance analysis.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from utils.io_utils import write_text_file, ensure_directory_exists
from utils.distance_utils import get_distance_label, get_output_suffix

logger = logging.getLogger(__name__)


def generate_short_distance_report(df, stats: Dict[str, Any],
                                  config: Dict[str, Any], distance_type: str) -> None:
    """Generate comprehensive short distance analysis report."""
    output_dir = Path(config['short_distance_analysis']['output_dir'])
    coordinate_type = config['short_distance_analysis']['coordinate_type']
    comparison_mode = config['short_distance_analysis'].get('comparison_mode', 'ground_truth')
    
    # Create subdirectory
    suffix = get_output_suffix(distance_type)
    mode_suffix = 'comparison' if comparison_mode == 'method_comparison' else 'ground_truth'
    reports_dir = output_dir / 'reports' / f'{suffix}_{coordinate_type}_{mode_suffix}'
    ensure_directory_exists(reports_dir)
    
    report_file = reports_dir / f'short_distance_report_{suffix}.txt'
    distance_label = get_distance_label(distance_type)
    
    report_lines = _build_report_header(distance_label, coordinate_type, comparison_mode, config, stats)
    report_lines.extend(_build_overall_statistics(stats))
    report_lines.extend(_build_phase_statistics(stats))
    report_lines.extend(_build_event_station_statistics(stats))
    
    if 'clusters' in stats and stats['clusters']:
        report_lines.extend(_build_cluster_statistics(stats))
    
    if comparison_mode == 'method_comparison':
        report_lines.extend(_build_comparison_summary(stats, config))
    
    write_text_file(report_file, report_lines)
    logger.info(f"{distance_label} short distance report saved to: {report_file}")


def _build_report_header(distance_label: str, coordinate_type: str,
                        comparison_mode: str, config: Dict[str, Any],
                        stats: Dict[str, Any]) -> list:
    """Build report header section."""
    distance_cutoff = stats.get('distance_cutoff_km', 0)
    
    lines = [
        f"SHORT {distance_label.upper()} DISTANCE TRAVEL TIME ANALYSIS REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Distance type: {distance_label}",
        f"Distance cutoff: ≤ {distance_cutoff} km",
        f"Coordinate type: {coordinate_type}",
        f"Comparison mode: {comparison_mode}",
        ""
    ]
    
    if comparison_mode == 'method_comparison':
        method1_name = config['short_distance_analysis'].get('method1_name', 'Method 1')
        method2_name = config['short_distance_analysis'].get('method2_name', 'Method 2')
        method1_csv = config['short_distance_analysis'].get('method1_csv', 'N/A')
        method2_csv = config['short_distance_analysis'].get('method2_csv', 'N/A')
        
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
        travel_time_csv = config['short_distance_analysis'].get('travel_time_csv', 'N/A')
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
        f"Unique events: {overall['unique_events']}",
        f"Unique stations: {overall['unique_stations']}",
        "",
        "Distance Statistics:",
        f"  Range: {overall['distance_min']:.3f} - {overall['distance_max']:.3f} km",
        f"  Mean: {overall['distance_mean']:.3f} km",
        f"  Median: {overall['distance_median']:.3f} km",
        "",
        "Residual Statistics:",
        f"  Mean: {overall['residual_mean']:.4f} s",
        f"  Std dev: {overall['residual_std']:.4f} s",
        f"  RMS: {overall['residual_rms']:.4f} s",
        f"  Range: {overall['residual_min']:.4f} - {overall['residual_max']:.4f} s",
        f"  Median: {overall['residual_median']:.4f} s",
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
                "",
                "Distance Statistics:",
                f"  Mean: {phase_stats['distance_mean']:.3f} km",
                f"  Std dev: {phase_stats['distance_std']:.3f} km",
                "",
                "Residual Statistics:",
                f"  Mean: {phase_stats['residual_mean']:.4f} s",
                f"  Std dev: {phase_stats['residual_std']:.4f} s",
                f"  RMS: {phase_stats['residual_rms']:.4f} s",
                f"  Range: {phase_stats['residual_min']:.4f} - {phase_stats['residual_max']:.4f} s",
                "",
                "Correlation:",
                f"  Distance-Residual: {phase_stats['distance_residual_corr']:.4f}",
                ""
            ])
    
    return lines


def _build_event_station_statistics(stats: Dict[str, Any]) -> list:
    """Build event and station statistics sections."""
    lines = []
    
    if 'events' in stats:
        event_stats = stats['events']
        lines.extend([
            "EVENT STATISTICS",
            "-" * 40,
            f"Unique events: {event_stats['unique_events']}",
            f"Mean picks per event: {event_stats['mean_picks_per_event']:.2f}",
            f"Median picks per event: {event_stats['median_picks_per_event']:.1f}",
            f"Picks per event range: {event_stats['min_picks_per_event']} - {event_stats['max_picks_per_event']}",
            ""
        ])
    
    if 'stations' in stats:
        station_stats = stats['stations']
        lines.extend([
            "STATION STATISTICS",
            "-" * 40,
            f"Unique stations: {station_stats['unique_stations']}",
            f"Mean picks per station: {station_stats['mean_picks_per_station']:.2f}",
            f"Median picks per station: {station_stats['median_picks_per_station']:.1f}",
            f"Picks per station range: {station_stats['min_picks_per_station']} - {station_stats['max_picks_per_station']}",
            ""
        ])
    
    return lines


def _build_cluster_statistics(stats: Dict[str, Any]) -> list:
    """Build cluster statistics section."""
    cluster_stats = stats.get('clusters', {})
    
    if not cluster_stats:
        return []
    
    lines = [
        "CLUSTER STATISTICS",
        "-" * 40,
        f"Number of defined clusters: {len(cluster_stats)}",
        ""
    ]
    
    for cluster_name, cluster_data in cluster_stats.items():
        if cluster_data.get('count', 0) > 0:
            lines.extend([
                f"Cluster: {cluster_name}",
                f"  Description: {cluster_data.get('description', 'N/A')}",
                f"  Data points: {cluster_data['count']}",
                f"  Unique events: {cluster_data['unique_events']}",
                f"  Unique stations: {cluster_data['unique_stations']}",
                f"  Mean distance: {cluster_data['distance_mean']:.3f} km",
                f"  Mean residual: {cluster_data['residual_mean']:.4f} s",
                f"  Std residual: {cluster_data['residual_std']:.4f} s",
                f"  RMS residual: {cluster_data['residual_rms']:.4f} s",
                ""
            ])
        else:
            lines.extend([
                f"Cluster: {cluster_name}",
                f"  Description: {cluster_data.get('description', 'N/A')}",
                f"  {cluster_data.get('message', 'No data')}",
                ""
            ])
    
    return lines


def _build_comparison_summary(stats: Dict[str, Any], config: Dict[str, Any]) -> list:
    """Build method comparison summary section."""
    method1_name = stats.get('method1_name', 'Method 1')
    method2_name = stats.get('method2_name', 'Method 2')
    
    overall = stats['overall']
    
    lines = [
        "METHOD COMPARISON SUMMARY",
        "-" * 40,
        f"Comparing: {method1_name} vs {method2_name}",
        f"Residuals: {method1_name} - {method2_name}",
        "",
        "Overall Residual Analysis:",
        f"  Mean residual: {overall['residual_mean']:.4f} s",
        f"  RMS residual: {overall['residual_rms']:.4f} s",
        ""
    ]
    
    # Add interpretation
    mean_residual = overall['residual_mean']
    rms_residual = overall['residual_rms']
    
    lines.append("INTERPRETATION:")
    
    if abs(mean_residual) < 0.1:
        lines.append(f"  Small mean bias (|mean| < 0.1 s)")
        lines.append(f"  Methods show similar performance on average")
    elif mean_residual > 0.1:
        lines.append(f"  Positive mean bias ({mean_residual:.4f} s)")
        lines.append(f"  {method1_name} tends to predict larger travel times than {method2_name}")
    else:
        lines.append(f"  Negative mean bias ({mean_residual:.4f} s)")
        lines.append(f"  {method1_name} tends to predict smaller travel times than {method2_name}")
    
    lines.append("")
    
    if rms_residual < 0.2:
        lines.append(f"  Excellent agreement (RMS < 0.2 s)")
    elif rms_residual < 0.5:
        lines.append(f"  Good agreement (RMS < 0.5 s)")
    elif rms_residual < 1.0:
        lines.append(f"  Moderate agreement (RMS < 1.0 s)")
    else:
        lines.append(f"  Significant differences (RMS ≥ 1.0 s)")
    
    lines.append("")
    
    # Phase-specific comparison
    if 'p_wave' in stats and 's_wave' in stats:
        p_rms = stats['p_wave']['residual_rms']
        s_rms = stats['s_wave']['residual_rms']
        
        lines.extend([
            "Phase-specific Performance:",
            f"  P-wave RMS: {p_rms:.4f} s",
            f"  S-wave RMS: {s_rms:.4f} s",
        ])
        
        if p_rms < s_rms:
            lines.append(f"  Better agreement for P-waves")
        else:
            lines.append(f"  Better agreement for S-waves")
        
        lines.append("")
    
    return lines