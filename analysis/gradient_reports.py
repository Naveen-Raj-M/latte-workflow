"""
Report generation functions for gradient and search direction analysis.

This module generates comprehensive text reports summarizing gradient statistics,
search directions, and convergence diagnostics.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from utils.io_utils import write_text_file, ensure_directory_exists

logger = logging.getLogger(__name__)


def generate_velocity_gradient_report(df: pd.DataFrame,
                                     available_fields: Dict[str, List[str]],
                                     stats: Dict[str, Any],
                                     config: Dict[str, Any]) -> None:
    """
    Generate comprehensive report for velocity gradients and search directions.
    
    Args:
        df: Velocity dataframe
        available_fields: Available fields dictionary
        stats: Statistics dictionary
        config: Configuration dictionary
    """
    output_dir = Path(config['gradient_analysis']['velocity_output_dir'])
    reports_dir = output_dir / 'reports'
    ensure_directory_exists(reports_dir)
    
    report_file = reports_dir / 'velocity_gradient_analysis_report.txt'
    
    report_lines = [
        "=" * 80,
        "VELOCITY GRADIENT AND SEARCH DIRECTION ANALYSIS REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Input file: {config['gradient_analysis']['velocity_input_csv']}",
        "",
        "AVAILABLE FIELDS",
        "-" * 40,
        f"Gradient fields: {', '.join(available_fields['gradients']) if available_fields['gradients'] else 'None'}",
        f"Search direction fields: {', '.join(available_fields['search']) if available_fields['search'] else 'None'}",
        "",
    ]
    
    # Gradient statistics
    if 'gradients' in stats and stats['gradients']:
        report_lines.extend(_format_gradient_statistics(stats['gradients'], 'velocity'))
    
    # Search direction statistics
    if 'search' in stats and stats['search']:
        report_lines.extend(_format_search_statistics(stats['search'], 'velocity'))
    
    # Alignment statistics
    if 'alignment' in stats and stats['alignment']:
        report_lines.extend(_format_alignment_statistics(stats['alignment']))
    
    # Depth-dependent statistics
    if 'depth_dependent' in stats and stats['depth_dependent']:
        report_lines.extend(_format_depth_statistics(stats['depth_dependent']))
    
    # Convergence diagnostics
    report_lines.extend(_format_convergence_diagnostics(stats, available_fields))
    
    # Write report
    write_text_file(report_file, report_lines)
    logger.info(f"Velocity gradient report saved to: {report_file}")


def generate_source_gradient_report(df: pd.DataFrame,
                                   available_fields: Dict[str, List[str]],
                                   stats: Dict[str, Any],
                                   config: Dict[str, Any]) -> None:
    """
    Generate comprehensive report for source parameter gradients.
    
    Args:
        df: Source dataframe
        available_fields: Available fields dictionary
        stats: Statistics dictionary
        config: Configuration dictionary
    """
    output_dir = Path(config['gradient_analysis']['source_output_dir'])
    reports_dir = output_dir / 'reports'
    ensure_directory_exists(reports_dir)
    
    report_file = reports_dir / 'source_gradient_analysis_report.txt'
    
    report_lines = [
        "=" * 80,
        "SOURCE PARAMETER GRADIENT AND SEARCH DIRECTION ANALYSIS REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Input file: {config['gradient_analysis']['source_input_csv']}",
        "",
        "AVAILABLE FIELDS",
        "-" * 40,
        f"Gradient fields: {', '.join(available_fields['gradients']) if available_fields['gradients'] else 'None'}",
        f"Search direction fields: {', '.join(available_fields['search']) if available_fields['search'] else 'None'}",
        "",
    ]
    
    # Gradient statistics
    if 'gradients' in stats and stats['gradients']:
        report_lines.extend(_format_gradient_statistics(stats['gradients'], 'source'))
    
    # Search direction statistics
    if 'search' in stats and stats['search']:
        report_lines.extend(_format_search_statistics(stats['search'], 'source'))
    
    # Alignment statistics
    if 'alignment' in stats and stats['alignment']:
        report_lines.extend(_format_alignment_statistics(stats['alignment']))
    
    # Spatial gradient magnitudes
    if 'spatial_gradients' in stats and stats['spatial_gradients']:
        report_lines.extend(_format_spatial_gradient_statistics(stats['spatial_gradients']))
    
    # Convergence diagnostics
    report_lines.extend(_format_convergence_diagnostics(stats, available_fields))
    
    # Write report
    write_text_file(report_file, report_lines)
    logger.info(f"Source gradient report saved to: {report_file}")


def _format_gradient_statistics(grad_stats: Dict[str, Any], 
                                analysis_type: str) -> List[str]:
    """Format gradient statistics section."""
    lines = [
        "GRADIENT STATISTICS",
        "=" * 80,
        ""
    ]
    
    for param, stats in grad_stats.items():
        unit = _get_gradient_unit(param, analysis_type)
        
        lines.extend([
            f"{param.upper()} GRADIENT",
            "-" * 40,
            f"Data points: {stats['count']}",
            f"Mean: {stats['mean']:.6e} {unit}",
            f"Std: {stats['std']:.6e} {unit}",
            f"Median: {stats['median']:.6e} {unit}",
            f"Range: [{stats['min']:.6e}, {stats['max']:.6e}] {unit}",
            f"Mean absolute: {stats['abs_mean']:.6e} {unit}",
            f"Max absolute: {stats['abs_max']:.6e} {unit}",
            f"RMS: {stats['rms']:.6e} {unit}",
            "",
            "Sign distribution:",
            f"  Positive: {stats['positive_count']} ({stats['positive_count']/stats['count']*100:.1f}%)",
            f"  Negative: {stats['negative_count']} ({stats['negative_count']/stats['count']*100:.1f}%)",
            f"  Near-zero: {stats['zero_count']} ({stats['zero_count']/stats['count']*100:.1f}%)",
            "",
        ])
    
    return lines


def _format_search_statistics(search_stats: Dict[str, Any],
                              analysis_type: str) -> List[str]:
    """Format search direction statistics section."""
    lines = [
        "SEARCH DIRECTION STATISTICS",
        "=" * 80,
        ""
    ]
    
    for param, stats in search_stats.items():
        unit = _get_search_unit(param, analysis_type)
        
        lines.extend([
            f"{param.upper()} SEARCH DIRECTION",
            "-" * 40,
            f"Data points: {stats['count']}",
            f"Mean: {stats['mean']:.6e} {unit}",
            f"Std: {stats['std']:.6e} {unit}",
            f"Median: {stats['median']:.6e} {unit}",
            f"Range: [{stats['min']:.6e}, {stats['max']:.6e}] {unit}",
            f"Mean absolute: {stats['abs_mean']:.6e} {unit}",
            f"Max absolute: {stats['abs_max']:.6e} {unit}",
            f"RMS: {stats['rms']:.6e} {unit}",
            "",
            "Sign distribution:",
            f"  Positive: {stats['positive_count']} ({stats['positive_count']/stats['count']*100:.1f}%)",
            f"  Negative: {stats['negative_count']} ({stats['negative_count']/stats['count']*100:.1f}%)",
            f"  Near-zero: {stats['zero_count']} ({stats['zero_count']/stats['count']*100:.1f}%)",
            "",
        ])
    
    return lines


def _format_alignment_statistics(alignment_stats: Dict[str, Any]) -> List[str]:
    """Format gradient-search alignment statistics."""
    lines = [
        "GRADIENT-SEARCH ALIGNMENT ANALYSIS",
        "=" * 80,
        ""
    ]
    
    for param, stats in alignment_stats.items():
        lines.extend([
            f"{param.upper()} ALIGNMENT",
            "-" * 40,
            f"Cosine similarity: {stats['cosine_similarity']:.4f}",
            f"Same sign count: {stats['same_sign_count']}",
            f"Opposite sign count: {stats['opposite_sign_count']}",
            f"Same sign percentage: {stats['same_sign_percent']:.1f}%",
            ""
        ])
        
        # Interpretation
        if stats['cosine_similarity'] > 0.8:
            interpretation = "Strong alignment (descent direction)"
        elif stats['cosine_similarity'] > 0.5:
            interpretation = "Moderate alignment"
        elif stats['cosine_similarity'] > -0.5:
            interpretation = "Weak alignment"
        else:
            interpretation = "Strong anti-alignment (ascent direction)"
        
        lines.extend([
            f"Interpretation: {interpretation}",
            ""
        ])
    
    return lines


def _format_depth_statistics(depth_stats: Dict[str, Any]) -> List[str]:
    """Format depth-dependent statistics."""
    lines = [
        "DEPTH-DEPENDENT GRADIENT STATISTICS",
        "=" * 80,
        ""
    ]
    
    for param, bin_data in depth_stats.items():
        lines.extend([
            f"{param.upper()} GRADIENT BY DEPTH",
            "-" * 40,
        ])
        
        # Extract and sort by depth
        depth_entries = []
        for bin_interval, data in bin_data.items():
            if hasattr(bin_interval, 'mid'):
                depth_entries.append((bin_interval.mid, data))
        
        depth_entries.sort(key=lambda x: x[0])
        
        for depth, data in depth_entries:
            lines.append(f"  Depth {depth:.1f} km: mean={data['mean']:.6e}, "
                        f"std={data['std']:.6e}, count={data['count']}")
        
        lines.append("")
    
    return lines


def _format_spatial_gradient_statistics(spatial_stats: Dict[str, Any]) -> List[str]:
    """Format spatial gradient magnitude statistics."""
    lines = [
        "SPATIAL GRADIENT MAGNITUDES",
        "=" * 80,
        ""
    ]
    
    if 'horizontal' in spatial_stats:
        h_stats = spatial_stats['horizontal']
        lines.extend([
            "HORIZONTAL GRADIENT MAGNITUDE",
            "-" * 40,
            f"Mean magnitude: {h_stats['mean_magnitude']:.6e} m",
            f"Max magnitude: {h_stats['max_magnitude']:.6e} m",
            f"Std magnitude: {h_stats['std_magnitude']:.6e} m",
            ""
        ])
    
    if '3d' in spatial_stats:
        td_stats = spatial_stats['3d']
        lines.extend([
            "3D GRADIENT MAGNITUDE",
            "-" * 40,
            f"Mean magnitude: {td_stats['mean_magnitude']:.6e} m",
            f"Max magnitude: {td_stats['max_magnitude']:.6e} m",
            f"Std magnitude: {td_stats['std_magnitude']:.6e} m",
            ""
        ])
    
    return lines


def _format_convergence_diagnostics(stats: Dict[str, Any],
                                   available_fields: Dict[str, List[str]]) -> List[str]:
    """Format convergence diagnostic information."""
    lines = [
        "CONVERGENCE DIAGNOSTICS",
        "=" * 80,
        ""
    ]
    
    # Check gradient magnitudes
    if 'gradients' in stats and stats['gradients']:
        lines.append("GRADIENT MAGNITUDE ANALYSIS")
        lines.append("-" * 40)
        
        for param, grad_stats in stats['gradients'].items():
            rms = grad_stats['rms']
            abs_mean = grad_stats['abs_mean']
            
            # Simple heuristic for convergence assessment
            if abs_mean < 1e-6:
                status = "Very small (near convergence)"
            elif abs_mean < 1e-4:
                status = "Small (approaching convergence)"
            elif abs_mean < 1e-2:
                status = "Moderate (active optimization)"
            else:
                status = "Large (early optimization)"
            
            lines.append(f"  {param.upper()}: RMS={rms:.6e}, Status: {status}")
        
        lines.append("")
    
    # Check gradient-search alignment
    if 'alignment' in stats and stats['alignment']:
        lines.append("DESCENT DIRECTION ANALYSIS")
        lines.append("-" * 40)
        
        for param, align_stats in stats['alignment'].items():
            cos_sim = align_stats['cosine_similarity']
            same_pct = align_stats['same_sign_percent']
            
            if cos_sim > 0.9 and same_pct > 90:
                assessment = "Excellent descent direction"
            elif cos_sim > 0.7 and same_pct > 80:
                assessment = "Good descent direction"
            elif cos_sim > 0.5:
                assessment = "Acceptable descent direction"
            else:
                assessment = "Poor descent direction - check optimization"
            
            lines.append(f"  {param.upper()}: {assessment}")
            lines.append(f"    (cosine similarity={cos_sim:.3f}, same sign={same_pct:.1f}%)")
        
        lines.append("")
    
    # Overall assessment
    lines.extend([
        "OVERALL ASSESSMENT",
        "-" * 40,
    ])
    
    if 'gradients' in stats and stats['gradients']:
        # Calculate average RMS across all gradients
        avg_rms = np.mean([s['rms'] for s in stats['gradients'].values()])
        
        if avg_rms < 1e-5:
            overall = "Model appears to be near convergence (very small gradients)"
        elif avg_rms < 1e-3:
            overall = "Model is approaching convergence (small gradients)"
        elif avg_rms < 1e-1:
            overall = "Model is actively optimizing (moderate gradients)"
        else:
            overall = "Model is in early optimization phase (large gradients)"
        
        lines.append(f"  {overall}")
        lines.append(f"  Average gradient RMS: {avg_rms:.6e}")
    else:
        lines.append("  Insufficient data for convergence assessment")
    
    lines.extend(["", "=" * 80])
    
    return lines


def _get_gradient_unit(param: str, analysis_type: str) -> str:
    """Get unit string for gradient."""
    if analysis_type == 'velocity':
        if param in ['vp', 'vs']:
            return '[(km/s) per unit]'
    else:  # source
        if param in ['sx', 'sy', 'sz']:
            return '[m per unit]'
        elif param == 'st0':
            return '[s per unit]'
    return '[per unit]'


def _get_search_unit(param: str, analysis_type: str) -> str:
    """Get unit string for search direction."""
    if analysis_type == 'velocity':
        if param in ['vp', 'vs']:
            return '[km/s]'
    else:  # source
        if param in ['sx', 'sy', 'sz']:
            return '[m]'
        elif param == 'st0':
            return '[s]'
    return '[]'