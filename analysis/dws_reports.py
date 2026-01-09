"""
Report generation for DWS (Derivative Weight Sum) analysis.

This module generates text-based summary reports of DWS analysis results.
"""

import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime
import math

logger = logging.getLogger(__name__)


def safe_format_float(value: Optional[Union[float, int]], format_spec: str = '.2f', default: str = 'N/A') -> str:
    """
    Safely format a numeric value, handling None and NaN.

    Args:
        value: The value to format
        format_spec: Format specification (e.g., '.2f', '.6e')
        default: Default string to return if value is None or NaN

    Returns:
        Formatted string
    """
    if value is None:
        return default
    try:
        if math.isnan(value):
            return default
    except (TypeError, ValueError):
        pass

    try:
        return f"{value:{format_spec}}"
    except (TypeError, ValueError):
        return default


def generate_dws_report(
    statistics: Dict[str, Any],
    metadata: Dict[str, Any],
    results: Dict[str, Any],
    output_file: str
) -> None:
    """
    Generate comprehensive DWS analysis report.

    Args:
        statistics: DWS statistics dictionary
        metadata: DWS metadata dictionary
        results: Analysis results dictionary
        output_file: Output file path
    """
    logger.info(f"Generating DWS analysis report: {output_file}")

    with open(output_file, 'w') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("DWS (DERIVATIVE WEIGHT SUM) ANALYSIS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")

        # Metadata
        f.write("FILE INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Format: {metadata.get('format', 'N/A')}\n")
        f.write(f"File size: {safe_format_float(metadata.get('file_size_mb'), '.2f')} MB\n")
        if 'data_type' in metadata:
            f.write(f"Data type: {metadata['data_type']}\n")
        if 'description' in metadata:
            f.write(f"Description: {metadata['description']}\n")
        if 'units' in metadata:
            f.write(f"Units: {metadata['units']}\n")
        f.write("\n")

        # Grid Information
        f.write("GRID INFORMATION\n")
        f.write("-"*80 + "\n")
        if 'grid_dimensions' in metadata:
            dims = metadata['grid_dimensions']
            f.write(f"Grid dimensions: {dims.get('nx', 'N/A')} x {dims.get('ny', 'N/A')} x {dims.get('nz', 'N/A')}\n")
        if 'grid_spacing' in metadata:
            spacing = metadata['grid_spacing']
            dx_str = safe_format_float(spacing.get('dx'), '.3f')
            dy_str = safe_format_float(spacing.get('dy'), '.3f')
            dz_str = safe_format_float(spacing.get('dz'), '.3f')
            f.write(f"Grid spacing: dx={dx_str} km, dy={dy_str} km, dz={dz_str} km\n")
        if 'grid_origin' in metadata:
            origin = metadata['grid_origin']
            x_str = safe_format_float(origin.get('x'), '.3f')
            y_str = safe_format_float(origin.get('y'), '.3f')
            z_str = safe_format_float(origin.get('z'), '.3f')
            f.write(f"Grid origin: x={x_str} km, y={y_str} km, z={z_str} km\n")
        if 'bounds' in metadata:
            bounds = metadata['bounds']
            x_min = safe_format_float(bounds['x'][0], '.3f') if bounds.get('x') else 'N/A'
            x_max = safe_format_float(bounds['x'][1], '.3f') if bounds.get('x') else 'N/A'
            y_min = safe_format_float(bounds['y'][0], '.3f') if bounds.get('y') else 'N/A'
            y_max = safe_format_float(bounds['y'][1], '.3f') if bounds.get('y') else 'N/A'
            z_min = safe_format_float(bounds['z'][0], '.3f') if bounds.get('z') else 'N/A'
            z_max = safe_format_float(bounds['z'][1], '.3f') if bounds.get('z') else 'N/A'
            f.write(f"X bounds: [{x_min}, {x_max}] km\n")
            f.write(f"Y bounds: [{y_min}, {y_max}] km\n")
            f.write(f"Z bounds: [{z_min}, {z_max}] km\n")
        f.write("\n")

        # Coverage Statistics
        f.write("COVERAGE STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total cells: {statistics.get('total_cells', 'N/A'):,}\n" if isinstance(statistics.get('total_cells'), int) else f"Total cells: {statistics.get('total_cells', 'N/A')}\n")
        f.write(f"Non-zero cells: {statistics.get('nonzero_cells', 'N/A'):,}\n" if isinstance(statistics.get('nonzero_cells'), int) else f"Non-zero cells: {statistics.get('nonzero_cells', 'N/A')}\n")
        f.write(f"Coverage: {safe_format_float(statistics.get('coverage_percentage'), '.2f')}%\n")
        f.write("\n")

        # DWS Value Statistics
        f.write("DWS VALUE STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Minimum: {safe_format_float(statistics.get('dws_min'), '.6e')} km²\n")
        f.write(f"Maximum: {safe_format_float(statistics.get('dws_max'), '.6e')} km²\n")
        f.write(f"Mean: {safe_format_float(statistics.get('dws_mean'), '.6e')} km²\n")
        f.write(f"Median: {safe_format_float(statistics.get('dws_median'), '.6e')} km²\n")
        f.write(f"Std Dev: {safe_format_float(statistics.get('dws_std'), '.6e')} km²\n")
        f.write("\n")

        # Percentiles
        f.write("DWS PERCENTILES\n")
        f.write("-"*80 + "\n")
        percentiles = statistics.get('dws_percentiles', {})
        f.write(f"25th percentile: {safe_format_float(percentiles.get('25'), '.6e')} km²\n")
        f.write(f"50th percentile: {safe_format_float(percentiles.get('50'), '.6e')} km²\n")
        f.write(f"75th percentile: {safe_format_float(percentiles.get('75'), '.6e')} km²\n")
        f.write(f"90th percentile: {safe_format_float(percentiles.get('90'), '.6e')} km²\n")
        f.write(f"95th percentile: {safe_format_float(percentiles.get('95'), '.6e')} km²\n")
        f.write(f"99th percentile: {safe_format_float(percentiles.get('99'), '.6e')} km²\n")
        f.write("\n")

        # Visualization Summary
        f.write("VISUALIZATION SUMMARY\n")
        f.write("-"*80 + "\n")

        # Horizontal slices
        h_slices = results.get('plots', {}).get('horizontal_slices', [])
        if h_slices:
            n_success = sum(1 for s in h_slices if s.get('status') == 'success')
            f.write(f"Horizontal slices: {n_success}/{len(h_slices)} successful\n")
            if n_success > 0:
                depths = [s.get('depth_km') for s in h_slices if s.get('status') == 'success']
                depths_str = ', '.join([safe_format_float(d, '.2f') for d in depths if d is not None])
                if depths_str:
                    f.write(f"  Depths: {depths_str} km\n")

        # Vertical slices
        v_slices = results.get('plots', {}).get('vertical_slices', [])
        if v_slices:
            n_success = sum(1 for s in v_slices if s.get('status') == 'success')
            f.write(f"Vertical slices: {n_success}/{len(v_slices)} successful\n")

        # Coverage maps
        c_maps = results.get('plots', {}).get('coverage_maps', [])
        if c_maps:
            n_success = sum(1 for m in c_maps if m.get('status') == 'success')
            f.write(f"Coverage maps: {n_success}/{len(c_maps)} successful\n")

        # Histograms
        histograms = results.get('plots', {}).get('histograms', [])
        if histograms:
            n_success = sum(1 for h in histograms if h.get('status') == 'success')
            f.write(f"Histograms: {n_success}/{len(histograms)} successful\n")

        f.write("\n")

        # Footer
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    logger.info(f"DWS analysis report saved: {output_file}")
