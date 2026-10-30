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
    output_file: str,
    use_m2_units: bool = False
) -> None:
    """
    Generate comprehensive DWS analysis report.

    Args:
        statistics: DWS statistics dictionary
        metadata: DWS metadata dictionary
        results: Analysis results dictionary
        output_file: Output file path
        use_m2_units: If True, display values in m² instead of km²
    """
    logger.info(f"Generating DWS analysis report: {output_file}")

    # Unit conversion
    unit_label = 'm²' if use_m2_units else 'km²'
    unit_conversion = 1e6 if use_m2_units else 1.0

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
        dws_units = metadata.get('units.dws', metadata.get('units'))
        if dws_units:
            f.write(f"Units: {dws_units}\n")
        f.write("\n")

        # Grid Information
        f.write("GRID INFORMATION\n")
        f.write("-"*80 + "\n")
        if 'nx' in metadata:
            f.write(f"Grid dimensions: {metadata.get('nx', 'N/A')} x {metadata.get('ny', 'N/A')} x {metadata.get('nz', 'N/A')}\n")
        if 'dx' in metadata:
            dx_str = safe_format_float(metadata.get('dx'), '.3f')
            dy_str = safe_format_float(metadata.get('dy'), '.3f')
            dz_str = safe_format_float(metadata.get('dz'), '.3f')
            f.write(f"Grid spacing: dx={dx_str} km, dy={dy_str} km, dz={dz_str} km\n")
        if 'origin_x' in metadata:
            x_str = safe_format_float(metadata.get('origin_x'), '.3f')
            y_str = safe_format_float(metadata.get('origin_y'), '.3f')
            z_str = safe_format_float(metadata.get('origin_z'), '.3f')
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
        # Apply unit conversion to statistics
        dws_min = statistics.get('dws_min')
        dws_max = statistics.get('dws_max')
        dws_mean = statistics.get('dws_mean')
        dws_median = statistics.get('dws_median')
        dws_std = statistics.get('dws_std')

        f.write(f"Minimum: {safe_format_float(dws_min * unit_conversion if dws_min is not None else None, '.6e')} {unit_label}\n")
        f.write(f"Maximum: {safe_format_float(dws_max * unit_conversion if dws_max is not None else None, '.6e')} {unit_label}\n")
        f.write(f"Mean: {safe_format_float(dws_mean * unit_conversion if dws_mean is not None else None, '.6e')} {unit_label}\n")
        f.write(f"Median: {safe_format_float(dws_median * unit_conversion if dws_median is not None else None, '.6e')} {unit_label}\n")
        f.write(f"Std Dev: {safe_format_float(dws_std * unit_conversion if dws_std is not None else None, '.6e')} {unit_label}\n")
        f.write("\n")

        # Percentiles
        f.write("DWS PERCENTILES\n")
        f.write("-"*80 + "\n")
        percentiles = statistics.get('dws_percentiles', {})
        # Apply unit conversion to percentiles
        p25 = percentiles.get('25')
        p50 = percentiles.get('50')
        p75 = percentiles.get('75')
        p90 = percentiles.get('90')
        p95 = percentiles.get('95')
        p99 = percentiles.get('99')

        f.write(f"25th percentile: {safe_format_float(p25 * unit_conversion if p25 is not None else None, '.6e')} {unit_label}\n")
        f.write(f"50th percentile: {safe_format_float(p50 * unit_conversion if p50 is not None else None, '.6e')} {unit_label}\n")
        f.write(f"75th percentile: {safe_format_float(p75 * unit_conversion if p75 is not None else None, '.6e')} {unit_label}\n")
        f.write(f"90th percentile: {safe_format_float(p90 * unit_conversion if p90 is not None else None, '.6e')} {unit_label}\n")
        f.write(f"95th percentile: {safe_format_float(p95 * unit_conversion if p95 is not None else None, '.6e')} {unit_label}\n")
        f.write(f"99th percentile: {safe_format_float(p99 * unit_conversion if p99 is not None else None, '.6e')} {unit_label}\n")
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
