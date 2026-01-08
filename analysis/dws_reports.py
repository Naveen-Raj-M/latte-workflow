"""
Report generation for DWS (Derivative Weight Sum) analysis.

This module generates text-based summary reports of DWS analysis results.
"""

import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


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
        f.write(f"File size: {metadata.get('file_size_mb', 0):.2f} MB\n")
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
            f.write(f"Grid spacing: dx={spacing.get('dx', 'N/A'):.3f} km, dy={spacing.get('dy', 'N/A'):.3f} km, dz={spacing.get('dz', 'N/A'):.3f} km\n")
        if 'grid_origin' in metadata:
            origin = metadata['grid_origin']
            f.write(f"Grid origin: x={origin.get('x', 'N/A'):.3f} km, y={origin.get('y', 'N/A'):.3f} km, z={origin.get('z', 'N/A'):.3f} km\n")
        if 'bounds' in metadata:
            bounds = metadata['bounds']
            f.write(f"X bounds: [{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}] km\n")
            f.write(f"Y bounds: [{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}] km\n")
            f.write(f"Z bounds: [{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}] km\n")
        f.write("\n")

        # Coverage Statistics
        f.write("COVERAGE STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total cells: {statistics['total_cells']:,}\n")
        f.write(f"Non-zero cells: {statistics['nonzero_cells']:,}\n")
        f.write(f"Coverage: {statistics['coverage_percentage']:.2f}%\n")
        f.write("\n")

        # DWS Value Statistics
        f.write("DWS VALUE STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Minimum: {statistics['dws_min']:.6e} km²\n")
        f.write(f"Maximum: {statistics['dws_max']:.6e} km²\n")
        f.write(f"Mean: {statistics['dws_mean']:.6e} km²\n")
        f.write(f"Median: {statistics['dws_median']:.6e} km²\n")
        f.write(f"Std Dev: {statistics['dws_std']:.6e} km²\n")
        f.write("\n")

        # Percentiles
        f.write("DWS PERCENTILES\n")
        f.write("-"*80 + "\n")
        percentiles = statistics['dws_percentiles']
        f.write(f"25th percentile: {percentiles['25']:.6e} km²\n")
        f.write(f"50th percentile: {percentiles['50']:.6e} km²\n")
        f.write(f"75th percentile: {percentiles['75']:.6e} km²\n")
        f.write(f"90th percentile: {percentiles['90']:.6e} km²\n")
        f.write(f"95th percentile: {percentiles['95']:.6e} km²\n")
        f.write(f"99th percentile: {percentiles['99']:.6e} km²\n")
        f.write("\n")

        # Visualization Summary
        f.write("VISUALIZATION SUMMARY\n")
        f.write("-"*80 + "\n")

        # Horizontal slices
        h_slices = results['plots']['horizontal_slices']
        if h_slices:
            n_success = sum(1 for s in h_slices if s['status'] == 'success')
            f.write(f"Horizontal slices: {n_success}/{len(h_slices)} successful\n")
            if n_success > 0:
                depths = [s['depth_km'] for s in h_slices if s['status'] == 'success']
                f.write(f"  Depths: {', '.join([f'{d:.2f}' for d in depths])} km\n")

        # Vertical slices
        v_slices = results['plots']['vertical_slices']
        if v_slices:
            n_success = sum(1 for s in v_slices if s['status'] == 'success')
            f.write(f"Vertical slices: {n_success}/{len(v_slices)} successful\n")

        # Coverage maps
        c_maps = results['plots']['coverage_maps']
        if c_maps:
            n_success = sum(1 for m in c_maps if m['status'] == 'success')
            f.write(f"Coverage maps: {n_success}/{len(c_maps)} successful\n")

        # Histograms
        histograms = results['plots']['histograms']
        if histograms:
            n_success = sum(1 for h in histograms if h['status'] == 'success')
            f.write(f"Histograms: {n_success}/{len(histograms)} successful\n")

        f.write("\n")

        # Footer
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    logger.info(f"DWS analysis report saved: {output_file}")
