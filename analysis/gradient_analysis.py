"""
Gradient and search direction analysis module.

This module provides comprehensive analysis of gradients and search directions
from LATTE tomographic inversion, including statistical analysis, spatial patterns,
and convergence diagnostics.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from utils.io_utils import read_csv_file, write_text_file, ensure_directory_exists
from utils.plot_utils import setup_plot_style, get_color_scheme
from . import apply_depth_filter

logger = logging.getLogger(__name__)


def analyze_gradients_and_search(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function for gradient and search direction analysis.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    gradient_cfg = config.get('gradient_analysis', {})
    if not gradient_cfg.get('enabled', False):
        logger.info("Gradient and search direction analysis is disabled.")
        return {}
    
    logger.info("Starting gradient and search direction analysis.")
    
    # Setup plot style
    setup_plot_style(config.get('analysis', {}))
    
    # Determine analysis type(s)
    analysis_type = gradient_cfg.get('analysis_type', ['velocity'])
    if isinstance(analysis_type, str):
        analysis_type = [analysis_type.lower()]
    else:
        analysis_type = [t.lower() for t in analysis_type]
    
    logger.info(f"Analysis types: {analysis_type}")
    
    results = {}
    
    if 'velocity' in analysis_type:
        logger.info("Running velocity gradient analysis...")
        results['velocity'] = analyze_velocity_gradients(config)
    
    if 'source' in analysis_type:
        logger.info("Running source gradient analysis...")
        results['source'] = analyze_source_gradients(config)
    
    # Handle unknown types gracefully
    known_types = {'velocity', 'source'}
    unknown_types = set(analysis_type) - known_types
    if unknown_types:
        logger.warning(f"Unknown analysis types ignored: {unknown_types}")
    
    return results



def analyze_velocity_gradients(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze velocity model gradients and search directions.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of statistics and results
    """
    logger.info("Analyzing velocity gradients and search directions")
    
    input_csv = config['gradient_analysis']['velocity_input_csv']
    output_dir = Path(config['gradient_analysis']['velocity_output_dir'])
    
    logger.info(f"Loading velocity data from: {input_csv}")
    df = read_csv_file(input_csv)

    # Apply depth filtering (velocity gradients use Z_grid_km for depth)
    max_depth_km = config.get('analysis', {}).get('max_depth_km', None)
    if max_depth_km is not None:
        df = apply_depth_filter(df, max_depth_km, depth_column='Z_grid_km',
                               data_type="velocity gradient data")

    # Determine which fields are available
    available_fields = _detect_available_gradient_fields(df, 'velocity')
    
    if not available_fields['gradients'] and not available_fields['search']:
        logger.warning("No gradient or search direction fields found in data")
        return {}
    
    logger.info(f"Found gradient fields: {available_fields['gradients']}")
    logger.info(f"Found search direction fields: {available_fields['search']}")
    
    # Compute statistics
    logger.info("Computing gradient statistics")
    stats = compute_velocity_gradient_statistics(df, available_fields, config)
    
    # Create plots
    logger.info("Creating gradient analysis plots")
    from .gradient_plots import create_velocity_gradient_plots
    create_velocity_gradient_plots(df, available_fields, stats, config)
    
    # Generate report
    if config['gradient_analysis'].get('generate_report', True):
        logger.info("Generating gradient analysis report")
        from .gradient_reports import generate_velocity_gradient_report
        generate_velocity_gradient_report(df, available_fields, stats, config)
    
    logger.info("Velocity gradient analysis completed")
    return stats


def analyze_source_gradients(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze source parameter gradients and search directions.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of statistics and results
    """
    logger.info("Analyzing source parameter gradients and search directions")
    
    input_csv = config['gradient_analysis']['source_input_csv']
    output_dir = Path(config['gradient_analysis']['source_output_dir'])
    
    logger.info(f"Loading source data from: {input_csv}")
    df = read_csv_file(input_csv)

    # Apply depth filtering (source gradients use event_z_transformed for depth)
    max_depth_km = config.get('analysis', {}).get('max_depth_km', None)
    if max_depth_km is not None:
        df = apply_depth_filter(df, max_depth_km, depth_column='event_z_transformed',
                               data_type="source gradient data")

    # Determine which fields are available
    available_fields = _detect_available_gradient_fields(df, 'source')
    
    if not available_fields['gradients'] and not available_fields['search']:
        logger.warning("No gradient or search direction fields found in data")
        return {}
    
    logger.info(f"Found gradient fields: {available_fields['gradients']}")
    logger.info(f"Found search direction fields: {available_fields['search']}")
    
    # Compute statistics
    logger.info("Computing source gradient statistics")
    stats = compute_source_gradient_statistics(df, available_fields, config)
    
    # Create plots
    logger.info("Creating source gradient plots")
    from .gradient_plots import create_source_gradient_plots
    create_source_gradient_plots(df, available_fields, stats, config)
    
    # Generate report
    if config['gradient_analysis'].get('generate_report', True):
        logger.info("Generating source gradient report")
        from .gradient_reports import generate_source_gradient_report
        generate_source_gradient_report(df, available_fields, stats, config)
    
    logger.info("Source gradient analysis completed")
    return stats


def _detect_available_gradient_fields(df: pd.DataFrame, 
                                      analysis_type: str) -> Dict[str, List[str]]:
    """
    Detect which gradient and search direction fields are available in the data.
    
    Args:
        df: Input dataframe
        analysis_type: 'velocity' or 'source'
        
    Returns:
        Dictionary with 'gradients' and 'search' lists
    """
    available = {'gradients': [], 'search': []}
    
    if analysis_type == 'velocity':
        # Check for velocity gradient fields
        possible_grads = ['grad_vp', 'grad_vs']
        possible_search = ['srch_vp', 'srch_vs']
    else:  # source
        # Check for source gradient fields
        possible_grads = ['grad_sx', 'grad_sy', 'grad_sz', 'grad_st0']
        possible_search = ['srch_sx', 'srch_sy', 'srch_sz', 'srch_st0']
    
    for field in possible_grads:
        if field in df.columns:
            # Check if column has non-NaN values
            if df[field].notna().any():
                available['gradients'].append(field)
    
    for field in possible_search:
        if field in df.columns:
            # Check if column has non-NaN values
            if df[field].notna().any():
                available['search'].append(field)
    
    return available


def compute_velocity_gradient_statistics(df: pd.DataFrame, 
                                        available_fields: Dict[str, List[str]],
                                        config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for velocity gradients and search directions.
    
    Args:
        df: Velocity dataframe with gradient/search fields
        available_fields: Dictionary of available fields
        config: Configuration dictionary
        
    Returns:
        Dictionary of statistics
    """
    stats = {'type': 'velocity', 'available_fields': available_fields}
    
    # Gradient statistics
    if available_fields['gradients']:
        stats['gradients'] = {}
        for field in available_fields['gradients']:
            param = field.replace('grad_', '')
            data = df[field].dropna()
            
            if len(data) > 0:
                stats['gradients'][param] = {
                    'count': int(len(data)),
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'median': float(data.median()),
                    'abs_mean': float(data.abs().mean()),
                    'abs_max': float(data.abs().max()),
                    'rms': float(np.sqrt(np.mean(data**2))),
                    'positive_count': int((data > 0).sum()),
                    'negative_count': int((data < 0).sum()),
                    'zero_count': int((np.abs(data) < 1e-10).sum())
                }
    
    # Search direction statistics
    if available_fields['search']:
        stats['search'] = {}
        for field in available_fields['search']:
            param = field.replace('srch_', '')
            data = df[field].dropna()
            
            if len(data) > 0:
                stats['search'][param] = {
                    'count': int(len(data)),
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'median': float(data.median()),
                    'abs_mean': float(data.abs().mean()),
                    'abs_max': float(data.abs().max()),
                    'rms': float(np.sqrt(np.mean(data**2))),
                    'positive_count': int((data > 0).sum()),
                    'negative_count': int((data < 0).sum())
                }
    
    # Gradient-search alignment (if both available)
    if available_fields['gradients'] and available_fields['search']:
        stats['alignment'] = _compute_gradient_search_alignment(df, available_fields)
    
    # Depth-dependent statistics for velocity
    if 'Z_grid_km' in df.columns:
        stats['depth_dependent'] = _compute_depth_dependent_gradient_stats(
            df, available_fields, config
        )
    
    return stats


def compute_source_gradient_statistics(df: pd.DataFrame, 
                                      available_fields: Dict[str, List[str]],
                                      config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for source parameter gradients.
    
    Args:
        df: Source dataframe with gradient/search fields
        available_fields: Dictionary of available fields
        config: Configuration dictionary
        
    Returns:
        Dictionary of statistics
    """
    stats = {'type': 'source', 'available_fields': available_fields}
    
    # Gradient statistics
    if available_fields['gradients']:
        stats['gradients'] = {}
        for field in available_fields['gradients']:
            param = field.replace('grad_', '')
            data = df[field].dropna()
            
            if len(data) > 0:
                stats['gradients'][param] = {
                    'count': int(len(data)),
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'median': float(data.median()),
                    'abs_mean': float(data.abs().mean()),
                    'abs_max': float(data.abs().max()),
                    'rms': float(np.sqrt(np.mean(data**2))),
                    'positive_count': int((data > 0).sum()),
                    'negative_count': int((data < 0).sum()),
                    'zero_count': int((np.abs(data) < 1e-10).sum())
                }
    
    # Search direction statistics
    if available_fields['search']:
        stats['search'] = {}
        for field in available_fields['search']:
            param = field.replace('srch_', '')
            data = df[field].dropna()
            
            if len(data) > 0:
                stats['search'][param] = {
                    'count': int(len(data)),
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'median': float(data.median()),
                    'abs_mean': float(data.abs().mean()),
                    'abs_max': float(data.abs().max()),
                    'rms': float(np.sqrt(np.mean(data**2)))
                }
    
    # Gradient-search alignment
    if available_fields['gradients'] and available_fields['search']:
        stats['alignment'] = _compute_gradient_search_alignment(df, available_fields)
    
    # Spatial gradients for 3D source parameters
    if 'grad_sx' in available_fields['gradients'] and 'grad_sy' in available_fields['gradients']:
        stats['spatial_gradients'] = _compute_spatial_gradient_magnitudes(df, available_fields)
    
    return stats


def _compute_gradient_search_alignment(df: pd.DataFrame,
                                      available_fields: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Compute alignment between gradients and search directions.
    
    Args:
        df: Dataframe with gradient and search fields
        available_fields: Available fields dictionary
        
    Returns:
        Dictionary of alignment statistics
    """
    alignment = {}
    
    for grad_field in available_fields['gradients']:
        param = grad_field.replace('grad_', '')
        search_field = f'srch_{param}'
        
        if search_field in available_fields['search']:
            # Get valid data
            valid_mask = df[grad_field].notna() & df[search_field].notna()
            grad_data = df.loc[valid_mask, grad_field]
            search_data = df.loc[valid_mask, search_field]
            
            if len(grad_data) > 0:
                # Compute dot product (normalized)
                dot_product = (grad_data * search_data).sum()
                grad_norm = np.sqrt((grad_data ** 2).sum())
                search_norm = np.sqrt((search_data ** 2).sum())
                
                if grad_norm > 0 and search_norm > 0:
                    cosine_similarity = dot_product / (grad_norm * search_norm)
                else:
                    cosine_similarity = 0.0
                
                # Count alignment direction
                same_sign = ((grad_data > 0) & (search_data > 0)) | ((grad_data < 0) & (search_data < 0))
                
                alignment[param] = {
                    'cosine_similarity': float(cosine_similarity),
                    'same_sign_count': int(same_sign.sum()),
                    'opposite_sign_count': int((~same_sign).sum()),
                    'same_sign_percent': float(same_sign.sum() / len(same_sign) * 100)
                }
    
    return alignment


def _compute_depth_dependent_gradient_stats(df: pd.DataFrame,
                                           available_fields: Dict[str, List[str]],
                                           config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute depth-dependent gradient statistics.
    
    Args:
        df: Dataframe with depth coordinate
        available_fields: Available fields dictionary
        config: Configuration dictionary
        
    Returns:
        Dictionary of depth-dependent statistics
    """
    depth_stats = {}
    
    n_bins = config['gradient_analysis'].get('depth_bins', 10)
    
    # Create depth bins
    df_copy = df.copy()
    df_copy['depth_bin'] = pd.cut(df_copy['Z_grid_km'], bins=n_bins)
    
    # Compute statistics for each gradient field
    for grad_field in available_fields['gradients']:
        param = grad_field.replace('grad_', '')
        
        bin_stats = df_copy.groupby('depth_bin')[grad_field].agg([
            'count', 'mean', 'std', 
            lambda x: np.sqrt(np.mean(x**2)),  # RMS
            lambda x: x.abs().mean()  # Mean absolute
        ]).round(6)
        
        bin_stats.columns = ['count', 'mean', 'std', 'rms', 'abs_mean']
        
        # Filter bins with sufficient data
        min_points = config['gradient_analysis'].get('min_points_per_bin', 10)
        bin_stats = bin_stats[bin_stats['count'] >= min_points]
        
        if not bin_stats.empty:
            depth_stats[param] = bin_stats.to_dict('index')
    
    return depth_stats


def _compute_spatial_gradient_magnitudes(df: pd.DataFrame,
                                        available_fields: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Compute spatial gradient magnitudes for source parameters.
    
    Args:
        df: Dataframe with spatial gradient fields
        available_fields: Available fields dictionary
        
    Returns:
        Dictionary of spatial gradient statistics
    """
    spatial_stats = {}
    
    # Horizontal gradient magnitude
    if 'grad_sx' in available_fields['gradients'] and 'grad_sy' in available_fields['gradients']:
        valid_mask = df['grad_sx'].notna() & df['grad_sy'].notna()
        grad_sx = df.loc[valid_mask, 'grad_sx']
        grad_sy = df.loc[valid_mask, 'grad_sy']
        
        horiz_grad_mag = np.sqrt(grad_sx**2 + grad_sy**2)
        
        spatial_stats['horizontal'] = {
            'mean_magnitude': float(horiz_grad_mag.mean()),
            'max_magnitude': float(horiz_grad_mag.max()),
            'std_magnitude': float(horiz_grad_mag.std())
        }
    
    # 3D gradient magnitude
    if all(f'grad_{p}' in available_fields['gradients'] for p in ['sx', 'sy', 'sz']):
        valid_mask = df['grad_sx'].notna() & df['grad_sy'].notna() & df['grad_sz'].notna()
        grad_sx = df.loc[valid_mask, 'grad_sx']
        grad_sy = df.loc[valid_mask, 'grad_sy']
        grad_sz = df.loc[valid_mask, 'grad_sz']
        
        total_grad_mag = np.sqrt(grad_sx**2 + grad_sy**2 + grad_sz**2)
        
        spatial_stats['3d'] = {
            'mean_magnitude': float(total_grad_mag.mean()),
            'max_magnitude': float(total_grad_mag.max()),
            'std_magnitude': float(total_grad_mag.std())
        }
    
    return spatial_stats