"""
Velocity model comparison analysis module.

This module provides comprehensive comparison analysis between velocity models
including spatial projections, profile analysis, and statistical comparisons.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

from utils.io_utils import read_csv_file, write_text_file, ensure_directory_exists
from utils.plot_utils import setup_plot_style
from utils.velocity_io import read_velocity_file

from .velocity_statistics import (
    compute_velocity_differences,
    compute_velocity_statistics,
    compute_log_ratios
)
from .velocity_spatial import find_extreme_velocity_profiles, find_extreme_ratio_change_profiles
from .velocity_plots import (
    create_basic_comparison_plots,
    create_spatial_projection_plots,
    create_profile_comparison_plots,
    create_slice_heatmap_plots,
    create_ratio_analysis_plots,
    create_domain_overview_plot,
    create_plane_specific_ratio_plots
)
from .basin_boundaries import load_basin_boundaries
from .velocity_reports import generate_velocity_comparison_report

logger = logging.getLogger(__name__)


def analyze_velocity_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function for velocity model comparison analysis.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    logger.info("Starting velocity model comparison analysis")
    
    # Setup plot style
    setup_plot_style(config['analysis'])
    
    # Load and validate data (support CSV and HDF5)
    # Determine file paths (check for HDF5 first, then CSV)
    model1_path = (
        config['velocity_analysis'].get('model1_hdf5') or
        config['velocity_analysis']['model1_csv']
    )
    model2_path = (
        config['velocity_analysis'].get('model2_hdf5') or
        config['velocity_analysis']['model2_csv']
    )

    use_coarse = config['velocity_analysis'].get('use_coarse_dataset', False)
    dataset = 'velocity_coarse' if use_coarse else 'velocity'

    logger.info(f"Loading velocity models:")
    logger.info(f"  Model 1: {model1_path}")
    logger.info(f"  Model 2: {model2_path}")
    if use_coarse:
        logger.info("  Using coarse datasets for faster analysis")

    # Read velocity files
    df1 = read_velocity_file(model1_path, dataset=dataset, show_progress=False)
    df2 = read_velocity_file(model2_path, dataset=dataset, show_progress=False)

    # Validate the loaded models
    logger.info(f"Model 1: {len(df1)} rows, {len(df1.columns)} columns")
    logger.info(f"Model 2: {len(df2)} rows, {len(df2.columns)} columns")

    # Validate structure
    if len(df1) != len(df2):
        raise ValueError(f"Row count mismatch: Model 1 has {len(df1)}, Model 2 has {len(df2)}")

    required_cols = ['vp', 'vs', 'x', 'y', 'z']
    for col in required_cols:
        if col not in df1.columns:
            raise ValueError(f"Required column '{col}' missing from Model 1")
        if col not in df2.columns:
            raise ValueError(f"Required column '{col}' missing from Model 2")

    # Check coordinate consistency if enabled
    if config.get('quality_control', {}).get('check_coordinate_consistency', True):
        coord_cols = ['x', 'y', 'z']
        for col in coord_cols:
            if not np.allclose(df1[col].values, df2[col].values, rtol=1e-6):
                logger.warning(f"Coordinate column '{col}' differs between models")

    logger.info("Velocity model validation completed")

    # Compute differences and statistics
    logger.info("Computing velocity differences and statistics")
    differences = compute_velocity_differences(df1, df2, config)
    stats = compute_velocity_statistics(df1, df2, differences, config)
    
    # Compute log ratios if enabled
    if config['velocity_analysis'].get('compute_log_ratios', True):
        log_ratios = compute_log_ratios(df1, df2, config)
        differences.update(log_ratios)
        stats['log_ratios'] = {
            'vp': compute_log_ratio_statistics(log_ratios['vp_log_ratio']),
            'vs': compute_log_ratio_statistics(log_ratios['vs_log_ratio'])
        }
    
    # Find extreme profiles only when needed by downstream plots
    vel_cfg = config['velocity_analysis']
    need_extremes = (
        vel_cfg.get('enable_slice_analysis', True) or
        vel_cfg.get('compute_geometric_mean_ratios', True) or
        vel_cfg.get('enable_domain_overview', True)
    )
    need_ratio_extremes = vel_cfg.get('enable_profile_analysis', True)

    extremes = None
    if need_extremes:
        logger.info("Finding extreme velocity profiles")
        extremes = find_extreme_velocity_profiles(df1, differences, config)

    ratio_extremes = None
    if need_ratio_extremes:
        logger.info("Finding extreme ratio change profiles")
        ratio_extremes = find_extreme_ratio_change_profiles(df1, df2, config)

    # Create plots
    logger.info("Creating velocity comparison plots")
    create_all_velocity_plots(df1, df2, differences, extremes, ratio_extremes, config)
    
    # Generate summary report
    logger.info("Generating velocity comparison report")
    generate_velocity_comparison_report(df1, df2, differences, stats, extremes, config)
    
    logger.info("Velocity model comparison analysis completed")
    return stats


def load_and_validate_velocity_models(model1_csv: str, model2_csv: str, 
                                     config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and validate velocity model CSV files.
    
    Args:
        model1_csv: Path to first model CSV
        model2_csv: Path to second model CSV
        config: Configuration dictionary
        
    Returns:
        Tuple of (model1_df, model2_df)
    """
    logger.info("Loading and validating velocity models")
    
    df1 = read_csv_file(model1_csv)
    df2 = read_csv_file(model2_csv)
    
    logger.info(f"Model 1: {len(df1)} rows, {len(df1.columns)} columns")
    logger.info(f"Model 2: {len(df2)} rows, {len(df2.columns)} columns")
    
    # Validate structure
    if len(df1) != len(df2):
        raise ValueError(f"Row count mismatch: Model 1 has {len(df1)}, Model 2 has {len(df2)}")
    
    required_cols = ['vp', 'vs', 'x', 'y', 'z']
    for col in required_cols:
        if col not in df1.columns:
            raise ValueError(f"Required column '{col}' missing from Model 1")
        if col not in df2.columns:
            raise ValueError(f"Required column '{col}' missing from Model 2")
    
    # Check coordinate consistency
    if config['quality_control'].get('check_coordinate_consistency', True):
        coord_cols = ['x', 'y', 'z']
        for col in coord_cols:
            if not np.allclose(df1[col].values, df2[col].values, rtol=1e-6):
                logger.warning(f"Coordinate column '{col}' differs between models")

    logger.info("Velocity model validation completed")
    return df1, df2


def compute_log_ratio_statistics(log_ratios: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics for logarithmic ratios.
    
    Args:
        log_ratios: Array of log ratios
        
    Returns:
        Dictionary of statistics
    """
    return {
        'mean': float(np.mean(log_ratios)),
        'std': float(np.std(log_ratios)),
        'min': float(np.min(log_ratios)),
        'max': float(np.max(log_ratios)),
        'abs_mean': float(np.mean(np.abs(log_ratios))),
        'rms': float(np.sqrt(np.mean(log_ratios**2))),
        'equivalent_percent_change': float((np.exp(np.mean(log_ratios)) - 1) * 100),
        'points_increased': int(np.sum(log_ratios > 0)),
        'points_decreased': int(np.sum(log_ratios < 0)),
        'points_unchanged': int(np.sum(np.abs(log_ratios) < 1e-6))
    }


def create_all_velocity_plots(df1: pd.DataFrame, df2: pd.DataFrame,
                             differences: Dict[str, np.ndarray],
                             extremes: Dict[str, Any],
                             ratio_extremes: Dict[str, Any],
                             config: Dict[str, Any]) -> None:
    """
    Create all velocity comparison plots.

    Args:
        df1: First velocity model dataframe
        df2: Second velocity model dataframe
        differences: Differences dictionary
        extremes: Extreme profiles dictionary (difference-based)
        ratio_extremes: Extreme profiles dictionary (ratio-based)
        config: Configuration dictionary
    """
    output_dir = Path(config['velocity_analysis']['output_dir'])
    plots_dir = output_dir / 'plots'
    ensure_directory_exists(plots_dir)

    output_format = config['analysis'].get('output_format', 'png')

    vel_cfg = config['velocity_analysis']

    # Pre-load basin boundaries once and cache on config to avoid repeated disk reads
    if vel_cfg.get('plot_basin_boundaries', False):
        basin_config = config.get('basin_boundaries', {})
        _basin_data = load_basin_boundaries(basin_config)
        if _basin_data:
            logger.info(f"Loaded {len(_basin_data)} basin boundaries for XY plane overlays")
            config['_basin_data_cache'] = _basin_data
            config['_basin_config_cache'] = basin_config
        else:
            logger.warning("plot_basin_boundaries is true but no basin data was loaded — check basin_boundaries.data_dir and basins list")

    # 1. Basic comparison plots (scatter + histograms)
    if vel_cfg.get('enable_basic_comparison_plots', True):
        logger.info("Creating basic comparison plots")
        create_basic_comparison_plots(differences, plots_dir, output_format, config)

    # 2. Spatial projection plots (RMS)
    if vel_cfg.get('enable_spatial_projections', True):
        logger.info("Creating spatial projection plots")
        create_spatial_projection_plots(df1, differences, plots_dir, output_format, config)

    # 3. Vertical profile plots (using ratio-based extremes)
    if vel_cfg.get('enable_profile_analysis', True):
        logger.info("Creating profile comparison plots")
        create_profile_comparison_plots(df1, df2, differences, ratio_extremes, plots_dir, output_format, config)

    # 4. Horizontal slice heatmaps (using difference-based extremes)
    if vel_cfg.get('enable_slice_analysis', True):
        logger.info("Creating slice heatmap plots")
        create_slice_heatmap_plots(df1, df2, differences, extremes, plots_dir, output_format, config)

    # 5. Ratio analysis plots (using difference-based extremes)
    if vel_cfg.get('compute_geometric_mean_ratios', True):
        logger.info("Creating ratio analysis plots")
        create_ratio_analysis_plots(df1, df2, differences, extremes, plots_dir, output_format, config)

    # 6. Domain overview (using difference-based extremes)
    if vel_cfg.get('enable_domain_overview', True):
        logger.info("Creating domain overview plot")
        create_domain_overview_plot(df1, extremes, plots_dir, output_format, config)

    # 7. Plane-specific ratio plots (optional)
    if vel_cfg.get('enable_plane_ratio_plots', False):
        logger.info("Creating plane-specific ratio plots")
        create_plane_specific_ratio_plots(df1, df2, plots_dir, output_format, config)

    logger.info(f"All plots saved to: {plots_dir}")