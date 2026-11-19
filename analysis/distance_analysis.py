"""
Distance-dependent travel time analysis module.

This module analyzes how travel time residuals vary with distance,
including correlation analysis, binned statistics, and ratio analysis.
Supports both hypocentral (3D) and epicentral (2D horizontal) distances.
Supports both ground_truth and method_comparison modes.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from utils.io_utils import ensure_directory_exists
from utils.plot_utils import setup_plot_style
from utils.distance_utils import validate_distance_types
from utils.logging_utils import ProcessTimer

from .distance_data import load_and_merge_distance_data, prepare_method_comparison_data
from .distance_statistics import compute_distance_statistics
from .distance_plots import create_distance_analysis_plots
from .distance_reports import generate_distance_analysis_report

logger = logging.getLogger(__name__)

# Not converted to the spherical coordinate system: every entry point raises
SPHERICAL_UNSUPPORTED = ("Distance analysis is not available for the spherical coordinate system; this "
                         "branch is spherical only and the module still uses Cartesian coordinates")


def analyze_distance_dependent_residuals(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function for distance-dependent travel time analysis.
    Supports both hypocentral and epicentral distance analysis.
    Supports both ground_truth and method_comparison modes.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    raise NotImplementedError(SPHERICAL_UNSUPPORTED)
    logger.info("Starting distance-dependent travel time analysis")
    
    # Setup plot style
    setup_plot_style(config['analysis'])
    
    # Validate distance types
    distance_types = validate_distance_types(
        config['distance_analysis'].get('distance_types', ['hypocentral'])
    )
    
    # Check comparison mode
    comparison_mode = config['distance_analysis'].get('comparison_mode', 'ground_truth')
    logger.info(f"Comparison mode: {comparison_mode}")
    
    # Load and prepare data based on comparison mode
    if comparison_mode == 'method_comparison':
        df_merged = prepare_method_comparison_data(config)
    else:  # ground_truth mode
        df_merged = load_and_merge_distance_data(config)
    
    # Process each distance type
    all_stats = {}
    for distance_type in distance_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {distance_type.upper()} distance analysis")
        logger.info(f"{'='*60}")
        
        # Compute distance statistics
        with ProcessTimer(f"{distance_type} distance statistics computation"):
            stats = compute_distance_statistics(df_merged, config, distance_type)
        
        # Create plots
        with ProcessTimer(f"{distance_type} distance plots generation"):
            create_distance_analysis_plots(df_merged, stats, config, distance_type)
        
        # Generate report
        with ProcessTimer(f"{distance_type} distance report generation"):
            generate_distance_analysis_report(df_merged, stats, config, distance_type)
        
        all_stats[distance_type] = stats
    
    logger.info("Distance-dependent analysis completed successfully")
    return all_stats