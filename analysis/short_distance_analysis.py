"""
Short distance travel time analysis module.

This module analyzes travel time residuals for short source-receiver distances,
with support for both ground_truth and method_comparison modes.
Supports both hypocentral (3D) and epicentral (2D horizontal) distances.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any

from utils.io_utils import ensure_directory_exists
from utils.plot_utils import setup_plot_style
from utils.distance_utils import validate_distance_types
from utils.logging_utils import ProcessTimer

from .short_distance_data import (
    load_and_filter_short_distance_data,
    prepare_short_distance_comparison_data
)
from .short_distance_statistics import compute_short_distance_statistics
from .short_distance_plots import create_short_distance_plots
from .short_distance_reports import generate_short_distance_report

logger = logging.getLogger(__name__)

# Not converted to the spherical coordinate system: every entry point raises
SPHERICAL_UNSUPPORTED = ("Short distance analysis is not available for the spherical coordinate system; this "
                         "branch is spherical only and the module still uses Cartesian coordinates")


def analyze_short_distance_residuals(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function for short distance travel time analysis.
    Supports both hypocentral and epicentral distance analysis.
    Supports both ground_truth and method_comparison modes.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    raise NotImplementedError(SPHERICAL_UNSUPPORTED)
    logger.info("Starting short distance travel time analysis")
    
    # Setup plot style
    setup_plot_style(config['analysis'])
    
    # Validate distance types
    distance_types = validate_distance_types(
        config['short_distance_analysis'].get('distance_types', ['hypocentral'])
    )
    
    # Check comparison mode
    comparison_mode = config['short_distance_analysis'].get('comparison_mode', 'ground_truth')
    distance_cutoff = config['short_distance_analysis']['distance_cutoff_km']
    
    logger.info(f"Comparison mode: {comparison_mode}")
    logger.info(f"Distance cutoff: {distance_cutoff} km")
    
    # Load and prepare data based on comparison mode
    if comparison_mode == 'method_comparison':
        df_filtered = prepare_short_distance_comparison_data(config)
    else:  # ground_truth mode
        df_filtered = load_and_filter_short_distance_data(config)
    
    # Process each distance type
    all_stats = {}
    for distance_type in distance_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {distance_type.upper()} short distance analysis")
        logger.info(f"{'='*60}")
        
        # Compute statistics
        with ProcessTimer(f"{distance_type} short distance statistics computation"):
            stats = compute_short_distance_statistics(df_filtered, config, distance_type)
        
        # Create plots
        with ProcessTimer(f"{distance_type} short distance plots generation"):
            create_short_distance_plots(df_filtered, stats, config, distance_type)
        
        # Generate report
        with ProcessTimer(f"{distance_type} short distance report generation"):
            generate_short_distance_report(df_filtered, stats, config, distance_type)
        
        all_stats[distance_type] = stats
    
    logger.info("Short distance analysis completed successfully")
    return all_stats