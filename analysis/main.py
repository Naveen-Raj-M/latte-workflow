"""
Main entry point for LATTE analysis module.

This script provides the main interface for running different analysis modules
with Hydra configuration management.

Updated to support method comparison mode for distance and short distance analysis.
"""

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import sys

from utils.logging_utils import setup_logging, ProcessTimer
from utils.io_utils import ensure_directory_exists

# Import analysis functions
from .traveltime_analysis import analyze_travel_time_residuals
from .velocity_analysis import analyze_velocity_models
from .distance_analysis import analyze_distance_dependent_residuals
from .short_distance_analysis import analyze_short_distance_residuals
from .source_analysis import analyze_source_corrections
from .gradient_analysis import analyze_gradients_and_search

logger = logging.getLogger(__name__)


def run_travel_time_analysis(cfg: DictConfig) -> None:
    """Run travel time residual analysis."""
    if not cfg.travel_time_analysis.enabled:
        logger.info("Travel time analysis disabled, skipping")
        return
    
    logger.info("=" * 60)
    logger.info("STARTING TRAVEL TIME ANALYSIS")
    logger.info("=" * 60)
    
    with ProcessTimer("Travel time analysis"):
        try:
            ensure_directory_exists(cfg.travel_time_analysis.output_dir)
            results = analyze_travel_time_residuals(OmegaConf.to_container(cfg, resolve=True))
            logger.info("Travel time analysis completed successfully")
            logger.info(f"Results saved to: {cfg.travel_time_analysis.output_dir}")
        except Exception as e:
            logger.error(f"Travel time analysis failed: {e}")
            raise


def run_velocity_analysis(cfg: DictConfig) -> None:
    """Run velocity model comparison analysis."""
    if not cfg.velocity_analysis.enabled:
        logger.info("Velocity analysis disabled, skipping")
        return
    
    logger.info("=" * 60)
    logger.info("STARTING VELOCITY ANALYSIS")
    logger.info("=" * 60)
    
    with ProcessTimer("Velocity analysis"):
        try:
            ensure_directory_exists(cfg.velocity_analysis.output_dir)
            results = analyze_velocity_models(OmegaConf.to_container(cfg, resolve=True))
            logger.info("Velocity analysis completed successfully")
            logger.info(f"Results saved to: {cfg.velocity_analysis.output_dir}")
        except Exception as e:
            logger.error(f"Velocity analysis failed: {e}")
            raise


def run_distance_analysis(cfg: DictConfig) -> None:
    """Run distance-dependent residual analysis."""
    if not cfg.distance_analysis.enabled:
        logger.info("Distance analysis disabled, skipping")
        return
    
    logger.info("=" * 60)
    logger.info("STARTING DISTANCE ANALYSIS")
    logger.info("=" * 60)
    
    # Get distance types from config
    distance_types = cfg.distance_analysis.get('distance_types', ['hypocentral'])
    if isinstance(distance_types, str):
        distance_types = [distance_types]
    
    comparison_mode = cfg.distance_analysis.get('comparison_mode', 'ground_truth')
    
    logger.info(f"Comparison mode: {comparison_mode}")
    logger.info(f"Distance types to analyze: {distance_types}")
    
    with ProcessTimer("Distance analysis"):
        try:
            ensure_directory_exists(cfg.distance_analysis.output_dir)
            results = analyze_distance_dependent_residuals(OmegaConf.to_container(cfg, resolve=True))
            
            logger.info("Distance analysis completed successfully")
            logger.info(f"Results saved to: {cfg.distance_analysis.output_dir}")
            
            # Log summary for each distance type
            for distance_type in distance_types:
                if distance_type in results:
                    stats = results[distance_type]
                    logger.info(f"{distance_type.upper()} distance summary:")
                    logger.info(f"  Total points: {stats['overall']['count']}")
                    logger.info(f"  Mean distance: {stats['overall']['distance_mean']:.3f} km")
                    logger.info(f"  Distance-residual correlation: {stats['overall']['residual_distance_correlation']:.4f}")
                    
        except Exception as e:
            logger.error(f"Distance analysis failed: {e}")
            raise


def run_short_distance_analysis(cfg: DictConfig) -> None:
    """Run short distance residual analysis."""
    if not cfg.short_distance_analysis.enabled:
        logger.info("Short distance analysis disabled, skipping")
        return
    
    logger.info("=" * 60)
    logger.info("STARTING SHORT DISTANCE ANALYSIS")
    logger.info("=" * 60)
    
    # Get distance types from config
    distance_types = cfg.short_distance_analysis.get('distance_types', ['hypocentral'])
    if isinstance(distance_types, str):
        distance_types = [distance_types]
    
    comparison_mode = cfg.short_distance_analysis.get('comparison_mode', 'ground_truth')
    distance_cutoff = cfg.short_distance_analysis.distance_cutoff_km
    
    logger.info(f"Comparison mode: {comparison_mode}")
    logger.info(f"Distance types to analyze: {distance_types}")
    logger.info(f"Distance cutoff: {distance_cutoff} km")
    
    with ProcessTimer("Short distance analysis"):
        try:
            ensure_directory_exists(cfg.short_distance_analysis.output_dir)
            results = analyze_short_distance_residuals(OmegaConf.to_container(cfg, resolve=True))
            
            logger.info("Short distance analysis completed successfully")
            logger.info(f"Results saved to: {cfg.short_distance_analysis.output_dir}")
            
            # Log summary for each distance type
            for distance_type in distance_types:
                if distance_type in results:
                    stats = results[distance_type]
                    logger.info(f"{distance_type.upper()} short distance summary:")
                    logger.info(f"  Total points: {stats['overall']['count']}")
                    logger.info(f"  Unique events: {stats['overall']['unique_events']}")
                    logger.info(f"  Unique stations: {stats['overall']['unique_stations']}")
                    logger.info(f"  RMS residual: {stats['overall']['residual_rms']:.4f} s")
                    
        except Exception as e:
            logger.error(f"Short distance analysis failed: {e}")
            raise


def run_source_analysis(cfg: DictConfig) -> None:
    """Run source correction analysis."""
    if not cfg.source_analysis.enabled:
        logger.info("Source correction analysis disabled, skipping")
        return
    
    logger.info("=" * 60)
    logger.info("STARTING SOURCE CORRECTION ANALYSIS")
    logger.info("=" * 60)
    
    with ProcessTimer("Source correction analysis"):
        try:
            ensure_directory_exists(cfg.source_analysis.output_dir)
            results = analyze_source_corrections(OmegaConf.to_container(cfg, resolve=True))
            logger.info("Source correction analysis completed successfully")
            logger.info(f"Results saved to: {cfg.source_analysis.output_dir}")
        except Exception as e:
            logger.error(f"Source correction analysis failed: {e}")
            raise


def run_gradient_analysis(cfg: DictConfig) -> None:
    """Run gradient and search direction analysis."""
    if not cfg.gradient_analysis.get('enabled', False):
        logger.info("Gradient and search direction analysis disabled, skipping")
        return
    
    logger.info("=" * 60)
    logger.info("STARTING GRADIENT AND SEARCH DIRECTION ANALYSIS")
    logger.info("=" * 60)
    
    with ProcessTimer("Gradient analysis"):
        try:
            ensure_directory_exists(cfg.gradient_analysis.velocity_output_dir)
            ensure_directory_exists(cfg.gradient_analysis.source_output_dir)
            results = analyze_gradients_and_search(OmegaConf.to_container(cfg, resolve=True))
            logger.info("Gradient analysis completed successfully")
            logger.info(f"Velocity Gradient Results saved to: {cfg.gradient_analysis.velocity_output_dir}")
            logger.info(f"Source Gradient Results saved to: {cfg.gradient_analysis.source_output_dir}")
        except Exception as e:
            logger.error(f"Gradient analysis failed: {e}")
            raise


def run_single_function(cfg: DictConfig, function_name: str) -> None:
    """
    Run a single analysis function.

    Args:
        cfg: Hydra configuration
        function_name: Name of the function to run
    """
    function_map = {
        'travel_time_analysis': run_travel_time_analysis,
        'velocity_analysis': run_velocity_analysis,
        'distance_analysis': run_distance_analysis,
        'short_distance_analysis': run_short_distance_analysis,
        'source_analysis': run_source_analysis,
        'gradient_analysis': run_gradient_analysis,
    }

    if function_name not in function_map:
        logger.error(f"Unknown function: {function_name}")
        logger.error(f"Available functions: {list(function_map.keys())}")
        sys.exit(1)

    logger.info(f"Running single function: {function_name}")

    # Initialize configuration settings
    initialize_analysis_configuration(cfg)

    function_map[function_name](cfg)


def initialize_analysis_configuration(cfg: DictConfig) -> None:
    """
    Initialize and validate analysis configuration settings.

    Args:
        cfg: Hydra configuration
    """
    logger.info("Initializing analysis configuration...")

    # Initialize and log analysis settings
    logger.info(f"Output format: {cfg.analysis.output_format}")
    logger.info(f"DPI: {cfg.analysis.dpi}")
    logger.info(f"Save intermediate results: {cfg.analysis.save_intermediate_results}")
    logger.info(f"Phase type: {cfg.analysis.get('phase_type', 'both')}")
    if cfg.analysis.get('max_depth_km'):
        logger.info(f"Max depth filter: {cfg.analysis.max_depth_km} km")

    # Initialize plot style settings
    plot_style = cfg.analysis.plot_style
    logger.info("Plot style configuration:")
    logger.info(f"  Font family: {plot_style.font_family}")
    logger.info(f"  Font size: {plot_style.font_size}")
    logger.info(f"  Max plot points: {plot_style.max_plot_points}")

    # Initialize paths configuration
    if hasattr(cfg, 'paths'):
        logger.info("Path configuration:")
        logger.info(f"  Input directory: {cfg.paths.input_dir}")
        logger.info(f"  Output base directory: {cfg.paths.output_base_dir}")
        logger.info(f"  Plots subdirectory: {cfg.paths.plots_subdir}")
        logger.info(f"  Statistics subdirectory: {cfg.paths.stats_subdir}")
        logger.info(f"  Reports subdirectory: {cfg.paths.reports_subdir}")

        # Ensure output directories exist
        from pathlib import Path
        output_base = Path(cfg.paths.output_base_dir)
        ensure_directory_exists(output_base)
        ensure_directory_exists(output_base / cfg.paths.plots_subdir)
        ensure_directory_exists(output_base / cfg.paths.stats_subdir)
        ensure_directory_exists(output_base / cfg.paths.reports_subdir)

    # Initialize performance settings
    if hasattr(cfg, 'performance'):
        logger.info("Performance configuration:")
        logger.info(f"  Max memory: {cfg.performance.max_memory_gb} GB")
        logger.info(f"  Subsample large datasets: {cfg.performance.subsample_large_datasets}")
        logger.info(f"  Subsample threshold: {cfg.performance.subsample_threshold} points")
        logger.info(f"  Parallel processing: {cfg.performance.parallel_processing}")
        if cfg.performance.parallel_processing:
            logger.info(f"  Number of jobs: {cfg.performance.n_jobs}")

    # Initialize logging settings
    if hasattr(cfg, 'logging'):
        logger.info("Logging configuration:")
        logger.info(f"  Log to file: {cfg.logging.log_to_file}")
        if cfg.logging.log_to_file:
            logger.info(f"  Log file prefix: {cfg.logging.log_file_prefix}")
        logger.info(f"  Detailed progress: {cfg.logging.detailed_progress}")
        logger.info(f"  Profile memory: {cfg.logging.profile_memory}")
        logger.info(f"  Profile timing: {cfg.logging.profile_timing}")

    # Initialize quality control settings
    if hasattr(cfg, 'quality_control'):
        logger.info("Quality control configuration:")
        logger.info(f"  Validate input data: {cfg.quality_control.validate_input_data}")
        logger.info(f"  Check coordinate consistency: {cfg.quality_control.check_coordinate_consistency}")
        logger.info(f"  Warn on missing data: {cfg.quality_control.warn_missing_data}")
        logger.info(f"  Error on empty results: {cfg.quality_control.error_on_empty_results}")

    logger.info("Configuration initialization complete")


def run_analysis_pipeline(cfg: DictConfig) -> None:
    """
    Run the complete analysis pipeline.

    Args:
        cfg: Hydra configuration
    """
    logger.info("Starting LATTE analysis pipeline")
    logger.info(f"Configuration mode: {cfg.get('mode', 'full_pipeline')}")

    # Initialize configuration settings
    initialize_analysis_configuration(cfg)
    
    with ProcessTimer("Complete analysis pipeline"):
        analyses = [
            ('Travel Time Analysis', run_travel_time_analysis),
            ('Velocity Analysis', run_velocity_analysis),
            ('Distance Analysis', run_distance_analysis),
            ('Short Distance Analysis', run_short_distance_analysis),
            ('Source Correction Analysis', run_source_analysis),
            ('Gradient Analysis', run_gradient_analysis),
        ]
        
        completed_analyses = []
        failed_analyses = []
        
        for analysis_name, analysis_func in analyses:
            try:
                analysis_func(cfg)
                completed_analyses.append(analysis_name)
            except Exception as e:
                logger.error(f"{analysis_name} failed: {e}")
                failed_analyses.append((analysis_name, str(e)))
                
                if cfg.get('strict_mode', False):
                    logger.error("Strict mode enabled, stopping pipeline")
                    break
                else:
                    logger.warning(f"Continuing with remaining analyses...")
        
        # Summary report
        logger.info("=" * 60)
        logger.info("ANALYSIS PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Completed analyses: {len(completed_analyses)}")
        for analysis in completed_analyses:
            logger.info(f"  ✓ {analysis}")
        
        if failed_analyses:
            logger.info(f"Failed analyses: {len(failed_analyses)}")
            for analysis, error in failed_analyses:
                logger.info(f"  ✗ {analysis}: {error}")
        
        if not failed_analyses:
            logger.info("All enabled analyses completed successfully!")
        elif len(completed_analyses) > 0:
            logger.warning("Pipeline completed with some failures")
        else:
            logger.error("All analyses failed")
            sys.exit(1)


def validate_configuration(cfg: DictConfig) -> None:
    """
    Validate the configuration for common issues.
    
    Args:
        cfg: Hydra configuration
    """
    logger.info("Validating configuration...")
    
    # Check if at least one analysis is enabled
    enabled_analyses = []
    if cfg.travel_time_analysis.get('enabled', False):
        enabled_analyses.append('travel_time_analysis')
    if cfg.velocity_analysis.get('enabled', False):
        enabled_analyses.append('velocity_analysis')
    if cfg.distance_analysis.get('enabled', False):
        enabled_analyses.append('distance_analysis')
    if cfg.short_distance_analysis.get('enabled', False):
        enabled_analyses.append('short_distance_analysis')
    if cfg.source_analysis.get('enabled', False):
        enabled_analyses.append('source_analysis')
    if cfg.gradient_analysis.get('enabled', False):
        enabled_analyses.append('gradient_analysis')
    
    if not enabled_analyses:
        logger.error("No analysis modules are enabled!")
        logger.error("Please enable at least one analysis module in the configuration")
        sys.exit(1)
    
    logger.info(f"Enabled analyses: {enabled_analyses}")
    
    # Validate travel time analysis configuration
    if cfg.travel_time_analysis.get('enabled', False):
        _validate_travel_time_config(cfg)
    
    # Validate source analysis configuration
    if cfg.source_analysis.get('enabled', False):
        _validate_source_config(cfg)
    
    # Validate velocity analysis configuration
    if cfg.velocity_analysis.get('enabled', False):
        _validate_velocity_config(cfg)
    
    # Validate distance analysis configuration
    if cfg.distance_analysis.get('enabled', False):
        _validate_distance_config(cfg)
    
    # Validate short distance analysis configuration
    if cfg.short_distance_analysis.get('enabled', False):
        _validate_short_distance_config(cfg)
    
    # Validate gradient analysis configuration
    if cfg.gradient_analysis.get('enabled', False):
        _validate_gradient_config(cfg)
    
    logger.info("Configuration validation completed")


def _validate_travel_time_config(cfg: DictConfig) -> None:
    """Validate travel time analysis configuration."""
    comparison_mode = cfg.travel_time_analysis.get('comparison_mode', 'ground_truth')
    
    if comparison_mode == 'method_comparison':
        method1_csv = cfg.travel_time_analysis.get('method1_csv')
        method2_csv = cfg.travel_time_analysis.get('method2_csv')
        
        if not method1_csv or not method2_csv:
            logger.error("Travel time analysis in method_comparison mode requires both method1_csv and method2_csv")
            sys.exit(1)
        
        if cfg.get('quality_control', {}).get('validate_input_data', True):
            if not Path(method1_csv).exists():
                logger.error(f"Travel time Method 1 CSV not found: {method1_csv}")
                sys.exit(1)
            if not Path(method2_csv).exists():
                logger.error(f"Travel time Method 2 CSV not found: {method2_csv}")
                sys.exit(1)
        
        logger.info(f"Travel time analysis: method_comparison mode")
        logger.info(f"  Method 1: {cfg.travel_time_analysis.get('method1_name', 'Method 1')}")
        logger.info(f"  Method 2: {cfg.travel_time_analysis.get('method2_name', 'Method 2')}")
        
    elif comparison_mode == 'ground_truth':
        input_csv = cfg.travel_time_analysis.get('input_csv')
        
        if not input_csv:
            logger.error("Travel time analysis in ground_truth mode requires input_csv")
            sys.exit(1)
        
        if cfg.get('quality_control', {}).get('validate_input_data', True):
            if not Path(input_csv).exists():
                logger.error(f"Travel time CSV not found: {input_csv}")
                sys.exit(1)
        
        logger.info(f"Travel time analysis: ground_truth mode")
    else:
        logger.error(f"Invalid travel time comparison_mode: {comparison_mode}")
        logger.error("Valid modes: 'ground_truth', 'method_comparison'")
        sys.exit(1)


def _validate_source_config(cfg: DictConfig) -> None:
    """Validate source analysis configuration."""
    comparison_mode = cfg.source_analysis.get('comparison_mode', 'ground_truth')
    
    if comparison_mode == 'method_comparison':
        method1_csv = cfg.source_analysis.get('method1_csv')
        method2_csv = cfg.source_analysis.get('method2_csv')
        
        if not method1_csv or not method2_csv:
            logger.error("Source analysis in method_comparison mode requires both method1_csv and method2_csv")
            sys.exit(1)
        
        if cfg.get('quality_control', {}).get('validate_input_data', True):
            if not Path(method1_csv).exists():
                logger.error(f"Source Method 1 CSV not found: {method1_csv}")
                sys.exit(1)
            if not Path(method2_csv).exists():
                logger.error(f"Source Method 2 CSV not found: {method2_csv}")
                sys.exit(1)
        
        logger.info(f"Source analysis: method_comparison mode")
        logger.info(f"  Method 1: {cfg.source_analysis.get('method1_name', 'Method 1')}")
        logger.info(f"  Method 2: {cfg.source_analysis.get('method2_name', 'Method 2')}")
        
    elif comparison_mode == 'ground_truth':
        input_csv = cfg.source_analysis.get('input_csv')
        
        if not input_csv:
            logger.error("Source analysis in ground_truth mode requires input_csv")
            sys.exit(1)
        
        if cfg.get('quality_control', {}).get('validate_input_data', True):
            if not Path(input_csv).exists():
                logger.error(f"Source CSV not found: {input_csv}")
                sys.exit(1)
        
        logger.info(f"Source analysis: ground_truth mode")
    else:
        logger.error(f"Invalid source comparison_mode: {comparison_mode}")
        logger.error("Valid modes: 'ground_truth', 'method_comparison'")
        sys.exit(1)


def _validate_velocity_config(cfg: DictConfig) -> None:
    """Validate velocity analysis configuration."""
    if cfg.get('quality_control', {}).get('validate_input_data', True):
        model1_csv = cfg.velocity_analysis.get('model1_csv')
        model2_csv = cfg.velocity_analysis.get('model2_csv')
        if model1_csv and not Path(model1_csv).exists():
            logger.error(f"Velocity model 1 CSV not found: {model1_csv}")
            sys.exit(1)
        if model2_csv and not Path(model2_csv).exists():
            logger.error(f"Velocity model 2 CSV not found: {model2_csv}")
            sys.exit(1)


def _validate_distance_config(cfg: DictConfig) -> None:
    """Validate distance analysis configuration."""
    comparison_mode = cfg.distance_analysis.get('comparison_mode', 'ground_truth')
    
    # Validate distance types
    distance_types = cfg.distance_analysis.get('distance_types', ['hypocentral'])
    if isinstance(distance_types, str):
        distance_types = [distance_types]
    
    valid_types = []
    for dt in distance_types:
        if dt in ['hypocentral', 'epicentral']:
            valid_types.append(dt)
        else:
            logger.warning(f"Invalid distance type '{dt}' in distance_analysis config")
    
    if not valid_types:
        logger.warning("No valid distance types in distance_analysis, defaulting to 'hypocentral'")
    
    if comparison_mode == 'method_comparison':
        method1_csv = cfg.distance_analysis.get('method1_csv')
        method2_csv = cfg.distance_analysis.get('method2_csv')
        station_csv = cfg.distance_analysis.get('station_csv')
        events_csv = cfg.distance_analysis.get('events_csv')
        
        if not method1_csv or not method2_csv:
            logger.error("Distance analysis in method_comparison mode requires both method1_csv and method2_csv")
            sys.exit(1)
        
        if not station_csv or not events_csv:
            logger.error("Distance analysis requires station_csv and events_csv for both modes")
            sys.exit(1)
        
        if cfg.get('quality_control', {}).get('validate_input_data', True):
            if not Path(method1_csv).exists():
                logger.error(f"Distance Method 1 CSV not found: {method1_csv}")
                sys.exit(1)
            if not Path(method2_csv).exists():
                logger.error(f"Distance Method 2 CSV not found: {method2_csv}")
                sys.exit(1)
            if not Path(station_csv).exists():
                logger.error(f"Station CSV not found: {station_csv}")
                sys.exit(1)
            if not Path(events_csv).exists():
                logger.error(f"Events CSV not found: {events_csv}")
                sys.exit(1)
        
        logger.info(f"Distance analysis: method_comparison mode")
        logger.info(f"  Method 1: {cfg.distance_analysis.get('method1_name', 'Method 1')}")
        logger.info(f"  Method 2: {cfg.distance_analysis.get('method2_name', 'Method 2')}")
        
    elif comparison_mode == 'ground_truth':
        if cfg.get('quality_control', {}).get('validate_input_data', True):
            files_to_check = [
                ('travel_time_csv', 'Distance analysis travel time CSV'),
                ('station_csv', 'Distance analysis station CSV'),
                ('events_csv', 'Distance analysis events CSV')
            ]
            for file_key, description in files_to_check:
                file_path = cfg.distance_analysis.get(file_key)
                if file_path and not Path(file_path).exists():
                    logger.error(f"{description} not found: {file_path}")
                    sys.exit(1)
        
        logger.info(f"Distance analysis: ground_truth mode")
    else:
        logger.error(f"Invalid distance analysis comparison_mode: {comparison_mode}")
        logger.error("Valid modes: 'ground_truth', 'method_comparison'")
        sys.exit(1)


def _validate_short_distance_config(cfg: DictConfig) -> None:
    """Validate short distance analysis configuration."""
    comparison_mode = cfg.short_distance_analysis.get('comparison_mode', 'ground_truth')
    
    # Validate distance types
    distance_types = cfg.short_distance_analysis.get('distance_types', ['hypocentral'])
    if isinstance(distance_types, str):
        distance_types = [distance_types]
    
    valid_types = []
    for dt in distance_types:
        if dt in ['hypocentral', 'epicentral']:
            valid_types.append(dt)
        else:
            logger.warning(f"Invalid distance type '{dt}' in short_distance_analysis config")
    
    if not valid_types:
        logger.warning("No valid distance types in short_distance_analysis, defaulting to 'hypocentral'")
    
    if comparison_mode == 'method_comparison':
        method1_csv = cfg.short_distance_analysis.get('method1_csv')
        method2_csv = cfg.short_distance_analysis.get('method2_csv')
        station_csv = cfg.short_distance_analysis.get('station_csv')
        events_csv = cfg.short_distance_analysis.get('events_csv')
        
        if not method1_csv or not method2_csv:
            logger.error("Short distance analysis in method_comparison mode requires both method1_csv and method2_csv")
            sys.exit(1)
        
        if not station_csv or not events_csv:
            logger.error("Short distance analysis requires station_csv and events_csv for both modes")
            sys.exit(1)
        
        if cfg.get('quality_control', {}).get('validate_input_data', True):
            if not Path(method1_csv).exists():
                logger.error(f"Short distance Method 1 CSV not found: {method1_csv}")
                sys.exit(1)
            if not Path(method2_csv).exists():
                logger.error(f"Short distance Method 2 CSV not found: {method2_csv}")
                sys.exit(1)
            if not Path(station_csv).exists():
                logger.error(f"Station CSV not found: {station_csv}")
                sys.exit(1)
            if not Path(events_csv).exists():
                logger.error(f"Events CSV not found: {events_csv}")
                sys.exit(1)
        
        logger.info(f"Short distance analysis: method_comparison mode")
        logger.info(f"  Method 1: {cfg.short_distance_analysis.get('method1_name', 'Method 1')}")
        logger.info(f"  Method 2: {cfg.short_distance_analysis.get('method2_name', 'Method 2')}")
        
    elif comparison_mode == 'ground_truth':
        if cfg.get('quality_control', {}).get('validate_input_data', True):
            files_to_check = [
                ('travel_time_csv', 'Short distance analysis travel time CSV'),
                ('station_csv', 'Short distance analysis station CSV'),
                ('events_csv', 'Short distance analysis events CSV')
            ]
            for file_key, description in files_to_check:
                file_path = cfg.short_distance_analysis.get(file_key)
                if file_path and not Path(file_path).exists():
                    logger.error(f"{description} not found: {file_path}")
                    sys.exit(1)
        
        logger.info(f"Short distance analysis: ground_truth mode")
    else:
        logger.error(f"Invalid short distance analysis comparison_mode: {comparison_mode}")
        logger.error("Valid modes: 'ground_truth', 'method_comparison'")
        sys.exit(1)


def _validate_gradient_config(cfg: DictConfig) -> None:
    """Validate gradient analysis configuration."""
    if cfg.get('quality_control', {}).get('validate_input_data', True):
        if "velocity" in cfg.gradient_analysis.get('analysis_type', []):
            velocity_gradient_csv = cfg.gradient_analysis.get('velocity_input_csv')
            if velocity_gradient_csv and not Path(velocity_gradient_csv).exists():
                logger.error(f"Velocity gradient analysis CSV not found: {velocity_gradient_csv}")
                sys.exit(1)
        
        if "source" in cfg.gradient_analysis.get('analysis_type', []):
            source_gradient_csv = cfg.gradient_analysis.get('source_input_csv')
            if source_gradient_csv and not Path(source_gradient_csv).exists():
                logger.error(f"Source gradient analysis CSV not found: {source_gradient_csv}")
                sys.exit(1)


@hydra.main(version_base=None, config_path="../configs/analysis", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for analysis with Hydra.
    
    Args:
        cfg: Hydra configuration
    """
    # Setup logging
    setup_logging(level=cfg.analysis.log_level)
    logger = logging.getLogger(__name__)
    
    # Log configuration in debug mode
    if cfg.analysis.log_level.upper() == "DEBUG":
        logger.debug("Full configuration:")
        logger.debug(f"\n{OmegaConf.to_yaml(cfg)}")
    
    # Validate configuration
    validate_configuration(cfg)
    
    # Check if we should run a single function
    single_function = cfg.get('single_function', None)
    if single_function:
        run_single_function(cfg, single_function)
    else:
        run_analysis_pipeline(cfg)


if __name__ == "__main__":
    main()