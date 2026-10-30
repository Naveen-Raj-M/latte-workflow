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
from utils.io_utils import ensure_directory_exists, resolve_csv_path

# Import analysis functions
from .traveltime_analysis import analyze_travel_time_residuals
from .velocity_analysis import analyze_velocity_models
from .distance_analysis import analyze_distance_dependent_residuals
from .short_distance_analysis import analyze_short_distance_residuals
from .source_analysis import analyze_source_corrections
from .gradient_analysis import analyze_gradients_and_search
from .dws_analysis import analyze_dws
from .raypath_analysis import analyze_ray_paths
from .ray_density_analysis import analyze_ray_density
from .tradeoff_analysis import analyze_tradeoff

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


def run_dws_analysis(cfg: DictConfig) -> None:
    """Run DWS (Derivative Weight Sum) analysis."""
    if not cfg.dws_analysis.get('enabled', False):
        logger.info("DWS analysis disabled, skipping")
        return

    logger.info("=" * 60)
    logger.info("STARTING DWS ANALYSIS")
    logger.info("=" * 60)

    with ProcessTimer("DWS analysis"):
        try:
            ensure_directory_exists(cfg.dws_analysis.output_dir)
            results = analyze_dws(OmegaConf.to_container(cfg, resolve=True))
            logger.info("DWS analysis completed successfully")
            logger.info(f"Results saved to: {cfg.dws_analysis.output_dir}")

            # Log summary
            if 'statistics' in results:
                stats = results['statistics']
                logger.info(f"DWS summary:")
                logger.info(f"  Coverage: {stats['coverage_percentage']:.2f}%")
                logger.info(f"  DWS range: [{stats['dws_min']:.6e}, {stats['dws_max']:.6e}] km²")
                logger.info(f"  Mean DWS: {stats['dws_mean']:.6e} km²")

        except Exception as e:
            logger.error(f"DWS analysis failed: {e}")
            raise


def run_raypath_analysis(cfg: DictConfig) -> None:
    """Run ray path visualization analysis."""
    if not cfg.raypath_analysis.get('enabled', False):
        logger.info("Ray path analysis disabled, skipping")
        return

    logger.info("=" * 60)
    logger.info("STARTING RAY PATH ANALYSIS")
    logger.info("=" * 60)

    with ProcessTimer("Ray path analysis"):
        try:
            ensure_directory_exists(cfg.raypath_analysis.output_dir)
            results = analyze_ray_paths(OmegaConf.to_container(cfg, resolve=True))
            logger.info("Ray path analysis completed successfully")
            logger.info(f"Results saved to: {cfg.raypath_analysis.output_dir}")

            # Log summary
            if 'statistics' in results:
                stats = results['statistics']
                logger.info(f"Ray path summary:")
                logger.info(f"  P-wave: {stats.get('p_wave_shots', 0)} shots, {stats.get('p_wave_rays', 0)} rays")
                logger.info(f"  S-wave: {stats.get('s_wave_shots', 0)} shots, {stats.get('s_wave_rays', 0)} rays")
                logger.info(f"  Cache size: {stats.get('file_size_mb', 0):.2f} MB")

        except Exception as e:
            logger.error(f"Ray path analysis failed: {e}")
            raise


def run_ray_density_analysis(cfg: DictConfig) -> None:
    """Run ray density analysis using actual traced ray paths."""
    if not cfg.ray_density_analysis.get('enabled', False):
        logger.info("Ray density analysis disabled, skipping")
        return

    logger.info("=" * 60)
    logger.info("STARTING RAY DENSITY ANALYSIS")
    logger.info("=" * 60)

    with ProcessTimer("Ray density analysis"):
        try:
            ensure_directory_exists(cfg.ray_density_analysis.output_dir)
            results = analyze_ray_density(OmegaConf.to_container(cfg, resolve=True))
            logger.info("Ray density analysis completed successfully")
            logger.info(f"Results saved to: {cfg.ray_density_analysis.output_dir}")

            # Log summary
            if 'statistics' in results:
                for phase, stats in results['statistics'].items():
                    logger.info(f"{phase.upper()}-wave density summary:")
                    logger.info(f"  Coverage: {stats['coverage_percentage']:.1f}%")
                    logger.info(f"  Density range: [{stats['min_density']}, {stats['max_density']}]")
                    logger.info(f"  Mean density: {stats['mean_density']:.2f}")

        except Exception as e:
            logger.error(f"Ray density analysis failed: {e}")
            raise


def run_tradeoff_analysis(cfg: DictConfig) -> None:
    """Run tradeoff (data misfit vs model variance) analysis."""
    if not cfg.tradeoff_analysis.get('enabled', False):
        logger.info("Tradeoff analysis disabled, skipping")
        return

    logger.info("=" * 60)
    logger.info("STARTING TRADEOFF ANALYSIS")
    logger.info("=" * 60)

    with ProcessTimer("Tradeoff analysis"):
        try:
            ensure_directory_exists(cfg.tradeoff_analysis.output_dir)
            results = analyze_tradeoff(OmegaConf.to_container(cfg, resolve=True))
            logger.info("Tradeoff analysis completed successfully")
            logger.info(f"Results saved to: {cfg.tradeoff_analysis.output_dir}")

            # Log brief summary
            if 'tradeoff_table' in results:
                df = results['tradeoff_table']
                if not df.empty:
                    logger.info(f"  Iterations included: {len(df)}")
                    logger.info(
                        f"  Iteration range: "
                        f"[{df['iteration'].min()}, {df['iteration'].max()}]"
                    )
            elif 'by_depth' in results:
                for depth_km, df in results['by_depth'].items():
                    if not df.empty:
                        logger.info(
                            f"  Z={depth_km:.2f} km: {len(df)} iterations"
                        )
        except Exception as e:
            logger.error(f"Tradeoff analysis failed: {e}")
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
        'dws_analysis': run_dws_analysis,
        'raypath_analysis': run_raypath_analysis,
        'ray_density_analysis': run_ray_density_analysis,
        'tradeoff_analysis': run_tradeoff_analysis,
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
    logger.info(f"Phase type: {cfg.analysis.get('phase_type', 'both')}")

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
        logger.info(f"  Parallel processing: {cfg.performance.parallel_processing}")
        if cfg.performance.parallel_processing:
            logger.info(f"  Number of jobs: {cfg.performance.n_jobs}")

    # Initialize quality control settings
    if hasattr(cfg, 'quality_control'):
        logger.info("Quality control configuration:")
        logger.info(f"  Validate input data: {cfg.quality_control.validate_input_data}")
        logger.info(f"  Check coordinate consistency: {cfg.quality_control.check_coordinate_consistency}")

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
            ('Ray Path Analysis', run_raypath_analysis),
            ('DWS Analysis', run_dws_analysis),
            ('Ray Density Analysis', run_ray_density_analysis),
            ('Tradeoff Analysis', run_tradeoff_analysis),
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
    if cfg.raypath_analysis.get('enabled', False):
        enabled_analyses.append('raypath_analysis')
    if cfg.dws_analysis.get('enabled', False):
        enabled_analyses.append('dws_analysis')
    if cfg.ray_density_analysis.get('enabled', False):
        enabled_analyses.append('ray_density_analysis')
    if cfg.tradeoff_analysis.get('enabled', False):
        enabled_analyses.append('tradeoff_analysis')

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

    # Validate tradeoff analysis configuration
    if cfg.tradeoff_analysis.get('enabled', False):
        _validate_tradeoff_config(cfg)

    logger.info("Configuration validation completed")


def _validate_travel_time_config(cfg: DictConfig) -> None:
    """Validate travel time analysis configuration (supports CSV and HDF5)."""
    comparison_mode = cfg.travel_time_analysis.get('comparison_mode', 'ground_truth')

    if comparison_mode == 'method_comparison':
        method1_csv = cfg.travel_time_analysis.get('method1_csv')
        method2_csv = cfg.travel_time_analysis.get('method2_csv')
        method1_hdf5 = cfg.travel_time_analysis.get('method1_hdf5')
        method2_hdf5 = cfg.travel_time_analysis.get('method2_hdf5')

        # Check if at least one format is provided for each method
        if not (method1_csv or method1_hdf5) or not (method2_csv or method2_hdf5):
            logger.error("Travel time analysis in method_comparison mode requires input files for both methods")
            logger.error("Provide either CSV (method1_csv/method2_csv) or HDF5 (method1_hdf5/method2_hdf5) for each method")
            sys.exit(1)

        if cfg.get('quality_control', {}).get('validate_input_data', True):
            # Validate Method 1
            if method1_hdf5 and Path(method1_hdf5).exists():
                logger.info(f"  Method 1: Using HDF5 file: {method1_hdf5}")
            elif method1_csv and Path(resolve_csv_path(method1_csv)).exists():
                logger.info(f"  Method 1: Using CSV file: {method1_csv}")
            else:
                logger.error(f"Travel time Method 1 file not found (tried CSV: {method1_csv}, HDF5: {method1_hdf5})")
                sys.exit(1)

            # Validate Method 2
            if method2_hdf5 and Path(method2_hdf5).exists():
                logger.info(f"  Method 2: Using HDF5 file: {method2_hdf5}")
            elif method2_csv and Path(resolve_csv_path(method2_csv)).exists():
                logger.info(f"  Method 2: Using CSV file: {method2_csv}")
            else:
                logger.error(f"Travel time Method 2 file not found (tried CSV: {method2_csv}, HDF5: {method2_hdf5})")
                sys.exit(1)

        logger.info(f"Travel time analysis: method_comparison mode")
        logger.info(f"  Method 1: {cfg.travel_time_analysis.get('method1_name', 'Method 1')}")
        logger.info(f"  Method 2: {cfg.travel_time_analysis.get('method2_name', 'Method 2')}")

    elif comparison_mode == 'ground_truth':
        input_csv = cfg.travel_time_analysis.get('input_csv')
        input_hdf5 = cfg.travel_time_analysis.get('input_hdf5')

        if not input_csv and not input_hdf5:
            logger.error("Travel time analysis in ground_truth mode requires either input_csv or input_hdf5")
            sys.exit(1)

        if cfg.get('quality_control', {}).get('validate_input_data', True):
            # Prefer HDF5 if available, fallback to CSV
            if input_hdf5 and Path(input_hdf5).exists():
                logger.info(f"Travel time analysis: Using HDF5 file: {input_hdf5}")
            elif input_csv and Path(resolve_csv_path(input_csv)).exists():
                logger.info(f"Travel time analysis: Using CSV file: {input_csv}")
            else:
                logger.error(f"Travel time input file not found (tried CSV: {input_csv}, HDF5: {input_hdf5})")
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
            if not Path(resolve_csv_path(method1_csv)).exists():
                logger.error(f"Source Method 1 CSV not found: {method1_csv}")
                sys.exit(1)
            if not Path(resolve_csv_path(method2_csv)).exists():
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
            if not Path(resolve_csv_path(input_csv)).exists():
                logger.error(f"Source CSV not found: {input_csv}")
                sys.exit(1)
        
        logger.info(f"Source analysis: ground_truth mode")
    else:
        logger.error(f"Invalid source comparison_mode: {comparison_mode}")
        logger.error("Valid modes: 'ground_truth', 'method_comparison'")
        sys.exit(1)


def _validate_velocity_config(cfg: DictConfig) -> None:
    """Validate velocity analysis configuration (supports CSV and HDF5)."""
    if cfg.get('quality_control', {}).get('validate_input_data', True):
        model1_csv = cfg.velocity_analysis.get('model1_csv')
        model2_csv = cfg.velocity_analysis.get('model2_csv')
        model1_hdf5 = cfg.velocity_analysis.get('model1_hdf5')
        model2_hdf5 = cfg.velocity_analysis.get('model2_hdf5')

        # Validate Model 1
        if model1_csv or model1_hdf5:
            if model1_hdf5 and Path(model1_hdf5).exists():
                logger.info(f"Velocity model 1: Using HDF5 file: {model1_hdf5}")
            elif model1_csv and Path(model1_csv).exists():
                logger.info(f"Velocity model 1: Using CSV file: {model1_csv}")
            else:
                logger.error(f"Velocity model 1 file not found (tried CSV: {model1_csv}, HDF5: {model1_hdf5})")
                sys.exit(1)

        # Validate Model 2
        if model2_csv or model2_hdf5:
            if model2_hdf5 and Path(model2_hdf5).exists():
                logger.info(f"Velocity model 2: Using HDF5 file: {model2_hdf5}")
            elif model2_csv and Path(model2_csv).exists():
                logger.info(f"Velocity model 2: Using CSV file: {model2_csv}")
            else:
                logger.error(f"Velocity model 2 file not found (tried CSV: {model2_csv}, HDF5: {model2_hdf5})")
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
            if not Path(resolve_csv_path(method1_csv)).exists():
                logger.error(f"Distance Method 1 CSV not found: {method1_csv}")
                sys.exit(1)
            if not Path(resolve_csv_path(method2_csv)).exists():
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
                if file_path and not Path(resolve_csv_path(file_path)).exists():
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
            if not Path(resolve_csv_path(method1_csv)).exists():
                logger.error(f"Short distance Method 1 CSV not found: {method1_csv}")
                sys.exit(1)
            if not Path(resolve_csv_path(method2_csv)).exists():
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
                if file_path and not Path(resolve_csv_path(file_path)).exists():
                    logger.error(f"{description} not found: {file_path}")
                    sys.exit(1)
        
        logger.info(f"Short distance analysis: ground_truth mode")
    else:
        logger.error(f"Invalid short distance analysis comparison_mode: {comparison_mode}")
        logger.error("Valid modes: 'ground_truth', 'method_comparison'")
        sys.exit(1)


def _validate_gradient_config(cfg: DictConfig) -> None:
    """Validate gradient analysis configuration (supports CSV and HDF5)."""
    if cfg.get('quality_control', {}).get('validate_input_data', True):
        if "velocity" in cfg.gradient_analysis.get('analysis_type', []):
            velocity_gradient_csv = cfg.gradient_analysis.get('velocity_input_csv')
            velocity_gradient_hdf5 = cfg.gradient_analysis.get('velocity_input_hdf5')

            if velocity_gradient_csv or velocity_gradient_hdf5:
                if velocity_gradient_hdf5 and Path(velocity_gradient_hdf5).exists():
                    logger.info(f"Velocity gradient analysis: Using HDF5 file: {velocity_gradient_hdf5}")
                elif velocity_gradient_csv and Path(velocity_gradient_csv).exists():
                    logger.info(f"Velocity gradient analysis: Using CSV file: {velocity_gradient_csv}")
                else:
                    logger.error(f"Velocity gradient file not found (tried CSV: {velocity_gradient_csv}, HDF5: {velocity_gradient_hdf5})")
                    sys.exit(1)

        if "source" in cfg.gradient_analysis.get('analysis_type', []):
            source_gradient_csv = cfg.gradient_analysis.get('source_input_csv')
            source_gradient_hdf5 = cfg.gradient_analysis.get('source_input_hdf5')

            if source_gradient_csv or source_gradient_hdf5:
                if source_gradient_hdf5 and Path(source_gradient_hdf5).exists():
                    logger.info(f"Source gradient analysis: Using HDF5 file: {source_gradient_hdf5}")
                elif source_gradient_csv and Path(resolve_csv_path(source_gradient_csv)).exists():
                    logger.info(f"Source gradient analysis: Using CSV file: {source_gradient_csv}")
                else:
                    logger.error(f"Source gradient file not found (tried CSV: {source_gradient_csv}, HDF5: {source_gradient_hdf5})")
                    sys.exit(1)


def _validate_tradeoff_config(cfg: DictConfig) -> None:
    """Validate tradeoff analysis configuration."""
    tcfg = cfg.tradeoff_analysis

    # y_axis_mode
    y_mode = tcfg.get('y_axis_mode', 'data_misfit')
    if y_mode not in ('data_misfit', 'rms', 'data_variance'):
        logger.error(
            f"Invalid tradeoff_analysis y_axis_mode: '{y_mode}'. "
            "Valid options: 'data_misfit', 'rms', 'data_variance'"
        )
        sys.exit(1)

    # data_misfit mode requirements
    if y_mode == 'data_misfit':
        if not tcfg.get('data_misfit_file_path'):
            logger.error(
                "tradeoff_analysis: data_misfit_file_path is required "
                "when y_axis_mode = 'data_misfit'"
            )
            sys.exit(1)
        if cfg.get('quality_control', {}).get('validate_input_data', True):
            fp = tcfg.get('data_misfit_file_path')
            if fp and not Path(fp).exists():
                logger.error(f"tradeoff_analysis: data_misfit_file_path not found: {fp}")
                sys.exit(1)

    # rms mode requirements
    if y_mode == 'rms':
        if not tcfg.get('inversion_base_path'):
            logger.error(
                "tradeoff_analysis: inversion_base_path is required "
                "when y_axis_mode = 'rms'"
            )
            sys.exit(1)
        phase_mode = tcfg.get('phase_mode', 'both')
        if phase_mode not in ('both', 'p', 's', 'separate'):
            logger.error(
                f"Invalid tradeoff_analysis phase_mode: '{phase_mode}'. "
                "Valid options: 'both', 'p', 's', 'separate'"
            )
            sys.exit(1)

    # Model paths
    if not tcfg.get('initial_model_csv') and not tcfg.get('initial_model_hdf5'):
        logger.error(
            "tradeoff_analysis: initial_model_csv or initial_model_hdf5 is required"
        )
        sys.exit(1)
    if not tcfg.get('inversion_analysis_dir'):
        logger.error("tradeoff_analysis: inversion_analysis_dir is required")
        sys.exit(1)

    if cfg.get('quality_control', {}).get('validate_input_data', True):
        hdf5 = tcfg.get('initial_model_hdf5')
        csv = tcfg.get('initial_model_csv')
        if hdf5 and not Path(hdf5).exists():
            logger.error(
                f"tradeoff_analysis: initial_model_hdf5 not found: {hdf5}"
            )
            sys.exit(1)
        elif not hdf5 and csv and not Path(csv).exists():
            logger.error(
                f"tradeoff_analysis: initial_model_csv not found: {csv}"
            )
            sys.exit(1)

        inv_dir = tcfg.get('inversion_analysis_dir')
        if inv_dir and not Path(inv_dir).exists():
            logger.error(
                f"tradeoff_analysis: inversion_analysis_dir not found: {inv_dir}"
            )
            sys.exit(1)

    # Region mode
    mv_cfg = tcfg.get('model_variance', {})
    region = mv_cfg.get('region', 'full')
    if region not in ('full', 'z_bounded', 'depth_layer'):
        logger.error(
            f"Invalid tradeoff_analysis model_variance.region: '{region}'. "
            "Valid options: 'full', 'z_bounded', 'depth_layer'"
        )
        sys.exit(1)
    if region == 'depth_layer':
        layers = mv_cfg.get('depth_layers', [])
        if not layers:
            logger.error(
                "tradeoff_analysis: model_variance.depth_layers must be non-empty "
                "when region = 'depth_layer'"
            )
            sys.exit(1)

    logger.info("Tradeoff analysis configuration validated")


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