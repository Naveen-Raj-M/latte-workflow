"""
Main entry point for LATTE analysis module.

This script provides the main interface for running different analysis modules
with Hydra configuration management.
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
    
    with ProcessTimer("Distance analysis"):
        try:
            ensure_directory_exists(cfg.distance_analysis.output_dir)
            results = analyze_distance_dependent_residuals(OmegaConf.to_container(cfg, resolve=True))
            logger.info("Distance analysis completed successfully")
            logger.info(f"Results saved to: {cfg.distance_analysis.output_dir}")
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
    
    with ProcessTimer("Short distance analysis"):
        try:
            ensure_directory_exists(cfg.short_distance_analysis.output_dir)
            results = analyze_short_distance_residuals(OmegaConf.to_container(cfg, resolve=True))
            logger.info("Short distance analysis completed successfully")
            logger.info(f"Results saved to: {cfg.short_distance_analysis.output_dir}")
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
            ensure_directory_exists(cfg.gradient_analysis.output_dir)
            results = analyze_gradients_and_search(OmegaConf.to_container(cfg, resolve=True))
            logger.info("Gradient analysis completed successfully")
            logger.info(f"Results saved to: {cfg.gradient_analysis.output_dir}")
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
    function_map[function_name](cfg)


def run_analysis_pipeline(cfg: DictConfig) -> None:
    """
    Run the complete analysis pipeline.
    
    Args:
        cfg: Hydra configuration
    """
    logger.info("Starting LATTE analysis pipeline")
    logger.info(f"Configuration mode: {cfg.get('mode', 'full_pipeline')}")
    
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
    
    # Check file paths if quality control is enabled
    if cfg.get('quality_control', {}).get('validate_input_data', True):
        missing_files = []
        
        # Check travel time analysis files
        if cfg.travel_time_analysis.get('enabled', False):
            tt_csv = cfg.travel_time_analysis.get('input_csv')
            if tt_csv and not Path(tt_csv).exists():
                missing_files.append(f"Travel time CSV: {tt_csv}")
        
        # Check velocity analysis files
        if cfg.velocity_analysis.get('enabled', False):
            model1_csv = cfg.velocity_analysis.get('model1_csv')
            model2_csv = cfg.velocity_analysis.get('model2_csv')
            if model1_csv and not Path(model1_csv).exists():
                missing_files.append(f"Velocity model 1 CSV: {model1_csv}")
            if model2_csv and not Path(model2_csv).exists():
                missing_files.append(f"Velocity model 2 CSV: {model2_csv}")
        
        # Check distance analysis files
        if cfg.distance_analysis.get('enabled', False):
            files_to_check = [
                ('travel_time_csv', 'Distance analysis travel time CSV'),
                ('station_csv', 'Distance analysis station CSV'),
                ('events_csv', 'Distance analysis events CSV')
            ]
            for file_key, description in files_to_check:
                file_path = cfg.distance_analysis.get(file_key)
                if file_path and not Path(file_path).exists():
                    missing_files.append(f"{description}: {file_path}")
        
        # Check short distance analysis files
        if cfg.short_distance_analysis.get('enabled', False):
            files_to_check = [
                ('travel_time_csv', 'Short distance analysis travel time CSV'),
                ('station_csv', 'Short distance analysis station CSV'),
                ('events_csv', 'Short distance analysis events CSV')
            ]
            for file_key, description in files_to_check:
                file_path = cfg.short_distance_analysis.get(file_key)
                if file_path and not Path(file_path).exists():
                    missing_files.append(f"{description}: {file_path}")
        
        # Check source correction analysis files
        if cfg.source_analysis.get('enabled', False):
            source_csv = cfg.source_analysis.get('input_csv')
            if source_csv and not Path(source_csv).exists():
                missing_files.append(f"Source correction CSV: {source_csv}")
        
        # Check gradient analysis files
        if cfg.gradient_analysis.get('enabled', False):
            gradient_csv = cfg.gradient_analysis.get('input_csv')
            if gradient_csv and not Path(gradient_csv).exists():
                missing_files.append(f"Gradient analysis CSV: {gradient_csv}")
        
        if missing_files:
            logger.error("Missing input files:")
            for missing_file in missing_files:
                logger.error(f"  - {missing_file}")
            
            if cfg.get('quality_control', {}).get('error_on_empty_results', True):
                sys.exit(1)
            else:
                logger.warning("Continuing despite missing files...")
    
    logger.info("Configuration validation completed")


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