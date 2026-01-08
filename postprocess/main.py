#!/usr/bin/env python3
"""
Main script for LATTE post-processing.
Converts LATTE tomography binary outputs to CSV format for analysis.

Usage:
    python main.py

The configuration is loaded from ../configs/postprocess/default.yaml
"""
import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.io_utils import ensure_directory_exists
from utils.logging_utils import setup_logging

from .postprocess_velocity import create_velocity_csv_from_binaries
from .postprocess_traveltime import (
    create_synthetic_travel_time_csv,
    create_synthetic_travel_time_csv_with_exchange
)
from .postprocess_source import process_source_updates
from .postprocess_raypath import create_ray_path_visualizations, create_ray_path_visualizations_by_pairs
from .postprocess_dws import create_dws_from_ray_paths


def postprocess_latte_outputs(config: DictConfig) -> None:
    """Main post-processing function."""
    print("=== LATTE POST-PROCESSING: BINARY TO CSV CONVERSION ===")
    print(f"Configuration loaded from: {config.config_file if hasattr(config, 'config_file') else 'default'}")

    # Log data paths
    print(f"\nData paths:")
    print(f"  Base directory: {config.data.base_dir}")
    if hasattr(config.data, 'analysis_output_dir'):
        print(f"  Analysis output directory: {config.data.analysis_output_dir}")
        # Ensure analysis output directory exists
        ensure_directory_exists(config.data.analysis_output_dir)

    # Create output directories
    if config.velocity_processing.enabled:
        ensure_directory_exists(os.path.dirname(config.velocity_processing.velocity_output_csv))

    if config.travel_time_processing.enabled:
        ensure_directory_exists(os.path.dirname(config.travel_time_processing.synthetic_tt_output_csv))

    if hasattr(config, "source_processing") and config.source_processing.enabled:
        ensure_directory_exists(os.path.dirname(config.source_processing.output_csv))
    
    # Process velocity binaries
    if config.velocity_processing.enabled:
        print(f"\nVelocity processing: ENABLED")
        try:
            phase_type = config.general.get('phase_type', 'both')
            max_depth_km = config.general.get('max_depth_km', None)
            create_velocity_csv_from_binaries(config.velocity_processing, phase_type, max_depth_km)
        except Exception as e:
            print(f"Error in velocity processing: {e}")
            if config.strict_mode:
                raise
    else:
        print(f"\nVelocity processing: DISABLED")
    
    # Process travel time binaries
    if config.travel_time_processing.enabled:
        print(f"\nTravel time processing: ENABLED")
        use_exchange = config.travel_time_processing.get('source_receiver_exchange', False)
        use_source_correction = config.travel_time_processing.get('source_correction', False)
        phase_type = config.general.get('phase_type', 'both')
        max_depth_km = config.general.get('max_depth_km', None)
        removed_shots = config.travel_time_processing.get('removed_shots', [])

        # Get st0 precision from config
        st0_precision = 'single'  # default
        if hasattr(config.general, 'precision') and hasattr(config.general.precision, 'st0'):
            st0_precision = config.general.precision.st0

        # Log removed shots if any
        if removed_shots:
            print(f"  Removed shots (filtered during geometry loading): {removed_shots}")

        try:
            tt_config_dict = {
                'ground_truth_csv': config.travel_time_processing.ground_truth_csv,
                'synthetic_tt_dir': config.travel_time_processing.synthetic_tt_dir,
                'shot_map_file': config.travel_time_processing.shot_map_file,
                'synthetic_tt_output_csv': config.travel_time_processing.synthetic_tt_output_csv,
                'use_source_correction': use_source_correction,
                'removed_shots': removed_shots
            }
            if use_source_correction:
                tt_config_dict['source_correction_file'] = config.travel_time_processing.source_correction_file
            if use_exchange:
                print("Using SOURCE-RECEIVER EXCHANGE mode")
                tt_config_dict['exchange_geometry_file'] = config.travel_time_processing.exchange_geometry_file
                tt_config_dict['exchange_mapping_file'] = config.travel_time_processing.exchange_mapping_file
                tt_config_dict['virtual_shot_mapping_file'] = config.travel_time_processing.get('virtual_shot_mapping_file', None)
                tt_config = OmegaConf.create(tt_config_dict)
                create_synthetic_travel_time_csv_with_exchange(tt_config, phase_type, max_depth_km, st0_precision)
            else:
                print("Using STANDARD mode (no exchange)")
                tt_config = OmegaConf.create(tt_config_dict)
                create_synthetic_travel_time_csv(tt_config, phase_type, max_depth_km, st0_precision)
        except Exception as e:
            print(f"Error in travel time processing: {e}")
            if config.strict_mode:
                raise
    else:
        print(f"\nTravel time processing: DISABLED")

    # Process source parameter updates
    if hasattr(config, "source_processing") and config.source_processing.enabled:
        print(f"\nSource parameter processing: ENABLED")
        try:
            # Base configuration
            source_cfg_dict = {
                "sx_file": config.source_processing.sx_file,
                "sy_file": config.source_processing.sy_file,
                "sz_file": config.source_processing.sz_file,
                "st0_file": config.source_processing.st0_file,
                "grid_dimensions": config.source_processing.grid_dimensions,
                "travel_time_csv": config.travel_time_processing.synthetic_tt_output_csv,
                "output_csv": config.source_processing.output_csv,
                "enabled": True
            }
            
            # Add optional gradient/search configuration
            if hasattr(config.source_processing, 'parameters_to_process'):
                source_cfg_dict['parameters_to_process'] = config.source_processing.parameters_to_process
                
                # Add gradient and search files if specified
                for param in ['sx', 'sy', 'sz', 'st0']:
                    grad_key = f'grad_{param}_file'
                    if hasattr(config.source_processing, grad_key):
                        source_cfg_dict[grad_key] = getattr(config.source_processing, grad_key)
                    
                    srch_key = f'srch_{param}_file'
                    if hasattr(config.source_processing, srch_key):
                        source_cfg_dict[srch_key] = getattr(config.source_processing, srch_key)
            
            source_cfg = OmegaConf.create(source_cfg_dict)
            # Set parent to allow access to general.precision settings
            OmegaConf.set_struct(source_cfg, False)
            source_cfg._parent = config
            OmegaConf.set_struct(source_cfg, True)
            process_source_updates(source_cfg)
        except Exception as e:
            print(f"Error in source parameter processing: {e}")
            if config.strict_mode:
                raise
    else:
        print(f"\nSource parameter processing: DISABLED")

    # Process ray path visualization
    if hasattr(config, "ray_path_visualization") and config.ray_path_visualization.enabled:
        print(f"\nRay path visualization: ENABLED")
        try:
            # Choose parallelization mode
            parallel_mode = "by_shots"  # default
            if hasattr(config.ray_path_visualization, 'parallel') and hasattr(config.ray_path_visualization.parallel, 'mode'):
                parallel_mode = config.ray_path_visualization.parallel.mode

            if parallel_mode == "by_pairs":
                print(f"  Using PAIR-LEVEL parallelization (better load balancing)")
                create_ray_path_visualizations_by_pairs(config.ray_path_visualization)
            else:
                print(f"  Using SHOT-LEVEL parallelization (lower I/O overhead)")
                create_ray_path_visualizations(config.ray_path_visualization)
        except Exception as e:
            print(f"Error in ray path visualization: {e}")
            if config.strict_mode:
                raise
    else:
        print(f"\nRay path visualization: DISABLED")

    # Process DWS computation
    if hasattr(config, "dws_computation") and config.dws_computation.enabled:
        print(f"\nDWS computation: ENABLED")
        try:
            create_dws_from_ray_paths(config.dws_computation)
        except Exception as e:
            print(f"Error in DWS computation: {e}")
            if config.strict_mode:
                raise
    else:
        print(f"\nDWS computation: DISABLED")

    print("\n=== POST-PROCESSING COMPLETE ===")


@hydra.main(version_base=None, config_path="../configs/postprocess", config_name="default")
def main(cfg: DictConfig) -> None:
    setup_logging(level=cfg.general.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting LATTE post-processing")
    if cfg.general.log_level.upper() == "DEBUG":
        logger.debug("Full configuration:")
        logger.debug(f"\n{OmegaConf.to_yaml(cfg)}")
    try:
        postprocess_latte_outputs(cfg)
        logger.info("Post-processing completed successfully!")
    except Exception as e:
        logger.error(f"Post-processing failed: {e}")
        raise


if __name__ == "__main__":
    main()