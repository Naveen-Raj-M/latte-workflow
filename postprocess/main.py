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
            create_velocity_csv_from_binaries(config.velocity_processing)
        except Exception as e:
            print(f"Error in velocity processing: {e}")
            if config.general.strict_mode:
                raise
    else:
        print(f"\nVelocity processing: DISABLED")
    
    # Process travel time binaries
    if config.travel_time_processing.enabled:
        print(f"\nTravel time processing: ENABLED")
        use_exchange = config.travel_time_processing.get('source_receiver_exchange', False)
        use_source_correction = config.travel_time_processing.get('source_correction', False)

        try:
            tt_config_dict = {
                'ground_truth_csv': config.travel_time_processing.ground_truth_csv,
                'synthetic_tt_dir': config.travel_time_processing.synthetic_tt_dir,
                'shot_map_file': config.travel_time_processing.shot_map_file,
                'synthetic_tt_output_csv': config.travel_time_processing.synthetic_tt_output_csv,
                'use_source_correction': use_source_correction,
                'phase_type': config.travel_time_processing.get('phase_type', 'both'),
            }
            if use_source_correction:
                tt_config_dict['source_correction_file'] = config.travel_time_processing.source_correction_file
            if use_exchange:
                print("Using SOURCE-RECEIVER EXCHANGE mode")
                tt = config.travel_time_processing
                # geometry_dir rebuilds the exchange; exchange_geometry_file is an optional LATTE output used as a check
                for key in ('geometry_dir', 'geometry_file', 'exchange_geometry_file',
                            'filtered_shot_ids', 'filtered_receiver_ids', 'coordinate_tolerance'):
                    if tt.get(key, None) is not None:
                        tt_config_dict[key] = tt.get(key)
                tt_config = OmegaConf.create(tt_config_dict)
                create_synthetic_travel_time_csv_with_exchange(tt_config)
            else:
                print("Using STANDARD mode (no exchange)")
                tt_config = OmegaConf.create(tt_config_dict)
                create_synthetic_travel_time_csv(tt_config)
        except Exception as e:
            print(f"Error in travel time processing: {e}")
            if config.general.strict_mode:
                raise
    else:
        print(f"\nTravel time processing: DISABLED")

    # Process source parameter updates
    if hasattr(config, "source_processing") and config.source_processing.enabled:
        print(f"\nSource parameter processing: ENABLED")
        try:
            # Base configuration
            source_cfg_dict = {
                "slon_file": config.source_processing.slon_file,
                "slat_file": config.source_processing.slat_file,
                "sdepth_file": config.source_processing.sdepth_file,
                "st0_file": config.source_processing.st0_file,
                "initial_geometry_csv": config.source_processing.get("initial_geometry_csv", None),
                "filtered_shot_ids": list(config.source_processing.get("filtered_shot_ids", None) or []),
                "travel_time_csv": config.travel_time_processing.synthetic_tt_output_csv,
                "output_csv": config.source_processing.output_csv,
                "enabled": True
            }
            
            # Add optional gradient/search configuration
            if hasattr(config.source_processing, 'parameters_to_process'):
                source_cfg_dict['parameters_to_process'] = config.source_processing.parameters_to_process
                source_cfg_dict['process_gradient'] = config.source_processing.get('process_gradient', False)
                source_cfg_dict['process_search'] = config.source_processing.get('process_search', False)

                # Add gradient and search files if specified
                for param in ['slon', 'slat', 'sdepth', 'st0']:
                    grad_key = f'grad_{param}_file'
                    if hasattr(config.source_processing, grad_key):
                        source_cfg_dict[grad_key] = getattr(config.source_processing, grad_key)

                    srch_key = f'srch_{param}_file'
                    if hasattr(config.source_processing, srch_key):
                        source_cfg_dict[srch_key] = getattr(config.source_processing, srch_key)
            
            source_cfg = OmegaConf.create(source_cfg_dict)
            # Set parent to allow access to the parent configuration
            OmegaConf.set_struct(source_cfg, False)
            source_cfg._parent = config
            OmegaConf.set_struct(source_cfg, True)
            process_source_updates(source_cfg)
        except Exception as e:
            print(f"Error in source parameter processing: {e}")
            if config.general.strict_mode:
                raise
    else:
        print(f"\nSource parameter processing: DISABLED")

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