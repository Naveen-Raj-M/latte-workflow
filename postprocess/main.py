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
from .postprocess_dws import compute_dws_from_precomputed_rays
from .compute_ray_paths import (
    compute_all_ray_paths,
    load_event_id_list_and_convert_to_shot_ids,
    get_cache_path,
    get_cache_stats,
    is_cache_valid,
    load_ray_paths_from_cache
)


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

    # Process ray path diagnosis (generates ray path cache and DWS files)
    # Visualization is handled in the analysis module
    if hasattr(config, "ray_path_diagnosis") and config.ray_path_diagnosis.enabled:
        print(f"\nRay path diagnosis: ENABLED")
        try:
            diag_config = config.ray_path_diagnosis

            # Check if DWS computation is requested
            compute_dws = getattr(diag_config, 'compute_dws', False)

            # Get cache settings (caching is always enabled)
            force_recompute = getattr(diag_config, 'force_recompute', False)

            # Print cache status
            cache_path = get_cache_path(diag_config)
            print(f"\n  Cache settings:")
            print(f"    Ray path cache: {cache_path}")
            print(f"    Force recompute: {force_recompute}")

            if is_cache_valid(cache_path, diag_config):
                cache_stats = get_cache_stats(cache_path)
                print(f"    Cache status: VALID")
                print(f"    Cached rays: P={cache_stats['p_wave_rays']}, S={cache_stats['s_wave_rays']}")
                print(f"    Cache size: {cache_stats['file_size_mb']:.2f} MB")
            else:
                print(f"    Cache status: INVALID or NOT FOUND")

            if compute_dws:
                print(f"  Outputs: Ray path cache + DWS file")
                print(f"  DWS output: {diag_config.dws_output_file}")
            else:
                print(f"  Outputs: Ray path cache only")

            print(f"\n  NOTE: Visualizations have been moved to the analysis module")
            print(f"        Run analysis/main.py with raypath_analysis and dws_analysis enabled")

            # Event filtering/subsampling (optimization when DWS is disabled)
            event_id_list_file = getattr(diag_config, 'event_id_list_file', None)
            subsample_fraction = getattr(diag_config, 'subsample_fraction', 1.0)
            filter_shot_ids = None
            subsample_during_computation = False

            # Only apply filtering/subsampling if DWS is disabled (optimization)
            if not compute_dws:
                # Priority: event_id_list_file > subsample_fraction
                if event_id_list_file and event_id_list_file != "null":
                    shot_map_file = getattr(diag_config, 'shot_map_file', None)
                    if shot_map_file:
                        filter_shot_ids = load_event_id_list_and_convert_to_shot_ids(
                            event_id_list_file, shot_map_file
                        )
                        if filter_shot_ids:
                            subsample_during_computation = True
                            print(f"\n  OPTIMIZATION: Computing rays for {len(filter_shot_ids)} selected events only")
                    else:
                        print(f"  WARNING: event_id_list_file provided but shot_map_file not found")
                elif subsample_fraction < 1.0:
                    subsample_during_computation = True
                    print(f"\n  OPTIMIZATION: Computing {subsample_fraction*100:.1f}% of ray paths")

            if compute_dws or (not subsample_during_computation):
                print(f"\n  Computing ALL ray paths for complete cache...")

            precomputed_rays = compute_all_ray_paths(
                diag_config,
                subsample_during_computation,
                filter_shot_ids
            )

            # Compute DWS if requested
            if compute_dws:
                compute_dws_from_precomputed_rays(diag_config, precomputed_rays)

            # Clear precomputed rays from memory
            del precomputed_rays
            import gc
            gc.collect()

            print(f"\n  Ray path cache saved: {cache_path}")
            if compute_dws:
                print(f"  DWS file saved: {diag_config.dws_output_file}")

        except Exception as e:
            print(f"Error in ray path diagnosis: {e}")
            if config.strict_mode:
                raise
    else:
        print(f"\nRay path diagnosis: DISABLED")

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