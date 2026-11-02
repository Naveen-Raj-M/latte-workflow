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
from utils.velocity_io import resolve_reference_grid

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
)
from .simplify_ray_paths import simplify_ray_paths_main


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
                "sx_file": config.source_processing.sx_file,
                "sy_file": config.source_processing.sy_file,
                "sz_file": config.source_processing.sz_file,
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
                for param in ['sx', 'sy', 'sz', 'st0']:
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

    # Process ray paths (generates ray path cache and DWS files)
    # Visualization is handled in the analysis module
    if hasattr(config, "ray_path_processing") and config.ray_path_processing.enabled:
        print(f"\nRay path processing: ENABLED")
        try:
            diag_config = config.ray_path_processing

            # Grid: from the reference velocity model; the configured grid is only a check
            # (or the grid itself when no reference model is set)
            ref_file = diag_config.get('reference_velocity_file', None)
            cfg_grid = diag_config.get('grid', None)
            grid = resolve_reference_grid(
                reference_file=ref_file or None,
                config_grid=OmegaConf.to_container(cfg_grid) if cfg_grid else None,
                source="ray_path_processing",
            )
            OmegaConf.set_struct(diag_config, False)
            diag_config.grid = OmegaConf.create(grid)
            OmegaConf.set_struct(diag_config, True)
            print(f"  Grid: {grid['nx']}x{grid['ny']}x{grid['nz']}, "
                  f"d=({grid['dx']}, {grid['dy']}, {grid['dz']}) km")

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

            # Set load_cache_for_downstream automatically based on DWS requirement
            OmegaConf.set_struct(diag_config, False)
            diag_config.load_cache_for_downstream = compute_dws
            OmegaConf.set_struct(diag_config, True)

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
            print(f"Error in ray path processing: {e}")
            if config.general.strict_mode:
                raise
    else:
        print(f"\nRay path processing: DISABLED")

    # Ray path simplification (optional, for reducing memory in analysis); can run on an existing cache
    simp_cfg = config.get("ray_path_processing", {}).get("simplification", None)
    if simp_cfg is not None and simp_cfg.get("enabled", False):
        print(f"\nRay path simplification: ENABLED")
        try:
            simplify_ray_paths_main(config.ray_path_processing)
        except Exception as e:
            print(f"Error in ray path simplification: {e}")
            if config.general.strict_mode:
                raise
    else:
        print(f"\nRay path simplification: DISABLED")

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