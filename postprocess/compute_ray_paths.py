"""
Shared ray path computation module for LATTE post-processing.

This module computes ray paths from gradient fields and stores them in memory
for use by both visualization and DWS computation, avoiding duplicate computation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from omegaconf import DictConfig
from tqdm import tqdm
from joblib import Parallel, delayed

from postprocess.postprocess_raypath import (
    process_single_ray,
    get_performance_setting,
    get_tracing_setting
)


def compute_all_ray_paths(config: DictConfig) -> Dict[str, Dict]:
    """
    Compute all ray paths for configured phase types (P and/or S waves).

    This is the shared function that traces all rays once and returns structured data
    that can be used by both visualization and DWS computation modules.

    Args:
        config: Ray path diagnosis configuration object containing:
            - ground_truth_csv: Path to ground truth traveltime data
            - gradient_dir: Directory containing gradient files
            - grid: Grid dimensions
            - phase_type: 'p', 's', or 'both'
            - precision: Numerical precision ('single' or 'double')
            - max_shots: Maximum shots to process (0 = all)
            - dws.weight_column: Weight column name (optional)

    Returns:
        Dictionary with phase as keys ('p', 's') and values containing:
            {
                'ray_paths': List of ray path arrays (N, 3) for each source-receiver pair
                'sources': List of source positions (x, y, z) in km
                'receivers': List of receiver positions (x, y, z) in km
                'shot_ids': List of shot IDs
                'weights': List of observation weights
            }
    """
    print("\n" + "="*80)
    print("COMPUTING RAY PATHS (SHARED COMPUTATION)")
    print("="*80)

    # Load ground truth data
    print(f"\nLoading ground truth data: {config.ground_truth_csv}")
    df_gt = pd.read_csv(config.ground_truth_csv)

    # Convert coordinates from meters to kilometers
    coordinate_columns = [
        'event_x_transformed', 'event_y_transformed', 'event_z_transformed',
        'station_x_transformed', 'station_y_transformed', 'station_z_transformed'
    ]
    for col in coordinate_columns:
        if col in df_gt.columns:
            df_gt[col] = df_gt[col] / 1000.0
    print("  Converted coordinates from meters to kilometers")
    print(f"  Total source-receiver pairs: {len(df_gt)}")

    # Determine phase types to process
    phase_type = getattr(config, 'phase_type', 'both')
    phase_types = []
    if phase_type in ['both', 'p']:
        phase_types.append('p')
    if phase_type in ['both', 's']:
        phase_types.append('s')

    result = {}

    # Get performance settings from global config
    parallel_processing = get_performance_setting(config, 'parallel_processing', True)
    n_jobs = get_performance_setting(config, 'n_jobs', 1) if parallel_processing else 1
    backend = get_performance_setting(config, 'backend', 'loky')
    batch_size = get_performance_setting(config, 'batch_size', 'auto')
    pre_dispatch = get_performance_setting(config, 'pre_dispatch', '2*n_jobs')

    # Get tracing parameters from global config
    tracing_params = {
        'max_steps': get_tracing_setting(config, 'max_steps', 10000),
        'step_size': get_tracing_setting(config, 'step_size', 0.1),
        'tolerance': get_tracing_setting(config, 'tolerance', 0.5)
    }
    interpolation_method = get_tracing_setting(config, 'interpolation_method', 'nearest')

    # Get common parameters
    gradient_dir = config.gradient_dir
    precision = getattr(config, 'precision', 'single')
    max_shots = getattr(config, 'max_shots', 0)

    # Get weight column (for DWS)
    weight_column = None
    if hasattr(config, 'dws') and hasattr(config.dws, 'weight_column'):
        weight_column = config.dws.weight_column

    print(f"\nGlobal settings:")
    print(f"  Parallel processing: {parallel_processing} (n_jobs={n_jobs}, backend={backend})")
    print(f"  Ray tracing: max_steps={tracing_params['max_steps']}, step_size={tracing_params['step_size']}, interpolation={interpolation_method}")
    print(f"  Grid: {config.grid.nx}x{config.grid.ny}x{config.grid.nz}, spacing={config.grid.dx}x{config.grid.dy}x{config.grid.dz} km")

    for phase in phase_types:
        print(f"\n{'='*80}")
        print(f"Computing {phase.upper()}-wave ray paths")
        print(f"{'='*80}")

        # Filter by phase if column exists
        if 'phase' in df_gt.columns:
            df_phase = df_gt[df_gt['phase'].str.upper() == phase.upper()].copy()
        else:
            df_phase = df_gt.copy()

        # Prepare task list
        tasks = []
        shot_ids_list = []
        sources_list = []
        receivers_list = []
        weights_list = []

        # Get unique shots
        unique_shots = df_phase['shot_id'].unique()

        # Limit number of shots if specified
        if max_shots > 0:
            unique_shots = unique_shots[:max_shots]
            print(f"  Limited to {max_shots} shots")

        # Get weight column if it exists
        has_weights = weight_column is not None and weight_column in df_phase.columns
        if has_weights:
            print(f"  Using weight column: '{weight_column}'")
        else:
            print(f"  No weights found, using uniform weights (1.0)")

        for shot_id in unique_shots:
            shot_data = df_phase[df_phase['shot_id'] == shot_id]
            if len(shot_data) == 0:
                continue

            # Get source position
            source_row = shot_data.iloc[0]
            source_pos = (
                source_row['event_x_transformed'],
                source_row['event_y_transformed'],
                source_row['event_z_transformed']
            )

            # Create task for each receiver
            for _, row in shot_data.iterrows():
                receiver_pos = (
                    row['station_x_transformed'],
                    row['station_y_transformed'],
                    row['station_z_transformed']
                )
                weight = row[weight_column] if has_weights else 1.0

                tasks.append((shot_id, source_pos, receiver_pos))
                shot_ids_list.append(shot_id)
                sources_list.append(source_pos)
                receivers_list.append(receiver_pos)
                weights_list.append(weight)

        print(f"  Created {len(tasks)} ray tracing tasks ({len(unique_shots)} shots)")

        # Trace all rays in parallel
        if parallel_processing and n_jobs != 1:
            print(f"  Tracing rays using {n_jobs if n_jobs > 0 else 'all'} CPU cores...")
        else:
            print(f"  Tracing rays sequentially...")

        ray_paths_all = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            batch_size=batch_size,
            pre_dispatch=pre_dispatch,
            verbose=0
        )(
            delayed(process_single_ray)(
                shot_id,
                source_pos,
                receiver_pos,
                gradient_dir,
                phase,
                precision,
                tracing_params,
                interpolation_method
            )
            for shot_id, source_pos, receiver_pos in tqdm(tasks, desc=f"Tracing {phase}-wave rays")
        )

        # Filter out None values (failed rays)
        valid_indices = [i for i, rp in enumerate(ray_paths_all) if rp is not None]
        ray_paths_valid = [ray_paths_all[i] for i in valid_indices]
        sources_valid = [sources_list[i] for i in valid_indices]
        receivers_valid = [receivers_list[i] for i in valid_indices]
        shot_ids_valid = [shot_ids_list[i] for i in valid_indices]
        weights_valid = [weights_list[i] for i in valid_indices]

        print(f"  Successfully traced {len(ray_paths_valid)}/{len(tasks)} ray paths ({100*len(ray_paths_valid)/len(tasks):.1f}%)")
        print(f"  Failed rays: {len(tasks) - len(ray_paths_valid)}")

        result[phase] = {
            'ray_paths': ray_paths_valid,
            'sources': sources_valid,
            'receivers': receivers_valid,
            'shot_ids': shot_ids_valid,
            'weights': weights_valid
        }

    print("\n" + "="*80)
    print("RAY PATH COMPUTATION COMPLETE")
    print("="*80)

    return result
