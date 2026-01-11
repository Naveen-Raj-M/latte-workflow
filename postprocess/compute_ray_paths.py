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
    get_tracing_setting,
    load_shot_mapping
)


def load_event_id_list_and_convert_to_shot_ids(
    event_id_list_file: str,
    shot_map_file: str
) -> Optional[set]:
    """
    Load event IDs from a file and convert them to shot IDs using shot mapping.

    Args:
        event_id_list_file: Path to text file with event IDs (one per line)
        shot_map_file: Path to shot mapping file (Shot_ID -> Original_Event_ID)

    Returns:
        Set of shot IDs corresponding to the event IDs in the list, or None if files not found
    """
    if not event_id_list_file or event_id_list_file == "null":
        return None

    try:
        # Read event ID list
        print(f"\nLoading event ID list: {event_id_list_file}")
        with open(event_id_list_file, 'r') as f:
            event_ids = set(line.strip() for line in f if line.strip())
        print(f"  Found {len(event_ids)} event IDs in list")

        # Load shot mapping (Shot_ID -> Event_ID)
        print(f"Loading shot mapping: {shot_map_file}")
        shot_map = load_shot_mapping(shot_map_file)

        # Create reverse mapping (Event_ID -> Shot_ID)
        event_to_shot = {event_id: shot_id for shot_id, event_id in shot_map.items()}

        # Convert event IDs to shot IDs
        shot_ids = set()
        missing_events = []
        for event_id in event_ids:
            if event_id in event_to_shot:
                shot_ids.add(event_to_shot[event_id])
            else:
                missing_events.append(event_id)

        if missing_events:
            print(f"  WARNING: {len(missing_events)} event IDs not found in shot mapping:")
            for event_id in missing_events[:5]:  # Show first 5
                print(f"    - {event_id}")
            if len(missing_events) > 5:
                print(f"    ... and {len(missing_events) - 5} more")

        print(f"  Converted to {len(shot_ids)} shot IDs")
        return shot_ids

    except FileNotFoundError as e:
        print(f"  ERROR: File not found: {e}")
        return None
    except Exception as e:
        print(f"  ERROR: Failed to load event ID list: {e}")
        return None


def compute_all_ray_paths(
    config: DictConfig,
    subsample_during_computation: bool = False,
    filter_shot_ids: Optional[set] = None
) -> Dict[str, Dict]:
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
            - subsample_fraction: Fraction of rays to compute/plot (0.0-1.0)
            - dws.weight_column: Weight column name (optional)
        subsample_during_computation: If True, apply subsampling before ray tracing.
            Should be True only when visualization-only mode is enabled (no DWS).
        filter_shot_ids: Optional set of shot IDs to filter by. If provided:
            - When subsample_during_computation=True: only compute rays for these shots
            - When subsample_during_computation=False: compute all rays but mark which to visualize

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

        # Apply event filtering if provided and we're in visualization-only mode
        if filter_shot_ids is not None and subsample_during_computation:
            n_shots_before = len(unique_shots)
            unique_shots = np.array([s for s in unique_shots if s in filter_shot_ids])
            print(f"  EVENT FILTER: Selected {len(unique_shots)}/{n_shots_before} shots from event ID list")

        # Limit number of shots if specified (applied after event filtering)
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

        # Apply subsampling during computation if requested (visualization-only mode)
        if subsample_during_computation:
            subsample_fraction = getattr(config, 'subsample_fraction', 1.0)
            if subsample_fraction < 1.0:
                n_tasks_original = len(tasks)
                n_tasks_to_compute = max(1, int(n_tasks_original * subsample_fraction))
                # Random subsample for representative ray coverage
                indices = np.random.choice(n_tasks_original, n_tasks_to_compute, replace=False)
                indices = sorted(indices)  # Sort to maintain some spatial coherence

                # Subsample all lists
                tasks = [tasks[i] for i in indices]
                shot_ids_list = [shot_ids_list[i] for i in indices]
                sources_list = [sources_list[i] for i in indices]
                receivers_list = [receivers_list[i] for i in indices]
                weights_list = [weights_list[i] for i in indices]

                print(f"  OPTIMIZATION: Subsampling for visualization-only mode")
                print(f"  Computing {n_tasks_to_compute}/{n_tasks_original} ray paths ({subsample_fraction*100:.1f}%)")
            else:
                print(f"  No subsampling (subsample_fraction = {subsample_fraction})")

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

        # If event filtering is active but we computed all rays (DWS mode),
        # filter rays for visualization while keeping all for DWS
        if filter_shot_ids is not None and not subsample_during_computation:
            # Filter rays for visualization (keep rays only for specified events)
            viz_indices = [i for i in range(len(shot_ids_valid)) if shot_ids_valid[i] in filter_shot_ids]
            print(f"  EVENT FILTER: Will plot {len(viz_indices)}/{len(ray_paths_valid)} rays matching event ID list")
            print(f"  (All {len(ray_paths_valid)} rays will be used for DWS computation)")

            result[phase] = {
                'ray_paths': ray_paths_valid,
                'sources': sources_valid,
                'receivers': receivers_valid,
                'shot_ids': shot_ids_valid,
                'weights': weights_valid,
                # Store filtered versions for visualization
                'viz_ray_paths': [ray_paths_valid[i] for i in viz_indices],
                'viz_sources': [sources_valid[i] for i in viz_indices],
                'viz_receivers': [receivers_valid[i] for i in viz_indices],
                'viz_shot_ids': [shot_ids_valid[i] for i in viz_indices]
            }
        else:
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
