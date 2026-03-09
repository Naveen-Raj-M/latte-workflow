"""
Add Gaussian or uniform noise to synthetic traveltime binary files.

This module reads synthetic traveltime binary files from LATTE, adds
configurable noise, and writes new binary files for inversion testing.

Supports two modes:
- Standard: reads shot_N_traveltime_p.bin directly, applies noise, writes output.
- Exchange: LATTE was run with stations as virtual sources (source-receiver exchange).
  Reads shot_<virtual_id>_traveltime_p.bin files, re-assembles them into the
  standard shot_N format (receiver order matching shot_N_geometry.txt), applies
  noise, and writes output identical in format to the standard mode.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)


def read_traveltime_binary(bin_path: Path, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Read traveltime binary file.

    Args:
        bin_path: Path to binary file
        dtype: Data type (default: float32)

    Returns:
        1D array of travel times
    """
    if not bin_path.exists():
        raise FileNotFoundError(f"Binary file not found: {bin_path}")

    data = np.fromfile(bin_path, dtype=dtype)
    return data


def write_traveltime_binary(
    data: np.ndarray,
    output_path: Path,
    dtype: np.dtype = np.float32,
) -> None:
    """
    Write traveltime binary file.

    Args:
        data: 1D array of travel times
        output_path: Output file path
        dtype: Data type (default: float32)
    """
    data_typed = data.astype(dtype)
    with open(output_path, 'wb') as f:
        f.write(data_typed.tobytes())


def generate_gaussian_noise(
    size: int,
    mean: float,
    std: float,
    random_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Generate Gaussian (normal) noise.

    Args:
        size: Number of samples
        mean: Mean of distribution
        std: Standard deviation
        random_state: Random state for reproducibility

    Returns:
        Noise array
    """
    if random_state is None:
        random_state = np.random.RandomState()

    noise = random_state.normal(mean, std, size)
    return noise


def generate_uniform_noise(
    size: int,
    low: float,
    high: float,
    random_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Generate uniform noise.

    Args:
        size: Number of samples
        low: Lower bound
        high: Upper bound
        random_state: Random state for reproducibility

    Returns:
        Noise array
    """
    if random_state is None:
        random_state = np.random.RandomState()

    noise = random_state.uniform(low, high, size)
    return noise


def add_noise_to_traveltime(
    traveltime: np.ndarray,
    noise_config: Dict,
    phase: str,
    random_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Add noise to traveltime array.

    Args:
        traveltime: Original traveltime array
        noise_config: Noise configuration dictionary
        phase: "p" or "s" for phase-specific settings
        random_state: Random state for reproducibility

    Returns:
        Noisy traveltime array
    """
    noise_type = noise_config.get("noise_type", "gaussian")

    # Check for phase-specific overrides
    phase_config = noise_config.get(f"{phase}_wave", {})

    if noise_type == "gaussian":
        # Get parameters (phase-specific or default)
        mean = phase_config.get("mean", noise_config["gaussian"]["mean"])
        std = phase_config.get("std", noise_config["gaussian"]["std"])

        noise = generate_gaussian_noise(len(traveltime), mean, std, random_state)

        logger.debug(f"  Generated Gaussian noise: mean={mean:.4f}s, std={std:.4f}s")

    elif noise_type == "uniform":
        low = noise_config["uniform"]["low"]
        high = noise_config["uniform"]["high"]

        noise = generate_uniform_noise(len(traveltime), low, high, random_state)

        logger.debug(f"  Generated uniform noise: [{low:.4f}s, {high:.4f}s]")

    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    # Add noise
    noisy_traveltime = traveltime + noise

    # Log statistics
    logger.debug(f"  Noise range: {noise.min():.4f}s to {noise.max():.4f}s")
    logger.debug(f"  Noise RMS: {np.sqrt(np.mean(noise**2)):.4f}s")

    return noisy_traveltime


def process_traveltime_directory(
    input_dir: Path,
    output_dir: Path,
    ground_truth_csv: Path,
    noise_config: Dict,
    random_seed: Optional[int] = None,
) -> Dict[str, int]:
    """
    Process all traveltime binary files in a directory.

    Args:
        input_dir: Directory containing synthetic traveltime binaries
        output_dir: Output directory for noisy traveltimes
        ground_truth_csv: CSV with metadata (to identify all shot/phase combinations)
        noise_config: Noise configuration
        random_seed: Random seed for reproducibility

    Returns:
        Statistics dictionary
    """
    logger.info("=" * 80)
    logger.info("STEP 2: Adding Noise to Synthetic Traveltimes")
    logger.info("=" * 80)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize random state
    random_state = np.random.RandomState(random_seed)

    # Read ground truth CSV to get all shot/phase combinations
    logger.info(f"Reading ground truth CSV: {ground_truth_csv}")
    df = pd.read_csv(ground_truth_csv)

    # Get unique shot_id and phase combinations
    combinations = df.groupby(["shot_id", "phase"]).size().reset_index(name="count")
    logger.info(f"Found {len(combinations)} unique shot/phase combinations")

    # Determine output precision
    output_precision = noise_config.get("output_precision", "single")
    output_dtype = np.float32 if output_precision == "single" else np.float64

    # Process each combination
    stats = {"processed": 0, "skipped": 0, "total_samples": 0}

    for _, row in tqdm(combinations.iterrows(), total=len(combinations), desc="Processing traveltimes"):
        shot_id = int(row["shot_id"])
        phase = row["phase"].lower()
        expected_count = int(row["count"])

        # Input file path
        input_file = input_dir / f"shot_{shot_id}_traveltime_{phase}.bin"

        if not input_file.exists():
            logger.warning(f"File not found, skipping: {input_file}")
            stats["skipped"] += 1
            continue

        # Read binary
        try:
            traveltime = read_traveltime_binary(input_file)
        except Exception as e:
            logger.error(f"Error reading {input_file}: {e}")
            stats["skipped"] += 1
            continue

        # Verify size
        if len(traveltime) != expected_count:
            logger.warning(
                f"Size mismatch for shot {shot_id} phase {phase}: "
                f"expected {expected_count}, got {len(traveltime)}"
            )

        # Add noise
        noisy_traveltime = add_noise_to_traveltime(
            traveltime, noise_config, phase, random_state
        )

        # Output file path
        output_file = output_dir / f"shot_{shot_id}_traveltime_{phase}.bin"

        # Write binary
        write_traveltime_binary(noisy_traveltime, output_file, output_dtype)

        stats["processed"] += 1
        stats["total_samples"] += len(noisy_traveltime)

    # Log summary
    logger.info("\nNoise addition completed:")
    logger.info(f"  Processed: {stats['processed']} files")
    logger.info(f"  Skipped: {stats['skipped']} files")
    logger.info(f"  Total samples: {stats['total_samples']}")

    # Create a summary CSV with statistics
    summary_data = []
    for _, row in combinations.iterrows():
        shot_id = int(row["shot_id"])
        phase = row["phase"].lower()

        input_file = input_dir / f"shot_{shot_id}_traveltime_{phase}.bin"
        output_file = output_dir / f"shot_{shot_id}_traveltime_{phase}.bin"

        if output_file.exists():
            # Read both files to compute statistics
            original = read_traveltime_binary(input_file)
            noisy = read_traveltime_binary(output_file, dtype=output_dtype)
            noise = noisy - original

            summary_data.append({
                "shot_id": shot_id,
                "phase": phase,
                "num_receivers": len(original),
                "noise_mean": noise.mean(),
                "noise_std": noise.std(),
                "noise_min": noise.min(),
                "noise_max": noise.max(),
                "noise_rms": np.sqrt(np.mean(noise**2)),
            })

    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_dir / "noise_statistics.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"\nNoise statistics saved to: {summary_csv}")

    # Log overall statistics
    if len(summary_df) > 0:
        logger.info("\nOverall noise statistics:")
        logger.info(f"  Mean: {summary_df['noise_mean'].mean():.6f} s")
        logger.info(f"  Std: {summary_df['noise_std'].mean():.6f} s")
        logger.info(f"  RMS: {summary_df['noise_rms'].mean():.6f} s")

    # Create a comprehensive CSV with all ground truth columns + synthetic_tt
    logger.info("\nCreating noisy_traveltimes.csv with all metadata...")
    create_noisy_traveltimes_csv(df, output_dir, output_dir, output_dtype)

    return stats


def create_noisy_traveltimes_csv(
    ground_truth_df: pd.DataFrame,
    noisy_traveltime_dir: Path,
    output_dir: Path,
    dtype: np.dtype = np.float32,
) -> None:
    """
    Create a CSV file with all ground truth columns plus synthetic_tt from noisy binaries.

    Args:
        ground_truth_df: Ground truth DataFrame with all metadata
        noisy_traveltime_dir: Directory containing noisy traveltime binary files
        output_dir: Output directory for CSV
        dtype: Data type of binary files
    """
    # Create a copy of ground truth dataframe to preserve all columns
    output_df = ground_truth_df.copy()

    # Add synthetic_tt column (initialize with NaN)
    output_df["synthetic_tt"] = np.nan

    # Group by shot_id and phase to read corresponding binary files
    grouped = ground_truth_df.groupby(["shot_id", "phase"])

    for (shot_id, phase), group in tqdm(grouped, desc="Creating noisy traveltimes CSV"):
        phase_lower = phase.lower()
        binary_file = noisy_traveltime_dir / f"shot_{shot_id}_traveltime_{phase_lower}.bin"

        if binary_file.exists():
            try:
                # Read noisy traveltime binary
                noisy_tt = read_traveltime_binary(binary_file, dtype=dtype)

                # Get the indices for this shot_id and phase in the original dataframe
                indices = group.index

                # Subtract configured origin time so synthetic_tt is physical travel time only,
                # consistent with observed_tt (which carries no offset) and postprocessing convention.
                configured_origin_time = group["origin_time_s"].iloc[0]
                noisy_tt = noisy_tt - configured_origin_time

                # Verify size matches
                if len(noisy_tt) != len(indices):
                    logger.warning(
                        f"Size mismatch for shot {shot_id} phase {phase}: "
                        f"binary has {len(noisy_tt)} values, CSV has {len(indices)} rows"
                    )
                    # Take minimum to avoid index errors
                    n_values = min(len(noisy_tt), len(indices))
                    output_df.loc[indices[:n_values], "synthetic_tt"] = noisy_tt[:n_values]
                else:
                    # Fill synthetic_tt for this shot/phase combination
                    output_df.loc[indices, "synthetic_tt"] = noisy_tt

            except Exception as e:
                logger.error(f"Error reading {binary_file}: {e}")
        else:
            logger.warning(f"Binary file not found: {binary_file}")

    # Save to CSV
    output_csv = output_dir / "noisy_traveltimes.csv"
    output_df.to_csv(output_csv, index=False)
    logger.info(f"Noisy traveltimes CSV saved to: {output_csv}")

    # Log how many rows have valid synthetic_tt values
    valid_count = output_df["synthetic_tt"].notna().sum()
    total_count = len(output_df)
    logger.info(f"  {valid_count}/{total_count} rows have valid synthetic_tt values")


def read_shot_geometry_file(geometry_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a LATTE shot geometry file.

    File format (blank lines are separators, ignored):
        shot_id
        ns
        sx1 sy1 sz1
        ...
        nr
        rx1 ry1 rz1
        ...

    Args:
        geometry_file: Path to shot_N_geometry.txt

    Returns:
        (source_coords, receiver_coords) as float64 arrays of shape (ns, 3)
        and (nr, 3) in **meters** (LATTE native units).
    """
    lines = geometry_file.read_text().split('\n')
    non_empty = [l.strip() for l in lines if l.strip()]

    idx = 0
    idx += 1  # shot_id line

    ns = int(non_empty[idx]); idx += 1
    source_coords = []
    for _ in range(ns):
        parts = non_empty[idx].split()
        source_coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
        idx += 1

    nr = int(non_empty[idx]); idx += 1
    receiver_coords = []
    for _ in range(nr):
        parts = non_empty[idx].split()
        receiver_coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
        idx += 1

    return np.array(source_coords, dtype=np.float64), np.array(receiver_coords, dtype=np.float64)


def load_exchange_geometry(exchange_geometry_file: Path) -> Dict[int, Tuple[float, float, float]]:
    """
    Load the exchange geometry file that maps virtual shot ID to station
    coordinates (in km, matching station_x/y/z_transformed in the ground-truth CSV).

    File format (one entry per line, comment lines start with '#'):
        virtual_id  x_km  y_km  z_km

    Args:
        exchange_geometry_file: Path to the exchange geometry file.

    Returns:
        Dict mapping virtual_id (int) to (x, y, z) in km.
    """
    virtual_id_to_coord: Dict[int, Tuple[float, float, float]] = {}
    with open(exchange_geometry_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 4:
                    virtual_id = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    virtual_id_to_coord[virtual_id] = (x, y, z)
    return virtual_id_to_coord


def _find_virtual_id(
    rx_km: float,
    ry_km: float,
    rz_km: float,
    virtual_coords: np.ndarray,
    virtual_ids: List[int],
    tolerance: float,
) -> Optional[int]:
    """
    Return the virtual_id whose coordinates are closest to (rx, ry, rz) within
    *tolerance* km, or None if no match is found.
    """
    dists = np.sqrt(
        (virtual_coords[:, 0] - rx_km) ** 2
        + (virtual_coords[:, 1] - ry_km) ** 2
        + (virtual_coords[:, 2] - rz_km) ** 2
    )
    min_idx = int(np.argmin(dists))
    if dists[min_idx] <= tolerance:
        return virtual_ids[min_idx]
    return None


def process_traveltime_directory_with_exchange(
    input_dir: Path,
    output_dir: Path,
    geometry_dir: Path,
    ground_truth_csv: Path,
    exchange_geometry_file: Path,
    noise_config: Dict,
    removed_shots: Optional[List[int]] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, int]:
    """
    Process traveltime binary files produced by a source-receiver exchange
    forward run, re-assemble them into the standard shot format, apply noise,
    and write output binaries that are **identical in format** to those produced
    by the standard (no-exchange) mode.

    In exchange mode LATTE uses each station as a virtual source. The file
    ``shot_<virtual_id>_traveltime_{p|s}.bin`` therefore contains one traveltime
    per real shot (event), stored in compacted array order (skipping any shots
    removed by LATTE geometry filtering).

    This function inverts that layout:
      - For each real shot_id, read ``shot_<real_id>_geometry.txt`` to learn
        which receivers are expected and in what order.
      - For each receiver, find the corresponding virtual station binary and
        extract the traveltime at the position of ``real_id`` in the compacted
        array.
      - Apply noise and write ``shot_<real_id>_traveltime_{p|s}.bin`` so that
        the k-th value corresponds to the k-th receiver listed in the geometry
        file — exactly as a standard LATTE output would be.

    Args:
        input_dir: Directory containing exchange traveltime binaries
                   (``shot_<virtual_id>_traveltime_{p|s}.bin``).
        output_dir: Directory for output standard traveltime binaries
                    (``shot_<real_id>_traveltime_{p|s}.bin``).
        geometry_dir: Directory containing ``shot_N_geometry.txt`` files.
        ground_truth_csv: CSV with metadata (shot_id, phase, station,
                          origin_time_s, …).
        exchange_geometry_file: File mapping virtual_id → (x, y, z) in km
                                (same coordinate system as
                                station_x/y/z_transformed in the CSV).
        noise_config: Noise configuration dictionary (same schema as for the
                      standard ``process_traveltime_directory``).
        removed_shots: Shot IDs that LATTE removed during geometry filtering.
                       These compact the binary arrays. Defaults to ``[]``.
        random_seed: Random seed for reproducibility.

    Returns:
        Statistics dictionary with keys 'processed', 'skipped', 'total_samples'.
    """
    logger.info("=" * 80)
    logger.info("STEP 2: Adding Noise to Synthetic Traveltimes (SOURCE-RECEIVER EXCHANGE MODE)")
    logger.info("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)
    random_state = np.random.RandomState(random_seed)
    removed_shots = list(removed_shots) if removed_shots else []
    removed_set = set(removed_shots)

    output_precision = noise_config.get("output_precision", "single")
    output_dtype = np.float32 if output_precision == "single" else np.float64

    # ------------------------------------------------------------------ #
    # 1. Ground-truth CSV: enumerate real shots and phases                #
    # ------------------------------------------------------------------ #
    logger.info(f"Reading ground truth CSV: {ground_truth_csv}")
    df = pd.read_csv(ground_truth_csv)
    unique_shot_ids = sorted(df['shot_id'].unique())
    phases_in_data = sorted(df['phase'].str.upper().unique())
    logger.info(f"  Unique shots: {len(unique_shot_ids)}, phases: {phases_in_data}")

    # ------------------------------------------------------------------ #
    # 2. Build shot_id ↔ array-index mapping (accounts for removed shots) #
    # ------------------------------------------------------------------ #
    if removed_set:
        logger.info(f"Removed shots (filtered by LATTE): {sorted(removed_set)}")
    valid_shot_ids = [sid for sid in unique_shot_ids if sid not in removed_set]
    shot_id_to_array_index = {sid: idx for idx, sid in enumerate(valid_shot_ids)}
    expected_array_size = len(valid_shot_ids)
    logger.info(f"  Valid shots after removal: {expected_array_size}")

    # ------------------------------------------------------------------ #
    # 3. Exchange geometry: virtual_id → (x, y, z) in km                 #
    # ------------------------------------------------------------------ #
    logger.info(f"Loading exchange geometry: {exchange_geometry_file}")
    virtual_id_to_coord = load_exchange_geometry(exchange_geometry_file)
    virtual_ids = list(virtual_id_to_coord.keys())
    virtual_coords = np.array([virtual_id_to_coord[vid] for vid in virtual_ids], dtype=np.float64)
    logger.info(f"  Loaded {len(virtual_ids)} virtual station coordinates")

    coordinate_tolerance = noise_config.get("coordinate_tolerance", 0.01)  # km

    # ------------------------------------------------------------------ #
    # 4. Load all exchange binaries into memory (one array per            #
    #    virtual_id × phase).  Each array has shape (expected_array_size,) #
    # ------------------------------------------------------------------ #
    logger.info("Loading exchange traveltime binaries into memory...")
    exchange_data: Dict[Tuple[int, str], np.ndarray] = {}

    for phase in phases_in_data:
        phase_lower = phase.lower()
        loaded = 0
        for vid in virtual_ids:
            bin_path = input_dir / f"shot_{vid}_traveltime_{phase_lower}.bin"
            if not bin_path.exists():
                continue
            data = read_traveltime_binary(bin_path)
            if data is None:
                logger.warning(f"  Could not read {bin_path.name}, skipping")
                continue
            if len(data) != expected_array_size:
                raise ValueError(
                    f"Array size mismatch for {bin_path.name}: "
                    f"expected {expected_array_size} elements (valid shots after "
                    f"removing {sorted(removed_set) if removed_set else 'none'}), "
                    f"got {len(data)}. "
                    f"Check the removed_shots configuration."
                )
            exchange_data[(vid, phase)] = data
            loaded += 1
        logger.info(f"  {phase}-wave: loaded {loaded}/{len(virtual_ids)} binaries")

    # ------------------------------------------------------------------ #
    # 5. For every real shot, read its geometry file, match each receiver  #
    #    to a virtual station, assemble the traveltime array, apply noise, #
    #    and write the output binary.                                      #
    # ------------------------------------------------------------------ #
    stats = {"processed": 0, "skipped": 0, "total_samples": 0}
    summary_data = []

    logger.info("\nAssembling standard-format output binaries per shot...")
    for shot_id in tqdm(unique_shot_ids, desc="Processing shots"):
        if shot_id in removed_set:
            stats["skipped"] += 1
            continue

        array_idx = shot_id_to_array_index[shot_id]

        # Read geometry file for this shot
        geom_file = geometry_dir / f"shot_{shot_id}_geometry.txt"
        if not geom_file.exists():
            logger.warning(f"Geometry file not found, skipping shot {shot_id}: {geom_file}")
            stats["skipped"] += 1
            continue

        try:
            _, receiver_coords_m = read_shot_geometry_file(geom_file)
        except Exception as e:
            logger.error(f"Error parsing {geom_file}: {e}")
            stats["skipped"] += 1
            continue

        # Convert receiver coords from meters to km for matching
        receiver_coords_km = receiver_coords_m / 1000.0
        nr = len(receiver_coords_km)

        for phase in phases_in_data:
            phase_lower = phase.lower()

            # Build traveltime array in geometry-file receiver order
            tt_array = np.zeros(nr, dtype=np.float64)
            unmatched_indices = []

            for i, (rx, ry, rz) in enumerate(receiver_coords_km):
                vid = _find_virtual_id(rx, ry, rz, virtual_coords, virtual_ids, coordinate_tolerance)
                if vid is None:
                    unmatched_indices.append(i)
                    continue
                exchange_arr = exchange_data.get((vid, phase))
                if exchange_arr is None:
                    unmatched_indices.append(i)
                    continue
                tt_array[i] = exchange_arr[array_idx]

            if unmatched_indices:
                logger.warning(
                    f"  Shot {shot_id} phase {phase}: "
                    f"{len(unmatched_indices)}/{nr} receivers could not be matched "
                    f"to a virtual station within {coordinate_tolerance} km tolerance. "
                    f"Unmatched positions set to 0.0. "
                    f"Check exchange_geometry_file and coordinate_tolerance."
                )

            # Apply noise
            noisy_tt = add_noise_to_traveltime(tt_array, noise_config, phase_lower, random_state)

            # Write output binary — format identical to standard (no-exchange) mode
            output_file = output_dir / f"shot_{shot_id}_traveltime_{phase_lower}.bin"
            write_traveltime_binary(noisy_tt, output_file, output_dtype)

            stats["processed"] += 1
            stats["total_samples"] += nr

            summary_data.append({
                "shot_id": shot_id,
                "phase": phase_lower,
                "num_receivers": nr,
                "unmatched_receivers": len(unmatched_indices),
            })

    # ------------------------------------------------------------------ #
    # 6. Log summary and write noise statistics CSV                        #
    # ------------------------------------------------------------------ #
    logger.info("\nExchange noise addition completed:")
    logger.info(f"  Processed: {stats['processed']} shot-phase binaries")
    logger.info(f"  Skipped: {stats['skipped']} shots (removed or missing geometry)")
    logger.info(f"  Total receiver-phase entries: {stats['total_samples']}")

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv = output_dir / "noise_statistics.csv"
        summary_df.to_csv(summary_csv, index=False)
        total_unmatched = summary_df["unmatched_receivers"].sum()
        if total_unmatched > 0:
            logger.warning(f"  Total unmatched receiver positions across all shots: {total_unmatched}")
        logger.info(f"  Summary saved to: {summary_csv}")

    # Create the noisy_traveltimes.csv (same post-step as standard mode)
    logger.info("\nCreating noisy_traveltimes.csv with all metadata...")
    create_noisy_traveltimes_csv(df, output_dir, output_dir, output_dtype)

    return stats


if __name__ == "__main__":
    # Example standalone usage
    import yaml

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        full_config = yaml.safe_load(f)

    # Extract relevant configs
    input_dir = Path(full_config["input"]["synthetic_traveltime_dir"])
    output_dir = Path(full_config["general"]["output_dir"]) / "noisy_traveltimes"
    ground_truth_csv = Path(full_config["input"]["ground_truth_csv"])
    noise_config = full_config["traveltime_noise"]
    random_seed = full_config["general"].get("random_seed")

    # Run
    process_traveltime_directory(
        input_dir, output_dir, ground_truth_csv, noise_config, random_seed
    )
