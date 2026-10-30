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

from utils.io_utils import resolve_csv_path
from utils.exchange import (
    resolve_layout, match_stations, station_coordinates, check_binary_counts, check_array_size,
)
from utils.traveltime_io import read_traveltime_csv, write_traveltime_csv, event_station_distance_m

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


def _read_ground_truth(ground_truth_csv) -> pd.DataFrame:
    """
    Read the ground-truth CSV (legacy column names are accepted).

    The hypocentral source-receiver distance is not stored in the file; it is added here as
    the in-memory column ``distance_m`` (metres), used by the magnitude-distance noise model.
    It is removed again before any CSV is written.
    """
    df = read_traveltime_csv(ground_truth_csv)
    coord_cols = [f"{p}_{a}" for p in ("event", "station") for a in "xyz"]
    if all(c in df.columns for c in coord_cols):
        df["distance_m"] = event_station_distance_m(df, "hypocentral")
    return df


def compute_snr_sigma(
    distance_m: float,
    magnitude: float,
    phase: str,
    snr_config: Dict,
) -> float:
    """
    Compute per-pick traveltime uncertainty (std dev) from an SNR-based model.

    SNR(M, r) = S0 * (10^M / r^gamma) * exp(-kappa * r)
    SNR       = min(SNR, snr_max)
    sigma_t   = 1 / (2 * pi * f * SNR)

    ``distance_m`` is converted to km internally; ``kappa`` therefore has
    units of 1/km.  Returns 0.0 when SNR <= 0 or distance <= 0.

    Args:
        distance_m: Hypocentral distance in metres (from the ``distance_m``
                    column of the ground-truth CSV).
        magnitude:  Event magnitude.
        phase:      "p" or "s".
        snr_config: Sub-config dict/DictConfig with keys S0, gamma, kappa,
                    snr_max, p_wave_frequency, s_wave_frequency.

    Returns:
        sigma_t in seconds.
    """
    S0 = float(snr_config.get("S0", 0.2))
    gamma = float(snr_config.get("gamma", 1.0))
    kappa = float(snr_config.get("kappa", 0.003))
    snr_max = float(snr_config.get("snr_max", 50.0))

    if phase.lower() == "p":
        f = float(snr_config.get("p_wave_frequency", 10.0))
    else:
        f = float(snr_config.get("s_wave_frequency", 5.0))

    r_km = distance_m / 1000.0
    if r_km <= 0:
        return 0.0

    snr = S0 * (10.0 ** magnitude / r_km ** gamma) * np.exp(-kappa * r_km)
    if snr <= 0:
        return 0.0

    snr = min(snr, snr_max)
    return 1.0 / (2.0 * np.pi * f * snr)


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
    df = _read_ground_truth(ground_truth_csv)

    # Get unique shot_id and phase combinations
    combinations = df.groupby(["shot_id", "phase"]).size().reset_index(name="count")
    logger.info(f"Found {len(combinations)} unique shot/phase combinations")

    # Determine output precision
    output_precision = noise_config.get("output_precision", "single")
    output_dtype = np.float32 if output_precision == "single" else np.float64

    # ------------------------------------------------------------------ #
    # Magnitude-distance (SNR-based) noise setup                          #
    # ------------------------------------------------------------------ #
    md_noise_cfg = noise_config.get("magnitude_distance_noise", None) or {}
    md_noise_enabled = bool(md_noise_cfg.get("enabled", False)) if md_noise_cfg else False
    has_md_columns = False
    noise_tt_by_index: Optional[Dict[int, float]] = None  # df-index -> noise added

    if md_noise_enabled:
        missing_cols = [c for c in ("distance_m", "magnitude") if c not in df.columns]
        if missing_cols:
            msg = (
                f"WARNING: magnitude_distance_noise is enabled but column(s) "
                f"{missing_cols} not found in ground truth CSV. "
                f"Setting noise std to 0 and continuing workflow."
            )
            print(msg)
            logger.warning(msg)
            md_noise_enabled = False  # fall through to zero-noise path
        else:
            has_md_columns = True
            noise_tt_by_index = {}

            # Pre-compute all sigmas to report statistics
            all_sigmas = np.array([
                compute_snr_sigma(
                    float(r["distance_m"]), float(r["magnitude"]),
                    r["phase"].lower(), md_noise_cfg
                )
                for _, r in df.iterrows()
            ])
            stat_msg = (
                f"Magnitude-distance noise sigma_t: "
                f"min={all_sigmas.min():.6f} s, max={all_sigmas.max():.6f} s"
            )
            print(stat_msg)
            logger.info(stat_msg)

            # Build a fast lookup: (shot_id, phase_lower) -> group DataFrame
            groups_by_shot_phase: Dict[tuple, pd.DataFrame] = {
                (int(sid), ph.lower()): grp
                for (sid, ph), grp in df.groupby(["shot_id", "phase"])
            }

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
        if md_noise_enabled and has_md_columns:
            group = groups_by_shot_phase.get((shot_id, phase))
            if group is not None and len(group) == len(traveltime):
                sigmas = np.array([
                    compute_snr_sigma(
                        float(r["distance_m"]), float(r["magnitude"]),
                        phase, md_noise_cfg
                    )
                    for _, r in group.iterrows()
                ])
                noise = np.array([
                    random_state.normal(0.0, s) if s > 0 else 0.0
                    for s in sigmas
                ])
                noisy_traveltime = traveltime + noise
                for idx, nv in zip(group.index, noise):
                    noise_tt_by_index[idx] = float(nv)
            else:
                logger.warning(
                    f"Group size mismatch for shot {shot_id} phase {phase} "
                    f"(group={len(group) if group is not None else 'None'}, "
                    f"binary={len(traveltime)}); falling back to standard noise."
                )
                noisy_traveltime = add_noise_to_traveltime(
                    traveltime, noise_config, phase, random_state
                )
        elif md_noise_enabled and not has_md_columns:
            # Columns were missing — zero noise, copy original
            noisy_traveltime = traveltime.copy()
        else:
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

    # Build extra_columns DataFrame for magnitude-distance noise picks
    extra_columns: Optional[pd.DataFrame] = None
    if noise_tt_by_index:
        idx_list = list(noise_tt_by_index.keys())
        extra_columns = pd.DataFrame(
            {
                "magnitude": df.loc[idx_list, "magnitude"].values,
                "noise_tt": [noise_tt_by_index[i] for i in idx_list],
            },
            index=idx_list,
        )

    # Create a comprehensive CSV with all ground truth columns, replacing observed_tt with noisy values
    logger.info("\nCreating noisy_traveltimes.csv with all metadata...")
    create_noisy_traveltimes_csv(df, output_dir, output_dir, output_dtype, extra_columns=extra_columns)

    return stats


def create_noisy_traveltimes_csv(
    ground_truth_df: pd.DataFrame,
    noisy_traveltime_dir: Path,
    output_dir: Path,
    dtype: np.dtype = np.float32,
    extra_columns: Optional[pd.DataFrame] = None,
) -> None:
    """
    Create a CSV file with all ground truth columns, replacing observed_tt with noisy values.

    Args:
        ground_truth_df: Ground truth DataFrame with all metadata
        noisy_traveltime_dir: Directory containing noisy traveltime binary files
        output_dir: Output directory for CSV
        dtype: Data type of binary files
        extra_columns: Optional DataFrame (same index as ground_truth_df) with
                       additional columns to append (e.g. ``magnitude``,
                       ``noise_tt`` from magnitude-distance noise mode).
                       Only columns present in this DataFrame are added.
    """
    # Create a copy of ground truth dataframe to preserve all columns
    output_df = ground_truth_df.copy()

    # Reset observed_tt to NaN so rows without a matching binary are clearly missing
    output_df["observed_tt"] = np.nan

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

                # Subtract configured origin time so the noisy traveltime is physical travel time only,
                # consistent with the original observed_tt convention (no offset) and postprocessing.
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
                    output_df.loc[indices[:n_values], "observed_tt"] = noisy_tt[:n_values]
                else:
                    # Replace observed_tt with noisy traveltime for this shot/phase combination
                    output_df.loc[indices, "observed_tt"] = noisy_tt

            except Exception as e:
                logger.error(f"Error reading {binary_file}: {e}")
        else:
            logger.warning(f"Binary file not found: {binary_file}")

    # Attach per-pick extra columns (e.g. magnitude, noise_tt) when provided
    if extra_columns is not None and not extra_columns.empty:
        for col in extra_columns.columns:
            output_df[col] = extra_columns[col]

    # Save to CSV
    output_csv = output_dir / "noisy_traveltimes.csv"
    # distance_m is an in-memory working column and is not written
    output_df = output_df.drop(columns=["distance_m"], errors="ignore")
    write_traveltime_csv(output_df, output_csv, data_type="noisy_traveltime")
    logger.info(f"Noisy traveltimes CSV saved to: {output_csv}")

    # Log how many rows have valid observed_tt values
    valid_count = output_df["observed_tt"].notna().sum()
    total_count = len(output_df)
    logger.info(f"  {valid_count}/{total_count} rows have valid observed_tt (noisy) values")


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


def process_traveltime_directory_with_exchange(
    input_dir: Path,
    output_dir: Path,
    geometry_dir: Path,
    ground_truth_csv: Path,
    noise_config: Dict,
    exchange_geometry_file: Optional[Path] = None,
    filtered_shot_ids: Optional[List[int]] = None,
    filtered_receiver_ids: Optional[List[str]] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, int]:
    """
    Process traveltime binary files produced by a source-receiver exchange
    forward run, re-assemble them into the standard shot format, apply noise,
    and write output binaries **identical in format** to those produced by a
    standard (no-exchange) forward run.

    The exchange is rebuilt from the geometry files exactly as LATTE does it (virtual sources are the
    unique receivers in order of first appearance, array index = kept shot in geometry.txt order).
    ``exchange_geometry_file`` (LATTE output) is optional: if given it is the virtual-source table and
    mismatches with the geometry only warn; without it any count mismatch raises an error.

    Receiver ordering in the output binary follows the sorted station name
    order, consistent with how the standard postprocess module reads LATTE
    output binaries (sort_values('station')).

    Args:
        input_dir: Directory with exchange binaries (shot_<virtual_id>_traveltime_{p|s}.bin).
        output_dir: Directory for output standard binaries (shot_<id>_traveltime_{p|s}.bin).
        geometry_dir: Directory with geometry.txt and shot_N_geometry.txt files.
        ground_truth_csv: CSV with shot_id, phase, station, origin_time_s, and
                          station_x/y/z columns.
        exchange_geometry_file: Optional LATTE exchange_geometry.txt (metres).
        noise_config: Noise configuration dict (same schema as standard mode).
        filtered_shot_ids: Shot IDs LATTE dropped (compact the binary arrays).
        filtered_receiver_ids: Station names LATTE dropped.
        random_seed: Random seed for reproducibility.

    Returns:
        Statistics dict with keys 'processed', 'skipped', 'total_samples'.
    """
    logger.info("=" * 80)
    logger.info("STEP 2: Adding Noise to Synthetic Traveltimes (SOURCE-RECEIVER EXCHANGE MODE)")
    logger.info("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)
    random_state = np.random.RandomState(random_seed)
    removed_set = set(int(x) for x in (filtered_shot_ids or []))

    output_precision = noise_config.get("output_precision", "single")
    output_dtype = np.float32 if output_precision == "single" else np.float64

    # ------------------------------------------------------------------ #
    # 1. Ground-truth CSV                                                  #
    # ------------------------------------------------------------------ #
    logger.info(f"Reading ground truth CSV: {ground_truth_csv}")
    df = _read_ground_truth(ground_truth_csv)
    logger.info(f"  Ground truth: {len(df)} records")
    logger.info(f"  Unique shots: {df['shot_id'].nunique()}, unique stations: {df['station'].nunique()}")

    unique_shot_ids = sorted(df['shot_id'].unique())
    phases_in_data = sorted(df['phase'].str.upper().unique())
    logger.info(f"  Phases: {phases_in_data}")

    # ------------------------------------------------------------------ #
    # 1b. Magnitude-distance (SNR-based) noise setup                      #
    # ------------------------------------------------------------------ #
    md_noise_cfg = noise_config.get("magnitude_distance_noise", None) or {}
    md_noise_enabled = bool(md_noise_cfg.get("enabled", False)) if md_noise_cfg else False
    has_md_columns = False
    md_noise_extra: Dict[Tuple[int, str, str], Tuple[float, float]] = {}  # (shot_id, station, phase_upper) -> (noise_tt, magnitude)

    if md_noise_enabled:
        missing_cols = [c for c in ("distance_m", "magnitude") if c not in df.columns]
        if missing_cols:
            msg = (
                f"WARNING: magnitude_distance_noise is enabled but column(s) "
                f"{missing_cols} not found in ground truth CSV. "
                f"Setting noise std to 0 and continuing workflow."
            )
            print(msg)
            logger.warning(msg)
            md_noise_enabled = False
        else:
            has_md_columns = True
            all_sigmas = np.array([
                compute_snr_sigma(
                    float(r["distance_m"]), float(r["magnitude"]),
                    r["phase"].lower(), md_noise_cfg
                )
                for _, r in df.iterrows()
            ])
            stat_msg = (
                f"Magnitude-distance noise sigma_t: "
                f"min={all_sigmas.min():.6f} s, max={all_sigmas.max():.6f} s"
            )
            print(stat_msg)
            logger.info(stat_msg)

    # ------------------------------------------------------------------ #
    # 2. Exchange layout: kept shots (array index -> shot_id) and virtual   #
    #    sources, rebuilt from the geometry files                          #
    # ------------------------------------------------------------------ #
    coordinate_tolerance = noise_config.get("coordinate_tolerance", 1.0)  # m
    layout = resolve_layout(
        df,
        geometry_dir=geometry_dir,
        exchange_geometry_file=exchange_geometry_file,
        filtered_shot_ids=filtered_shot_ids or [],
        filtered_receiver_ids=filtered_receiver_ids or [],
        tolerance=coordinate_tolerance,
    )
    shot_id_to_array_index = {sid: idx for idx, sid in enumerate(layout.shot_ids)}
    expected_array_size = layout.n_shots
    logger.info(f"Exchange layout from {layout.source}: {layout.n_virtual} virtual sources, {layout.n_shots} shots")
    if removed_set:
        logger.info(f"Filtered shots (dropped by LATTE): {sorted(removed_set)}")

    # ------------------------------------------------------------------ #
    # 3. Match virtual sources → station names                            #
    # ------------------------------------------------------------------ #
    virtual_to_stations = match_stations(layout.virtual_coords, station_coordinates(df), layout.matching_tolerance(coordinate_tolerance))
    station_to_virtual_id: Dict[str, int] = {
        name: vid for vid, names in virtual_to_stations.items() for name in names
    }
    if len(virtual_to_stations) < layout.n_virtual:
        logger.warning(f"  {layout.n_virtual - len(virtual_to_stations)} virtual sources matched no station "
                       f"within {coordinate_tolerance} m")

    # ------------------------------------------------------------------ #
    # 4. Load all exchange binaries into memory                           #
    # ------------------------------------------------------------------ #
    logger.info(f"\nLoading exchange traveltime binaries from: {input_dir}")
    check_binary_counts(layout, input_dir, phases_in_data)
    exchange_data: Dict[Tuple[int, str], np.ndarray] = {}

    for phase in phases_in_data:
        phase_lower = phase.lower()
        loaded = 0
        for vid in range(1, layout.n_virtual + 1):
            bin_path = input_dir / f"shot_{vid}_traveltime_{phase_lower}.bin"
            if not bin_path.exists():
                continue
            data = read_traveltime_binary(bin_path)
            if data is None:
                logger.warning(f"  Could not read {bin_path.name}, skipping")
                continue
            check_array_size(layout, bin_path.name, len(data))
            exchange_data[(vid, phase)] = data
            loaded += 1
        logger.info(f"  {phase}-wave: loaded {loaded}/{layout.n_virtual} binaries")

    # ------------------------------------------------------------------ #
    # 5. For every real shot, assemble traveltime array in sorted-station  #
    #    order (matching standard LATTE output convention), apply noise,   #
    #    and write output binary.                                           #
    # ------------------------------------------------------------------ #
    stats = {"processed": 0, "skipped": 0, "total_samples": 0}
    missing_pairs: List[Tuple] = []
    noisy_tt_map: Dict[Tuple[int, str, str], float] = {}  # (shot_id, station, phase_upper) -> physical tt

    # Pre-load geometry receiver counts for validation (if geometry_dir is provided)
    geom_receiver_counts: Dict[int, int] = {}
    if geometry_dir is not None and geometry_dir.exists():
        logger.info(f"\nValidating receiver counts against geometry files in: {geometry_dir}")
        count_ok = 0
        for shot_id in unique_shot_ids:
            if shot_id in removed_set:
                continue
            geom_file = geometry_dir / f"shot_{shot_id}_geometry.txt"
            if geom_file.exists():
                try:
                    _, recv = read_shot_geometry_file(geom_file)
                    geom_receiver_counts[shot_id] = len(recv)
                    count_ok += 1
                except Exception:
                    pass
        logger.info(f"  Loaded receiver counts for {count_ok}/{len(unique_shot_ids) - len(removed_set)} shots")

    logger.info("\nAssembling standard-format output binaries per shot...")
    for shot_id in tqdm(unique_shot_ids, desc="Processing shots"):
        if shot_id in removed_set:
            stats["skipped"] += 1
            continue

        array_idx = shot_id_to_array_index[shot_id]
        shot_gt = df[df['shot_id'] == shot_id]

        for phase in phases_in_data:
            phase_lower = phase.lower()
            phase_gt = shot_gt[shot_gt['phase'] == phase]

            if phase_gt.empty:
                continue

            # Sort by station name — same order as standard postprocess mode
            phase_gt_sorted = phase_gt.sort_values('station')
            stations_sorted = phase_gt_sorted['station'].tolist()
            nr = len(stations_sorted)

            # Validate against geometry file receiver count if available
            geom_nr = geom_receiver_counts.get(shot_id)
            if geom_nr is not None and geom_nr != nr:
                logger.warning(
                    f"  Shot {shot_id} phase {phase}: ground truth has {nr} receivers "
                    f"but geometry file lists {geom_nr}. Using ground truth count."
                )

            matched_stations: List[str] = []
            tt_list: List[float] = []

            for station in stations_sorted:
                vid = station_to_virtual_id.get(station)
                if vid is None:
                    missing_pairs.append((shot_id, station, phase))
                    continue
                exchange_arr = exchange_data.get((vid, phase))
                if exchange_arr is None:
                    missing_pairs.append((shot_id, station, phase))
                    continue
                matched_stations.append(station)
                tt_list.append(float(exchange_arr[array_idx]))

            if not matched_stations:
                continue

            tt_array = np.array(tt_list, dtype=np.float64)

            # Apply noise
            if md_noise_enabled and has_md_columns:
                gt_by_station = phase_gt_sorted.set_index('station')
                sigmas = np.array([
                    compute_snr_sigma(
                        float(gt_by_station.loc[st, 'distance_m']),
                        float(gt_by_station.loc[st, 'magnitude']),
                        phase_lower, md_noise_cfg
                    )
                    for st in matched_stations
                ])
                noise_vals = np.array([
                    random_state.normal(0.0, s) if s > 0 else 0.0
                    for s in sigmas
                ])
                noisy_tt = tt_array + noise_vals
            elif md_noise_enabled and not has_md_columns:
                noisy_tt = tt_array.copy()
                noise_vals = np.zeros(len(tt_array))
            else:
                noisy_tt = add_noise_to_traveltime(tt_array, noise_config, phase_lower, random_state)
                noise_vals = None

            # Write binary with only matched entries (shorter than full geometry count when stations are missing)
            output_file = output_dir / f"shot_{shot_id}_traveltime_{phase_lower}.bin"
            write_traveltime_binary(noisy_tt, output_file, output_dtype)

            # Store physical tt (noisy - origin_time) for CSV building
            origin_time_per_station = phase_gt_sorted.set_index('station')['origin_time_s'].to_dict()
            gt_by_station_mag = phase_gt_sorted.set_index('station')['magnitude'].to_dict() if has_md_columns else {}
            for i, (station, noisy_val) in enumerate(zip(matched_stations, noisy_tt)):
                ot = origin_time_per_station.get(station, phase_gt_sorted['origin_time_s'].iloc[0])
                noisy_tt_map[(shot_id, station, phase)] = float(noisy_val) - ot
                if noise_vals is not None and has_md_columns:
                    md_noise_extra[(shot_id, station, phase)] = (
                        float(noise_vals[i]),
                        float(gt_by_station_mag.get(station, float('nan'))),
                    )

            stats["processed"] += 1
            stats["total_samples"] += len(matched_stations)

    # ------------------------------------------------------------------ #
    # 6. Summary                                                           #
    # ------------------------------------------------------------------ #
    logger.info("\nExchange noise addition completed:")
    logger.info(f"  Processed: {stats['processed']} shot-phase binaries")
    logger.info(f"  Skipped:   {stats['skipped']} shots (removed or not in ground truth)")
    logger.info(f"  Total receiver entries written: {stats['total_samples']}")
    if missing_pairs:
        logger.warning(f"  {len(missing_pairs)} (shot, station, phase) pairs had no matching virtual binary")

    # Noise statistics CSV (receiver counts per shot-phase)
    summary_data = []
    for shot_id in unique_shot_ids:
        if shot_id in removed_set:
            continue
        for phase in phases_in_data:
            phase_lower = phase.lower()
            output_file = output_dir / f"shot_{shot_id}_traveltime_{phase_lower}.bin"
            if output_file.exists():
                actual_nr = len(read_traveltime_binary(output_file, dtype=output_dtype))
                summary_data.append({
                    "shot_id": shot_id,
                    "phase": phase_lower,
                    "num_receivers": actual_nr,
                })
    if summary_data:
        summary_csv = output_dir / "noise_statistics.csv"
        pd.DataFrame(summary_data).to_csv(summary_csv, index=False)
        logger.info(f"  Summary saved to: {summary_csv}")

    # Build noisy_traveltimes.csv directly from in-memory map — only rows with a matching
    # virtual binary are included; missing stations are dropped (not set to -50).
    logger.info("\nCreating noisy_traveltimes.csv with all metadata...")
    if noisy_tt_map:
        map_records = [
            {'shot_id': k[0], 'station': k[1], 'phase_upper': k[2], 'observed_tt': v}
            for k, v in noisy_tt_map.items()
        ]
        map_df = pd.DataFrame(map_records)

        # Attach magnitude-distance noise extra columns when available
        if md_noise_extra:
            extra_records = [
                {'shot_id': k[0], 'station': k[1], 'phase_upper': k[2],
                 'noise_tt': v[0], 'magnitude': v[1]}
                for k, v in md_noise_extra.items()
            ]
            extra_df = pd.DataFrame(extra_records)
            map_df = map_df.merge(extra_df, on=['shot_id', 'station', 'phase_upper'], how='left')

        df_work = df.copy()
        df_work['phase_upper'] = df_work['phase'].str.upper()
        df_work = df_work.drop(columns=['observed_tt'], errors='ignore')
        output_df = df_work.merge(map_df, on=['shot_id', 'station', 'phase_upper'], how='inner')
        output_df = output_df.drop(columns=['phase_upper'])
    else:
        output_df = df.iloc[:0].copy()
        output_df['observed_tt'] = pd.Series(dtype=float)
    output_csv = output_dir / "noisy_traveltimes.csv"
    # distance_m is an in-memory working column and is not written
    output_df = output_df.drop(columns=["distance_m"], errors="ignore")
    write_traveltime_csv(output_df, output_csv, data_type="noisy_traveltime")
    logger.info(f"Noisy traveltimes CSV saved to: {output_csv}")
    removed_count = len(df) - len(output_df)
    logger.info(f"  {len(output_df)}/{len(df)} rows have valid observed_tt (noisy) values"
                + (f" ({removed_count} removed — no matching virtual binary)" if removed_count else ""))

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
