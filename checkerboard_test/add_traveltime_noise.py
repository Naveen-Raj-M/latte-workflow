"""
Add Gaussian or uniform noise to synthetic traveltime binary files.

This module reads synthetic traveltime binary files from LATTE, adds
configurable noise, and writes new binary files for inversion testing.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
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
