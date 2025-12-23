"""
Create checkerboard velocity model for resolution testing.

This module reads an input velocity model CSV, applies checkerboard perturbations
using sinusoidal or block patterns, and outputs:
- Updated CSV with perturbed velocities
- Binary files (vp_checkerboard.bin, vs_checkerboard.bin) for LATTE
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def create_sinusoidal_perturbation(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    wavelength_x: float,
    wavelength_y: float,
    wavelength_z: float,
    amplitude: float,
    phase_shift_x: float = 0.0,
    phase_shift_y: float = 0.0,
    phase_shift_z: float = 0.0,
) -> np.ndarray:
    """
    Create sinusoidal checkerboard perturbation pattern.

    Args:
        x, y, z: Coordinate arrays (km)
        wavelength_x, y, z: Wavelengths in each direction (km)
        amplitude: Perturbation amplitude (fraction, e.g., 0.05 for ±5%)
        phase_shift_x, y, z: Phase shifts (radians)

    Returns:
        Perturbation factor array (e.g., 1.0 ± amplitude)
    """
    perturbation = amplitude * (
        np.sin(2 * np.pi * x / wavelength_x + phase_shift_x)
        * np.sin(2 * np.pi * y / wavelength_y + phase_shift_y)
        * np.sin(2 * np.pi * z / wavelength_z + phase_shift_z)
    )

    # Return as multiplicative factor: 1.0 ± perturbation
    return 1.0 + perturbation


def create_block_perturbation(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    wavelength_x: float,
    wavelength_y: float,
    wavelength_z: float,
    amplitude: float,
    phase_shift_x: float = 0.0,
    phase_shift_y: float = 0.0,
    phase_shift_z: float = 0.0,
) -> np.ndarray:
    """
    Create block (square wave) checkerboard perturbation pattern.

    Args:
        x, y, z: Coordinate arrays (km)
        wavelength_x, y, z: Wavelengths in each direction (km)
        amplitude: Perturbation amplitude (fraction, e.g., 0.05 for ±5%)
        phase_shift_x, y, z: Phase shifts (radians)

    Returns:
        Perturbation factor array (e.g., 1.0 ± amplitude)
    """
    # Convert coordinates to phase
    phase_x = (2 * np.pi * x / wavelength_x + phase_shift_x) % (2 * np.pi)
    phase_y = (2 * np.pi * y / wavelength_y + phase_shift_y) % (2 * np.pi)
    phase_z = (2 * np.pi * z / wavelength_z + phase_shift_z) % (2 * np.pi)

    # Create square waves (-1 or +1)
    block_x = np.where(phase_x < np.pi, -1.0, 1.0)
    block_y = np.where(phase_y < np.pi, -1.0, 1.0)
    block_z = np.where(phase_z < np.pi, -1.0, 1.0)

    # Combine: product gives checkerboard pattern
    perturbation = amplitude * block_x * block_y * block_z

    return 1.0 + perturbation


def apply_checkerboard_perturbation(
    df: pd.DataFrame,
    config: Dict,
) -> pd.DataFrame:
    """
    Apply checkerboard perturbations to velocity model.

    Args:
        df: Velocity model DataFrame with X_grid_km, Y_grid_km, Z_grid_km, vp, vs
        config: Checkerboard configuration dictionary

    Returns:
        DataFrame with perturbed velocities
    """
    df = df.copy()

    # Extract coordinates
    x = df["X_grid_km"].values
    y = df["Y_grid_km"].values
    z = df["Z_grid_km"].values

    # Extract parameters
    pattern_type = config.get("pattern_type", "sinusoidal")
    amplitude = config.get("amplitude", 0.05)
    wavelength_x = config.get("wavelength_x_km", 10.0)
    wavelength_y = config.get("wavelength_y_km", 10.0)
    wavelength_z = config.get("wavelength_z_km", 5.0)
    phase_shift_x = config.get("phase_shift_x", 0.0)
    phase_shift_y = config.get("phase_shift_y", 0.0)
    phase_shift_z = config.get("phase_shift_z", 0.0)

    logger.info(f"Creating {pattern_type} checkerboard pattern:")
    logger.info(f"  Amplitude: ±{amplitude * 100:.1f}%")
    logger.info(f"  Wavelengths: X={wavelength_x} km, Y={wavelength_y} km, Z={wavelength_z} km")

    # Create perturbation pattern
    if pattern_type == "sinusoidal":
        perturbation_factor = create_sinusoidal_perturbation(
            x, y, z,
            wavelength_x, wavelength_y, wavelength_z,
            amplitude,
            phase_shift_x, phase_shift_y, phase_shift_z,
        )
    elif pattern_type == "block":
        perturbation_factor = create_block_perturbation(
            x, y, z,
            wavelength_x, wavelength_y, wavelength_z,
            amplitude,
            phase_shift_x, phase_shift_y, phase_shift_z,
        )
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")

    # Apply to vp
    if config.get("apply_to_vp", True):
        vp_original = df["vp"].values
        vp_perturbed = vp_original * perturbation_factor

        # Clip to valid range
        vp_min = config.get("vp_min", 500.0)
        vp_max = config.get("vp_max", 10000.0)
        vp_perturbed = np.clip(vp_perturbed, vp_min, vp_max)

        df["vp"] = vp_perturbed

        logger.info(f"  Vp: {vp_perturbed.min():.1f} - {vp_perturbed.max():.1f} m/s")
        logger.info(f"  Vp perturbation range: {((vp_perturbed / vp_original - 1.0) * 100).min():.2f}% "
                    f"to {((vp_perturbed / vp_original - 1.0) * 100).max():.2f}%")

    # Apply to vs
    if config.get("apply_to_vs", True):
        vs_original = df["vs"].values
        vs_perturbed = vs_original * perturbation_factor

        # Clip to valid range
        vs_min = config.get("vs_min", 500.0)
        vs_max = config.get("vs_max", 6000.0)
        vs_perturbed = np.clip(vs_perturbed, vs_min, vs_max)

        df["vs"] = vs_perturbed

        logger.info(f"  Vs: {vs_perturbed.min():.1f} - {vs_perturbed.max():.1f} m/s")
        logger.info(f"  Vs perturbation range: {((vs_perturbed / vs_original - 1.0) * 100).min():.2f}% "
                    f"to {((vs_perturbed / vs_original - 1.0) * 100).max():.2f}%")

    return df


def get_grid_dimensions(df: pd.DataFrame) -> Tuple[int, int, int]:
    """
    Extract grid dimensions from velocity model DataFrame.

    Args:
        df: DataFrame with X_grid_km, Y_grid_km, Z_grid_km columns

    Returns:
        (nx, ny, nz) grid dimensions
    """
    # Get unique values for each dimension
    x_unique = df["X_grid_km"].unique()
    y_unique = df["Y_grid_km"].unique()
    z_unique = df["Z_grid_km"].unique()

    nx = len(x_unique)
    ny = len(y_unique)
    nz = len(z_unique)

    logger.info(f"Grid dimensions: nx={nx}, ny={ny}, nz={nz}")

    return nx, ny, nz


def write_velocity_binary(
    df: pd.DataFrame,
    output_path: Path,
    velocity_field: str,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """
    Write velocity binary file in LATTE format.

    Format:
    - Header: 3 x int32 (nx, ny, nz)
    - Data: nx*ny*nz x float32 in Fortran (column-major) order, units: km/s

    Args:
        df: DataFrame with velocity data
        output_path: Output binary file path
        velocity_field: "vp" or "vs"
        nx, ny, nz: Grid dimensions
    """
    logger.info(f"Writing {velocity_field} binary: {output_path}")

    # Get velocity values (m/s from CSV)
    velocity_ms = df[velocity_field].values

    # Convert m/s to km/s
    velocity_kms = velocity_ms / 1000.0

    # Reshape to 3D grid
    # Assuming data is sorted by z, then y, then x (typical CSV order)
    # Need to reshape to (nz, ny, nx) and then convert to Fortran order
    try:
        velocity_3d = velocity_kms.reshape((nz, ny, nx), order='C')
    except ValueError:
        logger.error(f"Cannot reshape velocity array of size {len(velocity_kms)} "
                     f"to ({nz}, {ny}, {nx})")
        raise

    # Convert to Fortran order for LATTE
    velocity_fortran = np.asfortranarray(velocity_3d.astype(np.float32))

    # Write binary file
    with open(output_path, 'wb') as f:
        # Write header: nx, ny, nz as int32
        f.write(np.array([nx, ny, nz], dtype=np.int32).tobytes())

        # Write data
        f.write(velocity_fortran.tobytes())

    logger.info(f"  Binary file written: {output_path.stat().st_size} bytes")
    logger.info(f"  Velocity range: {velocity_kms.min():.3f} - {velocity_kms.max():.3f} km/s")


def create_checkerboard_model(
    input_csv: Path,
    output_dir: Path,
    config: Dict,
) -> Tuple[pd.DataFrame, Path, Path]:
    """
    Main function to create checkerboard velocity model.

    Args:
        input_csv: Path to input velocity model CSV
        output_dir: Output directory
        config: Checkerboard configuration

    Returns:
        (perturbed_df, vp_bin_path, vs_bin_path)
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Creating Checkerboard Velocity Model")
    logger.info("=" * 80)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input CSV
    logger.info(f"Reading input velocity model: {input_csv}")
    df = pd.read_csv(input_csv)
    logger.info(f"  Loaded {len(df)} grid points")

    # Verify required columns
    required_cols = ["X_grid_km", "Y_grid_km", "Z_grid_km", "vp", "vs"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Apply checkerboard perturbation
    df_perturbed = apply_checkerboard_perturbation(df, config)

    # Get grid dimensions
    nx, ny, nz = get_grid_dimensions(df_perturbed)

    # Write perturbed CSV
    csv_output = output_dir / "velocity_model_checkerboard.csv"
    logger.info(f"Writing perturbed velocity model CSV: {csv_output}")
    df_perturbed.to_csv(csv_output, index=False)

    # Write binary files
    vp_bin_path = output_dir / "vp_checkerboard.bin"
    vs_bin_path = output_dir / "vs_checkerboard.bin"

    write_velocity_binary(df_perturbed, vp_bin_path, "vp", nx, ny, nz)
    write_velocity_binary(df_perturbed, vs_bin_path, "vs", nx, ny, nz)

    logger.info("Checkerboard model creation completed successfully!")

    return df_perturbed, vp_bin_path, vs_bin_path


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
    input_csv = Path(full_config["input"]["velocity_model_csv"])
    output_dir = Path(full_config["general"]["output_dir"]) / "checkerboard_model"
    checkerboard_config = full_config["checkerboard"]

    # Run
    create_checkerboard_model(input_csv, output_dir, checkerboard_config)
