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

from utils.velocity_io import read_velocity_file, write_velocity_file

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
        vp_original = df["vp"].values * 1000 # Convert to m/s
        vp_perturbed = vp_original * perturbation_factor

        # Clip to valid range
        vp_min = config.get("vp_min", 500.0)
        vp_max = config.get("vp_max", 10000.0)
        vp_perturbed = np.clip(vp_perturbed, vp_min, vp_max)

        df["vp"] = vp_perturbed / 1000.0 # Store back in km/s

        logger.info(f"  Vp: {vp_perturbed.min():.1f} - {vp_perturbed.max():.1f} m/s")
        logger.info(f"  Vp perturbation range: {((vp_perturbed / vp_original - 1.0) * 100).min():.2f}% "
                    f"to {((vp_perturbed / vp_original - 1.0) * 100).max():.2f}%")

    # Apply to vs
    if config.get("apply_to_vs", True):
        vs_original = df["vs"].values * 1000 # Convert to m/s
        vs_perturbed = vs_original * perturbation_factor

        # Clip to valid range
        vs_min = config.get("vs_min", 500.0)
        vs_max = config.get("vs_max", 6000.0)
        vs_perturbed = np.clip(vs_perturbed, vs_min, vs_max)

        df["vs"] = vs_perturbed / 1000.0 # Store back in km/s

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
    Write velocity binary file in LATTE-compatible Fortran format.

    Format:
    - Raw binary (no header), Fortran-order (column-major)
    - Data: nz*ny*nx x float32, units: m/s
    - Dimension order: (nz, ny, nx)

    Args:
        df: DataFrame with velocity data and X_grid_km, Y_grid_km, Z_grid_km columns
        output_path: Output binary file path
        velocity_field: "vp" or "vs"
        nx, ny, nz: Grid dimensions
    """
    logger.info(f"Writing {velocity_field} binary: {output_path}")

    # Create 3D grid array with natural LATTE shape (nz, ny, nx) in Fortran-order
    velocity_3d = np.zeros((nz, ny, nx), dtype=np.float32, order='F')

    # Compute grid parameters from data
    x_min = df["X_grid_km"].min()
    y_min = df["Y_grid_km"].min()
    z_min = df["Z_grid_km"].min()

    # Infer grid spacing from sorted unique values
    x_unique = np.sort(df["X_grid_km"].unique())
    y_unique = np.sort(df["Y_grid_km"].unique())
    z_unique = np.sort(df["Z_grid_km"].unique())

    dx = x_unique[1] - x_unique[0] if len(x_unique) > 1 else 1.0
    dy = y_unique[1] - y_unique[0] if len(y_unique) > 1 else 1.0
    dz = z_unique[1] - z_unique[0] if len(z_unique) > 1 else 1.0

    # Fill grid using i, j, k indices computed from coordinates
    points_processed = 0
    for _, row in df.iterrows():
        # Compute 1-based indices, then convert to 0-based
        i = int(np.round((row["X_grid_km"] - x_min) / dx))
        j = int(np.round((row["Y_grid_km"] - y_min) / dy))
        k = int(np.round((row["Z_grid_km"] - z_min) / dz))

        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
            # Convert km/s to m/s
            velocity_3d[k, j, i] = row[velocity_field] * 1000.0  # Index as [depth, y, x]
            points_processed += 1

    logger.info(f"  Processed {points_processed} grid points")

    # Write binary file in Fortran-order (no header)
    velocity_3d.flatten('F').tofile(output_path)

    # Verify the written file
    velocity_check = np.fromfile(output_path, dtype=np.float32).reshape((nz, ny, nx), order='F')

    logger.info(f"  Binary file written: {output_path.stat().st_size} bytes")
    logger.info(f"  Velocity range: {velocity_3d[velocity_3d > 0].min():.1f} - {velocity_3d.max():.1f} m/s")
    logger.info(f"  Verification: {np.allclose(velocity_3d, velocity_check)}")


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

    # Read input velocity model (CSV or H5)
    logger.info(f"Reading input velocity model: {input_csv}")
    df = read_velocity_file(input_csv, show_progress=True)
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

    # Write perturbed velocity model (CSV and/or H5 based on config)
    output_format = config.get('output_format', 'csv')
    output_csv = config.get('output_csv', output_dir / "velocity_model_checkerboard.csv")
    output_hdf5 = config.get('output_hdf5', output_dir / "velocity_model_checkerboard.h5")
    hdf5_options = config.get('hdf5_options', {})

    # Ensure output paths are Path objects
    if isinstance(output_csv, str):
        output_csv = Path(output_csv)
    if isinstance(output_hdf5, str):
        output_hdf5 = Path(output_hdf5)

    # Write CSV if requested
    if output_format in ['csv', 'both']:
        logger.info(f"Writing perturbed velocity model CSV: {output_csv}")
        write_velocity_file(df_perturbed, output_csv, format='csv', show_progress=True)

    # Write HDF5 if requested
    if output_format in ['hdf5', 'both']:
        logger.info(f"Writing perturbed velocity model H5: {output_hdf5}")

        # Prepare metadata
        metadata = {
            'checkerboard_config': {
                'pattern_type': config.get('pattern_type'),
                'amplitude': config.get('amplitude'),
                'wavelength_x_km': config.get('wavelength_x_km'),
                'wavelength_y_km': config.get('wavelength_y_km'),
                'wavelength_z_km': config.get('wavelength_z_km'),
            },
            'grid_dimensions': {
                'nx': nx,
                'ny': ny,
                'nz': nz
            }
        }

        # Extract H5 options
        compression = hdf5_options.get('compression', 'gzip')
        compression_level = hdf5_options.get('compression_level', 4)
        n_jobs = hdf5_options.get('n_jobs', 1)
        show_progress = hdf5_options.get('show_progress', True)
        coarse_grid_config = None
        if hdf5_options.get('create_coarse_dataset', False):
            coarse_grid_config = hdf5_options.get('coarse_grid')

        write_velocity_file(
            df_perturbed,
            output_hdf5,
            format='hdf5',
            compression=compression,
            compression_opts=compression_level,
            metadata=metadata,
            coarse_grid_config=coarse_grid_config,
            n_jobs=n_jobs,
            show_progress=show_progress
        )

    # Write binary files for LATTE (always generated)
    vp_bin_path = output_dir / "vp_checkerboard.bin"
    vs_bin_path = output_dir / "vs_checkerboard.bin"

    logger.info(f"Writing binary velocity files for LATTE:")
    logger.info(f"  - {vp_bin_path}")
    logger.info(f"  - {vs_bin_path}")
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
