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
    pattern_dimension: str = "3d",
    polarity: str = "positive",
) -> np.ndarray:
    """
    Create sinusoidal checkerboard perturbation pattern.

    Args:
        x, y, z: Coordinate arrays (km)
        wavelength_x, y, z: Wavelengths in each direction (km)
            wavelength_z is ignored when pattern_dimension is "2d"
        amplitude: Perturbation amplitude (fraction, e.g., 0.05 for ±5%)
        phase_shift_x, y, z: Phase shifts (radians)
        pattern_dimension: "3d" (full 3D) or "2d" (no depth variation)
        polarity: "positive" (default) or "negative" (inverts the pattern)

    Returns:
        Perturbation factor array (e.g., 1.0 ± amplitude)
    """
    eff_amplitude = amplitude if polarity == "positive" else -amplitude

    if pattern_dimension == "2d":
        perturbation = eff_amplitude * (
            np.sin(2 * np.pi * x / wavelength_x + phase_shift_x)
            * np.sin(2 * np.pi * y / wavelength_y + phase_shift_y)
        )
    else:  # "3d"
        perturbation = eff_amplitude * (
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
    pattern_dimension: str = "3d",
    polarity: str = "positive",
) -> np.ndarray:
    """
    Create block (square wave) checkerboard perturbation pattern.

    Args:
        x, y, z: Coordinate arrays (km)
        wavelength_x, y, z: Wavelengths in each direction (km)
            wavelength_z is ignored when pattern_dimension is "2d"
        amplitude: Perturbation amplitude (fraction, e.g., 0.05 for ±5%)
        phase_shift_x, y, z: Phase shifts (radians)
        pattern_dimension: "3d" (full 3D) or "2d" (no depth variation)
        polarity: "positive" (default) or "negative" (inverts the pattern)

    Returns:
        Perturbation factor array (e.g., 1.0 ± amplitude)
    """
    eff_amplitude = amplitude if polarity == "positive" else -amplitude

    # Convert coordinates to phase
    phase_x = (2 * np.pi * x / wavelength_x + phase_shift_x) % (2 * np.pi)
    phase_y = (2 * np.pi * y / wavelength_y + phase_shift_y) % (2 * np.pi)

    # Create square waves (-1 or +1)
    block_x = np.where(phase_x < np.pi, -1.0, 1.0)
    block_y = np.where(phase_y < np.pi, -1.0, 1.0)

    if pattern_dimension == "2d":
        # No depth variation: Z component is omitted (effectively +1)
        perturbation = eff_amplitude * block_x * block_y
    else:  # "3d"
        phase_z = (2 * np.pi * z / wavelength_z + phase_shift_z) % (2 * np.pi)
        block_z = np.where(phase_z < np.pi, -1.0, 1.0)
        perturbation = eff_amplitude * block_x * block_y * block_z

    return 1.0 + perturbation


def create_planar_perturbation(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    wavelength_x: float,
    wavelength_y: float,
    amplitude: float,
    phase_shift_x: float,
    phase_shift_y: float,
    positive_depths: list,
    negative_depths: list,
    pattern_type: str,
) -> np.ndarray:
    """
    Create planar checkerboard: XY pattern applied only at specified depth planes.
    All other grid points are left unperturbed (factor = 1.0).

    Args:
        x, y, z: Coordinate arrays (km)
        wavelength_x, wavelength_y: XY cell sizes (km)
        amplitude: Perturbation amplitude (fraction, e.g., 0.05 for ±5%)
        phase_shift_x, phase_shift_y: Phase offsets (radians)
        positive_depths: Depths (km) receiving +amplitude perturbation
        negative_depths: Depths (km) receiving -amplitude perturbation
        pattern_type: "sinusoidal" or "block"

    Returns:
        Perturbation factor array (1.0 where unperturbed, 1.0 ± amplitude at planes)
    """
    perturbation_factor = np.ones(len(x), dtype=np.float64)
    z_unique = np.unique(z)

    planes = (
        [(float(d), "positive") for d in positive_depths]
        + [(float(d), "negative") for d in negative_depths]
    )

    for depth_km, polarity in planes:
        eff_amplitude = amplitude if polarity == "positive" else -amplitude

        nearest_z = z_unique[np.argmin(np.abs(z_unique - depth_km))]
        mask = np.isclose(z, nearest_z, atol=1e-6)

        if not mask.any():
            logger.warning(
                f"No grid points at depth {depth_km} km (nearest: {nearest_z:.3f} km) — skipping."
            )
            continue

        logger.info(
            f"  Plane {depth_km} km → {nearest_z:.3f} km, polarity={polarity}, points={mask.sum()}"
        )

        x_plane = x[mask]
        y_plane = y[mask]

        if pattern_type == "sinusoidal":
            xy_perturbation = eff_amplitude * (
                np.sin(2 * np.pi * x_plane / wavelength_x + phase_shift_x)
                * np.sin(2 * np.pi * y_plane / wavelength_y + phase_shift_y)
            )
        elif pattern_type == "block":
            phase_x = (2 * np.pi * x_plane / wavelength_x + phase_shift_x) % (2 * np.pi)
            phase_y = (2 * np.pi * y_plane / wavelength_y + phase_shift_y) % (2 * np.pi)
            block_x = np.where(phase_x < np.pi, -1.0, 1.0)
            block_y = np.where(phase_y < np.pi, -1.0, 1.0)
            xy_perturbation = eff_amplitude * block_x * block_y
        else:
            raise ValueError(f"Unknown pattern_type for planar perturbation: {pattern_type!r}")

        perturbation_factor[mask] = 1.0 + xy_perturbation

    return perturbation_factor


def apply_checkerboard_perturbation(
    df: pd.DataFrame,
    config: Dict,
) -> pd.DataFrame:
    """
    Apply checkerboard perturbations to velocity model.

    Args:
        df: Velocity model DataFrame with X_grid_km, Y_grid_km, Z_grid_km, vp, vs
            NOTE: This function modifies the input DataFrame in place for memory efficiency
        config: Checkerboard configuration dictionary

    Returns:
        DataFrame with perturbed velocities (same object as input)
    """
    # Note: We modify df in place to avoid memory issues with large datasets

    # Extract coordinates
    x = df["X_grid_km"].values
    y = df["Y_grid_km"].values
    z = df["Z_grid_km"].values

    # Extract parameters
    pattern_type = config.get("pattern_type", "sinusoidal")
    pattern_dimension = config.get("pattern_dimension", "3d")
    polarity = config.get("origin_polarity", "positive")
    amplitude = config.get("amplitude", 0.05)
    wavelength_x = config.get("wavelength_x_km", 10.0)
    wavelength_y = config.get("wavelength_y_km", 10.0)
    wavelength_z = config.get("wavelength_z_km", 5.0)
    phase_shift_x = config.get("phase_shift_x", 0.0)
    phase_shift_y = config.get("phase_shift_y", 0.0)
    phase_shift_z = config.get("phase_shift_z", 0.0)

    if pattern_dimension not in ("3d", "2d", "planar"):
        raise ValueError(f"Unknown pattern_dimension: {pattern_dimension!r}. Must be '3d', '2d', or 'planar'.")
    if polarity not in ("positive", "negative"):
        raise ValueError(f"Unknown origin_polarity: {polarity!r}. Must be 'positive' or 'negative'.")

    logger.info(f"Creating {pattern_type} checkerboard pattern:")
    logger.info(f"  Dimension: {pattern_dimension}")
    logger.info(f"  Amplitude: ±{amplitude * 100:.1f}%")
    logger.info(f"  Wavelengths: X={wavelength_x} km, Y={wavelength_y} km"
                + (f", Z={wavelength_z} km" if pattern_dimension == "3d" else " (Z ignored)"))
    if pattern_dimension != "planar":
        logger.info(f"  Origin polarity: {polarity}")

    # Create perturbation pattern
    if pattern_dimension == "planar":
        positive_depths = list(config.get("depth_planes_positive_polarity", []) or [])
        negative_depths = list(config.get("depth_planes_negative_polarity", []) or [])
        if not positive_depths and not negative_depths:
            raise ValueError(
                "pattern_dimension is 'planar' but both depth_planes_positive_polarity "
                "and depth_planes_negative_polarity are empty. At least one depth is required."
            )
        logger.info(f"  Positive-polarity planes: {positive_depths}")
        logger.info(f"  Negative-polarity planes: {negative_depths}")
        perturbation_factor = create_planar_perturbation(
            x, y, z,
            wavelength_x, wavelength_y,
            amplitude,
            phase_shift_x, phase_shift_y,
            positive_depths,
            negative_depths,
            pattern_type,
        )
    elif pattern_type == "sinusoidal":
        perturbation_factor = create_sinusoidal_perturbation(
            x, y, z,
            wavelength_x, wavelength_y, wavelength_z,
            amplitude,
            phase_shift_x, phase_shift_y, phase_shift_z,
            pattern_dimension=pattern_dimension,
            polarity=polarity,
        )
    elif pattern_type == "block":
        perturbation_factor = create_block_perturbation(
            x, y, z,
            wavelength_x, wavelength_y, wavelength_z,
            amplitude,
            phase_shift_x, phase_shift_y, phase_shift_z,
            pattern_dimension=pattern_dimension,
            polarity=polarity,
        )
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type!r}. Must be 'sinusoidal' or 'block'.")

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

    # Fill grid using vectorized operations (much faster than iterrows)
    # Compute all indices at once using vectorized numpy operations
    i_indices = np.round((df["X_grid_km"].values - x_min) / dx).astype(np.int32)
    j_indices = np.round((df["Y_grid_km"].values - y_min) / dy).astype(np.int32)
    k_indices = np.round((df["Z_grid_km"].values - z_min) / dz).astype(np.int32)

    # Filter valid indices
    valid_mask = (
        (i_indices >= 0) & (i_indices < nx) &
        (j_indices >= 0) & (j_indices < ny) &
        (k_indices >= 0) & (k_indices < nz)
    )

    # Apply valid indices to fill the 3D grid
    i_valid = i_indices[valid_mask]
    j_valid = j_indices[valid_mask]
    k_valid = k_indices[valid_mask]
    velocity_values = df[velocity_field].values[valid_mask] * 1000.0  # Convert km/s to m/s

    # Fill the grid (vectorized assignment)
    velocity_3d[k_valid, j_valid, i_valid] = velocity_values

    points_processed = valid_mask.sum()
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
