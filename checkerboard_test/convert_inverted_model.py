"""
Convert inverted velocity model from LATTE binary format to CSV.

This module reads the inverted velocity binary files (updated_vp.bin, updated_vs.bin)
from LATTE and converts them to CSV format, either by sampling at coarse grid points
or expanding to the full fine grid.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from scipy.interpolate import RegularGridInterpolator

from utils.velocity_io import read_velocity_file, write_velocity_file

logger = logging.getLogger(__name__)


def load_velocity_binary(
    bin_file: Path,
    nx: int,
    ny: int,
    nz: int,
    convert_to_kms: bool = True,
) -> np.ndarray:
    """
    Load velocity binary file from LATTE.

    Format: Raw array of float32 values in Fortran order, units: m/s

    Args:
        bin_file: Path to binary file
        nx, ny, nz: Grid dimensions
        convert_to_kms: Convert from m/s to km/s

    Returns:
        3D array of velocities (nz, ny, nx)
    """
    logger.info(f"Loading binary file: {bin_file}")

    if not bin_file.exists():
        raise FileNotFoundError(f"Binary file not found: {bin_file}")

    # Read data
    data = np.fromfile(bin_file, dtype=np.float32)

    # Expected size
    expected_size = nx * ny * nz
    if data.size != expected_size:
        raise ValueError(
            f"Size mismatch: expected {expected_size} ({nx}×{ny}×{nz}), "
            f"got {data.size}"
        )

    # Reshape to 3D (Fortran order)
    array_3d = data.reshape((nz, ny, nx), order='F')

    # Convert units if requested
    if convert_to_kms:
        array_3d = array_3d / 1000.0  # m/s to km/s
        logger.info(f"  Velocity range: {array_3d.min():.3f} - {array_3d.max():.3f} km/s")
    else:
        logger.info(f"  Velocity range: {array_3d.min():.1f} - {array_3d.max():.1f} m/s")

    return array_3d


def apply_chapman_inverse_transform(
    velocity: np.ndarray,
    z_coords: np.ndarray,
    earth_radius_km: float = 6371.0,
) -> np.ndarray:
    """
    Apply Chapman inverse transform to convert from spherical to Cartesian.

    Formula: v_cartesian = exp(-depth/R) * v_spherical

    Args:
        velocity: Velocity array
        z_coords: Depth coordinates (km)
        earth_radius_km: Earth radius

    Returns:
        Transformed velocity array
    """
    logger.info("Applying Chapman inverse transform")

    # Create depth grid matching velocity array shape
    if velocity.ndim == 3:
        nz, ny, nx = velocity.shape
        depth_grid = np.tile(z_coords[:, np.newaxis, np.newaxis], (1, ny, nx))
    else:
        raise ValueError("Expected 3D velocity array")

    # Apply transform
    velocity_cartesian = np.exp(-depth_grid / earth_radius_km) * velocity

    logger.info(f"  Transformed velocity range: {velocity_cartesian.min():.3f} - "
                f"{velocity_cartesian.max():.3f}")

    return velocity_cartesian


def sample_at_coarse_points(
    velocity_3d: np.ndarray,
    coarse_df: pd.DataFrame,
    fine_grid_config: Dict,
) -> pd.DataFrame:
    """
    Sample fine grid velocities at coarse grid points.

    Args:
        velocity_3d: 3D velocity array (nz, ny, nx) from LATTE
        coarse_df: Coarse grid DataFrame with X_grid_km, Y_grid_km, Z_grid_km
        fine_grid_config: Fine grid configuration (nx, ny, nz, dx, dy, dz, origin)

    Returns:
        Updated DataFrame with sampled velocities
    """
    logger.info("Sampling velocities at coarse grid points")

    df = coarse_df.copy()

    # Extract fine grid parameters
    nx = fine_grid_config["nx"]
    ny = fine_grid_config["ny"]
    nz = fine_grid_config["nz"]
    dx = fine_grid_config["dx"]
    dy = fine_grid_config["dy"]
    dz = fine_grid_config["dz"]
    origin_x = fine_grid_config["origin_x"]
    origin_y = fine_grid_config["origin_y"]
    origin_z = fine_grid_config["origin_z"]

    # Create fine grid coordinate arrays
    x_fine = origin_x + np.arange(nx) * dx
    y_fine = origin_y + np.arange(ny) * dy
    z_fine = origin_z + np.arange(nz) * dz

    # Create interpolator
    # Note: velocity_3d is (nz, ny, nx), so points are (z, y, x)
    interpolator = RegularGridInterpolator(
        (z_fine, y_fine, x_fine),
        velocity_3d,
        method='linear',
        bounds_error=False,
        fill_value=None,
    )

    # Get coarse grid coordinates
    coarse_points = np.column_stack([
        df["Z_grid_km"].values,
        df["Y_grid_km"].values,
        df["X_grid_km"].values,
    ])

    # Interpolate
    sampled_velocity = interpolator(coarse_points)

    logger.info(f"  Sampled {len(df)} points")
    logger.info(f"  Sampled velocity range: {np.nanmin(sampled_velocity):.3f} - "
                f"{np.nanmax(sampled_velocity):.3f}")

    return sampled_velocity


def expand_to_fine_grid(
    velocity_3d: np.ndarray,
    coarse_df: pd.DataFrame,
    fine_grid_config: Dict,
    field_name: str,
) -> pd.DataFrame:
    """
    Expand to full fine grid, creating all grid points.

    Args:
        velocity_3d: 3D velocity array (nz, ny, nx) from LATTE
        coarse_df: Coarse grid DataFrame (used as template for other columns)
        fine_grid_config: Fine grid configuration
        field_name: Name for velocity field ("vp" or "vs")

    Returns:
        Full fine grid DataFrame
    """
    logger.info(f"Expanding to full fine grid for {field_name}")

    # Extract fine grid parameters
    nx = fine_grid_config["nx"]
    ny = fine_grid_config["ny"]
    nz = fine_grid_config["nz"]
    dx = fine_grid_config["dx"]
    dy = fine_grid_config["dy"]
    dz = fine_grid_config["dz"]
    origin_x = fine_grid_config["origin_x"]
    origin_y = fine_grid_config["origin_y"]
    origin_z = fine_grid_config["origin_z"]

    # Create fine grid coordinate arrays
    x_fine = origin_x + np.arange(nx) * dx
    y_fine = origin_y + np.arange(ny) * dy
    z_fine = origin_z + np.arange(nz) * dz

    # Create meshgrid
    Z, Y, X = np.meshgrid(z_fine, y_fine, x_fine, indexing='ij')

    # Flatten
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    velocity_flat = velocity_3d.flatten()

    # Create DataFrame
    df = pd.DataFrame({
        "X_grid_km": x_flat,
        "Y_grid_km": y_flat,
        "Z_grid_km": z_flat,
        field_name: velocity_flat,  # Keep in km/s for CSV
    })

    # Add grid indices (1-based, Fortran convention)
    # i corresponds to x, j to y, k to z
    df["i"] = np.tile(np.arange(1, nx + 1), ny * nz)
    df["j"] = np.tile(np.repeat(np.arange(1, ny + 1), nx), nz)
    df["k"] = np.repeat(np.arange(1, nz + 1), nx * ny)

    logger.info(f"  Created {len(df)} fine grid points")

    return df


def convert_inverted_model(
    input_model_dir: Path,
    output_dir: Path,
    input_csv: Path,
    config: Dict,
) -> pd.DataFrame:
    """
    Main function to convert inverted velocity model to CSV.

    Args:
        input_model_dir: Directory containing updated_vp.bin, updated_vs.bin
        output_dir: Output directory
        input_csv: Input (coarse) velocity model CSV
        config: Conversion configuration

    Returns:
        Converted DataFrame
    """
    logger.info("=" * 80)
    logger.info("STEP 3: Converting Inverted Model to CSV")
    logger.info("=" * 80)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract config
    grid_config = config["grid"]
    processing_mode = config.get("processing_mode", "sample")
    convert_to_kms = config.get("convert_to_kms", True)
    apply_chapman = config.get("apply_chapman_inverse", False)
    earth_radius = config.get("earth_radius_km", 6371.0)

    # Load binary files
    vp_bin = input_model_dir / "updated_vp.bin"
    vs_bin = input_model_dir / "updated_vs.bin"

    vp_3d = load_velocity_binary(vp_bin, grid_config["nx"], grid_config["ny"],
                                  grid_config["nz"], convert_to_kms)
    vs_3d = load_velocity_binary(vs_bin, grid_config["nx"], grid_config["ny"],
                                  grid_config["nz"], convert_to_kms)

    # Apply Chapman inverse transform if requested
    if apply_chapman:
        z_coords = grid_config["origin_z"] + np.arange(grid_config["nz"]) * grid_config["dz"]
        vp_3d = apply_chapman_inverse_transform(vp_3d, z_coords, earth_radius)
        vs_3d = apply_chapman_inverse_transform(vs_3d, z_coords, earth_radius)

    # Process based on mode
    if processing_mode == "sample":
        # Read input model (CSV or H5)
        logger.info(f"Reading input model: {input_csv}")
        df = read_velocity_file(input_csv, show_progress=True)
        logger.info(f"  Loaded {len(df)} coarse grid points")

        # Sample velocities
        vp_sampled = sample_at_coarse_points(vp_3d, df, grid_config)
        vs_sampled = sample_at_coarse_points(vs_3d, df, grid_config)

        # Update DataFrame (keep in km/s for CSV)
        df["vp"] = vp_sampled if convert_to_kms else vp_sampled * 1000.0
        df["vs"] = vs_sampled if convert_to_kms else vs_sampled * 1000.0

    elif processing_mode == "expand":
        # Expand to full fine grid
        df_vp = expand_to_fine_grid(vp_3d, None, grid_config, "vp")
        df_vs = expand_to_fine_grid(vs_3d, None, grid_config, "vs")

        # Merge
        df = df_vp.copy()
        df["vs"] = df_vs["vs"]

    else:
        raise ValueError(f"Unknown processing_mode: {processing_mode}")

    # Write output model (CSV and/or H5 based on config)
    output_format = config.get('output_format', 'csv')
    output_csv = config.get('output_csv', output_dir / "velocity_model_inverted.csv")
    output_hdf5 = config.get('output_hdf5', output_dir / "velocity_model_inverted.h5")
    hdf5_options = config.get('hdf5_options', {})

    # Ensure output paths are Path objects
    if isinstance(output_csv, str):
        output_csv = Path(output_csv)
    if isinstance(output_hdf5, str):
        output_hdf5 = Path(output_hdf5)

    # Write CSV if requested
    if output_format in ['csv', 'both']:
        logger.info(f"Writing inverted model CSV: {output_csv}")
        write_velocity_file(df, output_csv, format='csv', show_progress=True)

    # Write HDF5 if requested
    if output_format in ['hdf5', 'both']:
        logger.info(f"Writing inverted model H5: {output_hdf5}")

        # Prepare metadata
        metadata = {
            'grid_dimensions': {
                'nx': grid_config['nx'],
                'ny': grid_config['ny'],
                'nz': grid_config['nz']
            },
            'grid_spacing': {
                'dx': grid_config['dx'],
                'dy': grid_config['dy'],
                'dz': grid_config['dz']
            },
            'processing_mode': processing_mode,
            'apply_chapman_inverse': apply_chapman,
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
            df,
            output_hdf5,
            format='hdf5',
            compression=compression,
            compression_opts=compression_level,
            metadata=metadata,
            coarse_grid_config=coarse_grid_config,
            n_jobs=n_jobs,
            show_progress=show_progress
        )

    # Log statistics
    logger.info("\nInverted model statistics:")
    logger.info(f"  Vp: {df['vp'].min():.3f} - {df['vp'].max():.3f} km/s")
    logger.info(f"  Vs: {df['vs'].min():.3f} - {df['vs'].max():.3f} km/s")
    logger.info(f"  Total grid points: {len(df)}")

    logger.info("Model conversion completed successfully!")

    return df


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
    input_model_dir = Path(full_config["input"]["inverted_model_dir"])
    output_dir = Path(full_config["general"]["output_dir"]) / "inverted_model"
    input_csv = Path(full_config["input"]["velocity_model_csv"])
    conversion_config = full_config["model_conversion"]

    # Run
    convert_inverted_model(input_model_dir, output_dir, input_csv, conversion_config)
