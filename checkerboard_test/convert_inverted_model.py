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

from utils.velocity_io import (
    read_velocity_file, write_velocity_file, read_velocity_metadata,
    resolve_binary_grid, resolve_grid_metadata, get_fine_grid_metadata,
    normalize_metadata, GRID_KEYS, FINE_GRID_KEYS,
)
from utils.column_utils import normalize_coordinate_columns

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


def sample_at_coarse_points(
    velocity_3d: np.ndarray,
    coarse_df: pd.DataFrame,
    fine_grid_config: Dict,
    batch_size: int = 1000000,
) -> pd.DataFrame:
    """
    Sample fine grid velocities at coarse grid points.

    Args:
        velocity_3d: 3D velocity array (nz, ny, nx) from LATTE
        coarse_df: Coarse grid DataFrame with x, y, z
        fine_grid_config: Fine grid configuration (nx, ny, nz, dx, dy, dz, origin)
        batch_size: Number of points to process at once (default: 1M)

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
        df["z"].values,
        df["y"].values,
        df["x"].values,
    ])

    # Interpolate in batches to avoid memory issues
    n_points = len(coarse_points)
    n_batches = (n_points + batch_size - 1) // batch_size

    logger.info(f"  Processing {n_points} points in {n_batches} batches")

    sampled_velocity = np.zeros(n_points, dtype=np.float32)

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_points)
        batch_points = coarse_points[start_idx:end_idx]

        sampled_velocity[start_idx:end_idx] = interpolator(batch_points)

        if (i + 1) % 5 == 0 or i == n_batches - 1:
            logger.info(f"  Processed batch {i+1}/{n_batches} "
                       f"({end_idx}/{n_points} points)")

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
        "x": x_flat,
        "y": y_flat,
        "z": z_flat,
        field_name: velocity_flat,  # Keep in km/s for CSV
    })

    logger.info(f"  Created {len(df)} fine grid points")

    return df


def _resolve_conversion_grid(input_csv: Optional[Path], config: Dict,
                             binary_files) -> Dict:
    """
    Grid of the inverted binaries (the fine model grid).

    Read from the metadata of the input velocity model (checked against its x/y/z
    columns). Only if the input model has no grid metadata (legacy file) is the grid
    inferred from its columns, and as a last resort taken from ``config['grid']``.
    """
    metadata, df_xyz = {}, None
    if input_csv is not None and Path(input_csv).exists():
        metadata = read_velocity_metadata(input_csv)
        if not all(k in get_fine_grid_metadata(metadata) for k in GRID_KEYS):
            df_xyz = read_velocity_file(input_csv, show_progress=False, columns=['x', 'y', 'z'])
    else:
        logger.warning("No input velocity model found; the grid must be given in the configuration")
    return resolve_binary_grid(
        metadata=metadata, df=df_xyz, config_grid=config.get("grid"),
        binary_files=binary_files,
        source=Path(input_csv).name if input_csv is not None else "model_conversion",
    )


def _same_grid(a: Dict, b: Dict, tol: float = 1e-6) -> bool:
    return all(
        np.isclose(float(a[k]), float(b[k]), rtol=tol, atol=tol)
        for k in GRID_KEYS if a.get(k) is not None and b.get(k) is not None
    )


def _carry_over_geography(df_fine: pd.DataFrame, input_csv: Optional[Path],
                          grid: Dict) -> pd.DataFrame:
    """
    Fill lat/lon/depth of the full fine-grid DataFrame from the input model.

    Only possible when the input model is on the same grid; otherwise they stay NaN.
    """
    for column in ("lat", "lon", "depth"):
        df_fine[column] = np.nan
    if input_csv is None or not Path(input_csv).exists():
        logger.warning("No input model to take lat/lon/depth from; they are written as NaN")
        return df_fine

    df_in = read_velocity_file(input_csv, show_progress=False)
    in_grid = resolve_grid_metadata(df_in, source=Path(input_csv).name)
    if not _same_grid(in_grid, grid) or not {"lat", "lon", "depth"} <= set(df_in.columns):
        logger.warning("Input model is not on the fine grid (or has no lat/lon/depth); "
                       "lat/lon/depth are written as NaN")
        return df_fine

    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    idx = [np.round((df_in[a].values - grid[f"origin_{a}"]) / grid.get(f"d{a}", 1.0)).astype(int)
           for a in ("x", "y", "z")]
    for column in ("lat", "lon", "depth"):
        values = np.full((nz, ny, nx), np.nan)
        values[idx[2], idx[1], idx[0]] = df_in[column].values
        df_fine[column] = values.ravel()  # (nz, ny, nx) C-order matches the x-fastest fine grid
    return df_fine


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
    processing_mode = config.get("processing_mode", "sample")
    convert_to_kms = config.get("convert_to_kms", True)

    # Binary files
    vp_bin = input_model_dir / "updated_vp.bin"
    vs_bin = input_model_dir / "updated_vs.bin"

    # Grid of the binaries: from the input model's metadata
    grid_config = _resolve_conversion_grid(input_csv, config, [vp_bin, vs_bin])

    vp_3d = load_velocity_binary(vp_bin, grid_config["nx"], grid_config["ny"],
                                  grid_config["nz"], convert_to_kms)
    vs_3d = load_velocity_binary(vs_bin, grid_config["nx"], grid_config["ny"],
                                  grid_config["nz"], convert_to_kms)

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

        # Same points as the input model: carry over its metadata, adding the fine grid
        file_metadata = normalize_metadata(df.attrs.get("metadata"))
        own_grid = resolve_grid_metadata(df, file_metadata, source=Path(input_csv).name)
        file_metadata.update(own_grid)
        if not _same_grid(own_grid, grid_config):
            file_metadata.update({f"fine_{k}": v for k, v in grid_config.items()})

    elif processing_mode == "expand":
        # Expand to full fine grid
        df_vp = expand_to_fine_grid(vp_3d, None, grid_config, "vp")
        df_vs = expand_to_fine_grid(vs_3d, None, grid_config, "vs")

        # Merge
        df = df_vp.copy()
        df["vs"] = df_vs["vs"]
        df = _carry_over_geography(df, input_csv, grid_config)

        # Geography from the input model, grid of the full fine grid
        input_metadata = read_velocity_metadata(input_csv) if input_csv is not None and Path(input_csv).exists() else {}
        file_metadata = {k: v for k, v in input_metadata.items()
                         if k not in GRID_KEYS and k not in FINE_GRID_KEYS and not k.startswith("stride_")}
        file_metadata.update(grid_config)

    else:
        raise ValueError(f"Unknown processing_mode: {processing_mode}")

    file_metadata["data_type"] = "velocity"
    file_metadata["processing_mode"] = processing_mode

    # Write output model (CSV and/or H5 based on config)
    output_format = config.get('output_format', 'csv')
    output_csv = config.get('output_csv', output_dir / "velocity_model_final.csv")
    output_hdf5 = config.get('output_hdf5', output_dir / "velocity_model_final.h5")
    hdf5_options = config.get('hdf5_options', {})

    # Ensure output paths are Path objects
    if isinstance(output_csv, str):
        output_csv = Path(output_csv)
    if isinstance(output_hdf5, str):
        output_hdf5 = Path(output_hdf5)

    # Write CSV if requested
    if output_format in ['csv', 'both']:
        logger.info(f"Writing inverted model CSV: {output_csv}")
        write_velocity_file(df, output_csv, format='csv', metadata=file_metadata, show_progress=True)

    # Write HDF5 if requested
    if output_format in ['hdf5', 'both']:
        logger.info(f"Writing inverted model H5: {output_hdf5}")

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
            metadata=file_metadata,
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
