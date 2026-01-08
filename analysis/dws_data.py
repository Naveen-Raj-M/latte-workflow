"""
Data loading and preparation for DWS (Derivative Weight Sum) analysis.

This module handles loading DWS data from HDF5 files and preparing it
for statistical analysis and visualization.
"""

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

from utils.velocity_io import read_velocity_file, get_velocity_file_metadata

logger = logging.getLogger(__name__)


def load_dws_data(
    filepath: str,
    dataset: str = 'velocity',
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Load DWS data from HDF5 file.

    The DWS values are stored in the 'vp' column of the velocity format.

    Args:
        filepath: Path to DWS HDF5 file
        dataset: Dataset name ('velocity' or 'velocity_coarse')
        show_progress: Whether to show progress bar

    Returns:
        DataFrame with DWS data including columns:
        - X_grid_km, Y_grid_km, Z_grid_km: Grid coordinates
        - i, j, k: Grid indices
        - dws: DWS values (extracted from vp column)
    """
    logger.info(f"Loading DWS data from {filepath}")

    # Load using velocity_io
    df = read_velocity_file(
        filepath,
        format='hdf5',
        dataset=dataset,
        show_progress=show_progress
    )

    # Extract DWS from vp column and rename
    df['dws'] = df['vp']

    # Log statistics
    logger.info(f"Loaded {len(df):,} DWS grid points")
    logger.info(f"  DWS range: [{df['dws'].min():.6f}, {df['dws'].max():.6f}]")
    logger.info(f"  Non-zero cells: {(df['dws'] > 0).sum():,} ({100*(df['dws'] > 0).sum()/len(df):.2f}%)")

    return df


def get_dws_metadata(filepath: str) -> Dict[str, Any]:
    """
    Extract metadata from DWS HDF5 file.

    Args:
        filepath: Path to DWS file

    Returns:
        Dictionary with metadata
    """
    logger.info(f"Extracting metadata from {filepath}")

    metadata = get_velocity_file_metadata(filepath)

    # Check if this is a DWS file
    with h5py.File(filepath, 'r') as f:
        if 'velocity' in f:
            dset = f['velocity']
            if 'data_type' in dset.attrs:
                metadata['data_type'] = dset.attrs['data_type']
            if 'description' in dset.attrs:
                metadata['description'] = dset.attrs['description']
            if 'units' in dset.attrs:
                metadata['units'] = dset.attrs['units']

    return metadata


def extract_horizontal_slice(
    df: pd.DataFrame,
    depth_km: float,
    tolerance_km: float = 0.01
) -> pd.DataFrame:
    """
    Extract horizontal slice at specified depth.

    Args:
        df: DWS DataFrame
        depth_km: Depth in km (positive downward)
        tolerance_km: Tolerance for depth matching

    Returns:
        DataFrame containing points at the specified depth
    """
    # Filter by depth
    mask = np.abs(df['Z_grid_km'] - depth_km) < tolerance_km

    df_slice = df[mask].copy()

    if len(df_slice) == 0:
        logger.warning(f"No data found at depth {depth_km} km (tolerance: {tolerance_km} km)")
        # Try to find nearest depth
        depths = df['Z_grid_km'].unique()
        nearest_depth = depths[np.argmin(np.abs(depths - depth_km))]
        logger.info(f"  Nearest available depth: {nearest_depth} km")

    return df_slice


def extract_vertical_slice(
    df: pd.DataFrame,
    slice_type: str,
    position: float,
    tolerance_km: float = 0.01
) -> pd.DataFrame:
    """
    Extract vertical slice along X or Y direction.

    Args:
        df: DWS DataFrame
        slice_type: 'xz' (constant Y) or 'yz' (constant X)
        position: Position value in km
        tolerance_km: Tolerance for position matching

    Returns:
        DataFrame containing points along the slice
    """
    if slice_type == 'xz':
        # Slice at constant Y
        mask = np.abs(df['Y_grid_km'] - position) < tolerance_km
    elif slice_type == 'yz':
        # Slice at constant X
        mask = np.abs(df['X_grid_km'] - position) < tolerance_km
    else:
        raise ValueError(f"Invalid slice_type: {slice_type}. Use 'xz' or 'yz'")

    df_slice = df[mask].copy()

    if len(df_slice) == 0:
        logger.warning(f"No data found at {slice_type} position {position} km (tolerance: {tolerance_km} km)")

    return df_slice


def compute_dws_grid_3d(
    df: pd.DataFrame,
    nx: int,
    ny: int,
    nz: int
) -> np.ndarray:
    """
    Convert DWS DataFrame to 3D grid array.

    Args:
        df: DWS DataFrame
        nx, ny, nz: Grid dimensions

    Returns:
        3D numpy array of DWS values (nz, ny, nx)
    """
    # Initialize grid
    dws_grid = np.zeros((nz, ny, nx))

    # Fill grid
    for _, row in df.iterrows():
        i = int(row['i'])
        j = int(row['j'])
        k = int(row['k'])
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
            dws_grid[k, j, i] = row['dws']

    return dws_grid


def filter_dws_by_threshold(
    df: pd.DataFrame,
    min_threshold: float = 0.0,
    max_threshold: Optional[float] = None
) -> pd.DataFrame:
    """
    Filter DWS data by threshold values.

    Args:
        df: DWS DataFrame
        min_threshold: Minimum DWS value to include
        max_threshold: Maximum DWS value to include (None for no upper limit)

    Returns:
        Filtered DataFrame
    """
    mask = df['dws'] >= min_threshold

    if max_threshold is not None:
        mask &= df['dws'] <= max_threshold

    df_filtered = df[mask].copy()

    logger.info(
        f"Filtered DWS data: {len(df_filtered):,}/{len(df):,} points "
        f"({100*len(df_filtered)/len(df):.2f}%) with DWS in "
        f"[{min_threshold}, {max_threshold if max_threshold is not None else 'inf'}]"
    )

    return df_filtered


def compute_coverage_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute coverage statistics for DWS data.

    Args:
        df: DWS DataFrame

    Returns:
        Dictionary with coverage statistics
    """
    total_cells = len(df)
    nonzero_cells = (df['dws'] > 0).sum()
    coverage_pct = 100 * nonzero_cells / total_cells if total_cells > 0 else 0

    stats = {
        'total_cells': total_cells,
        'nonzero_cells': int(nonzero_cells),
        'coverage_percentage': coverage_pct,
        'dws_min': float(df['dws'].min()),
        'dws_max': float(df['dws'].max()),
        'dws_mean': float(df['dws'].mean()),
        'dws_median': float(df['dws'].median()),
        'dws_std': float(df['dws'].std()),
        'dws_percentiles': {
            '25': float(df['dws'].quantile(0.25)),
            '50': float(df['dws'].quantile(0.50)),
            '75': float(df['dws'].quantile(0.75)),
            '90': float(df['dws'].quantile(0.90)),
            '95': float(df['dws'].quantile(0.95)),
            '99': float(df['dws'].quantile(0.99))
        }
    }

    return stats
