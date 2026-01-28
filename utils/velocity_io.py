"""
Velocity file I/O utilities for the LATTE tomography package.

This module provides a unified interface for reading and writing velocity files
in both CSV and HDF5 formats, with support for:
- Automatic format detection
- Coarse grid subsampling
- Parallel processing
- Progress tracking with tqdm
- HDF5 compression

Author: LATTE Workflow Team
"""

import os
import logging
from pathlib import Path
from typing import Union, Dict, Any, Optional, List, Tuple
import warnings

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from joblib import Parallel, delayed

from utils.io_utils import ensure_directory_exists

logger = logging.getLogger(__name__)

# Expected column names for velocity DataFrames (17 columns)
VELOCITY_COLUMNS = [
    'X_grid_km', 'Y_grid_km', 'Z_grid_km',  # Grid coordinates
    'x', 'y', 'z',  # Alternate coordinate names
    'i_raw', 'j_raw', 'k_raw',  # Raw grid indices
    'i', 'j', 'k',  # Integer grid indices
    'vp', 'vs',  # Velocity values
    'lon', 'lat', 'depth'  # Geographic coordinates
]


def _detect_format(filepath: Union[str, Path]) -> str:
    """
    Auto-detect file format from extension.

    Args:
        filepath: Path to file

    Returns:
        'csv' or 'hdf5'

    Raises:
        ValueError: If extension is not recognized
    """
    ext = Path(filepath).suffix.lower()
    if ext == '.csv':
        return 'csv'
    elif ext in ['.h5', '.hdf5']:
        return 'hdf5'
    else:
        raise ValueError(
            f"Cannot detect format from extension '{ext}'. "
            f"Use .csv, .h5, or .hdf5, or specify format explicitly."
        )


def _validate_coarse_grid_config(
    coarse_config: Dict[str, float],
    fine_dx: float,
    fine_dy: float,
    fine_dz: float
) -> Tuple[int, int, int]:
    """
    Validate coarse grid configuration and compute stride factors.

    Args:
        coarse_config: Dictionary with dx_coarse, dy_coarse, dz_coarse
        fine_dx: Fine grid spacing in X dimension
        fine_dy: Fine grid spacing in Y dimension
        fine_dz: Fine grid spacing in Z dimension

    Returns:
        Tuple of (stride_x, stride_y, stride_z)

    Raises:
        ValueError: If coarse spacing is not integral multiple of fine spacing
    """
    dx_coarse = coarse_config['dx_coarse']
    dy_coarse = coarse_config['dy_coarse']
    dz_coarse = coarse_config['dz_coarse']

    # Validate integral multiples
    ratio_x = dx_coarse / fine_dx
    ratio_y = dy_coarse / fine_dy
    ratio_z = dz_coarse / fine_dz

    tol = 1e-6

    if abs(ratio_x - round(ratio_x)) > tol:
        raise ValueError(
            f"dx_coarse ({dx_coarse}) must be an integral multiple of "
            f"fine dx ({fine_dx}). Ratio: {ratio_x}"
        )
    if abs(ratio_y - round(ratio_y)) > tol:
        raise ValueError(
            f"dy_coarse ({dy_coarse}) must be an integral multiple of "
            f"fine dy ({fine_dy}). Ratio: {ratio_y}"
        )
    if abs(ratio_z - round(ratio_z)) > tol:
        raise ValueError(
            f"dz_coarse ({dz_coarse}) must be an integral multiple of "
            f"fine dz ({fine_dz}). Ratio: {ratio_z}"
        )

    stride_x = round(ratio_x)
    stride_y = round(ratio_y)
    stride_z = round(ratio_z)

    return stride_x, stride_y, stride_z


def _subsample_chunk(
    df_chunk: pd.DataFrame,
    x_coarse: List[float],
    y_coarse: List[float],
    z_coarse: List[float],
    tol: float = 1e-6
) -> pd.DataFrame:
    """
    Subsample a chunk of DataFrame to coarse grid points.

    This is a helper function for parallel processing.

    Args:
        df_chunk: Chunk of velocity DataFrame
        x_coarse: List of coarse X coordinates
        y_coarse: List of coarse Y coordinates
        z_coarse: List of coarse Z coordinates
        tol: Tolerance for coordinate matching

    Returns:
        Filtered DataFrame with only coarse grid points
    """
    # Use isin for exact matches (faster)
    mask = (
        df_chunk['X_grid_km'].isin(x_coarse) &
        df_chunk['Y_grid_km'].isin(y_coarse) &
        df_chunk['Z_grid_km'].isin(z_coarse)
    )
    return df_chunk[mask].copy()


def subsample_to_coarse_grid(
    df: pd.DataFrame,
    dx_coarse: float,
    dy_coarse: float,
    dz_coarse: float,
    fine_dx: float,
    fine_dy: float,
    fine_dz: float,
    n_jobs: int = 1,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Create coarse-sampled version of velocity grid with parallel processing.

    Args:
        df: Fine grid DataFrame
        dx_coarse: Coarse grid spacing in X (km)
        dy_coarse: Coarse grid spacing in Y (km)
        dz_coarse: Coarse grid spacing in Z (km)
        fine_dx: Fine grid spacing in X (km)
        fine_dy: Fine grid spacing in Y (km)
        fine_dz: Fine grid spacing in Z (km)
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for no parallelization)
        show_progress: Whether to show progress bar

    Returns:
        Coarse-sampled DataFrame

    Raises:
        ValueError: If coarse spacing is not integral multiple of fine spacing

    Algorithm:
        1. Validate coarse spacing is integral multiple of fine spacing
        2. Compute stride factors
        3. Extract unique coordinates per dimension
        4. Select every Nth coordinate using stride
        5. Ensure grid boundaries are included (edge case handling)
        6. Filter DataFrame to coarse grid points using parallel processing
        7. Validate output dimensions
    """
    if show_progress:
        logger.info(
            f"Subsampling velocity grid: "
            f"({fine_dx}, {fine_dy}, {fine_dz}) km → "
            f"({dx_coarse}, {dy_coarse}, {dz_coarse}) km"
        )

    # Validate and compute stride factors
    coarse_config = {
        'dx_coarse': dx_coarse,
        'dy_coarse': dy_coarse,
        'dz_coarse': dz_coarse
    }
    stride_x, stride_y, stride_z = _validate_coarse_grid_config(
        coarse_config, fine_dx, fine_dy, fine_dz
    )

    # Get unique coordinates for each dimension (sorted)
    x_unique = sorted(df['X_grid_km'].unique())
    y_unique = sorted(df['Y_grid_km'].unique())
    z_unique = sorted(df['Z_grid_km'].unique())

    # Select every Nth coordinate
    x_coarse = x_unique[::stride_x]
    y_coarse = y_unique[::stride_y]
    z_coarse = z_unique[::stride_z]

    # Ensure grid boundaries are included (edge case handling)
    if x_unique[-1] not in x_coarse:
        x_coarse.append(x_unique[-1])
    if y_unique[-1] not in y_coarse:
        y_coarse.append(y_unique[-1])
    if z_unique[-1] not in z_coarse:
        z_coarse.append(z_unique[-1])

    # Expected dimensions
    nx_coarse = len(x_coarse)
    ny_coarse = len(y_coarse)
    nz_coarse = len(z_coarse)
    expected_points = nx_coarse * ny_coarse * nz_coarse

    if show_progress:
        logger.info(
            f"Coarse grid: {nx_coarse} × {ny_coarse} × {nz_coarse} = "
            f"{expected_points:,} points (stride: {stride_x}, {stride_y}, {stride_z})"
        )

    # Filter DataFrame to coarse grid points
    if n_jobs == 1:
        # No parallelization
        if show_progress:
            with tqdm(total=len(df), desc="Filtering to coarse grid") as pbar:
                df_coarse = _subsample_chunk(df, x_coarse, y_coarse, z_coarse)
                pbar.update(len(df))
        else:
            df_coarse = _subsample_chunk(df, x_coarse, y_coarse, z_coarse)
    else:
        # Parallel processing
        # Determine number of chunks
        n_cores = n_jobs if n_jobs > 0 else -1
        n_chunks = max(os.cpu_count() if n_cores == -1 else n_cores, 1)
        chunks = np.array_split(df, n_chunks)

        if show_progress:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_subsample_chunk)(chunk, x_coarse, y_coarse, z_coarse)
                for chunk in tqdm(chunks, desc="Subsampling chunks")
            )
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_subsample_chunk)(chunk, x_coarse, y_coarse, z_coarse)
                for chunk in chunks
            )

        df_coarse = pd.concat(results, ignore_index=True)

    # Validate output dimensions
    actual_points = len(df_coarse)
    if actual_points != expected_points:
        warnings.warn(
            f"Coarse grid point mismatch: expected {expected_points:,}, "
            f"got {actual_points:,}. This may indicate irregular grid spacing."
        )

    if show_progress:
        reduction_ratio = len(df) / len(df_coarse)
        logger.info(
            f"Subsampling complete: {len(df):,} → {len(df_coarse):,} points "
            f"({reduction_ratio:.1f}x reduction)"
        )

    return df_coarse


def _read_csv_velocity(
    filepath: Union[str, Path],
    show_progress: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Read velocity from CSV file with progress tracking.

    Args:
        filepath: Path to CSV file
        show_progress: Whether to show progress bar
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        Velocity DataFrame
    """
    if show_progress:
        # Count total lines for progress bar
        with open(filepath, 'r') as f:
            total_lines = sum(1 for _ in f) - 1  # Subtract header

        # Read with progress
        chunksize = kwargs.pop('chunksize', 10000)
        chunks = pd.read_csv(filepath, chunksize=chunksize, **kwargs)

        with tqdm(total=total_lines, desc=f"Reading {Path(filepath).name}") as pbar:
            df_list = []
            for chunk in chunks:
                df_list.append(chunk)
                pbar.update(len(chunk))

        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.read_csv(filepath, **kwargs)

    return df


def _read_hdf5_velocity(
    filepath: Union[str, Path],
    dataset: str = 'velocity',
    show_progress: bool = True,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Read velocity from HDF5 file.

    Args:
        filepath: Path to HDF5 file
        dataset: Dataset name ('velocity' or 'velocity_coarse')
        show_progress: Whether to show progress bar
        columns: Optional list of column names to read (default: read all)

    Returns:
        Velocity DataFrame
    """
    desc = f"Reading {Path(filepath).name}"
    if dataset == 'velocity_coarse':
        desc += " (coarse)"

    if show_progress:
        with tqdm(total=100, desc=desc) as pbar:
            with h5py.File(filepath, 'r', locking=False) as f:
                pbar.update(10)

                if dataset not in f:
                    raise KeyError(
                        f"Dataset '{dataset}' not found in {filepath}. "
                        f"Available datasets: {list(f.keys())}"
                    )

                pbar.set_description(f"{desc} - Loading data")

                # Get column names from attributes
                if 'columns' in f[dataset].attrs:
                    all_columns = f[dataset].attrs['columns']
                    if isinstance(all_columns[0], bytes):
                        all_columns = [c.decode('utf-8') for c in all_columns]
                else:
                    # Use default column names
                    all_columns = VELOCITY_COLUMNS

                # Load data (all columns or selected columns)
                if columns is None:
                    data = f[dataset][:]
                    selected_columns = all_columns
                else:
                    # Read only selected columns
                    col_indices = [all_columns.index(c) for c in columns if c in all_columns]
                    data = f[dataset][:, col_indices]
                    selected_columns = [all_columns[i] for i in col_indices]

                pbar.update(70)

                pbar.set_description(f"{desc} - Creating DataFrame")
                df = pd.DataFrame(data, columns=selected_columns)
                pbar.update(20)

        logger.info(f"Loaded {len(df):,} points from {filepath}:{dataset}")
    else:
        with h5py.File(filepath, 'r', locking=False) as f:
            if dataset not in f:
                raise KeyError(
                    f"Dataset '{dataset}' not found in {filepath}. "
                    f"Available datasets: {list(f.keys())}"
                )

            # Get column names from attributes
            if 'columns' in f[dataset].attrs:
                all_columns = f[dataset].attrs['columns']
                if isinstance(all_columns[0], bytes):
                    all_columns = [c.decode('utf-8') for c in all_columns]
            else:
                all_columns = VELOCITY_COLUMNS

            # Load data (all columns or selected columns)
            if columns is None:
                data = f[dataset][:]
                selected_columns = all_columns
            else:
                # Read only selected columns
                col_indices = [all_columns.index(c) for c in columns if c in all_columns]
                data = f[dataset][:, col_indices]
                selected_columns = [all_columns[i] for i in col_indices]

            df = pd.DataFrame(data, columns=selected_columns)

    return df


def _write_hdf5_velocity(
    filepath: Union[str, Path],
    df: pd.DataFrame,
    dataset: str,
    metadata: Optional[Dict[str, Any]],
    compression: str,
    compression_opts: Optional[int],
    show_progress: bool = True,
    chunk_size: int = 1000000
) -> None:
    """
    Write velocity to HDF5 file with metadata and compression.

    Args:
        filepath: Output HDF5 file path
        df: Velocity DataFrame
        dataset: Dataset name
        metadata: Metadata dictionary to store as attributes
        compression: Compression type ('gzip', 'lzf', None)
        compression_opts: Compression level (0-9 for gzip)
        show_progress: Whether to show progress bar
        chunk_size: Number of rows to write at once (default: 1M)
    """
    mode = 'a' if os.path.exists(filepath) and dataset != 'velocity' else 'w'

    desc = f"Writing {Path(filepath).name}"
    if dataset == 'velocity_coarse':
        desc += " (coarse)"

    n_rows = len(df)
    n_cols = len(df.columns)
    n_chunks = (n_rows + chunk_size - 1) // chunk_size

    if show_progress:
        with tqdm(total=100, desc=desc) as pbar:
            with h5py.File(filepath, mode, locking=False) as f:
                pbar.update(5)

                # Remove existing dataset if present
                if dataset in f:
                    del f[dataset]

                pbar.set_description(f"{desc} - Creating dataset")
                # Create empty dataset first (without loading all data)
                dset = f.create_dataset(
                    dataset,
                    shape=(n_rows, n_cols),
                    dtype=np.float64,
                    compression=compression if compression != 'null' else None,
                    compression_opts=compression_opts if compression == 'gzip' else None,
                    chunks=True
                )
                pbar.update(10)

                pbar.set_description(f"{desc} - Writing data")
                # Write data in chunks
                for i in range(n_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, n_rows)

                    # Get chunk data
                    chunk_data = df.iloc[start_idx:end_idx].values

                    # Write chunk to dataset
                    dset[start_idx:end_idx, :] = chunk_data

                    # Update progress (70% for writing data)
                    pbar.update(70 / n_chunks)

                pbar.set_description(f"{desc} - Writing metadata")
                # Store column names as attribute
                dset.attrs['columns'] = [c.encode('utf-8') for c in df.columns.tolist()]

                # Store metadata as attributes
                if metadata:
                    for key, value in metadata.items():
                        if isinstance(value, dict):
                            # Flatten nested dicts
                            for subkey, subvalue in value.items():
                                attr_key = f"{key}.{subkey}"
                                dset.attrs[attr_key] = subvalue
                        else:
                            dset.attrs[key] = value

                pbar.update(15)

        # Log file size
        file_size_mb = os.path.getsize(filepath) / (1024 ** 2)
        logger.info(
            f"Wrote {len(df):,} points to {filepath}:{dataset} "
            f"({file_size_mb:.1f} MB, {n_chunks} chunks)"
        )
    else:
        with h5py.File(filepath, mode, locking=False) as f:
            if dataset in f:
                del f[dataset]

            # Create empty dataset
            dset = f.create_dataset(
                dataset,
                shape=(n_rows, n_cols),
                dtype=np.float64,
                compression=compression if compression != 'null' else None,
                compression_opts=compression_opts if compression == 'gzip' else None,
                chunks=True
            )

            # Write data in chunks
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_rows)
                chunk_data = df.iloc[start_idx:end_idx].values
                dset[start_idx:end_idx, :] = chunk_data

            dset.attrs['columns'] = [c.encode('utf-8') for c in df.columns.tolist()]

            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            attr_key = f"{key}.{subkey}"
                            dset.attrs[attr_key] = subvalue
                    else:
                        dset.attrs[key] = value


def read_velocity_file(
    filepath: Union[str, Path],
    format: str = 'auto',
    dataset: str = 'velocity',
    show_progress: bool = True,
    columns: Optional[List[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Read velocity file in CSV or HDF5 format.

    Args:
        filepath: Path to velocity file (.csv or .h5/.hdf5)
        format: 'auto' (detect from extension), 'csv', or 'hdf5'
        dataset: HDF5 dataset name ('velocity' or 'velocity_coarse')
        show_progress: Whether to show progress bar
        columns: Optional list of column names to read (HDF5 only, for memory efficiency)
        **kwargs: Additional arguments for pd.read_csv (CSV only)

    Returns:
        DataFrame with velocity model data

    Columns:
        X_grid_km, Y_grid_km, Z_grid_km, x, y, z,
        i_raw, j_raw, k_raw, i, j, k,
        vp, vs, lon, lat, depth

    Example:
        >>> df = read_velocity_file('velocity_initial.csv')
        >>> df = read_velocity_file('velocity_final.h5', dataset='velocity_coarse')
        >>> # Memory efficient: read only necessary columns
        >>> df = read_velocity_file('velocity.h5', columns=['X_grid_km', 'Y_grid_km', 'Z_grid_km', 'vp'])
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Velocity file not found: {filepath}")

    # Detect format
    if format == 'auto':
        format = _detect_format(filepath)

    # Read based on format
    if format == 'csv':
        if columns is not None:
            kwargs['usecols'] = columns
        df = _read_csv_velocity(filepath, show_progress=show_progress, **kwargs)
    elif format == 'hdf5':
        df = _read_hdf5_velocity(filepath, dataset=dataset, show_progress=show_progress, columns=columns)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'csv' or 'hdf5'.")

    return df


def write_velocity_file(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    format: str = 'auto',
    coarse_grid_config: Optional[Dict[str, float]] = None,
    compression: str = 'gzip',
    compression_opts: int = 4,
    metadata: Optional[Dict[str, Any]] = None,
    show_progress: bool = True,
    n_jobs: int = 1,
    **kwargs
) -> None:
    """
    Write velocity file in CSV or HDF5 format.

    Args:
        df: Velocity DataFrame to write
        filepath: Output path (.csv or .h5/.hdf5)
        format: 'auto' (detect from extension), 'csv', or 'hdf5'
        coarse_grid_config: Optional config for coarse subsampling
            {
                'dx_coarse': 2.0,  # km, multiple of fine dx
                'dy_coarse': 2.0,  # km, multiple of fine dy
                'dz_coarse': 2.0,  # km, multiple of fine dz
            }
        compression: HDF5 compression ('gzip', 'lzf', None)
        compression_opts: Compression level (0-9 for gzip)
        metadata: Metadata dictionary (for HDF5 attributes)
        show_progress: Whether to show progress bar
        n_jobs: Number of parallel jobs for coarse subsampling
        **kwargs: Additional arguments for to_csv

    Example:
        >>> write_velocity_file(df, 'velocity.csv')
        >>> write_velocity_file(
        ...     df, 'velocity.h5',
        ...     coarse_grid_config={'dx_coarse': 2.0, 'dy_coarse': 2.0, 'dz_coarse': 2.0},
        ...     metadata={'nx': 100, 'ny': 100, 'nz': 50}
        ... )
    """
    filepath = Path(filepath)
    ensure_directory_exists(filepath.parent)

    # Detect format
    if format == 'auto':
        format = _detect_format(filepath)

    # Write based on format
    if format == 'csv':
        if show_progress:
            with tqdm(total=100, desc=f"Writing {filepath.name}") as pbar:
                pbar.set_description(f"Writing {filepath.name}")
                df.to_csv(filepath, index=False, **kwargs)
                pbar.update(100)

            file_size_mb = os.path.getsize(filepath) / (1024 ** 2)
            logger.info(f"Wrote {len(df):,} points to {filepath} ({file_size_mb:.1f} MB)")
        else:
            df.to_csv(filepath, index=False, **kwargs)

    elif format == 'hdf5':
        # Write main dataset
        _write_hdf5_velocity(
            filepath, df, 'velocity', metadata,
            compression, compression_opts, show_progress
        )

        # Create coarse dataset if requested
        if coarse_grid_config:
            if show_progress:
                logger.info("Creating coarse-sampled dataset...")

            # Extract fine grid spacing from metadata or compute from data
            if metadata and 'grid_spacing' in metadata:
                fine_dx = metadata['grid_spacing']['dx']
                fine_dy = metadata['grid_spacing']['dy']
                fine_dz = metadata['grid_spacing']['dz']
            else:
                # Compute from data
                x_unique = sorted(df['X_grid_km'].unique())
                y_unique = sorted(df['Y_grid_km'].unique())
                z_unique = sorted(df['Z_grid_km'].unique())
                fine_dx = x_unique[1] - x_unique[0] if len(x_unique) > 1 else 1.0
                fine_dy = y_unique[1] - y_unique[0] if len(y_unique) > 1 else 1.0
                fine_dz = z_unique[1] - z_unique[0] if len(z_unique) > 1 else 1.0

            # Subsample to coarse grid
            df_coarse = subsample_to_coarse_grid(
                df,
                dx_coarse=coarse_grid_config['dx_coarse'],
                dy_coarse=coarse_grid_config['dy_coarse'],
                dz_coarse=coarse_grid_config['dz_coarse'],
                fine_dx=fine_dx,
                fine_dy=fine_dy,
                fine_dz=fine_dz,
                n_jobs=n_jobs,
                show_progress=show_progress
            )

            # Compute coarse grid dimensions
            nx_coarse = int(df_coarse['i'].max())
            ny_coarse = int(df_coarse['j'].max())
            nz_coarse = int(df_coarse['k'].max())

            # Add stride information to metadata
            coarse_metadata = metadata.copy() if metadata else {}
            stride_x, stride_y, stride_z = _validate_coarse_grid_config(
                coarse_grid_config, fine_dx, fine_dy, fine_dz
            )
            coarse_metadata['coarse_dimensions'] = {
                'nx_coarse': nx_coarse,
                'ny_coarse': ny_coarse,
                'nz_coarse': nz_coarse
            }
            coarse_metadata['coarse_spacing'] = coarse_grid_config
            coarse_metadata['stride'] = {
                'stride_x': stride_x,
                'stride_y': stride_y,
                'stride_z': stride_z
            }

            # Write coarse dataset
            _write_hdf5_velocity(
                filepath, df_coarse, 'velocity_coarse', coarse_metadata,
                compression, compression_opts, show_progress
            )

    else:
        raise ValueError(f"Unknown format: {format}. Use 'csv' or 'hdf5'.")


def get_velocity_file_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract metadata from velocity file.

    Args:
        filepath: Path to velocity file

    Returns:
        Dictionary with metadata:
        {
            'format': 'csv' or 'hdf5',
            'num_points': int,
            'grid_dimensions': {'nx': int, 'ny': int, 'nz': int},
            'grid_spacing': {'dx': float, 'dy': float, 'dz': float},
            'grid_origin': {'x': float, 'y': float, 'z': float},
            'bounds': {'x': (min, max), 'y': (min, max), 'z': (min, max)},
            'has_coarse_dataset': bool,  # HDF5 only
            'datasets': List[str],  # HDF5 only
            'file_size_mb': float,
        }
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    file_size_mb = os.path.getsize(filepath) / (1024 ** 2)
    format_type = _detect_format(filepath)

    metadata = {
        'format': format_type,
        'file_size_mb': file_size_mb,
    }

    if format_type == 'csv':
        # Read first few rows to get metadata
        df_sample = pd.read_csv(filepath, nrows=1000)
        metadata['num_points'] = sum(1 for _ in open(filepath)) - 1  # Subtract header

    elif format_type == 'hdf5':
        with h5py.File(filepath, 'r', locking=False) as f:
            metadata['datasets'] = list(f.keys())
            metadata['has_coarse_dataset'] = 'velocity_coarse' in f

            if 'velocity' in f:
                dset = f['velocity']
                metadata['num_points'] = dset.shape[0]

                # Get column names
                if 'columns' in dset.attrs:
                    columns = dset.attrs['columns']
                    if isinstance(columns[0], bytes):
                        columns = [c.decode('utf-8') for c in columns]
                else:
                    columns = VELOCITY_COLUMNS

                df_sample = pd.DataFrame(dset[:1000], columns=columns)

                # Extract metadata from attributes
                for key in dset.attrs.keys():
                    if key != 'columns':  # Skip columns attribute
                        metadata[key] = dset.attrs[key]
            else:
                raise KeyError(f"Dataset 'velocity' not found in {filepath}")

    # Compute grid properties from sample data
    if 'df_sample' in locals():
        metadata['grid_dimensions'] = {
            'nx': int(df_sample['i'].max()),
            'ny': int(df_sample['j'].max()),
            'nz': int(df_sample['k'].max())
        }

        # Compute grid spacing
        x_unique = sorted(df_sample['X_grid_km'].unique())
        y_unique = sorted(df_sample['Y_grid_km'].unique())
        z_unique = sorted(df_sample['Z_grid_km'].unique())

        metadata['grid_spacing'] = {
            'dx': x_unique[1] - x_unique[0] if len(x_unique) > 1 else None,
            'dy': y_unique[1] - y_unique[0] if len(y_unique) > 1 else None,
            'dz': z_unique[1] - z_unique[0] if len(z_unique) > 1 else None
        }

        metadata['grid_origin'] = {
            'x': x_unique[0] if len(x_unique) > 0 else None,
            'y': y_unique[0] if len(y_unique) > 0 else None,
            'z': z_unique[0] if len(z_unique) > 0 else None
        }

        metadata['bounds'] = {
            'x': (df_sample['X_grid_km'].min(), df_sample['X_grid_km'].max()),
            'y': (df_sample['Y_grid_km'].min(), df_sample['Y_grid_km'].max()),
            'z': (df_sample['Z_grid_km'].min(), df_sample['Z_grid_km'].max())
        }

    return metadata
