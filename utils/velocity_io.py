"""
Velocity file I/O utilities for the LATTE tomography package.

This module provides a unified interface for reading and writing velocity files
in both CSV and HDF5 formats, with support for:
- Automatic format detection
- A single column convention and a single metadata schema for every file
- Coarse grid subsampling
- Parallel processing
- Progress tracking with tqdm
- HDF5 compression
- Legacy files (``X_grid_km`` style names, grid index columns, nested/missing metadata)

Columns
-------
Velocity files contain ``lat, lon, depth, x, y, z, vp, vs`` (``x, y, z`` are Cartesian km,
``depth`` is absolute depth in km). Gradient/search modules add the optional columns
``grad_vp, grad_vs, srch_vp, srch_vs``. DWS files contain ``dws`` instead of ``vp, vs``.

Metadata
--------
Every file carries the same flat metadata. In HDF5 it is stored as attributes of the
dataset; in CSV it is stored as ``# key: value`` comment lines at the top of the file.

- Grid: ``nx, ny, nz, dx, dy, dz, origin_x, origin_y, origin_z`` (km, Cartesian)
- Geography: ``origin_lon, origin_lat, origin_depth_km, transformation_method, invert_y_axis``
- ``data_type`` (``velocity`` or ``dws``)
- Coarse files additionally carry ``fine_nx ... fine_origin_z`` (the grid of the
  original fine model) and ``stride_x, stride_y, stride_z``
- ``units.<name>``: the unit of every column and grid key (km, km/s, deg, ...)
- Module specific provenance (e.g. ``interpolation_mode``) as extra keys

Grid spacing and origin are defined once, in preprocessing. Every other module reads them
from this metadata and cross-checks them against the x/y/z columns.

Author: LATTE Workflow Team
"""

import os
import logging
from pathlib import Path
from typing import Union, Dict, Any, Optional, List, Tuple, Sequence
import warnings

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from joblib import Parallel, delayed

from utils.io_utils import ensure_directory_exists, resolve_velocity_path
from utils.csv_metadata import flatten_metadata, read_csv_header_metadata, csv_header_lines, to_builtin
from utils.column_utils import (
    normalize_coordinate_columns,
    resolve_column_names,
    LEGACY_COORDINATE_COLUMNS,
    REMOVED_INDEX_COLUMNS,
)

logger = logging.getLogger(__name__)

# Standard column names for velocity DataFrames
GEO_COLUMNS = ['lat', 'lon', 'depth']
COORDINATE_COLUMNS = ['x', 'y', 'z']
VELOCITY_VALUE_COLUMNS = ['vp', 'vs']
DWS_VALUE_COLUMNS = ['dws']
VELOCITY_COLUMNS = GEO_COLUMNS + COORDINATE_COLUMNS + VELOCITY_VALUE_COLUMNS
OPTIONAL_VELOCITY_COLUMNS = ['grad_vp', 'grad_vs', 'srch_vp', 'srch_vs']
DWS_COLUMNS = GEO_COLUMNS + COORDINATE_COLUMNS + DWS_VALUE_COLUMNS

# Units of the columns and grid keys, written as ``units.<name>`` metadata
VELOCITY_UNITS = {
    'lat': 'deg', 'lon': 'deg', 'depth': 'km',
    'x': 'km', 'y': 'km', 'z': 'km',
    'vp': 'km/s', 'vs': 'km/s',
    'dws': 'km^2',
    'dx': 'km', 'dy': 'km', 'dz': 'km',
    'origin_x': 'km', 'origin_y': 'km', 'origin_z': 'km',
    'origin_depth_km': 'km', 'origin_lon': 'deg', 'origin_lat': 'deg',
}

# Standard metadata keys
GRID_KEYS = ('nx', 'ny', 'nz', 'dx', 'dy', 'dz', 'origin_x', 'origin_y', 'origin_z')
FINE_GRID_KEYS = tuple(f'fine_{k}' for k in GRID_KEYS)
GEO_METADATA_KEYS = ('origin_lon', 'origin_lat', 'origin_depth_km',
                     'transformation_method', 'invert_y_axis')
_INTEGER_KEYS = ('nx', 'ny', 'nz', 'fine_nx', 'fine_ny', 'fine_nz',
                 'stride_x', 'stride_y', 'stride_z')

# Legacy metadata key -> current key
_LEGACY_GRID_METADATA = {
    **{f'grid_dimensions.{a}': a for a in ('nx', 'ny', 'nz')},
    **{f'grid_spacing.{a}': a for a in ('dx', 'dy', 'dz')},
    'grid_origin.x': 'origin_x', 'grid_origin.y': 'origin_y', 'grid_origin.z': 'origin_z',
}
_LEGACY_COARSE_METADATA = {
    'coarse_dimensions.nx_coarse': 'nx', 'coarse_dimensions.ny_coarse': 'ny',
    'coarse_dimensions.nz_coarse': 'nz',
    'coarse_spacing.dx_coarse': 'dx', 'coarse_spacing.dy_coarse': 'dy',
    'coarse_spacing.dz_coarse': 'dz',
}
_LEGACY_STRIDE_METADATA = {f'stride.stride_{a}': f'stride_{a}' for a in 'xyz'}
# Legacy keys that no longer exist
_DROPPED_METADATA_KEYS = ('columns',)

_CSV_COMMENT = '#'


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

_to_builtin = to_builtin


def normalize_metadata(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert metadata of any vintage to the current flat schema.

    Accepts the legacy nested names (``grid_dimensions.nx``, ``grid_spacing.dx``,
    ``grid_origin.x``, ``stride.stride_x``) as well as the legacy coarse-dataset keys
    (where ``grid_*`` described the fine grid and ``coarse_dimensions``/``coarse_spacing``
    described the coarse one).
    """
    flat = flatten_metadata(raw)
    out: Dict[str, Any] = {}
    legacy_grid: Dict[str, Any] = {}
    legacy_coarse: Dict[str, Any] = {}

    for key, value in flat.items():
        if key in _DROPPED_METADATA_KEYS:
            continue
        if key in _LEGACY_GRID_METADATA:
            legacy_grid[_LEGACY_GRID_METADATA[key]] = value
        elif key in _LEGACY_COARSE_METADATA:
            legacy_coarse[_LEGACY_COARSE_METADATA[key]] = value
        elif key in _LEGACY_STRIDE_METADATA:
            out[_LEGACY_STRIDE_METADATA[key]] = value
        else:
            out[key] = value

    if legacy_coarse:
        # Legacy coarse dataset: grid_* described the fine grid, which is also its origin
        for key, value in legacy_grid.items():
            out[f'fine_{key}'] = value
        for key in ('origin_x', 'origin_y', 'origin_z'):
            if key in legacy_grid:
                out.setdefault(key, legacy_grid[key])
        out.update(legacy_coarse)
    else:
        for key, value in legacy_grid.items():
            out.setdefault(key, value)

    for key in _INTEGER_KEYS:
        if key in out and out[key] is not None:
            out[key] = int(out[key])
    return out


def infer_grid_metadata(df: pd.DataFrame, tol: float = 1e-6) -> Dict[str, Any]:
    """
    Infer ``nx..origin_z`` from the x/y/z columns of a regular-grid DataFrame.

    Spacing is the median coordinate step (omitted for an axis with a single point).
    A warning is logged when the grid is not regular.
    """
    df = normalize_coordinate_columns(df)
    grid: Dict[str, Any] = {}
    for axis in COORDINATE_COLUMNS:
        if axis not in df.columns:
            continue
        values = np.unique(np.round(df[axis].to_numpy(dtype=float), 6))
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        grid[f'n{axis}'] = int(len(values))
        grid[f'origin_{axis}'] = float(values[0])
        if len(values) > 1:
            steps = np.diff(values)
            grid[f'd{axis}'] = float(np.median(steps))
            # A shorter last step is expected in coarse grids that had the boundary point added
            body = steps[:-1] if len(steps) > 1 else steps
            if body.max() - body.min() > max(tol, 1e-3 * abs(grid[f'd{axis}'])):
                logger.warning(
                    f"Irregular grid spacing along {axis}: steps range "
                    f"{steps.min():.6g} - {steps.max():.6g} km (using median {grid[f'd{axis}']:.6g})"
                )
    return grid


def _values_match(a: Any, b: Any, tol: float = 1e-6) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=tol, atol=tol))


def resolve_grid_metadata(df: Optional[pd.DataFrame] = None,
                          metadata: Optional[Dict[str, Any]] = None,
                          source: str = 'velocity model') -> Dict[str, Any]:
    """
    Return the grid (``nx..origin_z``) of a velocity model.

    The metadata is authoritative. When ``df`` is given, the metadata is cross-checked
    against the x/y/z columns and a warning is logged for every disagreement. Keys the
    metadata does not provide (legacy files) are inferred from the columns, with a warning.

    Args:
        df: Velocity DataFrame (optional). If it was read by ``read_velocity_file`` and
            ``metadata`` is None, the metadata read from the file is used.
        metadata: File metadata (any vintage).
        source: Name used in log messages.

    Returns:
        Dict with the grid keys that could be determined.
    """
    if metadata is None and df is not None:
        metadata = df.attrs.get('metadata')
    meta = normalize_metadata(metadata)
    inferred = infer_grid_metadata(df) if df is not None else {}

    grid: Dict[str, Any] = {}
    inferred_keys = []
    for key in GRID_KEYS:
        meta_value, data_value = meta.get(key), inferred.get(key)
        if meta_value is not None:
            grid[key] = meta_value
            if data_value is not None and not _values_match(meta_value, data_value):
                logger.warning(
                    f"{source}: metadata {key}={meta_value} does not match the value "
                    f"{data_value} found in the x/y/z columns; using the metadata value"
                )
        elif data_value is not None:
            grid[key] = data_value
            inferred_keys.append(key)

    if inferred_keys:
        logger.warning(
            f"{source}: grid metadata missing for {inferred_keys} (legacy file?); "
            f"inferred from the x/y/z columns"
        )
    return grid


def get_fine_grid_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the grid of the original fine model from file metadata.

    Coarse files store it as ``fine_*``; for fine files it is the file's own grid.
    """
    meta = normalize_metadata(metadata)
    prefix = 'fine_' if any(k in meta for k in FINE_GRID_KEYS) else ''
    return {k: meta[prefix + k] for k in GRID_KEYS if meta.get(prefix + k) is not None}


def resolve_binary_grid(metadata: Optional[Dict[str, Any]] = None,
                        df: Optional[pd.DataFrame] = None,
                        config_grid: Optional[Dict[str, Any]] = None,
                        binary_files: Sequence[Union[str, Path]] = (),
                        bytes_per_value: int = 4,
                        source: str = 'velocity model') -> Dict[str, Any]:
    """
    Determine the grid of LATTE binary files (always the fine model grid).

    Preference order:
      1. The fine grid in the file metadata.
      2. (legacy file without grid metadata) the grid inferred from the x/y/z columns.
      3. (legacy file, last resort) ``config_grid`` from the configuration.
    For legacy files the candidates are checked against the size of the first existing
    binary file, since a coarse file's columns describe the coarse grid, not the binaries.

    Raises:
        ValueError: If no complete grid can be found.
    """
    meta = normalize_metadata(metadata if metadata is not None
                              else (df.attrs.get('metadata') if df is not None else None))
    fine = get_fine_grid_metadata(meta)
    if all(k in fine for k in GRID_KEYS):
        if df is not None and not any(k in meta for k in FINE_GRID_KEYS):
            resolve_grid_metadata(df, meta, source)  # cross-check with the columns
        return fine

    candidates = []
    if df is not None:
        inferred = infer_grid_metadata(df)
        if all(k in inferred for k in GRID_KEYS):
            candidates.append(('x/y/z columns', inferred))
    if config_grid:
        cfg = _config_grid_to_metadata(config_grid)
        if all(k in cfg for k in GRID_KEYS):
            candidates.append(('configuration', cfg))
    if not candidates:
        raise ValueError(
            f"{source}: no grid metadata and the grid cannot be inferred from the columns; "
            f"define the grid ({', '.join(GRID_KEYS)}) in the configuration"
        )

    existing = [Path(f) for f in binary_files if f and Path(f).exists()]
    chosen_name, chosen = candidates[0]
    if existing:
        size = existing[0].stat().st_size
        for name, grid in candidates:
            if grid['nx'] * grid['ny'] * grid['nz'] * bytes_per_value == size:
                chosen_name, chosen = name, grid
                break
        else:
            logger.warning(f"{source}: no candidate grid matches the size of {existing[0].name}")
    logger.warning(f"{source}: no fine-grid metadata (legacy file?); using the grid from {chosen_name}")
    return chosen


def _config_grid_to_metadata(config_grid: Any) -> Dict[str, Any]:
    """Convert a configuration ``grid`` block to the metadata key convention."""
    cfg = dict(config_grid)
    out = {}
    for key in GRID_KEYS:
        if cfg.get(key) is not None:
            out[key] = cfg[key]
    for axis in COORDINATE_COLUMNS:  # accept x_min/y_min/z_min as origin
        if f'origin_{axis}' not in out and cfg.get(f'{axis}_min') is not None:
            out[f'origin_{axis}'] = cfg[f'{axis}_min']
    return normalize_metadata(out)


def build_coarse_metadata(df_coarse: pd.DataFrame,
                          fine_metadata: Dict[str, Any],
                          coarse_config: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Build the metadata of a coarse file derived from a fine model.

    The coarse file's own grid is taken from ``df_coarse``; the fine grid is stored as
    ``fine_*`` and the strides as ``stride_x/y/z`` (when ``coarse_config`` is given).
    """
    fine_meta = normalize_metadata(fine_metadata)
    fine_grid = get_fine_grid_metadata(fine_meta)
    meta = {k: v for k, v in fine_meta.items()
            if k not in GRID_KEYS and k not in FINE_GRID_KEYS and not k.startswith('stride_')}
    meta.update(infer_grid_metadata(df_coarse))
    for key, value in fine_grid.items():
        meta[f'fine_{key}'] = value
    if coarse_config:
        strides = _validate_coarse_grid_config(
            coarse_config, *_fine_spacing(fine_grid, coarse_config))
        meta.update(stride_x=strides[0], stride_y=strides[1], stride_z=strides[2])
    return meta


def _fine_spacing(fine_grid: Dict[str, Any], coarse_config: Dict[str, float]) -> Tuple[float, float, float]:
    """Fine spacing; a single-point axis (no spacing) uses the coarse spacing, i.e. stride 1."""
    return tuple(
        fine_grid[f'd{a}'] if fine_grid.get(f'd{a}') is not None else coarse_config[f'd{a}_coarse']
        for a in COORDINATE_COLUMNS
    )


_read_csv_header_metadata = read_csv_header_metadata
_csv_header_lines = csv_header_lines


def read_velocity_metadata(filepath: Union[str, Path], dataset: str = 'velocity') -> Dict[str, Any]:
    """
    Read the metadata stored in a velocity file (no data is loaded).

    Works for current and legacy files; the result always uses the current flat schema.
    A legacy file without metadata gives an empty dict.
    """
    filepath = Path(resolve_velocity_path(filepath))
    format_type = _detect_format(filepath)
    if format_type == 'csv':
        raw = _read_csv_header_metadata(filepath)
    else:
        with h5py.File(filepath, 'r', locking=False) as f:
            if dataset not in f:
                raise KeyError(f"Dataset '{dataset}' not found in {filepath}. "
                               f"Available datasets: {list(f.keys())}")
            raw = {k: _to_builtin(v) for k, v in f[dataset].attrs.items()}
    return normalize_metadata(raw)


def get_velocity_grid(filepath: Union[str, Path], dataset: str = 'velocity',
                      verify: bool = True) -> Dict[str, Any]:
    """
    Read a file's metadata with the grid resolved and (optionally) verified.

    With ``verify`` the x/y/z columns are loaded and compared to the metadata; legacy
    files without grid metadata get it from the columns. The returned dict is the
    file metadata with the grid keys filled in.
    """
    filepath = Path(resolve_velocity_path(filepath))
    metadata = read_velocity_metadata(filepath, dataset=dataset)
    complete = all(metadata.get(k) is not None for k in GRID_KEYS if k not in ('dx', 'dy', 'dz'))
    df_xyz = None
    if verify or not complete:
        df_xyz = read_velocity_file(filepath, dataset=dataset, show_progress=False,
                                    columns=COORDINATE_COLUMNS)
    grid = resolve_grid_metadata(df_xyz, metadata, source=filepath.name)
    return {**metadata, **grid}


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def standardize_velocity_columns(df: pd.DataFrame,
                                 value_columns: Sequence[str] = tuple(VELOCITY_VALUE_COLUMNS)
                                 ) -> pd.DataFrame:
    """
    Return ``df`` with the standard columns in the standard order.

    Legacy coordinate names are renamed and grid index columns dropped. Missing
    ``lat/lon/depth`` are added as NaN. Optional gradient/search columns follow the value
    columns when present. Any other column is kept at the end with a warning.

    Raises:
        ValueError: If x/y/z or a value column is missing.
    """
    df = normalize_coordinate_columns(df)
    required = COORDINATE_COLUMNS + list(value_columns)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Velocity data is missing required columns: {missing}")

    df = df.copy()
    for column in GEO_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    optional = [c for c in OPTIONAL_VELOCITY_COLUMNS if c in df.columns]
    ordered = GEO_COLUMNS + COORDINATE_COLUMNS + list(value_columns) + optional
    extras = [c for c in df.columns if c not in ordered]
    if extras:
        logger.warning(f"Velocity data has non-standard columns {extras}; keeping them")
    return df[ordered + extras]


# ---------------------------------------------------------------------------
# Format detection and coarse subsampling
# ---------------------------------------------------------------------------

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
        df_chunk['x'].isin(x_coarse) &
        df_chunk['y'].isin(y_coarse) &
        df_chunk['z'].isin(z_coarse)
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
    x_unique = sorted(df['x'].unique())
    y_unique = sorted(df['y'].unique())
    z_unique = sorted(df['z'].unique())

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
    columns: Optional[List[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Read velocity from CSV file with progress tracking.

    Metadata comment lines (``# key: value``) at the top of the file are skipped.

    Args:
        filepath: Path to CSV file
        show_progress: Whether to show progress bar
        columns: Optional list of (current) column names to read; legacy names are resolved
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        Velocity DataFrame
    """
    kwargs.setdefault('comment', _CSV_COMMENT)
    if columns is not None:
        available = pd.read_csv(filepath, nrows=0, comment=_CSV_COMMENT).columns
        kwargs['usecols'] = resolve_column_names(columns, available)

    if show_progress:
        # Count total lines for progress bar
        with open(filepath, 'r') as f:
            total_lines = sum(1 for line in f if not line.startswith(_CSV_COMMENT)) - 1  # Subtract header

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
        columns: Optional list of (current) column names to read (default: read all);
            legacy names are resolved

    Returns:
        Velocity DataFrame
    """
    desc = f"Reading {Path(filepath).name}"
    if dataset == 'velocity_coarse':
        desc += " (coarse)"

    def _load(pbar=None) -> pd.DataFrame:
        with h5py.File(filepath, 'r', locking=False) as f:
            if pbar is not None:
                pbar.update(10)

            if dataset not in f:
                raise KeyError(
                    f"Dataset '{dataset}' not found in {filepath}. "
                    f"Available datasets: {list(f.keys())}"
                )

            if pbar is not None:
                pbar.set_description(f"{desc} - Loading data")

            # Get column names from attributes
            if 'columns' in f[dataset].attrs:
                all_columns = list(f[dataset].attrs['columns'])
                if all_columns and isinstance(all_columns[0], bytes):
                    all_columns = [c.decode('utf-8') for c in all_columns]
            else:
                # Use default column names
                all_columns = list(VELOCITY_COLUMNS)

            # Load data (all columns or selected columns)
            if columns is None:
                data = f[dataset][:]
                selected_columns = all_columns
            else:
                # Read only selected columns (h5py needs increasing indices)
                wanted = resolve_column_names(columns, all_columns)
                col_indices = sorted({all_columns.index(c) for c in wanted})
                data = f[dataset][:, col_indices]
                selected_columns = [all_columns[i] for i in col_indices]

            if pbar is not None:
                pbar.update(70)
                pbar.set_description(f"{desc} - Creating DataFrame")
            df = pd.DataFrame(data, columns=selected_columns)
            if pbar is not None:
                pbar.update(20)
            return df

    if show_progress:
        with tqdm(total=100, desc=desc) as pbar:
            df = _load(pbar)
        logger.info(f"Loaded {len(df):,} points from {filepath}:{dataset}")
    else:
        df = _load()
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
        df: Velocity DataFrame (already standardized)
        dataset: Dataset name
        metadata: Flat metadata dictionary to store as attributes
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
    metadata = flatten_metadata(metadata)

    def _write(pbar=None) -> None:
        def _step(description=None):
            if pbar is not None:
                if description:
                    pbar.set_description(f"{desc} - {description}")
                pbar.update(1)

        with h5py.File(filepath, mode, locking=False) as f:
            _step()  # file opened

            # Remove existing dataset if present
            if dataset in f:
                del f[dataset]

            # Create empty dataset first (without loading all data)
            dset = f.create_dataset(
                dataset,
                shape=(n_rows, n_cols),
                dtype=np.float64,
                compression=compression if compression != 'null' else None,
                compression_opts=compression_opts if compression == 'gzip' else None,
                chunks=True
            )
            _step("Writing data")

            # Write data in chunks
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_rows)
                dset[start_idx:end_idx, :] = df.iloc[start_idx:end_idx].values
                _step()

            if pbar is not None:
                pbar.set_description(f"{desc} - Writing metadata")
            # Store column names and metadata as attributes
            dset.attrs['columns'] = [c.encode('utf-8') for c in df.columns.tolist()]
            for key, value in metadata.items():
                dset.attrs[key] = value
            _step()  # metadata written

    if show_progress:
        with tqdm(total=1 + 1 + n_chunks + 1, desc=desc) as pbar:
            _write(pbar)
    else:
        _write()


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

    Legacy files are accepted: ``X_grid_km``/``Y_grid_km``/``Z_grid_km`` are renamed to
    ``x``/``y``/``z`` and grid index columns (``i``, ``j``, ``k``, ``i_raw``, ...) are
    dropped, each with a warning. The file metadata (current or legacy schema, normalized
    to the current one) is attached as ``df.attrs['metadata']``; use
    ``resolve_grid_metadata(df)`` to get the verified grid.

    Args:
        filepath: Path to velocity file (.csv or .h5/.hdf5)
        format: 'auto' (detect from extension), 'csv', or 'hdf5'
        dataset: HDF5 dataset name ('velocity' or 'velocity_coarse')
        show_progress: Whether to show progress bar
        columns: Optional list of column names to read (for memory efficiency)
        **kwargs: Additional arguments for pd.read_csv (CSV only)

    Returns:
        DataFrame with velocity model data

    Columns:
        lat, lon, depth, x, y, z, vp, vs
        (plus grad_vp, grad_vs, srch_vp, srch_vs in gradient/search files, or dws in DWS files)

    Example:
        >>> df = read_velocity_file('velocity_model_init.csv')
        >>> df = read_velocity_file('velocity_model_final.h5', dataset='velocity_coarse')
        >>> # Memory efficient: read only necessary columns
        >>> df = read_velocity_file('velocity.h5', columns=['x', 'y', 'z', 'vp'])
        >>> grid = resolve_grid_metadata(df)
    """
    filepath = Path(resolve_velocity_path(filepath))
    if not filepath.exists():
        raise FileNotFoundError(f"Velocity file not found: {filepath}")

    # Detect format
    if format == 'auto':
        format = _detect_format(filepath)

    # Read based on format
    if format == 'csv':
        metadata = _read_csv_header_metadata(filepath)
        df = _read_csv_velocity(filepath, show_progress=show_progress, columns=columns, **kwargs)
    elif format == 'hdf5':
        metadata = read_velocity_metadata(filepath, dataset=dataset)
        df = _read_hdf5_velocity(filepath, dataset=dataset, show_progress=show_progress, columns=columns)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'csv' or 'hdf5'.")

    df = normalize_coordinate_columns(df, context=filepath.name)
    df.attrs['metadata'] = normalize_metadata(metadata)
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

    The columns are standardized (``lat, lon, depth, x, y, z, vp, vs`` plus optional
    gradient/search columns; missing ``lat/lon/depth`` are written as NaN) and the file
    metadata is written in both formats: HDF5 attributes, or ``# key: value`` lines at the
    top of a CSV. The grid keys (``nx..origin_z``) are taken from ``metadata`` when
    given, cross-checked against the x/y/z columns, and inferred from the columns
    otherwise.

    Args:
        df: Velocity DataFrame to write
        filepath: Output path (.csv or .h5/.hdf5)
        format: 'auto' (detect from extension), 'csv', or 'hdf5'
        coarse_grid_config: Optional config for coarse subsampling (HDF5 only)
            {
                'dx_coarse': 2.0,  # km, multiple of fine dx
                'dy_coarse': 2.0,  # km, multiple of fine dy
                'dz_coarse': 2.0,  # km, multiple of fine dz
            }
        compression: HDF5 compression ('gzip', 'lzf', None)
        compression_opts: Compression level (0-9 for gzip)
        metadata: Metadata dictionary (see module docstring). ``data_type='dws'`` writes a
            DWS file (``dws`` column instead of ``vp, vs``).
        show_progress: Whether to show progress bar
        n_jobs: Number of parallel jobs for coarse subsampling
        **kwargs: Additional arguments for to_csv

    Example:
        >>> write_velocity_file(df, 'velocity.csv', metadata={'origin_lon': -108.0})
        >>> write_velocity_file(
        ...     df, 'velocity.h5',
        ...     coarse_grid_config={'dx_coarse': 2.0, 'dy_coarse': 2.0, 'dz_coarse': 2.0},
        ... )
    """
    filepath = Path(filepath)
    ensure_directory_exists(filepath.parent)

    # Detect format
    if format == 'auto':
        format = _detect_format(filepath)

    metadata = normalize_metadata(metadata)
    data_type = metadata.setdefault('data_type', 'velocity')
    value_columns = DWS_VALUE_COLUMNS if data_type == 'dws' else VELOCITY_VALUE_COLUMNS
    df = standardize_velocity_columns(df, value_columns)

    # Grid metadata: given values are authoritative, cross-checked against the columns
    metadata.update(resolve_grid_metadata(df, metadata, source=filepath.name))

    # Units of every known column / grid key
    for name in list(df.columns) + [k for k in metadata if k in VELOCITY_UNITS]:
        if name in VELOCITY_UNITS:
            metadata.setdefault(f'units.{name}', VELOCITY_UNITS[name])

    # Write based on format
    if format == 'csv':
        with open(filepath, 'w', newline='') as f:
            f.writelines(_csv_header_lines(flatten_metadata(metadata)))
            if show_progress:
                with tqdm(total=100, desc=f"Writing {filepath.name}") as pbar:
                    df.to_csv(f, index=False, **kwargs)
                    pbar.update(100)
            else:
                df.to_csv(f, index=False, **kwargs)
        if show_progress:
            file_size_mb = os.path.getsize(filepath) / (1024 ** 2)
            logger.info(f"Wrote {len(df):,} points to {filepath} ({file_size_mb:.1f} MB)")

    elif format == 'hdf5':
        # Write main dataset
        _write_hdf5_velocity(
            filepath, df, 'velocity', metadata,
            compression, compression_opts, show_progress
        )

        # Create coarse dataset if requested
        if coarse_grid_config:
            # Accept the configuration spelling (dx/dy/dz) as well as dx_coarse/dy_coarse/dz_coarse
            coarse_grid_config = {
                f'd{a}_coarse': coarse_grid_config[f'd{a}_coarse'] if f'd{a}_coarse' in coarse_grid_config
                else coarse_grid_config[f'd{a}']
                for a in COORDINATE_COLUMNS
            }
            if show_progress:
                logger.info("Creating coarse-sampled dataset...")

            fine_grid = get_fine_grid_metadata(metadata)
            fine_dx, fine_dy, fine_dz = _fine_spacing(fine_grid, coarse_grid_config)

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

            # Write coarse dataset
            _write_hdf5_velocity(
                filepath, df_coarse, 'velocity_coarse',
                build_coarse_metadata(df_coarse, metadata, coarse_grid_config),
                compression, compression_opts, show_progress
            )
    else:
        raise ValueError(f"Unknown format: {format}. Use 'csv' or 'hdf5'.")


def get_velocity_file_metadata(filepath: Union[str, Path], dataset: str = 'velocity') -> Dict[str, Any]:
    """
    Extract metadata from a velocity file.

    Args:
        filepath: Path to velocity file
        dataset: HDF5 dataset name

    Returns:
        Dictionary with the file's flat metadata (see module docstring; the grid keys are
        verified against the x/y/z columns and inferred from them if the file has none),
        plus:
        {
            'format': 'csv' or 'hdf5',
            'num_points': int,
            'bounds': {'x': (min, max), 'y': (min, max), 'z': (min, max)},
            'has_coarse_dataset': bool,  # HDF5 only
            'datasets': List[str],  # HDF5 only
            'file_size_mb': float,
        }
    """
    filepath = Path(resolve_velocity_path(filepath))
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    format_type = _detect_format(filepath)
    info: Dict[str, Any] = {
        'format': format_type,
        'file_size_mb': os.path.getsize(filepath) / (1024 ** 2),
    }

    if format_type == 'hdf5':
        with h5py.File(filepath, 'r', locking=False) as f:
            info['datasets'] = list(f.keys())
            info['has_coarse_dataset'] = 'velocity_coarse' in f
            if dataset not in f:
                raise KeyError(f"Dataset '{dataset}' not found in {filepath}")

    metadata = read_velocity_metadata(filepath, dataset=dataset)
    df_xyz = read_velocity_file(filepath, dataset=dataset, show_progress=False,
                                columns=COORDINATE_COLUMNS)
    info['num_points'] = len(df_xyz)
    info['bounds'] = {axis: (float(df_xyz[axis].min()), float(df_xyz[axis].max())) for axis in COORDINATE_COLUMNS}

    grid = resolve_grid_metadata(df_xyz, metadata, source=filepath.name)
    return {**metadata, **grid, **info}
