"""
Velocity file I/O utilities for the LATTE tomography package (spherical grid).

This module provides a unified interface for reading and writing velocity files in CSV and
HDF5 formats, with:
- Automatic format detection
- A single column convention and a single metadata schema for every file
- Coarse grid subsampling
- Parallel processing and progress tracking
- HDF5 compression

Columns
-------
Velocity files contain ``lon, lat, depth, vp, vs``: longitude and WGS84 geodetic latitude in
degrees, depth below mean sea level in km, velocities in km/s. Gradient/search modules add the
optional columns ``grad_vp, grad_vs, srch_vp, srch_vs``.

Metadata
--------
Every file carries the same flat metadata. In HDF5 it is stored as attributes of the dataset;
in CSV it is stored as ``# key: value`` comment lines at the top of the file.

- Grid: ``nlon, nlat, ndepth, dlon, dlat, ddepth, olon, olat, odepth`` (degrees for the
  longitude and latitude keys, km for the depth keys); see ``utils.spherical``
- ``coordinate_system`` (``spherical``) and ``earth_radius_km``
- ``data_type`` (``velocity``)
- Coarse files additionally carry ``fine_nlon ... fine_odepth`` (the grid of the original fine
  model) and ``stride_lon, stride_lat, stride_depth``
- ``units.<name>``: the unit of every column and grid key
- Module specific provenance as extra keys

Grid spacing and origin are defined once, in preprocessing. Every other module reads them from
this metadata and cross-checks them against the lon/lat/depth columns.
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
from utils.spherical import GRID_KEYS, GRID_AXES, EARTH_RADIUS_KM

logger = logging.getLogger(__name__)

# Standard column names for velocity DataFrames
GEO_COLUMNS = list(GRID_AXES)              # lon, lat, depth: the grid axes
COORDINATE_COLUMNS = GEO_COLUMNS
VELOCITY_VALUE_COLUMNS = ['vp', 'vs']
VELOCITY_COLUMNS = GEO_COLUMNS + VELOCITY_VALUE_COLUMNS
OPTIONAL_VELOCITY_COLUMNS = ['grad_vp', 'grad_vs', 'srch_vp', 'srch_vs']

# Units of the columns and grid keys, written as ``units.<name>`` metadata
VELOCITY_UNITS = {
    'lon': 'deg', 'lat': 'deg', 'depth': 'km',
    'vp': 'km/s', 'vs': 'km/s',
    'dlon': 'deg', 'dlat': 'deg', 'ddepth': 'km',
    'olon': 'deg', 'olat': 'deg', 'odepth': 'km',
    'earth_radius_km': 'km',
}

# Standard metadata keys
FINE_GRID_KEYS = tuple(f'fine_{k}' for k in GRID_KEYS)
_INTEGER_KEYS = ('nlon', 'nlat', 'ndepth', 'fine_nlon', 'fine_nlat', 'fine_ndepth',
                 'stride_lon', 'stride_lat', 'stride_depth')
# Spacing key and stride key of each axis
_SPACING = {'lon': 'dlon', 'lat': 'dlat', 'depth': 'ddepth'}

_CSV_COMMENT = '#'


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

_to_builtin = to_builtin


def normalize_metadata(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Return the metadata as a flat dictionary with plain Python values, the integer grid keys
    as ``int``, and ``coordinate_system`` and ``earth_radius_km`` set (``spherical`` and the
    default radius when absent).

    Raises:
        ValueError: If the metadata names another coordinate system.
    """
    out = {k: _to_builtin(v) for k, v in flatten_metadata(raw).items()}
    for key in _INTEGER_KEYS:
        if out.get(key) is not None:
            out[key] = int(round(float(out[key])))
    system = out.setdefault('coordinate_system', 'spherical')
    if system != 'spherical':
        raise ValueError(f"Coordinate system '{system}' is not supported; this workflow is spherical only")
    out.setdefault('earth_radius_km', EARTH_RADIUS_KM)
    return out


def infer_grid_metadata(df: pd.DataFrame, tol: float = 1e-6) -> Dict[str, Any]:
    """
    Infer ``nlon..odepth`` from the lon/lat/depth columns of a regular-grid DataFrame.

    Spacing is the median coordinate step (omitted for an axis with a single point).
    A warning is logged when the grid is not regular.
    """
    grid: Dict[str, Any] = {}
    for axis in GRID_AXES:
        if axis not in df.columns:
            continue
        values = np.unique(np.round(df[axis].to_numpy(dtype=float), 8))
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        grid[f'n{axis}'] = int(len(values))
        grid[f'o{axis}'] = float(values[0])
        if len(values) > 1:
            steps = np.diff(values)
            spacing = float(np.median(steps))
            grid[_SPACING[axis]] = spacing
            # A shorter last step is expected in coarse grids that had the boundary point added
            body = steps[:-1] if len(steps) > 1 else steps
            if body.max() - body.min() > max(tol, 1e-3 * abs(spacing)):
                logger.warning(
                    f"Irregular grid spacing along {axis}: steps range "
                    f"{steps.min():.6g} - {steps.max():.6g} (using median {spacing:.6g})"
                )
    return grid


def _values_match(a: Any, b: Any, tol: float = 1e-6) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=tol, atol=tol))


def resolve_grid_metadata(df: Optional[pd.DataFrame] = None,
                          metadata: Optional[Dict[str, Any]] = None,
                          source: str = 'velocity model') -> Dict[str, Any]:
    """
    Return the grid (``nlon..odepth``) of a velocity model.

    The metadata is authoritative. When ``df`` is given, the metadata is cross-checked against
    the lon/lat/depth columns and a warning is logged for every disagreement. Keys the metadata
    does not provide are inferred from the columns, with a warning.

    Args:
        df: Velocity DataFrame (optional). If it was read by ``read_velocity_file`` and
            ``metadata`` is None, the metadata read from the file is used.
        metadata: File metadata.
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
                    f"{data_value} found in the lon/lat/depth columns; using the metadata value"
                )
        elif data_value is not None:
            grid[key] = data_value
            inferred_keys.append(key)
    if inferred_keys:
        logger.warning(
            f"{source}: grid metadata missing for {inferred_keys}; "
            f"inferred from the lon/lat/depth columns"
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
    Determine the grid of LATTE binary files (the model grid the binaries were written on).

    The grid in the file metadata is the reference (or, without grid metadata, the grid inferred
    from the lon/lat/depth columns). ``config_grid`` is an additional check. The candidates are
    tested against the number of values in the existing binary files:
      - the reference matches: it is used (a warning is logged if the configuration differs);
      - only the configuration matches: it is used, with a warning;
      - neither matches: ``ValueError``.
    Without existing binary files the reference is used (or the configuration if there is no
    reference), and a warning is logged if the two differ.

    Raises:
        ValueError: If no complete grid can be found, or no candidate matches the binaries.
    """
    meta = normalize_metadata(metadata if metadata is not None
                              else (df.attrs.get('metadata') if df is not None else None))
    fine = get_fine_grid_metadata(meta)

    candidates = []  # (name, grid), highest priority first
    if all(k in fine for k in GRID_KEYS):
        if df is not None and not any(k in meta for k in FINE_GRID_KEYS):
            resolve_grid_metadata(df, meta, source)  # cross-check with the columns
        candidates.append(('file metadata', fine))
    elif df is not None:
        inferred = infer_grid_metadata(df)
        if all(k in inferred for k in GRID_KEYS):
            logger.warning(f"{source}: no grid metadata; grid inferred from the lon/lat/depth columns")
            candidates.append(('lon/lat/depth columns', inferred))
    if config_grid:
        cfg = _config_grid_to_metadata(config_grid)
        if all(k in cfg for k in GRID_KEYS):
            candidates.append(('configuration', cfg))
        else:
            logger.warning(f"{source}: configured grid is incomplete (needs {', '.join(GRID_KEYS)}); ignored")
    if not candidates:
        raise ValueError(
            f"{source}: no grid metadata and the grid cannot be inferred from the columns; "
            f"define the grid ({', '.join(GRID_KEYS)}) in the configuration"
        )

    def n_values(grid):
        return int(grid['nlon']) * int(grid['nlat']) * int(grid['ndepth'])

    existing = [Path(f) for f in binary_files if f and Path(f).exists()]
    if not existing:
        name, grid = candidates[0]
        if len(candidates) > 1 and not _same_grid_values(grid, candidates[1][1]):
            logger.warning(f"{source}: grid from {name} differs from the {candidates[1][0]} grid and "
                           f"no binary file exists to check against; using {name}")
        return grid

    sizes = {f.name: f.stat().st_size for f in existing}
    matching = [(name, grid) for name, grid in candidates
                if all(size == n_values(grid) * bytes_per_value for size in sizes.values())]
    if not matching:
        found = ', '.join(f"{n} has {sz // bytes_per_value:,}" for n, sz in sizes.items())
        tried = ', '.join(f"{name} {grid['nlon']}x{grid['nlat']}x{grid['ndepth']}={n_values(grid):,}"
                          for name, grid in candidates)
        raise ValueError(f"{source}: the binary files match none of the candidate grids "
                         f"(expected {tried}; {found} values)")
    name, grid = matching[0]
    if name != candidates[0][0]:
        logger.warning(f"{source}: the {candidates[0][0]} grid does not match the number of values in the "
                       f"binary files; using the {name} grid instead")
    elif len(candidates) > 1 and not _same_grid_values(grid, candidates[1][1]):
        other = candidates[1][0]
        logger.warning(f"{source}: the {other} grid differs from the {name} grid"
                       + ("" if len(matching) > 1 else " and does not match the binary files")
                       + f"; using the {name} grid")
    return grid


def _same_grid_values(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return all(_values_match(a[k], b[k]) for k in GRID_KEYS)


def resolve_reference_grid(reference_file: Optional[Union[str, Path]] = None,
                           config_grid: Optional[Dict[str, Any]] = None,
                           source: str = 'grid') -> Dict[str, Any]:
    """
    Grid (``nlon..odepth``) of the LATTE model from a reference velocity file.

    The reference file is the truth: its (fine-grid) metadata, or the grid inferred from its
    lon/lat/depth columns. ``config_grid`` is an additional check; every value that differs
    from the reference logs a warning. If ``reference_file`` is None, or no grid can be read
    from it, ``config_grid`` is used with a warning.

    Raises:
        ValueError: If no complete grid is available.
    """
    cfg = None
    if config_grid:
        cfg = _config_grid_to_metadata(config_grid)
        if not all(k in cfg for k in GRID_KEYS):
            logger.warning(f"{source}: configured grid is incomplete (needs {', '.join(GRID_KEYS)}); ignored")
            cfg = None
    if reference_file is None:
        if cfg is None:
            raise ValueError(f"{source}: no reference velocity file and no complete grid in the configuration")
        logger.warning(f"{source}: no reference velocity file; using the grid from the configuration")
        return cfg

    name = Path(reference_file).name
    meta = read_velocity_metadata(reference_file)
    grid = get_fine_grid_metadata(meta)
    if not all(k in grid for k in GRID_KEYS):
        df_axes = read_velocity_file(reference_file, show_progress=False, columns=GEO_COLUMNS)
        try:
            grid = resolve_binary_grid(metadata=meta, df=df_axes, source=name)
        except ValueError as e:
            if cfg is None:
                raise
            logger.warning(f"{source}: {e}; using the grid from the configuration")
            return cfg
    if cfg is not None:
        diffs = [f"{k}: reference {grid[k]} vs configured {cfg[k]}"
                 for k in GRID_KEYS if not _values_match(grid[k], cfg[k])]
        if diffs:
            logger.warning(f"{source}: configured grid differs from {name} ({'; '.join(diffs)}); "
                           f"using the grid from {name}")
    return grid


def _config_grid_to_metadata(config_grid: Any) -> Dict[str, Any]:
    """Convert a configuration ``grid`` block (keys ``nlon..odepth``) to metadata."""
    cfg = dict(config_grid)
    return normalize_metadata({key: cfg[key] for key in GRID_KEYS if cfg.get(key) is not None})


def build_coarse_metadata(df_coarse: pd.DataFrame,
                          fine_metadata: Dict[str, Any],
                          coarse_config: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Build the metadata of a coarse file derived from a fine model.

    The coarse file's own grid is taken from ``df_coarse``; the fine grid is stored as
    ``fine_*`` and the strides as ``stride_lon/lat/depth`` (when ``coarse_config`` is given).
    """
    fine_meta = normalize_metadata(fine_metadata)
    fine_grid = get_fine_grid_metadata(fine_meta)
    meta = {k: v for k, v in fine_meta.items()
            if k not in GRID_KEYS and k not in FINE_GRID_KEYS and not k.startswith('stride_')}
    meta.update(infer_grid_metadata(df_coarse))
    for key, value in fine_grid.items():
        meta[f'fine_{key}'] = value
    if coarse_config:
        strides = _validate_coarse_grid_config(coarse_config, *_fine_spacing(fine_grid, coarse_config))
        meta.update(stride_lon=strides[0], stride_lat=strides[1], stride_depth=strides[2])
    return meta


def _fine_spacing(fine_grid: Dict[str, Any], coarse_config: Dict[str, float]) -> Tuple[float, float, float]:
    """Fine spacing; a single-point axis (no spacing) uses the coarse spacing, i.e. stride 1."""
    return tuple(
        fine_grid[_SPACING[a]] if fine_grid.get(_SPACING[a]) is not None else coarse_config[f'{_SPACING[a]}_coarse']
        for a in GRID_AXES
    )


_read_csv_header_metadata = read_csv_header_metadata
_csv_header_lines = csv_header_lines


def read_velocity_metadata(filepath: Union[str, Path], dataset: str = 'velocity') -> Dict[str, Any]:
    """Read the metadata stored in a velocity file (no data is loaded)."""
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

    With ``verify`` the lon/lat/depth columns are loaded and compared to the metadata; a file
    without grid metadata gets it from the columns. The returned dict is the file metadata with
    the grid keys filled in.
    """
    filepath = Path(resolve_velocity_path(filepath))
    metadata = read_velocity_metadata(filepath, dataset=dataset)
    complete = all(metadata.get(k) is not None for k in GRID_KEYS)
    df_axes = None
    if verify or not complete:
        df_axes = read_velocity_file(filepath, dataset=dataset, show_progress=False, columns=GEO_COLUMNS)
    grid = resolve_grid_metadata(df_axes, metadata, source=filepath.name)
    return {**metadata, **grid}


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def standardize_velocity_columns(df: pd.DataFrame,
                                 value_columns: Sequence[str] = tuple(VELOCITY_VALUE_COLUMNS)
                                 ) -> pd.DataFrame:
    """
    Return ``df`` with the standard columns in the standard order: ``lon, lat, depth``, the value
    columns, then the optional gradient/search columns when present. Any other column is kept at
    the end with a warning.

    Raises:
        ValueError: If lon/lat/depth or a value column is missing.
    """
    required = GEO_COLUMNS + list(value_columns)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Velocity data is missing required columns: {missing}")
    optional = [c for c in OPTIONAL_VELOCITY_COLUMNS if c in df.columns]
    ordered = GEO_COLUMNS + list(value_columns) + optional
    extras = [c for c in df.columns if c not in ordered]
    if extras:
        logger.warning(f"Velocity data has non-standard columns {extras}; keeping them")
    return df[ordered + extras]


# ---------------------------------------------------------------------------
# Format detection and coarse subsampling
# ---------------------------------------------------------------------------

def _detect_format(filepath: Union[str, Path]) -> str:
    """Detect the file format ('csv' or 'hdf5') from the extension."""
    ext = Path(filepath).suffix.lower()
    if ext == '.csv':
        return 'csv'
    elif ext in ['.h5', '.hdf5']:
        return 'hdf5'
    raise ValueError(
        f"Cannot detect format from extension '{ext}'. "
        f"Use .csv, .h5, or .hdf5, or specify format explicitly."
    )


def _validate_coarse_grid_config(coarse_config: Dict[str, float],
                                 fine_dlon: float, fine_dlat: float, fine_ddepth: float
                                 ) -> Tuple[int, int, int]:
    """
    Stride factors of a coarse grid: each coarse spacing (``dlon_coarse``, ``dlat_coarse``,
    ``ddepth_coarse``) must be an integral multiple of the fine spacing.

    Raises:
        ValueError: If a coarse spacing is not an integral multiple of the fine spacing.
    """
    strides = []
    for axis, fine in zip(GRID_AXES, (fine_dlon, fine_dlat, fine_ddepth)):
        key = f'{_SPACING[axis]}_coarse'
        ratio = coarse_config[key] / fine
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(f"{key} ({coarse_config[key]}) must be an integral multiple of the fine "
                             f"{_SPACING[axis]} ({fine}). Ratio: {ratio}")
        strides.append(int(round(ratio)))
    return tuple(strides)


def _subsample_chunk(df_chunk: pd.DataFrame, lon_coarse: List[float], lat_coarse: List[float],
                     depth_coarse: List[float]) -> pd.DataFrame:
    """Rows of a chunk that lie on the coarse grid."""
    mask = (df_chunk['lon'].isin(lon_coarse) & df_chunk['lat'].isin(lat_coarse)
            & df_chunk['depth'].isin(depth_coarse))
    return df_chunk[mask].copy()


def subsample_to_coarse_grid(
    df: pd.DataFrame,
    dlon_coarse: float,
    dlat_coarse: float,
    ddepth_coarse: float,
    fine_dlon: float,
    fine_dlat: float,
    fine_ddepth: float,
    n_jobs: int = 1,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Coarse-sampled version of a velocity grid: every stride-th node along longitude, latitude and
    depth, plus the last node of each axis.

    Args:
        df: Fine grid DataFrame
        dlon_coarse, dlat_coarse: Coarse spacing in degrees, multiples of the fine spacing
        ddepth_coarse: Coarse spacing in km, a multiple of the fine spacing
        fine_dlon, fine_dlat, fine_ddepth: Fine spacing
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for no parallelization)
        show_progress: Whether to show progress

    Raises:
        ValueError: If a coarse spacing is not an integral multiple of the fine spacing
    """
    if show_progress:
        logger.info(f"Subsampling velocity grid: ({fine_dlon}°, {fine_dlat}°, {fine_ddepth} km) → "
                    f"({dlon_coarse}°, {dlat_coarse}°, {ddepth_coarse} km)")
    strides = _validate_coarse_grid_config(
        {'dlon_coarse': dlon_coarse, 'dlat_coarse': dlat_coarse, 'ddepth_coarse': ddepth_coarse},
        fine_dlon, fine_dlat, fine_ddepth)

    coarse = []
    for axis, stride in zip(GRID_AXES, strides):
        values = sorted(df[axis].unique())
        selected = values[::stride]
        if values[-1] not in selected:
            selected.append(values[-1])
        coarse.append(selected)
    expected_points = int(np.prod([len(c) for c in coarse]))
    if show_progress:
        logger.info(f"Coarse grid: {len(coarse[0])} × {len(coarse[1])} × {len(coarse[2])} = "
                    f"{expected_points:,} points (stride: {strides[0]}, {strides[1]}, {strides[2]})")

    if n_jobs == 1:
        df_coarse = _subsample_chunk(df, *coarse)
    else:
        n_chunks = max(os.cpu_count() if n_jobs <= 0 else n_jobs, 1)
        chunks = np.array_split(df, n_chunks)
        results = Parallel(n_jobs=n_jobs)(delayed(_subsample_chunk)(chunk, *coarse) for chunk in chunks)
        df_coarse = pd.concat(results, ignore_index=True)

    if len(df_coarse) != expected_points:
        warnings.warn(f"Coarse grid point mismatch: expected {expected_points:,}, got {len(df_coarse):,}. "
                      f"This may indicate irregular grid spacing.")
    if show_progress:
        logger.info(f"Subsampling complete: {len(df):,} → {len(df_coarse):,} points "
                    f"({len(df) / max(len(df_coarse), 1):.1f}x reduction)")
    return df_coarse


# ---------------------------------------------------------------------------
# Read and write
# ---------------------------------------------------------------------------

def _read_csv_velocity(filepath: Union[str, Path], show_progress: bool = True,
                       columns: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
    """Read a velocity CSV; the metadata comment lines at the top are skipped."""
    kwargs.setdefault('comment', _CSV_COMMENT)
    if columns is not None:
        available = pd.read_csv(filepath, nrows=0, comment=_CSV_COMMENT).columns
        kwargs['usecols'] = [c for c in columns if c in available]
    if show_progress:
        with open(filepath, 'r') as f:
            total_lines = sum(1 for line in f if not line.startswith(_CSV_COMMENT)) - 1
        chunksize = kwargs.pop('chunksize', 10000)
        chunks = pd.read_csv(filepath, chunksize=chunksize, **kwargs)
        with tqdm(total=total_lines, desc=f"Reading {Path(filepath).name}") as pbar:
            df_list = []
            for chunk in chunks:
                df_list.append(chunk)
                pbar.update(len(chunk))
        return pd.concat(df_list, ignore_index=True)
    return pd.read_csv(filepath, **kwargs)


def _read_hdf5_velocity(filepath: Union[str, Path], dataset: str = 'velocity',
                        show_progress: bool = True, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a velocity dataset ('velocity' or 'velocity_coarse') from an HDF5 file."""
    with h5py.File(filepath, 'r', locking=False) as f:
        if dataset not in f:
            raise KeyError(f"Dataset '{dataset}' not found in {filepath}. Available datasets: {list(f.keys())}")
        if 'columns' in f[dataset].attrs:
            all_columns = [c.decode('utf-8') if isinstance(c, bytes) else c for c in f[dataset].attrs['columns']]
        else:
            all_columns = list(VELOCITY_COLUMNS)
        if columns is None:
            data = f[dataset][:]
            selected = all_columns
        else:
            indices = sorted({all_columns.index(c) for c in columns if c in all_columns})
            data = f[dataset][:, indices]
            selected = [all_columns[i] for i in indices]
    df = pd.DataFrame(data, columns=selected)
    if show_progress:
        logger.info(f"Loaded {len(df):,} points from {filepath}:{dataset}")
    return df


def _write_hdf5_velocity(filepath: Union[str, Path], df: pd.DataFrame, dataset: str,
                         metadata: Optional[Dict[str, Any]], compression: str,
                         compression_opts: Optional[int], show_progress: bool = True,
                         chunk_size: int = 1000000) -> None:
    """Write a velocity dataset to an HDF5 file, with its metadata as attributes."""
    mode = 'a' if os.path.exists(filepath) and dataset != 'velocity' else 'w'
    metadata = flatten_metadata(metadata)
    n_rows = len(df)
    with h5py.File(filepath, mode, locking=False) as f:
        if dataset in f:
            del f[dataset]
        dset = f.create_dataset(
            dataset, shape=(n_rows, len(df.columns)), dtype=np.float64,
            compression=compression if compression not in (None, 'null') else None,
            compression_opts=compression_opts if compression == 'gzip' else None,
            chunks=True)
        starts = range(0, n_rows, chunk_size)
        for start in (tqdm(starts, desc=f"Writing {Path(filepath).name}") if show_progress else starts):
            dset[start:start + chunk_size, :] = df.iloc[start:start + chunk_size].values
        dset.attrs['columns'] = [c.encode('utf-8') for c in df.columns.tolist()]
        for key, value in metadata.items():
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
    Read a velocity file in CSV or HDF5 format.

    The file metadata is attached as ``df.attrs['metadata']``; use ``resolve_grid_metadata(df)``
    to get the verified grid.

    Args:
        filepath: Path to velocity file (.csv or .h5/.hdf5)
        format: 'auto' (detect from extension), 'csv', or 'hdf5'
        dataset: HDF5 dataset name ('velocity' or 'velocity_coarse')
        show_progress: Whether to show progress
        columns: Optional list of column names to read
        **kwargs: Additional arguments for pd.read_csv (CSV only)

    Returns:
        DataFrame with columns lon, lat, depth, vp, vs (plus grad_*/srch_* in gradient files)
    """
    filepath = Path(resolve_velocity_path(filepath))
    if not filepath.exists():
        raise FileNotFoundError(f"Velocity file not found: {filepath}")
    if format == 'auto':
        format = _detect_format(filepath)
    if format == 'csv':
        metadata = _read_csv_header_metadata(filepath)
        df = _read_csv_velocity(filepath, show_progress=show_progress, columns=columns, **kwargs)
    elif format == 'hdf5':
        metadata = read_velocity_metadata(filepath, dataset=dataset)
        df = _read_hdf5_velocity(filepath, dataset=dataset, show_progress=show_progress, columns=columns)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'csv' or 'hdf5'.")
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
    Write a velocity file in CSV or HDF5 format.

    The columns are standardized (``lon, lat, depth, vp, vs`` plus optional gradient/search
    columns) and the file metadata is written in both formats: HDF5 attributes, or
    ``# key: value`` lines at the top of a CSV. The grid keys (``nlon..odepth``) are taken from
    ``metadata`` when given, cross-checked against the lon/lat/depth columns, and inferred from
    the columns otherwise.

    Args:
        df: Velocity DataFrame to write
        filepath: Output path (.csv or .h5/.hdf5)
        format: 'auto' (detect from extension), 'csv', or 'hdf5'
        coarse_grid_config: Optional coarse subsampling (HDF5 only): ``dlon``, ``dlat`` (degrees)
            and ``ddepth`` (km), each an integral multiple of the fine spacing (the ``*_coarse``
            spellings are accepted too)
        compression: HDF5 compression ('gzip', 'lzf', None)
        compression_opts: Compression level (0-9 for gzip)
        metadata: Metadata dictionary (see module docstring)
        show_progress: Whether to show progress
        n_jobs: Number of parallel jobs for coarse subsampling
        **kwargs: Additional arguments for to_csv
    """
    filepath = Path(filepath)
    ensure_directory_exists(filepath.parent)
    if format == 'auto':
        format = _detect_format(filepath)

    metadata = normalize_metadata(metadata)
    metadata.setdefault('data_type', 'velocity')
    df = standardize_velocity_columns(df, VELOCITY_VALUE_COLUMNS)
    metadata.update(resolve_grid_metadata(df, metadata, source=filepath.name))
    for name in list(df.columns) + [k for k in metadata if k in VELOCITY_UNITS]:
        if name in VELOCITY_UNITS:
            metadata.setdefault(f'units.{name}', VELOCITY_UNITS[name])

    if format == 'csv':
        with open(filepath, 'w', newline='') as f:
            f.writelines(_csv_header_lines(flatten_metadata(metadata)))
            df.to_csv(f, index=False, **kwargs)
        if show_progress:
            logger.info(f"Wrote {len(df):,} points to {filepath} "
                        f"({os.path.getsize(filepath) / (1024 ** 2):.1f} MB)")
    elif format == 'hdf5':
        _write_hdf5_velocity(filepath, df, 'velocity', metadata, compression, compression_opts, show_progress)
        if coarse_grid_config:
            coarse_grid_config = {
                f'{key}_coarse': coarse_grid_config[f'{key}_coarse'] if f'{key}_coarse' in coarse_grid_config
                else coarse_grid_config[key]
                for key in ('dlon', 'dlat', 'ddepth')
            }
            fine_dlon, fine_dlat, fine_ddepth = _fine_spacing(get_fine_grid_metadata(metadata), coarse_grid_config)
            df_coarse = subsample_to_coarse_grid(
                df, coarse_grid_config['dlon_coarse'], coarse_grid_config['dlat_coarse'],
                coarse_grid_config['ddepth_coarse'], fine_dlon, fine_dlat, fine_ddepth,
                n_jobs=n_jobs, show_progress=show_progress)
            _write_hdf5_velocity(filepath, df_coarse, 'velocity_coarse',
                                 build_coarse_metadata(df_coarse, metadata, coarse_grid_config),
                                 compression, compression_opts, show_progress)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'csv' or 'hdf5'.")


def get_velocity_file_metadata(filepath: Union[str, Path], dataset: str = 'velocity') -> Dict[str, Any]:
    """
    Metadata of a velocity file with the grid verified against the columns, plus ``format``,
    ``num_points``, ``bounds`` (min and max of lon, lat, depth), ``file_size_mb``, and for HDF5
    ``datasets`` and ``has_coarse_dataset``.
    """
    filepath = Path(resolve_velocity_path(filepath))
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    format_type = _detect_format(filepath)
    info: Dict[str, Any] = {'format': format_type, 'file_size_mb': os.path.getsize(filepath) / (1024 ** 2)}
    if format_type == 'hdf5':
        with h5py.File(filepath, 'r', locking=False) as f:
            info['datasets'] = list(f.keys())
            info['has_coarse_dataset'] = 'velocity_coarse' in f
            if dataset not in f:
                raise KeyError(f"Dataset '{dataset}' not found in {filepath}")
    metadata = read_velocity_metadata(filepath, dataset=dataset)
    df_axes = read_velocity_file(filepath, dataset=dataset, show_progress=False, columns=GEO_COLUMNS)
    info['num_points'] = len(df_axes)
    info['bounds'] = {axis: (float(df_axes[axis].min()), float(df_axes[axis].max())) for axis in GEO_COLUMNS}
    grid = resolve_grid_metadata(df_axes, metadata, source=filepath.name)
    return {**metadata, **grid, **info}
