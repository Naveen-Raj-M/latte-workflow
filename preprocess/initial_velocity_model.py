"""Initial velocity model creation and binary file generation (spherical Earth).

This module builds a 3D velocity model from an initial 1D model on the grid of the spherical
LATTE branch: regular in longitude and WGS84 geodetic latitude (degrees) and in depth below
mean sea level (km). It writes the model as CSV/HDF5 (km/s) and as LATTE binaries (m/s, float32,
Fortran order (ndepth, nlat, nlon)), and the grid lines of the LATTE parameter file.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.interpolate import griddata
from scipy.ndimage import convolve
from joblib import Parallel, delayed
from tqdm import tqdm

from utils.io_utils import ensure_directory_exists
from utils.velocity_io import write_velocity_file, build_coarse_metadata, subsample_to_coarse_grid
from utils.spherical import (EARTH_RADIUS_KM, GRID_AXES, check_grid, grid_axes,
                             latte_grid_lines, node_count)


def load_initial_model(filepath: str,
                       max_depth_km: Optional[float] = None) -> pd.DataFrame:
    """Load initial 1D velocity model from CSV file.

    Expected CSV format: depth,vp,vs (with header)
    - depth in km (positive down)
    - vp, vs in km/s

    Args:
        filepath: Path to initial 1D model CSV file.
        max_depth_km: Maximum depth to load (filters data).

    Returns:
        DataFrame with columns [depth, vp, vs], sorted by depth.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If CSV format is invalid.
    """
    logger = logging.getLogger(__name__)

    if not Path(filepath).exists():
        raise FileNotFoundError(f"Initial model file not found: {filepath}")

    logger.info(f"Loading initial 1D model from: {filepath}")

    # Read CSV file
    try:
        # First, try reading with header
        model_df = pd.read_csv(filepath)

        # Check if the file has no header (first row contains numeric data)
        # If column names look like numbers, it means there's no header
        if all(col.replace('.', '').replace('-', '').isdigit() or
               col.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit()
               for col in model_df.columns):
            # Re-read without header and assign column names
            model_df = pd.read_csv(filepath, header=None)
            # Assuming format: depth, radius, vp, vs
            if len(model_df.columns) == 4:
                model_df.columns = ['depth', 'radius', 'vp', 'vs']
            elif len(model_df.columns) == 3:
                model_df.columns = ['depth', 'vp', 'vs']
            else:
                raise ValueError(
                    f"Unexpected number of columns: {len(model_df.columns)}. "
                    f"Expected 3 (depth, vp, vs) or 4 (depth, radius, vp, vs)"
                )
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    # Validate required columns (radius column is optional)
    required_cols = ['depth', 'vp', 'vs']
    missing_cols = [col for col in required_cols if col not in model_df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV missing required columns: {missing_cols}. "
            f"Expected: {required_cols}. Found columns: {list(model_df.columns)}"
        )

    # Keep only required columns (drop radius if present)
    model_df = model_df[required_cols].copy()

    # Filter by max depth if specified
    if max_depth_km is not None:
        model_df = model_df[model_df['depth'] <= max_depth_km].copy()
        logger.info(f"Filtered model to max_depth_km = {max_depth_km} km")

    # Sort by depth
    model_df = model_df.sort_values('depth').reset_index(drop=True)

    logger.info(f"Loaded {len(model_df)} depth levels")
    logger.info(f"Depth range: {model_df['depth'].min():.2f} - "
               f"{model_df['depth'].max():.2f} km")

    return model_df


def get_velocity_for_depth(target_depth: float,
                           model_df: pd.DataFrame) -> Tuple[float, float]:
    """Find Vp and Vs from 1D model for a given depth.

    Uses layer-based lookup: selects the layer with the largest depth
    that is less than or equal to target_depth.

    Args:
        target_depth: Depth in km (positive down, negative = above MSL).
        model_df: DataFrame with columns [depth, vp, vs].

    Returns:
        Tuple of (vp, vs) in km/s.

    Raises:
        ValueError: If model_df is empty.

    Note:
        For depths above MSL (negative values), uses the model value at that depth
        if the model contains data there. Falls back to the shallowest layer only
        when the target depth is shallower than all entries in the model.
    """
    if len(model_df) == 0:
        raise ValueError("The 1D model data is empty.")

    depths = model_df['depth'].values
    vp_values = model_df['vp'].values
    vs_values = model_df['vs'].values

    # If target_depth is shallower than the first layer, use first layer
    if target_depth < depths[0]:
        return vp_values[0], vs_values[0]

    # If target_depth is deeper than the last layer, use last layer
    if target_depth > depths[-1]:
        return vp_values[-1], vs_values[-1]

    # Find the best matching layer (largest depth <= target_depth)
    idx = np.searchsorted(depths, target_depth, side='right') - 1
    idx = max(0, min(idx, len(depths) - 1))  # Ensure valid index

    return vp_values[idx], vs_values[idx]


def _snap_down(value: float, origin: float, step: float) -> float:
    """Largest node ``origin + i * step`` (integer i) at or below ``value``."""
    return origin + np.floor((value - origin) / step + 1e-9) * step


def _snap_up(value: float, origin: float, step: float) -> float:
    """Smallest node ``origin + i * step`` (integer i) at or above ``value``."""
    return origin + np.ceil((value - origin) / step - 1e-9) * step


def _axis_from_data(name: str, unit: str, origin: float, step: float,
                    data_min: float, data_max: float, logger) -> Tuple[float, float]:
    """Grid range of one axis covering the data: it starts at ``origin`` and is extended
    beyond it, with a warning, only if data lie on the negative side of the origin."""
    lo = origin
    if data_min < origin:
        lo = _snap_down(data_min, origin, step)
        logger.warning(f"Sources/receivers lie beyond the grid origin along {name}: data min "
                       f"{data_min:.6g} {unit} < origin {origin:.6g} {unit}. Extending the grid to {lo:.6g} {unit}.")
    hi = _snap_up(data_max, origin, step)
    logger.info(f"Data {name} range: [{data_min:.6g}, {data_max:.6g}] {unit} -> grid [{lo:.6g}, {hi:.6g}] {unit}")
    return lo, hi


def define_spherical_grid(lon_range: Tuple[float, float], lat_range: Tuple[float, float],
                          depth_range: Tuple[float, float],
                          grid_spacing: Tuple[float, float, float]) -> dict:
    """Grid definition (``nlon .. odepth``) of a grid from its first to last node on each axis.

    Args:
        lon_range: (west, east) node longitudes, degrees.
        lat_range: (south, north) node geodetic latitudes, degrees.
        depth_range: (top, bottom) node depths below MSL, km.
        grid_spacing: (dlon, dlat) in degrees and ddepth in km.
    """
    grid = {}
    for (lo, hi), step, axis in zip((lon_range, lat_range, depth_range), grid_spacing, GRID_AXES):
        if hi < lo:
            raise ValueError(f"Grid {axis} range is empty: [{lo}, {hi}]")
        grid[f'n{axis}'] = node_count(lo, hi, step)
        grid[f'd{axis}'] = float(step)
        grid[f'o{axis}'] = float(lo)
    return grid


def create_spherical_grid(grid: dict) -> pd.DataFrame:
    """All nodes of a grid as a DataFrame with columns [lon, lat, depth] (degrees, degrees,
    km below MSL), depth varying fastest, then latitude, then longitude."""
    logger = logging.getLogger(__name__)
    lons, lats, depths = grid_axes(grid)

    logger.info("Creating spherical grid:")
    logger.info(f"  lon: [{lons[0]:.6g}, {lons[-1]:.6g}] deg, {len(lons)} points")
    logger.info(f"  lat: [{lats[0]:.6g}, {lats[-1]:.6g}] deg, {len(lats)} points")
    logger.info(f"  depth: [{depths[0]:.6g}, {depths[-1]:.6g}] km, {len(depths)} points")

    lon_grid, lat_grid, depth_grid = np.meshgrid(lons, lats, depths, indexing='ij')
    grid_df = pd.DataFrame({
        'lon': lon_grid.ravel(),
        'lat': lat_grid.ravel(),
        'depth': depth_grid.ravel()
    })

    logger.info(f"Created grid with {len(grid_df)} points")
    return grid_df


def _grid_indices(grid_df: pd.DataFrame, grid: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """0-based (i, j, k) array indices of the grid points along lon, lat and depth."""
    return tuple(
        np.round((grid_df[axis].values - grid[f'o{axis}']) / grid[f'd{axis}']).astype(int)
        for axis in GRID_AXES
    )


def _get_velocities_vectorized(depths_array, initial_model):
    """Vectorized velocity lookup for an array of depths.

    Args:
        depths_array: NumPy array of depths (km)
        initial_model: DataFrame with columns [depth, vp, vs]

    Returns:
        Tuple of (vp_array, vs_array) in km/s
    """
    model_depths = initial_model['depth'].values
    model_vp = initial_model['vp'].values
    model_vs = initial_model['vs'].values

    # Step lookup: find the deepest layer with depth <= target depth
    indices = np.searchsorted(model_depths, depths_array, side='right') - 1

    # Clamp indices to valid range
    indices = np.clip(indices, 0, len(model_depths) - 1)

    # Lookup velocities
    vp_array = model_vp[indices]
    vs_array = model_vs[indices]

    return vp_array, vs_array


def _assign_velocities_vectorized(grid_df, initial_model, logger):
    """Assign Vp and Vs of the 1D model to every grid point by its depth (no chunking).

    Best for grids that fit comfortably in memory.

    Returns:
        DataFrame with added vp, vs columns
    """
    logger.info("Using full vectorization (grid fits in memory)")
    logger.info("Looking up velocities from initial model...")
    vp_vals, vs_vals = _get_velocities_vectorized(grid_df['depth'].values, initial_model)
    grid_df['vp'] = vp_vals
    grid_df['vs'] = vs_vals
    return grid_df


def _assign_velocities_chunked_vectorized(grid_df, initial_model, chunk_size, n_jobs, logger):
    """Assign Vp and Vs of the 1D model by depth, in parallel chunks, for large grids.

    Returns:
        DataFrame with added vp, vs columns
    """
    logger.info(f"Using chunked vectorization (large grid: {len(grid_df):,} points)")
    logger.info(f"Chunk size: {chunk_size:,}, Workers: {n_jobs if n_jobs != -1 else 'all cores'}")

    depth_vals = grid_df['depth'].values
    chunk_ranges = [
        (i, min(i + chunk_size, len(grid_df)))
        for i in range(0, len(grid_df), chunk_size)
    ]
    logger.info(f"Processing in {len(chunk_ranges)} chunks...")

    def process_chunk_vectorized(start_idx, end_idx):
        vp_chunk, vs_chunk = _get_velocities_vectorized(depth_vals[start_idx:end_idx], initial_model)
        return (start_idx, vp_chunk, vs_chunk)

    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_chunk_vectorized)(start, end)
        for start, end in tqdm(chunk_ranges, desc="Processing chunks", unit="chunk")
    )

    logger.info("Combining results...")
    vp_vals = np.empty(len(grid_df), dtype=np.float64)
    vs_vals = np.empty(len(grid_df), dtype=np.float64)
    for start_idx, vp_chunk, vs_chunk in results:
        end_idx = start_idx + len(vp_chunk)
        vp_vals[start_idx:end_idx] = vp_chunk
        vs_vals[start_idx:end_idx] = vs_chunk

    grid_df['vp'] = vp_vals
    grid_df['vs'] = vs_vals
    return grid_df


def _grid_from_config(vel_config: DictConfig, events_df: pd.DataFrame,
                      stations_df: pd.DataFrame, logger) -> dict:
    """Grid definition from the configuration: automatic bounds covering the events and
    stations, or manual bounds."""
    dlon, dlat, ddepth = (float(v) for v in vel_config.grid_spacing)

    if vel_config.use_auto_bounds:
        logger.info("Auto-detecting grid bounds from filtered events and stations...")
        both = pd.concat([events_df[['longitude', 'latitude', 'depth']],
                          stations_df[['longitude', 'latitude', 'depth']]], ignore_index=True)
        origin_lon = float(vel_config.origin_lon)
        origin_lat = float(vel_config.origin_lat)
        origin_depth = float(vel_config.origin_depth_km)

        lon_range = _axis_from_data('longitude', 'deg', origin_lon, dlon,
                                    both['longitude'].min(), both['longitude'].max(), logger)
        lat_range = _axis_from_data('latitude', 'deg', origin_lat, dlat,
                                    both['latitude'].min(), both['latitude'].max(), logger)
        depth_lo, depth_hi = _axis_from_data('depth', 'km', origin_depth, ddepth,
                                             both['depth'].min(), both['depth'].max(), logger)
        # Deepest node: max_depth_km if specified, otherwise the deepest data
        if vel_config.max_depth_km is not None:
            depth_hi = _snap_up(float(vel_config.max_depth_km), origin_depth, ddepth)
            logger.info(f"Using max_depth_km={vel_config.max_depth_km} km -> deepest node {depth_hi:.6g} km")
        depth_range = (depth_lo, depth_hi)
    else:
        logger.info("Using manual bounds for grid")
        lon_range = (float(vel_config.lon_min), float(vel_config.lon_max))
        lat_range = (float(vel_config.lat_min), float(vel_config.lat_max))
        depth_range = (float(vel_config.depth_min), float(vel_config.depth_max))

    return define_spherical_grid(lon_range, lat_range, depth_range, (dlon, dlat, ddepth))


def write_latte_grid_file(filepath: Path, grid: dict, earth_radius_km: float) -> None:
    """Write the grid lines of a spherical LATTE parameter file (lengths in metres)."""
    ensure_directory_exists(filepath.parent)
    with open(filepath, 'w') as f:
        f.write("\n".join(latte_grid_lines(grid, earth_radius_km)) + "\n")
    logging.getLogger(__name__).info(f"LATTE grid parameters written: {filepath}")


def create_velocity_model(config: DictConfig,
                          events_df: pd.DataFrame,
                          stations_df: pd.DataFrame) -> None:
    """Create the initial velocity model on the spherical grid and generate binaries.

    The grid is regular in longitude, geodetic latitude (degrees) and depth below MSL (km).
    Every node takes the Vp and Vs of the 1D model at its depth.

    Args:
        config: Full configuration.
        events_df: Events with longitude, latitude and depth (km).
        stations_df: Stations with longitude, latitude and depth (km).
    """
    logger = logging.getLogger(__name__)
    vel_config = config.initial_velocity_model

    if not vel_config.enabled:
        logger.info("Initial velocity model generation disabled")
        return

    logger.info("Creating initial velocity model...")
    earth_radius_km = float(config.coordinate_transform.get('earth_radius_km', EARTH_RADIUS_KM))

    # Load initial 1D model
    initial_model = load_initial_model(
        vel_config.initial_model_file,
        max_depth_km=vel_config.max_depth_km
    )

    # Grid definition, stored in the metadata of every output file; downstream modules
    # read it from there instead of from their configuration.
    grid_meta = _grid_from_config(vel_config, events_df, stations_df, logger)
    check_grid(grid_meta, earth_radius_km)
    grid_spacing = (grid_meta['dlon'], grid_meta['dlat'], grid_meta['ddepth'])

    grid_df = create_spherical_grid(grid_meta)

    # Get parallel processing configuration
    parallel_config = config.get('parallel', {})
    n_jobs = parallel_config.get('num_workers', None)
    if n_jobs is None:
        n_jobs = -1  # Use all CPU cores

    grid_chunk_size = parallel_config.get('grid_chunk_size', 50000)

    # Hybrid vectorization: Choose approach based on grid size
    vectorize_threshold = parallel_config.get('vectorize_max_points', 100_000_000)  # 100M points

    logger.info(f"Assigning velocities from initial model...")
    logger.info(f"Grid size: {len(grid_df):,} points")
    logger.info(f"Vectorization threshold: {vectorize_threshold:,} points")

    # Estimate memory requirements (rough estimate: 8 bytes per float * 5 arrays)
    estimated_memory_mb = (len(grid_df) * 8 * 5) / (1024 * 1024)
    logger.info(f"Estimated memory requirement: ~{estimated_memory_mb:.1f} MB")

    if len(grid_df) < vectorize_threshold:
        grid_df = _assign_velocities_vectorized(grid_df, initial_model, logger)
    else:
        grid_df = _assign_velocities_chunked_vectorized(
            grid_df, initial_model, grid_chunk_size, n_jobs, logger
        )

    # Save velocity model (CSV and/or HDF5)
    output_format = vel_config.get('output_format', 'csv')
    logger.info(f"Saving velocity model in format: {output_format}")

    # Metadata written to every output (HDF5 attributes / CSV header)
    metadata = {
        **grid_meta,
        'coordinate_system': 'spherical',
        'earth_radius_km': earth_radius_km,
        'data_type': 'velocity',
    }

    # Prepare coarse grid config if enabled (dlon, dlat in degrees, ddepth in km)
    coarse_config = None
    hdf5_opts = vel_config.get('hdf5_options', {})
    if hdf5_opts.get('create_coarse_dataset', False) or hdf5_opts.get('write_coarse_csv', False):
        coarse_config = {
            'dlon_coarse': float(hdf5_opts.coarse_grid.dlon),
            'dlat_coarse': float(hdf5_opts.coarse_grid.dlat),
            'ddepth_coarse': float(hdf5_opts.coarse_grid.ddepth),
        }

    # Write CSV if requested
    if output_format in ['csv', 'both']:
        output_csv = Path(vel_config.output_csv)
        ensure_directory_exists(output_csv.parent)
        logger.info(f"Saving velocity model CSV: {output_csv}")
        write_velocity_file(
            grid_df,
            output_csv,
            format='csv',
            metadata=metadata,
            show_progress=hdf5_opts.get('show_progress', True)
        )

    # Write HDF5 if requested
    if output_format in ['hdf5', 'both']:
        output_hdf5 = Path(vel_config.output_hdf5)
        ensure_directory_exists(output_hdf5.parent)
        logger.info(f"Saving velocity model HDF5: {output_hdf5}")
        write_velocity_file(
            grid_df,
            output_hdf5,
            format='hdf5',
            coarse_grid_config=coarse_config,
            compression=hdf5_opts.get('compression', 'gzip'),
            compression_opts=hdf5_opts.get('compression_level', 4),
            metadata=metadata,
            show_progress=hdf5_opts.get('show_progress', True),
            n_jobs=hdf5_opts.get('n_jobs', 1)
        )

        # Log file size comparison if both formats written
        if output_format == 'both':
            csv_size = output_csv.stat().st_size / (1024**2)
            hdf5_size = output_hdf5.stat().st_size / (1024**2)
            compression_ratio = csv_size / hdf5_size if hdf5_size > 0 else 0
            logger.info(
                f"File sizes: CSV={csv_size:.1f}MB, HDF5={hdf5_size:.1f}MB "
                f"(compression ratio: {compression_ratio:.1f}x)"
            )

    # Write separate coarse CSV if requested
    if hdf5_opts.get('write_coarse_csv', False) and coarse_config:
        output_coarse_csv = Path(hdf5_opts.get('output_coarse_csv',
                                               vel_config.output_csv.replace('.csv', '_coarse.csv')))
        ensure_directory_exists(output_coarse_csv.parent)

        logger.info(f"Creating coarse-sampled CSV: {output_coarse_csv}")

        grid_df_coarse = subsample_to_coarse_grid(
            grid_df,
            dlon_coarse=coarse_config['dlon_coarse'],
            dlat_coarse=coarse_config['dlat_coarse'],
            ddepth_coarse=coarse_config['ddepth_coarse'],
            fine_dlon=grid_meta['dlon'],
            fine_dlat=grid_meta['dlat'],
            fine_ddepth=grid_meta['ddepth'],
            n_jobs=hdf5_opts.get('n_jobs', 1),
            show_progress=hdf5_opts.get('show_progress', True)
        )

        # Write coarse CSV (its own grid plus the fine grid and strides in the metadata)
        write_velocity_file(
            grid_df_coarse,
            output_coarse_csv,
            format='csv',
            metadata=build_coarse_metadata(grid_df_coarse, metadata, coarse_config),
            show_progress=hdf5_opts.get('show_progress', True)
        )

        coarse_size = output_coarse_csv.stat().st_size / (1024**2)
        logger.info(f"Coarse CSV written: {len(grid_df_coarse):,} points ({coarse_size:.1f}MB)")

    # Grid lines of the spherical LATTE parameter file
    latte_grid_file = config.coordinate_transform.get('latte_grid_file', None)
    if latte_grid_file:
        write_latte_grid_file(Path(latte_grid_file), grid_meta, earth_radius_km)

    nlon, nlat, ndepth = grid_meta['nlon'], grid_meta['nlat'], grid_meta['ndepth']
    dlon, dlat, ddepth = grid_spacing
    num_sources = len(events_df)

    logger.info("Velocity model created successfully")
    logger.info(f"  Grid dimensions (nlon, nlat, ndepth): {nlon} x {nlat} x {ndepth}")
    logger.info(f"  Grid spacing (dlon, dlat, ddepth): {dlon:g} deg x {dlat:g} deg x {ddepth:g} km")
    logger.info(f"  Total grid points: {len(grid_df):,}")
    logger.info(f"  Events used for grid bounds: {num_sources}")
    logger.info(f"  Vp range: {grid_df['vp'].min():.3f} - "
               f"{grid_df['vp'].max():.3f} km/s")
    logger.info(f"  Vs range: {grid_df['vs'].min():.3f} - "
               f"{grid_df['vs'].max():.3f} km/s")
    logger.info(f"  Depth range: {grid_df['depth'].min():.2f} - "
               f"{grid_df['depth'].max():.2f} km")

    # Generate binary files
    _generate_binary_files(config, grid_df, grid_meta)

    # Print final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("VELOCITY MODEL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Events used for grid bounds: {num_sources}")
    logger.info(f"Grid dimensions (nlon, nlat, ndepth): {nlon} x {nlat} x {ndepth}")
    logger.info(f"Grid spacing (dlon, dlat, ddepth): {dlon:g} deg x {dlat:g} deg x {ddepth:g} km")
    logger.info(f"Grid origin (olon, olat, odepth): {grid_meta['olon']:g} deg, "
                f"{grid_meta['olat']:g} deg, {grid_meta['odepth']:g} km")
    logger.info(f"Total grid points: {len(grid_df):,}")
    logger.info("=" * 60)


def _fill_grid_vectorized(grid_df, vel_col, nlon, nlat, ndepth, indices, logger):
    """Fill 3D grid array using vectorization (no chunking).

    Args:
        grid_df: DataFrame with velocity data
        vel_col: Column name ('vp' or 'vs')
        nlon, nlat, ndepth: Grid dimensions
        indices: 0-based (i, j, k) array indices of the grid points along lon, lat, depth
        logger: Logger instance

    Returns:
        3D numpy array in Fortran order
    """
    logger.info(f"Using full vectorization to fill {vel_col.upper()} grid")

    # Create 3D grid array in Fortran-order
    grid_3d = np.zeros((ndepth, nlat, nlon), dtype=np.float32, order='F')

    # Extract indices and velocities as numpy arrays
    i_vals, j_vals, k_vals = indices
    vel_vals = grid_df[vel_col].values * 1000  # Convert to m/s

    # Create mask for valid indices
    valid_mask = (
        (i_vals >= 0) & (i_vals < nlon) &
        (j_vals >= 0) & (j_vals < nlat) &
        (k_vals >= 0) & (k_vals < ndepth)
    )

    # Filter to valid indices
    k_valid = k_vals[valid_mask]
    j_valid = j_vals[valid_mask]
    i_valid = i_vals[valid_mask]
    vel_valid = vel_vals[valid_mask]

    # Vectorized assignment
    logger.info(f"Assigning {len(vel_valid):,} values to grid...")
    grid_3d[k_valid, j_valid, i_valid] = vel_valid

    return grid_3d


def _fill_grid_chunked_vectorized(grid_df, vel_col, nlon, nlat, ndepth, indices, chunk_size, n_jobs, logger):
    """Fill 3D grid array using chunked vectorization.

    Args:
        grid_df: DataFrame with velocity data
        vel_col: Column name ('vp' or 'vs')
        nlon, nlat, ndepth: Grid dimensions
        indices: 0-based (i, j, k) array indices of the grid points along lon, lat, depth
        chunk_size: Points per chunk
        n_jobs: Number of workers
        logger: Logger instance

    Returns:
        3D numpy array in Fortran order
    """
    logger.info(f"Using chunked vectorization to fill {vel_col.upper()} grid")
    logger.info(f"Chunk size: {chunk_size:,}, Workers: {n_jobs if n_jobs != -1 else 'all cores'}")

    # Create 3D grid array in Fortran-order
    grid_3d = np.zeros((ndepth, nlat, nlon), dtype=np.float32, order='F')

    # Extract arrays once
    i_vals, j_vals, k_vals = indices
    vel_vals = grid_df[vel_col].values * 1000

    # Calculate chunks
    num_chunks = (len(grid_df) + chunk_size - 1) // chunk_size
    logger.info(f"Processing {len(grid_df):,} points in {num_chunks} chunks...")

    def process_chunk(start_idx, end_idx):
        """Process a chunk and return indices and values."""
        i_chunk = i_vals[start_idx:end_idx]
        j_chunk = j_vals[start_idx:end_idx]
        k_chunk = k_vals[start_idx:end_idx]
        vel_chunk = vel_vals[start_idx:end_idx]

        # Filter valid indices
        valid_mask = (
            (i_chunk >= 0) & (i_chunk < nlon) &
            (j_chunk >= 0) & (j_chunk < nlat) &
            (k_chunk >= 0) & (k_chunk < ndepth)
        )

        return (
            k_chunk[valid_mask],
            j_chunk[valid_mask],
            i_chunk[valid_mask],
            vel_chunk[valid_mask]
        )

    # Process chunks in parallel
    chunk_ranges = [
        (i, min(i + chunk_size, len(grid_df)))
        for i in range(0, len(grid_df), chunk_size)
    ]

    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_chunk)(start, end)
        for start, end in tqdm(chunk_ranges, desc=f"Filling {vel_col.upper()} grid", unit="chunk")
    )

    # Combine results into grid
    logger.info("Combining results into 3D array...")
    for k_chunk, j_chunk, i_chunk, vel_chunk in results:
        grid_3d[k_chunk, j_chunk, i_chunk] = vel_chunk

    return grid_3d


def _generate_binary_files(config: DictConfig, grid_df: pd.DataFrame,
                           grid_meta: dict) -> None:
    """Generate binary velocity files from grid DataFrame.

    Args:
        config: Configuration.
        grid_df: DataFrame with velocity grid data (lon, lat, depth, vp, vs).
        grid_meta: Grid definition (nlon .. odepth).
    """
    logger = logging.getLogger(__name__)
    vel_config = config.initial_velocity_model

    logger.info("Creating velocity binary files...")

    # Get grid dimensions
    nlon, nlat, ndepth = grid_meta['nlon'], grid_meta['nlat'], grid_meta['ndepth']
    indices = _grid_indices(grid_df, grid_meta)

    logger.info(f"Grid dimensions (nlon, nlat, ndepth): {nlon} x {nlat} x {ndepth}")

    # Get parallel processing configuration
    parallel_config = config.get('parallel', {})
    n_jobs = parallel_config.get('num_workers', None)
    if n_jobs is None:
        n_jobs = -1  # Use all CPU cores

    grid_chunk_size = parallel_config.get('grid_chunk_size', 50000)
    vectorize_threshold = parallel_config.get('vectorize_max_points', 100_000_000)

    # Determine which phases to process
    phase_type = vel_config.get('phase_type', 'both')
    phases_to_process = []

    if phase_type in ['both', 'p']:
        phases_to_process.append(('p', vel_config.vp_binary))
    if phase_type in ['both', 's']:
        phases_to_process.append(('s', vel_config.vs_binary))

    # Process each phase
    for phase, output_path in phases_to_process:
        logger.info(f"Processing {phase.upper()}-wave velocity...")

        vel_col = 'vp' if phase == 'p' else 'vs'

        # Always use full vectorization - numpy fancy indexing is fast and memory-efficient
        # The chunked parallel version causes memory issues due to process forking
        grid_3d = _fill_grid_vectorized(grid_df, vel_col, nlon, nlat, ndepth, indices, logger)

        # Interpolate if requested
        if vel_config.get('interpolate', False):
            neighbor_threshold = vel_config.get('neighbor_threshold', 5)
            grid_3d = _interpolate_internal_gaps(grid_3d, neighbor_threshold)

        # Write binary file
        _write_velocity_binary(Path(output_path), grid_3d)

    logger.info("Velocity binary files created successfully")


def _interpolate_internal_gaps(model: np.ndarray,
                               neighbor_threshold: int) -> np.ndarray:
    """Interpolate internal gaps in the velocity model.

    Args:
        model: 3D velocity array.
        neighbor_threshold: Minimum neighbors to interpolate.

    Returns:
        Interpolated 3D velocity array.
    """
    logger = logging.getLogger(__name__)

    known_indices = np.array(np.nonzero(model)).T

    if known_indices.shape[0] < 4:
        logger.warning("Not enough data points for interpolation. Skipping.")
        return model

    known_values = model[tuple(known_indices.T)]
    binary_mask = model > 0

    # Create kernel for neighbor counting
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0

    # Count neighbors
    neighbor_counts = convolve(
        binary_mask.astype(np.uint8),
        kernel,
        mode='constant',
        cval=0
    )

    # Find gaps to fill
    gaps_mask = (model == 0) & (neighbor_counts >= neighbor_threshold)
    gap_count = np.sum(gaps_mask)

    if gap_count == 0:
        logger.info("No internal gaps found requiring interpolation")
        return model

    logger.info(f"Interpolating {gap_count} internal gaps...")

    # Get gap indices
    gap_indices = np.array(np.nonzero(gaps_mask)).T

    # Interpolate
    try:
        interpolated_values = griddata(
            known_indices,
            known_values,
            gap_indices,
            method='nearest'
        )

        # Fill gaps
        model_filled = model.copy()
        for idx, value in zip(gap_indices, interpolated_values):
            model_filled[tuple(idx)] = value

        logger.info("Interpolation completed successfully")
        return model_filled

    except Exception as e:
        logger.warning(f"Interpolation failed: {e}. Using original model.")
        return model


def _write_velocity_binary(filepath: Path, grid: np.ndarray) -> None:
    """Write velocity grid to binary file.

    Format: Raw binary (no header), Fortran-order (column-major).

    Args:
        filepath: Output file path.
        grid: 3D velocity array (ndepth, nlat, nlon) in Fortran-order.
    """
    logger = logging.getLogger(__name__)

    ensure_directory_exists(filepath.parent)

    ndepth, nlat, nlon = grid.shape

    # Write in Fortran-order to match LATTE's column-major format
    grid.astype(np.float32).flatten('F').tofile(str(filepath))

    # Verify the written file
    grid_check = np.fromfile(str(filepath), dtype=np.float32).reshape((ndepth, nlat, nlon), order='F')

    logger.info(f"Written binary file: {filepath}")
    logger.info(f"  File size: {filepath.stat().st_size} bytes")
    logger.info(f"  Shape: {grid.shape}")
    logger.info(f"  Value range: {grid.min():.1f} - {grid.max():.1f} m/s")
    logger.info(f"  Verification: {np.allclose(grid, grid_check)}")
