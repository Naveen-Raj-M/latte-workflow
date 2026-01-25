"""Initial velocity model creation and binary file generation.

This module combines velocity model creation from IASP91 with binary file
generation. It creates a 3D velocity model on a Cartesian grid and writes
both CSV and binary formats.
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
from utils.velocity_io import write_velocity_file


def load_iasp91_model(filepath: str,
                      max_depth_km: Optional[float] = None) -> pd.DataFrame:
    """Load IASP91 velocity model from CSV file.

    Expected CSV format: depth,vp,vs (with header)
    - depth in km (positive down)
    - vp, vs in km/s

    Args:
        filepath: Path to IASP91 CSV file.
        max_depth_km: Maximum depth to load (filters data).

    Returns:
        DataFrame with columns [depth, vp, vs], sorted by depth.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If CSV format is invalid.
    """
    logger = logging.getLogger(__name__)

    if not Path(filepath).exists():
        raise FileNotFoundError(f"IASP91 model file not found: {filepath}")

    logger.info(f"Loading IASP91 model from: {filepath}")

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
        For depths above MSL (negative values), uses velocities at MSL (depth=0).
    """
    if len(model_df) == 0:
        raise ValueError("The 1D model data is empty.")

    depths = model_df['depth'].values
    vp_values = model_df['vp'].values
    vs_values = model_df['vs'].values

    # For depths above MSL (negative), clamp to MSL (depth=0) and use shallowest layer
    if target_depth <= 0:
        # Find velocity at depth=0 or use first layer if depth=0 not in model
        zero_idx = np.where(depths == 0)[0]
        if len(zero_idx) > 0:
            return vp_values[zero_idx[0]], vs_values[zero_idx[0]]
        else:
            return vp_values[0], vs_values[0]

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


def create_cartesian_grid(min_coords: Tuple[float, float, float],
                          max_coords: Tuple[float, float, float],
                          grid_spacing: Tuple[float, float, float]) -> pd.DataFrame:
    """Create a 3D Cartesian grid of points.

    Args:
        min_coords: (x_min, y_min, z_min) in km.
        max_coords: (x_max, y_max, z_max) in km.
        grid_spacing: (dx, dy, dz) spacing in km.

    Returns:
        DataFrame with columns [x, y, z] in km.
    """
    logger = logging.getLogger(__name__)

    x_min, y_min, z_min = min_coords
    x_max, y_max, z_max = max_coords
    dx, dy, dz = grid_spacing

    # Create grid arrays
    xs = np.arange(x_min, x_max + dx/2, dx)
    ys = np.arange(y_min, y_max + dy/2, dy)
    zs = np.arange(z_min, z_max + dz/2, dz)

    logger.info(f"Creating Cartesian grid:")
    logger.info(f"  x: [{x_min:.2f}, {x_max:.2f}] km, {len(xs)} points")
    logger.info(f"  y: [{y_min:.2f}, {y_max:.2f}] km, {len(ys)} points")
    logger.info(f"  z: [{z_min:.2f}, {z_max:.2f}] km, {len(zs)} points")

    # Create meshgrid
    x_grid, y_grid, z_grid = np.meshgrid(xs, ys, zs, indexing='ij')

    # Flatten and create DataFrame with capital case columns
    grid_df = pd.DataFrame({
        'X_grid_km': x_grid.ravel(),
        'Y_grid_km': y_grid.ravel(),
        'Z_grid_km': z_grid.ravel()
    })

    # Also add lowercase for compatibility
    grid_df['x'] = grid_df['X_grid_km']
    grid_df['y'] = grid_df['Y_grid_km']
    grid_df['z'] = grid_df['Z_grid_km']

    logger.info(f"Created grid with {len(grid_df)} points")

    return grid_df


def compute_grid_indices(grid_df: pd.DataFrame, min_coords: Tuple[float, float, float],
                         grid_spacing: Tuple[float, float, float]) -> pd.DataFrame:
    """Compute 1-based grid indices for Fortran compatibility.

    Args:
        grid_df: DataFrame with [X_grid_km, Y_grid_km, Z_grid_km] or [x, y, z] columns.
        min_coords: (x_min, y_min, z_min) for grid origin.
        grid_spacing: (dx, dy, dz) spacing in km.

    Returns:
        DataFrame with added [i_raw, j_raw, k_raw, i, j, k] columns.
    """
    x_min, y_min, z_min = min_coords
    dx, dy, dz = grid_spacing

    # Use capital case if available, otherwise lowercase
    x_col = 'X_grid_km' if 'X_grid_km' in grid_df.columns else 'x'
    y_col = 'Y_grid_km' if 'Y_grid_km' in grid_df.columns else 'y'
    z_col = 'Z_grid_km' if 'Z_grid_km' in grid_df.columns else 'z'

    # Calculate raw (floating-point) indices
    grid_df['i_raw'] = (grid_df[x_col] - x_min) / dx
    grid_df['j_raw'] = (grid_df[y_col] - y_min) / dy
    grid_df['k_raw'] = (grid_df[z_col] - z_min) / dz

    # Calculate final integer indices (1-based for Fortran compatibility)
    grid_df['i'] = np.round(grid_df['i_raw']).astype(int) + 1
    grid_df['j'] = np.round(grid_df['j_raw']).astype(int) + 1
    grid_df['k'] = np.round(grid_df['k_raw']).astype(int) + 1

    return grid_df


def _get_velocities_vectorized(depths_array, iasp91_model):
    """Vectorized velocity lookup for an array of depths.

    Args:
        depths_array: NumPy array of depths (km)
        iasp91_model: DataFrame with columns [depth, vp, vs]

    Returns:
        Tuple of (vp_array, vs_array) in km/s
    """
    iasp91_depths = iasp91_model['depth'].values
    iasp91_vp = iasp91_model['vp'].values
    iasp91_vs = iasp91_model['vs'].values

    # Handle depths above MSL (clamp to 0)
    depths_clamped = np.maximum(depths_array, 0.0)

    # Use searchsorted to find layer indices for all points at once
    # searchsorted finds insertion point, we want the layer before that
    indices = np.searchsorted(iasp91_depths, depths_clamped, side='right') - 1

    # Clamp indices to valid range
    indices = np.clip(indices, 0, len(iasp91_depths) - 1)

    # Lookup velocities
    vp_array = iasp91_vp[indices]
    vs_array = iasp91_vs[indices]

    return vp_array, vs_array


def _cartesian_to_geographic_vectorized(x_km, y_km, z_km, origin_lon, origin_lat,
                                         origin_depth, invert_y_axis):
    """Vectorized conversion of Cartesian coordinates to geographic.

    Args:
        x_km, y_km, z_km: NumPy arrays of Cartesian coordinates (km)
        origin_lon, origin_lat, origin_depth: Origin point
        invert_y_axis: Y-axis direction flag

    Returns:
        Tuple of (lon_array, lat_array, depth_array)
    """
    EARTH_RADIUS_KM = 6371.0

    origin_lat_rad = np.radians(origin_lat)

    # Vectorized conversion
    y_multiplier = -1 if invert_y_axis else 1
    lat = origin_lat + y_multiplier * (y_km / EARTH_RADIUS_KM) * (180.0 / np.pi)
    lon = origin_lon + (x_km / (EARTH_RADIUS_KM * np.cos(origin_lat_rad))) * (180.0 / np.pi)
    depth = z_km + origin_depth

    return lon, lat, depth


def _assign_velocities_vectorized(grid_df, iasp91_model, origin_lon, origin_lat,
                                   origin_depth, invert_y_axis, logger):
    """Fully vectorized velocity assignment (no chunking).

    Best for grids that fit comfortably in memory.

    Args:
        grid_df: DataFrame with grid points
        iasp91_model: Velocity model DataFrame
        origin_lon, origin_lat, origin_depth: Origin parameters
        invert_y_axis: Y-axis direction flag
        logger: Logger instance

    Returns:
        DataFrame with added vp, vs, lon, lat, depth columns
    """
    logger.info("Using full vectorization (grid fits in memory)")

    # Extract columns as numpy arrays
    x_col = 'X_grid_km' if 'X_grid_km' in grid_df.columns else 'x'
    y_col = 'Y_grid_km' if 'Y_grid_km' in grid_df.columns else 'y'
    z_col = 'Z_grid_km' if 'Z_grid_km' in grid_df.columns else 'z'

    x_vals = grid_df[x_col].values
    y_vals = grid_df[y_col].values
    z_vals = grid_df[z_col].values

    # Vectorized operations on entire grid
    logger.info("Computing absolute depths...")
    absolute_depths = z_vals + origin_depth

    logger.info("Looking up velocities from IASP91 model...")
    vp_vals, vs_vals = _get_velocities_vectorized(absolute_depths, iasp91_model)

    logger.info("Converting to geographic coordinates...")
    lon_vals, lat_vals, depth_vals = _cartesian_to_geographic_vectorized(
        x_vals, y_vals, z_vals, origin_lon, origin_lat, origin_depth, invert_y_axis
    )

    # Add to DataFrame (no copy needed - assigning to full DataFrame, not a slice)
    grid_df['vp'] = vp_vals
    grid_df['vs'] = vs_vals
    grid_df['lon'] = lon_vals
    grid_df['lat'] = lat_vals
    grid_df['depth'] = depth_vals

    return grid_df


def _assign_velocities_chunked_vectorized(grid_df, iasp91_model, origin_lon, origin_lat,
                                          origin_depth, invert_y_axis, chunk_size, n_jobs, logger):
    """Vectorized processing within chunks for large grids.

    Combines parallelism with vectorization for optimal performance on large grids.

    Args:
        grid_df: DataFrame with grid points
        iasp91_model: Velocity model DataFrame
        origin_lon, origin_lat, origin_depth: Origin parameters
        invert_y_axis: Y-axis direction flag
        chunk_size: Number of points per chunk
        n_jobs: Number of parallel workers
        logger: Logger instance

    Returns:
        DataFrame with added vp, vs, lon, lat, depth columns
    """
    logger.info(f"Using chunked vectorization (large grid: {len(grid_df):,} points)")
    logger.info(f"Chunk size: {chunk_size:,}, Workers: {n_jobs if n_jobs != -1 else 'all cores'}")

    # Extract columns as numpy arrays
    x_col = 'X_grid_km' if 'X_grid_km' in grid_df.columns else 'x'
    y_col = 'Y_grid_km' if 'Y_grid_km' in grid_df.columns else 'y'
    z_col = 'Z_grid_km' if 'Z_grid_km' in grid_df.columns else 'z'

    x_vals = grid_df[x_col].values
    y_vals = grid_df[y_col].values
    z_vals = grid_df[z_col].values

    # Calculate chunks
    num_chunks = (len(grid_df) + chunk_size - 1) // chunk_size
    logger.info(f"Processing in {num_chunks} chunks...")

    def process_chunk_vectorized(start_idx, end_idx):
        """Process a chunk using vectorization."""
        # Slice arrays (no copy, just views)
        x_chunk = x_vals[start_idx:end_idx]
        y_chunk = y_vals[start_idx:end_idx]
        z_chunk = z_vals[start_idx:end_idx]

        # Vectorized operations on chunk
        absolute_depths = z_chunk + origin_depth
        vp_chunk, vs_chunk = _get_velocities_vectorized(absolute_depths, iasp91_model)
        lon_chunk, lat_chunk, depth_chunk = _cartesian_to_geographic_vectorized(
            x_chunk, y_chunk, z_chunk, origin_lon, origin_lat, origin_depth, invert_y_axis
        )

        return (start_idx, vp_chunk, vs_chunk, lon_chunk, lat_chunk, depth_chunk)

    # Process chunks in parallel
    chunk_ranges = [
        (i, min(i + chunk_size, len(grid_df)))
        for i in range(0, len(grid_df), chunk_size)
    ]

    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_chunk_vectorized)(start, end)
        for start, end in tqdm(chunk_ranges, desc="Processing chunks", unit="chunk")
    )

    # Combine results
    logger.info("Combining results...")
    vp_vals = np.empty(len(grid_df), dtype=np.float64)
    vs_vals = np.empty(len(grid_df), dtype=np.float64)
    lon_vals = np.empty(len(grid_df), dtype=np.float64)
    lat_vals = np.empty(len(grid_df), dtype=np.float64)
    depth_vals = np.empty(len(grid_df), dtype=np.float64)

    for start_idx, vp_chunk, vs_chunk, lon_chunk, lat_chunk, depth_chunk in results:
        end_idx = start_idx + len(vp_chunk)
        vp_vals[start_idx:end_idx] = vp_chunk
        vs_vals[start_idx:end_idx] = vs_chunk
        lon_vals[start_idx:end_idx] = lon_chunk
        lat_vals[start_idx:end_idx] = lat_chunk
        depth_vals[start_idx:end_idx] = depth_chunk

    # Add to DataFrame (no copy needed - assigning to full DataFrame, not a slice)
    grid_df['vp'] = vp_vals
    grid_df['vs'] = vs_vals
    grid_df['lon'] = lon_vals
    grid_df['lat'] = lat_vals
    grid_df['depth'] = depth_vals

    return grid_df


def _process_grid_chunk(chunk_rows, iasp91_model, origin_lon, origin_lat,
                        origin_depth, invert_y_axis, x_col, y_col, z_col):
    """Process a chunk of grid points to assign velocities.

    This function is designed to be called in parallel for chunks of grid points.
    DEPRECATED: Use vectorized versions instead.

    Returns:
        List of tuples: (index, vp, vs, lon, lat, depth)
    """
    results = []
    for idx, row in chunk_rows:
        x_km = row[x_col]
        y_km = row[y_col]
        z_km = row[z_col]

        # Convert relative z to absolute depth
        absolute_depth = z_km + origin_depth
        vp, vs = get_velocity_for_depth(absolute_depth, iasp91_model)

        # Convert Cartesian back to geographic coordinates
        lon, lat, depth = cartesian_to_geographic(
            x_km, y_km, z_km, origin_lon, origin_lat, origin_depth, invert_y_axis
        )

        results.append((idx, vp, vs, lon, lat, depth))

    return results


def cartesian_to_geographic(x_km: float, y_km: float, z_km: float,
                           origin_lon: float, origin_lat: float, origin_depth: float,
                           invert_y_axis: bool = True) -> Tuple[float, float, float]:
    """Convert Cartesian coordinates back to geographic coordinates.

    Args:
        x_km: X coordinate in km (East).
        y_km: Y coordinate in km (direction depends on invert_y_axis).
        z_km: Z coordinate in km (depth relative to origin).
        origin_lon: Origin longitude in degrees.
        origin_lat: Origin latitude in degrees.
        origin_depth: Origin depth in km.
        invert_y_axis: If True, positive y means south (lat decreases).
                       If False, positive y means north (lat increases).

    Returns:
        Tuple of (longitude, latitude, depth) in degrees and km.
    """
    EARTH_RADIUS_KM = 6371.0

    # Convert back to lat/lon
    origin_lat_rad = np.radians(origin_lat)

    # Reverse the transformation
    # If invert_y_axis=True: y = -1 * (lat - origin_lat) => lat = origin_lat - y
    # If invert_y_axis=False: y = (lat - origin_lat) => lat = origin_lat + y
    y_multiplier = -1 if invert_y_axis else 1
    lat = origin_lat + y_multiplier * (y_km / EARTH_RADIUS_KM) * (180.0 / np.pi)
    lon = origin_lon + (x_km / (EARTH_RADIUS_KM * np.cos(origin_lat_rad))) * (180.0 / np.pi)
    depth = z_km + origin_depth

    return lon, lat, depth


def create_velocity_model(config: DictConfig,
                          events_cart_df: pd.DataFrame,
                          stations_cart_df: pd.DataFrame) -> None:
    """Create initial velocity model on Cartesian grid and generate binaries.

    Args:
        config: Full configuration.
        events_cart_df: DataFrame with event Cartesian coordinates.
        stations_cart_df: DataFrame with station Cartesian coordinates.
    """
    logger = logging.getLogger(__name__)
    vel_config = config.initial_velocity_model

    if not vel_config.enabled:
        logger.info("Initial velocity model generation disabled")
        return

    logger.info("Creating initial velocity model...")

    # Load IASP91 model
    iasp91_model = load_iasp91_model(
        vel_config.iasp91_model_file,
        max_depth_km=vel_config.max_depth_km
    )

    # Get origin parameters
    origin_lon = vel_config.origin_lon
    origin_lat = vel_config.origin_lat
    origin_depth = vel_config.origin_depth_km

    # Get y-axis inversion setting from coordinate_transform config
    invert_y_axis = config.coordinate_transform.get('invert_y_axis', True)
    logger.info(f"Using y-axis mode: {'inverted (positive south)' if invert_y_axis else 'standard (positive north)'}")

    # Use capital case columns if available, otherwise lowercase
    x_col = 'X_grid_km' if 'X_grid_km' in events_cart_df.columns else 'x'
    y_col = 'Y_grid_km' if 'Y_grid_km' in events_cart_df.columns else 'y'
    z_col = 'Z_grid_km' if 'Z_grid_km' in events_cart_df.columns else 'z'

    # Determine grid bounds
    if vel_config.use_auto_bounds:
        logger.info("Auto-detecting grid bounds from filtered events and stations...")

        # Get grid spacing
        dx, dy, dz = vel_config.grid_spacing_km

        # Start from origin (0, 0) for X and Y
        min_x = 0.0
        min_y = 0.0

        # Auto-detect max X and Y bounds from data
        data_max_x = max(events_cart_df[x_col].max(), stations_cart_df[x_col].max())
        data_max_y = max(events_cart_df[y_col].max(), stations_cart_df[y_col].max())

        # Extend to next grid point beyond data maximum
        max_x = np.ceil(data_max_x / dx) * dx
        max_y = np.ceil(data_max_y / dy) * dy

        logger.info(f"Data max X: {data_max_x:.2f} km -> Extended to grid point: {max_x:.2f} km")
        logger.info(f"Data max Y: {data_max_y:.2f} km -> Extended to grid point: {max_y:.2f} km")

        # For Z: start from minimum data depth
        data_min_z = min(events_cart_df[z_col].min(), stations_cart_df[z_col].min())
        min_z = np.floor(data_min_z / dz) * dz  # Extend to grid point below minimum

        # For Z max: use max_depth_km if specified, otherwise use data maximum
        if vel_config.max_depth_km is not None:
            # max_depth_km is actual depth, convert to Z_grid_km
            # Z_grid_km = actual_depth - origin_depth
            max_z_from_config = vel_config.max_depth_km - origin_depth
            logger.info(f"Using max_depth_km={vel_config.max_depth_km} km (Z_grid_km={max_z_from_config:.2f} km)")
            max_z = np.ceil(max_z_from_config / dz) * dz  # Extend to next grid point
        else:
            data_max_z = max(events_cart_df[z_col].max(), stations_cart_df[z_col].max())
            max_z = np.ceil(data_max_z / dz) * dz  # Extend to next grid point
            logger.info(f"Data max Z: {data_max_z:.2f} km -> Extended to grid point: {max_z:.2f} km")

        min_coords = (min_x, min_y, min_z)
        max_coords = (max_x, max_y, max_z)
    else:
        logger.info("Using manual bounds for grid")
        min_coords = (vel_config.x_min, vel_config.y_min, vel_config.z_min)
        max_coords = (vel_config.x_max, vel_config.y_max, vel_config.z_max)

    logger.info(f"Grid bounds (Cartesian):")
    logger.info(f"  x: [{min_coords[0]:.2f}, {max_coords[0]:.2f}] km")
    logger.info(f"  y: [{min_coords[1]:.2f}, {max_coords[1]:.2f}] km")
    logger.info(f"  z: [{min_coords[2]:.2f}, {max_coords[2]:.2f}] km")

    # Create Cartesian grid
    grid_spacing = tuple(vel_config.grid_spacing_km)
    grid_df = create_cartesian_grid(min_coords, max_coords, grid_spacing)

    # Add grid indices
    grid_df = compute_grid_indices(grid_df, min_coords, grid_spacing)

    # Get parallel processing configuration
    parallel_config = config.get('parallel', {})
    n_jobs = parallel_config.get('num_workers', None)
    if n_jobs is None:
        n_jobs = -1  # Use all CPU cores

    grid_chunk_size = parallel_config.get('grid_chunk_size', 50000)

    # Hybrid vectorization: Choose approach based on grid size
    vectorize_threshold = parallel_config.get('vectorize_max_points', 100_000_000)  # 100M points

    logger.info(f"Assigning velocities from IASP91 model...")
    logger.info(f"Grid size: {len(grid_df):,} points")
    logger.info(f"Vectorization threshold: {vectorize_threshold:,} points")

    # Estimate memory requirements (rough estimate: 8 bytes per float * 8 arrays)
    estimated_memory_mb = (len(grid_df) * 8 * 8) / (1024 * 1024)
    logger.info(f"Estimated memory requirement: ~{estimated_memory_mb:.1f} MB")

    # Choose processing strategy
    if len(grid_df) < vectorize_threshold:
        # FAST PATH: Full vectorization for grids that fit in memory
        grid_df = _assign_velocities_vectorized(
            grid_df, iasp91_model, origin_lon, origin_lat,
            origin_depth, invert_y_axis, logger
        )
    else:
        # SAFE PATH: Chunked vectorization for large grids
        grid_df = _assign_velocities_chunked_vectorized(
            grid_df, iasp91_model, origin_lon, origin_lat,
            origin_depth, invert_y_axis, grid_chunk_size, n_jobs, logger
        )

    # Save velocity model (CSV and/or HDF5)
    output_format = vel_config.get('output_format', 'csv')
    logger.info(f"Saving velocity model in format: {output_format}")

    # Prepare metadata for HDF5
    metadata = {
        'grid_dimensions': {'nx': int(grid_df['i'].max()), 'ny': int(grid_df['j'].max()), 'nz': int(grid_df['k'].max())},
        'grid_spacing': {'dx': grid_spacing[0], 'dy': grid_spacing[1], 'dz': grid_spacing[2]},
        'grid_origin': {
            'x': float(grid_df['X_grid_km'].min()),
            'y': float(grid_df['Y_grid_km'].min()),
            'z': float(grid_df['Z_grid_km'].min())
        },
        'origin_lon': origin_lon,
        'origin_lat': origin_lat,
        'origin_depth_km': origin_depth,
        'transformation_method': 'iasp91_cartesian',
        'invert_y_axis': invert_y_axis,
    }

    # Prepare coarse grid config if enabled
    coarse_config = None
    hdf5_opts = vel_config.get('hdf5_options', {})
    if hdf5_opts.get('create_coarse_dataset', False) or hdf5_opts.get('write_coarse_csv', False):
        coarse_config = {
            'dx_coarse': hdf5_opts.coarse_grid.dx,
            'dy_coarse': hdf5_opts.coarse_grid.dy,
            'dz_coarse': hdf5_opts.coarse_grid.dz,
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
        from utils.velocity_io import subsample_to_coarse_grid

        output_coarse_csv = Path(hdf5_opts.get('output_coarse_csv',
                                               vel_config.output_csv.replace('.csv', '_coarse.csv')))
        ensure_directory_exists(output_coarse_csv.parent)

        logger.info(f"Creating coarse-sampled CSV: {output_coarse_csv}")

        # Extract fine grid spacing from metadata
        fine_dx = metadata['grid_spacing']['dx']
        fine_dy = metadata['grid_spacing']['dy']
        fine_dz = metadata['grid_spacing']['dz']

        # Generate coarse grid
        grid_df_coarse = subsample_to_coarse_grid(
            grid_df,
            dx_coarse=coarse_config['dx_coarse'],
            dy_coarse=coarse_config['dy_coarse'],
            dz_coarse=coarse_config['dz_coarse'],
            fine_dx=fine_dx,
            fine_dy=fine_dy,
            fine_dz=fine_dz,
            n_jobs=hdf5_opts.get('n_jobs', 1),
            show_progress=hdf5_opts.get('show_progress', True)
        )

        # Write coarse CSV
        write_velocity_file(
            grid_df_coarse,
            output_coarse_csv,
            format='csv',
            show_progress=hdf5_opts.get('show_progress', True)
        )

        coarse_size = output_coarse_csv.stat().st_size / (1024**2)
        logger.info(f"Coarse CSV written: {len(grid_df_coarse):,} points ({coarse_size:.1f}MB)")

    # Calculate grid dimensions
    nx = grid_df['i'].max()
    ny = grid_df['j'].max()
    nz = grid_df['k'].max()
    dx, dy, dz = grid_spacing

    # Count number of sources (events)
    num_sources = len(events_cart_df)

    logger.info("Velocity model created successfully")
    logger.info(f"  Grid dimensions (nx, ny, nz): {nx} x {ny} x {nz}")
    logger.info(f"  Grid spacing (dx, dy, dz): {dx:.2f} x {dy:.2f} x {dz:.2f} km")
    logger.info(f"  Total grid points: {len(grid_df):,}")
    logger.info(f"  Number of sources (filtered events): {num_sources}")
    logger.info(f"  Vp range: {grid_df['vp'].min():.3f} - "
               f"{grid_df['vp'].max():.3f} km/s")
    logger.info(f"  Vs range: {grid_df['vs'].min():.3f} - "
               f"{grid_df['vs'].max():.3f} km/s")
    logger.info(f"  Depth range: {grid_df['depth'].min():.2f} - "
               f"{grid_df['depth'].max():.2f} km")

    # Generate binary files
    _generate_binary_files(config, grid_df, min_coords, grid_spacing)

    # Print final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("VELOCITY MODEL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Number of sources (filtered events): {num_sources}")
    logger.info(f"Grid dimensions (nx, ny, nz): {nx} x {ny} x {nz}")
    logger.info(f"Grid spacing (dx, dy, dz): {dx:.2f} km x {dy:.2f} km x {dz:.2f} km")
    logger.info(f"Total grid points: {len(grid_df):,}")
    logger.info("=" * 60)


def _fill_grid_vectorized(grid_df, vel_col, nx, ny, nz, logger):
    """Fill 3D grid array using vectorization (no chunking).

    Args:
        grid_df: DataFrame with velocity data
        vel_col: Column name ('vp' or 'vs')
        nx, ny, nz: Grid dimensions
        logger: Logger instance

    Returns:
        3D numpy array in Fortran order
    """
    logger.info(f"Using full vectorization to fill {vel_col.upper()} grid")

    # Create 3D grid array in Fortran-order
    grid_3d = np.zeros((nz, ny, nx), dtype=np.float32, order='F')

    # Extract indices and velocities as numpy arrays
    i_vals = grid_df['i'].values - 1  # Convert to 0-based
    j_vals = grid_df['j'].values - 1
    k_vals = grid_df['k'].values - 1
    vel_vals = grid_df[vel_col].values * 1000  # Convert to m/s

    # Create mask for valid indices
    valid_mask = (
        (i_vals >= 0) & (i_vals < nx) &
        (j_vals >= 0) & (j_vals < ny) &
        (k_vals >= 0) & (k_vals < nz)
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


def _fill_grid_chunked_vectorized(grid_df, vel_col, nx, ny, nz, chunk_size, n_jobs, logger):
    """Fill 3D grid array using chunked vectorization.

    Args:
        grid_df: DataFrame with velocity data
        vel_col: Column name ('vp' or 'vs')
        nx, ny, nz: Grid dimensions
        chunk_size: Points per chunk
        n_jobs: Number of workers
        logger: Logger instance

    Returns:
        3D numpy array in Fortran order
    """
    logger.info(f"Using chunked vectorization to fill {vel_col.upper()} grid")
    logger.info(f"Chunk size: {chunk_size:,}, Workers: {n_jobs if n_jobs != -1 else 'all cores'}")

    # Create 3D grid array in Fortran-order
    grid_3d = np.zeros((nz, ny, nx), dtype=np.float32, order='F')

    # Extract arrays once
    i_vals = grid_df['i'].values - 1
    j_vals = grid_df['j'].values - 1
    k_vals = grid_df['k'].values - 1
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
            (i_chunk >= 0) & (i_chunk < nx) &
            (j_chunk >= 0) & (j_chunk < ny) &
            (k_chunk >= 0) & (k_chunk < nz)
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


def _fill_grid_chunk(chunk_rows, vel_col, nx, ny, nz):
    """Fill a chunk of grid points into the velocity array.

    DEPRECATED: Use vectorized versions instead.

    Returns:
        List of tuples: (k, j, i, velocity_value)
    """
    results = []
    for _, row in chunk_rows:
        i = int(row['i']) - 1  # Convert to 0-based
        j = int(row['j']) - 1
        k = int(row['k']) - 1
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
            velocity_ms = row[vel_col] * 1000  # Convert to m/s
            results.append((k, j, i, velocity_ms))
    return results


def _generate_binary_files(config: DictConfig, grid_df: pd.DataFrame,
                           min_coords: Tuple[float, float, float],
                           grid_spacing: Tuple[float, float, float]) -> None:
    """Generate binary velocity files from grid DataFrame.

    Args:
        config: Configuration.
        grid_df: DataFrame with velocity grid data.
        min_coords: Grid origin coordinates.
        grid_spacing: Grid spacing.
    """
    logger = logging.getLogger(__name__)
    vel_config = config.initial_velocity_model

    logger.info("Creating velocity binary files...")

    # Get grid dimensions
    nx = grid_df['i'].max()
    ny = grid_df['j'].max()
    nz = grid_df['k'].max()

    logger.info(f"Grid dimensions: {nx} x {ny} x {nz}")

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
        grid_3d = _fill_grid_vectorized(grid_df, vel_col, nx, ny, nz, logger)

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
        grid: 3D velocity array (nz, ny, nx) in Fortran-order.
    """
    logger = logging.getLogger(__name__)

    ensure_directory_exists(filepath.parent)

    nz, ny, nx = grid.shape

    # Write in Fortran-order to match LATTE's column-major format
    grid.astype(np.float32).flatten('F').tofile(str(filepath))

    # Verify the written file
    grid_check = np.fromfile(str(filepath), dtype=np.float32).reshape((nz, ny, nx), order='F')

    logger.info(f"Written binary file: {filepath}")
    logger.info(f"  File size: {filepath.stat().st_size} bytes")
    logger.info(f"  Shape: {grid.shape}")
    logger.info(f"  Value range: {grid.min():.1f} - {grid.max():.1f} m/s")
    logger.info(f"  Verification: {np.allclose(grid, grid_check)}")
