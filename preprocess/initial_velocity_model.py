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

    # Assign velocities based on depth (z coordinate + origin_depth)
    logger.info("Assigning velocities from IASP91 model...")
    vp_values = []
    vs_values = []
    lon_values = []
    lat_values = []
    depth_values = []

    # Use capital case columns
    x_col = 'X_grid_km' if 'X_grid_km' in grid_df.columns else 'x'
    y_col = 'Y_grid_km' if 'Y_grid_km' in grid_df.columns else 'y'
    z_col = 'Z_grid_km' if 'Z_grid_km' in grid_df.columns else 'z'

    for _, row in grid_df.iterrows():
        x_km = row[x_col]
        y_km = row[y_col]
        z_km = row[z_col]

        # Convert relative z to absolute depth
        absolute_depth = z_km + origin_depth
        vp, vs = get_velocity_for_depth(absolute_depth, iasp91_model)
        vp_values.append(vp)
        vs_values.append(vs)

        # Convert Cartesian back to geographic coordinates
        lon, lat, depth = cartesian_to_geographic(x_km, y_km, z_km, origin_lon, origin_lat, origin_depth, invert_y_axis)
        lon_values.append(lon)
        lat_values.append(lat)
        depth_values.append(depth)

    grid_df['vp'] = vp_values
    grid_df['vs'] = vs_values
    grid_df['lon'] = lon_values
    grid_df['lat'] = lat_values
    grid_df['depth'] = depth_values

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
    if hdf5_opts.get('create_coarse_dataset', False):
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

        # Create 3D grid array in Fortran-order to match LATTE
        vel_col = 'vp' if phase == 'p' else 'vs'
        grid_3d = np.zeros((nz, ny, nx), dtype=np.float32, order='F')

        # Fill grid (using 1-based indices)
        for _, row in grid_df.iterrows():
            i = int(row['i']) - 1  # Convert to 0-based
            j = int(row['j']) - 1
            k = int(row['k']) - 1
            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                grid_3d[k, j, i] = row[vel_col] * 1000  # Convert to m/s, index as [depth, y, x]

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
