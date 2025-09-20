"""
Coordinate transformation functions for LATTE preprocessing.
Functions for converting geodetic to Cartesian coordinates using exact or flattening methods.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
from omegaconf import DictConfig

from utils.io_utils import ensure_directory_exists

# Check for optional dependencies
try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

# Constants
EARTH_RADIUS_KM = 6371.0
GEODETIC_CRS = "EPSG:4326"
ECEF_CRS = "EPSG:4978"


# ============================================================================
# I/O Functions (preserved from your original scripts)
# ============================================================================

def load_station_data(file_path: str) -> pd.DataFrame:
    """Load seismic station coordinates from a 'stacoords' file (6 cols)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Station data file not found at: {file_path}")

    stations = []
    with open(file_path, "r") as f:
        for ln, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 6:
                print(f"[stations] Skipping line {ln}: expected 6 columns, got {len(parts)}")
                continue
            station = parts[-3]
            try:
                lat = float(parts[-2])
                lon = float(parts[-1])
                # elevation (m, positive up) -> depth (km, positive down)
                depth = -float(parts[-4]) / 1000.0
            except ValueError:
                print(f"[stations] Skipping line {ln}: parse error.")
                continue
            stations.append({"station": station, "lat": lat, "lon": lon, "depth": depth})
    return pd.DataFrame(stations)


def load_event_data(file_path: str) -> pd.DataFrame:
    """Load seismic event data from a 'phasePickings.txt' file (9 cols)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Event data file not found at: {file_path}")

    events = []
    with open(file_path, "r") as f:
        for ln, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            try:
                events.append({
                    "event_id": parts[8],
                    "lat": float(parts[5]),
                    "lon": float(parts[6]),
                    "depth": float(parts[7])
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(events)


def load_corrected_event_data(file_path: str) -> pd.DataFrame:
    """
    Load corrected seismic event data from a .dat file after source correction.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Corrected event data file not found at: {file_path}")
    
    events = []
    
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
                
            # Check if this is an event line (10 columns)
            if len(parts) == 10:
                try:
                    year = int(parts[0])
                    julian_day = int(parts[1])
                    hour = int(parts[2])
                    minute = int(parts[3])
                    second = float(parts[4])
                    lat = float(parts[5])
                    lon = float(parts[6])
                    depth = float(parts[7])
                    event_id = parts[8]
                    weighted_rms = float(parts[9])
                    
                    events.append({
                        "event_id": event_id,
                        "year": year,
                        "julian_day": julian_day,
                        "hour": hour,
                        "minute": minute,
                        "second": second,
                        "lat": lat,
                        "lon": lon,
                        "depth": depth,
                        "weighted_rms": weighted_rms
                    })
                    
                except (ValueError, IndexError):
                    continue
    
    return pd.DataFrame(events)


def load_velocity_data(file_path: str) -> pd.DataFrame:
    """Load velocity grid coordinates from an initial velocity file (5 cols)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Velocity data file not found at: {file_path}")

    grids = []
    with open(file_path, "r") as f:
        for ln, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    grids.append({
                        "lon": float(parts[0]),
                        "lat": float(parts[1]),
                        "depth": float(parts[2]),
                        "vp": float(parts[3]),
                        "vs": float(parts[4])
                    })
                except ValueError:
                    print(f"[velocity] Skipping line {ln}: parse error.")
                    continue
            else:
                print(f"[velocity] Skipping line {ln}: expected 5 columns, got {len(parts)}")
                continue
    return pd.DataFrame(grids)


# ============================================================================
# Exact Transformation Functions (your spherical_to_cartesian_exact.py)
# ============================================================================

def _convert_to_ecef(df: pd.DataFrame) -> pd.DataFrame:
    """Add ECEF X/Y/Z (m) from lon/lat/depth(km)."""
    if not HAS_PYPROJ:
        raise ImportError("pyproj is required for exact coordinate transformations. Install with: pip install pyproj")
    
    transformer = Transformer.from_crs(GEODETIC_CRS, ECEF_CRS, always_xy=True)
    height_meters = -df["depth"].values * 1000.0
    X, Y, Z = transformer.transform(df["lon"].values, df["lat"].values, height_meters)
    df_out = df.copy()
    df_out["X_ecef"], df_out["Y_ecef"], df_out["Z_ecef"] = X, Y, Z
    return df_out


def _shift_ecef_origin(df: pd.DataFrame, origin_lon: float, origin_lat: float, origin_depth_km: float) -> pd.DataFrame:
    """Translate ECEF to be relative to a TRANSLATION origin (m)."""
    if not HAS_PYPROJ:
        raise ImportError("pyproj is required for exact coordinate transformations")
    
    transformer = Transformer.from_crs(GEODETIC_CRS, ECEF_CRS, always_xy=True)
    origin_height_m = -origin_depth_km * 1000.0
    X0, Y0, Z0 = transformer.transform(origin_lon, origin_lat, origin_height_m)
    df_out = df.copy()
    df_out["X_local"] = df_out["X_ecef"] - X0
    df_out["Y_local"] = df_out["Y_ecef"] - Y0
    df_out["Z_local"] = df_out["Z_ecef"] - Z0
    return df_out


def _rotate_to_enu(df: pd.DataFrame, origin_lon: float, origin_lat: float) -> pd.DataFrame:
    """Add ENU columns (m). Rotates around the TRANSLATION origin."""
    lon_rad = np.radians(origin_lon)
    lat_rad = np.radians(origin_lat)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    R = np.array([
        [-sin_lon,             cos_lon,              0.0],
        [-sin_lat * cos_lon,  -sin_lat * sin_lon,    cos_lat],
        [ cos_lat * cos_lon,   cos_lat * sin_lon,    sin_lat],
    ])
    local = df[["X_local", "Y_local", "Z_local"]].values
    enu = (R @ local.T).T
    out = df.copy()
    out["E"], out["N"], out["U"] = enu[:, 0], enu[:, 1], enu[:, 2]
    return out


def _rotate_to_geocentric_normal(df: pd.DataFrame, rotation_lon: float, rotation_lat: float, rotation_depth_km: float) -> pd.DataFrame:
    """Rotate so Z_sym aligns with the geocentric radial of a ROTATION center."""
    if not HAS_PYPROJ:
        raise ImportError("pyproj is required for exact coordinate transformations")
    
    transformer = Transformer.from_crs(GEODETIC_CRS, ECEF_CRS, always_xy=True)
    origin_height_m = -rotation_depth_km * 1000.0
    X0, Y0, Z0 = transformer.transform(rotation_lon, rotation_lat, origin_height_m)
    up = np.array([X0, Y0, Z0])
    up /= np.linalg.norm(up)
    lon_rad = np.radians(rotation_lon)
    east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0.0])
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    north /= np.linalg.norm(north)
    R = np.vstack([east, north, up])
    local = df[["X_local", "Y_local", "Z_local"]].values
    sym = (R @ local.T).T
    out = df.copy()
    out["X_sym"], out["Y_sym"], out["Z_sym"] = sym[:, 0], sym[:, 1], sym[:, 2]
    return out


def exact_geodetic_to_local_cartesian(
    df: pd.DataFrame,
    origin_lon: float,
    origin_lat: float,
    origin_depth_km: float,
    rotate_to_enu: bool = True,
    align_to_geocentric_normal: bool = False,
    rotation_center_lon: Optional[float] = None,
    rotation_center_lat: Optional[float] = None
) -> pd.DataFrame:
    """Convert (lon, lat, depth[km]) -> local systems using exact transformations."""
    df_ecef = _convert_to_ecef(df)
    df_local = _shift_ecef_origin(df_ecef, origin_lon, origin_lat, origin_depth_km)
    df_out = df_local.copy()
    
    if rotate_to_enu:
        enu = _rotate_to_enu(df_local, origin_lon, origin_lat)
        for c in ("E", "N", "U"):
            df_out[c] = enu[c]
            df_out[f"{c}_km"] = enu[c] / 1000.0
    
    if align_to_geocentric_normal:
        rot_lon = rotation_center_lon if rotation_center_lon is not None else origin_lon
        rot_lat = rotation_center_lat if rotation_center_lat is not None else origin_lat
        sym = _rotate_to_geocentric_normal(df_local, rot_lon, rot_lat, origin_depth_km)
        for c in ("X_sym", "Y_sym", "Z_sym"):
            df_out[c] = sym[c]
            df_out[f"{c}_km"] = sym[c] / 1000.0
        df_out["Depth_sym_km"] = -df_out["Z_sym_km"]
    
    return df_out


# ============================================================================
# Flattening Transformation Functions (your spherical_to_cartesian_flattening.py)
# ============================================================================

def simple_flattening_conversion(df: pd.DataFrame, origin_lon: float, origin_lat: float, origin_depth: float) -> pd.DataFrame:
    """Simple geometric conversion from geodetic (lon, lat) to local Cartesian (x, y) in km."""
    df_out = df.copy()
    lon_rad = np.radians(df_out['lon'])
    lat_rad = np.radians(df_out['lat'])
    origin_lon_rad = np.radians(origin_lon)
    origin_lat_rad = np.radians(origin_lat)
    df_out['x_km'] = EARTH_RADIUS_KM * (lon_rad - origin_lon_rad) * np.cos(origin_lat_rad)
    df_out['y_km'] = -1 * (EARTH_RADIUS_KM * (lat_rad - origin_lat_rad))  # to make the axes positive
    df_out['z_km'] = df_out['depth'] - origin_depth
    return df_out


def chapman_flattening_conversion(df: pd.DataFrame, origin_lon: float, origin_lat: float, origin_depth: float) -> pd.DataFrame:
    """Chapman (1973) Earth-flattening transformation for seismic wave studies."""
    df_out = df.copy()
    
    # First do horizontal coordinate transformation (same as simple method)
    lon_rad = np.radians(df_out['lon'])
    lat_rad = np.radians(df_out['lat'])
    origin_lon_rad = np.radians(origin_lon)
    origin_lat_rad = np.radians(origin_lat)
    df_out['x_km'] = EARTH_RADIUS_KM * (lon_rad - origin_lon_rad) * np.cos(origin_lat_rad)
    df_out['y_km'] = -1 * (EARTH_RADIUS_KM * (lat_rad - origin_lat_rad))
    
    # Chapman depth transformation: z_flat = r * ln(r/(r-z))
    r = EARTH_RADIUS_KM
    # Handle the case where depth might be negative (above sea level)
    depth_positive = np.maximum(df_out['depth'], 0.001)  # Avoid log(inf)
    df_out['z_km'] = r * np.log(r / (r - depth_positive)) - origin_depth
    
    # Chapman velocity transformations: v_flat = exp(z/r) * v
    if 'vp' in df_out.columns:
        df_out['vp_flat'] = np.exp(depth_positive / r) * df_out['vp']
        # Keep original velocities for reference
        df_out['vp_original'] = df_out['vp']
        df_out['vp'] = df_out['vp_flat']  # Replace with flattened velocities
    
    if 'vs' in df_out.columns:
        df_out['vs_flat'] = np.exp(depth_positive / r) * df_out['vs']
        # Keep original velocities for reference
        df_out['vs_original'] = df_out['vs']
        df_out['vs'] = df_out['vs_flat']  # Replace with flattened velocities
    
    return df_out


def flattening_conversion(df: pd.DataFrame, origin_lon: float, origin_lat: float, origin_depth: float, use_chapman: bool = False) -> pd.DataFrame:
    """
    Main conversion function that switches between methods based on use_chapman flag.
    """
    if use_chapman:
        return chapman_flattening_conversion(df, origin_lon, origin_lat, origin_depth)
    else:
        return simple_flattening_conversion(df, origin_lon, origin_lat, origin_depth)


# ============================================================================
# Grid Processing Functions
# ============================================================================

def process_dataframe_for_exact_grid(
    df: pd.DataFrame,
    global_origin_mins: Tuple[float, float, float],
    grid_spacing: Tuple[float, float, float],
    use_geocentric_normal: bool = False,
) -> pd.DataFrame:
    """Process coordinates for exact transformation grid mapping."""
    min_x, min_y, min_z = global_origin_mins
    dx_km, dy_km, dz_km = grid_spacing
    out = df.copy()

    # Define the final grid coordinate system
    if use_geocentric_normal:
        x_coords = out["X_sym_km"]
        y_coords = out["Y_sym_km"]
        z_coords = out["Depth_sym_km"]  # Already positive down
    else:  # Default to ENU
        x_coords = out["E_km"]
        y_coords = -out["N_km"]  # Y-axis points South
        z_coords = -out["U_km"]  # Z-axis points Down
    
    out["X_grid_km"] = x_coords - min_x
    out["Y_grid_km"] = y_coords - min_y
    out["Z_grid_km"] = z_coords - min_z

    # Calculate raw (floating-point) indices from the shifted coordinates
    out["i_raw"] = out["X_grid_km"] / dx_km
    out["j_raw"] = out["Y_grid_km"] / dy_km
    out["k_raw"] = out["Z_grid_km"] / dz_km

    # Calculate final integer indices (1-based for Fortran compatibility)
    out["i"] = np.round(out["i_raw"]).astype(int) + 1
    out["j"] = np.round(out["j_raw"]).astype(int) + 1
    out["k"] = np.round(out["k_raw"]).astype(int) + 1
    return out


def process_dataframe_for_flattening_grid(
    df: pd.DataFrame,
    grid_spacing: Tuple[float, float, float],
) -> pd.DataFrame:
    """Process coordinates for flattening transformation grid mapping."""
    dx_km, dy_km, dz_km = grid_spacing
    out = df.copy()

    # The (x_km, y_km, z_km) are now the final grid coordinates
    out["X_grid_km"] = out["x_km"]
    out["Y_grid_km"] = out["y_km"]
    out["Z_grid_km"] = out["z_km"]

    # Calculate raw (floating-point) indices
    out["i_raw"] = out["X_grid_km"] / dx_km
    out["j_raw"] = out["Y_grid_km"] / dy_km
    out["k_raw"] = out["Z_grid_km"] / dz_km

    # Calculate final integer indices (1-based for Fortran compatibility)
    out["i"] = np.round(out["i_raw"]).astype(int) + 1
    out["j"] = np.round(out["j_raw"]).astype(int) + 1
    out["k"] = np.round(out["k_raw"]).astype(int) + 1
    return out


def _global_grid_mins(dfs: List[pd.DataFrame], use_geocentric_normal: bool) -> Tuple[float, float, float]:
    """Compute consistent global minima from a list of dataframes to define the grid origin."""
    if use_geocentric_normal:
        all_x = pd.concat([df["X_sym_km"] for df in dfs])
        all_y = pd.concat([df["Y_sym_km"] for df in dfs])
        all_z = pd.concat([df["Depth_sym_km"] for df in dfs])  # Positive down
    else:  # Default to ENU
        all_x = pd.concat([df["E_km"] for df in dfs])
        all_y = pd.concat([-df["N_km"] for df in dfs])  # South is positive Y
        all_z = pd.concat([-df["U_km"] for df in dfs])  # Down is positive Z
    return (all_x.min(), all_y.min(), all_z.min())


def _grid_extents_and_dims(dfs: List[pd.DataFrame]) -> Tuple[int, int, int]:
    """Compute nx, ny, nz from the final grid indices of all dataframes."""
    all_i = pd.concat([df["i"] for df in dfs])
    all_j = pd.concat([df["j"] for df in dfs])
    all_k = pd.concat([df["k"] for df in dfs])

    # Grid dimensions are calculated from the full range of final integer indices
    nx = (all_i.max() - all_i.min()) + 1
    ny = (all_j.max() - all_j.min()) + 1
    nz = (all_k.max() - all_k.min()) + 1
    return nx, ny, nz


# ============================================================================
# Main Coordinate Transformation Function
# ============================================================================

def transform_coordinates(config: DictConfig) -> None:
    """
    Main function to transform coordinates from geodetic to Cartesian.
    Supports both exact and flattening transformations.
    """
    print("=" * 60)
    print("COORDINATE TRANSFORMATION")
    print("=" * 60)
    
    # Load data files
    print(f"Loading station data from: {config.station_file}")
    stations_df = load_station_data(config.station_file)
    
    # Choose between original or corrected event data
    if config.use_corrected_phase_pickings:
        print(f"Loading corrected event data from: {config.corrected_event_file}")
        events_df = load_corrected_event_data(config.corrected_event_file)
        event_file_suffix = "_corrected"
    else:
        print(f"Loading original event data from: {config.event_file}")
        events_df = load_event_data(config.event_file)
        event_file_suffix = ""
    
    print(f"Loading velocity data from: {config.velocity_file}")
    velocity_df = load_velocity_data(config.velocity_file)
    
    print(f"Loaded {len(events_df)} events, {len(stations_df)} stations, {len(velocity_df)} velocity points")
    
    # Setup output directory
    ensure_directory_exists(config.output_dir)
    
    # Transform based on method
    if config.transformation_method == "exact":
        print(f"\nProcessing coordinates with EXACT transformation...")
        if not HAS_PYPROJ:
            raise ImportError("pyproj is required for exact transformations. Install with: pip install pyproj")
        
        # Calculate rotation center from min/max extent of ALL data
        print("Calculating geographic midpoint of all data for vertical alignment...")
        combined_coords = pd.concat([
            stations_df[['lon', 'lat']],
            events_df[['lon', 'lat']],
            velocity_df[['lon', 'lat']]
        ])
        min_lon, max_lon = combined_coords['lon'].min(), combined_coords['lon'].max()
        min_lat, max_lat = combined_coords['lat'].min(), combined_coords['lat'].max()
        center_lon = (min_lon + max_lon) / 2.0
        center_lat = (min_lat + max_lat) / 2.0
        print(f"Data Extents: Lon ({min_lon:.4f}, {max_lon:.4f}), Lat ({min_lat:.4f}, {max_lat:.4f})")
        print(f"Rotation Center (Midpoint): Lon={center_lon:.4f}, Lat={center_lat:.4f}")
        print(f"Translation Origin: Lon={config.origin_lon}, Lat={config.origin_lat}")
        
        # Transform all dataframes to local Cartesian coordinates
        common_kwargs = {
            "origin_lon": config.origin_lon, 
            "origin_lat": config.origin_lat, 
            "origin_depth_km": config.origin_depth_km,
            "rotate_to_enu": config.exact_transform.rotate_to_enu, 
            "align_to_geocentric_normal": config.exact_transform.align_to_geocentric_normal,
            "rotation_center_lon": center_lon, 
            "rotation_center_lat": center_lat,
        }
        stations_tf = exact_geodetic_to_local_cartesian(df=stations_df, **common_kwargs)
        events_tf = exact_geodetic_to_local_cartesian(df=events_df, **common_kwargs)
        velocity_tf = exact_geodetic_to_local_cartesian(df=velocity_df, **common_kwargs)
        
        # Find global minima and process for grid
        transformed_dfs = [stations_tf, events_tf, velocity_tf]
        global_origin_mins = _global_grid_mins(transformed_dfs, config.exact_transform.align_to_geocentric_normal)
        grid_spacing_km = tuple(config.grid_spacing_km)
        print(f"Global origin minimums (X, Y, Z in km): {global_origin_mins}")
        
        # Map all dataframes to the final grid
        stations_processed_df = process_dataframe_for_exact_grid(
            stations_tf, global_origin_mins, grid_spacing_km, config.exact_transform.align_to_geocentric_normal
        )
        events_processed_df = process_dataframe_for_exact_grid(
            events_tf, global_origin_mins, grid_spacing_km, config.exact_transform.align_to_geocentric_normal
        )
        velocity_processed_df = process_dataframe_for_exact_grid(
            velocity_tf, global_origin_mins, grid_spacing_km, config.exact_transform.align_to_geocentric_normal
        )
        
        method_suffix = "_exact"
        
    elif config.transformation_method == "flattening":
        print(f"\nProcessing coordinates with FLATTENING transformation...")
        method_name = "Chapman (1973)" if config.use_chapman_flattening else "Simple geometric"
        print(f"Method: {method_name}")
        print(f"Origin: Lon={config.origin_lon}, Lat={config.origin_lat}, Depth={config.origin_depth_km}")
        
        # Transform using flattening method
        stations_tf = flattening_conversion(stations_df, config.origin_lon, config.origin_lat, config.origin_depth_km, config.use_chapman_flattening)
        events_tf = flattening_conversion(events_df, config.origin_lon, config.origin_lat, config.origin_depth_km, config.use_chapman_flattening)
        velocity_tf = flattening_conversion(velocity_df, config.origin_lon, config.origin_lat, config.origin_depth_km, config.use_chapman_flattening)
        
        # Process for grid
        grid_spacing_km = tuple(config.grid_spacing_km)
        stations_processed_df = process_dataframe_for_flattening_grid(stations_tf, grid_spacing_km)
        events_processed_df = process_dataframe_for_flattening_grid(events_tf, grid_spacing_km)
        velocity_processed_df = process_dataframe_for_flattening_grid(velocity_tf, grid_spacing_km)
        
        method_suffix = "_chapman" if config.use_chapman_flattening else ""
        
    else:
        raise ValueError(f"Unknown transformation method: {config.transformation_method}")
    
    # Calculate final grid dimensions
    processed_dfs = [stations_processed_df, events_processed_df, velocity_processed_df]
    nx, ny, nz = _grid_extents_and_dims(processed_dfs)
    
    # Report results
    print("\n--- Processing Complete ---")
    print(f"Transformation method: {config.transformation_method}")
    if config.transformation_method == "flattening":
        print(f"Flattening type: {'Chapman (1973)' if config.use_chapman_flattening else 'Simple geometric'}")
    print(f"Event data source: {'Corrected (.dat)' if config.use_corrected_phase_pickings else 'Original (.txt)'}")
    
    # Show sample data
    station_cols_to_show = ["station", "X_grid_km", "Y_grid_km", "Z_grid_km", "i", "j", "k"]
    print("\nSample of Processed Station Data:")
    print(stations_processed_df[station_cols_to_show].head())
    
    event_cols_to_show = ["event_id", "X_grid_km", "Y_grid_km", "Z_grid_km", "i", "j", "k"]
    if config.use_corrected_phase_pickings and 'weighted_rms' in events_processed_df.columns:
        event_cols_to_show.append("weighted_rms")
    available_event_cols = [col for col in event_cols_to_show if col in events_processed_df.columns]
    print("\nSample of Processed Event Data:")
    print(events_processed_df[available_event_cols].head())
    
    # Show velocity data columns based on method
    base_cols = ["lon", "lat", "depth", "X_grid_km", "Y_grid_km", "Z_grid_km", "i", "j", "k"]
    if config.transformation_method == "flattening" and config.use_chapman_flattening and 'vp_flat' in velocity_processed_df.columns:
        cols_to_show = base_cols + ["vp_original", "vp_flat", "vs_original", "vs_flat"]
    else:
        cols_to_show = base_cols + (["vp", "vs"] if 'vp' in velocity_processed_df.columns else [])
    available_cols = [col for col in cols_to_show if col in velocity_processed_df.columns]
    print("\nSample of Processed Velocity Data:")
    print(velocity_processed_df[available_cols].head())
    
    # Save results
    print(f"\nSaving results to '{config.output_dir}':")
    stations_processed_df.to_csv(os.path.join(config.output_dir, f"stations_cartesian{method_suffix}.csv"), index=False)
    events_processed_df.to_csv(os.path.join(config.output_dir, f"events_cartesian{method_suffix}{event_file_suffix}.csv"), index=False)
    velocity_processed_df.to_csv(os.path.join(config.output_dir, f"velocity_cartesian{method_suffix}.csv"), index=False)
    
    print(f"  - stations_cartesian{method_suffix}.csv")
    print(f"  - events_cartesian{method_suffix}{event_file_suffix}.csv")
    print(f"  - velocity_cartesian{method_suffix}.csv")
    
    print(f"\nGrid Dimensions: nx={nx}, ny={ny}, nz={nz}")
    print(f"Grid spacing: dx={config.grid_spacing_km[0]} km, dy={config.grid_spacing_km[1]} km, dz={config.grid_spacing_km[2]} km")
    
    if config.transformation_method == "flattening" and config.use_chapman_flattening:
        print("\nNOTE: Chapman transformation applied to velocities.")
        print("Original velocities preserved as 'vp_original' and 'vs_original' columns.")
        
    if config.use_corrected_phase_pickings:
        print("\nNOTE: Using corrected phase pickings data with weighted RMS values.")
    
    print("Coordinate transformation completed successfully.")