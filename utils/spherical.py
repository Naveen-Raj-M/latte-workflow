"""
Spherical-Earth geometry of the workflow, identical to that of the spherical LATTE branch
(``src/module_spherical.f90``).

The Earth is a sphere of radius ``earth_radius_km`` (default 6371 km). A point given by its
longitude (degrees), WGS84 geodetic latitude (degrees) and depth below mean sea level (km)
lies on the sphere at

    longitude  lambda = lon
    latitude   theta  = geocentric latitude of the geodetic latitude lat
    radius     r      = earth_radius_km - depth

where the geodetic-to-geocentric relation is the exact one on the WGS84 ellipsoid surface,
``tan(theta) = (1 - e^2) tan(lat)``. The same relation is used at every depth, so a column of
constant (lon, lat) is a radial line of the sphere. Every distance and angle below is the exact
one of this sphere.

The model grid is regular in longitude (degrees), geodetic latitude (degrees) and depth (km).
Its keys are ``nlon, nlat, ndepth, dlon, dlat, ddepth, olon, olat, odepth``; ``olon`` and
``olat`` are the west and south edges and ``odepth`` the top. LATTE uses the same grid with
lengths in metres.

All functions accept scalars or numpy arrays (broadcast against each other).
"""

from typing import Dict, List, Tuple

import numpy as np

# WGS84 ellipsoid, used only to convert geodetic latitude
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

#: Default radius of the spherical Earth, km (the LATTE default, 6371000 m)
EARTH_RADIUS_KM = 6371.0

#: Grid metadata keys, in the order lon, lat, depth for each kind
GRID_KEYS = ('nlon', 'nlat', 'ndepth', 'dlon', 'dlat', 'ddepth', 'olon', 'olat', 'odepth')

#: Grid axes (velocity table columns), in the order of the grid keys
GRID_AXES = ('lon', 'lat', 'depth')


# ---------------------------------------------------------------------------
# Latitude
# ---------------------------------------------------------------------------

def geodetic_to_geocentric(lat_deg):
    """Geocentric latitude (radians) of a WGS84 geodetic latitude (degrees)."""
    p = np.radians(lat_deg)
    return np.arctan2((1.0 - WGS84_E2) * np.sin(p), np.cos(p))


def geocentric_to_geodetic(theta_rad):
    """WGS84 geodetic latitude (degrees) of a geocentric latitude (radians)."""
    return np.degrees(np.arctan2(np.sin(theta_rad), (1.0 - WGS84_E2) * np.cos(theta_rad)))


def dgeocentric_dgeodetic(lat_deg):
    """Derivative of the geocentric latitude with respect to the geodetic latitude
    (both in radians), at a geodetic latitude in degrees."""
    p = np.radians(lat_deg)
    return (1.0 - WGS84_E2) / (np.cos(p) ** 2 + (1.0 - WGS84_E2) ** 2 * np.sin(p) ** 2)


# ---------------------------------------------------------------------------
# Positions and distances
# ---------------------------------------------------------------------------

def to_ecef(lon_deg, lat_deg, depth_km, earth_radius_km: float = EARTH_RADIUS_KM) -> np.ndarray:
    """Earth-centred Cartesian position (km) of (lon, geodetic lat, depth below MSL in km);
    the last axis of the result is x, y, z."""
    lon_deg, lat_deg, depth_km = np.broadcast_arrays(
        np.asarray(lon_deg, dtype=float), np.asarray(lat_deg, dtype=float),
        np.asarray(depth_km, dtype=float))
    r = earth_radius_km - depth_km
    theta = geodetic_to_geocentric(lat_deg)
    lam = np.radians(lon_deg)
    return np.stack([r * np.cos(theta) * np.cos(lam),
                     r * np.cos(theta) * np.sin(lam),
                     r * np.sin(theta)], axis=-1)


def unit_vector(lon_deg, lat_deg) -> np.ndarray:
    """Unit vector of the radial line through (lon, geodetic lat); last axis x, y, z."""
    return to_ecef(lon_deg, lat_deg, 0.0, earth_radius_km=1.0)


def hypocentral_distance_km(lon1, lat1, depth1, lon2, lat2, depth2,
                            earth_radius_km: float = EARTH_RADIUS_KM):
    """Straight-line distance (km) between two points, as LATTE measures offsets."""
    d = to_ecef(lon1, lat1, depth1, earth_radius_km) - to_ecef(lon2, lat2, depth2, earth_radius_km)
    return np.linalg.norm(d, axis=-1)


def central_angle(lon1, lat1, lon2, lat2):
    """Angle (radians) between the radial lines through two points, from
    ``atan2(|u1 x u2|, u1 . u2)``, accurate at all distances."""
    u1 = unit_vector(lon1, lat1)
    u2 = unit_vector(lon2, lat2)
    u1, u2 = np.broadcast_arrays(u1, u2)
    return np.arctan2(np.linalg.norm(np.cross(u1, u2), axis=-1), np.sum(u1 * u2, axis=-1))


def epicentral_distance_km(lon1, lat1, lon2, lat2, earth_radius_km: float = EARTH_RADIUS_KM):
    """Great-circle distance (km) at the Earth's surface between the radial lines through
    two points."""
    return earth_radius_km * central_angle(lon1, lat1, lon2, lat2)


def local_basis(lon_deg, lat_deg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit vectors east, north and down at (lon, geodetic lat); last axis x, y, z."""
    lon_deg, lat_deg = np.broadcast_arrays(np.asarray(lon_deg, dtype=float),
                                           np.asarray(lat_deg, dtype=float))
    theta = geodetic_to_geocentric(lat_deg)
    lam = np.radians(lon_deg)
    zero = np.zeros_like(lam)
    east = np.stack([-np.sin(lam), np.cos(lam), zero], axis=-1)
    north = np.stack([-np.sin(theta) * np.cos(lam), -np.sin(theta) * np.sin(lam), np.cos(theta)], axis=-1)
    down = -np.stack([np.cos(theta) * np.cos(lam), np.cos(theta) * np.sin(lam), np.sin(theta)], axis=-1)
    return east, north, down


def east_north_down_km(lon0, lat0, depth0, lon, lat, depth,
                       earth_radius_km: float = EARTH_RADIUS_KM) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Components (km) of the straight-line displacement from point 0 to a point, along the
    east, north and down unit vectors at point 0."""
    d = to_ecef(lon, lat, depth, earth_radius_km) - to_ecef(lon0, lat0, depth0, earth_radius_km)
    east, north, down = local_basis(lon0, lat0)
    return np.sum(d * east, axis=-1), np.sum(d * north, axis=-1), np.sum(d * down, axis=-1)


def azimuth_deg(lon1, lat1, lon2, lat2):
    """Azimuth (degrees clockwise from north, 0 to 360) at point 1 of the great circle toward
    point 2, on the sphere of geocentric latitudes."""
    th1 = geodetic_to_geocentric(lat1)
    th2 = geodetic_to_geocentric(lat2)
    dlam = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    y = np.sin(dlam) * np.cos(th2)
    x = np.cos(th1) * np.sin(th2) - np.sin(th1) * np.cos(th2) * np.cos(dlam)
    return np.mod(np.degrees(np.arctan2(y, x)), 360.0)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def grid_axes(grid: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Node coordinates of the grid along longitude, latitude and depth."""
    lon = grid['olon'] + np.arange(int(grid['nlon'])) * grid['dlon']
    lat = grid['olat'] + np.arange(int(grid['nlat'])) * grid['dlat']
    depth = grid['odepth'] + np.arange(int(grid['ndepth'])) * grid['ddepth']
    return lon, lat, depth


def node_count(lo: float, hi: float, step: float) -> int:
    """Number of grid nodes from ``lo`` to ``hi`` (both included) at spacing ``step``."""
    return int(round((hi - lo) / step)) + 1


def km_per_degree(lat_deg, depth_km=0.0, earth_radius_km: float = EARTH_RADIUS_KM) -> Tuple[np.ndarray, np.ndarray]:
    """Length (km) of one degree of longitude (along the parallel) and of one degree of
    geodetic latitude (along the meridian), at a geodetic latitude and depth."""
    r = earth_radius_km - np.asarray(depth_km, dtype=float)
    lon_len = r * np.cos(geodetic_to_geocentric(lat_deg)) * np.pi / 180.0
    lat_len = r * dgeocentric_dgeodetic(lat_deg) * np.pi / 180.0
    return lon_len, lat_len


def check_grid(grid: Dict, earth_radius_km: float = EARTH_RADIUS_KM) -> None:
    """Raise ``ValueError`` unless the grid satisfies the checks of the spherical LATTE branch:
    at least 2 nodes per axis, positive spacing, longitude cells covering at most 360 degrees,
    latitude cells (half an interval beyond the first and last rows) strictly between -90 and
    90 degrees, and the deepest cell above the centre of the Earth."""
    n = [int(grid[k]) for k in ('nlon', 'nlat', 'ndepth')]
    d = [float(grid[k]) for k in ('dlon', 'dlat', 'ddepth')]
    if min(n) < 2:
        raise ValueError(f"nlon, nlat and ndepth must each be >= 2 (got {n})")
    if min(d) <= 0:
        raise ValueError(f"dlon, dlat and ddepth must be > 0 (got {d})")
    if n[0] * d[0] > 360.0:
        raise ValueError("the longitude cells cover more than 360 degrees")
    if grid['olat'] - 0.5 * d[1] <= -90.0 or grid['olat'] + (n[1] - 0.5) * d[1] >= 90.0:
        raise ValueError("the latitude cells must lie strictly between -90 and 90 degrees")
    if grid['odepth'] + (n[2] - 0.5) * d[2] >= earth_radius_km:
        raise ValueError("the deepest cell must lie above the centre of the Earth")


def latte_grid_lines(grid: Dict, earth_radius_km: float = EARTH_RADIUS_KM) -> List[str]:
    """Grid lines of a parameter file of the spherical LATTE branch (lengths in metres)."""
    return [
        f"nlon = {int(grid['nlon'])}",
        f"nlat = {int(grid['nlat'])}",
        f"ndepth = {int(grid['ndepth'])}",
        f"dlon = {float(grid['dlon']):.10g}",
        f"dlat = {float(grid['dlat']):.10g}",
        f"ddepth = {float(grid['ddepth']) * 1000.0:.10g}",
        f"olon = {float(grid['olon']):.10g}",
        f"olat = {float(grid['olat']):.10g}",
        f"odepth = {float(grid['odepth']) * 1000.0:.10g}",
        f"earth_radius = {earth_radius_km * 1000.0:.10g}",
    ]
