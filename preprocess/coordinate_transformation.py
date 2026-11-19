"""Coordinates of the LATTE preprocessing (spherical Earth).

The spherical LATTE branch takes positions as longitude, WGS84 geodetic latitude and depth
below mean sea level, the coordinates of the catalog. No transformation is applied; this module
checks the coordinates of events and stations and the grid configuration, and puts the
coordinate columns in a standard form:

    longitude  degrees
    latitude   degrees, WGS84 geodetic
    depth      km below mean sea level (stations above sea level have negative depth)

The geometry of the spherical Earth is in ``utils.spherical``.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from utils.spherical import EARTH_RADIUS_KM

COORDINATE_COLUMNS = ['longitude', 'latitude', 'depth']


def standardize_coordinates(df: pd.DataFrame, label: str,
                            earth_radius_km: float = EARTH_RADIUS_KM) -> pd.DataFrame:
    """Return a copy of ``df`` with the columns ``longitude``, ``latitude`` and ``depth``.

    ``lon`` and ``lat`` are accepted as aliases. Raises ``ValueError`` if a coordinate is
    missing or not finite, a latitude is outside [-90, 90] degrees, or a depth reaches the
    centre of the Earth.

    Args:
        df: Events or stations.
        label: Name used in error messages ('events' or 'stations').
        earth_radius_km: Radius of the spherical Earth, km.
    """
    out = df.rename(columns={'lon': 'longitude', 'lat': 'latitude'}).copy()
    missing = [c for c in COORDINATE_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"{label} lack the coordinate columns {missing}")
    values = out[COORDINATE_COLUMNS].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} have missing or non-finite coordinates")
    if np.any(np.abs(out['latitude']) > 90.0):
        raise ValueError(f"{label} have latitudes outside [-90, 90] degrees")
    if np.any(out['depth'] >= earth_radius_km):
        raise ValueError(f"{label} have depths at or below the centre of the Earth")
    return out


def check_grid_configuration(coord_config: DictConfig) -> None:
    """Raise ``ValueError`` unless the grid origin and spacing are valid: three positive
    spacings (dlon and dlat in degrees, ddepth in km), an origin latitude in (-90, 90) and a
    positive Earth radius."""
    spacing = list(coord_config.grid_spacing)
    if len(spacing) != 3 or min(spacing) <= 0:
        raise ValueError(f"coordinate_transform.grid_spacing must be three positive values "
                         f"[dlon_deg, dlat_deg, ddepth_km] (got {spacing})")
    if not -90.0 < float(coord_config.origin_lat) < 90.0:
        raise ValueError("coordinate_transform.origin_lat must lie in (-90, 90) degrees")
    if float(coord_config.get('earth_radius_km', EARTH_RADIUS_KM)) <= 0:
        raise ValueError("coordinate_transform.earth_radius_km must be > 0")


def transform_coordinates(config: DictConfig, events_df: pd.DataFrame,
                          stations_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Check and standardize the coordinates of events and stations.

    Args:
        config: Full configuration.
        events_df: Events with geodetic coordinates.
        stations_df: Stations with geodetic coordinates.

    Returns:
        Tuple of (events_df, stations_df) with the columns longitude, latitude, depth.
    """
    logger = logging.getLogger(__name__)
    coord_config = config.coordinate_transform
    check_grid_configuration(coord_config)
    radius = float(coord_config.get('earth_radius_km', EARTH_RADIUS_KM))

    events = standardize_coordinates(events_df, 'events', radius)
    stations = standardize_coordinates(stations_df, 'stations', radius)

    logger.info(f"Spherical Earth of radius {radius} km; coordinates are longitude, WGS84 geodetic "
                f"latitude (degrees) and depth below MSL (km)")
    for label, df in (('Events', events), ('Stations', stations)):
        logger.info(f"  {label}: lon [{df['longitude'].min():.4f}, {df['longitude'].max():.4f}], "
                    f"lat [{df['latitude'].min():.4f}, {df['latitude'].max():.4f}], "
                    f"depth [{df['depth'].min():.3f}, {df['depth'].max():.3f}] km")
    return events, stations
