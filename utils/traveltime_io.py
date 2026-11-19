"""
Reading and writing the per-pick travel-time tables.

Three CSV files share one layout, each extending the previous one:

- ground truth (preprocess): ``shot_id, event_id, station, phase, observed_tt, weight,
  event_longitude, event_latitude, event_depth_km, station_longitude, station_latitude,
  station_depth_km, origin_time_s`` (+ ``magnitude`` when requested)
- synthetic (postprocess): the above plus ``synthetic_tt`` (and ``corrected_origin_time_s``
  when a source correction was applied)
- source relocation (postprocess): the synthetic table plus the corrected event position and
  origin time, and the optional ``grad_*``/``srch_*`` columns

Longitudes and WGS84 geodetic latitudes are in degrees, depths in km below mean sea level
(station elevations are negative depths), times in seconds. The units are written as
``units.<column>`` metadata in the CSV header (``# key: value`` lines).

Residuals and distances are not stored; use ``add_residual_s`` and ``event_station_distance_m``.
The dropped columns (``weighted_rms``, ``residual``, ``residual_s``, ``distance_m``) are removed
on read.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from utils.csv_metadata import csv_header_lines, flatten_metadata
from utils.io_utils import read_csv_file, ensure_directory_exists
from utils.spherical import EARTH_RADIUS_KM, hypocentral_distance_km, epicentral_distance_km

logger = logging.getLogger(__name__)

# Units of the travel-time table columns
TRAVELTIME_UNITS = {
    'observed_tt': 's',
    'synthetic_tt': 's',
    'origin_time_s': 's',
    'corrected_origin_time_s': 's',
    **{f'{prefix}_{name}': unit
       for prefix in ('event', 'station', 'corrected_event')
       for name, unit in (('longitude', 'deg'), ('latitude', 'deg'), ('depth_km', 'km'))},
}

_LENGTH_TO_KM = {'m': 1e-3, 'km': 1.0}


def get_units(df: pd.DataFrame) -> Dict[str, str]:
    """Units of the columns: from the ``units.*`` file metadata, else the standard units."""
    metadata = df.attrs.get('metadata') or {}
    units = {col: TRAVELTIME_UNITS[col] for col in df.columns if col in TRAVELTIME_UNITS}
    units.update({key[len('units.'):]: value for key, value in metadata.items()
                  if key.startswith('units.') and key[len('units.'):] in df.columns})
    return units


def km_per_unit(df: pd.DataFrame, column: str) -> float:
    """Factor converting ``column`` to km, from its unit."""
    unit = get_units(df).get(column)
    if unit not in _LENGTH_TO_KM:
        raise ValueError(f"Column '{column}' has no length unit (found {unit!r})")
    return _LENGTH_TO_KM[unit]


def read_traveltime_csv(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Read a ground-truth, synthetic or source-relocation CSV.

    The standard name is preferred and legacy file names are searched (with a warning);
    legacy column names are renamed and dropped columns removed (see the module docstring).
    The header metadata is attached as ``df.attrs['metadata']``.
    """
    return read_csv_file(filepath, **kwargs)


def write_traveltime_csv(df: pd.DataFrame, filepath: Union[str, Path], data_type: str,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Write a travel-time table with ``units.*`` and ``data_type`` metadata in the header.

    Args:
        df: Table to write.
        filepath: Output path.
        data_type: ``geometry_traveltime``, ``synthetic_traveltime`` or ``source_relocation``.
        metadata: Additional metadata (merged over what is derived), e.g. ``earth_radius_km``.
    """
    filepath = Path(filepath)
    ensure_directory_exists(filepath.parent)

    meta: Dict[str, Any] = {'data_type': data_type, 'coordinate_system': 'spherical',
                            'earth_radius_km': EARTH_RADIUS_KM}
    meta.update({f'units.{col}': unit for col, unit in TRAVELTIME_UNITS.items() if col in df.columns})
    meta.update(flatten_metadata(metadata))

    with open(filepath, 'w', newline='') as f:
        f.writelines(csv_header_lines(meta))
        df.to_csv(f, index=False)


def add_residual_s(df: pd.DataFrame) -> pd.DataFrame:
    """Add the in-memory column ``residual_s = observed_tt - synthetic_tt`` (never written to files)."""
    df = df.copy()
    df['residual_s'] = df['observed_tt'] - df['synthetic_tt']
    return df


def event_station_distance_m(df: pd.DataFrame, kind: str = 'hypocentral') -> pd.Series:
    """
    Source-receiver distance in metres, as the spherical LATTE branch measures it.

    Args:
        df: Table with ``event_longitude/latitude/depth_km`` and
            ``station_longitude/latitude/depth_km``.
        kind: ``hypocentral``: the straight line between the Earth-centred positions;
            ``epicentral``: the great-circle distance at the Earth's surface.

    The Earth radius is the ``earth_radius_km`` of the file metadata, or 6371 km.
    """
    radius = float((df.attrs.get('metadata') or {}).get('earth_radius_km', EARTH_RADIUS_KM))
    if kind == 'hypocentral':
        km = hypocentral_distance_km(
            df['event_longitude'].to_numpy(float), df['event_latitude'].to_numpy(float),
            df['event_depth_km'].to_numpy(float), df['station_longitude'].to_numpy(float),
            df['station_latitude'].to_numpy(float), df['station_depth_km'].to_numpy(float), radius)
    elif kind == 'epicentral':
        km = epicentral_distance_km(
            df['event_longitude'].to_numpy(float), df['event_latitude'].to_numpy(float),
            df['station_longitude'].to_numpy(float), df['station_latitude'].to_numpy(float), radius)
    else:
        raise ValueError(f"Unknown distance kind: {kind!r}")
    return pd.Series(km * 1000.0, index=df.index)


def convert_coordinates_to_km(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with the Cartesian coordinate columns converted to km.

    The conversion uses the units of the columns (``units.*`` header metadata, metres for
    legacy files). Ray tracing, DWS and ray density work in km internally, so this is called
    once, right after reading the table.
    """
    df = df.copy()
    columns = [f'{prefix}_{axis}' for prefix in ('event', 'station', 'corrected_event')
               for axis in ('x', 'y', 'z') if f'{prefix}_{axis}' in df.columns]
    for column in columns:
        df[column] = df[column] * km_per_unit(df, column)
    metadata = dict(df.attrs.get('metadata') or {})
    metadata.update({f'units.{column}': 'km' for column in columns})
    df.attrs['metadata'] = metadata
    return df
