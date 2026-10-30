"""
Reading and writing the per-pick travel-time tables.

Three CSV files share one layout, each extending the previous one:

- ground truth (preprocess): ``shot_id, event_id, station, phase, observed_tt, weight,
  event_longitude, event_latitude, event_depth_km, station_longitude, station_latitude,
  station_depth_km, event_x, event_y, event_z, station_x, station_y, station_z,
  origin_time_s`` (+ ``magnitude`` when requested)
- synthetic (postprocess): the above plus ``synthetic_tt`` (and ``corrected_origin_time_s``
  when a source correction was applied)
- source relocation (postprocess): the synthetic table plus ``corrected_event_x/y/z``,
  ``corrected_origin_time_s`` and the optional ``grad_*``/``srch_*`` columns

Units are the same as in the legacy files: the Cartesian coordinates (``*_x/_y/_z``) are in
metres, depths in km, angles in degrees, times in seconds. They are written as ``units.<column>``
metadata in the CSV header (``# key: value`` lines). Files without that header (legacy files)
are read with the same units, so nothing is converted.

Residuals and distances are not stored; use ``add_residual_s`` and ``event_station_distance_m``.
Legacy column names (``event_x_transformed``, ...) and the dropped columns (``weighted_rms``,
``residual``, ``residual_s``, ``distance_m``) are handled on read.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from utils.csv_metadata import csv_header_lines, flatten_metadata
from utils.io_utils import read_csv_file, ensure_directory_exists

logger = logging.getLogger(__name__)

# Units of the travel-time table columns (identical to the legacy files)
TRAVELTIME_UNITS = {
    'observed_tt': 's',
    'synthetic_tt': 's',
    'origin_time_s': 's',
    'corrected_origin_time_s': 's',
    'event_longitude': 'deg', 'event_latitude': 'deg', 'event_depth_km': 'km',
    'station_longitude': 'deg', 'station_latitude': 'deg', 'station_depth_km': 'km',
    **{f'{prefix}_{axis}': 'm'
       for prefix in ('event', 'station', 'corrected_event') for axis in ('x', 'y', 'z')},
}

_LENGTH_TO_KM = {'m': 1e-3, 'km': 1.0}


def get_units(df: pd.DataFrame) -> Dict[str, str]:
    """Units of the columns: from the ``units.*`` file metadata, else the legacy units."""
    metadata = df.attrs.get('metadata') or {}
    units = {col: TRAVELTIME_UNITS[col] for col in df.columns if col in TRAVELTIME_UNITS}
    units.update({key[len('units.'):]: value for key, value in metadata.items()
                  if key.startswith('units.') and key[len('units.'):] in df.columns})
    return units


def km_per_unit(df: pd.DataFrame, column: str) -> float:
    """Factor converting ``column`` to km, from its unit (metres for the coordinate columns)."""
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
        metadata: Additional metadata (merged over what is derived).
    """
    filepath = Path(filepath)
    ensure_directory_exists(filepath.parent)

    meta: Dict[str, Any] = {'data_type': data_type}
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
    Source-receiver distance in metres from the Cartesian coordinates.

    Args:
        df: Table with ``event_x/y/z`` and ``station_x/y/z`` (in the file's length unit).
        kind: ``hypocentral`` (3D) or ``epicentral`` (horizontal).
    """
    # km_per_unit gives km per column unit; multiply by 1000 for metres
    diffs = [(df[f'event_{a}'] - df[f'station_{a}']) * km_per_unit(df, f'event_{a}') * 1000.0
             for a in ('x', 'y', 'z')]
    if kind == 'epicentral':
        diffs = diffs[:2]
    elif kind != 'hypocentral':
        raise ValueError(f"Unknown distance kind: {kind!r}")
    return np.sqrt(sum(d ** 2 for d in diffs))


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
