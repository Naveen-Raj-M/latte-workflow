"""
Column-name conventions shared by velocity models, event tables and station tables.

Grid positions are ``lon``, ``lat`` (degrees, WGS84 geodetic) and ``depth`` (km below mean
sea level). Grid index columns (``i``, ``j``, ``k``, ``i_raw``, ``j_raw``, ``k_raw``) are not
part of any file; they are dropped on read, with a warning.
"""

import logging
from typing import Iterable, List

import pandas as pd

logger = logging.getLogger(__name__)

# Grid index columns that are not part of any file
REMOVED_INDEX_COLUMNS = ('i_raw', 'j_raw', 'k_raw', 'i', 'j', 'k')

_warned = set()


def _warn_once(key, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


def normalize_coordinate_columns(df: pd.DataFrame, drop_index_columns: bool = True,
                                 context: str = '') -> pd.DataFrame:
    """
    Return ``df`` without grid index columns (``i``, ``j``, ``k``, ``i_raw``, ...) when
    ``drop_index_columns`` is True. The input DataFrame is not modified; it is returned as is
    when nothing needs changing.
    """
    index_found = [c for c in REMOVED_INDEX_COLUMNS if c in df.columns] if drop_index_columns else []
    if not index_found:
        return df
    label = f" in {context}" if context else ""
    _warn_once(('index', context, tuple(index_found)),
               f"Grid index columns {index_found}{label} are not used and were dropped")
    return df.drop(columns=index_found)


def resolve_column_names(requested: Iterable[str], available: Iterable[str]) -> List[str]:
    """The requested column names that are present in ``available``, in the requested order."""
    available = set(available)
    return [name for name in requested if name in available]


# ---------------------------------------------------------------------------
# Geometry / travel-time tables (ground-truth, synthetic and source-relocation CSVs)
# ---------------------------------------------------------------------------

# Columns that are not written; residuals and distances are computed where needed
REMOVED_GEOMETRY_COLUMNS = ('weighted_rms', 'residual', 'residual_s', 'distance_m')


def normalize_geometry_columns(df: pd.DataFrame, context: str = '') -> pd.DataFrame:
    """
    Return ``df`` without the columns ``weighted_rms``, ``residual``, ``residual_s`` and
    ``distance_m``; residuals and distances are computed from the other columns where needed.
    The input DataFrame is not modified; it is returned as is when nothing changes.
    """
    removed_found = [c for c in REMOVED_GEOMETRY_COLUMNS if c in df.columns]
    if not removed_found:
        return df
    label = f" in {context}" if context else ""
    _warn_once(('removed_geometry', context, tuple(removed_found)),
               f"Columns {removed_found}{label} are not used and were dropped; "
               f"residuals and distances are computed from the other columns")
    return df.drop(columns=removed_found)
