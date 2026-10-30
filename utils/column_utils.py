"""
Column-name conventions shared by velocity models, event tables and station tables.

Cartesian coordinates are always ``x``, ``y``, ``z`` (km). Older files used
``X_grid_km``, ``Y_grid_km``, ``Z_grid_km`` (often alongside duplicate lowercase ``x``,
``y``, ``z``) and carried grid index columns (``i``, ``j``, ``k``, ``i_raw``, ``j_raw``,
``k_raw``). Those are still readable: the legacy coordinate names are renamed and the
index columns are dropped, with a warning.
"""

import logging
from typing import Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Legacy coordinate column -> current name
LEGACY_COORDINATE_COLUMNS = {
    'X_grid_km': 'x',
    'Y_grid_km': 'y',
    'Z_grid_km': 'z',
}

# Current coordinate column -> legacy name
_CURRENT_TO_LEGACY = {new: old for old, new in LEGACY_COORDINATE_COLUMNS.items()}

# Grid index columns that are no longer part of any file
REMOVED_INDEX_COLUMNS = ('i_raw', 'j_raw', 'k_raw', 'i', 'j', 'k')

_warned = set()


def _warn_once(key, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


def normalize_coordinate_columns(df: pd.DataFrame, drop_index_columns: bool = True,
                                 context: str = '') -> pd.DataFrame:
    """
    Return ``df`` using the current coordinate names, accepting legacy files.

    - ``X_grid_km``/``Y_grid_km``/``Z_grid_km`` are renamed to ``x``/``y``/``z``. If the
      legacy file also has lowercase ``x``/``y``/``z`` (duplicates), those are replaced.
    - ``i``, ``j``, ``k``, ``i_raw``, ``j_raw``, ``k_raw`` are dropped when
      ``drop_index_columns`` is True.

    The input DataFrame is not modified; it is returned as is when nothing needs changing.
    """
    legacy_found = [c for c in LEGACY_COORDINATE_COLUMNS if c in df.columns]
    index_found = [c for c in REMOVED_INDEX_COLUMNS if c in df.columns] if drop_index_columns else []
    if not legacy_found and not index_found:
        return df

    label = f" in {context}" if context else ""
    if legacy_found:
        _warn_once(('legacy', context, tuple(legacy_found)),
                   f"Legacy coordinate columns {legacy_found}{label} renamed to "
                   f"{[LEGACY_COORDINATE_COLUMNS[c] for c in legacy_found]}")
    if index_found:
        _warn_once(('index', context, tuple(index_found)),
                   f"Grid index columns {index_found}{label} are no longer used and were dropped")

    df_out = df.copy()
    duplicates = [LEGACY_COORDINATE_COLUMNS[c] for c in legacy_found
                  if LEGACY_COORDINATE_COLUMNS[c] in df_out.columns]
    df_out = df_out.drop(columns=duplicates + index_found)
    df_out = df_out.rename(columns=LEGACY_COORDINATE_COLUMNS)
    return df_out


def resolve_column_names(requested: Iterable[str], available: Iterable[str]) -> List[str]:
    """
    Map requested (current) column names onto the names present in a file.

    A requested ``x``/``y``/``z`` that is missing falls back to the legacy
    ``X_grid_km``/``Y_grid_km``/``Z_grid_km``. Requested columns that are not present
    under either name are skipped.
    """
    available = list(available)
    resolved = []
    for name in requested:
        if name in available:
            resolved.append(name)
        elif _CURRENT_TO_LEGACY.get(name) in available:
            resolved.append(_CURRENT_TO_LEGACY[name])
    return resolved
