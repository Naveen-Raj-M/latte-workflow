"""
Source-receiver exchange layout, rebuilt from the geometry files.

With ``yn_exchange_sr`` (always on for ``tloc``) LATTE's ``divide_shots`` turns every unique
receiver into a virtual source and every kept shot into a virtual receiver:

- Geometry shots are loaded in ``geometry.txt`` order; shots with no valid source or receiver,
  or excluded by ``sid_select``/``sid_exclude`` etc., are dropped and the rest are compacted.
- Virtual source ``i`` (1-based) is the i-th distinct receiver, scanning the receivers of the kept
  shots in order and keeping the first appearance (flit ``unique``, exact single-precision equality).
- ``shot_<i>_traveltime_{p,s}.bin`` holds one float32 per kept shot, in the same order.

Positions are those of the geometry files: longitude and WGS84 geodetic latitude (degrees) and depth
below mean sea level (m). Distances between positions (tolerances, matching to stations) are
straight-line distances in metres between the Earth-centred positions on the spherical Earth.

``exchange_geometry.txt`` (virtual id, lon, lat, depth_m) written by LATTE is optional: when given it is
used as the virtual-source table and the geometry-derived table is only compared against it.
"""

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.spherical import EARTH_RADIUS_KM, to_ecef

logger = logging.getLogger(__name__)

STATION_COLUMNS = ["station_longitude", "station_latitude", "station_depth_km"]

FILTER_HINT = (
    "If LATTE dropped shots or receivers (e.g. sid_select/sid_exclude, no valid receivers), list them "
    "with the config keys filtered_shot_ids / filtered_receiver_ids."
)


class ExchangeLayout:
    """Result of the exchange: kept shot ids in LATTE order and the virtual-source coordinates."""

    def __init__(self, shot_ids: List[int], virtual_coords: np.ndarray, source: str):
        self.shot_ids = list(shot_ids)  # array index -> shot id
        # row i -> virtual id i + 1: lon (deg), lat (deg), depth (m)
        self.virtual_coords = np.asarray(virtual_coords, dtype=np.float64)
        self.source = source  # 'geometry' | 'exchange_geometry'
        # True when virtual_coords came from LATTE's own exchange_geometry.txt
        self.from_exchange_geometry = source == "exchange_geometry"

    def matching_tolerance(self, tolerance: float) -> float:
        """Station-matching tolerance (m). exchange_geometry.txt holds LATTE's single-precision
        positions, which differ from the stations by up to the float32 resolution."""
        if self.from_exchange_geometry:
            return max(tolerance, float(float32_resolution_m(self.virtual_coords).max()))
        return tolerance

    @property
    def n_virtual(self) -> int:
        return len(self.virtual_coords)

    @property
    def n_shots(self) -> int:
        return len(self.shot_ids)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_shot_geometry(path) -> Tuple[int, np.ndarray]:
    """Return (shot_id, receivers[n, 3] float64: lon deg, lat deg, depth m) from one
    shot_<n>_geometry.txt (4 or 5 columns)."""
    tokens = Path(path).read_text().split()
    shot_id, n_sources = int(tokens[0]), int(tokens[1])
    n_receivers = int(tokens[2 + 4 * n_sources])
    values = np.array(tokens[3 + 4 * n_sources:], dtype=np.float64)
    return shot_id, values.reshape(n_receivers, -1)[:, :3]


def read_geometry(geometry_dir, geometry_file: str = "geometry.txt") -> Tuple[List[int], List[np.ndarray]]:
    """Read all shots listed in geometry.txt, in file order (the order LATTE loads them)."""
    geometry_dir = Path(geometry_dir)
    names = (geometry_dir / geometry_file).read_text().split()
    shot_ids, receivers = [], []
    for name in names:
        shot_id, rec = read_shot_geometry(geometry_dir / name)
        shot_ids.append(shot_id)
        receivers.append(rec)
    return shot_ids, receivers


def read_exchange_geometry(path) -> Dict[int, Tuple[float, float, float]]:
    """Read exchange_geometry.txt: ``virtual_id lon lat depth_m`` per line, '#' comments."""
    table: Dict[int, Tuple[float, float, float]] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 4:
                    table[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return table


# --------------------------------------------------------------------------- #
# Positions
# --------------------------------------------------------------------------- #
def earth_radius_km(df: pd.DataFrame) -> float:
    """Earth radius (km) of a travel-time table: its ``earth_radius_km`` metadata, or 6371 km."""
    return float((df.attrs.get("metadata") or {}).get("earth_radius_km", EARTH_RADIUS_KM))


def ecef_m(coords: np.ndarray, radius_km: float = EARTH_RADIUS_KM) -> np.ndarray:
    """Earth-centred positions (m) of rows (lon deg, lat deg, depth m)."""
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    return 1000.0 * to_ecef(coords[:, 0], coords[:, 1], coords[:, 2] / 1000.0, radius_km)


def float32_resolution_m(coords: np.ndarray) -> np.ndarray:
    """Upper bound (m) per row of the distance between neighbouring single-precision positions:
    the float32 spacing of lon and lat (at most 111.2 km per degree on the 6371 km sphere,
    1.1 % margin for a larger radius) plus that of the depth."""
    c = np.abs(np.asarray(coords, dtype=np.float64).reshape(-1, 3)).astype("<f4")
    spacing = np.spacing(c).astype(np.float64)
    return 112.5e3 * (spacing[:, 0] + spacing[:, 1]) + spacing[:, 2]


def distances_m(a: np.ndarray, b: np.ndarray, radius_km: float = EARTH_RADIUS_KM) -> np.ndarray:
    """Straight-line distances (m) between every row of ``a`` and every row of ``b``: shape [len(a), len(b)]."""
    return np.linalg.norm(ecef_m(a, radius_km)[:, None, :] - ecef_m(b, radius_km)[None, :, :], axis=2)


# --------------------------------------------------------------------------- #
# The exchange itself
# --------------------------------------------------------------------------- #
def first_appearance_unique(coords: np.ndarray) -> np.ndarray:
    """Indices of the first occurrence of each distinct row (float32 equality), in order of appearance."""
    _, first_seen = np.unique(coords.astype("<f4"), axis=0, return_index=True)
    return np.sort(first_seen)


def station_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """One row per station name with station_longitude/latitude (deg) and station_depth_km from the
    ground-truth table."""
    stations = df[["station"] + STATION_COLUMNS].drop_duplicates("station").reset_index(drop=True)
    stations.attrs["metadata"] = dict(df.attrs.get("metadata") or {})
    return stations


def station_positions(df: pd.DataFrame) -> np.ndarray:
    """Station positions of a table as rows (lon deg, lat deg, depth m), the units of the geometry files."""
    lon, lat, depth_km = (df[c].to_numpy(dtype=np.float64) for c in STATION_COLUMNS)
    return np.column_stack([lon, lat, 1000.0 * depth_km])


def compute_layout(
    shot_ids: Sequence[int],
    receivers: Sequence[np.ndarray],
    filtered_shot_ids: Iterable[int] = (),
    filtered_receiver_coords: Optional[np.ndarray] = None,
    tolerance: float = 1.0,
    radius_km: float = EARTH_RADIUS_KM,
) -> ExchangeLayout:
    """
    Reproduce LATTE's exchange from geometry shots (in geometry.txt order).

    Filtered shots are dropped before the unique-receiver scan, exactly as LATTE compacts ``gmtr``,
    so they can change the virtual-source numbering. Filtered receivers (positions as rows lon, lat,
    depth m, matched within ``tolerance`` metres) are removed from every shot; a shot left without
    receivers is dropped.
    """
    drop_shots = set(int(s) for s in filtered_shot_ids)
    kept_ids, kept_rec = [], []
    for sid, rec in zip(shot_ids, receivers):
        if sid in drop_shots:
            continue
        if filtered_receiver_coords is not None and len(filtered_receiver_coords):
            d = distances_m(rec, filtered_receiver_coords, radius_km)
            rec = rec[~(d <= tolerance).any(axis=1)]
        if len(rec) == 0:
            continue
        kept_ids.append(sid)
        kept_rec.append(rec)
    if not kept_ids:
        raise ValueError("No shots left after applying filtered_shot_ids / filtered_receiver_ids")
    all_rec = np.concatenate(kept_rec)
    return ExchangeLayout(kept_ids, all_rec[first_appearance_unique(all_rec)], "geometry")


def match_stations(virtual_coords: np.ndarray, stations: pd.DataFrame, tolerance: float) -> Dict[int, List[str]]:
    """
    Virtual id -> every station name within ``tolerance`` metres (straight-line distance; co-located
    stations share a virtual source). Unmatched virtual ids are absent from the result.
    """
    radius = earth_radius_km(stations)
    station_ecef = ecef_m(station_positions(stations), radius)
    virtual_ecef = ecef_m(virtual_coords, radius)
    names = stations["station"].tolist()
    matched: Dict[int, List[str]] = {}
    for i, v in enumerate(virtual_ecef):
        hit = np.where(np.linalg.norm(station_ecef - v, axis=1) <= tolerance)[0]
        if hit.size:
            matched[i + 1] = [names[k] for k in hit]
    return matched


# --------------------------------------------------------------------------- #
# Resolution and validation
# --------------------------------------------------------------------------- #
def _compare_virtual_tables(computed: np.ndarray, given: np.ndarray, tolerance: float,
                            radius_km: float = EARTH_RADIUS_KM) -> None:
    """Warn if the geometry-derived virtual sources differ from exchange_geometry.txt."""
    if len(computed) != len(given):
        logger.warning(
            f"Exchange check: geometry gives {len(computed)} virtual sources but exchange_geometry.txt has "
            f"{len(given)}. Proceeding with exchange_geometry.txt. {FILTER_HINT}"
        )
        return
    # LATTE stores positions in single precision, so allow for the float32 resolution
    slack = tolerance + 2 * float32_resolution_m(given)
    dist = np.linalg.norm(ecef_m(computed, radius_km) - ecef_m(given, radius_km), axis=1)
    bad = np.where(dist > slack)[0]
    if bad.size:
        logger.warning(
            f"Exchange check: {bad.size}/{len(given)} virtual sources differ between the geometry-derived order "
            f"and exchange_geometry.txt by more than the tolerance (first: virtual id {bad[0] + 1}, "
            f"{dist[bad[0]]:.4f} m). Proceeding with exchange_geometry.txt. {FILTER_HINT}"
        )


def resolve_layout(
    df_ground_truth: pd.DataFrame,
    geometry_dir=None,
    geometry_file: str = "geometry.txt",
    exchange_geometry_file=None,
    filtered_shot_ids: Iterable[int] = (),
    filtered_receiver_ids: Iterable[str] = (),
    tolerance: float = 1.0,
) -> ExchangeLayout:
    """
    Build the exchange layout and check it against the reference (ground-truth) table.

    - ``geometry_dir`` given: the layout is computed from the geometry files like LATTE does.
    - ``exchange_geometry_file`` given (optional): its virtual-source table is used; if the geometry is
      also available the computed one is compared and mismatches only warn.
    - neither: error. With only ``exchange_geometry_file`` (legacy), the shot order is the sorted
      shot ids of the ground truth minus ``filtered_shot_ids``.

    Without ``exchange_geometry_file`` any mismatch with the reference raises.
    """
    filtered_shot_ids = [int(s) for s in (filtered_shot_ids or [])]
    filtered_receiver_ids = [str(s) for s in (filtered_receiver_ids or [])]
    if geometry_dir is None and exchange_geometry_file is None:
        raise ValueError("Source-receiver exchange needs geometry_dir (preferred) or exchange_geometry_file")

    radius = earth_radius_km(df_ground_truth)
    stations = station_coordinates(df_ground_truth)
    unknown = [s for s in filtered_receiver_ids if s not in set(stations["station"])]
    if unknown:
        logger.warning(f"filtered_receiver_ids not found in the ground truth: {unknown}")
    filtered_positions = station_positions(stations[stations["station"].isin(filtered_receiver_ids)])

    # Reference (ground truth) after the user's filters
    ref = df_ground_truth[~df_ground_truth["shot_id"].isin(filtered_shot_ids)
                          & ~df_ground_truth["station"].isin(filtered_receiver_ids)]
    ref_shots = sorted(ref["shot_id"].unique())

    computed = None
    if geometry_dir is not None:
        shot_ids, receivers = read_geometry(geometry_dir, geometry_file)
        computed = compute_layout(shot_ids, receivers, filtered_shot_ids, filtered_positions, tolerance, radius)

    if exchange_geometry_file is not None:
        table = read_exchange_geometry(exchange_geometry_file)
        if sorted(table) != list(range(1, len(table) + 1)):
            raise ValueError(f"{exchange_geometry_file}: virtual ids are not 1..{len(table)}")
        given = np.array([table[i] for i in range(1, len(table) + 1)], dtype=np.float64)
        if computed is not None:
            _compare_virtual_tables(computed.virtual_coords, given, tolerance, radius)
            layout = ExchangeLayout(computed.shot_ids, given, "exchange_geometry")
        else:
            layout = ExchangeLayout(ref_shots, given, "exchange_geometry")
    else:
        layout = computed

    # --- Reference checks (item: shot order and counts) ---
    if sorted(layout.shot_ids) != ref_shots:
        missing = sorted(set(layout.shot_ids) - set(ref_shots))[:10]
        extra = sorted(set(ref_shots) - set(layout.shot_ids))[:10]
        raise ValueError(
            f"Shots in the geometry ({layout.n_shots}) do not match the reference ground truth ({len(ref_shots)}): "
            f"only in geometry {missing}, only in ground truth {extra}. {FILTER_HINT}"
        )
    if layout.shot_ids != ref_shots:
        logger.warning(
            "Shot order in geometry.txt differs from the sorted shot ids of the ground truth; "
            "using the geometry.txt order (this is the order of the LATTE binary arrays)."
        )

    n_ref_locations = len(np.unique(station_positions(ref).astype("<f4"), axis=0))
    if n_ref_locations != layout.n_virtual:
        msg = (f"Reference ground truth has {n_ref_locations} distinct receiver locations but the exchange has "
               f"{layout.n_virtual} virtual sources ({layout.source}). {FILTER_HINT}")
        if layout.from_exchange_geometry:
            logger.warning(msg)
        else:
            raise ValueError(msg)
    return layout


def check_binary_counts(layout: ExchangeLayout, synthetic_dir, phases: Iterable[str]) -> None:
    """Number of shot_*_traveltime_<phase>.bin files must equal the number of virtual sources."""
    synthetic_dir = Path(synthetic_dir)
    for phase in phases:
        n_files = len(list(synthetic_dir.glob(f"shot_*_traveltime_{phase.lower()}.bin")))
        if n_files != layout.n_virtual:
            msg = (f"{synthetic_dir}: found {n_files} {phase}-wave binaries but the exchange has "
                   f"{layout.n_virtual} virtual sources ({layout.source}). {FILTER_HINT}")
            if layout.from_exchange_geometry:
                logger.warning(msg)
            else:
                raise ValueError(msg)


def check_array_size(layout: ExchangeLayout, bin_name: str, actual: int) -> None:
    """Each binary holds one value per kept shot; a mismatch means shots were filtered inside LATTE."""
    if actual != layout.n_shots:
        raise ValueError(
            f"Array size mismatch for {bin_name}: expected {layout.n_shots} elements (kept shots from "
            f"{layout.source}), got {actual}. {FILTER_HINT}"
        )
