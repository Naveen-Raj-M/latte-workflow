"""One-time QuakeML loading shared by visualization, HDF5 conversion and preprocessing.

The QuakeML files are parsed once WITHOUT any filtering. Each consumer
(raw data visualization, HDF5 conversion, the preprocessing pipeline) then
applies its own filters to the returned tables (see catalog_filters).

Optionally a secondary catalog (e.g. USGS) is loaded with the same parsing
options and appended to the same tables; rows are distinguishable through the
``catalog`` column.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from obspy import read_events
from obspy.geodetics import gps2dist_azimuth
from omegaconf import DictConfig
from tqdm import tqdm

from preprocess.data_loader import (
    calculate_distances,
    get_qml_files,
    load_station_inventory,
)

EVENT_COLUMNS = ['event_id', 'catalog', 'origin_time', 'latitude', 'longitude',
                 'depth', 'magnitude', 'magnitude_type', 'num_picks', 'num_arrivals']
STATION_COLUMNS = ['station_id', 'latitude', 'longitude', 'depth']
PICK_COLUMNS = ['pick_id', 'event_id', 'catalog', 'station_id', 'pick_time',
                'phase_hint', 'evaluation_mode', 'evaluation_status',
                'epicentral_distance', 'hypocentral_distance']
ARRIVAL_COLUMNS = ['arrival_id', 'event_id', 'catalog', 'pick_id', 'station_id',
                   'phase', 'azimuth', 'distance', 'time_residual', 'time_weight',
                   'epicentral_distance', 'hypocentral_distance']
NUMERIC_COLUMNS = ['latitude', 'longitude', 'depth', 'magnitude', 'azimuth',
                   'distance', 'time_residual', 'time_weight',
                   'epicentral_distance', 'hypocentral_distance']


@dataclass
class CatalogData:
    """Events, stations, arrivals and picks parsed from QuakeML."""
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    stations: pd.DataFrame = field(default_factory=pd.DataFrame)
    arrivals: pd.DataFrame = field(default_factory=pd.DataFrame)
    picks: pd.DataFrame = field(default_factory=pd.DataFrame)

    def copy(self) -> 'CatalogData':
        return CatalogData(self.events.copy(), self.stations.copy(),
                           self.arrivals.copy(), self.picks.copy())

    def summary(self) -> str:
        return (f"{len(self.events)} events, {len(self.stations)} stations, "
                f"{len(self.arrivals)} arrivals, {len(self.picks)} picks")


# ---------------------------------------------------------------------------
# Station inventories
# ---------------------------------------------------------------------------

def load_station_locations(station_file: str) -> Dict[str, List[Tuple[float, float]]]:
    """Load EVERY distinct location listed for each station code.

    load_station_inventory() keeps one entry per code (last wins), but some codes
    are listed at several positions. Comparing a code against only the last of
    those would flag a false conflict when merging catalogs.

    Args:
        station_file: Path to station inventory file.

    Returns:
        Dictionary mapping station code to a list of (lat, lon).
    """
    locations = defaultdict(list)
    try:
        with open(station_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0].startswith('#') or len(parts) < 4:
                    continue
                latlon = (float(parts[1]), float(parts[2]))
                if latlon not in locations[parts[0]]:
                    locations[parts[0]].append(latlon)
    except (TypeError, OSError):
        pass
    return locations


def merge_station_inventories(primary: Dict, secondary: Dict, secondary_cfg,
                              primary_locations: Optional[Dict] = None
                              ) -> Tuple[Dict, Dict, Dict]:
    """Merge a secondary station inventory into the primary one, keeping IDs unique.

    Station codes are bare (no network prefix), so the same code can denote
    different stations in the two catalogs. Codes present in both lists are
    compared by distance and handled in four cases:

      1. same code, identical location      -> unchanged, one shared station
      2. same code, < match_tolerance_m     -> suffixed only if
                                               suffix_near_duplicate_stations
      3. different code                     -> unchanged (nothing to resolve)
      4. same code, > match_tolerance_m     -> suffixed if
                                               suffix_conflicting_stations

    Suffixes are applied to secondary stations only; primary IDs never change.

    Args:
        primary: Primary station inventory.
        secondary: Secondary station inventory.
        secondary_cfg: ``data.secondary_catalogs`` settings plus the ``station_suffix`` of this catalog.
        primary_locations: Optional output of load_station_locations() for the
            primary file; the closest listed location decides the match.

    Returns:
        Tuple of (merged inventory, {secondary code: output id}, report dict).
    """
    suffix = secondary_cfg.get('station_suffix', '_usgs')
    identical_tol = secondary_cfg.get('identical_tolerance_m', 1.0)
    match_tol = secondary_cfg.get('match_tolerance_m', 200.0)
    suffix_near = secondary_cfg.get('suffix_near_duplicate_stations', False)
    suffix_conflict = secondary_cfg.get('suffix_conflicting_stations', True)

    merged = dict(primary)
    id_map = {}
    report = {'identical': [], 'near_duplicate': [], 'conflict': [],
              'secondary_only': [], 'unresolved_conflict': []}

    for sta_id, info in secondary.items():
        if sta_id not in primary:
            merged[sta_id] = info
            id_map[sta_id] = sta_id
            report['secondary_only'].append(sta_id)
            continue

        if primary_locations and sta_id in primary_locations:
            candidates = primary_locations[sta_id]
        else:
            candidates = [(primary[sta_id]['lat'], primary[sta_id]['lon'])]
        dist_m = min(gps2dist_azimuth(lat, lon, info['lat'], info['lon'])[0]
                     for lat, lon in candidates)

        if dist_m <= identical_tol:
            id_map[sta_id] = sta_id
            report['identical'].append((sta_id, dist_m))
            continue

        if dist_m <= match_tol:
            case, apply_suffix = 'near_duplicate', suffix_near
        else:
            case, apply_suffix = 'conflict', suffix_conflict

        if apply_suffix:
            new_id = f'{sta_id}{suffix}'
            merged[new_id] = info
            id_map[sta_id] = new_id
            report[case].append((sta_id, new_id, dist_m))
        else:
            # Not suffixed: secondary picks fall back to the primary coordinates
            id_map[sta_id] = sta_id
            report[case].append((sta_id, sta_id, dist_m))
            if case == 'conflict':
                report['unresolved_conflict'].append((sta_id, dist_m))

    return merged, id_map, report


def print_station_merge_report(report: Dict, secondary_cfg, label: str = 'secondary') -> None:
    """Print what merge_station_inventories() did, per case."""
    suffix_near = secondary_cfg.get('suffix_near_duplicate_stations', False)
    suffix_conflict = secondary_cfg.get('suffix_conflicting_stations', True)
    match_tol = secondary_cfg.get('match_tolerance_m', 200.0)

    print(f"\nStation ID reconciliation (merged inventory vs '{label}'):")
    print(f"  Secondary-only codes (added as-is)  : {len(report['secondary_only'])}")
    print(f"  Same code, identical location       : {len(report['identical'])}")
    print(f"  Same code, < {match_tol:g} m apart "
          f"({'suffixed' if suffix_near else 'shared'}): {len(report['near_duplicate'])}")
    for sta_id, out_id, dist_m in report['near_duplicate']:
        print(f"      {sta_id:<10} {dist_m:8.1f} m  -> {out_id}")
    print(f"  Same code, > {match_tol:g} m apart "
          f"({'suffixed' if suffix_conflict else 'shared'}): {len(report['conflict'])}")
    for sta_id, out_id, dist_m in report['conflict']:
        print(f"      {sta_id:<10} {dist_m / 1000:8.1f} km -> {out_id}")

    if report['unresolved_conflict']:
        print("\n  WARNING: these codes denote different stations in the two catalogs "
              "but were NOT suffixed, so secondary picks use the primary coordinates:")
        for sta_id, dist_m in report['unresolved_conflict']:
            print(f"      {sta_id:<10} off by {dist_m / 1000:.1f} km")
        print("  Set data.secondary_catalogs.suffix_conflicting_stations=true to separate them.")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _select_origin(event, use_preferred: bool, index: int):
    if use_preferred:
        return event.preferred_origin() or (event.origins[0] if event.origins else None)
    return event.origins[index] if len(event.origins) > index else None


def _select_magnitude(event, use_preferred: bool, index: int):
    if use_preferred:
        return event.preferred_magnitude() or (event.magnitudes[0] if event.magnitudes else None)
    return event.magnitudes[index] if len(event.magnitudes) > index else None


def extract_raw_event_data(catalog, station_inventory: Dict, options: Dict,
                           catalog_label: str = '',
                           station_id_map: Optional[Dict] = None
                           ) -> Tuple[List, List, List, Set]:
    """Extract events, picks and arrivals from a catalog WITHOUT filtering.

    Station coordinates are looked up by the raw QuakeML station code; only the
    ID written to the output rows is remapped through ``station_id_map``.

    Args:
        catalog: ObsPy catalog.
        station_inventory: Station inventory (raw code -> lat/lon/elevation).
        options: Dict with use_preferred_origin/origin_index/
            use_preferred_magnitude/magnitude_index.
        catalog_label: Value for the ``catalog`` column.
        station_id_map: Optional {raw code: output id} remap.

    Returns:
        Tuple of (events, picks, arrivals, stations_used as output ids).
    """
    station_id_map = station_id_map or {}
    events, picks, arrivals, stations_used = [], [], [], set()

    for event in catalog:
        origin = _select_origin(event, options['use_preferred_origin'], options['origin_index'])
        if origin is None:
            continue
        magnitude = _select_magnitude(event, options['use_preferred_magnitude'],
                                      options['magnitude_index'])

        event_id = str(event.resource_id.id)
        src_lat, src_lon = origin.latitude, origin.longitude
        src_depth = origin.depth / 1000.0 if origin.depth is not None else 0.0  # km

        pick_dict = {pick.resource_id.id: pick for pick in event.picks}
        distance_cache = {}  # raw station code -> (epi, hypo, azimuth) or None

        def station_geometry(raw_id):
            if raw_id not in distance_cache:
                sta = station_inventory.get(raw_id)
                distance_cache[raw_id] = None if sta is None else calculate_distances(
                    src_lat, src_lon, src_depth, sta['lat'], sta['lon'], sta['elevation'])
            return distance_cache[raw_id]

        event_arrivals, event_picks = [], []
        event_stations = set()

        for arrival in (origin.arrivals or []):
            pick = pick_dict.get(arrival.pick_id.id)
            if pick is None:
                continue
            raw_id = getattr(getattr(pick, 'waveform_id', None), 'station_code', None)
            if raw_id is None:
                continue
            station_id = station_id_map.get(raw_id, raw_id)
            geometry = station_geometry(raw_id)
            epi, hypo, az = geometry if geometry else (None, None, None)

            event_arrivals.append({
                'arrival_id': str(arrival.resource_id.id),
                'event_id': event_id,
                'catalog': catalog_label,
                'pick_id': arrival.pick_id.id,
                'station_id': station_id,
                'phase': arrival.phase or '',
                'azimuth': az if az is not None else arrival.azimuth,
                'distance': arrival.distance,
                'time_residual': arrival.time_residual,
                'time_weight': arrival.time_weight,
                'epicentral_distance': epi,
                'hypocentral_distance': hypo,
            })
            event_stations.add(station_id)

        for pick_id, pick in pick_dict.items():
            raw_id = getattr(getattr(pick, 'waveform_id', None), 'station_code', None)
            if raw_id is None:
                continue
            station_id = station_id_map.get(raw_id, raw_id)
            geometry = station_geometry(raw_id)
            event_picks.append({
                'pick_id': str(pick_id),
                'event_id': event_id,
                'catalog': catalog_label,
                'station_id': station_id,
                'pick_time': str(pick.time),
                'phase_hint': pick.phase_hint or '',
                'evaluation_mode': pick.evaluation_mode or '',
                'evaluation_status': pick.evaluation_status or '',
                'epicentral_distance': geometry[0] if geometry else None,
                'hypocentral_distance': geometry[1] if geometry else None,
            })
            event_stations.add(station_id)

        if not event_arrivals and not event_picks:
            continue

        events.append({
            'event_id': event_id,
            'catalog': catalog_label,
            'origin_time': str(origin.time),
            'latitude': src_lat,
            'longitude': src_lon,
            'depth': src_depth,  # km, positive = below MSL
            'magnitude': magnitude.mag if magnitude else None,
            'magnitude_type': (magnitude.magnitude_type or '') if magnitude else '',
            'num_picks': len(event_picks),
            'num_arrivals': len(event_arrivals),
        })
        arrivals.extend(event_arrivals)
        picks.extend(event_picks)
        stations_used.update(event_stations)

    return events, picks, arrivals, stations_used


def _process_single_qml_file(qml_file: str, station_inventory: Dict, options: Dict,
                             catalog_label: str, station_id_map: Optional[Dict]):
    """Multiprocessing worker. Returns (events, picks, arrivals, stations, error)."""
    try:
        catalog = read_events(qml_file)
        return (*extract_raw_event_data(catalog, station_inventory, options,
                                        catalog_label, station_id_map), None)
    except Exception as e:
        return [], [], [], set(), f"{qml_file}: {e}"


def _parse_catalog(qml_files: List[str], station_inventory: Dict, options: Dict,
                   num_workers: int, catalog_label: str,
                   station_id_map: Optional[Dict], desc: str):
    worker = partial(_process_single_qml_file, station_inventory=station_inventory,
                     options=options, catalog_label=catalog_label,
                     station_id_map=station_id_map)
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(worker, qml_files), total=len(qml_files), desc=desc))

    events, picks, arrivals, stations, errors = [], [], [], set(), []
    for ev, pk, ar, st, err in results:
        events.extend(ev)
        picks.extend(pk)
        arrivals.extend(ar)
        stations.update(st)
        if err:
            errors.append(err)
    if errors:
        print(f"  WARNING: {len(errors)} file(s) could not be parsed (first: {errors[0]})")
        logging.getLogger(__name__).warning(f"{len(errors)} QuakeML files failed to parse")
    return events, picks, arrivals, stations


def _to_frame(rows: List[Dict], columns: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _discover_files(data_dir: str, years, max_files, label: str) -> List[str]:
    print(f"\nSearching for {label} QML files in: {data_dir}")
    qml_files, info = get_qml_files(data_dir, years, max_files)
    if info['years_found']:
        print(f"  Year directories: {', '.join(info['years_found'])}")
    for yr, cnt in info['files_per_year'].items():
        print(f"    {yr}/: {cnt} QML file(s)")
    if info['direct_files_count']:
        print(f"  Direct files: {info['direct_files_count']} QML file(s)")
    print(f"  Total: {len(qml_files)} QML file(s)")
    return qml_files


def _secondary_catalog_entries(secondary_cfg) -> List[Dict]:
    """Zip the parallel ``data.secondary_catalogs`` lists into one dict per catalog.

    Raises:
        ValueError: If quakeml_dir, station_file, catalog_label and station_suffix
            do not have the same length.
    """
    keys = ['quakeml_dir', 'station_file', 'catalog_label', 'station_suffix']
    lists = {}
    for key in keys:
        value = secondary_cfg.get(key)
        lists[key] = [] if value is None else ([value] if isinstance(value, str) else list(value))
    lengths = {key: len(values) for key, values in lists.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            "data.secondary_catalogs: quakeml_dir, station_file, catalog_label and "
            f"station_suffix must have the same length, got {lengths}")
    return [dict(zip(keys, values)) for values in zip(*(lists[k] for k in keys))]


def load_quakeml_catalog(config: DictConfig) -> CatalogData:
    """Parse the configured QuakeML catalog(s) once, without filtering.

    Reads ``data.quakeml_dir``/``data.station_file`` and, when enabled, every
    catalog listed in ``data.secondary_catalogs``.

    Args:
        config: Full configuration.

    Returns:
        CatalogData with events, stations, arrivals and picks.
    """
    data_cfg = config.data
    secondary_cfg = data_cfg.get('secondary_catalogs') or {}
    entries = _secondary_catalog_entries(secondary_cfg) if secondary_cfg.get('enabled', False) else []
    options = {
        'use_preferred_origin': bool(data_cfg.use_preferred_origin),
        'origin_index': int(data_cfg.get('origin_index', 0)),
        'use_preferred_magnitude': bool(data_cfg.use_preferred_magnitude),
        'magnitude_index': int(data_cfg.get('magnitude_index', 0)),
    }

    print("\nLoading data from QuakeML files (parsed once, unfiltered)...")
    print(f"Loading station inventory from: {data_cfg.station_file}")
    inventory = load_station_inventory(data_cfg.station_file)
    print(f"  Loaded {len(inventory)} stations")

    # Each additional catalog is merged into the inventory accumulated so far
    merged_inventory = dict(inventory)
    locations = load_station_locations(data_cfg.station_file)
    catalogs = [{'label': data_cfg.get('catalog_label', 'texnet'), 'dir': data_cfg.quakeml_dir,
                 'years': data_cfg.years, 'max_files': data_cfg.max_files,
                 'inventory': inventory, 'id_map': None, 'desc': 'Primary'}]
    for i, entry in enumerate(entries):
        label = entry['catalog_label']
        print(f"Loading station inventory of '{label}' from: {entry['station_file']}")
        entry_inventory = load_station_inventory(entry['station_file'])
        print(f"  Loaded {len(entry_inventory)} stations")
        merge_cfg = {**dict(secondary_cfg), 'station_suffix': entry['station_suffix']}
        merged_inventory, id_map, report = merge_station_inventories(
            merged_inventory, entry_inventory, merge_cfg, primary_locations=locations)
        for code, out_id in id_map.items():
            latlon = (entry_inventory[code]['lat'], entry_inventory[code]['lon'])
            if latlon not in locations[out_id]:
                locations[out_id].append(latlon)
        print_station_merge_report(report, merge_cfg, label)
        print(f"  Merged inventory: {len(merged_inventory)} unique stations")
        catalogs.append({'label': label, 'dir': entry['quakeml_dir'],
                         'years': secondary_cfg.get('years'),
                         'max_files': secondary_cfg.get('max_files'),
                         'inventory': entry_inventory, 'id_map': id_map,
                         'desc': f'Secondary {i + 1}'})

    for cat in catalogs:
        cat['files'] = _discover_files(cat['dir'], cat['years'], cat['max_files'], cat['label'])
    if not any(cat['files'] for cat in catalogs):
        print("No QML files found.")
        return CatalogData()

    num_workers = config.parallel.num_workers or cpu_count()
    print(f"\nParsing QML files using {num_workers} workers...")

    all_events, all_picks, all_arrivals, all_stations = [], [], [], set()
    for cat in catalogs:
        if not cat['files']:
            continue
        ev, pk, ar, st = _parse_catalog(cat['files'], cat['inventory'], options, num_workers,
                                        cat['label'], cat['id_map'], cat['desc'])
        print(f"  {cat['desc']} ({cat['label']}): {len(ev)} events, {len(pk)} picks, {len(ar)} arrivals")
        all_events.extend(ev)
        all_picks.extend(pk)
        all_arrivals.extend(ar)
        all_stations.update(st)

    stations = pd.DataFrame(
        [{'station_id': sid, 'latitude': merged_inventory[sid]['lat'],
          'longitude': merged_inventory[sid]['lon'],
          'depth': -merged_inventory[sid]['elevation'] / 1000.0}  # km, negative = above MSL
         for sid in sorted(all_stations) if sid in merged_inventory],
        columns=STATION_COLUMNS)

    catalog = CatalogData(
        events=_to_frame(all_events, EVENT_COLUMNS),
        stations=stations,
        arrivals=_to_frame(all_arrivals, ARRIVAL_COLUMNS),
        picks=_to_frame(all_picks, PICK_COLUMNS),
    )
    print(f"\nRaw QuakeML data: {catalog.summary()}")
    return catalog
