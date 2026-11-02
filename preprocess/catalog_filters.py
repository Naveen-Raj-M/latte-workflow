"""Filtering of parsed QuakeML catalogs.

Every consumer of the one-time QuakeML load (raw data visualization, HDF5
conversion, preprocessing pipeline) applies its own ``filters`` node through
apply_catalog_filters(). After every stage the tables are synchronised so that
arrivals/picks only reference remaining events, every remaining event has at
least one arrival or pick, and only referenced stations are kept.

Supported filter keys (all optional, null = no filter):
    manual_picks_only, magnitude_min/max, lat_min/max, lon_min/max,
    depth_min/max, epicentral_distance_min/max, hypocentral_distance_min/max,
    require_both_p_and_s, azimuthal_gap_max,
    min_picks_per_event, max_picks_per_event.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from preprocess.quakeml_loader import CatalogData


def compute_azimuthal_gap(arrivals: pd.DataFrame) -> pd.Series:
    """Largest azimuthal gap (degrees) per event, from arrival azimuths.

    Args:
        arrivals: Arrivals with ``event_id`` and ``azimuth`` columns.

    Returns:
        Series indexed by event_id (events without >= 2 azimuths get 360).
    """
    if arrivals.empty or 'azimuth' not in arrivals.columns:
        return pd.Series(dtype=float)

    def gap(azimuths: pd.Series) -> float:
        az = np.sort(azimuths.values)
        if len(az) < 2:
            return 360.0
        return float(max(np.diff(az).max(), 360.0 - az[-1] + az[0]))

    return arrivals.dropna(subset=['azimuth']).groupby('event_id')['azimuth'].apply(gap)


def sync_catalog(data: CatalogData) -> CatalogData:
    """Make events, stations, arrivals and picks mutually consistent.

    Drops arrivals/picks of removed events, events without any arrival or pick,
    stations not referenced by any remaining arrival or pick, and recomputes
    ``num_arrivals``/``num_picks``.
    """
    events, arrivals, picks = data.events, data.arrivals, data.picks
    for _ in range(2):  # dropping empty events can orphan nothing further, but be safe
        event_ids = set(events['event_id'])
        arrivals = arrivals[arrivals['event_id'].isin(event_ids)]
        picks = picks[picks['event_id'].isin(event_ids)]
        has_data = set(arrivals['event_id']) | set(picks['event_id'])
        events = events[events['event_id'].isin(has_data)]

    n_arr = arrivals['event_id'].value_counts()
    n_pick = picks['event_id'].value_counts()
    events = events.copy()
    events['num_arrivals'] = events['event_id'].map(n_arr).fillna(0).astype(int)
    events['num_picks'] = events['event_id'].map(n_pick).fillna(0).astype(int)

    used = set(arrivals['station_id']) | set(picks['station_id'])
    stations = data.stations[data.stations['station_id'].isin(used)]
    return CatalogData(events.reset_index(drop=True), stations.reset_index(drop=True),
                       arrivals.reset_index(drop=True), picks.reset_index(drop=True))


def restrict_picks_to_arrivals(data: CatalogData) -> CatalogData:
    """Keep only picks referenced by an arrival (``save_all_picks: false``)."""
    picks = data.picks[data.picks['pick_id'].isin(set(data.arrivals['pick_id']))]
    return CatalogData(data.events, data.stations, data.arrivals, picks)


def _range_mask(series: pd.Series, lo, hi, keep_nan: bool = False) -> pd.Series:
    mask = pd.Series(True, index=series.index)
    if lo is not None:
        mask &= series >= lo
    if hi is not None:
        mask &= series <= hi
    return mask | series.isna() if keep_nan else mask


def _phase_letter(phase: pd.Series) -> pd.Series:
    first = phase.fillna('').astype(str).str.upper().str[:1]
    return first.where(first.isin(['P', 'S']))


def apply_catalog_filters(data: CatalogData, filters, label: str = '') -> CatalogData:
    """Apply a ``filters`` node to a parsed catalog.

    Args:
        data: Unfiltered catalog (not modified).
        filters: Mapping/DictConfig with the filter keys listed in the module docstring.
        label: Name used in the printed stage report.

    Returns:
        New, synchronised CatalogData.
    """
    def get(key):
        return filters.get(key)

    def has(*keys):
        return any(get(k) is not None for k in keys)

    stages: List[Tuple[str, CatalogData]] = [('Initial (raw data)', data)]
    data = sync_catalog(data.copy())

    def record(name):
        stages.append((name, data))

    # Pick evaluation mode
    if get('manual_picks_only'):
        if data.picks.empty:
            pass
        elif 'evaluation_mode' not in data.picks.columns:
            raise ValueError("manual_picks_only=True but 'evaluation_mode' column not found in picks.")
        else:
            picks = data.picks[data.picks['evaluation_mode'] == 'manual']
            arrivals = data.arrivals[data.arrivals['pick_id'].isin(set(picks['pick_id']))]
            data = sync_catalog(CatalogData(data.events, data.stations, arrivals, picks))
        record('manual_picks_only')

    # Event-level: magnitude (events without magnitude are kept), location, depth
    if has('magnitude_min', 'magnitude_max', 'lat_min', 'lat_max', 'lon_min', 'lon_max',
           'depth_min', 'depth_max'):
        ev = data.events
        mask = (_range_mask(ev['magnitude'], get('magnitude_min'), get('magnitude_max'), keep_nan=True)
                & _range_mask(ev['latitude'], get('lat_min'), get('lat_max'))
                & _range_mask(ev['longitude'], get('lon_min'), get('lon_max'))
                & _range_mask(ev['depth'], get('depth_min'), get('depth_max')))
        st = data.stations
        st_mask = (_range_mask(st['latitude'], get('lat_min'), get('lat_max'))
                   & _range_mask(st['longitude'], get('lon_min'), get('lon_max')))
        valid_stations = set(st.loc[st_mask, 'station_id'])
        arrivals = data.arrivals[data.arrivals['station_id'].isin(valid_stations)]
        picks = data.picks[data.picks['station_id'].isin(valid_stations)]
        data = sync_catalog(CatalogData(ev[mask], st, arrivals, picks))
        record('magnitude/location/depth')

    # Distances (arrivals and picks without station coordinates fail an active filter)
    if has('epicentral_distance_min', 'epicentral_distance_max',
           'hypocentral_distance_min', 'hypocentral_distance_max'):
        def distance_ok(df):
            mask = pd.Series(True, index=df.index)
            for kind in ('epicentral', 'hypocentral'):
                lo, hi = get(f'{kind}_distance_min'), get(f'{kind}_distance_max')
                if lo is not None or hi is not None:
                    col = df[f'{kind}_distance']
                    mask &= _range_mask(col, lo, hi) & col.notna()
            return mask
        data = sync_catalog(CatalogData(data.events, data.stations,
                                        data.arrivals[distance_ok(data.arrivals)],
                                        data.picks[distance_ok(data.picks)]))
        record('distance')

    # Event-station pairs must have both a P and an S arrival
    if get('require_both_p_and_s'):
        letters = _phase_letter(data.arrivals['phase'])
        pairs = (data.arrivals.assign(letter=letters).dropna(subset=['letter'])
                 .groupby(['event_id', 'station_id'])['letter'].agg(set))
        good = pairs[pairs.apply(lambda s: {'P', 'S'} <= s)].index
        pair_index = lambda df: pd.MultiIndex.from_frame(df[['event_id', 'station_id']])
        data = sync_catalog(CatalogData(
            data.events, data.stations,
            data.arrivals[pair_index(data.arrivals).isin(good)],
            data.picks[pair_index(data.picks).isin(good)]))
        record('require_both_p_and_s')

    # Azimuthal gap
    if get('azimuthal_gap_max') is not None:
        gap = data.events['event_id'].map(compute_azimuthal_gap(data.arrivals)).fillna(360.0)
        data = sync_catalog(CatalogData(data.events[gap <= get('azimuthal_gap_max')].copy(),
                                        data.stations, data.arrivals, data.picks))
        record('azimuthal_gap_max')

    # Picks per event (counts reflect the picks that survived the filters above)
    if has('min_picks_per_event', 'max_picks_per_event'):
        mask = _range_mask(data.events['num_picks'], get('min_picks_per_event'),
                           get('max_picks_per_event'))
        data = sync_catalog(CatalogData(data.events[mask], data.stations,
                                        data.arrivals, data.picks))
        record('picks_per_event')

    _print_stage_report(label, stages)
    return data


def _print_stage_report(label: str, stages: List[Tuple[str, CatalogData]]) -> None:
    print(f"\nFilter summary{f' ({label})' if label else ''}:")
    print(f"  {'Stage':<28}{'Events':>10}{'Stations':>10}{'Arrivals':>12}{'Picks':>12}")
    for name, d in stages:
        print(f"  {name:<28}{len(d.events):>10}{len(d.stations):>10}"
              f"{len(d.arrivals):>12}{len(d.picks):>12}")
    if len(stages) == 1:
        print("  (no filters active)")
