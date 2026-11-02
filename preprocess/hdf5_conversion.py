"""QuakeML to HDF5 conversion.

Writes the (filtered) catalog parsed by quakeml_loader to an HDF5 file with the
tables ``events``, ``stations``, ``picks`` and ``arrivals`` that
data_loader.load_from_hdf5() and the raw data visualization read.
"""

from pathlib import Path

import h5py
import pandas as pd
from omegaconf import DictConfig

from preprocess.catalog_filters import (
    apply_catalog_filters,
    compute_azimuthal_gap,
    restrict_picks_to_arrivals,
    sync_catalog,
)
from preprocess.quakeml_loader import CatalogData


def get_phase_type(phase_hint: pd.Series) -> pd.Series:
    """'P' for P-type phases (P, Pg, Pn, pP, ...), 'S' for S-type, else NaN."""
    upper = phase_hint.fillna('').astype(str).str.upper()
    is_p = upper.str.startswith('P') | upper.str.endswith('P')
    is_s = ~is_p & (upper.str.startswith('S') | upper.str.endswith('S'))
    return pd.Series(pd.NA, index=phase_hint.index, dtype=object).mask(is_p, 'P').mask(is_s, 'S')


def select_single_pick_per_phase(picks: pd.DataFrame) -> pd.DataFrame:
    """Keep one P-type and one S-type pick per event-station pair.

    Priority: exact P/S over variants (Pg, Pn, Sg, Sn, ...), then
    manual > automatic > unknown evaluation mode. Unclassifiable picks are dropped.
    """
    if picks.empty:
        return picks
    picks = picks.assign(_type=get_phase_type(picks['phase_hint'])).dropna(subset=['_type'])
    exact = picks['phase_hint'].isin(['P', 'S'])
    mode_priority = picks['evaluation_mode'].map({'manual': 3, 'automatic': 2}).fillna(1)
    picks = picks.assign(_score=exact.map({True: 10, False: 5}) * 10 + mode_priority)
    picks = picks.sort_values('_score', ascending=False, kind='stable')
    picks = picks.drop_duplicates(['event_id', 'station_id', '_type'])
    return picks.drop(columns=['_type', '_score']).sort_index()


def _select_fields(df: pd.DataFrame, fields) -> pd.DataFrame:
    if not fields or df.empty:
        return df
    return df[[col for col in fields if col in df.columns]]


def _write_tables(tables: dict, output_file: Path) -> None:
    try:
        for i, (key, df) in enumerate(tables.items()):
            df.to_hdf(output_file, key=key, mode='w' if i == 0 else 'a',
                      complevel=9, complib='zlib')
    except Exception as e:
        print(f"  Warning: Compression failed ({e}), saving without compression...")
        for i, (key, df) in enumerate(tables.items()):
            df.to_hdf(output_file, key=key, mode='w' if i == 0 else 'a')


def convert_to_hdf5(config: DictConfig, raw: CatalogData) -> Path:
    """Filter the parsed catalog with ``hdf5_conversion.filters`` and write it to HDF5.

    Args:
        config: Full configuration.
        raw: Unfiltered catalog from load_quakeml_catalog().

    Returns:
        Path of the written HDF5 file.
    """
    cfg = config.hdf5_conversion
    output_file = Path(cfg.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting QuakeML catalog to HDF5: {output_file}")
    data = raw.copy()

    # Pick preparation: runs before the filters so that events left without any
    # data are dropped by the filter step.
    if cfg.phase_hint_from_arrival and not data.picks.empty and not data.arrivals.empty:
        # Many picks carry no phase_hint while the arrival always names the phase
        arrival_phase = (data.arrivals.drop_duplicates('pick_id')
                         .set_index('pick_id')['phase'])
        missing = data.picks['phase_hint'] == ''
        data.picks.loc[missing, 'phase_hint'] = (
            data.picks.loc[missing, 'pick_id'].map(arrival_phase).fillna(''))
    if cfg.require_station_coordinates:
        # Picks of stations absent from the station list leave dangling references
        known = set(data.stations['station_id'])
        data.arrivals = data.arrivals[data.arrivals['station_id'].isin(known)]
        data.picks = data.picks[data.picks['station_id'].isin(known)]
    if not cfg.save_all_picks:
        data = restrict_picks_to_arrivals(data)

    data = apply_catalog_filters(data, cfg.filters, label='hdf5_conversion')

    if cfg.single_pick_per_phase:
        picks = select_single_pick_per_phase(data.picks)
        arrivals = data.arrivals[data.arrivals['pick_id'].isin(set(picks['pick_id']))]
        data = CatalogData(data.events, data.stations, arrivals, picks)
    data = sync_catalog(data)

    if data.events.empty:
        print("  No events passed the filters; nothing written. Adjust hdf5_conversion.filters.")
        return output_file

    events = data.events
    if 'azimuthal_gap' in (cfg.event_fields or []):
        events = events.assign(azimuthal_gap=events['event_id'].map(
            compute_azimuthal_gap(data.arrivals)).fillna(360.0))

    tables = {'events': _select_fields(events, cfg.event_fields)}
    for key, df, fields in (('picks', data.picks, cfg.pick_fields),
                            ('arrivals', data.arrivals, cfg.arrival_fields),
                            ('stations', data.stations, cfg.station_fields)):
        if not df.empty:
            tables[key] = _select_fields(df, fields)
    _write_tables(tables, output_file)

    with h5py.File(output_file, 'a') as f:
        f['events'].attrs['description'] = 'Seismic event catalog'
        f['events'].attrs['depth_convention'] = 'positive = below MSL (km)'
        if 'stations' in tables:
            f['stations'].attrs['description'] = 'Station inventory'
            f['stations'].attrs['depth_convention'] = 'negative = above MSL (km)'

    ev = data.events
    print(f"  Events: {len(ev)}  Picks: {len(data.picks)}  "
          f"Arrivals: {len(data.arrivals)}  Stations: {len(data.stations)}")
    print(f"  Latitude: {ev['latitude'].min():.2f}° - {ev['latitude'].max():.2f}°, "
          f"Longitude: {ev['longitude'].min():.2f}° - {ev['longitude'].max():.2f}°, "
          f"Depth: {ev['depth'].min():.2f} - {ev['depth'].max():.2f} km")
    if ev['magnitude'].notna().any():
        print(f"  Magnitude: {ev['magnitude'].min():.2f} - {ev['magnitude'].max():.2f} "
              f"(mean {ev['magnitude'].mean():.2f})")
    if ev['catalog'].nunique() > 1:
        print("  Events per catalog: " + ", ".join(
            f"{k}: {v}" for k, v in ev['catalog'].value_counts().items()))
    print(f"  Saved: {output_file}")
    return output_file
