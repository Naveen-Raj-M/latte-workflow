"""Raw data visualization of QuakeML catalogs.

Works on the catalog parsed once by quakeml_loader. The module applies its own
``raw_data_visualization.filters`` (each defaulting to the global ``filters``),
then produces:

- source/receiver map view and depth cross-sections
- histograms: source/receiver location, time, magnitude, epicentral and
  hypocentral distance, picks per event, magnitude vs picks
- cumulative azimuthal gap distribution

Styling comes from the unified ``plots`` section, with optional overrides in
``raw_data_visualization.plots``.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from preprocess.catalog_filters import apply_catalog_filters, compute_azimuthal_gap
from preprocess.plot_settings import resolve_plot_settings, save_figure
from preprocess.quakeml_loader import CatalogData
from preprocess.visualization import plot_station_event_views
from utils.io_utils import ensure_directory_exists

STATS_BOX = dict(boxstyle='round', facecolor='wheat', alpha=0.8)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _new_axes(ps: DictConfig, columns: int = 1):
    """Histogram-style figure; two-panel figures are 1.5x wider."""
    width, height = ps.figure.histogram_size
    return plt.subplots(1, columns, figsize=(width * (1.5 if columns == 2 else 1), height))


def _style_axes(ax, ps: DictConfig, xlabel: str, ylabel: str, title: str,
                grid_axis: str = 'both') -> None:
    fs, fw = ps.fontsize, ps.fontweight
    ax.set_xlabel(xlabel, fontsize=fs.axis_label, fontweight=fw.axis_label)
    ax.set_ylabel(ylabel, fontsize=fs.axis_label, fontweight=fw.axis_label)
    ax.set_title(title, fontsize=fs.title, fontweight=fw.title)
    ax.tick_params(axis='both', labelsize=fs.axis_ticks)
    ax.grid(True, alpha=0.3, linestyle='--', axis=grid_axis)


def _stats_box(ax, ps: DictConfig, text: str, right: bool = True) -> None:
    if ps.histograms.show_statistics:
        ax.text(0.98 if right else 0.02, 0.97, text, transform=ax.transAxes,
                fontsize=ps.fontsize.statistics, va='top',
                ha='right' if right else 'left', bbox=STATS_BOX)


def _describe(values: np.ndarray, unit: str = '', fmt: str = '.2f') -> str:
    u = f' {unit}' if unit else ''
    return (f'Mean: {values.mean():{fmt}}{u}\nMedian: {np.median(values):{fmt}}{u}\n'
            f'Std Dev: {values.std():{fmt}}{u}\n'
            f'Range: [{values.min():{fmt}}, {values.max():{fmt}}]{u}')


def _report(path: Path, what: str) -> None:
    print(f"  Saved {what}: {path}")


def _histogram(ax, values, ps: DictConfig, color: str, edge_width: float = 0.5, **kwargs):
    return ax.hist(values, bins=kwargs.pop('bins', ps.histograms.bins), alpha=ps.histograms.alpha,
                   color=color, edgecolor='black', linewidth=edge_width, **kwargs)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_source_receiver_histograms(data: CatalogData, out: Path, ps: DictConfig, rv: DictConfig) -> None:
    """Latitude, longitude and depth histograms for sources and receivers."""
    quantities = [('latitude', 'Latitude (degrees)', 'Latitude'),
                  ('longitude', 'Longitude (degrees)', 'Longitude'),
                  ('depth', None, 'Depth')]
    depth_labels = {'sources': 'Depth (km, positive = below MSL)',
                    'receivers': 'Depth (km, negative = above MSL)'}
    for who, df, color, noun in (('sources', data.events, ps.histograms.colors.sources, 'Sources (Events)'),
                                 ('receivers', data.stations, ps.histograms.colors.receivers,
                                  'Receivers (Stations)')):
        if df.empty:
            continue
        for column, xlabel, name in quantities:
            fig, ax = _new_axes(ps)
            _histogram(ax, df[column].dropna(), ps, color)
            _style_axes(ax, ps, xlabel or depth_labels[who], 'Count',
                        f'{name} Distribution: {noun}')
            fig.tight_layout()
            _report(save_figure(fig, out, f'histogram_{column}_{who}', ps), f'{name.lower()} histogram')


_TIME_BINS = {'daily': ('D', 'Date', 1), 'weekly': ('W', 'Week', 7),
              'monthly': ('ME', 'Month', 28), 'yearly': ('YE', 'Year', 350)}


def plot_temporal_histogram(data: CatalogData, out: Path, ps: DictConfig, rv: DictConfig) -> None:
    """Number of events per day/week/month/year."""
    if data.events.empty:
        return
    binning = rv.histogram_temporal.time_binning
    freq, xlabel, width_days = _TIME_BINS.get(binning, _TIME_BINS['monthly'])
    times = pd.to_datetime(data.events['origin_time'], utc=True, errors='coerce').dropna()
    counts = pd.Series(1, index=times.dt.tz_localize(None)).resample(freq).sum()

    fig, ax = _new_axes(ps)
    ax.bar(counts.index, counts.values, width=width_days, color=ps.histograms.colors.temporal,
           edgecolor='black', linewidth=0.5)
    _style_axes(ax, ps, xlabel, 'Number of Events',
                f'Temporal Distribution of Seismic Events ({binning.capitalize()})', grid_axis='y')
    plt.setp(ax.get_xticklabels(), rotation=ps.histograms.rotate_time_labels, ha='right')
    fig.tight_layout()
    _report(save_figure(fig, out, f'histogram_temporal_{binning}', ps), 'temporal histogram')


def plot_magnitude_histogram(data: CatalogData, out: Path, ps: DictConfig, rv: DictConfig) -> None:
    """Magnitude distribution (log count axis)."""
    mags = data.events['magnitude'].dropna().values
    if len(mags) == 0:
        print("  No events with magnitude values to plot.")
        return
    fig, ax = _new_axes(ps)
    _histogram(ax, mags, ps, ps.histograms.colors.magnitude, edge_width=1.2)
    ax.set_yscale('log')
    _style_axes(ax, ps, 'Magnitude', 'Count', 'Magnitude Distribution')
    _stats_box(ax, ps, f'Total Events: {len(mags)}\n' + _describe(mags))
    fig.tight_layout()
    _report(save_figure(fig, out, 'histogram_magnitude', ps), 'magnitude histogram')


def _plot_distance_per_magnitude_bin(data: CatalogData, out: Path, ps: DictConfig, rv: DictConfig,
                                     column: str, label: str) -> None:
    """One distance histogram per magnitude bin, starting at filters.magnitude_min."""
    settings = rv.distance_histograms
    mag_min = rv.filters.get('magnitude_min')
    arrivals = data.arrivals.merge(data.events.dropna(subset=['magnitude'])[['event_id', 'magnitude']],
                                   on='event_id').dropna(subset=[column])
    if arrivals.empty:
        print(f"  No arrivals with valid {label.lower()} and magnitude for per-bin plots.")
        return

    size = settings.magnitude_bin_size
    start = float(mag_min) if mag_min is not None else 0.0
    edges = np.arange(start, arrivals['magnitude'].max() + size, size)
    short = column.replace('_distance', '')
    subdir = out / f'{short}_per_magnitude_bin'
    ensure_directory_exists(subdir)

    saved = 0
    for low, high in zip(edges[:-1], edges[1:]):
        distances = arrivals.loc[(arrivals['magnitude'] >= low) & (arrivals['magnitude'] < high), column].values
        if len(distances) == 0:
            continue
        fig, ax = _new_axes(ps)
        _histogram(ax, distances, ps, ps.histograms.colors[f'{short}_distance'].histogram)
        _style_axes(ax, ps, f'{label} (km)', 'Count',
                    f'{label} Distribution\nMagnitude: [{low:.1f}, {high:.1f})', grid_axis='y')
        ax.axvline(distances.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {distances.mean():.2f} km')
        ax.axvline(np.median(distances), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(distances):.2f} km')
        ax.legend(fontsize=ps.fontsize.legend)
        _stats_box(ax, ps, f'Total Arrivals: {len(distances)}\n' + _describe(distances, 'km'))
        fig.tight_layout()
        save_figure(fig, subdir, f'histogram_{short}_mag_{low:.1f}_{high:.1f}', ps)
        saved += 1
    print(f"  Saved {saved} per-magnitude-bin {label.lower()} histograms to: {subdir}")


def _plot_distance_histogram(data: CatalogData, out: Path, ps: DictConfig, rv: DictConfig,
                             column: str, label: str) -> None:
    distances = data.arrivals[column].dropna().values
    if len(distances) == 0:
        print(f"  No arrivals with valid {label.lower()} values to plot.")
        return
    settings = rv.distance_histograms
    plot_type = settings.plot_type
    colors = ps.histograms.colors[column]
    median, p90 = np.median(distances), np.percentile(distances, 90)

    fig, axes = _new_axes(ps, 2 if plot_type == 'both' else 1)
    axes = np.atleast_1d(axes)
    hist_ax = axes[0] if plot_type in ('histogram', 'both') else None
    cum_ax = axes[-1] if plot_type in ('cumulative', 'both') else None

    if hist_ax is not None:
        _histogram(hist_ax, distances, ps, colors.histogram)
        _style_axes(hist_ax, ps, f'{label} (km)', 'Count', f'{label} Distribution', grid_axis='y')
        hist_ax.axvline(distances.mean(), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {distances.mean():.2f} km')
        hist_ax.axvline(median, color='orange', linestyle='--', linewidth=2,
                        label=f'Median: {median:.2f} km')
        hist_ax.legend(fontsize=ps.fontsize.legend)
        _stats_box(hist_ax, ps, f'Total Arrivals: {len(distances)}\n' + _describe(distances, 'km'))
    if cum_ax is not None:
        ordered = np.sort(distances)
        cumulative = np.arange(1, len(ordered) + 1) / len(ordered) * 100
        cum_ax.plot(ordered, cumulative, color=colors.cumulative, linewidth=2)
        cum_ax.fill_between(ordered, cumulative, alpha=0.3, color=colors.cumulative)
        _style_axes(cum_ax, ps, f'{label} (km)', 'Cumulative Percentage (%)',
                    f'Cumulative {label} Distribution')
        cum_ax.axvline(median, color='orange', linestyle='--', linewidth=1.5,
                       label=f'Median: {median:.2f} km')
        cum_ax.axvline(p90, color='red', linestyle='--', linewidth=1.5, label=f'90th %ile: {p90:.2f} km')
        cum_ax.legend(fontsize=ps.fontsize.legend)

    fig.tight_layout()
    _report(save_figure(fig, out, f'histogram_{column}_{plot_type}', ps), f'{label.lower()} plot')

    if settings.enable_magnitude_bin:
        _plot_distance_per_magnitude_bin(data, out, ps, rv, column, label)


def plot_epicentral_distance_histogram(data, out, ps, rv) -> None:
    """Epicentral distance histogram and/or cumulative distribution."""
    _plot_distance_histogram(data, out, ps, rv, 'epicentral_distance', 'Epicentral Distance')


def plot_hypocentral_distance_histogram(data, out, ps, rv) -> None:
    """Hypocentral distance histogram and/or cumulative distribution."""
    _plot_distance_histogram(data, out, ps, rv, 'hypocentral_distance', 'Hypocentral Distance')


def plot_picks_per_event_histogram(data: CatalogData, out: Path, ps: DictConfig, rv: DictConfig) -> None:
    """Distribution of the number of picks per event."""
    counts = data.picks.groupby('event_id').size().values
    if len(counts) == 0:
        print("  No picks to plot.")
        return
    fig, ax = _new_axes(ps)
    _histogram(ax, counts, ps, ps.histograms.colors.picks_per_event, edge_width=1.2)
    _style_axes(ax, ps, 'Picks per Event', 'Count', 'Picks per Event Distribution')
    _stats_box(ax, ps, f'Total Events: {len(counts)}\n' + _describe(counts, fmt='.1f'))
    fig.tight_layout()
    _report(save_figure(fig, out, 'histogram_picks_per_event', ps), 'picks per event histogram')


def plot_azimuthal_gap_cumulative(data: CatalogData, out: Path, ps: DictConfig, rv: DictConfig) -> None:
    """Cumulative distribution of the per-event azimuthal gap."""
    gaps = compute_azimuthal_gap(data.arrivals).values
    if len(gaps) == 0:
        print("  No events with azimuthal gap values to plot.")
        return
    style = ps.histograms.azimuthal_gap
    ordered = np.sort(gaps)
    fig, ax = _new_axes(ps)
    ax.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered) * 100,
            linewidth=style.line_width, color=style.line_color, linestyle=style.line_style,
            label='Cumulative Distribution')
    if style.show_percentiles:
        for q, color in ((25, 'gray'), (50, 'black'), (75, 'gray')):
            ax.axvline(np.percentile(gaps, q), color=color, linestyle='--', linewidth=1.5,
                       alpha=0.7, label=f'{q}th percentile: {np.percentile(gaps, q):.1f}°')
    _style_axes(ax, ps, 'Azimuthal Gap (degrees)', 'Cumulative Percentage (%)',
                'Azimuthal Gap Cumulative Distribution')
    ax.legend(loc='lower right', fontsize=ps.fontsize.legend)
    ax.set_ylim(0, 100)
    _stats_box(ax, ps, f'Total Events: {len(gaps)}\n' + _describe(gaps, fmt='.1f'), right=False)
    fig.tight_layout()
    _report(save_figure(fig, out, 'cumulative_azimuthal_gap', ps), 'azimuthal gap plot')


def plot_magnitude_picks_histogram(data: CatalogData, out: Path, ps: DictConfig, rv: DictConfig) -> None:
    """Total picks per magnitude bin and cumulative share of picks vs magnitude."""
    picks_per_event = data.picks.groupby('event_id').size().rename('n_picks')
    events = (data.events.dropna(subset=['magnitude']).merge(picks_per_event, on='event_id')
              .sort_values('magnitude'))
    if events.empty:
        print("  No events with valid magnitude and picks to plot.")
        return
    mags, n_picks = events['magnitude'].values, events['n_picks'].values
    plot_type = rv.histogram_magnitude_picks.plot_type
    colors = ps.histograms.colors.magnitude_picks

    fig, axes = _new_axes(ps, 2 if plot_type == 'both' else 1)
    axes = np.atleast_1d(axes)
    hist_ax = axes[0] if plot_type in ('histogram', 'both') else None
    cum_ax = axes[-1] if plot_type in ('cumulative', 'both') else None

    if hist_ax is not None:
        edges = np.histogram_bin_edges(mags, bins=ps.histograms.bins)
        totals, _ = np.histogram(mags, bins=edges, weights=n_picks)
        hist_ax.bar((edges[:-1] + edges[1:]) / 2, totals, width=np.diff(edges) * 0.9,
                    alpha=ps.histograms.alpha, color=colors.histogram, edgecolor='black', linewidth=0.5)
        _style_axes(hist_ax, ps, 'Magnitude', 'Total Picks', 'Magnitude vs Picks Distribution', grid_axis='y')
        hist_ax.axvline(mags.mean(), color='red', linestyle='--', linewidth=2,
                        label=f'Mean Mag: {mags.mean():.2f}')
        hist_ax.axvline(np.median(mags), color='orange', linestyle='--', linewidth=2,
                        label=f'Median Mag: {np.median(mags):.2f}')
        hist_ax.legend(fontsize=ps.fontsize.legend)
        _stats_box(hist_ax, ps,
                   f'Total Events: {len(events)}\nTotal Picks: {n_picks.sum()}\n'
                   f'Mean Mag: {mags.mean():.2f}\nMedian Mag: {np.median(mags):.2f}\n'
                   f'Avg Picks/Event: {n_picks.mean():.1f}\nMedian Picks/Event: {np.median(n_picks):.1f}')
    if cum_ax is not None:
        share = np.cumsum(n_picks) / n_picks.sum() * 100
        cum_ax.plot(mags, share, color=colors.cumulative, linewidth=2)
        cum_ax.fill_between(mags, share, alpha=0.3, color=colors.cumulative)
        _style_axes(cum_ax, ps, 'Magnitude', 'Cumulative Percentage of Picks (%)',
                    'Cumulative Picks vs Magnitude')
        for pct, color in ((50, 'orange'), (90, 'red')):
            mag_at = mags[np.argmin(np.abs(share - pct))]
            cum_ax.axvline(mag_at, color=color, linestyle='--', linewidth=1.5,
                           label=f'{pct}% picks: Mag {mag_at:.2f}')
        cum_ax.legend(fontsize=ps.fontsize.legend)

    fig.tight_layout()
    _report(save_figure(fig, out, f'histogram_magnitude_picks_{plot_type}', ps), 'magnitude vs picks plot')


def plot_source_receiver_maps(data: CatalogData, out: Path, ps: DictConfig, rv: DictConfig) -> None:
    """Map view and depth cross-sections of sources and receivers."""
    primary = None
    if rv.source_receiver_maps.plot_picks:
        primary = set(data.arrivals['station_id'])  # others are "picks only" receivers
    receivers = data.stations if rv.source_receiver_maps.plot_picks else data.stations[
        data.stations['station_id'].isin(set(data.arrivals['station_id']))]
    plot_station_event_views(data.events, receivers, data.picks, out, ps,
                             filters=rv.filters, primary_station_ids=primary)


PLOT_FUNCTIONS: Dict[str, Callable] = {
    'source_receiver_maps': plot_source_receiver_maps,
    'histogram_source_receivers': plot_source_receiver_histograms,
    'histogram_temporal': plot_temporal_histogram,
    'histogram_magnitude': plot_magnitude_histogram,
    'histogram_epicentral_distance': plot_epicentral_distance_histogram,
    'histogram_hypocentral_distance': plot_hypocentral_distance_histogram,
    'histogram_picks_per_event': plot_picks_per_event_histogram,
    'cumulative_azimuthal_gap': plot_azimuthal_gap_cumulative,
    'histogram_magnitude_picks': plot_magnitude_picks_histogram,
}


def create_raw_data_visualization(config: DictConfig, raw: CatalogData) -> Optional[Path]:
    """Filter the parsed QuakeML catalog and generate the selected raw data plots.

    Args:
        config: Full configuration.
        raw: Unfiltered catalog from load_quakeml_catalog().

    Returns:
        Output directory, or None when no events passed the filters.
    """
    rv = config.raw_data_visualization
    ps = resolve_plot_settings(config, rv)
    out = Path(rv.output_dir)
    ensure_directory_exists(out)

    data = apply_catalog_filters(raw, rv.filters, label='raw_data_visualization')
    print(f"\nRaw data after filtering: {data.summary()}")
    if data.events.empty:
        print("No events passed raw_data_visualization.filters. Adjust the filters.")
        return None

    selected = [name for name, enabled in rv.plots_to_generate.items() if enabled]
    for name in selected:
        print(f"\nGenerating {name}...")
        PLOT_FUNCTIONS[name](data, out, ps, rv)
    if not selected:
        print("No plots selected in raw_data_visualization.plots_to_generate.")
    return out
