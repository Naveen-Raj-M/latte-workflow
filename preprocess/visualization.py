"""Visualization functions for seismic data and velocity models.

This module provides functions for visualizing:
- Station and event locations (map view and cross-sections); shared by the
  raw data visualization and the final data visualization
- Velocity model (depth slices and profiles)

All styling comes from the unified ``plots`` configuration section
(see plot_settings.resolve_plot_settings).
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Set

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.interpolate import griddata as scipy_griddata

from preprocess.plot_settings import resolve_plot_settings, save_figure
from utils.io_utils import ensure_directory_exists
from utils.velocity_io import read_velocity_file

KM_PER_DEG = 111.0

# view -> (x column, y column, x label, title, event colour column, colormap, colorbar label)
_VIEWS = {
    'map': ('longitude', 'latitude', 'Longitude (°)', 'Event Locations - Map View',
            'depth', 'hot_r', 'Depth (km)'),
    'lat_depth': ('latitude', 'depth', 'Latitude (°)', 'Event Locations - Latitude vs Depth',
                  'magnitude', 'viridis', 'Magnitude'),
    'lon_depth': ('longitude', 'depth', 'Longitude (°)', 'Event Locations - Longitude vs Depth',
                  'magnitude', 'viridis', 'Magnitude'),
}
_VIEW_FILES = {'map': 'map_view_lat_lon', 'lat_depth': 'cross_section_lat_depth',
               'lon_depth': 'cross_section_lon_depth'}


def _load_basin_boundaries(basin_config: DictConfig) -> Dict:
    """Load basin boundary polygons and resolve per-basin styles.

    Args:
        basin_config: ``plots.basin_boundaries`` settings. Each basin entry may
            override any style key (line_color, line_width, line_style,
            line_alpha, label_color, label_fontsize, label_fontweight,
            label_offset_x, label_offset_y); ``label: null`` hides its label.

    Returns:
        Dictionary basin name -> {'lons', 'lats', 'style': dict, 'label': str|None}.
    """
    if not basin_config.get('enabled', False):
        return {}

    style_keys = ['line_color', 'line_width', 'line_style', 'line_alpha', 'label_color',
                  'label_fontsize', 'label_fontweight', 'label_offset_x', 'label_offset_y']
    basin_data = {}
    data_dir = Path(basin_config.data_dir)

    for basin in basin_config.basins:
        csv_file = data_dir / basin['file']
        if not csv_file.exists():
            logging.getLogger(__name__).warning(f"Basin boundary file not found: {csv_file}")
            continue

        # Space-delimited file (despite .csv extension)
        df = pd.read_csv(csv_file, sep=r'\s+')
        style = {k: basin.get(k, basin_config.get(k)) for k in style_keys}
        if style['label_color'] is None:
            style['label_color'] = style['line_color']
        name = basin.get('label') or basin['file'].replace('.csv', '')
        basin_data[name] = {'lons': df['POINT_X'].values, 'lats': df['POINT_Y'].values,
                            'style': style,
                            'label': basin.get('label') if basin_config.get('show_labels', True) else None}
    return basin_data


def _plot_basin_boundary(ax, basin_data: Dict, xlim=None, ylim=None) -> None:
    """Plot basin boundaries (and labels inside the axis limits) on a map view axis."""
    for data in basin_data.values():
        lons, lats, style = data['lons'], data['lats'], data['style']
        ax.plot(lons, lats, color=style['line_color'], linewidth=style['line_width'],
                linestyle=style['line_style'], alpha=style['line_alpha'], zorder=4)

        if data['label']:
            label_x = np.mean(lons) + style['label_offset_x']
            label_y = np.mean(lats) + style['label_offset_y']
            inside = ((xlim is None or xlim[0] <= label_x <= xlim[1]) and
                      (ylim is None or ylim[0] <= label_y <= ylim[1]))
            if inside:
                ax.text(label_x, label_y, data['label'], fontsize=style['label_fontsize'],
                        fontweight=style['label_fontweight'], color=style['label_color'],
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor=style['label_color'], alpha=0.7),
                        zorder=5)


def _normalize_coordinate_names(df: pd.DataFrame) -> pd.DataFrame:
    """Accept 'lat'/'lon' as aliases of 'latitude'/'longitude'."""
    return df.rename(columns={'lat': 'latitude', 'lon': 'longitude'}) if (
        'lat' in df.columns or 'lon' in df.columns) else df


def _pick_segments(view: str, picks_df: pd.DataFrame, events_df: pd.DataFrame,
                   stations_df: pd.DataFrame) -> np.ndarray:
    """Event-to-station segments (N, 2, 2) for the unique event-station pairs with picks."""
    x_col, y_col = _VIEWS[view][0], _VIEWS[view][1]
    pairs = picks_df[['event_id', 'station_id']].drop_duplicates()
    event_xy = events_df.set_index('event_id')[[x_col, y_col]]
    station_xy = stations_df.drop_duplicates('station_id').set_index('station_id')[[x_col, y_col]]
    merged = (pairs.merge(event_xy, left_on='event_id', right_index=True)
                   .merge(station_xy, left_on='station_id', right_index=True,
                          suffixes=('_event', '_station')))
    if merged.empty:
        return np.empty((0, 2, 2))
    return np.stack([merged[[f'{x_col}_event', f'{y_col}_event']].values,
                     merged[[f'{x_col}_station', f'{y_col}_station']].values], axis=1)


def _plot_view(view: str, events_df: pd.DataFrame, stations_df: pd.DataFrame,
               picks_df: Optional[pd.DataFrame], output_dir: Path, settings: DictConfig,
               filters, origin: Optional[tuple], primary_station_ids: Optional[Set[str]],
               receiver_label: str) -> Path:
    """Create one event/receiver view (map, latitude-depth or longitude-depth)."""
    x_col, y_col, xlabel, title, color_col, cmap, cbar_label = _VIEWS[view]
    fs = settings.fontsize
    ev, rcv = settings.events, settings.receivers

    fig, ax = plt.subplots(figsize=tuple(settings.figure.size))

    # Receivers, optionally split into arrival receivers and "picks only" receivers
    receivers = stations_df.drop_duplicates(subset=['longitude', 'latitude', 'depth'])
    if primary_station_ids is None:
        groups = [(receivers, rcv.color, f'{receiver_label} (n={len(receivers)})')]
    else:
        primary = receivers[receivers['station_id'].isin(primary_station_ids)]
        picks_only = receivers[~receivers['station_id'].isin(primary_station_ids)]
        groups = [(primary, rcv.color, f'{receiver_label} - Arrivals (n={len(primary)})'),
                  (picks_only, rcv.picks_only_color, f'{receiver_label} - Picks Only (n={len(picks_only)})')]
    for group, color, label in groups:
        if len(group) > 0:
            ax.scatter(group[x_col], group[y_col], s=rcv.size, c=color, marker=rcv.marker,
                       alpha=0.8, edgecolors='black', linewidths=0.5, label=label, zorder=2)

    # Events: sized by magnitude, coloured by depth (map) / magnitude (cross-sections)
    if len(events_df) > 0:
        mags = events_df['magnitude'].fillna(3.0).values
        sizes = mags**2 * ev.size / 9
        label = f'Events (n={len(events_df)})'
        if ev.show_colorbar:
            scatter = ax.scatter(events_df[x_col], events_df[y_col], s=sizes,
                                 c=events_df[color_col].fillna(3.0) if color_col == 'magnitude'
                                 else events_df[color_col],
                                 cmap=cmap, marker=ev.marker, alpha=ev.alpha,
                                 edgecolors='black', linewidths=0.5, label=label, zorder=3,
                                 **({'vmin': 0} if view == 'map' else {}))
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label, fontsize=fs.colorbar_label)
            cbar.ax.tick_params(labelsize=fs.colorbar_ticks)
        else:
            ax.scatter(events_df[x_col], events_df[y_col], s=sizes, c=ev.color,
                       marker=ev.marker, alpha=ev.alpha, edgecolors='black',
                       linewidths=0.5, label=label, zorder=3)

    # Pick lines (LineCollection for speed)
    lines = settings.pick_lines
    if lines.enabled and picks_df is not None and not picks_df.empty:
        segments = _pick_segments(view, picks_df, events_df, stations_df)
        if len(segments) > 0:
            ax.add_collection(LineCollection(segments, colors=lines.color, alpha=lines.alpha,
                                             linewidths=lines.linewidth, zorder=1,
                                             rasterized=True))

    ax.set_xlabel(xlabel, fontsize=fs.axis_label, fontweight=settings.fontweight.axis_label)
    ax.set_ylabel('Latitude (°)' if view == 'map' else 'Depth (km)', fontsize=fs.axis_label,
                  fontweight=settings.fontweight.axis_label)
    ax.set_title(title, fontsize=fs.title, fontweight=settings.fontweight.title)
    ax.tick_params(axis='both', labelsize=fs.axis_ticks)
    ax.grid(True, alpha=0.3)
    if view != 'map':
        ax.invert_yaxis()  # depth increases downward
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=fs.legend)

    _apply_limits(ax, view, events_df, stations_df, filters, origin, settings)

    if view == 'map':
        basin_data = _load_basin_boundaries(settings.basin_boundaries)
        if basin_data:
            _plot_basin_boundary(ax, basin_data, xlim=ax.get_xlim(), ylim=ax.get_ylim())

    fig.tight_layout()
    output_file = save_figure(fig, output_dir, _VIEW_FILES[view], settings)
    logging.getLogger(__name__).info(f"Saved: {output_file}")
    return output_file


def _apply_limits(ax, view: str, events_df, stations_df, filters, origin, settings) -> None:
    """Axis limits from the data extent, overridden by the filters' bounds."""
    filters = filters or {}
    lon_lo, lon_hi = filters.get('lon_min'), filters.get('lon_max')
    lat_lo, lat_hi = filters.get('lat_min'), filters.get('lat_max')

    all_lat = np.concatenate([events_df['latitude'].values.astype(float),
                              stations_df['latitude'].values.astype(float)])
    mean_lat = float(np.mean(all_lat)) if len(all_lat) else 32.0

    if view == 'map':
        all_lon = np.concatenate([events_df['longitude'].values.astype(float),
                                  stations_df['longitude'].values.astype(float)])
        if len(all_lon) > 0:
            lons, lats = all_lon, all_lat
            if origin is not None:  # model domain starts at the Cartesian origin
                lons, lats = np.append(lons, origin[0]), np.append(lats, origin[1])
            lon_margin = max((lons.max() - lons.min()) * 0.02, 0.01)
            lat_margin = max((lats.max() - lats.min()) * 0.02, 0.01)
            xlim = [lons.min() - lon_margin, lons.max() + lon_margin]
            ylim = [lats.min() - lat_margin, lats.max() + lat_margin]
        else:
            xlim, ylim = list(ax.get_xlim()), list(ax.get_ylim())
        for lim, lo, hi in ((xlim, lon_lo, lon_hi), (ylim, lat_lo, lat_hi)):
            if lo is not None:
                lim[0] = lo
            if hi is not None:
                lim[1] = hi
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if settings.equal_aspect:
            # 1 degree of latitude spans 1/cos(lat) times more km than 1 degree of longitude
            ax.set_aspect(1.0 / np.cos(np.radians(mean_lat)), adjustable='box')
        return

    lo, hi = (lon_lo, lon_hi) if view == 'lon_depth' else (lat_lo, lat_hi)
    if lo is not None or hi is not None:
        cur = ax.get_xlim()
        ax.set_xlim(lo if lo is not None else cur[0], hi if hi is not None else cur[1])

    depth_lo, depth_hi = filters.get('depth_min'), filters.get('depth_max')
    bottom = settings.max_depth if settings.max_depth is not None else depth_hi
    if bottom is not None or depth_lo is not None:
        cur = ax.get_ylim()  # inverted axis: (deepest, shallowest)
        ax.set_ylim(bottom if bottom is not None else cur[0],
                    depth_lo if depth_lo is not None else cur[1])
    if settings.equal_aspect:
        km_per_deg = KM_PER_DEG * (np.cos(np.radians(mean_lat)) if view == 'lon_depth' else 1.0)
        ax.set_aspect(1.0 / km_per_deg, adjustable='box')


def create_station_event_visualization(config: DictConfig,
                                       events_df: pd.DataFrame,
                                       stations_df: pd.DataFrame,
                                       picks_df: Optional[pd.DataFrame] = None) -> None:
    """Create the final data visualization of stations and events.

    Creates three plots: map view, latitude vs depth, longitude vs depth.
    Styling comes from ``plots`` with ``final_data_visualization.plots`` overrides.

    Args:
        config: Full configuration.
        events_df: DataFrame with event data.
        stations_df: DataFrame with station data.
        picks_df: Optional DataFrame with pick data for connection lines.
    """
    vis_config = config.final_data_visualization.station_event_vis
    settings = resolve_plot_settings(config, config.final_data_visualization)
    origin = (config.coordinate_transform.get('origin_lon'),
              config.coordinate_transform.get('origin_lat'))
    plot_station_event_views(
        events_df, stations_df, picks_df, Path(vis_config.output_dir), settings,
        filters=config.get('filters'),
        origin=origin if None not in origin else None)


def plot_station_event_views(events_df: pd.DataFrame, stations_df: pd.DataFrame,
                             picks_df: Optional[pd.DataFrame], output_dir: Path,
                             settings: DictConfig, filters=None,
                             origin: Optional[tuple] = None,
                             primary_station_ids: Optional[Set[str]] = None,
                             receiver_label: str = 'Receivers') -> None:
    """Create the map view and the two depth cross-sections.

    Args:
        events_df: Events (latitude, longitude, depth, magnitude).
        stations_df: Stations (station_id, latitude, longitude, depth).
        picks_df: Optional picks (event_id, station_id) for connection lines.
        output_dir: Output directory.
        settings: Resolved plot settings.
        filters: Filters node; its lat/lon/depth bounds set the axis limits.
        origin: Optional (lon, lat) included in the map extent.
        primary_station_ids: When given, stations outside this set are drawn as
            "picks only" receivers.
        receiver_label: Legend label for receivers.
    """
    logger = logging.getLogger(__name__)
    if len(events_df) == 0:
        logger.warning("No events to visualize")
        return

    events_df = _normalize_coordinate_names(events_df)
    stations_df = _normalize_coordinate_names(stations_df)
    if 'magnitude' not in events_df.columns:
        events_df = events_df.assign(magnitude=3.0)
    if stations_df.empty:
        stations_df = pd.DataFrame({c: pd.Series(dtype=float) for c in
                                    ('latitude', 'longitude', 'depth')}).assign(
                                        station_id=pd.Series(dtype=object))
    ensure_directory_exists(output_dir)
    logger.info("Creating station and event visualizations...")
    for view in _VIEWS:
        _plot_view(view, events_df, stations_df, picks_df, output_dir, settings, filters,
                   origin, primary_station_ids, receiver_label)
    logger.info(f"Visualizations saved to: {output_dir}")


def create_velocity_model_visualization(config: DictConfig,
                                        velocity_file: str) -> None:
    """Create velocity model visualizations.

    Creates depth slices and/or profiles through the velocity model.

    Args:
        config: Full configuration.
        velocity_file: Path to velocity model file (CSV or HDF5).
    """
    logger = logging.getLogger(__name__)
    vis_config = config.final_data_visualization.velocity_model_vis

    if not vis_config.enabled:
        logger.info("Velocity model visualization disabled")
        return

    logger.info("Creating velocity model visualizations...")

    # Load velocity model (supports both CSV and HDF5 formats)
    vel_df = read_velocity_file(velocity_file, show_progress=True)

    settings = resolve_plot_settings(config, config.final_data_visualization)

    # Create output directory
    output_dir = Path(vis_config.output_dir)
    ensure_directory_exists(output_dir)

    # Create depth slices
    if vis_config.plot_depth_slices:
        _plot_depth_slices(
            vel_df,
            vis_config.depth_slices,
            output_dir,
            settings
        )

    # Create profiles
    if vis_config.plot_profiles:
        _plot_velocity_profiles(
            vel_df,
            vis_config.profile_type,
            output_dir,
            settings
        )

    logger.info(f"Velocity visualizations saved to: {output_dir}")


def _plot_depth_slices(vel_df: pd.DataFrame, depth_slices: list,
                      output_dir: Path, settings: DictConfig) -> None:
    """Plot horizontal depth slices using filled contours (viridis/plasma).

    Matches the analysis workflow style: contourf with 20 levels, griddata
    interpolation, and shared colorbar normalization.  When the model is
    laterally uniform (1D initial model), an annotation is added and the
    colorbar range is centred on the uniform value so the fill is still
    informative.
    """
    logger = logging.getLogger(__name__)

    z_col, x_col, y_col = 'z', 'x', 'y'
    x_label = 'X (km)'
    y_label = 'Y (km)'

    for depth in depth_slices:
        # Find closest depth in data
        depths = vel_df[z_col].unique()
        closest_depth = depths[np.argmin(np.abs(depths - depth))]

        slice_df = vel_df[vel_df[z_col] == closest_depth].copy()

        if len(slice_df) == 0:
            logger.warning(f"No data at depth {depth} km, skipping")
            continue

        # Build regular meshgrid for interpolation
        x_unique = np.sort(slice_df[x_col].unique())
        y_unique = np.sort(slice_df[y_col].unique())
        X, Y = np.meshgrid(x_unique, y_unique)

        # Interpolate Vp and Vs onto the regular grid
        Z_vp = scipy_griddata(
            (slice_df[x_col].values, slice_df[y_col].values),
            slice_df['vp'].values, (X, Y), method='linear'
        )
        Z_vs = scipy_griddata(
            (slice_df[x_col].values, slice_df[y_col].values),
            slice_df['vs'].values, (X, Y), method='linear'
        )

        # Detect laterally uniform (1D) model
        vp_std = float(np.nanstd(Z_vp))
        vs_std = float(np.nanstd(Z_vs))
        is_1d = vp_std < 1e-6 and vs_std < 1e-6

        vp_val = float(np.nanmean(Z_vp))
        vs_val = float(np.nanmean(Z_vs))

        # Colour range — for a 1D model add a small symmetric range so
        # contourf can render a meaningful filled plot
        if is_1d:
            half = max(vp_val * 0.02, 0.05)   # ±2 % or ±0.05 km/s
            vp_min, vp_max = vp_val - half, vp_val + half
            half = max(vs_val * 0.02, 0.05)
            vs_min, vs_max = vs_val - half, vs_val + half
        else:
            vp_min, vp_max = float(np.nanmin(Z_vp)), float(np.nanmax(Z_vp))
            vs_min, vs_max = float(np.nanmin(Z_vs)), float(np.nanmax(Z_vs))
            # Guard against near-zero range
            if vp_max - vp_min < 1e-10:
                vp_min, vp_max = vp_val - 0.05, vp_val + 0.05
            if vs_max - vs_min < 1e-10:
                vs_min, vs_max = vs_val - 0.05, vs_val + 0.05

        norm_vp = Normalize(vmin=vp_min, vmax=vp_max)
        norm_vs = Normalize(vmin=vs_min, vmax=vs_max)

        title_suffix = '  [1D — Laterally Uniform]' if is_1d else ''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Vp panel ---
        ax1.contourf(X, Y, Z_vp, levels=20, cmap='viridis',
                     vmin=vp_min, vmax=vp_max, extend='both')
        ax1.set_title(f'Vp at {closest_depth:.1f} km depth', fontsize=13)
        ax1.set_xlabel(x_label, fontsize=11)
        ax1.set_ylabel(y_label, fontsize=11)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(
            plt.cm.ScalarMappable(norm=norm_vp, cmap='viridis'),
            ax=ax1, label='Vp (km/s)'
        )
        if is_1d:
            ax1.text(0.5, 0.5, f'Laterally uniform\nVp = {vp_val:.3f} km/s',
                     transform=ax1.transAxes, ha='center', va='center',
                     fontsize=12, color='white', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.55))

        # --- Vs panel ---
        ax2.contourf(X, Y, Z_vs, levels=20, cmap='plasma',
                     vmin=vs_min, vmax=vs_max, extend='both')
        ax2.set_title(f'Vs at {closest_depth:.1f} km depth', fontsize=13)
        ax2.set_xlabel(x_label, fontsize=11)
        ax2.set_ylabel(y_label, fontsize=11)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(
            plt.cm.ScalarMappable(norm=norm_vs, cmap='plasma'),
            ax=ax2, label='Vs (km/s)'
        )
        if is_1d:
            ax2.text(0.5, 0.5, f'Laterally uniform\nVs = {vs_val:.3f} km/s',
                     transform=ax2.transAxes, ha='center', va='center',
                     fontsize=12, color='white', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.55))

        plt.suptitle(
            f'Velocity Depth Slice at {closest_depth:.1f} km{title_suffix}',
            fontsize=15, fontweight='bold'
        )
        plt.subplots_adjust(left=0.08, right=0.95, top=0.88,
                            bottom=0.10, wspace=0.32)

        output_file = save_figure(fig, output_dir,
                                  f'velocity_slice_depth_{closest_depth:.1f}km', settings)
        logger.info(f"Saved depth slice: {output_file}")


def _plot_velocity_profiles(vel_df: pd.DataFrame, profile_type: str,
                           output_dir: Path, settings: DictConfig) -> None:
    """Plot velocity profiles (1D depth profiles)."""
    logger = logging.getLogger(__name__)

    # Group by depth and compute mean velocities
    depth_profile = vel_df.groupby('depth')[['vp', 'vs']].mean().reset_index()
    depth_profile = depth_profile.sort_values('depth')

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Vp profile
    ax1.plot(depth_profile['vp'], depth_profile['depth'],
            'b-', linewidth=2, label='Vp', drawstyle='steps-pre')
    ax1.set_xlabel('Vp (km/s)', fontsize=12)
    ax1.set_ylabel('Depth (km)', fontsize=12)
    ax1.set_title('P-wave Velocity Profile', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot Vs profile
    ax2.plot(depth_profile['vs'], depth_profile['depth'],
            'r-', linewidth=2, label='Vs', drawstyle='steps-pre')
    ax2.set_xlabel('Vs (km/s)', fontsize=12)
    ax2.set_ylabel('Depth (km)', fontsize=12)
    ax2.set_title('S-wave Velocity Profile', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    output_file = save_figure(fig, output_dir, 'velocity_profiles', settings)

    logger.info(f"Saved velocity profiles: {output_file}")
