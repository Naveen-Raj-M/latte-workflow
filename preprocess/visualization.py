"""Visualization functions for seismic data and velocity models.

This module provides functions for visualizing:
- Station and event locations (map view and cross-sections)
- Velocity model (depth slices and profiles)
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from utils.io_utils import ensure_directory_exists


def _load_basin_boundaries(config: DictConfig) -> Dict:
    """Load basin boundary data from CSV files.

    Args:
        config: Basin boundaries configuration.

    Returns:
        Dictionary with basin name -> {'lons': array, 'lats': array, 'label': str}
    """
    if not config.get('enabled', False):
        return {}

    basin_data = {}
    data_dir = Path(config.data_dir)

    for basin in config.basins:
        csv_file = data_dir / basin['file']

        if not csv_file.exists():
            logging.getLogger(__name__).warning(
                f"Basin boundary file not found: {csv_file}"
            )
            continue

        # Read space-delimited file (despite .csv extension)
        df = pd.read_csv(csv_file, delim_whitespace=True)

        # Extract longitude and latitude
        lons = df['POINT_X'].values
        lats = df['POINT_Y'].values

        # Extract basin name from file or label
        basin_name = basin.get('label', basin['file'].replace('.csv', ''))

        basin_data[basin_name] = {
            'lons': lons,
            'lats': lats,
            'label': basin.get('label', basin_name)
        }

    return basin_data


def _plot_basin_boundary(ax, basin_data: Dict, config: DictConfig,
                         xlim: Optional[tuple] = None,
                         ylim: Optional[tuple] = None) -> None:
    """Plot basin boundaries on map view axis.

    Args:
        ax: Matplotlib axis.
        basin_data: Dictionary from _load_basin_boundaries().
        config: Basin boundaries configuration.
        xlim: X-axis limits (lon_min, lon_max).
        ylim: Y-axis limits (lat_min, lat_max).
    """
    for basin_name, data in basin_data.items():
        lons = data['lons']
        lats = data['lats']
        label = data['label']

        # Plot boundary polygon
        ax.plot(lons, lats,
               color=config.line_color,
               linewidth=config.line_width,
               linestyle=config.line_style,
               alpha=config.line_alpha,
               zorder=4)

        # Plot labels if enabled
        if config.get('show_labels', True):
            # Calculate centroid for label placement
            centroid_lon = np.mean(lons)
            centroid_lat = np.mean(lats)

            # Apply offsets
            label_x = centroid_lon + config.label_offset_x
            label_y = centroid_lat + config.label_offset_y

            # Only plot label if within bounds
            plot_label = True
            if xlim is not None:
                if label_x < xlim[0] or label_x > xlim[1]:
                    plot_label = False
            if ylim is not None:
                if label_y < ylim[0] or label_y > ylim[1]:
                    plot_label = False

            if plot_label:
                ax.text(label_x, label_y, label,
                       fontsize=config.label_fontsize,
                       fontweight=config.label_fontweight,
                       color=config.line_color,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor=config.line_color, alpha=0.7),
                       zorder=5)


def create_station_event_visualization(config: DictConfig,
                                       events_df: pd.DataFrame,
                                       stations_df: pd.DataFrame,
                                       picks_df: Optional[pd.DataFrame] = None) -> None:
    """Create visualizations of stations and events.

    Creates three plots:
    1. Map view (latitude vs longitude)
    2. Cross-section (latitude vs depth)
    3. Cross-section (longitude vs depth)

    Args:
        config: Full configuration.
        events_df: DataFrame with event data.
        stations_df: DataFrame with station data.
        picks_df: Optional DataFrame with pick data for connection lines.
    """
    logger = logging.getLogger(__name__)
    vis_config = config.visualization.station_event_vis

    # Check if station/event visualization is enabled
    if not vis_config.get('enabled', True):
        logger.info("Station/event visualization disabled")
        return

    if len(events_df) == 0:
        logger.warning("No events to visualize")
        return

    logger.info("Creating station and event visualizations...")

    # Create output directory
    output_dir = Path(vis_config.output_dir)
    ensure_directory_exists(output_dir)

    # Load basin boundaries if enabled
    basin_boundaries = {}
    if hasattr(vis_config, 'basin_boundaries'):
        basin_boundaries = _load_basin_boundaries(vis_config.basin_boundaries)
        if basin_boundaries:
            logger.info(f"Loaded {len(basin_boundaries)} basin boundaries")

    # Extract data
    # Use 'lat' and 'lon' if available, otherwise 'latitude' and 'longitude'
    lat_col = 'lat' if 'lat' in events_df.columns else 'latitude'
    lon_col = 'lon' if 'lon' in events_df.columns else 'longitude'

    source_lats = events_df[lat_col].values
    source_lons = events_df[lon_col].values
    source_depths = events_df['depth'].values
    source_mags = events_df.get('magnitude', pd.Series([3.0] * len(events_df))).fillna(3.0).values

    if len(stations_df) > 0:
        # Use 'lat' and 'lon' if available, otherwise 'latitude' and 'longitude'
        lat_col = 'lat' if 'lat' in stations_df.columns else 'latitude'
        lon_col = 'lon' if 'lon' in stations_df.columns else 'longitude'

        receiver_lats = stations_df[lat_col].values
        receiver_lons = stations_df[lon_col].values
        receiver_depths = stations_df['depth'].values
    else:
        receiver_lats = np.array([])
        receiver_lons = np.array([])
        receiver_depths = np.array([])

    # Figure 1: Map view (Latitude vs Longitude)
    _plot_map_view(
        source_lons, source_lats, source_depths, source_mags,
        receiver_lons, receiver_lats,
        output_dir / 'map_view_lat_lon.png',
        vis_config, config, basin_boundaries,
        events_df, stations_df, picks_df
    )

    # Figure 2: Cross-section (Latitude vs Depth)
    _plot_cross_section_lat(
        source_lats, source_depths, source_mags,
        receiver_lats, receiver_depths,
        output_dir / 'cross_section_lat_depth.png',
        vis_config, config,
        events_df, stations_df, picks_df
    )

    # Figure 3: Cross-section (Longitude vs Depth)
    _plot_cross_section_lon(
        source_lons, source_depths, source_mags,
        receiver_lons, receiver_depths,
        output_dir / 'cross_section_lon_depth.png',
        vis_config, config,
        events_df, stations_df, picks_df
    )

    logger.info(f"Visualizations saved to: {output_dir}")


def _plot_map_view(source_lons: np.ndarray, source_lats: np.ndarray,
                  source_depths: np.ndarray, source_mags: np.ndarray,
                  receiver_lons: np.ndarray, receiver_lats: np.ndarray,
                  output_file: Path, vis_config: DictConfig,
                  full_config: DictConfig, basin_boundaries: Dict,
                  events_df: Optional[pd.DataFrame] = None,
                  stations_df: Optional[pd.DataFrame] = None,
                  picks_df: Optional[pd.DataFrame] = None) -> None:
    """Create map view plot."""
    fig, ax = plt.subplots(
        figsize=tuple(vis_config.figure_size),
        dpi=vis_config.figure_dpi
    )

    # Get fontsize config
    fontsize = vis_config.get('fontsize', {})
    fs_title = fontsize.get('title', 14)
    fs_axis_label = fontsize.get('axis_label', 12)
    fs_axis_ticks = fontsize.get('axis_ticks', 10)
    fs_legend = fontsize.get('legend', 9)
    fs_cbar_label = fontsize.get('colorbar_label', 12)
    fs_cbar_ticks = fontsize.get('colorbar_ticks', 10)

    # Plot events (colored by depth, sized by magnitude)
    if len(source_lons) > 0:
        sizes = source_mags**2 * vis_config.source_size / 9
        scatter = ax.scatter(
            source_lons, source_lats,
            s=sizes, c=source_depths, cmap='hot_r',
            marker=vis_config.source_marker, alpha=0.7,
            edgecolors='black', linewidths=0.5,
            label=f'Events (n={len(source_lons)})',
            vmin=0, zorder=3
        )
        cbar = plt.colorbar(scatter, ax=ax, label='Depth (km)')
        cbar.set_label('Depth (km)', fontsize=fs_cbar_label)
        cbar.ax.tick_params(labelsize=fs_cbar_ticks)

    # Plot receivers
    if len(receiver_lons) > 0:
        ax.scatter(
            receiver_lons, receiver_lats,
            s=vis_config.receiver_size, c=vis_config.receiver_color,
            marker=vis_config.receiver_marker, alpha=0.8,
            edgecolors='black', linewidths=0.5,
            label=f'Receivers (n={len(receiver_lons)})',
            zorder=2
        )

    # Plot pick lines connecting events to receivers using LineCollection for speed
    if (vis_config.get('plot_pick_lines', False) and
        picks_df is not None and not picks_df.empty and
        events_df is not None and stations_df is not None):

        pick_alpha = vis_config.get('pick_alpha', 0.3)

        # Use 'lat' and 'lon' if available, otherwise 'latitude' and 'longitude'
        event_lat_col = 'lat' if 'lat' in events_df.columns else 'latitude'
        event_lon_col = 'lon' if 'lon' in events_df.columns else 'longitude'
        station_lat_col = 'lat' if 'lat' in stations_df.columns else 'latitude'
        station_lon_col = 'lon' if 'lon' in stations_df.columns else 'longitude'

        # Create lookup dictionaries for coordinates (vectorized using set_index)
        event_coords = events_df.set_index('event_id')[[event_lon_col, event_lat_col]]
        station_coords = stations_df.set_index('station_id')[[station_lon_col, station_lat_col]]

        # Vectorized: merge picks with event and station coordinates
        picks_with_coords = picks_df.merge(
            event_coords, left_on='event_id', right_index=True, how='inner'
        ).merge(
            station_coords, left_on='station_id', right_index=True, how='inner',
            suffixes=('_event', '_station')
        )

        if len(picks_with_coords) > 0:
            # Build segments array for LineCollection
            # Each segment is [[event_lon, event_lat], [station_lon, station_lat]]
            event_lons = picks_with_coords[f'{event_lon_col}_event'].values
            event_lats = picks_with_coords[f'{event_lat_col}_event'].values
            station_lons = picks_with_coords[f'{station_lon_col}_station'].values
            station_lats = picks_with_coords[f'{station_lat_col}_station'].values

            # Create segments as (N, 2, 2) array: N lines, 2 points each, 2 coords (lon, lat)
            segments = np.stack([
                np.column_stack([event_lons, event_lats]),
                np.column_stack([station_lons, station_lats])
            ], axis=1)

            # Create LineCollection for efficient rendering
            lc = LineCollection(
                segments,
                colors='gray',
                alpha=pick_alpha,
                linewidths=0.5,
                zorder=1,
                rasterized=True  # Improves performance for large numbers of lines
            )
            ax.add_collection(lc)

    ax.set_xlabel('Longitude (°)', fontsize=fs_axis_label)
    ax.set_ylabel('Latitude (°)', fontsize=fs_axis_label)
    ax.set_title('Event Locations - Map View', fontsize=fs_title, fontweight='bold')
    ax.tick_params(axis='both', labelsize=fs_axis_ticks)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=fs_legend)

    if len(source_lons) > 0 or len(receiver_lons) > 0:
        ax.axis('equal')

    # Apply geographic filters to axis limits if specified
    filters = full_config.get('filters', {})
    if filters.get('lon_min') is not None or filters.get('lon_max') is not None:
        current_xlim = ax.get_xlim()
        new_xlim = [
            filters.get('lon_min') if filters.get('lon_min') is not None else current_xlim[0],
            filters.get('lon_max') if filters.get('lon_max') is not None else current_xlim[1]
        ]
        ax.set_xlim(new_xlim)

    if filters.get('lat_min') is not None or filters.get('lat_max') is not None:
        current_ylim = ax.get_ylim()
        new_ylim = [
            filters.get('lat_min') if filters.get('lat_min') is not None else current_ylim[0],
            filters.get('lat_max') if filters.get('lat_max') is not None else current_ylim[1]
        ]
        ax.set_ylim(new_ylim)

    # Plot basin boundaries if enabled
    if basin_boundaries and hasattr(vis_config, 'basin_boundaries'):
        if vis_config.basin_boundaries.get('enabled', False):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            _plot_basin_boundary(ax, basin_boundaries, vis_config.basin_boundaries,
                               xlim=xlim, ylim=ylim)

    plt.tight_layout()
    fig.savefig(output_file, dpi=vis_config.figure_dpi, bbox_inches='tight')
    plt.close(fig)

    logging.getLogger(__name__).info(f"Saved: {output_file}")


def _plot_cross_section_lat(source_lats: np.ndarray, source_depths: np.ndarray,
                            source_mags: np.ndarray, receiver_lats: np.ndarray,
                            receiver_depths: np.ndarray, output_file: Path,
                            vis_config: DictConfig, full_config: DictConfig,
                            events_df: Optional[pd.DataFrame] = None,
                            stations_df: Optional[pd.DataFrame] = None,
                            picks_df: Optional[pd.DataFrame] = None) -> None:
    """Create latitude vs depth cross-section plot."""
    fig, ax = plt.subplots(
        figsize=tuple(vis_config.figure_size),
        dpi=vis_config.figure_dpi
    )

    # Get fontsize config
    fontsize = vis_config.get('fontsize', {})
    fs_title = fontsize.get('title', 14)
    fs_axis_label = fontsize.get('axis_label', 12)
    fs_axis_ticks = fontsize.get('axis_ticks', 10)
    fs_legend = fontsize.get('legend', 9)
    fs_cbar_label = fontsize.get('colorbar_label', 12)
    fs_cbar_ticks = fontsize.get('colorbar_ticks', 10)

    # Plot events (colored by magnitude, sized by magnitude)
    if len(source_lats) > 0:
        sizes = source_mags**2 * vis_config.source_size / 9
        scatter = ax.scatter(
            source_lats, source_depths,
            s=sizes, c=source_mags, cmap='viridis',
            marker=vis_config.source_marker, alpha=0.7,
            edgecolors='black', linewidths=0.5,
            label=f'Events (n={len(source_lats)})',
            zorder=3
        )
        cbar = plt.colorbar(scatter, ax=ax, label='Magnitude')
        cbar.set_label('Magnitude', fontsize=fs_cbar_label)
        cbar.ax.tick_params(labelsize=fs_cbar_ticks)

    # Plot receivers
    if len(receiver_lats) > 0:
        ax.scatter(
            receiver_lats, receiver_depths,
            s=vis_config.receiver_size, c=vis_config.receiver_color,
            marker=vis_config.receiver_marker, alpha=0.8,
            edgecolors='black', linewidths=0.5,
            label=f'Receivers (n={len(receiver_lats)})',
            zorder=2
        )

    # Plot pick lines connecting events to receivers using LineCollection for speed
    if (vis_config.get('plot_pick_lines', False) and
        picks_df is not None and not picks_df.empty and
        events_df is not None and stations_df is not None):

        pick_alpha = vis_config.get('pick_alpha', 0.3)

        # Use 'lat' if available, otherwise 'latitude'
        event_lat_col = 'lat' if 'lat' in events_df.columns else 'latitude'
        station_lat_col = 'lat' if 'lat' in stations_df.columns else 'latitude'

        # Create coordinate DataFrames for vectorized merge
        event_coords = events_df.set_index('event_id')[[event_lat_col, 'depth']].rename(
            columns={event_lat_col: 'lat', 'depth': 'depth'}
        )
        station_coords = stations_df.set_index('station_id')[[station_lat_col, 'depth']].rename(
            columns={station_lat_col: 'lat', 'depth': 'depth'}
        )

        # Vectorized: merge picks with event and station coordinates
        picks_with_coords = picks_df.merge(
            event_coords, left_on='event_id', right_index=True, how='inner'
        ).merge(
            station_coords, left_on='station_id', right_index=True, how='inner',
            suffixes=('_event', '_station')
        )

        if len(picks_with_coords) > 0:
            # Build segments array for LineCollection
            event_lats = picks_with_coords['lat_event'].values
            event_depths = picks_with_coords['depth_event'].values
            station_lats = picks_with_coords['lat_station'].values
            station_depths = picks_with_coords['depth_station'].values

            # Create segments as (N, 2, 2) array: N lines, 2 points each, 2 coords (lat, depth)
            segments = np.stack([
                np.column_stack([event_lats, event_depths]),
                np.column_stack([station_lats, station_depths])
            ], axis=1)

            # Create LineCollection for efficient rendering
            lc = LineCollection(
                segments,
                colors='gray',
                alpha=pick_alpha,
                linewidths=0.5,
                zorder=1,
                rasterized=True  # Improves performance for large numbers of lines
            )
            ax.add_collection(lc)

    ax.set_xlabel('Latitude (°)', fontsize=fs_axis_label)
    ax.set_ylabel('Depth (km)', fontsize=fs_axis_label)
    ax.set_title('Event Locations - Latitude vs Depth',
                fontsize=fs_title, fontweight='bold')
    ax.tick_params(axis='both', labelsize=fs_axis_ticks)
    ax.invert_yaxis()  # Depth increases downward
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=fs_legend)

    # Apply latitude filter to x-axis
    filters = full_config.get('filters', {})
    if filters.get('lat_min') is not None or filters.get('lat_max') is not None:
        current_xlim = ax.get_xlim()
        new_xlim = [
            filters.get('lat_min') if filters.get('lat_min') is not None else current_xlim[0],
            filters.get('lat_max') if filters.get('lat_max') is not None else current_xlim[1]
        ]
        ax.set_xlim(new_xlim)

    # Apply depth filter to y-axis (inverted)
    if filters.get('depth_min') is not None or filters.get('depth_max') is not None:
        current_ylim = ax.get_ylim()
        new_ylim = [
            filters.get('depth_max') if filters.get('depth_max') is not None else current_ylim[0],
            filters.get('depth_min') if filters.get('depth_min') is not None else current_ylim[1]
        ]
        ax.set_ylim(new_ylim)

    plt.tight_layout()
    fig.savefig(output_file, dpi=vis_config.figure_dpi, bbox_inches='tight')
    plt.close(fig)

    logging.getLogger(__name__).info(f"Saved: {output_file}")


def _plot_cross_section_lon(source_lons: np.ndarray, source_depths: np.ndarray,
                            source_mags: np.ndarray, receiver_lons: np.ndarray,
                            receiver_depths: np.ndarray, output_file: Path,
                            vis_config: DictConfig, full_config: DictConfig,
                            events_df: Optional[pd.DataFrame] = None,
                            stations_df: Optional[pd.DataFrame] = None,
                            picks_df: Optional[pd.DataFrame] = None) -> None:
    """Create longitude vs depth cross-section plot."""
    fig, ax = plt.subplots(
        figsize=tuple(vis_config.figure_size),
        dpi=vis_config.figure_dpi
    )

    # Get fontsize config
    fontsize = vis_config.get('fontsize', {})
    fs_title = fontsize.get('title', 14)
    fs_axis_label = fontsize.get('axis_label', 12)
    fs_axis_ticks = fontsize.get('axis_ticks', 10)
    fs_legend = fontsize.get('legend', 9)
    fs_cbar_label = fontsize.get('colorbar_label', 12)
    fs_cbar_ticks = fontsize.get('colorbar_ticks', 10)

    # Plot events (colored by magnitude, sized by magnitude)
    if len(source_lons) > 0:
        sizes = source_mags**2 * vis_config.source_size / 9
        scatter = ax.scatter(
            source_lons, source_depths,
            s=sizes, c=source_mags, cmap='viridis',
            marker=vis_config.source_marker, alpha=0.7,
            edgecolors='black', linewidths=0.5,
            label=f'Events (n={len(source_lons)})',
            zorder=3
        )
        cbar = plt.colorbar(scatter, ax=ax, label='Magnitude')
        cbar.set_label('Magnitude', fontsize=fs_cbar_label)
        cbar.ax.tick_params(labelsize=fs_cbar_ticks)

    # Plot receivers
    if len(receiver_lons) > 0:
        ax.scatter(
            receiver_lons, receiver_depths,
            s=vis_config.receiver_size, c=vis_config.receiver_color,
            marker=vis_config.receiver_marker, alpha=0.8,
            edgecolors='black', linewidths=0.5,
            label=f'Receivers (n={len(receiver_lons)})',
            zorder=2
        )

    # Plot pick lines connecting events to receivers using LineCollection for speed
    if (vis_config.get('plot_pick_lines', False) and
        picks_df is not None and not picks_df.empty and
        events_df is not None and stations_df is not None):

        pick_alpha = vis_config.get('pick_alpha', 0.3)

        # Use 'lon' if available, otherwise 'longitude'
        event_lon_col = 'lon' if 'lon' in events_df.columns else 'longitude'
        station_lon_col = 'lon' if 'lon' in stations_df.columns else 'longitude'

        # Create coordinate DataFrames for vectorized merge
        event_coords = events_df.set_index('event_id')[[event_lon_col, 'depth']].rename(
            columns={event_lon_col: 'lon', 'depth': 'depth'}
        )
        station_coords = stations_df.set_index('station_id')[[station_lon_col, 'depth']].rename(
            columns={station_lon_col: 'lon', 'depth': 'depth'}
        )

        # Vectorized: merge picks with event and station coordinates
        picks_with_coords = picks_df.merge(
            event_coords, left_on='event_id', right_index=True, how='inner'
        ).merge(
            station_coords, left_on='station_id', right_index=True, how='inner',
            suffixes=('_event', '_station')
        )

        if len(picks_with_coords) > 0:
            # Build segments array for LineCollection
            event_lons = picks_with_coords['lon_event'].values
            event_depths = picks_with_coords['depth_event'].values
            station_lons = picks_with_coords['lon_station'].values
            station_depths = picks_with_coords['depth_station'].values

            # Create segments as (N, 2, 2) array: N lines, 2 points each, 2 coords (lon, depth)
            segments = np.stack([
                np.column_stack([event_lons, event_depths]),
                np.column_stack([station_lons, station_depths])
            ], axis=1)

            # Create LineCollection for efficient rendering
            lc = LineCollection(
                segments,
                colors='gray',
                alpha=pick_alpha,
                linewidths=0.5,
                zorder=1,
                rasterized=True  # Improves performance for large numbers of lines
            )
            ax.add_collection(lc)

    ax.set_xlabel('Longitude (°)', fontsize=fs_axis_label)
    ax.set_ylabel('Depth (km)', fontsize=fs_axis_label)
    ax.set_title('Event Locations - Longitude vs Depth',
                fontsize=fs_title, fontweight='bold')
    ax.tick_params(axis='both', labelsize=fs_axis_ticks)
    ax.invert_yaxis()  # Depth increases downward
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=fs_legend)

    # Apply longitude filter to x-axis
    filters = full_config.get('filters', {})
    if filters.get('lon_min') is not None or filters.get('lon_max') is not None:
        current_xlim = ax.get_xlim()
        new_xlim = [
            filters.get('lon_min') if filters.get('lon_min') is not None else current_xlim[0],
            filters.get('lon_max') if filters.get('lon_max') is not None else current_xlim[1]
        ]
        ax.set_xlim(new_xlim)

    # Apply depth filter to y-axis (inverted)
    if filters.get('depth_min') is not None or filters.get('depth_max') is not None:
        current_ylim = ax.get_ylim()
        new_ylim = [
            filters.get('depth_max') if filters.get('depth_max') is not None else current_ylim[0],
            filters.get('depth_min') if filters.get('depth_min') is not None else current_ylim[1]
        ]
        ax.set_ylim(new_ylim)

    plt.tight_layout()
    fig.savefig(output_file, dpi=vis_config.figure_dpi, bbox_inches='tight')
    plt.close(fig)

    logging.getLogger(__name__).info(f"Saved: {output_file}")


def create_velocity_model_visualization(config: DictConfig,
                                        velocity_csv: str) -> None:
    """Create velocity model visualizations.

    Creates depth slices and/or profiles through the velocity model.

    Args:
        config: Full configuration.
        velocity_csv: Path to velocity model CSV file.
    """
    logger = logging.getLogger(__name__)
    vis_config = config.visualization.velocity_model_vis

    if not vis_config.enabled:
        logger.info("Velocity model visualization disabled")
        return

    logger.info("Creating velocity model visualizations...")

    # Load velocity model
    vel_df = pd.read_csv(velocity_csv)

    # Create output directory
    output_dir = Path(vis_config.output_dir)
    ensure_directory_exists(output_dir)

    # Create depth slices
    if vis_config.plot_depth_slices:
        _plot_depth_slices(
            vel_df,
            vis_config.depth_slices,
            output_dir,
            config
        )

    # Create profiles
    if vis_config.plot_profiles:
        _plot_velocity_profiles(
            vel_df,
            vis_config.profile_type,
            output_dir,
            config
        )

    logger.info(f"Velocity visualizations saved to: {output_dir}")


def _plot_depth_slices(vel_df: pd.DataFrame, depth_slices: list,
                      output_dir: Path, config: DictConfig) -> None:
    """Plot horizontal slices at specified depths."""
    logger = logging.getLogger(__name__)

    for depth in depth_slices:
        # Find closest depth in data
        depths = vel_df['depth'].unique()
        closest_depth = depths[np.argmin(np.abs(depths - depth))]

        # Extract data at this depth
        slice_df = vel_df[vel_df['depth'] == closest_depth]

        if len(slice_df) == 0:
            logger.warning(f"No data at depth {depth} km, skipping")
            continue

        # Create figure with two subplots (Vp and Vs)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

        # Plot Vp
        scatter1 = ax1.scatter(
            slice_df['lon'], slice_df['lat'],
            c=slice_df['vp'], cmap='seismic', s=50,
            edgecolors='black', linewidths=0.5
        )
        ax1.set_xlabel('Longitude (°)')
        ax1.set_ylabel('Latitude (°)')
        ax1.set_title(f'Vp at {closest_depth:.1f} km depth')
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Vp (km/s)')

        # Plot Vs
        scatter2 = ax2.scatter(
            slice_df['lon'], slice_df['lat'],
            c=slice_df['vs'], cmap='seismic', s=50,
            edgecolors='black', linewidths=0.5
        )
        ax2.set_xlabel('Longitude (°)')
        ax2.set_ylabel('Latitude (°)')
        ax2.set_title(f'Vs at {closest_depth:.1f} km depth')
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Vs (km/s)')

        plt.tight_layout()
        output_file = output_dir / f'velocity_slice_depth_{closest_depth:.1f}km.png'
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved depth slice: {output_file}")


def _plot_velocity_profiles(vel_df: pd.DataFrame, profile_type: str,
                           output_dir: Path, config: DictConfig) -> None:
    """Plot velocity profiles (1D depth profiles)."""
    logger = logging.getLogger(__name__)

    # Group by depth and compute mean velocities
    depth_profile = vel_df.groupby('depth')[['vp', 'vs']].mean().reset_index()
    depth_profile = depth_profile.sort_values('depth')

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)

    # Plot Vp profile
    ax1.plot(depth_profile['vp'], depth_profile['depth'],
            'b-', linewidth=2, label='Vp')
    ax1.set_xlabel('Vp (km/s)', fontsize=12)
    ax1.set_ylabel('Depth (km)', fontsize=12)
    ax1.set_title('P-wave Velocity Profile', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot Vs profile
    ax2.plot(depth_profile['vs'], depth_profile['depth'],
            'r-', linewidth=2, label='Vs')
    ax2.set_xlabel('Vs (km/s)', fontsize=12)
    ax2.set_ylabel('Depth (km)', fontsize=12)
    ax2.set_title('S-wave Velocity Profile', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    output_file = output_dir / 'velocity_profiles.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved velocity profiles: {output_file}")
