"""Visualization functions for seismic data and velocity models.

This module provides functions for visualizing:
- Station and event locations (map view and cross-sections)
- Velocity model (depth slices and profiles)
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from utils.io_utils import ensure_directory_exists


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

    if len(events_df) == 0:
        logger.warning("No events to visualize")
        return

    logger.info("Creating station and event visualizations...")

    # Create output directory
    output_dir = Path(vis_config.output_dir)
    ensure_directory_exists(output_dir)

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
        vis_config
    )

    # Figure 2: Cross-section (Latitude vs Depth)
    _plot_cross_section_lat(
        source_lats, source_depths, source_mags,
        receiver_lats, receiver_depths,
        output_dir / 'cross_section_lat_depth.png',
        vis_config
    )

    # Figure 3: Cross-section (Longitude vs Depth)
    _plot_cross_section_lon(
        source_lons, source_depths, source_mags,
        receiver_lons, receiver_depths,
        output_dir / 'cross_section_lon_depth.png',
        vis_config
    )

    logger.info(f"Visualizations saved to: {output_dir}")


def _plot_map_view(source_lons: np.ndarray, source_lats: np.ndarray,
                  source_depths: np.ndarray, source_mags: np.ndarray,
                  receiver_lons: np.ndarray, receiver_lats: np.ndarray,
                  output_file: Path, config: DictConfig) -> None:
    """Create map view plot."""
    fig, ax = plt.subplots(
        figsize=tuple(config.figure_size),
        dpi=config.figure_dpi
    )

    # Plot events (colored by depth, sized by magnitude)
    if len(source_lons) > 0:
        sizes = source_mags**2 * config.source_size / 9
        scatter = ax.scatter(
            source_lons, source_lats,
            s=sizes, c=source_depths, cmap='hot_r',
            marker=config.source_marker, alpha=0.7,
            edgecolors='black', linewidths=0.5,
            label=f'Events (n={len(source_lons)})',
            vmin=0, zorder=3
        )
        plt.colorbar(scatter, ax=ax, label='Depth (km)')

    # Plot receivers
    if len(receiver_lons) > 0:
        ax.scatter(
            receiver_lons, receiver_lats,
            s=config.receiver_size, c=config.receiver_color,
            marker=config.receiver_marker, alpha=0.8,
            edgecolors='black', linewidths=0.5,
            label=f'Receivers (n={len(receiver_lons)})',
            zorder=2
        )

    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)
    ax.set_title('Event Locations - Map View', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    if len(source_lons) > 0 or len(receiver_lons) > 0:
        ax.axis('equal')

    plt.tight_layout()
    fig.savefig(output_file, dpi=config.figure_dpi, bbox_inches='tight')
    plt.close(fig)

    logging.getLogger(__name__).info(f"Saved: {output_file}")


def _plot_cross_section_lat(source_lats: np.ndarray, source_depths: np.ndarray,
                            source_mags: np.ndarray, receiver_lats: np.ndarray,
                            receiver_depths: np.ndarray, output_file: Path,
                            config: DictConfig) -> None:
    """Create latitude vs depth cross-section plot."""
    fig, ax = plt.subplots(
        figsize=tuple(config.figure_size),
        dpi=config.figure_dpi
    )

    # Plot events (colored by magnitude, sized by magnitude)
    if len(source_lats) > 0:
        sizes = source_mags**2 * config.source_size / 9
        scatter = ax.scatter(
            source_lats, source_depths,
            s=sizes, c=source_mags, cmap='viridis',
            marker=config.source_marker, alpha=0.7,
            edgecolors='black', linewidths=0.5,
            label=f'Events (n={len(source_lats)})',
            zorder=3
        )
        plt.colorbar(scatter, ax=ax, label='Magnitude')

    # Plot receivers
    if len(receiver_lats) > 0:
        ax.scatter(
            receiver_lats, receiver_depths,
            s=config.receiver_size, c=config.receiver_color,
            marker=config.receiver_marker, alpha=0.8,
            edgecolors='black', linewidths=0.5,
            label=f'Receivers (n={len(receiver_lats)})',
            zorder=2
        )

    ax.set_xlabel('Latitude (°)', fontsize=12)
    ax.set_ylabel('Depth (km)', fontsize=12)
    ax.set_title('Event Locations - Latitude vs Depth',
                fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Depth increases downward
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_file, dpi=config.figure_dpi, bbox_inches='tight')
    plt.close(fig)

    logging.getLogger(__name__).info(f"Saved: {output_file}")


def _plot_cross_section_lon(source_lons: np.ndarray, source_depths: np.ndarray,
                            source_mags: np.ndarray, receiver_lons: np.ndarray,
                            receiver_depths: np.ndarray, output_file: Path,
                            config: DictConfig) -> None:
    """Create longitude vs depth cross-section plot."""
    fig, ax = plt.subplots(
        figsize=tuple(config.figure_size),
        dpi=config.figure_dpi
    )

    # Plot events (colored by magnitude, sized by magnitude)
    if len(source_lons) > 0:
        sizes = source_mags**2 * config.source_size / 9
        scatter = ax.scatter(
            source_lons, source_depths,
            s=sizes, c=source_mags, cmap='viridis',
            marker=config.source_marker, alpha=0.7,
            edgecolors='black', linewidths=0.5,
            label=f'Events (n={len(source_lons)})',
            zorder=3
        )
        plt.colorbar(scatter, ax=ax, label='Magnitude')

    # Plot receivers
    if len(receiver_lons) > 0:
        ax.scatter(
            receiver_lons, receiver_depths,
            s=config.receiver_size, c=config.receiver_color,
            marker=config.receiver_marker, alpha=0.8,
            edgecolors='black', linewidths=0.5,
            label=f'Receivers (n={len(receiver_lons)})',
            zorder=2
        )

    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Depth (km)', fontsize=12)
    ax.set_title('Event Locations - Longitude vs Depth',
                fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Depth increases downward
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_file, dpi=config.figure_dpi, bbox_inches='tight')
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
