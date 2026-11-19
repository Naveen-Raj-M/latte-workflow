"""
Basin boundary loading and plotting for LATTE analysis module.

This module provides functions for loading and plotting basin boundaries on map views
(longitude on the horizontal axis, latitude on the vertical axis). The boundary files are the
WGS84 vertex files: whitespace-delimited, ``POINT_X`` = longitude and ``POINT_Y`` = geodetic
latitude, in degrees.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_basin_boundaries(config: Dict) -> Dict:
    """Load basin boundary vertices (WGS84 longitude and latitude, degrees).

    Args:
        config: Basin boundaries configuration dictionary with keys:
            - data_dir: str - Directory containing the WGS84 basin boundary files
            - basins: list - List of basin configs with 'file' and 'label' keys

    Returns:
        Dictionary with basin name -> {'lon': array, 'lat': array, 'label': str}
        Returns empty dict if no basins configured or data directory not found.
    """
    basin_data = {}
    data_dir = Path(config.get('data_dir', ''))

    if not data_dir.exists():
        logger.warning(f"Basin boundaries data directory not found: {data_dir}")
        return {}

    basins = config.get('basins', [])
    if not basins:
        logger.warning("No basins configured in basin_boundaries")
        return {}

    for basin in basins:
        csv_file = data_dir / basin['file']

        if not csv_file.exists():
            logger.warning(f"Basin boundary file not found: {csv_file}")
            continue

        try:
            # Whitespace-delimited file (despite the .csv extension)
            df = pd.read_csv(csv_file, sep=r'\s+')

            # POINT_X is the longitude, POINT_Y the geodetic latitude (degrees)
            lon = df['POINT_X'].to_numpy(dtype=float)
            lat = df['POINT_Y'].to_numpy(dtype=float)

            # Extract basin name from file or label
            basin_name = basin.get('label', basin['file'].replace('.csv', ''))

            basin_data[basin_name] = {
                'lon': lon,
                'lat': lat,
                'label': basin.get('label', basin_name)
            }

            logger.debug(f"Loaded basin boundary '{basin_name}' with {len(lon)} vertices")

        except Exception as e:
            logger.warning(f"Error loading basin boundary {csv_file}: {e}")
            continue

    if basin_data:
        logger.info(f"Loaded {len(basin_data)} basin boundaries")

    return basin_data


def plot_basin_boundaries(ax, basin_data: Dict, config: Dict,
                          xlim: Optional[tuple] = None,
                          ylim: Optional[tuple] = None) -> None:
    """Plot basin boundaries on a map-view axis (longitude on x, latitude on y).

    Args:
        ax: Matplotlib axis to plot on.
        basin_data: Dictionary from load_basin_boundaries().
        config: Basin boundaries configuration with styling options:
            - line_color: str - Line color (default: 'darkgreen')
            - line_width: float - Line width (default: 2)
            - line_style: str - Line style (default: '-')
            - line_alpha: float - Line transparency (default: 0.8)
            - show_labels: bool - Whether to show basin labels (default: True)
            - label_fontsize: int - Label font size (default: 12)
            - label_fontweight: str - Label font weight (default: 'bold')
            - label_offset_lon: float - Longitude offset of labels in degrees (default: 0)
            - label_offset_lat: float - Latitude offset of labels in degrees (default: 0)
        xlim: Optional longitude limits for label visibility check.
        ylim: Optional latitude limits for label visibility check.
    """
    if not basin_data:
        return

    # Get styling configuration with defaults
    line_color = config.get('line_color', 'darkgreen')
    line_width = config.get('line_width', 2)
    line_style = config.get('line_style', '-')
    line_alpha = config.get('line_alpha', 0.8)
    show_labels = config.get('show_labels', True)
    label_fontsize = config.get('label_fontsize', 12)
    label_fontweight = config.get('label_fontweight', 'bold')
    label_offset_lon = config.get('label_offset_lon', 0.0)
    label_offset_lat = config.get('label_offset_lat', 0.0)

    for basin_name, data in basin_data.items():
        lon = data['lon']
        lat = data['lat']
        label = data['label']

        # Plot boundary polygon
        ax.plot(lon, lat,
                color=line_color,
                linewidth=line_width,
                linestyle=line_style,
                alpha=line_alpha,
                zorder=4)

        # Plot labels if enabled
        if show_labels:
            # Mean of the vertices for label placement, plus the offsets (degrees)
            label_x = np.mean(lon) + label_offset_lon
            label_y = np.mean(lat) + label_offset_lat

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
                        fontsize=label_fontsize,
                        fontweight=label_fontweight,
                        color=line_color,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor=line_color, alpha=0.7),
                        zorder=5)
