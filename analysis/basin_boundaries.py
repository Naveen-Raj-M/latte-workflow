"""
Basin boundary loading and plotting for LATTE analysis module.

This module provides functions for loading and plotting basin boundaries
in Cartesian (transformed) coordinates for ray density and ray path plots.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_basin_boundaries(config: Dict) -> Dict:
    """Load basin boundary data from CSV files in Cartesian coordinates.

    Args:
        config: Basin boundaries configuration dictionary with keys:
            - data_dir: str - Directory containing basin boundary CSV files
            - basins: list - List of basin configs with 'file' and 'label' keys

    Returns:
        Dictionary with basin name -> {'x': array, 'y': array, 'label': str}
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
            # Read space-delimited file (same format as preprocessing output)
            df = pd.read_csv(csv_file, delim_whitespace=True)

            # Extract X and Y coordinates (Cartesian, in km)
            x_coords = df['POINT_X'].values
            y_coords = df['POINT_Y'].values

            # Extract basin name from file or label
            basin_name = basin.get('label', basin['file'].replace('.csv', ''))

            basin_data[basin_name] = {
                'x': x_coords,
                'y': y_coords,
                'label': basin.get('label', basin_name)
            }

            logger.debug(f"Loaded basin boundary '{basin_name}' with {len(x_coords)} vertices")

        except Exception as e:
            logger.warning(f"Error loading basin boundary {csv_file}: {e}")
            continue

    if basin_data:
        logger.info(f"Loaded {len(basin_data)} basin boundaries")

    return basin_data


def plot_basin_boundaries(ax, basin_data: Dict, config: Dict,
                          xlim: Optional[tuple] = None,
                          ylim: Optional[tuple] = None) -> None:
    """Plot basin boundaries on an axis.

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
            - label_offset_x: float - X offset for labels in km (default: 0)
            - label_offset_y: float - Y offset for labels in km (default: 0)
        xlim: Optional X-axis limits for label visibility check.
        ylim: Optional Y-axis limits for label visibility check.
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
    label_offset_x = config.get('label_offset_x', 0.0)
    label_offset_y = config.get('label_offset_y', 0.0)

    for basin_name, data in basin_data.items():
        x_coords = data['x']
        y_coords = data['y']
        label = data['label']

        # Plot boundary polygon
        ax.plot(x_coords, y_coords,
                color=line_color,
                linewidth=line_width,
                linestyle=line_style,
                alpha=line_alpha,
                zorder=4)

        # Plot labels if enabled
        if show_labels:
            # Calculate centroid for label placement
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)

            # Apply offsets (in km for Cartesian coordinates)
            label_x = centroid_x + label_offset_x
            label_y = centroid_y + label_offset_y

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
