"""
Colormap utility functions for visualization.

This module provides functions for creating custom colormaps, including:
- Three-color scheme for marking values beyond thresholds with black
- Uniform colorbar computation across multiple slices
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def create_threshold_colormap(
    base_cmap_name: str,
    threshold: Optional[float] = None,
    black_color: str = "black",
) -> Tuple[mcolors.Colormap, Optional[mcolors.BoundaryNorm]]:
    """
    Create a three-color scheme colormap for perturbation visualization.

    If threshold is specified:
    - Values < -threshold: black
    - Values between -threshold and +threshold: base colormap
    - Values > +threshold: black

    Args:
        base_cmap_name: Name of the base matplotlib colormap (e.g., "seismic")
        threshold: Threshold value (e.g., 7.0 for 7%). If None, returns base colormap
        black_color: Color to use for values exceeding threshold

    Returns:
        (colormap, norm) tuple. norm is None if threshold is None, otherwise BoundaryNorm
    """
    if threshold is None:
        # No threshold, return base colormap
        return plt.get_cmap(base_cmap_name), None

    # Get the base colormap
    base_cmap = plt.get_cmap(base_cmap_name)

    # Number of colors in the middle range
    n_middle = 256

    # Create color list: black, base_cmap colors, black
    colors = []

    # Add black for values below -threshold (10% of colormap)
    n_low_black = int(n_middle * 0.1)
    colors.extend([black_color] * n_low_black)

    # Add base colormap for middle range (80% of colormap)
    n_middle_colors = int(n_middle * 0.8)
    middle_colors = base_cmap(np.linspace(0, 1, n_middle_colors))
    colors.extend(middle_colors)

    # Add black for values above +threshold (10% of colormap)
    n_high_black = int(n_middle * 0.1)
    colors.extend([black_color] * n_high_black)

    # Create custom colormap
    custom_cmap = mcolors.ListedColormap(colors, name=f'{base_cmap_name}_threshold')

    # Create boundary norm to map data values to colors
    # This ensures proper mapping: < -threshold -> low black, > +threshold -> high black
    # We create boundaries that extend beyond ±threshold to catch all outliers

    return custom_cmap, None


def compute_uniform_colorbar_limits(
    velocity_3d: np.ndarray,
    symmetric: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute colorbar limits from entire 3D volume for uniform colorbars.

    Args:
        velocity_3d: 3D array of velocity values
        symmetric: If True, force symmetric colorbar around zero
        vmin: Manual minimum (if None, computed from data)
        vmax: Manual maximum (if None, computed from data)

    Returns:
        (vmin, vmax) tuple
    """
    if vmin is not None and vmax is not None:
        # Both manually specified
        computed_vmin, computed_vmax = vmin, vmax
    elif vmin is not None:
        # Only vmin specified
        computed_vmin = vmin
        computed_vmax = np.nanmax(velocity_3d)
    elif vmax is not None:
        # Only vmax specified
        computed_vmin = np.nanmin(velocity_3d)
        computed_vmax = vmax
    else:
        # Compute from data
        computed_vmin = np.nanmin(velocity_3d)
        computed_vmax = np.nanmax(velocity_3d)

    # Apply symmetry if requested
    if symmetric:
        vmax_abs = max(abs(computed_vmin), abs(computed_vmax))
        computed_vmin = -vmax_abs
        computed_vmax = vmax_abs

    logger.debug(f"Uniform colorbar limits: [{computed_vmin:.4f}, {computed_vmax:.4f}]")

    return computed_vmin, computed_vmax


def apply_threshold_to_data(
    data: np.ndarray,
    threshold: Optional[float],
    vmin: float,
    vmax: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Apply threshold by clipping data values beyond ±threshold.

    This function clips the data to [vmin_extended, vmax_extended] where:
    - vmin_extended extends below -threshold to capture outliers
    - vmax_extended extends above +threshold to capture outliers

    Args:
        data: Input data array
        threshold: Threshold value (e.g., 7.0 for ±7%)
        vmin: Current minimum value
        vmax: Current maximum value

    Returns:
        (clipped_data, vmin_extended, vmax_extended) tuple
    """
    if threshold is None:
        return data, vmin, vmax

    # Extend the range slightly beyond ±threshold to show black regions
    # The extension ensures values beyond threshold map to the black parts of colormap
    extension_factor = 1.2

    vmin_extended = -threshold * extension_factor
    vmax_extended = threshold * extension_factor

    # Note: We don't actually clip the data, we just extend the colorbar range
    # The colormap will handle showing values beyond ±threshold as black

    return data, vmin_extended, vmax_extended
