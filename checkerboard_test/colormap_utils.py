"""
Colormap utility functions for visualization.

This module provides functions for creating custom colormaps, including:
- Three-color scheme for marking values beyond thresholds with black
- Uniform colorbar computation across multiple slices
- Symmetric colorbar centered at zero with asymmetric data ranges
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
) -> mcolors.Colormap:
    """
    Create a colormap with black color for out-of-range values.

    The colormap uses set_over() and set_under() to mark values
    that exceed ±threshold as black. The actual threshold clipping
    is handled by setting vmin=-threshold and vmax=threshold on the plot.

    Args:
        base_cmap_name: Name of the base matplotlib colormap (e.g., "seismic")
        threshold: Threshold value (e.g., 7.0 for 7%). If None, returns base colormap
        black_color: Color to use for values exceeding threshold

    Returns:
        Colormap with over/under colors set to black
    """
    # Get the base colormap and make a copy
    base_cmap = plt.get_cmap(base_cmap_name).copy()

    if threshold is not None:
        # Set colors for values that exceed the threshold
        base_cmap.set_over(black_color)
        base_cmap.set_under(black_color)

    return base_cmap


def create_centered_norm(
    vmin: float,
    vmax: float,
    vcenter: float = 0.0,
) -> mcolors.TwoSlopeNorm:
    """
    Create a norm that centers the colormap at a specific value.

    This allows asymmetric data ranges (e.g., -5% to +7%) while keeping
    the colormap centered at zero (so white/neutral color is at zero).

    Args:
        vmin: Minimum value
        vmax: Maximum value
        vcenter: Center value (default 0.0)

    Returns:
        TwoSlopeNorm for centered colormap
    """
    # Ensure vmin < vcenter < vmax
    if vmin >= vcenter:
        vmin = vcenter - 1e-6
    if vmax <= vcenter:
        vmax = vcenter + 1e-6

    return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


def create_threshold_colormap_with_black(
    base_cmap_name: str,
    vmin: float,
    vmax: float,
    threshold: float,
    n_colors: int = 256,
) -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
    """
    Create a colormap with black regions for values exceeding ±threshold.

    The colormap shows:
    - Black for values < -threshold
    - Base colormap for values in [-threshold, +threshold], centered at 0
    - Black for values > +threshold

    Args:
        base_cmap_name: Name of the base matplotlib colormap (e.g., "seismic")
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
        threshold: Threshold value (values with |v| > threshold are black)
        n_colors: Number of colors in the colormap

    Returns:
        (colormap, norm) tuple
    """
    base_cmap = plt.get_cmap(base_cmap_name)

    # Create boundaries for the colormap segments
    # We need to handle cases where vmin > -threshold or vmax < threshold
    boundaries = []
    colors_list = []

    # Add lower black region if vmin < -threshold
    if vmin < -threshold:
        boundaries.append(vmin)
        # Number of colors for this segment proportional to its range
        n_low_black = max(1, int(n_colors * (-threshold - vmin) / (vmax - vmin)))
        colors_list.extend(['black'] * n_low_black)

    # Add the threshold boundaries
    boundaries.append(-threshold)
    boundaries.append(threshold)

    # Add colors for the middle region (from base colormap)
    n_middle = max(1, int(n_colors * (2 * threshold) / (vmax - vmin)))
    # Sample from base colormap, centered at 0.5 (which is white for seismic)
    middle_colors = [base_cmap(i / (n_middle - 1)) if n_middle > 1 else base_cmap(0.5)
                     for i in range(n_middle)]
    colors_list.extend(middle_colors)

    # Add upper black region if vmax > threshold
    if vmax > threshold:
        boundaries.append(vmax)
        n_high_black = max(1, int(n_colors * (vmax - threshold) / (vmax - vmin)))
        colors_list.extend(['black'] * n_high_black)

    # Handle edge cases where threshold exceeds data range
    if len(boundaries) < 2:
        # Fallback to base colormap
        return base_cmap, None

    # Create the colormap
    custom_cmap = mcolors.ListedColormap(colors_list, name=f'{base_cmap_name}_threshold')

    # Create BoundaryNorm
    norm = mcolors.BoundaryNorm(boundaries, custom_cmap.N)

    return custom_cmap, norm


def create_threshold_segmented_colormap(
    base_cmap_name: str,
    vmin: float,
    vmax: float,
    threshold: float,
    symmetric: bool = True,
    n_samples: int = 50,
) -> Tuple[mcolors.LinearSegmentedColormap, mcolors.Normalize]:
    """
    Create a segmented colormap with black for values exceeding threshold.

    The colorbar shows the full range (vmin to vmax), but:
    - Values in [-threshold, +threshold] use the base colormap (centered at 0)
    - Values outside ±threshold are black

    Args:
        base_cmap_name: Name of base colormap
        vmin: Global minimum value
        vmax: Global maximum value
        threshold: Threshold for black regions (values with |v| > threshold are black)
        symmetric: Whether to center colormap at zero
        n_samples: Number of color samples for smooth gradient in threshold region

    Returns:
        (colormap, norm) tuple
    """
    base_cmap = plt.get_cmap(base_cmap_name)

    # Calculate normalized positions for the boundaries
    total_range = vmax - vmin
    if total_range <= 0:
        total_range = 1e-6

    # Position in [0, 1] normalized space
    pos_neg_thresh = max(0.0, (-threshold - vmin) / total_range)
    pos_pos_thresh = min(1.0, (threshold - vmin) / total_range)

    black = (0.0, 0.0, 0.0)

    # Build the colormap segments
    cdict = {'red': [], 'green': [], 'blue': []}

    for channel_idx, channel in enumerate(['red', 'green', 'blue']):
        segments = []

        # Black region from vmin to -threshold (if applicable)
        if vmin < -threshold:
            segments.append((0.0, black[channel_idx], black[channel_idx]))
            segments.append((pos_neg_thresh, black[channel_idx], black[channel_idx]))

        # Gradient region from -threshold to +threshold
        # Sample the base colormap and map to the threshold region
        for i in range(n_samples):
            # Position in base colormap [0, 1]
            t = i / (n_samples - 1) if n_samples > 1 else 0.5
            color = base_cmap(t)

            # Position in final colormap
            if vmin < -threshold and vmax > threshold:
                # Both black regions exist
                pos = pos_neg_thresh + t * (pos_pos_thresh - pos_neg_thresh)
            elif vmin < -threshold:
                # Only lower black region
                pos = pos_neg_thresh + t * (1.0 - pos_neg_thresh)
            elif vmax > threshold:
                # Only upper black region
                pos = t * pos_pos_thresh
            else:
                # No black regions
                pos = t

            segments.append((pos, color[channel_idx], color[channel_idx]))

        # Black region from +threshold to vmax (if applicable)
        if vmax > threshold:
            segments.append((pos_pos_thresh, black[channel_idx], black[channel_idx]))
            segments.append((1.0, black[channel_idx], black[channel_idx]))

        cdict[channel] = segments

    # Create the colormap
    custom_cmap = mcolors.LinearSegmentedColormap(f'{base_cmap_name}_threshold', cdict)

    # Use standard normalization with global vmin/vmax
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    return custom_cmap, norm


def compute_uniform_colorbar_limits(
    velocity_3d: np.ndarray,
    symmetric: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    force_equal_magnitude: bool = True,
) -> Tuple[float, float]:
    """
    Compute colorbar limits from entire 3D volume for uniform colorbars.

    Args:
        velocity_3d: 3D array of velocity values
        symmetric: If True, center colorbar around zero (use TwoSlopeNorm for actual centering)
        vmin: Manual minimum (if None, computed from data)
        vmax: Manual maximum (if None, computed from data)
        force_equal_magnitude: If True and symmetric, force vmin=-vmax_abs and vmax=vmax_abs
                               If False and symmetric, keep actual data range but centered at zero

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

    # Apply symmetry if requested AND force_equal_magnitude is True
    if symmetric and force_equal_magnitude:
        vmax_abs = max(abs(computed_vmin), abs(computed_vmax))
        computed_vmin = -vmax_abs
        computed_vmax = vmax_abs

    logger.debug(f"Uniform colorbar limits: [{computed_vmin:.4f}, {computed_vmax:.4f}]")

    return computed_vmin, computed_vmax


def get_threshold_colorbar_limits(
    data_vmin: float,
    data_vmax: float,
    threshold: Optional[float],
    use_uniform: bool,
) -> Tuple[float, float]:
    """
    Determine colorbar limits considering threshold and uniform settings.

    Logic:
    - If use_uniform=True: Return data_vmin/data_vmax (global limits), ignoring threshold for range
    - If use_uniform=False and threshold is set: Return ±threshold as limits (local with threshold clipping)
    - If use_uniform=False and no threshold: Return data_vmin/data_vmax (local limits)

    The threshold affects which values are shown as black via set_over/set_under,
    but uniform colorbar should still show the full global range.

    Args:
        data_vmin: Data minimum (global if uniform, local if not)
        data_vmax: Data maximum (global if uniform, local if not)
        threshold: Amplitude threshold for black coloring
        use_uniform: Whether uniform colorbar is enabled

    Returns:
        (vmin, vmax) for the colorbar
    """
    if use_uniform:
        # Uniform colorbar: always use global data range
        return data_vmin, data_vmax
    elif threshold is not None:
        # Local colorbar with threshold: use ±threshold as limits
        # Values beyond will be shown as black via set_over/set_under
        return -threshold, threshold
    else:
        # Local colorbar without threshold: use data range
        return data_vmin, data_vmax
