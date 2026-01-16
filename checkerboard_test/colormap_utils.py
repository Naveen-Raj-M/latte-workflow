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
