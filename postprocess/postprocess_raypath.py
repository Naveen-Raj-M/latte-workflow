"""
Post-processing functions for LATTE ray path computation.
Traces ray paths using traveltime gradient fields and saves to HDF5 cache.

This module reads traveltime gradient binary files (gradient_px.bin, gradient_py.bin, gradient_pz.bin)
and traces ray paths from receivers back to sources using the steepest descent algorithm.

Visualization has been moved to the analysis module.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from omegaconf import DictConfig
from tqdm import tqdm
import warnings
from joblib import Parallel, delayed

from utils.io_utils import ensure_directory_exists, check_file_exists

from utils.io_utils import ensure_directory_exists, check_file_exists


def get_performance_setting(config: DictConfig, key: str, default):
    """
    Get performance setting from global performance config.

    Args:
        config: Configuration object
        key: Setting key (e.g., 'n_jobs', 'backend', 'batch_size', 'pre_dispatch')
        default: Default value if not found

    Returns:
        Setting value with proper fallback
    """
    # Check global performance section
    if hasattr(config, '_parent') and hasattr(config._parent, 'performance'):
        perf_config = config._parent.performance
        if hasattr(perf_config, key):
            return getattr(perf_config, key)
    elif hasattr(config, 'performance'):
        perf_config = config.performance
        if hasattr(perf_config, key):
            return getattr(perf_config, key)

    # Fallback to default
    return default


def get_tracing_setting(config: DictConfig, key: str, default):
    """
    Get ray tracing setting from global tracing config.

    Args:
        config: Configuration object
        key: Setting key (e.g., 'max_steps', 'step_size', 'tolerance', 'interpolation_method')
        default: Default value if not found

    Returns:
        Setting value with proper fallback
    """
    # Check global tracing section
    if hasattr(config, '_parent') and hasattr(config._parent, 'tracing'):
        tracing_config = config._parent.tracing
        if hasattr(tracing_config, key):
            return getattr(tracing_config, key)
    elif hasattr(config, 'tracing'):
        tracing_config = config.tracing
        if hasattr(tracing_config, key):
            return getattr(tracing_config, key)

    # Fallback to default
    return default


def read_gradient_metadata(gradient_dir: str, shot_id: int) -> Dict[str, float]:
    """
    Read gradient metadata file for a specific shot.

    Args:
        gradient_dir: Directory containing gradient files
        shot_id: Shot ID

    Returns:
        Dictionary containing local grid parameters
    """
    metadata_file = f"{gradient_dir}/shot_{shot_id}_gradient_metadata.txt"

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Gradient metadata file not found: {metadata_file}")

    metadata = {}
    with open(metadata_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            # Parse key = value format
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Convert to appropriate type
                if key in ['nx', 'ny', 'nz']:
                    metadata[key] = int(value)
                else:
                    metadata[key] = float(value)

    # CRITICAL FIX: LATTE writes grid spacing in METERS, but we need KILOMETERS
    # Convert dx, dy, dz from meters to kilometers
    if 'dx' in metadata:
        metadata['dx'] = metadata['dx'] / 1000.0
    if 'dy' in metadata:
        metadata['dy'] = metadata['dy'] / 1000.0
    if 'dz' in metadata:
        metadata['dz'] = metadata['dz'] / 1000.0

    # Convert origin coordinates from meters to kilometers
    if 'origin_x' in metadata:
        metadata['origin_x'] = metadata['origin_x'] / 1000.0
    if 'origin_y' in metadata:
        metadata['origin_y'] = metadata['origin_y'] / 1000.0
    if 'origin_z' in metadata:
        metadata['origin_z'] = metadata['origin_z'] / 1000.0

    return metadata


def read_gradient_binaries(gradient_dir: str, shot_id: int, phase: str = 'p',
                           precision: str = 'single') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read gradient binary files for a specific shot.

    Args:
        gradient_dir: Directory containing gradient files
        shot_id: Shot ID
        phase: 'p' or 's' wave
        precision: 'single' or 'double'

    Returns:
        Tuple of (grad_x, grad_y, grad_z) arrays
    """
    dtype = np.float64 if precision == 'double' else np.float32

    # File paths
    grad_px_file = f"{gradient_dir}/shot_{shot_id}_gradient_{phase}x.bin"
    grad_py_file = f"{gradient_dir}/shot_{shot_id}_gradient_{phase}y.bin"
    grad_pz_file = f"{gradient_dir}/shot_{shot_id}_gradient_{phase}z.bin"

    # Check if files exist
    if not os.path.exists(grad_px_file):
        raise FileNotFoundError(f"Gradient file not found: {grad_px_file}")

    # Read gradient files
    grad_x = np.fromfile(grad_px_file, dtype=dtype)
    grad_z = np.fromfile(grad_pz_file, dtype=dtype)

    # Check if 3D (has Y component)
    if os.path.exists(grad_py_file):
        grad_y = np.fromfile(grad_py_file, dtype=dtype)
    else:
        grad_y = None

    return grad_x, grad_y, grad_z


def reshape_gradients(grad_x: np.ndarray, grad_y: Optional[np.ndarray], grad_z: np.ndarray,
                     grid_dims: Dict[str, int]) -> Tuple[np.ndarray, ...]:
    """
    Reshape gradient arrays to 3D grid.

    Args:
        grad_x, grad_y, grad_z: Flattened gradient arrays
        grid_dims: Dictionary with nx, ny, nz, dx, dy, dz, origin_x, origin_y, origin_z

    Returns:
        Reshaped gradient arrays
    """
    nx, ny, nz = grid_dims['nx'], grid_dims['ny'], grid_dims['nz']

    if grad_y is not None:
        # 3D case - use Fortran order since LATTE writes in column-major format
        grad_x = grad_x.reshape((nz, ny, nx), order='F')
        grad_y = grad_y.reshape((nz, ny, nx), order='F')
        grad_z = grad_z.reshape((nz, ny, nx), order='F')
        return grad_x, grad_y, grad_z
    else:
        # 2D case - use Fortran order since LATTE writes in column-major format
        grad_x = grad_x.reshape((nz, nx), order='F')
        grad_z = grad_z.reshape((nz, nx), order='F')
        return grad_x, None, grad_z


def interpolate_gradient_trilinear(grad_x: np.ndarray, grad_y: Optional[np.ndarray], grad_z: np.ndarray,
                                   pos: np.ndarray, grid_dims: Dict[str, int], is_3d: bool) -> np.ndarray:
    """
    Interpolate gradient values using trilinear interpolation.

    Args:
        grad_x, grad_y, grad_z: Gradient field arrays
        pos: Current position [x, y, z] in km
        grid_dims: Grid dimensions dictionary
        is_3d: Whether this is a 3D problem

    Returns:
        Interpolated gradient vector [gx, gy, gz]
    """
    nx, ny, nz = grid_dims['nx'], grid_dims['ny'], grid_dims['nz']
    dx, dy, dz = grid_dims['dx'], grid_dims['dy'], grid_dims['dz']
    ox, oy, oz = grid_dims['origin_x'], grid_dims['origin_y'], grid_dims['origin_z']

    if is_3d:
        # Compute continuous grid coordinates
        x_coord = (pos[0] - ox) / dx
        y_coord = (pos[1] - oy) / dy
        z_coord = (pos[2] - oz) / dz

        # Get integer indices (floor)
        i0 = int(np.floor(x_coord))
        j0 = int(np.floor(y_coord))
        k0 = int(np.floor(z_coord))

        # Get upper indices (clamp to grid bounds)
        i1 = min(i0 + 1, nx - 1)
        j1 = min(j0 + 1, ny - 1)
        k1 = min(k0 + 1, nz - 1)

        # Clamp lower indices too
        i0 = max(0, i0)
        j0 = max(0, j0)
        k0 = max(0, k0)

        # Compute interpolation weights
        wx = x_coord - i0
        wy = y_coord - j0
        wz = z_coord - k0

        # Trilinear interpolation for each gradient component
        # Sample at 8 corners of the cube
        gx = (1-wx)*(1-wy)*(1-wz)*grad_x[k0, j0, i0] + \
             (  wx)*(1-wy)*(1-wz)*grad_x[k0, j0, i1] + \
             (1-wx)*(  wy)*(1-wz)*grad_x[k0, j1, i0] + \
             (  wx)*(  wy)*(1-wz)*grad_x[k0, j1, i1] + \
             (1-wx)*(1-wy)*(  wz)*grad_x[k1, j0, i0] + \
             (  wx)*(1-wy)*(  wz)*grad_x[k1, j0, i1] + \
             (1-wx)*(  wy)*(  wz)*grad_x[k1, j1, i0] + \
             (  wx)*(  wy)*(  wz)*grad_x[k1, j1, i1]

        gy = (1-wx)*(1-wy)*(1-wz)*grad_y[k0, j0, i0] + \
             (  wx)*(1-wy)*(1-wz)*grad_y[k0, j0, i1] + \
             (1-wx)*(  wy)*(1-wz)*grad_y[k0, j1, i0] + \
             (  wx)*(  wy)*(1-wz)*grad_y[k0, j1, i1] + \
             (1-wx)*(1-wy)*(  wz)*grad_y[k1, j0, i0] + \
             (  wx)*(1-wy)*(  wz)*grad_y[k1, j0, i1] + \
             (1-wx)*(  wy)*(  wz)*grad_y[k1, j1, i0] + \
             (  wx)*(  wy)*(  wz)*grad_y[k1, j1, i1]

        gz = (1-wx)*(1-wy)*(1-wz)*grad_z[k0, j0, i0] + \
             (  wx)*(1-wy)*(1-wz)*grad_z[k0, j0, i1] + \
             (1-wx)*(  wy)*(1-wz)*grad_z[k0, j1, i0] + \
             (  wx)*(  wy)*(1-wz)*grad_z[k0, j1, i1] + \
             (1-wx)*(1-wy)*(  wz)*grad_z[k1, j0, i0] + \
             (  wx)*(1-wy)*(  wz)*grad_z[k1, j0, i1] + \
             (1-wx)*(  wy)*(  wz)*grad_z[k1, j1, i0] + \
             (  wx)*(  wy)*(  wz)*grad_z[k1, j1, i1]

        return np.array([gx, gy, gz])
    else:
        # 2D case: bilinear interpolation
        x_coord = (pos[0] - ox) / dx
        z_coord = (pos[2] - oz) / dz

        # Get integer indices (floor)
        i0 = int(np.floor(x_coord))
        k0 = int(np.floor(z_coord))

        # Get upper indices (clamp to grid bounds)
        i1 = min(i0 + 1, nx - 1)
        k1 = min(k0 + 1, nz - 1)

        # Clamp lower indices too
        i0 = max(0, i0)
        k0 = max(0, k0)

        # Compute interpolation weights
        wx = x_coord - i0
        wz = z_coord - k0

        # Bilinear interpolation
        gx = (1-wx)*(1-wz)*grad_x[k0, i0] + \
             (  wx)*(1-wz)*grad_x[k0, i1] + \
             (1-wx)*(  wz)*grad_x[k1, i0] + \
             (  wx)*(  wz)*grad_x[k1, i1]

        gz = (1-wx)*(1-wz)*grad_z[k0, i0] + \
             (  wx)*(1-wz)*grad_z[k0, i1] + \
             (1-wx)*(  wz)*grad_z[k1, i0] + \
             (  wx)*(  wz)*grad_z[k1, i1]

        return np.array([gx, 0.0, gz])


def compute_adaptive_step_size(gradient_mag: float, base_step: float, min_step: float, max_step: float, scaling: str = 'inverse') -> float:
    """
    Compute adaptive step size based on gradient magnitude.

    Small gradients -> larger steps to traverse flat regions faster
    Large gradients -> smaller steps for precision

    Args:
        gradient_mag: Magnitude of the gradient
        base_step: Base step size
        min_step: Minimum allowed step size
        max_step: Maximum allowed step size
        scaling: Scaling method ('inverse', 'sqrt_inverse', or 'fixed')

    Returns:
        Adaptive step size in km
    """
    if scaling == 'fixed':
        return base_step
    elif scaling == 'sqrt_inverse':
        # Square root inverse scaling (gentler adaptation)
        if gradient_mag > 1e-6:
            step = base_step / np.sqrt(gradient_mag)
        else:
            step = max_step
    else:  # 'inverse'
        # Inverse scaling (stronger adaptation)
        if gradient_mag > 1e-6:
            step = base_step / gradient_mag
        else:
            step = max_step

    return np.clip(step, min_step, max_step)


def check_convergence(current_pos: np.ndarray, source_pos: np.ndarray,
                      prev_distance: float, oscillation_count: int,
                      max_oscillations: int, tolerance: float, oscillation_tol: float) -> Tuple[Optional[bool], int, str]:
    """
    Monitor ray progress and detect failure modes.

    Args:
        current_pos: Current position
        source_pos: Target source position
        prev_distance: Distance at previous step
        oscillation_count: Current oscillation counter
        max_oscillations: Maximum allowed oscillations
        tolerance: Distance tolerance for success
        oscillation_tol: Distance change threshold for oscillation

    Returns:
        Tuple of (converged/failed/None, updated oscillation_count, status_message)
    """
    current_distance = np.linalg.norm(current_pos - source_pos)

    # Check if distance is decreasing (with small tolerance for numerical noise)
    distance_change = prev_distance - current_distance

    if distance_change < oscillation_tol:
        # Not making progress
        oscillation_count += 1
    else:
        # Making progress, reset counter
        oscillation_count = 0

    # Terminate if stuck oscillating
    if oscillation_count > max_oscillations:
        return False, oscillation_count, "STUCK_OSCILLATING"

    # Success if within tolerance
    if current_distance < tolerance:
        return True, oscillation_count, "SUCCESS"

    return None, oscillation_count, "IN_PROGRESS"


def compute_hybrid_direction(gradient: np.ndarray, current_pos: np.ndarray,
                            source_pos: np.ndarray, alpha: float,
                            adaptive: bool, distance_threshold: float,
                            gradient_mag: float, min_gradient: float) -> np.ndarray:
    """
    Blend gradient direction with direct source direction.

    Args:
        gradient: Gradient vector (not normalized)
        current_pos: Current position
        source_pos: Target source position
        alpha: Weight for gradient direction (0=direct, 1=pure gradient)
        adaptive: Whether to adaptively adjust alpha
        distance_threshold: Distance threshold for adaptive alpha
        gradient_mag: Magnitude of gradient
        min_gradient: Minimum gradient threshold

    Returns:
        Combined direction vector (normalized)
    """
    # Gradient-based direction (negative gradient points toward source)
    grad_mag = np.linalg.norm(gradient)
    if grad_mag > 1e-10:
        grad_dir = -gradient / grad_mag
    else:
        grad_dir = np.zeros_like(gradient)

    # Direct source direction
    source_vec = source_pos - current_pos
    source_dist = np.linalg.norm(source_vec)
    if source_dist > 1e-10:
        source_dir = source_vec / source_dist
    else:
        return grad_dir

    # Adaptive alpha adjustment
    effective_alpha = alpha
    if adaptive:
        # Reduce alpha (use more direct direction) when:
        # 1. Far from source
        # 2. Gradient is weak
        if source_dist > distance_threshold:
            effective_alpha *= 0.5
        if gradient_mag < min_gradient * 10:
            effective_alpha *= 0.3

    # Blend directions
    combined_dir = effective_alpha * grad_dir + (1 - effective_alpha) * source_dir
    combined_mag = np.linalg.norm(combined_dir)

    if combined_mag > 1e-10:
        return combined_dir / combined_mag
    else:
        return source_dir


def trace_ray_path(source_pos: Tuple[float, float, float],
                   receiver_pos: Tuple[float, float, float],
                   grad_x: np.ndarray, grad_y: Optional[np.ndarray], grad_z: np.ndarray,
                   grid_dims: Dict[str, int],
                   max_steps: int = 10000,
                   step_size: float = 0.1,
                   tolerance: float = 0.5,
                   interpolation_method: str = 'nearest',
                   # Advanced options (from RAY_PATH_ISSUES.md improvements)
                   adaptive_step_size: bool = False,
                   min_step_size: float = 0.01,
                   max_step_size: float = 1.0,
                   step_size_scaling: str = 'inverse',
                   min_gradient_magnitude: float = 1e-6,
                   use_direct_fallback: bool = False,
                   enable_convergence_check: bool = False,
                   max_oscillations: int = 100,
                   oscillation_tolerance: float = 1e-3,
                   hybrid_mode: bool = False,
                   hybrid_alpha: float = 0.7,
                   hybrid_adaptive_alpha: bool = False,
                   hybrid_distance_threshold: float = 100.0,
                   momentum_mode: bool = False,
                   momentum_beta: float = 0.9,
                   momentum_learning_rate: float = 0.1,
                   diagnostic_mode: bool = False) -> np.ndarray:
    """
    Trace ray path from receiver to source using improved gradient descent.

    This function implements the improvements outlined in RAY_PATH_ISSUES.md:
    - Priority 1: Adaptive step sizing, gradient magnitude thresholding, convergence monitoring
    - Priority 2: Hybrid ray tracing, momentum-based tracing, diagnostic output

    The ray follows the negative gradient of the traveltime field:
    dx/ds = -∇T / |∇T|

    Args:
        source_pos: (x, y, z) source coordinates in km
        receiver_pos: (x, y, z) receiver coordinates in km
        grad_x, grad_y, grad_z: Gradient fields (∂T/∂x, ∂T/∂y, ∂T/∂z)
        grid_dims: Grid dimensions and spacing
        max_steps: Maximum number of integration steps
        step_size: Integration step size in km (base step for adaptive mode)
        tolerance: Distance tolerance to source in km
        interpolation_method: 'nearest' or 'trilinear'

        # Priority 1 improvements
        adaptive_step_size: Enable adaptive step sizing based on gradient magnitude
        min_step_size: Minimum step size in km
        max_step_size: Maximum step size in km
        step_size_scaling: Scaling method ('inverse', 'sqrt_inverse', 'fixed')
        min_gradient_magnitude: Minimum reliable gradient magnitude (s/km)
        use_direct_fallback: Fall back to direct source direction if gradient too small
        enable_convergence_check: Monitor convergence and detect oscillations
        max_oscillations: Maximum consecutive non-improving steps
        oscillation_tolerance: Distance change threshold for oscillation detection (km)

        # Priority 2 improvements
        hybrid_mode: Blend gradient and direct source directions
        hybrid_alpha: Weight for gradient direction (0=direct, 1=pure gradient)
        hybrid_adaptive_alpha: Adaptively adjust alpha based on distance/gradient
        hybrid_distance_threshold: Distance (km) for adaptive alpha adjustment
        momentum_mode: Use momentum to smooth oscillations
        momentum_beta: Momentum coefficient (0=no momentum, 1=full momentum)
        momentum_learning_rate: Learning rate for momentum updates
        diagnostic_mode: Enable detailed diagnostic logging

    Returns:
        Array of ray path coordinates (n_points, 3)
        If diagnostic_mode is True, also stores diagnostic info in ray_path.diagnostics
    """
    is_3d = grad_y is not None

    # Extract grid parameters
    nx, ny, nz = grid_dims['nx'], grid_dims['ny'], grid_dims['nz']
    dx, dy, dz = grid_dims['dx'], grid_dims['dy'], grid_dims['dz']
    ox, oy, oz = grid_dims['origin_x'], grid_dims['origin_y'], grid_dims['origin_z']

    # Initialize ray path and tracking variables
    ray_path = [receiver_pos]
    current_pos = np.array(receiver_pos, dtype=np.float64)
    source_pos_arr = np.array(source_pos, dtype=np.float64)

    # Convergence monitoring
    prev_distance = np.linalg.norm(current_pos - source_pos_arr)
    oscillation_count = 0
    termination_reason = "MAX_STEPS"

    # Momentum tracking
    velocity = np.zeros(3, dtype=np.float64) if momentum_mode else None

    # Diagnostic tracking
    diagnostics = {
        'gradient_magnitudes': [],
        'distances_to_source': [],
        'step_sizes': [],
        'termination_reason': '',
        'final_distance': 0.0,
        'oscillation_count': 0,
        'gradient_fallback_count': 0
    } if diagnostic_mode else None

    for step in range(max_steps):
        # Check if we've reached the source
        dist_to_source = np.linalg.norm(current_pos - source_pos_arr)

        if diagnostic_mode:
            diagnostics['distances_to_source'].append(dist_to_source)

        if dist_to_source < tolerance:
            ray_path.append(source_pos)
            termination_reason = "SUCCESS"
            if diagnostic_mode:
                diagnostics['termination_reason'] = termination_reason
                diagnostics['final_distance'] = dist_to_source
            break

        # Convergence check (Priority 1)
        if enable_convergence_check:
            converged, oscillation_count, status = check_convergence(
                current_pos, source_pos_arr, prev_distance,
                oscillation_count, max_oscillations, tolerance, oscillation_tolerance
            )

            if converged is True:
                ray_path.append(source_pos)
                termination_reason = "SUCCESS"
                if diagnostic_mode:
                    diagnostics['termination_reason'] = termination_reason
                    diagnostics['final_distance'] = dist_to_source
                    diagnostics['oscillation_count'] = oscillation_count
                break
            elif converged is False:
                termination_reason = status
                if diagnostic_mode:
                    diagnostics['termination_reason'] = termination_reason
                    diagnostics['final_distance'] = dist_to_source
                    diagnostics['oscillation_count'] = oscillation_count
                break

            prev_distance = dist_to_source

        # Get gradient at current position
        if interpolation_method == 'trilinear':
            gradient = interpolate_gradient_trilinear(
                grad_x, grad_y, grad_z, current_pos, grid_dims, is_3d
            )

            # Check bounds
            if is_3d:
                x_coord = (current_pos[0] - ox) / dx
                y_coord = (current_pos[1] - oy) / dy
                z_coord = (current_pos[2] - oz) / dz
                if not (-0.5 <= x_coord <= nx - 0.5 and -0.5 <= y_coord <= ny - 0.5 and -0.5 <= z_coord <= nz - 0.5):
                    termination_reason = "OUT_OF_BOUNDS"
                    if diagnostic_mode:
                        diagnostics['termination_reason'] = termination_reason
                        diagnostics['final_distance'] = dist_to_source
                    break
            else:
                x_coord = (current_pos[0] - ox) / dx
                z_coord = (current_pos[2] - oz) / dz
                if not (-0.5 <= x_coord <= nx - 0.5 and -0.5 <= z_coord <= nz - 0.5):
                    termination_reason = "OUT_OF_BOUNDS"
                    if diagnostic_mode:
                        diagnostics['termination_reason'] = termination_reason
                        diagnostics['final_distance'] = dist_to_source
                    break
        else:
            # Nearest neighbor interpolation
            if is_3d:
                i = int((current_pos[0] - ox) / dx)
                j = int((current_pos[1] - oy) / dy)
                k = int((current_pos[2] - oz) / dz)

                if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
                    termination_reason = "OUT_OF_BOUNDS"
                    if diagnostic_mode:
                        diagnostics['termination_reason'] = termination_reason
                        diagnostics['final_distance'] = dist_to_source
                    break

                gx = grad_x[k, j, i]
                gy = grad_y[k, j, i]
                gz = grad_z[k, j, i]
                gradient = np.array([gx, gy, gz])
            else:
                i = int((current_pos[0] - ox) / dx)
                k = int((current_pos[2] - oz) / dz)

                if not (0 <= i < nx and 0 <= k < nz):
                    termination_reason = "OUT_OF_BOUNDS"
                    if diagnostic_mode:
                        diagnostics['termination_reason'] = termination_reason
                        diagnostics['final_distance'] = dist_to_source
                    break

                gx = grad_x[k, i]
                gz = grad_z[k, i]
                gradient = np.array([gx, 0.0, gz])

        # Check gradient magnitude (Priority 1)
        grad_magnitude = np.linalg.norm(gradient)

        if diagnostic_mode:
            diagnostics['gradient_magnitudes'].append(grad_magnitude)

        # Gradient magnitude thresholding with direct fallback
        if grad_magnitude < min_gradient_magnitude and use_direct_fallback:
            # Fall back to direct source direction
            source_vec = source_pos_arr - current_pos
            source_dist = np.linalg.norm(source_vec)
            if source_dist > 1e-10:
                direction = source_vec / source_dist
            else:
                termination_reason = "GRADIENT_TOO_SMALL"
                if diagnostic_mode:
                    diagnostics['termination_reason'] = termination_reason
                    diagnostics['final_distance'] = dist_to_source
                    diagnostics['gradient_fallback_count'] = diagnostics.get('gradient_fallback_count', 0) + 1
                break

            if diagnostic_mode:
                diagnostics['gradient_fallback_count'] = diagnostics.get('gradient_fallback_count', 0) + 1
        elif grad_magnitude < 1e-10:
            # Gradient too small even without threshold
            termination_reason = "GRADIENT_TOO_SMALL"
            if diagnostic_mode:
                diagnostics['termination_reason'] = termination_reason
                diagnostics['final_distance'] = dist_to_source
            break
        else:
            # Compute direction based on mode
            if hybrid_mode:
                # Hybrid mode: blend gradient and direct directions (Priority 2)
                direction = compute_hybrid_direction(
                    gradient, current_pos, source_pos_arr,
                    hybrid_alpha, hybrid_adaptive_alpha,
                    hybrid_distance_threshold, grad_magnitude,
                    min_gradient_magnitude
                )
            else:
                # Standard gradient descent
                direction = -gradient / grad_magnitude

        # Compute step size
        if adaptive_step_size:
            # Adaptive step sizing (Priority 1)
            current_step = compute_adaptive_step_size(
                grad_magnitude, step_size, min_step_size,
                max_step_size, step_size_scaling
            )
        else:
            current_step = step_size

        if diagnostic_mode:
            diagnostics['step_sizes'].append(current_step)

        # Update position
        if momentum_mode:
            # Momentum-based update (Priority 2)
            velocity = momentum_beta * velocity + momentum_learning_rate * direction
            new_pos = current_pos + velocity
        else:
            # Standard update
            new_pos = current_pos + current_step * direction

        # In 2D case, keep Y coordinate fixed
        if not is_3d:
            new_pos[1] = current_pos[1]

        current_pos = new_pos
        ray_path.append(tuple(current_pos))

    # Store diagnostics if enabled
    if diagnostic_mode:
        diagnostics['final_distance'] = np.linalg.norm(current_pos - source_pos_arr)
        if not diagnostics['termination_reason']:
            diagnostics['termination_reason'] = termination_reason

    ray_path_array = np.array(ray_path)

    # Attach diagnostics as attribute if enabled
    if diagnostic_mode:
        ray_path_array.diagnostics = diagnostics

    return ray_path_array


def load_shot_mapping(shot_map_file: str) -> Dict[int, str]:
    """Load shot ID to event ID mapping."""
    shot_map = {}
    df = pd.read_csv(shot_map_file, sep='\t')
    for _, row in df.iterrows():
        shot_map[int(row['Shot_ID'])] = str(row['Original_Event_ID'])
    return shot_map


def get_advanced_tracing_params(config: DictConfig) -> Dict:
    """
    Extract advanced ray tracing parameters from configuration.

    Args:
        config: Configuration object

    Returns:
        Dictionary of advanced ray tracing parameters
    """
    return {
        # Priority 1 improvements
        'adaptive_step_size': get_tracing_setting(config, 'adaptive_step_size', False),
        'min_step_size': get_tracing_setting(config, 'min_step_size', 0.01),
        'max_step_size': get_tracing_setting(config, 'max_step_size', 1.0),
        'step_size_scaling': get_tracing_setting(config, 'step_size_scaling', 'inverse'),
        'min_gradient_magnitude': get_tracing_setting(config, 'min_gradient_magnitude', 1e-6),
        'use_direct_fallback': get_tracing_setting(config, 'use_direct_fallback', False),
        'enable_convergence_check': get_tracing_setting(config, 'enable_convergence_check', False),
        'max_oscillations': get_tracing_setting(config, 'max_oscillations', 100),
        'oscillation_tolerance': get_tracing_setting(config, 'oscillation_tolerance', 1e-3),

        # Priority 2 improvements
        'hybrid_mode': get_tracing_setting(config, 'hybrid_mode', False),
        'hybrid_alpha': get_tracing_setting(config, 'hybrid_alpha', 0.7),
        'hybrid_adaptive_alpha': get_tracing_setting(config, 'hybrid_adaptive_alpha', False),
        'hybrid_distance_threshold': get_tracing_setting(config, 'hybrid_distance_threshold', 100.0),
        'momentum_mode': get_tracing_setting(config, 'momentum_mode', False),
        'momentum_beta': get_tracing_setting(config, 'momentum_beta', 0.9),
        'momentum_learning_rate': get_tracing_setting(config, 'momentum_learning_rate', 0.1),
        'diagnostic_mode': get_tracing_setting(config, 'diagnostic_mode', False)
    }


def process_single_ray(shot_id: int,
                       source_pos: Tuple[float, float, float],
                       receiver_pos: Tuple[float, float, float],
                       gradient_dir: str,
                       phase: str,
                       precision: str,
                       tracing_params: Dict,
                       interpolation_method: str = 'nearest',
                       advanced_params: Optional[Dict] = None) -> Optional[np.ndarray]:
    """
    Process a single shot-receiver pair and trace its ray path.

    This function is designed for fine-grained parallelization over individual rays.

    Args:
        shot_id: Shot ID
        source_pos: Source position (x, y, z) in km
        receiver_pos: Receiver position (x, y, z) in km
        gradient_dir: Directory containing gradient files
        phase: Wave phase ('p' or 's')
        precision: Numerical precision ('single' or 'double')
        tracing_params: Ray tracing parameters dictionary
        interpolation_method: 'nearest' or 'trilinear' interpolation method
        advanced_params: Optional dictionary with advanced ray tracing parameters

    Returns:
        Ray path array or None if failed
    """
    # Default advanced parameters if not provided
    if advanced_params is None:
        advanced_params = {}

    try:
        # Read gradient metadata to get the local grid parameters for this shot
        metadata = read_gradient_metadata(gradient_dir, shot_id)

        # Create local grid_dims from metadata
        local_grid_dims = {
            'nx': metadata['nx'],
            'ny': metadata['ny'],
            'nz': metadata['nz'],
            'dx': metadata['dx'],
            'dy': metadata['dy'],
            'dz': metadata['dz'],
            'origin_x': metadata['origin_x'],
            'origin_y': metadata['origin_y'],
            'origin_z': metadata['origin_z']
        }

        # Read gradient files
        grad_x, grad_y, grad_z = read_gradient_binaries(
            gradient_dir, shot_id, phase, precision
        )

        # Reshape gradients using the local grid dimensions
        grad_x, grad_y, grad_z = reshape_gradients(
            grad_x, grad_y, grad_z, local_grid_dims
        )

        # Trace ray path with advanced parameters
        ray_path = trace_ray_path(
            source_pos,
            receiver_pos,
            grad_x, grad_y, grad_z,
            local_grid_dims,
            max_steps=tracing_params['max_steps'],
            step_size=tracing_params['step_size'],
            tolerance=tracing_params['tolerance'],
            interpolation_method=interpolation_method,
            **advanced_params  # Pass all advanced parameters
        )

        return ray_path if len(ray_path) > 1 else None

    except Exception:
        return None


def process_single_shot(shot_id: int,
                        df_gt: pd.DataFrame,
                        gradient_dir: str,
                        phase: str,
                        precision: str,
                        grid_dims: Dict,  # Not used - kept for backward compatibility
                        tracing_params: Dict,
                        verbose: bool,
                        interpolation_method: str = 'nearest',
                        advanced_params: Optional[Dict] = None) -> Tuple[List[np.ndarray], Tuple, List[Tuple]]:
    """
    Process a single shot and trace all its ray paths.

    Args:
        shot_id: Shot ID to process
        df_gt: Ground truth dataframe
        gradient_dir: Directory containing gradient files
        phase: Wave phase ('p' or 's')
        precision: Numerical precision ('single' or 'double')
        grid_dims: Grid dimension dictionary (will be overridden by metadata)
        tracing_params: Ray tracing parameters dictionary
        verbose: Whether to print warnings
        interpolation_method: 'nearest' or 'trilinear' interpolation method
        advanced_params: Optional dictionary with advanced ray tracing parameters

    Returns:
        Tuple of (ray_paths, source_position, receiver_positions)
    """
    import time
    t_start = time.time()

    # Default advanced parameters if not provided
    if advanced_params is None:
        advanced_params = {}

    try:
        # Read gradient metadata to get the local grid parameters for this shot
        metadata = read_gradient_metadata(gradient_dir, shot_id)

        # Create local grid_dims from metadata (overrides the global grid_dims)
        local_grid_dims = {
            'nx': metadata['nx'],
            'ny': metadata['ny'],
            'nz': metadata['nz'],
            'dx': metadata['dx'],
            'dy': metadata['dy'],
            'dz': metadata['dz'],
            'origin_x': metadata['origin_x'],
            'origin_y': metadata['origin_y'],
            'origin_z': metadata['origin_z']
        }

        # Read gradient files
        grad_x, grad_y, grad_z = read_gradient_binaries(
            gradient_dir, shot_id, phase, precision
        )

        # Reshape gradients using the local grid dimensions
        grad_x, grad_y, grad_z = reshape_gradients(
            grad_x, grad_y, grad_z, local_grid_dims
        )

        # Get source-receiver pairs for this shot
        shot_data = df_gt[df_gt['shot_id'] == shot_id]

        if len(shot_data) == 0:
            return [], None, []

        # Get source position
        source_row = shot_data.iloc[0]
        source_pos = (
            source_row['event_x_transformed'],
            source_row['event_y_transformed'],
            source_row['event_z_transformed']
        )

        ray_paths = []
        receiver_positions = []

        # Trace rays for each receiver
        slow_rays_count = 0
        for _, row in shot_data.iterrows():
            receiver_pos = (
                row['station_x_transformed'],
                row['station_y_transformed'],
                row['station_z_transformed']
            )
            receiver_positions.append(receiver_pos)

            # Trace ray path using the local grid dimensions for this shot
            ray_path = trace_ray_path(
                source_pos,
                receiver_pos,
                grad_x, grad_y, grad_z,
                local_grid_dims,
                max_steps=tracing_params['max_steps'],
                step_size=tracing_params['step_size'],
                tolerance=tracing_params['tolerance'],
                interpolation_method=interpolation_method,
                **advanced_params  # Pass all advanced parameters
            )

            if len(ray_path) > 1:
                ray_paths.append(ray_path)
                # Track rays that hit max_steps (potentially problematic)
                if len(ray_path) >= tracing_params['max_steps'] * 0.9:
                    slow_rays_count += 1

        # Warn if shot took too long to process
        elapsed = time.time() - t_start
        if verbose and elapsed > 5.0:
            print(f"  WARNING: Shot {shot_id} took {elapsed:.1f}s ({len(ray_paths)} rays, {slow_rays_count} hit max_steps)")

        return ray_paths, source_pos, receiver_positions

    except FileNotFoundError:
        if verbose:
            warnings.warn(f"Gradient files not found for shot {shot_id}")
        return [], None, []
    except Exception as e:
        if verbose:
            warnings.warn(f"Error processing shot {shot_id}: {e}")
        return [], None, []

