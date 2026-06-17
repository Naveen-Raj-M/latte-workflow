"""
Create spike velocity model for resolution testing.

This module reads an input velocity model CSV, superimposes one or more localized
Gaussian velocity anomalies ("spikes"), and outputs:
- Updated CSV with perturbed velocities
- Binary files (vp_spike.bin, vs_spike.bin) for LATTE

Three anomaly kinds are supported:
- "3d"        : axis-aligned 3D Gaussian ellipsoid, sigma = (sigma_x, sigma_y, sigma_z)
- "2d"        : in-plane Gaussian blob confined to a single depth layer,
                sigma = (sigma_x, sigma_y); the z value of the center picks the layer
- "arbitrary" : Gaussian ellipsoid whose major axis follows an arbitrary direction
                vector, sigma = (sigma_major, sigma_minor1, sigma_minor2)

Each anomaly has independent amplitude and polarity for P and S. Perturbations from
overlapping anomalies are summed before being applied multiplicatively to the model.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.velocity_io import read_velocity_file, write_velocity_file

# Grid/binary helpers are perturbation-agnostic and reused from the checkerboard module.
from checkerboard_test.create_checkerboard_model import (
    get_grid_dimensions,
    write_velocity_binary,
)

logger = logging.getLogger(__name__)

_EPS = 1e-12


def _signed_amplitude(amplitude: float, polarity: str, anomaly_name: str, field: str) -> float:
    """Return the amplitude with sign applied from the polarity string."""
    if polarity not in ("positive", "negative"):
        raise ValueError(
            f"Anomaly {anomaly_name!r}: unknown polarity_{field}={polarity!r}. "
            "Must be 'positive' or 'negative'."
        )
    return float(amplitude) if polarity == "positive" else -float(amplitude)


def _apply_truncation(
    shape: np.ndarray,
    squared_radius: np.ndarray,
    truncate_at_sigma: Optional[float],
) -> np.ndarray:
    """
    Zero the Gaussian shape beyond ``truncate_at_sigma`` standard deviations.

    Args:
        shape: Unit Gaussian shape array (peak 1 at center)
        squared_radius: Normalized squared radius (sum of (d_i / sigma_i)**2)
        truncate_at_sigma: Cutoff in standard deviations, or None for no truncation

    Returns:
        The (possibly) truncated shape array
    """
    if truncate_at_sigma is not None:
        shape = np.where(squared_radius > float(truncate_at_sigma) ** 2, 0.0, shape)
    return shape


def gaussian_3d_shape(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    center: Tuple[float, float, float],
    sigma: Tuple[float, float, float],
    truncate_at_sigma: Optional[float] = None,
) -> np.ndarray:
    """
    Unit (peak 1) axis-aligned 3D Gaussian shape.

    Args:
        x, y, z: Coordinate arrays (km)
        center: (cx, cy, cz) anomaly center (km)
        sigma: (sigma_x, sigma_y, sigma_z) standard deviations (km)
        truncate_at_sigma: Optional cutoff in standard deviations

    Returns:
        Shape array in [0, 1]
    """
    cx, cy, cz = center
    sx, sy, sz = sigma
    if min(sx, sy, sz) <= 0:
        raise ValueError(f"3d anomaly sigma values must be positive, got {sigma}")

    squared_radius = (
        ((x - cx) / sx) ** 2
        + ((y - cy) / sy) ** 2
        + ((z - cz) / sz) ** 2
    )
    shape = np.exp(-0.5 * squared_radius)
    return _apply_truncation(shape, squared_radius, truncate_at_sigma)


def gaussian_2d_layer_shape(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    center: Tuple[float, float, float],
    sigma: Tuple[float, float],
    truncate_at_sigma: Optional[float] = None,
) -> np.ndarray:
    """
    Unit (peak 1) in-plane 2D Gaussian confined to a single depth layer.

    The depth layer is the grid layer nearest to ``center[2]``. All points off that
    layer receive zero perturbation.

    Args:
        x, y, z: Coordinate arrays (km)
        center: (cx, cy, cz) anomaly center (km); cz selects the depth layer
        sigma: (sigma_x, sigma_y) in-plane standard deviations (km)
        truncate_at_sigma: Optional cutoff in standard deviations

    Returns:
        Shape array in [0, 1] (nonzero only on the selected layer)
    """
    cx, cy, cz = center
    sx, sy = sigma
    if min(sx, sy) <= 0:
        raise ValueError(f"2d anomaly sigma values must be positive, got {sigma}")

    # Snap to nearest depth layer present in the grid.
    z_unique = np.unique(z)
    nearest_z = z_unique[np.argmin(np.abs(z_unique - cz))]
    layer_mask = np.isclose(z, nearest_z, atol=1e-6)

    shape = np.zeros_like(x, dtype=np.float64)
    if not layer_mask.any():
        logger.warning(f"2d anomaly: no grid points near depth {cz} km — skipping.")
        return shape

    if not np.isclose(nearest_z, cz, atol=1e-6):
        logger.info(f"  2d anomaly depth {cz} km snapped to nearest layer {nearest_z:.3f} km")

    squared_radius = (
        ((x[layer_mask] - cx) / sx) ** 2
        + ((y[layer_mask] - cy) / sy) ** 2
    )
    layer_shape = np.exp(-0.5 * squared_radius)
    layer_shape = _apply_truncation(layer_shape, squared_radius, truncate_at_sigma)
    shape[layer_mask] = layer_shape
    return shape


def _orthonormal_basis(
    direction: np.ndarray,
    minor1_direction: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a right-handed orthonormal basis whose first axis is the major axis.

    Args:
        direction: Major-axis direction vector (need not be unit length)
        minor1_direction: Optional vector used to fix the first minor axis. It is
            orthogonalized against the major axis. If omitted (or degenerate), a
            reference axis least aligned with the major axis is used instead.

    Returns:
        (e_major, e_minor1, e_minor2) orthonormal vectors
    """
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < _EPS:
        raise ValueError("arbitrary anomaly: direction vector must be non-zero")
    e_major = direction / norm

    candidate = None
    if minor1_direction is not None:
        candidate = np.asarray(minor1_direction, dtype=np.float64)
        candidate = candidate - np.dot(candidate, e_major) * e_major
        if np.linalg.norm(candidate) < _EPS:
            logger.warning(
                "arbitrary anomaly: minor1_direction is parallel to the major axis — "
                "choosing minor axes automatically."
            )
            candidate = None

    if candidate is None:
        # Pick the global axis least aligned with the major axis for numerical stability.
        axes = np.eye(3)
        ref = axes[int(np.argmin(np.abs(e_major)))]
        candidate = ref - np.dot(ref, e_major) * e_major

    e_minor1 = candidate / np.linalg.norm(candidate)
    e_minor2 = np.cross(e_major, e_minor1)
    e_minor2 = e_minor2 / np.linalg.norm(e_minor2)
    return e_major, e_minor1, e_minor2


def gaussian_arbitrary_shape(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    center: Tuple[float, float, float],
    direction: Tuple[float, float, float],
    sigma: Tuple[float, float, float],
    minor1_direction: Optional[Tuple[float, float, float]] = None,
    truncate_at_sigma: Optional[float] = None,
) -> np.ndarray:
    """
    Unit (peak 1) 3D Gaussian ellipsoid oriented along an arbitrary direction.

    The major axis follows ``direction``; two minor axes are taken perpendicular to it.

    Args:
        x, y, z: Coordinate arrays (km)
        center: (cx, cy, cz) anomaly center (km)
        direction: Major-axis direction vector
        sigma: (sigma_major, sigma_minor1, sigma_minor2) standard deviations (km)
        minor1_direction: Optional vector fixing the first minor axis orientation
        truncate_at_sigma: Optional cutoff in standard deviations

    Returns:
        Shape array in [0, 1]
    """
    cx, cy, cz = center
    s_major, s_minor1, s_minor2 = sigma
    if min(s_major, s_minor1, s_minor2) <= 0:
        raise ValueError(f"arbitrary anomaly sigma values must be positive, got {sigma}")

    e_major, e_minor1, e_minor2 = _orthonormal_basis(direction, minor1_direction)

    # Offsets from center, projected onto the rotated axes.
    dx = x - cx
    dy = y - cy
    dz = z - cz
    u = dx * e_major[0] + dy * e_major[1] + dz * e_major[2]
    v = dx * e_minor1[0] + dy * e_minor1[1] + dz * e_minor1[2]
    w = dx * e_minor2[0] + dy * e_minor2[1] + dz * e_minor2[2]

    squared_radius = (
        (u / s_major) ** 2
        + (v / s_minor1) ** 2
        + (w / s_minor2) ** 2
    )
    shape = np.exp(-0.5 * squared_radius)
    return _apply_truncation(shape, squared_radius, truncate_at_sigma)


def compute_anomaly_shape(
    anomaly: Dict,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    default_truncate_at_sigma: Optional[float],
) -> np.ndarray:
    """
    Dispatch to the correct Gaussian shape function for one anomaly.

    Args:
        anomaly: Anomaly configuration dictionary
        x, y, z: Coordinate arrays (km)
        default_truncate_at_sigma: Global truncation, used if the anomaly has none

    Returns:
        Unit Gaussian shape array (peak 1 at the anomaly center)
    """
    name = anomaly.get("name", "unnamed")
    anomaly_type = anomaly.get("type")
    center = tuple(float(c) for c in anomaly.get("center"))
    if len(center) != 3:
        raise ValueError(f"Anomaly {name!r}: 'center' must be [x, y, z], got {center}")

    sigma = [float(s) for s in anomaly.get("sigma")]
    truncate_at_sigma = anomaly.get("truncate_at_sigma", default_truncate_at_sigma)

    if anomaly_type == "3d":
        if len(sigma) != 3:
            raise ValueError(f"Anomaly {name!r}: 3d type needs sigma=[sx, sy, sz], got {sigma}")
        return gaussian_3d_shape(x, y, z, center, tuple(sigma), truncate_at_sigma)

    if anomaly_type == "2d":
        if len(sigma) != 2:
            raise ValueError(f"Anomaly {name!r}: 2d type needs sigma=[sx, sy], got {sigma}")
        return gaussian_2d_layer_shape(x, y, z, center, tuple(sigma), truncate_at_sigma)

    if anomaly_type == "arbitrary":
        if len(sigma) != 3:
            raise ValueError(
                f"Anomaly {name!r}: arbitrary type needs "
                f"sigma=[major, minor1, minor2], got {sigma}"
            )
        direction = anomaly.get("direction")
        if direction is None:
            raise ValueError(f"Anomaly {name!r}: arbitrary type requires a 'direction' vector")
        direction = tuple(float(d) for d in direction)
        minor1_direction = anomaly.get("minor1_direction")
        if minor1_direction is not None:
            minor1_direction = tuple(float(d) for d in minor1_direction)
        return gaussian_arbitrary_shape(
            x, y, z, center, direction, tuple(sigma),
            minor1_direction=minor1_direction,
            truncate_at_sigma=truncate_at_sigma,
        )

    raise ValueError(
        f"Anomaly {name!r}: unknown type {anomaly_type!r}. "
        "Must be '3d', '2d', or 'arbitrary'."
    )


def apply_spike_perturbation(
    df: pd.DataFrame,
    config: Dict,
) -> pd.DataFrame:
    """
    Apply Gaussian spike perturbations to a velocity model.

    Perturbation factors from all anomalies are summed (in fractional units) and applied
    multiplicatively: v_perturbed = v_original * (1 + sum_of_fractional_perturbations).

    Args:
        df: Velocity model DataFrame with X_grid_km, Y_grid_km, Z_grid_km, vp, vs.
            NOTE: This function modifies the input DataFrame in place for memory efficiency.
        config: Spike configuration dictionary

    Returns:
        DataFrame with perturbed velocities (same object as input)
    """
    x = df["X_grid_km"].values
    y = df["Y_grid_km"].values
    z = df["Z_grid_km"].values

    apply_to_vp = config.get("apply_to_vp", True)
    apply_to_vs = config.get("apply_to_vs", True)
    default_truncate = config.get("truncate_at_sigma", None)

    anomalies = list(config.get("anomalies", []) or [])
    if not anomalies:
        raise ValueError(
            "spike.anomalies is empty. At least one Gaussian anomaly is required."
        )

    logger.info(f"Creating spike model from {len(anomalies)} Gaussian anomal"
                f"{'y' if len(anomalies) == 1 else 'ies'}:")

    # Fractional perturbation accumulators for each field.
    perturbation_vp = np.zeros(len(x), dtype=np.float64)
    perturbation_vs = np.zeros(len(x), dtype=np.float64)

    for idx, anomaly in enumerate(anomalies):
        name = anomaly.get("name", f"anomaly_{idx + 1}")
        anomaly_type = anomaly.get("type")

        shape = compute_anomaly_shape(anomaly, x, y, z, default_truncate)

        amp_p = _signed_amplitude(
            anomaly.get("amplitude_p", 0.0), anomaly.get("polarity_p", "positive"), name, "p"
        )
        amp_s = _signed_amplitude(
            anomaly.get("amplitude_s", 0.0), anomaly.get("polarity_s", "positive"), name, "s"
        )

        center = tuple(float(c) for c in anomaly.get("center"))
        logger.info(
            f"  [{idx + 1}] {name}: type={anomaly_type}, center={center}, "
            f"sigma={list(anomaly.get('sigma'))}, "
            f"amp_p={amp_p * 100:+.1f}%, amp_s={amp_s * 100:+.1f}%, "
            f"affected_points={(shape > 0).sum()}"
        )

        if apply_to_vp:
            perturbation_vp += amp_p * shape
        if apply_to_vs:
            perturbation_vs += amp_s * shape

    # Apply to vp
    if apply_to_vp:
        vp_original = df["vp"].values * 1000.0  # Convert to m/s
        vp_perturbed = vp_original * (1.0 + perturbation_vp)

        vp_min = config.get("vp_min", 500.0)
        vp_max = config.get("vp_max", 10000.0)
        vp_perturbed = np.clip(vp_perturbed, vp_min, vp_max)

        df["vp"] = vp_perturbed / 1000.0  # Store back in km/s

        with np.errstate(divide="ignore", invalid="ignore"):
            vp_pct = (vp_perturbed / vp_original - 1.0) * 100.0
        logger.info(f"  Vp: {vp_perturbed.min():.1f} - {vp_perturbed.max():.1f} m/s")
        logger.info(f"  Vp perturbation range: {np.nanmin(vp_pct):.2f}% to {np.nanmax(vp_pct):.2f}%")

    # Apply to vs
    if apply_to_vs:
        vs_original = df["vs"].values * 1000.0  # Convert to m/s
        vs_perturbed = vs_original * (1.0 + perturbation_vs)

        vs_min = config.get("vs_min", 500.0)
        vs_max = config.get("vs_max", 6000.0)
        vs_perturbed = np.clip(vs_perturbed, vs_min, vs_max)

        df["vs"] = vs_perturbed / 1000.0  # Store back in km/s

        with np.errstate(divide="ignore", invalid="ignore"):
            vs_pct = (vs_perturbed / vs_original - 1.0) * 100.0
        logger.info(f"  Vs: {vs_perturbed.min():.1f} - {vs_perturbed.max():.1f} m/s")
        logger.info(f"  Vs perturbation range: {np.nanmin(vs_pct):.2f}% to {np.nanmax(vs_pct):.2f}%")

    return df


def create_spike_model(
    input_csv: Path,
    output_dir: Path,
    config: Dict,
) -> Tuple[pd.DataFrame, Path, Path]:
    """
    Main function to create a spike (Gaussian-anomaly) velocity model.

    Args:
        input_csv: Path to input velocity model CSV (or HDF5)
        output_dir: Output directory
        config: Spike configuration

    Returns:
        (perturbed_df, vp_bin_path, vs_bin_path)
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Creating Spike Velocity Model")
    logger.info("=" * 80)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input velocity model (CSV or H5)
    logger.info(f"Reading input velocity model: {input_csv}")
    df = read_velocity_file(input_csv, show_progress=True)
    logger.info(f"  Loaded {len(df)} grid points")

    # Verify required columns
    required_cols = ["X_grid_km", "Y_grid_km", "Z_grid_km", "vp", "vs"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Apply spike perturbation
    df_perturbed = apply_spike_perturbation(df, config)

    # Get grid dimensions
    nx, ny, nz = get_grid_dimensions(df_perturbed)

    # Write perturbed velocity model (CSV and/or H5 based on config)
    output_format = config.get("output_format", "csv")
    output_csv = config.get("output_csv", output_dir / "velocity_model_spike.csv")
    output_hdf5 = config.get("output_hdf5", output_dir / "velocity_model_spike.h5")
    hdf5_options = config.get("hdf5_options", {})

    # Ensure output paths are Path objects
    if isinstance(output_csv, str):
        output_csv = Path(output_csv)
    if isinstance(output_hdf5, str):
        output_hdf5 = Path(output_hdf5)

    # Write CSV if requested
    if output_format in ["csv", "both"]:
        logger.info(f"Writing perturbed velocity model CSV: {output_csv}")
        write_velocity_file(df_perturbed, output_csv, format="csv", show_progress=True)

    # Write HDF5 if requested
    if output_format in ["hdf5", "both"]:
        logger.info(f"Writing perturbed velocity model H5: {output_hdf5}")

        # Prepare metadata
        anomalies = list(config.get("anomalies", []) or [])
        metadata = {
            "spike_config": {
                "n_anomalies": len(anomalies),
                "anomaly_types": [a.get("type") for a in anomalies],
                "truncate_at_sigma": config.get("truncate_at_sigma"),
            },
            "grid_dimensions": {
                "nx": nx,
                "ny": ny,
                "nz": nz,
            },
        }

        # Extract H5 options
        compression = hdf5_options.get("compression", "gzip")
        compression_level = hdf5_options.get("compression_level", 4)
        n_jobs = hdf5_options.get("n_jobs", 1)
        show_progress = hdf5_options.get("show_progress", True)
        coarse_grid_config = None
        if hdf5_options.get("create_coarse_dataset", False):
            coarse_grid_config = hdf5_options.get("coarse_grid")

        write_velocity_file(
            df_perturbed,
            output_hdf5,
            format="hdf5",
            compression=compression,
            compression_opts=compression_level,
            metadata=metadata,
            coarse_grid_config=coarse_grid_config,
            n_jobs=n_jobs,
            show_progress=show_progress,
        )

    # Write binary files for LATTE (always generated)
    vp_bin_path = output_dir / "vp_spike.bin"
    vs_bin_path = output_dir / "vs_spike.bin"

    logger.info(f"Writing binary velocity files for LATTE:")
    logger.info(f"  - {vp_bin_path}")
    logger.info(f"  - {vs_bin_path}")
    write_velocity_binary(df_perturbed, vp_bin_path, "vp", nx, ny, nz)
    write_velocity_binary(df_perturbed, vs_bin_path, "vs", nx, ny, nz)

    logger.info("Spike model creation completed successfully!")

    return df_perturbed, vp_bin_path, vs_bin_path


if __name__ == "__main__":
    # Example standalone usage
    import yaml

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        full_config = yaml.safe_load(f)

    # Extract relevant configs
    input_csv = Path(full_config["input"]["velocity_model_csv"])
    output_dir = Path(full_config["general"]["output_dir"]) / "spike_model"
    spike_config = full_config["spike"]

    # Run
    create_spike_model(input_csv, output_dir, spike_config)
