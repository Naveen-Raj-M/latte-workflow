"""
LATTE Spike Resolution Test Module

This module provides tools for conducting spike resolution tests for LATTE
tomographic inversion. Instead of a periodic checkerboard pattern, it places one
or more localized Gaussian velocity anomalies of arbitrary shape, orientation,
amplitude and polarity, then checks how well the inversion recovers them.

Supported anomaly kinds (see configs/spike_test/default.yaml):
- "3d"        : axis-aligned 3D Gaussian ellipsoid
- "2d"        : in-plane Gaussian blob confined to a single depth layer
- "arbitrary" : Gaussian ellipsoid whose major axis follows a direction vector

Components:
- create_spike_model: Create synthetic spike velocity models
- visualize_slices: Visualize input/spike/inverted models and patterns
- main: Main workflow orchestrator

The traveltime-noise, inverted-model-conversion, grid utilities, recovery analysis
and per-slice plotting primitives are perturbation-agnostic and are reused from the
checkerboard_test module.

Usage:
    # Run complete workflow
    python -m spike_test.main --config-name=default

    # Or import individual components
    from spike_test import create_spike_model
"""

__version__ = "4.0.0"
__author__ = "Naveen Raj Manoharan"

from .create_spike_model import (
    create_spike_model,
    apply_spike_perturbation,
    compute_anomaly_shape,
    gaussian_3d_shape,
    gaussian_2d_layer_shape,
    gaussian_arbitrary_shape,
)

from .visualize_slices import visualize_spike_test

__all__ = [
    # Main functions
    "create_spike_model",
    "visualize_spike_test",
    # Spike utilities
    "apply_spike_perturbation",
    "compute_anomaly_shape",
    "gaussian_3d_shape",
    "gaussian_2d_layer_shape",
    "gaussian_arbitrary_shape",
]
