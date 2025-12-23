"""
LATTE Checkerboard Resolution Test Module

This module provides tools for conducting checkerboard resolution tests
for LATTE tomographic inversion.

Components:
- create_checkerboard_model: Create synthetic checkerboard velocity models
- add_traveltime_noise: Add noise to synthetic traveltimes
- convert_inverted_model: Convert inverted binary models to CSV
- visualize_slices: Visualize velocity model slices
- main: Main workflow orchestrator

Usage:
    # Run complete workflow
    python -m checkerboard_test.main --config config.yaml

    # Or import individual components
    from checkerboard_test import create_checkerboard_model
"""

__version__ = "1.0.0"
__author__ = "LATTE Workflow"

from .create_checkerboard_model import (
    create_checkerboard_model,
    create_sinusoidal_perturbation,
    create_block_perturbation,
    apply_checkerboard_perturbation,
)

from .add_traveltime_noise import (
    process_traveltime_directory,
    add_noise_to_traveltime,
    read_traveltime_binary,
    write_traveltime_binary,
)

from .convert_inverted_model import (
    convert_inverted_model,
    load_velocity_binary,
    sample_at_coarse_points,
    expand_to_fine_grid,
)

from .visualize_slices import (
    visualize_checkerboard_test,
    visualize_model,
    plot_xy_slice,
    plot_xz_slice,
    plot_yz_slice,
)

__all__ = [
    # Main functions
    "create_checkerboard_model",
    "process_traveltime_directory",
    "convert_inverted_model",
    "visualize_checkerboard_test",
    # Checkerboard utilities
    "create_sinusoidal_perturbation",
    "create_block_perturbation",
    "apply_checkerboard_perturbation",
    # Traveltime utilities
    "add_noise_to_traveltime",
    "read_traveltime_binary",
    "write_traveltime_binary",
    # Model conversion utilities
    "load_velocity_binary",
    "sample_at_coarse_points",
    "expand_to_fine_grid",
    # Visualization utilities
    "visualize_model",
    "plot_xy_slice",
    "plot_xz_slice",
    "plot_yz_slice",
]
