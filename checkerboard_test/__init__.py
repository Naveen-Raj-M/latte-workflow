"""
LATTE Checkerboard Resolution Test Module

This module provides tools for conducting checkerboard resolution tests
for LATTE tomographic inversion.

Components:
- create_checkerboard_model: Create synthetic checkerboard velocity models
- add_traveltime_noise: Add noise to synthetic traveltimes
- convert_inverted_model: Convert inverted binary models to CSV
- visualize_slices: Visualize velocity model slices
- visualize_model: Visualize input/checkerboard/inverted models
- visualize_difference: Visualize difference plots
- visualize_pattern: Visualize pattern plots with threshold support
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

from .visualize_slices import visualize_checkerboard_test

from .visualize_model import (
    visualize_velocity_model,
    plot_model_xy_slice,
    plot_model_xz_slice,
    plot_model_yz_slice,
)

from .visualize_difference import (
    visualize_difference,
    plot_difference_xy_slice,
    plot_difference_xz_slice,
    plot_difference_yz_slice,
)

from .visualize_pattern import (
    visualize_pattern,
    plot_pattern_xy_slice,
    plot_pattern_xz_slice,
    plot_pattern_yz_slice,
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
    # Model visualization
    "visualize_velocity_model",
    "plot_model_xy_slice",
    "plot_model_xz_slice",
    "plot_model_yz_slice",
    # Difference visualization
    "visualize_difference",
    "plot_difference_xy_slice",
    "plot_difference_xz_slice",
    "plot_difference_yz_slice",
    # Pattern visualization
    "visualize_pattern",
    "plot_pattern_xy_slice",
    "plot_pattern_xz_slice",
    "plot_pattern_yz_slice",
]
