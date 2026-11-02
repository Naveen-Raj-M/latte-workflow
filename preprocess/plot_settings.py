"""Shared plot settings for the preprocess visualizations.

The top-level ``plots`` section of the configuration holds the unified plotting
controls. Each visualization section (``raw_data_visualization`` and
``final_data_visualization``) inherits it and may override any key through its
own optional ``plots`` node.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf


def resolve_plot_settings(config: DictConfig, section: DictConfig) -> DictConfig:
    """Merge the global ``plots`` section with a visualization section's overrides.

    Args:
        config: Full configuration.
        section: Visualization section (may contain an optional ``plots`` node).

    Returns:
        Plot settings with the section's overrides applied on top of ``plots``.
    """
    base = config.get('plots') or {}
    override = section.get('plots') or {}
    return OmegaConf.merge(base, override)


def save_figure(fig, output_dir: Path, name: str, settings: DictConfig) -> Path:
    """Save and close a figure using the figure dpi and format from the settings.

    Args:
        fig: Matplotlib figure.
        output_dir: Output directory.
        name: File name without extension.
        settings: Plot settings from resolve_plot_settings().

    Returns:
        Path of the saved file.
    """
    output_file = Path(output_dir) / f"{name}.{settings.figure.format}"
    fig.savefig(output_file, dpi=settings.figure.dpi, bbox_inches='tight')
    plt.close(fig)
    return output_file
