#!/usr/bin/env python3
"""
Standalone script for visualizing ray paths from LATTE gradient files.

This script can be run independently or as part of the main postprocessing workflow.

Usage:
    python visualize_raypaths.py

Configuration is loaded from ../configs/postprocess/default.yaml
You can override the configuration file:
    python visualize_raypaths.py --config-name=custom_config
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from utils.logging_utils import setup_logging
from postprocess_raypath import create_ray_path_visualizations


@hydra.main(version_base=None, config_path="../configs/postprocess", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main entry point for ray path visualization."""
    setup_logging(level=cfg.general.log_level)
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("LATTE RAY PATH VISUALIZATION")
    logger.info("="*80)

    if not hasattr(cfg, 'ray_path_visualization'):
        logger.error("ray_path_visualization configuration not found in config file")
        return

    if not cfg.ray_path_visualization.enabled:
        logger.warning("Ray path visualization is DISABLED in configuration")
        logger.info("Set ray_path_visualization.enabled = true in your config file")
        return

    if cfg.general.log_level.upper() == "DEBUG":
        logger.debug("Ray path visualization configuration:")
        logger.debug(f"\n{OmegaConf.to_yaml(cfg.ray_path_visualization)}")

    try:
        create_ray_path_visualizations(cfg.ray_path_visualization)
        logger.info("Ray path visualization completed successfully!")
    except Exception as e:
        logger.error(f"Ray path visualization failed: {e}")
        raise


if __name__ == "__main__":
    main()
