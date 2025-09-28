#!/usr/bin/env python3
"""
Main script for LATTE post-processing.
Converts LATTE tomography binary outputs to CSV format for analysis.

Usage:
    python main.py
    
The configuration is loaded from ../configs/postprocess/default.yaml
"""

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from postprocess_tools import postprocess_latte_outputs

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

@hydra.main(version_base=None, config_path="../configs", config_name="postprocess/default")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for post-processing with Hydra.
    
    Args:
        cfg: Hydra configuration
    """
    # Setup logging
    setup_logging(level=cfg.general.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting LATTE post-processing")
    
    # Log configuration (only in debug mode)
    if cfg.general.log_level.upper() == "DEBUG":
        logger.debug("Full configuration:")
        logger.debug(f"\n{OmegaConf.to_yaml(cfg)}")
    
    try:
        # Run post-processing
        postprocess_latte_outputs(cfg)
        logger.info("Post-processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Post-processing failed: {e}")
        raise

if __name__ == "__main__":
    main()