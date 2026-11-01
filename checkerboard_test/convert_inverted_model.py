"""
Convert inverted velocity model from LATTE binary format to CSV/HDF5.

This module runs the same conversion as the postprocess module
(postprocess.postprocess_velocity.create_velocity_csv_from_binaries), driven by a
model_conversion configuration that uses the postprocess ``velocity_processing`` keys:
the grid of the binaries comes from the metadata of the input velocity model (checked
against the binary sizes), the model is either sampled at the input points
(``sample_at_coarse``) or expanded to the fine grid (``expand_to_fine``), and the output is
written as CSV and/or HDF5 with the grid in its metadata.
"""

import logging
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from postprocess.postprocess_velocity import create_velocity_csv_from_binaries

logger = logging.getLogger(__name__)


def convert_inverted_model(
    input_model_dir: Path,
    output_dir: Path,
    input_csv: Path,
    config: Union[Dict, DictConfig],
) -> pd.DataFrame:
    """
    Convert the inverted velocity binaries to a velocity model file.

    Args:
        input_model_dir: Directory containing updated_vp.bin, updated_vs.bin (used when
            ``vp_binary``/``vs_binary`` are not set in the configuration)
        output_dir: Output directory (used when the output paths are not set in the configuration)
        input_csv: Input velocity model (CSV or HDF5) written by preprocess; it provides
            the grid metadata and the points to sample
        config: model_conversion configuration (postprocess velocity_processing keys)

    Returns:
        Converted DataFrame
    """
    logger.info("=" * 80)
    logger.info("STEP 3: Converting Inverted Model to CSV")
    logger.info("=" * 80)

    container = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else dict(config)
    cfg = OmegaConf.create(container)

    cfg.reference_velocity_file = str(input_csv)
    # Gradient and search direction fields are not part of the inverted model conversion
    cfg.process_gradient = False
    cfg.process_search = False
    cfg.setdefault("vp_binary", str(Path(input_model_dir) / "updated_vp.bin"))
    cfg.setdefault("vs_binary", str(Path(input_model_dir) / "updated_vs.bin"))
    cfg.setdefault("velocity_output_csv", str(Path(output_dir) / "velocity_model_final.csv"))
    cfg.setdefault("velocity_output_hdf5", str(Path(output_dir) / "velocity_model_final.h5"))

    df = create_velocity_csv_from_binaries(cfg)

    logger.info("\nInverted model statistics:")
    for column, label in (("vp", "Vp"), ("vs", "Vs")):
        if column in df.columns:
            logger.info(f"  {label}: {df[column].min():.3f} - {df[column].max():.3f} km/s")
    logger.info(f"  Total grid points: {len(df)}")
    logger.info("Model conversion completed successfully!")

    return df
