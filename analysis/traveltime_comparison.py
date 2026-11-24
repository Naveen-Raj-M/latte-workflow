"""
Travel time method comparison utilities.

This module handles merging and preparing data for method-to-method comparison.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def prepare_comparison_data(df1: pd.DataFrame, df2: pd.DataFrame, 
                           config: Dict[str, Any]) -> pd.DataFrame:
    """
    Prepare data for method-to-method comparison.
    
    Merges two datasets and computes differences between methods.
    
    Args:
        df1: Method 1 dataframe
        df2: Method 2 dataframe
        config: Configuration dictionary
        
    Returns:
        Merged dataframe with comparison columns
    """
    logger.info(f"Method 1 data points: {len(df1)}")
    logger.info(f"Method 2 data points: {len(df2)}")
    
    # Identify merge keys
    merge_keys = ['event_id', 'station', 'phase']
    
    # Verify merge keys exist in both dataframes
    for key in merge_keys:
        if key not in df1.columns:
            raise ValueError(f"Missing merge key '{key}' in Method 1 data")
        if key not in df2.columns:
            raise ValueError(f"Missing merge key '{key}' in Method 2 data")
    
    # Merge datasets
    logger.info(f"Merging datasets on keys: {merge_keys}")
    df_merged = pd.merge(
        df1, df2, 
        on=merge_keys, 
        suffixes=('_method1', '_method2'),
        how='inner'
    )
    
    logger.info(f"Merged data points: {len(df_merged)}")
    
    if len(df_merged) == 0:
        raise ValueError("No matching records found between Method 1 and Method 2 data")
    
    # For method comparison, we treat Method 1 as "observed" and Method 2 as "synthetic"
    # This allows reuse of existing plotting and analysis code
    df_merged['observed_tt'] = df_merged['synthetic_tt_method1']
    df_merged['synthetic_tt'] = df_merged['synthetic_tt_method2']
    
    # Compute residual (Method2 - Method1)
    df_merged['residual_s'] = df_merged['synthetic_tt'] - df_merged['observed_tt']
    
    # Also keep observed travel time if it exists (for reference)
    if 'observed_tt_method1' in df_merged.columns:
        df_merged['ground_truth_tt'] = df_merged['observed_tt_method1']
    
    # Weights: use Method 1 (the reference) so weighted statistics work downstream
    if 'weight_method1' in df_merged.columns:
        df_merged['weight'] = df_merged['weight_method1']
    elif 'weight' not in df_merged.columns:
        raise ValueError("Missing 'weight' column in Method 1 data")

    # Clean up intermediate columns to avoid confusion
    drop_cols = [col for col in df_merged.columns 
                 if col.endswith('_method1') or col.endswith('_method2')]
    df_merged = df_merged.drop(columns=drop_cols)
    
    # Log statistics
    logger.info(f"Residual statistics (Method2 - Method1):")
    logger.info(f"  Mean: {df_merged['residual_s'].mean():.4f} s")
    logger.info(f"  Std: {df_merged['residual_s'].std():.4f} s")
    logger.info(f"  RMS: {np.sqrt(np.mean(df_merged['residual_s']**2)):.4f} s")
    
    return df_merged