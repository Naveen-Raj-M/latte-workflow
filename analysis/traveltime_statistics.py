"""
Travel time statistics computation.

This module handles all statistical calculations for travel time analysis.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def compute_travel_time_statistics(df: pd.DataFrame, 
                                  config: Dict[str, Any],
                                  comparison_mode: str = 'ground_truth') -> Dict[str, Any]:
    """
    Compute comprehensive travel time statistics.
    
    Args:
        df: Travel time dataframe
        config: Configuration dictionary
        comparison_mode: 'ground_truth' or 'method_comparison'
        
    Returns:
        Dictionary of statistics
    """
    stats = {'comparison_mode': comparison_mode}
    
    # Overall statistics
    residuals = df['residual_s']
    stats['overall'] = {
        'count': len(df),
        'mean_residual': float(residuals.mean()),
        'std_residual': float(residuals.std()),
        'median_residual': float(residuals.median()),
        'min_residual': float(residuals.min()),
        'max_residual': float(residuals.max()),
        'rms_residual': float(np.sqrt(np.mean(residuals**2))),
        'mae_residual': float(np.mean(np.abs(residuals))),
    }
    
    # Phase-specific statistics
    for phase in ['P', 'S']:
        phase_data = df[df['phase'] == phase]
        if not phase_data.empty:
            phase_residuals = phase_data['residual_s']
            obs_tt = phase_data['observed_tt']
            syn_tt = phase_data['synthetic_tt']
            
            stats[f'{phase.lower()}_wave'] = {
                'count': len(phase_data),
                'mean_residual': float(phase_residuals.mean()),
                'std_residual': float(phase_residuals.std()),
                'rms_residual': float(np.sqrt(np.mean(phase_residuals**2))),
                'mae_residual': float(np.mean(np.abs(phase_residuals))),
                'mean_observed_tt': float(obs_tt.mean()),
                'mean_synthetic_tt': float(syn_tt.mean()),
                'correlation': float(obs_tt.corr(syn_tt)),
            }
            
            # Logarithmic ratio analysis if enabled
            if config['travel_time_analysis'].get('enable_ln_ratio_analysis', True):
                valid_mask = (obs_tt > 0) & (syn_tt > 0)
                if valid_mask.sum() > 0:
                    ln_ratios = np.log(syn_tt[valid_mask] / obs_tt[valid_mask])
                    stats[f'{phase.lower()}_wave']['ln_ratio'] = {
                        'mean': float(ln_ratios.mean()),
                        'std': float(ln_ratios.std()),
                        'median': float(ln_ratios.median()),
                        'equivalent_percent_change': float((np.exp(ln_ratios.mean()) - 1) * 100)
                    }
    
    # Station statistics
    station_stats = df.groupby('station')['residual_s'].agg([
        'count', 'mean', 'std'
    ]).round(4)
    stats['station_summary'] = {
        'unique_stations': int(df['station'].nunique()),
        'top_5_by_count': station_stats.nlargest(5, 'count').to_dict('index')
    }
    
    # Event statistics  
    event_stats = df.groupby('event_id')['residual_s'].agg([
        'count', 'mean', 'std'
    ]).round(4)
    stats['event_summary'] = {
        'unique_events': int(df['event_id'].nunique()),
        'top_5_by_count': event_stats.nlargest(5, 'count').to_dict('index')
    }
    
    return stats