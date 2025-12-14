# Analysis Module Configuration

## Overview

The analysis module provides comprehensive statistical and visual analysis tools for LATTE tomography results, including velocity models, travel times, source corrections, and distance-based metrics.

## Configuration Files

- `default.yaml` - Default configuration with all available options
- `config.yaml` - User configuration (overrides defaults)

## Key Configuration Sections

### Global Settings

- **log_level**: `DEBUG | INFO | WARNING | ERROR`
- **output_format**: `png | pdf | svg`
- **phase_type**: `both | p | s`
- **max_depth_km**: Depth filtering threshold (null to disable)

### Analysis Modules

#### Gradient Analysis
Analyzes gradients and search directions for velocity or source parameters.

- **analysis_type**: `[velocity, source]`
- **enabled**: Enable/disable module

#### Travel Time Analysis
Compares synthetic vs observed travel times or between methods.

- **comparison_mode**: `ground_truth | method_comparison`
- **enable_log_log_plots**: Logarithmic scale analysis
- **enable_ln_ratio_analysis**: Natural log ratio analysis

#### Velocity Analysis
Compares velocity models with spatial projections and depth profiles.

- **comparison_mode**: `initial_vs_final | method_comparison | custom`
- **interpolation_method**: `linear | cubic | nearest`
- **enable_spatial_projections**: 2D/3D spatial visualizations
- **enable_plane_ratio_plots**: Velocity ratio plots on specified planes
  - **plane_type**: `xy | xz | yz`

#### Distance Analysis
Analyzes travel time residuals as function of source-receiver distance.

- **comparison_mode**: `ground_truth | method_comparison`
- **coordinate_type**: `transformed | original`
- **distance_types**: `[hypocentral, epicentral]`
- **correlation_method**: `pearson | spearman`

#### Short Distance Analysis
Focused analysis on short-distance observations with spatial filtering.

- **comparison_mode**: `ground_truth | method_comparison`
- **distance_cutoff_km**: Maximum distance threshold
- **coordinate_type**: `transformed | original`
- **distance_types**: `[hypocentral, epicentral]`
- **phase_filter**: `null | P | S`

#### Source Correction Analysis
Analyzes event source parameter corrections (location and origin time).

- **comparison_mode**: `ground_truth | method_comparison`
- **enable_depth_analysis**: Depth-binned statistics
- **origin_time_metadata_dir**: Path to origin time metadata

## Usage

```bash
python -m analysis.main --config-name=your_config
```

## Output Structure

```
results/analysis/
├── plots/          # Visualizations (PNG/PDF/SVG)
├── statistics/     # Statistical summaries (CSV)
└── reports/        # Detailed analysis reports (TXT/MD)
```

## Notes

- Ensure `phase_type` matches preprocessing/postprocessing settings
- Set `max_depth_km` consistently across all modules
- For `method_comparison` mode, provide both method CSV files
- Gradient analysis requires postprocessed gradient/search direction data
