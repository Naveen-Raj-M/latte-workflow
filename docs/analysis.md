# LATTE Analysis Module Documentation

## Overview

The analysis module provides comprehensive visualization and statistical analysis of LATTE tomographic inversion results. It processes CSV outputs from post-processing to generate publication-quality plots, detailed reports, and quality metrics for evaluating inversion performance.

---

## Features

### Analysis Components

- **Gradient Analysis**: Velocity and source gradient/search direction analysis with depth-dependent statistics
- **Travel Time Analysis**: Residual distributions, phase-specific statistics, and logarithmic ratio analysis
- **Velocity Model Comparison**: Spatial projections, vertical profiles, and statistical comparisons between models
- **Distance-Dependent Analysis**: Correlation studies and residual patterns as functions of source-receiver distance
- **Short Distance Analysis**: Detailed analysis with travel path visualization and spatial clustering
- **Source Correction Analysis**: Hypocenter relocation evaluation and correction magnitude assessment

### Output Characteristics

- Publication-quality visualizations (PNG, PDF, SVG)
- Comprehensive statistical reports
- Modular design for flexible analysis workflows
- Consistent styling across all plots

---

## Getting Started

### Quick Start

```bash
# Run complete analysis pipeline
python -m analysis.main

# View configuration options
python -m analysis.main --help
```

### Configuration

Configuration file location: `configs/analysis/default.yaml`

#### Required Path Updates

Update input file paths to match your processed data:

```yaml
travel_time_analysis:
  input_csv: "data/processed/synthetic_travel_times_latte_output.csv"
  output_dir: "results/analysis/travel_time"

velocity_analysis:
  model1_csv: "data/processed/velocity_cartesian_initial.csv"
  model2_csv: "data/processed/velocity_cartesian_latte_output.csv"
  output_dir: "results/analysis/velocity_comparison"

distance_analysis:
  travel_time_csv: "data/processed/synthetic_travel_times_latte_output.csv"
  station_csv: "data/processed/stations_cartesian.csv"
  events_csv: "data/processed/events_cartesian.csv"
  output_dir: "results/analysis/distance"
```

---

## Configuration Reference

### Global Settings

```yaml
analysis:
  log_level: INFO              # DEBUG | INFO | WARNING | ERROR
  output_format: png           # png | pdf | svg
  dpi: 300                     # Resolution for plots
  save_intermediate_results: true

  # Phase filtering
  phase_type: both             # both | p | s

  # Depth filtering
  max_depth_km: null           # Maximum depth in km (null to disable)

  plot_style:
    font_family: sans-serif
    font_size: 14
    dpi: 300
    colors:
      p_wave_dark: "#1f4e79"
      s_wave_dark: "#2e7d32"
      event_color: "#ff6b35"
```

**Key Parameters**:
- `phase_type`: Filter data by wave type. Must match preprocessing/postprocessing settings.
  - `both`: Analyze P and S waves together
  - `p`: P-waves only
  - `s`: S-waves only
- `max_depth_km`: Filter events by depth. Set to `null` to disable. Should match preprocessing/postprocessing configuration.

### Gradient Analysis

```yaml
gradient_analysis:
  enabled: false
  analysis_type: ["velocity", "source"]  # velocity | source

  # Velocity gradient analysis
  velocity_input_csv: "path/to/postprocessed/velocity_gradients.csv"
  velocity_output_dir: "results/analysis/velocity_gradient"

  # Source gradient analysis
  source_input_csv: "path/to/postprocessed/source_gradients.csv"
  source_output_dir: "results/analysis/source_gradient"

  # Analysis parameters
  histogram_bins: 50
  depth_bins: 10
  min_points_per_bin: 10

  # Output options
  generate_report: true
  create_histograms: true
  create_alignment_plots: true
  create_spatial_plots: true
  create_depth_profiles: true
```

**Purpose**: Analyze gradients and search directions from postprocessed inversion results.

**Analysis Types**:
- `velocity`: Velocity model gradients and update directions
- `source`: Source parameter gradients and update directions

**Key Visualizations**:
- Gradient magnitude histograms
- Alignment between gradients and search directions
- Depth-dependent gradient statistics
- Spatial distribution of gradient magnitudes

**Requirements**: Postprocessed gradient/search direction CSV files from velocity or source updates.

### Travel Time Analysis

```yaml
travel_time_analysis:
  enabled: true
  comparison_mode: "ground_truth"  # ground_truth | method_comparison

  # Ground truth mode (compare observed vs synthetic)
  input_csv: "path/to/synthetic_travel_times_latte_output.csv"
  output_dir: "results/analysis/travel_time"

  # Method comparison mode (compare two methods)
  method1_csv: "path/to/method1_travel_times.csv"
  method2_csv: "path/to/method2_travel_times.csv"
  method1_name: "Method 1"
  method2_name: "Method 2"

  # Analysis options
  enable_log_log_plots: true
  enable_ln_ratio_analysis: true
  outlier_threshold_sigma: 3.0
  histogram_bins: 50
  max_plot_points: 50000

  # Report generation
  generate_summary_report: true
```

**Comparison Modes**:
- `ground_truth`: Compare observed vs synthetic travel times (standard LATTE workflow)
- `method_comparison`: Compare travel time predictions from two different methods

**Key Parameters**:
- `enable_log_log_plots`: Log-scale visualizations for dynamic range assessment
- `enable_ln_ratio_analysis`: Multiplicative error analysis using logarithmic ratios
- `max_plot_points`: Subsample large datasets for performance

### Velocity Analysis

```yaml
velocity_analysis:
  enabled: true
  model1_csv: "path/to/velocity_cartesian_initial.csv"
  model2_csv: "path/to/velocity_cartesian_latte_output.csv"
  output_dir: "results/analysis/velocity_comparison"

  # Optional reference model for vertical profiles
  initial_velocity_model_csv: "path/to/initial_velocity.csv"

  # Comparison settings
  comparison_mode: "initial_vs_final"  # initial_vs_final | method_comparison | custom
  model1_name: "Initial Model"
  model2_name: "Final Model"

  # Analysis features
  compute_log_ratios: true
  compute_geometric_mean_ratios: true
  enable_spatial_projections: true
  enable_profile_analysis: true
  enable_slice_analysis: true
  enable_plane_ratio_plots: true

  # Spatial settings
  interpolation_method: "cubic"  # linear | cubic | nearest
  grid_resolution: 100
  contour_levels: 20

  # Plane ratio plots configuration
  plane_ratio_plots:
    planes:
      - plane_type: "xy"  # xy | xz | yz
        coordinates: [10.0, 20.0, 30.0]  # Slice positions in km
        label: "Horizontal Plane (xy)"
      - plane_type: "xz"
        coordinates: [400.0, 500.0]
        label: "Vertical Plane (xz)"
      - plane_type: "yz"
        coordinates: [300.0, 400.0]
        label: "Vertical Plane (yz)"
```

**Comparison Modes**:
- `initial_vs_final`: Compare starting and inverted models
- `method_comparison`: Compare results from different methods
- `custom`: Use custom titles specified in configuration

**Plane Ratio Plots**:
- `plane_type`: Orientation of slice plane
  - `xy`: Horizontal plane at specified depth coordinates
  - `xz`: Vertical plane at specified y coordinates
  - `yz`: Vertical plane at specified x coordinates
- `coordinates`: List of positions where slices are taken (in km)
- `initial_velocity_model_csv`: Optional reference model overlaid on vertical profiles

### Distance Analysis

```yaml
distance_analysis:
  enabled: true
  comparison_mode: "ground_truth"  # ground_truth | method_comparison

  # Ground truth mode
  travel_time_csv: "path/to/synthetic_travel_times_latte_output.csv"
  station_csv: "path/to/stations_cartesian.csv"
  events_csv: "path/to/events_cartesian.csv"
  output_dir: "results/analysis/distance"

  # Method comparison mode
  method1_csv: "path/to/method1_travel_times.csv"
  method2_csv: "path/to/method2_travel_times.csv"
  method1_name: "Method 1"
  method2_name: "Method 2"

  # Coordinate system
  coordinate_type: "transformed"  # transformed | original

  # Distance calculation
  distance_types: ["hypocentral"]  # hypocentral | epicentral

  # Analysis options
  enable_ratio_analysis: true
  enable_binned_analysis: true
  enable_correlation_analysis: true
  distance_bins: 20

  # Plotting options
  enable_log_plots: true
  correlation_method: "pearson"  # pearson | spearman
  fit_polynomial_degree: 1
```

**Distance Types**:
- `hypocentral`: 3D distance including depth (default)
- `epicentral`: 2D horizontal distance only
- Can specify both: `distance_types: ["hypocentral", "epicentral"]`

**Comparison Modes**:
- `ground_truth`: Analyze residuals vs distance
- `method_comparison`: Compare two methods' distance-dependent performance

### Short Distance Analysis

```yaml
short_distance_analysis:
  enabled: true
  comparison_mode: "ground_truth"  # ground_truth | method_comparison

  # Ground truth mode
  travel_time_csv: "path/to/synthetic_travel_times_latte_output.csv"
  station_csv: "path/to/stations_cartesian.csv"
  events_csv: "path/to/events_cartesian.csv"
  output_dir: "results/analysis/short_distance"

  # Method comparison mode
  method1_csv: "path/to/method1_travel_times.csv"
  method2_csv: "path/to/method2_travel_times.csv"
  method1_name: "Method 1"
  method2_name: "Method 2"

  # Distance settings
  distance_cutoff_km: 20.0
  coordinate_type: "transformed"
  distance_types: ["hypocentral"]  # hypocentral | epicentral
  phase_filter: null  # null | "P" | "S"

  # Display options
  show_events: true
  show_stations: true
  show_travel_paths: true

  # Analysis features
  enable_detailed_stats: true
  enable_residual_histograms: true
  enable_zoom_clusters: true

  # Zoom clusters for detailed regional analysis
  zoom_clusters:
    - name: "Central_Region"
      x_min: 350.0
      x_max: 450.0
      y_min: 300.0
      y_max: 400.0
      description: "Dense cluster in central region"
```

**Distance Types**:
- `hypocentral`: 3D distance including depth
- `epicentral`: 2D horizontal distance only

**Comparison Modes**:
- `ground_truth`: Analyze short-distance residual patterns
- `method_comparison`: Compare two methods at short distances

### Source Correction Analysis

```yaml
source_analysis:
  enabled: true
  comparison_mode: "ground_truth"  # ground_truth | method_comparison

  # Ground truth mode
  input_csv: "path/to/source_updated_travel_times.csv"
  output_dir: "results/analysis/source_corrections"

  # Method comparison mode
  method1_csv: "path/to/method1_source.csv"
  method2_csv: "path/to/method2_source.csv"
  method1_name: "Method 1"
  method2_name: "Method 2"

  # Analysis options
  use_log_scale: false
  enable_depth_analysis: true

  # Origin time metadata (preferred method)
  origin_time_metadata_dir: "path/to/preprocessing/metadata"

  # Deprecated: Manual origin time zeroing
  zero_initial_origin_time: true  # Kept for backwards compatibility

  # Depth binning
  depth_bins: 10
  min_events_per_bin: 5

  # Plotting options
  histogram_bins: 50
  max_arrows_map: 5000

  # Output options
  generate_summary_report: true
  save_corrected_data: true
```

**Origin Time Metadata**:
- `origin_time_metadata_dir`: Automatically loads origin times from preprocessing metadata (recommended)
- `zero_initial_origin_time`: Deprecated but retained for backwards compatibility. Use `origin_time_metadata_dir` instead.

**Comparison Modes**:
- `ground_truth`: Evaluate source corrections from inversion
- `method_comparison`: Compare source corrections from two different methods

---

## Usage Examples

### Standard Workflow

```bash
# Run all enabled analyses
python -m analysis.main
```

### Single Module Execution

```bash
# Run only gradient analysis
python -m analysis.main single_function=gradient_analysis

# Run only travel time analysis
python -m analysis.main single_function=travel_time_analysis

# Run only velocity analysis
python -m analysis.main single_function=velocity_analysis

# Run only distance analysis
python -m analysis.main single_function=distance_analysis
```

### Command-Line Overrides

```bash
# Change output format and quality
python -m analysis.main analysis.output_format=pdf analysis.dpi=600

# Enable debug logging
python -m analysis.main analysis.log_level=DEBUG

# Disable specific analysis features
python -m analysis.main \
  travel_time_analysis.enable_ln_ratio_analysis=false \
  velocity_analysis.enable_spatial_projections=false

# Custom velocity comparison
python -m analysis.main \
  velocity_analysis.comparison_mode=custom \
  velocity_analysis.custom_title1="Method A" \
  velocity_analysis.custom_title2="Method B"

# Method comparison for travel times
python -m analysis.main \
  travel_time_analysis.comparison_mode=method_comparison \
  travel_time_analysis.method1_csv=data/method1.csv \
  travel_time_analysis.method2_csv=data/method2.csv

# Filter by phase type and depth
python -m analysis.main \
  analysis.phase_type=p \
  analysis.max_depth_km=30.0
```

### Custom Configuration Files

```bash
# Use alternative configuration
python -m analysis.main --config-name=high_quality

# Specify custom config path
python -m analysis.main --config-path=/custom/config/path
```

---

## Output Files

### Gradient Analysis

**Directory**: `{output_dir}/gradient_analysis/`

**Plots**:
- `gradient_magnitude_histogram_{type}.png`: Distribution of gradient magnitudes
- `search_direction_histogram_{type}.png`: Distribution of search direction magnitudes
- `alignment_scatter_{type}.png`: Gradient vs search direction alignment
- `spatial_gradient_map_{type}.png`: Geographic distribution of gradient magnitudes
- `depth_profile_{type}.png`: Gradient statistics by depth

**Reports**:
- `gradient_analysis_report_{type}.txt`: Statistical summary of gradients and search directions

**Key Metrics**:
- Gradient magnitude statistics (mean, median, RMS)
- Search direction statistics
- Alignment angles between gradients and search directions
- Depth-dependent gradient characteristics

### Travel Time Analysis

**Directory**: `{output_dir}/travel_time/`

**Plots**:
- `synthetic_vs_observed.png`: Scatter plots with 1:1 reference lines
- `residual_histograms.png`: Distribution analysis with statistical overlays
- `synthetic_vs_observed_loglog.png`: Log-scale comparisons
- `ln_ratio_histograms.png`: Logarithmic ratio distributions
- `ln_ratio_analysis.png`: Ratio vs time analysis

**Reports**:
- `travel_time_analysis_report.txt`: Comprehensive statistical summary

**Key Metrics**:
- Phase-specific RMS residuals
- Mean and median residuals
- Travel time correlations
- Logarithmic ratio statistics

### Velocity Analysis

**Directory**: `{output_dir}/velocity_comparison/`

**Plots**:
- `basic_velocity_comparison.png`: Scatter plots and difference histograms
- `spatial_velocity_projections.png`: RMS spatial distribution (XY, XZ, YZ planes)
- `spatial_geometric_ratio_projections.png`: Ratio-based spatial analysis
- `velocity_vertical_profiles.png`: 1D profiles at extreme difference locations
- `depth_slice_heatmaps.png`: Horizontal slice comparisons
- `vp_ratio_analysis.png`: Vp ratio heatmaps at significant depths
- `vs_ratio_analysis.png`: Vs ratio heatmaps at significant depths
- `domain_overview.png`: Overall domain visualization

**Reports**:
- `velocity_comparison_report.txt`: Statistical comparison summary

**Key Metrics**:
- Velocity correlations
- Mean and RMS differences
- Percent changes
- Vp/Vs ratio evolution

### Distance Analysis

**Directory**: `{output_dir}/distance/`

**Plots**:
- `residuals_vs_distance.png`: Residual-distance relationships with trend lines
- `correlation_analysis.png`: Phase-specific correlations
- `binned_distance_analysis.png`: Binned statistics with error bars
- `ratios_vs_distance.png`: Travel time ratios vs distance

**Reports**:
- `distance_analysis_report.txt`: Correlation and trend analysis

**Key Metrics**:
- Distance-residual correlations
- Binned mean residuals and RMS
- Polynomial fit coefficients

### Short Distance Analysis

**Directory**: `{output_dir}/short_distance/`

**Plots**:
- `short_distance_travel_paths_{coord}.png`: Spatial path visualization with residual color coding
- `short_distance_histograms.png`: Residual distributions and Q-Q plots
- `short_distance_correlation_{coord}.png`: Distance-residual relationships
- `zoom_cluster_{name}_{coord}.png`: Detailed regional analysis

**Reports**:
- `short_distance_analysis_report.txt`: Spatial analysis summary

**Key Metrics**:
- Short-range residual statistics
- Spatial clustering metrics
- Regional statistics

### Source Correction Analysis

**Directory**: `{output_dir}/source_corrections/`

**Plots**:
- `hist_{parameter}.png`: Correction magnitude histograms for each parameter
- `scatter_relationships.png`: Relationships between different corrections
- `geographic_corrections_{plane}.png`: Geographic maps showing correction vectors
- `depth_analysis.png`: Original vs corrected depths
- `correction_summary.png`: Box plots of all corrections

**Reports**:
- `source_correction_report.txt`: Correction statistics summary

**Key Metrics**:
- Mean correction magnitudes
- Horizontal and vertical shifts
- Temporal corrections
- Depth-dependent statistics

---

## Analysis Details

### Gradient Analysis

**Purpose**: Analyze gradients and search directions from velocity or source inversion updates.

**Workflow**:
1. Load postprocessed gradient and search direction CSV files
2. Compute magnitude statistics for gradients and search directions
3. Calculate alignment between gradients and search directions
4. Analyze depth-dependent gradient patterns
5. Generate spatial distribution maps
6. Create comprehensive statistical reports

**Key Visualizations**:
- Gradient magnitude histograms showing distribution of update strengths
- Alignment scatter plots revealing optimization behavior
- Spatial maps showing geographic patterns in gradients
- Depth profiles identifying depth-dependent gradient characteristics

**Requirements**: Postprocessed CSV files containing gradient and search direction data from velocity or source parameter updates.

### Travel Time Analysis

**Purpose**: Evaluate fit between observed and synthetic travel times.

**Workflow**:
1. Load travel time CSV with residuals
2. Validate and clean data (remove outliers, NaN values)
3. Compute phase-specific statistics
4. Generate comparison plots (linear and log scales)
5. Perform logarithmic ratio analysis for multiplicative errors
6. Create detailed statistical report

**Key Visualizations**:
- Observed vs synthetic scatter plots with correlation metrics
- Residual histograms showing distribution characteristics
- Log-log plots for dynamic range assessment
- ln(ratio) analysis for systematic bias detection

### Velocity Analysis

**Purpose**: Compare velocity models to assess inversion results.

**Workflow**:
1. Load two velocity model CSVs
2. Validate coordinate consistency
3. Compute differences and statistics
4. Find extreme difference locations
5. Generate spatial projections (XY, XZ, YZ)
6. Create vertical profiles at extreme locations
7. Analyze horizontal slices at significant depths
8. Compute geometric mean ratios

**Key Visualizations**:
- Scatter plots with 1:1 lines showing model agreement
- Spatial RMS projections revealing 3D difference patterns
- Vertical profiles comparing models at extreme locations
- Depth slice heatmaps showing lateral variations
- Ratio analysis identifying fractional changes

### Distance Analysis

**Purpose**: Investigate travel time residual dependencies on source-receiver distance.

**Workflow**:
1. Load travel time, station, and event CSVs
2. Merge datasets and calculate 3D distances
3. Compute distance-residual correlations
4. Perform binned analysis for distance ranges
5. Analyze travel time ratios vs distance
6. Generate correlation plots with trend lines

**Key Visualizations**:
- Residual vs distance scatter plots with polynomial fits
- Binned mean residuals with error bars
- Phase-specific correlation analysis
- Ratio analysis in linear and log scales

### Short Distance Analysis

**Purpose**: Detailed analysis of short source-receiver paths with spatial context.

**Workflow**:
1. Filter data by distance cutoff
2. Calculate spatial statistics
3. Visualize travel paths with residual color coding
4. Identify spatial clusters
5. Perform zoom analysis on defined regions
6. Generate residual intensity maps

**Key Visualizations**:
- Travel path maps showing event-station pairs
- Residual intensity maps by station location
- Zoom cluster analysis for detailed regions
- Distance-residual correlations for short ranges

### Source Correction Analysis

**Purpose**: Evaluate hypocenter relocation and origin time corrections.

**Workflow**:
1. Load CSV with before/after correction columns
2. Compute correction deltas (spatial and temporal)
3. Calculate correction magnitudes
4. Analyze depth-dependent patterns
5. Generate geographic correction maps
6. Create statistical summaries

**Key Visualizations**:
- Correction magnitude histograms for all parameters
- Geographic maps with correction vectors
- Original vs corrected depth scatter plots
- Depth-binned correction statistics
- Temporal evolution of corrections

---

## Performance Optimization

### Memory Management

```yaml
analysis:
  plot_style:
    max_plot_points: 50000  # Automatic subsampling for large datasets

performance:
  max_memory_gb: 8.0
  subsample_large_datasets: true
```

### Processing Speed

```yaml
velocity_analysis:
  grid_resolution: 50          # Lower resolution for faster processing
  interpolation_method: linear  # Faster than cubic

short_distance_analysis:
  enable_zoom_clusters: false  # Skip detailed cluster analysis
```

---

## Quality Control

### Input Validation

```yaml
quality_control:
  validate_input_data: true
  check_coordinate_consistency: true
  error_on_empty_results: true
  
  # Data validation thresholds
  max_velocity_km_s: 15.0
  min_velocity_km_s: 0.5
  max_travel_time_s: 1000.0
```

**Automatic Checks**:
- File existence and format validation
- Coordinate system consistency between files
- Data range reasonableness
- Sufficient data points for analysis

---

## Advanced Usage

### Custom Analysis Workflows

Create specialized configurations for different analysis stages:

```yaml
# configs/analysis/initial_assessment.yaml
defaults:
  - default

# First pass: quick validation
travel_time_analysis:
  enabled: true
  histogram_bins: 30
  max_plot_points: 10000

velocity_analysis:
  enabled: false
```

```yaml
# configs/analysis/detailed_analysis.yaml
defaults:
  - default

# Second pass: comprehensive analysis
single_function: velocity_analysis
velocity_analysis:
  enable_spatial_projections: true
  grid_resolution: 200
  enable_profile_analysis: true
  enable_slice_analysis: true
```

### Parameter Sweeps

```bash
# Test different distance cutoffs
for cutoff in 10 20 30; do
  python -m analysis.main \
    single_function=short_distance_analysis \
    short_distance_analysis.distance_cutoff_km=$cutoff \
    short_distance_analysis.output_dir=results/short_${cutoff}km
done
```

### Batch Processing

```bash
# Process multiple model comparisons
for model in run1 run2 run3; do
  python -m analysis.main \
    single_function=velocity_analysis \
    velocity_analysis.model2_csv=data/${model}_velocity.csv \
    velocity_analysis.output_dir=results/${model}_comparison
done
```

---

## Integration Workflow

### Complete Pipeline

```
Preprocessing → LATTE Inversion → Post-Processing → Analysis
```

**Example Directory Structure**:

```
project/
├── data/
│   ├── raw/
│   │   ├── stacoords.txt
│   │   └── phasePickings.txt
│   └── processed/
│       ├── velocity_cartesian_initial.csv
│       ├── velocity_cartesian_latte_output.csv
│       ├── synthetic_travel_times_latte_output.csv
│       ├── stations_cartesian.csv
│       └── events_cartesian.csv
├── results/
│   └── analysis/
│       ├── travel_time/
│       ├── velocity_comparison/
│       ├── distance/
│       └── short_distance/
└── configs/
    └── analysis/
        └── default.yaml
```

---

## Troubleshooting

### Configuration Issues
- Verify YAML syntax and indentation
- Ensure all input file paths exist
- Check that output directories are writable

### Data Issues
- Confirm CSV files have required columns
- Verify coordinate consistency across files
- Check for sufficient data points in each analysis

### Performance Issues
- Reduce `max_plot_points` for large datasets
- Lower `grid_resolution` for faster interpolation
- Disable optional analysis features

### Plot Quality Issues
- Increase `dpi` for higher resolution
- Use PDF format for vector graphics
- Adjust font sizes for better readability

---

## Best Practices

1. **Start with quick assessment**: Run with default settings first to identify issues
2. **Use debug logging**: Enable `DEBUG` level when troubleshooting
3. **Validate inputs**: Check data quality before running full analysis
4. **Organize outputs**: Use descriptive output directory names
5. **Document parameters**: Record configuration used for each analysis run
6. **Review logs**: Check log files for warnings or errors
7. **Compare incrementally**: Analyze changes between successive inversion iterations

---

## API Reference

### Main Analysis Functions

#### `analyze_gradients(config: Dict[str, Any])`
Gradient and search direction analysis for velocity or source parameters.

**Outputs**: Gradient histograms, alignment plots, spatial maps, depth profiles, and statistical reports

#### `analyze_travel_time_residuals(config: Dict[str, Any])`
Comprehensive travel time residual analysis.

**Outputs**: Plots and statistical report

#### `analyze_velocity_models(config: Dict[str, Any])`
Velocity model comparison with spatial analysis.

**Outputs**: Multiple plot types and comparison report

#### `analyze_distance_dependent_residuals(config: Dict[str, Any])`
Distance-dependent residual analysis.

**Outputs**: Correlation plots and statistical report

#### `analyze_short_distance_residuals(config: Dict[str, Any])`
Focused short-distance analysis with spatial visualization.

**Outputs**: Travel path maps and cluster analysis plots

#### `analyze_source_corrections(config: Dict[str, Any])`
Source parameter correction analysis.

**Outputs**: Correction magnitude plots and statistical report

---

**Version:** 1.0.0  
**Last Updated:** 2025