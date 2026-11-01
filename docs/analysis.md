# Analysis Pipeline

The analysis pipeline provides post-inversion diagnostics for LATTE tomography results. It reads post-processed CSVs, HDF5 files, and ray path caches to produce publication-quality plots and statistical reports. All ten analysis modules are independent and disabled by default; enable only the modules relevant to a given evaluation.

> **Configuration reference:** See [`configs/analysis/README.md`](../configs/analysis/README.md) for a complete description of every configuration parameter and all required file paths.

---

## Table of Contents

1. [Modules](#1-modules)
2. [Outputs](#2-outputs)
3. [Usage and Configuration](#3-usage-and-configuration)
4. [Logging](#4-logging)
5. [Troubleshooting](#5-troubleshooting)
6. [Best Practices](#6-best-practices)

---

## 1. Modules

The pipeline is driven by `analysis/main.py`. Each module is called in sequence and is guarded by an `enabled` flag. Failures in one module are caught and logged without stopping the remaining modules.

---

### 1.1 Travel Time Residual Analysis (`traveltime_analysis.py`)

Analyzes residuals between synthetic travel time predictions and observed ground truth picks.

**What it does:**

- **`ground_truth` mode:** loads a single merged travel time CSV and computes the distribution of residuals (`observed_tt − synthetic_tt`).
- **`method_comparison` mode:** loads two separate residual CSVs and computes their difference, identifying picks present in only one file (mismatch log) and flagging outliers that exceed physical thresholds.
- Applies outlier rejection using a configurable sigma threshold (`outlier_threshold_sigma`, default 3.0).
- Computes summary statistics: mean, standard deviation, median, and weighted RMS per phase and overall.
- Generates histograms, scatter plots, phase-separated plots, and log-ratio plots.
- Writes a plain-text summary report and optional mismatch/outlier logs.

**Key configuration:**

| Key | Purpose |
|---|---|
| `travel_time_analysis.enabled` | Enable/disable this module |
| `travel_time_analysis.comparison_mode` | `ground_truth` or `method_comparison` |
| `travel_time_analysis.input_csv` | Merged TT CSV (ground_truth mode) |
| `travel_time_analysis.method1_csv` / `method2_csv` | Two residual CSVs (comparison mode) |
| `travel_time_analysis.outlier_threshold_sigma` | Sigma cutoff for outlier rejection |
| `travel_time_analysis.min/max_residual_threshold` | Hard residual bounds (±50 s default) |

---

### 1.2 Velocity Model Analysis (`velocity_analysis.py`)

Compares two velocity models and characterizes spatial velocity variations.

**What it does:**

- Loads two velocity model files (CSV or HDF5) representing, for example, the initial and final models or two method variants.
- Computes point-wise differences (Δvp, Δvs) and log-ratios (ln(v₂/v₁)).
- Produces spatial projections onto the XY, XZ, and YZ planes by averaging or slicing the 3D model.
- Generates depth-slice heatmaps, 1D profile plots, a domain overview, and a velocity ratio analysis.
- Reports per-plane statistics in a plain-text summary report.

**Key configuration:**

| Key | Purpose |
|---|---|
| `velocity_analysis.enabled` | Enable/disable this module |
| `velocity_analysis.comparison_mode` | `initial_vs_final` / `method_comparison` / `custom` |
| `velocity_analysis.model1_csv` / `model2_csv` | Paths to velocity model files |
| `velocity_analysis.use_coarse_dataset` | Use the `velocity_coarse` HDF5 dataset for speed |
| `velocity_analysis.compute_log_ratios` | Compute ln-ratio in addition to differences |
| `velocity_analysis.enable_*_plots` | Toggle individual plot types |

---

### 1.3 Distance-Dependent Residual Analysis (`distance_analysis.py`)

Examines how travel time residuals vary with source-receiver distance to identify systematic distance-dependent bias.

**What it does:**

- Merges the travel time CSV with event and station Cartesian coordinates to compute hypocentral and/or epicentral distances.
- Bins picks by distance and computes mean residual and standard deviation per bin.
- Computes Pearson or Spearman correlation between residual and distance.
- Fits a low-degree polynomial to the residual–distance trend.
- Produces scatter plots, binned statistics plots, correlation analysis plots, and (optionally) ratio plots.
- Supports `ground_truth` (single file) and `method_comparison` (two files) modes.

**Key configuration:**

| Key | Purpose |
|---|---|
| `distance_analysis.enabled` | Enable/disable this module |
| `distance_analysis.comparison_mode` | `ground_truth` or `method_comparison` |
| `distance_analysis.travel_time_csv` | Merged TT CSV path |
| `distance_analysis.distance_types` | `["hypocentral"]` and/or `["epicentral"]` |
| `distance_analysis.distance_bins` | Number of distance bins |
| `distance_analysis.correlation_method` | `pearson` or `spearman` |
| `distance_analysis.enable_ratio_analysis` | Generate ratio plots |

---

### 1.4 Short-Distance Residual Analysis (`short_distance_analysis.py`)

Focuses analysis on near-field event–station pairs, where 3D velocity heterogeneity has the strongest influence on residuals.

**What it does:**

- Applies a distance cutoff (`distance_cutoff_km`) to retain only short-path picks.
- Plots a geographic map of the retained event–station pairs with travel path lines.
- Identifies spatial clusters of nearby events and renders a zoomed panel for each cluster.
- Computes residual histograms for the near-field subset.
- Reports statistics broken down by distance type and phase.

**Key configuration:**

| Key | Purpose |
|---|---|
| `short_distance_analysis.enabled` | Enable/disable this module |
| `short_distance_analysis.distance_cutoff_km` | Maximum source-receiver distance to include |
| `short_distance_analysis.distance_types` | `hypocentral` and/or `epicentral` |
| `short_distance_analysis.phase_filter` | `null` (both), `P`, or `S` |

---

### 1.5 Source Parameter Analysis (`source_analysis.py`)

Analyzes hypocenter location corrections and origin time adjustments produced by the inversion.

**What it does:**

- Reads the source parameters CSV (postprocess output) with columns for updated coordinates and origin time.
- Computes correction vectors: Δx, Δy, Δz (km) and Δt₀ (s) relative to the preprocess starting locations.
- Plots a geographic correction vector map, magnitude histograms for each parameter, depth-dependent correction profiles, and a spatial distribution.
- Optionally writes a new CSV with the corrected coordinates and grouping labels (depth bin, zone).
- Supports `ground_truth` and `method_comparison` modes.

**Key configuration:**

| Key | Purpose |
|---|---|
| `source_analysis.enabled` | Enable/disable this module |
| `source_analysis.comparison_mode` | `ground_truth` or `method_comparison` |
| `source_analysis.input_csv` | Source parameters CSV path |
| `source_analysis.use_log_scale` | Log-scale histograms |
| `source_analysis.enable_depth_analysis` | Depth-dependent correction plots |
| `source_analysis.save_corrected_data` | Write corrected source CSV |

---

### 1.6 Gradient and Search Direction Analysis (`gradient_analysis.py`)

Diagnoses inversion convergence by inspecting the velocity and source gradient fields and their alignment with search directions.

**What it does:**

- Loads velocity and/or source gradient fields from the postprocess output CSV or HDF5.
- Computes gradient magnitude distributions and per-parameter histograms.
- Plots the alignment between gradient and search direction vectors (scatter with cosine similarity).
- Maps the spatial distribution of gradient magnitudes in depth slices and 1D depth profiles.
- Reports mean, maximum, and spatial distribution statistics.

**Key configuration:**

| Key | Purpose |
|---|---|
| `gradient_analysis.enabled` | Enable/disable this module |
| `gradient_analysis.analysis_type` | `["velocity"]`, `["source"]`, or `["velocity", "source"]` |
| `gradient_analysis.velocity_input_csv` | Velocity model CSV with grad/srch columns |
| `gradient_analysis.source_input_csv` | Source parameters CSV with grad/srch columns |

---

### 1.7 Derivative Weight Sum Analysis (`dws_analysis.py`)

Visualizes the volumetric ray path coverage using the DWS grid computed during postprocessing.

**What it does:**

- Reads the DWS HDF5 file and reshapes it to the 3D grid `(nz, ny, nx)`.
- Generates horizontal depth-slice plots (linear and log scale) at specified depths.
- Generates vertical XZ and YZ cross-section plots at specified positions.
- Produces binary coverage maps (cells with DWS > `coverage_threshold` are marked as covered).
- Computes coverage statistics: total cells, covered cells, percentage, min/max/mean/median DWS values.

**Key configuration:**

| Key | Purpose |
|---|---|
| `dws_analysis.enabled` | Enable/disable this module |
| `dws_analysis.input_hdf5` | Path to DWS HDF5 file |
| `dws_analysis.coverage_threshold` | DWS value above which a cell is considered covered |
| `dws_analysis.horizontal_slice_depths` | List of depths (km) for horizontal slice plots |
| `dws_analysis.vertical_slices.xz_positions` | Y-positions (km) for XZ vertical slices |
| `dws_analysis.vertical_slices.yz_positions` | X-positions (km) for YZ vertical slices |
| `dws_analysis.enable_log_scale` | Logarithmic color scale for DWS plots |
| `dws_analysis.use_m2_units` | Interpret DWS values as m² rather than km² |

---

### 1.8 Ray Path Visualization (`raypath_analysis.py`)

Projects 3D ray paths onto the XY, XZ, and YZ planes to visualize the ray coverage geometry.

**What it does:**

- Loads ray paths from the postprocess HDF5 cache, optionally filtering by event ID list, maximum ray count, or a random sub-sample fraction.
- Optionally filters to only receiver–event pairs listed in an anomalous pick CSV (matched by coordinate proximity within `filter_match_tolerance_km`).
- Renders one projection plot per plane, with configurable ray color, transparency, source markers, and receiver markers.
- Overlays optional basin boundary polygons.
- Reports the number of rays plotted and per-phase counts.

**Key configuration:**

| Key | Purpose |
|---|---|
| `raypath_analysis.enabled` | Enable/disable this module |
| `raypath_analysis.input_cache` | Path to ray path HDF5 cache |
| `raypath_analysis.event_id_list_file` | Filter to specific events (null = all) |
| `raypath_analysis.max_rays` | Cap total rays loaded (null = no limit) |
| `raypath_analysis.subsample_fraction` | Random fraction of rays to plot (0.0–1.0) |
| `raypath_analysis.filter_csv` | Anomalous pick CSV for receiver filtering |
| `raypath_analysis.filter_match_tolerance_km` | Coordinate match tolerance for receiver filter |

---

### 1.9 Ray Density Analysis (`ray_density_analysis.py`)

Quantifies the number of ray paths passing through each grid cell and maps coverage patterns.

**What it does:**

- Loads ray paths from the HDF5 cache and samples each path at `num_samples_per_ray` uniformly spaced points.
- Assigns each sample point to its nearest grid cell and increments that cell's ray count.
- Applies a threshold mask: cells with count < `threshold` are shown as masked (uncovered).
- Generates XY, XZ, and YZ projection maps (column sum over the third dimension), horizontal depth-slice maps at specified depths, and a depth-layer count map showing how many depth layers meet the threshold at each XY location.
- Reports coverage statistics.

**Key configuration:**

| Key | Purpose |
|---|---|
| `ray_density_analysis.enabled` | Enable/disable this module |
| `ray_density_analysis.input_cache` | Path to ray path HDF5 cache |
| `ray_density_analysis.grid.*` | Must match LATTE inversion grid |
| `ray_density_analysis.threshold` | Minimum ray count to mark a cell as covered |
| `ray_density_analysis.num_samples_per_ray` | Points sampled per ray for grid assignment |
| `ray_density_analysis.depth_slices` | Depths (km) for horizontal slice plots |

---

### 1.10 Tradeoff Curve Analysis (`tradeoff_analysis.py`)

Constructs and analyzes the L-curve (data misfit vs. model variance) to identify the optimal regularization point.

**What it does:**

- **`data_misfit` mode:** reads a whitespace-separated three-column file (`iteration | data_misfit | normalized_data_misfit`) produced by LATTE.
- **`rms` mode:** computes the per-iteration RMS from travel time residual CSVs located under `inversion_base_path` by iterating over per-iteration directories.
- Computes model variance from the velocity model CSV at each iteration, restricted to a configurable spatial region (`full`, `z_bounded`, or `depth_layer`) and aggregated as mean or maximum percentage change.
- Supports per-phase analysis with `phase_mode: separate` to generate independent Vp and Vs curves.
- Plots the tradeoff curve with optional iteration labels, a polynomial fit, and a gold-star annotation at the estimated optimal point.

**Key configuration:**

| Key | Purpose |
|---|---|
| `tradeoff_analysis.enabled` | Enable/disable this module |
| `tradeoff_analysis.y_axis_mode` | `data_misfit` (file) or `rms` (computed) |
| `tradeoff_analysis.data_misfit_file_path` | Path to data misfit file (data_misfit mode) |
| `tradeoff_analysis.inversion_base_path` | Root directory of per-iteration results (rms mode) |
| `tradeoff_analysis.phase_mode` | `both`, `p`, `s`, or `separate` |
| `tradeoff_analysis.model_variance.region` | `full`, `z_bounded`, or `depth_layer` |
| `tradeoff_analysis.model_variance.aggregation` | `mean` or `max` |

---

## 2. Outputs

All outputs are plots and text reports; no new data files are produced. Plot format (`png`, `pdf`, or `svg`) and DPI are controlled by `analysis.output_format` and `analysis.dpi`. Each module writes into its own `output_dir`.

| Module | Plots | Report |
|---|---|---|
| Travel time | Residual histogram, scatter, by-phase, ratio, method comparison | `travel_time_analysis_report.txt` |
| Velocity | Comparison, spatial projections, profile, slice heatmap, ratio, domain overview | `velocity_comparison_report.txt` |
| Distance | Residual vs. distance scatter, binned, correlation, ratio | `distance_analysis_report.txt` |
| Short distance | Map, path visualization, cluster zooms, histogram | `short_distance_analysis_report.txt` |
| Source | Correction map, histograms, depth profile, spatial distribution | `source_correction_analysis_report.txt` |
| Gradient | Histograms, alignment scatter, spatial distribution, depth profiles | `gradient_analysis_report.txt` |
| DWS | Horizontal slices (linear/log), vertical slices, coverage map, histogram | `dws_analysis_report.txt` |
| Ray path | XY, XZ, YZ projections | `ray_path_analysis_report.txt` |
| Ray density | XY/XZ/YZ projections, depth slices, threshold count map | `ray_density_analysis_report.txt` |
| Tradeoff | Tradeoff curve, iteration-labeled curve, per-phase curves | `tradeoff_analysis_report.txt` |

---

## 3. Usage and Configuration

### Running the pipeline

```bash
# Use the default configuration
python -m analysis.main

# Use a named configuration
python -m analysis.main --config-name example
```

Hydra writes run metadata to:

```
outputs/YYYY-MM-DD/HH-MM-SS_analysis_<config_name>/
```

### CLI overrides

```bash
# Enable only travel time analysis
python -m analysis.main travel_time_analysis.enabled=true

# Enable velocity analysis and change comparison mode
python -m analysis.main \
  velocity_analysis.enabled=true \
  velocity_analysis.comparison_mode=initial_vs_final \
  velocity_analysis.model1_csv=/data/initial/velocity_model_init.csv \
  velocity_analysis.model2_csv=/data/iter10/velocity_model_final.csv

# Enable DWS analysis with custom depth slices
python -m analysis.main \
  dws_analysis.enabled=true \
  "dws_analysis.horizontal_slice_depths=[5.0,10.0,15.0,20.0]"

# Enable ray density and match grid to LATTE
python -m analysis.main \
  ray_density_analysis.enabled=true \
  ray_density_analysis.grid.nx=60 \
  ray_density_analysis.grid.ny=60 \
  ray_density_analysis.grid.nz=50

# Enable tradeoff analysis in RMS mode
python -m analysis.main \
  tradeoff_analysis.enabled=true \
  tradeoff_analysis.y_axis_mode=rms \
  tradeoff_analysis.inversion_base_path=/data/iterations

# Change global output format and DPI
python -m analysis.main analysis.output_format=pdf analysis.dpi=600

# Change log level
python -m analysis.main analysis.log_level=DEBUG
```

### Minimal configuration checklist

Each module requires only its `enabled` flag and the relevant input paths. The following keys are common across modules:

```yaml
analysis:
  log_level: INFO
  output_format: png     # png | pdf | svg
  dpi: 300
  phase_type: both       # both | p | s
  plot_style:
    font_size: 14

# Optional but recommended — add basin boundary overlays
basin_boundaries:
  data_dir: /path/to/basin_files/cartesian
  basins:
    - {file: delaware.csv, label: Delaware Basin}

# Example: enable travel time and velocity modules
travel_time_analysis:
  enabled: true
  comparison_mode: ground_truth
  input_csv: /path/to/synthetic_traveltimes.csv
  output_dir: /path/to/output/travel_time

velocity_analysis:
  enabled: true
  comparison_mode: initial_vs_final
  model1_csv: /path/to/initial/velocity_model_init.csv
  model2_csv: /path/to/final/velocity_model_final.csv
  output_dir: /path/to/output/velocity
```

Refer to `configs/analysis/example.yaml` for a complete working template.

---

## 4. Logging

The log level is controlled by `analysis.log_level` (default: `INFO`). Valid values are `DEBUG`, `INFO`, `WARNING`, and `ERROR`. A `ProcessTimer` context manager wraps each module and logs its wall-clock duration.

| Level | What is logged |
|---|---|
| `DEBUG` | Input data shapes; filter counts before and after; interpolation fallback events; per-iteration model variance values |
| `INFO` | Module start/end with timing; input record counts; summary statistics (mean, std, RMS); output directory paths; plot file names |
| `WARNING` | Missing optional columns (gradient, search); picks present in one file only (comparison mode); depth slices outside grid bounds; out-of-range DWS values |
| `ERROR` | Missing required input files; coordinate inconsistencies; failed plot generation |

Key log messages to monitor:

```
# Confirms module timing
[ProcessTimer] Travel time analysis completed in X.X s

# Summary statistics — verify plausible values
RMS residual (all picks):   X.XXX s
Mean residual:              X.XXX ± X.XXX s

# DWS coverage check
DWS covered cells: N / TOTAL (P.P%)

# Ray density check
Rays plotted: N (P-wave: M, S-wave: K)
```

---

## 5. Troubleshooting

**Module is enabled but produces no output**
Confirm that all required input paths exist and are readable. Check the log for `ERROR` or `WARNING` messages. Run with `analysis.log_level=DEBUG` to see per-step progress.

**Travel time residual plots show bimodal distribution**
A bimodal pattern typically indicates a systematic origin time offset or an incorrect configured origin time that was not subtracted during postprocessing. Verify that `origin_time.fixed_value` in the preprocess config matches what was used when building the travel time binaries.

**Velocity comparison shows no change between models**
Confirm that `model1_csv` and `model2_csv` point to different files (initial vs. final iteration). If `use_coarse_dataset: true`, confirm that the `velocity_coarse` dataset exists in both HDF5 files.

**DWS plot is uniformly zero or very small**
The DWS HDF5 was computed from a subsampled ray set. Rerun postprocess DWS with `subsample_fraction: 1.0`. Also check that `dws_analysis.use_m2_units` is consistent with the units used during computation.

**Ray path or ray density plots are blank**
The HDF5 cache may be empty or corrupted. Verify the cache file size and run postprocess `ray_path_diagnosis` with `force_recompute: true`. Also check that `phase_type` is consistent between cache creation and analysis.

**Tradeoff curve shows only one point**
In `rms` mode, the pipeline expects one per-iteration residual CSV under `inversion_base_path/iter_N/`. Confirm the directory naming convention matches the pattern expected by the module, or switch to `data_misfit` mode and provide the three-column file directly.

**Source correction vectors appear unrealistically large**
The number of kept shots (`initial_geometry_csv` minus `source_processing.filtered_shot_ids`) must match the binary size, otherwise postprocess raises an error; check `filtered_shot_ids` against the shots LATTE removed. A size mismatch causes incorrect binary indexing, producing spurious correction values.

**Basin boundary overlays do not appear**
Boundary files must be in Cartesian coordinates (x, y in km), not geographic. Use the Cartesian basin boundary CSVs produced by the preprocess coordinate transformation stage. Confirm `basin_boundaries.data_dir` points to the correct directory.

---

## 6. Best Practices

- **Enable modules one at a time during setup.** This isolates configuration issues and avoids confusing errors from multiple modules running simultaneously.
- **Start with travel time and velocity analysis.** These two modules directly assess inversion quality and should be run before more specialized diagnostics (DWS, gradients, tradeoff).
- **Use `method_comparison` mode to evaluate multiple inversions.** Rather than interpreting each model independently, direct comparison between two runs highlights meaningful differences and suppresses common artifacts.
- **Set consistent `phase_type` across all modules.** Mixing phase filters (e.g., `both` in one module and `p` in another) makes cross-module comparison difficult.
- **Use `pdf` output format for publications.** PNG is sufficient for rapid iteration; switch to PDF or SVG for final figures to preserve vector quality.
- **Verify DWS coverage before interpreting velocity anomalies.** Velocity changes in cells with DWS below the coverage threshold are poorly constrained. Use the DWS coverage map to mask low-coverage regions in velocity plots.
- **Run ray density analysis before DWS.** Ray density is faster to compute (no weighting) and gives a quick qualitative view of coverage. Use it to tune grid parameters before committing to a full DWS computation.
- **Inspect gradient histograms to diagnose convergence.** Gradient magnitudes that do not decrease between iterations suggest a step size or regularization problem in the inversion, not a data quality issue.
- **Archive the config snapshot from each Hydra run.** Hydra saves a copy of the resolved configuration in `outputs/`. Keeping these snapshots alongside results makes it straightforward to reproduce any analysis plot.
