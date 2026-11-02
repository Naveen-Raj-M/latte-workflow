# Analysis Pipeline

The analysis pipeline provides post-inversion diagnostics for LATTE tomography results. It reads the CSV, HDF5 and ray path cache files written by [postprocess](postprocess.md) and produces plots, statistics and text reports. All ten modules are independent and disabled by default; enable only the modules needed for an evaluation.

> **Configuration reference:** See [`configs/analysis/README.md`](../configs/analysis/README.md) for a complete description of every configuration parameter.

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

The pipeline is driven by `analysis/main.py`. The modules run in the order of the sections below, except that ray path analysis runs before DWS analysis. Each module is guarded by an `enabled` flag. A failing module is logged and the next one runs; add `+strict_mode=true` to stop at the first failure.

---

### 1.1 Travel Time Residual Analysis (`traveltime_analysis.py`)

Analyses the residuals between synthetic travel times and the observed travel times.

**What it does:**

- **`ground_truth` mode:** reads one travel time CSV and analyses the residual distribution (`observed_tt − synthetic_tt`).
- **`method_comparison` mode:** reads two CSVs, compares them pick by pick, and logs the picks that are present in only one file.
- Excludes residuals outside `min_residual_threshold` / `max_residual_threshold` and flags outliers beyond `outlier_threshold_sigma`.
- Computes mean, standard deviation, median and weighted RMS, per phase and overall.
- Plots residual histograms, synthetic-versus-observed scatter plots (linear and log–log) and natural-log ratio plots.
- Writes a text report and optional mismatch and outlier logs.

**Key configuration:**

| Key | Purpose |
|---|---|
| `travel_time_analysis.enabled` | Enable/disable this module |
| `travel_time_analysis.comparison_mode` | `ground_truth` or `method_comparison` |
| `travel_time_analysis.input_csv` | Travel time CSV (`ground_truth` mode) |
| `travel_time_analysis.method1_csv`, `method2_csv` | Two CSVs (`method_comparison` mode) |
| `travel_time_analysis.outlier_threshold_sigma` | Outlier cutoff in standard deviations |
| `travel_time_analysis.min_residual_threshold`, `max_residual_threshold` | Residual bounds in s (±50 by default) |

---

### 1.2 Velocity Model Analysis (`velocity_analysis.py`)

Compares two velocity models and characterises their spatial differences.

**What it does:**

- Reads two velocity files (CSV or HDF5), for example the initial and final model, or the models of two methods.
- Computes differences (Δvp, Δvs), percent changes and log ratios (ln(v₂/v₁)), and geometric-mean Vp/Vs ratios.
- Produces spatial projections on the XY, XZ and YZ planes, depth-slice heatmaps, vertical profiles at the locations of largest difference, per-plane ratio slices and a domain overview.
- Writes a text report with statistics per plane.

**Key configuration:**

| Key | Purpose |
|---|---|
| `velocity_analysis.enabled` | Enable/disable this module |
| `velocity_analysis.comparison_mode` | `initial_vs_final` / `method_comparison` / `custom` (sets plot labels) |
| `velocity_analysis.model1_csv`, `model2_csv` (or `*_hdf5`) | The two models |
| `velocity_analysis.use_coarse_dataset` | Read the `velocity_coarse` HDF5 dataset for speed |
| `velocity_analysis.compute_log_ratios` | Also compute ln-ratios |
| `velocity_analysis.enable_*` | Switch individual plot groups |
| `velocity_analysis.plane_ratio_plots.planes` | Planes and coordinates of the ratio slices |

---

### 1.3 Distance-Dependent Residual Analysis (`distance_analysis.py`)

Examines how the residuals depend on source–receiver distance.

**What it does:**

- Merges the travel time CSV with the event and station Cartesian coordinates to compute hypocentral and/or epicentral distances.
- Bins the picks by distance and computes the mean residual and standard deviation per bin.
- Computes the Pearson or Spearman correlation between residual and distance, and fits a polynomial trend.
- Plots residuals versus distance, binned statistics, the correlation analysis and (optionally) ratios versus distance.
- Supports `ground_truth` and `method_comparison` modes.

**Key configuration:**

| Key | Purpose |
|---|---|
| `distance_analysis.enabled` | Enable/disable this module |
| `distance_analysis.comparison_mode` | `ground_truth` or `method_comparison` |
| `distance_analysis.travel_time_csv`, `station_csv`, `events_csv` | Inputs (`ground_truth` mode) |
| `distance_analysis.distance_types` | `hypocentral` and/or `epicentral` |
| `distance_analysis.distance_bins` | Number of distance bins |
| `distance_analysis.correlation_method` | `pearson` or `spearman` |

---

### 1.4 Short-Distance Residual Analysis (`short_distance_analysis.py`)

Restricts the residual analysis to near-field event–station pairs.

**What it does:**

- Keeps only the picks within `distance_cutoff_km`.
- Plots an overview map of the retained pairs with travel path lines.
- Renders a zoomed map for every cluster defined in `zoom_clusters`.
- Plots residual histograms and residual-versus-distance for the subset.
- Reports statistics by distance type and phase.

**Key configuration:**

| Key | Purpose |
|---|---|
| `short_distance_analysis.enabled` | Enable/disable this module |
| `short_distance_analysis.distance_cutoff_km` | Maximum source–receiver distance |
| `short_distance_analysis.distance_types` | `hypocentral` and/or `epicentral` |
| `short_distance_analysis.phase_filter` | `null` (use `analysis.phase_type`), `P` or `S` |
| `short_distance_analysis.zoom_clusters` | Cartesian bounding boxes of the zoom panels |

---

### 1.5 Source Parameter Analysis (`source_analysis.py`)

Analyses the hypocenter and origin time corrections produced by the inversion.

**What it does:**

- Reads `source_relocation.csv` and computes the corrections Δx, Δy, Δz (km) and Δt₀ (s) relative to the starting locations. `parameters_to_process` selects the parameters (for example `["sx", "sy", "sz"]` when origin times were not updated).
- Plots correction vector maps, histograms, scatter relationships between corrections, depth-dependent corrections and a summary.
- Optionally writes a CSV with the computed deltas (`source_corrections_with_deltas.csv`).
- Supports `ground_truth` and `method_comparison` modes.

**Key configuration:**

| Key | Purpose |
|---|---|
| `source_analysis.enabled` | Enable/disable this module |
| `source_analysis.comparison_mode` | `ground_truth` or `method_comparison` |
| `source_analysis.input_csv` | Source relocation CSV |
| `source_analysis.parameters_to_process` | Subset of `sx`, `sy`, `sz`, `st0` |
| `source_analysis.enable_depth_analysis` | Depth-dependent correction plots |
| `source_analysis.save_corrected_data` | Write the CSV with deltas |

---

### 1.6 Gradient and Search Direction Analysis (`gradient_analysis.py`)

Diagnoses inversion convergence from the gradient fields and their alignment with the search directions.

**What it does:**

- Reads velocity and/or source gradient and search direction columns from the postprocess CSV or HDF5 files.
- Plots gradient and search direction histograms, gradient–search alignment (cosine similarity), spatial maps of the gradient magnitude, and depth profiles.
- Writes a report per analysis type.

**Key configuration:**

| Key | Purpose |
|---|---|
| `gradient_analysis.enabled` | Enable/disable this module |
| `gradient_analysis.analysis_type` | `["velocity"]`, `["source"]` or both |
| `gradient_analysis.velocity_input_csv` | Velocity file with gradient/search columns |
| `gradient_analysis.source_input_csv` | Source file with gradient/search columns |

---

### 1.7 Ray Path Visualization (`raypath_analysis.py`)

Projects ray paths onto the XY, XZ and YZ planes.

**What it does:**

- Reads rays from the ray path cache, optionally restricted by `event_id_list_file`, `max_rays` or `subsample_fraction`.
- Optionally keeps only the receivers listed in `filter_csv` (matched by coordinates within `filter_match_tolerance_km`).
- Plots one projection per plane and phase with configurable ray, source and receiver styles, and optional basin boundaries.
- Writes statistics and a text report.

**Key configuration:**

| Key | Purpose |
|---|---|
| `raypath_analysis.enabled` | Enable/disable this module |
| `raypath_analysis.input_cache` | Ray path HDF5 cache |
| `raypath_analysis.event_id_list_file`, `shot_map_file` | Restrict to listed events |
| `raypath_analysis.max_rays`, `subsample_fraction` | Limit the number of rays |
| `raypath_analysis.filter_csv`, `filter_match_tolerance_km` | Receiver filter |
| `raypath_analysis.plot.*` | Styling |

---

### 1.8 Derivative Weight Sum Analysis (`dws_analysis.py`)

Visualises the ray coverage of the model using the DWS grid.

**What it does:**

- Reads the DWS HDF5 file (columns `x`, `y`, `z`, `dws`) and reshapes it to the model grid.
- Plots horizontal depth slices and vertical XZ and YZ slices, in linear and logarithmic scale.
- Plots binary coverage maps (cells with DWS above `coverage_threshold` are covered) and DWS histograms.
- Reports the number of covered cells and the minimum, maximum, mean and median DWS.

**Key configuration:**

| Key | Purpose |
|---|---|
| `dws_analysis.enabled` | Enable/disable this module |
| `dws_analysis.input_hdf5` | DWS HDF5 file |
| `dws_analysis.coverage_threshold` | DWS above which a cell counts as covered |
| `dws_analysis.horizontal_slice_depths` | Depths (km) of the horizontal slices |
| `dws_analysis.vertical_slices.xz_positions`, `yz_positions` | Positions (km) of the vertical slices |
| `dws_analysis.enable_log_scale` | Add logarithmic plots |
| `dws_analysis.use_m2_units` | Show DWS in m² instead of km² |

---

### 1.9 Ray Density Analysis (`ray_density_analysis.py`)

Counts the rays that pass through each grid cell.

**What it does:**

- Samples every ray at `num_samples_per_ray` points and assigns the samples to their nearest grid cell.
- Masks cells whose count is below `threshold`.
- Plots XY, XZ and YZ projections (sum over the third axis), horizontal depth slices, and a map of how many depth layers reach the threshold at each XY location.
- Optionally plots the density along user-defined **oblique vertical sections** (`enable_oblique_sections`, `oblique_sections`) and a plan view of the section traces (`enable_oblique_section_plan_view`).
- Saves statistics per phase.

**Key configuration:**

| Key | Purpose |
|---|---|
| `ray_density_analysis.enabled` | Enable/disable this module |
| `ray_density_analysis.input_cache` | Ray path HDF5 cache |
| `ray_density_analysis.grid.*` | Model grid; the metadata of the ray path cache takes precedence, configured values fill missing keys |
| `ray_density_analysis.threshold` | Minimum ray count for a covered cell |
| `ray_density_analysis.num_samples_per_ray` | Samples per ray |
| `ray_density_analysis.depth_slices` | Depths (km) of the horizontal slices |
| `ray_density_analysis.oblique_sections` | Sections defined by `label`, `x1`, `y1`, `x2`, `y2` (km) |

---

### 1.10 Tradeoff Curve Analysis (`tradeoff_analysis.py`)

Builds the tradeoff (L-curve) between data fit and model change across inversion iterations, or across a parameter sweep.

**What it does:**

- **X axis:** `x_axis_mode: iteration` finds the `iteration_<N>/` directories under `inversion_base_path`; `x_axis_mode: parameter` uses the entries listed in `parameter_mode` (one directory and iteration per parameter value, for P and S).
- **Y axis (`y_axis_mode`):**
  - `data_misfit`: reads the three-column LATTE misfit file (`iteration | data_misfit | normalized_data_misfit`);
  - `rms`: computes the RMS residual from the travel time CSV of each iteration (`use_weighted_rms` selects pick weights);
  - `data_variance`: mean squared residual (s²).
  
  `use_log_residuals` replaces the residual by the absolute log difference of observed and synthetic travel times.
- **Model change (X axis values):** computed from the velocity files of each iteration relative to the initial model, over a region (`full`, `z_bounded` or `depth_layer`, with optional `x_bounds`/`y_bounds`), as `mean`, `max`, `model_variance` or `rms_variation` (with `exclude_zero_change` to skip unchanged nodes), optionally as a percentage.
- `phase_mode: separate` produces independent Vp and Vs curves.
- Plots the curve with optional iteration labels, colouring by iteration and an elbow marker, and writes statistics CSVs and a report.

**Key configuration:**

| Key | Purpose |
|---|---|
| `tradeoff_analysis.enabled` | Enable/disable this module |
| `tradeoff_analysis.x_axis_mode` | `iteration` or `parameter` |
| `tradeoff_analysis.y_axis_mode` | `data_misfit`, `rms` or `data_variance` |
| `tradeoff_analysis.data_misfit_file_path` | LATTE misfit file (`data_misfit` mode) |
| `tradeoff_analysis.inversion_base_path` | Directory with `iteration_<N>/` results |
| `tradeoff_analysis.phase_mode` | `both`, `p`, `s` or `separate` |
| `tradeoff_analysis.model_variance.*` | Region, aggregation and units of the model change |

---

## 2. Outputs

Analysis produces plots, statistics and text reports; it writes no new data tables except the optional source-delta CSV. Plot format (`png`, `pdf`, `svg`) and DPI are set by `analysis.output_format` and `analysis.dpi`. Each module writes into its own `output_dir`, usually with the subdirectories `plots/`, `reports/` and (for some modules) `statistics/`.

| Module | Main plots | Report |
|---|---|---|
| Travel time | `residual_histograms`, `synthetic_vs_observed`, `synthetic_vs_observed_loglog`, `ln_ratio_analysis`, `ln_ratio_histograms` | `travel_time_analysis_report.txt` |
| Velocity | `basic_velocity_comparison`, `spatial_velocity_projections`, `spatial_geometric_ratio_projections`, `velocity_vertical_profiles`, `depth_slice_heatmaps`, `xy/xz/yz_plane_ratio_*`, `domain_overview` | `velocity_comparison_report.txt` |
| Distance | `residuals_vs_distance`, `binned_distance_analysis`, `correlation_analysis`, `ratios_vs_distance` | `distance_analysis_report_*.txt` |
| Short distance | `overview_map`, `residual_histograms`, `distance_vs_residual`, `cluster_<name>` | `short_distance_report_*.txt` |
| Source | `hist_*`, `scatter_relationships`, `geographic_corrections_<plane>`, `depth_analysis`, `correction_summary` | `source_correction_report.txt` |
| Gradient | `*_gradient_histograms`, `*_search_histograms`, `*_gradient_search_alignment`, `velocity_gradient_*_xy_projection`, `velocity_gradient_depth_profiles` | `velocity_gradient_analysis_report.txt`, `source_gradient_analysis_report.txt` |
| Ray path | `ray_paths_<phase>_wave_<plane>`, `raypath_summary` | `raypath_analysis_report.txt` |
| DWS | `dws_horizontal_depth_<d>km_{linear,log}`, `dws_vertical_{xz,yz}_pos_<p>km_*`, `dws_coverage_depth_<d>km`, `dws_histogram_{linear,log}` | `dws_analysis_report.txt` |
| Ray density | `ray_density_<phase>_wave_<plane>`, `…_xy_threshold_count`, `ray_density_depth_<d>km`, `…_oblique_<label>` | statistics JSON |
| Tradeoff | `tradeoff_curve`, `tradeoff_curve_<phase>`, `tradeoff_curve_depth_<d>km` | `tradeoff_analysis_report.txt`, `tradeoff_statistics.csv` |

---

## 3. Usage and Configuration

### Running the pipeline

```bash
# Default configuration (configs/analysis/default.yaml)
python -m analysis.main

# A named configuration
python -m analysis.main --config-name example
```

Hydra writes the resolved configuration and the log to:

```
outputs/YYYY-MM-DD/HH-MM-SS_analysis_<config_name>/
```

### CLI overrides

```bash
# Travel time analysis only
python -m analysis.main travel_time_analysis.enabled=true

# Velocity analysis with explicit models
python -m analysis.main \
  velocity_analysis.enabled=true \
  velocity_analysis.comparison_mode=initial_vs_final \
  velocity_analysis.model1_csv=/data/preprocess/velocity_model_init.csv \
  velocity_analysis.model2_csv=/data/iter10/velocity_model_final.csv

# DWS slices at chosen depths
python -m analysis.main \
  dws_analysis.enabled=true \
  "dws_analysis.horizontal_slice_depths=[5.0,10.0,15.0,20.0]"

# Ray density on the LATTE grid
python -m analysis.main \
  ray_density_analysis.enabled=true \
  ray_density_analysis.grid.nx=60 \
  ray_density_analysis.grid.ny=60 \
  ray_density_analysis.grid.nz=50

# Tradeoff curve from per-iteration residuals
python -m analysis.main \
  tradeoff_analysis.enabled=true \
  tradeoff_analysis.y_axis_mode=rms \
  tradeoff_analysis.inversion_base_path=/data/inversion/run_1

# Global plot settings
python -m analysis.main analysis.output_format=pdf analysis.dpi=600

# Stop at the first failing module
python -m analysis.main +strict_mode=true

# Log verbosity
python -m analysis.main analysis.log_level=DEBUG
```

### Minimal configuration checklist

Each module needs its `enabled` flag and its input paths. Settings shared by all modules:

```yaml
analysis:
  log_level: INFO
  output_format: png     # png | pdf | svg
  dpi: 300
  phase_type: both       # both | p | s

paths:
  input_dir: /path/to/postprocess/output
  output_base_dir: /path/to/analysis/output

# Optional: basin boundary overlays
basin_boundaries:
  data_dir: /path/to/basin_files/cartesian
  basins:
    - {file: basin_a.csv, label: Basin A}

travel_time_analysis:
  enabled: true
  comparison_mode: ground_truth
  input_csv: /path/to/synthetic_traveltimes.csv
  output_dir: /path/to/output/travel_time

velocity_analysis:
  enabled: true
  comparison_mode: initial_vs_final
  model1_csv: /path/to/velocity_model_init.csv
  model2_csv: /path/to/velocity_model_final.csv
  output_dir: /path/to/output/velocity
```

`configs/analysis/example.yaml` is a ready-to-copy template.

---

## 4. Logging

The log level is set by `analysis.log_level` (default `INFO`; `DEBUG`, `INFO`, `WARNING`, `ERROR`). A timer wraps every module and logs its duration (`Completed <module> in <time>`); at the end the pipeline prints the lists of completed and failed modules.

| Level | What is logged |
|---|---|
| `DEBUG` | Input shapes; counts before and after filters; per-iteration model variance |
| `INFO` | Module start and end with timing; input record counts; summary statistics; output directories |
| `WARNING` | Missing optional columns; picks present in one file only (comparison mode); depth slices outside the grid; out-of-range values |
| `ERROR` | Missing required inputs; failed plots; the failure of a module |

---

## 5. Troubleshooting

**A module is enabled but writes nothing**
Check that its input paths exist, then look for `ERROR` and `WARNING` lines in the log. `analysis.log_level=DEBUG` shows the progress of each step.

**The pipeline summary lists a failed module**
The failure is logged with its message; the other modules still ran. Fix the cause (usually an input path) and re-run with only that module enabled.

**Travel time residuals are bimodal**
This usually points to an origin time offset that was not removed. Make sure `origin_time.fixed_value` of the preprocess run matches the value used for the travel time binaries, and that source correction in postprocess is consistent with the run.

**Velocity analysis fails with `attempt to get argmin of an empty sequence`**
The two models are identical, so there is no difference to locate. Check that `model1_csv` and `model2_csv` point to different models. With `use_coarse_dataset: true`, both HDF5 files must contain the `velocity_coarse` dataset.

**DWS plots are zero or very small**
The DWS file was computed from a subsampled ray set. Re-run postprocess with `subsample_fraction: 1.0` and `compute_dws: true`, and check that `use_m2_units` matches the intended unit.

**Ray path or ray density plots are empty**
The cache may be empty or damaged. Check its size and rebuild it with `force_recompute: true` in postprocess. `phase_type` must be consistent between postprocess and analysis.

**Tradeoff curve shows only one point**
In `iteration` mode the module looks for directories named `iteration_<N>` under `inversion_base_path`, each holding the travel time and velocity files matched by `traveltime_csv_pattern` and `velocity_csv_pattern`. Check the directory names and patterns, or use `data_misfit` mode with the misfit file.

**Source corrections look unrealistically large**
The number of kept shots (`initial_geometry_csv` minus `filtered_shot_ids`) must equal the size of the source binaries; otherwise postprocess stops with an error. Check `filtered_shot_ids` against the shots removed by LATTE.

**Basin boundaries are not drawn**
Boundary files must be in Cartesian coordinates (x, y in km), as written to `coordinate_transform/basin_boundaries/` by preprocess. Check `basin_boundaries.data_dir` and that `plot_basin_boundaries: true` is set for the module.

---

## 6. Best Practices

- **Enable modules one at a time during setup.** This isolates configuration problems.
- **Start with travel time and velocity analysis.** They assess inversion quality directly and should come before the specialised diagnostics.
- **Use `method_comparison` to compare inversions.** A direct comparison highlights real differences and suppresses shared artifacts.
- **Keep `phase_type` consistent across modules.** Mixed phase filters make results hard to compare.
- **Use `pdf` for publication figures.** Use `png` for fast iteration.
- **Check coverage before interpreting anomalies.** Velocity changes in cells with low DWS or ray density are poorly constrained.
- **Run ray density before DWS.** It is cheaper and gives a quick view of the coverage; use it to check the grid settings first.
- **Inspect the gradient histograms.** Gradient magnitudes that do not decrease between iterations point to a step-size or regularisation problem rather than a data problem.
- **Keep the Hydra run directory.** `outputs/` holds the resolved configuration of every run; archive it next to the results to reproduce any figure.
