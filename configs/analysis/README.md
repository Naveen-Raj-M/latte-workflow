# Analysis Configuration

## Overview

The analysis pipeline provides post-inversion analysis of [LATTE](https://github.com/lanl/latte_traveltime) tomography results. It reads post-processed outputs (velocity models, synthetic travel times, source parameters, ray path caches, and DWS grids) and produces plots, statistics, and reports for evaluating inversion quality.

All analysis modules are independently enabled and support two comparison modes:
- **`ground_truth`** — compare inverted results against observed/reference data
- **`method_comparison`** — compare two inversion methods directly

**Available analysis modules:**
1. Gradient analysis — velocity and source gradient/search direction diagnostics
2. Travel time analysis — residual statistics and distributions
3. Velocity analysis — model comparison, spatial differences, and ratios
4. Distance analysis — residual dependence on source–receiver distance
5. Short-distance analysis — residual analysis restricted to near-field pairs
6. Source analysis — hypocenter correction distributions and spatial patterns
7. DWS analysis — derivative weight sum model coverage and resolution
8. Ray path analysis — 3D ray path projection visualization
9. Ray density analysis — ray count per grid cell and coverage mapping
10. Tradeoff analysis — data misfit vs. model variance (L-curve)

---

## Quick Start

### Minimal configuration

Copy `example.yaml` and update the file paths for your dataset:

```yaml
# configs/analysis/example.yaml
defaults:
  - default
  - _self_

paths:
  input_dir: "/path/to/postprocessed/output"
  output_base_dir: "/path/to/analysis/output"

basin_boundaries:
  data_dir: "/path/to/preprocessed/coordinate_transform/basin_boundaries"

travel_time_analysis:
  enabled: true
  input_csv: "/path/to/synthetic_traveltimes.csv"
  output_dir: "/path/to/analysis/output/travel_time"

velocity_analysis:
  enabled: true
  model1_csv: "/path/to/velocity_model_init.csv"
  model2_csv: "/path/to/velocity_model_final.csv"
  output_dir: "/path/to/analysis/output/velocity_comparison"
```

All other parameters inherit their defaults from `default.yaml`.

### Run

```bash
python3 -m analysis.main --config-name example
```

### CLI overrides

Any key can be overridden at the command line using Hydra dot-notation without modifying the YAML file:

```bash
python3 -m analysis.main --config-name example \
    analysis.phase_type=p \
    travel_time_analysis.outlier_threshold_sigma=2.5 \
    velocity_analysis.use_coarse_dataset=true
```

---

## Configuration Reference

### `paths`

Global path patterns used as defaults across all analysis modules. Module-level path keys override these when specified.

| Key | Default | Description |
|-----|---------|-------------|
| `input_dir` | `data/processed` | Root directory for all input files. Used to resolve relative paths within modules. |
| `output_base_dir` | `results/analysis` | Root directory for all analysis outputs. |
| `travel_time_csv_pattern` | `synthetic_traveltimes.csv` | Glob pattern for travel time residual CSVs. |
| `velocity_csv_pattern` | `velocity_model_final*.csv` | Glob pattern for velocity model CSVs. |
| `station_csv_name` | `stations_cartesian.csv` | Filename for the Cartesian station list. |
| `events_csv_name` | `events_cartesian*.csv` | Glob pattern for the Cartesian events file. |
| `source_csv_pattern` | `source_relocation.csv` | Glob pattern for source parameter CSVs. |
| `plots_subdir` | `plots` | Subdirectory name for plot outputs within each module's output directory. |
| `stats_subdir` | `statistics` | Subdirectory name for statistics outputs. |
| `reports_subdir` | `reports` | Subdirectory name for text report outputs. |

---

### `analysis`

Global settings applied to all enabled modules.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `log_level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | Logging verbosity. |
| `output_format` | `png` | `png`, `pdf`, `svg` | Plot file format. |
| `dpi` | `300` | — | Plot resolution in DPI. |
| `phase_type` | `both` | `both`, `p`, `s` | Phase filter applied globally before each module runs. |

#### `analysis.plot_style`

Matplotlib style settings applied globally before each module runs.

| Key | Default | Description |
|-----|---------|-------------|
| `font_family` | `sans-serif` | Font family for all plot text. |
| `font_size` | `14` | Base font size (pt). |
| `title_size` | `16` | Axes title font size (pt). |
| `label_size` | `14` | Axes label font size (pt). |
| `tick_size` | `12` | Tick label font size (pt). |
| `legend_size` | `12` | Legend font size (pt). |
| `figure_title_size` | `18` | Figure suptitle font size (pt). |
| `line_width` | `2` | Default line width. |
| `marker_size` | `4` | Default marker size. |
| `alpha` | `0.7` | Default element transparency (0 = transparent, 1 = opaque). |
| `grid_alpha` | `0.3` | Grid line transparency. |
| `max_plot_points` | `100000` | Scatter plots downsample to this limit to keep rendering fast. |

#### `analysis.plot_style.colors`

Hex color palette used across all plot modules.

| Key | Default | Used for |
|-----|---------|----------|
| `p_wave_dark` | `#1f4e79` | Primary P-wave series. |
| `p_wave_light` | `#5b9bd5` | Secondary P-wave series. |
| `s_wave_dark` | `#2e7d32` | Primary S-wave series. |
| `s_wave_light` | `#66bb6a` | Secondary S-wave series. |
| `event_color` | `#ff6b35` | Event/source markers. |
| `station_color` | `#4a4a4a` | Station markers. |
| `neutral_color` | `#d32f2f` | Neutral reference elements. |
| `grid_color` | `#cccccc` | Grid lines. |
| `positive_residual` | `#ff4444` | Positive residual markers. |
| `negative_residual` | `#4444ff` | Negative residual markers. |
| `zero_line` | `#000000` | Zero-reference lines. |

---

### `basin_boundaries`

Optional basin boundary overlays drawn on XY-plane plots across all modules that support `plot_basin_boundaries`. Boundary files should be Cartesian-transformed CSVs produced by the preprocess module.

| Key | Default | Description |
|-----|---------|-------------|
| `data_dir` | — | Directory containing basin boundary CSV files. |
| `basins` | — | List of basin entries. Each entry requires `file` (CSV filename) and `label` (display name). |
| `line_color` | `darkgreen` | Boundary line color. |
| `line_width` | `2` | Boundary line width. |
| `line_style` | `-` | Line style: `-`, `--`, `:`, `-.`. |
| `line_alpha` | `0.8` | Boundary line transparency (0 = transparent, 1 = opaque). |
| `show_labels` | `true` | Annotate each basin with its label. |
| `label_fontsize` | `12` | Basin label font size (pt). |
| `label_fontweight` | `bold` | Font weight: `normal`, `bold`. |
| `label_offset_x` / `label_offset_y` | `0.0` | km | Label offset from the basin centroid. |

---

### `gradient_analysis`

Analyzes velocity and source gradient fields alongside search directions to diagnose inversion convergence.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `false` | `true`, `false` | Enable this module. |
| `analysis_type` | `["velocity", "source"]` | subset of `velocity`, `source` | Which gradient types to process. |
| `velocity_input_csv` | — | — | Path to the post-processed velocity CSV. HDF5 takes precedence if `velocity_input_hdf5` is set. |
| `velocity_input_hdf5` | `null` | — | Path to the velocity HDF5 file. `null` = use CSV. |
| `velocity_output_dir` | — | — | Output directory for velocity gradient plots and reports. |
| `source_input_csv` | — | — | Path to the post-processed source parameter CSV. |
| `source_input_hdf5` | `null` | — | Path to the source HDF5 file. `null` = use CSV. |
| `source_output_dir` | — | — | Output directory for source gradient plots and reports. |
| `histogram_bins` | `50` | — | Number of bins for gradient magnitude histograms. |
| `depth_bins` | `10` | — | Number of depth bins for depth-profile plots. |
| `min_points_per_bin` | `10` | — | Minimum points per bin; bins below this are excluded. |
| `generate_report` | `true` | `true`, `false` | Write a text summary report. |
| `create_histograms` | `true` | `true`, `false` | Generate gradient magnitude distribution histograms. |
| `create_alignment_plots` | `true` | `true`, `false` | Generate gradient–search direction alignment plots. |
| `create_spatial_plots` | `true` | `true`, `false` | Generate spatial gradient magnitude maps. |
| `create_depth_profiles` | `true` | `true`, `false` | Generate gradient depth profile plots. |
| `plot_basin_boundaries` | `false` | `true`, `false` | Overlay basin boundaries on XY spatial plots. |

---

### `travel_time_analysis`

Analyzes travel time residuals between synthetic predictions and observed ground truth, or between two methods.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `false` | `true`, `false` | Enable this module. |
| `comparison_mode` | `ground_truth` | `ground_truth`, `method_comparison` | `ground_truth`: single CSV compared against zero. `method_comparison`: two CSVs compared against each other. |
| `input_csv` | `null` | — | Travel time residual CSV. Used when `comparison_mode: ground_truth`. |
| `output_dir` | `null` | — | Output directory for plots and reports. |
| `method1_csv` / `method2_csv` | `null` | — | Travel time CSVs for each method. Used when `comparison_mode: method_comparison`. |
| `method1_name` / `method2_name` | `Method 1` / `Method 2` | — | Display names for each method in plots. |
| `enable_log_log_plots` | `true` | `true`, `false` | Generate log–log scale residual plots. |
| `enable_ln_ratio_analysis` | `true` | `true`, `false` | Compute and plot natural log ratio statistics. |
| `enable_corrected_format` | `true` | `true`, `false` | Parse picks in LATTE corrected-residual format. |
| `outlier_threshold_sigma` | `3.0` | — | Residuals beyond N·σ are flagged as outliers. |
| `min_residual_threshold` | `-50.0` | s | Residuals below this are treated as unrealistic and excluded. |
| `max_residual_threshold` | `50.0` | s | Residuals above this are treated as unrealistic and excluded. |
| `histogram_bins` | `50` | — | Number of bins for residual histograms. |
| `scatter_alpha` | `0.6` | 0–1 | Scatter plot marker transparency. |
| `scatter_size` | `15` | — | Scatter plot marker size. |
| `generate_summary_report` | `true` | `true`, `false` | Write a text summary report. |
| `include_mismatch_log` | `true` | `true`, `false` | Log picks present in only one file (method comparison mode). |
| `include_unrealistic_log` | `true` | `true`, `false` | Log picks flagged by residual thresholds. |

---

### `velocity_analysis`

Compares two velocity models by computing absolute differences, percent changes, log ratios, and spatial projections.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `false` | `true`, `false` | Enable this module. |
| `comparison_mode` | `initial_vs_final` | `initial_vs_final`, `method_comparison`, `custom` | Controls plot title labelling. |
| `model1_csv` / `model2_csv` | — | — | Paths to velocity model CSVs. HDF5 takes precedence when `model1_hdf5`/`model2_hdf5` is set. |
| `model1_hdf5` / `model2_hdf5` | `null` | — | Paths to velocity model HDF5 files. `null` = use CSV. |
| `output_dir` | — | — | Output directory for plots and reports. |
| `initial_velocity_model_csv` / `initial_velocity_model_hdf5` | `null` | — | Optional reference model for three-way comparisons. HDF5 takes precedence. |
| `use_coarse_dataset` | `false` | `true`, `false` | Read the `velocity_coarse` HDF5 dataset for faster analysis. |
| `model1_name` / `model2_name` | `Initial Model` / `Final Model` | — | Display names used in `initial_vs_final` mode. |
| `custom_title1` / `custom_title2` | `Model A` / `Model B` | — | Display names used in `custom` mode. |
| `compute_log_ratios` | `true` | `true`, `false` | Compute and plot logarithmic velocity ratios. |
| `compute_geometric_mean_ratios` | `true` | `true`, `false` | Compute geometric mean Vp/Vs ratios. |
| `enable_basic_comparison_plots` | `true` | `true`, `false` | Scatter plots and histograms of velocity differences. |
| `enable_spatial_projections` | `true` | `true`, `false` | RMS difference projections on XY, XZ, YZ planes. |
| `enable_profile_analysis` | `true` | `true`, `false` | Vertical profiles at locations of maximum difference. |
| `enable_slice_analysis` | `true` | `true`, `false` | Horizontal slice heatmaps at specified depths. |
| `enable_plane_ratio_plots` | `true` | `true`, `false` | Per-plane ratio slice plots (configured via `plane_ratio_plots`). |
| `enable_domain_overview` | `true` | `true`, `false` | Spatial distribution overview of the model domain. |
| `plot_basin_boundaries` | `false` | `true`, `false` | Overlay basin boundaries on XY plane plots. |
| `interpolation_method` | `cubic` | `linear`, `cubic`, `nearest` | Interpolation method for gridded slice plots. |
| `grid_resolution` | `100` | — | Grid resolution for interpolated slices. |
| `contour_levels` | `20` | — | Number of contour levels in slice plots. |
| `velocity_colormap` | `viridis` | — | Colormap for velocity magnitude plots. |
| `difference_colormap` | `RdBu_r` | — | Colormap for difference plots. |
| `ratio_colormap` | `RdBu_r` | — | Colormap for ratio plots. |

#### `velocity_analysis.plane_ratio_plots`

Configures per-plane ratio slice outputs. Each entry in `planes` defines one slice group.

| Key | Options | Description |
|-----|---------|-------------|
| `plane_type` | `xy`, `xz`, `yz` | Plane orientation. |
| `coordinates` | list of floats (km) | Fixed coordinate values at which slices are taken (depths for `xy`, positions for `xz`/`yz`). Empty list = auto. |
| `label` | string | Display label for the plane group in plots. |

---

### `distance_analysis`

Analyzes the correlation between travel time residuals and source–receiver distance.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `false` | `true`, `false` | Enable this module. |
| `comparison_mode` | `ground_truth` | `ground_truth`, `method_comparison` | Data input mode. |
| `travel_time_csv` | `null` | — | Travel time CSV. Used in `ground_truth` mode. |
| `station_csv` | `null` | — | Station Cartesian coordinate CSV. Used in `ground_truth` mode. |
| `events_csv` | `null` | — | Events Cartesian coordinate CSV. Used in `ground_truth` mode. |
| `output_dir` | `null` | — | Output directory for plots and reports. |
| `method1_csv` / `method2_csv` | `null` | — | Travel time CSVs for each method. Used in `method_comparison` mode. |
| `method1_name` / `method2_name` | `Method 1` / `Method 2` | — | Display names for each method. |
| `coordinate_type` | `transformed` | `transformed`, `original` | Which coordinate columns to use for distance calculation. |
| `distance_types` | `["hypocentral"]` | `hypocentral`, `epicentral` | Distance metrics to compute. Can include both. |
| `enable_ratio_analysis` | `true` | `true`, `false` | Compute and plot logarithmic ratio vs. distance. |
| `enable_binned_analysis` | `true` | `true`, `false` | Compute RMS statistics in distance bins. |
| `enable_correlation_analysis` | `true` | `true`, `false` | Compute and plot residual–distance correlation. |
| `enable_log_plots` | `true` | `true`, `false` | Generate log-scale distance axis plots. |
| `distance_bins` | `20` | — | Number of bins for binned distance analysis. |
| `max_distance_plot` | `null` | km | Upper distance limit for plots. `null` = auto from data. |
| `min_distance_plot` | `0.0` | km | Lower distance limit for plots. |
| `correlation_method` | `pearson` | `pearson`, `spearman` | Correlation coefficient method. |
| `fit_polynomial_degree` | `1` | — | Degree of polynomial fit overlaid on scatter plots. |

---

### `short_distance_analysis`

Identical structure to `distance_analysis` but restricted to source–receiver pairs within a configurable distance cutoff.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `false` | `true`, `false` | Enable this module. |
| `comparison_mode` | `ground_truth` | `ground_truth`, `method_comparison` | Data input mode. Same inputs as `distance_analysis`. |
| `travel_time_csv` / `station_csv` / `events_csv` | `null` | — | Same as `distance_analysis`. Used in `ground_truth` mode. |
| `output_dir` | `null` | — | Output directory for plots and reports. |
| `method1_csv` / `method2_csv` | `null` | — | Travel time CSVs for each method. Used in `method_comparison` mode. |
| `distance_cutoff_km` | `20.0` | km | Only source–receiver pairs within this range are analysed. |
| `coordinate_type` | `transformed` | `transformed`, `original` | Coordinate columns used for distance calculation. |
| `distance_types` | `["hypocentral"]` | `hypocentral`, `epicentral` | Distance metrics to compute. |
| `phase_filter` | `null` | `null`, `P`, `S` | Phase filter override. `null` = use `analysis.phase_type`. |
| `min_residual_threshold` | `-5.0` | s | Residuals below this are excluded. |
| `max_residual_threshold` | `5.0` | s | Residuals above this are excluded. |
| `show_events` | `true` | `true`, `false` | Plot event locations on the map. |
| `show_stations` | `true` | `true`, `false` | Plot station locations on the map. |
| `show_travel_paths` | `true` | `true`, `false` | Draw source–receiver path lines on the map. |
| `enable_detailed_stats` | `true` | `true`, `false` | Compute per-event and per-station statistics. |
| `enable_residual_histograms` | `true` | `true`, `false` | Generate residual distribution histograms. |
| `enable_zoom_clusters` | `true` | `true`, `false` | Render per-cluster zoom map panels defined in `zoom_clusters`. |

#### `short_distance_analysis.zoom_clusters`

List of spatial cluster definitions for zoomed map panels. Only rendered when `enable_zoom_clusters: true`.

| Key | Unit | Description |
|-----|------|-------------|
| `name` | — | Cluster identifier used in output filenames. |
| `x_min`, `x_max`, `y_min`, `y_max` | km | Cartesian bounding box of the cluster. |
| `description` | — | Human-readable description included in reports. |

---

### `source_analysis`

Analyzes source parameter corrections (location shifts and origin time changes) produced by the inversion.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `false` | `true`, `false` | Enable this module. |
| `comparison_mode` | `ground_truth` | `ground_truth`, `method_comparison` | Data input mode. |
| `input_csv` | `null` | — | Source parameter CSV from the postprocess module. Used in `ground_truth` mode. |
| `output_dir` | `results/analysis/source_corrections` | — | Output directory for plots and reports. |
| `method1_csv` / `method2_csv` | `null` | — | Source CSVs for each method. Used in `method_comparison` mode. |
| `method1_name` / `method2_name` | `Method 1` / `Method 2` | — | Display names for each method. |
| `use_log_scale` | `false` | `true`, `false` | Use log scale on correction magnitude axes. |
| `enable_depth_analysis` | `true` | `true`, `false` | Generate correction vs. depth profiles. |
| `origin_time_metadata_dir` | `null` | — | Directory containing origin time metadata CSVs. `null` = skip temporal analysis. |
| `zero_initial_origin_time` | `true` | `true`, `false` | Treat the initial origin time as 0 s when computing corrections. |
| `depth_bins` | `10` | — | Number of depth bins for depth-profile analysis. |
| `min_events_per_bin` | `5` | — | Minimum events required per depth bin. |
| `histogram_bins` | `50` | — | Number of bins for correction magnitude histograms. |
| `max_arrows_map` | `5000` | — | Maximum correction vectors drawn on the map plot. |
| `generate_summary_report` | `true` | `true`, `false` | Write a text summary report. |
| `save_corrected_data` | `true` | `true`, `false` | Write `source_corrections_with_deltas.csv` with computed correction deltas. |
| `plot_basin_boundaries` | `false` | `true`, `false` | Overlay basin boundaries on the XY geographic map. |

---

### `dws_analysis`

Analyzes the Derivative Weight Sum (DWS) grid to assess model coverage and data sensitivity.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `false` | `true`, `false` | Enable this module. |
| `input_hdf5` | — | — | Path to the DWS HDF5 file produced by the postprocess module. |
| `output_dir` | derived | — | Output directory for plots and reports. |
| `enable_log_scale` | `true` | `true`, `false` | Produce both linear and log-scale plots. |
| `enable_coverage_maps` | `true` | `true`, `false` | Produce binary covered/uncovered coverage maps. |
| `enable_histograms` | `true` | `true`, `false` | Generate DWS value distribution histograms. |
| `enable_vertical_slices` | `true` | `true`, `false` | Generate vertical cross-section slice plots. |
| `use_m2_units` | `false` | `true`, `false` | Display DWS in m² instead of km² (1 km² = 1,000,000 m²). |
| `coverage_threshold` | `0.0` | — | Cells with DWS > threshold are considered covered in coverage maps. |
| `horizontal_slice_depths` | `[5,10,15,20,30,40,50]` | km | Depths at which horizontal slice plots are generated. |
| `generate_report` | `true` | `true`, `false` | Write a text coverage summary report. |
| `plot_basin_boundaries` | `false` | `true`, `false` | Overlay basin boundaries on horizontal slice and coverage map plots. |

#### `dws_analysis.vertical_slices`

| Key | Unit | Description |
|-----|------|-------------|
| `xz_positions` | km | List of constant Y values at which XZ vertical slices are plotted. |
| `yz_positions` | km | List of constant X values at which YZ vertical slices are plotted. |

#### `dws_analysis.plot_config`

| Key | Default | Description |
|-----|---------|-------------|
| `figure_width` / `figure_height` | `12` / `10` | Figure dimensions in inches. |
| `colormap` | `viridis` | Colormap for linear-scale plots. Options: `viridis`, `plasma`, `inferno`, `jet`. |
| `colormap_log` | `viridis` | Colormap for log-scale plots. |
| `equal_aspect` | `false` | Enforce equal axis aspect ratio. |
| `grid` | `true` | Show grid lines. |
| `depth_tolerance` | `0.01` | km | Tolerance for horizontal slice depth matching. |
| `position_tolerance` | `0.01` | km | Tolerance for vertical slice position matching. |

---

### `raypath_analysis`

Visualizes traced ray paths from the HDF5 cache as 2D projections onto the XY, XZ, and YZ planes.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `false` | `true`, `false` | Enable this module. |
| `input_cache` | derived | — | Path to the HDF5 ray path cache produced by the postprocess module. |
| `output_dir` | derived | — | Output directory for projection plots and reports. |
| `phase_type` | `both` | `both`, `p`, `s` | Phases to visualize. |
| `event_id_list_file` | `null` | — | Path to a text file with one event ID per line. `null` = no filter. |
| `shot_map_file` | `null` | — | Shot-to-event mapping file. Required when `event_id_list_file` is set. |
| `max_rays` | `null` | — | Maximum number of rays to load. `null` = all rays. |
| `subsample_fraction` | `1.0` | 0.0–1.0 | Fraction of loaded rays to plot. Applied after `max_rays`. |
| `filter_csv` | `null` | — | CSV for receiver-level filtering (e.g. anomalous picks export). `null` = disabled. |
| `filter_match_tolerance_km` | `0.5` | km | Receiver coordinate match tolerance for `filter_csv`. |
| `generate_report` | `true` | `true`, `false` | Write a ray path summary report. |
| `plot_basin_boundaries` | `false` | `true`, `false` | Overlay basin boundaries on the XY projection plot. |

#### `raypath_analysis.plot`

| Key | Default | Description |
|-----|---------|-------------|
| `planes` | `["xy","xz","yz"]` | Projection planes to generate. |
| `figure_width` / `figure_height` | `12` / `10` | Figure dimensions in inches. |
| `ray_color` | `blue` | Ray path line color. |
| `ray_alpha` | `0.3` | Ray path transparency (0 = transparent, 1 = opaque). |
| `ray_linewidth` | `0.5` | Ray path line width. |
| `source_color` | `red` | Source marker color. |
| `source_marker` | `*` | Source marker style. Options: `*`, `o`, `^`, `v`, `s`, `D`. |
| `source_size` | `100` | Source marker size. |
| `receiver_color` | `green` | Receiver marker color. |
| `receiver_marker` | `^` | Receiver marker style. Options: `*`, `o`, `^`, `v`, `s`, `D`. |
| `receiver_size` | `50` | Receiver marker size. |
| `grid` | `true` | Show grid lines. |
| `equal_aspect` | `true` | Enforce equal axis aspect ratio. |

---

### `ray_density_analysis`

Computes per-cell ray counts on the inversion grid and visualizes spatial coverage via projection maps and depth slices.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `false` | `true`, `false` | Enable this module. |
| `input_cache` | derived | — | Path to the HDF5 ray path cache. |
| `output_dir` | derived | — | Output directory for density maps and reports. |
| `phase_type` | `both` | `both`, `p`, `s` | Phases to include in density computation. |
| `threshold` | `0` | — | Cells with ray count below this are masked in plots. |
| `num_samples_per_ray` | `1000` | — | Number of sample points per ray used for grid intersection. |
| `n_jobs` | `-1` | — | Parallel workers: `-1` = all cores, `1` = serial, `N` = N cores. |
| `batch_size` | `1000` | — | Number of rays processed per parallel batch. |
| `event_id_list_file` | `null` | — | Path to event ID filter file. `null` = no filter. |
| `shot_map_file` | `null` | — | Shot map file. Required when `event_id_list_file` is set. |
| `max_rays` | `null` | — | Maximum rays to load. `null` = all. |
| `subsample_fraction` | `1.0` | 0.0–1.0 | Fraction of loaded rays to use. |
| `planes` | `["xy","xz","yz"]` | — | Projection planes to generate. |
| `enable_xy_threshold_count` | `true` | `true`, `false` | Map showing the count of depth layers with ray density ≥ threshold. |
| `enable_depth_slices` | `true` | `true`, `false` | Horizontal slice plots at specified depths. |
| `depth_slices` | `[5,10,15,20,30,40,50]` | km | Depths at which horizontal slice plots are generated. |
| `plot_basin_boundaries` | `false` | `true`, `false` | Overlay basin boundaries on XY projection and depth slice plots. |
| `figure_width` / `figure_height` | `12` / `10` | — | Figure dimensions in inches. |
| `colormap` | `hot` | `hot`, `viridis`, `plasma`, `inferno`, `jet` | Colormap for density plots. |

#### `ray_density_analysis.grid`

Cartesian grid definition used to bin ray paths. Must match the inversion grid exactly.

| Key | Unit | Description |
|-----|------|-------------|
| `nx`, `ny`, `nz` | — | Number of grid nodes along x, y, z axes. |
| `dx`, `dy`, `dz` | km | Grid spacing along each axis. |
| `x_min`, `y_min`, `z_min` | km | Cartesian origin of the grid. |

---

### `tradeoff_analysis`

Generates the inversion L-curve (data misfit vs. model variance) across iterations to identify the optimal regularization point.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `false` | `true`, `false` | Enable this module. |
| `y_axis_mode` | `data_misfit` | `data_misfit`, `rms` | `data_misfit`: read a pre-computed misfit file. `rms`: compute RMS from per-iteration residual CSVs. |
| `data_misfit_file_path` | `null` | — | Path to `data_misfit.txt` (3-column: iteration, data_misfit, normalized_data_misfit). Used when `y_axis_mode: data_misfit`. |
| `use_normalized_misfit` | `false` | `true`, `false` | Use the normalized misfit column instead of the absolute column. |
| `inversion_base_path` | `null` | — | Directory containing `iteration_*/` subdirs with travel time residual CSVs. Used when `y_axis_mode: rms`. |
| `traveltime_csv_pattern` | `synthetic_traveltimes.csv` | — | Glob pattern for travel time CSVs within each iteration directory. |
| `phase_mode` | `both` | `both`, `p`, `s`, `separate` | Phase filter for RMS computation. `separate` = plot Vp and Vs curves independently. |
| `min_residual_threshold` | `-50.0` | s | Lower bound for RMS computation. |
| `max_residual_threshold` | `50.0` | s | Upper bound for RMS computation. |
| `rms_comparison_mode` | `ground_truth` | `ground_truth`, `method_comparison` | `ground_truth`: single residual set. `method_comparison`: compare two methods. |
| `method1_csv` | `null` | — | Static travel time CSV for Method 1 (method comparison mode). |
| `method2_csv_pattern` | `synthetic_traveltimes.csv` | — | Glob pattern for Method 2 CSVs per iteration. |
| `initial_model_csv` / `initial_model_hdf5` | `null` | — | Path to the initial velocity model for variance computation. HDF5 takes precedence. |
| `inversion_analysis_dir` | `null` | — | Directory containing per-iteration post-processed velocity outputs. |
| `velocity_dataset` | `velocity` | `velocity`, `velocity_coarse` | HDF5 dataset name to read for velocity model. |
| `output_dir` | derived | — | Output directory for tradeoff curve plots. |

#### `tradeoff_analysis.model_variance`

Controls how model variance (X-axis of the L-curve) is computed.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `region` | `full` | `full`, `z_bounded`, `depth_layer` | Grid region over which variance is averaged. |
| `z_bounds` | `[null, null]` | km | `[z_min, z_max]` bounds. Used when `region: z_bounded`. |
| `depth_layers` | `[]` | km | Specific depth layers for stratified variance. Used when `region: depth_layer`. |
| `depth_tolerance_km` | `0.5` | km | Z-matching tolerance for depth layer mode. |
| `x_bounds` / `y_bounds` | `[null, null]` | km | Optional spatial clip applied before variance computation. `null` = no clip. |
| `use_percentage_change` | `false` | `true`, `false` | Express variance as mean(\|Δv\| / \|v_initial\|) × 100 (%). |
| `aggregation` | `mean` | `mean`, `max` | Aggregation method over selected grid points. |

#### `tradeoff_analysis.plot`

| Key | Default | Description |
|-----|---------|-------------|
| `figure_width` / `figure_height` | `10` / `8` | Figure dimensions in inches. |
| `show_grid` | `true` | Show grid lines. |
| `show_iteration_labels` | `true` | Annotate each data point with its iteration number. |
| `use_log_scale_x` / `use_log_scale_y` | `false` | Log scale on X (model variance) or Y (data misfit/RMS) axis. |
| `x_label` / `y_label` | `null` | Axis labels. `null` = auto-derived from mode and phase. |
| `marker_style` | `o` | Data point marker (any matplotlib marker). |
| `color_by_iteration` | `false` | Color each point by iteration number using `colormap`. |
| `colormap` | `viridis` | Colormap used when `color_by_iteration: true`. |
| `highlight_optimal` | `false` | Annotate the elbow/optimal point with a gold star. |
| `vp_color` / `vs_color` | `null` | Line colors for Vp/Vs curves. `null` = use global palette colors. |

---

### `performance`

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `parallel_processing` | `false` | `true`, `false` | Enable parallel processing. Currently used by `dws_analysis` only. |
| `n_jobs` | `1` | `-1`, `1`, `N` | Number of parallel workers. `1` = serial. |

---

### `quality_control`

| Key | Default | Description |
|-----|---------|-------------|
| `validate_input_data` | `true` | Abort if any required input file is missing before a module runs. |
| `check_coordinate_consistency` | `true` | Warn when coordinate columns differ between two input models. |

---

## Output Structure

Each module writes to its own `output_dir`. Within each directory the layout follows:

```
output_dir/
├── plots/
│   ├── <module>_<plot_type>.png
│   └── ...
├── reports/
│   ├── <module>_report.txt
│   └── <module>_summary.txt
└── statistics/
    └── <module>_stats.json
```

Hydra also writes a timestamped run log to:
```
outputs/<YYYY-MM-DD>/<HH-MM-SS>_analysis_<config_name>/
```

---

## Important Notes

- **All analysis modules are disabled by default.** Explicitly set `enabled: true` for each module to run.
- **`ray_density_analysis.grid` must match the inversion grid exactly.** Mismatched dimensions produce silently incorrect density counts.
- **`raypath_analysis` and `ray_density_analysis` both require the HDF5 ray path cache** produced by `ray_path_diagnosis` in the postprocess module. Run postprocessing before analysis.
- **`event_id_list_file` requires `shot_map_file`** in both `raypath_analysis` and `ray_density_analysis`. Omitting the shot map when an event filter is set will raise an error.
- **`method_comparison` mode requires both `method1_csv` and `method2_csv`** to be valid paths. Missing either raises a validation error at startup.
- **`tradeoff_analysis` in `rms` mode requires `inversion_base_path`** to point to a directory with `iteration_*/` subdirectories, each containing travel time residual CSVs matching `traveltime_csv_pattern`.
- **`use_coarse_dataset: true` in `velocity_analysis`** requires the HDF5 file to contain a `velocity_coarse` dataset, created by the postprocess module when `hdf5_options.create_coarse_dataset: true` is set.
- **`basin_boundaries` is optional.** Modules only draw boundaries when `plot_basin_boundaries: true` and `basin_boundaries.data_dir` contains readable CSV files.
- **`plane_ratio_plots.coordinates` accepts an empty list** (`[]`), which triggers automatic coordinate selection from the data range.
