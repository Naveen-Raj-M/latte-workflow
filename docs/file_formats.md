# LATTE Workflow — File Format Reference

This document describes all input and output file formats across the three pipeline stages: **Preprocess**, **Postprocess**, and **Analysis**. For each file, the format, structure, column/key definitions (with units), and whether it is required or optional are specified.

---

## Table of Contents

1. [Preprocess](#1-preprocess)
   - [Inputs](#11-inputs)
   - [Outputs](#12-outputs)
2. [Postprocess](#2-postprocess)
   - [Inputs](#21-inputs)
   - [Outputs](#22-outputs)
3. [Analysis](#3-analysis)
   - [Inputs](#31-inputs)
   - [Outputs](#32-outputs)
4. [Configuration Files](#4-configuration-files)
5. [File Format Summary](#5-file-format-summary)

---

## File Naming

The per-pick CSV tables and velocity models use the same names as the published data package (`analysis/geometry_and_travel_times/`):

| Table | Standard name | Legacy names (read-only fallback) |
|---|---|---|
| Ground truth travel times (preprocess) | `geometry_ground_truth_traveltimes.csv` | `ground_truth_traveltimes.csv`, `ground_truth_travel_times_corrected.csv`, `ground_truth_travel_times.csv` |
| Synthetic travel times (postprocess) | `synthetic_traveltimes.csv` | `synthetic_travel_times_latte_output.csv`, `travel_time_residuals*.csv` (iteration scans) |
| Source relocation (postprocess) | `source_relocation.csv` | `source_parameters_latte_output.csv`, `synthetic_travel_times_latte_output.csv`, `source_corrections*.csv` (iteration scans) |
| Initial velocity model (preprocess) | `velocity_model_init.{csv,h5}` (also `_coarse`, `_exact`, `_chapman` variants) | `velocity_model_cartesian`, `velocity_initial_cartesian`, `velocity_cartesian_initial`, `velocity_cartesian` |
| Final velocity model (postprocess, spike/checkerboard inverted model) | `velocity_model_final.{csv,h5}` (also `_coarse`) | `velocity_cartesian_latte_output`, `velocity_cartesian_updated`, `velocity_model_inverted` |

New files are always written with the standard names. When a configured input is missing, the same directory is searched for the standard name and then the legacy names; a warning is logged whenever a legacy file is used.

---

## 1. Preprocess

### 1.1 Inputs

---

#### QuakeML Event Catalog
| Property | Value |
|---|---|
| **Required** | Yes (if `use_hdf5: false`) |
| **Format** | XML (ObsPy QuakeML) |
| **Extension** | `.qml` |
| **Config key** | `data.quakeml_dir` |

Seismic event catalog read via `obspy.read_events()`. The directory is scanned recursively, including year subdirectories. Contains events, origins, picks, arrivals, and magnitudes.

---

#### Seismic Data — HDF5
| Property | Value |
|---|---|
| **Required** | Yes (if `use_hdf5: true`) |
| **Format** | HDF5 |
| **Extension** | `.h5` / `.hdf5` |
| **Config key** | `data.hdf5_file` |

Read via `pd.read_hdf()`. The file must contain the following datasets:

**Dataset: `events`**

| Key | Description | Units |
|---|---|---|
| `event_id` | Unique event identifier | — |
| `origin_time` | Event origin time | UTC datetime string |
| `latitude` | Event latitude | degrees |
| `longitude` | Event longitude | degrees |
| `depth` | Hypocenter depth (positive down) | km |
| `magnitude` | Event magnitude | — |

**Dataset: `stations`**

| Key | Description | Units |
|---|---|---|
| `station_id` | Unique station identifier | — |
| `latitude` | Station latitude | degrees |
| `longitude` | Station longitude | degrees |
| `depth` | Station depth (negative = above MSL) | km |

**Dataset: `arrivals`**

| Key | Description | Units |
|---|---|---|
| `arrival_id` | Unique arrival identifier | — |
| `event_id` | Associated event identifier | — |
| `pick_id` | Associated pick identifier | — |
| `station_id` | Associated station identifier | — |
| `phase` | Phase label (P or S) | — |
| `azimuth` | Source-to-receiver azimuth | degrees |
| `distance` | Source-receiver distance | km |
| `time_residual` | Observed minus predicted travel time | s |
| `time_weight` | Arrival weight used in inversion | — |
| `epicentral_distance` | Epicentral distance | km |
| `hypocentral_distance` | Hypocentral (slant) distance | km |

**Dataset: `picks`** *(optional)*

| Key | Description | Units |
|---|---|---|
| `pick_id` | Unique pick identifier | — |
| `event_id` | Associated event identifier | — |
| `station_id` | Associated station identifier | — |
| `pick_time` | Absolute pick time | UTC datetime string |
| `phase_hint` | Phase label | — |
| `evaluation_mode` | Manual or automatic | — |

---

#### Station Inventory
| Property | Value |
|---|---|
| **Required** | Yes |
| **Format** | Plain text, whitespace-delimited |
| **Extension** | `.txt` |
| **Config key** | `data.station_file` |

Lines beginning with `#` are treated as comments and skipped.

| Column | Description | Units |
|---|---|---|
| `ID` | Station identifier | — |
| `latitude` | Station latitude | degrees |
| `longitude` | Station longitude | degrees |
| `elevation_meters` | Station elevation | m |

---

#### Initial 1D Velocity Model
| Property | Value |
|---|---|
| **Required** | Yes |
| **Format** | CSV |
| **Extension** | `.csv` |
| **Config key** | `initial_velocity_model.initial_model_file` |

Supports two column layouts:

**3-column layout:**

| Column | Description | Units |
|---|---|---|
| `depth` | Depth (positive down) | km |
| `vp` | P-wave velocity | km/s |
| `vs` | S-wave velocity | km/s |

**4-column layout:**

| Column | Description | Units |
|---|---|---|
| `depth` | Depth (positive down) | km |
| `radius` | Earth radius at layer | km |
| `vp` | P-wave velocity | km/s |
| `vs` | S-wave velocity | km/s |

---

#### Basin Boundary Files
| Property | Value |
|---|---|
| **Required** | No |
| **Format** | CSV |
| **Extension** | `.csv` |
| **Config key** | `coordinate_transform.basin_boundaries.data_dir` |

One file per basin. Used as overlays on visualization plots.

| Column | Description | Units |
|---|---|---|
| `latitude` | Vertex latitude | degrees |
| `longitude` | Vertex longitude | degrees |

---

### 1.2 Outputs

---

#### Geodetic Events
| Property | Value |
|---|---|
| **Format** | CSV |
| **Config key** | `coordinate_transform.events_geodetic_csv` |

| Column | Description | Units |
|---|---|---|
| `event_id` | Unique event identifier | — |
| `origin_time` | Event origin time | UTC datetime string |
| `latitude` | Event latitude | degrees |
| `longitude` | Event longitude | degrees |
| `depth` | Hypocenter depth (positive below MSL) | km |
| `magnitude` | Event magnitude | — |
| `magnitude_type` | Magnitude scale | — |
| `num_picks` | Number of picks | — |
| `num_arrivals` | Number of arrivals used | — |

---

#### Geodetic Stations
| Property | Value |
|---|---|
| **Format** | CSV |
| **Config key** | `coordinate_transform.stations_geodetic_csv` |

| Column | Description | Units |
|---|---|---|
| `station_id` | Unique station identifier | — |
| `latitude` | Station latitude | degrees |
| `longitude` | Station longitude | degrees |
| `depth` | Depth (negative = above MSL) | km |

---

#### Cartesian Events
| Property | Value |
|---|---|
| **Format** | CSV |
| **Config key** | `coordinate_transform.events_cartesian_csv` |

| Column | Description | Units |
|---|---|---|
| `event_id` | Unique event identifier | — |
| `origin_time` | Event origin time | UTC datetime string |
| `x` | Easting (Cartesian) | km |
| `y` | Northing (Cartesian) | km |
| `z` | Depth (Cartesian, positive down) | km |

---

#### Cartesian Stations
| Property | Value |
|---|---|
| **Format** | CSV |
| **Config key** | `coordinate_transform.stations_cartesian_csv` |

| Column | Description | Units |
|---|---|---|
| `station_id` | Unique station identifier | — |
| `x` | Easting (Cartesian) | km |
| `y` | Northing (Cartesian) | km |
| `z` | Depth (Cartesian) | km |

---

#### Cartesian Picks
| Property | Value |
|---|---|
| **Format** | CSV |
| **Config key** | `coordinate_transform.picks_cartesian_csv` |

| Column | Description | Units |
|---|---|---|
| `pick_id` | Unique pick identifier | — |
| `event_id` | Associated event identifier | — |
| `station_id` | Associated station identifier | — |
| `pick_time` | Absolute pick time | UTC datetime string |
| `phase_hint` | Phase label | — |

---

#### Ground Truth Travel Times
| Property | Value |
|---|---|
| **Format** | CSV |
| **Config key** | `geometry_traveltime.csv_output_dir` (filename: `geometry_ground_truth_traveltimes.csv`) |
| **Condition** | Written only if `generate_travel_time_csv: true` |

| Column | Description | Units |
|---|---|---|
| `event_id` | Event identifier | — |
| `station_id` | Station identifier | — |
| `phase` | Phase label (P or S) | — |
| `travel_time` | Observed travel time | s |
| `distance` | Source-receiver distance | km |
| `azimuth` | Source-to-receiver azimuth | degrees |
| `weight` | Arrival weight | — |
| `epicentral_distance` | Epicentral distance | km |
| `hypocentral_distance` | Hypocentral distance | km |
| `magnitude` | Event magnitude *(optional)* | — |

---

#### Initial Velocity Model — CSV
| Property | Value |
|---|---|
| **Format** | CSV |
| **Config key** | `initial_velocity_model.output_csv` |
| **Condition** | Written when `output_format` is `csv` or `both` |

One row per grid point (nx × ny × nz total rows).

| Column | Description | Units |
|---|---|---|
| `X_grid_km` | Grid x-coordinate (origin-relative) | km |
| `Y_grid_km` | Grid y-coordinate (origin-relative) | km |
| `Z_grid_km` | Grid z-coordinate (origin-relative) | km |
| `x` | Absolute Cartesian x | km |
| `y` | Absolute Cartesian y | km |
| `z` | Absolute Cartesian z | km |
| `i_raw` | Raw grid index along x | — |
| `j_raw` | Raw grid index along y | — |
| `k_raw` | Raw grid index along z | — |
| `i` | Grid index along x | — |
| `j` | Grid index along y | — |
| `k` | Grid index along z | — |
| `vp` | P-wave velocity | km/s |
| `vs` | S-wave velocity | km/s |
| `lon` | Geographic longitude | degrees |
| `lat` | Geographic latitude | degrees |
| `depth` | Depth (positive down) | km |

---

#### Initial Velocity Model — HDF5
| Property | Value |
|---|---|
| **Format** | HDF5 |
| **Config key** | `initial_velocity_model.output_hdf5` |
| **Condition** | Written when `output_format` is `hdf5` or `both` |

| Dataset | Shape | Description | Units |
|---|---|---|---|
| `velocity` | `(n_points, 17)` | All grid point data (same columns as velocity CSV) | km / km/s |
| `velocity_coarse` *(optional)* | `(n_coarse, 17)` | Subsampled grid for fast analysis | km / km/s |

File-level attributes include: `columns`, `nx`, `ny`, `nz`, `dx`, `dy`, `dz`, `origin_x`, `origin_y`, `origin_z`. Datasets are stored with gzip compression (level configurable).

---

#### Initial Velocity Model — Binary
| Property | Value |
|---|---|
| **Format** | Raw binary (float32) |
| **Config keys** | `initial_velocity_model.vp_binary`, `initial_velocity_model.vs_binary` |

One binary file each for Vp and Vs. Contains a 1D array of velocity values in Fortran-style ordering (x varies fastest, z varies slowest). Read back with `np.fromfile(dtype=np.float32)` and reshape to `(nx, ny, nz)`.

---

#### Geometry Files
| Property | Value |
|---|---|
| **Format** | Text (Fortran-format) |
| **Extension** | `.txt` |
| **Config key** | `geometry_traveltime.geometry_dir` |
| **Naming** | `geometry_<source_id>.txt` |

One file per source event. Contains event-receiver geometry used as input to the ray tracer.

---

#### Travel Time Files
| Property | Value |
|---|---|
| **Format** | Text / binary |
| **Extension** | `.txt` |
| **Config key** | `geometry_traveltime.traveltime_dir` |
| **Naming** | `traveltime_<source_id>.txt` |

Synthetic travel times between each source and its receivers, computed during preprocessing.

---

#### Shot Binary Files
| Property | Value |
|---|---|
| **Format** | Raw binary (Fortran stream) |
| **Extension** | `.bin` |
| **Config key** | `geometry_traveltime.source_dir` |
| **Naming** | `shot_<id>.bin` |

One file per event. Stores receiver geometry in a binary format compatible with the inversion kernel. Written via `np.ndarray.tofile()` and `struct.pack()`.

---

#### Basin Boundaries — Cartesian *(optional)*
| Property | Value |
|---|---|
| **Format** | CSV |
| **Config key** | `coordinate_transform.output_dir/basin_boundaries/<label>.csv` |

| Column | Description | Units |
|---|---|---|
| `x` | Easting (Cartesian) | km |
| `y` | Northing (Cartesian) | km |

---

#### Visualization Plots
| Property | Value |
|---|---|
| **Format** | PNG (default; configurable) |
| **Config key** | `visualization.*_vis.output_dir` |

| File | Description |
|---|---|
| `station_event_map_geodetic.png` | Geographic map of stations and events |
| `station_event_cross_section_xz.png` | Vertical cross-section (XZ plane) |
| `station_event_cross_section_yz.png` | Vertical cross-section (YZ plane) |
| `station_event_map_cartesian.png` | Cartesian map of stations and events |
| `velocity_depth_<depth>km_vp.png` | Vp depth slice at specified depth |
| `velocity_depth_<depth>km_vs.png` | Vs depth slice at specified depth |
| `velocity_profile_lat.png` | Velocity profile along latitude |
| `velocity_profile_lon.png` | Velocity profile along longitude |
| `ray_density_xy_projection.png` | Ray density map view |
| `ray_density_xz_projection.png` | Ray density XZ slice |
| `ray_density_yz_projection.png` | Ray density YZ slice |
| `ray_path_density_report.txt` | Ray density summary statistics |

---

## 2. Postprocess

### 2.1 Inputs

---

#### Updated Velocity Binaries
| Property | Value |
|---|---|
| **Required** | Yes |
| **Format** | Raw binary (float32) |
| **Extension** | `.bin` |
| **Config keys** | `velocity_processing.vp_binary`, `velocity_processing.vs_binary` |

Output from the inversion kernel. Same memory layout as the preprocess velocity binaries: 1D Fortran-ordered array, shape `(nx × ny × nz,)`, dtype `float32`.

---

#### Ground Truth Travel Times
| Property | Value |
|---|---|
| **Required** | Yes |
| **Format** | CSV |
| **Config key** | `travel_time_processing.ground_truth_csv` |

Same schema as the preprocess output `geometry_ground_truth_traveltimes.csv` (see [§1.2](#ground-truth-travel-times)).

---

#### Synthetic Travel Time Binaries
| Property | Value |
|---|---|
| **Required** | Yes |
| **Format** | Raw binary (float32) |
| **Extension** | `.bin` |
| **Config key** | `travel_time_processing.synthetic_tt_dir` |
| **Naming** | `tt_<shot_id>.bin` |

One file per shot. Contains computed synthetic travel times from the inversion kernel.

---

#### Shot Map File
| Property | Value |
|---|---|
| **Required** | Yes |
| **Format** | Plain text, tab-separated |
| **Extension** | `.txt` |
| **Config key** | `travel_time_processing.shot_map_file` |

| Column | Description | Units |
|---|---|---|
| `Shot_ID` | Integer shot index used by the kernel | — |
| `Original_Event_ID` | String event identifier from catalog | — |

---

#### Source Parameter Binaries
| Property | Value |
|---|---|
| **Required** | Yes |
| **Format** | Raw binary (float32) |
| **Extension** | `.bin` |
| **Config keys** | `source_processing.sx_file`, `sy_file`, `sz_file`, `st0_file` |

One value per source (shot). Arrays of length `n_sources`.

| File | Description | Units |
|---|---|---|
| `sx_file` | Source x-coordinate (Cartesian) | km |
| `sy_file` | Source y-coordinate (Cartesian) | km |
| `sz_file` | Source z-coordinate (Cartesian) | km |
| `st0_file` | Source origin time correction | s |

---

#### Source Correction File *(optional)*
| Property | Value |
|---|---|
| **Required** | No |
| **Format** | Raw binary (float32) |
| **Config key** | `source_processing.source_correction_file` |

Corrected origin times (one float32 per source) to be applied on top of the baseline `st0`.

---

#### Gradient / Search Direction Binaries *(optional)*
| Property | Value |
|---|---|
| **Required** | No |
| **Format** | Raw binary (float32) |
| **Extension** | `.bin` |
| **Config keys** | `velocity_processing.grad_vp`, `grad_vs`, `srch_vp`, `srch_vs`; `source_processing.grad_sx`, etc. |

| File pattern | Description |
|---|---|
| `grad_vp.bin`, `grad_vs.bin` | Velocity model gradient |
| `srch_vp.bin`, `srch_vs.bin` | Velocity search direction |
| `grad_sx.bin`, `grad_sy.bin`, `grad_sz.bin`, `grad_st0.bin` | Source parameter gradient |

---

#### Ray Path Binary Files
| Property | Value |
|---|---|
| **Required** | Yes (for ray-path / DWS steps) |
| **Format** | Raw binary (Fortran stream) |
| **Extension** | `.bin` |
| **Config key** | `ray_path_diagnosis.raypath_binary_dir` |
| **Naming** | `shot_<shot_id>_raypath_<phase>.bin` |

Binary layout per file:

| Field | Type | Description |
|---|---|---|
| `nrays` | `int32` | Number of rays in this shot |
| `ray_npoints[nrays]` | `int32[nrays]` | Number of points per ray |
| `x, y, z` coordinates | `float32` | Ray point coordinates (in **meters**) |

---

### 2.2 Outputs

---

#### Updated Velocity Model — CSV
| Property | Value |
|---|---|
| **Format** | CSV |
| **Config key** | `velocity_processing.velocity_output_csv` |

Same 17-column schema as the preprocess [Initial Velocity Model — CSV](#initial-velocity-model--csv).

---

#### Updated Velocity Model — HDF5
| Property | Value |
|---|---|
| **Format** | HDF5 |
| **Config key** | `velocity_processing.velocity_output_hdf5` |

Same dataset structure and attributes as the preprocess [Initial Velocity Model — HDF5](#initial-velocity-model--hdf5).

---

#### Synthetic Travel Times — CSV
| Property | Value |
|---|---|
| **Format** | CSV |
| **Config key** | `travel_time_processing.synthetic_tt_output_csv` |

| Column | Description | Units |
|---|---|---|
| `event_id` | Event identifier | — |
| `station_id` | Station identifier | — |
| `phase` | Phase label (P or S) | — |
| `shot_id` | Integer shot index | — |
| `receiver_id` | Integer receiver index | — |
| `synthetic_travel_time` | Computed synthetic travel time | s |

---

#### Source Parameters — CSV
| Property | Value |
|---|---|
| **Format** | CSV |
| **Config key** | `source_processing.output_csv` |

| Column | Description | Units |
|---|---|---|
| `event_id` | Event identifier | — |
| `sx` | Source x-coordinate (Cartesian) | km |
| `sy` | Source y-coordinate (Cartesian) | km |
| `sz` | Source z-coordinate (Cartesian) | km |
| `st0` | Source origin time correction | s |
| `grad_sx`, `grad_sy`, `grad_sz`, `grad_st0` | Source gradients *(optional)* | km⁻¹ / s⁻¹ |
| `srch_sx`, `srch_sy`, `srch_sz`, `srch_st0` | Search directions *(optional)* | — |

---

#### Ray Path Cache — HDF5
| Property | Value |
|---|---|
| **Format** | HDF5 |
| **Config key** | `ray_path_diagnosis.ray_path_output_file` |

Group hierarchy: `/p_wave/shots/<shot_id>/` and `/s_wave/shots/<shot_id>/`

| Dataset | Shape | Description | Units |
|---|---|---|---|
| `source` | `(1, 3)` | Source (x, y, z) coordinates | km |
| `receivers` | `(n_recv, 3)` | Receiver (x, y, z) coordinates | km |
| `weights` | `(n_recv,)` | Per-ray weights | — |
| `ray_lengths` | `(n_recv,)` | Number of points per ray | — |
| `ray_data` | `(total_pts, 3)` | Concatenated ray (x, y, z) coordinates | km |

Stored with gzip compression (level configurable via `compression_level`).

---

#### Derivative Weight Sum (DWS) — HDF5
| Property | Value |
|---|---|
| **Format** | HDF5 |
| **Config key** | `ray_path_diagnosis.dws_output_file` |

| Dataset | Shape | Description | Units |
|---|---|---|---|
| `dws_vp` | `(nx, ny, nz)` | DWS for P-wave model | km² |
| `dws_vs` | `(nx, ny, nz)` | DWS for S-wave model | km² |
| `grid_info` | scalar | Grid dimensions and spacing metadata | — |

---

#### Simplified Ray Path Cache — HDF5 *(optional)*
| Property | Value |
|---|---|
| **Required** | No |
| **Format** | HDF5 |
| **Config key** | `ray_path_simplification.output_cache_path` |

Identical structure to the [Ray Path Cache — HDF5](#ray-path-cache--hdf5). Ray paths are simplified using the Ramer–Douglas–Peucker algorithm; tolerance is controlled by `ray_path_simplification.epsilon` (km).

---

## 3. Analysis

### 3.1 Inputs

---

#### Travel Time CSVs
| Property | Value |
|---|---|
| **Required** | Yes (one or two depending on mode) |
| **Format** | CSV |
| **Config keys** | `travel_time_analysis.input_csv` (ground_truth mode); `method1_csv`, `method2_csv` (comparison mode) |

Same schema as [Ground Truth Travel Times](#ground-truth-travel-times) or [Synthetic Travel Times — CSV](#synthetic-travel-times--csv) depending on analysis mode.

---

#### Velocity Model Files
| Property | Value |
|---|---|
| **Required** | Yes (one or two depending on mode) |
| **Format** | CSV or HDF5 |
| **Config keys** | `velocity_analysis.model1_csv`, `velocity_analysis.model2_csv` (or HDF5 equivalents) |

Same schema as [Initial Velocity Model — CSV](#initial-velocity-model--csv) or [HDF5](#initial-velocity-model--hdf5).

---

#### Events — Cartesian CSV
| Property | Value |
|---|---|
| **Required** | Yes (for distance analysis) |
| **Format** | CSV |
| **Config key** | `distance_analysis.events_csv` |

Same schema as [Cartesian Events](#cartesian-events).

---

#### Stations — Cartesian CSV
| Property | Value |
|---|---|
| **Required** | Yes (for distance analysis) |
| **Format** | CSV |
| **Config key** | `distance_analysis.station_csv` |

Same schema as [Cartesian Stations](#cartesian-stations).

---

#### Source Parameters CSV
| Property | Value |
|---|---|
| **Required** | Yes (for source analysis) |
| **Format** | CSV |
| **Config key** | `source_analysis.input_csv` |

Same schema as the postprocess [Source Parameters — CSV](#source-parameters--csv), optionally including `delta_sx`, `delta_sy`, `delta_sz`, `delta_st0` correction columns.

---

#### DWS — HDF5
| Property | Value |
|---|---|
| **Required** | Yes (for DWS analysis) |
| **Format** | HDF5 |
| **Config key** | `dws_analysis.input_hdf5` |

Same structure as the postprocess [Derivative Weight Sum — HDF5](#derivative-weight-sum-dws--hdf5). Units are km² by default; set `use_m2_units: true` to interpret as m².

---

#### Ray Path Cache — HDF5
| Property | Value |
|---|---|
| **Required** | Yes (for ray path / ray density analysis) |
| **Format** | HDF5 |
| **Config key** | `raypath_analysis.input_cache` |

Same structure as the postprocess [Ray Path Cache — HDF5](#ray-path-cache--hdf5).

---

#### Data Misfit File
| Property | Value |
|---|---|
| **Required** | Yes (for tradeoff analysis) |
| **Format** | Plain text, tab-separated |
| **Extension** | `.txt` |
| **Config key** | `tradeoff_analysis.data_misfit_file_path` |

| Column | Description | Units |
|---|---|---|
| `iteration` | Inversion iteration index | — |
| `data_misfit` | Absolute data misfit | s (or dimensionless) |
| `normalized_data_misfit` | Data misfit normalized by initial value | — |

---

#### Basin Boundary Files *(optional)*
| Property | Value |
|---|---|
| **Required** | No |
| **Format** | CSV |
| **Config key** | `basin_boundaries.data_dir` |

Same schema as the preprocess [Basin Boundaries — Cartesian](#basin-boundaries--cartesian-optional) (x, y in km).

---

#### Gradient / Search Direction Files *(optional)*
| Property | Value |
|---|---|
| **Required** | No |
| **Format** | CSV or HDF5 |
| **Config keys** | `gradient_analysis.velocity_input_csv`, `gradient_analysis.source_input_csv` |

Velocity gradient file shares the velocity model CSV schema with additional columns `grad_vp` and `grad_vs`. Source gradient file shares the source parameters CSV schema with `grad_sx`, `grad_sy`, `grad_sz`, `grad_st0` columns.

---

### 3.2 Outputs

All analysis outputs are plots and text reports. No new data files are produced.

---

#### Travel Time Analysis
| Output | Path | Format |
|---|---|---|
| Residual histogram | `travel_time_analysis.output_dir/travel_time_residuals_histogram.png` | PNG |
| Residual scatter | `travel_time_analysis.output_dir/travel_time_residuals_scatter.png` | PNG |
| Residuals by phase | `travel_time_analysis.output_dir/travel_time_residuals_by_phase.png` | PNG |
| Method comparison | `travel_time_analysis.output_dir/travel_time_comparison_method1_vs_method2.png` | PNG |
| Summary report | `travel_time_analysis.output_dir/travel_time_analysis_report.txt` | Plain text |

---

#### Velocity Analysis
| Output | Path | Format |
|---|---|---|
| Comparison (basic) | `velocity_analysis.output_dir/velocity_comparison_basic.png` | PNG |
| Spatial projections | `velocity_analysis.output_dir/velocity_spatial_projection_xy/xz/yz.png` | PNG |
| Profile analysis | `velocity_analysis.output_dir/velocity_profile_analysis.png` | PNG |
| Slice heatmap | `velocity_analysis.output_dir/velocity_slice_heatmap.png` | PNG |
| Velocity ratio | `velocity_analysis.output_dir/velocity_ratio_analysis.png` | PNG |
| Domain overview | `velocity_analysis.output_dir/velocity_domain_overview.png` | PNG |
| Summary report | `velocity_analysis.output_dir/velocity_comparison_report.txt` | Plain text |

---

#### Distance Analysis
| Output | Path | Format |
|---|---|---|
| Residual vs distance | `distance_analysis.output_dir/residual_vs_distance_scatter.png` | PNG |
| Binned residuals | `distance_analysis.output_dir/residual_vs_distance_binned.png` | PNG |
| Correlation plot | `distance_analysis.output_dir/residual_vs_distance_correlation.png` | PNG |
| Ratio analysis | `distance_analysis.output_dir/residual_distance_ratio.png` | PNG |
| Summary report | `distance_analysis.output_dir/distance_analysis_report.txt` | Plain text |

---

#### Short-Distance Analysis
| Output | Path | Format |
|---|---|---|
| Short-distance map | `short_distance_analysis.output_dir/short_distance_map.png` | PNG |
| Residual histogram | `short_distance_analysis.output_dir/short_distance_residual_histogram.png` | PNG |
| Cluster zoom plots | `short_distance_analysis.output_dir/short_distance_zoom_clusters.png` | PNG |
| Coverage plot | `short_distance_analysis.output_dir/short_distance_event_station_coverage.png` | PNG |
| Summary report | `short_distance_analysis.output_dir/short_distance_analysis_report.txt` | Plain text |

---

#### Source Analysis
| Output | Path | Format |
|---|---|---|
| Source correction map | `source_analysis.output_dir/source_corrections_map.png` | PNG |
| Correction histograms | `source_analysis.output_dir/source_corrections_histograms.png` | PNG |
| Depth analysis | `source_analysis.output_dir/source_corrections_depth_analysis.png` | PNG |
| Spatial distribution | `source_analysis.output_dir/source_corrections_spatial_distribution.png` | PNG |
| Summary report | `source_analysis.output_dir/source_correction_analysis_report.txt` | Plain text |
| Corrected source CSV *(optional)* | `source_analysis.output_dir/source_corrections_with_deltas.csv` | CSV |

The optional CSV has the same schema as the input source parameters file plus `depth_bin` and `zone` grouping columns.

---

#### Gradient Analysis
| Output | Path | Format |
|---|---|---|
| Gradient histograms | `gradient_analysis.velocity_output_dir/gradient_histograms.png` | PNG |
| Alignment scatter | `gradient_analysis.velocity_output_dir/gradient_alignment_scatter.png` | PNG |
| Spatial distribution | `gradient_analysis.velocity_output_dir/gradient_spatial_distribution.png` | PNG |
| Depth profiles | `gradient_analysis.velocity_output_dir/gradient_depth_profiles.png` | PNG |
| Summary report | `gradient_analysis.velocity_output_dir/gradient_analysis_report.txt` | Plain text |

---

#### DWS Analysis
| Output | Path | Format |
|---|---|---|
| Horizontal depth slices | `dws_analysis.output_dir/dws_horizontal_depth_<depth>km_linear/log.png` | PNG |
| Vertical XZ slices | `dws_analysis.output_dir/dws_vertical_xz_position_<y>km.png` | PNG |
| Vertical YZ slices | `dws_analysis.output_dir/dws_vertical_yz_position_<x>km.png` | PNG |
| Histogram (linear/log) | `dws_analysis.output_dir/dws_histogram_linear/log.png` | PNG |
| Coverage map | `dws_analysis.output_dir/dws_coverage_map.png` | PNG |
| Summary report | `dws_analysis.output_dir/dws_analysis_report.txt` | Plain text |

---

#### Ray Path Analysis
| Output | Path | Format |
|---|---|---|
| XY projection | `raypath_analysis.output_dir/ray_paths_xy_projection.png` | PNG |
| XZ projection | `raypath_analysis.output_dir/ray_paths_xz_projection.png` | PNG |
| YZ projection | `raypath_analysis.output_dir/ray_paths_yz_projection.png` | PNG |
| Summary report | `raypath_analysis.output_dir/ray_path_analysis_report.txt` | Plain text |

---

#### Ray Density Analysis
| Output | Path | Format |
|---|---|---|
| XY projection | `ray_density_analysis.output_dir/ray_density_xy_projection.png` | PNG |
| XZ projection | `ray_density_analysis.output_dir/ray_density_xz_projection.png` | PNG |
| YZ projection | `ray_density_analysis.output_dir/ray_density_yz_projection.png` | PNG |
| Depth layer count | `ray_density_analysis.output_dir/ray_density_xy_threshold_count.png` | PNG |
| Horizontal depth slices | `ray_density_analysis.output_dir/ray_density_horizontal_depth_<depth>km.png` | PNG |
| Summary report | `ray_density_analysis.output_dir/ray_density_analysis_report.txt` | Plain text |

---

#### Tradeoff Analysis
| Output | Path | Format |
|---|---|---|
| Tradeoff curve | `tradeoff_analysis.output_dir/tradeoff_curve.png` | PNG |
| Tradeoff with iterations | `tradeoff_analysis.output_dir/tradeoff_curve_with_iterations.png` | PNG |
| Summary report | `tradeoff_analysis.output_dir/tradeoff_analysis_report.txt` | Plain text |

---

## 4. Configuration Files

All pipelines use [Hydra](https://hydra.cc/) for configuration management.

| Pipeline | Default config path |
|---|---|
| Preprocess | `configs/preprocess/default.yaml` |
| Postprocess | `configs/postprocess/default.yaml` |
| Analysis | `configs/analysis/default.yaml` |

Each YAML file is structured into named sections corresponding to pipeline sub-modules (e.g., `data`, `coordinate_transform`, `geometry_traveltime`, `velocity_processing`, `travel_time_analysis`). All file paths in these configs accept absolute or relative paths; relative paths are resolved from the working directory at runtime.

---

## 5. File Format Summary

| Format | Extensions | Pipelines | Role |
|---|---|---|---|
| CSV | `.csv` | All | Tabular event, station, velocity, travel time, and source data |
| HDF5 | `.h5`, `.hdf5` | All | Seismic catalog input, velocity models, ray path cache, DWS |
| Binary (float32) | `.bin` | Preprocess, Postprocess | Velocity fields, source parameters, synthetic travel times, ray paths |
| Plain text | `.txt` | All | Station inventory, shot maps, geometry/traveltime files, reports |
| QuakeML (XML) | `.qml` | Preprocess | ObsPy seismic event catalog |
| YAML | `.yaml` | All | Hydra pipeline configuration |
| PNG / PDF / SVG | `.png`, `.pdf`, `.svg` | All | Visualization outputs (format configurable) |
