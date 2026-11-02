# LATTE Workflow — File Format Reference

This document describes all input and output file formats of the workflow: **Preprocess**, **Postprocess**, **Analysis**, and the **resolution tests**. For each file it gives the format, the structure, column or key definitions with units, and whether the file is required or optional.

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
4. [Resolution Tests](#4-resolution-tests)
5. [Configuration Files](#5-configuration-files)
6. [File Format Summary](#6-file-format-summary)

---

## Conventions

**Binary files.** All `.bin` files exchanged with LATTE are headerless, little-endian **float32** arrays (ray path files additionally contain `int32` counts). LATTE reads and writes single precision only. Read them with `np.fromfile(path, dtype=np.float32)`.

**Model binaries.** A velocity model is a 1D array in Fortran order with shape `(nz, ny, nx)`: z varies fastest, x slowest. Reshape with `array.reshape((nz, ny, nx), order="F")`.

**Units.**

| Quantity | Unit |
|---|---|
| Geometry files, `sx/sy/sz.bin`, `exchange_geometry.txt`, Cartesian coordinates in the travel time CSVs | m |
| Velocity binaries (`vp_init.bin`, `updated_vp.bin`, …) | m/s |
| Velocity model CSV/HDF5 files (coordinates, grid spacing, velocities) | km, km/s |
| Cartesian event/station CSVs of preprocess, ray path cache, DWS grid | km |
| Times (travel times, origin times) | s |
| Angles | degrees |

**Cartesian frame.** x east, z down; y points south when `coordinate_transform.invert_y_axis` is true. The origin is set in the preprocess configuration.

**CSV metadata.** Travel time tables and velocity-format CSV files start with `# key: value` comment lines (values are JSON). Readers skip them (`pandas.read_csv(..., comment="#")`). In HDF5 files the same keys are dataset attributes.

**Standard file names.**

| Table | Name |
|---|---|
| Ground truth travel times (preprocess) | `geometry_ground_truth_traveltimes.csv` |
| Synthetic travel times (postprocess) | `synthetic_traveltimes.csv` |
| Source relocation (postprocess) | `source_relocation.csv` |
| Initial velocity model (preprocess) | `velocity_model_init.{csv,h5}` (coarse: `velocity_model_init_coarse.csv`) |
| Final velocity model (postprocess, resolution tests) | `velocity_model_final.{csv,h5}` (coarse: `velocity_model_final_coarse.csv`) |

---

## 1. Preprocess

### 1.1 Inputs

---

#### QuakeML Event Catalog
| Property | Value |
|---|---|
| **Required** | Yes, when QuakeML is parsed (`data.use_hdf5: false`, raw data visualization, HDF5 conversion) |
| **Format** | XML (QuakeML) |
| **Extension** | `.qml` |
| **Config key** | `data.quakeml_dir` |

Read with `obspy.read_events()`. Files sit directly in the directory or in four-digit year subdirectories (`<quakeml_dir>/<year>/<file>.qml`). Each file contains events, origins, picks, arrivals and magnitudes. Additional catalogs are given in `data.secondary_catalogs`.

---

#### Seismic Catalog — HDF5
| Property | Value |
|---|---|
| **Required** | Yes, when `data.use_hdf5: true` |
| **Format** | HDF5 (pandas tables) |
| **Extension** | `.h5` / `.hdf5` |
| **Config key** | `data.hdf5_file` |

Written by the HDF5 conversion step and read with `pd.read_hdf()`. The file holds four tables.

**Table `events`**

| Column | Description | Units |
|---|---|---|
| `event_id` | Unique event identifier | — |
| `catalog` | Catalog label (`data.catalog_label`) | — |
| `origin_time` | Origin time | UTC datetime string |
| `latitude`, `longitude` | Hypocenter position | degrees |
| `depth` | Hypocenter depth (positive down) | km |
| `magnitude`, `magnitude_type` | Magnitude and its scale | — |
| `num_picks`, `num_arrivals` | Number of picks and arrivals | — |
| `azimuthal_gap` | Largest azimuthal gap *(optional column)* | degrees |

**Table `stations`**

| Column | Description | Units |
|---|---|---|
| `station_id` | Unique station identifier | — |
| `latitude`, `longitude` | Station position | degrees |
| `depth` | Station depth (negative above mean sea level) | km |

**Table `arrivals`**

| Column | Description | Units |
|---|---|---|
| `arrival_id`, `event_id`, `pick_id`, `station_id` | Identifiers | — |
| `catalog` | Catalog label | — |
| `phase` | Phase label (`P` or `S`) | — |
| `azimuth` | Source-to-receiver azimuth | degrees |
| `distance` | Source–receiver distance of the arrival | km |
| `time_residual` | Observed minus predicted travel time | s |
| `time_weight` | Arrival weight | — |
| `epicentral_distance`, `hypocentral_distance` | Distances | km |

**Table `picks`** *(optional)*

| Column | Description | Units |
|---|---|---|
| `pick_id`, `event_id`, `station_id` | Identifiers | — |
| `catalog` | Catalog label | — |
| `pick_time` | Absolute pick time | UTC datetime string |
| `phase_hint` | Phase label | — |
| `evaluation_mode` | `manual` or `automatic` | — |

The columns written to each table are selected with `hdf5_conversion.event_fields`, `station_fields`, `pick_fields` and `arrival_fields`.

---

#### Station Inventory
| Property | Value |
|---|---|
| **Required** | Yes, when QuakeML is parsed |
| **Format** | Plain text, whitespace-delimited |
| **Config key** | `data.station_file` |

Lines starting with `#` are comments.

| Column | Description | Units |
|---|---|---|
| 1 | Station identifier | — |
| 2 | Latitude | degrees |
| 3 | Longitude | degrees |
| 4 | Elevation | m |

---

#### Initial 1D Velocity Model
| Property | Value |
|---|---|
| **Required** | Yes |
| **Format** | CSV |
| **Config key** | `initial_velocity_model.initial_model_file` |

Two column layouts are accepted.

| Layout | Columns |
|---|---|
| 3 columns | `depth` (km, positive down), `vp` (km/s), `vs` (km/s) |
| 4 columns | `depth` (km), `radius` (km), `vp` (km/s), `vs` (km/s) |

---

#### Basin Boundary Files *(optional)*
| Property | Value |
|---|---|
| **Required** | No |
| **Format** | CSV |
| **Config keys** | `coordinate_transform.basin_boundaries.data_dir`, `plots.basin_boundaries.data_dir` |

One file per basin, used as plot overlays.

| Column | Description | Units |
|---|---|---|
| `latitude`, `longitude` | Polygon vertex | degrees |

---

### 1.2 Outputs

---

#### Geodetic Events and Stations
| Property | Value |
|---|---|
| **Format** | CSV |
| **Config keys** | `coordinate_transform.events_geodetic_csv`, `coordinate_transform.stations_geodetic_csv` |

Events: `event_id, origin_time, latitude, longitude, depth, magnitude`. Stations: `station_id, latitude, longitude, depth`. Units as in the HDF5 tables.

---

#### Cartesian Events
| Property | Value |
|---|---|
| **Format** | CSV |
| **Config key** | `coordinate_transform.events_cartesian_csv` |

| Column | Description | Units |
|---|---|---|
| `event_id`, `origin_time`, `latitude`, `longitude`, `depth`, `magnitude` | As in the geodetic file | see above |
| `x`, `y`, `z` | Cartesian coordinates (z positive down) | km |

---

#### Cartesian Stations
| Property | Value |
|---|---|
| **Format** | CSV |
| **Config key** | `coordinate_transform.stations_cartesian_csv` |

| Column | Description | Units |
|---|---|---|
| `station_id`, `latitude`, `longitude`, `depth` | As in the geodetic file | see above |
| `x`, `y`, `z` | Cartesian coordinates | km |

---

#### Cartesian Picks
| Property | Value |
|---|---|
| **Format** | CSV |
| **Config key** | `coordinate_transform.picks_cartesian_csv` |

| Column | Description | Units |
|---|---|---|
| `event_id`, `station_id` | Identifiers | — |
| `phase_hint` | Phase label | — |
| `physical_travel_time` | Pick time minus origin time | s |
| `weight` | Pick weight | — |

---

#### Ground Truth Travel Times
| Property | Value |
|---|---|
| **Format** | CSV |
| **Config key** | `geometry_traveltime.csv_output_dir` (file `geometry_ground_truth_traveltimes.csv`) |
| **Condition** | Written when `generate_travel_time_csv: true` |

One row per pick. The synthetic travel time table and the source relocation table use the same layout and add columns.

| Column | Description | Units |
|---|---|---|
| `shot_id` | Shot number; matches the numbering of the geometry and binary files | — |
| `event_id` | Event identifier | — |
| `station` | Station name | — |
| `phase` | `P` or `S` | — |
| `observed_tt` | Observed (physical) travel time | s |
| `weight` | Pick weight | — |
| `event_longitude`, `event_latitude` | Event position | degrees |
| `event_depth_km` | Event depth, absolute, positive down | km |
| `station_longitude`, `station_latitude` | Station position | degrees |
| `station_depth_km` | Station depth, absolute, positive down (negative above sea level) | km |
| `event_x`, `event_y`, `event_z` | Event Cartesian coordinates relative to the transformation origin | m |
| `station_x`, `station_y`, `station_z` | Station Cartesian coordinates | m |
| `origin_time_s` | Configured origin time | s |
| `magnitude` | Event magnitude *(only with `include_magnitude_in_csv: true`)* | — |

The header carries `data_type` (`geometry_traveltime`) and `units.<column>` for every column. Residuals (`observed_tt − synthetic_tt`) and source–receiver distances are not stored; they are computed from these columns where needed.

---

#### Geometry Files
| Property | Value |
|---|---|
| **Format** | Text |
| **Config key** | `geometry_traveltime.geometry_dir` |
| **Naming** | `shot_<N>_geometry.txt`, with `N` the shot number starting at 1 |

One file per shot (event). Layout:

```
<shot id>
                                    (blank line)
1                                   number of sources
x y z t0                            source position (m) and origin time (s)
                                    (blank line)
<n receivers>
x y z weight                        one line per receiver (m, m, m, weight)
...
```

With `geometry_traveltime.separate_weight_p_s: true` every receiver line has five columns: `x y z weight_p weight_s`.

`geometry.txt` in the same directory lists the geometry file names in shot order (the index read by LATTE).

---

#### Shot Map
| Property | Value |
|---|---|
| **Format** | Plain text, tab-separated |
| **Location** | `<geometry_dir>/shot_to_source_id_map.txt` |

| Column | Description |
|---|---|
| `Shot_ID` | Integer shot number |
| `Original_Event_ID` | Event identifier from the catalog |

---

#### Ground Truth Travel Time Binaries
| Property | Value |
|---|---|
| **Format** | Raw binary (float32) |
| **Config key** | `geometry_traveltime.traveltime_dir` |
| **Naming** | `shot_<N>_traveltime_p.bin`, `shot_<N>_traveltime_s.bin` |

One value per receiver, in the receiver order of the geometry file. The values are `physical_travel_time + configured origin time` (seconds).

---

#### Source Binaries
| Property | Value |
|---|---|
| **Format** | Raw binary (float32) |
| **Config key** | `geometry_traveltime.source_dir` |
| **Condition** | `generate_source_binaries: true` |

One value per shot, in shot order.

| File | Description | Units |
|---|---|---|
| `sx.bin`, `sy.bin`, `sz.bin` | Source coordinates | m |
| `st0.bin` | Source origin times | s |

---

#### Initial Velocity Model — CSV
| Property | Value |
|---|---|
| **Format** | CSV |
| **Config key** | `initial_velocity_model.output_csv` |
| **Condition** | `output_format` is `csv` or `both` |

One row per grid point (`nx × ny × nz` rows). Every velocity-format file of the workflow (initial and final models, resolution-test models, coarse files) uses these columns and the same metadata.

| Column | Description | Units |
|---|---|---|
| `lat`, `lon` | Geographic position (NaN where not available) | degrees |
| `depth` | Absolute depth, positive down (NaN where not available) | km |
| `x`, `y`, `z` | Cartesian coordinates | km |
| `vp`, `vs` | P and S velocity | km/s |
| `grad_vp`, `grad_vs`, `srch_vp`, `srch_vs` | Gradient and search direction *(optional, written by postprocess on request)* | — |

DWS files (`ray_path_processing.dws_output_file`) have the same layout with a `dws` column in place of `vp` and `vs`.

**Metadata.** Every file carries the same flat metadata, as `# key: value` lines in CSV and as dataset attributes in HDF5.

| Key | Description |
|---|---|
| `nx`, `ny`, `nz` | Number of grid points along x, y, z |
| `dx`, `dy`, `dz` | Grid spacing (km) |
| `origin_x`, `origin_y`, `origin_z` | Cartesian coordinates of the first grid point (km) |
| `origin_lon`, `origin_lat`, `origin_depth_km` | Geographic origin of the Cartesian frame |
| `transformation_method`, `invert_y_axis` | Coordinate transformation used by preprocess |
| `data_type` | `velocity` or `dws` |
| `units.<name>` | Unit of every column and grid key |
| `fine_nx` … `fine_origin_z`, `stride_x/y/z` | *Coarse files only:* grid of the original fine model and the subsampling strides |
| `interpolation_mode`, `phase_type`, `checkerboard_config.*`, `spike_config.*`, `description` | Module-specific provenance *(optional)* |

The grid is defined once, in preprocess. All later stages read it from this metadata and compare it with the `x`, `y`, `z` columns, logging a warning for any disagreement (the metadata value is used). If a file has no grid metadata, the grid is inferred from the columns, with a warning. Grid keys are needed in a module's configuration only when there is no velocity file to read the grid from (`ray_path_processing.grid` when `reference_velocity_file` is `null`), or as an additional check (`velocity_processing.model_grid`). The ray path cache stores the grid of the run, and ray density analysis reads it from there; `ray_density_analysis.grid` fills only keys that the cache lacks.

---

#### Initial Velocity Model — HDF5
| Property | Value |
|---|---|
| **Format** | HDF5 |
| **Config key** | `initial_velocity_model.output_hdf5` |
| **Condition** | `output_format` is `hdf5` or `both` |

| Dataset | Shape | Description | Units |
|---|---|---|---|
| `velocity` | `(n_points, n_columns)` | All grid points, same columns as the CSV | km, km/s |
| `velocity_coarse` *(optional)* | `(n_coarse, n_columns)` | Subsampled grid for fast analysis | km, km/s |

The dataset attributes hold the metadata above and `columns` (the column names). `velocity_coarse` has its own grid keys plus the `fine_*` keys. Data is stored with gzip compression (level set by `compression_level`).

---

#### Initial Velocity Model — Binaries
| Property | Value |
|---|---|
| **Format** | Raw binary (float32) |
| **Config keys** | `initial_velocity_model.vp_binary`, `initial_velocity_model.vs_binary` |

One file each for Vp and Vs in **m/s**, Fortran order, shape `(nz, ny, nx)` (see [Conventions](#conventions)).

---

#### Basin Boundaries — Cartesian *(optional)*
| Property | Value |
|---|---|
| **Format** | CSV |
| **Location** | `<coordinate_transform.output_dir>/basin_boundaries/<label>.csv` |

| Column | Description | Units |
|---|---|---|
| `x`, `y` | Cartesian polygon vertex | km |

---

#### Visualization Plots
| Property | Value |
|---|---|
| **Format** | PNG by default (`plots.figure.format`: `png`, `pdf`, `svg`) |
| **Config keys** | `final_data_visualization.*.output_dir`, `raw_data_visualization.output_dir` |

| File | Description |
|---|---|
| `map_view_lat_lon` | Map of events and stations |
| `cross_section_lat_depth`, `cross_section_lon_depth` | Depth cross-sections |
| `velocity_slice_depth_<d>km` | Vp and Vs depth slice |
| `velocity_profiles` | 1D velocity profiles |
| `histogram_*`, `cumulative_azimuthal_gap` | Raw-data histograms (see [preprocess.md](preprocess.md)) |

---

## 2. Postprocess

### 2.1 Inputs

---

#### Updated Velocity Binaries
| Property | Value |
|---|---|
| **Required** | Yes (velocity processing) |
| **Format** | Raw binary (float32) |
| **Config keys** | `velocity_processing.vp_binary`, `velocity_processing.vs_binary` |

Written by LATTE: velocities in m/s, Fortran order, shape `(nz, ny, nx)`.

---

#### Ground Truth Travel Times
| Property | Value |
|---|---|
| **Required** | Yes (travel time processing) |
| **Format** | CSV |
| **Config key** | `travel_time_processing.ground_truth_csv` |

The preprocess table described in [§1.2](#ground-truth-travel-times).

---

#### Synthetic Travel Time Binaries
| Property | Value |
|---|---|
| **Required** | Yes (travel time processing) |
| **Format** | Raw binary (float32) |
| **Config key** | `travel_time_processing.synthetic_tt_dir` |
| **Naming** | `shot_<id>_traveltime_p.bin`, `shot_<id>_traveltime_s.bin` |

One file per shot and phase, with one value per receiver in the order of the geometry file. The values are `physical_tt + origin time`. In source–receiver exchange mode the files belong to the virtual sources and hold one value per shot kept by LATTE.

---

#### Shot Map
| Property | Value |
|---|---|
| **Required** | Yes (travel time and ray path processing) |
| **Format** | Plain text, tab-separated |
| **Config key** | `travel_time_processing.shot_map_file` |

Same file as the preprocess [Shot Map](#shot-map).

---

#### Geometry Directory *(exchange mode)*
| Property | Value |
|---|---|
| **Required** | In source–receiver exchange mode, unless `exchange_geometry_file` is given |
| **Format** | Text |
| **Config key** | `travel_time_processing.geometry_dir` |

The preprocess geometry directory (`geometry.txt` and `shot_<N>_geometry.txt`).

---

#### Exchange Geometry *(optional)*
| Property | Value |
|---|---|
| **Required** | No |
| **Format** | Plain text |
| **Config key** | `travel_time_processing.exchange_geometry_file` |

Written by LATTE. One line per virtual source: `virtual_id x y z` (coordinates in m). When given it is used as the virtual-source table.

---

#### Source Binaries
| Property | Value |
|---|---|
| **Required** | Yes (source processing) |
| **Format** | Raw binary (float32) |
| **Config keys** | `source_processing.sx_file`, `sy_file`, `sz_file`, `st0_file` |

Written by LATTE, one value per shot kept by LATTE (the shots of `initial_geometry_csv` that are not in `filtered_shot_ids`, in ascending shot id).

| File | Description | Units |
|---|---|---|
| `updated_sx.bin`, `updated_sy.bin`, `updated_sz.bin` | Updated source coordinates | m |
| `updated_st0.bin` | Updated origin times | s |

`updated_st0.bin` is also read by `travel_time_processing.source_correction_file` when `source_correction: true`; there it is indexed by shot id.

---

#### Gradient / Search Direction Binaries *(optional)*
| Property | Value |
|---|---|
| **Required** | No |
| **Format** | Raw binary (float32) |
| **Config keys** | `velocity_processing.grad_*_binary`, `srch_*_binary`; `source_processing.grad_*_file`, `srch_*_file` |

| File | Description |
|---|---|
| `grad_vp.bin`, `grad_vs.bin` | Velocity gradient (model grid, Fortran order) |
| `srch_vp.bin`, `srch_vs.bin` | Velocity search direction |
| `grad_sx.bin`, `grad_sy.bin`, `grad_sz.bin`, `grad_st0.bin` | Source gradient (one value per kept shot) |
| `srch_sx.bin`, `srch_sy.bin`, `srch_sz.bin`, `srch_st0.bin` | Source search direction |

---

#### Ray Path Binaries
| Property | Value |
|---|---|
| **Required** | Yes (ray path and DWS processing) |
| **Format** | Raw binary |
| **Config key** | `ray_path_processing.raypath_binary_dir` |
| **Naming** | `shot_<shot_id>_raypath_<phase>.bin` with `phase` = `p` or `s` |

Written by LATTE (`raypath_save = true`).

| Field | Type | Description |
|---|---|---|
| `nrays` | `int32` | Number of rays in the shot |
| `ray_npoints[nrays]` | `int32` | Number of points of each ray |
| coordinates | `float32` | Ray points `x, y, z`, consecutive per point, rays one after another (m) |

---

### 2.2 Outputs

---

#### Updated Velocity Model — CSV and HDF5
| Property | Value |
|---|---|
| **Format** | CSV / HDF5 |
| **Config keys** | `velocity_processing.velocity_output_csv`, `velocity_output_hdf5` |

Same columns, datasets and metadata as the preprocess [initial velocity model](#initial-velocity-model--csv), with the updated `vp` and `vs` (km/s).

---

#### Synthetic Travel Times — CSV
| Property | Value |
|---|---|
| **Format** | CSV |
| **Config key** | `travel_time_processing.synthetic_tt_output_csv` |

The ground truth table plus:

| Column | Description | Units |
|---|---|---|
| `synthetic_tt` | Synthetic travel time with the origin time removed | s |
| `corrected_origin_time_s` | Corrected origin time *(only when a source correction was applied)* | s |

The header carries `data_type: "synthetic_traveltime"` and `units.*` entries as in the ground truth table.

---

#### Source Relocation — CSV
| Property | Value |
|---|---|
| **Format** | CSV |
| **Config key** | `source_processing.output_csv` |

The synthetic travel time table plus:

| Column | Description | Units |
|---|---|---|
| `corrected_event_x`, `corrected_event_y`, `corrected_event_z` | Updated source coordinates | m |
| `corrected_origin_time_s` | Updated origin time | s |
| `grad_*`, `srch_*` | Gradient and search direction columns *(optional)* | — |

Shots that LATTE removed (`filtered_shot_ids`) keep empty corrected columns. Header `data_type: "source_relocation"`.

---

#### Ray Path Cache — HDF5
| Property | Value |
|---|---|
| **Format** | HDF5 |
| **Config key** | `ray_path_processing.ray_path_output_file` |

Groups: `/p_wave/shots/<shot_id>/` and `/s_wave/shots/<shot_id>/`.

| Dataset | Shape | Description | Units |
|---|---|---|---|
| `source` | `(1, 3)` | Source `x, y, z` | km |
| `receivers` | `(n_recv, 3)` | Receiver `x, y, z` | km |
| `weights` | `(n_recv,)` | Per-ray weights | — |
| `ray_lengths` | `(n_recv,)` | Number of points per ray | — |
| `ray_data` | `(total_points, 3)` | Concatenated ray coordinates | km |

Data is stored with the configured compression. The file also stores the grid of the run and a hash of the grid and binary-directory settings; the cache is rebuilt when they change.

---

#### Derivative Weight Sum — HDF5
| Property | Value |
|---|---|
| **Format** | HDF5 (velocity format) |
| **Config key** | `ray_path_processing.dws_output_file` |

Dataset `velocity` with the columns `lat, lon, depth, x, y, z, dws` and the grid metadata with `data_type: "dws"` (see [Initial Velocity Model — CSV](#initial-velocity-model--csv)). `dws` is the sum of `weight × (ray length in the cell)²` over all rays, in km².

---

#### Simplified Ray Path Cache — HDF5 *(optional)*
| Property | Value |
|---|---|
| **Format** | HDF5 |
| **Config key** | `ray_path_processing.simplification.output_cache_path` |

Same structure as the [ray path cache](#ray-path-cache--hdf5); rays are simplified with the Ramer–Douglas–Peucker algorithm with tolerance `simplification.epsilon` (km).

---

## 3. Analysis

### 3.1 Inputs

---

#### Travel Time CSVs
| Property | Value |
|---|---|
| **Required** | Yes (one or two files, by mode) |
| **Format** | CSV |
| **Config keys** | `travel_time_analysis.input_csv` (`ground_truth`); `method1_csv`, `method2_csv` (`method_comparison`); `distance_analysis.travel_time_csv`, … |

The postprocess [synthetic travel time table](#synthetic-travel-times--csv).

---

#### Velocity Model Files
| Property | Value |
|---|---|
| **Required** | Yes (velocity analysis, one or two files) |
| **Format** | CSV or HDF5 |
| **Config keys** | `velocity_analysis.model1_csv`, `model2_csv` (or `*_hdf5`) |

The velocity-format files described in [§1.2](#initial-velocity-model--csv).

---

#### Cartesian Events and Stations
| Property | Value |
|---|---|
| **Required** | Yes (distance and short-distance analysis) |
| **Format** | CSV |
| **Config keys** | `distance_analysis.events_csv`, `distance_analysis.station_csv` |

The preprocess [Cartesian events](#cartesian-events) and [Cartesian stations](#cartesian-stations).

---

#### Source Relocation CSV
| Property | Value |
|---|---|
| **Required** | Yes (source and source-gradient analysis) |
| **Format** | CSV |
| **Config key** | `source_analysis.input_csv` |

The postprocess [source relocation table](#source-relocation--csv).

---

#### DWS — HDF5
| Property | Value |
|---|---|
| **Required** | Yes (DWS analysis) |
| **Format** | HDF5 |
| **Config key** | `dws_analysis.input_hdf5` |

The postprocess [DWS file](#derivative-weight-sum--hdf5). Values are km² by default; `use_m2_units: true` shows m².

---

#### Ray Path Cache — HDF5
| Property | Value |
|---|---|
| **Required** | Yes (ray path and ray density analysis) |
| **Format** | HDF5 |
| **Config keys** | `raypath_analysis.input_cache`, `ray_density_analysis.input_cache` |

The postprocess [ray path cache](#ray-path-cache--hdf5).

---

#### Data Misfit File
| Property | Value |
|---|---|
| **Required** | Yes (tradeoff analysis in `data_misfit` mode) |
| **Format** | Plain text, whitespace-separated |
| **Config key** | `tradeoff_analysis.data_misfit_file_path` |

| Column | Description | Units |
|---|---|---|
| 1 `iteration` | Inversion iteration | — |
| 2 `data_misfit` | Absolute data misfit | s (or dimensionless) |
| 3 `normalized_data_misfit` | Misfit normalised by its initial value | — |

---

#### Iteration Directories
| Property | Value |
|---|---|
| **Required** | Yes (tradeoff analysis in `rms` / `data_variance` mode or for the model change) |
| **Config keys** | `tradeoff_analysis.inversion_base_path`, `inversion_analysis_dir` |

Directories named `iteration_<N>/`, each holding the postprocess travel time CSV and velocity file of that iteration.

---

#### Basin Boundary Files *(optional)*
| Property | Value |
|---|---|
| **Required** | No |
| **Format** | CSV |
| **Config key** | `basin_boundaries.data_dir` |

The Cartesian [basin boundaries](#basin-boundaries--cartesian-optional) (`x`, `y` in km).

---

#### Gradient / Search Direction Files *(optional)*
| Property | Value |
|---|---|
| **Required** | No |
| **Format** | CSV or HDF5 |
| **Config keys** | `gradient_analysis.velocity_input_csv`, `gradient_analysis.source_input_csv` (or `*_hdf5`) |

The velocity file with `grad_vp`, `grad_vs`, `srch_vp`, `srch_vs`, and the source relocation table with `grad_*` and `srch_*` columns.

---

### 3.2 Outputs

Analysis writes plots, statistics and text reports. The only data table is the optional source-delta CSV. Every module writes into its own `output_dir`, usually with `plots/`, `reports/` and (for some modules) `statistics/` subdirectories. The plot extension follows `analysis.output_format`.

| Module | Output | Format |
|---|---|---|
| Travel time | `plots/residual_histograms`, `synthetic_vs_observed`, `synthetic_vs_observed_loglog`, `ln_ratio_analysis`, `ln_ratio_histograms`; `reports/travel_time_analysis_report.txt` | Image / text |
| Velocity | `plots/basic_velocity_comparison`, `spatial_velocity_projections`, `spatial_geometric_ratio_projections`, `velocity_vertical_profiles`, `depth_slice_heatmaps`, `xy_plane_ratio_*`, `xz_plane_ratio_*`, `yz_plane_ratio_*`, `domain_overview`; `reports/velocity_comparison_report.txt` | Image / text |
| Distance | `plots/<subdir>/residuals_vs_distance`, `binned_distance_analysis`, `correlation_analysis`, `ratios_vs_distance`; `reports/<subdir>/distance_analysis_report_*.txt` | Image / text |
| Short distance | `plots/<subdir>/overview_map`, `residual_histograms`, `distance_vs_residual`, `cluster_<name>`; `reports/<subdir>/short_distance_report_*.txt` | Image / text |
| Source | `plots/hist_*`, `scatter_relationships`, `geographic_corrections_<plane>`, `depth_analysis`, `correction_summary`; `reports/source_correction_report.txt`; `source_corrections_with_deltas.csv` *(optional)* | Image / text / CSV |
| Gradient | `plots/*_gradient_histograms`, `*_search_histograms`, `*_gradient_search_alignment`, `velocity_gradient_<param>_xy_projection`, `velocity_gradient_depth_profiles`; `reports/velocity_gradient_analysis_report.txt`, `source_gradient_analysis_report.txt` | Image / text |
| Ray path | `plots/ray_paths_<phase>_wave_<plane>`, `raypath_summary`; `statistics/raypath_statistics.json`; `raypath_analysis_report.txt` | Image / JSON / text |
| DWS | `plots/dws_horizontal_depth_<d>km_{linear,log}`, `dws_vertical_{xz,yz}_pos_<p>km_{linear,log}`, `dws_coverage_depth_<d>km`, `dws_histogram_{linear,log}`; `statistics/dws_statistics.json`; `reports/dws_analysis_report.txt` | Image / JSON / text |
| Ray density | `plots/ray_density_<phase>_wave_<plane>`, `ray_density_<phase>_wave_xy_threshold_count`, `ray_density_depth_<d>km`, `ray_density_<phase>_wave_oblique_<label>`, `ray_density_oblique_section_plan_view`; `statistics/ray_density_<phase>_wave_statistics.json` | Image / JSON |
| Tradeoff | `plots/tradeoff_curve`, `tradeoff_curve_<phase>`, `tradeoff_curve_depth_<d>km`; `statistics/tradeoff_statistics.csv`; `reports/tradeoff_analysis_report.txt` | Image / CSV / text |

---

## 4. Resolution Tests

The checkerboard and spike tests ([resolution_tests.md](resolution_tests.md)) reuse the formats above. Their own files:

| File | Format | Description |
|---|---|---|
| `checkerboard_model/velocity_model_checkerboard.{csv,h5}`, `spike_model/velocity_model_spike.{csv,h5}` | CSV / HDF5 | Perturbed model in the velocity format (km/s) |
| `checkerboard_model/vp_checkerboard.bin`, `vs_checkerboard.bin`; `spike_model/vp_spike.bin`, `vs_spike.bin` | float32 | Perturbed Vp and Vs binaries (m/s) for LATTE |
| `noisy_traveltimes/shot_<N>_traveltime_{p,s}.bin` | float32 | Synthetic travel times with noise added, same layout as the synthetic binaries |
| `noisy_traveltimes/noisy_traveltimes.csv` | CSV | Noisy travel times with the ground truth metadata, `data_type: "noisy_traveltime"`; `observed_tt` holds the noisy value |
| `inverted_model/velocity_model_final.{csv,h5}` | CSV / HDF5 | Inverted model converted by the same code as the postprocess velocity stage |
| `visualizations/` | Image | Slice plots (model, difference, pattern) |

---

## 5. Configuration Files

All pipelines use [Hydra](https://hydra.cc/) for configuration.

| Pipeline | Default configuration |
|---|---|
| Preprocess | `configs/preprocess/default.yaml` |
| Postprocess | `configs/postprocess/default.yaml` |
| Analysis | `configs/analysis/default.yaml` |
| Checkerboard test | `configs/checkerboard_test/default.yaml` |
| Spike test | `configs/spike_test/default.yaml` |

Each YAML file is organised in named sections for the pipeline modules (for example `data`, `coordinate_transform`, `geometry_traveltime`, `velocity_processing`, `travel_time_analysis`). File paths may be absolute or relative to the working directory.

---

## 6. File Format Summary

| Format | Extensions | Pipelines | Role |
|---|---|---|---|
| CSV | `.csv` | All | Events, stations, velocity models, travel times, source tables |
| HDF5 | `.h5`, `.hdf5` | All | Seismic catalog, velocity models, ray path cache, DWS |
| Binary (float32) | `.bin` | Preprocess, Postprocess, Resolution tests | Velocity models, source parameters, travel times, ray paths |
| Plain text | `.txt` | All | Station inventory, geometry files, shot map, reports |
| QuakeML (XML) | `.qml` | Preprocess | Seismic event catalog |
| YAML | `.yaml` | All | Hydra configuration |
| PNG / PDF / SVG | `.png`, `.pdf`, `.svg` | All | Plots |
