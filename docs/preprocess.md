# Preprocess Pipeline

The preprocess pipeline transforms a raw seismic catalog (QuakeML files or an HDF5 file) into the binary and tabular inputs required by the LATTE tomography inversion. It covers one-time QuakeML parsing, catalog filtering, optional raw-data plots and HDF5 conversion, coordinate transformation, geometry and travel time generation, initial velocity model creation, and final-data visualization.

> **Configuration reference:** See [`configs/preprocess/README.md`](../configs/preprocess/README.md) for a complete description of every configuration parameter.

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

The pipeline is driven by `preprocess/main.py`. It runs the steps below in order; each step is switched on or off by its own `enabled` flag, and `preprocessing.run_pipeline: false` stops the run after the raw-data steps (1.2 and 1.3).

```
QuakeML parse (once) ──► raw data visualization ──► HDF5 conversion ──► pipeline:
                                                                         data loading
                                                                         → coordinate transform
                                                                         → geometry / travel times
                                                                         → initial velocity model
                                                                         → final data visualization
```

---

### 1.1 QuakeML Loading and Catalog Filters (`quakeml_loader.py`, `catalog_filters.py`)

Parses the QuakeML files once, without filtering, and hands the parsed tables (events, stations, picks, arrivals) to every step that needs them. Each consumer then applies its own filters through `catalog_filters.py`.

**What it does:**

- Scans `data.quakeml_dir` for year-named subdirectories (for example `2020/`, `2021/`) and for `.qml` files placed directly in the directory. `data.years` restricts the scan to specific years and `data.max_files` caps the number of files.
- Parses files in parallel (`parallel.num_workers`, `parallel.chunk_size`).
- Reads station locations from `data.station_file` (whitespace-delimited: ID, latitude, longitude, elevation in m).
- Selects the preferred origin and magnitude of each event (`data.use_preferred_origin`, `data.use_preferred_magnitude`) or the entry given by `data.origin_index` / `data.magnitude_index`.
- Tags every row with a `catalog` label (`data.catalog_label`).
- Optionally appends **secondary catalogs** (`data.secondary_catalogs`). When the same station code appears in several catalogs, stations at the same location (within `identical_tolerance_m`) are shared; conflicting stations of an additional catalog are renamed by appending `station_suffix`.
- Filters, per consumer, with the keys `manual_picks_only`, `magnitude_min/max`, `lat_min/max`, `lon_min/max`, `depth_min/max`, `epicentral_distance_min/max`, `hypocentral_distance_min/max`, and, for the raw-data and HDF5 steps, `require_both_p_and_s`, `azimuthal_gap_max`, `min_picks_per_event`, `max_picks_per_event`. After every filter the tables are synchronised: arrivals and picks reference only remaining events, every remaining event has at least one arrival or pick, and only referenced stations are kept.

**Key configuration:**

| Key | Purpose |
|---|---|
| `data.quakeml_dir` | Root directory of QuakeML files |
| `data.station_file` | Station inventory |
| `data.years` / `data.max_files` | Restrict the files that are parsed |
| `data.secondary_catalogs.*` | Additional QuakeML catalogs and station-merge rules |
| `parallel.num_workers` | Worker count; `null` uses all cores |
| `filters.*` | Global filters used by the pipeline |

---

### 1.2 Raw Data Visualization (`raw_data_visualization.py`)

Plots the parsed QuakeML catalog before it enters the pipeline. It has its own filters (`raw_data_visualization.filters`), which default to the global `filters` and can be overridden individually. It always reads QuakeML, independent of `data.use_hdf5`.

**Plots** (select with `raw_data_visualization.plots_to_generate`):

| Plot key | Output |
|---|---|
| `source_receiver_maps` | Map view and latitude/longitude depth cross-sections of events and receivers |
| `histogram_source_receivers` | Latitude, longitude and depth histograms of sources and receivers |
| `histogram_temporal` | Events over time (`daily`, `weekly`, `monthly`, `yearly` bins) |
| `histogram_magnitude` | Magnitude histogram |
| `histogram_epicentral_distance`, `histogram_hypocentral_distance` | Distance histograms and cumulative curves, optionally per magnitude bin |
| `histogram_picks_per_event` | Picks-per-event histogram |
| `cumulative_azimuthal_gap` | Cumulative distribution of the azimuthal gap |
| `histogram_magnitude_picks` | Magnitude versus number of picks |

Styling comes from the shared `plots` section (see [§1.8](#18-final-data-visualization-visualizationpy)).

---

### 1.3 HDF5 Conversion (`hdf5_conversion.py`)

Writes the filtered QuakeML catalog to an HDF5 file, so later runs can read it with `data.use_hdf5: true` instead of re-parsing thousands of XML files.

**What it does:**

- Applies `hdf5_conversion.filters` (defaulting to the global `filters`).
- Writes the tables `events`, `stations`, `picks` and `arrivals` to `hdf5_conversion.output_file` (default: `data.hdf5_file`) with `DataFrame.to_hdf`.
- Keeps one P-type and one S-type pick per event–station pair when `single_pick_per_phase: true` (exact `P`/`S` is preferred over `Pg`/`Pn`/…; manual over automatic).
- Takes the phase from the arrival when a pick has no phase hint (`phase_hint_from_arrival`), and drops picks and arrivals of stations missing from the station list (`require_station_coordinates`).
- Writes only the columns listed in `event_fields`, `station_fields`, `pick_fields`, `arrival_fields`; an empty list writes all columns.

The conversion runs before the pipeline, so with `data.use_hdf5: true` the pipeline reads the freshly written file.

---

### 1.4 Data Loading (`data_loader.py`)

Loads events, stations, picks and arrivals for the pipeline and applies the global filters.

**What it does:**

- `data.use_hdf5: true` reads the `events`, `stations`, `arrivals` and (optionally) `picks` tables from `data.hdf5_file` with `pd.read_hdf()`.
- `data.use_hdf5: false` uses the QuakeML catalog parsed in [§1.1](#11-quakeml-loading-and-catalog-filters-quakeml_loaderpy-catalog_filterspy).
- Applies the filters in order: magnitude → geographic bounding box → depth → hypocentral / epicentral distance → manual picks only.
- Computes epicentral and hypocentral distances for every event–station pair.

---

### 1.5 Coordinate Transformation (`coordinate_transformation.py`)

Converts geodetic coordinates (longitude, latitude, depth in km) to a local Cartesian frame (x east, y, z down; km) with a first-order flattening transformation.

**What it does:**

- Places the Cartesian origin at `(origin_lon, origin_lat, origin_depth_km)`.
- Computes:
  - `x = R · (λ − λ₀) · cos(φ₀)`
  - `y = ±R · (φ − φ₀)` (sign set by `invert_y_axis`; `true` makes +y point south)
  - `z = depth − origin_depth_km`
- Writes events and stations in geodetic and Cartesian form, and the picks in Cartesian form.
- Optionally transforms basin boundary polygons from geographic to Cartesian coordinates for plot overlays.

**Key configuration:**

| Key | Purpose |
|---|---|
| `coordinate_transform.enabled` | Enable/disable this step |
| `coordinate_transform.origin_lon/lat/depth_km` | Cartesian origin |
| `coordinate_transform.grid_spacing_km` | `[dx, dy, dz]` of the velocity model grid |
| `coordinate_transform.invert_y_axis` | `true` = +y points south |
| `coordinate_transform.basin_boundaries.*` | Basin polygons to transform |

---

### 1.6 Geometry and Travel Time Generation (`geometry_traveltime.py`)

Writes the geometry files, travel time binaries and source binaries read by the LATTE inversion.

**What it does:**

- Matches picks to Cartesian event and station coordinates and converts coordinates to **metres**.
- Rejects picks with a negative travel time or an apparent velocity outside `p_velocity_range_m_s` (default 500–10 000 m/s) or `s_velocity_range_m_s` (default 500–6 000 m/s).
- Applies the phase filter: `both` keeps only the stations where an event has both a P and an S pick; `p` or `s` keeps the picks of that phase only.
- Assigns pick weights in one of three ways: unit weights (`apply_default_weights`, `default_weight`), **global-scale azimuthal weighting** (`apply_azimuthal_weighting`: picks are counted per station and azimuthal bin across all events and weighted by 1/√N or 1/N), or no weighting. `apply_default_weights` and `apply_azimuthal_weighting` are mutually exclusive.
- With `separate_weight_p_s: true`, writes 5-column geometry files (`x y z weight_p weight_s`); `default_weight_p` and `default_weight_s` are the base weights, and they scale the azimuthal weights when azimuthal weighting is on.
- Writes one geometry file and one travel time binary per phase for every event (one *shot* per event, numbered from 1).
- Writes `geometry.txt` (an index of the geometry files) and `shot_to_source_id_map.txt` (shot index ↔ event ID).
- Writes the source binaries `sx.bin`, `sy.bin`, `sz.bin` (metres) and `st0.bin` (seconds), one value per shot, in shot order.
- Writes a human-readable `geometry_ground_truth_traveltimes.csv` when `generate_travel_time_csv` is true.

> **Important:** The travel time binaries contain `physical_travel_time + configured_origin_time`. The CSV column `observed_tt` contains the physical travel time only. Keep the distinction in mind when post-processing.

> **Important:** After this step, set the number of sources `ns` in the LATTE parameter file to the number of shots reported in the log.

**Key configuration:**

| Key | Purpose |
|---|---|
| `geometry_traveltime.enabled` | Enable/disable this step |
| `geometry_traveltime.phase_type` | `both` / `p` / `s` |
| `geometry_traveltime.p_velocity_range_m_s`, `s_velocity_range_m_s` | Acceptable apparent velocity range (m/s) |
| `geometry_traveltime.apply_default_weights`, `default_weight` | Uniform pick weights |
| `geometry_traveltime.apply_azimuthal_weighting`, `azimuthal_bin_size_deg`, `azimuthal_weight_formula` | Azimuthal weights (`inverse_sqrt` or `inverse`) |
| `geometry_traveltime.separate_weight_p_s`, `default_weight_p`, `default_weight_s` | Separate P and S weights |
| `geometry_traveltime.generate_travel_time_csv` | Write the human-readable CSV |
| `geometry_traveltime.include_magnitude_in_csv` | Add a `magnitude` column to the CSV |
| `origin_time.mode` | `fixed` (constant) or `unix_epoch` |
| `origin_time.fixed_value` | Origin time (s) when the mode is `fixed` |

---

### 1.7 Initial Velocity Model (`initial_velocity_model.py`)

Extrudes a one-dimensional depth–velocity profile onto a regular three-dimensional Cartesian grid to create the starting model of the inversion.

**What it does:**

- Reads a 1D model CSV with columns `depth, vp, vs` (or `depth, radius, vp, vs`).
- Builds the grid from `grid_spacing_km`. Bounds are taken from the transformed events and stations (`use_auto_bounds: true`) or from `x_min` … `z_max`.
- Interpolates Vp and Vs linearly in depth; large grids (more than `parallel.vectorize_max_points` points) are processed in chunks of `parallel.grid_chunk_size`.
- Writes the model as CSV, HDF5 or both (`output_format`). Every file carries the grid metadata (`nx`, `dx`, `origin_x`, …) used by all later stages.
- Optionally writes a coarse-resolution dataset or CSV (`hdf5_options.create_coarse_dataset`, `write_coarse_csv`) for faster analysis.
- Always writes the LATTE binaries `vp_init.bin` and `vs_init.bin` in **m/s**, float32, in Fortran order with shape `(nz, ny, nx)` (z varies fastest).

**Key configuration:**

| Key | Purpose |
|---|---|
| `initial_velocity_model.enabled` | Enable/disable this step |
| `initial_velocity_model.initial_model_file` | Path to the 1D velocity CSV |
| `initial_velocity_model.max_depth_km` | Model entries below this depth are ignored |
| `initial_velocity_model.use_auto_bounds` | Infer the grid extent from the data |
| `initial_velocity_model.output_format` | `csv` / `hdf5` / `both` |
| `initial_velocity_model.hdf5_options.*` | Compression and coarse-grid options |

---

### 1.8 Final Data Visualization (`visualization.py`)

Generates quality-control plots of the data written by the pipeline.

**What it does:**

- **Station/event plots** (`final_data_visualization.station_event_vis`): map view and latitude–depth and longitude–depth cross-sections of events and stations, with optional pick lines and basin boundary overlays.
- **Velocity model plots** (`final_data_visualization.velocity_model_vis`, disabled by default): Vp and Vs depth slices at `depth_slices` and 1D profiles along latitude and/or longitude.
- All styling (figure size, DPI, format, fonts, marker styles, basin boundaries) comes from the shared top-level `plots` section. `raw_data_visualization.plots` and `final_data_visualization.plots` override individual keys for their own plots.

---

## 2. Outputs

All paths below are relative to `data.output_base_dir`. See [`file_formats.md`](file_formats.md) for column and key definitions and units.

| File | Format | Step | Description |
|---|---|---|---|
| `<hdf5_conversion.output_file>` | HDF5 | HDF5 conversion | Filtered catalog tables |
| `raw_data_visualization/*.png` | PNG | Raw data plots | Catalog maps and histograms |
| `events_geodetic.csv`, `stations_geodetic.csv` | CSV | Data loading | Filtered events and stations in geographic coordinates |
| `coordinate_transform/events_cartesian.csv` | CSV | Coord. transform | Events in Cartesian km |
| `coordinate_transform/stations_cartesian.csv` | CSV | Coord. transform | Stations in Cartesian km |
| `coordinate_transform/picks_cartesian.csv` | CSV | Coord. transform | Picks |
| `coordinate_transform/basin_boundaries/*.csv` | CSV | Coord. transform | Cartesian basin polygons (optional) |
| `geometry/shot_N_geometry.txt` | Text | Geometry/TT | Per-shot geometry for LATTE |
| `geometry/geometry.txt` | Text | Geometry/TT | Index of the geometry files |
| `geometry/shot_to_source_id_map.txt` | Text | Geometry/TT | Shot index ↔ event ID |
| `ground_truth_traveltimes/shot_N_traveltime_{p,s}.bin` | Binary (float32) | Geometry/TT | Per-shot travel times |
| `model/sx.bin`, `sy.bin`, `sz.bin`, `st0.bin` | Binary (float32) | Geometry/TT | Source coordinates (m) and origin times (s) |
| `geometry_csv/geometry_ground_truth_traveltimes.csv` | CSV | Geometry/TT | Per-pick observed travel times |
| `velocity_model_init.csv`, `velocity_model_init.h5` | CSV / HDF5 | Velocity model | Initial 3D model (km/s) |
| `model/vp_init.bin`, `model/vs_init.bin` | Binary (float32) | Velocity model | LATTE velocity binaries (m/s) |
| `visualizations/` | PNG | Final visualization | Maps, cross-sections, depth slices, profiles |

---

## 3. Usage and Configuration

### Running the pipeline

```bash
# Default configuration (configs/preprocess/default.yaml)
python -m preprocess.main

# A named configuration (configs/preprocess/<name>.yaml)
python -m preprocess.main --config-name example
```

Hydra writes the resolved configuration and the log to a timestamped directory:

```
outputs/YYYY-MM-DD/HH-MM-SS_preprocess_<config_name>/
```

### CLI overrides

Any configuration key can be overridden with Hydra dot-notation:

```bash
# Read the catalog from HDF5
python -m preprocess.main data.use_hdf5=true data.hdf5_file=/data/catalog.h5

# Parse QuakeML, write an HDF5 file, and stop before the pipeline
python -m preprocess.main \
  data.use_hdf5=false \
  hdf5_conversion.enabled=true \
  preprocessing.run_pipeline=false

# Plot the raw catalog only
python -m preprocess.main \
  raw_data_visualization.enabled=true \
  preprocessing.run_pipeline=false

# Origin and grid spacing
python -m preprocess.main \
  coordinate_transform.origin_lon=-104.5 \
  coordinate_transform.origin_lat=31.8 \
  coordinate_transform.origin_depth_km=0.0 \
  coordinate_transform.grid_spacing_km="[2.0, 2.0, 1.0]"

# Magnitude and depth filters
python -m preprocess.main filters.magnitude_min=2.0 filters.depth_max=40.0

# Quick test with a few files
python -m preprocess.main data.use_hdf5=false data.max_files=50 parallel.num_workers=4

# Plots only (skip all processing steps)
python -m preprocess.main \
  coordinate_transform.enabled=false \
  geometry_traveltime.enabled=false \
  initial_velocity_model.enabled=false

# Log verbosity
python -m preprocess.main preprocessing.log_level=DEBUG
```

### Minimal configuration checklist

```yaml
preprocessing:
  output_dir: /path/to/output

data:
  use_hdf5: true
  hdf5_file: /path/to/seismic_data.h5
  quakeml_dir: /path/to/quakeml          # used when use_hdf5 is false or for QuakeML-based steps
  station_file: /path/to/stations.txt    # used when QuakeML is parsed
  output_base_dir: /path/to/output

coordinate_transform:
  origin_lon: -104.5
  origin_lat: 31.8
  origin_depth_km: 0.0

initial_velocity_model:
  initial_model_file: /path/to/1d_model.csv
```

`configs/preprocess/example.yaml` is a ready-to-copy template.

---

## 4. Logging

The log level is set by `preprocessing.log_level` (default `INFO`; `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). Each long-running step reports its wall-clock time (`Completed <step> in <time>`).

| Level | What is logged |
|---|---|
| `DEBUG` | Resolved configuration; per-pick rejection reasons; coordinate ranges |
| `INFO` | Step start and end; file and event counts; filter results; grid dimensions; output paths |
| `WARNING` | Uneven azimuthal bins; missing optional files; stations absent from the inventory; renamed stations from secondary catalogs |
| `ERROR` | Configuration validation failures; invalid parameter combinations |

Hydra also stores the console output and the resolved configuration in the run directory under `outputs/`.

Messages to check after a run:

```
# Geometry / travel time summary: confirm pick acceptance
Pick filtering summary:
  Events parsed: N in -> M with valid picks
  Picks accepted: N
  Picks rejected (negative travel time): N
  Picks rejected (unrealistic velocity): N

# LATTE parameter file: update the number of sources
IMPORTANT: Update 'ns' parameter in Fortran input to N
```

---

## 5. Troubleshooting

**No events after filtering**
Compare the filter bounds with the catalog extent. Run with `preprocessing.log_level=DEBUG` to see the count after each filter. Widen all filters, confirm that data loads, then tighten them one at a time.

**`No events parsed from QuakeML`**
Check `data.quakeml_dir` and `data.years`. Files must sit in the root of `quakeml_dir` or in four-digit year subdirectories (`<quakeml_dir>/<year>/<file>.qml`).

**Station not found for an arrival**
The station ID in the arrival does not match any entry in `data.station_file`. Check that both use the same naming and look for trailing whitespace or case differences.

**`ValueError: apply_default_weights and apply_azimuthal_weighting are mutually exclusive`**
Enable only one weighting scheme.

**`Unknown raw_data_visualization.plots_to_generate entries`**
A key in `plots_to_generate` is misspelled; the error lists the valid keys.

**Geometry stage produces zero sources**
All picks were rejected by the apparent-velocity filter or the phase filter. Widen `p_velocity_range_m_s` / `s_velocity_range_m_s`, or change `phase_type` from `both` to `p` or `s` to find the phase that causes the rejections.

**Velocity model binary size does not match LATTE**
LATTE reports a dimension mismatch. Make sure `coordinate_transform.grid_spacing_km`, the grid bounds and `initial_velocity_model.max_depth_km` produce the `(nx, ny, nz)` set in the LATTE parameter file.

**Large grid runs out of memory**
Lower `parallel.vectorize_max_points` to switch to chunked processing sooner, or lower `parallel.grid_chunk_size`.

---

## 6. Best Practices

- **Convert the catalog to HDF5 once.** Parsing QuakeML is the slowest step. Run `hdf5_conversion` with `preprocessing.run_pipeline=false`, then work from the HDF5 file with `data.use_hdf5=true`.
- **Start with a small test run.** Use `data.max_files=50` and `parallel.num_workers=4` to validate the configuration before processing the full catalog.
- **Look at the raw catalog first.** The raw-data plots show the magnitude, distance and azimuthal-gap distributions, which guide the filter settings.
- **Keep one named config per study area.** Copy `example.yaml`, name it after the study, and keep it under version control, so runs are reproducible and CLI override chains stay short.
- **Check the pick acceptance rate.** The `Pick filtering summary` block shows how many picks were accepted. A low rate (< 50%) suggests that the velocity ranges or the filters need adjustment.
- **Use a fixed, physically meaningful origin time.** With `origin_time.mode: fixed`, a large constant offset (for example 50 s) separates the physical travel time from the stored value and makes debugging easier.
- **Prefer azimuthal weighting for uneven station coverage.** It reduces directional bias; do not combine it with `apply_default_weights`.
- **Update `ns` in the LATTE parameter file** after the geometry step, using the value printed in the log.
- **Keep `output_format: both` for the velocity model** during development: the CSV is easy to inspect, the HDF5 file is fast to read.
- **Process large catalogs year by year** with `data.years` to find year-specific data-quality problems.
