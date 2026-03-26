# Preprocess Pipeline

The preprocess pipeline transforms raw seismic data (QuakeML or HDF5 catalog) into the binary and tabular inputs required by the LATTE tomography inversion kernel. It covers data loading and filtering, coordinate transformation, geometry and travel time file generation, initial velocity model creation, and optional visualization.

> **Configuration reference:** See [`configs/preprocess/README.md`](../configs/preprocess/README.md) for a complete description of every configuration parameter and all required file paths.

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

The pipeline is driven by `preprocess/main.py` and executes the following modules in order.

---

### 1.1 Data Loading (`data_loader.py`)

Loads seismic events, stations, picks, and arrivals from either an HDF5 file or a directory of QuakeML (`.qml`) files, then applies all configured filters.

**What it does:**

- Reads the station inventory from a whitespace-delimited text file (`data.station_file`).
- If `data.use_hdf5: true`, reads events, stations, arrivals, and optionally picks from an HDF5 file via `pd.read_hdf()`.
- If `data.use_hdf5: false`, scans `data.quakeml_dir` for year-named subdirectories (e.g., `2020/`, `2021/`) or `.qml` files directly in the root directory, then processes them in parallel using `multiprocessing.Pool`.
- Applies filters in order: magnitude → geographic bounding box → depth → hypocentral/epicentral distance → manual picks only (if `filters.manual_picks_only: true`).
- Computes epicentral and hypocentral distances for every event–station pair.

**Key configuration:**

| Key | Purpose |
|---|---|
| `data.use_hdf5` | Switch between HDF5 and QuakeML input |
| `data.hdf5_file` | Path to the HDF5 catalog |
| `data.quakeml_dir` | Root directory of QuakeML files |
| `data.station_file` | Station inventory (ID lat lon elevation_m) |
| `data.years` | Restrict to a list of years (null = all) |
| `data.max_files` | Cap the number of QML files processed |
| `parallel.num_workers` | Worker count; null uses all available cores |
| `filters.*` | Magnitude, geographic, depth, distance bounds |

---

### 1.2 Coordinate Transformation (`coordinate_transformation.py`)

Converts geodetic coordinates (longitude, latitude, depth in km) to a local Cartesian frame (x east, y north/south, z down in km) using a first-order flattening transformation.

**What it does:**

- Sets the Cartesian origin at `(origin_lon, origin_lat, origin_depth_km)`.
- Computes:
  - `x = R · (λ − λ₀) · cos(φ₀)`
  - `y = ±R · (φ − φ₀)` (sign controlled by `invert_y_axis`)
  - `z = depth − origin_depth_km`
- Produces separate CSV files for events, stations, and picks in the Cartesian frame.
- Optionally transforms basin boundary polygons from geodetic to Cartesian for use in visualization overlays.
- Writes grid indices (`i`, `j`, `k`) alongside coordinates for downstream modules.

**Key configuration:**

| Key | Purpose |
|---|---|
| `coordinate_transform.enabled` | Enable/disable this stage |
| `coordinate_transform.origin_lon/lat/depth_km` | Cartesian origin |
| `coordinate_transform.grid_spacing_km` | `[dx, dy, dz]` — consumed by velocity model |
| `coordinate_transform.invert_y_axis` | If true, y increases southward |
| `coordinate_transform.basin_boundaries.data_dir` | Directory of basin polygon CSVs |

---

### 1.3 Geometry and Travel Time Generation (`geometry_traveltime.py`)

Produces the geometry text files, travel time binary files, and source binary files (`sx.bin`, `sy.bin`, `sz.bin`, `st0.bin`) consumed by the LATTE inversion kernel.

**What it does:**

- Parses picks and matches them to Cartesian event and station coordinates.
- Filters picks by apparent velocity range to reject physically unrealistic arrivals (default P: 500–10 000 m/s; S: 500–6 000 m/s).
- Applies phase-type filtering: `both` requires co-located P and S picks at the same station; `p` or `s` uses only the specified phase.
- Optionally applies **global-scale azimuthal weighting**: a two-pass scheme that counts picks per (station, 30° azimuthal bin) across all events and assigns weights of 1/√N or 1/N to reduce directional bias.
- Writes one geometry text file and one pair of travel time binary files (float32, little-endian) per source event.
- Writes four source binary files containing updated event coordinates and origin times.
- Writes a `shot_to_source_id_map.txt` that links integer shot indices to original event IDs.
- Optionally writes a human-readable `ground_truth_travel_times.csv`.

> **Important:** Binary travel time files contain `physical_travel_time + configured_origin_time`. The CSV `observed_tt` column contains physical travel times only. Keep this distinction in mind when post-processing.

> **Important:** After this step, update the `ns` parameter in the LATTE Fortran input file to match the number of sources reported in the log.

**Key configuration:**

| Key | Purpose |
|---|---|
| `geometry_traveltime.enabled` | Enable/disable this stage |
| `geometry_traveltime.phase_type` | `both` / `p` / `s` |
| `geometry_traveltime.p_velocity_range_ms` | Acceptable apparent P-velocity range (m/s) |
| `geometry_traveltime.s_velocity_range_ms` | Acceptable apparent S-velocity range (m/s) |
| `geometry_traveltime.apply_azimuthal_weighting` | Enable global azimuthal weighting |
| `geometry_traveltime.azimuthal_bin_size_deg` | Azimuthal bin width (degrees) |
| `geometry_traveltime.generate_travel_time_csv` | Write the human-readable CSV |
| `origin_time.mode` | `fixed` (constant offset) or `unix_epoch` |
| `origin_time.fixed_value` | Origin time value (seconds) when mode is `fixed` |

---

### 1.4 Initial Velocity Model (`initial_velocity_model.py`)

Extrudes a one-dimensional depth–velocity profile onto a three-dimensional Cartesian grid to produce the starting model for the inversion.

**What it does:**

- Reads a 1D velocity model CSV with columns `depth, vp, vs` (or `depth, radius, vp, vs`).
- Constructs a regular 3D Cartesian grid. Grid bounds are inferred automatically from the transformed event and station coordinates, or set manually via configuration.
- Performs linear interpolation of Vp and Vs at each grid node depth.
- For large grids (> `vectorize_max_points`, default 100 M points), uses a chunked strategy to avoid excessive memory consumption.
- Optionally interpolates over any internal grid gaps.
- Writes the model in the formats specified by `output_format`: CSV, HDF5, or both.
- Always writes float32 binary files (`vp_init.bin`, `vs_init.bin`) in Fortran-ordered layout (`[nz, ny, nx]`).

**Key configuration:**

| Key | Purpose |
|---|---|
| `initial_velocity_model.enabled` | Enable/disable this stage |
| `initial_velocity_model.initial_model_file` | Path to 1D velocity CSV |
| `initial_velocity_model.max_depth_km` | Maximum depth of the model domain |
| `initial_velocity_model.output_format` | `csv` / `hdf5` / `both` |
| `initial_velocity_model.auto_bounds` | Infer grid extent from data extent |
| `parallel.vectorize_max_points` | Threshold for switching to chunked mode |
| `parallel.grid_chunk_size` | Chunk size for chunked vectorization |

---

### 1.5 Visualization (`visualization.py`, `ray_density_visualization.py`)

Generates diagnostic plots to verify the processed data before inversion.

**What it does:**

- **Station/event maps:** geographic map of events (colored by depth, sized by magnitude) and stations, plus vertical cross-sections in the XZ and YZ planes. Optionally overlays basin boundary polygons.
- **Velocity model plots:** depth-slice contour maps of Vp and Vs at specified depths, and 1D profiles along latitude or longitude.
- **Ray density analysis:** traces straight-line rays between every accepted event–station pair through the 3D grid using a DDA (Digital Differential Analyzer) algorithm. Produces XY, XZ, and YZ density projections and a per-depth coverage map.

**Key configuration:**

| Key | Purpose |
|---|---|
| `visualization.enabled` | Master switch |
| `visualization.station_event_vis.enabled` | Station/event maps |
| `visualization.velocity_model_vis.enabled` | Velocity model plots |
| `visualization.ray_density.enabled` | Ray density analysis |
| `visualization.ray_density.coverage_threshold` | Minimum ray count to mark a cell as covered |
| `visualization.figure_dpi` | Output figure DPI |

---

## 2. Outputs

All outputs are written under `preprocessing.output_dir`. The table below summarises the primary outputs; see [`docs/file_formats.md`](file_formats.md) for complete column/key definitions and units.

| File | Format | Stage | Description |
|---|---|---|---|
| `events_geodetic.csv` | CSV | Data loading | Filtered events in geographic coordinates |
| `stations_geodetic.csv` | CSV | Data loading | Filtered stations in geographic coordinates |
| `events_cartesian.csv` | CSV | Coord. transform | Events in Cartesian km |
| `stations_cartesian.csv` | CSV | Coord. transform | Stations in Cartesian km |
| `picks_cartesian.csv` | CSV | Coord. transform | Picks with Cartesian coordinates |
| `geometry/shot_N_geometry.txt` | Text | Geom./TT | Per-event geometry file for LATTE |
| `traveltime/shot_N_traveltime_{p,s}.bin` | Binary (float32) | Geom./TT | Per-event travel time arrays |
| `source/sx.bin`, `sy.bin`, `sz.bin`, `st0.bin` | Binary (float32) | Geom./TT | Source coordinate and origin time arrays |
| `shot_to_source_id_map.txt` | Text | Geom./TT | Shot index ↔ event ID mapping |
| `ground_truth_travel_times.csv` | CSV | Geom./TT | Human-readable observed travel times |
| `velocity_model_cartesian.csv` | CSV | Velocity model | 3D grid with Vp and Vs |
| `velocity_model_cartesian.h5` | HDF5 | Velocity model | Same data with compression |
| `vp_init.bin`, `vs_init.bin` | Binary (float32) | Velocity model | LATTE-ready velocity binaries |
| `plots/` | PNG | Visualization | Maps, cross-sections, depth slices, ray density |

---

## 3. Usage and Configuration

### Running the pipeline

```bash
# Use the default configuration
python -m preprocess.main

# Use a named configuration file (e.g., configs/preprocess/example.yaml)
python -m preprocess.main --config-name example

# Use a region-specific configuration
python -m preprocess.main --config-name delaware_basin_config_1
```

Hydra writes all run outputs (config snapshot, logs) to a timestamped directory:

```
outputs/YYYY-MM-DD/HH-MM-SS_preprocess_<config_name>/
```

### CLI overrides

Any configuration key can be overridden at the command line using Hydra dot-notation:

```bash
# Switch to HDF5 input
python -m preprocess.main data.use_hdf5=true data.hdf5_file=/data/catalog.h5

# Override origin and grid spacing
python -m preprocess.main \
  coordinate_transform.origin_lon=-104.5 \
  coordinate_transform.origin_lat=31.8 \
  coordinate_transform.origin_depth_km=0.0 \
  coordinate_transform.grid_spacing_km="[2.0, 2.0, 1.0]"

# Restrict by magnitude and depth
python -m preprocess.main filters.magnitude_min=2.0 filters.depth_max=40.0

# Limit files for a quick test
python -m preprocess.main data.max_files=50 parallel.num_workers=4

# Enable only visualization (skips all processing stages)
python -m preprocess.main \
  coordinate_transform.enabled=false \
  geometry_traveltime.enabled=false \
  initial_velocity_model.enabled=false \
  visualization.enabled=true

# Change log verbosity
python -m preprocess.main preprocessing.log_level=DEBUG
```

### Minimal configuration checklist

The following keys must be set before the pipeline can run:

```yaml
preprocessing:
  output_dir: /path/to/output

data:
  use_hdf5: false
  quakeml_dir: /path/to/qml_files     # or hdf5_file if use_hdf5: true
  station_file: /path/to/stations.txt

coordinate_transform:
  origin_lon: -104.5
  origin_lat: 31.8
  origin_depth_km: 0.0

initial_velocity_model:
  initial_model_file: /path/to/1d_model.csv
```

Refer to `configs/preprocess/example.yaml` for a working minimal configuration template.

---

## 4. Logging

The pipeline uses Python's `logging` module throughout. The log level is controlled by `preprocessing.log_level` (default: `INFO`). Valid values are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.

| Level | What is logged |
|---|---|
| `DEBUG` | Full YAML configuration dump; per-pick velocity rejection reasons; coordinate ranges after transformation |
| `INFO` | Stage start/end; file counts; filter results (before vs. after counts); grid dimensions; velocity ranges; output file paths |
| `WARNING` | Uneven azimuthal bins; missing optional files; stations not found in inventory; picks with negative travel times; large grid memory warnings |
| `ERROR` | Configuration validation failures; invalid parameter combinations |

Log messages from each module are prefixed with the module name (e.g., `preprocess.data_loader`, `preprocess.geometry_traveltime`).

Hydra also writes a copy of the resolved configuration and the console output to the timestamped run directory under `outputs/`.

Key console messages to watch for:

```
# Geometry/TT summary — confirm pick acceptance rates
Pick filtering summary:
  Events parsed:              N in -> M with valid picks
  Picks accepted:             N
  Picks rejected (negative travel time): N
  Picks rejected (unrealistic velocity): N

# Important — Fortran input update required
IMPORTANT: Update 'ns' parameter in Fortran input to N
```

---

## 5. Troubleshooting

**No events after filtering**
Check filter bounds against the catalog extent. Run with `preprocessing.log_level=DEBUG` to see per-filter counts. Temporarily set all filters to their widest valid range to confirm data loads correctly, then tighten incrementally.

**Station not found for arrival**
The station ID in the arrival record does not match any entry in `data.station_file`. Verify that the station file uses the same naming convention as the catalog. Check for trailing whitespace or case mismatches.

**`ValueError: origin_time.mode must be 'fixed' or 'unix_epoch'`**
The `origin_time.mode` key is misspelled or set to an unsupported value. The only valid options are `fixed` and `unix_epoch`.

**`ValueError: apply_default_weights and apply_azimuthal_weighting are mutually exclusive`**
Only one weighting scheme can be active at a time. Set the other to `false`.

**Geometry stage produces zero sources**
All picks were rejected by the velocity range filter or the phase-type filter. Widen `p_velocity_range_ms` / `s_velocity_range_ms`, or switch `phase_type` from `both` to `p` or `s` to identify which phase is causing rejections.

**Velocity model binary has wrong size**
The LATTE Fortran kernel reports a dimension mismatch. Ensure `coordinate_transform.grid_spacing_km` and `initial_velocity_model.max_depth_km` produce the same `(nx, ny, nz)` that is configured in the Fortran input file.

**Large grid runs out of memory**
Reduce `parallel.vectorize_max_points` to trigger the chunked path sooner, or reduce `parallel.grid_chunk_size` to process smaller chunks at a time.

**Ray density visualization is skipped**
`coordinate_transform.enabled` must be `true` for ray density analysis to run. If coordinate transformation has already been run and CSVs exist, set `coordinate_transform.enabled=true` and all other stages to `false` to reuse the existing data for visualization.

---

## 6. Best Practices

- **Start with a small test run.** Use `data.max_files=50` and `parallel.num_workers=4` to validate the configuration before processing the full catalog.
- **Use named config files for each region.** Copy `example.yaml`, name it after the study area, and commit it to version control. This makes runs reproducible and avoids long CLI override chains.
- **Always verify pick counts before inversion.** The "Pick filtering summary" block in the log confirms how many picks were accepted. A low acceptance rate (< 50%) suggests filter settings or velocity range thresholds need adjustment.
- **Set `origin_time.mode: fixed` with a physically meaningful offset.** Using a large fixed offset (e.g., 50 s) separates the physical travel time from the encoded value and makes debugging easier.
- **Do not mix `apply_default_weights` and `apply_azimuthal_weighting`.** They are mutually exclusive; using azimuthal weighting is recommended when the azimuthal distribution of stations is uneven.
- **Confirm `ns` in the Fortran input file.** After the geometry/TT stage completes, the log prints the required `ns` value. Update the LATTE Fortran input before launching the inversion.
- **Keep `output_format: both`** for the velocity model during development so you have both a human-readable CSV and a fast HDF5 file available for inspection.
- **Use `data.years` to process catalogs incrementally.** For large multi-year catalogs, process one year at a time to diagnose year-specific data quality issues.
