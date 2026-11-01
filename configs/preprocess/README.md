# Preprocessing Configuration

## Overview

The preprocessing pipeline prepares seismic data for [LATTE](https://github.com/lanl/latte_traveltime) tomography. It reads seismic event data from **QuakeML** or **HDF5** format, applies spatial and magnitude filters, transforms geodetic coordinates (lon/lat/depth) to Cartesian (x/y/z km) via a flattening transformation, and generates the geometry, travel time, and initial velocity model files required by the inversion.

**Pipeline steps (in order):**
1. Load data (QuakeML or HDF5)
2. Transform coordinates (geodetic → Cartesian)
3. Create geometry and travel time files
4. Create initial 3D velocity model and binaries
5. Generate visualizations

---

## Quick Start

### Minimal configuration

Copy `example.yaml` and update the file paths for your dataset:

```yaml
# configs/preprocess/example.yaml
defaults:
  - default
  - _self_

preprocessing:
  output_dir: "/path/to/output"

data:
  use_hdf5: true
  hdf5_file: "/path/to/seismic_data.h5"
  quakeml_dir: "/path/to/quakeml"        # used when use_hdf5 is false
  station_file: "/path/to/station_list.txt"
  output_base_dir: "/path/to/output"

coordinate_transform:
  origin_lon: -108.0
  origin_lat: 34.5
  origin_depth_km: 0.0

initial_velocity_model:
  initial_model_file: "/path/to/initial_model.csv"
```

All other parameters inherit their defaults from `default.yaml`.

### Run

```bash
python3 -m preprocess.main --config-name example
```

### CLI overrides

Any key can be overridden at the command line using Hydra dot-notation without modifying the YAML file:

```bash
python3 -m preprocess.main --config-name example \
    data.use_hdf5=false \
    data.quakeml_dir=/new/path/to/quakeml \
    filters.magnitude_min=2.0 \
    filters.depth_max=30.0
```

---

## Configuration Reference

### `preprocessing`

| Key | Default | Description |
|-----|---------|-------------|
| `output_dir` | `"/path/to/output/files"` | Root output directory for all generated files. |
| `log_level` | `INFO` | Logging verbosity. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |

---

### `data`

Controls the data source and loading behavior.

| Key | Default | Description |
|-----|---------|-------------|
| `use_hdf5` | `true` | If `true`, load from HDF5; if `false`, load from QuakeML directory. |
| `hdf5_file` | — | Path to the `.h5` seismic data file. Used when `use_hdf5: true`. |
| `quakeml_dir` | — | Directory containing QuakeML files. Used when `use_hdf5: false`. |
| `station_file` | — | Path to the station list (`.txt`). Used when `use_hdf5: false`. |
| `output_base_dir` | — | Base directory from which all output sub-paths are derived. |
| `max_files` | `null` | Limit number of QuakeML files loaded. `null` = all files. Useful for testing. |
| `years` | `null` | Year subdirectories to scan (e.g., `[2020, 2021]`). `null` = auto-detect. |
| `use_preferred_origin` | `true` | Use the preferred origin from each event. |
| `use_preferred_magnitude` | `true` | Use the preferred magnitude from each event. |
| `save_all_picks` | `false` | If `false`, only picks associated with the selected origin are saved. |

---

### `parallel`

Controls parallel processing behavior.

| Key | Default | Description |
|-----|---------|-------------|
| `num_workers` | `null` | Number of worker processes. `null` = all CPU cores. |
| `chunk_size` | `100` | Events per chunk for pick parsing. |
| `grid_chunk_size` | `50000` | Grid points per chunk for velocity model creation. |
| `vectorize_max_points` | `100000000` | Max grid points for full vectorization (~6 GB RAM at 100 M). |

---

### `filters`

All filters default to `null` (no filter applied).

| Key | Unit | Description |
|-----|------|-------------|
| `manual_picks_only` | — | If `true`, only picks with `evaluation_mode='manual'` are kept. |
| `magnitude_min` / `magnitude_max` | — | Magnitude range filter. |
| `lat_min` / `lat_max` | degrees | Latitude bounding box. |
| `lon_min` / `lon_max` | degrees | Longitude bounding box. |
| `depth_min` / `depth_max` | km | Depth range filter. |
| `hypocentral_distance_min` / `hypocentral_distance_max` | km | 3D source–receiver distance filter. Requires station coordinates. |
| `epicentral_distance_min` / `epicentral_distance_max` | km | 2D surface source–receiver distance filter. Requires station coordinates. |

---

### `origin_time`

Controls how event origin times are encoded in output files.

| Key | Options | Description |
|-----|---------|-------------|
| `mode` | `fixed`, `unix_epoch` | `fixed`: all origins set to `fixed_value`; `unix_epoch`: use actual Unix timestamps. |
| `fixed_value` | — (seconds) | Value assigned to all origin times when `mode: fixed`. Default `0.0`. |

---

### `coordinate_transform`

Converts geodetic (lon/lat/depth) coordinates to a local Cartesian frame (x/y/z km) using a flattening transformation.

| Key | Default | Unit | Description |
|-----|---------|------|-------------|
| `enabled` | `true` | — | Enable/disable this step. |
| `origin_lon` | `-108.0` | degrees | Longitude of the Cartesian origin. |
| `origin_lat` | `34.5` | degrees | Latitude of the Cartesian origin. |
| `origin_depth_km` | `0.0` | km | Depth of the Cartesian origin (positive = below MSL). |
| `grid_spacing_km` | `[1.0, 1.0, 1.0]` | km | Grid spacing as `[dx, dy, dz]`. |
| `invert_y_axis` | `true` | — | If `true`, +y points south (latitude decreases); if `false`, +y points north. |
| `output_dir` | derived | — | Output directory for coordinate transform CSVs. |

**Output CSVs** (paths derived from `output_dir`):

| Key | Description |
|-----|-------------|
| `events_cartesian_csv` | Events in Cartesian coordinates (post-filtering). |
| `stations_cartesian_csv` | Stations in Cartesian coordinates (post-filtering). |
| `picks_cartesian_csv` | Picks in Cartesian coordinates (post-filtering). |
| `events_geodetic_csv` | Events in geodetic coordinates (post-filtering). |
| `stations_geodetic_csv` | Stations in geodetic coordinates (post-filtering). |

**`basin_boundaries`** (optional, disabled by default): Reprojects basin boundary CSV files from geographic to Cartesian coordinates. Set `enabled: true` and provide `data_dir` with a list of `basins` entries (`file`, `label`).

---

### `geometry_traveltime`

Generates per-event geometry and ground-truth travel time files for the tomography inversion.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable/disable this step. |
| `geometry_dir` | derived | Output directory for geometry binary files. |
| `traveltime_dir` | derived | Output directory for ground-truth travel time files. |
| `source_dir` | derived | Output directory for source (event) binary files. |
| `generate_travel_time_csv` | `true` | Write a human-readable travel time CSV in addition to binaries. |
| `csv_output_dir` | derived | Output directory for travel time CSVs. |
| `include_magnitude_in_csv` | `false` | Append a magnitude column to the travel time CSV. |
| `generate_source_binaries` | `true` | Write source location binary files. |
| `apply_default_weights` | `false` | Assign a uniform weight to all picks. Mutually exclusive with `apply_azimuthal_weighting`. |
| `default_weight` | `1.0` | Weight value used when `apply_default_weights: true`. |
| `apply_azimuthal_weighting` | `false` | Weight picks by azimuthal coverage. Mutually exclusive with `apply_default_weights`. |
| `azimuthal_bin_size_deg` | `30` | degrees | Azimuthal bin width. 30° produces 12 bins. Must divide 360 evenly. |
| `azimuthal_weight_formula` | `inverse_sqrt` | Weight formula: `inverse_sqrt` = 1/√N, `inverse` = 1/N. |
| `phase_type` | `both` | Phases to process: `both`, `p`, or `s`. |
| `p_velocity_range_m_s` | `[500, 10000]` | m/s | Apparent P-velocity range. Picks outside this range are discarded. |
| `s_velocity_range_m_s` | `[500, 6000]` | m/s | Apparent S-velocity range. Picks outside this range are discarded. |

---

### `initial_velocity_model`

Builds a 3D velocity model by extruding a 1D depth profile (CSV) onto the Cartesian grid.

| Key | Default | Unit | Description |
|-----|---------|------|-------------|
| `enabled` | `true` | — | Enable/disable this step. |
| `initial_model_file` | — | — | Path to the 1D initial velocity model CSV. |
| `max_depth_km` | `100.0` | km | Model entries below this depth are ignored. |
| `use_auto_bounds` | `true` | — | If `true`, grid bounds are computed from the transformed events/stations. |
| `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, `z_max` | `null` | km | Manual Cartesian grid bounds; used only when `use_auto_bounds: false`. |
| `output_format` | `csv` | — | Output format: `csv`, `hdf5`, or `both`. |
| `output_csv` | derived | — | Path for the Cartesian velocity model CSV. |
| `output_hdf5` | derived | — | Path for the Cartesian velocity model HDF5 file. |
| `vp_binary` | derived | — | Path for the Vp binary file. |
| `vs_binary` | derived | — | Path for the Vs binary file. |
| `interpolate` | `true` | — | Interpolate velocity values onto the grid. |
| `neighbor_threshold` | `5` | — | Minimum number of neighbors for interpolation. |
| `phase_type` | `both` | — | Which phase binaries to write: `both`, `p`, or `s`. |

**`hdf5_options`** (used when `output_format` is `hdf5` or `both`):

| Key | Default | Description |
|-----|---------|-------------|
| `compression` | `gzip` | HDF5 compression: `gzip`, `lzf`, or `null`. |
| `compression_level` | `4` | Gzip compression level (0–9). |
| `parallel_processing` | `true` | Use parallel processing for HDF5 writes. |
| `n_jobs` | `-1` | Number of parallel jobs: `-1` = all cores. |
| `create_coarse_dataset` | `false` | Write a coarse-grid dataset inside the HDF5 file. |
| `write_coarse_csv` | `false` | Write a separate coarse-grid CSV file. |
| `coarse_grid` | `dx/dy/dz: 2.0` | km | Coarse grid spacing. Must be integer multiples of the fine grid spacing. |

---

### `visualization`

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Master switch for all visualizations. |

**`station_event_vis`** — Map view and cross-section plots of stations and events.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable station/event visualization. |
| `output_dir` | derived | Output directory for figures. |
| `figure_dpi` | `150` | Figure resolution in DPI. |
| `figure_size` | `[12, 10]` | Figure size in inches `[width, height]`. |
| `plot_pick_lines` | `true` | Draw source–receiver ray path lines. |
| `pick_alpha` | `0.3` | Transparency of pick lines (0 = transparent, 1 = opaque). |

**`velocity_model_vis`** — Depth slice and 1D profile plots of the velocity model (disabled by default).

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Enable velocity model visualization. |
| `plot_depth_slices` | `true` | Plot Vp/Vs contourf maps at specified depths. |
| `depth_slices` | `[5,10,20,30,40,50]` | km | Depths at which to plot slices. |
| `plot_profiles` | `true` | Plot 1D velocity profiles. |
| `profile_type` | `both` | Profile orientation: `lat`, `lon`, or `both`. |

---

## Output Structure

```
output_base_dir/
├── coordinate_transform/
│   ├── events_cartesian.csv
│   ├── stations_cartesian.csv
│   └── picks_cartesian.csv
├── events_geodetic.csv
├── stations_geodetic.csv
├── geometry/                        # Per-event geometry binary files
├── ground_truth_traveltimes/        # Per-event travel time binary files
├── model/
│   ├── vp_init.bin
│   ├── vs_init.bin
│   └── <source binaries>
├── velocity_model_init.csv     # (or .h5)
├── geometry_csv/                    # Human-readable travel time CSVs
└── visualizations/
    ├── <station_event plots>
    └── <velocity_model plots>
```

Hydra also writes a timestamped run log to:
```
outputs/<YYYY-MM-DD>/<HH-MM-SS>_preprocess_<config_name>/
```

---

## Important Notes

- **`apply_default_weights` and `apply_azimuthal_weighting` are mutually exclusive.** Enabling both will raise a `ValueError` at startup.
- **Filtered data is authoritative.** The CSV files saved under `coordinate_transform/` contain only events, stations, and picks that passed all geometry and velocity-range filters. Visualizations use this filtered data.
- **`coarse_grid` spacing must be integer multiples of the fine grid spacing** (`coordinate_transform.grid_spacing_km`). Non-integer multiples will produce incorrect coarse grids.
- **`azimuthal_bin_size_deg` should evenly divide 360.** Non-even divisors are allowed but will produce uneven bin coverage (a warning is logged).
- **`max_files`** is intended for development and testing only. Leave as `null` for production runs.
