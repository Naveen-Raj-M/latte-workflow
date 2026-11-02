# Preprocessing Configuration

## Overview

The preprocessing pipeline prepares seismic data for [LATTE](https://github.com/lanl/latte_traveltime) tomography. It reads a catalog from **QuakeML** or **HDF5**, filters it, transforms geodetic coordinates (lon/lat/depth) to a Cartesian frame (x/y/z in km), and writes the geometry, travel time and initial velocity model files required by the inversion.

**Steps (in order, each with its own switch):**
1. Parse the QuakeML files once (when any enabled step needs them)
2. Raw data visualization (`raw_data_visualization`)
3. QuakeML to HDF5 conversion (`hdf5_conversion`)
4. Load data and apply the global filters (`data`, `filters`)
5. Transform coordinates (`coordinate_transform`)
6. Create geometry and travel time files (`geometry_traveltime`)
7. Create the initial 3D velocity model and binaries (`initial_velocity_model`)
8. Final data visualization (`final_data_visualization`)

`preprocessing.run_pipeline: false` stops after steps 2 and 3.

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
  quakeml_dir: "/path/to/quakeml"        # used when QuakeML is parsed
  station_file: "/path/to/station_list.txt"
  output_base_dir: "/path/to/output"

filters:
  magnitude_min: 2.0

coordinate_transform:
  origin_lon: -104.0
  origin_lat: 31.5
  origin_depth_km: 0.0

initial_velocity_model:
  initial_model_file: "/path/to/initial_model.csv"
```

All other parameters inherit their defaults from `default.yaml`.

### Run

```bash
python -m preprocess.main --config-name example
```

### CLI overrides

Any key can be overridden with Hydra dot-notation:

```bash
python -m preprocess.main --config-name example \
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
| `output_dir` | `"/path/to/output/files"` | Root output directory; the subdirectories `coordinate_transform`, `geometry`, `model` and `visualizations` are created here. |
| `log_level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `run_pipeline` | `true` | `false` skips coordinate transform, geometry/travel times, velocity model and final visualization. Use with `raw_data_visualization` / `hdf5_conversion`. |

---

### `data`

Catalog input and loading behaviour.

| Key | Default | Description |
|-----|---------|-------------|
| `use_hdf5` | `true` | `true` loads the catalog from `hdf5_file`; `false` parses QuakeML. |
| `hdf5_file` | — | HDF5 catalog (`events`, `stations`, `arrivals`, `picks` tables). |
| `quakeml_dir` | — | Directory with the QuakeML files: `.qml` files in the directory itself and/or in four-digit year subdirectories (`<quakeml_dir>/<year>/<file>.qml`). |
| `station_file` | — | Station inventory (`ID latitude longitude elevation_m`). Used when QuakeML is parsed. |
| `output_base_dir` | — | Base directory of all output paths. |
| `max_files` | `null` | Limit the number of QuakeML files (testing). `null` = all. |
| `years` | `null` | Year subdirectories to read, for example `[2020, 2021]`. `null` = auto-detect. |
| `catalog_label` | `"catalog_1"` | Value written to the `catalog` column. |
| `use_preferred_origin` | `true` | Use the preferred origin of each event. |
| `origin_index` | `0` | Origin used when `use_preferred_origin` is `false`. |
| `use_preferred_magnitude` | `true` | Use the preferred magnitude of each event. |
| `magnitude_index` | `0` | Magnitude used when `use_preferred_magnitude` is `false`. |
| `save_all_picks` | `false` | `false` keeps only the picks of the selected origin (pipeline). |

#### `data.secondary_catalogs`

Optional additional QuakeML catalogs appended to the same tables (rows are distinguished by the `catalog` column). Station codes that occur in several catalogs are merged by location: within `identical_tolerance_m` they are one shared station; closer than `match_tolerance_m` they are renamed only if `suffix_near_duplicate_stations` is `true`; farther apart they are renamed if `suffix_conflicting_stations` is `true`. Only stations of the additional catalogs are renamed, by appending `station_suffix`.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Load the additional catalogs. |
| `quakeml_dir` | `[…]` | List of QuakeML directories, one per additional catalog. |
| `station_file` | `[…]` | List of station files, same order. |
| `catalog_label` | `["catalog_2"]` | List of labels. |
| `station_suffix` | `["_catalog_2"]` | List of suffixes for renamed stations. |
| `years`, `max_files` | `null` | As in `data`; shared by all additional catalogs. |
| `identical_tolerance_m` | `1.0` | m; same-location tolerance. |
| `match_tolerance_m` | `200.0` | m; near-duplicate tolerance. |
| `suffix_near_duplicate_stations` | `false` | Rename near-duplicate stations. |
| `suffix_conflicting_stations` | `true` | Rename stations farther apart than `match_tolerance_m`. |

---

### `parallel`

| Key | Default | Description |
|-----|---------|-------------|
| `num_workers` | `null` | Worker processes. `null` = all CPU cores. |
| `chunk_size` | `100` | Events per chunk for pick parsing. |
| `grid_chunk_size` | `50000` | Grid points per chunk for the velocity model. |
| `vectorize_max_points` | `100000000` | Largest grid processed in one vectorized pass (about 6 GB RAM at 100 M). |

---

### `filters`

Global filters of the pipeline. All default to `null` / `false` (no filter).

| Key | Unit | Description |
|-----|------|-------------|
| `manual_picks_only` | — | Keep only picks with `evaluation_mode='manual'`. |
| `magnitude_min` / `magnitude_max` | — | Magnitude range. |
| `lat_min` / `lat_max` | degrees | Latitude bounding box. |
| `lon_min` / `lon_max` | degrees | Longitude bounding box. |
| `depth_min` / `depth_max` | km | Depth range. |
| `hypocentral_distance_min` / `hypocentral_distance_max` | km | 3D source–receiver distance. Needs station coordinates. |
| `epicentral_distance_min` / `epicentral_distance_max` | km | Surface source–receiver distance. Needs station coordinates. |

---

### `raw_data_visualization`

Plots of the parsed QuakeML catalog. Always reads QuakeML, independent of `data.use_hdf5`.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Enable this step. |
| `output_dir` | `${data.output_base_dir}/raw_data_visualization` | Output directory. |
| `filters.*` | global `filters` | Filters of this step only; each key defaults to the global filter of the same name. |
| `filters.require_both_p_and_s` | `false` | Keep only event–station pairs with both P and S arrivals. |
| `filters.azimuthal_gap_max` | `null` | degrees; maximum azimuthal gap. |
| `filters.min_picks_per_event`, `max_picks_per_event` | `null` | Bounds on the number of picks per event. |
| `plots_to_generate.<plot>` | see below | Switch for each plot. |
| `source_receiver_maps.plot_picks` | `true` | Also show receivers that only have picks. |
| `histogram_temporal.time_binning` | `"monthly"` | `daily`, `weekly`, `monthly`, `yearly`. |
| `distance_histograms.plot_type` | `"both"` | `histogram`, `cumulative`, `both`. |
| `distance_histograms.enable_magnitude_bin` | `false` | One histogram per magnitude bin. |
| `distance_histograms.magnitude_bin_size` | `0.2` | Bin width; bins start at `filters.magnitude_min` (0 if `null`). |
| `histogram_magnitude_picks.plot_type` | `"both"` | `histogram`, `cumulative`, `both`. |
| `plots` | `pick_lines.enabled: false` | Overrides of the global `plots` section for this step. |

`plots_to_generate` keys (default `true` for `source_receiver_maps`, `false` for the others): `source_receiver_maps`, `histogram_source_receivers`, `histogram_temporal`, `histogram_magnitude`, `histogram_epicentral_distance`, `histogram_hypocentral_distance`, `histogram_picks_per_event`, `cumulative_azimuthal_gap`, `histogram_magnitude_picks`.

---

### `hdf5_conversion`

Writes the filtered QuakeML catalog to the HDF5 file read by `data.use_hdf5`. Runs before the pipeline.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Enable this step. |
| `output_file` | `${data.hdf5_file}` | HDF5 file to write. |
| `filters.*` | global `filters` | Filters of this step only (same keys as `filters`). |
| `save_all_picks` | `true` | `false` keeps only picks referenced by an arrival. |
| `single_pick_per_phase` | `true` | Keep one P-type and one S-type pick per event–station pair (exact `P`/`S` over `Pg`/`Pn`/…; manual over automatic). |
| `phase_hint_from_arrival` | `true` | Use the arrival's phase when a pick has no phase hint. |
| `require_station_coordinates` | `true` | Drop picks and arrivals of stations missing from the station list. |
| `event_fields` | see `default.yaml` | Columns of the `events` table; `azimuthal_gap` is also available. Empty = all. |
| `station_fields` | `[station_id, latitude, longitude, depth]` | Columns of the `stations` table. |
| `pick_fields` | see `default.yaml` | Columns of the `picks` table. |
| `arrival_fields` | `[]` | Columns of the `arrivals` table. Empty = all. |

---

### `origin_time`

How origin times are encoded in the output files.

| Key | Options | Description |
|-----|---------|-------------|
| `mode` | `fixed`, `unix_epoch` | `fixed`: all origin times set to `fixed_value`; `unix_epoch`: actual Unix timestamps. |
| `fixed_value` | seconds | Value used when `mode: fixed`. Default `0.0`. |

---

### `coordinate_transform`

Converts geodetic coordinates to a local Cartesian frame with a flattening transformation.

| Key | Default | Unit | Description |
|-----|---------|------|-------------|
| `enabled` | `true` | — | Enable this step. |
| `origin_lon` | `-108.0` | degrees | Longitude of the Cartesian origin. |
| `origin_lat` | `34.5` | degrees | Latitude of the Cartesian origin. |
| `origin_depth_km` | `0.0` | km | Depth of the origin (positive below sea level). |
| `grid_spacing_km` | `[1.0, 1.0, 1.0]` | km | `[dx, dy, dz]` of the velocity model grid. |
| `invert_y_axis` | `true` | — | `true`: +y points south; `false`: +y points north. |
| `output_dir` | `${data.output_base_dir}/coordinate_transform` | — | Directory of the Cartesian CSVs. |

**Output CSVs:**

| Key | Default |
|-----|---------|
| `events_cartesian_csv` | `${output_dir}/events_cartesian.csv` |
| `stations_cartesian_csv` | `${output_dir}/stations_cartesian.csv` |
| `picks_cartesian_csv` | `${output_dir}/picks_cartesian.csv` |
| `events_geodetic_csv` | `${data.output_base_dir}/events_geodetic.csv` |
| `stations_geodetic_csv` | `${data.output_base_dir}/stations_geodetic.csv` |

**`basin_boundaries`** (disabled by default): reprojects basin boundary CSVs to Cartesian coordinates and writes them to `<output_dir>/basin_boundaries/`. Set `enabled: true`, `data_dir`, and a list of `basins` entries (`file`, `label`).

---

### `geometry_traveltime`

Writes the per-event geometry files, travel time binaries and source binaries.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable this step. |
| `events_csv`, `stations_csv` | from `coordinate_transform` | Inputs. |
| `geometry_dir` | `${data.output_base_dir}/geometry` | Geometry files, `geometry.txt`, `shot_to_source_id_map.txt`. |
| `traveltime_dir` | `${data.output_base_dir}/ground_truth_traveltimes` | Travel time binaries. |
| `source_dir` | `${data.output_base_dir}/model` | Source binaries. |
| `generate_travel_time_csv` | `true` | Write the human-readable travel time CSV. |
| `csv_output_dir` | `${data.output_base_dir}/geometry_csv` | Directory of the CSV. |
| `include_magnitude_in_csv` | `false` | Add a `magnitude` column to the CSV. |
| `generate_source_binaries` | `true` | Write `sx.bin`, `sy.bin`, `sz.bin`, `st0.bin`. |
| `apply_default_weights` | `false` | Uniform weight for all picks. Mutually exclusive with `apply_azimuthal_weighting`. |
| `default_weight` | `1.0` | Weight used with `apply_default_weights`. |
| `separate_weight_p_s` | `false` | `true`: 5-column geometry files (`x y z weight_p weight_s`); `false`: 4 columns (`x y z weight`). |
| `default_weight_p`, `default_weight_s` | `1.0` | Base P and S weights; applied as they are, or as a scale on the azimuthal weights. |
| `apply_azimuthal_weighting` | `false` | Weight picks by azimuthal coverage (counts per station and bin across all events). Mutually exclusive with `apply_default_weights`. |
| `azimuthal_bin_size_deg` | `30` | degrees; bin width (30° gives 12 bins). Should divide 360. |
| `azimuthal_weight_formula` | `"inverse_sqrt"` | `inverse_sqrt` = 1/√N, `inverse` = 1/N. |
| `phase_type` | `"both"` | `both` (stations with P and S), `p`, or `s`. |
| `p_velocity_range_m_s` | `[500, 10000]` | m/s; picks with an apparent P velocity outside the range are rejected. |
| `s_velocity_range_m_s` | `[500, 6000]` | m/s; same for S. |

---

### `initial_velocity_model`

Builds a 3D model by extruding a 1D depth profile onto the Cartesian grid.

| Key | Default | Unit | Description |
|-----|---------|------|-------------|
| `enabled` | `true` | — | Enable this step. |
| `initial_model_file` | — | — | 1D model CSV (`depth, vp, vs` or `depth, radius, vp, vs`). |
| `max_depth_km` | `100.0` | km | Entries below this depth are ignored. |
| `origin_lon/lat/depth_km`, `grid_spacing_km` | from `coordinate_transform` | — | Inherited. |
| `use_auto_bounds` | `true` | — | Compute the grid bounds from the transformed events and stations. |
| `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, `z_max` | `null` | km | Manual bounds, used when `use_auto_bounds: false`. |
| `output_format` | `"csv"` | — | `csv`, `hdf5`, or `both`. |
| `output_csv` | `${data.output_base_dir}/velocity_model_init.csv` | — | CSV path. |
| `output_hdf5` | `${data.output_base_dir}/velocity_model_init.h5` | — | HDF5 path. |
| `vp_binary`, `vs_binary` | `${data.output_base_dir}/model/vp_init.bin`, `vs_init.bin` | — | LATTE binaries (m/s). |
| `interpolate` | `true` | — | Interpolate velocity values onto the grid. |
| `neighbor_threshold` | `5` | — | Minimum number of neighbours for interpolation. |
| `phase_type` | `"both"` | — | Which binaries to write: `both`, `p`, `s`. |

**`hdf5_options`** (used with `output_format` `hdf5` or `both`):

| Key | Default | Description |
|-----|---------|-------------|
| `compression` | `"gzip"` | `gzip`, `lzf`, or `null`. |
| `compression_level` | `4` | Gzip level 0–9. |
| `parallel_processing` | `true` | Parallel HDF5 preparation. |
| `n_jobs` | `-1` | `-1` = all cores. |
| `show_progress` | `true` | Progress bars. |
| `create_coarse_dataset` | `false` | Add a coarse dataset (`velocity_coarse`) to the HDF5 file. |
| `write_coarse_csv` | `false` | Write a coarse CSV (`output_coarse_csv`). |
| `coarse_grid.dx/dy/dz` | `2.0` | km; integer multiples of the fine spacing. |

---

### `final_data_visualization`

Plots of the data written by the pipeline.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Master switch. |
| `plots` | `{}` | Overrides of the global `plots` section for this step. |

**`station_event_vis`**: map view and cross-sections of stations and events.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable these plots. |
| `output_dir` | `${data.output_base_dir}/visualizations` | Output directory. |

**`velocity_model_vis`** (disabled by default): depth slices and 1D profiles.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Enable these plots. |
| `output_dir` | `${data.output_base_dir}/visualizations` | Output directory. |
| `plot_depth_slices` | `true` | Vp and Vs maps at `depth_slices`. |
| `depth_slices` | `[5, 10, 20, 30, 40, 50]` | km. |
| `plot_profiles` | `true` | 1D profiles. |
| `profile_type` | `"both"` | `lat`, `lon`, or `both`. |

---

### `plots`

Unified plot settings, inherited by `raw_data_visualization` and `final_data_visualization`; each can override any key in its own `plots` node.

| Section | Keys |
|---------|------|
| `figure` | `dpi` (300), `format` (`png`, `pdf`, `svg`), `size` (`[12, 10]`), `histogram_size` (`[12, 7]`). |
| `fontsize`, `fontweight` | Sizes of title, axis labels, ticks, legend, colorbar, statistics; weights of title and axis label. |
| `events` | `marker`, `size`, `color`, `alpha`, `show_colorbar` (events are sized by magnitude and coloured by depth or magnitude). |
| `receivers` | `marker`, `size`, `color`, `picks_only_color`. |
| `pick_lines` | `enabled`, `color`, `alpha`, `linewidth` (one line per event–station pair). |
| `equal_aspect` | Equal physical scale in km on both axes (`false`). |
| `max_depth` | km; deepest depth in cross-sections (`null` = auto). |
| `basin_boundaries` | `enabled`, `show_labels`, `data_dir`, `basins` (`file`, `label`, optional per-basin style), `line_color`, `line_width`, `line_style`, `line_alpha`, `label_color`, `label_fontsize`, `label_fontweight`, `label_offset_x/y`. Shown on map views only. |
| `histograms` | `bins`, `alpha`, `show_statistics`, `rotate_time_labels`, `colors`, `azimuthal_gap` line style. |

---

## Output Structure

```
output_base_dir/
├── <hdf5_file>                      # hdf5_conversion
├── raw_data_visualization/          # raw catalog plots
├── events_geodetic.csv
├── stations_geodetic.csv
├── coordinate_transform/
│   ├── events_cartesian.csv
│   ├── stations_cartesian.csv
│   ├── picks_cartesian.csv
│   └── basin_boundaries/            # optional
├── geometry/                        # shot_N_geometry.txt, geometry.txt, shot_to_source_id_map.txt
├── ground_truth_traveltimes/        # shot_N_traveltime_{p,s}.bin
├── geometry_csv/
│   └── geometry_ground_truth_traveltimes.csv
├── model/
│   ├── sx.bin  sy.bin  sz.bin  st0.bin
│   └── vp_init.bin  vs_init.bin
├── velocity_model_init.csv          # and/or .h5
└── visualizations/
```

Hydra also writes the resolved configuration and the log to:
```
outputs/<YYYY-MM-DD>/<HH-MM-SS>_preprocess_<config_name>/
```

---

## Important Notes

- **`apply_default_weights` and `apply_azimuthal_weighting` are mutually exclusive.** Enabling both raises a `ValueError` at startup.
- **QuakeML is parsed once.** The raw data plots, the HDF5 conversion and the pipeline (with `use_hdf5: false`) share one parse; each applies its own filters.
- **The CSV files in `coordinate_transform/` contain only the data that passed all filters**, and the visualizations use these files.
- **`coarse_grid` spacing must be an integer multiple of `grid_spacing_km`.**
- **`azimuthal_bin_size_deg` should divide 360.** Other values are accepted with a warning and give uneven bins.
- **Geometry files and source binaries are in metres**, velocity binaries in m/s, and the velocity CSV/HDF5 files in km and km/s.
- **`max_files` is meant for tests.** Leave it `null` for production runs.
