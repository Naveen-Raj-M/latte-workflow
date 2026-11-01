# Postprocess Pipeline

The postprocess pipeline converts LATTE inversion binary outputs into human-readable CSV and HDF5 files, computes ray path caches, and derives model coverage diagnostics (Derivative Weight Sum). All five processing stages are independent and disabled by default; enable only the stages relevant to each run.

> **Configuration reference:** See [`configs/postprocess/README.md`](../configs/postprocess/README.md) for a complete description of every configuration parameter and all required file paths.

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

The pipeline is driven by `postprocess/main.py` and executes the following five stages in order. Each stage is guarded by an `enabled` flag and is wrapped in error handling controlled by `general.strict_mode`.

---

### 1.1 Velocity Processing (`postprocess_velocity.py`)

Reads updated Vp and Vs binary files produced by LATTE and projects them onto a reference coordinate grid, writing the result in CSV, HDF5, or both formats.

**What it does:**

- Loads a reference velocity file (the preprocess output CSV or HDF5) to obtain the coarse grid coordinate frame.
- Reads `vp_binary` and `vs_binary` (float32, m/s) and converts units to km/s.
- Operates in one of two modes:
  - **`sample_at_coarse` (default):** For each coarse reference grid point, finds the nearest node on the fine inversion grid and reads the velocity there. Uses `scipy.interpolate.RegularGridInterpolator` as a fallback for non-exact matches.
  - **`expand_to_fine`:** Constructs the full fine grid, interpolates coarse velocity fields onto it, and writes one row per fine grid node.
- Optionally loads gradient and search direction binaries (`grad_vp.bin`, `srch_vp.bin`, etc.) and appends them as extra columns.
- Writes output in the configured format (CSV, HDF5, or both) and optionally generates a coarse-resolution preview CSV.

**Key configuration:**

| Key | Purpose |
|---|---|
| `velocity_processing.enabled` | Enable/disable this stage |
| `velocity_processing.vp_binary` / `vs_binary` | Paths to LATTE output velocity binaries |
| `velocity_processing.reference_velocity_file` | Preprocess CSV or HDF5 reference grid |
| `velocity_processing.model_grid.*` | *Optional.* Additional check of the binary grid. The reference file's metadata is the reference; if only `model_grid` matches the binary size it is used with a warning, and if neither matches an error is raised |
| `velocity_processing.interpolation_mode` | `sample_at_coarse` or `expand_to_fine` |
| `velocity_processing.velocity_output_format` | `csv` / `hdf5` / `both` |
| `velocity_processing.process_gradient` | Load and append gradient binaries |
| `velocity_processing.process_search` | Load and append search direction binaries |

---

### 1.2 Travel Time Processing (`postprocess_traveltime.py`)

Merges observed ground truth travel times with synthetic predictions from the inversion kernel, computing per-pick residuals.

**What it does:**

- Loads the ground truth CSV produced by the preprocess pipeline and builds an O(1) lookup table keyed on `(shot_id, station, phase)`.
- Loads the shot-to-event mapping file (`shot_to_source_id_map.txt`) to link integer shot indices back to original event IDs.
- Reads one synthetic travel time binary per shot per phase. For each binary value:
  1. Subtracts the configured origin time offset (the binary stores `physical_tt + origin_time`).
  2. Optionally applies a per-shot source correction (corrected origin time from `source_correction_file`).
- Supports **source-receiver exchange mode** (`source_receiver_exchange: true`): the pipeline rebuilds LATTE's exchange from `geometry_dir` (virtual sources = unique receivers in first-appearance order, array index = kept shot in `geometry.txt` order), optionally checked against `exchange_geometry_file`, and matches virtual source coordinates to station names before reading the binaries.
- Reports the per-pick residuals (`observed_tt − synthetic_tt`) as a weighted RMS (the residual column is not written).
- Writes a single merged CSV containing both observed and synthetic columns, plus all available metadata.

> **Critical (exchange mode):** The binary array length must equal the number of kept shots and the number of binaries per phase must equal the number of virtual sources. A mismatch raises an error (a warning if `exchange_geometry_file` is given); if LATTE filtered shots or receivers, set `filtered_shot_ids` / `filtered_receiver_ids`.

**Key configuration:**

| Key | Purpose |
|---|---|
| `travel_time_processing.enabled` | Enable/disable this stage |
| `travel_time_processing.ground_truth_csv` | Path to preprocess ground truth CSV |
| `travel_time_processing.synthetic_tt_dir` | Directory of `tt_*.bin` files from LATTE |
| `travel_time_processing.shot_map_file` | `shot_to_source_id_map.txt` |
| `travel_time_processing.phase_type` | `both` / `p` / `s` |
| `travel_time_processing.source_receiver_exchange` | Enable reciprocity mode |
| `travel_time_processing.use_source_correction` | Apply corrected origin times |
| `travel_time_processing.source_correction_file` | Binary file with corrected st0 values |

---

### 1.3 Source Parameter Processing (`postprocess_source.py`)

Reads updated hypocenter coordinates and origin times from LATTE binary files and merges them with the travel time CSV, computing correction magnitudes.

**What it does:**

- Reads four required binary files (`sx_file`, `sy_file`, `sz_file`, `st0_file`) as float32 arrays with one value per shot LATTE kept. Array index k is the k-th shot of `initial_geometry_csv` (ascending shot id) that is not in `filtered_shot_ids`.
- Optionally reads gradient and search direction binaries for all four source parameters.
- Checks that the number of kept shots (`initial_geometry_csv` minus `filtered_shot_ids`) equals the binary size (error otherwise), and that shot IDs in the travel time CSV exist in `initial_geometry_csv`.
- Computes per-shot correction magnitudes: spatial shift (√(Δx² + Δy² + Δz²) in km) and origin time shift (Δst0 in s).
- Appends corrected coordinate columns and optional diagnostic columns to the travel time CSV and writes a new output file.

**Key configuration:**

| Key | Purpose |
|---|---|
| `source_processing.enabled` | Enable/disable this stage |
| `source_processing.sx_file` / `sy_file` / `sz_file` / `st0_file` | Paths to source binary files |
| `source_processing.initial_geometry_csv` | Initial locations (default: `travel_time_processing.ground_truth_csv`); needs `shot_id`, `event_x/y/z`, `origin_time_s` |
| `source_processing.filtered_shot_ids` | Shots LATTE removed (default: `travel_time_processing.filtered_shot_ids`) |
| `source_processing.output_csv` | Output file path |
| `source_processing.process_gradient` | Append source gradient columns |
| `source_processing.process_search` | Append source search direction columns |

---

### 1.4 Ray Path Diagnosis (`compute_ray_paths.py`, `postprocess_raypath.py`, `postprocess_dws.py`)

Reads precomputed ray path binaries from the LATTE forward solver, builds an incremental HDF5 cache, and optionally computes the Derivative Weight Sum (DWS) as a volumetric coverage metric.

**What it does:**

**Cache computation (`compute_ray_paths.py`):**
- Computes an MD5 hash of the critical configuration parameters (grid dimensions, spacing, origin, binary directory) and stores it in the cache. If the config changes, the cache is automatically invalidated.
- Identifies which shots are already cached and processes only new shots, enabling incremental updates.
- Reads one binary ray path file per shot per phase (format: header `nrays`, then `ray_npoints[nrays]`, then concatenated float32 `(x, y, z)` coordinates in meters).
- Saves ray paths to the HDF5 cache in batches, grouped by phase (`/p_wave/shots/<id>/` and `/s_wave/shots/<id>/`).
- Supports event filtering (`event_id_list_file`) and random sub-sampling (`subsample_fraction`) for large datasets.

**Ray tracing from gradients (`postprocess_raypath.py`):**
- When precomputed binary ray paths are unavailable, traces rays using traveltime gradient fields via steepest descent (receiver-to-source).
- Reads per-shot gradient metadata and binary fields (`dT/dx`, `dT/dy`, `dT/dz`).
- Supports nearest-neighbor and trilinear interpolation of gradient values during tracing.

**DWS computation (`postprocess_dws.py`):**
- Accumulates the weighted squared path length of every ray segment within each grid cell: DWS = Σ(w · L²) where w is the pick weight and L is the segment length in km.
- Runs in parallel (configurable backend and worker count).
- Saves the DWS grid to an HDF5 file in velocity model format, with `dws_vp` and `dws_vs` datasets of shape `(nz, ny, nx)`.

**Ray path simplification (`simplify_ray_paths.py`):**
- Optionally reduces the memory footprint of the HDF5 cache by applying the Ramer–Douglas–Peucker (RDP) algorithm to each ray path.
- The tolerance `epsilon` (km) controls aggressiveness: 0.01 km is gentle, 0.10 km is aggressive.

**Key configuration:**

| Key | Purpose |
|---|---|
| `ray_path_diagnosis.enabled` | Enable/disable this stage |
| `ray_path_diagnosis.raypath_binary_dir` | Directory of LATTE ray path binaries |
| `ray_path_diagnosis.ray_path_output_file` | HDF5 cache output path |
| `ray_path_diagnosis.dws_output_file` | DWS HDF5 output path |
| `ray_path_diagnosis.grid.*` | Must match the LATTE inversion grid exactly |
| `ray_path_diagnosis.compute_dws` | Compute DWS after building the cache |
| `ray_path_diagnosis.force_recompute` | Ignore existing cache and recompute |
| `ray_path_diagnosis.fix_corrupted_shots` | Detect and reprocess corrupted cache entries |
| `ray_path_diagnosis.subsample_fraction` | Fraction of events to process (0.0–1.0) |
| `ray_path_diagnosis.compression` | HDF5 compression: `gzip` / `lzf` / `none` |
| `ray_path_simplification.enabled` | Enable RDP simplification |
| `ray_path_simplification.epsilon` | RDP tolerance in km |

---

## 2. Outputs

All outputs are written to paths specified in the configuration. See [`docs/file_formats.md`](file_formats.md) for complete column/key definitions and units.

| File | Format | Stage | Description |
|---|---|---|---|
| `velocity_model_final.csv` | CSV | Velocity | Updated Vp/Vs on reference grid |
| `velocity_model_final.h5` | HDF5 | Velocity | Same data with compression |
| `synthetic_traveltimes.csv` | CSV | Travel time | Observed + synthetic TT and ground-truth metadata (residuals are computed, not stored) |
| `source_relocation.csv` | CSV | Source | Updated source coordinates and origin times |
| `ray_paths.h5` | HDF5 | Ray path | Cached ray paths grouped by phase and shot |
| `ray_paths_simplified.h5` | HDF5 | Ray path | RDP-simplified ray paths (optional) |
| `dws.h5` | HDF5 | DWS | Volumetric DWS coverage grid `(nz, ny, nx)` |

---

## 3. Usage and Configuration

### Running the pipeline

```bash
# Use the default configuration
python -m postprocess.main

# Use a named configuration
python -m postprocess.main --config-name example
```

Hydra writes all run outputs to:

```
outputs/YYYY-MM-DD/HH-MM-SS_postprocess_<config_name>/
```

### CLI overrides

```bash
# Enable only velocity processing
python -m postprocess.main \
  velocity_processing.enabled=true \
  velocity_processing.vp_binary=/latte/out/updated_vp.bin \
  velocity_processing.vs_binary=/latte/out/updated_vs.bin

# Enable travel time processing and change phase
python -m postprocess.main \
  travel_time_processing.enabled=true \
  travel_time_processing.phase_type=p

# Enable ray path cache and DWS, limit to 50% of events
python -m postprocess.main \
  ray_path_diagnosis.enabled=true \
  ray_path_diagnosis.compute_dws=true \
  ray_path_diagnosis.subsample_fraction=0.5

# Force cache recomputation
python -m postprocess.main \
  ray_path_diagnosis.enabled=true \
  ray_path_diagnosis.force_recompute=true

# Run in non-strict mode (continue on error)
python -m postprocess.main general.strict_mode=false

# Change log level
python -m postprocess.main general.log_level=DEBUG
```

### Minimal configuration checklist

Before enabling each stage, confirm the following paths and values are set:

```yaml
general:
  strict_mode: true
  log_level: INFO

data:
  base_dir: /path/to/latte/output

# Velocity processing
velocity_processing:
  enabled: true
  vp_binary: ${data.base_dir}/model/updated_vp.bin
  vs_binary: ${data.base_dir}/model/updated_vs.bin
  reference_velocity_file: /path/to/preprocess/velocity_model_init.csv
  # The binary grid comes from the reference file's metadata; model_grid is an optional check:
  # model_grid: {nx: 60, ny: 60, nz: 50, dx: 2.0, dy: 2.0, dz: 1.0,
  #             origin_x: -60.0, origin_y: -60.0, origin_z: 0.0}

# Travel time processing
travel_time_processing:
  enabled: true
  ground_truth_csv: /path/to/preprocess/geometry_ground_truth_traveltimes.csv
  synthetic_tt_dir: ${data.base_dir}/synthetic_traveltimes
  shot_map_file: /path/to/preprocess/shot_to_source_id_map.txt

# Source processing
source_processing:
  enabled: true
  # initial_geometry_csv / filtered_shot_ids default to the travel_time_processing values
  sx_file: ${data.base_dir}/sources/sx.bin
  sy_file: ${data.base_dir}/sources/sy.bin
  sz_file: ${data.base_dir}/sources/sz.bin
  st0_file: ${data.base_dir}/sources/st0.bin
```

Refer to `configs/postprocess/example.yaml` for a complete working template.

---

## 4. Logging

The log level is controlled by `general.log_level` (default: `INFO`). Valid values are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.

| Level | What is logged |
|---|---|
| `DEBUG` | Full YAML configuration dump; per-field value ranges for gradient/search binaries; coordinate ranges; interpolation fallback events |
| `INFO` | Stage start/end; file loading summaries; binary value ranges; output row counts; RMS residuals; DWS statistics; cache hit/miss counts |
| `WARNING` | Source corrections all zero (likely incorrect file); unmatched virtual stations (exchange mode); optional binary files missing; corrupted cache entries detected |
| `ERROR` | Binary array size mismatch; missing required files; out-of-range shot IDs; cache config hash mismatch |

Key log messages to monitor:

```
# Travel time — confirm coverage
Final: M / N ground truth pairs found (P.P% coverage)
RMS residual: X.XXX s

# Source processing — verify corrections are non-trivial
Mean spatial shift: X.XXX km (std: X.XXX km)
Mean origin time shift: X.XXX s (std: X.XXX s)

# Ray path cache — confirm incremental progress
Cache valid. N shots already computed, M new shots to process.

# DWS — verify coverage
DWS non-zero cells: N / TOTAL (P.P%)
```

---

## 5. Troubleshooting

**`velocity_processing`: binary array size does not match grid**
The product `nx × ny × nz` of the grid (from the reference file's metadata, or `model_grid` if only that matches) must exactly match the number of float32 values in the binary file; otherwise an error is raised. The LATTE input and output models may have different dimensions. Cross-check the dimensions in the LATTE Fortran input file, and make sure the reference file is the fine model written by preprocess.

**`travel_time_processing`: binary array size mismatch (exchange mode)**
In exchange mode the binary array length must equal the number of shots LATTE kept. The error prints expected vs. actual sizes. If LATTE dropped shots, list them in `travel_time_processing.filtered_shot_ids` (and dropped stations in `filtered_receiver_ids`).

**`travel_time_processing`: low coverage percentage (< 80%)**
Many ground truth picks are not found in the synthetic binaries. Confirm that `shot_map_file` was generated by the same preprocess run that produced the binaries, and that `phase_type` is consistent between preprocess and postprocess.

**`source_processing`: corrections are unexpectedly zero**
If the source parameter log reports all corrections at zero, a size mismatch between the binaries and the kept shots raises an error; if the stage failed, check `filtered_shot_ids` against the shots LATTE removed.

**`ray_path_diagnosis`: cache invalidated after config change**
Any change to grid dimensions, spacing, origin, or `raypath_binary_dir` invalidates the existing cache. Either revert the config change or set `force_recompute: true` to rebuild from scratch.

**`ray_path_diagnosis`: corrupted shots detected**
Enable `fix_corrupted_shots: true` to automatically detect and reprocess incomplete cache entries. Check `general.log_level=DEBUG` to see which shot IDs are affected.

**DWS grid is uniformly zero**
Confirm that `compute_dws: true` is set and that `ray_path_output_file` points to a non-empty cache. If the cache was built with `subsample_fraction < 1.0`, the DWS will reflect only the sampled rays.

**Slow ray path computation**
Increase `performance.n_jobs` or switch `performance.backend` from `loky` to `multiprocessing`. For very large datasets, use `subsample_fraction` to compute a representative subset first.

**HDF5 cache file grows too large**
Enable `ray_path_simplification` with `epsilon: 0.05` to reduce point density. Alternatively, reduce `compression_level` for faster writes at the cost of larger files.

---

## 6. Best Practices

- **Enable stages incrementally.** Run velocity processing first to confirm the grid is correct before enabling travel time or source processing. Add DWS last, as it is the most compute-intensive stage.
- **Keep `strict_mode: true` in production.** Silent failures in binary conversions can propagate undetected errors into the analysis. Let the pipeline stop and report the cause.
- **Verify the grid matches LATTE.** The grid comes from the reference file's metadata; log warnings about metadata/column mismatches or about an inferred grid (legacy file) point to a wrong reference file. Optionally set `model_grid` from the LATTE Fortran input file as an additional check.
- **Use the incremental cache.** Do not set `force_recompute: true` routinely. The cache stores completed shots, so interrupted runs resume automatically.
- **Set a meaningful `compression` level.** `gzip` with `compression_level: 4` gives a good balance of file size and read/write speed for the ray path HDF5 cache.
- **Use `subsample_fraction` for exploratory runs.** A 10–20% sample is usually sufficient to assess DWS coverage patterns before committing to a full computation.
- **Validate residuals before analysis.** Check the weighted RMS in the travel time processing log. Values significantly higher than the preprocess acceptance range suggest a configuration mismatch.
- **Archive the `shot_to_source_id_map.txt`.** This file links shot indices to event IDs across the entire workflow. Losing it makes it impossible to trace results back to individual events.
