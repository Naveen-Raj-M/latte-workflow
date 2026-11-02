# Postprocess Pipeline

The postprocess pipeline converts the binary outputs of a LATTE run into CSV and HDF5 files, builds a ray path cache, and derives a model-coverage diagnostic (Derivative Weight Sum, DWS). All stages are independent and disabled by default; enable only the stages needed for a run.

> **Configuration reference:** See [`configs/postprocess/README.md`](../configs/postprocess/README.md) for a complete description of every configuration parameter.

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

The pipeline is driven by `postprocess/main.py` and runs the stages below in order. Every stage has an `enabled` flag. With `general.strict_mode: true` (default) the run stops at the first error; with `false` a failed stage is reported and the next one runs.

---

### 1.1 Velocity Processing (`postprocess_velocity.py`)

Reads the updated Vp and Vs binaries from LATTE and writes them as CSV and/or HDF5 on a reference grid.

**What it does:**

- Reads `vp_binary` and `vs_binary` (float32, m/s), reshapes them to `(nz, ny, nx)` in Fortran order, and converts them to km/s.
- Takes the grid of the binaries from the metadata of `reference_velocity_file` (the CSV or HDF5 model written by [preprocess](preprocess.md)). The grid is tested against the number of values in the binaries. The optional `model_grid` block is an additional check; if only `model_grid` matches the binary size it is used, with a warning, and if neither matches an error is raised.
- Runs in one of two modes:
  - **`sample_at_coarse` (default):** samples the output grid at every point of the reference file. Use this when the LATTE input and output models have different dimensions.
  - **`expand_to_fine`:** writes the full output grid.
- Optionally appends gradient and search direction fields (`grad_vp.bin`, `srch_vp.bin`, …) as extra columns.
- Optionally writes a coarse-resolution dataset or CSV for faster analysis.

**Key configuration:**

| Key | Purpose |
|---|---|
| `velocity_processing.enabled` | Enable/disable this stage |
| `velocity_processing.vp_binary`, `vs_binary` | LATTE output velocity binaries |
| `velocity_processing.reference_velocity_file` | Reference model providing the grid |
| `velocity_processing.model_grid.*` | Optional extra check of the binary grid |
| `velocity_processing.interpolation_mode` | `sample_at_coarse` or `expand_to_fine` |
| `velocity_processing.velocity_output_format` | `csv` / `hdf5` / `both` |
| `velocity_processing.process_gradient`, `process_search` | Append gradient / search direction columns |

---

### 1.2 Travel Time Processing (`postprocess_traveltime.py`)

Merges the observed travel times with the synthetic travel times predicted by LATTE.

**What it does:**

- Reads the ground truth CSV written by preprocess and the shot map `shot_to_source_id_map.txt`.
- Reads the synthetic binaries `shot_<id>_traveltime_{p,s}.bin` (float32) from `synthetic_tt_dir`. The binaries store `physical_tt + origin_time`; the configured origin time is subtracted.
- With `source_correction: true`, replaces the origin time with the corrected value from `updated_st0.bin` (indexed by shot id).
- In **source–receiver exchange mode** (`source_receiver_exchange: true`) LATTE treats each unique receiver as a virtual source. The pipeline rebuilds that layout from the geometry files exactly as LATTE does (virtual sources are the unique receivers in order of first appearance; the virtual receivers are the shots kept by LATTE, in `geometry.txt` order) and matches virtual sources to station names by coordinates.
- Writes one CSV with the observed and synthetic travel times and all metadata. Residuals (`observed_tt − synthetic_tt`) are not stored; the log reports the weighted RMS.

> **Exchange mode:** The length of each binary must equal the number of shots kept by LATTE, and the number of binaries per phase must equal the number of virtual sources. A mismatch raises an error (a warning if `exchange_geometry_file` is given). If LATTE dropped shots or receivers, list them in `filtered_shot_ids` and `filtered_receiver_ids`.

**Key configuration:**

| Key | Purpose |
|---|---|
| `travel_time_processing.enabled` | Enable/disable this stage |
| `travel_time_processing.ground_truth_csv` | Preprocess ground truth CSV |
| `travel_time_processing.synthetic_tt_dir` | Directory of synthetic travel time binaries |
| `travel_time_processing.shot_map_file` | `shot_to_source_id_map.txt` |
| `travel_time_processing.phase_type` | `both` / `p` / `s` |
| `travel_time_processing.source_receiver_exchange` | Exchange (reciprocity) mode |
| `travel_time_processing.geometry_dir` | Geometry files used to rebuild the exchange |
| `travel_time_processing.filtered_shot_ids`, `filtered_receiver_ids` | Shots / stations dropped by LATTE |
| `travel_time_processing.source_correction`, `source_correction_file` | Apply corrected origin times |

---

### 1.3 Source Parameter Processing (`postprocess_source.py`)

Reads the updated hypocenters and origin times and merges them with the travel time table.

**What it does:**

- Reads `updated_sx.bin`, `updated_sy.bin`, `updated_sz.bin`, `updated_st0.bin` (float32; coordinates in m, origin times in s), one value per shot kept by LATTE. Array index *k* belongs to the *k*-th shot of `initial_geometry_csv` (ascending shot id) that is not in `filtered_shot_ids`. `parameters_to_process` selects which of the four files are read.
- Checks that the number of kept shots equals the binary size (an error otherwise) and that the shot ids of the travel time CSV exist in `initial_geometry_csv`.
- Optionally reads gradient and search direction binaries for the four parameters.
- Adds `corrected_event_x/y/z` and `corrected_origin_time_s` to the travel time table and writes it as `source_relocation.csv`. The log reports the mean and maximum shifts in m.

**Key configuration:**

| Key | Purpose |
|---|---|
| `source_processing.enabled` | Enable/disable this stage |
| `source_processing.sx_file`, `sy_file`, `sz_file`, `st0_file` | Updated source binaries |
| `source_processing.parameters_to_process` | Subset of `sx`, `sy`, `sz`, `st0` |
| `source_processing.initial_geometry_csv` | Initial locations (default: the ground truth CSV) |
| `source_processing.filtered_shot_ids` | Shots removed by LATTE (default: the travel time setting) |
| `source_processing.process_gradient`, `process_search` | Append gradient / search direction columns |

---

### 1.4 Ray Path Processing (`compute_ray_paths.py`, `read_raypath_binary.py`, `postprocess_dws.py`, `simplify_ray_paths.py`)

Reads the ray paths written by LATTE, stores them in an incremental HDF5 cache, and optionally computes the Derivative Weight Sum.

> **LATTE setting:** Ray paths are written by LATTE when `raypath_save = true` is set in its parameter file (output directory `raypath_dir`). This is available in the [forked LATTE repository](https://github.com/Naveen-Raj-M/latte_traveltime).

**Cache (`compute_ray_paths.py`, `read_raypath_binary.py`):**

- Reads one binary per shot and phase, `shot_<id>_raypath_<phase>.bin`: an `int32` ray count, `int32[nrays]` points per ray, then `float32` x, y, z coordinates in metres.
- Matches each ray to its receiver and converts the coordinates to km.
- Writes the cache grouped by phase and shot (`/p_wave/shots/<id>/`, `/s_wave/shots/<id>/`) in batches.
- Stores an MD5 hash of the critical settings (grid, binary directory); a changed setting invalidates the cache. Shots already cached are skipped, so interrupted runs resume. `fix_corrupted_shots` re-computes damaged entries, and `force_recompute` rebuilds everything.
- `event_id_list_file` and `subsample_fraction` restrict the cache to a subset of events when `compute_dws` is false.

**DWS (`postprocess_dws.py`):**

- Accumulates `DWS = Σ(w · L²)` per grid cell, where `w` is the pick weight and `L` is the ray length inside the cell in km.
- Runs in parallel (`performance.n_jobs`, `performance.backend`).
- Writes the grid as a velocity-format HDF5 file with the columns `x`, `y`, `z`, `dws` (see [file_formats.md](file_formats.md)).

**Simplification (`simplify_ray_paths.py`):**

- Reduces the size of the cache with the Ramer–Douglas–Peucker algorithm. `epsilon` (km) sets the tolerance: 0.01 is gentle, 0.10 is aggressive.
- Runs independently of `ray_path_processing.enabled`, so an existing cache can be simplified on its own.

**Key configuration:**

| Key | Purpose |
|---|---|
| `ray_path_processing.enabled` | Enable/disable this stage |
| `ray_path_processing.raypath_binary_dir` | Directory of LATTE ray path binaries (required) |
| `ray_path_processing.reference_velocity_file` | Reference model giving the LATTE grid (default: `velocity_processing.reference_velocity_file`) |
| `ray_path_processing.grid.*` | Optional check of the reference grid; the grid itself if the reference is `null` |
| `ray_path_processing.ray_path_output_file` | HDF5 cache path |
| `ray_path_processing.dws_output_file` | DWS HDF5 path |
| `ray_path_processing.compute_dws` | Compute DWS after building the cache |
| `ray_path_processing.force_recompute`, `fix_corrupted_shots` | Cache control |
| `ray_path_processing.subsample_fraction`, `event_id_list_file` | Event subset |
| `ray_path_processing.simplification.*` | RDP simplification |

---

## 2. Outputs

All outputs are written to paths set in the configuration; the defaults below are relative to `data.analysis_output_dir`. See [`file_formats.md`](file_formats.md) for column and key definitions and units.

| File | Format | Stage | Description |
|---|---|---|---|
| `velocity_model_final.csv` | CSV | Velocity | Updated Vp/Vs (km/s) on the reference grid |
| `velocity_model_final.h5` | HDF5 | Velocity | Same data, compressed |
| `synthetic_traveltimes.csv` | CSV | Travel time | Observed and synthetic travel times with metadata |
| `source_relocation.csv` | CSV | Source | Updated source coordinates and origin times |
| `ray_paths/ray_paths.h5` | HDF5 | Ray path | Ray path cache by phase and shot |
| `ray_paths/ray_paths_simplified.h5` | HDF5 | Ray path | RDP-simplified cache (optional) |
| `dws_values.h5` | HDF5 | DWS | DWS on the model grid |

---

## 3. Usage and Configuration

### Running the pipeline

```bash
# Default configuration (configs/postprocess/default.yaml)
python -m postprocess.main

# A named configuration
python -m postprocess.main --config-name example
```

Hydra writes the resolved configuration and the log to:

```
outputs/YYYY-MM-DD/HH-MM-SS_postprocess_<config_name>/
```

### CLI overrides

```bash
# Velocity processing only
python -m postprocess.main \
  velocity_processing.enabled=true \
  velocity_processing.vp_binary=/latte/out/model/updated_vp.bin \
  velocity_processing.vs_binary=/latte/out/model/updated_vs.bin

# Travel times for P only
python -m postprocess.main \
  travel_time_processing.enabled=true \
  travel_time_processing.phase_type=p

# Source–receiver exchange
python -m postprocess.main \
  travel_time_processing.enabled=true \
  travel_time_processing.source_receiver_exchange=true \
  travel_time_processing.geometry_dir=/path/to/preprocess/geometry

# Ray path cache and DWS
python -m postprocess.main \
  ray_path_processing.enabled=true \
  ray_path_processing.compute_dws=true

# Ray path cache for half of the events (no DWS)
python -m postprocess.main \
  ray_path_processing.enabled=true \
  ray_path_processing.subsample_fraction=0.5

# Rebuild the cache from scratch
python -m postprocess.main \
  ray_path_processing.enabled=true \
  ray_path_processing.force_recompute=true

# Continue after a failed stage
python -m postprocess.main general.strict_mode=false

# Log verbosity
python -m postprocess.main general.log_level=DEBUG
```

### Minimal configuration checklist

```yaml
general:
  strict_mode: true
  log_level: INFO

data:
  base_dir: /path/to/latte/output
  analysis_output_dir: /path/to/analysis/output

velocity_processing:
  enabled: true
  reference_velocity_file: /path/to/preprocess/velocity_model_init.csv
  # vp_binary / vs_binary default to ${data.base_dir}/model/updated_{vp,vs}.bin

travel_time_processing:
  enabled: true
  ground_truth_csv: /path/to/preprocess/geometry_csv/geometry_ground_truth_traveltimes.csv
  shot_map_file: /path/to/preprocess/geometry/shot_to_source_id_map.txt
  # synthetic_tt_dir defaults to ${data.base_dir}/synthetic

source_processing:
  enabled: true
  # source binaries default to ${data.base_dir}/model/updated_{sx,sy,sz,st0}.bin
  # initial_geometry_csv and filtered_shot_ids default to the travel time values

ray_path_processing:
  enabled: true
  raypath_binary_dir: /path/to/latte/raypaths
```

`configs/postprocess/example.yaml` is a ready-to-copy template.

---

## 4. Logging

The log level is set by `general.log_level` (default `INFO`; `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).

| Level | What is logged |
|---|---|
| `DEBUG` | Resolved configuration; value ranges of gradient and search binaries; coordinate ranges |
| `INFO` | Stage start and end; file loading summaries; binary value ranges; output row counts; weighted RMS; correction statistics; cache status; DWS statistics |
| `WARNING` | Corrections that are all zero; unmatched virtual stations (exchange mode); missing optional binaries; corrupted cache entries; grid mismatches between reference file and configuration |
| `ERROR` | Binary size mismatches; missing required files; unknown shot ids; cache hash mismatch |

Messages to check after a run:

```
# Travel times: confirm that the residual level is plausible
Weighted RMS residual: X.XXXX s

# Source processing: confirm that the corrections are not trivial
3D spatial shift: mean=X.XXX m, max=X.XXX m

# Ray path cache
Cache status: VALID
Cached rays: P=N, S=M

# DWS coverage
Non-zero cells: N / TOTAL (P.PP%)
```

---

## 5. Troubleshooting

**Velocity processing: binary size does not match the grid**
The product `nx × ny × nz` of the grid must equal the number of float32 values in the binary. The grid comes from the metadata of the reference file; make sure the reference file is the model written by preprocess for this LATTE run, and compare it with the dimensions in the LATTE parameter file. `model_grid` can be set as an additional check.

**Travel time processing: binary array size mismatch (exchange mode)**
The binary length must equal the number of shots kept by LATTE. The error prints the expected and actual sizes. List the shots LATTE dropped in `filtered_shot_ids` and the dropped stations in `filtered_receiver_ids`.

**Travel time processing: low coverage**
Many ground truth picks have no synthetic value. Confirm that `shot_map_file` and `ground_truth_csv` come from the same preprocess run that produced the LATTE input, and that `phase_type` is the same in both stages.

**Source processing: size mismatch or zero corrections**
The number of kept shots (`initial_geometry_csv` minus `filtered_shot_ids`) must equal the size of every source binary. If all corrections are zero, the binaries probably come from a run without source updates.

**Ray path processing: cache invalidated after a configuration change**
A change of grid, spacing, origin or `raypath_binary_dir` invalidates the cache. Revert the change or set `force_recompute: true`.

**Ray path processing: corrupted shots**
Keep `fix_corrupted_shots: true` to re-compute incomplete entries; use `general.log_level=DEBUG` to list the shot ids.

**DWS is zero or very small**
Check that `compute_dws: true` is set and that the cache is not empty. A cache built with `subsample_fraction < 1.0` covers only the sampled events.

**Ray path computation is slow**
Increase `performance.n_jobs`, or try another `performance.backend`. For a first look, use `subsample_fraction` on a representative subset.

**The HDF5 cache is too large**
Enable `ray_path_processing.simplification` with `epsilon: 0.05`.

---

## 6. Best Practices

- **Enable stages one at a time.** Start with velocity processing to confirm the grid, then add travel times and sources. Enable DWS last; it is the most expensive stage.
- **Keep `strict_mode: true`.** A silently skipped stage can propagate wrong data into the analysis.
- **Use the reference file written by preprocess.** Its metadata defines the grid for every later stage; warnings about metadata and column mismatches point to a wrong reference file.
- **Use the incremental cache.** Do not set `force_recompute: true` routinely; interrupted runs resume automatically.
- **Use `subsample_fraction` for exploratory runs.** A 10–20% sample is usually enough to see the coverage pattern before a full computation.
- **Check the weighted RMS** in the travel time log before moving to the analysis; a value far above the pick uncertainties suggests a configuration mismatch.
- **Archive `shot_to_source_id_map.txt`.** It links shot indices to event IDs for the whole workflow.
- **Process several iterations with `shell_scripts/process_inversion.sh`.** It runs postprocess and analysis for every selected iteration; see [`shell_scripts/README.md`](../shell_scripts/README.md).
