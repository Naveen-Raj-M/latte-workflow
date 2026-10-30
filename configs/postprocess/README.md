# Post-Processing Configuration

## Overview

The post-processing pipeline converts [LATTE](https://github.com/lanl/latte_traveltime) tomography binary outputs into CSV and HDF5 formats for analysis and visualization. It reads inversion results (velocity models, synthetic travel times, source parameters, and ray path binaries) and produces structured output files for downstream interpretation.

**Pipeline stages (each independently enabled):**
1. Velocity model processing — converts Vp/Vs binary outputs to CSV/HDF5
2. Travel time processing — merges ground truth and synthetic travel times
3. Source parameter processing — extracts updated source locations and origin times
4. Ray path diagnosis — builds HDF5 ray path cache and computes DWS sensitivity
5. Ray path simplification — reduces ray path memory footprint via RDP algorithm

---

## Quick Start

### Minimal configuration

Copy `example.yaml` and update the file paths for your dataset:

```yaml
# configs/postprocess/example.yaml
defaults:
  - default
  - _self_

data:
  base_dir: "/path/to/latte/output"
  analysis_output_dir: "/path/to/analysis/output"

velocity_processing:
  enabled: true
  reference_velocity_file: "/path/to/velocity_cartesian_final.csv"
  # fine_grid is optional; see "velocity_processing.fine_grid" below

travel_time_processing:
  enabled: true
  ground_truth_csv: "/path/to/geometry_ground_truth_traveltimes.csv"
  shot_map_file: "/path/to/shot_to_source_id_map.txt"

source_processing:
  enabled: true
  n_sources: 4495

ray_path_diagnosis:
  enabled: true
  raypath_binary_dir: "/path/to/raypath/binaries"
  grid:
    nx: 401
    ny: 361
    nz: 43
    dx: 0.5
    dy: 0.5
    dz: 0.5
```

All other parameters inherit their defaults from `default.yaml`.

### Run

```bash
python3 -m postprocess.main --config-name example
```

### CLI overrides

Any key can be overridden at the command line using Hydra dot-notation without modifying the YAML file:

```bash
python3 -m postprocess.main --config-name example \
    velocity_processing.enabled=true \
    velocity_processing.velocity_output_format=hdf5 \
    performance.n_jobs=8
```

---

## Configuration Reference

### `general`

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `strict_mode` | `true` | `true`, `false` | If `true`, the pipeline stops on the first error. If `false`, it logs the error and continues to the next stage. |
| `log_level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | Controls logging verbosity. |

---

### `data`

Top-level path configuration shared across all processing stages.

| Key | Description |
|-----|-------------|
| `base_dir` | Root directory of LATTE inversion outputs (binary files, model files). |
| `analysis_output_dir` | Root directory where all post-processed results are written. |

---

### `performance`

Global parallel processing settings applied across all enabled modules.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `parallel_processing` | `true` | `true`, `false` | Master switch for parallel execution across modules. |
| `n_jobs` | `-1` | `-1`, `1`, `N` | Number of parallel workers. `-1` = all CPU cores, `1` = serial, `N` = N cores. |
| `backend` | `multiprocessing` | `loky`, `multiprocessing`, `threading` | joblib parallel backend. |
| `batch_size` | `auto` | `auto`, integer, `null` | Batch size for parallel dispatch. `auto` = joblib default. |
| `pre_dispatch` | `2*n_jobs` | — | Number of batches pre-dispatched to workers. |
| `show_progress` | `true` | `true`, `false` | Show tqdm progress bars during processing. |

---

### `velocity_processing`

Converts Vp/Vs binary output files from LATTE to CSV or HDF5 format by sampling the fine-grid binary onto the reference coordinate grid.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `false` | `true`, `false` | Enable this stage. |
| `phase_type` | `both` | `both`, `p`, `s` | Which velocity phases to process. |
| `reference_velocity_file` | — | `.csv`, `.h5`, `.hdf5` | Reference velocity file providing the coordinate grid. Format auto-detected from extension. |
| `vp_binary` | `${data.base_dir}/model/updated_vp.bin` | — | Path to the Vp binary output from LATTE. |
| `vs_binary` | `${data.base_dir}/model/updated_vs.bin` | — | Path to the Vs binary output from LATTE. |
| `interpolation_mode` | `sample_at_coarse` | `sample_at_coarse`, `expand_to_fine` | `sample_at_coarse`: sample fine-grid values at coarse reference points. `expand_to_fine`: expand to the full fine grid. |
| `velocity_output_format` | `csv` | `csv`, `hdf5`, `both` | Output format for the velocity model. |
| `velocity_output_csv` | derived | — | Output path for the CSV velocity file. |
| `velocity_output_hdf5` | derived | — | Output path for the HDF5 velocity file. |

#### `velocity_processing.fine_grid` *(optional)*

The 3D Cartesian grid of the LATTE binary output is read from the metadata of `reference_velocity_file` (written by preprocess) and checked against that file's `x`, `y`, `z` columns. Define `fine_grid` only for legacy reference files that have no grid metadata and whose columns cannot give the grid (for example a legacy coarse CSV). It is then used as a last resort, after checking it against the size of the binary files.

| Key | Unit | Description |
|-----|------|-------------|
| `nx`, `ny`, `nz` | — | Number of grid nodes along x, y, z axes. |
| `dx`, `dy`, `dz` | km | Grid spacing along each axis. |
| `origin_x`, `origin_y`, `origin_z` | km | Cartesian origin of the grid. |

#### `velocity_processing.hdf5_options`

Used when `velocity_output_format` is `hdf5` or `both`.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `compression` | `gzip` | `gzip`, `lzf`, `null` | HDF5 dataset compression algorithm. |
| `compression_level` | `4` | `0`–`9` | Gzip compression level. `4` is a balanced default. |
| `use_coarse_dataset` | `false` | `true`, `false` | Read from the coarse dataset in the HDF5 file for faster processing. |
| `create_coarse_dataset` | `false` | `true`, `false` | Write a coarse-grid dataset inside the HDF5 output file. |
| `write_coarse_csv` | `false` | `true`, `false` | Write a separate coarse-resolution CSV file. |
| `output_coarse_csv` | derived | — | Path for the coarse CSV output when `write_coarse_csv: true`. |
| `coarse_grid.dx/dy/dz` | `2.0` | km | Coarse grid spacing. Must be integer multiples of the fine grid spacing. |

#### Gradient and search direction outputs

| Key | Default | Description |
|-----|---------|-------------|
| `process_gradient` | `false` | Load and output gradient fields (`grad_vp`, `grad_vs`). |
| `process_search` | `false` | Load and output search direction fields (`srch_vp`, `srch_vs`). |
| `parameters_to_process` | `["vp", "vs"]` | Which parameters to include. |
| `grad_vp_binary` / `grad_vs_binary` | derived | Paths to gradient binary files. |
| `srch_vp_binary` / `srch_vs_binary` | derived | Paths to search direction binary files. |

---

### `travel_time_processing`

Merges observed ground truth travel times with synthetic predictions from LATTE binary outputs, producing a unified comparison CSV.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `false` | `true`, `false` | Enable this stage. |
| `phase_type` | `both` | `both`, `p`, `s` | Which phases to process. |
| `ground_truth_csv` | — | — | Path to the observed travel time CSV. Must contain `shot_id`, `receiver_id`, `phase_type`, `travel_time`, and `weight` columns. |
| `synthetic_tt_dir` | derived | — | Directory containing LATTE-generated per-shot travel time binaries. |
| `shot_map_file` | — | — | Text file mapping shot IDs to original source event IDs. |
| `synthetic_tt_output_csv` | derived | — | Output path for the merged travel time CSV. |

#### Source–receiver exchange (reciprocity-based inversion)

| Key | Default | Description |
|-----|---------|-------------|
| `source_receiver_exchange` | `false` | Enable reciprocity-based inversion mode where source and receiver roles are swapped. |
| `exchange_mapping_file` | derived | Maps original shot IDs to virtual shot IDs used in exchange. |
| `exchange_geometry_file` | derived | Geometry file for the exchange configuration. |
| `virtual_shot_mapping_file` | derived | Maps virtual shots back to original event IDs. |

#### Source correction

| Key | Default | Description |
|-----|---------|-------------|
| `source_correction` | `false` | Apply origin time correction from LATTE source binary. |
| `source_correction_file` | derived | Path to the updated source time binary (`updated_st0.bin`). |

---

### `source_processing`

Reads updated source location and origin time binaries from the LATTE inversion and writes them to CSV, optionally merged with travel time results.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Enable this stage. |
| `sx_file` / `sy_file` / `sz_file` | derived | Paths to updated source x/y/z location binaries. |
| `st0_file` | derived | Path to the updated source origin time binary. |
| `n_sources` | `4495` | Number of sources. Must match the binary file size. |
| `output_csv` | derived | Output path for the source parameter CSV. |

#### Gradient and search direction outputs

| Key | Default | Description |
|-----|---------|-------------|
| `process_gradient` | `false` | Load and output gradient fields for source parameters. |
| `process_search` | `false` | Load and output search direction fields for source parameters. |
| `parameters_to_process` | `["sx","sy","sz","st0"]` | Which source parameters to include. |
| `grad_sx/sy/sz/st0_file` | derived | Paths to gradient binary files for each source parameter. |
| `srch_sx/sy/sz/st0_file` | derived | Paths to search direction binary files for each source parameter. |

---

### `ray_path_diagnosis`

Reads precomputed ray path binaries from the LATTE forward solver, builds an HDF5 cache, and optionally computes the Derivative Weight Sum (DWS) — a sensitivity metric indicating the weighted ray path length through each grid cell.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `false` | `true`, `false` | Enable this stage. |
| `raypath_binary_dir` | `null` | — | Directory containing `shot_*_raypath_*.bin` files from LATTE. |
| `ground_truth_csv` | derived | — | CSV providing observation weights for DWS computation. |
| `shot_map_file` | derived | — | Shot-to-event mapping file. |
| `ray_path_output_file` | derived | — | Output path for the HDF5 ray path cache. |
| `dws_output_file` | derived | — | Output path for the DWS HDF5 file. |
| `phase_type` | `both` | `both`, `p`, `s` | Phases to process. |
| `compute_dws` | `true` | `true`, `false` | Compute and write the Derivative Weight Sum grid. |
| `force_recompute` | `false` | `true`, `false` | Recompute all shots even if the cache already exists. |
| `fix_corrupted_shots` | `true` | `true`, `false` | Detect and recompute only corrupted cache entries. |
| `compression` | `gzip` | `gzip`, `lzf`, `null` | HDF5 compression for the ray path cache. |
| `compression_level` | `4` | `0`–`9` | Gzip compression level. |

#### `ray_path_diagnosis.grid`

Cartesian grid definition used to bin ray paths and compute DWS. Must match the inversion grid.

| Key | Unit | Description |
|-----|------|-------------|
| `nx`, `ny`, `nz` | — | Number of grid nodes along x, y, z axes. |
| `dx`, `dy`, `dz` | km | Grid spacing along each axis. |
| `origin_x`, `origin_y`, `origin_z` | km | Cartesian origin of the grid. |

#### Event filtering (cache-only optimization; applies only when `compute_dws: false`)

| Key | Default | Description |
|-----|---------|-------------|
| `event_id_list_file` | `null` | Path to a text file with one event ID per line. `null` = use all events. |
| `subsample_fraction` | `1.0` | Fraction of events to sample when `event_id_list_file` is `null`. `1.0` = all events. |

#### `ray_path_diagnosis.dws`

| Key | Default | Description |
|-----|---------|-------------|
| `weight_column` | `weight` | Column name in the ground truth CSV to use as DWS weights. Falls back to `1.0` if not found. |

---

### `ray_path_simplification`

Reduces the memory footprint of the ray path HDF5 cache using the Ramer–Douglas–Peucker (RDP) algorithm, removing collinear points within a tolerance threshold. Useful for large datasets prior to visualization or further analysis.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `false` | `true`, `false` | Enable this stage. |
| `input_cache_path` | derived | — | Path to the input ray path HDF5 cache (output of `ray_path_diagnosis`). |
| `output_cache_path` | derived | — | Path for the simplified HDF5 output. |
| `epsilon` | `0.05` | km | RDP simplification tolerance. `0.01` = gentle, `0.05` = moderate, `0.10` = aggressive. |
| `phase_type` | `both` | `both`, `p`, `s` | Phases to simplify. |

---

## Output Structure

```
analysis_output_dir/
├── velocity_model_final.csv       # Vp/Vs model (CSV)
├── velocity_model_final.h5        # Vp/Vs model (HDF5)
├── velocity_model_final_coarse.csv  # (optional) coarse-sampled CSV
├── synthetic_traveltimes.csv          # Ground truth + synthetic TT comparison
├── source_relocation.csv              # Updated source locations and origin times
├── dws_values.h5                             # Derivative Weight Sum grid
└── ray_paths/
    ├── ray_paths.h5                          # Full ray path HDF5 cache
    └── ray_paths_simplified.h5              # RDP-simplified ray path cache
```

Hydra also writes a timestamped run log to:
```
outputs/<YYYY-MM-DD>/<HH-MM-SS>_postprocess_<config_name>/
```

---

## Important Notes

- **All processing stages are disabled by default.** Explicitly set `enabled: true` for each stage you want to run.
- **The grid comes from the reference file's metadata.** Use a reference file written by preprocess (or by a previous postprocess run); `fine_grid` is only a fallback for legacy files and must then match the LATTE inversion grid exactly.
- **`n_sources` must match the size of the source binary files.** A mismatch will raise a validation error at startup.
- **`coarse_grid` spacing must be integer multiples of the fine grid spacing.** Non-integer multiples will produce incorrect coarse grids.
- **`strict_mode: true` is recommended for production runs.** In lenient mode (`false`), a failed stage is skipped silently and downstream stages that depend on its output may also fail or produce incomplete results.
- **`ray_path_diagnosis` requires `raypath_binary_dir` to be set.** It defaults to `null` and will raise an error if left unset when the stage is enabled.
- **`ray_path_simplification` should run after `ray_path_diagnosis`.** It reads the HDF5 cache produced by that stage; if the cache does not exist the stage will fail.
- **`source_receiver_exchange` requires all three exchange mapping files** (`exchange_mapping_file`, `exchange_geometry_file`, `virtual_shot_mapping_file`) to be valid paths. Missing any one will raise an error.
- **`event_id_list_file` and `subsample_fraction` filtering applies only when `compute_dws: false`.** When DWS is enabled, all events are processed to ensure a complete sensitivity map.
