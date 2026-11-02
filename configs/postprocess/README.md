# Post-Processing Configuration

## Overview

The post-processing pipeline converts the binary outputs of a [LATTE](https://github.com/lanl/latte_traveltime) run into CSV and HDF5 files for analysis. It reads inversion results (velocity models, synthetic travel times, source parameters, ray path binaries) and writes structured files for the [analysis pipeline](../analysis/README.md).

**Stages (each independently enabled):**
1. Velocity processing: Vp/Vs binaries to CSV/HDF5
2. Travel time processing: merge observed and synthetic travel times
3. Source processing: updated source locations and origin times
4. Ray path processing: HDF5 ray path cache and Derivative Weight Sum (DWS)
5. Ray path simplification: reduce the ray path cache with the Ramer–Douglas–Peucker algorithm

---

## Quick Start

### Minimal configuration

Copy `example.yaml` and update the file paths for your run:

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
  reference_velocity_file: "/path/to/velocity_model_init.csv"
  # model_grid is optional; see "velocity_processing.model_grid" below

travel_time_processing:
  enabled: true
  ground_truth_csv: "/path/to/geometry_ground_truth_traveltimes.csv"
  shot_map_file: "/path/to/shot_to_source_id_map.txt"

source_processing:
  enabled: true
  # initial_geometry_csv and filtered_shot_ids default to the travel_time_processing values

ray_path_processing:
  enabled: true
  raypath_binary_dir: "/path/to/raypath/binaries"
  # the grid comes from the reference velocity model; an optional grid block is only a check
```

All other parameters inherit their defaults from `default.yaml`.

### Run

```bash
python -m postprocess.main --config-name example
```

### CLI overrides

```bash
python -m postprocess.main --config-name example \
    velocity_processing.velocity_output_format=hdf5 \
    performance.n_jobs=8
```

---

## Configuration Reference

### `general`

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `strict_mode` | `true` | `true`, `false` | `true`: stop at the first error. `false`: log the error and continue with the next stage. |
| `log_level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | Logging verbosity. |

---

### `data`

Paths shared by all stages.

| Key | Description |
|-----|-------------|
| `base_dir` | Root directory of the LATTE outputs (binaries and models). |
| `analysis_output_dir` | Root directory of all post-processed results. |

---

### `performance`

Global parallel processing settings.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `parallel_processing` | `true` | `true`, `false` | Master switch. |
| `n_jobs` | `-1` | `-1`, `1`, `N` | Workers; `-1` = all cores, `1` = serial. |
| `backend` | `multiprocessing` | `loky`, `multiprocessing`, `threading` | joblib backend. |
| `batch_size` | `auto` | `auto`, integer, `null` | Batch size of the parallel dispatch. |
| `pre_dispatch` | `2*n_jobs` | — | Batches dispatched ahead to the workers. |
| `show_progress` | `true` | `true`, `false` | Progress bars. |

---

### `velocity_processing`

Converts the Vp/Vs binaries of LATTE to CSV or HDF5 on the reference grid.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `false` | `true`, `false` | Enable this stage. |
| `phase_type` | `both` | `both`, `p`, `s` | Velocity fields to process. |
| `reference_velocity_file` | — | `.csv`, `.h5`, `.hdf5` | Model written by preprocess; its metadata gives the grid. |
| `vp_binary` | `${data.base_dir}/model/updated_vp.bin` | — | LATTE Vp output (m/s). |
| `vs_binary` | `${data.base_dir}/model/updated_vs.bin` | — | LATTE Vs output (m/s). |
| `interpolation_mode` | `sample_at_coarse` | `sample_at_coarse`, `expand_to_fine` | `sample_at_coarse`: sample the output at the points of the reference file. `expand_to_fine`: write the full output grid. |
| `velocity_output_format` | `csv` | `csv`, `hdf5`, `both` | Output format. |
| `velocity_output_csv` | `${data.analysis_output_dir}/velocity_model_final.csv` | — | CSV path. |
| `velocity_output_hdf5` | `${data.analysis_output_dir}/velocity_model_final.h5` | — | HDF5 path. |

#### `velocity_processing.model_grid` *(optional)*

The grid of the LATTE binaries. The metadata of `reference_velocity_file` is the reference (if a file has no grid metadata, the grid is inferred from its `x`, `y`, `z` columns); `model_grid` is an additional check. Both are tested against the number of values in `vp_binary` / `vs_binary`: if the metadata grid matches it is used (with a warning if `model_grid` differs); if only `model_grid` matches it is used, with a warning; if neither matches, an error is raised. The LATTE input and output models may have different dimensions.

| Key | Unit | Description |
|-----|------|-------------|
| `nx`, `ny`, `nz` | — | Number of grid points along x, y, z. |
| `dx`, `dy`, `dz` | km | Grid spacing. |
| `origin_x`, `origin_y`, `origin_z` | km | Cartesian origin of the grid. |

#### `velocity_processing.hdf5_options`

Used when `velocity_output_format` is `hdf5` or `both`.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `compression` | `gzip` | `gzip`, `lzf`, `null` | HDF5 compression. |
| `compression_level` | `4` | `0`–`9` | Gzip level. |
| `use_coarse_dataset` | `false` | `true`, `false` | Read the reference model from its coarse dataset. |
| `create_coarse_dataset` | `false` | `true`, `false` | Write a coarse dataset (`velocity_coarse`) into the output file. |
| `write_coarse_csv` | `false` | `true`, `false` | Write a coarse CSV. |
| `output_coarse_csv` | `${data.analysis_output_dir}/velocity_model_final_coarse.csv` | — | Path of the coarse CSV. |
| `coarse_grid.dx/dy/dz` | `2.0` | km | Coarse spacing; integer multiples of the fine spacing. |

#### Gradient and search direction outputs

| Key | Default | Description |
|-----|---------|-------------|
| `process_gradient` | `false` | Add `grad_vp`, `grad_vs`. |
| `process_search` | `false` | Add `srch_vp`, `srch_vs`. |
| `parameters_to_process` | `["vp", "vs"]` | Parameters to include. |
| `grad_vp_binary`, `grad_vs_binary` | `${data.base_dir}/model/grad_{vp,vs}.bin` | Gradient binaries. |
| `srch_vp_binary`, `srch_vs_binary` | `${data.base_dir}/model/srch_{vp,vs}.bin` | Search direction binaries. |

---

### `travel_time_processing`

Merges the observed travel times with the synthetic travel times of LATTE.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `false` | `true`, `false` | Enable this stage. |
| `phase_type` | `both` | `both`, `p`, `s` | Phases to process. |
| `ground_truth_csv` | — | — | Preprocess `geometry_ground_truth_traveltimes.csv`. |
| `synthetic_tt_dir` | `${data.base_dir}/synthetic` | — | Directory of `shot_<id>_traveltime_{p,s}.bin`. |
| `shot_map_file` | — | — | `shot_to_source_id_map.txt`. |
| `synthetic_tt_output_csv` | `${data.analysis_output_dir}/synthetic_traveltimes.csv` | — | Output CSV. |

#### Source–receiver exchange (reciprocity)

| Key | Default | Description |
|-----|---------|-------------|
| `source_receiver_exchange` | `false` | LATTE uses the receivers as virtual sources. |
| `geometry_dir` | `null` | Directory with `geometry.txt` and `shot_<n>_geometry.txt` (recommended). The exchange is rebuilt from it as LATTE does: virtual sources are the unique receivers in order of first appearance, virtual receivers are the kept shots in `geometry.txt` order. |
| `geometry_file` | `geometry.txt` | Name of the geometry index in `geometry_dir`. |
| `exchange_geometry_file` | `null` | Optional LATTE `exchange_geometry.txt`, used as the virtual-source table. With `geometry_dir` also set, mismatches only warn; without it, any mismatch between geometry, ground truth and binaries raises an error. |
| `filtered_shot_ids` | `[]` | Shot ids LATTE dropped (`sid_select`, `sid_exclude`, no valid sources or receivers). |
| `filtered_receiver_ids` | `[]` | Station names dropped from every shot before the virtual sources are numbered. |
| `coordinate_tolerance` | `1.0` | m; tolerance for matching geometry receivers to ground truth stations. |

#### Source correction

| Key | Default | Description |
|-----|---------|-------------|
| `source_correction` | `false` | Use the corrected origin times of the LATTE source update. |
| `source_correction_file` | `${data.base_dir}/model/updated_st0.bin` | Updated origin time binary, indexed by shot id. |

---

### `source_processing`

Reads the updated source locations and origin times and writes them, merged with the travel time table, to CSV.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Enable this stage. |
| `parameters_to_process` | `["sx", "sy", "sz", "st0"]` | Parameters that are read and validated. |
| `sx_file`, `sy_file`, `sz_file` | `${data.base_dir}/model/updated_{sx,sy,sz}.bin` | Updated source coordinates (m). |
| `st0_file` | `${data.base_dir}/model/updated_st0.bin` | Updated origin times (s). |
| `initial_geometry_csv` | `${travel_time_processing.ground_truth_csv}` | Initial source locations; needs `shot_id`, `event_x/y/z`, `origin_time_s`. Its shots minus `filtered_shot_ids` are the shots in the binaries; that count must equal the number of float32 values in every source binary, otherwise an error is raised. Binary index *k* is the *k*-th kept shot in ascending shot id. |
| `filtered_shot_ids` | `${travel_time_processing.filtered_shot_ids}` | Shots LATTE removed. They have no binary entry; their corrected columns stay empty. |
| `output_csv` | `${data.analysis_output_dir}/source_relocation.csv` | Output CSV. |

#### Gradient and search direction outputs

| Key | Default | Description |
|-----|---------|-------------|
| `process_gradient` | `false` | Add the source gradient columns. |
| `process_search` | `false` | Add the source search direction columns. |
| `grad_sx_file` … `grad_st0_file` | `${data.base_dir}/model/grad_*.bin` | Gradient binaries. |
| `srch_sx_file` … `srch_st0_file` | `${data.base_dir}/model/srch_*.bin` | Search direction binaries. |

---

### `ray_path_processing`

Reads the ray path binaries written by LATTE (`raypath_save = true`), builds the HDF5 cache and optionally computes the Derivative Weight Sum (DWS), a measure of the weighted ray path length in each grid cell.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `false` | `true`, `false` | Enable this stage. |
| `raypath_binary_dir` | `null` | — | Directory with the `shot_*_raypath_*.bin` files. Required. |
| `reference_velocity_file` | `${velocity_processing.reference_velocity_file}` | — | Model whose metadata gives the LATTE grid. `null`: use `grid` (with a warning). |
| `ground_truth_csv` | `${travel_time_processing.synthetic_tt_output_csv}` | — | CSV providing the pick weights. |
| `shot_map_file` | `${travel_time_processing.shot_map_file}` | — | Shot-to-event map. |
| `ray_path_output_file` | `${data.analysis_output_dir}/ray_paths/ray_paths.h5` | — | HDF5 cache. |
| `dws_output_file` | `${data.analysis_output_dir}/dws_values.h5` | — | DWS file. |
| `phase_type` | `both` | `both`, `p`, `s` | Phases to process. |
| `compute_dws` | `false` | `true`, `false` | Compute DWS after building the cache. |
| `force_recompute` | `false` | `true`, `false` | Recompute all shots even if cached. |
| `fix_corrupted_shots` | `true` | `true`, `false` | Detect and recompute damaged cache entries only. |
| `compression` | `gzip` | `gzip`, `lzf`, `null` | HDF5 compression of the cache. |
| `compression_level` | `4` | `0`–`9` | Gzip level. |

#### `ray_path_processing.grid` *(optional)*

The Cartesian grid used to bin the rays and compute DWS. It is read from the metadata of `reference_velocity_file`; `grid` is an additional check, and a difference logs a warning (the reference file is used). If `reference_velocity_file` is `null`, `grid` is used and must be complete.

| Key | Unit | Description |
|-----|------|-------------|
| `nx`, `ny`, `nz` | — | Number of grid points. |
| `dx`, `dy`, `dz` | km | Grid spacing. |
| `origin_x`, `origin_y`, `origin_z` | km | Cartesian origin. |

#### Event filtering *(cache only; applies when `compute_dws: false`)*

| Key | Default | Description |
|-----|---------|-------------|
| `event_id_list_file` | `null` | Text file with one event id per line. `null` = all events. |
| `subsample_fraction` | `1.0` | Fraction of events when `event_id_list_file` is `null`. |

#### `ray_path_processing.dws`

| Key | Default | Description |
|-----|---------|-------------|
| `weight_column` | `weight` | Column of the ground truth CSV used as DWS weight; 1.0 if missing. |

#### `ray_path_processing.simplification`

Reduces the size of the cache with the Ramer–Douglas–Peucker algorithm. It runs independently of `ray_path_processing.enabled`, so an existing cache can be simplified on its own.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `false` | `true`, `false` | Enable simplification. |
| `input_cache_path` | `${ray_path_processing.ray_path_output_file}` | — | Cache to simplify. |
| `output_cache_path` | `${data.analysis_output_dir}/ray_paths/ray_paths_simplified.h5` | — | Simplified cache. |
| `epsilon` | `0.05` | km | Tolerance: `0.01` gentle, `0.05` moderate, `0.10` aggressive. |
| `phase_type` | `both` | `both`, `p`, `s` | Phases to simplify. |

---

## Output Structure

```
analysis_output_dir/
├── velocity_model_final.csv           # Vp/Vs model (CSV)
├── velocity_model_final.h5            # Vp/Vs model (HDF5)
├── velocity_model_final_coarse.csv    # optional coarse CSV
├── synthetic_traveltimes.csv          # observed + synthetic travel times
├── source_relocation.csv              # updated source locations and origin times
├── dws_values.h5                      # Derivative Weight Sum grid
└── ray_paths/
    ├── ray_paths.h5                   # ray path cache
    └── ray_paths_simplified.h5        # simplified cache (optional)
```

Hydra also writes the resolved configuration and the log to:
```
outputs/<YYYY-MM-DD>/<HH-MM-SS>_postprocess_<config_name>/
```

---

## Important Notes

- **All stages are disabled by default.** Set `enabled: true` for each stage you want to run.
- **The grid comes from the metadata of the reference file.** Use the model written by preprocess; `model_grid` is an additional check and is used only if the metadata grid does not match the binaries.
- **Source binaries must hold one value per kept shot.** Kept shots = shots of `initial_geometry_csv` minus `filtered_shot_ids`; a different count raises an error. List the shots LATTE removed in `filtered_shot_ids`.
- **`strict_mode: true` is recommended.** In lenient mode a failed stage is skipped and later stages that depend on it may fail or give incomplete results.
- **`ray_path_processing.raypath_binary_dir` must be set** when the stage is enabled.
- **Simplification needs an existing cache.** It reads the HDF5 cache written by `ray_path_processing`.
- **`source_receiver_exchange` needs `geometry_dir` or `exchange_geometry_file`.** If the shot or virtual-source counts of geometry, ground truth and binaries disagree (for example because LATTE filtered shots), the stage raises an error unless `exchange_geometry_file` is given; list dropped shots and stations in `filtered_shot_ids` / `filtered_receiver_ids`.
- **`event_id_list_file` and `subsample_fraction` apply only when `compute_dws: false`.** With DWS enabled all events are processed, to give a complete sensitivity map.
- **Units:** source binaries and the coordinates in the travel time tables are in metres; the velocity files are in km and km/s; the ray path cache and the DWS grid are in km.
