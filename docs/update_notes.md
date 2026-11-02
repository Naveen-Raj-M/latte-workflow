# Update Notes

This file records the changes between releases and the fixes made after a release. Add a new entry at the top for every fix (see [Recording a fix](#recording-a-fix)).

---

## Table of Contents

1. [v4.0.0](#1-v400)
   - [Summary](#11-summary)
   - [New features](#12-new-features)
   - [Changed behaviour](#13-changed-behaviour)
   - [Removed](#14-removed)
   - [Migrating from v3.0.0](#15-migrating-from-v300)
   - [Dependencies and environments](#16-dependencies-and-environments)
   - [Testing status](#17-testing-status)
   - [Known limitations](#18-known-limitations)
2. [Recording a fix](#recording-a-fix)
3. [Fixes after v4.0.0](#fixes-after-v400)

---

## 1. v4.0.0

### 1.1 Summary

v4.0.0 is a refactoring and extension release that prepares the toolkit for publication. Almost the whole workflow was restructured, and the core functionality was re-tested with the new structure (see [§1.7](#17-testing-status)).

| Area | v3.0.0 | v4.0.0 |
|---|---|---|
| Catalog input | Event data read per run from QuakeML or HDF5 | QuakeML parsed once and shared; per-step filters; raw catalog plots; QuakeML → HDF5 conversion; secondary catalogs |
| Grid definition | Grid keys repeated in the configuration of every module | Defined once in preprocess, stored as file metadata, read by every later stage |
| Table layout | Several per-pick and velocity layouts, names and units | One layout per table, with units written into the file header |
| Source–receiver exchange | Needed LATTE exchange mapping files | Rebuilt from the geometry files; handles shots and receivers dropped by LATTE |
| Ray paths | Cache and DWS keyed to a configured grid | Grid from the reference model, stored with the cache; DWS stored in the velocity format |
| Resolution tests | Checkerboard test | Checkerboard test (more patterns and plots) and a new spike test |
| Cluster scripts | — | Slurm and shell scripts for forward runs, inversions, resolution tests and batch processing |
| Plotting | Per-module plot options | Unified `plots` section for the preprocess plots |
| Code structure | Partly object oriented | Function based |
| Tests | Unit tests in `tests/` | Removed; the workflow is validated by end-to-end runs |
| Documentation | Three pipeline documents | Rewritten, plus update notes and a resolution test document |

---

### 1.2 New features

**Preprocess**
- One-time QuakeML parsing (`quakeml_loader.py`) shared by raw data visualization, HDF5 conversion and the pipeline; per-step filters (`catalog_filters.py`), including `require_both_p_and_s`, `azimuthal_gap_max`, `min_picks_per_event`, `max_picks_per_event`.
- Raw data visualization of the QuakeML catalog (`raw_data_visualization`): maps, temporal, magnitude, distance, picks-per-event and azimuthal-gap plots.
- QuakeML → HDF5 conversion (`hdf5_conversion`) with single-pick-per-phase selection and column selection.
- Secondary catalogs with station merging rules (`data.secondary_catalogs`).
- Separate P and S pick weights (`separate_weight_p_s`, `default_weight_p`, `default_weight_s`), and 5-column geometry files.
- `preprocessing.run_pipeline` to run only the raw-data steps.
- Unified `plots` section shared by `raw_data_visualization` and `final_data_visualization`.

**Postprocess**
- Source–receiver exchange rebuilt from the geometry files (`geometry_dir`), with optional cross-check against `exchange_geometry.txt`; `filtered_shot_ids` and `filtered_receiver_ids` for shots and stations dropped by LATTE.
- Source processing validates the binaries against the shots kept by LATTE (`initial_geometry_csv`, `filtered_shot_ids`).

**Analysis**
- Tradeoff analysis: parameter sweeps (`x_axis_mode: parameter`), `data_variance` Y axis, optional weighted RMS, log-residuals, `model_variance` and `rms_variation` aggregations with zero-change exclusion, `write_report`.
- Ray density analysis along oblique vertical sections, with a plan-view map of the section traces.
- Source analysis for a subset of the source parameters (`parameters_to_process`); comparison mode uses the corrected locations.
- Iteration 0 is included in tradeoff reports and statistics.

**Resolution tests**
- Spike test (`spike_test`): 3D, single-layer 2D and arbitrarily oriented Gaussian anomalies with independent P and S amplitude and polarity.
- Checkerboard test: Vp/Vs-ratio plots and colour thresholds with optional saturation marking (`mark_threshold_black`).
- Both tests handle source–receiver exchange and shots dropped by LATTE.

**Cluster scripts** (`shell_scripts/`): Slurm jobs for forward runs, inversions, checkerboard and spike tests, and `process_inversion.sh` for batch postprocessing and analysis of iterations. See [`shell_scripts/README.md`](../shell_scripts/README.md).

**Environment files:** `requirements.txt`, `environment.yml`, `environment-tested.yml`.

---

### 1.3 Changed behaviour

| Topic | Change |
|---|---|
| Grid | The grid (`nx`, `dx`, `origin_x`, …) is stored in the metadata of every velocity-format file and read from there. `model_grid` and `ray_path_processing.grid` are optional checks (or fallbacks), and `ray_density_analysis.grid` fills only keys that the ray path cache lacks. A disagreement logs a warning and the metadata value is used. |
| Velocity files | One column layout (`lat, lon, depth, x, y, z, vp, vs`, optional gradient/search columns) and one metadata schema for all files, including DWS (`dws` instead of `vp`, `vs`) and resolution-test models. |
| Travel time tables | Ground truth, synthetic and source-relocation tables share one layout. Cartesian coordinates are in metres, units are recorded in the header. Residuals and distances are no longer stored; they are computed where needed. |
| Observed travel times | Noisy travel time tables store the noisy value in `observed_tt`. |
| Source binaries | The number of values must match the number of shots kept by LATTE; a mismatch is an error. Previously the number of sources was configured. |
| DWS | Written as a velocity-format HDF5 file (`dws_values.h5`, columns `x, y, z, dws`). |
| Weights | A fixed P and S weight can be combined with azimuthal weighting. |
| Ray paths | The grid of the cache comes from the reference velocity model and is stored with the cache; the ray density analysis reads it from there. |
| Analysis output | Plot, report and statistics files follow one layout per module; the generic `plots_subdir`, `reports_subdir` and `stats_subdir` settings were removed. |

---

### 1.4 Removed

- The dataset-specific preprocessing module and its configuration. Use the general preprocess pipeline with the catalog formats it supports.
- Ray density plots in preprocess (the module lives in analysis only).
- Configuration keys that had no effect, and precision options (binaries are always float32): `traveltime_noise.output_precision`, `precision` of the gradient reader, `travel_time_analysis.enable_corrected_format`, `preprocess.use_parallel`, `postprocess.n_sources`, `postprocess.exchange_mapping_file`, `postprocess.virtual_shot_mapping_file`, `analysis.quality_control`, `analysis.zero_initial_origin_time`, `analysis.origin_time_metadata_dir`.
- Unit tests (`tests/`).
- `seaborn` import in `utils/plot_utils.py` (it was unused).

---

### 1.5 Migrating from v3.0.0

Files and configurations of v3.0.0 can still be read in most cases: the readers look for the current name first and then for the old names, and log a **warning** whenever an old name or layout is used. New files are always written with the current names. Rename old files when convenient so that the warnings disappear.

**File names**

| Table | Current name | Old names that are still found |
|---|---|---|
| Ground truth travel times | `geometry_ground_truth_traveltimes.csv` | `ground_truth_traveltimes.csv`, `ground_truth_travel_times_corrected.csv`, `ground_truth_travel_times.csv` |
| Synthetic travel times | `synthetic_traveltimes.csv` | `synthetic_travel_times_latte_output.csv`, `travel_time_residuals*.csv` |
| Source relocation | `source_relocation.csv` | `source_parameters_latte_output.csv`, `source_corrections*.csv`, and the synthetic table that held the source columns |
| Initial velocity model | `velocity_model_init.{csv,h5}` (also `_coarse`, `_exact`) | `velocity_model_cartesian`, `velocity_initial_cartesian`, `velocity_cartesian_initial`, `velocity_cartesian` |
| Final velocity model | `velocity_model_final.{csv,h5}` (also `_coarse`) | `velocity_cartesian_latte_output`, `velocity_cartesian_updated`, `velocity_model_inverted` |

**Columns**

| Table | Change |
|---|---|
| Velocity files | `X_grid_km`, `Y_grid_km`, `Z_grid_km` (and duplicate lower-case `x`, `y`, `z`) are renamed to `x`, `y`, `z`. The grid index columns `i`, `j`, `k`, `i_raw`, `j_raw`, `k_raw` are dropped. A file without grid metadata is read with the grid inferred from its columns. DWS values stored in `vp` are read as `dws`. Nested metadata names (`grid_dimensions.nx`, `grid_spacing.dx`, `grid_origin.x`) are read as the flat keys. |
| Travel time tables | `event_x_transformed` … `corrected_event_z_transformed` lose the `_transformed` suffix. The columns `weighted_rms`, `residual`, `residual_s` and `distance_m` are ignored. Files without the header are read with the same units, so nothing is converted. |

**Configuration keys**

| v3.0.0 | v4.0.0 |
|---|---|
| `visualization` (preprocess) | `final_data_visualization`, with styling in `plots` |
| `ray_path_diagnosis`, `ray_path_simplification` (postprocess) | `ray_path_processing`, `ray_path_processing.simplification` |
| `fine_grid` (postprocess velocity) | optional `model_grid` |
| `removed_shots` (resolution tests) | `filtered_shot_ids` |
| `ray_density_analysis.grid.x_min`, `y_min`, `z_min` | `origin_x`, `origin_y`, `origin_z` |
| Exchange mapping files | `geometry_dir` (and optionally `exchange_geometry_file`) |

**Behaviour to check after migrating**

- Source binaries from a LATTE run in which shots were dropped need `filtered_shot_ids`.
- Ray density analysis takes the grid from the metadata of the ray path cache. If it warns that a cache lacks grid keys, the configured `ray_density_analysis.grid` is used; rebuilding the cache (`force_recompute: true`) stores the grid in it.

---

### 1.6 Dependencies and environments

- Supported Python: 3.10 – 3.13.
- Direct dependencies: `numpy`, `pandas` (2.x), `scipy`, `matplotlib`, `h5py`, `tables` (PyTables, needed by `pandas.read_hdf` / `to_hdf`), `hydra-core` (brings `omegaconf` and `PyYAML`), `obspy`, `joblib`, `tqdm`, `numba`.
- Removed: `seaborn` (unused), and the optional `pyproj` and `imageio` that earlier documentation listed but the code never imported.
- `pandas` is limited to `<3`: the workflow was developed and tested with pandas 2.x.
- Environment files: `requirements.txt` (pip), `environment.yml` (conda, version ranges), `environment-tested.yml` (conda, exact versions of the last test).

---

### 1.7 Testing status

The functionality was tested at the development stage. Almost the whole workflow was then refactored for publication, and the core functionality was tested again with the refactored code. **Some parts of the workflow have not been re-tested since the refactoring and may contain bugs.** If you find one, report it as described in [Recording a fix](#recording-a-fix).

For this release the environments in [§1.6](#16-dependencies-and-environments) were built from scratch (conda-forge with Python 3.13.18, and a pip virtual environment; the modules were also imported with Python 3.10). A synthetic HDF5 catalog was run through preprocess (HDF5 input, coordinate transform, geometry and travel times, velocity model as CSV and HDF5, plots), postprocess (velocity, travel time and source processing) and analysis (travel time and velocity modules). Not re-run for this release: QuakeML parsing, raw data plots, HDF5 conversion, the remaining analysis modules, ray paths and DWS, exchange mode, the resolution tests and the cluster scripts.

---

### 1.8 Known limitations

- **Single precision only.** All binaries exchanged with LATTE are float32; the code reads and writes float32 and has no precision options.
- **Identical models in velocity analysis.** Comparing two identical models stops with `attempt to get argmin of an empty sequence`.

---

## Recording a fix

The core functionality was tested at the development stage, and the workflow was refactored almost completely for publication. If something does not work, please raise an issue in the repository and describe it in four parts:

1. **Problem** — what fails, in one or two sentences, with the command, configuration and the error message.
2. **Root cause** — where in the code and why it happens.
3. **Potential solution** — the change that fixes it.
4. **Behaviour before and after** — what the workflow does now and what it will do after the fix (outputs, file formats, defaults).

When the fix is merged, add an entry to [Fixes after v4.0.0](#fixes-after-v400) with the same four parts and refer to it from the issue. Keep entries short; newest first.

Template:

```markdown
### YYYY-MM-DD — <short title>

- **Problem:** …
- **Root cause:** `path/to/file.py` — …
- **Solution:** …
- **Behaviour before:** …
- **Behaviour after:** …
- **Issue:** #<number>
```

---

## Fixes after v4.0.0

No fixes recorded yet.
