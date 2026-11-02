# Checkerboard Test Configuration

## Overview

The checkerboard test checks how well a LATTE inversion recovers a known periodic pattern. A perturbed model is created from the starting model, LATTE computes synthetic travel times for it, noise is added to them, and the noisy data are inverted. The recovered model is compared with the input pattern. See [`docs/resolution_tests.md`](../../docs/resolution_tests.md) for the workflow and [`shell_scripts/README.md`](../../shell_scripts/README.md) for the cluster jobs.

**Steps (switched in `workflow`, LATTE runs between them):**
1. Create the checkerboard model (`checkerboard`)
2. Add noise to the synthetic travel times (`traveltime_noise`)
3. Convert the inverted model (`model_conversion`)
4. Visualize (`visualization`)

The spike test uses the same structure; see [`configs/spike_test/README.md`](../spike_test/README.md).

---

## Quick Start

### Minimal configuration

Create a new file in `configs/checkerboard_test/` that includes `default` and sets the paths:

```yaml
# configs/checkerboard_test/my_test.yaml
defaults:
  - default
  - _self_

general:
  output_dir: "/path/to/output/checkerboard_test"

input:
  velocity_model_csv: "/path/to/preprocess/velocity_model_init.csv"
  geometry_dir: "/path/to/preprocess/geometry"
  synthetic_traveltime_dir: "/path/to/latte/forward_run/synthetic"
  ground_truth_csv: "/path/to/preprocess/geometry_csv/geometry_ground_truth_traveltimes.csv"
  inverted_model_dir: "/path/to/latte/inversion/model"

checkerboard:
  wavelength_x_km: 20.0
  wavelength_y_km: 20.0
  wavelength_z_km: 10.0
  amplitude: 0.05

workflow:
  step1_create_checkerboard: true
```

All other parameters inherit their defaults from `default.yaml`. All four workflow steps are `false` by default.

### Run

```bash
python -m checkerboard_test.main --config-name my_test
```

Run the steps in order, with LATTE in between:

```bash
python -m checkerboard_test.main --config-name my_test workflow.step1_create_checkerboard=true
# LATTE forward run on <output_dir>/checkerboard_model/vp_checkerboard.bin, vs_checkerboard.bin
python -m checkerboard_test.main --config-name my_test workflow.step2_add_noise=true
# LATTE inversion of <output_dir>/noisy_traveltimes
python -m checkerboard_test.main --config-name my_test \
    workflow.step3_convert_inverted=true workflow.step4_visualize=true
```

---

## Configuration Reference

### `general`

| Key | Default | Description |
|-----|---------|-------------|
| `output_dir` | `"/path/to/output/checkerboard_test"` | Root of all outputs (`checkerboard_model/`, `noisy_traveltimes/`, `inverted_model/`, `visualizations/`). |
| `log_level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `random_seed` | `42` | Seed of the noise generator; `null` = random. |

---

### `input`

| Key | Default | Description |
|-----|---------|-------------|
| `velocity_model_csv` | — | Starting model written by preprocess (`x, y, z, vp, vs`). |
| `velocity_model_hdf5` | `null` | Starting model as HDF5; takes precedence over the CSV. |
| `geometry_dir` | — | Preprocess geometry directory (reused for the noise step and for exchange runs). |
| `synthetic_traveltime_dir` | — | LATTE forward-run output with `shot_<id>_traveltime_{p,s}.bin`. |
| `ground_truth_csv` | — | Preprocess `geometry_ground_truth_traveltimes.csv` (metadata of the noisy table). |
| `inverted_model_dir` | — | LATTE inversion output with `updated_vp.bin`, `updated_vs.bin`. |
| `exchange_geometry_file` | `null` | Optional LATTE `exchange_geometry.txt` (`virtual_id x y z`, metres), used as the virtual-source table in exchange mode; `geometry_dir` then cross-checks it. |

---

### `checkerboard`

Step 1: builds the perturbed model.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `true` | — | Enable the checkerboard creation. |
| `pattern_type` | `sinusoidal` | `sinusoidal`, `block` | Smooth or sharp-edged cells. |
| `pattern_dimension` | `3d` | `3d`, `2d`, `planar` | `3d`: variation in x, y, z. `2d`: no depth variation. `planar`: anomalies only on the listed depth planes. |
| `origin_polarity` | `positive` | `positive`, `negative` | `negative` inverts the pattern. Ignored for `planar`. |
| `depth_planes_positive_polarity` | `[]` | km | Depth planes with a positive anomaly (`planar`; snapped to the nearest grid point). |
| `depth_planes_negative_polarity` | `[]` | km | Depth planes with a negative anomaly. Both lists empty is an error for `planar`. |
| `amplitude` | `0.05` | — | Fractional perturbation (0.05 = ±5%). |
| `wavelength_x_km`, `wavelength_y_km`, `wavelength_z_km` | `10.0`, `10.0`, `5.0` | km | Cell sizes. `wavelength_z_km` is ignored for `2d` and `planar`. |
| `phase_shift_x`, `phase_shift_y`, `phase_shift_z` | `0.0` | radians | Phase shifts. `phase_shift_z` is ignored for `2d` and `planar`. |
| `apply_to_vp`, `apply_to_vs` | `true` | — | Fields that are perturbed. |
| `vp_min`, `vp_max` | `500.0`, `10000.0` | m/s | Clip range of the perturbed Vp. |
| `vs_min`, `vs_max` | `500.0`, `6000.0` | m/s | Clip range of the perturbed Vs. |
| `output_format` | `csv` | `csv`, `hdf5`, `both` | Format of the perturbed model. The LATTE binaries are always written. |
| `output_csv` | `${general.output_dir}/checkerboard_model/velocity_model_checkerboard.csv` | — | CSV path. |
| `output_hdf5` | `${general.output_dir}/checkerboard_model/velocity_model_checkerboard.h5` | — | HDF5 path. |

The binaries `vp_checkerboard.bin` and `vs_checkerboard.bin` (m/s, float32) are written to `checkerboard_model/`.

**`hdf5_options`** (with `output_format` `hdf5` or `both`):

| Key | Default | Description |
|-----|---------|-------------|
| `compression` | `gzip` | `gzip`, `lzf`, or `null`. |
| `compression_level` | `4` | Gzip level 0–9. |
| `parallel_processing` | `true` | Parallel coarse-grid subsampling. |
| `n_jobs` | `-1` | `-1` = all cores. |
| `show_progress` | `true` | Progress bars. |
| `create_coarse_dataset` | `false` | Add a coarse dataset to the HDF5 file. |
| `coarse_grid.dx/dy/dz` | `2.0` | km; integer multiples of the fine spacing. |

---

### `traveltime_noise`

Step 2: adds noise to the synthetic travel times of the perturbed model.

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `enabled` | `true` | — | Enable the noise step. |
| `source_receiver_exchange` | `false` | — | `true` if the LATTE forward run used the stations as virtual sources. |
| `filtered_shot_ids` | `[]` | — | Shots dropped by LATTE (`sid_select`, `sid_exclude`, no valid receivers). |
| `filtered_receiver_ids` | `[]` | — | Station names dropped by LATTE. |
| `coordinate_tolerance` | `0.01` | m | Tolerance for matching geometry receivers to ground truth stations. |
| `noise_type` | `gaussian` | `gaussian`, `uniform` | Noise distribution. |
| `gaussian.mean`, `gaussian.std` | `0.0`, `0.01` | s | Gaussian parameters. |
| `uniform.low`, `uniform.high` | `-0.02`, `0.02` | s | Uniform range. |
| `p_wave.mean`, `p_wave.std` | `0.0`, `0.01` | s | Phase-specific override for P. |
| `s_wave.mean`, `s_wave.std` | `0.0`, `0.02` | s | Phase-specific override for S. |

The noisy binaries (float32) and `noisy_traveltimes.csv` are written to `noisy_traveltimes/`.

**`magnitude_distance_noise`**: per-pick noise from a signal-to-noise model, `σ_t = 1 / (2π f · SNR)`.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Use this model instead of the fixed noise. |
| `S0` | `0.2` | Reference SNR amplitude. |
| `gamma` | `1.0` | Geometric spreading exponent. |
| `kappa` | `0.003` | Attenuation coefficient (1/km). |
| `snr_max` | `50.0` | Cap of the SNR. |
| `p_wave_frequency`, `s_wave_frequency` | `10.0`, `5.0` | Hz; dominant frequencies of P and S. |

---

### `model_conversion`

Step 3: converts the inverted binaries to the velocity format with the same code and keys as the postprocess [`velocity_processing`](../postprocess/README.md#velocity_processing).

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable the conversion. |
| `phase_type` | `both` | `both`, `p`, `s`. |
| `output_dir` | `${general.output_dir}/inverted_model` | Output directory. |
| `vp_binary`, `vs_binary` | `${input.inverted_model_dir}/updated_{vp,vs}.bin` | Inverted binaries. |
| `model_grid.*` | see `default.yaml` | Optional check of the binary grid (`nx`, `ny`, `nz`, `dx`, `dy`, `dz`, `origin_x/y/z`). The grid comes from the metadata of the input model; `model_grid` is used only if that grid does not match the binaries. |
| `interpolation_mode` | `sample_at_coarse` | `sample_at_coarse` or `expand_to_fine`. |
| `velocity_output_format` | `csv` | `csv`, `hdf5`, `both`. |
| `velocity_output_csv`, `velocity_output_hdf5` | `${model_conversion.output_dir}/velocity_model_final.{csv,h5}` | Output paths. |
| `hdf5_options.*` | see `default.yaml` | Compression, coarse dataset and coarse CSV, as in postprocess. |
| `performance.n_jobs`, `performance.show_progress` | `-1`, `true` | Parallelism and progress bars. |

---

### `visualization`

Step 4: slice plots of the models.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Master switch. |
| `output_dir` | `null` | Output directory; `null` = `${general.output_dir}/visualizations`. |
| `inverted_model_dir` | `null` | Directory of an inverted model to plot instead of the one written by step 3. |
| `planes` | `["xy", "xz", "yz"]` | `xy` depth slices, `xz` and `yz` vertical slices. |
| `velocity_field` | `["vp"]` | Any of `vp`, `vs`, `vpvs_ratio`. |
| `slice_unit` | `indices` | `indices` or `km`; unit of `xy_slices`, `xz_slices`, `yz_slices`. |
| `spacing_unit` | `indices` | `indices` or `km`; unit of `slice_spacing`. |
| `tolerance_km` | `1e-6` | Tolerance of the km-to-index conversion. |
| `xy_slices`, `xz_slices`, `yz_slices` | `null` | Explicit slice positions; `null` = use `slice_spacing`. |
| `slice_spacing.xy/xz/yz` | `5`, `20`, `20` | Regular slice spacing. |
| `figure.dpi`, `figure.figsize`, `figure.format` | `150`, `[10, 8]`, `png` | `png`, `pdf`, `svg`. |
| `figure.recovery_contour_color`, `…_linestyle`, `…_linewidth` | `green`, `-`, `2` | Style of the recovery contours. |
| `show_colorbar`, `show_grid`, `show_title` | `true` | Annotations. |
| `font_size.title/labels/colorbar/ticks` | `14`, `12`, `10`, `10` | Font sizes. |
| `vpvs_ratio.colormap` | `seismic` | Colormap of the Vp/Vs plots. |
| `vpvs_ratio.uniform_cbar` | `false` | One colour range for all slices. |
| `vpvs_ratio.range` | `null` | `[min, max]`; values outside are drawn black. `null` = autoscale. |
| `basin_boundaries.*` | disabled | Basin outlines on `xy` slices (`enabled`, `data_dir`, `basins`, line and label style). |

**Plot groups.** Each group has `enabled` (default `false`) and `colormap` (default `seismic`):

| Group | Content | Additional keys |
|-------|---------|-----------------|
| `visualize_input` | Starting model | — |
| `visualize_checkerboard` | Perturbed model | — |
| `visualize_inverted` | Inverted model | — |
| `visualize_difference` | Inverted minus perturbed model (Vp, Vs) | `centered_at_zero` (`true`), `uniform_cbar` (`false`) |
| `visualize_checkerboard_pattern` | Input pattern, % perturbation of the starting model (Vp/Vs ratio for `vpvs_ratio`) | `centered_at_zero`, `uniform_cbar`, `threshold` (`null`), `mark_threshold_black` (`true`), `show_recovery_contours` (`false`), `recovery_threshold_pct` (`3.0`) |
| `visualize_inverted_pattern` | Recovered pattern, % perturbation of the starting model | as above, plus `min_cbar_pct` (`null`) |

`threshold` limits the colour range to ±threshold %; with `mark_threshold_black: true` values beyond it are drawn black, otherwise they saturate at the colour of the threshold. `min_cbar_pct` makes the colour bar show at least ±that value. A cell is fairly recovered (`show_recovery_contours`) when the recovered perturbation has the same sign as the input and a magnitude of at least `recovery_threshold_pct`. `centered_at_zero`, `threshold` and the recovery contours apply to Vp and Vs only.

---

### `workflow`

| Key | Default | Description |
|-----|---------|-------------|
| `step1_create_checkerboard` | `false` | Create the perturbed model. |
| `step2_add_noise` | `false` | Add noise to the synthetic travel times. |
| `step3_convert_inverted` | `false` | Convert the inverted model. |
| `step4_visualize` | `false` | Create the plots. |

---

## Output Structure

```
output_dir/
├── checkerboard_model/      # velocity_model_checkerboard.{csv,h5}, vp_checkerboard.bin, vs_checkerboard.bin
├── noisy_traveltimes/       # shot_<N>_traveltime_{p,s}.bin, noisy_traveltimes.csv
├── inverted_model/          # velocity_model_final.{csv,h5}
└── visualizations/
```

Hydra also writes the resolved configuration and the log to:
```
outputs/<YYYY-MM-DD>/<HH-MM-SS>_checkerboard_test_<config_name>/
```

---

## Important Notes

- **LATTE runs between the steps and is not started by this code.** Step 2 needs the forward run on the perturbed model, step 3 the inversion of the noisy data.
- **Use the same geometry and weights as the real inversion**, so that the data coverage is identical.
- **Perturbed velocities are clipped** to the `vp_*`/`vs_*` bounds; a large `amplitude` may be cut.
- **Exchange runs:** set `source_receiver_exchange: true` and list shots and stations dropped by LATTE in `filtered_shot_ids` / `filtered_receiver_ids`; otherwise a count mismatch raises an error.
- **Binaries are float32 (m/s for models, s for travel times).**
- **The grid comes from the metadata of the input model.** `model_grid` is only an additional check.
