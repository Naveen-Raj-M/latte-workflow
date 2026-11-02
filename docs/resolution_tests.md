# Resolution Tests

The resolution tests assess how well an inversion can recover known structure. A synthetic perturbation is added to the starting model, LATTE computes synthetic travel times for the perturbed model, noise is added to them, and the noisy data are inverted. The recovered model is then compared with the input perturbation.

Two tests share one workflow:

| Test | Perturbation | Module |
|---|---|---|
| **Checkerboard** | Periodic pattern of alternating positive and negative anomalies | `checkerboard_test` |
| **Spike** | One or more localised Gaussian anomalies of arbitrary shape and orientation | `spike_test` |

> **Configuration reference:** The configuration keys are defined in [`configs/checkerboard_test/default.yaml`](../configs/checkerboard_test/default.yaml) and [`configs/spike_test/default.yaml`](../configs/spike_test/default.yaml); their comments describe every key. The two files have the same structure.

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

Each test is driven by `<test>.main` and consists of four steps, switched on or off in the `workflow` section. LATTE runs between the steps and is not started by the Python code; see [`shell_scripts/README.md`](../shell_scripts/README.md) for the job scripts.

```
step 1  create perturbed model        ──►  LATTE forward run on the perturbed model (one iteration)
step 2  add noise to travel times     ──►  LATTE inversion of the noisy travel times
step 3  convert the inverted model
step 4  visualize
```

---

### 1.1 Step 1 — Create the Perturbed Model

`workflow.step1_create_checkerboard` / `workflow.step1_create_spike`

Reads the starting model written by preprocess (CSV or HDF5) and writes the perturbed model as CSV and/or HDF5 plus the LATTE binaries.

**Checkerboard (`create_checkerboard_model.py`):**

- `pattern_type`: `sinusoidal` or `block`.
- `pattern_dimension`: `3d`, `2d` (no depth variation) or `planar` (anomalies on listed depth planes, with positive and negative polarity set by `depth_planes_positive_polarity` and `depth_planes_negative_polarity`).
- `amplitude` is the fractional perturbation (0.05 = ±5%); `wavelength_{x,y,z}_km` are the cell sizes and `phase_shift_{x,y,z}` the phase shifts; `origin_polarity` flips the pattern.

**Spike (`create_spike_model.py`):**

- Each entry of `spike.anomalies` has a `name`, a `type`, a `center` (km), `sigma` (km), and independent `amplitude_p`, `amplitude_s`, `polarity_p`, `polarity_s`.
- `type: 3d` is an axis-aligned Gaussian ellipsoid (`sigma = [σx, σy, σz]`); `2d` is a Gaussian blob in one depth layer selected by the z value of the centre (`sigma = [σx, σy]`); `arbitrary` is a Gaussian ellipsoid whose major axis follows the vector `direction` (`sigma = [σmajor, σminor₁, σminor₂]`).
- Perturbations of overlapping anomalies are summed before they are applied to the model. `truncate_at_sigma` sets the anomaly to zero beyond that many standard deviations (`null` = no truncation).

Both tests clip the perturbed velocities to `vp_min`/`vp_max` and `vs_min`/`vs_max` (m/s) and can restrict the perturbation to Vp or Vs (`apply_to_vp`, `apply_to_vs`). The binaries are always written, in m/s, float32.

---

### 1.2 Step 2 — Add Noise to the Synthetic Travel Times

`workflow.step2_add_noise` (`add_traveltime_noise.py`, shared by both tests)

Reads the synthetic travel time binaries computed by LATTE for the perturbed model and writes noisy versions that serve as the observed data of the inversion.

- `noise_type`: `gaussian` or `uniform`; `p_wave` and `s_wave` override the mean and standard deviation per phase.
- `magnitude_distance_noise` optionally derives the noise of each pick from the event magnitude and the hypocentral distance (a signal-to-noise model with the parameters `S0`, `gamma`, `kappa`, `snr_max` and the dominant frequencies).
- With `source_receiver_exchange: true` the binaries are read in LATTE's exchange layout and mapped back to the original shots, using the geometry directory (`input.geometry_dir`) and optionally `input.exchange_geometry_file`. `filtered_shot_ids` and `filtered_receiver_ids` list shots and stations that LATTE dropped.
- Writes the noisy binaries and `noisy_traveltimes.csv`, which carries the ground truth metadata with the noisy values in `observed_tt`.
- The noisy binaries are written as float32, the only precision LATTE reads.

---

### 1.3 Step 3 — Convert the Inverted Model

`workflow.step3_convert_inverted` (`convert_inverted_model.py`)

Converts the inverted binaries `updated_vp.bin` and `updated_vs.bin` to the velocity format with the same code and the same keys as the postprocess [velocity stage](postprocess.md#11-velocity-processing-postprocess_velocitypy) (`model_conversion` section: `interpolation_mode`, `velocity_output_format`, `hdf5_options`, optional `model_grid`). The grid comes from the metadata of the input model.

---

### 1.4 Step 4 — Visualize

`workflow.step4_visualize` (`visualize_slices.py`, `visualize_model.py`, `visualize_difference.py`, `visualize_pattern.py`, `recovery_analysis.py`)

Plots slices of the models on the planes listed in `visualization.planes` (`xy` depth slices, `xz` and `yz` vertical slices), at `xy_slices`/`xz_slices`/`yz_slices` or at a regular spacing (`slice_spacing`), in grid indices or km (`slice_unit`, `spacing_unit`). The fields are `vp`, `vs` and `vpvs_ratio` (`visualization.velocity_field`).

| Plot group (switch) | Content |
|---|---|
| `visualize_input` | Starting model |
| `visualize_checkerboard` (spike test: `visualize_spike`) | Perturbed model |
| `visualize_inverted` | Inverted model |
| `visualize_difference` | Inverted minus perturbed model |
| `visualize_checkerboard_pattern` (spike test: `visualize_spike_pattern`) | Input pattern as percent perturbation of the starting model |
| `visualize_inverted_pattern` | Recovered pattern as percent perturbation of the starting model |

The pattern plots support a colour threshold (`threshold`, `mark_threshold_black`), a minimum colour range (`min_cbar_pct`) and **recovery contours** (`show_recovery_contours`, `recovery_threshold_pct`): a cell is fairly recovered when the recovered perturbation has the same sign as the input and its magnitude is at least the threshold. Basin boundaries can be overlaid on `xy` slices.

---

## 2. Outputs

All outputs are written under `general.output_dir`. See [`file_formats.md`](file_formats.md#4-resolution-tests) for the file formats.

| Path | Step | Description |
|---|---|---|
| `checkerboard_model/` or `spike_model/` | 1 | Perturbed model (CSV/HDF5) and `vp_*.bin`, `vs_*.bin` for LATTE |
| `noisy_traveltimes/` | 2 | Noisy travel time binaries and `noisy_traveltimes.csv` |
| `inverted_model/` | 3 | Inverted model (`velocity_model_final.csv` / `.h5`) |
| `visualizations/` | 4 | Slice plots |

---

## 3. Usage and Configuration

### Running the workflow

```bash
# Checkerboard test
python -m checkerboard_test.main --config-name default

# Spike test
python -m spike_test.main --config-name default
```

Copy `default.yaml` to a new file in the same directory (with `defaults: [default, _self_]` or a full copy), set the paths, and select it with `--config-name`. The usual sequence is:

```bash
# 1. perturbed model
python -m spike_test.main --config-name my_test workflow.step1_create_spike=true

# 2. (LATTE forward run on the perturbed model)

# 3. noisy travel times
python -m spike_test.main --config-name my_test workflow.step2_add_noise=true

# 4. (LATTE inversion of the noisy travel times)

# 5. convert and plot the inverted model
python -m spike_test.main --config-name my_test \
  workflow.step3_convert_inverted=true workflow.step4_visualize=true
```

All steps are disabled in `default.yaml` and are switched on from the command line or in the copied configuration. For the checkerboard test the step names are `step1_create_checkerboard`, `step2_add_noise`, `step3_convert_inverted`, `step4_visualize`; for the spike test `step1_create_spike` replaces the first.

On a cluster, the LATTE runs and steps 2 can be chained with the Slurm scripts in [`shell_scripts/`](../shell_scripts/README.md).

### Minimal configuration checklist

```yaml
general:
  output_dir: /path/to/output/spike_test

input:
  velocity_model_csv: /path/to/preprocess/velocity_model_init.csv   # or velocity_model_hdf5
  geometry_dir: /path/to/preprocess/geometry
  synthetic_traveltime_dir: /path/to/latte/forward_run/synthetic
  ground_truth_csv: /path/to/preprocess/geometry_csv/geometry_ground_truth_traveltimes.csv
  inverted_model_dir: /path/to/latte/inversion/model

traveltime_noise:
  noise_type: gaussian
  p_wave: {mean: 0.0, std: 0.01}     # seconds
  s_wave: {mean: 0.0, std: 0.02}

workflow:
  step1_create_spike: true
```

---

## 4. Logging

The log level is set by `general.log_level` (default `INFO`). Every step logs a banner (for example `STEP 4: Creating Visualizations`) and a summary of what it wrote. Hydra stores the resolved configuration in `outputs/YYYY-MM-DD/HH-MM-SS_<test>_<config_name>/`.

---

## 5. Troubleshooting

**Step 2 stops with a shot or receiver count mismatch**
LATTE dropped shots or receivers (for example through shot selection, or shots without valid receivers). List them in `traveltime_noise.filtered_shot_ids` and `filtered_receiver_ids`. In exchange mode also check that `input.geometry_dir` is the geometry used by LATTE.

**Step 2 cannot find travel time files**
The files must be named `shot_<id>_traveltime_p.bin` / `shot_<id>_traveltime_s.bin` and lie in `input.synthetic_traveltime_dir`. This is the `dir_synthetic` of the LATTE forward run.

**Step 3 reports a binary size mismatch**
The inverted binaries must match the grid of the input model. The grid is read from the model's metadata; `model_conversion.model_grid` is used only as an additional check.

**The perturbation is not visible after clipping**
The perturbed velocities are clipped to `vp_min`/`vp_max` and `vs_min`/`vs_max`. Widen the bounds or reduce `amplitude`.

**An anomaly of the spike test is missing**
Check that `center` lies inside the model, and that `sigma` is large enough compared with the grid spacing; an anomaly narrower than one cell vanishes. For `type: 2d` the z value of `center` selects the layer.

**No recovery contours**
Set `show_recovery_contours: true` in the pattern plot sections and choose a `recovery_threshold_pct` below the amplitude of the input pattern.

---

## 6. Best Practices

- **Use the same geometry, weights and phases as the real inversion.** A resolution test is only meaningful when the data coverage is identical; run it on the same preprocess output.
- **Use realistic noise.** Take the standard deviations from the pick uncertainties of the real data, or use the magnitude- and distance-dependent model.
- **Choose anomaly sizes with the ray coverage in mind.** Checkerboard cells smaller than the resolution length produce no meaningful recovery; vary `wavelength_*_km` or the spike `sigma` to map the resolution.
- **Check the perturbation before running LATTE.** Enable `visualize_checkerboard` (spike test: `visualize_spike`) and inspect the slices first.
- **Keep inversion settings identical to the real run.** Regularisation, step sizes and number of iterations determine the recovery.
- **Use the recovery contours to mark well-resolved regions** in figures; state the threshold in the caption.
- **Keep single precision.** All binaries exchanged with LATTE are float32.
