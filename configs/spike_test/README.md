# Spike Test Configuration

## Overview

The spike test checks how well a LATTE inversion recovers localised anomalies. One or more Gaussian anomalies of arbitrary shape, orientation, amplitude and polarity are added to the starting model, LATTE computes synthetic travel times for the perturbed model, noise is added to them, and the noisy data are inverted. The recovered model is compared with the input anomalies. See [`docs/resolution_tests.md`](../../docs/resolution_tests.md) for the workflow and [`shell_scripts/README.md`](../../shell_scripts/README.md) for the cluster jobs.

**Steps (switched in `workflow`, LATTE runs between them):**
1. Create the spike model (`spike`)
2. Add noise to the synthetic travel times (`traveltime_noise`)
3. Convert the inverted model (`model_conversion`)
4. Visualize (`visualization`)

`spike_test` has the same structure as the checkerboard test. Only the model creation (`spike`), the names of two plot groups and the workflow switch of step 1 differ. All other sections have the same keys, defaults and meaning as in [`configs/checkerboard_test/README.md`](../checkerboard_test/README.md):

| Section | Reference |
|---------|-----------|
| `general`, `input` | [checkerboard `general`, `input`](../checkerboard_test/README.md#general) |
| `traveltime_noise` | [checkerboard `traveltime_noise`](../checkerboard_test/README.md#traveltime_noise) |
| `model_conversion` | [checkerboard `model_conversion`](../checkerboard_test/README.md#model_conversion) |
| `visualization` | [checkerboard `visualization`](../checkerboard_test/README.md#visualization) |

---

## Quick Start

### Minimal configuration

Create a new file in `configs/spike_test/` that includes `default` and sets the paths and anomalies:

```yaml
# configs/spike_test/my_test.yaml
defaults:
  - default
  - _self_

general:
  output_dir: "/path/to/output/spike_test"

input:
  velocity_model_csv: "/path/to/preprocess/velocity_model_init.csv"
  geometry_dir: "/path/to/preprocess/geometry"
  synthetic_traveltime_dir: "/path/to/latte/forward_run/synthetic"
  ground_truth_csv: "/path/to/preprocess/geometry_csv/geometry_ground_truth_traveltimes.csv"
  inverted_model_dir: "/path/to/latte/inversion/model"

spike:
  anomalies:
    - name: "blob"
      type: "3d"
      center: [100.0, 80.0, 10.0]
      sigma: [5.0, 5.0, 3.0]
      amplitude_p: 0.10
      amplitude_s: 0.10
      polarity_p: "positive"
      polarity_s: "positive"

workflow:
  step1_create_spike: true
```

All other parameters inherit their defaults from `default.yaml`. All four workflow steps are `false` by default. A list in a derived configuration replaces the list of `default.yaml`, so `anomalies` must be given in full.

### Run

```bash
python -m spike_test.main --config-name my_test
```

Run the steps in order, with LATTE in between:

```bash
python -m spike_test.main --config-name my_test workflow.step1_create_spike=true
# LATTE forward run on <output_dir>/spike_model/vp_spike.bin, vs_spike.bin
python -m spike_test.main --config-name my_test workflow.step2_add_noise=true
# LATTE inversion of <output_dir>/noisy_traveltimes
python -m spike_test.main --config-name my_test \
    workflow.step3_convert_inverted=true workflow.step4_visualize=true
```

---

## Configuration Reference

### `spike`

Step 1: builds the perturbed model. The perturbations of all anomalies are summed and applied multiplicatively to the model.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable the spike creation. |
| `apply_to_vp`, `apply_to_vs` | `true` | Global switches for the fields that are perturbed (per-anomaly amplitudes still apply). |
| `vp_min`, `vp_max` | `500.0`, `10000.0` | m/s; clip range of the perturbed Vp. |
| `vs_min`, `vs_max` | `500.0`, `6000.0` | m/s; clip range of the perturbed Vs. |
| `truncate_at_sigma` | `4.0` | The Gaussian is set to zero beyond this many standard deviations from the centre. `null` = no truncation. Can be overridden per anomaly. |
| `anomalies` | 3 examples | List of anomalies (below). |
| `output_format` | `csv` | `csv`, `hdf5`, `both`. The LATTE binaries are always written. |
| `output_csv` | `${general.output_dir}/spike_model/velocity_model_spike.csv` | CSV path. |
| `output_hdf5` | `${general.output_dir}/spike_model/velocity_model_spike.h5` | HDF5 path. |
| `hdf5_options.*` | see `default.yaml` | `compression`, `compression_level`, `parallel_processing`, `n_jobs`, `show_progress`, `create_coarse_dataset`, `coarse_grid.dx/dy/dz`; same as in the checkerboard test. |

The binaries `vp_spike.bin` and `vs_spike.bin` (m/s, float32) are written to `spike_model/`.

#### `spike.anomalies`

Every entry is independent. Common fields:

| Key | Description |
|-----|-------------|
| `name` | Label used in the log. |
| `type` | `3d`, `2d` or `arbitrary`. |
| `center` | `[x, y, z]` in km (Cartesian frame of the model). |
| `sigma` | Standard deviations in km; the length depends on `type`. |
| `amplitude_p`, `amplitude_s` | Peak fractional perturbation of Vp and Vs (0.10 = 10%). |
| `polarity_p`, `polarity_s` | `positive` or `negative`. |
| `truncate_at_sigma` | Optional override of the global truncation. |

Shape fields:

| `type` | Shape | `sigma` | Extra field |
|--------|-------|---------|-------------|
| `3d` | Axis-aligned Gaussian ellipsoid | `[σx, σy, σz]` | — |
| `2d` | Gaussian blob confined to one depth layer; the z value of `center` selects the layer | `[σx, σy]` | — |
| `arbitrary` | Gaussian ellipsoid whose major axis follows a direction | `[σmajor, σminor₁, σminor₂]` | `direction: [dx, dy, dz]`, the major axis; optional `minor1_direction: [x, y, z]` fixes the first minor axis (otherwise a perpendicular pair is chosen automatically) |

Example with all three kinds:

```yaml
spike:
  anomalies:
    - name: "blob_3d"
      type: "3d"
      center: [100.0, 80.0, 10.0]
      sigma: [5.0, 5.0, 3.0]
      amplitude_p: 0.10
      amplitude_s: 0.10
      polarity_p: "positive"
      polarity_s: "positive"
    - name: "blob_2d_layer"
      type: "2d"
      center: [120.0, 90.0, 15.0]     # z = 15 km selects the layer
      sigma: [8.0, 4.0]
      amplitude_p: 0.08
      amplitude_s: 0.08
      polarity_p: "negative"
      polarity_s: "negative"
    - name: "elongated_dyke"
      type: "arbitrary"
      center: [80.0, 70.0, 12.0]
      direction: [1.0, 1.0, 0.5]      # major axis points NE and slightly down
      sigma: [12.0, 3.0, 3.0]
      amplitude_p: 0.12
      amplitude_s: 0.10
      polarity_p: "positive"
      polarity_s: "negative"
```

P and S may have different amplitudes and opposite polarity, which also perturbs the Vp/Vs ratio.

---

### `visualization`

Same keys as in the checkerboard test. Differences in the plot groups:

| Spike test group | Checkerboard group | Content |
|------------------|--------------------|---------|
| `visualize_spike` | `visualize_checkerboard` | Perturbed model |
| `visualize_spike_pattern` | `visualize_checkerboard_pattern` | Input anomalies as % perturbation of the starting model. Keys: `centered_at_zero`, `uniform_cbar`, `threshold`, `show_recovery_contours`, `recovery_threshold_pct`. It has no `mark_threshold_black`: values beyond `threshold` are drawn black. |

`visualize_input`, `visualize_inverted`, `visualize_difference` and `visualize_inverted_pattern` are the same as in the checkerboard test.

---

### `workflow`

| Key | Default | Description |
|-----|---------|-------------|
| `step1_create_spike` | `false` | Create the perturbed model. |
| `step2_add_noise` | `false` | Add noise to the synthetic travel times. |
| `step3_convert_inverted` | `false` | Convert the inverted model. |
| `step4_visualize` | `false` | Create the plots. |

---

## Output Structure

```
output_dir/
├── spike_model/             # velocity_model_spike.{csv,h5}, vp_spike.bin, vs_spike.bin
├── noisy_traveltimes/       # shot_<N>_traveltime_{p,s}.bin, noisy_traveltimes.csv
├── inverted_model/          # velocity_model_final.{csv,h5}
└── visualizations/
```

Hydra also writes the resolved configuration and the log to:
```
outputs/<YYYY-MM-DD>/<HH-MM-SS>_spike_test_<config_name>/
```

---

## Important Notes

- **LATTE runs between the steps and is not started by this code.**
- **Anomalies narrower than a grid cell vanish.** Choose `sigma` larger than the grid spacing, and check that `center` lies inside the model.
- **Anomalies that overlap add up**, and the sum is clipped to the `vp_*`/`vs_*` bounds.
- **`2d` anomalies sit on one layer.** The layer is the grid depth nearest to the z value of `center`.
- **Exchange runs, filtered shots and the grid:** see the notes of the [checkerboard test](../checkerboard_test/README.md#important-notes).
