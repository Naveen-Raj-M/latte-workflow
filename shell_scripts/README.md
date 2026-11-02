# Shell Scripts

Scripts for running LATTE on a cluster (Slurm) and processing its output with the
`postprocess` and `analysis` modules. All paths in the scripts are placeholders
(`/path/to/...`, `your_allocation`, ...); edit them before use.

| File | What it does |
|---|---|
| `forward_job_submission.slurm` | Slurm job: one LATTE forward run (`x_eikonal3`) from a parameter file. |
| `inversion_job_submission.slurm` | Slurm job: one LATTE inversion (`x_tloc3`) per run name in `RUN_NAMES`. |
| `checkerboard_test_job_submission.slurm` | Slurm job: runs the checkerboard test for every name in `tests`. |
| `checkerboard_test_workflow.sh` | Called by the job above, once per test. Not run directly. |
| `spike_test_job_submission.slurm` | Slurm job: runs the spike test for every name in `tests`. |
| `spike_test_workflow.sh` | Called by the job above, once per test. Not run directly. |
| `process_inversion.sh` | Converts LATTE output to CSV (`postprocess.main`) and runs travel time, velocity and source analysis (`analysis.main`) for chosen runs and iterations. |

## Common setup

Every `.slurm` file loads the Intel and MPI modules (`intel/25.1`, `impi`), activates
the Python virtual environment, and sets `OMP_NUM_THREADS`. Before submitting, edit:

- `#SBATCH` header: partition (`-p`), allocation (`-A`), wall time (`-t`), email.
- `-N` and `-n`: nodes = MPI ranks x OpenMP threads / cores per node. `-n` must equal
  `NUM_RANK` in the script (the number of receivers in the final data).
- Paths: virtual environment, `BINDIR` (directory with `x_tloc3` / `x_eikonal3`), and
  the run directories.

## Forward modelling

```bash
sbatch /home/rajmanoharann/latte-workflow/shell_scripts/forward_job_submission.slurm
```

Set `RUN`, `BASE_DIR`, `EXECUTABLE`. Needs `<BASE_DIR>/<RUN>.rb`.
Log: `<BASE_DIR>/logs/run_<RUN>_simulation_log_<timestamp>.txt`.

## Inversion

```bash
sbatch /home/rajmanoharann/latte-workflow/shell_scripts/inversion_job_submission.slurm
```

Set `BASE_DIR` and `RUN_NAMES`. Each run needs `<BASE_DIR>/<run>.rb`; LATTE writes
`<BASE_DIR>/<run>/iteration_<N>/{model,synthetic}/`. Runs go one after another inside
the same job, so size `-t` for all of them. Log per run: `<BASE_DIR>/logs/`.

## Checkerboard and spike tests

```bash
sbatch /home/rajmanoharann/latte-workflow/shell_scripts/checkerboard_test_job_submission.slurm
sbatch /home/rajmanoharann/latte-workflow/shell_scripts/spike_test_job_submission.slurm
```

Prerequisite: create the perturbed model first with `checkerboard_test.main` or
`spike_test.main` (`workflow.step1_create_checkerboard=true` / `workflow.step1_create_spike=true`).
This writes the model binaries and you provide the LATTE parameter files:

```
<BASE_..._TEST_DIR>/inversion_runs_perturbed/<test>.rb   # LATTE on the perturbed model
<BASE_..._TEST_DIR>/inversion_runs/<test>.rb             # LATTE inversion of noisy data
```

The config named by `CONFIG_FILE` must live in `configs/checkerboard_test/` or
`configs/spike_test/` and include `default` in its `defaults` list.

In the job file, list the test names in `tests` and edit the `CFG` block. Each test
runs these steps, switched on or off by the `RUN_*` flags (a failed test is logged and
the next one still runs):

1. `RUN_INVERSION_PERTURBED`: LATTE on the perturbed model (one iteration) to produce synthetic travel times.
2. `RUN_ADD_NOISE`: adds noise to those travel times (`*.main`, `workflow.step2_add_noise`), giving `<test>/noisy_traveltimes/`.
3. `RUN_INVERSION`: LATTE inversion of the noisy travel times.

Set `SOURCE_RECEIVER_EXCHANGE=true` if LATTE used stations as virtual sources.
LATTE logs go to `inversion_runs_perturbed/logs/` and `inversion_runs/logs/`.

To resume after a failure, set the finished steps' flags to `false` and resubmit.

Converting and visualizing the inverted model (`workflow.step3_convert_inverted`,
`workflow.step4_visualize`) is not part of these scripts; run `checkerboard_test.main`
or `spike_test.main` manually for those.

## Processing inversion output

`process_inversion.sh` runs on a login or compute node, not through Slurm. Edit the
CONFIGURATION section at the top of the script, then:

```bash
bash /home/rajmanoharann/latte-workflow/shell_scripts/process_inversion.sh > /path/to/process_inversion.log 2>&1 &
```

Follow progress with `tail -f /path/to/process_inversion.log`.

For each run in `RUN_NAMES` and each selected iteration (`ITERATION_MODE` = `range`,
`step` or `list`) it reads `<BASE_INVERSION_DIR>/<run>/iteration_<N>/` and writes to
`<BASE_ANALYSIS_DIR>/<run>/iteration_<N>/`:

| Stage | Switch | Output |
|---|---|---|
| Postprocess velocity | `DO_VELOCITY_PROCESSING` | `velocity_model_final.csv` |
| Postprocess travel time | `DO_TRAVELTIME_PROCESSING` | `synthetic_traveltimes.csv` |
| Postprocess source | `DO_SOURCE_PROCESSING` | `source_relocation.csv` |
| Analysis travel time | `DO_TRAVELTIME_ANALYSIS` | `traveltime_analysis/` |
| Analysis velocity | `DO_VELOCITY_ANALYSIS` | `velocity_analysis/` (starting model vs. inverted model) |
| Analysis source | `DO_SOURCE_ANALYSIS` | `source_analysis/` |

Notes:

- Run it from anywhere; the script changes to `LATTE_WORKFLOW_DIR` itself.
- Source processing reuses the travel time CSV, so run travel time processing first (or in the same pass).
- For checkerboard and spike tests, set `GROUND_TRUTH_CSV` to the noisy travel time CSV (`<test>/noisy_traveltimes/noisy_traveltimes.csv`).
- Set `SOURCE_RECEIVER_EXCHANGE=true` (with `GEOMETRY_DIR`) if the inversion used it.
- Failed iterations are skipped and listed at the end; the script then exits with status 1.
- Both stages use the default configs (`configs/postprocess/default.yaml`, `configs/analysis/default.yaml`); all settings are passed as Hydra overrides.
