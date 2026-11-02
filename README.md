# LATTE Tomography Toolkit

A Python toolkit for seismic tomography data processing, analysis and visualization around the **LATTE** (Los Alamos Travel-Time package based on Eikonal Equation) framework.

## Overview

The toolkit covers regional and local-scale seismic tomography studies from end to end: preparing LATTE inputs from raw catalogs, converting inversion results into analysis-ready tables, diagnosing the results, and running resolution tests.

> **Recommended LATTE version:** The workflow can be used with the original [LATTE](https://github.com/lanl/latte_traveltime). A [forked LATTE repository](https://github.com/Naveen-Raj-M/latte_traveltime) is suggested, though not mandatory; it adds ray path computation and other improvements that this workflow uses (ray path and DWS processing).

## Related Publications and Code

- **LATTE Framework**: [LATTE: open-source, high-performance traveltime computation, tomography and source location in acoustic and elastic media](https://academic.oup.com/gji/article/241/2/1275/8046728) — Geophysical Journal International, 2025
- **Original LATTE code**: [LANL LATTE repository](https://github.com/lanl/latte_traveltime)
- **Forked LATTE code**: [Naveen-Raj-M/latte_traveltime](https://github.com/Naveen-Raj-M/latte_traveltime)

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Naveen-Raj-M/latte-workflow.git
cd latte-workflow
```

Create the Python environment with either conda or pip.

**conda (recommended):**

```bash
conda env create -f environment.yml
conda activate latte-workflow
```

`environment-tested.yml` pins the exact versions of the last test run:

```bash
conda env create -f environment-tested.yml
```

**pip:**

```bash
python -m venv latte-workflow-venv
source latte-workflow-venv/bin/activate
pip install -r requirements.txt
```

**Dependencies** (Python 3.10 – 3.13):

| Package | Used for |
|---|---|
| NumPy, pandas (2.x), SciPy | Numerical work and tables |
| Matplotlib | Plots |
| h5py, PyTables (`tables`) | HDF5 files; PyTables is needed by `pandas.read_hdf` / `to_hdf` |
| Hydra (`hydra-core`, brings OmegaConf and PyYAML) | Configuration |
| ObsPy | QuakeML parsing and geodetic distances |
| joblib, tqdm | Parallel processing and progress bars |
| Numba | Ray path simplification |

Check the installation:

```bash
python -c "import preprocess.main, postprocess.main, analysis.main; print('ok')"
```

### Basic Usage

All pipelines are driven by [Hydra](https://hydra.cc) configuration. Copy the relevant `example.yaml` from `configs/<pipeline>/`, update the file paths, and run it from the repository root:

```bash
# Preprocessing
python -m preprocess.main --config-name=example

# Post-processing
python -m postprocess.main --config-name=example

# Analysis
python -m analysis.main --config-name=example
```

Individual parameters can be overridden on the command line:

```bash
# A single parameter
python -m analysis.main --config-name=example travel_time_analysis.enabled=true

# Several parameters
python -m postprocess.main --config-name=example \
  velocity_processing.enabled=true \
  travel_time_processing.enabled=true
```

> The parameters, override syntax and example configurations are described in [`configs/preprocess/README.md`](configs/preprocess/README.md), [`configs/postprocess/README.md`](configs/postprocess/README.md) and [`configs/analysis/README.md`](configs/analysis/README.md).

---

## Workflow Components

### Preprocessing

The preprocessing pipeline transforms a raw catalog into the binary and tabular inputs of the LATTE inversion. It reads **QuakeML** or **HDF5** catalogs and produces geometry files, travel time tables and binaries, source binaries, and an initial 3D velocity model.

**Steps (executed in order):**
1. **QuakeML Loading** — parses the QuakeML files once (optionally with secondary catalogs) and applies per-step filters
2. **Raw Data Visualization** — maps and histograms of the QuakeML catalog, with their own filters (optional)
3. **HDF5 Conversion** — writes the filtered catalog to an HDF5 file for fast reuse (optional)
4. **Data Loading** — reads events, stations, picks and arrivals and applies the global filters
5. **Coordinate Transformation** — converts geodetic (lon/lat/depth) to local Cartesian (x/y/z km) coordinates with a first-order flattening transformation
6. **Geometry and Travel Time Generation** — writes source–receiver geometry, travel time binaries and source binaries in LATTE format, with optional azimuthal and separate P/S weights
7. **Initial Velocity Model** — extrudes a 1D Vp/Vs profile onto the 3D grid; writes CSV/HDF5 and the LATTE binaries
8. **Final Data Visualization** — quality-control plots of the processed data

Details: [docs/preprocess.md](docs/preprocess.md).

---

### Post-Processing

The post-processing pipeline converts the binary outputs of LATTE into CSV and HDF5 files. The stages are independent and disabled by default.

**Stages:**
1. **Velocity Processing** — reads the updated Vp/Vs binaries and writes them on the reference grid; optionally adds gradient and search direction fields
2. **Travel Time Processing** — merges observed and synthetic travel times, including source–receiver exchange runs
3. **Source Processing** — extracts updated hypocenters and origin times
4. **Ray Path Processing** — builds an HDF5 ray path cache from the LATTE ray path binaries and computes the Derivative Weight Sum (DWS)
5. **Ray Path Simplification** — shrinks the cache with the Ramer–Douglas–Peucker algorithm

Details: [docs/postprocess.md](docs/postprocess.md).

---

### Analysis

The analysis pipeline provides post-inversion diagnostics. It reads the post-processed files and writes plots, statistics and reports. All ten modules are independent and disabled by default.

**Modules:**
1. **Travel Time Residual Analysis** — residual distributions and statistics; ground truth and method-comparison modes
2. **Velocity Model Analysis** — differences and ratios between two models, projections, slices and profiles
3. **Distance-Dependent Residual Analysis** — residual dependence on source–receiver distance
4. **Short-Distance Analysis** — near-field residuals with spatial clusters
5. **Source Correction Analysis** — hypocenter and origin time corrections
6. **Gradient Analysis** — gradient and search direction diagnostics
7. **DWS Analysis** — Derivative Weight Sum coverage maps
8. **Ray Path Analysis** — ray path projections
9. **Ray Density Analysis** — ray counts per cell, depth slices and oblique sections
10. **Tradeoff Analysis** — data misfit versus model change across iterations or parameter values

Details: [docs/analysis.md](docs/analysis.md).

---

### Resolution Tests

The checkerboard and spike tests check how well an inversion recovers known structure: a perturbed model is created, noise is added to its synthetic travel times, the noisy data are inverted, and the recovered model is compared with the input.

- **Checkerboard test** (`checkerboard_test`) — periodic sinusoidal or block patterns in 3D, 2D or on selected depth planes
- **Spike test** (`spike_test`) — localised Gaussian anomalies (3D, single layer, arbitrary orientation)

Details: [docs/resolution_tests.md](docs/resolution_tests.md).

---

### Cluster Scripts

`shell_scripts/` contains Slurm jobs for LATTE forward runs, inversions and resolution tests, and a script that post-processes and analyses selected inversion iterations. Details: [shell_scripts/README.md](shell_scripts/README.md).

---

## Complete Workflow

```
1. Preprocessing            2. LATTE                    3. Post-Processing           4. Analysis
   ↓                           ↓                            ↓                            ↓
Input Data                 Binary Outputs               Analysis-Ready Files         Results & Plots
├─ Event catalog           ├─ updated_vp.bin            ├─ velocity_model_final.csv   ├─ Residual plots
│  (QuakeML / HDF5)        ├─ updated_vs.bin            ├─ synthetic_traveltimes.csv  ├─ Velocity comparisons
├─ Station inventory       ├─ updated_sx/sy/sz/st0.bin  ├─ source_relocation.csv      ├─ DWS coverage maps
└─ 1D reference model      ├─ synthetic travel times    ├─ ray_paths.h5               ├─ Ray path projections
                           └─ ray path binaries         └─ dws_values.h5              ├─ L-curve tradeoff
                                                                                       └─ Statistical reports
```

---

## Repository Structure

```
latte-workflow/
├── preprocess/                      # Preprocessing pipeline
│   ├── main.py
│   ├── quakeml_loader.py
│   ├── catalog_filters.py
│   ├── raw_data_visualization.py
│   ├── hdf5_conversion.py
│   ├── data_loader.py
│   ├── coordinate_transformation.py
│   ├── geometry_traveltime.py
│   ├── initial_velocity_model.py
│   ├── visualization.py
│   └── plot_settings.py
│
├── postprocess/                     # Post-processing pipeline
│   ├── main.py
│   ├── postprocess_velocity.py
│   ├── postprocess_traveltime.py
│   ├── postprocess_source.py
│   ├── compute_ray_paths.py
│   ├── read_raypath_binary.py
│   ├── postprocess_raypath.py
│   ├── postprocess_dws.py
│   └── simplify_ray_paths.py
│
├── analysis/                        # Analysis pipeline
│   ├── main.py
│   ├── filters.py
│   ├── traveltime_analysis.py
│   ├── velocity_analysis.py
│   ├── distance_analysis.py
│   ├── short_distance_analysis.py
│   ├── source_analysis.py
│   ├── gradient_analysis.py
│   ├── dws_analysis.py
│   ├── raypath_analysis.py
│   ├── ray_density_analysis.py
│   ├── tradeoff_analysis.py
│   └── [supporting data / statistics / plot / report modules]
│
├── checkerboard_test/               # Checkerboard resolution test
├── spike_test/                      # Spike resolution test
│
├── utils/                           # Shared utilities
│   ├── io_utils.py
│   ├── velocity_io.py
│   ├── traveltime_io.py
│   ├── csv_metadata.py
│   ├── column_utils.py
│   ├── exchange.py
│   ├── distance_utils.py
│   ├── origin_time_utils.py
│   ├── logging_utils.py
│   └── plot_utils.py
│
├── configs/                         # Hydra configuration files
│   ├── preprocess/                  # default.yaml, example.yaml, README.md
│   ├── postprocess/                 # default.yaml, example.yaml, README.md
│   ├── analysis/                    # default.yaml, example.yaml, README.md
│   ├── checkerboard_test/           # default.yaml, README.md
│   └── spike_test/                  # default.yaml, README.md
│
├── shell_scripts/                   # Slurm jobs and batch processing scripts
│
├── docs/                            # Detailed documentation
│   ├── preprocess.md
│   ├── postprocess.md
│   ├── analysis.md
│   ├── resolution_tests.md
│   ├── file_formats.md
│   └── update_notes.md
│
├── requirements.txt                 # pip dependencies
├── environment.yml                  # conda environment
└── environment-tested.yml           # conda environment with the tested versions
```

---

## Documentation

| Document | Description |
|---|---|
| [docs/preprocess.md](docs/preprocess.md) | Module reference for each preprocessing step, configuration keys and outputs |
| [docs/postprocess.md](docs/postprocess.md) | Module reference for each post-processing stage, configuration keys and outputs |
| [docs/analysis.md](docs/analysis.md) | Module reference for each analysis module, configuration keys and plots |
| [docs/resolution_tests.md](docs/resolution_tests.md) | Checkerboard and spike resolution tests |
| [docs/file_formats.md](docs/file_formats.md) | Specification of all input and output file formats |
| [docs/update_notes.md](docs/update_notes.md) | Changes of the releases, known limitations, and the log of fixes |
| [configs/preprocess/README.md](configs/preprocess/README.md) | All preprocessing parameters with defaults and descriptions |
| [configs/postprocess/README.md](configs/postprocess/README.md) | All post-processing parameters with defaults and descriptions |
| [configs/analysis/README.md](configs/analysis/README.md) | All analysis parameters with defaults and descriptions |
| [configs/checkerboard_test/README.md](configs/checkerboard_test/README.md) | All checkerboard test parameters with defaults and descriptions |
| [configs/spike_test/README.md](configs/spike_test/README.md) | All spike test parameters with defaults and descriptions |
| [shell_scripts/README.md](shell_scripts/README.md) | Slurm jobs and processing scripts |

---

## Compatibility

- **Python**: 3.10 – 3.13
- **LATTE**: the original [LATTE](https://github.com/lanl/latte_traveltime) or the [forked repository](https://github.com/Naveen-Raj-M/latte_traveltime) (suggested; adds ray path computation and other improvements)
- **Binary format**: all binary files exchanged with LATTE are headerless little-endian **float32** (Fortran stream access); LATTE reads and writes single precision only
- **Operating system**: developed and tested on Linux
- **Input formats**: QuakeML, HDF5, plain-text station files

---

## Reporting Issues

The core functionality was tested at the development stage. Almost the whole workflow was then refactored to prepare this release, and the core functionality was tested again with the refactored code. Some parts may therefore still contain bugs. If you find one, please open an issue in this repository and describe:

1. **Problem** — what fails, concisely, with the command, the configuration and the error message
2. **Root cause** — where in the code and why it happens
3. **Potential solution** — the change that would fix it
4. **Behaviour before and after the fix** — what the workflow does now, and what it will do afterwards

Fixes are recorded in [docs/update_notes.md](docs/update_notes.md) (section *Recording a fix*) with the same four parts; please refer to that entry from the issue.

---

## Contributing

When contributing to this repository:

1. Keep the modular design: one module per task, configured through Hydra
2. Add docstrings and type hints
3. Test new functionality end to end on a small example before submitting
4. Update the configuration files and their README sections together
5. Follow PEP 8
6. Document new features in the relevant document under `docs/`
7. Record fixes in [docs/update_notes.md](docs/update_notes.md)

## License

Released under the MIT License (see [LICENSE](LICENSE)). Please cite the relevant papers when using this code in publications.

## Acknowledgments

- The LANL LATTE development team for the tomography framework
- The contributors to the open-source scientific Python ecosystem
- The seismological community for data standards and best practices

---

## Support

For questions about:
- **LATTE**: the [original repository](https://github.com/lanl/latte_traveltime) or the [forked repository](https://github.com/Naveen-Raj-M/latte_traveltime)
- **This toolkit**: open an issue in this repository (see [Reporting Issues](#reporting-issues))
- **File formats**: [docs/file_formats.md](docs/file_formats.md)

---

*This toolkit is designed to work with the LATTE tomography framework and is maintained independently.*
