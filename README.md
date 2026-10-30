# LATTE Tomography Toolkit

A comprehensive Python toolkit for seismic tomography data processing, analysis, and visualization using the [LATTE (Los Alamos Travel Time Estimation)](https://academic.oup.com/gji/article/241/2/1275/8046728) framework.

## Overview

This toolkit provides end-to-end capabilities for regional and local-scale seismic tomography studies, from raw data preprocessing through tomographic inversion to results analysis and visualization.

> **Note:** This workflow is designed to work with the [`dev2` branch](https://github.com/Naveen-Raj-M/latte_traveltime/tree/dev2) of the forked LATTE repository at [https://github.com/Naveen-Raj-M/latte_traveltime.git](https://github.com/Naveen-Raj-M/latte_traveltime.git), which contains important functionality required by this workflow.

## Related Publications

- **LATTE Framework**: [LATTE: open-source, high-performance traveltime computation, tomography and source location in acoustic and elastic media](https://academic.oup.com/gji/article/241/2/1275/8046728) — Geophysical Journal International, 2025
- **Original LATTE Code**: [LANL LATTE Repository](https://github.com/lanl/latte_traveltime)
- **Data Source**: [Mapping the 3-D Lithospheric Structure of the Greater Permian Basin in West Texas and Southeast New Mexico for Earthquake Monitoring](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JB018351) — Huang et al., Journal of Geophysical Research: Solid Earth, 2019

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd latte-workflow

# Install dependencies
pip install -r requirements.txt
```

**Required Dependencies:**
- Python 3.8+
- NumPy, Pandas, SciPy
- Matplotlib, imageio
- Hydra, OmegaConf
- ObsPy (for QuakeML input)
- pyproj (optional, for exact coordinate transformations)

### Basic Usage

All pipelines are driven by [Hydra](https://hydra.cc) configuration. Copy the relevant `example.yaml` from `configs/<pipeline>/`, update the file paths, and run:

```bash
# Run preprocessing pipeline
python -m preprocess.main --config-name=example

# Run post-processing pipeline
python -m postprocess.main --config-name=example

# Run analysis pipeline
python -m analysis.main --config-name=example
```

Individual parameters can be overridden directly on the command line without modifying the config file:

```bash
# Override a single parameter
python -m analysis.main --config-name=example travel_time_analysis.enabled=true

# Override multiple parameters
python -m postprocess.main --config-name=example \
  velocity_processing.enabled=true \
  travel_time_processing.enabled=true
```

> For full details on available parameters, override syntax, and example configs, see the configuration documentation in [`configs/preprocess/README.md`](configs/preprocess/README.md), [`configs/postprocess/README.md`](configs/postprocess/README.md), and [`configs/analysis/README.md`](configs/analysis/README.md).

---

## Workflow Components

### Preprocessing

The preprocessing pipeline transforms raw seismic data into binary and tabular inputs required by the LATTE inversion kernel. It accepts event catalogs in **QuakeML** or **HDF5** format and produces geometry files, travel time tables, and initial 3D velocity model binaries.

**Pipeline stages (executed in order):**
1. **Data Loading** — reads events, stations, picks, and arrivals; applies magnitude, geographic, depth, and distance filters
2. **Coordinate Transformation** — converts geodetic (lon/lat/depth) to local Cartesian coordinates (x/y/z km) via a first-order flattening transformation
3. **Geometry & Travel Time Generation** — constructs source–receiver geometry files and ground truth travel time tables in LATTE-compatible format
4. **Initial Velocity Model** — generates a 3D initial Vp/Vs model from a 1D reference profile and writes it as Fortran-compatible binary
5. **Visualization** — produces quality control plots and animations

For detailed documentation, see [docs/preprocess.md](docs/preprocess.md).

---

### Post-Processing

The post-processing pipeline converts LATTE binary inversion outputs into analysis-ready CSV and HDF5 files. All five stages are independent and disabled by default; enable only those relevant to a given run.

**Pipeline stages:**
1. **Velocity Processing** — reads updated Vp/Vs binary files and projects them onto the reference coordinate grid; optionally extracts gradient and search direction binaries
2. **Travel Time Processing** — merges observed ground truth travel times with synthetic predictions from the inversion kernel, computing per-pick residuals
3. **Source Parameter Processing** — extracts updated hypocenter locations and origin times from binary outputs
4. **Ray Path Diagnosis** — builds an HDF5 ray path cache and computes Derivative Weight Sum (DWS) sensitivity grids for model coverage assessment
5. **Ray Path Simplification** — reduces ray path memory footprint using the Ramer–Douglas–Peucker algorithm

For detailed documentation, see [docs/postprocess.md](docs/postprocess.md).

---

### Analysis

The analysis pipeline provides post-inversion diagnostics for LATTE tomography results. It reads post-processed CSV, HDF5, and ray path cache files to produce publication-quality plots and statistical reports. All ten modules are independent and disabled by default.

**Analysis modules:**
1. **Travel Time Residual Analysis** — residual distributions, histograms, scatter plots, and summary statistics; supports ground truth and method-to-method comparison modes
2. **Velocity Model Analysis** — point-wise differences and log-ratios between two models; spatial projections, depth slices, and 1D profile plots
3. **Distance-Dependent Residual Analysis** — residual dependence on source–receiver distance; identifies systematic distance-dependent bias
4. **Short-Distance Analysis** — residual analysis restricted to near-field source–receiver pairs with spatial clustering
5. **Source Correction Analysis** — hypocenter relocation evaluation and spatial distribution of correction magnitudes
6. **Gradient Analysis** — velocity and source gradient / search direction diagnostics from inversion optimization
7. **DWS Analysis** — Derivative Weight Sum model coverage and resolution assessment
8. **Ray Path Analysis** — 3D ray path projection visualization
9. **Ray Density Analysis** — ray count per grid cell and coverage mapping
10. **Tradeoff Analysis** — data misfit vs. model variance L-curve for regularization selection

For detailed documentation, see [docs/analysis.md](docs/analysis.md).

---

## Complete Workflow

```
1. Preprocessing            2. LATTE Inversion          3. Post-Processing           4. Analysis
   ↓                           ↓                            ↓                            ↓
Input Data                 Binary Outputs               Analysis-Ready Files         Results & Plots
├─ Event catalog           ├─ updated_vp.bin            ├─ velocity_model_final.csv   ├─ Residual plots
│  (QuakeML / HDF5)        ├─ updated_vs.bin            ├─ synthetic_traveltimes.csv  ├─ Velocity comparisons
├─ Station inventory       ├─ updated_sx/sy/sz.bin      ├─ source_relocation.csv     ├─ DWS coverage maps
└─ 1D reference model      ├─ updated_st0.bin           ├─ ray_paths.h5              ├─ Ray path projections
                           ├─ synthetic_traveltimes/    └─ dws_grid.csv              ├─ L-curve tradeoff
                           └─ ray_path binaries                                      └─ Statistical reports
```

---

## Repository Structure

```
latte-workflow/
├── preprocess/                      # Preprocessing pipeline
│   ├── main.py
│   ├── data_loader.py
│   ├── coordinate_transformation.py
│   ├── initial_velocity_model.py
│   ├── geometry_traveltime.py
│   ├── visualization.py
│   └── ray_density_visualization.py
│
├── postprocess/                     # Post-processing pipeline
│   ├── main.py
│   ├── postprocess_velocity.py
│   ├── postprocess_traveltime.py
│   ├── postprocess_source.py
│   ├── postprocess_raypath.py
│   ├── postprocess_dws.py
│   ├── compute_ray_paths.py
│   ├── read_raypath_binary.py
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
│   └── [supporting plot/data/report modules]
│
├── utils/                           # Shared utilities
│   ├── io_utils.py
│   ├── logging_utils.py
│   ├── plot_utils.py
│   ├── distance_utils.py
│   ├── origin_time_utils.py
│   └── velocity_io.py
│
├── configs/                         # Hydra configuration files
│   ├── preprocess/
│   │   ├── default.yaml             # Default configuration
│   │   ├── example.yaml             # Minimal template to copy
│   │   └── README.md                # Configuration reference
│   ├── postprocess/
│   │   ├── default.yaml
│   │   ├── example.yaml
│   │   └── README.md
│   └── analysis/
│       ├── default.yaml
│       ├── example.yaml
│       ├── high_quality.yaml
│       ├── traveltime_only.yaml
│       ├── velocity_only.yaml
│       └── README.md
│
├── docs/                            # Detailed documentation
│   ├── preprocess.md
│   ├── postprocess.md
│   ├── analysis.md
│   └── file_formats.md
│
└── tests/                           # Unit tests
```

---

## Documentation

| Document | Description |
|---|---|
| [docs/preprocess.md](docs/preprocess.md) | Module-level reference for each preprocessing stage, configuration keys, and outputs |
| [docs/postprocess.md](docs/postprocess.md) | Module-level reference for each post-processing stage, configuration keys, and outputs |
| [docs/analysis.md](docs/analysis.md) | Module-level reference for each analysis module, configuration keys, and plot descriptions |
| [docs/file_formats.md](docs/file_formats.md) | Complete specification of all input and output file formats across all three pipelines |
| [configs/preprocess/README.md](configs/preprocess/README.md) | All preprocessing configuration parameters with defaults and descriptions |
| [configs/postprocess/README.md](configs/postprocess/README.md) | All post-processing configuration parameters with defaults and descriptions |
| [configs/analysis/README.md](configs/analysis/README.md) | All analysis configuration parameters with defaults and descriptions |

---

## Compatibility

- **Python**: 3.8+
- **LATTE**: Requires the [`dev2` branch](https://github.com/Naveen-Raj-M/latte_traveltime/tree/dev2) of [https://github.com/Naveen-Raj-M/latte_traveltime.git](https://github.com/Naveen-Raj-M/latte_traveltime.git)
- **Fortran Binary Format**: Binary outputs use Fortran stream access format (float32 or float64)
- **Operating Systems**: Linux, macOS, Windows
- **Input Data Formats**: QuakeML, HDF5, plain-text station files

---

## Contributing

When contributing to this repository:

1. Maintain modular design principles across all components
2. Add comprehensive docstrings and type hints
3. Include unit tests for new functionality
4. Update configuration schemas as needed
5. Follow PEP 8 style guidelines
6. Document new features in relevant README sections
7. Consider cross-module compatibility

## License

This project is intended for academic and research use. Please cite the relevant papers when using this code in publications.

## Acknowledgments

- LANL LATTE development team for the original tomography framework
- Huang et al. for the Permian Basin dataset and methodological insights
- Contributors to the open-source scientific Python ecosystem
- Seismological community for data standards and best practices

---

## Support

For questions about:
- **LATTE Framework**: Refer to the [forked LATTE repository (dev2 branch)](https://github.com/Naveen-Raj-M/latte_traveltime/tree/dev2) or the [original LATTE repository](https://github.com/lanl/latte_traveltime)
- **This Toolkit**: Open an issue in this repository
- **Data Formats**: See [docs/file_formats.md](docs/file_formats.md) for a complete specification of all input and output formats
- **Scientific Applications**: Review the cited literature

---

*This toolkit is designed to work with the LATTE tomographic framework but is maintained independently. It aims to provide a complete workflow solution for seismic tomography research and applications.*
