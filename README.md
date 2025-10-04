# LATTE Tomography Toolkit

A comprehensive Python toolkit for seismic tomography data processing, analysis, and visualization using the [LATTE (Los Alamos Travel Time Estimation)](https://academic.oup.com/gji/article/241/2/1275/8046728) framework.

## Overview

This toolkit provides end-to-end capabilities for regional and local-scale seismic tomography studies, from raw data preprocessing through tomographic inversion to results analysis and visualization.

## Related Publications

- **LATTE Framework**: [LATTE: open-source, high-performance traveltime computation, tomography and source location in acoustic and elastic media](https://academic.oup.com/gji/article/241/2/1275/8046728) - Geophysical Journal International, 2025
- **Original LATTE Code**: [LANL LATTE Repository](https://github.com/lanl/latte_traveltime)
- **Data Source**: [Mapping the 3-D Lithospheric Structure of the Greater Permian Basin in West Texas and Southeast New Mexico for Earthquake Monitoring](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JB018351) - Huang et al., Journal of Geophysical Research: Solid Earth, 2019

## Quick Start

### Installation

```bash
# Clone the repository
git clone 
cd latte_toolkit

# Install dependencies
pip install -r requirements.txt
```

**Required Dependencies:**
- Python 3.8+
- NumPy, Pandas, SciPy
- Matplotlib, imageio
- Hydra, OmegaConf
- pyproj (optional, for exact transformations)

### Basic Usage

```bash
# Run preprocessing pipeline
python -m preprocess.main

# Run post-processing pipeline
python -m postprocess.main

# Override configuration parameters
python -m postprocess.main velocity_processing.enabled=false
```

## Toolkit Components

### Preprocessing Module

Complete data preparation pipeline for tomographic inversion:

- **Coordinate Transformation** - Convert geodetic to Cartesian coordinates using exact ECEF or Earth-flattening methods
- **Velocity Model Creation** - Generate 3D initial models from 1D reference models
- **Geometry & Travel Time Processing** - Parse phase picks, validate data quality, and create LATTE-compatible files
- **Binary File Generation** - Convert velocity models to Fortran-compatible binary format
- **Visualization** - Generate quality control plots and animations

For detailed preprocessing documentation, see [Preprocessing Guide](docs/preprocess.md).

### Post-Processing Module

Converts LATTE tomography binary outputs to analysis-ready CSV format:

- **Velocity Model Conversion** - Extract updated Vp and Vs models from binary outputs
- **Synthetic Travel Time Processing** - Combine synthetic travel times with ground truth metadata
- **Source Parameter Updates** - Process updated source locations and origin times
- **Coordinate Transformation Support** - Optional Chapman inverse transformation for spherical coordinates
- **Source-Receiver Exchange** - Support for reciprocity-based inversions

**Key Features:**
- Self-contained CSV outputs requiring no additional mapping files
- Automatic residual calculation (observed - synthetic)
- Support for source correction and location updates
- Comprehensive metadata preservation
- Hydra-based configuration with command-line overrides

For detailed post-processing documentation, see [Post-Processing Guide](docs/postprocess.md).

## Configuration System

Uses [Hydra](https://hydra.cc/) for flexible YAML-based configuration:

```yaml
# Post-processing example configuration
velocity_processing:
  enabled: true
  original_velocity_csv: "/path/to/velocity_cartesian_final.csv"
  vp_binary: "${data.base_dir}/model/updated_vp.bin"
  vs_binary: "${data.base_dir}/model/updated_vs.bin"
  velocity_output_csv: "${data.analysis_output_dir}/velocity_cartesian_latte_output.csv"
  use_chapman_inverse: false

travel_time_processing:
  enabled: true
  ground_truth_csv: "/path/to/ground_truth_travel_times_corrected.csv"
  synthetic_tt_dir: "${data.base_dir}/synthetic_traveltimes"
  synthetic_tt_output_csv: "${data.analysis_output_dir}/synthetic_travel_times_latte_output.csv"
  source_receiver_exchange: false
  source_correction: false
```

## Complete Workflow

```
1. Preprocessing          2. LATTE Inversion       3. Post-Processing
   ↓                         ↓                        ↓
Input Data              Binary Outputs           Analysis-Ready CSVs
├─ stacoords.txt        ├─ updated_vp.bin        ├─ velocity_cartesian_latte_output.csv
├─ phasePickings.txt    ├─ updated_vs.bin        ├─ synthetic_travel_times_latte_output.csv
├─ vmodel-PB1D.txt      ├─ updated_sx.bin        └─ [Optional source corrections]
└─ vmodel-PB3D.txt      ├─ updated_sy.bin
                        ├─ updated_sz.bin
                        ├─ updated_st0.bin
                        └─ synthetic_traveltimes/
```

## Repository Structure

```
latte_toolkit/
├── preprocess/              # Preprocessing pipeline
│   ├── main.py
│   ├── coordinate_transformation.py
│   ├── initial_velocity_model.py
│   ├── geometry_traveltime.py
│   ├── velocity_binary.py
│   └── visualization.py
├── postprocess/             # Post-processing pipeline
│   ├── main.py
│   ├── postprocess_velocity.py
│   ├── postprocess_traveltime.py
│   └── postprocess_source.py
├── utils/                   # Shared utilities
│   ├── io_utils.py
│   └── logging_utils.py
├── configs/                 # Configuration files
│   ├── preprocess/
│   │   └── default.yaml
│   └── postprocess/
│       └── default.yaml
└── docs/                    # Documentation
    ├── preprocess.md
    └── postprocess.md
```

## Post-Processing Quick Examples

### Standard Workflow

```bash
# 1. Configure paths in configs/postprocess/default.yaml
# 2. Run post-processing
python -m postprocess.main

# 3. Check outputs
ls outputs/analysis/
```

### Enable Source Correction

```bash
python -m postprocess.main \
  travel_time_processing.source_correction=true
```

### Use Source-Receiver Exchange

```bash
python -m postprocess.main \
  travel_time_processing.source_receiver_exchange=true
```

### Enable Chapman Transformation

```bash
python -m postprocess.main \
  velocity_processing.use_chapman_inverse=true
```

### Process Only Velocity Data

```bash
python -m postprocess.main \
  travel_time_processing.enabled=false \
  source_processing.enabled=false
```

## Output Files

### Velocity CSV
Contains updated velocity models with preserved coordinate systems and grid structure. Includes Vp, Vs values in km/s with optional Chapman inverse transformation.

### Travel Time CSV
Comprehensive dataset combining observed and synthetic travel times with complete metadata including event/station coordinates, residuals, and weights.

### Source Updates CSV (Optional)
Enhanced travel time CSV with corrected source locations (x, y, z) and origin times when source parameter inversion is enabled.

## Troubleshooting

**Module Import Errors:**
Run from the parent directory: `python -m preprocess.main`

**Missing pyproj:**
Install for exact transformations: `pip install pyproj`

**Configuration Not Found:**
Verify file is at `configs/preprocess/default.yaml`

**Grid Dimension Warnings:**
The toolkit auto-detects dimensions from your data

For detailed troubleshooting, see the [preprocessing guide](docs/preprocess.md).


## Compatibility

- **Python**: 3.8+
- **LATTE**: Compatible with LANL LATTE traveltime tomography framework
- **Fortran Codes**: Binary outputs use Fortran stream access format
- **Operating Systems**: Cross-platform (Linux, macOS, Windows)
- **Data Formats**: NetCDF, VTK, CSV, binary formats

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

## Support

For questions about:
- **LATTE Framework**: Refer to the [original LATTE repository](https://github.com/lanl/latte_traveltime)
- **This Toolkit**: Open an issue in this repository
- **Data Format Questions**: Consult the original data publications
- **Scientific Applications**: Review the cited literature

---

*This toolkit is designed to work with the LATTE tomographic framework but is maintained independently. It aims to provide a complete workflow solution for seismic tomography research and applications.*