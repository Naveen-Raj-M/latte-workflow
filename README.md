# LATTE Tomography Toolkit

A comprehensive Python toolkit for seismic tomography data processing, analysis, and visualization using the [LATTE (Los Alamos Travel Time Estimation)](https://academic.oup.com/gji/article/241/2/1275/8046728) framework.

## Overview

This toolkit provides end-to-end capabilities for regional and local-scale seismic tomography studies, from raw data preprocessing through tomographic inversion to results analysis and visualization.

## Related Publications

- **LATTE Framework**: [LATTE: open-source, high-performance traveltime computation, tomography and source location in acoustic and elastic media](https://academic.oup.com/gji/article/241/2/1275/8046728) - Geophysical Journal International, 2025
- **Original LATTE Code**: [LANL LATTE Repository](https://github.com/lanl/latte_traveltime)
- **Data Source**: [Mapping the 3-D Lithospheric Structure of the Greater Permian Basin in West Texas and Southeast New Mexico for Earthquake Monitoring](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JB018351) - Huang et al., Journal of Geophysical Research: Solid Earth, 2019

**Key Features:**
- Multiple coordinate transformation methods (exact ECEF and Earth-flattening)
- Automated velocity model generation from 1D to 3D
- Phase pick processing with quality control filtering
- LATTE-compatible binary file generation
- Comprehensive visualization tools

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
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
# Run complete preprocessing pipeline
python -m preprocess.main

# Run with custom configuration
python -m preprocess.main --config-path=/path/to/configs --config-name=my_config

# Run individual modules
python -m preprocess.main single_function=coordinate_transform
```

## Toolkit Components

### Preprocessing Module

Complete data preparation pipeline for tomographic inversion:

- **Coordinate Transformation** - Convert geodetic to Cartesian coordinates using exact ECEF or Earth-flattening methods
- **Velocity Model Creation** - Generate 3D initial models from 1D reference models
- **Geometry & Travel Time Processing** - Parse phase picks, validate data quality, and create LATTE-compatible files
- **Binary File Generation** - Convert velocity models to Fortran-compatible binary format
- **Visualization** - Generate quality control plots and animations

### Configuration System

Uses [Hydra](https://hydra.cc/) for flexible YAML-based configuration:

```yaml
coordinate_transform:
  transformation_method: "flattening"  # or "exact"
  origin_lon: -108.0
  origin_lat: 34.5
  grid_spacing_km: [20.0, 20.0, 2.0]

geometry_traveltime:
  p_velocity_range_m_s: [500, 10000]
  s_velocity_range_m_s: [500, 6000]
```

Override any parameter from the command line:

```bash
python -m preprocess.main coordinate_transform.transformation_method=exact
```

## Data Flow

```
Input Data                 Preprocessing              Output (LATTE-ready)
├─ stacoords.txt          ├─ Coordinate Transform   ├─ Cartesian CSVs
├─ phasePickings.txt      ├─ Velocity Model         ├─ Geometry files
├─ vmodel-PB1D.txt        ├─ Phase Processing       ├─ Travel time binaries
└─ vmodel-PB3D.txt        └─ Binary Conversion      └─ Velocity binaries
```

## Key Capabilities

### Coordinate Transformation

**Exact Transformation:**
- Precise ECEF (Earth-Centered Earth-Fixed) to local ENU conversion
- Accounts for Earth's curvature
- Requires `pyproj`

**Flattening Transformation:**
- Simple geometric projection for regional studies
- Optional Chapman (1973) transformation for seismic wave corrections
- Fast and lightweight

### Quality Control

- Velocity range validation for P and S waves
- Travel time consistency checks
- Automatic filtering of unrealistic picks
- Station-event pair validation
- Comprehensive visualization for inspection

### LATTE Compatibility

Generates all files required by LATTE:
- Source and receiver geometry files
- P and S wave travel time binaries
- Initial velocity model binaries
- Source location files (sx, sy, sz, st0)

## File Formats

### Input Requirements

- **Stations:** 6-column format with elevation, latitude, longitude
- **Events:** 9-column phasePickings.txt or 10-column corrected .dat format
- **Velocity Models:** 1D (depth, Vp, Vs) and 3D (lon, lat, depth, Vp, Vs)

### Output Files

- **CSV:** Transformed coordinates with grid indices
- **Binary:** Fortran-compatible 32-bit float, column-major order
- **Geometry:** Text files describing source-receiver pairs
- **Visualizations:** PDF plots and GIF animations

## Documentation

- **[Preprocessing Guide](docs/preprocess.md)** - Detailed preprocessing documentation
- **Configuration Examples** - See `configs/preprocess/` directory
- **API Reference** - See docstrings in source files

## Repository Structure

```
latte_toolkit/
├── preprocess/              # Preprocessing pipeline
│   ├── main.py             # Pipeline orchestration
│   ├── coordinate_transformation.py
│   ├── initial_velocity_model.py
│   ├── geometry_traveltime.py
│   ├── velocity_binary.py
│   └── visualization.py
├── utils/                   # Shared utilities
│   ├── io_utils.py
│   └── logging_utils.py
├── configs/                 # Configuration files
│   └── preprocess/
│       └── default.yaml
└── docs/                    # Documentation
    └── preprocess.md
```

## Related Resources

- **LATTE Framework:** [Geophysical Journal International (2025)](https://academic.oup.com/gji/article/241/2/1275/8046728)
- **LATTE Code:** [LANL Repository](https://github.com/lanl/latte_traveltime)
- **Reference Dataset:** [Huang et al., JGR (2019)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JB018351)

## Examples

### Standard Workflow

```bash
# 1. Configure your data paths
cp configs/preprocess/default.yaml my_study.yaml
# Edit my_study.yaml with your file paths

# 2. Run preprocessing
python -m preprocess.main --config-name=my_study

# 3. Check outputs
ls outputs/
```

### Chapman Workflow

For studies requiring Earth-flattening corrections:

```bash
python -m preprocess.main \
  coordinate_transform.use_chapman_flattening=true \
  velocity_binary.use_chapman_velocities=true
```

### Using Corrected Phase Picks

```bash
python -m preprocess.main \
  coordinate_transform.use_corrected_phase_pickings=true \
  geometry_traveltime.use_corrected_phase_pickings=true
```

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