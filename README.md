# LATTE Tomography Toolkit

A comprehensive Python toolkit for seismic tomography data processing, analysis, and visualization using the [LATTE (Los Alamos Travel Time Estimation)](https://academic.oup.com/gji/article/241/2/1275/8046728) methodology.

## Overview

This repository contains a complete toolkit for seismic tomography workflows, from raw data preprocessing through tomographic inversion to results analysis and visualization. The toolkit is designed to work with the LATTE framework and provides end-to-end capabilities for regional and local-scale seismic tomography studies.

## Related Publications

- **LATTE Framework**: [Travel time tomography with adaptive dictionaries](https://academic.oup.com/gji/article/241/2/1275/8046728) - Geophysical Journal International, 2025
- **Original LATTE Code**: [LANL LATTE Repository](https://github.com/lanl/latte_traveltime)
- **Data Source**: [Mapping the 3-D Lithospheric Structure of the Greater Permian Basin in West Texas and Southeast New Mexico for Earthquake Monitoring](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JB018351) - Huang et al., Journal of Geophysical Research: Solid Earth, 2019

## Toolkit Components

### 1. Preprocessing Module (`preprocessing/`)

Comprehensive data preparation pipeline for tomographic inversion:

- **Coordinate Transformation** (`coordinate_transformation.py`)
  - Geodetic to Cartesian coordinate conversion
  - Support for exact transformations (using pyproj/ECEF) and Earth-flattening methods
  - Chapman (1973) flattening transformation for seismic studies
  - Grid mapping with configurable spacing

- **Initial Velocity Model Creation** (`initial_velocity_model.py`)
  - Convert 1D velocity models to 3D initial models
  - Depth-based velocity assignment with extrapolation
  - Validation and quality checking

- **Geometry and Travel Time Processing** (`geometry_traveltime.py`)
  - Phase pick parsing and filtering
  - Travel time calculation and validation
  - Geometry file generation for LATTE
  - Binary travel time file creation
  - Support for both original and corrected phase data

- **Velocity Binary File Creation** (`velocity_binary.py`)
  - CSV to binary conversion for velocity models
  - Fortran-compatible binary format
  - Gap interpolation and grid completion
  - Support for Chapman-transformed velocities

- **Visualization** (`visualization.py`)
  - 3D station and event plotting
  - Velocity model visualization
  - Animated model rotation
  - Comparison plots for quality assurance

### 2. Postprocessing Module (`postprocessing/`) *[Planned]*

Interpretation of tomographic inversion results:

### 3. Analysis Module (`analysis/`) *[Planned]*

Advanced analysis tools for seismic tomography research:

### Utility Modules (`utils/`)

- **I/O Utilities** (`io_utils.py`): File operations, CSV handling, data validation
- **Logging Utilities** (`logging_utils.py`): Process timing, progress tracking, debug logging
- **Plotting Utilities** (`plot_utils.py`): *[Planned]* Standardized visualization functions

### Repository Structure

```
latte_toolkit/
├── preprocessing/
│   ├── __init__.py
│   ├── main.py                      # Main pipeline orchestration
│   ├── coordinate_transformation.py  # Coordinate conversion
│   ├── initial_velocity_model.py    # 1D to 3D model creation
│   ├── geometry_traveltime.py       # Phase processing
│   ├── velocity_binary.py           # Binary file creation
│   ├── visualization.py             # Plotting and visualization
│   └── utils/
│       ├── __init__.py
│       ├── io_utils.py              # I/O utilities
│       └── logging_utils.py         # Logging utilities
├── postprocessing/                  # [Planned] Results analysis
├── analysis/                        # [Planned] Advanced analysis
├── utils/                           # Shared utilities
│   ├── __init__.py
│   ├── io_utils.py
│   ├── logging_utils.py
│   ├── plot_utils.py               # [Planned]
├── configs/                        # Hydra configuration files
├── examples/                       # Example workflows and tutorials
├── tests/                          # Unit and integration tests
└── README.md
```

## Usage

### Configuration-Based Processing

The toolkit uses [Hydra](https://hydra.cc/) for configuration management. Create configuration files in YAML format to specify input files, processing parameters, and output locations.

### Running the Preprocessing Pipeline

```bash
# Run complete preprocessing pipeline
python -m latte_toolkit.preprocessing.main

# Run with custom configuration
python -m latte_toolkit.preprocessing.main --config-path=/path/to/configs --config-name=my_config

# Run individual preprocessing modules
python -m latte_toolkit.preprocessing.main single_function=coordinate_transform
```

### Future Usage (Postprocessing and Analysis)

```bash
# Planned postprocessing capabilities
python -m latte_toolkit.postprocessing.main --config-name=postprocess_config

# Planned analysis workflows
python -m latte_toolkit.analysis.main --config-name=analysis_config
```

## Data Requirements and Formats

### Input Data (Preprocessing)

1. **Station Coordinates**: 6-column format (stacoords file)
   ```
   [unused] [unused] [unused] elevation(m) station_id latitude longitude
   ```

2. **Event Data**: 
   - Original: 9-column phasePickings.txt format
   - Corrected: 10-column .dat format with weighted RMS

3. **Phase Picks**: 
   - P and S wave arrival times
   - Station and event associations
   - Quality weights

4. **Velocity Models**:
   - 1D reference models (depth, Vp, Vs)
   - 3D template grids

### Output Data (LATTE-Compatible)

- **Cartesian Coordinates**: CSV files with transformed coordinates and grid indices
- **Geometry Files**: Text files describing source-receiver geometry
- **Travel Time Binaries**: Binary files with P and S wave travel times
- **Velocity Binaries**: Fortran-compatible binary velocity models
- **Visualizations**: 3D plots and animations for quality control

## Coordinate Transformation Methods

### Exact Transformation
Uses precise geodetic-to-ECEF-to-local conversions with pyproj:
- ECEF (Earth-Centered Earth-Fixed) coordinate system
- ENU (East-North-Up) local coordinates
- Optional geocentric normal alignment

### Earth Flattening Transformation
Implements spherical Earth approximations:
- **Simple Geometric**: Basic latitude/longitude to Cartesian
- **Chapman (1973)**: Earth-flattening transformation with velocity corrections

## Quality Control Features

- Velocity range validation for P and S waves
- Travel time sanity checks
- Coordinate boundary validation
- Grid completeness verification
- Comprehensive visualization tools for data inspection
- Automated quality reports

## Configuration Example

```yaml
preprocessing:
  coordinate_transform:
    enabled: true
    transformation_method: "exact"  # or "flattening"
    origin_lon: -118.0
    origin_lat: 34.0
    origin_depth_km: 0.0
    grid_spacing_km: [1.0, 1.0, 1.0]
    
  geometry_traveltime:
    enabled: true
    use_corrected_phase_pickings: false
    p_velocity_range_m_s: [4000, 9000]
    s_velocity_range_m_s: [2000, 5500]
    
  visualization:
    enabled: true
    view_elevation: 30
    view_from_direction: "southeast"

# Planned configurations
postprocessing:

analysis:
```

## Complete Workflow

### 1. Data Preparation (Preprocessing)
- Load and validate input data
- Transform coordinates to local Cartesian system
- Create initial velocity models
- Process phase picks and generate geometry files
- Create binary files for LATTE

### 2. Tomographic Inversion (External - LATTE)
- Run LATTE tomographic inversion
- Generate velocity models and resolution matrices

### 3. Results Analysis (Postprocessing)

### 4. Scientific Interpretation (Analysis)

## Development Roadmap

### Current Status
- ✅ Complete preprocessing pipeline
- ✅ Coordinate transformation modules
- ✅ Basic visualization capabilities
- ✅ Configuration management system

### Phase 1 (Next)
- 🔄 Postprocessing module framework
- 🔄 Model validation tools
- 🔄 Enhanced visualization capabilities
- 🔄 Format conversion utilities

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