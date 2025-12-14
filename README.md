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

# Run analysis pipeline
python -m analysis.main

# Override configuration parameters
python -m analysis.main analysis.output_format=pdf
```

## Toolkit Components

### Preprocessing Module

Complete data preparation pipeline for tomographic inversion:

- **Coordinate Transformation** - Convert geodetic to Cartesian coordinates using exact ECEF or Earth-flattening methods
- **Velocity Model Creation** - Generate 3D initial models from 1D reference models
- **Geometry & Travel Time Processing** - Parse phase picks, validate data quality, and create LATTE-compatible files
- **Binary File Generation** - Convert velocity models to Fortran-compatible binary format
- **Visualization** - Generate quality control plots and animations

**Key Features:**
- Multiple coordinate transformation methods (exact, flattening, Chapman)
- Phase type selection (both P+S, P-only, S-only)
- Origin time modes (fixed, unix_epoch)
- Configurable binary precision (single/double)
- Depth filtering capabilities
- Automated velocity model generation
- Quality control filtering
- LATTE-compatible binary outputs

For detailed preprocessing documentation, see [Preprocessing Guide](docs/preprocess.md).

### Post-Processing Module

Converts LATTE tomography binary outputs to analysis-ready CSV format:

- **Velocity Model Conversion** - Extract updated Vp and Vs models from binary outputs
- **Synthetic Travel Time Processing** - Combine synthetic travel times with ground truth metadata
- **Source Parameter Updates** - Process updated source locations and origin times
- **Gradient & Search Direction Processing** - Optional extraction of optimization gradients and search directions
- **Coordinate Transformation Support** - Optional Chapman inverse transformation for spherical coordinates
- **Source-Receiver Exchange** - Support for reciprocity-based inversions

**Key Features:**
- Self-contained CSV outputs requiring no additional mapping files
- Phase type configuration matching preprocessing
- Configurable binary precision (single/double for all files)
- Interpolation modes (sample at coarse points or expand to fine grid)
- Automatic residual calculation (observed - synthetic)
- Support for source correction and location updates
- Gradient/search direction extraction for sensitivity analysis
- Comprehensive metadata preservation

For detailed post-processing documentation, see [Post-Processing Guide](docs/postprocess.md).

### Analysis Module

Comprehensive analysis and visualization of tomographic inversion results:

- **Gradient Analysis** - Analyze optimization gradients and search directions for velocity and source parameters
- **Travel Time Analysis** - Residual statistics, phase comparisons, and logarithmic ratio analysis
- **Velocity Model Comparison** - Spatial projections, profile analysis, and statistical comparisons
- **Distance-Dependent Analysis** - Correlation studies and binned statistics by source-receiver distance
- **Short Distance Analysis** - Focused analysis with spatial clustering and travel path visualization
- **Source Correction Analysis** - Hypocenter relocation evaluation and correction magnitude assessment

**Key Features:**
- Professional publication-quality visualizations (PNG, PDF, SVG)
- Method comparison modes (ground truth vs method-to-method)
- Distance type selection (3D hypocentral or 2D epicentral)
- Phase-specific analysis (P-wave, S-wave, or combined)
- Plane-based ratio plots at custom coordinates
- Comprehensive statistical reporting
- Modular analysis with flexible configuration
- Multi-scale spatial analysis
- Automated quality metrics

For detailed analysis documentation, see [Analysis Guide](docs/analysis.md).

## Complete Workflow

```
1. Preprocessing          2. LATTE Inversion       3. Post-Processing        4. Analysis
   ↓                         ↓                        ↓                         ↓
Input Data              Binary Outputs           Analysis-Ready CSVs       Results & Plots
├─ stacoords.txt        ├─ updated_vp.bin        ├─ velocity_cartesian     ├─ Residual plots
├─ phasePickings.txt    ├─ updated_vs.bin        ├─ synthetic_travel_times ├─ Velocity comparisons
├─ vmodel-PB1D.txt      ├─ updated_sx.bin        └─ [Optional corrections] ├─ Statistical reports
└─ vmodel-PB3D.txt      ├─ updated_sy.bin                                  └─ Quality metrics
                        ├─ updated_sz.bin
                        ├─ updated_st0.bin
                        └─ synthetic_traveltimes/
```

## Repository Structure

```
latte-workflow/
├── preprocess/                      # Preprocessing pipeline
│   ├── __init__.py
│   ├── main.py
│   ├── coordinate_transformation.py
│   ├── initial_velocity_model.py
│   ├── geometry_traveltime.py
│   ├── velocity_binary.py
│   └── visualization.py
│
├── postprocess/                     # Post-processing pipeline
│   ├── __init__.py
│   ├── main.py
│   ├── postprocess_velocity.py
│   ├── postprocess_traveltime.py
│   └── postprocess_source.py
│
├── analysis/                        # Analysis pipeline
│   ├── __init__.py
│   ├── main.py
│   ├── filters.py
│   ├── gradient_analysis.py         # Gradient & search direction analysis
│   ├── gradient_plots.py
│   ├── gradient_reports.py
│   ├── traveltime_analysis.py       # Travel time residual analysis
│   ├── traveltime_comparison.py
│   ├── traveltime_plots.py
│   ├── traveltime_statistics.py
│   ├── velocity_analysis.py         # Velocity model comparison
│   ├── velocity_plots.py
│   ├── velocity_reports.py
│   ├── velocity_spatial.py
│   ├── velocity_statistics.py
│   ├── distance_analysis.py         # Distance-dependent analysis
│   ├── distance_data.py
│   ├── distance_plots.py
│   ├── distance_reports.py
│   ├── distance_statistics.py
│   ├── short_distance_analysis.py   # Short-distance focused analysis
│   ├── short_distance_data.py
│   ├── short_distance_plots.py
│   ├── short_distance_reports.py
│   ├── short_distance_statistics.py
│   ├── source_analysis.py           # Source correction analysis
│   └── source_plots.py
│
├── utils/                           # Shared utilities
│   ├── __init__.py
│   ├── io_utils.py
│   ├── logging_utils.py
│   ├── plot_utils.py
│   ├── distance_utils.py
│   └── origin_time_utils.py
│
├── configs/                         # Configuration files
│   ├── preprocess/
│   │   ├── default.yaml
│   │   └── README.md
│   ├── postprocess/
│   │   ├── default.yaml
│   │   └── README.md
│   └── analysis/
│       ├── default.yaml
│       ├── config.yaml
│       ├── high_quality.yaml
│       ├── traveltime_only.yaml
│       ├── velocity_only.yaml
│       └── README.md
│
└── docs/                            # Documentation
    ├── preprocess.md
    ├── postprocess.md
    └── analysis.md
```

## Analysis Quick Examples

### Standard Workflow

```bash
# 1. Configure paths in configs/analysis/default.yaml
# 2. Run complete analysis
python -m analysis.main

# 3. Check outputs
ls results/analysis/
```

### Run Single Analysis Module

```bash
# Gradient analysis only
python -m analysis.main single_function=gradient_analysis

# Travel time analysis only
python -m analysis.main single_function=travel_time_analysis

# Velocity comparison only
python -m analysis.main single_function=velocity_analysis

# Distance analysis only
python -m analysis.main single_function=distance_analysis
```

### Customize Output Quality

```bash
# High-resolution PDF output
python -m analysis.main analysis.output_format=pdf analysis.dpi=600

# Publication quality with custom styling
python -m analysis.main \
  analysis.output_format=pdf \
  analysis.dpi=600 \
  analysis.plot_style.font_size=16
```

### Enable/Disable Specific Features

```bash
# Enable all velocity analysis features
python -m analysis.main \
  velocity_analysis.enable_spatial_projections=true \
  velocity_analysis.compute_geometric_mean_ratios=true

# Method comparison mode
python -m analysis.main \
  travel_time_analysis.comparison_mode=method_comparison \
  travel_time_analysis.method1_csv=/path/to/method1.csv \
  travel_time_analysis.method2_csv=/path/to/method2.csv

# Phase-specific analysis
python -m analysis.main analysis.phase_type=p

# Disable log-scale plots
python -m analysis.main \
  travel_time_analysis.enable_log_log_plots=false
```

## Output Files

### Preprocessing Outputs
Binary files and CSV datasets ready for LATTE inversion with complete coordinate mappings and quality control plots.

### Post-Processing Outputs
Analysis-ready CSV files containing updated velocity models, synthetic travel times with residuals, and optional source corrections.

### Analysis Outputs
- **Plots**: Publication-quality visualizations (PNG, PDF, SVG)
- **Reports**: Comprehensive statistical summaries in text format
- **Statistics**: Detailed metrics for quality assessment

## Documentation

- **[Preprocessing Guide](docs/preprocess.md)** - Detailed preprocessing documentation
- **[Post-Processing Guide](docs/postprocess.md)** - Detailed post-processing documentation
- **[Analysis Guide](docs/analysis.md)** - Detailed analysis documentation
- **Configuration Examples** - See `configs/` directory
- **API Reference** - See docstrings in source files

## Troubleshooting

**Module Import Errors:**
Run from the parent directory: `python -m analysis.main`

**Missing Dependencies:**
Install required packages: `pip install -r requirements.txt`

**Configuration Not Found:**
Verify file is at `configs/analysis/default.yaml`

**Memory Issues:**
Reduce plot resolution: `analysis.plot_style.max_plot_points=20000`

For detailed troubleshooting, see the respective module documentation.

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