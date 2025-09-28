# LATTE Post-Processing Documentation

## Overview

The LATTE post-processing module converts binary outputs from LATTE tomography inversions into CSV format for downstream analysis. This ensures compatibility with existing analysis workflows while maintaining data consistency with preprocessing outputs.

## Features

- **Velocity Model Conversion**: Converts `updated_vp.bin` and `updated_vs.bin` to structured CSV format
- **Synthetic Travel Time Processing**: Combines synthetic travel time binaries with ground truth metadata
- **Coordinate Transformation Support**: Optional Chapman inverse transformation for velocity coordinates
- **Self-Contained Outputs**: Creates standalone CSV files requiring no additional mapping files
- **Hydra Configuration**: Integrated configuration management with logging support

## Module Structure

```
latte_postprocess/
├── __init__.py                    # Package initialization and exports
├── postprocess_functions.py      # Core processing functions
├── main.py                       # Main execution script with Hydra integration
└── docs/
    └── postprocess.md            # This documentation
```

## Configuration

### File Location
```
configs/postprocess/default.yaml
```

### Configuration Structure

```yaml
# General settings
general:
  strict_mode: true        # Stop on first error vs. continue processing
  log_level: INFO         # DEBUG, INFO, WARNING, ERROR

# Data paths
data:
  base_dir: "/path/to/latte/output"
  analysis_output_dir: "/path/to/analysis/output"

# Velocity processing
velocity_processing:
  enabled: true
  original_velocity_csv: "/full/path/to/velocity_cartesian_final.csv"
  vp_binary: "${data.base_dir}/model/updated_vp.bin"
  vs_binary: "${data.base_dir}/model/updated_vs.bin"
  velocity_output_csv: "${data.analysis_output_dir}/velocity_cartesian_latte_output.csv"
  use_chapman_inverse: false
  earth_radius_km: 6371.0

# Travel time processing
travel_time_processing:
  enabled: true
  ground_truth_csv: "/full/path/to/ground_truth_travel_times_corrected.csv"
  synthetic_tt_dir: "${data.base_dir}/synthetic_traveltimes"
  shot_map_file: "/full/path/to/shot_to_source_id_map.txt"
  synthetic_tt_output_csv: "${data.analysis_output_dir}/synthetic_travel_times_latte_output.csv"
```

### Required Path Updates

Update these paths to match your directory structure:

```yaml
velocity_processing:
  original_velocity_csv: "/home/user/project/preprocessing/velocity_cartesian_final.csv"

travel_time_processing:
  ground_truth_csv: "/home/user/project/preprocessing/ground_truth_travel_times_corrected.csv"
  shot_map_file: "/home/user/project/preprocessing/shot_to_source_id_map.txt"
```

## Usage

### Basic Execution

```bash
python main.py
```

The script automatically loads configuration from `../configs/postprocess/default.yaml`.

### Hydra Overrides

Override configuration parameters from command line:

```bash
# Enable only velocity processing
python main.py travel_time_processing.enabled=false

# Change output directory
python main.py data.analysis_output_dir="/new/output/path"

# Enable Chapman transformation
python main.py velocity_processing.use_chapman_inverse=true

# Set debug logging
python main.py general.log_level=DEBUG
```

### Alternative Configurations

```bash
# Use different config file
python main.py --config-name=postprocess/production

# Specify config directory
python main.py --config-path=/custom/config/path
```

## Outputs

### Velocity CSV (`velocity_cartesian_latte_output.csv`)

Contains updated velocity model with original structure preserved:

| Column | Description |
|--------|-------------|
| `lon`, `lat`, `depth` | Original geographic coordinates |
| `vp`, `vs` | **Updated** velocity values from inversion |
| `x_km`, `y_km`, `z_km` | Cartesian coordinates |
| `X_grid_km`, `Y_grid_km`, `Z_grid_km` | Grid coordinates |
| `i_raw`, `j_raw`, `k_raw` | Raw grid indices |
| `i`, `j`, `k` | Final grid indices |

**Key Features:**
- Maintains exact same structure as original velocity CSV
- Converts velocities from m/s to km/s
- Optional Chapman inverse transformation
- Preserves all coordinate systems and indexing

### Travel Time CSV (`synthetic_travel_times_latte_output.csv`)

Combines observed and synthetic travel times with comprehensive metadata:

| Column Group | Columns | Description |
|--------------|---------|-------------|
| **Identifiers** | `shot_id`, `event_id`, `station`, `phase` | Event and station identification |
| **Travel Times** | `observed_tt`, `synthetic_tt`, `residual_s` | Time measurements and residuals |
| **Metadata** | `weight`, `origin_time_s`, `weighted_rms` | Pick weights and timing |
| **Original Coordinates** | `event_longitude`, `event_latitude`, `event_depth_km` | Geographic event location |
| | `station_longitude`, `station_latitude`, `station_depth_km` | Geographic station location |
| **Transformed Coordinates** | `event_x_transformed`, `event_y_transformed`, `event_z_transformed` | Cartesian event coordinates |
| | `station_x_transformed`, `station_y_transformed`, `station_z_transformed` | Cartesian station coordinates |
| **Derived** | `distance_m` | Source-receiver distance |

**Key Features:**
- Self-contained: no external mapping files required
- Consistent ordering with ground truth data
- Automatic residual calculation (observed - synthetic)
- Complete coordinate information for spatial analysis

## Processing Details

### Velocity Processing

1. **Load Original Structure**: Reads original velocity CSV to preserve grid dimensions and coordinate systems
2. **Binary Loading**: Reads LATTE velocity binaries using little-endian 32-bit floats
3. **Grid Matching**: Maps binary array indices to CSV grid coordinates using `(i-1, j-1, k-1)` indexing
4. **Unit Conversion**: Converts velocities from m/s to km/s
5. **Coordinate Transformation**: Applies Chapman inverse transformation if enabled
6. **Output Generation**: Creates CSV with updated velocities preserving original structure

### Travel Time Processing

1. **Template Loading**: Uses ground truth CSV as template for structure and metadata
2. **Shot Mapping**: Loads shot ID to event ID mapping from preprocessing
3. **Binary Processing**: Reads synthetic travel time binaries for each shot and phase
4. **Station Matching**: Ensures consistent station ordering between ground truth and synthetic data
5. **Quality Control**: Filters unrealistic synthetic times (>1e10 seconds)
6. **Residual Calculation**: Computes residuals as observed - synthetic
7. **Metadata Preservation**: Transfers all coordinates and metadata from ground truth

### Chapman Transformation

When `use_chapman_inverse: true`:

```
v_spherical = exp(-depth/R) * v_flattened
```

Where:
- `R = 6371.0 km` (Earth radius)
- `depth` in km (positive downward)
- Handles negative depths (above sea level) gracefully

## Error Handling

### Strict Mode (`strict_mode: true`)
- Stops processing on first error
- Recommended for production workflows

### Permissive Mode (`strict_mode: false`)
- Continues processing despite errors
- Logs warnings for failed components
- Useful for debugging and partial processing

### Common Issues

1. **File Not Found**: Check paths in configuration
2. **Grid Dimension Mismatch**: Verify binary file matches original CSV grid
3. **Station Ordering Mismatch**: Ensure ground truth CSV has consistent station ordering
4. **Binary Format Issues**: Confirm LATTE outputs use little-endian 32-bit floats

## Integration Workflow

### Complete Processing Pipeline

1. **Preprocessing**: Generate ground truth CSV and coordinate mappings
2. **LATTE Inversion**: Run tomography to produce binary outputs
3. **Post-processing**: Convert binaries to CSV format (this module)
4. **Analysis**: Use CSV files for comparison and visualization

### Directory Structure Example

```
project/
├── preprocessing/
│   ├── ground_truth_travel_times_corrected.csv
│   ├── velocity_cartesian_final.csv
│   └── shot_to_source_id_map.txt
├── latte_run/
│   ├── model/
│   │   ├── updated_vp.bin
│   │   └── updated_vs.bin
│   └── synthetic_traveltimes/
│       ├── shot_1_traveltime_p.bin
│       ├── shot_1_traveltime_s.bin
│       └── ...
└── analysis/
    ├── velocity_cartesian_latte_output.csv
    └── synthetic_travel_times_latte_output.csv
```

## API Reference

### Core Functions

#### `create_velocity_csv_from_binaries(config: DictConfig)`
Converts velocity binary files to CSV format.

**Parameters:**
- `config`: Configuration containing file paths and processing options

**Outputs:**
- CSV file with updated velocity values

#### `create_synthetic_travel_time_csv(config: DictConfig)`
Combines synthetic travel time binaries with ground truth metadata.

**Parameters:**
- `config`: Configuration containing file paths

**Outputs:**
- CSV file with observed and synthetic travel times

#### `postprocess_latte_outputs(config: DictConfig)`
Main processing function that orchestrates velocity and travel time processing.

**Parameters:**
- `config`: Complete configuration object

### Utility Functions

#### `get_grid_dimensions_from_csv(df: DataFrame) -> Tuple[int, int, int]`
Extracts grid dimensions from velocity CSV.

#### `load_velocity_binary(bin_file: str, shape: Tuple, velocity_type: str) -> ndarray`
Loads and validates velocity binary files.

#### `apply_chapman_inverse_transform(velocity: float, depth: float, radius: float) -> float`
Applies Chapman inverse coordinate transformation.

## Performance Considerations

- **Memory Usage**: Velocity binaries are loaded entirely into memory
- **Processing Time**: Scales with number of shots and stations
- **Disk I/O**: Sequential binary file reading optimized for SSD storage

## Logging

### Log Levels

- **DEBUG**: Full configuration dump, detailed processing steps
- **INFO**: Major processing stages, file counts, basic statistics
- **WARNING**: Non-fatal issues, missing files, mismatched counts
- **ERROR**: Fatal errors requiring intervention

### Log Output Example

```
2024-01-15 14:30:15 - __main__ - INFO - Starting LATTE post-processing
2024-01-15 14:30:15 - __main__ - INFO - Velocity processing: ENABLED
2024-01-15 14:30:16 - __main__ - INFO - Grid dimensions detected: NX=100, NY=80, NZ=40
2024-01-15 14:30:17 - __main__ - INFO - Updated Vp: 32000 points
2024-01-15 14:30:17 - __main__ - INFO - Travel time processing: ENABLED
2024-01-15 14:30:20 - __main__ - INFO - Synthetic travel time CSV written: 15420 records
2024-01-15 14:30:20 - __main__ - INFO - Post-processing completed successfully!
```

## Troubleshooting

### Configuration Issues
- Verify all file paths exist and are accessible
- Check YAML syntax with proper indentation
- Ensure variable substitutions (`${data.base_dir}`) resolve correctly

### Data Issues
- Confirm binary files are from same LATTE run as ground truth
- Verify grid dimensions match between velocity CSV and binaries
- Check that shot numbering is consistent across files

### Performance Issues
- Monitor memory usage for large velocity models
- Consider processing subsets for very large datasets
- Verify sufficient disk space for output files