# LATTE Post-Processing Documentation

## Overview

The post-processing module converts LATTE tomography binary outputs into analysis-ready CSV files. It bridges the gap between inversion results and downstream analysis workflows while maintaining consistency with preprocessing outputs.

---

## Features

### Core Capabilities

- **Velocity Model Conversion**: Transforms `updated_vp.bin` and `updated_vs.bin` into structured CSV format
- **Synthetic Travel Time Processing**: Merges synthetic travel time binaries with ground truth metadata
- **Source Parameter Updates**: Processes updated source locations and origin times
- **Source-Receiver Exchange**: Handles reciprocity-based inversion outputs
- **Coordinate Transformations**: Optional Chapman inverse transformation for spherical coordinates

### Output Characteristics

- Self-contained CSV files requiring no external mapping files
- Automatic residual calculation (observed - synthetic)
- Complete metadata preservation from preprocessing
- Consistent structure with ground truth data

---

## Getting Started

### Quick Start

```bash
# Run with default configuration
python -m postprocess.main

# View available options
python -m postprocess.main --help
```

### Configuration

Configuration file location: `configs/postprocess/default.yaml`

#### Required Path Updates

Before running, update these paths to match your directory structure:

```yaml
data:
  base_dir: "/path/to/latte/output"
  analysis_output_dir: "/path/to/analysis/output"

velocity_processing:
  original_velocity_csv: "/full/path/to/velocity_cartesian_final.csv"

travel_time_processing:
  ground_truth_csv: "/full/path/to/ground_truth_travel_times_corrected.csv"
  shot_map_file: "/full/path/to/shot_to_source_id_map.txt"
```

---

## Configuration Reference

### General Settings

```yaml
general:
  strict_mode: true        # Stop on first error (true) vs. continue (false)
  log_level: INFO          # DEBUG | INFO | WARNING | ERROR
```

### Velocity Processing

```yaml
velocity_processing:
  enabled: true
  original_velocity_csv: "/path/to/velocity_cartesian_final.csv"
  vp_binary: "${data.base_dir}/model/updated_vp.bin"
  vs_binary: "${data.base_dir}/model/updated_vs.bin"
  grid_dimensions: [401, 361, 43]  # [nx, ny, nz]; auto-detected if null
  velocity_output_csv: "${data.analysis_output_dir}/velocity_cartesian_latte_output.csv"
  use_chapman_inverse: false       # Apply Chapman transformation
  earth_radius_km: 6371.0
```

### Travel Time Processing

```yaml
travel_time_processing:
  enabled: true
  ground_truth_csv: "/path/to/ground_truth_travel_times_corrected.csv"
  synthetic_tt_dir: "${data.base_dir}/synthetic_traveltimes"
  shot_map_file: "/path/to/shot_to_source_id_map.txt"
  synthetic_tt_output_csv: "${data.analysis_output_dir}/synthetic_travel_times_latte_output.csv"
  
  # Reciprocity-based inversion
  source_receiver_exchange: false
  exchange_mapping_file: "${data.base_dir}/exchange_mapping.txt"
  exchange_geometry_file: "${data.base_dir}/exchange_geometry.txt"
  virtual_shot_mapping_file: "${data.base_dir}/shot_mapping.txt"  # Optional
  coordinate_tolerance: 1.0  # meters
  
  # Source correction
  source_correction: false
  source_correction_file: "${data.base_dir}/model/updated_st0.bin"
```

### Source Processing

```yaml
source_processing:
  enabled: true
  sx_file: "${data.base_dir}/model/updated_sx.bin"
  sy_file: "${data.base_dir}/model/updated_sy.bin"
  sz_file: "${data.base_dir}/model/updated_sz.bin"
  st0_file: "${data.base_dir}/model/updated_st0.bin"
  grid_dimensions: [4495]  # Number of sources
  output_csv: "${data.analysis_output_dir}/source_updated_travel_times.csv"
```

---

## Usage Examples

### Standard Processing

```bash
python -m postprocess.main
```

### Command-Line Overrides

```bash
# Disable travel time processing
python -m postprocess.main travel_time_processing.enabled=false

# Change output directory
python -m postprocess.main data.analysis_output_dir="/new/output/path"

# Enable Chapman transformation
python -m postprocess.main velocity_processing.use_chapman_inverse=true

# Enable debug logging
python -m postprocess.main general.log_level=DEBUG
```

### Custom Configuration File

```bash
# Use alternative config
python -m postprocess.main --config-name=production

# Specify custom config path
python -m postprocess.main --config-path=/custom/config/path
```

### Advanced Workflows

#### Source-Receiver Exchange Mode

```bash
python -m postprocess.main \
  travel_time_processing.source_receiver_exchange=true \
  travel_time_processing.exchange_geometry_file="/path/to/exchange_geometry.txt" \
  travel_time_processing.exchange_mapping_file="/path/to/exchange_mapping.txt"
```

#### Source Correction

```bash
python -m postprocess.main \
  travel_time_processing.source_correction=true \
  travel_time_processing.source_correction_file="/path/to/updated_st0.bin"
```

#### Combined Source Updates

```bash
python -m postprocess.main \
  source_processing.enabled=true \
  travel_time_processing.source_correction=true
```

---

## Output Files

### Velocity CSV

**File**: `velocity_cartesian_latte_output.csv`

**Structure**: Preserves original velocity CSV format with updated Vp and Vs values.

**Key Columns**:
- `lon`, `lat`, `depth`: Geographic coordinates
- `vp`, `vs`: Updated velocity values (km/s)
- `x_km`, `y_km`, `z_km`: Cartesian coordinates
- `X_grid_km`, `Y_grid_km`, `Z_grid_km`: Grid coordinates
- `i`, `j`, `k`: Grid indices

**Features**:
- Automatic conversion from m/s to km/s
- Optional Chapman inverse transformation
- Preserves all coordinate systems

### Travel Time CSV

**File**: `synthetic_travel_times_latte_output.csv`

**Structure**: Combines observed and synthetic travel times with comprehensive metadata.

**Key Columns**:

| Column Group | Description |
|--------------|-------------|
| **Identifiers** | `shot_id`, `event_id`, `station`, `phase` |
| **Travel Times** | `observed_tt`, `synthetic_tt`, `residual_s` |
| **Metadata** | `weight`, `origin_time_s`, `weighted_rms` |
| **Event Coordinates** | Geographic: `event_longitude`, `event_latitude`, `event_depth_km` |
|  | Cartesian: `event_x_transformed`, `event_y_transformed`, `event_z_transformed` |
| **Station Coordinates** | Geographic: `station_longitude`, `station_latitude`, `station_depth_km` |
|  | Cartesian: `station_x_transformed`, `station_y_transformed`, `station_z_transformed` |
| **Derived** | `distance_m` |

**Optional Columns** (when source correction enabled):
- `corrected_origin_time_s`: Updated origin time

**Features**:
- Self-contained with complete metadata
- Residuals calculated as: observed - synthetic
- Consistent ordering with ground truth

### Source Updates CSV

**File**: `source_updated_travel_times.csv`

**Additional Columns** (beyond travel time CSV):
- `corrected_event_x_transformed`: Updated event X coordinate (m)
- `corrected_event_y_transformed`: Updated event Y coordinate (m)
- `corrected_event_z_transformed`: Updated event Z coordinate (m)
- `corrected_origin_time_s`: Updated origin time (s)

---

## Processing Details

### Velocity Processing

**Workflow**:
1. Load original velocity CSV to preserve structure
2. Read binary files (`updated_vp.bin`, `updated_vs.bin`)
3. Match binary indices to CSV grid using `(i-1, j-1, k-1)` mapping
4. Convert velocities from m/s to km/s
5. Apply Chapman transformation (if enabled)
6. Generate output CSV with preserved structure

**Chapman Transformation**:
```
v_spherical = exp(-depth/R) * v_flattened
```
Where `R = 6371.0 km` (Earth radius), depth in km (positive downward)

**Binary Format**: Little-endian 32-bit float, C-order (row-major)

### Travel Time Processing

#### Standard Mode

**Workflow**:
1. Load ground truth CSV as template
2. Load shot ID to event ID mapping
3. Process each shot and phase:
   - Read synthetic travel time binary
   - Match stations with ground truth ordering
   - Apply source correction (if enabled)
   - Calculate residuals
4. Generate output CSV with preserved metadata

**Binary Naming**: `shot_{shot_id}_traveltime_{p|s}.bin`

#### Source-Receiver Exchange Mode

**Workflow**:
1. Load virtual station geometry and match to actual stations
2. For each virtual station (one binary file per station per phase):
   - Read travel time array (indexed by shot_id)
   - Apply source correction per shot
   - Match with ground truth records
3. Generate output CSV

**Binary Naming**: `shot_{virtual_id}_traveltime_{p|s}.bin`  
**Array Structure**: Index = shot_id - 1 (0-based)

### Source Parameter Processing

**Workflow**:
1. Read binary files: `updated_sx.bin`, `updated_sy.bin`, `updated_sz.bin`, `updated_st0.bin`
2. Create corrections lookup (1-based shot_id → corrections)
3. Load travel time CSV
4. Apply corrections to each record
5. Calculate correction magnitudes
6. Generate output CSV with corrected parameters

**Binary Format**: One value per source, 1-based indexing

---

## Error Handling

### Strict Mode (`strict_mode: true`)
- Stops processing on first error
- Recommended for production
- Ensures data consistency

### Permissive Mode (`strict_mode: false`)
- Continues despite errors
- Logs warnings for failed components
- Useful for debugging

### Common Issues

| Issue | Solution |
|-------|----------|
| **File Not Found** | Check all paths in configuration |
| **Grid Dimension Mismatch** | Verify binary dimensions match velocity CSV grid |
| **Station Ordering Mismatch** | Ensure consistent station sorting in ground truth |
| **Binary Format Issues** | Confirm little-endian 32-bit float format |
| **Virtual Station Mismatch** | Adjust `coordinate_tolerance` parameter |

---

## Integration Workflow

### Complete Pipeline

```
Preprocessing → LATTE Inversion → Post-Processing → Analysis
```

**Directory Structure Example**:

```
project/
├── preprocessing/
│   ├── ground_truth_travel_times_corrected.csv
│   ├── velocity_cartesian_final.csv
│   └── shot_to_source_id_map.txt
├── latte_run/
│   ├── model/
│   │   ├── updated_vp.bin
│   │   ├── updated_vs.bin
│   │   ├── updated_sx.bin
│   │   ├── updated_sy.bin
│   │   ├── updated_sz.bin
│   │   └── updated_st0.bin
│   └── synthetic_traveltimes/
│       ├── shot_1_traveltime_p.bin
│       ├── shot_1_traveltime_s.bin
│       └── ...
└── analysis/
    ├── velocity_cartesian_latte_output.csv
    ├── synthetic_travel_times_latte_output.csv
    └── source_updated_travel_times.csv
```

---

## API Reference

### Main Functions

#### `postprocess_latte_outputs(config: DictConfig)`
Orchestrates all post-processing operations.

**Parameters**: Configuration object  
**Outputs**: CSV files as specified in config

#### `create_velocity_csv_from_binaries(config: DictConfig)`
Converts velocity binaries to CSV.

**Outputs**: Velocity CSV with updated Vp/Vs values

#### `create_synthetic_travel_time_csv(config: DictConfig)`
Standard travel time processing.

**Outputs**: Travel time CSV with residuals

#### `create_synthetic_travel_time_csv_with_exchange(config: DictConfig)`
Exchange mode travel time processing.

**Outputs**: Travel time CSV using reciprocity

#### `process_source_updates(config: DictConfig)`
Processes source parameter updates.

**Outputs**: Travel time CSV with corrected source parameters

---

## Logging

### Log Levels

- **DEBUG**: Detailed processing steps, configuration dump
- **INFO**: Major stages, file counts, statistics
- **WARNING**: Non-fatal issues, missing files
- **ERROR**: Fatal errors requiring intervention

### Example Output

```
2024-01-15 14:30:15 - INFO - Starting LATTE post-processing
2024-01-15 14:30:15 - INFO - Velocity processing: ENABLED
2024-01-15 14:30:16 - INFO - Grid dimensions: NX=401, NY=361, NZ=43
2024-01-15 14:30:17 - INFO - Updated Vp: 6253440 points
2024-01-15 14:30:17 - INFO - Travel time processing: ENABLED
2024-01-15 14:30:20 - INFO - Total records: 15420
2024-01-15 14:30:20 - INFO - RMS residual: 0.0847 s
2024-01-15 14:30:20 - INFO - Post-processing completed successfully
```

---

## Troubleshooting

### Configuration Issues
- Verify YAML syntax and indentation
- Check that all file paths exist
- Ensure variable substitutions resolve correctly (`${data.base_dir}`)

### Data Issues
- Confirm binary files match ground truth versions
- Verify grid dimensions consistency
- Check shot numbering alignment across files

### Performance Issues
- Monitor memory usage for large models
- Consider processing subsets for testing
- Verify sufficient disk space

### Exchange Mode Issues
- Adjust `coordinate_tolerance` if stations don't match
- Verify exchange geometry file format
- Check virtual station ID ranges

---

## Performance Notes

- **Memory**: Velocity binaries loaded entirely into memory
- **Processing Time**: Scales with number of shots × stations
- **Disk I/O**: Optimized for SSD storage
- **Typical Runtime**: Minutes for thousands of shots

---

## Best Practices

1. **Always validate configuration** before running on production data
2. **Use strict mode** for final processing runs
3. **Check output statistics** to verify reasonable residuals
4. **Preserve original files** - post-processing creates new files
5. **Document processing parameters** used for each inversion
6. **Review log files** for warnings or errors

---

**Version:** 1.0.0  
**Last Updated:** 2025