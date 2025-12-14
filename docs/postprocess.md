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
  phase_type: both         # both | p | s
  max_depth_km: null       # Depth filtering (km); null to disable
  precision:
    default: single        # single | double - Most binaries
    st0: single            # single | double - Origin time files
```

#### Phase Type Configuration

Controls which phase files are expected during processing.

| Phase Type | P-wave Files | S-wave Files | Use Case |
|------------|-------------|-------------|----------|
| `both` | Required | Required | Joint P+S inversion |
| `p` | Required | Not expected | P-wave only inversion |
| `s` | Not expected | Required | S-wave only inversion |

**Critical**: Must match preprocessing `phase_type` setting. Mismatches cause file not found errors.

**Binary Naming**: `shot_{id}_traveltime_{p\|s}.bin`

#### Binary Precision Configuration

Specifies floating-point precision for reading binary files.

| Precision | Bytes/Value | NumPy Type | Use Case |
|-----------|------------|------------|----------|
| `single` | 4 | `float32` | Standard LATTE output |
| `double` | 8 | `float64` | High-precision requirements |

**precision.default**: Applies to velocity (`vp`, `vs`) and source coordinate (`sx`, `sy`, `sz`) binaries.

**precision.st0**: Applies to origin time binaries (`updated_st0.bin`, `grad_st0.bin`, `srch_st0.bin`).

**Critical**: `precision.st0` must match preprocessing `origin_time_precision` setting.

#### Depth Filtering

Removes points deeper than specified threshold from output CSVs.

```yaml
max_depth_km: 30.0  # Remove points with Z_grid_km > 30 km
```

- Applied after all processing (velocity, travel time, source)
- Filters on `Z_grid_km` column
- Set to `null` to disable
- **Recommendation**: Match preprocessing `max_depth_km` for consistency

### Velocity Processing

```yaml
velocity_processing:
  enabled: true
  original_velocity_csv: "/path/to/velocity_cartesian_final.csv"
  vp_binary: "${data.base_dir}/model/updated_vp.bin"
  vs_binary: "${data.base_dir}/model/updated_vs.bin"
  fine_grid:                       # LATTE internal grid specification
    nx: 401
    ny: 361
    nz: 43
    dx: 0.5                        # Grid spacing (km)
    dy: 0.5
    dz: 0.5
    origin_x: 0.0                  # Grid origin (km)
    origin_y: 0.0
    origin_z: 0.0
  interpolation_mode: sample_at_coarse  # sample_at_coarse | expand_to_fine
  velocity_output_csv: "${data.analysis_output_dir}/velocity_cartesian_latte_output.csv"
  use_chapman_inverse: false       # Apply Chapman transformation
  earth_radius_km: 6371.0
  parameters_to_process: [vp, vs] # Process gradient/search direction files
  grad_vp_binary: "${data.base_dir}/model/grad_vp.bin"
  grad_vs_binary: "${data.base_dir}/model/grad_vs.bin"
  srch_vp_binary: "${data.base_dir}/model/srch_vp.bin"
  srch_vs_binary: "${data.base_dir}/model/srch_vs.bin"
```

#### Fine Grid Specification

The **fine_grid** defines LATTE's internal computation grid (from binary files).

**Parameters**:
- `nx, ny, nz`: Grid dimensions in each direction
- `dx, dy, dz`: Uniform grid spacing (km)
- `origin_x, origin_y, origin_z`: Grid origin coordinates (km)

**Source**: Typically extracted from LATTE configuration or binary metadata.

**Coarse Grid**: Auto-detected from `original_velocity_csv` structure. Represents the preprocessing output grid, which may differ from fine grid if LATTE uses mesh refinement.

#### Interpolation Mode

Controls output grid resolution and memory usage.

| Mode | Behavior | Output Size | Memory | Use Case |
|------|----------|-------------|--------|----------|
| `sample_at_coarse` | Sample fine grid values at coarse grid points | Coarse grid size | Low | Standard workflow |
| `expand_to_fine` | Output full fine grid with interpolated metadata | Fine grid size | High | High-resolution analysis |

**sample_at_coarse** (Default):
- Preserves coarse grid structure from preprocessing
- Samples fine grid velocities at coarse grid locations
- Output CSV size matches input CSV
- Recommended for most workflows

**expand_to_fine**:
- Creates all fine grid points
- Interpolates coarse CSV metadata to fine grid
- Assigns velocities directly from binaries
- Significantly larger output files
- Use when fine grid resolution is required for analysis

#### Gradient and Search Direction Processing

Optionally process gradient and search direction fields from optimization.

```yaml
parameters_to_process: [vp, vs]  # Or [vp], [vs], or []
```

**Files Required** (per parameter):
- `grad_{vp|vs}.bin`: Gradient of objective function
- `srch_{vp|vs}.bin`: Search direction from optimization

**Output CSV Columns** (when enabled):
- `grad_vp`, `grad_vs`: Gradient magnitude at each grid point
- `srch_vp`, `srch_vs`: Search direction magnitude

**Use Cases**:
- Analyzing sensitivity patterns
- Identifying well-constrained regions
- Diagnosing convergence issues
- Understanding optimization behavior

**Notes**:
- Set `parameters_to_process: []` to skip gradient processing
- Files are optional; missing files are logged and skipped
- Gradient files use same precision as velocity binaries

### Travel Time Processing

```yaml
travel_time_processing:
  enabled: true
  ground_truth_csv: "/path/to/ground_truth_travel_times_corrected.csv"
  synthetic_tt_dir: "${data.base_dir}/synthetic_traveltimes"
  shot_map_file: "/path/to/shot_to_source_id_map.txt"
  removed_shots: []  # List of 1-based shot IDs filtered by LATTE
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

#### Removed Shots Configuration

Specifies shots that were filtered out by LATTE during geometry loading.

```yaml
removed_shots: [5, 12, 47, 103]  # 1-based shot IDs
```

**Why This Is Needed**:
- LATTE may exclude shots that fail geometry validation
- Common reasons: missing stations, insufficient ray coverage, geometry constraints
- Preprocessing shot numbering includes all shots
- LATTE's internal shot indexing excludes removed shots

**Impact on Processing**:
- Standard mode: Minimal impact (files for removed shots simply don't exist)
- **Source-receiver exchange mode**: Critical for correct array indexing

**Source-Receiver Exchange Complications**:
- Exchange mode uses dense arrays indexed by shot_id
- Array size = number of valid shots (after removal)
- Array index = `(shot_id - 1) - number_of_removed_shots_before_this_id`
- Without `removed_shots` list, mapping becomes incorrect

**Example**:
```yaml
# Preprocessing: 100 shots (IDs 1-100)
# LATTE removes shots: [5, 12, 47]
# LATTE internal: 97 shots with renumbered indices

removed_shots: [5, 12, 47]  # Must match LATTE's exclusion list
```

**Determining Removed Shots**:
- Check LATTE log files for geometry filtering messages
- Compare preprocessing shot count vs. LATTE's reported shot count
- Verify by checking which `shot_*_traveltime_*.bin` files exist

### Source Processing

```yaml
source_processing:
  enabled: true
  sx_file: "${data.base_dir}/model/updated_sx.bin"
  sy_file: "${data.base_dir}/model/updated_sy.bin"
  sz_file: "${data.base_dir}/model/updated_sz.bin"
  st0_file: "${data.base_dir}/model/updated_st0.bin"
  grid_dimensions: [4495]  # Number of sources
  output_csv: "${data.analysis_output_dir}/source_parameters_latte_output.csv"
  parameters_to_process: [sx, sy, sz, st0]  # Process gradient/search direction files
  grad_sx_file: "${data.base_dir}/model/grad_sx.bin"
  grad_sy_file: "${data.base_dir}/model/grad_sy.bin"
  grad_sz_file: "${data.base_dir}/model/grad_sz.bin"
  grad_st0_file: "${data.base_dir}/model/grad_st0.bin"
  srch_sx_file: "${data.base_dir}/model/srch_sx.bin"
  srch_sy_file: "${data.base_dir}/model/srch_sy.bin"
  srch_sz_file: "${data.base_dir}/model/srch_sz.bin"
  srch_st0_file: "${data.base_dir}/model/srch_st0.bin"
```

#### Gradient and Search Direction Processing

Optionally process gradient and search direction fields for source parameters.

```yaml
parameters_to_process: [sx, sy, sz, st0]  # Or any subset, or []
```

**Files Required** (per parameter):
- `grad_{sx|sy|sz|st0}.bin`: Gradient of objective function w.r.t. source parameter
- `srch_{sx|sy|sz|st0}.bin`: Search direction for source parameter

**Output CSV Columns** (when enabled):
- `grad_sx`, `grad_sy`, `grad_sz`, `grad_st0`: Gradient values per source
- `srch_sx`, `srch_sy`, `srch_sz`, `srch_st0`: Search direction values per source

**Use Cases**:
- Identifying poorly constrained sources
- Analyzing sensitivity to source position/timing
- Understanding optimization convergence patterns
- Diagnosing source location quality

**Precision**:
- Coordinate gradients (`sx`, `sy`, `sz`) use `precision.default`
- Origin time gradients (`st0`) use `precision.st0`

**Notes**:
- Set `parameters_to_process: []` to skip gradient processing
- Files are optional; missing files are logged and skipped
- Each binary file contains one value per source

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

**Optional Columns** (when gradient processing enabled):
- `grad_vp`, `grad_vs`: Gradient magnitudes
- `srch_vp`, `srch_vs`: Search direction magnitudes

**Features**:
- Automatic conversion from m/s to km/s
- Optional Chapman inverse transformation
- Preserves all coordinate systems
- Grid resolution depends on `interpolation_mode` setting

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

**File**: `source_parameters_latte_output.csv`

**Key Columns**:
- `shot_id`: Source identifier (1-based)
- `sx_km`, `sy_km`, `sz_km`: Updated source coordinates (km)
- `st0_s`: Updated origin time (s)

**Optional Columns** (when gradient processing enabled):
- `grad_sx`, `grad_sy`, `grad_sz`: Coordinate gradients
- `grad_st0`: Origin time gradient
- `srch_sx`, `srch_sy`, `srch_sz`: Coordinate search directions
- `srch_st0`: Origin time search direction

**Features**:
- One row per source
- Coordinate corrections relative to preprocessing values
- Gradient values indicate sensitivity/constraint quality

---

## Processing Details

### Velocity Processing

**Workflow**:
1. Load original velocity CSV to preserve structure
2. Read fine grid specification and detect coarse grid from CSV
3. Read binary files (`updated_vp.bin`, `updated_vs.bin`) on fine grid
4. Optionally read gradient and search direction binaries
5. Apply processing mode:
   - **sample_at_coarse**: Sample fine grid at coarse points
   - **expand_to_fine**: Interpolate to full fine grid
6. Convert velocities from m/s to km/s
7. Apply Chapman transformation (if enabled)
8. Apply depth filtering (if enabled)
9. Generate output CSV

**Grid Relationship**:
- **Fine Grid**: LATTE's internal computation grid (from binaries)
- **Coarse Grid**: Preprocessing output grid (from CSV)
- Fine grid typically equals or refines coarse grid

**Chapman Transformation**:
```
v_spherical = exp(-depth/R) * v_flattened
```
Where `R = 6371.0 km` (Earth radius), depth in km (positive downward)

**Binary Format**:
- Precision determined by `precision.default` setting
- Single: 4 bytes/value (float32), Double: 8 bytes/value (float64)
- Storage order: C-order (row-major), dimensions `[nx, ny, nz]`

### Travel Time Processing

#### Standard Mode

**Workflow**:
1. Load ground truth CSV as template
2. Load shot ID to event ID mapping
3. Identify removed shots (if specified)
4. Process each valid shot and phase:
   - Read synthetic travel time binary
   - Match stations with ground truth ordering
   - Apply source correction (if enabled)
   - Calculate residuals
5. Apply depth filtering (if enabled)
6. Generate output CSV with preserved metadata

**Binary Naming**: `shot_{shot_id}_traveltime_{p|s}.bin`

**Phase-Specific Processing**:
- `phase_type: both`: Process both P and S binaries for each shot
- `phase_type: p`: Process only P binaries
- `phase_type: s`: Process only S binaries

**Precision**: Determined by `precision.default` for travel times

#### Source-Receiver Exchange Mode

**Workflow**:
1. Load virtual station geometry and match to actual stations
2. Load removed shots list for correct array indexing
3. For each virtual station (one binary file per station per phase):
   - Read travel time array (indexed by shot_id, accounting for removed shots)
   - Apply source correction per shot
   - Match with ground truth records
4. Apply depth filtering (if enabled)
5. Generate output CSV

**Binary Naming**: `shot_{virtual_id}_traveltime_{p|s}.bin`
**Array Structure**: Index calculated accounting for removed shots

**Removed Shots Impact**:
- Array index must account for shots filtered by LATTE
- Incorrect `removed_shots` list causes wrong travel time assignments
- Critical for data integrity in exchange mode

### Source Parameter Processing

**Workflow**:
1. Read binary files: `updated_sx.bin`, `updated_sy.bin`, `updated_sz.bin`, `updated_st0.bin`
2. Optionally read gradient and search direction binaries
3. Validate source count matches `grid_dimensions`
4. Generate output CSV with one row per source
5. Apply depth filtering (if enabled)

**Binary Format**:
- One value per source
- 1-based source/shot indexing
- Precision:
  - Coordinates (`sx`, `sy`, `sz`): `precision.default`
  - Origin time (`st0`): `precision.st0`

**Gradient/Search Files**:
- Same format as main parameter files
- Optional; missing files logged and skipped
- Units match corresponding parameter (m for coordinates, s for time)

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
| **File Not Found** | Check paths; verify `phase_type` matches available binaries |
| **Grid Dimension Mismatch** | Verify `fine_grid` dimensions match binary file sizes |
| **Station Ordering Mismatch** | Ensure consistent station sorting in ground truth |
| **Binary Format Issues** | Check `precision.default` and `precision.st0` settings |
| **Virtual Station Mismatch** | Adjust `coordinate_tolerance` parameter |
| **Wrong Travel Times in Exchange Mode** | Verify `removed_shots` list matches LATTE filtering |
| **Phase Type Mismatch** | Ensure `phase_type` matches preprocessing setting |
| **Memory Issues with expand_to_fine** | Use `sample_at_coarse` mode or increase available memory |

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
│   │   ├── updated_st0.bin
│   │   ├── grad_vp.bin           # Optional gradient files
│   │   ├── grad_vs.bin
│   │   ├── srch_vp.bin           # Optional search direction files
│   │   ├── srch_vs.bin
│   │   ├── grad_sx.bin
│   │   ├── grad_sy.bin
│   │   ├── grad_sz.bin
│   │   ├── grad_st0.bin
│   │   ├── srch_sx.bin
│   │   ├── srch_sy.bin
│   │   ├── srch_sz.bin
│   │   └── srch_st0.bin
│   └── synthetic_traveltimes/
│       ├── shot_1_traveltime_p.bin
│       ├── shot_1_traveltime_s.bin
│       └── ...
└── analysis/
    ├── velocity_cartesian_latte_output.csv
    ├── synthetic_travel_times_latte_output.csv
    └── source_parameters_latte_output.csv
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
  - `sample_at_coarse`: Memory ~ coarse grid size
  - `expand_to_fine`: Memory ~ fine grid size (potentially much larger)
- **Processing Time**:
  - Scales with number of shots × stations for travel times
  - Gradient processing adds minimal overhead
- **Disk I/O**: Optimized for SSD storage
- **Typical Runtime**: Minutes for thousands of shots
- **Precision Impact**: Double precision doubles memory and I/O requirements

---

## Best Practices

1. **Always validate configuration** before running on production data
2. **Use strict mode** for final processing runs
3. **Match preprocessing settings**:
   - `phase_type` must match preprocessing
   - `precision.st0` must match `origin_time_precision`
   - `max_depth_km` should match for consistency
4. **For source-receiver exchange**:
   - Carefully specify `removed_shots` from LATTE logs
   - Verify shot count matches expected value
5. **Choose interpolation mode wisely**:
   - Use `sample_at_coarse` for standard workflows
   - Use `expand_to_fine` only when fine resolution is required
6. **Gradient processing**:
   - Enable only when needed for analysis
   - Set `parameters_to_process: []` to skip if not required
7. **Check output statistics** to verify reasonable residuals
8. **Preserve original files** - post-processing creates new files
9. **Document processing parameters** used for each inversion
10. **Review log files** for warnings or missing gradient files

---

**Version:** 1.0.0  
**Last Updated:** 2025