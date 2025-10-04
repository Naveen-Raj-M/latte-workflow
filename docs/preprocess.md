# LATTE Preprocessing Guide

Comprehensive guide for the LATTE preprocessing module, which handles all data preparation for seismic tomography inversion.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Coordinate Transformation](#coordinate-transformation)
5. [Initial Velocity Model](#initial-velocity-model)
6. [Geometry & Travel Time Processing](#geometry--travel-time-processing)
7. [Velocity Binary Files](#velocity-binary-files)
8. [Visualization](#visualization)
9. [File Formats](#file-formats)
10. [Troubleshooting](#troubleshooting)

## Overview

The preprocessing pipeline consists of five components that can run individually or as a complete workflow:

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| Coordinate Transform | Convert geodetic to Cartesian | Station/event/velocity text files | Cartesian CSVs with grid indices |
| Initial 3D Model | Generate 3D from 1D model | 1D velocity model, 3D template | 3D initial velocity model |
| Geometry & Travel Time | Process phase picks | Phase pickings, Cartesian CSVs | Geometry files, travel time binaries |
| Velocity Binary | Create binary files | Velocity CSV | Fortran-compatible binaries |
| Visualization | Generate QC plots | Cartesian CSVs, binaries | PDF plots, GIF animations |

## Quick Start

### Complete Pipeline

```bash
python -m preprocess.main
```

### Individual Functions

```bash
# Coordinate transformation
python -m preprocess.main single_function=coordinate_transform

# Velocity model creation
python -m preprocess.main single_function=initial_3d_model

# Geometry and travel times
python -m preprocess.main single_function=geometry_traveltime

# Binary conversion
python -m preprocess.main single_function=velocity_binary

# Visualization
python -m preprocess.main single_function=visualization
```

## Configuration

### Configuration File

All preprocessing is controlled through `configs/preprocess/default.yaml`. Create custom configurations by copying and modifying:

```yaml
data:
  base_dir: "/path/to/your/data"
  output_base_dir: "./outputs"

coordinate_transform:
  enabled: true
  transformation_method: "flattening"
  origin_lon: -108.0
  origin_lat: 34.5
  origin_depth_km: -4.0
  grid_spacing_km: [20.0, 20.0, 2.0]
```

### Command Line Overrides

```bash
# Override data directory
python -m preprocess.main data.base_dir=/path/to/my/data

# Change transformation method
python -m preprocess.main coordinate_transform.transformation_method=exact

# Use corrected phase picks
python -m preprocess.main geometry_traveltime.use_corrected_phase_pickings=true
```

## Coordinate Transformation

Converts geographic coordinates (longitude, latitude, depth) to local Cartesian grid coordinates.

### Transformation Methods

#### Flattening (Default)

Fast geometric projection suitable for regional studies:

```yaml
coordinate_transform:
  transformation_method: "flattening"
  use_chapman_flattening: false  # Enable Chapman (1973) correction
```

**Chapman Flattening:**
- Applies Earth-flattening transformation for seismic studies
- Corrects both depths and velocities
- Preserves original values in separate columns

#### Exact Transformation

Precise ECEF to local ENU conversion:

```yaml
coordinate_transform:
  transformation_method: "exact"
  exact_transform:
    rotate_to_enu: true
    align_to_geocentric_normal: false
```

Requires `pyproj`: `pip install pyproj`

### Input Files

```yaml
station_file: "${data.base_dir}/stacoords.txt"           # 6 columns
event_file: "${data.base_dir}/phasePickings.txt"         # 9 columns
corrected_event_file: "${data.base_dir}/sphfdloc.dat"   # 10 columns (optional)
velocity_file: "${data.base_dir}/vmodel-PB3D.txt"        # 5 columns
final_velocity_file: "${data.base_dir}/vmodel-final.txt" # 5 columns (optional)
```

### Output Files

- `stations_cartesian.csv` - Station coordinates in Cartesian grid
- `events_cartesian.csv` - Event coordinates in Cartesian grid
- `velocity_cartesian.csv` - Velocity model in Cartesian grid
- `velocity_final_cartesian.csv` - Final velocity model (if enabled)

### Key Options

```yaml
use_corrected_phase_pickings: false  # Use corrected event data
process_final_velocity: false        # Process final velocity model
grid_spacing_km: [20.0, 20.0, 2.0]  # Grid spacing [dx, dy, dz]
```

Grid indices (i, j, k) are 1-based for Fortran compatibility.

## Initial Velocity Model

Creates a 3D initial velocity model by mapping a 1D reference model onto a 3D spatial grid.

### Configuration

```yaml
initial_3d_model:
  enabled: true
  model_1d_file: "${data.base_dir}/vmodel-PB1D.txt"
  template_3d_file: "${data.base_dir}/vmodel-PB3D.txt"
  output_file: "${data.output_base_dir}/initial_3d_model_generated.txt"
  
  precision:
    longitude: 6
    latitude: 7
    depth: 8
    vp: 8
    vs: 8
  field_width: 11
```

### Velocity Mapping

For each 3D grid point:
1. Find the deepest 1D layer at or above the grid point depth
2. Assign that layer's velocities
3. For shallow points: use first layer
4. For deep points: use deepest layer

### Input Format

**1D Model (3 columns):**
```
# depth(km)  vp(km/s)  vs(km/s)
0.0          5.0       2.9
1.0          5.5       3.2
2.0          6.0       3.5
```

**3D Template (5 columns):**
```
# lon    lat    depth   vp    vs
-108.0  34.5   0.0     0.0   0.0
-108.0  34.5   2.0     0.0   0.0
```

## Geometry & Travel Time Processing

Processes seismic phase picks to create geometry files and travel time data for LATTE.

### Configuration

```yaml
geometry_traveltime:
  enabled: true
  use_corrected_phase_pickings: false
  generate_source_binaries: true
  zero_origin_time: true
  source_binary_format: "travel_time"
  
  # Velocity filtering
  p_velocity_range_m_s: [500, 10000]
  s_velocity_range_m_s: [500, 6000]
  
  # Event/station filtering
  filter_events: false
  events_to_discard_file: "${data.base_dir}/events_to_discard.txt"
  filter_stations: false
  stations_to_discard_file: "${data.base_dir}/stations_to_discard.txt"
```

### Quality Control Filters

**Automatic Filtering:**
- Stations must exist in station CSV
- Events must exist in event CSV
- Coordinates must be non-negative
- Travel times must be positive
- Apparent velocities must be within specified ranges
- Both P and S picks required for each station-event pair

**Manual Filtering:**
- Discard specific events by ID
- Discard specific stations by name

### Output Files

**Geometry Files:**
- `geometry/shot_N_geometry.txt` - One per event
- `geometry/geometry.txt` - Index file
- `geometry/shot_to_source_id_map.txt` - Event ID mapping

**Travel Time Binaries:**
- `ground_truth_traveltimes/shot_N_traveltime_p.bin` - P-wave times
- `ground_truth_traveltimes/shot_N_traveltime_s.bin` - S-wave times

**Source Files (optional):**
- `model/sx.bin`, `sy.bin`, `sz.bin` - Source coordinates
- `model/st0.bin` - Origin times

**CSV Export (optional):**
- `geometry_csv/ground_truth_travel_times.csv` - Comprehensive data with metadata

### CSV Export

Enable comprehensive CSV output:

```yaml
generate_travel_time_csv: true
csv_output_dir: "${data.output_base_dir}/geometry_csv"
```

Includes: original coordinates, transformed coordinates, travel times, distances, weights, residuals (for corrected data).

### Phase Pick Formats

**Original (phasePickings.txt):**
```
# Event header (9 cols): year jday hour min sec lat lon depth event_id
2023 100 10 30 45.123 34.5 -108.0 5.0 event001
# Picks (7+ cols): station ? jday hour min sec phase [weight]
STA1 ? 100 10 31 2.456 P 1.0
```

**Corrected (sphfdloc.dat):**
```
# Event (10 cols): year jday hour min sec lat lon depth event_id weighted_rms
2023 100 10 30 45.123 34.5 -108.0 5.0 event001 0.25
# Picks (9 cols): station year jday hour min sec phase uncertainty residual
STA1 2023 100 10 31 2.456 P 0.05 -0.02
```

## Velocity Binary Files

Converts velocity models from CSV to Fortran-compatible binary format.

### Configuration

```yaml
velocity_binary:
  enabled: true
  input_csv: "${coordinate_transform.output_dir}/velocity_cartesian.csv"
  vp_output: "${data.output_base_dir}/vp_init.bin"
  vs_output: "${data.output_base_dir}/vs_init.bin"
  
  use_chapman_velocities: true
  interpolate: true
  neighbor_threshold: 5
  
  # Final velocity processing (optional)
  create_final_binaries: false
  final_input_csv: "${coordinate_transform.output_dir}/velocity_final_cartesian.csv"
  vp_final_output: "${data.output_base_dir}/vp_final.bin"
  vs_final_output: "${data.output_base_dir}/vs_final.bin"
```

### Processing Features

**Grid Dimensions:**
- Auto-detected from CSV data
- Uses maximum i, j, k values
- Warns if dimensions differ from expected

**Chapman Velocities:**
- Uses `vp_flat`/`vs_flat` if `use_chapman_velocities: true`
- Uses `vp_original`/`vs_original` otherwise
- Velocities converted from km/s to m/s

**Gap Interpolation:**
- Fills internal gaps using nearest-neighbor method
- Only fills points with sufficient neighbors
- Preserves boundary zeros

### Binary Format

- **Data Type:** 32-bit float (np.float32)
- **Byte Order:** Little-endian (native)
- **Array Order:** Fortran column-major
- **Access:** Direct stream (no record markers)
- **Units:** m/s (converted from km/s)

### Verification

The module performs automatic verification:
- Checks file size matches expected dimensions
- Reads back data to verify integrity
- Reports velocity ranges
- Confirms array matching

## Visualization

Creates quality control plots and animations.

### Station and Event Visualization

```yaml
visualization:
  station_event_vis:
    station_file: "${coordinate_transform.output_dir}/stations_cartesian.csv"
    event_file: "${coordinate_transform.output_dir}/events_cartesian.csv"
    output_image: "${data.output_base_dir}/visualization_station_events.pdf"
    view_from_direction: "southeast"  # south, east, west, north, southeast
    view_elevation: 20
```

Creates 3D scatter plot showing spatial distribution.

### Velocity Visualization

```yaml
velocity_vis:
  csv_file: "${coordinate_transform.output_dir}/velocity_cartesian.csv"
  bin_file: "${data.output_base_dir}/vp_init.bin"
  coord_system: "flattened"  # or "enu"
```

Generates comparison plots from CSV and binary files.

### Velocity Animation

```yaml
velocity_animation:
  bin_file: "${data.output_base_dir}/vp_init.bin"
  output_gif: "${data.output_base_dir}/velocity_animation.gif"
  rotation_frames: 120
  frame_duration: 0.05
```

Creates rotating 3D animation of velocity model.

## File Formats

### Input File Formats

**Station Coordinates (6 columns):**
```
network station_code elevation(m) latitude longitude depth_info
XX      STA1        1500.0       34.5     -108.0    0
```

**1D Velocity Model (3 columns):**
```
depth(km)  vp(km/s)  vs(km/s)
0.0        5.0       2.9
```

**3D Velocity Model (5 columns):**
```
lon      lat    depth(km)  vp(km/s)  vs(km/s)
-108.0   34.5   0.0        5.0       2.9
```

### Output CSV Columns

**Cartesian Coordinates:**
- Original: `lon`, `lat`, `depth`, `station`/`event_id`
- Grid: `X_grid_km`, `Y_grid_km`, `Z_grid_km`
- Indices: `i`, `j`, `k` (1-based)
- Chapman: `vp`, `vs`, `vp_original`, `vs_original`, `vp_flat`, `vs_flat`

**Travel Time CSV:**
- Identifiers: `shot_id`, `event_id`, `station`, `phase`
- Times: `observed_tt`, `weight`, `residual`
- Original coords: `event_longitude`, `station_latitude`, etc.
- Transformed: `event_x_transformed`, `station_z_transformed`, etc.
- Derived: `distance_m`, `origin_time_s`

## Troubleshooting

### Common Issues

**Import Error:**
```
ImportError: attempted relative import beyond top-level package
```
**Solution:** Run from parent directory: `python -m preprocess.main`

**Missing pyproj:**
```
ImportError: No module named 'pyproj'
```
**Solution:** `pip install pyproj` or use `transformation_method: "flattening"`

**Config Not Found:**
```
Cannot find primary config 'default.yaml'
```
**Solution:** Check file exists at `configs/preprocess/default.yaml`

**Grid Dimension Warning:**
```
Auto-detected dimensions: NX=45, NY=40, NZ=50
Common/expected dimensions: NX=41, NY=37, NZ=41
```
**Solution:** This is informational. The toolkit auto-detects correct dimensions.

**Negative Travel Times:**
```
Warning: Event X, station Y skipped (reason: negative travel time: -2.5s)
```
**Solution:** Check phase pick times and event origin times. May indicate data quality issues.

**Unrealistic Velocities:**
```
Warning: Event X, station Y (P-phase) skipped (reason: unrealistic velocity: 15000 m/s)
```
**Solution:** Adjust `p_velocity_range_m_s` or investigate data quality.

### Debugging Tips

1. **Enable Debug Logging:**
   ```yaml
   preprocessing:
     log_level: DEBUG
   ```

2. **Test Individual Components:**
   ```bash
   python -m preprocess.main single_function=coordinate_transform
   ```

3. **Check File Permissions:**
   Ensure write access to output directories

4. **Validate Input Data:**
   - Check file formats match specifications
   - Verify coordinate ranges are reasonable
   - Ensure no missing required columns

5. **Monitor Output:**
   - Check CSV files for expected columns
   - Verify binary file sizes
   - Review visualization plots

### Performance Considerations

- **Memory:** Large velocity models may require 4-8 GB RAM
- **Processing Time:** Coordinate transformation: seconds to minutes depending on grid size
- **Disk Space:** Binary files are typically 50-80% smaller than text files

## Examples

### Basic Workflow

```bash
# 1. Set up configuration
cp configs/preprocess/default.yaml my_config.yaml
# Edit my_config.yaml with your paths

# 2. Run preprocessing
python -m preprocess.main --config-name=my_config

# 3. Verify outputs
ls outputs/coordinate_transform/
ls outputs/geometry/
ls outputs/ground_truth_traveltimes/
```

### Chapman Workflow

```bash
python -m preprocess.main \
  coordinate_transform.use_chapman_flattening=true \
  velocity_binary.use_chapman_velocities=true
```

### Exact Transformation

```bash
python -m preprocess.main \
  coordinate_transform.transformation_method=exact \
  coordinate_transform.exact_transform.rotate_to_enu=true
```

### Custom Grid Spacing

```bash
python -m preprocess.main \
  coordinate_transform.grid_spacing_km=[10.0,10.0,1.0]
```

### Using Corrected Data

```bash
python -m preprocess.main \
  coordinate_transform.use_corrected_phase_pickings=true \
  geometry_traveltime.use_corrected_phase_pickings=true
```

### Filtering Events

```bash
# Create events_to_discard.txt with:
# Shot_ID    Original_Event_ID
# 1          event001
# 5          event005

python -m preprocess.main \
  geometry_traveltime.filter_events=true \
  geometry_traveltime.events_to_discard_file=/path/to/events_to_discard.txt
```

---

For additional help, open an issue in the repository or consult the main README.