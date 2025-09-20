# LATTE Preprocessing

The LATTE preprocessing module handles all data preparation tasks required for seismic tomography inversion. It provides a comprehensive suite of tools for coordinate transformation, velocity model creation, geometry file generation, and data visualization.

## Overview

The preprocessing pipeline consists of five main components that can be run individually or as a complete workflow:

1. **Coordinate Transformation** - Convert geodetic coordinates to Cartesian grid coordinates
2. **Initial 3D Model Creation** - Generate 3D velocity models from 1D reference models
3. **Geometry & Travel Time Processing** - Create geometry files and travel time data from phase picks
4. **Velocity Binary Generation** - Convert velocity models to binary format for inversion
5. **Visualization** - Generate plots and animations for quality control

## Quick Start

### Complete Pipeline

Run the entire preprocessing workflow:

```bash
cd /path/to/latte-workflow
python -m preprocess.main --config-path=configs --config-name=preprocess/default
```

### Individual Functions

Run specific preprocessing steps:

```bash
# Coordinate transformation only
python -m preprocess.main single_function=coordinate_transform

# Initial 3D model creation only
python -m preprocess.main single_function=initial_3d_model

# Geometry and travel time processing only
python -m preprocess.main single_function=geometry_traveltime

# Velocity binary creation only
python -m preprocess.main single_function=velocity_binary

# Visualization only
python -m preprocess.main single_function=visualization
```

## Configuration

### Basic Configuration

All preprocessing is controlled through a YAML configuration file. Create your configuration by copying and modifying the default:

```yaml
# configs/preprocess/default.yaml
data:
  base_dir: "/path/to/your/data"
  output_base_dir: "./outputs"

coordinate_transform:
  enabled: true
  transformation_method: "flattening"  # or "exact"
  origin_lon: -108.0
  origin_lat: 34.5
  origin_depth_km: -4.0
```

### Custom Configuration

Override settings for your specific dataset:

```bash
# Use custom data directory
python -m preprocess.main data.base_dir=/path/to/my/data

# Use exact coordinate transformation
python -m preprocess.main coordinate_transform.transformation_method=exact

# Use corrected phase pickings
python -m preprocess.main geometry_traveltime.use_corrected_phase_pickings=true
```

## Detailed Component Guide

### 1. Coordinate Transformation

Converts geographic coordinates (longitude, latitude, depth) to local Cartesian coordinates suitable for tomographic inversion.

#### Transformation Methods

**Flattening Transformation** (Default)
- Simple geometric projection onto a flat Earth
- Optional Chapman (1973) Earth-flattening for seismic waves
- Fast and suitable for regional studies

**Exact Transformation**
- ECEF (Earth-Centered Earth-Fixed) to local ENU (East-North-Up)
- Accounts for Earth's curvature precisely
- Requires `pyproj` library

#### Input Files
- `stacoords.txt` - Station coordinates (6 columns)
- `phasePickings.txt` - Event data (9 columns) or `sphfdloc.dat` (corrected events)
- `vmodel-PB3D.txt` - 3D velocity model template (5 columns)

#### Output Files
- `stations_cartesian.csv` - Station coordinates in Cartesian grid
- `events_cartesian.csv` - Event coordinates in Cartesian grid  
- `velocity_cartesian.csv` - Velocity model in Cartesian grid

#### Configuration Options

```yaml
coordinate_transform:
  transformation_method: "flattening"  # "exact" or "flattening"
  use_chapman_flattening: false        # Apply Chapman transformation
  use_corrected_phase_pickings: false  # Use corrected event data
  origin_lon: -108.0                   # Origin longitude
  origin_lat: 34.5                     # Origin latitude
  origin_depth_km: -4.0                # Origin depth (km)
  grid_spacing_km: [20.0, 20.0, 2.0]  # Grid spacing [dx, dy, dz]
```

### 2. Initial 3D Model Creation

Creates a 3D initial velocity model by mapping velocities from a 1D reference model onto a 3D spatial grid.

#### Input Files
- `vmodel-PB1D.txt` - 1D velocity model (depth, vp, vs)
- `vmodel-PB3D.txt` - 3D template grid (lon, lat, depth, vp, vs)

#### Output Files
- `initial_3d_model_generated.txt` - 3D initial velocity model

#### Velocity Mapping Logic
- Finds the deepest 1D layer at or above each 3D grid point depth
- Uses first layer velocities for shallow extrapolation
- Uses deepest layer velocities for deep extrapolation

#### Configuration Options

```yaml
initial_3d_model:
  model_1d_file: "${data.base_dir}/vmodel-PB1D.txt"
  template_3d_file: "${data.base_dir}/vmodel-PB3D.txt"
  output_file: "${data.output_base_dir}/initial_3d_model_generated.txt"
  precision:
    longitude: 6    # Decimal places for longitude
    latitude: 7     # Decimal places for latitude
    depth: 8        # Decimal places for depth
    vp: 8          # Decimal places for Vp
    vs: 8          # Decimal places for Vs
  field_width: 11   # Field width for formatting
```

### 3. Geometry & Travel Time Processing

Processes seismic phase picks to create geometry files and travel time data required for tomographic inversion.

#### Input Files
- `phasePickings.txt` or `sphfdloc.dat` - Phase pick data
- `stations_cartesian.csv` - Station coordinates from coordinate transformation
- `events_cartesian.csv` - Event coordinates from coordinate transformation

#### Output Files
- `geometry/shot_N_geometry.txt` - Geometry files for each event
- `geometry/geometry.txt` - Index of geometry files
- `geometry/shot_to_source_id_map.txt` - Event ID mapping
- `ground_truth_traveltimes/shot_N_traveltime_p.bin` - P-wave travel times
- `ground_truth_traveltimes/shot_N_traveltime_s.bin` - S-wave travel times
- `model/sx.bin`, `sy.bin`, `sz.bin`, `st0.bin` - Source coordinates and origin times

#### Quality Control Filters
- Velocity range filtering for realistic apparent velocities
- Station/event filtering based on coordinate availability
- Travel time validation (removes negative times)
- Requires both P and S picks for each station-event pair

#### Configuration Options

```yaml
geometry_traveltime:
  use_corrected_phase_pickings: false
  generate_source_binaries: true
  zero_origin_time: true
  source_binary_format: "travel_time"  # "travel_time" or "fortran"
  p_velocity_range_m_s: [500, 10000]   # P-wave velocity filter (m/s)
  s_velocity_range_m_s: [500, 6000]    # S-wave velocity filter (m/s)
  filter_events: false                 # Enable event filtering
  filter_stations: false               # Enable station filtering
```

### 4. Velocity Binary Generation

Converts velocity models from CSV format to binary format required by the tomographic inversion code.

#### Input Files
- `velocity_cartesian.csv` - Velocity model from coordinate transformation

#### Output Files
- `vp_init.bin` - P-wave velocity model (binary)
- `vs_init.bin` - S-wave velocity model (binary)

#### Processing Features
- Auto-detection of grid dimensions from CSV data
- Chapman velocity handling (uses flattened or original velocities)
- Internal gap interpolation using nearest-neighbor method
- Fortran-compatible binary format (stream access, column-major order)

#### Configuration Options

```yaml
velocity_binary:
  use_chapman_velocities: true    # Use Chapman-transformed velocities
  interpolate: true               # Fill internal gaps
  neighbor_threshold: 5           # Minimum neighbors for interpolation
```

### 5. Visualization

Creates quality control plots and animations for visual inspection of processed data.

#### Station and Event Visualization
- 3D scatter plots of station and event locations
- Configurable viewing angles and perspectives

#### Velocity Model Visualization
- Comparison plots from CSV and binary data
- 3D scatter plots with velocity color-coding

#### Animated Visualization
- Rotating 3D animations of velocity models
- GIF output for presentations

#### Configuration Options

```yaml
visualization:
  station_event_vis:
    view_from_direction: "southeast"  # Viewing direction
    view_elevation: 20                # Elevation angle
  velocity_vis:
    coord_system: "flattened"         # Coordinate system for plots
  velocity_animation:
    rotation_frames: 120              # Number of animation frames
    frame_duration: 0.05              # Frame duration (seconds)
```

## File Formats

### Input File Formats

**Station Coordinates (`stacoords.txt`)**
```
# 6 columns: network station_code elevation latitude longitude depth_info
XX STA1 1500.0 34.5 -108.0 0
```

**Phase Picks (`phasePickings.txt`)**
```
# Event header (9 columns): year julian_day hour minute second latitude longitude depth event_id
2023 100 10 30 45.123 34.5 -108.0 5.0 event001
# Pick lines (7+ columns): station ? julian_day hour minute second phase [weight]
STA1 ? 100 10 31 2.456 P 1.0
```

**1D Velocity Model (`vmodel-PB1D.txt`)**
```
# 3 columns: depth(km) vp(km/s) vs(km/s)
0.0 5.0 2.9
1.0 5.5 3.2
```

### Output File Formats

**Cartesian Coordinates CSV**
- Contains original coordinates plus Cartesian grid coordinates
- Includes grid indices (i, j, k) for Fortran compatibility
- For Chapman transformation, includes both original and flattened velocities

**Binary Velocity Files**
- 32-bit float, Fortran column-major order
- Direct stream access (no record markers)
- Units: m/s (converted from km/s in CSV)

## Error Handling

The preprocessing pipeline includes comprehensive error handling:

- **File Validation**: Checks for file existence and format
- **Coordinate Validation**: Verifies coordinate ranges and projections
- **Velocity Validation**: Ensures Vp > Vs and positive values
- **Travel Time Validation**: Removes unrealistic apparent velocities
- **Grid Validation**: Checks for consistent grid dimensions

## Performance Considerations

- **Memory Usage**: Large velocity models may require significant RAM
- **Processing Time**: Coordinate transformation scales with number of points
- **Disk Space**: Binary files are typically smaller than text equivalents
- **Parallel Processing**: Individual functions can be run in parallel for different datasets

## Troubleshooting

### Common Issues

**Import Errors**
```bash
ImportError: attempted relative import beyond top-level package
```
Solution: Run from the parent directory containing the package

**Config File Not Found**
```bash
Cannot find primary config 'default.yaml'
```
Solution: Check config file path and name (should be `configs/preprocess/default.yaml`)

**Missing Dependencies**
```bash
ImportError: No module named 'pyproj'
```
Solution: Install required dependencies: `pip install pyproj imageio`

**Grid Dimension Mismatches**
Solution: Enable auto-detection or verify CSV grid indices are consistent

### Debugging Tips

1. **Enable Debug Logging**: Set `preprocessing.log_level: DEBUG`
2. **Run Individual Functions**: Test each component separately
3. **Check File Permissions**: Ensure write access to output directories
4. **Validate Input Data**: Use built-in validation functions
5. **Monitor Memory Usage**: For large datasets, consider processing in chunks

## Migration from Original Scripts

If migrating from standalone scripts, the functional equivalents are:

| Original Script | LATTE Function |
|----------------|----------------|
| `spherical_to_cartesian_exact.py` | `coordinate_transform` (exact method) |
| `spherical_to_cartesian_flattening.py` | `coordinate_transform` (flattening method) |
| `create_3d_init_vel_model.py` | `initial_3d_model` |
| `create_geometry_and_traveltime_files.py` | `geometry_traveltime` |
| `create_velocity_bin.py` | `velocity_binary` |
| `visualize_*.py` | `visualization` |

The LATTE functions preserve all original logic while adding configuration management, error handling, and integration capabilities.

## Examples

### Example 1: Basic Workflow

```bash
# 1. Set up configuration
cp configs/preprocess/default.yaml my_config.yaml
# Edit my_config.yaml with your paths

# 2. Run complete pipeline
python -m preprocess.main --config-name=my_config

# 3. Check outputs
ls outputs/
```

### Example 2: Chapman Workflow

```bash
# Enable Chapman flattening for velocity transformation
python -m preprocess.main \
  coordinate_transform.use_chapman_flattening=true \
  velocity_binary.use_chapman_velocities=true
```

### Example 3: Custom Origin

```bash
# Set custom coordinate origin
python -m preprocess.main \
  coordinate_transform.origin_lon=-110.0 \
  coordinate_transform.origin_lat=35.0 \
  coordinate_transform.origin_depth_km=-2.0
```

### Example 4: Corrected Phase Picks

```bash
# Use corrected event locations and phase picks
python -m preprocess.main \
  coordinate_transform.use_corrected_phase_pickings=true \
  geometry_traveltime.use_corrected_phase_pickings=true
```

This documentation provides a complete reference for using the LATTE preprocessing module effectively and efficiently for seismic tomography data preparation.