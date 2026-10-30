# Preprocessing Module Configuration

## Overview

The preprocessing module transforms raw seismic data into LATTE-compatible formats, including coordinate transformations, velocity model preparation, and geometry/travel time file generation.

## Configuration Files

- `default.yaml` - Default configuration with all available options
- Project-specific configs override defaults via Hydra

## Key Configuration Sections

### Origin Time Configuration

Controls event origin time handling throughout the workflow.

- **mode**: `fixed | unix_epoch`
  - `fixed`: Set all events to fixed value (seconds)
  - `unix_epoch`: Convert to Unix timestamps
- **fixed_value**: Time value when mode=`fixed`

### Coordinate Transformation

Converts geographic (lon/lat/depth) to local Cartesian coordinates.

- **transformation_method**: `exact | flattening`
  - `exact`: Geodetic transformation via pyproj
  - `flattening`: Earth-flattening approximation
- **max_depth_km**: Depth filtering threshold (null to disable)
- **exact_transform.rotate_to_enu**: Rotate to East-North-Up frame
- **exact_transform.align_to_geocentric_normal**: Geocentric alignment

### Initial 3D Model Generation

Creates initial 3D velocity model from 1D reference and 3D template.

- **enabled**: Generate initial model from 1D profile
- **model_1d_file**: 1D velocity profile input
- **template_3d_file**: 3D grid template
- **output_file**: Generated 3D model output

### Geometry and Travel Time Files

Generates LATTE input files for geometry and observed travel times.

- **phase_type**: `both | p | s`
  - `both`: Requires P and S picks; writes both binaries
  - `p`: P-wave only
  - `s`: S-wave only
- **source_binary_format**: `travel_time | fortran`
  - `travel_time`: Little-endian 32-bit float
  - `fortran`: Big-endian for Fortran compatibility
- **origin_time_precision**: `single | double`
  - `single`: 4-byte float (real*4)
  - `double`: 8-byte double (real*8)
- **use_corrected_phase_pickings**: Use relocated source parameters
- **generate_source_binaries**: Create source binary files (sx, sy, sz, st0)
- **apply_default_weights**: Override all weights with default value

### Velocity Binary Generation

Creates binary velocity files for LATTE input.

- **phase_type**: `both | p | s`
- **interpolate**: Fill internal grid gaps
- **create_final_binaries**: Generate binaries from inverted model

### Visualization

Generates 3D visualizations of stations, events, and velocity models.

- **view_from_direction**: `south | east | west | north | southeast`
- **coord_system**: `flattened | enu`
- **enabled**: Enable visualization generation

## Usage

```bash
python -m preprocess.main --config-name=your_config
```

## Output Structure

```
output/
├── stations_cartesian.csv
├── events_cartesian.csv
├── velocity_model_init.csv
├── model/
│   ├── vp_init.bin
│   ├── vs_init.bin
│   └── sx.bin, sy.bin, sz.bin, st0.bin
├── geometry/
│   └── geometry.txt
└── ground_truth_traveltimes/
    ├── P_traveltime.bin
    └── S_traveltime.bin
```

## Critical Notes

- **phase_type** must be consistent across preprocessing, postprocessing, and analysis
- **origin_time.mode** metadata is saved and auto-loaded by postprocessing/analysis
- **origin_time_precision** should match LATTE binary precision requirements
- **max_depth_km** filtering is applied after transformation
- Use **initial_velocity_file** from generated 3D model, not original input
- Set **use_corrected_phase_pickings=true** when using relocated sources
