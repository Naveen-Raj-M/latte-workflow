# Postprocessing Module Configuration

## Overview

The postprocessing module converts LATTE binary outputs to CSV format for analysis, including updated velocity models, synthetic travel times, and corrected source parameters.

## Configuration Files

- `default.yaml` - Default configuration with all available options
- Project-specific configs (e.g., `tloc.yaml`) override defaults

## Key Configuration Sections

### General Settings

- **phase_type**: `both | p | s` (must match preprocessing)
- **log_level**: `DEBUG | INFO | WARNING | ERROR | CRITICAL`
- **strict_mode**: Stop on first error (true) or continue (false)
- **max_depth_km**: Depth filtering threshold (null to disable)

### Binary Precision

- **precision.default**: `single | double` - Default for most binaries (typically `single`)
- **precision.st0**: `single | double` - Origin time binary precision

**Note**: Must match preprocessing `origin_time_precision` setting.

### Velocity Processing

Converts LATTE velocity binaries to CSV format.

- **interpolation_mode**: `sample_at_coarse | expand_to_fine`
  - `sample_at_coarse`: Sample fine grid at coarse points (output = coarse size)
  - `expand_to_fine`: Full fine grid output (output = fine size)
- **use_chapman_inverse**: Apply inverse Chapman transformation
- **parameters_to_process**: `[vp, vs]` - Process gradients/search directions
  - Requires corresponding `grad_*.bin` and `srch_*.bin` files

#### Grid Specification

- **fine_grid**: LATTE internal grid (from binary files)
  - `nx, ny, nz`: Grid dimensions
  - `dx, dy, dz`: Grid spacing (km)
  - `origin_x, origin_y, origin_z`: Grid origin (km)
- **coarse_grid**: Auto-detected from CSV if not specified

### Travel Time Processing

Converts synthetic travel time binaries and merges with ground truth.

- **source_receiver_exchange**: Enable reciprocity-based inversion mode
  - Requires `exchange_mapping_file` and `exchange_geometry_file`
- **source_correction**: Apply corrected origin times from `updated_st0.bin`
- **removed_shots**: List of 1-based shot IDs filtered by LATTE during geometry loading
  - Required for correct mapping in source-receiver exchange mode

### Source Processing

Converts updated source parameter binaries to CSV.

- **grid_dimensions**: Number of sources (e.g., `[4495]`)
- **parameters_to_process**: `[sx, sy, sz, st0]` - Process gradients/search directions
  - Requires corresponding `grad_*.bin` and `srch_*.bin` files

## Usage

```bash
python -m postprocess.main --config-name=your_config
```

## Output Structure

```
analysis_output/
├── velocity_cartesian_latte_output.csv
├── synthetic_travel_times_latte_output.csv
└── source_parameters_latte_output.csv
```

### CSV Column Structure

**Velocity CSV**:
- Coordinates: `X_grid_km`, `Y_grid_km`, `Z_grid_km`
- Velocities: `Vp_km_s`, `Vs_km_s`
- Optional: `grad_vp`, `grad_vs`, `srch_vp`, `srch_vs`

**Travel Time CSV**:
- Merged ground truth and synthetic travel times
- Optional: Corrected origin times if `source_correction=true`

**Source Parameters CSV**:
- Source coordinates: `sx_km`, `sy_km`, `sz_km`
- Origin time: `st0_s`
- Optional: Gradients and search directions for each parameter

## Critical Notes

- **phase_type** must match preprocessing configuration
- **precision.st0** must match preprocessing `origin_time_precision`
- For **source_receiver_exchange**, specify `removed_shots` if applicable
- **interpolation_mode** choice affects output CSV size and memory usage
- Gradient/search direction processing is optional; set `parameters_to_process=[]` to skip
- **max_depth_km** filtering should match preprocessing value
