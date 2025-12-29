"""
Unit tests for velocity_io module.

Tests cover:
- CSV and HDF5 reading/writing
- Format auto-detection
- Coarse subsampling with edge cases
- Parallel processing
- Metadata extraction
- Error handling
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

# Add parent directory to path for imports
import sys
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import directly to avoid package issues
import utils.velocity_io as velocity_io

read_velocity_file = velocity_io.read_velocity_file
write_velocity_file = velocity_io.write_velocity_file
subsample_to_coarse_grid = velocity_io.subsample_to_coarse_grid
get_velocity_file_metadata = velocity_io.get_velocity_file_metadata
_detect_format = velocity_io._detect_format
_validate_coarse_grid_config = velocity_io._validate_coarse_grid_config
VELOCITY_COLUMNS = velocity_io.VELOCITY_COLUMNS

# Dummy to satisfy old-style import structure
class DummyModule:
    def __init__(self):
        pass

if False:  # Never executed, just for type checking
    from utils.velocity_io import (
    read_velocity_file,
    write_velocity_file,
    subsample_to_coarse_grid,
    get_velocity_file_metadata,
    _detect_format,
    _validate_coarse_grid_config,
    VELOCITY_COLUMNS
)


class TestFormatDetection:
    """Test file format auto-detection."""

    def test_detect_csv_format(self):
        """Test CSV format detection."""
        assert _detect_format('test.csv') == 'csv'
        assert _detect_format('/path/to/file.csv') == 'csv'

    def test_detect_hdf5_format(self):
        """Test HDF5 format detection."""
        assert _detect_format('test.h5') == 'hdf5'
        assert _detect_format('test.hdf5') == 'hdf5'
        assert _detect_format('/path/to/file.HDF5') == 'hdf5'

    def test_invalid_extension_raises_error(self):
        """Test that invalid extension raises ValueError."""
        with pytest.raises(ValueError, match="Cannot detect format"):
            _detect_format('test.txt')


class TestCoarseGridValidation:
    """Test coarse grid configuration validation."""

    def test_valid_integral_multiples(self):
        """Test validation with valid integral multiples."""
        config = {'dx_coarse': 2.0, 'dy_coarse': 3.0, 'dz_coarse': 4.0}
        stride_x, stride_y, stride_z = _validate_coarse_grid_config(
            config, fine_dx=1.0, fine_dy=1.0, fine_dz=1.0
        )
        assert stride_x == 2
        assert stride_y == 3
        assert stride_z == 4

    def test_non_integral_multiple_raises_error(self):
        """Test that non-integral multiples raise ValueError."""
        config = {'dx_coarse': 1.5, 'dy_coarse': 2.0, 'dz_coarse': 2.0}
        with pytest.raises(ValueError, match="integral multiple"):
            _validate_coarse_grid_config(
                config, fine_dx=1.0, fine_dy=1.0, fine_dz=1.0
            )

    def test_fine_spacing_of_2km(self):
        """Test validation with fine spacing of 2 km."""
        config = {'dx_coarse': 4.0, 'dy_coarse': 6.0, 'dz_coarse': 2.0}
        stride_x, stride_y, stride_z = _validate_coarse_grid_config(
            config, fine_dx=2.0, fine_dy=2.0, fine_dz=2.0
        )
        assert stride_x == 2
        assert stride_y == 3
        assert stride_z == 1


@pytest.fixture
def sample_velocity_df():
    """
    Create sample velocity DataFrame for testing.

    Creates a 5x5x5 grid with 1 km spacing.
    """
    nx, ny, nz = 5, 5, 5
    dx, dy, dz = 1.0, 1.0, 1.0

    data = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                data.append({
                    'X_grid_km': i * dx,
                    'Y_grid_km': j * dy,
                    'Z_grid_km': k * dz,
                    'x': i * dx,
                    'y': j * dy,
                    'z': k * dz,
                    'i_raw': float(i),
                    'j_raw': float(j),
                    'k_raw': float(k),
                    'i': i + 1,
                    'j': j + 1,
                    'k': k + 1,
                    'vp': 5.0 + k * 0.1,
                    'vs': 3.0 + k * 0.05,
                    'lon': -108.0 + i * 0.01,
                    'lat': 34.5 + j * 0.01,
                    'depth': k * dz,
                })

    return pd.DataFrame(data)


@pytest.fixture
def large_velocity_df():
    """
    Create larger velocity DataFrame for parallel processing tests.

    Creates a 10x10x10 grid.
    """
    nx, ny, nz = 10, 10, 10
    dx, dy, dz = 0.5, 0.5, 0.5

    data = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                data.append({
                    'X_grid_km': i * dx,
                    'Y_grid_km': j * dy,
                    'Z_grid_km': k * dz,
                    'x': i * dx,
                    'y': j * dy,
                    'z': k * dz,
                    'i_raw': float(i),
                    'j_raw': float(j),
                    'k_raw': float(k),
                    'i': i + 1,
                    'j': j + 1,
                    'k': k + 1,
                    'vp': 6.0 + k * 0.15,
                    'vs': 3.5 + k * 0.08,
                    'lon': -105.0 + i * 0.005,
                    'lat': 33.0 + j * 0.005,
                    'depth': k * dz,
                })

    return pd.DataFrame(data)


class TestCSVReadWrite:
    """Test CSV reading and writing."""

    def test_csv_write_and_read(self, sample_velocity_df):
        """Test basic CSV write and read."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            # Write CSV
            write_velocity_file(sample_velocity_df, filepath, show_progress=False)

            # Read CSV
            df_read = read_velocity_file(filepath, show_progress=False)

            # Verify data
            assert len(df_read) == len(sample_velocity_df)
            assert list(df_read.columns) == list(sample_velocity_df.columns)
            pd.testing.assert_frame_equal(df_read, sample_velocity_df, rtol=1e-5)

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_csv_auto_format_detection(self, sample_velocity_df):
        """Test automatic format detection for CSV."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            write_velocity_file(sample_velocity_df, filepath, format='auto', show_progress=False)
            df_read = read_velocity_file(filepath, format='auto', show_progress=False)

            assert len(df_read) == len(sample_velocity_df)

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestHDF5ReadWrite:
    """Test HDF5 reading and writing."""

    def test_hdf5_write_and_read(self, sample_velocity_df):
        """Test basic HDF5 write and read."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filepath = f.name

        try:
            # Prepare metadata
            metadata = {
                'grid_dimensions': {'nx': 5, 'ny': 5, 'nz': 5},
                'grid_spacing': {'dx': 1.0, 'dy': 1.0, 'dz': 1.0},
                'origin_lon': -108.0,
                'origin_lat': 34.5,
            }

            # Write HDF5
            write_velocity_file(
                sample_velocity_df, filepath,
                format='hdf5',
                metadata=metadata,
                show_progress=False
            )

            # Read HDF5
            df_read = read_velocity_file(filepath, format='hdf5', show_progress=False)

            # Verify data
            assert len(df_read) == len(sample_velocity_df)
            pd.testing.assert_frame_equal(df_read, sample_velocity_df, rtol=1e-5)

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_hdf5_with_compression(self, sample_velocity_df):
        """Test HDF5 writing with different compression options."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filepath = f.name

        try:
            # Write with gzip compression
            write_velocity_file(
                sample_velocity_df, filepath,
                format='hdf5',
                compression='gzip',
                compression_opts=4,
                show_progress=False
            )

            df_read = read_velocity_file(filepath, show_progress=False)
            assert len(df_read) == len(sample_velocity_df)

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_hdf5_auto_format_detection(self, sample_velocity_df):
        """Test automatic format detection for HDF5."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filepath = f.name

        try:
            write_velocity_file(sample_velocity_df, filepath, format='auto', show_progress=False)
            df_read = read_velocity_file(filepath, format='auto', show_progress=False)

            assert len(df_read) == len(sample_velocity_df)

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestCoarseSubsampling:
    """Test coarse grid subsampling functionality."""

    def test_basic_subsampling(self, sample_velocity_df):
        """Test basic coarse grid subsampling."""
        df_coarse = subsample_to_coarse_grid(
            sample_velocity_df,
            dx_coarse=2.0, dy_coarse=2.0, dz_coarse=2.0,
            fine_dx=1.0, fine_dy=1.0, fine_dz=1.0,
            n_jobs=1,
            show_progress=False
        )

        # With stride=2 on 5x5x5 grid: indices 0,2,4 → 3 points per dimension
        # Expected: 3x3x3 = 27 points
        assert len(df_coarse) == 27

        # Verify coarse points are subset of fine
        for _, row in df_coarse.iterrows():
            match = sample_velocity_df[
                (sample_velocity_df['X_grid_km'] == row['X_grid_km']) &
                (sample_velocity_df['Y_grid_km'] == row['Y_grid_km']) &
                (sample_velocity_df['Z_grid_km'] == row['Z_grid_km'])
            ]
            assert len(match) == 1, "Coarse point not found in fine grid"

    def test_subsampling_with_parallel(self, large_velocity_df):
        """Test coarse subsampling with parallel processing."""
        # Single-threaded
        df_coarse_serial = subsample_to_coarse_grid(
            large_velocity_df,
            dx_coarse=1.0, dy_coarse=1.0, dz_coarse=1.0,
            fine_dx=0.5, fine_dy=0.5, fine_dz=0.5,
            n_jobs=1,
            show_progress=False
        )

        # Multi-threaded
        df_coarse_parallel = subsample_to_coarse_grid(
            large_velocity_df,
            dx_coarse=1.0, dy_coarse=1.0, dz_coarse=1.0,
            fine_dx=0.5, fine_dy=0.5, fine_dz=0.5,
            n_jobs=2,
            show_progress=False
        )

        # Results should be identical
        assert len(df_coarse_serial) == len(df_coarse_parallel)
        pd.testing.assert_frame_equal(
            df_coarse_serial.sort_values(['X_grid_km', 'Y_grid_km', 'Z_grid_km']).reset_index(drop=True),
            df_coarse_parallel.sort_values(['X_grid_km', 'Y_grid_km', 'Z_grid_km']).reset_index(drop=True),
            rtol=1e-5
        )

    def test_subsampling_edge_case_grid_boundary(self, sample_velocity_df):
        """Test that grid boundaries are always included."""
        df_coarse = subsample_to_coarse_grid(
            sample_velocity_df,
            dx_coarse=3.0, dy_coarse=3.0, dz_coarse=3.0,
            fine_dx=1.0, fine_dy=1.0, fine_dz=1.0,
            n_jobs=1,
            show_progress=False
        )

        # Check that maximum coordinates are present
        assert df_coarse['X_grid_km'].max() == sample_velocity_df['X_grid_km'].max()
        assert df_coarse['Y_grid_km'].max() == sample_velocity_df['Y_grid_km'].max()
        assert df_coarse['Z_grid_km'].max() == sample_velocity_df['Z_grid_km'].max()

        # Check that minimum coordinates are present
        assert df_coarse['X_grid_km'].min() == sample_velocity_df['X_grid_km'].min()
        assert df_coarse['Y_grid_km'].min() == sample_velocity_df['Y_grid_km'].min()
        assert df_coarse['Z_grid_km'].min() == sample_velocity_df['Z_grid_km'].min()

    def test_subsampling_non_integral_raises_error(self, sample_velocity_df):
        """Test that non-integral coarse spacing raises error."""
        with pytest.raises(ValueError, match="integral multiple"):
            subsample_to_coarse_grid(
                sample_velocity_df,
                dx_coarse=1.5, dy_coarse=2.0, dz_coarse=2.0,
                fine_dx=1.0, fine_dy=1.0, fine_dz=1.0,
                n_jobs=1,
                show_progress=False
            )


class TestHDF5WithCoarseDataset:
    """Test HDF5 writing with coarse dataset generation."""

    def test_write_with_coarse_dataset(self, sample_velocity_df):
        """Test writing HDF5 with automatic coarse dataset generation."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                'grid_dimensions': {'nx': 5, 'ny': 5, 'nz': 5},
                'grid_spacing': {'dx': 1.0, 'dy': 1.0, 'dz': 1.0},
            }

            coarse_config = {
                'dx_coarse': 2.0,
                'dy_coarse': 2.0,
                'dz_coarse': 2.0,
            }

            # Write with coarse dataset
            write_velocity_file(
                sample_velocity_df, filepath,
                format='hdf5',
                coarse_grid_config=coarse_config,
                metadata=metadata,
                show_progress=False,
                n_jobs=1
            )

            # Read fine dataset
            df_fine = read_velocity_file(filepath, dataset='velocity', show_progress=False)
            assert len(df_fine) == 125  # 5x5x5

            # Read coarse dataset
            df_coarse = read_velocity_file(filepath, dataset='velocity_coarse', show_progress=False)
            assert len(df_coarse) == 27  # 3x3x3

            # Verify coarse is subset of fine
            for _, row in df_coarse.iterrows():
                match = df_fine[
                    (df_fine['X_grid_km'] == row['X_grid_km']) &
                    (df_fine['Y_grid_km'] == row['Y_grid_km']) &
                    (df_fine['Z_grid_km'] == row['Z_grid_km'])
                ]
                assert len(match) == 1

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_read_nonexistent_dataset_raises_error(self, sample_velocity_df):
        """Test that reading non-existent dataset raises KeyError."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filepath = f.name

        try:
            # Write without coarse dataset
            write_velocity_file(
                sample_velocity_df, filepath,
                format='hdf5',
                show_progress=False
            )

            # Try to read non-existent coarse dataset
            with pytest.raises(KeyError, match="not found"):
                read_velocity_file(filepath, dataset='velocity_coarse', show_progress=False)

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestMetadataExtraction:
    """Test metadata extraction from velocity files."""

    def test_csv_metadata_extraction(self, sample_velocity_df):
        """Test metadata extraction from CSV file."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            write_velocity_file(sample_velocity_df, filepath, show_progress=False)

            metadata = get_velocity_file_metadata(filepath)

            assert metadata['format'] == 'csv'
            assert metadata['num_points'] == 125
            assert metadata['grid_dimensions']['nx'] == 5
            assert metadata['grid_dimensions']['ny'] == 5
            assert metadata['grid_dimensions']['nz'] == 5
            assert metadata['file_size_mb'] > 0

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_hdf5_metadata_extraction(self, sample_velocity_df):
        """Test metadata extraction from HDF5 file."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filepath = f.name

        try:
            metadata_in = {
                'grid_dimensions': {'nx': 5, 'ny': 5, 'nz': 5},
                'grid_spacing': {'dx': 1.0, 'dy': 1.0, 'dz': 1.0},
            }

            write_velocity_file(
                sample_velocity_df, filepath,
                format='hdf5',
                metadata=metadata_in,
                show_progress=False
            )

            metadata_out = get_velocity_file_metadata(filepath)

            assert metadata_out['format'] == 'hdf5'
            assert metadata_out['num_points'] == 125
            assert 'velocity' in metadata_out['datasets']
            assert metadata_out['has_coarse_dataset'] == False
            assert metadata_out['file_size_mb'] > 0

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_hdf5_with_coarse_metadata(self, sample_velocity_df):
        """Test metadata extraction from HDF5 with coarse dataset."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filepath = f.name

        try:
            metadata_in = {
                'grid_dimensions': {'nx': 5, 'ny': 5, 'nz': 5},
                'grid_spacing': {'dx': 1.0, 'dy': 1.0, 'dz': 1.0},
            }

            coarse_config = {'dx_coarse': 2.0, 'dy_coarse': 2.0, 'dz_coarse': 2.0}

            write_velocity_file(
                sample_velocity_df, filepath,
                format='hdf5',
                coarse_grid_config=coarse_config,
                metadata=metadata_in,
                show_progress=False
            )

            metadata_out = get_velocity_file_metadata(filepath)

            assert metadata_out['has_coarse_dataset'] == True
            assert 'velocity_coarse' in metadata_out['datasets']

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestFileNotFound:
    """Test error handling for missing files."""

    def test_read_nonexistent_file(self):
        """Test that reading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_velocity_file('/nonexistent/path/to/file.csv')

    def test_metadata_nonexistent_file(self):
        """Test that metadata extraction on non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            get_velocity_file_metadata('/nonexistent/path/to/file.h5')


class TestRoundTrip:
    """Test round-trip conversion between formats."""

    def test_csv_to_hdf5_roundtrip(self, sample_velocity_df):
        """Test converting CSV to HDF5 and back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            h5_path = Path(tmpdir) / "test.h5"

            # Write CSV
            write_velocity_file(sample_velocity_df, csv_path, show_progress=False)

            # Read CSV and write HDF5
            df_csv = read_velocity_file(csv_path, show_progress=False)
            write_velocity_file(df_csv, h5_path, show_progress=False)

            # Read HDF5
            df_h5 = read_velocity_file(h5_path, show_progress=False)

            # Verify
            pd.testing.assert_frame_equal(df_csv, df_h5, rtol=1e-5)
            pd.testing.assert_frame_equal(sample_velocity_df, df_h5, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
