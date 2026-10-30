"""
I/O utility functions for the LATTE tomography package.
These functions can be reused across preprocessing, postprocessing, and analysis modules.
"""

import os
import re
import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import json
import pickle
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Standard CSV file names
# --------------------------------------------------------------------------- #
# Default names of the per-pick tables, matching the published data package
# (analysis/geometry_and_travel_times/). Legacy names are only fallbacks for inputs.
GROUND_TRUTH_CSV_NAME = "geometry_ground_truth_traveltimes.csv"
SYNTHETIC_TT_CSV_NAME = "synthetic_traveltimes.csv"
SOURCE_RELOCATION_CSV_NAME = "source_relocation.csv"

_CSV_NAMES = {
    'ground_truth': (
        GROUND_TRUTH_CSV_NAME,
        ("ground_truth_traveltimes.csv",
         "ground_truth_travel_times_corrected.csv",
         "ground_truth_travel_times.csv"),
    ),
    'synthetic': (
        SYNTHETIC_TT_CSV_NAME,
        ("synthetic_travel_times_latte_output.csv",),
    ),
    # The source columns were once appended to the synthetic travel time table
    'source': (
        SOURCE_RELOCATION_CSV_NAME,
        ("source_parameters_latte_output.csv",
         "synthetic_travel_times_latte_output.csv"),
    ),
}

# Legacy glob patterns used when scanning iteration directories
_LEGACY_CSV_GLOBS = {
    SYNTHETIC_TT_CSV_NAME: ("synthetic_travel_times_latte_output.csv", "travel_time_residuals*.csv"),
    SOURCE_RELOCATION_CSV_NAME: ("source_parameters_latte_output.csv", "source_corrections*.csv"),
    GROUND_TRUTH_CSV_NAME: ("ground_truth_traveltimes.csv", "ground_truth_travel_times*.csv"),
}

# Velocity models: standard stem and legacy stems. The extension (.csv, .h5, .hdf5) is kept, and a
# trailing grid/method suffix (_coarse, _exact, _chapman) carries over to the standard name.
VELOCITY_INIT_STEM = "velocity_model_init"
VELOCITY_FINAL_STEM = "velocity_model_final"

_VELOCITY_STEMS = {
    'init': (
        VELOCITY_INIT_STEM,
        ("velocity_model_cartesian", "velocity_initial_cartesian",
         "velocity_cartesian_initial", "velocity_cartesian"),
    ),
    'final': (
        VELOCITY_FINAL_STEM,
        ("velocity_cartesian_latte_output", "velocity_cartesian_updated", "velocity_model_inverted"),
    ),
}
_VELOCITY_EXTENSIONS = ('.csv', '.h5', '.hdf5')
_VELOCITY_SUFFIX_RE = re.compile(r'^(?P<stem>.+?)(?P<suffix>(?:_exact|_chapman)?(?:_coarse)?)$')

_logger = logging.getLogger(__name__)


def _csv_role_from_name(name: str) -> Optional[str]:
    for role, (new_name, legacy_names) in _CSV_NAMES.items():
        if name == new_name:
            return role
    for role, (_, legacy_names) in _CSV_NAMES.items():
        if name in legacy_names:
            return role
    return None


def resolve_csv_path(filepath: Union[str, Path], role: Optional[str] = None) -> str:
    """
    Resolve the standard per-pick CSV (ground truth, synthetic travel times, source
    relocation) to a file that exists, preferring the standard name.

    If filepath exists it is returned as is. Otherwise the same directory is searched
    for the standard name and then the legacy names. A warning is logged whenever a
    legacy name is used. Paths with other names, or with no match, are returned
    unchanged so the caller reports its usual error.

    Args:
        filepath: Configured CSV path
        role: 'ground_truth', 'synthetic' or 'source'; inferred from the file name if omitted

    Returns:
        Path of the file to read
    """
    path = Path(filepath)
    role = role or _csv_role_from_name(path.name)
    if role not in _CSV_NAMES:
        return str(filepath)

    standard, legacy_names = _CSV_NAMES[role]
    if path.exists():
        if path.name in legacy_names:
            _logger.warning(f"Legacy CSV name '{path.name}' in use; standard name is '{standard}'")
        return str(filepath)

    for name in (standard, *legacy_names):
        candidate = path.parent / name
        if candidate.exists():
            if name != standard:
                _logger.warning(f"'{standard}' not found in {path.parent}; "
                                f"using legacy file '{name}' (rename it to '{standard}')")
            return str(candidate)
    return str(filepath)


def glob_csv(directory: Union[str, Path], pattern: str, recursive: bool = False) -> List[Path]:
    """
    Sorted glob of directory for pattern, with legacy fallback when the pattern is a
    standard CSV name. A warning is logged if only legacy names match.
    """
    directory = Path(directory)
    find = directory.rglob if recursive else directory.glob
    matches = sorted(find(pattern))
    if matches:
        return matches
    for legacy in _LEGACY_CSV_GLOBS.get(pattern, ()):
        matches = sorted(find(legacy))
        if matches:
            _logger.warning(f"No '{pattern}' in {directory}; using legacy file(s) matching "
                            f"'{legacy}' (rename to '{pattern}')")
            return matches
    return []


def resolve_velocity_path(filepath: Union[str, Path]) -> str:
    """
    Resolve a velocity model file (CSV or HDF5) to one that exists, preferring the standard
    stems 'velocity_model_init' and 'velocity_model_final'.

    Behaves like resolve_csv_path: an existing path is returned as is, otherwise the same
    directory is searched for the standard stem and then the legacy stems with the same
    extension and suffix. A warning is logged whenever a legacy name is used. Other names, or
    no match, return the path unchanged.
    """
    path = Path(filepath)
    if path.suffix.lower() not in _VELOCITY_EXTENSIONS:
        return str(filepath)
    match = _VELOCITY_SUFFIX_RE.match(path.stem)
    stem, suffix = match.group('stem'), match.group('suffix')
    role = next((r for r, (std, legacy) in _VELOCITY_STEMS.items() if stem == std or stem in legacy), None)
    if role is None:
        return str(filepath)

    standard, legacy_stems = _VELOCITY_STEMS[role]
    standard_name = f"{standard}{suffix}{path.suffix}"
    if path.exists():
        if stem != standard:
            _logger.warning(f"Legacy velocity model name '{path.name}' in use; standard name is '{standard_name}'")
        return str(filepath)

    for name_stem in (standard, *legacy_stems):
        candidate = path.parent / f"{name_stem}{suffix}{path.suffix}"
        if candidate.exists():
            if name_stem != standard:
                _logger.warning(f"'{standard_name}' not found in {path.parent}; "
                                f"using legacy file '{candidate.name}' (rename it to '{standard_name}')")
            return str(candidate)
    return str(filepath)


def check_file_exists(filepath: Union[str, Path]) -> None:
    """
    Check if a file exists and raise informative error if not.
    
    Args:
        filepath: Path to check
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")


def ensure_directory_exists(directory: Union[str, Path]) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path to create
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_file_extension(filepath: Union[str, Path]) -> str:
    """
    Get file extension in lowercase.
    
    Args:
        filepath: Path to file
        
    Returns:
        File extension (e.g., '.txt', '.csv')
    """
    return Path(filepath).suffix.lower()


def read_text_file(filepath: Union[str, Path], 
                  skip_comments: bool = True,
                  comment_char: str = '#') -> List[str]:
    """
    Read text file and return list of lines.
    
    Args:
        filepath: Path to text file
        skip_comments: Whether to skip comment lines
        comment_char: Character that indicates comments
        
    Returns:
        List of lines (stripped)
    """
    check_file_exists(filepath)
    
    lines = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if skip_comments and line.startswith(comment_char):
                continue
            lines.append(line)
    
    return lines


def write_text_file(filepath: Union[str, Path], 
                   lines: List[str],
                   header: Optional[str] = None) -> None:
    """
    Write lines to text file.
    
    Args:
        filepath: Output file path
        lines: List of lines to write
        header: Optional header to write first
    """
    ensure_directory_exists(Path(filepath).parent)
    
    with open(filepath, 'w') as f:
        if header:
            f.write(f"{header}\n")
        for line in lines:
            f.write(f"{line}\n")


def read_csv_file(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Read CSV file into DataFrame with common preprocessing.
    
    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        pandas DataFrame
    """
    filepath = resolve_velocity_path(resolve_csv_path(filepath))
    check_file_exists(filepath)
    
    # Default arguments
    default_kwargs = {
        'comment': '#',
        'skipinitialspace': True,
        'na_values': ['', 'nan', 'NaN', 'NULL', 'null']
    }
    default_kwargs.update(kwargs)
    
    return pd.read_csv(filepath, **default_kwargs)


def write_csv_file(df: pd.DataFrame, 
                  filepath: Union[str, Path],
                  **kwargs) -> None:
    """
    Write DataFrame to CSV file.
    
    Args:
        df: DataFrame to write
        filepath: Output file path
        **kwargs: Additional arguments for DataFrame.to_csv
    """
    ensure_directory_exists(Path(filepath).parent)
    
    # Default arguments
    default_kwargs = {
        'index': False,
        'float_format': '%.6f'
    }
    default_kwargs.update(kwargs)
    
    df.to_csv(filepath, **default_kwargs)


def save_json(data: Dict[Any, Any], 
             filepath: Union[str, Path],
             indent: int = 2) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        indent: JSON indentation
    """
    ensure_directory_exists(Path(filepath).parent)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: Union[str, Path]) -> Dict[Any, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary loaded from JSON
    """
    check_file_exists(filepath)
    
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        filepath: Output file path
    """
    ensure_directory_exists(Path(filepath).parent)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    check_file_exists(filepath)
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_numpy_array(array: np.ndarray, 
                    filepath: Union[str, Path],
                    compressed: bool = True) -> None:
    """
    Save numpy array to file.
    
    Args:
        array: Numpy array to save
        filepath: Output file path
        compressed: Whether to use compression
    """
    ensure_directory_exists(Path(filepath).parent)
    
    if compressed:
        np.savez_compressed(filepath, array=array)
    else:
        np.save(filepath, array)


def load_numpy_array(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load numpy array from file.
    
    Args:
        filepath: Path to numpy file
        
    Returns:
        Loaded numpy array
    """
    check_file_exists(filepath)
    
    if str(filepath).endswith('.npz'):
        with np.load(filepath) as data:
            return data['array']
    else:
        return np.load(filepath)


def get_file_size(filepath: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        filepath: Path to file
        
    Returns:
        File size in bytes
    """
    check_file_exists(filepath)
    return os.path.getsize(filepath)


def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive file information.
    
    Args:
        filepath: Path to file
        
    Returns:
        Dictionary with file information
    """
    check_file_exists(filepath)
    
    path = Path(filepath)
    stat = path.stat()
    
    return {
        'name': path.name,
        'stem': path.stem,
        'suffix': path.suffix,
        'size_bytes': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'modified_time': stat.st_mtime,
        'created_time': stat.st_ctime,
        'is_file': path.is_file(),
        'is_dir': path.is_dir(),
        'absolute_path': str(path.absolute())
    }


def list_files_by_extension(directory: Union[str, Path], 
                          extension: str,
                          recursive: bool = False) -> List[Path]:
    """
    List all files with specific extension in directory.
    
    Args:
        directory: Directory to search
        extension: File extension (with or without dot)
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    # Ensure extension starts with dot
    if not extension.startswith('.'):
        extension = f'.{extension}'
    
    pattern = f"**/*{extension}" if recursive else f"*{extension}"
    return list(directory.glob(pattern))


class FileLogger:
    """
    Simple file logger for tracking file operations.
    """
    
    def __init__(self, log_file: Union[str, Path]):
        """
        Initialize file logger.
        
        Args:
            log_file: Path to log file
        """
        self.log_file = Path(log_file)
        ensure_directory_exists(self.log_file.parent)
        
        # Setup logger
        self.logger = logging.getLogger(f"file_logger_{id(self)}")
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_operation(self, operation: str, filepath: Union[str, Path], 
                     success: bool = True, details: str = "") -> None:
        """
        Log file operation.
        
        Args:
            operation: Type of operation (read, write, delete, etc.)
            filepath: Path that was operated on
            success: Whether operation succeeded
            details: Additional details
        """
        status = "SUCCESS" if success else "FAILED"
        message = f"{operation.upper()} - {status} - {filepath}"
        if details:
            message += f" - {details}"
        
        if success:
            self.logger.info(message)
        else:
            self.logger.error(message)