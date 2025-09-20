"""
I/O utility functions for the LATTE tomography package.
These functions can be reused across preprocessing, postprocessing, and analysis modules.
"""

import os
import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import json
import pickle
import numpy as np
import pandas as pd


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