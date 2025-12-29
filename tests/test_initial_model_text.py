#!/usr/bin/env python3
"""
Enhanced file comparison script with support for:
- Text files with numerical precision handling
- Binary files (numpy arrays, raw bytes)
- CSV files with column-specific comparison
- Single files or directory comparison
"""

import re
import sys
import os
import csv
import struct
from typing import Tuple, List, Optional, Set, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass
import numpy as np


# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PATHS AS NEEDED
# ============================================================================

# Define your comparison pairs here
COMPARISON_CONFIGS = [
    {
        "name": "Model Output Comparison",
        "path1": "/home/rajmanoharann/dino_latte_comparison/test/flattened_transformation/input_files_corrected/geometry",
        "path2": "/home/rajmanoharann/dino_latte_comparison/flattened_transformation/input_files/geometry_corrected",
        "file_patterns": ["*.bin", "*.dat", "*.txt", "*.csv"],
        "tolerance": 1e-6,
        "relative_tolerance": 1e-6,
        "ignore_whitespace": True,
        "recursive": True,
        "verbose": False,
        "compare_type": "directory",  # "file" or "directory"
        
        # Binary file settings
        "binary_formats": {
            "*.bin": {"dtype": "float32", "endian": "little"},  # numpy float32 little-endian
            "*.dat": {"dtype": "float32", "endian": "big"},     # numpy float32 big-endian
            "*.model": {"format": "raw_bytes"},                 # raw bytes comparison
        },
        
        # CSV file settings
        "csv_settings": {
            "columns_to_compare": None,  # None = compare all, or ["col1", "col2", ...]
            "ignore_extra_columns": True,  # Handle extra columns gracefully
            "case_sensitive_columns": False,  # Column name matching
        }
    },
    # {
    #     "name": "Single Model File",
    #     "path1": "/path/to/model1.bin",
    #     "path2": "/path/to/model2.bin",
    #     "tolerance": 1e-8,
    #     "relative_tolerance": 1e-8,
    #     "compare_type": "file",
    #     "verbose": True,
        
    #     "binary_formats": {
    #         "*.bin": {"dtype": "float32", "endian": "little"},
    #     }
    # },
    # {
    #     "name": "CSV Data Comparison",
    #     "path1": "/path/to/data1.csv", 
    #     "path2": "/path/to/data2.csv",
    #     "tolerance": 1e-9,
    #     "relative_tolerance": 1e-9,
    #     "compare_type": "file",
    #     "verbose": True,
        
    #     "csv_settings": {
    #         "columns_to_compare": ["value1", "value2", "result"],  # Only compare these columns
    #         "ignore_extra_columns": True,
    #         "case_sensitive_columns": False,
    #     }
    # },
]

# Global settings
SHOW_SUMMARY = True
QUIET_MODE = False

# ============================================================================
# END CONFIGURATION SECTION
# ============================================================================


@dataclass
class ComparisonResult:
    """Results of a file comparison operation."""
    files_compared: int = 0
    files_identical: int = 0
    files_different: int = 0
    missing_files: int = 0
    error_files: int = 0
    differences: List[str] = None
    
    def __post_init__(self):
        if self.differences is None:
            self.differences = []


def is_number(s: str) -> bool:
    """Check if a string represents a number (int or float)."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def compare_numbers(val1: Union[str, float, int], val2: Union[str, float, int], 
                   tolerance: float = 1e-9, relative_tolerance: float = 1e-9) -> bool:
    """
    Compare two numerical values with tolerance for precision differences.
    Handles both string and numeric inputs.
    """
    try:
        if isinstance(val1, str):
            val1 = float(val1)
        if isinstance(val2, str):
            val2 = float(val2)
        
        val1 = float(val1)
        val2 = float(val2)
        
        # Handle special cases
        if val1 == val2:
            return True
        
        # Handle NaN values
        if np.isnan(val1) and np.isnan(val2):
            return True
        if np.isnan(val1) or np.isnan(val2):
            return False
        
        # Handle infinity
        if np.isinf(val1) and np.isinf(val2):
            return np.sign(val1) == np.sign(val2)
        if np.isinf(val1) or np.isinf(val2):
            return False
        
        # Check if both are zero (or very close to zero)
        if abs(val1) < tolerance and abs(val2) < tolerance:
            return True
        
        # Use relative tolerance for larger numbers
        if abs(val1) > tolerance or abs(val2) > tolerance:
            relative_diff = abs(val1 - val2) / max(abs(val1), abs(val2))
            return relative_diff <= relative_tolerance
        
        # Use absolute tolerance for small numbers
        return abs(val1 - val2) <= tolerance
        
    except (ValueError, TypeError):
        # If conversion fails, fall back to string comparison
        return str(val1) == str(val2)


def read_binary_file(filepath: str, dtype: str = "float32", endian: str = "little") -> np.ndarray:
    """
    Read binary file as numpy array with specified dtype and endianness.
    Supports formats like those created with numpy's tofile() method.
    """
    # Map endianness
    endian_char = '<' if endian == "little" else '>'
    
    # Map common dtype names
    dtype_map = {
        "float32": f"{endian_char}f4",
        "float64": f"{endian_char}f8", 
        "int32": f"{endian_char}i4",
        "int64": f"{endian_char}i8",
        "uint32": f"{endian_char}u4",
        "uint64": f"{endian_char}u8",
    }
    
    numpy_dtype = dtype_map.get(dtype, f"{endian_char}{dtype}")
    
    try:
        return np.fromfile(filepath, dtype=numpy_dtype)
    except Exception as e:
        raise ValueError(f"Failed to read binary file {filepath}: {e}")


def compare_binary_files(file1_path: str, file2_path: str, config: dict) -> Tuple[bool, List[str]]:
    """Compare binary files based on configuration."""
    differences = []
    
    # Determine format from file extension
    file_ext = "*" + Path(file1_path).suffix.lower()
    binary_format = config.get("binary_formats", {}).get(file_ext, {})
    
    if binary_format.get("format") == "raw_bytes":
        # Raw bytes comparison
        try:
            with open(file1_path, 'rb') as f1, open(file2_path, 'rb') as f2:
                data1 = f1.read()
                data2 = f2.read()
            
            if len(data1) != len(data2):
                differences.append(f"Different file sizes: {len(data1)} vs {len(data2)} bytes")
                return False, differences
            
            if data1 != data2:
                # Find first difference
                for i, (b1, b2) in enumerate(zip(data1, data2)):
                    if b1 != b2:
                        differences.append(f"First byte difference at offset {i}: 0x{b1:02x} vs 0x{b2:02x}")
                        break
                return False, differences
            
            return True, []
            
        except Exception as e:
            differences.append(f"Error reading binary files: {e}")
            return False, differences
    
    else:
        # Numpy array comparison
        dtype = binary_format.get("dtype", "float32")
        endian = binary_format.get("endian", "little")
        
        try:
            array1 = read_binary_file(file1_path, dtype, endian)
            array2 = read_binary_file(file2_path, dtype, endian)
        except Exception as e:
            differences.append(f"Error reading binary files as {dtype}: {e}")
            return False, differences
        
        if array1.shape != array2.shape:
            differences.append(f"Different array shapes: {array1.shape} vs {array2.shape}")
            return False, differences
        
        tolerance = config.get("tolerance", 1e-9)
        relative_tolerance = config.get("relative_tolerance", 1e-9)
        
        # Compare arrays element by element for detailed reporting
        differences_found = 0
        max_reported_diffs = 10  # Limit output
        
        for i, (val1, val2) in enumerate(zip(array1.flat, array2.flat)):
            if not compare_numbers(val1, val2, tolerance, relative_tolerance):
                differences_found += 1
                if len(differences) < max_reported_diffs:
                    differences.append(f"Element {i}: {val1} vs {val2}")
        
        if differences_found > 0:
            if differences_found > max_reported_diffs:
                differences.append(f"... and {differences_found - max_reported_diffs} more differences")
            differences.insert(0, f"Total different elements: {differences_found}/{len(array1.flat)}")
            return False, differences
        
        return True, []


def normalize_column_name(name: str, case_sensitive: bool = True) -> str:
    """Normalize column name for comparison."""
    name = name.strip()
    if not case_sensitive:
        name = name.lower()
    return name


def compare_csv_files(file1_path: str, file2_path: str, config: dict) -> Tuple[bool, List[str]]:
    """Compare CSV files with column-specific comparison."""
    differences = []
    csv_settings = config.get("csv_settings", {})
    columns_to_compare = csv_settings.get("columns_to_compare", None)
    ignore_extra_columns = csv_settings.get("ignore_extra_columns", True)
    case_sensitive = csv_settings.get("case_sensitive_columns", False)
    
    tolerance = config.get("tolerance", 1e-9)
    relative_tolerance = config.get("relative_tolerance", 1e-9)
    
    try:
        # Read CSV files
        with open(file1_path, 'r', newline='', encoding='utf-8') as f1:
            reader1 = csv.DictReader(f1)
            data1 = list(reader1)
            headers1 = reader1.fieldnames or []
        
        with open(file2_path, 'r', newline='', encoding='utf-8') as f2:
            reader2 = csv.DictReader(f2)
            data2 = list(reader2)
            headers2 = reader2.fieldnames or []
    
    except Exception as e:
        differences.append(f"Error reading CSV files: {e}")
        return False, differences
    
    # Normalize headers
    norm_headers1 = {normalize_column_name(h, case_sensitive): h for h in headers1}
    norm_headers2 = {normalize_column_name(h, case_sensitive): h for h in headers2}
    
    # Determine columns to compare
    if columns_to_compare is None:
        # Compare all common columns
        common_cols = set(norm_headers1.keys()) & set(norm_headers2.keys())
        if not ignore_extra_columns:
            extra_in_1 = set(norm_headers1.keys()) - set(norm_headers2.keys())
            extra_in_2 = set(norm_headers2.keys()) - set(norm_headers1.keys())
            if extra_in_1:
                differences.append(f"Extra columns in file 1: {[norm_headers1[c] for c in extra_in_1]}")
            if extra_in_2:
                differences.append(f"Extra columns in file 2: {[norm_headers2[c] for c in extra_in_2]}")
    else:
        # Compare only specified columns
        norm_cols_to_compare = {normalize_column_name(c, case_sensitive) for c in columns_to_compare}
        common_cols = norm_cols_to_compare & set(norm_headers1.keys()) & set(norm_headers2.keys())
        
        missing_in_1 = norm_cols_to_compare - set(norm_headers1.keys())
        missing_in_2 = norm_cols_to_compare - set(norm_headers2.keys())
        
        if missing_in_1:
            differences.append(f"Columns missing in file 1: {list(missing_in_1)}")
        if missing_in_2:
            differences.append(f"Columns missing in file 2: {list(missing_in_2)}")
    
    if not common_cols:
        differences.append("No common columns found for comparison")
        return False, differences
    
    # Compare row counts
    if len(data1) != len(data2):
        differences.append(f"Different number of rows: {len(data1)} vs {len(data2)}")
    
    # Compare data row by row
    max_rows = max(len(data1), len(data2))
    max_reported_diffs = 20
    row_diffs = 0
    
    for row_idx in range(max_rows):
        if row_idx >= len(data1):
            differences.append(f"Row {row_idx + 1}: Missing in file 1")
            row_diffs += 1
            continue
        if row_idx >= len(data2):
            differences.append(f"Row {row_idx + 1}: Missing in file 2")
            row_diffs += 1
            continue
        
        row1 = data1[row_idx]
        row2 = data2[row_idx]
        
        for norm_col in common_cols:
            orig_col1 = norm_headers1[norm_col]
            orig_col2 = norm_headers2[norm_col]
            
            val1 = row1.get(orig_col1, "").strip()
            val2 = row2.get(orig_col2, "").strip()
            
            # Try numeric comparison first
            if is_number(val1) and is_number(val2):
                if not compare_numbers(val1, val2, tolerance, relative_tolerance):
                    if row_diffs < max_reported_diffs:
                        differences.append(f"Row {row_idx + 1}, Column '{orig_col1}': {val1} vs {val2}")
                    row_diffs += 1
            else:
                # String comparison
                if val1 != val2:
                    if row_diffs < max_reported_diffs:
                        differences.append(f"Row {row_idx + 1}, Column '{orig_col1}': '{val1}' vs '{val2}'")
                    row_diffs += 1
    
    if row_diffs > max_reported_diffs:
        differences.append(f"... and {row_diffs - max_reported_diffs} more row differences")
    
    return len(differences) == 0, differences


def extract_numbers_and_text(line: str) -> List[Tuple[str, bool]]:
    """Extract numbers and text from a line, preserving order."""
    number_pattern = r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
    
    tokens = []
    last_end = 0
    
    for match in re.finditer(number_pattern, line):
        if match.start() > last_end:
            text_part = line[last_end:match.start()]
            if text_part:
                tokens.append((text_part, False))
        
        tokens.append((match.group(), True))
        last_end = match.end()
    
    if last_end < len(line):
        remaining_text = line[last_end:]
        if remaining_text:
            tokens.append((remaining_text, False))
    
    if not tokens:
        tokens.append((line, False))
    
    return tokens


def compare_lines(line1: str, line2: str, tolerance: float = 1e-9,
                 relative_tolerance: float = 1e-9, ignore_whitespace: bool = True) -> Tuple[bool, str]:
    """Compare two lines with intelligent number handling."""
    if ignore_whitespace:
        line1 = line1.strip()
        line2 = line2.strip()
    
    if line1 == line2:
        return True, ""
    
    tokens1 = extract_numbers_and_text(line1)
    tokens2 = extract_numbers_and_text(line2)
    
    if len(tokens1) != len(tokens2):
        return False, f"Different number of tokens: {len(tokens1)} vs {len(tokens2)}"
    
    differences = []
    for i, ((token1, is_num1), (token2, is_num2)) in enumerate(zip(tokens1, tokens2)):
        if is_num1 and is_num2:
            if not compare_numbers(token1, token2, tolerance, relative_tolerance):
                differences.append(f"Number mismatch at position {i}: {token1} vs {token2}")
        elif is_num1 != is_num2:
            differences.append(f"Type mismatch at position {i}: {'number' if is_num1 else 'text'} vs {'number' if is_num2 else 'text'}")
        else:
            if token1 != token2:
                differences.append(f"Text mismatch at position {i}: '{token1}' vs '{token2}'")
    
    if differences:
        return False, "; ".join(differences)
    return True, ""


def get_file_type(filepath: str) -> str:
    """Determine file type based on extension."""
    ext = Path(filepath).suffix.lower()
    if ext == '.csv':
        return 'csv'
    elif ext in ['.bin', '.dat', '.model']:
        return 'binary'
    else:
        return 'text'


def compare_files(file1_path: str, file2_path: str, config: dict) -> Tuple[bool, List[str]]:
    """Compare two files based on their type and configuration."""
    if not Path(file1_path).exists():
        return False, [f"File not found: {file1_path}"]
    if not Path(file2_path).exists():
        return False, [f"File not found: {file2_path}"]
    
    file_type = get_file_type(file1_path)
    verbose = config.get("verbose", False)
    
    if file_type == 'csv':
        return compare_csv_files(file1_path, file2_path, config)
    elif file_type == 'binary':
        return compare_binary_files(file1_path, file2_path, config)
    else:
        # Text file comparison
        return compare_text_files(file1_path, file2_path, config)


def compare_text_files(file1_path: str, file2_path: str, config: dict) -> Tuple[bool, List[str]]:
    """Compare text files with precision handling."""
    tolerance = config.get("tolerance", 1e-9)
    relative_tolerance = config.get("relative_tolerance", 1e-9)
    ignore_whitespace = config.get("ignore_whitespace", True)
    verbose = config.get("verbose", False)
    
    try:
        with open(file1_path, 'r', encoding='utf-8') as f1, \
             open(file2_path, 'r', encoding='utf-8') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
    except Exception as e:
        return False, [f"Error reading files: {e}"]
    
    differences = []
    files_same = True
    
    if len(lines1) != len(lines2):
        files_same = False
        differences.append(f"Different number of lines: {len(lines1)} vs {len(lines2)}")
        if verbose and not QUIET_MODE:
            print(f"File 1 has {len(lines1)} lines, File 2 has {len(lines2)} lines")
    
    max_lines = max(len(lines1), len(lines2))
    for line_num in range(max_lines):
        line1 = lines1[line_num] if line_num < len(lines1) else ""
        line2 = lines2[line_num] if line_num < len(lines2) else ""
        
        if line_num >= len(lines1):
            files_same = False
            differences.append(f"Line {line_num + 1}: File 1 ends, File 2 has: '{line2.rstrip()}'")
        elif line_num >= len(lines2):
            files_same = False
            differences.append(f"Line {line_num + 1}: File 2 ends, File 1 has: '{line1.rstrip()}'")
        else:
            is_same, diff_desc = compare_lines(line1, line2, tolerance, relative_tolerance, ignore_whitespace)
            if not is_same:
                files_same = False
                differences.append(f"Line {line_num + 1}: {diff_desc}")
                if verbose and not QUIET_MODE:
                    print(f"Line {line_num + 1}:")
                    print(f"  File 1: '{line1.rstrip()}'")
                    print(f"  File 2: '{line2.rstrip()}'")
                    print(f"  Issue: {diff_desc}")
    
    return files_same, differences


def get_matching_files(directory: str, patterns: List[str], recursive: bool = False) -> Set[str]:
    """Get all files in directory matching the given patterns."""
    from fnmatch import fnmatch
    
    directory = Path(directory)
    matching_files = set()
    
    if recursive:
        for pattern in patterns:
            matching_files.update(str(f.relative_to(directory)) for f in directory.rglob(pattern) if f.is_file())
    else:
        for pattern in patterns:
            matching_files.update(str(f.relative_to(directory)) for f in directory.glob(pattern) if f.is_file())
    
    return matching_files


def compare_directories(dir1: str, dir2: str, config: dict) -> ComparisonResult:
    """Compare all matching files between two directories."""
    result = ComparisonResult()
    verbose = config.get("verbose", False)
    
    dir1_path = Path(dir1)
    dir2_path = Path(dir2)
    
    if not dir1_path.exists():
        result.differences.append(f"Directory 1 does not exist: {dir1}")
        result.error_files += 1
        return result
    
    if not dir2_path.exists():
        result.differences.append(f"Directory 2 does not exist: {dir2}")
        result.error_files += 1
        return result
    
    patterns = config.get("file_patterns", ["*.txt", "*.log", "*.csv", "*.bin", "*.dat"])
    recursive = config.get("recursive", False)
    
    files1 = get_matching_files(dir1, patterns, recursive)
    files2 = get_matching_files(dir2, patterns, recursive)
    all_files = files1.union(files2)
    
    if not all_files:
        result.differences.append("No matching files found in either directory")
        return result
    
    if not QUIET_MODE:
        print(f"Found {len(all_files)} files to compare")
        if verbose:
            print(f"Files in dir1: {len(files1)}, Files in dir2: {len(files2)}")
    
    for relative_path in sorted(all_files):
        file1_path = dir1_path / relative_path
        file2_path = dir2_path / relative_path
        
        if not file1_path.exists():
            result.missing_files += 1
            result.differences.append(f"Missing in dir1: {relative_path}")
            continue
            
        if not file2_path.exists():
            result.missing_files += 1
            result.differences.append(f"Missing in dir2: {relative_path}")
            continue
        
        result.files_compared += 1
        files_same, file_diffs = compare_files(str(file1_path), str(file2_path), config)
        
        if files_same:
            result.files_identical += 1
            if verbose and not QUIET_MODE:
                print(f"✓ {relative_path}")
        else:
            result.files_different += 1
            result.differences.append(f"File differs: {relative_path}")
            for diff in file_diffs:
                result.differences.append(f"  {diff}")
            if verbose and not QUIET_MODE:
                print(f"✗ {relative_path}")
    
    return result


def run_comparison(config: dict) -> ComparisonResult:
    """Run a single comparison configuration."""
    if not QUIET_MODE:
        print(f"Running comparison: {config['name']}")
        print(f"Path 1: {config['path1']}")
        print(f"Path 2: {config['path2']}")
    
    compare_type = config.get("compare_type", "directory")
    path1 = Path(config["path1"])
    path2 = Path(config["path2"])
    
    if compare_type == "file" or (path1.is_file() and path2.is_file()):
        # Single file comparison
        result = ComparisonResult()
        result.files_compared = 1
        files_same, file_diffs = compare_files(str(path1), str(path2), config)
        
        if files_same:
            result.files_identical = 1
        else:
            result.files_different = 1
            result.differences = file_diffs
            
    elif compare_type == "directory" or (path1.is_dir() and path2.is_dir()):
        # Directory comparison
        result = compare_directories(str(path1), str(path2), config)
    else:
        result = ComparisonResult()
        result.differences.append("Path types don't match or paths don't exist")
        result.error_files = 1
    
    return result


def print_results(result: ComparisonResult, config_name: str):
    """Print comparison results."""
    if QUIET_MODE:
        return
    
    print("\n" + "="*60)
    print(f"COMPARISON SUMMARY: {config_name}")
    print("="*60)
    
    if result.error_files > 0:
        print(f"❌ Errors encountered: {result.error_files}")
    
    if result.files_compared > 0:
        print(f"📁 Files compared: {result.files_compared}")
        print(f"✅ Identical files: {result.files_identical}")
        print(f"❌ Different files: {result.files_different}")
    
    if result.missing_files > 0:
        print(f"⚠️  Missing files: {result.missing_files}")
    
    if result.differences:
        print(f"\n📝 Differences found: {len(result.differences)}")
        for diff in result.differences:
            print(f"  {diff}")
    
    overall_success = (result.files_different == 0 and 
                      result.missing_files == 0 and 
                      result.error_files == 0 and 
                      result.files_compared > 0)
    
    print(f"\n🎯 Overall result: {'✅ PASS' if overall_success else '❌ FAIL'}")


def main():
    """Main function - runs all configured comparisons."""
    if not COMPARISON_CONFIGS:
        print("No comparison configurations defined!")
        print("Please modify the COMPARISON_CONFIGS list in the script.")
        sys.exit(1)
    
    if not QUIET_MODE:
        print("🔍 Starting file comparisons...")
        print(f"Found {len(COMPARISON_CONFIGS)} comparison configurations")
    
    all_results = []
    overall_success = True
    
    for config in COMPARISON_CONFIGS:
        if not QUIET_MODE:
            print("\n" + "-"*60)
        
        result = run_comparison(config)
        all_results.append((config["name"], result))
        
        comparison_failed = (result.files_different > 0 or 
                           result.missing_files > 0 or 
                           result.error_files > 0 or
                           result.files_compared == 0)
        
        if comparison_failed:
            overall_success = False
        
        if SHOW_SUMMARY and not QUIET_MODE:
            print_results(result, config["name"])
    
    # Print overall summary
    if not QUIET_MODE and len(COMPARISON_CONFIGS) > 1:
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)
        
        total_compared = sum(result.files_compared for _, result in all_results)
        total_identical = sum(result.files_identical for _, result in all_results)
        total_different = sum(result.files_different for _, result in all_results)
        total_missing = sum(result.missing_files for _, result in all_results)
        total_errors = sum(result.error_files for _, result in all_results)
        
        print(f"📊 Total comparisons: {len(COMPARISON_CONFIGS)}")
        print(f"📁 Total files compared: {total_compared}")
        print(f"✅ Total identical: {total_identical}")
        print(f"❌ Total different: {total_different}")
        if total_missing > 0:
            print(f"⚠️  Total missing: {total_missing}")
        if total_errors > 0:
            print(f"🚨 Total errors: {total_errors}")
        
        print(f"\n🎯 Final result: {'✅ ALL PASSED' if overall_success else '❌ SOME FAILED'}")
        
        if not overall_success:
            print("\n❌ Failed comparisons:")
            for name, result in all_results:
                comparison_failed = (result.files_different > 0 or 
                                   result.missing_files > 0 or 
                                   result.error_files > 0 or
                                   result.files_compared == 0)
                if comparison_failed:
                    print(f"  - {name}")
    
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()