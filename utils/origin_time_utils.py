"""
Origin time utilities for LATTE workflow.

This module provides functions for handling event origin times across preprocessing,
postprocessing, and analysis modules with support for:
- Fixed origin time (all events at same time, e.g., 0s, 10s, 100s)
- Unix epoch mode (convert year + Julian day to Unix timestamp)
"""

from datetime import datetime, timezone
from typing import Dict, Tuple
import json
import os


def convert_to_unix_epoch(year: int, jday: int, hour: int, minute: int, second: float) -> float:
    """
    Convert year, Julian day, and time components to Unix epoch timestamp.

    Args:
        year: Year (e.g., 2008)
        jday: Julian day (1-366)
        hour: Hour (0-23)
        minute: Minute (0-59)
        second: Second with fractional part (0-59.999...)

    Returns:
        Unix epoch timestamp in seconds (seconds since Jan 1, 1970 UTC)
    """
    # Create datetime for January 1st of the given year
    jan_1 = datetime(year, 1, 1, tzinfo=timezone.utc)

    # Calculate total seconds from Julian day and time
    total_seconds = (jday - 1) * 86400 + hour * 3600 + minute * 60 + second

    # Get Unix timestamp for Jan 1 + offset
    jan_1_timestamp = jan_1.timestamp()

    return jan_1_timestamp + total_seconds


def get_origin_time_for_event(
    original_time_s: float,
    mode: str,
    fixed_value: float = 0.0,
    year: int = None,
    jday: int = None,
    hour: int = None,
    minute: int = None,
    second: float = None
) -> float:
    """
    Get the origin time for an event based on the configured mode.

    Args:
        original_time_s: Original origin time in seconds from start of year
        mode: Origin time mode - "fixed" or "unix_epoch"
        fixed_value: Value to use when mode is "fixed" (in seconds)
        year: Year (required for unix_epoch mode)
        jday: Julian day (required for unix_epoch mode)
        hour: Hour (required for unix_epoch mode)
        minute: Minute (required for unix_epoch mode)
        second: Second (required for unix_epoch mode)

    Returns:
        Origin time in seconds based on the mode

    Raises:
        ValueError: If mode is invalid or required parameters are missing
    """
    if mode == "fixed":
        return fixed_value

    elif mode == "unix_epoch":
        if any(param is None for param in [year, jday, hour, minute, second]):
            raise ValueError("unix_epoch mode requires year, jday, hour, minute, and second")
        return convert_to_unix_epoch(year, jday, hour, minute, second)

    else:
        raise ValueError(f"Invalid origin time mode: {mode}. Must be 'fixed' or 'unix_epoch'")


def save_origin_time_metadata(
    output_dir: str,
    mode: str,
    fixed_value: float = 0.0,
    event_original_times: Dict[str, float] = None
) -> None:
    """
    Save origin time configuration metadata for use in postprocessing and analysis.

    Args:
        output_dir: Directory to save metadata file
        mode: Origin time mode used
        fixed_value: Fixed value used (if mode is "fixed")
        event_original_times: Mapping of event_id -> original origin time in seconds
    """
    metadata = {
        'mode': mode,
        'fixed_value': fixed_value,
        'event_original_times': event_original_times or {}
    }

    os.makedirs(output_dir, exist_ok=True)
    metadata_file = os.path.join(output_dir, 'origin_time_metadata.json')

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Origin time metadata saved to: {metadata_file}")


def load_origin_time_metadata(metadata_dir: str) -> Dict:
    """
    Load origin time configuration metadata from preprocessing.

    Args:
        metadata_dir: Directory containing metadata file

    Returns:
        Dictionary with 'mode', 'fixed_value', and 'event_original_times'

    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    metadata_file = os.path.join(metadata_dir, 'origin_time_metadata.json')

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(
            f"Origin time metadata not found: {metadata_file}\n"
            "This file should be created during preprocessing."
        )

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    return metadata


def get_corrected_origin_time(
    event_id: str,
    corrected_time_from_inversion: float,
    metadata: Dict
) -> float:
    """
    Get the actual corrected origin time by adding back the original time.

    This is used in postprocessing to convert the inverted origin time
    back to the original time reference.

    Args:
        event_id: Event identifier
        corrected_time_from_inversion: Corrected time from inversion (relative to mode)
        metadata: Origin time metadata from preprocessing

    Returns:
        Actual corrected origin time in the original reference frame
    """
    mode = metadata['mode']

    if mode == "fixed":
        # For fixed mode, the correction from inversion is relative to the fixed value
        # The actual origin time is: original_time + (inverted_time - fixed_value)
        original_time = metadata['event_original_times'].get(event_id, 0.0)
        fixed_value = metadata['fixed_value']
        return original_time + (corrected_time_from_inversion - fixed_value)

    elif mode == "unix_epoch":
        # For unix_epoch mode, the corrected time is already in Unix epoch
        return corrected_time_from_inversion

    else:
        raise ValueError(f"Invalid origin time mode in metadata: {mode}")
