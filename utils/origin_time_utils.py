"""
Origin time utilities for LATTE workflow.

This module provides functions for handling event origin times across preprocessing,
postprocessing, and analysis modules with support for:
- Fixed origin time (all events at same time, e.g., 0s, 10s, 100s)
- Unix epoch mode (convert year + Julian day to Unix timestamp)
"""

from datetime import datetime, timezone
from typing import Tuple


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
