"""
Metadata stored in the header of CSV files as ``# key: value`` comment lines.

Values are JSON encoded. Readers that pass ``comment='#'`` to ``pandas.read_csv`` skip
these lines. Nested dictionaries are flattened to ``key.subkey`` (for example
``units.event_x``).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

COMMENT_CHAR = '#'


def to_builtin(value: Any) -> Any:
    """Convert numpy / bytes values to plain Python values."""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [to_builtin(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


def flatten_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten nested dicts to ``key.subkey``, convert to plain Python types, drop None values."""
    flat: Dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if isinstance(value, dict):
            for subkey, subvalue in flatten_metadata(value).items():
                flat[f'{key}.{subkey}'] = subvalue
        elif value is not None:
            flat[key] = to_builtin(value)
    return flat


def read_csv_header_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Read the ``# key: value`` comment lines at the top of a CSV file."""
    raw: Dict[str, Any] = {}
    with open(filepath, 'r') as f:
        for line in f:
            if not line.startswith(COMMENT_CHAR):
                break
            key, sep, value = line[1:].strip().partition(':')
            if not sep:
                continue
            value = value.strip()
            try:
                raw[key.strip()] = json.loads(value)
            except ValueError:
                raw[key.strip()] = value
    return raw


def csv_header_lines(metadata: Dict[str, Any]) -> List[str]:
    """Format flat metadata as CSV header comment lines."""
    return [f"{COMMENT_CHAR} {key}: {json.dumps(value)}\n" for key, value in metadata.items()]
