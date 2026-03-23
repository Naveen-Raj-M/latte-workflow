"""
Logging utilities for the LATTE tomography package.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


class _LevelBasedFormatter(logging.Formatter):
    """Uses a clean format for INFO and below, full format for WARNING+."""

    def __init__(self, fmt_info: str, fmt_warning: str,
                 datefmt: str = '%Y-%m-%d %H:%M:%S'):
        super().__init__(datefmt=datefmt)
        self._fmt_info = fmt_info
        self._fmt_warning = fmt_warning

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno <= logging.INFO:
            self._style._fmt = self._fmt_info
        else:
            self._style._fmt = self._fmt_warning
        return super().format(record)


def setup_logging(level: str = "INFO",
                 log_file: Optional[Union[str, Path]] = None,
                 format_string: Optional[str] = None) -> None:
    """
    Setup logging configuration for the LATTE package.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Optional custom format string
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    # Full format string (used for WARNING+ on console, and for any added file handler)
    full_format = (
        format_string if format_string is not None else
        '%(asctime)s - %(name)s - %(levelname)s - '
        '%(filename)s:%(lineno)d - %(message)s'
    )

    # Clean format for INFO messages on console (no timestamp/module prefix)
    clean_format = '%(message)s'

    # Remove existing console (StreamHandler) handlers added by Hydra to avoid
    # duplicate output.  FileHandlers (e.g. Hydra's log file) are kept intact.
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler):
            root_logger.removeHandler(handler)
    root_logger.setLevel(numeric_level)

    # Console handler: INFO prints clean message only; WARNING+ prints full format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(_LevelBasedFormatter(clean_format, full_format))
    root_logger.addHandler(console_handler)

    # Optional extra file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter(full_format, datefmt='%Y-%m-%d %H:%M:%S'))
        root_logger.addHandler(file_handler)


class ProcessTimer:
    """
    Context manager for timing processes and logging results.
    """
    
    def __init__(self, process_name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize process timer.
        
        Args:
            process_name: Name of the process being timed
            logger: Logger to use (creates new one if None)
        """
        self.process_name = process_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.process_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log results."""
        self.end_time = datetime.now()
        elapsed = self.end_time - self.start_time
        
        if exc_type is None:
            self.logger.info(f"Completed {self.process_name} in {elapsed}")
        else:
            self.logger.error(f"Failed {self.process_name} after {elapsed}: {exc_val}")
    
    @property
    def elapsed_time(self):
        """Get elapsed time if timing is complete."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class ProgressLogger:
    """
    Logger for tracking progress of long-running operations.
    """
    
    def __init__(self, total_items: int, 
                 process_name: str = "Processing",
                 log_interval: int = 1000,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize progress logger.
        
        Args:
            total_items: Total number of items to process
            process_name: Name of the process
            log_interval: Log progress every N items
            logger: Logger to use
        """
        self.total_items = total_items
        self.process_name = process_name
        self.log_interval = log_interval
        self.logger = logger or logging.getLogger(__name__)
        self.current_item = 0
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1) -> None:
        """
        Update progress counter.
        
        Args:
            increment: Number of items processed
        """
        self.current_item += increment
        
        if self.current_item % self.log_interval == 0 or self.current_item == self.total_items:
            self._log_progress()
    
    def _log_progress(self) -> None:
        """Log current progress."""
        elapsed = datetime.now() - self.start_time
        progress_pct = (self.current_item / self.total_items) * 100
        
        if self.current_item > 0:
            rate = self.current_item / elapsed.total_seconds()
            eta_seconds = (self.total_items - self.current_item) / rate
            eta = f"{eta_seconds:.0f}s"
        else:
            rate = 0
            eta = "unknown"
        
        self.logger.info(
            f"{self.process_name}: {self.current_item}/{self.total_items} "
            f"({progress_pct:.1f}%) - Rate: {rate:.1f} items/s - ETA: {eta}"
        )


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a logger with specified name and level.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    return logger


def log_function_call(func):
    """
    Decorator to log function calls with arguments and execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        # Log function call
        args_str = ', '.join([str(arg) for arg in args])
        kwargs_str = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
        all_args = ', '.join(filter(None, [args_str, kwargs_str]))
        
        logger.debug(f"Calling {func.__name__}({all_args})")
        
        # Time execution
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            elapsed = datetime.now() - start_time
            logger.debug(f"Completed {func.__name__} in {elapsed}")
            return result
        except Exception as e:
            elapsed = datetime.now() - start_time
            logger.error(f"Failed {func.__name__} after {elapsed}: {e}")
            raise
    
    return wrapper