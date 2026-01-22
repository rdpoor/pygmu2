"""
Configuration and error handling utilities for pygmu2.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from enum import Enum
from typing import Type, Optional
from pygmu2.logger import get_logger

logger = get_logger(__name__)


class ErrorMode(Enum):
    """
    Error handling mode for pygmu2 operations.
    
    STRICT: All errors raise exceptions (default, fail-fast)
    LENIENT: Non-fatal errors become warnings, execution continues
    """
    STRICT = "strict"
    LENIENT = "lenient"


# Module-level default error mode
DEFAULT_ERROR_MODE: ErrorMode = ErrorMode.STRICT


def set_error_mode(mode: ErrorMode) -> None:
    """
    Set the default error mode for all pygmu2 operations.
    
    Args:
        mode: The error mode to use
    """
    global DEFAULT_ERROR_MODE
    DEFAULT_ERROR_MODE = mode


def get_error_mode() -> ErrorMode:
    """
    Get the current default error mode.
    
    Returns:
        The current error mode
    """
    return DEFAULT_ERROR_MODE


def handle_error(
    message: str,
    fatal: bool = False,
    error_mode: Optional[ErrorMode] = None,
    exception_class: Type[Exception] = RuntimeError,
) -> bool:
    """
    Handle an error based on the error mode.
    
    In STRICT mode (or if fatal=True), raises an exception.
    In LENIENT mode (and fatal=False), logs a warning and returns True.
    
    Args:
        message: Error description
        fatal: If True, always raise regardless of mode
        error_mode: Override the default error mode (optional)
        exception_class: Exception type to raise (default: RuntimeError)
    
    Returns:
        True if operation should continue (warning was issued)
    
    Raises:
        exception_class: If in STRICT mode or fatal=True
    
    Example:
        # In strict mode, raises RuntimeError
        # In lenient mode, logs warning and returns True
        if self._started:
            if handle_error("Already started."):
                return  # Continue in lenient mode
        
        # Always raises regardless of mode
        if self._source is None:
            handle_error("No source set.", fatal=True)
    """
    mode = error_mode if error_mode is not None else DEFAULT_ERROR_MODE
    
    if fatal or mode == ErrorMode.STRICT:
        raise exception_class(message)
    else:
        logger.warning(message)
        return True
