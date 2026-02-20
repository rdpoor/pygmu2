"""
Logging configuration for pygmu2

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import logging
import sys
from typing import Optional


def set_global_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        log_file: Optional file path to write logs to
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=format_string,
        handlers=handlers,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logger = logging.getLogger("pygmu2")
    logger.setLevel(log_level)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Logger name (defaults to 'pygmu2')
        
    Returns:
        Logger instance
    """
    if name is None:
        name = "pygmu2"
    return logging.getLogger(name)
