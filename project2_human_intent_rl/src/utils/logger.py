"""
Logging configuration and utilities for the Human Intent Recognition project.

This module provides centralized logging configuration with support for
multiple output formats, log levels, and file/console output.
"""

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'ENDC': '\033[0m'       # End color
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors for console output."""
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['ENDC']}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    enable_console: bool = True,
    enable_colors: bool = True
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, only console logging is used
        log_format: Custom log format string
        enable_console: Whether to enable console logging
        enable_colors: Whether to use colored console output
    
    Raises:
        ValueError: If invalid log level is provided
    """
    # Validate log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Default log format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if log file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': log_format,
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'colored': {
                '()': ColoredFormatter,
                'format': log_format,
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {},
        'root': {
            'level': log_level.upper(),
            'handlers': []
        }
    }
    
    # Add console handler
    if enable_console:
        config['handlers']['console'] = {
            'level': log_level.upper(),
            'class': 'logging.StreamHandler',
            'stream': sys.stdout,
            'formatter': 'colored' if enable_colors else 'standard'
        }
        config['root']['handlers'].append('console')
    
    # Add file handler
    if log_file:
        config['handlers']['file'] = {
            'level': log_level.upper(),
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': log_file,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'standard'
        }
        config['root']['handlers'].append('file')
    
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def setup_from_env() -> None:
    """
    Set up logging configuration from environment variables.
    
    Reads configuration from:
    - LOG_LEVEL: Logging level (default: INFO)
    - LOG_FILE: Log file path (optional)
    - LOG_FORMAT: Log format string (optional)
    """
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_file = os.getenv('LOG_FILE')
    log_format = os.getenv('LOG_FORMAT')
    
    setup_logging(
        log_level=log_level,
        log_file=log_file,
        log_format=log_format
    )


# Module-level logger
logger = get_logger(__name__)