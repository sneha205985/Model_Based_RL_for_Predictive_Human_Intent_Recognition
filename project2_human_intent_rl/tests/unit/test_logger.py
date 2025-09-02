"""
Unit tests for logging utilities.
"""

import pytest
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from src.utils.logger import (
    setup_logging,
    get_logger,
    setup_from_env,
    ColoredFormatter
)


class TestColoredFormatter:
    """Test cases for ColoredFormatter."""
    
    def test_colored_formatter_format(self):
        """Test that colored formatter adds colors to log records."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        
        # Create a log record
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert 'INFO' in formatted
        assert 'Test message' in formatted
        # Check that color codes are present (ANSI escape sequences)
        assert '\033[32m' in formatted  # Green color for INFO
        assert '\033[0m' in formatted   # End color


class TestSetupLogging:
    """Test cases for setup_logging function."""
    
    def test_setup_logging_with_defaults(self):
        """Test setup_logging with default parameters."""
        setup_logging()
        
        # Check that root logger is configured
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) > 0
    
    def test_setup_logging_with_file(self):
        """Test setup_logging with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            setup_logging(
                log_level="DEBUG",
                log_file=str(log_file),
                enable_console=False
            )
            
            # Test that log file directory is created
            assert log_file.parent.exists()
            
            # Test logging to file
            logger = get_logger("test")
            logger.info("Test message")
            
            # Check that file was created
            assert log_file.exists()
    
    def test_setup_logging_invalid_level(self):
        """Test setup_logging with invalid log level."""
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging(log_level="INVALID_LEVEL")
    
    def test_setup_logging_custom_format(self):
        """Test setup_logging with custom format."""
        custom_format = "%(name)s: %(message)s"
        
        setup_logging(
            log_format=custom_format,
            enable_console=True,
            enable_colors=False
        )
        
        # This test mainly ensures no exceptions are raised
        logger = get_logger("test")
        logger.info("Test message")
    
    def test_setup_logging_no_console(self):
        """Test setup_logging with console disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            setup_logging(
                log_file=str(log_file),
                enable_console=False
            )
            
            # Verify file logging works
            logger = get_logger("test")
            logger.info("Test message")
            assert log_file.exists()


class TestGetLogger:
    """Test cases for get_logger function."""
    
    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
    
    def test_get_logger_same_name_same_instance(self):
        """Test that get_logger returns same instance for same name."""
        logger1 = get_logger("same_name")
        logger2 = get_logger("same_name")
        assert logger1 is logger2


class TestSetupFromEnv:
    """Test cases for setup_from_env function."""
    
    @patch.dict('os.environ', {
        'LOG_LEVEL': 'DEBUG',
        'LOG_FORMAT': '%(name)s - %(message)s'
    })
    def test_setup_from_env_with_env_vars(self):
        """Test setup_from_env with environment variables."""
        setup_from_env()
        
        # Check that environment variables were used
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
    
    @patch.dict('os.environ', {}, clear=True)
    def test_setup_from_env_with_defaults(self):
        """Test setup_from_env with default values."""
        setup_from_env()
        
        # Check that defaults were used
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
    
    @patch.dict('os.environ', {
        'LOG_LEVEL': 'WARNING',
        'LOG_FILE': '/tmp/test.log',
        'LOG_FORMAT': 'Custom: %(message)s'
    })
    def test_setup_from_env_all_variables(self):
        """Test setup_from_env with all environment variables set."""
        setup_from_env()
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING


@pytest.mark.unit
class TestLoggingIntegration:
    """Integration tests for logging functionality."""
    
    def test_logger_hierarchy(self):
        """Test logger hierarchy works correctly."""
        setup_logging(log_level="DEBUG")
        
        parent_logger = get_logger("parent")
        child_logger = get_logger("parent.child")
        
        assert child_logger.parent == parent_logger
    
    def test_log_message_flow(self):
        """Test that log messages flow through the system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "flow_test.log"
            
            setup_logging(
                log_level="INFO",
                log_file=str(log_file),
                enable_console=False
            )
            
            logger = get_logger("flow_test")
            test_message = "Flow test message"
            logger.info(test_message)
            
            # Read log file and verify message
            with open(log_file, 'r') as f:
                content = f.read()
                assert test_message in content
                assert "flow_test" in content
    
    def test_different_log_levels(self):
        """Test that different log levels work correctly."""
        setup_logging(log_level="WARNING")
        
        logger = get_logger("level_test")
        
        # These should work without raising exceptions
        logger.debug("Debug message")    # Should not appear
        logger.info("Info message")      # Should not appear  
        logger.warning("Warning message") # Should appear
        logger.error("Error message")    # Should appear
        logger.critical("Critical message") # Should appear