"""
Integration Module for Model-Based RL Human Intent Recognition System

This module provides integration components that connect various system modules
including Bayesian RL agents, human behavior models, safety systems, and 
control interfaces.

Main Components:
- HRIBayesianRLIntegration: Main integration class
- SystemOrchestrator: Coordinates multiple system components  
- IntegrationConfig: Configuration management for integrated systems
- ComponentRegistry: Service registry for system components

The integration layer ensures seamless communication between:
- Sensor fusion and data processing
- Intent prediction and uncertainty quantification
- Safety assessment and constraint handling
- Control generation and execution
"""

from typing import Optional, Dict, Any, List, Tuple
import logging

# Version information
__version__ = "1.0.0"
__author__ = "Model-Based RL Team"
__email__ = "research@anthropic.com"

# Configure module-level logging
logger = logging.getLogger(__name__)

# Import main integration components
try:
    from .hri_bayesian_rl import HRIBayesianRLIntegration, HRIBayesianRLConfig
    from .system_orchestrator import SystemOrchestrator, OrchestrationConfig
    from .component_registry import ComponentRegistry, ServiceInterface
    from .integration_utils import IntegrationError, validate_system_state
    
    __all__ = [
        'HRIBayesianRLIntegration',
        'HRIBayesianRLConfig', 
        'SystemOrchestrator',
        'OrchestrationConfig',
        'ComponentRegistry',
        'ServiceInterface',
        'IntegrationError',
        'validate_system_state'
    ]
    
    logger.info(f"Integration module v{__version__} loaded successfully")
    
except ImportError as e:
    logger.warning(f"Some integration components not available: {e}")
    # Fallback imports for basic functionality
    __all__ = ['logger']

# Module-level configuration
DEFAULT_CONFIG = {
    'enable_safety_checks': True,
    'real_time_mode': True,
    'logging_level': 'INFO',
    'component_timeout': 5.0,  # seconds
    'max_retry_attempts': 3
}

def get_integration_info() -> Dict[str, Any]:
    """
    Get information about the integration module.
    
    Returns:
        Dict containing module version, available components, and configuration
    """
    return {
        'version': __version__,
        'author': __author__, 
        'available_components': __all__,
        'default_config': DEFAULT_CONFIG
    }

def configure_integration(config: Dict[str, Any]) -> bool:
    """
    Configure integration module with custom settings.
    
    Args:
        config: Dictionary of configuration parameters
        
    Returns:
        True if configuration successful, False otherwise
    """
    try:
        # Validate configuration parameters
        for key, value in config.items():
            if key in DEFAULT_CONFIG:
                DEFAULT_CONFIG[key] = value
                logger.debug(f"Updated {key} = {value}")
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to configure integration module: {e}")
        return False