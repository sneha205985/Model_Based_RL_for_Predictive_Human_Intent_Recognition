"""
Optional Dependencies Manager
Model-Based RL Human Intent Recognition System

This module manages optional dependencies with graceful fallbacks,
ensuring core functionality works without optional packages.

Features:
- Automatic dependency detection
- Graceful fallback mechanisms  
- User-friendly installation messages
- Feature availability checking

Author: Enhanced Dependencies Team
Date: September 2025
"""

import logging
import sys
from typing import Dict, List, Optional, Any, Callable
from functools import wraps

logger = logging.getLogger(__name__)

# Track availability of optional dependencies
_OPTIONAL_DEPS = {}

def check_optional_dependency(module_name: str, feature_name: str = None, 
                            install_command: str = None) -> bool:
    """
    Check if an optional dependency is available.
    
    Args:
        module_name: Name of the module to check
        feature_name: Human-readable feature name
        install_command: Installation command suggestion
    
    Returns:
        True if module is available, False otherwise
    """
    if module_name in _OPTIONAL_DEPS:
        return _OPTIONAL_DEPS[module_name]['available']
    
    try:
        __import__(module_name)
        _OPTIONAL_DEPS[module_name] = {
            'available': True,
            'feature_name': feature_name or module_name,
            'install_command': install_command or f"pip install {module_name}"
        }
        return True
    except ImportError:
        _OPTIONAL_DEPS[module_name] = {
            'available': False,
            'feature_name': feature_name or module_name,
            'install_command': install_command or f"pip install {module_name}"
        }
        return False

def require_optional(module_name: str, feature_name: str = None, 
                    install_command: str = None, fallback_message: str = None):
    """
    Decorator to require optional dependency with graceful fallback.
    
    Args:
        module_name: Required module name
        feature_name: Human-readable feature name
        install_command: Installation command
        fallback_message: Message when feature unavailable
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not check_optional_dependency(module_name, feature_name, install_command):
                dep_info = _OPTIONAL_DEPS[module_name]
                message = (fallback_message or 
                          f"{dep_info['feature_name']} not available. "
                          f"Install with: {dep_info['install_command']}")
                logger.warning(message)
                
                # Return None or raise exception based on context
                if 'fallback' in kwargs:
                    return kwargs['fallback']
                return None
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Initialize common optional dependencies
OPTIONAL_FEATURES = {
    'h5py': {
        'feature_name': 'HDF5 Data Storage',
        'install_command': 'pip install h5py',
        'description': 'Advanced data storage in HDF5 format'
    },
    'dash': {
        'feature_name': 'Interactive Dashboards', 
        'install_command': 'pip install dash dash-bootstrap-components',
        'description': 'Real-time interactive web dashboards'
    },
    'bokeh': {
        'feature_name': 'Interactive Plotting',
        'install_command': 'pip install bokeh',
        'description': 'Advanced interactive visualizations'
    },
    'altair': {
        'feature_name': 'Statistical Visualizations',
        'install_command': 'pip install altair',
        'description': 'Grammar of graphics statistical plots'
    },
    'xarray': {
        'feature_name': 'N-dimensional Arrays',
        'install_command': 'pip install xarray',
        'description': 'Labeled multi-dimensional arrays'
    },
    'dask': {
        'feature_name': 'Parallel Computing',
        'install_command': 'pip install dask',
        'description': 'Parallel and distributed computing'
    },
    'netCDF4': {
        'feature_name': 'NetCDF Data Format',
        'install_command': 'pip install netCDF4',
        'description': 'Network Common Data Format support'
    },
    'kaleido': {
        'feature_name': 'Static Plot Export',
        'install_command': 'pip install kaleido',
        'description': 'Export plotly figures to static images'
    },
    'psutil': {
        'feature_name': 'System Monitoring',
        'install_command': 'pip install psutil',
        'description': 'System and process monitoring'
    }
}

# Check all optional dependencies on import
for module_name, info in OPTIONAL_FEATURES.items():
    check_optional_dependency(
        module_name, 
        info['feature_name'], 
        info['install_command']
    )

def get_feature_status() -> Dict[str, Dict[str, Any]]:
    """
    Get status of all optional features.
    
    Returns:
        Dictionary with feature availability and installation info
    """
    return _OPTIONAL_DEPS.copy()

def print_feature_status():
    """Print status of all optional features."""
    print("Optional Feature Status:")
    print("=" * 50)
    
    available_count = 0
    total_count = len(_OPTIONAL_DEPS)
    
    for module_name, info in _OPTIONAL_DEPS.items():
        status = "✅ Available" if info['available'] else "❌ Missing"
        feature_info = OPTIONAL_FEATURES.get(module_name, {})
        description = feature_info.get('description', 'No description')
        
        print(f"{status} {info['feature_name']}")
        print(f"    {description}")
        
        if not info['available']:
            print(f"    Install: {info['install_command']}")
        
        if info['available']:
            available_count += 1
        print()
    
    print(f"Summary: {available_count}/{total_count} optional features available")
    
    if available_count < total_count:
        print("\nTo install all optional features:")
        print("pip install -r requirements-optional.txt")

def safe_import(module_name: str, feature_name: str = None, 
               install_command: str = None, default=None):
    """
    Safely import a module with fallback.
    
    Args:
        module_name: Module to import
        feature_name: Human-readable feature name
        install_command: Installation command
        default: Default value if import fails
    
    Returns:
        Imported module or default value
    """
    if check_optional_dependency(module_name, feature_name, install_command):
        return __import__(module_name)
    else:
        dep_info = _OPTIONAL_DEPS[module_name]
        logger.debug(f"{dep_info['feature_name']} not available - using fallback")
        return default

# Convenience functions for common imports
def get_h5py(warn: bool = True):
    """Get h5py with fallback handling."""
    module = safe_import('h5py', 'HDF5 Data Storage', 'pip install h5py')
    if module is None and warn:
        logger.warning("HDF5 functionality disabled - install h5py for advanced data storage")
    return module

def get_dash(warn: bool = True):
    """Get dash with fallback handling.""" 
    module = safe_import('dash', 'Interactive Dashboards', 
                        'pip install dash dash-bootstrap-components')
    if module is None and warn:
        logger.warning("Interactive dashboards disabled - install dash for web interfaces")
    return module

def get_bokeh(warn: bool = True):
    """Get bokeh with fallback handling."""
    module = safe_import('bokeh', 'Interactive Plotting', 'pip install bokeh')
    if module is None and warn:
        logger.warning("Interactive plotting disabled - install bokeh for advanced visualizations")
    return module

def get_altair(warn: bool = True):
    """Get altair with fallback handling."""
    module = safe_import('altair', 'Statistical Visualizations', 'pip install altair')
    if module is None and warn:
        logger.warning("Statistical visualizations disabled - install altair for grammar of graphics")
    return module

# Feature availability flags (for easy checking)
HAS_H5PY = check_optional_dependency('h5py')
HAS_DASH = check_optional_dependency('dash') 
HAS_BOKEH = check_optional_dependency('bokeh')
HAS_ALTAIR = check_optional_dependency('altair')
HAS_XARRAY = check_optional_dependency('xarray')
HAS_DASK = check_optional_dependency('dask')
HAS_NETCDF = check_optional_dependency('netCDF4')
HAS_KALEIDO = check_optional_dependency('kaleido')
HAS_PSUTIL = check_optional_dependency('psutil')

def get_installation_guide() -> str:
    """Generate installation guide for missing features."""
    missing_features = [
        (name, info) for name, info in _OPTIONAL_DEPS.items()
        if not info['available']
    ]
    
    if not missing_features:
        return "All optional features are available! 🎉"
    
    guide = "Optional Features Installation Guide\n"
    guide += "=" * 40 + "\n\n"
    
    guide += "Missing features:\n"
    for module_name, info in missing_features:
        feature_info = OPTIONAL_FEATURES.get(module_name, {})
        guide += f"• {info['feature_name']}: {feature_info.get('description', '')}\n"
        guide += f"  Install: {info['install_command']}\n\n"
    
    guide += "Install all optional features:\n"
    guide += "pip install -r requirements-optional.txt\n\n"
    
    guide += "Or install individual features as needed:\n"
    for module_name, info in missing_features[:3]:  # Show first 3
        guide += f"pip install {module_name}\n"
    
    return guide

if __name__ == "__main__":
    print_feature_status()
    print("\n" + get_installation_guide())