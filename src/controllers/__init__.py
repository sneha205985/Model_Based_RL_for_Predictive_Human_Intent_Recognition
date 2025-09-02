"""
Controllers module for Model Predictive Control implementations.

This module contains working implementations for:
- Model Predictive Control (MPC)
- Robot control and trajectory optimization
- Safety constraints and collision avoidance
"""

from .mpc_controller import MPCController

__all__ = [
    "MPCController",
]