"""
Control module for Model Predictive Control implementation.

This module provides:
- MPCController: Core MPC implementation
- HRIMPCController: Human-Robot Interaction aware MPC
- Safety constraints and robustness features
"""

from .mpc_controller import MPCController, MPCConfiguration, MPCResult, MPCStatus, SolverType
from .hri_mpc import HRIMPCController, HRIConfiguration, HumanState, InteractionPhase, create_default_hri_mpc
from .safety_constraints import (
    SafetyConstraint, CollisionAvoidanceConstraint, WorkspaceLimitConstraint, 
    JointLimitConstraint, ControlBarrierConstraint, SafetyMonitor, 
    RobustMPCFormulation, create_default_safety_constraints
)

__all__ = [
    # Core MPC
    'MPCController', 'MPCConfiguration', 'MPCResult', 'MPCStatus', 'SolverType',
    
    # HRI MPC
    'HRIMPCController', 'HRIConfiguration', 'HumanState', 'InteractionPhase',
    'create_default_hri_mpc',
    
    # Safety constraints
    'SafetyConstraint', 'CollisionAvoidanceConstraint', 'WorkspaceLimitConstraint',
    'JointLimitConstraint', 'ControlBarrierConstraint', 'SafetyMonitor',
    'RobustMPCFormulation', 'create_default_safety_constraints'
]