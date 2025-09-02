"""
System Robustness and Error Handling Package

This package provides comprehensive error handling, fault tolerance,
and recovery mechanisms for the HRI Bayesian RL system to ensure
reliable operation in real-world scenarios.

Modules:
- error_handler: Comprehensive error handling and recovery system
- system_resilience: System resilience and fault tolerance mechanisms

Author: Phase 5 Implementation
Date: 2024
"""

from .error_handler import (
    RobustErrorHandler,
    ErrorHandlingConfig,
    ErrorInfo,
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
    CircuitBreaker,
    ErrorRecoveryManager,
    SystemHealthMonitor,
    robust_operation,
    get_global_error_handler,
    initialize_global_error_handler,
    shutdown_global_error_handler
)

from .system_resilience import (
    ResilienceOrchestrator,
    ResilienceConfig,
    ComponentManager,
    FaultDetector,
    StateManager,
    ComponentInfo,
    FaultEvent,
    FaultType,
    SystemState,
    ComponentStatus,
    create_resilience_system,
    get_global_resilience_system,
    initialize_global_resilience,
    shutdown_global_resilience
)

__all__ = [
    # Error handler components
    'RobustErrorHandler',
    'ErrorHandlingConfig',
    'ErrorInfo',
    'ErrorSeverity',
    'ErrorCategory', 
    'RecoveryStrategy',
    'CircuitBreaker',
    'ErrorRecoveryManager',
    'SystemHealthMonitor',
    'robust_operation',
    'get_global_error_handler',
    'initialize_global_error_handler',
    'shutdown_global_error_handler',
    
    # Resilience system components
    'ResilienceOrchestrator',
    'ResilienceConfig',
    'ComponentManager',
    'FaultDetector',
    'StateManager',
    'ComponentInfo',
    'FaultEvent',
    'FaultType',
    'SystemState',
    'ComponentStatus',
    'create_resilience_system',
    'get_global_resilience_system',
    'initialize_global_resilience',
    'shutdown_global_resilience'
]