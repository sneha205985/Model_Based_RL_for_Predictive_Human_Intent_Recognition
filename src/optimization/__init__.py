"""
Performance Optimization Package

This package provides comprehensive performance monitoring, profiling,
and optimization capabilities for the HRI Bayesian RL system.

Modules:
- performance_profiler: Comprehensive performance profiling and monitoring
- system_optimizer: System-level optimization utilities

Author: Phase 5 Implementation
Date: 2024
"""

from .performance_profiler import (
    ComprehensiveProfiler,
    ProfilerConfiguration,
    PerformanceMetrics,
    ProfilerType,
    OptimizationLevel,
    TimingProfiler,
    MemoryProfiler,
    GPUMonitor,
    SystemMonitor,
    PerformanceOptimizer,
    create_profiler,
    get_global_profiler,
    profile_function,
    start_global_profiling,
    stop_global_profiling
)

from .system_optimizer import (
    SystemOptimizer,
    OptimizationConfig,
    OptimizationStrategy,
    ResourceType,
    SmartCache,
    ThreadPoolOptimizer,
    MemoryOptimizer,
    ObjectPool,
    CPUOptimizer,
    optimize_performance,
    get_global_optimizer,
    start_global_optimization,
    stop_global_optimization,
    get_optimization_report
)

__all__ = [
    # Main profiler
    'ComprehensiveProfiler',
    'ProfilerConfiguration',
    'PerformanceMetrics',
    'ProfilerType',
    'OptimizationLevel',
    
    # Individual profiler components
    'TimingProfiler',
    'MemoryProfiler',
    'GPUMonitor',
    'SystemMonitor',
    'PerformanceOptimizer',
    
    # Profiler convenience functions
    'create_profiler',
    'get_global_profiler',
    'profile_function',
    'start_global_profiling',
    'stop_global_profiling',
    
    # System optimizer
    'SystemOptimizer',
    'OptimizationConfig',
    'OptimizationStrategy',
    'ResourceType',
    
    # Optimizer components
    'SmartCache',
    'ThreadPoolOptimizer',
    'MemoryOptimizer',
    'ObjectPool',
    'CPUOptimizer',
    
    # Optimizer convenience functions
    'optimize_performance',
    'get_global_optimizer',
    'start_global_optimization',
    'stop_global_optimization',
    'get_optimization_report'
]