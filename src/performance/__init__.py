"""
Performance Monitoring and Benchmarking Framework

This module provides comprehensive performance monitoring and benchmarking:
- Real-time performance tracking with microsecond precision
- Production monitoring with automated alerting
- Statistical performance validation
- Load testing and scalability analysis

Components:
    comprehensive_benchmarking: Performance benchmarking with statistical validation
    production_monitoring: Real-time monitoring and alerting system
"""

__version__ = "1.0.0"

# Import main components with error handling
try:
    from .comprehensive_benchmarking import (
        run_performance_benchmarks,
        HighPrecisionTimer,
        SystemResourceMonitor,
        StatisticalAnalyzer as PerformanceStatisticalAnalyzer
    )
except ImportError as e:
    run_performance_benchmarks = None
    HighPrecisionTimer = None
    SystemResourceMonitor = None
    PerformanceStatisticalAnalyzer = None
    import warnings
    warnings.warn(f"Could not import benchmarking components: {e}")

try:
    from .production_monitoring import (
        ProductionMonitoringContext,
        MetricCollector,
        AnomalyDetector,
        AlertManager
    )
except ImportError as e:
    ProductionMonitoringContext = None
    MetricCollector = None
    AnomalyDetector = None
    AlertManager = None
    import warnings
    warnings.warn(f"Could not import monitoring components: {e}")

__all__ = [
    "run_performance_benchmarks",
    "HighPrecisionTimer", 
    "SystemResourceMonitor",
    "PerformanceStatisticalAnalyzer",
    "ProductionMonitoringContext",
    "MetricCollector",
    "AnomalyDetector", 
    "AlertManager"
]