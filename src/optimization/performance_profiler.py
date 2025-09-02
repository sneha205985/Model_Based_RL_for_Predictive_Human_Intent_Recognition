"""
Performance Profiling and Optimization System

This module provides comprehensive performance monitoring, profiling,
and optimization capabilities for the HRI Bayesian RL system.

Features:
- Real-time performance monitoring
- Code profiling with detailed timing analysis
- Memory usage tracking and optimization
- GPU utilization monitoring (if available)
- Performance bottleneck detection
- Automated optimization recommendations
- Performance regression detection
- Scalability analysis

Author: Phase 5 Implementation
Date: 2024
"""

import numpy as np
import time
import psutil
import threading
import functools
import cProfile
import pstats
import io
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import tracemalloc
import gc
import sys
import os

# Try to import GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Memory profiling
try:
    from memory_profiler import profile as memory_profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    def memory_profile(func):
        return func

# Line profiler
try:
    from line_profiler import LineProfiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProfilerType(Enum):
    """Types of profiling"""
    FUNCTION_TIMING = auto()
    MEMORY_USAGE = auto()
    CPU_PROFILING = auto()
    GPU_MONITORING = auto()
    SYSTEM_RESOURCES = auto()
    CUSTOM_METRICS = auto()


class OptimizationLevel(Enum):
    """Optimization levels"""
    NONE = auto()
    BASIC = auto()
    AGGRESSIVE = auto()
    EXPERIMENTAL = auto()


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    timestamp: float
    function_name: str
    execution_time: float
    memory_usage: float
    cpu_percent: float
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    thread_id: int = 0
    process_id: int = 0
    call_count: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'function_name': self.function_name,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'cpu_percent': self.cpu_percent,
            'gpu_usage': self.gpu_usage,
            'gpu_memory': self.gpu_memory,
            'thread_id': self.thread_id,
            'process_id': self.process_id,
            'call_count': self.call_count
        }


@dataclass
class ProfilerConfiguration:
    """Configuration for performance profiler"""
    # Profiling settings
    enable_function_timing: bool = True
    enable_memory_profiling: bool = True
    enable_cpu_profiling: bool = True
    enable_gpu_monitoring: bool = True
    enable_system_monitoring: bool = True
    
    # Monitoring intervals
    system_monitor_interval: float = 1.0  # seconds
    memory_check_interval: float = 0.5
    gpu_check_interval: float = 1.0
    
    # Data retention
    max_records: int = 10000
    data_retention_days: int = 30
    
    # Output settings
    output_directory: str = "performance_data"
    auto_export: bool = True
    export_interval: float = 300.0  # 5 minutes
    
    # Optimization settings
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    auto_optimize: bool = False
    optimization_threshold: float = 0.8  # Trigger optimization at 80% resource usage
    
    # Alert settings
    enable_alerts: bool = True
    memory_alert_threshold: float = 0.85  # 85% memory usage
    cpu_alert_threshold: float = 0.90     # 90% CPU usage
    execution_time_alert_threshold: float = 1.0  # 1 second


class TimingProfiler:
    """Function timing profiler with decorator support"""
    
    def __init__(self, config: ProfilerConfiguration):
        """Initialize timing profiler"""
        self.config = config
        self.timing_data = defaultdict(list)
        self.call_counts = defaultdict(int)
        self._lock = threading.Lock()
        
    def profile_function(self, func_name: str = None):
        """Decorator for profiling function execution time"""
        def decorator(func):
            nonlocal func_name
            if func_name is None:
                func_name = f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.config.enable_function_timing:
                    return func(*args, **kwargs)
                
                start_time = time.perf_counter()
                start_memory = self._get_current_memory()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    end_memory = self._get_current_memory()
                    
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    # Record metrics
                    metrics = PerformanceMetrics(
                        timestamp=time.time(),
                        function_name=func_name,
                        execution_time=execution_time,
                        memory_usage=memory_delta,
                        cpu_percent=psutil.cpu_percent(),
                        thread_id=threading.get_ident(),
                        process_id=os.getpid()
                    )
                    
                    self._record_timing(metrics)
                    
                    # Check for alerts
                    if (self.config.enable_alerts and 
                        execution_time > self.config.execution_time_alert_threshold):
                        logger.warning(f"Slow function detected: {func_name} took {execution_time:.3f}s")
            
            return wrapper
        return decorator
    
    def _get_current_memory(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def _record_timing(self, metrics: PerformanceMetrics):
        """Record timing metrics"""
        with self._lock:
            self.timing_data[metrics.function_name].append(metrics)
            self.call_counts[metrics.function_name] += 1
            
            # Limit data retention
            if len(self.timing_data[metrics.function_name]) > self.config.max_records:
                self.timing_data[metrics.function_name] = self.timing_data[metrics.function_name][-self.config.max_records:]
    
    def get_function_stats(self, function_name: str = None) -> Dict[str, Any]:
        """Get statistics for function(s)"""
        with self._lock:
            if function_name:
                if function_name not in self.timing_data:
                    return {}
                
                timings = [m.execution_time for m in self.timing_data[function_name]]
                memory_usage = [m.memory_usage for m in self.timing_data[function_name]]
                
                return {
                    'function_name': function_name,
                    'call_count': self.call_counts[function_name],
                    'total_time': sum(timings),
                    'average_time': np.mean(timings),
                    'min_time': min(timings),
                    'max_time': max(timings),
                    'std_time': np.std(timings),
                    'average_memory': np.mean(memory_usage),
                    'total_memory': sum(memory_usage)
                }
            else:
                # Return stats for all functions
                stats = {}
                for func_name in self.timing_data:
                    stats[func_name] = self.get_function_stats(func_name)
                return stats
    
    def get_slowest_functions(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get the n slowest functions by average execution time"""
        function_stats = self.get_function_stats()
        sorted_functions = sorted(
            [(name, stats['average_time']) for name, stats in function_stats.items()],
            key=lambda x: x[1], reverse=True
        )
        return sorted_functions[:n]


class MemoryProfiler:
    """Memory usage profiler"""
    
    def __init__(self, config: ProfilerConfiguration):
        """Initialize memory profiler"""
        self.config = config
        self.memory_snapshots = deque(maxlen=config.max_records)
        self.peak_memory = 0.0
        self._monitoring_active = False
        self._monitor_thread = None
        
        # Start memory tracing
        if config.enable_memory_profiling:
            tracemalloc.start()
    
    def start_monitoring(self):
        """Start continuous memory monitoring"""
        if not self.config.enable_memory_profiling:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Memory monitoring loop"""
        while self._monitoring_active:
            try:
                # Get current memory usage
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                # Update peak memory
                self.peak_memory = max(self.peak_memory, memory_mb)
                
                # Record snapshot
                snapshot = {
                    'timestamp': time.time(),
                    'rss_mb': memory_mb,
                    'vms_mb': memory_info.vms / (1024 * 1024),
                    'percent': process.memory_percent(),
                    'available_mb': psutil.virtual_memory().available / (1024 * 1024)
                }
                
                # Add tracemalloc data if available
                if tracemalloc.is_tracing():
                    current, peak = tracemalloc.get_traced_memory()
                    snapshot['traced_current_mb'] = current / (1024 * 1024)
                    snapshot['traced_peak_mb'] = peak / (1024 * 1024)
                
                self.memory_snapshots.append(snapshot)
                
                # Check alerts
                if (self.config.enable_alerts and 
                    snapshot['percent'] / 100 > self.config.memory_alert_threshold):
                    logger.warning(f"High memory usage: {snapshot['percent']:.1f}%")
                
                time.sleep(self.config.memory_check_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(5.0)
    
    def take_snapshot(self) -> Dict[str, Any]:
        """Take memory snapshot with detailed breakdown"""
        if not tracemalloc.is_tracing():
            return {}
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        memory_breakdown = {
            'timestamp': time.time(),
            'total_mb': sum(stat.size for stat in top_stats) / (1024 * 1024),
            'top_allocations': []
        }
        
        # Get top 10 memory allocators
        for index, stat in enumerate(top_stats[:10]):
            memory_breakdown['top_allocations'].append({
                'filename': stat.traceback.format()[0],
                'size_mb': stat.size / (1024 * 1024),
                'count': stat.count
            })
        
        return memory_breakdown
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        if not self.memory_snapshots:
            return {}
        
        memory_values = [s['rss_mb'] for s in self.memory_snapshots]
        
        return {
            'current_memory_mb': memory_values[-1] if memory_values else 0,
            'peak_memory_mb': self.peak_memory,
            'average_memory_mb': np.mean(memory_values),
            'min_memory_mb': min(memory_values),
            'max_memory_mb': max(memory_values),
            'memory_growth_mb': memory_values[-1] - memory_values[0] if len(memory_values) > 1 else 0,
            'snapshots_count': len(self.memory_snapshots)
        }


class GPUMonitor:
    """GPU usage monitor (NVIDIA only)"""
    
    def __init__(self, config: ProfilerConfiguration):
        """Initialize GPU monitor"""
        self.config = config
        self.gpu_data = deque(maxlen=config.max_records)
        self.gpu_available = GPU_AVAILABLE
        self._monitoring_active = False
        self._monitor_thread = None
        
        if self.gpu_available:
            self.device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"Found {self.device_count} GPU device(s)")
        else:
            logger.info("GPU monitoring not available")
    
    def start_monitoring(self):
        """Start GPU monitoring"""
        if not self.gpu_available or not self.config.enable_gpu_monitoring:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("GPU monitoring stopped")
    
    def _monitor_loop(self):
        """GPU monitoring loop"""
        while self._monitoring_active:
            try:
                gpu_stats = []
                
                for i in range(self.device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Get utilization
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    # Get memory info
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    # Get temperature
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    # Get power usage
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    except:
                        power = None
                    
                    gpu_stats.append({
                        'device_id': i,
                        'gpu_utilization': utilization.gpu,
                        'memory_utilization': utilization.memory,
                        'memory_used_mb': memory_info.used / (1024 * 1024),
                        'memory_total_mb': memory_info.total / (1024 * 1024),
                        'memory_free_mb': memory_info.free / (1024 * 1024),
                        'temperature_c': temperature,
                        'power_watts': power
                    })
                
                # Record snapshot
                self.gpu_data.append({
                    'timestamp': time.time(),
                    'devices': gpu_stats
                })
                
                time.sleep(self.config.gpu_check_interval)
                
            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")
                time.sleep(5.0)
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU statistics"""
        if not self.gpu_data:
            return {}
        
        latest = self.gpu_data[-1]
        
        stats = {
            'timestamp': latest['timestamp'],
            'device_count': len(latest['devices']),
            'devices': {}
        }
        
        for device in latest['devices']:
            device_id = device['device_id']
            stats['devices'][device_id] = {
                'current_utilization': device['gpu_utilization'],
                'memory_utilization': device['memory_utilization'],
                'memory_used_mb': device['memory_used_mb'],
                'memory_total_mb': device['memory_total_mb'],
                'temperature': device['temperature_c'],
                'power_usage': device['power_watts']
            }
        
        return stats


class SystemMonitor:
    """System-wide resource monitor"""
    
    def __init__(self, config: ProfilerConfiguration):
        """Initialize system monitor"""
        self.config = config
        self.system_data = deque(maxlen=config.max_records)
        self._monitoring_active = False
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start system monitoring"""
        if not self.config.enable_system_monitoring:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """System monitoring loop"""
        while self._monitoring_active:
            try:
                # CPU info
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_count = psutil.cpu_count()
                cpu_freq = psutil.cpu_freq()
                
                # Memory info
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                
                # Disk info
                disk = psutil.disk_usage('/')
                
                # Network info
                network = psutil.net_io_counters()
                
                # Process info
                process = psutil.Process()
                
                snapshot = {
                    'timestamp': time.time(),
                    'cpu': {
                        'percent': cpu_percent,
                        'count': cpu_count,
                        'frequency_mhz': cpu_freq.current if cpu_freq else None
                    },
                    'memory': {
                        'total_mb': memory.total / (1024 * 1024),
                        'available_mb': memory.available / (1024 * 1024),
                        'used_mb': memory.used / (1024 * 1024),
                        'percent': memory.percent
                    },
                    'swap': {
                        'total_mb': swap.total / (1024 * 1024),
                        'used_mb': swap.used / (1024 * 1024),
                        'percent': swap.percent
                    },
                    'disk': {
                        'total_gb': disk.total / (1024**3),
                        'used_gb': disk.used / (1024**3),
                        'free_gb': disk.free / (1024**3),
                        'percent': (disk.used / disk.total) * 100
                    },
                    'network': {
                        'bytes_sent': network.bytes_sent,
                        'bytes_recv': network.bytes_recv,
                        'packets_sent': network.packets_sent,
                        'packets_recv': network.packets_recv
                    },
                    'process': {
                        'cpu_percent': process.cpu_percent(),
                        'memory_mb': process.memory_info().rss / (1024 * 1024),
                        'num_threads': process.num_threads(),
                        'num_fds': process.num_fds() if hasattr(process, 'num_fds') else None
                    }
                }
                
                self.system_data.append(snapshot)
                
                # Check CPU alert
                if (self.config.enable_alerts and 
                    cpu_percent > self.config.cpu_alert_threshold * 100):
                    logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                
                time.sleep(self.config.system_monitor_interval)
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(5.0)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        if not self.system_data:
            return {}
        
        latest = self.system_data[-1]
        
        # Calculate averages over recent data
        recent_data = list(self.system_data)[-60:]  # Last 60 samples
        
        avg_cpu = np.mean([d['cpu']['percent'] for d in recent_data])
        avg_memory = np.mean([d['memory']['percent'] for d in recent_data])
        
        return {
            'timestamp': latest['timestamp'],
            'current': latest,
            'averages': {
                'cpu_percent': avg_cpu,
                'memory_percent': avg_memory,
                'samples': len(recent_data)
            },
            'alerts': {
                'high_cpu': avg_cpu > self.config.cpu_alert_threshold * 100,
                'high_memory': avg_memory > self.config.memory_alert_threshold * 100
            }
        }


class PerformanceOptimizer:
    """Automatic performance optimization"""
    
    def __init__(self, config: ProfilerConfiguration):
        """Initialize performance optimizer"""
        self.config = config
        self.optimization_history = []
        
    def analyze_bottlenecks(self, timing_profiler: TimingProfiler, 
                          memory_profiler: MemoryProfiler) -> List[Dict[str, Any]]:
        """Analyze performance bottlenecks"""
        bottlenecks = []
        
        # Function timing bottlenecks
        slow_functions = timing_profiler.get_slowest_functions(10)
        for func_name, avg_time in slow_functions:
            if avg_time > 0.1:  # Functions taking more than 100ms
                bottlenecks.append({
                    'type': 'slow_function',
                    'function': func_name,
                    'average_time': avg_time,
                    'severity': 'high' if avg_time > 1.0 else 'medium',
                    'recommendation': self._get_timing_recommendation(func_name, avg_time)
                })
        
        # Memory bottlenecks
        memory_stats = memory_profiler.get_memory_stats()
        if memory_stats.get('memory_growth_mb', 0) > 100:  # Growing by more than 100MB
            bottlenecks.append({
                'type': 'memory_growth',
                'growth_mb': memory_stats['memory_growth_mb'],
                'severity': 'high',
                'recommendation': 'Check for memory leaks and optimize data structures'
            })
        
        return bottlenecks
    
    def _get_timing_recommendation(self, func_name: str, avg_time: float) -> str:
        """Get optimization recommendation for slow function"""
        if 'numpy' in func_name.lower() or 'array' in func_name.lower():
            return "Consider vectorization or using more efficient NumPy operations"
        elif 'loop' in func_name.lower() or 'iterate' in func_name.lower():
            return "Consider vectorization, caching, or parallel processing"
        elif 'io' in func_name.lower() or 'read' in func_name.lower() or 'write' in func_name.lower():
            return "Consider asynchronous I/O, buffering, or caching"
        elif avg_time > 5.0:
            return "Consider breaking into smaller functions or using parallel processing"
        else:
            return "Profile individual lines to identify specific bottlenecks"
    
    def suggest_optimizations(self, bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Suggest optimization strategies"""
        suggestions = []
        
        high_severity_count = sum(1 for b in bottlenecks if b['severity'] == 'high')
        
        if high_severity_count > 0:
            suggestions.append(f"Found {high_severity_count} high-severity bottlenecks requiring immediate attention")
        
        # Function-specific suggestions
        slow_functions = [b for b in bottlenecks if b['type'] == 'slow_function']
        if slow_functions:
            suggestions.append("Consider implementing caching for frequently called slow functions")
            suggestions.append("Profile slow functions line-by-line to identify specific bottlenecks")
        
        # Memory suggestions
        memory_issues = [b for b in bottlenecks if b['type'] == 'memory_growth']
        if memory_issues:
            suggestions.append("Implement proper resource cleanup and garbage collection")
            suggestions.append("Consider using memory-efficient data structures")
        
        # General optimization suggestions based on configuration
        if self.config.optimization_level == OptimizationLevel.AGGRESSIVE:
            suggestions.extend([
                "Consider using JIT compilation (e.g., Numba) for compute-intensive functions",
                "Implement parallel processing for independent computations",
                "Use memory mapping for large datasets",
                "Consider using GPU acceleration for mathematical operations"
            ])
        
        return suggestions


class ComprehensiveProfiler:
    """Main profiler orchestrating all monitoring components"""
    
    def __init__(self, config: ProfilerConfiguration):
        """Initialize comprehensive profiler"""
        self.config = config
        
        # Initialize components
        self.timing_profiler = TimingProfiler(config)
        self.memory_profiler = MemoryProfiler(config)
        self.gpu_monitor = GPUMonitor(config)
        self.system_monitor = SystemMonitor(config)
        self.optimizer = PerformanceOptimizer(config)
        
        # State management
        self.is_active = False
        self.start_time = None
        
        # Create output directory
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info("Comprehensive profiler initialized")
    
    def start_profiling(self):
        """Start all profiling components"""
        if self.is_active:
            logger.warning("Profiler already active")
            return
        
        self.is_active = True
        self.start_time = time.time()
        
        # Start all monitors
        self.memory_profiler.start_monitoring()
        self.gpu_monitor.start_monitoring()
        self.system_monitor.start_monitoring()
        
        logger.info("Profiling started")
    
    def stop_profiling(self):
        """Stop all profiling components"""
        if not self.is_active:
            return
        
        # Stop all monitors
        self.memory_profiler.stop_monitoring()
        self.gpu_monitor.stop_monitoring()
        self.system_monitor.stop_monitoring()
        
        self.is_active = False
        
        # Generate final report
        if self.config.auto_export:
            self.export_performance_report()
        
        logger.info("Profiling stopped")
    
    def profile_function(self, func_name: str = None):
        """Decorator for profiling functions"""
        return self.timing_profiler.profile_function(func_name)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        runtime = time.time() - (self.start_time or time.time())
        
        stats = {
            'profiler_info': {
                'runtime_seconds': runtime,
                'is_active': self.is_active,
                'config': {
                    'optimization_level': self.config.optimization_level.name,
                    'enable_function_timing': self.config.enable_function_timing,
                    'enable_memory_profiling': self.config.enable_memory_profiling,
                    'enable_gpu_monitoring': self.config.enable_gpu_monitoring
                }
            },
            'function_timing': self.timing_profiler.get_function_stats(),
            'memory_usage': self.memory_profiler.get_memory_stats(),
            'gpu_usage': self.gpu_monitor.get_gpu_stats(),
            'system_resources': self.system_monitor.get_system_stats()
        }
        
        return stats
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Perform comprehensive performance analysis"""
        # Get bottlenecks
        bottlenecks = self.optimizer.analyze_bottlenecks(
            self.timing_profiler, 
            self.memory_profiler
        )
        
        # Get optimization suggestions
        suggestions = self.optimizer.suggest_optimizations(bottlenecks)
        
        # Get comprehensive stats
        stats = self.get_comprehensive_stats()
        
        analysis = {
            'analysis_timestamp': time.time(),
            'bottlenecks': bottlenecks,
            'optimization_suggestions': suggestions,
            'performance_stats': stats,
            'summary': {
                'total_bottlenecks': len(bottlenecks),
                'high_severity_issues': sum(1 for b in bottlenecks if b['severity'] == 'high'),
                'functions_profiled': len(self.timing_profiler.timing_data),
                'total_function_calls': sum(self.timing_profiler.call_counts.values())
            }
        }
        
        return analysis
    
    def export_performance_report(self, filename: str = None) -> str:
        """Export comprehensive performance report"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Generate analysis
        analysis = self.analyze_performance()
        
        # Add metadata
        analysis['report_metadata'] = {
            'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'profiler_version': '1.0',
            'python_version': sys.version,
            'platform': sys.platform
        }
        
        # Save report
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Performance report exported to {filepath}")
        return str(filepath)


# Convenience functions and decorators
def create_profiler(optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
                   output_dir: str = "performance_data") -> ComprehensiveProfiler:
    """Create profiler with default configuration"""
    config = ProfilerConfiguration(
        optimization_level=optimization_level,
        output_directory=output_dir,
        enable_alerts=True,
        auto_export=True
    )
    return ComprehensiveProfiler(config)


# Global profiler instance for easy access
_global_profiler = None


def get_global_profiler() -> ComprehensiveProfiler:
    """Get global profiler instance"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = create_profiler()
    return _global_profiler


def profile_function(func_name: str = None):
    """Global function profiler decorator"""
    return get_global_profiler().profile_function(func_name)


def start_global_profiling():
    """Start global profiling"""
    get_global_profiler().start_profiling()


def stop_global_profiling() -> str:
    """Stop global profiling and return report path"""
    profiler = get_global_profiler()
    profiler.stop_profiling()
    return profiler.export_performance_report()


# Example usage and testing
if __name__ == "__main__":
    # Test comprehensive profiler
    logger.info("Testing Comprehensive Performance Profiler")
    
    # Create profiler
    config = ProfilerConfiguration(
        output_directory="test_performance",
        enable_gpu_monitoring=GPU_AVAILABLE,
        optimization_level=OptimizationLevel.AGGRESSIVE
    )
    
    profiler = ComprehensiveProfiler(config)
    
    # Start profiling
    profiler.start_profiling()
    
    # Define test functions
    @profiler.profile_function("test_computation")
    def slow_computation():
        """Simulate slow computation"""
        result = 0
        for i in range(1000000):
            result += i * i
        return result
    
    @profiler.profile_function("test_memory_allocation")
    def memory_intensive():
        """Simulate memory-intensive operation"""
        big_list = [i for i in range(100000)]
        big_array = np.random.random((1000, 1000))
        return len(big_list) + big_array.sum()
    
    @profiler.profile_function("test_nested_calls")
    def nested_function_test():
        """Test nested function calls"""
        result1 = slow_computation()
        result2 = memory_intensive()
        return result1 + result2
    
    try:
        # Run test functions
        logger.info("Running test functions...")
        
        for i in range(5):
            slow_computation()
            memory_intensive()
            nested_function_test()
            time.sleep(0.5)  # Allow monitoring to collect data
        
        # Get analysis
        analysis = profiler.analyze_performance()
        
        logger.info("Performance analysis completed!")
        logger.info(f"Functions profiled: {analysis['summary']['functions_profiled']}")
        logger.info(f"Total function calls: {analysis['summary']['total_function_calls']}")
        logger.info(f"Bottlenecks found: {analysis['summary']['total_bottlenecks']}")
        
        # Print bottlenecks
        for bottleneck in analysis['bottlenecks']:
            logger.info(f"Bottleneck: {bottleneck['type']} - {bottleneck.get('function', 'N/A')} - {bottleneck['severity']}")
        
        # Print suggestions
        for suggestion in analysis['optimization_suggestions']:
            logger.info(f"Suggestion: {suggestion}")
        
        # Export report
        report_path = profiler.export_performance_report()
        logger.info(f"Report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Profiling test failed: {e}")
    finally:
        profiler.stop_profiling()
    
    print("Performance profiler test completed!")