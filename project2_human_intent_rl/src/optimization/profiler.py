"""
Computational Bottleneck Analysis and System Profiler
Model-Based RL Human Intent Recognition System

This module provides comprehensive system profiling capabilities for identifying
computational bottlenecks, memory usage patterns, GPU utilization, and I/O performance.
"""

import time
import psutil
import threading
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import cProfile
import pstats
import io
import tracemalloc
import gc
import sys
import os
from contextlib import contextmanager
import functools
import json
from datetime import datetime
import logging

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ProfilerConfig:
    """Configuration for system profiling."""
    enable_line_profiling: bool = True
    enable_memory_profiling: bool = True
    enable_gpu_profiling: bool = True
    enable_io_profiling: bool = True
    sampling_interval: float = 0.1  # seconds
    memory_threshold_mb: float = 100.0  # Alert threshold
    cpu_threshold_percent: float = 80.0  # Alert threshold
    profile_duration: Optional[float] = None  # None for manual control


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_peak_mb: float = 0.0
    gpu_usage_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    io_read_mb: float = 0.0
    io_write_mb: float = 0.0
    function_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    bottlenecks: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)


@dataclass
class ProfilingResult:
    """Container for profiling results."""
    function_name: str
    metrics: PerformanceMetrics
    detailed_stats: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class LineProfiler:
    """Line-by-line execution profiler."""
    
    def __init__(self):
        self.profiler = None
        self.stats = {}
        
    def start_profiling(self):
        """Start line-by-line profiling."""
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return statistics."""
        if self.profiler:
            self.profiler.disable()
            
            # Get statistics
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats()
            
            # Parse statistics
            stats_output = s.getvalue()
            
            # Get top functions by time
            ps.sort_stats('tottime')
            top_functions = []
            for func_info in ps.stats.items():
                func_name = f"{func_info[0][0]}:{func_info[0][1]}({func_info[0][2]})"
                stats = func_info[1]
                top_functions.append({
                    'function': func_name,
                    'calls': stats[0],
                    'total_time': stats[2],
                    'cumulative_time': stats[3],
                    'per_call': stats[2] / stats[0] if stats[0] > 0 else 0
                })
            
            # Sort by total time descending
            top_functions.sort(key=lambda x: x['total_time'], reverse=True)
            
            return {
                'raw_output': stats_output,
                'top_functions': top_functions[:20],  # Top 20 functions
                'total_calls': sum(func[1][0] for func in ps.stats.values()),
                'total_time': sum(func[1][2] for func in ps.stats.values())
            }
        
        return {}


class MemoryProfiler:
    """Memory usage and allocation profiler."""
    
    def __init__(self):
        self.start_snapshot = None
        self.peak_memory = 0.0
        self.memory_timeline = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_profiling(self):
        """Start memory profiling."""
        tracemalloc.start()
        self.start_snapshot = tracemalloc.take_snapshot()
        self.peak_memory = 0.0
        self.memory_timeline = []
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory)
        self.monitor_thread.start()
        
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop memory profiling and return statistics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
        if self.start_snapshot:
            current_snapshot = tracemalloc.take_snapshot()
            
            # Calculate memory growth
            top_stats = current_snapshot.compare_to(
                self.start_snapshot, 'lineno'
            )
            
            # Get top memory consumers
            top_memory = []
            for stat in top_stats[:10]:
                top_memory.append({
                    'file': stat.traceback.format()[0] if stat.traceback.format() else 'unknown',
                    'size_diff_mb': stat.size_diff / (1024 * 1024),
                    'count_diff': stat.count_diff,
                    'size_mb': stat.size / (1024 * 1024)
                })
            
            # Current memory usage
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            tracemalloc.stop()
            
            return {
                'peak_memory_mb': self.peak_memory,
                'current_memory_mb': current_memory,
                'memory_timeline': self.memory_timeline,
                'top_allocations': top_memory,
                'total_allocations': len(current_snapshot.traces)
            }
        
        return {}
    
    def _monitor_memory(self):
        """Monitor memory usage in background thread."""
        while self.monitoring:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                self.peak_memory = max(self.peak_memory, memory_mb)
                
                self.memory_timeline.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_mb
                })
                
                time.sleep(0.1)
            except Exception:
                break


class GPUProfiler:
    """GPU utilization and memory profiler."""
    
    def __init__(self):
        self.gpu_timeline = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_profiling(self):
        """Start GPU profiling."""
        if not GPU_AVAILABLE:
            return
            
        self.gpu_timeline = []
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_gpu)
        self.monitor_thread.start()
        
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop GPU profiling and return statistics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
        if not GPU_AVAILABLE or not self.gpu_timeline:
            return {'gpu_available': False}
        
        # Calculate statistics
        gpu_usage = [entry['usage'] for entry in self.gpu_timeline]
        memory_usage = [entry['memory_used'] for entry in self.gpu_timeline]
        
        return {
            'gpu_available': True,
            'avg_gpu_usage': np.mean(gpu_usage),
            'max_gpu_usage': np.max(gpu_usage),
            'avg_memory_usage_mb': np.mean(memory_usage),
            'max_memory_usage_mb': np.max(memory_usage),
            'timeline': self.gpu_timeline
        }
    
    def _monitor_gpu(self):
        """Monitor GPU usage in background thread."""
        while self.monitoring:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Monitor first GPU
                    self.gpu_timeline.append({
                        'timestamp': time.time(),
                        'usage': gpu.load * 100,
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'temperature': gpu.temperature
                    })
                
                time.sleep(0.1)
            except Exception:
                break


class IOProfiler:
    """I/O performance profiler."""
    
    def __init__(self):
        self.start_counters = None
        self.io_timeline = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_profiling(self):
        """Start I/O profiling."""
        process = psutil.Process()
        self.start_counters = process.io_counters()
        self.io_timeline = []
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_io)
        self.monitor_thread.start()
        
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop I/O profiling and return statistics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
        if self.start_counters:
            process = psutil.Process()
            end_counters = process.io_counters()
            
            read_mb = (end_counters.read_bytes - self.start_counters.read_bytes) / (1024 * 1024)
            write_mb = (end_counters.write_bytes - self.start_counters.write_bytes) / (1024 * 1024)
            
            return {
                'total_read_mb': read_mb,
                'total_write_mb': write_mb,
                'read_count': end_counters.read_count - self.start_counters.read_count,
                'write_count': end_counters.write_count - self.start_counters.write_count,
                'timeline': self.io_timeline
            }
        
        return {}
    
    def _monitor_io(self):
        """Monitor I/O performance in background thread."""
        while self.monitoring:
            try:
                process = psutil.Process()
                counters = process.io_counters()
                
                self.io_timeline.append({
                    'timestamp': time.time(),
                    'read_bytes': counters.read_bytes,
                    'write_bytes': counters.write_bytes,
                    'read_count': counters.read_count,
                    'write_count': counters.write_count
                })
                
                time.sleep(0.1)
            except Exception:
                break


class BottleneckAnalyzer:
    """Analyzes profiling results to identify bottlenecks."""
    
    @staticmethod
    def analyze_cpu_bottlenecks(line_stats: Dict[str, Any], 
                               threshold_ms: float = 10.0) -> List[str]:
        """Identify CPU bottlenecks from line profiling."""
        bottlenecks = []
        
        if 'top_functions' in line_stats:
            for func in line_stats['top_functions']:
                if func['total_time'] * 1000 > threshold_ms:
                    bottlenecks.append(
                        f"CPU bottleneck: {func['function']} "
                        f"({func['total_time']*1000:.1f}ms, {func['calls']} calls)"
                    )
        
        return bottlenecks
    
    @staticmethod
    def analyze_memory_bottlenecks(memory_stats: Dict[str, Any], 
                                 threshold_mb: float = 100.0) -> List[str]:
        """Identify memory bottlenecks."""
        bottlenecks = []
        
        if memory_stats.get('peak_memory_mb', 0) > threshold_mb:
            bottlenecks.append(
                f"Memory bottleneck: Peak usage {memory_stats['peak_memory_mb']:.1f}MB"
            )
        
        if 'top_allocations' in memory_stats:
            for alloc in memory_stats['top_allocations'][:3]:
                if alloc['size_diff_mb'] > threshold_mb / 10:
                    bottlenecks.append(
                        f"Memory allocation bottleneck: {alloc['file']} "
                        f"({alloc['size_diff_mb']:.1f}MB growth)"
                    )
        
        return bottlenecks
    
    @staticmethod
    def analyze_gpu_bottlenecks(gpu_stats: Dict[str, Any], 
                              usage_threshold: float = 90.0,
                              memory_threshold: float = 80.0) -> List[str]:
        """Identify GPU bottlenecks."""
        bottlenecks = []
        
        if not gpu_stats.get('gpu_available', False):
            return bottlenecks
        
        if gpu_stats.get('max_gpu_usage', 0) > usage_threshold:
            bottlenecks.append(
                f"GPU utilization bottleneck: {gpu_stats['max_gpu_usage']:.1f}% peak usage"
            )
        
        if 'max_memory_usage_mb' in gpu_stats and 'timeline' in gpu_stats:
            for entry in gpu_stats['timeline']:
                if entry.get('memory_total', 1) > 0:
                    memory_percent = (entry['memory_used'] / entry['memory_total']) * 100
                    if memory_percent > memory_threshold:
                        bottlenecks.append(
                            f"GPU memory bottleneck: {memory_percent:.1f}% memory usage"
                        )
                        break
        
        return bottlenecks
    
    @staticmethod
    def generate_optimization_suggestions(bottlenecks: List[str]) -> List[str]:
        """Generate optimization suggestions based on bottlenecks."""
        suggestions = []
        
        for bottleneck in bottlenecks:
            if "CPU bottleneck" in bottleneck:
                suggestions.extend([
                    "Consider algorithm optimization or parallel processing",
                    "Profile individual functions for further optimization",
                    "Implement caching for repeated computations"
                ])
            elif "Memory bottleneck" in bottleneck:
                suggestions.extend([
                    "Implement memory pooling for frequent allocations",
                    "Use more efficient data structures",
                    "Consider streaming processing for large datasets"
                ])
            elif "GPU utilization bottleneck" in bottleneck:
                suggestions.extend([
                    "Optimize GPU kernel efficiency",
                    "Consider batch processing optimization",
                    "Balance CPU-GPU workload distribution"
                ])
            elif "GPU memory bottleneck" in bottleneck:
                suggestions.extend([
                    "Reduce batch sizes or model complexity",
                    "Implement gradient checkpointing",
                    "Use mixed precision training"
                ])
        
        return list(set(suggestions))  # Remove duplicates


class SystemProfiler:
    """Main system profiler orchestrating all profiling components."""
    
    def __init__(self, config: ProfilerConfig = None):
        self.config = config or ProfilerConfig()
        self.line_profiler = LineProfiler()
        self.memory_profiler = MemoryProfiler()
        self.gpu_profiler = GPUProfiler()
        self.io_profiler = IOProfiler()
        self.analyzer = BottleneckAnalyzer()
        
        self.profiling_active = False
        self.start_time = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def start_profiling(self):
        """Start comprehensive system profiling."""
        self.start_time = time.time()
        self.profiling_active = True
        
        self.logger.info("Starting comprehensive system profiling...")
        
        if self.config.enable_line_profiling:
            self.line_profiler.start_profiling()
            
        if self.config.enable_memory_profiling:
            self.memory_profiler.start_profiling()
            
        if self.config.enable_gpu_profiling:
            self.gpu_profiler.start_profiling()
            
        if self.config.enable_io_profiling:
            self.io_profiler.start_profiling()
    
    def stop_profiling(self) -> ProfilingResult:
        """Stop profiling and return comprehensive results."""
        if not self.profiling_active:
            raise RuntimeError("Profiling not active")
        
        execution_time = time.time() - self.start_time
        self.profiling_active = False
        
        self.logger.info("Stopping profiling and analyzing results...")
        
        # Collect results from all profilers
        line_stats = {}
        memory_stats = {}
        gpu_stats = {}
        io_stats = {}
        
        if self.config.enable_line_profiling:
            line_stats = self.line_profiler.stop_profiling()
            
        if self.config.enable_memory_profiling:
            memory_stats = self.memory_profiler.stop_profiling()
            
        if self.config.enable_gpu_profiling:
            gpu_stats = self.gpu_profiler.stop_profiling()
            
        if self.config.enable_io_profiling:
            io_stats = self.io_profiler.stop_profiling()
        
        # Analyze bottlenecks
        bottlenecks = []
        bottlenecks.extend(self.analyzer.analyze_cpu_bottlenecks(line_stats))
        bottlenecks.extend(self.analyzer.analyze_memory_bottlenecks(memory_stats))
        bottlenecks.extend(self.analyzer.analyze_gpu_bottlenecks(gpu_stats))
        
        # Generate optimization suggestions
        suggestions = self.analyzer.generate_optimization_suggestions(bottlenecks)
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            cpu_usage_percent=psutil.cpu_percent(),
            memory_usage_mb=memory_stats.get('current_memory_mb', 0),
            memory_peak_mb=memory_stats.get('peak_memory_mb', 0),
            gpu_usage_percent=gpu_stats.get('avg_gpu_usage', 0),
            gpu_memory_mb=gpu_stats.get('avg_memory_usage_mb', 0),
            io_read_mb=io_stats.get('total_read_mb', 0),
            io_write_mb=io_stats.get('total_write_mb', 0),
            function_calls=line_stats.get('total_calls', 0),
            bottlenecks=bottlenecks,
            optimization_suggestions=suggestions
        )
        
        # Create detailed stats
        detailed_stats = {
            'line_profiling': line_stats,
            'memory_profiling': memory_stats,
            'gpu_profiling': gpu_stats,
            'io_profiling': io_stats,
            'system_info': self._get_system_info()
        }
        
        return ProfilingResult(
            function_name="system_wide",
            metrics=metrics,
            detailed_stats=detailed_stats
        )
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, ProfilingResult]:
        """Profile a specific function call."""
        self.start_profiling()
        
        try:
            result = func(*args, **kwargs)
            profiling_result = self.stop_profiling()
            profiling_result.function_name = func.__name__
            return result, profiling_result
        except Exception as e:
            if self.profiling_active:
                self.stop_profiling()
            raise e
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'platform': sys.platform,
            'python_version': sys.version,
            'gpu_available': GPU_AVAILABLE,
            'torch_available': TORCH_AVAILABLE
        }
    
    @contextmanager
    def profile_context(self):
        """Context manager for profiling code blocks."""
        self.start_profiling()
        try:
            yield self
        finally:
            if self.profiling_active:
                result = self.stop_profiling()
                self.logger.info(f"Profiling completed: {len(result.metrics.bottlenecks)} bottlenecks found")
    
    def save_results(self, result: ProfilingResult, filepath: str):
        """Save profiling results to file."""
        # Convert to serializable format
        data = {
            'function_name': result.function_name,
            'timestamp': result.timestamp.isoformat(),
            'metrics': {
                'execution_time': result.metrics.execution_time,
                'cpu_usage_percent': result.metrics.cpu_usage_percent,
                'memory_usage_mb': result.metrics.memory_usage_mb,
                'memory_peak_mb': result.metrics.memory_peak_mb,
                'gpu_usage_percent': result.metrics.gpu_usage_percent,
                'gpu_memory_mb': result.metrics.gpu_memory_mb,
                'io_read_mb': result.metrics.io_read_mb,
                'io_write_mb': result.metrics.io_write_mb,
                'function_calls': result.metrics.function_calls,
                'bottlenecks': result.metrics.bottlenecks,
                'optimization_suggestions': result.metrics.optimization_suggestions
            },
            'detailed_stats': result.detailed_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Profiling results saved to {filepath}")
    
    def generate_report(self, result: ProfilingResult) -> str:
        """Generate a comprehensive profiling report."""
        report = []
        report.append("=" * 80)
        report.append("SYSTEM PERFORMANCE PROFILING REPORT")
        report.append("=" * 80)
        report.append(f"Function: {result.function_name}")
        report.append(f"Timestamp: {result.timestamp}")
        report.append(f"Execution Time: {result.metrics.execution_time:.3f}s")
        report.append("")
        
        # Performance Summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 40)
        report.append(f"CPU Usage: {result.metrics.cpu_usage_percent:.1f}%")
        report.append(f"Memory Usage: {result.metrics.memory_usage_mb:.1f}MB")
        report.append(f"Memory Peak: {result.metrics.memory_peak_mb:.1f}MB")
        report.append(f"GPU Usage: {result.metrics.gpu_usage_percent:.1f}%")
        report.append(f"GPU Memory: {result.metrics.gpu_memory_mb:.1f}MB")
        report.append(f"I/O Read: {result.metrics.io_read_mb:.1f}MB")
        report.append(f"I/O Write: {result.metrics.io_write_mb:.1f}MB")
        report.append(f"Function Calls: {result.metrics.function_calls}")
        report.append("")
        
        # Bottlenecks
        if result.metrics.bottlenecks:
            report.append("IDENTIFIED BOTTLENECKS")
            report.append("-" * 40)
            for bottleneck in result.metrics.bottlenecks:
                report.append(f"• {bottleneck}")
            report.append("")
        
        # Optimization Suggestions
        if result.metrics.optimization_suggestions:
            report.append("OPTIMIZATION SUGGESTIONS")
            report.append("-" * 40)
            for suggestion in result.metrics.optimization_suggestions:
                report.append(f"• {suggestion}")
            report.append("")
        
        # Top Functions
        if 'line_profiling' in result.detailed_stats:
            line_stats = result.detailed_stats['line_profiling']
            if 'top_functions' in line_stats:
                report.append("TOP TIME-CONSUMING FUNCTIONS")
                report.append("-" * 40)
                for func in line_stats['top_functions'][:10]:
                    report.append(
                        f"{func['function']}: {func['total_time']*1000:.1f}ms "
                        f"({func['calls']} calls)"
                    )
                report.append("")
        
        return "\n".join(report)


def profile_decorator(config: ProfilerConfig = None):
    """Decorator for profiling function calls."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = SystemProfiler(config)
            result, profiling_result = profiler.profile_function(func, *args, **kwargs)
            
            # Log results
            logging.getLogger(__name__).info(
                f"Profiled {func.__name__}: "
                f"{profiling_result.metrics.execution_time:.3f}s, "
                f"{len(profiling_result.metrics.bottlenecks)} bottlenecks"
            )
            
            return result
        return wrapper
    return decorator


# Example usage and testing functions
def create_example_workload():
    """Create example computational workload for testing."""
    # CPU-intensive task
    def cpu_intensive():
        return sum(i**2 for i in range(100000))
    
    # Memory-intensive task
    def memory_intensive():
        data = []
        for i in range(10000):
            data.append(np.random.random((100, 100)))
        return len(data)
    
    # I/O intensive task
    def io_intensive():
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            for i in range(10000):
                f.write(f"Line {i}: " + "x" * 100 + "\n")
            temp_path = f.name
        
        # Read it back
        with open(temp_path, 'r') as f:
            lines = f.readlines()
        
        os.unlink(temp_path)
        return len(lines)
    
    return cpu_intensive, memory_intensive, io_intensive


if __name__ == "__main__":
    # Example usage
    profiler = SystemProfiler()
    
    # Create test workload
    cpu_task, memory_task, io_task = create_example_workload()
    
    # Profile different workloads
    print("Profiling CPU-intensive workload...")
    _, cpu_result = profiler.profile_function(cpu_task)
    
    print("Profiling memory-intensive workload...")
    _, memory_result = profiler.profile_function(memory_task)
    
    print("Profiling I/O-intensive workload...")
    _, io_result = profiler.profile_function(io_task)
    
    # Generate reports
    print("\nCPU Task Report:")
    print(profiler.generate_report(cpu_result))
    
    print("\nMemory Task Report:")
    print(profiler.generate_report(memory_result))
    
    print("\nI/O Task Report:")
    print(profiler.generate_report(io_result))