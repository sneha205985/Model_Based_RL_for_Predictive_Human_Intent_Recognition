"""
System-Level Performance Optimization

This module provides system-level optimization utilities including
automatic parameter tuning, resource allocation optimization,
and runtime performance enhancements for the HRI Bayesian RL system.

Features:
- Automatic hyperparameter optimization
- Dynamic resource allocation
- Runtime system optimization
- Cache management and optimization
- Thread pool optimization
- Memory management strategies

Author: Phase 5 Implementation
Date: 2024
"""

import numpy as np
import threading
import multiprocessing
import gc
import os
import sys
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
from collections import defaultdict, LRUCache
import functools
import weakref

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategies"""
    CONSERVATIVE = auto()
    BALANCED = auto()
    AGGRESSIVE = auto()
    MEMORY_OPTIMIZED = auto()
    SPEED_OPTIMIZED = auto()
    ADAPTIVE = auto()


class ResourceType(Enum):
    """Types of system resources"""
    CPU = auto()
    MEMORY = auto()
    DISK = auto()
    NETWORK = auto()
    GPU = auto()


@dataclass
class OptimizationConfig:
    """Configuration for system optimization"""
    # General settings
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    auto_optimize: bool = True
    optimization_interval: float = 60.0  # seconds
    
    # Threading optimization
    max_threads: Optional[int] = None  # Auto-detect if None
    thread_pool_size: Optional[int] = None
    enable_thread_optimization: bool = True
    
    # Memory optimization
    enable_memory_optimization: bool = True
    gc_threshold: float = 0.8  # Trigger GC at 80% memory usage
    cache_size_limit: int = 1000  # Maximum items in caches
    memory_pool_size: Optional[int] = None
    
    # CPU optimization
    enable_cpu_optimization: bool = True
    cpu_affinity: Optional[List[int]] = None
    process_priority: str = "normal"  # "low", "normal", "high"
    
    # I/O optimization
    enable_io_optimization: bool = True
    io_buffer_size: int = 65536  # 64KB
    async_io_enabled: bool = True
    
    # Cache optimization
    enable_caching: bool = True
    cache_strategy: str = "lru"  # "lru", "lfu", "fifo"
    cache_stats_enabled: bool = True
    
    # Resource monitoring
    resource_monitor_interval: float = 5.0
    resource_threshold_cpu: float = 0.8
    resource_threshold_memory: float = 0.8


class SmartCache:
    """Intelligent caching system with automatic optimization"""
    
    def __init__(self, max_size: int = 1000, strategy: str = "lru"):
        """Initialize smart cache"""
        self.max_size = max_size
        self.strategy = strategy
        self.cache = {}
        self.access_counts = defaultdict(int)
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.RLock()
        
    def get(self, key: str, default=None):
        """Get item from cache"""
        with self._lock:
            if key in self.cache:
                self.hit_count += 1
                self.access_counts[key] += 1
                self.access_times[key] = time.time()
                return self.cache[key]
            else:
                self.miss_count += 1
                return default
    
    def put(self, key: str, value: Any):
        """Put item in cache"""
        with self._lock:
            if len(self.cache) >= self.max_size:
                self._evict_item()
            
            self.cache[key] = value
            self.access_counts[key] += 1
            self.access_times[key] = time.time()
    
    def _evict_item(self):
        """Evict item based on strategy"""
        if not self.cache:
            return
        
        if self.strategy == "lru":
            # Remove least recently used
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        elif self.strategy == "lfu":
            # Remove least frequently used
            oldest_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
        else:  # fifo
            # Remove first inserted (oldest in cache)
            oldest_key = next(iter(self.cache))
        
        del self.cache[oldest_key]
        del self.access_counts[oldest_key]
        del self.access_times[oldest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    def clear(self):
        """Clear cache"""
        with self._lock:
            self.cache.clear()
            self.access_counts.clear()
            self.access_times.clear()
            self.hit_count = 0
            self.miss_count = 0


class ThreadPoolOptimizer:
    """Optimizes thread pool configuration based on workload"""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize thread pool optimizer"""
        self.config = config
        self.optimal_size = self._calculate_optimal_pool_size()
        self.thread_pool = None
        self.performance_history = []
        
    def _calculate_optimal_pool_size(self) -> int:
        """Calculate optimal thread pool size"""
        cpu_count = multiprocessing.cpu_count()
        
        if self.config.thread_pool_size:
            return self.config.thread_pool_size
        
        if self.config.strategy == OptimizationStrategy.MEMORY_OPTIMIZED:
            return min(cpu_count, 4)  # Conservative for memory
        elif self.config.strategy == OptimizationStrategy.SPEED_OPTIMIZED:
            return min(cpu_count * 2, 32)  # Aggressive for speed
        else:  # Balanced
            return min(cpu_count + 2, 16)
    
    def get_optimized_thread_pool(self) -> ThreadPoolExecutor:
        """Get optimized thread pool"""
        if self.thread_pool is None:
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.optimal_size,
                thread_name_prefix="HRI_Optimized"
            )
        return self.thread_pool
    
    def shutdown(self):
        """Shutdown thread pool"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)


class MemoryOptimizer:
    """Memory usage optimization and management"""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize memory optimizer"""
        self.config = config
        self.memory_pools = {}
        self.weak_references = weakref.WeakSet()
        
    def optimize_garbage_collection(self):
        """Optimize garbage collection based on memory usage"""
        memory_percent = psutil.virtual_memory().percent / 100.0
        
        if memory_percent > self.config.gc_threshold:
            # Force garbage collection
            collected = gc.collect()
            logger.info(f"Forced garbage collection: {collected} objects collected")
            
            # Adjust GC thresholds for more aggressive collection
            current = gc.get_threshold()
            gc.set_threshold(
                max(100, current[0] // 2),
                max(10, current[1] // 2),
                max(10, current[2] // 2)
            )
    
    def create_object_pool(self, pool_name: str, factory_func: Callable, 
                          initial_size: int = 10) -> 'ObjectPool':
        """Create object pool for expensive objects"""
        if pool_name not in self.memory_pools:
            self.memory_pools[pool_name] = ObjectPool(factory_func, initial_size)
        return self.memory_pools[pool_name]
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """Monitor current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'process_memory_mb': memory_info.rss / (1024 * 1024),
            'process_memory_percent': process.memory_percent(),
            'system_memory_percent': system_memory.percent,
            'system_available_mb': system_memory.available / (1024 * 1024)
        }
    
    def suggest_memory_optimizations(self, memory_stats: Dict[str, float]) -> List[str]:
        """Suggest memory optimizations based on usage"""
        suggestions = []
        
        if memory_stats['process_memory_percent'] > 50:
            suggestions.append("Consider implementing object pooling for frequently created objects")
            suggestions.append("Review data structures for memory efficiency")
        
        if memory_stats['system_memory_percent'] > 80:
            suggestions.append("System memory usage is high - consider reducing cache sizes")
            suggestions.append("Enable more aggressive garbage collection")
        
        return suggestions


class ObjectPool:
    """Generic object pool for memory optimization"""
    
    def __init__(self, factory_func: Callable, initial_size: int = 10):
        """Initialize object pool"""
        self.factory_func = factory_func
        self.pool = []
        self.in_use = set()
        self._lock = threading.Lock()
        
        # Pre-populate pool
        for _ in range(initial_size):
            self.pool.append(factory_func())
    
    def acquire(self):
        """Acquire object from pool"""
        with self._lock:
            if self.pool:
                obj = self.pool.pop()
            else:
                obj = self.factory_func()
            
            self.in_use.add(id(obj))
            return obj
    
    def release(self, obj):
        """Release object back to pool"""
        with self._lock:
            obj_id = id(obj)
            if obj_id in self.in_use:
                self.in_use.remove(obj_id)
                self.pool.append(obj)
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics"""
        with self._lock:
            return {
                'available': len(self.pool),
                'in_use': len(self.in_use),
                'total_created': len(self.pool) + len(self.in_use)
            }


class CPUOptimizer:
    """CPU utilization optimization"""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize CPU optimizer"""
        self.config = config
        self.cpu_count = multiprocessing.cpu_count()
        self.process = psutil.Process()
        
    def optimize_cpu_affinity(self):
        """Optimize CPU affinity based on configuration"""
        if not self.config.cpu_affinity:
            return
        
        try:
            self.process.cpu_affinity(self.config.cpu_affinity)
            logger.info(f"CPU affinity set to: {self.config.cpu_affinity}")
        except Exception as e:
            logger.warning(f"Failed to set CPU affinity: {e}")
    
    def optimize_process_priority(self):
        """Optimize process priority"""
        try:
            if self.config.process_priority == "high":
                priority = psutil.HIGH_PRIORITY_CLASS if sys.platform == "win32" else -10
            elif self.config.process_priority == "low":
                priority = psutil.IDLE_PRIORITY_CLASS if sys.platform == "win32" else 10
            else:
                return  # Normal priority, no change needed
            
            self.process.nice(priority)
            logger.info(f"Process priority set to: {self.config.process_priority}")
        except Exception as e:
            logger.warning(f"Failed to set process priority: {e}")
    
    def get_cpu_optimization_recommendations(self) -> List[str]:
        """Get CPU optimization recommendations"""
        recommendations = []
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent > 80:
            recommendations.append("High CPU usage detected - consider parallel processing")
            recommendations.append("Profile code to identify CPU bottlenecks")
        
        if self.cpu_count > 4 and not self.config.cpu_affinity:
            recommendations.append("Consider setting CPU affinity for better cache locality")
        
        return recommendations


class SystemOptimizer:
    """Main system optimizer coordinating all optimization components"""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize system optimizer"""
        self.config = config
        
        # Initialize components
        self.cache_manager = {}
        self.thread_optimizer = ThreadPoolOptimizer(config)
        self.memory_optimizer = MemoryOptimizer(config)
        self.cpu_optimizer = CPUOptimizer(config)
        
        # Optimization state
        self.optimization_active = False
        self.optimization_thread = None
        self.optimization_stats = {
            'optimizations_applied': 0,
            'last_optimization': None,
            'performance_improvements': []
        }
        
        logger.info("System optimizer initialized")
    
    def start_optimization(self):
        """Start automatic system optimization"""
        if self.optimization_active:
            logger.warning("Optimization already active")
            return
        
        self.optimization_active = True
        
        # Apply initial optimizations
        self._apply_static_optimizations()
        
        # Start continuous optimization thread
        if self.config.auto_optimize:
            self.optimization_thread = threading.Thread(
                target=self._optimization_loop, 
                daemon=True
            )
            self.optimization_thread.start()
        
        logger.info("System optimization started")
    
    def stop_optimization(self):
        """Stop system optimization"""
        self.optimization_active = False
        
        if self.optimization_thread:
            self.optimization_thread.join(timeout=10)
        
        # Cleanup resources
        self.thread_optimizer.shutdown()
        
        logger.info("System optimization stopped")
    
    def _apply_static_optimizations(self):
        """Apply one-time static optimizations"""
        try:
            # CPU optimizations
            if self.config.enable_cpu_optimization:
                self.cpu_optimizer.optimize_cpu_affinity()
                self.cpu_optimizer.optimize_process_priority()
            
            # Python-specific optimizations
            self._optimize_python_settings()
            
            self.optimization_stats['optimizations_applied'] += 1
            self.optimization_stats['last_optimization'] = time.time()
            
        except Exception as e:
            logger.error(f"Static optimization failed: {e}")
    
    def _optimize_python_settings(self):
        """Optimize Python interpreter settings"""
        # Optimize hash randomization for better performance
        if 'PYTHONHASHSEED' not in os.environ:
            os.environ['PYTHONHASHSEED'] = '0'
        
        # Set garbage collection thresholds based on strategy
        if self.config.strategy == OptimizationStrategy.MEMORY_OPTIMIZED:
            gc.set_threshold(100, 10, 10)  # More frequent GC
        elif self.config.strategy == OptimizationStrategy.SPEED_OPTIMIZED:
            gc.set_threshold(2000, 20, 20)  # Less frequent GC
    
    def _optimization_loop(self):
        """Continuous optimization loop"""
        while self.optimization_active:
            try:
                # Monitor system resources
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent / 100.0
                
                # Apply dynamic optimizations based on resource usage
                if memory_percent > self.config.resource_threshold_memory:
                    self.memory_optimizer.optimize_garbage_collection()
                
                # Update optimization stats
                if cpu_percent > self.config.resource_threshold_cpu:
                    self.optimization_stats['performance_improvements'].append({
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory_percent,
                        'action': 'resource_optimization'
                    })
                
                time.sleep(self.config.optimization_interval)
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                time.sleep(30)  # Wait before retrying
    
    def create_optimized_cache(self, cache_name: str, max_size: int = None) -> SmartCache:
        """Create optimized cache"""
        if max_size is None:
            max_size = self.config.cache_size_limit
        
        cache = SmartCache(
            max_size=max_size,
            strategy=self.config.cache_strategy
        )
        
        self.cache_manager[cache_name] = cache
        return cache
    
    def get_optimized_thread_pool(self) -> ThreadPoolExecutor:
        """Get optimized thread pool"""
        return self.thread_optimizer.get_optimized_thread_pool()
    
    def create_object_pool(self, pool_name: str, factory_func: Callable, 
                          initial_size: int = 10) -> ObjectPool:
        """Create optimized object pool"""
        return self.memory_optimizer.create_object_pool(pool_name, factory_func, initial_size)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        # Collect performance metrics
        memory_stats = self.memory_optimizer.monitor_memory_usage()
        cpu_recommendations = self.cpu_optimizer.get_cpu_optimization_recommendations()
        memory_recommendations = self.memory_optimizer.suggest_memory_optimizations(memory_stats)
        
        # Collect cache statistics
        cache_stats = {}
        for name, cache in self.cache_manager.items():
            cache_stats[name] = cache.get_stats()
        
        report = {
            'optimization_config': {
                'strategy': self.config.strategy.name,
                'auto_optimize': self.config.auto_optimize,
                'optimization_active': self.optimization_active
            },
            'optimization_stats': self.optimization_stats,
            'performance_metrics': {
                'memory': memory_stats,
                'cpu_count': self.cpu_optimizer.cpu_count,
                'thread_pool_size': self.thread_optimizer.optimal_size
            },
            'cache_statistics': cache_stats,
            'recommendations': {
                'cpu': cpu_recommendations,
                'memory': memory_recommendations
            },
            'timestamp': time.time()
        }
        
        return report
    
    def export_optimization_report(self, filepath: str = None) -> str:
        """Export optimization report to file"""
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"optimization_report_{timestamp}.json"
        
        report = self.get_optimization_report()
        
        with open(filepath, 'w') as f:
            import json
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Optimization report exported to {filepath}")
        return filepath


# Decorator for automatic optimization
def optimize_performance(strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
    """Decorator to automatically optimize function performance"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create temporary optimizer for this function
            config = OptimizationConfig(strategy=strategy, auto_optimize=False)
            optimizer = SystemOptimizer(config)
            
            # Apply optimizations
            optimizer._apply_static_optimizations()
            
            # Execute function
            try:
                return func(*args, **kwargs)
            finally:
                # Cleanup
                optimizer.stop_optimization()
        
        return wrapper
    return decorator


# Global optimizer instance
_global_optimizer = None


def get_global_optimizer() -> SystemOptimizer:
    """Get global system optimizer"""
    global _global_optimizer
    if _global_optimizer is None:
        config = OptimizationConfig()
        _global_optimizer = SystemOptimizer(config)
    return _global_optimizer


def start_global_optimization():
    """Start global system optimization"""
    get_global_optimizer().start_optimization()


def stop_global_optimization():
    """Stop global system optimization"""
    get_global_optimizer().stop_optimization()


def get_optimization_report() -> Dict[str, Any]:
    """Get global optimization report"""
    return get_global_optimizer().get_optimization_report()


# Example usage and testing
if __name__ == "__main__":
    # Test system optimizer
    logger.info("Testing System Optimizer")
    
    # Create optimizer with aggressive strategy
    config = OptimizationConfig(
        strategy=OptimizationStrategy.AGGRESSIVE,
        auto_optimize=True,
        optimization_interval=5.0
    )
    
    optimizer = SystemOptimizer(config)
    
    try:
        # Start optimization
        optimizer.start_optimization()
        
        # Test cache
        cache = optimizer.create_optimized_cache("test_cache", max_size=100)
        
        # Test cache operations
        for i in range(150):  # More than cache size
            cache.put(f"key_{i}", f"value_{i}")
        
        # Check cache stats
        cache_stats = cache.get_stats()
        logger.info(f"Cache stats: {cache_stats}")
        
        # Test object pool
        def create_test_object():
            return {"id": time.time(), "data": [1, 2, 3, 4, 5]}
        
        object_pool = optimizer.create_object_pool("test_pool", create_test_object, 5)
        
        # Test object pool
        objects = []
        for _ in range(10):
            obj = object_pool.acquire()
            objects.append(obj)
        
        for obj in objects:
            object_pool.release(obj)
        
        pool_stats = object_pool.get_stats()
        logger.info(f"Object pool stats: {pool_stats}")
        
        # Test thread pool
        thread_pool = optimizer.get_optimized_thread_pool()
        
        def test_task(x):
            return x * x
        
        futures = []
        for i in range(20):
            future = thread_pool.submit(test_task, i)
            futures.append(future)
        
        results = [f.result() for f in futures]
        logger.info(f"Thread pool test completed: {len(results)} results")
        
        # Wait a bit for optimization loop
        time.sleep(10)
        
        # Get optimization report
        report = optimizer.get_optimization_report()
        logger.info("Optimization report generated:")
        logger.info(f"  Optimizations applied: {report['optimization_stats']['optimizations_applied']}")
        logger.info(f"  Cache hit rates: {[stats['hit_rate'] for stats in report['cache_statistics'].values()]}")
        logger.info(f"  Memory usage: {report['performance_metrics']['memory']['process_memory_mb']:.1f} MB")
        
        # Export report
        report_file = optimizer.export_optimization_report("test_optimization_report.json")
        logger.info(f"Report saved to: {report_file}")
        
    except Exception as e:
        logger.error(f"System optimizer test failed: {e}")
    finally:
        optimizer.stop_optimization()
    
    print("System optimizer test completed!")