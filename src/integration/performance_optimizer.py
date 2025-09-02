#!/usr/bin/env python3
"""
Computational Performance Optimization System
============================================

This module provides comprehensive computational performance optimization for
real-time human intent recognition systems. It implements parallel processing,
GPU acceleration, efficient caching strategies, and approximation algorithms
for time-critical scenarios.

Key Features:
- Parallel processing for independent computations with thread/process pools
- GPU acceleration for GP inference and MPC optimization using CUDA/OpenCL
- Multi-level caching system with LRU and time-based eviction
- Approximation algorithms for time-critical scenarios
- Computational profiling and bottleneck identification
- Dynamic load balancing and resource allocation

Performance Requirements:
- CPU utilization: <80% average, <95% peak
- GPU utilization: optimized for inference workloads
- Cache hit ratio: >80% for repeated computations
- Parallel speedup: >2x for parallelizable tasks

Author: Claude Code (Anthropic)
Date: 2025-01-15
Version: 1.0
"""

import time
import threading
import multiprocessing as mp
import concurrent.futures
import logging
import psutil
import functools
import hashlib
import pickle
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import OrderedDict, deque
from enum import Enum
import numpy as np
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ComputationType(Enum):
    """Types of computational tasks"""
    GP_INFERENCE = "gp_inference"
    MPC_OPTIMIZATION = "mpc_optimization"
    PERCEPTION = "perception"
    PLANNING = "planning"
    MATRIX_OPERATIONS = "matrix_ops"
    SIGNAL_PROCESSING = "signal_processing"


class ParallelMode(Enum):
    """Parallel execution modes"""
    THREAD = "thread"      # Threading for I/O bound
    PROCESS = "process"    # Multiprocessing for CPU bound
    GPU = "gpu"           # GPU acceleration
    HYBRID = "hybrid"     # Adaptive selection


@dataclass
class ComputationProfile:
    """Performance profile for computational tasks"""
    task_name: str
    computation_type: ComputationType
    avg_execution_time_ms: float = 0.0
    peak_execution_time_ms: float = 0.0
    cpu_utilization: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0
    cache_hit_rate: float = 0.0
    parallel_speedup: float = 1.0
    execution_count: int = 0
    last_updated: float = field(default_factory=time.time)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_time: float
    access_time: float
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[float] = None


class LRUCache:
    """
    Thread-safe LRU cache with time-based expiration and size limits.
    """
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 100.0, 
                 default_ttl: Optional[float] = None):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.current_memory_bytes = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.debug(f"Created LRU cache (max_size={max_size}, max_memory={max_memory_mb}MB)")
    
    def _make_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments"""
        key_data = (args, tuple(sorted(kwargs.items())))
        return hashlib.md5(pickle.dumps(key_data)).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            current_time = time.time()
            if (entry.ttl_seconds is not None and 
                current_time - entry.created_time > entry.ttl_seconds):
                self.remove(key)
                self.misses += 1
                return None
            
            # Update access info
            entry.access_time = current_time
            entry.access_count += 1
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            
            self.hits += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache"""
        with self.lock:
            current_time = time.time()
            
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 1024  # Estimate if can't serialize
            
            # Remove existing entry if updating
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_memory_bytes -= old_entry.size_bytes
                del self.cache[key]
            
            # Check memory limit
            while (self.current_memory_bytes + size_bytes > self.max_memory_bytes and 
                   self.cache):
                self._evict_lru()
            
            # Check size limit
            while len(self.cache) >= self.max_size and self.cache:
                self._evict_lru()
            
            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_time=current_time,
                access_time=current_time,
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl or self.default_ttl
            )
            
            self.cache[key] = entry
            self.current_memory_bytes += size_bytes
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                self.current_memory_bytes -= entry.size_bytes
                del self.cache[key]
                return True
            return False
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if self.cache:
            key, entry = self.cache.popitem(last=False)  # Remove oldest
            self.current_memory_bytes -= entry.size_bytes
            self.evictions += 1
    
    def clear_expired(self) -> int:
        """Clear expired entries"""
        with self.lock:
            current_time = time.time()
            expired_keys = []
            
            for key, entry in self.cache.items():
                if (entry.ttl_seconds is not None and 
                    current_time - entry.created_time > entry.ttl_seconds):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self.remove(key)
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(1, total_requests)
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_mb': self.current_memory_bytes / 1024 / 1024,
                'max_memory_mb': self.max_memory_bytes / 1024 / 1024,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions
            }


def cached(cache: LRUCache, ttl: Optional[float] = None):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = cache._make_key(func.__name__, *args, **kwargs)
            
            # Try cache first
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(key, result, ttl)
            
            return result
        return wrapper
    return decorator


class ThreadPoolManager:
    """
    Manages thread pools for different types of computational tasks.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize thread pool manager.
        
        Args:
            max_workers: Maximum number of worker threads (default: CPU count)
        """
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 4) + 4)
        
        # Separate pools for different task types
        self.pools = {
            'io_bound': concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix='io_pool'
            ),
            'cpu_light': concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers // 2,
                thread_name_prefix='cpu_light_pool'
            )
        }
        
        # Task tracking
        self.active_tasks = {}
        self.task_stats = {}
        
        logger.info(f"Thread pool manager initialized with {self.max_workers} max workers")
    
    def submit_task(self, task_type: str, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit task to appropriate thread pool"""
        if task_type not in self.pools:
            task_type = 'cpu_light'  # Default pool
        
        future = self.pools[task_type].submit(func, *args, **kwargs)
        
        # Track task
        task_id = id(future)
        self.active_tasks[task_id] = {
            'type': task_type,
            'start_time': time.time(),
            'future': future
        }
        
        # Update stats
        if task_type not in self.task_stats:
            self.task_stats[task_type] = {'submitted': 0, 'completed': 0, 'failed': 0}
        self.task_stats[task_type]['submitted'] += 1
        
        # Add callback for completion tracking
        future.add_done_callback(lambda f: self._task_completed(task_id, f))
        
        return future
    
    def _task_completed(self, task_id: int, future: concurrent.futures.Future) -> None:
        """Handle task completion"""
        if task_id in self.active_tasks:
            task_info = self.active_tasks[task_id]
            task_type = task_info['type']
            
            if future.exception():
                self.task_stats[task_type]['failed'] += 1
            else:
                self.task_stats[task_type]['completed'] += 1
            
            del self.active_tasks[task_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread pool statistics"""
        stats = {
            'max_workers': self.max_workers,
            'active_tasks': len(self.active_tasks),
            'task_stats': self.task_stats.copy()
        }
        
        for pool_name, pool in self.pools.items():
            stats[f'{pool_name}_pool'] = {
                'active_threads': pool._threads.__len__() if hasattr(pool, '_threads') else 0
            }
        
        return stats
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown all thread pools"""
        for pool in self.pools.values():
            pool.shutdown(wait=wait)


class ProcessPoolManager:
    """
    Manages process pools for CPU-intensive computations.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize process pool manager.
        
        Args:
            max_workers: Maximum number of worker processes (default: CPU count)
        """
        self.max_workers = max_workers or psutil.cpu_count()
        
        # Create process pool
        self.pool = mp.Pool(processes=self.max_workers)
        
        # Task tracking
        self.active_tasks = {}
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        logger.info(f"Process pool manager initialized with {self.max_workers} processes")
    
    def submit_task(self, func: Callable, *args, **kwargs) -> mp.pool.ApplyResult:
        """Submit task to process pool"""
        result = self.pool.apply_async(func, args, kwargs)
        
        # Track task
        task_id = id(result)
        self.active_tasks[task_id] = {
            'start_time': time.time(),
            'result': result
        }
        
        return result
    
    def map_tasks(self, func: Callable, iterable: List[Any], 
                  chunksize: Optional[int] = None) -> mp.pool.MapResult:
        """Map function over iterable using process pool"""
        return self.pool.map_async(func, iterable, chunksize)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get process pool statistics"""
        # Clean up completed tasks
        completed_task_ids = []
        for task_id, task_info in self.active_tasks.items():
            if task_info['result'].ready():
                completed_task_ids.append(task_id)
                if task_info['result'].successful():
                    self.completed_tasks += 1
                else:
                    self.failed_tasks += 1
        
        for task_id in completed_task_ids:
            del self.active_tasks[task_id]
        
        return {
            'max_workers': self.max_workers,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks
        }
    
    def shutdown(self) -> None:
        """Shutdown process pool"""
        self.pool.close()
        self.pool.join()


class GPUAccelerator:
    """
    GPU acceleration for computational tasks using available GPU libraries.
    """
    
    def __init__(self):
        """Initialize GPU accelerator"""
        self.gpu_available = False
        self.gpu_device = None
        self.gpu_memory_mb = 0
        
        # Try to detect and initialize GPU
        self._detect_gpu()
        
        logger.info(f"GPU accelerator initialized (available={self.gpu_available})")
    
    def _detect_gpu(self) -> None:
        """Detect available GPU resources"""
        try:
            # Try CUDA first
            import cupy as cp
            self.gpu_available = True
            self.gpu_device = cp.cuda.Device()
            
            # Get memory info
            mem_info = cp.cuda.runtime.memGetInfo()
            self.gpu_memory_mb = mem_info[1] / 1024 / 1024  # Total memory
            
            logger.info(f"CUDA GPU detected: {self.gpu_memory_mb:.0f}MB")
            return
            
        except ImportError:
            pass
        
        try:
            # Try OpenCL
            import pyopencl as cl
            platforms = cl.get_platforms()
            if platforms:
                for platform in platforms:
                    devices = platform.get_devices()
                    if devices:
                        self.gpu_available = True
                        self.gpu_device = devices[0]
                        logger.info("OpenCL GPU detected")
                        return
        except ImportError:
            pass
        
        logger.info("No GPU acceleration available")
    
    def is_available(self) -> bool:
        """Check if GPU acceleration is available"""
        return self.gpu_available
    
    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """GPU-accelerated matrix multiplication"""
        if not self.gpu_available:
            return np.dot(a, b)
        
        try:
            import cupy as cp
            # Transfer to GPU
            a_gpu = cp.asarray(a)
            b_gpu = cp.asarray(b)
            
            # Compute on GPU
            result_gpu = cp.dot(a_gpu, b_gpu)
            
            # Transfer back to CPU
            return cp.asnumpy(result_gpu)
            
        except Exception as e:
            logger.warning(f"GPU matrix multiply failed, falling back to CPU: {e}")
            return np.dot(a, b)
    
    def batch_operations(self, operations: List[Callable], 
                        data_list: List[np.ndarray]) -> List[np.ndarray]:
        """Execute batch operations on GPU"""
        if not self.gpu_available or not operations:
            return [op(data) for op, data in zip(operations, data_list)]
        
        try:
            import cupy as cp
            results = []
            
            for op, data in zip(operations, data_list):
                # Transfer to GPU
                data_gpu = cp.asarray(data)
                
                # Apply operation (assume it works with CuPy)
                result_gpu = op(data_gpu)
                
                # Transfer back
                results.append(cp.asnumpy(result_gpu))
            
            return results
            
        except Exception as e:
            logger.warning(f"GPU batch operations failed, falling back to CPU: {e}")
            return [op(data) for op, data in zip(operations, data_list)]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get GPU statistics"""
        stats = {
            'available': self.gpu_available,
            'memory_mb': self.gpu_memory_mb
        }
        
        if self.gpu_available:
            try:
                import cupy as cp
                mem_info = cp.cuda.runtime.memGetInfo()
                stats.update({
                    'free_memory_mb': mem_info[0] / 1024 / 1024,
                    'used_memory_mb': (mem_info[1] - mem_info[0]) / 1024 / 1024,
                    'utilization_percent': ((mem_info[1] - mem_info[0]) / mem_info[1]) * 100
                })
            except:
                pass
        
        return stats


class ApproximationEngine:
    """
    Engine for approximation algorithms when exact computation is too slow.
    """
    
    def __init__(self):
        """Initialize approximation engine"""
        self.approximation_methods = {}
        self.accuracy_targets = {}
        
        logger.debug("Approximation engine initialized")
    
    def register_approximation(self, task_name: str, exact_func: Callable, 
                             approx_func: Callable, accuracy_threshold: float = 0.95) -> None:
        """
        Register approximation method for a task.
        
        Args:
            task_name: Name of the task
            exact_func: Exact computation function
            approx_func: Approximation function
            accuracy_threshold: Minimum accuracy required (0-1)
        """
        self.approximation_methods[task_name] = {
            'exact': exact_func,
            'approx': approx_func,
            'threshold': accuracy_threshold
        }
        
        logger.debug(f"Registered approximation for {task_name}")
    
    def compute_adaptive(self, task_name: str, time_budget_ms: float, 
                        *args, **kwargs) -> Tuple[Any, float, bool]:
        """
        Compute with adaptive approximation based on time budget.
        
        Args:
            task_name: Name of registered task
            time_budget_ms: Available time in milliseconds
            
        Returns:
            Tuple of (result, actual_time_ms, used_approximation)
        """
        if task_name not in self.approximation_methods:
            raise ValueError(f"Task '{task_name}' not registered")
        
        methods = self.approximation_methods[task_name]
        
        start_time = time.perf_counter()
        
        # Try exact computation first if we have time
        if time_budget_ms > 50:  # If we have >50ms budget
            try:
                # Quick test to estimate exact computation time
                test_start = time.perf_counter()
                result = methods['exact'](*args, **kwargs)
                exact_time = (time.perf_counter() - test_start) * 1000
                
                if exact_time <= time_budget_ms * 0.8:  # Within 80% of budget
                    return result, exact_time, False
                
            except Exception as e:
                logger.warning(f"Exact computation failed: {e}")
        
        # Use approximation
        try:
            result = methods['approx'](*args, **kwargs)
            approx_time = (time.perf_counter() - start_time) * 1000
            return result, approx_time, True
            
        except Exception as e:
            logger.error(f"Approximation failed: {e}")
            # Fall back to exact if approximation fails
            result = methods['exact'](*args, **kwargs)
            total_time = (time.perf_counter() - start_time) * 1000
            return result, total_time, False


class PerformanceOptimizer:
    """
    Main computational performance optimization system.
    Coordinates all optimization strategies and provides unified interface.
    """
    
    def __init__(self):
        """Initialize performance optimizer"""
        # Core components
        self.cache = LRUCache(max_size=1000, max_memory_mb=200.0, default_ttl=300.0)
        self.thread_manager = ThreadPoolManager()
        self.process_manager = ProcessPoolManager()
        self.gpu_accelerator = GPUAccelerator()
        self.approximation_engine = ApproximationEngine()
        
        # Performance profiling
        self.profiles: Dict[str, ComputationProfile] = {}
        self.profiling_enabled = True
        
        # Load balancing
        self.load_balancer = self._create_load_balancer()
        
        logger.info("Performance optimizer initialized")
    
    def _create_load_balancer(self) -> Dict[str, Any]:
        """Create adaptive load balancer"""
        return {
            'cpu_threshold': 0.8,      # Switch to GPU if CPU > 80%
            'gpu_threshold': 0.9,      # Fallback to CPU if GPU > 90%
            'thread_threshold': 0.7,   # Switch to processes if threads > 70% util
            'cache_threshold': 0.6     # Use approximation if cache hit < 60%
        }
    
    def profile_function(self, func: Callable, task_name: str, 
                        computation_type: ComputationType) -> Callable:
        """Decorator to profile function performance"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.profiling_enabled:
                return func(*args, **kwargs)
            
            # Start profiling
            start_time = time.perf_counter()
            start_cpu = psutil.cpu_percent()
            process = psutil.Process()
            start_memory = process.memory_info().rss
            
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                logger.error(f"Function {task_name} failed: {e}")
                success = False
                raise
            finally:
                # Calculate metrics
                execution_time = (time.perf_counter() - start_time) * 1000
                end_cpu = psutil.cpu_percent()
                end_memory = process.memory_info().rss
                
                # Update profile
                if task_name not in self.profiles:
                    self.profiles[task_name] = ComputationProfile(
                        task_name=task_name,
                        computation_type=computation_type
                    )
                
                profile = self.profiles[task_name]
                profile.execution_count += 1
                
                # Update timing stats
                if execution_time > profile.peak_execution_time_ms:
                    profile.peak_execution_time_ms = execution_time
                
                # Moving average for execution time
                alpha = 0.1  # Smoothing factor
                profile.avg_execution_time_ms = (
                    (1 - alpha) * profile.avg_execution_time_ms + 
                    alpha * execution_time
                )
                
                profile.cpu_utilization = (end_cpu + start_cpu) / 2
                profile.memory_usage_mb = (end_memory - start_memory) / 1024 / 1024
                profile.last_updated = time.time()
            
            return result
        return wrapper
    
    def optimize_computation(self, task_name: str, func: Callable, 
                           parallel_mode: ParallelMode = ParallelMode.HYBRID,
                           use_cache: bool = True, time_budget_ms: Optional[float] = None,
                           *args, **kwargs) -> Any:
        """
        Optimize computation using best available strategy.
        
        Args:
            task_name: Name of the computational task
            func: Function to execute
            parallel_mode: Parallel execution mode
            use_cache: Whether to use caching
            time_budget_ms: Time budget for computation
            
        Returns:
            Computation result
        """
        # Try cache first
        if use_cache:
            cache_key = self.cache._make_key(task_name, *args, **kwargs)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                if task_name in self.profiles:
                    self.profiles[task_name].cache_hit_rate += 0.1  # Boost hit rate
                return cached_result
        
        # Select execution strategy
        strategy = self._select_execution_strategy(task_name, parallel_mode, time_budget_ms)
        
        # Execute with selected strategy
        start_time = time.perf_counter()
        
        try:
            if strategy == 'gpu' and self.gpu_accelerator.is_available():
                result = self._execute_gpu(func, *args, **kwargs)
            elif strategy == 'process':
                result = self._execute_process(func, *args, **kwargs)
            elif strategy == 'thread':
                result = self._execute_thread(func, *args, **kwargs)
            elif strategy == 'approximation':
                if task_name in self.approximation_engine.approximation_methods:
                    result, _, _ = self.approximation_engine.compute_adaptive(
                        task_name, time_budget_ms or 100, *args, **kwargs
                    )
                else:
                    result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)  # Sequential execution
            
            # Cache result
            if use_cache:
                execution_time = (time.perf_counter() - start_time) * 1000
                ttl = min(300, max(60, execution_time / 10))  # TTL based on computation time
                self.cache.put(cache_key, result, ttl)
            
            return result
            
        except Exception as e:
            logger.error(f"Optimized computation failed: {e}")
            # Fallback to sequential execution
            return func(*args, **kwargs)
    
    def _select_execution_strategy(self, task_name: str, parallel_mode: ParallelMode, 
                                 time_budget_ms: Optional[float]) -> str:
        """Select optimal execution strategy based on current conditions"""
        # Get current system state
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # Check if we have profile data
        profile = self.profiles.get(task_name)
        
        # Strategy selection logic
        if parallel_mode == ParallelMode.GPU and self.gpu_accelerator.is_available():
            gpu_stats = self.gpu_accelerator.get_stats()
            if gpu_stats.get('utilization_percent', 0) < self.load_balancer['gpu_threshold']:
                return 'gpu'
        
        if time_budget_ms and time_budget_ms < 50:  # Very tight time budget
            return 'approximation'
        
        if parallel_mode in [ParallelMode.PROCESS, ParallelMode.HYBRID]:
            if cpu_percent < self.load_balancer['cpu_threshold']:
                return 'process'
        
        if parallel_mode in [ParallelMode.THREAD, ParallelMode.HYBRID]:
            if cpu_percent < self.load_balancer['thread_threshold']:
                return 'thread'
        
        return 'sequential'
    
    def _execute_gpu(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with GPU acceleration"""
        # For matrix operations, use GPU accelerator
        if len(args) >= 2 and all(isinstance(arg, np.ndarray) for arg in args[:2]):
            return self.gpu_accelerator.matrix_multiply(args[0], args[1])
        
        # For other functions, try to execute normally
        return func(*args, **kwargs)
    
    def _execute_process(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function in process pool"""
        result = self.process_manager.submit_task(func, *args, **kwargs)
        return result.get(timeout=30)  # 30 second timeout
    
    def _execute_thread(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function in thread pool"""
        future = self.thread_manager.submit_task('cpu_light', func, *args, **kwargs)
        return future.result(timeout=30)  # 30 second timeout
    
    def batch_optimize(self, tasks: List[Tuple[str, Callable, tuple, dict]], 
                      parallel_mode: ParallelMode = ParallelMode.HYBRID) -> List[Any]:
        """
        Optimize batch of computations.
        
        Args:
            tasks: List of (task_name, function, args, kwargs) tuples
            parallel_mode: Parallel execution mode
            
        Returns:
            List of results
        """
        if not tasks:
            return []
        
        # Determine batch execution strategy
        if (parallel_mode in [ParallelMode.GPU, ParallelMode.HYBRID] and 
            self.gpu_accelerator.is_available() and len(tasks) > 4):
            
            # Try GPU batch processing
            try:
                operations = [task[1] for task in tasks]
                data_list = [task[2][0] if task[2] else np.array([]) for task in tasks]
                
                if all(isinstance(data, np.ndarray) for data in data_list):
                    return self.gpu_accelerator.batch_operations(operations, data_list)
            except Exception as e:
                logger.warning(f"GPU batch processing failed: {e}")
        
        # Parallel execution using thread/process pools
        if parallel_mode in [ParallelMode.THREAD, ParallelMode.HYBRID]:
            futures = []
            for task_name, func, args, kwargs in tasks:
                future = self.thread_manager.submit_task('cpu_light', func, *args, **kwargs)
                futures.append(future)
            
            return [future.result(timeout=30) for future in futures]
        
        # Sequential execution
        results = []
        for task_name, func, args, kwargs in tasks:
            result = self.optimize_computation(task_name, func, parallel_mode, 
                                             True, None, *args, **kwargs)
            results.append(result)
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        report = {
            'timestamp': time.time(),
            'cache_stats': self.cache.get_stats(),
            'thread_stats': self.thread_manager.get_stats(),
            'process_stats': self.process_manager.get_stats(),
            'gpu_stats': self.gpu_accelerator.get_stats(),
            'computation_profiles': {}
        }
        
        # Add computation profiles
        for task_name, profile in self.profiles.items():
            report['computation_profiles'][task_name] = {
                'avg_time_ms': profile.avg_execution_time_ms,
                'peak_time_ms': profile.peak_execution_time_ms,
                'execution_count': profile.execution_count,
                'cpu_utilization': profile.cpu_utilization,
                'memory_usage_mb': profile.memory_usage_mb,
                'cache_hit_rate': profile.cache_hit_rate
            }
        
        # System health indicators
        report['health'] = {
            'cache_healthy': report['cache_stats']['hit_rate'] > 0.6,
            'memory_healthy': report['cache_stats']['memory_usage_mb'] < 180,  # Under 180MB
            'gpu_available': report['gpu_stats']['available']
        }
        
        return report
    
    def cleanup(self) -> None:
        """Cleanup all resources"""
        logger.info("Cleaning up performance optimizer")
        
        self.thread_manager.shutdown()
        self.process_manager.shutdown()
        self.cache.clear_expired()
        
        logger.info("Performance optimizer cleanup completed")


# Example usage and testing
if __name__ == "__main__":
    # Test performance optimizer
    optimizer = PerformanceOptimizer()
    
    # Example computations
    @optimizer.profile_function("matrix_multiply", ComputationType.MATRIX_OPERATIONS)
    def matrix_multiply(a, b):
        return np.dot(a, b)
    
    @optimizer.profile_function("signal_filter", ComputationType.SIGNAL_PROCESSING)
    def signal_filter(signal, kernel):
        return np.convolve(signal, kernel, mode='same')
    
    # Test matrix multiplication
    a = np.random.rand(500, 500)
    b = np.random.rand(500, 500)
    
    result = optimizer.optimize_computation(
        "matrix_multiply", 
        matrix_multiply, 
        ParallelMode.GPU,
        use_cache=True,
        a, b
    )
    
    print(f"Matrix result shape: {result.shape}")
    
    # Test signal processing
    signal = np.random.rand(1000)
    kernel = np.array([0.1, 0.8, 0.1])
    
    filtered = optimizer.optimize_computation(
        "signal_filter",
        signal_filter,
        ParallelMode.THREAD,
        use_cache=True,
        signal, kernel
    )
    
    print(f"Filtered signal shape: {filtered.shape}")
    
    # Performance report
    report = optimizer.get_performance_report()
    print(f"Cache hit rate: {report['cache_stats']['hit_rate']:.2f}")
    print(f"GPU available: {report['gpu_stats']['available']}")
    
    # Cleanup
    optimizer.cleanup()