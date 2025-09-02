"""
Memory Management and Optimization Module
Model-Based RL Human Intent Recognition System

This module provides advanced memory management including object pooling,
efficient data structures, memory monitoring, and garbage collection optimization.
"""

import gc
import sys
import threading
import time
import weakref
from typing import Dict, List, Any, Optional, Type, Union, Generic, TypeVar
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import numpy as np
import psutil
import logging
from contextlib import contextmanager
import tracemalloc
from functools import wraps
import pickle
import mmap
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

T = TypeVar('T')


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    enable_pooling: bool = True
    pool_max_size: int = 1000
    enable_monitoring: bool = True
    monitoring_interval: float = 1.0  # seconds
    gc_threshold_mb: float = 1000.0  # MB
    enable_compression: bool = True
    use_memory_mapping: bool = True
    cache_size_limit_mb: float = 500.0
    enable_weak_references: bool = True
    preallocation_size: int = 100


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    rss_mb: float = 0.0  # Resident Set Size
    vms_mb: float = 0.0  # Virtual Memory Size
    percent_used: float = 0.0
    available_mb: float = 0.0
    swap_mb: float = 0.0
    pool_usage: Dict[str, int] = field(default_factory=dict)
    gc_collections: int = 0
    object_counts: Dict[str, int] = field(default_factory=dict)
    memory_leaks: List[str] = field(default_factory=list)


class ObjectPool(Generic[T]):
    """Thread-safe object pool for efficient memory reuse."""
    
    def __init__(self, factory: callable, max_size: int = 100, reset_method: str = None):
        self.factory = factory
        self.max_size = max_size
        self.reset_method = reset_method
        self.pool = deque()
        self.created_count = 0
        self.borrowed_count = 0
        self.returned_count = 0
        self.lock = threading.Lock()
        
    def borrow(self) -> T:
        """Borrow an object from the pool."""
        with self.lock:
            if self.pool:
                obj = self.pool.popleft()
            else:
                obj = self.factory()
                self.created_count += 1
            
            self.borrowed_count += 1
            return obj
    
    def return_object(self, obj: T) -> None:
        """Return an object to the pool."""
        with self.lock:
            # Reset object if reset method is specified
            if self.reset_method and hasattr(obj, self.reset_method):
                getattr(obj, self.reset_method)()
            
            # Only keep if pool not full
            if len(self.pool) < self.max_size:
                self.pool.append(obj)
                self.returned_count += 1
            else:
                # Let object be garbage collected
                del obj
    
    def clear(self):
        """Clear the pool."""
        with self.lock:
            self.pool.clear()
    
    def stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self.lock:
            return {
                'pool_size': len(self.pool),
                'created_count': self.created_count,
                'borrowed_count': self.borrowed_count,
                'returned_count': self.returned_count,
                'reuse_rate': self.returned_count / max(1, self.borrowed_count)
            }


class NumpyArrayPool:
    """Specialized pool for NumPy arrays."""
    
    def __init__(self, max_size: int = 100):
        self.pools = defaultdict(lambda: ObjectPool(
            factory=lambda: None,  # Will be set dynamically
            max_size=max_size
        ))
        self.lock = threading.Lock()
    
    def get_array(self, shape: tuple, dtype: np.dtype = np.float64) -> np.ndarray:
        """Get a NumPy array from the pool."""
        key = (shape, dtype)
        
        with self.lock:
            if key not in self.pools:
                self.pools[key] = ObjectPool(
                    factory=lambda: np.zeros(shape, dtype=dtype),
                    max_size=50,
                    reset_method='fill'
                )
        
        array = self.pools[key].borrow()
        array.fill(0)  # Reset array
        return array
    
    def return_array(self, array: np.ndarray) -> None:
        """Return a NumPy array to the pool."""
        key = (array.shape, array.dtype)
        
        with self.lock:
            if key in self.pools:
                self.pools[key].return_object(array)
    
    def clear_all(self):
        """Clear all array pools."""
        with self.lock:
            for pool in self.pools.values():
                pool.clear()
            self.pools.clear()


class CircularBuffer:
    """Memory-efficient circular buffer implementation."""
    
    def __init__(self, max_size: int, dtype: np.dtype = np.float64):
        self.max_size = max_size
        self.buffer = np.zeros(max_size, dtype=dtype)
        self.head = 0
        self.size = 0
        self.lock = threading.Lock()
    
    def append(self, value: Union[float, int]) -> None:
        """Append a value to the buffer."""
        with self.lock:
            self.buffer[self.head] = value
            self.head = (self.head + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
    
    def get_recent(self, n: int = None) -> np.ndarray:
        """Get the n most recent values."""
        n = n or self.size
        n = min(n, self.size)
        
        with self.lock:
            if n == 0:
                return np.array([])
            
            if self.size < self.max_size:
                # Buffer not full yet
                return self.buffer[:self.size][-n:]
            else:
                # Buffer is full, need to handle wrap-around
                if n <= self.head:
                    return self.buffer[self.head - n:self.head]
                else:
                    # Wrap around
                    part1 = self.buffer[self.max_size - (n - self.head):]
                    part2 = self.buffer[:self.head]
                    return np.concatenate([part1, part2])
    
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.head = 0
            self.size = 0
            self.buffer.fill(0)


class CompressedStorage:
    """Compressed storage for large data structures."""
    
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level
        self.storage = {}
        self.access_times = {}
        self.lock = threading.Lock()
    
    def store(self, key: str, data: Any) -> None:
        """Store data with compression."""
        import zlib
        
        with self.lock:
            # Serialize and compress
            serialized = pickle.dumps(data)
            compressed = zlib.compress(serialized, level=self.compression_level)
            
            self.storage[key] = compressed
            self.access_times[key] = time.time()
    
    def retrieve(self, key: str) -> Any:
        """Retrieve and decompress data."""
        import zlib
        
        with self.lock:
            if key not in self.storage:
                return None
            
            compressed = self.storage[key]
            serialized = zlib.decompress(compressed)
            data = pickle.loads(serialized)
            
            self.access_times[key] = time.time()
            return data
    
    def remove(self, key: str) -> bool:
        """Remove stored data."""
        with self.lock:
            if key in self.storage:
                del self.storage[key]
                del self.access_times[key]
                return True
            return False
    
    def get_compression_ratio(self, key: str) -> Optional[float]:
        """Get compression ratio for a key."""
        if key not in self.storage:
            return None
        
        # Estimate original size by temporary decompression
        data = self.retrieve(key)
        original_size = len(pickle.dumps(data))
        compressed_size = len(self.storage[key])
        
        return compressed_size / original_size
    
    def cleanup_old(self, max_age_seconds: float = 3600):
        """Remove old entries."""
        current_time = time.time()
        
        with self.lock:
            keys_to_remove = [
                key for key, access_time in self.access_times.items()
                if current_time - access_time > max_age_seconds
            ]
            
            for key in keys_to_remove:
                self.remove(key)


class MemoryMappedArray:
    """Memory-mapped array for large datasets."""
    
    def __init__(self, filepath: str, shape: tuple, dtype: np.dtype = np.float64,
                 mode: str = 'w+'):
        self.filepath = filepath
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        
        # Calculate size
        itemsize = np.dtype(dtype).itemsize
        self.size = int(np.prod(shape) * itemsize)
        
        # Create memory-mapped file
        self.file = None
        self.mmap = None
        self.array = None
        self._create_mapping()
    
    def _create_mapping(self):
        """Create the memory mapping."""
        # Create or open file
        self.file = open(self.filepath, 'r+b' if 'r' in self.mode else 'w+b')
        
        if 'w' in self.mode:
            # Ensure file is large enough
            self.file.seek(self.size - 1)
            self.file.write(b'\0')
            self.file.flush()
        
        # Create memory map
        self.mmap = mmap.mmap(self.file.fileno(), self.size)
        
        # Create NumPy array view
        self.array = np.frombuffer(self.mmap, dtype=self.dtype).reshape(self.shape)
    
    def __getitem__(self, key):
        """Array-like access."""
        return self.array[key]
    
    def __setitem__(self, key, value):
        """Array-like assignment."""
        self.array[key] = value
    
    def flush(self):
        """Flush changes to disk."""
        if self.mmap:
            self.mmap.flush()
    
    def close(self):
        """Close the memory mapping."""
        if self.mmap:
            self.mmap.close()
        if self.file:
            self.file.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class MemoryMonitor:
    """Continuous memory usage monitoring."""
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.monitoring = False
        self.monitor_thread = None
        self.stats_history = deque(maxlen=1000)
        self.alerts = []
        self.lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Start tracemalloc for detailed tracking
        if self.config.enable_monitoring:
            tracemalloc.start()
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                stats = self._collect_stats()
                
                with self.lock:
                    self.stats_history.append(stats)
                
                # Check for alerts
                self._check_alerts(stats)
                
                time.sleep(self.config.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {e}")
                break
    
    def _collect_stats(self) -> MemoryStats:
        """Collect current memory statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        stats = MemoryStats(
            rss_mb=memory_info.rss / (1024 * 1024),
            vms_mb=memory_info.vms / (1024 * 1024),
            percent_used=system_memory.percent,
            available_mb=system_memory.available / (1024 * 1024),
            swap_mb=psutil.swap_memory().used / (1024 * 1024)
        )
        
        # Add GC stats
        stats.gc_collections = sum(gc.get_count())
        
        # Add object counts
        if hasattr(gc, 'get_objects'):
            objects = gc.get_objects()
            object_types = defaultdict(int)
            for obj in objects:
                obj_type = type(obj).__name__
                object_types[obj_type] += 1
            stats.object_counts = dict(object_types)
        
        return stats
    
    def _check_alerts(self, stats: MemoryStats):
        """Check for memory-related alerts."""
        alerts = []
        
        # High memory usage
        if stats.rss_mb > self.config.gc_threshold_mb:
            alerts.append(f"High memory usage: {stats.rss_mb:.1f}MB")
        
        # High system memory usage
        if stats.percent_used > 90:
            alerts.append(f"System memory critical: {stats.percent_used:.1f}%")
        
        # Swap usage
        if stats.swap_mb > 100:
            alerts.append(f"Swap memory usage: {stats.swap_mb:.1f}MB")
        
        if alerts:
            with self.lock:
                self.alerts.extend(alerts)
            
            for alert in alerts:
                self.logger.warning(f"Memory alert: {alert}")
    
    def get_current_stats(self) -> Optional[MemoryStats]:
        """Get current memory statistics."""
        with self.lock:
            return self.stats_history[-1] if self.stats_history else None
    
    def get_stats_history(self, n: int = 100) -> List[MemoryStats]:
        """Get recent statistics history."""
        with self.lock:
            return list(self.stats_history)[-n:]
    
    def get_alerts(self, clear: bool = True) -> List[str]:
        """Get current alerts."""
        with self.lock:
            alerts = self.alerts.copy()
            if clear:
                self.alerts.clear()
            return alerts


class SmartGarbageCollector:
    """Intelligent garbage collection manager."""
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.logger = logging.getLogger(__name__)
        
        # GC statistics
        self.gc_stats = {
            'manual_collections': 0,
            'total_freed_mb': 0.0,
            'average_collection_time': 0.0
        }
        
        # Set GC thresholds
        self._optimize_gc_thresholds()
    
    def _optimize_gc_thresholds(self):
        """Optimize garbage collection thresholds."""
        # Get current thresholds
        current = gc.get_threshold()
        
        # Increase thresholds for better performance (less frequent GC)
        new_thresholds = (
            current[0] * 2,  # Generation 0
            current[1] * 2,  # Generation 1
            current[2] * 2   # Generation 2
        )
        
        gc.set_threshold(*new_thresholds)
        self.logger.info(f"GC thresholds optimized: {current} -> {new_thresholds}")
    
    def collect_if_needed(self, memory_mb: float) -> bool:
        """Collect garbage if memory usage is high."""
        if memory_mb > self.config.gc_threshold_mb:
            return self.force_collection()
        return False
    
    def force_collection(self) -> bool:
        """Force garbage collection."""
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        # Collect all generations
        collected = 0
        for generation in range(3):
            collected += gc.collect(generation)
        
        memory_after = self._get_memory_usage()
        collection_time = time.time() - start_time
        freed_mb = max(0, memory_before - memory_after)
        
        # Update statistics
        self.gc_stats['manual_collections'] += 1
        self.gc_stats['total_freed_mb'] += freed_mb
        
        # Update average collection time
        prev_avg = self.gc_stats['average_collection_time']
        count = self.gc_stats['manual_collections']
        self.gc_stats['average_collection_time'] = (prev_avg * (count - 1) + collection_time) / count
        
        self.logger.info(
            f"GC completed: {collected} objects collected, "
            f"{freed_mb:.1f}MB freed in {collection_time:.3f}s"
        )
        
        return freed_mb > 0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return psutil.Process().memory_info().rss / (1024 * 1024)
    
    def analyze_memory_leaks(self) -> List[str]:
        """Analyze potential memory leaks."""
        leaks = []
        
        # Get object counts
        objects = gc.get_objects()
        object_counts = defaultdict(int)
        
        for obj in objects:
            obj_type = type(obj).__name__
            object_counts[obj_type] += 1
        
        # Check for common leak patterns
        suspicious_types = ['list', 'dict', 'function', 'frame']
        for obj_type in suspicious_types:
            count = object_counts.get(obj_type, 0)
            if count > 10000:  # Threshold for suspicion
                leaks.append(f"High {obj_type} count: {count}")
        
        # Check for uncollectable objects
        uncollectable = len(gc.garbage)
        if uncollectable > 0:
            leaks.append(f"Uncollectable objects: {uncollectable}")
        
        return leaks


class MemoryManager:
    """Main memory management system."""
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        
        # Initialize components
        self.pools = {}
        self.numpy_pool = NumpyArrayPool(self.config.pool_max_size)
        self.compressed_storage = CompressedStorage()
        self.monitor = MemoryMonitor(config)
        self.gc_manager = SmartGarbageCollector(config)
        
        # Memory-mapped files
        self.memory_maps = {}
        
        # Weak reference tracking
        self.weak_refs = weakref.WeakValueDictionary()
        
        self.logger = logging.getLogger(__name__)
        
        # Start monitoring if enabled
        if self.config.enable_monitoring:
            self.monitor.start_monitoring()
    
    def create_pool(self, name: str, factory: callable, max_size: int = None,
                   reset_method: str = None) -> ObjectPool:
        """Create a new object pool."""
        max_size = max_size or self.config.pool_max_size
        pool = ObjectPool(factory, max_size, reset_method)
        self.pools[name] = pool
        return pool
    
    def get_numpy_array(self, shape: tuple, dtype: np.dtype = np.float64) -> np.ndarray:
        """Get a NumPy array from the pool."""
        return self.numpy_pool.get_array(shape, dtype)
    
    def return_numpy_array(self, array: np.ndarray):
        """Return a NumPy array to the pool."""
        self.numpy_pool.return_array(array)
    
    def create_circular_buffer(self, size: int, dtype: np.dtype = np.float64) -> CircularBuffer:
        """Create a memory-efficient circular buffer."""
        return CircularBuffer(size, dtype)
    
    def create_memory_mapped_array(self, filepath: str, shape: tuple,
                                 dtype: np.dtype = np.float64, mode: str = 'w+') -> MemoryMappedArray:
        """Create a memory-mapped array."""
        if not self.config.use_memory_mapping:
            raise RuntimeError("Memory mapping is disabled")
        
        mmap_array = MemoryMappedArray(filepath, shape, dtype, mode)
        self.memory_maps[filepath] = mmap_array
        return mmap_array
    
    def store_compressed(self, key: str, data: Any):
        """Store data with compression."""
        if self.config.enable_compression:
            self.compressed_storage.store(key, data)
        else:
            # Store without compression (just use pickle)
            self.compressed_storage.storage[key] = pickle.dumps(data)
    
    def retrieve_compressed(self, key: str) -> Any:
        """Retrieve compressed data."""
        return self.compressed_storage.retrieve(key)
    
    def add_weak_reference(self, key: str, obj: Any):
        """Add weak reference to track object lifecycle."""
        if self.config.enable_weak_references:
            self.weak_refs[key] = obj
    
    def get_memory_stats(self) -> Optional[MemoryStats]:
        """Get current memory statistics."""
        stats = self.monitor.get_current_stats()
        
        if stats:
            # Add pool statistics
            for name, pool in self.pools.items():
                stats.pool_usage[name] = pool.stats()['pool_size']
        
        return stats
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization."""
        optimizations = []
        
        # Force garbage collection
        freed_mb = self.gc_manager.force_collection()
        if freed_mb > 0:
            optimizations.append(f"GC freed {freed_mb:.1f}MB")
        
        # Clear old compressed storage
        self.compressed_storage.cleanup_old()
        optimizations.append("Cleaned old compressed storage")
        
        # Clear unused pools
        cleared_pools = 0
        for name, pool in self.pools.items():
            if pool.stats()['reuse_rate'] < 0.1:  # Low reuse rate
                pool.clear()
                cleared_pools += 1
        
        if cleared_pools > 0:
            optimizations.append(f"Cleared {cleared_pools} underutilized pools")
        
        # Flush memory-mapped files
        for mmap_array in self.memory_maps.values():
            mmap_array.flush()
        optimizations.append("Flushed memory-mapped files")
        
        return {
            'optimizations': optimizations,
            'memory_freed_mb': freed_mb,
            'pools_cleared': cleared_pools
        }
    
    def cleanup(self):
        """Cleanup all memory resources."""
        self.logger.info("Starting memory manager cleanup")
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Clear all pools
        for pool in self.pools.values():
            pool.clear()
        self.pools.clear()
        
        self.numpy_pool.clear_all()
        
        # Close memory-mapped files
        for mmap_array in self.memory_maps.values():
            mmap_array.close()
        self.memory_maps.clear()
        
        # Clear compressed storage
        self.compressed_storage.storage.clear()
        
        # Final garbage collection
        self.gc_manager.force_collection()
        
        self.logger.info("Memory manager cleanup completed")
    
    @contextmanager
    def temporary_array(self, shape: tuple, dtype: np.dtype = np.float64):
        """Context manager for temporary array usage."""
        array = self.get_numpy_array(shape, dtype)
        try:
            yield array
        finally:
            self.return_numpy_array(array)
    
    def memory_efficient_decorator(self, func):
        """Decorator for memory-efficient function execution."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get initial memory
            initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            try:
                result = func(*args, **kwargs)
                
                # Check if cleanup is needed
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                if current_memory - initial_memory > 100:  # 100MB increase
                    self.gc_manager.collect_if_needed(current_memory)
                
                return result
            except Exception as e:
                # Cleanup on exception
                self.gc_manager.force_collection()
                raise
        
        return wrapper


def create_test_workload():
    """Create test workload for memory management testing."""
    memory_manager = MemoryManager()
    
    # Test array pooling
    arrays = []
    for i in range(100):
        array = memory_manager.get_numpy_array((1000, 1000))
        array.fill(i)
        arrays.append(array)
    
    # Return arrays to pool
    for array in arrays:
        memory_manager.return_numpy_array(array)
    
    # Test compressed storage
    large_data = {f'key_{i}': np.random.randn(1000, 1000) for i in range(10)}
    for key, data in large_data.items():
        memory_manager.store_compressed(key, data)
    
    # Test circular buffer
    buffer = memory_manager.create_circular_buffer(1000)
    for i in range(2000):  # More than buffer size
        buffer.append(i)
    
    recent_values = buffer.get_recent(100)
    print(f"Recent values: {recent_values[-10:]}")  # Last 10 values
    
    # Get memory stats
    stats = memory_manager.get_memory_stats()
    if stats:
        print(f"Memory usage: {stats.rss_mb:.1f}MB")
        print(f"Pool usage: {stats.pool_usage}")
    
    # Optimize memory
    optimization_result = memory_manager.optimize_memory()
    print(f"Memory optimization: {optimization_result}")
    
    # Cleanup
    memory_manager.cleanup()


if __name__ == "__main__":
    # Run test workload
    create_test_workload()