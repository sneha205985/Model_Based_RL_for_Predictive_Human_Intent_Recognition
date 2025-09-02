#!/usr/bin/env python3
"""
Memory Management & Optimization System
=======================================

This module provides comprehensive memory management and optimization for 
real-time human intent recognition systems. It implements bounded memory 
buffers, efficient data structures, memory pooling, and garbage collection
optimization for real-time constraints.

Key Features:
- Bounded memory buffers for streaming data with overflow protection
- Memory pool management for frequent allocations/deallocations  
- Efficient data structures optimized for real-time access patterns
- Garbage collection optimization and monitoring
- Memory usage profiling and leak detection
- Predictable memory access patterns for real-time guarantees

Performance Requirements:
- Memory usage: <2GB bounded growth
- Allocation/deallocation: <1ms for pooled objects
- GC pause: <10ms maximum
- Memory fragmentation: <20%

Author: Claude Code (Anthropic)
Date: 2025-01-15
Version: 1.0
"""

import threading
import time
import gc
import sys
import mmap
import os
import psutil
import logging
import tracemalloc
from typing import Dict, Any, Optional, List, Tuple, Generic, TypeVar, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import numpy as np
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

T = TypeVar('T')


class MemoryPoolType(Enum):
    """Memory pool types for different object sizes"""
    SMALL = "small"      # <1KB objects
    MEDIUM = "medium"    # 1KB-100KB objects  
    LARGE = "large"      # >100KB objects


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_allocated_mb: float = 0.0
    peak_usage_mb: float = 0.0
    current_usage_mb: float = 0.0
    pool_usage_mb: Dict[str, float] = field(default_factory=dict)
    gc_collections: int = 0
    gc_time_ms: float = 0.0
    fragmentation_percent: float = 0.0
    active_buffers: int = 0
    buffer_overflow_count: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass 
class PooledObject:
    """Object wrapper for memory pooling"""
    obj: Any
    pool_type: MemoryPoolType
    allocated_time: float
    last_used: float
    size_bytes: int
    in_use: bool = False


class ObjectPool(Generic[T]):
    """
    Thread-safe object pool for memory management.
    Reduces allocation/deallocation overhead for frequently used objects.
    """
    
    def __init__(self, factory: callable, pool_type: MemoryPoolType, 
                 max_size: int = 100, cleanup_func: callable = None):
        """
        Initialize object pool.
        
        Args:
            factory: Function to create new objects
            pool_type: Type of memory pool
            max_size: Maximum pool size
            cleanup_func: Optional cleanup function for objects
        """
        self.factory = factory
        self.pool_type = pool_type
        self.max_size = max_size
        self.cleanup_func = cleanup_func
        
        self.available = deque()
        self.in_use = set()
        self.lock = threading.RLock()
        self.total_created = 0
        self.total_reused = 0
        
        logger.debug(f"Created object pool for {pool_type.value} objects (max_size={max_size})")
    
    def acquire(self) -> T:
        """Acquire object from pool or create new one"""
        with self.lock:
            if self.available:
                pooled_obj = self.available.popleft()
                pooled_obj.in_use = True
                pooled_obj.last_used = time.time()
                self.in_use.add(pooled_obj)
                self.total_reused += 1
                return pooled_obj.obj
            else:
                # Create new object
                obj = self.factory()
                pooled_obj = PooledObject(
                    obj=obj,
                    pool_type=self.pool_type,
                    allocated_time=time.time(),
                    last_used=time.time(),
                    size_bytes=sys.getsizeof(obj),
                    in_use=True
                )
                self.in_use.add(pooled_obj)
                self.total_created += 1
                return obj
    
    def release(self, obj: T) -> None:
        """Release object back to pool"""
        with self.lock:
            # Find pooled object
            pooled_obj = None
            for po in self.in_use:
                if po.obj is obj:
                    pooled_obj = po
                    break
            
            if pooled_obj is None:
                logger.warning("Attempted to release object not from this pool")
                return
            
            self.in_use.remove(pooled_obj)
            
            # Clean object if cleanup function provided
            if self.cleanup_func:
                try:
                    self.cleanup_func(obj)
                except Exception as e:
                    logger.error(f"Error cleaning pooled object: {e}")
                    return  # Don't return to pool if cleanup failed
            
            # Return to pool if under limit
            if len(self.available) < self.max_size:
                pooled_obj.in_use = False
                self.available.append(pooled_obj)
            # Otherwise let it be garbage collected
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self.lock:
            return {
                'pool_type': self.pool_type.value,
                'available': len(self.available),
                'in_use': len(self.in_use),
                'total_created': self.total_created,
                'total_reused': self.total_reused,
                'reuse_ratio': self.total_reused / max(1, self.total_created + self.total_reused),
                'max_size': self.max_size
            }
    
    def cleanup_stale_objects(self, max_age_seconds: float = 300.0) -> int:
        """Clean up stale objects that haven't been used recently"""
        with self.lock:
            current_time = time.time()
            cleaned = 0
            
            # Clean available objects
            new_available = deque()
            for pooled_obj in self.available:
                if current_time - pooled_obj.last_used < max_age_seconds:
                    new_available.append(pooled_obj)
                else:
                    cleaned += 1
            
            self.available = new_available
            return cleaned


class BoundedMemoryBuffer(Generic[T]):
    """
    Thread-safe bounded memory buffer with overflow protection.
    Implements circular buffer semantics for streaming data.
    """
    
    def __init__(self, capacity: int, name: str = "buffer", 
                 overflow_strategy: str = "overwrite"):
        """
        Initialize bounded memory buffer.
        
        Args:
            capacity: Maximum number of items
            name: Buffer name for monitoring
            overflow_strategy: "overwrite", "drop", or "block"
        """
        self.capacity = capacity
        self.name = name
        self.overflow_strategy = overflow_strategy
        
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0
        self.lock = threading.RLock()
        
        # Statistics
        self.total_puts = 0
        self.total_gets = 0
        self.overflow_count = 0
        self.created_time = time.time()
        
        logger.debug(f"Created bounded buffer '{name}' (capacity={capacity}, strategy={overflow_strategy})")
    
    def put(self, item: T, timeout: Optional[float] = None) -> bool:
        """
        Put item in buffer.
        
        Args:
            item: Item to put
            timeout: Timeout in seconds (for "block" strategy)
            
        Returns:
            bool: True if item was added successfully
        """
        start_time = time.time() if timeout else None
        
        with self.lock:
            self.total_puts += 1
            
            if self.size < self.capacity:
                self.buffer[self.tail] = item
                self.tail = (self.tail + 1) % self.capacity
                self.size += 1
                return True
            
            # Buffer is full - handle overflow
            if self.overflow_strategy == "overwrite":
                # Overwrite oldest item
                self.buffer[self.tail] = item
                self.tail = (self.tail + 1) % self.capacity
                self.head = (self.head + 1) % self.capacity
                self.overflow_count += 1
                return True
            
            elif self.overflow_strategy == "drop":
                # Drop new item
                self.overflow_count += 1
                return False
            
            elif self.overflow_strategy == "block":
                # Block until space available
                while self.size >= self.capacity:
                    if timeout and (time.time() - start_time) > timeout:
                        return False
                    time.sleep(0.001)  # 1ms wait
                
                # Space now available
                self.buffer[self.tail] = item
                self.tail = (self.tail + 1) % self.capacity
                self.size += 1
                return True
            
            return False
    
    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """
        Get item from buffer.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Item or None if timeout/empty
        """
        start_time = time.time() if timeout else None
        
        with self.lock:
            while self.size == 0:
                if timeout is None:
                    return None
                
                if (time.time() - start_time) > timeout:
                    return None
                
                time.sleep(0.001)  # 1ms wait
            
            item = self.buffer[self.head]
            self.buffer[self.head] = None  # Help GC
            self.head = (self.head + 1) % self.capacity
            self.size -= 1
            self.total_gets += 1
            
            return item
    
    def peek(self) -> Optional[T]:
        """Peek at oldest item without removing"""
        with self.lock:
            if self.size == 0:
                return None
            return self.buffer[self.head]
    
    def peek_latest(self) -> Optional[T]:
        """Peek at newest item without removing"""
        with self.lock:
            if self.size == 0:
                return None
            latest_idx = (self.tail - 1) % self.capacity
            return self.buffer[latest_idx]
    
    def clear(self) -> None:
        """Clear all items from buffer"""
        with self.lock:
            for i in range(self.capacity):
                self.buffer[i] = None
            self.head = 0
            self.tail = 0
            self.size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            uptime = time.time() - self.created_time
            return {
                'name': self.name,
                'capacity': self.capacity,
                'size': self.size,
                'utilization': self.size / self.capacity,
                'total_puts': self.total_puts,
                'total_gets': self.total_gets,
                'overflow_count': self.overflow_count,
                'overflow_rate': self.overflow_count / max(1, self.total_puts),
                'throughput_hz': (self.total_puts + self.total_gets) / max(1, uptime),
                'strategy': self.overflow_strategy
            }


class MemoryMappedBuffer:
    """
    Memory-mapped buffer for large data structures.
    Provides efficient access to large datasets without full memory loading.
    """
    
    def __init__(self, size_bytes: int, filename: Optional[str] = None):
        """
        Initialize memory-mapped buffer.
        
        Args:
            size_bytes: Size of buffer in bytes
            filename: Optional file backing (None for anonymous mapping)
        """
        self.size_bytes = size_bytes
        self.filename = filename
        self.fd = None
        self.mmap_obj = None
        
        try:
            if filename:
                # File-backed mapping
                self.fd = os.open(filename, os.O_CREAT | os.O_TRUNC | os.O_RDWR)
                os.write(self.fd, b'\0' * size_bytes)  # Initialize file
                self.mmap_obj = mmap.mmap(self.fd, size_bytes)
            else:
                # Anonymous mapping
                self.mmap_obj = mmap.mmap(-1, size_bytes)
            
            logger.debug(f"Created memory-mapped buffer: {size_bytes} bytes")
            
        except Exception as e:
            logger.error(f"Failed to create memory-mapped buffer: {e}")
            self.cleanup()
            raise
    
    def read(self, offset: int, length: int) -> bytes:
        """Read data from buffer"""
        if not self.mmap_obj:
            raise RuntimeError("Buffer not initialized")
        
        self.mmap_obj.seek(offset)
        return self.mmap_obj.read(length)
    
    def write(self, offset: int, data: bytes) -> int:
        """Write data to buffer"""
        if not self.mmap_obj:
            raise RuntimeError("Buffer not initialized")
        
        self.mmap_obj.seek(offset)
        return self.mmap_obj.write(data)
    
    def flush(self) -> None:
        """Flush changes to backing store"""
        if self.mmap_obj:
            self.mmap_obj.flush()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.mmap_obj:
            self.mmap_obj.close()
            self.mmap_obj = None
        
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        
        if self.filename and os.path.exists(self.filename):
            try:
                os.unlink(self.filename)
            except OSError:
                pass


class RealtimeGCManager:
    """
    Garbage collection manager optimized for real-time systems.
    Minimizes GC pause times and provides predictable memory management.
    """
    
    def __init__(self, max_pause_ms: float = 10.0):
        """
        Initialize GC manager.
        
        Args:
            max_pause_ms: Maximum allowed GC pause time
        """
        self.max_pause_ms = max_pause_ms
        self.gc_stats = []
        self.last_gc_time = time.time()
        
        # Configure GC thresholds for real-time performance
        # More frequent generation 0 collections, less frequent gen 1/2
        gc.set_threshold(700, 10, 5)  # Default: 700, 10, 10
        
        logger.info(f"Initialized real-time GC manager (max_pause={max_pause_ms}ms)")
    
    def incremental_collect(self, generation: int = 0) -> float:
        """
        Perform incremental garbage collection.
        
        Args:
            generation: GC generation (0=youngest, fastest)
            
        Returns:
            float: GC pause time in milliseconds
        """
        start_time = time.perf_counter()
        
        # Perform collection
        collected = gc.collect(generation)
        
        pause_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Track statistics
        self.gc_stats.append({
            'timestamp': time.time(),
            'generation': generation,
            'objects_collected': collected,
            'pause_time_ms': pause_time_ms
        })
        
        # Keep only recent stats
        if len(self.gc_stats) > 1000:
            self.gc_stats = self.gc_stats[-500:]
        
        if pause_time_ms > self.max_pause_ms:
            logger.warning(f"GC pause time ({pause_time_ms:.1f}ms) exceeded limit ({self.max_pause_ms}ms)")
        
        return pause_time_ms
    
    def adaptive_collect(self) -> float:
        """
        Perform adaptive garbage collection based on system state.
        
        Returns:
            float: Total pause time in milliseconds
        """
        total_pause = 0.0
        
        # Always collect generation 0 (fastest)
        pause = self.incremental_collect(generation=0)
        total_pause += pause
        
        # Conditionally collect higher generations based on time budget
        if pause < self.max_pause_ms * 0.5:  # If gen 0 was fast
            remaining_budget = self.max_pause_ms - pause
            
            if remaining_budget > 2.0:  # At least 2ms remaining
                pause = self.incremental_collect(generation=1)
                total_pause += pause
                
                remaining_budget -= pause
                if remaining_budget > 3.0:  # At least 3ms remaining
                    pause = self.incremental_collect(generation=2)
                    total_pause += pause
        
        self.last_gc_time = time.time()
        return total_pause
    
    def get_gc_stats(self) -> Dict[str, Any]:
        """Get GC statistics"""
        if not self.gc_stats:
            return {}
        
        recent_stats = self.gc_stats[-100:]  # Last 100 collections
        pause_times = [s['pause_time_ms'] for s in recent_stats]
        
        return {
            'total_collections': len(self.gc_stats),
            'recent_collections': len(recent_stats),
            'avg_pause_ms': np.mean(pause_times),
            'max_pause_ms': np.max(pause_times),
            'min_pause_ms': np.min(pause_times),
            'pause_limit_exceeded': sum(1 for p in pause_times if p > self.max_pause_ms),
            'last_collection': self.gc_stats[-1]['timestamp'] if self.gc_stats else 0
        }


class MemoryManager:
    """
    Main memory management system for real-time applications.
    Coordinates all memory optimization strategies.
    """
    
    def __init__(self, memory_limit_mb: int = 2048):
        """
        Initialize memory manager.
        
        Args:
            memory_limit_mb: Memory usage limit in MB
        """
        self.memory_limit_mb = memory_limit_mb
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        
        # Object pools for different sizes
        self.pools = {
            MemoryPoolType.SMALL: ObjectPool(
                factory=lambda: bytearray(1024),  # 1KB objects
                pool_type=MemoryPoolType.SMALL,
                max_size=200
            ),
            MemoryPoolType.MEDIUM: ObjectPool(
                factory=lambda: bytearray(50 * 1024),  # 50KB objects
                pool_type=MemoryPoolType.MEDIUM,
                max_size=50
            ),
            MemoryPoolType.LARGE: ObjectPool(
                factory=lambda: bytearray(1024 * 1024),  # 1MB objects
                pool_type=MemoryPoolType.LARGE,
                max_size=10
            )
        }
        
        # Buffer management
        self.buffers: Dict[str, BoundedMemoryBuffer] = {}
        self.mmap_buffers: Dict[str, MemoryMappedBuffer] = {}
        
        # GC management
        self.gc_manager = RealtimeGCManager(max_pause_ms=10.0)
        
        # Statistics tracking
        self.stats_history = deque(maxlen=1000)
        
        # Memory monitoring
        self.memory_monitor_enabled = True
        self._start_memory_monitoring()
        
        logger.info(f"Memory manager initialized (limit={memory_limit_mb}MB)")
    
    def create_buffer(self, name: str, capacity: int, 
                     overflow_strategy: str = "overwrite") -> BoundedMemoryBuffer:
        """Create bounded memory buffer"""
        if name in self.buffers:
            raise ValueError(f"Buffer '{name}' already exists")
        
        buffer = BoundedMemoryBuffer(capacity, name, overflow_strategy)
        self.buffers[name] = buffer
        
        logger.debug(f"Created buffer '{name}' (capacity={capacity})")
        return buffer
    
    def get_buffer(self, name: str) -> Optional[BoundedMemoryBuffer]:
        """Get buffer by name"""
        return self.buffers.get(name)
    
    def create_mmap_buffer(self, name: str, size_bytes: int, 
                          filename: Optional[str] = None) -> MemoryMappedBuffer:
        """Create memory-mapped buffer"""
        if name in self.mmap_buffers:
            raise ValueError(f"Memory-mapped buffer '{name}' already exists")
        
        mmap_buffer = MemoryMappedBuffer(size_bytes, filename)
        self.mmap_buffers[name] = mmap_buffer
        
        logger.debug(f"Created memory-mapped buffer '{name}' ({size_bytes} bytes)")
        return mmap_buffer
    
    def get_pooled_object(self, size_bytes: int) -> Any:
        """Get object from appropriate pool based on size"""
        if size_bytes <= 1024:
            pool_type = MemoryPoolType.SMALL
        elif size_bytes <= 100 * 1024:
            pool_type = MemoryPoolType.MEDIUM
        else:
            pool_type = MemoryPoolType.LARGE
        
        return self.pools[pool_type].acquire()
    
    def return_pooled_object(self, obj: Any, size_bytes: int) -> None:
        """Return object to appropriate pool"""
        if size_bytes <= 1024:
            pool_type = MemoryPoolType.SMALL
        elif size_bytes <= 100 * 1024:
            pool_type = MemoryPoolType.MEDIUM
        else:
            pool_type = MemoryPoolType.LARGE
        
        self.pools[pool_type].release(obj)
    
    def perform_maintenance(self) -> None:
        """Perform memory maintenance operations"""
        # GC management
        gc_pause = self.gc_manager.adaptive_collect()
        
        # Pool cleanup
        for pool in self.pools.values():
            cleaned = pool.cleanup_stale_objects(max_age_seconds=300.0)
            if cleaned > 0:
                logger.debug(f"Cleaned {cleaned} stale objects from {pool.pool_type.value} pool")
        
        # Check memory usage
        current_usage = self.get_current_memory_usage()
        if current_usage > self.memory_limit_bytes * 0.9:  # 90% of limit
            logger.warning(f"Memory usage high: {current_usage / 1024 / 1024:.1f}MB")
            self._aggressive_cleanup()
    
    def _aggressive_cleanup(self) -> None:
        """Perform aggressive cleanup when memory usage is high"""
        logger.info("Performing aggressive memory cleanup")
        
        # Clear old statistics
        while len(self.stats_history) > 100:
            self.stats_history.popleft()
        
        # Clean all pools
        for pool in self.pools.values():
            pool.cleanup_stale_objects(max_age_seconds=60.0)  # More aggressive
        
        # Force GC
        for generation in range(3):
            gc.collect(generation)
        
        # Defragment if possible
        try:
            gc.collect()
            gc.collect()  # Double collection sometimes helps
        except Exception as e:
            logger.error(f"Error during aggressive cleanup: {e}")
    
    def get_current_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        process = psutil.Process()
        return process.memory_info().rss
    
    def get_memory_stats(self) -> MemoryStats:
        """Get comprehensive memory statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        stats = MemoryStats(
            current_usage_mb=memory_info.rss / 1024 / 1024,
            peak_usage_mb=memory_info.peak_wss / 1024 / 1024 if hasattr(memory_info, 'peak_wss') else 0,
            gc_collections=sum(gc.get_count()),
            active_buffers=len(self.buffers) + len(self.mmap_buffers),
        )
        
        # Pool usage
        for pool_type, pool in self.pools.items():
            pool_stats = pool.get_stats()
            stats.pool_usage_mb[pool_type.value] = pool_stats['in_use'] * 0.001  # Rough estimate
        
        # Buffer overflow counts
        stats.buffer_overflow_count = sum(
            buffer.overflow_count for buffer in self.buffers.values()
        )
        
        # GC statistics
        gc_stats = self.gc_manager.get_gc_stats()
        if gc_stats:
            stats.gc_time_ms = gc_stats.get('avg_pause_ms', 0)
        
        self.stats_history.append(stats)
        return stats
    
    def _start_memory_monitoring(self) -> None:
        """Start background memory monitoring"""
        def monitor_loop():
            while self.memory_monitor_enabled:
                try:
                    self.get_memory_stats()
                    time.sleep(1.0)  # Monitor every second
                except Exception as e:
                    logger.error(f"Error in memory monitoring: {e}")
                    time.sleep(5.0)  # Back off on error
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.debug("Started memory monitoring thread")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system memory health"""
        stats = self.get_memory_stats()
        
        # Health indicators
        memory_health = stats.current_usage_mb < self.memory_limit_mb * 0.8
        gc_health = stats.gc_time_ms < 10.0  # GC pauses under 10ms
        buffer_health = stats.buffer_overflow_count < 100  # Less than 100 overflows
        
        return {
            'overall_healthy': memory_health and gc_health and buffer_health,
            'memory_healthy': memory_health,
            'gc_healthy': gc_health,
            'buffer_healthy': buffer_health,
            'current_usage_mb': stats.current_usage_mb,
            'usage_percentage': (stats.current_usage_mb / self.memory_limit_mb) * 100,
            'gc_pause_ms': stats.gc_time_ms,
            'buffer_overflows': stats.buffer_overflow_count
        }
    
    def cleanup(self) -> None:
        """Cleanup all resources"""
        logger.info("Cleaning up memory manager")
        
        self.memory_monitor_enabled = False
        
        # Cleanup buffers
        for buffer in self.buffers.values():
            buffer.clear()
        
        # Cleanup memory-mapped buffers
        for mmap_buffer in self.mmap_buffers.values():
            mmap_buffer.cleanup()
        
        # Force final GC
        gc.collect()
        
        logger.info("Memory manager cleanup completed")


# Example usage and testing
if __name__ == "__main__":
    # Test memory manager
    memory_manager = MemoryManager(memory_limit_mb=512)  # 512MB limit for testing
    
    # Create buffers
    sensor_buffer = memory_manager.create_buffer("sensors", capacity=100)
    perception_buffer = memory_manager.create_buffer("perception", capacity=50)
    
    # Test object pooling
    small_obj = memory_manager.get_pooled_object(512)  # Small object
    medium_obj = memory_manager.get_pooled_object(10 * 1024)  # Medium object
    
    # Use objects
    print(f"Got objects: small={len(small_obj)}, medium={len(medium_obj)}")
    
    # Return to pool
    memory_manager.return_pooled_object(small_obj, 512)
    memory_manager.return_pooled_object(medium_obj, 10 * 1024)
    
    # Test buffers
    for i in range(150):  # More than capacity to test overflow
        sensor_buffer.put(f"sensor_data_{i}")
    
    print(f"Buffer stats: {sensor_buffer.get_stats()}")
    
    # Memory statistics
    stats = memory_manager.get_memory_stats()
    print(f"Memory stats: {stats.current_usage_mb:.1f}MB")
    
    # Health check
    health = memory_manager.get_system_health()
    print(f"System healthy: {health['overall_healthy']}")
    
    # Cleanup
    memory_manager.cleanup()