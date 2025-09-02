"""
Caching and Precomputation System
Model-Based RL Human Intent Recognition System

This module provides intelligent caching mechanisms including result caching,
precomputation systems, cache invalidation, and distributed caching support.
"""

import hashlib
import pickle
import time
import threading
import os
import sqlite3
import json
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod
from functools import wraps, lru_cache
from contextlib import contextmanager
import numpy as np
import logging
from pathlib import Path
import weakref

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    max_memory_mb: float = 1000.0
    max_disk_mb: float = 5000.0
    default_ttl_seconds: int = 3600  # 1 hour
    enable_persistence: bool = True
    enable_compression: bool = True
    enable_statistics: bool = True
    cache_dir: str = "cache"
    enable_distributed: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    precompute_batch_size: int = 100
    background_cleanup_interval: int = 300  # 5 minutes


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    total_keys: int = 0
    oldest_key_age: float = 0.0
    hit_rate: float = 0.0
    precomputed_results: int = 0
    cache_saves: int = 0


class CacheKey:
    """Intelligent cache key generator."""
    
    @staticmethod
    def generate(func_name: str, args: tuple, kwargs: dict, 
                include_code_hash: bool = False) -> str:
        """Generate a cache key from function call."""
        # Create hashable representation
        key_data = {
            'function': func_name,
            'args': CacheKey._make_hashable(args),
            'kwargs': CacheKey._make_hashable(kwargs)
        }
        
        # Optionally include function code hash for invalidation
        if include_code_hash:
            key_data['code_hash'] = CacheKey._get_function_hash(func_name)
        
        # Create hash
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    @staticmethod
    def _make_hashable(obj: Any) -> Any:
        """Convert object to hashable representation."""
        if isinstance(obj, dict):
            return tuple(sorted((k, CacheKey._make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, (list, tuple)):
            return tuple(CacheKey._make_hashable(item) for item in obj)
        elif isinstance(obj, np.ndarray):
            return ('ndarray', obj.shape, obj.dtype.str, hash(obj.data.tobytes()))
        elif hasattr(obj, '__dict__'):
            return ('object', type(obj).__name__, CacheKey._make_hashable(obj.__dict__))
        else:
            return obj
    
    @staticmethod
    def _get_function_hash(func_name: str) -> str:
        """Get hash of function code."""
        # This is a simplified version - in practice, you'd want more robust code hashing
        return hashlib.md5(func_name.encode()).hexdigest()


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.stats = CacheStats()
    
    def get(self, key: str) -> Tuple[Any, bool]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value, timestamp, size = self.cache.pop(key)
                self.cache[key] = (value, timestamp, size)
                self.stats.hits += 1
                return value, True
            else:
                self.stats.misses += 1
                return None, False
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put item in cache."""
        import sys
        
        with self.lock:
            # Calculate size
            size = sys.getsizeof(value)
            timestamp = time.time()
            
            # Remove if already exists
            if key in self.cache:
                old_value, old_timestamp, old_size = self.cache.pop(key)
                self.stats.memory_usage_mb -= old_size / (1024 * 1024)
            
            # Add new item
            self.cache[key] = (value, timestamp, size)
            self.stats.memory_usage_mb += size / (1024 * 1024)
            self.stats.total_keys = len(self.cache)
            
            # Evict if necessary
            while len(self.cache) > self.max_size:
                self._evict_oldest()
    
    def _evict_oldest(self):
        """Evict oldest item."""
        if self.cache:
            key, (value, timestamp, size) = self.cache.popitem(last=False)
            self.stats.memory_usage_mb -= size / (1024 * 1024)
            self.stats.evictions += 1
            self.stats.total_keys -= 1
    
    def clear(self):
        """Clear all items."""
        with self.lock:
            self.cache.clear()
            self.stats = CacheStats()
    
    def cleanup_expired(self, current_time: float, ttl: int):
        """Remove expired items."""
        with self.lock:
            expired_keys = [
                key for key, (value, timestamp, size) in self.cache.items()
                if current_time - timestamp > ttl
            ]
            
            for key in expired_keys:
                value, timestamp, size = self.cache.pop(key)
                self.stats.memory_usage_mb -= size / (1024 * 1024)
                self.stats.total_keys -= 1


class DiskCache:
    """Persistent disk-based cache."""
    
    def __init__(self, cache_dir: str, max_size_mb: float = 1000.0):
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = self.cache_dir / "cache_metadata.db"
        self._init_database()
        
        self.lock = threading.RLock()
        self.stats = CacheStats()
        
        # Update initial stats
        self._update_disk_usage()
    
    def _init_database(self):
        """Initialize SQLite database for metadata."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key TEXT PRIMARY KEY,
                filepath TEXT NOT NULL,
                timestamp REAL NOT NULL,
                size INTEGER NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_access REAL NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
    
    def get(self, key: str) -> Tuple[Any, bool]:
        """Get item from disk cache."""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute(
                "SELECT filepath, timestamp FROM cache_metadata WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            
            if row:
                filepath, timestamp = row
                file_path = self.cache_dir / filepath
                
                if file_path.exists():
                    try:
                        with open(file_path, 'rb') as f:
                            value = pickle.load(f)
                        
                        # Update access statistics
                        conn.execute(
                            "UPDATE cache_metadata SET access_count = access_count + 1, "
                            "last_access = ? WHERE key = ?",
                            (time.time(), key)
                        )
                        conn.commit()
                        
                        self.stats.hits += 1
                        return value, True
                    except Exception:
                        # Remove corrupted entry
                        self._remove_key(key, conn)
                else:
                    # Remove missing file entry
                    self._remove_key(key, conn)
            
            conn.close()
            self.stats.misses += 1
            return None, False
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put item in disk cache."""
        with self.lock:
            # Create unique filename
            filename = f"{key[:16]}_{int(time.time())}.pkl"
            filepath = self.cache_dir / filename
            
            # Serialize to disk
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(value, f)
                
                size = filepath.stat().st_size
                timestamp = time.time()
                
                # Update database
                conn = sqlite3.connect(str(self.db_path))
                
                # Remove old entry if exists
                self._remove_key(key, conn)
                
                # Add new entry
                conn.execute(
                    "INSERT INTO cache_metadata "
                    "(key, filepath, timestamp, size, last_access) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (key, filename, timestamp, size, timestamp)
                )
                conn.commit()
                conn.close()
                
                # Update stats
                self.stats.disk_usage_mb += size / (1024 * 1024)
                self.stats.total_keys += 1
                
                # Cleanup if over limit
                self._cleanup_if_needed()
                
            except Exception as e:
                # Cleanup on failure
                if filepath.exists():
                    filepath.unlink()
                raise e
    
    def _remove_key(self, key: str, conn: sqlite3.Connection = None):
        """Remove key and associated file."""
        if conn is None:
            conn = sqlite3.connect(str(self.db_path))
            close_conn = True
        else:
            close_conn = False
        
        cursor = conn.execute(
            "SELECT filepath, size FROM cache_metadata WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()
        
        if row:
            filepath, size = row
            file_path = self.cache_dir / filepath
            
            # Remove file
            if file_path.exists():
                file_path.unlink()
            
            # Remove database entry
            conn.execute("DELETE FROM cache_metadata WHERE key = ?", (key,))
            
            # Update stats
            self.stats.disk_usage_mb -= size / (1024 * 1024)
            self.stats.total_keys -= 1
        
        if close_conn:
            conn.commit()
            conn.close()
    
    def _cleanup_if_needed(self):
        """Cleanup cache if over size limit."""
        if self.stats.disk_usage_mb > self.max_size_mb:
            conn = sqlite3.connect(str(self.db_path))
            
            # Get least recently used items
            cursor = conn.execute(
                "SELECT key FROM cache_metadata ORDER BY last_access ASC"
            )
            
            while self.stats.disk_usage_mb > self.max_size_mb * 0.8:  # Clean to 80%
                row = cursor.fetchone()
                if not row:
                    break
                
                self._remove_key(row[0], conn)
                self.stats.evictions += 1
            
            conn.commit()
            conn.close()
    
    def _update_disk_usage(self):
        """Update disk usage statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute("SELECT SUM(size), COUNT(*) FROM cache_metadata")
        row = cursor.fetchone()
        
        if row[0]:
            self.stats.disk_usage_mb = row[0] / (1024 * 1024)
            self.stats.total_keys = row[1]
        
        conn.close()
    
    def cleanup_expired(self, ttl: int):
        """Remove expired items."""
        current_time = time.time()
        
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            
            # Get expired keys
            cursor = conn.execute(
                "SELECT key FROM cache_metadata WHERE ? - timestamp > ?",
                (current_time, ttl)
            )
            
            expired_keys = [row[0] for row in cursor.fetchall()]
            
            for key in expired_keys:
                self._remove_key(key, conn)
            
            conn.commit()
            conn.close()


class DistributedCache:
    """Redis-based distributed cache."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available for distributed caching")
        
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        self.stats = CacheStats()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Tuple[Any, bool]:
        """Get item from distributed cache."""
        try:
            data = self.redis_client.get(key)
            if data:
                value = pickle.loads(data)
                with self.lock:
                    self.stats.hits += 1
                return value, True
            else:
                with self.lock:
                    self.stats.misses += 1
                return None, False
        except Exception:
            with self.lock:
                self.stats.misses += 1
            return None, False
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put item in distributed cache."""
        try:
            data = pickle.dumps(value)
            if ttl:
                self.redis_client.setex(key, ttl, data)
            else:
                self.redis_client.set(key, data)
            
            with self.lock:
                self.stats.total_keys += 1
        except Exception:
            pass  # Fail silently for distributed cache
    
    def clear(self):
        """Clear distributed cache."""
        self.redis_client.flushdb()
        with self.lock:
            self.stats = CacheStats()


class PrecomputationEngine:
    """Engine for precomputing and caching results."""
    
    def __init__(self, cache_system: 'CacheSystem'):
        self.cache_system = cache_system
        self.precomputation_tasks = {}
        self.task_queue = []
        self.worker_threads = []
        self.running = False
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
    
    def register_precomputation(self, name: str, func: Callable, 
                               input_generator: Callable, batch_size: int = 100,
                               priority: int = 1):
        """Register a function for precomputation."""
        self.precomputation_tasks[name] = {
            'function': func,
            'input_generator': input_generator,
            'batch_size': batch_size,
            'priority': priority,
            'completed': 0,
            'total': 0
        }
        
        self.logger.info(f"Registered precomputation task: {name}")
    
    def start_precomputation(self, num_workers: int = 2):
        """Start precomputation workers."""
        if self.running:
            return
        
        self.running = True
        
        # Generate tasks
        self._generate_tasks()
        
        # Start worker threads
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.worker_threads.append(worker)
        
        self.logger.info(f"Started {num_workers} precomputation workers")
    
    def stop_precomputation(self):
        """Stop precomputation workers."""
        self.running = False
        
        for worker in self.worker_threads:
            if worker.is_alive():
                worker.join(timeout=5.0)
        
        self.worker_threads.clear()
        self.logger.info("Stopped precomputation workers")
    
    def _generate_tasks(self):
        """Generate precomputation tasks."""
        for name, task_info in self.precomputation_tasks.items():
            try:
                inputs = task_info['input_generator']()
                
                # Split into batches
                batch_size = task_info['batch_size']
                batches = [inputs[i:i+batch_size] for i in range(0, len(inputs), batch_size)]
                
                for batch in batches:
                    self.task_queue.append({
                        'name': name,
                        'function': task_info['function'],
                        'inputs': batch,
                        'priority': task_info['priority']
                    })
                
                task_info['total'] = len(batches)
                
            except Exception as e:
                self.logger.error(f"Error generating tasks for {name}: {e}")
        
        # Sort by priority
        self.task_queue.sort(key=lambda x: x['priority'], reverse=True)
    
    def _worker_loop(self, worker_id: int):
        """Main worker loop."""
        while self.running and self.task_queue:
            try:
                with self.lock:
                    if not self.task_queue:
                        break
                    task = self.task_queue.pop(0)
                
                self._process_task(task, worker_id)
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
    
    def _process_task(self, task: Dict[str, Any], worker_id: int):
        """Process a precomputation task."""
        name = task['name']
        func = task['function']
        inputs = task['inputs']
        
        for input_data in inputs:
            try:
                # Generate cache key
                cache_key = CacheKey.generate(func.__name__, (input_data,), {})
                
                # Check if already cached
                if self.cache_system.get(cache_key)[1]:
                    continue  # Already cached
                
                # Compute result
                result = func(input_data)
                
                # Cache result
                self.cache_system.put(cache_key, result)
                
                # Update statistics
                with self.lock:
                    self.precomputation_tasks[name]['completed'] += 1
                
            except Exception as e:
                self.logger.error(f"Precomputation error for {name}: {e}")
    
    def get_progress(self) -> Dict[str, Dict[str, int]]:
        """Get precomputation progress."""
        progress = {}
        for name, task_info in self.precomputation_tasks.items():
            progress[name] = {
                'completed': task_info['completed'],
                'total': task_info['total'],
                'percentage': (task_info['completed'] / max(1, task_info['total'])) * 100
            }
        return progress


class CacheSystem:
    """Main caching system orchestrator."""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        
        # Initialize cache layers
        self.memory_cache = LRUCache(max_size=1000)
        
        if self.config.enable_persistence:
            self.disk_cache = DiskCache(
                self.config.cache_dir, 
                self.config.max_disk_mb
            )
        else:
            self.disk_cache = None
        
        if self.config.enable_distributed and REDIS_AVAILABLE:
            try:
                self.distributed_cache = DistributedCache(
                    self.config.redis_host,
                    self.config.redis_port,
                    self.config.redis_db
                )
            except Exception:
                self.distributed_cache = None
        else:
            self.distributed_cache = None
        
        # Initialize precomputation engine
        self.precomputation_engine = PrecomputationEngine(self)
        
        # Background cleanup thread
        self.cleanup_thread = None
        self.running = False
        
        # Statistics
        self.global_stats = CacheStats()
        
        self.logger = logging.getLogger(__name__)
        
        if self.config.enable_statistics:
            self._start_background_cleanup()
    
    def get(self, key: str) -> Tuple[Any, bool]:
        """Get item from cache hierarchy."""
        # Try memory cache first
        value, found = self.memory_cache.get(key)
        if found:
            self.global_stats.hits += 1
            return value, True
        
        # Try disk cache
        if self.disk_cache:
            value, found = self.disk_cache.get(key)
            if found:
                # Promote to memory cache
                self.memory_cache.put(key, value)
                self.global_stats.hits += 1
                return value, True
        
        # Try distributed cache
        if self.distributed_cache:
            value, found = self.distributed_cache.get(key)
            if found:
                # Promote to memory and disk cache
                self.memory_cache.put(key, value)
                if self.disk_cache:
                    self.disk_cache.put(key, value)
                self.global_stats.hits += 1
                return value, True
        
        self.global_stats.misses += 1
        return None, False
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put item in cache hierarchy."""
        ttl = ttl or self.config.default_ttl_seconds
        
        # Store in all available caches
        self.memory_cache.put(key, value, ttl)
        
        if self.disk_cache:
            self.disk_cache.put(key, value, ttl)
        
        if self.distributed_cache:
            self.distributed_cache.put(key, value, ttl)
        
        self.global_stats.cache_saves += 1
    
    def invalidate(self, key: str) -> None:
        """Invalidate item from all caches."""
        # Remove from memory
        with self.memory_cache.lock:
            if key in self.memory_cache.cache:
                value, timestamp, size = self.memory_cache.cache.pop(key)
                self.memory_cache.stats.memory_usage_mb -= size / (1024 * 1024)
                self.memory_cache.stats.total_keys -= 1
        
        # Remove from disk
        if self.disk_cache:
            self.disk_cache._remove_key(key)
        
        # Remove from distributed cache
        if self.distributed_cache:
            try:
                self.distributed_cache.redis_client.delete(key)
            except Exception:
                pass
    
    def clear_all(self):
        """Clear all caches."""
        self.memory_cache.clear()
        
        if self.disk_cache:
            # Clear database and files
            import shutil
            shutil.rmtree(self.config.cache_dir, ignore_errors=True)
            os.makedirs(self.config.cache_dir, exist_ok=True)
            self.disk_cache._init_database()
        
        if self.distributed_cache:
            self.distributed_cache.clear()
        
        self.global_stats = CacheStats()
    
    def cached(self, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
        """Decorator for caching function results."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = CacheKey.generate(func.__name__, args, kwargs)
                
                # Try to get from cache
                result, found = self.get(cache_key)
                if found:
                    return result
                
                # Compute result
                result = func(*args, **kwargs)
                
                # Store in cache
                self.put(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def precompute(self, name: str, func: Callable, input_generator: Callable,
                  batch_size: int = None):
        """Register function for precomputation."""
        batch_size = batch_size or self.config.precompute_batch_size
        self.precomputation_engine.register_precomputation(
            name, func, input_generator, batch_size
        )
    
    def start_precomputation(self, num_workers: int = 2):
        """Start precomputation."""
        self.precomputation_engine.start_precomputation(num_workers)
    
    def stop_precomputation(self):
        """Stop precomputation."""
        self.precomputation_engine.stop_precomputation()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            'global': {
                'hits': self.global_stats.hits,
                'misses': self.global_stats.misses,
                'hit_rate': self.global_stats.hits / max(1, self.global_stats.hits + self.global_stats.misses),
                'cache_saves': self.global_stats.cache_saves
            },
            'memory': {
                'hits': self.memory_cache.stats.hits,
                'misses': self.memory_cache.stats.misses,
                'evictions': self.memory_cache.stats.evictions,
                'usage_mb': self.memory_cache.stats.memory_usage_mb,
                'keys': len(self.memory_cache.cache)
            }
        }
        
        if self.disk_cache:
            stats['disk'] = {
                'hits': self.disk_cache.stats.hits,
                'misses': self.disk_cache.stats.misses,
                'evictions': self.disk_cache.stats.evictions,
                'usage_mb': self.disk_cache.stats.disk_usage_mb,
                'keys': self.disk_cache.stats.total_keys
            }
        
        if self.distributed_cache:
            stats['distributed'] = {
                'hits': self.distributed_cache.stats.hits,
                'misses': self.distributed_cache.stats.misses
            }
        
        # Precomputation progress
        stats['precomputation'] = self.precomputation_engine.get_progress()
        
        return stats
    
    def _start_background_cleanup(self):
        """Start background cleanup thread."""
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while self.running:
            try:
                time.sleep(self.config.background_cleanup_interval)
                
                # Cleanup expired items
                ttl = self.config.default_ttl_seconds
                self.memory_cache.cleanup_expired(time.time(), ttl)
                
                if self.disk_cache:
                    self.disk_cache.cleanup_expired(ttl)
                
            except Exception as e:
                self.logger.error(f"Background cleanup error: {e}")
    
    def cleanup(self):
        """Cleanup cache system."""
        self.running = False
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        
        self.precomputation_engine.stop_precomputation()
        
        self.logger.info("Cache system cleaned up")


# Example usage functions
def create_example_cached_functions(cache_system: CacheSystem):
    """Create example cached functions."""
    
    @cache_system.cached(ttl=3600)
    def expensive_computation(n: int) -> float:
        """Simulate expensive computation."""
        time.sleep(0.1)  # Simulate computation time
        return sum(i**2 for i in range(n))
    
    @cache_system.cached(ttl=1800)
    def matrix_operation(matrix: np.ndarray) -> np.ndarray:
        """Cached matrix operation."""
        return matrix @ matrix.T
    
    return expensive_computation, matrix_operation


def test_caching_system():
    """Test the caching system."""
    config = CacheConfig(
        max_memory_mb=100.0,
        max_disk_mb=500.0,
        enable_persistence=True,
        enable_statistics=True
    )
    
    cache_system = CacheSystem(config)
    
    # Create cached functions
    expensive_comp, matrix_op = create_example_cached_functions(cache_system)
    
    print("Testing caching system...")
    
    # Test expensive computation
    start_time = time.time()
    result1 = expensive_comp(1000)
    first_call_time = time.time() - start_time
    
    start_time = time.time()
    result2 = expensive_comp(1000)  # Should be cached
    second_call_time = time.time() - start_time
    
    print(f"First call: {first_call_time:.3f}s")
    print(f"Second call (cached): {second_call_time:.3f}s")
    print(f"Speedup: {first_call_time / second_call_time:.1f}x")
    
    # Test matrix operations
    matrix = np.random.randn(100, 100)
    
    start_time = time.time()
    result_matrix1 = matrix_op(matrix)
    matrix_first_time = time.time() - start_time
    
    start_time = time.time()
    result_matrix2 = matrix_op(matrix)  # Should be cached
    matrix_second_time = time.time() - start_time
    
    print(f"Matrix first call: {matrix_first_time:.3f}s")
    print(f"Matrix second call (cached): {matrix_second_time:.3f}s")
    
    # Show statistics
    stats = cache_system.get_statistics()
    print(f"\nCache Statistics:")
    print(f"Hit rate: {stats['global']['hit_rate']:.2%}")
    print(f"Memory usage: {stats['memory']['usage_mb']:.1f}MB")
    if 'disk' in stats:
        print(f"Disk usage: {stats['disk']['usage_mb']:.1f}MB")
    
    # Cleanup
    cache_system.cleanup()


if __name__ == "__main__":
    test_caching_system()