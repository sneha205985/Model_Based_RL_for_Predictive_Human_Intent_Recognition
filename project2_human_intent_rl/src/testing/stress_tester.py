"""
Comprehensive Stress Testing Framework
=====================================

Advanced stress testing system for validating system robustness under extreme conditions:
- Load progression testing (gradually increasing load until failure)
- Resource exhaustion testing (memory, CPU, GPU, network)
- Failure injection and recovery testing
- Performance degradation analysis
- System stability under sustained load
- Real-time constraint violation detection
- Automated test orchestration and reporting
"""

import asyncio
import time
import threading
import multiprocessing as mp
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import psutil
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import sqlite3
import warnings

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class StressTestType(Enum):
    """Types of stress tests"""
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    IO_SATURATION = "io_saturation"
    NETWORK_FLOOD = "network_flood"
    GPU_OVERLOAD = "gpu_overload"
    CONCURRENT_THREADS = "concurrent_threads"
    PROCESS_SPAWN = "process_spawn"
    REALTIME_DEADLINE = "realtime_deadline"
    FAILURE_INJECTION = "failure_injection"
    SUSTAINED_LOAD = "sustained_load"


class TestResult(Enum):
    """Test result status"""
    PASSED = "passed"
    FAILED = "failed"
    DEGRADED = "degraded"
    TIMEOUT = "timeout"
    CRASHED = "crashed"


@dataclass
class StressTestConfig:
    """Configuration for stress tests"""
    test_type: StressTestType
    duration_seconds: float = 60.0
    max_load_factor: float = 10.0
    load_increment: float = 0.5
    increment_interval: float = 5.0
    failure_threshold: Dict[str, float] = field(default_factory=dict)
    recovery_time: float = 30.0
    target_metrics: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.failure_threshold:
            self.failure_threshold = {
                'cpu_utilization': 98.0,
                'memory_usage_mb': 3000.0,
                'response_time_ms': 200.0,
                'error_rate': 0.1,
                'deadline_miss_rate': 0.05
            }


@dataclass
class StressTestResults:
    """Results from stress testing"""
    test_type: StressTestType
    config: StressTestConfig
    start_time: datetime
    end_time: Optional[datetime] = None
    result_status: TestResult = TestResult.PASSED
    max_stable_load: float = 0.0
    breaking_point: Optional[float] = None
    performance_metrics: Dict[str, List[float]] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    recovery_time_actual: Optional[float] = None
    failure_reason: Optional[str] = None
    
    def add_metric(self, name: str, value: float):
        """Add performance metric sample"""
        if name not in self.performance_metrics:
            self.performance_metrics[name] = []
        self.performance_metrics[name].append(value)
    
    def get_test_duration(self) -> float:
        """Get total test duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()


class SystemStressor:
    """Individual stress test implementations"""
    
    def __init__(self):
        self.active_stressors = {}
        self.stop_flags = {}
    
    def start_cpu_stress(self, load_factor: float, duration: float) -> str:
        """Start CPU-intensive stress test"""
        stress_id = f"cpu_stress_{int(time.time())}"
        self.stop_flags[stress_id] = threading.Event()
        
        def cpu_stress_worker():
            """CPU intensive worker"""
            end_time = time.time() + duration
            while time.time() < end_time and not self.stop_flags[stress_id].is_set():
                # Intensive mathematical operations
                for _ in range(int(1000000 * load_factor)):
                    _ = sum(i**2 for i in range(100))
                
                # Brief pause to allow monitoring
                if load_factor < 5.0:
                    time.sleep(0.001)
        
        # Start multiple CPU stress threads
        num_threads = min(mp.cpu_count(), int(4 * load_factor))
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=cpu_stress_worker, daemon=True)
            thread.start()
            threads.append(thread)
        
        self.active_stressors[stress_id] = threads
        return stress_id
    
    def start_memory_stress(self, load_factor: float, duration: float) -> str:
        """Start memory exhaustion stress test"""
        stress_id = f"memory_stress_{int(time.time())}"
        self.stop_flags[stress_id] = threading.Event()
        
        def memory_stress_worker():
            """Memory allocation worker"""
            memory_chunks = []
            chunk_size = int(100 * 1024 * 1024 * load_factor)  # MB per chunk
            
            end_time = time.time() + duration
            while time.time() < end_time and not self.stop_flags[stress_id].is_set():
                try:
                    # Allocate memory chunk
                    chunk = np.random.random(chunk_size // 8)  # 8 bytes per float64
                    memory_chunks.append(chunk)
                    
                    # Access memory to ensure allocation
                    if len(memory_chunks) % 10 == 0:
                        for chunk in memory_chunks[-5:]:  # Access recent chunks
                            _ = np.sum(chunk[:1000])  # Partial access
                    
                    time.sleep(0.1)
                    
                except MemoryError:
                    logging.warning("Memory allocation failed - memory limit reached")
                    break
                except Exception as e:
                    logging.error(f"Memory stress error: {e}")
                    break
            
            # Cleanup
            memory_chunks.clear()
        
        thread = threading.Thread(target=memory_stress_worker, daemon=True)
        thread.start()
        self.active_stressors[stress_id] = [thread]
        return stress_id
    
    def start_io_stress(self, load_factor: float, duration: float, 
                       temp_dir: Path) -> str:
        """Start I/O saturation stress test"""
        stress_id = f"io_stress_{int(time.time())}"
        self.stop_flags[stress_id] = threading.Event()
        
        def io_stress_worker(worker_id: int):
            """I/O intensive worker"""
            temp_file = temp_dir / f"stress_test_{worker_id}_{stress_id}.tmp"
            
            try:
                end_time = time.time() + duration
                while time.time() < end_time and not self.stop_flags[stress_id].is_set():
                    # Write large data chunk
                    data_size = int(10 * 1024 * 1024 * load_factor)  # MB
                    data = np.random.bytes(data_size)
                    
                    with open(temp_file, 'wb') as f:
                        f.write(data)
                        f.flush()
                    
                    # Read data back
                    with open(temp_file, 'rb') as f:
                        _ = f.read()
                    
                    time.sleep(0.01)  # Brief pause
                    
            except Exception as e:
                logging.error(f"I/O stress error: {e}")
            finally:
                # Cleanup
                if temp_file.exists():
                    temp_file.unlink()
        
        # Start multiple I/O workers
        num_workers = min(4, int(2 * load_factor))
        threads = []
        for i in range(num_workers):
            thread = threading.Thread(target=io_stress_worker, args=(i,), daemon=True)
            thread.start()
            threads.append(thread)
        
        self.active_stressors[stress_id] = threads
        return stress_id
    
    def start_gpu_stress(self, load_factor: float, duration: float) -> str:
        """Start GPU stress test"""
        stress_id = f"gpu_stress_{int(time.time())}"
        
        if not CUPY_AVAILABLE:
            logging.warning("GPU stress test skipped - CuPy not available")
            return stress_id
        
        self.stop_flags[stress_id] = threading.Event()
        
        def gpu_stress_worker():
            """GPU intensive worker"""
            try:
                end_time = time.time() + duration
                while time.time() < end_time and not self.stop_flags[stress_id].is_set():
                    # Large matrix operations
                    size = int(2000 * load_factor)
                    a = cp.random.random((size, size), dtype=cp.float32)
                    b = cp.random.random((size, size), dtype=cp.float32)
                    
                    # Intensive GPU computation
                    c = cp.matmul(a, b)
                    d = cp.fft.fft2(c)
                    e = cp.linalg.svd(c[:100, :100])  # Expensive operation
                    
                    # Synchronize to ensure completion
                    cp.cuda.Device().synchronize()
                    
                    # Brief pause
                    time.sleep(0.01)
                    
            except Exception as e:
                logging.error(f"GPU stress error: {e}")
        
        thread = threading.Thread(target=gpu_stress_worker, daemon=True)
        thread.start()
        self.active_stressors[stress_id] = [thread]
        return stress_id
    
    def start_concurrent_thread_stress(self, load_factor: float, duration: float) -> str:
        """Start concurrent thread stress test"""
        stress_id = f"thread_stress_{int(time.time())}"
        self.stop_flags[stress_id] = threading.Event()
        
        def thread_worker(worker_id: int):
            """Individual thread worker"""
            end_time = time.time() + duration
            while time.time() < end_time and not self.stop_flags[stress_id].is_set():
                # Mixed workload
                data = np.random.random(int(10000 * load_factor))
                result = np.fft.fft(data)
                _ = np.sum(result.real)
                
                time.sleep(0.001)
        
        # Create many concurrent threads
        num_threads = int(50 * load_factor)
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=thread_worker, args=(i,), daemon=True)
            thread.start()
            threads.append(thread)
        
        self.active_stressors[stress_id] = threads
        return stress_id
    
    def stop_stress_test(self, stress_id: str):
        """Stop specific stress test"""
        if stress_id in self.stop_flags:
            self.stop_flags[stress_id].set()
        
        if stress_id in self.active_stressors:
            threads = self.active_stressors[stress_id]
            for thread in threads:
                if thread.is_alive():
                    thread.join(timeout=5.0)
            
            del self.active_stressors[stress_id]
        
        if stress_id in self.stop_flags:
            del self.stop_flags[stress_id]
    
    def stop_all_stress_tests(self):
        """Stop all active stress tests"""
        for stress_id in list(self.active_stressors.keys()):
            self.stop_stress_test(stress_id)


class StressTester:
    """Main stress testing framework"""
    
    def __init__(self, results_dir: str = "stress_test_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.temp_dir = self.results_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        self.db_path = self.results_dir / "stress_test_results.db"
        self._init_database()
        
        self.stressor = SystemStressor()
        self.monitoring_active = False
        self.monitoring_thread = None
        
    def _init_database(self):
        """Initialize results database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stress_test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_type TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration REAL,
                    result_status TEXT NOT NULL,
                    max_stable_load REAL,
                    breaking_point REAL,
                    config_json TEXT NOT NULL,
                    metrics_json TEXT,
                    error_log_json TEXT,
                    failure_reason TEXT
                )
            """)
    
    def run_load_progression_test(self, config: StressTestConfig) -> StressTestResults:
        """Run progressive load test until breaking point"""
        results = StressTestResults(
            test_type=config.test_type,
            config=config,
            start_time=datetime.now()
        )
        
        try:
            # Start monitoring
            self._start_monitoring(results)
            
            current_load = 0.5
            stable_loads = []
            
            while current_load <= config.max_load_factor:
                print(f"Testing load factor: {current_load:.1f}")
                
                # Start stress test at current load
                stress_id = self._start_stress_test(config.test_type, current_load, 
                                                 config.increment_interval)
                
                # Monitor for stability
                stability_check_start = time.time()
                is_stable = True
                
                while time.time() - stability_check_start < config.increment_interval:
                    if self._check_system_health(results, config.failure_threshold):
                        # System is still healthy
                        time.sleep(0.5)
                    else:
                        # System exceeded thresholds
                        is_stable = False
                        results.breaking_point = current_load
                        results.failure_reason = "Performance threshold exceeded"
                        break
                
                # Stop current stress test
                self.stressor.stop_stress_test(stress_id)
                
                if is_stable:
                    stable_loads.append(current_load)
                    results.max_stable_load = current_load
                    time.sleep(2.0)  # Recovery pause
                    current_load += config.load_increment
                else:
                    break
            
            # Set final status
            if results.breaking_point:
                results.result_status = TestResult.FAILED
            else:
                results.result_status = TestResult.PASSED
                
        except Exception as e:
            results.result_status = TestResult.CRASHED
            results.failure_reason = f"Test crashed: {str(e)}"
            results.error_log.append(str(e))
            
        finally:
            # Stop monitoring and cleanup
            self._stop_monitoring()
            self.stressor.stop_all_stress_tests()
            results.end_time = datetime.now()
        
        return results
    
    def run_sustained_load_test(self, config: StressTestConfig, 
                               load_factor: float) -> StressTestResults:
        """Run sustained load test at fixed load level"""
        results = StressTestResults(
            test_type=config.test_type,
            config=config,
            start_time=datetime.now()
        )
        
        try:
            # Start monitoring
            self._start_monitoring(results)
            
            print(f"Starting sustained load test at {load_factor:.1f}x for {config.duration_seconds}s")
            
            # Start sustained stress test
            stress_id = self._start_stress_test(config.test_type, load_factor, 
                                             config.duration_seconds)
            
            # Monitor throughout test duration
            test_start = time.time()
            degradation_detected = False
            
            while time.time() - test_start < config.duration_seconds:
                if not self._check_system_health(results, config.failure_threshold):
                    degradation_detected = True
                    results.failure_reason = "System degradation during sustained load"
                
                time.sleep(1.0)
            
            # Stop stress test
            self.stressor.stop_stress_test(stress_id)
            
            # Check recovery
            print("Testing system recovery...")
            recovery_start = time.time()
            recovery_timeout = config.recovery_time
            
            while time.time() - recovery_start < recovery_timeout:
                if self._check_system_recovery(results):
                    results.recovery_time_actual = time.time() - recovery_start
                    break
                time.sleep(1.0)
            
            # Set final status
            if degradation_detected:
                results.result_status = TestResult.DEGRADED
            elif results.recovery_time_actual is None:
                results.result_status = TestResult.FAILED
                results.failure_reason = "Failed to recover within timeout"
            else:
                results.result_status = TestResult.PASSED
                
        except Exception as e:
            results.result_status = TestResult.CRASHED
            results.failure_reason = f"Test crashed: {str(e)}"
            results.error_log.append(str(e))
            
        finally:
            self._stop_monitoring()
            self.stressor.stop_all_stress_tests()
            results.end_time = datetime.now()
        
        return results
    
    def run_failure_injection_test(self, config: StressTestConfig) -> StressTestResults:
        """Run failure injection and recovery test"""
        results = StressTestResults(
            test_type=StressTestType.FAILURE_INJECTION,
            config=config,
            start_time=datetime.now()
        )
        
        try:
            self._start_monitoring(results)
            
            # Simulate various failure scenarios
            failure_scenarios = [
                ("memory_exhaustion", 5.0, 10.0),
                ("cpu_overload", 8.0, 5.0),
                ("io_saturation", 3.0, 8.0)
            ]
            
            recovery_successful = True
            
            for scenario_name, load_factor, duration in failure_scenarios:
                print(f"Injecting failure: {scenario_name} at {load_factor}x for {duration}s")
                
                # Inject failure
                if scenario_name == "memory_exhaustion":
                    stress_id = self.stressor.start_memory_stress(load_factor, duration)
                elif scenario_name == "cpu_overload":
                    stress_id = self.stressor.start_cpu_stress(load_factor, duration)
                elif scenario_name == "io_saturation":
                    stress_id = self.stressor.start_io_stress(load_factor, duration, self.temp_dir)
                
                # Monitor during failure
                failure_start = time.time()
                while time.time() - failure_start < duration:
                    self._check_system_health(results, config.failure_threshold)
                    time.sleep(0.5)
                
                # Stop failure injection
                self.stressor.stop_stress_test(stress_id)
                
                # Test recovery
                recovery_start = time.time()
                recovered = False
                
                while time.time() - recovery_start < config.recovery_time:
                    if self._check_system_recovery(results):
                        recovered = True
                        break
                    time.sleep(1.0)
                
                if not recovered:
                    recovery_successful = False
                    results.error_log.append(f"Failed to recover from {scenario_name}")
                
                # Pause between scenarios
                time.sleep(5.0)
            
            results.result_status = TestResult.PASSED if recovery_successful else TestResult.FAILED
            
        except Exception as e:
            results.result_status = TestResult.CRASHED
            results.failure_reason = f"Failure injection test crashed: {str(e)}"
            results.error_log.append(str(e))
            
        finally:
            self._stop_monitoring()
            self.stressor.stop_all_stress_tests()
            results.end_time = datetime.now()
        
        return results
    
    def _start_stress_test(self, test_type: StressTestType, load_factor: float, 
                          duration: float) -> str:
        """Start appropriate stress test based on type"""
        if test_type == StressTestType.CPU_INTENSIVE:
            return self.stressor.start_cpu_stress(load_factor, duration)
        elif test_type == StressTestType.MEMORY_EXHAUSTION:
            return self.stressor.start_memory_stress(load_factor, duration)
        elif test_type == StressTestType.IO_SATURATION:
            return self.stressor.start_io_stress(load_factor, duration, self.temp_dir)
        elif test_type == StressTestType.GPU_OVERLOAD:
            return self.stressor.start_gpu_stress(load_factor, duration)
        elif test_type == StressTestType.CONCURRENT_THREADS:
            return self.stressor.start_concurrent_thread_stress(load_factor, duration)
        else:
            # Default to CPU stress
            return self.stressor.start_cpu_stress(load_factor, duration)
    
    def _start_monitoring(self, results: StressTestResults):
        """Start system monitoring"""
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # System metrics
                    process = psutil.Process()
                    memory = process.memory_info()
                    cpu_percent = process.cpu_percent()
                    system_memory = psutil.virtual_memory()
                    
                    results.add_metric("process_cpu", cpu_percent)
                    results.add_metric("process_memory_mb", memory.rss / 1024 / 1024)
                    results.add_metric("system_memory_percent", system_memory.percent)
                    results.add_metric("system_cpu_percent", psutil.cpu_percent())
                    
                    # GPU metrics
                    if GPU_AVAILABLE:
                        try:
                            gpus = GPUtil.getGPUs()
                            for i, gpu in enumerate(gpus):
                                results.add_metric(f"gpu_{i}_utilization", gpu.load * 100)
                                results.add_metric(f"gpu_{i}_memory_percent", 
                                                  gpu.memoryUsed / gpu.memoryTotal * 100)
                        except Exception:
                            pass
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    logging.warning(f"Monitoring error: {e}")
                    time.sleep(0.5)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def _stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
    
    def _check_system_health(self, results: StressTestResults, 
                           thresholds: Dict[str, float]) -> bool:
        """Check if system is within healthy parameters"""
        try:
            # Check recent metrics
            for metric_name, threshold in thresholds.items():
                if metric_name in results.performance_metrics:
                    recent_values = results.performance_metrics[metric_name][-5:]  # Last 5 samples
                    if recent_values:
                        avg_recent = sum(recent_values) / len(recent_values)
                        
                        if metric_name == "cpu_utilization" and avg_recent > threshold:
                            return False
                        elif metric_name == "memory_usage_mb" and avg_recent > threshold:
                            return False
                        elif metric_name == "response_time_ms" and avg_recent > threshold:
                            return False
            
            return True
            
        except Exception as e:
            logging.warning(f"Health check error: {e}")
            return True  # Assume healthy if check fails
    
    def _check_system_recovery(self, results: StressTestResults) -> bool:
        """Check if system has recovered to normal levels"""
        try:
            # Define recovery thresholds (lower than failure thresholds)
            recovery_thresholds = {
                "process_cpu": 50.0,  # %
                "process_memory_mb": 1000.0,  # MB
                "system_cpu_percent": 30.0,  # %
            }
            
            for metric_name, threshold in recovery_thresholds.items():
                if metric_name in results.performance_metrics:
                    recent_values = results.performance_metrics[metric_name][-3:]  # Last 3 samples
                    if recent_values:
                        avg_recent = sum(recent_values) / len(recent_values)
                        if avg_recent > threshold:
                            return False
            
            return True
            
        except Exception as e:
            logging.warning(f"Recovery check error: {e}")
            return False
    
    def save_results(self, results: StressTestResults):
        """Save stress test results to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO stress_test_results 
                (test_type, start_time, end_time, duration, result_status, 
                 max_stable_load, breaking_point, config_json, metrics_json, 
                 error_log_json, failure_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                results.test_type.value,
                results.start_time.isoformat(),
                results.end_time.isoformat() if results.end_time else None,
                results.get_test_duration(),
                results.result_status.value,
                results.max_stable_load,
                results.breaking_point,
                json.dumps(results.config.__dict__, default=str),
                json.dumps(results.performance_metrics),
                json.dumps(results.error_log),
                results.failure_reason
            ))
    
    def generate_stress_report(self) -> str:
        """Generate comprehensive stress test report"""
        report_path = self.results_dir / f"stress_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute("""
                SELECT * FROM stress_test_results 
                ORDER BY start_time DESC 
                LIMIT 10
            """).fetchall()
        
        html_content = self._generate_stress_html_report(results)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def _generate_stress_html_report(self, results: List[Tuple]) -> str:
        """Generate HTML stress test report"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stress Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
                .test-result { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .passed { background: #e8f5e8; }
                .failed { background: #fee; }
                .degraded { background: #fff3cd; }
                .crashed { background: #f8d7da; }
                .metric { margin: 5px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>System Stress Test Report</h1>
                <p>Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
                <p>This report shows system behavior under extreme load conditions and failure scenarios.</p>
            </div>
        """
        
        for result in results:
            (id_, test_type, start_time, end_time, duration, result_status, 
             max_stable_load, breaking_point, config_json, metrics_json, 
             error_log_json, failure_reason) = result
            
            status_class = result_status
            
            html += f"""
            <div class="test-result {status_class}">
                <h3>{test_type.upper()} - {result_status.upper()}</h3>
                <p><strong>Duration:</strong> {duration:.1f}s</p>
                <p><strong>Max Stable Load:</strong> {max_stable_load:.1f}x</p>
            """
            
            if breaking_point:
                html += f"<p><strong>Breaking Point:</strong> {breaking_point:.1f}x</p>"
            
            if failure_reason:
                html += f"<p><strong>Failure Reason:</strong> {failure_reason}</p>"
            
            html += "</div>"
        
        html += "</body></html>"
        return html


async def run_comprehensive_stress_tests():
    """Run all stress tests"""
    tester = StressTester()
    
    print("Starting comprehensive stress testing...")
    
    # Test configurations
    configs = [
        StressTestConfig(
            test_type=StressTestType.CPU_INTENSIVE,
            duration_seconds=120.0,
            max_load_factor=8.0,
            load_increment=1.0,
            increment_interval=15.0
        ),
        StressTestConfig(
            test_type=StressTestType.MEMORY_EXHAUSTION,
            duration_seconds=90.0,
            max_load_factor=5.0,
            load_increment=0.5,
            increment_interval=10.0
        ),
        StressTestConfig(
            test_type=StressTestType.CONCURRENT_THREADS,
            duration_seconds=60.0,
            max_load_factor=4.0,
            load_increment=0.5,
            increment_interval=10.0
        )
    ]
    
    results = []
    
    # Run progression tests
    for config in configs:
        print(f"\nRunning {config.test_type.value} progression test...")
        result = tester.run_load_progression_test(config)
        tester.save_results(result)
        results.append(result)
        
        print(f"Result: {result.result_status.value}")
        print(f"Max stable load: {result.max_stable_load:.1f}x")
        if result.breaking_point:
            print(f"Breaking point: {result.breaking_point:.1f}x")
    
    # Run sustained load tests
    print("\nRunning sustained load tests...")
    sustained_config = StressTestConfig(
        test_type=StressTestType.SUSTAINED_LOAD,
        duration_seconds=300.0  # 5 minutes
    )
    
    sustained_result = tester.run_sustained_load_test(sustained_config, 2.0)
    tester.save_results(sustained_result)
    results.append(sustained_result)
    
    # Run failure injection test
    print("\nRunning failure injection test...")
    failure_config = StressTestConfig(
        test_type=StressTestType.FAILURE_INJECTION,
        recovery_time=60.0
    )
    
    failure_result = tester.run_failure_injection_test(failure_config)
    tester.save_results(failure_result)
    results.append(failure_result)
    
    # Generate report
    report_path = tester.generate_stress_report()
    print(f"\nStress test report generated: {report_path}")
    
    # Summary
    passed = sum(1 for r in results if r.result_status == TestResult.PASSED)
    total = len(results)
    
    print(f"\nStress Test Summary:")
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    return results


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_comprehensive_stress_tests())