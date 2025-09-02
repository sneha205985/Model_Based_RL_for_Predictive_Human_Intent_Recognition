"""
Performance Benchmarking and Stress Testing Framework
==================================================

Comprehensive benchmarking system for validating real-time performance requirements:
- Decision cycle timing validation (<100ms guaranteed)
- Memory usage monitoring and bounds checking (<2GB)
- CPU utilization tracking (<80% average, <95% peak)
- Network latency measurement (<10ms)
- GPU utilization monitoring
- System stress testing under various load conditions
- Performance regression detection
- Automated reporting and validation
"""

import time
import psutil
import asyncio
import threading
import statistics
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import sqlite3
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import logging
import warnings

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    warnings.warn("GPUtil not available. GPU monitoring disabled.")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results"""
    timing_metrics: Dict[str, List[float]] = field(default_factory=dict)
    memory_metrics: Dict[str, List[float]] = field(default_factory=dict)
    cpu_metrics: Dict[str, List[float]] = field(default_factory=dict)
    gpu_metrics: Dict[str, List[float]] = field(default_factory=dict)
    network_metrics: Dict[str, List[float]] = field(default_factory=dict)
    custom_metrics: Dict[str, List[Any]] = field(default_factory=dict)
    test_duration: float = 0.0
    test_timestamp: datetime = field(default_factory=datetime.now)
    test_name: str = ""
    
    def add_timing(self, name: str, value: float):
        """Add timing measurement"""
        if name not in self.timing_metrics:
            self.timing_metrics[name] = []
        self.timing_metrics[name].append(value)
    
    def add_memory(self, name: str, value: float):
        """Add memory measurement (in MB)"""
        if name not in self.memory_metrics:
            self.memory_metrics[name] = []
        self.memory_metrics[name].append(value)
    
    def add_cpu(self, name: str, value: float):
        """Add CPU measurement (percentage)"""
        if name not in self.cpu_metrics:
            self.cpu_metrics[name] = []
        self.cpu_metrics[name].append(value)
    
    def add_gpu(self, name: str, value: float):
        """Add GPU measurement"""
        if name not in self.gpu_metrics:
            self.gpu_metrics[name] = []
        self.gpu_metrics[name].append(value)
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate statistical summary of all metrics"""
        summary = {
            'test_name': self.test_name,
            'test_duration': self.test_duration,
            'timestamp': self.test_timestamp.isoformat(),
            'timing_summary': {},
            'memory_summary': {},
            'cpu_summary': {},
            'gpu_summary': {}
        }
        
        for category, metrics_dict in [
            ('timing_summary', self.timing_metrics),
            ('memory_summary', self.memory_metrics),
            ('cpu_summary', self.cpu_metrics),
            ('gpu_summary', self.gpu_metrics)
        ]:
            for name, values in metrics_dict.items():
                if values:
                    summary[category][name] = {
                        'count': len(values),
                        'mean': statistics.mean(values),
                        'median': statistics.median(values),
                        'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values),
                        'p95': np.percentile(values, 95),
                        'p99': np.percentile(values, 99)
                    }
        
        return summary


class SystemProfiler:
    """Real-time system performance profiler"""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.profiling_active = False
        self.metrics = PerformanceMetrics()
        self._profiling_thread = None
        self._start_time = None
        
    def start_profiling(self, test_name: str = ""):
        """Start continuous system profiling"""
        if self.profiling_active:
            return
        
        self.profiling_active = True
        self.metrics = PerformanceMetrics()
        self.metrics.test_name = test_name
        self._start_time = time.time()
        
        self._profiling_thread = threading.Thread(target=self._profile_loop, daemon=True)
        self._profiling_thread.start()
    
    def stop_profiling(self) -> PerformanceMetrics:
        """Stop profiling and return collected metrics"""
        if not self.profiling_active:
            return self.metrics
        
        self.profiling_active = False
        if self._profiling_thread:
            self._profiling_thread.join(timeout=1.0)
        
        self.metrics.test_duration = time.time() - self._start_time
        return self.metrics
    
    def _profile_loop(self):
        """Background profiling loop"""
        process = psutil.Process()
        
        while self.profiling_active:
            try:
                # Memory metrics
                memory_info = process.memory_info()
                system_memory = psutil.virtual_memory()
                
                self.metrics.add_memory('process_rss', memory_info.rss / 1024 / 1024)  # MB
                self.metrics.add_memory('process_vms', memory_info.vms / 1024 / 1024)  # MB
                self.metrics.add_memory('system_used', system_memory.used / 1024 / 1024)  # MB
                self.metrics.add_memory('system_available', system_memory.available / 1024 / 1024)  # MB
                
                # CPU metrics
                cpu_percent = process.cpu_percent()
                system_cpu = psutil.cpu_percent(interval=None)
                
                self.metrics.add_cpu('process_cpu', cpu_percent)
                self.metrics.add_cpu('system_cpu', system_cpu)
                
                # GPU metrics (if available)
                if GPU_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        for i, gpu in enumerate(gpus):
                            self.metrics.add_gpu(f'gpu_{i}_utilization', gpu.load * 100)
                            self.metrics.add_gpu(f'gpu_{i}_memory', gpu.memoryUsed)
                            self.metrics.add_gpu(f'gpu_{i}_temperature', gpu.temperature)
                    except Exception:
                        pass
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logging.warning(f"Profiling error: {e}")
                time.sleep(self.sampling_interval)


@contextmanager
def timing_context(name: str, metrics: PerformanceMetrics):
    """Context manager for timing measurements"""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration = (end_time - start_time) * 1000  # Convert to milliseconds
        metrics.add_timing(name, duration)


class PerformanceBenchmark:
    """Main benchmarking framework"""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.db_path = self.results_dir / "benchmark_results.db"
        self._init_database()
        
        # Performance requirements (from original spec)
        self.requirements = {
            'decision_cycle_max': 100.0,  # ms
            'memory_max': 2048.0,  # MB
            'cpu_average_max': 80.0,  # %
            'cpu_peak_max': 95.0,  # %
            'network_latency_max': 10.0,  # ms
            'emergency_response_max': 10.0,  # ms
        }
    
    def _init_database(self):
        """Initialize SQLite database for results storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    duration REAL NOT NULL,
                    metrics_json TEXT NOT NULL,
                    passed BOOLEAN NOT NULL,
                    notes TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_requirements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    requirement_name TEXT UNIQUE NOT NULL,
                    threshold_value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    description TEXT
                )
            """)
            
            # Insert default requirements
            requirements_data = [
                ('decision_cycle_max', 100.0, 'ms', 'Maximum decision cycle time'),
                ('memory_max', 2048.0, 'MB', 'Maximum memory usage'),
                ('cpu_average_max', 80.0, '%', 'Maximum average CPU utilization'),
                ('cpu_peak_max', 95.0, '%', 'Maximum peak CPU utilization'),
                ('network_latency_max', 10.0, 'ms', 'Maximum network latency'),
                ('emergency_response_max', 10.0, 'ms', 'Maximum emergency response time')
            ]
            
            conn.executemany("""
                INSERT OR REPLACE INTO performance_requirements 
                (requirement_name, threshold_value, unit, description)
                VALUES (?, ?, ?, ?)
            """, requirements_data)
    
    def run_realtime_decision_benchmark(self, iterations: int = 1000, 
                                      load_factor: float = 1.0) -> PerformanceMetrics:
        """Benchmark real-time decision cycle performance"""
        profiler = SystemProfiler(sampling_interval=0.01)  # High frequency sampling
        profiler.start_profiling("realtime_decision_benchmark")
        
        # Simulate decision cycle workload
        for i in range(iterations):
            with timing_context("decision_cycle", profiler.metrics):
                # Simulate perception phase
                self._simulate_perception_workload(load_factor)
                
                # Simulate prediction phase  
                self._simulate_prediction_workload(load_factor)
                
                # Simulate planning phase
                self._simulate_planning_workload(load_factor)
                
                # Simulate control phase
                self._simulate_control_workload(load_factor)
            
            # Maintain real-time constraints
            if i % 10 == 0:  # Every 10 iterations
                await asyncio.sleep(0.001)  # Yield control
        
        return profiler.stop_profiling()
    
    def run_memory_stress_test(self, duration_minutes: int = 5, 
                              memory_pressure_mb: int = 1500) -> PerformanceMetrics:
        """Test memory management under stress conditions"""
        profiler = SystemProfiler(sampling_interval=0.1)
        profiler.start_profiling("memory_stress_test")
        
        # Allocate memory in chunks to simulate real workload
        memory_chunks = []
        chunk_size = 50 * 1024 * 1024  # 50MB chunks
        target_chunks = memory_pressure_mb // 50
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        while time.time() < end_time:
            # Allocate memory
            if len(memory_chunks) < target_chunks:
                with timing_context("memory_allocation", profiler.metrics):
                    chunk = np.random.random(chunk_size // 8)  # 8 bytes per float64
                    memory_chunks.append(chunk)
            
            # Simulate memory access patterns
            if memory_chunks:
                with timing_context("memory_access", profiler.metrics):
                    random_chunk = np.random.choice(memory_chunks)
                    _ = np.sum(random_chunk)  # Force memory access
            
            # Occasionally deallocate memory
            if len(memory_chunks) > 5 and np.random.random() < 0.1:
                with timing_context("memory_deallocation", profiler.metrics):
                    memory_chunks.pop(np.random.randint(0, len(memory_chunks)))
            
            time.sleep(0.1)
        
        # Cleanup
        memory_chunks.clear()
        
        return profiler.stop_profiling()
    
    def run_concurrent_load_test(self, num_workers: int = 4, 
                               tasks_per_worker: int = 100) -> PerformanceMetrics:
        """Test system performance under concurrent load"""
        profiler = SystemProfiler(sampling_interval=0.05)
        profiler.start_profiling("concurrent_load_test")
        
        def worker_task(worker_id: int, num_tasks: int) -> List[float]:
            """Individual worker task"""
            task_times = []
            for i in range(num_tasks):
                start_time = time.perf_counter()
                
                # Simulate mixed workload
                self._simulate_perception_workload(0.8)
                self._simulate_prediction_workload(0.6)
                
                # Add some GPU work if available
                if CUPY_AVAILABLE:
                    try:
                        data = cp.random.random((1000, 1000))
                        result = cp.matmul(data, data.T)
                        cp.cuda.Device().synchronize()
                    except Exception:
                        pass
                
                end_time = time.perf_counter()
                task_times.append((end_time - start_time) * 1000)
                
                if i % 10 == 0:
                    time.sleep(0.001)  # Brief yield
            
            return task_times
        
        # Execute concurrent workers
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(worker_task, i, tasks_per_worker) 
                for i in range(num_workers)
            ]
            
            # Collect results
            all_task_times = []
            for future in as_completed(futures):
                try:
                    task_times = future.result()
                    all_task_times.extend(task_times)
                except Exception as e:
                    logging.error(f"Worker task failed: {e}")
        
        # Add task timing to metrics
        for task_time in all_task_times:
            profiler.metrics.add_timing("concurrent_task", task_time)
        
        return profiler.stop_profiling()
    
    def run_emergency_response_test(self, num_tests: int = 100) -> PerformanceMetrics:
        """Test emergency response time requirements"""
        metrics = PerformanceMetrics()
        metrics.test_name = "emergency_response_test"
        
        for i in range(num_tests):
            # Simulate emergency condition detection
            with timing_context("emergency_detection", metrics):
                time.sleep(0.001)  # Simulate detection latency
            
            # Simulate emergency response
            with timing_context("emergency_response", metrics):
                # Simulate immediate safety actions
                self._simulate_emergency_response()
        
        return metrics
    
    def run_network_latency_test(self, num_tests: int = 100) -> PerformanceMetrics:
        """Test network communication latency"""
        metrics = PerformanceMetrics()
        metrics.test_name = "network_latency_test"
        
        # Simulate various network operations
        for i in range(num_tests):
            with timing_context("sensor_data_fetch", metrics):
                time.sleep(0.002)  # Simulate sensor data retrieval
            
            with timing_context("control_command_send", metrics):
                time.sleep(0.001)  # Simulate control command transmission
            
            with timing_context("status_update", metrics):
                time.sleep(0.001)  # Simulate status update
        
        return metrics
    
    def validate_requirements(self, metrics: PerformanceMetrics) -> Dict[str, bool]:
        """Validate performance metrics against requirements"""
        results = {}
        summary = metrics.get_summary()
        
        # Decision cycle timing
        if 'decision_cycle' in summary['timing_summary']:
            max_decision_time = summary['timing_summary']['decision_cycle']['max']
            results['decision_cycle_requirement'] = max_decision_time <= self.requirements['decision_cycle_max']
        
        # Memory usage
        if 'process_rss' in summary['memory_summary']:
            max_memory = summary['memory_summary']['process_rss']['max']
            results['memory_requirement'] = max_memory <= self.requirements['memory_max']
        
        # CPU utilization
        if 'process_cpu' in summary['cpu_summary']:
            avg_cpu = summary['cpu_summary']['process_cpu']['mean']
            max_cpu = summary['cpu_summary']['process_cpu']['max']
            results['cpu_average_requirement'] = avg_cpu <= self.requirements['cpu_average_max']
            results['cpu_peak_requirement'] = max_cpu <= self.requirements['cpu_peak_max']
        
        # Emergency response
        if 'emergency_response' in summary['timing_summary']:
            max_response = summary['timing_summary']['emergency_response']['max']
            results['emergency_response_requirement'] = max_response <= self.requirements['emergency_response_max']
        
        # Network latency
        network_times = []
        for metric in ['sensor_data_fetch', 'control_command_send', 'status_update']:
            if metric in summary['timing_summary']:
                network_times.append(summary['timing_summary'][metric]['max'])
        
        if network_times:
            max_network_latency = max(network_times)
            results['network_latency_requirement'] = max_network_latency <= self.requirements['network_latency_max']
        
        return results
    
    def save_results(self, metrics: PerformanceMetrics, validation_results: Dict[str, bool]):
        """Save benchmark results to database"""
        all_passed = all(validation_results.values()) if validation_results else False
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO benchmark_results 
                (test_name, timestamp, duration, metrics_json, passed, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                metrics.test_name,
                metrics.test_timestamp.isoformat(),
                metrics.test_duration,
                json.dumps(metrics.get_summary()),
                all_passed,
                json.dumps(validation_results)
            ))
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive benchmark report"""
        if output_path is None:
            output_path = self.results_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute("""
                SELECT test_name, timestamp, duration, metrics_json, passed, notes
                FROM benchmark_results
                ORDER BY timestamp DESC
                LIMIT 20
            """).fetchall()
        
        # Generate HTML report
        html_content = self._generate_html_report(results)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def _simulate_perception_workload(self, load_factor: float = 1.0):
        """Simulate perception processing workload"""
        # Simulate image processing
        size = int(1000 * load_factor)
        data = np.random.random((size, size))
        
        # Simulate feature extraction
        result = np.fft.fft2(data[:100, :100])
        
        # Simulate some computation delay
        time.sleep(0.001 * load_factor)
    
    def _simulate_prediction_workload(self, load_factor: float = 1.0):
        """Simulate prediction model workload"""
        # Simulate neural network forward pass
        size = int(500 * load_factor)
        weights = np.random.random((size, size))
        inputs = np.random.random(size)
        
        # Matrix multiplication
        result = np.dot(weights, inputs)
        
        time.sleep(0.002 * load_factor)
    
    def _simulate_planning_workload(self, load_factor: float = 1.0):
        """Simulate motion planning workload"""
        # Simulate trajectory optimization
        n_points = int(100 * load_factor)
        trajectory = np.random.random((n_points, 3))
        
        # Simulate constraint checking
        for i in range(n_points):
            constraint_check = np.linalg.norm(trajectory[i])
        
        time.sleep(0.003 * load_factor)
    
    def _simulate_control_workload(self, load_factor: float = 1.0):
        """Simulate control computation workload"""
        # Simulate PID control calculations
        error = np.random.random()
        integral = np.random.random()
        derivative = np.random.random()
        
        # Control output
        output = 0.1 * error + 0.05 * integral + 0.02 * derivative
        
        time.sleep(0.001 * load_factor)
    
    def _simulate_emergency_response(self):
        """Simulate emergency response actions"""
        # Simulate immediate safety actions
        time.sleep(0.005)  # 5ms response time
    
    def _generate_html_report(self, results: List[Tuple]) -> str:
        """Generate HTML benchmark report"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Benchmark Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .test-result { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .passed { background: #e8f5e8; border-color: #4CAF50; }
                .failed { background: #fee; border-color: #f44336; }
                .metrics { margin: 10px 0; }
                .metric-item { margin: 5px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Real-Time System Performance Benchmark Report</h1>
                <p>Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
                <p>Requirements Summary:</p>
                <ul>
                    <li>Decision cycle: &lt; 100ms</li>
                    <li>Memory usage: &lt; 2GB</li>
                    <li>CPU average: &lt; 80%</li>
                    <li>CPU peak: &lt; 95%</li>
                    <li>Emergency response: &lt; 10ms</li>
                    <li>Network latency: &lt; 10ms</li>
                </ul>
            </div>
        """
        
        for result in results:
            test_name, timestamp, duration, metrics_json, passed, notes = result
            metrics = json.loads(metrics_json)
            validation = json.loads(notes) if notes else {}
            
            status_class = "passed" if passed else "failed"
            status_text = "PASSED" if passed else "FAILED"
            
            html += f"""
            <div class="test-result {status_class}">
                <h3>{test_name} - {status_text}</h3>
                <p><strong>Timestamp:</strong> {timestamp}</p>
                <p><strong>Duration:</strong> {duration:.2f}s</p>
                
                <div class="metrics">
                    <h4>Performance Metrics:</h4>
            """
            
            for category in ['timing_summary', 'memory_summary', 'cpu_summary']:
                if category in metrics and metrics[category]:
                    html += f"<h5>{category.replace('_', ' ').title()}:</h5>"
                    for metric_name, metric_data in metrics[category].items():
                        html += f"""
                        <div class="metric-item">
                            <strong>{metric_name}:</strong> 
                            Mean: {metric_data.get('mean', 0):.2f}, 
                            Max: {metric_data.get('max', 0):.2f}, 
                            P95: {metric_data.get('p95', 0):.2f}
                        </div>
                        """
            
            if validation:
                html += "<h4>Requirement Validation:</h4><ul>"
                for req, passed in validation.items():
                    status = "✓" if passed else "✗"
                    html += f"<li>{status} {req}: {'PASSED' if passed else 'FAILED'}</li>"
                html += "</ul>"
            
            html += "</div></div>"
        
        html += "</body></html>"
        return html


async def run_comprehensive_benchmark():
    """Run all benchmark tests"""
    benchmark = PerformanceBenchmark()
    
    print("Starting comprehensive performance benchmark...")
    
    # Real-time decision cycle test
    print("1. Running real-time decision cycle benchmark...")
    metrics1 = benchmark.run_realtime_decision_benchmark(iterations=500, load_factor=1.0)
    validation1 = benchmark.validate_requirements(metrics1)
    benchmark.save_results(metrics1, validation1)
    
    # Memory stress test
    print("2. Running memory stress test...")
    metrics2 = benchmark.run_memory_stress_test(duration_minutes=2, memory_pressure_mb=1500)
    validation2 = benchmark.validate_requirements(metrics2)
    benchmark.save_results(metrics2, validation2)
    
    # Concurrent load test
    print("3. Running concurrent load test...")
    metrics3 = benchmark.run_concurrent_load_test(num_workers=4, tasks_per_worker=50)
    validation3 = benchmark.validate_requirements(metrics3)
    benchmark.save_results(metrics3, validation3)
    
    # Emergency response test
    print("4. Running emergency response test...")
    metrics4 = benchmark.run_emergency_response_test(num_tests=100)
    validation4 = benchmark.validate_requirements(metrics4)
    benchmark.save_results(metrics4, validation4)
    
    # Network latency test
    print("5. Running network latency test...")
    metrics5 = benchmark.run_network_latency_test(num_tests=100)
    validation5 = benchmark.validate_requirements(metrics5)
    benchmark.save_results(metrics5, validation5)
    
    # Generate report
    print("6. Generating benchmark report...")
    report_path = benchmark.generate_report()
    print(f"Benchmark report generated: {report_path}")
    
    # Summary
    all_tests = [validation1, validation2, validation3, validation4, validation5]
    total_requirements = sum(len(v) for v in all_tests)
    passed_requirements = sum(sum(v.values()) for v in all_tests)
    
    print(f"\nBenchmark Summary:")
    print(f"Total requirements tested: {total_requirements}")
    print(f"Requirements passed: {passed_requirements}")
    print(f"Pass rate: {passed_requirements/total_requirements*100:.1f}%")
    
    return report_path


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_comprehensive_benchmark())