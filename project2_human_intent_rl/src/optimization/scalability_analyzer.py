"""
Scalability Analysis Module
Model-Based RL Human Intent Recognition System

This module provides comprehensive scalability analysis including horizontal scaling,
vertical scaling, load testing, resource planning, and distributed deployment analysis.
"""

import time
import threading
import multiprocessing as mp
import queue
import statistics
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple, NamedTuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import defaultdict, deque
import psutil
import logging
import json
from abc import ABC, abstractmethod
import socket
import requests
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    import kubernetes
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False


@dataclass
class ScalabilityConfig:
    """Configuration for scalability analysis."""
    max_workers: int = mp.cpu_count() * 2
    load_test_duration: int = 60  # seconds
    ramp_up_time: int = 10  # seconds
    target_throughput: float = 100.0  # requests/second
    max_response_time: float = 1.0  # seconds
    cpu_threshold: float = 80.0  # percent
    memory_threshold: float = 80.0  # percent
    enable_distributed_testing: bool = False
    test_endpoints: List[str] = field(default_factory=list)
    enable_auto_scaling: bool = False
    min_replicas: int = 1
    max_replicas: int = 10


@dataclass
class PerformanceMetrics:
    """Performance metrics for scalability analysis."""
    throughput: float = 0.0  # requests/second
    response_time_p50: float = 0.0  # median
    response_time_p95: float = 0.0  # 95th percentile
    response_time_p99: float = 0.0  # 99th percentile
    error_rate: float = 0.0  # percentage
    cpu_usage: float = 0.0  # percentage
    memory_usage: float = 0.0  # MB
    network_io: float = 0.0  # MB/s
    disk_io: float = 0.0  # MB/s
    concurrent_users: int = 0
    success_count: int = 0
    error_count: int = 0


@dataclass
class ScalabilityTestResult:
    """Results from scalability testing."""
    config: ScalabilityConfig
    metrics: PerformanceMetrics
    scaling_recommendations: List[str] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    cost_analysis: Dict[str, float] = field(default_factory=dict)
    timeline_data: List[Dict[str, Any]] = field(default_factory=list)


class LoadGenerator:
    """Generate load for scalability testing."""
    
    def __init__(self, config: ScalabilityConfig):
        self.config = config
        self.running = False
        self.workers = []
        self.results_queue = queue.Queue()
        self.logger = logging.getLogger(__name__)
        
    def generate_load(self, target_function: Callable, 
                     args: Tuple = (), kwargs: Dict = None,
                     concurrent_users: int = 10) -> List[Dict[str, Any]]:
        """Generate load against a target function."""
        kwargs = kwargs or {}
        results = []
        
        self.running = True
        
        # Calculate requests per worker
        total_duration = self.config.load_test_duration
        requests_per_second = self.config.target_throughput
        total_requests = int(requests_per_second * total_duration)
        requests_per_worker = max(1, total_requests // concurrent_users)
        
        # Start workers
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            # Submit tasks
            futures = []
            for worker_id in range(concurrent_users):
                future = executor.submit(
                    self._worker_load_test, 
                    worker_id, target_function, args, kwargs, 
                    requests_per_worker, total_duration
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    worker_results = future.result()
                    results.extend(worker_results)
                except Exception as e:
                    self.logger.error(f"Worker error: {e}")
        
        self.running = False
        return results
    
    def _worker_load_test(self, worker_id: int, target_function: Callable,
                         args: Tuple, kwargs: Dict, num_requests: int,
                         total_duration: float) -> List[Dict[str, Any]]:
        """Individual worker load test."""
        results = []
        start_time = time.time()
        
        for request_id in range(num_requests):
            if time.time() - start_time > total_duration:
                break
            
            request_start = time.time()
            success = True
            error_msg = None
            
            try:
                result = target_function(*args, **kwargs)
            except Exception as e:
                success = False
                error_msg = str(e)
                result = None
            
            request_end = time.time()
            response_time = request_end - request_start
            
            results.append({
                'worker_id': worker_id,
                'request_id': request_id,
                'timestamp': request_start,
                'response_time': response_time,
                'success': success,
                'error': error_msg,
                'result_size': len(str(result)) if result else 0
            })
            
            # Small delay to control rate
            if num_requests > 0:
                expected_interval = total_duration / num_requests
                actual_time = time.time() - start_time
                expected_time = request_id * expected_interval
                
                if actual_time < expected_time:
                    time.sleep(expected_time - actual_time)
        
        return results
    
    def generate_web_load(self, urls: List[str], 
                         concurrent_users: int = 10) -> List[Dict[str, Any]]:
        """Generate HTTP load against web endpoints."""
        if not urls:
            return []
        
        results = []
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            # Submit HTTP load tests
            futures = []
            for worker_id in range(concurrent_users):
                future = executor.submit(
                    self._worker_http_load_test, 
                    worker_id, urls, self.config.load_test_duration
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    worker_results = future.result()
                    results.extend(worker_results)
                except Exception as e:
                    self.logger.error(f"HTTP worker error: {e}")
        
        return results
    
    def _worker_http_load_test(self, worker_id: int, urls: List[str], 
                              duration: float) -> List[Dict[str, Any]]:
        """HTTP load test worker."""
        results = []
        start_time = time.time()
        request_id = 0
        
        import requests
        session = requests.Session()
        
        while time.time() - start_time < duration:
            url = urls[request_id % len(urls)]
            request_start = time.time()
            success = True
            error_msg = None
            status_code = 0
            content_length = 0
            
            try:
                response = session.get(url, timeout=10)
                status_code = response.status_code
                content_length = len(response.content)
                success = status_code < 400
                
                if not success:
                    error_msg = f"HTTP {status_code}"
                    
            except Exception as e:
                success = False
                error_msg = str(e)
            
            request_end = time.time()
            response_time = request_end - request_start
            
            results.append({
                'worker_id': worker_id,
                'request_id': request_id,
                'timestamp': request_start,
                'response_time': response_time,
                'success': success,
                'error': error_msg,
                'url': url,
                'status_code': status_code,
                'content_length': content_length
            })
            
            request_id += 1
        
        session.close()
        return results


class ResourceMonitor:
    """Monitor system resources during scalability tests."""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = deque(maxlen=10000)
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self, interval: float = 1.0):
        """Start resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, args=(interval,)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current resource metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_mb = memory.used / (1024 * 1024)
        memory_percent = memory.percent
        
        # Network metrics
        network = psutil.net_io_counters()
        
        # Disk metrics
        disk = psutil.disk_io_counters()
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info().rss / (1024 * 1024)
        process_cpu = process.cpu_percent()
        
        return {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count,
            'memory_mb': memory_mb,
            'memory_percent': memory_percent,
            'memory_available_mb': memory.available / (1024 * 1024),
            'network_bytes_sent': network.bytes_sent,
            'network_bytes_recv': network.bytes_recv,
            'disk_read_bytes': disk.read_bytes if disk else 0,
            'disk_write_bytes': disk.write_bytes if disk else 0,
            'process_memory_mb': process_memory,
            'process_cpu_percent': process_cpu,
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        }
    
    def get_metrics_summary(self, window_seconds: int = 60) -> Dict[str, Any]:
        """Get metrics summary for recent window."""
        if not self.metrics_history:
            return {}
        
        current_time = time.time()
        recent_metrics = [
            m for m in self.metrics_history 
            if current_time - m['timestamp'] <= window_seconds
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate statistics
        cpu_values = [m['cpu_percent'] for m in recent_metrics]
        memory_values = [m['memory_percent'] for m in recent_metrics]
        process_cpu_values = [m['process_cpu_percent'] for m in recent_metrics]
        process_memory_values = [m['process_memory_mb'] for m in recent_metrics]
        
        return {
            'cpu_avg': statistics.mean(cpu_values),
            'cpu_max': max(cpu_values),
            'cpu_min': min(cpu_values),
            'memory_avg': statistics.mean(memory_values),
            'memory_max': max(memory_values),
            'process_cpu_avg': statistics.mean(process_cpu_values),
            'process_memory_avg': statistics.mean(process_memory_values),
            'process_memory_max': max(process_memory_values),
            'samples': len(recent_metrics),
            'duration': window_seconds
        }


class HorizontalScaler:
    """Horizontal scaling analysis and management."""
    
    def __init__(self, config: ScalabilityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze_scaling_requirements(self, 
                                   current_metrics: PerformanceMetrics,
                                   target_throughput: float) -> Dict[str, Any]:
        """Analyze horizontal scaling requirements."""
        current_throughput = current_metrics.throughput
        
        if current_throughput <= 0:
            return {'error': 'No current throughput data available'}
        
        # Calculate scaling factor
        scaling_factor = target_throughput / current_throughput
        
        # Consider response time degradation
        if current_metrics.response_time_p95 > self.config.max_response_time:
            # Need more aggressive scaling due to response time issues
            scaling_factor *= 1.5
        
        # Calculate required instances
        current_instances = 1  # Assume single instance currently
        required_instances = max(1, int(scaling_factor * current_instances))
        
        # Resource requirements per instance
        cpu_per_instance = current_metrics.cpu_usage
        memory_per_instance = current_metrics.memory_usage
        
        total_cpu_cores = required_instances * (cpu_per_instance / 100) * psutil.cpu_count()
        total_memory_gb = required_instances * memory_per_instance / 1024
        
        # Network and storage scaling
        network_scaling = scaling_factor
        storage_scaling = scaling_factor if current_metrics.disk_io > 0 else 1.0
        
        recommendations = []
        if required_instances > current_instances:
            recommendations.append(f"Scale out to {required_instances} instances")
        
        if current_metrics.error_rate > 5.0:
            recommendations.append("High error rate - investigate before scaling")
        
        if current_metrics.response_time_p99 > self.config.max_response_time * 2:
            recommendations.append("Consider vertical scaling first")
        
        return {
            'current_throughput': current_throughput,
            'target_throughput': target_throughput,
            'scaling_factor': scaling_factor,
            'current_instances': current_instances,
            'required_instances': required_instances,
            'total_cpu_cores': total_cpu_cores,
            'total_memory_gb': total_memory_gb,
            'network_scaling_factor': network_scaling,
            'storage_scaling_factor': storage_scaling,
            'recommendations': recommendations
        }
    
    def simulate_horizontal_scaling(self, target_function: Callable,
                                  instance_counts: List[int]) -> Dict[int, PerformanceMetrics]:
        """Simulate horizontal scaling with different instance counts."""
        results = {}
        
        for instance_count in instance_counts:
            self.logger.info(f"Testing with {instance_count} simulated instances")
            
            # Simulate by using process pools
            load_generator = LoadGenerator(self.config)
            monitor = ResourceMonitor()
            
            monitor.start_monitoring()
            
            # Generate load with workers simulating instances
            start_time = time.time()
            load_results = load_generator.generate_load(
                target_function, 
                concurrent_users=instance_count * 5  # Scale users with instances
            )
            end_time = time.time()
            
            monitor.stop_monitoring()
            
            # Calculate metrics
            metrics = self._calculate_metrics(load_results, end_time - start_time)
            
            # Add resource metrics
            resource_summary = monitor.get_metrics_summary()
            if resource_summary:
                metrics.cpu_usage = resource_summary['cpu_avg']
                metrics.memory_usage = resource_summary['process_memory_avg']
            
            results[instance_count] = metrics
        
        return results
    
    def _calculate_metrics(self, load_results: List[Dict[str, Any]], 
                          duration: float) -> PerformanceMetrics:
        """Calculate performance metrics from load test results."""
        if not load_results:
            return PerformanceMetrics()
        
        # Filter successful requests
        successful_requests = [r for r in load_results if r['success']]
        failed_requests = [r for r in load_results if not r['success']]
        
        # Response times
        response_times = [r['response_time'] for r in successful_requests]
        
        metrics = PerformanceMetrics()
        
        if response_times:
            metrics.response_time_p50 = np.percentile(response_times, 50)
            metrics.response_time_p95 = np.percentile(response_times, 95)
            metrics.response_time_p99 = np.percentile(response_times, 99)
        
        # Throughput
        metrics.throughput = len(successful_requests) / duration if duration > 0 else 0
        
        # Error rate
        total_requests = len(load_results)
        metrics.error_rate = (len(failed_requests) / total_requests * 100) if total_requests > 0 else 0
        
        # Counts
        metrics.success_count = len(successful_requests)
        metrics.error_count = len(failed_requests)
        metrics.concurrent_users = len(set(r['worker_id'] for r in load_results))
        
        return metrics


class VerticalScaler:
    """Vertical scaling analysis and recommendations."""
    
    def __init__(self, config: ScalabilityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_vertical_scaling(self, current_metrics: PerformanceMetrics,
                               target_improvement: float = 2.0) -> Dict[str, Any]:
        """Analyze vertical scaling options."""
        current_cpu = current_metrics.cpu_usage
        current_memory = current_metrics.memory_usage
        current_throughput = current_metrics.throughput
        
        recommendations = []
        
        # CPU scaling analysis
        if current_cpu > self.config.cpu_threshold:
            cpu_scaling_factor = target_improvement
            recommended_cpu_cores = psutil.cpu_count() * cpu_scaling_factor
            recommendations.append(
                f"CPU bottleneck detected. Recommend {recommended_cpu_cores:.0f} CPU cores "
                f"(current: {psutil.cpu_count()})"
            )
        
        # Memory scaling analysis
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        if current_memory / 1024 > total_memory_gb * 0.8:
            memory_scaling_factor = target_improvement
            recommended_memory_gb = total_memory_gb * memory_scaling_factor
            recommendations.append(
                f"Memory bottleneck detected. Recommend {recommended_memory_gb:.1f}GB RAM "
                f"(current: {total_memory_gb:.1f}GB)"
            )
        
        # Storage scaling analysis
        if current_metrics.disk_io > 100:  # MB/s threshold
            recommendations.append("Consider SSD storage for better I/O performance")
        
        # Estimate performance improvement
        bottleneck_factor = 1.0
        if current_cpu > 90:
            bottleneck_factor = min(bottleneck_factor, 90 / current_cpu)
        
        if current_memory / 1024 > total_memory_gb * 0.9:
            bottleneck_factor = min(bottleneck_factor, 0.5)  # Memory pressure impact
        
        estimated_throughput_improvement = 1.0 / bottleneck_factor if bottleneck_factor < 1.0 else 1.0
        
        # Cost-benefit analysis
        cpu_cost_factor = target_improvement if current_cpu > self.config.cpu_threshold else 1.0
        memory_cost_factor = target_improvement if current_memory / 1024 > total_memory_gb * 0.8 else 1.0
        
        return {
            'current_cpu_usage': current_cpu,
            'current_memory_gb': current_memory / 1024,
            'current_throughput': current_throughput,
            'recommended_cpu_scaling': cpu_cost_factor,
            'recommended_memory_scaling': memory_cost_factor,
            'estimated_throughput_improvement': estimated_throughput_improvement,
            'recommendations': recommendations,
            'cost_multiplier': max(cpu_cost_factor, memory_cost_factor)
        }
    
    def benchmark_resource_scaling(self, target_function: Callable,
                                 cpu_limits: List[int] = None,
                                 memory_limits: List[int] = None) -> Dict[str, Any]:
        """Benchmark performance under different resource constraints."""
        import os
        
        cpu_limits = cpu_limits or [1, 2, 4, 8]
        memory_limits = memory_limits or [512, 1024, 2048, 4096]  # MB
        
        results = {
            'cpu_scaling': {},
            'memory_scaling': {}
        }
        
        # Test CPU scaling (simulated by limiting worker processes)
        for cpu_limit in cpu_limits:
            if cpu_limit <= psutil.cpu_count():
                self.logger.info(f"Testing with {cpu_limit} CPU cores")
                
                # Simulate by limiting concurrent processes
                with ProcessPoolExecutor(max_workers=cpu_limit) as executor:
                    start_time = time.time()
                    
                    # Submit multiple tasks
                    futures = []
                    for _ in range(cpu_limit * 10):  # 10 tasks per core
                        future = executor.submit(target_function)
                        futures.append(future)
                    
                    # Wait for completion
                    completed = 0
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=10)
                            completed += 1
                        except Exception:
                            pass
                    
                    end_time = time.time()
                    
                    throughput = completed / (end_time - start_time)
                    results['cpu_scaling'][cpu_limit] = {
                        'throughput': throughput,
                        'duration': end_time - start_time,
                        'completed_tasks': completed
                    }
        
        return results


class AutoScaler:
    """Automatic scaling based on metrics."""
    
    def __init__(self, config: ScalabilityConfig):
        self.config = config
        self.current_replicas = config.min_replicas
        self.scaling_history = deque(maxlen=100)
        self.logger = logging.getLogger(__name__)
    
    def should_scale(self, metrics: PerformanceMetrics) -> Tuple[bool, str, int]:
        """Determine if scaling is needed."""
        if not self.config.enable_auto_scaling:
            return False, "Auto-scaling disabled", self.current_replicas
        
        # Scale up conditions
        if (metrics.cpu_usage > self.config.cpu_threshold or
            metrics.response_time_p95 > self.config.max_response_time or
            metrics.error_rate > 5.0):
            
            if self.current_replicas < self.config.max_replicas:
                new_replicas = min(
                    self.current_replicas * 2,
                    self.config.max_replicas
                )
                return True, "Scale up", new_replicas
        
        # Scale down conditions
        if (metrics.cpu_usage < self.config.cpu_threshold * 0.3 and
            metrics.response_time_p95 < self.config.max_response_time * 0.5 and
            metrics.error_rate < 1.0):
            
            if self.current_replicas > self.config.min_replicas:
                new_replicas = max(
                    self.current_replicas // 2,
                    self.config.min_replicas
                )
                return True, "Scale down", new_replicas
        
        return False, "No scaling needed", self.current_replicas
    
    def execute_scaling(self, new_replicas: int, reason: str) -> bool:
        """Execute scaling action."""
        old_replicas = self.current_replicas
        
        # Record scaling decision
        self.scaling_history.append({
            'timestamp': time.time(),
            'old_replicas': old_replicas,
            'new_replicas': new_replicas,
            'reason': reason
        })
        
        # Update current replicas
        self.current_replicas = new_replicas
        
        self.logger.info(
            f"Scaled from {old_replicas} to {new_replicas} replicas. Reason: {reason}"
        )
        
        return True


class ScalabilityAnalyzer:
    """Main scalability analysis system."""
    
    def __init__(self, config: ScalabilityConfig = None):
        self.config = config or ScalabilityConfig()
        self.load_generator = LoadGenerator(self.config)
        self.resource_monitor = ResourceMonitor()
        self.horizontal_scaler = HorizontalScaler(self.config)
        self.vertical_scaler = VerticalScaler(self.config)
        self.auto_scaler = AutoScaler(self.config)
        
        self.logger = logging.getLogger(__name__)
    
    def run_comprehensive_analysis(self, target_function: Callable,
                                 test_scenarios: List[Dict[str, Any]] = None) -> ScalabilityTestResult:
        """Run comprehensive scalability analysis."""
        self.logger.info("Starting comprehensive scalability analysis")
        
        test_scenarios = test_scenarios or [
            {'concurrent_users': 10, 'duration': 60},
            {'concurrent_users': 50, 'duration': 60},
            {'concurrent_users': 100, 'duration': 60}
        ]
        
        all_results = []
        timeline_data = []
        
        for scenario in test_scenarios:
            self.logger.info(f"Running scenario: {scenario}")
            
            # Configure for this scenario
            original_duration = self.config.load_test_duration
            self.config.load_test_duration = scenario.get('duration', 60)
            
            # Start monitoring
            self.resource_monitor.start_monitoring()
            
            # Run load test
            start_time = time.time()
            load_results = self.load_generator.generate_load(
                target_function,
                concurrent_users=scenario['concurrent_users']
            )
            end_time = time.time()
            
            # Stop monitoring
            self.resource_monitor.stop_monitoring()
            
            # Calculate metrics
            metrics = self.horizontal_scaler._calculate_metrics(
                load_results, end_time - start_time
            )
            
            # Add resource metrics
            resource_summary = self.resource_monitor.get_metrics_summary()
            if resource_summary:
                metrics.cpu_usage = resource_summary['cpu_avg']
                metrics.memory_usage = resource_summary['process_memory_avg']
            
            metrics.concurrent_users = scenario['concurrent_users']
            all_results.append(metrics)
            
            # Add to timeline
            timeline_data.append({
                'scenario': scenario,
                'metrics': metrics,
                'timestamp': start_time
            })
            
            # Restore original config
            self.config.load_test_duration = original_duration
        
        # Analyze results
        final_metrics = all_results[-1] if all_results else PerformanceMetrics()
        
        # Get scaling recommendations
        horizontal_analysis = self.horizontal_scaler.analyze_scaling_requirements(
            final_metrics, self.config.target_throughput
        )
        
        vertical_analysis = self.vertical_scaler.analyze_vertical_scaling(
            final_metrics
        )
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(final_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            final_metrics, horizontal_analysis, vertical_analysis
        )
        
        # Resource planning
        resource_requirements = self._calculate_resource_requirements(
            horizontal_analysis, vertical_analysis
        )
        
        # Cost analysis
        cost_analysis = self._estimate_costs(resource_requirements)
        
        return ScalabilityTestResult(
            config=self.config,
            metrics=final_metrics,
            scaling_recommendations=recommendations,
            bottlenecks=bottlenecks,
            resource_requirements=resource_requirements,
            cost_analysis=cost_analysis,
            timeline_data=timeline_data
        )
    
    def _identify_bottlenecks(self, metrics: PerformanceMetrics) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        if metrics.cpu_usage > self.config.cpu_threshold:
            bottlenecks.append(f"CPU bottleneck: {metrics.cpu_usage:.1f}% usage")
        
        if metrics.memory_usage > 1024 * 0.8:  # 80% of available memory
            bottlenecks.append(f"Memory bottleneck: {metrics.memory_usage:.0f}MB usage")
        
        if metrics.response_time_p95 > self.config.max_response_time:
            bottlenecks.append(
                f"Response time bottleneck: {metrics.response_time_p95:.3f}s P95"
            )
        
        if metrics.error_rate > 5.0:
            bottlenecks.append(f"High error rate: {metrics.error_rate:.1f}%")
        
        if metrics.throughput < self.config.target_throughput * 0.8:
            bottlenecks.append(
                f"Low throughput: {metrics.throughput:.1f} req/s "
                f"(target: {self.config.target_throughput})"
            )
        
        return bottlenecks
    
    def _generate_recommendations(self, metrics: PerformanceMetrics,
                                horizontal_analysis: Dict[str, Any],
                                vertical_analysis: Dict[str, Any]) -> List[str]:
        """Generate scaling recommendations."""
        recommendations = []
        
        # Add horizontal scaling recommendations
        if 'recommendations' in horizontal_analysis:
            recommendations.extend(horizontal_analysis['recommendations'])
        
        # Add vertical scaling recommendations
        if 'recommendations' in vertical_analysis:
            recommendations.extend(vertical_analysis['recommendations'])
        
        # Performance-based recommendations
        if metrics.response_time_p99 > self.config.max_response_time * 3:
            recommendations.append("Consider caching to reduce response times")
        
        if metrics.error_rate > 10:
            recommendations.append("Investigate error causes before scaling")
        
        # Cost optimization
        cpu_scaling = vertical_analysis.get('recommended_cpu_scaling', 1.0)
        instance_scaling = horizontal_analysis.get('scaling_factor', 1.0)
        
        if cpu_scaling < instance_scaling:
            recommendations.append("Vertical scaling more cost-effective than horizontal")
        else:
            recommendations.append("Horizontal scaling recommended over vertical")
        
        return recommendations
    
    def _calculate_resource_requirements(self, horizontal_analysis: Dict[str, Any],
                                       vertical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate total resource requirements."""
        return {
            'cpu_cores': horizontal_analysis.get('total_cpu_cores', 1),
            'memory_gb': horizontal_analysis.get('total_memory_gb', 1),
            'instances': horizontal_analysis.get('required_instances', 1),
            'network_bandwidth_multiplier': horizontal_analysis.get('network_scaling_factor', 1),
            'storage_multiplier': horizontal_analysis.get('storage_scaling_factor', 1),
            'estimated_improvement': vertical_analysis.get('estimated_throughput_improvement', 1)
        }
    
    def _estimate_costs(self, resource_requirements: Dict[str, Any]) -> Dict[str, float]:
        """Estimate costs for scaling."""
        # Simplified cost model (would need real cloud pricing)
        cpu_cost_per_core_hour = 0.05
        memory_cost_per_gb_hour = 0.01
        instance_cost_per_hour = 0.10
        network_cost_per_gb = 0.02
        
        cpu_cores = resource_requirements.get('cpu_cores', 1)
        memory_gb = resource_requirements.get('memory_gb', 1)
        instances = resource_requirements.get('instances', 1)
        
        monthly_hours = 24 * 30  # 720 hours per month
        
        return {
            'cpu_cost_monthly': cpu_cores * cpu_cost_per_core_hour * monthly_hours,
            'memory_cost_monthly': memory_gb * memory_cost_per_gb_hour * monthly_hours,
            'instance_cost_monthly': instances * instance_cost_per_hour * monthly_hours,
            'total_monthly': (
                cpu_cores * cpu_cost_per_core_hour * monthly_hours +
                memory_gb * memory_cost_per_gb_hour * monthly_hours +
                instances * instance_cost_per_hour * monthly_hours
            )
        }
    
    def generate_scaling_report(self, result: ScalabilityTestResult) -> str:
        """Generate comprehensive scaling report."""
        report = []
        report.append("=" * 80)
        report.append("SCALABILITY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Performance Summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 40)
        report.append(f"Throughput: {result.metrics.throughput:.2f} req/s")
        report.append(f"Response Time P50: {result.metrics.response_time_p50:.3f}s")
        report.append(f"Response Time P95: {result.metrics.response_time_p95:.3f}s")
        report.append(f"Response Time P99: {result.metrics.response_time_p99:.3f}s")
        report.append(f"Error Rate: {result.metrics.error_rate:.2f}%")
        report.append(f"CPU Usage: {result.metrics.cpu_usage:.1f}%")
        report.append(f"Memory Usage: {result.metrics.memory_usage:.0f}MB")
        report.append(f"Concurrent Users: {result.metrics.concurrent_users}")
        report.append("")
        
        # Bottlenecks
        if result.bottlenecks:
            report.append("IDENTIFIED BOTTLENECKS")
            report.append("-" * 40)
            for bottleneck in result.bottlenecks:
                report.append(f"• {bottleneck}")
            report.append("")
        
        # Scaling Recommendations
        if result.scaling_recommendations:
            report.append("SCALING RECOMMENDATIONS")
            report.append("-" * 40)
            for recommendation in result.scaling_recommendations:
                report.append(f"• {recommendation}")
            report.append("")
        
        # Resource Requirements
        if result.resource_requirements:
            report.append("RESOURCE REQUIREMENTS")
            report.append("-" * 40)
            req = result.resource_requirements
            report.append(f"CPU Cores: {req.get('cpu_cores', 0):.1f}")
            report.append(f"Memory: {req.get('memory_gb', 0):.1f} GB")
            report.append(f"Instances: {req.get('instances', 0)}")
            report.append("")
        
        # Cost Analysis
        if result.cost_analysis:
            report.append("COST ANALYSIS (Monthly)")
            report.append("-" * 40)
            costs = result.cost_analysis
            report.append(f"CPU Cost: ${costs.get('cpu_cost_monthly', 0):.2f}")
            report.append(f"Memory Cost: ${costs.get('memory_cost_monthly', 0):.2f}")
            report.append(f"Instance Cost: ${costs.get('instance_cost_monthly', 0):.2f}")
            report.append(f"Total Cost: ${costs.get('total_monthly', 0):.2f}")
            report.append("")
        
        return "\n".join(report)


# Example usage
def create_test_workload():
    """Create a test workload for scalability analysis."""
    def cpu_intensive_task():
        """CPU-intensive task for testing."""
        # Simulate computation
        result = sum(i**2 for i in range(10000))
        time.sleep(0.01)  # Small delay
        return result
    
    def memory_intensive_task():
        """Memory-intensive task for testing."""
        # Create large array
        data = np.random.randn(1000, 1000)
        result = np.sum(data)
        return result
    
    def mixed_workload_task():
        """Mixed CPU and memory task."""
        # CPU work
        cpu_result = sum(i**2 for i in range(5000))
        
        # Memory work
        data = np.random.randn(500, 500)
        memory_result = np.sum(data)
        
        return cpu_result + memory_result
    
    return cpu_intensive_task, memory_intensive_task, mixed_workload_task


if __name__ == "__main__":
    # Run scalability analysis example
    config = ScalabilityConfig(
        target_throughput=50.0,
        load_test_duration=30,  # Shorter for testing
        max_response_time=0.5
    )
    
    analyzer = ScalabilityAnalyzer(config)
    
    # Create test workloads
    cpu_task, memory_task, mixed_task = create_test_workload()
    
    print("Running scalability analysis...")
    
    # Test different workloads
    for task_name, task in [("CPU Task", cpu_task), ("Mixed Task", mixed_task)]:
        print(f"\nAnalyzing {task_name}...")
        
        result = analyzer.run_comprehensive_analysis(
            task,
            test_scenarios=[
                {'concurrent_users': 5, 'duration': 20},
                {'concurrent_users': 15, 'duration': 20}
            ]
        )
        
        # Generate report
        report = analyzer.generate_scaling_report(result)
        print(report)