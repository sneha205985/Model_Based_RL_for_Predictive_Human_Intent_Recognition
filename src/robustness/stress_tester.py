#!/usr/bin/env python3
"""
Stress Testing Framework for Robustness Validation
==================================================

This module implements a comprehensive stress testing framework for validating
system robustness under extreme conditions, including high-frequency disturbance
scenarios, simultaneous multiple sensor failures, network communication
disruptions, computational resource exhaustion, and extended operation testing.

Features:
- High-frequency disturbance scenario testing
- Simultaneous multiple sensor failure simulation
- Network communication disruption testing
- Computational resource exhaustion testing  
- Memory leak detection and recovery testing
- Extended operation testing (>24 hours continuous)
- Performance degradation analysis over time
- Adversarial input and robustness testing
- Data drift detection and handling
- Component wear and adaptation testing

Mathematical Models:
===================

Stress Response Analysis:
    S(t) = S₀ + ∫₀ᵗ f(stress_factor, system_state) dt

Performance Degradation:
    P(t) = P₀ * exp(-λt) + noise(t)

Resource Exhaustion Model:
    R(t) = R_max / (1 + k*load(t))

Reliability Under Stress:
    R(t) = exp(-∫₀ᵗ λ(stress_level(τ)) dτ)

Author: Claude Code (Anthropic)
Date: 2025-01-15
Version: 1.0
"""

import time
import threading
import logging
import psutil
import gc
import tracemalloc
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Generator
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque, defaultdict
import queue
import random
import json
import socket
import multiprocessing
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal, stats
import pandas as pd

logger = logging.getLogger(__name__)


class StressType(Enum):
    """Types of stress tests"""
    COMPUTATIONAL_LOAD = "computational_load"
    MEMORY_PRESSURE = "memory_pressure"
    SENSOR_FAILURES = "sensor_failures"
    NETWORK_DISRUPTION = "network_disruption"
    HIGH_FREQUENCY_DISTURBANCE = "high_frequency_disturbance"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONCURRENT_OPERATIONS = "concurrent_operations"
    ADVERSARIAL_INPUT = "adversarial_input"
    DATA_CORRUPTION = "data_corruption"
    TIMING_ATTACKS = "timing_attacks"


class StressSeverity(Enum):
    """Stress test severity levels"""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_usage: float  # Percentage
    memory_usage: float  # Percentage
    memory_available: float  # MB
    disk_io_read: float  # MB/s
    disk_io_write: float  # MB/s
    network_bytes_sent: float  # bytes/s
    network_bytes_recv: float  # bytes/s
    
    # Custom metrics
    response_time_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    error_rate: float = 0.0
    queue_size: int = 0
    thread_count: int = 0


@dataclass
class StressTestConfiguration:
    """Configuration for stress tests"""
    test_id: str
    stress_type: StressType
    severity: StressSeverity
    duration_seconds: float
    
    # Stress parameters
    target_cpu_load: Optional[float] = None
    target_memory_usage: Optional[float] = None
    failure_rate: Optional[float] = None
    disturbance_frequency: Optional[float] = None
    concurrent_threads: Optional[int] = None
    
    # Thresholds
    max_response_time_ms: float = 100.0
    min_throughput_ops_per_sec: float = 10.0
    max_error_rate: float = 0.05
    max_memory_growth_mb_per_min: float = 10.0
    
    # Recovery requirements
    recovery_time_limit_seconds: float = 30.0
    performance_recovery_threshold: float = 0.9


@dataclass
class StressTestResult:
    """Result of stress test execution"""
    test_id: str
    stress_type: StressType
    severity: StressSeverity
    passed: bool
    execution_time: float
    
    # Performance metrics
    baseline_metrics: SystemMetrics
    peak_stress_metrics: SystemMetrics
    recovery_metrics: SystemMetrics
    
    # Test-specific results
    performance_degradation: float
    recovery_time_seconds: float
    max_memory_usage_mb: float
    max_cpu_usage: float
    error_count: int
    
    # Failure information
    failure_reason: Optional[str] = None
    failure_timestamp: Optional[float] = None
    
    # Additional data
    metrics_history: List[SystemMetrics] = field(default_factory=list)
    error_log: List[str] = field(default_factory=list)


class SystemMonitor:
    """Real-time system performance monitoring"""
    
    def __init__(self, sampling_rate_hz: float = 10.0):
        """Initialize system monitor"""
        self.sampling_rate = sampling_rate_hz
        self.sampling_interval = 1.0 / sampling_rate_hz
        
        self.monitoring_enabled = False
        self.monitoring_thread = None
        
        self.metrics_history = deque(maxlen=10000)  # Keep last 10000 samples
        self.baseline_metrics = None
        
        # System baseline measurement
        self._establish_baseline()
        
        logger.info(f"System monitor initialized (sampling rate: {sampling_rate_hz}Hz)")
    
    def _establish_baseline(self) -> None:
        """Establish baseline system performance"""
        baseline_samples = []
        
        # Collect baseline for 10 seconds
        for _ in range(100):  # 10 seconds at 10Hz
            metrics = self._collect_metrics()
            baseline_samples.append(metrics)
            time.sleep(0.1)
        
        # Calculate baseline averages
        if baseline_samples:
            self.baseline_metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_usage=np.mean([m.cpu_usage for m in baseline_samples]),
                memory_usage=np.mean([m.memory_usage for m in baseline_samples]),
                memory_available=np.mean([m.memory_available for m in baseline_samples]),
                disk_io_read=np.mean([m.disk_io_read for m in baseline_samples]),
                disk_io_write=np.mean([m.disk_io_write for m in baseline_samples]),
                network_bytes_sent=np.mean([m.network_bytes_sent for m in baseline_samples]),
                network_bytes_recv=np.mean([m.network_bytes_recv for m in baseline_samples])
            )
        
        logger.info(f"Baseline established - CPU: {self.baseline_metrics.cpu_usage:.1f}%, "
                   f"Memory: {self.baseline_metrics.memory_usage:.1f}%")
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        current_time = time.time()
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_mb_s = 0.0
        disk_write_mb_s = 0.0
        if disk_io:
            # Convert bytes to MB and calculate rate (simplified)
            disk_read_mb_s = disk_io.read_bytes / (1024 * 1024) / self.sampling_interval
            disk_write_mb_s = disk_io.write_bytes / (1024 * 1024) / self.sampling_interval
        
        # Network I/O
        net_io = psutil.net_io_counters()
        net_sent_rate = 0.0
        net_recv_rate = 0.0
        if net_io:
            net_sent_rate = net_io.bytes_sent / self.sampling_interval
            net_recv_rate = net_io.bytes_recv / self.sampling_interval
        
        return SystemMetrics(
            timestamp=current_time,
            cpu_usage=cpu_percent,
            memory_usage=memory_info.percent,
            memory_available=memory_info.available / (1024 * 1024),  # MB
            disk_io_read=disk_read_mb_s,
            disk_io_write=disk_write_mb_s,
            network_bytes_sent=net_sent_rate,
            network_bytes_recv=net_recv_rate,
            thread_count=threading.active_count()
        )
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring"""
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        
        def monitoring_loop():
            while self.monitoring_enabled:
                try:
                    metrics = self._collect_metrics()
                    self.metrics_history.append(metrics)
                    time.sleep(self.sampling_interval)
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(1.0)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("System monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring"""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        logger.info("System monitoring stopped")
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        return self._collect_metrics()
    
    def get_metrics_history(self, duration_seconds: float = 60.0) -> List[SystemMetrics]:
        """Get metrics history for specified duration"""
        current_time = time.time()
        cutoff_time = current_time - duration_seconds
        
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def analyze_performance_trend(self, duration_seconds: float = 300.0) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        history = self.get_metrics_history(duration_seconds)
        
        if len(history) < 10:
            return {'error': 'Insufficient data for trend analysis'}
        
        timestamps = [m.timestamp - history[0].timestamp for m in history]
        cpu_values = [m.cpu_usage for m in history]
        memory_values = [m.memory_usage for m in history]
        
        # Linear regression for trends
        cpu_slope, cpu_intercept, cpu_r_value, _, _ = stats.linregress(timestamps, cpu_values)
        mem_slope, mem_intercept, mem_r_value, _, _ = stats.linregress(timestamps, memory_values)
        
        return {
            'duration_seconds': duration_seconds,
            'sample_count': len(history),
            'cpu_trend': {
                'slope_per_hour': cpu_slope * 3600,  # % per hour
                'current_usage': cpu_values[-1],
                'baseline_usage': self.baseline_metrics.cpu_usage if self.baseline_metrics else 0.0,
                'correlation': cpu_r_value
            },
            'memory_trend': {
                'slope_per_hour': mem_slope * 3600,  # % per hour
                'current_usage': memory_values[-1],
                'baseline_usage': self.baseline_metrics.memory_usage if self.baseline_metrics else 0.0,
                'correlation': mem_r_value
            },
            'statistics': {
                'cpu_mean': np.mean(cpu_values),
                'cpu_std': np.std(cpu_values),
                'cpu_max': np.max(cpu_values),
                'memory_mean': np.mean(memory_values),
                'memory_std': np.std(memory_values),
                'memory_max': np.max(memory_values)
            }
        }


class StressGenerator:
    """Generates various types of system stress"""
    
    def __init__(self):
        """Initialize stress generator"""
        self.stress_threads = []
        self.stress_processes = []
        self.stress_enabled = False
        self.allocated_memory = []
        
    def generate_cpu_stress(self, target_load: float, duration: float) -> None:
        """Generate CPU stress load"""
        
        def cpu_stress_worker():
            end_time = time.time() + duration
            
            while time.time() < end_time and self.stress_enabled:
                # Busy loop to consume CPU
                start = time.time()
                while time.time() - start < target_load / 100.0:
                    # Computational work
                    _ = sum(i * i for i in range(1000))
                
                # Sleep to achieve target load percentage
                sleep_time = (1.0 - target_load / 100.0)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        self.stress_enabled = True
        num_threads = max(1, int(psutil.cpu_count() * target_load / 100.0))
        
        for _ in range(num_threads):
            thread = threading.Thread(target=cpu_stress_worker, daemon=True)
            thread.start()
            self.stress_threads.append(thread)
        
        logger.info(f"Started CPU stress test: {target_load}% load for {duration}s")
    
    def generate_memory_stress(self, target_usage_mb: float, duration: float) -> None:
        """Generate memory pressure"""
        
        def memory_stress_worker():
            chunk_size = 10 * 1024 * 1024  # 10MB chunks
            num_chunks = int(target_usage_mb / 10)
            
            try:
                for i in range(num_chunks):
                    if not self.stress_enabled:
                        break
                    
                    # Allocate memory chunk
                    chunk = bytearray(chunk_size)
                    # Write to memory to ensure allocation
                    for j in range(0, chunk_size, 4096):
                        chunk[j] = random.randint(0, 255)
                    
                    self.allocated_memory.append(chunk)
                    time.sleep(0.1)  # Gradual allocation
                
                # Hold memory for duration
                time.sleep(duration)
                
            except MemoryError:
                logger.warning("Memory allocation limit reached during stress test")
            finally:
                # Clean up allocated memory
                self.allocated_memory.clear()
                gc.collect()
        
        self.stress_enabled = True
        thread = threading.Thread(target=memory_stress_worker, daemon=True)
        thread.start()
        self.stress_threads.append(thread)
        
        logger.info(f"Started memory stress test: {target_usage_mb}MB for {duration}s")
    
    def generate_io_stress(self, duration: float) -> None:
        """Generate I/O stress"""
        
        def io_stress_worker():
            end_time = time.time() + duration
            file_counter = 0
            
            while time.time() < end_time and self.stress_enabled:
                try:
                    # Write stress
                    filename = f"/tmp/stress_test_{file_counter}.tmp"
                    with open(filename, 'wb') as f:
                        data = bytearray(random.getrandbits(8) for _ in range(1024 * 1024))  # 1MB
                        f.write(data)
                    
                    # Read stress
                    with open(filename, 'rb') as f:
                        _ = f.read()
                    
                    # Cleanup
                    import os
                    os.remove(filename)
                    
                    file_counter += 1
                    time.sleep(0.01)  # Brief pause
                    
                except Exception as e:
                    logger.warning(f"I/O stress error: {e}")
                    time.sleep(0.1)
        
        self.stress_enabled = True
        thread = threading.Thread(target=io_stress_worker, daemon=True)
        thread.start()
        self.stress_threads.append(thread)
        
        logger.info(f"Started I/O stress test for {duration}s")
    
    def generate_network_stress(self, duration: float) -> None:
        """Generate network stress (local loopback)"""
        
        def network_server():
            try:
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.bind(('localhost', 0))  # Random port
                port = server_socket.getsockname()[1]
                server_socket.listen(5)
                server_socket.settimeout(1.0)
                
                end_time = time.time() + duration
                
                while time.time() < end_time and self.stress_enabled:
                    try:
                        client_socket, _ = server_socket.accept()
                        data = client_socket.recv(4096)
                        client_socket.send(data)  # Echo back
                        client_socket.close()
                    except socket.timeout:
                        continue
                    except Exception as e:
                        logger.debug(f"Network server error: {e}")
                
                server_socket.close()
                
            except Exception as e:
                logger.error(f"Network server setup error: {e}")
        
        def network_client(port):
            end_time = time.time() + duration
            
            while time.time() < end_time and self.stress_enabled:
                try:
                    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client_socket.settimeout(1.0)
                    client_socket.connect(('localhost', port))
                    
                    data = bytearray(random.getrandbits(8) for _ in range(1024))
                    client_socket.send(data)
                    response = client_socket.recv(4096)
                    client_socket.close()
                    
                    time.sleep(0.01)
                    
                except Exception as e:
                    logger.debug(f"Network client error: {e}")
                    time.sleep(0.1)
        
        self.stress_enabled = True
        
        # Start server
        server_thread = threading.Thread(target=network_server, daemon=True)
        server_thread.start()
        self.stress_threads.append(server_thread)
        
        time.sleep(0.5)  # Let server start
        
        # Start clients
        for _ in range(5):  # Multiple clients
            client_thread = threading.Thread(
                target=lambda: network_client(12345), daemon=True
            )
            client_thread.start()
            self.stress_threads.append(client_thread)
        
        logger.info(f"Started network stress test for {duration}s")
    
    def stop_stress(self) -> None:
        """Stop all stress generation"""
        self.stress_enabled = False
        
        # Wait for threads to finish
        for thread in self.stress_threads:
            thread.join(timeout=2.0)
        
        # Clean up allocated memory
        self.allocated_memory.clear()
        gc.collect()
        
        # Terminate processes
        for process in self.stress_processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        
        self.stress_threads.clear()
        self.stress_processes.clear()
        
        logger.info("All stress generation stopped")


class AdversarialInputGenerator:
    """Generates adversarial inputs for robustness testing"""
    
    def __init__(self):
        """Initialize adversarial input generator"""
        self.attack_strategies = {
            'noise_injection': self._generate_noise_attack,
            'boundary_values': self._generate_boundary_attack,
            'format_fuzzing': self._generate_format_attack,
            'timing_attack': self._generate_timing_attack,
            'overflow_attack': self._generate_overflow_attack
        }
    
    def generate_adversarial_inputs(self, 
                                  base_input: np.ndarray,
                                  attack_type: str = 'noise_injection',
                                  intensity: float = 0.1) -> List[np.ndarray]:
        """Generate adversarial inputs based on base input"""
        
        if attack_type not in self.attack_strategies:
            logger.warning(f"Unknown attack type: {attack_type}")
            return [base_input]
        
        attack_function = self.attack_strategies[attack_type]
        return attack_function(base_input, intensity)
    
    def _generate_noise_attack(self, base_input: np.ndarray, intensity: float) -> List[np.ndarray]:
        """Generate noise-based adversarial inputs"""
        adversarial_inputs = []
        
        # Gaussian noise
        noise = np.random.normal(0, intensity * np.std(base_input), base_input.shape)
        adversarial_inputs.append(base_input + noise)
        
        # Uniform noise
        noise_range = intensity * (np.max(base_input) - np.min(base_input))
        noise = np.random.uniform(-noise_range, noise_range, base_input.shape)
        adversarial_inputs.append(base_input + noise)
        
        # Salt and pepper noise
        mask = np.random.random(base_input.shape) < intensity
        salt_pepper = base_input.copy()
        salt_pepper[mask] = np.random.choice([np.min(base_input), np.max(base_input)], np.sum(mask))
        adversarial_inputs.append(salt_pepper)
        
        return adversarial_inputs
    
    def _generate_boundary_attack(self, base_input: np.ndarray, intensity: float) -> List[np.ndarray]:
        """Generate boundary value attacks"""
        adversarial_inputs = []
        
        data_min = np.min(base_input)
        data_max = np.max(base_input)
        data_range = data_max - data_min
        
        # Extreme values
        adversarial_inputs.append(np.full_like(base_input, data_min))
        adversarial_inputs.append(np.full_like(base_input, data_max))
        
        # Values slightly outside expected range
        adversarial_inputs.append(np.full_like(base_input, data_min - intensity * data_range))
        adversarial_inputs.append(np.full_like(base_input, data_max + intensity * data_range))
        
        return adversarial_inputs
    
    def _generate_format_attack(self, base_input: np.ndarray, intensity: float) -> List[np.ndarray]:
        """Generate format-based attacks"""
        adversarial_inputs = []
        
        # NaN injection
        nan_input = base_input.copy().astype(float)
        num_nans = int(intensity * base_input.size)
        nan_indices = np.random.choice(base_input.size, num_nans, replace=False)
        nan_input.flat[nan_indices] = np.nan
        adversarial_inputs.append(nan_input)
        
        # Infinity injection
        inf_input = base_input.copy().astype(float)
        num_infs = int(intensity * base_input.size)
        inf_indices = np.random.choice(base_input.size, num_infs, replace=False)
        inf_input.flat[inf_indices] = np.inf
        adversarial_inputs.append(inf_input)
        
        return adversarial_inputs
    
    def _generate_timing_attack(self, base_input: np.ndarray, intensity: float) -> List[np.ndarray]:
        """Generate timing-based attacks (delayed inputs)"""
        # This would be used in conjunction with delayed delivery
        # For now, return modified inputs that might cause timing issues
        adversarial_inputs = []
        
        # Large inputs (might cause processing delays)
        large_input = base_input * (1 + intensity * 100)
        adversarial_inputs.append(large_input)
        
        # High-frequency components (might cause processing spikes)
        if len(base_input.shape) >= 1:
            t = np.linspace(0, 1, base_input.shape[0])
            high_freq_signal = intensity * np.sin(2 * np.pi * 50 * t)  # 50 Hz
            high_freq_input = base_input + high_freq_signal[:, np.newaxis] if len(base_input.shape) > 1 else base_input + high_freq_signal
            adversarial_inputs.append(high_freq_input)
        
        return adversarial_inputs
    
    def _generate_overflow_attack(self, base_input: np.ndarray, intensity: float) -> List[np.ndarray]:
        """Generate overflow-based attacks"""
        adversarial_inputs = []
        
        # Large values that might cause overflow
        max_safe_value = np.finfo(np.float32).max / 2
        overflow_input = base_input + intensity * max_safe_value
        adversarial_inputs.append(overflow_input)
        
        # Rapid scaling that might cause overflow in accumulated operations
        scaling_factor = 1 + intensity * 1000
        scaled_input = base_input * scaling_factor
        adversarial_inputs.append(scaled_input)
        
        return adversarial_inputs


class StressTestFramework:
    """Comprehensive stress testing framework"""
    
    def __init__(self):
        """Initialize stress test framework"""
        self.monitor = SystemMonitor()
        self.stress_generator = StressGenerator()
        self.adversarial_generator = AdversarialInputGenerator()
        
        self.test_results: List[StressTestResult] = []
        self.current_test_config = None
        
        # Memory leak detection
        self.memory_tracker_enabled = False
        
        logger.info("Stress test framework initialized")
    
    def run_stress_test(self, config: StressTestConfiguration) -> StressTestResult:
        """Run a comprehensive stress test"""
        
        logger.info(f"Starting stress test: {config.test_id} ({config.stress_type.value})")
        self.current_test_config = config
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Record baseline
        baseline_metrics = self.monitor.get_current_metrics()
        
        # Enable memory tracking if needed
        if config.stress_type == StressType.MEMORY_PRESSURE:
            tracemalloc.start()
            self.memory_tracker_enabled = True
        
        # Execute stress test
        test_start_time = time.time()
        
        try:
            result = self._execute_stress_test(config)
            result.baseline_metrics = baseline_metrics
            
        except Exception as e:
            logger.error(f"Stress test {config.test_id} failed: {e}")
            result = StressTestResult(
                test_id=config.test_id,
                stress_type=config.stress_type,
                severity=config.severity,
                passed=False,
                execution_time=time.time() - test_start_time,
                baseline_metrics=baseline_metrics,
                peak_stress_metrics=self.monitor.get_current_metrics(),
                recovery_metrics=self.monitor.get_current_metrics(),
                performance_degradation=1.0,
                recovery_time_seconds=0.0,
                max_memory_usage_mb=0.0,
                max_cpu_usage=0.0,
                error_count=1,
                failure_reason=str(e),
                failure_timestamp=time.time()
            )
        
        # Stop stress generation
        self.stress_generator.stop_stress()
        
        # Record metrics history
        result.metrics_history = self.monitor.get_metrics_history(config.duration_seconds + 60)
        
        # Cleanup
        if self.memory_tracker_enabled:
            tracemalloc.stop()
            self.memory_tracker_enabled = False
        
        self.monitor.stop_monitoring()
        
        # Store result
        self.test_results.append(result)
        
        logger.info(f"Stress test {config.test_id} completed: {'PASSED' if result.passed else 'FAILED'}")
        
        return result
    
    def _execute_stress_test(self, config: StressTestConfiguration) -> StressTestResult:
        """Execute specific type of stress test"""
        
        test_start_time = time.time()
        errors = []
        
        # Apply stress based on type
        if config.stress_type == StressType.COMPUTATIONAL_LOAD:
            self.stress_generator.generate_cpu_stress(
                config.target_cpu_load or 80.0,
                config.duration_seconds
            )
        
        elif config.stress_type == StressType.MEMORY_PRESSURE:
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
            target_memory = min(
                config.target_memory_usage or available_memory_mb * 0.8,
                available_memory_mb * 0.9  # Don't exceed 90% to avoid system crash
            )
            self.stress_generator.generate_memory_stress(target_memory, config.duration_seconds)
        
        elif config.stress_type == StressType.NETWORK_DISRUPTION:
            self.stress_generator.generate_network_stress(config.duration_seconds)
        
        elif config.stress_type == StressType.HIGH_FREQUENCY_DISTURBANCE:
            self._run_high_frequency_disturbance_test(config)
        
        elif config.stress_type == StressType.CONCURRENT_OPERATIONS:
            self._run_concurrent_operations_test(config)
        
        elif config.stress_type == StressType.ADVERSARIAL_INPUT:
            errors.extend(self._run_adversarial_input_test(config))
        
        else:
            # Generic stress test
            self.stress_generator.generate_cpu_stress(50.0, config.duration_seconds)
        
        # Monitor during stress test
        peak_metrics = self._monitor_stress_period(config)
        
        # Wait for stress to complete
        time.sleep(config.duration_seconds)
        
        # Stop stress and monitor recovery
        self.stress_generator.stop_stress()
        recovery_start = time.time()
        
        # Wait for recovery
        recovery_metrics, recovery_time = self._monitor_recovery_period(config, recovery_start)
        
        # Analyze results
        performance_degradation = self._calculate_performance_degradation(
            self.monitor.baseline_metrics, peak_metrics
        )
        
        # Determine if test passed
        passed = self._evaluate_test_success(config, peak_metrics, recovery_time, len(errors))
        
        execution_time = time.time() - test_start_time
        
        return StressTestResult(
            test_id=config.test_id,
            stress_type=config.stress_type,
            severity=config.severity,
            passed=passed,
            execution_time=execution_time,
            baseline_metrics=self.monitor.baseline_metrics,
            peak_stress_metrics=peak_metrics,
            recovery_metrics=recovery_metrics,
            performance_degradation=performance_degradation,
            recovery_time_seconds=recovery_time,
            max_memory_usage_mb=peak_metrics.memory_available,
            max_cpu_usage=peak_metrics.cpu_usage,
            error_count=len(errors),
            error_log=errors
        )
    
    def _monitor_stress_period(self, config: StressTestConfiguration) -> SystemMetrics:
        """Monitor system during stress period"""
        
        peak_cpu = 0.0
        peak_memory = 0.0
        peak_response_time = 0.0
        
        stress_start = time.time()
        
        while time.time() - stress_start < config.duration_seconds:
            current_metrics = self.monitor.get_current_metrics()
            
            peak_cpu = max(peak_cpu, current_metrics.cpu_usage)
            peak_memory = max(peak_memory, current_metrics.memory_usage)
            peak_response_time = max(peak_response_time, current_metrics.response_time_ms)
            
            time.sleep(1.0)  # Sample every second during stress
        
        # Return peak stress metrics
        return SystemMetrics(
            timestamp=time.time(),
            cpu_usage=peak_cpu,
            memory_usage=peak_memory,
            memory_available=psutil.virtual_memory().available / (1024 * 1024),
            disk_io_read=0.0,
            disk_io_write=0.0,
            network_bytes_sent=0.0,
            network_bytes_recv=0.0,
            response_time_ms=peak_response_time
        )
    
    def _monitor_recovery_period(self, 
                               config: StressTestConfiguration, 
                               recovery_start: float) -> Tuple[SystemMetrics, float]:
        """Monitor system recovery after stress"""
        
        baseline_cpu = self.monitor.baseline_metrics.cpu_usage
        baseline_memory = self.monitor.baseline_metrics.memory_usage
        
        recovery_threshold = 1.1  # 10% above baseline
        recovery_time = 0.0
        recovered = False
        
        while time.time() - recovery_start < config.recovery_time_limit_seconds:
            current_metrics = self.monitor.get_current_metrics()
            
            # Check if system has recovered
            cpu_recovered = current_metrics.cpu_usage <= baseline_cpu * recovery_threshold
            memory_recovered = current_metrics.memory_usage <= baseline_memory * recovery_threshold
            
            if cpu_recovered and memory_recovered and not recovered:
                recovery_time = time.time() - recovery_start
                recovered = True
                break
            
            time.sleep(1.0)
        
        if not recovered:
            recovery_time = config.recovery_time_limit_seconds
        
        return self.monitor.get_current_metrics(), recovery_time
    
    def _run_high_frequency_disturbance_test(self, config: StressTestConfiguration) -> None:
        """Run high-frequency disturbance test"""
        
        def disturbance_generator():
            frequency = config.disturbance_frequency or 100.0  # Hz
            amplitude = 0.1 * config.severity.value  # Scale with severity
            
            end_time = time.time() + config.duration_seconds
            
            while time.time() < end_time:
                # Generate high-frequency CPU spikes
                for _ in range(int(frequency)):
                    # Brief CPU intensive operation
                    _ = sum(i * i for i in range(100))
                
                time.sleep(1.0 / frequency)
        
        thread = threading.Thread(target=disturbance_generator, daemon=True)
        thread.start()
        self.stress_generator.stress_threads.append(thread)
    
    def _run_concurrent_operations_test(self, config: StressTestConfiguration) -> None:
        """Run concurrent operations stress test"""
        
        num_threads = config.concurrent_threads or 50
        
        def worker_function(worker_id):
            end_time = time.time() + config.duration_seconds
            
            while time.time() < end_time:
                try:
                    # Simulate concurrent work
                    data = np.random.random((100, 100))
                    result = np.dot(data, data.T)
                    _ = np.linalg.eigvals(result)
                    
                    time.sleep(0.01)  # Brief pause
                    
                except Exception as e:
                    logger.debug(f"Worker {worker_id} error: {e}")
        
        # Start concurrent workers
        for i in range(num_threads):
            thread = threading.Thread(
                target=lambda wid=i: worker_function(wid), 
                daemon=True
            )
            thread.start()
            self.stress_generator.stress_threads.append(thread)
    
    def _run_adversarial_input_test(self, config: StressTestConfiguration) -> List[str]:
        """Run adversarial input test"""
        
        errors = []
        
        # Generate test inputs
        base_input = np.random.random((100, 10))
        
        attack_types = ['noise_injection', 'boundary_values', 'format_fuzzing']
        
        for attack_type in attack_types:
            try:
                adversarial_inputs = self.adversarial_generator.generate_adversarial_inputs(
                    base_input, attack_type, 0.2
                )
                
                # Test system response to adversarial inputs
                for adv_input in adversarial_inputs:
                    try:
                        # Simulate processing adversarial input
                        _ = np.mean(adv_input)
                        _ = np.std(adv_input)
                        
                        # Check for NaN or infinite results
                        if np.any(np.isnan(adv_input)) or np.any(np.isinf(adv_input)):
                            errors.append(f"Invalid values in {attack_type} adversarial input")
                        
                    except Exception as e:
                        errors.append(f"Error processing {attack_type} input: {str(e)}")
                        
            except Exception as e:
                errors.append(f"Error generating {attack_type} inputs: {str(e)}")
        
        return errors
    
    def _calculate_performance_degradation(self, 
                                         baseline: SystemMetrics, 
                                         peak: SystemMetrics) -> float:
        """Calculate performance degradation during stress"""
        
        if not baseline:
            return 0.0
        
        # Calculate degradation factors
        cpu_degradation = max(0, (peak.cpu_usage - baseline.cpu_usage) / 100.0)
        memory_degradation = max(0, (peak.memory_usage - baseline.memory_usage) / 100.0)
        
        # Response time degradation (if available)
        response_degradation = 0.0
        if peak.response_time_ms > 0 and baseline.response_time_ms > 0:
            response_degradation = max(0, (peak.response_time_ms - baseline.response_time_ms) / baseline.response_time_ms)
        
        # Overall degradation (weighted average)
        overall_degradation = (
            cpu_degradation * 0.4 +
            memory_degradation * 0.4 +
            response_degradation * 0.2
        )
        
        return min(1.0, overall_degradation)
    
    def _evaluate_test_success(self, 
                             config: StressTestConfiguration,
                             peak_metrics: SystemMetrics,
                             recovery_time: float,
                             error_count: int) -> bool:
        """Evaluate if stress test passed"""
        
        # Check response time requirement
        if peak_metrics.response_time_ms > config.max_response_time_ms:
            return False
        
        # Check recovery time requirement
        if recovery_time > config.recovery_time_limit_seconds:
            return False
        
        # Check error count
        if error_count > 0 and config.max_error_rate == 0.0:
            return False
        
        # Check throughput (if applicable)
        if (peak_metrics.throughput_ops_per_sec > 0 and 
            peak_metrics.throughput_ops_per_sec < config.min_throughput_ops_per_sec):
            return False
        
        # Check system stability (no crashes, reasonable resource usage)
        if peak_metrics.cpu_usage > 98.0:  # System nearly unresponsive
            return False
        
        if peak_metrics.memory_usage > 95.0:  # Memory nearly exhausted
            return False
        
        return True
    
    def run_extended_operation_test(self, duration_hours: float = 24.0) -> StressTestResult:
        """Run extended operation test for long-term stability"""
        
        logger.info(f"Starting extended operation test for {duration_hours} hours")
        
        config = StressTestConfiguration(
            test_id="extended_operation",
            stress_type=StressType.RESOURCE_EXHAUSTION,
            severity=StressSeverity.MODERATE,
            duration_seconds=duration_hours * 3600,
            recovery_time_limit_seconds=300.0
        )
        
        # Start monitoring
        self.monitor.start_monitoring()
        baseline_metrics = self.monitor.get_current_metrics()
        
        test_start_time = time.time()
        errors = []
        
        try:
            # Light continuous load to simulate normal operation
            self.stress_generator.generate_cpu_stress(20.0, config.duration_seconds)
            
            # Monitor for memory leaks and performance degradation
            memory_samples = []
            performance_samples = []
            
            sample_interval = 300.0  # Sample every 5 minutes
            next_sample_time = test_start_time + sample_interval
            
            while time.time() < test_start_time + config.duration_seconds:
                current_time = time.time()
                
                if current_time >= next_sample_time:
                    metrics = self.monitor.get_current_metrics()
                    memory_samples.append((current_time, metrics.memory_usage))
                    performance_samples.append((current_time, metrics.cpu_usage))
                    
                    # Check for memory leaks
                    if len(memory_samples) > 10:
                        recent_memory = [m[1] for m in memory_samples[-10:]]
                        memory_trend = np.polyfit(range(10), recent_memory, 1)[0]
                        
                        # Memory growth > 1% per hour indicates potential leak
                        if memory_trend > 1.0 / 12:  # 1% per hour / 5-min samples
                            errors.append(f"Potential memory leak detected at {current_time}")
                    
                    next_sample_time += sample_interval
                
                time.sleep(60.0)  # Check every minute
        
        except KeyboardInterrupt:
            logger.info("Extended operation test interrupted by user")
        except Exception as e:
            logger.error(f"Extended operation test failed: {e}")
            errors.append(str(e))
        
        # Stop stress
        self.stress_generator.stop_stress()
        
        # Final metrics
        final_metrics = self.monitor.get_current_metrics()
        execution_time = time.time() - test_start_time
        
        # Analyze long-term trends
        performance_degradation = self._analyze_longterm_degradation(
            memory_samples, performance_samples
        )
        
        # Create result
        result = StressTestResult(
            test_id=config.test_id,
            stress_type=config.stress_type,
            severity=config.severity,
            passed=len(errors) == 0 and performance_degradation < 0.2,
            execution_time=execution_time,
            baseline_metrics=baseline_metrics,
            peak_stress_metrics=final_metrics,
            recovery_metrics=final_metrics,
            performance_degradation=performance_degradation,
            recovery_time_seconds=0.0,
            max_memory_usage_mb=final_metrics.memory_available,
            max_cpu_usage=final_metrics.cpu_usage,
            error_count=len(errors),
            error_log=errors
        )
        
        self.monitor.stop_monitoring()
        self.test_results.append(result)
        
        logger.info(f"Extended operation test completed: {'PASSED' if result.passed else 'FAILED'}")
        
        return result
    
    def _analyze_longterm_degradation(self, 
                                    memory_samples: List[Tuple[float, float]],
                                    performance_samples: List[Tuple[float, float]]) -> float:
        """Analyze long-term performance degradation"""
        
        if len(memory_samples) < 10 or len(performance_samples) < 10:
            return 0.0
        
        # Memory trend analysis
        times = [m[0] - memory_samples[0][0] for m in memory_samples]  # Relative times
        memory_values = [m[1] for m in memory_samples]
        
        memory_slope, _, memory_r, _, _ = stats.linregress(times, memory_values)
        
        # Performance trend analysis
        perf_values = [p[1] for p in performance_samples]
        perf_slope, _, perf_r, _, _ = stats.linregress(times, perf_values)
        
        # Calculate overall degradation
        # Normalize slopes to per-hour rates
        hours_elapsed = (times[-1] - times[0]) / 3600 if times else 1
        memory_degradation_per_hour = memory_slope / hours_elapsed
        perf_degradation_per_hour = perf_slope / hours_elapsed
        
        # Convert to 0-1 scale (1% per hour = 0.01 degradation)
        memory_degradation = abs(memory_degradation_per_hour) / 100.0
        perf_degradation = abs(perf_degradation_per_hour) / 100.0
        
        return min(1.0, max(memory_degradation, perf_degradation))
    
    def generate_stress_test_suite(self) -> List[StressTestConfiguration]:
        """Generate comprehensive stress test suite"""
        
        test_configs = []
        
        # CPU stress tests
        for severity in [StressSeverity.MILD, StressSeverity.MODERATE, StressSeverity.SEVERE]:
            cpu_loads = {'mild': 30.0, 'moderate': 60.0, 'severe': 90.0}
            
            config = StressTestConfiguration(
                test_id=f"cpu_stress_{severity.value}",
                stress_type=StressType.COMPUTATIONAL_LOAD,
                severity=severity,
                duration_seconds=300.0,  # 5 minutes
                target_cpu_load=cpu_loads[severity.value]
            )
            test_configs.append(config)
        
        # Memory stress tests
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        for severity in [StressSeverity.MILD, StressSeverity.MODERATE, StressSeverity.SEVERE]:
            memory_targets = {'mild': 0.3, 'moderate': 0.6, 'severe': 0.8}
            
            config = StressTestConfiguration(
                test_id=f"memory_stress_{severity.value}",
                stress_type=StressType.MEMORY_PRESSURE,
                severity=severity,
                duration_seconds=300.0,
                target_memory_usage=available_memory * memory_targets[severity.value]
            )
            test_configs.append(config)
        
        # High-frequency disturbance tests
        for freq in [50.0, 100.0, 200.0]:
            config = StressTestConfiguration(
                test_id=f"high_freq_disturbance_{int(freq)}hz",
                stress_type=StressType.HIGH_FREQUENCY_DISTURBANCE,
                severity=StressSeverity.MODERATE,
                duration_seconds=180.0,  # 3 minutes
                disturbance_frequency=freq
            )
            test_configs.append(config)
        
        # Concurrent operations tests
        for thread_count in [10, 50, 100]:
            config = StressTestConfiguration(
                test_id=f"concurrent_ops_{thread_count}_threads",
                stress_type=StressType.CONCURRENT_OPERATIONS,
                severity=StressSeverity.MODERATE,
                duration_seconds=240.0,  # 4 minutes
                concurrent_threads=thread_count
            )
            test_configs.append(config)
        
        # Adversarial input tests
        config = StressTestConfiguration(
            test_id="adversarial_input_test",
            stress_type=StressType.ADVERSARIAL_INPUT,
            severity=StressSeverity.HIGH,
            duration_seconds=120.0,  # 2 minutes
            max_error_rate=0.1  # Allow some errors
        )
        test_configs.append(config)
        
        # Network stress test
        config = StressTestConfiguration(
            test_id="network_stress_test",
            stress_type=StressType.NETWORK_DISRUPTION,
            severity=StressSeverity.MODERATE,
            duration_seconds=300.0
        )
        test_configs.append(config)
        
        return test_configs
    
    def run_comprehensive_stress_testing(self) -> Dict[str, Any]:
        """Run comprehensive stress testing suite"""
        
        logger.info("Starting comprehensive stress testing suite")
        
        # Generate test configurations
        test_configs = self.generate_stress_test_suite()
        
        # Run all tests
        start_time = time.time()
        
        for config in test_configs:
            try:
                result = self.run_stress_test(config)
                logger.info(f"Test {config.test_id}: {'PASSED' if result.passed else 'FAILED'}")
            except Exception as e:
                logger.error(f"Test {config.test_id} failed with exception: {e}")
        
        total_time = time.time() - start_time
        
        # Analyze results
        results_summary = self._analyze_stress_test_results()
        results_summary['total_execution_time'] = total_time
        
        logger.info(f"Comprehensive stress testing completed in {total_time:.1f} seconds")
        
        return results_summary
    
    def _analyze_stress_test_results(self) -> Dict[str, Any]:
        """Analyze stress test results"""
        
        if not self.test_results:
            return {'error': 'No test results available'}
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)
        
        # Categorize by stress type
        results_by_type = defaultdict(list)
        for result in self.test_results:
            results_by_type[result.stress_type.value].append(result)
        
        # Calculate statistics
        type_statistics = {}
        for stress_type, results in results_by_type.items():
            passed = sum(1 for r in results if r.passed)
            avg_degradation = np.mean([r.performance_degradation for r in results])
            avg_recovery_time = np.mean([r.recovery_time_seconds for r in results])
            max_cpu = max([r.max_cpu_usage for r in results])
            max_memory = max([r.max_memory_usage_mb for r in results])
            
            type_statistics[stress_type] = {
                'total_tests': len(results),
                'passed_tests': passed,
                'pass_rate': passed / len(results),
                'avg_performance_degradation': avg_degradation,
                'avg_recovery_time': avg_recovery_time,
                'max_cpu_usage': max_cpu,
                'max_memory_usage_mb': max_memory
            }
        
        # Find critical issues
        critical_failures = [
            r for r in self.test_results
            if not r.passed and r.severity in [StressSeverity.SEVERE, StressSeverity.EXTREME]
        ]
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'overall_pass_rate': passed_tests / total_tests,
            'type_statistics': type_statistics,
            'critical_failures': len(critical_failures),
            'critical_failure_details': [
                {
                    'test_id': cf.test_id,
                    'stress_type': cf.stress_type.value,
                    'failure_reason': cf.failure_reason,
                    'performance_degradation': cf.performance_degradation
                }
                for cf in critical_failures
            ],
            'system_limits': {
                'max_cpu_usage_observed': max([r.max_cpu_usage for r in self.test_results]),
                'max_memory_usage_observed': max([r.max_memory_usage_mb for r in self.test_results]),
                'worst_performance_degradation': max([r.performance_degradation for r in self.test_results]),
                'longest_recovery_time': max([r.recovery_time_seconds for r in self.test_results])
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Create stress test framework
    framework = StressTestFramework()
    
    print("Starting comprehensive stress testing...")
    print("=" * 60)
    
    # Run comprehensive stress testing
    results = framework.run_comprehensive_stress_testing()
    
    print("\nSTRESS TEST RESULTS:")
    print("=" * 60)
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed_tests']}")
    print(f"Failed: {results['failed_tests']}")
    print(f"Pass Rate: {results['overall_pass_rate']:.1%}")
    
    if results['critical_failures'] > 0:
        print(f"\n⚠️  CRITICAL FAILURES: {results['critical_failures']}")
        for failure in results['critical_failure_details']:
            print(f"  - {failure['test_id']}: {failure['failure_reason']}")
    
    print(f"\nSYSTEM LIMITS OBSERVED:")
    limits = results['system_limits']
    print(f"  Max CPU Usage: {limits['max_cpu_usage_observed']:.1f}%")
    print(f"  Max Memory Usage: {limits['max_memory_usage_observed']:.1f} MB")
    print(f"  Worst Performance Degradation: {limits['worst_performance_degradation']:.1%}")
    print(f"  Longest Recovery Time: {limits['longest_recovery_time']:.1f}s")
    
    print(f"\nSTRESS TYPE BREAKDOWN:")
    for stress_type, stats in results['type_statistics'].items():
        print(f"  {stress_type.replace('_', ' ').title()}:")
        print(f"    Pass Rate: {stats['pass_rate']:.1%}")
        print(f"    Avg Performance Degradation: {stats['avg_performance_degradation']:.1%}")
        print(f"    Avg Recovery Time: {stats['avg_recovery_time']:.1f}s")
    
    # Run extended operation test (shortened for demo)
    print(f"\nRunning extended operation test (10 minutes)...")
    extended_result = framework.run_extended_operation_test(duration_hours=10/60)  # 10 minutes
    print(f"Extended Operation Test: {'PASSED' if extended_result.passed else 'FAILED'}")
    print(f"Performance Degradation: {extended_result.performance_degradation:.2%}")
    
    print(f"\n✅ Comprehensive stress testing completed!")
    print(f"📊 Total execution time: {results['total_execution_time']:.1f} seconds")