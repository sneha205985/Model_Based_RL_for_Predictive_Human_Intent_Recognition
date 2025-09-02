#!/usr/bin/env python3
"""
Production Performance Benchmark
Model-Based RL Human Intent Recognition System

Validates real-time <10ms decision cycles in production environment.
Comprehensive performance testing with detailed metrics collection.

Author: Production Team
Date: September 2025
"""

import time
import json
import logging
import statistics
import threading
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import psutil
import numpy as np

# Import project components
from src.system.human_intent_rl_system import HumanIntentRLSystem
from src.utils.optional_dependencies import get_dash, HAS_PSUTIL

class ProductionBenchmark:
    """
    Production-grade performance benchmark for validating 
    real-time decision cycle performance claims.
    """
    
    def __init__(self, target_ms: float = 10.0):
        """
        Initialize benchmark with performance targets.
        
        Args:
            target_ms: Maximum acceptable decision cycle time (ms)
        """
        self.target_ms = target_ms
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'target_performance': f"<{target_ms}ms",
            'environment': 'production',
            'system_info': self._get_system_info(),
            'benchmarks': {}
        }
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/production_benchmark.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize system components
        try:
            self.rl_system = HumanIntentRLSystem()
            self.logger.info("RL System initialized successfully")
        except Exception as e:
            self.logger.warning(f"RL System initialization with defaults: {e}")
            self.rl_system = HumanIntentRLSystem({})

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for performance context."""
        if HAS_PSUTIL:
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
                'platform': platform.platform()
            }
        return {'platform': 'unknown', 'monitoring': 'limited'}

    def run_decision_cycle_benchmark(self, num_iterations: int = 1000) -> Dict[str, Any]:
        """
        Benchmark core decision cycle performance.
        
        Args:
            num_iterations: Number of decision cycles to test
            
        Returns:
            Benchmark results with timing statistics
        """
        self.logger.info(f"Running decision cycle benchmark ({num_iterations} iterations)")
        
        decision_times = []
        successful_cycles = 0
        
        for i in range(num_iterations):
            try:
                # Simulate realistic human intent data
                intent_data = np.random.rand(10, 3)  # 10 observations, 3 features
                
                start_time = time.perf_counter()
                
                # Execute decision cycle
                prediction = self.rl_system.predict_intent(intent_data)
                control_action = self.rl_system.generate_control_action(prediction)
                safety_check = self.rl_system.validate_safety(control_action)
                
                end_time = time.perf_counter()
                cycle_time_ms = (end_time - start_time) * 1000
                
                decision_times.append(cycle_time_ms)
                successful_cycles += 1
                
                if i % 100 == 0:
                    self.logger.debug(f"Cycle {i}: {cycle_time_ms:.2f}ms")
                    
            except Exception as e:
                self.logger.warning(f"Decision cycle {i} failed: {e}")
                
        # Calculate statistics
        if decision_times:
            stats = {
                'total_cycles': num_iterations,
                'successful_cycles': successful_cycles,
                'success_rate': f"{(successful_cycles/num_iterations)*100:.1f}%",
                'mean_ms': round(statistics.mean(decision_times), 3),
                'median_ms': round(statistics.median(decision_times), 3),
                'std_ms': round(statistics.stdev(decision_times) if len(decision_times) > 1 else 0, 3),
                'min_ms': round(min(decision_times), 3),
                'max_ms': round(max(decision_times), 3),
                'p95_ms': round(np.percentile(decision_times, 95), 3),
                'p99_ms': round(np.percentile(decision_times, 99), 3),
                'target_compliance': f"{sum(1 for t in decision_times if t < self.target_ms)}/{len(decision_times)}",
                'compliance_rate': f"{(sum(1 for t in decision_times if t < self.target_ms)/len(decision_times))*100:.1f}%"
            }
            
            self.logger.info(f"Decision Cycle Results: {stats['mean_ms']:.2f}ms avg, "
                           f"{stats['compliance_rate']} within target")
            
            return {
                'status': 'PASS' if stats['mean_ms'] < self.target_ms else 'FAIL',
                'statistics': stats,
                'raw_times': decision_times[:100]  # Store first 100 for analysis
            }
        else:
            return {
                'status': 'FAIL',
                'error': 'No successful decision cycles completed'
            }

    def run_concurrent_load_test(self, concurrent_threads: int = 10, 
                                duration_seconds: int = 60) -> Dict[str, Any]:
        """
        Test performance under concurrent load.
        
        Args:
            concurrent_threads: Number of concurrent decision threads
            duration_seconds: Test duration in seconds
            
        Returns:
            Load test results
        """
        self.logger.info(f"Running concurrent load test ({concurrent_threads} threads, "
                        f"{duration_seconds}s)")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        all_times = []
        thread_results = []
        
        def worker_thread():
            """Worker thread for concurrent testing."""
            thread_times = []
            thread_cycles = 0
            
            while time.time() < end_time:
                try:
                    intent_data = np.random.rand(5, 3)
                    
                    cycle_start = time.perf_counter()
                    prediction = self.rl_system.predict_intent(intent_data)
                    cycle_end = time.perf_counter()
                    
                    cycle_time_ms = (cycle_end - cycle_start) * 1000
                    thread_times.append(cycle_time_ms)
                    thread_cycles += 1
                    
                except Exception as e:
                    self.logger.debug(f"Thread error: {e}")
                    
            return {'times': thread_times, 'cycles': thread_cycles}
        
        # Execute concurrent threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_threads) as executor:
            futures = [executor.submit(worker_thread) for _ in range(concurrent_threads)]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    thread_results.append(result)
                    all_times.extend(result['times'])
                except Exception as e:
                    self.logger.error(f"Thread execution error: {e}")
        
        # Analyze results
        if all_times:
            total_cycles = sum(r['cycles'] for r in thread_results)
            throughput = total_cycles / duration_seconds
            
            stats = {
                'duration_seconds': duration_seconds,
                'concurrent_threads': concurrent_threads,
                'total_cycles': total_cycles,
                'throughput_cycles_per_sec': round(throughput, 1),
                'mean_ms': round(statistics.mean(all_times), 3),
                'p95_ms': round(np.percentile(all_times, 95), 3),
                'compliance_rate': f"{(sum(1 for t in all_times if t < self.target_ms)/len(all_times))*100:.1f}%",
                'thread_performance': [
                    {
                        'thread': i,
                        'cycles': r['cycles'],
                        'avg_ms': round(statistics.mean(r['times']) if r['times'] else 0, 3)
                    }
                    for i, r in enumerate(thread_results)
                ]
            }
            
            self.logger.info(f"Load Test Results: {throughput:.1f} cycles/sec, "
                           f"{stats['mean_ms']:.2f}ms avg")
            
            return {
                'status': 'PASS' if stats['mean_ms'] < self.target_ms else 'FAIL',
                'statistics': stats
            }
        else:
            return {
                'status': 'FAIL',
                'error': 'No cycles completed during load test'
            }

    def run_memory_performance_test(self) -> Dict[str, Any]:
        """Test memory usage and performance correlation."""
        self.logger.info("Running memory performance analysis")
        
        if not HAS_PSUTIL:
            return {
                'status': 'SKIP',
                'reason': 'psutil not available for memory monitoring'
            }
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        memory_samples = []
        performance_samples = []
        
        for i in range(100):
            # Take memory snapshot
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            
            # Performance test
            intent_data = np.random.rand(20, 5)  # Larger data
            start_time = time.perf_counter()
            prediction = self.rl_system.predict_intent(intent_data)
            end_time = time.perf_counter()
            
            performance_samples.append((end_time - start_time) * 1000)
            
        memory_growth = max(memory_samples) - initial_memory
        avg_performance = statistics.mean(performance_samples)
        
        return {
            'status': 'PASS' if avg_performance < self.target_ms else 'WARN',
            'initial_memory_mb': round(initial_memory, 2),
            'peak_memory_mb': round(max(memory_samples), 2),
            'memory_growth_mb': round(memory_growth, 2),
            'avg_performance_ms': round(avg_performance, 3),
            'memory_stable': memory_growth < 10  # Less than 10MB growth
        }

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """
        Run complete production benchmark suite.
        
        Returns:
            Comprehensive benchmark results
        """
        self.logger.info("Starting comprehensive production benchmark")
        benchmark_start = time.time()
        
        # Run all benchmark tests
        self.results['benchmarks']['decision_cycles'] = self.run_decision_cycle_benchmark(1000)
        self.results['benchmarks']['concurrent_load'] = self.run_concurrent_load_test(10, 60)
        self.results['benchmarks']['memory_performance'] = self.run_memory_performance_test()
        
        # Overall assessment
        total_time = time.time() - benchmark_start
        self.results['execution_time_seconds'] = round(total_time, 2)
        
        # Determine overall status
        test_statuses = [
            self.results['benchmarks']['decision_cycles']['status'],
            self.results['benchmarks']['concurrent_load']['status']
        ]
        
        if all(status == 'PASS' for status in test_statuses):
            overall_status = 'PASS'
        elif any(status == 'FAIL' for status in test_statuses):
            overall_status = 'FAIL'
        else:
            overall_status = 'WARN'
        
        self.results['overall_status'] = overall_status
        self.results['summary'] = self._generate_summary()
        
        self.logger.info(f"Benchmark completed in {total_time:.1f}s - Status: {overall_status}")
        
        return self.results

    def _generate_summary(self) -> Dict[str, str]:
        """Generate executive summary of benchmark results."""
        decision_result = self.results['benchmarks']['decision_cycles']
        load_result = self.results['benchmarks']['concurrent_load']
        
        if decision_result['status'] == 'PASS':
            avg_ms = decision_result['statistics']['mean_ms']
            compliance = decision_result['statistics']['compliance_rate']
            decision_summary = f"✅ Decision cycles: {avg_ms}ms avg ({compliance} compliant)"
        else:
            decision_summary = "❌ Decision cycles: Failed to meet performance target"
        
        if load_result['status'] == 'PASS':
            throughput = load_result['statistics']['throughput_cycles_per_sec']
            load_summary = f"✅ Load test: {throughput} cycles/sec sustained"
        else:
            load_summary = "❌ Load test: Failed under concurrent load"
        
        return {
            'decision_cycles': decision_summary,
            'load_performance': load_summary,
            'production_ready': self.results['overall_status'] == 'PASS'
        }

    def save_results(self, filename: str = None):
        """Save benchmark results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics/production_benchmark_{timestamp}.json"
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Benchmark results saved to {filename}")


def main():
    """Main execution function."""
    import sys
    import platform
    
    print("=" * 60)
    print("Production Performance Benchmark")
    print("Model-Based RL Human Intent Recognition")
    print("=" * 60)
    
    # Initialize and run benchmark
    benchmark = ProductionBenchmark(target_ms=10.0)
    results = benchmark.run_comprehensive_benchmark()
    
    # Save results
    benchmark.save_results()
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(f"Overall Status: {results['overall_status']}")
    print(f"Execution Time: {results['execution_time_seconds']}s")
    print("\nKey Metrics:")
    
    for test_name, test_results in results['benchmarks'].items():
        status = test_results['status']
        print(f"  {test_name}: {status}")
        
    print("\nSummary:")
    for key, summary in results['summary'].items():
        print(f"  {summary}")
    
    # Exit with appropriate code
    exit_code = 0 if results['overall_status'] == 'PASS' else 1
    print(f"\nExiting with code: {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()