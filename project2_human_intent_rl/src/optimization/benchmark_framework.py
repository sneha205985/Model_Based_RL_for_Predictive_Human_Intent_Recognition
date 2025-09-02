"""
Benchmarking Framework for Performance Validation
Model-Based RL Human Intent Recognition System

This module provides comprehensive benchmarking capabilities including performance
regression testing, comparative analysis, automated benchmarking, and reporting.
"""

import time
import statistics
import json
import csv
import threading
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Callable, Tuple, NamedTuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import psutil
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats as scipy_stats

try:
    import gperftools
    GPERFTOOLS_AVAILABLE = True
except ImportError:
    GPERFTOOLS_AVAILABLE = False


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    num_iterations: int = 100
    warmup_iterations: int = 10
    timeout_seconds: float = 300.0
    enable_profiling: bool = True
    enable_memory_tracking: bool = True
    enable_regression_testing: bool = True
    baseline_tolerance_percent: float = 10.0
    statistical_significance_level: float = 0.05
    output_directory: str = "benchmark_results"
    enable_parallel_benchmarks: bool = False
    max_workers: int = mp.cpu_count()
    enable_visualization: bool = True
    save_raw_data: bool = True


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    success: bool
    error_message: Optional[str] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    iteration: int = 0


@dataclass
class BenchmarkSummary:
    """Summary statistics for a benchmark."""
    name: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    p50_time: float
    p95_time: float
    p99_time: float
    mean_memory: float
    max_memory: float
    success_rate: float
    total_iterations: int
    throughput: float  # operations per second
    regression_detected: bool = False
    baseline_comparison: Optional[Dict[str, float]] = None
    custom_metrics_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)


class BenchmarkMetrics:
    """Collects and manages benchmark metrics."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None
        self.custom_metrics = {}
        self.process = psutil.Process()
    
    def start(self):
        """Start collecting metrics."""
        self.start_time = time.perf_counter()
        self.start_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.start_cpu = self.process.cpu_percent()
        self.custom_metrics.clear()
    
    def stop(self) -> Tuple[float, float, float]:
        """Stop collecting and return metrics."""
        end_time = time.perf_counter()
        end_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        end_cpu = self.process.cpu_percent()
        
        execution_time = end_time - self.start_time if self.start_time else 0
        memory_usage = max(0, end_memory - self.start_memory) if self.start_memory else end_memory
        cpu_usage = max(self.start_cpu, end_cpu) if self.start_cpu else end_cpu
        
        return execution_time, memory_usage, cpu_usage
    
    def add_custom_metric(self, name: str, value: float):
        """Add a custom metric."""
        self.custom_metrics[name] = value


class BenchmarkRunner:
    """Runs individual benchmark functions."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def run_single_benchmark(self, func: Callable, args: tuple = (), 
                           kwargs: dict = None, name: str = None) -> BenchmarkResult:
        """Run a single benchmark iteration."""
        kwargs = kwargs or {}
        name = name or func.__name__
        
        metrics = BenchmarkMetrics()
        success = True
        error_message = None
        
        try:
            metrics.start()
            result = func(*args, **kwargs)
            execution_time, memory_usage, cpu_usage = metrics.stop()
            
            # Allow functions to return custom metrics
            if isinstance(result, dict) and 'custom_metrics' in result:
                metrics.custom_metrics.update(result['custom_metrics'])
        
        except Exception as e:
            execution_time, memory_usage, cpu_usage = metrics.stop()
            success = False
            error_message = str(e)
            self.logger.error(f"Benchmark {name} failed: {e}")
        
        return BenchmarkResult(
            name=name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            success=success,
            error_message=error_message,
            custom_metrics=metrics.custom_metrics.copy()
        )
    
    def run_benchmark_suite(self, func: Callable, args: tuple = (), 
                          kwargs: dict = None, name: str = None) -> List[BenchmarkResult]:
        """Run a complete benchmark suite with warmup and iterations."""
        kwargs = kwargs or {}
        name = name or func.__name__
        
        results = []
        
        self.logger.info(f"Starting benchmark: {name}")
        
        # Warmup iterations
        self.logger.info(f"Running {self.config.warmup_iterations} warmup iterations")
        for i in range(self.config.warmup_iterations):
            try:
                func(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"Warmup iteration {i} failed: {e}")
        
        # Main benchmark iterations
        self.logger.info(f"Running {self.config.num_iterations} benchmark iterations")
        for i in range(self.config.num_iterations):
            result = self.run_single_benchmark(func, args, kwargs, name)
            result.iteration = i
            results.append(result)
            
            if not result.success and len([r for r in results if not r.success]) > self.config.num_iterations * 0.1:
                self.logger.warning(f"High failure rate in benchmark {name}, stopping early")
                break
        
        return results
    
    def run_parallel_benchmarks(self, benchmarks: List[Dict[str, Any]]) -> Dict[str, List[BenchmarkResult]]:
        """Run multiple benchmarks in parallel."""
        results = {}
        
        if not self.config.enable_parallel_benchmarks:
            # Run sequentially
            for benchmark in benchmarks:
                name = benchmark['name']
                results[name] = self.run_benchmark_suite(**benchmark)
        else:
            # Run in parallel
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_name = {}
                
                for benchmark in benchmarks:
                    name = benchmark['name']
                    future = executor.submit(self.run_benchmark_suite, **benchmark)
                    future_to_name[future] = name
                
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        results[name] = future.result(timeout=self.config.timeout_seconds)
                    except Exception as e:
                        self.logger.error(f"Parallel benchmark {name} failed: {e}")
                        results[name] = []
        
        return results


class BenchmarkAnalyzer:
    """Analyzes benchmark results and detects regressions."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_results(self, results: List[BenchmarkResult]) -> BenchmarkSummary:
        """Analyze benchmark results and generate summary."""
        if not results:
            return BenchmarkSummary(
                name="empty", mean_time=0, std_time=0, min_time=0, max_time=0,
                p50_time=0, p95_time=0, p99_time=0, mean_memory=0, max_memory=0,
                success_rate=0, total_iterations=0, throughput=0
            )
        
        # Filter successful results
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return BenchmarkSummary(
                name=results[0].name, mean_time=0, std_time=0, min_time=0, max_time=0,
                p50_time=0, p95_time=0, p99_time=0, mean_memory=0, max_memory=0,
                success_rate=0, total_iterations=len(results), throughput=0
            )
        
        # Extract metrics
        execution_times = [r.execution_time for r in successful_results]
        memory_usages = [r.memory_usage for r in successful_results]
        
        # Calculate time statistics
        mean_time = statistics.mean(execution_times)
        std_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        # Percentiles
        p50_time = np.percentile(execution_times, 50)
        p95_time = np.percentile(execution_times, 95)
        p99_time = np.percentile(execution_times, 99)
        
        # Memory statistics
        mean_memory = statistics.mean(memory_usages)
        max_memory = max(memory_usages)
        
        # Success rate
        success_rate = len(successful_results) / len(results)
        
        # Throughput
        throughput = 1.0 / mean_time if mean_time > 0 else 0
        
        # Custom metrics summary
        custom_metrics_summary = {}
        if successful_results and successful_results[0].custom_metrics:
            for metric_name in successful_results[0].custom_metrics.keys():
                metric_values = [r.custom_metrics.get(metric_name, 0) for r in successful_results]
                custom_metrics_summary[metric_name] = {
                    'mean': statistics.mean(metric_values),
                    'std': statistics.stdev(metric_values) if len(metric_values) > 1 else 0,
                    'min': min(metric_values),
                    'max': max(metric_values)
                }
        
        return BenchmarkSummary(
            name=results[0].name,
            mean_time=mean_time,
            std_time=std_time,
            min_time=min_time,
            max_time=max_time,
            p50_time=p50_time,
            p95_time=p95_time,
            p99_time=p99_time,
            mean_memory=mean_memory,
            max_memory=max_memory,
            success_rate=success_rate,
            total_iterations=len(results),
            throughput=throughput,
            custom_metrics_summary=custom_metrics_summary
        )
    
    def compare_with_baseline(self, current_summary: BenchmarkSummary,
                            baseline_summary: BenchmarkSummary) -> Dict[str, float]:
        """Compare current results with baseline."""
        if not baseline_summary:
            return {}
        
        comparison = {}
        
        # Time comparison
        if baseline_summary.mean_time > 0:
            time_change = ((current_summary.mean_time - baseline_summary.mean_time) / 
                          baseline_summary.mean_time) * 100
            comparison['time_change_percent'] = time_change
        
        # Memory comparison
        if baseline_summary.mean_memory > 0:
            memory_change = ((current_summary.mean_memory - baseline_summary.mean_memory) / 
                           baseline_summary.mean_memory) * 100
            comparison['memory_change_percent'] = memory_change
        
        # Throughput comparison
        if baseline_summary.throughput > 0:
            throughput_change = ((current_summary.throughput - baseline_summary.throughput) / 
                               baseline_summary.throughput) * 100
            comparison['throughput_change_percent'] = throughput_change
        
        return comparison
    
    def detect_regression(self, current_summary: BenchmarkSummary,
                         baseline_summary: BenchmarkSummary) -> Tuple[bool, List[str]]:
        """Detect performance regressions."""
        if not baseline_summary:
            return False, []
        
        issues = []
        regression_detected = False
        tolerance = self.config.baseline_tolerance_percent
        
        comparison = self.compare_with_baseline(current_summary, baseline_summary)
        
        # Check time regression
        time_change = comparison.get('time_change_percent', 0)
        if time_change > tolerance:
            issues.append(f"Time regression: {time_change:.1f}% slower than baseline")
            regression_detected = True
        
        # Check memory regression
        memory_change = comparison.get('memory_change_percent', 0)
        if memory_change > tolerance:
            issues.append(f"Memory regression: {memory_change:.1f}% more memory than baseline")
            regression_detected = True
        
        # Check throughput regression
        throughput_change = comparison.get('throughput_change_percent', 0)
        if throughput_change < -tolerance:
            issues.append(f"Throughput regression: {abs(throughput_change):.1f}% lower than baseline")
            regression_detected = True
        
        return regression_detected, issues
    
    def statistical_significance_test(self, results1: List[BenchmarkResult],
                                    results2: List[BenchmarkResult]) -> Dict[str, Any]:
        """Perform statistical significance test between two result sets."""
        times1 = [r.execution_time for r in results1 if r.success]
        times2 = [r.execution_time for r in results2 if r.success]
        
        if len(times1) < 3 or len(times2) < 3:
            return {'significant': False, 'reason': 'Insufficient data'}
        
        # Welch's t-test (assumes unequal variances)
        t_stat, p_value = scipy_stats.ttest_ind(times1, times2, equal_var=False)
        
        significant = p_value < self.config.statistical_significance_level
        
        return {
            'significant': significant,
            'p_value': p_value,
            't_statistic': t_stat,
            'alpha': self.config.statistical_significance_level,
            'mean_diff': statistics.mean(times2) - statistics.mean(times1)
        }


class BenchmarkVisualizer:
    """Creates visualizations for benchmark results."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_execution_times(self, summaries: Dict[str, BenchmarkSummary],
                           filename: str = "execution_times.png") -> str:
        """Plot execution time comparison."""
        if not self.config.enable_visualization:
            return ""
        
        plt.figure(figsize=(12, 8))
        
        names = list(summaries.keys())
        means = [s.mean_time for s in summaries.values()]
        stds = [s.std_time for s in summaries.values()]
        
        # Bar plot with error bars
        bars = plt.bar(names, means, yerr=stds, capsize=5, alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std,
                    f'{mean:.3f}s', ha='center', va='bottom')
        
        plt.title('Benchmark Execution Times')
        plt.xlabel('Benchmark')
        plt.ylabel('Execution Time (seconds)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath)
        plt.close()
        
        return str(filepath)
    
    def plot_memory_usage(self, summaries: Dict[str, BenchmarkSummary],
                         filename: str = "memory_usage.png") -> str:
        """Plot memory usage comparison."""
        if not self.config.enable_visualization:
            return ""
        
        plt.figure(figsize=(12, 8))
        
        names = list(summaries.keys())
        mean_memory = [s.mean_memory for s in summaries.values()]
        max_memory = [s.max_memory for s in summaries.values()]
        
        x = np.arange(len(names))
        width = 0.35
        
        plt.bar(x - width/2, mean_memory, width, label='Mean Memory', alpha=0.7)
        plt.bar(x + width/2, max_memory, width, label='Max Memory', alpha=0.7)
        
        plt.title('Benchmark Memory Usage')
        plt.xlabel('Benchmark')
        plt.ylabel('Memory Usage (MB)')
        plt.xticks(x, names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath)
        plt.close()
        
        return str(filepath)
    
    def plot_throughput(self, summaries: Dict[str, BenchmarkSummary],
                       filename: str = "throughput.png") -> str:
        """Plot throughput comparison."""
        if not self.config.enable_visualization:
            return ""
        
        plt.figure(figsize=(12, 8))
        
        names = list(summaries.keys())
        throughput = [s.throughput for s in summaries.values()]
        
        bars = plt.bar(names, throughput, alpha=0.7, color='green')
        
        # Add value labels
        for bar, tput in zip(bars, throughput):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{tput:.1f}', ha='center', va='bottom')
        
        plt.title('Benchmark Throughput')
        plt.xlabel('Benchmark')
        plt.ylabel('Throughput (operations/second)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath)
        plt.close()
        
        return str(filepath)
    
    def plot_timeline(self, results: List[BenchmarkResult],
                     filename: str = "timeline.png") -> str:
        """Plot execution time over iterations."""
        if not self.config.enable_visualization or not results:
            return ""
        
        plt.figure(figsize=(12, 8))
        
        iterations = [r.iteration for r in results if r.success]
        times = [r.execution_time for r in results if r.success]
        
        plt.plot(iterations, times, marker='o', alpha=0.7)
        plt.axhline(y=statistics.mean(times), color='red', linestyle='--', 
                   label=f'Mean: {statistics.mean(times):.3f}s')
        
        plt.title(f'Execution Time Timeline - {results[0].name}')
        plt.xlabel('Iteration')
        plt.ylabel('Execution Time (seconds)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath)
        plt.close()
        
        return str(filepath)


class BenchmarkReporter:
    """Generates benchmark reports."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_json_report(self, summaries: Dict[str, BenchmarkSummary],
                           baseline_summaries: Dict[str, BenchmarkSummary] = None,
                           filename: str = "benchmark_report.json") -> str:
        """Generate JSON report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'benchmarks': {},
            'summary': {}
        }
        
        total_benchmarks = len(summaries)
        total_regressions = 0
        
        for name, summary in summaries.items():
            benchmark_data = asdict(summary)
            
            # Add baseline comparison if available
            if baseline_summaries and name in baseline_summaries:
                baseline = baseline_summaries[name]
                benchmark_data['baseline_comparison'] = asdict(baseline)
                
                # Check for regression
                regression_detected, issues = BenchmarkAnalyzer(self.config).detect_regression(
                    summary, baseline
                )
                benchmark_data['regression_detected'] = regression_detected
                benchmark_data['regression_issues'] = issues
                
                if regression_detected:
                    total_regressions += 1
            
            report['benchmarks'][name] = benchmark_data
        
        # Summary statistics
        if summaries:
            all_times = [s.mean_time for s in summaries.values()]
            report['summary'] = {
                'total_benchmarks': total_benchmarks,
                'total_regressions': total_regressions,
                'avg_execution_time': statistics.mean(all_times),
                'total_throughput': sum(s.throughput for s in summaries.values())
            }
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(filepath)
    
    def generate_csv_report(self, summaries: Dict[str, BenchmarkSummary],
                          filename: str = "benchmark_results.csv") -> str:
        """Generate CSV report."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = [
                'name', 'mean_time', 'std_time', 'min_time', 'max_time',
                'p50_time', 'p95_time', 'p99_time', 'mean_memory', 'max_memory',
                'success_rate', 'total_iterations', 'throughput', 'regression_detected'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for name, summary in summaries.items():
                row = asdict(summary)
                # Remove complex fields that don't fit in CSV
                row.pop('baseline_comparison', None)
                row.pop('custom_metrics_summary', None)
                writer.writerow(row)
        
        return str(filepath)
    
    def generate_html_report(self, summaries: Dict[str, BenchmarkSummary],
                           baseline_summaries: Dict[str, BenchmarkSummary] = None,
                           visualization_paths: Dict[str, str] = None,
                           filename: str = "benchmark_report.html") -> str:
        """Generate HTML report."""
        html_content = self._create_html_template(summaries, baseline_summaries, visualization_paths)
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        return str(filepath)
    
    def _create_html_template(self, summaries: Dict[str, BenchmarkSummary],
                            baseline_summaries: Dict[str, BenchmarkSummary] = None,
                            visualization_paths: Dict[str, str] = None) -> str:
        """Create HTML report template."""
        visualization_paths = visualization_paths or {}
        
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .regression { background-color: #ffebee; }
        .improvement { background-color: #e8f5e8; }
        .chart { margin: 20px 0; text-align: center; }
        .summary { background-color: #f5f5f5; padding: 15px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Benchmark Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>Total Benchmarks:</strong> {total_benchmarks}</p>
        <p><strong>Regressions Detected:</strong> {regressions}</p>
    </div>
""".format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_benchmarks=len(summaries),
            regressions=sum(1 for s in summaries.values() if s.regression_detected)
        )
        
        # Add visualizations
        if visualization_paths:
            html += "<h2>Performance Charts</h2>\n"
            for chart_name, path in visualization_paths.items():
                if Path(path).exists():
                    html += f'<div class="chart"><img src="{Path(path).name}" alt="{chart_name}"></div>\n'
        
        # Add benchmark table
        html += """
    <h2>Benchmark Results</h2>
    <table>
        <tr>
            <th>Benchmark</th>
            <th>Mean Time (s)</th>
            <th>P95 Time (s)</th>
            <th>Memory (MB)</th>
            <th>Throughput (ops/s)</th>
            <th>Success Rate</th>
            <th>Status</th>
        </tr>
"""
        
        for name, summary in summaries.items():
            row_class = ""
            status = "OK"
            
            if summary.regression_detected:
                row_class = "regression"
                status = "REGRESSION"
            elif baseline_summaries and name in baseline_summaries:
                baseline = baseline_summaries[name]
                if summary.mean_time < baseline.mean_time * 0.9:  # 10% improvement
                    row_class = "improvement"
                    status = "IMPROVED"
            
            html += f"""
        <tr class="{row_class}">
            <td>{name}</td>
            <td>{summary.mean_time:.4f}</td>
            <td>{summary.p95_time:.4f}</td>
            <td>{summary.mean_memory:.1f}</td>
            <td>{summary.throughput:.1f}</td>
            <td>{summary.success_rate:.1%}</td>
            <td>{status}</td>
        </tr>
"""
        
        html += """
    </table>
</body>
</html>
"""
        
        return html


class BenchmarkFramework:
    """Main benchmarking framework."""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.runner = BenchmarkRunner(self.config)
        self.analyzer = BenchmarkAnalyzer(self.config)
        self.visualizer = BenchmarkVisualizer(self.config)
        self.reporter = BenchmarkReporter(self.config)
        
        # Storage for baselines
        self.baselines = {}
        self.baseline_file = Path(self.config.output_directory) / "baselines.json"
        
        self.logger = logging.getLogger(__name__)
        
        # Load existing baselines
        self._load_baselines()
    
    def benchmark(self, func: Callable, args: tuple = (), kwargs: dict = None,
                 name: str = None) -> BenchmarkSummary:
        """Run a single benchmark."""
        results = self.runner.run_benchmark_suite(func, args, kwargs, name)
        summary = self.analyzer.analyze_results(results)
        
        # Check for regression if baseline exists
        if self.config.enable_regression_testing:
            baseline = self.baselines.get(summary.name)
            if baseline:
                regression_detected, issues = self.analyzer.detect_regression(summary, baseline)
                summary.regression_detected = regression_detected
                
                if regression_detected:
                    self.logger.warning(f"Regression detected in {summary.name}: {issues}")
        
        return summary
    
    def benchmark_suite(self, benchmarks: List[Dict[str, Any]]) -> Dict[str, BenchmarkSummary]:
        """Run a suite of benchmarks."""
        # Run all benchmarks
        all_results = self.runner.run_parallel_benchmarks(benchmarks)
        
        # Analyze results
        summaries = {}
        for name, results in all_results.items():
            summaries[name] = self.analyzer.analyze_results(results)
        
        # Check regressions
        if self.config.enable_regression_testing:
            for name, summary in summaries.items():
                baseline = self.baselines.get(name)
                if baseline:
                    regression_detected, issues = self.analyzer.detect_regression(summary, baseline)
                    summary.regression_detected = regression_detected
        
        return summaries
    
    def set_baseline(self, name: str, summary: BenchmarkSummary):
        """Set a benchmark result as baseline for regression testing."""
        self.baselines[name] = summary
        self._save_baselines()
        self.logger.info(f"Set baseline for {name}")
    
    def generate_full_report(self, summaries: Dict[str, BenchmarkSummary],
                           report_name: str = "benchmark") -> Dict[str, str]:
        """Generate complete benchmark report with all formats."""
        output_files = {}
        
        # Generate visualizations
        visualization_paths = {}
        if self.config.enable_visualization:
            visualization_paths['execution_times'] = self.visualizer.plot_execution_times(
                summaries, f"{report_name}_execution_times.png"
            )
            visualization_paths['memory_usage'] = self.visualizer.plot_memory_usage(
                summaries, f"{report_name}_memory_usage.png"
            )
            visualization_paths['throughput'] = self.visualizer.plot_throughput(
                summaries, f"{report_name}_throughput.png"
            )
        
        # Generate reports
        output_files['json'] = self.reporter.generate_json_report(
            summaries, self.baselines, f"{report_name}.json"
        )
        output_files['csv'] = self.reporter.generate_csv_report(
            summaries, f"{report_name}.csv"
        )
        output_files['html'] = self.reporter.generate_html_report(
            summaries, self.baselines, visualization_paths, f"{report_name}.html"
        )
        
        output_files.update(visualization_paths)
        
        return output_files
    
    def _load_baselines(self):
        """Load baseline benchmarks from file."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    data = json.load(f)
                
                for name, baseline_data in data.items():
                    # Convert dict back to BenchmarkSummary
                    self.baselines[name] = BenchmarkSummary(**baseline_data)
                
                self.logger.info(f"Loaded {len(self.baselines)} baselines")
            except Exception as e:
                self.logger.error(f"Failed to load baselines: {e}")
    
    def _save_baselines(self):
        """Save baseline benchmarks to file."""
        try:
            # Create output directory
            self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dict for JSON serialization
            data = {name: asdict(summary) for name, summary in self.baselines.items()}
            
            with open(self.baseline_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Failed to save baselines: {e}")
    
    @contextmanager
    def benchmark_context(self, name: str):
        """Context manager for benchmarking code blocks."""
        metrics = BenchmarkMetrics()
        metrics.start()
        
        try:
            yield metrics
        finally:
            execution_time, memory_usage, cpu_usage = metrics.stop()
            
            result = BenchmarkResult(
                name=name,
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                success=True,
                custom_metrics=metrics.custom_metrics.copy()
            )
            
            self.logger.info(f"Benchmark {name}: {execution_time:.4f}s")


# Example benchmark functions
def create_example_benchmarks():
    """Create example benchmark functions for testing."""
    
    def quick_computation():
        """Quick computation benchmark."""
        return sum(i**2 for i in range(1000))
    
    def medium_computation():
        """Medium computation benchmark."""
        result = 0
        for i in range(10000):
            result += i**2
        return result
    
    def memory_intensive():
        """Memory-intensive benchmark."""
        data = [np.random.randn(100, 100) for _ in range(10)]
        return sum(np.sum(arr) for arr in data)
    
    def custom_metrics_benchmark():
        """Benchmark that returns custom metrics."""
        start = time.time()
        result = sum(i**3 for i in range(5000))
        custom_time = time.time() - start
        
        return {
            'result': result,
            'custom_metrics': {
                'custom_timing': custom_time,
                'result_magnitude': abs(result)
            }
        }
    
    return {
        'quick_computation': {'func': quick_computation, 'name': 'Quick Computation'},
        'medium_computation': {'func': medium_computation, 'name': 'Medium Computation'},
        'memory_intensive': {'func': memory_intensive, 'name': 'Memory Intensive'},
        'custom_metrics': {'func': custom_metrics_benchmark, 'name': 'Custom Metrics'}
    }


if __name__ == "__main__":
    # Example usage
    config = BenchmarkConfig(
        num_iterations=50,
        warmup_iterations=5,
        enable_visualization=True,
        enable_regression_testing=True
    )
    
    framework = BenchmarkFramework(config)
    
    # Get example benchmarks
    example_benchmarks = create_example_benchmarks()
    
    print("Running benchmark suite...")
    
    # Prepare benchmark list
    benchmark_list = [
        {
            'func': bench_info['func'],
            'name': bench_info['name']
        }
        for bench_info in example_benchmarks.values()
    ]
    
    # Run benchmarks
    summaries = framework.benchmark_suite(benchmark_list)
    
    # Set baselines (first run)
    for name, summary in summaries.items():
        framework.set_baseline(name, summary)
    
    # Generate reports
    output_files = framework.generate_full_report(summaries, "example_benchmarks")
    
    print("\nBenchmark Summary:")
    for name, summary in summaries.items():
        print(f"{name}:")
        print(f"  Mean time: {summary.mean_time:.4f}s")
        print(f"  Throughput: {summary.throughput:.1f} ops/s")
        print(f"  Success rate: {summary.success_rate:.1%}")
        if summary.regression_detected:
            print(f"  ⚠️  REGRESSION DETECTED")
        print()
    
    print("\nGenerated reports:")
    for report_type, filepath in output_files.items():
        print(f"  {report_type}: {filepath}")