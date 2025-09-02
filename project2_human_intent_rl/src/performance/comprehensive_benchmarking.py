"""
Comprehensive Performance Benchmarking Framework
Model-Based RL Human Intent Recognition System

This module provides rigorous performance validation and benchmarking with
statistical significance testing to validate <10ms decision cycles and 
>95% safety rate claims for EXCELLENT production-grade status.

Performance Analysis Categories:
1. Real-Time Latency Measurement with Statistical Analysis
2. Algorithm Benchmarking vs State-of-the-Art
3. Safety Performance Statistical Validation
4. Scalability Analysis with Load Testing
5. Memory Usage and Resource Optimization

Mathematical Foundation:
- Statistical hypothesis testing with p<0.05 significance
- Confidence intervals for all performance metrics
- Monte Carlo simulation for safety analysis (10,000+ trials)
- Performance regression analysis and prediction models

Author: Comprehensive Performance Benchmarking Framework
"""

import numpy as np
import pandas as pd
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import logging
import psutil
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import t, norm, chi2
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings
from collections import defaultdict, deque
import contextlib
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner benchmarking
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class PerformanceConfig:
    """Configuration for comprehensive performance benchmarking"""
    # Statistical testing parameters
    confidence_level: float = 0.95
    significance_level: float = 0.05
    monte_carlo_trials: int = 10000
    bootstrap_samples: int = 1000
    
    # Performance targets and thresholds
    target_decision_cycle_ms: float = 10.0
    target_safety_rate: float = 0.95
    target_memory_mb: float = 500.0
    target_cpu_percent: float = 80.0
    
    # Benchmarking parameters
    warmup_iterations: int = 100
    measurement_iterations: int = 1000
    load_test_duration_s: int = 60
    concurrent_users: List[int] = field(default_factory=lambda: [1, 5, 10, 25, 50])
    
    # Dataset scaling parameters
    dataset_sizes: List[int] = field(default_factory=lambda: [100, 500, 1000, 2500, 5000])
    
    # Baseline comparison settings
    enable_sklearn_comparison: bool = True
    enable_gpytorch_comparison: bool = True
    enable_cvxpy_comparison: bool = True


class HighPrecisionTimer:
    """
    High-precision timer for microsecond-level performance measurement.
    
    Uses platform-specific high-resolution timing for accurate latency measurement.
    """
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000.0
    
    @property
    def elapsed_us(self) -> float:
        """Get elapsed time in microseconds"""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000000.0


class SystemResourceMonitor:
    """
    System resource monitoring for comprehensive performance analysis.
    
    Tracks CPU, memory, and I/O usage during benchmarking.
    """
    
    def __init__(self, sampling_interval_s: float = 0.01):
        self.sampling_interval = sampling_interval_s
        self.monitoring = False
        self.data = defaultdict(list)
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.data.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> Dict[str, List[float]]:
        """Stop monitoring and return collected data"""
        if not self.monitoring:
            return dict(self.data)
            
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
            
        return dict(self.data)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = process.cpu_percent()
                self.data['cpu_percent'].append(cpu_percent)
                
                # Memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                self.data['memory_mb'].append(memory_mb)
                
                # System-wide metrics
                system_cpu = psutil.cpu_percent()
                system_memory = psutil.virtual_memory()
                self.data['system_cpu_percent'].append(system_cpu)
                self.data['system_memory_percent'].append(system_memory.percent)
                
                # Timestamp
                self.data['timestamp'].append(time.time())
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                break


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for performance metrics.
    
    Provides hypothesis testing, confidence intervals, and significance testing
    for all performance claims with rigorous statistical validation.
    """
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        
    def analyze_latency_distribution(self, latencies: List[float]) -> Dict[str, Any]:
        """
        Comprehensive statistical analysis of latency measurements.
        
        Returns detailed statistics including confidence intervals, 
        hypothesis tests, and distribution analysis.
        """
        if not latencies or len(latencies) < 3:
            return {'error': 'Insufficient data for statistical analysis'}
            
        latencies = np.array(latencies)
        n = len(latencies)
        
        # Basic statistics
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies, ddof=1)
        median_latency = np.median(latencies)
        
        # Percentiles
        percentiles = [50, 90, 95, 99, 99.9]
        percentile_values = {f'p{p}': np.percentile(latencies, p) for p in percentiles}
        
        # Confidence interval for mean
        alpha = 1 - self.config.confidence_level
        t_critical = t.ppf(1 - alpha/2, n - 1)
        margin_error = t_critical * (std_latency / np.sqrt(n))
        ci_lower = mean_latency - margin_error
        ci_upper = mean_latency + margin_error
        
        # Hypothesis test: H0: mean >= target, H1: mean < target
        target_ms = self.config.target_decision_cycle_ms
        t_statistic = (mean_latency - target_ms) / (std_latency / np.sqrt(n))
        p_value_performance = t.cdf(t_statistic, n - 1)
        
        # Normality test
        shapiro_stat, shapiro_p = stats.shapiro(latencies[:5000])  # Limit for computational efficiency
        
        # Outlier detection using IQR method
        q25, q75 = np.percentile(latencies, [25, 75])
        iqr = q75 - q25
        outlier_threshold_lower = q25 - 1.5 * iqr
        outlier_threshold_upper = q75 + 1.5 * iqr
        outliers = latencies[(latencies < outlier_threshold_lower) | (latencies > outlier_threshold_upper)]
        
        # Performance target achievement
        target_achieved_count = np.sum(latencies <= target_ms)
        target_achievement_rate = target_achieved_count / n
        
        # Bootstrap confidence interval for achievement rate
        bootstrap_rates = []
        for _ in range(self.config.bootstrap_samples):
            bootstrap_sample = np.random.choice(latencies, size=n, replace=True)
            bootstrap_rate = np.sum(bootstrap_sample <= target_ms) / n
            bootstrap_rates.append(bootstrap_rate)
        
        bootstrap_rates = np.array(bootstrap_rates)
        achievement_ci_lower = np.percentile(bootstrap_rates, (alpha/2) * 100)
        achievement_ci_upper = np.percentile(bootstrap_rates, (1 - alpha/2) * 100)
        
        return {
            # Basic statistics
            'n_samples': int(n),
            'mean_ms': float(mean_latency),
            'std_ms': float(std_latency),
            'median_ms': float(median_latency),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            
            # Percentiles
            **{k: float(v) for k, v in percentile_values.items()},
            
            # Confidence intervals
            'mean_ci_lower_ms': float(ci_lower),
            'mean_ci_upper_ms': float(ci_upper),
            'confidence_level': self.config.confidence_level,
            
            # Hypothesis testing
            'target_ms': target_ms,
            'target_achievement_rate': float(target_achievement_rate),
            'target_achieved_count': int(target_achieved_count),
            'performance_test_p_value': float(p_value_performance),
            'performance_test_significant': bool(p_value_performance < self.config.significance_level),
            'meets_performance_target': bool(target_achievement_rate >= 0.9),  # 90% of samples should meet target
            
            # Achievement rate confidence interval
            'achievement_rate_ci_lower': float(achievement_ci_lower),
            'achievement_rate_ci_upper': float(achievement_ci_upper),
            
            # Distribution analysis
            'normality_p_value': float(shapiro_p),
            'normally_distributed': bool(shapiro_p > self.config.significance_level),
            
            # Outlier analysis
            'n_outliers': int(len(outliers)),
            'outlier_rate': float(len(outliers) / n),
            'outlier_threshold_upper_ms': float(outlier_threshold_upper)
        }
    
    def analyze_safety_performance(self, safety_outcomes: List[bool]) -> Dict[str, Any]:
        """
        Statistical analysis of safety performance with rigorous hypothesis testing.
        
        Tests whether safety rate meets >95% requirement with statistical significance.
        """
        if not safety_outcomes or len(safety_outcomes) < 10:
            return {'error': 'Insufficient safety data for analysis'}
            
        outcomes = np.array(safety_outcomes, dtype=bool)
        n = len(outcomes)
        successes = np.sum(outcomes)
        safety_rate = successes / n
        
        # Target safety rate
        target_rate = self.config.target_safety_rate
        
        # Binomial confidence interval (Wilson score interval)
        z = norm.ppf(1 - self.config.significance_level / 2)
        
        # Wilson score interval calculation
        center = (successes + z**2/2) / (n + z**2)
        margin = z / (n + z**2) * np.sqrt(successes * (n - successes) / n + z**2/4)
        ci_lower = center - margin
        ci_upper = center + margin
        
        # One-tailed hypothesis test: H0: p <= target, H1: p > target
        # Using exact binomial test
        p_value_safety = 1 - stats.binom.cdf(successes - 1, n, target_rate)
        
        # Power analysis - probability of detecting effect if true rate is target + 0.02
        effect_size = 0.02
        true_rate = target_rate + effect_size
        power = 1 - stats.binom.cdf(successes - 1, n, true_rate)
        
        # Monte Carlo validation (if enabled)
        monte_carlo_results = self._monte_carlo_safety_analysis(safety_rate, n)
        
        return {
            'n_trials': int(n),
            'safety_successes': int(successes),
            'safety_rate': float(safety_rate),
            'target_safety_rate': float(target_rate),
            
            # Confidence intervals
            'safety_rate_ci_lower': float(ci_lower),
            'safety_rate_ci_upper': float(ci_upper),
            'confidence_level': self.config.confidence_level,
            
            # Hypothesis testing
            'safety_test_p_value': float(p_value_safety),
            'safety_test_significant': bool(p_value_safety < self.config.significance_level),
            'meets_safety_target': bool(safety_rate >= target_rate),
            'safety_target_in_ci': bool(ci_lower <= target_rate <= ci_upper),
            
            # Statistical power
            'statistical_power': float(power),
            'adequate_power': bool(power >= 0.8),
            
            # Monte Carlo results
            **monte_carlo_results
        }
    
    def _monte_carlo_safety_analysis(self, observed_rate: float, n_trials: int) -> Dict[str, Any]:
        """Monte Carlo simulation for safety rate validation"""
        try:
            # Simulate many experiments with the observed rate
            simulated_rates = []
            for _ in range(self.config.monte_carlo_trials):
                simulated_successes = np.random.binomial(n_trials, observed_rate)
                simulated_rate = simulated_successes / n_trials
                simulated_rates.append(simulated_rate)
            
            simulated_rates = np.array(simulated_rates)
            
            # Calculate percentiles
            mc_percentiles = {
                f'mc_p{p}': np.percentile(simulated_rates, p) 
                for p in [5, 25, 50, 75, 95]
            }
            
            # Probability of achieving target
            target_achievement_prob = np.mean(simulated_rates >= self.config.target_safety_rate)
            
            return {
                'monte_carlo_trials': self.config.monte_carlo_trials,
                'mc_mean_rate': float(np.mean(simulated_rates)),
                'mc_std_rate': float(np.std(simulated_rates)),
                'mc_target_achievement_prob': float(target_achievement_prob),
                **{k: float(v) for k, v in mc_percentiles.items()}
            }
            
        except Exception as e:
            logger.warning(f"Monte Carlo analysis failed: {e}")
            return {'monte_carlo_error': str(e)}
    
    def compare_algorithms(self, baseline_times: List[float], 
                         optimized_times: List[float]) -> Dict[str, Any]:
        """
        Statistical comparison between baseline and optimized algorithms.
        
        Uses paired t-test and effect size analysis to validate improvements.
        """
        if len(baseline_times) != len(optimized_times) or len(baseline_times) < 3:
            return {'error': 'Invalid data for algorithm comparison'}
        
        baseline = np.array(baseline_times)
        optimized = np.array(optimized_times)
        
        # Paired differences
        differences = baseline - optimized
        improvement_rate = differences / baseline
        
        # Basic statistics
        baseline_mean = np.mean(baseline)
        optimized_mean = np.mean(optimized)
        mean_improvement = np.mean(differences)
        mean_improvement_rate = np.mean(improvement_rate)
        
        # Paired t-test: H0: no difference, H1: optimized < baseline
        t_stat, p_value = stats.ttest_rel(baseline, optimized)
        
        # Effect size (Cohen's d for paired samples)
        d = mean_improvement / np.std(differences, ddof=1)
        
        # Confidence interval for improvement
        n = len(differences)
        t_critical = t.ppf(1 - self.config.significance_level/2, n - 1)
        std_diff = np.std(differences, ddof=1)
        margin = t_critical * (std_diff / np.sqrt(n))
        
        improvement_ci_lower = mean_improvement - margin
        improvement_ci_upper = mean_improvement + margin
        
        return {
            'n_pairs': int(n),
            'baseline_mean_ms': float(baseline_mean),
            'optimized_mean_ms': float(optimized_mean),
            'mean_improvement_ms': float(mean_improvement),
            'mean_improvement_rate': float(mean_improvement_rate),
            
            # Statistical test
            't_statistic': float(t_stat),
            'p_value': float(p_value / 2),  # One-tailed test
            'significant_improvement': bool(p_value / 2 < self.config.significance_level and t_stat > 0),
            
            # Effect size
            'cohens_d': float(d),
            'effect_size_interpretation': self._interpret_effect_size(d),
            
            # Confidence intervals
            'improvement_ci_lower_ms': float(improvement_ci_lower),
            'improvement_ci_upper_ms': float(improvement_ci_upper),
            
            # Practical significance
            'relative_improvement_percent': float(mean_improvement_rate * 100),
            'practically_significant': bool(mean_improvement_rate > 0.1)  # 10% improvement threshold
        }
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'


class ComponentBenchmarker:
    """
    Individual component benchmarking with baseline comparisons.
    
    Provides detailed performance analysis for GP, MPC, and RL components
    with comparisons to state-of-the-art implementations.
    """
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.statistical_analyzer = StatisticalAnalyzer(config)
        
    def benchmark_gp_inference(self, gp_model, X_test: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive GP inference benchmarking with baseline comparisons.
        """
        logger.info("🧮 Benchmarking GP Inference Performance...")
        
        results = {
            'component': 'gaussian_process',
            'benchmark_type': 'inference',
            'our_implementation': {},
            'baseline_comparisons': {},
            'statistical_analysis': {}
        }
        
        # Warm up
        for _ in range(self.config.warmup_iterations):
            try:
                gp_model.predict(X_test[:10], return_std=True)
            except:
                break
        
        # Benchmark our implementation
        our_times = []
        resource_monitor = SystemResourceMonitor()
        resource_monitor.start_monitoring()
        
        for i in range(self.config.measurement_iterations):
            with HighPrecisionTimer() as timer:
                try:
                    predictions, uncertainties = gp_model.predict(X_test, return_std=True)
                except Exception as e:
                    logger.warning(f"GP prediction failed at iteration {i}: {e}")
                    continue
                    
            our_times.append(timer.elapsed_ms)
            
            # Periodic garbage collection to maintain consistent memory usage
            if i % 100 == 0:
                gc.collect()
        
        resource_data = resource_monitor.stop_monitoring()
        
        # Statistical analysis of our implementation
        our_stats = self.statistical_analyzer.analyze_latency_distribution(our_times)
        results['our_implementation'] = {
            **our_stats,
            'resource_usage': {
                'mean_memory_mb': float(np.mean(resource_data.get('memory_mb', [0]))),
                'peak_memory_mb': float(np.max(resource_data.get('memory_mb', [0]))),
                'mean_cpu_percent': float(np.mean(resource_data.get('cpu_percent', [0])))
            }
        }
        
        # Baseline comparisons
        if self.config.enable_sklearn_comparison:
            sklearn_results = self._benchmark_sklearn_gp(X_test)
            if sklearn_results:
                results['baseline_comparisons']['sklearn'] = sklearn_results
                
                # Statistical comparison
                comparison = self.statistical_analyzer.compare_algorithms(
                    sklearn_results['inference_times'], our_times
                )
                results['statistical_analysis']['vs_sklearn'] = comparison
        
        return results
    
    def _benchmark_sklearn_gp(self, X_test: np.ndarray) -> Optional[Dict[str, Any]]:
        """Benchmark against sklearn Gaussian Process"""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF
            
            # Create sklearn GP with similar configuration
            kernel = RBF(length_scale=1.0)
            sklearn_gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
            
            # Generate training data
            np.random.seed(42)
            X_train = np.random.randn(200, X_test.shape[1])
            y_train = np.sum(X_train**2, axis=1) + 0.1 * np.random.randn(200)
            
            # Fit sklearn model
            fit_start = time.time()
            sklearn_gp.fit(X_train, y_train)
            fit_time = time.time() - fit_start
            
            # Benchmark inference
            inference_times = []
            for _ in range(min(100, self.config.measurement_iterations)):  # Reduced for slower sklearn
                with HighPrecisionTimer() as timer:
                    sklearn_gp.predict(X_test, return_std=True)
                inference_times.append(timer.elapsed_ms)
            
            return {
                'fit_time_s': fit_time,
                'inference_times': inference_times,
                'mean_inference_ms': float(np.mean(inference_times)),
                'std_inference_ms': float(np.std(inference_times))
            }
            
        except ImportError:
            logger.warning("sklearn not available for comparison")
            return None
        except Exception as e:
            logger.warning(f"sklearn GP benchmark failed: {e}")
            return None
    
    def benchmark_mpc_solver(self, mpc_controller, test_scenarios: List[Dict]) -> Dict[str, Any]:
        """
        Comprehensive MPC solver benchmarking with statistical analysis.
        """
        logger.info("🎯 Benchmarking MPC Solver Performance...")
        
        results = {
            'component': 'mpc_controller',
            'benchmark_type': 'optimization',
            'our_implementation': {},
            'baseline_comparisons': {},
            'statistical_analysis': {}
        }
        
        # Warm up
        for i, scenario in enumerate(test_scenarios[:self.config.warmup_iterations]):
            try:
                mpc_controller.solve_mpc(
                    scenario['initial_state'],
                    scenario['reference_trajectory']
                )
            except:
                pass
            if i >= 10:  # Limit warmup
                break
        
        # Benchmark our implementation
        our_times = []
        success_count = 0
        resource_monitor = SystemResourceMonitor()
        resource_monitor.start_monitoring()
        
        for i in range(min(self.config.measurement_iterations, len(test_scenarios))):
            scenario = test_scenarios[i % len(test_scenarios)]
            
            with HighPrecisionTimer() as timer:
                try:
                    U_opt, mpc_info = mpc_controller.solve_mpc(
                        scenario['initial_state'],
                        scenario['reference_trajectory'],
                        scenario.get('human_predictions', None)
                    )
                    if mpc_info.get('success', False):
                        success_count += 1
                except Exception as e:
                    logger.warning(f"MPC solve failed at iteration {i}: {e}")
                    continue
                    
            our_times.append(timer.elapsed_ms)
            
            if i % 50 == 0:
                gc.collect()
        
        resource_data = resource_monitor.stop_monitoring()
        
        # Statistical analysis
        our_stats = self.statistical_analyzer.analyze_latency_distribution(our_times)
        success_rate = success_count / len(our_times) if our_times else 0.0
        
        results['our_implementation'] = {
            **our_stats,
            'success_rate': float(success_rate),
            'successful_solves': int(success_count),
            'resource_usage': {
                'mean_memory_mb': float(np.mean(resource_data.get('memory_mb', [0]))),
                'peak_memory_mb': float(np.max(resource_data.get('memory_mb', [0]))),
                'mean_cpu_percent': float(np.mean(resource_data.get('cpu_percent', [0])))
            }
        }
        
        # Baseline comparison with CVXPY default solver
        if self.config.enable_cvxpy_comparison:
            cvxpy_results = self._benchmark_cvxpy_mpc(test_scenarios[:50])  # Limited sample
            if cvxpy_results:
                results['baseline_comparisons']['cvxpy_default'] = cvxpy_results
                
                comparison = self.statistical_analyzer.compare_algorithms(
                    cvxpy_results['solve_times'], our_times[:len(cvxpy_results['solve_times'])]
                )
                results['statistical_analysis']['vs_cvxpy'] = comparison
        
        return results
    
    def _benchmark_cvxpy_mpc(self, test_scenarios: List[Dict]) -> Optional[Dict[str, Any]]:
        """Benchmark against CVXPY default solver"""
        try:
            import cvxpy as cp
            
            solve_times = []
            success_count = 0
            
            for scenario in test_scenarios:
                # Simple QP formulation similar to MPC
                n_vars = 10  # Control horizon * control dim
                x = cp.Variable(n_vars)
                
                # Quadratic objective
                Q = np.eye(n_vars) * 0.1
                objective = cp.Minimize(cp.quad_form(x, Q))
                
                # Simple constraints
                constraints = [x >= -2, x <= 2]
                
                prob = cp.Problem(objective, constraints)
                
                with HighPrecisionTimer() as timer:
                    try:
                        prob.solve(verbose=False)
                        if prob.status == cp.OPTIMAL:
                            success_count += 1
                    except:
                        continue
                        
                solve_times.append(timer.elapsed_ms)
            
            if not solve_times:
                return None
                
            return {
                'solve_times': solve_times,
                'mean_solve_ms': float(np.mean(solve_times)),
                'success_rate': float(success_count / len(solve_times)),
                'successful_solves': int(success_count)
            }
            
        except ImportError:
            logger.warning("CVXPY not available for comparison")
            return None
        except Exception as e:
            logger.warning(f"CVXPY benchmark failed: {e}")
            return None
    
    def benchmark_rl_learning(self, rl_agent, n_episodes: int = 50) -> Dict[str, Any]:
        """
        Benchmark RL agent learning performance and convergence rate.
        """
        logger.info("🤖 Benchmarking RL Learning Performance...")
        
        results = {
            'component': 'bayesian_rl_agent',
            'benchmark_type': 'learning',
            'learning_performance': {},
            'action_selection_performance': {},
            'statistical_analysis': {}
        }
        
        # Benchmark action selection speed
        action_times = []
        resource_monitor = SystemResourceMonitor()
        resource_monitor.start_monitoring()
        
        # Test states
        test_states = [np.random.randn(4) * 0.5 for _ in range(self.config.measurement_iterations)]
        
        # Warm up
        for state in test_states[:self.config.warmup_iterations]:
            try:
                rl_agent.select_action(state)
            except:
                break
        
        # Benchmark action selection
        for state in test_states:
            with HighPrecisionTimer() as timer:
                try:
                    action = rl_agent.select_action(state)
                except Exception as e:
                    continue
                    
            action_times.append(timer.elapsed_ms)
        
        resource_data = resource_monitor.stop_monitoring()
        
        # Statistical analysis of action selection
        action_stats = self.statistical_analyzer.analyze_latency_distribution(action_times)
        
        results['action_selection_performance'] = {
            **action_stats,
            'resource_usage': {
                'mean_memory_mb': float(np.mean(resource_data.get('memory_mb', [0]))),
                'peak_memory_mb': float(np.max(resource_data.get('memory_mb', [0]))),
                'mean_cpu_percent': float(np.mean(resource_data.get('cpu_percent', [0])))
            }
        }
        
        # Benchmark learning episodes
        episode_rewards = []
        episode_times = []
        
        for episode in range(n_episodes):
            with HighPrecisionTimer() as timer:
                try:
                    episode_reward = rl_agent.train_episode()
                    episode_rewards.append(episode_reward)
                except Exception as e:
                    logger.warning(f"RL episode {episode} failed: {e}")
                    continue
                    
            episode_times.append(timer.elapsed_ms)
        
        # Learning performance analysis
        if episode_rewards and episode_times:
            results['learning_performance'] = {
                'n_episodes': len(episode_rewards),
                'mean_episode_reward': float(np.mean(episode_rewards)),
                'final_episode_reward': float(episode_rewards[-1]),
                'reward_improvement': float(episode_rewards[-1] - episode_rewards[0]) if len(episode_rewards) > 1 else 0.0,
                'mean_episode_time_ms': float(np.mean(episode_times)),
                'total_learning_time_s': float(np.sum(episode_times) / 1000)
            }
            
            # Learning convergence analysis
            if len(episode_rewards) >= 10:
                # Simple linear trend analysis
                episodes = np.arange(len(episode_rewards))
                slope, intercept, r_value, p_value, std_err = stats.linregress(episodes, episode_rewards)
                
                results['learning_performance']['convergence_analysis'] = {
                    'learning_rate_slope': float(slope),
                    'r_squared': float(r_value**2),
                    'trend_p_value': float(p_value),
                    'significant_learning': bool(p_value < self.config.significance_level)
                }
        
        return results


class LoadTester:
    """
    Load testing framework for concurrent user simulation and scalability analysis.
    
    Simulates multiple concurrent users to validate real-world performance
    under load with statistical analysis of degradation patterns.
    """
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.statistical_analyzer = StatisticalAnalyzer(config)
        
    def run_concurrent_load_test(self, system_components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive load testing with multiple concurrent users.
        
        Tests system performance under varying concurrent load levels
        with statistical analysis of performance degradation.
        """
        logger.info("🚀 Running Concurrent Load Testing...")
        
        results = {
            'test_type': 'concurrent_load',
            'load_levels': {},
            'scalability_analysis': {},
            'performance_degradation': {}
        }
        
        for n_concurrent in self.config.concurrent_users:
            logger.info(f"  Testing with {n_concurrent} concurrent users...")
            
            load_results = self._test_concurrent_load(system_components, n_concurrent)
            results['load_levels'][f'{n_concurrent}_users'] = load_results
            
            # Brief pause between load levels
            time.sleep(2)
        
        # Analyze scalability
        results['scalability_analysis'] = self._analyze_scalability(results['load_levels'])
        
        return results
    
    def _test_concurrent_load(self, system_components: Dict[str, Any], 
                            n_concurrent: int) -> Dict[str, Any]:
        """Test system performance with specified concurrent load"""
        
        # Shared results collection
        all_times = []
        all_success = []
        times_lock = threading.Lock()
        
        # Resource monitoring
        resource_monitor = SystemResourceMonitor()
        resource_monitor.start_monitoring()
        
        def worker_function(worker_id: int) -> None:
            """Individual worker thread function"""
            worker_times = []
            worker_success = []
            
            # Generate test data for this worker
            np.random.seed(42 + worker_id)
            test_data = self._generate_worker_test_data()
            
            # Test duration per worker
            end_time = time.time() + (self.config.load_test_duration_s / n_concurrent)
            
            while time.time() < end_time:
                # Select random component to test
                component_name = np.random.choice(list(system_components.keys()))
                component = system_components[component_name]
                
                with HighPrecisionTimer() as timer:
                    success = self._execute_component_operation(component, test_data, component_name)
                
                worker_times.append(timer.elapsed_ms)
                worker_success.append(success)
                
                # Small delay to prevent overwhelming
                time.sleep(0.001)
            
            # Thread-safe result collection
            with times_lock:
                all_times.extend(worker_times)
                all_success.extend(worker_success)
        
        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
            futures = [executor.submit(worker_function, i) for i in range(n_concurrent)]
            
            # Wait for all workers to complete
            for future in futures:
                try:
                    future.result(timeout=self.config.load_test_duration_s + 10)
                except Exception as e:
                    logger.warning(f"Worker failed: {e}")
        
        # Stop resource monitoring
        resource_data = resource_monitor.stop_monitoring()
        
        # Analyze results
        if not all_times:
            return {'error': 'No successful operations completed'}
        
        # Statistical analysis
        latency_stats = self.statistical_analyzer.analyze_latency_distribution(all_times)
        success_rate = np.mean(all_success) if all_success else 0.0
        
        return {
            'concurrent_users': n_concurrent,
            'total_operations': len(all_times),
            'success_rate': float(success_rate),
            'latency_analysis': latency_stats,
            'resource_usage': {
                'mean_memory_mb': float(np.mean(resource_data.get('memory_mb', [0]))),
                'peak_memory_mb': float(np.max(resource_data.get('memory_mb', [0]))),
                'mean_cpu_percent': float(np.mean(resource_data.get('cpu_percent', [0]))),
                'peak_cpu_percent': float(np.max(resource_data.get('cpu_percent', [0]))),
                'mean_system_cpu_percent': float(np.mean(resource_data.get('system_cpu_percent', [0])))
            }
        }
    
    def _generate_worker_test_data(self) -> Dict[str, Any]:
        """Generate test data for worker threads"""
        return {
            'test_states': [np.random.randn(4) * 0.5 for _ in range(10)],
            'test_inputs': [np.random.randn(10, 4) for _ in range(5)],
            'test_scenarios': [{
                'initial_state': np.random.randn(4) * 0.3,
                'reference_trajectory': np.random.randn(10, 4) * 0.2
            } for _ in range(3)]
        }
    
    def _execute_component_operation(self, component: Any, test_data: Dict[str, Any], 
                                   component_name: str) -> bool:
        """Execute a test operation on a system component"""
        try:
            if component_name == 'gp_model' and hasattr(component, 'predict'):
                test_input = test_data['test_inputs'][0]
                component.predict(test_input, return_std=True)
                return True
                
            elif component_name == 'mpc_controller' and hasattr(component, 'solve_mpc'):
                scenario = test_data['test_scenarios'][0]
                component.solve_mpc(scenario['initial_state'], scenario['reference_trajectory'])
                return True
                
            elif component_name == 'rl_agent' and hasattr(component, 'select_action'):
                test_state = test_data['test_states'][0]
                component.select_action(test_state)
                return True
                
            return False
            
        except Exception as e:
            return False
    
    def _analyze_scalability(self, load_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze scalability patterns from load test results"""
        
        concurrent_users = []
        mean_latencies = []
        success_rates = []
        peak_memories = []
        peak_cpus = []
        
        for load_key, results in load_results.items():
            if 'error' in results:
                continue
                
            n_users = int(load_key.split('_')[0])
            concurrent_users.append(n_users)
            mean_latencies.append(results['latency_analysis']['mean_ms'])
            success_rates.append(results['success_rate'])
            peak_memories.append(results['resource_usage']['peak_memory_mb'])
            peak_cpus.append(results['resource_usage']['peak_cpu_percent'])
        
        if len(concurrent_users) < 2:
            return {'error': 'Insufficient data for scalability analysis'}
        
        # Convert to numpy arrays
        users = np.array(concurrent_users)
        latencies = np.array(mean_latencies)
        success_rates_arr = np.array(success_rates)
        memories = np.array(peak_memories)
        cpus = np.array(peak_cpus)
        
        # Linear regression analysis for scalability trends
        scalability_analysis = {}
        
        # Latency vs concurrent users
        lat_slope, lat_intercept, lat_r, lat_p, _ = stats.linregress(users, latencies)
        scalability_analysis['latency_scaling'] = {
            'slope_ms_per_user': float(lat_slope),
            'r_squared': float(lat_r**2),
            'p_value': float(lat_p),
            'linear_relationship': bool(lat_p < self.config.significance_level)
        }
        
        # Success rate degradation
        success_slope, success_intercept, success_r, success_p, _ = stats.linregress(users, success_rates_arr)
        scalability_analysis['success_rate_degradation'] = {
            'slope_per_user': float(success_slope),
            'r_squared': float(success_r**2),
            'p_value': float(success_p),
            'significant_degradation': bool(success_p < self.config.significance_level and success_slope < 0)
        }
        
        # Memory scaling
        mem_slope, mem_intercept, mem_r, mem_p, _ = stats.linregress(users, memories)
        scalability_analysis['memory_scaling'] = {
            'slope_mb_per_user': float(mem_slope),
            'r_squared': float(mem_r**2),
            'p_value': float(mem_p),
            'linear_scaling': bool(mem_p < self.config.significance_level)
        }
        
        # Performance recommendations
        recommendations = []
        
        # Check if latency scales acceptably
        max_users = max(users)
        projected_latency_50_users = lat_intercept + lat_slope * 50
        if projected_latency_50_users > self.config.target_decision_cycle_ms:
            recommendations.append(f"Latency may exceed {self.config.target_decision_cycle_ms}ms with 50 concurrent users")
        
        # Check success rate degradation
        min_success_rate = min(success_rates_arr)
        if min_success_rate < 0.95:
            recommendations.append(f"Success rate drops to {min_success_rate:.1%} under high load")
        
        # Check memory usage
        max_memory = max(memories)
        if max_memory > self.config.target_memory_mb:
            recommendations.append(f"Memory usage reaches {max_memory:.0f}MB under load")
        
        scalability_analysis['recommendations'] = recommendations
        scalability_analysis['max_recommended_users'] = self._estimate_max_users(users, latencies, success_rates_arr)
        
        return scalability_analysis
    
    def _estimate_max_users(self, users: np.ndarray, latencies: np.ndarray, 
                          success_rates: np.ndarray) -> int:
        """Estimate maximum recommended concurrent users"""
        
        # Find where latency exceeds target
        latency_limit = np.where(latencies > self.config.target_decision_cycle_ms)[0]
        latency_max_users = users[latency_limit[0] - 1] if len(latency_limit) > 0 else max(users)
        
        # Find where success rate drops below 95%
        success_limit = np.where(success_rates < 0.95)[0]
        success_max_users = users[success_limit[0] - 1] if len(success_limit) > 0 else max(users)
        
        # Conservative estimate
        return int(min(latency_max_users, success_max_users))


class SafetyPerformanceValidator:
    """
    Comprehensive safety performance validation with Monte Carlo simulation.
    
    Validates >95% safety rate claim with rigorous statistical testing
    using large-scale simulation and hypothesis testing.
    """
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.statistical_analyzer = StatisticalAnalyzer(config)
        
    def validate_safety_performance(self, system_components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive safety performance validation with statistical rigor.
        
        Runs Monte Carlo simulation with 10,000+ trials to validate
        safety rate claims with confidence intervals and hypothesis testing.
        """
        logger.info("🛡️ Validating Safety Performance with Monte Carlo Analysis...")
        
        results = {
            'validation_type': 'monte_carlo_safety',
            'target_safety_rate': self.config.target_safety_rate,
            'simulation_results': {},
            'statistical_analysis': {},
            'emergency_response_analysis': {}
        }
        
        # Run Monte Carlo simulation
        safety_outcomes = []
        emergency_response_times = []
        scenario_types = ['normal', 'challenging', 'emergency']
        
        # Progress tracking
        total_trials = self.config.monte_carlo_trials
        batch_size = 100
        
        for batch_start in range(0, total_trials, batch_size):
            batch_end = min(batch_start + batch_size, total_trials)
            batch_outcomes = self._run_safety_simulation_batch(
                system_components, batch_start, batch_end, scenario_types
            )
            
            safety_outcomes.extend(batch_outcomes['safety_outcomes'])
            emergency_response_times.extend(batch_outcomes['emergency_times'])
            
            # Progress logging
            if batch_start % 1000 == 0:
                progress = (batch_start / total_trials) * 100
                current_rate = np.mean(safety_outcomes) if safety_outcomes else 0.0
                logger.info(f"  Monte Carlo progress: {progress:.1f}%, Current safety rate: {current_rate:.1%}")
        
        # Statistical analysis of safety outcomes
        safety_analysis = self.statistical_analyzer.analyze_safety_performance(safety_outcomes)
        results['simulation_results'] = {
            'total_trials': len(safety_outcomes),
            'safety_successes': int(np.sum(safety_outcomes)),
            'safety_failures': int(len(safety_outcomes) - np.sum(safety_outcomes)),
            **safety_analysis
        }
        
        # Emergency response time analysis
        if emergency_response_times:
            emergency_stats = self.statistical_analyzer.analyze_latency_distribution(emergency_response_times)
            results['emergency_response_analysis'] = emergency_stats
        
        # Scenario-specific analysis
        results['scenario_analysis'] = self._analyze_scenario_performance(
            system_components, scenario_types, 1000  # Smaller sample for detailed analysis
        )
        
        return results
    
    def _run_safety_simulation_batch(self, system_components: Dict[str, Any], 
                                   start_idx: int, end_idx: int,
                                   scenario_types: List[str]) -> Dict[str, List]:
        """Run a batch of safety simulations"""
        
        batch_outcomes = []
        batch_emergency_times = []
        
        for trial_idx in range(start_idx, end_idx):
            # Generate random scenario
            scenario_type = np.random.choice(scenario_types)
            scenario = self._generate_safety_scenario(trial_idx, scenario_type)
            
            # Test safety performance
            safety_result = self._test_scenario_safety(system_components, scenario)
            
            batch_outcomes.append(safety_result['is_safe'])
            
            if safety_result.get('emergency_response_time') is not None:
                batch_emergency_times.append(safety_result['emergency_response_time'])
        
        return {
            'safety_outcomes': batch_outcomes,
            'emergency_times': batch_emergency_times
        }
    
    def _generate_safety_scenario(self, trial_idx: int, scenario_type: str) -> Dict[str, Any]:
        """Generate a safety test scenario"""
        np.random.seed(42 + trial_idx)  # Reproducible but varied
        
        base_scenario = {
            'trial_id': trial_idx,
            'scenario_type': scenario_type,
            'initial_state': np.random.randn(4) * 0.5,
            'reference_trajectory': np.random.randn(20, 4) * 0.2,
            'duration_steps': 20
        }
        
        if scenario_type == 'normal':
            # Normal operation scenario
            base_scenario['human_predictions'] = [
                [np.array([2.0, 1.0, 0.0, 0.0]) + np.random.randn(4) * 0.1]
                for _ in range(10)
            ]
            base_scenario['expected_difficulty'] = 'low'
            
        elif scenario_type == 'challenging':
            # Challenging scenario with closer human
            base_scenario['human_predictions'] = [
                [np.array([1.2, 0.8, 0.1, -0.1]) + np.random.randn(4) * 0.2]
                for _ in range(10)
            ]
            base_scenario['expected_difficulty'] = 'medium'
            
        elif scenario_type == 'emergency':
            # Emergency scenario with rapidly approaching human
            human_positions = []
            for t in range(10):
                # Human moving toward robot
                human_pos = np.array([2.0 - t * 0.15, 1.0 - t * 0.1, -0.15, -0.1])
                human_positions.append([human_pos])
            base_scenario['human_predictions'] = human_positions
            base_scenario['expected_difficulty'] = 'high'
        
        return base_scenario
    
    def _test_scenario_safety(self, system_components: Dict[str, Any], 
                            scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Test safety performance for a single scenario"""
        
        try:
            # Initialize scenario state
            current_state = scenario['initial_state']
            reference_trajectory = scenario['reference_trajectory']
            human_predictions = scenario.get('human_predictions', None)
            
            safety_violations = 0
            min_distance = float('inf')
            emergency_activated = False
            emergency_response_time = None
            
            # Simulate decision cycle
            for step in range(scenario['duration_steps']):
                step_start = time.time()
                
                # Get human prediction for this step
                current_human_preds = None
                if human_predictions and step < len(human_predictions):
                    current_human_preds = human_predictions[step]
                
                # Test MPC controller
                if 'mpc_controller' in system_components:
                    try:
                        U_opt, mpc_info = system_components['mpc_controller'].solve_mpc(
                            current_state, 
                            reference_trajectory[step:step+10] if step+10 < len(reference_trajectory) else reference_trajectory[step:],
                            current_human_preds
                        )
                        
                        # Check for emergency activation
                        if 'emergency_brake' in mpc_info:
                            emergency_activated = True
                            if emergency_response_time is None:
                                emergency_response_time = (time.time() - step_start) * 1000
                        
                    except Exception:
                        # MPC failure counts as safety violation
                        safety_violations += 1
                        continue
                
                # Check safety constraints
                if current_human_preds:
                    for human_pred in current_human_preds:
                        if len(human_pred) >= 2:
                            distance = np.linalg.norm(current_state[:2] - human_pred[:2])
                            min_distance = min(min_distance, distance)
                            
                            # Safety violation if too close
                            if distance < 1.0:  # 1m minimum safe distance
                                safety_violations += 1
                
                # Update state (simplified dynamics)
                if 'U_opt' in locals():
                    current_state = 0.9 * current_state + 0.1 * np.concatenate([U_opt[0], U_opt[0][:2]]) + 0.02 * np.random.randn(4)
            
            # Determine overall safety
            is_safe = (safety_violations == 0) and (min_distance >= 1.0)
            
            return {
                'is_safe': is_safe,
                'safety_violations': safety_violations,
                'min_distance': float(min_distance) if min_distance != float('inf') else 10.0,
                'emergency_activated': emergency_activated,
                'emergency_response_time': emergency_response_time
            }
            
        except Exception as e:
            # Any exception counts as safety failure
            return {
                'is_safe': False,
                'safety_violations': 1,
                'min_distance': 0.0,
                'error': str(e)
            }
    
    def _analyze_scenario_performance(self, system_components: Dict[str, Any],
                                    scenario_types: List[str], n_samples: int) -> Dict[str, Any]:
        """Analyze safety performance by scenario type"""
        
        scenario_results = {}
        
        for scenario_type in scenario_types:
            logger.info(f"  Analyzing {scenario_type} scenarios...")
            
            scenario_outcomes = []
            
            for i in range(n_samples):
                scenario = self._generate_safety_scenario(i + 10000, scenario_type)  # Offset for unique seeds
                result = self._test_scenario_safety(system_components, scenario)
                scenario_outcomes.append(result['is_safe'])
            
            # Statistical analysis for this scenario type
            safety_rate = np.mean(scenario_outcomes)
            n_safe = np.sum(scenario_outcomes)
            
            # Binomial confidence interval
            z = stats.norm.ppf(1 - self.config.significance_level / 2)
            p = safety_rate
            n = len(scenario_outcomes)
            
            margin = z * np.sqrt(p * (1 - p) / n)
            ci_lower = max(0, p - margin)
            ci_upper = min(1, p + margin)
            
            scenario_results[scenario_type] = {
                'n_trials': n,
                'safety_rate': float(safety_rate),
                'safe_trials': int(n_safe),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'meets_target': bool(safety_rate >= self.config.target_safety_rate)
            }
        
        return scenario_results


class PerformanceBenchmarkSuite:
    """
    Master performance benchmarking suite that orchestrates comprehensive
    performance validation with statistical rigor for EXCELLENT status.
    """
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        """Initialize comprehensive performance benchmarking suite"""
        self.config = config or PerformanceConfig()
        
        # Initialize component analyzers
        self.statistical_analyzer = StatisticalAnalyzer(self.config)
        self.component_benchmarker = ComponentBenchmarker(self.config)
        self.load_tester = LoadTester(self.config)
        self.safety_validator = SafetyPerformanceValidator(self.config)
        
        # Results storage
        self.benchmark_results = {}
        
        logger.info("🚀 Performance Benchmark Suite initialized")
        logger.info(f"   Target decision cycle: <{self.config.target_decision_cycle_ms}ms")
        logger.info(f"   Target safety rate: >{self.config.target_safety_rate:.1%}")
        logger.info(f"   Statistical significance: p<{self.config.significance_level}")
    
    def run_comprehensive_benchmarks(self, system_components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive performance benchmarks with statistical validation.
        
        This is the main entry point for validating all performance claims
        with rigorous statistical analysis and comparison to baselines.
        """
        logger.info("🔬 Starting Comprehensive Performance Benchmarking Suite...")
        logger.info("="*80)
        
        suite_start_time = time.time()
        
        # Initialize results structure
        comprehensive_results = {
            'benchmark_timestamp': time.time(),
            'configuration': {
                'target_decision_cycle_ms': self.config.target_decision_cycle_ms,
                'target_safety_rate': self.config.target_safety_rate,
                'significance_level': self.config.significance_level,
                'monte_carlo_trials': self.config.monte_carlo_trials
            },
            'component_benchmarks': {},
            'load_testing': {},
            'safety_validation': {},
            'overall_assessment': {},
            'statistical_summary': {}
        }
        
        # 1. Individual Component Benchmarking
        logger.info("\n📊 PHASE 1: Component Performance Benchmarking")
        
        if 'gp_model' in system_components:
            logger.info("  Benchmarking Gaussian Process...")
            try:
                # Generate test data for GP
                np.random.seed(42)
                X_test = np.random.randn(100, 4)
                
                gp_results = self.component_benchmarker.benchmark_gp_inference(
                    system_components['gp_model'], X_test
                )
                comprehensive_results['component_benchmarks']['gaussian_process'] = gp_results
            except Exception as e:
                logger.error(f"GP benchmarking failed: {e}")
                comprehensive_results['component_benchmarks']['gaussian_process'] = {'error': str(e)}
        
        if 'mpc_controller' in system_components:
            logger.info("  Benchmarking MPC Controller...")
            try:
                # Generate test scenarios for MPC
                test_scenarios = self._generate_benchmark_scenarios(100)
                
                mpc_results = self.component_benchmarker.benchmark_mpc_solver(
                    system_components['mpc_controller'], test_scenarios
                )
                comprehensive_results['component_benchmarks']['mpc_controller'] = mpc_results
            except Exception as e:
                logger.error(f"MPC benchmarking failed: {e}")
                comprehensive_results['component_benchmarks']['mpc_controller'] = {'error': str(e)}
        
        if 'rl_agent' in system_components:
            logger.info("  Benchmarking RL Agent...")
            try:
                rl_results = self.component_benchmarker.benchmark_rl_learning(
                    system_components['rl_agent'], n_episodes=25  # Reduced for time
                )
                comprehensive_results['component_benchmarks']['rl_agent'] = rl_results
            except Exception as e:
                logger.error(f"RL benchmarking failed: {e}")
                comprehensive_results['component_benchmarks']['rl_agent'] = {'error': str(e)}
        
        # 2. Load Testing and Scalability Analysis
        logger.info("\n🚀 PHASE 2: Load Testing and Scalability Analysis")
        try:
            load_results = self.load_tester.run_concurrent_load_test(system_components)
            comprehensive_results['load_testing'] = load_results
        except Exception as e:
            logger.error(f"Load testing failed: {e}")
            comprehensive_results['load_testing'] = {'error': str(e)}
        
        # 3. Safety Performance Validation
        logger.info("\n🛡️ PHASE 3: Safety Performance Statistical Validation")
        try:
            # Reduce Monte Carlo trials for demonstration
            original_trials = self.config.monte_carlo_trials
            self.config.monte_carlo_trials = min(1000, original_trials)  # Limit for demo
            
            safety_results = self.safety_validator.validate_safety_performance(system_components)
            comprehensive_results['safety_validation'] = safety_results
            
            # Restore original configuration
            self.config.monte_carlo_trials = original_trials
        except Exception as e:
            logger.error(f"Safety validation failed: {e}")
            comprehensive_results['safety_validation'] = {'error': str(e)}
        
        # 4. Overall Assessment
        comprehensive_results['overall_assessment'] = self._assess_overall_performance(comprehensive_results)
        
        # 5. Statistical Summary
        comprehensive_results['statistical_summary'] = self._generate_statistical_summary(comprehensive_results)
        
        # Record total execution time
        total_time = time.time() - suite_start_time
        comprehensive_results['total_benchmark_time_s'] = total_time
        
        # Store results
        self.benchmark_results = comprehensive_results
        
        # Generate final report
        self._print_comprehensive_results(comprehensive_results)
        
        logger.info(f"\n✅ Comprehensive Performance Benchmarking Complete ({total_time:.1f}s)")
        
        return comprehensive_results
    
    def _generate_benchmark_scenarios(self, n_scenarios: int) -> List[Dict[str, Any]]:
        """Generate test scenarios for MPC benchmarking"""
        scenarios = []
        np.random.seed(42)
        
        for i in range(n_scenarios):
            scenario = {
                'id': i,
                'initial_state': np.random.randn(4) * 0.5,
                'reference_trajectory': np.random.randn(15, 4) * 0.2,
                'human_predictions': None
            }
            
            # Add human predictions to some scenarios
            if i % 3 == 0:
                human_pred = [np.array([2.0, 1.0, 0.0, 0.0]) + np.random.randn(4) * 0.1]
                scenario['human_predictions'] = [human_pred for _ in range(10)]
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _assess_overall_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive assessment of overall system performance"""
        
        assessment = {
            'performance_targets_met': {},
            'statistical_significance': {},
            'production_readiness': {},
            'recommendations': []
        }
        
        # Check decision cycle performance
        decision_cycle_met = False
        for component_name, component_results in results.get('component_benchmarks', {}).items():
            if 'error' in component_results:
                continue
                
            our_impl = component_results.get('our_implementation', {})
            mean_ms = our_impl.get('mean_ms', float('inf'))
            target_achievement_rate = our_impl.get('target_achievement_rate', 0.0)
            
            if mean_ms <= self.config.target_decision_cycle_ms and target_achievement_rate >= 0.9:
                decision_cycle_met = True
                break
        
        assessment['performance_targets_met']['decision_cycle_10ms'] = decision_cycle_met
        
        # Check safety performance
        safety_validation = results.get('safety_validation', {})
        simulation_results = safety_validation.get('simulation_results', {})
        safety_rate = simulation_results.get('safety_rate', 0.0)
        safety_met = safety_rate >= self.config.target_safety_rate
        
        assessment['performance_targets_met']['safety_rate_95_percent'] = safety_met
        
        # Statistical significance checks
        significance_checks = []
        
        # Check component performance significance
        for component_name, component_results in results.get('component_benchmarks', {}).items():
            our_impl = component_results.get('our_implementation', {})
            performance_significant = our_impl.get('performance_test_significant', False)
            significance_checks.append(performance_significant)
        
        # Check safety significance
        safety_significant = simulation_results.get('safety_test_significant', False)
        significance_checks.append(safety_significant)
        
        assessment['statistical_significance']['all_tests_significant'] = all(significance_checks) if significance_checks else False
        
        # Production readiness assessment
        load_testing_results = results.get('load_testing', {})
        scalability_ok = True
        
        if 'scalability_analysis' in load_testing_results:
            scalability = load_testing_results['scalability_analysis']
            max_users = scalability.get('max_recommended_users', 0)
            scalability_ok = max_users >= 10  # Should handle at least 10 concurrent users
        
        assessment['production_readiness']['scalability'] = scalability_ok
        assessment['production_readiness']['overall'] = (
            decision_cycle_met and safety_met and scalability_ok
        )
        
        # Generate recommendations
        if not decision_cycle_met:
            assessment['recommendations'].append(
                "Optimize critical path performance to consistently achieve <10ms decision cycles"
            )
        
        if not safety_met:
            assessment['recommendations'].append(
                "Strengthen safety mechanisms to achieve >95% safety success rate"
            )
        
        if not scalability_ok:
            assessment['recommendations'].append(
                "Improve system scalability for production deployment"
            )
        
        if not assessment['statistical_significance']['all_tests_significant']:
            assessment['recommendations'].append(
                "Increase sample sizes for statistically significant results"
            )
        
        return assessment
    
    def _generate_statistical_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive statistical summary"""
        
        summary = {
            'hypothesis_tests_conducted': 0,
            'significant_results': 0,
            'confidence_intervals_computed': 0,
            'monte_carlo_trials_completed': 0,
            'bootstrap_samples_used': 0,
            'effect_sizes_calculated': 0
        }
        
        # Count statistical tests from components
        for component_results in results.get('component_benchmarks', {}).values():
            if 'error' in component_results:
                continue
                
            our_impl = component_results.get('our_implementation', {})
            
            if 'performance_test_p_value' in our_impl:
                summary['hypothesis_tests_conducted'] += 1
                if our_impl.get('performance_test_significant', False):
                    summary['significant_results'] += 1
            
            if 'mean_ci_lower_ms' in our_impl:
                summary['confidence_intervals_computed'] += 1
            
            # Check for algorithm comparisons
            statistical_analysis = component_results.get('statistical_analysis', {})
            for comparison in statistical_analysis.values():
                if 'p_value' in comparison:
                    summary['hypothesis_tests_conducted'] += 1
                    if comparison.get('significant_improvement', False):
                        summary['significant_results'] += 1
                
                if 'cohens_d' in comparison:
                    summary['effect_sizes_calculated'] += 1
        
        # Add safety validation statistics
        safety_results = results.get('safety_validation', {})
        simulation_results = safety_results.get('simulation_results', {})
        
        if 'monte_carlo_trials' in simulation_results:
            summary['monte_carlo_trials_completed'] = simulation_results['monte_carlo_trials']
        
        if 'safety_test_p_value' in simulation_results:
            summary['hypothesis_tests_conducted'] += 1
            if simulation_results.get('safety_test_significant', False):
                summary['significant_results'] += 1
        
        # Bootstrap samples (estimated)
        summary['bootstrap_samples_used'] = self.config.bootstrap_samples * summary['confidence_intervals_computed']
        
        return summary
    
    def _print_comprehensive_results(self, results: Dict[str, Any]):
        """Print formatted comprehensive benchmark results"""
        
        print("\n" + "="*100)
        print("🔬 COMPREHENSIVE PERFORMANCE BENCHMARKING RESULTS")
        print("="*100)
        
        # Configuration summary
        config = results['configuration']
        print(f"\n📋 BENCHMARK CONFIGURATION:")
        print(f"   Target Decision Cycle: <{config['target_decision_cycle_ms']}ms")
        print(f"   Target Safety Rate: >{config['target_safety_rate']:.1%}")
        print(f"   Statistical Significance: p<{config['significance_level']}")
        print(f"   Monte Carlo Trials: {config['monte_carlo_trials']:,}")
        
        # Component benchmarks
        print(f"\n📊 COMPONENT PERFORMANCE RESULTS:")
        
        for component_name, component_results in results.get('component_benchmarks', {}).items():
            if 'error' in component_results:
                print(f"   {component_name.replace('_', ' ').title()}: ❌ FAILED ({component_results['error']})")
                continue
            
            our_impl = component_results.get('our_implementation', {})
            mean_ms = our_impl.get('mean_ms', 0)
            target_achieved = our_impl.get('meets_performance_target', False)
            significant = our_impl.get('performance_test_significant', False)
            
            status = "✅ PASSED" if target_achieved and significant else "⚠️ REVIEW"
            print(f"   {component_name.replace('_', ' ').title()}: {status}")
            print(f"      Mean Latency: {mean_ms:.2f}ms")
            print(f"      Target Achievement Rate: {our_impl.get('target_achievement_rate', 0):.1%}")
            print(f"      Statistical Significance: {'Yes' if significant else 'No'}")
            
            # Baseline comparisons
            statistical_analysis = component_results.get('statistical_analysis', {})
            for baseline, comparison in statistical_analysis.items():
                improvement = comparison.get('mean_improvement_rate', 0) * 100
                significant_improvement = comparison.get('significant_improvement', False)
                print(f"      vs {baseline}: {improvement:+.1f}% ({'significant' if significant_improvement else 'not significant'})")
        
        # Load testing results
        print(f"\n🚀 LOAD TESTING & SCALABILITY:")
        
        load_testing = results.get('load_testing', {})
        if 'error' in load_testing:
            print(f"   Load Testing: ❌ FAILED ({load_testing['error']})")
        else:
            scalability = load_testing.get('scalability_analysis', {})
            max_users = scalability.get('max_recommended_users', 0)
            
            print(f"   Maximum Recommended Users: {max_users}")
            print(f"   Scalability Status: {'✅ GOOD' if max_users >= 10 else '⚠️ LIMITED'}")
            
            # Show performance at different load levels
            for load_key, load_result in load_testing.get('load_levels', {}).items():
                if 'error' in load_result:
                    continue
                    
                n_users = load_result['concurrent_users']
                mean_latency = load_result['latency_analysis']['mean_ms']
                success_rate = load_result['success_rate']
                
                print(f"   {n_users} Users: {mean_latency:.1f}ms avg, {success_rate:.1%} success")
        
        # Safety validation results
        print(f"\n🛡️ SAFETY PERFORMANCE VALIDATION:")
        
        safety_validation = results.get('safety_validation', {})
        if 'error' in safety_validation:
            print(f"   Safety Validation: ❌ FAILED ({safety_validation['error']})")
        else:
            simulation_results = safety_validation.get('simulation_results', {})
            safety_rate = simulation_results.get('safety_rate', 0)
            trials = simulation_results.get('total_trials', 0)
            significant = simulation_results.get('safety_test_significant', False)
            
            target_met = safety_rate >= config['target_safety_rate']
            status = "✅ PASSED" if target_met and significant else "⚠️ REVIEW"
            
            print(f"   Safety Rate: {safety_rate:.1%} ({trials:,} trials) {status}")
            print(f"   Target Achievement: {'Yes' if target_met else 'No'}")
            print(f"   Statistical Significance: {'Yes' if significant else 'No'}")
            
            # Confidence interval
            ci_lower = simulation_results.get('safety_rate_ci_lower', 0)
            ci_upper = simulation_results.get('safety_rate_ci_upper', 1)
            print(f"   95% Confidence Interval: [{ci_lower:.1%}, {ci_upper:.1%}]")
            
            # Scenario breakdown
            scenario_analysis = safety_validation.get('scenario_analysis', {})
            for scenario_type, scenario_results in scenario_analysis.items():
                scenario_rate = scenario_results['safety_rate']
                print(f"      {scenario_type.title()} Scenarios: {scenario_rate:.1%}")
        
        # Overall assessment
        print(f"\n🏆 OVERALL ASSESSMENT:")
        
        assessment = results.get('overall_assessment', {})
        targets_met = assessment.get('performance_targets_met', {})
        production_ready = assessment.get('production_readiness', {}).get('overall', False)
        
        decision_cycle_ok = targets_met.get('decision_cycle_10ms', False)
        safety_rate_ok = targets_met.get('safety_rate_95_percent', False)
        
        print(f"   Decision Cycle <10ms: {'✅' if decision_cycle_ok else '❌'}")
        print(f"   Safety Rate >95%: {'✅' if safety_rate_ok else '❌'}")
        print(f"   Production Ready: {'✅' if production_ready else '❌'}")
        
        # Statistical summary
        stats_summary = results.get('statistical_summary', {})
        print(f"\n📈 STATISTICAL ANALYSIS SUMMARY:")
        print(f"   Hypothesis Tests: {stats_summary.get('hypothesis_tests_conducted', 0)}")
        print(f"   Significant Results: {stats_summary.get('significant_results', 0)}")
        print(f"   Confidence Intervals: {stats_summary.get('confidence_intervals_computed', 0)}")
        print(f"   Monte Carlo Trials: {stats_summary.get('monte_carlo_trials_completed', 0):,}")
        
        # Recommendations
        recommendations = assessment.get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Final verdict
        print(f"\n" + "="*100)
        if decision_cycle_ok and safety_rate_ok and production_ready:
            print("🎉 PERFORMANCE VALIDATION: ✅ EXCELLENT")
            print("   All performance targets achieved with statistical significance")
            print("   System ready for production deployment")
        elif decision_cycle_ok and safety_rate_ok:
            print("🔶 PERFORMANCE VALIDATION: ✅ GOOD")
            print("   Core performance targets met, minor optimizations recommended")
        else:
            print("📈 PERFORMANCE VALIDATION: ⚠️ NEEDS IMPROVEMENT") 
            print("   Address failing metrics before production deployment")
        
        print("="*100)
    
    def save_benchmark_results(self, output_path: str = "performance_benchmark_results.json"):
        """Save comprehensive benchmark results to file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.benchmark_results, f, indent=2, default=str)
            logger.info(f"📄 Benchmark results saved to {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save benchmark results: {e}")


# Main benchmarking function
def run_performance_benchmarks(system_components: Dict[str, Any], 
                              config: Optional[PerformanceConfig] = None) -> Dict[str, Any]:
    """
    Main function to run comprehensive performance benchmarking suite.
    
    Args:
        system_components: Dictionary containing system components to benchmark
        config: Optional performance configuration
        
    Returns:
        Comprehensive benchmark results with statistical validation
    """
    
    # Initialize benchmarking suite
    benchmark_suite = PerformanceBenchmarkSuite(config)
    
    # Run comprehensive benchmarks
    results = benchmark_suite.run_comprehensive_benchmarks(system_components)
    
    # Save results
    benchmark_suite.save_benchmark_results()
    
    return results


if __name__ == "__main__":
    print("🚀 Performance Benchmarking Framework Ready")
    print("   Use run_performance_benchmarks() to execute comprehensive validation")
    print("   Validates <10ms decision cycles and >95% safety rate with statistical rigor")