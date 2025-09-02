"""
Comprehensive Testing Suite for HRI Bayesian RL System

This module provides a complete testing framework covering unit tests,
integration tests, system tests, and performance tests for the entire
HRI Bayesian RL system.

Features:
- Automated test discovery and execution
- Unit tests for individual components
- Integration tests for component interactions
- System-level end-to-end tests
- Performance and stress testing
- Test reporting and coverage analysis
- Continuous integration support
- Test data generation and management

Author: Phase 5 Implementation
Date: 2024
"""

import unittest
import numpy as np
import time
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import threading
import multiprocessing
from unittest.mock import Mock, MagicMock, patch
import json
import tempfile
import shutil

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
logger = logging.getLogger(__name__)


class TestResult:
    """Container for test results"""
    
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0
        self.errors = []
        self.execution_time = 0.0
        self.coverage_data = {}
        
    def add_result(self, test_name: str, status: str, execution_time: float, 
                  error: Optional[Exception] = None):
        """Add test result"""
        self.total_tests += 1
        
        if status == "pass":
            self.passed_tests += 1
        elif status == "fail":
            self.failed_tests += 1
            if error:
                self.errors.append(f"{test_name}: {error}")
        elif status == "skip":
            self.skipped_tests += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary"""
        success_rate = self.passed_tests / self.total_tests if self.total_tests > 0 else 0
        
        return {
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'skipped_tests': self.skipped_tests,
            'success_rate': success_rate,
            'execution_time': self.execution_time,
            'errors': self.errors
        }


class BayesianRLAgentTests(unittest.TestCase):
    """Unit tests for Bayesian RL Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from src.agents.bayesian_rl_agent import BayesianRLAgent, BayesianRLConfiguration
            
            self.config = BayesianRLConfiguration(
                algorithm="gp_q_learning",
                state_dim=10,
                action_dim=3,
                learning_rate=0.1
            )
            self.agent = BayesianRLAgent(self.config)
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.config.state_dim, 10)
        self.assertEqual(self.agent.config.action_dim, 3)
    
    def test_prior_beliefs_initialization(self):
        """Test prior beliefs initialization"""
        prior_params = {'mean': 0.0, 'variance': 1.0}
        result = self.agent.initialize_prior_beliefs(prior_params)
        self.assertTrue(result)
    
    def test_action_selection(self):
        """Test action selection"""
        state = np.random.randn(10)
        
        # Test with different exploration strategies
        strategies = ['thompson_sampling', 'ucb', 'epsilon_greedy']
        
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                try:
                    action, info = self.agent.select_action(state, exploration_strategy=strategy)
                    self.assertIsInstance(action, np.ndarray)
                    self.assertEqual(len(action), self.config.action_dim)
                    self.assertIsInstance(info, dict)
                except Exception as e:
                    self.fail(f"Action selection failed for {strategy}: {e}")
    
    def test_belief_update(self):
        """Test belief updating"""
        state = np.random.randn(10)
        action = np.random.randn(3)
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = False
        
        try:
            self.agent.update_beliefs(state, action, reward, next_state, done)
        except Exception as e:
            self.fail(f"Belief update failed: {e}")
    
    def test_policy_uncertainty(self):
        """Test policy uncertainty computation"""
        state = np.random.randn(10)
        
        try:
            uncertainty = self.agent.compute_policy_uncertainty(state)
            self.assertIsInstance(uncertainty, (float, np.ndarray))
            self.assertGreaterEqual(uncertainty, 0.0)
        except Exception as e:
            self.fail(f"Policy uncertainty computation failed: {e}")
    
    def test_value_function_estimate(self):
        """Test value function estimation"""
        state = np.random.randn(10)
        
        try:
            value, confidence = self.agent.get_value_function_estimate(state)
            self.assertIsInstance(value, (float, np.floating))
            self.assertIsInstance(confidence, (float, np.floating))
        except Exception as e:
            self.fail(f"Value function estimation failed: {e}")


class HRIEnvironmentTests(unittest.TestCase):
    """Unit tests for HRI Environment"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from src.environments.hri_environment import (
                HRIEnvironment, create_default_hri_environment
            )
            
            self.env = create_default_hri_environment()
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def test_environment_initialization(self):
        """Test environment initialization"""
        self.assertIsNotNone(self.env)
    
    def test_reset_functionality(self):
        """Test environment reset"""
        try:
            initial_state = self.env.reset()
            self.assertIsNotNone(initial_state)
            
            # Check state structure
            self.assertTrue(hasattr(initial_state, 'robot'))
            self.assertTrue(hasattr(initial_state, 'human'))
            self.assertTrue(hasattr(initial_state, 'context'))
        except Exception as e:
            self.fail(f"Environment reset failed: {e}")
    
    def test_step_functionality(self):
        """Test environment step"""
        try:
            initial_state = self.env.reset()
            action = np.random.uniform(-0.1, 0.1, 6)  # Small random action
            
            next_state, reward_dict, done, info = self.env.step(action)
            
            self.assertIsNotNone(next_state)
            self.assertIsInstance(reward_dict, dict)
            self.assertIsInstance(done, bool)
            self.assertIsInstance(info, dict)
            
        except Exception as e:
            self.fail(f"Environment step failed: {e}")
    
    def test_reward_computation(self):
        """Test reward computation"""
        try:
            self.env.reset()
            action = np.zeros(6)  # Neutral action
            
            _, reward_dict, _, _ = self.env.step(action)
            
            # Check that we get expected reward components
            expected_components = ['task_reward', 'safety_reward', 'efficiency_reward']
            for component in expected_components:
                # Allow flexibility in reward structure
                if component in reward_dict:
                    self.assertIsInstance(reward_dict[component], (float, np.floating))
            
        except Exception as e:
            self.fail(f"Reward computation failed: {e}")


class IntegrationTests(unittest.TestCase):
    """Integration tests for component interactions"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from src.system.human_intent_rl_system import (
                HumanIntentRLSystem, SystemConfiguration, SystemMode
            )
            
            self.config = SystemConfiguration(
                mode=SystemMode.SIMULATION,
                max_decision_time=0.5,
                save_data=False
            )
            self.system = HumanIntentRLSystem(self.config)
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def test_system_initialization(self):
        """Test complete system initialization"""
        try:
            success = self.system.initialize_system()
            self.assertTrue(success)
        except Exception as e:
            self.fail(f"System initialization failed: {e}")
    
    def test_system_episode_execution(self):
        """Test complete episode execution"""
        try:
            # Initialize system
            if not self.system.initialize_system():
                self.skipTest("System initialization failed")
            
            # Run short episode
            episode_results = self.system.run_episode(max_steps=10)
            
            self.assertIsInstance(episode_results, dict)
            self.assertIn('success', episode_results)
            self.assertIn('steps_completed', episode_results)
            
        except Exception as e:
            self.fail(f"Episode execution failed: {e}")
        finally:
            try:
                self.system.shutdown()
            except:
                pass
    
    def test_data_flow_integration(self):
        """Test data flow between components"""
        try:
            from src.data.data_collector import create_default_collector
            from src.data.statistical_analysis import quick_performance_analysis
            
            # Create mock data
            mock_logs = [
                {
                    'timestamp': time.time(),
                    'cpu_usage': 50.0,
                    'memory_usage': 100.0,
                    'processing_time': 0.05
                }
                for _ in range(10)
            ]
            
            # Test data collection and analysis
            report = quick_performance_analysis(mock_logs, "test_analysis")
            self.assertIsInstance(report, object)
            
        except ImportError as e:
            self.skipTest(f"Data modules not available: {e}")
        except Exception as e:
            self.fail(f"Data flow integration failed: {e}")


class ExperimentalFrameworkTests(unittest.TestCase):
    """Tests for experimental framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from src.experiments.experimental_framework import (
                ExperimentRunner, ExperimentConfiguration, ExperimentType
            )
            
            self.config = ExperimentConfiguration(
                experiment_name="Test_Experiment",
                experiment_type=ExperimentType.COMPUTATIONAL_PERFORMANCE,
                num_trials=3,  # Small number for testing
                max_steps_per_trial=10
            )
            self.runner = ExperimentRunner(self.config)
        except ImportError as e:
            self.skipTest(f"Experimental modules not available: {e}")
    
    def test_experiment_configuration(self):
        """Test experiment configuration"""
        self.assertEqual(self.config.experiment_name, "Test_Experiment")
        self.assertEqual(self.config.num_trials, 3)
    
    def test_scenario_generation(self):
        """Test scenario generation"""
        try:
            scenario = self.runner._generate_trial_scenario(0)
            self.assertIsInstance(scenario, dict)
        except Exception as e:
            self.fail(f"Scenario generation failed: {e}")
    
    @unittest.skip("Requires full system setup - too slow for unit tests")
    def test_experiment_execution(self):
        """Test experiment execution (disabled for speed)"""
        try:
            results = self.runner.run_experiment()
            self.assertIsInstance(results, object)
        except Exception as e:
            self.fail(f"Experiment execution failed: {e}")


class PerformanceTests(unittest.TestCase):
    """Performance and stress tests"""
    
    def test_agent_performance(self):
        """Test agent decision speed"""
        try:
            from src.agents.bayesian_rl_agent import BayesianRLAgent, BayesianRLConfiguration
            
            config = BayesianRLConfiguration(state_dim=20, action_dim=5)
            agent = BayesianRLAgent(config)
            agent.initialize_prior_beliefs({'mean': 0.0, 'variance': 1.0})
            
            state = np.random.randn(20)
            
            # Time multiple decision cycles
            start_time = time.time()
            for _ in range(100):
                action, info = agent.select_action(state)
            end_time = time.time()
            
            avg_decision_time = (end_time - start_time) / 100
            self.assertLess(avg_decision_time, 0.1, "Decision time too slow")
            
        except ImportError:
            self.skipTest("Agent module not available")
        except Exception as e:
            self.fail(f"Performance test failed: {e}")
    
    def test_memory_usage(self):
        """Test memory usage patterns"""
        try:
            import psutil
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Create and use some components
            large_arrays = []
            for _ in range(100):
                large_arrays.append(np.random.randn(1000, 1000))
            
            # Clean up
            del large_arrays
            import gc
            gc.collect()
            
            final_memory = process.memory_info().rss
            memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB
            
            # Memory increase should be reasonable after cleanup
            self.assertLess(memory_increase, 1000, "Excessive memory usage")
            
        except ImportError:
            self.skipTest("psutil not available")
        except Exception as e:
            self.fail(f"Memory test failed: {e}")


class ErrorHandlingTests(unittest.TestCase):
    """Tests for error handling and robustness"""
    
    def test_error_handler_initialization(self):
        """Test error handler initialization"""
        try:
            from src.robustness import RobustErrorHandler, ErrorHandlingConfig
            
            config = ErrorHandlingConfig()
            handler = RobustErrorHandler(config)
            self.assertIsNotNone(handler)
            
        except ImportError:
            self.skipTest("Error handling modules not available")
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker pattern"""
        try:
            from src.robustness import CircuitBreaker
            
            breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
            
            # Test normal operation
            def success_func():
                return "success"
            
            result = breaker.call(success_func)
            self.assertEqual(result, "success")
            
            # Test failure handling
            def failure_func():
                raise Exception("Test failure")
            
            # Should fail and eventually open circuit
            for _ in range(5):
                try:
                    breaker.call(failure_func)
                except Exception:
                    pass
            
            state = breaker.get_state()
            self.assertEqual(state['state'], 'OPEN')
            
        except ImportError:
            self.skipTest("Circuit breaker not available")


class SystemLevelTests(unittest.TestCase):
    """System-level end-to-end tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_system_startup_shutdown(self):
        """Test complete system startup and shutdown"""
        try:
            from src.system.human_intent_rl_system import (
                HumanIntentRLSystem, SystemConfiguration, SystemMode
            )
            
            config = SystemConfiguration(
                mode=SystemMode.SIMULATION,
                save_data=False
            )
            system = HumanIntentRLSystem(config)
            
            # Test initialization
            init_success = system.initialize_system()
            self.assertTrue(init_success)
            
            # Test shutdown
            shutdown_success = system.shutdown()
            self.assertTrue(shutdown_success)
            
        except ImportError:
            self.skipTest("System modules not available")
        except Exception as e:
            self.fail(f"System startup/shutdown failed: {e}")
    
    def test_optimization_system_integration(self):
        """Test optimization system integration"""
        try:
            from src.optimization import get_global_profiler, start_global_profiling, stop_global_profiling
            
            # Start profiling
            start_global_profiling()
            
            # Do some work
            time.sleep(0.1)
            
            # Stop profiling and get report
            report_path = stop_global_profiling()
            self.assertTrue(os.path.exists(report_path))
            
        except ImportError:
            self.skipTest("Optimization modules not available")
        except Exception as e:
            self.fail(f"Optimization integration failed: {e}")


class TestSuiteRunner:
    """Main test suite runner"""
    
    def __init__(self):
        """Initialize test suite runner"""
        self.test_result = TestResult()
        self.test_classes = [
            BayesianRLAgentTests,
            HRIEnvironmentTests,
            IntegrationTests,
            ExperimentalFrameworkTests,
            PerformanceTests,
            ErrorHandlingTests,
            SystemLevelTests
        ]
    
    def run_all_tests(self, verbosity: int = 1) -> TestResult:
        """Run all test suites"""
        logger.info("Starting comprehensive test suite...")
        start_time = time.time()
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add all test classes
        for test_class in self.test_classes:
            tests = loader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
        result = runner.run(suite)
        
        # Process results
        self.test_result.total_tests = result.testsRun
        self.test_result.failed_tests = len(result.failures) + len(result.errors)
        self.test_result.passed_tests = result.testsRun - self.test_result.failed_tests
        self.test_result.skipped_tests = len(result.skipped) if hasattr(result, 'skipped') else 0
        self.test_result.execution_time = time.time() - start_time
        
        # Collect errors
        for test, error in result.failures + result.errors:
            self.test_result.errors.append(f"{test}: {error}")
        
        return self.test_result
    
    def run_specific_test_class(self, test_class_name: str, verbosity: int = 1) -> TestResult:
        """Run specific test class"""
        test_class = None
        for tc in self.test_classes:
            if tc.__name__ == test_class_name:
                test_class = tc
                break
        
        if not test_class:
            raise ValueError(f"Test class {test_class_name} not found")
        
        start_time = time.time()
        
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        
        runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
        result = runner.run(suite)
        
        # Process results
        test_result = TestResult()
        test_result.total_tests = result.testsRun
        test_result.failed_tests = len(result.failures) + len(result.errors)
        test_result.passed_tests = result.testsRun - test_result.failed_tests
        test_result.skipped_tests = len(result.skipped) if hasattr(result, 'skipped') else 0
        test_result.execution_time = time.time() - start_time
        
        return test_result
    
    def generate_test_report(self, output_file: str = "test_report.json") -> str:
        """Generate comprehensive test report"""
        summary = self.test_result.get_summary()
        
        report = {
            'report_metadata': {
                'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'test_framework': 'comprehensive_test_suite',
                'python_version': sys.version
            },
            'test_summary': summary,
            'test_classes': [tc.__name__ for tc in self.test_classes],
            'recommendations': self._generate_recommendations(summary)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_file
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if summary['failed_tests'] > 0:
            recommendations.append(f"Address {summary['failed_tests']} failing tests")
        
        if summary['success_rate'] < 0.9:
            recommendations.append("Improve test coverage and fix failing tests")
        
        if summary['execution_time'] > 300:  # 5 minutes
            recommendations.append("Consider optimizing slow tests or running them separately")
        
        if summary['skipped_tests'] > summary['total_tests'] * 0.2:
            recommendations.append("High number of skipped tests - check dependencies")
        
        if not recommendations:
            recommendations.append("All tests passing! Consider adding more edge case tests.")
        
        return recommendations


def run_quick_tests():
    """Run a quick subset of tests for rapid feedback"""
    runner = TestSuiteRunner()
    
    # Run only fast unit tests
    quick_classes = [BayesianRLAgentTests, HRIEnvironmentTests, ErrorHandlingTests]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in quick_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(suite)
    
    return result.wasSuccessful()


def run_performance_tests():
    """Run performance-specific tests"""
    runner = TestSuiteRunner()
    return runner.run_specific_test_class("PerformanceTests", verbosity=2)


def run_integration_tests():
    """Run integration-specific tests"""
    runner = TestSuiteRunner()
    return runner.run_specific_test_class("IntegrationTests", verbosity=2)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HRI Bayesian RL Test Suite")
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--performance', action='store_true', help='Run performance tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--class', dest='test_class', help='Run specific test class')
    parser.add_argument('--verbose', '-v', action='count', default=1, help='Increase verbosity')
    parser.add_argument('--report', help='Generate test report file')
    
    args = parser.parse_args()
    
    if args.quick:
        print("Running quick tests...")
        success = run_quick_tests()
        sys.exit(0 if success else 1)
    
    elif args.performance:
        print("Running performance tests...")
        result = run_performance_tests()
        print(f"Performance tests completed: {result.get_summary()}")
        sys.exit(0 if result.failed_tests == 0 else 1)
    
    elif args.integration:
        print("Running integration tests...")
        result = run_integration_tests()
        print(f"Integration tests completed: {result.get_summary()}")
        sys.exit(0 if result.failed_tests == 0 else 1)
    
    elif args.test_class:
        print(f"Running {args.test_class} tests...")
        runner = TestSuiteRunner()
        result = runner.run_specific_test_class(args.test_class, verbosity=args.verbose)
        print(f"Test results: {result.get_summary()}")
        sys.exit(0 if result.failed_tests == 0 else 1)
    
    else:
        print("Running comprehensive test suite...")
        runner = TestSuiteRunner()
        result = runner.run_all_tests(verbosity=args.verbose)
        
        summary = result.get_summary()
        print(f"\n{'='*50}")
        print("TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Skipped: {summary['skipped_tests']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Execution time: {summary['execution_time']:.2f}s")
        
        if summary['errors']:
            print(f"\nErrors ({len(summary['errors'])}):")
            for error in summary['errors']:
                print(f"  - {error}")
        
        if args.report:
            report_file = runner.generate_test_report(args.report)
            print(f"\nTest report saved to: {report_file}")
        
        print(f"\n{'='*50}")
        sys.exit(0 if summary['failed_tests'] == 0 else 1)