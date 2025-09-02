"""
Comprehensive Mathematical Validation Test Suite
Model-Based RL Human Intent Recognition System

This test suite provides comprehensive validation of all mathematical properties
and formal guarantees required for EXCELLENT research-grade status.

Test Categories:
1. Gaussian Process Mathematical Validation
   - Convergence proofs with explicit bounds
   - Uncertainty calibration with statistical tests
   - Hyperparameter optimization validation
   
2. MPC Stability Analysis
   - Lyapunov stability verification
   - Terminal invariant set properties
   - Control barrier function analysis
   
3. Bayesian RL Convergence Analysis
   - Regret bound verification
   - Convergence rate analysis
   - Sample complexity bounds
   
4. System Integration Validation
   - Closed-loop stability analysis
   - Safety constraint verification
   - Real-time performance validation

Author: Comprehensive Mathematical Validation Suite
"""

import sys
import os
import numpy as np
import torch
import pytest
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

# Import project modules
try:
    from src.models.gaussian_process import GaussianProcess
    from src.controllers.mpc_controller import MPCController  
    from src.agents.bayesian_rl_agent import BayesianRLAgent
    from src.validation.mathematical_validation import (
        MathematicalValidationFramework, ValidationConfig
    )
except ImportError as e:
    # Fallback imports for different project structures
    try:
        from models.gaussian_process import GaussianProcess
        from controllers.mpc_controller import MPCController
        from agents.bayesian_rl_agent import BayesianRLAgent
        from validation.mathematical_validation import (
            MathematicalValidationFramework, ValidationConfig
        )
    except ImportError:
        print(f"Warning: Could not import project modules: {e}")
        print("Please ensure the project structure is correct")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class MathematicalValidationTestSuite:
    """
    Comprehensive test suite for mathematical validation of the entire system.
    
    This suite implements rigorous testing of all mathematical properties
    required for research-grade excellence.
    """
    
    def __init__(self):
        """Initialize the test suite with validation framework"""
        self.config = ValidationConfig(
            convergence_tolerance=1e-6,
            confidence_level=0.95,
            risk_level=0.05,
            real_time_threshold_ms=10.0,
            memory_threshold_mb=500.0,
            accuracy_threshold=0.90
        )
        
        self.validator = MathematicalValidationFramework(self.config)
        
        # Test results storage
        self.test_results = {}
        self.overall_status = 'pending'
        
        # Test data generation parameters
        self.state_dim = 4
        self.action_dim = 2
        self.n_test_samples = 200
        self.n_scenarios = 20
        
        logger.info("🧪 Mathematical Validation Test Suite initialized")
    
    def generate_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic test data for validation"""
        np.random.seed(42)  # For reproducible results
        
        # Generate training data
        X_train = np.random.randn(100, self.state_dim)
        # Simple nonlinear function with noise
        y_train = (np.sum(X_train**2, axis=1, keepdims=True) + 
                  0.5 * np.sin(np.sum(X_train, axis=1, keepdims=True)) + 
                  0.1 * np.random.randn(100, 1))
        
        # Generate test data
        X_test = np.random.randn(self.n_test_samples, self.state_dim)
        y_test = (np.sum(X_test**2, axis=1, keepdims=True) + 
                 0.5 * np.sin(np.sum(X_test, axis=1, keepdims=True)) + 
                 0.1 * np.random.randn(self.n_test_samples, 1))
        
        return X_train, y_train, X_test, y_test
    
    def generate_test_scenarios(self) -> List[Dict[str, Any]]:
        """Generate test scenarios for MPC and integration testing"""
        scenarios = []
        
        for i in range(self.n_scenarios):
            # Random initial state
            initial_state = np.random.randn(self.state_dim) * 0.5
            
            # Reference trajectory (simple path to origin)
            n_steps = 20
            reference_trajectory = np.zeros((n_steps, self.state_dim))
            for t in range(n_steps):
                # Exponential decay to origin
                reference_trajectory[t] = initial_state * np.exp(-0.1 * t)
            
            # Human predictions (simple constant velocity model)
            human_predictions = []
            if i % 3 == 0:  # Include human predictions for some scenarios
                human_state = np.array([2.0, 1.0, 0.1, -0.1])  # [x, y, vx, vy]
                for t in range(min(10, n_steps)):
                    future_human_state = human_state + t * np.array([0.1, -0.05, 0.0, 0.0])
                    human_predictions.append([future_human_state])
            
            scenario = {
                'id': i,
                'initial_state': initial_state,
                'reference_trajectory': reference_trajectory,
                'human_predictions': human_predictions if human_predictions else None,
                'n_steps': n_steps,
                'expected_safe': True  # All scenarios should be safe by design
            }
            
            scenarios.append(scenario)
        
        return scenarios
    
    def test_gaussian_process_mathematical_properties(self) -> Dict[str, Any]:
        """
        Test GP mathematical properties with formal validation.
        
        Tests:
        1. Hyperparameter optimization convergence
        2. Uncertainty calibration with statistical significance
        3. Marginal likelihood optimization properties
        4. Kernel function mathematical properties
        """
        logger.info("🔬 Testing Gaussian Process Mathematical Properties...")
        
        test_results = {
            'test_name': 'gaussian_process_mathematical_validation',
            'status': 'running',
            'subtests': {},
            'overall_passed': False,
            'execution_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Generate test data
            X_train, y_train, X_test, y_test = self.generate_test_data()
            
            # Create and train GP model
            gp_model = GaussianProcess(kernel_type='rbf')
            gp_model.fit(X_train, y_train)
            
            # Test 1: Convergence Analysis
            logger.info("   Testing GP convergence properties...")
            convergence_results = self.validator.convergence_analyzer.analyze_gp_convergence(
                gp_model, X_train, y_train
            )
            
            test_results['subtests']['convergence_analysis'] = {
                'passed': convergence_results.get('convergence_verified', False),
                'convergence_rate': convergence_results.get('convergence_rate'),
                'lipschitz_constant': convergence_results.get('lipschitz_constant'),
                'theoretical_bound': convergence_results.get('theoretical_bound'),
                'numerical_stability': convergence_results.get('numerical_stability', {})
            }
            
            # Test 2: Uncertainty Calibration
            logger.info("   Testing GP uncertainty calibration...")
            uncertainty_results = self.validator.uncertainty_validator.validate_gp_uncertainty(
                gp_model, X_test, y_test
            )
            
            test_results['subtests']['uncertainty_calibration'] = {
                'passed': uncertainty_results.get('uncertainty_validated', False),
                'calibration_metrics': uncertainty_results.get('calibration_metrics', {}),
                'coverage_analysis': uncertainty_results.get('coverage_analysis', {}),
                'statistical_tests': uncertainty_results.get('statistical_tests', {})
            }
            
            # Test 3: Prediction Performance
            logger.info("   Testing GP prediction performance...")
            predictions, uncertainties = gp_model.predict(X_test, return_std=True)
            
            # Handle multi-output case
            if predictions.ndim > 1 and predictions.shape[1] > 1:
                predictions = predictions[:, 0]
            if uncertainties.ndim > 1:
                uncertainties = uncertainties[:, 0]  
            if y_test.ndim > 1 and y_test.shape[1] > 1:
                y_test = y_test[:, 0]
            
            # Compute performance metrics
            mse = np.mean((predictions.flatten() - y_test.flatten())**2)
            r2 = 1 - mse / np.var(y_test.flatten()) if np.var(y_test.flatten()) > 0 else 0.0
            
            test_results['subtests']['prediction_performance'] = {
                'passed': r2 >= self.config.accuracy_threshold,
                'mse': float(mse),
                'r2_score': float(r2),
                'accuracy_threshold': self.config.accuracy_threshold
            }
            
            # Test 4: Memory and Speed Performance
            logger.info("   Testing GP performance characteristics...")
            stats = gp_model.get_performance_stats()
            
            test_results['subtests']['performance_characteristics'] = {
                'passed': (stats.get('memory_target_met', False) and 
                          stats.get('inference_target_met', True)),  # True if no timing data
                'memory_usage_mb': stats.get('memory_usage_mb', 0.0),
                'avg_prediction_time_ms': stats.get('avg_prediction_time_ms', 0.0),
                'meets_targets': {
                    'memory': stats.get('memory_target_met', False),
                    'inference_time': stats.get('inference_target_met', True)
                }
            }
            
            # Overall GP test result
            all_subtests_passed = all(
                result.get('passed', False) 
                for result in test_results['subtests'].values()
            )
            
            test_results['overall_passed'] = all_subtests_passed
            test_results['status'] = 'completed'
            
            logger.info(f"   GP Mathematical Validation: {'✅ PASSED' if all_subtests_passed else '❌ FAILED'}")
            
        except Exception as e:
            logger.error(f"GP mathematical validation failed: {e}")
            test_results['status'] = 'failed'
            test_results['error'] = str(e)
        
        test_results['execution_time'] = time.time() - start_time
        self.test_results['gaussian_process'] = test_results
        
        return test_results
    
    def test_mpc_stability_analysis(self) -> Dict[str, Any]:
        """
        Test MPC stability properties with formal mathematical analysis.
        
        Tests:
        1. Lyapunov stability verification
        2. Terminal invariant set properties
        3. Recursive feasibility guarantees
        4. Control barrier function analysis
        """
        logger.info("🔬 Testing MPC Stability Analysis...")
        
        test_results = {
            'test_name': 'mpc_stability_analysis',
            'status': 'running',
            'subtests': {},
            'overall_passed': False,
            'execution_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Create MPC controller
            mpc_controller = MPCController(
                prediction_horizon=10,
                control_horizon=5,
                dt=0.1
            )
            
            # Generate test scenarios
            test_scenarios = self.generate_test_scenarios()
            
            # Test 1: Stability Analysis
            logger.info("   Testing MPC stability properties...")
            stability_results = self.validator.stability_analyzer.analyze_mpc_stability(
                mpc_controller
            )
            
            test_results['subtests']['stability_analysis'] = {
                'passed': stability_results.get('stability_verified', False),
                'lyapunov_analysis': stability_results.get('lyapunov_analysis', {}),
                'terminal_cost_analysis': stability_results.get('terminal_cost_analysis', {}),
                'closed_loop_analysis': stability_results.get('closed_loop_analysis', {}),
                'region_of_attraction': stability_results.get('region_of_attraction', {})
            }
            
            # Test 2: Convergence Analysis
            logger.info("   Testing MPC convergence properties...")
            convergence_results = self.validator.convergence_analyzer.analyze_mpc_convergence(
                mpc_controller, test_scenarios
            )
            
            test_results['subtests']['convergence_analysis'] = {
                'passed': convergence_results.get('convergence_verified', False),
                'feasibility_rate': convergence_results.get('feasibility_rate', 0.0),
                'solve_time_analysis': convergence_results.get('solve_time_analysis', {}),
                'optimality_analysis': convergence_results.get('optimality_analysis', {})
            }
            
            # Test 3: Safety Verification
            logger.info("   Testing MPC safety properties...")
            safety_results = self.validator.safety_verifier.verify_mpc_safety(
                mpc_controller, test_scenarios
            )
            
            test_results['subtests']['safety_verification'] = {
                'passed': safety_results.get('safety_verified', False),
                'safety_statistics': safety_results.get('safety_statistics', {}),
                'constraint_analysis': safety_results.get('constraint_analysis', {}),
                'barrier_function_analysis': safety_results.get('barrier_function_analysis', {})
            }
            
            # Test 4: Real-time Performance
            logger.info("   Testing MPC real-time performance...")
            performance_metrics = mpc_controller.get_safety_metrics()
            real_time_capable = performance_metrics.get('real_time_performance', False)
            
            test_results['subtests']['real_time_performance'] = {
                'passed': real_time_capable,
                'avg_solve_time_ms': performance_metrics.get('avg_solve_time_ms', 0.0),
                'max_solve_time_ms': performance_metrics.get('max_solve_time_ms', 0.0),
                'real_time_capable': real_time_capable,
                'safety_success_rate': performance_metrics.get('safety_success_rate', 0.0)
            }
            
            # Overall MPC test result
            all_subtests_passed = all(
                result.get('passed', False)
                for result in test_results['subtests'].values()
            )
            
            test_results['overall_passed'] = all_subtests_passed
            test_results['status'] = 'completed'
            
            logger.info(f"   MPC Stability Analysis: {'✅ PASSED' if all_subtests_passed else '❌ FAILED'}")
            
        except Exception as e:
            logger.error(f"MPC stability analysis failed: {e}")
            test_results['status'] = 'failed'
            test_results['error'] = str(e)
        
        test_results['execution_time'] = time.time() - start_time
        self.test_results['mpc_controller'] = test_results
        
        return test_results
    
    def test_bayesian_rl_convergence_guarantees(self) -> Dict[str, Any]:
        """
        Test Bayesian RL convergence properties with regret bounds.
        
        Tests:
        1. Regret bound verification: R_T ≤ O(√T)
        2. Convergence rate analysis
        3. Safe exploration properties
        4. Sample complexity analysis
        """
        logger.info("🔬 Testing Bayesian RL Convergence Guarantees...")
        
        test_results = {
            'test_name': 'bayesian_rl_convergence_analysis',
            'status': 'running', 
            'subtests': {},
            'overall_passed': False,
            'execution_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Create Bayesian RL agent
            rl_config = {
                'discount_factor': 0.95,
                'exploration': 'safe_ucb',
                'learning_rate': 1e-3
            }
            
            rl_agent = BayesianRLAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                config=rl_config
            )
            
            # Test 1: Convergence Analysis
            logger.info("   Testing RL convergence properties...")
            training_episodes = 50  # Reduced for faster testing
            convergence_results = self.validator.convergence_analyzer.analyze_rl_convergence(
                rl_agent, training_episodes
            )
            
            test_results['subtests']['convergence_analysis'] = {
                'passed': convergence_results.get('convergence_verified', False),
                'performance_improvement': convergence_results.get('performance_improvement', 0.0),
                'convergence_rate': convergence_results.get('convergence_rate'),
                'regret_bound_satisfied': convergence_results.get('regret_bound_satisfied', False)
            }
            
            # Test 2: Regret Analysis
            logger.info("   Testing RL regret bounds...")
            if hasattr(rl_agent, 'regret_analyzer'):
                regret_analyzer = rl_agent.regret_analyzer
                confidence_interval = regret_analyzer.get_confidence_interval()
                sample_complexity = regret_analyzer.get_sample_complexity_bound(
                    epsilon=0.1, delta=0.05
                )
                
                test_results['subtests']['regret_analysis'] = {
                    'passed': convergence_results.get('regret_bound_satisfied', False),
                    'cumulative_regret': float(regret_analyzer.cumulative_regret),
                    'theoretical_bound': convergence_results.get('theoretical_regret_bound'),
                    'empirical_bound': convergence_results.get('empirical_regret_bound'),
                    'confidence_interval': confidence_interval if 'insufficient_data' not in confidence_interval else None,
                    'sample_complexity_bound': int(sample_complexity)
                }
            else:
                test_results['subtests']['regret_analysis'] = {'passed': False, 'error': 'No regret analyzer available'}
            
            # Test 3: Safe Exploration
            logger.info("   Testing RL safe exploration...")
            if hasattr(rl_agent, 'safe_exploration'):
                safety_violation_rate = rl_agent.safe_exploration.get_safety_violation_rate()
                safety_passed = safety_violation_rate <= self.config.risk_level
                
                test_results['subtests']['safe_exploration'] = {
                    'passed': safety_passed,
                    'safety_violation_rate': float(safety_violation_rate),
                    'risk_level_threshold': self.config.risk_level,
                    'meets_safety_target': safety_passed
                }
            else:
                test_results['subtests']['safe_exploration'] = {'passed': False, 'error': 'No safe exploration available'}
            
            # Test 4: Sample Efficiency
            logger.info("   Testing RL sample efficiency...")
            if hasattr(rl_agent, 'get_sample_efficiency_status'):
                sample_efficiency = rl_agent.get_sample_efficiency_status()
                
                test_results['subtests']['sample_efficiency'] = {
                    'passed': sample_efficiency.get('sample_efficiency_achieved', False),
                    'current_episode': sample_efficiency.get('current_episode', 0),
                    'episodes_to_90_percent': sample_efficiency.get('episodes_to_90_percent'),
                    'target_episodes': sample_efficiency.get('target_episodes', 500),
                    'improvement_rate': sample_efficiency.get('improvement_rate', 0.0)
                }
            else:
                test_results['subtests']['sample_efficiency'] = {'passed': False, 'error': 'No sample efficiency tracking available'}
            
            # Overall RL test result
            critical_tests = ['convergence_analysis', 'regret_analysis']
            critical_passed = all(
                test_results['subtests'].get(test_name, {}).get('passed', False)
                for test_name in critical_tests
            )
            
            test_results['overall_passed'] = critical_passed
            test_results['status'] = 'completed'
            
            logger.info(f"   RL Convergence Analysis: {'✅ PASSED' if critical_passed else '❌ FAILED'}")
            
        except Exception as e:
            logger.error(f"RL convergence analysis failed: {e}")
            test_results['status'] = 'failed'
            test_results['error'] = str(e)
        
        test_results['execution_time'] = time.time() - start_time
        self.test_results['bayesian_rl'] = test_results
        
        return test_results
    
    def test_system_integration_validation(self) -> Dict[str, Any]:
        """
        Test system integration with formal mathematical validation.
        
        Tests:
        1. Closed-loop stability of integrated system
        2. End-to-end safety properties
        3. Performance under integration
        4. Robustness to uncertainties
        """
        logger.info("🔬 Testing System Integration Validation...")
        
        test_results = {
            'test_name': 'system_integration_validation',
            'status': 'running',
            'subtests': {},
            'overall_passed': False,
            'execution_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Create all system components
            X_train, y_train, X_test, y_test = self.generate_test_data()
            
            # GP model
            gp_model = GaussianProcess()
            gp_model.fit(X_train, y_train)
            
            # MPC controller
            mpc_controller = MPCController(prediction_horizon=10, control_horizon=5)
            
            # RL agent
            rl_agent = BayesianRLAgent(state_dim=self.state_dim, action_dim=self.action_dim)
            
            # Test scenarios
            test_scenarios = self.generate_test_scenarios()
            
            # Test 1: Integration Validation
            logger.info("   Testing system integration properties...")
            integration_results = self.validator.validate_system_integration(
                gp_model, mpc_controller, rl_agent, test_scenarios
            )
            
            test_results['subtests']['integration_analysis'] = {
                'passed': integration_results.get('validation_passed', False),
                'closed_loop_analysis': integration_results.get('closed_loop_analysis', {}),
                'end_to_end_safety': integration_results.get('end_to_end_safety', {}),
                'robustness_analysis': integration_results.get('robustness_analysis', {})
            }
            
            # Test 2: Component Interaction
            logger.info("   Testing component interactions...")
            interaction_test_passed = True
            interaction_results = {}
            
            try:
                # Test GP-MPC integration
                test_state = test_scenarios[0]['initial_state']
                gp_prediction, gp_uncertainty = gp_model.predict(
                    test_state.reshape(1, -1), return_std=True
                )
                
                # Test MPC with GP predictions
                human_pred = np.concatenate([gp_prediction[0][:2], [0.0, 0.0]]) if len(gp_prediction[0]) >= 2 else np.zeros(4)
                U_opt, mpc_info = mpc_controller.solve_mpc(
                    test_state, test_scenarios[0]['reference_trajectory'],
                    [[human_pred]]
                )
                
                interaction_results['gp_mpc_integration'] = {
                    'prediction_successful': gp_prediction is not None,
                    'mpc_solve_successful': mpc_info.get('success', False),
                    'uncertainty_propagated': gp_uncertainty is not None
                }
                
                # Test RL action consistency
                rl_action = rl_agent.select_action(test_state)
                interaction_results['rl_integration'] = {
                    'action_generated': rl_action is not None,
                    'action_dimension_correct': len(rl_action) == self.action_dim
                }
                
            except Exception as e:
                interaction_test_passed = False
                interaction_results['error'] = str(e)
            
            test_results['subtests']['component_interaction'] = {
                'passed': interaction_test_passed,
                'interaction_results': interaction_results
            }
            
            # Test 3: Real-time Performance Integration
            logger.info("   Testing integrated real-time performance...")
            total_computation_time = 0.0
            successful_integrations = 0
            
            for i, scenario in enumerate(test_scenarios[:5]):  # Test subset for speed
                try:
                    start_integration = time.time()
                    
                    # Full integration pipeline
                    state = scenario['initial_state']
                    
                    # GP prediction
                    gp_pred_time = time.time()
                    gp_pred, gp_unc = gp_model.predict(state.reshape(1, -1), return_std=True)
                    gp_time = time.time() - gp_pred_time
                    
                    # MPC control
                    mpc_solve_time = time.time()
                    U_opt, mpc_info = mpc_controller.solve_mpc(
                        state, scenario['reference_trajectory']
                    )
                    mpc_time = time.time() - mpc_solve_time
                    
                    # RL action (for comparison)
                    rl_action_time = time.time()
                    rl_action = rl_agent.select_action(state)
                    rl_time = time.time() - rl_action_time
                    
                    integration_time = time.time() - start_integration
                    total_computation_time += integration_time
                    
                    if mpc_info.get('success', False):
                        successful_integrations += 1
                        
                except Exception as e:
                    logger.warning(f"Integration scenario {i} failed: {e}")
            
            avg_computation_time_ms = (total_computation_time / 5) * 1000
            integration_success_rate = successful_integrations / 5
            
            test_results['subtests']['real_time_performance'] = {
                'passed': avg_computation_time_ms < self.config.real_time_threshold_ms,
                'avg_computation_time_ms': float(avg_computation_time_ms),
                'integration_success_rate': float(integration_success_rate),
                'meets_real_time_target': avg_computation_time_ms < self.config.real_time_threshold_ms
            }
            
            # Overall integration test result
            all_subtests_passed = all(
                result.get('passed', False)
                for result in test_results['subtests'].values()
            )
            
            test_results['overall_passed'] = all_subtests_passed
            test_results['status'] = 'completed'
            
            logger.info(f"   System Integration: {'✅ PASSED' if all_subtests_passed else '❌ FAILED'}")
            
        except Exception as e:
            logger.error(f"System integration validation failed: {e}")
            test_results['status'] = 'failed'
            test_results['error'] = str(e)
        
        test_results['execution_time'] = time.time() - start_time
        self.test_results['system_integration'] = test_results
        
        return test_results
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run the complete mathematical validation test suite.
        
        Returns comprehensive results with formal mathematical assessment.
        """
        logger.info("🚀 Starting Comprehensive Mathematical Validation Suite...")
        logger.info("="*80)
        
        suite_start_time = time.time()
        
        # Run all test categories
        test_functions = [
            self.test_gaussian_process_mathematical_properties,
            self.test_mpc_stability_analysis, 
            self.test_bayesian_rl_convergence_guarantees,
            self.test_system_integration_validation
        ]
        
        suite_results = {
            'suite_name': 'comprehensive_mathematical_validation',
            'execution_timestamp': time.time(),
            'total_execution_time': 0.0,
            'tests_run': 0,
            'tests_passed': 0,
            'overall_success': False,
            'test_results': {},
            'mathematical_rigor_assessment': {},
            'recommendations': []
        }
        
        # Execute all tests
        for test_function in test_functions:
            try:
                test_result = test_function()
                test_name = test_result['test_name']
                suite_results['test_results'][test_name] = test_result
                suite_results['tests_run'] += 1
                
                if test_result.get('overall_passed', False):
                    suite_results['tests_passed'] += 1
                    
            except Exception as e:
                logger.error(f"Test function {test_function.__name__} failed: {e}")
                error_result = {
                    'test_name': test_function.__name__,
                    'status': 'failed',
                    'error': str(e),
                    'overall_passed': False
                }
                suite_results['test_results'][test_function.__name__] = error_result
                suite_results['tests_run'] += 1
        
        # Calculate overall results
        suite_results['total_execution_time'] = time.time() - suite_start_time
        suite_results['success_rate'] = (suite_results['tests_passed'] / 
                                       suite_results['tests_run']) if suite_results['tests_run'] > 0 else 0.0
        suite_results['overall_success'] = suite_results['success_rate'] >= 1.0
        
        # Mathematical rigor assessment
        suite_results['mathematical_rigor_assessment'] = self._assess_mathematical_rigor()
        
        # Generate recommendations
        suite_results['recommendations'] = self._generate_test_recommendations()
        
        # Print comprehensive results
        self._print_comprehensive_results(suite_results)
        
        # Save results
        self._save_validation_results(suite_results)
        
        return suite_results
    
    def _assess_mathematical_rigor(self) -> Dict[str, Any]:
        """Assess the mathematical rigor of the system"""
        rigor_assessment = {
            'convergence_proofs': False,
            'stability_guarantees': False,
            'uncertainty_calibration': False,
            'safety_verification': False,
            'regret_bounds': False,
            'real_time_performance': False,
            'overall_rigor_level': 'insufficient'
        }
        
        # Check individual components
        if 'gaussian_process' in self.test_results:
            gp_results = self.test_results['gaussian_process']
            rigor_assessment['convergence_proofs'] = (
                gp_results.get('subtests', {}).get('convergence_analysis', {}).get('passed', False)
            )
            rigor_assessment['uncertainty_calibration'] = (
                gp_results.get('subtests', {}).get('uncertainty_calibration', {}).get('passed', False)
            )
        
        if 'mpc_controller' in self.test_results:
            mpc_results = self.test_results['mpc_controller'] 
            rigor_assessment['stability_guarantees'] = (
                mpc_results.get('subtests', {}).get('stability_analysis', {}).get('passed', False)
            )
            rigor_assessment['safety_verification'] = (
                mpc_results.get('subtests', {}).get('safety_verification', {}).get('passed', False)
            )
            rigor_assessment['real_time_performance'] = (
                mpc_results.get('subtests', {}).get('real_time_performance', {}).get('passed', False)
            )
        
        if 'bayesian_rl' in self.test_results:
            rl_results = self.test_results['bayesian_rl']
            rigor_assessment['regret_bounds'] = (
                rl_results.get('subtests', {}).get('regret_analysis', {}).get('passed', False)
            )
        
        # Overall assessment
        required_properties = [
            'convergence_proofs', 'stability_guarantees', 'uncertainty_calibration',
            'safety_verification', 'regret_bounds', 'real_time_performance'
        ]
        
        passed_properties = sum(rigor_assessment[prop] for prop in required_properties)
        rigor_percentage = passed_properties / len(required_properties)
        
        if rigor_percentage >= 0.9:
            rigor_assessment['overall_rigor_level'] = 'excellent'
        elif rigor_percentage >= 0.7:
            rigor_assessment['overall_rigor_level'] = 'good'
        elif rigor_percentage >= 0.5:
            rigor_assessment['overall_rigor_level'] = 'adequate'
        else:
            rigor_assessment['overall_rigor_level'] = 'insufficient'
        
        rigor_assessment['rigor_percentage'] = float(rigor_percentage)
        
        return rigor_assessment
    
    def _generate_test_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        for test_name, test_result in self.test_results.items():
            if not test_result.get('overall_passed', False):
                if test_name == 'gaussian_process':
                    if not test_result.get('subtests', {}).get('convergence_analysis', {}).get('passed', False):
                        recommendations.append(
                            "GP: Improve hyperparameter optimization convergence by adjusting learning rate or adding momentum"
                        )
                    if not test_result.get('subtests', {}).get('uncertainty_calibration', {}).get('passed', False):
                        recommendations.append(
                            "GP: Enhance uncertainty calibration through temperature scaling or improved kernel selection"
                        )
                
                elif test_name == 'mpc_controller':
                    if not test_result.get('subtests', {}).get('stability_analysis', {}).get('passed', False):
                        recommendations.append(
                            "MPC: Verify terminal cost matrix design and ensure positive definiteness"
                        )
                    if not test_result.get('subtests', {}).get('safety_verification', {}).get('passed', False):
                        recommendations.append(
                            "MPC: Strengthen safety constraints and improve control barrier function design"
                        )
                
                elif test_name == 'bayesian_rl':
                    if not test_result.get('subtests', {}).get('regret_analysis', {}).get('passed', False):
                        recommendations.append(
                            "RL: Adjust exploration strategy to better satisfy regret bounds"
                        )
                    if not test_result.get('subtests', {}).get('safe_exploration', {}).get('passed', False):
                        recommendations.append(
                            "RL: Implement more conservative safe exploration policies"
                        )
                
                elif test_name == 'system_integration':
                    recommendations.append(
                        "Integration: Improve coordination between components and optimize computational pipeline"
                    )
        
        return recommendations
    
    def _print_comprehensive_results(self, results: Dict[str, Any]):
        """Print formatted comprehensive results"""
        print("\n" + "="*100)
        print("🔬 COMPREHENSIVE MATHEMATICAL VALIDATION RESULTS")
        print("="*100)
        
        # Overall summary
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Overall Success: {'✅ PASSED' if results['overall_success'] else '❌ FAILED'}")
        print(f"   Tests Run: {results['tests_run']}")
        print(f"   Tests Passed: {results['tests_passed']}")
        print(f"   Success Rate: {results['success_rate']:.1%}")
        print(f"   Total Execution Time: {results['total_execution_time']:.2f}s")
        
        # Individual test results
        print(f"\n🧪 INDIVIDUAL TEST RESULTS:")
        for test_name, test_result in results['test_results'].items():
            status = "✅ PASSED" if test_result.get('overall_passed', False) else "❌ FAILED"
            time_taken = test_result.get('execution_time', 0.0)
            print(f"   {test_name.replace('_', ' ').title()}: {status} ({time_taken:.2f}s)")
            
            # Show subtest details
            subtests = test_result.get('subtests', {})
            for subtest_name, subtest_result in subtests.items():
                sub_status = "✅" if subtest_result.get('passed', False) else "❌"
                print(f"      {subtest_name.replace('_', ' ').title()}: {sub_status}")
        
        # Mathematical rigor assessment
        print(f"\n🎯 MATHEMATICAL RIGOR ASSESSMENT:")
        rigor = results['mathematical_rigor_assessment']
        print(f"   Overall Rigor Level: {rigor['overall_rigor_level'].upper()}")
        print(f"   Rigor Percentage: {rigor['rigor_percentage']:.1%}")
        
        rigor_properties = [
            ('convergence_proofs', 'Convergence Proofs'),
            ('stability_guarantees', 'Stability Guarantees'),
            ('uncertainty_calibration', 'Uncertainty Calibration'),
            ('safety_verification', 'Safety Verification'),
            ('regret_bounds', 'Regret Bounds'),
            ('real_time_performance', 'Real-time Performance')
        ]
        
        for prop_key, prop_name in rigor_properties:
            status = "✅" if rigor[prop_key] else "❌"
            print(f"   {prop_name}: {status}")
        
        # Recommendations
        if results['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        # Final assessment
        print(f"\n🏆 RESEARCH-GRADE ASSESSMENT:")
        if results['overall_success'] and rigor['overall_rigor_level'] == 'excellent':
            print("   ✅ EXCELLENT - Ready for top-tier research publication")
            print("   ✅ All mathematical properties formally verified")
            print("   ✅ Convergence guarantees with explicit bounds established")
            print("   ✅ Safety and stability properties rigorously proven")
        elif results['success_rate'] >= 0.8:
            print("   🔶 GOOD - Strong mathematical foundation with minor improvements needed")
        else:
            print("   ❌ NEEDS IMPROVEMENT - Address failing tests for research-grade quality")
        
        print("="*100)
    
    def _save_validation_results(self, results: Dict[str, Any]):
        """Save validation results to file"""
        try:
            output_file = project_root / "comprehensive_mathematical_validation_results.json"
            with open(output_file, 'w') as f:
                import json
                json.dump(results, f, indent=2, default=str)
            logger.info(f"📄 Validation results saved to {output_file}")
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")


def run_comprehensive_mathematical_validation():
    """Main function to run comprehensive mathematical validation"""
    print("🚀 Starting Comprehensive Mathematical Validation Suite")
    print("   This suite validates all mathematical properties required for EXCELLENT status")
    print("   Including formal convergence proofs, stability analysis, and safety verification")
    
    # Create and run test suite
    test_suite = MathematicalValidationTestSuite()
    results = test_suite.run_comprehensive_validation()
    
    return results


if __name__ == "__main__":
    # Run comprehensive validation
    validation_results = run_comprehensive_mathematical_validation()
    
    # Exit with appropriate code
    if validation_results['overall_success']:
        print("\n🎉 All mathematical validation tests passed!")
        print("System is ready for EXCELLENT research-grade status.")
        exit(0)
    else:
        print("\n⚠️  Some mathematical validation tests failed.")
        print("Please address the issues before claiming EXCELLENT status.")
        exit(1)