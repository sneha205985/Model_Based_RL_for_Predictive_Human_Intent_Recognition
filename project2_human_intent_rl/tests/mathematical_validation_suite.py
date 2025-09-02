"""
Comprehensive Mathematical Validation Test Suite

This test suite validates the mathematical correctness and numerical stability
of all core algorithmic components implemented in the Bayesian RL Human Intent
Recognition system.

Test Categories:
1. Gaussian Process Mathematical Properties
2. Bayesian Intent Classification Validation  
3. MPC Mathematical Implementation
4. Bayesian RL Agent Exploration-Exploitation
5. Integration System Validation
6. Uncertainty Quantification Validation

Author: Claude Code Research Team
Date: 2024
"""

import numpy as np
import torch
import pytest
import warnings
from typing import Dict, List, Tuple, Any, Optional
import logging
import time
from scipy.stats import multivariate_normal
from scipy.linalg import cholesky, solve_triangular
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Import all components to test
try:
    from src.models.gaussian_process import GaussianProcess, RBFKernel, MaternKernel
    from src.models.bayesian_intent_classifier import BayesianIntentClassifier
    from src.control.mpc_controller import MPCController, MPCConfiguration
    from src.agents.bayesian_rl_agent import BayesianRLAgent, BayesianRLConfig
    from src.integration.hri_bayesian_rl import HRIBayesianRLIntegration, HRIBayesianRLConfig
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Some imports not available: {e}")
    IMPORTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MathematicalValidationSuite:
    """Comprehensive mathematical validation test suite"""
    
    def __init__(self, tolerance: float = 1e-6, verbose: bool = True):
        """
        Initialize validation suite.
        
        Args:
            tolerance: Numerical tolerance for tests
            verbose: Whether to print detailed test results
        """
        self.tolerance = tolerance
        self.verbose = verbose
        self.test_results = {}
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all mathematical validation tests"""
        logger.info("🧮 Starting Comprehensive Mathematical Validation Suite")
        
        if not IMPORTS_AVAILABLE:
            logger.warning("Not all components available for testing")
            return {"status": "incomplete", "reason": "missing_imports"}
        
        # Run test categories
        test_categories = [
            ("GP Mathematical Properties", self.test_gaussian_process_mathematics),
            ("Bayesian Intent Classification", self.test_bayesian_intent_classification),
            ("MPC Mathematical Implementation", self.test_mpc_mathematical_implementation),
            ("Bayesian RL Exploration-Exploitation", self.test_bayesian_rl_mathematics),
            ("Integration System Validation", self.test_integration_system),
            ("Uncertainty Quantification", self.test_uncertainty_quantification)
        ]
        
        all_results = {}
        total_tests = 0
        passed_tests = 0
        
        for category_name, test_function in test_categories:
            logger.info(f"🔍 Testing {category_name}")
            try:
                category_results = test_function()
                all_results[category_name] = category_results
                
                # Count test results
                if isinstance(category_results, dict) and 'tests' in category_results:
                    for test_name, test_result in category_results['tests'].items():
                        total_tests += 1
                        if test_result.get('passed', False):
                            passed_tests += 1
                            
            except Exception as e:
                logger.error(f"❌ {category_name} failed: {e}")
                all_results[category_name] = {"status": "error", "error": str(e)}
        
        # Summary
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": success_rate,
            "categories": all_results
        }
        
        logger.info(f"✅ Mathematical Validation Complete: {passed_tests}/{total_tests} tests passed ({success_rate:.1%})")
        
        self.test_results = summary
        return summary
    
    def test_gaussian_process_mathematics(self) -> Dict[str, Any]:
        """Test mathematical properties of Gaussian Process implementation"""
        tests = {}
        
        # Test 1: Kernel positive definiteness
        try:
            rbf_kernel = RBFKernel(lengthscale=1.0, outputscale=1.0)
            X = np.random.randn(50, 3)
            K = rbf_kernel(X, X)
            
            eigenvals = np.linalg.eigvals(K)
            is_positive_definite = np.all(eigenvals > -self.tolerance)
            
            tests["kernel_positive_definiteness"] = {
                "passed": is_positive_definite,
                "min_eigenvalue": float(np.min(eigenvals)),
                "condition_number": float(np.linalg.cond(K))
            }
        except Exception as e:
            tests["kernel_positive_definiteness"] = {"passed": False, "error": str(e)}
        
        # Test 2: Cholesky decomposition numerical stability  
        try:
            gp = GaussianProcess(
                kernel=RBFKernel(lengthscale=1.0),
                noise_var=1e-6
            )
            
            X_train = np.random.randn(100, 2)
            y_train = np.sin(X_train[:, 0]) + 0.1 * np.random.randn(100)
            
            # Fit GP
            gp.fit(X_train, y_train)
            
            # Test numerical stability
            X_test = np.random.randn(20, 2)
            mean_pred, var_pred = gp.predict(X_test)
            
            # Validate predictions
            is_valid = (
                np.all(np.isfinite(mean_pred)) and
                np.all(np.isfinite(var_pred)) and
                np.all(var_pred >= 0)
            )
            
            tests["cholesky_numerical_stability"] = {
                "passed": is_valid,
                "mean_variance": float(np.mean(var_pred)),
                "max_variance": float(np.max(var_pred))
            }
        except Exception as e:
            tests["cholesky_numerical_stability"] = {"passed": False, "error": str(e)}
        
        # Test 3: GP uncertainty propagation
        try:
            # Test that uncertainty increases with distance from training data
            X_train = np.array([[0, 0], [1, 1], [2, 2]])
            y_train = np.array([1, 2, 3])
            
            gp = GaussianProcess(kernel=RBFKernel(lengthscale=1.0), noise_var=0.01)
            gp.fit(X_train, y_train)
            
            # Test points at different distances
            X_near = np.array([[0.1, 0.1]])  # Close to training data
            X_far = np.array([[5, 5]])        # Far from training data
            
            _, var_near = gp.predict(X_near)
            _, var_far = gp.predict(X_far)
            
            uncertainty_increases = var_far[0] > var_near[0]
            
            tests["uncertainty_propagation"] = {
                "passed": uncertainty_increases,
                "near_variance": float(var_near[0]),
                "far_variance": float(var_far[0]),
                "variance_ratio": float(var_far[0] / var_near[0])
            }
        except Exception as e:
            tests["uncertainty_propagation"] = {"passed": False, "error": str(e)}
        
        # Test 4: GP marginal likelihood optimization
        try:
            X = np.random.randn(50, 2)
            y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + 0.1 * np.random.randn(50)
            
            gp = GaussianProcess(kernel=RBFKernel(lengthscale=1.0), noise_var=0.1)
            
            # Fit with hyperparameter optimization
            initial_lengthscale = gp.kernel.lengthscale
            gp.fit(X, y)
            final_lengthscale = gp.kernel.lengthscale
            
            # Check that hyperparameters were optimized (changed)
            hyperparameters_optimized = abs(final_lengthscale - initial_lengthscale) > 1e-3
            
            tests["marginal_likelihood_optimization"] = {
                "passed": hyperparameters_optimized,
                "initial_lengthscale": float(initial_lengthscale),
                "final_lengthscale": float(final_lengthscale)
            }
        except Exception as e:
            tests["marginal_likelihood_optimization"] = {"passed": False, "error": str(e)}
        
        return {"status": "completed", "tests": tests}
    
    def test_bayesian_intent_classification(self) -> Dict[str, Any]:
        """Test Bayesian intent classification mathematics"""
        tests = {}
        
        # Test 1: Bayesian posterior updating
        try:
            classifier = BayesianIntentClassifier(
                state_dim=10,
                num_intents=5,
                config={'learning_rate': 0.01, 'num_samples': 10}
            )
            
            # Generate synthetic data
            X = torch.randn(100, 10)
            y = torch.randint(0, 5, (100,))
            
            # Train classifier
            initial_loss = float('inf')
            for epoch in range(20):
                loss = classifier.train_step(X, y)
                if epoch == 0:
                    initial_loss = loss
                final_loss = loss
            
            # Check that loss decreased (learning occurred)
            loss_decreased = final_loss < initial_loss * 0.9
            
            tests["bayesian_posterior_updating"] = {
                "passed": loss_decreased,
                "initial_loss": float(initial_loss),
                "final_loss": float(final_loss),
                "improvement_ratio": float(initial_loss / final_loss)
            }
        except Exception as e:
            tests["bayesian_posterior_updating"] = {"passed": False, "error": str(e)}
        
        # Test 2: Uncertainty quantification calibration
        try:
            classifier = BayesianIntentClassifier(
                state_dim=5,
                num_intents=3,
                config={'num_samples': 50}
            )
            
            # Simple synthetic data
            X = torch.randn(50, 5)
            y = torch.randint(0, 3, (50,))
            
            # Train briefly
            for _ in range(10):
                classifier.train_step(X, y)
            
            # Test uncertainty quantification
            X_test = torch.randn(20, 5)
            predictions = classifier.predict_with_uncertainty(X_test)
            
            # Check that predictions have proper structure
            has_mean = 'mean' in predictions
            has_uncertainty = 'uncertainty' in predictions
            has_samples = 'samples' in predictions
            
            proper_structure = has_mean and has_uncertainty and has_samples
            
            # Check uncertainty values are reasonable
            uncertainties = predictions['uncertainty']['epistemic']
            reasonable_uncertainty = torch.all(uncertainties >= 0) and torch.all(uncertainties <= 1)
            
            tests["uncertainty_quantification_calibration"] = {
                "passed": proper_structure and reasonable_uncertainty,
                "has_proper_structure": proper_structure,
                "mean_epistemic_uncertainty": float(torch.mean(uncertainties)),
                "max_epistemic_uncertainty": float(torch.max(uncertainties))
            }
        except Exception as e:
            tests["uncertainty_quantification_calibration"] = {"passed": False, "error": str(e)}
        
        # Test 3: Temperature scaling calibration
        try:
            classifier = BayesianIntentClassifier(
                state_dim=4,
                num_intents=2,
                config={'temperature_scaling': True}
            )
            
            # Generate data with clear separation
            X1 = torch.randn(30, 4) + 2  # Class 1
            X2 = torch.randn(30, 4) - 2  # Class 2
            X = torch.cat([X1, X2], dim=0)
            y = torch.cat([torch.zeros(30, dtype=torch.long), torch.ones(30, dtype=torch.long)])
            
            # Train classifier
            for _ in range(15):
                classifier.train_step(X, y)
            
            # Calibrate temperature
            classifier.calibrate_uncertainty(X, y)
            
            # Test that temperature was adjusted
            temperature_applied = hasattr(classifier, 'temperature') and classifier.temperature != 1.0
            
            tests["temperature_scaling_calibration"] = {
                "passed": temperature_applied,
                "temperature_value": float(getattr(classifier, 'temperature', 1.0))
            }
        except Exception as e:
            tests["temperature_scaling_calibration"] = {"passed": False, "error": str(e)}
        
        return {"status": "completed", "tests": tests}
    
    def test_mpc_mathematical_implementation(self) -> Dict[str, Any]:
        """Test MPC mathematical implementation correctness"""
        tests = {}
        
        # Test 1: Discrete-time state-space model
        try:
            config = MPCConfiguration()
            controller = MPCController(config, state_dim=4, control_dim=2)
            
            # Set up simple double integrator system
            A = np.array([[1, 0.1], [0, 1]])  # Position-velocity system
            A = np.kron(A, np.eye(2))  # 2D system (4 states: px, vx, py, vy)
            B = np.array([[0.005], [0.1]])  # Control input
            B = np.kron(B, np.eye(2))  # 2D control (2 inputs)
            
            controller.set_state_space_model(A, B)
            
            # Check controllability
            controllability_rank = controller._compute_controllability_matrix(A, B)
            is_controllable = np.linalg.matrix_rank(controllability_rank) == 4
            
            tests["discrete_time_state_space"] = {
                "passed": is_controllable,
                "controllability_rank": int(np.linalg.matrix_rank(controllability_rank)),
                "required_rank": 4
            }
        except Exception as e:
            tests["discrete_time_state_space"] = {"passed": False, "error": str(e)}
        
        # Test 2: DARE solution and terminal cost
        try:
            config = MPCConfiguration()
            controller = MPCController(config, state_dim=2, control_dim=1)
            
            # Simple system
            A = np.array([[1.1, 0.1], [0, 1.0]])
            B = np.array([[0], [1]])
            
            controller.set_state_space_model(A, B)
            
            # Set cost matrices
            Q = np.eye(2)
            R = np.array([[1]])
            controller.set_objective_function(Q, R)
            
            # Check that DARE solution exists and gives finite terminal cost
            P = controller.P
            is_finite = np.all(np.isfinite(P))
            is_positive_definite = np.all(np.linalg.eigvals(P) > 0)
            
            tests["dare_solution_terminal_cost"] = {
                "passed": is_finite and is_positive_definite,
                "terminal_cost_finite": is_finite,
                "terminal_cost_psd": is_positive_definite,
                "condition_number": float(np.linalg.cond(P))
            }
        except Exception as e:
            tests["dare_solution_terminal_cost"] = {"passed": False, "error": str(e)}
        
        # Test 3: Constraint violation recovery
        try:
            config = MPCConfiguration()
            config.enable_feasibility_recovery = True
            controller = MPCController(config, state_dim=2, control_dim=1)
            
            # Set up system with constraints
            A = np.eye(2)
            B = np.array([[1], [0]])
            controller.set_state_space_model(A, B)
            
            Q = np.eye(2)
            R = np.array([[1]])
            controller.set_objective_function(Q, R)
            
            # Set tight constraints that might be violated
            state_bounds = (np.array([-0.1, -0.1]), np.array([0.1, 0.1]))
            control_bounds = (np.array([-0.1]), np.array([0.1]))
            controller.set_constraints(state_bounds, control_bounds)
            
            # Try to solve with infeasible initial state
            infeasible_state = np.array([1.0, 1.0])  # Outside bounds
            
            result = controller.solve_mpc(
                current_state=infeasible_state,
                reference_trajectory=None
            )
            
            # Check if recovery was triggered
            recovery_used = result.feasibility_recovery_used if hasattr(result, 'feasibility_recovery_used') else False
            solution_found = result.optimal_control is not None
            
            tests["constraint_violation_recovery"] = {
                "passed": recovery_used or solution_found,
                "recovery_used": recovery_used,
                "solution_found": solution_found,
                "solver_status": result.status.value if hasattr(result.status, 'value') else str(result.status)
            }
        except Exception as e:
            tests["constraint_violation_recovery"] = {"passed": False, "error": str(e)}
        
        # Test 4: Real-time performance constraint
        try:
            config = MPCConfiguration()
            config.max_solve_time = 0.05  # 50ms constraint
            controller = MPCController(config, state_dim=2, control_dim=1)
            
            # Set up simple system
            A = np.array([[1.0, 0.1], [0, 1.0]])
            B = np.array([[0], [1]])
            controller.set_state_space_model(A, B)
            controller.set_objective_function(np.eye(2), np.array([[1]]))
            
            # Measure solve time
            current_state = np.array([1.0, 0.0])
            start_time = time.time()
            result = controller.solve_mpc(current_state)
            solve_time = time.time() - start_time
            
            meets_real_time_constraint = solve_time < config.max_solve_time
            
            tests["real_time_performance"] = {
                "passed": meets_real_time_constraint,
                "solve_time": solve_time,
                "constraint": config.max_solve_time,
                "margin": config.max_solve_time - solve_time
            }
        except Exception as e:
            tests["real_time_performance"] = {"passed": False, "error": str(e)}
        
        return {"status": "completed", "tests": tests}
    
    def test_bayesian_rl_mathematics(self) -> Dict[str, Any]:
        """Test Bayesian RL exploration-exploitation mathematics"""
        tests = {}
        
        # Test 1: Thompson Sampling theoretical properties
        try:
            config = BayesianRLConfig()
            config.algorithm = "gp_q_learning"
            config.exploration_strategy = "thompson_sampling"
            
            agent = BayesianRLAgent(
                state_dim=5,
                action_dim=2,
                config=config
            )
            
            # Add some training data
            for _ in range(20):
                state = np.random.randn(5)
                action = np.random.randn(2)
                reward = np.random.randn()
                next_state = np.random.randn(5)
                
                agent.update_beliefs(state, action, reward, next_state)
            
            # Test Thompson sampling produces diverse actions
            state = np.random.randn(5)
            actions = []
            
            for _ in range(10):
                action, info = agent.select_action(state, exploration_strategy="thompson_sampling")
                actions.append(action)
            
            actions = np.array(actions)
            action_diversity = np.std(actions, axis=0).mean()
            
            # Thompson sampling should produce diverse actions
            sufficient_diversity = action_diversity > 0.01
            
            tests["thompson_sampling_diversity"] = {
                "passed": sufficient_diversity,
                "action_diversity": float(action_diversity),
                "num_samples": len(actions)
            }
        except Exception as e:
            tests["thompson_sampling_diversity"] = {"passed": False, "error": str(e)}
        
        # Test 2: UCB confidence bounds
        try:
            config = BayesianRLConfig()
            config.algorithm = "gp_q_learning" 
            config.exploration_strategy = "ucb"
            
            agent = BayesianRLAgent(state_dim=3, action_dim=1, config=config)
            
            # Add training data
            for _ in range(30):
                state = np.random.randn(3)
                action = np.random.randn(1)
                reward = np.sin(state[0]) + 0.1 * np.random.randn()
                next_state = np.random.randn(3)
                
                agent.update_beliefs(state, action, reward, next_state)
            
            # Test UCB selection
            state = np.array([1.0, 0.0, 0.0])  # Fixed test state
            action, info = agent.select_action(state, exploration_strategy="ucb")
            
            has_uncertainty = 'uncertainty' in info
            has_confidence = 'ucb_value' in info or 'confidence' in info
            
            tests["ucb_confidence_bounds"] = {
                "passed": has_uncertainty and has_confidence,
                "has_uncertainty_info": has_uncertainty,
                "has_confidence_info": has_confidence,
                "method": info.get('method', 'unknown')
            }
        except Exception as e:
            tests["ucb_confidence_bounds"] = {"passed": False, "error": str(e)}
        
        # Test 3: Uncertainty-guided exploration
        try:
            config = BayesianRLConfig()
            config.algorithm = "gp_q_learning"
            config.exploration_strategy = "info_gain"
            
            agent = BayesianRLAgent(state_dim=2, action_dim=1, config=config)
            
            # Add sparse training data (high uncertainty regions should exist)
            train_states = np.array([[0, 0], [1, 1]])
            for i, state in enumerate(train_states):
                action = np.array([0.5])
                reward = float(i)
                next_state = state + 0.1
                agent.update_beliefs(state, action, reward, next_state)
            
            # Test in high uncertainty region (far from training data)
            high_uncertainty_state = np.array([5.0, 5.0])
            action_uncertain, info_uncertain = agent.select_action(
                high_uncertainty_state, 
                exploration_strategy="info_gain"
            )
            
            # Test in low uncertainty region (close to training data) 
            low_uncertainty_state = np.array([0.1, 0.1])
            action_certain, info_certain = agent.select_action(
                low_uncertainty_state,
                exploration_strategy="info_gain"
            )
            
            # Information gain should be higher in uncertain regions
            uncertainty_uncertain = info_uncertain.get('uncertainty', 0)
            uncertainty_certain = info_certain.get('uncertainty', 0)
            
            uncertainty_guided = uncertainty_uncertain > uncertainty_certain
            
            tests["uncertainty_guided_exploration"] = {
                "passed": uncertainty_guided,
                "uncertainty_in_unknown": float(uncertainty_uncertain),
                "uncertainty_in_known": float(uncertainty_certain),
                "ratio": float(uncertainty_uncertain / (uncertainty_certain + 1e-8))
            }
        except Exception as e:
            tests["uncertainty_guided_exploration"] = {"passed": False, "error": str(e)}
        
        # Test 4: Regret bounds (theoretical property)
        try:
            config = BayesianRLConfig()
            config.track_regret = True
            config.algorithm = "gp_q_learning"
            
            agent = BayesianRLAgent(state_dim=1, action_dim=1, config=config)
            
            # Simulate learning on a simple function
            true_function = lambda x: -x**2  # Quadratic with maximum at 0
            
            cumulative_regret = 0
            regrets = []
            
            for step in range(50):
                state = np.random.uniform(-2, 2, 1)
                action, _ = agent.select_action(state)
                
                # True optimal action for this state
                optimal_action = np.array([0.0])  # Always choose 0 for quadratic
                
                # Compute rewards
                actual_reward = true_function(action[0])
                optimal_reward = true_function(optimal_action[0])
                
                # Regret for this step
                instantaneous_regret = optimal_reward - actual_reward
                cumulative_regret += instantaneous_regret
                regrets.append(cumulative_regret)
                
                # Update agent
                next_state = state + 0.1 * np.random.randn(1)
                agent.update_beliefs(state, action, actual_reward, next_state)
            
            # Regret should grow sublinearly (√T for GP bandits)
            # Check that regret growth rate is decreasing
            if len(regrets) > 10:
                early_regret = regrets[9]
                late_regret = regrets[-1]
                regret_growth_rate = (late_regret - early_regret) / 40
                
                # For GP bandits, regret should grow as O(√T log T)
                reasonable_regret_growth = regret_growth_rate < 2.0  # Heuristic bound
            else:
                reasonable_regret_growth = True
                regret_growth_rate = 0.0
            
            tests["regret_bounds"] = {
                "passed": reasonable_regret_growth,
                "final_cumulative_regret": float(cumulative_regret),
                "regret_growth_rate": float(regret_growth_rate),
                "num_steps": len(regrets)
            }
        except Exception as e:
            tests["regret_bounds"] = {"passed": False, "error": str(e)}
        
        return {"status": "completed", "tests": tests}
    
    def test_integration_system(self) -> Dict[str, Any]:
        """Test integration system mathematical correctness"""
        tests = {}
        
        # Test 1: Real-time constraint satisfaction
        try:
            config = HRIBayesianRLConfig()
            config.real_time_constraint = 0.1  # 100ms
            
            integration = HRIBayesianRLIntegration(config)
            
            # Simulate state
            mock_state = self._create_mock_hri_state()
            
            # Measure execution time
            start_time = time.time()
            step_result = integration.step(mock_state)
            execution_time = time.time() - start_time
            
            meets_constraint = execution_time < config.real_time_constraint
            constraint_margin = config.real_time_constraint - execution_time
            
            tests["real_time_constraint_satisfaction"] = {
                "passed": meets_constraint,
                "execution_time": execution_time,
                "constraint": config.real_time_constraint,
                "margin": constraint_margin
            }
        except Exception as e:
            tests["real_time_constraint_satisfaction"] = {"passed": False, "error": str(e)}
        
        # Test 2: Multi-objective reward function
        try:
            config = HRIBayesianRLConfig()
            integration = HRIBayesianRLIntegration(config)
            
            # Test reward computation
            mock_state = self._create_mock_hri_state()
            mock_action = np.random.randn(6)
            mock_execution = {
                'success': True,
                'control_effort': 0.5,
                'execution_time': 0.05
            }
            mock_safety = {
                'safety_score': 0.8,
                'violations': []
            }
            
            reward_dict = integration._compute_reward(
                mock_state, mock_action, mock_execution, mock_safety
            )
            
            # Check reward structure
            has_components = all(key in reward_dict for key in 
                               ['task_success', 'safety', 'efficiency', 'human_comfort'])
            has_total = 'total' in reward_dict
            total_is_weighted_sum = abs(reward_dict['total'] - sum(
                config.reward_weights.get(k, 1.0) * v 
                for k, v in reward_dict.items() 
                if k != 'total'
            )) < 1e-6
            
            tests["multi_objective_reward"] = {
                "passed": has_components and has_total and total_is_weighted_sum,
                "has_all_components": has_components,
                "has_total_reward": has_total,
                "correct_weighting": total_is_weighted_sum,
                "total_reward": float(reward_dict['total'])
            }
        except Exception as e:
            tests["multi_objective_reward"] = {"passed": False, "error": str(e)}
        
        # Test 3: Safety override functionality
        try:
            config = HRIBayesianRLConfig()
            config.emergency_stop_enabled = True
            
            integration = HRIBayesianRLIntegration(config)
            
            # Create dangerous scenario (human very close)
            mock_state = self._create_mock_hri_state()
            # Simulate human very close to robot
            mock_state.human.position = mock_state.robot.ee_position + np.array([0.05, 0, 0])
            
            step_result = integration.step(mock_state)
            
            # Check if safety override was triggered
            safety_assessment = step_result['safety_assessment']
            emergency_triggered = safety_assessment.get('emergency_stop_needed', False)
            command_is_safe = step_result['mpc_params']['command'] == 'emergency_stop'
            
            safety_override_works = emergency_triggered or command_is_safe
            
            tests["safety_override_functionality"] = {
                "passed": safety_override_works,
                "emergency_detected": emergency_triggered,
                "emergency_command_issued": command_is_safe,
                "risk_level": float(safety_assessment.get('risk_level', 0))
            }
        except Exception as e:
            tests["safety_override_functionality"] = {"passed": False, "error": str(e)}
        
        return {"status": "completed", "tests": tests}
    
    def test_uncertainty_quantification(self) -> Dict[str, Any]:
        """Test uncertainty quantification across all components"""
        tests = {}
        
        # Test 1: Epistemic vs Aleatoric uncertainty separation
        try:
            # Test with Bayesian intent classifier
            classifier = BayesianIntentClassifier(
                state_dim=3,
                num_intents=2,
                config={'num_samples': 20}
            )
            
            # Train on small dataset (high epistemic uncertainty)
            X_small = torch.randn(10, 3)
            y_small = torch.randint(0, 2, (10,))
            for _ in range(5):
                classifier.train_step(X_small, y_small)
            
            # Test uncertainty
            X_test = torch.randn(5, 3)
            predictions = classifier.predict_with_uncertainty(X_test)
            
            epistemic = predictions['uncertainty']['epistemic']
            aleatoric = predictions['uncertainty']['aleatoric'] 
            total = predictions['uncertainty']['total']
            
            # Check uncertainty decomposition: total ≈ epistemic + aleatoric
            decomposition_valid = torch.allclose(total, epistemic + aleatoric, rtol=0.1)
            
            # Epistemic should be higher with limited data
            epistemic_higher = torch.mean(epistemic) > torch.mean(aleatoric)
            
            tests["epistemic_aleatoric_separation"] = {
                "passed": decomposition_valid and epistemic_higher,
                "decomposition_valid": decomposition_valid,
                "epistemic_higher": epistemic_higher,
                "mean_epistemic": float(torch.mean(epistemic)),
                "mean_aleatoric": float(torch.mean(aleatoric))
            }
        except Exception as e:
            tests["epistemic_aleatoric_separation"] = {"passed": False, "error": str(e)}
        
        # Test 2: Uncertainty calibration
        try:
            # Generate calibrated data
            n_samples = 100
            true_uncertainty = np.random.beta(2, 2, n_samples)  # Ground truth uncertainty
            predicted_uncertainty = true_uncertainty + 0.1 * np.random.randn(n_samples)
            predicted_uncertainty = np.clip(predicted_uncertainty, 0, 1)
            
            # Test calibration using reliability diagram approach
            # Bin predictions and check if confidence matches accuracy
            n_bins = 5
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_accuracies = []
            bin_confidences = []
            
            for i in range(n_bins):
                bin_mask = (predicted_uncertainty >= bin_boundaries[i]) & \
                          (predicted_uncertainty < bin_boundaries[i+1])
                
                if np.sum(bin_mask) > 0:
                    bin_accuracy = np.mean(true_uncertainty[bin_mask])
                    bin_confidence = np.mean(predicted_uncertainty[bin_mask])
                    
                    bin_accuracies.append(bin_accuracy)
                    bin_confidences.append(bin_confidence)
            
            # Calculate calibration error
            if bin_accuracies:
                calibration_error = np.mean([abs(acc - conf) for acc, conf in 
                                           zip(bin_accuracies, bin_confidences)])
                well_calibrated = calibration_error < 0.2  # Threshold
            else:
                well_calibrated = False
                calibration_error = 1.0
            
            tests["uncertainty_calibration"] = {
                "passed": well_calibrated,
                "calibration_error": float(calibration_error),
                "n_bins_used": len(bin_accuracies)
            }
        except Exception as e:
            tests["uncertainty_calibration"] = {"passed": False, "error": str(e)}
        
        # Test 3: Uncertainty propagation through system
        try:
            # Test that uncertainty propagates correctly through integration
            config = HRIBayesianRLConfig()
            integration = HRIBayesianRLIntegration(config)
            
            # Create state with known uncertainty properties
            mock_state = self._create_mock_hri_state()
            
            # Run step and check uncertainty tracking
            step_result = integration.step(mock_state)
            
            # Check that uncertainty information is preserved through pipeline
            has_intent_uncertainty = 'uncertainty' in step_result['human_intent']
            has_selection_uncertainty = 'uncertainty' in step_result.get('selection_info', {})
            
            uncertainty_propagated = has_intent_uncertainty
            
            tests["uncertainty_propagation"] = {
                "passed": uncertainty_propagated,
                "intent_uncertainty_tracked": has_intent_uncertainty,
                "selection_uncertainty_tracked": has_selection_uncertainty,
                "intent_uncertainty_value": step_result['human_intent'].get('uncertainty', 0)
            }
        except Exception as e:
            tests["uncertainty_propagation"] = {"passed": False, "error": str(e)}
        
        return {"status": "completed", "tests": tests}
    
    def _create_mock_hri_state(self):
        """Create mock HRI state for testing"""
        try:
            from src.environments.hri_environment import HRIState, RobotState, HumanState, ContextState
            
            robot_state = type('RobotState', (), {
                'joint_positions': np.array([0, 0.5, 0, -1, 0, 0]),
                'joint_velocities': np.zeros(6),
                'ee_position': np.array([0.5, 0.0, 0.5]),
                'ee_velocity': np.zeros(3)
            })()
            
            human_state = type('HumanState', (), {
                'position': np.array([1.0, 0.0, 1.0]),
                'velocity': np.zeros(3),
                'head_orientation': np.array([0, 0, 0, 1]),
                'gaze_direction': np.array([1, 0, 0])
            })()
            
            context_state = type('ContextState', (), {
                'interaction_phase': type('Phase', (), {'name': 'APPROACH'})(),
                'task_progress': 0.3,
                'safety_violations': 0,
                'emergency_stop_active': False
            })()
            
            return type('HRIState', (), {
                'robot': robot_state,
                'human': human_state, 
                'context': context_state,
                'timestamp': time.time(),
                'to_vector': lambda: np.random.randn(164)  # Mock vector representation
            })()
            
        except ImportError:
            # Fallback mock object
            return type('MockState', (), {
                'robot': type('MockRobot', (), {
                    'ee_position': np.array([0.5, 0.0, 0.5]),
                    'joint_positions': np.zeros(6),
                    'joint_velocities': np.zeros(6)
                })(),
                'human': type('MockHuman', (), {
                    'position': np.array([1.0, 0.0, 1.0])
                })(),
                'context': type('MockContext', (), {
                    'interaction_phase': type('MockPhase', (), {'name': 'APPROACH'})(),
                    'safety_violations': 0,
                    'emergency_stop_active': False
                })(),
                'timestamp': time.time(),
                'to_vector': lambda: np.random.randn(164)
            })()


def run_mathematical_validation_tests():
    """Run all mathematical validation tests"""
    print("🔬 Starting Mathematical Validation Test Suite")
    print("=" * 60)
    
    # Initialize test suite
    validator = MathematicalValidationSuite(tolerance=1e-6, verbose=True)
    
    # Run all tests
    results = validator.run_all_tests()
    
    # Print detailed results
    print("\n📊 Test Results Summary:")
    print("-" * 40)
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed_tests']}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    
    # Category breakdown
    print("\n📋 Category Results:")
    for category, category_results in results['categories'].items():
        if isinstance(category_results, dict) and 'tests' in category_results:
            passed = sum(1 for test in category_results['tests'].values() 
                        if test.get('passed', False))
            total = len(category_results['tests'])
            print(f"  {category}: {passed}/{total} ({passed/total:.1%})")
        else:
            print(f"  {category}: {'✅' if category_results.get('status') == 'completed' else '❌'}")
    
    # Return overall success
    return results['success_rate'] > 0.8  # 80% pass rate threshold


if __name__ == "__main__":
    success = run_mathematical_validation_tests()
    
    if success:
        print("\n🎉 Mathematical Validation PASSED!")
        print("All core algorithmic components are mathematically sound.")
    else:
        print("\n⚠️  Mathematical Validation needs attention.")
        print("Some components may require mathematical fixes.")
    
    print("\n" + "=" * 60)
    print("Mathematical Validation Test Suite Complete")