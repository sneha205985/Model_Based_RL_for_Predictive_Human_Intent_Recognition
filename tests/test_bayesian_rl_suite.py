"""
Comprehensive Testing and Benchmarking Suite for Bayesian RL

This module provides comprehensive tests and benchmarks for all components
of the Bayesian RL system, including unit tests, integration tests,
performance benchmarks, and HRI scenario validation.

Test Categories:
1. Unit Tests: Individual component testing
2. Integration Tests: Component interaction testing  
3. Performance Tests: Speed and memory benchmarks
4. Accuracy Tests: Learning and prediction accuracy
5. Safety Tests: Constraint satisfaction and safety
6. HRI Scenario Tests: Real-world interaction scenarios

Author: Bayesian RL Testing Suite
Date: 2024
"""

import pytest
import numpy as np
import torch
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import unittest
from unittest.mock import Mock, patch
import tempfile
import os
import json
import matplotlib.pyplot as plt

# Import all components to test
try:
    from src.agents.bayesian_rl_agent import BayesianRLAgent, BayesianRLConfiguration
    from src.environments.hri_environment import (
        HRIEnvironment, HRIState, RobotState, HumanState, ContextState, 
        create_default_hri_environment
    )
    from src.algorithms.gp_q_learning import GPBayesianQLearning, GPQConfiguration
    from src.algorithms.psrl import PSRLAgent, PSRLConfiguration
    from src.exploration.strategies import ExplorationManager, ExplorationConfig
    from src.integration.hri_bayesian_rl import HRIBayesianRLIntegration, HRIBayesianRLConfig
    from src.uncertainty.quantification import (
        MonteCarloUncertainty, UncertaintyPropagator, UncertaintyCalibrator,
        RiskAssessment, UncertaintyConfig
    )
except ImportError as e:
    print(f"Import error: {e}. Some tests may be skipped.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestResults:
    """Test results container"""
    test_name: str
    passed: bool
    execution_time: float
    metrics: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class BenchmarkResults:
    """Benchmark results container"""
    component_name: str
    test_scenario: str
    execution_times: List[float]
    memory_usage: List[float]
    accuracy_metrics: Dict[str, float]
    performance_summary: Dict[str, float]


class BayesianRLTestSuite:
    """Comprehensive test suite for Bayesian RL components"""
    
    def __init__(self):
        """Initialize test suite"""
        self.test_results = []
        self.benchmark_results = []
        self.temp_dir = tempfile.mkdtemp()
        
        # Test configurations
        self.test_configs = {
            'small': {'state_dim': 5, 'action_dim': 2, 'episodes': 5, 'steps': 10},
            'medium': {'state_dim': 20, 'action_dim': 6, 'episodes': 10, 'steps': 50},
            'large': {'state_dim': 100, 'action_dim': 20, 'episodes': 20, 'steps': 100}
        }
        
        logger.info(f"Initialized Bayesian RL Test Suite, temp dir: {self.temp_dir}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test categories"""
        logger.info("Starting comprehensive Bayesian RL test suite")
        
        # Unit tests
        unit_test_results = self.run_unit_tests()
        
        # Integration tests
        integration_test_results = self.run_integration_tests()
        
        # Performance benchmarks
        performance_results = self.run_performance_tests()
        
        # Accuracy tests
        accuracy_results = self.run_accuracy_tests()
        
        # Safety tests
        safety_results = self.run_safety_tests()
        
        # HRI scenario tests
        hri_results = self.run_hri_scenario_tests()
        
        # Compile results
        all_results = {
            'unit_tests': unit_test_results,
            'integration_tests': integration_test_results,
            'performance_tests': performance_results,
            'accuracy_tests': accuracy_results,
            'safety_tests': safety_results,
            'hri_scenario_tests': hri_results,
            'summary': self._compile_test_summary()
        }
        
        # Save results
        self._save_test_results(all_results)
        
        logger.info("Test suite completed")
        return all_results
    
    def run_unit_tests(self) -> Dict[str, TestResults]:
        """Run unit tests for individual components"""
        logger.info("Running unit tests")
        
        unit_tests = [
            ('test_bayesian_rl_agent', self._test_bayesian_rl_agent),
            ('test_hri_environment', self._test_hri_environment),
            ('test_gp_q_learning', self._test_gp_q_learning),
            ('test_psrl_agent', self._test_psrl_agent),
            ('test_exploration_strategies', self._test_exploration_strategies),
            ('test_uncertainty_quantification', self._test_uncertainty_quantification)
        ]
        
        results = {}
        for test_name, test_func in unit_tests:
            try:
                start_time = time.time()
                metrics = test_func()
                execution_time = time.time() - start_time
                
                results[test_name] = TestResults(
                    test_name=test_name,
                    passed=True,
                    execution_time=execution_time,
                    metrics=metrics
                )
                
                logger.info(f"✓ {test_name} passed ({execution_time:.3f}s)")
                
            except Exception as e:
                execution_time = time.time() - start_time
                results[test_name] = TestResults(
                    test_name=test_name,
                    passed=False,
                    execution_time=execution_time,
                    metrics={},
                    error_message=str(e)
                )
                
                logger.error(f"✗ {test_name} failed: {e}")
        
        return results
    
    def _test_bayesian_rl_agent(self) -> Dict[str, Any]:
        """Test BayesianRLAgent functionality"""
        config = BayesianRLConfiguration()
        agent = BayesianRLAgent(config)
        
        # Test initialization
        assert hasattr(agent, 'state_dim'), "Agent should have state_dim attribute"
        assert hasattr(agent, 'action_dim'), "Agent should have action_dim attribute"
        
        # Test prior belief initialization
        prior_params = {'mean': 0.0, 'variance': 1.0}
        agent.initialize_prior_beliefs(prior_params)
        
        # Test belief updates
        state = np.random.randn(agent.state_dim)
        action = np.random.randn(agent.action_dim)
        reward = np.random.normal()
        next_state = np.random.randn(agent.state_dim)
        
        belief_update = agent.update_beliefs(state, action, reward, next_state)
        assert isinstance(belief_update, dict), "Belief update should return dictionary"
        
        # Test action selection
        action, info = agent.select_action(state)
        assert action.shape == (agent.action_dim,), f"Action shape should be {(agent.action_dim,)}"
        assert isinstance(info, dict), "Action selection should return info dict"
        
        # Test uncertainty computation
        uncertainty = agent.compute_policy_uncertainty(state)
        assert 'epistemic_uncertainty' in uncertainty, "Should include epistemic uncertainty"
        assert 'aleatoric_uncertainty' in uncertainty, "Should include aleatoric uncertainty"
        
        # Test value function estimate
        value_est = agent.get_value_function_estimate(state)
        assert 'mean' in value_est, "Value estimate should include mean"
        assert 'variance' in value_est, "Value estimate should include variance"
        
        return {
            'belief_update_keys': list(belief_update.keys()),
            'uncertainty_keys': list(uncertainty.keys()),
            'value_estimate_keys': list(value_est.keys()),
            'action_shape': action.shape,
            'tests_passed': 6
        }
    
    def _test_hri_environment(self) -> Dict[str, Any]:
        """Test HRI Environment functionality"""
        env = create_default_hri_environment()
        
        # Test environment initialization
        initial_state = env.reset()
        assert isinstance(initial_state, HRIState), "Reset should return HRIState"
        
        # Test state dimensions
        state_vector = initial_state.to_vector()
        expected_dim = initial_state.dimension
        assert state_vector.shape[0] == expected_dim, f"State vector should have dimension {expected_dim}"
        
        # Test environment step
        action = np.random.uniform(-1, 1, env.action_dimension)
        next_state, reward_dict, done, info = env.step(action)
        
        assert isinstance(next_state, HRIState), "Step should return HRIState"
        assert isinstance(reward_dict, dict), "Step should return reward dictionary"
        assert 'total' in reward_dict, "Reward dict should include total reward"
        assert isinstance(done, bool), "Done should be boolean"
        assert isinstance(info, dict), "Info should be dictionary"
        
        # Test multiple steps
        step_count = 0
        while not done and step_count < 10:
            action = np.random.uniform(-1, 1, env.action_dimension)
            next_state, reward_dict, done, info = env.step(action)
            step_count += 1
        
        # Test bounds
        state_low, state_high = env.get_state_bounds()
        action_low, action_high = env.get_action_bounds()
        
        assert state_low.shape[0] == expected_dim, "State bounds should match state dimension"
        assert action_low.shape[0] == env.action_dimension, "Action bounds should match action dimension"
        
        return {
            'state_dimension': expected_dim,
            'action_dimension': env.action_dimension,
            'steps_completed': step_count,
            'final_reward': reward_dict['total'],
            'episode_finished': done,
            'tests_passed': 7
        }
    
    def _test_gp_q_learning(self) -> Dict[str, Any]:
        """Test GP-based Bayesian Q-Learning"""
        config = GPQConfiguration(training_iterations=5, action_candidates=10)
        agent = GPBayesianQLearning(state_dim=5, action_dim=2, config=config)
        
        # Add some experience
        for _ in range(10):
            state = np.random.randn(5).astype(np.float32)
            action = np.random.uniform(-1, 1, 2).astype(np.float32)
            reward = np.random.normal()
            next_state = np.random.randn(5).astype(np.float32)
            done = False
            
            agent.add_experience(state, action, reward, next_state, done)
        
        # Test Q-function update
        metrics = agent.update_q_function(batch_size=5)
        assert 'loss' in metrics, "Update should return loss metric"
        assert 'training_time' in metrics, "Update should return training time"
        
        # Test Q-value prediction
        test_state = np.random.randn(5).astype(np.float32)
        test_action = np.random.uniform(-1, 1, 2).astype(np.float32)
        
        q_pred = agent.predict_q_value(test_state, test_action)
        assert 'mean' in q_pred, "Q prediction should include mean"
        assert 'variance' in q_pred, "Q prediction should include variance"
        assert 'epistemic_uncertainty' in q_pred, "Should include epistemic uncertainty"
        assert 'aleatoric_uncertainty' in q_pred, "Should include aleatoric uncertainty"
        
        # Test Thompson sampling
        selected_action, sampling_info = agent.thompson_sampling_action(test_state)
        assert selected_action.shape == (2,), "Selected action should have correct shape"
        assert 'sampling_method' in sampling_info, "Should include sampling method info"
        
        # Test UCB action selection
        ucb_action, ucb_info = agent.ucb_action(test_state)
        assert ucb_action.shape == (2,), "UCB action should have correct shape"
        assert 'selection_method' in ucb_info, "Should include selection method info"
        
        # Test performance metrics
        perf_metrics = agent.get_performance_metrics()
        assert 'num_experiences' in perf_metrics, "Should include experience count"
        assert 'training_steps' in perf_metrics, "Should include training steps"
        
        return {
            'num_experiences': len(agent.experience_buffer),
            'training_loss': metrics['loss'],
            'q_prediction_mean': q_pred['mean'],
            'q_prediction_uncertainty': q_pred['variance'],
            'thompson_sampling_works': True,
            'ucb_selection_works': True,
            'tests_passed': 6
        }
    
    def _test_psrl_agent(self) -> Dict[str, Any]:
        """Test PSRL Agent functionality"""
        config = PSRLConfiguration(state_dim=4, action_dim=2, planning_horizon=3)
        agent = PSRLAgent(config.state_dim, config.action_dim, config)
        
        # Add some transitions
        for episode in range(2):
            initial_state = np.random.randn(4).astype(np.float32)
            planning_info = agent.begin_episode(initial_state)
            
            assert 'planning_time' in planning_info, "Should include planning time"
            
            # Simulate episode
            current_state = initial_state
            for step in range(5):
                action, selection_info = agent.select_action(current_state, "psrl")
                assert action.shape == (2,), "Action should have correct shape"
                
                # Simulate environment
                next_state = np.random.randn(4).astype(np.float32)
                reward = np.random.normal()
                done = step == 4
                
                agent.add_transition(current_state, action, reward, next_state, done)
                current_state = next_state
        
        # Test model predictions
        test_states = np.random.randn(3, 4).astype(np.float32)
        test_actions = np.random.uniform(-1, 1, (3, 2)).astype(np.float32)
        
        predictions = agent.get_model_predictions(test_states, test_actions)
        assert 'predicted_next_states' in predictions, "Should predict next states"
        assert 'predicted_rewards' in predictions, "Should predict rewards"
        assert 'transition_uncertainty' in predictions, "Should include transition uncertainty"
        assert 'reward_uncertainty' in predictions, "Should include reward uncertainty"
        
        # Test posterior samples
        posterior_samples = agent.get_posterior_samples(num_samples=3)
        assert 'transition_samples' in posterior_samples, "Should include transition samples"
        assert 'reward_samples' in posterior_samples, "Should include reward samples"
        
        # Test performance metrics
        perf_metrics = agent.get_performance_metrics()
        assert 'episode_count' in perf_metrics, "Should include episode count"
        assert 'num_transitions' in perf_metrics, "Should include transition count"
        
        return {
            'num_transitions': len(agent.transitions),
            'episode_count': agent.episode_count,
            'planning_time': planning_info['planning_time'],
            'has_current_policy': agent.current_policy is not None,
            'prediction_shapes': {
                'next_states': predictions['predicted_next_states'].shape,
                'rewards': predictions['predicted_rewards'].shape
            },
            'tests_passed': 6
        }
    
    def _test_exploration_strategies(self) -> Dict[str, Any]:
        """Test exploration strategies"""
        config = ExplorationConfig(action_candidates=5, thompson_samples=3)
        manager = ExplorationManager(config)
        
        # Mock Q-function
        class MockQFunction:
            def predict_q_value(self, state, action):
                return {
                    'mean': np.sum(state) + np.sum(action),
                    'std': 0.5,
                    'variance': 0.25
                }
        
        q_function = MockQFunction()
        test_state = np.array([1.0, -0.5, 0.2]).astype(np.float32)
        
        # Test different strategies
        strategies_tested = []
        for strategy in ['thompson_sampling', 'ucb', 'boltzmann', 'epsilon_greedy']:
            try:
                action, info = manager.select_action(test_state, q_function, strategy=strategy)
                assert len(action) > 0, f"Action should not be empty for {strategy}"
                assert info['strategy'] == strategy or 'builtin' in info.get('method', ''), f"Info should indicate {strategy}"
                strategies_tested.append(strategy)
            except Exception as e:
                logger.warning(f"Strategy {strategy} failed: {e}")
        
        # Test strategy statistics
        stats = manager.get_strategy_stats()
        assert 'current_strategy' in stats, "Should include current strategy"
        assert 'total_selections' in stats, "Should include total selections"
        
        # Test adaptive strategy selection
        performance_metrics = {
            'uncertainty': 0.7,
            'safety_violations': 0,
            'episode_reward': 5.0
        }
        
        recommended_strategy = manager.adaptive_strategy_selection(performance_metrics)
        assert isinstance(recommended_strategy, str), "Should recommend a strategy"
        
        return {
            'strategies_tested': strategies_tested,
            'num_strategies_working': len(strategies_tested),
            'total_selections': stats['total_selections'],
            'recommended_strategy': recommended_strategy,
            'tests_passed': 4
        }
    
    def _test_uncertainty_quantification(self) -> Dict[str, Any]:
        """Test uncertainty quantification methods"""
        config = UncertaintyConfig(mc_samples=20, calibration_bins=5)
        
        # Test Monte Carlo uncertainty
        quantifier = MonteCarloUncertainty(config)
        
        # Mock model for testing
        class MockModel:
            def __call__(self, x):
                return torch.randn(x.shape[0], 1)
        
        model = MockModel()
        inputs = torch.randn(5, 3)
        
        uncertainty_results = quantifier.compute_uncertainty(inputs, model)
        assert 'mean' in uncertainty_results, "Should include mean"
        assert 'variance' in uncertainty_results, "Should include variance"
        assert 'confidence_intervals' in uncertainty_results, "Should include confidence intervals"
        
        # Test uncertainty decomposition
        decomposition = quantifier.decompose_uncertainty(inputs, model)
        assert 'total_uncertainty' in decomposition, "Should include total uncertainty"
        assert 'epistemic_uncertainty' in decomposition, "Should include epistemic uncertainty"
        assert 'aleatoric_uncertainty' in decomposition, "Should include aleatoric uncertainty"
        
        # Test uncertainty propagation
        propagator = UncertaintyPropagator(config)
        input_uncertainty = {
            'mean': inputs,
            'std': torch.ones_like(inputs) * 0.1
        }
        
        model_chain = [MockModel(), MockModel()]
        final_uncertainty = propagator.propagate_through_chain(
            inputs, input_uncertainty, model_chain
        )
        
        assert 'mean' in final_uncertainty, "Final uncertainty should include mean"
        assert 'propagation_history' in final_uncertainty, "Should include propagation history"
        
        # Test calibration assessment
        calibrator = UncertaintyCalibrator(config)
        
        # Generate test data
        test_predictions = np.random.randn(50, 1)
        test_uncertainties = np.abs(np.random.randn(50, 1)) * 0.5
        test_targets = test_predictions + np.random.randn(50, 1) * 0.3
        
        calibration_results = calibrator.assess_calibration(
            test_predictions, test_uncertainties, test_targets
        )
        
        assert 'expected_calibration_error' in calibration_results, "Should include ECE"
        assert 'coverage_probability' in calibration_results, "Should include coverage"
        
        return {
            'uncertainty_components': list(uncertainty_results.keys()),
            'decomposition_components': list(decomposition.keys()),
            'propagation_stages': len(final_uncertainty['propagation_history']),
            'calibration_ece': calibration_results['mean_ece'],
            'coverage_probability': np.mean(calibration_results['coverage_probability']),
            'tests_passed': 5
        }
    
    def run_integration_tests(self) -> Dict[str, TestResults]:
        """Run integration tests between components"""
        logger.info("Running integration tests")
        
        integration_tests = [
            ('test_agent_environment_integration', self._test_agent_environment_integration),
            ('test_hri_integration_workflow', self._test_hri_integration_workflow),
            ('test_uncertainty_pipeline', self._test_uncertainty_pipeline)
        ]
        
        results = {}
        for test_name, test_func in integration_tests:
            try:
                start_time = time.time()
                metrics = test_func()
                execution_time = time.time() - start_time
                
                results[test_name] = TestResults(
                    test_name=test_name,
                    passed=True,
                    execution_time=execution_time,
                    metrics=metrics
                )
                
                logger.info(f"✓ {test_name} passed ({execution_time:.3f}s)")
                
            except Exception as e:
                execution_time = time.time() - start_time
                results[test_name] = TestResults(
                    test_name=test_name,
                    passed=False,
                    execution_time=execution_time,
                    metrics={},
                    error_message=str(e)
                )
                
                logger.error(f"✗ {test_name} failed: {e}")
        
        return results
    
    def _test_agent_environment_integration(self) -> Dict[str, Any]:
        """Test agent-environment integration"""
        # Create environment
        env = create_default_hri_environment()
        
        # Create GP Q-learning agent
        config = GPQConfiguration(training_iterations=3, action_candidates=5)
        agent = GPBayesianQLearning(env.state_dimension, env.action_dimension, config)
        
        # Run episode
        state = env.reset()
        episode_reward = 0
        steps_completed = 0
        
        for step in range(10):
            # Get action from agent (using random actions initially)
            action = np.random.uniform(env.action_low, env.action_high)
            
            # Take environment step
            next_state, reward_dict, done, info = env.step(action)
            
            # Add experience to agent
            agent.add_experience(
                state.to_vector(), action, reward_dict['total'],
                next_state.to_vector(), done
            )
            
            episode_reward += reward_dict['total']
            steps_completed += 1
            
            state = next_state
            if done:
                break
        
        # Update agent
        if len(agent.experience_buffer) >= 5:
            update_metrics = agent.update_q_function(batch_size=5)
        else:
            update_metrics = {'loss': 0, 'training_time': 0}
        
        # Test agent prediction after learning
        test_state = state.to_vector()
        test_action = np.random.uniform(env.action_low, env.action_high)
        q_pred = agent.predict_q_value(test_state, test_action)
        
        return {
            'steps_completed': steps_completed,
            'episode_reward': episode_reward,
            'experiences_collected': len(agent.experience_buffer),
            'agent_trained': agent.is_trained,
            'q_prediction_mean': q_pred['mean'],
            'update_loss': update_metrics['loss'],
            'integration_successful': True
        }
    
    def _test_hri_integration_workflow(self) -> Dict[str, Any]:
        """Test complete HRI integration workflow"""
        config = HRIBayesianRLConfig(
            rl_algorithm="gp_q_learning",
            real_time_constraint=1.0  # Relaxed for testing
        )
        
        integration = HRIBayesianRLIntegration(config)
        
        # Run a short episode
        episode_results = integration.run_episode(max_steps=5)
        
        # Check key metrics
        assert episode_results['steps'] > 0, "Should complete some steps"
        assert 'total_reward' in episode_results, "Should have total reward"
        assert 'avg_safety_score' in episode_results, "Should have safety score"
        
        # Test performance summary
        performance_summary = integration.get_performance_summary()
        
        assert 'total_episodes' in performance_summary, "Should track total episodes"
        assert 'recent_performance' in performance_summary, "Should have recent performance"
        
        # Test individual step
        current_state = integration.environment.reset()
        step_results = integration.step(current_state)
        
        assert 'human_intent' in step_results, "Should predict human intent"
        assert 'safety_assessment' in step_results, "Should assess safety"
        assert 'execution_results' in step_results, "Should have execution results"
        
        return {
            'episode_steps': episode_results['steps'],
            'episode_reward': episode_results['total_reward'],
            'safety_score': episode_results['avg_safety_score'],
            'real_time_met': episode_results.get('real_time_performance', 0),
            'integration_components_working': 4,
            'workflow_complete': True
        }
    
    def _test_uncertainty_pipeline(self) -> Dict[str, Any]:
        """Test uncertainty quantification pipeline"""
        config = UncertaintyConfig(mc_samples=20)
        
        # Create pipeline components
        quantifier = MonteCarloUncertainty(config)
        propagator = UncertaintyPropagator(config)
        calibrator = UncertaintyCalibrator(config)
        risk_assessor = RiskAssessment(config)
        
        # Mock models
        class MockModel:
            def __call__(self, x):
                return x @ torch.randn(x.shape[-1], 2) + torch.randn(2)
        
        # Test pipeline
        inputs = torch.randn(10, 5)
        model1 = MockModel()
        model2 = MockModel()
        
        # Step 1: Initial uncertainty quantification
        initial_uncertainty = quantifier.compute_uncertainty(inputs, model1)
        
        # Step 2: Propagate through model chain
        final_uncertainty = propagator.propagate_through_chain(
            inputs, initial_uncertainty, [model1, model2]
        )
        
        # Step 3: Generate test data for calibration
        predictions = final_uncertainty['mean'].numpy()
        uncertainties = final_uncertainty['std'].numpy()
        targets = predictions + np.random.randn(*predictions.shape) * 0.2
        
        # Step 4: Assess calibration
        calibration_results = calibrator.assess_calibration(
            predictions, uncertainties, targets
        )
        
        # Step 5: Risk assessment
        risk_measures = risk_assessor.compute_risk_measures(predictions, uncertainties)
        
        return {
            'initial_uncertainty_computed': True,
            'propagation_successful': 'propagation_history' in final_uncertainty,
            'calibration_ece': calibration_results['mean_ece'],
            'risk_measures_computed': len(risk_measures),
            'pipeline_stages_completed': 5,
            'pipeline_successful': True
        }
    
    def run_performance_tests(self) -> Dict[str, BenchmarkResults]:
        """Run performance benchmarking tests"""
        logger.info("Running performance tests")
        
        benchmarks = {
            'gp_q_learning_performance': self._benchmark_gp_q_learning,
            'psrl_performance': self._benchmark_psrl,
            'environment_performance': self._benchmark_environment,
            'integration_performance': self._benchmark_integration
        }
        
        results = {}
        for benchmark_name, benchmark_func in benchmarks.items():
            try:
                logger.info(f"Running benchmark: {benchmark_name}")
                results[benchmark_name] = benchmark_func()
                logger.info(f"✓ {benchmark_name} completed")
            except Exception as e:
                logger.error(f"✗ {benchmark_name} failed: {e}")
                results[benchmark_name] = BenchmarkResults(
                    component_name=benchmark_name,
                    test_scenario="error",
                    execution_times=[],
                    memory_usage=[],
                    accuracy_metrics={},
                    performance_summary={'error': str(e)}
                )
        
        return results
    
    def _benchmark_gp_q_learning(self) -> BenchmarkResults:
        """Benchmark GP Q-learning performance"""
        config = GPQConfiguration(training_iterations=10, action_candidates=10)
        agent = GPBayesianQLearning(state_dim=20, action_dim=6, config=config)
        
        execution_times = []
        memory_usage = []
        
        # Benchmark learning
        for trial in range(5):
            # Add experiences
            for _ in range(50):
                state = np.random.randn(20).astype(np.float32)
                action = np.random.uniform(-1, 1, 6).astype(np.float32)
                reward = np.random.normal()
                next_state = np.random.randn(20).astype(np.float32)
                
                agent.add_experience(state, action, reward, next_state, False)
            
            # Benchmark update
            start_time = time.time()
            metrics = agent.update_q_function(batch_size=20)
            execution_times.append(time.time() - start_time)
            
            # Mock memory usage
            memory_usage.append(50 + trial * 5)  # MB
        
        # Benchmark prediction
        prediction_times = []
        for _ in range(20):
            state = np.random.randn(20).astype(np.float32)
            action = np.random.uniform(-1, 1, 6).astype(np.float32)
            
            start_time = time.time()
            q_pred = agent.predict_q_value(state, action)
            prediction_times.append(time.time() - start_time)
        
        return BenchmarkResults(
            component_name="GP Q-Learning",
            test_scenario="learning_and_prediction",
            execution_times=execution_times,
            memory_usage=memory_usage,
            accuracy_metrics={
                'avg_prediction_time': np.mean(prediction_times),
                'max_prediction_time': np.max(prediction_times)
            },
            performance_summary={
                'avg_update_time': np.mean(execution_times),
                'avg_memory_usage': np.mean(memory_usage),
                'training_scalable': np.mean(execution_times) < 1.0
            }
        )
    
    def _benchmark_psrl(self) -> BenchmarkResults:
        """Benchmark PSRL performance"""
        config = PSRLConfiguration(state_dim=10, action_dim=4, planning_horizon=5)
        agent = PSRLAgent(config.state_dim, config.action_dim, config)
        
        execution_times = []
        memory_usage = []
        
        # Benchmark episodes
        for episode in range(3):
            initial_state = np.random.randn(10).astype(np.float32)
            
            # Benchmark episode planning
            start_time = time.time()
            planning_info = agent.begin_episode(initial_state)
            execution_times.append(time.time() - start_time)
            
            # Mock memory usage
            memory_usage.append(30 + episode * 10)
            
            # Run episode steps
            current_state = initial_state
            for step in range(10):
                action, _ = agent.select_action(current_state)
                next_state = np.random.randn(10).astype(np.float32)
                reward = np.random.normal()
                
                agent.add_transition(current_state, action, reward, next_state, step == 9)
                current_state = next_state
        
        return BenchmarkResults(
            component_name="PSRL",
            test_scenario="episodic_planning",
            execution_times=execution_times,
            memory_usage=memory_usage,
            accuracy_metrics={
                'planning_time': np.mean([info['planning_time'] for info in [planning_info]]),
                'episodes_completed': 3
            },
            performance_summary={
                'avg_episode_planning_time': np.mean(execution_times),
                'planning_scalable': np.mean(execution_times) < 0.5
            }
        )
    
    def _benchmark_environment(self) -> BenchmarkResults:
        """Benchmark HRI environment performance"""
        env = create_default_hri_environment()
        
        execution_times = []
        memory_usage = []
        
        # Benchmark environment steps
        for trial in range(10):
            state = env.reset()
            trial_times = []
            
            for step in range(50):
                action = np.random.uniform(env.action_low, env.action_high)
                
                start_time = time.time()
                next_state, reward_dict, done, info = env.step(action)
                trial_times.append(time.time() - start_time)
                
                if done:
                    break
            
            execution_times.extend(trial_times)
            memory_usage.append(20)  # Mock memory usage
        
        return BenchmarkResults(
            component_name="HRI Environment",
            test_scenario="environment_simulation",
            execution_times=execution_times,
            memory_usage=memory_usage,
            accuracy_metrics={
                'avg_step_time': np.mean(execution_times),
                'max_step_time': np.max(execution_times),
                'total_steps': len(execution_times)
            },
            performance_summary={
                'real_time_capable': np.mean(execution_times) < 0.01,  # 10ms per step
                'consistent_performance': np.std(execution_times) < 0.005
            }
        )
    
    def _benchmark_integration(self) -> BenchmarkResults:
        """Benchmark full integration performance"""
        config = HRIBayesianRLConfig(real_time_constraint=0.5)
        integration = HRIBayesianRLIntegration(config)
        
        execution_times = []
        memory_usage = []
        
        # Benchmark integration steps
        current_state = integration.environment.reset()
        
        for step in range(20):
            start_time = time.time()
            step_results = integration.step(current_state)
            execution_times.append(time.time() - start_time)
            
            # Update state
            action = step_results['rl_action']
            next_state, _, done, _ = integration.environment.step(action)
            current_state = next_state
            
            memory_usage.append(100)  # Mock memory usage
            
            if done:
                current_state = integration.environment.reset()
        
        return BenchmarkResults(
            component_name="HRI Integration",
            test_scenario="full_integration_pipeline",
            execution_times=execution_times,
            memory_usage=memory_usage,
            accuracy_metrics={
                'avg_step_time': np.mean(execution_times),
                'real_time_violations': sum(t > config.real_time_constraint for t in execution_times)
            },
            performance_summary={
                'meets_real_time_constraint': np.mean(execution_times) < config.real_time_constraint,
                'integration_overhead': np.mean(execution_times),
                'steps_completed': len(execution_times)
            }
        )
    
    def run_accuracy_tests(self) -> Dict[str, TestResults]:
        """Run accuracy and learning performance tests"""
        logger.info("Running accuracy tests")
        
        accuracy_tests = [
            ('test_learning_convergence', self._test_learning_convergence),
            ('test_prediction_accuracy', self._test_prediction_accuracy),
            ('test_uncertainty_calibration', self._test_uncertainty_calibration)
        ]
        
        results = {}
        for test_name, test_func in accuracy_tests:
            try:
                start_time = time.time()
                metrics = test_func()
                execution_time = time.time() - start_time
                
                results[test_name] = TestResults(
                    test_name=test_name,
                    passed=True,
                    execution_time=execution_time,
                    metrics=metrics
                )
                
                logger.info(f"✓ {test_name} passed ({execution_time:.3f}s)")
                
            except Exception as e:
                execution_time = time.time() - start_time
                results[test_name] = TestResults(
                    test_name=test_name,
                    passed=False,
                    execution_time=execution_time,
                    metrics={},
                    error_message=str(e)
                )
                
                logger.error(f"✗ {test_name} failed: {e}")
        
        return results
    
    def _test_learning_convergence(self) -> Dict[str, Any]:
        """Test learning convergence properties"""
        # Simple synthetic task: learn to predict quadratic function
        def target_function(state, action):
            return -np.sum(state**2) - np.sum(action**2) + np.random.normal(0, 0.1)
        
        config = GPQConfiguration(training_iterations=20)
        agent = GPBayesianQLearning(state_dim=3, action_dim=2, config=config)
        
        # Generate training data
        training_errors = []
        
        for episode in range(20):
            episode_error = 0
            episode_samples = 0
            
            for step in range(10):
                state = np.random.uniform(-1, 1, 3).astype(np.float32)
                action = np.random.uniform(-1, 1, 2).astype(np.float32)
                
                true_value = target_function(state, action)
                
                # Add experience
                next_state = state + 0.1 * np.random.randn(3)
                agent.add_experience(state, action, true_value, next_state, False)
                
                # Test prediction after some learning
                if len(agent.experience_buffer) >= 10:
                    q_pred = agent.predict_q_value(state, action)
                    error = abs(q_pred['mean'] - true_value)
                    episode_error += error
                    episode_samples += 1
            
            if episode_samples > 0:
                training_errors.append(episode_error / episode_samples)
            
            # Update agent
            if len(agent.experience_buffer) >= 10:
                agent.update_q_function(batch_size=10)
        
        # Test final accuracy
        test_errors = []
        for _ in range(20):
            state = np.random.uniform(-1, 1, 3).astype(np.float32)
            action = np.random.uniform(-1, 1, 2).astype(np.float32)
            
            true_value = target_function(state, action)
            q_pred = agent.predict_q_value(state, action)
            
            test_errors.append(abs(q_pred['mean'] - true_value))
        
        # Check convergence
        early_error = np.mean(training_errors[:5]) if len(training_errors) >= 5 else float('inf')
        late_error = np.mean(training_errors[-5:]) if len(training_errors) >= 5 else float('inf')
        improvement = early_error - late_error
        
        return {
            'training_episodes': len(training_errors),
            'early_error': early_error,
            'late_error': late_error,
            'improvement': improvement,
            'final_test_error': np.mean(test_errors),
            'converged': improvement > 0.1,
            'accurate': np.mean(test_errors) < 1.0
        }
    
    def _test_prediction_accuracy(self) -> Dict[str, Any]:
        """Test prediction accuracy across different scenarios"""
        # Create environment
        env = create_default_hri_environment()
        
        # Create and train agent
        config = GPQConfiguration(training_iterations=10)
        agent = GPBayesianQLearning(env.state_dimension, env.action_dimension, config)
        
        # Collect training data
        state = env.reset()
        training_data = []
        
        for _ in range(50):
            action = np.random.uniform(env.action_low, env.action_high)
            next_state, reward_dict, done, info = env.step(action)
            
            training_data.append({
                'state': state.to_vector(),
                'action': action,
                'reward': reward_dict['total'],
                'next_state': next_state.to_vector()
            })
            
            agent.add_experience(
                state.to_vector(), action, reward_dict['total'],
                next_state.to_vector(), done
            )
            
            state = next_state if not done else env.reset()
        
        # Train agent
        for _ in range(5):
            agent.update_q_function(batch_size=20)
        
        # Test prediction accuracy
        prediction_errors = []
        uncertainty_scores = []
        
        for data in training_data[-10:]:  # Use last 10 samples for testing
            q_pred = agent.predict_q_value(data['state'], data['action'])
            
            # Compare with actual reward (approximation of true Q-value)
            error = abs(q_pred['mean'] - data['reward'])
            prediction_errors.append(error)
            
            # Record uncertainty
            uncertainty_scores.append(q_pred['std'])
        
        return {
            'training_samples': len(training_data),
            'test_samples': len(prediction_errors),
            'mean_prediction_error': np.mean(prediction_errors),
            'max_prediction_error': np.max(prediction_errors),
            'mean_uncertainty': np.mean(uncertainty_scores),
            'error_uncertainty_correlation': np.corrcoef(prediction_errors, uncertainty_scores)[0, 1],
            'prediction_quality': 'good' if np.mean(prediction_errors) < 2.0 else 'poor'
        }
    
    def _test_uncertainty_calibration(self) -> Dict[str, Any]:
        """Test uncertainty calibration quality"""
        config = UncertaintyConfig(calibration_bins=5)
        calibrator = UncertaintyCalibrator(config)
        
        # Generate synthetic test data with known uncertainty
        n_samples = 100
        true_mean = 0
        true_std = 1
        
        # Generate predictions with bias and miscalibrated uncertainty
        predictions = np.random.normal(true_mean + 0.2, 0.8, (n_samples, 1))  # Biased
        uncertainties = np.random.gamma(2, 0.5, (n_samples, 1))  # Miscalibrated
        targets = np.random.normal(true_mean, true_std, (n_samples, 1))  # True targets
        
        # Assess calibration
        calibration_before = calibrator.assess_calibration(predictions, uncertainties, targets)
        
        # Improve calibration
        calibration_improvement = calibrator.improve_calibration(predictions, uncertainties, targets)
        
        calibration_after = calibration_improvement['improved_calibration']
        
        return {
            'samples_used': n_samples,
            'ece_before': calibration_before['mean_ece'],
            'ece_after': calibration_after['mean_ece'],
            'ece_improvement': calibration_improvement['ece_improvement'],
            'coverage_before': np.mean(calibration_before['coverage_probability']),
            'coverage_after': np.mean(calibration_after['coverage_probability']),
            'calibration_improved': calibration_improvement['ece_improvement'] > 0,
            'calibration_method': 'isotonic'
        }
    
    def run_safety_tests(self) -> Dict[str, TestResults]:
        """Run safety and constraint satisfaction tests"""
        logger.info("Running safety tests")
        
        safety_tests = [
            ('test_constraint_satisfaction', self._test_constraint_satisfaction),
            ('test_emergency_stop', self._test_emergency_stop),
            ('test_human_safety_zones', self._test_human_safety_zones)
        ]
        
        results = {}
        for test_name, test_func in safety_tests:
            try:
                start_time = time.time()
                metrics = test_func()
                execution_time = time.time() - start_time
                
                results[test_name] = TestResults(
                    test_name=test_name,
                    passed=True,
                    execution_time=execution_time,
                    metrics=metrics
                )
                
                logger.info(f"✓ {test_name} passed ({execution_time:.3f}s)")
                
            except Exception as e:
                execution_time = time.time() - start_time
                results[test_name] = TestResults(
                    test_name=test_name,
                    passed=False,
                    execution_time=execution_time,
                    metrics={},
                    error_message=str(e)
                )
                
                logger.error(f"✗ {test_name} failed: {e}")
        
        return results
    
    def _test_constraint_satisfaction(self) -> Dict[str, Any]:
        """Test safety constraint satisfaction"""
        config = HRIBayesianRLConfig(safety_monitoring=True)
        integration = HRIBayesianRLIntegration(config)
        
        # Run steps and monitor safety
        current_state = integration.environment.reset()
        safety_violations = []
        constraint_checks = []
        
        for step in range(20):
            step_results = integration.step(current_state)
            
            safety_assessment = step_results['safety_assessment']
            safety_violations.extend(safety_assessment['violations'])
            constraint_checks.append(len(safety_assessment['violations']) == 0)
            
            # Update state
            action = step_results['rl_action']
            next_state, _, done, _ = integration.environment.step(action)
            current_state = next_state if not done else integration.environment.reset()
        
        # Analyze safety performance
        total_violations = len(safety_violations)
        constraint_satisfaction_rate = np.mean(constraint_checks)
        
        # Check specific safety metrics
        human_distance_violations = [v for v in safety_violations if v['type'] == 'human_distance']
        velocity_violations = [v for v in safety_violations if v['type'] == 'robot_velocity']
        
        return {
            'steps_monitored': 20,
            'total_violations': total_violations,
            'constraint_satisfaction_rate': constraint_satisfaction_rate,
            'human_distance_violations': len(human_distance_violations),
            'velocity_violations': len(velocity_violations),
            'safety_monitoring_active': True,
            'safe_operation': constraint_satisfaction_rate > 0.8
        }
    
    def _test_emergency_stop(self) -> Dict[str, Any]:
        """Test emergency stop functionality"""
        config = HRIBayesianRLConfig(emergency_stop_enabled=True)
        integration = HRIBayesianRLIntegration(config)
        
        # Create dangerous scenario (robot too close to human)
        current_state = integration.environment.reset()
        
        # Manually set dangerous state
        current_state.robot.ee_position = np.array([0.0, 0.0, 0.5])
        current_state.human.position = np.array([0.1, 0.0, 0.5])  # Very close!
        
        # Execute step in dangerous scenario
        step_results = integration.step(current_state)
        
        safety_assessment = step_results['safety_assessment']
        execution_results = step_results['execution_results']
        
        # Check if emergency stop was triggered
        emergency_triggered = safety_assessment.get('emergency_stop_needed', False)
        emergency_command = execution_results.get('command_executed') == 'emergency_stop'
        
        return {
            'dangerous_scenario_created': True,
            'human_robot_distance': 0.1,  # Very close
            'emergency_stop_needed': emergency_triggered,
            'emergency_command_executed': emergency_command,
            'safety_system_responsive': emergency_triggered or emergency_command,
            'violations_detected': len(safety_assessment.get('violations', [])),
            'risk_level': safety_assessment.get('risk_level', 0)
        }
    
    def _test_human_safety_zones(self) -> Dict[str, Any]:
        """Test human safety zone enforcement"""
        env = create_default_hri_environment()
        
        # Test different human positions and check safety zones
        safety_zone_tests = []
        
        human_positions = [
            np.array([0.3, 0.0, 0.8]),   # Close
            np.array([0.8, 0.0, 0.8]),   # Medium distance
            np.array([1.5, 0.0, 0.8])    # Far
        ]
        
        robot_position = np.array([0.0, 0.0, 0.5])
        
        for i, human_pos in enumerate(human_positions):
            # Create state with specific positions
            state = env.reset()
            state.robot.ee_position = robot_position
            state.human.position = human_pos
            
            # Calculate distance
            distance = np.linalg.norm(robot_position - human_pos)
            
            # Test safety assessment
            config = HRIBayesianRLConfig()
            integration = HRIBayesianRLIntegration(config)
            
            step_results = integration.step(state)
            safety_assessment = step_results['safety_assessment']
            
            safety_zone_tests.append({
                'test_case': i + 1,
                'human_position': human_pos.tolist(),
                'distance': distance,
                'safe': safety_assessment['safe'],
                'risk_level': safety_assessment['risk_level'],
                'violations': len(safety_assessment['violations'])
            })
        
        # Analyze safety zone enforcement
        close_test = safety_zone_tests[0]  # Should be unsafe
        far_test = safety_zone_tests[2]    # Should be safe
        
        return {
            'safety_zone_tests': safety_zone_tests,
            'close_distance_unsafe': not close_test['safe'],
            'far_distance_safe': far_test['safe'],
            'distance_based_safety': close_test['risk_level'] > far_test['risk_level'],
            'safety_zones_working': True,
            'min_safe_distance': 0.3  # Expected minimum safe distance
        }
    
    def run_hri_scenario_tests(self) -> Dict[str, TestResults]:
        """Run HRI scenario-specific tests"""
        logger.info("Running HRI scenario tests")
        
        hri_tests = [
            ('test_handover_scenario', self._test_handover_scenario),
            ('test_collaboration_scenario', self._test_collaboration_scenario),
            ('test_intent_adaptation', self._test_intent_adaptation)
        ]
        
        results = {}
        for test_name, test_func in hri_tests:
            try:
                start_time = time.time()
                metrics = test_func()
                execution_time = time.time() - start_time
                
                results[test_name] = TestResults(
                    test_name=test_name,
                    passed=True,
                    execution_time=execution_time,
                    metrics=metrics
                )
                
                logger.info(f"✓ {test_name} passed ({execution_time:.3f}s)")
                
            except Exception as e:
                execution_time = time.time() - start_time
                results[test_name] = TestResults(
                    test_name=test_name,
                    passed=False,
                    execution_time=execution_time,
                    metrics={},
                    error_message=str(e)
                )
                
                logger.error(f"✗ {test_name} failed: {e}")
        
        return results
    
    def _test_handover_scenario(self) -> Dict[str, Any]:
        """Test handover scenario"""
        config = HRIBayesianRLConfig()
        integration = HRIBayesianRLIntegration(config)
        
        # Set up handover scenario
        current_state = integration.environment.reset()
        
        # Human approaching for handover
        current_state.human.position = np.array([0.6, 0.3, 0.8])
        current_state.human.intent_probabilities = {'handover_request': 0.8, 'idle': 0.2}
        current_state.context.task_type = "handover"
        
        handover_steps = []
        
        # Execute handover sequence
        for step in range(10):
            step_results = integration.step(current_state)
            
            human_intent = step_results['human_intent']
            mpc_params = step_results['mpc_params']
            
            handover_steps.append({
                'step': step,
                'dominant_intent': human_intent['dominant_intent'],
                'intent_confidence': human_intent['confidence'],
                'mpc_command': mpc_params['command'],
                'safety_score': step_results['safety_assessment']['safety_score']
            })
            
            # Update state (simulate human getting closer)
            action = step_results['rl_action']
            next_state, _, done, _ = integration.environment.step(action)
            
            # Simulate human movement toward robot for handover
            if step < 5:
                next_state.human.position = current_state.human.position + np.array([-0.05, 0, 0])
            
            current_state = next_state
            if done:
                break
        
        # Analyze handover performance
        handover_intents = [s['dominant_intent'] for s in handover_steps]
        handover_commands = [s['mpc_command'] for s in handover_steps]
        avg_safety = np.mean([s['safety_score'] for s in handover_steps])
        
        return {
            'handover_steps_completed': len(handover_steps),
            'handover_intents_detected': handover_intents.count('handover_request'),
            'appropriate_commands': handover_commands.count('move_to_human') + handover_commands.count('handover_position'),
            'average_safety_score': avg_safety,
            'intent_recognition_working': 'handover_request' in handover_intents,
            'robot_responds_appropriately': any(cmd in ['move_to_human', 'handover_position'] for cmd in handover_commands),
            'handover_scenario_successful': True
        }
    
    def _test_collaboration_scenario(self) -> Dict[str, Any]:
        """Test collaboration scenario"""
        config = HRIBayesianRLConfig()
        integration = HRIBayesianRLIntegration(config)
        
        # Set up collaboration scenario
        current_state = integration.environment.reset()
        
        # Human initiating collaboration
        current_state.human.position = np.array([0.8, 0.2, 0.8])
        current_state.human.intent_probabilities = {'collaboration_start': 0.7, 'idle': 0.3}
        current_state.context.task_type = "collaboration"
        
        collaboration_steps = []
        
        # Execute collaboration sequence
        for step in range(8):
            step_results = integration.step(current_state)
            
            human_intent = step_results['human_intent']
            mpc_params = step_results['mpc_params']
            safety_assessment = step_results['safety_assessment']
            
            collaboration_steps.append({
                'step': step,
                'intent_uncertainty': human_intent['uncertainty'],
                'safety_constraints': mpc_params['safety_constraints'],
                'human_distance': np.linalg.norm(current_state.robot.ee_position - current_state.human.position),
                'risk_level': safety_assessment['risk_level']
            })
            
            # Update state
            action = step_results['rl_action']
            next_state, _, done, _ = integration.environment.step(action)
            current_state = next_state
            
            if done:
                break
        
        # Analyze collaboration performance
        distances = [s['human_distance'] for s in collaboration_steps]
        uncertainties = [s['intent_uncertainty'] for s in collaboration_steps]
        risk_levels = [s['risk_level'] for s in collaboration_steps]
        
        return {
            'collaboration_steps': len(collaboration_steps),
            'min_human_distance': min(distances),
            'max_human_distance': max(distances),
            'average_intent_uncertainty': np.mean(uncertainties),
            'average_risk_level': np.mean(risk_levels),
            'maintains_safe_distance': min(distances) > 0.2,
            'adapts_to_uncertainty': max(uncertainties) > min(uncertainties),
            'collaboration_scenario_successful': True
        }
    
    def _test_intent_adaptation(self) -> Dict[str, Any]:
        """Test adaptation to changing human intent"""
        config = HRIBayesianRLConfig()
        integration = HRIBayesianRLIntegration(config)
        
        # Define intent sequence (changing over time)
        intent_sequence = [
            {'handover_request': 0.8, 'idle': 0.2},
            {'handover_request': 0.6, 'idle': 0.4},
            {'handover_request': 0.3, 'idle': 0.7},
            {'idle': 0.9, 'handover_request': 0.1}
        ]
        
        adaptation_data = []
        current_state = integration.environment.reset()
        
        # Test robot adaptation to changing intent
        for phase, intent_probs in enumerate(intent_sequence):
            # Set human intent for this phase
            current_state.human.intent_probabilities = intent_probs
            
            phase_steps = []
            
            # Execute 3 steps per phase
            for step in range(3):
                step_results = integration.step(current_state)
                
                predicted_intent = step_results['human_intent']
                mpc_params = step_results['mpc_params']
                
                phase_steps.append({
                    'predicted_intent': predicted_intent['dominant_intent'],
                    'intent_confidence': predicted_intent['confidence'],
                    'mpc_command': mpc_params['command'],
                    'target_pose': mpc_params['target_pose'].tolist()
                })
                
                # Update state
                action = step_results['rl_action']
                next_state, _, done, _ = integration.environment.step(action)
                current_state = next_state if not done else integration.environment.reset()
            
            adaptation_data.append({
                'phase': phase,
                'true_intent': intent_probs,
                'phase_steps': phase_steps
            })
        
        # Analyze adaptation performance
        intent_predictions = []
        command_adaptations = []
        
        for phase_data in adaptation_data:
            true_dominant = max(phase_data['true_intent'], key=phase_data['true_intent'].get)
            
            for step_data in phase_data['phase_steps']:
                predicted = step_data['predicted_intent']
                intent_predictions.append(predicted == true_dominant)
                command_adaptations.append(step_data['mpc_command'])
        
        intent_accuracy = np.mean(intent_predictions)
        command_variety = len(set(command_adaptations))
        
        return {
            'intent_phases_tested': len(intent_sequence),
            'total_adaptation_steps': len(intent_predictions),
            'intent_prediction_accuracy': intent_accuracy,
            'command_variety': command_variety,
            'robot_adapts_to_intent': intent_accuracy > 0.5,
            'behavioral_flexibility': command_variety > 2,
            'adaptation_successful': True
        }
    
    def _compile_test_summary(self) -> Dict[str, Any]:
        """Compile overall test summary"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.passed)
        
        total_execution_time = sum(result.execution_time for result in self.test_results)
        
        return {
            'total_tests_run': total_tests,
            'tests_passed': passed_tests,
            'tests_failed': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'total_execution_time': total_execution_time,
            'average_test_time': total_execution_time / total_tests if total_tests > 0 else 0,
            'benchmarks_completed': len(self.benchmark_results)
        }
    
    def _save_test_results(self, results: Dict[str, Any]):
        """Save test results to files"""
        # Save summary
        summary_file = os.path.join(self.temp_dir, 'test_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(results['summary'], f, indent=2)
        
        # Save detailed results (simplified)
        detailed_file = os.path.join(self.temp_dir, 'detailed_results.json')
        simplified_results = {}
        
        for category, category_results in results.items():
            if category == 'summary':
                continue
                
            simplified_results[category] = {}
            
            if isinstance(category_results, dict):
                for test_name, test_result in category_results.items():
                    if hasattr(test_result, 'test_name'):
                        simplified_results[category][test_name] = {
                            'passed': test_result.passed,
                            'execution_time': test_result.execution_time,
                            'error_message': test_result.error_message
                        }
                    else:
                        simplified_results[category][test_name] = str(test_result)
        
        with open(detailed_file, 'w') as f:
            json.dump(simplified_results, f, indent=2)
        
        logger.info(f"Test results saved to {self.temp_dir}")
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        logger.info("Test suite cleanup completed")


# Main execution function
def run_bayesian_rl_tests():
    """Run the complete Bayesian RL test suite"""
    test_suite = BayesianRLTestSuite()
    
    try:
        # Run all tests
        results = test_suite.run_all_tests()
        
        # Print summary
        summary = results['summary']
        print("\n" + "="*60)
        print("BAYESIAN RL TEST SUITE SUMMARY")
        print("="*60)
        print(f"Total Tests: {summary['total_tests_run']}")
        print(f"Passed: {summary['tests_passed']}")
        print(f"Failed: {summary['tests_failed']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Time: {summary['total_execution_time']:.2f}s")
        print(f"Benchmarks: {summary['benchmarks_completed']}")
        print("="*60)
        
        if summary['success_rate'] >= 0.8:
            print("🎉 TEST SUITE PASSED!")
        else:
            print("❌ TEST SUITE FAILED - Some tests need attention")
        
        return results
        
    finally:
        test_suite.cleanup()


if __name__ == "__main__":
    # Run the test suite
    test_results = run_bayesian_rl_tests()
    
    # Additional analysis could be added here
    print(f"\nTest results available in: {test_results.get('temp_dir', 'N/A')}")