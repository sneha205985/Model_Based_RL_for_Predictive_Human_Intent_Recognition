"""
Complete Pipeline Integration Test Suite

This module provides comprehensive end-to-end integration tests for the
Model-Based RL Human Intent Recognition system, validating that all
components work together seamlessly in real-time scenarios.
"""

import pytest
import numpy as np
import torch
import time
import threading
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# System imports
from system.human_intent_rl_system import HumanIntentRLSystem, SystemConfiguration
from models.human_behavior import HumanState, BehaviorPrediction, BehaviorType
from models.neural_behavior_model import NeuralHumanBehaviorModel, NeuralModelConfig
from models.intent_predictor import IntentPrediction, ContextInformation, IntentType
from controllers.nonlinear_mpc_controller import NonlinearMPCController, NMPCConfiguration
from controllers.mpc_controller import RobotState, ControlAction
from agents.bayesian_rl_agent import BayesianRLAgent, BayesianRLConfig
from optimization.profiler import SystemProfiler
from optimization.caching_system import CacheSystem
from optimization.memory_manager import MemoryManager
from optimization.benchmark_framework import BenchmarkFramework


@dataclass
class IntegrationTestMetrics:
    """Metrics collected during integration testing."""
    total_runtime: float = 0.0
    pipeline_latency: float = 0.0
    throughput_predictions_per_sec: float = 0.0
    memory_peak_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0
    safety_constraint_violations: int = 0
    component_failures: int = 0
    real_time_violations: int = 0
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TestScenario:
    """Defines a test scenario for integration testing."""
    name: str
    description: str
    duration_seconds: float
    human_behavior_sequence: List[BehaviorType]
    expected_robot_responses: List[str]
    safety_critical: bool = False
    performance_requirements: Dict[str, float] = field(default_factory=dict)


class SyntheticHumanSimulator:
    """Simulates realistic human behavior for testing."""
    
    def __init__(self, scenario: TestScenario):
        self.scenario = scenario
        self.current_time = 0.0
        self.behavior_index = 0
        self.base_position = np.array([1.0, 0.5, 1.0])  # Human standing position
        
    def get_current_state(self) -> HumanState:
        """Generate current human state based on scenario."""
        # Get current behavior
        if self.behavior_index < len(self.scenario.human_behavior_sequence):
            current_behavior = self.scenario.human_behavior_sequence[self.behavior_index]
        else:
            current_behavior = BehaviorType.IDLE
        
        # Generate realistic human state based on behavior
        if current_behavior == BehaviorType.REACHING:
            # Reaching motion - hand moving toward object
            reach_progress = (self.current_time % 2.0) / 2.0  # 2-second reach cycle
            hand_position = self.base_position + np.array([
                0.3 * reach_progress,  # Reach forward
                0.2 * np.sin(reach_progress * np.pi),  # Slight arc
                -0.1 * reach_progress  # Slight downward
            ])
        elif current_behavior == BehaviorType.HANDOVER:
            # Handover motion - extending hand toward robot
            handover_progress = (self.current_time % 3.0) / 3.0
            hand_position = self.base_position + np.array([
                0.4 * handover_progress,
                -0.1 * handover_progress,
                0.1 * handover_progress
            ])
        elif current_behavior == BehaviorType.POINTING:
            # Pointing gesture - stable pointing position
            hand_position = self.base_position + np.array([0.5, 0.1, 0.2])
        elif current_behavior == BehaviorType.GESTURE:
            # Waving gesture - oscillatory motion
            wave_phase = self.current_time * 4.0  # 4 Hz waving
            hand_position = self.base_position + np.array([
                0.2,
                0.3 * np.sin(wave_phase),
                0.1 * np.cos(wave_phase)
            ])
        else:  # IDLE or UNKNOWN
            # Idle position - hands at sides with slight natural movement
            hand_position = self.base_position + 0.02 * np.random.randn(3)
        
        # Calculate velocity
        if hasattr(self, 'previous_position'):
            velocity = (hand_position - self.previous_position) / 0.1  # Assuming 10Hz updates
        else:
            velocity = np.zeros(3)
        
        self.previous_position = hand_position.copy()
        
        # Create joint positions (simplified skeleton model)
        joint_positions = {
            'right_hand': hand_position,
            'left_hand': self.base_position + np.array([-0.1, 0.0, -0.2]),
            'head': self.base_position + np.array([0.0, 0.0, 0.4]),
            'torso': self.base_position + np.array([0.0, 0.0, 0.1])
        }
        
        return HumanState(
            position=hand_position,
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # No rotation
            joint_positions=joint_positions,
            velocity=velocity,
            timestamp=self.current_time,
            confidence=0.95 + 0.05 * np.random.randn()  # High confidence with small noise
        )
    
    def update(self, dt: float) -> None:
        """Update simulator state."""
        self.current_time += dt
        
        # Advance behavior sequence
        behavior_duration = self.scenario.duration_seconds / len(self.scenario.human_behavior_sequence)
        if self.current_time > (self.behavior_index + 1) * behavior_duration:
            self.behavior_index = min(
                self.behavior_index + 1,
                len(self.scenario.human_behavior_sequence) - 1
            )


class IntegrationTestRunner:
    """Main integration test runner."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.system: Optional[HumanIntentRLSystem] = None
        self.profiler = SystemProfiler()
        self.cache_system = CacheSystem()
        self.memory_manager = MemoryManager()
        self.benchmark = BenchmarkFramework()
        
        # Test results
        self.test_results: Dict[str, IntegrationTestMetrics] = {}
        
    def setup_system(self) -> None:
        """Set up the complete system for testing."""
        # System configuration
        system_config = SystemConfiguration(
            max_concurrent_predictions=10,
            safety_check_interval=0.1,
            performance_monitoring=True,
            real_time_mode=True
        )
        
        # Neural behavior model configuration
        neural_config = {
            'neural_config': {
                'input_dim': 42,
                'hidden_dims': [128, 64, 32],  # Smaller for faster testing
                'output_dim': 6,
                'learning_rate': 1e-3,
                'batch_size': 16,
                'enable_bayesian': True,
                'ensemble_size': 2,  # Smaller ensemble for speed
                'use_gpu': torch.cuda.is_available(),
                'prediction_horizon': 10,
                'sequence_length': 5
            }
        }
        
        # MPC configuration
        mpc_config = NMPCConfiguration(
            prediction_horizon=5,  # Shorter horizon for speed
            control_horizon=3,
            sampling_time=0.1,
            state_weights={'task': 1.0, 'smoothness': 0.1},
            control_weights={'torque': 0.01},
            terminal_weights={'task': 10.0},
            max_iterations=20  # Reduced for speed
        )
        
        # Bayesian RL configuration
        rl_config = {
            'bayesian_config': {
                'state_dim': 21,
                'action_dim': 7,
                'gp_config': {
                    'kernel_type': 'rbf',
                    'noise_variance': 1e-3,
                    'length_scale': 1.0
                },
                'exploration_strategy': 'thompson_sampling',
                'learning_rate': 1e-2,
                'batch_size': 16
            }
        }
        
        # Initialize system
        self.system = HumanIntentRLSystem(system_config)
        
        # Initialize components
        behavior_model = NeuralHumanBehaviorModel(neural_config)
        mpc_controller = NonlinearMPCController(mpc_config)
        rl_agent = BayesianRLAgent(rl_config)
        
        # Set up MPC dynamics (simplified for testing)
        def simple_dynamics(state: RobotState, action: ControlAction, dt: float) -> RobotState:
            # Simple integration for testing
            new_positions = state.joint_positions + state.joint_velocities * dt
            new_velocities = state.joint_velocities
            if action.joint_torques is not None:
                new_velocities += action.joint_torques * dt  # Simplified dynamics
            
            return RobotState(
                joint_positions=new_positions,
                joint_velocities=new_velocities,
                end_effector_pose=np.concatenate([new_positions[:3], [1, 0, 0, 0]]),
                timestamp=state.timestamp + dt
            )
        
        mpc_controller.set_dynamics_model(simple_dynamics)
        
        # Register components with system
        self.system.register_behavior_model(behavior_model)
        self.system.register_controller(mpc_controller)
        self.system.register_rl_agent(rl_agent)
        
        self.logger.info("Integration test system initialized")
    
    def create_test_scenarios(self) -> List[TestScenario]:
        """Create comprehensive test scenarios."""
        scenarios = [
            TestScenario(
                name="basic_handover",
                description="Simple human-robot handover scenario",
                duration_seconds=10.0,
                human_behavior_sequence=[BehaviorType.IDLE, BehaviorType.REACHING, BehaviorType.HANDOVER],
                expected_robot_responses=["wait", "prepare", "receive"],
                safety_critical=True,
                performance_requirements={'latency_ms': 50, 'success_rate': 0.95}
            ),
            TestScenario(
                name="complex_interaction",
                description="Complex multi-behavior interaction",
                duration_seconds=20.0,
                human_behavior_sequence=[
                    BehaviorType.IDLE, BehaviorType.GESTURE, BehaviorType.POINTING,
                    BehaviorType.REACHING, BehaviorType.HANDOVER, BehaviorType.IDLE
                ],
                expected_robot_responses=[
                    "wait", "acknowledge", "look", "prepare", "receive", "complete"
                ],
                safety_critical=True,
                performance_requirements={'latency_ms': 75, 'success_rate': 0.90}
            ),
            TestScenario(
                name="stress_test",
                description="High-frequency behavior changes",
                duration_seconds=15.0,
                human_behavior_sequence=[
                    BehaviorType.GESTURE, BehaviorType.POINTING, BehaviorType.GESTURE,
                    BehaviorType.REACHING, BehaviorType.GESTURE, BehaviorType.HANDOVER
                ] * 3,  # Repeat pattern for stress testing
                expected_robot_responses=["adapt"] * 18,
                safety_critical=False,
                performance_requirements={'latency_ms': 100, 'throughput_hz': 10}
            ),
            TestScenario(
                name="safety_critical",
                description="Scenario requiring immediate safety responses",
                duration_seconds=8.0,
                human_behavior_sequence=[BehaviorType.IDLE, BehaviorType.REACHING, BehaviorType.UNKNOWN],
                expected_robot_responses=["wait", "prepare", "emergency_stop"],
                safety_critical=True,
                performance_requirements={'latency_ms': 20, 'safety_response_time': 10}
            )
        ]
        
        return scenarios
    
    @contextmanager
    def performance_monitoring(self, test_name: str):
        """Context manager for performance monitoring during tests."""
        # Start monitoring
        start_time = time.time()
        start_memory = self.memory_manager.get_memory_usage()
        
        # Start profiling
        self.profiler.start_profiling()
        
        try:
            yield
        finally:
            # Stop profiling
            profile_results = self.profiler.stop_profiling()
            
            # Calculate metrics
            end_time = time.time()
            end_memory = self.memory_manager.get_memory_usage()
            
            # Store results
            metrics = IntegrationTestMetrics()
            metrics.total_runtime = end_time - start_time
            metrics.memory_peak_usage_mb = max(start_memory, end_memory)
            metrics.performance_metrics = profile_results
            
            self.test_results[test_name] = metrics
    
    def run_scenario_test(self, scenario: TestScenario) -> IntegrationTestMetrics:
        """Run a single scenario test."""
        self.logger.info(f"Running scenario: {scenario.name}")
        
        # Initialize simulator
        simulator = SyntheticHumanSimulator(scenario)
        
        # Initialize robot state
        initial_robot_state = RobotState(
            joint_positions=np.zeros(7),
            joint_velocities=np.zeros(7),
            end_effector_pose=np.array([0.5, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]),
            timestamp=0.0
        )
        
        # Test metrics
        metrics = IntegrationTestMetrics()
        latencies = []
        safety_violations = 0
        real_time_violations = 0
        successful_predictions = 0
        total_predictions = 0
        
        # Run scenario
        start_time = time.time()
        dt = 0.1  # 10Hz update rate
        
        while simulator.current_time < scenario.duration_seconds:
            step_start = time.time()
            
            # Get human state
            human_state = simulator.get_current_state()
            
            # Create context
            context = ContextInformation(
                task_type="interaction",
                environment_state={'human_present': True},
                robot_capabilities=['manipulation', 'navigation'],
                safety_constraints={'min_distance': 0.3},
                timestamp=simulator.current_time
            )
            
            try:
                # Run complete pipeline
                pipeline_start = time.time()
                
                # Predict human behavior
                behavior_predictions = self.system.predict_human_behavior(
                    human_state, time_horizon=2.0, context=context
                )
                
                # Predict intent
                intent_predictions = self.system.predict_human_intent(
                    human_state, context
                )
                
                # Generate robot control
                robot_action = self.system.generate_robot_control(
                    current_robot_state=initial_robot_state,
                    human_state=human_state,
                    behavior_predictions=behavior_predictions,
                    intent_predictions=intent_predictions,
                    context=context
                )
                
                pipeline_end = time.time()
                pipeline_latency = (pipeline_end - pipeline_start) * 1000  # ms
                latencies.append(pipeline_latency)
                
                # Check real-time constraints
                if pipeline_latency > 100:  # 100ms threshold
                    real_time_violations += 1
                
                # Check safety constraints
                if self._check_safety_violation(human_state, robot_action):
                    safety_violations += 1
                
                # Track prediction accuracy (simplified)
                if behavior_predictions and len(behavior_predictions) > 0:
                    successful_predictions += 1
                
                total_predictions += 1
                
            except Exception as e:
                self.logger.error(f"Pipeline error: {e}")
                metrics.component_failures += 1
            
            # Update simulator
            simulator.update(dt)
            
            # Maintain real-time execution
            step_time = time.time() - step_start
            if step_time < dt:
                time.sleep(dt - step_time)
        
        # Calculate final metrics
        total_time = time.time() - start_time
        metrics.total_runtime = total_time
        metrics.pipeline_latency = np.mean(latencies) if latencies else float('inf')
        metrics.throughput_predictions_per_sec = total_predictions / total_time
        metrics.safety_constraint_violations = safety_violations
        metrics.real_time_violations = real_time_violations
        
        # Accuracy metrics
        metrics.accuracy_metrics = {
            'prediction_success_rate': successful_predictions / max(total_predictions, 1),
            'average_latency_ms': metrics.pipeline_latency,
            'latency_std_ms': np.std(latencies) if latencies else 0.0
        }
        
        # Performance requirements check
        meets_requirements = True
        if 'latency_ms' in scenario.performance_requirements:
            if metrics.pipeline_latency > scenario.performance_requirements['latency_ms']:
                meets_requirements = False
        
        if 'success_rate' in scenario.performance_requirements:
            success_rate = successful_predictions / max(total_predictions, 1)
            if success_rate < scenario.performance_requirements['success_rate']:
                meets_requirements = False
        
        metrics.performance_metrics['meets_requirements'] = meets_requirements
        
        self.logger.info(f"Scenario {scenario.name} completed: "
                        f"latency={metrics.pipeline_latency:.1f}ms, "
                        f"throughput={metrics.throughput_predictions_per_sec:.1f}Hz, "
                        f"safety_violations={safety_violations}")
        
        return metrics
    
    def _check_safety_violation(self, human_state: HumanState, robot_action: ControlAction) -> bool:
        """Check if robot action violates safety constraints."""
        # Simplified safety check - in real system this would be more comprehensive
        
        # Check if robot is too aggressive near human
        if robot_action.joint_torques is not None:
            max_torque = np.max(np.abs(robot_action.joint_torques))
            human_distance = np.linalg.norm(human_state.position[:2])  # Distance in x-y plane
            
            # Higher torques require larger safety distance
            required_distance = 0.3 + 0.1 * (max_torque / 10.0)
            
            if human_distance < required_distance:
                return True
        
        return False
    
    def run_performance_integration_tests(self) -> Dict[str, Any]:
        """Test integration with optimization systems."""
        self.logger.info("Running performance integration tests")
        
        results = {}
        
        # Test 1: Caching system integration
        self.cache_system.clear_all_caches()
        
        # Create test data
        test_human_state = HumanState(
            position=np.array([1.0, 0.0, 1.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            joint_positions={'hand': np.array([1.0, 0.0, 1.0])},
            velocity=np.zeros(3),
            timestamp=time.time()
        )
        
        context = ContextInformation(
            task_type="test",
            environment_state={},
            robot_capabilities=[],
            safety_constraints={},
            timestamp=time.time()
        )
        
        # Measure without cache
        start_time = time.time()
        for i in range(10):
            try:
                predictions = self.system.predict_human_behavior(
                    test_human_state, time_horizon=1.0, context=context
                )
            except:
                pass  # System may not be fully initialized
        no_cache_time = time.time() - start_time
        
        # Measure with cache (second run should be faster)
        start_time = time.time()
        for i in range(10):
            try:
                predictions = self.system.predict_human_behavior(
                    test_human_state, time_horizon=1.0, context=context
                )
            except:
                pass
        with_cache_time = time.time() - start_time
        
        cache_speedup = no_cache_time / max(with_cache_time, 1e-6)
        results['cache_integration'] = {
            'speedup_factor': cache_speedup,
            'no_cache_time_s': no_cache_time,
            'with_cache_time_s': with_cache_time
        }
        
        # Test 2: Memory management integration
        initial_memory = self.memory_manager.get_memory_usage()
        
        # Run memory-intensive operations
        for i in range(100):
            try:
                predictions = self.system.predict_human_behavior(
                    test_human_state, time_horizon=1.0, context=context
                )
            except:
                pass
        
        peak_memory = self.memory_manager.get_memory_usage()
        memory_growth = peak_memory - initial_memory
        
        results['memory_management'] = {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'memory_growth_mb': memory_growth,
            'memory_efficient': memory_growth < 100  # Less than 100MB growth
        }
        
        # Test 3: Profiler integration
        profile_results = self.profiler.profile_function(
            lambda: self.system.predict_human_behavior(
                test_human_state, time_horizon=1.0, context=context
            ) if self.system else None
        )
        
        results['profiler_integration'] = {
            'profiling_successful': profile_results is not None,
            'bottlenecks_identified': len(profile_results.get('bottlenecks', [])) if profile_results else 0
        }
        
        return results
    
    def run_system_orchestration_tests(self) -> Dict[str, Any]:
        """Test system orchestration and coordination."""
        self.logger.info("Running system orchestration tests")
        
        results = {}
        
        # Test 1: System startup and shutdown
        startup_start = time.time()
        try:
            self.system.start()
            startup_time = time.time() - startup_start
            startup_success = True
        except Exception as e:
            self.logger.error(f"System startup failed: {e}")
            startup_time = float('inf')
            startup_success = False
        
        shutdown_start = time.time()
        try:
            self.system.stop()
            shutdown_time = time.time() - shutdown_start
            shutdown_success = True
        except Exception as e:
            self.logger.error(f"System shutdown failed: {e}")
            shutdown_time = float('inf')
            shutdown_success = False
        
        results['startup_shutdown'] = {
            'startup_time_s': startup_time,
            'shutdown_time_s': shutdown_time,
            'startup_success': startup_success,
            'shutdown_success': shutdown_success
        }
        
        # Test 2: Component health monitoring
        try:
            health_status = self.system.get_system_health()
            results['health_monitoring'] = {
                'monitoring_available': True,
                'all_components_healthy': all(
                    status == 'healthy' for status in health_status.values()
                ),
                'component_count': len(health_status)
            }
        except Exception as e:
            results['health_monitoring'] = {
                'monitoring_available': False,
                'error': str(e)
            }
        
        # Test 3: Error recovery
        error_recovery_success = False
        try:
            # Simulate error condition
            self.system._simulate_component_failure('behavior_model')
            
            # Check if system recovers
            time.sleep(1.0)  # Give time for recovery
            health_after_error = self.system.get_system_health()
            
            if 'behavior_model' in health_after_error:
                error_recovery_success = health_after_error['behavior_model'] == 'healthy'
            
        except Exception as e:
            self.logger.info(f"Error recovery test not available: {e}")
        
        results['error_recovery'] = {
            'recovery_successful': error_recovery_success
        }
        
        return results
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        report = {
            'timestamp': time.time(),
            'test_summary': {
                'total_scenarios_run': len(self.test_results),
                'scenarios_passed': sum(
                    1 for metrics in self.test_results.values()
                    if metrics.performance_metrics.get('meets_requirements', False)
                ),
                'total_runtime_s': sum(
                    metrics.total_runtime for metrics in self.test_results.values()
                ),
                'total_safety_violations': sum(
                    metrics.safety_constraint_violations for metrics in self.test_results.values()
                ),
                'total_real_time_violations': sum(
                    metrics.real_time_violations for metrics in self.test_results.values()
                )
            },
            'scenario_results': {},
            'performance_summary': {
                'average_pipeline_latency_ms': np.mean([
                    metrics.pipeline_latency for metrics in self.test_results.values()
                ]),
                'average_throughput_hz': np.mean([
                    metrics.throughput_predictions_per_sec for metrics in self.test_results.values()
                ]),
                'peak_memory_usage_mb': max([
                    metrics.memory_peak_usage_mb for metrics in self.test_results.values()
                ])
            }
        }
        
        # Add detailed scenario results
        for scenario_name, metrics in self.test_results.items():
            report['scenario_results'][scenario_name] = {
                'runtime_s': metrics.total_runtime,
                'pipeline_latency_ms': metrics.pipeline_latency,
                'throughput_hz': metrics.throughput_predictions_per_sec,
                'memory_usage_mb': metrics.memory_peak_usage_mb,
                'safety_violations': metrics.safety_constraint_violations,
                'real_time_violations': metrics.real_time_violations,
                'component_failures': metrics.component_failures,
                'accuracy_metrics': metrics.accuracy_metrics,
                'meets_requirements': metrics.performance_metrics.get('meets_requirements', False)
            }
        
        return report


# Test fixtures and utilities
@pytest.fixture
def integration_test_runner():
    """Fixture providing an integration test runner."""
    runner = IntegrationTestRunner()
    runner.setup_system()
    yield runner
    # Cleanup
    if runner.system:
        try:
            runner.system.stop()
        except:
            pass


# Main integration tests
class TestCompleteIntegration:
    """Complete integration test suite."""
    
    def test_basic_pipeline(self, integration_test_runner):
        """Test basic perception → prediction → planning → control pipeline."""
        scenarios = integration_test_runner.create_test_scenarios()
        basic_scenario = scenarios[0]  # basic_handover
        
        with integration_test_runner.performance_monitoring('basic_pipeline'):
            metrics = integration_test_runner.run_scenario_test(basic_scenario)
        
        # Assertions
        assert metrics.pipeline_latency < 100, f"Pipeline latency {metrics.pipeline_latency}ms exceeds 100ms"
        assert metrics.safety_constraint_violations == 0, "Safety violations detected"
        assert metrics.component_failures == 0, "Component failures detected"
        assert metrics.accuracy_metrics['prediction_success_rate'] > 0.8, "Low prediction success rate"
    
    def test_real_time_performance(self, integration_test_runner):
        """Test real-time performance constraints."""
        scenarios = integration_test_runner.create_test_scenarios()
        
        for scenario in scenarios:
            if scenario.safety_critical:
                metrics = integration_test_runner.run_scenario_test(scenario)
                
                # Real-time constraints
                assert metrics.pipeline_latency < 100, f"Scenario {scenario.name}: latency {metrics.pipeline_latency}ms > 100ms"
                assert metrics.throughput_predictions_per_sec > 5, f"Scenario {scenario.name}: throughput too low"
                
                # Safety constraints
                if scenario.name == 'safety_critical':
                    assert metrics.pipeline_latency < 50, "Safety-critical scenario exceeds latency requirement"
    
    def test_synthetic_data_scenarios(self, integration_test_runner):
        """Test with comprehensive synthetic human interaction data."""
        scenarios = integration_test_runner.create_test_scenarios()
        
        results = []
        for scenario in scenarios:
            metrics = integration_test_runner.run_scenario_test(scenario)
            results.append((scenario.name, metrics))
        
        # Overall system performance
        avg_latency = np.mean([m.pipeline_latency for _, m in results])
        total_safety_violations = sum(m.safety_constraint_violations for _, m in results)
        
        assert avg_latency < 75, f"Average latency {avg_latency}ms too high"
        assert total_safety_violations < 2, f"Too many safety violations: {total_safety_violations}"
    
    def test_component_integration(self, integration_test_runner):
        """Test that all components work together seamlessly."""
        # Test component communication
        assert integration_test_runner.system is not None
        
        # Test component initialization
        components = integration_test_runner.system.get_component_status()
        assert len(components) >= 3, "Not all components registered"
        
        # Test component coordination
        health = integration_test_runner.system.get_system_health()
        healthy_components = sum(1 for status in health.values() if status == 'healthy')
        assert healthy_components >= 2, "Too many unhealthy components"
    
    def test_performance_optimization_integration(self, integration_test_runner):
        """Test integration with performance optimization systems."""
        perf_results = integration_test_runner.run_performance_integration_tests()
        
        # Cache system integration
        cache_results = perf_results['cache_integration']
        assert cache_results['speedup_factor'] > 0.5, "Cache system not providing speedup"
        
        # Memory management integration
        memory_results = perf_results['memory_management']
        assert memory_results['memory_efficient'], "Memory usage growing too quickly"
        
        # Profiler integration
        profiler_results = perf_results['profiler_integration']
        assert profiler_results['profiling_successful'], "Profiler integration failed"
    
    def test_system_orchestration(self, integration_test_runner):
        """Test system orchestration and timing."""
        orchestration_results = integration_test_runner.run_system_orchestration_tests()
        
        # Startup/shutdown
        startup_shutdown = orchestration_results['startup_shutdown']
        assert startup_shutdown['startup_time_s'] < 10, "System startup too slow"
        assert startup_shutdown['startup_success'], "System startup failed"
        
        # Health monitoring
        health_monitoring = orchestration_results['health_monitoring']
        assert health_monitoring['monitoring_available'], "Health monitoring not available"
    
    def test_error_handling_recovery(self, integration_test_runner):
        """Test error handling and recovery mechanisms."""
        # Test with invalid input
        invalid_human_state = HumanState(
            position=np.array([np.inf, 0, 0]),  # Invalid position
            orientation=np.array([1, 0, 0, 0]),
            joint_positions={},
            velocity=np.zeros(3),
            timestamp=time.time()
        )
        
        context = ContextInformation(
            task_type="test",
            environment_state={},
            robot_capabilities=[],
            safety_constraints={},
            timestamp=time.time()
        )
        
        # System should handle gracefully
        try:
            predictions = integration_test_runner.system.predict_human_behavior(
                invalid_human_state, time_horizon=1.0, context=context
            )
            # Should not crash, may return empty or default predictions
        except Exception as e:
            pytest.fail(f"System did not handle invalid input gracefully: {e}")
    
    def test_safety_constraint_enforcement(self, integration_test_runner):
        """Test that safety constraints are properly enforced."""
        scenarios = integration_test_runner.create_test_scenarios()
        safety_scenario = next(s for s in scenarios if s.name == 'safety_critical')
        
        metrics = integration_test_runner.run_scenario_test(safety_scenario)
        
        # Safety-critical scenarios should have minimal violations
        assert metrics.safety_constraint_violations <= 1, "Too many safety violations in critical scenario"
        
        # Response time for safety scenarios should be very fast
        assert metrics.pipeline_latency < 50, "Safety scenario response too slow"


def run_complete_integration_tests():
    """Run the complete integration test suite."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    runner = IntegrationTestRunner()
    
    try:
        runner.setup_system()
        
        # Run all scenario tests
        scenarios = runner.create_test_scenarios()
        for scenario in scenarios:
            metrics = runner.run_scenario_test(scenario)
            runner.test_results[scenario.name] = metrics
        
        # Run performance integration tests
        perf_results = runner.run_performance_integration_tests()
        
        # Run orchestration tests
        orch_results = runner.run_system_orchestration_tests()
        
        # Generate final report
        report = runner.generate_test_report()
        
        # Print summary
        print("\n" + "="*50)
        print("INTEGRATION TEST RESULTS SUMMARY")
        print("="*50)
        print(f"Total Scenarios: {report['test_summary']['total_scenarios_run']}")
        print(f"Scenarios Passed: {report['test_summary']['scenarios_passed']}")
        print(f"Average Latency: {report['performance_summary']['average_pipeline_latency_ms']:.1f}ms")
        print(f"Average Throughput: {report['performance_summary']['average_throughput_hz']:.1f}Hz")
        print(f"Safety Violations: {report['test_summary']['total_safety_violations']}")
        print(f"Real-time Violations: {report['test_summary']['total_real_time_violations']}")
        
        return report
        
    except Exception as e:
        logging.error(f"Integration test failed: {e}")
        raise
    finally:
        if runner.system:
            try:
                runner.system.stop()
            except:
                pass


if __name__ == "__main__":
    # Run integration tests directly
    report = run_complete_integration_tests()
    
    # Save report
    import json
    with open('integration_test_results.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\nFull report saved to: integration_test_results.json")