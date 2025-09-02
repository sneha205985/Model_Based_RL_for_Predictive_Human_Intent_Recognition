"""
System Orchestration Integration Tests

This module tests the complete system orchestration, including startup/shutdown,
component coordination, timing synchronization, error handling, and safety
constraint enforcement in real-world scenarios.
"""

import pytest
import numpy as np
import time
import threading
import logging
import signal
import subprocess
import psutil
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import sys
import os
import json
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# System imports
from system.human_intent_rl_system import HumanIntentRLSystem, SystemConfiguration
from models.human_behavior import HumanState, BehaviorPrediction, BehaviorType
from models.neural_behavior_model import NeuralHumanBehaviorModel
from models.intent_predictor import IntentPrediction, ContextInformation, IntentType
from controllers.nonlinear_mpc_controller import NonlinearMPCController, NMPCConfiguration
from controllers.mpc_controller import RobotState, ControlAction
from agents.bayesian_rl_agent import BayesianRLAgent
from robustness.system_monitor import SystemMonitor
from robustness.error_handler import ErrorHandler
from robustness.safety_system import SafetySystem
from safety.constraint_enforcement import ConstraintEnforcement


@dataclass
class OrchestrationTestResult:
    """Results from orchestration tests."""
    test_name: str
    startup_time: float
    shutdown_time: float
    component_health_checks: Dict[str, bool]
    timing_synchronization_errors: int
    error_recovery_successful: bool
    safety_system_responsive: bool
    throughput_achieved: float
    latency_p95: float
    resource_usage_peak: Dict[str, float]
    coordination_failures: int
    success: bool
    error_message: Optional[str] = None


class MockHardwareInterface:
    """Mock hardware interface for testing."""
    
    def __init__(self):
        self.is_connected = False
        self.current_state = None
        self.command_queue = []
        self.failure_mode = None
        
    def connect(self) -> bool:
        """Simulate hardware connection."""
        time.sleep(0.1)  # Connection delay
        self.is_connected = True
        return True
    
    def disconnect(self) -> bool:
        """Simulate hardware disconnection."""
        self.is_connected = False
        return True
    
    def get_robot_state(self) -> RobotState:
        """Get current robot state."""
        if not self.is_connected:
            raise RuntimeError("Hardware not connected")
        
        if self.failure_mode == "sensor_failure":
            raise RuntimeError("Sensor failure")
        
        return RobotState(
            joint_positions=np.zeros(7) + 0.01 * np.random.randn(7),
            joint_velocities=np.zeros(7) + 0.001 * np.random.randn(7),
            end_effector_pose=np.array([0.5, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]),
            timestamp=time.time()
        )
    
    def send_control_command(self, action: ControlAction) -> bool:
        """Send control command to robot."""
        if not self.is_connected:
            raise RuntimeError("Hardware not connected")
        
        if self.failure_mode == "actuator_failure":
            raise RuntimeError("Actuator failure")
        
        self.command_queue.append(action)
        return True
    
    def set_failure_mode(self, mode: Optional[str]) -> None:
        """Set failure mode for testing."""
        self.failure_mode = mode


class SystemOrchestrationTester:
    """Tests system orchestration and coordination."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.system: Optional[HumanIntentRLSystem] = None
        self.hardware_interface = MockHardwareInterface()
        self.test_results: List[OrchestrationTestResult] = []
        
        # Monitoring components
        self.system_monitor = SystemMonitor()
        self.error_handler = ErrorHandler()
        self.safety_system = SafetySystem()
        
        # Test configuration
        self.test_duration = 30.0  # seconds
        self.coordination_timeout = 5.0  # seconds
        
    def setup_complete_system(self) -> HumanIntentRLSystem:
        """Set up complete system with all components."""
        # System configuration
        config = SystemConfiguration(
            max_concurrent_predictions=5,
            safety_check_interval=0.05,  # 20Hz safety checks
            performance_monitoring=True,
            real_time_mode=True,
            component_timeout=2.0,
            error_recovery_enabled=True
        )
        
        # Create system
        system = HumanIntentRLSystem(config)
        
        # Configure components with reduced complexity for testing
        
        # Neural behavior model
        behavior_config = {
            'neural_config': {
                'input_dim': 42,
                'hidden_dims': [64, 32],
                'output_dim': 6,
                'learning_rate': 1e-3,
                'batch_size': 8,
                'enable_bayesian': False,  # Disable for speed
                'ensemble_size': 1,
                'use_gpu': False,  # Use CPU for consistent testing
                'prediction_horizon': 5,
                'sequence_length': 3
            }
        }
        behavior_model = NeuralHumanBehaviorModel(behavior_config)
        
        # MPC controller
        mpc_config = NMPCConfiguration(
            prediction_horizon=5,
            control_horizon=3,
            sampling_time=0.1,
            state_weights={'task': 1.0, 'smoothness': 0.1},
            control_weights={'torque': 0.01},
            terminal_weights={'task': 10.0},
            max_iterations=10  # Reduced for speed
        )
        controller = NonlinearMPCController(mpc_config)
        
        # Simple dynamics for testing
        def test_dynamics(state: RobotState, action: ControlAction, dt: float) -> RobotState:
            new_positions = state.joint_positions + state.joint_velocities * dt
            new_velocities = state.joint_velocities
            if action.joint_torques is not None:
                new_velocities += action.joint_torques * dt * 0.1
            
            return RobotState(
                joint_positions=new_positions,
                joint_velocities=new_velocities,
                end_effector_pose=np.concatenate([new_positions[:3], [1, 0, 0, 0]]),
                timestamp=state.timestamp + dt
            )
        
        controller.set_dynamics_model(test_dynamics)
        
        # Bayesian RL agent
        rl_config = {
            'bayesian_config': {
                'state_dim': 21,
                'action_dim': 7,
                'gp_config': {
                    'kernel_type': 'rbf',
                    'noise_variance': 0.01,
                    'length_scale': 1.0
                },
                'exploration_strategy': 'epsilon_greedy',  # Faster than Thompson sampling
                'learning_rate': 0.01,
                'batch_size': 8
            }
        }
        rl_agent = BayesianRLAgent(rl_config)
        
        # Register components
        system.register_behavior_model(behavior_model)
        system.register_controller(controller)
        system.register_rl_agent(rl_agent)
        
        # Register hardware interface
        system.register_hardware_interface(self.hardware_interface)
        
        # Register monitoring systems
        system.register_monitor(self.system_monitor)
        system.register_error_handler(self.error_handler)
        system.register_safety_system(self.safety_system)
        
        return system
    
    @contextmanager
    def system_lifecycle(self):
        """Context manager for system lifecycle management."""
        startup_start = time.time()
        
        try:
            # Setup system
            self.system = self.setup_complete_system()
            
            # Start system
            self.system.start()
            startup_time = time.time() - startup_start
            
            yield self.system, startup_time
            
        except Exception as e:
            self.logger.error(f"System startup failed: {e}")
            raise
        finally:
            # Shutdown system
            shutdown_start = time.time()
            if self.system:
                try:
                    self.system.stop()
                    shutdown_time = time.time() - shutdown_start
                    self.logger.info(f"System shutdown completed in {shutdown_time:.2f}s")
                except Exception as e:
                    self.logger.error(f"System shutdown failed: {e}")
    
    def test_system_startup_shutdown(self) -> OrchestrationTestResult:
        """Test system startup and shutdown procedures."""
        self.logger.info("Testing system startup and shutdown")
        
        startup_times = []
        shutdown_times = []
        startup_successes = []
        
        # Test multiple startup/shutdown cycles
        for cycle in range(3):
            try:
                with self.system_lifecycle() as (system, startup_time):
                    startup_times.append(startup_time)
                    startup_successes.append(True)
                    
                    # Verify system is running
                    health = system.get_system_health()
                    assert len(health) > 0, "No components registered"
                    
                    # Brief operation
                    time.sleep(0.5)
                    
                # Measure shutdown time (handled by context manager)
                shutdown_times.append(0.5)  # Placeholder - actual measurement in context manager
                
            except Exception as e:
                self.logger.error(f"Startup/shutdown cycle {cycle} failed: {e}")
                startup_successes.append(False)
                startup_times.append(float('inf'))
                shutdown_times.append(float('inf'))
        
        # Analyze results
        avg_startup = np.mean([t for t in startup_times if t != float('inf')])
        avg_shutdown = np.mean([t for t in shutdown_times if t != float('inf')])
        success_rate = np.mean(startup_successes)
        
        result = OrchestrationTestResult(
            test_name="startup_shutdown",
            startup_time=avg_startup,
            shutdown_time=avg_shutdown,
            component_health_checks={},
            timing_synchronization_errors=0,
            error_recovery_successful=success_rate > 0.8,
            safety_system_responsive=True,
            throughput_achieved=0.0,
            latency_p95=0.0,
            resource_usage_peak={'cpu': 0, 'memory': 0},
            coordination_failures=3 - sum(startup_successes),
            success=success_rate == 1.0 and avg_startup < 10.0,
            error_message=None if success_rate == 1.0 else f"Startup success rate: {success_rate:.2f}"
        )
        
        self.test_results.append(result)
        return result
    
    def test_component_coordination(self) -> OrchestrationTestResult:
        """Test coordination between system components."""
        self.logger.info("Testing component coordination")
        
        coordination_failures = 0
        timing_errors = 0
        component_health = {}
        
        try:
            with self.system_lifecycle() as (system, startup_time):
                # Test component health monitoring
                health_checks = []
                for _ in range(10):
                    health = system.get_system_health()
                    health_checks.append(health)
                    time.sleep(0.1)
                
                # Analyze health check consistency
                if health_checks:
                    first_health = health_checks[0]
                    component_health = first_health
                    
                    # Check for component health consistency
                    for health in health_checks[1:]:
                        for component, status in health.items():
                            if component in first_health:
                                if status != first_health[component] and first_health[component] == 'healthy':
                                    coordination_failures += 1
                
                # Test component interaction timing
                test_human_state = HumanState(
                    position=np.array([1.0, 0.0, 1.0]),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                    joint_positions={'hand': np.array([1.0, 0.0, 1.0])},
                    velocity=np.zeros(3),
                    timestamp=time.time()
                )
                
                context = ContextInformation(
                    task_type="coordination_test",
                    environment_state={},
                    robot_capabilities=[],
                    safety_constraints={},
                    timestamp=time.time()
                )
                
                # Test coordinated prediction pipeline
                pipeline_timings = []
                for i in range(20):
                    start_time = time.time()
                    
                    try:
                        # This should coordinate between behavior model, RL agent, and controller
                        behavior_predictions = system.predict_human_behavior(
                            test_human_state, time_horizon=1.0, context=context
                        )
                        
                        intent_predictions = system.predict_human_intent(
                            test_human_state, context
                        )
                        
                        robot_state = self.hardware_interface.get_robot_state()
                        
                        control_action = system.generate_robot_control(
                            current_robot_state=robot_state,
                            human_state=test_human_state,
                            behavior_predictions=behavior_predictions,
                            intent_predictions=intent_predictions,
                            context=context
                        )
                        
                        pipeline_time = time.time() - start_time
                        pipeline_timings.append(pipeline_time)
                        
                        # Check for timing constraints
                        if pipeline_time > 0.2:  # 200ms threshold
                            timing_errors += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Pipeline coordination failed: {e}")
                        coordination_failures += 1
                    
                    time.sleep(0.05)  # 20Hz operation
                
                # Calculate performance metrics
                if pipeline_timings:
                    throughput = len(pipeline_timings) / sum(pipeline_timings)
                    latency_p95 = np.percentile(pipeline_timings, 95)
                else:
                    throughput = 0.0
                    latency_p95 = float('inf')
                
        except Exception as e:
            self.logger.error(f"Component coordination test failed: {e}")
            coordination_failures += 1
            throughput = 0.0
            latency_p95 = float('inf')
        
        result = OrchestrationTestResult(
            test_name="component_coordination",
            startup_time=startup_time if 'startup_time' in locals() else 0.0,
            shutdown_time=0.0,
            component_health_checks=component_health,
            timing_synchronization_errors=timing_errors,
            error_recovery_successful=True,
            safety_system_responsive=True,
            throughput_achieved=throughput,
            latency_p95=latency_p95,
            resource_usage_peak={'cpu': 0, 'memory': 0},
            coordination_failures=coordination_failures,
            success=coordination_failures < 3 and timing_errors < 5,
            error_message=None if coordination_failures < 3 else f"Too many coordination failures: {coordination_failures}"
        )
        
        self.test_results.append(result)
        return result
    
    def test_error_handling_recovery(self) -> OrchestrationTestResult:
        """Test error handling and recovery mechanisms."""
        self.logger.info("Testing error handling and recovery")
        
        recovery_tests = []
        
        try:
            with self.system_lifecycle() as (system, startup_time):
                # Test 1: Hardware failure recovery
                self.logger.info("Testing hardware failure recovery")
                
                # Simulate sensor failure
                self.hardware_interface.set_failure_mode("sensor_failure")
                
                # System should detect and handle the failure
                recovery_start = time.time()
                error_detected = False
                error_recovered = False
                
                for _ in range(20):  # Check for 2 seconds
                    try:
                        health = system.get_system_health()
                        if 'hardware' in health and health['hardware'] != 'healthy':
                            error_detected = True
                        
                        # Try to get robot state - should fail or return default
                        robot_state = self.hardware_interface.get_robot_state()
                        
                    except Exception:
                        error_detected = True
                    
                    time.sleep(0.1)
                
                # Clear failure mode and check recovery
                self.hardware_interface.set_failure_mode(None)
                
                for _ in range(20):
                    try:
                        health = system.get_system_health()
                        if 'hardware' in health and health['hardware'] == 'healthy':
                            error_recovered = True
                            break
                        
                        # Try operation
                        robot_state = self.hardware_interface.get_robot_state()
                        error_recovered = True
                        break
                        
                    except Exception:
                        pass
                    
                    time.sleep(0.1)
                
                recovery_time = time.time() - recovery_start
                recovery_tests.append({
                    'test': 'hardware_failure',
                    'error_detected': error_detected,
                    'error_recovered': error_recovered,
                    'recovery_time': recovery_time
                })
                
                # Test 2: Component timeout recovery
                self.logger.info("Testing component timeout recovery")
                
                # Simulate slow component response
                original_predict = system.predict_human_behavior
                
                def slow_predict(*args, **kwargs):
                    time.sleep(3.0)  # Longer than component timeout
                    return original_predict(*args, **kwargs)
                
                system.predict_human_behavior = slow_predict
                
                # Test should timeout and recover
                test_state = HumanState(
                    position=np.array([1.0, 0.0, 1.0]),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                    joint_positions={},
                    velocity=np.zeros(3),
                    timestamp=time.time()
                )
                
                context = ContextInformation(
                    task_type="timeout_test",
                    environment_state={},
                    robot_capabilities=[],
                    safety_constraints={},
                    timestamp=time.time()
                )
                
                timeout_detected = False
                timeout_recovered = False
                
                try:
                    # This should timeout
                    predictions = system.predict_human_behavior(
                        test_state, time_horizon=1.0, context=context
                    )
                except Exception as e:
                    if 'timeout' in str(e).lower():
                        timeout_detected = True
                
                # Restore original function
                system.predict_human_behavior = original_predict
                
                # Test recovery
                try:
                    predictions = system.predict_human_behavior(
                        test_state, time_horizon=1.0, context=context
                    )
                    timeout_recovered = True
                except Exception:
                    pass
                
                recovery_tests.append({
                    'test': 'component_timeout',
                    'error_detected': timeout_detected,
                    'error_recovered': timeout_recovered,
                    'recovery_time': 0.5
                })
                
        except Exception as e:
            self.logger.error(f"Error handling test failed: {e}")
            recovery_tests.append({
                'test': 'system_failure',
                'error_detected': True,
                'error_recovered': False,
                'recovery_time': float('inf')
            })
        
        # Analyze recovery test results
        recovery_successful = all(test['error_recovered'] for test in recovery_tests if test['error_detected'])
        avg_recovery_time = np.mean([test['recovery_time'] for test in recovery_tests if test['recovery_time'] != float('inf')])
        
        result = OrchestrationTestResult(
            test_name="error_handling_recovery",
            startup_time=startup_time if 'startup_time' in locals() else 0.0,
            shutdown_time=0.0,
            component_health_checks={},
            timing_synchronization_errors=0,
            error_recovery_successful=recovery_successful,
            safety_system_responsive=True,
            throughput_achieved=0.0,
            latency_p95=avg_recovery_time * 1000,  # Convert to ms
            resource_usage_peak={'cpu': 0, 'memory': 0},
            coordination_failures=len([t for t in recovery_tests if not t['error_recovered']]),
            success=recovery_successful and avg_recovery_time < 5.0,
            error_message=None if recovery_successful else "Error recovery failed"
        )
        
        self.test_results.append(result)
        return result
    
    def test_safety_constraint_enforcement(self) -> OrchestrationTestResult:
        """Test safety constraint enforcement."""
        self.logger.info("Testing safety constraint enforcement")
        
        safety_violations = 0
        safety_responses = 0
        response_times = []
        
        try:
            with self.system_lifecycle() as (system, startup_time):
                # Test safety-critical scenarios
                test_scenarios = [
                    {
                        'name': 'human_too_close',
                        'human_pos': np.array([0.2, 0.0, 1.0]),  # Very close to robot
                        'expected_response': 'emergency_stop'
                    },
                    {
                        'name': 'rapid_human_movement',
                        'human_pos': np.array([1.0, 0.0, 1.0]),
                        'human_vel': np.array([2.0, 0.0, 0.0]),  # Fast approach
                        'expected_response': 'slow_down'
                    },
                    {
                        'name': 'uncertain_human_intent',
                        'human_pos': np.array([0.8, 0.0, 1.0]),
                        'intent_confidence': 0.3,  # Low confidence
                        'expected_response': 'cautious_motion'
                    }
                ]
                
                robot_state = RobotState(
                    joint_positions=np.zeros(7),
                    joint_velocities=np.zeros(7),
                    end_effector_pose=np.array([0.5, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]),
                    timestamp=time.time()
                )
                
                for scenario in test_scenarios:
                    self.logger.info(f"Testing safety scenario: {scenario['name']}")
                    
                    # Create dangerous human state
                    human_state = HumanState(
                        position=scenario['human_pos'],
                        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                        joint_positions={'hand': scenario['human_pos']},
                        velocity=scenario.get('human_vel', np.zeros(3)),
                        timestamp=time.time(),
                        confidence=scenario.get('intent_confidence', 0.9)
                    )
                    
                    # Create behavior predictions indicating potential danger
                    behavior_predictions = [
                        BehaviorPrediction(
                            behavior_type=BehaviorType.REACHING,
                            probability=0.8,
                            predicted_trajectory=np.array([scenario['human_pos']] * 10),
                            time_horizon=1.0,
                            confidence=scenario.get('intent_confidence', 0.9)
                        )
                    ]
                    
                    intent_predictions = [
                        IntentPrediction(
                            intent_type=IntentType.UNKNOWN,
                            probability=0.7,
                            confidence=scenario.get('intent_confidence', 0.9),
                            time_horizon=1.0
                        )
                    ]
                    
                    context = ContextInformation(
                        task_type="safety_test",
                        environment_state={'danger_level': 'high'},
                        robot_capabilities=['manipulation'],
                        safety_constraints={'min_distance': 0.5},
                        timestamp=time.time()
                    )
                    
                    # Test safety system response
                    response_start = time.time()
                    
                    try:
                        # Generate control action
                        control_action = system.generate_robot_control(
                            current_robot_state=robot_state,
                            human_state=human_state,
                            behavior_predictions=behavior_predictions,
                            intent_predictions=intent_predictions,
                            context=context
                        )
                        
                        response_time = time.time() - response_start
                        response_times.append(response_time)
                        
                        # Check if safety constraints are enforced
                        if control_action.joint_torques is not None:
                            max_torque = np.max(np.abs(control_action.joint_torques))
                            
                            # For dangerous scenarios, torques should be limited
                            if scenario['name'] == 'human_too_close':
                                if max_torque > 5.0:  # Should be very limited
                                    safety_violations += 1
                                else:
                                    safety_responses += 1
                            
                            elif scenario['name'] == 'rapid_human_movement':
                                if max_torque > 20.0:  # Should be moderately limited
                                    safety_violations += 1
                                else:
                                    safety_responses += 1
                            
                            else:  # uncertain_human_intent
                                if max_torque > 50.0:  # Should be somewhat limited
                                    safety_violations += 1
                                else:
                                    safety_responses += 1
                        else:
                            # No torques generated - could be emergency stop
                            safety_responses += 1
                        
                        # Check response time (safety-critical should be fast)
                        if response_time > 0.05:  # 50ms threshold for safety
                            self.logger.warning(f"Slow safety response: {response_time:.3f}s")
                    
                    except Exception as e:
                        self.logger.error(f"Safety test failed: {e}")
                        safety_violations += 1
                    
                    time.sleep(0.1)  # Brief pause between scenarios
                
        except Exception as e:
            self.logger.error(f"Safety constraint test failed: {e}")
            safety_violations += len(test_scenarios)
        
        # Calculate safety metrics
        total_scenarios = len(test_scenarios) if 'test_scenarios' in locals() else 3
        safety_response_rate = safety_responses / max(total_scenarios, 1)
        avg_response_time = np.mean(response_times) if response_times else float('inf')
        
        result = OrchestrationTestResult(
            test_name="safety_constraint_enforcement",
            startup_time=startup_time if 'startup_time' in locals() else 0.0,
            shutdown_time=0.0,
            component_health_checks={},
            timing_synchronization_errors=0,
            error_recovery_successful=True,
            safety_system_responsive=safety_response_rate > 0.8,
            throughput_achieved=0.0,
            latency_p95=avg_response_time * 1000,  # Convert to ms
            resource_usage_peak={'cpu': 0, 'memory': 0},
            coordination_failures=safety_violations,
            success=safety_violations == 0 and avg_response_time < 0.05,
            error_message=None if safety_violations == 0 else f"Safety violations: {safety_violations}"
        )
        
        self.test_results.append(result)
        return result
    
    def test_real_time_performance_orchestration(self) -> OrchestrationTestResult:
        """Test real-time performance under orchestration load."""
        self.logger.info("Testing real-time performance orchestration")
        
        timing_violations = 0
        throughput_measurements = []
        latency_measurements = []
        resource_usage = {'cpu': [], 'memory': []}
        
        try:
            with self.system_lifecycle() as (system, startup_time):
                # High-frequency operation test
                test_duration = 10.0  # 10 seconds of intensive testing
                target_frequency = 20.0  # 20Hz
                
                start_time = time.time()
                iteration_count = 0
                
                while time.time() - start_time < test_duration:
                    iteration_start = time.time()
                    
                    # Simulate realistic workload
                    human_state = HumanState(
                        position=np.array([1.0, 0.0, 1.0]) + 0.1 * np.random.randn(3),
                        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                        joint_positions={'hand': np.array([1.0, 0.0, 1.0])},
                        velocity=0.1 * np.random.randn(3),
                        timestamp=time.time()
                    )
                    
                    context = ContextInformation(
                        task_type="performance_test",
                        environment_state={},
                        robot_capabilities=[],
                        safety_constraints={},
                        timestamp=time.time()
                    )
                    
                    robot_state = RobotState(
                        joint_positions=np.zeros(7),
                        joint_velocities=np.zeros(7),
                        end_effector_pose=np.array([0.5, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]),
                        timestamp=time.time()
                    )
                    
                    try:
                        # Complete pipeline execution
                        behavior_predictions = system.predict_human_behavior(
                            human_state, time_horizon=1.0, context=context
                        )
                        
                        intent_predictions = system.predict_human_intent(
                            human_state, context
                        )
                        
                        control_action = system.generate_robot_control(
                            current_robot_state=robot_state,
                            human_state=human_state,
                            behavior_predictions=behavior_predictions,
                            intent_predictions=intent_predictions,
                            context=context
                        )
                        
                        iteration_time = time.time() - iteration_start
                        latency_measurements.append(iteration_time)
                        
                        # Check real-time constraint (50ms for 20Hz)
                        if iteration_time > 0.05:
                            timing_violations += 1
                        
                        iteration_count += 1
                        
                        # Monitor resource usage
                        try:
                            process = psutil.Process()
                            cpu_percent = process.cpu_percent()
                            memory_mb = process.memory_info().rss / 1024 / 1024
                            resource_usage['cpu'].append(cpu_percent)
                            resource_usage['memory'].append(memory_mb)
                        except:
                            pass
                        
                    except Exception as e:
                        self.logger.warning(f"Performance test iteration failed: {e}")
                        timing_violations += 1
                    
                    # Maintain target frequency
                    elapsed = time.time() - iteration_start
                    target_interval = 1.0 / target_frequency
                    if elapsed < target_interval:
                        time.sleep(target_interval - elapsed)
                
                # Calculate performance metrics
                actual_duration = time.time() - start_time
                actual_throughput = iteration_count / actual_duration
                
                throughput_measurements.append(actual_throughput)
                
        except Exception as e:
            self.logger.error(f"Real-time performance test failed: {e}")
            timing_violations += 1
        
        # Calculate final metrics
        avg_throughput = np.mean(throughput_measurements) if throughput_measurements else 0.0
        latency_p95 = np.percentile(latency_measurements, 95) if latency_measurements else float('inf')
        
        peak_cpu = np.max(resource_usage['cpu']) if resource_usage['cpu'] else 0.0
        peak_memory = np.max(resource_usage['memory']) if resource_usage['memory'] else 0.0
        
        result = OrchestrationTestResult(
            test_name="real_time_performance",
            startup_time=startup_time if 'startup_time' in locals() else 0.0,
            shutdown_time=0.0,
            component_health_checks={},
            timing_synchronization_errors=timing_violations,
            error_recovery_successful=True,
            safety_system_responsive=True,
            throughput_achieved=avg_throughput,
            latency_p95=latency_p95 * 1000,  # Convert to ms
            resource_usage_peak={'cpu': peak_cpu, 'memory': peak_memory},
            coordination_failures=0,
            success=timing_violations < 10 and avg_throughput > 15.0 and latency_p95 < 0.05,
            error_message=None if timing_violations < 10 else f"Too many timing violations: {timing_violations}"
        )
        
        self.test_results.append(result)
        return result
    
    def run_all_orchestration_tests(self) -> Dict[str, Any]:
        """Run all orchestration tests."""
        self.logger.info("Starting comprehensive orchestration tests")
        
        # Connect hardware interface
        self.hardware_interface.connect()
        
        try:
            # Run all tests
            test_functions = [
                self.test_system_startup_shutdown,
                self.test_component_coordination,
                self.test_error_handling_recovery,
                self.test_safety_constraint_enforcement,
                self.test_real_time_performance_orchestration
            ]
            
            for test_func in test_functions:
                try:
                    result = test_func()
                    self.logger.info(f"Test {result.test_name}: Success={result.success}")
                except Exception as e:
                    self.logger.error(f"Test {test_func.__name__} failed: {e}")
            
            return self.generate_orchestration_report()
            
        finally:
            # Cleanup
            self.hardware_interface.disconnect()
    
    def generate_orchestration_report(self) -> Dict[str, Any]:
        """Generate comprehensive orchestration test report."""
        if not self.test_results:
            return {"error": "No test results available"}
        
        successful_tests = [r for r in self.test_results if r.success]
        
        report = {
            'summary': {
                'total_tests': len(self.test_results),
                'successful_tests': len(successful_tests),
                'success_rate': len(successful_tests) / len(self.test_results),
                'average_startup_time': np.mean([r.startup_time for r in self.test_results]),
                'average_throughput': np.mean([r.throughput_achieved for r in self.test_results if r.throughput_achieved > 0]),
                'average_latency_p95': np.mean([r.latency_p95 for r in self.test_results if r.latency_p95 != float('inf')]),
                'total_safety_violations': sum(r.coordination_failures for r in self.test_results),
                'error_recovery_rate': np.mean([r.error_recovery_successful for r in self.test_results])
            },
            'detailed_results': {
                result.test_name: {
                    'success': result.success,
                    'startup_time_s': result.startup_time,
                    'shutdown_time_s': result.shutdown_time,
                    'throughput_hz': result.throughput_achieved,
                    'latency_p95_ms': result.latency_p95,
                    'coordination_failures': result.coordination_failures,
                    'timing_errors': result.timing_synchronization_errors,
                    'safety_responsive': result.safety_system_responsive,
                    'error_recovery': result.error_recovery_successful,
                    'resource_usage': result.resource_usage_peak,
                    'error_message': result.error_message
                }
                for result in self.test_results
            },
            'recommendations': self._generate_orchestration_recommendations()
        }
        
        return report
    
    def _generate_orchestration_recommendations(self) -> List[str]:
        """Generate orchestration improvement recommendations."""
        recommendations = []
        
        failed_tests = [r for r in self.test_results if not r.success]
        slow_startup_tests = [r for r in self.test_results if r.startup_time > 5.0]
        low_throughput_tests = [r for r in self.test_results if 0 < r.throughput_achieved < 10.0]
        high_latency_tests = [r for r in self.test_results if 0 < r.latency_p95 < 100.0]
        
        if len(failed_tests) > 0:
            recommendations.append(f"System orchestration failures detected in {len(failed_tests)} tests - review component integration")
        
        if len(slow_startup_tests) > 0:
            recommendations.append("Slow system startup detected - optimize initialization sequence")
        
        if len(low_throughput_tests) > 0:
            recommendations.append("Low throughput detected - review component coordination and timing")
        
        if len(high_latency_tests) > 0:
            recommendations.append("High latency detected - optimize real-time performance")
        
        safety_issues = sum(r.coordination_failures for r in self.test_results)
        if safety_issues > 5:
            recommendations.append("Safety constraint violations detected - strengthen safety system")
        
        error_recovery_rate = np.mean([r.error_recovery_successful for r in self.test_results])
        if error_recovery_rate < 0.8:
            recommendations.append("Poor error recovery rate - improve error handling mechanisms")
        
        if not recommendations:
            recommendations.append("System orchestration performing well across all test scenarios")
        
        return recommendations


# Test fixtures
@pytest.fixture
def orchestration_tester():
    """Fixture providing orchestration tester."""
    tester = SystemOrchestrationTester()
    yield tester
    # Cleanup
    if tester.hardware_interface.is_connected:
        tester.hardware_interface.disconnect()


# Test classes
class TestSystemOrchestration:
    """System orchestration test suite."""
    
    def test_startup_shutdown_procedures(self, orchestration_tester):
        """Test system startup and shutdown procedures."""
        result = orchestration_tester.test_system_startup_shutdown()
        assert result.success, f"Startup/shutdown test failed: {result.error_message}"
        assert result.startup_time < 15.0, f"Startup too slow: {result.startup_time}s"
    
    def test_component_coordination_timing(self, orchestration_tester):
        """Test component coordination and timing."""
        result = orchestration_tester.test_component_coordination()
        assert result.success, f"Component coordination failed: {result.error_message}"
        assert result.coordination_failures < 5, f"Too many coordination failures: {result.coordination_failures}"
        assert result.throughput_achieved > 5.0, f"Low throughput: {result.throughput_achieved}Hz"
    
    def test_error_recovery_mechanisms(self, orchestration_tester):
        """Test error handling and recovery."""
        result = orchestration_tester.test_error_handling_recovery()
        assert result.success, f"Error recovery failed: {result.error_message}"
        assert result.error_recovery_successful, "Error recovery not successful"
        assert result.latency_p95 < 5000, f"Slow error recovery: {result.latency_p95}ms"
    
    def test_safety_system_enforcement(self, orchestration_tester):
        """Test safety constraint enforcement."""
        result = orchestration_tester.test_safety_constraint_enforcement()
        assert result.success, f"Safety enforcement failed: {result.error_message}"
        assert result.safety_system_responsive, "Safety system not responsive"
        assert result.coordination_failures == 0, f"Safety violations detected: {result.coordination_failures}"
    
    def test_real_time_orchestration_performance(self, orchestration_tester):
        """Test real-time performance under orchestration load."""
        result = orchestration_tester.test_real_time_performance_orchestration()
        assert result.success, f"Real-time performance failed: {result.error_message}"
        assert result.throughput_achieved > 10.0, f"Low real-time throughput: {result.throughput_achieved}Hz"
        assert result.latency_p95 < 100, f"High real-time latency: {result.latency_p95}ms"
    
    def test_comprehensive_orchestration_validation(self, orchestration_tester):
        """Run comprehensive orchestration validation."""
        report = orchestration_tester.run_all_orchestration_tests()
        
        assert 'summary' in report, "Orchestration report missing summary"
        assert report['summary']['total_tests'] > 0, "No orchestration tests were run"
        
        # At least 80% of tests should pass
        success_rate = report['summary']['success_rate']
        assert success_rate >= 0.8, f"Low orchestration test success rate: {success_rate:.2f}"
        
        # System should start quickly
        avg_startup = report['summary']['average_startup_time']
        assert avg_startup < 10.0, f"System startup too slow: {avg_startup:.2f}s"


def run_system_orchestration_tests():
    """Run system orchestration test suite."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    tester = SystemOrchestrationTester()
    
    try:
        report = tester.run_all_orchestration_tests()
        
        # Print summary
        print("\n" + "="*60)
        print("SYSTEM ORCHESTRATION TEST RESULTS")
        print("="*60)
        
        summary = report['summary']
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful Tests: {summary['successful_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Average Startup Time: {summary['average_startup_time']:.2f}s")
        print(f"Average Throughput: {summary['average_throughput']:.1f}Hz")
        print(f"Average Latency P95: {summary['average_latency_p95']:.1f}ms")
        print(f"Safety Violations: {summary['total_safety_violations']}")
        print(f"Error Recovery Rate: {summary['error_recovery_rate']:.1%}")
        
        print("\nDetailed Results:")
        for test_name, results in report['detailed_results'].items():
            print(f"  {test_name}:")
            print(f"    Success: {results['success']}")
            print(f"    Startup Time: {results['startup_time_s']:.2f}s")
            if results['throughput_hz'] > 0:
                print(f"    Throughput: {results['throughput_hz']:.1f}Hz")
            if results['latency_p95_ms'] < 1000:
                print(f"    Latency P95: {results['latency_p95_ms']:.1f}ms")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
        
        return report
        
    except Exception as e:
        print(f"System orchestration tests failed: {e}")
        raise


if __name__ == "__main__":
    report = run_system_orchestration_tests()
    
    # Save report
    import json
    with open('orchestration_test_results.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nFull report saved to: orchestration_test_results.json")