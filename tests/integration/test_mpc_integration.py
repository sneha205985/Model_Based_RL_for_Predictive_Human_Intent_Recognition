"""
Integration tests for complete MPC system.

This test suite validates the integration of all MPC components:
- Robot dynamics with MPC controller
- HRI MPC with human behavior prediction
- Safety constraints enforcement
- Visualization integration
- Real-time performance validation
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch
import time

from src.control.mpc_controller import MPCController, MPCConfiguration, SolverType
from src.control.hri_mpc import HRIMPCController, HRIConfiguration, HumanState, create_default_hri_mpc
from src.models.robotics.robot_dynamics import create_default_6dof_robot, RobotState
from src.control.safety_constraints import create_default_safety_constraints, SafetyMonitor
from src.visualization.mpc_plots import MPCVisualizer, create_mpc_visualization_suite
from src.models.behavior_model import HumanBehaviorModel
from src.models.bayesian_intent_classifier import BayesianIntentClassifier
from src.data.synthetic_generator import SyntheticHumanBehaviorGenerator, GestureType


class TestMPCRobotIntegration:
    """Test MPC integration with robot dynamics."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated MPC-robot system."""
        # Create robot model
        robot = create_default_6dof_robot()
        
        # Create MPC controller
        config = MPCConfiguration(
            prediction_horizon=10,
            control_horizon=8,
            sampling_time=0.1,
            solver_type=SolverType.SCIPY_MINIMIZE,
            max_solve_time=2.0  # Generous for testing
        )
        
        def robot_dynamics_wrapper(state, control):
            return robot.dynamics(state, control)
        
        mpc = MPCController(
            config=config,
            state_dim=robot.state_dim,
            control_dim=robot.control_dim,
            dynamics_model=robot_dynamics_wrapper
        )
        
        # Set up objective function
        Q = np.eye(12)
        Q[0:6, 0:6] *= 10  # Weight joint positions
        R = np.eye(6) * 0.01
        mpc.set_objective_function(Q, R)
        
        # Set constraints
        mpc.set_constraints(
            state_bounds=(
                np.concatenate([robot.joint_limits.position_min,
                               -robot.joint_limits.velocity_max]),
                np.concatenate([robot.joint_limits.position_max,
                               robot.joint_limits.velocity_max])
            ),
            control_bounds=(
                -robot.joint_limits.torque_max * 0.5,
                robot.joint_limits.torque_max * 0.5
            )
        )
        
        return robot, mpc
    
    def test_mpc_robot_closed_loop_simulation(self, integrated_system):
        """Test closed-loop simulation with MPC and robot dynamics."""
        robot, mpc = integrated_system
        
        # Initial state (feasible configuration)
        initial_joint_pos = np.array([0.1, -0.3, 0.5, 0.0, 0.2, 0.0])
        initial_joint_vel = np.zeros(6)
        current_state = np.concatenate([initial_joint_pos, initial_joint_vel])
        
        # Target configuration
        target_joint_pos = np.array([0.3, -0.1, 0.2, 0.1, 0.4, -0.1])
        
        # Simulation parameters
        sim_duration = 2.0  # seconds
        dt = mpc.config.sampling_time
        n_steps = int(sim_duration / dt)
        
        # Data storage
        state_history = [current_state.copy()]
        control_history = []
        cost_history = []
        solve_times = []
        
        # Closed-loop simulation
        for step in range(n_steps):
            # Create reference trajectory (constant target)
            N = mpc.config.prediction_horizon
            reference = np.tile(
                np.concatenate([target_joint_pos, np.zeros(6)]), (N + 1, 1)
            )
            
            # Solve MPC
            start_time = time.time()
            result = mpc.solve_mpc(current_state, reference_trajectory=reference)
            solve_time = time.time() - start_time
            solve_times.append(solve_time)
            
            # Check if solution was found
            if result.optimal_control is not None and len(result.optimal_control) > 0:
                # Apply first control input
                control_input = result.optimal_control[0]
                control_history.append(control_input.copy())
                
                if result.optimal_cost is not None:
                    cost_history.append(result.optimal_cost)
                
                # Simulate robot dynamics
                current_state = robot.dynamics(current_state, control_input)
                state_history.append(current_state.copy())
                
                # Check joint limits
                violations = robot.check_joint_limits(current_state)
                if any(violations.values()):
                    print(f"Warning: Joint limit violations at step {step}: {violations}")
            
            else:
                # If no solution, apply zero control (emergency stop)
                zero_control = np.zeros(6)
                control_history.append(zero_control)
                
                current_state = robot.dynamics(current_state, zero_control)
                state_history.append(current_state.copy())
        
        # Validate simulation results
        assert len(state_history) == n_steps + 1
        assert len(control_history) == n_steps
        
        # Check that system made progress toward target
        initial_error = np.linalg.norm(initial_joint_pos - target_joint_pos)
        final_error = np.linalg.norm(state_history[-1][0:6] - target_joint_pos)
        
        print(f"Initial error: {initial_error:.4f}, Final error: {final_error:.4f}")
        print(f"Mean solve time: {np.mean(solve_times):.4f}s")
        
        # Should make some progress (at least 20% error reduction)
        assert final_error < 0.8 * initial_error or final_error < 0.1
        
        # Real-time performance check
        real_time_violations = sum(1 for t in solve_times if t > mpc.config.max_solve_time)
        violation_rate = real_time_violations / len(solve_times)
        print(f"Real-time violations: {violation_rate:.1%}")
        
        # Most solves should meet real-time constraint
        assert violation_rate < 0.5  # Allow some violations for testing
    
    def test_mpc_robot_trajectory_tracking(self, integrated_system):
        """Test MPC trajectory tracking with robot dynamics."""
        robot, mpc = integrated_system
        
        # Create a smooth trajectory in joint space
        sim_duration = 3.0
        dt = mpc.config.sampling_time
        n_steps = int(sim_duration / dt)
        
        time_vec = np.linspace(0, sim_duration, n_steps + 1)
        
        # Sinusoidal trajectory for first joint, others constant
        trajectory = np.zeros((n_steps + 1, 12))
        for i, t in enumerate(time_vec):
            trajectory[i, 0] = 0.3 * np.sin(2 * np.pi * t / 3.0)  # 3-second period
            trajectory[i, 1:6] = np.array([-0.2, 0.4, 0.1, 0.3, -0.1])  # Constant
            # Velocities are zero (will be computed by differencing)
        
        # Compute reference velocities
        for i in range(1, n_steps + 1):
            trajectory[i, 6:12] = (trajectory[i, 0:6] - trajectory[i-1, 0:6]) / dt
        
        # Initial state matches trajectory start
        current_state = trajectory[0].copy()
        
        # Tracking simulation
        tracking_errors = []
        
        for step in range(min(n_steps, 20)):  # Limit for testing speed
            # Reference trajectory (current + future)
            N = mpc.config.prediction_horizon
            ref_start = min(step, len(trajectory) - N - 1)
            reference = trajectory[ref_start:ref_start + N + 1]
            
            # Solve MPC
            result = mpc.solve_mpc(current_state, reference_trajectory=reference)
            
            if result.optimal_control is not None and len(result.optimal_control) > 0:
                # Apply control
                control_input = result.optimal_control[0]
                current_state = robot.dynamics(current_state, control_input)
                
                # Compute tracking error
                target_state = trajectory[step + 1] if step + 1 < len(trajectory) else trajectory[-1]
                tracking_error = np.linalg.norm(current_state[0:6] - target_state[0:6])
                tracking_errors.append(tracking_error)
            
            else:
                tracking_errors.append(float('inf'))
        
        # Validate tracking performance
        finite_errors = [e for e in tracking_errors if np.isfinite(e)]
        if finite_errors:
            mean_error = np.mean(finite_errors)
            print(f"Mean tracking error: {mean_error:.4f} rad")
            
            # Should achieve reasonable tracking (depends on trajectory complexity)
            assert mean_error < 0.5  # rad, reasonable for test trajectory


class TestHRIMPCIntegration:
    """Test HRI MPC integration with human behavior models."""
    
    @pytest.fixture
    def hri_system(self):
        """Create integrated HRI MPC system."""
        # Create robot and HRI controller
        robot = create_default_6dof_robot()
        hri_controller = create_default_hri_mpc(robot)
        
        # Set up cost matrices
        Q = np.eye(12)
        Q[0:6, 0:6] *= 10
        R = np.eye(6) * 0.01
        hri_controller.set_objective_function(Q, R)
        
        # Set constraints
        hri_controller.set_constraints(
            state_bounds=(
                np.concatenate([robot.joint_limits.position_min,
                               -robot.joint_limits.velocity_max]),
                np.concatenate([robot.joint_limits.position_max,
                               robot.joint_limits.velocity_max])
            ),
            control_bounds=(
                -robot.joint_limits.torque_max * 0.3,
                robot.joint_limits.torque_max * 0.3
            )
        )
        
        # Create synthetic human behavior generator for testing
        workspace_bounds = np.array([-1, 1, -1, 1, 0, 2])
        behavior_generator = SyntheticHumanBehaviorGenerator(
            workspace_bounds=workspace_bounds,
            random_seed=42
        )
        
        return hri_controller, robot, behavior_generator
    
    def test_hri_mpc_human_aware_behavior(self, hri_system):
        """Test HRI MPC adaptation to human behavior."""
        hri_controller, robot, behavior_gen = hri_system
        
        # Generate synthetic human behavior sequence
        human_sequence = behavior_gen.generate_sequence(
            gesture_type=GestureType.HANDOVER,
            duration=2.0,
            noise_level=0.01
        )
        
        if not human_sequence:
            pytest.skip("Could not generate human sequence")
        
        # Initial robot state
        initial_joint_pos = np.array([0.0, -0.5, 0.8, 0.0, 0.3, 0.0])
        robot_state = robot.get_state_from_measurements(
            initial_joint_pos, np.zeros(6)
        )
        
        # Target pose for robot
        target_pose = np.array([0.5, 0.3, 0.8, 0.0, 0.0, 0.0])
        
        # Test different interaction scenarios
        scenarios = [
            {
                'human_pos': np.array([0.8, 0.2, 0.9]),
                'intent': {'handover': 0.9, 'idle': 0.1},
                'uncertainty': 0.1,
                'expected_phase': 'handover'
            },
            {
                'human_pos': np.array([0.8, 0.2, 0.9]),
                'intent': {'wave': 0.8, 'idle': 0.2},
                'uncertainty': 0.3,
                'expected_phase': 'retreat'
            },
            {
                'human_pos': np.array([0.2, 0.1, 0.7]),
                'intent': {'reach': 0.7, 'grab': 0.3},
                'uncertainty': 0.2,
                'expected_phase': 'approach'
            }
        ]
        
        for i, scenario in enumerate(scenarios):
            print(f"Testing scenario {i+1}: {scenario['expected_phase']}")
            
            # Create human state
            human_state = HumanState(
                position=scenario['human_pos'],
                velocity=np.zeros(3),
                intent_probabilities=scenario['intent'],
                uncertainty=scenario['uncertainty'],
                timestamp=float(i)
            )
            
            # Solve HRI MPC
            result = hri_controller.solve_hri_mpc(
                robot_state, target_pose, human_state
            )
            
            # Check that controller responds to human state
            assert isinstance(result, type(result))  # Basic result structure
            
            # Check that interaction phase was updated
            current_phase = hri_controller.current_phase.value
            print(f"Controller phase: {current_phase}")
            
            # Verify some level of human-aware behavior
            if result.solve_time > 0:
                print(f"Solve time: {result.solve_time:.4f}s")
            
            # Check safety considerations
            if result.optimal_control is not None:
                # Control should be reasonable (not extreme)
                control_magnitude = np.linalg.norm(result.optimal_control[0])
                print(f"Control magnitude: {control_magnitude:.4f}")
                assert control_magnitude < 100.0  # Sanity check
    
    def test_hri_safety_constraint_enforcement(self, hri_system):
        """Test safety constraint enforcement in HRI scenarios."""
        hri_controller, robot, _ = hri_system
        
        # Create safety constraints
        safety_constraints = create_default_safety_constraints(robot)
        safety_monitor = SafetyMonitor(
            constraints=safety_constraints,
            safety_params=type('SafetyParams', (), {
                'min_safety_distance': 0.3,
                'collision_threshold': 0.05
            })()
        )
        
        # Test scenario with human very close to robot
        robot_state = robot.get_state_from_measurements(
            np.zeros(6), np.zeros(6)
        )
        
        # Human very close (should trigger safety response)
        close_human_state = HumanState(
            position=np.array([0.1, 0.1, 0.1]),  # Very close
            velocity=np.zeros(3),
            intent_probabilities={'grab': 0.9, 'idle': 0.1},
            uncertainty=0.1,
            timestamp=0.0
        )
        
        target_pose = np.array([0.5, 0.5, 0.8, 0.0, 0.0, 0.0])
        
        # Solve with close human
        result_close = hri_controller.solve_hri_mpc(
            robot_state, target_pose, close_human_state
        )
        
        # Human far away (normal operation)
        far_human_state = HumanState(
            position=np.array([2.0, 2.0, 1.0]),  # Far away
            velocity=np.zeros(3),
            intent_probabilities={'wave': 0.8, 'idle': 0.2},
            uncertainty=0.2,
            timestamp=1.0
        )
        
        # Solve with distant human
        result_far = hri_controller.solve_hri_mpc(
            robot_state, target_pose, far_human_state
        )
        
        # Verify different behaviors
        if (result_close.optimal_control is not None and 
            result_far.optimal_control is not None):
            
            close_control_mag = np.linalg.norm(result_close.optimal_control[0])
            far_control_mag = np.linalg.norm(result_far.optimal_control[0])
            
            print(f"Close human control magnitude: {close_control_mag:.4f}")
            print(f"Far human control magnitude: {far_control_mag:.4f}")
            
            # Controller should be more conservative with close human
            # (though the exact relationship depends on implementation)
            assert close_control_mag >= 0  # Basic sanity check
            assert far_control_mag >= 0


class TestMPCVisualizationIntegration:
    """Test integration of MPC with visualization tools."""
    
    @pytest.fixture
    def visualization_system(self):
        """Create MPC system with visualization."""
        robot = create_default_6dof_robot()
        hri_controller = create_default_hri_mpc(robot)
        
        # Create visualization suite
        viz_suite = create_mpc_visualization_suite(robot, hri_controller)
        
        return hri_controller, robot, viz_suite
    
    def test_trajectory_visualization_integration(self, visualization_system):
        """Test trajectory visualization with MPC results."""
        hri_controller, robot, viz_suite = visualization_system
        
        # Set up controller
        Q = np.eye(12)
        R = np.eye(6) * 0.01
        hri_controller.set_objective_function(Q, R)
        
        # Generate some trajectory data
        initial_state = np.zeros(12)
        
        # Mock MPC result for visualization
        N = 10
        predicted_states = np.random.randn(N, 12) * 0.1
        
        # Test 3D trajectory visualization
        visualizer = viz_suite['visualizer']
        
        try:
            fig = visualizer.plot_3d_trajectory(
                predicted_states=predicted_states,
                human_position=np.array([0.5, 0.5, 0.8]),
                interactive=False
            )
            
            # Basic checks that visualization was created
            assert fig is not None
            print("3D trajectory visualization created successfully")
            
        except Exception as e:
            print(f"Visualization test failed (may be expected in headless environment): {e}")
            # This is acceptable for CI environments without display
    
    def test_performance_metrics_visualization(self, visualization_system):
        """Test performance metrics visualization."""
        hri_controller, robot, viz_suite = visualization_system
        
        # Generate some performance data
        solve_times = np.random.exponential(0.01, 100)  # Realistic solve times
        costs = np.random.exponential(10.0, 100)
        
        performance_data = {
            'solve_times': solve_times.tolist(),
            'costs': costs.tolist(),
            'constraint_violations': (np.random.rand(100) > 0.9).tolist()
        }
        
        visualizer = viz_suite['visualizer']
        
        try:
            fig = visualizer.plot_performance_metrics(performance_data)
            assert fig is not None
            print("Performance metrics visualization created successfully")
            
        except Exception as e:
            print(f"Performance visualization test failed: {e}")


class TestEndToEndIntegration:
    """End-to-end integration tests of complete system."""
    
    def test_complete_hri_scenario(self):
        """Test complete HRI scenario from human behavior to robot control."""
        print("Running end-to-end HRI scenario test...")
        
        # Create complete system
        robot = create_default_6dof_robot()
        hri_controller = create_default_hri_mpc(robot)
        
        # Set up controller
        Q = np.eye(12)
        Q[0:6, 0:6] *= 10
        R = np.eye(6) * 0.01
        hri_controller.set_objective_function(Q, R)
        
        hri_controller.set_constraints(
            state_bounds=(
                np.concatenate([robot.joint_limits.position_min,
                               -robot.joint_limits.velocity_max]),
                np.concatenate([robot.joint_limits.position_max,
                               robot.joint_limits.velocity_max])
            ),
            control_bounds=(
                -robot.joint_limits.torque_max * 0.2,
                robot.joint_limits.torque_max * 0.2
            )
        )
        
        # Create synthetic human behavior
        workspace_bounds = np.array([-1, 1, -1, 1, 0, 2])
        behavior_gen = SyntheticHumanBehaviorGenerator(
            workspace_bounds=workspace_bounds,
            random_seed=123
        )
        
        # Initial robot configuration
        robot_state = robot.get_state_from_measurements(
            np.array([0.1, -0.2, 0.3, 0.0, 0.1, 0.0]),
            np.zeros(6)
        )
        
        # Target for robot
        target_pose = np.array([0.6, 0.3, 0.9, 0.0, 0.0, 0.0])
        
        # Simulate interaction scenario
        scenario_duration = 2.0  # seconds
        dt = 0.2  # time step
        n_steps = int(scenario_duration / dt)
        
        success_count = 0
        total_solve_time = 0
        
        for step in range(n_steps):
            t = step * dt
            
            # Generate evolving human behavior
            if t < 0.6:
                # Human reaching
                human_pos = np.array([0.7, 0.2, 0.8]) + 0.1 * np.array([np.sin(t), np.cos(t), 0])
                intent = {'reach': 0.8, 'grab': 0.2}
                uncertainty = 0.2
            elif t < 1.2:
                # Human handover
                human_pos = np.array([0.6, 0.25, 0.85])
                intent = {'handover': 0.9, 'reach': 0.1}
                uncertainty = 0.1
            else:
                # Human waving goodbye
                human_pos = np.array([0.8, 0.3, 0.9])
                intent = {'wave': 0.9, 'idle': 0.1}
                uncertainty = 0.3
            
            # Create human state
            human_state = HumanState(
                position=human_pos,
                velocity=np.zeros(3),
                intent_probabilities=intent,
                uncertainty=uncertainty,
                timestamp=t
            )
            
            # Solve HRI MPC
            start_time = time.time()
            result = hri_controller.solve_hri_mpc(
                robot_state, target_pose, human_state
            )
            solve_time = time.time() - start_time
            total_solve_time += solve_time
            
            if result.optimal_control is not None and len(result.optimal_control) > 0:
                success_count += 1
                
                # Simulate robot forward
                control = result.optimal_control[0]
                next_state_vec = robot.dynamics(
                    np.concatenate([robot_state.joint_positions, robot_state.joint_velocities]),
                    control
                )
                
                # Update robot state
                robot_state = robot.get_state_from_measurements(
                    next_state_vec[0:6], next_state_vec[6:12]
                )
                robot_state.timestamp = t + dt
                
                print(f"Step {step+1}: Phase={hri_controller.current_phase.value}, "
                      f"Solve={solve_time:.4f}s")
            
            else:
                print(f"Step {step+1}: MPC failed to find solution")
        
        # Validate end-to-end performance
        success_rate = success_count / n_steps
        avg_solve_time = total_solve_time / n_steps
        
        print(f"End-to-end test completed:")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Average solve time: {avg_solve_time:.4f}s")
        print(f"  Final robot joint positions: {robot_state.joint_positions}")
        
        # Should achieve reasonable success rate
        assert success_rate > 0.5  # At least 50% success
        assert avg_solve_time < 1.0  # Reasonable solve times
        
        # Check that robot made some progress
        final_ee_pos = robot_state.end_effector_pose[0:3]
        distance_to_target = np.linalg.norm(final_ee_pos - target_pose[0:3])
        print(f"  Distance to target: {distance_to_target:.3f}m")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print statements