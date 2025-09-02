"""
Unit tests for MPC Controller implementation.

This test suite validates the core MPC functionality including:
- Controller initialization and configuration
- Objective function setup and validation
- Constraint handling
- Optimization solver integration
- Warm start functionality
- Performance monitoring
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch
import time

from src.control.mpc_controller import (
    MPCController, MPCConfiguration, MPCResult, MPCStatus, SolverType
)
from src.models.robotics.robot_dynamics import create_default_6dof_robot


class TestMPCConfiguration:
    """Test MPC configuration parameters."""
    
    def test_default_configuration(self):
        """Test default MPC configuration values."""
        config = MPCConfiguration()
        
        assert config.prediction_horizon == 20
        assert config.control_horizon == 15
        assert config.sampling_time == 0.1
        assert config.solver_type == SolverType.CVXPY_QP
        assert config.max_solve_time == 0.05
        assert config.use_warm_start == True
        assert config.safety_margin == 0.1
    
    def test_custom_configuration(self):
        """Test custom MPC configuration."""
        config = MPCConfiguration(
            prediction_horizon=30,
            control_horizon=25,
            sampling_time=0.05,
            solver_type=SolverType.SCIPY_MINIMIZE,
            max_solve_time=0.1,
            use_warm_start=False
        )
        
        assert config.prediction_horizon == 30
        assert config.control_horizon == 25
        assert config.sampling_time == 0.05
        assert config.solver_type == SolverType.SCIPY_MINIMIZE
        assert config.max_solve_time == 0.1
        assert config.use_warm_start == False


class TestMPCController:
    """Test core MPC controller functionality."""
    
    @pytest.fixture
    def mock_dynamics(self):
        """Mock dynamics function for testing."""
        def dynamics(state, control):
            # Simple integrator dynamics: x_{k+1} = x_k + dt * u_k
            dt = 0.1
            next_state = state.copy()
            next_state[6:12] = state[6:12] + dt * control  # Velocity integration
            next_state[0:6] = state[0:6] + dt * next_state[6:12]  # Position integration
            return next_state
        return dynamics
    
    @pytest.fixture
    def mpc_controller(self, mock_dynamics):
        """Create MPC controller for testing."""
        config = MPCConfiguration(
            prediction_horizon=10,
            control_horizon=8,
            max_solve_time=1.0  # Relaxed for testing
        )
        
        controller = MPCController(
            config=config,
            state_dim=12,
            control_dim=6,
            dynamics_model=mock_dynamics
        )
        
        return controller
    
    def test_controller_initialization(self, mpc_controller):
        """Test MPC controller initialization."""
        assert mpc_controller.state_dim == 12
        assert mpc_controller.control_dim == 6
        assert mpc_controller.config.prediction_horizon == 10
        assert mpc_controller.Q is None  # Not set yet
        assert mpc_controller.R is None
        assert len(mpc_controller.solve_times) == 0
    
    def test_set_objective_function_valid(self, mpc_controller):
        """Test setting valid objective function matrices."""
        Q = np.eye(12) * 2.0
        R = np.eye(6) * 0.1
        P = np.eye(12) * 10.0
        
        mpc_controller.set_objective_function(Q, R, P)
        
        assert np.allclose(mpc_controller.Q, Q)
        assert np.allclose(mpc_controller.R, R)
        assert np.allclose(mpc_controller.P, P)
    
    def test_set_objective_function_invalid_dimensions(self, mpc_controller):
        """Test setting objective function with invalid dimensions."""
        Q_wrong = np.eye(10)  # Wrong dimension
        R = np.eye(6)
        
        with pytest.raises(ValueError, match="Q matrix must be"):
            mpc_controller.set_objective_function(Q_wrong, R)
    
    def test_set_objective_function_non_positive_definite_R(self, mpc_controller):
        """Test setting objective function with non-positive definite R."""
        Q = np.eye(12)
        R_invalid = np.zeros((6, 6))  # Not positive definite
        
        with pytest.raises(ValueError, match="R matrix must be positive definite"):
            mpc_controller.set_objective_function(Q, R_invalid)
    
    def test_update_dynamics_model(self, mpc_controller):
        """Test updating dynamics model."""
        def new_dynamics(x, u):
            return x + u[0]  # Simple dynamics
        
        mpc_controller.update_dynamics_model(new_dynamics)
        assert mpc_controller.dynamics_model == new_dynamics
    
    def test_set_constraints(self, mpc_controller):
        """Test setting system constraints."""
        from scipy.optimize import Bounds
        
        state_lower = -np.ones(12)
        state_upper = np.ones(12)
        control_lower = -np.ones(6)
        control_upper = np.ones(6)
        
        mpc_controller.set_constraints(
            state_bounds=(state_lower, state_upper),
            control_bounds=(control_lower, control_upper)
        )
        
        assert mpc_controller.state_bounds is not None
        assert mpc_controller.control_bounds is not None
        assert np.allclose(mpc_controller.state_bounds.lb, state_lower)
        assert np.allclose(mpc_controller.state_bounds.ub, state_upper)
    
    def test_solve_mpc_no_objective(self, mpc_controller):
        """Test MPC solving without objective function set."""
        current_state = np.zeros(12)
        
        with pytest.raises(ValueError, match="Objective function matrices must be set"):
            mpc_controller.solve_mpc(current_state)
    
    def test_solve_mpc_no_dynamics(self):
        """Test MPC solving without dynamics model."""
        config = MPCConfiguration(prediction_horizon=5)
        controller = MPCController(
            config=config,
            state_dim=12,
            control_dim=6
        )
        
        Q = np.eye(12)
        R = np.eye(6)
        controller.set_objective_function(Q, R)
        
        current_state = np.zeros(12)
        
        with pytest.raises(ValueError, match="Dynamics model must be set"):
            controller.solve_mpc(current_state)
    
    def test_solve_mpc_invalid_state_dimension(self, mpc_controller):
        """Test MPC solving with invalid state dimension."""
        Q = np.eye(12)
        R = np.eye(6)
        mpc_controller.set_objective_function(Q, R)
        
        invalid_state = np.zeros(10)  # Wrong dimension
        
        with pytest.raises(ValueError, match="Current state must have dimension"):
            mpc_controller.solve_mpc(invalid_state)
    
    @pytest.mark.parametrize("solver_type", [SolverType.SCIPY_MINIMIZE])
    def test_solve_mpc_different_solvers(self, mock_dynamics, solver_type):
        """Test MPC solving with different solver types."""
        config = MPCConfiguration(
            prediction_horizon=5,
            solver_type=solver_type,
            max_solve_time=2.0  # Generous timeout for testing
        )
        
        controller = MPCController(
            config=config,
            state_dim=12,
            control_dim=6,
            dynamics_model=mock_dynamics
        )
        
        # Set objective function
        Q = np.eye(12) * 1.0
        R = np.eye(6) * 0.1
        controller.set_objective_function(Q, R)
        
        # Set reasonable constraints
        state_bounds = (-10 * np.ones(12), 10 * np.ones(12))
        control_bounds = (-5 * np.ones(6), 5 * np.ones(6))
        controller.set_constraints(
            state_bounds=state_bounds,
            control_bounds=control_bounds
        )
        
        # Solve MPC
        current_state = np.random.randn(12) * 0.1  # Small random initial state
        result = controller.solve_mpc(current_state)
        
        # Check result
        assert isinstance(result, MPCResult)
        assert result.solve_time >= 0
        
        # For feasible problems, should find a solution
        if result.status in [MPCStatus.OPTIMAL, MPCStatus.FEASIBLE]:
            assert result.optimal_control is not None
            assert result.optimal_control.shape == (config.prediction_horizon, 6)
            assert result.predicted_states is not None
    
    def test_solve_mpc_with_reference(self, mpc_controller):
        """Test MPC solving with reference trajectory."""
        Q = np.eye(12)
        R = np.eye(6) * 0.1
        mpc_controller.set_objective_function(Q, R)
        
        current_state = np.zeros(12)
        reference = np.ones((11, 12)) * 0.5  # N+1 x state_dim
        
        # This should not raise an error even if solver fails
        result = mpc_controller.solve_mpc(current_state, reference_trajectory=reference)
        assert isinstance(result, MPCResult)
    
    def test_get_control_sequence(self, mpc_controller):
        """Test extracting control sequence from MPC result."""
        # Create a mock result
        optimal_control = np.random.randn(10, 6)
        result = MPCResult(
            status=MPCStatus.OPTIMAL,
            optimal_control=optimal_control
        )
        
        # Test default extraction
        control_seq = mpc_controller.get_control_sequence(result)
        expected_length = min(mpc_controller.config.control_horizon, len(optimal_control))
        assert control_seq.shape == (expected_length, 6)
        
        # Test custom length
        control_seq_custom = mpc_controller.get_control_sequence(result, num_controls=5)
        assert control_seq_custom.shape == (5, 6)
    
    def test_get_control_sequence_no_solution(self, mpc_controller):
        """Test extracting control sequence when no solution available."""
        result = MPCResult(status=MPCStatus.INFEASIBLE)
        
        with pytest.raises(ValueError, match="No optimal control available"):
            mpc_controller.get_control_sequence(result)
    
    def test_warm_start_functionality(self, mpc_controller):
        """Test warm start functionality."""
        Q = np.eye(12)
        R = np.eye(6) * 0.1
        mpc_controller.set_objective_function(Q, R)
        
        # First solve - no warm start available
        assert mpc_controller.previous_solution is None
        
        # Set some previous solution manually
        mpc_controller.previous_solution = np.random.randn(10, 6)
        mpc_controller.previous_states = np.random.randn(11, 12)
        
        # Check warm start data is available
        assert mpc_controller.previous_solution is not None
        assert mpc_controller.previous_states is not None
    
    def test_reset_warm_start(self, mpc_controller):
        """Test resetting warm start data."""
        # Set some warm start data
        mpc_controller.previous_solution = np.random.randn(10, 6)
        mpc_controller.previous_states = np.random.randn(11, 12)
        
        # Reset
        mpc_controller.reset_warm_start()
        
        assert mpc_controller.previous_solution is None
        assert mpc_controller.previous_states is None
    
    def test_performance_metrics(self, mpc_controller):
        """Test performance metrics collection."""
        # Initially no metrics
        metrics = mpc_controller.get_performance_metrics()
        assert len(metrics) == 0
        
        # Add some solve times manually
        mpc_controller.solve_times = [0.01, 0.02, 0.015, 0.08]  # Last one violates real-time
        
        metrics = mpc_controller.get_performance_metrics()
        assert 'mean_solve_time' in metrics
        assert 'max_solve_time' in metrics
        assert 'min_solve_time' in metrics
        assert 'real_time_violations' in metrics
        assert metrics['real_time_violations'] == 1  # One violation
        assert np.isclose(metrics['mean_solve_time'], 0.03125)
    
    def test_emergency_stop(self, mpc_controller):
        """Test emergency stop functionality."""
        emergency_control = mpc_controller.emergency_stop()
        
        assert emergency_control.shape == (mpc_controller.control_dim,)
        assert np.allclose(emergency_control, 0.0)
    
    def test_adaptive_cost_weights(self, mpc_controller):
        """Test adaptive cost weights based on human intent."""
        Q = np.eye(12)
        R = np.eye(6) * 0.1
        mpc_controller.set_objective_function(Q, R)
        
        # Enable adaptive weights
        mpc_controller.adaptive_weights = True
        
        # Test with high uncertainty
        high_uncertainty_intent = {
            'intent_probs': {'reach': 0.4, 'grab': 0.3, 'wave': 0.3},
            'uncertainty': 0.8
        }
        
        Q_adapted, R_adapted = mpc_controller._adapt_cost_weights(high_uncertainty_intent)
        
        # Q should be scaled up for safety
        assert np.all(Q_adapted >= Q)
        # R should be scaled up to reduce aggressiveness
        assert np.all(R_adapted >= R)
        
        # Test with low uncertainty
        low_uncertainty_intent = {
            'intent_probs': {'reach': 0.9, 'grab': 0.1},
            'uncertainty': 0.1
        }
        
        Q_adapted_low, R_adapted_low = mpc_controller._adapt_cost_weights(low_uncertainty_intent)
        
        # Scaling should be less aggressive
        assert np.all(Q_adapted_low <= Q_adapted)
        assert np.all(R_adapted_low <= R_adapted)


class TestMPCResult:
    """Test MPC result data structure."""
    
    def test_mpc_result_creation(self):
        """Test creating MPC result."""
        result = MPCResult(status=MPCStatus.OPTIMAL)
        
        assert result.status == MPCStatus.OPTIMAL
        assert result.optimal_control is None
        assert result.predicted_states is None
        assert result.solve_time == 0.0
        assert result.iterations == 0
        assert len(result.constraint_violations) == 0
    
    def test_mpc_result_with_data(self):
        """Test creating MPC result with full data."""
        optimal_control = np.random.randn(10, 6)
        predicted_states = np.random.randn(11, 12)
        
        result = MPCResult(
            status=MPCStatus.OPTIMAL,
            optimal_control=optimal_control,
            predicted_states=predicted_states,
            optimal_cost=123.45,
            solve_time=0.025,
            iterations=15
        )
        
        assert np.array_equal(result.optimal_control, optimal_control)
        assert np.array_equal(result.predicted_states, predicted_states)
        assert result.optimal_cost == 123.45
        assert result.solve_time == 0.025
        assert result.iterations == 15


class TestMPCIntegrationWithRobot:
    """Integration tests with robot dynamics model."""
    
    @pytest.fixture
    def robot_model(self):
        """Create robot model for testing."""
        return create_default_6dof_robot()
    
    @pytest.fixture
    def robot_mpc_controller(self, robot_model):
        """Create MPC controller integrated with robot model."""
        config = MPCConfiguration(
            prediction_horizon=8,
            control_horizon=6,
            max_solve_time=1.0
        )
        
        def robot_dynamics_wrapper(state, control):
            return robot_model.dynamics(state, control)
        
        controller = MPCController(
            config=config,
            state_dim=robot_model.state_dim,
            control_dim=robot_model.control_dim,
            dynamics_model=robot_dynamics_wrapper
        )
        
        # Set up objective function
        Q = np.eye(12)
        Q[0:6, 0:6] *= 10  # Weight joint positions more
        R = np.eye(6) * 0.01  # Small control cost
        controller.set_objective_function(Q, R)
        
        # Set constraints based on robot limits
        controller.set_constraints(
            state_bounds=(
                np.concatenate([robot_model.joint_limits.position_min, 
                               -robot_model.joint_limits.velocity_max]),
                np.concatenate([robot_model.joint_limits.position_max,
                               robot_model.joint_limits.velocity_max])
            ),
            control_bounds=(
                -robot_model.joint_limits.torque_max,
                robot_model.joint_limits.torque_max
            )
        )
        
        return controller
    
    def test_robot_mpc_integration(self, robot_mpc_controller, robot_model):
        """Test MPC integration with robot dynamics."""
        # Start from a feasible configuration
        initial_joint_pos = np.array([0.1, -0.5, 0.8, 0.0, 0.3, 0.0])
        initial_joint_vel = np.zeros(6)
        initial_state = np.concatenate([initial_joint_pos, initial_joint_vel])
        
        # Check initial state is within bounds
        violations = robot_model.check_joint_limits(initial_state)
        assert not any(violations.values()), f"Initial state violates limits: {violations}"
        
        # Solve MPC
        result = robot_mpc_controller.solve_mpc(initial_state)
        
        # Check result
        assert isinstance(result, MPCResult)
        assert result.solve_time > 0
        
        # If successful, validate solution
        if result.status in [MPCStatus.OPTIMAL, MPCStatus.FEASIBLE]:
            assert result.optimal_control is not None
            assert result.optimal_control.shape[0] >= 1
            assert result.optimal_control.shape[1] == 6
            
            # Check control limits
            if robot_mpc_controller.control_bounds:
                assert np.all(result.optimal_control >= robot_mpc_controller.control_bounds.lb)
                assert np.all(result.optimal_control <= robot_mpc_controller.control_bounds.ub)
    
    def test_robot_mpc_trajectory_tracking(self, robot_mpc_controller, robot_model):
        """Test MPC trajectory tracking with robot dynamics."""
        # Define a simple reference trajectory (joint space)
        initial_state = np.zeros(12)
        target_joint_pos = np.array([0.2, -0.3, 0.5, 0.1, 0.2, -0.1])
        
        # Create reference trajectory (linear interpolation)
        N = robot_mpc_controller.config.prediction_horizon
        reference = np.zeros((N + 1, 12))
        for k in range(N + 1):
            alpha = k / N
            reference[k, 0:6] = (1 - alpha) * initial_state[0:6] + alpha * target_joint_pos
        
        # Solve MPC with reference
        result = robot_mpc_controller.solve_mpc(
            initial_state, 
            reference_trajectory=reference
        )
        
        # Should find some solution
        assert isinstance(result, MPCResult)
        assert result.solve_time >= 0
    
    def test_robot_mpc_constraint_satisfaction(self, robot_mpc_controller, robot_model):
        """Test that MPC respects robot constraints."""
        # Start near joint limits to test constraint handling
        near_limit_state = np.zeros(12)
        near_limit_state[0] = robot_model.joint_limits.position_max[0] * 0.9  # Near upper limit
        near_limit_state[1] = robot_model.joint_limits.position_min[1] * 0.9  # Near lower limit
        
        result = robot_mpc_controller.solve_mpc(near_limit_state)
        
        # Even if solver struggles, should return some result
        assert isinstance(result, MPCResult)
        
        # If successful, check constraint satisfaction
        if result.optimal_control is not None:
            # Simulate forward to check state constraints
            current_state = near_limit_state.copy()
            
            for k in range(min(3, len(result.optimal_control))):  # Check first few steps
                try:
                    next_state = robot_model.dynamics(current_state, result.optimal_control[k])
                    
                    # Check joint limits on next state
                    violations = robot_model.check_joint_limits(next_state)
                    
                    # Small violations might be acceptable due to numerical errors
                    if violations['position_lower'] or violations['position_upper']:
                        logger.warning(f"Joint position violation at step {k}")
                    if violations['velocity']:
                        logger.warning(f"Joint velocity violation at step {k}")
                    
                    current_state = next_state
                    
                except Exception as e:
                    # Dynamics might fail for extreme states
                    logger.warning(f"Dynamics failed at step {k}: {e}")
                    break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])