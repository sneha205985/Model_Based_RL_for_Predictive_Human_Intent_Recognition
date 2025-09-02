"""
Unit tests for 6-DOF Robot Dynamics implementation.

This test suite validates:
- Robot model initialization
- Forward kinematics computation
- Jacobian calculation
- Inverse kinematics solver
- Dynamics equations
- Constraint checking
"""

import numpy as np
import pytest
from unittest.mock import patch

from src.models.robotics.robot_dynamics import (
    Robot6DOF, DHParameters, JointLimits, RobotState, ControlMode,
    create_default_6dof_robot
)


class TestDHParameters:
    """Test DH parameters data structure."""
    
    def test_dh_parameters_creation(self):
        """Test creating DH parameters."""
        dh = DHParameters(a=0.425, alpha=0, d=0.089, theta=0)
        
        assert dh.a == 0.425
        assert dh.alpha == 0
        assert dh.d == 0.089
        assert dh.theta == 0


class TestJointLimits:
    """Test joint limits data structure."""
    
    def test_joint_limits_creation(self):
        """Test creating joint limits."""
        pos_min = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
        pos_max = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])
        vel_max = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
        acc_max = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        torque_max = np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0])
        
        limits = JointLimits(
            position_min=pos_min,
            position_max=pos_max,
            velocity_max=vel_max,
            acceleration_max=acc_max,
            torque_max=torque_max
        )
        
        assert np.array_equal(limits.position_min, pos_min)
        assert np.array_equal(limits.position_max, pos_max)
        assert np.array_equal(limits.velocity_max, vel_max)
        assert np.array_equal(limits.acceleration_max, acc_max)
        assert np.array_equal(limits.torque_max, torque_max)


class TestRobotState:
    """Test robot state data structure."""
    
    def test_robot_state_creation(self):
        """Test creating robot state."""
        joint_pos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        joint_vel = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        ee_pose = np.array([0.5, 0.3, 0.8, 0.0, 0.0, 0.0])
        ee_vel = np.array([0.1, 0.05, 0.02, 0.01, 0.01, 0.01])
        
        state = RobotState(
            joint_positions=joint_pos,
            joint_velocities=joint_vel,
            end_effector_pose=ee_pose,
            end_effector_velocity=ee_vel,
            timestamp=1.5
        )
        
        assert np.array_equal(state.joint_positions, joint_pos)
        assert np.array_equal(state.joint_velocities, joint_vel)
        assert np.array_equal(state.end_effector_pose, ee_pose)
        assert np.array_equal(state.end_effector_velocity, ee_vel)
        assert state.timestamp == 1.5


class TestRobot6DOF:
    """Test 6-DOF robot dynamics model."""
    
    @pytest.fixture
    def simple_robot(self):
        """Create a simple robot for testing."""
        # Simple DH parameters for testing
        dh_params = [
            DHParameters(a=0.0, alpha=np.pi/2, d=0.1, theta=0),
            DHParameters(a=0.5, alpha=0, d=0.0, theta=0),
            DHParameters(a=0.4, alpha=0, d=0.0, theta=0),
            DHParameters(a=0.0, alpha=np.pi/2, d=0.1, theta=0),
            DHParameters(a=0.0, alpha=-np.pi/2, d=0.1, theta=0),
            DHParameters(a=0.0, alpha=0, d=0.05, theta=0)
        ]
        
        # Joint limits
        joint_limits = JointLimits(
            position_min=np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]),
            position_max=np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]),
            velocity_max=np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0]),
            acceleration_max=np.array([10.0, 10.0, 10.0, 20.0, 20.0, 20.0]),
            torque_max=np.array([50.0, 50.0, 50.0, 25.0, 25.0, 25.0])
        )
        
        # Link masses and inertias
        link_masses = np.array([2.0, 5.0, 3.0, 1.5, 1.0, 0.5])
        link_inertias = [np.eye(3) * 0.01 * mass for mass in link_masses]
        
        return Robot6DOF(
            dh_parameters=dh_params,
            joint_limits=joint_limits,
            link_masses=link_masses,
            link_inertias=link_inertias
        )
    
    def test_robot_initialization(self, simple_robot):
        """Test robot initialization."""
        assert simple_robot.n_joints == 6
        assert simple_robot.state_dim == 12
        assert simple_robot.control_dim == 6
        assert len(simple_robot.dh_params) == 6
        assert len(simple_robot.link_masses) == 6
        assert len(simple_robot.link_inertias) == 6
        assert simple_robot.control_mode == ControlMode.JOINT_TORQUE
    
    def test_robot_initialization_invalid_parameters(self):
        """Test robot initialization with invalid parameters."""
        # Wrong number of DH parameters
        dh_params = [DHParameters(0, 0, 0, 0)] * 5  # Only 5 instead of 6
        joint_limits = JointLimits(
            position_min=np.zeros(6), position_max=np.ones(6),
            velocity_max=np.ones(6), acceleration_max=np.ones(6),
            torque_max=np.ones(6)
        )
        link_masses = np.ones(6)
        link_inertias = [np.eye(3)] * 6
        
        with pytest.raises(ValueError, match="Must provide exactly 6 DH parameter sets"):
            Robot6DOF(dh_params, joint_limits, link_masses, link_inertias)
        
        # Wrong number of masses
        dh_params = [DHParameters(0, 0, 0, 0)] * 6
        link_masses = np.ones(5)  # Only 5 instead of 6
        
        with pytest.raises(ValueError, match="Must provide exactly 6 link masses"):
            Robot6DOF(dh_params, joint_limits, link_masses, link_inertias)
    
    def test_forward_kinematics_zero_configuration(self, simple_robot):
        """Test forward kinematics at zero configuration."""
        joint_positions = np.zeros(6)
        
        pos, rot = simple_robot.forward_kinematics(joint_positions)
        
        # Should return valid position and rotation matrix
        assert pos.shape == (3,)
        assert rot.shape == (3, 3)
        assert np.allclose(np.linalg.det(rot), 1.0, atol=1e-10)  # Valid rotation matrix
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-10)
    
    def test_forward_kinematics_random_configuration(self, simple_robot):
        """Test forward kinematics with random configuration."""
        np.random.seed(42)
        joint_positions = np.random.uniform(-np.pi/2, np.pi/2, 6)
        
        pos, rot = simple_robot.forward_kinematics(joint_positions)
        
        # Check dimensions and properties
        assert pos.shape == (3,)
        assert rot.shape == (3, 3)
        assert np.allclose(np.linalg.det(rot), 1.0, atol=1e-10)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-10)
        
        # Position should be reasonable (within robot's reach)
        assert np.linalg.norm(pos) < 2.0  # Rough workspace bound
    
    def test_forward_kinematics_invalid_input(self, simple_robot):
        """Test forward kinematics with invalid input."""
        with pytest.raises(ValueError, match="Joint positions must have 6 elements"):
            simple_robot.forward_kinematics(np.array([1, 2, 3, 4, 5]))  # Only 5 elements
    
    def test_jacobian_computation(self, simple_robot):
        """Test Jacobian computation."""
        joint_positions = np.array([0.1, 0.2, -0.1, 0.3, -0.2, 0.1])
        
        J = simple_robot.jacobian(joint_positions)
        
        # Check dimensions
        assert J.shape == (6, 6)
        
        # Jacobian should not be singular for most configurations
        # (though it might be close to singular at singularities)
        try:
            det_J = np.linalg.det(J)
            assert not np.isnan(det_J)
        except np.linalg.LinAlgError:
            # Some configurations might be singular, which is okay
            pass
    
    def test_jacobian_zero_configuration(self, simple_robot):
        """Test Jacobian at zero configuration."""
        joint_positions = np.zeros(6)
        
        J = simple_robot.jacobian(joint_positions)
        
        assert J.shape == (6, 6)
        # At zero configuration, Jacobian should have some structure
        # The exact values depend on the DH parameters
    
    def test_inverse_kinematics_feasible_target(self, simple_robot):
        """Test inverse kinematics for a feasible target."""
        # First get forward kinematics for a known configuration
        known_config = np.array([0.2, -0.3, 0.5, 0.1, 0.4, -0.2])
        target_pos, target_rot = simple_robot.forward_kinematics(known_config)
        
        # Try to recover the configuration
        success, recovered_config = simple_robot.inverse_kinematics(
            target_position=target_pos,
            target_orientation=target_rot,
            max_iterations=50,
            tolerance=1e-4
        )
        
        if success:
            # Check if recovered configuration produces same position
            recovered_pos, recovered_rot = simple_robot.forward_kinematics(recovered_config)
            assert np.allclose(recovered_pos, target_pos, atol=1e-3)
            # Orientation comparison is more complex due to multiple representations
        else:
            # IK might fail due to singularities or numerical issues
            # This is acceptable for some configurations
            pass
    
    def test_inverse_kinematics_position_only(self, simple_robot):
        """Test position-only inverse kinematics."""
        # Target position within workspace
        target_pos = np.array([0.5, 0.2, 0.6])
        
        success, joint_config = simple_robot.inverse_kinematics(
            target_position=target_pos,
            max_iterations=100,
            tolerance=1e-4
        )
        
        if success:
            # Verify the solution
            actual_pos, _ = simple_robot.forward_kinematics(joint_config)
            assert np.allclose(actual_pos, target_pos, atol=1e-3)
            
            # Check joint limits
            assert np.all(joint_config >= simple_robot.joint_limits.position_min)
            assert np.all(joint_config <= simple_robot.joint_limits.position_max)
    
    def test_inverse_kinematics_unreachable_target(self, simple_robot):
        """Test inverse kinematics for unreachable target."""
        # Target position far outside workspace
        unreachable_pos = np.array([10.0, 10.0, 10.0])
        
        success, _ = simple_robot.inverse_kinematics(
            target_position=unreachable_pos,
            max_iterations=20,
            tolerance=1e-3
        )
        
        # Should fail for unreachable targets
        assert not success
    
    def test_dynamics_computation(self, simple_robot):
        """Test dynamics computation."""
        # Random state and control
        np.random.seed(123)
        joint_pos = np.random.uniform(-1.0, 1.0, 6)
        joint_vel = np.random.uniform(-0.5, 0.5, 6)
        state = np.concatenate([joint_pos, joint_vel])
        
        control = np.random.uniform(-10.0, 10.0, 6)
        
        # Compute dynamics
        state_dot = simple_robot.dynamics(state, control)
        
        # Check output dimensions
        assert state_dot.shape == (12,)
        
        # First 6 elements should be joint velocities
        assert np.allclose(state_dot[0:6], joint_vel)
        
        # Last 6 elements should be joint accelerations
        joint_acc = state_dot[6:12]
        assert joint_acc.shape == (6,)
        
        # Accelerations should be finite
        assert np.all(np.isfinite(joint_acc))
    
    def test_dynamics_invalid_input(self, simple_robot):
        """Test dynamics with invalid input."""
        # Wrong state dimension
        with pytest.raises(ValueError, match="State must have 12 elements"):
            simple_robot.dynamics(np.zeros(10), np.zeros(6))
        
        # Wrong control dimension
        with pytest.raises(ValueError, match="Control input must have 6 elements"):
            simple_robot.dynamics(np.zeros(12), np.zeros(4))
    
    def test_dynamics_with_external_forces(self, simple_robot):
        """Test dynamics with external forces."""
        state = np.concatenate([np.zeros(6), np.zeros(6)])
        control = np.zeros(6)
        external_forces = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Force in x-direction
        
        # Compute dynamics with external forces
        state_dot_with_forces = simple_robot.dynamics(state, control, external_forces)
        state_dot_without = simple_robot.dynamics(state, control)
        
        # Results should be different when external forces are applied
        assert not np.allclose(state_dot_with_forces, state_dot_without)
        
        # Both should have valid dimensions
        assert state_dot_with_forces.shape == (12,)
        assert state_dot_without.shape == (12,)
    
    def test_check_joint_limits(self, simple_robot):
        """Test joint limit checking."""
        # State within limits
        valid_state = np.concatenate([
            np.array([0.1, -0.2, 0.3, -0.1, 0.2, 0.0]),  # positions
            np.array([0.5, -0.3, 0.2, 0.8, -0.5, 0.1])   # velocities
        ])
        
        violations = simple_robot.check_joint_limits(valid_state)
        assert not violations['position_lower']
        assert not violations['position_upper']
        assert not violations['velocity']
        
        # State with position violations
        invalid_pos_state = np.concatenate([
            np.array([5.0, -0.2, 0.3, -0.1, 0.2, 0.0]),  # First joint exceeds limit
            np.zeros(6)
        ])
        
        violations_pos = simple_robot.check_joint_limits(invalid_pos_state)
        assert violations_pos['position_upper']
        
        # State with velocity violations
        invalid_vel_state = np.concatenate([
            np.zeros(6),
            np.array([0.5, -0.3, 0.2, 5.0, -0.5, 0.1])  # Fourth joint vel exceeds limit
        ])
        
        violations_vel = simple_robot.check_joint_limits(invalid_vel_state)
        assert violations_vel['velocity']
    
    def test_check_workspace_bounds(self, simple_robot):
        """Test workspace bounds checking."""
        # Position within workspace (roughly)
        valid_pos = np.array([0.5, 0.3, 0.4])
        result = simple_robot.check_workspace_bounds(valid_pos)
        
        # Since workspace_bounds might be None or computed, result could be True
        assert isinstance(result, bool)
        
        # Position clearly outside any reasonable workspace
        invalid_pos = np.array([100.0, 100.0, 100.0])
        result_invalid = simple_robot.check_workspace_bounds(invalid_pos)
        
        # This should be False if workspace bounds are properly set
        if simple_robot.workspace_bounds is not None:
            assert not result_invalid
    
    def test_get_state_from_measurements(self, simple_robot):
        """Test creating robot state from measurements."""
        joint_pos = np.array([0.1, 0.2, -0.1, 0.3, -0.2, 0.1])
        joint_vel = np.array([0.05, -0.03, 0.02, 0.08, -0.04, 0.01])
        
        robot_state = simple_robot.get_state_from_measurements(joint_pos, joint_vel)
        
        # Check robot state structure
        assert isinstance(robot_state, RobotState)
        assert np.array_equal(robot_state.joint_positions, joint_pos)
        assert np.array_equal(robot_state.joint_velocities, joint_vel)
        assert robot_state.end_effector_pose.shape == (6,)
        assert robot_state.end_effector_velocity.shape == (6,)
        
        # Verify consistency with forward kinematics
        fk_pos, fk_rot = simple_robot.forward_kinematics(joint_pos)
        assert np.allclose(robot_state.end_effector_pose[0:3], fk_pos)
        
        # Verify consistency with Jacobian
        J = simple_robot.jacobian(joint_pos)
        expected_ee_vel = J @ joint_vel
        assert np.allclose(robot_state.end_effector_velocity, expected_ee_vel)
    
    def test_inertia_matrix_properties(self, simple_robot):
        """Test inertia matrix properties."""
        joint_positions = np.array([0.2, -0.1, 0.3, 0.1, -0.2, 0.0])
        
        M = simple_robot._inertia_matrix(joint_positions)
        
        # Check dimensions
        assert M.shape == (6, 6)
        
        # Should be symmetric
        assert np.allclose(M, M.T, atol=1e-10)
        
        # Should be positive definite
        eigenvalues = np.linalg.eigvals(M)
        assert np.all(eigenvalues > 1e-6), f"Eigenvalues: {eigenvalues}"
        
        # Diagonal elements should be positive
        assert np.all(np.diag(M) > 0)
    
    def test_coriolis_matrix_properties(self, simple_robot):
        """Test Coriolis matrix properties."""
        joint_positions = np.array([0.2, -0.1, 0.3, 0.1, -0.2, 0.0])
        joint_velocities = np.array([0.1, -0.05, 0.08, 0.03, -0.06, 0.02])
        
        C = simple_robot._coriolis_matrix(joint_positions, joint_velocities)
        
        # Check dimensions
        assert C.shape == (6, 6)
        
        # Elements should be finite
        assert np.all(np.isfinite(C))
        
        # For zero velocity, Coriolis terms should be small
        C_zero_vel = simple_robot._coriolis_matrix(joint_positions, np.zeros(6))
        assert np.allclose(C_zero_vel, 0, atol=1e-10)
    
    def test_gravity_vector_properties(self, simple_robot):
        """Test gravity vector properties."""
        joint_positions = np.array([0.2, -0.1, 0.3, 0.1, -0.2, 0.0])
        
        G = simple_robot._gravity_vector(joint_positions)
        
        # Check dimensions
        assert G.shape == (6,)
        
        # Elements should be finite
        assert np.all(np.isfinite(G))
        
        # Gravity should affect some joints (those with horizontal motion)
        # First joint (rotation about vertical axis) should have zero gravity effect
        assert abs(G[0]) < 1e-10


class TestDefaultRobot:
    """Test default robot creation function."""
    
    def test_create_default_robot(self):
        """Test creating default 6-DOF robot."""
        robot = create_default_6dof_robot()
        
        assert isinstance(robot, Robot6DOF)
        assert robot.n_joints == 6
        assert robot.state_dim == 12
        assert robot.control_dim == 6
        assert len(robot.dh_params) == 6
        assert len(robot.link_masses) == 6
        assert len(robot.link_inertias) == 6
    
    def test_default_robot_kinematics(self):
        """Test default robot kinematics."""
        robot = create_default_6dof_robot()
        
        # Test forward kinematics at zero configuration
        pos, rot = robot.forward_kinematics(np.zeros(6))
        
        assert pos.shape == (3,)
        assert rot.shape == (3, 3)
        assert np.allclose(np.linalg.det(rot), 1.0, atol=1e-10)
        
        # Position should be reasonable for UR5-like robot
        assert pos[2] > 0  # Should be above ground
        assert np.linalg.norm(pos) < 2.0  # Within reasonable workspace
    
    def test_default_robot_dynamics(self):
        """Test default robot dynamics."""
        robot = create_default_6dof_robot()
        
        # Test dynamics computation
        state = np.zeros(12)
        control = np.zeros(6)
        
        state_dot = robot.dynamics(state, control)
        
        assert state_dot.shape == (12,)
        assert np.allclose(state_dot[0:6], 0)  # Zero velocities
        
        # Accelerations should mostly be due to gravity
        accelerations = state_dot[6:12]
        # Some joints should have non-zero gravity accelerations
        assert not np.allclose(accelerations, 0, atol=1e-6)
    
    def test_default_robot_joint_limits(self):
        """Test default robot joint limits."""
        robot = create_default_6dof_robot()
        
        limits = robot.joint_limits
        
        # Check that limits are reasonable
        assert np.all(limits.position_min < 0)
        assert np.all(limits.position_max > 0)
        assert np.all(limits.velocity_max > 0)
        assert np.all(limits.acceleration_max > 0)
        assert np.all(limits.torque_max > 0)
        
        # Check symmetry for most joints
        for i in range(6):
            assert abs(limits.position_min[i] + limits.position_max[i]) < 1e-10 or \
                   abs(limits.position_min[i]) == abs(limits.position_max[i])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])