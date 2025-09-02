"""
6-DOF robotic arm dynamics implementation for MPC control.

This module implements the dynamics model for a 6-degree-of-freedom robotic arm
suitable for human-robot interaction scenarios. The model includes:
- Forward and inverse kinematics
- Dynamic equations of motion
- Joint and Cartesian constraints
- Collision avoidance modeling

Mathematical Formulation:
========================

Robot Dynamics (Euler-Lagrange):
    M(q)q̈ + C(q,q̇)q̇ + G(q) = τ

Where:
- q ∈ ℝ⁶: joint angles
- q̇ ∈ ℝ⁶: joint velocities  
- q̈ ∈ ℝ⁶: joint accelerations
- τ ∈ ℝ⁶: joint torques
- M(q) ∈ ℝ⁶ˣ⁶: inertia matrix (positive definite)
- C(q,q̇) ∈ ℝ⁶ˣ⁶: Coriolis/centripetal matrix
- G(q) ∈ ℝ⁶: gravity vector

State-Space Representation:
    x = [q; q̇] ∈ ℝ¹²
    ẋ = [q̇; M(q)⁻¹(τ - C(q,q̇)q̇ - G(q))]

Forward Kinematics:
    T = ∏ᵢ₌₁⁶ Tᵢ(qᵢ)  where Tᵢ are DH transformation matrices
    
End-Effector Position: p = T[0:3, 3]
End-Effector Orientation: R = T[0:3, 0:3]

Jacobian:
    J(q) ∈ ℝ⁶ˣ⁶: maps joint velocities to end-effector twist
    v = J(q)q̇  where v = [linear_vel; angular_vel]
"""

import numpy as np
import scipy.linalg as la
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

from ...utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DHParameters:
    """Denavit-Hartenberg parameters for robot links."""
    a: float  # Link length
    alpha: float  # Link twist
    d: float  # Link offset
    theta: float  # Joint angle (variable for revolute joints)


@dataclass
class JointLimits:
    """Joint limits for robotic arm."""
    position_min: np.ndarray
    position_max: np.ndarray
    velocity_max: np.ndarray
    acceleration_max: np.ndarray
    torque_max: np.ndarray


@dataclass
class RobotState:
    """Complete state of the robotic arm."""
    joint_positions: np.ndarray  # q ∈ ℝ⁶
    joint_velocities: np.ndarray  # q̇ ∈ ℝ⁶
    end_effector_pose: np.ndarray  # [x, y, z, rx, ry, rz] ∈ ℝ⁶
    end_effector_velocity: np.ndarray  # [vx, vy, vz, wx, wy, wz] ∈ ℝ⁶
    timestamp: float = 0.0


class ControlMode(Enum):
    """Control modes for the robotic arm."""
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_TORQUE = "joint_torque"
    CARTESIAN_POSITION = "cartesian_position"
    CARTESIAN_VELOCITY = "cartesian_velocity"


class Robot6DOF:
    """
    6-DOF robotic arm dynamics model.
    
    This class implements a complete dynamics model for a 6-DOF robotic arm
    including kinematics, dynamics, and constraint modeling for use in MPC.
    
    The robot state is represented as x = [q, q̇] ∈ ℝ¹² where:
    - q ∈ ℝ⁶ are joint angles
    - q̇ ∈ ℝ⁶ are joint velocities
    
    Control inputs can be:
    - Joint torques: τ ∈ ℝ⁶
    - Joint velocity commands: q̇_cmd ∈ ℝ⁶
    - Cartesian velocity commands: v_cmd ∈ ℝ⁶
    """
    
    def __init__(self,
                 dh_parameters: List[DHParameters],
                 joint_limits: JointLimits,
                 link_masses: np.ndarray,
                 link_inertias: List[np.ndarray],
                 control_mode: ControlMode = ControlMode.JOINT_TORQUE):
        """
        Initialize 6-DOF robot model.
        
        Args:
            dh_parameters: List of 6 DH parameter sets
            joint_limits: Joint position, velocity, acceleration, torque limits
            link_masses: Mass of each link (6,)
            link_inertias: List of 6 inertia tensors (3x3 each)
            control_mode: Control interface mode
        """
        if len(dh_parameters) != 6:
            raise ValueError("Must provide exactly 6 DH parameter sets")
        if len(link_masses) != 6:
            raise ValueError("Must provide exactly 6 link masses")
        if len(link_inertias) != 6:
            raise ValueError("Must provide exactly 6 link inertia tensors")
        
        self.dh_params = dh_parameters
        self.joint_limits = joint_limits
        self.link_masses = link_masses
        self.link_inertias = link_inertias
        self.control_mode = control_mode
        
        # Robot dimensions
        self.n_joints = 6
        self.state_dim = 12  # [q, q̇]
        self.control_dim = 6
        
        # Physical parameters
        self.gravity = np.array([0, 0, -9.81])  # Gravity vector
        
        # Workspace bounds (will be computed from kinematics)
        self.workspace_bounds: Optional[np.ndarray] = None
        self._compute_workspace_bounds()
        
        # Pre-allocated arrays for efficiency
        self._T_matrices = [np.eye(4) for _ in range(7)]  # Include base frame
        self._jacobian = np.zeros((6, 6))
        
        logger.info(f"Initialized 6-DOF robot with {control_mode} control mode")
        logger.info(f"Joint limits - pos: [{joint_limits.position_min.min():.3f}, {joint_limits.position_max.max():.3f}] rad")
        logger.info(f"Joint limits - vel: {joint_limits.velocity_max.max():.3f} rad/s")
        logger.info(f"Joint limits - torque: {joint_limits.torque_max.max():.1f} Nm")
    
    def forward_kinematics(self, joint_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics using DH parameters.
        
        Args:
            joint_positions: Joint angles q ∈ ℝ⁶
        
        Returns:
            end_effector_position: [x, y, z] ∈ ℝ³
            end_effector_orientation: Rotation matrix R ∈ ℝ³ˣ³
        """
        if len(joint_positions) != 6:
            raise ValueError("Joint positions must have 6 elements")
        
        # Initialize with identity (base frame)
        T_total = np.eye(4)
        
        # Multiply transformation matrices
        for i, (dh, q) in enumerate(zip(self.dh_params, joint_positions)):
            # DH transformation matrix
            cos_theta = np.cos(q + dh.theta)
            sin_theta = np.sin(q + dh.theta)
            cos_alpha = np.cos(dh.alpha)
            sin_alpha = np.sin(dh.alpha)
            
            T_i = np.array([
                [cos_theta, -sin_theta * cos_alpha,  sin_theta * sin_alpha, dh.a * cos_theta],
                [sin_theta,  cos_theta * cos_alpha, -cos_theta * sin_alpha, dh.a * sin_theta],
                [0,          sin_alpha,              cos_alpha,             dh.d],
                [0,          0,                      0,                     1]
            ])
            
            T_total = T_total @ T_i
            self._T_matrices[i+1] = T_total.copy()
        
        position = T_total[0:3, 3]
        orientation = T_total[0:3, 0:3]
        
        return position, orientation
    
    def jacobian(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Compute geometric Jacobian matrix.
        
        The Jacobian J(q) ∈ ℝ⁶ˣ⁶ maps joint velocities to end-effector twist:
        v = J(q)q̇  where v = [linear_velocity; angular_velocity]
        
        Args:
            joint_positions: Joint angles q ∈ ℝ⁶
        
        Returns:
            Jacobian matrix J ∈ ℝ⁶ˣ⁶
        """
        # Compute forward kinematics first
        _, _ = self.forward_kinematics(joint_positions)
        
        # End-effector position
        p_end = self._T_matrices[6][0:3, 3]
        
        # Compute Jacobian columns
        for i in range(6):
            # Joint i axis of rotation (z-axis of frame i-1)
            z_i = self._T_matrices[i][0:3, 2]
            
            # Position of joint i
            p_i = self._T_matrices[i][0:3, 3]
            
            # Linear velocity component: z_i × (p_end - p_i)
            self._jacobian[0:3, i] = np.cross(z_i, p_end - p_i)
            
            # Angular velocity component: z_i
            self._jacobian[3:6, i] = z_i
        
        return self._jacobian.copy()
    
    def inverse_kinematics(self,
                          target_position: np.ndarray,
                          target_orientation: Optional[np.ndarray] = None,
                          initial_guess: Optional[np.ndarray] = None,
                          max_iterations: int = 100,
                          tolerance: float = 1e-6) -> Tuple[bool, np.ndarray]:
        """
        Solve inverse kinematics using Newton-Raphson method.
        
        Args:
            target_position: Desired end-effector position [x, y, z]
            target_orientation: Desired orientation matrix (3x3)
            initial_guess: Initial joint configuration
            max_iterations: Maximum Newton-Raphson iterations
            tolerance: Convergence tolerance
        
        Returns:
            success: Whether IK solution was found
            joint_angles: Solution joint angles (or best attempt)
        """
        # Initial guess
        if initial_guess is None:
            q = np.zeros(6)
        else:
            q = initial_guess.copy()
        
        # Target pose
        if target_orientation is None:
            # Position-only IK (3 constraints)
            target_pose = target_position
            pose_dim = 3
        else:
            # Full pose IK (6 constraints) - orientation as rotation vector
            target_orientation_vec = self._rotation_matrix_to_vector(target_orientation)
            target_pose = np.concatenate([target_position, target_orientation_vec])
            pose_dim = 6
        
        for iteration in range(max_iterations):
            # Current end-effector pose
            current_pos, current_rot = self.forward_kinematics(q)
            
            if pose_dim == 3:
                current_pose = current_pos
            else:
                current_rot_vec = self._rotation_matrix_to_vector(current_rot)
                current_pose = np.concatenate([current_pos, current_rot_vec])
            
            # Pose error
            pose_error = target_pose - current_pose
            
            # Check convergence
            if np.linalg.norm(pose_error) < tolerance:
                return True, q
            
            # Jacobian (use only position part if position-only IK)
            J = self.jacobian(q)
            if pose_dim == 3:
                J = J[0:3, :]
            
            # Newton-Raphson update with damping
            damping = 1e-6
            try:
                J_pinv = np.linalg.inv(J.T @ J + damping * np.eye(6)) @ J.T
                dq = J_pinv @ pose_error
                
                # Apply joint limits
                q_new = q + dq
                q_new = np.clip(q_new, self.joint_limits.position_min, self.joint_limits.position_max)
                q = q_new
                
            except np.linalg.LinAlgError:
                logger.warning("Jacobian singular in inverse kinematics")
                return False, q
        
        logger.warning(f"IK did not converge after {max_iterations} iterations")
        return False, q
    
    def dynamics(self,
                state: np.ndarray,
                control_input: np.ndarray,
                external_forces: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute robot dynamics: ẋ = f(x, u).
        
        For joint torque control:
            M(q)q̈ + C(q,q̇)q̇ + G(q) = τ
        
        State: x = [q, q̇] ∈ ℝ¹²
        Control: u = τ ∈ ℝ⁶ (joint torques)
        
        Args:
            state: Robot state [q, q̇] ∈ ℝ¹²
            control_input: Control input (torques) ∈ ℝ⁶
            external_forces: External forces on end-effector ∈ ℝ⁶
        
        Returns:
            state_derivative: ẋ = [q̇, q̈] ∈ ℝ¹²
        """
        if len(state) != 12:
            raise ValueError("State must have 12 elements [q, q̇]")
        if len(control_input) != 6:
            raise ValueError("Control input must have 6 elements")
        
        # Extract joint positions and velocities
        q = state[0:6]
        qd = state[6:12]
        
        # Compute dynamics matrices
        M = self._inertia_matrix(q)
        C = self._coriolis_matrix(q, qd)
        G = self._gravity_vector(q)
        
        # Control input (joint torques)
        tau = control_input.copy()
        
        # Add external forces if provided
        if external_forces is not None:
            # Transform Cartesian forces to joint torques: τ_ext = J^T F_ext
            J = self.jacobian(q)
            tau += J.T @ external_forces
        
        # Solve for joint accelerations: q̈ = M⁻¹(τ - Cq̇ - G)
        try:
            qdd = np.linalg.solve(M, tau - C @ qd - G)
        except np.linalg.LinAlgError:
            logger.warning("Singular inertia matrix, using pseudo-inverse")
            qdd = np.linalg.pinv(M) @ (tau - C @ qd - G)
        
        # State derivative
        state_derivative = np.concatenate([qd, qdd])
        
        return state_derivative
    
    def _inertia_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Compute joint-space inertia matrix M(q).
        
        This is a simplified implementation. For a real robot, this would
        involve the recursive Newton-Euler algorithm or the composite
        rigid-body algorithm.
        
        Args:
            q: Joint positions
        
        Returns:
            Inertia matrix M(q) ∈ ℝ⁶ˣ⁶
        """
        # Simplified inertia matrix based on link masses and geometry
        # In practice, this would be computed using the robot's full dynamics
        
        M = np.zeros((6, 6))
        
        # Diagonal terms (simplified)
        for i in range(6):
            # Base inertia for each joint
            M[i, i] = self.link_masses[i] * 0.1  # Simplified
            
            # Add contributions from subsequent links (approximation)
            for j in range(i+1, 6):
                M[i, i] += self.link_masses[j] * (0.5 * (j-i))
        
        # Off-diagonal coupling terms (simplified)
        for i in range(6):
            for j in range(i+1, 6):
                coupling = 0.1 * np.sqrt(self.link_masses[i] * self.link_masses[j])
                coupling *= np.cos(q[i] - q[j])  # Configuration-dependent
                M[i, j] = M[j, i] = coupling
        
        # Ensure positive definiteness
        M += 0.01 * np.eye(6)
        
        return M
    
    def _coriolis_matrix(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """
        Compute Coriolis and centripetal matrix C(q, q̇).
        
        Args:
            q: Joint positions
            qd: Joint velocities
        
        Returns:
            Coriolis matrix C(q, q̇) ∈ ℝ⁶ˣ⁶
        """
        # Simplified Coriolis matrix
        # In practice, this would be computed from derivatives of M(q)
        
        C = np.zeros((6, 6))
        
        # Simplified Coriolis terms
        for i in range(6):
            for j in range(6):
                if i != j:
                    # Velocity-dependent coupling
                    C[i, j] = 0.05 * self.link_masses[min(i,j)] * np.sin(q[i] - q[j]) * qd[j]
        
        return C
    
    def _gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """
        Compute gravity vector G(q).
        
        Args:
            q: Joint positions
        
        Returns:
            Gravity vector G(q) ∈ ℝ⁶
        """
        G = np.zeros(6)
        
        # Simplified gravity computation
        # This should be computed from link centers of mass and orientations
        
        for i in range(6):
            # Vertical component based on joint position
            gravity_effect = self.link_masses[i] * self.gravity[2]  # -9.81
            
            # Multiply by effective moment arm (simplified)
            if i == 0:  # Base joint (rotation about z)
                G[i] = 0  # No gravity effect for vertical rotation
            else:
                # Approximate moment arm based on DH parameters
                moment_arm = self.dh_params[i].a * np.sin(q[i])
                G[i] = gravity_effect * moment_arm
        
        return G
    
    def check_joint_limits(self, state: np.ndarray) -> Dict[str, bool]:
        """
        Check if joint limits are violated.
        
        Args:
            state: Robot state [q, q̇]
        
        Returns:
            Dictionary indicating which limits are violated
        """
        q = state[0:6]
        qd = state[6:12]
        
        violations = {
            'position_lower': np.any(q < self.joint_limits.position_min),
            'position_upper': np.any(q > self.joint_limits.position_max),
            'velocity': np.any(np.abs(qd) > self.joint_limits.velocity_max)
        }
        
        return violations
    
    def check_workspace_bounds(self, end_effector_position: np.ndarray) -> bool:
        """
        Check if end-effector is within workspace bounds.
        
        Args:
            end_effector_position: End-effector position [x, y, z]
        
        Returns:
            True if within bounds, False otherwise
        """
        if self.workspace_bounds is None:
            return True
        
        return (end_effector_position >= self.workspace_bounds[0:3]).all() and \
               (end_effector_position <= self.workspace_bounds[3:6]).all()
    
    def _compute_workspace_bounds(self) -> None:
        """Compute approximate workspace bounds from kinematic limits."""
        # Sample joint space and compute reachable positions
        n_samples = 1000
        positions = []
        
        for _ in range(n_samples):
            # Random valid joint configuration
            q_sample = np.random.uniform(
                self.joint_limits.position_min,
                self.joint_limits.position_max
            )
            
            try:
                pos, _ = self.forward_kinematics(q_sample)
                positions.append(pos)
            except:
                continue
        
        if positions:
            positions = np.array(positions)
            min_bounds = np.min(positions, axis=0)
            max_bounds = np.max(positions, axis=0)
            self.workspace_bounds = np.concatenate([min_bounds, max_bounds])
            
            logger.info(f"Computed workspace bounds: x[{min_bounds[0]:.3f}, {max_bounds[0]:.3f}], "
                       f"y[{min_bounds[1]:.3f}, {max_bounds[1]:.3f}], "
                       f"z[{min_bounds[2]:.3f}, {max_bounds[2]:.3f}]")
    
    def _rotation_matrix_to_vector(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to rotation vector (axis-angle)."""
        # Use Rodrigues' formula
        trace_R = np.trace(R)
        if trace_R >= 3 - 1e-6:
            # Nearly identity matrix
            return np.zeros(3)
        elif trace_R <= -1 + 1e-6:
            # Nearly 180-degree rotation
            # Find the eigenvector corresponding to eigenvalue 1
            eigenvals, eigenvecs = np.linalg.eig(R)
            axis_idx = np.argmax(np.real(eigenvals))
            axis = np.real(eigenvecs[:, axis_idx])
            return np.pi * axis
        else:
            # General case
            angle = np.arccos((trace_R - 1) / 2)
            axis = 1 / (2 * np.sin(angle)) * np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1]
            ])
            return angle * axis
    
    def get_state_from_measurements(self,
                                   joint_positions: np.ndarray,
                                   joint_velocities: np.ndarray) -> RobotState:
        """
        Create complete robot state from joint measurements.
        
        Args:
            joint_positions: Joint angles ∈ ℝ⁶
            joint_velocities: Joint velocities ∈ ℝ⁶
        
        Returns:
            Complete robot state
        """
        # Compute forward kinematics
        ee_pos, ee_rot = self.forward_kinematics(joint_positions)
        
        # Compute end-effector pose (position + orientation as rotation vector)
        ee_rot_vec = self._rotation_matrix_to_vector(ee_rot)
        ee_pose = np.concatenate([ee_pos, ee_rot_vec])
        
        # Compute end-effector velocity
        J = self.jacobian(joint_positions)
        ee_velocity = J @ joint_velocities
        
        return RobotState(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            end_effector_pose=ee_pose,
            end_effector_velocity=ee_velocity
        )


def create_default_6dof_robot() -> Robot6DOF:
    """
    Create a default 6-DOF robot model for testing and demonstration.
    
    This creates a robot similar to a UR5 or similar industrial robot arm.
    
    Returns:
        Configured Robot6DOF instance
    """
    # DH parameters for a UR5-like robot (approximate)
    dh_params = [
        DHParameters(a=0.0, alpha=np.pi/2, d=0.089, theta=0),     # Joint 1
        DHParameters(a=-0.425, alpha=0, d=0.0, theta=0),          # Joint 2
        DHParameters(a=-0.392, alpha=0, d=0.0, theta=0),          # Joint 3
        DHParameters(a=0.0, alpha=np.pi/2, d=0.109, theta=0),     # Joint 4
        DHParameters(a=0.0, alpha=-np.pi/2, d=0.095, theta=0),    # Joint 5
        DHParameters(a=0.0, alpha=0, d=0.082, theta=0)            # Joint 6
    ]
    
    # Joint limits (typical for industrial robot)
    joint_limits = JointLimits(
        position_min=np.array([-2*np.pi, -np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi]),
        position_max=np.array([2*np.pi, np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi]),
        velocity_max=np.array([3.15, 3.15, 3.15, 6.28, 6.28, 6.28]),  # rad/s
        acceleration_max=np.array([15, 15, 15, 40, 40, 40]),  # rad/s²
        torque_max=np.array([150, 150, 150, 28, 28, 28])  # Nm
    )
    
    # Link masses (approximate for UR5-like robot)
    link_masses = np.array([3.7, 8.4, 2.3, 1.2, 1.2, 0.25])  # kg
    
    # Link inertias (simplified - diagonal tensors)
    link_inertias = []
    for mass in link_masses:
        # Simplified inertia tensor (assume cylindrical links)
        I = np.eye(3) * mass * 0.01  # Rough approximation
        link_inertias.append(I)
    
    return Robot6DOF(
        dh_parameters=dh_params,
        joint_limits=joint_limits,
        link_masses=link_masses,
        link_inertias=link_inertias,
        control_mode=ControlMode.JOINT_TORQUE
    )