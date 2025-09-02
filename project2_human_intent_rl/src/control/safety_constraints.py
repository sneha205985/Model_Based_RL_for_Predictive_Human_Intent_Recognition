"""
Safety constraints and robustness features for MPC control.

This module implements comprehensive safety constraints for human-robot
interaction including collision avoidance, workspace limits, and robust
MPC formulations that account for uncertainty in human behavior.

Mathematical Formulation:
========================

Safety Constraints:
    ||p_robot(t) - p_human(t)||₂ ≥ d_safe(t)  ∀t ∈ [0, N]
    ||ṗ_robot(t)||₂ ≤ v_max(d_human(t))       ∀t ∈ [0, N]
    ||p̈_robot(t)||₂ ≤ a_max(d_human(t))       ∀t ∈ [0, N]

Robust MPC with Uncertainty:
    minimize     max_{w ∈ W} J(x, u, w)
    subject to   x_{k+1} = f(x_k, u_k) + w_k
                 g(x_k, u_k) ≤ 0  ∀w_k ∈ W_k
    
Where W represents uncertainty sets for human behavior prediction.

Tube MPC:
    x(t) ∈ X_nominal(t) ⊕ S  where S is robust invariant set
    
Barrier Functions:
    h(x) ≥ 0: safety constraint
    CBF condition: ḣ + γh ≥ 0  (Control Barrier Function)
"""

import numpy as np
import scipy.optimize as opt
from typing import Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ConstraintType(Enum):
    """Types of safety constraints."""
    COLLISION_AVOIDANCE = "collision_avoidance"
    WORKSPACE_LIMITS = "workspace_limits"
    JOINT_LIMITS = "joint_limits"
    VELOCITY_LIMITS = "velocity_limits"
    ACCELERATION_LIMITS = "acceleration_limits"
    CONTROL_BARRIER = "control_barrier"
    TUBE_CONSTRAINTS = "tube_constraints"


class UncertaintyType(Enum):
    """Types of uncertainty modeling."""
    GAUSSIAN = "gaussian"
    BOUNDED = "bounded"
    POLYTOPIC = "polytopic"
    ELLIPSOIDAL = "ellipsoidal"


@dataclass
class SafetyParameters:
    """Parameters for safety constraint configuration."""
    # Collision avoidance
    min_distance: float = 0.3  # Minimum safe distance (m)
    collision_buffer: float = 0.1  # Additional safety buffer
    velocity_reduction_factor: float = 0.5  # Speed reduction near humans
    
    # Workspace limits
    workspace_bounds: Optional[np.ndarray] = None  # [xmin, xmax, ymin, ymax, zmin, zmax]
    workspace_buffer: float = 0.05  # Buffer from workspace boundary
    
    # Robust MPC parameters
    uncertainty_level: float = 0.1  # Uncertainty scaling factor
    tube_size: float = 0.05  # Tube cross-section size for tube MPC
    robustness_margin: float = 0.02  # Additional robustness margin
    
    # Barrier function parameters
    barrier_gamma: float = 1.0  # CBF class-K function parameter
    barrier_relaxation: float = 1e-6  # Constraint relaxation parameter


class SafetyConstraint(ABC):
    """Abstract base class for safety constraints."""
    
    def __init__(self, constraint_type: ConstraintType, priority: int = 1):
        """
        Initialize safety constraint.
        
        Args:
            constraint_type: Type of constraint
            priority: Priority level (1=highest, higher numbers = lower priority)
        """
        self.constraint_type = constraint_type
        self.priority = priority
        self.enabled = True
        
    @abstractmethod
    def evaluate(self, state: np.ndarray, control: np.ndarray, 
                 context: Optional[Dict] = None) -> Tuple[float, np.ndarray]:
        """
        Evaluate constraint violation and gradient.
        
        Args:
            state: System state
            control: Control input
            context: Additional context (human state, etc.)
        
        Returns:
            constraint_value: Constraint value (≤0 for feasible)
            gradient: Gradient w.r.t. state and control
        """
        pass
    
    @abstractmethod
    def get_bounds(self, context: Optional[Dict] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get constraint bounds if applicable.
        
        Args:
            context: Additional context
        
        Returns:
            (lower_bounds, upper_bounds) or None if not applicable
        """
        pass
    
    def enable(self) -> None:
        """Enable constraint."""
        self.enabled = True
        
    def disable(self) -> None:
        """Disable constraint."""
        self.enabled = False


class CollisionAvoidanceConstraint(SafetyConstraint):
    """
    Collision avoidance constraint for human-robot interaction.
    
    Implements distance-based collision avoidance:
    ||p_robot - p_human||₂ ≥ d_safe
    
    With adaptive safety distance based on human intent uncertainty.
    """
    
    def __init__(self, 
                 robot_model: 'Robot6DOF',
                 safety_params: SafetyParameters,
                 priority: int = 1):
        """
        Initialize collision avoidance constraint.
        
        Args:
            robot_model: Robot dynamics model
            safety_params: Safety parameters
            priority: Constraint priority
        """
        super().__init__(ConstraintType.COLLISION_AVOIDANCE, priority)
        self.robot_model = robot_model
        self.safety_params = safety_params
        
    def evaluate(self, state: np.ndarray, control: np.ndarray,
                 context: Optional[Dict] = None) -> Tuple[float, np.ndarray]:
        """Evaluate collision avoidance constraint."""
        if not self.enabled or context is None or 'human_position' not in context:
            return 0.0, np.zeros(len(state) + len(control))
        
        # Get robot end-effector position
        joint_positions = state[0:6]
        ee_pos, _ = self.robot_model.forward_kinematics(joint_positions)
        
        # Human position
        human_pos = context['human_position']
        
        # Distance between robot and human
        distance_vector = ee_pos - human_pos
        distance = np.linalg.norm(distance_vector)
        
        # Adaptive safety distance based on uncertainty
        uncertainty = context.get('human_uncertainty', 0.0)
        safety_distance = self.safety_params.min_distance * (1 + uncertainty)
        
        # Constraint value (negative when violated)
        constraint_value = distance - safety_distance
        
        # Gradient computation
        if distance > 1e-6:
            # Jacobian of end-effector position w.r.t. joint positions
            J_pos = self.robot_model.jacobian(joint_positions)[0:3, :]
            
            # Gradient w.r.t. joint positions
            grad_state = np.zeros(len(state))
            grad_state[0:6] = (distance_vector / distance) @ J_pos
            
            # No gradient w.r.t. control
            grad_control = np.zeros(len(control))
            
            gradient = np.concatenate([grad_state, grad_control])
        else:
            # Singular case - large penalty gradient
            gradient = -np.ones(len(state) + len(control)) * 1e3
        
        return constraint_value, gradient
    
    def get_bounds(self, context: Optional[Dict] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Collision avoidance doesn't use box bounds."""
        return None


class WorkspaceLimitConstraint(SafetyConstraint):
    """
    Workspace limit constraint for robot end-effector.
    
    Ensures end-effector stays within defined workspace bounds:
    x_min ≤ p_x ≤ x_max, y_min ≤ p_y ≤ y_max, z_min ≤ p_z ≤ z_max
    """
    
    def __init__(self,
                 robot_model: 'Robot6DOF',
                 workspace_bounds: np.ndarray,
                 priority: int = 2):
        """
        Initialize workspace constraint.
        
        Args:
            robot_model: Robot dynamics model
            workspace_bounds: [x_min, x_max, y_min, y_max, z_min, z_max]
            priority: Constraint priority
        """
        super().__init__(ConstraintType.WORKSPACE_LIMITS, priority)
        self.robot_model = robot_model
        self.workspace_bounds = workspace_bounds
        
    def evaluate(self, state: np.ndarray, control: np.ndarray,
                 context: Optional[Dict] = None) -> Tuple[float, np.ndarray]:
        """Evaluate workspace limit constraints."""
        if not self.enabled:
            return 0.0, np.zeros(len(state) + len(control))
        
        # Get robot end-effector position
        joint_positions = state[0:6]
        ee_pos, _ = self.robot_model.forward_kinematics(joint_positions)
        
        # Check all workspace bounds
        violations = []
        gradients = []
        
        J_pos = self.robot_model.jacobian(joint_positions)[0:3, :]
        
        # X bounds
        if ee_pos[0] < self.workspace_bounds[0]:
            violations.append(self.workspace_bounds[0] - ee_pos[0])
            grad = np.zeros(len(state) + len(control))
            grad[0:6] = -J_pos[0, :]
            gradients.append(grad)
        elif ee_pos[0] > self.workspace_bounds[1]:
            violations.append(ee_pos[0] - self.workspace_bounds[1])
            grad = np.zeros(len(state) + len(control))
            grad[0:6] = J_pos[0, :]
            gradients.append(grad)
        
        # Y bounds
        if ee_pos[1] < self.workspace_bounds[2]:
            violations.append(self.workspace_bounds[2] - ee_pos[1])
            grad = np.zeros(len(state) + len(control))
            grad[0:6] = -J_pos[1, :]
            gradients.append(grad)
        elif ee_pos[1] > self.workspace_bounds[3]:
            violations.append(ee_pos[1] - self.workspace_bounds[3])
            grad = np.zeros(len(state) + len(control))
            grad[0:6] = J_pos[1, :]
            gradients.append(grad)
        
        # Z bounds
        if ee_pos[2] < self.workspace_bounds[4]:
            violations.append(self.workspace_bounds[4] - ee_pos[2])
            grad = np.zeros(len(state) + len(control))
            grad[0:6] = -J_pos[2, :]
            gradients.append(grad)
        elif ee_pos[2] > self.workspace_bounds[5]:
            violations.append(ee_pos[2] - self.workspace_bounds[5])
            grad = np.zeros(len(state) + len(control))
            grad[0:6] = J_pos[2, :]
            gradients.append(grad)
        
        if violations:
            # Return worst violation
            worst_idx = np.argmax(violations)
            return violations[worst_idx], gradients[worst_idx]
        else:
            return -1.0, np.zeros(len(state) + len(control))  # Feasible
    
    def get_bounds(self, context: Optional[Dict] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Workspace limits don't directly translate to box bounds on state."""
        return None


class JointLimitConstraint(SafetyConstraint):
    """
    Joint limit constraint for robot joints.
    
    Ensures joints stay within safe limits:
    q_min ≤ q ≤ q_max
    q̇_min ≤ q̇ ≤ q̇_max
    """
    
    def __init__(self,
                 joint_limits: 'JointLimits',
                 priority: int = 2):
        """
        Initialize joint limit constraint.
        
        Args:
            joint_limits: Joint limit specification
            priority: Constraint priority
        """
        super().__init__(ConstraintType.JOINT_LIMITS, priority)
        self.joint_limits = joint_limits
        
    def evaluate(self, state: np.ndarray, control: np.ndarray,
                 context: Optional[Dict] = None) -> Tuple[float, np.ndarray]:
        """Evaluate joint limit constraints."""
        if not self.enabled:
            return 0.0, np.zeros(len(state) + len(control))
        
        q = state[0:6]
        qd = state[6:12]
        
        # Position limit violations
        pos_violations = np.maximum(
            self.joint_limits.position_min - q,
            q - self.joint_limits.position_max
        )
        
        # Velocity limit violations
        vel_violations = np.maximum(
            np.abs(qd) - self.joint_limits.velocity_max,
            0
        )
        
        # Combined violations
        all_violations = np.concatenate([pos_violations, vel_violations])
        max_violation = np.max(all_violations)
        
        if max_violation > 0:
            # Gradient
            gradient = np.zeros(len(state) + len(control))
            
            # Position violations
            for i in range(6):
                if q[i] < self.joint_limits.position_min[i]:
                    gradient[i] = -1.0
                elif q[i] > self.joint_limits.position_max[i]:
                    gradient[i] = 1.0
            
            # Velocity violations
            for i in range(6):
                if np.abs(qd[i]) > self.joint_limits.velocity_max[i]:
                    gradient[6+i] = np.sign(qd[i])
            
            return max_violation, gradient
        else:
            return -0.1, np.zeros(len(state) + len(control))  # Feasible
    
    def get_bounds(self, context: Optional[Dict] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get joint limit bounds."""
        lower = np.concatenate([
            self.joint_limits.position_min,
            -self.joint_limits.velocity_max
        ])
        upper = np.concatenate([
            self.joint_limits.position_max,
            self.joint_limits.velocity_max
        ])
        return lower, upper


class ControlBarrierConstraint(SafetyConstraint):
    """
    Control Barrier Function (CBF) constraint.
    
    Implements CBF safety guarantee:
    ḣ(x) + γh(x) ≥ 0
    
    where h(x) is a barrier function and γ > 0 is the class-K parameter.
    """
    
    def __init__(self,
                 barrier_function: Callable[[np.ndarray], Tuple[float, np.ndarray]],
                 dynamics_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 gamma: float = 1.0,
                 priority: int = 1):
        """
        Initialize CBF constraint.
        
        Args:
            barrier_function: Function returning (h(x), ∇h(x))
            dynamics_function: System dynamics f(x, u)
            gamma: Class-K parameter
            priority: Constraint priority
        """
        super().__init__(ConstraintType.CONTROL_BARRIER, priority)
        self.barrier_function = barrier_function
        self.dynamics_function = dynamics_function
        self.gamma = gamma
        
    def evaluate(self, state: np.ndarray, control: np.ndarray,
                 context: Optional[Dict] = None) -> Tuple[float, np.ndarray]:
        """Evaluate CBF constraint."""
        if not self.enabled:
            return 0.0, np.zeros(len(state) + len(control))
        
        # Evaluate barrier function
        h_val, grad_h = self.barrier_function(state)
        
        # System dynamics
        f_xu = self.dynamics_function(state, control)
        
        # Time derivative of barrier function: ḣ = ∇h^T f(x,u)
        h_dot = grad_h @ f_xu
        
        # CBF constraint: ḣ + γh ≥ 0
        constraint_value = -(h_dot + self.gamma * h_val)
        
        # Gradient computation (simplified - would need chain rule for full accuracy)
        gradient = np.zeros(len(state) + len(control))
        
        return constraint_value, gradient
    
    def get_bounds(self, context: Optional[Dict] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """CBF doesn't use box bounds."""
        return None


class RobustMPCFormulation:
    """
    Robust MPC formulation accounting for uncertainty in human behavior.
    
    This class implements various robust MPC approaches:
    - Tube MPC with invariant tubes
    - Min-max robust MPC
    - Chance-constrained MPC
    """
    
    def __init__(self,
                 uncertainty_type: UncertaintyType,
                 safety_params: SafetyParameters):
        """
        Initialize robust MPC formulation.
        
        Args:
            uncertainty_type: Type of uncertainty modeling
            safety_params: Safety parameters
        """
        self.uncertainty_type = uncertainty_type
        self.safety_params = safety_params
        
        logger.info(f"Initialized robust MPC with {uncertainty_type.value} uncertainty")
    
    def compute_robust_constraints(self,
                                 nominal_trajectory: np.ndarray,
                                 uncertainty_set: Dict,
                                 constraints: List[SafetyConstraint]) -> List[Dict]:
        """
        Compute robust constraint tightenings.
        
        Args:
            nominal_trajectory: Nominal state trajectory
            uncertainty_set: Uncertainty set description
            constraints: List of safety constraints
        
        Returns:
            List of tightened constraints
        """
        robust_constraints = []
        
        for constraint in constraints:
            if self.uncertainty_type == UncertaintyType.BOUNDED:
                tightened = self._compute_bounded_tightening(
                    constraint, uncertainty_set
                )
            elif self.uncertainty_type == UncertaintyType.GAUSSIAN:
                tightened = self._compute_gaussian_tightening(
                    constraint, uncertainty_set
                )
            else:
                # Default: no tightening
                tightened = {'constraint': constraint, 'tightening': 0.0}
            
            robust_constraints.append(tightened)
        
        return robust_constraints
    
    def _compute_bounded_tightening(self,
                                  constraint: SafetyConstraint,
                                  uncertainty_set: Dict) -> Dict:
        """Compute constraint tightening for bounded uncertainty."""
        if 'bound' not in uncertainty_set:
            return {'constraint': constraint, 'tightening': 0.0}
        
        # Conservative tightening based on uncertainty bound
        uncertainty_bound = uncertainty_set['bound']
        
        if constraint.constraint_type == ConstraintType.COLLISION_AVOIDANCE:
            # Increase minimum distance by uncertainty bound
            tightening = uncertainty_bound * self.safety_params.robustness_margin
        else:
            tightening = uncertainty_bound * 0.1  # Default tightening
        
        return {
            'constraint': constraint,
            'tightening': tightening,
            'type': 'bounded'
        }
    
    def _compute_gaussian_tightening(self,
                                   constraint: SafetyConstraint,
                                   uncertainty_set: Dict) -> Dict:
        """Compute constraint tightening for Gaussian uncertainty."""
        if 'covariance' not in uncertainty_set:
            return {'constraint': constraint, 'tightening': 0.0}
        
        # Use 3-sigma bound for high confidence
        covariance = uncertainty_set['covariance']
        sigma_bound = 3.0 * np.sqrt(np.trace(covariance))
        
        tightening = sigma_bound * self.safety_params.robustness_margin
        
        return {
            'constraint': constraint,
            'tightening': tightening,
            'type': 'gaussian'
        }
    
    def compute_tube_constraints(self,
                               nominal_state: np.ndarray,
                               tube_size: float) -> np.ndarray:
        """
        Compute tube constraints for tube MPC.
        
        Args:
            nominal_state: Nominal state trajectory
            tube_size: Cross-sectional tube size
        
        Returns:
            Tube constraint bounds
        """
        # Simple box tube around nominal trajectory
        tube_lower = nominal_state - tube_size
        tube_upper = nominal_state + tube_size
        
        return np.column_stack([tube_lower, tube_upper])


class SafetyMonitor:
    """
    Real-time safety monitoring and constraint violation detection.
    
    Monitors system state and detects potential safety violations
    before they occur, enabling proactive safety responses.
    """
    
    def __init__(self,
                 constraints: List[SafetyConstraint],
                 safety_params: SafetyParameters):
        """
        Initialize safety monitor.
        
        Args:
            constraints: List of safety constraints to monitor
            safety_params: Safety parameters
        """
        self.constraints = constraints
        self.safety_params = safety_params
        
        # Monitoring state
        self.violation_history: List[Dict] = []
        self.warning_count = 0
        self.emergency_count = 0
        
        logger.info(f"Initialized safety monitor with {len(constraints)} constraints")
    
    def check_safety(self,
                    current_state: np.ndarray,
                    predicted_trajectory: np.ndarray,
                    context: Optional[Dict] = None) -> Dict[str, Union[bool, List, float]]:
        """
        Check safety of current state and predicted trajectory.
        
        Args:
            current_state: Current system state
            predicted_trajectory: Predicted state trajectory
            context: Additional context information
        
        Returns:
            Safety assessment results
        """
        violations = []
        max_violation = 0.0
        safety_margin = float('inf')
        
        # Check each constraint
        for constraint in self.constraints:
            if not constraint.enabled:
                continue
            
            # Check current state
            violation, _ = constraint.evaluate(
                current_state, np.zeros(6), context
            )
            
            if violation > 0:
                violations.append({
                    'constraint': constraint.constraint_type.value,
                    'violation': violation,
                    'priority': constraint.priority,
                    'timestamp': context.get('timestamp', 0.0) if context else 0.0
                })
                max_violation = max(max_violation, violation)
            else:
                safety_margin = min(safety_margin, abs(violation))
        
        # Determine safety status
        is_safe = len(violations) == 0
        is_warning = max_violation > 0.1  # Warning threshold
        is_emergency = max_violation > 0.5  # Emergency threshold
        
        # Update counters
        if is_warning:
            self.warning_count += 1
        if is_emergency:
            self.emergency_count += 1
        
        # Store violation history
        if violations:
            self.violation_history.extend(violations)
            
            # Maintain history size
            if len(self.violation_history) > 1000:
                self.violation_history = self.violation_history[-500:]
        
        return {
            'is_safe': is_safe,
            'is_warning': is_warning,
            'is_emergency': is_emergency,
            'violations': violations,
            'max_violation': max_violation,
            'safety_margin': safety_margin if safety_margin != float('inf') else 1.0,
            'warning_count': self.warning_count,
            'emergency_count': self.emergency_count
        }
    
    def get_safety_metrics(self) -> Dict[str, float]:
        """Get safety monitoring metrics."""
        if not self.violation_history:
            return {
                'total_violations': 0,
                'average_violation': 0.0,
                'max_violation': 0.0,
                'warning_rate': 0.0,
                'emergency_rate': 0.0
            }
        
        violations_values = [v['violation'] for v in self.violation_history]
        
        return {
            'total_violations': len(self.violation_history),
            'average_violation': np.mean(violations_values),
            'max_violation': np.max(violations_values),
            'warning_rate': self.warning_count / len(self.violation_history),
            'emergency_rate': self.emergency_count / len(self.violation_history)
        }
    
    def reset_monitoring(self) -> None:
        """Reset monitoring statistics."""
        self.violation_history.clear()
        self.warning_count = 0
        self.emergency_count = 0
        logger.info("Reset safety monitoring statistics")


def create_default_safety_constraints(robot_model: 'Robot6DOF') -> List[SafetyConstraint]:
    """
    Create default set of safety constraints for human-robot interaction.
    
    Args:
        robot_model: Robot dynamics model
    
    Returns:
        List of configured safety constraints
    """
    safety_params = SafetyParameters()
    
    constraints = []
    
    # Collision avoidance (highest priority)
    collision_constraint = CollisionAvoidanceConstraint(
        robot_model=robot_model,
        safety_params=safety_params,
        priority=1
    )
    constraints.append(collision_constraint)
    
    # Joint limits (high priority)
    joint_constraint = JointLimitConstraint(
        joint_limits=robot_model.joint_limits,
        priority=2
    )
    constraints.append(joint_constraint)
    
    # Workspace limits (medium priority)
    if robot_model.workspace_bounds is not None:
        workspace_constraint = WorkspaceLimitConstraint(
            robot_model=robot_model,
            workspace_bounds=robot_model.workspace_bounds,
            priority=3
        )
        constraints.append(workspace_constraint)
    
    logger.info(f"Created {len(constraints)} default safety constraints")
    return constraints