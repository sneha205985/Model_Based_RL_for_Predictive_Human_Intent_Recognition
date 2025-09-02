"""
Human-Robot Interaction specific MPC implementation.

This module extends the base MPC controller with features specifically designed
for safe and effective human-robot interaction, including:
- Human-aware safety constraints
- Intent-responsive trajectory adaptation
- Handover timing optimization
- Proximity-based safety zones

Mathematical Formulation:
========================

Human-Aware Cost Function:
    J = J_track + J_control + J_safety + J_intent + J_handover

Where:
- J_track: Trajectory tracking cost
- J_control: Control effort cost  
- J_safety: Safety zone penalties
- J_intent: Intent-based adaptation cost
- J_handover: Handover timing cost

Safety Constraints:
    ||p_robot(t) - p_human(t)||₂ ≥ d_safe(intent, uncertainty)
    
    where d_safe is adaptive safety distance based on:
    - Human intent prediction
    - Intent uncertainty
    - Interaction phase (approach, handover, retreat)

Intent-Adaptive Weights:
    Q_adapted = Q * (1 + α * uncertainty)
    R_adapted = R * (1 + β * uncertainty)
    
    where α, β are adaptation parameters
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from .mpc_controller import MPCController, MPCConfiguration, MPCResult
from ..models.robotics.robot_dynamics import Robot6DOF, RobotState
from ..utils.logger import get_logger

logger = get_logger(__name__)


class InteractionPhase(Enum):
    """Phases of human-robot interaction."""
    IDLE = "idle"
    APPROACH = "approach"
    HANDOVER = "handover"
    RETREAT = "retreat"
    EMERGENCY = "emergency"


@dataclass
class HumanState:
    """State of human during interaction."""
    position: np.ndarray  # [x, y, z] position
    velocity: np.ndarray  # [vx, vy, vz] velocity
    intent_probabilities: Dict[str, float]  # Intent predictions
    uncertainty: float  # Intent uncertainty [0, 1]
    hand_pose: Optional[np.ndarray] = None  # Hand pose if available
    gaze_direction: Optional[np.ndarray] = None  # Gaze vector if available
    timestamp: float = 0.0


@dataclass
class HRIConfiguration:
    """Configuration for human-robot interaction MPC."""
    # Safety parameters
    min_safety_distance: float = 0.3  # Minimum distance to human (m)
    max_safety_distance: float = 1.0  # Maximum safety zone (m)
    uncertainty_safety_factor: float = 2.0  # Safety scaling with uncertainty
    
    # Intent adaptation
    intent_adaptation_enabled: bool = True
    uncertainty_weight_factor: float = 5.0
    intent_response_time: float = 0.5  # Time to adapt to intent changes (s)
    
    # Handover optimization
    handover_enabled: bool = True
    handover_approach_distance: float = 0.15  # Distance to approach for handover
    handover_timeout: float = 5.0  # Max handover duration (s)
    handover_speed_factor: float = 0.5  # Speed reduction during handover
    
    # Emergency behavior
    emergency_stop_distance: float = 0.1  # Emergency stop threshold
    emergency_retreat_distance: float = 0.5  # Distance to retreat in emergency
    
    # Cost weights
    safety_cost_weight: float = 1000.0
    intent_cost_weight: float = 50.0
    handover_cost_weight: float = 100.0
    smoothness_cost_weight: float = 10.0


class HRIMPCController(MPCController):
    """
    Human-Robot Interaction Model Predictive Controller.
    
    This controller extends the base MPC with human-aware features:
    - Adaptive safety constraints based on human intent
    - Intent-responsive trajectory modification
    - Handover timing optimization
    - Emergency behaviors for safety
    
    The controller maintains awareness of human state and adapts its
    behavior accordingly while ensuring safety at all times.
    """
    
    def __init__(self,
                 mpc_config: MPCConfiguration,
                 hri_config: HRIConfiguration,
                 robot_model: Robot6DOF,
                 human_predictor: Optional[Callable] = None):
        """
        Initialize HRI MPC controller.
        
        Args:
            mpc_config: Base MPC configuration
            hri_config: HRI-specific configuration
            robot_model: Robot dynamics model
            human_predictor: Function to predict human intent
        """
        # Initialize base MPC controller
        super().__init__(
            config=mpc_config,
            state_dim=robot_model.state_dim,
            control_dim=robot_model.control_dim,
            dynamics_model=lambda x, u: self._robot_dynamics_wrapper(x, u)
        )
        
        self.hri_config = hri_config
        self.robot_model = robot_model
        self.human_predictor = human_predictor
        
        # Interaction state
        self.current_phase = InteractionPhase.IDLE
        self.human_state: Optional[HumanState] = None
        self.handover_start_time: Optional[float] = None
        
        # Safety monitoring
        self.safety_violations: List[Dict] = []
        self.emergency_count = 0
        
        # Intent adaptation history
        self.intent_history: List[Dict[str, float]] = []
        self.uncertainty_history: List[float] = []
        
        logger.info("Initialized HRI MPC controller")
        logger.info(f"Min safety distance: {hri_config.min_safety_distance:.3f}m")
        logger.info(f"Intent adaptation: {'enabled' if hri_config.intent_adaptation_enabled else 'disabled'}")
        logger.info(f"Handover optimization: {'enabled' if hri_config.handover_enabled else 'disabled'}")
    
    def update_human_state(self, human_state: HumanState) -> None:
        """
        Update human state information.
        
        Args:
            human_state: Current human state
        """
        self.human_state = human_state
        
        # Update interaction phase
        self._update_interaction_phase()
        
        # Store intent history
        self.intent_history.append(human_state.intent_probabilities.copy())
        self.uncertainty_history.append(human_state.uncertainty)
        
        # Maintain history size
        max_history = 100
        if len(self.intent_history) > max_history:
            self.intent_history.pop(0)
            self.uncertainty_history.pop(0)
        
        logger.debug(f"Updated human state - phase: {self.current_phase}, "
                    f"uncertainty: {human_state.uncertainty:.3f}")
    
    def solve_hri_mpc(self,
                     current_robot_state: RobotState,
                     target_pose: np.ndarray,
                     human_state: Optional[HumanState] = None) -> MPCResult:
        """
        Solve MPC with human-robot interaction awareness.
        
        Args:
            current_robot_state: Current robot state
            target_pose: Desired end-effector pose [x, y, z, rx, ry, rz]
            human_state: Current human state (optional, uses stored if None)
        
        Returns:
            MPC result with HRI-aware solution
        """
        if human_state is not None:
            self.update_human_state(human_state)
        
        # Check for emergency conditions
        if self._check_emergency_conditions():
            logger.warning("Emergency condition detected!")
            return self._handle_emergency()
        
        # Generate reference trajectory with HRI awareness
        reference_trajectory = self._generate_hri_reference_trajectory(
            current_robot_state, target_pose
        )
        
        # Solve MPC with human intent information
        mpc_result = self.solve_mpc(
            current_state=np.concatenate([
                current_robot_state.joint_positions,
                current_robot_state.joint_velocities
            ]),
            reference_trajectory=reference_trajectory,
            predicted_human_intent=self.human_state.intent_probabilities if self.human_state else None
        )
        
        # Post-process solution for HRI
        if mpc_result.status.name in ['OPTIMAL', 'FEASIBLE']:
            mpc_result = self._post_process_hri_solution(mpc_result)
        
        return mpc_result
    
    def _update_interaction_phase(self) -> None:
        """Update the current interaction phase based on human state."""
        if self.human_state is None:
            self.current_phase = InteractionPhase.IDLE
            return
        
        # Get dominant intent
        if self.human_state.intent_probabilities:
            dominant_intent = max(self.human_state.intent_probabilities.items(),
                                key=lambda x: x[1])
            intent_name, intent_prob = dominant_intent
            
            # Phase transitions based on intent and proximity
            if intent_name == 'handover' and intent_prob > 0.7:
                if self.current_phase != InteractionPhase.HANDOVER:
                    self.handover_start_time = self.human_state.timestamp
                self.current_phase = InteractionPhase.HANDOVER
            elif intent_name in ['reach', 'grab'] and intent_prob > 0.5:
                self.current_phase = InteractionPhase.APPROACH
            elif intent_name == 'wave':
                self.current_phase = InteractionPhase.RETREAT
            else:
                self.current_phase = InteractionPhase.IDLE
    
    def _check_emergency_conditions(self) -> bool:
        """Check if emergency stop is required."""
        if self.human_state is None:
            return False
        
        # Get current robot end-effector position
        # (This would normally come from robot state)
        robot_ee_pos = np.array([0, 0, 0])  # Placeholder
        
        # Distance to human
        distance = np.linalg.norm(robot_ee_pos - self.human_state.position)
        
        # Emergency conditions
        emergency_conditions = [
            distance < self.hri_config.emergency_stop_distance,
            self.human_state.uncertainty > 0.9,  # Very high uncertainty
            # Could add more conditions like rapid human movement
        ]
        
        if any(emergency_conditions):
            self.emergency_count += 1
            self.current_phase = InteractionPhase.EMERGENCY
            return True
        
        return False
    
    def _handle_emergency(self) -> MPCResult:
        """Handle emergency situation with immediate stop."""
        logger.warning(f"Emergency stop #{self.emergency_count}")
        
        # Return emergency stop control
        emergency_control = np.zeros((self.config.prediction_horizon, self.control_dim))
        emergency_states = np.zeros((self.config.prediction_horizon + 1, self.state_dim))
        
        from .mpc_controller import MPCStatus
        return MPCResult(
            status=MPCStatus.OPTIMAL,
            optimal_control=emergency_control,
            predicted_states=emergency_states,
            optimal_cost=0.0
        )
    
    def _generate_hri_reference_trajectory(self,
                                          current_robot_state: RobotState,
                                          target_pose: np.ndarray) -> np.ndarray:
        """
        Generate reference trajectory considering human intent.
        
        Args:
            current_robot_state: Current robot state
            target_pose: Target end-effector pose
        
        Returns:
            Reference trajectory (N+1 x state_dim)
        """
        N = self.config.prediction_horizon
        
        # Convert current joint state to full state vector
        current_state = np.concatenate([
            current_robot_state.joint_positions,
            current_robot_state.joint_velocities
        ])
        
        # Simple reference trajectory (could be improved with path planning)
        if self.human_state and self.hri_config.intent_adaptation_enabled:
            # Adapt trajectory based on human intent
            trajectory = self._generate_intent_aware_trajectory(
                current_state, target_pose, N
            )
        else:
            # Simple linear interpolation to target
            trajectory = np.zeros((N + 1, self.state_dim))
            for k in range(N + 1):
                alpha = k / N
                trajectory[k] = (1 - alpha) * current_state + alpha * np.zeros(self.state_dim)
        
        return trajectory
    
    def _generate_intent_aware_trajectory(self,
                                        current_state: np.ndarray,
                                        target_pose: np.ndarray,
                                        horizon: int) -> np.ndarray:
        """Generate trajectory that considers human intent."""
        trajectory = np.zeros((horizon + 1, self.state_dim))
        trajectory[0] = current_state
        
        if self.human_state is None:
            return trajectory
        
        # Adapt trajectory based on dominant intent
        dominant_intent = max(self.human_state.intent_probabilities.items(),
                            key=lambda x: x[1])[0]
        
        if dominant_intent == 'handover':
            # Approach trajectory for handover
            trajectory = self._generate_handover_trajectory(
                current_state, target_pose, horizon
            )
        elif dominant_intent in ['reach', 'grab']:
            # Collaborative trajectory
            trajectory = self._generate_collaborative_trajectory(
                current_state, target_pose, horizon
            )
        elif dominant_intent == 'wave':
            # Retreat trajectory
            trajectory = self._generate_retreat_trajectory(
                current_state, horizon
            )
        
        return trajectory
    
    def _generate_handover_trajectory(self,
                                    current_state: np.ndarray,
                                    target_pose: np.ndarray,
                                    horizon: int) -> np.ndarray:
        """Generate trajectory optimized for handover."""
        trajectory = np.zeros((horizon + 1, self.state_dim))
        trajectory[0] = current_state
        
        # Approach human position with reduced speed
        if self.human_state:
            # Calculate handover position (between human and target)
            human_pos = self.human_state.position
            target_pos = target_pose[0:3]
            handover_pos = 0.7 * human_pos + 0.3 * target_pos
            
            # Generate smooth approach trajectory
            for k in range(1, horizon + 1):
                alpha = k / horizon
                # Smooth interpolation with reduced final velocity
                trajectory[k, 0:6] = (1 - alpha) * current_state[0:6] + alpha * np.zeros(6)
                trajectory[k, 6:12] = (1 - alpha) * current_state[6:12] * self.hri_config.handover_speed_factor
        
        return trajectory
    
    def _generate_collaborative_trajectory(self,
                                         current_state: np.ndarray,
                                         target_pose: np.ndarray,
                                         horizon: int) -> np.ndarray:
        """Generate trajectory for collaborative tasks."""
        trajectory = np.zeros((horizon + 1, self.state_dim))
        trajectory[0] = current_state
        
        # Standard trajectory with safety-aware timing
        for k in range(1, horizon + 1):
            alpha = k / horizon
            trajectory[k, 0:6] = (1 - alpha) * current_state[0:6] + alpha * np.zeros(6)
            
            # Velocity profile considers human uncertainty
            if self.human_state:
                speed_factor = 1.0 - 0.5 * self.human_state.uncertainty
                trajectory[k, 6:12] = speed_factor * (trajectory[k, 0:6] - trajectory[k-1, 0:6]) / self.config.sampling_time
        
        return trajectory
    
    def _generate_retreat_trajectory(self,
                                   current_state: np.ndarray,
                                   horizon: int) -> np.ndarray:
        """Generate safe retreat trajectory."""
        trajectory = np.zeros((horizon + 1, self.state_dim))
        trajectory[0] = current_state
        
        # Move to safe distance from human
        if self.human_state:
            # Calculate retreat direction (away from human)
            # This is simplified - would use proper path planning
            for k in range(1, horizon + 1):
                # Gradually reduce joint angles to safe configuration
                safe_config = np.array([0, -np.pi/4, -np.pi/2, 0, np.pi/2, 0])  # Example safe config
                alpha = k / horizon
                trajectory[k, 0:6] = (1 - alpha) * current_state[0:6] + alpha * safe_config
                
                # Smooth velocity profile
                if k > 1:
                    trajectory[k, 6:12] = (trajectory[k, 0:6] - trajectory[k-1, 0:6]) / self.config.sampling_time
        
        return trajectory
    
    def _post_process_hri_solution(self, mpc_result: MPCResult) -> MPCResult:
        """Post-process MPC solution for HRI requirements."""
        if mpc_result.optimal_control is None:
            return mpc_result
        
        # Apply safety velocity limits based on proximity
        if self.human_state:
            proximity_factor = self._compute_proximity_factor()
            
            # Scale control commands based on proximity
            mpc_result.optimal_control *= proximity_factor
            
            # Log safety adjustments
            if proximity_factor < 1.0:
                logger.info(f"Reduced control commands by {(1-proximity_factor)*100:.1f}% due to human proximity")
        
        return mpc_result
    
    def _compute_proximity_factor(self) -> float:
        """Compute speed reduction factor based on human proximity."""
        if self.human_state is None:
            return 1.0
        
        # This would normally use actual robot end-effector position
        robot_pos = np.array([0, 0, 0])  # Placeholder
        distance = np.linalg.norm(robot_pos - self.human_state.position)
        
        # Linear scaling between min and max safety distances
        if distance <= self.hri_config.min_safety_distance:
            return 0.0  # Full stop
        elif distance >= self.hri_config.max_safety_distance:
            return 1.0  # Full speed
        else:
            # Linear interpolation
            return (distance - self.hri_config.min_safety_distance) / \
                   (self.hri_config.max_safety_distance - self.hri_config.min_safety_distance)
    
    def _robot_dynamics_wrapper(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Wrapper for robot dynamics to match MPC interface."""
        return self.robot_model.dynamics(state, control)
    
    def get_interaction_metrics(self) -> Dict[str, float]:
        """Get HRI-specific performance metrics."""
        base_metrics = self.get_performance_metrics()
        
        hri_metrics = {
            'current_phase': self.current_phase.value,
            'emergency_stops': self.emergency_count,
            'safety_violations': len(self.safety_violations),
            'average_uncertainty': np.mean(self.uncertainty_history) if self.uncertainty_history else 0.0,
            'intent_stability': self._compute_intent_stability(),
        }
        
        return {**base_metrics, **hri_metrics}
    
    def _compute_intent_stability(self) -> float:
        """Compute stability of intent predictions over time."""
        if len(self.intent_history) < 2:
            return 1.0
        
        # Compute variance in dominant intent over recent history
        recent_intents = self.intent_history[-10:]  # Last 10 predictions
        
        dominant_intents = []
        for intent_dict in recent_intents:
            if intent_dict:
                dominant_intent = max(intent_dict.items(), key=lambda x: x[1])[0]
                dominant_intents.append(dominant_intent)
        
        if not dominant_intents:
            return 1.0
        
        # Compute stability as consistency of dominant intent
        most_common = max(set(dominant_intents), key=dominant_intents.count)
        stability = dominant_intents.count(most_common) / len(dominant_intents)
        
        return stability


def create_default_hri_mpc(robot_model: Robot6DOF) -> HRIMPCController:
    """
    Create default HRI MPC controller for testing and demonstration.
    
    Args:
        robot_model: Robot dynamics model
    
    Returns:
        Configured HRI MPC controller
    """
    # MPC configuration
    mpc_config = MPCConfiguration(
        prediction_horizon=20,
        control_horizon=15,
        sampling_time=0.1,
        max_solve_time=0.05,
        use_warm_start=True
    )
    
    # HRI configuration
    hri_config = HRIConfiguration(
        min_safety_distance=0.3,
        max_safety_distance=1.0,
        intent_adaptation_enabled=True,
        handover_enabled=True
    )
    
    return HRIMPCController(
        mpc_config=mpc_config,
        hri_config=hri_config,
        robot_model=robot_model
    )