"""
Human-Robot Interaction Environment for Bayesian RL

This module implements the environment model for human-robot interaction scenarios,
providing state/action/reward spaces with uncertainty modeling for Bayesian RL.

Mathematical Foundation:
- State Space: S = S_robot × S_human × S_context
- Action Space: A = continuous/discrete robot control commands  
- Reward Function: R(s,a,s') = w_task*R_task + w_safety*R_safety + w_efficiency*R_efficiency
- Transition Dynamics: P(s'|s,a) with epistemic and aleatoric uncertainty

Author: Bayesian RL Implementation
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractionPhase(Enum):
    """Interaction phases for HRI scenarios"""
    IDLE = auto()
    APPROACH = auto()
    HANDOVER = auto()
    COLLABORATION = auto()
    RETREAT = auto()
    EMERGENCY = auto()


class ActionType(Enum):
    """Types of robot actions"""
    CONTINUOUS = auto()  # Joint velocities, end-effector velocities
    DISCRETE = auto()    # High-level commands (move_to, grasp, etc.)
    HYBRID = auto()      # Mixed continuous and discrete


@dataclass
class RobotState:
    """Robot state representation"""
    # Joint configuration (6-DOF)
    joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(6))
    joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros(6))
    
    # End-effector pose (position + orientation)
    ee_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ee_orientation: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))  # quaternion
    ee_velocity: np.ndarray = field(default_factory=lambda: np.zeros(6))  # linear + angular
    
    # Additional robot state
    gripper_state: float = 0.0  # 0 = open, 1 = closed
    payload_mass: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to flat vector representation"""
        return np.concatenate([
            self.joint_positions,
            self.joint_velocities,
            self.ee_position,
            self.ee_orientation,
            self.ee_velocity,
            [self.gripper_state, self.payload_mass]
        ])
    
    @classmethod
    def from_vector(cls, vector: np.ndarray) -> 'RobotState':
        """Create from flat vector representation"""
        return cls(
            joint_positions=vector[:6],
            joint_velocities=vector[6:12],
            ee_position=vector[12:15],
            ee_orientation=vector[15:19],
            ee_velocity=vector[19:25],
            gripper_state=vector[25],
            payload_mass=vector[26]
        )
    
    @property
    def dimension(self) -> int:
        """State vector dimension"""
        return 27


@dataclass
class HumanState:
    """Human state representation"""
    # Physical state
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pose_keypoints: np.ndarray = field(default_factory=lambda: np.zeros(51))  # 17 keypoints × 3
    
    # Intent and behavior
    intent_probabilities: Dict[str, float] = field(default_factory=dict)
    predicted_trajectory: Optional[np.ndarray] = None  # Future positions
    attention_focus: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Interaction state
    engagement_level: float = 0.5  # 0 = not engaged, 1 = fully engaged
    comfort_level: float = 0.5     # 0 = uncomfortable, 1 = comfortable
    trust_level: float = 0.5       # 0 = no trust, 1 = full trust
    
    # Uncertainty measures
    position_uncertainty: float = 0.1
    intent_uncertainty: float = 0.5
    behavior_uncertainty: float = 0.3
    
    def to_vector(self) -> np.ndarray:
        """Convert to flat vector representation"""
        intent_vec = np.array(list(self.intent_probabilities.values()))
        if len(intent_vec) == 0:
            intent_vec = np.zeros(5)  # Default intent categories
            
        trajectory_vec = self.predicted_trajectory.flatten() if self.predicted_trajectory is not None else np.zeros(15)  # 5 future steps × 3
        
        return np.concatenate([
            self.position,
            self.velocity, 
            self.pose_keypoints,
            intent_vec,
            trajectory_vec,
            self.attention_focus,
            [self.engagement_level, self.comfort_level, self.trust_level],
            [self.position_uncertainty, self.intent_uncertainty, self.behavior_uncertainty]
        ])
    
    @property
    def dimension(self) -> int:
        """State vector dimension"""
        return 3 + 3 + 51 + 5 + 15 + 3 + 3 + 3  # 86


@dataclass
class ContextState:
    """Environmental and task context"""
    # Task information
    task_type: str = "handover"
    task_progress: float = 0.0
    task_priority: float = 0.5
    
    # Environmental factors
    workspace_obstacles: List[np.ndarray] = field(default_factory=list)
    lighting_conditions: float = 1.0  # 0 = dark, 1 = bright
    noise_level: float = 0.0          # 0 = quiet, 1 = loud
    
    # Interaction context
    interaction_phase: InteractionPhase = InteractionPhase.IDLE
    time_in_phase: float = 0.0
    phase_success_probability: float = 0.5
    
    # Safety context
    safety_violations: int = 0
    emergency_stop_active: bool = False
    min_human_distance: float = float('inf')
    
    def to_vector(self) -> np.ndarray:
        """Convert to flat vector representation"""
        # Encode task type as one-hot
        task_types = ["handover", "collaboration", "navigation", "manipulation", "inspection"]
        task_one_hot = np.zeros(len(task_types))
        try:
            task_one_hot[task_types.index(self.task_type)] = 1.0
        except ValueError:
            pass  # Unknown task type
        
        # Encode interaction phase
        phase_one_hot = np.zeros(len(InteractionPhase))
        phase_one_hot[self.interaction_phase.value - 1] = 1.0
        
        # Flatten obstacles (up to 5 obstacles, 6D each: position + size)
        obstacles_vec = np.zeros(30)  # 5 × 6
        for i, obs in enumerate(self.workspace_obstacles[:5]):
            if len(obs) >= 6:
                obstacles_vec[i*6:(i+1)*6] = obs[:6]
        
        return np.concatenate([
            task_one_hot,
            [self.task_progress, self.task_priority],
            obstacles_vec,
            [self.lighting_conditions, self.noise_level],
            phase_one_hot,
            [self.time_in_phase, self.phase_success_probability],
            [self.safety_violations, float(self.emergency_stop_active), self.min_human_distance]
        ])
    
    @property
    def dimension(self) -> int:
        """State vector dimension"""
        return 5 + 2 + 30 + 2 + 6 + 2 + 3  # 50


@dataclass
class HRIState:
    """Complete HRI state combining robot, human, and context"""
    robot: RobotState = field(default_factory=RobotState)
    human: HumanState = field(default_factory=HumanState)
    context: ContextState = field(default_factory=ContextState)
    timestamp: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to flat vector representation"""
        return np.concatenate([
            self.robot.to_vector(),
            self.human.to_vector(),
            self.context.to_vector(),
            [self.timestamp]
        ])
    
    @classmethod
    def from_vector(cls, vector: np.ndarray) -> 'HRIState':
        """Create from flat vector representation"""
        robot_dim = RobotState().dimension
        human_dim = HumanState().dimension
        context_dim = ContextState().dimension
        
        return cls(
            robot=RobotState.from_vector(vector[:robot_dim]),
            human=HumanState.from_vector(vector[robot_dim:robot_dim+human_dim]),
            context=ContextState.from_vector(vector[robot_dim+human_dim:robot_dim+human_dim+context_dim]),
            timestamp=vector[-1]
        )
    
    @property
    def dimension(self) -> int:
        """Total state vector dimension"""
        return self.robot.dimension + self.human.dimension + self.context.dimension + 1  # 164


@dataclass
class RobotAction:
    """Robot action representation"""
    # Control commands
    joint_commands: np.ndarray = field(default_factory=lambda: np.zeros(6))  # Joint velocities/torques
    gripper_command: float = 0.0  # Gripper control
    
    # High-level commands (for discrete actions)
    move_command: Optional[str] = None  # "move_to", "grasp", "release", etc.
    target_pose: Optional[np.ndarray] = None  # Target end-effector pose
    
    # Motion parameters
    velocity_scale: float = 1.0    # Speed scaling factor
    force_limit: float = 50.0      # Force/torque limits
    
    # Safety parameters
    collision_checking: bool = True
    emergency_stop: bool = False
    
    def to_vector(self) -> np.ndarray:
        """Convert to flat vector representation"""
        # Encode high-level command
        commands = ["none", "move_to", "grasp", "release", "wait", "stop"]
        command_one_hot = np.zeros(len(commands))
        if self.move_command in commands:
            command_one_hot[commands.index(self.move_command)] = 1.0
        
        target_vec = self.target_pose if self.target_pose is not None else np.zeros(7)  # position + quaternion
        
        return np.concatenate([
            self.joint_commands,
            [self.gripper_command],
            command_one_hot,
            target_vec,
            [self.velocity_scale, self.force_limit],
            [float(self.collision_checking), float(self.emergency_stop)]
        ])
    
    @property
    def dimension(self) -> int:
        """Action vector dimension"""
        return 6 + 1 + 6 + 7 + 2 + 2  # 24


class RewardFunction:
    """Multi-objective reward function for HRI"""
    
    def __init__(self, weights: Dict[str, float] = None):
        """Initialize reward function with component weights"""
        self.weights = weights or {
            'task_success': 1.0,
            'safety': 2.0,          # Higher weight for safety
            'efficiency': 0.5,
            'human_comfort': 1.0,
            'smoothness': 0.3
        }
    
    def compute_reward(self, state: HRIState, action: RobotAction, next_state: HRIState) -> Dict[str, float]:
        """
        Compute multi-component reward
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            Dictionary with reward components and total reward
        """
        rewards = {}
        
        # Task success reward
        rewards['task_success'] = self._task_success_reward(state, action, next_state)
        
        # Safety reward (negative cost for violations)
        rewards['safety'] = self._safety_reward(state, action, next_state)
        
        # Efficiency reward (negative cost for time/energy)
        rewards['efficiency'] = self._efficiency_reward(state, action, next_state)
        
        # Human comfort reward
        rewards['human_comfort'] = self._human_comfort_reward(state, action, next_state)
        
        # Motion smoothness reward
        rewards['smoothness'] = self._smoothness_reward(state, action, next_state)
        
        # Compute total weighted reward
        rewards['total'] = sum(self.weights[key] * rewards[key] for key in self.weights.keys())
        
        return rewards
    
    def _task_success_reward(self, state: HRIState, action: RobotAction, next_state: HRIState) -> float:
        """Reward for task progress and completion"""
        progress_reward = next_state.context.task_progress - state.context.task_progress
        
        # Bonus for task completion
        completion_bonus = 0.0
        if next_state.context.task_progress >= 1.0:
            completion_bonus = 10.0
            
        # Phase-specific rewards
        phase_reward = 0.0
        if next_state.context.interaction_phase != state.context.interaction_phase:
            # Successful phase transitions
            phase_reward = 1.0
        
        return progress_reward + completion_bonus + phase_reward
    
    def _safety_reward(self, state: HRIState, action: RobotAction, next_state: HRIState) -> float:
        """Reward for maintaining safety (negative cost for violations)"""
        safety_reward = 0.0
        
        # Distance to human (closer = more negative reward if too close)
        human_distance = np.linalg.norm(next_state.robot.ee_position - next_state.human.position)
        min_safe_distance = 0.3  # 30cm minimum
        
        if human_distance < min_safe_distance:
            safety_reward -= (min_safe_distance - human_distance) * 10.0
        
        # Velocity limits
        max_velocity = np.max(np.abs(next_state.robot.joint_velocities))
        if max_velocity > 2.0:  # rad/s
            safety_reward -= (max_velocity - 2.0) * 5.0
            
        # Safety violations
        safety_violations = next_state.context.safety_violations - state.context.safety_violations
        safety_reward -= safety_violations * 20.0
        
        # Emergency stop penalty
        if next_state.context.emergency_stop_active:
            safety_reward -= 50.0
            
        return safety_reward
    
    def _efficiency_reward(self, state: HRIState, action: RobotAction, next_state: HRIState) -> float:
        """Reward for efficient motion and energy use"""
        # Energy cost (proportional to control effort)
        energy_cost = np.sum(np.square(action.joint_commands))
        
        # Time cost (encourage faster task completion)
        time_cost = 1.0  # Constant time penalty
        
        # Path efficiency (penalize unnecessary movements)
        movement_magnitude = np.linalg.norm(
            next_state.robot.ee_position - state.robot.ee_position
        )
        
        # Reward useful movement, penalize excessive movement
        movement_reward = min(movement_magnitude * 2.0, 1.0)
        if movement_magnitude > 0.1:  # 10cm
            movement_reward -= (movement_magnitude - 0.1) * 2.0
        
        return movement_reward - 0.01 * energy_cost - 0.1 * time_cost
    
    def _human_comfort_reward(self, state: HRIState, action: RobotAction, next_state: HRIState) -> float:
        """Reward for maintaining human comfort and trust"""
        comfort_reward = 0.0
        
        # Reward increased comfort/trust levels
        comfort_change = next_state.human.comfort_level - state.human.comfort_level
        trust_change = next_state.human.trust_level - state.human.trust_level
        engagement_change = next_state.human.engagement_level - state.human.engagement_level
        
        comfort_reward += comfort_change * 5.0
        comfort_reward += trust_change * 3.0
        comfort_reward += engagement_change * 2.0
        
        # Penalize rapid/jerky movements that might startle human
        velocity_change = np.linalg.norm(
            next_state.robot.joint_velocities - state.robot.joint_velocities
        )
        if velocity_change > 1.0:
            comfort_reward -= velocity_change * 2.0
        
        return comfort_reward
    
    def _smoothness_reward(self, state: HRIState, action: RobotAction, next_state: HRIState) -> float:
        """Reward for smooth, predictable motion"""
        # Acceleration penalty (difference in velocities)
        acceleration = np.linalg.norm(
            next_state.robot.joint_velocities - state.robot.joint_velocities
        )
        
        # Jerk penalty would require storing previous accelerations
        smoothness_reward = -acceleration * 0.5
        
        # Reward consistent velocity directions
        velocity_consistency = np.dot(
            state.robot.joint_velocities / (np.linalg.norm(state.robot.joint_velocities) + 1e-6),
            next_state.robot.joint_velocities / (np.linalg.norm(next_state.robot.joint_velocities) + 1e-6)
        )
        smoothness_reward += max(0, velocity_consistency) * 0.5
        
        return smoothness_reward


class TransitionDynamics:
    """Probabilistic transition dynamics with uncertainty"""
    
    def __init__(self, robot_model=None, human_behavior_model=None):
        """Initialize with models for robot and human behavior"""
        self.robot_model = robot_model
        self.human_behavior_model = human_behavior_model
        
        # Uncertainty parameters
        self.process_noise_std = 0.01  # Process noise standard deviation
        self.observation_noise_std = 0.005  # Observation noise
        
    def predict_transition(self, state: HRIState, action: RobotAction, dt: float = 0.1) -> Tuple[HRIState, Dict[str, float]]:
        """
        Predict next state with uncertainty estimates
        
        Args:
            state: Current state
            action: Action to take
            dt: Time step
            
        Returns:
            Tuple of (predicted_next_state, uncertainty_estimates)
        """
        # Predict robot state evolution
        next_robot_state = self._predict_robot_dynamics(state.robot, action, dt)
        
        # Predict human state evolution
        next_human_state = self._predict_human_dynamics(state.human, state.context, dt)
        
        # Update context
        next_context_state = self._update_context(state.context, state.robot, next_robot_state, next_human_state, dt)
        
        # Create next state
        next_state = HRIState(
            robot=next_robot_state,
            human=next_human_state,
            context=next_context_state,
            timestamp=state.timestamp + dt
        )
        
        # Compute uncertainty estimates
        uncertainty = self._compute_transition_uncertainty(state, action, next_state)
        
        return next_state, uncertainty
    
    def _predict_robot_dynamics(self, robot_state: RobotState, action: RobotAction, dt: float) -> RobotState:
        """Predict robot state evolution using dynamics model"""
        if self.robot_model is not None:
            # Use physics-based robot model if available
            return self.robot_model.integrate_dynamics(robot_state, action, dt)
        else:
            # Simple kinematic integration
            next_positions = robot_state.joint_positions + robot_state.joint_velocities * dt
            next_velocities = action.joint_commands  # Assume velocity control
            
            # Update end-effector state (simplified)
            ee_linear_vel = action.joint_commands[:3] * 0.1  # Simplified mapping
            next_ee_position = robot_state.ee_position + ee_linear_vel * dt
            
            # Add process noise
            next_positions += np.random.normal(0, self.process_noise_std, 6)
            next_velocities += np.random.normal(0, self.process_noise_std, 6)
            
            return RobotState(
                joint_positions=next_positions,
                joint_velocities=next_velocities,
                ee_position=next_ee_position,
                ee_orientation=robot_state.ee_orientation,  # Simplified
                ee_velocity=np.concatenate([ee_linear_vel, np.zeros(3)]),
                gripper_state=action.gripper_command,
                payload_mass=robot_state.payload_mass
            )
    
    def _predict_human_dynamics(self, human_state: HumanState, context: ContextState, dt: float) -> HumanState:
        """Predict human state evolution"""
        if self.human_behavior_model is not None:
            # Use learned human behavior model
            return self.human_behavior_model.predict_next_state(human_state, context, dt)
        else:
            # Simple motion model
            next_position = human_state.position + human_state.velocity * dt
            
            # Add uncertainty to human motion
            position_noise = np.random.normal(0, human_state.position_uncertainty, 3)
            next_position += position_noise
            
            # Update intent (simplified)
            next_intent = human_state.intent_probabilities.copy()
            
            # Slowly evolve comfort/trust/engagement (simplified)
            comfort_change = np.random.normal(0, 0.01)
            next_comfort = np.clip(human_state.comfort_level + comfort_change, 0, 1)
            
            trust_change = np.random.normal(0, 0.005)
            next_trust = np.clip(human_state.trust_level + trust_change, 0, 1)
            
            return HumanState(
                position=next_position,
                velocity=human_state.velocity,
                pose_keypoints=human_state.pose_keypoints,
                intent_probabilities=next_intent,
                predicted_trajectory=human_state.predicted_trajectory,
                attention_focus=human_state.attention_focus,
                engagement_level=human_state.engagement_level,
                comfort_level=next_comfort,
                trust_level=next_trust,
                position_uncertainty=human_state.position_uncertainty,
                intent_uncertainty=human_state.intent_uncertainty,
                behavior_uncertainty=human_state.behavior_uncertainty
            )
    
    def _update_context(self, context: ContextState, robot_state: RobotState, 
                       next_robot_state: RobotState, next_human_state: HumanState, dt: float) -> ContextState:
        """Update environmental and task context"""
        # Update task progress (simplified)
        progress_increment = 0.01 if context.interaction_phase != InteractionPhase.IDLE else 0.0
        next_progress = min(context.task_progress + progress_increment, 1.0)
        
        # Update interaction phase (simplified state machine)
        next_phase = self._update_interaction_phase(context, next_robot_state, next_human_state)
        
        # Update safety metrics
        human_distance = np.linalg.norm(next_robot_state.ee_position - next_human_state.position)
        
        return ContextState(
            task_type=context.task_type,
            task_progress=next_progress,
            task_priority=context.task_priority,
            workspace_obstacles=context.workspace_obstacles,
            lighting_conditions=context.lighting_conditions,
            noise_level=context.noise_level,
            interaction_phase=next_phase,
            time_in_phase=context.time_in_phase + dt if next_phase == context.interaction_phase else 0.0,
            phase_success_probability=context.phase_success_probability,
            safety_violations=context.safety_violations,
            emergency_stop_active=context.emergency_stop_active,
            min_human_distance=min(context.min_human_distance, human_distance)
        )
    
    def _update_interaction_phase(self, context: ContextState, robot_state: RobotState, human_state: HumanState) -> InteractionPhase:
        """Update interaction phase based on states"""
        current_phase = context.interaction_phase
        
        # Simplified phase transition logic
        human_distance = np.linalg.norm(robot_state.ee_position - human_state.position)
        
        if human_distance < 0.2:  # Very close
            if current_phase == InteractionPhase.APPROACH:
                return InteractionPhase.HANDOVER
        elif human_distance < 0.5:  # Moderate distance
            if current_phase == InteractionPhase.IDLE:
                return InteractionPhase.APPROACH
        else:  # Far away
            if current_phase in [InteractionPhase.HANDOVER, InteractionPhase.COLLABORATION]:
                return InteractionPhase.RETREAT
        
        return current_phase
    
    def _compute_transition_uncertainty(self, state: HRIState, action: RobotAction, next_state: HRIState) -> Dict[str, float]:
        """Compute uncertainty estimates for the transition"""
        return {
            'robot_position_std': self.process_noise_std,
            'robot_velocity_std': self.process_noise_std,
            'human_position_std': next_state.human.position_uncertainty,
            'human_intent_std': next_state.human.intent_uncertainty,
            'context_uncertainty': 0.1,
            'total_state_uncertainty': 0.05
        }


class HRIEnvironment:
    """Complete HRI Environment for Bayesian RL"""
    
    def __init__(self, 
                 reward_weights: Dict[str, float] = None,
                 robot_model=None,
                 human_behavior_model=None,
                 action_type: ActionType = ActionType.CONTINUOUS,
                 dt: float = 0.1):
        """
        Initialize HRI Environment
        
        Args:
            reward_weights: Weights for multi-objective reward function
            robot_model: Physics-based robot dynamics model
            human_behavior_model: Human behavior prediction model
            action_type: Type of actions (continuous/discrete/hybrid)
            dt: Time step for dynamics integration
        """
        self.reward_function = RewardFunction(reward_weights)
        self.dynamics = TransitionDynamics(robot_model, human_behavior_model)
        self.action_type = action_type
        self.dt = dt
        
        # Environment state
        self.current_state: Optional[HRIState] = None
        self.episode_length = 0
        self.max_episode_length = 1000
        
        # Action and observation spaces
        self.state_dimension = HRIState().dimension
        self.action_dimension = RobotAction().dimension
        
        # Action bounds (for continuous actions)
        self.action_low = np.full(self.action_dimension, -1.0)
        self.action_high = np.full(self.action_dimension, 1.0)
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_safety_violations = 0
        
    def reset(self, initial_state: Optional[HRIState] = None) -> HRIState:
        """
        Reset environment to initial state
        
        Args:
            initial_state: Optional initial state, otherwise use default
            
        Returns:
            Initial state
        """
        if initial_state is not None:
            self.current_state = initial_state
        else:
            # Create default initial state
            robot_state = RobotState(
                joint_positions=np.array([0, 0, 0, 0, 0, 0]),
                ee_position=np.array([0.5, 0.0, 0.5])
            )
            human_state = HumanState(
                position=np.array([0.8, 0.3, 0.8]),
                intent_probabilities={'handover': 0.7, 'idle': 0.3}
            )
            context_state = ContextState(
                task_type="handover",
                interaction_phase=InteractionPhase.IDLE
            )
            self.current_state = HRIState(robot=robot_state, human=human_state, context=context_state)
        
        self.episode_length = 0
        self.episode_rewards = []
        self.episode_safety_violations = 0
        
        logger.info(f"Environment reset. Initial state dimension: {self.current_state.dimension}")
        return self.current_state
    
    def step(self, action: Union[np.ndarray, RobotAction]) -> Tuple[HRIState, Dict[str, float], bool, Dict[str, Any]]:
        """
        Execute one environment step
        
        Args:
            action: Action to execute (vector or RobotAction object)
            
        Returns:
            Tuple of (next_state, reward_dict, done, info)
        """
        if self.current_state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        
        # Convert action vector to RobotAction if needed
        if isinstance(action, np.ndarray):
            robot_action = self._vector_to_action(action)
        else:
            robot_action = action
        
        # Apply dynamics to get next state
        next_state, transition_uncertainty = self.dynamics.predict_transition(
            self.current_state, robot_action, self.dt
        )
        
        # Compute reward
        reward_dict = self.reward_function.compute_reward(
            self.current_state, robot_action, next_state
        )
        
        # Check if episode is done
        done = self._is_episode_done(next_state)
        
        # Create info dictionary
        info = {
            'transition_uncertainty': transition_uncertainty,
            'safety_violations': next_state.context.safety_violations,
            'task_progress': next_state.context.task_progress,
            'human_distance': np.linalg.norm(next_state.robot.ee_position - next_state.human.position),
            'interaction_phase': next_state.context.interaction_phase.name,
            'episode_length': self.episode_length
        }
        
        # Update environment state
        self.current_state = next_state
        self.episode_length += 1
        self.episode_rewards.append(reward_dict['total'])
        
        if next_state.context.safety_violations > self.episode_safety_violations:
            self.episode_safety_violations = next_state.context.safety_violations
        
        return next_state, reward_dict, done, info
    
    def _vector_to_action(self, action_vector: np.ndarray) -> RobotAction:
        """Convert action vector to RobotAction object"""
        if len(action_vector) != self.action_dimension:
            raise ValueError(f"Action vector dimension {len(action_vector)} != expected {self.action_dimension}")
        
        return RobotAction(
            joint_commands=action_vector[:6],
            gripper_command=action_vector[6],
            # Decode other fields from remaining vector elements
            velocity_scale=np.clip(action_vector[7], 0.1, 2.0),
            force_limit=np.clip(action_vector[8], 10.0, 100.0)
        )
    
    def _is_episode_done(self, state: HRIState) -> bool:
        """Check if episode should terminate"""
        # Task completion
        if state.context.task_progress >= 1.0:
            return True
        
        # Safety violations
        if state.context.emergency_stop_active:
            return True
        
        # Maximum episode length
        if self.episode_length >= self.max_episode_length:
            return True
        
        # Human too far away (task failure)
        human_distance = np.linalg.norm(state.robot.ee_position - state.human.position)
        if human_distance > 2.0:  # 2m
            return True
        
        return False
    
    def get_observation(self) -> np.ndarray:
        """Get current observation (state vector)"""
        if self.current_state is None:
            raise ValueError("Environment not initialized")
        return self.current_state.to_vector()
    
    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get state space bounds"""
        # Define reasonable bounds for state variables
        state_low = np.full(self.state_dimension, -10.0)
        state_high = np.full(self.state_dimension, 10.0)
        
        # More specific bounds for some variables
        # Joint positions: [-2π, 2π]
        state_low[:6] = -2 * np.pi
        state_high[:6] = 2 * np.pi
        
        # Probabilities: [0, 1]
        prob_indices = [15, 16, 17, 18, 19]  # Intent probabilities
        state_low[prob_indices] = 0.0
        state_high[prob_indices] = 1.0
        
        return state_low, state_high
    
    def get_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get action space bounds"""
        return self.action_low.copy(), self.action_high.copy()
    
    def render(self, mode: str = 'text') -> Optional[str]:
        """Render current environment state"""
        if self.current_state is None:
            return "Environment not initialized"
        
        if mode == 'text':
            state_summary = f"""
HRI Environment State:
- Robot End-Effector: {self.current_state.robot.ee_position}
- Human Position: {self.current_state.human.position}
- Task Progress: {self.current_state.context.task_progress:.2f}
- Interaction Phase: {self.current_state.context.interaction_phase.name}
- Episode Length: {self.episode_length}
- Safety Violations: {self.current_state.context.safety_violations}
- Human Distance: {np.linalg.norm(self.current_state.robot.ee_position - self.current_state.human.position):.2f}m
            """
            return state_summary
        
        return None


def create_default_hri_environment(**kwargs) -> HRIEnvironment:
    """Create HRI environment with default configuration"""
    default_reward_weights = {
        'task_success': 1.0,
        'safety': 2.0,
        'efficiency': 0.5,
        'human_comfort': 1.0,
        'smoothness': 0.3
    }
    
    return HRIEnvironment(
        reward_weights=default_reward_weights,
        action_type=ActionType.CONTINUOUS,
        dt=0.1,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = create_default_hri_environment()
    
    # Reset environment
    initial_state = env.reset()
    print(f"Initial state dimension: {initial_state.dimension}")
    print(f"Action dimension: {env.action_dimension}")
    
    # Take a few steps
    for step in range(5):
        # Random action
        action = np.random.uniform(env.action_low, env.action_high)
        
        # Execute step
        next_state, reward_dict, done, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"Reward: {reward_dict['total']:.3f}")
        print(f"Task Progress: {next_state.context.task_progress:.2f}")
        print(f"Human Distance: {info['human_distance']:.2f}m")
        print(f"Done: {done}")
        
        if done:
            break
    
    print("\nEnvironment test completed successfully!")