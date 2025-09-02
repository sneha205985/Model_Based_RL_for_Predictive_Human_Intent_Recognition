"""
Realistic HRI Simulation Environment

This module provides a realistic physics-based simulation environment
for human-robot interaction scenarios with configurable human behaviors,
sensor noise, and dynamic environments.

Features:
- Physics-based robot dynamics
- Realistic human behavior models
- Sensor noise and uncertainty
- Dynamic obstacle environments
- Configurable interaction scenarios

Author: Phase 5 Implementation
Date: 2024
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import threading
import queue

# Import system components
try:
    from src.environments.hri_environment import (
        HRIEnvironment, HRIState, RobotState, HumanState, ContextState,
        InteractionPhase, create_default_hri_environment
    )
    from src.system.human_intent_rl_system import HumanIntentRLSystem, SystemConfiguration, SystemMode
except ImportError as e:
    logging.warning(f"Import error: {e}. Some components may not be available.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HumanBehaviorType(Enum):
    """Types of human behavior patterns"""
    NORMAL = auto()
    AGGRESSIVE = auto()
    CAUTIOUS = auto()
    UNPREDICTABLE = auto()
    COLLABORATIVE = auto()


@dataclass
class SimulationConfiguration:
    """Configuration for realistic HRI simulation"""
    # Physics parameters
    gravity: float = 9.81
    timestep: float = 0.01  # 100 Hz simulation
    
    # Robot parameters
    robot_mass: float = 50.0  # kg
    robot_max_velocity: float = 2.0  # m/s
    robot_max_acceleration: float = 5.0  # m/s²
    
    # Human parameters
    human_mass: float = 70.0  # kg
    human_reaction_time: float = 0.3  # seconds
    human_max_velocity: float = 1.5  # m/s
    
    # Sensor noise parameters
    position_noise_std: float = 0.01  # meters
    velocity_noise_std: float = 0.05  # m/s
    intent_noise_std: float = 0.1     # intent uncertainty
    
    # Environment parameters
    workspace_bounds: List[float] = field(default_factory=lambda: [-2, 2, -2, 2, 0, 2])  # [x_min, x_max, y_min, y_max, z_min, z_max]
    enable_obstacles: bool = True
    dynamic_obstacles: bool = False
    
    # Scenario parameters
    scenario_duration: float = 30.0  # seconds
    human_behavior_type: HumanBehaviorType = HumanBehaviorType.NORMAL
    interaction_complexity: str = "medium"  # "simple", "medium", "complex"


class RealisticHumanBehaviorModel:
    """Realistic human behavior model with various behavior patterns"""
    
    def __init__(self, config: SimulationConfiguration):
        """Initialize human behavior model"""
        self.config = config
        self.behavior_type = config.human_behavior_type
        
        # Internal state
        self.current_intent = "idle"
        self.intent_history = []
        self.position_history = []
        self.velocity_history = []
        
        # Behavior parameters
        self._setup_behavior_parameters()
        
        # Random number generator
        self.rng = np.random.RandomState(42)
        
        logger.info(f"Initialized human behavior model: {self.behavior_type.name}")
    
    def _setup_behavior_parameters(self):
        """Setup behavior-specific parameters"""
        if self.behavior_type == HumanBehaviorType.NORMAL:
            self.intent_change_probability = 0.05
            self.movement_smoothness = 0.8
            self.reaction_speed = 1.0
            self.predictability = 0.8
            
        elif self.behavior_type == HumanBehaviorType.AGGRESSIVE:
            self.intent_change_probability = 0.15
            self.movement_smoothness = 0.4
            self.reaction_speed = 1.5
            self.predictability = 0.6
            
        elif self.behavior_type == HumanBehaviorType.CAUTIOUS:
            self.intent_change_probability = 0.02
            self.movement_smoothness = 0.9
            self.reaction_speed = 0.7
            self.predictability = 0.9
            
        elif self.behavior_type == HumanBehaviorType.UNPREDICTABLE:
            self.intent_change_probability = 0.3
            self.movement_smoothness = 0.3
            self.reaction_speed = 1.2
            self.predictability = 0.3
            
        else:  # COLLABORATIVE
            self.intent_change_probability = 0.1
            self.movement_smoothness = 0.85
            self.reaction_speed = 1.1
            self.predictability = 0.85
    
    def update_behavior(self, robot_state: RobotState, context: ContextState, dt: float) -> HumanState:
        """Update human behavior based on robot state and context"""
        # Get current human state or create default
        current_human = self._get_current_human_state()
        
        # Update intent based on context and robot behavior
        new_intent = self._update_intent(current_human, robot_state, context)
        
        # Update position based on intent and robot position
        new_position, new_velocity = self._update_position(
            current_human, robot_state, new_intent, dt
        )
        
        # Add sensor noise
        noisy_position = self._add_position_noise(new_position)
        noisy_velocity = self._add_velocity_noise(new_velocity)
        
        # Update intent probabilities with uncertainty
        intent_probs = self._compute_intent_probabilities(new_intent, robot_state)
        
        # Create new human state
        human_state = HumanState(
            position=noisy_position,
            velocity=noisy_velocity,
            pose_keypoints=self._generate_pose_keypoints(noisy_position),
            intent_probabilities=intent_probs,
            predicted_trajectory=self._predict_trajectory(noisy_position, noisy_velocity),
            attention_focus=robot_state.ee_position,
            engagement_level=self._compute_engagement_level(robot_state),
            comfort_level=self._compute_comfort_level(robot_state, noisy_position),
            trust_level=self._compute_trust_level(context),
            position_uncertainty=self.config.position_noise_std,
            intent_uncertainty=self._compute_intent_uncertainty(new_intent),
            behavior_uncertainty=1.0 - self.predictability
        )
        
        # Update history
        self._update_history(human_state)
        
        return human_state
    
    def _get_current_human_state(self) -> HumanState:
        """Get current human state or create default"""
        if not self.position_history:
            # Create initial human state
            initial_position = np.array([0.8, 0.3, 0.8])  # 80cm away from robot
            return HumanState(
                position=initial_position,
                velocity=np.zeros(3),
                intent_probabilities={'idle': 1.0}
            )
        else:
            # Use last known state
            return HumanState(
                position=self.position_history[-1],
                velocity=self.velocity_history[-1] if self.velocity_history else np.zeros(3),
                intent_probabilities={self.current_intent: 1.0}
            )
    
    def _update_intent(self, current_human: HumanState, robot_state: RobotState, 
                      context: ContextState) -> str:
        """Update human intent based on context and robot behavior"""
        # Distance to robot
        distance_to_robot = np.linalg.norm(current_human.position - robot_state.ee_position)
        
        # Intent transition probabilities based on distance and context
        if distance_to_robot < 0.4:  # Very close
            if context.interaction_phase == InteractionPhase.APPROACH:
                # Likely to start handover
                if self.rng.random() < 0.7:
                    return "handover_request"
            elif context.interaction_phase == InteractionPhase.HANDOVER:
                return "handover_give"
                
        elif distance_to_robot < 0.8:  # Medium distance
            if self.current_intent == "idle" and self.rng.random() < self.intent_change_probability:
                # Might start approaching
                return "handover_request"
                
        elif distance_to_robot > 1.5:  # Far away
            if self.current_intent != "idle" and self.rng.random() < self.intent_change_probability:
                # Might give up and go idle
                return "idle"
        
        # Random intent changes based on behavior type
        if self.rng.random() < self.intent_change_probability:
            possible_intents = ["idle", "handover_request", "collaboration_start", "obstacle_avoidance"]
            
            # Remove current intent from possibilities
            if self.current_intent in possible_intents:
                possible_intents.remove(self.current_intent)
            
            if possible_intents:
                return self.rng.choice(possible_intents)
        
        return self.current_intent
    
    def _update_position(self, current_human: HumanState, robot_state: RobotState,
                        intent: str, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Update human position based on intent and robot position"""
        current_pos = current_human.position
        current_vel = current_human.velocity
        
        # Target position based on intent
        target_pos = self._get_target_position(intent, robot_state.ee_position)
        
        # Desired velocity toward target
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0.01:  # Avoid division by zero
            direction_normalized = direction / distance
            
            # Speed based on intent and behavior type
            if intent == "handover_request":
                desired_speed = 0.3 * self.reaction_speed  # Approach slowly
            elif intent == "handover_give":
                desired_speed = 0.1 * self.reaction_speed  # Very slow for precision
            elif intent == "obstacle_avoidance":
                desired_speed = 0.5 * self.reaction_speed  # Move away quickly
            elif intent == "collaboration_start":
                desired_speed = 0.2 * self.reaction_speed  # Moderate speed
            else:  # idle
                desired_speed = 0.05 * self.reaction_speed  # Very slow drift
            
            desired_velocity = direction_normalized * min(desired_speed, distance / dt)
        else:
            desired_velocity = np.zeros(3)
        
        # Smooth velocity changes
        alpha = self.movement_smoothness
        new_velocity = alpha * current_vel + (1 - alpha) * desired_velocity
        
        # Limit maximum velocity
        speed = np.linalg.norm(new_velocity)
        if speed > self.config.human_max_velocity:
            new_velocity = new_velocity / speed * self.config.human_max_velocity
        
        # Update position
        new_position = current_pos + new_velocity * dt
        
        # Keep within workspace bounds
        bounds = self.config.workspace_bounds
        new_position[0] = np.clip(new_position[0], bounds[0] + 0.2, bounds[1] - 0.2)
        new_position[1] = np.clip(new_position[1], bounds[2] + 0.2, bounds[3] - 0.2)
        new_position[2] = np.clip(new_position[2], bounds[4] + 0.5, bounds[5])
        
        return new_position, new_velocity
    
    def _get_target_position(self, intent: str, robot_ee_position: np.ndarray) -> np.ndarray:
        """Get target position based on intent"""
        if intent == "handover_request":
            # Move closer to robot for handover
            direction = robot_ee_position - np.array([0.8, 0.3, 0.8])
            direction[2] = 0  # Keep same height
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            target = robot_ee_position - direction * 0.4  # Stop 40cm away
            target[2] = 0.8  # Keep handover height
            return target
            
        elif intent == "handover_give":
            # Get very close for handover
            direction = robot_ee_position - np.array([0.8, 0.3, 0.8])
            direction[2] = 0
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            target = robot_ee_position - direction * 0.25  # Very close
            target[2] = 0.8
            return target
            
        elif intent == "obstacle_avoidance":
            # Move away from robot
            direction = np.array([0.8, 0.3, 0.8]) - robot_ee_position
            direction[2] = 0
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            target = robot_ee_position + direction * 1.2  # Move away
            target[2] = 0.8
            return target
            
        elif intent == "collaboration_start":
            # Move to collaboration position
            return np.array([0.6, 0.2, 0.8])
            
        else:  # idle
            # Stay at comfortable distance
            return np.array([0.9, 0.3, 0.8])
    
    def _add_position_noise(self, position: np.ndarray) -> np.ndarray:
        """Add realistic sensor noise to position"""
        noise = self.rng.normal(0, self.config.position_noise_std, 3)
        return position + noise
    
    def _add_velocity_noise(self, velocity: np.ndarray) -> np.ndarray:
        """Add realistic sensor noise to velocity"""
        noise = self.rng.normal(0, self.config.velocity_noise_std, 3)
        return velocity + noise
    
    def _compute_intent_probabilities(self, dominant_intent: str, robot_state: RobotState) -> Dict[str, float]:
        """Compute intent probability distribution"""
        # Base probabilities
        base_probs = {
            "idle": 0.1,
            "handover_request": 0.1,
            "handover_give": 0.05,
            "collaboration_start": 0.05,
            "obstacle_avoidance": 0.05
        }
        
        # Boost probability of dominant intent
        base_probs[dominant_intent] = 0.6 + 0.3 * self.predictability
        
        # Add uncertainty based on behavior type
        uncertainty = 1.0 - self.predictability
        for intent in base_probs:
            if intent != dominant_intent:
                base_probs[intent] += uncertainty * 0.1
        
        # Normalize
        total = sum(base_probs.values())
        return {intent: prob / total for intent, prob in base_probs.items()}
    
    def _generate_pose_keypoints(self, position: np.ndarray) -> np.ndarray:
        """Generate realistic pose keypoints"""
        # Simplified pose generation (17 keypoints * 3 coordinates)
        keypoints = np.zeros(51)
        
        # Head position (relative to center)
        keypoints[0:3] = position + np.array([0, 0, 0.3])  # Head above center
        
        # Shoulder positions
        keypoints[3:6] = position + np.array([0.2, 0, 0.2])   # Right shoulder
        keypoints[6:9] = position + np.array([-0.2, 0, 0.2])  # Left shoulder
        
        # Add some noise to make it realistic
        noise = self.rng.normal(0, 0.02, 51)
        keypoints += noise
        
        return keypoints
    
    def _predict_trajectory(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Predict future trajectory"""
        # Simple linear prediction for next 5 time steps
        trajectory = np.zeros((5, 3))
        dt = 0.5  # 0.5 second intervals
        
        for i in range(5):
            trajectory[i] = position + velocity * dt * (i + 1)
            # Add some uncertainty
            trajectory[i] += self.rng.normal(0, 0.05 * (i + 1), 3)
        
        return trajectory.flatten()
    
    def _compute_engagement_level(self, robot_state: RobotState) -> float:
        """Compute human engagement level with robot"""
        base_engagement = 0.5
        
        if self.current_intent in ["handover_request", "handover_give", "collaboration_start"]:
            base_engagement = 0.8
        elif self.current_intent == "obstacle_avoidance":
            base_engagement = 0.2
        
        # Add behavior-specific modulation
        if self.behavior_type == HumanBehaviorType.COLLABORATIVE:
            base_engagement += 0.2
        elif self.behavior_type == HumanBehaviorType.CAUTIOUS:
            base_engagement -= 0.1
        
        return np.clip(base_engagement, 0.0, 1.0)
    
    def _compute_comfort_level(self, robot_state: RobotState, position: np.ndarray) -> float:
        """Compute human comfort level"""
        # Distance-based comfort
        distance = np.linalg.norm(position - robot_state.ee_position)
        
        if distance > 1.0:
            distance_comfort = 1.0  # Comfortable when far
        elif distance > 0.5:
            distance_comfort = 0.8  # Slightly uncomfortable when medium distance
        elif distance > 0.3:
            distance_comfort = 0.6  # Uncomfortable when close
        else:
            distance_comfort = 0.3  # Very uncomfortable when too close
        
        # Velocity-based comfort (prefer slow movements)
        robot_speed = np.linalg.norm(robot_state.joint_velocities)
        if robot_speed < 0.5:
            speed_comfort = 1.0
        elif robot_speed < 1.0:
            speed_comfort = 0.8
        else:
            speed_comfort = 0.5
        
        # Behavior type modulation
        comfort_modifier = 1.0
        if self.behavior_type == HumanBehaviorType.CAUTIOUS:
            comfort_modifier = 0.8
        elif self.behavior_type == HumanBehaviorType.AGGRESSIVE:
            comfort_modifier = 1.2
        
        base_comfort = (distance_comfort * 0.7 + speed_comfort * 0.3) * comfort_modifier
        return np.clip(base_comfort, 0.0, 1.0)
    
    def _compute_trust_level(self, context: ContextState) -> float:
        """Compute human trust level in robot"""
        base_trust = 0.7
        
        # Decrease trust with safety violations
        if context.safety_violations > 0:
            trust_penalty = min(0.5, context.safety_violations * 0.1)
            base_trust -= trust_penalty
        
        # Increase trust with successful interactions
        if context.task_progress > 0.5:
            base_trust += 0.2 * context.task_progress
        
        # Behavior type effects
        if self.behavior_type == HumanBehaviorType.CAUTIOUS:
            base_trust -= 0.1  # More skeptical
        elif self.behavior_type == HumanBehaviorType.COLLABORATIVE:
            base_trust += 0.1  # More trusting
        
        return np.clip(base_trust, 0.0, 1.0)
    
    def _compute_intent_uncertainty(self, intent: str) -> float:
        """Compute uncertainty in intent recognition"""
        base_uncertainty = 1.0 - self.predictability
        
        # Some intents are inherently more uncertain
        if intent == "handover_give":
            base_uncertainty += 0.1  # Precise timing is uncertain
        elif intent == "obstacle_avoidance":
            base_uncertainty += 0.2  # Reaction can be unpredictable
        
        # Add noise based on configuration
        noise = self.rng.normal(0, self.config.intent_noise_std)
        
        return np.clip(base_uncertainty + noise, 0.1, 0.9)
    
    def _update_history(self, human_state: HumanState):
        """Update behavior history"""
        self.current_intent = max(human_state.intent_probabilities, 
                                key=human_state.intent_probabilities.get)
        self.intent_history.append(self.current_intent)
        self.position_history.append(human_state.position.copy())
        self.velocity_history.append(human_state.velocity.copy())
        
        # Keep limited history
        max_history = 100
        if len(self.intent_history) > max_history:
            self.intent_history = self.intent_history[-max_history:]
            self.position_history = self.position_history[-max_history:]
            self.velocity_history = self.velocity_history[-max_history:]


class RealisticHRISimulator:
    """
    Realistic HRI Simulator with physics-based dynamics and human behavior
    
    Provides a high-fidelity simulation environment for testing
    human-robot interaction algorithms.
    """
    
    def __init__(self, config: SimulationConfiguration):
        """Initialize realistic HRI simulator"""
        self.config = config
        
        # Create base HRI environment
        self.base_environment = create_default_hri_environment()
        
        # Create realistic human behavior model
        self.human_behavior = RealisticHumanBehaviorModel(config)
        
        # Simulation state
        self.current_time = 0.0
        self.step_count = 0
        self.is_running = False
        
        # Performance tracking
        self.simulation_times = []
        self.behavior_update_times = []
        
        logger.info("Initialized realistic HRI simulator")
    
    def reset(self, scenario_params: Dict[str, Any] = None) -> HRIState:
        """Reset simulator with optional scenario configuration"""
        # Configure scenario if provided
        if scenario_params:
            self._configure_scenario(scenario_params)
        
        # Reset base environment
        initial_state = self.base_environment.reset()
        
        # Reset human behavior model
        self.human_behavior.current_intent = "idle"
        self.human_behavior.intent_history = []
        self.human_behavior.position_history = []
        self.human_behavior.velocity_history = []
        
        # Reset simulation state
        self.current_time = 0.0
        self.step_count = 0
        
        # Update human behavior for initial state
        updated_human = self.human_behavior.update_behavior(
            initial_state.robot, initial_state.context, self.config.timestep
        )
        
        # Create realistic initial state
        realistic_state = HRIState(
            robot=initial_state.robot,
            human=updated_human,
            context=initial_state.context,
            timestamp=self.current_time
        )
        
        logger.debug(f"Simulator reset with scenario: {scenario_params}")
        return realistic_state
    
    def step(self, action: np.ndarray) -> Tuple[HRIState, Dict[str, Any], bool, Dict[str, Any]]:
        """Execute one simulation step with realistic dynamics"""
        step_start_time = time.time()
        
        # Get current state
        current_state = self.base_environment.current_state
        if current_state is None:
            raise RuntimeError("Simulator not initialized. Call reset() first.")
        
        # Execute action in base environment
        next_state, reward_dict, done, info = self.base_environment.step(action)
        
        # Update human behavior with realistic dynamics
        behavior_start_time = time.time()
        updated_human = self.human_behavior.update_behavior(
            next_state.robot, next_state.context, self.config.timestep
        )
        behavior_update_time = time.time() - behavior_start_time
        self.behavior_update_times.append(behavior_update_time)
        
        # Create realistic next state
        realistic_next_state = HRIState(
            robot=next_state.robot,
            human=updated_human,
            context=next_state.context,
            timestamp=self.current_time
        )
        
        # Add realistic sensor noise and delays
        noisy_state = self._add_sensor_effects(realistic_next_state)
        
        # Update simulation time
        self.current_time += self.config.timestep
        self.step_count += 1
        
        # Check termination conditions
        done = done or self._check_realistic_termination(noisy_state)
        
        # Add simulation-specific info
        simulation_time = time.time() - step_start_time
        self.simulation_times.append(simulation_time)
        
        sim_info = {
            'simulation_time': simulation_time,
            'behavior_update_time': behavior_update_time,
            'human_intent': updated_human.intent_probabilities,
            'human_uncertainty': updated_human.intent_uncertainty,
            'realistic_effects_applied': True
        }
        info.update(sim_info)
        
        return noisy_state, reward_dict, done, info
    
    def _configure_scenario(self, scenario_params: Dict[str, Any]):
        """Configure simulation scenario"""
        # Update human behavior type
        if 'human_behavior_type' in scenario_params:
            behavior_name = scenario_params['human_behavior_type']
            if behavior_name in [b.name.lower() for b in HumanBehaviorType]:
                self.config.human_behavior_type = HumanBehaviorType[behavior_name.upper()]
                self.human_behavior = RealisticHumanBehaviorModel(self.config)
        
        # Update noise levels
        if 'noise_level' in scenario_params:
            noise_multiplier = scenario_params['noise_level']
            self.config.position_noise_std *= noise_multiplier
            self.config.velocity_noise_std *= noise_multiplier
            self.config.intent_noise_std *= noise_multiplier
        
        # Update other parameters
        for param, value in scenario_params.items():
            if hasattr(self.config, param):
                setattr(self.config, param, value)
    
    def _add_sensor_effects(self, state: HRIState) -> HRIState:
        """Add realistic sensor noise and measurement delays"""
        # The human behavior model already adds position and velocity noise
        # Here we can add additional sensor effects
        
        # Add measurement delay (simplified - would require state buffer in real implementation)
        # For now, just add small additional noise
        
        additional_position_noise = np.random.normal(0, self.config.position_noise_std * 0.1, 3)
        
        # Create new human state with additional noise
        noisy_human = HumanState(
            position=state.human.position + additional_position_noise,
            velocity=state.human.velocity,
            pose_keypoints=state.human.pose_keypoints,
            intent_probabilities=state.human.intent_probabilities,
            predicted_trajectory=state.human.predicted_trajectory,
            attention_focus=state.human.attention_focus,
            engagement_level=state.human.engagement_level,
            comfort_level=state.human.comfort_level,
            trust_level=state.human.trust_level,
            position_uncertainty=state.human.position_uncertainty + self.config.position_noise_std * 0.1,
            intent_uncertainty=state.human.intent_uncertainty,
            behavior_uncertainty=state.human.behavior_uncertainty
        )
        
        return HRIState(
            robot=state.robot,
            human=noisy_human,
            context=state.context,
            timestamp=state.timestamp
        )
    
    def _check_realistic_termination(self, state: HRIState) -> bool:
        """Check realistic termination conditions"""
        # Time-based termination
        if self.current_time >= self.config.scenario_duration:
            return True
        
        # Safety-based termination (human moves too far away)
        human_robot_distance = np.linalg.norm(state.robot.ee_position - state.human.position)
        if human_robot_distance > 3.0:  # 3m maximum interaction range
            return True
        
        # Human comfort-based termination
        if state.human.comfort_level < 0.1:  # Human very uncomfortable
            return True
        
        return False
    
    def get_simulation_metrics(self) -> Dict[str, float]:
        """Get simulation performance metrics"""
        if not self.simulation_times:
            return {}
        
        return {
            'avg_simulation_time': np.mean(self.simulation_times),
            'max_simulation_time': np.max(self.simulation_times),
            'avg_behavior_update_time': np.mean(self.behavior_update_times) if self.behavior_update_times else 0.0,
            'total_steps': self.step_count,
            'simulation_frequency': 1.0 / self.config.timestep,
            'real_time_factor': np.mean(self.simulation_times) / self.config.timestep
        }
    
    def run_autonomous_simulation(self, duration: float) -> List[HRIState]:
        """Run autonomous simulation for analysis"""
        trajectory = []
        
        # Reset simulator
        current_state = self.reset()
        trajectory.append(current_state)
        
        start_time = time.time()
        while time.time() - start_time < duration:
            # Generate random action (for demonstration)
            action = np.random.uniform(-0.1, 0.1, 6)  # Small random movements
            
            # Step simulation
            next_state, _, done, _ = self.step(action)
            trajectory.append(next_state)
            
            if done:
                break
        
        return trajectory


def create_handover_simulation(behavior_type: HumanBehaviorType = HumanBehaviorType.NORMAL) -> RealisticHRISimulator:
    """Create simulation configured for handover scenarios"""
    config = SimulationConfiguration(
        timestep=0.05,  # 20 Hz for handover precision
        human_behavior_type=behavior_type,
        position_noise_std=0.005,  # Low noise for precise handover
        scenario_duration=20.0,
        interaction_complexity="medium"
    )
    return RealisticHRISimulator(config)


def create_safety_simulation(behavior_type: HumanBehaviorType = HumanBehaviorType.UNPREDICTABLE) -> RealisticHRISimulator:
    """Create simulation configured for safety testing"""
    config = SimulationConfiguration(
        timestep=0.01,  # High frequency for safety
        human_behavior_type=behavior_type,
        position_noise_std=0.02,  # Higher noise for uncertainty
        intent_noise_std=0.2,
        scenario_duration=15.0,
        interaction_complexity="complex"
    )
    return RealisticHRISimulator(config)


def create_adaptation_simulation(behavior_type: HumanBehaviorType = HumanBehaviorType.NORMAL) -> RealisticHRISimulator:
    """Create simulation configured for adaptation testing"""
    config = SimulationConfiguration(
        timestep=0.05,
        human_behavior_type=behavior_type,
        scenario_duration=30.0,  # Longer for adaptation
        interaction_complexity="medium"
    )
    return RealisticHRISimulator(config)


# Example usage and testing
if __name__ == "__main__":
    # Test realistic HRI simulator
    logger.info("Testing Realistic HRI Simulator")
    
    # Create simulator with normal human behavior
    config = SimulationConfiguration(
        human_behavior_type=HumanBehaviorType.NORMAL,
        scenario_duration=10.0
    )
    
    simulator = RealisticHRISimulator(config)
    
    # Test reset and step
    initial_state = simulator.reset()
    logger.info(f"Initial human intent: {initial_state.human.intent_probabilities}")
    logger.info(f"Initial human position: {initial_state.human.position}")
    
    # Run some steps
    for step in range(20):
        # Random action
        action = np.random.uniform(-0.05, 0.05, 6)
        
        # Step simulation
        next_state, reward, done, info = simulator.step(action)
        
        if step % 5 == 0:
            logger.info(f"Step {step}: Intent={max(next_state.human.intent_probabilities, key=next_state.human.intent_probabilities.get)}")
            logger.info(f"  Human position: {next_state.human.position}")
            logger.info(f"  Comfort level: {next_state.human.comfort_level:.2f}")
            logger.info(f"  Simulation time: {info['simulation_time']:.4f}s")
        
        if done:
            logger.info(f"Simulation terminated at step {step}")
            break
    
    # Get simulation metrics
    metrics = simulator.get_simulation_metrics()
    logger.info(f"Simulation metrics: {metrics}")
    
    # Test different behavior types
    for behavior_type in [HumanBehaviorType.AGGRESSIVE, HumanBehaviorType.CAUTIOUS]:
        logger.info(f"\nTesting {behavior_type.name} behavior...")
        
        config.human_behavior_type = behavior_type
        sim = RealisticHRISimulator(config)
        
        state = sim.reset()
        for i in range(5):
            action = np.random.uniform(-0.02, 0.02, 6)
            state, _, done, _ = sim.step(action)
            
            if done:
                break
        
        logger.info(f"Final human comfort: {state.human.comfort_level:.2f}")
        logger.info(f"Final intent uncertainty: {state.human.intent_uncertainty:.2f}")
    
    print("Realistic HRI Simulator test completed!")