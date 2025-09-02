"""
Human-Robot Interaction Bayesian RL Integration

This module integrates the Bayesian RL agent with human behavior models
and MPC controllers for real-time human-robot interaction scenarios.

Integration Components:
1. Human behavior prediction → RL state representation
2. Bayesian RL policy → MPC constraints and objectives  
3. MPC output → RL action execution and learning
4. Safety monitoring and constraint enforcement

Mathematical Foundation:
- State: s = [s_robot, s_human, s_context, s_intent]
- Action: a = [a_robot_high_level] → MPC(a) → τ_robot
- Reward: R(s,a,s') = R_task + R_safety + R_efficiency + R_human_comfort
- Policy: π*(s) with uncertainty quantification

Author: Bayesian RL Integration
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import time
from abc import ABC, abstractmethod

# Import components from previous phases
try:
    from src.agents.bayesian_rl_agent import BayesianRLAgent
    # Define missing classes locally to avoid import errors
    @dataclass
    class HumanState:
        position: np.ndarray = field(default_factory=lambda: np.zeros(3))
        velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
        gaze_direction: np.ndarray = field(default_factory=lambda: np.zeros(3))
        intent_confidence: float = 0.5
        
    @dataclass
    class ContextState:
        workspace_objects: List[Dict] = field(default_factory=list)
        task_phase: str = "unknown"
        safety_constraints: Dict = field(default_factory=dict)
        
    @dataclass
    class RobotAction:
        linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
        angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
        gripper_action: float = 0.0
        
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error: {e}. Some components may not be available.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegrationMode(Enum):
    """Integration modes for Bayesian RL with HRI"""
    HIERARCHICAL = auto()      # RL provides high-level commands to MPC
    DIRECT_CONTROL = auto()    # RL directly controls robot actions
    COOPERATIVE = auto()       # RL and MPC collaborate on control
    ADVISORY = auto()         # RL advises MPC on strategy


@dataclass
class HRIBayesianRLConfig:
    """Configuration for HRI Bayesian RL integration"""
    # Integration mode
    integration_mode: IntegrationMode = IntegrationMode.HIERARCHICAL
    
    # Bayesian RL configuration
    rl_algorithm: str = "gp_q_learning"  # "gp_q_learning", "psrl", "variational"
    use_exploration_manager: bool = True
    
    # Human behavior integration
    use_human_intent_prediction: bool = True
    intent_update_frequency: int = 5  # Steps between intent updates
    intent_prediction_horizon: int = 10
    
    # MPC integration
    use_mpc_controller: bool = True
    mpc_update_frequency: int = 1  # Steps between MPC updates
    mpc_constraint_adaptation: bool = True
    
    # Safety integration
    safety_monitoring: bool = True
    safety_violation_threshold: float = 0.05
    emergency_stop_enabled: bool = True
    
    # Learning parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    replay_buffer_size: int = 10000
    target_update_frequency: int = 100
    
    # Performance parameters
    real_time_constraint: float = 0.1  # Maximum decision time (seconds)
    uncertainty_threshold: float = 0.2  # High uncertainty threshold
    
    # Reward function weights
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'task_success': 1.0,
        'safety': 2.0,
        'efficiency': 0.5,
        'human_comfort': 1.0,
        'learning_progress': 0.3
    })


class HumanIntentPredictor:
    """
    Human intent prediction interface for Bayesian RL
    
    Integrates with human behavior models to provide intent predictions
    with uncertainty quantification for the RL agent.
    """
    
    def __init__(self, config: HRIBayesianRLConfig):
        """Initialize human intent predictor"""
        self.config = config
        self.intent_history = []
        self.prediction_uncertainty = 0.5
        
        # Intent categories for HRI tasks
        self.intent_categories = [
            'handover_request',
            'handover_give', 
            'collaboration_start',
            'collaboration_end',
            'obstacle_avoidance',
            'idle',
            'emergency_stop'
        ]
        
        logger.info("Initialized human intent predictor")
    
    def predict_intent(self, human_state: HumanState, 
                      context_state: ContextState) -> Dict[str, Any]:
        """
        Predict human intent with uncertainty
        
        Args:
            human_state: Current human state
            context_state: Current context
            
        Returns:
            Dictionary with intent predictions and uncertainty
        """
        # Simple rule-based predictor (would be replaced with learned model)
        intent_probs = np.zeros(len(self.intent_categories))
        
        # Distance-based intent inference
        if hasattr(human_state, 'position') and len(human_state.position) >= 3:
            # Close distance suggests handover intent
            if np.linalg.norm(human_state.position) < 0.5:
                intent_probs[0] = 0.7  # handover_request
                intent_probs[5] = 0.3  # idle
            else:
                intent_probs[5] = 0.8  # idle
                intent_probs[0] = 0.2  # handover_request
        else:
            # Default distribution
            intent_probs[5] = 0.6  # idle
            intent_probs[0] = 0.4  # handover_request
        
        # Add uncertainty based on context
        if context_state.interaction_phase.name == 'APPROACH':
            intent_probs[0] += 0.2  # More likely to be handover
            intent_probs[5] -= 0.2
        
        # Normalize probabilities
        intent_probs = np.clip(intent_probs, 0, 1)
        intent_probs /= np.sum(intent_probs)
        
        # Compute uncertainty (entropy-based)
        entropy = -np.sum(intent_probs * np.log(intent_probs + 1e-8))
        uncertainty = entropy / np.log(len(self.intent_categories))  # Normalized entropy
        
        intent_dict = dict(zip(self.intent_categories, intent_probs))
        
        # Update history
        self.intent_history.append({
            'timestamp': time.time(),
            'intent_probs': intent_dict,
            'uncertainty': uncertainty
        })
        
        # Keep limited history
        if len(self.intent_history) > 100:
            self.intent_history.pop(0)
        
        return {
            'intent_probabilities': intent_dict,
            'uncertainty': uncertainty,
            'dominant_intent': self.intent_categories[np.argmax(intent_probs)],
            'confidence': np.max(intent_probs),
            'prediction_horizon': self.config.intent_prediction_horizon
        }
    
    def get_intent_trajectory(self, steps: int = 10) -> Dict[str, Any]:
        """Predict intent evolution over multiple steps"""
        # Simple trajectory prediction (would use temporal models in practice)
        if not self.intent_history:
            return {'trajectory': [], 'uncertainty_trajectory': []}
        
        last_intent = self.intent_history[-1]['intent_probs']
        trajectory = []
        uncertainty_trajectory = []
        
        for step in range(steps):
            # Add some temporal dynamics (random walk for now)
            noise = np.random.normal(0, 0.1, len(self.intent_categories))
            next_probs = np.array(list(last_intent.values())) + noise
            next_probs = np.clip(next_probs, 0, 1)
            next_probs /= np.sum(next_probs)
            
            next_intent = dict(zip(self.intent_categories, next_probs))
            trajectory.append(next_intent)
            
            # Uncertainty increases with prediction horizon
            uncertainty = 0.3 + 0.1 * step
            uncertainty_trajectory.append(min(uncertainty, 1.0))
            
            last_intent = next_intent
        
        return {
            'trajectory': trajectory,
            'uncertainty_trajectory': uncertainty_trajectory
        }


class MPCIntegrationLayer:
    """
    Integration layer between Bayesian RL and MPC controller
    
    Translates high-level RL decisions to MPC constraints and objectives,
    and provides feedback from MPC execution to RL learning.
    """
    
    def __init__(self, config: HRIBayesianRLConfig):
        """Initialize MPC integration layer"""
        self.config = config
        self.mpc_controller = None  # Would be initialized with actual MPC controller
        
        # Command mapping
        self.high_level_commands = [
            'move_to_human',
            'handover_position', 
            'retreat_safe',
            'collaboration_mode',
            'emergency_stop',
            'idle_position'
        ]
        
        logger.info("Initialized MPC integration layer")
    
    def translate_rl_action(self, rl_action: np.ndarray, 
                           current_state: HRIState,
                           human_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate RL action to MPC parameters
        
        Args:
            rl_action: High-level action from RL agent
            current_state: Current HRI state
            human_intent: Human intent prediction
            
        Returns:
            Dictionary with MPC parameters
        """
        # Map continuous RL action to discrete command + parameters
        command_idx = int(np.clip(rl_action[0] * len(self.high_level_commands), 
                                0, len(self.high_level_commands) - 1))
        command = self.high_level_commands[command_idx]
        
        # Extract parameters from remaining action dimensions
        if len(rl_action) > 1:
            command_params = rl_action[1:]
        else:
            command_params = np.array([0.0])
        
        # Adapt MPC constraints based on human intent
        safety_constraints = self._adapt_safety_constraints(human_intent, current_state)
        
        # Adapt cost function weights
        cost_weights = self._adapt_cost_weights(human_intent, current_state)
        
        # Generate target pose based on command
        target_pose = self._generate_target_pose(command, command_params, current_state)
        
        return {
            'command': command,
            'command_params': command_params,
            'target_pose': target_pose,
            'safety_constraints': safety_constraints,
            'cost_weights': cost_weights,
            'prediction_horizon': 15,  # MPC horizon
            'human_prediction': human_intent
        }
    
    def _adapt_safety_constraints(self, human_intent: Dict[str, Any], 
                                current_state: HRIState) -> Dict[str, float]:
        """Adapt safety constraints based on human intent"""
        base_safe_distance = 0.3  # meters
        
        # Increase safety distance if human intent is uncertain
        uncertainty = human_intent.get('uncertainty', 0.5)
        safety_distance = base_safe_distance * (1 + uncertainty)
        
        # Adapt based on dominant intent
        dominant_intent = human_intent.get('dominant_intent', 'idle')
        
        if dominant_intent == 'handover_request':
            safety_distance *= 0.8  # Allow closer approach for handover
        elif dominant_intent == 'emergency_stop':
            safety_distance *= 2.0  # Extra safety margin
        
        return {
            'min_human_distance': safety_distance,
            'max_velocity': 1.0 if dominant_intent != 'emergency_stop' else 0.1,
            'max_acceleration': 2.0,
            'workspace_bounds': [-2, 2, -2, 2, 0, 2]  # [x_min, x_max, y_min, y_max, z_min, z_max]
        }
    
    def _adapt_cost_weights(self, human_intent: Dict[str, Any], 
                          current_state: HRIState) -> Dict[str, float]:
        """Adapt MPC cost function weights based on intent"""
        base_weights = {
            'position_error': 1.0,
            'velocity_smoothness': 0.1,
            'control_effort': 0.01,
            'safety_cost': 2.0,
            'human_comfort': 1.0
        }
        
        # Modify weights based on intent
        dominant_intent = human_intent.get('dominant_intent', 'idle')
        uncertainty = human_intent.get('uncertainty', 0.5)
        
        if dominant_intent == 'handover_request':
            base_weights['position_error'] *= 1.5  # Emphasize reaching target
            base_weights['human_comfort'] *= 1.2
        elif dominant_intent == 'emergency_stop':
            base_weights['safety_cost'] *= 3.0
            base_weights['velocity_smoothness'] *= 0.5  # Allow rapid stopping
        
        # Increase safety weight with uncertainty
        base_weights['safety_cost'] *= (1 + uncertainty)
        
        return base_weights
    
    def _generate_target_pose(self, command: str, params: np.ndarray, 
                            current_state: HRIState) -> np.ndarray:
        """Generate target pose based on command"""
        # Default to current end-effector position
        target_pose = current_state.robot.ee_position.copy()
        
        if command == 'move_to_human':
            # Move towards human position (with safety margin)
            human_pos = current_state.human.position
            direction = human_pos - current_state.robot.ee_position
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm > 0:
                # Move 70% of the way towards human (with safety margin)
                target_pose = current_state.robot.ee_position + 0.7 * direction
        
        elif command == 'handover_position':
            # Position for handover (slightly in front of human)
            human_pos = current_state.human.position
            # Assume handover position is 20cm in front of human
            target_pose = human_pos + np.array([0.2, 0, 0])
        
        elif command == 'retreat_safe':
            # Move to safe distance from human
            human_pos = current_state.human.position
            direction = current_state.robot.ee_position - human_pos
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm > 0 and direction_norm < 1.0:
                # Move to 1m away from human
                target_pose = human_pos + (direction / direction_norm) * 1.0
        
        elif command == 'idle_position':
            # Default idle position
            target_pose = np.array([0.5, 0.0, 0.5])
        
        # Apply parameter modulations if provided
        if len(params) >= 3:
            target_pose += params[:3] * 0.1  # Small modulations
        
        return target_pose
    
    def execute_mpc_command(self, mpc_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute MPC command and return execution results
        
        Args:
            mpc_params: MPC parameters from RL translation
            
        Returns:
            Dictionary with execution results
        """
        # This would interface with actual MPC controller
        # For now, simulate execution results
        
        execution_time = np.random.uniform(0.05, 0.15)  # Simulate MPC solve time
        success = np.random.random() > 0.05  # 95% success rate
        
        # Simulate robot motion
        target_pose = mpc_params['target_pose']
        current_pose = np.array([0.5, 0.0, 0.5])  # Mock current pose
        
        if success:
            # Simulate movement towards target
            next_pose = current_pose + 0.1 * (target_pose - current_pose)
            control_effort = np.linalg.norm(target_pose - current_pose) * 0.5
        else:
            next_pose = current_pose
            control_effort = 0.0
        
        return {
            'success': success,
            'execution_time': execution_time,
            'next_pose': next_pose,
            'control_effort': control_effort,
            'safety_violations': 0,
            'constraint_violations': 0,
            'mpc_iterations': np.random.randint(10, 50),
            'command_executed': mpc_params['command']
        }


class SafetyMonitor:
    """
    Real-time safety monitoring for HRI Bayesian RL
    
    Monitors safety constraints and can override RL decisions
    if safety violations are detected.
    """
    
    def __init__(self, config: HRIBayesianRLConfig):
        """Initialize safety monitor"""
        self.config = config
        self.safety_violations = []
        self.emergency_stop_active = False
        
        # Safety thresholds
        self.min_human_distance = 0.2  # meters
        self.max_robot_velocity = 2.0   # m/s
        self.max_robot_acceleration = 5.0  # m/s²
        
        logger.info("Initialized safety monitor")
    
    def check_safety(self, current_state: HRIState, 
                    proposed_action: np.ndarray,
                    mpc_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check safety constraints for current state and proposed action
        
        Args:
            current_state: Current HRI state
            proposed_action: Proposed RL action
            mpc_params: Translated MPC parameters
            
        Returns:
            Dictionary with safety assessment
        """
        violations = []
        risk_level = 0.0
        
        # Check human-robot distance
        human_distance = np.linalg.norm(
            current_state.robot.ee_position - current_state.human.position
        )
        
        if human_distance < self.min_human_distance:
            violations.append({
                'type': 'human_distance',
                'severity': 'high',
                'value': human_distance,
                'threshold': self.min_human_distance,
                'risk_increase': 0.5
            })
            risk_level += 0.5
        
        # Check robot velocity
        robot_velocity = np.linalg.norm(current_state.robot.joint_velocities)
        if robot_velocity > self.max_robot_velocity:
            violations.append({
                'type': 'robot_velocity',
                'severity': 'medium',
                'value': robot_velocity,
                'threshold': self.max_robot_velocity,
                'risk_increase': 0.3
            })
            risk_level += 0.3
        
        # Check MPC constraints
        if 'safety_constraints' in mpc_params:
            safety_constraints = mpc_params['safety_constraints']
            
            if human_distance < safety_constraints.get('min_human_distance', self.min_human_distance):
                violations.append({
                    'type': 'mpc_safety_distance',
                    'severity': 'high', 
                    'value': human_distance,
                    'threshold': safety_constraints['min_human_distance'],
                    'risk_increase': 0.4
                })
                risk_level += 0.4
        
        # Check emergency conditions
        emergency_stop_needed = (
            risk_level > 0.7 or 
            current_state.context.emergency_stop_active or
            any(v['severity'] == 'high' for v in violations)
        )
        
        # Update safety history
        safety_record = {
            'timestamp': time.time(),
            'violations': violations,
            'risk_level': risk_level,
            'emergency_stop_needed': emergency_stop_needed,
            'human_distance': human_distance,
            'robot_velocity': robot_velocity
        }
        
        self.safety_violations.append(safety_record)
        
        # Keep limited history
        if len(self.safety_violations) > 1000:
            self.safety_violations.pop(0)
        
        return {
            'safe': len(violations) == 0,
            'violations': violations,
            'risk_level': risk_level,
            'emergency_stop_needed': emergency_stop_needed,
            'recommended_action': 'emergency_stop' if emergency_stop_needed else 'continue',
            'safety_score': max(0, 1 - risk_level)
        }
    
    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get safety monitoring statistics"""
        if not self.safety_violations:
            return {'total_records': 0}
        
        recent_violations = self.safety_violations[-100:]  # Last 100 records
        
        violation_types = {}
        risk_levels = []
        
        for record in recent_violations:
            risk_levels.append(record['risk_level'])
            for violation in record['violations']:
                vtype = violation['type']
                if vtype not in violation_types:
                    violation_types[vtype] = 0
                violation_types[vtype] += 1
        
        return {
            'total_records': len(self.safety_violations),
            'recent_records': len(recent_violations),
            'violation_types': violation_types,
            'avg_risk_level': np.mean(risk_levels) if risk_levels else 0.0,
            'max_risk_level': np.max(risk_levels) if risk_levels else 0.0,
            'emergency_stops': sum(1 for r in recent_violations if r['emergency_stop_needed']),
            'safety_rate': sum(1 for r in recent_violations if len(r['violations']) == 0) / len(recent_violations) if recent_violations else 1.0
        }


class HRIBayesianRLIntegration:
    """
    Main integration class for HRI Bayesian RL
    
    Coordinates all components: RL agent, human behavior prediction,
    MPC controller, and safety monitoring for seamless HRI.
    """
    
    def __init__(self, config: HRIBayesianRLConfig = None):
        """Initialize HRI Bayesian RL integration"""
        self.config = config or HRIBayesianRLConfig()
        
        # Initialize components
        self.intent_predictor = HumanIntentPredictor(self.config)
        self.mpc_integration = MPCIntegrationLayer(self.config)
        self.safety_monitor = SafetyMonitor(self.config)
        
        # Initialize RL agent based on configuration
        self.rl_agent = self._initialize_rl_agent()
        
        # Initialize exploration manager
        if self.config.use_exploration_manager:
            exploration_config = ExplorationConfig()
            self.exploration_manager = ExplorationManager(exploration_config)
        else:
            self.exploration_manager = None
        
        # Initialize environment
        self.environment = HRIEnvironment()
        
        # Performance tracking
        self.episode_count = 0
        self.step_count = 0
        self.performance_history = []
        
        logger.info("Initialized HRI Bayesian RL integration")
    
    def _initialize_rl_agent(self) -> Union['BayesianRLAgent', 'GPBayesianQLearning', 'PSRLAgent']:
        """Initialize RL agent based on configuration"""
        state_dim = 164  # From HRIState
        action_dim = 6   # High-level action dimension
        
        if self.config.rl_algorithm == "gp_q_learning":
            gp_config = GPQConfiguration()
            return GPBayesianQLearning(state_dim, action_dim, gp_config)
        
        elif self.config.rl_algorithm == "psrl":
            psrl_config = PSRLConfiguration(state_dim=state_dim, action_dim=action_dim)
            return PSRLAgent(state_dim, action_dim, psrl_config)
        
        else:
            # Default to Bayesian RL agent
            rl_config = BayesianRLConfiguration()
            return BayesianRLAgent(rl_config)
    
    def step(self, current_state: HRIState) -> Dict[str, Any]:
        """
        Execute one integration step
        
        Args:
            current_state: Current HRI state
            
        Returns:
            Dictionary with step results
        """
        step_start_time = time.time()
        
        # 1. Predict human intent
        human_intent = self.intent_predictor.predict_intent(
            current_state.human, current_state.context
        )
        
        # 2. Select RL action
        state_vector = current_state.to_vector()
        
        if hasattr(self.rl_agent, 'select_action'):
            # Use Bayesian RL agent
            rl_action, selection_info = self.rl_agent.select_action(state_vector)
        elif self.exploration_manager:
            # Use exploration manager with RL agent
            rl_action, selection_info = self.exploration_manager.select_action(
                state_vector, self.rl_agent
            )
        else:
            # Fallback: random action
            rl_action = np.random.uniform(-1, 1, 6)
            selection_info = {'strategy': 'random'}
        
        # 3. Translate RL action to MPC parameters
        mpc_params = self.mpc_integration.translate_rl_action(
            rl_action, current_state, human_intent
        )
        
        # 4. Safety check
        safety_assessment = self.safety_monitor.check_safety(
            current_state, rl_action, mpc_params
        )
        
        # 5. Execute action (with safety override if needed)
        if safety_assessment['emergency_stop_needed']:
            # Override with emergency stop
            mpc_params['command'] = 'emergency_stop'
            mpc_params['target_pose'] = current_state.robot.ee_position  # Stay in place
            
            execution_results = {
                'success': True,
                'execution_time': 0.001,
                'next_pose': current_state.robot.ee_position,
                'control_effort': 0.0,
                'safety_violations': 0,
                'constraint_violations': 0,
                'command_executed': 'emergency_stop'
            }
        else:
            # Execute MPC command
            execution_results = self.mpc_integration.execute_mpc_command(mpc_params)
        
        # 6. Compute reward
        reward = self._compute_reward(current_state, rl_action, execution_results, safety_assessment)
        
        # 7. Update RL agent (if experience is available)
        if hasattr(self.rl_agent, 'add_experience'):
            # Simulate next state (would come from environment in practice)
            next_state = self._simulate_next_state(current_state, execution_results)
            
            self.rl_agent.add_experience(
                state_vector, rl_action, reward['total'], 
                next_state.to_vector(), False
            )
            
            # Periodic learning updates
            if self.step_count % 10 == 0:
                if hasattr(self.rl_agent, 'update_q_function'):
                    self.rl_agent.update_q_function(batch_size=self.config.batch_size)
        
        # 8. Track performance
        step_time = time.time() - step_start_time
        
        step_results = {
            'step_count': self.step_count,
            'step_time': step_time,
            'human_intent': human_intent,
            'rl_action': rl_action,
            'selection_info': selection_info,
            'mpc_params': mpc_params,
            'safety_assessment': safety_assessment,
            'execution_results': execution_results,
            'reward': reward,
            'real_time_constraint_met': step_time < self.config.real_time_constraint
        }
        
        self.step_count += 1
        return step_results
    
    def _compute_reward(self, state: HRIState, action: np.ndarray,
                       execution_results: Dict[str, Any], 
                       safety_assessment: Dict[str, Any]) -> Dict[str, float]:
        """Compute multi-objective reward"""
        rewards = {}
        
        # Task success reward
        rewards['task_success'] = 1.0 if execution_results['success'] else -0.5
        
        # Safety reward
        safety_score = safety_assessment['safety_score']
        rewards['safety'] = safety_score * 2 - 1  # Map [0,1] to [-1,1]
        
        # Efficiency reward (penalize high control effort and time)
        control_effort = execution_results.get('control_effort', 0)
        execution_time = execution_results.get('execution_time', 0.1)
        rewards['efficiency'] = -0.1 * control_effort - 0.5 * execution_time
        
        # Human comfort reward (based on distance and smooth motion)
        human_distance = np.linalg.norm(state.robot.ee_position - state.human.position)
        comfort_distance = max(0, (human_distance - 0.2) / 1.0)  # Comfort zone 0.2-1.2m
        rewards['human_comfort'] = comfort_distance
        
        # Learning progress reward (encourage exploration)
        if hasattr(self.rl_agent, 'get_performance_metrics'):
            metrics = self.rl_agent.get_performance_metrics()
            exploration_bonus = 0.1 if metrics.get('is_trained', False) else 0.2
            rewards['learning_progress'] = exploration_bonus
        else:
            rewards['learning_progress'] = 0.1
        
        # Compute weighted total reward
        total_reward = sum(
            self.config.reward_weights.get(key, 1.0) * value 
            for key, value in rewards.items()
        )
        
        rewards['total'] = total_reward
        return rewards
    
    def _simulate_next_state(self, current_state: HRIState, 
                           execution_results: Dict[str, Any]) -> HRIState:
        """Simulate next state based on execution results"""
        # Create new state with updated robot pose
        next_robot_state = current_state.robot
        if 'next_pose' in execution_results:
            next_robot_state.ee_position = execution_results['next_pose']
        
        # Simple human state evolution (would use human behavior model)
        next_human_state = current_state.human
        
        # Update context
        next_context_state = current_state.context
        if execution_results.get('safety_violations', 0) > 0:
            next_context_state.safety_violations += execution_results['safety_violations']
        
        return HRIState(
            robot=next_robot_state,
            human=next_human_state,
            context=next_context_state,
            timestamp=current_state.timestamp + 0.1
        )
    
    def run_episode(self, max_steps: int = 100) -> Dict[str, Any]:
        """
        Run complete episode of HRI interaction
        
        Args:
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary with episode results
        """
        # Initialize environment
        current_state = self.environment.reset()
        
        episode_rewards = []
        episode_safety_scores = []
        episode_steps = []
        
        for step in range(max_steps):
            # Execute step
            step_results = self.step(current_state)
            
            # Track metrics
            episode_rewards.append(step_results['reward']['total'])
            episode_safety_scores.append(step_results['safety_assessment']['safety_score'])
            episode_steps.append(step_results)
            
            # Update state (simulate environment step)
            next_state, reward_dict, done, info = self.environment.step(
                step_results['rl_action']
            )
            current_state = next_state
            
            if done:
                break
        
        # Episode summary
        episode_results = {
            'episode_count': self.episode_count,
            'steps': len(episode_steps),
            'total_reward': sum(episode_rewards),
            'avg_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'avg_safety_score': np.mean(episode_safety_scores) if episode_safety_scores else 0,
            'safety_violations': sum(s['safety_assessment'].get('violations', []) for s in episode_steps),
            'real_time_performance': sum(s['real_time_constraint_met'] for s in episode_steps) / len(episode_steps),
            'detailed_steps': episode_steps
        }
        
        self.performance_history.append(episode_results)
        self.episode_count += 1
        
        logger.info(f"Episode {self.episode_count} completed: "
                   f"reward={episode_results['total_reward']:.3f}, "
                   f"safety={episode_results['avg_safety_score']:.3f}, "
                   f"steps={episode_results['steps']}")
        
        return episode_results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.performance_history:
            return {'no_episodes': True}
        
        recent_episodes = self.performance_history[-10:]  # Last 10 episodes
        
        # Aggregate metrics
        avg_reward = np.mean([ep['avg_reward'] for ep in recent_episodes])
        avg_safety = np.mean([ep['avg_safety_score'] for ep in recent_episodes])
        avg_steps = np.mean([ep['steps'] for ep in recent_episodes])
        success_rate = sum(ep['total_reward'] > 0 for ep in recent_episodes) / len(recent_episodes)
        
        # Component-specific metrics
        rl_metrics = {}
        if hasattr(self.rl_agent, 'get_performance_metrics'):
            rl_metrics = self.rl_agent.get_performance_metrics()
        
        safety_stats = self.safety_monitor.get_safety_statistics()
        
        exploration_stats = {}
        if self.exploration_manager:
            exploration_stats = self.exploration_manager.get_strategy_stats()
        
        return {
            'total_episodes': self.episode_count,
            'total_steps': self.step_count,
            'recent_performance': {
                'avg_reward': avg_reward,
                'avg_safety_score': avg_safety,
                'avg_steps_per_episode': avg_steps,
                'success_rate': success_rate
            },
            'rl_agent_metrics': rl_metrics,
            'safety_statistics': safety_stats,
            'exploration_statistics': exploration_stats,
            'integration_config': {
                'rl_algorithm': self.config.rl_algorithm,
                'integration_mode': self.config.integration_mode.name,
                'real_time_constraint': self.config.real_time_constraint
            }
        }


# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing HRI Bayesian RL Integration")
    
    # Configuration
    config = HRIBayesianRLConfig(
        rl_algorithm="gp_q_learning",
        integration_mode=IntegrationMode.HIERARCHICAL,
        real_time_constraint=0.2
    )
    
    # Initialize integration
    hri_integration = HRIBayesianRLIntegration(config)
    
    # Run test episode
    episode_results = hri_integration.run_episode(max_steps=20)
    
    logger.info(f"Test episode results: {episode_results}")
    
    # Get performance summary
    performance_summary = hri_integration.get_performance_summary()
    logger.info(f"Performance summary: {performance_summary}")
    
    print("HRI Bayesian RL Integration test completed successfully!")