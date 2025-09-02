"""
Synthetic human behavior dataset generator for human-robot interaction.

This module generates realistic synthetic human behavior data including:
- Hand/arm trajectories for reaching motions
- Gaze patterns (eye tracking simulation)
- Gesture sequences (wave, point, grab, handover)
- Temporal patterns with realistic noise

Mathematical formulations are included for each trajectory generation method.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import pickle
import json

from ..models.human_behavior import BehaviorType, HumanState, BehaviorPrediction
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GestureType(Enum):
    """Extended gesture types for synthetic data generation."""
    WAVE = "wave"
    POINT = "point"
    GRAB = "grab"
    HANDOVER = "handover"
    REACH = "reach"
    IDLE = "idle"


@dataclass
class TrajectoryParameters:
    """
    Parameters for trajectory generation.
    
    Attributes:
        start_position: Starting 3D position [x, y, z]
        end_position: Target 3D position [x, y, z]
        duration: Trajectory duration (seconds)
        velocity_profile: Velocity profile type ('bell_curve', 'linear', 'exponential')
        noise_std: Standard deviation for Gaussian noise
        intermediate_points: Optional intermediate waypoints
    """
    start_position: np.ndarray
    end_position: np.ndarray
    duration: float
    velocity_profile: str = 'bell_curve'
    noise_std: float = 0.01
    intermediate_points: Optional[List[np.ndarray]] = None
    
    def __post_init__(self) -> None:
        """Validate trajectory parameters."""
        if self.start_position.shape != (3,):
            raise ValueError("Start position must be 3D")
        if self.end_position.shape != (3,):
            raise ValueError("End position must be 3D")
        if self.duration <= 0:
            raise ValueError("Duration must be positive")
        if self.noise_std < 0:
            raise ValueError("Noise standard deviation must be non-negative")


@dataclass
class GazeParameters:
    """
    Parameters for gaze pattern generation.
    
    Attributes:
        focal_points: List of 3D points to focus on
        dwell_times: Time spent focusing on each point
        saccade_duration: Duration of rapid eye movements
        fixation_noise: Standard deviation of fixation noise
        pursuit_gain: Gain for smooth pursuit movements
    """
    focal_points: List[np.ndarray]
    dwell_times: List[float]
    saccade_duration: float = 0.05
    fixation_noise: float = 0.002
    pursuit_gain: float = 0.95


@dataclass
class SyntheticSequence:
    """
    Complete synthetic interaction sequence.
    
    Attributes:
        sequence_id: Unique identifier for the sequence
        gesture_type: Type of gesture being performed
        hand_trajectory: 3D hand position over time [N, 3]
        gaze_trajectory: 3D gaze direction over time [N, 3]
        timestamps: Time stamps for each observation [N]
        intent_labels: Ground truth intent at each time step [N]
        context_info: Environmental context information
        noise_level: Applied noise level for this sequence
    """
    sequence_id: str
    gesture_type: GestureType
    hand_trajectory: np.ndarray
    gaze_trajectory: np.ndarray
    timestamps: np.ndarray
    intent_labels: List[BehaviorType]
    context_info: Dict[str, Any]
    noise_level: float


class SyntheticHumanBehaviorGenerator:
    """
    Generator for synthetic human behavior data.
    
    This class implements mathematical models for generating realistic
    human motion patterns, gaze behaviors, and gesture sequences for
    training and testing human-robot interaction algorithms.
    
    Mathematical Foundation:
    - Trajectories use minimum jerk principles: x(t) = a₀ + a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵
    - Gaze follows Fitts' law: MT = a + b·log₂(2D/W)
    - Noise follows multivariate Gaussian: N(0, Σ)
    """
    
    def __init__(
        self,
        workspace_bounds: np.ndarray,
        sampling_frequency: float = 30.0,
        random_seed: Optional[int] = None
    ) -> None:
        """
        Initialize the synthetic behavior generator.
        
        Args:
            workspace_bounds: 3D workspace boundaries [x_min, x_max, y_min, y_max, z_min, z_max]
            sampling_frequency: Data sampling frequency (Hz)
            random_seed: Random seed for reproducible generation
        """
        self.workspace_bounds = workspace_bounds
        self.sampling_frequency = sampling_frequency
        self.dt = 1.0 / sampling_frequency
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Workspace constraints
        self.x_bounds = workspace_bounds[:2]
        self.y_bounds = workspace_bounds[2:4]  
        self.z_bounds = workspace_bounds[4:6]
        
        # Gesture-specific parameters
        self._initialize_gesture_parameters()
        
        logger.info(f"Initialized synthetic generator with {sampling_frequency}Hz sampling")
    
    def _initialize_gesture_parameters(self) -> None:
        """Initialize parameters for different gesture types."""
        self.gesture_params = {
            GestureType.WAVE: {
                'amplitude': 0.15,
                'frequency': 2.0,
                'cycles': 2.5,
                'base_duration': 2.0
            },
            GestureType.POINT: {
                'extension_ratio': 0.8,
                'hold_duration': 1.0,
                'base_duration': 1.5
            },
            GestureType.GRAB: {
                'approach_speed': 0.3,
                'grasp_duration': 0.5,
                'base_duration': 2.0
            },
            GestureType.HANDOVER: {
                'approach_phase': 0.4,
                'present_phase': 0.4,
                'wait_phase': 0.2,
                'base_duration': 3.0
            },
            GestureType.REACH: {
                'acceleration_phase': 0.3,
                'deceleration_phase': 0.7,
                'base_duration': 1.5
            }
        }
    
    def generate_minimum_jerk_trajectory(
        self,
        params: TrajectoryParameters,
        include_derivatives: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate minimum jerk trajectory between two points.
        
        Mathematical formulation:
        For minimum jerk trajectory from (0,0) to (T,D):
        x(t) = D * (10(t/T)³ - 15(t/T)⁴ + 6(t/T)⁵)
        
        Args:
            params: Trajectory parameters
            include_derivatives: Whether to compute velocity and acceleration
            
        Returns:
            Tuple of (timestamps, positions, velocities, accelerations)
        """
        # Time vector
        n_points = int(params.duration * self.sampling_frequency)
        t = np.linspace(0, params.duration, n_points)
        tau = t / params.duration  # Normalized time [0, 1]
        
        # Minimum jerk profile (normalized)
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        
        # Handle intermediate waypoints using piecewise minimum jerk
        if params.intermediate_points is not None:
            positions = self._generate_waypoint_trajectory(params, t)
        else:
            # Direct trajectory
            displacement = params.end_position - params.start_position
            positions = params.start_position[None, :] + s[:, None] * displacement[None, :]
        
        # Add realistic noise
        if params.noise_std > 0:
            noise = np.random.normal(0, params.noise_std, positions.shape)
            positions += noise
        
        # Compute derivatives if requested
        velocities = np.zeros_like(positions)
        accelerations = np.zeros_like(positions)
        
        if include_derivatives:
            # Velocity: ds/dt = (30t² - 60t³ + 30t⁴) / T
            s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / params.duration
            
            if params.intermediate_points is None:
                velocities = s_dot[:, None] * displacement[None, :]
            else:
                # Numerical differentiation for waypoint trajectories
                velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * self.dt)
                velocities[0] = (positions[1] - positions[0]) / self.dt
                velocities[-1] = (positions[-1] - positions[-2]) / self.dt
            
            # Acceleration: d²s/dt² = (60t - 180t² + 120t³) / T²
            s_ddot = (60 * tau - 180 * tau**2 + 120 * tau**3) / (params.duration**2)
            
            if params.intermediate_points is None:
                accelerations = s_ddot[:, None] * displacement[None, :]
            else:
                # Numerical second derivative
                accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (2 * self.dt)
                accelerations[0] = (velocities[1] - velocities[0]) / self.dt
                accelerations[-1] = (velocities[-1] - velocities[-2]) / self.dt
        
        return t, positions, velocities, accelerations
    
    def _generate_waypoint_trajectory(
        self,
        params: TrajectoryParameters,
        t: np.ndarray
    ) -> np.ndarray:
        """Generate trajectory through multiple waypoints."""
        waypoints = [params.start_position] + params.intermediate_points + [params.end_position]
        n_segments = len(waypoints) - 1
        segment_duration = params.duration / n_segments
        
        positions = np.zeros((len(t), 3))
        
        for i in range(n_segments):
            # Time indices for this segment
            t_start = i * segment_duration
            t_end = (i + 1) * segment_duration
            
            segment_mask = (t >= t_start) & (t < t_end)
            if i == n_segments - 1:  # Include endpoint for last segment
                segment_mask = (t >= t_start) & (t <= t_end)
            
            t_segment = t[segment_mask] - t_start
            tau_segment = t_segment / segment_duration
            
            # Minimum jerk for this segment
            s = 10 * tau_segment**3 - 15 * tau_segment**4 + 6 * tau_segment**5
            
            displacement = waypoints[i + 1] - waypoints[i]
            positions[segment_mask] = waypoints[i][None, :] + s[:, None] * displacement[None, :]
        
        return positions
    
    def generate_gaze_trajectory(
        self,
        params: GazeParameters,
        duration: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate realistic gaze trajectory using saccades and fixations.
        
        Mathematical model:
        - Saccadic movements: exponential approach to target
        - Fixation stability: small random walk with return tendency
        - Fitts' law for saccade timing: MT = a + b·log₂(2D/W)
        
        Args:
            params: Gaze pattern parameters
            duration: Total trajectory duration
            
        Returns:
            Tuple of (timestamps, gaze_directions)
        """
        n_points = int(duration * self.sampling_frequency)
        t = np.linspace(0, duration, n_points)
        gaze_trajectory = np.zeros((n_points, 3))
        
        current_gaze = params.focal_points[0].copy()
        current_time = 0.0
        focal_idx = 0
        
        for i, time_point in enumerate(t):
            if focal_idx < len(params.focal_points) - 1:
                # Check if it's time to move to next focal point
                if time_point >= current_time + params.dwell_times[focal_idx]:
                    focal_idx += 1
                    current_time = time_point
            
            target_gaze = params.focal_points[min(focal_idx, len(params.focal_points) - 1)]
            
            # Saccadic movement (exponential approach)
            gaze_error = target_gaze - current_gaze
            saccade_velocity = gaze_error / params.saccade_duration
            
            # Apply smooth pursuit with gain
            current_gaze += params.pursuit_gain * saccade_velocity * self.dt
            
            # Add fixation noise (small random walk)
            fixation_noise = np.random.normal(0, params.fixation_noise, 3)
            current_gaze += fixation_noise
            
            gaze_trajectory[i] = current_gaze
        
        return t, gaze_trajectory
    
    def generate_gesture_sequence(
        self,
        gesture_type: GestureType,
        context: Dict[str, Any],
        noise_level: float = 0.02
    ) -> SyntheticSequence:
        """
        Generate complete gesture sequence with hand and gaze trajectories.
        
        Args:
            gesture_type: Type of gesture to generate
            context: Environmental context (objects, workspace, etc.)
            noise_level: Amount of noise to add to trajectories
            
        Returns:
            Complete synthetic sequence
        """
        # Get gesture-specific parameters
        gesture_params = self.gesture_params[gesture_type]
        base_duration = gesture_params['base_duration']
        
        # Generate start and end positions based on gesture type
        start_pos, end_pos, waypoints = self._get_gesture_positions(gesture_type, context)
        
        # Create trajectory parameters
        traj_params = TrajectoryParameters(
            start_position=start_pos,
            end_position=end_pos,
            duration=base_duration,
            noise_std=noise_level,
            intermediate_points=waypoints
        )
        
        # Generate hand trajectory
        timestamps, hand_positions, hand_velocities, hand_accelerations = \
            self.generate_minimum_jerk_trajectory(traj_params)
        
        # Generate corresponding gaze trajectory
        gaze_params = self._get_gaze_parameters(gesture_type, context, hand_positions)
        _, gaze_trajectory = self.generate_gaze_trajectory(gaze_params, base_duration)
        
        # Generate intent labels over time
        intent_labels = self._generate_intent_labels(gesture_type, len(timestamps))
        
        # Create sequence ID
        sequence_id = f"{gesture_type.value}_{np.random.randint(100000, 999999)}"
        
        return SyntheticSequence(
            sequence_id=sequence_id,
            gesture_type=gesture_type,
            hand_trajectory=hand_positions,
            gaze_trajectory=gaze_trajectory,
            timestamps=timestamps,
            intent_labels=intent_labels,
            context_info=context,
            noise_level=noise_level
        )
    
    def _get_gesture_positions(
        self,
        gesture_type: GestureType,
        context: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[List[np.ndarray]]]:
        """Get start/end positions and waypoints for specific gesture."""
        # Default start position (shoulder height, neutral)
        start_pos = np.array([
            np.random.uniform(self.x_bounds[0] + 0.1, self.x_bounds[1] - 0.1),
            np.random.uniform(self.y_bounds[0] + 0.2, self.y_bounds[1] - 0.2),
            np.random.uniform(self.z_bounds[0] + 0.8, self.z_bounds[1] - 0.2)
        ])
        
        waypoints = None
        
        if gesture_type == GestureType.WAVE:
            # Wave motion with oscillatory waypoints
            end_pos = start_pos.copy()
            amplitude = self.gesture_params[GestureType.WAVE]['amplitude']
            waypoints = []
            
            # Create oscillatory waypoints
            for i in range(1, 4):
                way_point = start_pos.copy()
                way_point[1] += amplitude * (-1)**i  # Oscillate in Y
                waypoints.append(way_point)
        
        elif gesture_type == GestureType.POINT:
            # Point toward object or location
            if 'target_object' in context:
                target_pos = np.array(context['target_object']['position'])
                direction = (target_pos - start_pos)
                direction /= np.linalg.norm(direction)
                
                extension = self.gesture_params[GestureType.POINT]['extension_ratio']
                end_pos = start_pos + extension * direction
            else:
                # Default pointing direction
                end_pos = start_pos + np.array([0.3, 0.2, -0.1])
        
        elif gesture_type == GestureType.GRAB:
            # Reach and grasp object
            if 'target_object' in context:
                end_pos = np.array(context['target_object']['position'])
                # Add slight offset for natural approach
                end_pos += np.random.normal(0, 0.02, 3)
            else:
                end_pos = np.array([
                    np.random.uniform(self.x_bounds[0] + 0.2, self.x_bounds[1] - 0.2),
                    np.random.uniform(self.y_bounds[0] + 0.1, self.y_bounds[1] - 0.1),
                    np.random.uniform(self.z_bounds[0] + 0.6, self.z_bounds[0] + 1.0)
                ])
        
        elif gesture_type == GestureType.HANDOVER:
            # Move toward robot/human for handover
            handover_zone = context.get('handover_zone', [0.0, 0.3, 0.9])
            end_pos = np.array(handover_zone)
            end_pos += np.random.normal(0, 0.05, 3)
            
            # Add waypoint for natural approach
            waypoints = [start_pos + 0.3 * (end_pos - start_pos)]
        
        elif gesture_type == GestureType.REACH:
            # Simple reaching motion
            end_pos = np.array([
                np.random.uniform(self.x_bounds[0] + 0.15, self.x_bounds[1] - 0.15),
                np.random.uniform(self.y_bounds[0] + 0.1, self.y_bounds[1] - 0.1),
                np.random.uniform(self.z_bounds[0] + 0.5, self.z_bounds[1] - 0.3)
            ])
        
        else:  # IDLE
            end_pos = start_pos + np.random.normal(0, 0.05, 3)
        
        return start_pos, end_pos, waypoints
    
    def _get_gaze_parameters(
        self,
        gesture_type: GestureType,
        context: Dict[str, Any],
        hand_positions: np.ndarray
    ) -> GazeParameters:
        """Generate gaze parameters based on gesture and context."""
        focal_points = []
        dwell_times = []
        
        # Start by looking at hand
        focal_points.append(hand_positions[0])
        dwell_times.append(0.3)
        
        if gesture_type == GestureType.POINT:
            # Look at target being pointed at
            if 'target_object' in context:
                focal_points.append(np.array(context['target_object']['position']))
                dwell_times.append(1.0)
            
            # Return gaze to hand
            focal_points.append(hand_positions[-1])
            dwell_times.append(0.5)
        
        elif gesture_type == GestureType.GRAB:
            # Track hand movement toward object
            mid_point = hand_positions[len(hand_positions) // 2]
            focal_points.append(mid_point)
            dwell_times.append(0.8)
            
            # Focus on grasp location
            focal_points.append(hand_positions[-1])
            dwell_times.append(1.0)
        
        elif gesture_type == GestureType.HANDOVER:
            # Look between hand and robot/receiver
            if 'robot_position' in context:
                robot_pos = np.array(context['robot_position'])
                focal_points.append(robot_pos)
                dwell_times.append(0.8)
            
            # Look at handover location
            focal_points.append(hand_positions[-1])
            dwell_times.append(1.5)
        
        else:
            # Default: track hand movement
            if len(hand_positions) > 10:
                focal_points.append(hand_positions[len(hand_positions) // 2])
                dwell_times.append(0.8)
            
            focal_points.append(hand_positions[-1])
            dwell_times.append(1.0)
        
        return GazeParameters(
            focal_points=focal_points,
            dwell_times=dwell_times,
            saccade_duration=0.05,
            fixation_noise=0.002,
            pursuit_gain=0.9
        )
    
    def _generate_intent_labels(
        self,
        gesture_type: GestureType,
        n_timesteps: int
    ) -> List[BehaviorType]:
        """Generate ground truth intent labels over trajectory."""
        labels = []
        
        # Map gesture types to behavior types
        gesture_to_behavior = {
            GestureType.WAVE: BehaviorType.GESTURE,
            GestureType.POINT: BehaviorType.POINTING,
            GestureType.GRAB: BehaviorType.REACHING,
            GestureType.HANDOVER: BehaviorType.HANDOVER,
            GestureType.REACH: BehaviorType.REACHING,
            GestureType.IDLE: BehaviorType.IDLE
        }
        
        main_behavior = gesture_to_behavior.get(gesture_type, BehaviorType.UNKNOWN)
        
        # Generate temporal progression of intents
        for i in range(n_timesteps):
            progress = i / n_timesteps
            
            if progress < 0.1:
                # Initial preparation phase
                labels.append(BehaviorType.IDLE)
            elif progress > 0.9:
                # Final completion phase
                labels.append(BehaviorType.IDLE if gesture_type != GestureType.HANDOVER 
                            else BehaviorType.HANDOVER)
            else:
                # Main gesture execution
                labels.append(main_behavior)
        
        return labels
    
    def generate_dataset(
        self,
        n_sequences: int = 1000,
        gesture_distribution: Optional[Dict[GestureType, float]] = None,
        noise_range: Tuple[float, float] = (0.005, 0.03),
        save_path: Optional[str] = None
    ) -> List[SyntheticSequence]:
        """
        Generate a complete synthetic dataset.
        
        Args:
            n_sequences: Number of sequences to generate
            gesture_distribution: Probability distribution over gesture types
            noise_range: Range of noise levels to apply
            save_path: Optional path to save the dataset
            
        Returns:
            List of synthetic sequences
        """
        if gesture_distribution is None:
            # Default uniform distribution
            gesture_distribution = {gesture: 1.0 / len(GestureType) 
                                  for gesture in GestureType}
        
        # Normalize distribution
        total = sum(gesture_distribution.values())
        gesture_distribution = {k: v/total for k, v in gesture_distribution.items()}
        
        sequences = []
        logger.info(f"Generating {n_sequences} synthetic sequences")
        
        for i in range(n_sequences):
            # Sample gesture type
            gesture_type = np.random.choice(
                list(gesture_distribution.keys()),
                p=list(gesture_distribution.values())
            )
            
            # Sample noise level
            noise_level = np.random.uniform(noise_range[0], noise_range[1])
            
            # Generate context
            context = self._generate_context()
            
            # Generate sequence
            try:
                sequence = self.generate_gesture_sequence(
                    gesture_type, context, noise_level
                )
                sequences.append(sequence)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Generated {i + 1}/{n_sequences} sequences")
                    
            except Exception as e:
                logger.warning(f"Failed to generate sequence {i}: {e}")
                continue
        
        logger.info(f"Successfully generated {len(sequences)} sequences")
        
        # Save dataset if path provided
        if save_path:
            self.save_dataset(sequences, save_path)
        
        return sequences
    
    def _generate_context(self) -> Dict[str, Any]:
        """Generate random context for a sequence."""
        context = {
            'workspace_bounds': self.workspace_bounds.tolist(),
            'n_objects': np.random.randint(0, 4),
            'lighting_condition': np.random.choice(['bright', 'normal', 'dim']),
            'handover_zone': [
                np.random.uniform(self.x_bounds[0], self.x_bounds[1]),
                np.random.uniform(self.y_bounds[0], self.y_bounds[1]),
                np.random.uniform(self.z_bounds[0] + 0.7, self.z_bounds[1] - 0.1)
            ]
        }
        
        # Add random objects
        if context['n_objects'] > 0:
            objects = []
            for i in range(context['n_objects']):
                obj = {
                    'id': f'object_{i}',
                    'position': [
                        np.random.uniform(self.x_bounds[0] + 0.1, self.x_bounds[1] - 0.1),
                        np.random.uniform(self.y_bounds[0] + 0.1, self.y_bounds[1] - 0.1),
                        np.random.uniform(self.z_bounds[0] + 0.5, self.z_bounds[0] + 1.2)
                    ],
                    'type': np.random.choice(['cup', 'bottle', 'tool', 'box'])
                }
                objects.append(obj)
            
            context['objects'] = objects
            
            # Sometimes make one object the target
            if np.random.random() < 0.7:
                context['target_object'] = np.random.choice(objects)
        
        # Add robot position
        context['robot_position'] = [
            np.random.uniform(self.x_bounds[0], self.x_bounds[1]),
            np.random.uniform(self.y_bounds[0] - 0.3, self.y_bounds[0]),
            np.random.uniform(self.z_bounds[0] + 0.5, self.z_bounds[1])
        ]
        
        return context
    
    def save_dataset(self, sequences: List[SyntheticSequence], save_path: str) -> None:
        """
        Save dataset to disk in multiple formats.
        
        Args:
            sequences: List of synthetic sequences
            save_path: Base path for saving (without extension)
        """
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle for Python
        with open(f"{save_path}.pkl", 'wb') as f:
            pickle.dump(sequences, f)
        
        # Save metadata as JSON
        metadata = {
            'n_sequences': len(sequences),
            'gesture_types': [seq.gesture_type.value for seq in sequences],
            'avg_duration': np.mean([seq.timestamps[-1] - seq.timestamps[0] 
                                   for seq in sequences]),
            'workspace_bounds': self.workspace_bounds.tolist(),
            'sampling_frequency': self.sampling_frequency
        }
        
        with open(f"{save_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset saved to {save_path} (pickle) and {save_path}_metadata.json")
    
    @classmethod
    def load_dataset(cls, load_path: str) -> List[SyntheticSequence]:
        """
        Load dataset from disk.
        
        Args:
            load_path: Path to dataset file (with or without .pkl extension)
            
        Returns:
            List of synthetic sequences
        """
        if not load_path.endswith('.pkl'):
            load_path += '.pkl'
        
        with open(load_path, 'rb') as f:
            sequences = pickle.load(f)
        
        logger.info(f"Loaded {len(sequences)} sequences from {load_path}")
        return sequences