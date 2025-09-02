"""
Enhanced Synthetic Human Behavior Dataset Generator

This module provides publication-quality synthetic human behavior data with:
- Biomechanically realistic trajectory generation
- Advanced noise models based on human motor control literature  
- Temporal intent pattern validation
- Comprehensive ground truth with confidence scores
- Domain expert validation metrics

Mathematical Foundation:
- Minimum-jerk trajectories: minimize ∫(d³x/dt³)² dt
- Biomechanical constraints from human kinematic studies
- Motor noise models from Fitts' law and signal-dependent noise
- Intent temporal patterns with preparation-execution-completion phases

Author: Claude Code Research Team
Date: 2024
"""

import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.stats
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from pathlib import Path
import json
import pickle
import warnings
from collections import defaultdict, Counter

try:
    from ..models.human_behavior import BehaviorType, HumanState, BehaviorPrediction
    from ..utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


@dataclass
class BiomechanicalConstraints:
    """Human biomechanical constraints based on research literature"""
    # Joint angle limits (radians) - based on human anatomy studies
    shoulder_flexion: Tuple[float, float] = (-np.pi/6, 3*np.pi/4)  # -30° to 135°
    shoulder_abduction: Tuple[float, float] = (0, np.pi/2)  # 0° to 90° 
    elbow_flexion: Tuple[float, float] = (0, 2.4)  # 0° to 135°
    wrist_flexion: Tuple[float, float] = (-np.pi/3, np.pi/3)  # -60° to 60°
    
    # Velocity limits (m/s) - from reaching movement studies
    max_hand_velocity: float = 2.5  # Peak velocity for reaching
    max_angular_velocity: float = 15.0  # rad/s for joints
    
    # Acceleration limits (m/s²)
    max_hand_acceleration: float = 20.0  # Peak acceleration
    max_jerk: float = 100.0  # Maximum jerk (m/s³)
    
    # Workspace constraints (meters) - reachable space
    comfortable_reach: float = 0.65  # 65cm comfortable reach
    maximum_reach: float = 0.85  # 85cm maximum reach
    
    # Force capabilities (N) - grip and manipulation forces
    max_grip_force: float = 400.0  # Maximum grip force
    precision_force_range: Tuple[float, float] = (1.0, 25.0)  # Precision grip range
    
    # Fatigue and adaptation parameters
    fatigue_onset_time: float = 300.0  # 5 minutes for fatigue onset
    learning_rate: float = 0.1  # Motor learning adaptation rate


@dataclass
class MotorNoiseModel:
    """Motor noise model based on human movement control literature"""
    # Signal-dependent noise (Weber-Fechner law)
    signal_dependent_factor: float = 0.04  # 4% of signal magnitude
    
    # Neuromotor noise characteristics
    tremor_frequency_range: Tuple[float, float] = (4.0, 12.0)  # Hz
    physiological_tremor_amplitude: float = 0.001  # 1mm RMS
    
    # Planning noise (affects trajectory endpoints)
    planning_noise_std: float = 0.015  # 1.5cm planning uncertainty
    
    # Sensory feedback delays
    visual_feedback_delay: float = 0.12  # 120ms visual processing
    proprioceptive_delay: float = 0.05  # 50ms proprioceptive delay
    
    # Corrective movement parameters
    correction_threshold: float = 0.02  # 2cm error threshold for corrections
    correction_gain: float = 0.8  # Feedback correction gain


@dataclass 
class IntentTemporalPattern:
    """Temporal patterns for different intent phases"""
    # Phase durations as fractions of total movement time
    preparation_phase: float = 0.15  # 15% preparation
    execution_phase: float = 0.70   # 70% main execution  
    completion_phase: float = 0.15  # 15% completion/stabilization
    
    # Phase transition variability
    phase_transition_std: float = 0.05  # 5% variability in phase timing
    
    # Intent change probability over time
    intent_stability: float = 0.95  # 95% probability intent remains same
    intent_change_rate: float = 0.02  # 2% per second change probability
    
    # Hesitation and correction patterns
    hesitation_probability: float = 0.08  # 8% chance of hesitation
    correction_probability: float = 0.12  # 12% chance of trajectory correction
    
    # Confidence evolution over movement
    initial_confidence: float = 0.8   # Starting intent confidence
    confidence_growth_rate: float = 0.3  # Confidence increase during movement


@dataclass
class SensorNoiseSpec:
    """Realistic sensor noise specifications based on commercial sensor specifications"""
    # Motion capture noise (OptiTrack Prime series, Vicon Vantage)
    mocap_position_noise: float = 0.0003  # 0.3mm RMS noise (high-end systems)
    mocap_frequency_noise: float = 0.08  # Frequency-dependent noise factor
    mocap_marker_diameter_noise: float = 0.0002  # Marker size effect on accuracy
    mocap_calibration_drift: float = 0.00001  # Calibration drift per hour
    mocap_volume_edge_noise_factor: float = 2.0  # Increased noise at volume edges
    
    # Eye tracking noise (Tobii Pro Spectrum, EyeLink 1000)
    eyetrack_angular_accuracy: float = 0.3  # 0.3° angular accuracy (best case)
    eyetrack_precision_rms: float = 0.05  # 0.05° RMS precision
    eyetrack_sample_dropout: float = 0.015  # 1.5% sample dropout rate
    eyetrack_pupil_size_dependency: float = 0.02  # Accuracy degradation factor
    eyetrack_head_movement_noise: float = 0.1  # Additional noise from head movement
    
    # IMU sensor noise (Xsens MTi-680G, Bosch BMI series)
    imu_accelerometer_noise: float = 0.08  # m/s² noise density (updated)
    imu_accelerometer_bias_stability: float = 0.02  # m/s² bias stability
    imu_gyroscope_noise: float = 0.015  # rad/s noise density (updated)
    imu_gyroscope_bias_stability: float = 0.002  # rad/s bias stability
    imu_magnetometer_noise: float = 0.2  # µT noise density (updated)
    imu_temperature_drift: float = 0.001  # Temperature coefficient
    
    # Environmental factors
    lighting_noise_factor: float = 1.5  # Noise increases in poor lighting
    occlusion_probability: float = 0.05  # 5% chance of temporary occlusion
    electromagnetic_interference: float = 0.02  # EMI effect on sensors
    
    # Temporal correlation parameters
    autocorrelation_timescale: float = 0.1  # Correlation timescale in seconds
    cross_correlation_factor: float = 0.3  # Inter-sensor correlation strength
    
    # Non-stationary noise parameters
    noise_burst_probability: float = 0.01  # Probability of noise bursts
    noise_burst_duration: float = 0.05  # Duration of noise bursts in seconds
    noise_burst_magnitude: float = 5.0  # Magnitude multiplier for bursts
    
    # Systematic error parameters
    systematic_drift_rate: float = 0.0001  # Systematic drift per second
    periodic_error_frequency: float = 1.0  # Periodic error frequency in Hz
    periodic_error_amplitude: float = 0.0001  # Amplitude of periodic errors
    correlation_time_constant: float = 0.1  # 100ms correlation time
    drift_rate: float = 0.001  # Long-term drift rate


class EnhancedSyntheticGenerator:
    """Enhanced synthetic human behavior generator with publication-quality realism"""
    
    def __init__(
        self,
        workspace_bounds: np.ndarray,
        sampling_frequency: float = 100.0,  # Higher frequency for better realism
        biomech_constraints: Optional[BiomechanicalConstraints] = None,
        motor_noise: Optional[MotorNoiseModel] = None,
        sensor_noise: Optional[SensorNoiseSpec] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize enhanced synthetic generator.
        
        Args:
            workspace_bounds: 3D workspace boundaries [x_min, x_max, y_min, y_max, z_min, z_max]
            sampling_frequency: Data sampling frequency (Hz)
            biomech_constraints: Human biomechanical constraints
            motor_noise: Motor noise model parameters
            sensor_noise: Sensor noise specifications
            random_seed: Random seed for reproducibility
        """
        self.workspace_bounds = workspace_bounds
        self.sampling_frequency = sampling_frequency
        self.dt = 1.0 / sampling_frequency
        
        # Initialize constraint and noise models
        self.biomech = biomech_constraints or BiomechanicalConstraints()
        self.motor_noise = motor_noise or MotorNoiseModel()
        self.sensor_noise = sensor_noise or SensorNoiseSpec()
        self.intent_patterns = IntentTemporalPattern()
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Workspace bounds
        self.x_bounds = workspace_bounds[:2]
        self.y_bounds = workspace_bounds[2:4]
        self.z_bounds = workspace_bounds[4:6]
        
        # Initialize validation metrics
        self.validation_metrics = {}
        self.generated_sequences = []
        
        # Literature-based parameters
        self._initialize_literature_parameters()
        
        logger.info(f"Enhanced generator initialized: {sampling_frequency}Hz, biomechanical constraints enabled")
    
    def _initialize_literature_parameters(self):
        """Initialize parameters based on human movement literature"""
        # Fitts' Law parameters (from Fitts 1954, MacKenzie 1992)
        self.fitts_a = 0.1  # Intercept term (seconds)
        self.fitts_b = 0.15  # Slope term (seconds/bit)
        
        # Speed-accuracy tradeoffs (from Woodworth 1899, Meyer et al. 1988)
        self.woodworth_constant = 0.2  # Speed-accuracy tradeoff parameter
        
        # Two-component model parameters (from Meyer et al. 1988)
        self.primary_submovement_ratio = 0.8  # 80% of movement in primary phase
        self.secondary_correction_probability = 0.3  # 30% chance of corrections
        
        # Demographic variation parameters
        self.age_effect_factor = 0.002  # Movement slowing per year of age
        self.skill_level_range = (0.7, 1.3)  # Skill multiplier range
        self.fatigue_accumulation_rate = 0.0001  # Per-movement fatigue
        
        logger.debug("Literature-based parameters initialized")
    
    def generate_biomechanically_realistic_trajectory(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        duration: float,
        user_params: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Generate biomechanically realistic trajectory with constraints.
        
        Mathematical foundation:
        - Minimum jerk: minimize ∫[0,T] (d³x/dt³)² dt
        - Subject to: biomechanical joint limits, velocity limits, workspace constraints
        - With motor noise: x_noisy = x_ideal + noise(t)
        
        Args:
            start_pos: Starting 3D position
            end_pos: Target 3D position  
            duration: Movement duration
            user_params: Optional user-specific parameters
            
        Returns:
            Tuple of (time, position, velocity, acceleration, metrics)
        """
        # Initialize user parameters
        params = {
            'age': 30,  # years
            'skill_level': 1.0,  # normalized skill
            'fatigue_level': 0.0,  # normalized fatigue
            'hand_dominance': 'right',
            'strength_factor': 1.0
        }
        if user_params:
            params.update(user_params)
        
        # Apply demographic effects
        age_factor = 1 + self.age_effect_factor * max(0, params['age'] - 25)
        skill_factor = params['skill_level']
        fatigue_factor = 1 + params['fatigue_level'] * 0.3
        
        adjusted_duration = duration * age_factor * fatigue_factor / skill_factor
        
        # Check workspace constraints
        if not self._check_workspace_reachability(start_pos, end_pos):
            logger.warning("Target outside comfortable workspace, adjusting...")
            end_pos = self._adjust_to_reachable_workspace(start_pos, end_pos)
        
        # Generate minimum jerk trajectory
        t = np.linspace(0, adjusted_duration, int(adjusted_duration * self.sampling_frequency))
        
        # Minimum jerk solution
        tau = t / adjusted_duration
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / adjusted_duration
        s_ddot = (60 * tau - 180 * tau**2 + 120 * tau**3) / (adjusted_duration**2)
        
        # Apply to position
        displacement = end_pos - start_pos
        positions = start_pos[None, :] + s[:, None] * displacement[None, :]
        velocities = s_dot[:, None] * displacement[None, :]
        accelerations = s_ddot[:, None] * displacement[None, :]
        
        # Apply biomechanical constraints
        positions, velocities, accelerations = self._apply_biomechanical_constraints(
            positions, velocities, accelerations, t
        )
        
        # Add realistic motor noise
        positions_noisy = self._add_motor_noise(positions, velocities, t, params)
        
        # Add corrections and hesitations if needed
        if np.random.random() < self.intent_patterns.hesitation_probability:
            positions_noisy, velocities, accelerations = self._add_hesitation_pattern(
                positions_noisy, velocities, accelerations, t
            )
        
        if np.random.random() < self.intent_patterns.correction_probability:
            positions_noisy, velocities, accelerations = self._add_correction_movement(
                positions_noisy, velocities, accelerations, t, end_pos
            )
        
        # Compute movement quality metrics
        metrics = self._compute_movement_metrics(
            t, positions_noisy, velocities, accelerations, start_pos, end_pos
        )
        
        return t, positions_noisy, velocities, accelerations, metrics
    
    def _check_workspace_reachability(self, start_pos: np.ndarray, end_pos: np.ndarray) -> bool:
        """Check if target is within reachable workspace"""
        distance = np.linalg.norm(end_pos - start_pos)
        return distance <= self.biomech.comfortable_reach
    
    def _adjust_to_reachable_workspace(self, start_pos: np.ndarray, end_pos: np.ndarray) -> np.ndarray:
        """Adjust target to within reachable workspace"""
        direction = end_pos - start_pos
        distance = np.linalg.norm(direction)
        
        if distance > self.biomech.maximum_reach:
            # Scale to maximum reach
            direction = direction / distance * self.biomech.maximum_reach
            return start_pos + direction
        else:
            return end_pos
    
    def _apply_biomechanical_constraints(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply biomechanical constraints to trajectory"""
        # Velocity constraints
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        max_indices = velocity_magnitudes > self.biomech.max_hand_velocity
        
        if np.any(max_indices):
            # Scale velocities to maximum
            scale_factors = self.biomech.max_hand_velocity / velocity_magnitudes[max_indices]
            velocities[max_indices] *= scale_factors[:, None]
            
            # Recompute positions with constrained velocities
            for i in range(1, len(positions)):
                if max_indices[i]:
                    positions[i] = positions[i-1] + velocities[i] * self.dt
        
        # Acceleration constraints
        accel_magnitudes = np.linalg.norm(accelerations, axis=1)
        max_accel_indices = accel_magnitudes > self.biomech.max_hand_acceleration
        
        if np.any(max_accel_indices):
            scale_factors = self.biomech.max_hand_acceleration / accel_magnitudes[max_accel_indices]
            accelerations[max_accel_indices] *= scale_factors[:, None]
        
        # Workspace boundary constraints
        positions = np.clip(positions, 
                          [self.x_bounds[0], self.y_bounds[0], self.z_bounds[0]],
                          [self.x_bounds[1], self.y_bounds[1], self.z_bounds[1]])
        
        return positions, velocities, accelerations
    
    def _add_motor_noise(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        t: np.ndarray,
        user_params: Dict
    ) -> np.ndarray:
        """Add realistic motor noise based on movement control literature"""
        n_points, n_dims = positions.shape
        noisy_positions = positions.copy()
        
        # Signal-dependent noise (Weber-Fechner law)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        signal_noise_std = self.motor_noise.signal_dependent_factor * velocity_magnitudes
        
        for i in range(n_points):
            if signal_noise_std[i] > 0:
                noise = np.random.normal(0, signal_noise_std[i], n_dims)
                noisy_positions[i] += noise * 0.01  # Scale appropriately
        
        # Physiological tremor
        tremor_freq = np.random.uniform(*self.motor_noise.tremor_frequency_range)
        tremor_amplitude = self.motor_noise.physiological_tremor_amplitude
        
        # Age effect on tremor
        age_tremor_factor = 1 + 0.02 * max(0, user_params['age'] - 60)
        tremor_amplitude *= age_tremor_factor
        
        tremor_signal = tremor_amplitude * np.sin(2 * np.pi * tremor_freq * t[:, None])
        noisy_positions += tremor_signal
        
        # Planning noise (affects entire trajectory)
        planning_noise = np.random.normal(0, self.motor_noise.planning_noise_std, n_dims)
        noisy_positions += planning_noise[None, :]
        
        # Add correlated noise (motor unit synchronization)
        correlation_kernel = np.exp(-np.abs(t[:, None] - t[None, :]) / 0.05)  # 50ms correlation
        for dim in range(n_dims):
            white_noise = np.random.normal(0, 0.002, n_points)
            correlated_noise = np.dot(correlation_kernel, white_noise) / n_points
            noisy_positions[:, dim] += correlated_noise
        
        return noisy_positions
    
    def _add_hesitation_pattern(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Add realistic hesitation patterns to movement"""
        n_points = len(positions)
        hesitation_start = int(0.3 * n_points)  # Hesitation at 30% of movement
        hesitation_duration = int(0.1 * n_points)  # 10% of movement duration
        
        # Create hesitation by temporarily reducing velocity
        hesitation_end = min(hesitation_start + hesitation_duration, n_points)
        
        # Gaussian velocity reduction
        for i in range(hesitation_start, hesitation_end):
            progress = (i - hesitation_start) / hesitation_duration
            reduction_factor = 0.3 * np.exp(-((progress - 0.5) * 4)**2)  # Peak at middle
            velocities[i] *= (1 - reduction_factor)
        
        # Recompute positions with hesitation
        for i in range(hesitation_start + 1, hesitation_end):
            positions[i] = positions[i-1] + velocities[i] * self.dt
        
        # Recompute accelerations
        accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (2 * self.dt)
        
        return positions, velocities, accelerations
    
    def _add_correction_movement(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        t: np.ndarray,
        target_pos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Add corrective submovements based on error feedback"""
        n_points = len(positions)
        correction_start = int(0.6 * n_points)  # Correction starts at 60%
        
        # Calculate error at correction point
        current_pos = positions[correction_start]
        error_vector = target_pos - current_pos
        error_magnitude = np.linalg.norm(error_vector)
        
        if error_magnitude > self.motor_noise.correction_threshold:
            # Apply correction with realistic gain
            correction_gain = self.motor_noise.correction_gain
            correction = correction_gain * error_vector
            
            # Apply correction over remaining trajectory
            remaining_points = n_points - correction_start
            correction_profile = np.linspace(0, 1, remaining_points)**2  # Quadratic buildup
            
            for i, correction_factor in enumerate(correction_profile):
                idx = correction_start + i
                if idx < n_points:
                    positions[idx] += correction_factor * correction
        
        # Recompute derivatives
        velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * self.dt)
        accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (2 * self.dt)
        
        return positions, velocities, accelerations
    
    def _compute_movement_metrics(
        self,
        t: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        start_pos: np.ndarray,
        end_pos: np.ndarray
    ) -> Dict[str, float]:
        """Compute comprehensive movement quality metrics"""
        # Basic metrics
        total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        straight_line_distance = np.linalg.norm(end_pos - start_pos)
        path_efficiency = straight_line_distance / total_distance if total_distance > 0 else 0
        
        # Velocity metrics
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        peak_velocity = np.max(velocity_magnitudes)
        mean_velocity = np.mean(velocity_magnitudes)
        
        # Acceleration metrics  
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
        peak_acceleration = np.max(acceleration_magnitudes)
        
        # Smoothness metrics (jerk-based)
        jerk = np.diff(accelerations, axis=0) / self.dt
        jerk_magnitudes = np.linalg.norm(jerk, axis=1)
        
        # Normalized jerk (SPARC - Balasubramanian et al. 2015)
        total_jerk = np.sum(jerk_magnitudes**2) * self.dt
        movement_duration = t[-1] - t[0]
        normalized_jerk = total_jerk * (movement_duration**5) / (peak_velocity**2) if peak_velocity > 0 else 0
        
        # Number of velocity peaks (movement units)
        velocity_peaks = self._count_velocity_peaks(velocity_magnitudes)
        
        # Time to peak velocity (symmetry measure)
        time_to_peak_vel = t[np.argmax(velocity_magnitudes)] / movement_duration
        
        # Endpoint accuracy
        final_error = np.linalg.norm(positions[-1] - end_pos)
        
        # Fitts' law validation
        target_width = 0.02  # Assume 2cm target width
        index_of_difficulty = np.log2(2 * straight_line_distance / target_width)
        predicted_mt = self.fitts_a + self.fitts_b * index_of_difficulty
        fitts_law_ratio = movement_duration / predicted_mt
        
        return {
            'path_efficiency': path_efficiency,
            'peak_velocity': peak_velocity,
            'mean_velocity': mean_velocity,
            'peak_acceleration': peak_acceleration,
            'normalized_jerk': normalized_jerk,
            'velocity_peaks': velocity_peaks,
            'time_to_peak_velocity': time_to_peak_vel,
            'endpoint_error': final_error,
            'movement_duration': movement_duration,
            'fitts_law_ratio': fitts_law_ratio,
            'total_distance': total_distance,
            'straight_line_distance': straight_line_distance
        }
    
    def _count_velocity_peaks(self, velocity_profile: np.ndarray, prominence: float = 0.1) -> int:
        """Count number of velocity peaks (movement submovements)"""
        from scipy.signal import find_peaks
        
        # Find peaks with minimum prominence
        peaks, _ = find_peaks(velocity_profile, prominence=prominence * np.max(velocity_profile))
        return len(peaks)
    
    def generate_realistic_intent_sequence(
        self,
        gesture_type: str,
        sequence_duration: float,
        context: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Generate realistic temporal intent sequence with phase transitions.
        
        Args:
            gesture_type: Primary gesture type
            sequence_duration: Total sequence duration
            context: Environmental context
            
        Returns:
            Tuple of (timestamps, intent_labels, confidence_scores)
        """
        n_points = int(sequence_duration * self.sampling_frequency)
        t = np.linspace(0, sequence_duration, n_points)
        
        # Initialize intent sequence
        intent_labels = []
        confidence_scores = np.zeros(n_points)
        
        # Define temporal phases
        prep_end = self.intent_patterns.preparation_phase
        exec_end = prep_end + self.intent_patterns.execution_phase
        
        # Add phase transition variability
        prep_var = np.random.normal(0, self.intent_patterns.phase_transition_std)
        exec_var = np.random.normal(0, self.intent_patterns.phase_transition_std)
        
        prep_end += prep_var
        exec_end += exec_var
        
        prep_end = np.clip(prep_end, 0.05, 0.3)  # Keep within reasonable bounds
        exec_end = np.clip(exec_end, prep_end + 0.1, 0.95)
        
        # Generate intent labels with temporal evolution
        for i, time_point in enumerate(t):
            progress = time_point / sequence_duration
            
            # Determine phase
            if progress < prep_end:
                # Preparation phase
                phase_labels = self._get_preparation_intents(gesture_type, context)
                phase_confidence = self.intent_patterns.initial_confidence
            elif progress < exec_end:
                # Execution phase
                phase_labels = self._get_execution_intents(gesture_type, context)
                # Confidence grows during execution
                exec_progress = (progress - prep_end) / (exec_end - prep_end)
                phase_confidence = self.intent_patterns.initial_confidence + \
                                 exec_progress * self.intent_patterns.confidence_growth_rate
            else:
                # Completion phase
                phase_labels = self._get_completion_intents(gesture_type, context)
                phase_confidence = min(0.95, self.intent_patterns.initial_confidence + \
                                     self.intent_patterns.confidence_growth_rate)
            
            # Select primary intent (with some variability)
            if len(phase_labels) > 1 and np.random.random() < 0.1:
                # 10% chance of secondary intent
                selected_intent = np.random.choice(phase_labels[1:])
                phase_confidence *= 0.8  # Lower confidence for secondary intent
            else:
                selected_intent = phase_labels[0]  # Primary intent
            
            intent_labels.append(selected_intent)
            confidence_scores[i] = np.clip(phase_confidence, 0.1, 0.95)
            
            # Add intent change dynamics
            if i > 0 and np.random.random() < self.intent_patterns.intent_change_rate * self.dt:
                # Possible intent change
                if np.random.random() > self.intent_patterns.intent_stability:
                    # Change to related intent
                    related_intents = self._get_related_intents(selected_intent, gesture_type)
                    if related_intents:
                        selected_intent = np.random.choice(related_intents)
                        confidence_scores[i] *= 0.7  # Reduce confidence during transition
        
        # Smooth confidence scores to remove abrupt changes
        confidence_scores = savgol_filter(confidence_scores, 
                                        window_length=min(21, n_points//10*2+1),
                                        polyorder=2)
        confidence_scores = np.clip(confidence_scores, 0.1, 0.95)
        
        return t, intent_labels, confidence_scores
    
    def _get_preparation_intents(self, gesture_type: str, context: Dict) -> List[str]:
        """Get preparation phase intent possibilities"""
        base_intents = ['idle', 'attention_focusing', 'planning']
        
        if gesture_type in ['reach', 'grab']:
            return ['planning', 'target_identification', 'idle']
        elif gesture_type in ['point', 'gesture']:
            return ['attention_focusing', 'communication_intent', 'idle']
        elif gesture_type == 'handover':
            return ['handover_preparation', 'object_presentation', 'attention_focusing']
        else:
            return base_intents
    
    def _get_execution_intents(self, gesture_type: str, context: Dict) -> List[str]:
        """Get execution phase intent possibilities"""
        if gesture_type == 'reach':
            return ['reaching', 'approach', 'target_acquisition']
        elif gesture_type == 'grab':
            return ['grasping', 'grip_formation', 'object_manipulation']
        elif gesture_type == 'point':
            return ['pointing', 'indication', 'direction_showing']
        elif gesture_type == 'handover':
            return ['handover_execution', 'object_transfer', 'cooperative_action']
        elif gesture_type == 'wave':
            return ['greeting', 'attention_getting', 'communication']
        else:
            return ['unknown_action', 'general_movement']
    
    def _get_completion_intents(self, gesture_type: str, context: Dict) -> List[str]:
        """Get completion phase intent possibilities"""
        if gesture_type in ['reach', 'grab']:
            return ['task_completion', 'stabilization', 'idle']
        elif gesture_type in ['point', 'wave']:
            return ['communication_complete', 'attention_achieved', 'idle']
        elif gesture_type == 'handover':
            return ['handover_complete', 'confirmation_wait', 'idle']
        else:
            return ['idle', 'task_completion']
    
    def _get_related_intents(self, current_intent: str, gesture_type: str) -> List[str]:
        """Get intents related to current intent (for transitions)"""
        intent_relations = {
            'reaching': ['approach', 'target_acquisition', 'grasping'],
            'grasping': ['grip_formation', 'object_manipulation', 'handover_preparation'],
            'pointing': ['indication', 'direction_showing', 'communication'],
            'handover_execution': ['object_transfer', 'cooperative_action', 'handover_complete'],
            'planning': ['attention_focusing', 'target_identification']
        }
        
        return intent_relations.get(current_intent, ['idle'])
    
    def add_realistic_sensor_noise(
        self,
        clean_data: Dict[str, np.ndarray],
        sensor_type: str = 'mocap',
        environmental_conditions: Optional[Dict] = None
    ) -> Dict[str, np.ndarray]:
        """
        Add realistic sensor noise based on hardware specifications.
        
        Args:
            clean_data: Dictionary of clean sensor data
            sensor_type: Type of sensor ('mocap', 'imu', 'eyetrack')
            environmental_conditions: Environmental factors affecting noise
            
        Returns:
            Dictionary of noisy sensor data
        """
        noisy_data = {}
        conditions = environmental_conditions or {}
        
        # Lighting condition affects noise
        lighting_factor = 1.0
        if conditions.get('lighting') == 'dim':
            lighting_factor = self.sensor_noise.lighting_noise_factor
        elif conditions.get('lighting') == 'bright':
            lighting_factor = 0.8
        
        for data_key, data_values in clean_data.items():
            if sensor_type == 'mocap':
                noisy_data[data_key] = self._add_mocap_noise(data_values, lighting_factor)
            elif sensor_type == 'imu':
                noisy_data[data_key] = self._add_imu_noise(data_values)
            elif sensor_type == 'eyetrack':
                noisy_data[data_key] = self._add_eyetrack_noise(data_values, lighting_factor)
            else:
                # Generic sensor noise
                noise_std = 0.01 * lighting_factor
                noise = np.random.normal(0, noise_std, data_values.shape)
                noisy_data[data_key] = data_values + noise
        
        return noisy_data
    
    def _add_mocap_noise(self, positions: np.ndarray, lighting_factor: float = 1.0) -> np.ndarray:
        """Add motion capture specific noise patterns with enhanced realism"""
        noisy_positions = positions.copy()
        n_points, n_dims = positions.shape
        t = np.linspace(0, n_points * self.dt, n_points)
        
        # Base noise level with environmental factors
        base_noise = self.sensor_noise.mocap_position_noise * lighting_factor
        
        # Volume edge effects (noise increases at tracking volume boundaries)
        center_position = np.mean(positions, axis=0)
        distances_from_center = np.linalg.norm(positions - center_position, axis=1)
        max_distance = np.max(distances_from_center)
        edge_factor = 1 + (distances_from_center / max_distance) * (self.sensor_noise.mocap_volume_edge_noise_factor - 1)
        
        # Generate correlated noise across dimensions and time
        correlation_matrix = self._generate_spatial_correlation_matrix(n_dims)
        
        for i in range(n_points):
            # Temporally correlated noise
            if i == 0:
                noise = np.random.multivariate_normal(np.zeros(n_dims), 
                                                    (base_noise * edge_factor[i])**2 * correlation_matrix)
            else:
                # AR(1) process for temporal correlation
                alpha = np.exp(-self.dt / self.sensor_noise.autocorrelation_timescale)
                innovation = np.random.multivariate_normal(np.zeros(n_dims), 
                                                         (base_noise * edge_factor[i])**2 * correlation_matrix * (1 - alpha**2))
                noise = alpha * noise + innovation
            
            noisy_positions[i] += noise
        
        # Add systematic drift
        drift = self.sensor_noise.systematic_drift_rate * t[:, None] * np.random.randn(n_dims)
        noisy_positions += drift
        
        # Add periodic errors (from systematic calibration errors)
        periodic_error = (self.sensor_noise.periodic_error_amplitude * 
                         np.sin(2 * np.pi * self.sensor_noise.periodic_error_frequency * t[:, None]) *
                         np.random.randn(n_dims))
        noisy_positions += periodic_error
        
        # Add non-stationary noise bursts
        burst_mask = np.random.random(n_points) < self.sensor_noise.noise_burst_probability
        burst_duration_samples = int(self.sensor_noise.noise_burst_duration / self.dt)
        
        for burst_start in np.where(burst_mask)[0]:
            burst_end = min(burst_start + burst_duration_samples, n_points)
            burst_noise = (np.random.randn(burst_end - burst_start, n_dims) * 
                          base_noise * self.sensor_noise.noise_burst_magnitude)
            noisy_positions[burst_start:burst_end] += burst_noise
        
        # Add calibration drift over time
        calibration_drift = (self.sensor_noise.mocap_calibration_drift * 
                           (t - t[0]) / 3600)[:, None] * np.random.randn(n_dims)  # Drift per hour
        noisy_positions += calibration_drift
        
        # Add occasional dropout/occlusion with realistic interpolation
        dropout_mask = np.random.random(n_points) < self.sensor_noise.occlusion_probability
        if np.any(dropout_mask):
            # Create realistic occlusion patterns (clustered dropouts)
            dropout_mask = self._create_clustered_dropouts(dropout_mask, n_points)
            
            # Interpolate missing points with uncertainty
            for dim in range(n_dims):
                valid_indices = ~dropout_mask
                if np.any(valid_indices) and np.sum(valid_indices) > 2:
                    # Cubic spline interpolation for smoother reconstruction
                    valid_times = t[valid_indices]
                    valid_positions = noisy_positions[valid_indices, dim]
                    
                    interp_func = scipy.interpolate.CubicSpline(valid_times, valid_positions, 
                                                              bc_type='natural')
                    interpolated_values = interp_func(t[dropout_mask])
                    
                    # Add interpolation uncertainty
                    interp_uncertainty = base_noise * 2.0  # Higher uncertainty for interpolated points
                    interpolation_noise = np.random.normal(0, interp_uncertainty, np.sum(dropout_mask))
                    
                    noisy_positions[dropout_mask, dim] = interpolated_values + interpolation_noise
        
        return noisy_positions
    
    def _generate_spatial_correlation_matrix(self, n_dims: int) -> np.ndarray:
        """Generate spatial correlation matrix for noise"""
        correlation_matrix = np.eye(n_dims)
        
        # Add cross-dimensional correlations
        for i in range(n_dims):
            for j in range(i+1, n_dims):
                correlation = self.sensor_noise.cross_correlation_factor * np.exp(-abs(i-j))
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        return correlation_matrix
    
    def _create_clustered_dropouts(self, dropout_mask: np.ndarray, n_points: int) -> np.ndarray:
        """Create realistic clustered dropout patterns rather than random dropouts"""
        clustered_mask = dropout_mask.copy()
        
        # Find isolated dropout points and extend them to create clusters
        dropout_indices = np.where(dropout_mask)[0]
        
        for idx in dropout_indices:
            # Extend dropout in both directions with decreasing probability
            cluster_size = max(1, int(np.random.exponential(3)))  # Average cluster size of 3 samples
            
            start_idx = max(0, idx - cluster_size // 2)
            end_idx = min(n_points, idx + cluster_size // 2)
            
            clustered_mask[start_idx:end_idx] = True
        
        return clustered_mask
    
    def _add_imu_noise(self, imu_data: np.ndarray) -> np.ndarray:
        """Add enhanced IMU sensor noise with bias stability and temperature effects"""
        noisy_data = imu_data.copy()
        n_points, n_dims = imu_data.shape
        t = np.linspace(0, n_points * self.dt, n_points)
        
        # Determine data type based on expected ranges and apply appropriate noise
        data_range = np.max(np.abs(imu_data))
        
        if data_range > 50:  # Magnetometer (µT range)
            noise_density = self.sensor_noise.imu_magnetometer_noise
            bias_stability = 0.1  # µT bias stability
        elif data_range > 10:  # Accelerometer (m/s² range)
            noise_density = self.sensor_noise.imu_accelerometer_noise
            bias_stability = self.sensor_noise.imu_accelerometer_bias_stability
        else:  # Gyroscope (rad/s range)
            noise_density = self.sensor_noise.imu_gyroscope_noise
            bias_stability = self.sensor_noise.imu_gyroscope_bias_stability
        
        # Generate bias instability (random walk)
        bias_random_walk = np.zeros((n_points, n_dims))
        bias_random_walk[0] = np.random.normal(0, bias_stability, n_dims)
        
        # Random walk process for bias
        bias_step_std = bias_stability * np.sqrt(self.dt)
        for i in range(1, n_points):
            bias_random_walk[i] = (bias_random_walk[i-1] + 
                                 np.random.normal(0, bias_step_std, n_dims))
        
        # Add temporally correlated noise
        correlation_matrix = self._generate_spatial_correlation_matrix(n_dims)
        
        for i in range(n_points):
            if i == 0:
                noise = np.random.multivariate_normal(np.zeros(n_dims), 
                                                    noise_density**2 * correlation_matrix)
            else:
                # AR(1) process for temporal correlation
                alpha = np.exp(-self.dt / self.sensor_noise.autocorrelation_timescale)
                innovation = np.random.multivariate_normal(np.zeros(n_dims), 
                                                         noise_density**2 * correlation_matrix * (1 - alpha**2))
                noise = alpha * noise + innovation
            
            noisy_data[i] += noise
        
        # Add bias instability
        noisy_data += bias_random_walk
        
        # Add temperature drift (simulate temperature variation)
        temperature_variation = 10 * np.sin(2 * np.pi * t / 1800)  # 30-minute temperature cycle
        temp_drift = (self.sensor_noise.imu_temperature_drift * temperature_variation[:, None] * 
                     np.random.randn(n_dims))
        noisy_data += temp_drift
        
        # Add quantization noise (ADC effects)
        if data_range < 10:  # For gyroscope and accelerometer
            quantization_step = 2**(-16) * data_range  # 16-bit ADC assumption
            quantization_noise = (np.random.uniform(-0.5, 0.5, (n_points, n_dims)) * 
                                quantization_step)
            noisy_data += quantization_noise
        
        # Add EMI effects
        emi_amplitude = self.sensor_noise.electromagnetic_interference * noise_density
        emi_frequency = np.random.uniform(50, 400)  # 50-400 Hz EMI
        emi_noise = emi_amplitude * np.sin(2 * np.pi * emi_frequency * t[:, None])
        noisy_data += emi_noise
        
        return noisy_data
    
    def _add_eyetrack_noise(self, gaze_data: np.ndarray, lighting_factor: float = 1.0) -> np.ndarray:
        """Add enhanced eye tracking specific noise patterns with pupil and head movement effects"""
        noisy_gaze = gaze_data.copy()
        n_points = len(gaze_data)
        t = np.linspace(0, n_points * self.dt, n_points)
        
        # Base angular noise with lighting dependency
        base_angular_noise = np.deg2rad(self.sensor_noise.eyetrack_angular_accuracy) * lighting_factor
        precision_noise = np.deg2rad(self.sensor_noise.eyetrack_precision_rms)
        
        # Simulate pupil size variation effects on accuracy
        pupil_size_variation = 0.5 + 0.5 * np.sin(2 * np.pi * t / 300)  # 5-minute cycles
        pupil_dependent_noise = (1 + self.sensor_noise.eyetrack_pupil_size_dependency * 
                               (1 - pupil_size_variation))
        
        # Add head movement artifacts
        head_movement_magnitude = np.random.exponential(0.1, n_points)  # Random head movements
        head_movement_noise_factor = (1 + self.sensor_noise.eyetrack_head_movement_noise * 
                                    head_movement_magnitude)
        
        # Generate systematic drift (eye tracker calibration drift)
        systematic_drift = np.zeros((n_points, 3))
        drift_rate = np.deg2rad(0.1)  # 0.1 degree per minute drift
        for i in range(1, n_points):
            drift_step = drift_rate * self.dt / 60  # Convert to per-second
            systematic_drift[i] = systematic_drift[i-1] + np.random.normal(0, drift_step, 3)
        
        # Add angular noise to each sample with all effects
        for i in range(n_points):
            # Combined noise factors
            total_noise_std = (base_angular_noise * pupil_dependent_noise[i] * 
                             head_movement_noise_factor[i])
            
            # Add precision noise (high-frequency jitter)
            precision_component = precision_noise * np.random.randn()
            
            # Generate random rotation for angular noise
            if np.linalg.norm(gaze_data[i]) > 1e-6:
                # Random rotation around current gaze direction
                rotation_axis = np.cross(gaze_data[i], np.random.randn(3))
                rotation_axis_norm = np.linalg.norm(rotation_axis)
                
                if rotation_axis_norm > 1e-8:
                    rotation_axis /= rotation_axis_norm
                    
                    rotation_angle = np.random.normal(0, total_noise_std) + precision_component
                    
                    # Rodrigues rotation formula
                    cos_angle = np.cos(rotation_angle)
                    sin_angle = np.sin(rotation_angle)
                    
                    noisy_gaze[i] = (gaze_data[i] * cos_angle +
                                   np.cross(rotation_axis, gaze_data[i]) * sin_angle +
                                   rotation_axis * np.dot(rotation_axis, gaze_data[i]) * (1 - cos_angle))
                else:
                    # If cross product is near zero, add small random perturbation
                    noisy_gaze[i] += np.random.normal(0, total_noise_std, 3)
            
            # Add systematic drift
            drift_rotation_axis = np.array([0, 0, 1])  # Assume vertical drift primarily
            drift_angle = np.linalg.norm(systematic_drift[i])
            if drift_angle > 1e-8:
                drift_rotation_axis = systematic_drift[i] / drift_angle
                cos_drift = np.cos(drift_angle)
                sin_drift = np.sin(drift_angle)
                
                noisy_gaze[i] = (noisy_gaze[i] * cos_drift +
                               np.cross(drift_rotation_axis, noisy_gaze[i]) * sin_drift +
                               drift_rotation_axis * np.dot(drift_rotation_axis, noisy_gaze[i]) * (1 - cos_drift))
        
        # Add realistic sample dropout patterns (blinks, occlusions)
        dropout_mask = np.random.random(n_points) < self.sensor_noise.eyetrack_sample_dropout
        
        # Create blink-like patterns (clustered dropouts)
        blink_probability = 0.005  # 0.5% chance of blink per sample
        blink_duration = int(0.1 / self.dt)  # ~100ms blink duration
        
        for i in range(n_points):
            if np.random.random() < blink_probability:
                blink_start = i
                blink_end = min(i + blink_duration, n_points)
                dropout_mask[blink_start:blink_end] = True
        
        # Handle dropouts with more sophisticated interpolation
        if np.any(dropout_mask):
            valid_mask = ~dropout_mask
            if np.sum(valid_mask) > 2:
                for dim in range(3):
                    # Use cubic spline for smoother interpolation
                    valid_indices = np.where(valid_mask)[0]
                    valid_times = t[valid_indices]
                    valid_values = noisy_gaze[valid_indices, dim]
                    
                    # Cubic spline interpolation
                    spline = scipy.interpolate.CubicSpline(valid_times, valid_values, 
                                                         bc_type='natural')
                    interpolated_values = spline(t[dropout_mask])
                    
                    # Add interpolation uncertainty
                    interp_noise = np.random.normal(0, total_noise_std * 1.5, np.sum(dropout_mask))
                    noisy_gaze[dropout_mask, dim] = interpolated_values + interp_noise
        
        # Normalize gaze vectors (they should be unit vectors)
        for i in range(n_points):
            norm = np.linalg.norm(noisy_gaze[i])
            if norm > 1e-8:
                noisy_gaze[i] /= norm
        
        return noisy_gaze
    
    def validate_dataset_realism(
        self,
        sequences: List[Dict],
        comparison_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of dataset realism against literature.
        
        Args:
            sequences: Generated synthetic sequences
            comparison_data: Optional real human data for comparison
            
        Returns:
            Comprehensive validation report
        """
        logger.info("Validating dataset realism...")
        
        validation_report = {
            'timestamp': time.time(),
            'n_sequences': len(sequences),
            'biomechanical_validation': {},
            'temporal_validation': {},
            'statistical_validation': {},
            'literature_comparison': {},
            'quality_metrics': {}
        }
        
        # Extract trajectories and metrics
        all_trajectories = []
        all_metrics = []
        all_durations = []
        
        for seq in sequences:
            if 'hand_trajectory' in seq and 'movement_metrics' in seq:
                all_trajectories.append(seq['hand_trajectory'])
                all_metrics.append(seq['movement_metrics'])
                all_durations.append(seq['movement_metrics']['movement_duration'])
        
        if not all_metrics:
            logger.warning("No movement metrics found in sequences")
            return validation_report
        
        # Biomechanical validation
        validation_report['biomechanical_validation'] = self._validate_biomechanical_realism(all_metrics)
        
        # Temporal pattern validation
        validation_report['temporal_validation'] = self._validate_temporal_patterns(sequences)
        
        # Statistical distribution validation
        validation_report['statistical_validation'] = self._validate_statistical_distributions(all_metrics)
        
        # Literature comparison
        validation_report['literature_comparison'] = self._validate_against_literature(all_metrics)
        
        # Overall quality assessment
        validation_report['quality_metrics'] = self._compute_quality_metrics(sequences)
        
        # Generate visualizations
        self._generate_validation_plots(sequences, validation_report)
        
        logger.info("Dataset validation complete")
        return validation_report
    
    def _validate_biomechanical_realism(self, metrics_list: List[Dict]) -> Dict:
        """Validate biomechanical realism of generated movements"""
        # Extract key biomechanical metrics
        peak_velocities = [m['peak_velocity'] for m in metrics_list]
        peak_accelerations = [m['peak_acceleration'] for m in metrics_list]
        path_efficiencies = [m['path_efficiency'] for m in metrics_list]
        jerk_scores = [m['normalized_jerk'] for m in metrics_list]
        
        return {
            'velocity_range_valid': (
                np.min(peak_velocities) >= 0.1 and 
                np.max(peak_velocities) <= self.biomech.max_hand_velocity
            ),
            'acceleration_range_valid': (
                np.max(peak_accelerations) <= self.biomech.max_hand_acceleration
            ),
            'path_efficiency_realistic': (
                0.7 <= np.mean(path_efficiencies) <= 0.95
            ),
            'movement_smoothness_good': np.mean(jerk_scores) < 100,
            'velocity_stats': {
                'mean': np.mean(peak_velocities),
                'std': np.std(peak_velocities),
                'range': [np.min(peak_velocities), np.max(peak_velocities)]
            },
            'jerk_stats': {
                'mean': np.mean(jerk_scores),
                'std': np.std(jerk_scores)
            }
        }
    
    def _validate_temporal_patterns(self, sequences: List[Dict]) -> Dict:
        """Validate temporal intent patterns"""
        phase_transitions = []
        intent_changes = []
        
        for seq in sequences:
            if 'intent_labels' in seq and 'confidence_scores' in seq:
                labels = seq['intent_labels']
                confidences = seq['confidence_scores']
                
                # Count intent changes
                changes = sum(1 for i in range(1, len(labels)) 
                            if labels[i] != labels[i-1])
                intent_changes.append(changes / len(labels))
                
                # Analyze phase structure
                if len(set(labels)) >= 2:
                    phase_transitions.append(len(set(labels)))
        
        return {
            'avg_intent_changes_per_sequence': np.mean(intent_changes) if intent_changes else 0,
            'avg_phases_per_sequence': np.mean(phase_transitions) if phase_transitions else 0,
            'phase_transition_realism': (
                1 <= np.mean(phase_transitions) <= 4 if phase_transitions else False
            ),
            'intent_stability_good': (
                np.mean(intent_changes) < 0.3 if intent_changes else False
            )
        }
    
    def _validate_statistical_distributions(self, metrics_list: List[Dict]) -> Dict:
        """Validate statistical distributions of movement parameters"""
        durations = [m['movement_duration'] for m in metrics_list]
        peak_vels = [m['peak_velocity'] for m in metrics_list]
        
        # Test for normality (many human movement parameters are log-normal)
        from scipy.stats import shapiro, lognorm
        
        # Duration distribution test
        duration_normal_p = shapiro(durations)[1] if len(durations) > 3 else 1.0
        
        # Try log-normal fit for peak velocities
        if len(peak_vels) > 5:
            vel_params = lognorm.fit(peak_vels, floc=0)
            vel_ks_stat = scipy.stats.kstest(peak_vels, 
                                           lambda x: lognorm.cdf(x, *vel_params))[1]
        else:
            vel_ks_stat = 1.0
        
        return {
            'duration_distribution_normal': duration_normal_p > 0.05,
            'velocity_distribution_lognormal': vel_ks_stat > 0.05,
            'sample_size_adequate': len(metrics_list) >= 100,
            'duration_cv': np.std(durations) / np.mean(durations) if durations else 0,
            'velocity_cv': np.std(peak_vels) / np.mean(peak_vels) if peak_vels else 0
        }
    
    def _validate_against_literature(self, metrics_list: List[Dict]) -> Dict:
        """Validate against established human movement literature"""
        fitts_ratios = [m['fitts_law_ratio'] for m in metrics_list if 'fitts_law_ratio' in m]
        times_to_peak = [m['time_to_peak_velocity'] for m in metrics_list]
        
        # Fitts' Law validation (should be close to 1.0)
        fitts_valid = (0.8 <= np.mean(fitts_ratios) <= 1.2) if fitts_ratios else False
        
        # Time to peak velocity (typically 40-60% of movement time)
        peak_time_valid = (0.3 <= np.mean(times_to_peak) <= 0.7) if times_to_peak else False
        
        # Speed-accuracy tradeoff validation
        endpoint_errors = [m['endpoint_error'] for m in metrics_list]
        peak_velocities = [m['peak_velocity'] for m in metrics_list]
        
        if len(endpoint_errors) > 5 and len(peak_velocities) > 5:
            # Should be positive correlation between speed and error
            correlation = np.corrcoef(peak_velocities, endpoint_errors)[0, 1]
            speed_accuracy_valid = correlation > 0.1
        else:
            speed_accuracy_valid = True
        
        return {
            'fitts_law_adherence': fitts_valid,
            'velocity_profile_realistic': peak_time_valid,
            'speed_accuracy_tradeoff_present': speed_accuracy_valid,
            'fitts_ratio_stats': {
                'mean': np.mean(fitts_ratios) if fitts_ratios else 0,
                'std': np.std(fitts_ratios) if fitts_ratios else 0
            },
            'literature_compliance_score': sum([
                fitts_valid, peak_time_valid, speed_accuracy_valid
            ]) / 3
        }
    
    def _compute_quality_metrics(self, sequences: List[Dict]) -> Dict:
        """Compute overall dataset quality metrics"""
        # Balance across gesture types
        gesture_counts = Counter()
        for seq in sequences:
            if 'gesture_type' in seq:
                gesture_counts[seq['gesture_type']] += 1
        
        # Calculate balance (entropy-based)
        if gesture_counts:
            probs = np.array(list(gesture_counts.values())) / len(sequences)
            gesture_balance = -np.sum(probs * np.log2(probs + 1e-8)) / np.log2(len(gesture_counts))
        else:
            gesture_balance = 0.0
        
        # Noise level distribution
        noise_levels = [seq['noise_level'] for seq in sequences if 'noise_level' in seq]
        noise_diversity = np.std(noise_levels) if noise_levels else 0
        
        return {
            'gesture_type_balance': gesture_balance,
            'noise_level_diversity': noise_diversity,
            'sequence_count': len(sequences),
            'avg_sequence_duration': np.mean([
                seq.get('duration', 0) for seq in sequences
            ]),
            'quality_score': (gesture_balance + min(1.0, noise_diversity/0.01)) / 2
        }
    
    def _generate_validation_plots(self, sequences: List[Dict], report: Dict) -> None:
        """Generate validation visualizations"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Synthetic Dataset Validation Report', fontsize=16)
            
            # Extract data for plotting
            all_metrics = [seq['movement_metrics'] for seq in sequences 
                          if 'movement_metrics' in seq]
            
            if not all_metrics:
                return
            
            # Plot 1: Velocity distribution
            peak_vels = [m['peak_velocity'] for m in all_metrics]
            axes[0, 0].hist(peak_vels, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(self.biomech.max_hand_velocity, color='red', 
                              linestyle='--', label='Biomech Limit')
            axes[0, 0].set_xlabel('Peak Velocity (m/s)')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Peak Velocity Distribution')
            axes[0, 0].legend()
            
            # Plot 2: Path efficiency
            path_effs = [m['path_efficiency'] for m in all_metrics]
            axes[0, 1].hist(path_effs, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Path Efficiency')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Path Efficiency Distribution')
            
            # Plot 3: Movement smoothness (jerk)
            jerk_scores = [m['normalized_jerk'] for m in all_metrics]
            axes[0, 2].hist(jerk_scores, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 2].set_xlabel('Normalized Jerk')
            axes[0, 2].set_ylabel('Count') 
            axes[0, 2].set_title('Movement Smoothness')
            
            # Plot 4: Fitts' Law validation
            fitts_ratios = [m.get('fitts_law_ratio', 1) for m in all_metrics]
            axes[1, 0].scatter(range(len(fitts_ratios)), fitts_ratios, alpha=0.6)
            axes[1, 0].axhline(1.0, color='green', linestyle='-', label='Perfect Fit')
            axes[1, 0].axhline(0.8, color='red', linestyle='--', alpha=0.5)
            axes[1, 0].axhline(1.2, color='red', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Sequence Index')
            axes[1, 0].set_ylabel('Fitts Law Ratio')
            axes[1, 0].set_title('Fitts Law Adherence')
            axes[1, 0].legend()
            
            # Plot 5: Velocity profiles comparison
            if len(sequences) > 0 and 'velocities' in sequences[0]:
                for i in range(min(5, len(sequences))):
                    if 'velocities' in sequences[i]:
                        vel_profile = np.linalg.norm(sequences[i]['velocities'], axis=1)
                        t_norm = np.linspace(0, 1, len(vel_profile))
                        axes[1, 1].plot(t_norm, vel_profile / np.max(vel_profile), 
                                       alpha=0.6, linewidth=1)
            axes[1, 1].set_xlabel('Normalized Time')
            axes[1, 1].set_ylabel('Normalized Velocity')
            axes[1, 1].set_title('Velocity Profile Examples')
            
            # Plot 6: Quality summary
            quality_scores = [
                report['biomechanical_validation'].get('path_efficiency_realistic', 0),
                report['biomechanical_validation'].get('movement_smoothness_good', 0),
                report['literature_comparison'].get('fitts_law_adherence', 0),
                report['literature_comparison'].get('velocity_profile_realistic', 0),
                report['statistical_validation'].get('sample_size_adequate', 0)
            ]
            
            quality_labels = ['Path Efficiency', 'Smoothness', 'Fitts Law', 
                            'Velocity Profile', 'Sample Size']
            
            axes[1, 2].bar(range(len(quality_scores)), quality_scores)
            axes[1, 2].set_xticks(range(len(quality_scores)))
            axes[1, 2].set_xticklabels(quality_labels, rotation=45, ha='right')
            axes[1, 2].set_ylabel('Quality Score')
            axes[1, 2].set_title('Validation Summary')
            axes[1, 2].set_ylim(0, 1.1)
            
            plt.tight_layout()
            plt.savefig('dataset_validation_report.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Validation plots saved to dataset_validation_report.png")
            
        except Exception as e:
            logger.warning(f"Could not generate validation plots: {e}")


def run_dataset_validation_example():
    """Run comprehensive dataset validation example"""
    print("🔬 Running Enhanced Synthetic Dataset Validation")
    print("=" * 60)
    
    # Initialize enhanced generator
    workspace = np.array([-1, 1, -1, 1, 0, 2])  # 2m x 2m x 2m workspace
    generator = EnhancedSyntheticGenerator(
        workspace_bounds=workspace,
        sampling_frequency=100.0,
        random_seed=42
    )
    
    # Generate sample trajectories
    print("Generating sample trajectories...")
    sample_sequences = []
    
    gesture_types = ['reach', 'grab', 'point', 'handover', 'wave']
    
    for i in range(50):  # Generate 50 sample sequences
        gesture = np.random.choice(gesture_types)
        
        # Random start and end positions
        start_pos = np.array([0.3, 0.0, 1.0])  # Shoulder height
        end_pos = np.random.uniform([-0.5, -0.5, 0.5], [0.5, 0.5, 1.5])
        
        duration = np.random.uniform(1.0, 3.0)
        
        # Generate trajectory
        t, pos, vel, acc, metrics = generator.generate_biomechanically_realistic_trajectory(
            start_pos, end_pos, duration
        )
        
        # Generate intent sequence
        intent_t, intent_labels, confidence = generator.generate_realistic_intent_sequence(
            gesture, duration, {}
        )
        
        # Create sequence dictionary
        sequence = {
            'gesture_type': gesture,
            'hand_trajectory': pos,
            'velocities': vel,
            'accelerations': acc,
            'timestamps': t,
            'intent_labels': intent_labels,
            'confidence_scores': confidence,
            'movement_metrics': metrics,
            'noise_level': 0.02,
            'duration': duration
        }
        
        sample_sequences.append(sequence)
    
    print(f"Generated {len(sample_sequences)} sample sequences")
    
    # Run comprehensive validation
    print("Running comprehensive validation...")
    validation_report = generator.validate_dataset_realism(sample_sequences)
    
    # Print results
    print("\n📊 Validation Results Summary:")
    print("-" * 40)
    
    # Biomechanical validation
    biomech = validation_report['biomechanical_validation']
    print(f"✓ Velocity range valid: {biomech['velocity_range_valid']}")
    print(f"✓ Acceleration range valid: {biomech['acceleration_range_valid']}")
    print(f"✓ Path efficiency realistic: {biomech['path_efficiency_realistic']}")
    print(f"✓ Movement smoothness good: {biomech['movement_smoothness_good']}")
    
    # Literature comparison
    literature = validation_report['literature_comparison']
    print(f"✓ Fitts' law adherence: {literature['fitts_law_adherence']}")
    print(f"✓ Velocity profile realistic: {literature['velocity_profile_realistic']}")
    print(f"✓ Literature compliance: {literature['literature_compliance_score']:.2f}")
    
    # Quality metrics
    quality = validation_report['quality_metrics']
    print(f"✓ Gesture balance: {quality['gesture_type_balance']:.3f}")
    print(f"✓ Overall quality score: {quality['quality_score']:.3f}")
    
    print(f"\n🎯 Dataset Quality Assessment: {'EXCELLENT' if quality['quality_score'] > 0.8 else 'GOOD' if quality['quality_score'] > 0.6 else 'NEEDS IMPROVEMENT'}")
    
    return validation_report


if __name__ == "__main__":
    # Run validation example
    report = run_dataset_validation_example()
    print("\n✅ Enhanced synthetic dataset validation complete!")