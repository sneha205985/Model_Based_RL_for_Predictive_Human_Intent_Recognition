"""
Feature extraction pipeline for human behavior analysis.

This module extracts kinematic, temporal, and spatial features from
human movement trajectories for intent recognition and behavior modeling.

Mathematical foundations:
- Kinematic features: velocity v = dx/dt, acceleration a = dv/dt
- Temporal features: periodicity via FFT, rhythm analysis
- Spatial features: curvature κ = |v × a| / |v|³, workspace analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import signal
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd

from ..utils.logger import get_logger
from ..data.synthetic_generator import SyntheticSequence, GestureType

logger = get_logger(__name__)


class FeatureType(Enum):
    """Types of features that can be extracted."""
    KINEMATIC = "kinematic"
    TEMPORAL = "temporal" 
    SPATIAL = "spatial"
    STATISTICAL = "statistical"
    FREQUENCY = "frequency"


@dataclass
class FeatureConfig:
    """
    Configuration for feature extraction.
    
    Attributes:
        window_size: Size of sliding window for local features
        overlap: Overlap between windows (0-1)
        smooth_trajectories: Whether to smooth trajectories before extraction
        smoothing_window: Window size for trajectory smoothing
        extract_derivatives: Whether to extract velocity and acceleration
        frequency_bands: Frequency bands for spectral analysis
        spatial_resolution: Resolution for spatial binning
    """
    window_size: int = 30
    overlap: float = 0.5
    smooth_trajectories: bool = True
    smoothing_window: int = 5
    extract_derivatives: bool = True
    frequency_bands: List[Tuple[float, float]] = None
    spatial_resolution: float = 0.1
    
    def __post_init__(self) -> None:
        """Set default frequency bands if not provided."""
        if self.frequency_bands is None:
            self.frequency_bands = [
                (0.1, 2.0),   # Low frequency (general motion)
                (2.0, 8.0),   # Mid frequency (gesture dynamics) 
                (8.0, 15.0),  # High frequency (fine motor control)
            ]


@dataclass  
class ExtractedFeatures:
    """
    Container for extracted features from a trajectory.
    
    Attributes:
        sequence_id: Identifier for the source sequence
        kinematic_features: Velocity, acceleration, jerk features
        temporal_features: Timing, rhythm, periodicity features  
        spatial_features: Curvature, workspace, path features
        statistical_features: Mean, std, percentile features
        frequency_features: Spectral power, dominant frequencies
        feature_names: Names of all extracted features
        timestamps: Timestamps for windowed features
    """
    sequence_id: str
    kinematic_features: np.ndarray
    temporal_features: np.ndarray
    spatial_features: np.ndarray
    statistical_features: np.ndarray
    frequency_features: np.ndarray
    feature_names: List[str]
    timestamps: np.ndarray
    
    def get_all_features(self) -> np.ndarray:
        """Concatenate all feature types into single array."""
        features = []
        for feature_array in [
            self.kinematic_features,
            self.temporal_features, 
            self.spatial_features,
            self.statistical_features,
            self.frequency_features
        ]:
            if feature_array.size > 0:
                if feature_array.ndim == 1:
                    features.append(feature_array[None, :])  # Add time dimension
                else:
                    features.append(feature_array)
        
        if features:
            return np.concatenate(features, axis=-1)
        else:
            return np.array([[]])
    
    def get_feature_dict(self) -> Dict[str, np.ndarray]:
        """Get features as dictionary with named keys."""
        all_features = self.get_all_features()
        if all_features.size == 0:
            return {}
        
        feature_dict = {}
        for i, name in enumerate(self.feature_names):
            if i < all_features.shape[-1]:
                feature_dict[name] = all_features[:, i] if all_features.ndim > 1 else all_features[i]
        
        return feature_dict


class FeatureExtractor:
    """
    Comprehensive feature extraction for human behavior analysis.
    
    This class implements various feature extraction methods for analyzing
    human movement trajectories, including kinematic, temporal, spatial,
    statistical, and frequency domain features.
    """
    
    def __init__(self, config: FeatureConfig = None):
        """
        Initialize feature extractor.
        
        Args:
            config: Feature extraction configuration
        """
        self.config = config if config is not None else FeatureConfig()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        logger.info("Initialized feature extractor")
    
    def extract_features(
        self,
        sequence: SyntheticSequence,
        feature_types: List[FeatureType] = None
    ) -> ExtractedFeatures:
        """
        Extract features from a synthetic sequence.
        
        Args:
            sequence: Synthetic behavior sequence
            feature_types: Types of features to extract (default: all)
            
        Returns:
            Extracted features container
        """
        if feature_types is None:
            feature_types = list(FeatureType)
        
        # Smooth trajectories if requested
        trajectory = sequence.hand_trajectory.copy()
        if self.config.smooth_trajectories:
            trajectory = self._smooth_trajectory(trajectory)
        
        # Extract derivatives
        velocities, accelerations = self._compute_derivatives(
            trajectory, sequence.timestamps
        )
        
        # Initialize feature containers
        all_features = []
        feature_names = []
        
        # Extract each feature type
        if FeatureType.KINEMATIC in feature_types:
            kinematic_feats, kinematic_names = self._extract_kinematic_features(
                trajectory, velocities, accelerations
            )
            all_features.append(kinematic_feats)
            feature_names.extend(kinematic_names)
        else:
            kinematic_feats = np.array([])
        
        if FeatureType.TEMPORAL in feature_types:
            temporal_feats, temporal_names = self._extract_temporal_features(
                trajectory, sequence.timestamps
            )
            all_features.append(temporal_feats)
            feature_names.extend(temporal_names)
        else:
            temporal_feats = np.array([])
        
        if FeatureType.SPATIAL in feature_types:
            spatial_feats, spatial_names = self._extract_spatial_features(
                trajectory, velocities, accelerations
            )
            all_features.append(spatial_feats)
            feature_names.extend(spatial_names)
        else:
            spatial_feats = np.array([])
        
        if FeatureType.STATISTICAL in feature_types:
            statistical_feats, statistical_names = self._extract_statistical_features(
                trajectory, velocities, accelerations
            )
            all_features.append(statistical_feats)
            feature_names.extend(statistical_names)
        else:
            statistical_feats = np.array([])
        
        if FeatureType.FREQUENCY in feature_types:
            frequency_feats, frequency_names = self._extract_frequency_features(
                trajectory, sequence.timestamps
            )
            all_features.append(frequency_feats)
            feature_names.extend(frequency_names)
        else:
            frequency_feats = np.array([])
        
        return ExtractedFeatures(
            sequence_id=sequence.sequence_id,
            kinematic_features=kinematic_feats,
            temporal_features=temporal_feats,
            spatial_features=spatial_feats,
            statistical_features=statistical_feats,
            frequency_features=frequency_feats,
            feature_names=feature_names,
            timestamps=sequence.timestamps
        )
    
    def _smooth_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Smooth trajectory using Savitzky-Golay filter.
        
        Args:
            trajectory: Raw trajectory [N, 3]
            
        Returns:
            Smoothed trajectory [N, 3]
        """
        window = min(self.config.smoothing_window, trajectory.shape[0] // 2)
        if window % 2 == 0:
            window += 1  # Ensure odd window size
        
        if window < 3:
            return trajectory
        
        smoothed = np.zeros_like(trajectory)
        for dim in range(trajectory.shape[1]):
            smoothed[:, dim] = signal.savgol_filter(
                trajectory[:, dim], window, polyorder=2
            )
        
        return smoothed
    
    def _compute_derivatives(
        self,
        trajectory: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute velocity and acceleration using numerical differentiation.
        
        Mathematical formulation:
        v(t) = dx/dt ≈ (x(t+h) - x(t-h)) / (2h)
        a(t) = dv/dt ≈ (v(t+h) - v(t-h)) / (2h)
        
        Args:
            trajectory: Position trajectory [N, 3]
            timestamps: Time stamps [N]
            
        Returns:
            Tuple of (velocities [N, 3], accelerations [N, 3])
        """
        dt = np.diff(timestamps)
        dt_mean = np.mean(dt)
        
        # Velocity using central difference
        velocities = np.zeros_like(trajectory)
        velocities[1:-1] = (trajectory[2:] - trajectory[:-2]) / (2 * dt_mean)
        velocities[0] = (trajectory[1] - trajectory[0]) / dt[0]
        velocities[-1] = (trajectory[-1] - trajectory[-2]) / dt[-1]
        
        # Acceleration using central difference on velocities
        accelerations = np.zeros_like(trajectory)
        accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (2 * dt_mean)
        accelerations[0] = (velocities[1] - velocities[0]) / dt[0]
        accelerations[-1] = (velocities[-1] - velocities[-2]) / dt[-1]
        
        return velocities, accelerations
    
    def _extract_kinematic_features(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract kinematic features from motion data.
        
        Features include:
        - Speed (magnitude of velocity)
        - Acceleration magnitude
        - Jerk (derivative of acceleration)
        - Angular velocity
        - Smoothness metrics
        
        Args:
            positions: Position data [N, 3]
            velocities: Velocity data [N, 3]  
            accelerations: Acceleration data [N, 3]
            
        Returns:
            Tuple of (features [M], feature_names)
        """
        features = []
        names = []
        
        # Speed and acceleration magnitudes
        speeds = np.linalg.norm(velocities, axis=1)
        acc_mags = np.linalg.norm(accelerations, axis=1)
        
        # Basic statistics of speed and acceleration
        features.extend([
            np.mean(speeds), np.std(speeds), np.max(speeds),
            np.mean(acc_mags), np.std(acc_mags), np.max(acc_mags)
        ])
        names.extend([
            'speed_mean', 'speed_std', 'speed_max',
            'acc_mean', 'acc_std', 'acc_max'
        ])
        
        # Jerk (rate of change of acceleration)
        if len(accelerations) > 2:
            jerks = np.diff(accelerations, axis=0)
            jerk_mags = np.linalg.norm(jerks, axis=1)
            features.extend([np.mean(jerk_mags), np.std(jerk_mags)])
            names.extend(['jerk_mean', 'jerk_std'])
        else:
            features.extend([0.0, 0.0])
            names.extend(['jerk_mean', 'jerk_std'])
        
        # Smoothness - spectral arc length
        if len(speeds) > 10:
            # Normalize speed profile
            speed_norm = (speeds - np.mean(speeds)) / (np.std(speeds) + 1e-8)
            
            # Compute spectral arc length (smoothness metric)
            freqs = np.fft.fftfreq(len(speed_norm))
            fft_speed = np.fft.fft(speed_norm)
            power_spectrum = np.abs(fft_speed) ** 2
            
            # Spectral arc length = integral of |dP/df|
            spectral_arc_length = np.sum(np.abs(np.diff(power_spectrum)))
            features.append(spectral_arc_length)
            names.append('smoothness_sal')
        else:
            features.append(0.0)
            names.append('smoothness_sal')
        
        # Number of movement units (velocity peaks)
        if len(speeds) > 5:
            peaks, _ = signal.find_peaks(speeds, height=np.mean(speeds))
            features.append(len(peaks))
            names.append('n_movement_units')
        else:
            features.append(0)
            names.append('n_movement_units')
        
        # Movement efficiency (path length ratio)
        total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        direct_distance = np.linalg.norm(positions[-1] - positions[0])
        efficiency = direct_distance / (total_distance + 1e-8)
        features.append(efficiency)
        names.append('movement_efficiency')
        
        return np.array(features), names
    
    def _extract_temporal_features(
        self,
        positions: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract temporal features from trajectory timing.
        
        Features include:
        - Movement duration
        - Pause analysis
        - Rhythm and periodicity
        - Temporal asymmetry
        
        Args:
            positions: Position data [N, 3]
            timestamps: Time stamps [N]
            
        Returns:
            Tuple of (features [M], feature_names)
        """
        features = []
        names = []
        
        # Basic temporal features
        duration = timestamps[-1] - timestamps[0]
        n_samples = len(timestamps)
        sampling_rate = n_samples / duration if duration > 0 else 0
        
        features.extend([duration, sampling_rate])
        names.extend(['duration', 'sampling_rate'])
        
        # Movement phases analysis
        velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        speed_threshold = np.mean(velocities) * 0.1  # 10% of mean speed
        
        # Identify movement and pause phases
        moving_mask = velocities > speed_threshold
        
        # Count number of movement phases
        movement_phases = 0
        in_movement = False
        for is_moving in moving_mask:
            if is_moving and not in_movement:
                movement_phases += 1
                in_movement = True
            elif not is_moving:
                in_movement = False
        
        features.append(movement_phases)
        names.append('n_movement_phases')
        
        # Movement vs pause time ratio
        if len(moving_mask) > 0:
            movement_time_ratio = np.sum(moving_mask) / len(moving_mask)
        else:
            movement_time_ratio = 0
        features.append(movement_time_ratio)
        names.append('movement_time_ratio')
        
        # Periodicity analysis via autocorrelation
        if len(velocities) > 20:
            # Normalize velocity signal
            vel_norm = (velocities - np.mean(velocities)) / (np.std(velocities) + 1e-8)
            
            # Compute autocorrelation
            autocorr = np.correlate(vel_norm, vel_norm, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find dominant period
            if len(autocorr) > 10:
                peaks, _ = signal.find_peaks(autocorr[1:], height=0.3)  # Exclude zero lag
                if len(peaks) > 0:
                    dominant_period = peaks[0] + 1  # Add 1 for excluded zero lag
                    periodicity_strength = autocorr[dominant_period]
                else:
                    dominant_period = 0
                    periodicity_strength = 0
            else:
                dominant_period = 0
                periodicity_strength = 0
            
            features.extend([dominant_period, periodicity_strength])
            names.extend(['dominant_period', 'periodicity_strength'])
        else:
            features.extend([0, 0])
            names.extend(['dominant_period', 'periodicity_strength'])
        
        # Time to peak velocity
        if len(velocities) > 0:
            peak_vel_idx = np.argmax(velocities)
            time_to_peak = peak_vel_idx / len(velocities)  # Normalized
            features.append(time_to_peak)
            names.append('time_to_peak_velocity')
        else:
            features.append(0.5)  # Default middle
            names.append('time_to_peak_velocity')
        
        return np.array(features), names
    
    def _extract_spatial_features(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract spatial features from trajectory geometry.
        
        Features include:
        - Path curvature
        - Workspace utilization  
        - Directional changes
        - 3D geometric properties
        
        Mathematical formulation for curvature:
        κ = |v × a| / |v|³
        
        Args:
            positions: Position data [N, 3]
            velocities: Velocity data [N, 3]
            accelerations: Acceleration data [N, 3]
            
        Returns:
            Tuple of (features [M], feature_names)
        """
        features = []
        names = []
        
        # Path length and displacement
        if len(positions) > 1:
            path_segments = np.diff(positions, axis=0)
            segment_lengths = np.linalg.norm(path_segments, axis=1)
            total_path_length = np.sum(segment_lengths)
            
            displacement = np.linalg.norm(positions[-1] - positions[0])
            path_efficiency = displacement / (total_path_length + 1e-8)
        else:
            total_path_length = 0
            displacement = 0
            path_efficiency = 1
        
        features.extend([total_path_length, displacement, path_efficiency])
        names.extend(['path_length', 'displacement', 'path_efficiency'])
        
        # Workspace bounding box
        if len(positions) > 0:
            bbox_size = np.ptp(positions, axis=0)  # Peak-to-peak (max - min)
            workspace_volume = np.prod(bbox_size)
            
            features.extend([*bbox_size, workspace_volume])
            names.extend(['workspace_x', 'workspace_y', 'workspace_z', 'workspace_volume'])
        else:
            features.extend([0, 0, 0, 0])
            names.extend(['workspace_x', 'workspace_y', 'workspace_z', 'workspace_volume'])
        
        # Curvature analysis
        if len(positions) > 2:
            # Compute curvature at each point: κ = |v × a| / |v|³
            v_mag = np.linalg.norm(velocities, axis=1)
            
            # Cross product |v × a|
            cross_products = np.cross(velocities, accelerations)
            if cross_products.ndim == 1:  # Handle 2D case
                cross_mag = np.abs(cross_products)
            else:
                cross_mag = np.linalg.norm(cross_products, axis=1)
            
            # Curvature κ = |v × a| / |v|³
            curvature = cross_mag / (v_mag**3 + 1e-8)
            
            # Remove invalid values
            curvature = curvature[np.isfinite(curvature)]
            
            if len(curvature) > 0:
                features.extend([
                    np.mean(curvature), np.std(curvature), 
                    np.max(curvature), np.median(curvature)
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            names.extend(['curvature_mean', 'curvature_std', 'curvature_max', 'curvature_median'])
        else:
            features.extend([0, 0, 0, 0])
            names.extend(['curvature_mean', 'curvature_std', 'curvature_max', 'curvature_median'])
        
        # Directional changes
        if len(positions) > 2:
            # Compute direction vectors
            directions = positions[1:] - positions[:-1]
            direction_norms = np.linalg.norm(directions, axis=1)
            
            # Normalize directions
            valid_dirs = direction_norms > 1e-8
            if np.any(valid_dirs):
                directions[valid_dirs] = directions[valid_dirs] / direction_norms[valid_dirs, None]
                
                # Compute angular changes between consecutive directions
                if len(directions) > 1:
                    dot_products = np.sum(directions[:-1] * directions[1:], axis=1)
                    # Clamp to valid range for arccos
                    dot_products = np.clip(dot_products, -1.0, 1.0)
                    angles = np.arccos(dot_products)
                    
                    # Remove invalid values
                    angles = angles[np.isfinite(angles)]
                    
                    if len(angles) > 0:
                        total_angular_change = np.sum(angles)
                        mean_angular_change = np.mean(angles)
                        n_direction_changes = np.sum(angles > np.pi/4)  # 45 degree threshold
                    else:
                        total_angular_change = 0
                        mean_angular_change = 0
                        n_direction_changes = 0
                else:
                    total_angular_change = 0
                    mean_angular_change = 0
                    n_direction_changes = 0
            else:
                total_angular_change = 0
                mean_angular_change = 0
                n_direction_changes = 0
        else:
            total_angular_change = 0
            mean_angular_change = 0
            n_direction_changes = 0
        
        features.extend([total_angular_change, mean_angular_change, n_direction_changes])
        names.extend(['total_angular_change', 'mean_angular_change', 'n_direction_changes'])
        
        # Trajectory spread (variance in each dimension)
        if len(positions) > 0:
            position_variance = np.var(positions, axis=0)
            features.extend(position_variance.tolist())
            names.extend(['pos_var_x', 'pos_var_y', 'pos_var_z'])
        else:
            features.extend([0, 0, 0])
            names.extend(['pos_var_x', 'pos_var_y', 'pos_var_z'])
        
        return np.array(features), names
    
    def _extract_statistical_features(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract statistical features from trajectory data.
        
        Features include:
        - Moments (mean, variance, skewness, kurtosis)
        - Percentiles
        - Distribution properties
        
        Args:
            positions: Position data [N, 3]
            velocities: Velocity data [N, 3]
            accelerations: Acceleration data [N, 3]
            
        Returns:
            Tuple of (features [M], feature_names)
        """
        from scipy import stats
        
        features = []
        names = []
        
        # Position statistics for each dimension
        for dim in range(3):
            dim_name = ['x', 'y', 'z'][dim]
            pos_dim = positions[:, dim]
            
            if len(pos_dim) > 0:
                features.extend([
                    np.mean(pos_dim), np.std(pos_dim),
                    stats.skew(pos_dim), stats.kurtosis(pos_dim),
                    np.percentile(pos_dim, 25), np.percentile(pos_dim, 75)
                ])
                names.extend([
                    f'pos_{dim_name}_mean', f'pos_{dim_name}_std',
                    f'pos_{dim_name}_skew', f'pos_{dim_name}_kurt',
                    f'pos_{dim_name}_q25', f'pos_{dim_name}_q75'
                ])
            else:
                features.extend([0, 0, 0, 0, 0, 0])
                names.extend([
                    f'pos_{dim_name}_mean', f'pos_{dim_name}_std',
                    f'pos_{dim_name}_skew', f'pos_{dim_name}_kurt',
                    f'pos_{dim_name}_q25', f'pos_{dim_name}_q75'
                ])
        
        # Velocity magnitude statistics
        vel_mags = np.linalg.norm(velocities, axis=1)
        if len(vel_mags) > 0:
            features.extend([
                np.mean(vel_mags), np.std(vel_mags),
                stats.skew(vel_mags), stats.kurtosis(vel_mags),
                np.percentile(vel_mags, 10), np.percentile(vel_mags, 90)
            ])
            names.extend([
                'vel_mean', 'vel_std', 'vel_skew', 'vel_kurt',
                'vel_p10', 'vel_p90'
            ])
        else:
            features.extend([0, 0, 0, 0, 0, 0])
            names.extend([
                'vel_mean', 'vel_std', 'vel_skew', 'vel_kurt',
                'vel_p10', 'vel_p90'
            ])
        
        # Acceleration magnitude statistics
        acc_mags = np.linalg.norm(accelerations, axis=1)
        if len(acc_mags) > 0:
            features.extend([
                np.mean(acc_mags), np.std(acc_mags),
                np.percentile(acc_mags, 95)  # 95th percentile for peak detection
            ])
            names.extend(['acc_mean', 'acc_std', 'acc_p95'])
        else:
            features.extend([0, 0, 0])
            names.extend(['acc_mean', 'acc_std', 'acc_p95'])
        
        return np.array(features), names
    
    def _extract_frequency_features(
        self,
        positions: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract frequency domain features using FFT analysis.
        
        Features include:
        - Spectral power in different bands
        - Dominant frequencies
        - Spectral centroid and spread
        
        Args:
            positions: Position data [N, 3]
            timestamps: Time stamps [N]
            
        Returns:
            Tuple of (features [M], feature_names)
        """
        features = []
        names = []
        
        if len(positions) < 10:  # Need minimum samples for FFT
            # Return zero features
            n_bands = len(self.config.frequency_bands)
            features.extend([0] * (n_bands * 3 + 6))  # 3 dims * n_bands + 6 spectral features
            names.extend([f'power_{i}_{dim}' for i in range(n_bands) for dim in ['x', 'y', 'z']])
            names.extend(['dominant_freq', 'spectral_centroid', 'spectral_spread',
                         'spectral_rolloff', 'spectral_flux', 'zero_crossing_rate'])
            return np.array(features), names
        
        # Compute sampling rate
        dt = np.mean(np.diff(timestamps))
        fs = 1.0 / dt
        
        # For each spatial dimension
        for dim in range(3):
            dim_name = ['x', 'y', 'z'][dim]
            signal_data = positions[:, dim]
            
            # Remove DC component
            signal_data = signal_data - np.mean(signal_data)
            
            # Apply window function
            windowed_signal = signal_data * np.hanning(len(signal_data))
            
            # Compute FFT
            fft_data = np.fft.fft(windowed_signal)
            freqs = np.fft.fftfreq(len(fft_data), dt)
            
            # Power spectrum (positive frequencies only)
            pos_freqs = freqs[:len(freqs)//2]
            power_spectrum = np.abs(fft_data[:len(freqs)//2])**2
            
            # Normalize power spectrum
            power_spectrum = power_spectrum / np.sum(power_spectrum + 1e-8)
            
            # Extract power in frequency bands
            for i, (f_low, f_high) in enumerate(self.config.frequency_bands):
                band_mask = (pos_freqs >= f_low) & (pos_freqs <= f_high)
                band_power = np.sum(power_spectrum[band_mask])
                features.append(band_power)
                names.append(f'power_band_{i}_{dim_name}')
        
        # Global spectral features (using velocity magnitude)
        velocities = np.diff(positions, axis=0)
        vel_mags = np.linalg.norm(velocities, axis=1)
        
        if len(vel_mags) > 5:
            vel_mags = vel_mags - np.mean(vel_mags)
            vel_windowed = vel_mags * np.hanning(len(vel_mags))
            
            fft_vel = np.fft.fft(vel_windowed)
            freqs_vel = np.fft.fftfreq(len(fft_vel), dt)
            pos_freqs_vel = freqs_vel[:len(freqs_vel)//2]
            power_vel = np.abs(fft_vel[:len(freqs_vel)//2])**2
            power_vel = power_vel / (np.sum(power_vel) + 1e-8)
            
            # Dominant frequency
            if len(power_vel) > 0:
                dominant_freq = pos_freqs_vel[np.argmax(power_vel)]
            else:
                dominant_freq = 0
            
            # Spectral centroid
            spectral_centroid = np.sum(pos_freqs_vel * power_vel) / (np.sum(power_vel) + 1e-8)
            
            # Spectral spread
            spectral_spread = np.sqrt(np.sum((pos_freqs_vel - spectral_centroid)**2 * power_vel) / 
                                    (np.sum(power_vel) + 1e-8))
            
            # Spectral rolloff (frequency below which 85% of energy is contained)
            cumulative_power = np.cumsum(power_vel)
            rolloff_idx = np.where(cumulative_power >= 0.85 * cumulative_power[-1])[0]
            spectral_rolloff = pos_freqs_vel[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
            
            # Spectral flux (rate of change in spectrum)
            if len(power_vel) > 1:
                spectral_flux = np.mean(np.diff(power_vel)**2)
            else:
                spectral_flux = 0
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(vel_mags)) != 0)
            zero_crossing_rate = zero_crossings / len(vel_mags)
            
        else:
            dominant_freq = 0
            spectral_centroid = 0
            spectral_spread = 0
            spectral_rolloff = 0
            spectral_flux = 0
            zero_crossing_rate = 0
        
        features.extend([
            dominant_freq, spectral_centroid, spectral_spread,
            spectral_rolloff, spectral_flux, zero_crossing_rate
        ])
        names.extend([
            'dominant_freq', 'spectral_centroid', 'spectral_spread',
            'spectral_rolloff', 'spectral_flux', 'zero_crossing_rate'
        ])
        
        return np.array(features), names
    
    def extract_batch_features(
        self,
        sequences: List[SyntheticSequence],
        feature_types: List[FeatureType] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Extract features from a batch of sequences.
        
        Args:
            sequences: List of synthetic sequences
            feature_types: Types of features to extract
            normalize: Whether to normalize features
            
        Returns:
            Tuple of (feature_matrix [N, M], feature_names, sequence_ids)
        """
        extracted_features = []
        sequence_ids = []
        
        logger.info(f"Extracting features from {len(sequences)} sequences...")
        
        for i, sequence in enumerate(sequences):
            try:
                features = self.extract_features(sequence, feature_types)
                extracted_features.append(features)
                sequence_ids.append(sequence.sequence_id)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(sequences)} sequences")
                    
            except Exception as e:
                logger.warning(f"Failed to extract features from sequence {sequence.sequence_id}: {e}")
                continue
        
        if not extracted_features:
            raise ValueError("No features could be extracted from any sequence")
        
        # Convert to matrix format
        feature_matrix = []
        feature_names = extracted_features[0].feature_names
        
        for features in extracted_features:
            all_feats = features.get_all_features()
            if all_feats.size > 0:
                # Take mean over time if features are temporal
                if all_feats.ndim > 1:
                    feat_vector = np.mean(all_feats, axis=0)
                else:
                    feat_vector = all_feats
                feature_matrix.append(feat_vector)
            else:
                # Handle empty features
                feature_matrix.append(np.zeros(len(feature_names)))
        
        feature_matrix = np.array(feature_matrix)
        
        # Normalize features if requested
        if normalize and feature_matrix.shape[0] > 1:
            if not self.is_fitted:
                feature_matrix = self.scaler.fit_transform(feature_matrix)
                self.is_fitted = True
            else:
                feature_matrix = self.scaler.transform(feature_matrix)
        
        logger.info(f"Extracted feature matrix: {feature_matrix.shape}")
        return feature_matrix, feature_names, sequence_ids
    
    def save_features(
        self,
        feature_matrix: np.ndarray,
        feature_names: List[str],
        sequence_ids: List[str],
        filepath: str
    ) -> None:
        """
        Save extracted features to file.
        
        Args:
            feature_matrix: Feature matrix [N, M]
            feature_names: List of feature names
            sequence_ids: List of sequence identifiers
            filepath: Output file path
        """
        # Create DataFrame for easy saving
        df = pd.DataFrame(feature_matrix, columns=feature_names)
        df['sequence_id'] = sequence_ids
        
        # Save as CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Features saved to {filepath}")
    
    @classmethod
    def load_features(cls, filepath: str) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Load features from file.
        
        Args:
            filepath: Input file path
            
        Returns:
            Tuple of (feature_matrix, feature_names, sequence_ids)
        """
        df = pd.read_csv(filepath)
        sequence_ids = df['sequence_id'].tolist()
        feature_names = [col for col in df.columns if col != 'sequence_id']
        feature_matrix = df[feature_names].values
        
        logger.info(f"Loaded features from {filepath}: {feature_matrix.shape}")
        return feature_matrix, feature_names, sequence_ids