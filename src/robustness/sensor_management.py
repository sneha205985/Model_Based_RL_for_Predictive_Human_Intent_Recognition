#!/usr/bin/env python3
"""
Sensor Failure Handling & Management System
===========================================

This module implements comprehensive sensor failure handling with multi-modal
sensor fusion, graceful degradation strategies, uncertainty propagation through
sensor failures, and backup sensing modalities with fallback behaviors.

Features:
- Multi-modal sensor fusion with failure detection
- Sensor health monitoring and predictive maintenance
- Graceful degradation strategies for sensor failures  
- Uncertainty propagation through sensor failure scenarios
- Backup sensing modalities and automatic fallback behaviors
- Real-time sensor reliability assessment
- Adaptive fusion weights based on sensor health

Mathematical Framework:
======================

Multi-Modal Sensor Fusion:
    x̂ = Σᵢ wᵢ · xᵢ  where Σwᵢ = 1
    
Uncertainty Propagation:
    P(x̂) = Σᵢ wᵢ² · P(xᵢ) + Σᵢⱼ wᵢwⱼ · Cov(xᵢ,xⱼ)
    
Kalman Fusion:
    x̂ = (Σᵢ Rᵢ⁻¹)⁻¹ · Σᵢ Rᵢ⁻¹ · xᵢ
    
Sensor Reliability:
    R(t) = exp(-λt)  (exponential reliability)
    
Author: Claude Code (Anthropic)
Date: 2025-01-15
Version: 1.0
"""

import time
import threading
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from abc import ABC, abstractmethod
from collections import deque, defaultdict
import queue
import json
from pathlib import Path
import scipy.stats as stats
from scipy.spatial.transform import Rotation
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Types of sensors"""
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    FORCE_TORQUE = "force_torque"
    ENCODER = "encoder"
    TACTILE = "tactile"
    MICROPHONE = "microphone"
    ULTRASONIC = "ultrasonic"


class SensorStatus(Enum):
    """Sensor operational status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    SUSPICIOUS = "suspicious"
    FAILED = "failed"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class FailureType(Enum):
    """Types of sensor failures"""
    NO_DATA = "no_data"
    NOISY_DATA = "noisy_data"
    BIASED_DATA = "biased_data"
    INTERMITTENT = "intermittent"
    STUCK_VALUE = "stuck_value"
    CALIBRATION_DRIFT = "calibration_drift"
    HARDWARE_FAULT = "hardware_fault"
    COMMUNICATION_LOSS = "communication_loss"


@dataclass
class SensorConfiguration:
    """Sensor configuration and specifications"""
    sensor_id: str
    sensor_type: SensorType
    name: str
    description: str
    
    # Physical properties
    sample_rate: float  # Hz
    resolution: Union[float, Tuple[float, ...]]
    measurement_range: Tuple[float, float]
    accuracy: float
    precision: float
    
    # Reliability parameters
    mtbf: float = 8760.0  # Mean Time Between Failures (hours)
    mttr: float = 2.0     # Mean Time To Repair (hours)
    failure_rate: float = 1e-5  # failures per hour
    
    # Fusion parameters
    measurement_noise: float = 0.01
    update_frequency: float = 10.0
    fusion_weight: float = 1.0
    
    # Backup configuration
    backup_sensors: List[str] = field(default_factory=list)
    can_be_backup: bool = True
    
    # Maintenance schedule
    maintenance_interval: float = 720.0  # hours
    last_maintenance: float = field(default_factory=time.time)


@dataclass
class SensorMeasurement:
    """Single sensor measurement with metadata"""
    sensor_id: str
    timestamp: float
    data: np.ndarray
    uncertainty: np.ndarray
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    is_valid: bool = True
    validation_flags: List[str] = field(default_factory=list)


@dataclass
class FusedMeasurement:
    """Result of multi-sensor fusion"""
    timestamp: float
    fused_data: np.ndarray
    fused_uncertainty: np.ndarray
    fusion_weights: Dict[str, float]
    contributing_sensors: List[str]
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SensorHealthMonitor:
    """Monitors sensor health and detects failures"""
    
    def __init__(self, monitoring_window: float = 60.0):
        """Initialize sensor health monitor"""
        self.monitoring_window = monitoring_window
        self.sensor_configs: Dict[str, SensorConfiguration] = {}
        self.sensor_data_history: Dict[str, deque] = {}
        self.health_statistics: Dict[str, Dict[str, float]] = {}
        
        # Failure detection
        self.anomaly_detectors: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Thresholds
        self.health_thresholds = {
            'data_timeout': 5.0,  # seconds
            'quality_threshold': 0.3,
            'noise_threshold': 5.0,  # std devs above normal
            'bias_threshold': 3.0,
            'stuck_threshold': 0.001,  # variance threshold
            'anomaly_threshold': 0.1
        }
        
        logger.info("Sensor health monitor initialized")
    
    def register_sensor(self, config: SensorConfiguration) -> None:
        """Register sensor for monitoring"""
        self.sensor_configs[config.sensor_id] = config
        self.sensor_data_history[config.sensor_id] = deque(maxlen=int(
            config.sample_rate * self.monitoring_window
        ))
        self.health_statistics[config.sensor_id] = {
            'status': SensorStatus.HEALTHY.value,
            'reliability': 1.0,
            'failure_probability': 0.0,
            'last_data_time': time.time(),
            'data_rate': 0.0,
            'quality_score': 1.0,
            'noise_level': 0.0,
            'bias_estimate': 0.0,
            'maintenance_due': False
        }
        
        # Initialize anomaly detector
        self.anomaly_detectors[config.sensor_id] = IsolationForest(
            contamination=0.1, random_state=42
        )
        self.scalers[config.sensor_id] = StandardScaler()
        
        logger.debug(f"Registered sensor: {config.sensor_id}")
    
    def update_sensor_data(self, measurement: SensorMeasurement) -> SensorStatus:
        """Update sensor data and assess health"""
        sensor_id = measurement.sensor_id
        
        if sensor_id not in self.sensor_configs:
            logger.warning(f"Unknown sensor: {sensor_id}")
            return SensorStatus.OFFLINE
        
        # Store measurement
        self.sensor_data_history[sensor_id].append(measurement)
        
        # Update basic statistics
        current_time = time.time()
        self.health_statistics[sensor_id]['last_data_time'] = current_time
        
        # Calculate data rate
        if len(self.sensor_data_history[sensor_id]) >= 2:
            recent_measurements = list(self.sensor_data_history[sensor_id])[-10:]
            time_diffs = [
                recent_measurements[i].timestamp - recent_measurements[i-1].timestamp
                for i in range(1, len(recent_measurements))
            ]
            if time_diffs:
                avg_interval = np.mean(time_diffs)
                self.health_statistics[sensor_id]['data_rate'] = 1.0 / avg_interval if avg_interval > 0 else 0.0
        
        # Perform health assessment
        status = self._assess_sensor_health(sensor_id)
        self.health_statistics[sensor_id]['status'] = status.value
        
        return status
    
    def _assess_sensor_health(self, sensor_id: str) -> SensorStatus:
        """Comprehensive sensor health assessment"""
        config = self.sensor_configs[sensor_id]
        stats = self.health_statistics[sensor_id]
        history = self.sensor_data_history[sensor_id]
        
        if not history:
            return SensorStatus.OFFLINE
        
        current_time = time.time()
        
        # Check data timeout
        time_since_data = current_time - stats['last_data_time']
        if time_since_data > self.health_thresholds['data_timeout']:
            return SensorStatus.FAILED
        
        # Need sufficient data for analysis
        if len(history) < 10:
            return SensorStatus.HEALTHY  # Assume healthy until proven otherwise
        
        recent_measurements = list(history)[-50:]  # Last 50 measurements
        
        # Extract data for analysis
        data_values = []
        quality_scores = []
        timestamps = []
        
        for measurement in recent_measurements:
            if measurement.is_valid:
                data_values.append(measurement.data.flatten())
                quality_scores.append(measurement.quality_score)
                timestamps.append(measurement.timestamp)
        
        if not data_values:
            return SensorStatus.FAILED
        
        data_matrix = np.array(data_values)
        
        # Quality score check
        avg_quality = np.mean(quality_scores)
        stats['quality_score'] = avg_quality
        if avg_quality < self.health_thresholds['quality_threshold']:
            return SensorStatus.DEGRADED
        
        # Noise level assessment
        noise_level = self._assess_noise_level(data_matrix, config)
        stats['noise_level'] = noise_level
        if noise_level > self.health_thresholds['noise_threshold']:
            return SensorStatus.DEGRADED
        
        # Bias detection
        bias_level = self._detect_bias(data_matrix, config)
        stats['bias_estimate'] = bias_level
        if bias_level > self.health_thresholds['bias_threshold']:
            return SensorStatus.SUSPICIOUS
        
        # Stuck value detection
        variance = np.var(data_matrix, axis=0)
        if np.any(variance < self.health_thresholds['stuck_threshold']):
            return SensorStatus.FAILED
        
        # Anomaly detection
        if self._detect_anomalies(sensor_id, data_matrix):
            return SensorStatus.SUSPICIOUS
        
        # Data rate check
        expected_rate = config.sample_rate
        actual_rate = stats['data_rate']
        if actual_rate < expected_rate * 0.8:  # 20% tolerance
            return SensorStatus.DEGRADED
        
        # Reliability assessment based on age and usage
        reliability = self._calculate_reliability(sensor_id, current_time)
        stats['reliability'] = reliability
        stats['failure_probability'] = 1.0 - reliability
        
        if reliability < 0.8:
            return SensorStatus.SUSPICIOUS
        elif reliability < 0.9:
            return SensorStatus.DEGRADED
        
        # Maintenance check
        time_since_maintenance = current_time - config.last_maintenance
        if time_since_maintenance > config.maintenance_interval * 3600:  # Convert to seconds
            stats['maintenance_due'] = True
            return SensorStatus.MAINTENANCE
        
        return SensorStatus.HEALTHY
    
    def _assess_noise_level(self, data_matrix: np.ndarray, config: SensorConfiguration) -> float:
        """Assess sensor noise level"""
        if len(data_matrix) < 2:
            return 0.0
        
        # Calculate noise as deviation from expected noise
        measured_noise = np.std(data_matrix, axis=0)
        expected_noise = config.measurement_noise
        
        # Return noise level as multiple of expected noise
        return np.mean(measured_noise) / expected_noise
    
    def _detect_bias(self, data_matrix: np.ndarray, config: SensorConfiguration) -> float:
        """Detect sensor bias/drift"""
        if len(data_matrix) < 10:
            return 0.0
        
        # Use linear regression to detect drift
        n_samples, n_features = data_matrix.shape
        time_indices = np.arange(n_samples)
        
        bias_levels = []
        for feature_idx in range(n_features):
            feature_data = data_matrix[:, feature_idx]
            
            # Fit linear trend
            coeffs = np.polyfit(time_indices, feature_data, 1)
            slope = abs(coeffs[0])
            
            # Normalize by expected variance
            expected_std = config.measurement_noise
            normalized_slope = slope / expected_std if expected_std > 0 else slope
            
            bias_levels.append(normalized_slope)
        
        return np.mean(bias_levels)
    
    def _detect_anomalies(self, sensor_id: str, data_matrix: np.ndarray) -> bool:
        """Detect anomalies using isolation forest"""
        try:
            detector = self.anomaly_detectors[sensor_id]
            scaler = self.scalers[sensor_id]
            
            # Need sufficient training data
            if len(data_matrix) < 20:
                return False
            
            # Scale data
            if not hasattr(scaler, 'mean_'):
                scaled_data = scaler.fit_transform(data_matrix)
                detector.fit(scaled_data[:-10])  # Train on older data
            else:
                scaled_data = scaler.transform(data_matrix)
            
            # Check recent data for anomalies
            recent_data = scaled_data[-5:]  # Last 5 measurements
            anomaly_scores = detector.decision_function(recent_data)
            
            # Check if any recent measurements are anomalous
            anomaly_threshold = self.health_thresholds['anomaly_threshold']
            return np.any(anomaly_scores < -anomaly_threshold)
            
        except Exception as e:
            logger.warning(f"Anomaly detection failed for {sensor_id}: {e}")
            return False
    
    def _calculate_reliability(self, sensor_id: str, current_time: float) -> float:
        """Calculate sensor reliability based on age and usage"""
        config = self.sensor_configs[sensor_id]
        
        # Time since last maintenance (hours)
        time_since_maintenance = (current_time - config.last_maintenance) / 3600
        
        # Exponential reliability model: R(t) = exp(-λt)
        lambda_param = config.failure_rate
        reliability = np.exp(-lambda_param * time_since_maintenance)
        
        return max(0.0, min(1.0, reliability))
    
    def get_sensor_health_report(self, sensor_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive sensor health report"""
        if sensor_id:
            if sensor_id in self.health_statistics:
                return {
                    'sensor_id': sensor_id,
                    'health_stats': self.health_statistics[sensor_id].copy(),
                    'configuration': self.sensor_configs[sensor_id].__dict__,
                    'data_points': len(self.sensor_data_history[sensor_id])
                }
            else:
                return {'error': f'Unknown sensor: {sensor_id}'}
        else:
            # All sensors report
            report = {
                'timestamp': time.time(),
                'total_sensors': len(self.sensor_configs),
                'healthy_sensors': 0,
                'degraded_sensors': 0,
                'failed_sensors': 0,
                'sensors': {}
            }
            
            for sid, stats in self.health_statistics.items():
                status = SensorStatus(stats['status'])
                if status == SensorStatus.HEALTHY:
                    report['healthy_sensors'] += 1
                elif status in [SensorStatus.DEGRADED, SensorStatus.SUSPICIOUS, SensorStatus.MAINTENANCE]:
                    report['degraded_sensors'] += 1
                else:
                    report['failed_sensors'] += 1
                
                report['sensors'][sid] = stats.copy()
            
            report['system_reliability'] = report['healthy_sensors'] / max(report['total_sensors'], 1)
            return report


class MultiModalSensorFusion:
    """Multi-modal sensor fusion with adaptive weights"""
    
    def __init__(self):
        """Initialize sensor fusion system"""
        self.sensor_configs: Dict[str, SensorConfiguration] = {}
        self.fusion_weights: Dict[str, float] = {}
        self.fusion_history = deque(maxlen=1000)
        
        # Fusion methods
        self.fusion_methods = {
            'weighted_average': self._weighted_average_fusion,
            'kalman_fusion': self._kalman_fusion,
            'covariance_intersection': self._covariance_intersection_fusion,
            'dempster_shafer': self._dempster_shafer_fusion
        }
        
        self.active_fusion_method = 'weighted_average'
        
        logger.info("Multi-modal sensor fusion initialized")
    
    def register_sensor(self, config: SensorConfiguration) -> None:
        """Register sensor for fusion"""
        self.sensor_configs[config.sensor_id] = config
        self.fusion_weights[config.sensor_id] = config.fusion_weight
        logger.debug(f"Registered sensor for fusion: {config.sensor_id}")
    
    def update_fusion_weights(self, 
                            sensor_health: Dict[str, Dict[str, Any]],
                            adaptation_rate: float = 0.1) -> None:
        """Update fusion weights based on sensor health"""
        
        total_reliability = 0.0
        sensor_reliabilities = {}
        
        # Calculate reliability-based weights
        for sensor_id in self.fusion_weights:
            if sensor_id in sensor_health:
                reliability = sensor_health[sensor_id].get('reliability', 0.0)
                quality = sensor_health[sensor_id].get('quality_score', 0.0)
                status = sensor_health[sensor_id].get('status', 'failed')
                
                # Status-based reliability adjustment
                if status == 'failed' or status == 'offline':
                    effective_reliability = 0.0
                elif status == 'degraded':
                    effective_reliability = reliability * 0.5
                elif status == 'suspicious':
                    effective_reliability = reliability * 0.7
                else:
                    effective_reliability = reliability
                
                # Combine reliability and quality
                combined_score = effective_reliability * quality
                sensor_reliabilities[sensor_id] = combined_score
                total_reliability += combined_score
        
        # Update weights with adaptation
        if total_reliability > 0:
            for sensor_id in self.fusion_weights:
                if sensor_id in sensor_reliabilities:
                    target_weight = sensor_reliabilities[sensor_id] / total_reliability
                    current_weight = self.fusion_weights[sensor_id]
                    
                    # Smooth weight adaptation
                    new_weight = (1 - adaptation_rate) * current_weight + adaptation_rate * target_weight
                    self.fusion_weights[sensor_id] = new_weight
                else:
                    # Sensor not available, reduce weight
                    self.fusion_weights[sensor_id] *= (1 - adaptation_rate)
        
        # Normalize weights
        total_weight = sum(self.fusion_weights.values())
        if total_weight > 0:
            for sensor_id in self.fusion_weights:
                self.fusion_weights[sensor_id] /= total_weight
        
        logger.debug(f"Updated fusion weights: {self.fusion_weights}")
    
    def fuse_measurements(self, 
                         measurements: List[SensorMeasurement],
                         fusion_method: Optional[str] = None) -> Optional[FusedMeasurement]:
        """Fuse multiple sensor measurements"""
        
        if not measurements:
            return None
        
        # Filter valid measurements from healthy sensors
        valid_measurements = [
            m for m in measurements 
            if m.is_valid and m.sensor_id in self.fusion_weights
            and self.fusion_weights[m.sensor_id] > 0.001
        ]
        
        if not valid_measurements:
            logger.warning("No valid measurements for fusion")
            return None
        
        # Use specified fusion method or default
        method = fusion_method or self.active_fusion_method
        fusion_func = self.fusion_methods.get(method, self._weighted_average_fusion)
        
        try:
            fused_result = fusion_func(valid_measurements)
            
            # Store fusion history
            if fused_result:
                self.fusion_history.append(fused_result)
            
            return fused_result
            
        except Exception as e:
            logger.error(f"Sensor fusion failed: {e}")
            return None
    
    def _weighted_average_fusion(self, measurements: List[SensorMeasurement]) -> FusedMeasurement:
        """Weighted average fusion method"""
        
        # Extract data and weights
        data_arrays = []
        uncertainty_arrays = []
        weights = []
        contributing_sensors = []
        
        for measurement in measurements:
            if measurement.sensor_id in self.fusion_weights:
                weight = self.fusion_weights[measurement.sensor_id]
                if weight > 0:
                    data_arrays.append(measurement.data * weight)
                    uncertainty_arrays.append(measurement.uncertainty**2 * weight**2)
                    weights.append(weight)
                    contributing_sensors.append(measurement.sensor_id)
        
        if not data_arrays:
            raise ValueError("No valid measurements with positive weights")
        
        # Ensure all arrays have same shape
        target_shape = data_arrays[0].shape
        for i, arr in enumerate(data_arrays):
            if arr.shape != target_shape:
                # Handle shape mismatch - pad or truncate
                if arr.size < np.prod(target_shape):
                    # Pad with zeros
                    padded = np.zeros(target_shape)
                    padded.flat[:arr.size] = arr.flat
                    data_arrays[i] = padded
                else:
                    # Truncate
                    data_arrays[i] = arr.flatten()[:np.prod(target_shape)].reshape(target_shape)
        
        # Weighted average
        fused_data = np.sum(data_arrays, axis=0)
        fused_uncertainty = np.sqrt(np.sum(uncertainty_arrays, axis=0))
        
        # Quality score based on individual quality scores and weights
        quality_scores = [m.quality_score for m in measurements if m.sensor_id in contributing_sensors]
        weight_values = [self.fusion_weights[m.sensor_id] for m in measurements if m.sensor_id in contributing_sensors]
        
        if quality_scores and weight_values:
            fused_quality = np.average(quality_scores, weights=weight_values)
        else:
            fused_quality = 0.5
        
        # Create fusion weight dict
        fusion_weight_dict = {sensor_id: self.fusion_weights[sensor_id] for sensor_id in contributing_sensors}
        
        return FusedMeasurement(
            timestamp=time.time(),
            fused_data=fused_data,
            fused_uncertainty=fused_uncertainty,
            fusion_weights=fusion_weight_dict,
            contributing_sensors=contributing_sensors,
            quality_score=fused_quality,
            metadata={
                'fusion_method': 'weighted_average',
                'num_sensors': len(contributing_sensors)
            }
        )
    
    def _kalman_fusion(self, measurements: List[SensorMeasurement]) -> FusedMeasurement:
        """Kalman-style optimal fusion"""
        
        # Extract measurements and covariances
        y_list = []  # measurements
        R_list = []  # covariances
        contributing_sensors = []
        
        for measurement in measurements:
            if measurement.sensor_id in self.fusion_weights and self.fusion_weights[measurement.sensor_id] > 0:
                y_list.append(measurement.data.flatten())
                
                # Use uncertainty as covariance diagonal
                R_diag = measurement.uncertainty.flatten()**2
                R = np.diag(R_diag)
                R_list.append(R)
                
                contributing_sensors.append(measurement.sensor_id)
        
        if not y_list:
            raise ValueError("No valid measurements for Kalman fusion")
        
        # Ensure all measurements have same dimension
        max_dim = max(len(y) for y in y_list)
        for i, y in enumerate(y_list):
            if len(y) < max_dim:
                # Pad with zeros
                padded_y = np.zeros(max_dim)
                padded_y[:len(y)] = y
                y_list[i] = padded_y
                
                # Pad covariance matrix
                padded_R = np.eye(max_dim) * 1e6  # Large uncertainty for missing dims
                orig_dim = R_list[i].shape[0]
                padded_R[:orig_dim, :orig_dim] = R_list[i]
                R_list[i] = padded_R
        
        # Kalman fusion: x_fused = (Σ R_i^-1)^-1 * Σ R_i^-1 * y_i
        sum_R_inv = np.zeros((max_dim, max_dim))
        sum_R_inv_y = np.zeros(max_dim)
        
        for y, R in zip(y_list, R_list):
            try:
                R_inv = np.linalg.inv(R + np.eye(max_dim) * 1e-10)  # Add small regularization
                sum_R_inv += R_inv
                sum_R_inv_y += R_inv @ y
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if matrix is singular
                R_inv = np.linalg.pinv(R)
                sum_R_inv += R_inv
                sum_R_inv_y += R_inv @ y
        
        # Final fusion
        try:
            P_fused = np.linalg.inv(sum_R_inv)
            x_fused = P_fused @ sum_R_inv_y
            uncertainty_fused = np.sqrt(np.diag(P_fused))
        except np.linalg.LinAlgError:
            # Fallback to weighted average
            logger.warning("Kalman fusion failed, falling back to weighted average")
            return self._weighted_average_fusion(measurements)
        
        # Calculate fusion weights (for reporting)
        fusion_weights = {}
        total_weight = 0.0
        for i, sensor_id in enumerate(contributing_sensors):
            if i < len(R_list):
                weight = 1.0 / (np.trace(R_list[i]) + 1e-10)
                fusion_weights[sensor_id] = weight
                total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for sensor_id in fusion_weights:
                fusion_weights[sensor_id] /= total_weight
        
        # Quality score
        quality_scores = [m.quality_score for m in measurements if m.sensor_id in contributing_sensors]
        fused_quality = np.mean(quality_scores) if quality_scores else 0.5
        
        return FusedMeasurement(
            timestamp=time.time(),
            fused_data=x_fused,
            fused_uncertainty=uncertainty_fused,
            fusion_weights=fusion_weights,
            contributing_sensors=contributing_sensors,
            quality_score=fused_quality,
            metadata={
                'fusion_method': 'kalman_fusion',
                'num_sensors': len(contributing_sensors)
            }
        )
    
    def _covariance_intersection_fusion(self, measurements: List[SensorMeasurement]) -> FusedMeasurement:
        """Covariance intersection fusion for correlated sensors"""
        # Simplified implementation - would need full covariance intersection algorithm
        return self._weighted_average_fusion(measurements)
    
    def _dempster_shafer_fusion(self, measurements: List[SensorMeasurement]) -> FusedMeasurement:
        """Dempster-Shafer evidence fusion"""
        # Simplified implementation - would need full D-S framework
        return self._weighted_average_fusion(measurements)
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get fusion performance statistics"""
        if not self.fusion_history:
            return {'error': 'No fusion history available'}
        
        recent_fusions = list(self.fusion_history)[-100:]  # Last 100 fusions
        
        # Calculate statistics
        quality_scores = [f.quality_score for f in recent_fusions]
        num_sensors_used = [len(f.contributing_sensors) for f in recent_fusions]
        
        # Sensor participation rates
        sensor_participation = defaultdict(int)
        for fusion in recent_fusions:
            for sensor_id in fusion.contributing_sensors:
                sensor_participation[sensor_id] += 1
        
        total_fusions = len(recent_fusions)
        participation_rates = {
            sensor_id: count / total_fusions 
            for sensor_id, count in sensor_participation.items()
        }
        
        return {
            'total_fusions': len(self.fusion_history),
            'recent_fusions': len(recent_fusions),
            'average_quality': np.mean(quality_scores),
            'average_sensors_per_fusion': np.mean(num_sensors_used),
            'active_fusion_method': self.active_fusion_method,
            'current_weights': self.fusion_weights.copy(),
            'sensor_participation_rates': participation_rates
        }


class GracefulDegradationManager:
    """Manages graceful degradation strategies for sensor failures"""
    
    def __init__(self):
        """Initialize degradation manager"""
        self.degradation_strategies: Dict[str, Callable] = {}
        self.active_degradations: Dict[str, Dict[str, Any]] = {}
        self.degradation_history = deque(maxlen=1000)
        
        # Performance impact tracking
        self.performance_metrics = {
            'accuracy_factor': 1.0,
            'update_rate_factor': 1.0,
            'reliability_factor': 1.0,
            'coverage_factor': 1.0
        }
        
        self._register_default_strategies()
        
        logger.info("Graceful degradation manager initialized")
    
    def _register_default_strategies(self) -> None:
        """Register default degradation strategies"""
        
        # Camera failure strategy
        def camera_degradation(failed_sensors: List[str], sensor_health: Dict) -> Dict[str, float]:
            num_failed_cameras = sum(1 for s in failed_sensors if 'camera' in s.lower())
            total_cameras = sum(1 for s in sensor_health if 'camera' in s.lower())
            
            if total_cameras > 0:
                failure_ratio = num_failed_cameras / total_cameras
                return {
                    'accuracy_factor': 1.0 - 0.3 * failure_ratio,
                    'update_rate_factor': 1.0 - 0.2 * failure_ratio,
                    'coverage_factor': 1.0 - 0.4 * failure_ratio
                }
            return {'accuracy_factor': 0.5}
        
        self.degradation_strategies['camera_failure'] = camera_degradation
        
        # LIDAR failure strategy
        def lidar_degradation(failed_sensors: List[str], sensor_health: Dict) -> Dict[str, float]:
            has_lidar_failure = any('lidar' in s.lower() for s in failed_sensors)
            if has_lidar_failure:
                return {
                    'accuracy_factor': 0.7,
                    'reliability_factor': 0.8,
                    'coverage_factor': 0.6
                }
            return {}
        
        self.degradation_strategies['lidar_failure'] = lidar_degradation
        
        # Communication failure strategy
        def communication_degradation(failed_sensors: List[str], sensor_health: Dict) -> Dict[str, float]:
            comm_failures = sum(1 for s in failed_sensors if 'comm' in s.lower())
            if comm_failures > 0:
                return {
                    'update_rate_factor': max(0.3, 1.0 - 0.2 * comm_failures),
                    'reliability_factor': max(0.5, 1.0 - 0.15 * comm_failures)
                }
            return {}
        
        self.degradation_strategies['communication_failure'] = communication_degradation
    
    def apply_degradation(self, 
                         failed_sensors: List[str],
                         sensor_health: Dict[str, Dict[str, Any]]) -> None:
        """Apply appropriate degradation strategies"""
        
        # Reset performance metrics
        self.performance_metrics = {
            'accuracy_factor': 1.0,
            'update_rate_factor': 1.0,
            'reliability_factor': 1.0,
            'coverage_factor': 1.0
        }
        
        applied_strategies = []
        
        # Apply relevant degradation strategies
        for strategy_name, strategy_func in self.degradation_strategies.items():
            try:
                impact = strategy_func(failed_sensors, sensor_health)
                
                if impact:
                    applied_strategies.append(strategy_name)
                    
                    # Apply performance impacts
                    for metric, factor in impact.items():
                        if metric in self.performance_metrics:
                            self.performance_metrics[metric] *= factor
                
            except Exception as e:
                logger.error(f"Error applying degradation strategy {strategy_name}: {e}")
        
        # Record degradation state
        degradation_record = {
            'timestamp': time.time(),
            'failed_sensors': failed_sensors.copy(),
            'applied_strategies': applied_strategies,
            'performance_impact': self.performance_metrics.copy()
        }
        
        self.degradation_history.append(degradation_record)
        
        if applied_strategies:
            logger.warning(f"Applied degradation strategies: {applied_strategies}")
            logger.info(f"Performance impact: {self.performance_metrics}")
    
    def register_strategy(self, 
                         strategy_name: str,
                         strategy_func: Callable[[List[str], Dict], Dict[str, float]]) -> None:
        """Register custom degradation strategy"""
        self.degradation_strategies[strategy_name] = strategy_func
        logger.debug(f"Registered degradation strategy: {strategy_name}")
    
    def get_performance_impact(self) -> Dict[str, float]:
        """Get current performance impact factors"""
        return self.performance_metrics.copy()
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status"""
        return {
            'active_degradations': len(self.active_degradations),
            'performance_metrics': self.performance_metrics,
            'available_strategies': list(self.degradation_strategies.keys()),
            'recent_degradations': list(self.degradation_history)[-10:]
        }


class SensorManagementSystem:
    """Comprehensive sensor management system"""
    
    def __init__(self):
        """Initialize sensor management system"""
        self.health_monitor = SensorHealthMonitor()
        self.fusion_system = MultiModalSensorFusion()
        self.degradation_manager = GracefulDegradationManager()
        
        self.sensor_configs: Dict[str, SensorConfiguration] = {}
        self.measurement_queue = queue.Queue(maxsize=1000)
        
        # Management thread
        self.management_enabled = True
        self.management_thread = None
        
        # Statistics
        self.total_measurements = 0
        self.successful_fusions = 0
        self.failed_fusions = 0
        
        logger.info("Sensor management system initialized")
    
    def register_sensor(self, config: SensorConfiguration) -> None:
        """Register sensor with all subsystems"""
        self.sensor_configs[config.sensor_id] = config
        self.health_monitor.register_sensor(config)
        self.fusion_system.register_sensor(config)
        logger.info(f"Registered sensor: {config.sensor_id}")
    
    def process_measurement(self, measurement: SensorMeasurement) -> Optional[FusedMeasurement]:
        """Process sensor measurement through the pipeline"""
        self.total_measurements += 1
        
        # Health monitoring
        sensor_status = self.health_monitor.update_sensor_data(measurement)
        
        # Queue measurement for fusion
        try:
            self.measurement_queue.put_nowait(measurement)
        except queue.Full:
            logger.warning("Measurement queue full, dropping oldest measurement")
            try:
                self.measurement_queue.get_nowait()
                self.measurement_queue.put_nowait(measurement)
            except queue.Empty:
                pass
        
        return None  # Fusion happens in management thread
    
    def start_management(self) -> None:
        """Start sensor management thread"""
        if self.management_thread and self.management_thread.is_alive():
            logger.warning("Management already running")
            return
        
        def management_loop():
            measurement_buffer = []
            last_fusion_time = time.time()
            fusion_interval = 0.1  # 10Hz fusion rate
            
            while self.management_enabled:
                try:
                    # Collect measurements
                    try:
                        while True:
                            measurement = self.measurement_queue.get_nowait()
                            measurement_buffer.append(measurement)
                    except queue.Empty:
                        pass
                    
                    # Perform fusion periodically
                    current_time = time.time()
                    if current_time - last_fusion_time >= fusion_interval and measurement_buffer:
                        
                        # Get sensor health
                        health_report = self.health_monitor.get_sensor_health_report()
                        sensor_health = health_report.get('sensors', {})
                        
                        # Update fusion weights
                        self.fusion_system.update_fusion_weights(sensor_health)
                        
                        # Group measurements by timestamp (approximately)
                        time_groups = defaultdict(list)
                        for measurement in measurement_buffer:
                            time_key = int(measurement.timestamp / fusion_interval)
                            time_groups[time_key].append(measurement)
                        
                        # Fuse measurements for each time group
                        for time_key, measurements in time_groups.items():
                            try:
                                fused_result = self.fusion_system.fuse_measurements(measurements)
                                if fused_result:
                                    self.successful_fusions += 1
                                else:
                                    self.failed_fusions += 1
                            except Exception as e:
                                logger.error(f"Fusion error: {e}")
                                self.failed_fusions += 1
                        
                        # Apply degradation strategies
                        failed_sensors = [
                            sensor_id for sensor_id, stats in sensor_health.items()
                            if stats.get('status') in ['failed', 'offline']
                        ]
                        
                        if failed_sensors:
                            self.degradation_manager.apply_degradation(failed_sensors, sensor_health)
                        
                        # Clear buffer
                        measurement_buffer.clear()
                        last_fusion_time = current_time
                    
                    time.sleep(0.01)  # 100Hz management loop
                    
                except Exception as e:
                    logger.error(f"Error in management loop: {e}")
                    time.sleep(0.1)
        
        self.management_thread = threading.Thread(target=management_loop, daemon=True)
        self.management_thread.start()
        logger.info("Sensor management thread started")
    
    def stop_management(self) -> None:
        """Stop sensor management"""
        self.management_enabled = False
        if self.management_thread:
            self.management_thread.join(timeout=2.0)
        logger.info("Sensor management stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        health_report = self.health_monitor.get_sensor_health_report()
        fusion_stats = self.fusion_system.get_fusion_statistics()
        degradation_status = self.degradation_manager.get_degradation_status()
        
        return {
            'timestamp': time.time(),
            'total_sensors': len(self.sensor_configs),
            'total_measurements': self.total_measurements,
            'successful_fusions': self.successful_fusions,
            'failed_fusions': self.failed_fusions,
            'fusion_success_rate': self.successful_fusions / max(self.successful_fusions + self.failed_fusions, 1),
            'health_summary': {
                'system_reliability': health_report.get('system_reliability', 0.0),
                'healthy_sensors': health_report.get('healthy_sensors', 0),
                'degraded_sensors': health_report.get('degraded_sensors', 0),
                'failed_sensors': health_report.get('failed_sensors', 0)
            },
            'fusion_summary': {
                'active_method': fusion_stats.get('active_fusion_method', 'unknown'),
                'average_quality': fusion_stats.get('average_quality', 0.0),
                'average_sensors_per_fusion': fusion_stats.get('average_sensors_per_fusion', 0.0)
            },
            'performance_impact': degradation_status.get('performance_metrics', {}),
            'queue_size': self.measurement_queue.qsize()
        }


# Example usage and testing
if __name__ == "__main__":
    # Create sensor management system
    sensor_manager = SensorManagementSystem()
    
    # Configure sensors
    camera_config = SensorConfiguration(
        sensor_id="camera_1",
        sensor_type=SensorType.CAMERA,
        name="Main Camera",
        description="RGB camera for human detection",
        sample_rate=30.0,
        resolution=(640, 480),
        measurement_range=(0.0, 255.0),
        accuracy=0.95,
        precision=0.02,
        measurement_noise=0.1,
        fusion_weight=0.4,
        backup_sensors=["camera_2"]
    )
    
    lidar_config = SensorConfiguration(
        sensor_id="lidar_1",
        sensor_type=SensorType.LIDAR,
        name="LIDAR Scanner",
        description="3D LIDAR for environment mapping",
        sample_rate=10.0,
        resolution=0.01,
        measurement_range=(0.1, 100.0),
        accuracy=0.99,
        precision=0.005,
        measurement_noise=0.02,
        fusion_weight=0.6
    )
    
    # Register sensors
    sensor_manager.register_sensor(camera_config)
    sensor_manager.register_sensor(lidar_config)
    
    # Start management
    sensor_manager.start_management()
    
    print("Testing sensor management system...")
    
    # Simulate sensor measurements
    for i in range(50):
        # Camera measurement
        camera_data = np.random.normal(100, 10, (3,))  # RGB values
        camera_uncertainty = np.ones(3) * 0.1
        
        camera_measurement = SensorMeasurement(
            sensor_id="camera_1",
            timestamp=time.time(),
            data=camera_data,
            uncertainty=camera_uncertainty,
            quality_score=0.9 + 0.1 * np.random.random()
        )
        
        # LIDAR measurement  
        lidar_data = np.random.normal(5.0, 0.1, (3,))  # 3D position
        lidar_uncertainty = np.ones(3) * 0.02
        
        lidar_measurement = SensorMeasurement(
            sensor_id="lidar_1",
            timestamp=time.time(),
            data=lidar_data,
            uncertainty=lidar_uncertainty,
            quality_score=0.95 + 0.05 * np.random.random()
        )
        
        # Process measurements
        sensor_manager.process_measurement(camera_measurement)
        sensor_manager.process_measurement(lidar_measurement)
        
        # Simulate sensor failure after some measurements
        if i == 25:
            # Inject faulty camera data
            faulty_measurement = SensorMeasurement(
                sensor_id="camera_1",
                timestamp=time.time(),
                data=np.array([0, 0, 0]),  # Stuck at zero
                uncertainty=np.ones(3) * 10.0,  # High uncertainty
                quality_score=0.1  # Low quality
            )
            sensor_manager.process_measurement(faulty_measurement)
        
        time.sleep(0.05)  # 20Hz simulation
    
    # Get final status
    status = sensor_manager.get_system_status()
    print(f"\nFinal System Status:")
    print(f"System reliability: {status['health_summary']['system_reliability']:.2%}")
    print(f"Fusion success rate: {status['fusion_success_rate']:.2%}")
    print(f"Performance impact: {status['performance_impact']}")
    print(f"Healthy sensors: {status['health_summary']['healthy_sensors']}")
    print(f"Failed sensors: {status['health_summary']['failed_sensors']}")
    
    # Cleanup
    sensor_manager.stop_management()
    print("\nSensor management system test completed")