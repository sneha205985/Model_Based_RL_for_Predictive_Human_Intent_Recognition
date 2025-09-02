"""
Data validation and preprocessing utilities for human behavior modeling.

This module provides comprehensive validation, preprocessing, and quality
assessment tools for trajectory data and human behavior sequences.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer

from ..data.synthetic_generator import SyntheticSequence, GestureType
from ..models.human_behavior import HumanState, BehaviorType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ValidationLevel(Enum):
    """Levels of data validation strictness."""
    BASIC = "basic"           # Basic checks (NaN, shape, range)
    STANDARD = "standard"     # Standard checks + statistical tests
    STRICT = "strict"         # All checks + outlier detection + consistency


class DataQualityIssue(Enum):
    """Types of data quality issues."""
    MISSING_VALUES = "missing_values"
    OUTLIERS = "outliers"
    INVALID_RANGE = "invalid_range"
    INCONSISTENT_SAMPLING = "inconsistent_sampling"
    INSUFFICIENT_DATA = "insufficient_data"
    CORRUPT_TRAJECTORY = "corrupt_trajectory"
    TEMPORAL_GAPS = "temporal_gaps"
    UNREALISTIC_VALUES = "unrealistic_values"


@dataclass
class ValidationResult:
    """
    Result of data validation process.
    
    Attributes:
        is_valid: Whether data passes validation
        issues: List of detected issues
        quality_score: Overall quality score (0-1)
        statistics: Validation statistics
        recommendations: Suggested fixes
    """
    is_valid: bool
    issues: List[DataQualityIssue]
    quality_score: float
    statistics: Dict[str, Any]
    recommendations: List[str]
    
    def __post_init__(self) -> None:
        """Ensure quality score is in valid range."""
        self.quality_score = np.clip(self.quality_score, 0.0, 1.0)


@dataclass
class PreprocessingConfig:
    """
    Configuration for data preprocessing.
    
    Attributes:
        interpolation_method: Method for filling gaps ('linear', 'cubic', 'akima')
        outlier_method: Outlier detection method ('iqr', 'zscore', 'isolation')
        outlier_threshold: Threshold for outlier detection
        smoothing_method: Smoothing method ('savgol', 'gaussian', 'moving_average')
        smoothing_params: Parameters for smoothing
        normalization_method: Normalization method ('standard', 'minmax', 'robust')
        imputation_method: Method for missing value imputation ('mean', 'median', 'knn')
        validate_after_preprocessing: Whether to validate after preprocessing
    """
    interpolation_method: str = 'linear'
    outlier_method: str = 'iqr'
    outlier_threshold: float = 2.0
    smoothing_method: str = 'savgol'
    smoothing_params: Dict[str, Any] = None
    normalization_method: str = 'standard'
    imputation_method: str = 'median'
    validate_after_preprocessing: bool = True
    
    def __post_init__(self) -> None:
        """Set default smoothing parameters."""
        if self.smoothing_params is None:
            self.smoothing_params = {'window_length': 5, 'polyorder': 2}


class DataValidator:
    """
    Comprehensive data validator for human behavior sequences.
    
    This class implements various validation checks for trajectory data,
    including statistical tests, outlier detection, and consistency checks.
    """
    
    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        workspace_bounds: Optional[np.ndarray] = None,
        expected_sampling_rate: float = 30.0
    ):
        """
        Initialize data validator.
        
        Args:
            validation_level: Level of validation strictness
            workspace_bounds: Expected workspace bounds [x_min, x_max, y_min, y_max, z_min, z_max]
            expected_sampling_rate: Expected data sampling rate (Hz)
        """
        self.validation_level = validation_level
        self.workspace_bounds = workspace_bounds
        self.expected_sampling_rate = expected_sampling_rate
        
        # Validation thresholds
        self.max_velocity = 2.0        # m/s
        self.max_acceleration = 10.0   # m/s²
        self.min_sequence_length = 5
        self.max_temporal_gap = 0.5    # seconds
        
        logger.info(f"Initialized data validator with {validation_level.value} level")
    
    def validate_sequence(self, sequence: SyntheticSequence) -> ValidationResult:
        """
        Validate a single synthetic sequence.
        
        Args:
            sequence: Synthetic sequence to validate
            
        Returns:
            Validation result
        """
        issues = []
        statistics = {}
        recommendations = []
        
        # Basic validation
        basic_issues, basic_stats = self._validate_basic(sequence)
        issues.extend(basic_issues)
        statistics.update(basic_stats)
        
        if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            # Standard validation
            std_issues, std_stats = self._validate_standard(sequence)
            issues.extend(std_issues)
            statistics.update(std_stats)
        
        if self.validation_level == ValidationLevel.STRICT:
            # Strict validation
            strict_issues, strict_stats = self._validate_strict(sequence)
            issues.extend(strict_issues)
            statistics.update(strict_stats)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues)
        
        # Compute quality score
        quality_score = self._compute_quality_score(issues, statistics)
        
        # Determine if valid
        is_valid = self._determine_validity(issues, quality_score)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            quality_score=quality_score,
            statistics=statistics,
            recommendations=recommendations
        )
    
    def _validate_basic(
        self, 
        sequence: SyntheticSequence
    ) -> Tuple[List[DataQualityIssue], Dict[str, Any]]:
        """Perform basic validation checks."""
        issues = []
        stats = {}
        
        # Check for missing or invalid data
        trajectory = sequence.hand_trajectory
        timestamps = sequence.timestamps
        
        # Check for NaN values
        if np.any(np.isnan(trajectory)):
            issues.append(DataQualityIssue.MISSING_VALUES)
        
        if np.any(np.isnan(timestamps)):
            issues.append(DataQualityIssue.MISSING_VALUES)
        
        # Check sequence length
        if len(trajectory) < self.min_sequence_length:
            issues.append(DataQualityIssue.INSUFFICIENT_DATA)
        
        stats['sequence_length'] = len(trajectory)
        stats['duration'] = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        
        # Check shape consistency
        if trajectory.shape[1] != 3:
            issues.append(DataQualityIssue.CORRUPT_TRAJECTORY)
        
        if len(trajectory) != len(timestamps):
            issues.append(DataQualityIssue.CORRUPT_TRAJECTORY)
        
        # Check workspace bounds
        if self.workspace_bounds is not None:
            x_bounds = self.workspace_bounds[:2]
            y_bounds = self.workspace_bounds[2:4]
            z_bounds = self.workspace_bounds[4:6]
            
            x_violations = np.sum((trajectory[:, 0] < x_bounds[0]) | (trajectory[:, 0] > x_bounds[1]))
            y_violations = np.sum((trajectory[:, 1] < y_bounds[0]) | (trajectory[:, 1] > y_bounds[1]))
            z_violations = np.sum((trajectory[:, 2] < z_bounds[0]) | (trajectory[:, 2] > z_bounds[1]))
            
            total_violations = x_violations + y_violations + z_violations
            if total_violations > 0:
                issues.append(DataQualityIssue.INVALID_RANGE)
                stats['workspace_violations'] = int(total_violations)
        
        # Check temporal consistency
        if len(timestamps) > 1:
            time_diffs = np.diff(timestamps)
            if np.any(time_diffs <= 0):
                issues.append(DataQualityIssue.INCONSISTENT_SAMPLING)
            
            # Check for large gaps
            if np.any(time_diffs > self.max_temporal_gap):
                issues.append(DataQualityIssue.TEMPORAL_GAPS)
            
            stats['sampling_rate'] = 1.0 / np.mean(time_diffs) if np.mean(time_diffs) > 0 else 0
            stats['sampling_std'] = np.std(time_diffs)
        
        return issues, stats
    
    def _validate_standard(
        self,
        sequence: SyntheticSequence
    ) -> Tuple[List[DataQualityIssue], Dict[str, Any]]:
        """Perform standard validation checks."""
        issues = []
        stats = {}
        
        trajectory = sequence.hand_trajectory
        timestamps = sequence.timestamps
        
        if len(trajectory) < 2:
            return issues, stats
        
        # Compute derivatives
        dt = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 1.0
        velocities = np.diff(trajectory, axis=0) / dt
        speeds = np.linalg.norm(velocities, axis=1)
        
        if len(velocities) > 1:
            accelerations = np.diff(velocities, axis=0) / dt
            acc_magnitudes = np.linalg.norm(accelerations, axis=1)
        else:
            accelerations = np.array([]).reshape(0, 3)
            acc_magnitudes = np.array([])
        
        # Check for unrealistic velocities
        if np.any(speeds > self.max_velocity):
            issues.append(DataQualityIssue.UNREALISTIC_VALUES)
            stats['max_speed'] = float(np.max(speeds))
            stats['speed_violations'] = int(np.sum(speeds > self.max_velocity))
        
        # Check for unrealistic accelerations
        if len(acc_magnitudes) > 0 and np.any(acc_magnitudes > self.max_acceleration):
            issues.append(DataQualityIssue.UNREALISTIC_VALUES)
            stats['max_acceleration'] = float(np.max(acc_magnitudes))
            stats['acceleration_violations'] = int(np.sum(acc_magnitudes > self.max_acceleration))
        
        # Statistical checks
        stats['mean_speed'] = float(np.mean(speeds))
        stats['std_speed'] = float(np.std(speeds))
        
        if len(acc_magnitudes) > 0:
            stats['mean_acceleration'] = float(np.mean(acc_magnitudes))
            stats['std_acceleration'] = float(np.std(acc_magnitudes))
        
        # Check for statistical anomalies
        # Position statistics
        for dim, name in enumerate(['x', 'y', 'z']):
            pos_dim = trajectory[:, dim]
            stats[f'pos_{name}_mean'] = float(np.mean(pos_dim))
            stats[f'pos_{name}_std'] = float(np.std(pos_dim))
            stats[f'pos_{name}_range'] = float(np.ptp(pos_dim))
            
            # Check for excessive variance
            if np.std(pos_dim) > 1.0:  # More than 1 meter std
                issues.append(DataQualityIssue.UNREALISTIC_VALUES)
        
        return issues, stats
    
    def _validate_strict(
        self,
        sequence: SyntheticSequence
    ) -> Tuple[List[DataQualityIssue], Dict[str, Any]]:
        """Perform strict validation checks."""
        issues = []
        stats = {}
        
        trajectory = sequence.hand_trajectory
        timestamps = sequence.timestamps
        
        if len(trajectory) < 10:  # Need sufficient data for strict validation
            return issues, stats
        
        # Outlier detection for positions
        for dim in range(3):
            pos_dim = trajectory[:, dim]
            outliers = self._detect_outliers(pos_dim, method='iqr')
            if len(outliers) > len(pos_dim) * 0.05:  # More than 5% outliers
                issues.append(DataQualityIssue.OUTLIERS)
                stats[f'outliers_dim_{dim}'] = len(outliers)
        
        # Check trajectory smoothness
        smoothness_score = self._compute_smoothness_score(trajectory, timestamps)
        stats['smoothness_score'] = float(smoothness_score)
        
        if smoothness_score < 0.3:  # Low smoothness threshold
            issues.append(DataQualityIssue.UNREALISTIC_VALUES)
        
        # Check gesture consistency with label
        gesture_consistency = self._check_gesture_consistency(sequence)
        stats['gesture_consistency'] = float(gesture_consistency)
        
        if gesture_consistency < 0.5:
            issues.append(DataQualityIssue.INCONSISTENT_SAMPLING)
        
        # Check for periodic patterns (for gestures like wave)
        if sequence.gesture_type in [GestureType.WAVE]:
            periodicity_score = self._check_periodicity(trajectory, timestamps)
            stats['periodicity_score'] = float(periodicity_score)
            
            if periodicity_score < 0.3:
                issues.append(DataQualityIssue.UNREALISTIC_VALUES)
        
        return issues, stats
    
    def _detect_outliers(
        self,
        data: np.ndarray,
        method: str = 'iqr',
        threshold: float = 2.0
    ) -> np.ndarray:
        """
        Detect outliers in data using specified method.
        
        Args:
            data: Input data array
            method: Outlier detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Array of outlier indices
        """
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers = np.where(z_scores > threshold)[0]
            
        elif method == 'modified_zscore':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            outliers = np.where(np.abs(modified_z_scores) > threshold)[0]
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return outliers
    
    def _compute_smoothness_score(
        self,
        trajectory: np.ndarray,
        timestamps: np.ndarray
    ) -> float:
        """
        Compute trajectory smoothness score.
        
        Uses spectral arc length as smoothness metric:
        Lower values indicate smoother trajectories.
        
        Args:
            trajectory: Position trajectory [N, 3]
            timestamps: Time stamps [N]
            
        Returns:
            Smoothness score (0-1, higher is smoother)
        """
        if len(trajectory) < 10:
            return 1.0  # Too short to assess
        
        dt = np.mean(np.diff(timestamps))
        velocities = np.diff(trajectory, axis=0) / dt
        speeds = np.linalg.norm(velocities, axis=1)
        
        if len(speeds) < 5:
            return 1.0
        
        try:
            # Normalize speed profile
            speed_mean = np.mean(speeds)
            speed_std = np.std(speeds)
            
            if speed_std < 1e-8:
                return 1.0  # Constant speed is perfectly smooth
            
            speed_norm = (speeds - speed_mean) / speed_std
            
            # Compute power spectral density
            freqs = np.fft.fftfreq(len(speed_norm), dt)
            fft_speed = np.fft.fft(speed_norm)
            power_spectrum = np.abs(fft_speed) ** 2
            
            # Spectral arc length
            spectral_arc_length = np.sum(np.abs(np.diff(power_spectrum)))
            
            # Normalize to [0, 1] range (higher = smoother)
            max_sal = len(speeds) * np.var(speeds)  # Rough upper bound
            smoothness_score = 1.0 - min(spectral_arc_length / max_sal, 1.0)
            
            return max(0.0, smoothness_score)
            
        except Exception as e:
            logger.warning(f"Smoothness computation failed: {e}")
            return 0.5  # Default moderate score
    
    def _check_gesture_consistency(self, sequence: SyntheticSequence) -> float:
        """
        Check consistency between gesture type and trajectory characteristics.
        
        Args:
            sequence: Synthetic sequence to check
            
        Returns:
            Consistency score (0-1)
        """
        trajectory = sequence.hand_trajectory
        gesture_type = sequence.gesture_type
        
        if len(trajectory) < 3:
            return 1.0
        
        # Compute trajectory characteristics
        displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
        path_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        efficiency = displacement / (path_length + 1e-8)
        
        # Expected characteristics for each gesture type
        expected = {
            GestureType.WAVE: {'min_path_length': 0.3, 'max_efficiency': 0.5},
            GestureType.POINT: {'min_displacement': 0.2, 'min_efficiency': 0.7},
            GestureType.GRAB: {'min_displacement': 0.15, 'min_efficiency': 0.6},
            GestureType.HANDOVER: {'min_displacement': 0.2, 'min_efficiency': 0.5},
            GestureType.REACH: {'min_displacement': 0.1, 'min_efficiency': 0.6},
            GestureType.IDLE: {'max_displacement': 0.1, 'max_path_length': 0.2}
        }
        
        if gesture_type not in expected:
            return 0.5  # Unknown gesture type
        
        expectations = expected[gesture_type]
        score = 1.0
        
        # Check expectations
        for metric, threshold in expectations.items():
            if metric == 'min_displacement' and displacement < threshold:
                score *= 0.5
            elif metric == 'max_displacement' and displacement > threshold:
                score *= 0.5
            elif metric == 'min_path_length' and path_length < threshold:
                score *= 0.5
            elif metric == 'max_path_length' and path_length > threshold:
                score *= 0.5
            elif metric == 'min_efficiency' and efficiency < threshold:
                score *= 0.7
            elif metric == 'max_efficiency' and efficiency > threshold:
                score *= 0.7
        
        return max(0.0, score)
    
    def _check_periodicity(
        self,
        trajectory: np.ndarray,
        timestamps: np.ndarray
    ) -> float:
        """
        Check for periodic patterns in trajectory (for wave gestures).
        
        Args:
            trajectory: Position trajectory [N, 3]
            timestamps: Time stamps [N]
            
        Returns:
            Periodicity score (0-1)
        """
        if len(trajectory) < 20:
            return 1.0
        
        try:
            # Use y-dimension (side-to-side) for wave detection
            y_signal = trajectory[:, 1]
            
            # Remove trend
            y_detrended = y_signal - np.linspace(y_signal[0], y_signal[-1], len(y_signal))
            
            # Compute autocorrelation
            autocorr = np.correlate(y_detrended, y_detrended, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find peaks in autocorrelation (excluding zero lag)
            if len(autocorr) > 10:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(autocorr[1:], height=0.3)
                
                if len(peaks) > 0:
                    # Strongest periodic component
                    strongest_peak = peaks[np.argmax(autocorr[peaks + 1])]
                    periodicity_strength = autocorr[strongest_peak + 1]
                    return min(1.0, periodicity_strength * 2)  # Scale to [0, 1]
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Periodicity check failed: {e}")
            return 0.5
    
    def _compute_quality_score(
        self,
        issues: List[DataQualityIssue],
        statistics: Dict[str, Any]
    ) -> float:
        """
        Compute overall data quality score.
        
        Args:
            issues: List of detected issues
            statistics: Validation statistics
            
        Returns:
            Quality score (0-1, higher is better)
        """
        # Base score
        score = 1.0
        
        # Penalize issues based on severity
        issue_penalties = {
            DataQualityIssue.MISSING_VALUES: 0.3,
            DataQualityIssue.CORRUPT_TRAJECTORY: 0.4,
            DataQualityIssue.INSUFFICIENT_DATA: 0.2,
            DataQualityIssue.UNREALISTIC_VALUES: 0.2,
            DataQualityIssue.INVALID_RANGE: 0.15,
            DataQualityIssue.INCONSISTENT_SAMPLING: 0.1,
            DataQualityIssue.TEMPORAL_GAPS: 0.1,
            DataQualityIssue.OUTLIERS: 0.05
        }
        
        for issue in issues:
            penalty = issue_penalties.get(issue, 0.1)
            score -= penalty
        
        # Bonus for good statistics
        if 'smoothness_score' in statistics:
            score += 0.1 * statistics['smoothness_score']
        
        if 'gesture_consistency' in statistics:
            score += 0.1 * statistics['gesture_consistency']
        
        return max(0.0, min(1.0, score))
    
    def _determine_validity(
        self,
        issues: List[DataQualityIssue],
        quality_score: float
    ) -> bool:
        """
        Determine if data is valid based on issues and quality score.
        
        Args:
            issues: List of detected issues
            quality_score: Overall quality score
            
        Returns:
            True if data is valid
        """
        # Critical issues that make data invalid
        critical_issues = {
            DataQualityIssue.MISSING_VALUES,
            DataQualityIssue.CORRUPT_TRAJECTORY,
            DataQualityIssue.INSUFFICIENT_DATA
        }
        
        # Check for critical issues
        if any(issue in critical_issues for issue in issues):
            return False
        
        # Check quality score threshold
        min_quality_thresholds = {
            ValidationLevel.BASIC: 0.5,
            ValidationLevel.STANDARD: 0.6,
            ValidationLevel.STRICT: 0.7
        }
        
        min_quality = min_quality_thresholds.get(self.validation_level, 0.6)
        return quality_score >= min_quality
    
    def _generate_recommendations(
        self,
        issues: List[DataQualityIssue]
    ) -> List[str]:
        """Generate recommendations to fix detected issues."""
        recommendations = []
        
        issue_fixes = {
            DataQualityIssue.MISSING_VALUES: "Use interpolation or imputation to fill missing values",
            DataQualityIssue.OUTLIERS: "Apply outlier detection and removal/correction",
            DataQualityIssue.INVALID_RANGE: "Check sensor calibration and workspace boundaries",
            DataQualityIssue.INCONSISTENT_SAMPLING: "Resample data to consistent time intervals",
            DataQualityIssue.INSUFFICIENT_DATA: "Collect more data points for this sequence",
            DataQualityIssue.CORRUPT_TRAJECTORY: "Check data collection pipeline for errors",
            DataQualityIssue.TEMPORAL_GAPS: "Use interpolation to fill temporal gaps",
            DataQualityIssue.UNREALISTIC_VALUES: "Apply smoothing and outlier removal"
        }
        
        for issue in set(issues):  # Remove duplicates
            if issue in issue_fixes:
                recommendations.append(issue_fixes[issue])
        
        if not recommendations:
            recommendations.append("Data quality is acceptable")
        
        return recommendations
    
    def validate_batch(
        self,
        sequences: List[SyntheticSequence]
    ) -> Tuple[List[ValidationResult], Dict[str, Any]]:
        """
        Validate a batch of sequences.
        
        Args:
            sequences: List of sequences to validate
            
        Returns:
            Tuple of (individual results, batch statistics)
        """
        results = []
        batch_stats = {
            'total_sequences': len(sequences),
            'valid_sequences': 0,
            'invalid_sequences': 0,
            'average_quality': 0.0,
            'common_issues': {},
            'quality_distribution': []
        }
        
        logger.info(f"Validating batch of {len(sequences)} sequences...")
        
        for i, sequence in enumerate(sequences):
            try:
                result = self.validate_sequence(sequence)
                results.append(result)
                
                if result.is_valid:
                    batch_stats['valid_sequences'] += 1
                else:
                    batch_stats['invalid_sequences'] += 1
                
                batch_stats['quality_distribution'].append(result.quality_score)
                
                # Track common issues
                for issue in result.issues:
                    issue_name = issue.value
                    if issue_name not in batch_stats['common_issues']:
                        batch_stats['common_issues'][issue_name] = 0
                    batch_stats['common_issues'][issue_name] += 1
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Validated {i + 1}/{len(sequences)} sequences")
                    
            except Exception as e:
                logger.error(f"Validation failed for sequence {sequence.sequence_id}: {e}")
                # Add failed validation result
                results.append(ValidationResult(
                    is_valid=False,
                    issues=[DataQualityIssue.CORRUPT_TRAJECTORY],
                    quality_score=0.0,
                    statistics={'error': str(e)},
                    recommendations=['Check data format and content']
                ))
                batch_stats['invalid_sequences'] += 1
                batch_stats['quality_distribution'].append(0.0)
        
        # Compute batch statistics
        if batch_stats['quality_distribution']:
            batch_stats['average_quality'] = np.mean(batch_stats['quality_distribution'])
            batch_stats['quality_std'] = np.std(batch_stats['quality_distribution'])
            batch_stats['quality_min'] = np.min(batch_stats['quality_distribution'])
            batch_stats['quality_max'] = np.max(batch_stats['quality_distribution'])
        
        valid_fraction = batch_stats['valid_sequences'] / len(sequences)
        logger.info(f"Batch validation complete: {valid_fraction:.2%} valid sequences")
        
        return results, batch_stats


class DataPreprocessor:
    """
    Data preprocessor for cleaning and preparing human behavior data.
    
    This class implements various preprocessing steps including interpolation,
    outlier removal, smoothing, and normalization.
    """
    
    def __init__(self, config: PreprocessingConfig = None):
        """
        Initialize data preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config if config is not None else PreprocessingConfig()
        self.validator = DataValidator(ValidationLevel.BASIC)
        
        # Initialize scalers
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        self.is_fitted = False
        
        logger.info("Initialized data preprocessor")
    
    def preprocess_sequence(self, sequence: SyntheticSequence) -> SyntheticSequence:
        """
        Preprocess a single sequence.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Preprocessed sequence
        """
        # Create copy to avoid modifying original
        processed_sequence = SyntheticSequence(
            sequence_id=sequence.sequence_id + "_processed",
            gesture_type=sequence.gesture_type,
            hand_trajectory=sequence.hand_trajectory.copy(),
            gaze_trajectory=sequence.gaze_trajectory.copy(),
            timestamps=sequence.timestamps.copy(),
            intent_labels=sequence.intent_labels.copy(),
            context_info=sequence.context_info.copy(),
            noise_level=sequence.noise_level
        )
        
        # Apply preprocessing steps
        processed_sequence.hand_trajectory = self._interpolate_missing_values(
            processed_sequence.hand_trajectory,
            processed_sequence.timestamps
        )
        
        processed_sequence.hand_trajectory = self._remove_outliers(
            processed_sequence.hand_trajectory
        )
        
        processed_sequence.hand_trajectory = self._smooth_trajectory(
            processed_sequence.hand_trajectory,
            processed_sequence.timestamps
        )
        
        # Validate after preprocessing if requested
        if self.config.validate_after_preprocessing:
            result = self.validator.validate_sequence(processed_sequence)
            if not result.is_valid:
                logger.warning(f"Sequence {sequence.sequence_id} still invalid after preprocessing")
        
        return processed_sequence
    
    def _interpolate_missing_values(
        self,
        trajectory: np.ndarray,
        timestamps: np.ndarray
    ) -> np.ndarray:
        """Interpolate missing values in trajectory."""
        if not np.any(np.isnan(trajectory)):
            return trajectory
        
        from scipy.interpolate import interp1d
        
        # Find valid points
        valid_mask = ~np.isnan(trajectory).any(axis=1)
        
        if np.sum(valid_mask) < 2:
            logger.warning("Too few valid points for interpolation")
            return trajectory
        
        valid_times = timestamps[valid_mask]
        valid_positions = trajectory[valid_mask]
        
        # Interpolate each dimension
        interpolated = trajectory.copy()
        
        for dim in range(trajectory.shape[1]):
            if self.config.interpolation_method == 'linear':
                interp_func = interp1d(
                    valid_times, valid_positions[:, dim], 
                    kind='linear', bounds_error=False, fill_value='extrapolate'
                )
            elif self.config.interpolation_method == 'cubic':
                interp_func = interp1d(
                    valid_times, valid_positions[:, dim],
                    kind='cubic', bounds_error=False, fill_value='extrapolate'
                )
            else:
                # Default to linear
                interp_func = interp1d(
                    valid_times, valid_positions[:, dim],
                    kind='linear', bounds_error=False, fill_value='extrapolate'
                )
            
            interpolated[:, dim] = interp_func(timestamps)
        
        return interpolated
    
    def _remove_outliers(self, trajectory: np.ndarray) -> np.ndarray:
        """Remove or correct outliers in trajectory."""
        cleaned = trajectory.copy()
        
        for dim in range(trajectory.shape[1]):
            data = trajectory[:, dim]
            
            if self.config.outlier_method == 'iqr':
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.config.outlier_threshold * IQR
                upper_bound = Q3 + self.config.outlier_threshold * IQR
                
                # Clip outliers
                cleaned[:, dim] = np.clip(data, lower_bound, upper_bound)
                
            elif self.config.outlier_method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                outlier_mask = z_scores > self.config.outlier_threshold
                
                # Replace outliers with median
                if np.any(outlier_mask):
                    cleaned[outlier_mask, dim] = np.median(data)
        
        return cleaned
    
    def _smooth_trajectory(
        self,
        trajectory: np.ndarray,
        timestamps: np.ndarray
    ) -> np.ndarray:
        """Smooth trajectory using specified method."""
        if len(trajectory) < 5:
            return trajectory
        
        smoothed = trajectory.copy()
        
        if self.config.smoothing_method == 'savgol':
            from scipy.signal import savgol_filter
            
            window_length = min(
                self.config.smoothing_params.get('window_length', 5),
                len(trajectory) // 2
            )
            if window_length % 2 == 0:
                window_length += 1
            
            polyorder = min(
                self.config.smoothing_params.get('polyorder', 2),
                window_length - 1
            )
            
            for dim in range(trajectory.shape[1]):
                smoothed[:, dim] = savgol_filter(
                    trajectory[:, dim], window_length, polyorder
                )
        
        elif self.config.smoothing_method == 'gaussian':
            from scipy.ndimage import gaussian_filter1d
            
            sigma = self.config.smoothing_params.get('sigma', 1.0)
            
            for dim in range(trajectory.shape[1]):
                smoothed[:, dim] = gaussian_filter1d(trajectory[:, dim], sigma)
        
        elif self.config.smoothing_method == 'moving_average':
            window_size = self.config.smoothing_params.get('window_size', 5)
            
            for dim in range(trajectory.shape[1]):
                smoothed[:, dim] = pd.Series(trajectory[:, dim]).rolling(
                    window=window_size, center=True
                ).mean().fillna(method='bfill').fillna(method='ffill').values
        
        return smoothed
    
    def preprocess_batch(
        self,
        sequences: List[SyntheticSequence]
    ) -> List[SyntheticSequence]:
        """
        Preprocess a batch of sequences.
        
        Args:
            sequences: List of input sequences
            
        Returns:
            List of preprocessed sequences
        """
        processed_sequences = []
        
        logger.info(f"Preprocessing batch of {len(sequences)} sequences...")
        
        for i, sequence in enumerate(sequences):
            try:
                processed_seq = self.preprocess_sequence(sequence)
                processed_sequences.append(processed_seq)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Preprocessed {i + 1}/{len(sequences)} sequences")
                    
            except Exception as e:
                logger.error(f"Preprocessing failed for sequence {sequence.sequence_id}: {e}")
                # Skip failed sequences or add original
                continue
        
        logger.info(f"Preprocessing complete: {len(processed_sequences)} sequences")
        return processed_sequences