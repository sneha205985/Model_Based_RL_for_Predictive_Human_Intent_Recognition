"""
Unit tests for feature extraction pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock

from src.data.feature_extraction import (
    FeatureExtractor,
    ExtractedFeatures,
    FeatureConfig,
    FeatureType
)
from src.data.synthetic_generator import SyntheticSequence, GestureType


class TestFeatureConfig:
    """Test cases for FeatureConfig data class."""
    
    def test_default_feature_config(self):
        """Test default feature configuration."""
        config = FeatureConfig()
        
        assert config.window_size == 30
        assert config.overlap == 0.5
        assert config.smooth_trajectories is True
        assert config.smoothing_window == 5
        assert config.extract_derivatives is True
        assert config.spatial_resolution == 0.1
        assert len(config.frequency_bands) == 3  # Default bands
    
    def test_custom_feature_config(self):
        """Test custom feature configuration."""
        custom_bands = [(0.5, 3.0), (3.0, 10.0)]
        
        config = FeatureConfig(
            window_size=50,
            overlap=0.3,
            smooth_trajectories=False,
            frequency_bands=custom_bands
        )
        
        assert config.window_size == 50
        assert config.overlap == 0.3
        assert config.smooth_trajectories is False
        assert config.frequency_bands == custom_bands


class TestExtractedFeatures:
    """Test cases for ExtractedFeatures data class."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample extracted features."""
        return ExtractedFeatures(
            sequence_id="test_001",
            kinematic_features=np.array([1.0, 2.0, 3.0]),
            temporal_features=np.array([4.0, 5.0]),
            spatial_features=np.array([6.0, 7.0, 8.0, 9.0]),
            statistical_features=np.array([10.0, 11.0]),
            frequency_features=np.array([12.0]),
            feature_names=['k1', 'k2', 'k3', 't1', 't2', 's1', 's2', 's3', 's4', 'st1', 'st2', 'f1'],
            timestamps=np.array([0.0, 0.1, 0.2])
        )
    
    def test_get_all_features(self, sample_features):
        """Test concatenating all feature types."""
        all_features = sample_features.get_all_features()
        
        expected_length = 3 + 2 + 4 + 2 + 1  # kinematic + temporal + spatial + statistical + frequency
        assert all_features.shape[-1] == expected_length
        
        # Should contain all features in order
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        if all_features.ndim == 1:
            np.testing.assert_array_equal(all_features, expected)
        else:
            np.testing.assert_array_equal(all_features[0], expected)
    
    def test_get_feature_dict(self, sample_features):
        """Test getting features as dictionary."""
        feature_dict = sample_features.get_feature_dict()
        
        assert len(feature_dict) == len(sample_features.feature_names)
        assert 'k1' in feature_dict
        assert 't1' in feature_dict
        assert 's1' in feature_dict
        
        # Check values match
        if isinstance(feature_dict['k1'], np.ndarray):
            if feature_dict['k1'].ndim > 0:
                assert feature_dict['k1'][0] == 1.0 or feature_dict['k1'] == 1.0
            else:
                assert feature_dict['k1'] == 1.0
    
    def test_empty_features(self):
        """Test behavior with empty features."""
        empty_features = ExtractedFeatures(
            sequence_id="empty",
            kinematic_features=np.array([]),
            temporal_features=np.array([]),
            spatial_features=np.array([]),
            statistical_features=np.array([]),
            frequency_features=np.array([]),
            feature_names=[],
            timestamps=np.array([])
        )
        
        all_features = empty_features.get_all_features()
        feature_dict = empty_features.get_feature_dict()
        
        assert all_features.shape == (1, 0) or all_features.size == 0
        assert len(feature_dict) == 0


class TestFeatureExtractor:
    """Test cases for FeatureExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create feature extractor instance."""
        return FeatureExtractor()
    
    @pytest.fixture
    def sample_sequence(self):
        """Create sample synthetic sequence."""
        np.random.seed(42)  # For reproducible tests
        
        # Create smooth trajectory (sine wave)
        t = np.linspace(0, 2, 60)  # 2 seconds at 30 Hz
        trajectory = np.column_stack([
            0.5 * np.sin(2 * np.pi * t),      # X: oscillatory
            0.3 * t,                          # Y: linear trend  
            1.0 + 0.1 * np.sin(4 * np.pi * t) # Z: higher frequency
        ])
        
        gaze_trajectory = trajectory + np.random.normal(0, 0.01, trajectory.shape)
        
        sequence = SyntheticSequence(
            sequence_id="test_wave",
            gesture_type=GestureType.WAVE,
            hand_trajectory=trajectory,
            gaze_trajectory=gaze_trajectory,
            timestamps=t,
            intent_labels=[],
            context_info={},
            noise_level=0.01
        )
        
        return sequence
    
    def test_extractor_initialization(self, extractor):
        """Test feature extractor initialization."""
        assert extractor.config is not None
        assert extractor.scaler is not None
        assert extractor.is_fitted is False
    
    def test_smooth_trajectory(self, extractor):
        """Test trajectory smoothing."""
        # Create noisy trajectory
        t = np.linspace(0, 1, 30)
        clean_trajectory = np.column_stack([t, t**2, np.sin(t)])
        noisy_trajectory = clean_trajectory + np.random.normal(0, 0.1, clean_trajectory.shape)
        
        smoothed = extractor._smooth_trajectory(noisy_trajectory)
        
        assert smoothed.shape == noisy_trajectory.shape
        
        # Smoothed trajectory should have less noise
        smooth_variance = np.var(np.diff(smoothed, axis=0))
        noisy_variance = np.var(np.diff(noisy_trajectory, axis=0))
        assert smooth_variance <= noisy_variance
    
    def test_compute_derivatives(self, extractor):
        """Test derivative computation."""
        # Simple polynomial trajectory
        t = np.linspace(0, 2, 60)
        trajectory = np.column_stack([t, t**2, t**3])
        
        velocities, accelerations = extractor._compute_derivatives(trajectory, t)
        
        assert velocities.shape == trajectory.shape
        assert accelerations.shape == trajectory.shape
        
        # Check that derivatives have expected properties
        # For x = t, velocity should be approximately 1
        assert np.abs(np.mean(velocities[:, 0]) - 1.0) < 0.1
        
        # For y = t^2, velocity should be approximately 2t
        expected_vel_y = 2 * t
        assert np.abs(np.mean(velocities[:, 1] - expected_vel_y)) < 0.5
    
    def test_extract_kinematic_features(self, extractor, sample_sequence):
        """Test kinematic feature extraction."""
        trajectory = sample_sequence.hand_trajectory
        timestamps = sample_sequence.timestamps
        
        velocities, accelerations = extractor._compute_derivatives(trajectory, timestamps)
        
        features, names = extractor._extract_kinematic_features(
            trajectory, velocities, accelerations
        )
        
        assert len(features) == len(names)
        assert len(features) > 0
        
        # Check for expected feature names
        expected_features = ['speed_mean', 'speed_std', 'speed_max', 
                           'acc_mean', 'acc_std', 'acc_max']
        for expected in expected_features:
            assert expected in names
        
        # Check that features are reasonable
        speed_mean_idx = names.index('speed_mean')
        assert features[speed_mean_idx] >= 0  # Speed should be non-negative
        
        efficiency_idx = names.index('movement_efficiency')
        assert 0 <= features[efficiency_idx] <= 1  # Efficiency should be [0, 1]
    
    def test_extract_temporal_features(self, extractor, sample_sequence):
        """Test temporal feature extraction."""
        trajectory = sample_sequence.hand_trajectory
        timestamps = sample_sequence.timestamps
        
        features, names = extractor._extract_temporal_features(trajectory, timestamps)
        
        assert len(features) == len(names)
        assert len(features) > 0
        
        # Check for expected features
        expected_features = ['duration', 'sampling_rate', 'n_movement_phases']
        for expected in expected_features:
            assert expected in names
        
        # Validate feature values
        duration_idx = names.index('duration')
        expected_duration = timestamps[-1] - timestamps[0]
        assert abs(features[duration_idx] - expected_duration) < 0.1
        
        sampling_rate_idx = names.index('sampling_rate')
        assert features[sampling_rate_idx] > 0
    
    def test_extract_spatial_features(self, extractor, sample_sequence):
        """Test spatial feature extraction."""
        trajectory = sample_sequence.hand_trajectory
        timestamps = sample_sequence.timestamps
        
        velocities, accelerations = extractor._compute_derivatives(trajectory, timestamps)
        
        features, names = extractor._extract_spatial_features(
            trajectory, velocities, accelerations
        )
        
        assert len(features) == len(names)
        assert len(features) > 0
        
        # Check for expected features
        expected_features = ['path_length', 'displacement', 'path_efficiency',
                           'workspace_x', 'workspace_y', 'workspace_z']
        for expected in expected_features:
            assert expected in names
        
        # Validate features
        path_length_idx = names.index('path_length')
        displacement_idx = names.index('displacement')
        efficiency_idx = names.index('path_efficiency')
        
        assert features[path_length_idx] >= features[displacement_idx]  # Path >= displacement
        assert 0 <= features[efficiency_idx] <= 1  # Efficiency in [0, 1]
    
    def test_extract_statistical_features(self, extractor, sample_sequence):
        """Test statistical feature extraction."""
        trajectory = sample_sequence.hand_trajectory
        timestamps = sample_sequence.timestamps
        
        velocities, accelerations = extractor._compute_derivatives(trajectory, timestamps)
        
        features, names = extractor._extract_statistical_features(
            trajectory, velocities, accelerations
        )
        
        assert len(features) == len(names)
        assert len(features) > 0
        
        # Check for position statistics
        pos_features = ['pos_x_mean', 'pos_y_mean', 'pos_z_mean',
                       'pos_x_std', 'pos_y_std', 'pos_z_std']
        for expected in pos_features:
            assert expected in names
        
        # Check velocity statistics
        vel_features = ['vel_mean', 'vel_std']
        for expected in vel_features:
            assert expected in names
    
    def test_extract_frequency_features(self, extractor, sample_sequence):
        """Test frequency domain feature extraction."""
        trajectory = sample_sequence.hand_trajectory
        timestamps = sample_sequence.timestamps
        
        features, names = extractor._extract_frequency_features(trajectory, timestamps)
        
        assert len(features) == len(names)
        assert len(features) > 0
        
        # Check for expected features
        expected_features = ['dominant_freq', 'spectral_centroid', 'spectral_spread']
        for expected in expected_features:
            assert expected in names
        
        # Validate frequency features
        dom_freq_idx = names.index('dominant_freq')
        assert features[dom_freq_idx] >= 0  # Frequency should be non-negative
        
        # Check power band features exist
        power_band_features = [name for name in names if name.startswith('power_band')]
        assert len(power_band_features) > 0
    
    def test_extract_features_all_types(self, extractor, sample_sequence):
        """Test extracting all feature types."""
        features = extractor.extract_features(sample_sequence)
        
        assert features.sequence_id == sample_sequence.sequence_id
        assert len(features.feature_names) > 0
        
        # Should have all feature types
        assert features.kinematic_features.size > 0
        assert features.temporal_features.size > 0
        assert features.spatial_features.size > 0
        assert features.statistical_features.size > 0
        assert features.frequency_features.size > 0
    
    def test_extract_features_subset(self, extractor, sample_sequence):
        """Test extracting subset of feature types."""
        feature_types = [FeatureType.KINEMATIC, FeatureType.TEMPORAL]
        
        features = extractor.extract_features(sample_sequence, feature_types)
        
        # Should only have specified feature types
        assert features.kinematic_features.size > 0
        assert features.temporal_features.size > 0
        assert features.spatial_features.size == 0
        assert features.statistical_features.size == 0
        assert features.frequency_features.size == 0
    
    def test_extract_batch_features(self, extractor):
        """Test batch feature extraction."""
        # Create multiple sequences
        sequences = []
        for i in range(3):
            t = np.linspace(0, 1, 30)
            trajectory = np.column_stack([
                np.sin(2 * np.pi * t + i),
                np.cos(2 * np.pi * t + i),
                t + i * 0.1
            ])
            
            sequence = SyntheticSequence(
                sequence_id=f"batch_{i}",
                gesture_type=GestureType.WAVE,
                hand_trajectory=trajectory,
                gaze_trajectory=trajectory.copy(),
                timestamps=t,
                intent_labels=[],
                context_info={},
                noise_level=0.01
            )
            sequences.append(sequence)
        
        feature_matrix, feature_names, sequence_ids = extractor.extract_batch_features(
            sequences, normalize=False
        )
        
        assert feature_matrix.shape[0] == len(sequences)
        assert feature_matrix.shape[1] == len(feature_names)
        assert len(sequence_ids) == len(sequences)
        assert all(sid.startswith('batch_') for sid in sequence_ids)
    
    def test_extract_batch_features_with_normalization(self, extractor):
        """Test batch feature extraction with normalization."""
        # Create sequences with different scales
        sequences = []
        for i in range(3):
            t = np.linspace(0, 1, 30)
            scale = (i + 1) * 2  # Different scales
            trajectory = np.column_stack([
                scale * np.sin(2 * np.pi * t),
                scale * np.cos(2 * np.pi * t),
                scale * t
            ])
            
            sequence = SyntheticSequence(
                sequence_id=f"norm_{i}",
                gesture_type=GestureType.REACH,
                hand_trajectory=trajectory,
                gaze_trajectory=trajectory.copy(),
                timestamps=t,
                intent_labels=[],
                context_info={},
                noise_level=0.01
            )
            sequences.append(sequence)
        
        feature_matrix, _, _ = extractor.extract_batch_features(
            sequences, normalize=True
        )
        
        # Check that normalization was applied
        assert extractor.is_fitted is True
        
        # Features should be approximately zero mean, unit variance
        means = np.mean(feature_matrix, axis=0)
        stds = np.std(feature_matrix, axis=0)
        
        # Most features should be normalized (allowing for some that might be constant)
        normalized_features = np.sum(np.abs(means) < 0.1)  # Near zero mean
        assert normalized_features > len(means) * 0.5  # At least half
    
    def test_save_and_load_features(self, extractor):
        """Test saving and loading features."""
        # Create test data
        feature_matrix = np.random.rand(5, 10)
        feature_names = [f'feature_{i}' for i in range(10)]
        sequence_ids = [f'seq_{i}' for i in range(5)]
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = f"{temp_dir}/test_features.csv"
            
            # Save features
            extractor.save_features(feature_matrix, feature_names, sequence_ids, filepath)
            
            # Load features
            loaded_matrix, loaded_names, loaded_ids = FeatureExtractor.load_features(filepath)
            
            np.testing.assert_array_equal(feature_matrix, loaded_matrix)
            assert feature_names == loaded_names
            assert sequence_ids == loaded_ids
    
    def test_feature_extraction_edge_cases(self, extractor):
        """Test feature extraction with edge cases."""
        # Very short sequence
        t = np.array([0.0, 0.1])
        trajectory = np.array([[0.0, 0.0, 1.0], [0.1, 0.1, 1.1]])
        
        short_sequence = SyntheticSequence(
            sequence_id="short",
            gesture_type=GestureType.IDLE,
            hand_trajectory=trajectory,
            gaze_trajectory=trajectory.copy(),
            timestamps=t,
            intent_labels=[],
            context_info={},
            noise_level=0.0
        )
        
        # Should not crash
        features = extractor.extract_features(short_sequence)
        assert features is not None
        assert len(features.feature_names) > 0
    
    def test_feature_extraction_constant_trajectory(self, extractor):
        """Test feature extraction with constant (no movement) trajectory."""
        t = np.linspace(0, 1, 30)
        trajectory = np.tile([0.5, 0.3, 1.2], (len(t), 1))  # Constant position
        
        constant_sequence = SyntheticSequence(
            sequence_id="constant",
            gesture_type=GestureType.IDLE,
            hand_trajectory=trajectory,
            gaze_trajectory=trajectory.copy(),
            timestamps=t,
            intent_labels=[],
            context_info={},
            noise_level=0.0
        )
        
        features = extractor.extract_features(constant_sequence)
        
        # Should handle zero movement gracefully
        all_features = features.get_all_features()
        assert np.all(np.isfinite(all_features))  # No NaN or inf values
        
        # Speed features should be zero or near zero
        feature_dict = features.get_feature_dict()
        if 'speed_mean' in feature_dict:
            speed_val = feature_dict['speed_mean']
            if hasattr(speed_val, '__len__'):
                assert np.all(np.array(speed_val) < 0.1)
            else:
                assert speed_val < 0.1


@pytest.mark.unit
class TestFeatureType:
    """Test cases for FeatureType enum."""
    
    def test_feature_type_values(self):
        """Test that feature type enum has expected values."""
        expected_types = ["kinematic", "temporal", "spatial", "statistical", "frequency"]
        actual_types = [ftype.value for ftype in FeatureType]
        
        for expected in expected_types:
            assert expected in actual_types
    
    def test_feature_type_membership(self):
        """Test feature type membership."""
        assert FeatureType.KINEMATIC in FeatureType
        assert FeatureType.TEMPORAL in FeatureType
        assert FeatureType.SPATIAL in FeatureType
        assert FeatureType.STATISTICAL in FeatureType
        assert FeatureType.FREQUENCY in FeatureType


class TestFeatureExtractionIntegration:
    """Integration tests for feature extraction pipeline."""
    
    def test_full_pipeline_wave_gesture(self):
        """Test complete feature extraction pipeline for wave gesture."""
        # Create realistic wave gesture
        t = np.linspace(0, 2, 60)  # 2 seconds
        x_wave = 0.3 * np.sin(3 * 2 * np.pi * t)  # 3 cycles
        y_trend = 0.1 * t  # Slight forward motion
        z_stable = 1.2 + 0.05 * np.sin(6 * 2 * np.pi * t)  # Small z variation
        
        trajectory = np.column_stack([x_wave, y_trend, z_stable])
        
        sequence = SyntheticSequence(
            sequence_id="integration_wave",
            gesture_type=GestureType.WAVE,
            hand_trajectory=trajectory,
            gaze_trajectory=trajectory + np.random.normal(0, 0.01, trajectory.shape),
            timestamps=t,
            intent_labels=[],
            context_info={'test': 'integration'},
            noise_level=0.02
        )
        
        extractor = FeatureExtractor()
        features = extractor.extract_features(sequence)
        
        # Validate extracted features make sense for wave gesture
        feature_dict = features.get_feature_dict()
        
        # Should have detected oscillatory motion
        if 'periodicity_strength' in feature_dict:
            periodicity = feature_dict['periodicity_strength']
            if hasattr(periodicity, '__len__'):
                assert np.any(np.array(periodicity) > 0.3)  # Some periodicity detected
            else:
                assert periodicity > 0.3
        
        # Should have reasonable path efficiency (not too direct due to oscillation)
        if 'path_efficiency' in feature_dict:
            efficiency = feature_dict['path_efficiency']
            if hasattr(efficiency, '__len__'):
                efficiency_val = efficiency[0] if len(efficiency) > 0 else efficiency
            else:
                efficiency_val = efficiency
            assert 0.1 <= efficiency_val <= 0.8  # Wave motion is not very efficient
    
    def test_feature_consistency_across_similar_sequences(self):
        """Test that similar sequences produce similar features."""
        extractor = FeatureExtractor()
        
        # Create two similar reaching gestures
        sequences = []
        for i in range(2):
            t = np.linspace(0, 1.5, 45)
            # Similar reaching motion with slight variations
            trajectory = np.column_stack([
                0.3 * t + 0.1 * i,  # Linear motion in X
                0.2 * t + 0.05 * i,  # Linear motion in Y
                1.0 + 0.1 * t + 0.02 * i  # Slight Z motion
            ])
            
            sequence = SyntheticSequence(
                sequence_id=f"similar_{i}",
                gesture_type=GestureType.REACH,
                hand_trajectory=trajectory,
                gaze_trajectory=trajectory.copy(),
                timestamps=t,
                intent_labels=[],
                context_info={},
                noise_level=0.01
            )
            sequences.append(sequence)
        
        # Extract features
        feature_matrix, feature_names, _ = extractor.extract_batch_features(
            sequences, normalize=False
        )
        
        # Features should be similar but not identical
        feature_diff = np.abs(feature_matrix[0] - feature_matrix[1])
        relative_diff = feature_diff / (np.abs(feature_matrix[0]) + 1e-8)
        
        # Most features should be similar (within 20% relative difference)
        similar_features = np.sum(relative_diff < 0.2)
        assert similar_features > len(feature_names) * 0.7  # At least 70% similar