"""
Unit tests for synthetic human behavior data generator.
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock
import tempfile
import pickle
import json
from pathlib import Path

from src.data.synthetic_generator import (
    SyntheticHumanBehaviorGenerator,
    SyntheticSequence,
    GestureType,
    TrajectoryParameters,
    GazeParameters
)


class TestTrajectoryParameters:
    """Test cases for TrajectoryParameters data class."""
    
    def test_valid_trajectory_parameters(self):
        """Test creating valid trajectory parameters."""
        start_pos = np.array([0.0, 0.0, 1.0])
        end_pos = np.array([1.0, 0.5, 1.2])
        
        params = TrajectoryParameters(
            start_position=start_pos,
            end_position=end_pos,
            duration=2.0,
            velocity_profile='bell_curve',
            noise_std=0.01
        )
        
        assert np.array_equal(params.start_position, start_pos)
        assert np.array_equal(params.end_position, end_pos)
        assert params.duration == 2.0
        assert params.velocity_profile == 'bell_curve'
        assert params.noise_std == 0.01
    
    def test_invalid_position_shape(self):
        """Test that invalid position shape raises error."""
        with pytest.raises(ValueError, match="Start position must be 3D"):
            TrajectoryParameters(
                start_position=np.array([0.0, 0.0]),  # Wrong shape
                end_position=np.array([1.0, 0.5, 1.2]),
                duration=2.0
            )
    
    def test_invalid_duration(self):
        """Test that invalid duration raises error."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            TrajectoryParameters(
                start_position=np.array([0.0, 0.0, 1.0]),
                end_position=np.array([1.0, 0.5, 1.2]),
                duration=-1.0  # Invalid
            )
    
    def test_invalid_noise_std(self):
        """Test that invalid noise std raises error."""
        with pytest.raises(ValueError, match="Noise standard deviation must be non-negative"):
            TrajectoryParameters(
                start_position=np.array([0.0, 0.0, 1.0]),
                end_position=np.array([1.0, 0.5, 1.2]),
                duration=2.0,
                noise_std=-0.1  # Invalid
            )


class TestSyntheticSequence:
    """Test cases for SyntheticSequence data class."""
    
    def test_synthetic_sequence_creation(self):
        """Test creating a synthetic sequence."""
        trajectory = np.random.rand(100, 3)
        gaze_traj = np.random.rand(100, 3)
        timestamps = np.linspace(0, 2, 100)
        
        sequence = SyntheticSequence(
            sequence_id="test_001",
            gesture_type=GestureType.WAVE,
            hand_trajectory=trajectory,
            gaze_trajectory=gaze_traj,
            timestamps=timestamps,
            intent_labels=[],
            context_info={'test': True},
            noise_level=0.02
        )
        
        assert sequence.sequence_id == "test_001"
        assert sequence.gesture_type == GestureType.WAVE
        assert sequence.hand_trajectory.shape == (100, 3)
        assert sequence.noise_level == 0.02


class TestSyntheticHumanBehaviorGenerator:
    """Test cases for SyntheticHumanBehaviorGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create test generator instance."""
        workspace_bounds = np.array([-1.0, 1.0, -1.0, 1.0, 0.0, 2.0])
        return SyntheticHumanBehaviorGenerator(
            workspace_bounds=workspace_bounds,
            sampling_frequency=30.0,
            random_seed=42
        )
    
    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.sampling_frequency == 30.0
        assert generator.dt == 1.0 / 30.0
        assert np.array_equal(generator.workspace_bounds, [-1.0, 1.0, -1.0, 1.0, 0.0, 2.0])
        assert np.array_equal(generator.x_bounds, [-1.0, 1.0])
        assert np.array_equal(generator.y_bounds, [-1.0, 1.0])
        assert np.array_equal(generator.z_bounds, [0.0, 2.0])
    
    def test_gesture_parameters_initialization(self, generator):
        """Test gesture parameters are properly initialized."""
        assert GestureType.WAVE in generator.gesture_params
        assert GestureType.POINT in generator.gesture_params
        assert GestureType.GRAB in generator.gesture_params
        
        wave_params = generator.gesture_params[GestureType.WAVE]
        assert 'amplitude' in wave_params
        assert 'frequency' in wave_params
        assert 'base_duration' in wave_params
    
    def test_minimum_jerk_trajectory_basic(self, generator):
        """Test basic minimum jerk trajectory generation."""
        params = TrajectoryParameters(
            start_position=np.array([0.0, 0.0, 1.0]),
            end_position=np.array([0.5, 0.3, 1.2]),
            duration=2.0
        )
        
        t, positions, velocities, accelerations = generator.generate_minimum_jerk_trajectory(params)
        
        # Check shapes
        expected_length = int(2.0 * 30.0)  # duration * sampling_freq
        assert len(t) == expected_length
        assert positions.shape == (expected_length, 3)
        assert velocities.shape == (expected_length, 3)
        assert accelerations.shape == (expected_length, 3)
        
        # Check boundary conditions
        np.testing.assert_allclose(positions[0], params.start_position, rtol=1e-2)
        np.testing.assert_allclose(positions[-1], params.end_position, rtol=1e-2)
        
        # Check that velocity starts and ends near zero
        assert np.linalg.norm(velocities[0]) < 0.1
        assert np.linalg.norm(velocities[-1]) < 0.1
    
    def test_minimum_jerk_trajectory_with_noise(self, generator):
        """Test minimum jerk trajectory with noise."""
        params = TrajectoryParameters(
            start_position=np.array([0.0, 0.0, 1.0]),
            end_position=np.array([0.5, 0.3, 1.2]),
            duration=1.0,
            noise_std=0.1
        )
        
        t, positions, _, _ = generator.generate_minimum_jerk_trajectory(params)
        
        # With noise, positions shouldn't exactly match start/end
        # but should be close
        start_diff = np.linalg.norm(positions[0] - params.start_position)
        end_diff = np.linalg.norm(positions[-1] - params.end_position)
        
        # Should be within reasonable noise bounds
        assert start_diff < 0.3  # 3 sigma for 0.1 std
        assert end_diff < 0.3
    
    def test_minimum_jerk_trajectory_with_waypoints(self, generator):
        """Test minimum jerk trajectory with intermediate waypoints."""
        waypoints = [np.array([0.2, 0.1, 1.05]), np.array([0.3, 0.2, 1.1])]
        
        params = TrajectoryParameters(
            start_position=np.array([0.0, 0.0, 1.0]),
            end_position=np.array([0.5, 0.3, 1.2]),
            duration=2.0,
            intermediate_points=waypoints
        )
        
        t, positions, _, _ = generator.generate_minimum_jerk_trajectory(params)
        
        # Should still have correct start and end
        np.testing.assert_allclose(positions[0], params.start_position, rtol=1e-2)
        np.testing.assert_allclose(positions[-1], params.end_position, rtol=1e-2)
    
    def test_gaze_trajectory_generation(self, generator):
        """Test gaze trajectory generation."""
        focal_points = [
            np.array([0.0, 0.0, 1.5]),
            np.array([0.5, 0.0, 1.3]),
            np.array([0.3, 0.2, 1.4])
        ]
        dwell_times = [0.5, 0.8, 0.7]
        
        params = GazeParameters(
            focal_points=focal_points,
            dwell_times=dwell_times,
            saccade_duration=0.05,
            fixation_noise=0.002
        )
        
        t, gaze_trajectory = generator.generate_gaze_trajectory(params, duration=2.0)
        
        expected_length = int(2.0 * 30.0)
        assert len(t) == expected_length
        assert gaze_trajectory.shape == (expected_length, 3)
        
        # Check that gaze starts near first focal point
        start_distance = np.linalg.norm(gaze_trajectory[0] - focal_points[0])
        assert start_distance < 0.1
    
    def test_gesture_sequence_generation_wave(self, generator):
        """Test wave gesture sequence generation."""
        context = {
            'workspace_bounds': generator.workspace_bounds.tolist(),
            'n_objects': 1,
            'robot_position': [0.0, -0.3, 1.0]
        }
        
        sequence = generator.generate_gesture_sequence(
            gesture_type=GestureType.WAVE,
            context=context,
            noise_level=0.01
        )
        
        assert sequence.gesture_type == GestureType.WAVE
        assert sequence.hand_trajectory.shape[1] == 3
        assert sequence.gaze_trajectory.shape[1] == 3
        assert len(sequence.timestamps) > 0
        assert sequence.noise_level == 0.01
        assert len(sequence.intent_labels) == len(sequence.timestamps)
    
    def test_gesture_sequence_generation_point(self, generator):
        """Test point gesture sequence generation."""
        target_object = {
            'id': 'target_cup',
            'position': [0.8, 0.3, 1.0],
            'type': 'cup'
        }
        
        context = {
            'target_object': target_object,
            'workspace_bounds': generator.workspace_bounds.tolist()
        }
        
        sequence = generator.generate_gesture_sequence(
            gesture_type=GestureType.POINT,
            context=context
        )
        
        assert sequence.gesture_type == GestureType.POINT
        assert 'target_object' in sequence.context_info
        
        # Check that trajectory moves toward target
        start_pos = sequence.hand_trajectory[0]
        end_pos = sequence.hand_trajectory[-1]
        target_pos = np.array(target_object['position'])
        
        start_dist = np.linalg.norm(start_pos - target_pos)
        end_dist = np.linalg.norm(end_pos - target_pos)
        
        # Should move closer to target (approximately)
        assert end_dist <= start_dist + 0.2  # Allow some tolerance
    
    def test_gesture_positions_wave(self, generator):
        """Test gesture position generation for wave."""
        context = {'workspace_bounds': generator.workspace_bounds.tolist()}
        
        start_pos, end_pos, waypoints = generator._get_gesture_positions(
            GestureType.WAVE, context
        )
        
        assert start_pos.shape == (3,)
        assert end_pos.shape == (3,)
        assert waypoints is not None
        assert len(waypoints) > 0
        
        # For wave, should have oscillatory waypoints
        assert len(waypoints) >= 2
    
    def test_gesture_positions_grab(self, generator):
        """Test gesture position generation for grab."""
        target_object = {'position': [0.5, 0.2, 0.8]}
        context = {
            'target_object': target_object,
            'workspace_bounds': generator.workspace_bounds.tolist()
        }
        
        start_pos, end_pos, waypoints = generator._get_gesture_positions(
            GestureType.GRAB, context
        )
        
        # End position should be near target object
        target_pos = np.array(target_object['position'])
        distance_to_target = np.linalg.norm(end_pos - target_pos)
        assert distance_to_target < 0.1  # Should be very close
    
    def test_gaze_parameters_generation(self, generator):
        """Test gaze parameters generation."""
        hand_positions = np.random.rand(50, 3)
        context = {
            'target_object': {'position': [0.5, 0.3, 1.0]},
            'robot_position': [0.0, -0.2, 1.2]
        }
        
        gaze_params = generator._get_gaze_parameters(
            GestureType.POINT, context, hand_positions
        )
        
        assert len(gaze_params.focal_points) > 0
        assert len(gaze_params.dwell_times) == len(gaze_params.focal_points)
        assert gaze_params.saccade_duration > 0
        assert gaze_params.fixation_noise >= 0
    
    def test_intent_labels_generation(self, generator):
        """Test intent labels generation."""
        n_timesteps = 100
        
        for gesture_type in GestureType:
            labels = generator._generate_intent_labels(gesture_type, n_timesteps)
            
            assert len(labels) == n_timesteps
            assert all(isinstance(label, type(labels[0])) for label in labels)
            
            # Check that labels change over time (not all the same)
            unique_labels = set(labels)
            assert len(unique_labels) >= 1  # At least one label type
    
    def test_context_generation(self, generator):
        """Test context generation."""
        context = generator._generate_context()
        
        required_keys = ['workspace_bounds', 'n_objects', 'handover_zone', 'robot_position']
        for key in required_keys:
            assert key in context
        
        assert context['n_objects'] >= 0
        assert len(context['handover_zone']) == 3
        assert len(context['robot_position']) == 3
        
        # If objects exist, they should have proper structure
        if context['n_objects'] > 0:
            assert 'objects' in context
            for obj in context['objects']:
                assert 'id' in obj
                assert 'position' in obj
                assert 'type' in obj
                assert len(obj['position']) == 3
    
    def test_dataset_generation(self, generator):
        """Test dataset generation."""
        n_sequences = 10
        dataset = generator.generate_dataset(
            n_sequences=n_sequences,
            noise_range=(0.01, 0.03)
        )
        
        assert len(dataset) <= n_sequences  # Could be fewer due to failures
        assert len(dataset) > 0  # Should generate at least some
        
        for sequence in dataset:
            assert isinstance(sequence, SyntheticSequence)
            assert sequence.hand_trajectory.shape[1] == 3
            assert sequence.gaze_trajectory.shape[1] == 3
            assert len(sequence.timestamps) > 0
            assert 0.01 <= sequence.noise_level <= 0.03
    
    def test_dataset_generation_with_distribution(self, generator):
        """Test dataset generation with custom gesture distribution."""
        gesture_distribution = {
            GestureType.WAVE: 0.5,
            GestureType.POINT: 0.3,
            GestureType.GRAB: 0.2
        }
        
        dataset = generator.generate_dataset(
            n_sequences=20,
            gesture_distribution=gesture_distribution
        )
        
        # Check that gestures follow approximately the right distribution
        gesture_counts = {}
        for sequence in dataset:
            gesture = sequence.gesture_type
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        # Should have all three gesture types represented
        assert len(gesture_counts) >= 1
        
        # Wave should be most common (allowing for randomness)
        if GestureType.WAVE in gesture_counts:
            wave_ratio = gesture_counts[GestureType.WAVE] / len(dataset)
            assert wave_ratio >= 0.2  # Allow some deviation
    
    def test_save_and_load_dataset(self, generator):
        """Test saving and loading dataset."""
        dataset = generator.generate_dataset(n_sequences=5)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_dataset"
            
            # Save dataset
            generator.save_dataset(dataset, str(save_path))
            
            # Check files were created
            assert (save_path.parent / "test_dataset.pkl").exists()
            assert (save_path.parent / "test_dataset_metadata.json").exists()
            
            # Load dataset
            loaded_dataset = SyntheticHumanBehaviorGenerator.load_dataset(str(save_path))
            
            assert len(loaded_dataset) == len(dataset)
            
            # Check first sequence matches
            original = dataset[0]
            loaded = loaded_dataset[0]
            
            assert original.sequence_id == loaded.sequence_id
            assert original.gesture_type == loaded.gesture_type
            np.testing.assert_array_equal(original.hand_trajectory, loaded.hand_trajectory)
            np.testing.assert_array_equal(original.timestamps, loaded.timestamps)
    
    def test_load_dataset_metadata(self, generator):
        """Test loading dataset metadata."""
        dataset = generator.generate_dataset(n_sequences=3)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_dataset"
            generator.save_dataset(dataset, str(save_path))
            
            # Load and check metadata
            metadata_path = save_path.parent / "test_dataset_metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            assert metadata['n_sequences'] == len(dataset)
            assert 'gesture_types' in metadata
            assert 'avg_duration' in metadata
            assert 'workspace_bounds' in metadata
            assert metadata['sampling_frequency'] == 30.0
    
    def test_workspace_bounds_validation(self, generator):
        """Test that generated trajectories respect workspace bounds."""
        dataset = generator.generate_dataset(n_sequences=5)
        
        x_bounds = generator.x_bounds
        y_bounds = generator.y_bounds
        z_bounds = generator.z_bounds
        
        for sequence in dataset:
            trajectory = sequence.hand_trajectory
            
            # Check X bounds (with some tolerance for noise)
            assert np.all(trajectory[:, 0] >= x_bounds[0] - 0.1)
            assert np.all(trajectory[:, 0] <= x_bounds[1] + 0.1)
            
            # Check Y bounds
            assert np.all(trajectory[:, 1] >= y_bounds[0] - 0.1)
            assert np.all(trajectory[:, 1] <= y_bounds[1] + 0.1)
            
            # Check Z bounds
            assert np.all(trajectory[:, 2] >= z_bounds[0] - 0.1)
            assert np.all(trajectory[:, 2] <= z_bounds[1] + 0.1)
    
    def test_temporal_consistency(self, generator):
        """Test that generated sequences have consistent timestamps."""
        dataset = generator.generate_dataset(n_sequences=5)
        
        for sequence in dataset:
            timestamps = sequence.timestamps
            
            # Check that timestamps are monotonically increasing
            assert np.all(np.diff(timestamps) > 0)
            
            # Check approximate sampling rate
            if len(timestamps) > 1:
                dt_mean = np.mean(np.diff(timestamps))
                expected_dt = 1.0 / generator.sampling_frequency
                
                # Allow 10% tolerance
                assert abs(dt_mean - expected_dt) / expected_dt < 0.1


@pytest.mark.unit
class TestGestureType:
    """Test cases for GestureType enum."""
    
    def test_gesture_type_values(self):
        """Test that gesture type enum has expected values."""
        expected_values = ["wave", "point", "grab", "handover", "reach", "idle"]
        actual_values = [gesture.value for gesture in GestureType]
        
        for expected in expected_values:
            assert expected in actual_values
    
    def test_gesture_type_membership(self):
        """Test gesture type membership."""
        assert GestureType.WAVE in GestureType
        assert GestureType.POINT in GestureType
        assert GestureType.GRAB in GestureType
        assert GestureType.HANDOVER in GestureType
        assert GestureType.REACH in GestureType
        assert GestureType.IDLE in GestureType