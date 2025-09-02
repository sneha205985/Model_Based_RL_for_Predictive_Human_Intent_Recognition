"""
Unit tests for human behavior modeling components.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.models.human_behavior import (
    HumanBehaviorModel, 
    HumanState, 
    BehaviorPrediction, 
    BehaviorType
)


class TestHumanState:
    """Test cases for HumanState data class."""
    
    def test_valid_human_state_creation(self, sample_human_state):
        """Test creating a valid human state."""
        assert sample_human_state.position.shape == (3,)
        assert sample_human_state.orientation.shape == (4,)
        assert sample_human_state.velocity.shape == (3,)
        assert 0.0 <= sample_human_state.confidence <= 1.0
    
    def test_invalid_position_shape(self):
        """Test that invalid position shape raises error."""
        with pytest.raises(ValueError, match="Position must be a 3D vector"):
            HumanState(
                position=np.array([1.0, 2.0]),  # Wrong shape
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                joint_positions={},
                velocity=np.array([0.0, 0.0, 0.0]),
                timestamp=0.0
            )
    
    def test_invalid_orientation_shape(self):
        """Test that invalid orientation shape raises error."""
        with pytest.raises(ValueError, match="Orientation must be a quaternion"):
            HumanState(
                position=np.array([1.0, 2.0, 3.0]),
                orientation=np.array([1.0, 0.0, 0.0]),  # Wrong shape
                joint_positions={},
                velocity=np.array([0.0, 0.0, 0.0]),
                timestamp=0.0
            )
    
    def test_invalid_confidence_range(self):
        """Test that invalid confidence range raises error."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            HumanState(
                position=np.array([1.0, 2.0, 3.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                joint_positions={},
                velocity=np.array([0.0, 0.0, 0.0]),
                timestamp=0.0,
                confidence=1.5  # Invalid
            )


class TestBehaviorPrediction:
    """Test cases for BehaviorPrediction data class."""
    
    def test_valid_behavior_prediction(self, sample_behavior_prediction):
        """Test creating a valid behavior prediction."""
        assert isinstance(sample_behavior_prediction.behavior_type, BehaviorType)
        assert 0.0 <= sample_behavior_prediction.probability <= 1.0
        assert 0.0 <= sample_behavior_prediction.confidence <= 1.0
        assert sample_behavior_prediction.time_horizon > 0
        assert sample_behavior_prediction.metadata == {}
    
    def test_invalid_probability(self):
        """Test that invalid probability raises error."""
        with pytest.raises(ValueError, match="Probability must be between 0 and 1"):
            BehaviorPrediction(
                behavior_type=BehaviorType.GESTURE,
                probability=1.5,  # Invalid
                predicted_trajectory=np.array([[0, 0, 0]]),
                time_horizon=1.0
            )
    
    def test_invalid_time_horizon(self):
        """Test that invalid time horizon raises error."""
        with pytest.raises(ValueError, match="Time horizon must be positive"):
            BehaviorPrediction(
                behavior_type=BehaviorType.GESTURE,
                probability=0.8,
                predicted_trajectory=np.array([[0, 0, 0]]),
                time_horizon=-1.0  # Invalid
            )


class MockHumanBehaviorModel(HumanBehaviorModel):
    """Mock implementation for testing."""
    
    def _initialize_model(self):
        self.model_initialized = True
    
    def observe(self, human_state):
        self.last_observation = human_state
    
    def predict_behavior(self, current_state, time_horizon, num_samples=1):
        return [BehaviorPrediction(
            behavior_type=BehaviorType.IDLE,
            probability=0.9,
            predicted_trajectory=np.array([[0, 0, 0], [0, 0, 0]]),
            time_horizon=time_horizon
        )]
    
    def update_model(self, observations, ground_truth=None):
        return {"loss": 0.1}
    
    def get_intent_probability(self, current_state, intent):
        return 0.5
    
    def _train_epoch(self, training_data, batch_size):
        return {"epoch_loss": 0.1}
    
    def _validate_epoch(self, validation_data):
        return {"val_loss": 0.2}


class TestHumanBehaviorModel:
    """Test cases for HumanBehaviorModel abstract class."""
    
    @pytest.fixture
    def model(self, mock_config):
        """Create mock model for testing."""
        return MockHumanBehaviorModel(mock_config)
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.config is not None
        assert model.is_trained is False
        assert model.training_history == []
        assert hasattr(model, 'model_initialized')
    
    def test_observe_method(self, model, sample_human_state):
        """Test observe method."""
        model.observe(sample_human_state)
        assert model.last_observation == sample_human_state
    
    def test_predict_behavior(self, model, sample_human_state):
        """Test behavior prediction."""
        predictions = model.predict_behavior(sample_human_state, 5.0)
        assert len(predictions) == 1
        assert isinstance(predictions[0], BehaviorPrediction)
        assert predictions[0].time_horizon == 5.0
    
    def test_update_model(self, model, sample_human_state):
        """Test model update."""
        observations = [sample_human_state]
        metrics = model.update_model(observations)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
    
    def test_get_intent_probability(self, model, sample_human_state):
        """Test intent probability calculation."""
        prob = model.get_intent_probability(sample_human_state, "test_intent")
        assert 0.0 <= prob <= 1.0
    
    def test_train_with_empty_data(self, model):
        """Test that training with empty data raises error."""
        with pytest.raises(ValueError, match="Training data cannot be empty"):
            model.train([])
    
    def test_train_with_valid_data(self, model, sample_human_state, sample_behavior_prediction):
        """Test training with valid data."""
        training_data = [(sample_human_state, sample_behavior_prediction)]
        history = model.train(training_data, epochs=2, batch_size=1)
        
        assert model.is_trained is True
        assert "epoch_metrics" in history
        assert len(model.training_history) == 2
    
    def test_save_untrained_model(self, model):
        """Test that saving untrained model raises error."""
        with pytest.raises(RuntimeError, match="Cannot save untrained model"):
            model.save_model("test_path.pkl")
    
    def test_get_model_info(self, model):
        """Test getting model information."""
        info = model.get_model_info()
        
        assert "model_type" in info
        assert "config" in info
        assert "is_trained" in info
        assert "training_epochs" in info
        assert "supported_behaviors" in info
        
        assert info["model_type"] == "MockHumanBehaviorModel"
        assert info["is_trained"] is False
        assert info["training_epochs"] == 0


@pytest.mark.unit
class TestBehaviorType:
    """Test cases for BehaviorType enum."""
    
    def test_behavior_type_values(self):
        """Test that behavior type enum has expected values."""
        expected_types = [
            "gesture", "handover", "reaching", "pointing", "idle", "unknown"
        ]
        actual_types = [behavior.value for behavior in BehaviorType]
        
        for expected in expected_types:
            assert expected in actual_types