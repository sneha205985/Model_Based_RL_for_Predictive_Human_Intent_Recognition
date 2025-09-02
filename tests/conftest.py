"""
Pytest configuration and shared fixtures.

This module defines common test fixtures and configuration that can be
used across all test modules in the project.
"""

import pytest
import numpy as np
from typing import Dict, Any
from unittest.mock import Mock

from src.models.human_behavior import HumanState, BehaviorPrediction, BehaviorType
from src.models.intent_predictor import IntentPrediction, IntentType, ContextInformation
from src.controllers.mpc_controller import RobotState, ControlAction, MPCConfiguration
from src.agents.bayesian_rl_agent import StateAction, Episode


@pytest.fixture
def sample_human_state() -> HumanState:
    """Create a sample human state for testing."""
    return HumanState(
        position=np.array([1.0, 0.5, 1.2]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # quaternion
        joint_positions={
            'head': np.array([1.0, 0.5, 1.5]),
            'left_hand': np.array([0.8, 0.3, 1.0]),
            'right_hand': np.array([1.2, 0.3, 1.0])
        },
        velocity=np.array([0.1, 0.0, 0.0]),
        timestamp=1234567890.0,
        confidence=0.95
    )


@pytest.fixture
def sample_robot_state() -> RobotState:
    """Create a sample robot state for testing."""
    return RobotState(
        joint_positions=np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0]),
        joint_velocities=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        end_effector_pose=np.array([0.5, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]),
        timestamp=1234567890.0
    )


@pytest.fixture
def sample_control_action() -> ControlAction:
    """Create a sample control action for testing."""
    return ControlAction(
        joint_torques=np.array([0.1, 0.2, 0.0, -0.1, 0.0, 0.1, 0.0]),
        execution_time=0.1,
        timestamp=1234567890.0
    )


@pytest.fixture
def sample_behavior_prediction() -> BehaviorPrediction:
    """Create a sample behavior prediction for testing."""
    return BehaviorPrediction(
        behavior_type=BehaviorType.REACHING,
        probability=0.8,
        predicted_trajectory=np.array([[1.0, 0.5, 1.2], [1.1, 0.5, 1.1]]),
        time_horizon=2.0,
        confidence=0.9
    )


@pytest.fixture
def sample_intent_prediction() -> IntentPrediction:
    """Create a sample intent prediction for testing."""
    return IntentPrediction(
        intent_type=IntentType.REACH_OBJECT,
        probability=0.75,
        confidence=0.85,
        target_object="cup",
        target_location=np.array([0.8, 0.3, 1.0]),
        time_to_completion=3.0
    )


@pytest.fixture
def sample_context_information() -> ContextInformation:
    """Create sample context information for testing."""
    return ContextInformation(
        objects_in_scene={
            "cup": {"position": [0.8, 0.3, 1.0], "type": "container"},
            "bottle": {"position": [1.2, 0.2, 1.0], "type": "container"}
        },
        robot_state={"status": "idle", "battery": 0.95},
        environment_constraints={"workspace_bounds": [-1.0, 1.5, -1.0, 1.0, 0.0, 2.0]},
        interaction_history=[]
    )


@pytest.fixture
def sample_state_action(
    sample_robot_state: RobotState,
    sample_human_state: HumanState,
    sample_control_action: ControlAction,
    sample_context_information: ContextInformation
) -> StateAction:
    """Create a sample state-action pair for testing."""
    return StateAction(
        robot_state=sample_robot_state,
        human_state=sample_human_state,
        context=sample_context_information,
        action=sample_control_action,
        timestamp=1234567890.0
    )


@pytest.fixture
def sample_episode(sample_state_action: StateAction) -> Episode:
    """Create a sample episode for testing."""
    return Episode(
        state_action_sequence=[sample_state_action],
        rewards=[1.0],
        terminal_state=sample_state_action,
        episode_return=1.0,
        episode_length=1,
        success=True
    )


@pytest.fixture
def sample_mpc_config() -> MPCConfiguration:
    """Create a sample MPC configuration for testing."""
    return MPCConfiguration(
        prediction_horizon=10,
        control_horizon=5,
        sampling_time=0.1,
        state_weights={"position": 1.0, "velocity": 0.1},
        control_weights={"torque": 0.01},
        terminal_weights={"position": 10.0}
    )


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Create a mock configuration dictionary for testing."""
    return {
        "model_type": "test_model",
        "learning_rate": 0.01,
        "batch_size": 32,
        "max_iterations": 100,
        "convergence_tolerance": 1e-6
    }


@pytest.fixture
def numpy_random_seed():
    """Set numpy random seed for reproducible tests."""
    np.random.seed(42)
    yield
    # Reset seed after test
    np.random.seed(None)


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return Mock()


class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def generate_trajectory(length: int, dims: int = 3) -> np.ndarray:
        """Generate a random trajectory for testing."""
        return np.random.rand(length, dims)
    
    @staticmethod
    def generate_human_states(count: int) -> list:
        """Generate a list of human states for testing."""
        states = []
        for i in range(count):
            state = HumanState(
                position=np.random.rand(3),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                joint_positions={
                    'head': np.random.rand(3),
                    'left_hand': np.random.rand(3),
                    'right_hand': np.random.rand(3)
                },
                velocity=np.random.rand(3) - 0.5,
                timestamp=1234567890.0 + i * 0.1,
                confidence=0.8 + 0.2 * np.random.rand()
            )
            states.append(state)
        return states


@pytest.fixture
def test_data_generator():
    """Provide test data generator utility."""
    return TestDataGenerator()