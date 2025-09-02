"""
Abstract base class for human behavior modeling.

This module defines the interface for modeling human behavior patterns
in human-robot interaction scenarios, including gesture recognition,
motion prediction, and intent inference.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
from dataclasses import dataclass
from enum import Enum


class BehaviorType(Enum):
    """Enumeration of different human behavior types."""
    GESTURE = "gesture"
    HANDOVER = "handover"
    REACHING = "reaching"
    POINTING = "pointing"
    IDLE = "idle"
    UNKNOWN = "unknown"


@dataclass
class HumanState:
    """
    Represents the current state of a human in the environment.
    
    Attributes:
        position: 3D position of the human (x, y, z)
        orientation: Quaternion representing orientation (w, x, y, z)
        joint_positions: Dictionary mapping joint names to 3D positions
        velocity: 3D velocity vector
        timestamp: Unix timestamp of the observation
        confidence: Confidence score of the state estimation (0-1)
    """
    position: np.ndarray
    orientation: np.ndarray
    joint_positions: Dict[str, np.ndarray]
    velocity: np.ndarray
    timestamp: float
    confidence: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate the human state data."""
        if self.position.shape != (3,):
            raise ValueError("Position must be a 3D vector")
        if self.orientation.shape != (4,):
            raise ValueError("Orientation must be a quaternion (4D vector)")
        if self.velocity.shape != (3,):
            raise ValueError("Velocity must be a 3D vector")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class BehaviorPrediction:
    """
    Represents a prediction of human behavior.
    
    Attributes:
        behavior_type: Predicted behavior type
        probability: Probability of this behavior (0-1)
        predicted_trajectory: Predicted future trajectory points
        time_horizon: Time horizon of the prediction (seconds)
        confidence: Overall confidence in the prediction (0-1)
        metadata: Additional metadata about the prediction
    """
    behavior_type: BehaviorType
    probability: float
    predicted_trajectory: np.ndarray
    time_horizon: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """Validate the behavior prediction data."""
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Probability must be between 0 and 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        if self.time_horizon <= 0:
            raise ValueError("Time horizon must be positive")
        if self.metadata is None:
            self.metadata = {}


class HumanBehaviorModel(ABC):
    """
    Abstract base class for human behavior modeling.
    
    This class defines the interface for models that can observe human behavior,
    learn patterns, and make predictions about future actions and intents.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the human behavior model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self.is_trained = False
        self.training_history: List[Dict[str, float]] = []
        self._initialize_model()
    
    @abstractmethod
    def _initialize_model(self) -> None:
        """
        Initialize the internal model architecture and parameters.
        
        This method should set up the model's internal structure,
        including neural networks, statistical models, or other
        computational components.
        """
        pass
    
    @abstractmethod
    def observe(self, human_state: HumanState) -> None:
        """
        Process a new observation of human state.
        
        Args:
            human_state: Current state of the human
        
        Note:
            This method should update the model's internal representation
            of the human's current state and may trigger online learning
            or adaptation mechanisms.
        """
        pass
    
    @abstractmethod
    def predict_behavior(
        self,
        current_state: HumanState,
        time_horizon: float,
        num_samples: int = 1
    ) -> List[BehaviorPrediction]:
        """
        Predict future human behavior based on current state.
        
        Args:
            current_state: Current human state
            time_horizon: Time horizon for prediction (seconds)
            num_samples: Number of prediction samples to generate
        
        Returns:
            List of behavior predictions sorted by probability (descending)
        
        Raises:
            RuntimeError: If model is not trained
        """
        pass
    
    @abstractmethod
    def update_model(
        self,
        observations: List[HumanState],
        ground_truth: Optional[List[BehaviorPrediction]] = None
    ) -> Dict[str, float]:
        """
        Update the model with new training data.
        
        Args:
            observations: List of human state observations
            ground_truth: Optional ground truth behavior labels
        
        Returns:
            Dictionary containing training metrics
        
        Note:
            This method implements online or batch learning to improve
            the model's predictive performance based on new data.
        """
        pass
    
    @abstractmethod
    def get_intent_probability(
        self,
        current_state: HumanState,
        intent: str
    ) -> float:
        """
        Get the probability of a specific intent given current state.
        
        Args:
            current_state: Current human state
            intent: Intent string (e.g., "reach_object", "handover")
        
        Returns:
            Probability of the specified intent (0-1)
        """
        pass
    
    def train(
        self,
        training_data: List[Tuple[HumanState, BehaviorPrediction]],
        validation_data: Optional[List[Tuple[HumanState, BehaviorPrediction]]] = None,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """
        Train the model on labeled data.
        
        Args:
            training_data: List of (state, behavior) pairs for training
            validation_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Training batch size
        
        Returns:
            Dictionary containing training history metrics
        
        Raises:
            ValueError: If training data is empty or invalid
        """
        if not training_data:
            raise ValueError("Training data cannot be empty")
        
        # Default implementation - subclasses should override for custom training
        self.training_history = []
        
        for epoch in range(epochs):
            epoch_metrics = self._train_epoch(training_data, batch_size)
            
            if validation_data:
                val_metrics = self._validate_epoch(validation_data)
                epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            
            self.training_history.append(epoch_metrics)
        
        self.is_trained = True
        return {"epoch_metrics": self.training_history}
    
    @abstractmethod
    def _train_epoch(
        self,
        training_data: List[Tuple[HumanState, BehaviorPrediction]],
        batch_size: int
    ) -> Dict[str, float]:
        """Train for one epoch and return metrics."""
        pass
    
    @abstractmethod
    def _validate_epoch(
        self,
        validation_data: List[Tuple[HumanState, BehaviorPrediction]]
    ) -> Dict[str, float]:
        """Validate model and return metrics."""
        pass
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        
        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        # Default implementation - subclasses should override
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'HumanBehaviorModel':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            Loaded model instance
        """
        import pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model configuration and training status.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_type": self.__class__.__name__,
            "config": self.config,
            "is_trained": self.is_trained,
            "training_epochs": len(self.training_history),
            "supported_behaviors": [behavior.value for behavior in BehaviorType]
        }