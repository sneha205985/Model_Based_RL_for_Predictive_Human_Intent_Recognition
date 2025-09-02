"""
Abstract base class for human intent prediction.

This module defines the interface for models that predict human intentions
and goals based on observed behavior patterns and contextual information.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum

from .human_behavior import HumanState, BehaviorPrediction


class IntentType(Enum):
    """Enumeration of different intent types."""
    REACH_OBJECT = "reach_object"
    HANDOVER_TO_ROBOT = "handover_to_robot"
    HANDOVER_TO_HUMAN = "handover_to_human"
    POINT_TO_LOCATION = "point_to_location"
    PICK_UP_OBJECT = "pick_up_object"
    PLACE_OBJECT = "place_object"
    GESTURE_COMMUNICATION = "gesture_communication"
    IDLE_WAITING = "idle_waiting"
    UNKNOWN = "unknown"


class UncertaintyType(Enum):
    """Types of uncertainty in intent prediction."""
    ALEATORIC = "aleatoric"  # Data uncertainty
    EPISTEMIC = "epistemic"  # Model uncertainty
    TOTAL = "total"          # Combined uncertainty


@dataclass
class IntentPrediction:
    """
    Represents a prediction of human intent.
    
    Attributes:
        intent_type: Predicted intent type
        probability: Probability of this intent (0-1)
        confidence: Confidence in the prediction (0-1)
        target_object: Target object ID if applicable
        target_location: Target 3D location if applicable
        time_to_completion: Estimated time to complete intent (seconds)
        uncertainty: Uncertainty estimates by type
        context_factors: Contextual factors influencing the prediction
    """
    intent_type: IntentType
    probability: float
    confidence: float
    target_object: Optional[str] = None
    target_location: Optional[np.ndarray] = None
    time_to_completion: Optional[float] = None
    uncertainty: Dict[UncertaintyType, float] = None
    context_factors: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """Validate the intent prediction data."""
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Probability must be between 0 and 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        if self.target_location is not None and self.target_location.shape != (3,):
            raise ValueError("Target location must be a 3D vector")
        if self.time_to_completion is not None and self.time_to_completion < 0:
            raise ValueError("Time to completion must be non-negative")
        if self.uncertainty is None:
            self.uncertainty = {}
        if self.context_factors is None:
            self.context_factors = {}


@dataclass
class ContextInformation:
    """
    Contextual information for intent prediction.
    
    Attributes:
        objects_in_scene: List of object IDs and their properties
        robot_state: Current robot state and capabilities
        environment_constraints: Physical constraints in the environment
        interaction_history: History of past interactions
        social_context: Social context factors (e.g., gaze, attention)
        temporal_context: Temporal patterns and routines
    """
    objects_in_scene: Dict[str, Dict[str, Any]]
    robot_state: Dict[str, Any]
    environment_constraints: Dict[str, Any]
    interaction_history: List[Dict[str, Any]]
    social_context: Dict[str, Any] = None
    temporal_context: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """Initialize optional fields."""
        if self.social_context is None:
            self.social_context = {}
        if self.temporal_context is None:
            self.temporal_context = {}


class IntentPredictor(ABC):
    """
    Abstract base class for human intent prediction.
    
    This class defines the interface for models that can predict human
    intentions based on observed behavior, contextual information, and
    learned patterns from previous interactions.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the intent predictor.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self.is_trained = False
        self.intent_history: List[IntentPrediction] = []
        self.supported_intents = list(IntentType)
        self._initialize_predictor()
    
    @abstractmethod
    def _initialize_predictor(self) -> None:
        """
        Initialize the internal predictor architecture and parameters.
        
        This method should set up the model's internal structure for
        intent prediction, including neural networks, probabilistic models,
        or other computational components.
        """
        pass
    
    @abstractmethod
    def predict_intent(
        self,
        human_state: HumanState,
        context: ContextInformation,
        time_horizon: float = 5.0
    ) -> List[IntentPrediction]:
        """
        Predict human intent based on current state and context.
        
        Args:
            human_state: Current human state observation
            context: Contextual information about the environment
            time_horizon: Time horizon for intent prediction (seconds)
        
        Returns:
            List of intent predictions sorted by probability (descending)
        
        Raises:
            RuntimeError: If predictor is not trained
        """
        pass
    
    @abstractmethod
    def predict_intent_sequence(
        self,
        state_sequence: List[HumanState],
        context: ContextInformation,
        sequence_length: int = 10
    ) -> List[List[IntentPrediction]]:
        """
        Predict sequence of intents based on state history.
        
        Args:
            state_sequence: Sequence of human states
            context: Contextual information
            sequence_length: Length of intent sequence to predict
        
        Returns:
            List of intent prediction lists for each time step
        """
        pass
    
    @abstractmethod
    def update_with_feedback(
        self,
        prediction: IntentPrediction,
        actual_intent: IntentType,
        feedback_type: str = "binary"
    ) -> Dict[str, float]:
        """
        Update the predictor with feedback about prediction accuracy.
        
        Args:
            prediction: Original intent prediction
            actual_intent: Observed actual intent
            feedback_type: Type of feedback ("binary", "continuous", "weighted")
        
        Returns:
            Dictionary containing update metrics
        """
        pass
    
    @abstractmethod
    def estimate_uncertainty(
        self,
        human_state: HumanState,
        context: ContextInformation,
        uncertainty_type: UncertaintyType = UncertaintyType.TOTAL
    ) -> Dict[IntentType, float]:
        """
        Estimate prediction uncertainty for each intent type.
        
        Args:
            human_state: Current human state
            context: Contextual information
            uncertainty_type: Type of uncertainty to estimate
        
        Returns:
            Dictionary mapping intent types to uncertainty values
        """
        pass
    
    def get_intent_transition_matrix(self) -> np.ndarray:
        """
        Get the learned intent transition matrix.
        
        Returns:
            Matrix of transition probabilities between intent types
        
        Note:
            Default implementation returns uniform transitions.
            Subclasses should override with learned transitions.
        """
        n_intents = len(self.supported_intents)
        return np.ones((n_intents, n_intents)) / n_intents
    
    def compute_intent_likelihood(
        self,
        intent: IntentType,
        human_state: HumanState,
        context: ContextInformation
    ) -> float:
        """
        Compute likelihood of specific intent given state and context.
        
        Args:
            intent: Intent type to evaluate
            human_state: Current human state
            context: Contextual information
        
        Returns:
            Likelihood of the specified intent (0-1)
        
        Note:
            Default implementation delegates to predict_intent.
            Subclasses may override for more efficient computation.
        """
        predictions = self.predict_intent(human_state, context)
        for pred in predictions:
            if pred.intent_type == intent:
                return pred.probability
        return 0.0
    
    def train_predictor(
        self,
        training_data: List[Tuple[HumanState, ContextInformation, IntentType]],
        validation_data: Optional[List[Tuple[HumanState, ContextInformation, IntentType]]] = None,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """
        Train the intent predictor on labeled data.
        
        Args:
            training_data: List of (state, context, intent) tuples for training
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
        
        training_history = []
        
        for epoch in range(epochs):
            epoch_metrics = self._train_predictor_epoch(training_data, batch_size)
            
            if validation_data:
                val_metrics = self._validate_predictor_epoch(validation_data)
                epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            
            training_history.append(epoch_metrics)
        
        self.is_trained = True
        return {"epoch_metrics": training_history}
    
    @abstractmethod
    def _train_predictor_epoch(
        self,
        training_data: List[Tuple[HumanState, ContextInformation, IntentType]],
        batch_size: int
    ) -> Dict[str, float]:
        """Train predictor for one epoch and return metrics."""
        pass
    
    @abstractmethod
    def _validate_predictor_epoch(
        self,
        validation_data: List[Tuple[HumanState, ContextInformation, IntentType]]
    ) -> Dict[str, float]:
        """Validate predictor and return metrics."""
        pass
    
    def save_predictor(self, filepath: str) -> None:
        """
        Save the trained predictor to disk.
        
        Args:
            filepath: Path to save the predictor
        
        Raises:
            RuntimeError: If predictor is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained predictor")
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_predictor(cls, filepath: str) -> 'IntentPredictor':
        """
        Load a trained predictor from disk.
        
        Args:
            filepath: Path to the saved predictor
        
        Returns:
            Loaded predictor instance
        """
        import pickle
        with open(filepath, 'rb') as f:
            predictor = pickle.load(f)
        return predictor
    
    def get_predictor_info(self) -> Dict[str, Any]:
        """
        Get information about the predictor configuration and status.
        
        Returns:
            Dictionary containing predictor information
        """
        return {
            "predictor_type": self.__class__.__name__,
            "config": self.config,
            "is_trained": self.is_trained,
            "supported_intents": [intent.value for intent in self.supported_intents],
            "prediction_history_length": len(self.intent_history)
        }