"""
Gaussian Process-based Human Behavior Model implementation.

This module implements a concrete HumanBehaviorModel using Gaussian Processes
for trajectory prediction and Bayesian inference for uncertainty quantification.

Mathematical Foundation:
- GP trajectory model: f(t) ~ GP(μ(t), k(t,t'))
- Predictive distribution: p(f*|X,y) = ∫ p(f*|f)p(f|X,y)df
- Bayesian model averaging for uncertainty
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle

from ..models.human_behavior import (
    HumanBehaviorModel, HumanState, BehaviorPrediction, BehaviorType
)
from ..models.gaussian_process import GaussianProcess, MultiOutputGP, GPParameters
from ..data.feature_extraction import FeatureExtractor, FeatureConfig, ExtractedFeatures
from ..data.synthetic_generator import SyntheticSequence
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GPBehaviorConfig:
    """
    Configuration for GP-based behavior model.
    
    Attributes:
        kernel_type: GP kernel type ('rbf', 'matern32', 'matern52')
        optimize_hyperparams: Whether to optimize GP hyperparameters
        prediction_horizon: Default prediction horizon (seconds)
        trajectory_window: Window size for trajectory context
        uncertainty_threshold: Threshold for high uncertainty detection
        online_update: Whether to enable online model updates
        feature_config: Configuration for feature extraction
    """
    kernel_type: str = 'matern52'
    optimize_hyperparams: bool = True
    prediction_horizon: float = 2.0
    trajectory_window: int = 50
    uncertainty_threshold: float = 0.5
    online_update: bool = True
    feature_config: Optional[FeatureConfig] = None
    
    def __post_init__(self) -> None:
        """Initialize feature config if not provided."""
        if self.feature_config is None:
            self.feature_config = FeatureConfig()


class GPHumanBehaviorModel(HumanBehaviorModel):
    """
    Gaussian Process-based human behavior model.
    
    This model uses GPs for smooth trajectory prediction with uncertainty
    quantification, combined with Bayesian inference for intent recognition.
    
    Key Features:
    - Multi-output GP for 3D trajectory prediction
    - Feature-based behavior classification
    - Online learning and adaptation
    - Uncertainty quantification for safe interaction
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize GP behavior model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        
        # Parse GP-specific configuration
        if 'gp_config' in config:
            self.gp_config = config['gp_config']
        else:
            self.gp_config = GPBehaviorConfig()
        
        # Initialize components
        self.trajectory_gp: Optional[MultiOutputGP] = None
        self.feature_extractor = FeatureExtractor(self.gp_config.feature_config)
        
        # Training data storage
        self.trajectory_data: List[Tuple[np.ndarray, np.ndarray]] = []
        self.behavior_labels: List[BehaviorType] = []
        
        # State tracking
        self.current_trajectory: List[np.ndarray] = []
        self.current_timestamps: List[float] = []
        self.last_prediction: Optional[BehaviorPrediction] = None
        
        logger.info(f"Initialized GP behavior model with {self.gp_config.kernel_type} kernel")
    
    def _initialize_model(self) -> None:
        """Initialize the GP trajectory model and feature extractor."""
        # Initialize GP parameters based on expected data characteristics
        initial_params = GPParameters(
            length_scale=0.5,  # Reasonable for human motion timescales
            output_scale=0.1,  # Typical position variance
            noise_variance=0.01,  # Measurement noise
            kernel_type=self.gp_config.kernel_type
        )
        
        # Create multi-output GP for 3D trajectories
        self.trajectory_gp = MultiOutputGP(
            kernel_type=self.gp_config.kernel_type,
            initial_params=initial_params,
            optimize_hyperparams=self.gp_config.optimize_hyperparams,
            n_outputs=3  # x, y, z positions
        )
        
        logger.info("GP behavior model initialized")
    
    def observe(self, human_state: HumanState) -> None:
        """
        Process new human state observation.
        
        Args:
            human_state: Current human state observation
        """
        # Add to current trajectory
        self.current_trajectory.append(human_state.position)
        self.current_timestamps.append(human_state.timestamp)
        
        # Maintain sliding window
        if len(self.current_trajectory) > self.gp_config.trajectory_window:
            self.current_trajectory.pop(0)
            self.current_timestamps.pop(0)
        
        # Online model update if enabled and sufficient data
        if (self.gp_config.online_update and 
            len(self.current_trajectory) >= 10 and 
            self.trajectory_gp is not None and 
            self.trajectory_gp.is_fitted):
            
            try:
                # Prepare recent trajectory data
                recent_positions = np.array(self.current_trajectory[-10:])
                recent_times = np.array(self.current_timestamps[-10:])
                recent_times = recent_times - recent_times[0]  # Normalize time
                
                # Update GP with recent data
                self.trajectory_gp.update(
                    recent_times[:, None], recent_positions
                )
                
                logger.debug(f"Updated GP model with recent trajectory data")
                
            except Exception as e:
                logger.warning(f"Online model update failed: {e}")
    
    def predict_behavior(
        self,
        current_state: HumanState,
        time_horizon: float,
        num_samples: int = 1
    ) -> List[BehaviorPrediction]:
        """
        Predict future human behavior using GP trajectory model.
        
        Mathematical approach:
        1. Use GP to predict trajectory: p(f*|X,y) 
        2. Extract features from predicted trajectory
        3. Classify behavior type with uncertainty
        4. Generate prediction with confidence bounds
        
        Args:
            current_state: Current human state
            time_horizon: Prediction time horizon (seconds)
            num_samples: Number of prediction samples
            
        Returns:
            List of behavior predictions with uncertainty
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Ensure we have sufficient trajectory context
        if len(self.current_trajectory) < 5:
            # Insufficient context - return low-confidence idle prediction
            return [BehaviorPrediction(
                behavior_type=BehaviorType.UNKNOWN,
                probability=0.1,
                predicted_trajectory=np.array([current_state.position]),
                time_horizon=time_horizon,
                confidence=0.1,
                metadata={'reason': 'insufficient_trajectory_context'}
            )]
        
        predictions = []
        
        for sample_idx in range(num_samples):
            try:
                # Predict trajectory using GP
                predicted_trajectory, prediction_uncertainty = self._predict_trajectory(
                    current_state, time_horizon
                )
                
                # Extract features from predicted trajectory
                behavior_type, probability, confidence = self._classify_behavior(
                    predicted_trajectory, prediction_uncertainty
                )
                
                # Create behavior prediction
                prediction = BehaviorPrediction(
                    behavior_type=behavior_type,
                    probability=probability,
                    predicted_trajectory=predicted_trajectory,
                    time_horizon=time_horizon,
                    confidence=confidence,
                    metadata={
                        'gp_uncertainty': np.mean(prediction_uncertainty),
                        'trajectory_length': len(predicted_trajectory),
                        'sample_index': sample_idx
                    }
                )
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.warning(f"Prediction sample {sample_idx} failed: {e}")
                # Add fallback prediction
                predictions.append(BehaviorPrediction(
                    behavior_type=BehaviorType.UNKNOWN,
                    probability=0.1,
                    predicted_trajectory=np.array([current_state.position]),
                    time_horizon=time_horizon,
                    confidence=0.1,
                    metadata={'error': str(e)}
                ))
        
        # Sort predictions by probability (descending)
        predictions.sort(key=lambda x: x.probability, reverse=True)
        
        # Cache last prediction
        if predictions:
            self.last_prediction = predictions[0]
        
        return predictions
    
    def _predict_trajectory(
        self,
        current_state: HumanState,
        time_horizon: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict trajectory using GP model.
        
        Args:
            current_state: Current human state
            time_horizon: Prediction horizon
            
        Returns:
            Tuple of (predicted_positions, uncertainties)
        """
        if self.trajectory_gp is None or not self.trajectory_gp.is_fitted:
            raise RuntimeError("GP trajectory model not fitted")
        
        # Prepare input data (recent trajectory)
        recent_positions = np.array(self.current_trajectory)
        recent_times = np.array(self.current_timestamps)
        
        # Normalize times relative to start
        recent_times = recent_times - recent_times[0]
        
        # Create prediction time points
        current_time = recent_times[-1]
        prediction_times = np.linspace(
            current_time,
            current_time + time_horizon,
            int(time_horizon * 30)  # 30 Hz prediction
        )
        
        # Make GP predictions
        predicted_means, predicted_stds = self.trajectory_gp.predict(
            prediction_times[:, None], return_std=True
        )
        
        return predicted_means, predicted_stds
    
    def _classify_behavior(
        self,
        trajectory: np.ndarray,
        uncertainty: np.ndarray
    ) -> Tuple[BehaviorType, float, float]:
        """
        Classify behavior type from predicted trajectory.
        
        This is a simplified classification based on trajectory characteristics.
        In practice, this would use trained classifiers.
        
        Args:
            trajectory: Predicted trajectory [N, 3]
            uncertainty: Prediction uncertainties [N, 3]
            
        Returns:
            Tuple of (behavior_type, probability, confidence)
        """
        if len(trajectory) < 2:
            return BehaviorType.IDLE, 0.5, 0.1
        
        # Compute trajectory characteristics
        total_displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
        velocities = np.diff(trajectory, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        max_speed = np.max(speeds) if len(speeds) > 0 else 0
        mean_uncertainty = np.mean(uncertainty)
        
        # Simple rule-based classification
        # (In practice, would use trained ML model)
        if total_displacement < 0.05:
            # Minimal movement - likely idle
            behavior_type = BehaviorType.IDLE
            probability = 0.8 - mean_uncertainty
        elif max_speed > 0.5:
            # Fast movement - likely reaching
            if total_displacement > 0.3:
                behavior_type = BehaviorType.REACHING
                probability = 0.7 - mean_uncertainty
            else:
                behavior_type = BehaviorType.GESTURE
                probability = 0.6 - mean_uncertainty
        elif total_displacement > 0.2:
            # Moderate movement - could be reaching or handover
            # Check if movement is toward interaction zone
            final_position = trajectory[-1]
            if final_position[1] < 0.2 and final_position[2] < 1.0:  # Lower, closer
                behavior_type = BehaviorType.HANDOVER
                probability = 0.6 - mean_uncertainty
            else:
                behavior_type = BehaviorType.REACHING
                probability = 0.5 - mean_uncertainty
        else:
            # Small movement - gesture or adjustment
            behavior_type = BehaviorType.GESTURE
            probability = 0.4 - mean_uncertainty
        
        # Ensure probability is in valid range
        probability = np.clip(probability, 0.1, 0.9)
        
        # Confidence inversely related to uncertainty
        confidence = 1.0 - min(mean_uncertainty / self.gp_config.uncertainty_threshold, 0.9)
        confidence = np.clip(confidence, 0.1, 0.9)
        
        return behavior_type, probability, confidence
    
    def update_model(
        self,
        observations: List[HumanState],
        ground_truth: Optional[List[BehaviorPrediction]] = None
    ) -> Dict[str, float]:
        """
        Update model with new observation data.
        
        Args:
            observations: List of human state observations
            ground_truth: Optional ground truth behavior labels
            
        Returns:
            Update metrics dictionary
        """
        if len(observations) < 2:
            logger.warning("Need at least 2 observations for trajectory modeling")
            return {'error': 'insufficient_data'}
        
        # Extract trajectory data
        positions = np.array([obs.position for obs in observations])
        timestamps = np.array([obs.timestamp for obs in observations])
        
        # Normalize timestamps
        timestamps = timestamps - timestamps[0]
        
        # Add to training data
        self.trajectory_data.append((timestamps[:, None], positions))
        
        # Add behavior labels if provided
        if ground_truth:
            for pred in ground_truth:
                self.behavior_labels.append(pred.behavior_type)
        
        # Refit GP model with accumulated data
        if self.trajectory_gp is not None:
            try:
                # Combine all trajectory data
                all_times = []
                all_positions = []
                
                for times, positions in self.trajectory_data[-10:]:  # Use recent data
                    all_times.append(times)
                    all_positions.append(positions)
                
                if all_times:
                    combined_times = np.vstack(all_times)
                    combined_positions = np.vstack(all_positions)
                    
                    # Refit GP
                    self.trajectory_gp.fit(combined_times, combined_positions)
                    
                    # Compute metrics
                    log_likelihood = self.trajectory_gp.log_marginal_likelihood()
                    
                    return {
                        'log_likelihood': log_likelihood,
                        'n_training_points': len(combined_times),
                        'update_success': True
                    }
                    
            except Exception as e:
                logger.error(f"GP model update failed: {e}")
                return {'error': str(e), 'update_success': False}
        
        return {'update_success': False}
    
    def get_intent_probability(
        self,
        current_state: HumanState,
        intent: str
    ) -> float:
        """
        Get probability of specific intent.
        
        Args:
            current_state: Current human state
            intent: Intent string identifier
            
        Returns:
            Intent probability (0-1)
        """
        # Map intent string to behavior type
        intent_mapping = {
            'reach': BehaviorType.REACHING,
            'handover': BehaviorType.HANDOVER,
            'gesture': BehaviorType.GESTURE,
            'point': BehaviorType.POINTING,
            'idle': BehaviorType.IDLE
        }
        
        target_behavior = intent_mapping.get(intent.lower())
        if target_behavior is None:
            logger.warning(f"Unknown intent: {intent}")
            return 0.0
        
        # Get behavior predictions
        try:
            predictions = self.predict_behavior(current_state, time_horizon=1.0)
            
            # Find matching behavior type
            for pred in predictions:
                if pred.behavior_type == target_behavior:
                    return pred.probability
            
            return 0.1  # Default low probability
            
        except Exception as e:
            logger.error(f"Intent probability computation failed: {e}")
            return 0.0
    
    def _train_epoch(
        self,
        training_data: List[Tuple[HumanState, BehaviorPrediction]],
        batch_size: int
    ) -> Dict[str, float]:
        """Train for one epoch and return metrics."""
        if not training_data:
            return {'epoch_loss': 1.0}
        
        # Group training data by sequences
        sequence_data = {}
        for state, prediction in training_data:
            seq_id = f"seq_{hash(state.timestamp) % 1000}"
            if seq_id not in sequence_data:
                sequence_data[seq_id] = {'states': [], 'predictions': []}
            sequence_data[seq_id]['states'].append(state)
            sequence_data[seq_id]['predictions'].append(prediction)
        
        total_loss = 0.0
        n_sequences = 0
        
        for seq_id, data in sequence_data.items():
            try:
                # Update model with sequence data
                metrics = self.update_model(data['states'], data['predictions'])
                if 'log_likelihood' in metrics:
                    total_loss += -metrics['log_likelihood']  # Negative log likelihood as loss
                n_sequences += 1
                
            except Exception as e:
                logger.warning(f"Training sequence {seq_id} failed: {e}")
                total_loss += 10.0  # High penalty for failed sequences
        
        avg_loss = total_loss / max(n_sequences, 1)
        return {'epoch_loss': avg_loss}
    
    def _validate_epoch(
        self,
        validation_data: List[Tuple[HumanState, BehaviorPrediction]]
    ) -> Dict[str, float]:
        """Validate model and return metrics."""
        if not validation_data:
            return {'val_loss': 1.0}
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for state, ground_truth in validation_data:
            try:
                # Make prediction
                predictions = self.predict_behavior(state, time_horizon=1.0)
                
                if predictions:
                    pred = predictions[0]
                    
                    # Compute classification accuracy
                    if pred.behavior_type == ground_truth.behavior_type:
                        correct_predictions += 1
                    
                    # Compute prediction loss (negative log likelihood)
                    prob_loss = -np.log(max(pred.probability, 1e-8))
                    total_loss += prob_loss
                
                total_predictions += 1
                
            except Exception as e:
                logger.warning(f"Validation prediction failed: {e}")
                total_loss += 10.0
                total_predictions += 1
        
        avg_loss = total_loss / max(total_predictions, 1)
        accuracy = correct_predictions / max(total_predictions, 1)
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_samples': total_predictions
        }
    
    def get_uncertainty_estimate(self) -> Dict[str, float]:
        """
        Get uncertainty estimates for current predictions.
        
        Returns:
            Dictionary of uncertainty metrics
        """
        if self.last_prediction is None:
            return {
                'epistemic_uncertainty': 1.0,  # Model uncertainty (high when untrained)
                'aleatoric_uncertainty': 0.5,  # Data uncertainty
                'total_uncertainty': 1.0
            }
        
        # Extract uncertainty from last prediction
        gp_uncertainty = self.last_prediction.metadata.get('gp_uncertainty', 0.5)
        confidence = self.last_prediction.confidence
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = 1.0 - confidence
        
        # Aleatoric uncertainty (data/observation uncertainty) 
        aleatoric = gp_uncertainty
        
        # Total uncertainty (combination)
        total = epistemic + aleatoric - (epistemic * aleatoric)
        
        return {
            'epistemic_uncertainty': float(epistemic),
            'aleatoric_uncertainty': float(aleatoric), 
            'total_uncertainty': float(total),
            'confidence': float(confidence)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        base_info = super().get_model_info()
        
        gp_info = {
            'gp_kernel_type': self.gp_config.kernel_type,
            'prediction_horizon': self.gp_config.prediction_horizon,
            'trajectory_window': self.gp_config.trajectory_window,
            'current_trajectory_length': len(self.current_trajectory),
            'n_training_sequences': len(self.trajectory_data),
        }
        
        if self.trajectory_gp and self.trajectory_gp.is_fitted:
            gp_info.update({
                'gp_log_likelihood': self.trajectory_gp.log_marginal_likelihood(),
                'gp_fitted': True
            })
        else:
            gp_info['gp_fitted'] = False
        
        base_info.update(gp_info)
        return base_info
    
    def save_model(self, filepath: str) -> None:
        """
        Save GP behavior model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        model_data = {
            'config': self.config,
            'gp_config': self.gp_config,
            'trajectory_gp': self.trajectory_gp,
            'feature_extractor': self.feature_extractor,
            'trajectory_data': self.trajectory_data[-50:],  # Keep recent data only
            'behavior_labels': self.behavior_labels[-50:],
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"GP behavior model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'GPHumanBehaviorModel':
        """
        Load GP behavior model from disk.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded GP behavior model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new model instance
        model = cls(model_data['config'])
        
        # Restore saved components
        model.gp_config = model_data['gp_config']
        model.trajectory_gp = model_data['trajectory_gp']
        model.feature_extractor = model_data['feature_extractor']
        model.trajectory_data = model_data['trajectory_data']
        model.behavior_labels = model_data['behavior_labels']
        model.is_trained = model_data['is_trained']
        model.training_history = model_data['training_history']
        
        logger.info(f"GP behavior model loaded from {filepath}")
        return model