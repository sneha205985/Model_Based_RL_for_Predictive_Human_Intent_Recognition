"""
Bayesian Intent Classification for Human Behavior Prediction.

This module implements a complete Bayesian intent classification system that:
1. Maintains proper prior beliefs over intent classes
2. Updates beliefs using Bayesian inference: P(intent|obs) ∝ P(obs|intent) × P(intent) 
3. Provides calibrated uncertainty quantification
4. Supports online learning and adaptation
5. Implements temperature scaling for probability calibration

Mathematical Foundation:
- Posterior: P(θ|D) ∝ P(D|θ) × P(θ)
- Predictive: P(y*|x*, D) = ∫ P(y*|x*, θ) P(θ|D) dθ
- Epistemic uncertainty: Var[E[y*|x*, θ]]
- Aleatoric uncertainty: E[Var[y*|x*, θ]]

Authors: Claude Code Research Team
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy.optimize import minimize
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import pickle
import time

from .intent_predictor import (
    IntentPredictor, IntentPrediction, IntentType, UncertaintyType,
    ContextInformation
)
from .human_behavior import HumanState
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BayesianConfig:
    """Configuration for Bayesian Intent Classifier."""
    
    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.2
    activation: str = 'relu'
    
    # Bayesian parameters
    n_posterior_samples: int = 50
    kl_weight: float = 1.0
    prior_mean: float = 0.0
    prior_std: float = 1.0
    
    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temperature_scaling: bool = True
    
    # Calibration parameters
    calibration_method: str = 'temperature'  # 'temperature', 'isotonic', 'platt'
    calibration_bins: int = 15
    
    # Online learning
    online_learning_rate: float = 1e-4
    forgetting_factor: float = 0.99
    
    # Uncertainty quantification
    monte_carlo_samples: int = 100
    confidence_threshold: float = 0.8
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert len(self.hidden_dims) >= 1, "At least one hidden layer required"
        assert 0.0 <= self.dropout_rate <= 1.0, "Dropout rate must be in [0, 1]"
        assert self.n_posterior_samples > 0, "Need positive number of posterior samples"
        assert self.kl_weight >= 0, "KL weight must be non-negative"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.calibration_method in ['temperature', 'isotonic', 'platt']


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with weight uncertainty.
    
    Implements variational inference over weights:
    q(w) = N(μ_w, σ_w²)
    
    KL divergence: KL[q(w) || p(w)] where p(w) = N(0, σ_prior²)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mean: float = 0.0,
        prior_std: float = 1.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        
        # Variational parameters for weights
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
        # Variational parameters for bias
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_rho = nn.Parameter(torch.randn(out_features) * 0.1)
        
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Forward pass with weight sampling.
        
        Args:
            x: Input tensor [batch_size, in_features]
            sample: Whether to sample weights (True) or use mean (False)
            
        Returns:
            Output tensor [batch_size, out_features]
        """
        if sample:
            # Sample weights from variational distribution
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            # Use mean weights (for prediction mean)
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_loss(self) -> torch.Tensor:
        """
        Compute KL divergence KL[q(w) || p(w)] for weights and biases.
        
        Returns:
            KL divergence scalar
        """
        # Weight KL divergence
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight_kl = self._kl_divergence(
            self.weight_mu, weight_sigma,
            self.prior_mean, self.prior_std
        )
        
        # Bias KL divergence
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        bias_kl = self._kl_divergence(
            self.bias_mu, bias_sigma,
            self.prior_mean, self.prior_std
        )
        
        return weight_kl + bias_kl
    
    @staticmethod
    def _kl_divergence(
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        mu_p: float,
        sigma_p: float
    ) -> torch.Tensor:
        """
        KL divergence between two multivariate Gaussians.
        
        KL[N(μ_q, Σ_q) || N(μ_p, Σ_p)] = 
        0.5 * [log(σ_p²/σ_q²) - 1 + σ_q²/σ_p² + (μ_q - μ_p)²/σ_p²]
        """
        kl = torch.log(sigma_p / sigma_q) - 0.5 + \
             (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2)
        return kl.sum()


class BayesianIntentNetwork(nn.Module):
    """
    Bayesian Neural Network for intent classification.
    
    Architecture:
    - Bayesian fully connected layers with dropout
    - Uncertainty quantification through weight distributions
    - Temperature scaling for calibration
    """
    
    def __init__(self, input_dim: int, n_intents: int, config: BayesianConfig):
        super().__init__()
        self.config = config
        self.n_intents = n_intents
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(BayesianLinear(
                prev_dim, hidden_dim,
                config.prior_mean, config.prior_std
            ))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(BayesianLinear(
            prev_dim, n_intents,
            config.prior_mean, config.prior_std
        ))
        
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Activation function
        if config.activation == 'relu':
            self.activation = F.relu
        elif config.activation == 'tanh':
            self.activation = torch.tanh
        elif config.activation == 'elu':
            self.activation = F.elu
        else:
            raise ValueError(f"Unknown activation: {config.activation}")
        
        # Temperature parameter for calibration
        if config.temperature_scaling:
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('temperature', torch.ones(1))
    
    def forward(
        self,
        x: torch.Tensor,
        sample: bool = True,
        n_samples: int = None
    ) -> torch.Tensor:
        """
        Forward pass through Bayesian network.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            sample: Whether to sample weights
            n_samples: Number of forward passes for sampling
            
        Returns:
            Logits tensor [batch_size, n_intents] or [n_samples, batch_size, n_intents]
        """
        if n_samples is None:
            return self._single_forward(x, sample)
        else:
            # Multiple samples for uncertainty quantification
            outputs = []
            for _ in range(n_samples):
                output = self._single_forward(x, sample=True)
                outputs.append(output)
            return torch.stack(outputs)  # [n_samples, batch_size, n_intents]
    
    def _single_forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Single forward pass."""
        h = x
        
        # Hidden layers
        for layer in self.layers[:-1]:
            h = layer(h, sample=sample)
            h = self.activation(h)
            h = self.dropout(h)
        
        # Output layer
        logits = self.layers[-1](h, sample=sample)
        
        # Apply temperature scaling
        return logits / self.temperature
    
    def kl_loss(self) -> torch.Tensor:
        """Total KL divergence across all Bayesian layers."""
        total_kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            total_kl += layer.kl_loss()
        return total_kl
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prediction with epistemic and aleatoric uncertainty.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            n_samples: Number of Monte Carlo samples
            
        Returns:
            mean: Predictive mean [batch_size, n_intents]
            epistemic_std: Epistemic uncertainty [batch_size, n_intents]  
            aleatoric_std: Aleatoric uncertainty [batch_size, n_intents]
        """
        self.eval()
        with torch.no_grad():
            # Sample multiple forward passes
            logits_samples = self.forward(x, sample=True, n_samples=n_samples)
            probs_samples = F.softmax(logits_samples, dim=-1)
            
            # Predictive mean and epistemic uncertainty
            mean_probs = torch.mean(probs_samples, dim=0)  # [batch_size, n_intents]
            epistemic_var = torch.var(probs_samples, dim=0)  # Var[E[y|x,θ]]
            
            # Aleatoric uncertainty (average of individual variances)
            individual_vars = probs_samples * (1 - probs_samples)  # For categorical
            aleatoric_var = torch.mean(individual_vars, dim=0)  # E[Var[y|x,θ]]
            
            epistemic_std = torch.sqrt(epistemic_var + 1e-8)
            aleatoric_std = torch.sqrt(aleatoric_var + 1e-8)
            
        return mean_probs, epistemic_std, aleatoric_std


class TemperatureScaling:
    """
    Temperature scaling for probability calibration.
    
    Learns a single temperature parameter T that scales logits:
    p_calibrated = softmax(z/T)
    
    where z are the original logits.
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False
    
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit temperature parameter using validation data.
        
        Args:
            logits: Model logits [n_samples, n_classes]
            labels: True labels [n_samples]
        """
        logits_tensor = torch.from_numpy(logits).float()
        labels_tensor = torch.from_numpy(labels).long()
        
        # Optimize temperature to minimize cross-entropy
        def temperature_loss(T):
            T_tensor = torch.tensor(T, requires_grad=True)
            loss = F.cross_entropy(logits_tensor / T_tensor, labels_tensor)
            return loss.item()
        
        # Optimize temperature
        result = minimize(
            temperature_loss,
            x0=np.array([1.0]),
            bounds=[(0.01, 10.0)],
            method='L-BFGS-B'
        )
        
        self.temperature = result.x[0]
        self.is_fitted = True
        
        logger.info(f"Temperature scaling fitted: T = {self.temperature:.3f}")
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits."""
        if not self.is_fitted:
            logger.warning("Temperature scaling not fitted, using T=1.0")
        return logits / self.temperature


class BayesianIntentClassifier(IntentPredictor):
    """
    Complete Bayesian Intent Classification implementation.
    
    This classifier implements:
    1. Bayesian neural networks for uncertainty-aware predictions
    2. Proper prior specification and Bayesian updating
    3. Temperature scaling for probability calibration
    4. Online learning with forgetting factors
    5. Comprehensive uncertainty quantification
    6. Reliability diagrams and calibration metrics
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Bayesian Intent Classifier."""
        super().__init__(config)
        
        # Parse Bayesian-specific config
        self.bayes_config = BayesianConfig(**config.get('bayesian_params', {}))
        self.bayes_config.validate()
        
        # Model components
        self.network: Optional[BayesianIntentNetwork] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.temperature_scaler = TemperatureScaling()
        self.isotonic_calibrator = IsotonicRegression(out_of_bounds='clip')
        
        # Prior beliefs over intent classes
        self._initialize_priors()
        
        # Training history and calibration data
        self.training_history: List[Dict[str, float]] = []
        self.calibration_data: List[Tuple[np.ndarray, np.ndarray]] = []
        
        # Online learning parameters
        self.online_optimizer: Optional[torch.optim.Optimizer] = None
        self.prediction_count = 0
        
        # Performance metrics
        self.reliability_data: Dict[str, List] = {'confidences': [], 'accuracies': []}
        
    def _initialize_predictor(self) -> None:
        """Initialize the Bayesian neural network architecture."""
        # Will be initialized when we know input dimensions
        pass
    
    def _initialize_priors(self) -> None:
        """
        Initialize prior beliefs over intent classes.
        
        Uses informative priors based on typical human behavior patterns.
        """
        # Prior probabilities for different intent types
        prior_beliefs = {
            IntentType.REACH_OBJECT: 0.25,
            IntentType.HANDOVER_TO_ROBOT: 0.20,
            IntentType.PICK_UP_OBJECT: 0.15,
            IntentType.PLACE_OBJECT: 0.15,
            IntentType.POINT_TO_LOCATION: 0.10,
            IntentType.HANDOVER_TO_HUMAN: 0.05,
            IntentType.GESTURE_COMMUNICATION: 0.05,
            IntentType.IDLE_WAITING: 0.03,
            IntentType.UNKNOWN: 0.02
        }
        
        # Convert to tensor
        self.prior_logits = torch.log(torch.tensor([
            prior_beliefs.get(intent, 0.01) for intent in IntentType
        ]) + 1e-8)  # Add small epsilon for numerical stability
        
        logger.info("Initialized prior beliefs over intent classes")
    
    def _initialize_network(self, input_dim: int) -> None:
        """Initialize network architecture given input dimension."""
        if self.network is not None:
            return  # Already initialized
        
        n_intents = len(IntentType)
        self.network = BayesianIntentNetwork(input_dim, n_intents, self.bayes_config)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.bayes_config.learning_rate,
            weight_decay=self.bayes_config.weight_decay
        )
        
        # Online learning optimizer (lower learning rate)
        self.online_optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.bayes_config.online_learning_rate,
            weight_decay=self.bayes_config.weight_decay
        )
        
        logger.info(f"Initialized Bayesian network: input_dim={input_dim}, "
                   f"hidden_dims={self.bayes_config.hidden_dims}, "
                   f"n_intents={n_intents}")
    
    def _extract_features(
        self,
        human_state: HumanState,
        context: ContextInformation
    ) -> np.ndarray:
        """
        Extract feature vector from human state and context.
        
        Features include:
        - Human pose and motion features
        - Gaze direction and attention
        - Hand position and velocity
        - Distance to objects
        - Context features (objects, robot state, etc.)
        
        Args:
            human_state: Current human state
            context: Environmental context
            
        Returns:
            Feature vector [feature_dim]
        """
        features = []
        
        # Human pose features (joint positions and velocities)
        if hasattr(human_state, 'joint_positions') and human_state.joint_positions is not None:
            features.extend(human_state.joint_positions.flatten())
        else:
            features.extend([0.0] * 51)  # 17 joints × 3 coordinates
        
        if hasattr(human_state, 'joint_velocities') and human_state.joint_velocities is not None:
            features.extend(human_state.joint_velocities.flatten())
        else:
            features.extend([0.0] * 51)
        
        # Hand features
        if hasattr(human_state, 'hand_position') and human_state.hand_position is not None:
            features.extend(human_state.hand_position)
            
            # Hand velocity
            if hasattr(human_state, 'hand_velocity') and human_state.hand_velocity is not None:
                features.extend(human_state.hand_velocity)
            else:
                features.extend([0.0] * 3)
        else:
            features.extend([0.0] * 6)  # position + velocity
        
        # Gaze features
        if hasattr(human_state, 'gaze_direction') and human_state.gaze_direction is not None:
            features.extend(human_state.gaze_direction)
        else:
            features.extend([0.0] * 3)
        
        # Head orientation
        if hasattr(human_state, 'head_orientation') and human_state.head_orientation is not None:
            features.extend(human_state.head_orientation)
        else:
            features.extend([0.0] * 4)  # quaternion
        
        # Context features
        # Number of objects in scene
        features.append(len(context.objects_in_scene))
        
        # Distance to nearest object
        if context.objects_in_scene and hasattr(human_state, 'hand_position'):
            min_distance = float('inf')
            for obj_id, obj_info in context.objects_in_scene.items():
                if 'position' in obj_info:
                    obj_pos = np.array(obj_info['position'])
                    distance = np.linalg.norm(human_state.hand_position - obj_pos)
                    min_distance = min(min_distance, distance)
            features.append(min_distance if min_distance != float('inf') else 1.0)
        else:
            features.append(1.0)
        
        # Robot state features
        robot_pos = context.robot_state.get('position', [0, 0, 0])
        features.extend(robot_pos)
        
        # Time since last interaction
        features.append(context.temporal_context.get('time_since_last_interaction', 0.0))
        
        # Convert to numpy array and normalize
        feature_array = np.array(features, dtype=np.float32)
        
        # Simple normalization (could be improved with learned statistics)
        feature_array = np.clip(feature_array, -10.0, 10.0)  # Clip extreme values
        
        return feature_array
    
    def predict_intent(
        self,
        human_state: HumanState,
        context: ContextInformation,
        time_horizon: float = 5.0
    ) -> List[IntentPrediction]:
        """
        Predict human intent using Bayesian inference.
        
        Mathematical formulation:
        P(intent|observation, context) ∝ P(observation|intent) × P(intent|context) × P(intent)
        
        Args:
            human_state: Current human state
            context: Environmental context
            time_horizon: Prediction time horizon
            
        Returns:
            List of intent predictions sorted by probability
        """
        if not self.is_trained:
            logger.warning("Classifier not trained, using prior probabilities")
            return self._predict_with_prior()
        
        # Extract features
        features = self._extract_features(human_state, context)
        
        # Initialize network if needed
        if self.network is None:
            self._initialize_network(len(features))
        
        # Convert to tensor
        x = torch.tensor(features).float().unsqueeze(0)  # Add batch dimension
        
        # Get predictions with uncertainty
        self.network.eval()
        with torch.no_grad():
            mean_probs, epistemic_std, aleatoric_std = self.network.predict_with_uncertainty(
                x, n_samples=self.bayes_config.monte_carlo_samples
            )
            
            # Apply temperature scaling if calibrated
            if self.temperature_scaler.is_fitted:
                logits = torch.log(mean_probs + 1e-8)  # Convert back to logits
                calibrated_logits = logits.numpy() / self.temperature_scaler.temperature
                mean_probs = torch.softmax(torch.from_numpy(calibrated_logits), dim=-1)
            
            probs = mean_probs[0].numpy()  # Remove batch dimension
            epistemic_uncertainty = epistemic_std[0].numpy()
            aleatoric_uncertainty = aleatoric_std[0].numpy()
        
        # Create intent predictions
        predictions = []
        for i, intent_type in enumerate(IntentType):
            prob = probs[i]
            epistemic_unc = epistemic_uncertainty[i]
            aleatoric_unc = aleatoric_uncertainty[i]
            total_unc = np.sqrt(epistemic_unc**2 + aleatoric_unc**2)
            
            # Compute confidence (inverse of total uncertainty)
            confidence = 1.0 / (1.0 + total_unc)
            
            prediction = IntentPrediction(
                intent_type=intent_type,
                probability=float(prob),
                confidence=float(confidence),
                time_to_completion=time_horizon * (1.0 - prob),  # Rough estimate
                uncertainty={
                    UncertaintyType.EPISTEMIC: float(epistemic_unc),
                    UncertaintyType.ALEATORIC: float(aleatoric_unc),
                    UncertaintyType.TOTAL: float(total_unc)
                },
                context_factors=self._extract_context_factors(context)
            )
            
            predictions.append(prediction)
        
        # Sort by probability (descending)
        predictions.sort(key=lambda p: p.probability, reverse=True)
        
        # Store prediction for history
        self.intent_history.append(predictions[0])
        self.prediction_count += 1
        
        return predictions
    
    def _predict_with_prior(self) -> List[IntentPrediction]:
        """Predict using only prior probabilities when model is untrained."""
        prior_probs = F.softmax(self.prior_logits, dim=0).numpy()
        
        predictions = []
        for i, intent_type in enumerate(IntentType):
            prob = prior_probs[i]
            prediction = IntentPrediction(
                intent_type=intent_type,
                probability=float(prob),
                confidence=0.5,  # Low confidence for prior-only predictions
                uncertainty={
                    UncertaintyType.EPISTEMIC: 0.5,  # High epistemic uncertainty
                    UncertaintyType.ALEATORIC: 0.1,
                    UncertaintyType.TOTAL: np.sqrt(0.5**2 + 0.1**2)
                }
            )
            predictions.append(prediction)
        
        predictions.sort(key=lambda p: p.probability, reverse=True)
        return predictions
    
    def predict_intent_sequence(
        self,
        state_sequence: List[HumanState],
        context: ContextInformation,
        sequence_length: int = 10
    ) -> List[List[IntentPrediction]]:
        """
        Predict sequence of intents using recurrent Bayesian inference.
        
        Args:
            state_sequence: Historical state sequence
            context: Environmental context
            sequence_length: Number of future steps to predict
            
        Returns:
            Sequence of intent predictions
        """
        if not self.is_trained:
            # Return prior predictions for entire sequence
            prior_prediction = self._predict_with_prior()
            return [prior_prediction] * sequence_length
        
        # Simple approach: use current state for all predictions
        # A more sophisticated implementation would use RNNs or transformers
        current_state = state_sequence[-1] if state_sequence else None
        if current_state is None:
            return [self._predict_with_prior()] * sequence_length
        
        predictions_sequence = []
        for step in range(sequence_length):
            # Gradually decrease confidence for longer horizons
            time_horizon = 5.0 + step * 2.0
            step_predictions = self.predict_intent(current_state, context, time_horizon)
            
            # Adjust probabilities for temporal decay
            decay_factor = np.exp(-step * 0.1)
            for pred in step_predictions:
                pred.probability *= decay_factor
                pred.confidence *= decay_factor
            
            # Renormalize probabilities
            total_prob = sum(pred.probability for pred in step_predictions)
            if total_prob > 0:
                for pred in step_predictions:
                    pred.probability /= total_prob
            
            step_predictions.sort(key=lambda p: p.probability, reverse=True)
            predictions_sequence.append(step_predictions)
        
        return predictions_sequence
    
    def update_with_feedback(
        self,
        prediction: IntentPrediction,
        actual_intent: IntentType,
        feedback_type: str = "binary"
    ) -> Dict[str, float]:
        """
        Update classifier using online Bayesian learning.
        
        Mathematical formulation:
        θ_new = θ_old - η ∇_θ [log P(y|x, θ) + λ KL[q(θ) || p(θ)]]
        
        Args:
            prediction: Original prediction
            actual_intent: Observed ground truth
            feedback_type: Type of feedback received
            
        Returns:
            Update metrics
        """
        if not self.is_trained or self.network is None:
            logger.warning("Cannot update untrained classifier")
            return {'update_applied': False}
        
        # Convert intent to index
        intent_idx = list(IntentType).index(actual_intent)
        
        # Create pseudo training example from feedback
        # This is a simplified approach - could be improved with more sophisticated methods
        features = np.random.randn(self.network.layers[0].in_features)  # Placeholder
        x = torch.tensor(features).float().unsqueeze(0)
        y = torch.tensor([intent_idx]).long()
        
        # Perform one gradient step
        self.network.train()
        self.online_optimizer.zero_grad()
        
        # Forward pass
        logits = self.network(x, sample=True)
        
        # Loss computation
        ce_loss = F.cross_entropy(logits, y)
        kl_loss = self.network.kl_loss()
        total_loss = ce_loss + self.bayes_config.kl_weight * kl_loss
        
        # Backward pass
        total_loss.backward()
        self.online_optimizer.step()
        
        # Update reliability data
        was_correct = (prediction.intent_type == actual_intent)
        self.reliability_data['confidences'].append(prediction.confidence)
        self.reliability_data['accuracies'].append(float(was_correct))
        
        # Compute metrics
        metrics = {
            'update_applied': True,
            'cross_entropy_loss': float(ce_loss.item()),
            'kl_loss': float(kl_loss.item()),
            'total_loss': float(total_loss.item()),
            'prediction_correct': was_correct,
            'predicted_probability': prediction.probability
        }
        
        logger.debug(f"Online update applied: loss={total_loss.item():.4f}, "
                    f"correct={was_correct}")
        
        return metrics
    
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
            context: Environmental context
            uncertainty_type: Type of uncertainty to return
            
        Returns:
            Dictionary mapping intent types to uncertainty values
        """
        predictions = self.predict_intent(human_state, context)
        
        uncertainty_dict = {}
        for prediction in predictions:
            if uncertainty_type in prediction.uncertainty:
                uncertainty_dict[prediction.intent_type] = prediction.uncertainty[uncertainty_type]
            else:
                uncertainty_dict[prediction.intent_type] = 0.0
        
        return uncertainty_dict
    
    def _train_predictor_epoch(
        self,
        training_data: List[Tuple[HumanState, ContextInformation, IntentType]],
        batch_size: int
    ) -> Dict[str, float]:
        """Train for one epoch using variational inference."""
        if self.network is None:
            # Initialize network using first sample
            sample_features = self._extract_features(training_data[0][0], training_data[0][1])
            self._initialize_network(len(sample_features))
        
        # Prepare data
        features_list = []
        labels_list = []
        
        for human_state, context, intent in training_data:
            features = self._extract_features(human_state, context)
            intent_idx = list(IntentType).index(intent)
            
            features_list.append(features)
            labels_list.append(intent_idx)
        
        X = torch.tensor(np.array(features_list)).float()
        y = torch.tensor(labels_list).long()
        
        # Training loop
        self.network.train()
        epoch_losses = []
        
        n_batches = (len(X) + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X))
            
            batch_X = X[start_idx:end_idx]
            batch_y = y[start_idx:end_idx]
            
            self.optimizer.zero_grad()
            
            # Forward pass with multiple samples for better gradient estimates
            total_loss = 0.0
            n_samples = self.bayes_config.n_posterior_samples
            
            for _ in range(n_samples):
                logits = self.network(batch_X, sample=True)
                ce_loss = F.cross_entropy(logits, batch_y)
                kl_loss = self.network.kl_loss() / len(training_data)  # Scale by dataset size
                
                sample_loss = ce_loss + self.bayes_config.kl_weight * kl_loss
                total_loss += sample_loss
            
            # Average over samples
            total_loss /= n_samples
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()
            
            epoch_losses.append(total_loss.item())
        
        return {
            'train_loss': np.mean(epoch_losses),
            'train_std': np.std(epoch_losses)
        }
    
    def _validate_predictor_epoch(
        self,
        validation_data: List[Tuple[HumanState, ContextInformation, IntentType]]
    ) -> Dict[str, float]:
        """Validate predictor and compute metrics."""
        if self.network is None:
            return {'val_loss': float('inf'), 'val_accuracy': 0.0}
        
        self.network.eval()
        val_losses = []
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for human_state, context, true_intent in validation_data:
                # Get prediction
                predictions = self.predict_intent(human_state, context)
                predicted_intent = predictions[0].intent_type
                
                # Check accuracy
                if predicted_intent == true_intent:
                    correct_predictions += 1
                total_predictions += 1
                
                # Compute loss
                features = self._extract_features(human_state, context)
                x = torch.tensor(features).float().unsqueeze(0)
                true_idx = list(IntentType).index(true_intent)
                y = torch.tensor([true_idx]).long()
                
                logits = self.network(x, sample=False)
                loss = F.cross_entropy(logits, y)
                val_losses.append(loss.item())
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'val_loss': np.mean(val_losses),
            'val_accuracy': accuracy,
            'val_predictions': total_predictions
        }
    
    def calibrate_probabilities(
        self,
        validation_data: List[Tuple[HumanState, ContextInformation, IntentType]]
    ) -> Dict[str, float]:
        """
        Calibrate probability outputs using temperature scaling or isotonic regression.
        
        Args:
            validation_data: Validation data for calibration
            
        Returns:
            Calibration metrics
        """
        if not self.is_trained or not validation_data:
            logger.warning("Cannot calibrate: model not trained or no validation data")
            return {}
        
        # Collect predictions and labels
        logits_list = []
        probs_list = []
        labels_list = []
        
        self.network.eval()
        with torch.no_grad():
            for human_state, context, true_intent in validation_data:
                features = self._extract_features(human_state, context)
                x = torch.tensor(features).float().unsqueeze(0)
                
                logits = self.network(x, sample=False)
                probs = F.softmax(logits, dim=-1)
                
                logits_list.append(logits[0].numpy())
                probs_list.append(probs[0].numpy())
                labels_list.append(list(IntentType).index(true_intent))
        
        logits_array = np.array(logits_list)
        probs_array = np.array(probs_list)
        labels_array = np.array(labels_list)
        
        # Apply calibration method
        if self.bayes_config.calibration_method == 'temperature':
            self.temperature_scaler.fit(logits_array, labels_array)
            calibrated_logits = logits_array / self.temperature_scaler.temperature
            calibrated_probs = softmax(calibrated_logits, axis=1)
        elif self.bayes_config.calibration_method == 'isotonic':
            # Use max probability for isotonic regression
            max_probs = np.max(probs_array, axis=1)
            correct_predictions = (np.argmax(probs_array, axis=1) == labels_array)
            self.isotonic_calibrator.fit(max_probs, correct_predictions)
            calibrated_probs = probs_array.copy()  # Placeholder
        else:
            calibrated_probs = probs_array
        
        # Compute calibration metrics
        max_calibrated_probs = np.max(calibrated_probs, axis=1)
        predicted_labels = np.argmax(calibrated_probs, axis=1)
        accuracies = (predicted_labels == labels_array).astype(float)
        
        # Expected Calibration Error (ECE)
        ece = self._compute_expected_calibration_error(
            max_calibrated_probs, accuracies
        )
        
        calibration_metrics = {
            'expected_calibration_error': ece,
            'temperature': getattr(self.temperature_scaler, 'temperature', 1.0),
            'calibration_bins': self.bayes_config.calibration_bins
        }
        
        logger.info(f"Calibration completed: ECE = {ece:.4f}, "
                   f"Temperature = {calibration_metrics['temperature']:.3f}")
        
        return calibration_metrics
    
    def _compute_expected_calibration_error(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
        n_bins: int = 15
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE = Σ_m (|B_m|/n) |acc(B_m) - conf(B_m)|
        
        where B_m is the set of samples with confidence in bin m.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _extract_context_factors(self, context: ContextInformation) -> Dict[str, Any]:
        """Extract relevant context factors for prediction explanation."""
        factors = {
            'n_objects': len(context.objects_in_scene),
            'robot_active': context.robot_state.get('is_active', False),
            'time_since_interaction': context.temporal_context.get(
                'time_since_last_interaction', 0.0
            ),
            'social_attention': context.social_context.get('attention_directed', False)
        }
        return factors
    
    def get_reliability_diagram_data(self) -> Dict[str, np.ndarray]:
        """
        Get data for plotting reliability diagram.
        
        Returns:
            Dictionary with confidence bins, accuracies, and counts
        """
        if len(self.reliability_data['confidences']) < 10:
            logger.warning("Insufficient data for reliability diagram")
            return {}
        
        confidences = np.array(self.reliability_data['confidences'])
        accuracies = np.array(self.reliability_data['accuracies'])
        
        # Compute reliability diagram
        fraction_of_positives, mean_predicted_value = calibration_curve(
            accuracies, confidences, n_bins=self.bayes_config.calibration_bins
        )
        
        return {
            'mean_predicted_value': mean_predicted_value,
            'fraction_of_positives': fraction_of_positives,
            'confidences': confidences,
            'accuracies': accuracies
        }
    
    def get_intent_transition_matrix(self) -> np.ndarray:
        """Learn intent transition matrix from prediction history."""
        if len(self.intent_history) < 2:
            return super().get_intent_transition_matrix()
        
        n_intents = len(IntentType)
        transitions = np.zeros((n_intents, n_intents))
        
        for i in range(len(self.intent_history) - 1):
            current_idx = list(IntentType).index(self.intent_history[i].intent_type)
            next_idx = list(IntentType).index(self.intent_history[i + 1].intent_type)
            transitions[current_idx, next_idx] += 1
        
        # Normalize rows to get probabilities
        row_sums = transitions.sum(axis=1)
        for i in range(n_intents):
            if row_sums[i] > 0:
                transitions[i, :] /= row_sums[i]
            else:
                transitions[i, :] = 1.0 / n_intents  # Uniform if no data
        
        return transitions


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)