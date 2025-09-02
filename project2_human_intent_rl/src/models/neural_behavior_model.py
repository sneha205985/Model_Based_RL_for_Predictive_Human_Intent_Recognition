"""
Neural Network-based Human Behavior Model implementation.

This module implements a comprehensive neural network approach to modeling
human behavior patterns using deep learning techniques with uncertainty
quantification for safe human-robot interaction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import logging
from dataclasses import dataclass, field
from collections import deque
import pickle
import json
from pathlib import Path

try:
    import numba
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .human_behavior import (
    HumanBehaviorModel, HumanState, BehaviorPrediction, BehaviorType
)


@dataclass
class NeuralModelConfig:
    """Configuration for Neural Behavior Model."""
    # Architecture parameters
    input_dim: int = 42  # Joint positions (3*7) + velocities (3*7) = 42
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    output_dim: int = 6  # Number of behavior types
    dropout_rate: float = 0.2
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    
    # Uncertainty quantification
    enable_bayesian: bool = True
    num_monte_carlo_samples: int = 50
    dropout_uncertainty: bool = True
    ensemble_size: int = 5
    
    # Sequence modeling
    sequence_length: int = 10
    use_lstm: bool = True
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    
    # Prediction parameters
    prediction_horizon: int = 20
    temporal_resolution: float = 0.1
    confidence_threshold: float = 0.7
    
    # Performance optimization
    use_gpu: bool = True
    compile_model: bool = True
    enable_mixed_precision: bool = True


class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight uncertainty."""
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.full((out_features, in_features), -3.0))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.full((out_features,), -3.0))
        
        # Prior
        self.prior_std = prior_std
        
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            # Sample weights from posterior
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)
            
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)
        else:
            # Use mean weights
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence with prior."""
        kl_weight = 0.5 * torch.sum(
            self.weight_logvar.exp() + self.weight_mu.pow(2) - self.weight_logvar - 1
        ) / (self.prior_std ** 2)
        
        kl_bias = 0.5 * torch.sum(
            self.bias_logvar.exp() + self.bias_mu.pow(2) - self.bias_logvar - 1
        ) / (self.prior_std ** 2)
        
        return kl_weight + kl_bias


class AttentionMechanism(nn.Module):
    """Multi-head attention for sequence processing."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)


class NeuralBehaviorNet(nn.Module):
    """Neural network for human behavior prediction."""
    
    def __init__(self, config: NeuralModelConfig):
        super().__init__()
        self.config = config
        
        # Input processing
        self.input_norm = nn.BatchNorm1d(config.input_dim)
        
        # Sequence modeling
        if config.use_lstm:
            self.lstm = nn.LSTM(
                config.input_dim,
                config.lstm_hidden_size,
                config.lstm_num_layers,
                batch_first=True,
                dropout=config.dropout_rate if config.lstm_num_layers > 1 else 0
            )
            feature_dim = config.lstm_hidden_size
        else:
            feature_dim = config.input_dim
            
        # Attention mechanism
        self.attention = AttentionMechanism(feature_dim)
        
        # Feature extraction layers
        layers = []
        prev_dim = feature_dim
        
        for hidden_dim in config.hidden_dims:
            if config.enable_bayesian:
                layers.append(BayesianLinear(prev_dim, hidden_dim))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_rate))
            prev_dim = hidden_dim
        
        self.feature_layers = nn.ModuleList(layers)
        
        # Output heads
        if config.enable_bayesian:
            self.behavior_head = BayesianLinear(prev_dim, config.output_dim)
            self.trajectory_head = BayesianLinear(prev_dim, config.prediction_horizon * 3)  # 3D trajectory
        else:
            self.behavior_head = nn.Linear(prev_dim, config.output_dim)
            self.trajectory_head = nn.Linear(prev_dim, config.prediction_horizon * 3)
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Linear(prev_dim, 1)
        
        # Intent probability head
        self.intent_head = nn.Linear(prev_dim, 10)  # 10 common intents
        
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        sample: bool = True
    ) -> Dict[str, torch.Tensor]:
        batch_size = x.size(0)
        
        # Input normalization
        if x.dim() == 3:  # Sequence input
            seq_len = x.size(1)
            x_reshaped = x.view(-1, x.size(-1))
            x_norm = self.input_norm(x_reshaped)
            x = x_norm.view(batch_size, seq_len, -1)
        else:
            x = self.input_norm(x)
        
        # Sequence processing
        if self.config.use_lstm and x.dim() == 3:
            lstm_out, (h_n, c_n) = self.lstm(x)
            # Use attention on LSTM output
            features = self.attention(lstm_out)
            # Use final timestep for predictions
            features = features[:, -1, :]
        else:
            features = x
        
        # Feature extraction
        for i, layer in enumerate(self.feature_layers):
            if isinstance(layer, BayesianLinear):
                features = layer(features, sample=sample)
            else:
                features = layer(features)
        
        # Predictions
        if self.config.enable_bayesian and isinstance(self.behavior_head, BayesianLinear):
            behavior_logits = self.behavior_head(features, sample=sample)
            trajectory_pred = self.trajectory_head(features, sample=sample)
        else:
            behavior_logits = self.behavior_head(features)
            trajectory_pred = self.trajectory_head(features)
        
        # Additional outputs
        uncertainty = torch.sigmoid(self.uncertainty_head(features))
        intent_logits = self.intent_head(features)
        
        outputs = {
            'behavior_logits': behavior_logits,
            'trajectory': trajectory_pred.view(batch_size, self.config.prediction_horizon, 3),
            'uncertainty': uncertainty,
            'intent_logits': intent_logits
        }
        
        if return_features:
            outputs['features'] = features
        
        return outputs
    
    def compute_kl_loss(self) -> torch.Tensor:
        """Compute KL divergence loss for Bayesian layers."""
        kl_loss = 0.0
        
        for layer in self.feature_layers:
            if isinstance(layer, BayesianLinear):
                kl_loss += layer.kl_divergence()
        
        if isinstance(self.behavior_head, BayesianLinear):
            kl_loss += self.behavior_head.kl_divergence()
        
        if isinstance(self.trajectory_head, BayesianLinear):
            kl_loss += self.trajectory_head.kl_divergence()
        
        return kl_loss


class NeuralHumanBehaviorModel(HumanBehaviorModel):
    """Neural network-based human behavior model implementation."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        # Extract neural model configuration
        neural_config_dict = config.get('neural_config', {})
        self.neural_config = NeuralModelConfig(**neural_config_dict)
        
        # Device configuration
        if self.neural_config.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.logger.info("Using GPU acceleration")
        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU")
        
        # Model ensemble for uncertainty quantification
        self.models: List[NeuralBehaviorNet] = []
        self.optimizers: List[optim.Optimizer] = []
        
        # Data management
        self.observation_buffer = deque(maxlen=10000)
        self.sequence_buffer = deque(maxlen=1000)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Performance tracking
        self.inference_times: List[float] = []
        self.prediction_cache: Dict[str, Tuple[List[BehaviorPrediction], float]] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_model(self) -> None:
        """Initialize the neural network models."""
        # Create model ensemble
        for i in range(self.neural_config.ensemble_size):
            model = NeuralBehaviorNet(self.neural_config).to(self.device)
            
            # Model compilation for optimization
            if self.neural_config.compile_model and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model)
                    self.logger.info(f"Model {i} compiled successfully")
                except Exception as e:
                    self.logger.warning(f"Model compilation failed: {e}")
            
            self.models.append(model)
            
            # Optimizer
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.neural_config.learning_rate,
                weight_decay=self.neural_config.weight_decay
            )
            self.optimizers.append(optimizer)
        
        self.logger.info(f"Initialized ensemble of {len(self.models)} neural models")
    
    def observe(self, human_state: HumanState) -> None:
        """Process a new observation of human state."""
        # Store observation
        self.observation_buffer.append(human_state)
        
        # Create sequence data if buffer is sufficient
        if len(self.observation_buffer) >= self.neural_config.sequence_length:
            sequence = list(self.observation_buffer)[-self.neural_config.sequence_length:]
            self.sequence_buffer.append(sequence)
        
        # Trigger online learning if enabled
        if hasattr(self.config, 'online_learning') and self.config['online_learning']:
            if len(self.sequence_buffer) >= self.neural_config.batch_size:
                self._online_update()
    
    def predict_behavior(
        self,
        current_state: HumanState,
        time_horizon: float,
        num_samples: int = 1
    ) -> List[BehaviorPrediction]:
        """Predict future human behavior based on current state."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained")
        
        start_time = time.time()
        
        # Create cache key
        state_key = self._create_state_key(current_state, time_horizon)
        
        # Check cache
        if state_key in self.prediction_cache:
            cached_predictions, cache_time = self.prediction_cache[state_key]
            if time.time() - cache_time < 1.0:  # 1 second cache
                return cached_predictions[:num_samples]
        
        # Prepare input sequence
        input_sequence = self._prepare_input_sequence(current_state)
        
        # Get ensemble predictions
        ensemble_predictions = []
        
        for model in self.models:
            model.eval()
            
            with torch.no_grad():
                # Multiple Monte Carlo samples for uncertainty
                mc_samples = []
                
                for _ in range(self.neural_config.num_monte_carlo_samples):
                    outputs = model(input_sequence, sample=True)
                    mc_samples.append(outputs)
                
                # Aggregate Monte Carlo samples
                aggregated_outputs = self._aggregate_mc_samples(mc_samples)
                ensemble_predictions.append(aggregated_outputs)
        
        # Ensemble aggregation
        final_predictions = self._aggregate_ensemble_predictions(ensemble_predictions)
        
        # Convert to BehaviorPrediction objects
        behavior_predictions = self._convert_to_behavior_predictions(
            final_predictions, current_state, time_horizon, num_samples
        )
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Cache results
        self.prediction_cache[state_key] = (behavior_predictions, time.time())
        
        # Limit cache size
        if len(self.prediction_cache) > 100:
            oldest_key = min(self.prediction_cache.keys(), 
                           key=lambda k: self.prediction_cache[k][1])
            del self.prediction_cache[oldest_key]
        
        return behavior_predictions[:num_samples]
    
    def update_model(
        self,
        observations: List[HumanState],
        ground_truth: Optional[List[BehaviorPrediction]] = None
    ) -> Dict[str, float]:
        """Update the model with new training data."""
        if not observations:
            return {'error': 1.0}
        
        # Prepare training data
        input_sequences, targets = self._prepare_training_data(observations, ground_truth)
        
        if input_sequences is None or len(input_sequences) == 0:
            return {'error': 1.0, 'message': 'No valid training sequences'}
        
        # Create data loader
        dataset = TensorDataset(input_sequences, targets['behavior'], targets['trajectory'])
        dataloader = DataLoader(
            dataset, 
            batch_size=self.neural_config.batch_size,
            shuffle=True
        )
        
        # Update each model in ensemble
        total_losses = []
        
        for model_idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            model.train()
            epoch_losses = []
            
            for batch_inputs, batch_behaviors, batch_trajectories in dataloader:
                batch_inputs = batch_inputs.to(self.device)
                batch_behaviors = batch_behaviors.to(self.device)
                batch_trajectories = batch_trajectories.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(batch_inputs)
                
                # Compute losses
                behavior_loss = F.cross_entropy(outputs['behavior_logits'], batch_behaviors)
                trajectory_loss = F.mse_loss(outputs['trajectory'], batch_trajectories)
                
                total_loss = behavior_loss + 0.1 * trajectory_loss
                
                # Add KL loss for Bayesian models
                if self.neural_config.enable_bayesian:
                    kl_loss = model.compute_kl_loss()
                    kl_weight = 1e-4  # Small weight for KL term
                    total_loss += kl_weight * kl_loss
                
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_losses.append(total_loss.item())
            
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                total_losses.append(avg_loss)
        
        metrics = {
            'loss': np.mean(total_losses) if total_losses else 1.0,
            'ensemble_std': np.std(total_losses) if len(total_losses) > 1 else 0.0,
            'models_updated': len(self.models)
        }
        
        return metrics
    
    def get_intent_probability(
        self,
        current_state: HumanState,
        intent: str
    ) -> float:
        """Get the probability of a specific intent given current state."""
        if not self.is_trained:
            return 0.5  # Uniform prior
        
        # Prepare input
        input_sequence = self._prepare_input_sequence(current_state)
        
        # Get predictions from ensemble
        intent_probs = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(input_sequence)
                intent_logits = outputs['intent_logits']
                intent_probs_model = F.softmax(intent_logits, dim=-1)
                intent_probs.append(intent_probs_model.cpu().numpy())
        
        # Average across ensemble
        avg_intent_probs = np.mean(intent_probs, axis=0)
        
        # Map intent string to index (simplified mapping)
        intent_map = {
            'reach_object': 0,
            'handover': 1,
            'pointing': 2,
            'gesture': 3,
            'idle': 4,
            'unknown': 5
        }
        
        intent_idx = intent_map.get(intent, -1)
        if intent_idx >= 0 and intent_idx < avg_intent_probs.shape[-1]:
            return float(avg_intent_probs[0, intent_idx])
        
        return 0.1  # Low probability for unknown intents
    
    def _train_epoch(
        self,
        training_data: List[Tuple[HumanState, BehaviorPrediction]],
        batch_size: int
    ) -> Dict[str, float]:
        """Train for one epoch and return metrics."""
        if not training_data:
            return {'loss': 1.0}
        
        # Extract states and behaviors
        states = [item[0] for item in training_data]
        behaviors = [item[1] for item in training_data]
        
        # Update model
        metrics = self.update_model(states, behaviors)
        
        self.current_epoch += 1
        return metrics
    
    def _validate_epoch(
        self,
        validation_data: List[Tuple[HumanState, BehaviorPrediction]]
    ) -> Dict[str, float]:
        """Validate model and return metrics."""
        if not validation_data:
            return {'val_loss': 1.0}
        
        # Prepare validation data
        states = [item[0] for item in validation_data]
        ground_truth = [item[1] for item in validation_data]
        
        input_sequences, targets = self._prepare_training_data(states, ground_truth)
        
        if input_sequences is None:
            return {'val_loss': 1.0}
        
        # Evaluate ensemble
        val_losses = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                input_sequences = input_sequences.to(self.device)
                targets['behavior'] = targets['behavior'].to(self.device)
                targets['trajectory'] = targets['trajectory'].to(self.device)
                
                outputs = model(input_sequences, sample=False)  # Use mean predictions
                
                behavior_loss = F.cross_entropy(outputs['behavior_logits'], targets['behavior'])
                trajectory_loss = F.mse_loss(outputs['trajectory'], targets['trajectory'])
                
                total_loss = behavior_loss + 0.1 * trajectory_loss
                val_losses.append(total_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        # Early stopping check
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return {
            'val_loss': avg_val_loss,
            'val_loss_std': np.std(val_losses),
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter
        }
    
    def _prepare_input_sequence(self, current_state: HumanState) -> torch.Tensor:
        """Prepare input sequence tensor from current state."""
        # Create feature vector from human state
        features = []
        
        # Position and orientation
        features.extend(current_state.position)
        features.extend(current_state.orientation)
        
        # Joint positions (if available)
        if current_state.joint_positions:
            for joint_name, joint_pos in current_state.joint_positions.items():
                features.extend(joint_pos)
        
        # Velocity
        features.extend(current_state.velocity)
        
        # Pad or truncate to expected input dimension
        while len(features) < self.neural_config.input_dim:
            features.append(0.0)
        features = features[:self.neural_config.input_dim]
        
        # Create sequence from recent observations
        if len(self.observation_buffer) >= self.neural_config.sequence_length:
            sequence_features = []
            recent_states = list(self.observation_buffer)[-self.neural_config.sequence_length:]
            
            for state in recent_states:
                state_features = self._extract_state_features(state)
                sequence_features.append(state_features)
            
            sequence_tensor = torch.FloatTensor(sequence_features).unsqueeze(0)  # Add batch dimension
        else:
            # Repeat current state if insufficient history
            features_tensor = torch.FloatTensor(features)
            sequence_tensor = features_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
            sequence_tensor = sequence_tensor.repeat(1, self.neural_config.sequence_length, 1)
        
        return sequence_tensor.to(self.device)
    
    def _extract_state_features(self, state: HumanState) -> List[float]:
        """Extract numerical features from human state."""
        features = []
        
        # Position and orientation
        features.extend(state.position.tolist())
        features.extend(state.orientation.tolist())
        
        # Joint positions
        if state.joint_positions:
            for joint_name in sorted(state.joint_positions.keys()):
                features.extend(state.joint_positions[joint_name].tolist())
        
        # Velocity
        features.extend(state.velocity.tolist())
        
        # Confidence
        features.append(state.confidence)
        
        # Pad to expected dimension
        while len(features) < self.neural_config.input_dim:
            features.append(0.0)
        
        return features[:self.neural_config.input_dim]
    
    def _prepare_training_data(
        self,
        observations: List[HumanState],
        ground_truth: Optional[List[BehaviorPrediction]] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """Prepare training data from observations and ground truth."""
        if len(observations) < self.neural_config.sequence_length:
            return None, None
        
        # Create sequences
        sequences = []
        behavior_targets = []
        trajectory_targets = []
        
        for i in range(len(observations) - self.neural_config.sequence_length + 1):
            # Extract sequence
            sequence = observations[i:i + self.neural_config.sequence_length]
            sequence_features = [self._extract_state_features(state) for state in sequence]
            sequences.append(sequence_features)
            
            # Extract targets
            if ground_truth and i < len(ground_truth):
                behavior = ground_truth[i]
                
                # Behavior type target
                behavior_idx = list(BehaviorType).index(behavior.behavior_type)
                behavior_targets.append(behavior_idx)
                
                # Trajectory target
                if behavior.predicted_trajectory.size > 0:
                    # Reshape to match expected output
                    traj = behavior.predicted_trajectory[:self.neural_config.prediction_horizon]
                    if traj.shape[0] < self.neural_config.prediction_horizon:
                        # Pad with last point
                        last_point = traj[-1] if len(traj) > 0 else np.zeros(3)
                        padding_needed = self.neural_config.prediction_horizon - traj.shape[0]
                        padding = np.tile(last_point, (padding_needed, 1))
                        traj = np.vstack([traj, padding])
                    
                    trajectory_targets.append(traj[:, :3])  # Use only x, y, z coordinates
                else:
                    # Default trajectory (stay at current position)
                    current_pos = sequence[-1].position[:3]
                    traj = np.tile(current_pos, (self.neural_config.prediction_horizon, 1))
                    trajectory_targets.append(traj)
            else:
                # Use heuristic targets if no ground truth
                behavior_targets.append(BehaviorType.UNKNOWN.value)
                current_pos = sequence[-1].position[:3]
                traj = np.tile(current_pos, (self.neural_config.prediction_horizon, 1))
                trajectory_targets.append(traj)
        
        if not sequences:
            return None, None
        
        # Convert to tensors
        input_sequences = torch.FloatTensor(sequences)
        behavior_targets = torch.LongTensor(behavior_targets)
        trajectory_targets = torch.FloatTensor(trajectory_targets)
        
        targets = {
            'behavior': behavior_targets,
            'trajectory': trajectory_targets
        }
        
        return input_sequences, targets
    
    def _aggregate_mc_samples(self, mc_samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate Monte Carlo samples."""
        aggregated = {}
        
        for key in mc_samples[0].keys():
            samples = [sample[key] for sample in mc_samples]
            stacked = torch.stack(samples, dim=0)
            
            if key in ['behavior_logits', 'intent_logits']:
                # For logits, take mean
                aggregated[key] = torch.mean(stacked, dim=0)
            elif key == 'uncertainty':
                # For uncertainty, take mean and add epistemic uncertainty
                mean_uncertainty = torch.mean(stacked, dim=0)
                epistemic_uncertainty = torch.var(stacked, dim=0)
                aggregated[key] = mean_uncertainty + epistemic_uncertainty
            else:
                # For other outputs, take mean
                aggregated[key] = torch.mean(stacked, dim=0)
        
        return aggregated
    
    def _aggregate_ensemble_predictions(
        self,
        ensemble_predictions: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate predictions from ensemble."""
        aggregated = {}
        
        for key in ensemble_predictions[0].keys():
            predictions = [pred[key] for pred in ensemble_predictions]
            stacked = torch.stack(predictions, dim=0)
            
            # Take mean across ensemble
            aggregated[key] = torch.mean(stacked, dim=0)
            
            # Add ensemble uncertainty for relevant outputs
            if key in ['behavior_logits', 'trajectory']:
                ensemble_var = torch.var(stacked, dim=0)
                aggregated[f'{key}_uncertainty'] = ensemble_var
        
        return aggregated
    
    def _convert_to_behavior_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        current_state: HumanState,
        time_horizon: float,
        num_samples: int
    ) -> List[BehaviorPrediction]:
        """Convert neural network outputs to BehaviorPrediction objects."""
        behavior_predictions = []
        
        # Get behavior probabilities
        behavior_logits = predictions['behavior_logits'].cpu().numpy()
        behavior_probs = torch.softmax(torch.from_numpy(behavior_logits), dim=-1).numpy()
        
        # Get trajectory prediction
        trajectory = predictions['trajectory'].cpu().numpy()
        
        # Get uncertainty
        uncertainty = predictions['uncertainty'].cpu().numpy()
        
        # Create predictions for top behaviors
        for i in range(min(num_samples, len(BehaviorType))):
            behavior_idx = np.argmax(behavior_probs[0])  # Assuming batch size 1
            behavior_prob = behavior_probs[0, behavior_idx]
            
            # Get corresponding behavior type
            behavior_type = list(BehaviorType)[behavior_idx]
            
            # Create prediction
            prediction = BehaviorPrediction(
                behavior_type=behavior_type,
                probability=float(behavior_prob),
                predicted_trajectory=trajectory[0],  # Remove batch dimension
                time_horizon=time_horizon,
                confidence=float(1.0 - uncertainty[0, 0]),
                metadata={
                    'neural_prediction': True,
                    'model_type': 'ensemble_neural',
                    'uncertainty': float(uncertainty[0, 0])
                }
            )
            
            behavior_predictions.append(prediction)
            
            # Set probability to 0 for next iteration
            behavior_probs[0, behavior_idx] = 0
        
        # Sort by probability
        behavior_predictions.sort(key=lambda x: x.probability, reverse=True)
        
        return behavior_predictions
    
    def _create_state_key(self, state: HumanState, time_horizon: float) -> str:
        """Create a cache key for the given state."""
        # Simple hash based on position, velocity, and time horizon
        pos_hash = hash(tuple(state.position))
        vel_hash = hash(tuple(state.velocity))
        return f"{pos_hash}_{vel_hash}_{time_horizon}"
    
    def _online_update(self) -> None:
        """Perform online model update with recent observations."""
        if len(self.sequence_buffer) < self.neural_config.batch_size:
            return
        
        # Get recent sequences
        recent_sequences = list(self.sequence_buffer)[-self.neural_config.batch_size:]
        
        # Convert to training format
        observations = []
        for sequence in recent_sequences:
            observations.extend(sequence)
        
        # Perform quick update
        self.update_model(observations)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get model performance metrics."""
        metrics = {}
        
        if self.inference_times:
            metrics.update({
                'avg_inference_time': np.mean(self.inference_times),
                'max_inference_time': np.max(self.inference_times),
                'inference_time_std': np.std(self.inference_times),
                'real_time_violations': np.sum(np.array(self.inference_times) > 0.01)  # 10ms threshold
            })
        
        metrics.update({
            'total_observations': len(self.observation_buffer),
            'total_sequences': len(self.sequence_buffer),
            'cache_size': len(self.prediction_cache),
            'current_epoch': self.current_epoch,
            'ensemble_size': len(self.models)
        })
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble models
        model_data = {
            'config': self.config,
            'neural_config': self.neural_config.__dict__,
            'models': [],
            'training_history': self.training_history,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss
        }
        
        # Save model state dicts
        for i, model in enumerate(self.models):
            model_data['models'].append(model.state_dict())
        
        torch.save(model_data, save_path)
        self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'NeuralHumanBehaviorModel':
        """Load a trained model from disk."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_data = torch.load(filepath, map_location=device)
        
        # Create model instance
        model = cls(model_data['config'])
        model.neural_config = NeuralModelConfig(**model_data['neural_config'])
        model.training_history = model_data['training_history']
        model.current_epoch = model_data['current_epoch']
        model.best_val_loss = model_data['best_val_loss']
        model.is_trained = True
        
        # Load model weights
        for i, state_dict in enumerate(model_data['models']):
            model.models[i].load_state_dict(state_dict)
        
        return model


if __name__ == "__main__":
    # Example usage
    config = {
        'neural_config': {
            'input_dim': 42,
            'hidden_dims': [256, 128, 64],
            'learning_rate': 1e-3,
            'batch_size': 32,
            'enable_bayesian': True,
            'ensemble_size': 3
        }
    }
    
    model = NeuralHumanBehaviorModel(config)
    print(f"Model initialized: {model.is_initialized}")
    print(f"Performance metrics: {model.get_performance_metrics()}")