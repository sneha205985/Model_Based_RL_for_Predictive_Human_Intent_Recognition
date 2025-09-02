"""
Gaussian Process-based Bayesian Q-Learning

This module implements GP-based Bayesian Q-learning for human-robot interaction,
providing uncertainty quantification over Q-functions and enabling principled exploration.

Mathematical Foundation:
- Q-function prior: Q(s,a) ~ GP(μ(s,a), k((s,a), (s',a')))
- Posterior: Q(s,a)|D ~ GP(μ_post(s,a), k_post((s,a), (s',a')))
- Bayesian update: p(Q|D) ∝ p(D|Q)p(Q)
- Thompson Sampling: π_t(a|s) ∝ P(a = argmax Q_t(s,a))

Author: Bayesian RL Implementation
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, AdditiveKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from abc import ABC, abstractmethod
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KernelType(Enum):
    """Types of GP kernels"""
    RBF = auto()
    MATERN_1_2 = auto()
    MATERN_3_2 = auto()
    MATERN_5_2 = auto()
    LINEAR_RBF = auto()
    ADDITIVE = auto()


@dataclass
class GPQConfiguration:
    """Configuration for GP-based Q-Learning"""
    # Kernel configuration
    kernel_type: KernelType = KernelType.RBF
    kernel_lengthscale: float = 1.0
    kernel_outputscale: float = 1.0
    
    # GP parameters
    likelihood_noise: float = 0.01
    mean_function: str = "constant"  # "constant" or "linear"
    
    # Training parameters
    training_iterations: int = 50
    learning_rate: float = 0.1
    
    # Memory and efficiency
    max_inducing_points: int = 500
    use_sparse_gp: bool = True
    
    # Exploration parameters
    thompson_samples: int = 1
    ucb_beta: float = 2.0
    
    # Q-learning parameters
    discount_factor: float = 0.99
    target_update_frequency: int = 10
    
    # Performance parameters
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


@dataclass
class Experience:
    """Single experience tuple"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    timestamp: float = field(default_factory=time.time)
    
    def to_tensor(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Convert to tensors"""
        return {
            'state': torch.tensor(self.state, device=device, dtype=torch.float32),
            'action': torch.tensor(self.action, device=device, dtype=torch.float32),
            'reward': torch.tensor(self.reward, device=device, dtype=torch.float32),
            'next_state': torch.tensor(self.next_state, device=device, dtype=torch.float32),
            'done': torch.tensor(self.done, device=device, dtype=torch.bool)
        }


class ExperienceBuffer:
    """Experience replay buffer for GP Q-learning"""
    
    def __init__(self, capacity: int = 10000):
        """Initialize buffer"""
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def add(self, experience: Experience):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences"""
        if len(self.buffer) < batch_size:
            return self.buffer[:]
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def get_all(self) -> List[Experience]:
        """Get all experiences"""
        return self.buffer[:]
    
    def __len__(self) -> int:
        return len(self.buffer)


class GPQFunction(ExactGP):
    """Gaussian Process Q-function model"""
    
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, 
                 likelihood: GaussianLikelihood, config: GPQConfiguration):
        """
        Initialize GP Q-function
        
        Args:
            train_x: Training inputs (state-action pairs)
            train_y: Training outputs (Q-values)
            likelihood: GP likelihood function
            config: GP configuration
        """
        super(GPQFunction, self).__init__(train_x, train_y, likelihood)
        
        self.config = config
        
        # Mean function
        if config.mean_function == "constant":
            self.mean_module = ConstantMean()
        elif config.mean_function == "linear":
            self.mean_module = LinearMean(train_x.size(-1))
        else:
            raise ValueError(f"Unknown mean function: {config.mean_function}")
        
        # Covariance function
        self.covar_module = self._create_kernel(config.kernel_type, train_x.size(-1))
        
    def _create_kernel(self, kernel_type: KernelType, input_dim: int):
        """Create GP kernel based on configuration"""
        if kernel_type == KernelType.RBF:
            base_kernel = RBFKernel(lengthscale=self.config.kernel_lengthscale)
            
        elif kernel_type == KernelType.MATERN_1_2:
            base_kernel = MaternKernel(nu=0.5, lengthscale=self.config.kernel_lengthscale)
            
        elif kernel_type == KernelType.MATERN_3_2:
            base_kernel = MaternKernel(nu=1.5, lengthscale=self.config.kernel_lengthscale)
            
        elif kernel_type == KernelType.MATERN_5_2:
            base_kernel = MaternKernel(nu=2.5, lengthscale=self.config.kernel_lengthscale)
            
        elif kernel_type == KernelType.LINEAR_RBF:
            from gpytorch.kernels import LinearKernel
            linear_kernel = LinearKernel()
            rbf_kernel = RBFKernel(lengthscale=self.config.kernel_lengthscale)
            base_kernel = linear_kernel + rbf_kernel
            
        elif kernel_type == KernelType.ADDITIVE:
            # Additive kernel for different input dimensions
            state_dim = input_dim - 6  # Assuming 6D action space
            state_kernel = RBFKernel(active_dims=torch.arange(state_dim))
            action_kernel = RBFKernel(active_dims=torch.arange(state_dim, input_dim))
            base_kernel = AdditiveKernel(state_kernel, action_kernel)
            
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        # Scale kernel
        return ScaleKernel(base_kernel, outputscale=self.config.kernel_outputscale)
    
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Forward pass through GP"""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class SparseGPQFunction(gpytorch.models.ApproximateGP):
    """Sparse Gaussian Process Q-function for scalability"""
    
    def __init__(self, inducing_points: torch.Tensor, config: GPQConfiguration):
        """Initialize sparse GP Q-function"""
        from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
        
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SparseGPQFunction, self).__init__(variational_strategy)
        
        self.config = config
        
        # Mean function
        if config.mean_function == "constant":
            self.mean_module = ConstantMean()
        elif config.mean_function == "linear":
            self.mean_module = LinearMean(inducing_points.size(-1))
        
        # Covariance function (same as exact GP)
        self.covar_module = self._create_kernel(config.kernel_type, inducing_points.size(-1))
    
    def _create_kernel(self, kernel_type: KernelType, input_dim: int):
        """Create GP kernel (same as exact GP)"""
        if kernel_type == KernelType.RBF:
            base_kernel = RBFKernel(lengthscale=self.config.kernel_lengthscale)
        # ... (same logic as GPQFunction)
        return ScaleKernel(base_kernel, outputscale=self.config.kernel_outputscale)
    
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Forward pass through sparse GP"""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GPBayesianQLearning:
    """
    Gaussian Process-based Bayesian Q-Learning
    
    Implements Bayesian Q-learning using Gaussian Processes to maintain
    uncertainty estimates over Q-functions and enable principled exploration.
    
    Mathematical Formulation:
    - Prior: Q(s,a) ~ GP(μ₀(s,a), k₀((s,a), (s',a')))
    - Likelihood: r + γ max_a' Q(s',a') | Q(s,a) ~ N(Q(s,a), σ²)
    - Posterior: Q(s,a)|D ~ GP(μ_post(s,a), k_post((s,a), (s',a')))
    
    Where:
    - μ_post(s,a) = μ₀(s,a) + k(s,a)ᵀ(K + σ²I)⁻¹(y - μ₀)
    - k_post = k₀ - k(s,a)ᵀ(K + σ²I)⁻¹k(s,a)
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: GPQConfiguration = None):
        """
        Initialize GP Bayesian Q-Learning
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space  
            config: GP configuration parameters
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or GPQConfiguration()
        
        # GP models
        self.gp_model: Optional[Union[GPQFunction, SparseGPQFunction]] = None
        self.likelihood = GaussianLikelihood(noise=torch.tensor(self.config.likelihood_noise))
        
        # Target network (for stable learning)
        self.target_gp_model: Optional[Union[GPQFunction, SparseGPQFunction]] = None
        self.target_likelihood = GaussianLikelihood(noise=torch.tensor(self.config.likelihood_noise))
        
        # Experience buffer
        self.experience_buffer = ExperienceBuffer()
        
        # Training state
        self.training_step = 0
        self.is_trained = False
        
        # Device and dtype
        self.device = torch.device(self.config.device)
        self.dtype = self.config.dtype
        
        # Performance tracking
        self.training_losses = []
        self.prediction_times = []
        
        logger.info(f"Initialized GP Bayesian Q-Learning with state_dim={state_dim}, action_dim={action_dim}")
    
    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float, 
                      next_state: np.ndarray, done: bool):
        """
        Add experience to buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = Experience(
            state=state.copy(),
            action=action.copy(), 
            reward=reward,
            next_state=next_state.copy(),
            done=done
        )
        self.experience_buffer.add(experience)
    
    def _create_state_action_input(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Combine states and actions into GP input"""
        return torch.cat([states, actions], dim=-1)
    
    def _initialize_gp_model(self, train_inputs: torch.Tensor, train_targets: torch.Tensor):
        """Initialize GP model with initial data"""
        if self.config.use_sparse_gp and len(train_inputs) > self.config.max_inducing_points:
            # Use sparse GP for large datasets
            inducing_indices = np.random.choice(
                len(train_inputs), self.config.max_inducing_points, replace=False
            )
            inducing_points = train_inputs[inducing_indices]
            
            self.gp_model = SparseGPQFunction(inducing_points, self.config)
            self.target_gp_model = SparseGPQFunction(inducing_points.clone(), self.config)
        else:
            # Use exact GP
            self.gp_model = GPQFunction(train_inputs, train_targets, self.likelihood, self.config)
            self.target_gp_model = GPQFunction(
                train_inputs.clone(), train_targets.clone(), self.target_likelihood, self.config
            )
        
        # Move to device
        self.gp_model = self.gp_model.to(self.device, self.dtype)
        self.target_gp_model = self.target_gp_model.to(self.device, self.dtype)
        self.likelihood = self.likelihood.to(self.device, self.dtype)
        self.target_likelihood = self.target_likelihood.to(self.device, self.dtype)
    
    def update_q_function(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Update Q-function using Bayesian learning
        
        Args:
            batch_size: Size of training batch
            
        Returns:
            Dictionary with training metrics
        """
        if len(self.experience_buffer) < batch_size:
            return {'loss': 0.0, 'num_samples': len(self.experience_buffer)}
        
        start_time = time.time()
        
        # Sample experiences
        experiences = self.experience_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.stack([torch.tensor(exp.state, dtype=self.dtype) for exp in experiences])
        actions = torch.stack([torch.tensor(exp.action, dtype=self.dtype) for exp in experiences])
        rewards = torch.stack([torch.tensor(exp.reward, dtype=self.dtype) for exp in experiences])
        next_states = torch.stack([torch.tensor(exp.next_state, dtype=self.dtype) for exp in experiences])
        dones = torch.stack([torch.tensor(exp.done, dtype=torch.bool) for exp in experiences])
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute target Q-values using target network
        with torch.no_grad():
            if self.target_gp_model is not None:
                next_q_values = self._compute_max_q_values(next_states)
                targets = rewards + self.config.discount_factor * next_q_values * (~dones)
            else:
                targets = rewards  # Bootstrap case
        
        # Create training inputs and targets
        train_inputs = self._create_state_action_input(states, actions)
        
        # Initialize or update GP model
        if self.gp_model is None:
            self._initialize_gp_model(train_inputs, targets)
        
        # Set models to training mode
        self.gp_model.train()
        self.likelihood.train()
        
        # Optimizer
        optimizer = torch.optim.Adam([
            {'params': self.gp_model.parameters()},
            {'params': self.likelihood.parameters()}
        ], lr=self.config.learning_rate)
        
        # Marginal log likelihood loss
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
        if isinstance(self.gp_model, SparseGPQFunction):
            mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.gp_model, num_data=len(train_inputs))
        
        # Training loop
        total_loss = 0.0
        for iteration in range(self.config.training_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            output = self.gp_model(train_inputs)
            loss = -mll(output, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Update target network periodically
        if self.training_step % self.config.target_update_frequency == 0:
            self._update_target_network()
        
        self.training_step += 1
        self.is_trained = True
        
        # Set to eval mode
        self.gp_model.eval()
        self.likelihood.eval()
        
        # Record metrics
        avg_loss = total_loss / self.config.training_iterations
        self.training_losses.append(avg_loss)
        
        training_time = time.time() - start_time
        
        return {
            'loss': avg_loss,
            'training_time': training_time,
            'num_samples': len(experiences),
            'training_step': self.training_step
        }
    
    def _compute_max_q_values(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute maximum Q-values over actions for given states
        
        Args:
            states: Batch of states
            
        Returns:
            Maximum Q-values
        """
        if self.target_gp_model is None:
            return torch.zeros(len(states), device=self.device, dtype=self.dtype)
        
        # Sample action candidates (could be improved with optimization)
        num_action_samples = 10
        action_samples = torch.rand(
            len(states), num_action_samples, self.action_dim,
            device=self.device, dtype=self.dtype
        ) * 2 - 1  # Sample in [-1, 1]
        
        # Evaluate Q-values for all state-action pairs
        max_q_values = torch.zeros(len(states), device=self.device, dtype=self.dtype)
        
        for i, state in enumerate(states):
            state_batch = state.unsqueeze(0).repeat(num_action_samples, 1)
            actions_batch = action_samples[i]
            
            state_action_inputs = self._create_state_action_input(state_batch, actions_batch)
            
            with torch.no_grad():
                self.target_gp_model.eval()
                predictions = self.target_gp_model(state_action_inputs)
                q_values = predictions.mean
                max_q_values[i] = torch.max(q_values)
        
        return max_q_values
    
    def _update_target_network(self):
        """Update target network with current network parameters"""
        if self.target_gp_model is not None and self.gp_model is not None:
            # Copy state dict
            self.target_gp_model.load_state_dict(self.gp_model.state_dict())
            self.target_likelihood.load_state_dict(self.likelihood.state_dict())
    
    def predict_q_value(self, state: np.ndarray, action: np.ndarray) -> Dict[str, float]:
        """
        Predict Q-value with uncertainty for state-action pair
        
        Args:
            state: State vector
            action: Action vector
            
        Returns:
            Dictionary with mean, variance, and confidence intervals
        """
        if self.gp_model is None or not self.is_trained:
            return {
                'mean': 0.0,
                'variance': 1.0,
                'std': 1.0,
                'lower_ci': -2.0,
                'upper_ci': 2.0,
                'epistemic_uncertainty': 1.0,
                'aleatoric_uncertainty': self.config.likelihood_noise
            }
        
        start_time = time.time()
        
        # Convert to tensors
        state_tensor = torch.tensor(state, device=self.device, dtype=self.dtype).unsqueeze(0)
        action_tensor = torch.tensor(action, device=self.device, dtype=self.dtype).unsqueeze(0)
        
        # Create input
        input_tensor = self._create_state_action_input(state_tensor, action_tensor)
        
        # Make prediction
        self.gp_model.eval()
        self.likelihood.eval()
        
        with torch.no_grad():
            # GP prediction
            f_pred = self.gp_model(input_tensor)
            
            # Likelihood prediction (includes observation noise)
            y_pred = self.likelihood(f_pred)
            
            mean = f_pred.mean.item()
            variance = f_pred.variance.item()
            std = f_pred.stddev.item()
            
            # Confidence intervals (2 standard deviations)
            lower_ci = mean - 2 * std
            upper_ci = mean + 2 * std
            
            # Decompose uncertainty
            epistemic_uncertainty = variance  # Model uncertainty
            aleatoric_uncertainty = self.likelihood.noise.item()  # Observation noise
        
        prediction_time = time.time() - start_time
        self.prediction_times.append(prediction_time)
        
        return {
            'mean': mean,
            'variance': variance,
            'std': std,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'prediction_time': prediction_time
        }
    
    def thompson_sampling_action(self, state: np.ndarray, action_candidates: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Select action using Thompson sampling
        
        Args:
            state: Current state
            action_candidates: Optional set of candidate actions
            
        Returns:
            Tuple of (selected_action, sampling_info)
        """
        if action_candidates is None:
            # Generate random action candidates
            num_candidates = 20
            action_candidates = np.random.uniform(
                -1, 1, (num_candidates, self.action_dim)
            ).astype(np.float32)
        
        if self.gp_model is None or not self.is_trained:
            # Random action if not trained
            selected_idx = np.random.choice(len(action_candidates))
            return action_candidates[selected_idx], {
                'sampling_method': 'random',
                'num_candidates': len(action_candidates),
                'selected_idx': selected_idx
            }
        
        # Sample from posterior for each action
        q_samples = []
        
        state_tensor = torch.tensor(state, device=self.device, dtype=self.dtype).unsqueeze(0)
        
        self.gp_model.eval()
        
        for action in action_candidates:
            action_tensor = torch.tensor(action, device=self.device, dtype=self.dtype).unsqueeze(0)
            input_tensor = self._create_state_action_input(state_tensor, action_tensor)
            
            with torch.no_grad():
                # Sample from GP posterior
                f_pred = self.gp_model(input_tensor)
                samples = f_pred.sample(torch.Size([self.config.thompson_samples]))
                q_samples.append(samples.mean().item())
        
        # Select action with highest sampled Q-value
        q_samples = np.array(q_samples)
        selected_idx = np.argmax(q_samples)
        selected_action = action_candidates[selected_idx]
        
        return selected_action, {
            'sampling_method': 'thompson',
            'num_candidates': len(action_candidates),
            'selected_idx': selected_idx,
            'q_samples': q_samples,
            'max_q_sample': q_samples[selected_idx]
        }
    
    def ucb_action(self, state: np.ndarray, action_candidates: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Select action using Upper Confidence Bound
        
        Args:
            state: Current state
            action_candidates: Optional set of candidate actions
            
        Returns:
            Tuple of (selected_action, ucb_info)
        """
        if action_candidates is None:
            # Generate random action candidates
            num_candidates = 20
            action_candidates = np.random.uniform(
                -1, 1, (num_candidates, self.action_dim)
            ).astype(np.float32)
        
        if self.gp_model is None or not self.is_trained:
            # Random action if not trained
            selected_idx = np.random.choice(len(action_candidates))
            return action_candidates[selected_idx], {
                'selection_method': 'random',
                'num_candidates': len(action_candidates),
                'selected_idx': selected_idx
            }
        
        # Compute UCB values for each action
        ucb_values = []
        means = []
        stds = []
        
        for action in action_candidates:
            q_pred = self.predict_q_value(state, action)
            mean = q_pred['mean']
            std = q_pred['std']
            
            ucb_value = mean + self.config.ucb_beta * std
            
            ucb_values.append(ucb_value)
            means.append(mean)
            stds.append(std)
        
        # Select action with highest UCB value
        ucb_values = np.array(ucb_values)
        selected_idx = np.argmax(ucb_values)
        selected_action = action_candidates[selected_idx]
        
        return selected_action, {
            'selection_method': 'ucb',
            'num_candidates': len(action_candidates),
            'selected_idx': selected_idx,
            'ucb_values': ucb_values,
            'means': np.array(means),
            'stds': np.array(stds),
            'beta': self.config.ucb_beta,
            'max_ucb': ucb_values[selected_idx]
        }
    
    def get_q_function_sample(self, num_points: int = 100) -> Dict[str, np.ndarray]:
        """
        Sample from posterior Q-function for visualization
        
        Args:
            num_points: Number of points to sample
            
        Returns:
            Dictionary with sample points and Q-values
        """
        if self.gp_model is None or not self.is_trained:
            return {
                'states': np.random.randn(num_points, self.state_dim),
                'actions': np.random.uniform(-1, 1, (num_points, self.action_dim)),
                'q_means': np.zeros(num_points),
                'q_stds': np.ones(num_points)
            }
        
        # Generate random state-action pairs
        states = np.random.randn(num_points, self.state_dim).astype(np.float32)
        actions = np.random.uniform(-1, 1, (num_points, self.action_dim)).astype(np.float32)
        
        # Predict Q-values
        q_means = []
        q_stds = []
        
        for i in range(num_points):
            q_pred = self.predict_q_value(states[i], actions[i])
            q_means.append(q_pred['mean'])
            q_stds.append(q_pred['std'])
        
        return {
            'states': states,
            'actions': actions,
            'q_means': np.array(q_means),
            'q_stds': np.array(q_stds)
        }
    
    def save_model(self, filepath: str):
        """Save GP model to file"""
        if self.gp_model is not None:
            torch.save({
                'gp_model_state_dict': self.gp_model.state_dict(),
                'likelihood_state_dict': self.likelihood.state_dict(),
                'config': self.config,
                'training_step': self.training_step,
                'is_trained': self.is_trained
            }, filepath)
            logger.info(f"GP Q-learning model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load GP model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.config = checkpoint['config']
        self.training_step = checkpoint['training_step']
        self.is_trained = checkpoint['is_trained']
        
        # Recreate model architecture (requires training data - would need to save that too)
        logger.warning("Loading GP model requires training data to recreate architecture")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'num_experiences': len(self.experience_buffer),
            'training_steps': self.training_step,
            'is_trained': self.is_trained,
            'avg_training_loss': np.mean(self.training_losses) if self.training_losses else 0.0,
            'avg_prediction_time': np.mean(self.prediction_times) if self.prediction_times else 0.0,
            'total_training_losses': len(self.training_losses),
            'total_predictions': len(self.prediction_times)
        }


# Example usage and testing
if __name__ == "__main__":
    # Test GP Bayesian Q-Learning
    logger.info("Testing GP Bayesian Q-Learning")
    
    # Configuration
    config = GPQConfiguration(
        kernel_type=KernelType.RBF,
        training_iterations=10,  # Reduced for testing
        max_inducing_points=50
    )
    
    # Initialize
    state_dim, action_dim = 10, 3
    gp_q_learner = GPBayesianQLearning(state_dim, action_dim, config)
    
    # Add some experiences
    for _ in range(20):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.uniform(-1, 1, action_dim).astype(np.float32)
        reward = np.random.normal(0, 1)
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = np.random.random() < 0.1
        
        gp_q_learner.add_experience(state, action, reward, next_state, done)
    
    # Train
    metrics = gp_q_learner.update_q_function(batch_size=10)
    logger.info(f"Training metrics: {metrics}")
    
    # Test prediction
    test_state = np.random.randn(state_dim).astype(np.float32)
    test_action = np.random.uniform(-1, 1, action_dim).astype(np.float32)
    
    q_pred = gp_q_learner.predict_q_value(test_state, test_action)
    logger.info(f"Q-value prediction: {q_pred}")
    
    # Test Thompson sampling
    selected_action, sampling_info = gp_q_learner.thompson_sampling_action(test_state)
    logger.info(f"Thompson sampling: action={selected_action}, info={sampling_info}")
    
    # Test UCB
    selected_action, ucb_info = gp_q_learner.ucb_action(test_state)
    logger.info(f"UCB selection: action={selected_action}, info={ucb_info}")
    
    # Performance metrics
    perf_metrics = gp_q_learner.get_performance_metrics()
    logger.info(f"Performance metrics: {perf_metrics}")
    
    print("GP Bayesian Q-Learning test completed successfully!")