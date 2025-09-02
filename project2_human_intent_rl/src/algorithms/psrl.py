"""
Posterior Sampling for Reinforcement Learning (PSRL)

This module implements PSRL, a model-based Bayesian RL algorithm that maintains
uncertainty over environment models and samples from the posterior to guide exploration.

Mathematical Foundation:
- Model posterior: P(M|D) ∝ P(D|M)P(M)
- Transition posterior: P(T|D) ∝ P(D|T)P(T)  
- Reward posterior: P(R|D) ∝ P(D|R)P(R)
- Policy: π* = argmax_π V^π(s) for sampled M ~ P(M|D)

Algorithm:
1. Maintain posterior beliefs over environment models
2. At each episode: sample model M ~ P(M|D)
3. Compute optimal policy π* for sampled model M
4. Execute π* and collect data D_new
5. Update posterior: P(M|D ∪ D_new)

Author: Bayesian RL Implementation
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal, Categorical, Dirichlet
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from abc import ABC, abstractmethod
import time
from scipy import linalg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of environment models"""
    GAUSSIAN_LINEAR = auto()    # Linear-Gaussian dynamics
    NEURAL_NETWORK = auto()     # Neural network dynamics
    GAUSSIAN_PROCESS = auto()   # GP dynamics (imported from gp_q_learning)
    TABULAR = auto()           # Tabular MDP (discrete spaces)


@dataclass
class PSRLConfiguration:
    """Configuration for PSRL"""
    # Model configuration
    model_type: ModelType = ModelType.GAUSSIAN_LINEAR
    
    # Linear-Gaussian model parameters
    state_dim: int = 10
    action_dim: int = 3
    
    # Prior parameters for linear dynamics: s' = A*s + B*a + w, w ~ N(0, Σ)
    transition_prior_precision: float = 1.0  # Precision of transition matrix prior
    noise_prior_alpha: float = 1.0          # Inverse-Wishart prior for noise covariance
    noise_prior_beta: float = 1.0           # Scale parameter
    
    # Reward model parameters
    reward_prior_mean: float = 0.0
    reward_prior_precision: float = 1.0
    reward_noise_alpha: float = 1.0
    reward_noise_beta: float = 1.0
    
    # Neural network model parameters
    hidden_layers: List[int] = field(default_factory=lambda: [128, 128])
    activation: str = "relu"
    dropout_rate: float = 0.1
    
    # PSRL algorithm parameters
    planning_horizon: int = 10
    planning_iterations: int = 50
    num_model_samples: int = 1
    
    # Optimization parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    
    # Performance parameters
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


@dataclass
class Transition:
    """Single transition tuple"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    timestamp: float = field(default_factory=time.time)


class BayesianLinearModel:
    """
    Bayesian linear model for dynamics: s' = A*s + B*a + w
    
    Maintains posterior distributions over parameters A, B and noise covariance Σ.
    Uses conjugate priors for analytical updates.
    
    Prior:
    - vec(A,B) ~ N(μ₀, Λ₀⁻¹)
    - Σ ~ IW(α₀, Β₀)
    
    Posterior (after n observations):
    - vec(A,B) ~ N(μₙ, Λₙ⁻¹)  
    - Σ ~ IW(αₙ, Βₙ)
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: PSRLConfiguration):
        """
        Initialize Bayesian linear model
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: PSRL configuration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Input dimension (state + action)
        self.input_dim = state_dim + action_dim
        
        # Prior parameters for transition parameters [A, B]
        # Vectorized form: θ = vec([A, B]) ∈ R^(state_dim * input_dim)
        self.prior_mean = torch.zeros(state_dim * self.input_dim, dtype=config.dtype)
        self.prior_precision = config.transition_prior_precision * torch.eye(
            state_dim * self.input_dim, dtype=config.dtype
        )
        
        # Prior for noise covariance (Inverse-Wishart)
        self.prior_alpha = config.noise_prior_alpha + state_dim - 1  # Degrees of freedom
        self.prior_beta = config.noise_prior_beta * torch.eye(state_dim, dtype=config.dtype)
        
        # Posterior parameters (initialized to prior)
        self.posterior_mean = self.prior_mean.clone()
        self.posterior_precision = self.prior_precision.clone()
        self.posterior_alpha = self.prior_alpha
        self.posterior_beta = self.prior_beta.clone()
        
        # Sufficient statistics
        self.XX = torch.zeros(self.input_dim, self.input_dim, dtype=config.dtype)  # ∑xᵢxᵢᵀ
        self.XY = torch.zeros(self.input_dim, state_dim, dtype=config.dtype)       # ∑xᵢyᵢᵀ
        self.YY = torch.zeros(state_dim, state_dim, dtype=config.dtype)           # ∑yᵢyᵢᵀ
        self.n_samples = 0
        
        logger.info(f"Initialized Bayesian linear model: state_dim={state_dim}, action_dim={action_dim}")
    
    def update(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor):
        """
        Update posterior beliefs with new data
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
            next_states: Batch of next states [batch_size, state_dim]
        """
        batch_size = states.shape[0]
        
        # Create input matrix X = [states, actions]
        X = torch.cat([states, actions], dim=-1)  # [batch_size, input_dim]
        Y = next_states  # [batch_size, state_dim]
        
        # Update sufficient statistics
        self.XX += X.T @ X
        self.XY += X.T @ Y
        self.YY += Y.T @ Y
        self.n_samples += batch_size
        
        # Update posterior for transition parameters θ
        # Posterior precision: Λₙ = Λ₀ + ∑XᵢXᵢᵀ ⊗ I
        XX_kron = torch.kron(self.XX, torch.eye(self.state_dim, dtype=self.config.dtype))
        self.posterior_precision = self.prior_precision + XX_kron
        
        # Posterior mean: μₙ = Λₙ⁻¹(Λ₀μ₀ + vec(∑XᵢYᵢᵀ))
        XY_vec = self.XY.T.contiguous().view(-1)  # vec(XY^T)
        precision_prior_mean = self.prior_precision @ self.prior_mean
        self.posterior_mean = torch.linalg.solve(
            self.posterior_precision, precision_prior_mean + XY_vec
        )
        
        # Update posterior for noise covariance Σ
        self.posterior_alpha = self.prior_alpha + self.n_samples
        
        # Compute residual sum of squares for noise covariance update
        # This requires the current estimate of parameters
        theta_map = self.posterior_mean  # MAP estimate
        AB_map = theta_map.view(self.state_dim, self.input_dim)  # Reshape to [state_dim, input_dim]
        
        # Predicted next states: Ŷ = X @ AB_map^T
        Y_pred = X @ AB_map.T
        residuals = Y - Y_pred
        
        # Update noise covariance posterior scale matrix
        residual_sum_squares = residuals.T @ residuals
        self.posterior_beta = self.prior_beta + residual_sum_squares
        
        # Add uncertainty from parameter estimation
        posterior_cov = torch.linalg.inv(self.posterior_precision)
        param_uncertainty = torch.zeros(self.state_dim, self.state_dim, dtype=self.config.dtype)
        
        # Approximate parameter uncertainty contribution (simplified)
        for i in range(self.state_dim):
            start_idx = i * self.input_dim
            end_idx = (i + 1) * self.input_dim
            param_cov_i = posterior_cov[start_idx:end_idx, start_idx:end_idx]
            param_uncertainty[i, i] += torch.trace(self.XX @ param_cov_i)
        
        self.posterior_beta += param_uncertainty
    
    def sample_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample transition model from posterior
        
        Returns:
            Tuple of (transition_matrix, noise_covariance)
            - transition_matrix: [state_dim, input_dim] matrix [A, B]
            - noise_covariance: [state_dim, state_dim] covariance matrix
        """
        # Sample noise covariance from Inverse-Wishart
        # Σ ~ IW(αₙ, Βₙ)
        try:
            # Sample from Wishart and invert
            wishart_sample = torch.from_numpy(
                linalg.inv(np.random.wishart(
                    int(self.posterior_alpha), 
                    self.posterior_beta.numpy()
                ))
            ).to(dtype=self.config.dtype)
        except np.linalg.LinAlgError:
            # Fallback to prior if sampling fails
            logger.warning("Wishart sampling failed, using prior")
            wishart_sample = torch.linalg.inv(self.posterior_beta) / self.posterior_alpha
        
        noise_covariance = wishart_sample
        
        # Sample transition parameters from multivariate normal
        # θ ~ N(μₙ, Λₙ⁻¹ ⊗ Σ)
        try:
            posterior_cov = torch.linalg.inv(self.posterior_precision)
            # Use Cholesky for numerical stability
            L = torch.linalg.cholesky(posterior_cov)
            noise_sample = torch.randn(self.state_dim * self.input_dim, dtype=self.config.dtype)
            theta_sample = self.posterior_mean + L @ noise_sample
        except torch.linalg.LinAlgError:
            # Fallback to MAP estimate
            logger.warning("Parameter sampling failed, using MAP estimate")
            theta_sample = self.posterior_mean
        
        # Reshape to transition matrix
        transition_matrix = theta_sample.view(self.state_dim, self.input_dim)
        
        return transition_matrix, noise_covariance
    
    def predict(self, states: torch.Tensor, actions: torch.Tensor, 
                sample_model: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next states (with uncertainty)
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Actions [batch_size, action_dim]
            sample_model: Whether to sample from posterior or use MAP
            
        Returns:
            Tuple of (predicted_next_states, prediction_covariance)
        """
        X = torch.cat([states, actions], dim=-1)
        
        if sample_model:
            # Sample model parameters
            AB, Sigma = self.sample_model()
            next_states_mean = X @ AB.T
            
            # Add noise
            noise_samples = MultivariateNormal(
                torch.zeros(self.state_dim, dtype=self.config.dtype), Sigma
            ).sample((X.shape[0],))
            
            next_states = next_states_mean + noise_samples
            prediction_cov = Sigma.unsqueeze(0).repeat(X.shape[0], 1, 1)
            
        else:
            # Use MAP estimate
            theta_map = self.posterior_mean
            AB_map = theta_map.view(self.state_dim, self.input_dim)
            next_states_mean = X @ AB_map.T
            
            # Compute predictive uncertainty
            posterior_cov = torch.linalg.inv(self.posterior_precision)
            noise_cov = self.posterior_beta / (self.posterior_alpha - self.state_dim - 1)
            
            # Predictive covariance includes parameter uncertainty
            prediction_cov = torch.zeros(X.shape[0], self.state_dim, self.state_dim, dtype=self.config.dtype)
            for i, x in enumerate(X):
                # Parameter uncertainty contribution
                x_kron = torch.kron(x, torch.eye(self.state_dim, dtype=self.config.dtype))
                param_uncertainty = x_kron.T @ posterior_cov @ x_kron
                prediction_cov[i] = noise_cov + param_uncertainty
            
            next_states = next_states_mean
        
        return next_states, prediction_cov


class BayesianRewardModel:
    """
    Bayesian linear reward model: r = w^T * [s, a] + ε
    
    Maintains posterior over reward function parameters w and noise variance σ².
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: PSRLConfiguration):
        """Initialize Bayesian reward model"""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.input_dim = state_dim + action_dim
        
        # Prior for reward parameters w ~ N(μ₀, Σ₀)
        self.prior_mean = torch.ones(self.input_dim, dtype=config.dtype) * config.reward_prior_mean
        self.prior_precision = config.reward_prior_precision * torch.eye(self.input_dim, dtype=config.dtype)
        
        # Prior for noise variance σ² ~ IG(α₀, β₀)
        self.prior_alpha = config.reward_noise_alpha
        self.prior_beta = config.reward_noise_beta
        
        # Posterior parameters
        self.posterior_mean = self.prior_mean.clone()
        self.posterior_precision = self.prior_precision.clone()
        self.posterior_alpha = self.prior_alpha
        self.posterior_beta = self.prior_beta
        
        # Sufficient statistics
        self.n_samples = 0
    
    def update(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor):
        """Update posterior with new reward observations"""
        X = torch.cat([states, actions], dim=-1)  # [batch_size, input_dim]
        y = rewards  # [batch_size]
        
        batch_size = X.shape[0]
        
        # Update posterior for reward parameters
        XTX = X.T @ X
        XTy = X.T @ y
        
        self.posterior_precision = self.prior_precision + XTX
        precision_prior_mean = self.prior_precision @ self.prior_mean
        self.posterior_mean = torch.linalg.solve(self.posterior_precision, precision_prior_mean + XTy)
        
        # Update noise variance posterior
        self.n_samples += batch_size
        self.posterior_alpha = self.prior_alpha + batch_size / 2
        
        # Compute residual sum of squares
        y_pred = X @ self.posterior_mean
        residuals = y - y_pred
        residual_ss = torch.sum(residuals**2)
        
        # Add parameter uncertainty
        posterior_cov = torch.linalg.inv(self.posterior_precision)
        param_uncertainty = torch.sum((X @ posterior_cov) * X, dim=1).sum()
        
        self.posterior_beta = self.prior_beta + 0.5 * (residual_ss + param_uncertainty)
    
    def sample_model(self) -> Tuple[torch.Tensor, float]:
        """Sample reward model parameters"""
        # Sample noise variance from Inverse-Gamma
        sigma_squared = self.posterior_beta / torch.distributions.Gamma(self.posterior_alpha, 1.0).sample()
        
        # Sample reward parameters from multivariate normal
        posterior_cov = torch.linalg.inv(self.posterior_precision) * sigma_squared
        try:
            w_sample = MultivariateNormal(self.posterior_mean, posterior_cov).sample()
        except RuntimeError:
            # Fallback to MAP estimate
            w_sample = self.posterior_mean
        
        return w_sample, sigma_squared.item()
    
    def predict(self, states: torch.Tensor, actions: torch.Tensor, 
               sample_model: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict rewards"""
        X = torch.cat([states, actions], dim=-1)
        
        if sample_model:
            w, sigma_squared = self.sample_model()
            rewards_mean = X @ w
            noise = torch.randn(X.shape[0], dtype=self.config.dtype) * np.sqrt(sigma_squared)
            rewards = rewards_mean + noise
            rewards_var = torch.full_like(rewards, sigma_squared)
        else:
            # MAP estimate
            rewards = X @ self.posterior_mean
            
            # Predictive variance
            posterior_cov = torch.linalg.inv(self.posterior_precision)
            noise_var = self.posterior_beta / (self.posterior_alpha - 1)
            rewards_var = torch.sum((X @ posterior_cov) * X, dim=1) + noise_var
        
        return rewards, rewards_var


class PSRLAgent:
    """
    Posterior Sampling for Reinforcement Learning (PSRL) Agent
    
    Implements Thompson sampling for Bayesian RL by maintaining posterior
    distributions over environment models and sampling from them to guide exploration.
    
    Algorithm:
    1. Maintain posterior beliefs P(M|D) over environment models
    2. At each episode: sample model M ~ P(M|D)  
    3. Solve for optimal policy π* in sampled model M
    4. Execute π* for one episode
    5. Update posterior with collected data
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: PSRLConfiguration = None):
        """
        Initialize PSRL agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: PSRL configuration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or PSRLConfiguration(state_dim=state_dim, action_dim=action_dim)
        
        # Initialize model components
        self.transition_model = BayesianLinearModel(state_dim, action_dim, self.config)
        self.reward_model = BayesianRewardModel(state_dim, action_dim, self.config)
        
        # Experience storage
        self.transitions = []
        
        # Current sampled model (for episode-wise planning)
        self.sampled_transition_matrix = None
        self.sampled_noise_covariance = None
        self.sampled_reward_weights = None
        self.sampled_reward_noise = None
        
        # Policy storage
        self.current_policy = None
        self.value_function = None
        
        # Performance tracking
        self.episode_count = 0
        self.planning_times = []
        self.model_sampling_times = []
        
        # Device
        self.device = torch.device(self.config.device)
        
        logger.info(f"Initialized PSRL agent: state_dim={state_dim}, action_dim={action_dim}")
    
    def add_transition(self, state: np.ndarray, action: np.ndarray, reward: float, 
                      next_state: np.ndarray, done: bool):
        """Add transition to experience buffer"""
        transition = Transition(
            state=state.copy(),
            action=action.copy(),
            reward=reward,
            next_state=next_state.copy(), 
            done=done
        )
        self.transitions.append(transition)
    
    def update_models(self):
        """Update posterior beliefs over models with all collected data"""
        if len(self.transitions) == 0:
            return
        
        # Convert transitions to tensors
        states = torch.stack([torch.tensor(t.state, dtype=self.config.dtype) for t in self.transitions])
        actions = torch.stack([torch.tensor(t.action, dtype=self.config.dtype) for t in self.transitions])
        rewards = torch.stack([torch.tensor(t.reward, dtype=self.config.dtype) for t in self.transitions])
        next_states = torch.stack([torch.tensor(t.next_state, dtype=self.config.dtype) for t in self.transitions])
        
        # Update transition model
        self.transition_model.update(states, actions, next_states)
        
        # Update reward model
        self.reward_model.update(states, actions, rewards)
        
        logger.info(f"Updated models with {len(self.transitions)} transitions")
    
    def sample_models(self):
        """Sample models from posterior for current episode"""
        start_time = time.time()
        
        # Sample transition model
        self.sampled_transition_matrix, self.sampled_noise_covariance = self.transition_model.sample_model()
        
        # Sample reward model  
        self.sampled_reward_weights, self.sampled_reward_noise = self.reward_model.sample_model()
        
        sampling_time = time.time() - start_time
        self.model_sampling_times.append(sampling_time)
    
    def plan_policy(self, initial_state: np.ndarray) -> Dict[str, Any]:
        """
        Plan optimal policy for sampled model using value iteration
        
        Args:
            initial_state: Starting state for planning
            
        Returns:
            Dictionary with planning results
        """
        start_time = time.time()
        
        if self.sampled_transition_matrix is None:
            self.sample_models()
        
        # Convert to tensors
        state_tensor = torch.tensor(initial_state, dtype=self.config.dtype)
        
        # Initialize value function
        # For continuous spaces, we'll use a simple discretization for planning
        # In practice, you might want to use function approximation
        
        # Simple planning: compute Q-values for action candidates around current state
        num_action_candidates = 20
        action_candidates = torch.rand(num_action_candidates, self.action_dim, dtype=self.config.dtype) * 2 - 1
        
        # Compute Q-values using sampled model
        q_values = torch.zeros(num_action_candidates, dtype=self.config.dtype)
        
        for i, action in enumerate(action_candidates):
            # Simulate trajectory with sampled model
            q_value = self._simulate_trajectory(state_tensor, action)
            q_values[i] = q_value
        
        # Create simple policy: choose action with highest Q-value
        best_action_idx = torch.argmax(q_values)
        self.current_policy = {
            'action_candidates': action_candidates,
            'q_values': q_values,
            'best_action': action_candidates[best_action_idx],
            'best_q_value': q_values[best_action_idx]
        }
        
        planning_time = time.time() - start_time
        self.planning_times.append(planning_time)
        
        return {
            'planning_time': planning_time,
            'num_action_candidates': num_action_candidates,
            'max_q_value': q_values[best_action_idx].item(),
            'q_value_std': q_values.std().item()
        }
    
    def _simulate_trajectory(self, initial_state: torch.Tensor, action: torch.Tensor, 
                           horizon: int = None) -> torch.Tensor:
        """
        Simulate trajectory with sampled model to compute Q-value
        
        Args:
            initial_state: Starting state
            action: Action to evaluate
            horizon: Planning horizon (uses config default if None)
            
        Returns:
            Estimated Q-value
        """
        if horizon is None:
            horizon = self.config.planning_horizon
        
        current_state = initial_state.clone()
        cumulative_reward = 0.0
        discount = 1.0
        
        for step in range(horizon):
            # Predict next state using sampled transition model
            state_action = torch.cat([current_state, action if step == 0 else torch.zeros_like(action)])
            next_state_mean = state_action @ self.sampled_transition_matrix.T
            
            # Add noise (optional - can be deterministic for planning)
            noise = MultivariateNormal(
                torch.zeros(self.state_dim, dtype=self.config.dtype),
                self.sampled_noise_covariance
            ).sample()
            next_state = next_state_mean + noise
            
            # Predict reward using sampled reward model
            reward_input = torch.cat([current_state, action if step == 0 else torch.zeros_like(action)])
            reward = reward_input @ self.sampled_reward_weights
            
            # Add reward noise
            reward += torch.randn(1, dtype=self.config.dtype) * np.sqrt(self.sampled_reward_noise)
            
            # Accumulate discounted reward
            cumulative_reward += discount * reward
            discount *= 0.99  # Discount factor
            
            # Update state (for multi-step planning, would need better action selection)
            current_state = next_state
        
        return cumulative_reward
    
    def select_action(self, state: np.ndarray, exploration_strategy: str = "psrl") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select action using PSRL (or fallback strategy)
        
        Args:
            state: Current state
            exploration_strategy: Strategy to use ("psrl", "random", "greedy")
            
        Returns:
            Tuple of (selected_action, selection_info)
        """
        if exploration_strategy == "psrl" and self.current_policy is not None:
            # Use planned policy from sampled model
            action = self.current_policy['best_action'].numpy()
            
            selection_info = {
                'strategy': 'psrl',
                'q_value': self.current_policy['best_q_value'].item(),
                'episode_count': self.episode_count,
                'model_sampled': True
            }
            
        elif exploration_strategy == "random" or self.current_policy is None:
            # Random exploration
            action = np.random.uniform(-1, 1, self.action_dim).astype(np.float32)
            
            selection_info = {
                'strategy': 'random',
                'q_value': 0.0,
                'episode_count': self.episode_count,
                'model_sampled': False
            }
            
        else:
            # Greedy with respect to posterior mean (not implemented)
            action = np.zeros(self.action_dim, dtype=np.float32)
            selection_info = {'strategy': 'greedy'}
        
        return action, selection_info
    
    def begin_episode(self, initial_state: np.ndarray):
        """Begin new episode: update models, sample new models, plan policy"""
        # Update posterior beliefs
        if len(self.transitions) > 0:
            self.update_models()
        
        # Sample new models for this episode
        self.sample_models()
        
        # Plan policy for sampled models
        planning_info = self.plan_policy(initial_state)
        
        self.episode_count += 1
        
        return planning_info
    
    def get_model_predictions(self, states: np.ndarray, actions: np.ndarray) -> Dict[str, Any]:
        """
        Get model predictions with uncertainty
        
        Args:
            states: Array of states [num_points, state_dim]
            actions: Array of actions [num_points, action_dim]
            
        Returns:
            Dictionary with predictions and uncertainties
        """
        states_tensor = torch.tensor(states, dtype=self.config.dtype)
        actions_tensor = torch.tensor(actions, dtype=self.config.dtype)
        
        # Transition predictions
        next_states, transition_cov = self.transition_model.predict(states_tensor, actions_tensor, sample_model=False)
        
        # Reward predictions
        rewards, reward_var = self.reward_model.predict(states_tensor, actions_tensor, sample_model=False)
        
        return {
            'predicted_next_states': next_states.numpy(),
            'transition_uncertainty': transition_cov.numpy(),
            'predicted_rewards': rewards.numpy(),
            'reward_uncertainty': reward_var.numpy(),
            'num_transitions': len(self.transitions)
        }
    
    def get_posterior_samples(self, num_samples: int = 10) -> Dict[str, List[Any]]:
        """Get samples from model posteriors for analysis"""
        transition_samples = []
        reward_samples = []
        
        for _ in range(num_samples):
            # Sample transition model
            trans_matrix, noise_cov = self.transition_model.sample_model()
            transition_samples.append({
                'transition_matrix': trans_matrix.numpy(),
                'noise_covariance': noise_cov.numpy()
            })
            
            # Sample reward model
            reward_weights, reward_noise = self.reward_model.sample_model()
            reward_samples.append({
                'reward_weights': reward_weights.numpy(),
                'reward_noise': reward_noise
            })
        
        return {
            'transition_samples': transition_samples,
            'reward_samples': reward_samples,
            'num_samples': num_samples
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'episode_count': self.episode_count,
            'num_transitions': len(self.transitions),
            'avg_planning_time': np.mean(self.planning_times) if self.planning_times else 0.0,
            'avg_model_sampling_time': np.mean(self.model_sampling_times) if self.model_sampling_times else 0.0,
            'total_planning_time': sum(self.planning_times),
            'has_current_policy': self.current_policy is not None,
            'transition_posterior_samples': self.transition_model.n_samples,
            'reward_posterior_samples': self.reward_model.n_samples
        }


# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing PSRL implementation")
    
    # Configuration
    config = PSRLConfiguration(
        state_dim=6,
        action_dim=2,
        planning_horizon=5,
        planning_iterations=10
    )
    
    # Initialize PSRL agent
    agent = PSRLAgent(config.state_dim, config.action_dim, config)
    
    # Simulate some experience
    for episode in range(3):
        # Begin episode
        initial_state = np.random.randn(config.state_dim).astype(np.float32)
        planning_info = agent.begin_episode(initial_state)
        logger.info(f"Episode {episode + 1} planning: {planning_info}")
        
        # Simulate episode
        current_state = initial_state
        for step in range(10):
            # Select action
            action, selection_info = agent.select_action(current_state, "psrl")
            
            # Simulate environment (simple linear dynamics for testing)
            noise = np.random.normal(0, 0.1, config.state_dim)
            next_state = 0.9 * current_state + 0.1 * np.concatenate([action, np.zeros(config.state_dim - config.action_dim)]) + noise
            reward = -np.sum(current_state**2) + np.random.normal(0, 0.1)  # Quadratic cost
            done = step == 9
            
            # Add to experience
            agent.add_transition(current_state, action, reward, next_state, done)
            
            current_state = next_state
            
            if done:
                break
    
    # Test model predictions
    test_states = np.random.randn(5, config.state_dim).astype(np.float32)
    test_actions = np.random.uniform(-1, 1, (5, config.action_dim)).astype(np.float32)
    
    predictions = agent.get_model_predictions(test_states, test_actions)
    logger.info(f"Model predictions shape: {predictions['predicted_next_states'].shape}")
    
    # Test posterior samples
    posterior_samples = agent.get_posterior_samples(num_samples=3)
    logger.info(f"Number of posterior samples: {len(posterior_samples['transition_samples'])}")
    
    # Performance metrics
    metrics = agent.get_performance_metrics()
    logger.info(f"Performance metrics: {metrics}")
    
    print("PSRL test completed successfully!")