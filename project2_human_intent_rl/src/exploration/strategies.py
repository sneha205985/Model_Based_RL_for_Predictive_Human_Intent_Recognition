"""
Bayesian RL Exploration Strategies

This module implements advanced exploration strategies for Bayesian RL,
including Thompson sampling, UCB variants, information gain maximization,
and safe exploration methods.

Mathematical Foundation:
- Thompson Sampling: π(a|s) = P(a = argmax_a Q(s,a)) under posterior
- UCB: a* = argmax_a [Q̄(s,a) + β√Var(Q(s,a))]
- Information Gain: I(θ; y) = H(y) - H(y|θ) = ∫p(y|θ)log[p(y|θ)/p(y)]dθdy
- Safe Exploration: subject to P(constraint violation) ≤ δ

Author: Bayesian RL Implementation
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal, Categorical
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from abc import ABC, abstractmethod
import time
from scipy.optimize import minimize
from scipy.stats import entropy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplorationStrategy(Enum):
    """Available exploration strategies"""
    THOMPSON_SAMPLING = auto()
    UCB = auto()
    UCB_V = auto()                # UCB with variance
    INFO_GAIN_MAX = auto()        # Information gain maximization
    BOLTZMANN = auto()            # Boltzmann exploration
    EPSILON_GREEDY = auto()       # ε-greedy
    SAFE_UCB = auto()            # Safe UCB with constraints
    OPTIMISTIC_SAMPLING = auto()  # Optimistic Thompson sampling
    PROBABILITY_MATCHING = auto() # Probability matching


@dataclass
class ExplorationConfig:
    """Configuration for exploration strategies"""
    # Thompson Sampling parameters
    thompson_samples: int = 10
    thompson_temperature: float = 1.0
    
    # UCB parameters
    ucb_beta: float = 2.0
    ucb_confidence: float = 0.95
    
    # Information gain parameters
    info_gain_samples: int = 100
    info_gain_candidates: int = 50
    mutual_info_method: str = "monte_carlo"  # "monte_carlo", "variational"
    
    # Boltzmann parameters
    boltzmann_temperature: float = 1.0
    boltzmann_decay: float = 0.999
    
    # Epsilon-greedy parameters
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    
    # Safe exploration parameters
    safety_threshold: float = 0.05
    safety_confidence: float = 0.95
    constraint_violation_cost: float = 100.0
    
    # General parameters
    action_candidates: int = 20
    optimization_steps: int = 50
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


class ExplorationStrategy(ABC):
    """Abstract base class for exploration strategies"""
    
    def __init__(self, config: ExplorationConfig):
        """Initialize exploration strategy"""
        self.config = config
        self.step_count = 0
        self.device = torch.device(config.device)
        
    @abstractmethod
    def select_action(self, state: np.ndarray, q_function: Any, 
                     action_candidates: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select action using exploration strategy
        
        Args:
            state: Current state
            q_function: Q-function model (GP, NN, etc.)
            action_candidates: Optional action candidates
            
        Returns:
            Tuple of (selected_action, selection_info)
        """
        pass
    
    def update(self, **kwargs):
        """Update strategy parameters (e.g., decay rates)"""
        self.step_count += 1


class ThompsonSamplingStrategy(ExplorationStrategy):
    """
    Thompson Sampling for Bayesian RL
    
    Samples from posterior distribution over Q-functions and acts greedily
    with respect to the sampled Q-function.
    
    Mathematical Formulation:
    1. Sample Q ~ P(Q|D)
    2. Choose a* = argmax_a Q(s,a)
    3. Probability of selecting action a: P(a) = P(Q(s,a) > Q(s,a') ∀a' ≠ a)
    """
    
    def select_action(self, state: np.ndarray, q_function: Any,
                     action_candidates: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action using Thompson sampling"""
        if action_candidates is None:
            action_candidates = self._generate_action_candidates(len(state) if hasattr(state, '__len__') else 3)
        
        # Sample Q-values from posterior for each action candidate
        q_samples = []
        q_means = []
        q_stds = []
        
        for action in action_candidates:
            if hasattr(q_function, 'predict_q_value'):
                # GP-based Q-function
                q_pred = q_function.predict_q_value(state, action)
                
                # Sample from posterior  
                q_sample = np.random.normal(q_pred['mean'], q_pred['std'])
                q_samples.append(q_sample)
                q_means.append(q_pred['mean'])
                q_stds.append(q_pred['std'])
                
            elif hasattr(q_function, 'thompson_sampling_action'):
                # Use built-in Thompson sampling
                selected_action, info = q_function.thompson_sampling_action(state, action_candidates)
                return selected_action, {
                    'strategy': 'thompson_sampling',
                    'method': 'builtin',
                    **info
                }
            else:
                # Fallback: assume deterministic Q-function
                q_value = q_function(state, action) if callable(q_function) else 0.0
                q_sample = q_value + np.random.normal(0, self.config.thompson_temperature)
                q_samples.append(q_sample)
                q_means.append(q_value)
                q_stds.append(self.config.thompson_temperature)
        
        # Select action with highest sampled Q-value
        q_samples = np.array(q_samples)
        selected_idx = np.argmax(q_samples)
        selected_action = action_candidates[selected_idx]
        
        return selected_action, {
            'strategy': 'thompson_sampling',
            'selected_idx': selected_idx,
            'q_samples': q_samples,
            'q_means': np.array(q_means),
            'q_stds': np.array(q_stds),
            'max_q_sample': q_samples[selected_idx],
            'num_candidates': len(action_candidates),
            'temperature': self.config.thompson_temperature
        }
    
    def _generate_action_candidates(self, action_dim: int) -> np.ndarray:
        """Generate random action candidates"""
        return np.random.uniform(-1, 1, (self.config.action_candidates, action_dim)).astype(np.float32)


class UCBStrategy(ExplorationStrategy):
    """
    Upper Confidence Bound (UCB) exploration
    
    Selects actions that maximize upper confidence bound:
    UCB(s,a) = Q̄(s,a) + β * √Var(Q(s,a))
    
    where β controls exploration-exploitation tradeoff.
    """
    
    def select_action(self, state: np.ndarray, q_function: Any,
                     action_candidates: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action using UCB"""
        if action_candidates is None:
            action_candidates = self._generate_action_candidates(len(state) if hasattr(state, '__len__') else 3)
        
        # Compute UCB values for each action candidate
        ucb_values = []
        means = []
        stds = []
        
        for action in action_candidates:
            if hasattr(q_function, 'predict_q_value'):
                # GP-based Q-function
                q_pred = q_function.predict_q_value(state, action)
                mean = q_pred['mean']
                std = q_pred['std']
                
            elif hasattr(q_function, 'ucb_action'):
                # Use built-in UCB
                selected_action, info = q_function.ucb_action(state, action_candidates)
                return selected_action, {
                    'strategy': 'ucb',
                    'method': 'builtin',
                    **info
                }
            else:
                # Fallback: assume deterministic Q-function  
                mean = q_function(state, action) if callable(q_function) else 0.0
                std = 1.0  # Default uncertainty
            
            # Compute UCB value
            ucb_value = mean + self.config.ucb_beta * std
            
            ucb_values.append(ucb_value)
            means.append(mean)
            stds.append(std)
        
        # Select action with highest UCB value
        ucb_values = np.array(ucb_values)
        selected_idx = np.argmax(ucb_values)
        selected_action = action_candidates[selected_idx]
        
        return selected_action, {
            'strategy': 'ucb',
            'selected_idx': selected_idx,
            'ucb_values': ucb_values,
            'means': np.array(means),
            'stds': np.array(stds),
            'beta': self.config.ucb_beta,
            'max_ucb': ucb_values[selected_idx],
            'num_candidates': len(action_candidates)
        }
    
    def _generate_action_candidates(self, action_dim: int) -> np.ndarray:
        """Generate random action candidates"""
        return np.random.uniform(-1, 1, (self.config.action_candidates, action_dim)).astype(np.float32)


class InformationGainStrategy(ExplorationStrategy):
    """
    Information Gain Maximization for Exploration
    
    Selects actions that maximize expected information gain about model parameters:
    I(θ; y) = E_{p(y|x,θ)}[log p(y|x,θ) - log p(y|x)]
    
    where y is the observed outcome and θ are model parameters.
    """
    
    def select_action(self, state: np.ndarray, q_function: Any,
                     action_candidates: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action using information gain maximization"""
        if action_candidates is None:
            action_candidates = self._generate_action_candidates(len(state) if hasattr(state, '__len__') else 3)
        
        # Compute information gain for each action candidate
        info_gains = []
        
        for action in action_candidates:
            info_gain = self._compute_information_gain(state, action, q_function)
            info_gains.append(info_gain)
        
        # Select action with highest information gain
        info_gains = np.array(info_gains)
        selected_idx = np.argmax(info_gains)
        selected_action = action_candidates[selected_idx]
        
        return selected_action, {
            'strategy': 'information_gain',
            'selected_idx': selected_idx,
            'info_gains': info_gains,
            'max_info_gain': info_gains[selected_idx],
            'num_candidates': len(action_candidates),
            'method': self.config.mutual_info_method
        }
    
    def _compute_information_gain(self, state: np.ndarray, action: np.ndarray, q_function: Any) -> float:
        """
        Compute expected information gain for state-action pair
        
        For GP-based Q-functions, information gain can be computed as:
        I = 0.5 * log(1 + σ²(x)/σ_noise²)
        
        where σ²(x) is the predictive variance at input x.
        """
        if hasattr(q_function, 'predict_q_value'):
            # GP-based Q-function
            q_pred = q_function.predict_q_value(state, action)
            
            # Information gain approximation using predictive variance
            predictive_variance = q_pred['variance']
            noise_variance = q_pred.get('aleatoric_uncertainty', 0.01)**2
            
            # Shannon information gain (in nats)
            info_gain = 0.5 * np.log(1 + predictive_variance / noise_variance)
            
            return info_gain
            
        elif hasattr(q_function, 'gp_model') and q_function.gp_model is not None:
            # Direct GP model access
            state_tensor = torch.tensor(state, dtype=self.config.dtype).unsqueeze(0)
            action_tensor = torch.tensor(action, dtype=self.config.dtype).unsqueeze(0)
            
            # Create input for GP
            if hasattr(q_function, '_create_state_action_input'):
                input_tensor = q_function._create_state_action_input(state_tensor, action_tensor)
            else:
                input_tensor = torch.cat([state_tensor, action_tensor], dim=-1)
            
            with torch.no_grad():
                # Compute predictive distribution
                q_function.gp_model.eval()
                pred = q_function.gp_model(input_tensor)
                
                # Information gain using mutual information
                # I(θ; y) = H(y) - H(y|θ) = 0.5 * log(2πeσ²) where σ² is predictive variance
                predictive_var = pred.variance.item()
                info_gain = 0.5 * np.log(2 * np.pi * np.e * predictive_var)
            
            return max(info_gain, 0.0)  # Ensure non-negative
        
        else:
            # Fallback: use variance heuristic based on model uncertainty
            # This is a simplified approach - real implementations would depend on the specific model
            return np.random.random()  # Random fallback
    
    def _monte_carlo_information_gain(self, state: np.ndarray, action: np.ndarray, 
                                    q_function: Any, num_samples: int = None) -> float:
        """Compute information gain using Monte Carlo estimation"""
        if num_samples is None:
            num_samples = self.config.info_gain_samples
        
        # This would require sampling from model posterior and computing mutual information
        # Simplified implementation for now
        if hasattr(q_function, 'predict_q_value'):
            q_pred = q_function.predict_q_value(state, action)
            return q_pred.get('epistemic_uncertainty', 0.0)
        
        return 0.0
    
    def _generate_action_candidates(self, action_dim: int) -> np.ndarray:
        """Generate action candidates"""
        return np.random.uniform(-1, 1, (self.config.action_candidates, action_dim)).astype(np.float32)


class BoltzmannStrategy(ExplorationStrategy):
    """
    Boltzmann (Softmax) Exploration
    
    Selects actions probabilistically based on Q-values:
    P(a|s) = exp(Q(s,a)/τ) / Σ_a' exp(Q(s,a')/τ)
    
    where τ is the temperature parameter.
    """
    
    def __init__(self, config: ExplorationConfig):
        """Initialize Boltzmann exploration"""
        super().__init__(config)
        self.current_temperature = config.boltzmann_temperature
    
    def select_action(self, state: np.ndarray, q_function: Any,
                     action_candidates: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action using Boltzmann exploration"""
        if action_candidates is None:
            action_candidates = self._generate_action_candidates(len(state) if hasattr(state, '__len__') else 3)
        
        # Compute Q-values for all action candidates
        q_values = []
        
        for action in action_candidates:
            if hasattr(q_function, 'predict_q_value'):
                q_pred = q_function.predict_q_value(state, action)
                q_value = q_pred['mean']
            else:
                q_value = q_function(state, action) if callable(q_function) else 0.0
            
            q_values.append(q_value)
        
        q_values = np.array(q_values)
        
        # Apply softmax with temperature
        exp_q = np.exp((q_values - np.max(q_values)) / self.current_temperature)  # Subtract max for numerical stability
        probabilities = exp_q / np.sum(exp_q)
        
        # Sample action according to probabilities
        selected_idx = np.random.choice(len(action_candidates), p=probabilities)
        selected_action = action_candidates[selected_idx]
        
        return selected_action, {
            'strategy': 'boltzmann',
            'selected_idx': selected_idx,
            'q_values': q_values,
            'probabilities': probabilities,
            'temperature': self.current_temperature,
            'entropy': entropy(probabilities),
            'num_candidates': len(action_candidates)
        }
    
    def update(self, **kwargs):
        """Update temperature (decay over time)"""
        super().update(**kwargs)
        self.current_temperature *= self.config.boltzmann_decay
        self.current_temperature = max(self.current_temperature, 0.01)  # Minimum temperature
    
    def _generate_action_candidates(self, action_dim: int) -> np.ndarray:
        """Generate action candidates"""
        return np.random.uniform(-1, 1, (self.config.action_candidates, action_dim)).astype(np.float32)


class EpsilonGreedyStrategy(ExplorationStrategy):
    """
    ε-greedy Exploration
    
    With probability ε, select random action.
    With probability 1-ε, select greedy action (highest Q-value).
    """
    
    def __init__(self, config: ExplorationConfig):
        """Initialize ε-greedy exploration"""
        super().__init__(config)
        self.current_epsilon = config.epsilon
    
    def select_action(self, state: np.ndarray, q_function: Any,
                     action_candidates: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action using ε-greedy"""
        if action_candidates is None:
            action_candidates = self._generate_action_candidates(len(state) if hasattr(state, '__len__') else 3)
        
        if np.random.random() < self.current_epsilon:
            # Random exploration
            selected_idx = np.random.choice(len(action_candidates))
            selected_action = action_candidates[selected_idx]
            
            return selected_action, {
                'strategy': 'epsilon_greedy',
                'action_type': 'exploration',
                'selected_idx': selected_idx,
                'epsilon': self.current_epsilon,
                'num_candidates': len(action_candidates)
            }
        
        else:
            # Greedy exploitation
            q_values = []
            
            for action in action_candidates:
                if hasattr(q_function, 'predict_q_value'):
                    q_pred = q_function.predict_q_value(state, action)
                    q_value = q_pred['mean']
                else:
                    q_value = q_function(state, action) if callable(q_function) else 0.0
                
                q_values.append(q_value)
            
            q_values = np.array(q_values)
            selected_idx = np.argmax(q_values)
            selected_action = action_candidates[selected_idx]
            
            return selected_action, {
                'strategy': 'epsilon_greedy',
                'action_type': 'exploitation',
                'selected_idx': selected_idx,
                'q_values': q_values,
                'max_q_value': q_values[selected_idx],
                'epsilon': self.current_epsilon,
                'num_candidates': len(action_candidates)
            }
    
    def update(self, **kwargs):
        """Update epsilon (decay over time)"""
        super().update(**kwargs)
        self.current_epsilon *= self.config.epsilon_decay
        self.current_epsilon = max(self.current_epsilon, self.config.epsilon_min)
    
    def _generate_action_candidates(self, action_dim: int) -> np.ndarray:
        """Generate action candidates"""
        return np.random.uniform(-1, 1, (self.config.action_candidates, action_dim)).astype(np.float32)


class SafeUCBStrategy(ExplorationStrategy):
    """
    Safe UCB with Constraints
    
    UCB exploration with safety constraints:
    a* = argmax_a UCB(s,a) subject to P(safety_violation(s,a)) ≤ δ
    
    Uses conservative estimates of safety constraint violations.
    """
    
    def select_action(self, state: np.ndarray, q_function: Any,
                     action_candidates: Optional[np.ndarray] = None,
                     safety_function: Optional[Callable] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action using Safe UCB"""
        if action_candidates is None:
            action_candidates = self._generate_action_candidates(len(state) if hasattr(state, '__len__') else 3)
        
        # Compute UCB values and safety estimates
        ucb_values = []
        safety_violations = []
        means = []
        stds = []
        
        for action in action_candidates:
            # Compute Q-value UCB
            if hasattr(q_function, 'predict_q_value'):
                q_pred = q_function.predict_q_value(state, action)
                mean = q_pred['mean']
                std = q_pred['std']
                ucb_value = mean + self.config.ucb_beta * std
            else:
                mean = q_function(state, action) if callable(q_function) else 0.0
                std = 1.0
                ucb_value = mean + self.config.ucb_beta * std
            
            # Compute safety constraint violation probability
            if safety_function is not None:
                safety_violation_prob = self._estimate_safety_violation(state, action, safety_function)
            else:
                # Default safety check: penalize large actions or extreme states
                action_magnitude = np.linalg.norm(action)
                state_magnitude = np.linalg.norm(state)
                safety_violation_prob = max(0, (action_magnitude - 1.0) * 0.1 + (state_magnitude - 5.0) * 0.05)
                safety_violation_prob = np.clip(safety_violation_prob, 0, 1)
            
            ucb_values.append(ucb_value)
            safety_violations.append(safety_violation_prob)
            means.append(mean)
            stds.append(std)
        
        # Filter safe actions
        ucb_values = np.array(ucb_values)
        safety_violations = np.array(safety_violations)
        safe_mask = safety_violations <= self.config.safety_threshold
        
        if not np.any(safe_mask):
            # No safe actions: choose least unsafe
            logger.warning("No safe actions found, choosing least unsafe")
            selected_idx = np.argmin(safety_violations)
            selection_type = "least_unsafe"
        else:
            # Choose highest UCB among safe actions
            safe_ucb_values = ucb_values.copy()
            safe_ucb_values[~safe_mask] = -np.inf  # Exclude unsafe actions
            selected_idx = np.argmax(safe_ucb_values)
            selection_type = "safe_ucb"
        
        selected_action = action_candidates[selected_idx]
        
        return selected_action, {
            'strategy': 'safe_ucb',
            'selection_type': selection_type,
            'selected_idx': selected_idx,
            'ucb_values': ucb_values,
            'safety_violations': safety_violations,
            'safe_mask': safe_mask,
            'num_safe_actions': np.sum(safe_mask),
            'safety_threshold': self.config.safety_threshold,
            'selected_safety_prob': safety_violations[selected_idx],
            'selected_ucb': ucb_values[selected_idx],
            'means': np.array(means),
            'stds': np.array(stds),
            'num_candidates': len(action_candidates)
        }
    
    def _estimate_safety_violation(self, state: np.ndarray, action: np.ndarray,
                                 safety_function: Callable) -> float:
        """Estimate probability of safety constraint violation"""
        try:
            # Assume safety_function returns (safety_value, uncertainty)
            result = safety_function(state, action)
            
            if isinstance(result, tuple):
                safety_value, uncertainty = result
                # Use conservative estimate: P(violation) based on lower confidence bound
                violation_threshold = 0.0  # Assumes negative values indicate violations
                lower_bound = safety_value - self.config.ucb_beta * uncertainty
                
                if lower_bound < violation_threshold:
                    # Conservative probability estimate
                    violation_prob = 0.5 * (1 + (violation_threshold - lower_bound) / (2 * uncertainty))
                    return np.clip(violation_prob, 0, 1)
                else:
                    return 0.0
            else:
                # Deterministic safety function
                return 1.0 if result < 0 else 0.0
                
        except Exception as e:
            logger.warning(f"Safety function evaluation failed: {e}")
            return 1.0  # Assume unsafe if evaluation fails
    
    def _generate_action_candidates(self, action_dim: int) -> np.ndarray:
        """Generate action candidates"""
        return np.random.uniform(-1, 1, (self.config.action_candidates, action_dim)).astype(np.float32)


class ExplorationManager:
    """
    Manager for different exploration strategies
    
    Provides a unified interface for selecting and switching between
    exploration strategies during learning.
    """
    
    def __init__(self, config: ExplorationConfig):
        """Initialize exploration manager"""
        self.config = config
        
        # Initialize all strategies
        self.strategies = {
            'thompson_sampling': ThompsonSamplingStrategy(config),
            'ucb': UCBStrategy(config),
            'information_gain': InformationGainStrategy(config),
            'boltzmann': BoltzmannStrategy(config),
            'epsilon_greedy': EpsilonGreedyStrategy(config),
            'safe_ucb': SafeUCBStrategy(config)
        }
        
        self.current_strategy = 'thompson_sampling'  # Default strategy
        self.strategy_history = []
        
    def select_action(self, state: np.ndarray, q_function: Any,
                     strategy: Optional[str] = None,
                     action_candidates: Optional[np.ndarray] = None,
                     safety_function: Optional[Callable] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select action using specified strategy
        
        Args:
            state: Current state
            q_function: Q-function model
            strategy: Strategy name (uses current_strategy if None)
            action_candidates: Optional action candidates
            safety_function: Optional safety constraint function
            
        Returns:
            Tuple of (selected_action, selection_info)
        """
        if strategy is None:
            strategy = self.current_strategy
        
        if strategy not in self.strategies:
            raise ValueError(f"Unknown exploration strategy: {strategy}")
        
        # Select action using chosen strategy
        strategy_obj = self.strategies[strategy]
        
        if strategy == 'safe_ucb':
            action, info = strategy_obj.select_action(state, q_function, action_candidates, safety_function)
        else:
            action, info = strategy_obj.select_action(state, q_function, action_candidates)
        
        # Update strategy parameters
        strategy_obj.update()
        
        # Track strategy usage
        self.strategy_history.append(strategy)
        
        # Add manager-level info
        info['exploration_manager'] = {
            'current_strategy': self.current_strategy,
            'strategy_used': strategy,
            'total_selections': len(self.strategy_history)
        }
        
        return action, info
    
    def set_strategy(self, strategy: str):
        """Set current exploration strategy"""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown exploration strategy: {strategy}")
        self.current_strategy = strategy
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get statistics about strategy usage"""
        from collections import Counter
        
        strategy_counts = Counter(self.strategy_history)
        
        return {
            'current_strategy': self.current_strategy,
            'total_selections': len(self.strategy_history),
            'strategy_counts': dict(strategy_counts),
            'strategy_frequencies': {k: v/len(self.strategy_history) for k, v in strategy_counts.items()} if self.strategy_history else {},
            'available_strategies': list(self.strategies.keys())
        }
    
    def adaptive_strategy_selection(self, performance_metrics: Dict[str, float]) -> str:
        """
        Adaptively select exploration strategy based on performance
        
        Args:
            performance_metrics: Dictionary with performance metrics
            
        Returns:
            Recommended strategy name
        """
        # Simple heuristic-based strategy selection
        # In practice, this could be more sophisticated (e.g., multi-armed bandit)
        
        uncertainty = performance_metrics.get('uncertainty', 0.5)
        safety_violations = performance_metrics.get('safety_violations', 0)
        episode_reward = performance_metrics.get('episode_reward', 0)
        
        # High uncertainty -> information gain or Thompson sampling
        if uncertainty > 0.7:
            return 'information_gain' if uncertainty > 0.8 else 'thompson_sampling'
        
        # Safety concerns -> safe exploration
        if safety_violations > 0:
            return 'safe_ucb'
        
        # Good performance -> exploit more
        if episode_reward > performance_metrics.get('average_reward', 0):
            return 'ucb'
        
        # Default
        return 'thompson_sampling'


# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing exploration strategies")
    
    # Mock Q-function for testing
    class MockQFunction:
        def predict_q_value(self, state, action):
            return {
                'mean': np.sum(state) + np.sum(action),
                'std': 0.5,
                'variance': 0.25,
                'epistemic_uncertainty': 0.3,
                'aleatoric_uncertainty': 0.1
            }
    
    # Configuration
    config = ExplorationConfig(
        thompson_samples=5,
        ucb_beta=1.5,
        action_candidates=10
    )
    
    # Initialize exploration manager
    exploration_manager = ExplorationManager(config)
    
    # Mock data
    state = np.array([1.0, 2.0, -0.5]).astype(np.float32)
    q_function = MockQFunction()
    
    # Test different strategies
    strategies_to_test = ['thompson_sampling', 'ucb', 'information_gain', 'boltzmann', 'epsilon_greedy']
    
    for strategy in strategies_to_test:
        action, info = exploration_manager.select_action(state, q_function, strategy=strategy)
        logger.info(f"{strategy}: action={action}, info={info['strategy']}")
    
    # Test safe UCB
    def mock_safety_function(state, action):
        # Return (safety_value, uncertainty)
        safety_value = -np.linalg.norm(action) + 0.5  # Prefer smaller actions
        uncertainty = 0.1
        return safety_value, uncertainty
    
    action, info = exploration_manager.select_action(
        state, q_function, 
        strategy='safe_ucb',
        safety_function=mock_safety_function
    )
    logger.info(f"Safe UCB: action={action}, safety_prob={info['selected_safety_prob']:.3f}")
    
    # Strategy statistics
    stats = exploration_manager.get_strategy_stats()
    logger.info(f"Strategy stats: {stats}")
    
    # Adaptive strategy selection
    performance_metrics = {
        'uncertainty': 0.8,
        'safety_violations': 0,
        'episode_reward': 10.0,
        'average_reward': 8.0
    }
    
    recommended_strategy = exploration_manager.adaptive_strategy_selection(performance_metrics)
    logger.info(f"Recommended strategy: {recommended_strategy}")
    
    print("Exploration strategies test completed successfully!")