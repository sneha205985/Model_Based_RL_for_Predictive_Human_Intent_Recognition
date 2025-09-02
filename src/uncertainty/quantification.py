"""
Uncertainty Quantification and Propagation for Bayesian RL

This module implements comprehensive uncertainty quantification methods
for Bayesian RL in human-robot interaction scenarios, including:
- Epistemic vs Aleatoric uncertainty decomposition
- Uncertainty propagation through model chains
- Calibration assessment and improvement
- Risk-aware decision making

Mathematical Foundation:
- Total Uncertainty: Var[y] = E[Var[y|θ]] + Var[E[y|θ]]
- Epistemic: Var[E[y|θ]] (reducible with more data)
- Aleatoric: E[Var[y|θ]] (irreducible noise)
- Propagation: σ²(f(x)) ≈ (∇f(x))ᵀ Σₓ (∇f(x)) (first-order)

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
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UncertaintyType(Enum):
    """Types of uncertainty"""
    EPISTEMIC = auto()      # Model uncertainty (reducible)
    ALEATORIC = auto()      # Data uncertainty (irreducible)
    TOTAL = auto()          # Combined uncertainty
    PREDICTIVE = auto()     # Predictive uncertainty


class PropagationMethod(Enum):
    """Methods for uncertainty propagation"""
    MONTE_CARLO = auto()           # Monte Carlo sampling
    TAYLOR_FIRST_ORDER = auto()    # First-order Taylor expansion
    TAYLOR_SECOND_ORDER = auto()   # Second-order Taylor expansion
    UNSCENTED_TRANSFORM = auto()   # Unscented transform
    PARTICLE_FILTER = auto()       # Particle filtering


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification"""
    # General parameters
    num_samples: int = 1000
    confidence_levels: List[float] = field(default_factory=lambda: [0.68, 0.95, 0.99])
    
    # Propagation method
    propagation_method: PropagationMethod = PropagationMethod.MONTE_CARLO
    
    # Monte Carlo parameters
    mc_samples: int = 1000
    mc_batch_size: int = 100
    
    # Unscented transform parameters
    ut_alpha: float = 1e-3
    ut_beta: float = 2.0
    ut_kappa: float = 0.0
    
    # Calibration parameters
    calibration_bins: int = 10
    calibration_method: str = "isotonic"  # "isotonic" or "sigmoid"
    
    # Risk assessment parameters
    risk_measures: List[str] = field(default_factory=lambda: ["var", "cvar", "worst_case"])
    cvar_alpha: float = 0.05  # Conditional Value at Risk confidence level
    
    # Performance parameters
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


class UncertaintyQuantifier(ABC):
    """Abstract base class for uncertainty quantification"""
    
    def __init__(self, config: UncertaintyConfig):
        """Initialize uncertainty quantifier"""
        self.config = config
        self.device = torch.device(config.device)
        self.calibration_data = []
        
    @abstractmethod
    def compute_uncertainty(self, inputs: torch.Tensor, 
                          model: Any) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty estimates
        
        Args:
            inputs: Input tensor
            model: Model to compute uncertainty for
            
        Returns:
            Dictionary with uncertainty estimates
        """
        pass
    
    @abstractmethod
    def decompose_uncertainty(self, inputs: torch.Tensor,
                            model: Any) -> Dict[str, torch.Tensor]:
        """
        Decompose total uncertainty into epistemic and aleatoric components
        
        Args:
            inputs: Input tensor
            model: Model to decompose uncertainty for
            
        Returns:
            Dictionary with uncertainty decomposition
        """
        pass


class MonteCarloUncertainty(UncertaintyQuantifier):
    """
    Monte Carlo uncertainty quantification
    
    Uses Monte Carlo sampling from model posterior to estimate
    predictive uncertainty and its decomposition.
    """
    
    def compute_uncertainty(self, inputs: torch.Tensor, 
                          model: Any) -> Dict[str, torch.Tensor]:
        """Compute uncertainty using Monte Carlo sampling"""
        batch_size = inputs.shape[0]
        num_samples = self.config.mc_samples
        
        # Collect samples from model
        samples = []
        
        for i in range(0, num_samples, self.config.mc_batch_size):
            batch_samples = min(self.config.mc_batch_size, num_samples - i)
            
            if hasattr(model, 'sample_predictions'):
                # Direct sampling method
                batch_predictions = model.sample_predictions(inputs, batch_samples)
                samples.append(batch_predictions)
                
            elif hasattr(model, 'gp_model') and model.gp_model is not None:
                # Gaussian Process model
                model.gp_model.eval()
                with torch.no_grad():
                    for _ in range(batch_samples):
                        pred = model.gp_model(inputs)
                        sample = pred.sample()
                        samples.append(sample.unsqueeze(0))
            
            elif hasattr(model, 'forward') and hasattr(model, 'enable_dropout'):
                # Neural network with dropout
                model.enable_dropout()
                batch_predictions = []
                
                with torch.no_grad():
                    for _ in range(batch_samples):
                        pred = model(inputs)
                        batch_predictions.append(pred.unsqueeze(0))
                
                if batch_predictions:
                    samples.append(torch.cat(batch_predictions, dim=0))
            
            else:
                # Fallback: add noise to deterministic predictions
                logger.warning("Model doesn't support sampling, using noise-based approximation")
                with torch.no_grad():
                    pred = model(inputs) if callable(model) else torch.zeros(batch_size, 1)
                    
                for _ in range(batch_samples):
                    noise = torch.randn_like(pred) * 0.1  # 10% noise
                    samples.append((pred + noise).unsqueeze(0))
        
        if not samples:
            # Emergency fallback
            samples = [torch.zeros(1, batch_size, 1) for _ in range(10)]
        
        # Combine samples
        all_samples = torch.cat(samples, dim=0)  # [num_samples, batch_size, output_dim]
        
        # Compute statistics
        mean = all_samples.mean(dim=0)
        variance = all_samples.var(dim=0)
        std = all_samples.std(dim=0)
        
        # Confidence intervals
        confidence_intervals = {}
        for conf_level in self.config.confidence_levels:
            alpha = (1 - conf_level) / 2
            lower_quantile = torch.quantile(all_samples, alpha, dim=0)
            upper_quantile = torch.quantile(all_samples, 1 - alpha, dim=0)
            
            confidence_intervals[f'ci_{int(conf_level*100)}'] = {
                'lower': lower_quantile,
                'upper': upper_quantile
            }
        
        return {
            'mean': mean,
            'variance': variance,
            'std': std,
            'samples': all_samples,
            'confidence_intervals': confidence_intervals,
            'num_samples': num_samples
        }
    
    def decompose_uncertainty(self, inputs: torch.Tensor,
                            model: Any) -> Dict[str, torch.Tensor]:
        """
        Decompose uncertainty into epistemic and aleatoric components
        
        Uses the law of total variance:
        Var[y] = E[Var[y|θ]] + Var[E[y|θ]]
        """
        # Get Monte Carlo samples
        mc_results = self.compute_uncertainty(inputs, model)
        all_samples = mc_results['samples']  # [num_samples, batch_size, output_dim]
        
        # Total uncertainty (predictive variance)
        total_variance = mc_results['variance']
        
        # Epistemic uncertainty: variance of the means
        # This is the variance across different model parameters
        sample_means = all_samples  # Each sample represents E[y|θᵢ]
        epistemic_variance = sample_means.var(dim=0)
        
        # Aleatoric uncertainty: mean of the variances
        # For this, we need within-sample variance, which requires multiple
        # samples per parameter setting. We'll approximate it.
        
        if hasattr(model, 'get_aleatoric_uncertainty'):
            # Direct method if available
            aleatoric_variance = model.get_aleatoric_uncertainty(inputs)
        else:
            # Approximation: total - epistemic
            aleatoric_variance = total_variance - epistemic_variance
            aleatoric_variance = torch.clamp(aleatoric_variance, min=0)  # Ensure non-negative
        
        # Uncertainty decomposition
        decomposition = {
            'total_uncertainty': total_variance,
            'epistemic_uncertainty': epistemic_variance,
            'aleatoric_uncertainty': aleatoric_variance,
            'epistemic_std': torch.sqrt(epistemic_variance),
            'aleatoric_std': torch.sqrt(aleatoric_variance),
            'total_std': torch.sqrt(total_variance)
        }
        
        # Relative contributions
        total_var_safe = total_variance + 1e-8  # Avoid division by zero
        decomposition['epistemic_ratio'] = epistemic_variance / total_var_safe
        decomposition['aleatoric_ratio'] = aleatoric_variance / total_var_safe
        
        return decomposition


class UncertaintyPropagator:
    """
    Uncertainty propagation through model chains
    
    Propagates uncertainty through sequences of models or transformations,
    accounting for correlations and nonlinearities.
    """
    
    def __init__(self, config: UncertaintyConfig):
        """Initialize uncertainty propagator"""
        self.config = config
        self.device = torch.device(config.device)
    
    def propagate_through_chain(self, inputs: torch.Tensor, 
                              input_uncertainty: Dict[str, torch.Tensor],
                              model_chain: List[Any]) -> Dict[str, torch.Tensor]:
        """
        Propagate uncertainty through a chain of models
        
        Args:
            inputs: Initial inputs
            input_uncertainty: Initial input uncertainty
            model_chain: List of models to propagate through
            
        Returns:
            Dictionary with final uncertainty estimates
        """
        current_inputs = inputs
        current_uncertainty = input_uncertainty
        
        propagation_history = []
        
        for i, model in enumerate(model_chain):
            logger.info(f"Propagating through model {i+1}/{len(model_chain)}")
            
            # Propagate to next stage
            if self.config.propagation_method == PropagationMethod.MONTE_CARLO:
                next_uncertainty = self._monte_carlo_propagation(
                    current_inputs, current_uncertainty, model
                )
            elif self.config.propagation_method == PropagationMethod.TAYLOR_FIRST_ORDER:
                next_uncertainty = self._taylor_first_order_propagation(
                    current_inputs, current_uncertainty, model
                )
            elif self.config.propagation_method == PropagationMethod.UNSCENTED_TRANSFORM:
                next_uncertainty = self._unscented_transform_propagation(
                    current_inputs, current_uncertainty, model
                )
            else:
                raise ValueError(f"Unknown propagation method: {self.config.propagation_method}")
            
            # Update for next iteration
            current_inputs = next_uncertainty['mean']
            current_uncertainty = next_uncertainty
            
            # Store history
            propagation_history.append({
                'stage': i,
                'model_type': type(model).__name__,
                'input_shape': current_inputs.shape,
                'uncertainty': {
                    'total_std': next_uncertainty['std'].mean().item(),
                    'max_std': next_uncertainty['std'].max().item()
                }
            })
        
        # Add propagation history
        current_uncertainty['propagation_history'] = propagation_history
        
        return current_uncertainty
    
    def _monte_carlo_propagation(self, inputs: torch.Tensor, 
                               input_uncertainty: Dict[str, torch.Tensor],
                               model: Any) -> Dict[str, torch.Tensor]:
        """Propagate uncertainty using Monte Carlo sampling"""
        num_samples = self.config.mc_samples
        batch_size = inputs.shape[0]
        
        # Sample from input distribution
        if 'samples' in input_uncertainty:
            input_samples = input_uncertainty['samples'][:num_samples]
        else:
            # Generate samples from mean and covariance
            input_mean = input_uncertainty.get('mean', inputs)
            input_std = input_uncertainty.get('std', torch.ones_like(inputs) * 0.1)
            
            input_samples = []
            for _ in range(num_samples):
                noise = torch.randn_like(input_mean)
                sample = input_mean + noise * input_std
                input_samples.append(sample.unsqueeze(0))
            
            input_samples = torch.cat(input_samples, dim=0)
        
        # Propagate samples through model
        output_samples = []
        
        for i in range(0, num_samples, self.config.mc_batch_size):
            batch_end = min(i + self.config.mc_batch_size, num_samples)
            batch_samples = input_samples[i:batch_end]
            
            # Reshape for model input if needed
            batch_samples = batch_samples.view(-1, *inputs.shape[1:])
            
            with torch.no_grad():
                if hasattr(model, '__call__'):
                    batch_outputs = model(batch_samples)
                else:
                    # Linear transformation fallback
                    batch_outputs = batch_samples @ torch.randn(inputs.shape[-1], inputs.shape[-1])
                
                # Reshape back to sample format
                batch_outputs = batch_outputs.view(batch_end - i, batch_size, -1)
                output_samples.append(batch_outputs)
        
        # Combine output samples
        all_output_samples = torch.cat(output_samples, dim=0)
        
        # Compute output statistics
        output_mean = all_output_samples.mean(dim=0)
        output_variance = all_output_samples.var(dim=0)
        output_std = all_output_samples.std(dim=0)
        
        return {
            'mean': output_mean,
            'variance': output_variance,
            'std': output_std,
            'samples': all_output_samples
        }
    
    def _taylor_first_order_propagation(self, inputs: torch.Tensor,
                                      input_uncertainty: Dict[str, torch.Tensor],
                                      model: Any) -> Dict[str, torch.Tensor]:
        """
        First-order Taylor expansion for uncertainty propagation
        
        σ²(f(x)) ≈ (∇f(x))ᵀ Σₓ (∇f(x))
        """
        # Enable gradient computation
        inputs_grad = inputs.clone().requires_grad_(True)
        
        # Forward pass
        with torch.enable_grad():
            outputs = model(inputs_grad)
            
        # Compute Jacobian
        batch_size = inputs.shape[0]
        output_dim = outputs.shape[-1] if outputs.dim() > 1 else 1
        input_dim = inputs.shape[-1]
        
        jacobian = torch.zeros(batch_size, output_dim, input_dim, device=self.device)
        
        for i in range(output_dim):
            if outputs.dim() > 1:
                grad_outputs = torch.zeros_like(outputs)
                grad_outputs[:, i] = 1.0
            else:
                grad_outputs = torch.ones_like(outputs)
            
            grads = torch.autograd.grad(
                outputs=outputs,
                inputs=inputs_grad,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            if grads is not None:
                jacobian[:, i, :] = grads
        
        # Input covariance matrix
        input_variance = input_uncertainty.get('variance', torch.ones_like(inputs) * 0.01)
        
        # Assuming diagonal covariance for simplicity
        input_cov = torch.diag_embed(input_variance)  # [batch_size, input_dim, input_dim]
        
        # Propagate uncertainty: σ²(f(x)) = J Σₓ Jᵀ
        output_cov = torch.matmul(torch.matmul(jacobian, input_cov), jacobian.transpose(-2, -1))
        
        # Extract variances (diagonal elements)
        output_variance = torch.diagonal(output_cov, dim1=-2, dim2=-1)
        output_std = torch.sqrt(torch.clamp(output_variance, min=1e-8))
        
        return {
            'mean': outputs.detach(),
            'variance': output_variance,
            'std': output_std,
            'jacobian': jacobian.detach(),
            'output_covariance': output_cov.detach()
        }
    
    def _unscented_transform_propagation(self, inputs: torch.Tensor,
                                       input_uncertainty: Dict[str, torch.Tensor],
                                       model: Any) -> Dict[str, torch.Tensor]:
        """
        Unscented Transform for uncertainty propagation
        
        Uses sigma points to capture mean and covariance through nonlinear transformations.
        """
        batch_size, input_dim = inputs.shape
        
        # UT parameters
        alpha = self.config.ut_alpha
        beta = self.config.ut_beta
        kappa = self.config.ut_kappa
        
        lambda_param = alpha**2 * (input_dim + kappa) - input_dim
        
        # Input statistics
        input_mean = input_uncertainty.get('mean', inputs)
        input_variance = input_uncertainty.get('variance', torch.ones_like(inputs) * 0.01)
        input_cov = torch.diag_embed(input_variance)
        
        # Generate sigma points
        num_sigma_points = 2 * input_dim + 1
        sigma_points = torch.zeros(batch_size, num_sigma_points, input_dim, device=self.device)
        
        # Weights
        w_m = torch.zeros(num_sigma_points, device=self.device)
        w_c = torch.zeros(num_sigma_points, device=self.device)
        
        # Central point
        sigma_points[:, 0, :] = input_mean
        w_m[0] = lambda_param / (input_dim + lambda_param)
        w_c[0] = lambda_param / (input_dim + lambda_param) + (1 - alpha**2 + beta)
        
        # Compute matrix square root (Cholesky decomposition)
        try:
            sqrt_cov = torch.linalg.cholesky((input_dim + lambda_param) * input_cov)
        except RuntimeError:
            # Fallback: use diagonal approximation
            sqrt_cov = torch.sqrt((input_dim + lambda_param) * input_variance).unsqueeze(-1) * torch.eye(input_dim, device=self.device)
        
        # Positive and negative sigma points
        for i in range(input_dim):
            sigma_points[:, i + 1, :] = input_mean + sqrt_cov[:, :, i]
            sigma_points[:, i + 1 + input_dim, :] = input_mean - sqrt_cov[:, :, i]
            
            w_m[i + 1] = w_m[i + 1 + input_dim] = 1 / (2 * (input_dim + lambda_param))
            w_c[i + 1] = w_c[i + 1 + input_dim] = 1 / (2 * (input_dim + lambda_param))
        
        # Propagate sigma points through model
        transformed_points = []
        
        for i in range(num_sigma_points):
            points_batch = sigma_points[:, i, :]  # [batch_size, input_dim]
            
            with torch.no_grad():
                transformed = model(points_batch)
                transformed_points.append(transformed.unsqueeze(1))
        
        transformed_points = torch.cat(transformed_points, dim=1)  # [batch_size, num_sigma_points, output_dim]
        
        # Compute output mean
        output_mean = torch.sum(w_m.unsqueeze(0).unsqueeze(-1) * transformed_points, dim=1)
        
        # Compute output covariance
        output_dim = transformed_points.shape[-1]
        output_cov = torch.zeros(batch_size, output_dim, output_dim, device=self.device)
        
        for i in range(num_sigma_points):
            diff = transformed_points[:, i, :] - output_mean  # [batch_size, output_dim]
            outer_prod = torch.matmul(diff.unsqueeze(-1), diff.unsqueeze(-2))  # [batch_size, output_dim, output_dim]
            output_cov += w_c[i] * outer_prod
        
        # Extract variances and standard deviations
        output_variance = torch.diagonal(output_cov, dim1=-2, dim2=-1)
        output_std = torch.sqrt(torch.clamp(output_variance, min=1e-8))
        
        return {
            'mean': output_mean,
            'variance': output_variance,
            'std': output_std,
            'covariance': output_cov,
            'sigma_points': sigma_points.detach(),
            'transformed_points': transformed_points.detach()
        }


class UncertaintyCalibrator:
    """
    Uncertainty calibration assessment and improvement
    
    Evaluates how well uncertainty estimates match actual prediction errors
    and provides methods to improve calibration.
    """
    
    def __init__(self, config: UncertaintyConfig):
        """Initialize uncertainty calibrator"""
        self.config = config
        self.calibration_history = []
        
    def assess_calibration(self, predictions: np.ndarray, 
                         uncertainties: np.ndarray, 
                         targets: np.ndarray) -> Dict[str, Any]:
        """
        Assess calibration of uncertainty estimates
        
        Args:
            predictions: Model predictions [N, output_dim]
            uncertainties: Uncertainty estimates [N, output_dim]
            targets: True targets [N, output_dim]
            
        Returns:
            Dictionary with calibration metrics
        """
        # Compute prediction errors
        errors = np.abs(predictions - targets)
        
        # Expected calibration error (ECE)
        ece_scores = []
        reliability_diagrams = []
        
        for dim in range(predictions.shape[1]):
            pred_dim = predictions[:, dim]
            unc_dim = uncertainties[:, dim]
            target_dim = targets[:, dim]
            error_dim = errors[:, dim]
            
            # Compute reliability diagram
            prob_true, prob_pred = calibration_curve(
                (error_dim <= unc_dim).astype(int),
                unc_dim / (unc_dim.max() + 1e-8),  # Normalize uncertainties
                n_bins=self.config.calibration_bins
            )
            
            reliability_diagrams.append({
                'prob_true': prob_true,
                'prob_pred': prob_pred,
                'dimension': dim
            })
            
            # Expected Calibration Error
            bin_boundaries = np.linspace(0, 1, self.config.calibration_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Find samples in this bin
                in_bin = (unc_dim >= bin_lower * unc_dim.max()) & (unc_dim < bin_upper * unc_dim.max())
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    # Accuracy in this bin
                    accuracy_in_bin = (error_dim[in_bin] <= unc_dim[in_bin]).mean()
                    
                    # Expected confidence in this bin
                    avg_confidence_in_bin = (bin_lower + bin_upper) / 2
                    
                    # Add to ECE
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            ece_scores.append(ece)
        
        # Sharpness (average uncertainty)
        sharpness = np.mean(uncertainties, axis=0)
        
        # Coverage probability (fraction of times true value falls within uncertainty)
        coverage = np.mean(errors <= uncertainties, axis=0)
        
        # Interval score (proper scoring rule for prediction intervals)
        alpha = 0.05  # 95% prediction intervals
        interval_scores = []
        
        for dim in range(predictions.shape[1]):
            lower = predictions[:, dim] - stats.norm.ppf(1 - alpha/2) * uncertainties[:, dim]
            upper = predictions[:, dim] + stats.norm.ppf(1 - alpha/2) * uncertainties[:, dim]
            
            # Interval score components
            width = upper - lower
            below = 2 * alpha * (lower - targets[:, dim]) * (targets[:, dim] < lower)
            above = 2 * alpha * (targets[:, dim] - upper) * (targets[:, dim] > upper)
            
            interval_score = width + below + above
            interval_scores.append(np.mean(interval_score))
        
        calibration_results = {
            'expected_calibration_error': ece_scores,
            'mean_ece': np.mean(ece_scores),
            'reliability_diagrams': reliability_diagrams,
            'sharpness': sharpness,
            'coverage_probability': coverage,
            'interval_scores': interval_scores,
            'mean_interval_score': np.mean(interval_scores),
            'calibration_timestamp': time.time()
        }
        
        self.calibration_history.append(calibration_results)
        
        return calibration_results
    
    def improve_calibration(self, predictions: np.ndarray,
                          uncertainties: np.ndarray,
                          targets: np.ndarray) -> Dict[str, Any]:
        """
        Improve uncertainty calibration using post-hoc methods
        
        Args:
            predictions: Model predictions [N, output_dim]
            uncertainties: Raw uncertainty estimates [N, output_dim]
            targets: True targets [N, output_dim]
            
        Returns:
            Dictionary with calibrated uncertainties and calibration functions
        """
        calibrated_uncertainties = uncertainties.copy()
        calibration_functions = []
        
        for dim in range(predictions.shape[1]):
            pred_dim = predictions[:, dim]
            unc_dim = uncertainties[:, dim]
            target_dim = targets[:, dim]
            
            # Compute prediction errors
            errors = np.abs(pred_dim - target_dim)
            
            if self.config.calibration_method == "isotonic":
                from sklearn.isotonic import IsotonicRegression
                
                # Fit isotonic regression to map uncertainties to empirical quantiles
                iso_reg = IsotonicRegression(out_of_bounds='clip')
                
                # Sort by uncertainty
                sorted_indices = np.argsort(unc_dim)
                sorted_uncertainties = unc_dim[sorted_indices]
                sorted_errors = errors[sorted_indices]
                
                # Compute empirical quantiles
                empirical_quantiles = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                
                # Fit calibration function
                iso_reg.fit(sorted_uncertainties, empirical_quantiles)
                
                # Apply calibration
                calibrated_unc = iso_reg.predict(unc_dim)
                calibration_functions.append(('isotonic', iso_reg))
                
            elif self.config.calibration_method == "sigmoid":
                from scipy.optimize import minimize_scalar
                
                # Fit sigmoid calibration function
                def sigmoid_loss(params):
                    a, b = params
                    calibrated = 1 / (1 + np.exp(-(a * unc_dim + b)))
                    
                    # Use Brier score as loss
                    targets_binary = (errors <= np.percentile(errors, 68)).astype(float)  # Within 1 std
                    return np.mean((calibrated - targets_binary)**2)
                
                # Optimize sigmoid parameters
                result = minimize(sigmoid_loss, [1.0, 0.0], method='L-BFGS-B')
                a_opt, b_opt = result.x
                
                # Apply sigmoid calibration
                calibrated_unc = 1 / (1 + np.exp(-(a_opt * unc_dim + b_opt)))
                calibration_functions.append(('sigmoid', (a_opt, b_opt)))
            
            else:
                # No calibration
                calibrated_unc = unc_dim
                calibration_functions.append(('none', None))
            
            calibrated_uncertainties[:, dim] = calibrated_unc
        
        # Assess improvement
        original_calibration = self.assess_calibration(predictions, uncertainties, targets)
        improved_calibration = self.assess_calibration(predictions, calibrated_uncertainties, targets)
        
        return {
            'calibrated_uncertainties': calibrated_uncertainties,
            'calibration_functions': calibration_functions,
            'original_calibration': original_calibration,
            'improved_calibration': improved_calibration,
            'ece_improvement': original_calibration['mean_ece'] - improved_calibration['mean_ece']
        }


class RiskAssessment:
    """
    Risk assessment based on uncertainty quantification
    
    Provides various risk measures and decision-making tools
    based on uncertainty estimates.
    """
    
    def __init__(self, config: UncertaintyConfig):
        """Initialize risk assessment"""
        self.config = config
        
    def compute_risk_measures(self, predictions: np.ndarray,
                            uncertainties: np.ndarray,
                            costs: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute various risk measures
        
        Args:
            predictions: Model predictions [N, output_dim]
            uncertainties: Uncertainty estimates [N, output_dim]
            costs: Optional cost function values [N, output_dim]
            
        Returns:
            Dictionary with risk measures
        """
        risk_measures = {}
        
        # Value at Risk (VaR) - worst case within confidence level
        for alpha in [0.05, 0.1, 0.25]:
            var_values = []
            for dim in range(predictions.shape[1]):
                pred_dist = stats.norm(predictions[:, dim], uncertainties[:, dim])
                var_value = pred_dist.ppf(alpha)  # α-quantile
                var_values.append(var_value)
            
            risk_measures[f'var_{int(alpha*100)}'] = var_values
        
        # Conditional Value at Risk (CVaR) - expected value beyond VaR
        cvar_alpha = self.config.cvar_alpha
        cvar_values = []
        
        for dim in range(predictions.shape[1]):
            pred_dist = stats.norm(predictions[:, dim], uncertainties[:, dim])
            var_threshold = pred_dist.ppf(cvar_alpha)
            
            # Monte Carlo estimate of CVaR
            samples = pred_dist.rvs(size=10000)
            tail_samples = samples[samples <= var_threshold]
            
            if len(tail_samples) > 0:
                cvar_value = np.mean(tail_samples)
            else:
                cvar_value = var_threshold
            
            cvar_values.append(cvar_value)
        
        risk_measures['conditional_var'] = cvar_values
        
        # Worst-case scenario (mean - 3*std)
        worst_case = predictions - 3 * uncertainties
        risk_measures['worst_case'] = worst_case
        
        # Expected shortfall
        expected_shortfall = []
        for dim in range(predictions.shape[1]):
            threshold = np.percentile(predictions[:, dim] - uncertainties[:, dim], 5)
            shortfall_values = predictions[:, dim] - threshold
            shortfall_values = shortfall_values[shortfall_values > 0]
            
            expected_shortfall.append(np.mean(shortfall_values) if len(shortfall_values) > 0 else 0.0)
        
        risk_measures['expected_shortfall'] = expected_shortfall
        
        # Uncertainty-adjusted costs (if provided)
        if costs is not None:
            # Risk-adjusted cost: cost + uncertainty penalty
            uncertainty_penalty = 0.1 * uncertainties  # 10% penalty per unit uncertainty
            risk_adjusted_cost = costs + uncertainty_penalty
            risk_measures['risk_adjusted_cost'] = risk_adjusted_cost
            
            # Robust optimization: cost under worst-case uncertainty
            robust_cost = costs + 2 * uncertainties  # Conservative approach
            risk_measures['robust_cost'] = robust_cost
        
        return risk_measures
    
    def risk_aware_decision(self, action_predictions: Dict[str, np.ndarray],
                          action_uncertainties: Dict[str, np.ndarray],
                          risk_preference: str = "neutral") -> Dict[str, Any]:
        """
        Make risk-aware decisions based on predictions and uncertainties
        
        Args:
            action_predictions: Dictionary mapping actions to predictions
            action_uncertainties: Dictionary mapping actions to uncertainties
            risk_preference: "averse", "neutral", "seeking"
            
        Returns:
            Dictionary with decision analysis
        """
        action_scores = {}
        action_risks = {}
        
        for action, predictions in action_predictions.items():
            uncertainties = action_uncertainties[action]
            
            # Compute risk measures for this action
            risks = self.compute_risk_measures(
                predictions.reshape(-1, 1) if predictions.ndim == 1 else predictions,
                uncertainties.reshape(-1, 1) if uncertainties.ndim == 1 else uncertainties
            )
            
            action_risks[action] = risks
            
            # Compute risk-adjusted score
            mean_prediction = np.mean(predictions)
            mean_uncertainty = np.mean(uncertainties)
            
            if risk_preference == "averse":
                # Penalize uncertainty heavily
                score = mean_prediction - 2.0 * mean_uncertainty
            elif risk_preference == "seeking":
                # Reward uncertainty (exploration)
                score = mean_prediction + 0.5 * mean_uncertainty
            else:  # neutral
                # Standard expected value
                score = mean_prediction
            
            action_scores[action] = score
        
        # Select best action
        best_action = max(action_scores.keys(), key=lambda x: action_scores[x])
        
        return {
            'recommended_action': best_action,
            'action_scores': action_scores,
            'action_risks': action_risks,
            'risk_preference': risk_preference,
            'score_difference': action_scores[best_action] - min(action_scores.values())
        }


# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing uncertainty quantification and propagation")
    
    # Configuration
    config = UncertaintyConfig(
        num_samples=100,
        mc_samples=50,
        propagation_method=PropagationMethod.MONTE_CARLO
    )
    
    # Mock model for testing
    class MockModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            
        def forward(self, x):
            return self.linear(x)
    
    # Test uncertainty quantification
    model = MockModel(5, 2)
    inputs = torch.randn(10, 5)
    
    quantifier = MonteCarloUncertainty(config)
    uncertainty_results = quantifier.compute_uncertainty(inputs, model)
    
    logger.info(f"Uncertainty results keys: {uncertainty_results.keys()}")
    logger.info(f"Mean uncertainty: {uncertainty_results['std'].mean().item():.3f}")
    
    # Test uncertainty decomposition
    decomposition = quantifier.decompose_uncertainty(inputs, model)
    logger.info(f"Epistemic ratio: {decomposition['epistemic_ratio'].mean().item():.3f}")
    
    # Test uncertainty propagation
    propagator = UncertaintyPropagator(config)
    model_chain = [MockModel(5, 4), MockModel(4, 3), MockModel(3, 1)]
    
    input_uncertainty = {
        'mean': inputs,
        'std': torch.ones_like(inputs) * 0.1
    }
    
    final_uncertainty = propagator.propagate_through_chain(
        inputs, input_uncertainty, model_chain
    )
    
    logger.info(f"Final uncertainty std: {final_uncertainty['std'].mean().item():.3f}")
    logger.info(f"Propagation stages: {len(final_uncertainty['propagation_history'])}")
    
    # Test calibration assessment
    calibrator = UncertaintyCalibrator(config)
    
    # Generate test data
    test_predictions = np.random.randn(100, 2)
    test_uncertainties = np.abs(np.random.randn(100, 2)) * 0.5
    test_targets = test_predictions + np.random.randn(100, 2) * 0.3
    
    calibration_results = calibrator.assess_calibration(
        test_predictions, test_uncertainties, test_targets
    )
    
    logger.info(f"Expected Calibration Error: {calibration_results['mean_ece']:.3f}")
    logger.info(f"Coverage probability: {np.mean(calibration_results['coverage_probability']):.3f}")
    
    # Test risk assessment
    risk_assessor = RiskAssessment(config)
    risk_measures = risk_assessor.compute_risk_measures(test_predictions, test_uncertainties)
    
    logger.info(f"VaR 5%: {np.mean(risk_measures['var_5']):.3f}")
    logger.info(f"Conditional VaR: {np.mean(risk_measures['conditional_var']):.3f}")
    
    print("Uncertainty quantification and propagation test completed successfully!")