"""
Bayesian Analysis Visualization Suite.

This module provides comprehensive visualization tools for Bayesian analysis
including posterior distributions, uncertainty quantification, model comparison,
and information-theoretic visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from scipy.special import kl_div
import arviz as az

from .core_utils import BaseVisualizer, PlotConfig, StatisticsCalculator, ValidationUtils
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PosteriorDistribution:
    """Container for posterior distribution data."""
    
    samples: np.ndarray
    parameter_names: List[str]
    log_likelihood: Optional[np.ndarray] = None
    prior_samples: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BayesianMetrics:
    """Container for Bayesian analysis metrics."""
    
    waic: float
    loo: float
    effective_sample_size: Dict[str, float]
    rhat: Dict[str, float]
    mcmc_diagnostics: Dict[str, Any]
    model_evidence: Optional[float] = None


class UncertaintyType(Enum):
    """Types of uncertainty."""
    EPISTEMIC = "epistemic"
    ALEATORIC = "aleatoric"
    TOTAL = "total"


class BayesianAnalyzer(BaseVisualizer):
    """
    Comprehensive Bayesian analysis visualization toolkit.
    
    Provides methods for:
    - Posterior distribution visualization
    - Uncertainty calibration plots
    - Model comparison visualizations
    - MCMC diagnostic plots
    - Information gain visualization
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize Bayesian analyzer.
        
        Args:
            config: Visualization configuration
        """
        super().__init__(config)
        
        # Bayesian-specific color scheme
        self.bayes_colors = {
            'posterior': '#3498DB',
            'prior': '#E74C3C',
            'likelihood': '#2ECC71',
            'prediction': '#F39C12',
            'uncertainty': '#9B59B6',
            'epistemic': '#E67E22',
            'aleatoric': '#1ABC9C'
        }
        
        logger.info("Initialized Bayesian analyzer")
    
    def plot_posterior_evolution(self,
                               posterior_history: List[PosteriorDistribution],
                               save_path: Optional[str] = None,
                               parameter_subset: Optional[List[str]] = None) -> Union[plt.Figure, go.Figure]:
        """
        Plot evolution of posterior distributions over time/iterations.
        
        Args:
            posterior_history: List of posterior distributions over time
            save_path: Path to save the plot
            parameter_subset: Subset of parameters to plot
            
        Returns:
            Figure object
        """
        self._validate_data(posterior_history)
        
        if self.config.interactive_engine == 'plotly':
            return self._plot_posterior_evolution_plotly(posterior_history, save_path, parameter_subset)
        else:
            return self._plot_posterior_evolution_matplotlib(posterior_history, save_path, parameter_subset)
    
    def _plot_posterior_evolution_matplotlib(self,
                                           posterior_history: List[PosteriorDistribution],
                                           save_path: Optional[str],
                                           parameter_subset: Optional[List[str]]) -> plt.Figure:
        """Create posterior evolution plot using matplotlib."""
        
        # Get parameter names
        if parameter_subset is None:
            parameter_names = posterior_history[0].parameter_names
        else:
            parameter_names = parameter_subset
        
        n_params = len(parameter_names)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(posterior_history)))
        
        for i, param_name in enumerate(parameter_names):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Plot posterior evolution
            for j, posterior in enumerate(posterior_history):
                if param_name in posterior.parameter_names:
                    param_idx = posterior.parameter_names.index(param_name)
                    samples = posterior.samples[:, param_idx]
                    
                    # Plot density
                    density = stats.gaussian_kde(samples)
                    x_range = np.linspace(samples.min(), samples.max(), 100)
                    y_density = density(x_range)
                    
                    ax.plot(x_range, y_density, color=colors[j], alpha=0.7,
                           label=f'Iteration {j+1}' if j < 5 else None)
                    
                    # Fill area for latest posterior
                    if j == len(posterior_history) - 1:
                        ax.fill_between(x_range, y_density, alpha=0.3, color=colors[j])
            
            # Plot prior if available
            if posterior_history[0].prior_samples is not None:
                param_idx = posterior_history[0].parameter_names.index(param_name)
                prior_samples = posterior_history[0].prior_samples[:, param_idx]
                prior_density = stats.gaussian_kde(prior_samples)
                x_prior = np.linspace(prior_samples.min(), prior_samples.max(), 100)
                y_prior = prior_density(x_prior)
                
                ax.plot(x_prior, y_prior, '--', color=self.bayes_colors['prior'],
                       linewidth=2, label='Prior')
            
            ax.set_xlabel(param_name)
            ax.set_ylabel('Density')
            ax.set_title(f'Posterior Evolution: {param_name}')
            if i == 0:  # Only show legend for first subplot
                ax.legend()
            ax.grid(True, alpha=self.config.grid_alpha)
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_uncertainty_calibration(self,
                                   predicted_probs: np.ndarray,
                                   true_labels: np.ndarray,
                                   n_bins: int = 10,
                                   save_path: Optional[str] = None) -> Union[plt.Figure, go.Figure]:
        """
        Plot uncertainty calibration curve.
        
        Args:
            predicted_probs: Predicted probabilities
            true_labels: True binary labels
            n_bins: Number of calibration bins
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        ValidationUtils.validate_array_shapes(predicted_probs, true_labels)
        ValidationUtils.validate_probability_data(predicted_probs, "predicted_probs")
        
        if self.config.interactive_engine == 'plotly':
            return self._plot_calibration_plotly(predicted_probs, true_labels, n_bins, save_path)
        else:
            return self._plot_calibration_matplotlib(predicted_probs, true_labels, n_bins, save_path)
    
    def _plot_calibration_matplotlib(self,
                                   predicted_probs: np.ndarray,
                                   true_labels: np.ndarray,
                                   n_bins: int,
                                   save_path: Optional[str]) -> plt.Figure:
        """Create calibration plot using matplotlib."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.config.figsize[0], self.config.figsize[1] // 2))
        
        # Compute calibration curve
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = true_labels[in_bin].mean()
                avg_confidence_in_bin = predicted_probs[in_bin].mean()
                
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
        
        # Calibration plot
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        ax1.plot(bin_confidences, bin_accuracies, 'o-', color=self.bayes_colors['posterior'],
                linewidth=2, markersize=8, label='Model')
        
        # Add bin counts as bar widths
        for i, (conf, acc, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
            bar_width = count / len(predicted_probs) * 0.1  # Scale bar width
            ax1.bar(conf, acc, width=bar_width, alpha=0.3, color=self.bayes_colors['posterior'])
        
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Plot')
        ax1.legend()
        ax1.grid(True, alpha=self.config.grid_alpha)
        
        # Reliability histogram
        ax2.hist(predicted_probs, bins=n_bins, alpha=0.7, color=self.bayes_colors['uncertainty'],
                edgecolor='black')
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Count')
        ax2.set_title('Prediction Distribution')
        ax2.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_information_gain(self,
                            information_gain_history: List[float],
                            timestamps: Optional[List[float]] = None,
                            save_path: Optional[str] = None,
                            show_cumulative: bool = True) -> Union[plt.Figure, go.Figure]:
        """
        Plot information gain during exploration/learning.
        
        Args:
            information_gain_history: History of information gain values
            timestamps: Optional timestamps
            save_path: Path to save the plot
            show_cumulative: Whether to show cumulative information gain
            
        Returns:
            Figure object
        """
        self._validate_data(information_gain_history)
        
        if timestamps is None:
            timestamps = list(range(len(information_gain_history)))
        
        ValidationUtils.validate_array_shapes(np.array(information_gain_history), np.array(timestamps))
        
        if self.config.interactive_engine == 'plotly':
            return self._plot_information_gain_plotly(information_gain_history, timestamps, 
                                                    save_path, show_cumulative)
        else:
            return self._plot_information_gain_matplotlib(information_gain_history, timestamps,
                                                        save_path, show_cumulative)
    
    def _plot_information_gain_matplotlib(self,
                                        information_gain_history: List[float],
                                        timestamps: List[float],
                                        save_path: Optional[str],
                                        show_cumulative: bool) -> plt.Figure:
        """Create information gain plot using matplotlib."""
        
        if show_cumulative:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.config.figsize[0], self.config.figsize[1]))
        else:
            fig, ax1 = plt.subplots(figsize=self.config.figsize)
            ax2 = None
        
        information_gain = np.array(information_gain_history)
        timestamps = np.array(timestamps)
        
        # Instantaneous information gain
        ax1.plot(timestamps, information_gain, 'o-', color=self.bayes_colors['posterior'],
                linewidth=2, markersize=4, alpha=0.8, label='Information Gain')
        
        # Add trend line
        if len(information_gain) > 5:
            z = np.polyfit(timestamps, information_gain, 2)
            p = np.poly1d(z)
            ax1.plot(timestamps, p(timestamps), '--', color=self.bayes_colors['uncertainty'],
                    alpha=0.7, label='Trend')
        
        # Highlight significant gains
        threshold = np.mean(information_gain) + np.std(information_gain)
        significant_gains = information_gain > threshold
        if np.any(significant_gains):
            ax1.scatter(timestamps[significant_gains], information_gain[significant_gains],
                       color='red', s=50, marker='*', label='Significant Gains', zorder=5)
        
        ax1.set_ylabel('Information Gain (bits)')
        ax1.set_title('Information Gain During Exploration')
        ax1.legend()
        ax1.grid(True, alpha=self.config.grid_alpha)
        
        if show_cumulative and ax2 is not None:
            # Cumulative information gain
            cumulative_gain = np.cumsum(information_gain)
            ax2.plot(timestamps, cumulative_gain, 'o-', color=self.bayes_colors['likelihood'],
                    linewidth=2, markersize=4, alpha=0.8, label='Cumulative Gain')
            
            ax2.set_xlabel('Time/Iteration')
            ax2.set_ylabel('Cumulative Information Gain (bits)')
            ax2.set_title('Cumulative Information Gain')
            ax2.legend()
            ax2.grid(True, alpha=self.config.grid_alpha)
        else:
            ax1.set_xlabel('Time/Iteration')
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_policy_entropy(self,
                          entropy_history: List[float],
                          timestamps: Optional[List[float]] = None,
                          save_path: Optional[str] = None,
                          include_target: Optional[float] = None) -> Union[plt.Figure, go.Figure]:
        """
        Plot policy entropy over time.
        
        Args:
            entropy_history: History of policy entropy values
            timestamps: Optional timestamps
            save_path: Path to save the plot
            include_target: Target entropy value (if applicable)
            
        Returns:
            Figure object
        """
        self._validate_data(entropy_history)
        
        if timestamps is None:
            timestamps = list(range(len(entropy_history)))
        
        ValidationUtils.validate_array_shapes(np.array(entropy_history), np.array(timestamps))
        
        if self.config.interactive_engine == 'plotly':
            return self._plot_entropy_plotly(entropy_history, timestamps, save_path, include_target)
        else:
            return self._plot_entropy_matplotlib(entropy_history, timestamps, save_path, include_target)
    
    def _plot_entropy_matplotlib(self,
                               entropy_history: List[float],
                               timestamps: List[float],
                               save_path: Optional[str],
                               include_target: Optional[float]) -> plt.Figure:
        """Create policy entropy plot using matplotlib."""
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        entropy = np.array(entropy_history)
        timestamps = np.array(timestamps)
        
        # Main entropy plot
        ax.plot(timestamps, entropy, 'o-', color=self.bayes_colors['uncertainty'],
               linewidth=2, markersize=4, alpha=0.8, label='Policy Entropy')
        
        # Add smoothed trend
        if len(entropy) > 10:
            window_size = min(10, len(entropy) // 4)
            smoothed = np.convolve(entropy, np.ones(window_size), 'valid') / window_size
            smoothed_times = timestamps[window_size//2:-(window_size//2)+1]
            ax.plot(smoothed_times, smoothed, '-', color=self.bayes_colors['posterior'],
                   linewidth=3, alpha=0.7, label='Smoothed Trend')
        
        # Target entropy line
        if include_target is not None:
            ax.axhline(y=include_target, color=self.bayes_colors['likelihood'],
                      linestyle='--', linewidth=2, label=f'Target Entropy ({include_target:.2f})')
        
        # Exploration phases
        # Identify phases with high/low entropy
        if len(entropy) > 5:
            entropy_std = np.std(entropy)
            high_entropy = entropy > np.mean(entropy) + 0.5 * entropy_std
            low_entropy = entropy < np.mean(entropy) - 0.5 * entropy_std
            
            # Highlight exploration phases
            if np.any(high_entropy):
                ax.fill_between(timestamps, 0, np.max(entropy), where=high_entropy,
                               alpha=0.2, color='orange', interpolate=True,
                               label='High Exploration')
            
            if np.any(low_entropy):
                ax.fill_between(timestamps, 0, np.max(entropy), where=low_entropy,
                               alpha=0.2, color='blue', interpolate=True,
                               label='Low Exploration')
        
        ax.set_xlabel('Time/Iteration')
        ax.set_ylabel('Policy Entropy (nats)')
        ax.set_title('Policy Entropy Evolution')
        ax.legend()
        ax.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_model_comparison(self,
                            model_metrics: Dict[str, BayesianMetrics],
                            save_path: Optional[str] = None,
                            comparison_metrics: List[str] = None) -> Union[plt.Figure, go.Figure]:
        """
        Plot Bayesian model comparison results.
        
        Args:
            model_metrics: Dictionary mapping model names to metrics
            save_path: Path to save the plot
            comparison_metrics: List of metrics to compare
            
        Returns:
            Figure object
        """
        self._validate_data(model_metrics)
        
        if comparison_metrics is None:
            comparison_metrics = ['waic', 'loo']
        
        if self.config.interactive_engine == 'plotly':
            return self._plot_model_comparison_plotly(model_metrics, save_path, comparison_metrics)
        else:
            return self._plot_model_comparison_matplotlib(model_metrics, save_path, comparison_metrics)
    
    def _plot_model_comparison_matplotlib(self,
                                        model_metrics: Dict[str, BayesianMetrics],
                                        save_path: Optional[str],
                                        comparison_metrics: List[str]) -> plt.Figure:
        """Create model comparison plot using matplotlib."""
        
        n_metrics = len(comparison_metrics)
        n_models = len(model_metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        model_names = list(model_metrics.keys())
        colors = self.get_color_palette(n_models, 'categorical')
        
        for i, metric in enumerate(comparison_metrics):
            ax = axes[i]
            
            # Extract metric values
            values = []
            for model_name in model_names:
                metrics = model_metrics[model_name]
                if hasattr(metrics, metric):
                    values.append(getattr(metrics, metric))
                else:
                    values.append(np.nan)
            
            # Create bar plot
            bars = ax.bar(model_names, values, color=colors, alpha=0.8)
            
            # Add value labels
            for bar, value in zip(bars, values):
                if not np.isnan(value):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*abs(height),
                           f'{value:.2f}', ha='center', va='bottom')
            
            # Highlight best model (lowest WAIC/LOO is better)
            if metric in ['waic', 'loo'] and not all(np.isnan(values)):
                best_idx = np.nanargmin(values)
                bars[best_idx].set_edgecolor('gold')
                bars[best_idx].set_linewidth(3)
            
            ax.set_ylabel(metric.upper())
            ax.set_title(f'Model Comparison: {metric.upper()}')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=self.config.grid_alpha, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_mcmc_diagnostics(self,
                            samples: np.ndarray,
                            parameter_names: List[str],
                            save_path: Optional[str] = None,
                            n_chains: int = 4) -> Union[plt.Figure, go.Figure]:
        """
        Plot MCMC diagnostic plots.
        
        Args:
            samples: MCMC samples [n_samples, n_params] or [n_chains, n_samples, n_params]
            parameter_names: Names of parameters
            save_path: Path to save the plot
            n_chains: Number of chains (if samples are not already separated)
            
        Returns:
            Figure object
        """
        self._validate_data(samples)
        
        # Ensure samples have chain dimension
        if samples.ndim == 2:
            # Reshape to [n_chains, samples_per_chain, n_params]
            n_samples, n_params = samples.shape
            samples_per_chain = n_samples // n_chains
            samples = samples[:samples_per_chain * n_chains].reshape(n_chains, samples_per_chain, n_params)
        
        if self.config.interactive_engine == 'plotly':
            return self._plot_mcmc_diagnostics_plotly(samples, parameter_names, save_path)
        else:
            return self._plot_mcmc_diagnostics_matplotlib(samples, parameter_names, save_path)
    
    def _plot_mcmc_diagnostics_matplotlib(self,
                                        samples: np.ndarray,
                                        parameter_names: List[str],
                                        save_path: Optional[str]) -> plt.Figure:
        """Create MCMC diagnostics using matplotlib."""
        
        n_chains, n_samples, n_params = samples.shape
        n_params_to_plot = min(6, n_params)  # Limit to 6 parameters for readability
        
        fig, axes = plt.subplots(n_params_to_plot, 2, figsize=(12, 3*n_params_to_plot))
        if n_params_to_plot == 1:
            axes = axes.reshape(1, -1)
        
        colors = self.get_color_palette(n_chains, 'categorical')
        
        for i in range(n_params_to_plot):
            param_name = parameter_names[i]
            
            # Trace plots
            ax_trace = axes[i, 0]
            for chain in range(n_chains):
                ax_trace.plot(samples[chain, :, i], color=colors[chain], alpha=0.7,
                             label=f'Chain {chain+1}' if i == 0 else "")
            
            ax_trace.set_ylabel(param_name)
            ax_trace.set_title(f'Trace Plot: {param_name}')
            if i == 0:
                ax_trace.legend()
            ax_trace.grid(True, alpha=self.config.grid_alpha)
            
            # Posterior distributions
            ax_post = axes[i, 1]
            all_samples = samples[:, :, i].flatten()
            
            # Overall posterior
            ax_post.hist(all_samples, bins=50, alpha=0.6, density=True,
                        color=self.bayes_colors['posterior'], label='Combined')
            
            # Individual chain distributions
            for chain in range(n_chains):
                chain_samples = samples[chain, :, i]
                density = stats.gaussian_kde(chain_samples)
                x_range = np.linspace(chain_samples.min(), chain_samples.max(), 100)
                ax_post.plot(x_range, density(x_range), color=colors[chain],
                           alpha=0.7, linewidth=2)
            
            ax_post.set_xlabel(param_name)
            ax_post.set_ylabel('Density')
            ax_post.set_title(f'Posterior: {param_name}')
            if i == 0:
                ax_post.legend()
            ax_post.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_uncertainty_decomposition(self,
                                     uncertainty_components: Dict[str, np.ndarray],
                                     timestamps: Optional[List[float]] = None,
                                     save_path: Optional[str] = None) -> Union[plt.Figure, go.Figure]:
        """
        Plot decomposition of uncertainty into epistemic and aleatoric components.
        
        Args:
            uncertainty_components: Dict with 'epistemic', 'aleatoric', 'total' arrays
            timestamps: Optional timestamps
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        self._validate_data(uncertainty_components)
        
        required_keys = ['epistemic', 'aleatoric']
        for key in required_keys:
            if key not in uncertainty_components:
                raise ValueError(f"Missing required uncertainty component: {key}")
        
        if timestamps is None:
            n_points = len(uncertainty_components['epistemic'])
            timestamps = list(range(n_points))
        
        if self.config.interactive_engine == 'plotly':
            return self._plot_uncertainty_decomposition_plotly(uncertainty_components, 
                                                             timestamps, save_path)
        else:
            return self._plot_uncertainty_decomposition_matplotlib(uncertainty_components,
                                                                 timestamps, save_path)
    
    def _plot_uncertainty_decomposition_matplotlib(self,
                                                 uncertainty_components: Dict[str, np.ndarray],
                                                 timestamps: List[float],
                                                 save_path: Optional[str]) -> plt.Figure:
        """Create uncertainty decomposition plot using matplotlib."""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.config.figsize[0], self.config.figsize[1]))
        
        timestamps = np.array(timestamps)
        epistemic = np.array(uncertainty_components['epistemic'])
        aleatoric = np.array(uncertainty_components['aleatoric'])
        
        # Compute total uncertainty
        if 'total' in uncertainty_components:
            total = np.array(uncertainty_components['total'])
        else:
            total = np.sqrt(epistemic**2 + aleatoric**2)
        
        # Stacked area plot
        ax1.fill_between(timestamps, 0, epistemic, alpha=0.7,
                        color=self.bayes_colors['epistemic'], label='Epistemic (Model)')
        ax1.fill_between(timestamps, epistemic, epistemic + aleatoric, alpha=0.7,
                        color=self.bayes_colors['aleatoric'], label='Aleatoric (Data)')
        
        # Total uncertainty line
        ax1.plot(timestamps, total, 'k-', linewidth=2, label='Total Uncertainty')
        
        ax1.set_ylabel('Uncertainty')
        ax1.set_title('Uncertainty Decomposition')
        ax1.legend()
        ax1.grid(True, alpha=self.config.grid_alpha)
        
        # Ratio plot
        epistemic_ratio = epistemic / (epistemic + aleatoric + 1e-8)  # Avoid division by zero
        aleatoric_ratio = aleatoric / (epistemic + aleatoric + 1e-8)
        
        ax2.plot(timestamps, epistemic_ratio, color=self.bayes_colors['epistemic'],
                linewidth=2, label='Epistemic Ratio')
        ax2.plot(timestamps, aleatoric_ratio, color=self.bayes_colors['aleatoric'],
                linewidth=2, label='Aleatoric Ratio')
        
        ax2.set_xlabel('Time/Iteration')
        ax2.set_ylabel('Uncertainty Ratio')
        ax2.set_title('Relative Uncertainty Contributions')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot(self, *args, **kwargs) -> Union[plt.Figure, go.Figure]:
        """Main plot method - delegates to appropriate visualization."""
        if 'plot_type' in kwargs:
            plot_type = kwargs.pop('plot_type')
            
            if plot_type == 'posterior_evolution':
                return self.plot_posterior_evolution(*args, **kwargs)
            elif plot_type == 'calibration':
                return self.plot_uncertainty_calibration(*args, **kwargs)
            elif plot_type == 'information_gain':
                return self.plot_information_gain(*args, **kwargs)
            elif plot_type == 'entropy':
                return self.plot_policy_entropy(*args, **kwargs)
            elif plot_type == 'model_comparison':
                return self.plot_model_comparison(*args, **kwargs)
            elif plot_type == 'mcmc_diagnostics':
                return self.plot_mcmc_diagnostics(*args, **kwargs)
            elif plot_type == 'uncertainty_decomposition':
                return self.plot_uncertainty_decomposition(*args, **kwargs)
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
        else:
            return self.plot_posterior_evolution(*args, **kwargs)


logger.info("Bayesian analysis visualization suite loaded successfully")