"""
Performance Analysis Visualization Suite.

This module provides comprehensive visualization tools for analyzing system performance
including success rates, learning curves, statistical comparisons, and benchmarking.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from sklearn.metrics import confusion_matrix, classification_report
import warnings

from .core_utils import BaseVisualizer, PlotConfig, StatisticsCalculator, PlotAnnotator, ValidationUtils
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics data."""
    
    success_rates: Dict[str, List[float]]
    completion_times: Dict[str, List[float]]
    learning_curves: Dict[str, Dict[str, List[float]]]
    statistical_tests: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class MetricType(Enum):
    """Types of performance metrics."""
    SUCCESS_RATE = "success_rate"
    COMPLETION_TIME = "completion_time"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    LOSS = "loss"
    REWARD = "reward"


class PerformanceAnalyzer(BaseVisualizer):
    """
    Comprehensive performance analysis visualization toolkit.
    
    Provides methods for:
    - Success rate comparisons
    - Learning curve analysis
    - Statistical significance testing
    - Performance distribution analysis
    - Multi-method benchmarking
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize performance analyzer.
        
        Args:
            config: Visualization configuration
        """
        super().__init__(config)
        self.statistics = StatisticsCalculator()
        self.annotator = PlotAnnotator()
        
        logger.info("Initialized performance analyzer")
    
    def plot_success_rate_comparison(self,
                                   success_rates: Dict[str, List[float]],
                                   save_path: Optional[str] = None,
                                   include_stats: bool = True,
                                   plot_type: str = 'bar') -> Union[plt.Figure, go.Figure]:
        """
        Plot success rate comparison across methods.
        
        Args:
            success_rates: Dictionary mapping method names to success rate lists
            save_path: Path to save the plot
            include_stats: Whether to include statistical annotations
            plot_type: Type of plot ('bar', 'box', 'violin')
            
        Returns:
            Figure object
        """
        self._validate_data(success_rates)
        
        # Validate probability data
        for method, rates in success_rates.items():
            ValidationUtils.validate_probability_data(np.array(rates), f"{method} success rates")
        
        if self.config.interactive_engine == 'plotly':
            return self._plot_success_rates_plotly(success_rates, plot_type, save_path, include_stats)
        else:
            return self._plot_success_rates_matplotlib(success_rates, plot_type, save_path, include_stats)
    
    def _plot_success_rates_matplotlib(self,
                                     success_rates: Dict[str, List[float]],
                                     plot_type: str,
                                     save_path: Optional[str],
                                     include_stats: bool) -> plt.Figure:
        """Create success rate plot using matplotlib."""
        
        methods = list(success_rates.keys())
        n_methods = len(methods)
        
        # Create figure based on plot type
        if plot_type == 'bar':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.config.figsize[0], self.config.figsize[1]))
        else:
            fig, ax1 = plt.subplots(figsize=self.config.figsize)
            ax2 = None
        
        colors = self.get_color_palette(n_methods, 'categorical')
        
        if plot_type == 'bar':
            # Mean success rates with error bars
            means = [np.mean(rates) for rates in success_rates.values()]
            stds = [np.std(rates) for rates in success_rates.values()]
            
            bars = ax1.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
            
            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                        f'{mean:.3f}±{std:.3f}',
                        ha='center', va='bottom', fontweight='bold')
            
            ax1.set_ylim(0, 1.1)
            ax1.set_ylabel('Success Rate')
            ax1.set_title('Mean Success Rates by Method')
            
            # Distribution comparison
            if ax2 is not None:
                all_data = list(success_rates.values())
                bp = ax2.boxplot(all_data, patch_artist=True, labels=methods)
                
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax2.set_ylabel('Success Rate')
                ax2.set_title('Success Rate Distributions')
        
        elif plot_type == 'box':
            data_list = []
            method_list = []
            
            for method, rates in success_rates.items():
                data_list.extend(rates)
                method_list.extend([method] * len(rates))
            
            df = pd.DataFrame({'Method': method_list, 'Success Rate': data_list})
            
            sns.boxplot(data=df, x='Method', y='Success Rate', ax=ax1, palette=colors)
            ax1.set_title('Success Rate Distributions')
        
        elif plot_type == 'violin':
            data_list = []
            method_list = []
            
            for method, rates in success_rates.items():
                data_list.extend(rates)
                method_list.extend([method] * len(rates))
            
            df = pd.DataFrame({'Method': method_list, 'Success Rate': data_list})
            
            sns.violinplot(data=df, x='Method', y='Success Rate', ax=ax1, palette=colors)
            ax1.set_title('Success Rate Distributions')
        
        # Add statistical annotations
        if include_stats and len(methods) > 1:
            self._add_statistical_significance(ax1, success_rates)
        
        # Apply styling
        ax1.grid(True, alpha=self.config.grid_alpha)
        if ax2 is not None:
            ax2.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def _plot_success_rates_plotly(self,
                                 success_rates: Dict[str, List[float]],
                                 plot_type: str,
                                 save_path: Optional[str],
                                 include_stats: bool) -> go.Figure:
        """Create interactive success rate plot using plotly."""
        
        if plot_type == 'bar':
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Mean Success Rates', 'Distribution Comparison'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
        else:
            fig = go.Figure()
        
        methods = list(success_rates.keys())
        colors = self.get_color_palette(len(methods), 'categorical')
        
        if plot_type == 'bar':
            # Mean success rates
            means = [np.mean(rates) for rates in success_rates.values()]
            stds = [np.std(rates) for rates in success_rates.values()]
            
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=means,
                    error_y=dict(type='data', array=stds, visible=True),
                    marker_color=colors,
                    text=[f'{m:.3f}±{s:.3f}' for m, s in zip(means, stds)],
                    textposition='outside',
                    name='Mean Success Rate'
                ),
                row=1, col=1
            )
            
            # Box plots
            for i, (method, rates) in enumerate(success_rates.items()):
                fig.add_trace(
                    go.Box(
                        y=rates,
                        name=method,
                        marker_color=colors[i],
                        boxmean='sd'
                    ),
                    row=1, col=2
                )
        
        elif plot_type in ['box', 'violin']:
            for i, (method, rates) in enumerate(success_rates.items()):
                if plot_type == 'box':
                    fig.add_trace(
                        go.Box(
                            y=rates,
                            name=method,
                            marker_color=colors[i],
                            boxmean='sd'
                        )
                    )
                else:  # violin
                    fig.add_trace(
                        go.Violin(
                            y=rates,
                            name=method,
                            fillcolor=colors[i],
                            line_color=colors[i],
                            meanline_visible=True
                        )
                    )
        
        # Update layout
        fig.update_layout(
            title='Success Rate Analysis',
            showlegend=True,
            height=600,
            template='plotly_white'
        )
        
        if plot_type == 'bar':
            fig.update_yaxes(title_text="Success Rate", range=[0, 1.1], row=1, col=1)
            fig.update_yaxes(title_text="Success Rate", range=[0, 1.1], row=1, col=2)
            fig.update_xaxes(title_text="Method", row=1, col=1)
            fig.update_xaxes(title_text="Method", row=1, col=2)
        else:
            fig.update_yaxes(title_text="Success Rate", range=[0, 1.1])
            fig.update_xaxes(title_text="Method")
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_learning_curves(self,
                           learning_data: Dict[str, Dict[str, List[float]]],
                           save_path: Optional[str] = None,
                           include_confidence: bool = True,
                           smoothing_window: Optional[int] = None) -> Union[plt.Figure, go.Figure]:
        """
        Plot learning curves with confidence intervals.
        
        Args:
            learning_data: Nested dict {method: {metric: values}}
            save_path: Path to save the plot
            include_confidence: Whether to include confidence bands
            smoothing_window: Window size for smoothing (optional)
            
        Returns:
            Figure object
        """
        self._validate_data(learning_data)
        
        if self.config.interactive_engine == 'plotly':
            return self._plot_learning_curves_plotly(learning_data, save_path, include_confidence, smoothing_window)
        else:
            return self._plot_learning_curves_matplotlib(learning_data, save_path, include_confidence, smoothing_window)
    
    def _plot_learning_curves_matplotlib(self,
                                       learning_data: Dict[str, Dict[str, List[float]]],
                                       save_path: Optional[str],
                                       include_confidence: bool,
                                       smoothing_window: Optional[int]) -> plt.Figure:
        """Create learning curves using matplotlib."""
        
        # Determine unique metrics across all methods
        all_metrics = set()
        for method_data in learning_data.values():
            all_metrics.update(method_data.keys())
        all_metrics = sorted(list(all_metrics))
        
        n_metrics = len(all_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        methods = list(learning_data.keys())
        colors = self.get_color_palette(len(methods), 'categorical')
        
        for i, metric in enumerate(all_metrics):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            for j, (method, method_data) in enumerate(learning_data.items()):
                if metric not in method_data:
                    continue
                
                values = np.array(method_data[metric])
                epochs = np.arange(1, len(values) + 1)
                
                # Apply smoothing if requested
                if smoothing_window:
                    values = self._smooth_curve(values, smoothing_window)
                
                # Plot main curve
                ax.plot(epochs, values, label=method, color=colors[j], linewidth=2, alpha=0.8)
                
                # Add confidence intervals if requested
                if include_confidence and len(values) > 10:
                    # Simple moving confidence interval
                    window = min(10, len(values) // 4)
                    ci_lower, ci_upper = self._compute_moving_confidence(values, window)
                    ax.fill_between(epochs, ci_lower, ci_upper, alpha=0.2, color=colors[j])
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'Learning Curve: {metric.replace("_", " ").title()}')
            ax.grid(True, alpha=self.config.grid_alpha)
            ax.legend()
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def _plot_learning_curves_plotly(self,
                                   learning_data: Dict[str, Dict[str, List[float]]],
                                   save_path: Optional[str],
                                   include_confidence: bool,
                                   smoothing_window: Optional[int]) -> go.Figure:
        """Create interactive learning curves using plotly."""
        
        # Determine unique metrics
        all_metrics = set()
        for method_data in learning_data.values():
            all_metrics.update(method_data.keys())
        all_metrics = sorted(list(all_metrics))
        
        n_metrics = len(all_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        subplot_titles = [metric.replace('_', ' ').title() for metric in all_metrics]
        
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        methods = list(learning_data.keys())
        colors = self.get_color_palette(len(methods), 'categorical')
        
        for i, metric in enumerate(all_metrics):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            
            for j, (method, method_data) in enumerate(learning_data.items()):
                if metric not in method_data:
                    continue
                
                values = np.array(method_data[metric])
                epochs = np.arange(1, len(values) + 1)
                
                # Apply smoothing if requested
                if smoothing_window:
                    values = self._smooth_curve(values, smoothing_window)
                
                # Add confidence bands if requested
                if include_confidence and len(values) > 10:
                    window = min(10, len(values) // 4)
                    ci_lower, ci_upper = self._compute_moving_confidence(values, window)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([epochs, epochs[::-1]]),
                            y=np.concatenate([ci_upper, ci_lower[::-1]]),
                            fill='toself',
                            fillcolor=colors[j],
                            opacity=0.2,
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False,
                            name=f'{method} CI'
                        ),
                        row=row, col=col
                    )
                
                # Main curve
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=values,
                        mode='lines',
                        name=method,
                        line=dict(color=colors[j], width=2),
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title='Learning Curves Analysis',
            height=400 * n_rows,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes labels
        for i in range(n_metrics):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            fig.update_xaxes(title_text="Epoch", row=row, col=col)
            fig.update_yaxes(title_text=all_metrics[i].replace('_', ' ').title(), row=row, col=col)
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_performance_heatmap(self,
                               performance_matrix: pd.DataFrame,
                               save_path: Optional[str] = None,
                               annotate_values: bool = True,
                               cmap: str = 'viridis') -> Union[plt.Figure, go.Figure]:
        """
        Plot performance heatmap showing metric values across methods/conditions.
        
        Args:
            performance_matrix: DataFrame with methods as rows, metrics as columns
            save_path: Path to save the plot
            annotate_values: Whether to annotate cell values
            cmap: Colormap name
            
        Returns:
            Figure object
        """
        self._validate_data(performance_matrix)
        
        if self.config.interactive_engine == 'plotly':
            return self._plot_heatmap_plotly(performance_matrix, save_path, annotate_values, cmap)
        else:
            return self._plot_heatmap_matplotlib(performance_matrix, save_path, annotate_values, cmap)
    
    def _plot_heatmap_matplotlib(self,
                               performance_matrix: pd.DataFrame,
                               save_path: Optional[str],
                               annotate_values: bool,
                               cmap: str) -> plt.Figure:
        """Create performance heatmap using matplotlib."""
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Create heatmap
        im = ax.imshow(performance_matrix.values, cmap=cmap, aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(performance_matrix.columns)))
        ax.set_yticks(np.arange(len(performance_matrix.index)))
        ax.set_xticklabels(performance_matrix.columns)
        ax.set_yticklabels(performance_matrix.index)
        
        # Rotate the tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Performance Value', rotation=-90, va="bottom")
        
        # Add text annotations
        if annotate_values:
            for i in range(len(performance_matrix.index)):
                for j in range(len(performance_matrix.columns)):
                    value = performance_matrix.iloc[i, j]
                    text = ax.text(j, i, f'{value:.3f}', ha="center", va="center", color="w")
        
        ax.set_title("Performance Metrics Heatmap")
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Methods")
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def _plot_heatmap_plotly(self,
                           performance_matrix: pd.DataFrame,
                           save_path: Optional[str],
                           annotate_values: bool,
                           cmap: str) -> go.Figure:
        """Create interactive performance heatmap using plotly."""
        
        fig = go.Figure(data=go.Heatmap(
            z=performance_matrix.values,
            x=performance_matrix.columns,
            y=performance_matrix.index,
            colorscale=cmap,
            text=performance_matrix.values.round(3) if annotate_values else None,
            texttemplate="%{text}" if annotate_values else None,
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Performance Metrics Heatmap',
            xaxis_title='Metrics',
            yaxis_title='Methods',
            height=max(400, len(performance_matrix.index) * 40),
            width=max(600, len(performance_matrix.columns) * 80)
        )
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_statistical_significance(self,
                                    data_groups: Dict[str, np.ndarray],
                                    test_type: str = 'ttest',
                                    save_path: Optional[str] = None,
                                    alpha: float = 0.05) -> Union[plt.Figure, go.Figure]:
        """
        Plot statistical significance test results.
        
        Args:
            data_groups: Dictionary mapping group names to data arrays
            test_type: Type of statistical test ('ttest', 'anova', 'mannwhitney')
            save_path: Path to save the plot
            alpha: Significance level
            
        Returns:
            Figure object with significance test results
        """
        self._validate_data(data_groups)
        
        # Perform statistical tests
        test_results = self._perform_statistical_tests(data_groups, test_type, alpha)
        
        if self.config.interactive_engine == 'plotly':
            return self._plot_significance_plotly(data_groups, test_results, save_path, alpha)
        else:
            return self._plot_significance_matplotlib(data_groups, test_results, save_path, alpha)
    
    def _plot_significance_matplotlib(self,
                                    data_groups: Dict[str, np.ndarray],
                                    test_results: Dict[str, Any],
                                    save_path: Optional[str],
                                    alpha: float) -> plt.Figure:
        """Create significance plot using matplotlib."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.config.figsize[0], self.config.figsize[1] // 2))
        
        # Box plot with significance annotations
        groups = list(data_groups.keys())
        data_list = list(data_groups.values())
        colors = self.get_color_palette(len(groups), 'categorical')
        
        bp = ax1.boxplot(data_list, patch_artist=True, labels=groups)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add significance bars
        if 'pairwise_results' in test_results:
            y_max = max([np.max(data) for data in data_list])
            y_offset = y_max * 0.05
            
            for i, (comparison, result) in enumerate(test_results['pairwise_results'].items()):
                if result['p_value'] < alpha:
                    group1, group2 = comparison.split(' vs ')
                    x1 = groups.index(group1) + 1
                    x2 = groups.index(group2) + 1
                    y = y_max + y_offset * (i + 1)
                    
                    # Significance level symbol
                    if result['p_value'] < 0.001:
                        sig_text = '***'
                    elif result['p_value'] < 0.01:
                        sig_text = '**'
                    elif result['p_value'] < 0.05:
                        sig_text = '*'
                    else:
                        sig_text = 'ns'
                    
                    ax1.plot([x1, x2], [y, y], 'k-', linewidth=1)
                    ax1.plot([x1, x1], [y-y_offset*0.1, y+y_offset*0.1], 'k-', linewidth=1)
                    ax1.plot([x2, x2], [y-y_offset*0.1, y+y_offset*0.1], 'k-', linewidth=1)
                    ax1.text((x1 + x2) / 2, y + y_offset*0.2, sig_text, ha='center', va='bottom')
        
        ax1.set_title('Data Distribution with Significance')
        ax1.grid(True, alpha=self.config.grid_alpha)
        
        # P-value matrix heatmap
        if 'pairwise_results' in test_results:
            n_groups = len(groups)
            p_matrix = np.ones((n_groups, n_groups))
            
            for comparison, result in test_results['pairwise_results'].items():
                group1, group2 = comparison.split(' vs ')
                i1, i2 = groups.index(group1), groups.index(group2)
                p_matrix[i1, i2] = result['p_value']
                p_matrix[i2, i1] = result['p_value']
            
            im = ax2.imshow(p_matrix, cmap='RdYlGn_r', vmin=0, vmax=alpha*2)
            ax2.set_xticks(range(n_groups))
            ax2.set_yticks(range(n_groups))
            ax2.set_xticklabels(groups, rotation=45)
            ax2.set_yticklabels(groups)
            
            # Add p-values as text
            for i in range(n_groups):
                for j in range(n_groups):
                    if i != j:
                        ax2.text(j, i, f'{p_matrix[i,j]:.3f}', ha="center", va="center", 
                               color="white" if p_matrix[i,j] < alpha else "black")
            
            ax2.set_title('P-value Matrix')
            plt.colorbar(im, ax=ax2, label='P-value')
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def _perform_statistical_tests(self,
                                 data_groups: Dict[str, np.ndarray],
                                 test_type: str,
                                 alpha: float) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        
        groups = list(data_groups.keys())
        results = {
            'test_type': test_type,
            'alpha': alpha,
            'pairwise_results': {}
        }
        
        if test_type == 'ttest':
            # Pairwise t-tests
            for i in range(len(groups)):
                for j in range(i+1, len(groups)):
                    group1, group2 = groups[i], groups[j]
                    data1, data2 = data_groups[group1], data_groups[group2]
                    
                    statistic, p_value = stats.ttest_ind(data1, data2)
                    effect_size = self.statistics.effect_size_cohens_d(data1, data2)
                    
                    results['pairwise_results'][f'{group1} vs {group2}'] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'effect_size': effect_size,
                        'significant': p_value < alpha
                    }
        
        elif test_type == 'anova':
            # One-way ANOVA
            statistic, p_value = stats.f_oneway(*data_groups.values())
            results['overall'] = {
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < alpha
            }
            
            # Post-hoc tests if significant
            if p_value < alpha:
                for i in range(len(groups)):
                    for j in range(i+1, len(groups)):
                        group1, group2 = groups[i], groups[j]
                        data1, data2 = data_groups[group1], data_groups[group2]
                        
                        statistic, p_value_pair = stats.ttest_ind(data1, data2)
                        # Bonferroni correction
                        n_comparisons = len(groups) * (len(groups) - 1) // 2
                        p_value_corrected = min(1.0, p_value_pair * n_comparisons)
                        
                        results['pairwise_results'][f'{group1} vs {group2}'] = {
                            'statistic': statistic,
                            'p_value': p_value_corrected,
                            'significant': p_value_corrected < alpha
                        }
        
        elif test_type == 'mannwhitney':
            # Mann-Whitney U test (non-parametric)
            for i in range(len(groups)):
                for j in range(i+1, len(groups)):
                    group1, group2 = groups[i], groups[j]
                    data1, data2 = data_groups[group1], data_groups[group2]
                    
                    statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                    
                    results['pairwise_results'][f'{group1} vs {group2}'] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < alpha
                    }
        
        return results
    
    def _smooth_curve(self, values: np.ndarray, window_size: int) -> np.ndarray:
        """Apply moving average smoothing to curve."""
        return np.convolve(values, np.ones(window_size), 'same') / window_size
    
    def _compute_moving_confidence(self, values: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute moving confidence intervals."""
        n = len(values)
        ci_lower = np.zeros(n)
        ci_upper = np.zeros(n)
        
        for i in range(n):
            start = max(0, i - window // 2)
            end = min(n, i + window // 2 + 1)
            window_data = values[start:end]
            
            mean = np.mean(window_data)
            std = np.std(window_data)
            
            ci_lower[i] = mean - 1.96 * std / np.sqrt(len(window_data))
            ci_upper[i] = mean + 1.96 * std / np.sqrt(len(window_data))
        
        return ci_lower, ci_upper
    
    def _add_statistical_significance(self, ax: plt.Axes, data_groups: Dict[str, List[float]]) -> None:
        """Add statistical significance annotations to plot."""
        groups = list(data_groups.keys())
        
        if len(groups) < 2:
            return
        
        # Perform pairwise t-tests
        y_max = ax.get_ylim()[1]
        y_offset = y_max * 0.05
        
        comparison_count = 0
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                data1 = np.array(data_groups[groups[i]])
                data2 = np.array(data_groups[groups[j]])
                
                _, p_value = stats.ttest_ind(data1, data2)
                
                if p_value < 0.05:
                    y = y_max + y_offset * (comparison_count + 1)
                    
                    if p_value < 0.001:
                        sig_text = '***'
                    elif p_value < 0.01:
                        sig_text = '**'
                    elif p_value < 0.05:
                        sig_text = '*'
                    else:
                        sig_text = 'ns'
                    
                    # Draw significance line
                    ax.plot([i, j], [y, y], 'k-', linewidth=1)
                    ax.plot([i, i], [y-y_offset*0.1, y+y_offset*0.1], 'k-', linewidth=1)
                    ax.plot([j, j], [y-y_offset*0.1, y+y_offset*0.1], 'k-', linewidth=1)
                    ax.text((i + j) / 2, y + y_offset*0.2, sig_text, ha='center', va='bottom')
                    
                    comparison_count += 1
    
    def plot(self, *args, **kwargs) -> Union[plt.Figure, go.Figure]:
        """Main plot method - delegates to appropriate visualization."""
        if 'plot_type' in kwargs:
            plot_type = kwargs.pop('plot_type')
            
            if plot_type == 'success_rates':
                return self.plot_success_rate_comparison(*args, **kwargs)
            elif plot_type == 'learning_curves':
                return self.plot_learning_curves(*args, **kwargs)
            elif plot_type == 'heatmap':
                return self.plot_performance_heatmap(*args, **kwargs)
            elif plot_type == 'significance':
                return self.plot_statistical_significance(*args, **kwargs)
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
        else:
            return self.plot_success_rate_comparison(*args, **kwargs)


logger.info("Performance analysis visualization suite loaded successfully")