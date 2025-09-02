"""
Statistical Analysis Framework for Visualizations.

This module provides comprehensive statistical analysis tools including
hypothesis testing, effect size calculations, multiple comparisons,
and advanced statistical visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Ellipse
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats
from scipy.stats import (
    ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, kruskal,
    f_oneway, chi2_contingency, pearsonr, spearmanr
)
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar
import scikit_posthocs as sp

from .core_utils import BaseVisualizer, PlotConfig, StatisticsCalculator, ValidationUtils
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StatisticalTest:
    """Container for statistical test results."""
    
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    description: str = ""
    assumptions_met: Optional[Dict[str, bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultipleComparisonResult:
    """Container for multiple comparison results."""
    
    method: str
    comparisons: List[Tuple[str, str]]
    p_values: List[float]
    adjusted_p_values: List[float]
    significant: List[bool]
    alpha: float = 0.05
    effect_sizes: Optional[List[float]] = None


class TestType(Enum):
    """Types of statistical tests."""
    TTEST_IND = "independent_t_test"
    TTEST_REL = "paired_t_test"
    MANNWHITNEY = "mann_whitney_u"
    WILCOXON = "wilcoxon_signed_rank"
    ANOVA = "one_way_anova"
    KRUSKAL = "kruskal_wallis"
    CHI2 = "chi_square"
    MCNEMAR = "mcnemar"


class EffectSizeType(Enum):
    """Types of effect size measures."""
    COHENS_D = "cohens_d"
    GLASS_DELTA = "glass_delta"
    HEDGES_G = "hedges_g"
    ETA_SQUARED = "eta_squared"
    OMEGA_SQUARED = "omega_squared"
    CRAMERS_V = "cramers_v"
    PEARSON_R = "pearson_r"


class StatisticalFramework(BaseVisualizer):
    """
    Comprehensive statistical analysis and visualization framework.
    
    Provides methods for:
    - Hypothesis testing with assumption checking
    - Effect size calculations
    - Multiple comparison procedures
    - Statistical power analysis
    - Bootstrap and permutation tests
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize statistical framework.
        
        Args:
            config: Visualization configuration
        """
        super().__init__(config)
        
        # Statistical significance colors
        self.sig_colors = {
            'significant': '#E74C3C',
            'not_significant': '#95A5A6',
            'highly_significant': '#C0392B',
            'trend': '#F39C12',
            'p_value': '#3498DB'
        }
        
        logger.info("Initialized statistical analysis framework")
    
    def perform_comprehensive_analysis(self,
                                     data_groups: Dict[str, np.ndarray],
                                     test_type: Optional[TestType] = None,
                                     alpha: float = 0.05,
                                     correction_method: str = 'fdr_bh') -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis on data groups.
        
        Args:
            data_groups: Dictionary mapping group names to data arrays
            test_type: Type of statistical test (auto-detected if None)
            alpha: Significance level
            correction_method: Multiple comparison correction method
            
        Returns:
            Dictionary containing all analysis results
        """
        self._validate_data(data_groups)
        
        results = {
            'descriptive_stats': self._compute_descriptive_stats(data_groups),
            'normality_tests': self._test_normality(data_groups),
            'homogeneity_tests': self._test_homogeneity(data_groups),
            'statistical_tests': [],
            'multiple_comparisons': None,
            'effect_sizes': {},
            'power_analysis': None,
            'recommendations': []
        }
        
        # Auto-detect appropriate test if not specified
        if test_type is None:
            test_type = self._recommend_test(data_groups, results)
            results['recommended_test'] = test_type.value
        
        # Perform main statistical tests
        if len(data_groups) == 2:
            # Two-group comparison
            group_names = list(data_groups.keys())
            data1, data2 = data_groups[group_names[0]], data_groups[group_names[1]]
            
            test_result = self._perform_two_group_test(data1, data2, test_type, alpha)
            results['statistical_tests'].append(test_result)
            
            # Effect size
            effect_size = self._compute_effect_size(data1, data2, EffectSizeType.COHENS_D)
            results['effect_sizes'][f"{group_names[0]}_vs_{group_names[1]}"] = effect_size
            
        elif len(data_groups) > 2:
            # Multi-group comparison
            test_result = self._perform_multi_group_test(data_groups, test_type, alpha)
            results['statistical_tests'].append(test_result)
            
            # Post-hoc tests if significant
            if test_result.p_value < alpha:
                multiple_comp = self._perform_post_hoc_tests(data_groups, correction_method, alpha)
                results['multiple_comparisons'] = multiple_comp
                
                # Pairwise effect sizes
                group_names = list(data_groups.keys())
                for i in range(len(group_names)):
                    for j in range(i+1, len(group_names)):
                        name1, name2 = group_names[i], group_names[j]
                        data1, data2 = data_groups[name1], data_groups[name2]
                        effect_size = self._compute_effect_size(data1, data2, EffectSizeType.COHENS_D)
                        results['effect_sizes'][f"{name1}_vs_{name2}"] = effect_size
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def plot_statistical_summary(self,
                               analysis_results: Dict[str, Any],
                               save_path: Optional[str] = None) -> Union[plt.Figure, go.Figure]:
        """
        Plot comprehensive statistical analysis summary.
        
        Args:
            analysis_results: Results from perform_comprehensive_analysis
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        if self.config.interactive_engine == 'plotly':
            return self._plot_statistical_summary_plotly(analysis_results, save_path)
        else:
            return self._plot_statistical_summary_matplotlib(analysis_results, save_path)
    
    def _plot_statistical_summary_matplotlib(self,
                                           analysis_results: Dict[str, Any],
                                           save_path: Optional[str]) -> plt.Figure:
        """Create statistical summary plot using matplotlib."""
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
        
        # Descriptive statistics table
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_descriptive_table(ax1, analysis_results['descriptive_stats'])
        
        # Effect sizes
        ax2 = fig.add_subplot(gs[0, 2])
        if analysis_results['effect_sizes']:
            self._plot_effect_sizes(ax2, analysis_results['effect_sizes'])
        
        # Statistical test results
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_test_results(ax3, analysis_results['statistical_tests'])
        
        # Multiple comparisons (if available)
        if analysis_results['multiple_comparisons']:
            ax4 = fig.add_subplot(gs[2, :2])
            self._plot_multiple_comparisons(ax4, analysis_results['multiple_comparisons'])
        
        # Assumptions tests
        ax5 = fig.add_subplot(gs[2, 2])
        self._plot_assumptions_summary(ax5, analysis_results)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_effect_sizes_analysis(self,
                                 effect_sizes: Dict[str, float],
                                 save_path: Optional[str] = None,
                                 include_interpretation: bool = True) -> Union[plt.Figure, go.Figure]:
        """
        Plot effect sizes with interpretation guidelines.
        
        Args:
            effect_sizes: Dictionary mapping comparison names to effect sizes
            save_path: Path to save the plot
            include_interpretation: Whether to include interpretation bands
            
        Returns:
            Figure object
        """
        self._validate_data(effect_sizes)
        
        if self.config.interactive_engine == 'plotly':
            return self._plot_effect_sizes_plotly(effect_sizes, save_path, include_interpretation)
        else:
            return self._plot_effect_sizes_matplotlib(effect_sizes, save_path, include_interpretation)
    
    def _plot_effect_sizes_matplotlib(self,
                                    effect_sizes: Dict[str, float],
                                    save_path: Optional[str],
                                    include_interpretation: bool) -> plt.Figure:
        """Create effect sizes plot using matplotlib."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        comparisons = list(effect_sizes.keys())
        values = list(effect_sizes.values())
        
        # Color code by effect size magnitude
        colors = []
        for value in values:
            abs_value = abs(value)
            if abs_value < 0.2:
                colors.append('#95A5A6')  # Negligible
            elif abs_value < 0.5:
                colors.append('#3498DB')  # Small
            elif abs_value < 0.8:
                colors.append('#F39C12')  # Medium
            else:
                colors.append('#E74C3C')  # Large
        
        # Horizontal bar plot
        bars = ax.barh(range(len(comparisons)), values, color=colors, alpha=0.8)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            x_pos = value + 0.02 if value >= 0 else value - 0.02
            ha = 'left' if value >= 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{value:.3f}',
                   ha=ha, va='center', fontweight='bold')
        
        # Add interpretation bands
        if include_interpretation:
            ax.axvspan(-0.2, 0.2, alpha=0.2, color='gray', label='Negligible (|d| < 0.2)')
            ax.axvspan(-0.5, -0.2, alpha=0.2, color='blue', label='Small (0.2 ≤ |d| < 0.5)')
            ax.axvspan(0.2, 0.5, alpha=0.2, color='blue')
            ax.axvspan(-0.8, -0.5, alpha=0.2, color='orange', label='Medium (0.5 ≤ |d| < 0.8)')
            ax.axvspan(0.5, 0.8, alpha=0.2, color='orange')
            ax.axvspan(-2, -0.8, alpha=0.2, color='red', label='Large (|d| ≥ 0.8)')
            ax.axvspan(0.8, 2, alpha=0.2, color='red')
        
        # Zero line
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        
        ax.set_yticks(range(len(comparisons)))
        ax.set_yticklabels(comparisons)
        ax.set_xlabel("Effect Size (Cohen's d)")
        ax.set_title('Effect Sizes Analysis')
        ax.grid(True, alpha=self.config.grid_alpha, axis='x')
        
        if include_interpretation:
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_power_analysis(self,
                          effect_sizes: np.ndarray,
                          sample_sizes: np.ndarray,
                          alpha: float = 0.05,
                          save_path: Optional[str] = None) -> Union[plt.Figure, go.Figure]:
        """
        Plot statistical power analysis.
        
        Args:
            effect_sizes: Array of effect sizes to analyze
            sample_sizes: Array of sample sizes to analyze
            alpha: Significance level
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        from statsmodels.stats.power import ttest_power
        
        if self.config.interactive_engine == 'plotly':
            return self._plot_power_analysis_plotly(effect_sizes, sample_sizes, alpha, save_path)
        else:
            return self._plot_power_analysis_matplotlib(effect_sizes, sample_sizes, alpha, save_path)
    
    def _plot_power_analysis_matplotlib(self,
                                      effect_sizes: np.ndarray,
                                      sample_sizes: np.ndarray,
                                      alpha: float,
                                      save_path: Optional[str]) -> plt.Figure:
        """Create power analysis plot using matplotlib."""
        
        from statsmodels.stats.power import ttest_power
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Power vs Sample Size (for different effect sizes)
        for effect_size in effect_sizes:
            power_values = [ttest_power(effect_size, n, alpha) for n in sample_sizes]
            ax1.plot(sample_sizes, power_values, 'o-', label=f'd = {effect_size:.2f}')
        
        ax1.axhline(y=0.8, color='red', linestyle='--', label='Power = 0.8')
        ax1.set_xlabel('Sample Size (per group)')
        ax1.set_ylabel('Statistical Power')
        ax1.set_title('Power vs Sample Size')
        ax1.legend()
        ax1.grid(True, alpha=self.config.grid_alpha)
        ax1.set_ylim(0, 1)
        
        # Power vs Effect Size (for different sample sizes)
        effect_range = np.linspace(0, max(effect_sizes), 100)
        for n in [10, 20, 50, 100]:
            power_values = [ttest_power(d, n, alpha) for d in effect_range]
            ax2.plot(effect_range, power_values, 'o-', label=f'n = {n}')
        
        ax2.axhline(y=0.8, color='red', linestyle='--', label='Power = 0.8')
        ax2.axvline(x=0.8, color='green', linestyle='--', label='Large Effect')
        ax2.set_xlabel("Effect Size (Cohen's d)")
        ax2.set_ylabel('Statistical Power')
        ax2.set_title('Power vs Effect Size')
        ax2.legend()
        ax2.grid(True, alpha=self.config.grid_alpha)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_bootstrap_analysis(self,
                              data: np.ndarray,
                              statistic: Callable = np.mean,
                              n_bootstrap: int = 1000,
                              confidence_level: float = 0.95,
                              save_path: Optional[str] = None) -> Union[plt.Figure, go.Figure]:
        """
        Plot bootstrap analysis results.
        
        Args:
            data: Input data for bootstrap analysis
            statistic: Statistic function to bootstrap
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        self._validate_data(data)
        
        # Perform bootstrap
        bootstrap_stats = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        if self.config.interactive_engine == 'plotly':
            return self._plot_bootstrap_plotly(data, bootstrap_stats, statistic, 
                                             confidence_level, save_path)
        else:
            return self._plot_bootstrap_matplotlib(data, bootstrap_stats, statistic,
                                                 confidence_level, save_path)
    
    def _plot_bootstrap_matplotlib(self,
                                 data: np.ndarray,
                                 bootstrap_stats: np.ndarray,
                                 statistic: Callable,
                                 confidence_level: float,
                                 save_path: Optional[str]) -> plt.Figure:
        """Create bootstrap analysis plot using matplotlib."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original data histogram
        ax1.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(statistic(data), color='red', linestyle='--', linewidth=2,
                   label=f'Observed: {statistic(data):.3f}')
        ax1.set_xlabel('Data Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Original Data Distribution')
        ax1.legend()
        ax1.grid(True, alpha=self.config.grid_alpha)
        
        # Bootstrap distribution
        ax2.hist(bootstrap_stats, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        
        # Confidence interval
        alpha = 1 - confidence_level
        lower_ci = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper_ci = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        ax2.axvline(lower_ci, color='red', linestyle='--', label=f'{confidence_level*100}% CI')
        ax2.axvline(upper_ci, color='red', linestyle='--')
        ax2.axvline(np.mean(bootstrap_stats), color='blue', linestyle='-', linewidth=2,
                   label=f'Bootstrap Mean: {np.mean(bootstrap_stats):.3f}')
        
        ax2.set_xlabel('Bootstrap Statistic')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Bootstrap Distribution')
        ax2.legend()
        ax2.grid(True, alpha=self.config.grid_alpha)
        
        # Q-Q plot
        from scipy.stats import probplot
        probplot(bootstrap_stats, dist="norm", plot=ax3)
        ax3.set_title('Bootstrap Distribution Q-Q Plot')
        ax3.grid(True, alpha=self.config.grid_alpha)
        
        # Bootstrap convergence
        cumulative_means = np.cumsum(bootstrap_stats) / np.arange(1, len(bootstrap_stats) + 1)
        ax4.plot(cumulative_means, color='blue', linewidth=2)
        ax4.axhline(np.mean(bootstrap_stats), color='red', linestyle='--',
                   label=f'Final Mean: {np.mean(bootstrap_stats):.3f}')
        ax4.set_xlabel('Bootstrap Sample')
        ax4.set_ylabel('Cumulative Mean')
        ax4.set_title('Bootstrap Convergence')
        ax4.legend()
        ax4.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def _compute_descriptive_stats(self, data_groups: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Compute descriptive statistics for all groups."""
        stats_dict = {}
        
        for group_name, data in data_groups.items():
            stats_dict[group_name] = {
                'count': len(data),
                'mean': np.mean(data),
                'std': np.std(data, ddof=1),
                'min': np.min(data),
                'q25': np.percentile(data, 25),
                'median': np.median(data),
                'q75': np.percentile(data, 75),
                'max': np.max(data),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            }
        
        return pd.DataFrame(stats_dict).T
    
    def _test_normality(self, data_groups: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """Test normality assumptions for all groups."""
        normality_results = {}
        
        for group_name, data in data_groups.items():
            # Shapiro-Wilk test
            shapiro_stat, shapiro_p = stats.shapiro(data)
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
            
            normality_results[group_name] = {
                'shapiro_wilk': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                'kolmogorov_smirnov': {'statistic': ks_stat, 'p_value': ks_p},
                'is_normal': shapiro_p > 0.05 and ks_p > 0.05
            }
        
        return normality_results
    
    def _test_homogeneity(self, data_groups: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Test homogeneity of variances."""
        if len(data_groups) < 2:
            return {}
        
        data_arrays = list(data_groups.values())
        
        # Levene's test
        levene_stat, levene_p = stats.levene(*data_arrays)
        
        # Bartlett's test
        bartlett_stat, bartlett_p = stats.bartlett(*data_arrays)
        
        return {
            'levene': {'statistic': levene_stat, 'p_value': levene_p},
            'bartlett': {'statistic': bartlett_stat, 'p_value': bartlett_p},
            'homogeneous': levene_p > 0.05
        }
    
    def _recommend_test(self, data_groups: Dict[str, np.ndarray], 
                       results: Dict[str, Any]) -> TestType:
        """Recommend appropriate statistical test based on data characteristics."""
        n_groups = len(data_groups)
        normality_results = results['normality_tests']
        homogeneity_results = results['homogeneity_tests']
        
        # Check if all groups are normally distributed
        all_normal = all(result['is_normal'] for result in normality_results.values())
        
        if n_groups == 2:
            if all_normal and homogeneity_results.get('homogeneous', True):
                return TestType.TTEST_IND
            else:
                return TestType.MANNWHITNEY
        else:
            if all_normal and homogeneity_results.get('homogeneous', True):
                return TestType.ANOVA
            else:
                return TestType.KRUSKAL
    
    def _perform_two_group_test(self, data1: np.ndarray, data2: np.ndarray,
                               test_type: TestType, alpha: float) -> StatisticalTest:
        """Perform two-group statistical test."""
        if test_type == TestType.TTEST_IND:
            statistic, p_value = ttest_ind(data1, data2)
            effect_size = self._compute_effect_size(data1, data2, EffectSizeType.COHENS_D)
            
        elif test_type == TestType.MANNWHITNEY:
            statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
            effect_size = None  # Effect size not standard for Mann-Whitney
            
        else:
            raise ValueError(f"Unsupported test type for two groups: {test_type}")
        
        return StatisticalTest(
            test_name=test_type.value,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            description=f"Comparison between two groups using {test_type.value}"
        )
    
    def _perform_multi_group_test(self, data_groups: Dict[str, np.ndarray],
                                 test_type: TestType, alpha: float) -> StatisticalTest:
        """Perform multi-group statistical test."""
        data_arrays = list(data_groups.values())
        
        if test_type == TestType.ANOVA:
            statistic, p_value = f_oneway(*data_arrays)
            effect_size = self._compute_eta_squared(data_arrays, statistic)
            
        elif test_type == TestType.KRUSKAL:
            statistic, p_value = kruskal(*data_arrays)
            effect_size = None  # Effect size calculation for Kruskal-Wallis is complex
            
        else:
            raise ValueError(f"Unsupported test type for multiple groups: {test_type}")
        
        return StatisticalTest(
            test_name=test_type.value,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            description=f"Comparison across {len(data_groups)} groups using {test_type.value}"
        )
    
    def _perform_post_hoc_tests(self, data_groups: Dict[str, np.ndarray],
                               correction_method: str, alpha: float) -> MultipleComparisonResult:
        """Perform post-hoc multiple comparisons."""
        group_names = list(data_groups.keys())
        comparisons = []
        p_values = []
        
        # Pairwise comparisons
        for i in range(len(group_names)):
            for j in range(i+1, len(group_names)):
                name1, name2 = group_names[i], group_names[j]
                data1, data2 = data_groups[name1], data_groups[name2]
                
                # Perform t-test
                _, p_value = ttest_ind(data1, data2)
                
                comparisons.append((name1, name2))
                p_values.append(p_value)
        
        # Multiple comparison correction
        rejected, adjusted_p_values, _, _ = multipletests(
            p_values, alpha=alpha, method=correction_method
        )
        
        return MultipleComparisonResult(
            method=correction_method,
            comparisons=comparisons,
            p_values=p_values,
            adjusted_p_values=adjusted_p_values.tolist(),
            significant=rejected.tolist(),
            alpha=alpha
        )
    
    def _compute_effect_size(self, data1: np.ndarray, data2: np.ndarray,
                            effect_type: EffectSizeType) -> float:
        """Compute effect size between two groups."""
        if effect_type == EffectSizeType.COHENS_D:
            return self.statistics.effect_size_cohens_d(data1, data2)
        else:
            raise NotImplementedError(f"Effect size type {effect_type} not implemented")
    
    def _compute_eta_squared(self, data_arrays: List[np.ndarray], f_statistic: float) -> float:
        """Compute eta-squared effect size for ANOVA."""
        # This is a simplified calculation
        return f_statistic / (f_statistic + len(data_arrays[0]) - len(data_arrays))
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate statistical analysis recommendations."""
        recommendations = []
        
        # Check assumptions
        normality_results = results['normality_tests']
        homogeneity_results = results['homogeneity_tests']
        
        if not all(result['is_normal'] for result in normality_results.values()):
            recommendations.append("Consider non-parametric tests due to non-normal distributions")
        
        if not homogeneity_results.get('homogeneous', True):
            recommendations.append("Consider Welch's t-test or non-parametric alternatives due to unequal variances")
        
        # Effect size interpretation
        for comparison, effect_size in results['effect_sizes'].items():
            abs_effect = abs(effect_size)
            if abs_effect < 0.2:
                recommendations.append(f"{comparison}: Negligible practical significance (|d| = {abs_effect:.3f})")
            elif abs_effect >= 0.8:
                recommendations.append(f"{comparison}: Large practical significance (|d| = {abs_effect:.3f})")
        
        # Multiple comparison considerations
        if results['multiple_comparisons']:
            n_comparisons = len(results['multiple_comparisons'].comparisons)
            if n_comparisons > 10:
                recommendations.append(f"Consider stricter alpha correction for {n_comparisons} comparisons")
        
        return recommendations
    
    def plot(self, *args, **kwargs) -> Union[plt.Figure, go.Figure]:
        """Main plot method - delegates to appropriate visualization."""
        if 'plot_type' in kwargs:
            plot_type = kwargs.pop('plot_type')
            
            if plot_type == 'statistical_summary':
                return self.plot_statistical_summary(*args, **kwargs)
            elif plot_type == 'effect_sizes':
                return self.plot_effect_sizes_analysis(*args, **kwargs)
            elif plot_type == 'power_analysis':
                return self.plot_power_analysis(*args, **kwargs)
            elif plot_type == 'bootstrap':
                return self.plot_bootstrap_analysis(*args, **kwargs)
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
        else:
            return self.plot_statistical_summary(*args, **kwargs)


logger.info("Statistical analysis framework loaded successfully")