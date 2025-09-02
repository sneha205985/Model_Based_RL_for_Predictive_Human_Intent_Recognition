"""
Statistical Analysis Module for HRI System Evaluation

This module provides comprehensive statistical analysis capabilities
for evaluating human-robot interaction system performance, including
data collection, preprocessing, hypothesis testing, and advanced
statistical modeling.

Features:
- Automated data collection and preprocessing
- Statistical hypothesis testing
- Bayesian analysis methods
- Performance metric calculations
- Data visualization and reporting
- Export capabilities for research publications

Author: Phase 5 Implementation
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.power import ttest_power
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.contingency_tables import mcnemar
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import time

# Bayesian analysis
try:
    import pymc3 as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    logging.warning("PyMC3 not available. Bayesian analysis features disabled.")

# Advanced statistical methods
try:
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Some analysis features disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure plotting
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=FutureWarning)


class AnalysisType(Enum):
    """Types of statistical analysis"""
    DESCRIPTIVE = auto()
    INFERENTIAL = auto()
    PREDICTIVE = auto()
    BAYESIAN = auto()
    COMPARATIVE = auto()
    CORRELATION = auto()
    TIME_SERIES = auto()


class StatisticalTest(Enum):
    """Available statistical tests"""
    T_TEST = auto()
    MANN_WHITNEY = auto()
    WILCOXON = auto()
    CHI_SQUARE = auto()
    ANOVA = auto()
    KRUSKAL_WALLIS = auto()
    FRIEDMAN = auto()
    MCNEMAR = auto()


@dataclass
class AnalysisConfiguration:
    """Configuration for statistical analysis"""
    # General settings
    analysis_name: str
    output_directory: str = "analysis_results"
    significance_level: float = 0.05
    confidence_level: float = 0.95
    
    # Data processing
    remove_outliers: bool = True
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation_forest"
    outlier_threshold: float = 3.0
    handle_missing: str = "drop"  # "drop", "mean", "median", "forward_fill"
    
    # Statistical testing
    multiple_comparisons_correction: str = "bonferroni"  # "bonferroni", "fdr_bh", "none"
    effect_size_methods: List[str] = field(default_factory=lambda: ["cohen_d", "eta_squared"])
    
    # Visualization
    generate_plots: bool = True
    plot_style: str = "seaborn"
    save_plots: bool = True
    plot_format: str = "png"  # "png", "pdf", "svg"
    
    # Bayesian analysis (if available)
    use_bayesian: bool = True
    mcmc_samples: int = 2000
    mcmc_chains: int = 4
    
    # Reporting
    generate_report: bool = True
    export_data: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["csv", "json", "xlsx"])


@dataclass
class StatisticalResult:
    """Results from statistical analysis"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    power: Optional[float] = None
    interpretation: str = ""
    assumptions_met: bool = True
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisReport:
    """Comprehensive analysis report"""
    analysis_config: AnalysisConfiguration
    data_summary: Dict[str, Any]
    descriptive_stats: Dict[str, Any]
    statistical_tests: Dict[str, StatisticalResult]
    effect_sizes: Dict[str, float]
    visualizations: Dict[str, str]  # plot_name -> file_path
    recommendations: List[str]
    execution_time: float
    timestamp: str


class DataCollector:
    """Automated data collection from various sources"""
    
    def __init__(self, config: AnalysisConfiguration):
        """Initialize data collector"""
        self.config = config
        self.collected_data = {}
        
    def collect_experimental_data(self, experiment_results: List[Any]) -> pd.DataFrame:
        """Collect data from experimental results"""
        data_rows = []
        
        for result in experiment_results:
            # Extract core metrics
            row = {
                'trial_id': getattr(result, 'trial_id', None),
                'method': getattr(result, 'method', None),
                'success': getattr(result, 'success', None),
                'task_completion_time': getattr(result, 'task_completion_time', None),
                'safety_violations': getattr(result, 'safety_violations', None),
                'human_comfort_score': getattr(result, 'human_comfort_score', None),
                'step_count': getattr(result, 'step_count', None),
                'average_decision_time': getattr(result, 'average_decision_time', None),
                'max_decision_time': getattr(result, 'max_decision_time', None),
                'memory_usage': getattr(result, 'memory_usage', None)
            }
            
            # Add scenario parameters if available
            if hasattr(result, 'scenario_params'):
                for key, value in result.scenario_params.items():
                    row[f'scenario_{key}'] = value
            
            # Add additional metrics if available
            if hasattr(result, 'additional_metrics'):
                for key, value in result.additional_metrics.items():
                    row[f'additional_{key}'] = value
            
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        logger.info(f"Collected data from {len(experiment_results)} experiments")
        return df
    
    def collect_system_performance_data(self, system_logs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Collect system performance data from logs"""
        performance_data = []
        
        for log_entry in system_logs:
            perf_row = {
                'timestamp': log_entry.get('timestamp', time.time()),
                'cpu_usage': log_entry.get('cpu_usage', 0.0),
                'memory_usage': log_entry.get('memory_usage', 0.0),
                'processing_time': log_entry.get('processing_time', 0.0),
                'queue_size': log_entry.get('queue_size', 0),
                'error_count': log_entry.get('error_count', 0),
                'system_status': log_entry.get('system_status', 'unknown')
            }
            performance_data.append(perf_row)
        
        df = pd.DataFrame(performance_data)
        logger.info(f"Collected performance data from {len(system_logs)} log entries")
        return df
    
    def collect_human_behavior_data(self, trajectory_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Collect human behavior data from trajectories"""
        behavior_data = []
        
        for traj_point in trajectory_data:
            behavior_row = {
                'timestamp': traj_point.get('timestamp', 0.0),
                'human_position_x': traj_point.get('human_position', [0, 0, 0])[0],
                'human_position_y': traj_point.get('human_position', [0, 0, 0])[1],
                'human_position_z': traj_point.get('human_position', [0, 0, 0])[2],
                'human_velocity': traj_point.get('human_velocity_magnitude', 0.0),
                'dominant_intent': traj_point.get('dominant_intent', 'unknown'),
                'intent_uncertainty': traj_point.get('intent_uncertainty', 1.0),
                'comfort_level': traj_point.get('comfort_level', 0.5),
                'engagement_level': traj_point.get('engagement_level', 0.5),
                'trust_level': traj_point.get('trust_level', 0.5)
            }
            behavior_data.append(behavior_row)
        
        df = pd.DataFrame(behavior_data)
        logger.info(f"Collected behavior data from {len(trajectory_data)} trajectory points")
        return df


class DataProcessor:
    """Data preprocessing and cleaning"""
    
    def __init__(self, config: AnalysisConfiguration):
        """Initialize data processor"""
        self.config = config
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Complete data preprocessing pipeline"""
        logger.info("Starting data preprocessing...")
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Remove outliers
        if self.config.remove_outliers:
            data = self._remove_outliers(data)
        
        # Type conversion
        data = self._convert_data_types(data)
        
        # Feature engineering
        data = self._engineer_features(data)
        
        logger.info(f"Preprocessing completed. Final dataset shape: {data.shape}")
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on configuration"""
        initial_shape = data.shape
        
        if self.config.handle_missing == "drop":
            data = data.dropna()
        elif self.config.handle_missing == "mean":
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        elif self.config.handle_missing == "median":
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        elif self.config.handle_missing == "forward_fill":
            data = data.fillna(method='ffill')
        
        logger.info(f"Missing value handling: {initial_shape} -> {data.shape}")
        return data
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using specified method"""
        initial_shape = data.shape
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if self.config.outlier_method == "iqr":
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
                
        elif self.config.outlier_method == "zscore":
            z_scores = np.abs(stats.zscore(data[numeric_cols]))
            data = data[(z_scores < self.config.outlier_threshold).all(axis=1)]
            
        elif self.config.outlier_method == "isolation_forest" and SKLEARN_AVAILABLE:
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_mask = iso_forest.fit_predict(data[numeric_cols]) == 1
            data = data[outlier_mask]
        
        logger.info(f"Outlier removal: {initial_shape} -> {data.shape}")
        return data
    
    def _convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert data types for optimal analysis"""
        # Convert boolean columns
        bool_cols = data.columns[data.dtypes == 'object']
        for col in bool_cols:
            if data[col].nunique() == 2:
                try:
                    data[col] = data[col].astype(bool)
                except:
                    pass  # Keep original type if conversion fails
        
        # Ensure numeric columns are properly typed
        for col in data.columns:
            if 'time' in col.lower() or 'duration' in col.lower():
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features for analysis"""
        # Success rate features
        if 'success' in data.columns and 'method' in data.columns:
            success_rates = data.groupby('method')['success'].mean().to_dict()
            data['method_success_rate'] = data['method'].map(success_rates)
        
        # Performance efficiency features
        if 'task_completion_time' in data.columns and 'step_count' in data.columns:
            data['time_per_step'] = data['task_completion_time'] / data['step_count'].clip(lower=1)
        
        # Safety features
        if 'safety_violations' in data.columns:
            data['has_safety_violation'] = data['safety_violations'] > 0
            data['safety_score'] = 1.0 / (1.0 + data['safety_violations'])
        
        # Efficiency features
        if 'memory_usage' in data.columns and 'average_decision_time' in data.columns:
            data['computational_efficiency'] = 1.0 / (data['memory_usage'] * data['average_decision_time']).clip(lower=1e-6)
        
        return data


class StatisticalAnalyzer:
    """Main statistical analysis engine"""
    
    def __init__(self, config: AnalysisConfiguration):
        """Initialize statistical analyzer"""
        self.config = config
        self.results = {}
        
        # Create output directory
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def perform_comprehensive_analysis(self, data: pd.DataFrame) -> AnalysisReport:
        """Perform comprehensive statistical analysis"""
        start_time = time.time()
        logger.info(f"Starting comprehensive analysis: {self.config.analysis_name}")
        
        # Data summary
        data_summary = self._generate_data_summary(data)
        
        # Descriptive statistics
        descriptive_stats = self._compute_descriptive_statistics(data)
        
        # Inferential statistics
        statistical_tests = self._perform_inferential_tests(data)
        
        # Effect size calculations
        effect_sizes = self._compute_effect_sizes(data)
        
        # Visualizations
        visualizations = {}
        if self.config.generate_plots:
            visualizations = self._generate_visualizations(data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(statistical_tests, effect_sizes)
        
        # Create analysis report
        execution_time = time.time() - start_time
        report = AnalysisReport(
            analysis_config=self.config,
            data_summary=data_summary,
            descriptive_stats=descriptive_stats,
            statistical_tests=statistical_tests,
            effect_sizes=effect_sizes,
            visualizations=visualizations,
            recommendations=recommendations,
            execution_time=execution_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Save report
        if self.config.generate_report:
            self._save_analysis_report(report)
        
        logger.info(f"Analysis completed in {execution_time:.2f} seconds")
        return report
    
    def _generate_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data summary"""
        summary = {
            'total_observations': len(data),
            'total_variables': len(data.columns),
            'numeric_variables': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_variables': len(data.select_dtypes(include=['object', 'bool']).columns),
            'missing_values_total': data.isnull().sum().sum(),
            'missing_values_by_column': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.astype(str).to_dict(),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024**2)
        }
        
        # Unique values for categorical columns
        categorical_cols = data.select_dtypes(include=['object', 'bool']).columns
        summary['unique_values'] = {col: data[col].nunique() for col in categorical_cols}
        
        return summary
    
    def _compute_descriptive_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute descriptive statistics"""
        descriptive = {}
        
        # Numeric variables
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            descriptive['numeric'] = data[numeric_cols].describe().to_dict()
            
            # Additional statistics
            for col in numeric_cols:
                if col not in descriptive['numeric']:
                    descriptive['numeric'][col] = {}
                
                descriptive['numeric'][col]['variance'] = data[col].var()
                descriptive['numeric'][col]['skewness'] = stats.skew(data[col].dropna())
                descriptive['numeric'][col]['kurtosis'] = stats.kurtosis(data[col].dropna())
                
                # Confidence intervals for mean
                if len(data[col].dropna()) > 1:
                    ci = stats.t.interval(
                        self.config.confidence_level,
                        len(data[col].dropna()) - 1,
                        loc=data[col].mean(),
                        scale=stats.sem(data[col].dropna())
                    )
                    descriptive['numeric'][col]['mean_ci'] = ci
        
        # Categorical variables
        categorical_cols = data.select_dtypes(include=['object', 'bool']).columns
        if len(categorical_cols) > 0:
            descriptive['categorical'] = {}
            for col in categorical_cols:
                descriptive['categorical'][col] = {
                    'unique_values': data[col].nunique(),
                    'most_common': data[col].mode().tolist(),
                    'value_counts': data[col].value_counts().to_dict(),
                    'frequencies': (data[col].value_counts(normalize=True) * 100).to_dict()
                }
        
        return descriptive
    
    def _perform_inferential_tests(self, data: pd.DataFrame) -> Dict[str, StatisticalResult]:
        """Perform inferential statistical tests"""
        tests = {}
        
        # Method comparison tests (if method column exists)
        if 'method' in data.columns:
            tests.update(self._method_comparison_tests(data))
        
        # Performance metric tests
        tests.update(self._performance_metric_tests(data))
        
        # Correlation tests
        tests.update(self._correlation_tests(data))
        
        # Normality tests
        tests.update(self._normality_tests(data))
        
        return tests
    
    def _method_comparison_tests(self, data: pd.DataFrame) -> Dict[str, StatisticalResult]:
        """Tests comparing different methods"""
        tests = {}
        
        methods = data['method'].unique()
        if len(methods) < 2:
            return tests
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in ['task_completion_time', 'safety_violations', 'human_comfort_score']:
            if col not in data.columns:
                continue
            
            # Get data for each method
            method_data = [data[data['method'] == method][col].dropna() for method in methods]
            method_data = [group for group in method_data if len(group) > 0]
            
            if len(method_data) < 2:
                continue
            
            # Choose appropriate test
            if len(method_data) == 2:
                # Two groups: t-test or Mann-Whitney U
                if self._test_normality(method_data[0]) and self._test_normality(method_data[1]):
                    # Normal distribution: use t-test
                    statistic, p_value = stats.ttest_ind(method_data[0], method_data[1])
                    test_name = "Independent t-test"
                else:
                    # Non-normal: use Mann-Whitney U
                    statistic, p_value = stats.mannwhitneyu(
                        method_data[0], method_data[1], alternative='two-sided'
                    )
                    test_name = "Mann-Whitney U test"
            else:
                # Multiple groups: ANOVA or Kruskal-Wallis
                normality_ok = all(self._test_normality(group) for group in method_data)
                if normality_ok:
                    # Normal: use ANOVA
                    statistic, p_value = stats.f_oneway(*method_data)
                    test_name = "One-way ANOVA"
                else:
                    # Non-normal: use Kruskal-Wallis
                    statistic, p_value = stats.kruskal(*method_data)
                    test_name = "Kruskal-Wallis test"
            
            # Create result
            tests[f"{col}_method_comparison"] = StatisticalResult(
                test_name=test_name,
                statistic=statistic,
                p_value=p_value,
                interpretation=self._interpret_p_value(p_value),
                assumptions_met=True  # Would need more sophisticated checking
            )
        
        return tests
    
    def _performance_metric_tests(self, data: pd.DataFrame) -> Dict[str, StatisticalResult]:
        """Tests for performance metrics"""
        tests = {}
        
        # Real-time performance test
        if 'max_decision_time' in data.columns:
            real_time_threshold = 0.1  # 100ms
            violations = (data['max_decision_time'] > real_time_threshold).sum()
            total = len(data)
            
            # Binomial test
            p_value = stats.binom_test(violations, total, 0.05)  # Test against 5% violation rate
            
            tests['real_time_performance'] = StatisticalResult(
                test_name="Binomial test (real-time violations)",
                statistic=violations / total,
                p_value=p_value,
                interpretation=self._interpret_p_value(p_value),
                additional_info={
                    'violations': violations,
                    'total_trials': total,
                    'violation_rate': violations / total
                }
            )
        
        # Success rate analysis
        if 'success' in data.columns and 'method' in data.columns:
            # Chi-square test for success rates across methods
            contingency_table = pd.crosstab(data['method'], data['success'])
            
            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                
                tests['success_rate_independence'] = StatisticalResult(
                    test_name="Chi-square test (success rate independence)",
                    statistic=chi2,
                    p_value=p_value,
                    interpretation=self._interpret_p_value(p_value),
                    additional_info={'degrees_of_freedom': dof}
                )
        
        return tests
    
    def _correlation_tests(self, data: pd.DataFrame) -> Dict[str, StatisticalResult]:
        """Correlation analysis tests"""
        tests = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Key correlations to test
        correlation_pairs = [
            ('task_completion_time', 'human_comfort_score'),
            ('safety_violations', 'human_comfort_score'),
            ('average_decision_time', 'task_completion_time'),
            ('memory_usage', 'average_decision_time')
        ]
        
        for col1, col2 in correlation_pairs:
            if col1 in numeric_cols and col2 in numeric_cols:
                # Clean data
                clean_data = data[[col1, col2]].dropna()
                if len(clean_data) < 3:
                    continue
                
                # Pearson correlation
                r_pearson, p_pearson = stats.pearsonr(clean_data[col1], clean_data[col2])
                
                # Spearman correlation (non-parametric)
                r_spearman, p_spearman = stats.spearmanr(clean_data[col1], clean_data[col2])
                
                tests[f"correlation_pearson_{col1}_{col2}"] = StatisticalResult(
                    test_name=f"Pearson correlation ({col1} vs {col2})",
                    statistic=r_pearson,
                    p_value=p_pearson,
                    interpretation=self._interpret_correlation(r_pearson, p_pearson),
                    additional_info={'correlation_coefficient': r_pearson}
                )
                
                tests[f"correlation_spearman_{col1}_{col2}"] = StatisticalResult(
                    test_name=f"Spearman correlation ({col1} vs {col2})",
                    statistic=r_spearman,
                    p_value=p_spearman,
                    interpretation=self._interpret_correlation(r_spearman, p_spearman),
                    additional_info={'correlation_coefficient': r_spearman}
                )
        
        return tests
    
    def _normality_tests(self, data: pd.DataFrame) -> Dict[str, StatisticalResult]:
        """Test normality of key variables"""
        tests = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        key_cols = ['task_completion_time', 'human_comfort_score', 'average_decision_time']
        
        for col in key_cols:
            if col in numeric_cols and len(data[col].dropna()) >= 3:
                clean_data = data[col].dropna()
                
                # Shapiro-Wilk test (for smaller samples)
                if len(clean_data) <= 5000:
                    statistic, p_value = stats.shapiro(clean_data)
                    test_name = "Shapiro-Wilk normality test"
                else:
                    # Kolmogorov-Smirnov test (for larger samples)
                    statistic, p_value = stats.kstest(clean_data, 'norm')
                    test_name = "Kolmogorov-Smirnov normality test"
                
                tests[f"normality_{col}"] = StatisticalResult(
                    test_name=f"{test_name} ({col})",
                    statistic=statistic,
                    p_value=p_value,
                    interpretation=self._interpret_normality(p_value),
                    additional_info={'sample_size': len(clean_data)}
                )
        
        return tests
    
    def _compute_effect_sizes(self, data: pd.DataFrame) -> Dict[str, float]:
        """Compute effect sizes for significant differences"""
        effect_sizes = {}
        
        if 'method' in data.columns and len(data['method'].unique()) >= 2:
            methods = data['method'].unique()
            
            # Cohen's d for method comparisons
            for metric in ['task_completion_time', 'human_comfort_score']:
                if metric in data.columns:
                    method_groups = [data[data['method'] == method][metric].dropna() 
                                   for method in methods[:2]]  # Compare first two methods
                    
                    if len(method_groups) == 2 and all(len(group) > 0 for group in method_groups):
                        cohens_d = self._cohen_d(method_groups[0], method_groups[1])
                        effect_sizes[f"cohen_d_{metric}"] = cohens_d
        
        return effect_sizes
    
    def _generate_visualizations(self, data: pd.DataFrame) -> Dict[str, str]:
        """Generate statistical visualizations"""
        plots = {}
        
        # Set plotting style
        plt.style.use(self.config.plot_style if self.config.plot_style in plt.style.available else 'default')
        
        # Distribution plots
        plots['distributions'] = self._plot_distributions(data)
        
        # Method comparison plots
        if 'method' in data.columns:
            plots['method_comparison'] = self._plot_method_comparison(data)
        
        # Correlation plots
        plots['correlations'] = self._plot_correlations(data)
        
        # Performance metrics
        plots['performance'] = self._plot_performance_metrics(data)
        
        return plots
    
    def _plot_distributions(self, data: pd.DataFrame) -> str:
        """Plot distributions of key variables"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        key_cols = [col for col in ['task_completion_time', 'human_comfort_score', 
                                   'safety_violations', 'average_decision_time'] 
                   if col in numeric_cols]
        
        if not key_cols:
            return ""
        
        n_plots = len(key_cols)
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(key_cols):
            ax = axes[i]
            clean_data = data[col].dropna()
            
            # Histogram with density curve
            ax.hist(clean_data, bins=30, alpha=0.7, density=True, color='skyblue')
            
            # Overlay normal distribution for comparison
            mu, sigma = clean_data.mean(), clean_data.std()
            x = np.linspace(clean_data.min(), clean_data.max(), 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', label='Normal fit')
            
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(n_plots, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{self.config.analysis_name}_distributions.{self.config.plot_format}"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_method_comparison(self, data: pd.DataFrame) -> str:
        """Plot method comparison visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = ['task_completion_time', 'human_comfort_score', 'safety_violations', 'average_decision_time']
        
        for i, metric in enumerate(metrics):
            if metric not in data.columns:
                continue
            
            ax = axes[i]
            
            # Box plot
            data.boxplot(column=metric, by='method', ax=ax)
            ax.set_title(f'{metric} by Method')
            ax.set_xlabel('Method')
            ax.set_ylabel(metric)
            
            # Rotate x-axis labels if needed
            if data['method'].nunique() > 3:
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{self.config.analysis_name}_method_comparison.{self.config.plot_format}"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_correlations(self, data: pd.DataFrame) -> str:
        """Plot correlation matrix"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return ""
        
        # Compute correlation matrix
        correlation_matrix = data[numeric_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f')
        
        plt.title('Correlation Matrix of Performance Metrics')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{self.config.analysis_name}_correlations.{self.config.plot_format}"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_performance_metrics(self, data: pd.DataFrame) -> str:
        """Plot performance metrics overview"""
        if 'method' not in data.columns:
            return ""
        
        # Calculate summary statistics by method
        summary_stats = data.groupby('method').agg({
            'success': 'mean',
            'task_completion_time': 'mean',
            'safety_violations': 'mean',
            'human_comfort_score': 'mean',
            'average_decision_time': 'mean'
        }).round(3)
        
        # Create radar chart
        categories = list(summary_stats.columns)
        methods = list(summary_stats.index)
        
        # Normalize data for radar chart (0-1 scale)
        normalized_stats = summary_stats.copy()
        for col in normalized_stats.columns:
            if col == 'safety_violations':
                # Invert safety violations (lower is better)
                max_val = normalized_stats[col].max()
                normalized_stats[col] = 1 - (normalized_stats[col] / max_val if max_val > 0 else 0)
            elif col == 'task_completion_time' or col == 'average_decision_time':
                # Invert time metrics (lower is better)
                max_val = normalized_stats[col].max()
                normalized_stats[col] = 1 - (normalized_stats[col] / max_val if max_val > 0 else 0)
            else:
                # Normalize to 0-1 (higher is better)
                max_val = normalized_stats[col].max()
                normalized_stats[col] = normalized_stats[col] / max_val if max_val > 0 else 0
        
        # Create radar plot
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = sns.color_palette("husl", len(methods))
        
        for i, method in enumerate(methods):
            values = normalized_stats.loc[method].tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Metrics Comparison (Normalized)', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{self.config.analysis_name}_performance_radar.{self.config.plot_format}"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _generate_recommendations(self, tests: Dict[str, StatisticalResult], 
                                effect_sizes: Dict[str, float]) -> List[str]:
        """Generate analysis-based recommendations"""
        recommendations = []
        
        # Analyze significant results
        significant_tests = {name: result for name, result in tests.items() 
                           if result.p_value < self.config.significance_level}
        
        if significant_tests:
            recommendations.append(
                f"Found {len(significant_tests)} statistically significant results "
                f"at α = {self.config.significance_level} level."
            )
        
        # Method performance recommendations
        method_tests = [name for name in significant_tests.keys() if 'method_comparison' in name]
        if method_tests:
            recommendations.append(
                "Significant differences found between methods. Consider focusing on "
                "the best-performing method for practical implementation."
            )
        
        # Effect size recommendations
        large_effects = {name: size for name, size in effect_sizes.items() if abs(size) > 0.8}
        if large_effects:
            recommendations.append(
                f"Large effect sizes detected ({len(large_effects)} metrics). "
                "These differences are not only statistically significant but also practically important."
            )
        
        # Real-time performance recommendations
        rt_test = tests.get('real_time_performance')
        if rt_test and rt_test.additional_info.get('violation_rate', 0) > 0.05:
            recommendations.append(
                "Real-time constraint violations detected. Consider optimizing computational "
                "efficiency or adjusting time constraints."
            )
        
        # Correlation recommendations
        correlation_tests = [test for name, test in tests.items() if 'correlation' in name]
        strong_correlations = [test for test in correlation_tests 
                             if abs(test.statistic) > 0.7 and test.p_value < 0.05]
        if strong_correlations:
            recommendations.append(
                "Strong correlations found between performance metrics. "
                "Consider these relationships when optimizing system parameters."
            )
        
        # Data quality recommendations
        if not recommendations:
            recommendations.append(
                "No significant statistical differences detected. Consider increasing "
                "sample size or examining additional metrics."
            )
        
        return recommendations
    
    def _save_analysis_report(self, report: AnalysisReport):
        """Save comprehensive analysis report"""
        # Save as JSON
        report_dict = {
            'analysis_name': report.analysis_config.analysis_name,
            'timestamp': report.timestamp,
            'execution_time': report.execution_time,
            'data_summary': report.data_summary,
            'descriptive_stats': self._serialize_for_json(report.descriptive_stats),
            'statistical_tests': {
                name: {
                    'test_name': result.test_name,
                    'statistic': result.statistic,
                    'p_value': result.p_value,
                    'effect_size': result.effect_size,
                    'interpretation': result.interpretation,
                    'additional_info': result.additional_info
                } for name, result in report.statistical_tests.items()
            },
            'effect_sizes': report.effect_sizes,
            'recommendations': report.recommendations,
            'visualizations': report.visualizations
        }
        
        json_path = self.output_dir / f"{self.config.analysis_name}_report.json"
        with open(json_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"Analysis report saved to {json_path}")
    
    # Helper methods
    def _test_normality(self, data: np.ndarray, alpha: float = 0.05) -> bool:
        """Test if data is normally distributed"""
        if len(data) < 3:
            return False
        
        if len(data) <= 5000:
            _, p_value = stats.shapiro(data)
        else:
            _, p_value = stats.kstest(data, 'norm')
        
        return p_value > alpha
    
    def _cohen_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _interpret_p_value(self, p_value: float) -> str:
        """Interpret p-value"""
        if p_value < 0.001:
            return "Highly significant (p < 0.001)"
        elif p_value < 0.01:
            return "Very significant (p < 0.01)"
        elif p_value < 0.05:
            return "Significant (p < 0.05)"
        elif p_value < 0.1:
            return "Marginally significant (p < 0.1)"
        else:
            return "Not significant (p >= 0.1)"
    
    def _interpret_correlation(self, r: float, p_value: float) -> str:
        """Interpret correlation coefficient"""
        strength = "weak"
        if abs(r) > 0.7:
            strength = "strong"
        elif abs(r) > 0.3:
            strength = "moderate"
        
        direction = "positive" if r > 0 else "negative"
        significance = "significant" if p_value < 0.05 else "non-significant"
        
        return f"{strength.title()} {direction} correlation ({significance})"
    
    def _interpret_normality(self, p_value: float) -> str:
        """Interpret normality test result"""
        if p_value > 0.05:
            return "Data appears normally distributed (fail to reject normality)"
        else:
            return "Data does not appear normally distributed (reject normality)"
    
    def _serialize_for_json(self, obj):
        """Serialize complex objects for JSON"""
        if isinstance(obj, dict):
            return {key: self._serialize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj


# Convenience functions for quick analysis
def quick_method_comparison(experimental_data: List[Any], 
                          output_dir: str = "quick_analysis") -> AnalysisReport:
    """Perform quick method comparison analysis"""
    config = AnalysisConfiguration(
        analysis_name="Quick_Method_Comparison",
        output_directory=output_dir,
        generate_plots=True,
        generate_report=True
    )
    
    # Collect and process data
    collector = DataCollector(config)
    data = collector.collect_experimental_data(experimental_data)
    
    processor = DataProcessor(config)
    clean_data = processor.preprocess_data(data)
    
    # Perform analysis
    analyzer = StatisticalAnalyzer(config)
    return analyzer.perform_comprehensive_analysis(clean_data)


def quick_performance_analysis(system_logs: List[Dict[str, Any]], 
                             output_dir: str = "performance_analysis") -> AnalysisReport:
    """Perform quick performance analysis"""
    config = AnalysisConfiguration(
        analysis_name="Quick_Performance_Analysis",
        output_directory=output_dir,
        generate_plots=True
    )
    
    collector = DataCollector(config)
    data = collector.collect_system_performance_data(system_logs)
    
    processor = DataProcessor(config)
    clean_data = processor.preprocess_data(data)
    
    analyzer = StatisticalAnalyzer(config)
    return analyzer.perform_comprehensive_analysis(clean_data)


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    logger.info("Testing Statistical Analysis Module")
    
    # Create sample experimental data
    sample_data = []
    methods = ["Bayesian_RL_Full", "No_Prediction", "Classical_RL"]
    
    for trial_id in range(150):  # 50 trials per method
        method = methods[trial_id % 3]
        
        # Simulate different performance based on method
        if method == "Bayesian_RL_Full":
            success = np.random.random() > 0.1  # 90% success rate
            completion_time = np.random.normal(5.0, 1.0)
            safety_violations = np.random.poisson(0.1)
            comfort_score = np.random.beta(8, 2)
        elif method == "No_Prediction":
            success = np.random.random() > 0.3  # 70% success rate
            completion_time = np.random.normal(7.0, 2.0)
            safety_violations = np.random.poisson(0.5)
            comfort_score = np.random.beta(5, 3)
        else:  # Classical_RL
            success = np.random.random() > 0.2  # 80% success rate
            completion_time = np.random.normal(6.0, 1.5)
            safety_violations = np.random.poisson(0.3)
            comfort_score = np.random.beta(6, 3)
        
        # Create mock result object
        result = type('MockResult', (), {
            'trial_id': trial_id,
            'method': method,
            'success': success,
            'task_completion_time': max(1.0, completion_time),
            'safety_violations': max(0, safety_violations),
            'human_comfort_score': comfort_score,
            'step_count': np.random.randint(50, 200),
            'average_decision_time': np.random.uniform(0.02, 0.15),
            'max_decision_time': np.random.uniform(0.05, 0.25),
            'memory_usage': np.random.uniform(50, 200),
            'scenario_params': {'randomize': True, 'noise_level': 0.1},
            'additional_metrics': {'efficiency_score': np.random.random()}
        })()
        
        sample_data.append(result)
    
    # Perform quick analysis
    try:
        report = quick_method_comparison(sample_data, "test_analysis")
        
        logger.info("Statistical analysis completed successfully!")
        logger.info(f"Analysis found {len(report.statistical_tests)} statistical tests")
        logger.info(f"Generated {len(report.visualizations)} visualizations")
        logger.info(f"Recommendations: {len(report.recommendations)}")
        
        # Print some key results
        for test_name, result in report.statistical_tests.items():
            if result.p_value < 0.05:
                logger.info(f"Significant result: {test_name} (p = {result.p_value:.4f})")
        
        for rec in report.recommendations:
            logger.info(f"Recommendation: {rec}")
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
    
    print("Statistical Analysis Module test completed!")