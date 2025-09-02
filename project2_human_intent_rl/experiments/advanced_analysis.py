"""
Advanced Analysis Tools and Learning Curve Analysis
==================================================

Advanced analytical tools for comprehensive experimental validation:

1. Learning curve analysis with statistical confidence bounds
2. Performance correlation analysis across metrics
3. Behavioral pattern analysis (clustering of successful strategies)
4. Failure mode analysis with root cause identification
5. Uncertainty analysis (when the system is/isn't confident)
6. Publication-quality visualization and reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
from pathlib import Path
import json
import logging
from datetime import datetime
from collections import defaultdict, Counter
import itertools

# Advanced analysis packages
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Some dimensionality reduction features limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive plots will be limited.")

# For change point detection
try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False


@dataclass
class LearningCurveAnalysis:
    """Results from learning curve analysis"""
    method_name: str
    raw_performance: List[float]
    smoothed_curve: List[float]
    confidence_bounds: Tuple[List[float], List[float]]  # (lower, upper)
    convergence_episode: Optional[int] = None
    convergence_value: Optional[float] = None
    convergence_confidence: float = 0.0
    plateau_episodes: Optional[Tuple[int, int]] = None
    learning_rate: Optional[float] = None
    asymptotic_performance: Optional[float] = None
    change_points: List[int] = field(default_factory=list)
    statistical_tests: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrelationAnalysis:
    """Results from correlation analysis between metrics"""
    correlation_matrix: pd.DataFrame
    significant_correlations: List[Tuple[str, str, float, float]]  # (metric1, metric2, r, p)
    factor_loadings: Optional[pd.DataFrame] = None
    explained_variance: Optional[List[float]] = None
    metric_clusters: Optional[Dict[str, List[str]]] = None


@dataclass
class BehavioralPattern:
    """Identified behavioral pattern"""
    pattern_id: str
    cluster_label: int
    success_rate: float
    episodes: List[int]
    characteristic_features: Dict[str, float]
    trajectory_signature: Optional[np.ndarray] = None
    temporal_pattern: Optional[List[float]] = None
    frequency: float = 0.0
    description: str = ""


@dataclass
class FailureMode:
    """Identified failure mode"""
    failure_id: str
    failure_type: str
    frequency: float
    episodes: List[int]
    root_causes: List[str]
    failure_signature: Dict[str, float]
    precondition_patterns: List[Dict[str, Any]]
    recovery_patterns: List[Dict[str, Any]]
    severity_score: float
    prevention_strategies: List[str] = field(default_factory=list)


@dataclass
class UncertaintyAnalysis:
    """Results from uncertainty analysis"""
    confidence_calibration: Dict[str, float]  # Reliability diagram data
    uncertainty_sources: Dict[str, float]  # Contribution of different sources
    prediction_intervals: Dict[str, Tuple[float, float]]
    overconfidence_regions: List[Tuple[float, float]]  # Performance ranges where overconfident
    underconfidence_regions: List[Tuple[float, float]]  # Performance ranges where underconfident
    optimal_confidence_threshold: float
    uncertainty_vs_performance: List[Tuple[float, float]]  # (uncertainty, performance) pairs


class AdvancedAnalyzer:
    """Main class for advanced experimental analysis"""
    
    def __init__(self, results_dir: str = "advanced_analysis"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Analysis results storage
        self.learning_curves = {}
        self.correlation_analyses = {}
        self.behavioral_patterns = {}
        self.failure_modes = {}
        self.uncertainty_analyses = {}
        
        # Visualization settings
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for advanced analysis"""
        logger = logging.getLogger("advanced_analysis")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler(self.results_dir / "advanced_analysis.log")
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_learning_curves(self, experiment_data: Dict[str, List[float]],
                              confidence_level: float = 0.95) -> Dict[str, LearningCurveAnalysis]:
        """Comprehensive learning curve analysis with statistical confidence bounds"""
        
        learning_curve_results = {}
        
        for method_name, performance_data in experiment_data.items():
            if not performance_data:
                continue
                
            self.logger.info(f"Analyzing learning curve for {method_name}")
            
            # Convert to numpy array
            raw_performance = np.array(performance_data)
            
            # Smooth the curve using moving average and LOWESS
            smoothed_curve = self._smooth_learning_curve(raw_performance)
            
            # Calculate confidence bounds using bootstrap
            confidence_bounds = self._calculate_confidence_bounds(
                raw_performance, confidence_level
            )
            
            # Detect convergence
            convergence_episode, convergence_value, convergence_confidence = self._detect_convergence(
                smoothed_curve
            )
            
            # Detect plateau
            plateau_episodes = self._detect_plateau(smoothed_curve)
            
            # Estimate learning rate
            learning_rate = self._estimate_learning_rate(smoothed_curve)
            
            # Estimate asymptotic performance
            asymptotic_performance = self._estimate_asymptotic_performance(smoothed_curve)
            
            # Detect change points
            change_points = self._detect_change_points(raw_performance)
            
            # Statistical tests for learning
            statistical_tests = self._perform_learning_statistical_tests(raw_performance)
            
            # Create analysis result
            analysis = LearningCurveAnalysis(
                method_name=method_name,
                raw_performance=raw_performance.tolist(),
                smoothed_curve=smoothed_curve.tolist(),
                confidence_bounds=confidence_bounds,
                convergence_episode=convergence_episode,
                convergence_value=convergence_value,
                convergence_confidence=convergence_confidence,
                plateau_episodes=plateau_episodes,
                learning_rate=learning_rate,
                asymptotic_performance=asymptotic_performance,
                change_points=change_points,
                statistical_tests=statistical_tests
            )
            
            learning_curve_results[method_name] = analysis
            
        self.learning_curves.update(learning_curve_results)
        return learning_curve_results
    
    def _smooth_learning_curve(self, performance: np.ndarray, 
                              window_size: Optional[int] = None) -> np.ndarray:
        """Smooth learning curve using multiple methods"""
        
        if window_size is None:
            window_size = max(10, len(performance) // 20)  # 5% of data points
        
        # Moving average smoothing
        if len(performance) > window_size:
            moving_avg = np.convolve(performance, np.ones(window_size)/window_size, mode='same')
        else:
            moving_avg = performance.copy()
        
        # Additional LOWESS smoothing if available
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            
            # Apply LOWESS
            x = np.arange(len(performance))
            lowess_result = lowess(performance, x, frac=0.3, return_sorted=False)
            
            # Combine moving average and LOWESS
            smoothed = 0.7 * moving_avg + 0.3 * lowess_result
            
        except ImportError:
            # Fallback to moving average only
            smoothed = moving_avg
        
        return smoothed
    
    def _calculate_confidence_bounds(self, performance: np.ndarray,
                                   confidence_level: float) -> Tuple[List[float], List[float]]:
        """Calculate confidence bounds using bootstrap"""
        
        n_bootstrap = 1000
        alpha = 1 - confidence_level
        
        # Bootstrap resampling
        bootstrap_curves = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(performance, size=len(performance), replace=True)
            
            # Smooth the bootstrap sample
            smoothed_bootstrap = self._smooth_learning_curve(bootstrap_sample)
            bootstrap_curves.append(smoothed_bootstrap)
        
        bootstrap_curves = np.array(bootstrap_curves)
        
        # Calculate percentile bounds
        lower_bound = np.percentile(bootstrap_curves, 100 * alpha/2, axis=0)
        upper_bound = np.percentile(bootstrap_curves, 100 * (1 - alpha/2), axis=0)
        
        return (lower_bound.tolist(), upper_bound.tolist())
    
    def _detect_convergence(self, smoothed_curve: np.ndarray, 
                          stability_threshold: float = 0.05) -> Tuple[Optional[int], Optional[float], float]:
        """Detect convergence point in learning curve"""
        
        if len(smoothed_curve) < 50:  # Need minimum data points
            return None, None, 0.0
        
        # Look for stability in the last portion of the curve
        min_stable_length = max(20, len(smoothed_curve) // 10)
        
        for start_idx in range(len(smoothed_curve) - min_stable_length):
            segment = smoothed_curve[start_idx:start_idx + min_stable_length]
            
            # Check if segment is stable (low variance relative to mean)
            if len(segment) > 1:
                coefficient_of_variation = np.std(segment) / (np.abs(np.mean(segment)) + 1e-8)
                
                if coefficient_of_variation < stability_threshold:
                    convergence_episode = start_idx
                    convergence_value = np.mean(segment)
                    
                    # Confidence based on how long the stability lasts
                    remaining_length = len(smoothed_curve) - start_idx
                    confidence = min(1.0, remaining_length / min_stable_length)
                    
                    return convergence_episode, convergence_value, confidence
        
        return None, None, 0.0
    
    def _detect_plateau(self, smoothed_curve: np.ndarray,
                       plateau_threshold: float = 0.02) -> Optional[Tuple[int, int]]:
        """Detect plateau regions in learning curve"""
        
        if len(smoothed_curve) < 30:
            return None
        
        # Calculate moving derivatives
        derivatives = np.gradient(smoothed_curve)
        
        # Find regions with low derivative magnitude
        low_derivative_mask = np.abs(derivatives) < plateau_threshold
        
        # Find contiguous plateau regions
        plateau_regions = []
        start_idx = None
        
        for i, is_plateau in enumerate(low_derivative_mask):
            if is_plateau and start_idx is None:
                start_idx = i
            elif not is_plateau and start_idx is not None:
                if i - start_idx > 10:  # Minimum plateau length
                    plateau_regions.append((start_idx, i-1))
                start_idx = None
        
        # Handle case where plateau extends to end
        if start_idx is not None and len(smoothed_curve) - start_idx > 10:
            plateau_regions.append((start_idx, len(smoothed_curve)-1))
        
        # Return the longest plateau
        if plateau_regions:
            longest_plateau = max(plateau_regions, key=lambda x: x[1] - x[0])
            return longest_plateau
        
        return None
    
    def _estimate_learning_rate(self, smoothed_curve: np.ndarray) -> Optional[float]:
        """Estimate learning rate from curve steepness"""
        
        if len(smoothed_curve) < 10:
            return None
        
        # Use first quarter of data to estimate initial learning rate
        initial_segment_length = max(10, len(smoothed_curve) // 4)
        initial_segment = smoothed_curve[:initial_segment_length]
        
        # Fit exponential curve: y = a * exp(-b * x) + c
        def exp_model(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        try:
            x_data = np.arange(len(initial_segment))
            
            # Initial parameter guess
            p0 = [initial_segment[0] - initial_segment[-1], 0.1, initial_segment[-1]]
            
            popt, _ = curve_fit(exp_model, x_data, initial_segment, p0=p0, maxfev=1000)
            
            # Learning rate is the decay parameter
            learning_rate = popt[1]
            
            return max(0.0, learning_rate)  # Ensure non-negative
            
        except Exception:
            # Fallback: simple slope calculation
            if len(initial_segment) > 1:
                slope = (initial_segment[-1] - initial_segment[0]) / (len(initial_segment) - 1)
                return abs(slope)
            
        return None
    
    def _estimate_asymptotic_performance(self, smoothed_curve: np.ndarray) -> Optional[float]:
        """Estimate asymptotic performance level"""
        
        if len(smoothed_curve) < 20:
            return None
        
        # Use last 25% of data to estimate asymptotic performance
        final_segment_length = max(10, len(smoothed_curve) // 4)
        final_segment = smoothed_curve[-final_segment_length:]
        
        # Return median of final segment (robust to outliers)
        return float(np.median(final_segment))
    
    def _detect_change_points(self, performance: np.ndarray) -> List[int]:
        """Detect change points in learning curve"""
        
        if not RUPTURES_AVAILABLE or len(performance) < 20:
            return []
        
        try:
            # Use Pelt algorithm for change point detection
            algo = rpt.Pelt(model="rbf").fit(performance.reshape(-1, 1))
            change_points = algo.predict(pen=10)
            
            # Remove the last point (end of data)
            if change_points and change_points[-1] == len(performance):
                change_points = change_points[:-1]
            
            return change_points
            
        except Exception:
            return []
    
    def _perform_learning_statistical_tests(self, performance: np.ndarray) -> Dict[str, Any]:
        """Perform statistical tests on learning curve"""
        
        tests = {}
        
        if len(performance) < 10:
            return tests
        
        # Test for trend (Mann-Kendall test)
        try:
            # Simplified Mann-Kendall test
            n = len(performance)
            s = 0
            
            for i in range(n-1):
                for j in range(i+1, n):
                    if performance[j] > performance[i]:
                        s += 1
                    elif performance[j] < performance[i]:
                        s -= 1
            
            # Calculate variance
            var_s = n * (n - 1) * (2 * n + 5) / 18
            
            # Calculate z-score
            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / np.sqrt(var_s)
            else:
                z = 0
            
            # Calculate p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            
            tests['mann_kendall'] = {
                'statistic': s,
                'z_score': z,
                'p_value': p_value,
                'trend': 'increasing' if s > 0 else 'decreasing' if s < 0 else 'no trend'
            }
            
        except Exception as e:
            self.logger.warning(f"Mann-Kendall test failed: {e}")
        
        # Test for stationarity (augmented Dickey-Fuller test)
        try:
            from statsmodels.tsa.stattools import adfuller
            
            adf_result = adfuller(performance)
            tests['adf_stationarity'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            }
            
        except ImportError:
            pass
        except Exception as e:
            self.logger.warning(f"ADF test failed: {e}")
        
        # Test for normality of residuals (if we can fit a trend)
        try:
            x = np.arange(len(performance))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, performance)
            
            predicted = slope * x + intercept
            residuals = performance - predicted
            
            # Shapiro-Wilk test for normality
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            
            tests['residual_normality'] = {
                'shapiro_statistic': shapiro_stat,
                'shapiro_p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05,
                'trend_r_squared': r_value**2
            }
            
        except Exception as e:
            self.logger.warning(f"Residual normality test failed: {e}")
        
        return tests
    
    def analyze_performance_correlations(self, metrics_data: Dict[str, Dict[str, List[float]]],
                                       method: str = 'pearson') -> CorrelationAnalysis:
        """Analyze correlations between performance metrics"""
        
        self.logger.info("Analyzing performance correlations")
        
        # Prepare data for correlation analysis
        all_metrics = {}
        
        # Flatten metrics data
        for method_name, method_metrics in metrics_data.items():
            for metric_name, metric_values in method_metrics.items():
                key = f"{method_name}_{metric_name}"
                all_metrics[key] = metric_values
        
        # Create DataFrame
        # Find the minimum length across all metrics
        min_length = min(len(values) for values in all_metrics.values())
        
        # Truncate all metrics to the same length
        truncated_metrics = {
            key: values[:min_length] for key, values in all_metrics.items()
        }
        
        df = pd.DataFrame(truncated_metrics)
        
        # Calculate correlation matrix
        correlation_matrix = df.corr(method=method)
        
        # Find significant correlations
        significant_correlations = []
        n = len(df)
        
        for i, metric1 in enumerate(correlation_matrix.columns):
            for j, metric2 in enumerate(correlation_matrix.columns):
                if i < j:  # Only upper triangle
                    r = correlation_matrix.iloc[i, j]
                    
                    if not np.isnan(r):
                        # Calculate p-value
                        if method == 'pearson':
                            t_stat = r * np.sqrt((n - 2) / (1 - r**2))
                            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                        else:
                            # For non-parametric correlations, use approximation
                            z = r * np.sqrt(n - 3)
                            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
                        
                        if p_value < 0.05 and abs(r) > 0.3:  # Significant and meaningful
                            significant_correlations.append((metric1, metric2, r, p_value))
        
        # Principal Component Analysis for factor loadings
        factor_loadings = None
        explained_variance = None
        
        try:
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df.fillna(0))
            
            # Perform PCA
            pca = PCA()
            pca.fit(scaled_data)
            
            # Create factor loadings DataFrame
            n_components = min(5, len(df.columns))  # Top 5 components
            factor_loadings = pd.DataFrame(
                pca.components_[:n_components].T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=df.columns
            )
            
            explained_variance = pca.explained_variance_ratio_[:n_components].tolist()
            
        except Exception as e:
            self.logger.warning(f"PCA analysis failed: {e}")
        
        # Metric clustering based on correlations
        metric_clusters = self._cluster_metrics(correlation_matrix)
        
        result = CorrelationAnalysis(
            correlation_matrix=correlation_matrix,
            significant_correlations=significant_correlations,
            factor_loadings=factor_loadings,
            explained_variance=explained_variance,
            metric_clusters=metric_clusters
        )
        
        self.correlation_analyses['performance_metrics'] = result
        return result
    
    def _cluster_metrics(self, correlation_matrix: pd.DataFrame) -> Dict[str, List[str]]:
        """Cluster metrics based on correlation patterns"""
        
        try:
            # Convert correlation to distance matrix
            distance_matrix = 1 - np.abs(correlation_matrix.fillna(0))
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(distance_matrix.values, method='ward')
            
            # Determine optimal number of clusters using silhouette score
            best_n_clusters = 2
            best_score = -1
            
            for n_clusters in range(2, min(8, len(correlation_matrix.columns))):
                from sklearn.cluster import AgglomerativeClustering
                
                clustering = AgglomerativeClustering(n_clusters=n_clusters)
                cluster_labels = clustering.fit_predict(distance_matrix.values)
                
                if len(set(cluster_labels)) > 1:  # More than one cluster
                    score = silhouette_score(distance_matrix.values, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
            
            # Final clustering
            clustering = AgglomerativeClustering(n_clusters=best_n_clusters)
            cluster_labels = clustering.fit_predict(distance_matrix.values)
            
            # Group metrics by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                cluster_name = f"Cluster_{label}"
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(correlation_matrix.columns[i])
            
            return clusters
            
        except Exception as e:
            self.logger.warning(f"Metric clustering failed: {e}")
            return {}
    
    def analyze_behavioral_patterns(self, trajectory_data: Dict[str, List[List[Tuple[float, float, float]]]],
                                  success_data: Dict[str, List[bool]]) -> Dict[str, List[BehavioralPattern]]:
        """Analyze behavioral patterns in trajectory data"""
        
        self.logger.info("Analyzing behavioral patterns")
        
        behavioral_patterns = {}
        
        for method_name in trajectory_data.keys():
            if method_name not in success_data:
                continue
            
            trajectories = trajectory_data[method_name]
            successes = success_data[method_name]
            
            if not trajectories or not successes:
                continue
            
            # Extract features from trajectories
            trajectory_features = self._extract_trajectory_features(trajectories)
            
            if trajectory_features is None or len(trajectory_features) == 0:
                continue
            
            # Perform clustering to identify behavioral patterns
            patterns = self._cluster_behavioral_patterns(
                trajectory_features, successes, method_name
            )
            
            behavioral_patterns[method_name] = patterns
        
        self.behavioral_patterns.update(behavioral_patterns)
        return behavioral_patterns
    
    def _extract_trajectory_features(self, trajectories: List[List[Tuple[float, float, float]]]) -> Optional[np.ndarray]:
        """Extract features from trajectory data"""
        
        features_list = []
        
        for trajectory in trajectories:
            if len(trajectory) < 2:
                continue
            
            # Convert to numpy array
            traj_array = np.array([(t[1], t[2], t[3]) for t in trajectory])  # (x, y, z) positions
            
            # Extract various features
            features = {}
            
            # Path length
            path_segments = np.diff(traj_array, axis=0)
            path_length = np.sum(np.linalg.norm(path_segments, axis=1))
            features['path_length'] = path_length
            
            # Straightness (ratio of direct distance to path length)
            direct_distance = np.linalg.norm(traj_array[-1] - traj_array[0])
            features['straightness'] = direct_distance / (path_length + 1e-8)
            
            # Average velocity
            time_diffs = np.diff([t[0] for t in trajectory])
            if len(time_diffs) > 0 and np.sum(time_diffs) > 0:
                features['avg_velocity'] = path_length / np.sum(time_diffs)
            else:
                features['avg_velocity'] = 0.0
            
            # Velocity variability
            velocities = np.linalg.norm(path_segments, axis=1) / (time_diffs + 1e-8)
            features['velocity_variability'] = np.std(velocities) if len(velocities) > 1 else 0.0
            
            # Smoothness (jerk)
            if len(traj_array) > 2:
                accelerations = np.diff(path_segments, axis=0)
                jerks = np.diff(accelerations, axis=0)
                features['smoothness'] = 1.0 / (1.0 + np.mean(np.linalg.norm(jerks, axis=1)))
            else:
                features['smoothness'] = 1.0
            
            # Trajectory duration
            features['duration'] = trajectory[-1][0] - trajectory[0][0]
            
            # Maximum distance from start
            distances_from_start = np.linalg.norm(traj_array - traj_array[0], axis=1)
            features['max_distance_from_start'] = np.max(distances_from_start)
            
            # Trajectory complexity (number of direction changes)
            if len(path_segments) > 1:
                direction_changes = 0
                for i in range(1, len(path_segments)):
                    dot_product = np.dot(path_segments[i-1], path_segments[i])
                    norm_product = (np.linalg.norm(path_segments[i-1]) * 
                                   np.linalg.norm(path_segments[i]))
                    
                    if norm_product > 1e-8:
                        cos_angle = np.clip(dot_product / norm_product, -1, 1)
                        angle = np.arccos(cos_angle)
                        
                        if angle > np.pi / 4:  # 45 degree threshold
                            direction_changes += 1
                
                features['direction_changes'] = direction_changes
            else:
                features['direction_changes'] = 0
            
            features_list.append(list(features.values()))
        
        if not features_list:
            return None
        
        return np.array(features_list)
    
    def _cluster_behavioral_patterns(self, trajectory_features: np.ndarray,
                                   successes: List[bool], method_name: str) -> List[BehavioralPattern]:
        """Cluster trajectory features to identify behavioral patterns"""
        
        patterns = []
        
        try:
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(trajectory_features)
            
            # Determine optimal number of clusters
            max_clusters = min(8, len(scaled_features) // 10)  # At least 10 samples per cluster
            
            if max_clusters < 2:
                return patterns
            
            best_n_clusters = 2
            best_score = -1
            
            for n_clusters in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scaled_features)
                
                if len(set(cluster_labels)) > 1:
                    score = silhouette_score(scaled_features, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
            
            # Final clustering
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            # Analyze each cluster
            feature_names = [
                'path_length', 'straightness', 'avg_velocity', 'velocity_variability',
                'smoothness', 'duration', 'max_distance_from_start', 'direction_changes'
            ]
            
            for cluster_id in range(best_n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_episodes = np.where(cluster_mask)[0].tolist()
                
                if len(cluster_episodes) == 0:
                    continue
                
                # Calculate success rate for this cluster
                cluster_successes = [successes[i] for i in cluster_episodes]
                success_rate = np.mean(cluster_successes)
                
                # Characteristic features (cluster center)
                cluster_center = kmeans.cluster_centers_[cluster_id]
                characteristic_features = dict(zip(feature_names, 
                                                 scaler.inverse_transform([cluster_center])[0]))
                
                # Create pattern description
                description = self._generate_pattern_description(characteristic_features, success_rate)
                
                pattern = BehavioralPattern(
                    pattern_id=f"{method_name}_pattern_{cluster_id}",
                    cluster_label=cluster_id,
                    success_rate=success_rate,
                    episodes=cluster_episodes,
                    characteristic_features=characteristic_features,
                    frequency=len(cluster_episodes) / len(trajectory_features),
                    description=description
                )
                
                patterns.append(pattern)
            
        except Exception as e:
            self.logger.warning(f"Behavioral pattern clustering failed for {method_name}: {e}")
        
        return patterns
    
    def _generate_pattern_description(self, features: Dict[str, float], success_rate: float) -> str:
        """Generate human-readable description of behavioral pattern"""
        
        descriptions = []
        
        # Velocity characteristics
        if features['avg_velocity'] > 0.5:
            descriptions.append("fast movement")
        elif features['avg_velocity'] < 0.2:
            descriptions.append("slow movement")
        else:
            descriptions.append("moderate movement")
        
        # Path characteristics
        if features['straightness'] > 0.8:
            descriptions.append("direct path")
        elif features['straightness'] < 0.5:
            descriptions.append("indirect path")
        
        # Smoothness
        if features['smoothness'] > 0.8:
            descriptions.append("smooth execution")
        elif features['smoothness'] < 0.5:
            descriptions.append("jerky execution")
        
        # Success assessment
        if success_rate > 0.8:
            descriptions.append("highly successful")
        elif success_rate < 0.5:
            descriptions.append("low success rate")
        
        return ", ".join(descriptions)
    
    def analyze_failure_modes(self, experiment_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[FailureMode]]:
        """Analyze failure modes and root causes"""
        
        self.logger.info("Analyzing failure modes")
        
        failure_modes = {}
        
        for method_name, results_list in experiment_results.items():
            
            # Extract failed episodes
            failed_episodes = []
            for i, result in enumerate(results_list):
                if not result.get('success', True):  # Assuming success field exists
                    failure_data = {
                        'episode': i,
                        'failure_reason': result.get('failure_reason', 'unknown'),
                        'metrics': result.get('metrics', {}),
                        'trajectory': result.get('trajectory_data', {}),
                        'events': result.get('event_log', [])
                    }
                    failed_episodes.append(failure_data)
            
            if not failed_episodes:
                failure_modes[method_name] = []
                continue
            
            # Analyze failure patterns
            method_failure_modes = self._identify_failure_modes(failed_episodes, method_name)
            failure_modes[method_name] = method_failure_modes
        
        self.failure_modes.update(failure_modes)
        return failure_modes
    
    def _identify_failure_modes(self, failed_episodes: List[Dict[str, Any]], 
                               method_name: str) -> List[FailureMode]:
        """Identify distinct failure modes from failed episodes"""
        
        failure_modes = []
        
        try:
            # Group failures by reason
            failure_groups = defaultdict(list)
            for episode in failed_episodes:
                reason = episode['failure_reason']
                failure_groups[reason].append(episode)
            
            # Analyze each failure type
            for failure_type, episodes in failure_groups.items():
                if len(episodes) < 2:  # Need multiple instances to identify pattern
                    continue
                
                # Extract common characteristics
                failure_signature = self._extract_failure_signature(episodes)
                
                # Identify root causes
                root_causes = self._identify_root_causes(episodes)
                
                # Analyze preconditions
                precondition_patterns = self._analyze_failure_preconditions(episodes)
                
                # Analyze recovery patterns
                recovery_patterns = self._analyze_recovery_patterns(episodes)
                
                # Calculate severity
                severity_score = self._calculate_failure_severity(episodes)
                
                # Generate prevention strategies
                prevention_strategies = self._generate_prevention_strategies(
                    failure_type, root_causes, failure_signature
                )
                
                failure_mode = FailureMode(
                    failure_id=f"{method_name}_{failure_type}",
                    failure_type=failure_type,
                    frequency=len(episodes) / len(failed_episodes),
                    episodes=[ep['episode'] for ep in episodes],
                    root_causes=root_causes,
                    failure_signature=failure_signature,
                    precondition_patterns=precondition_patterns,
                    recovery_patterns=recovery_patterns,
                    severity_score=severity_score,
                    prevention_strategies=prevention_strategies
                )
                
                failure_modes.append(failure_mode)
                
        except Exception as e:
            self.logger.warning(f"Failure mode identification failed for {method_name}: {e}")
        
        return failure_modes
    
    def _extract_failure_signature(self, episodes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract common characteristics of failure episodes"""
        
        signature = {}
        
        # Analyze metrics at failure
        all_metrics = defaultdict(list)
        
        for episode in episodes:
            metrics = episode.get('metrics', {})
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    all_metrics[metric_name].append(metric_value)
        
        # Calculate statistics for each metric
        for metric_name, values in all_metrics.items():
            if len(values) > 1:
                signature[f"{metric_name}_mean"] = np.mean(values)
                signature[f"{metric_name}_std"] = np.std(values)
                signature[f"{metric_name}_median"] = np.median(values)
        
        return signature
    
    def _identify_root_causes(self, episodes: List[Dict[str, Any]]) -> List[str]:
        """Identify root causes from failure episodes"""
        
        root_causes = []
        
        # Analyze event logs for common patterns
        all_events = []
        for episode in episodes:
            events = episode.get('events', [])
            all_events.extend([event.get('event_type', 'unknown') for event in events])
        
        # Find common problematic events
        event_counter = Counter(all_events)
        for event_type, count in event_counter.most_common():
            if count >= len(episodes) * 0.5:  # Present in at least 50% of failures
                if 'error' in event_type.lower() or 'fail' in event_type.lower():
                    root_causes.append(f"Frequent {event_type} events")
        
        # Analyze trajectory patterns
        trajectory_issues = self._analyze_trajectory_issues(episodes)
        root_causes.extend(trajectory_issues)
        
        # Analyze metric patterns
        metric_issues = self._analyze_metric_issues(episodes)
        root_causes.extend(metric_issues)
        
        return root_causes[:5]  # Limit to top 5 root causes
    
    def _analyze_trajectory_issues(self, episodes: List[Dict[str, Any]]) -> List[str]:
        """Analyze trajectory data for common issues"""
        
        issues = []
        
        trajectories = []
        for episode in episodes:
            traj_data = episode.get('trajectory', {})
            if traj_data:
                # Assume trajectory data is in the form we've seen before
                for phase_name, phase_traj in traj_data.items():
                    if isinstance(phase_traj, list) and len(phase_traj) > 1:
                        trajectories.append(phase_traj)
        
        if not trajectories:
            return issues
        
        # Analyze for common trajectory problems
        short_trajectories = 0
        erratic_trajectories = 0
        
        for traj in trajectories:
            # Check trajectory length
            if len(traj) < 5:
                short_trajectories += 1
            
            # Check for erratic movement
            if len(traj) > 2:
                positions = np.array([(t[1], t[2], t[3]) for t in traj])
                velocities = np.diff(positions, axis=0)
                
                if len(velocities) > 1:
                    velocity_magnitudes = np.linalg.norm(velocities, axis=1)
                    if np.std(velocity_magnitudes) > np.mean(velocity_magnitudes):
                        erratic_trajectories += 1
        
        if short_trajectories > len(trajectories) * 0.5:
            issues.append("Premature termination")
        
        if erratic_trajectories > len(trajectories) * 0.5:
            issues.append("Erratic movement patterns")
        
        return issues
    
    def _analyze_metric_issues(self, episodes: List[Dict[str, Any]]) -> List[str]:
        """Analyze metrics for common issues"""
        
        issues = []
        
        # Collect all metrics
        all_metrics = defaultdict(list)
        for episode in episodes:
            metrics = episode.get('metrics', {})
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    all_metrics[metric_name].append(metric_value)
        
        # Check for metric anomalies
        for metric_name, values in all_metrics.items():
            if len(values) > 2:
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Check for consistently low performance metrics
                if 'success' in metric_name.lower() or 'accuracy' in metric_name.lower():
                    if mean_val < 0.3:  # Low success/accuracy
                        issues.append(f"Low {metric_name}")
                
                # Check for high error metrics
                if 'error' in metric_name.lower() or 'violation' in metric_name.lower():
                    if mean_val > 0.5:  # High error rate
                        issues.append(f"High {metric_name}")
        
        return issues
    
    def _analyze_failure_preconditions(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze conditions that precede failures"""
        
        # This would analyze the state/conditions just before failure
        # For now, return placeholder
        return [{'type': 'analysis_placeholder', 'description': 'Precondition analysis not implemented'}]
    
    def _analyze_recovery_patterns(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze recovery attempts and patterns"""
        
        # This would analyze any recovery attempts made after detecting problems
        # For now, return placeholder
        return [{'type': 'recovery_placeholder', 'description': 'Recovery analysis not implemented'}]
    
    def _calculate_failure_severity(self, episodes: List[Dict[str, Any]]) -> float:
        """Calculate severity score for failure mode"""
        
        # Consider frequency and impact
        frequency_score = len(episodes) / 100  # Normalize by typical episode count
        
        # Look for safety-related failures (higher severity)
        safety_related = any(
            'safety' in episode.get('failure_reason', '').lower() or
            'collision' in episode.get('failure_reason', '').lower()
            for episode in episodes
        )
        
        severity = frequency_score * (2.0 if safety_related else 1.0)
        return min(1.0, severity)  # Cap at 1.0
    
    def _generate_prevention_strategies(self, failure_type: str, root_causes: List[str],
                                      failure_signature: Dict[str, float]) -> List[str]:
        """Generate prevention strategies based on failure analysis"""
        
        strategies = []
        
        # Generic strategies based on failure type
        if 'timeout' in failure_type.lower():
            strategies.append("Increase timeout limits or improve algorithm efficiency")
        
        if 'safety' in failure_type.lower():
            strategies.append("Implement more conservative safety margins")
            strategies.append("Add additional sensor verification")
        
        if 'sensor' in failure_type.lower():
            strategies.append("Implement sensor redundancy")
            strategies.append("Add sensor health monitoring")
        
        # Strategies based on root causes
        for cause in root_causes:
            if 'erratic' in cause.lower():
                strategies.append("Add motion smoothing filters")
            
            if 'premature' in cause.lower():
                strategies.append("Implement completion verification")
        
        return strategies[:3]  # Limit to top 3 strategies
    
    def analyze_uncertainty_calibration(self, predictions: Dict[str, List[Tuple[float, float]]],
                                      ground_truth: Dict[str, List[bool]]) -> Dict[str, UncertaintyAnalysis]:
        """Analyze uncertainty calibration and confidence"""
        
        self.logger.info("Analyzing uncertainty calibration")
        
        uncertainty_analyses = {}
        
        for method_name in predictions.keys():
            if method_name not in ground_truth:
                continue
            
            pred_data = predictions[method_name]  # List of (prediction, confidence) tuples
            true_data = ground_truth[method_name]  # List of actual outcomes
            
            if not pred_data or not true_data or len(pred_data) != len(true_data):
                continue
            
            analysis = self._perform_uncertainty_analysis(pred_data, true_data, method_name)
            uncertainty_analyses[method_name] = analysis
        
        self.uncertainty_analyses.update(uncertainty_analyses)
        return uncertainty_analyses
    
    def _perform_uncertainty_analysis(self, predictions: List[Tuple[float, float]],
                                    ground_truth: List[bool], method_name: str) -> UncertaintyAnalysis:
        """Perform uncertainty analysis for a single method"""
        
        # Extract predictions and confidences
        pred_values = [p[0] for p in predictions]
        confidences = [p[1] for p in predictions]
        
        # Reliability diagram (calibration)
        confidence_calibration = self._calculate_reliability_diagram(
            pred_values, confidences, ground_truth
        )
        
        # Identify uncertainty sources
        uncertainty_sources = self._analyze_uncertainty_sources(
            pred_values, confidences, ground_truth
        )
        
        # Calculate prediction intervals
        prediction_intervals = self._calculate_prediction_intervals(
            pred_values, confidences, ground_truth
        )
        
        # Identify over/under-confidence regions
        overconfidence_regions, underconfidence_regions = self._identify_confidence_regions(
            pred_values, confidences, ground_truth
        )
        
        # Find optimal confidence threshold
        optimal_threshold = self._find_optimal_confidence_threshold(
            pred_values, confidences, ground_truth
        )
        
        # Uncertainty vs performance relationship
        uncertainty_vs_performance = list(zip(
            [1 - c for c in confidences],  # Convert confidence to uncertainty
            [1.0 if gt else 0.0 for gt in ground_truth]  # Binary performance
        ))
        
        return UncertaintyAnalysis(
            confidence_calibration=confidence_calibration,
            uncertainty_sources=uncertainty_sources,
            prediction_intervals=prediction_intervals,
            overconfidence_regions=overconfidence_regions,
            underconfidence_regions=underconfidence_regions,
            optimal_confidence_threshold=optimal_threshold,
            uncertainty_vs_performance=uncertainty_vs_performance
        )
    
    def _calculate_reliability_diagram(self, predictions: List[float],
                                     confidences: List[float],
                                     ground_truth: List[bool]) -> Dict[str, float]:
        """Calculate reliability diagram data for calibration analysis"""
        
        # Bin confidences
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        calibration_data = {}
        
        for i in range(n_bins):
            bin_lower = bin_edges[i]
            bin_upper = bin_edges[i + 1]
            
            # Find predictions in this confidence bin
            in_bin = [(bin_lower <= c < bin_upper) for c in confidences]
            
            if i == n_bins - 1:  # Include upper bound for last bin
                in_bin = [(bin_lower <= c <= bin_upper) for c in confidences]
            
            bin_indices = [j for j, in_b in enumerate(in_bin) if in_b]
            
            if not bin_indices:
                continue
            
            # Calculate bin statistics
            bin_confidence = np.mean([confidences[j] for j in bin_indices])
            bin_accuracy = np.mean([ground_truth[j] for j in bin_indices])
            bin_count = len(bin_indices)
            
            bin_name = f"bin_{i}"
            calibration_data[f"{bin_name}_confidence"] = bin_confidence
            calibration_data[f"{bin_name}_accuracy"] = bin_accuracy
            calibration_data[f"{bin_name}_count"] = bin_count
            calibration_data[f"{bin_name}_calibration_error"] = abs(bin_confidence - bin_accuracy)
        
        # Overall calibration metrics
        if calibration_data:
            # Expected Calibration Error
            total_samples = sum([v for k, v in calibration_data.items() if k.endswith('_count')])
            ece = 0.0
            
            for i in range(n_bins):
                bin_name = f"bin_{i}"
                if f"{bin_name}_count" in calibration_data:
                    bin_weight = calibration_data[f"{bin_name}_count"] / total_samples
                    bin_error = calibration_data[f"{bin_name}_calibration_error"]
                    ece += bin_weight * bin_error
            
            calibration_data['expected_calibration_error'] = ece
        
        return calibration_data
    
    def _analyze_uncertainty_sources(self, predictions: List[float],
                                   confidences: List[float],
                                   ground_truth: List[bool]) -> Dict[str, float]:
        """Analyze sources of uncertainty"""
        
        sources = {}
        
        # Aleatoric uncertainty (data noise)
        # Estimated from prediction variance in similar regions
        pred_variance = np.var(predictions)
        sources['aleatoric'] = pred_variance
        
        # Epistemic uncertainty (model uncertainty)
        # Estimated from confidence variance
        confidence_variance = np.var(confidences)
        sources['epistemic'] = confidence_variance
        
        # Normalize to sum to 1
        total_uncertainty = pred_variance + confidence_variance
        if total_uncertainty > 0:
            sources['aleatoric'] = pred_variance / total_uncertainty
            sources['epistemic'] = confidence_variance / total_uncertainty
        
        return sources
    
    def _calculate_prediction_intervals(self, predictions: List[float],
                                      confidences: List[float],
                                      ground_truth: List[bool]) -> Dict[str, Tuple[float, float]]:
        """Calculate prediction intervals"""
        
        intervals = {}
        
        # Overall prediction interval
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        
        # 95% prediction interval
        intervals['95_percent'] = (
            pred_mean - 1.96 * pred_std,
            pred_mean + 1.96 * pred_std
        )
        
        # Confidence-weighted interval
        if confidences:
            weighted_mean = np.average(predictions, weights=confidences)
            weighted_std = np.sqrt(np.average((np.array(predictions) - weighted_mean)**2, weights=confidences))
            
            intervals['confidence_weighted'] = (
                weighted_mean - 1.96 * weighted_std,
                weighted_mean + 1.96 * weighted_std
            )
        
        return intervals
    
    def _identify_confidence_regions(self, predictions: List[float],
                                   confidences: List[float],
                                   ground_truth: List[bool]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Identify over and under-confidence regions"""
        
        # Sort by predictions to analyze regions
        sorted_indices = np.argsort(predictions)
        
        window_size = max(10, len(predictions) // 10)
        overconfidence_regions = []
        underconfidence_regions = []
        
        for i in range(0, len(predictions) - window_size, window_size // 2):
            window_indices = sorted_indices[i:i + window_size]
            
            window_confidences = [confidences[j] for j in window_indices]
            window_accuracy = [ground_truth[j] for j in window_indices]
            window_predictions = [predictions[j] for j in window_indices]
            
            avg_confidence = np.mean(window_confidences)
            avg_accuracy = np.mean(window_accuracy)
            
            pred_range = (min(window_predictions), max(window_predictions))
            
            # Overconfidence: confidence > accuracy
            if avg_confidence - avg_accuracy > 0.1:
                overconfidence_regions.append(pred_range)
            
            # Underconfidence: accuracy > confidence  
            elif avg_accuracy - avg_confidence > 0.1:
                underconfidence_regions.append(pred_range)
        
        return overconfidence_regions, underconfidence_regions
    
    def _find_optimal_confidence_threshold(self, predictions: List[float],
                                         confidences: List[float],
                                         ground_truth: List[bool]) -> float:
        """Find optimal confidence threshold for decision making"""
        
        thresholds = np.linspace(0.1, 0.9, 17)  # Test different thresholds
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in thresholds:
            # Make decisions based on threshold
            decisions = [c >= threshold for c in confidences]
            
            # Calculate performance (e.g., accuracy of high-confidence decisions)
            high_conf_indices = [i for i, d in enumerate(decisions) if d]
            
            if high_conf_indices:
                high_conf_accuracy = np.mean([ground_truth[i] for i in high_conf_indices])
                coverage = len(high_conf_indices) / len(predictions)
                
                # Score balances accuracy and coverage
                score = high_conf_accuracy * coverage
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
        
        return best_threshold
    
    def generate_publication_plots(self, output_dir: Optional[str] = None) -> List[str]:
        """Generate publication-quality plots"""
        
        if output_dir is None:
            output_dir = self.results_dir / "publication_plots"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        generated_plots = []
        
        # Learning curve plots
        if self.learning_curves:
            plot_path = self._create_learning_curves_plot(output_dir)
            if plot_path:
                generated_plots.append(plot_path)
        
        # Correlation heatmap
        if self.correlation_analyses:
            plot_path = self._create_correlation_heatmap(output_dir)
            if plot_path:
                generated_plots.append(plot_path)
        
        # Behavioral patterns plot
        if self.behavioral_patterns:
            plot_path = self._create_behavioral_patterns_plot(output_dir)
            if plot_path:
                generated_plots.append(plot_path)
        
        # Failure analysis plot
        if self.failure_modes:
            plot_path = self._create_failure_analysis_plot(output_dir)
            if plot_path:
                generated_plots.append(plot_path)
        
        # Uncertainty calibration plot
        if self.uncertainty_analyses:
            plot_path = self._create_uncertainty_calibration_plot(output_dir)
            if plot_path:
                generated_plots.append(plot_path)
        
        self.logger.info(f"Generated {len(generated_plots)} publication plots")
        return generated_plots
    
    def _create_learning_curves_plot(self, output_dir: Path) -> Optional[str]:
        """Create publication-quality learning curves plot"""
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Learning curves with confidence intervals
            ax1 = axes[0]
            
            for method_name, analysis in self.learning_curves.items():
                episodes = np.arange(len(analysis.smoothed_curve))
                
                # Plot main curve
                ax1.plot(episodes, analysis.smoothed_curve, label=method_name, linewidth=2)
                
                # Plot confidence bounds
                if analysis.confidence_bounds:
                    lower_bound, upper_bound = analysis.confidence_bounds
                    ax1.fill_between(episodes, lower_bound, upper_bound, alpha=0.2)
                
                # Mark convergence point
                if analysis.convergence_episode is not None:
                    ax1.axvline(x=analysis.convergence_episode, linestyle='--', alpha=0.7)
            
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Performance')
            ax1.set_title('Learning Curves with 95% Confidence Intervals')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Convergence comparison
            ax2 = axes[1]
            
            methods = list(self.learning_curves.keys())
            convergence_episodes = [
                self.learning_curves[m].convergence_episode or 0 
                for m in methods
            ]
            final_performance = [
                self.learning_curves[m].asymptotic_performance or 0
                for m in methods
            ]
            
            scatter = ax2.scatter(convergence_episodes, final_performance, 
                                s=100, alpha=0.7, c=range(len(methods)))
            
            for i, method in enumerate(methods):
                ax2.annotate(method, (convergence_episodes[i], final_performance[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax2.set_xlabel('Convergence Episode')
            ax2.set_ylabel('Final Performance')
            ax2.set_title('Convergence Speed vs Final Performance')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = output_dir / "learning_curves.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create learning curves plot: {e}")
            return None
    
    def _create_correlation_heatmap(self, output_dir: Path) -> Optional[str]:
        """Create correlation heatmap"""
        
        try:
            if 'performance_metrics' not in self.correlation_analyses:
                return None
            
            correlation_matrix = self.correlation_analyses['performance_metrics'].correlation_matrix
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create heatmap
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r',
                       center=0, square=True, fmt='.2f', cbar_kws={"shrink": 0.8})
            
            plt.title('Performance Metrics Correlation Matrix')
            plt.tight_layout()
            
            plot_path = output_dir / "correlation_heatmap.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create correlation heatmap: {e}")
            return None
    
    def _create_behavioral_patterns_plot(self, output_dir: Path) -> Optional[str]:
        """Create behavioral patterns visualization"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot behavioral patterns for first method
            method_name = list(self.behavioral_patterns.keys())[0]
            patterns = self.behavioral_patterns[method_name]
            
            if not patterns:
                return None
            
            # Plot 1: Success rate by pattern
            ax1 = axes[0, 0]
            pattern_ids = [p.pattern_id.split('_')[-1] for p in patterns]
            success_rates = [p.success_rate for p in patterns]
            
            bars = ax1.bar(pattern_ids, success_rates, alpha=0.7)
            ax1.set_xlabel('Behavioral Pattern')
            ax1.set_ylabel('Success Rate')
            ax1.set_title(f'Success Rate by Pattern ({method_name})')
            ax1.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{rate:.2f}', ha='center', va='bottom')
            
            # Plot 2: Pattern frequency
            ax2 = axes[0, 1]
            frequencies = [p.frequency for p in patterns]
            
            bars = ax2.bar(pattern_ids, frequencies, alpha=0.7, color='orange')
            ax2.set_xlabel('Behavioral Pattern')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Pattern Frequency')
            
            # Plot 3: Feature comparison
            ax3 = axes[1, 0]
            
            if patterns:
                feature_names = list(patterns[0].characteristic_features.keys())
                x = np.arange(len(feature_names))
                width = 0.8 / len(patterns)
                
                for i, pattern in enumerate(patterns):
                    values = [pattern.characteristic_features[f] for f in feature_names]
                    ax3.bar(x + i * width, values, width, label=f'Pattern {pattern_ids[i]}', alpha=0.7)
                
                ax3.set_xlabel('Feature')
                ax3.set_ylabel('Value')
                ax3.set_title('Characteristic Features by Pattern')
                ax3.set_xticks(x + width * (len(patterns) - 1) / 2)
                ax3.set_xticklabels(feature_names, rotation=45, ha='right')
                ax3.legend()
            
            # Plot 4: Pattern descriptions
            ax4 = axes[1, 1]
            ax4.axis('off')  # Remove axes
            
            # Display pattern descriptions as text
            y_pos = 0.9
            for i, pattern in enumerate(patterns):
                description = pattern.description[:50] + "..." if len(pattern.description) > 50 else pattern.description
                ax4.text(0.05, y_pos - i * 0.15, f"Pattern {pattern_ids[i]}: {description}",
                        transform=ax4.transAxes, fontsize=10, wrap=True)
            
            ax4.set_title('Pattern Descriptions')
            
            plt.tight_layout()
            
            plot_path = output_dir / "behavioral_patterns.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create behavioral patterns plot: {e}")
            return None
    
    def _create_failure_analysis_plot(self, output_dir: Path) -> Optional[str]:
        """Create failure analysis visualization"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Aggregate failure data across methods
            all_failures = []
            for method_name, method_failures in self.failure_modes.items():
                all_failures.extend(method_failures)
            
            if not all_failures:
                return None
            
            # Plot 1: Failure frequency by type
            ax1 = axes[0, 0]
            failure_types = [f.failure_type for f in all_failures]
            type_counts = Counter(failure_types)
            
            types, counts = zip(*type_counts.most_common())
            bars = ax1.bar(types, counts, alpha=0.7, color='red')
            ax1.set_xlabel('Failure Type')
            ax1.set_ylabel('Count')
            ax1.set_title('Failure Frequency by Type')
            
            # Rotate x-axis labels if needed
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Plot 2: Severity distribution
            ax2 = axes[0, 1]
            severities = [f.severity_score for f in all_failures]
            
            ax2.hist(severities, bins=10, alpha=0.7, color='orange', edgecolor='black')
            ax2.set_xlabel('Severity Score')
            ax2.set_ylabel('Count')
            ax2.set_title('Failure Severity Distribution')
            
            # Plot 3: Method comparison
            ax3 = axes[1, 0]
            
            method_failure_counts = {}
            for method_name, method_failures in self.failure_modes.items():
                method_failure_counts[method_name] = len(method_failures)
            
            if method_failure_counts:
                methods, counts = zip(*method_failure_counts.items())
                bars = ax3.bar(methods, counts, alpha=0.7, color='purple')
                ax3.set_xlabel('Method')
                ax3.set_ylabel('Total Failures')
                ax3.set_title('Failure Count by Method')
                
                plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            
            # Plot 4: Root causes
            ax4 = axes[1, 1]
            
            all_root_causes = []
            for failure in all_failures:
                all_root_causes.extend(failure.root_causes)
            
            if all_root_causes:
                cause_counts = Counter(all_root_causes)
                causes, counts = zip(*cause_counts.most_common(5))  # Top 5
                
                ax4.barh(causes, counts, alpha=0.7, color='green')
                ax4.set_xlabel('Count')
                ax4.set_title('Most Common Root Causes')
            
            plt.tight_layout()
            
            plot_path = output_dir / "failure_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create failure analysis plot: {e}")
            return None
    
    def _create_uncertainty_calibration_plot(self, output_dir: Path) -> Optional[str]:
        """Create uncertainty calibration plot"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Get first method's uncertainty analysis
            method_name = list(self.uncertainty_analyses.keys())[0]
            analysis = self.uncertainty_analyses[method_name]
            
            # Plot 1: Reliability diagram
            ax1 = axes[0, 0]
            
            # Extract calibration data
            confidences = []
            accuracies = []
            
            for key, value in analysis.confidence_calibration.items():
                if key.endswith('_confidence') and not key.startswith('expected'):
                    bin_name = key.replace('_confidence', '')
                    if f"{bin_name}_accuracy" in analysis.confidence_calibration:
                        confidences.append(value)
                        accuracies.append(analysis.confidence_calibration[f"{bin_name}_accuracy"])
            
            if confidences and accuracies:
                ax1.plot(confidences, accuracies, 'o-', label='Calibration curve')
                ax1.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
                ax1.set_xlabel('Confidence')
                ax1.set_ylabel('Accuracy')
                ax1.set_title(f'Reliability Diagram ({method_name})')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Plot 2: Uncertainty sources
            ax2 = axes[0, 1]
            
            if analysis.uncertainty_sources:
                sources = list(analysis.uncertainty_sources.keys())
                values = list(analysis.uncertainty_sources.values())
                
                colors = ['lightblue', 'lightcoral']
                wedges, texts, autotexts = ax2.pie(values, labels=sources, autopct='%1.1f%%',
                                                  colors=colors, startangle=90)
                ax2.set_title('Uncertainty Sources')
            
            # Plot 3: Confidence vs Performance
            ax3 = axes[1, 0]
            
            if analysis.uncertainty_vs_performance:
                uncertainties, performances = zip(*analysis.uncertainty_vs_performance)
                
                ax3.scatter(uncertainties, performances, alpha=0.6)
                
                # Add trend line
                z = np.polyfit(uncertainties, performances, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(uncertainties), max(uncertainties), 100)
                ax3.plot(x_trend, p(x_trend), "r--", alpha=0.8)
                
                ax3.set_xlabel('Uncertainty')
                ax3.set_ylabel('Performance')
                ax3.set_title('Uncertainty vs Performance')
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Confidence regions
            ax4 = axes[1, 1]
            
            # Visualize over/under confidence regions
            ax4.text(0.5, 0.8, f'Optimal Threshold: {analysis.optimal_confidence_threshold:.3f}',
                    transform=ax4.transAxes, ha='center', fontsize=12, weight='bold')
            
            if analysis.overconfidence_regions:
                ax4.text(0.1, 0.6, f'Overconfident regions: {len(analysis.overconfidence_regions)}',
                        transform=ax4.transAxes, fontsize=10, color='red')
            
            if analysis.underconfidence_regions:
                ax4.text(0.1, 0.4, f'Underconfident regions: {len(analysis.underconfidence_regions)}',
                        transform=ax4.transAxes, fontsize=10, color='blue')
            
            # Show ECE if available
            if 'expected_calibration_error' in analysis.confidence_calibration:
                ece = analysis.confidence_calibration['expected_calibration_error']
                ax4.text(0.1, 0.2, f'Expected Calibration Error: {ece:.4f}',
                        transform=ax4.transAxes, fontsize=10, weight='bold')
            
            ax4.set_title('Calibration Summary')
            ax4.axis('off')
            
            plt.tight_layout()
            
            plot_path = output_dir / "uncertainty_calibration.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create uncertainty calibration plot: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    analyzer = AdvancedAnalyzer()
    
    # Generate sample data
    np.random.seed(42)
    
    # Sample learning curves data
    learning_data = {
        'Method_A': np.random.exponential(0.1, 1000).cumsum(),
        'Method_B': np.random.exponential(0.08, 1000).cumsum(),
        'Method_C': np.random.exponential(0.12, 1000).cumsum()
    }
    
    # Analyze learning curves
    learning_results = analyzer.analyze_learning_curves(learning_data)
    
    print(f"Analyzed learning curves for {len(learning_results)} methods")
    
    for method, result in learning_results.items():
        print(f"{method}:")
        print(f"  Convergence episode: {result.convergence_episode}")
        print(f"  Final performance: {result.asymptotic_performance:.3f}")
        print(f"  Learning rate: {result.learning_rate:.6f}")
    
    # Generate publication plots
    plots = analyzer.generate_publication_plots()
    print(f"Generated {len(plots)} publication plots")