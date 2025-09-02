"""
Comprehensive Unit Tests for Visualization Suite.

This module provides comprehensive unit tests for all visualization components
including core utilities, analyzers, dashboards, and report generation.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Import visualization modules
from src.visualization.core_utils import (
    BaseVisualizer, PlotConfig, ColorPalette, StatisticsCalculator,
    PlotAnnotator, ValidationUtils, VisualizationError
)
from src.visualization.performance_analysis import (
    PerformanceAnalyzer, PerformanceMetrics, MetricType
)
from src.visualization.safety_analysis import (
    SafetyAnalyzer, SafetyEvent, SafetyMetrics, RiskLevel
)
from src.visualization.bayesian_analysis import (
    BayesianAnalyzer, PosteriorDistribution, BayesianMetrics
)
from src.visualization.statistical_framework import (
    StatisticalFramework, StatisticalTest, TestType
)
from src.visualization.interactive_dashboards import (
    InteractiveDashboard, DashboardData, DashboardConfig
)
from src.visualization.realtime_system import (
    RealTimeVisualizer, RealTimeData, StreamConfig, RealTimeDataGenerator
)
from src.visualization.report_generation import (
    AutomatedReportGenerator, ReportConfig, ReportType
)


class TestCoreUtils:
    """Test core visualization utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PlotConfig()
        self.palette = ColorPalette()
        self.stats_calc = StatisticsCalculator()
        
    def test_plot_config_initialization(self):
        """Test PlotConfig initialization."""
        config = PlotConfig(figsize=(10, 8), dpi=150)
        assert config.figsize == (10, 8)
        assert config.dpi == 150
        assert config.font_family == 'DejaVu Sans'
        
    def test_color_palette(self):
        """Test ColorPalette functionality."""
        assert len(self.palette.categorical) >= 10
        assert self.palette.primary == '#2E86C1'
        assert all(color.startswith('#') for color in self.palette.categorical)
        
    def test_statistics_calculator(self):
        """Test StatisticsCalculator methods."""
        data = np.random.normal(100, 15, 50)
        
        # Test confidence interval
        lower, upper = self.stats_calc.confidence_interval(data, 0.95)
        assert lower < np.mean(data) < upper
        
        # Test bootstrap confidence interval
        lower_boot, upper_boot = self.stats_calc.bootstrap_confidence_interval(
            data, np.mean, 0.95, 100
        )
        assert lower_boot < np.mean(data) < upper_boot
        
        # Test effect size
        data1 = np.random.normal(100, 15, 30)
        data2 = np.random.normal(110, 15, 30)
        effect_size = self.stats_calc.effect_size_cohens_d(data1, data2)
        assert isinstance(effect_size, float)
        
    def test_validation_utils(self):
        """Test ValidationUtils methods."""
        # Test array shape validation
        arr1 = np.random.randn(50)
        arr2 = np.random.randn(50)
        arr3 = np.random.randn(40)
        
        ValidationUtils.validate_array_shapes(arr1, arr2)  # Should not raise
        
        with pytest.raises(VisualizationError):
            ValidationUtils.validate_array_shapes(arr1, arr3)
        
        # Test finite data validation
        valid_data = np.random.randn(100)
        invalid_data = np.array([1, 2, np.inf, 4, np.nan])
        
        ValidationUtils.validate_finite_data(valid_data)  # Should not raise
        
        with pytest.raises(VisualizationError):
            ValidationUtils.validate_finite_data(invalid_data)
        
        # Test probability data validation
        valid_probs = np.array([0.1, 0.5, 0.8, 0.3])
        invalid_probs = np.array([0.1, 1.5, 0.8, -0.2])
        
        ValidationUtils.validate_probability_data(valid_probs)  # Should not raise
        
        with pytest.raises(VisualizationError):
            ValidationUtils.validate_probability_data(invalid_probs)


class TestPerformanceAnalysis:
    """Test performance analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PerformanceAnalyzer()
        self.success_rates = {
            'Method A': [0.85, 0.87, 0.82, 0.86, 0.84],
            'Method B': [0.78, 0.81, 0.79, 0.80, 0.77],
            'Method C': [0.91, 0.89, 0.92, 0.88, 0.90]
        }
        
    def test_success_rate_comparison_matplotlib(self):
        """Test success rate comparison with matplotlib."""
        self.analyzer.config.interactive_engine = 'matplotlib'
        
        fig = self.analyzer.plot_success_rate_comparison(
            self.success_rates, plot_type='bar'
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1
        plt.close(fig)
        
    def test_success_rate_comparison_plotly(self):
        """Test success rate comparison with plotly."""
        self.analyzer.config.interactive_engine = 'plotly'
        
        fig = self.analyzer.plot_success_rate_comparison(
            self.success_rates, plot_type='box'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        
    def test_learning_curves(self):
        """Test learning curves visualization."""
        learning_data = {
            'Method A': {
                'loss': [1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.2],
                'accuracy': [0.6, 0.7, 0.75, 0.8, 0.82, 0.84, 0.85]
            },
            'Method B': {
                'loss': [1.2, 0.9, 0.7, 0.5, 0.4, 0.35, 0.3],
                'accuracy': [0.55, 0.65, 0.72, 0.78, 0.8, 0.81, 0.82]
            }
        }
        
        fig = self.analyzer.plot_learning_curves(learning_data)
        assert isinstance(fig, (plt.Figure, go.Figure))
        
        if isinstance(fig, plt.Figure):
            plt.close(fig)
            
    def test_performance_heatmap(self):
        """Test performance heatmap."""
        performance_matrix = pd.DataFrame({
            'Accuracy': [0.85, 0.78, 0.91],
            'Precision': [0.82, 0.76, 0.89],
            'Recall': [0.87, 0.79, 0.92],
            'F1-Score': [0.84, 0.77, 0.90]
        }, index=['Method A', 'Method B', 'Method C'])
        
        fig = self.analyzer.plot_performance_heatmap(performance_matrix)
        assert isinstance(fig, (plt.Figure, go.Figure))
        
        if isinstance(fig, plt.Figure):
            plt.close(fig)


class TestSafetyAnalysis:
    """Test safety analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SafetyAnalyzer()
        self.distances = np.random.uniform(0.3, 2.0, 100)
        self.timestamps = np.linspace(0, 10, 100)
        self.safety_threshold = 0.5
        
    def test_distance_over_time(self):
        """Test distance over time visualization."""
        fig = self.analyzer.plot_distance_over_time(
            self.distances.tolist(),
            self.timestamps.tolist(),
            self.safety_threshold
        )
        
        assert isinstance(fig, (plt.Figure, go.Figure))
        
        if isinstance(fig, plt.Figure):
            plt.close(fig)
            
    def test_safety_violations_analysis(self):
        """Test safety violations analysis."""
        violations = [
            SafetyEvent(
                timestamp=1.0,
                event_type="distance_violation",
                severity="high",
                distance=0.3,
                position=np.array([1.0, 2.0, 0.5]),
                human_position=np.array([1.2, 2.1, 0.5]),
                robot_velocity=0.1
            ),
            SafetyEvent(
                timestamp=5.5,
                event_type="velocity_violation",
                severity="moderate",
                distance=0.8,
                position=np.array([2.0, 1.5, 0.8]),
                human_position=np.array([2.5, 1.8, 0.8]),
                robot_velocity=0.5
            )
        ]
        
        fig = self.analyzer.plot_safety_violations_analysis(violations)
        assert isinstance(fig, (plt.Figure, go.Figure))
        
        if isinstance(fig, plt.Figure):
            plt.close(fig)
            
    def test_risk_assessment(self):
        """Test risk assessment visualization."""
        risk_scores = np.random.uniform(0, 1, 100)
        
        risk_factors = {
            'distance_factor': np.random.uniform(0, 0.4, 100),
            'velocity_factor': np.random.uniform(0, 0.3, 100),
            'uncertainty_factor': np.random.uniform(0, 0.3, 100)
        }
        
        fig = self.analyzer.plot_risk_assessment(
            risk_scores.tolist(),
            self.timestamps.tolist(),
            risk_factors
        )
        
        assert isinstance(fig, (plt.Figure, go.Figure))
        
        if isinstance(fig, plt.Figure):
            plt.close(fig)


class TestBayesianAnalysis:
    """Test Bayesian analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = BayesianAnalyzer()
        
    def test_posterior_evolution(self):
        """Test posterior evolution visualization."""
        # Create mock posterior distributions
        posteriors = []
        for i in range(5):
            samples = np.random.multivariate_normal(
                mean=[0.5 + i*0.1, 1.0 - i*0.05],
                cov=[[0.1, 0.02], [0.02, 0.08]],
                size=1000
            )
            posterior = PosteriorDistribution(
                samples=samples,
                parameter_names=['param1', 'param2']
            )
            posteriors.append(posterior)
        
        fig = self.analyzer.plot_posterior_evolution(posteriors)
        assert isinstance(fig, (plt.Figure, go.Figure))
        
        if isinstance(fig, plt.Figure):
            plt.close(fig)
            
    def test_uncertainty_calibration(self):
        """Test uncertainty calibration plot."""
        n_samples = 1000
        predicted_probs = np.random.uniform(0, 1, n_samples)
        true_labels = np.random.binomial(1, predicted_probs)
        
        fig = self.analyzer.plot_uncertainty_calibration(
            predicted_probs, true_labels
        )
        
        assert isinstance(fig, (plt.Figure, go.Figure))
        
        if isinstance(fig, plt.Figure):
            plt.close(fig)
            
    def test_information_gain(self):
        """Test information gain visualization."""
        info_gain_history = np.random.exponential(0.5, 50)
        timestamps = list(range(50))
        
        fig = self.analyzer.plot_information_gain(
            info_gain_history.tolist(), timestamps
        )
        
        assert isinstance(fig, (plt.Figure, go.Figure))
        
        if isinstance(fig, plt.Figure):
            plt.close(fig)


class TestStatisticalFramework:
    """Test statistical analysis framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.framework = StatisticalFramework()
        self.data_groups = {
            'Group A': np.random.normal(100, 15, 50),
            'Group B': np.random.normal(105, 15, 50),
            'Group C': np.random.normal(95, 15, 50)
        }
        
    def test_comprehensive_analysis(self):
        """Test comprehensive statistical analysis."""
        results = self.framework.perform_comprehensive_analysis(
            self.data_groups, alpha=0.05
        )
        
        # Check structure of results
        assert 'descriptive_stats' in results
        assert 'normality_tests' in results
        assert 'statistical_tests' in results
        assert 'effect_sizes' in results
        assert 'recommendations' in results
        
        # Check descriptive stats
        assert isinstance(results['descriptive_stats'], pd.DataFrame)
        assert len(results['descriptive_stats']) == 3
        
        # Check statistical tests
        assert len(results['statistical_tests']) > 0
        assert isinstance(results['statistical_tests'][0], StatisticalTest)
        
    def test_effect_sizes_analysis(self):
        """Test effect sizes visualization."""
        effect_sizes = {
            'Group A vs B': 0.3,
            'Group A vs C': 0.7,
            'Group B vs C': -0.5
        }
        
        fig = self.framework.plot_effect_sizes_analysis(effect_sizes)
        assert isinstance(fig, (plt.Figure, go.Figure))
        
        if isinstance(fig, plt.Figure):
            plt.close(fig)
            
    def test_bootstrap_analysis(self):
        """Test bootstrap analysis."""
        data = np.random.normal(100, 15, 100)
        
        fig = self.framework.plot_bootstrap_analysis(
            data, statistic=np.mean, n_bootstrap=200
        )
        
        assert isinstance(fig, (plt.Figure, go.Figure))
        
        if isinstance(fig, plt.Figure):
            plt.close(fig)


class TestInteractiveDashboards:
    """Test interactive dashboard functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DashboardConfig()
        self.dashboard = InteractiveDashboard(dashboard_config=self.config)
        
    def test_dashboard_initialization(self):
        """Test dashboard initialization."""
        assert self.dashboard.dashboard_config.title == "Model-Based RL System Dashboard"
        assert self.dashboard.dashboard_config.update_interval == 1000
        assert isinstance(self.dashboard.data_buffer, DashboardData)
        
    def test_comprehensive_dashboard_creation(self):
        """Test comprehensive dashboard creation."""
        layout_config = {
            'layout': 'grid',
            'components': [
                {
                    'name': 'time_series',
                    'type': 'time_series',
                    'title': 'Performance Metrics',
                    'data': {
                        'x': list(range(50)),
                        'y': {
                            'metric1': np.random.randn(50).tolist(),
                            'metric2': np.random.randn(50).tolist()
                        }
                    }
                },
                {
                    'name': 'histogram',
                    'type': 'histogram',
                    'title': 'Data Distribution',
                    'data': {
                        'x': np.random.randn(1000).tolist()
                    }
                }
            ]
        }
        
        fig = self.dashboard.create_comprehensive_dashboard(layout_config)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        
    def test_parameter_sensitivity_dashboard(self):
        """Test parameter sensitivity dashboard."""
        parameter_ranges = {
            'learning_rate': (0.001, 0.1),
            'batch_size': (16, 128),
            'hidden_units': (64, 512)
        }
        
        def mock_sensitivity_function(params):
            # Mock function that returns a metric based on parameters
            return sum(params.values()) / len(params) * 0.8
        
        fig = self.dashboard.create_parameter_sensitivity_dashboard(
            parameter_ranges, mock_sensitivity_function
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestRealTimeSystem:
    """Test real-time visualization system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StreamConfig(buffer_size=100, update_rate=5.0)
        self.visualizer = RealTimeVisualizer(stream_config=self.config)
        self.data_generator = RealTimeDataGenerator(seed=42)
        
    def test_real_time_data_generation(self):
        """Test real-time data generation."""
        data = self.data_generator.generate_sample_data()
        
        assert isinstance(data, RealTimeData)
        assert isinstance(data.timestamp, float)
        assert 'position' in data.robot_state
        assert 'position' in data.human_state
        assert 'intent_probabilities' in data.predictions
        
    def test_add_data(self):
        """Test adding data to real-time system."""
        # Test with RealTimeData object
        rt_data = self.data_generator.generate_sample_data()
        self.visualizer.add_data(rt_data)
        
        assert len(self.visualizer.data_buffer) == 1
        
        # Test with dictionary
        dict_data = {
            'timestamp': time.time(),
            'robot_state': {'position': [1, 2, 3]},
            'performance_metrics': {'success_rate': 0.85}
        }
        self.visualizer.add_data(dict_data)
        
        assert len(self.visualizer.data_buffer) == 2
        
    def test_streaming_plot_creation(self):
        """Test streaming plot creation."""
        metrics = ['cpu_usage', 'memory_usage', 'temperature']
        
        def mock_data_source():
            return {
                'cpu_usage': np.random.uniform(20, 80),
                'memory_usage': np.random.uniform(30, 70),
                'temperature': np.random.uniform(25, 45)
            }
        
        fig = self.visualizer.create_streaming_plot(metrics, mock_data_source)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        plt.close(fig)
        
    def test_data_export(self):
        """Test data export functionality."""
        # Add some test data
        for _ in range(10):
            data = self.data_generator.generate_sample_data()
            self.visualizer.add_data(data)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            self.visualizer.export_data(tmp.name, format='json')
            
            # Verify file was created and has content
            with open(tmp.name, 'r') as f:
                exported_data = json.load(f)
                
            assert len(exported_data) == 10
            assert 'timestamp' in exported_data[0]
            
            # Cleanup
            Path(tmp.name).unlink()


class TestReportGeneration:
    """Test automated report generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ReportConfig(
            title="Test Analysis Report",
            author="Test Suite",
            output_formats=["html"]  # Only test HTML to avoid PDF dependencies
        )
        self.generator = AutomatedReportGenerator(self.config)
        
    def test_report_generator_initialization(self):
        """Test report generator initialization."""
        assert self.generator.config.title == "Test Analysis Report"
        assert self.generator.config.author == "Test Suite"
        assert hasattr(self.generator, 'performance_analyzer')
        assert hasattr(self.generator, 'template_env')
        
    def test_comprehensive_report_generation(self):
        """Test comprehensive report generation."""
        experimental_data = {
            'performance_data': {
                'success_rates': {
                    'Method A': [0.85, 0.87, 0.82],
                    'Method B': [0.78, 0.81, 0.79]
                },
                'summary_stats': {
                    'Method A': {'mean': 0.85, 'std': 0.02},
                    'Method B': {'mean': 0.79, 'std': 0.015}
                }
            },
            'safety_data': {
                'distances': [1.2, 1.0, 0.8, 0.6, 0.9, 1.1],
                'timestamps': [0, 1, 2, 3, 4, 5],
                'safety_threshold': 0.5,
                'violations': []
            },
            'experimental_setup': {
                'n_trials': 100,
                'methods': ['Method A', 'Method B'],
                'duration': '2 hours'
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            output_path = self.generator.generate_comprehensive_report(
                experimental_data, tmp.name, ReportType.COMPREHENSIVE
            )
            
            # Verify report was generated
            assert Path(output_path).exists()
            
            # Check HTML content
            with open(output_path, 'r') as f:
                html_content = f.read()
                
            assert 'Test Analysis Report' in html_content
            assert 'Performance Analysis' in html_content
            assert 'Safety Analysis' in html_content
            
            # Cleanup
            Path(output_path).unlink()
            
    def test_publication_figures_creation(self):
        """Test publication-ready figure creation."""
        experimental_data = {
            'performance_data': {
                'success_rates': {
                    'Method A': [0.85, 0.87, 0.82],
                    'Method B': [0.78, 0.81, 0.79]
                }
            }
        }
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            figure_paths = self.generator.create_publication_figures(
                experimental_data, tmp_dir
            )
            
            # Verify figures were created
            assert len(figure_paths) > 0
            
            for path in figure_paths:
                assert Path(path).exists()
                assert Path(path).suffix == '.png'


class TestIntegration:
    """Integration tests for the complete visualization suite."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.test_data = self._generate_comprehensive_test_data()
        
    def _generate_comprehensive_test_data(self) -> Dict[str, Any]:
        """Generate comprehensive test data for integration tests."""
        np.random.seed(42)
        
        return {
            'performance_data': {
                'success_rates': {
                    'Baseline': np.random.normal(0.75, 0.05, 20),
                    'Proposed': np.random.normal(0.85, 0.04, 20),
                    'Advanced': np.random.normal(0.82, 0.06, 20)
                },
                'completion_times': {
                    'Baseline': np.random.normal(12.0, 2.0, 20),
                    'Proposed': np.random.normal(10.5, 1.5, 20),
                    'Advanced': np.random.normal(11.2, 1.8, 20)
                },
                'learning_curves': {
                    'Baseline': {
                        'loss': np.exp(-np.linspace(0, 2, 50)) + 0.1,
                        'accuracy': 1 - np.exp(-np.linspace(0, 2, 50)) * 0.5
                    },
                    'Proposed': {
                        'loss': np.exp(-np.linspace(0, 2.5, 50)) + 0.05,
                        'accuracy': 1 - np.exp(-np.linspace(0, 2.5, 50)) * 0.4
                    }
                }
            },
            'safety_data': {
                'distances': np.concatenate([
                    np.random.uniform(0.8, 2.0, 80),  # Safe distances
                    np.random.uniform(0.2, 0.5, 5)   # Some violations
                ]),
                'timestamps': np.linspace(0, 100, 85),
                'safety_threshold': 0.5,
                'violations': [
                    SafetyEvent(
                        timestamp=45.2,
                        event_type="distance_violation",
                        severity="high",
                        distance=0.3,
                        position=np.array([1.0, 2.0, 0.5]),
                        human_position=np.array([1.2, 2.1, 0.5]),
                        robot_velocity=0.1
                    )
                ]
            },
            'experimental_setup': {
                'n_trials': 60,
                'methods': ['Baseline', 'Proposed', 'Advanced'],
                'duration': '4 hours',
                'environment': 'Simulation'
            }
        }
    
    def test_end_to_end_analysis_workflow(self):
        """Test complete end-to-end analysis workflow."""
        # 1. Performance Analysis
        perf_analyzer = PerformanceAnalyzer()
        success_fig = perf_analyzer.plot_success_rate_comparison(
            self.test_data['performance_data']['success_rates']
        )
        assert isinstance(success_fig, (plt.Figure, go.Figure))
        if isinstance(success_fig, plt.Figure):
            plt.close(success_fig)
        
        # 2. Safety Analysis
        safety_analyzer = SafetyAnalyzer()
        safety_fig = safety_analyzer.plot_distance_over_time(
            self.test_data['safety_data']['distances'].tolist(),
            self.test_data['safety_data']['timestamps'].tolist(),
            self.test_data['safety_data']['safety_threshold']
        )
        assert isinstance(safety_fig, (plt.Figure, go.Figure))
        if isinstance(safety_fig, plt.Figure):
            plt.close(safety_fig)
        
        # 3. Statistical Analysis
        stat_framework = StatisticalFramework()
        comparison_data = {
            'Baseline': self.test_data['performance_data']['success_rates']['Baseline'],
            'Proposed': self.test_data['performance_data']['success_rates']['Proposed']
        }
        stat_results = stat_framework.perform_comprehensive_analysis(comparison_data)
        assert 'statistical_tests' in stat_results
        
        # 4. Report Generation
        report_generator = AutomatedReportGenerator(
            ReportConfig(output_formats=["html"])
        )
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            report_path = report_generator.generate_comprehensive_report(
                self.test_data, tmp.name
            )
            
            assert Path(report_path).exists()
            Path(report_path).unlink()
    
    def test_real_time_to_report_pipeline(self):
        """Test pipeline from real-time data to final report."""
        # 1. Generate real-time data
        rt_visualizer = RealTimeVisualizer()
        data_generator = RealTimeDataGenerator(seed=123)
        
        # Collect data over time
        for _ in range(20):
            data = data_generator.generate_sample_data()
            rt_visualizer.add_data(data)
        
        assert len(rt_visualizer.data_buffer) == 20
        
        # 2. Export collected data
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            rt_visualizer.export_data(tmp.name, format='json')
            
            # Verify export
            with open(tmp.name, 'r') as f:
                exported_data = json.load(f)
            
            assert len(exported_data) == 20
            Path(tmp.name).unlink()
        
        # 3. Create dashboard from data
        dashboard = InteractiveDashboard()
        layout_config = {
            'layout': 'grid',
            'components': [
                {
                    'name': 'performance',
                    'type': 'time_series',
                    'title': 'System Performance',
                    'data': {
                        'x': list(range(20)),
                        'y': {'metric': np.random.randn(20).tolist()}
                    }
                }
            ]
        }
        
        dash_fig = dashboard.create_comprehensive_dashboard(layout_config)
        assert isinstance(dash_fig, go.Figure)


# Test fixtures and utilities
@pytest.fixture
def sample_performance_data():
    """Fixture providing sample performance data."""
    return {
        'success_rates': {
            'Method A': [0.85, 0.87, 0.82, 0.86, 0.84],
            'Method B': [0.78, 0.81, 0.79, 0.80, 0.77],
            'Method C': [0.91, 0.89, 0.92, 0.88, 0.90]
        },
        'completion_times': {
            'Method A': [10.2, 9.8, 10.5, 10.1, 10.3],
            'Method B': [11.5, 11.2, 11.8, 11.3, 11.6],
            'Method C': [9.5, 9.8, 9.2, 9.7, 9.4]
        }
    }


@pytest.fixture
def sample_safety_data():
    """Fixture providing sample safety data."""
    return {
        'distances': np.random.uniform(0.3, 2.0, 100),
        'timestamps': np.linspace(0, 50, 100),
        'safety_threshold': 0.5,
        'violations': [
            SafetyEvent(
                timestamp=15.5,
                event_type="distance_violation",
                severity="moderate",
                distance=0.4,
                position=np.array([1.5, 2.0, 0.8]),
                human_position=np.array([1.8, 2.2, 0.8]),
                robot_velocity=0.2
            )
        ]
    }


if __name__ == '__main__':
    # Run tests with detailed output
    pytest.main([__file__, '-v', '--tb=short'])