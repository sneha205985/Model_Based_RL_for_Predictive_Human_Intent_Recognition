"""
Comprehensive Visualization Module for Model-Based RL Human Intent Recognition.

This module provides a complete visualization and analysis suite including:
- Core visualization utilities and base classes
- Performance analysis and benchmarking
- Safety analysis and monitoring
- Bayesian analysis and uncertainty quantification
- Statistical analysis framework
- Interactive dashboards and real-time monitoring
- Automated report generation
- Human behavior analysis
- MPC visualization tools

Key Features:
- Publication-ready plots with consistent styling
- Interactive dashboards with real-time updates
- Statistical significance testing and effect size analysis
- Uncertainty visualization and calibration
- Safety monitoring and violation analysis
- Automated HTML/PDF report generation
- WebSocket-based real-time streaming
- Multi-format export capabilities
"""

from .core_utils import (
    BaseVisualizer, PlotConfig, ColorPalette, PlotType, ColorScheme,
    StatisticsCalculator, PlotAnnotator, ValidationUtils,
    create_publication_config, create_presentation_config, create_web_config
)

from .performance_analysis import (
    PerformanceAnalyzer, PerformanceMetrics, MetricType
)

from .safety_analysis import (
    SafetyAnalyzer, SafetyEvent, SafetyMetrics, SafetyViolationType, RiskLevel
)

from .bayesian_analysis import (
    BayesianAnalyzer, PosteriorDistribution, BayesianMetrics, UncertaintyType
)

from .statistical_framework import (
    StatisticalFramework, StatisticalTest, MultipleComparisonResult,
    TestType, EffectSizeType
)

from .interactive_dashboards import (
    InteractiveDashboard, DashboardData, DashboardConfig, DashboardComponent
)

from .realtime_system import (
    RealTimeVisualizer, RealTimeData, StreamConfig, UpdateStrategy,
    RealTimeDataGenerator
)

from .report_generation import (
    AutomatedReportGenerator, ReportSection, ReportConfig, ReportType
)

# Import existing visualization components
from .behavior_plots import (
    BehaviorVisualizer, quick_plot_trajectory, quick_plot_intents
)

from .mpc_plots import (
    MPCVisualizer, HRIVisualizationDashboard, create_mpc_visualization_suite
)

__all__ = [
    # Core utilities
    'BaseVisualizer', 'PlotConfig', 'ColorPalette', 'PlotType', 'ColorScheme',
    'StatisticsCalculator', 'PlotAnnotator', 'ValidationUtils',
    'create_publication_config', 'create_presentation_config', 'create_web_config',
    
    # Performance analysis
    'PerformanceAnalyzer', 'PerformanceMetrics', 'MetricType',
    
    # Safety analysis
    'SafetyAnalyzer', 'SafetyEvent', 'SafetyMetrics', 'SafetyViolationType', 'RiskLevel',
    
    # Bayesian analysis
    'BayesianAnalyzer', 'PosteriorDistribution', 'BayesianMetrics', 'UncertaintyType',
    
    # Statistical framework
    'StatisticalFramework', 'StatisticalTest', 'MultipleComparisonResult',
    'TestType', 'EffectSizeType',
    
    # Interactive dashboards
    'InteractiveDashboard', 'DashboardData', 'DashboardConfig', 'DashboardComponent',
    
    # Real-time system
    'RealTimeVisualizer', 'RealTimeData', 'StreamConfig', 'UpdateStrategy',
    'RealTimeDataGenerator',
    
    # Report generation
    'AutomatedReportGenerator', 'ReportSection', 'ReportConfig', 'ReportType',
    
    # Existing components
    'BehaviorVisualizer', 'quick_plot_trajectory', 'quick_plot_intents',
    'MPCVisualizer', 'HRIVisualizationDashboard', 'create_mpc_visualization_suite',
]

# Version information
__version__ = "1.0.0"
__author__ = "Claude Code System"

# Module-level configuration
import logging
logger = logging.getLogger(__name__)
logger.info("Comprehensive visualization suite loaded successfully")