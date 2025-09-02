"""
Data processing module for human behavior analysis.

This module contains tools for synthetic data generation, feature extraction,
data validation, data collection, and statistical analysis for human-robot 
interaction modeling.
"""

from .synthetic_generator import (
    SyntheticHumanBehaviorGenerator,
    SyntheticSequence,
    GestureType,
    TrajectoryParameters,
    GazeParameters
)

from .feature_extraction import (
    FeatureExtractor,
    ExtractedFeatures,
    FeatureConfig,
    FeatureType
)

from .validation import (
    DataValidator,
    DataPreprocessor,
    ValidationResult,
    ValidationLevel,
    PreprocessingConfig,
    DataQualityIssue
)

# New Phase 5 data collection and analysis components
from .data_collector import (
    HRIDataCollector,
    DataCollectionConfig, 
    DataRecord,
    DataType,
    DataFormat,
    create_default_collector,
    collect_experimental_session
)

from .statistical_analysis import (
    StatisticalAnalyzer,
    AnalysisConfiguration,
    AnalysisReport,
    StatisticalResult,
    AnalysisType,
    StatisticalTest,
    quick_method_comparison,
    quick_performance_analysis
)

__all__ = [
    # Synthetic data generation
    "SyntheticHumanBehaviorGenerator",
    "SyntheticSequence",
    "GestureType",
    "TrajectoryParameters",
    "GazeParameters",
    
    # Feature extraction
    "FeatureExtractor",
    "ExtractedFeatures",
    "FeatureConfig",
    "FeatureType",
    
    # Data validation and preprocessing
    "DataValidator",
    "DataPreprocessor",
    "ValidationResult",
    "ValidationLevel",
    "PreprocessingConfig",
    "DataQualityIssue",
    
    # Data collection (Phase 5)
    "HRIDataCollector",
    "DataCollectionConfig",
    "DataRecord", 
    "DataType",
    "DataFormat",
    "create_default_collector",
    "collect_experimental_session",
    
    # Statistical analysis (Phase 5)
    "StatisticalAnalyzer",
    "AnalysisConfiguration",
    "AnalysisReport",
    "StatisticalResult",
    "AnalysisType",
    "StatisticalTest",
    "quick_method_comparison",
    "quick_performance_analysis"
]