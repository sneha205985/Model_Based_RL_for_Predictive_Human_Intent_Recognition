"""
Experimental Research Validation Framework

This module provides comprehensive experimental validation capabilities including:
- Research validation orchestration  
- Statistical analysis with significance testing
- Ablation study frameworks
- Baseline comparison systems
- Publication-quality visualization

Components:
    research_validation: Main research validation framework
"""

__version__ = "1.0.0"

# Import main components with error handling
try:
    from .research_validation import (
        ResearchValidationFramework,
        StatisticalAnalyzer, 
        AblationStudyFramework,
        BaselineComparisonFramework,
        PublicationQualityVisualizer,
        ExperimentalConfig
    )
except ImportError as e:
    # Graceful fallback for missing dependencies
    ResearchValidationFramework = None
    StatisticalAnalyzer = None
    AblationStudyFramework = None 
    BaselineComparisonFramework = None
    PublicationQualityVisualizer = None
    ExperimentalConfig = None
    import warnings
    warnings.warn(f"Could not import research validation components: {e}")

__all__ = [
    "ResearchValidationFramework",
    "StatisticalAnalyzer",
    "AblationStudyFramework", 
    "BaselineComparisonFramework",
    "PublicationQualityVisualizer",
    "ExperimentalConfig"
]