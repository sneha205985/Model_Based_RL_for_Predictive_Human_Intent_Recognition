"""
Mathematical Validation and Convergence Analysis Framework

This module provides formal mathematical validation including:
- Convergence proofs for Gaussian Process hyperparameter optimization
- Lyapunov stability analysis for MPC controllers
- Bayesian RL regret bounds and convergence guarantees
- Uncertainty calibration and safety verification

Components:
    mathematical_validation: Core mathematical validation framework
"""

__version__ = "1.0.0"

# Import main components with error handling
try:
    from .mathematical_validation import (
        ConvergenceAnalyzer,
        StabilityAnalyzer,
        UncertaintyValidator,
        SafetyVerifier,
        MathematicalValidationFramework
    )
except ImportError as e:
    ConvergenceAnalyzer = None
    StabilityAnalyzer = None
    UncertaintyValidator = None
    SafetyVerifier = None
    MathematicalValidationFramework = None
    import warnings
    warnings.warn(f"Could not import mathematical validation components: {e}")

__all__ = [
    "ConvergenceAnalyzer",
    "StabilityAnalyzer", 
    "UncertaintyValidator",
    "SafetyVerifier",
    "MathematicalValidationFramework"
]