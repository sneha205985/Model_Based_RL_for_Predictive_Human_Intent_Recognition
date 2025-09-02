"""
Models module for human behavior and intent prediction.

This module contains working implementations for:
- Gaussian Process regression
- Human trajectory prediction
- Uncertainty quantification
"""

from .gaussian_process import GaussianProcess

__all__ = [
    "GaussianProcess",
]