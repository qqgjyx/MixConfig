"""
Predictor wrappers for MixConfig experiments.

Supports both neural and classical downstream predictors.
"""

from .neural import MLPPredictor, LitMLPPredictor
from .classical import XGBoostPredictor, RandomForestPredictor, LinearPredictor

__all__ = [
    # Neural
    "MLPPredictor",
    "LitMLPPredictor",
    # Classical
    "XGBoostPredictor",
    "RandomForestPredictor",
    "LinearPredictor",
]
