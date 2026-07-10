"""Estimators package for duckreg

Architecture follows OOP best practices:
- base.py: Abstract base class for all estimators
- core/results.py: Data containers (Single Responsibility)
- core/vcov.py: Bootstrap and variance-covariance computation (DRY)
- core/sql_builders.py: SQL query construction utilities (DRY)
- utils/summary.py: Unified formatting for regression and 2SLS results
- utils/name_utils.py: Coefficient naming utilities
- DuckLinearModel.py: Base class for OLS estimators
- Duck2SLS.py: IV/2SLS estimator (self-contained)
"""

# Base classes and constants
from .base import DuckEstimator, SEMethod

# Result containers
from ..core.results import RegressionResults, FirstStageResults, ModelSummary

# Unified summary formatting
from ..utils.summary import format_summary

# Base class
from .DuckLinearModel import DuckLinearModel

# Concrete estimators
from .DuckRegression import DuckRegression
from .Duck2SLS import Duck2SLS
from .DuckMundlakEventStudy import DuckMundlakEventStudy
from .DuckRidge import DuckRidge
from .DuckFE import DuckFE
from .DuckMediation import DuckMediation

__all__ = [
    # Base classes
    'DuckEstimator',
    'SEMethod',
    # Results
    'RegressionResults',
    'FirstStageResults',
    'ModelSummary',
    # Summary utilities
    'format_summary',
    # Base
    'DuckLinearModel',
    # Estimators
    'DuckRegression',
    'Duck2SLS',
    'DuckMundlakEventStudy',
    'DuckRidge',
    'DuckFE',
]
