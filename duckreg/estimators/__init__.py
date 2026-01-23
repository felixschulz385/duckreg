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
from .base import DuckEstimator, DuckReg, SEMethod

# Result containers
from ..core.results import RegressionResults, FirstStageResults, ModelSummary

# Bootstrap utilities
from ..core.vcov import (
    BootstrapExecutor,
    _bootstrap_iteration_iid,
    _bootstrap_iteration_cluster,
)

# Unified summary formatting
from ..utils.summary import (
    SummaryFormatter,
    format_summary,
    print_summary,
    to_tidy_df,
)

# Base class
from .DuckLinearModel import DuckLinearModel

# Concrete estimators
from .DuckRegression import DuckRegression
from .DuckMundlak import DuckMundlak
from .Duck2SLS import Duck2SLS
from .DuckDoubleDemeaning import DuckDoubleDemeaning
from .DuckMundlakEventStudy import DuckMundlakEventStudy
from .DuckRidge import DuckRidge

__all__ = [
    # Base classes
    'DuckEstimator',
    'DuckReg',
    'SEMethod',
    # Results
    'RegressionResults',
    'FirstStageResults',
    'ModelSummary',
    # Bootstrap utilities
    'BootstrapExecutor',
    # Summary utilities
    'SummaryFormatter',
    'format_summary',
    'print_summary',
    'to_tidy_df',
    # Base
    'DuckLinearModel',
    # Estimators
    'DuckRegression',
    'DuckMundlak',
    'Duck2SLS',
    'DuckDoubleDemeaning',
    'DuckMundlakEventStudy',
    'DuckRidge',
]
