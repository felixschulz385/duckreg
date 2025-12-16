"""Estimators package for duckreg

Architecture follows OOP best practices:
- results.py: Data containers (Single Responsibility)
- mixins.py: Reusable functionality (DRY, Interface Segregation)
- summary.py: Unified formatting for regression and 2SLS results
- DuckLinearModel.py: Base class for OLS estimators
- Duck2SLS.py: IV/2SLS estimator (self-contained)
"""

# Result containers
from .results import RegressionResults, FirstStageResults

# Mixins for composition
from .mixins import (
    VCovMixin,
    SQLBuilderMixin,
    MundlakMixin,
    BootstrapExecutor,
    _bootstrap_iteration_iid,
    _bootstrap_iteration_cluster,
)

# Unified summary formatting
from .summary import (
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

__all__ = [
    # Results
    'RegressionResults',
    'FirstStageResults',
    # Mixins
    'VCovMixin',
    'SQLBuilderMixin',
    'MundlakMixin',
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
]
