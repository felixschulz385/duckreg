"""
.. include:: ../README.md
"""

from ._version import __version__, get_version, get_version_info
from .duckreg import compressed_ols, DuckReg, SEMethod, FEMethod
from .fitters import (
    wls, 
    wls_duckdb, 
    ridge_closed_form, 
    ridge_closed_form_batch,
    NumpyFitter,
    DuckDBFitter,
    get_fitter,
    FitterResult,
)
from .estimators import DuckRegression, DuckMundlak, DuckDoubleDemeaning
from .estimators.results import RegressionResults, FirstStageResults, ModelSummary

__all__ = [
    # Version
    "__version__",
    "get_version",
    "get_version_info",
    # High-level API
    "compressed_ols",
    "DuckReg",
    "SEMethod",
    "FEMethod",
    # Estimators
    "DuckRegression",
    "DuckMundlak",
    "DuckDoubleDemeaning",
    # Result containers
    "RegressionResults",
    "FirstStageResults",
    "ModelSummary",
    # Fitters
    "wls",
    "wls_duckdb",
    "ridge_closed_form",
    "ridge_closed_form_batch",
    "NumpyFitter",
    "DuckDBFitter",
    "get_fitter",
    "FitterResult",
]
