"""
.. include:: ../README.md
"""

from ._version import __version__, get_version, get_version_info
from .duckreg import duckreg, compressed_ols, DuckReg, SEMethod
from .utils.api import FEMethod
from .core.fitters import (
    wls, 
    wls_duckdb, 
    ridge_closed_form, 
    ridge_closed_form_batch,
    NumpyFitter,
    DuckDBFitter,
    get_fitter,
    FitterResult,
)
from .estimators import DuckRegression
from .core.results import RegressionResults, FirstStageResults, ModelSummary

__all__ = [
    # Version
    "__version__",
    "get_version",
    "get_version_info",
    # High-level API
    "duckreg",
    "compressed_ols",  # Backward compatibility alias
    "DuckReg",
    "SEMethod",
    "FEMethod",
    # Estimators
    "DuckRegression",
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
