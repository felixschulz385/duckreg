"""
.. include:: ../README.md
"""

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

__all__ = [
    # High-level API
    "compressed_ols",
    "DuckReg",
    "SEMethod",
    "FEMethod",
    # Estimators
    "DuckRegression",
    "DuckMundlak",
    "DuckDoubleDemeaning",
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
