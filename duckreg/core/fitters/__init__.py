"""
duckreg.core.fitters
====================

OLS fitting split into three modules:

- base           — FitterResult, BaseFitter, compute_vcov_dispatch, helpers
- numpy_fitter   — NumpyFitter (in-memory numpy)
- duckdb_fitter  — DuckDBFitter (out-of-core DuckDB), get_fitter, convenience fns

All public symbols from the original ``fitters.py`` are re-exported here for
full backward compatibility.
"""

from .base import (
    DEFAULT_ALPHA,
    CONDITION_NUMBER_THRESHOLD,
    FitterResult,
    BaseFitter,
    _resolve_vcov_spec,
    _validate_and_prepare_data,
    _compute_weighted_matrices,
    compute_vcov_dispatch,
)
from .numpy_fitter import NumpyFitter
from .duckdb_fitter import (
    DuckDBFitter,
    get_fitter,
    wls,
    wls_duckdb,
    ridge_closed_form,
    ridge_closed_form_batch,
)

__all__ = [
    # Base
    'DEFAULT_ALPHA',
    'CONDITION_NUMBER_THRESHOLD',
    'FitterResult',
    'BaseFitter',
    'compute_vcov_dispatch',
    # Concrete fitters
    'NumpyFitter',
    'DuckDBFitter',
    # Factory
    'get_fitter',
    # Convenience
    'wls',
    'wls_duckdb',
    'ridge_closed_form',
    'ridge_closed_form_batch',
]
