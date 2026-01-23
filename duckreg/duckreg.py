"""High-level API for compressed OLS regression.

This module provides the main entry point for users to run compressed OLS regressions
with support for fixed effects, instrumental variables, and various standard error methods.
It handles formula parsing, data source resolution, and estimator selection.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .core.vcov import parse_vcov_specification, parse_cluster_vars, VcovTypeNotSupportedError
from .estimators.base import DuckEstimator, SEMethod
from .utils.api import FEMethod, _resolve_table_name, _resolve_db_path

logger = logging.getLogger(__name__)

# Backward compatibility - re-export from base
DuckReg = DuckEstimator


# ============================================================================
# High-level API
# ============================================================================

def compressed_ols(
    formula: str,
    data: str,
    subset: str = None,
    n_bootstraps: int = 100,
    cache_dir: str = None,
    round_strata: int = None,
    seed: int = 42,
    fe_method: str = FEMethod.AUTO,
    duckdb_kwargs: dict = None,
    db_name: str = None,
    n_jobs: int = 1,
    se_method: str = SEMethod.HC1,
    fitter: str = "numpy",
    **kwargs
) -> "DuckEstimator":
    """High-level API for compressed OLS regression with lfe-style formula.
    
    Orchestrates the entire regression workflow:
    1. Parse the lfe-style formula to extract outcomes, covariates, FE, IV, and clusters
    2. Resolve data source and database paths
    3. Select appropriate estimator based on model type (OLS, IV, or Mundlak)
    4. Fit the model and compute standard errors
    
    Args:
        formula: Regression formula in format "y ~ x1 + x2 | fe1 + fe2 | endog (inst1 + inst2) | cluster"
        data: Filepath to data file (.csv, .parquet, or directory of .parquet files)
        subset: SQL WHERE clause to subset data
        n_bootstraps: Number of bootstrap iterations (only used if se_method="BS")
        cache_dir: Directory for DuckDB cache files
        round_strata: Number of decimals to round strata columns
        seed: Random seed for reproducibility
        fe_method: Method for handling fixed effects ('mundlak' or 'demean')
        duckdb_kwargs: Dictionary of DuckDB configuration settings
        db_name: Full path to DuckDB database file
        n_jobs: Number of parallel jobs for bootstrap
        se_method: Method for computing standard errors ('iid', 'HC1', 'CRV1', 'BS', or 'none')
        fitter: Estimation method ('numpy' for in-memory, 'duckdb' for out-of-core)
        **kwargs: Additional arguments passed to estimator
    
    Returns:
        Fitted estimator object with results
    """
    logger.debug(f"=== compressed_ols START ===")
    
    # Import estimator classes and formula parser
    from .estimators import DuckRegression, DuckMundlak, Duck2SLS
    from .utils.formula_parser import FormulaParser
    
    # Parse the lfe-style formula to extract model components
    parsed_formula = FormulaParser().parse(formula)
    fe_cols = parsed_formula.get_fe_names()  # Fixed effect column names
    has_iv = parsed_formula.has_instruments()  # Does model have instrumental variables?
    
    logger.debug(f"Parsed: outcomes={parsed_formula.get_outcome_names()}, "
                 f"covariates={parsed_formula.get_covariate_names()}, "
                 f"fe={fe_cols}, cluster={parsed_formula.cluster}, "
                 f"has_iv={has_iv}, fitter={fitter}")
    
    # Resolve data paths: determine DuckDB database location and table reference
    # _resolve_db_path creates a cache database based on data path hash if db_name not specified
    # _resolve_table_name converts file path to appropriate DuckDB read function
    resolved_db = _resolve_db_path(data, cache_dir, db_name)
    table_name = _resolve_table_name(Path(data).resolve())
    
    # Resolve fixed effects method for 2SLS
    # For IV models, default to demeaning if AUTO is specified
    resolved_fe_method = fe_method
    if fe_method == FEMethod.AUTO:
        resolved_fe_method = FEMethod.DEMEAN
    
    # Select appropriate estimator class based on model type
    # IV models use Duck2SLS (2-stage least squares)
    # OLS models use either DuckMundlak (with Mundlak approach for FE) or DuckRegression
    if has_iv:
        # Instrumental variables: use 2SLS
        estimator = Duck2SLS(
            db_name=resolved_db,
            table_name=table_name,
            formula=parsed_formula,
            subset=subset,
            n_bootstraps=n_bootstraps if se_method == SEMethod.BS else 0,
            round_strata=round_strata,
            seed=seed,
            duckdb_kwargs=duckdb_kwargs,
            n_jobs=n_jobs,
            fitter=fitter,
            fe_method=resolved_fe_method,
            **kwargs
        )
    else:
        # OLS regression: choose between Mundlak or simple demeaning approach
        # Mundlak approach: absorb FE by demeaning with group means of all covariates
        # This enables consistent estimation with correlated random effects
        use_mundlak = bool(fe_cols) and fe_method == FEMethod.MUNDLAK
        if fe_cols and fe_method not in (FEMethod.MUNDLAK, FEMethod.DEMEAN, FEMethod.AUTO):
            raise ValueError(f"fe_method must be '{FEMethod.MUNDLAK}' or '{FEMethod.DEMEAN}'")
        
        estimator_class = DuckMundlak if use_mundlak else DuckRegression
        estimator = estimator_class(
            db_name=resolved_db,
            table_name=table_name,
            formula=parsed_formula,
            subset=subset,
            n_bootstraps=n_bootstraps if se_method == SEMethod.BS else 0,
            round_strata=round_strata,
            seed=seed,
            duckdb_kwargs=duckdb_kwargs,
            n_jobs=n_jobs,
            fitter=fitter,
            **kwargs
        )
    
    # Fit the model (triggers data preparation, compression, estimation, and SE computation)
    estimator.fit(se_method=se_method)
    
    logger.debug(f"=== compressed_ols END ===")
    return estimator


