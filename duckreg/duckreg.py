"""High-level API for compressed OLS regression.

This module provides the main entry point for users to run compressed OLS regressions
with support for fixed effects, instrumental variables, and various standard error methods.
It handles formula parsing, data source resolution, and estimator selection.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .core.vcov import parse_vcov_specification, parse_cluster_vars, VcovTypeNotSupportedError
from .estimators.base import DuckEstimator, SEMethod
from .utils.api import FEMethod, _resolve_data_source, _DUCKDB_VIEW_NAME

logger = logging.getLogger(__name__)

# Backward compatibility - re-export from base
DuckReg = DuckEstimator


# ============================================================================
# High-level API
# ============================================================================

def duckreg(
    formula: str,
    data: Any,
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
    remove_singletons: bool = True,
    ssc_dict: Optional[Dict[str, Any]] = None,
    **kwargs
) -> "DuckEstimator":
    """High-level API for DuckReg regression with lfe-style formula.

    Orchestrates the entire regression workflow:
    1. Parse the lfe-style formula to extract outcomes, covariates, FE, IV, and clusters
    2. Resolve data source and database paths
    3. Select appropriate estimator based on model type (OLS, IV, or Mundlak)
    4. Fit the model and compute standard errors

    Args:
        formula: Regression formula in format "y ~ x1 + x2 | fe1 + fe2 | endog (inst1 + inst2) | cluster"
        data: Data source.  Accepts a file path (str/Path) to a .parquet, .csv,
            .tsv, .json, .ndjson, .feather/.arrow file, a directory of .parquet
            files, an in-memory pandas/Polars/PyArrow DataFrame, or a DuckDB
            relation object.
        subset: SQL WHERE clause to subset data
        n_bootstraps: Number of bootstrap iterations (only used if se_method="BS")
        cache_dir: Directory for DuckDB cache files
        round_strata: Number of decimals to round strata columns
        seed: Random seed for reproducibility
        fe_method: Method for handling fixed effects ('mundlak' or 'demean');
            both 'mundlak' and 'demean' are dispatched through DuckFE.  ``"mundlak"`` is not
            recommended for unbalanced panels; use ``"demean"`` (iterative demeaning) instead.
        duckdb_kwargs: Dictionary of DuckDB configuration settings
        db_name: Full path to DuckDB database file
        n_jobs: Number of parallel jobs for bootstrap
        se_method: Method for computing standard errors ('iid', 'HC1', 'CRV1', 'BS', or 'none')
        fitter: Estimation method ('numpy' for in-memory, 'duckdb' for out-of-core)
        remove_singletons: If True, remove observations from singleton FE groups (default=True)
        ssc_dict: Optional small-sample correction overrides passed to every analytical
            vcov computation.  Keys: ``k_adj`` (bool), ``k_fixef`` (str: ``"full"`` |                ``"nonnested"`` | ``"none"``), ``G_adj`` (bool), ``G_df`` (str).
            When *None* (default), each vcov function uses fixest-style built-in defaults.
            Only applies to OLS estimators (DuckRegression / DuckFE); ignored for IV.
        **kwargs: Additional arguments passed to estimator

    Returns:
        Fitted estimator object with results
    """
    logger.debug(f"=== duckreg START ===")
    
    # Import estimator classes and formula parser
    from .estimators import DuckRegression, Duck2SLS, DuckFE
    from .utils.formula_parser import FormulaParser
    
    # Parse the lfe-style formula to extract model components
    parsed_formula = FormulaParser().parse(formula)
    fe_cols = parsed_formula.get_fe_names()  # Fixed effect column names
    has_iv = parsed_formula.has_instruments()  # Does model have instrumental variables?
    
    logger.debug(f"Parsed: outcomes={parsed_formula.get_outcome_names()}, "
                 f"covariates={parsed_formula.get_covariate_names()}, "
                 f"fe={fe_cols}, cluster={parsed_formula.cluster}, "
                 f"has_iv={has_iv}, fitter={fitter}")
    
    # Resolve data source: determine DuckDB database path, SQL table reference,
    # and (for in-memory objects) the object to register in the connection.
    resolved_db, table_name, obj_to_register = _resolve_data_source(
        data, cache_dir, db_name
    )
    
    # Resolve fixed effects method
    # For IV models, default to Mundlak if AUTO is specified
    # For OLS with FE, default to iterative demeaning if AUTO is specified
    resolved_fe_method = fe_method
    if fe_method == FEMethod.AUTO:
        resolved_fe_method = FEMethod.MUNDLAK if has_iv else (FEMethod.DEMEAN if fe_cols else None)
    
    # Select appropriate estimator based on model type
    # IV models use Duck2SLS (2-stage least squares)
    # OLS models with FE use DuckFE (transformer-backed); without FE use DuckRegression
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
            remove_singletons=remove_singletons,
            **kwargs
        )
    else:
        # OLS regression: route to DuckFE (transformer-backed) or DuckRegression (no FE)
        # DuckFE dispatches to the appropriate transformer based on method:
        # - 'mundlak': Mundlak device (levels model with group means + Wooldridge correction)
        # - 'iterative_demean': MAP demeaning (correct for any panel structure)
        if fe_cols:
            if resolved_fe_method == FEMethod.MUNDLAK:
                fe_method_str = "mundlak"
            elif resolved_fe_method == FEMethod.DEMEAN:
                fe_method_str = "iterative_demean"
            elif resolved_fe_method == FEMethod.AUTO_FE:
                raise NotImplementedError(
                    "fe_method='auto_fe' is experimental and has been temporarily disabled. "
                    "Use fe_method='demean' (iterative demeaning) instead."
                )
            else:
                raise ValueError(
                    f"With fixed effects, fe_method must be '{FEMethod.MUNDLAK}' "
                    f"or '{FEMethod.DEMEAN}', got '{resolved_fe_method}'"
                )
            estimator = DuckFE(
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
                remove_singletons=remove_singletons,
                method=fe_method_str,
                ssc_dict=ssc_dict,
                **kwargs
            )
        else:
            # No fixed effects: use standard regression
            estimator = DuckRegression(
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
                remove_singletons=remove_singletons,
                ssc_dict=ssc_dict,
                **kwargs
            )
    
    # For in-memory data sources (DataFrame, Arrow, Polars, DuckDB relation),
    # register the object as a view in the already-open DuckDB connection before
    # the fit pipeline touches the table.
    if obj_to_register is not None:
        estimator.conn.register(_DUCKDB_VIEW_NAME, obj_to_register)

    # Fit the model (triggers data preparation, compression, estimation, and SE computation)
    estimator.fit(se_method=se_method)

    logger.debug(f"=== duckreg END ===")
    return estimator


# Backward compatibility alias
compressed_ols = duckreg


