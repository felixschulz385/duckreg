import os
import sys
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List
import hashlib

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Constants (centralized for all modules)
# ============================================================================

class SEMethod:
    """Standard error computation methods"""
    ANALYTICAL = "analytical"
    BOOTSTRAP = "bootstrap"
    NONE = "none"


class FEMethod:
    """Fixed effects handling methods"""
    MUNDLAK = "mundlak"
    DEMEAN = "demean"
    AUTO = "auto"


# ============================================================================
# Data Source Utilities
# ============================================================================

_FILE_READERS = {".csv": "read_csv", ".parquet": "read_parquet"}


def _resolve_table_name(data_path: Path) -> str:
    """Create DuckDB table reference from data path"""
    if data_path.is_file():
        suffix = data_path.suffix.lower()
        if suffix not in _FILE_READERS:
            raise ValueError(f"Unsupported file format: {suffix}. Supported: {list(_FILE_READERS.keys())}")
        return f"{_FILE_READERS[suffix]}('{data_path}')"
    elif data_path.is_dir():
        return f"read_parquet('{data_path}/**/*.parquet')"
    raise ValueError(f"Data path not found: {data_path}")


def _resolve_db_path(data: str, cache_dir: Optional[str], db_name: Optional[str]) -> str:
    """Resolve database path from inputs"""
    if db_name is not None:
        db_path = Path(db_name)
        db_path.parent.mkdir(exist_ok=True, parents=True)
        return str(db_path)
    
    data_path = Path(data).resolve()
    cache_dir = Path(cache_dir) if cache_dir else (
        data_path.parent if data_path.is_file() else data_path
    ) / ".duckreg"
    
    cache_dir.mkdir(exist_ok=True, parents=True)
    data_hash = hashlib.md5(str(data_path).encode()).hexdigest()[:8]
    return str(cache_dir / f"duckreg_{data_hash}.db")


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
    se_method: str = SEMethod.ANALYTICAL,
    fitter: str = "numpy",
    **kwargs
) -> "DuckEstimator":
    """High-level API for compressed OLS regression with lfe-style formula
    
    Args:
        formula: Regression formula in format "y ~ x1 + x2 | fe1 + fe2 | endog (inst1 + inst2) | cluster"
        data: Filepath to data file (.csv, .parquet, or directory of .parquet files)
        subset: SQL WHERE clause to subset data
        n_bootstraps: Number of bootstrap iterations (only used if se_method="bootstrap")
        cache_dir: Directory for DuckDB cache files
        round_strata: Number of decimals to round strata columns
        seed: Random seed for reproducibility
        fe_method: Method for handling fixed effects ('mundlak' or 'demean')
        duckdb_kwargs: Dictionary of DuckDB configuration settings
        db_name: Full path to DuckDB database file
        n_jobs: Number of parallel jobs for bootstrap
        se_method: Method for computing standard errors ('analytical', 'bootstrap', or 'none')
        fitter: Estimation method ('numpy' for in-memory, 'duckdb' for out-of-core)
        **kwargs: Additional arguments passed to estimator
    
    Returns:
        Fitted estimator object
    """
    logger.debug(f"=== compressed_ols START ===")
    
    from .estimators import DuckRegression, DuckMundlak, Duck2SLS
    from .formula_parser import FormulaParser
    
    parsed_formula = FormulaParser().parse(formula)
    fe_cols = parsed_formula.get_fe_names()
    has_iv = parsed_formula.has_instruments()
    
    logger.debug(f"Parsed: outcomes={parsed_formula.get_outcome_names()}, "
                 f"covariates={parsed_formula.get_covariate_names()}, "
                 f"fe={fe_cols}, cluster={parsed_formula.cluster}, "
                 f"has_iv={has_iv}, fitter={fitter}")
    
    # Setup paths
    resolved_db = _resolve_db_path(data, cache_dir, db_name)
    table_name = _resolve_table_name(Path(data).resolve())
    
    # Resolve fe_method for 2SLS
    resolved_fe_method = fe_method
    if fe_method == FEMethod.AUTO:
        resolved_fe_method = FEMethod.DEMEAN
    
    # Determine estimator class
    if has_iv:
        estimator = Duck2SLS(
            db_name=resolved_db,
            table_name=table_name,
            formula=parsed_formula,
            subset=subset,
            n_bootstraps=n_bootstraps if se_method == SEMethod.BOOTSTRAP else 0,
            round_strata=round_strata,
            seed=seed,
            duckdb_kwargs=duckdb_kwargs,
            n_jobs=n_jobs,
            fitter=fitter,
            fe_method=resolved_fe_method,
            **kwargs
        )
    else:
        use_mundlak = bool(fe_cols) and fe_method == FEMethod.MUNDLAK
        if fe_cols and fe_method not in (FEMethod.MUNDLAK, FEMethod.DEMEAN, FEMethod.AUTO):
            raise ValueError(f"fe_method must be '{FEMethod.MUNDLAK}' or '{FEMethod.DEMEAN}'")
        
        estimator_class = DuckMundlak if use_mundlak else DuckRegression
        estimator = estimator_class(
            db_name=resolved_db,
            table_name=table_name,
            formula=parsed_formula,
            subset=subset,
            n_bootstraps=n_bootstraps if se_method == SEMethod.BOOTSTRAP else 0,
            round_strata=round_strata,
            seed=seed,
            duckdb_kwargs=duckdb_kwargs,
            n_jobs=n_jobs,
            fitter=fitter,
            **kwargs
        )
    
    estimator.fit(se_method=se_method)
    
    logger.debug(f"=== compressed_ols END ===")
    return estimator


# ============================================================================
# Base Estimator Class
# ============================================================================

class DuckEstimator(ABC):
    """Abstract base class for all DuckDB-based estimators.
    
    This provides the minimal interface that all estimators must implement,
    plus shared DuckDB connection management.
    """
    
    def __init__(
        self,
        db_name: str,
        table_name: str,
        seed: int,
        n_bootstraps: int = 0,
        fitter: str = "numpy",
        keep_connection_open: bool = False,
        round_strata: int = None,
        duckdb_kwargs: dict = None,
    ):
        logger.debug(f"DuckEstimator.__init__: db={db_name}, table={table_name}")
        
        self.db_name = db_name
        self.table_name = table_name
        self.n_bootstraps = n_bootstraps
        self.seed = seed
        self.fitter = fitter
        self.keep_connection_open = keep_connection_open
        self.round_strata = round_strata
        self.duckdb_kwargs = duckdb_kwargs
        
        # State
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self.rng: Optional[np.random.Generator] = None
        self.point_estimate: Optional[np.ndarray] = None
        self.vcov: Optional[np.ndarray] = None
        self.se: Optional[str] = None
        self.coef_names_: Optional[List[str]] = None
        self.n_obs: Optional[int] = None
        
        self._init_connection()

    def _init_connection(self):
        """Initialize DuckDB connection and RNG"""
        self.conn = duckdb.connect(self.db_name)
        self._apply_duckdb_config(self.duckdb_kwargs)
        self.rng = np.random.default_rng(self.seed)

    def _apply_duckdb_config(self, config: Optional[Dict[str, Any]]):
        """Apply DuckDB configuration settings"""
        if config:
            for key, value in config.items():
                self.conn.execute(f"SET {key} = '{value}'")

    def fit(self, se_method: str = SEMethod.ANALYTICAL):
        """Main fitting method - orchestrates the estimation pipeline.
        
        Subclasses should not override this; override the individual steps instead.
        """
        logger.debug(f"fit() START with se_method={se_method}")
        
        # Step 1: Prepare data (create tables, run first stages for IV, etc.)
        self.prepare_data()
        
        # Step 2: Compress data for efficient estimation
        self.compress_data()
        
        # Step 3: Estimate coefficients
        self.point_estimate = self.estimate()
        
        # Step 4: Compute standard errors
        self._compute_standard_errors(se_method)
        
        # Cleanup
        if not self.keep_connection_open:
            self.conn.close()
        
        logger.debug(f"fit() END")

    def _compute_standard_errors(self, se_method: str):
        """Dispatch standard error computation based on method"""
        if se_method == SEMethod.BOOTSTRAP:
            if self.n_bootstraps > 0:
                logger.debug("Computing bootstrap standard errors")
                self.vcov = self.bootstrap()
                self.se = "bootstrap"
        elif se_method == SEMethod.ANALYTICAL:
            logger.debug("Computing analytical standard errors")
            self.fit_vcov()
        elif se_method == SEMethod.NONE:
            logger.debug("Skipping standard error computation")
        else:
            logger.warning(f"Unknown se_method '{se_method}'")

    # -------------------------------------------------------------------------
    # Abstract methods - must be implemented by subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    def prepare_data(self):
        """Prepare data tables for estimation.
        
        This may include:
        - Creating design matrices
        - Running first-stage regressions (for IV)
        - Computing Mundlak means (for Mundlak approach)
        """
        pass

    @abstractmethod
    def compress_data(self):
        """Compress data for efficient estimation.
        
        Creates aggregated views/tables with sufficient statistics.
        """
        pass

    @abstractmethod
    def estimate(self) -> np.ndarray:
        """Estimate model coefficients.
        
        Returns:
            Array of coefficient estimates
        """
        pass

    @abstractmethod
    def fit_vcov(self):
        """Compute variance-covariance matrix analytically."""
        pass

    @abstractmethod
    def bootstrap(self) -> np.ndarray:
        """Compute variance-covariance matrix via bootstrap.
        
        Returns:
            Variance-covariance matrix
        """
        pass

    # -------------------------------------------------------------------------
    # Common utility methods
    # -------------------------------------------------------------------------

    def _get_table_columns(self, table_name: str) -> set:
        """Get column names from a table"""
        return set(
            self.conn.execute(f"SELECT column_name FROM (DESCRIBE {table_name})")
            .fetchdf()['column_name'].tolist()
        )

    def _build_where_clause(self, user_subset: Optional[str] = None) -> str:
        """Build WHERE clause with NULL checks and optional user subset"""
        if hasattr(self, 'formula'):
            return self.formula.get_where_clause_sql(user_subset)
        return f"WHERE ({user_subset})" if user_subset else ""

    def summary(self) -> Dict[str, Any]:
        """Provide results summary. Subclasses should override for richer output."""
        return {
            "point_estimate": self.point_estimate,
            "coef_names": self.coef_names_,
            "n_obs": self.n_obs,
            "se_type": self.se,
        }


# ============================================================================
# Backward compatibility alias
# ============================================================================

# Keep DuckReg as an alias for backward compatibility
DuckReg = DuckEstimator