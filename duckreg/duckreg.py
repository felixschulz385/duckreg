import os
import sys
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib

import duckdb
import numpy as np

# Configure logging for debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/duckreg_debug.log', mode='a')
    ]
)
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
) -> "DuckReg":
    """High-level API for compressed OLS regression with lfe-style formula
    
    Args:
        formula: Regression formula in format "y ~ x1 + x2 | fe1 + fe2 | iv1 + iv2 | cluster"
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
    
    from .estimators import DuckRegression, DuckMundlak
    from .formula_parser import FormulaParser
    
    parsed_formula = FormulaParser().parse(formula)
    fe_cols = parsed_formula.get_fe_names()
    
    logger.debug(f"Parsed: outcomes={parsed_formula.get_outcome_names()}, "
                 f"covariates={parsed_formula.get_covariate_names()}, "
                 f"fe={fe_cols}, cluster={parsed_formula.cluster}, fitter={fitter}")
    
    # Determine strategy
    use_mundlak = bool(fe_cols) and fe_method == FEMethod.MUNDLAK
    if fe_cols and fe_method not in (FEMethod.MUNDLAK, FEMethod.DEMEAN):
        raise ValueError(f"fe_method must be '{FEMethod.MUNDLAK}' or '{FEMethod.DEMEAN}'")
    
    # Setup paths
    resolved_db = _resolve_db_path(data, cache_dir, db_name)
    table_name = _resolve_table_name(Path(data).resolve())
    
    # Create and fit estimator
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
# Base class
# ============================================================================

class DuckReg(ABC):
    """Base class for DuckDB-based regression estimators"""
    
    def __init__(
        self,
        db_name: str,
        table_name: str,
        seed: int,
        n_bootstraps: int = 100,
        fitter: str = "numpy",
        keep_connection_open: bool = False,
        round_strata: int = None,
        duckdb_kwargs: dict = None,
        variable_casts: dict = None,
    ):
        logger.debug(f"DuckReg.__init__: db={db_name}, table={table_name}")
        
        self.db_name = db_name
        self.table_name = table_name
        self.n_bootstraps = n_bootstraps
        self.seed = seed
        self.fitter = fitter
        self.keep_connection_open = keep_connection_open
        self.round_strata = round_strata
        self.duckdb_kwargs = duckdb_kwargs
        self.variable_casts = variable_casts or {}
        
        self.conn = duckdb.connect(db_name)
        self._apply_duckdb_config(duckdb_kwargs)
        self.rng = np.random.default_rng(seed)

    def _apply_duckdb_config(self, config: Optional[Dict[str, Any]]):
        """Apply DuckDB configuration settings"""
        if config:
            for key, value in config.items():
                self.conn.execute(f"SET {key} = '{value}'")

    def _cast_col(self, col: str) -> str:
        """Apply casting to column if specified"""
        return f"CAST({col} AS {self.variable_casts[col]})" if col in self.variable_casts else col

    def fit(self, se_method: str = SEMethod.ANALYTICAL):
        """Fit the model: prepare data, compress, estimate, and compute standard errors"""
        logger.debug(f"fit() START with se_method={se_method}")
        
        self.prepare_data()
        self.compress_data()
        self.point_estimate = self.estimate()
        self._compute_standard_errors(se_method)
        
        if not self.keep_connection_open:
            self.conn.close()
        
        logger.debug(f"fit() END")

    def _compute_standard_errors(self, se_method: str):
        """Compute standard errors based on method"""
        handlers = {
            SEMethod.BOOTSTRAP: self._compute_bootstrap_se,
            SEMethod.ANALYTICAL: self._compute_analytical_se,
            SEMethod.NONE: lambda: logger.debug("Skipping standard error computation"),
        }
        handler = handlers.get(se_method)
        if handler:
            handler()
        else:
            logger.warning(f"Unknown se_method '{se_method}'")

    def _compute_bootstrap_se(self):
        if self.n_bootstraps > 0:
            logger.debug("Computing bootstrap standard errors")
            self.vcov = self.bootstrap()

    def _compute_analytical_se(self):
        logger.debug("Computing analytical standard errors")
        if hasattr(self, 'fit_vcov'):
            self.fit_vcov()
        else:
            logger.warning(f"{self.__class__.__name__} does not support analytical SEs")

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def compress_data(self):
        pass

    @abstractmethod
    def collect_data(self):
        pass

    @abstractmethod
    def estimate(self):
        pass

    @abstractmethod
    def bootstrap(self):
        pass

    def summary(self) -> dict:
        """Summary of regression"""
        result = {"point_estimate": self.point_estimate}
        if self.n_bootstraps > 0 and hasattr(self, 'vcov'):
            result["standard_error"] = np.sqrt(np.diag(self.vcov))
        return result

    def queries(self) -> dict:
        """Collect all query methods in the class"""
        return {name: getattr(self, name) for name in dir(self) if "query" in name}