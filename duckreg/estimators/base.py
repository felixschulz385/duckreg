"""Base estimator class for all DuckDB-based estimators"""
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

import duckdb
import numpy as np

logger = logging.getLogger(__name__)


class SEMethod:
    """Standard error computation methods"""
    IID = "iid"
    HC1 = "HC1"
    CRV1 = "CRV1"
    BS = "BS"
    NONE = "none"


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
        remove_singletons: bool = True,
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
        self.remove_singletons = remove_singletons
        
        # State
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self.rng: Optional[np.random.Generator] = None
        self.point_estimate: Optional[np.ndarray] = None
        self.vcov: Optional[np.ndarray] = None
        self.se: Optional[str] = None
        self.coef_names_: Optional[List[str]] = None
        self.n_obs: Optional[int] = None
        self.n_rows_dropped_singletons: int = 0
        
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

    def fit(self, se_method: str = SEMethod.IID):
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
        # When vcov_spec is set (via duckreg API), derive the effective method from it.
        # This ensures the parsed VcovSpec is used rather than the string fallback.
        vcov_spec = getattr(self, 'vcov_spec', None)
        effective = vcov_spec.vcov_detail if vcov_spec is not None else se_method

        if self.n_bootstraps > 0 and (effective == SEMethod.BS or se_method == SEMethod.BS):
            logger.debug("Computing bootstrap standard errors")
            self.vcov = self.bootstrap()
            self.se = "bootstrap"
        elif effective == SEMethod.NONE:
            logger.debug("Skipping standard error computation")
        elif effective in (SEMethod.IID, SEMethod.HC1, SEMethod.CRV1,
                           'HC2', 'HC3', 'CRV3', 'hetero', 'iid'):
            logger.debug(f"Computing {effective} standard errors")
            self.fit_vcov()
        else:
            logger.warning(f"Unknown se_method '{effective}'")

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
    def fit_vcov(self, se_method: str = SEMethod.HC1):
        """Compute variance-covariance matrix."""
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

    def _get_boolean_columns(self) -> set:
        """Get boolean columns from source table (cached)."""
        if hasattr(self, '_boolean_cols') and self._boolean_cols is not None:
            return self._boolean_cols
        
        all_cols = set(self.formula.get_source_columns_for_null_check())
        cols_sql = ', '.join(f"'{c}'" for c in all_cols)
        query = f"""
        SELECT column_name FROM (DESCRIBE SELECT * FROM {self.table_name})
        WHERE column_name IN ({cols_sql}) AND column_type = 'BOOLEAN'
        """
        self._boolean_cols = set(self.conn.execute(query).fetchdf()['column_name'].tolist())
        return self._boolean_cols

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

    def _build_qualify_singleton_filter(self, fe_col_sql_names: List[str]) -> str:
        """Build QUALIFY clause to exclude singleton groups from multiple FE columns.
        
        Uses window functions for efficiency - single pass filter vs multiple DELETE statements.
        A singleton group is one with exactly one observation.
        
        Args:
            fe_col_sql_names: List of SQL names of fixed effects columns
            
        Returns:
            QUALIFY clause excluding singletons (empty string if remove_singletons=False)
        """
        if not self.remove_singletons or not fe_col_sql_names:
            return ""
        
        # Build window function for all FE columns
        partition_clause = ', '.join(fe_col_sql_names)
        return f"QUALIFY count(*) OVER (PARTITION BY {partition_clause}) > 1"

    def _remove_singleton_observations(self, table_name: str, fe_col_sql_names: List[str]):
        """Remove observations from singleton FE groups if remove_singletons=True.
        
        Strategy:
        Uses ANTI JOIN to exclude observations from singleton groups (groups with < 2 obs).
        Processes each FE dimension sequentially since removing singletons in one dimension
        can create new singletons in another dimension.
        
        An observation is removed if it belongs to a singleton group in ANY FE dimension.
        
        Args:
            table_name: Name of the table to filter
            fe_col_sql_names: List of SQL names of fixed effects columns
        """
        if not self.remove_singletons or not fe_col_sql_names:
            return
        
        logger.debug(f"Removing singleton FE observations from {len(fe_col_sql_names)} FE groups")
        
        rows_before = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        # Process each FE dimension sequentially
        for fe_sql in fe_col_sql_names:
            # Use ANTI JOIN to exclude singleton groups directly
            self.conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT *
            FROM {table_name}
            ANTI JOIN (
                SELECT {fe_sql}
                FROM {table_name}
                GROUP BY {fe_sql}
                HAVING COUNT(*) < 2
            ) singletons
            USING ({fe_sql})
            """)
        
        rows_after = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        self.n_rows_dropped_singletons = rows_before - rows_after
        
        logger.debug(f"After singleton removal: {rows_after} observations "
                    f"({self.n_rows_dropped_singletons} rows removed)")

    def summary(self) -> Dict[str, Any]:
        """Provide results summary. Subclasses should override for richer output."""
        return {
            "point_estimate": self.point_estimate,
            "coef_names": self.coef_names_,
            "n_obs": self.n_obs,
            "se_type": self.se,
        }


# Backward compatibility alias
DuckReg = DuckEstimator
