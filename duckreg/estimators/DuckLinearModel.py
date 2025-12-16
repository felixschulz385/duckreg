"""
Base class for linear model estimators (DuckRegression, DuckMundlak).

This module provides the shared functionality for OLS estimators that use
either demeaning or Mundlak device for fixed effects.

Architecture:
- DuckLinearModel extends DuckEstimator and composes VCovMixin, SQLBuilderMixin
- Results containers are imported from results.py (Single Responsibility)
- Bootstrap and vcov helpers are imported from mixins.py (DRY)
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, List, Dict, Any

from ..demean import demean, _convert_to_int
from ..duckreg import DuckEstimator, SEMethod
from ..fitters import wls, NumpyFitter, DuckDBFitter, FitterResult
from ..formula_parser import cast_if_boolean, needs_quoting, quote_identifier, _make_sql_safe_name

# Import from refactored modules - Single Responsibility Principle
from .results import RegressionResults, FirstStageResults
from .mixins import (
    VCovMixin, SQLBuilderMixin, BootstrapExecutor,
    _bootstrap_iteration_iid, _bootstrap_iteration_cluster
)

logger = logging.getLogger(__name__)


# ============================================================================
# Base Linear Model
# ============================================================================

class DuckLinearModel(DuckEstimator, SQLBuilderMixin, VCovMixin):
    """Base class for single-stage linear models (OLS, Mundlak).
    
    This class handles:
    - Formula-based model specification
    - Data compression for efficient estimation
    - Coefficient estimation (numpy or duckdb)
    - Vcov computation (analytical or bootstrap)
    
    Subclasses should override:
    - prepare_data(): Create design matrix tables
    - compress_data(): Create compressed views
    - collect_data(): Extract arrays from compressed data
    """
    
    _COMPRESSED_VIEW = "_compressed_view"
    
    def __init__(
        self,
        db_name: str,
        table_name: str,
        seed: int,
        formula=None,
        n_bootstraps: int = 0,
        round_strata: int = None,
        duckdb_kwargs: dict = None,
        subset: str = None,
        n_jobs: int = 1,
        fitter: str = "numpy",
        **kwargs,
    ):
        if formula is None:
            raise ValueError("Formula object is required")
        
        super().__init__(
            db_name=db_name,
            table_name=table_name,
            seed=seed,
            n_bootstraps=n_bootstraps,
            round_strata=round_strata,
            duckdb_kwargs=duckdb_kwargs,
            fitter=fitter,
            **kwargs,
        )
        
        self.formula = formula
        self.n_jobs = n_jobs
        self.subset = subset
        
        # State
        self.strata_cols: List[str] = []
        self._boolean_cols: Optional[set] = None
        self._fitter_result: Optional[FitterResult] = None
        self._data_fetched: bool = False
        self.df_compressed: Optional[pd.DataFrame] = None
        self.agg_query: Optional[str] = None
        self.n_compressed_rows: Optional[int] = None
        self._results: Optional[RegressionResults] = None
        
        # Extract from formula
        self.outcome_vars = formula.get_outcome_names()
        self.covariates = formula.get_covariate_names()
        self.fe_cols = formula.get_fe_names()
        self.cluster_col = formula.cluster.name if formula.cluster else None
        
        if not self.outcome_vars:
            raise ValueError("No outcome variables provided")
        
        logger.debug(f"{self.__class__.__name__}: outcomes={self.outcome_vars}, "
                    f"covariates={self.covariates}, fe={self.fe_cols}, cluster={self.cluster_col}, "
                    f"fitter={fitter}")

    # -------------------------------------------------------------------------
    # Results property
    # -------------------------------------------------------------------------
    
    @property
    def results(self) -> Optional[RegressionResults]:
        """Get regression results as a RegressionResults object"""
        if self._results is not None:
            return self._results
        
        if not hasattr(self, 'point_estimate') or self.point_estimate is None:
            return None
        
        self._results = RegressionResults(
            coefficients=self.point_estimate,
            coef_names=getattr(self, 'coef_names_', []),
            vcov=getattr(self, 'vcov', None),
            n_obs=getattr(self, 'n_obs', None),
            n_compressed=self.n_compressed_rows,
            se_type=getattr(self, 'se', None),
        )
        return self._results

    # -------------------------------------------------------------------------
    # Shared utilities
    # -------------------------------------------------------------------------

    def _get_boolean_columns(self) -> set:
        """Detect BOOLEAN columns in source table (cached)"""
        if self._boolean_cols is not None:
            return self._boolean_cols
        
        all_columns = set(self.formula.get_source_columns_for_null_check())
        cols_sql = ', '.join(f"'{col}'" for col in all_columns)
        
        query = f"""
        SELECT column_name FROM (DESCRIBE SELECT * FROM {self.table_name})
        WHERE column_name IN ({cols_sql}) AND column_type = 'BOOLEAN'
        """
        self._boolean_cols = set(self.conn.execute(query).fetchdf()['column_name'].tolist())
        return self._boolean_cols

    def _get_cluster_col_for_vcov(self) -> Optional[str]:
        """Get cluster column name for vcov computation. Subclasses may override."""
        return self.cluster_col

    def _get_unit_col(self) -> Optional[str]:
        """Get unit column (first FE) for panel operations"""
        return self.fe_cols[0] if self.fe_cols else None

    def _ensure_data_fetched(self):
        """Fetch compressed data to memory if not already done"""
        if self._data_fetched:
            return
        
        if self.agg_query is None:
            raise ValueError("compress_data must be called before fetching data")
        
        self.df_compressed = self.conn.execute(
            f"SELECT * FROM {self._COMPRESSED_VIEW}"
        ).fetchdf()
        self._data_fetched = True
        self._compute_means()

    def _compute_means(self):
        """Compute mean columns from sum columns"""
        for outcome_var in self.formula.outcomes:
            sum_col = f"sum_{outcome_var.sql_name}"
            mean_col = f"mean_{outcome_var.sql_name}"
            
            if mean_col not in self.df_compressed.columns and sum_col in self.df_compressed.columns:
                self.df_compressed[mean_col] = (
                    self.df_compressed[sum_col] / self.df_compressed["count"]
                )

    def _create_compressed_view(self):
        """Create a view for compressed data without fetching"""
        if self.agg_query is None:
            raise ValueError("agg_query must be set before creating view")
        
        self.conn.execute(f"CREATE OR REPLACE VIEW {self._COMPRESSED_VIEW} AS {self.agg_query}")
        
        result = self.conn.execute(f"""
            SELECT SUM(count) as n_obs, COUNT(*) as n_compressed 
            FROM {self._COMPRESSED_VIEW}
        """).fetchone()
        self.n_obs = int(result[0]) if result[0] else 0
        self.n_compressed_rows = int(result[1]) if result[1] else 0

    def _get_view_columns(self) -> List[str]:
        """Get column names from compressed view"""
        return self.conn.execute(
            f"SELECT column_name FROM (DESCRIBE SELECT * FROM {self._COMPRESSED_VIEW})"
        ).fetchdf()['column_name'].tolist()

    # -------------------------------------------------------------------------
    # Abstract method to be implemented by subclasses
    # -------------------------------------------------------------------------

    def collect_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract y, X, weights arrays from compressed data.
        
        Must be implemented by subclasses (DuckRegression, DuckMundlak).
        
        Returns:
            Tuple of (y, X, weights) arrays
        """
        raise NotImplementedError("Subclasses must implement collect_data")

    # -------------------------------------------------------------------------
    # Estimation
    # -------------------------------------------------------------------------

    def estimate(self) -> np.ndarray:
        """Estimate coefficients using WLS"""
        if self.fitter == "duckdb":
            return self._estimate_duckdb()
        return self._estimate_numpy()
    
    def _estimate_numpy(self) -> np.ndarray:
        """Estimate using in-memory numpy WLS"""
        self._ensure_data_fetched()
        
        y, X, n = self.collect_data(data=self.df_compressed)
        cluster_ids = self._get_cluster_ids_from_df()
        
        numpy_fitter = NumpyFitter(alpha=1e-8, se_type="stata")
        self._fitter_result = numpy_fitter.fit(
            X=X, y=y, weights=n,
            coef_names=getattr(self, 'coef_names_', None),
            cluster_ids=cluster_ids,
            compute_vcov=True
        )
        
        self._update_coef_names()
        return self._fitter_result.coefficients
    
    def _estimate_duckdb(self) -> np.ndarray:
        """Estimate using DuckDB sufficient statistics"""
        x_cols = self._get_x_cols_for_duckdb()
        y_col = self._get_y_col_for_duckdb()
        cluster_col = self._get_cluster_col_for_vcov()
        view_cols = self._get_view_columns()
        
        duckdb_fitter = DuckDBFitter(conn=self.conn, alpha=1e-8, se_type="stata")
        self._fitter_result = duckdb_fitter.fit(
            table_name=self._COMPRESSED_VIEW,
            x_cols=x_cols,
            y_col=y_col,
            weight_col="count",
            add_intercept=self._needs_intercept_for_duckdb(),
            cluster_col=cluster_col if cluster_col in view_cols else None,
            compute_vcov=True
        )
        
        self._update_coef_names()
        return self._fitter_result.coefficients
    
    def _get_y_col_for_duckdb(self) -> str:
        """Get the y column name for DuckDB fitter"""
        outcome_sql_name = self.formula.get_outcome_sql_names()[0]
        col_name = f"sum_{outcome_sql_name}"
        return quote_identifier(col_name) if needs_quoting(col_name) else col_name

    def _get_x_cols_for_duckdb(self) -> List[str]:
        """Get x column names for DuckDB fitter"""
        non_intercept_sql_names = [var.sql_name for var in self.formula.covariates if not var.is_intercept()]
        return [quote_identifier(col) if needs_quoting(col) else col for col in non_intercept_sql_names]
    
    def _needs_intercept_for_duckdb(self) -> bool:
        """Determine if DuckDBFitter should add an intercept."""
        return not self.fe_cols

    def _get_cluster_ids_from_df(self) -> Optional[np.ndarray]:
        """Get cluster IDs from compressed dataframe"""
        cluster_col = self._get_cluster_col_for_vcov()
        if cluster_col and self.df_compressed is not None and cluster_col in self.df_compressed.columns:
            return self.df_compressed[cluster_col].values
        return None

    def _update_coef_names(self):
        """Update coefficient names from fitter result"""
        if self._fitter_result is not None and self._fitter_result.coef_names is not None:
            if not hasattr(self, 'coef_names_') or self.coef_names_ is None:
                self.coef_names_ = self._fitter_result.coef_names
        
        if len(self.outcome_vars) > 1 and hasattr(self, 'coef_names_') and self.coef_names_:
            self.coef_names_ = [f"{name}:{outcome}" 
                               for outcome in self.outcome_vars 
                               for name in self.coef_names_]

    # -------------------------------------------------------------------------
    # Variance-covariance
    # -------------------------------------------------------------------------

    def fit_vcov(self):
        """Compute variance-covariance matrix"""
        # If duckdb fitter already computed vcov, use it
        if self.fitter == "duckdb" and self._fitter_result is not None and self._fitter_result.vcov is not None:
            self.vcov = self._fitter_result.vcov
            self.se = self._fitter_result.se_type
            return
        
        self._ensure_data_fetched()
        
        y, X, n = self.collect_data(data=self.df_compressed)
        cluster_ids = self._get_cluster_ids_from_df()
        
        self.vcov, self.se, _ = self.compute_vcov_from_data(
            X=X, y=y, weights=n, cluster_ids=cluster_ids
        )
        self._results = None

    # -------------------------------------------------------------------------
    # Bootstrap
    # -------------------------------------------------------------------------

    def bootstrap(self) -> np.ndarray:
        """Run bootstrap to estimate variance-covariance matrix"""
        self._ensure_data_fetched()
        executor = BootstrapExecutor(self.n_bootstraps, self.n_jobs, self.rng)
        
        if self.cluster_col:
            boot_coefs, boot_sizes = self._run_cluster_bootstrap(executor)
        else:
            boot_coefs, boot_sizes = self._run_iid_bootstrap(executor)
        
        vcov = np.cov(boot_coefs.T, aweights=boot_sizes)
        self._results = None
        return np.expand_dims(vcov, axis=0) if vcov.ndim == 0 else vcov

    def _run_iid_bootstrap(self, executor: BootstrapExecutor) -> Tuple[np.ndarray, np.ndarray]:
        y, X, n = self.collect_data(data=self.df_compressed)
        n_rows = len(self.df_compressed)
        
        return executor.execute(
            _bootstrap_iteration_iid, (X, y, n, n_rows),
            args_builder=lambda b, seed: (X, y, n, n_rows, seed)
        )

    def _run_cluster_bootstrap(self, executor: BootstrapExecutor) -> Tuple[np.ndarray, np.ndarray]:
        df_clusters, cluster_col_name = self._get_cluster_data_for_bootstrap()
        df_clusters = df_clusters.dropna(subset=[cluster_col_name])
        
        unique_groups = df_clusters[cluster_col_name].unique()
        group_to_idx = {x: i for i, x in enumerate(unique_groups)}
        group_idx = df_clusters[cluster_col_name].map(group_to_idx).to_numpy(dtype=int)
        
        y, X, n = self.collect_data(data=df_clusters)
        n_unique_groups = len(unique_groups)
        
        return executor.execute(
            _bootstrap_iteration_cluster, (X, y, n, group_idx, n_unique_groups),
            args_builder=lambda b, seed: (X, y, n, group_idx, n_unique_groups, seed)
        )

    def _get_cluster_data_for_bootstrap(self) -> Tuple[pd.DataFrame, str]:
        """Get data and cluster column for bootstrap. Subclasses may override."""
        self._ensure_data_fetched()
        return self.df_compressed, self.cluster_col

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Provide comprehensive results summary"""
        return {
            "coefficients": self.results.to_dict() if self.results else None,
            "n_obs": getattr(self, 'n_obs', None),
            "n_compressed": self.n_compressed_rows,
            "estimator_type": self.__class__.__name__,
            "outcome_vars": self.outcome_vars,
            "covariates": self.covariates,
            "fe_cols": self.fe_cols,
            "cluster_col": self.cluster_col,
        }
    
    def summary_df(self) -> pd.DataFrame:
        """Get results as a DataFrame"""
        if self.results is None:
            return pd.DataFrame()
        return self.results.to_dataframe()
    
    def print_summary(self, precision: int = 4):
        """Print formatted results to console using unified formatter."""
        if self.results:
            fmt_print(self.results, precision=precision)
        else:
            print("No results available. Call fit() first.")
    
    def to_tidy_df(self) -> pd.DataFrame:
        """Get results as a tidy DataFrame using unified formatter."""
        if self.results:
            return fmt_tidy(self.results)
        return pd.DataFrame()
