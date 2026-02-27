"""
Base class for linear model estimators.

This module provides the shared functionality for OLS estimators that use
either demeaning or Mundlak device for fixed effects.

Architecture:
- DuckLinearModel extends DuckEstimator
- Results containers are imported from core/results.py (Single Responsibility)
- Bootstrap and vcov helpers are imported from core/vcov.py (DRY)
- SQL builders are imported from core/sql_builders.py (DRY)
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, List, Dict, Any

from ..core.demean import demean, _convert_to_int
from .base import DuckEstimator, SEMethod
from ..core.fitters import wls, NumpyFitter, DuckDBFitter, FitterResult
from ..utils.formula_parser import cast_if_boolean, needs_quoting, quote_identifier, _make_sql_safe_name

# Import from refactored modules - Single Responsibility Principle
from ..core.results import RegressionResults, FirstStageResults
from ..core.vcov import (
    BootstrapExecutor,
    _bootstrap_iteration_iid, _bootstrap_iteration_cluster
)
from ..utils.name_utils import build_coef_name_lists

logger = logging.getLogger(__name__)


# ============================================================================
# Base Linear Model
# ============================================================================

class DuckLinearModel(DuckEstimator):
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
        remove_singletons: bool = True,
        ssc_dict: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        ssc_dict : dict, optional
            Small-sample correction overrides forwarded to every analytical
            vcov computation (iid, HC1, CRV1, …).  Recognised keys:

            * ``k_adj``   – bool, default True.  Apply N/(N−k) adjustment.
            * ``k_fixef`` – str, ``"full"`` (default) | ``"nonnested"`` | ``"none"``.
              Controls how absorbed FE levels count toward df_k.
            * ``G_adj``   – bool, default True.  Apply G/(G−1) cluster factor.
            * ``G_df``    – str, ``"conventional"`` (default) | ``"min"``.

            When *None* (the default), each vcov function uses its own
            built-in defaults, which match fixest conventions.
        """
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
            remove_singletons=remove_singletons,
            **kwargs,
        )
        
        self.formula = formula
        self.n_jobs = n_jobs
        self.subset = subset
        self.ssc_dict = ssc_dict

        # State
        self.strata_cols: List[str] = []
        self._boolean_cols: Optional[set] = None
        self._fitter_result: Optional[FitterResult] = None
        self._data_fetched: bool = False
        self.df_compressed: Optional[pd.DataFrame] = None
        self.agg_query: Optional[str] = None
        self.n_compressed_rows: Optional[int] = None
        self._results: Optional[RegressionResults] = None
        self.vcov_meta: Optional[Dict[str, Any]] = None
        
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



    def _get_cluster_col_for_vcov(self) -> Optional[str]:
        """Get cluster column name for vcov computation. Subclasses may override."""
        return self.cluster_col

    def _get_unit_col(self) -> Optional[str]:
        """Get unit column (first FE) for panel operations"""
        return self.fe_cols[0] if self.fe_cols else None

    def _get_vcov_fe_params(self) -> Tuple[int, int, int, int]:
        """Return (k_fe, n_fe, k_fe_nested, n_fe_fully_nested) for VcovContext.

        Parameters
        ----------
        k_fe : int
            Total number of absorbed fixed-effect levels across all FE
            dimensions (i.e. sum of distinct-level counts per dimension).
            Used together with ``n_fe`` to adjust :math:`N - k - G` degrees
            of freedom inside :class:`~duckreg.core.vcov.VcovContext`.
        n_fe : int
            Number of FE dimensions (i.e. ``len(fe_cols)``).
        k_fe_nested : int
            Total FE levels in fully-nested dimensions (for nonnested SSC).
        n_fe_fully_nested : int
            Number of FE dimensions fully nested within another.

        Default ``(0, 0, 0, 0)`` is correct for:

        * :class:`DuckRegression` – no fixed effects at all.
        * :class:`DuckFE` (Mundlak method) – FE parameters appear as explicit
          regressors and are already counted in ``k``; no additional DOF
          adjustment is needed.

        Must be overridden for :class:`DuckFE` (iterative demean method), where
        fixed effects are absorbed via demeaning and are *not* present in ``X``.
        """
        return 0, 0, 0, 0

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
        
        Must be implemented by subclasses (DuckRegression, DuckFE).
        
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
            cluster_ids=cluster_ids
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
            cluster_col=cluster_col if cluster_col in view_cols else None
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
        """Update coefficient names from fitter result or formula.
        
        For DuckDB fitter, builds names from formula to avoid loading data.
        """
        # If we already have coef_names_, use them
        if hasattr(self, 'coef_names_') and self.coef_names_:
            pass
        # Try to get from fitter result
        elif self._fitter_result is not None and self._fitter_result.coef_names is not None:
            self.coef_names_ = self._fitter_result.coef_names
        # Build from formula for DuckDB fitter (avoids loading data)
        elif self.fitter == "duckdb":
            self.coef_names_ = self._build_coef_names_from_formula()
        
        if len(self.outcome_vars) > 1 and hasattr(self, 'coef_names_') and self.coef_names_:
            self.coef_names_ = [f"{name}:{outcome}" 
                               for outcome in self.outcome_vars 
                               for name in self.coef_names_]
    
    def _build_coef_names_from_formula(self) -> List[str]:
        """Build coefficient names from formula without loading data.
        
        Used by DuckDB fitter path to avoid memory loading.
        
        Returns:
            List of display names for coefficients
        """
        include_intercept = self._needs_intercept_for_duckdb()
        display_names, sql_names = build_coef_name_lists(
            formula=self.formula,
            fe_method='demean',  # DuckLinearModel uses demeaning by default
            include_intercept=include_intercept,
            fe_cols=None,  # No Mundlak means in base class
            is_iv=False
        )
        # Store sql_names for potential use in SQL column selection
        self._coef_sql_names = sql_names
        return display_names

    # -------------------------------------------------------------------------
    # Variance-covariance
    # -------------------------------------------------------------------------

    def fit_vcov(self, se_method: str = "HC1"):
        """Compute variance-covariance matrix.
        
        Args:
            se_method: Type of standard errors ('iid', 'HC1', 'CRV1')
        """
        if self.fitter == "duckdb":
            self._fit_vcov_duckdb(se_method=se_method)
        else:
            self._fit_vcov_numpy(se_method=se_method)
    
    def _fit_vcov_numpy(self, se_method: str = "HC1"):
        """Compute vcov using numpy (in-memory) backend via NumpyFitter"""
        self._ensure_data_fetched()
        
        y, X, n = self.collect_data(data=self.df_compressed)
        cluster_ids = self._get_cluster_ids_from_df()
        k_fe, n_fe, k_fe_nested, n_fe_fully_nested = self._get_vcov_fe_params()
        
        # Use NumpyFitter for vcov computation
        fitter = NumpyFitter(alpha=1e-8, se_type=se_method)
        
        vcov, vcov_meta, _ = fitter.fit_vcov(
            X=X,
            y=y,
            weights=n,
            coefficients=self.point_estimate,
            cluster_ids=cluster_ids,
            vcov_type=se_method,
            coef_names=getattr(self, 'coef_names_', None),
            k_fe=k_fe,
            n_fe=n_fe,
            k_fe_nested=k_fe_nested,
            n_fe_fully_nested=n_fe_fully_nested,
            ssc_dict=self.ssc_dict,
        )

        self.vcov = vcov
        self.vcov_meta = vcov_meta
        self.se = vcov_meta.get('vcov_type_detail', se_method)
        self._results = None

    def _fit_vcov_duckdb(self, se_method: str = "HC1"):
        """Compute vcov using DuckDB (out-of-core) backend"""
        # Option B fix: _get_vcov_fe_params() is called BEFORE the early-return
        # guard so the cached result is only reused when k_fe == 0 (no FE DOF
        # correction is needed).  When k_fe > 0 (DuckFE / iterative_demean),
        # any vcov cached inside DuckDBFitter.fit() was built with k_fe=0 and
        # would underestimate SEs.  Evaluating k_fe first ensures we always fall
        # through to the full DuckDBFitter.fit_vcov() path for FE models,
        # without touching DuckDBFitter or any other module.
        k_fe, n_fe, k_fe_nested, n_fe_fully_nested = self._get_vcov_fe_params()

        # Only reuse the fitter's cached vcov when there are no absorbed FE
        # levels (DuckRegression / no-FE case).  For k_fe > 0 (DuckFE),
        # always recompute via fit_vcov() so the DOF correction is applied.
        if k_fe == 0 and self._fitter_result is not None and self._fitter_result.vcov is not None:
            if self._fitter_result.se_type == se_method or se_method in self._fitter_result.se_type:
                self.vcov = self._fitter_result.vcov
                self.se = self._fitter_result.se_type
                self._results = None
                return
        
        # Compute using DuckDB fitter
        x_cols = self._get_x_cols_for_duckdb()
        y_col = self._get_y_col_for_duckdb()
        cluster_col = self._get_cluster_col_for_vcov()
        view_cols = self._get_view_columns()
        # k_fe, n_fe, k_fe_nested, n_fe_fully_nested already resolved above
        
        duckdb_fitter = DuckDBFitter(conn=self.conn, alpha=1e-8, se_type=se_method)
        vcov, vcov_meta, _ = duckdb_fitter.fit_vcov(
            table_name=self._COMPRESSED_VIEW,
            x_cols=x_cols,
            y_col=y_col,
            weight_col="count",
            add_intercept=self._needs_intercept_for_duckdb(),
            coefficients=self.point_estimate,
            cluster_col=cluster_col if cluster_col in view_cols else None,
            vcov_type=se_method,
            k_fe=k_fe,
            n_fe=n_fe,
            k_fe_nested=k_fe_nested,
            n_fe_fully_nested=n_fe_fully_nested,
            ssc_dict=self.ssc_dict,
        )

        self.vcov = vcov
        self.vcov_meta = vcov_meta
        self.se = vcov_meta.get('vcov_type_detail', se_method)
        self._results = None

    # -------------------------------------------------------------------------
    # Bootstrap
    # -------------------------------------------------------------------------

    def bootstrap(self) -> np.ndarray:
        """Run bootstrap to estimate variance-covariance matrix.
        
        Note: Bootstrap requires loading data into memory and is not compatible
        with the DuckDB fitter's out-of-core processing. Use analytical SEs instead.
        """
        if self.fitter == "duckdb":
            logger.warning(
                "Bootstrap requested with fitter='duckdb'. Bootstrap requires "
                "loading data into memory which defeats the purpose of out-of-core "
                "processing. Using analytical standard errors from DuckDB fitter instead."
            )
            self.fit_vcov()
            return self.vcov
        
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
        """Provide comprehensive and exhaustive results summary.
        
        Returns a dictionary containing all information needed to:
        - Reconstruct the analysis
        - Track provenance (version, timestamp)
        - Identify results from buggy versions
        
        Uses the standardized ModelSummary structure for consistency.
        
        Returns:
            Dictionary with model specification, results, and metadata
        """
        from ..core.results import ModelSummary
        return ModelSummary.from_estimator(self).to_dict()
    
    def summary_df(self) -> pd.DataFrame:
        """Get results as a DataFrame"""
        if self.results is None:
            return pd.DataFrame()
        return self.results.to_dataframe()
    
    def print_summary(self, precision: int = 4):
        """Print formatted results to console using unified formatter."""
        if self.results:
            from ..utils.summary import print_summary as fmt_print
            fmt_print(self.results, precision=precision)
        else:
            print("No results available. Call fit() first.")
    
    def to_tidy_df(self) -> pd.DataFrame:
        """Get results as a tidy DataFrame using unified formatter."""
        if self.results:
            from ..utils.summary import to_tidy_df as fmt_tidy
            return fmt_tidy(self.results)
        return pd.DataFrame()
