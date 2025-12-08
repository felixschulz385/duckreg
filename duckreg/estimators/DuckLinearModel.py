import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import sys
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field

from ..demean import demean, _convert_to_int
from ..duckreg import DuckReg
from ..fitters import wls, NumpyFitter, DuckDBFitter, FitterResult
from ..formula_parser import cast_if_boolean, needs_quoting, quote_identifier, _make_sql_safe_name

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG)


# ============================================================================
# Result containers
# ============================================================================

@dataclass
class RegressionResults:
    """Container for regression results with computed statistics"""
    coefficients: np.ndarray
    coef_names: List[str]
    vcov: Optional[np.ndarray] = None
    n_obs: Optional[int] = None
    n_compressed: Optional[int] = None
    se_type: Optional[str] = None
    
    @property
    def std_errors(self) -> Optional[np.ndarray]:
        if self.vcov is None:
            return None
        return np.sqrt(np.diag(self.vcov))
    
    @property
    def t_stats(self) -> Optional[np.ndarray]:
        se = self.std_errors
        if se is None:
            return None
        return self.coefficients.flatten() / se
    
    @property
    def p_values(self) -> Optional[np.ndarray]:
        t = self.t_stats
        if t is None:
            return None
        from scipy import stats
        return 2 * (1 - stats.norm.cdf(np.abs(t)))
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        data = {'coefficient': self.coefficients.flatten()}
        if self.std_errors is not None:
            data['std_error'] = self.std_errors
            data['t_stat'] = self.t_stats
            data['p_value'] = self.p_values
            data['ci_lower'] = self.coefficients.flatten() - 1.96 * self.std_errors
            data['ci_upper'] = self.coefficients.flatten() + 1.96 * self.std_errors
        return pd.DataFrame(data, index=self.coef_names)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            'coefficients': self.coefficients.flatten().tolist(),
            'coef_names': self.coef_names,
            'n_obs': self.n_obs,
            'n_compressed': self.n_compressed,
        }
        if self.vcov is not None:
            result.update({
                'vcov': self.vcov.tolist(),
                'std_errors': self.std_errors.tolist(),
                't_statistics': self.t_stats.tolist(),
                'p_values': self.p_values.tolist(),
                'se_type': self.se_type,
            })
        return result


@dataclass
class FirstStageResults:
    """Container for first-stage regression results with instrument diagnostics"""
    endog_var: str
    results: RegressionResults
    instrument_names: List[str]
    
    # Computed on demand
    _f_stat: Optional[float] = field(default=None, repr=False)
    _f_pvalue: Optional[float] = field(default=None, repr=False)
    
    @property
    def coefficients(self) -> np.ndarray:
        return self.results.coefficients
    
    @property
    def coef_names(self) -> List[str]:
        return self.results.coef_names
    
    @property
    def vcov(self) -> Optional[np.ndarray]:
        return self.results.vcov
    
    @property
    def n_obs(self) -> Optional[int]:
        return self.results.n_obs
    
    def compute_f_statistic(self) -> Tuple[Optional[float], Optional[float]]:
        """Compute F-statistic for joint significance of instruments"""
        if self.vcov is None:
            return None, None
        
        # Find instrument indices
        inst_indices = [i for i, name in enumerate(self.coef_names) if name in self.instrument_names]
        if not inst_indices:
            return None, None
        
        coefs = self.coefficients.flatten()
        inst_coefs = coefs[inst_indices]
        inst_vcov = self.vcov[np.ix_(inst_indices, inst_indices)]
        
        try:
            inst_vcov_inv = np.linalg.inv(inst_vcov)
            wald_stat = inst_coefs @ inst_vcov_inv @ inst_coefs
            n_inst = len(inst_indices)
            f_stat = wald_stat / n_inst
            
            # P-value
            from scipy import stats
            df1 = n_inst
            df2 = (self.n_obs or 1000) - len(coefs)
            f_pvalue = 1 - stats.f.cdf(f_stat, df1, df2)
            
            self._f_stat = float(f_stat)
            self._f_pvalue = float(f_pvalue)
            return self._f_stat, self._f_pvalue
        except np.linalg.LinAlgError:
            return None, None
    
    @property
    def f_statistic(self) -> Optional[float]:
        if self._f_stat is None:
            self.compute_f_statistic()
        return self._f_stat
    
    @property
    def f_pvalue(self) -> Optional[float]:
        if self._f_pvalue is None:
            self.compute_f_statistic()
        return self._f_pvalue
    
    @property
    def is_weak_instrument(self) -> Optional[bool]:
        """Check if F < 10 (Stock-Yogo rule of thumb)"""
        f = self.f_statistic
        return f < 10 if f is not None else None
    
    def get_instrument_stats(self) -> Dict[str, Dict[str, float]]:
        """Get individual instrument coefficient statistics"""
        if self.vcov is None:
            return {}
        
        coefs = self.coefficients.flatten()
        se = self.results.std_errors
        t_stats = self.results.t_stats
        p_vals = self.results.p_values
        
        stats = {}
        for i, name in enumerate(self.coef_names):
            if name in self.instrument_names:
                stats[name] = {
                    'coefficient': float(coefs[i]),
                    'std_error': float(se[i]),
                    't_statistic': float(t_stats[i]),
                    'p_value': float(p_vals[i]),
                }
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            'endogenous_variable': self.endog_var,
            **self.results.to_dict(),
            'instrument_names': self.instrument_names,
            'f_statistic': self.f_statistic,
            'f_pvalue': self.f_pvalue,
            'is_weak_instrument': self.is_weak_instrument,
            'instrument_statistics': self.get_instrument_stats(),
        }
        return result


# ============================================================================
# Bootstrap iteration functions (module-level for parallel execution)
# ============================================================================

def _bootstrap_iteration_iid(args: Tuple) -> Tuple[np.ndarray, float]:
    """Single bootstrap iteration for IID bootstrap (parallelizable)"""
    b, X, y, n, n_rows, seed = args
    rng = np.random.default_rng(seed)
    
    resampled_indices = rng.choice(n_rows, size=n_rows, replace=True)
    row_counts = np.bincount(resampled_indices, minlength=n_rows)
    n_boot = n * row_counts
    
    return wls(X, y, n_boot).flatten(), n_boot.sum()


def _bootstrap_iteration_cluster(args: Tuple) -> Tuple[np.ndarray, float]:
    """Single bootstrap iteration for cluster bootstrap (parallelizable)"""
    b, X, y, n, group_idx, n_unique_groups, seed = args
    rng = np.random.default_rng(seed)
    
    resampled_group_ids = rng.choice(n_unique_groups, size=n_unique_groups, replace=True)
    bootstrap_scale = np.bincount(resampled_group_ids, minlength=n_unique_groups)
    n_boot = n * bootstrap_scale[group_idx]
    
    return wls(X, y, n_boot).flatten(), n_boot.sum()


# ============================================================================
# Bootstrap executor helper
# ============================================================================

class BootstrapExecutor:
    """Encapsulates bootstrap execution logic (sequential or parallel)"""
    
    def __init__(self, n_bootstraps: int, n_jobs: int, rng: np.random.Generator):
        self.n_bootstraps = n_bootstraps
        self.n_jobs = n_jobs
        self.seeds = rng.integers(0, 2**31, size=n_bootstraps)
    
    def execute(self, iteration_func, base_args: Tuple, 
                args_builder=None) -> Tuple[np.ndarray, np.ndarray]:
        """Execute bootstrap iterations and return (coefficients, sizes)"""
        if args_builder is None:
            args_builder = lambda b, seed: base_args + (seed,)
        
        args_list = [(b,) + args_builder(b, self.seeds[b]) for b in range(self.n_bootstraps)]
        
        runner = self._run_sequential if self.n_jobs == 1 else self._run_parallel
        return runner(iteration_func, args_list)
    
    def _run_sequential(self, iteration_func, args_list) -> Tuple[np.ndarray, np.ndarray]:
        """Run bootstrap sequentially with progress bar"""
        print(f"Starting bootstrap with {self.n_bootstraps} iterations (sequential)")
        results = [iteration_func(args) for args in tqdm(args_list)]
        return self._parse_results(results)
    
    def _run_parallel(self, iteration_func, args_list) -> Tuple[np.ndarray, np.ndarray]:
        """Run bootstrap in parallel using joblib"""
        from joblib import Parallel, delayed
        
        print(f"Starting bootstrap with {self.n_bootstraps} iterations ({self.n_jobs} jobs)")
        results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(iteration_func)(args) for args in args_list
        )
        return self._parse_results(results)
    
    @staticmethod
    def _parse_results(results) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([r[0] for r in results]), np.array([r[1] for r in results])


# ============================================================================
# SQL Builder Mixin
# ============================================================================

class SQLBuilderMixin:
    """Mixin providing common SQL building utilities"""
    
    def _build_round_expr(self, col_expr: str, alias: str) -> Tuple[str, str]:
        """Build rounded expression for SELECT and GROUP BY
        
        Args:
            col_expr: The full SQL expression (e.g., "LN(ntl_harm + 0.01)")
            alias: The SQL-safe alias name to use (e.g., "log_ntl_harm_0_01")
        
        Returns:
            Tuple of (select_clause, group_by_clause)
            - select_clause includes alias (e.g., 'ROUND(expr, 5) AS alias')
            - group_by_clause references the alias directly (e.g., 'alias')
        """
        if self.round_strata is not None:
            rounded_expr = f"ROUND({col_expr}, {self.round_strata})"
            # In GROUP BY, reference the alias (not the full expression)
            return f"{rounded_expr} AS {alias}", alias
        # No rounding: use expression AS alias, and reference alias in GROUP BY
        return f"{col_expr} AS {alias}", alias

    def _build_agg_columns(self, vars: List, boolean_cols: set, unit_col: Optional[str],
                           include_sq: bool = True) -> List[str]:
        """Build aggregation SQL for outcome variables"""
        agg_parts = ["COUNT(*) as count"]
        for var in vars:
            expr = var.get_sql_expression(unit_col) if hasattr(var, 'get_sql_expression') else var
            var_name = var.name if hasattr(var, 'name') else var
            display_name = var.display_name if hasattr(var, 'display_name') else var_name
            
            if var_name in boolean_cols:
                expr = f"CAST({expr} AS SMALLINT)"
            
            # Use proper quoting for column aliases
            sum_alias = quote_identifier(f"sum_{display_name}")
            agg_parts.append(f"SUM({expr}) as {sum_alias}")
            if include_sq:
                sq_alias = quote_identifier(f"sum_{display_name}_sq")
                agg_parts.append(f"SUM(POW({expr}, 2)) as {sq_alias}")
        return agg_parts


# ============================================================================
# Shared base class for DuckRegression and DuckMundlak
# ============================================================================

class DuckLinearModel(DuckReg, SQLBuilderMixin):
    """Base class for linear models with shared estimation, bootstrap, and vcov logic"""
    
    _COMPRESSED_VIEW = "_compressed_view"
    
    def __init__(
        self,
        db_name: str,
        table_name: str,
        seed: int,
        formula=None,
        n_bootstraps: int = 100,
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
            variable_casts=None,
            fitter=fitter,
            **kwargs,
        )
        
        self.formula = formula
        self.n_jobs = n_jobs
        self.subset = subset
        self.strata_cols: List[str] = []
        self._boolean_cols: Optional[set] = None
        self._fitter_result: Optional[FitterResult] = None
        self._data_fetched: bool = False
        self.df_compressed: Optional[pd.DataFrame] = None
        self.agg_query: Optional[str] = None
        self.n_compressed_rows: Optional[int] = None
        self._results: Optional[RegressionResults] = None
        
        # Extract from formula - use raw names for internal logic
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
    # Abstract methods (must be implemented by subclasses)
    # -------------------------------------------------------------------------
    
    def _get_n_coefs(self) -> int:
        raise NotImplementedError
    
    def _get_cluster_data_for_bootstrap(self) -> Tuple[pd.DataFrame, Optional[str]]:
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Shared utility methods
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
        return self.cluster_col

    def _build_where_clause(self, user_subset: Optional[str] = None) -> str:
        return self.formula.get_where_clause_sql(user_subset)

    def _get_unit_col(self) -> Optional[str]:
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
            # Use sql_name for column lookup
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
        
        # Get both total observations and compressed row count
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
    # Estimation
    # -------------------------------------------------------------------------

    def estimate(self) -> np.ndarray:
        """Estimate coefficients using WLS (numpy or duckdb)"""
        estimator = self._estimate_duckdb if self.fitter == "duckdb" else self._estimate_numpy
        return estimator()
    
    def _estimate_numpy(self) -> np.ndarray:
        """Estimate using in-memory numpy WLS via NumpyFitter"""
        self._ensure_data_fetched()
        
        y, X, n = self.collect_data(data=self.df_compressed)
        cluster_ids = self._get_cluster_ids_from_df()
        
        numpy_fitter = NumpyFitter(alpha=1e-8, se_type="stata")
        self._fitter_result = numpy_fitter.fit(
            X=X, y=y, weights=n,
            coef_names=getattr(self, 'coef_names_', None),
            cluster_ids=cluster_ids,
            compute_vcov=False
        )
        
        self._update_coef_names()
        return self._fitter_result.coefficients
    
    def _estimate_duckdb(self) -> np.ndarray:
        """Estimate using DuckDB sufficient statistics (out-of-core)"""
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
        """Get the y column name for DuckDB fitter, using SQL-safe name"""
        # Use sql_name for the outcome
        outcome_sql_name = self.formula.get_outcome_sql_names()[0]
        col_name = f"sum_{outcome_sql_name}"
        # Quote if needed (though sql_name should already be safe)
        return quote_identifier(col_name) if needs_quoting(col_name) else col_name

    def _get_x_cols_for_duckdb(self) -> List[str]:
        """Get x column names for DuckDB fitter, using SQL-safe names"""
        simple_cov_sql_names = [var.sql_name for var in self.formula.covariates if not var.is_intercept()]
        avg_cols = [f"avg_{cov}_fe{i}" for i in range(len(self.fe_cols)) for cov in simple_cov_sql_names]
        # Use sql_names for covariates
        non_intercept_sql_names = [var.sql_name for var in self.formula.covariates if not var.is_intercept()]
        
        # Quote column names that need it
        all_cols = non_intercept_sql_names + avg_cols
        return [quote_identifier(col) if needs_quoting(col) else col for col in all_cols]
    
    def _get_non_intercept_covariate_names(self) -> List[str]:
        """Get simple covariate names excluding intercept"""
        return [var.name for var in self.formula.covariates if not var.is_intercept()]

    def _needs_intercept_for_duckdb(self) -> bool:
        """Determine if DuckDBFitter should add an intercept.
        
        Base implementation for demeaning approach:
        - Returns True only if there are no fixed effects
        - With FEs and demeaning, intercept is absorbed regardless of explicit request
        
        Subclasses (e.g., DuckMundlak) may override this.
        """
        # With demeaning, FEs absorb the intercept - ignore explicit intercept
        return not self.fe_cols

    def _get_cluster_ids_from_df(self) -> Optional[np.ndarray]:
        """Get cluster IDs from compressed dataframe"""
        cluster_col = self._get_cluster_col_for_vcov()
        if cluster_col and self.df_compressed is not None and cluster_col in self.df_compressed.columns:
            return self.df_compressed[cluster_col].values
        return None

    def _update_coef_names(self):
        """Update coefficient names from fitter result, handling multiple outcomes"""
        if self._fitter_result is not None and self._fitter_result.coef_names is not None:
            # Only update if we don't already have coef_names_ set by collect_data
            if not hasattr(self, 'coef_names_') or self.coef_names_ is None:
                self.coef_names_ = self._fitter_result.coef_names
        
        # Handle multiple outcomes by prefixing
        if len(self.outcome_vars) > 1 and hasattr(self, 'coef_names_') and self.coef_names_:
            self.coef_names_ = [f"{name}:{outcome}" 
                               for outcome in self.outcome_vars 
                               for name in self.coef_names_]

    # -------------------------------------------------------------------------
    # Variance-covariance estimation
    # -------------------------------------------------------------------------

    def fit_vcov(self):
        """Compute variance-covariance matrix (cluster-robust or HC1)"""
        if self.fitter == "duckdb" and self._fitter_result is not None and self._fitter_result.vcov is not None:
            self.vcov = self._fitter_result.vcov
            self.se = self._fitter_result.se_type
            return
        
        self._ensure_data_fetched()
        
        y, X, n = self.collect_data(data=self.df_compressed)
        cluster_ids = self._get_cluster_ids_from_df()
        
        numpy_fitter = NumpyFitter(alpha=1e-8, se_type="stata")
        result = numpy_fitter.fit(
            X=X, y=y, weights=n,
            coef_names=self.coef_names_,
            cluster_ids=cluster_ids,
            compute_vcov=True
        )
        
        self.vcov = result.vcov
        self.se = result.se_type
        self.n_bootstraps = 0
        self._results = None  # Reset cached results

    # -------------------------------------------------------------------------
    # Bootstrap methods
    # -------------------------------------------------------------------------

    def bootstrap(self) -> np.ndarray:
        """Run bootstrap to estimate variance-covariance matrix"""
        self._ensure_data_fetched()
        self.se = "bootstrap"
        executor = BootstrapExecutor(self.n_bootstraps, self.n_jobs, self.rng)
        
        runner = self._run_cluster_bootstrap if self.cluster_col else self._run_iid_bootstrap
        boot_coefs, boot_sizes = runner(executor)
        
        vcov = np.cov(boot_coefs.T, aweights=boot_sizes)
        self._results = None  # Reset cached results
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

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Provide comprehensive results summary"""
        result = {
            "coefficients": self.results.to_dict() if self.results else None,
            "n_obs": getattr(self, 'n_obs', None),
            "n_compressed": self.n_compressed_rows,
            "estimator_type": self.__class__.__name__,
            "outcome_vars": self.outcome_vars,
            "covariates": self.covariates,
            "fe_cols": self.fe_cols,
            "cluster_col": self.cluster_col,
        }
        return result
    
    def summary_df(self) -> pd.DataFrame:
        """Get results as a DataFrame"""
        if self.results is None:
            return pd.DataFrame()
        return self.results.to_dataframe()
