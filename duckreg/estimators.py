import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import sys
from typing import Tuple, Optional, List, Dict, Any

from .demean import demean, _convert_to_int
from .duckreg import DuckReg, wls

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG)


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
        
        if self.n_jobs == 1:
            return self._run_sequential(iteration_func, args_list)
        return self._run_parallel(iteration_func, args_list)
    
    def _run_sequential(self, iteration_func, args_list) -> Tuple[np.ndarray, np.ndarray]:
        """Run bootstrap sequentially with progress bar"""
        print(f"Starting bootstrap with {self.n_bootstraps} iterations (sequential)")
        results = [iteration_func(args) for args in tqdm(args_list)]
        return np.array([r[0] for r in results]), np.array([r[1] for r in results])
    
    def _run_parallel(self, iteration_func, args_list) -> Tuple[np.ndarray, np.ndarray]:
        """Run bootstrap in parallel using joblib"""
        from joblib import Parallel, delayed
        
        print(f"Starting bootstrap with {self.n_bootstraps} iterations ({self.n_jobs} jobs)")
        results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(iteration_func)(args) for args in args_list
        )
        return np.array([r[0] for r in results]), np.array([r[1] for r in results])


# ============================================================================
# Shared base class for DuckRegression and DuckMundlak
# ============================================================================

class DuckLinearModel(DuckReg):
    """Base class for linear models with shared estimation, bootstrap, and vcov logic"""
    
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
            **kwargs,
        )
        
        self.formula = formula
        self.n_jobs = n_jobs
        self.subset = subset
        self.strata_cols: List[str] = []
        self._boolean_cols: Optional[set] = None
        
        # Extract from formula
        self.outcome_vars = formula.get_outcome_names()
        self.covariates = formula.get_covariate_names()
        self.fe_cols = formula.get_fe_names()
        self.cluster_col = formula.cluster.name if formula.cluster else None
        
        if not self.outcome_vars:
            raise ValueError("No outcome variables provided")
        
        logger.debug(f"{self.__class__.__name__}: outcomes={self.outcome_vars}, "
                    f"covariates={self.covariates}, fe={self.fe_cols}, cluster={self.cluster_col}")

    # -------------------------------------------------------------------------
    # Abstract methods (must be implemented by subclasses)
    # -------------------------------------------------------------------------
    
    def _get_n_coefs(self) -> int:
        """Get number of coefficients based on model structure"""
        raise NotImplementedError
    
    def _get_cluster_data_for_bootstrap(self) -> Tuple[pd.DataFrame, Optional[str]]:
        """Get cluster-level data for bootstrap. Returns (df, cluster_col_name)"""
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
        """Get the cluster column name for analytical vcov"""
        return self.cluster_col

    def _build_where_clause(self, user_subset: Optional[str] = None) -> str:
        """Build WHERE clause using formula's method"""
        return self.formula.get_where_clause_sql(user_subset)

    def _get_unit_col(self) -> Optional[str]:
        """Get the first FE column as unit column for SQL expressions"""
        return self.fe_cols[0] if self.fe_cols else None

    # -------------------------------------------------------------------------
    # Estimation
    # -------------------------------------------------------------------------

    def estimate(self) -> np.ndarray:
        """Estimate coefficients using WLS"""
        y, X, n = self.collect_data(data=self.df_compressed)
        betahat = wls(X, y, n)
        
        # Expand coefficient names for multiple outcomes
        if len(self.outcome_vars) > 1:
            self.coef_names_ = [f"{name}:{outcome}" 
                               for outcome in self.outcome_vars 
                               for name in self.coef_names_]
        
        return betahat.flatten()

    # -------------------------------------------------------------------------
    # Variance-covariance estimation
    # -------------------------------------------------------------------------

    def fit_vcov(self):
        """Compute variance-covariance matrix (cluster-robust or HC1)"""
        y, X, n = self.collect_data(data=self.df_compressed)
        betahat = wls(X, y, n).flatten()
        self.n_bootstraps = 0
        
        # Compute bread: (X'WX)^{-1}
        w = n.reshape(-1, 1)
        bread = np.linalg.inv(X.T @ (X * w))
        
        cluster_col = self._get_cluster_col_for_vcov()
        
        if cluster_col and cluster_col in self.df_compressed.columns:
            self._compute_cluster_vcov(X, n, betahat, bread, cluster_col)
        else:
            self._compute_hc1_vcov(X, n, betahat, bread)

    def _compute_cluster_vcov(self, X: np.ndarray, n: np.ndarray, 
                              betahat: np.ndarray, bread: np.ndarray, 
                              cluster_col: str):
        """Compute cluster-robust standard errors (CR0)"""
        self.se = "cluster"
        
        yhat = (X @ betahat).reshape(-1, 1)
        yprime = self.df_compressed[f"sum_{self.outcome_vars[0]}"].values.reshape(-1, 1)
        residuals = yprime - n.reshape(-1, 1) * yhat
        
        clusters = self.df_compressed[cluster_col].values
        unique_clusters = np.unique(clusters)
        
        # Compute meat matrix via cluster scores
        meat = np.zeros((X.shape[1], X.shape[1]))
        for cluster in unique_clusters:
            mask = clusters == cluster
            score_g = (X[mask] * residuals[mask]).sum(axis=0, keepdims=True).T
            meat += score_g @ score_g.T
        
        # Small sample adjustment
        N, K, G = n.sum(), X.shape[1], len(unique_clusters)
        adjustment = (G / (G - 1)) * ((N - 1) / (N - K))
        self.vcov = adjustment * bread @ meat @ bread

    def _compute_hc1_vcov(self, X: np.ndarray, n: np.ndarray, 
                          betahat: np.ndarray, bread: np.ndarray):
        """Compute heteroskedasticity-robust standard errors (HC1)"""
        self.se = "hc1"
        
        yhat = (X @ betahat).reshape(-1, 1)
        sum_sq_col = f"sum_{self.outcome_vars[0]}_sq"
        
        if sum_sq_col in self.df_compressed.columns:
            yprime = self.df_compressed[f"sum_{self.outcome_vars[0]}"].values.reshape(-1, 1)
            yprimeprime = self.df_compressed[sum_sq_col].values.reshape(-1, 1)
            w_rss = yprimeprime - 2 * yhat * yprime + n.reshape(-1, 1) * (yhat**2)
        else:
            logger.warning("sum_sq column not available, using simple variance estimate")
            y_mean = self.df_compressed[f"mean_{self.outcome_vars[0]}"].values.reshape(-1, 1)
            w_rss = ((y_mean - yhat) ** 2) * n.reshape(-1, 1)
        
        meat = X.T @ (X * w_rss)
        n_nk = n.sum() / (n.sum() - X.shape[1])
        self.vcov = n_nk * (bread @ meat @ bread)

    # -------------------------------------------------------------------------
    # Bootstrap methods
    # -------------------------------------------------------------------------

    def bootstrap(self) -> np.ndarray:
        """Run bootstrap to estimate variance-covariance matrix"""
        self.se = "bootstrap"
        executor = BootstrapExecutor(self.n_bootstraps, self.n_jobs, self.rng)
        
        if self.cluster_col:
            boot_coefs, boot_sizes = self._run_cluster_bootstrap(executor)
        else:
            boot_coefs, boot_sizes = self._run_iid_bootstrap(executor)
        
        vcov = np.cov(boot_coefs.T, aweights=boot_sizes)
        return np.expand_dims(vcov, axis=0) if vcov.ndim == 0 else vcov

    def _run_iid_bootstrap(self, executor: BootstrapExecutor) -> Tuple[np.ndarray, np.ndarray]:
        """IID bootstrap - resample compressed dataframe"""
        y, X, n = self.collect_data(data=self.df_compressed)
        n_rows = len(self.df_compressed)
        
        return executor.execute(
            _bootstrap_iteration_iid,
            (X, y, n, n_rows),
            args_builder=lambda b, seed: (X, y, n, n_rows, seed)
        )

    def _run_cluster_bootstrap(self, executor: BootstrapExecutor) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster bootstrap"""
        df_clusters, cluster_col_name = self._get_cluster_data_for_bootstrap()
        df_clusters = df_clusters.dropna(subset=[cluster_col_name])
        
        unique_groups = df_clusters[cluster_col_name].unique()
        group_to_idx = {x: i for i, x in enumerate(unique_groups)}
        group_idx = df_clusters[cluster_col_name].map(group_to_idx).to_numpy(dtype=int)
        
        y, X, n = self.collect_data(data=df_clusters)
        n_unique_groups = len(unique_groups)
        
        return executor.execute(
            _bootstrap_iteration_cluster,
            (X, y, n, group_idx, n_unique_groups),
            args_builder=lambda b, seed: (X, y, n, group_idx, n_unique_groups, seed)
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Provide comprehensive results summary"""
        result = {
            "point_estimate": self.point_estimate,
            "coef_names": getattr(self, 'coef_names_', None),
            "n_obs": getattr(self, 'n_obs', None),
            "n_obs_compressed": len(self.df_compressed) if hasattr(self, 'df_compressed') else None,
            "estimator_type": self.__class__.__name__,
            "fe_method": "demean" if isinstance(self, DuckRegression) else "mundlak" if self.fe_cols else None,
            "outcome_vars": self.outcome_vars,
            "covariates": self.covariates,
            "fe_cols": self.fe_cols,
            "cluster_col": self.cluster_col
        }
        
        if hasattr(self, 'vcov'):
            result.update({
                "standard_error": np.sqrt(np.diag(self.vcov)),
                "vcov": self.vcov,
                "se_type": getattr(self, "se", "unknown")
            })
        
        return result


# ============================================================================
# DuckRegression (demeaning approach)
# ============================================================================

class DuckRegression(DuckLinearModel):
    """OLS with fixed effects via demeaning"""
    
    def __init__(self, rowid_col: str = "rowid", fitter: str = "numpy", **kwargs):
        super().__init__(**kwargs)
        self.rowid_col = rowid_col
        self.strata_cols = self.covariates + self.fe_cols

    def _get_n_coefs(self) -> int:
        n_covs = len(self.covariates) if self.fe_cols else len(self.covariates) + 1
        return n_covs * len(self.outcome_vars)

    def _get_cluster_data_for_bootstrap(self) -> Tuple[pd.DataFrame, Optional[str]]:
        return self.df_compressed, self.cluster_col

    def prepare_data(self):
        pass  # No preparation needed

    def compress_data(self):
        """Compress data by grouping on strata columns"""
        boolean_cols = self._get_boolean_columns()
        unit_col = self._get_unit_col()
        
        # Build SELECT and GROUP BY
        select_parts, group_by_parts = self._build_strata_select_sql(boolean_cols, unit_col)
        
        # Add cluster if present
        if self.cluster_col:
            cluster_expr = f"CAST({self.cluster_col} AS SMALLINT)" if self.cluster_col in boolean_cols else self.cluster_col
            select_parts.append(f"{cluster_expr} AS {self.cluster_col}")
            group_by_parts.append(cluster_expr)
        
        # Build aggregations
        agg_parts = self._build_outcome_agg_sql(boolean_cols, unit_col)
        
        self.agg_query = f"""
        SELECT {', '.join(select_parts)}, {', '.join(agg_parts)}
        FROM {self.table_name}
        {self._build_where_clause(self.subset)}
        GROUP BY {', '.join(group_by_parts)}
        """
        
        self.df_compressed = self.conn.execute(self.agg_query).fetchdf().dropna()
        
        # Set column names
        expected_cols = self.strata_cols.copy()
        if self.cluster_col:
            expected_cols.append(self.cluster_col)
        expected_cols.extend(["count"] + [f"sum_{v}" for v in self.outcome_vars] + 
                           [f"sum_{v}_sq" for v in self.outcome_vars])
        self.df_compressed.columns = expected_cols
        
        self.n_obs = int(self.df_compressed['count'].sum())
        
        # Compute means
        for var in self.outcome_vars:
            self.df_compressed[f"mean_{var}"] = self.df_compressed[f"sum_{var}"] / self.df_compressed["count"]

    def _build_strata_select_sql(self, boolean_cols: set, unit_col: Optional[str]) -> Tuple[List[str], List[str]]:
        """Build SELECT and GROUP BY parts for strata columns"""
        select_parts, group_by_parts = [], []
        
        for col in self.strata_cols:
            col_expr = self.formula.get_covariate_expression(col, unit_col, 'year', boolean_cols)
            if col_expr == col:
                col_expr = self.formula.get_fe_expression(col, boolean_cols)
            
            if self.round_strata is not None:
                select_parts.append(f"ROUND({col_expr}, {self.round_strata}) AS {col}")
                group_by_parts.append(f"ROUND({col_expr}, {self.round_strata})")
            else:
                select_parts.append(f"{col_expr} AS {col}")
                group_by_parts.append(col_expr)
        
        return select_parts, group_by_parts

    def _build_outcome_agg_sql(self, boolean_cols: set, unit_col: Optional[str]) -> List[str]:
        """Build aggregation SQL for outcomes"""
        agg_parts = ["COUNT(*) as count"]
        for var in self.formula.outcomes:
            expr = var.get_sql_expression(unit_col)
            if var.name in boolean_cols:
                expr = f"CAST({expr} AS SMALLINT)"
            agg_parts.append(f"SUM({expr}) as sum_{var.name}")
            agg_parts.append(f"SUM(POW({expr}, 2)) as sum_{var.name}_sq")
        return agg_parts

    def collect_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect and demean data"""
        y = data.filter(regex=f"mean_({'|'.join(self.outcome_vars)})", axis=1).values
        X = data[self.covariates].values
        n = data["count"].values

        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X

        if self.fe_cols:
            fe = _convert_to_int(data[self.fe_cols])
            fe = fe.reshape(-1, 1) if fe.ndim == 1 else fe
            y, _ = demean(x=y, flist=fe, weights=n)
            X, _ = demean(x=X, flist=fe, weights=n)
            self.coef_names_ = self.covariates.copy()
        else:
            X = np.c_[np.ones(X.shape[0]), X]
            self.coef_names_ = ['Intercept'] + self.covariates.copy()

        return y, X, n


# ============================================================================
# DuckMundlak (Mundlak device approach)
# ============================================================================

class DuckMundlak(DuckLinearModel):
    """OLS with fixed effects via Mundlak device"""
    
    _CLUSTER_ALIAS = "__cluster__"
    
    def _get_n_coefs(self) -> int:
        simple_covs = len(self.formula.get_simple_covariate_names())
        rhs_count = 1 + len(self.covariates) + len(self.fe_cols) * simple_covs
        return rhs_count * len(self.outcome_vars)

    def _get_cluster_data_for_bootstrap(self) -> Tuple[pd.DataFrame, str]:
        return self.df_compressed, self._CLUSTER_ALIAS

    def _get_cluster_col_for_vcov(self) -> str:
        return self._CLUSTER_ALIAS

    def prepare_data(self):
        """Create design matrix with Mundlak averages"""
        boolean_cols = self._get_boolean_columns()
        unit_col = self._get_unit_col()
        
        # Build SELECT parts using formula's SQL generators
        select_parts = [
            self.formula.get_fe_select_sql(boolean_cols),
            self.formula.get_outcomes_select_sql(unit_col, 'year', boolean_cols),
            self.formula.get_covariates_select_sql(unit_col, 'year', boolean_cols, include_interactions=True),
        ]
        
        cluster_sql = self.formula.get_cluster_select_sql(boolean_cols, self._CLUSTER_ALIAS, unit_col)
        if cluster_sql:
            select_parts.append(cluster_sql)
        
        # Create design matrix
        self.conn.execute(f"""
        CREATE OR REPLACE TABLE design_matrix AS
        SELECT {', '.join(p for p in select_parts if p)}
        FROM {self.table_name}
        {self._build_where_clause(self.subset)}
        """)
        
        # Add FE averages for Mundlak device
        self._add_fe_averages()

    def _add_fe_averages(self):
        """Add FE-level averages for Mundlak device"""
        simple_cov_names = self.formula.get_simple_covariate_names()
        
        for i, fe_col in enumerate(self.fe_cols):
            avg_cols = ", ".join([f"AVG({cov}) AS avg_{cov}_fe{i}" for cov in simple_cov_names])
            avg_col_list = ", ".join([f"fe{i}.avg_{cov}_fe{i}" for cov in simple_cov_names])
            
            self.conn.execute(f"""
            CREATE OR REPLACE TABLE design_matrix AS
            SELECT dm.*, {avg_col_list}
            FROM design_matrix dm
            JOIN (SELECT {fe_col}, {avg_cols} FROM design_matrix GROUP BY {fe_col}) fe{i} 
            ON dm.{fe_col} = fe{i}.{fe_col}
            """)

    def compress_data(self):
        """Compress design matrix"""
        simple_cov_names = self.formula.get_simple_covariate_names()
        cov_cols = list(self.covariates)
        avg_cols = [f"avg_{cov}_fe{i}" for i in range(len(self.fe_cols)) for cov in simple_cov_names]
        
        strata_cols_to_round = cov_cols + avg_cols
        self.strata_cols = strata_cols_to_round + [self._CLUSTER_ALIAS]
        
        # Build SELECT and GROUP BY with optional rounding
        if self.round_strata is not None:
            select_clause = ", ".join([f"ROUND({col}, {self.round_strata}) AS {col}" for col in strata_cols_to_round])
            group_by_clause = ", ".join([f"ROUND({col}, {self.round_strata})" for col in strata_cols_to_round])
        else:
            select_clause = ", ".join(strata_cols_to_round)
            group_by_clause = ", ".join(strata_cols_to_round)
        
        select_clause += f", {self._CLUSTER_ALIAS}"
        group_by_clause += f", {self._CLUSTER_ALIAS}"
        
        outcome_aggs = ", ".join([f"SUM({var}) as sum_{var}" for var in self.outcome_vars])
        
        self.agg_query = f"""
        SELECT {select_clause}, COUNT(*) as count, {outcome_aggs}
        FROM design_matrix
        GROUP BY {group_by_clause}
        """
        
        self.df_compressed = self.conn.execute(self.agg_query).fetchdf()
        self.n_obs = int(self.df_compressed['count'].sum())
        
        # Compute means
        for var in self.outcome_vars:
            self.df_compressed[f"mean_{var}"] = self.df_compressed[f"sum_{var}"] / self.df_compressed["count"]

    def collect_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect data with Mundlak averages as additional regressors"""
        simple_cov_names = self.formula.get_simple_covariate_names()
        avg_cols = [f"avg_{cov}_fe{i}" for i in range(len(self.fe_cols)) for cov in simple_cov_names]
        
        self.rhs = list(self.covariates) + avg_cols
        
        X = np.c_[np.ones(len(data)), data[self.rhs].values]
        self.coef_names_ = ['Intercept'] + self.rhs.copy()
        
        y = data[[f"mean_{var}" for var in self.outcome_vars]].values
        n = data["count"].values

        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X

        return y, X, n


# ============================================================================
# Legacy estimators (kept for backward compatibility)
# ============================================================================

class DuckMundlakEventStudy(DuckReg):
    """Event study estimator using Mundlak device"""
    def __init__(
        self,
        db_name: str,
        table_name: str,
        outcome_var: str,
        treatment_col: str,
        unit_col: str,
        time_col: str,
        cluster_col: str,
        pre_treat_interactions: bool = True,
        n_bootstraps: int = 100,
        duckdb_kwargs: dict = None,
        variable_casts: dict = None,
        **kwargs,
    ):
        super().__init__(
            db_name=db_name,
            table_name=table_name,
            n_bootstraps=n_bootstraps,
            duckdb_kwargs=duckdb_kwargs,
            variable_casts=variable_casts,
            **kwargs,
        )
        self.table_name = table_name
        self.outcome_var = outcome_var
        self.treatment_col = treatment_col
        self.unit_col = unit_col
        self.time_col = time_col
        self.num_periods = None
        self.cohorts = None
        self.time_dummies = None
        self.post_treatment_dummies = None
        self.transformed_query = None
        self.compression_query = None
        self.cluster_col = cluster_col
        self.pre_treat_interactions = pre_treat_interactions

    def prepare_data(self):
        # Create cohort data using CTE instead of temp table
        self.cohort_cte = f"""
        WITH cohort_data AS (
            SELECT *,
                   CASE WHEN cohort_min = 2147483647 THEN NULL ELSE cohort_min END as cohort,
                   CASE WHEN cohort_min IS NOT NULL AND cohort_min != 2147483647 THEN 1 ELSE 0 END as ever_treated
            FROM (
                SELECT *,
                       (SELECT MIN({self._cast_col(self.time_col)})
                        FROM {self.table_name} AS p2
                        WHERE p2.{self.unit_col} = p1.{self.unit_col} AND p2.{self.treatment_col} = 1
                       ) as cohort_min
                FROM {self.table_name} p1
            )
        )
        """
        #  retrieve_num_periods_and_cohorts using CTE instead of temp table
        self.num_periods = self.conn.execute(
            f"{self.cohort_cte} SELECT MAX({self.time_col}) FROM cohort_data"
        ).fetchone()[0]
        cohorts = self.conn.execute(
            f"{self.cohort_cte} SELECT DISTINCT cohort FROM cohort_data WHERE cohort IS NOT NULL"
        ).fetchall()
        self.cohorts = [row[0] for row in cohorts]
        # generate_time_dummies
        self.time_dummies = ",\n".join(
            [
                f"CASE WHEN {self.time_col} = {i} THEN 1 ELSE 0 END AS time_{i}"
                for i in range(self.num_periods + 1)
            ]
        )
        # generate cohort dummies
        cohort_intercepts = []
        for cohort in self.cohorts:
            cohort_intercepts.append(
                f"CASE WHEN cohort = {cohort} THEN 1 ELSE 0 END AS cohort_{cohort}"
            )
        self.cohort_intercepts = ",\n".join(cohort_intercepts)

        # generate_treatment_dummies
        treatment_dummies = []
        for cohort in self.cohorts:
            for i in range(self.num_periods + 1):
                treatment_dummies.append(
                    f"""CASE WHEN cohort = {cohort} AND
                        {self.time_col} = {i}
                        {f"AND {self.treatment_col} == 1" if not self.pre_treat_interactions else ""}
                        THEN 1 ELSE 0 END AS treatment_time_{cohort}_{i}"""
                )
        self.treatment_dummies = ",\n".join(treatment_dummies)

        #  create_transformed_query using CTE instead of temp table
        self.design_matrix_cte = f"""
        {self.cohort_cte},
        transformed_panel_data AS (
            SELECT
                p.{self.unit_col},
                p.{self.time_col},
                p.{self.treatment_col},
                p.{self.outcome_var},
                -- Intercept (constant term)
                1 AS intercept,
                -- cohort intercepts
                {self.cohort_intercepts},
                -- Time dummies for each period
                {self.time_dummies},
                -- Treated group interacted with treatment time dummies
                {self.treatment_dummies}
            FROM cohort_data p
        )
        """

    def compress_data(self):
        # Pre-compute RHS columns to avoid repeated string operations
        cohort_cols = [f"cohort_{cohort}" for cohort in self.cohorts]
        time_cols = [f"time_{i}" for i in range(self.num_periods + 1)]
        treatment_cols = [f"treatment_time_{cohort}_{i}" for cohort in self.cohorts for i in range(self.num_periods + 1)]
        
        rhs_cols = ["intercept"] + cohort_cols + time_cols + treatment_cols
        rhs_clause = ", ".join(rhs_cols)
        
        # Use single query with CTE instead of temp table
        self.compression_query = f"""
        {self.design_matrix_cte}
        SELECT
            {rhs_clause},
            COUNT(*) AS count,
            SUM({self.outcome_var}) AS sum_{self.outcome_var}
        FROM transformed_panel_data
        GROUP BY {rhs_clause}
        """
        
        self.df_compressed = self.conn.execute(self.compression_query).fetchdf()
        self.df_compressed[f"mean_{self.outcome_var}"] = (
            self.df_compressed[f"sum_{self.outcome_var}"] / self.df_compressed["count"]
        )
        
        # Store for later use
        self.rhs_cols = rhs_cols

    def collect_data(self, data):
        self._rhs_list = self.rhs_cols
        X = data[self._rhs_list].values
        y = data[f"mean_{self.outcome_var}"].values
        n = data["count"].values

        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        return y, X, n

    def estimate(self):
        y, X, n = self.collect_data(data=self.df_compressed)
        coef = wls(X, y, n)
        res = pd.DataFrame(
            {
                "est": coef.squeeze(),
            },
            index=self._rhs_list,
        )
        cohort_names = [x.split("_")[1] for x in self._rhs_list if "cohort_" in x]
        event_study_coefs = {}
        for c in cohort_names:
            offset = res.filter(regex=f"^cohort_{c}", axis=0).values
            event_study_coefs[c] = (
                res.filter(regex=f"treatment_time_{c}_", axis=0) + offset
            )

        return event_study_coefs

    def bootstrap(self):
        # list all clusters
        total_clusters = self.conn.execute(
            f"SELECT COUNT(DISTINCT {self.cluster_col}) FROM transformed_panel_data"
        ).fetchone()[0]
        boot_coefs = {str(cohort): [] for cohort in self.cohorts}
        # bootstrap loop
        for _ in tqdm(range(self.n_bootstraps)):
            resampled_clusters = (
                self.conn.execute(
                    f"SELECT UNNEST(ARRAY(SELECT {self.cluster_col} FROM transformed_panel_data ORDER BY RANDOM() LIMIT {total_clusters}))"
                )
                .fetchdf()
                .values.flatten()
                .tolist()
            )

            self.conn.execute(
                f"""
                CREATE TEMP TABLE resampled_transformed_panel_data AS
                SELECT * FROM transformed_panel_data
                WHERE {self.cluster_col} IN ({", ".join(map(str, resampled_clusters))})
            """
            )

            self.conn.execute(
                f"""
                CREATE TEMP TABLE resampled_compressed_panel_data AS
                SELECT
                    {self.rhs.replace(";", "")},
                    COUNT(*) AS count,
                    SUM({self.outcome_var}) AS sum_{self.outcome_var}
                FROM
                    resampled_transformed_panel_data
                GROUP BY
                    {self.rhs.replace(";", "")}
            """
            )

            df_boot = self.conn.execute(
                "SELECT * FROM resampled_compressed_panel_data"
            ).fetchdf()
            df_boot[f"mean_{self.outcome_var}"] = (
                df_boot[f"sum_{self.outcome_var}"] / df_boot["count"]
            )

            y, X, n = self.collect_data(data=df_boot)
            res = pd.DataFrame(
                {
                    "est": wls(X, y, n).squeeze(),
                },
                index=self._rhs_list,
            )
            cohort_names = [x.split("_")[1] for x in self._rhs_list if "cohort_" in x]
            for c in cohort_names:
                offset = res.filter(regex=f"^cohort_{c}", axis=0).values
                event_study_coefs = (
                    res.filter(regex=f"treatment_time_{c}_", axis=0) + offset
                )
                boot_coefs[c].append(event_study_coefs.values.flatten())

            self.conn.execute("DROP TABLE resampled_transformed_panel_data")
            self.conn.execute("DROP TABLE resampled_compressed_panel_data")
        # Calculate the covariance matrix for each cohort
        bootstrap_cov_matrix = {
            cohort: np.cov(np.array(coefs).T) for cohort, coefs in boot_coefs.items()
        }
        return bootstrap_cov_matrix

    def summary(self) -> dict:
        """Summary of event study regression (overrides the parent class method)

        Returns:
            dict of event study coefficients and their standard errors
        """
        if self.n_bootstraps > 0:
            summary_tables = {}
            for c in self.point_estimate.keys():
                point_estimate = self.point_estimate[c]
                se = np.sqrt(np.diag(self.vcov[c]))
                summary_tables[c] = pd.DataFrame(
                    np.c_[point_estimate, se],
                    columns=["point_estimate", "se"],
                    index=point_estimate.index,
                )
            return summary_tables
        return {"point_estimate": self.point_estimate}


################################################################################
class DuckDoubleDemeaning(DuckReg):
    def __init__(
        self,
        db_name: str,
        table_name: str,
        outcome_var: str,
        treatment_var: str,
        fe_cols: list,
        seed: int,
        n_bootstraps: int = 100,
        cluster_col: str = None,
        duckdb_kwargs: dict = None,
        variable_casts: dict = None,
        **kwargs,
    ):
        super().__init__(
            db_name=db_name,
            table_name=table_name,
            seed=seed,
            n_bootstraps=n_bootstraps,
            duckdb_kwargs=duckdb_kwargs,
            variable_casts=variable_casts,
            **kwargs,
        )
        self.outcome_var = outcome_var
        self.treatment_var = treatment_var
        self.fe_cols = fe_cols
        self.cluster_col = cluster_col

    def prepare_data(self):
        self.conn.execute(f"""
        CREATE TEMP TABLE overall_mean AS
        SELECT AVG({self._cast_col(self.treatment_var)}) AS mean_{self.treatment_var}
        FROM {self.table_name}
        """)

        for i, fe_col in enumerate(self.fe_cols):
            self.conn.execute(f"""
            CREATE TEMP TABLE fe_{i}_means AS
            SELECT {self._cast_col(fe_col)} AS {fe_col}, AVG({self._cast_col(self.treatment_var)}) AS mean_{self.treatment_var}_fe{i}
            FROM {self.table_name}
            GROUP BY {self._cast_col(fe_col)}
            """)

        join_clauses = [f"JOIN fe_{i}_means fe{i} ON t.{fe_col} = fe{i}.{fe_col}" for i, fe_col in enumerate(self.fe_cols)]
        demean_terms = [f"fe{i}.mean_{self.treatment_var}_fe{i}" for i in range(len(self.fe_cols))]
        demean_formula = f"t.{self._cast_col(self.treatment_var)} - {' - '.join(demean_terms)} + {len(self.fe_cols)-1} * om.mean_{self.treatment_var}"
        
        self.conn.execute(f"""
        CREATE TEMP TABLE multi_demeaned AS
        SELECT {", ".join([f"t.{fe_col}" for fe_col in self.fe_cols])}, t.{self.outcome_var}, {demean_formula} AS ddot_{self.treatment_var}
        FROM {self.table_name} t {" ".join(join_clauses)} CROSS JOIN overall_mean om
        """)

    def compress_data(self):
        self.df_compressed = self.conn.execute(f"""
        SELECT ddot_{self.treatment_var}, COUNT(*) as count, SUM({self.outcome_var}) as sum_{self.outcome_var}
        FROM multi_demeaned GROUP BY ddot_{self.treatment_var}
        """).fetchdf()
        
        self.n_obs = int(self.df_compressed['count'].sum())
        self.df_compressed[f"mean_{self.outcome_var}"] = self.df_compressed[f"sum_{self.outcome_var}"] / self.df_compressed["count"]

    def collect_data(self, data: pd.DataFrame):
        X = np.c_[np.ones(len(data)), data[f"ddot_{self.treatment_var}"].values]
        y = data[f"mean_{self.outcome_var}"].values.reshape(-1, 1)
        n = data["count"].values
        self.coef_names_ = ['Intercept', f'ddot_{self.treatment_var}']
        return y, X, n

    def estimate(self):
        y, X, n = self.collect_data(data=self.df_compressed)
        return wls(X, y, n)

    def bootstrap(self):
        boot_coefs = np.zeros((self.n_bootstraps, 2))

        if self.cluster_col is None:
            total_samples = self.conn.execute(f"SELECT COUNT(DISTINCT {self.fe_cols[0]}) FROM {self.table_name}").fetchone()[0]
            self.bootstrap_query = f"""
            SELECT ddot_{self.treatment_var}, COUNT(*) as count, SUM({self.outcome_var}) as sum_{self.outcome_var}
            FROM multi_demeaned WHERE {self.fe_cols[0]} IN (SELECT unnest((?))) GROUP BY ddot_{self.treatment_var}
            """
        else:
            total_samples = self.conn.execute(f"SELECT COUNT(DISTINCT {self.cluster_col}) FROM {self.table_name}").fetchone()[0]
            self.bootstrap_query = f"""
            SELECT ddot_{self.treatment_var}, COUNT(*) as count, SUM({self.outcome_var}) AS sum_{self.outcome_var}
            FROM multi_demeaned WHERE {self.cluster_col} IN (SELECT unnest((?))) GROUP BY ddot_{self.treatment_var}
            """

        for b in tqdm(range(self.n_bootstraps)):
            resampled_samples = self.rng.choice(total_samples, size=total_samples, replace=True)
            df_boot = self.conn.execute(self.bootstrap_query, [resampled_samples.tolist()]).fetchdf()
            df_boot[f"mean_{self.outcome_var}"] = df_boot[f"sum_{self.outcome_var}"] / df_boot["count"]
            y, X, n = self.collect_data(data=df_boot)
            boot_coefs[b, :] = wls(X, y, n).flatten()

        return np.cov(boot_coefs.T)

    def summary(self) -> dict:
        """Summary of double-demeaning regression"""
        result = {
            "point_estimate": self.point_estimate,
            "coef_names": getattr(self, 'coef_names_', None),
            "n_obs": getattr(self, 'n_obs', None),
            "n_obs_compressed": len(self.df_compressed) if hasattr(self, 'df_compressed') else None,
            "estimator_type": "DuckDoubleDemeaning",
            "fe_method": "double_demean",
            "outcome_var": self.outcome_var,
            "treatment_var": self.treatment_var,
            "fe_cols": self.fe_cols,
            "cluster_col": self.cluster_col
        }
        
        if self.n_bootstraps > 0:
            result.update({
                "standard_error": np.sqrt(np.diag(self.vcov)),
                "vcov": self.vcov,
                "se_type": "bootstrap",
            })
        
        return result