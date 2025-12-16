"""
Mixins providing reusable functionality for estimators.

This module follows Interface Segregation Principle - each mixin provides
a focused set of related methods that can be composed as needed.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional, Set
from tqdm import tqdm

from ..fitters import wls

logger = logging.getLogger(__name__)


# ============================================================================
# Bootstrap Helpers (module-level for parallel execution)
# ============================================================================

def _bootstrap_iteration_iid(args: Tuple) -> Tuple[np.ndarray, float]:
    """Single bootstrap iteration for IID bootstrap"""
    b, X, y, n, n_rows, seed = args
    rng = np.random.default_rng(seed)
    
    resampled_indices = rng.choice(n_rows, size=n_rows, replace=True)
    row_counts = np.bincount(resampled_indices, minlength=n_rows)
    n_boot = n * row_counts
    
    return wls(X, y, n_boot).flatten(), n_boot.sum()


def _bootstrap_iteration_cluster(args: Tuple) -> Tuple[np.ndarray, float]:
    """Single bootstrap iteration for cluster bootstrap"""
    b, X, y, n, group_idx, n_unique_groups, seed = args
    rng = np.random.default_rng(seed)
    
    resampled_group_ids = rng.choice(n_unique_groups, size=n_unique_groups, replace=True)
    bootstrap_scale = np.bincount(resampled_group_ids, minlength=n_unique_groups)
    n_boot = n * bootstrap_scale[group_idx]
    
    return wls(X, y, n_boot).flatten(), n_boot.sum()


class BootstrapExecutor:
    """Encapsulates bootstrap execution logic"""
    
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
        print(f"Starting bootstrap with {self.n_bootstraps} iterations (sequential)")
        results = [iteration_func(args) for args in tqdm(args_list)]
        return self._parse_results(results)
    
    def _run_parallel(self, iteration_func, args_list) -> Tuple[np.ndarray, np.ndarray]:
        from joblib import Parallel, delayed
        
        print(f"Starting bootstrap with {self.n_bootstraps} iterations ({self.n_jobs} jobs)")
        results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(iteration_func)(args) for args in args_list
        )
        return self._parse_results(results)
    
    def _parse_results(self, results) -> Tuple[np.ndarray, np.ndarray]:
        boot_coefs = np.array([r[0] for r in results])
        boot_sizes = np.array([r[1] for r in results])
        return boot_coefs, boot_sizes


# ============================================================================
# Variance-Covariance Computation Mixin
# ============================================================================

class VCovMixin:
    """Mixin providing variance-covariance computation methods.
    
    Requires the host class to have:
    - self.point_estimate: np.ndarray
    - self.n_obs: int
    - self.se: str (will be set by this mixin)
    """
    
    def compute_vcov_from_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        cluster_ids: Optional[np.ndarray] = None,
        se_type: str = "stata"
    ) -> Tuple[np.ndarray, str, Optional[int]]:
        """
        Compute variance-covariance matrix.
        
        Args:
            X: Design matrix
            y: Response variable
            weights: Frequency weights
            cluster_ids: Optional cluster identifiers
            se_type: SE type ("stata", "HC0", "HC1")
        
        Returns:
            Tuple of (vcov matrix, se_type string, n_clusters or None)
        """
        theta = self.point_estimate.flatten()
        n_obs = int(weights.sum())
        n_features = X.shape[1]
        
        # Compute residuals
        y_pred = X @ theta
        residuals = y.flatten() - y_pred
        
        # Compute XtX inverse
        sqrt_w = np.sqrt(weights).reshape(-1)
        Xw = X * sqrt_w.reshape(-1, 1)
        XtX = Xw.T @ Xw + 1e-8 * np.eye(n_features)
        
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)
        
        if cluster_ids is not None:
            vcov, n_clusters = self._compute_cluster_vcov(
                X, residuals, weights, cluster_ids, XtX_inv, n_obs, n_features, se_type
            )
            return vcov, "cluster", n_clusters
        else:
            vcov = self._compute_robust_vcov(
                X, residuals, weights, XtX_inv, n_obs, n_features, se_type
            )
            return vcov, se_type, None
    
    def _compute_cluster_vcov(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        weights: np.ndarray,
        cluster_ids: np.ndarray,
        XtX_inv: np.ndarray,
        n_obs: int,
        n_features: int,
        se_type: str
    ) -> Tuple[np.ndarray, int]:
        """Compute cluster-robust variance-covariance matrix."""
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)
        
        sqrt_w = np.sqrt(weights).reshape(-1)
        n_k = n_features
        
        meat = np.zeros((n_k, n_k))
        for cluster in unique_clusters:
            mask = cluster_ids == cluster
            X_g = X[mask] * sqrt_w[mask].reshape(-1, 1)
            e_g = residuals[mask] * sqrt_w[mask]
            score_g = (X_g * e_g.reshape(-1, 1)).sum(axis=0)
            meat += np.outer(score_g, score_g)
        
        # Small sample correction
        if se_type == "stata":
            correction = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_k))
        elif se_type == "HC0":
            correction = 1.0
        else:
            correction = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_k))
        
        vcov = correction * XtX_inv @ meat @ XtX_inv
        return 0.5 * (vcov + vcov.T), n_clusters
    
    def _compute_robust_vcov(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        weights: np.ndarray,
        XtX_inv: np.ndarray,
        n_obs: int,
        n_features: int,
        se_type: str
    ) -> np.ndarray:
        """Compute heteroskedasticity-robust variance-covariance matrix."""
        sqrt_w = np.sqrt(weights).reshape(-1)
        Xw = X * sqrt_w.reshape(-1, 1)
        
        # HC1 factor
        hc1_factor = n_obs / max(1, n_obs - n_features)
        resid_sq = (residuals * sqrt_w) ** 2
        meat = (Xw.T * resid_sq) @ Xw * hc1_factor
        
        vcov = XtX_inv @ meat @ XtX_inv
        return 0.5 * (vcov + vcov.T)


# ============================================================================
# SQL Builder Mixin
# ============================================================================

class SQLBuilderMixin:
    """Mixin providing SQL building utilities for DuckDB operations.
    
    Requires the host class to have:
    - self.conn: duckdb connection
    - self.table_name: str
    - self.formula: Formula object
    """
    
    def _get_boolean_columns(self) -> Set[str]:
        """Get boolean columns from source table (cached)."""
        if hasattr(self, '_boolean_cols'):
            return self._boolean_cols
        
        all_cols = set(self.formula.get_source_columns_for_null_check())
        cols_sql = ', '.join(f"'{c}'" for c in all_cols)
        query = f"""
        SELECT column_name FROM (DESCRIBE SELECT * FROM {self.table_name})
        WHERE column_name IN ({cols_sql}) AND column_type = 'BOOLEAN'
        """
        self._boolean_cols = set(self.conn.execute(query).fetchdf()['column_name'].tolist())
        return self._boolean_cols

    def _get_table_columns(self, table_name: str) -> Set[str]:
        """Get column names from a table."""
        return set(
            self.conn.execute(f"SELECT column_name FROM (DESCRIBE {table_name})")
            .fetchdf()['column_name'].tolist()
        )

    def _build_round_expr(self, expr: str, alias: str) -> Tuple[str, str]:
        """Build expression with optional rounding for data compression."""
        if hasattr(self, 'round_strata') and self.round_strata is not None:
            round_expr = f"ROUND({expr}, {self.round_strata})"
            return f"{round_expr} AS {alias}", round_expr
        return f"{expr} AS {alias}", expr

    def _build_agg_columns(self, outcome_vars, boolean_cols: Set[str], unit_col: Optional[str]) -> List[str]:
        """Build aggregation column expressions."""
        agg_parts = ["COUNT(*) as count"]
        for var in outcome_vars:
            col_expr = var.get_sql_expression(unit_col, 'year')
            from ..formula_parser import cast_if_boolean
            col_expr = cast_if_boolean(col_expr, var.name, boolean_cols)
            agg_parts.append(f"SUM({col_expr}) as sum_{var.sql_name}")
            agg_parts.append(f"SUM(({col_expr}) * ({col_expr})) as sum_{var.sql_name}_sq")
        return agg_parts


# ============================================================================
# Mundlak Device Mixin
# ============================================================================

class MundlakMixin:
    """Mixin providing Mundlak device (group means) computation.
    
    The Mundlak device absorbs fixed effects by including group means
    of covariates as additional regressors.
    
    Requires the host class to have:
    - self.formula: Formula object
    - self.fe_cols: List[str]
    """
    
    def compute_mundlak_means(
        self, 
        df: pd.DataFrame, 
        cov_cols: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute group means for Mundlak device.
        
        Args:
            df: DataFrame with data
            cov_cols: Columns to compute means for
            
        Returns:
            Tuple of (mean values array, column names)
        """
        mean_parts = []
        mean_names = []
        
        for i, fe_name in enumerate(self.fe_cols):
            # Find the FE column in the dataframe
            fe_col = self._get_fe_sql_name(fe_name)
            
            if fe_col not in df.columns:
                logger.debug(f"FE column {fe_col} not found in dataframe")
                continue
            
            for cov_col in cov_cols:
                if cov_col not in df.columns:
                    continue
                group_means = df.groupby(fe_col)[cov_col].transform('mean')
                mean_parts.append(group_means.values.reshape(-1, 1))
                mean_names.append(f"avg_{cov_col}_fe{i}")
        
        if mean_parts:
            return np.hstack(mean_parts), mean_names
        return np.empty((len(df), 0)), []
    
    def _get_fe_sql_name(self, fe_name: str) -> str:
        """Get SQL-safe column name for a fixed effect."""
        fe_var = self.formula.get_fe_by_name(fe_name)
        if fe_var:
            return fe_var.sql_name
        
        mfe = self.formula.get_merged_fe_by_name(fe_name)
        if mfe:
            return mfe.sql_name
        
        return fe_name
