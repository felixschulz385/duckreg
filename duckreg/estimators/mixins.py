"""
Mixins providing reusable functionality for estimators.

This module follows Interface Segregation Principle - each mixin provides
a focused set of related methods that can be composed as needed.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional, Set, Dict, Any
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
# SQL Builder Mixin
# ============================================================================

class SQLBuilderMixin:
    """Mixin providing SQL building utilities for DuckDB operations.
    
    Delegates to duckreg.core.sql_builders for actual query construction.
    
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
        """Build expression with optional rounding for data compression.
        
        Delegates to sql_builders.build_round_expr.
        """
        from ..core.sql_builders import build_round_expr
        round_strata = getattr(self, 'round_strata', None)
        return build_round_expr(expr, alias, round_strata)

    def _build_agg_columns(self, outcome_vars, boolean_cols: Set[str], unit_col: Optional[str]) -> List[str]:
        """Build aggregation column expressions.
        
        Delegates to sql_builders.build_agg_columns.
        """
        from ..core.sql_builders import build_agg_columns
        return build_agg_columns(outcome_vars, boolean_cols, unit_col)
    
    # -------------------------------------------------------------------------
    # Column List Builders for DuckDB Fitter
    # -------------------------------------------------------------------------
    
    def build_x_cols_for_duckdb(
        self, 
        fe_method: str = "demean",
        fe_cols: Optional[List[str]] = None,
        is_iv: bool = False,
        endogenous_vars: Optional[List[str]] = None
    ) -> List[str]:
        """Build X column names for DuckDB fitter.
        
        Args:
            fe_method: Method for handling fixed effects ('demean' or 'mundlak')
            fe_cols: List of fixed effect column names
            is_iv: Whether this is IV/2SLS regression
            endogenous_vars: List of endogenous variable display names (for IV)
            
        Returns:
            List of SQL column names for X matrix
        """
        x_cols = []
        
        if not is_iv:
            # Standard OLS case
            for var in self.formula.covariates:
                if not var.is_intercept():
                    x_cols.append(var.sql_name)
            
            # Mundlak means if applicable
            if fe_method == "mundlak" and fe_cols:
                simple_covs = [var for var in self.formula.covariates if not var.is_intercept()]
                for i in range(len(fe_cols)):
                    for var in simple_covs:
                        x_cols.append(f"avg_{var.sql_name}_fe{i}")
        else:
            # 2SLS case
            endogenous_set = set(endogenous_vars or [])
            
            # Exogenous covariates
            for var in self.formula.covariates:
                if not var.is_intercept() and var.display_name not in endogenous_set:
                    x_cols.append(var.sql_name)
            
            # Mundlak means for exogenous
            if fe_method == "mundlak" and fe_cols:
                for var in self.formula.covariates:
                    if not var.is_intercept() and var.display_name not in endogenous_set:
                        for i in range(len(fe_cols)):
                            x_cols.append(f"avg_{var.sql_name}_fe{i}")
                
                # Mundlak means for fitted endogenous
                for var in self.formula.endogenous:
                    for i in range(len(fe_cols)):
                        x_cols.append(f"avg_fitted_{var.sql_name}_fe{i}")
            
            # Fitted endogenous
            for var in self.formula.endogenous:
                x_cols.append(f"fitted_{var.sql_name}")
        
        return x_cols
    
    def build_z_cols_for_duckdb(
        self,
        fe_method: str = "demean",
        fe_cols: Optional[List[str]] = None,
        endogenous_vars: Optional[List[str]] = None
    ) -> List[str]:
        """Build instrument (Z) column names for first-stage DuckDB fitter.
        
        Args:
            fe_method: Method for handling fixed effects
            fe_cols: List of fixed effect column names
            endogenous_vars: List of endogenous variable display names
            
        Returns:
            List of SQL column names for Z matrix (exogenous + instruments + Mundlak means)
        """
        z_cols = []
        endogenous_set = set(endogenous_vars or [])
        
        # Exogenous covariates
        for var in self.formula.covariates:
            if not var.is_intercept() and var.display_name not in endogenous_set:
                z_cols.append(var.sql_name)
        
        # Instruments
        for var in self.formula.instruments:
            z_cols.append(var.sql_name)
        
        # Mundlak means if applicable
        if fe_method == "mundlak" and fe_cols:
            # Means of exogenous covariates
            for var in self.formula.covariates:
                if not var.is_intercept() and var.display_name not in endogenous_set:
                    for i in range(len(fe_cols)):
                        z_cols.append(f"avg_{var.sql_name}_fe{i}")
            
            # Means of instruments
            for var in self.formula.instruments:
                for i in range(len(fe_cols)):
                    z_cols.append(f"avg_{var.sql_name}_fe{i}")
        
        return z_cols
    
    def build_residual_x_cols_for_duckdb(
        self,
        fe_method: str = "demean",
        fe_cols: Optional[List[str]] = None,
        endogenous_vars: Optional[List[str]] = None
    ) -> List[str]:
        """Build X column names using ACTUAL endogenous for residual computation (2SLS).
        
        This is critical for correct 2SLS standard errors: residuals must use
        actual endogenous values, not fitted ones. The order matches
        build_x_cols_for_duckdb so coefficients align correctly.
        
        Args:
            fe_method: Method for handling fixed effects
            fe_cols: List of fixed effect column names
            endogenous_vars: List of endogenous variable display names
            
        Returns:
            List of SQL column expressions for residual computation
        """
        actual_x_cols = []
        endogenous_set = set(endogenous_vars or [])
        
        # Exogenous covariates (same as fitted)
        for var in self.formula.covariates:
            if not var.is_intercept() and var.display_name not in endogenous_set:
                actual_x_cols.append(var.sql_name)
        
        # Mundlak means for exogenous (same as fitted)
        if fe_method == "mundlak" and fe_cols:
            for var in self.formula.covariates:
                if not var.is_intercept() and var.display_name not in endogenous_set:
                    for i in range(len(fe_cols)):
                        actual_x_cols.append(f"avg_{var.sql_name}_fe{i}")
            
            # Mundlak means for fitted endogenous (keep fitted for consistency)
            for var in self.formula.endogenous:
                for i in range(len(fe_cols)):
                    actual_x_cols.append(f"avg_fitted_{var.sql_name}_fe{i}")
        
        # ACTUAL endogenous (key difference: use actual, not fitted)
        # For compressed data, these are sum columns divided by count
        for var in self.formula.endogenous:
            actual_x_cols.append(f"sum_{var.sql_name} / count")
        
        return actual_x_cols


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
