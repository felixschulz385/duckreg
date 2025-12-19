"""
Two-Stage Least Squares (2SLS) / Instrumental Variables Estimator

This module provides a self-contained 2SLS estimator that computes correct 
standard errors using residuals from actual endogenous variables.

Key Design Principles:
1. Self-contained: Does not delegate to sub-estimators to avoid state conflicts
2. Direct computation: Builds design matrices directly in Python/numpy or SQL
3. Correct SEs: Uses actual endogenous values for residual computation
4. Mundlak device: Handles FEs by including group means as regressors

Out-of-Core Processing:
    When using fitter='duckdb', all computations are done in SQL without loading
    full data into memory. First stage coefficients are computed via sufficient
    statistics, fitted values are computed in SQL, and second stage uses the
    DuckDBFitter with pre-computed residuals for correct 2SLS standard errors.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, List, Dict, Any

from ..formula_parser import cast_if_boolean, quote_identifier
from ..fitters import wls, NumpyFitter, DuckDBFitter
from ..duckreg import DuckEstimator

# Import from refactored modules - following DRY principle
from .results import RegressionResults, FirstStageResults
from .mixins import MundlakMixin, SQLBuilderMixin

logger = logging.getLogger(__name__)


class Duck2SLS(DuckEstimator, MundlakMixin, SQLBuilderMixin):
    """Two-Stage Least Squares (2SLS) / Instrumental Variables estimator.
    
    This estimator handles IV regression with:
    - Multiple endogenous variables
    - Fixed effects via Mundlak device
    - Cluster-robust and HC1 standard errors
    - Correct 2SLS standard errors using actual (not fitted) residuals
    
    The standard error formula uses residuals e = y - X_actual * β where X_actual
    contains the actual endogenous variables, not the fitted values from the
    first stage. This is the correct approach for 2SLS.
    
    Out-of-Core Processing:
        With fitter='duckdb', all computations are done in SQL:
        - First stage: DuckDBFitter computes sufficient statistics in SQL
        - Fitted values: Computed via SQL UPDATE statements
        - Second stage: DuckDBFitter with pre-computed residuals
        - Mundlak means: Computed via SQL GROUP BY
        
        This allows estimation on datasets larger than available memory.
    
    Estimation Pipeline:
    1. Create data table with all variables
    2. Run first stage for each endogenous variable (add Mundlak means in SQL)
    3. Add fitted values columns via SQL
    4. Estimate second stage coefficients 
    5. Compute residuals using actual endogenous and compute vcov
    """
    
    _DATA_TABLE = "iv_data"
    _COMPRESSED_VIEW = "iv_compressed"
    
    def __init__(
        self,
        db_name: str,
        table_name: str,
        formula,
        seed: int = 42,
        n_bootstraps: int = 0,
        round_strata: int = None,
        duckdb_kwargs: dict = None,
        subset: str = None,
        n_jobs: int = 1,
        fitter: str = "numpy",
        fe_method: str = "mundlak",
        **kwargs,
    ):
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
        self.fe_method = fe_method
        
        # Extract from formula
        self.outcome_vars = formula.get_outcome_names()
        self.fe_cols = formula.get_fe_names()
        self.cluster_col = formula.cluster.name if formula.cluster else None
        
        # IV-specific
        self.endogenous_vars = formula.get_endogenous_display_names()
        self.instrument_vars = formula.get_instrument_display_names()
        self.exogenous_vars = [
            var.display_name for var in formula.covariates 
            if not var.is_intercept() and var.display_name not in self.endogenous_vars
        ]
        self._has_intercept = any(var.is_intercept() for var in formula.covariates)
        
        # Storage
        self._first_stage_results: Dict[str, FirstStageResults] = {}
        self._results: Optional[RegressionResults] = None
        self.n_compressed_rows: Optional[int] = None
        
        # Design matrices (populated during estimation)
        self._y: Optional[np.ndarray] = None
        self._X_fitted: Optional[np.ndarray] = None
        self._X_actual: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None
        self._cluster_ids: Optional[np.ndarray] = None
        
        logger.debug(f"Duck2SLS: endogenous={self.endogenous_vars}, "
                    f"instruments={self.instrument_vars}, "
                    f"exogenous={self.exogenous_vars}, fe_method={fe_method}")

    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def first_stage(self) -> Dict[str, FirstStageResults]:
        """Get first stage results for all endogenous variables"""
        return self._first_stage_results
    
    @property
    def results(self) -> Optional[RegressionResults]:
        """Get regression results as RegressionResults object"""
        if self._results is not None:
            return self._results
        if self.point_estimate is None:
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
    
    def get_first_stage_f_stats(self) -> Dict[str, Optional[float]]:
        """Get F-statistics for all first stages"""
        return {endog: fs.f_statistic for endog, fs in self._first_stage_results.items()}
    
    def has_weak_instruments(self) -> bool:
        """Check if any first stage has weak instruments (F < 10)"""
        return any(
            fs.is_weak_instrument for fs in self._first_stage_results.values() 
            if fs.is_weak_instrument is not None
        )

    # =========================================================================
    # Main Pipeline (implements DuckEstimator interface)
    # =========================================================================
    
    def prepare_data(self):
        """Create data table with all needed columns"""
        self._create_data_table()
        if self.fe_cols and self.fe_method == "mundlak":
            self._add_mundlak_means_sql()
    
    def compress_data(self):
        """Run first stages and prepare for second stage estimation"""
        self._run_first_stages()
        if self.fitter == "numpy":
            self._build_design_matrices()
        else:
            self._create_second_stage_view()
    
    def estimate(self) -> np.ndarray:
        """Estimate coefficients using fitted endogenous values"""
        if self.fitter == "duckdb":
            return self._estimate_duckdb()
        return self._estimate_numpy()
    
    def _estimate_numpy(self) -> np.ndarray:
        """Estimate using in-memory numpy WLS"""
        if self._X_fitted is None:
            raise ValueError("Must call compress_data() before estimate()")
        return wls(self._X_fitted, self._y, self._weights)
    
    def _estimate_duckdb(self) -> np.ndarray:
        """Estimate using DuckDB sufficient statistics (out-of-core)"""
        x_cols = self._get_x_cols_for_duckdb()
        y_col = self._get_y_col_for_duckdb()
        
        duckdb_fitter = DuckDBFitter(conn=self.conn, alpha=1e-8, se_type="stata")
        self._fitter_result = duckdb_fitter.fit(
            table_name=self._COMPRESSED_VIEW,
            x_cols=x_cols,
            y_col=y_col,
            weight_col="count",
            add_intercept=True,
            cluster_col=self.cluster_col if self.cluster_col else None,
            compute_vcov=False  # Will compute separately with actual residuals
        )
        
        self.coef_names_ = self._build_coef_names_for_duckdb()
        self._coef_sql_names = self._build_coef_sql_names_for_duckdb()
        return self._fitter_result.coefficients
    
    def fit_vcov(self):
        """Compute vcov with correct 2SLS formula using actual residuals"""
        if self.fitter == "duckdb":
            self._fit_vcov_duckdb()
        else:
            self._fit_vcov_numpy()
    
    def _fit_vcov_numpy(self):
        """Compute vcov using numpy (in-memory)"""
        if self._X_actual is None or self._X_fitted is None:
            raise ValueError("Must call compress_data() before fit_vcov()")
        
        coefs = self.point_estimate.flatten()
        
        # Residuals using ACTUAL endogenous: e = y - X_actual * β
        residuals = self._y.flatten() - self._X_actual @ coefs
        
        if self._cluster_ids is not None:
            vcov, n_clusters = self._compute_cluster_robust_vcov(
                self._X_fitted, residuals, self._weights, self._cluster_ids
            )
            self.se = "cluster"
            self._n_clusters = n_clusters
        else:
            vcov = self._compute_hc1_vcov(self._X_fitted, residuals, self._weights)
            self.se = "HC1"
            self._n_clusters = None
        
        self.vcov = vcov
        self._results = None
    
    def _fit_vcov_duckdb(self):
        """Compute vcov using DuckDB with pre-computed residuals (out-of-core)"""
        # First add residual column to the view using actual endogenous
        self._add_actual_residuals_column()
        
        x_cols = self._get_x_cols_for_duckdb()
        y_col = self._get_y_col_for_duckdb()
        
        duckdb_fitter = DuckDBFitter(conn=self.conn, alpha=1e-8, se_type="stata")
        result = duckdb_fitter.fit(
            table_name=self._COMPRESSED_VIEW,
            x_cols=x_cols,
            y_col=y_col,
            weight_col="count",
            add_intercept=True,
            cluster_col=self.cluster_col if self.cluster_col else None,
            compute_vcov=True,
            coefficients=self.point_estimate,  # Use existing coefficients
            residual_col="residual_actual"  # Use actual residuals for 2SLS correction
        )
        
        self.vcov = result.vcov
        self.se = result.se_type
        self._n_clusters = result.n_clusters
        self._results = None
    
    def bootstrap(self) -> np.ndarray:
        """Bootstrap not yet implemented for 2SLS"""
        logger.warning("Bootstrap for 2SLS not implemented, using analytical SEs")
        self.fit_vcov()
        return self.vcov

    # =========================================================================
    # Data Preparation
    # =========================================================================
    
    def _create_data_table(self):
        """Create table with all variables needed for IV estimation"""
        boolean_cols = self._get_boolean_columns()
        unit_col = self.fe_cols[0] if self.fe_cols else None
        
        select_parts = []
        
        # Outcomes
        for var in self.formula.outcomes:
            expr = var.get_sql_expression(unit_col, 'year')
            expr = cast_if_boolean(expr, var.name, boolean_cols)
            select_parts.append(f"{expr} AS {var.sql_name}")
        
        # Exogenous covariates
        for var in self.formula.covariates:
            if var.is_intercept() or var.display_name in self.endogenous_vars:
                continue
            expr = var.get_sql_expression(unit_col, 'year')
            expr = cast_if_boolean(expr, var.name, boolean_cols)
            select_parts.append(f"{expr} AS {var.sql_name}")
        
        # Endogenous variables
        for var in self.formula.endogenous:
            expr = var.get_sql_expression(unit_col, 'year')
            expr = cast_if_boolean(expr, var.name, boolean_cols)
            select_parts.append(f"{expr} AS {var.sql_name}")
        
        # Instruments
        for var in self.formula.instruments:
            expr = var.get_sql_expression(unit_col, 'year')
            expr = cast_if_boolean(expr, var.name, boolean_cols)
            select_parts.append(f"{expr} AS {var.sql_name}")
        
        # Fixed effects
        fe_sql = self.formula.get_fe_select_sql(boolean_cols)
        if fe_sql:
            select_parts.append(fe_sql)
        
        # Cluster
        if self.cluster_col:
            select_parts.append(quote_identifier(self.cluster_col))
        
        self.conn.execute(f"""
        CREATE OR REPLACE TABLE {self._DATA_TABLE} AS
        SELECT {', '.join(select_parts)}
        FROM {self.table_name}
        {self._build_where_clause()}
        """)
        
        self.n_obs = self.conn.execute(
            f"SELECT COUNT(*) FROM {self._DATA_TABLE}"
        ).fetchone()[0]
        self.n_compressed_rows = self.n_obs

    def _get_boolean_columns(self) -> set:
        """Get boolean columns from source table"""
        all_cols = set(self.formula.get_source_columns_for_null_check())
        cols_sql = ', '.join(f"'{c}'" for c in all_cols)
        query = f"""
        SELECT column_name FROM (DESCRIBE SELECT * FROM {self.table_name})
        WHERE column_name IN ({cols_sql}) AND column_type = 'BOOLEAN'
        """
        return set(self.conn.execute(query).fetchdf()['column_name'].tolist())

    def _build_where_clause(self) -> str:
        """Build WHERE clause"""
        return self.formula.get_where_clause_sql(self.subset)

    def _get_table_columns(self, table_name: str) -> set:
        """Get column names from a table"""
        return set(
            self.conn.execute(f"SELECT column_name FROM (DESCRIBE {table_name})")
            .fetchdf()['column_name'].tolist()
        )
    
    def _add_mundlak_means_sql(self):
        """Add Mundlak group means columns to data table via SQL.
        
        This computes FE-level averages of covariates and instruments
        and adds them as columns, allowing DuckDB fitter to work out-of-core.
        """
        # Get columns to compute means for
        mean_cols = []
        for var in self.formula.covariates:
            if not var.is_intercept() and var.display_name not in self.endogenous_vars:
                mean_cols.append(var.sql_name)
        for var in self.formula.instruments:
            mean_cols.append(var.sql_name)
        
        if not mean_cols:
            return
        
        # Add mean columns for each FE
        for i, fe_col_name in enumerate(self.fe_cols):
            fe_var = self.formula.get_fe_by_name(fe_col_name)
            mfe = self.formula.get_merged_fe_by_name(fe_col_name)
            fe_sql_name = fe_var.sql_name if fe_var else (mfe.sql_name if mfe else fe_col_name)
            
            # Build avg column expressions
            avg_select_parts = []
            avg_col_aliases = []
            for col in mean_cols:
                avg_alias = f"avg_{col}_fe{i}"
                avg_select_parts.append(f"AVG({col}) AS {avg_alias}")
                avg_col_aliases.append(f"fe_means.{avg_alias}")
            
            avg_cols_sql = ", ".join(avg_select_parts)
            join_cols_sql = ", ".join(avg_col_aliases)
            
            self.conn.execute(f"""
            CREATE OR REPLACE TABLE {self._DATA_TABLE} AS
            SELECT t.*, {join_cols_sql}
            FROM {self._DATA_TABLE} t
            JOIN (
                SELECT {fe_sql_name}, {avg_cols_sql} 
                FROM {self._DATA_TABLE} 
                GROUP BY {fe_sql_name}
            ) fe_means ON t.{fe_sql_name} = fe_means.{fe_sql_name}
            """)
        
        logger.debug(f"Added Mundlak mean columns for {len(self.fe_cols)} FEs")
    
    def _create_second_stage_view(self):
        """Create compressed view for second stage estimation (DuckDB fitter).
        
        Groups by fitted endogenous + exogenous + Mundlak means and
        aggregates outcome for efficient sufficient statistics computation.
        """
        # Build the compression query
        outcome_var = self.formula.outcomes[0]
        
        # Columns to group by
        group_cols = []
        select_cols = []
        agg_cols = []  # Columns that need aggregation
        
        # Exogenous covariates
        for var in self.formula.covariates:
            if not var.is_intercept() and var.display_name not in self.endogenous_vars:
                group_cols.append(var.sql_name)
                select_cols.append(var.sql_name)
        
        # Fitted endogenous (group by fitted, aggregate actual)
        for var in self.formula.endogenous:
            fitted_col = f"fitted_{var.sql_name}"
            group_cols.append(fitted_col)
            select_cols.append(fitted_col)
            # Aggregate actual endogenous for residual computation
            agg_cols.append(f"SUM({var.sql_name}) as sum_{var.sql_name}")
        
        # Mundlak means
        if self.fe_cols and self.fe_method == "mundlak":
            for var in self.formula.covariates:
                if not var.is_intercept() and var.display_name not in self.endogenous_vars:
                    for i in range(len(self.fe_cols)):
                        col = f"avg_{var.sql_name}_fe{i}"
                        group_cols.append(col)
                        select_cols.append(col)
            for var in self.formula.instruments:
                for i in range(len(self.fe_cols)):
                    col = f"avg_{var.sql_name}_fe{i}"
                    group_cols.append(col)
                    select_cols.append(col)
            # Also add Mundlak means for fitted endogenous
            for var in self.formula.endogenous:
                for i in range(len(self.fe_cols)):
                    col = f"avg_fitted_{var.sql_name}_fe{i}"
                    group_cols.append(col)
                    select_cols.append(col)
        
        # Cluster column
        if self.cluster_col:
            group_cols.append(self.cluster_col)
            select_cols.append(self.cluster_col)
        
        # First add Mundlak means for fitted endogenous if needed
        if self.fe_cols and self.fe_method == "mundlak":
            self._add_fitted_mundlak_means_sql()
        
        # Build aggregation query
        group_by_sql = ", ".join(group_cols) if group_cols else "1"
        select_sql = ", ".join(select_cols) if select_cols else "1 AS dummy"
        
        # Add outcome aggregation
        agg_cols.append("COUNT(*) as count")
        agg_cols.append(f"SUM({outcome_var.sql_name}) as sum_{outcome_var.sql_name}")
        agg_sql = ", ".join(agg_cols)
        
        self.conn.execute(f"""
        CREATE OR REPLACE VIEW {self._COMPRESSED_VIEW} AS
        SELECT {select_sql},
               {agg_sql}
        FROM {self._DATA_TABLE}
        GROUP BY {group_by_sql}
        """)
        
        # Update observation counts
        result = self.conn.execute(f"""
            SELECT SUM(count) as n_obs, COUNT(*) as n_compressed 
            FROM {self._COMPRESSED_VIEW}
        """).fetchone()
        self.n_obs = int(result[0]) if result[0] else 0
        self.n_compressed_rows = int(result[1]) if result[1] else 0
        
        logger.debug(f"Created second stage view: {self.n_obs} obs, {self.n_compressed_rows} compressed")
    
    def _add_fitted_mundlak_means_sql(self):
        """Add Mundlak means for fitted endogenous variables"""
        for i, fe_col_name in enumerate(self.fe_cols):
            fe_var = self.formula.get_fe_by_name(fe_col_name)
            mfe = self.formula.get_merged_fe_by_name(fe_col_name)
            fe_sql_name = fe_var.sql_name if fe_var else (mfe.sql_name if mfe else fe_col_name)
            
            avg_select_parts = []
            avg_col_aliases = []
            
            for var in self.formula.endogenous:
                fitted_col = f"fitted_{var.sql_name}"
                avg_alias = f"avg_{fitted_col}_fe{i}"
                avg_select_parts.append(f"AVG({fitted_col}) AS {avg_alias}")
                avg_col_aliases.append(f"fe_means.{avg_alias}")
            
            if not avg_select_parts:
                continue
            
            avg_cols_sql = ", ".join(avg_select_parts)
            join_cols_sql = ", ".join(avg_col_aliases)
            
            self.conn.execute(f"""
            CREATE OR REPLACE TABLE {self._DATA_TABLE} AS
            SELECT t.*, {join_cols_sql}
            FROM {self._DATA_TABLE} t
            JOIN (
                SELECT {fe_sql_name}, {avg_cols_sql} 
                FROM {self._DATA_TABLE} 
                GROUP BY {fe_sql_name}
            ) fe_means ON t.{fe_sql_name} = fe_means.{fe_sql_name}
            """)
    
    def _add_actual_residuals_column(self):
        """Add residuals computed using actual endogenous to compressed view.
        
        For 2SLS, the residuals for vcov must use actual endogenous values:
        e = y - X_actual * β
        
        Since the data is compressed, we use mean values:
        e = (sum_y / count) - (sum of coef * variable values)
        """
        coefs = self.point_estimate.flatten()
        sql_names = self._coef_sql_names
        
        # Build the predicted value expression using SQL names
        pred_parts = []
        for sql_name, coef in zip(sql_names, coefs):
            if sql_name == 'Intercept':
                pred_parts.append(f"{coef}")
            elif sql_name in [v.sql_name for v in self.formula.endogenous]:
                # Use actual endogenous (aggregated as sum), not fitted
                pred_parts.append(f"({coef} * (sum_{sql_name} / count))")
            elif sql_name.startswith('avg_fitted_') and '_fe' in sql_name:
                # Mundlak mean of fitted - skip for residual computation
                # These are absorbed by FE and have small impact on residuals
                continue
            else:
                # Exogenous covariate or Mundlak mean - use SQL name directly
                pred_parts.append(f"({coef} * {quote_identifier(sql_name)})")
        
        pred_expr = " + ".join(pred_parts) if pred_parts else "0"
        outcome_var = self.formula.outcomes[0]
        
        # Materialize to temp table to avoid recursive view definition
        temp_table = f"{self._COMPRESSED_VIEW}_temp"
        self.conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        self.conn.execute(f"CREATE TEMP TABLE {temp_table} AS SELECT * FROM {self._COMPRESSED_VIEW}")
        
        # Recreate view with residual column from temp table
        self.conn.execute(f"""
        CREATE OR REPLACE VIEW {self._COMPRESSED_VIEW} AS
        SELECT *,
               (sum_{outcome_var.sql_name} / count) - ({pred_expr}) AS residual_actual
        FROM {temp_table}
        """)
    
    def _get_x_cols_for_duckdb(self) -> List[str]:
        """Get x column names for second stage DuckDB fitter"""
        x_cols = []
        
        # Exogenous covariates
        for var in self.formula.covariates:
            if not var.is_intercept() and var.display_name not in self.endogenous_vars:
                x_cols.append(var.sql_name)
        
        # Mundlak means for exogenous
        if self.fe_cols and self.fe_method == "mundlak":
            for var in self.formula.covariates:
                if not var.is_intercept() and var.display_name not in self.endogenous_vars:
                    for i in range(len(self.fe_cols)):
                        x_cols.append(f"avg_{var.sql_name}_fe{i}")
            for var in self.formula.instruments:
                for i in range(len(self.fe_cols)):
                    x_cols.append(f"avg_{var.sql_name}_fe{i}")
            # Mundlak means for fitted endogenous
            for var in self.formula.endogenous:
                for i in range(len(self.fe_cols)):
                    x_cols.append(f"avg_fitted_{var.sql_name}_fe{i}")
        
        # Fitted endogenous
        for var in self.formula.endogenous:
            x_cols.append(f"fitted_{var.sql_name}")
        
        return x_cols
    
    def _get_y_col_for_duckdb(self) -> str:
        """Get y column name for second stage DuckDB fitter"""
        outcome_var = self.formula.outcomes[0]
        return f"sum_{outcome_var.sql_name}"
    
    def _build_coef_names_for_duckdb(self) -> List[str]:
        """Build coefficient names for DuckDB fitter results"""
        coef_names = ['Intercept']
        
        # Exogenous covariates
        for var in self.formula.covariates:
            if not var.is_intercept() and var.display_name not in self.endogenous_vars:
                coef_names.append(var.display_name)
        
        # Mundlak means for exogenous
        if self.fe_cols and self.fe_method == "mundlak":
            for var in self.formula.covariates:
                if not var.is_intercept() and var.display_name not in self.endogenous_vars:
                    for i in range(len(self.fe_cols)):
                        coef_names.append(f"avg_{var.display_name}_fe{i}")
            for var in self.formula.instruments:
                for i in range(len(self.fe_cols)):
                    coef_names.append(f"avg_{var.display_name}_fe{i}")
            # Mundlak means for fitted endogenous
            for var in self.formula.endogenous:
                for i in range(len(self.fe_cols)):
                    coef_names.append(f"avg_{var.display_name}_fe{i}")
        
        # Endogenous
        for var in self.formula.endogenous:
            coef_names.append(var.display_name)
        
        return coef_names
    
    def _build_coef_sql_names_for_duckdb(self) -> List[str]:
        """Build SQL-safe coefficient names parallel to coef_names_"""
        coef_sql_names = ['Intercept']
        
        # Exogenous covariates
        for var in self.formula.covariates:
            if not var.is_intercept() and var.display_name not in self.endogenous_vars:
                coef_sql_names.append(var.sql_name)
        
        # Mundlak means
        if self.fe_cols and self.fe_method == "mundlak":
            for var in self.formula.covariates:
                if not var.is_intercept() and var.display_name not in self.endogenous_vars:
                    for i in range(len(self.fe_cols)):
                        coef_sql_names.append(f"avg_{var.sql_name}_fe{i}")
            for var in self.formula.instruments:
                for i in range(len(self.fe_cols)):
                    coef_sql_names.append(f"avg_{var.sql_name}_fe{i}")
            for var in self.formula.endogenous:
                for i in range(len(self.fe_cols)):
                    coef_sql_names.append(f"avg_fitted_{var.sql_name}_fe{i}")
        
        # Endogenous
        for var in self.formula.endogenous:
            coef_sql_names.append(var.sql_name)
        
        return coef_sql_names

    # =========================================================================
    # First Stage
    # =========================================================================
    
    def _run_first_stages(self):
        """Run first-stage regressions for all endogenous variables"""
        for endog_var in self.endogenous_vars:
            logger.debug(f"Running first stage for {endog_var}")
            if self.fitter == "duckdb":
                fs_result = self._run_single_first_stage_duckdb(endog_var)
            else:
                fs_result = self._run_single_first_stage_numpy(endog_var)
            self._first_stage_results[endog_var] = fs_result
            self._add_fitted_column(fs_result)
            logger.debug(f"First stage for {endog_var}: F-stat={fs_result.f_statistic}")
    
    def _run_single_first_stage_numpy(self, endog_var: str) -> FirstStageResults:
        """Run first-stage regression using numpy (in-memory)"""
        endog_var_obj = next(
            (v for v in self.formula.endogenous if v.display_name == endog_var), None
        )
        if endog_var_obj is None:
            raise ValueError(f"Endogenous variable not found: {endog_var}")
        
        df = self._fetch_first_stage_data()
        y_fs, X_fs, coef_names_fs = self._build_first_stage_matrices(df, endog_var_obj.sql_name)
        weights = np.ones(len(df))
        cluster_ids = df[self.cluster_col].values if self.cluster_col and self.cluster_col in df.columns else None
        
        coefs = wls(X_fs, y_fs, weights)
        
        fitter = NumpyFitter(alpha=1e-8, se_type="stata")
        result = fitter.fit(
            X=X_fs, y=y_fs, weights=weights,
            coef_names=coef_names_fs,
            cluster_ids=cluster_ids,
            compute_vcov=True
        )
        
        reg_results = RegressionResults(
            coefficients=coefs,
            coef_names=coef_names_fs,
            vcov=result.vcov,
            n_obs=len(df),
            se_type=result.se_type,
        )
        
        return FirstStageResults(
            endog_var=endog_var,
            results=reg_results,
            instrument_names=self.instrument_vars,
        )
    
    def _run_single_first_stage_duckdb(self, endog_var: str) -> FirstStageResults:
        """Run first-stage regression using DuckDB (out-of-core)"""
        endog_var_obj = next(
            (v for v in self.formula.endogenous if v.display_name == endog_var), None
        )
        if endog_var_obj is None:
            raise ValueError(f"Endogenous variable not found: {endog_var}")
        
        # Get column names for first stage
        x_cols = self._get_first_stage_x_cols()
        y_col = endog_var_obj.sql_name
        
        # Create first stage view (no compression, weight=1)
        self.conn.execute(f"""
        CREATE OR REPLACE VIEW first_stage_view AS
        SELECT *, 1 as count
        FROM {self._DATA_TABLE}
        WHERE {y_col} IS NOT NULL
        """)
        
        duckdb_fitter = DuckDBFitter(conn=self.conn, alpha=1e-8, se_type="stata")
        result = duckdb_fitter.fit(
            table_name="first_stage_view",
            x_cols=x_cols,
            y_col=y_col,
            weight_col="count",
            add_intercept=True,
            cluster_col=self.cluster_col if self.cluster_col else None,
            compute_vcov=True
        )
        
        coef_names_fs = self._build_first_stage_coef_names()
        
        reg_results = RegressionResults(
            coefficients=result.coefficients,
            coef_names=coef_names_fs,
            vcov=result.vcov,
            n_obs=result.n_obs,
            se_type=result.se_type,
        )
        
        return FirstStageResults(
            endog_var=endog_var,
            results=reg_results,
            instrument_names=self.instrument_vars,
        )
    
    def _get_first_stage_x_cols(self) -> List[str]:
        """Get x column names for first stage regression"""
        x_cols = []
        
        # Exogenous covariates
        for var in self.formula.covariates:
            if not var.is_intercept() and var.display_name not in self.endogenous_vars:
                x_cols.append(var.sql_name)
        
        # Instruments
        for var in self.formula.instruments:
            x_cols.append(var.sql_name)
        
        # Mundlak means (if added to table)
        if self.fe_cols and self.fe_method == "mundlak":
            for var in self.formula.covariates:
                if not var.is_intercept() and var.display_name not in self.endogenous_vars:
                    for i in range(len(self.fe_cols)):
                        x_cols.append(f"avg_{var.sql_name}_fe{i}")
            for var in self.formula.instruments:
                for i in range(len(self.fe_cols)):
                    x_cols.append(f"avg_{var.sql_name}_fe{i}")
        
        return x_cols
    
    def _build_first_stage_coef_names(self) -> List[str]:
        """Build coefficient names for first stage"""
        coef_names = ['Intercept']
        
        # Exogenous covariates
        for var in self.formula.covariates:
            if not var.is_intercept() and var.display_name not in self.endogenous_vars:
                coef_names.append(var.display_name)
        
        # Instruments
        for var in self.formula.instruments:
            coef_names.append(var.display_name)
        
        # Mundlak means
        if self.fe_cols and self.fe_method == "mundlak":
            for var in self.formula.covariates:
                if not var.is_intercept() and var.display_name not in self.endogenous_vars:
                    for i in range(len(self.fe_cols)):
                        coef_names.append(f"avg_{var.display_name}_fe{i}")
            for var in self.formula.instruments:
                for i in range(len(self.fe_cols)):
                    coef_names.append(f"avg_{var.display_name}_fe{i}")
        
        return coef_names
    
    def _fetch_first_stage_data(self) -> pd.DataFrame:
        """Fetch data for first stage (numpy fitter only)"""
        return self.conn.execute(f"SELECT * FROM {self._DATA_TABLE}").fetchdf().dropna()
    
    def _build_first_stage_matrices(
        self, df: pd.DataFrame, endog_sql_name: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Build y and X matrices for first stage with Mundlak device"""
        y = df[endog_sql_name].values.reshape(-1, 1)
        
        X_parts = [np.ones((len(df), 1))]
        coef_names = ['Intercept']
        
        # Exogenous covariates
        exog_cols = []
        for var in self.formula.covariates:
            if not var.is_intercept() and var.display_name not in self.endogenous_vars:
                if var.sql_name in df.columns:
                    exog_cols.append(var.sql_name)
                    coef_names.append(var.display_name)
        
        # Instruments
        inst_cols = []
        for var in self.formula.instruments:
            if var.sql_name in df.columns:
                inst_cols.append(var.sql_name)
                coef_names.append(var.display_name)
        
        if exog_cols:
            X_parts.append(df[exog_cols].values)
        if inst_cols:
            X_parts.append(df[inst_cols].values)
        
        # Mundlak means for FEs - uses MundlakMixin.compute_mundlak_means
        if self.fe_cols and self.fe_method == "mundlak":
            all_cov_cols = exog_cols + inst_cols
            if all_cov_cols:
                X_mundlak, names_mundlak = self.compute_mundlak_means(df, all_cov_cols)
                if X_mundlak.shape[1] > 0:
                    X_parts.append(X_mundlak)
                    coef_names.extend(names_mundlak)
        
        X = np.hstack(X_parts)
        return y, X, coef_names
    
    def _add_fitted_column(self, fs_result: FirstStageResults):
        """Add fitted values column to data table"""
        endog_var_obj = next(
            (v for v in self.formula.endogenous if v.display_name == fs_result.endog_var), None
        )
        if endog_var_obj is None:
            return
        
        fitted_col = f"fitted_{endog_var_obj.sql_name}"
        available_cols = self._get_table_columns(self._DATA_TABLE)
        
        # Build fitted expression including Mundlak means if they exist in table
        expr_parts = []
        for name, coef in zip(fs_result.coef_names, fs_result.coefficients.flatten()):
            if name == 'Intercept':
                expr_parts.append(f"{coef}")
            elif name.startswith('avg_') and '_fe' in name:
                # Mundlak means - use sql column name directly (avg_{sql_name}_fe{i})
                # Need to convert display name to sql name in the avg_ pattern
                parts = name.split('_')
                if len(parts) >= 3:
                    # Extract fe index and find the variable
                    fe_suffix = parts[-1]  # e.g., "fe0"
                    display_var_name = '_'.join(parts[1:-1])  # middle part is the display name
                    sql_col_name = None
                    for var in self.formula.covariates:
                        if var.display_name == display_var_name:
                            sql_col_name = f"avg_{var.sql_name}_{fe_suffix}"
                            break
                    if sql_col_name is None:
                        for var in self.formula.instruments:
                            if var.display_name == display_var_name:
                                sql_col_name = f"avg_{var.sql_name}_{fe_suffix}"
                                break
                    if sql_col_name and sql_col_name in available_cols:
                        expr_parts.append(f"({coef} * {quote_identifier(sql_col_name)})")
            else:
                sql_name = self._display_to_sql_name(name)
                if sql_name and sql_name in available_cols:
                    expr_parts.append(f"({coef} * {quote_identifier(sql_name)})")
        
        fitted_expr = " + ".join(expr_parts) if expr_parts else "0"
        
        self.conn.execute(f"""
        ALTER TABLE {self._DATA_TABLE} ADD COLUMN {quote_identifier(fitted_col)} DOUBLE
        """)
        self.conn.execute(f"""
        UPDATE {self._DATA_TABLE} SET {quote_identifier(fitted_col)} = {fitted_expr}
        """)
        
        logger.debug(f"Added fitted values column: {fitted_col}")
    
    def _display_to_sql_name(self, display_name: str) -> Optional[str]:
        """Map display name to sql_name"""
        for var in self.formula.covariates:
            if var.display_name == display_name:
                return var.sql_name
        for var in self.formula.instruments:
            if var.display_name == display_name:
                return var.sql_name
        for var in self.formula.endogenous:
            if var.display_name == display_name:
                return var.sql_name
        return None

    # =========================================================================
    # Second Stage Design Matrices
    # =========================================================================
    
    def _build_design_matrices(self):
        """Build design matrices for second stage estimation"""
        df = self._fetch_second_stage_data()
        
        # Outcome
        outcome_var = self.formula.outcomes[0]
        self._y = df[outcome_var.sql_name].values.reshape(-1, 1)
        
        # Build common X (exogenous + Mundlak means)
        X_common, coef_names, coef_names_sql = self._build_common_X(df)
        
        # Build endogenous columns
        X_endog_fitted, X_endog_actual, endog_names, endog_names_sql = self._build_endogenous_X(df)
        
        self._X_fitted = np.hstack([X_common, X_endog_fitted])
        self._X_actual = np.hstack([X_common, X_endog_actual])
        
        self.coef_names_ = coef_names + endog_names
        self._coef_sql_names = coef_names_sql + endog_names_sql
        self._weights = np.ones(len(df))
        
        if self.cluster_col and self.cluster_col in df.columns:
            self._cluster_ids = df[self.cluster_col].values
        else:
            self._cluster_ids = None
    
    def _fetch_second_stage_data(self) -> pd.DataFrame:
        """Fetch data for second stage"""
        return self.conn.execute(f"SELECT * FROM {self._DATA_TABLE}").fetchdf().dropna()
    
    def _build_common_X(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
        """Build exogenous part of design matrix
        
        Returns:
            Tuple of (X array, display names, SQL names)
        """
        X_parts = []
        coef_names = []
        coef_sql_names = []
        
        # Intercept
        if self._has_intercept or not self.fe_cols:
            X_parts.append(np.ones((len(df), 1)))
            coef_names.append('Intercept')
            coef_sql_names.append('Intercept')
        
        # Exogenous covariates
        exog_cols = []
        exog_names = []
        exog_sql_names = []
        for var in self.formula.covariates:
            if not var.is_intercept() and var.display_name not in self.endogenous_vars:
                if var.sql_name in df.columns:
                    exog_cols.append(var.sql_name)
                    exog_names.append(var.display_name)
                    exog_sql_names.append(var.sql_name)
        
        if exog_cols:
            X_parts.append(df[exog_cols].values)
            coef_names.extend(exog_names)
            coef_sql_names.extend(exog_sql_names)
        
        # Mundlak means - uses MundlakMixin.compute_mundlak_means
        if self.fe_cols and self.fe_method == "mundlak":
            # Get fitted endogenous column names
            fitted_cols = []
            for var in self.formula.endogenous:
                fc = f"fitted_{var.sql_name}"
                if fc in df.columns:
                    fitted_cols.append(fc)
            
            all_cov_cols = exog_cols + fitted_cols
            if all_cov_cols:
                X_mundlak, names_mundlak = self.compute_mundlak_means(df, all_cov_cols)
                if X_mundlak.shape[1] > 0:
                    X_parts.append(X_mundlak)
                    coef_names.extend(names_mundlak)
                    # Mundlak names are already SQL-safe (avg_{sql_col}_fe{i})
                    coef_sql_names.extend(names_mundlak)
        
        if X_parts:
            return np.hstack(X_parts), coef_names, coef_sql_names
        return np.ones((len(df), 1)), ['Intercept'], ['Intercept']
    
    def _build_endogenous_X(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Build endogenous part of design matrices (fitted and actual)
        
        Returns:
            Tuple of (X_fitted, X_actual, display names, SQL names)
        """
        X_fitted_parts = []
        X_actual_parts = []
        names = []
        sql_names = []
        
        for var in self.formula.endogenous:
            fitted_col = f"fitted_{var.sql_name}"
            
            if fitted_col in df.columns:
                X_fitted_parts.append(df[fitted_col].values.reshape(-1, 1))
            else:
                logger.warning(f"Fitted column {fitted_col} not found, using actual")
                X_fitted_parts.append(df[var.sql_name].values.reshape(-1, 1))
            
            X_actual_parts.append(df[var.sql_name].values.reshape(-1, 1))
            names.append(var.display_name)
            sql_names.append(var.sql_name)
        
        X_fitted = np.hstack(X_fitted_parts) if X_fitted_parts else np.empty((len(df), 0))
        X_actual = np.hstack(X_actual_parts) if X_actual_parts else np.empty((len(df), 0))
        
        return X_fitted, X_actual, names, sql_names

    # =========================================================================
    # Variance-Covariance Computation
    # =========================================================================
    
    def _compute_hc1_vcov(
        self, Z: np.ndarray, residuals: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """HC1 heteroskedasticity-robust 2SLS vcov"""
        n_obs = len(residuals)
        n_features = Z.shape[1]
        
        ZtZ = Z.T @ Z + 1e-8 * np.eye(n_features)
        try:
            ZtZ_inv = np.linalg.inv(ZtZ)
        except np.linalg.LinAlgError:
            ZtZ_inv = np.linalg.pinv(ZtZ)
        
        hc1_factor = n_obs / max(1, n_obs - n_features)
        resid_sq = residuals ** 2
        meat = (Z.T * resid_sq) @ Z * hc1_factor
        
        vcov = ZtZ_inv @ meat @ ZtZ_inv
        return 0.5 * (vcov + vcov.T)
    
    def _compute_cluster_robust_vcov(
        self, Z: np.ndarray, residuals: np.ndarray, 
        weights: np.ndarray, cluster_ids: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Cluster-robust 2SLS vcov"""
        n_obs = len(residuals)
        n_features = Z.shape[1]
        
        ZtZ = Z.T @ Z + 1e-8 * np.eye(n_features)
        try:
            ZtZ_inv = np.linalg.inv(ZtZ)
        except np.linalg.LinAlgError:
            ZtZ_inv = np.linalg.pinv(ZtZ)
        
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)
        
        meat = np.zeros((n_features, n_features))
        for cluster in unique_clusters:
            mask = cluster_ids == cluster
            score_g = (Z[mask] * residuals[mask, np.newaxis]).sum(axis=0)
            meat += np.outer(score_g, score_g)
        
        correction = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_features))
        
        vcov = correction * (ZtZ_inv @ meat @ ZtZ_inv)
        return 0.5 * (vcov + vcov.T), n_clusters

    # =========================================================================
    # Summary
    # =========================================================================
    
    def summary(self) -> Dict[str, Any]:
        """Provide comprehensive and exhaustive 2SLS results summary.
        
        Returns a dictionary containing all information needed to:
        - Reconstruct the analysis
        - Track provenance (version, timestamp)
        - Identify results from buggy versions
        
        Uses the standardized ModelSummary structure for consistency.
        
        Returns:
            Dictionary with model specification, results, first stages, and metadata
        """
        from .results import ModelSummary
        return ModelSummary.from_estimator(self).to_dict()
    
    def summary_df(self) -> pd.DataFrame:
        """Get results as a DataFrame (wide format with index).
        
        Returns:
            DataFrame with coefficient estimates and statistics as rows indexed by variable names
        """
        if self.results is None:
            return pd.DataFrame()
        return self.results.to_dataframe()
    
    def print_summary(self, precision: int = 4, include_diagnostics: bool = True):
        """Print formatted 2SLS results to console using unified formatter."""
        from .summary import print_summary as fmt_print
        fmt_print(self.summary(), precision=precision, include_diagnostics=include_diagnostics)
    
    def to_tidy_df(self) -> pd.DataFrame:
        """Get results as a tidy DataFrame using unified formatter."""
        from .summary import to_tidy_df as fmt_tidy
        if self.results:
            return fmt_tidy(self.results)
        return pd.DataFrame()