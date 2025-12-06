import numpy as np
import pandas as pd
from typing import Tuple, Optional, List

from ..formula_parser import needs_quoting, quote_identifier
from .DuckLinearModel import DuckLinearModel


# ============================================================================
# DuckMundlak (Mundlak device approach)
# ============================================================================

class DuckMundlak(DuckLinearModel):
    """OLS with fixed effects via Mundlak device"""
    
    _CLUSTER_ALIAS = "__cluster__"
    _DESIGN_MATRIX_TABLE = "design_matrix"

    def _get_n_coefs(self) -> int:
        simple_covs = len(self._get_non_intercept_covariate_names())
        rhs_count = 1 + len(self.covariates) + len(self.fe_cols) * simple_covs
        return rhs_count * len(self.outcome_vars)

    def _get_non_intercept_covariate_names(self) -> List[str]:
        """Get simple covariate names excluding intercept"""
        return [var.name for var in self.formula.covariates if not var.is_intercept()]

    def _has_intercept_in_covariates(self) -> bool:
        """Check if formula already includes an intercept"""
        return any(var.is_intercept() for var in self.formula.covariates)

    def _get_cluster_data_for_bootstrap(self) -> Tuple[pd.DataFrame, str]:
        self._ensure_data_fetched()
        return self.df_compressed, self._CLUSTER_ALIAS

    def _get_cluster_col_for_vcov(self) -> str:
        return self._CLUSTER_ALIAS

    def _needs_intercept_for_duckdb(self) -> bool:
        """Mundlak device always requires an intercept.
        
        Unlike demeaning which removes the intercept by centering data,
        the Mundlak approach adds group means as regressors but keeps
        the data in levels, so an intercept is always needed.
        """
        return True

    def prepare_data(self):
        """Create design matrix with Mundlak averages"""
        boolean_cols = self._get_boolean_columns()
        unit_col = self._get_unit_col()
        
        select_parts = [
            self.formula.get_fe_select_sql(boolean_cols),
            self.formula.get_outcomes_select_sql(unit_col, 'year', boolean_cols),
            self.formula.get_covariates_select_sql(unit_col, 'year', boolean_cols, include_interactions=True),
        ]
        
        cluster_sql = self.formula.get_cluster_select_sql(boolean_cols, self._CLUSTER_ALIAS, unit_col)
        if cluster_sql:
            select_parts.append(cluster_sql)
        
        self.conn.execute(f"""
        CREATE OR REPLACE TABLE {self._DESIGN_MATRIX_TABLE} AS
        SELECT {', '.join(p for p in select_parts if p)}
        FROM {self.table_name}
        {self._build_where_clause(self.subset)}
        """)
        
        self._add_fe_averages()

    def _add_fe_averages(self):
        """Add FE-level averages for Mundlak device (excluding intercept)"""
        
        simple_covs = [var for var in self.formula.covariates if not var.is_intercept()]
        
        if not simple_covs:
            return
        
        for i, fe_col in enumerate(self.fe_cols):
            # Build AVG expressions using SQL-safe names
            avg_parts = []
            avg_col_parts = []
            for cov_var in simple_covs:
                avg_alias = f"avg_{cov_var.sql_name}_fe{i}"
                avg_parts.append(f"AVG({cov_var.sql_name}) AS {avg_alias}")
                avg_col_parts.append(f"fe{i}.{avg_alias}")
            
            avg_cols = ", ".join(avg_parts)
            avg_col_list = ", ".join(avg_col_parts)
            
            # Find FE column's SQL name
            fe_var = self.formula.get_fe_by_name(fe_col)
            mfe = self.formula.get_merged_fe_by_name(fe_col)
            fe_sql_name = fe_var.sql_name if fe_var else (mfe.sql_name if mfe else fe_col)
            
            self.conn.execute(f"""
            CREATE OR REPLACE TABLE {self._DESIGN_MATRIX_TABLE} AS
            SELECT dm.*, {avg_col_list}
            FROM {self._DESIGN_MATRIX_TABLE} dm
            JOIN (SELECT {fe_sql_name}, {avg_cols} FROM {self._DESIGN_MATRIX_TABLE} GROUP BY {fe_sql_name}) fe{i} 
            ON dm.{fe_sql_name} = fe{i}.{fe_sql_name}
            """)

    def compress_data(self):
        """Compress design matrix - creates view, doesn't fetch for duckdb fitter"""
        
        simple_covs = [var for var in self.formula.covariates if not var.is_intercept()]
        
        # Get covariate SQL names (excluding intercept) - use sql_name for SQL
        cov_sql_names = [var.sql_name for var in simple_covs]
        avg_cols = [f"avg_{var.sql_name}_fe{i}" 
                    for i in range(len(self.fe_cols)) 
                    for var in simple_covs]
        
        strata_cols_to_round = cov_sql_names + avg_cols
        self.strata_cols = strata_cols_to_round + [self._CLUSTER_ALIAS]
        
        # Build select/group clauses - no quoting needed with sql_name
        select_parts, group_parts = [], []
        for col in strata_cols_to_round:
            select_expr, group_expr = self._build_round_expr(col, col)
            select_parts.append(select_expr)
            group_parts.append(group_expr)
        
        # Add cluster column (no rounding for cluster)
        select_parts.append(self._CLUSTER_ALIAS)
        group_parts.append(self._CLUSTER_ALIAS)
        
        select_clause = ", ".join(select_parts)
        group_by_clause = ", ".join(group_parts)
        
        # Build outcome aggregations using SQL-safe names
        outcome_aggs = []
        for var in self.formula.outcomes:
            outcome_aggs.append(f"SUM({var.sql_name}) as sum_{var.sql_name}")
        outcome_aggs_sql = ", ".join(outcome_aggs)
        
        self.agg_query = f"""
        SELECT {select_clause}, COUNT(*) as count, {outcome_aggs_sql}
        FROM {self._DESIGN_MATRIX_TABLE}
        GROUP BY {group_by_clause}
        """
        
        self._create_compressed_view()
        self._rhs_cols = cov_sql_names + avg_cols

    def _get_y_col_for_duckdb(self) -> str:
        """Get the y column name for DuckDB fitter"""
        outcome_var = self.formula.outcomes[0]
        return f"sum_{outcome_var.sql_name}"

    def _get_x_cols_for_duckdb(self) -> List[str]:
        simple_covs = [var for var in self.formula.covariates if not var.is_intercept()]
        avg_cols = [f"avg_{var.sql_name}_fe{i}" 
                    for i in range(len(self.fe_cols)) 
                    for var in simple_covs]
        
        # Return SQL-safe names (no quoting)
        return [var.sql_name for var in simple_covs] + avg_cols

    def collect_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect data with Mundlak averages as additional regressors"""
        simple_covs = [var for var in self.formula.covariates if not var.is_intercept()]
        
        # Build column names using sql_name for data access
        avg_cols_sql = [f"avg_{var.sql_name}_fe{i}" 
                        for i in range(len(self.fe_cols)) 
                        for var in simple_covs]
        rhs_cols_sql = [var.sql_name for var in simple_covs] + avg_cols_sql
        
        # Build display names for coefficients (user-facing)
        avg_cols_display = [f"avg_{var.display_name}_fe{i}" 
                            for i in range(len(self.fe_cols)) 
                            for var in simple_covs]
        
        # Always add intercept column
        X = np.c_[np.ones(len(data)), data[rhs_cols_sql].values]
        
        # Use display names for coefficient names (user-facing)
        self.coef_names_ = ['Intercept'] + [var.display_name for var in simple_covs] + avg_cols_display
        
        # Use SQL-safe names for outcome columns
        y_cols = [f"mean_{var.sql_name}" for var in self.formula.outcomes]
        y = data[y_cols].values
        n = data["count"].values

        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X

        return y, X, n

