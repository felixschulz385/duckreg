"""
OLS with fixed effects via Mundlak device.

The Mundlak approach includes group means of covariates as additional
regressors, which absorbs the fixed effects while keeping the data
in levels.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import logging

from ..utils.formula_parser import needs_quoting, quote_identifier
from .DuckLinearModel import DuckLinearModel
from ..core.sql_builders import compute_mundlak_means_numpy
from ..utils.name_utils import build_coef_name_lists

logger = logging.getLogger(__name__)


class DuckMundlak(DuckLinearModel):
    """OLS with fixed effects via Mundlak device.
    
    The Mundlak approach includes group means of covariates as additional
    regressors, which absorbs the fixed effects while keeping the data
    in levels. This allows estimation of between-group variation effects.
    
    Out-of-Core Processing:
        This estimator supports true out-of-core processing with fitter='duckdb'.
        Unlike DuckRegression (which requires demeaning in memory), Mundlak 
        computes all transformations in SQL, allowing estimation on datasets 
        larger than available memory.
        
        Example:
            model = DuckMundlak(..., fitter='duckdb')
            model.fit()  # No data loaded into memory
    """
    
    _CLUSTER_ALIAS = "__cluster__"
    _DESIGN_MATRIX_TABLE = "design_matrix"

    # -------------------------------------------------------------------------
    # Overrides for Mundlak-specific behavior
    # -------------------------------------------------------------------------

    def _get_cluster_data_for_bootstrap(self) -> Tuple[pd.DataFrame, str]:
        self._ensure_data_fetched()
        return self.df_compressed, self._CLUSTER_ALIAS

    def _get_cluster_col_for_vcov(self) -> str:
        return self._CLUSTER_ALIAS

    def _build_round_expr(self, expr: str, alias: str):
        """Build expression with optional rounding for data compression."""
        from ..core.sql_builders import build_round_expr
        return build_round_expr(expr, alias, self.round_strata)

    def _needs_intercept_for_duckdb(self) -> bool:
        """Mundlak device always requires an intercept."""
        return True
    
    def _build_coef_names_from_formula(self) -> List[str]:
        """Build coefficient names from formula for DuckDB fitter.
        
        Mundlak includes intercept + covariates + FE averages.
        
        Returns:
            List of display names for coefficients
        """
        display_names, sql_names = build_coef_name_lists(
            formula=self.formula,
            fe_method='mundlak',
            include_intercept=True,
            fe_cols=self.fe_cols,
            is_iv=False
        )
        # Store sql_names for SQL column selection
        self._coef_sql_names = sql_names
        return display_names

    # -------------------------------------------------------------------------
    # Data preparation
    # -------------------------------------------------------------------------

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
        
        cols = self._get_design_matrix_columns()
        logger.debug(f"Design matrix columns after prepare_data: {cols}")

    def _get_design_matrix_columns(self) -> List[str]:
        """Get column names from design matrix table"""
        return self.conn.execute(
            f"SELECT column_name FROM (DESCRIBE {self._DESIGN_MATRIX_TABLE})"
        ).fetchdf()['column_name'].tolist()

    def _add_fe_averages(self):
        """Add FE-level averages for Mundlak device (excluding intercept)"""
        from ..core.sql_builders import build_add_mundlak_means_sql
        
        simple_covs = [var for var in self.formula.covariates if not var.is_intercept()]
        
        if not simple_covs:
            return
        
        var_sql_names = [var.sql_name for var in simple_covs]
        
        for i, fe_col_name in enumerate(self.fe_cols):
            # Get FE SQL name
            fe_var = self.formula.get_fe_by_name(fe_col_name)
            mfe = self.formula.get_merged_fe_by_name(fe_col_name)
            fe_sql_name = fe_var.sql_name if fe_var else (mfe.sql_name if mfe else fe_col_name)
            
            # Use centralized SQL builder
            sql = build_add_mundlak_means_sql(
                table_name=self._DESIGN_MATRIX_TABLE,
                var_sql_names=var_sql_names,
                fe_col_sql_name=fe_sql_name,
                fe_index=i
            )
            self.conn.execute(sql)

    # -------------------------------------------------------------------------
    # Data compression
    # -------------------------------------------------------------------------

    def compress_data(self):
        """Compress design matrix"""
        simple_covs = [var for var in self.formula.covariates if not var.is_intercept()]
        
        cov_sql_names = [var.sql_name for var in simple_covs]
        avg_cols = [f"avg_{var.sql_name}_fe{i}" 
                    for i in range(len(self.fe_cols)) 
                    for var in simple_covs]
        
        # Verify columns exist
        available_cols = set(self._get_design_matrix_columns())
        missing_cov_cols = [c for c in cov_sql_names if c not in available_cols]
        missing_avg_cols = [c for c in avg_cols if c not in available_cols]
        
        if missing_cov_cols:
            logger.warning(f"Missing covariate columns in design matrix: {missing_cov_cols}")
        if missing_avg_cols:
            logger.warning(f"Missing avg columns in design matrix: {missing_avg_cols}")
        
        # Filter to existing columns
        cov_sql_names = [c for c in cov_sql_names if c in available_cols]
        avg_cols = [c for c in avg_cols if c in available_cols]
        
        strata_cols_to_round = cov_sql_names + avg_cols
        self.strata_cols = strata_cols_to_round + [self._CLUSTER_ALIAS]
        self._rhs_cols = cov_sql_names + avg_cols
        
        select_parts, group_parts = [], []
        for col in strata_cols_to_round:
            select_expr, group_expr = self._build_round_expr(col, col)
            select_parts.append(select_expr)
            group_parts.append(group_expr)
        
        select_parts.append(self._CLUSTER_ALIAS)
        group_parts.append(self._CLUSTER_ALIAS)
        
        select_clause = ", ".join(select_parts)
        group_by_clause = ", ".join(group_parts)
        
        # Add outcome aggregations (including sum_y_sq for exact variance computation)
        outcome_aggs = []
        for var in self.formula.outcomes:
            outcome_aggs.append(f"SUM({var.sql_name}) as sum_{var.sql_name}")
            outcome_aggs.append(f"SUM(({var.sql_name}) * ({var.sql_name})) as sum_{var.sql_name}_sq")
        outcome_aggs_sql = ", ".join(outcome_aggs)
        
        self.agg_query = f"""
        SELECT {select_clause}, COUNT(*) as count, {outcome_aggs_sql}
        FROM {self._DESIGN_MATRIX_TABLE}
        GROUP BY {group_by_clause}
        """
        
        logger.debug(f"Compress query: {self.agg_query}")
        self._create_compressed_view()

    # -------------------------------------------------------------------------
    # DuckDB fitter interface
    # -------------------------------------------------------------------------

    def _get_y_col_for_duckdb(self) -> str:
        outcome_var = self.formula.outcomes[0]
        return f"sum_{outcome_var.sql_name}"

    def _get_x_cols_for_duckdb(self) -> List[str]:
        """Get X column names for DuckDB fitter"""
        from ..core.sql_builders import build_x_cols_for_duckdb
        return build_x_cols_for_duckdb(
            formula=self.formula,
            fe_method='mundlak',
            fe_cols=self.fe_cols,
            is_iv=False
        )

    # -------------------------------------------------------------------------
    # Data collection
    # -------------------------------------------------------------------------

    def collect_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect data with Mundlak averages as additional regressors"""
        simple_covs = [var for var in self.formula.covariates if not var.is_intercept()]
        
        # Use stored _rhs_cols
        if hasattr(self, '_rhs_cols') and self._rhs_cols:
            rhs_cols_sql = self._rhs_cols
        else:
            avg_cols_sql = [f"avg_{var.sql_name}_fe{i}" 
                            for i in range(len(self.fe_cols)) 
                            for var in simple_covs]
            rhs_cols_sql = [var.sql_name for var in simple_covs] + avg_cols_sql
        
        # Verify columns exist
        missing_cols = [c for c in rhs_cols_sql if c not in data.columns]
        if missing_cols:
            logger.error(f"Missing columns in compressed data: {missing_cols}")
            logger.debug(f"Available columns: {list(data.columns)}")
            raise KeyError(f"Missing columns in compressed data: {missing_cols}")
        
        # Build display names for coefficients
        avg_cols_display = [f"avg_{var.display_name}_fe{i}" 
                            for i in range(len(self.fe_cols)) 
                            for var in simple_covs]
        
        X = np.c_[np.ones(len(data)), data[rhs_cols_sql].values]
        self.coef_names_ = ['Intercept'] + [var.display_name for var in simple_covs] + avg_cols_display
        
        y_cols = [f"sum_{var.sql_name}" for var in self.formula.outcomes]
        y = data[y_cols].values
        n = data["count"].values
        
        # Convert sum to mean for WLS (WLS will multiply by sqrt(n) internally)
        y = y / n.reshape(-1, 1)

        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X

        return y, X, n

