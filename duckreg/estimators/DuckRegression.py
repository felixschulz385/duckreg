import numpy as np
import pandas as pd
from typing import Tuple, Optional, List

from ..demean import demean, _convert_to_int
from ..formula_parser import quote_identifier, cast_if_boolean, _make_sql_safe_name
from .DuckLinearModel import DuckLinearModel


# ============================================================================
# DuckRegression (demeaning approach)
# ============================================================================

class DuckRegression(DuckLinearModel):
    """OLS with fixed effects via demeaning"""
    
    def __init__(self, rowid_col: str = "rowid", **kwargs):
        super().__init__(**kwargs)
        self.rowid_col = rowid_col
        # Use raw names for internal logic
        self.strata_cols = self.covariates + self.fe_cols

    def _get_n_coefs(self) -> int:
        n_covs = len(self.covariates) if self.fe_cols else len(self.covariates) + 1
        return n_covs * len(self.outcome_vars)

    def _get_cluster_data_for_bootstrap(self) -> Tuple[pd.DataFrame, Optional[str]]:
        self._ensure_data_fetched()
        return self.df_compressed, self.cluster_col

    def prepare_data(self):
        pass

    def compress_data(self):
        """Compress data by grouping on strata columns - creates view, doesn't fetch"""
        boolean_cols = self._get_boolean_columns()
        unit_col = self._get_unit_col()
        
        select_parts, group_by_parts = self._build_strata_select_sql(boolean_cols, unit_col)
        
        if self.cluster_col:
            cluster_expr = f"CAST({self.cluster_col} AS SMALLINT)" if self.cluster_col in boolean_cols else self.cluster_col
            select_parts.append(f"{cluster_expr} AS {self.cluster_col}")
            group_by_parts.append(cluster_expr)
        
        agg_parts = self._build_agg_columns(self.formula.outcomes, boolean_cols, unit_col)
        
        self.agg_query = f"""
        SELECT {', '.join(select_parts)}, {', '.join(agg_parts)}
        FROM {self.table_name}
        {self._build_where_clause(self.subset)}
        GROUP BY {', '.join(group_by_parts)}
        HAVING count IS NOT NULL
        """
        
        self._create_compressed_view()
        
        # Build expected columns using sql_name for covariates and display_name for outcomes
        strata_sql_names = []
        for col_name in self.strata_cols:
            # Get the Variable object
            var = (self.formula.get_covariate_by_name(col_name) or 
                   self.formula.get_fe_by_name(col_name))
            if var:
                strata_sql_names.append(var.sql_name)
            else:
                # Fallback for merged FEs or interactions
                interaction = self.formula.get_interaction_by_name(col_name)
                if interaction:
                    strata_sql_names.append(interaction.sql_name)
                else:
                    mfe = self.formula.get_merged_fe_by_name(col_name)
                    if mfe:
                        strata_sql_names.append(mfe.sql_name)
                    else:
                        strata_sql_names.append(col_name)
        
        self._expected_cols = (
            strata_sql_names + 
            ([self.cluster_col] if self.cluster_col else []) +
            ["count"] + 
            [f"sum_{v.sql_name}" for v in self.formula.outcomes] + 
            [f"sum_{v.sql_name}_sq" for v in self.formula.outcomes]
        )

    def _ensure_data_fetched(self):
        """Override to handle column renaming after fetch"""
        if self._data_fetched:
            return
        
        self.df_compressed = self.conn.execute(
            f"SELECT * FROM {self._COMPRESSED_VIEW}"
        ).fetchdf().dropna()
        
        if hasattr(self, '_expected_cols'):
            self.df_compressed.columns = self._expected_cols
        
        self._data_fetched = True
        self._compute_means()

    def _build_strata_select_sql(self, boolean_cols: set, unit_col: Optional[str]) -> Tuple[List[str], List[str]]:
        """Build SELECT and GROUP BY parts for strata columns"""
        select_parts, group_by_parts = [], []
        
        for col_name in self.strata_cols:
            # col_name is the raw name - use it to look up the Variable object
            var = (self.formula.get_covariate_by_name(col_name) or 
                   self.formula.get_fe_by_name(col_name))
            
            if var:
                col_expr = var.get_sql_expression(unit_col, 'year')
                col_expr = cast_if_boolean(col_expr, var.name, boolean_cols)
                # Use sql_name as the SQL-safe alias
                select_expr, group_expr = self._build_round_expr(col_expr, var.sql_name)
            else:
                # Handle interactions or merged FEs
                interaction = self.formula.get_interaction_by_name(col_name)
                if interaction:
                    col_expr = interaction.get_sql_expression(unit_col, 'year', boolean_cols)
                    select_expr, group_expr = self._build_round_expr(col_expr, interaction.sql_name)
                else:
                    mfe = self.formula.get_merged_fe_by_name(col_name)
                    if mfe:
                        col_expr = mfe.get_sql_expression(boolean_cols)
                        select_expr, group_expr = self._build_round_expr(col_expr, mfe.sql_name)
                    else:
                        # Fallback: use the raw column name
                        col_expr = self.formula.get_covariate_expression(col_name, unit_col, 'year', boolean_cols)
                        if col_expr == quote_identifier(col_name) or col_expr == col_name:
                            col_expr = self.formula.get_fe_expression(col_name, boolean_cols)
                        # Generate a SQL-safe name for fallback
                        safe_name = _make_sql_safe_name(col_name)
                        select_expr, group_expr = self._build_round_expr(col_expr, safe_name)
            
            select_parts.append(select_expr)
            group_by_parts.append(group_expr)
        
        return select_parts, group_by_parts

    def collect_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect and demean data"""
        # Use sql_name to find columns in the dataframe (they were aliased with sql_name)
        outcome_sql_names = self.formula.get_outcome_sql_names()
        pattern = f"mean_({'|'.join(outcome_sql_names)})"
        y = data.filter(regex=pattern, axis=1).values
        
        # Use sql_names for covariates too
        covariate_sql_names = self.formula.get_covariate_sql_names()
        X = data[covariate_sql_names].values
        n = data["count"].values

        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X

        if self.fe_cols:
            # Use sql_names for FE columns
            fe_sql_names = self.formula.get_fe_sql_names()
            fe = _convert_to_int(data[fe_sql_names])
            fe = fe.reshape(-1, 1) if fe.ndim == 1 else fe
            y, _ = demean(x=y, flist=fe, weights=n)
            X, _ = demean(x=X, flist=fe, weights=n)
            # For coef_names, use display names (for output)
            self.coef_names_ = self.formula.get_covariate_display_names()
        else:
            X = np.c_[np.ones(X.shape[0]), X]
            # For coef_names, use display names (for output)
            self.coef_names_ = ['Intercept'] + self.formula.get_covariate_display_names()

        return y, X, n