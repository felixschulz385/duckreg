import numpy as np
import pandas as pd
from typing import Tuple, Optional, List

from ..demean import demean, _convert_to_int
from ..formula_parser import quote_identifier
from .DuckLinearModel import DuckLinearModel


# ============================================================================
# DuckRegression (demeaning approach)
# ============================================================================

class DuckRegression(DuckLinearModel):
    """OLS with fixed effects via demeaning"""
    
    def __init__(self, rowid_col: str = "rowid", **kwargs):
        super().__init__(**kwargs)
        self.rowid_col = rowid_col
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
        
        # Set expected column names for later use
        self._expected_cols = (
            self.strata_cols.copy() + 
            ([self.cluster_col] if self.cluster_col else []) +
            ["count"] + 
            [f"sum_{v}" for v in self.outcome_vars] + 
            [f"sum_{v}_sq" for v in self.outcome_vars]
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
        
        for col in self.strata_cols:
            col_expr = self.formula.get_covariate_expression(col, unit_col, 'year', boolean_cols)
            if col_expr == quote_identifier(col) or col_expr == col:
                col_expr = self.formula.get_fe_expression(col, boolean_cols)
            
            select_expr, group_expr = self._build_round_expr(col_expr, col)
            select_parts.append(select_expr)
            group_by_parts.append(group_expr)
        
        return select_parts, group_by_parts

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