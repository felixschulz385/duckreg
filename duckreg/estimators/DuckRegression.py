import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, List

from ..core.demean import demean, _convert_to_int
from ..utils.formula_parser import quote_identifier, cast_if_boolean, _make_sql_safe_name
from .DuckLinearModel import DuckLinearModel

logger = logging.getLogger(__name__)


# ============================================================================
# DuckRegression (demeaning approach)
# ============================================================================

class DuckRegression(DuckLinearModel):
    """OLS with fixed effects via demeaning.
    
    Note: When using fixed effects (fe_cols), demeaning requires loading data
    into memory. For truly out-of-core FE handling, use DuckMundlak instead.
    The DuckDB fitter can only be used without fixed effects in this estimator.
    """
    
    def __init__(self, rowid_col: str = "rowid", **kwargs):
        super().__init__(**kwargs)
        self.rowid_col = rowid_col
        # Use raw names for internal logic
        self.strata_cols = self.covariates + self.fe_cols
        
        # Warn if using duckdb fitter with FEs (demeaning requires memory)
        if self.fitter == "duckdb" and self.fe_cols:
            logger.warning(
                f"DuckRegression with fixed effects uses demeaning which requires "
                f"loading data into memory. The 'duckdb' fitter cannot avoid memory "
                f"loading for this case. Consider using DuckMundlak for out-of-core "
                f"FE estimation, or use fitter='numpy' explicitly."
            )

    def _get_n_coefs(self) -> int:
        n_covs = len(self.covariates) if self.fe_cols else len(self.covariates) + 1
        return n_covs * len(self.outcome_vars)
    
    def _build_coef_names_from_formula(self) -> List[str]:
        """Build coefficient names from formula for DuckDB fitter.
        
        With FEs, no intercept is included (absorbed by demeaning).
        Without FEs, intercept is included.
        
        Returns:
            List of display names for coefficients
        """
        from ..utils.name_utils import build_coef_name_lists
        
        include_intercept = not bool(self.fe_cols)
        display_names, sql_names = build_coef_name_lists(
            formula=self.formula,
            fe_method='demean',
            include_intercept=include_intercept,
            fe_cols=None,  # Demeaning doesn't use Mundlak means
            is_iv=False
        )
        # Store sql_names for SQL column selection
        self._coef_sql_names = sql_names
        return display_names

    def _get_cluster_data_for_bootstrap(self) -> Tuple[pd.DataFrame, Optional[str]]:
        self._ensure_data_fetched()
        return self.df_compressed, self.cluster_col

    def _build_agg_columns(self, outcome_vars, boolean_cols, unit_col):
        """Build aggregation column expressions."""
        from ..core.sql_builders import build_agg_columns
        return build_agg_columns(outcome_vars, boolean_cols, unit_col)

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

    def _build_strata_select_sql(self, boolean_cols: set, unit_col: Optional[str]) -> Tuple[List[str], List[str]]:
        """Build SELECT and GROUP BY parts for strata columns.
        
        Delegates to centralized sql_builders.build_strata_select_sql.
        """
        from ..core.sql_builders import build_strata_select_sql
        return build_strata_select_sql(
            self.formula,
            self.strata_cols,
            boolean_cols,
            unit_col,
            self.round_strata
        )

    def collect_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect and demean data"""
        # Use explicit column names to avoid matching _sq columns with regex
        y_cols = [f"sum_{var.sql_name}" for var in self.formula.outcomes]
        y = data[y_cols].values
        n = data["count"].values
        
        # Convert sum to mean for WLS (WLS will multiply by sqrt(n) internally)
        y = y / n.reshape(-1, 1)
        
        # Use sql_names for covariates too
        covariate_sql_names = self.formula.get_covariate_sql_names()
        X = data[covariate_sql_names].values

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