import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, List

from ..utils.formula_parser import quote_identifier, cast_if_boolean, _make_sql_safe_name
from .DuckLinearModel import DuckLinearModel

logger = logging.getLogger(__name__)


# ============================================================================
# DuckRegression (demeaning approach)
# ============================================================================

class DuckRegression(DuckLinearModel):
    """Pooled OLS regression without fixed effects.
    
    This estimator handles basic OLS regression with an intercept.
    For fixed effects estimation, use DuckFE.
    
    Supports both in-memory (numpy) and out-of-core (duckdb) fitting.
    """
    
    def __init__(self, rowid_col: str = "rowid", **kwargs):
        super().__init__(**kwargs)
        self.rowid_col = rowid_col
        self.n_rows_dropped_singletons = 0
        
        # Pooled OLS - only use covariates for stratification
        self.strata_cols = self.covariates
        
        # Warn if FE columns are specified (should use different estimator)
        if self.fe_cols:
            raise ValueError(
                f"DuckRegression is for pooled OLS only (no fixed effects). "
                f"Use DuckFE for fixed effects estimation."
            )

    def _get_n_coefs(self) -> int:
        # Pooled OLS always includes intercept + covariates
        return (len(self.covariates) + 1) * len(self.outcome_vars)

    def _get_vcov_fe_params(self) -> Tuple[int, int, int, int]:
        """Return (k_fe, n_fe, k_fe_nested, n_fe_fully_nested) = (0, 0, 0, 0) for pooled OLS.

        DuckRegression is purely pooled OLS with no fixed effects, so no
        additional DOF correction is required.
        """
        return 0, 0, 0, 0
    
    def _build_coef_names_from_formula(self) -> List[str]:
        """Build coefficient names from formula for DuckDB fitter.
        
        Pooled OLS always includes an intercept + covariates.
        
        Returns:
            List of display names for coefficients
        """
        from ..utils.name_utils import build_coef_name_lists
        
        display_names, sql_names = build_coef_name_lists(
            formula=self.formula,
            fe_method=None,  # Pooled OLS, no FE
            include_intercept=True,
            fe_cols=None,
            is_iv=False
        )
        # Store sql_names for SQL column selection
        self._coef_sql_names = sql_names
        return display_names

    def _get_cluster_data_for_bootstrap(self) -> Tuple[pd.DataFrame, Optional[str]]:
        self._ensure_data_fetched()
        return self.df_compressed, self._effective_cluster_col

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
        
        eff_cluster = self._effective_cluster_col
        if eff_cluster:
            cluster_expr = f"CAST({eff_cluster} AS SMALLINT)" if eff_cluster in boolean_cols else eff_cluster
            select_parts.append(f"{cluster_expr} AS {eff_cluster}")
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
            ([self._effective_cluster_col] if self._effective_cluster_col else []) +
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
        """Collect data for pooled OLS (intercept + covariates)"""
        # Use explicit column names to avoid matching _sq columns with regex
        y_cols = [f"sum_{var.sql_name}" for var in self.formula.outcomes]
        y = data[y_cols].values
        n = data["count"].values
        
        # Convert sum to mean for WLS (WLS will multiply by sqrt(n) internally)
        y = y / n.reshape(-1, 1)
        
        # Use sql_names for covariates
        covariate_sql_names = self.formula.get_covariate_sql_names()
        X = data[covariate_sql_names].values

        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X

        # Pooled OLS: Add intercept column
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Set coefficient names (intercept + covariates)
        self.coef_names_ = ['Intercept'] + self.formula.get_covariate_display_names()

        return y, X, n