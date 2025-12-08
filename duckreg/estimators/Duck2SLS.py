import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, List, Dict, Any

from ..formula_parser import cast_if_boolean, quote_identifier
from ..fitters import NumpyFitter, DuckDBFitter
from .DuckLinearModel import DuckLinearModel, RegressionResults, FirstStageResults
from .DuckMundlak import DuckMundlak

logger = logging.getLogger(__name__)


# ============================================================================
# Duck2SLS Helper Classes
# ============================================================================

class StageFormulaBuilder:
    """Builds Formula objects for 2SLS stages from the original IV formula"""
    
    def __init__(self, original_formula: 'Formula'):
        self.original = original_formula
        self._simple_fes = self._convert_merged_fes_to_simple()
        # Check if original formula has an explicit intercept
        self._has_explicit_intercept = any(var.is_intercept() for var in self.original.covariates)
    
    def _convert_merged_fes_to_simple(self) -> Tuple:
        """Convert merged FEs to simple Variables (since they're pre-computed in stage tables)"""
        from ..formula_parser import Variable, VariableRole
        
        simple_fes = list(self.original.fixed_effects)
        for mfe in self.original.merged_fes:
            simple_fes.append(Variable(name=mfe.name, role=VariableRole.FIXED_EFFECT))
        return tuple(simple_fes)
    
    def build_first_stage(self, endog_var: str, endogenous_names: List[str]) -> 'Formula':
        """Build first-stage formula: endog ~ exogenous + instruments | FE | 0 | cluster
        
        Note: The first_stage_data table already has transformed values stored under
        the display_name. So we create Variables with transform=NONE but use the
        display_name as the column name.
        """
        from ..formula_parser import Formula, Variable, VariableRole, TransformType
        
        covariates = []
        
        # Add intercept first if present in original formula
        if self._has_explicit_intercept:
            covariates.append(Variable(
                name='_intercept',
                role=VariableRole.COVARIATE,
                display_name='_intercept'
            ))
        
        # Add exogenous covariates (excluding intercept since we handled it above)
        for var in self.original.covariates:
            if var.display_name not in endogenous_names and not var.is_intercept():
                covariates.append(var)
        
        # Add instruments as covariates
        for var in self.original.instruments:
            covariates.append(Variable(
                name=var.display_name,
                role=VariableRole.COVARIATE,
                transform=TransformType.NONE,
                transform_shift=0.0,
                lag=None,
                display_name=var.display_name
            ))
        
        # Find the endogenous variable to use as outcome
        endog_var_obj = next(
            (var for var in self.original.endogenous if var.display_name == endog_var),
            None
        )
        
        if endog_var_obj:
            outcome_var = Variable(
                name=endog_var_obj.display_name,
                role=VariableRole.OUTCOME,
                transform=TransformType.NONE,
                transform_shift=0.0,
                lag=None,
                display_name=endog_var_obj.display_name
            )
        else:
            outcome_var = Variable(name=endog_var, role=VariableRole.OUTCOME)
        
        return Formula(
            outcomes=(outcome_var,),
            covariates=tuple(covariates),
            interactions=(),
            fixed_effects=self._simple_fes,
            merged_fes=(),
            cluster=self.original.cluster,
            raw_formula=f"{endog_var} ~ first_stage",
            endogenous=(),
            instruments=(),
        )
    
    def build_second_stage(self, endogenous_names: List[str]) -> 'Formula':
        """Build second-stage formula: y ~ exogenous + fitted_endogenous | FE | 0 | cluster"""
        from ..formula_parser import Formula, Variable, VariableRole
        
        covariates = []
        
        # Add intercept first if present in original formula
        if self._has_explicit_intercept:
            covariates.append(Variable(
                name='_intercept',
                role=VariableRole.COVARIATE,
                display_name='_intercept'
            ))
        
        # Add exogenous covariates (excluding intercept and endogenous)
        for var in self.original.covariates:
            if var.display_name not in endogenous_names and not var.is_intercept():
                covariates.append(var)
        
        # Add fitted endogenous variables
        for endog in endogenous_names:
            covariates.append(Variable(
                name=f"fitted_{endog}",
                role=VariableRole.COVARIATE,
                display_name=f"fitted_{endog}"
            ))
        
        return Formula(
            outcomes=self.original.outcomes,
            covariates=tuple(covariates),
            interactions=(),
            fixed_effects=self._simple_fes,
            merged_fes=(),
            cluster=self.original.cluster,
            raw_formula="second_stage",
            endogenous=(),
            instruments=(),
        )


# ============================================================================
# Duck2SLS (Two-Stage Least Squares)
# ============================================================================

class Duck2SLS(DuckLinearModel):
    """Two-Stage Least Squares (2SLS) estimator
    
    Standard errors are computed using the correct 2SLS formula where residuals
    are calculated using actual endogenous values (not fitted values):
    
    e = y - X_exog * β_exog - X_endog_actual * β_endog
    
    The variance-covariance matrix uses the sandwich formula:
    V(β) = (Z'WZ)^(-1) * Z'W * Ω * WZ * (Z'WZ)^(-1)
    
    where Z contains fitted endogenous values (for the bread) and Ω is computed
    using residuals from actual endogenous values (for the meat).
    """
    
    _FIRST_STAGE_TABLE = "first_stage_data"
    _FITTED_COL_PREFIX = "fitted_"
    _ACTUAL_COL_PREFIX = "actual_"
    _RESIDUAL_COL = "_2sls_residual"
    _CORRECTED_VIEW = "_2sls_corrected_view"
    
    def __init__(self, fe_method: str = "demean", **kwargs):
        super().__init__(**kwargs)
        
        self.fe_method = fe_method
        self._formula_builder = StageFormulaBuilder(self.formula)
        
        # Extract IV-specific info from formula
        # Use display names for user-facing info and coefficient matching
        self.endogenous_vars = self.formula.get_endogenous_display_names()
        self.instrument_vars = self.formula.get_instrument_display_names()
        self.exogenous_vars = [var.display_name for var in self.formula.covariates 
                               if not var.is_intercept() and var.name not in self.formula.get_endogenous_names()]
        
        # Map from display name to Variable object for SQL generation
        self._endogenous_vars_map = {var.display_name: var for var in self.formula.endogenous}
        self._instrument_vars_map = {var.display_name: var for var in self.formula.instruments}
        
        # Storage for stage results
        self._first_stage_results: Dict[str, FirstStageResults] = {}
        self._second_stage_estimator: Optional[DuckLinearModel] = None
        
        logger.debug(f"Duck2SLS: endogenous={self.endogenous_vars}, "
                    f"instruments={self.instrument_vars}, "
                    f"exogenous={self.exogenous_vars}, fe_method={fe_method}")

    # -------------------------------------------------------------------------
    # First stage results property
    # -------------------------------------------------------------------------
    
    @property
    def first_stage(self) -> Dict[str, FirstStageResults]:
        """Get first stage results for all endogenous variables"""
        return self._first_stage_results
    
    def get_first_stage_f_stats(self) -> Dict[str, Optional[float]]:
        """Get F-statistics for all first stages"""
        return {endog: fs.f_statistic for endog, fs in self._first_stage_results.items()}
    
    def has_weak_instruments(self) -> bool:
        """Check if any first stage has weak instruments (F < 10)"""
        return any(fs.is_weak_instrument for fs in self._first_stage_results.values() if fs.is_weak_instrument is not None)

    # -------------------------------------------------------------------------
    # Abstract method implementations
    # -------------------------------------------------------------------------

    def _get_n_coefs(self) -> int:
        n_covs = len(self.exogenous_vars) + len(self.endogenous_vars)
        if not self.fe_cols:
            n_covs += 1
        return n_covs * len(self.outcome_vars)

    def _get_cluster_data_for_bootstrap(self) -> Tuple[pd.DataFrame, Optional[str]]:
        self._ensure_data_fetched()
        return self.df_compressed, self.cluster_col

    # -------------------------------------------------------------------------
    # Data preparation
    # -------------------------------------------------------------------------

    def prepare_data(self):
        """Prepare base data table and run first stage regressions"""
        self._create_first_stage_table()
        self._run_all_first_stages()

    def _create_first_stage_table(self):
        """Create table with all columns needed for both stages"""
        boolean_cols = self._get_boolean_columns()
        unit_col = self._get_unit_col()
        
        select_parts = self._build_first_stage_select(boolean_cols, unit_col)
        
        self.conn.execute(f"""
        CREATE OR REPLACE TABLE {self._FIRST_STAGE_TABLE} AS
        SELECT {', '.join(p for p in select_parts if p)}
        FROM {self.table_name}
        {self._build_where_clause(self.subset)}
        """)

    def _build_first_stage_select(self, boolean_cols: set, unit_col: Optional[str]) -> List[str]:
        """Build SELECT parts for first stage table.
        
        Important: For endogenous and instrument variables with transformations,
        we apply the transformation here and store the result under the DISPLAY name.
        This way, subsequent stages don't need to re-apply the transformation.
        """
        select_parts = []
        
        # Outcomes - use their select_sql which uses sql_name
        outcomes_sql = self.formula.get_outcomes_select_sql(unit_col, 'year', boolean_cols)
        if outcomes_sql:
            select_parts.append(outcomes_sql)
        
        # Exogenous covariates - use display names for user-facing columns
        endogenous_display_names = set(self.endogenous_vars)
        for var in self.formula.covariates:
            if var.display_name not in endogenous_display_names:
                if var.is_intercept():
                    # Add intercept constant
                    select_parts.append("1 AS _intercept")
                else:
                    # Store under display name for clarity
                    expr = var.get_sql_expression(unit_col, 'year')
                    expr = cast_if_boolean(expr, var.name, boolean_cols)
                    alias = quote_identifier(var.display_name)
                    select_parts.append(f"{expr} AS {alias}")
        
        # Endogenous variables - apply transformation and store under display name
        for var in self.formula.endogenous:
            expr = var.get_sql_expression(unit_col, 'year')
            expr = cast_if_boolean(expr, var.name, boolean_cols)
            alias = quote_identifier(var.display_name)
            select_parts.append(f"{expr} AS {alias}")
        
        # Instruments - apply transformation and store under display name
        for var in self.formula.instruments:
            expr = var.get_sql_expression(unit_col, 'year')
            expr = cast_if_boolean(expr, var.name, boolean_cols)
            alias = quote_identifier(var.display_name)
            select_parts.append(f"{expr} AS {alias}")
        
        # Fixed effects - use sql_names
        fe_sql = self.formula.get_fe_select_sql(boolean_cols)
        if fe_sql:
            select_parts.append(fe_sql)
        
        # Cluster - use raw name
        if self.cluster_col:
            cluster_quoted = quote_identifier(self.cluster_col)
            cluster_expr = (f"CAST({cluster_quoted} AS SMALLINT)" 
                          if self.cluster_col in boolean_cols else cluster_quoted)
            select_parts.append(f"{cluster_expr} AS {cluster_quoted}")
        
        return select_parts

    # -------------------------------------------------------------------------
    # First stage
    # -------------------------------------------------------------------------

    def _run_all_first_stages(self):
        """Run first-stage regressions for all endogenous variables"""
        for endog_var in self.endogenous_vars:
            logger.debug(f"Running first stage for {endog_var}")
            fs_result = self._run_single_first_stage(endog_var)
            self._first_stage_results[endog_var] = fs_result
            self._add_fitted_column(fs_result)
            logger.debug(f"First stage for {endog_var}: F-stat={fs_result.f_statistic}")

    def _run_single_first_stage(self, endog_var: str) -> FirstStageResults:
        """Run first-stage regression for one endogenous variable"""
        formula = self._formula_builder.build_first_stage(endog_var, self.endogenous_vars)
        estimator = self._create_stage_estimator(formula, n_bootstraps=0)
        
        estimator.prepare_data()
        estimator.compress_data()
        coefs = estimator.estimate()
        estimator.fit_vcov()  # Compute vcov for F-stat
        
        reg_results = RegressionResults(
            coefficients=coefs,
            coef_names=estimator.coef_names_,
            vcov=getattr(estimator, 'vcov', None),
            n_obs=getattr(estimator, 'n_obs', None),
            n_compressed=getattr(estimator, 'n_compressed_rows', None),
            se_type=getattr(estimator, 'se', None),
        )
        
        return FirstStageResults(
            endog_var=endog_var,
            results=reg_results,
            instrument_names=self.instrument_vars,
        )

    def _add_fitted_column(self, fs_result: FirstStageResults):
        """Add fitted values column to first stage table"""
        fitted_expr = self._build_fitted_expression(fs_result)
        # Use display name for the fitted column name
        fitted_col = f"{self._FITTED_COL_PREFIX}{fs_result.endog_var}"
        fitted_col_quoted = quote_identifier(fitted_col)
        
        self.conn.execute(f"""
        ALTER TABLE {self._FIRST_STAGE_TABLE} ADD COLUMN {fitted_col_quoted} DOUBLE
        """)
        self.conn.execute(f"""
        UPDATE {self._FIRST_STAGE_TABLE} SET {fitted_col_quoted} = {fitted_expr}
        """)
        
        logger.debug(f"Added fitted values column: {fitted_col}")

    def _build_fitted_expression(self, fs_result: FirstStageResults) -> str:
        """Build SQL expression for fitted values from first stage coefficients"""
        available_cols = self._get_table_columns(self._FIRST_STAGE_TABLE)
        
        expr_parts = []
        for name, coef in zip(fs_result.coef_names, fs_result.coefficients.flatten()):
            if name == 'Intercept':
                expr_parts.append(f"{coef}")
            elif name in available_cols:
                name_quoted = quote_identifier(name)
                expr_parts.append(f"({coef} * {name_quoted})")
            # Skip Mundlak average columns not in table
        
        return " + ".join(expr_parts) if expr_parts else "0"

    def _get_table_columns(self, table_name: str) -> set:
        """Get column names from a table"""
        return set(
            self.conn.execute(f"SELECT column_name FROM (DESCRIBE {table_name})")
            .fetchdf()['column_name'].tolist()
        )

    # -------------------------------------------------------------------------
    # Second stage
    # -------------------------------------------------------------------------

    def compress_data(self):
        """Create and prepare second stage estimator"""
        formula = self._formula_builder.build_second_stage(self.endogenous_vars)
        self._second_stage_estimator = self._create_stage_estimator(
            formula, n_bootstraps=self.n_bootstraps
        )
        
        self._second_stage_estimator.prepare_data()
        self._second_stage_estimator.compress_data()
        
        # Copy compressed data info
        self.agg_query = self._second_stage_estimator.agg_query
        self.n_obs = self._second_stage_estimator.n_obs
        self.n_compressed_rows = self._second_stage_estimator.n_compressed_rows

    def estimate(self) -> np.ndarray:
        """Run second stage estimation"""
        coefs = self._second_stage_estimator.estimate()
        
        self.coef_names_ = self._rename_fitted_coefs(self._second_stage_estimator.coef_names_)
        self._fitter_result = self._second_stage_estimator._fitter_result
        
        return coefs

    def _rename_fitted_coefs(self, coef_names: List[str]) -> List[str]:
        """Rename fitted_X coefficients back to X for cleaner output"""
        prefix = self._FITTED_COL_PREFIX
        prefix_len = len(prefix)
        
        renamed = []
        for name in coef_names:
            if name.startswith(prefix):
                # Check if the suffix matches any endogenous display name
                suffix = name[prefix_len:]
                if suffix in self.endogenous_vars:
                    renamed.append(suffix)
                else:
                    renamed.append(name)
            else:
                renamed.append(name)
        return renamed

    # -------------------------------------------------------------------------
    # Estimator factory
    # -------------------------------------------------------------------------

    def _create_stage_estimator(self, formula: 'Formula', n_bootstraps: int) -> DuckLinearModel:
        """Factory method to create estimator for a stage"""
        from .DuckRegression import DuckRegression
        
        EstimatorClass = DuckMundlak if (self.fe_cols and self.fe_method == "mundlak") else DuckRegression
        
        estimator = EstimatorClass(
            db_name=self.db_name,
            table_name=self._FIRST_STAGE_TABLE,
            formula=formula,
            seed=self.seed,
            n_bootstraps=n_bootstraps,
            round_strata=self.round_strata,
            duckdb_kwargs=self.duckdb_kwargs,
            subset=None,
            n_jobs=self.n_jobs if n_bootstraps > 0 else 1,
            fitter=self.fitter,
            keep_connection_open=True,
        )
        
        # Share connection and RNG
        estimator.conn = self.conn
        estimator.rng = self.rng
        
        return estimator

    # -------------------------------------------------------------------------
    # Delegation and vcov
    # -------------------------------------------------------------------------

    def _ensure_data_fetched(self):
        self._second_stage_estimator._ensure_data_fetched()
        self.df_compressed = self._second_stage_estimator.df_compressed
        self._data_fetched = True

    def collect_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._second_stage_estimator.collect_data(data)

    def fit_vcov(self):
        """Compute correct 2SLS variance-covariance matrix.
        
        For 2SLS, we need to compute residuals using actual endogenous values,
        not the fitted values from first stage. The formula is:
        
        V(β) = (Z'WZ)^(-1) * Z'W * Ω * WZ * (Z'WZ)^(-1)
        
        where Z is the matrix of instruments (exogenous + fitted endogenous),
        and Ω is computed using residuals from actual endogenous:
        e = y - X_actual * β_2sls
        """
        if self.fitter == "duckdb":
            self._compute_corrected_2sls_vcov_duckdb()
        else:
            self._compute_corrected_2sls_vcov_numpy()

    def _compute_corrected_2sls_vcov_numpy(self):
        """Compute corrected 2SLS vcov using NumpyFitter (in-memory)."""
        self._ensure_data_fetched()
        
        # Get the coefficient estimates
        coefs = self.point_estimate.flatten()
        
        # Get compressed data
        df = self.df_compressed.copy()
        
        # Add actual endogenous values to compressed data
        self._add_actual_endogenous_to_compressed()
        df = self.df_compressed  # Get updated dataframe
        
        # Get design matrix with fitted endogenous (the "Z" matrix for bread)
        y, X_fitted, weights = self._second_stage_estimator.collect_data(df)
        
        # Build X with actual endogenous to compute correct residuals
        X_actual = self._build_x_with_actual_endogenous(df)
        
        # Compute correct residuals: e = y - X_actual * β
        residuals = y.flatten() - X_actual @ coefs
        
        # Get cluster IDs if clustering
        cluster_col = self._second_stage_estimator._get_cluster_col_for_vcov()
        cluster_ids = None
        if cluster_col and cluster_col in df.columns:
            cluster_ids = df[cluster_col].values
        
        # Use NumpyFitter with custom residuals for vcov computation
        fitter = NumpyFitter(alpha=1e-8, se_type="stata")
        result = fitter.fit(
            X=X_fitted,
            y=y,
            weights=weights,
            coef_names=self.coef_names_,
            cluster_ids=cluster_ids,
            compute_vcov=True,
            residuals=residuals,
            coefficients=coefs
        )
        
        self.vcov = result.vcov
        self.se = result.se_type
        self._results = None

    def _add_actual_endogenous_to_compressed(self):
        """Add actual endogenous variable values to compressed dataframe.
        
        For 2SLS vcov correction, we need actual endogenous values (not fitted)
        to compute correct residuals. This aggregates actual values by the same
        strata used for compression.
        """
        if self.df_compressed is None:
            return
        
        # Get strata columns from second stage (excluding count and sum columns)
        strata_cols = [c for c in self._second_stage_estimator.strata_cols 
                      if c in self.df_compressed.columns and c != 'count']
        
        if not strata_cols:
            # No strata - just compute means directly
            for endog_var in self.endogenous_vars:
                actual_col = f"{self._ACTUAL_COL_PREFIX}{endog_var}"
                if actual_col not in self.df_compressed.columns:
                    # For single row or no strata, use mean from first stage table
                    endog_quoted = quote_identifier(endog_var)
                    mean_val = self.conn.execute(
                        f"SELECT AVG({endog_quoted}) FROM {self._FIRST_STAGE_TABLE}"
                    ).fetchone()[0]
                    self.df_compressed[actual_col] = mean_val
            return
        
        # Build aggregation query for actual endogenous values
        strata_quoted = [quote_identifier(c) for c in strata_cols]
        strata_sql = ", ".join(strata_quoted)
        
        agg_parts = [strata_sql]
        for endog_var in self.endogenous_vars:
            endog_quoted = quote_identifier(endog_var)
            actual_col = f"{self._ACTUAL_COL_PREFIX}{endog_var}"
            agg_parts.append(f"AVG({endog_quoted}) AS {quote_identifier(actual_col)}")
        
        query = f"""
        SELECT {', '.join(agg_parts)}
        FROM {self._FIRST_STAGE_TABLE}
        GROUP BY {strata_sql}
        """
        
        actual_df = self.conn.execute(query).fetchdf()
        
        # Merge with compressed data
        merge_cols = [c for c in strata_cols if c in self.df_compressed.columns and c in actual_df.columns]
        if merge_cols:
            # Drop any existing actual columns to avoid conflicts
            for endog_var in self.endogenous_vars:
                actual_col = f"{self._ACTUAL_COL_PREFIX}{endog_var}"
                if actual_col in self.df_compressed.columns:
                    self.df_compressed = self.df_compressed.drop(columns=[actual_col])
            
            self.df_compressed = self.df_compressed.merge(
                actual_df, on=merge_cols, how='left'
            )

    def _build_x_with_actual_endogenous(self, df: pd.DataFrame) -> np.ndarray:
        """Build design matrix using actual endogenous values instead of fitted.
        
        This is needed for computing correct 2SLS residuals for vcov.
        The matrix structure must match the fitted X exactly (same columns in same order).
        """
        second_stage = self._second_stage_estimator
        
        # Get RHS column names (what collect_data uses)
        if hasattr(second_stage, 'rhs') and second_stage.rhs is not None:
            rhs_cols = second_stage.rhs.copy()
        elif hasattr(second_stage, 'coef_names_') and second_stage.coef_names_ is not None:
            # Reconstruct from coef_names, excluding Intercept
            rhs_cols = [name for name in second_stage.coef_names_ if name != 'Intercept']
        else:
            raise ValueError("Cannot determine RHS columns for 2SLS vcov")
        
        # Build column list, replacing fitted with actual
        actual_cols = []
        for col in rhs_cols:
            if col.startswith(self._FITTED_COL_PREFIX):
                endog_name = col[len(self._FITTED_COL_PREFIX):]
                actual_col = f"{self._ACTUAL_COL_PREFIX}{endog_name}"
                if actual_col in df.columns:
                    actual_cols.append(actual_col)
                elif endog_name in df.columns:
                    # Fallback to raw endogenous name
                    actual_cols.append(endog_name)
                else:
                    # Last resort: use fitted (will give wrong SEs but won't crash)
                    logger.warning(f"Could not find actual values for {endog_name}, using fitted")
                    actual_cols.append(col)
            else:
                actual_cols.append(col)
        
        # Build X matrix with same structure as collect_data
        has_intercept = 'Intercept' in (second_stage.coef_names_ or [])
        
        X_data = df[actual_cols].values
        if has_intercept:
            X = np.c_[np.ones(len(df)), X_data]
        else:
            X = X_data
        
        return X

    def _compute_corrected_2sls_vcov_duckdb(self):
        """Compute corrected 2SLS vcov using DuckDBFitter (out-of-core)."""
        # Get the coefficient estimates
        coefs = self.point_estimate.flatten()
        
        # Add residual column to the compressed view using actual endogenous
        self._add_residual_column_to_view()
        
        # Get x_cols and other info from second stage estimator
        x_cols = self._second_stage_estimator._get_x_cols_for_duckdb()
        y_col = self._second_stage_estimator._get_y_col_for_duckdb()
        cluster_col = self._second_stage_estimator._get_cluster_col_for_vcov()
        view_cols = self._get_corrected_view_columns()
        
        # Use DuckDBFitter with pre-computed residuals
        fitter = DuckDBFitter(conn=self.conn, alpha=1e-8, se_type="stata")
        result = fitter.fit(
            table_name=self._CORRECTED_VIEW,
            x_cols=x_cols,
            y_col=y_col,
            weight_col="count",
            add_intercept=self._second_stage_estimator._needs_intercept_for_duckdb(),
            cluster_col=cluster_col if cluster_col in view_cols else None,
            compute_vcov=True,
            coefficients=coefs,
            residual_col=self._RESIDUAL_COL
        )
        
        self.vcov = result.vcov
        self.se = result.se_type
        self.coef_names_ = self._rename_fitted_coefs(result.coef_names)
        self._results = None

    def _get_corrected_view_columns(self) -> List[str]:
        """Get column names from corrected view"""
        return self.conn.execute(
            f"SELECT column_name FROM (DESCRIBE SELECT * FROM {self._CORRECTED_VIEW})"
        ).fetchdf()['column_name'].tolist()

    def _add_residual_column_to_view(self):
        """Create a view with the corrected residual column for 2SLS vcov.
        
        The residual is: e = mean_y - X_actual * β
        where X_actual uses actual endogenous values, not fitted.
        """
        coefs = self.point_estimate.flatten()
        second_stage_est = self._second_stage_estimator
        
        # Get the compressed view name
        base_view = second_stage_est._COMPRESSED_VIEW
        
        # Get coefficient names
        coef_names = second_stage_est.coef_names_ if hasattr(second_stage_est, 'coef_names_') else []
        
        # For Mundlak: we need actual endogenous means per FE group
        if isinstance(second_stage_est, DuckMundlak):
            self._create_corrected_view_mundlak(base_view, coefs, coef_names)
        else:
            # For DuckRegression (demeaning) - simpler case
            self._create_corrected_view_demean(base_view, coefs, coef_names)

    def _create_corrected_view_demean(self, base_view: str, coefs: np.ndarray, 
                                       coef_names: List[str]):
        """Create corrected view for demeaning case (simpler)."""
        y_col = self._second_stage_estimator._get_y_col_for_duckdb()
        
        # Build linear prediction, replacing fitted with actual endogenous
        pred_terms = []
        for i, (name, coef) in enumerate(zip(coef_names, coefs)):
            if name == 'Intercept':
                pred_terms.append(f"({coef})")
            else:
                # Check if this is a fitted endogenous variable (uses display name)
                col_name = name
                if name.startswith(self._FITTED_COL_PREFIX):
                    endog_display_name = name[len(self._FITTED_COL_PREFIX):]
                    if endog_display_name in self.endogenous_vars:
                        # Use actual endogenous value (stored under display name)
                        col_name = endog_display_name
                
                col_quoted = quote_identifier(col_name)
                pred_terms.append(f"({coef} * {col_quoted})")
        
        linear_pred = " + ".join(pred_terms) if pred_terms else "0"
        
        # residual = mean_y - linear_prediction
        # mean_y = sum_y / count
        residual_expr = f"(({y_col} / count) - ({linear_pred}))"
        
        self.conn.execute(f"""
        CREATE OR REPLACE VIEW {self._CORRECTED_VIEW} AS
        SELECT base.*, {residual_expr} AS {self._RESIDUAL_COL}
        FROM {base_view} base
        """)

    def _create_corrected_view_mundlak(self, base_view: str, coefs: np.ndarray,
                                        coef_names: List[str]):
        """Create corrected view for Mundlak case.
        
        For Mundlak, we need to join actual endogenous means per cluster/FE group.
        """
        y_col = self._second_stage_estimator._get_y_col_for_duckdb()
        cluster_col = self._second_stage_estimator._CLUSTER_ALIAS
        fs_cluster = self.cluster_col or (self.fe_cols[0] if self.fe_cols else None)
        
        if not fs_cluster:
            # No cluster/FE - fall back to simple case
            self._create_corrected_view_demean(base_view, coefs, coef_names)
            return
        
        # Build subqueries for actual endogenous means per cluster
        # Use display names since that's how they're stored in first_stage_table
        endog_subqueries = []
        endog_col_refs = {}
        
        for idx, endog_display_name in enumerate(self.endogenous_vars):
            endog_quoted = quote_identifier(endog_display_name)
            fs_cluster_quoted = quote_identifier(fs_cluster)
            actual_col = f"_actual_{idx}"
            table_alias = f"_endog_{idx}"
            
            subquery = f"""
            (SELECT {fs_cluster_quoted} AS _join_key_{idx}, 
                    AVG({endog_quoted}) AS {actual_col}
             FROM {self._FIRST_STAGE_TABLE}
             GROUP BY {fs_cluster_quoted}) {table_alias}
            """
            endog_subqueries.append((table_alias, f"_join_key_{idx}", subquery))
            endog_col_refs[endog_display_name] = f"{table_alias}.{actual_col}"
        
        # Build linear prediction with actual endogenous
        pred_terms = []
        for i, (name, coef) in enumerate(zip(coef_names, coefs)):
            if name == 'Intercept':
                pred_terms.append(f"({coef})")
            else:
                # Check if this is a fitted endogenous variable
                if name.startswith(self._FITTED_COL_PREFIX):
                    endog_display_name = name[len(self._FITTED_COL_PREFIX):]
                    if endog_display_name in endog_col_refs:
                        # Use actual endogenous from joined table
                        col_ref = endog_col_refs[endog_display_name]
                        pred_terms.append(f"({coef} * COALESCE({col_ref}, base.{quote_identifier(name)}))")
                        continue
                
                # Regular covariate
                col_quoted = quote_identifier(name)
                pred_terms.append(f"({coef} * base.{col_quoted})")
        
        linear_pred = " + ".join(pred_terms) if pred_terms else "0"
        residual_expr = f"(({y_col} / count) - ({linear_pred}))"
        
        # Build JOIN clauses
        join_clauses = []
        for table_alias, join_key, subquery in endog_subqueries:
            join_clauses.append(f"LEFT JOIN {subquery} ON base.{cluster_col} = {table_alias}.{join_key}")
        
        joins_sql = "\n".join(join_clauses)
        
        query = f"""
        CREATE OR REPLACE VIEW {self._CORRECTED_VIEW} AS
        SELECT base.*, {residual_expr} AS {self._RESIDUAL_COL}
        FROM {base_view} base
        {joins_sql}
        """
        self.conn.execute(query)

    # -------------------------------------------------------------------------
    # Bootstrap and summary
    # -------------------------------------------------------------------------

    def bootstrap(self) -> np.ndarray:
        """Run bootstrap for 2SLS - bootstrap both stages together."""
        # For proper 2SLS bootstrap, we should re-estimate both stages
        # For now, delegate to second stage bootstrap (which is approximate)
        self._second_stage_estimator.n_bootstraps = self.n_bootstraps
        vcov = self._second_stage_estimator.bootstrap()
        self.se = "bootstrap"
        self.vcov = vcov
        self._results = None
        return vcov

    def summary(self) -> Dict[str, Any]:
        """Comprehensive 2SLS results summary"""
        result = {
            "coefficients": self.results.to_dict() if self.results else None,
            "n_obs": getattr(self, 'n_obs', None),
            "n_compressed": self.n_compressed_rows,
            "estimator_type": "Duck2SLS",
            "fe_method": self.fe_method if self.fe_cols else None,
            "outcome_vars": self.outcome_vars,
            "exogenous_vars": self.exogenous_vars,
            "endogenous_vars": self.endogenous_vars,
            "instrument_vars": self.instrument_vars,
            "fe_cols": self.fe_cols,
            "cluster_col": self.cluster_col,
            "first_stage": {endog: fs.to_dict() for endog, fs in self._first_stage_results.items()},
            "weak_instruments": self.has_weak_instruments(),
        }
        return result