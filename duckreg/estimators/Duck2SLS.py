"""
Two-Stage Least Squares (2SLS) / Instrumental Variables Estimator

This module provides a self-contained 2SLS estimator that computes correct 
standard errors using residuals from actual endogenous variables.

Key Design Principles:
1. Self-contained: Does not delegate to sub-estimators to avoid state conflicts
2. Direct computation: Builds design matrices directly in Python/numpy
3. Correct SEs: Uses actual endogenous values for residual computation
4. Mundlak device: Handles FEs by including group means as regressors
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, List, Dict, Any

from ..formula_parser import cast_if_boolean, quote_identifier
from ..fitters import wls, NumpyFitter
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
    
    Estimation Pipeline:
    1. Create data table with all variables
    2. Run first stage for each endogenous variable
    3. Build design matrices (fitted and actual)
    4. Estimate coefficients using fitted endogenous
    5. Compute vcov using actual endogenous for residuals
    """
    
    _DATA_TABLE = "iv_data"
    
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
    
    def compress_data(self):
        """Run first stages and build design matrices"""
        self._run_first_stages()
        self._build_design_matrices()
    
    def estimate(self) -> np.ndarray:
        """Estimate coefficients using fitted endogenous values"""
        if self._X_fitted is None:
            raise ValueError("Must call compress_data() before estimate()")
        return wls(self._X_fitted, self._y, self._weights)
    
    def fit_vcov(self):
        """Compute vcov with correct 2SLS formula using actual residuals"""
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

    # =========================================================================
    # First Stage
    # =========================================================================
    
    def _run_first_stages(self):
        """Run first-stage regressions for all endogenous variables"""
        for endog_var in self.endogenous_vars:
            logger.debug(f"Running first stage for {endog_var}")
            fs_result = self._run_single_first_stage(endog_var)
            self._first_stage_results[endog_var] = fs_result
            self._add_fitted_column(fs_result)
            logger.debug(f"First stage for {endog_var}: F-stat={fs_result.f_statistic}")
    
    def _run_single_first_stage(self, endog_var: str) -> FirstStageResults:
        """Run first-stage regression for one endogenous variable"""
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
    
    def _fetch_first_stage_data(self) -> pd.DataFrame:
        """Fetch data for first stage"""
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
        
        # Build fitted expression (only for non-Mundlak terms - those are computed in Python)
        expr_parts = []
        for name, coef in zip(fs_result.coef_names, fs_result.coefficients.flatten()):
            if name == 'Intercept':
                expr_parts.append(f"{coef}")
            elif name.startswith('avg_') and '_fe' in name:
                # Skip Mundlak means - computed in Python
                continue
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
        X_common, coef_names = self._build_common_X(df)
        
        # Build endogenous columns
        X_endog_fitted, X_endog_actual, endog_names = self._build_endogenous_X(df)
        
        self._X_fitted = np.hstack([X_common, X_endog_fitted])
        self._X_actual = np.hstack([X_common, X_endog_actual])
        
        self.coef_names_ = coef_names + endog_names
        self._weights = np.ones(len(df))
        
        if self.cluster_col and self.cluster_col in df.columns:
            self._cluster_ids = df[self.cluster_col].values
        else:
            self._cluster_ids = None
    
    def _fetch_second_stage_data(self) -> pd.DataFrame:
        """Fetch data for second stage"""
        return self.conn.execute(f"SELECT * FROM {self._DATA_TABLE}").fetchdf().dropna()
    
    def _build_common_X(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Build exogenous part of design matrix"""
        X_parts = []
        coef_names = []
        
        # Intercept
        if self._has_intercept or not self.fe_cols:
            X_parts.append(np.ones((len(df), 1)))
            coef_names.append('Intercept')
        
        # Exogenous covariates
        exog_cols = []
        exog_names = []
        for var in self.formula.covariates:
            if not var.is_intercept() and var.display_name not in self.endogenous_vars:
                if var.sql_name in df.columns:
                    exog_cols.append(var.sql_name)
                    exog_names.append(var.display_name)
        
        if exog_cols:
            X_parts.append(df[exog_cols].values)
            coef_names.extend(exog_names)
        
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
        
        if X_parts:
            return np.hstack(X_parts), coef_names
        return np.ones((len(df), 1)), ['Intercept']
    
    def _build_endogenous_X(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Build endogenous part of design matrices (fitted and actual)"""
        X_fitted_parts = []
        X_actual_parts = []
        names = []
        
        for var in self.formula.endogenous:
            fitted_col = f"fitted_{var.sql_name}"
            
            if fitted_col in df.columns:
                X_fitted_parts.append(df[fitted_col].values.reshape(-1, 1))
            else:
                logger.warning(f"Fitted column {fitted_col} not found, using actual")
                X_fitted_parts.append(df[var.sql_name].values.reshape(-1, 1))
            
            X_actual_parts.append(df[var.sql_name].values.reshape(-1, 1))
            names.append(var.display_name)
        
        X_fitted = np.hstack(X_fitted_parts) if X_fitted_parts else np.empty((len(df), 0))
        X_actual = np.hstack(X_actual_parts) if X_actual_parts else np.empty((len(df), 0))
        
        return X_fitted, X_actual, names

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
        """Comprehensive 2SLS results summary"""
        return {
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
            "first_stage": {
                endog: fs.to_dict() for endog, fs in self._first_stage_results.items()
            },
            "weak_instruments": self.has_weak_instruments(),
        }
    
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