"""
Two-Stage Least Squares (2SLS) / Instrumental Variables Estimator

Refactored to use the same FETransformer architecture as DuckFE.  Fixed-effect
absorption in both stages is delegated to MundlakTransformer (or, for the
special case of no fixed effects, the stage is simply a plain OLS).

Pipeline
--------
1. ``prepare_data()``:  build ``_iv_staging`` from the source table using the
   same formula-helper pattern as DuckFE, plus a ``_row_idx`` column for
   reliable row-level joins between the transformer's ``design_matrix`` and the
   staging table.

2. ``compress_data()`` (= first + second stage setup):

   a. For each endogenous variable, run ``MundlakTransformer`` on
      ``_iv_staging`` with ``covariate_cols = exog + instruments``
      (Mundlak path), **or** demean all variables once up front via
      ``IterativeDemeanTransformer`` (demean path).

   b. Run ``MundlakTransformer`` on the updated ``_iv_staging`` with
      ``covariate_cols = exog + fitted_endog + instruments`` (all three so that
      both X-extra and Z-extra Mundlak means are available).  Create a
      compressed view (DuckDB path) or fetch numpy arrays directly.

Standard Error Computation
--------------------------
Second-stage SEs use the IV-adjusted sandwich estimator::

    V = Bread @ Meat_IV @ Bread

where::

    Bread   = (X̂'X̂)⁻¹
    Meat_IV = (X̂'Z)(Z'Z)⁻¹ Ω (Z'Z)⁻¹ (Z'X̂)
    Ω       = X'diag(e²w)X  (inner meat, evaluated at *actual* X)
    X̂      = design matrix with fitted endogenous
    Z       = instrument matrix (intercept + exog + instruments + FE controls)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import DuckEstimator
from ..core.transformers import MundlakTransformer, IterativeDemeanTransformer, AutoFETransformer
from ..core.fitters import NumpyFitter, DuckDBFitter
from ..core.vcov import VcovSpec
from ..core.results import RegressionResults, FirstStageResults
from ..utils.formula_parser import cast_if_boolean, quote_identifier
from ..utils.name_utils import build_coef_name_lists, build_first_stage_coef_names

logger = logging.getLogger(__name__)


class Duck2SLS(DuckEstimator):
    """Two-Stage Least Squares estimator using transformer-based FE absorption.

    Fixed effects are absorbed via the Mundlak device: group-means of
    covariates and instruments are added as explicit regressors.  The
    transformer handles FE profiling (fixed vs. asymptotic), Wooldridge
    correction for unbalanced panels, and all column bookkeeping.

    Parameters
    ----------
    db_name, table_name, formula
        Standard DuckEstimator / formula arguments.
    method : str
        FE absorption strategy.  ``'mundlak'`` (default) adds within-group
        means of all covariates and instruments.  ``'demean'`` absorbs FEs via
        iterative demeaning (MAP) and runs both stages without an intercept.
    fe_types, cardinality_threshold, singleton_threshold, max_fixed_fe_levels
        Forwarded to :class:`MundlakTransformer` for FE classification
        (``method='mundlak'`` only).
    max_iterations, tolerance
        Forwarded to :class:`IterativeDemeanTransformer` for MAP convergence
        (``method='demean'`` only).
    fitter : {'numpy', 'duckdb'}
        Estimation backend.  Note: when ``method='demean'``, the numpy
        backend is always used regardless of this setting.
    """

    _STAGING_TABLE   = "_iv_staging"
    _COMPRESSED_VIEW = "iv_compressed"
    _CLUSTER_ALIAS   = "__cluster__"
    _DEMEANED_STAGING = "_demeaned_staging"
    
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
        method: str = "mundlak",
        # backward-compat alias
        fe_method: str = None,
        vcov_spec: Optional[VcovSpec] = None,
        # Transformer tuning (forwarded to MundlakTransformer)
        fe_types: Optional[Dict[str, str]] = None,
        cardinality_threshold: int = 50,
        singleton_threshold: float = 0.1,
        max_fixed_fe_levels: int = 100,
        remove_singletons: bool = True,
        # IterativeDemeanTransformer tuning (used when method='demean')
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        # AutoFETransformer tuning (used when method='auto_fe')
        auto_fe_kwargs: Optional[Dict[str, Any]] = None,
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
            remove_singletons=remove_singletons,
            **kwargs,
        )

        self.formula    = formula
        self.n_jobs     = n_jobs
        self.subset     = subset
        self.method     = fe_method or method   # fe_method kept for compat
        self.vcov_spec  = vcov_spec
        self.ssc_dict   = vcov_spec.ssc.to_dict() if vcov_spec is not None else None

        if self.method not in ("mundlak", "demean", "auto_fe"):
            raise ValueError(
                f"Duck2SLS: unsupported method '{self.method}'.  "
                "Valid options are 'mundlak', 'demean', and 'auto_fe'."
            )

        self.max_iterations = max_iterations
        self.tolerance      = tolerance
        self.auto_fe_kwargs: Dict[str, Any] = auto_fe_kwargs or {}

        # Transformer tuning
        self.fe_types              = fe_types or {}
        self.cardinality_threshold = cardinality_threshold
        self.singleton_threshold   = singleton_threshold
        self.max_fixed_fe_levels   = max_fixed_fe_levels

        # Formula extractions
        self.outcome_vars    = formula.get_outcome_names()
        self.fe_cols         = formula.get_fe_names()
        self.cluster_col     = formula.cluster.name if formula.cluster else None
        self.endogenous_vars = formula.get_endogenous_display_names()
        self.instrument_vars = formula.get_instrument_display_names()
        self.exogenous_vars  = [
            v.display_name for v in formula.covariates
            if not v.is_intercept() and v.display_name not in
            formula.get_endogenous_display_names()
        ]

        # Internal state
        self._first_stage_results: Dict[str, FirstStageResults] = {}
        self._results: Optional[RegressionResults] = None
        self.n_compressed_rows: Optional[int] = None
        self.n_rows_dropped_singletons: int = 0

        # Second-stage transformer and cached SQL name lists
        self._ss_transformer:   Optional[MundlakTransformer] = None
        self._ss_result_table:  Optional[str]                = None
        self._exog_sql:         Optional[List[str]]          = None
        self._fitted_endog_sql: Optional[List[str]]          = None
        self._actual_endog_sql: Optional[List[str]]          = None
        self._outcome_sql:      Optional[List[str]]          = None
        self._inst_sql:         Optional[List[str]]          = None
        self._fitter_result     = None

        # Demean-path state
        self._demean_transformer:   Optional[IterativeDemeanTransformer] = None
        self._demean_result_table:  Optional[str]                        = None
        self._df_correction:        int                                  = 0

        # Numpy-path arrays
        self._y:           Optional[np.ndarray] = None
        self._X_fitted:    Optional[np.ndarray] = None
        self._X_actual:    Optional[np.ndarray] = None
        self._Z:           Optional[np.ndarray] = None
        self._weights:     Optional[np.ndarray] = None
        self._cluster_ids: Optional[np.ndarray] = None

        logger.debug(
            f"Duck2SLS: endogenous={self.endogenous_vars}, "
            f"instruments={self.instrument_vars}, method={self.method}"
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def _effective_cluster_col(self) -> Optional[str]:
        """Cluster column resolved from formula or ``vcov_spec.cluster_vars``."""
        if self.cluster_col:
            return self.cluster_col
        if (
            self.vcov_spec is not None
            and self.vcov_spec.is_clustered
            and self.vcov_spec.cluster_vars
        ):
            return self.vcov_spec.cluster_vars[0]
        return None

    @property
    def first_stage(self) -> Dict[str, FirstStageResults]:
        return self._first_stage_results

    @property
    def results(self) -> Optional[RegressionResults]:
        if self._results is not None:
            return self._results
        if self.point_estimate is None:
            return None
        self._results = RegressionResults(
            coefficients=self.point_estimate,
            coef_names=getattr(self, "coef_names_", []),
            vcov=getattr(self, "vcov", None),
            n_obs=getattr(self, "n_obs", None),
            n_compressed=self.n_compressed_rows,
            se_type=getattr(self, "se", None),
        )
        return self._results

    def get_first_stage_f_stats(self) -> Dict[str, Optional[float]]:
        """Get F-statistics for all first stages."""
        return {e: fs.f_statistic for e, fs in self._first_stage_results.items()}

    def has_weak_instruments(self) -> bool:
        """Check if any first stage has weak instruments (F < 10)."""
        return any(
            fs.is_weak_instrument
            for fs in self._first_stage_results.values()
            if fs.is_weak_instrument is not None
        )

    # =========================================================================
    # Transformer factory  (mirrors DuckFE._make_transformer)
    # =========================================================================

    def _resolve_fe_sql_names(self) -> List[str]:
        """Resolve FE logical names to SQL-safe names via formula metadata."""
        result = []
        for fe_name in self.fe_cols:
            fe_var = self.formula.get_fe_by_name(fe_name)
            mfe    = self.formula.get_merged_fe_by_name(fe_name)
            result.append(
                fe_var.sql_name if fe_var
                else (mfe.sql_name if mfe else fe_name)
            )
        return result

    def _make_transformer(
        self,
        covariate_cols: List[str],
        source_table: str,
        cluster_col: Optional[str],
    ):
        """Instantiate the appropriate FE transformer for *source_table*.

        Returns :class:`MundlakTransformer` for ``method='mundlak'`` and
        :class:`IterativeDemeanTransformer` for ``method='demean'``.
        The *covariate_cols* argument is used only for the Mundlak variant;
        the demean variant receives only FE / cluster / convergence settings.
        """
        fe_sql = self._resolve_fe_sql_names() if self.fe_cols else []
        if self.method == "demean":
            return IterativeDemeanTransformer(
                conn=self.conn,
                table_name=source_table,
                fe_cols=fe_sql,
                cluster_col=cluster_col,
                remove_singletons=False,   # singletons already handled in prepare_data
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
            )
        if self.method == "auto_fe":
            return AutoFETransformer(
                conn=self.conn,
                table_name=source_table,
                fe_cols=fe_sql,
                cluster_col=cluster_col,
                covariate_cols=covariate_cols,
                remove_singletons=False,
                fe_types=self.fe_types,
                cardinality_threshold=self.cardinality_threshold,
                singleton_threshold=self.singleton_threshold,
                max_fixed_fe_levels=self.max_fixed_fe_levels,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
                **self.auto_fe_kwargs,
            )
        return MundlakTransformer(
            conn=self.conn,
            table_name=source_table,
            fe_cols=fe_sql,
            cluster_col=cluster_col,
            covariate_cols=covariate_cols,
            remove_singletons=False,   # singletons already handled in prepare_data
            fe_types=self.fe_types,
            cardinality_threshold=self.cardinality_threshold,
            singleton_threshold=self.singleton_threshold,
            max_fixed_fe_levels=self.max_fixed_fe_levels,
        )

    # =========================================================================
    # Estimation pipeline  (DuckEstimator interface)
    # =========================================================================

    def prepare_data(self):
        """Build the staging table using formula helpers (mirrors DuckFE.prepare_data).

        Adds a ``_row_idx`` column (ROW_NUMBER) so that the transformer's
        ``design_matrix`` can be joined back to the staging table after first-stage
        fitted values have been computed.
        """
        boolean_cols = self._get_boolean_columns()
        unit_col     = self.fe_cols[0] if self.fe_cols else None
        eff_cluster  = self._effective_cluster_col
        cluster_alias = self._CLUSTER_ALIAS if eff_cluster else None

        select_parts = []

        fe_sql = self.formula.get_fe_select_sql(boolean_cols)
        if fe_sql:
            select_parts.append(fe_sql)

        outcomes_sql = self.formula.get_outcomes_select_sql(unit_col, "year", boolean_cols)
        if outcomes_sql:
            select_parts.append(outcomes_sql)

        # Exogenous covariates (exclude intercept and endogenous)
        for var in self.formula.covariates:
            if var.is_intercept() or var.display_name in self.endogenous_vars:
                continue
            expr = var.get_sql_expression(unit_col, "year")
            expr = cast_if_boolean(expr, var.name, boolean_cols)
            select_parts.append(f"{expr} AS {var.sql_name}")

        endog_sql = self.formula.get_endogenous_select_sql(unit_col, "year", boolean_cols)
        if endog_sql:
            select_parts.append(endog_sql)

        inst_sql = self.formula.get_instruments_select_sql(unit_col, "year", boolean_cols)
        if inst_sql:
            select_parts.append(inst_sql)

        # Pass eff_cluster as fallback so clusters via se_method dict are included.
        cluster_sql = self.formula.get_cluster_select_sql(
            boolean_cols, cluster_alias, eff_cluster
        )
        if cluster_sql:
            select_parts.append(cluster_sql)

        # Row index for reliable joins between design_matrix and staging table
        select_parts.append("ROW_NUMBER() OVER () AS _row_idx")

        self.conn.execute(f"""
        CREATE OR REPLACE TABLE {self._STAGING_TABLE} AS
        SELECT {', '.join(p for p in select_parts if p)}
        FROM {self.table_name}
        {self._build_where_clause(self.subset)}
        """)

        self._remove_singleton_fe_observations()

        self.n_obs = self.conn.execute(
            f"SELECT COUNT(*) FROM {self._STAGING_TABLE}"
        ).fetchone()[0]
        self.n_compressed_rows = self.n_obs

    def compress_data(self):
        """Run first stages then prepare the second-stage working table."""
        if self.method == "demean":
            self._demean_all_variables()
            self._run_first_stages_demean()
            self._setup_second_stage_demean()
        else:
            self._run_first_stages()
            self._setup_second_stage()

    def estimate(self) -> np.ndarray:
        """Estimate second-stage coefficients."""
        if self.fitter == "duckdb":
            if self.method == "demean":
                return self._estimate_duckdb_demean()
            return self._estimate_duckdb()
        return self._estimate_numpy()

    # =========================================================================
    # First stage
    # =========================================================================

    def _run_first_stages(self):
        """Run one first-stage regression per endogenous variable."""
        for endog_var in self.endogenous_vars:
            logger.debug(f"Duck2SLS: running first stage for {endog_var}")
            fs_result = self._run_single_first_stage(endog_var)
            self._first_stage_results[endog_var] = fs_result
            logger.debug(f"  F-stat={fs_result.f_statistic}")

    def _run_single_first_stage(self, endog_var: str) -> FirstStageResults:
        """Run one first-stage OLS via MundlakTransformer."""
        endog_var_obj = next(
            v for v in self.formula.endogenous if v.display_name == endog_var
        )
        exog_sql    = self._get_exog_sql()
        inst_sql    = self._get_inst_sql()
        cluster_col = self._CLUSTER_ALIAS if self.cluster_col else None

        # Transformer: adds Mundlak means of exog + instruments
        transformer = self._make_transformer(
            covariate_cols=exog_sql + inst_sql,
            source_table=self._STAGING_TABLE,
            cluster_col=cluster_col,
        )
        # Include _row_idx so we can join results back to the staging table
        variables    = [endog_var_obj.sql_name, "_row_idx"] + exog_sql + inst_sql
        result_table = transformer.fit_transform(variables, where_clause="")

        x_sql = exog_sql + inst_sql + transformer.extra_regressors
        y_sql = endog_var_obj.sql_name

        # For MAP/hybrid paths the transformer stores columns as resid_v instead
        # of v.  Materialise a normalised table with original-name columns so
        # _ols_numpy/_ols_duckdb and _add_fitted_column can access y_sql/x_sql
        # directly.  transform_query() is identity for Mundlak, so this is a
        # no-op for that path and safe to do unconditionally.
        _FS_NORM = "_fs_first_stage_norm"
        all_vars_to_rename = variables + transformer.extra_regressors
        rename_expr = transformer.transform_query(all_vars_to_rename)
        cluster_part = f", {cluster_col}" if cluster_col else ""
        self.conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE {_FS_NORM} AS
            SELECT {rename_expr}{cluster_part}
            FROM {result_table}
        """)
        result_table = _FS_NORM

        # Fit OLS on design_matrix
        if self.fitter == "duckdb":
            coefs, vcov, vcov_meta, n = self._ols_duckdb(
                result_table, x_sql, y_sql, cluster_col
            )
        else:
            coefs, vcov, vcov_meta, n = self._ols_numpy(
                result_table, x_sql, y_sql, cluster_col
            )

        # Copy fitted values back to staging table via _row_idx join
        self._add_fitted_column(
            design_table=result_table,
            endog_sql=endog_var_obj.sql_name,
            x_sql=x_sql,
            coefs=coefs,
        )

        coef_names = self._build_fs_display_names(exog_sql, inst_sql, transformer)

        reg_results = RegressionResults(
            coefficients=coefs.reshape(-1, 1),
            coef_names=coef_names,
            vcov=vcov,
            n_obs=n,
            se_type=vcov_meta.get("vcov_type_detail", "HC1"),
        )
        return FirstStageResults(
            endog_var=endog_var,
            results=reg_results,
            instrument_names=self.instrument_vars,
        )

    # =========================================================================
    # Demean path: FE absorption via iterative demeaning
    # =========================================================================

    def _demean_all_variables(self) -> str:
        """Absorb fixed effects via :class:`IterativeDemeanTransformer` (method='demean').

        Runs the transformer once on the union of outcomes, exogenous
        regressors, endogenous variables, and instruments.  The result is
        stored in the ``demeaned_data`` temp table.  A renaming view
        ``_demeaned_staging`` is created so that downstream SQL uses the
        original column names (``resid_v AS v``).

        Sets ``self._demean_transformer``, ``self._demean_result_table``, and
        ``self._df_correction`` (total absorbed FE levels).

        Returns
        -------
        str
            Name of the renaming view (``"_demeaned_staging"``).
        """
        exog_sql         = self._get_exog_sql()
        actual_endog_sql = [v.sql_name for v in self.formula.endogenous]
        outcome_sql      = [v.sql_name for v in self.formula.outcomes]
        inst_sql         = self._get_inst_sql()
        cluster_col      = self._CLUSTER_ALIAS if self._effective_cluster_col else None

        variables = outcome_sql + exog_sql + actual_endog_sql + inst_sql

        transformer = IterativeDemeanTransformer(
            conn=self.conn,
            table_name=self._STAGING_TABLE,
            fe_cols=self._resolve_fe_sql_names() if self.fe_cols else [],
            cluster_col=cluster_col,
            remove_singletons=False,   # singletons already handled in prepare_data
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )
        self._demean_result_table = transformer.fit_transform(variables, where_clause="")
        self._demean_transformer  = transformer
        self._df_correction       = transformer.df_correction

        # Build a renaming view so downstream SQL uses original column names
        rename_fragment = transformer.transform_query(variables)
        fe_sql_names    = self._resolve_fe_sql_names()
        select_parts    = []
        if fe_sql_names:
            select_parts.append(", ".join(fe_sql_names))
        if cluster_col:
            select_parts.append(cluster_col)
        select_parts.append(rename_fragment)

        self.conn.execute(f"""
        CREATE OR REPLACE VIEW {self._DEMEANED_STAGING} AS
        SELECT {', '.join(select_parts)}
        FROM {self._demean_result_table}
        """)

        logger.debug(
            f"_demean_all_variables: {len(variables)} variable(s) demeaned, "
            f"df_correction={self._df_correction}"
        )
        return self._DEMEANED_STAGING

    def _run_first_stages_demean(self):
        """Run first stages as plain OLS on the pre-demeaned table (no intercept).

        FEs are already absorbed by :meth:`_demean_all_variables`; no
        transformer is needed here.  Fitted values are written back to the
        underlying ``demeaned_data`` table so they can be exposed via the
        ``_demeaned_staging`` view after it is rebuilt in
        :meth:`_setup_second_stage_demean`.
        """
        exog_sql    = self._get_exog_sql()
        inst_sql    = self._get_inst_sql()
        cluster_col = self._CLUSTER_ALIAS if self.cluster_col else None
        x_sql       = exog_sql + inst_sql

        for endog_var in self.endogenous_vars:
            endog_var_obj = next(
                v for v in self.formula.endogenous if v.display_name == endog_var
            )
            if self.fitter == "duckdb":
                coefs, vcov, meta, n = self._ols_duckdb_no_intercept(
                    self._DEMEANED_STAGING, x_sql, endog_var_obj.sql_name, cluster_col
                )
            else:
                df = self.conn.execute(
                    f"SELECT * FROM {self._DEMEANED_STAGING}"
                ).fetchdf().dropna()
                n   = len(df)
                ids = (
                    df[cluster_col].values
                    if cluster_col and cluster_col in df.columns
                    else None
                )
                X = df[x_sql].values if x_sql else np.empty((n, 0))
                y = df[endog_var_obj.sql_name].values.reshape(-1, 1)
                w = np.ones(n)
                np_fitter = NumpyFitter(alpha=1e-8, se_type="stata")
                fit    = np_fitter.fit(X=X, y=y, weights=w, coef_names=x_sql)
                vcov, meta, _ = np_fitter.fit_vcov(
                    X=X, y=y, weights=w,
                    coefficients=fit.coefficients,
                    cluster_ids=ids,
                    existing_result=fit,
                )
                coefs = fit.coefficients.flatten()

            self._add_fitted_column_demean(
                endog_sql=endog_var_obj.sql_name,
                x_sql=x_sql,
                coefs=coefs,
            )

            reg_results = RegressionResults(
                coefficients=coefs.reshape(-1, 1),
                coef_names=x_sql,
                vcov=vcov,
                n_obs=n,
                se_type=meta.get("vcov_type_detail", "HC1"),
            )
            self._first_stage_results[endog_var] = FirstStageResults(
                endog_var=endog_var,
                results=reg_results,
                instrument_names=self.instrument_vars,
            )
            logger.debug(
                f"Demean first stage '{endog_var}': "
                f"F-stat={self._first_stage_results[endog_var].f_statistic}"
            )

    # =========================================================================
    # Second stage setup
    # =========================================================================

    def _setup_second_stage(self):
        """Run second-stage MundlakTransformer and prepare for estimation."""
        exog_sql         = self._get_exog_sql()
        fitted_endog_sql = [f"fitted_{v.sql_name}" for v in self.formula.endogenous]
        actual_endog_sql = [v.sql_name for v in self.formula.endogenous]
        outcome_sql      = [v.sql_name for v in self.formula.outcomes]
        inst_sql         = self._get_inst_sql()
        cluster_col      = self._CLUSTER_ALIAS if self.cluster_col else None

        # Cache for downstream helpers
        self._exog_sql         = exog_sql
        self._fitted_endog_sql = fitted_endog_sql
        self._actual_endog_sql = actual_endog_sql
        self._outcome_sql      = outcome_sql
        self._inst_sql         = inst_sql

        # covariate_cols covers exog + fitted_endog + inst so extra_regressors
        # contain Mundlak means for all three sets of columns
        self._ss_transformer = self._make_transformer(
            covariate_cols=exog_sql + fitted_endog_sql + inst_sql,
            source_table=self._STAGING_TABLE,
            cluster_col=cluster_col,
        )

        # Include _row_idx, actual endogenous (for residuals), instruments (for Z)
        variables = (
            outcome_sql
            + exog_sql
            + fitted_endog_sql
            + actual_endog_sql
            + inst_sql
            + ["_row_idx"]
        )
        self._ss_result_table = self._ss_transformer.fit_transform(
            variables, where_clause=""
        )
        logger.debug(
            f"Second-stage design_matrix: n_obs={self._ss_transformer.n_obs}, "
            f"extra_regressors={self._ss_transformer.extra_regressors}"
        )

        if self.fitter == "duckdb":
            self._create_compressed_view()
        else:
            self._build_numpy_arrays()

    def _create_compressed_view(self):
        """Create the compressed view for DuckDB-path estimation."""
        extra       = self._ss_transformer.extra_regressors
        cluster_col = self._CLUSTER_ALIAS if self.cluster_col else None

        # Group-by columns: X cols + instruments (for Z) + extra Mundlak cols
        group_cols = (
            self._exog_sql
            + self._fitted_endog_sql
            + self._inst_sql
            + extra
            + ([cluster_col] if cluster_col else [])
        )

        agg_parts = ["COUNT(*) AS count"]
        for sql_name in self._outcome_sql:
            agg_parts.append(f"SUM({sql_name}) AS sum_{sql_name}")
            agg_parts.append(f"SUM({sql_name} * {sql_name}) AS sum_{sql_name}_sq")
        # Aggregate actual endogenous for residual computation;
        # expose cell mean as the original column name for residual_x_cols.
        for sql_name in self._actual_endog_sql:
            agg_parts.append(f"SUM({sql_name}) AS sum_{sql_name}")
            agg_parts.append(f"SUM({sql_name} * {sql_name}) AS sum_{sql_name}_sq")
            agg_parts.append(f"SUM({sql_name}) / NULLIF(COUNT(*), 0) AS {sql_name}")

        group_by = ", ".join(group_cols)
        self.conn.execute(f"""
        CREATE OR REPLACE VIEW {self._COMPRESSED_VIEW} AS
        SELECT {group_by}, {', '.join(agg_parts)}
        FROM {self._ss_result_table}
        GROUP BY {group_by}
        """)

        result = self.conn.execute(
            f"SELECT SUM(count), COUNT(*) FROM {self._COMPRESSED_VIEW}"
        ).fetchone()
        self.n_obs             = int(result[0]) if result[0] else 0
        self.n_compressed_rows = int(result[1]) if result[1] else 0

    def _setup_second_stage_demean(self):
        """Rebuild the demeaned staging view (with fitted columns) and populate
        numpy arrays for second-stage estimation (demean path)."""
        exog_sql         = self._get_exog_sql()
        fitted_endog_sql = [f"fitted_{v.sql_name}" for v in self.formula.endogenous]
        actual_endog_sql = [v.sql_name for v in self.formula.endogenous]
        outcome_sql      = [v.sql_name for v in self.formula.outcomes]
        inst_sql         = self._get_inst_sql()
        cluster_col      = self._CLUSTER_ALIAS if self.cluster_col else None

        # Cache for downstream helpers
        self._exog_sql         = exog_sql
        self._fitted_endog_sql = fitted_endog_sql
        self._actual_endog_sql = actual_endog_sql
        self._outcome_sql      = outcome_sql
        self._inst_sql         = inst_sql

        # Rebuild _demeaned_staging to include the fitted endogenous columns
        # that were added to demeaned_data by _run_first_stages_demean.
        demean_vars     = outcome_sql + exog_sql + actual_endog_sql + inst_sql
        rename_fragment = self._demean_transformer.transform_query(demean_vars)
        fe_sql_names    = self._resolve_fe_sql_names()
        select_parts    = []
        if fe_sql_names:
            select_parts.append(", ".join(fe_sql_names))
        if cluster_col:
            select_parts.append(cluster_col)
        select_parts.append(rename_fragment)
        # Expose fitted columns directly (not resid_-prefixed)
        for fc in fitted_endog_sql:
            select_parts.append(fc)

        self.conn.execute(f"""
        CREATE OR REPLACE VIEW {self._DEMEANED_STAGING} AS
        SELECT {', '.join(select_parts)}
        FROM {self._demean_result_table}
        """)

        self._ss_result_table = self._DEMEANED_STAGING
        # No transformer needed for second stage; set to None so mundlak helpers
        # that check extra_regressors are not called accidentally.
        self._ss_transformer  = None

        if self.fitter == "duckdb":
            # Create a lightweight helper view that adds a constant weight column
            # and squared-outcome column so DuckDBFitter can operate OOC.
            outcome_col = self._outcome_sql[0]
            self.conn.execute(f"""
            CREATE OR REPLACE VIEW {self._DEMEANED_STAGING}_fit AS
            SELECT *, 1 AS count, {outcome_col} * {outcome_col} AS {outcome_col}_sq
            FROM {self._DEMEANED_STAGING}
            """)
            logger.debug("Created demeaned fit view for DuckDB second stage.")
        else:
            self._build_numpy_arrays_demean()

    def _estimate_duckdb_demean(self) -> np.ndarray:
        """Estimate second-stage coefficients via DuckDB (demean path, no intercept)."""
        x_sql       = self._exog_sql + self._fitted_endog_sql
        y_col       = self._outcome_sql[0]
        cluster_col = self._CLUSTER_ALIAS if self.cluster_col else None
        view        = f"{self._DEMEANED_STAGING}_fit"

        duckdb_fitter = DuckDBFitter(conn=self.conn, alpha=1e-8, se_type="stata")
        self._fitter_result = duckdb_fitter.fit(
            table_name=view,
            x_cols=x_sql,
            y_col=y_col,
            weight_col="count",
            add_intercept=False,
            cluster_col=cluster_col,
        )

        sql_to_display = {v.sql_name: v.display_name for v in self.formula.covariates}
        sql_to_display.update(
            {v.sql_name: v.display_name for v in self.formula.endogenous}
        )
        names = []
        for sql in self._exog_sql:
            names.append(sql_to_display.get(sql, sql))
        for sql in self._fitted_endog_sql:
            base = sql[len("fitted_"):] if sql.startswith("fitted_") else sql
            names.append(sql_to_display.get(base, base))
        self.coef_names_     = names
        self._coef_sql_names = x_sql
        self.n_obs           = self._fitter_result.n_obs
        return self._fitter_result.coefficients

    def _fit_vcov_duckdb_demean(self, vcov_spec: VcovSpec = None):
        """Compute IV vcov via DuckDB for the demean path (no intercept)."""
        if (
            self._fitter_result is not None
            and self._fitter_result.vcov is not None
        ):
            self.vcov        = self._fitter_result.vcov
            self.se          = self._fitter_result.se_type
            self._n_clusters = self._fitter_result.n_clusters
            self._results    = None
            return

        if vcov_spec is None:
            vcov_spec = VcovSpec.build('HC1', None, is_iv=True)
        x_sql       = self._exog_sql + self._fitted_endog_sql
        residual_x  = self._exog_sql + self._actual_endog_sql
        z_cols      = self._exog_sql + self._inst_sql
        y_col       = self._outcome_sql[0]
        cluster_col = self._CLUSTER_ALIAS if self.cluster_col else None
        view        = f"{self._DEMEANED_STAGING}_fit"

        duckdb_fitter = DuckDBFitter(conn=self.conn, alpha=1e-8, se_type=vcov_spec.vcov_detail)
        vcov, vcov_meta, _ = duckdb_fitter.fit_vcov(
            table_name=view,
            x_cols=x_sql,
            y_col=y_col,
            weight_col="count",
            add_intercept=False,
            coefficients=self.point_estimate,
            cluster_col=cluster_col,
            vcov_spec=vcov_spec,
            residual_x_cols=residual_x,
            z_cols=z_cols,
            is_iv=True,
            k_fe=self._df_correction,
        )
        self.vcov        = vcov
        self.se          = vcov_meta["vcov_type_detail"]
        self._n_clusters = vcov_meta.get("n_clusters")
        self._results    = None

    def _ols_duckdb_no_intercept(
        self,
        table: str,
        x_sql: List[str],
        y_sql: str,
        cluster_col: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray, dict, int]:
        """Fit OLS via DuckDBFitter without an intercept (for pre-demeaned data)."""
        self.conn.execute(
            f"CREATE OR REPLACE VIEW _fs_demean_view AS "
            f"SELECT *, 1 AS count FROM {table}"
        )
        fitter = DuckDBFitter(conn=self.conn, alpha=1e-8, se_type="stata")
        fit    = fitter.fit(
            table_name="_fs_demean_view",
            x_cols=x_sql,
            y_col=y_sql,
            weight_col="count",
            add_intercept=False,
            cluster_col=cluster_col,
        )
        vcov, meta, _ = fitter.fit_vcov(
            table_name="_fs_demean_view",
            x_cols=x_sql,
            y_col=y_sql,
            weight_col="count",
            add_intercept=False,
            cluster_col=cluster_col,
            coefficients=fit.coefficients,
            existing_result=fit,
        )
        return fit.coefficients.flatten(), vcov, meta, fit.n_obs

    def _add_fitted_column_demean(
        self,
        endog_sql: str,
        x_sql: List[str],
        coefs: np.ndarray,
    ):
        """Write first-stage fitted values into *_demean_result_table* and refresh view.

        Coefficients are ordered as [*x_sql] (no intercept term).
        The ``_DEMEANED_STAGING`` view is recreated after the column is added
        so it exposes the new ``fitted_`` column to downstream queries.
        """
        fitted_col   = f"fitted_{endog_sql}"
        resid_prefix = "resid_"

        if x_sql:
            expr_parts = [
                f"({float(coefs[i])} * {quote_identifier(resid_prefix + col)})"
                for i, col in enumerate(x_sql)
            ]
            fitted_expr = " + ".join(expr_parts)
        else:
            fitted_expr = "0.0"

        self.conn.execute(
            f"ALTER TABLE {self._demean_result_table} "
            f"ADD COLUMN {quote_identifier(fitted_col)} DOUBLE"
        )
        self.conn.execute(
            f"UPDATE {self._demean_result_table} "
            f"SET {quote_identifier(fitted_col)} = {fitted_expr}"
        )
        logger.debug(f"Added column '{fitted_col}' to demeaned result table.")

    def _build_numpy_arrays_demean(self):
        """Fetch the demeaned staging view and build numpy arrays (no intercept).

        Since all variables are already within-demeaned, the intercept is
        suppressed throughout (X_fitted, X_actual, and Z have no ones column).
        The DOF correction from :attr:`_df_correction` accounts for the
        absorbed FE levels in downstream vcov computations.
        """
        df = self.conn.execute(
            f"SELECT * FROM {self._ss_result_table}"
        ).fetchdf().dropna()
        n           = len(df)
        cluster_col = self._CLUSTER_ALIAS if self.cluster_col else None

        # Outcome
        self._y = df[self._outcome_sql[0]].values.reshape(-1, 1)

        # X_fitted: exog + fitted_endog  (no intercept)
        x_fitted_cols  = self._exog_sql + self._fitted_endog_sql
        self._X_fitted = df[x_fitted_cols].values

        # X_actual: exog + actual_endog  (no intercept, for residual sandwich)
        x_actual_cols  = self._exog_sql + self._actual_endog_sql
        self._X_actual = df[x_actual_cols].values

        # Z: exog + instruments  (no intercept)
        z_cols   = self._exog_sql + self._inst_sql
        self._Z  = df[z_cols].values

        self._weights = np.ones(n)
        self.n_obs    = n

        # Coefficient names (no 'Intercept')
        sql_to_display = {v.sql_name: v.display_name for v in self.formula.covariates}
        sql_to_display.update(
            {v.sql_name: v.display_name for v in self.formula.endogenous}
        )
        names = []
        for sql in self._exog_sql:
            names.append(sql_to_display.get(sql, sql))
        for sql in self._fitted_endog_sql:
            base = sql[len("fitted_"):] if sql.startswith("fitted_") else sql
            names.append(sql_to_display.get(base, base))
        self.coef_names_     = names
        self._coef_sql_names = x_fitted_cols

        self._cluster_ids = (
            df[cluster_col].values
            if cluster_col and cluster_col in df.columns
            else None
        )

        logger.debug(
            f"_build_numpy_arrays_demean: n={n}, "
            f"X_fitted.shape={self._X_fitted.shape}, "
            f"Z.shape={self._Z.shape}, df_correction={self._df_correction}"
        )

    def _build_numpy_arrays(self):
        """Fetch design_matrix and build numpy arrays for estimation."""
        df = self.conn.execute(
            f"SELECT * FROM {self._ss_result_table}"
        ).fetchdf().dropna()
        n = len(df)

        extra       = self._ss_transformer.extra_regressors
        x_extra     = self._x_extra(extra)
        z_extra     = self._z_extra(extra)
        cluster_col = self._CLUSTER_ALIAS if self.cluster_col else None

        # Outcome
        self._y = df[self._outcome_sql[0]].values.reshape(-1, 1)

        # X_fitted: intercept + exog + fitted_endog + x_extra
        x_fitted_cols    = self._exog_sql + self._fitted_endog_sql + x_extra
        self._X_fitted   = np.c_[np.ones(n), df[x_fitted_cols].values]

        # X_actual: same layout but with actual endogenous (for residuals)
        x_actual_cols    = self._exog_sql + self._actual_endog_sql + x_extra
        self._X_actual   = np.c_[np.ones(n), df[x_actual_cols].values]

        # Z: intercept + exog + instruments + z_extra
        z_cols   = self._exog_sql + self._inst_sql + z_extra
        self._Z  = np.c_[np.ones(n), df[z_cols].values]

        self._weights = np.ones(n)
        self.n_obs    = n

        # Coefficient names
        self.coef_names_     = self._build_second_stage_coef_names(x_extra)
        self._coef_sql_names = ["Intercept"] + x_fitted_cols

        self._cluster_ids = (
            df[cluster_col].values
            if cluster_col and cluster_col in df.columns
            else None
        )

    # =========================================================================
    # Estimation
    # =========================================================================

    def _estimate_numpy(self) -> np.ndarray:
        if self._X_fitted is None:
            raise RuntimeError("compress_data() must be called before estimate()")
        fitter = NumpyFitter(alpha=1e-8, se_type="stata")
        result = fitter.fit(
            X=self._X_fitted,
            y=self._y,
            weights=self._weights,
            coef_names=self.coef_names_,
        )
        return result.coefficients

    def _estimate_duckdb(self) -> np.ndarray:
        extra       = self._ss_transformer.extra_regressors
        x_extra     = self._x_extra(extra)
        x_sql       = self._exog_sql + self._fitted_endog_sql + x_extra
        residual_x  = self._exog_sql + self._actual_endog_sql + x_extra
        y_col       = f"sum_{self._outcome_sql[0]}"
        cluster_col = self._CLUSTER_ALIAS if self.cluster_col else None

        duckdb_fitter = DuckDBFitter(conn=self.conn, alpha=1e-8, se_type="stata")
        self._fitter_result = duckdb_fitter.fit(
            table_name=self._COMPRESSED_VIEW,
            x_cols=x_sql,
            y_col=y_col,
            weight_col="count",
            add_intercept=True,
            cluster_col=cluster_col,
            residual_x_cols=residual_x,
        )

        self.coef_names_     = self._build_second_stage_coef_names(x_extra)
        self._coef_sql_names = ["Intercept"] + x_sql
        return self._fitter_result.coefficients

    # =========================================================================
    # Variance-covariance
    # =========================================================================

    def fit_vcov(self, se_method: str = "HC1"):
        """Compute vcov with correct 2SLS formula using actual residuals."""
        vcov_spec = self.vcov_spec or VcovSpec.build(se_method, None, is_iv=True)
        if self.fitter == "duckdb":
            if self.method == "demean":
                self._fit_vcov_duckdb_demean(vcov_spec)
            else:
                self._fit_vcov_duckdb(vcov_spec)
        else:
            self._fit_vcov_numpy(vcov_spec)

    def _fit_vcov_numpy(self, vcov_spec: VcovSpec = None):
        if self._X_actual is None or self._X_fitted is None or self._Z is None:
            raise RuntimeError("compress_data() must be called before fit_vcov()")
        if vcov_spec is None:
            vcov_spec = VcovSpec.build('HC1', None, is_iv=True)
        fitter = NumpyFitter(alpha=1e-8, se_type=vcov_spec.vcov_detail)
        vcov, vcov_meta, _ = fitter.fit_vcov(
            X=self._X_fitted,
            y=self._y,
            weights=self._weights,
            coefficients=self.point_estimate,
            cluster_ids=self._cluster_ids,
            vcov_spec=vcov_spec,
            Z=self._Z,
            is_iv=True,
            residual_X=self._X_actual,
        )
        self.vcov        = vcov
        self.se          = vcov_meta["vcov_type_detail"]
        self._n_clusters = vcov_meta.get("n_clusters")
        self._results    = None

    def _fit_vcov_duckdb(self, vcov_spec: VcovSpec = None):
        if (
            self._fitter_result is not None
            and self._fitter_result.vcov is not None
        ):
            self.vcov        = self._fitter_result.vcov
            self.se          = self._fitter_result.se_type
            self._n_clusters = self._fitter_result.n_clusters
            self._results    = None
            return

        if vcov_spec is None:
            vcov_spec = VcovSpec.build('HC1', None, is_iv=True)
        extra      = self._ss_transformer.extra_regressors
        x_extra    = self._x_extra(extra)
        z_extra    = self._z_extra(extra)
        x_sql      = self._exog_sql + self._fitted_endog_sql + x_extra
        residual_x = self._exog_sql + self._actual_endog_sql + x_extra
        y_col      = f"sum_{self._outcome_sql[0]}"
        z_cols     = self._exog_sql + self._inst_sql + z_extra
        cluster_col = self._CLUSTER_ALIAS if self.cluster_col else None

        duckdb_fitter = DuckDBFitter(conn=self.conn, alpha=1e-8, se_type=vcov_spec.vcov_detail)
        vcov, vcov_meta, _ = duckdb_fitter.fit_vcov(
            table_name=self._COMPRESSED_VIEW,
            x_cols=x_sql,
            y_col=y_col,
            weight_col="count",
            add_intercept=True,
            coefficients=self.point_estimate,
            cluster_col=cluster_col,
            vcov_spec=vcov_spec,
            residual_x_cols=residual_x,
            z_cols=z_cols,
            is_iv=True,
        )
        self.vcov        = vcov
        self.se          = vcov_meta["vcov_type_detail"]
        self._n_clusters = vcov_meta.get("n_clusters")
        self._results    = None

    def bootstrap(self) -> np.ndarray:
        logger.warning("Bootstrap for 2SLS is not yet implemented; using analytical SEs.")
        self.fit_vcov()  # reads self.vcov_spec internally
        return self.vcov

    # =========================================================================
    # OLS helpers used inside the first-stage loop
    # =========================================================================

    def _ols_duckdb(
        self,
        table: str,
        x_sql: List[str],
        y_sql: str,
        cluster_col: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray, dict, int]:
        """Fit OLS via DuckDBFitter on *table* using column lists."""
        self.conn.execute(
            f"CREATE OR REPLACE VIEW _fs_view AS SELECT *, 1 AS count FROM {table}"
        )
        fitter = DuckDBFitter(conn=self.conn, alpha=1e-8, se_type="stata")
        fit    = fitter.fit(
            table_name="_fs_view",
            x_cols=x_sql,
            y_col=y_sql,
            weight_col="count",
            add_intercept=True,
            cluster_col=cluster_col,
        )
        vcov, meta, _ = fitter.fit_vcov(
            table_name="_fs_view",
            x_cols=x_sql,
            y_col=y_sql,
            weight_col="count",
            add_intercept=True,
            cluster_col=cluster_col,
            coefficients=fit.coefficients,
            existing_result=fit,
        )
        return fit.coefficients.flatten(), vcov, meta, fit.n_obs

    def _ols_numpy(
        self,
        table: str,
        x_sql: List[str],
        y_sql: str,
        cluster_col: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray, dict, int]:
        """Fit OLS via NumpyFitter on *table* using column lists."""
        df  = self.conn.execute(f"SELECT * FROM {table}").fetchdf().dropna()
        n   = len(df)
        y   = df[y_sql].values.reshape(-1, 1)
        X   = np.c_[np.ones(n), df[x_sql].values]
        w   = np.ones(n)
        ids = (
            df[cluster_col].values
            if cluster_col and cluster_col in df.columns
            else None
        )
        fitter = NumpyFitter(alpha=1e-8, se_type="stata")
        fit    = fitter.fit(X=X, y=y, weights=w, coef_names=["Intercept"] + x_sql)
        vcov, meta, _ = fitter.fit_vcov(
            X=X, y=y, weights=w,
            coefficients=fit.coefficients,
            cluster_ids=ids,
            existing_result=fit,
        )
        return fit.coefficients.flatten(), vcov, meta, n

    # =========================================================================
    # Fitted-value transfer  (design_matrix → staging table)
    # =========================================================================

    def _add_fitted_column(
        self,
        design_table: str,
        endog_sql: str,
        x_sql: List[str],
        coefs: np.ndarray,
    ):
        """Compute fitted values in *design_table* and copy back to staging.

        Both tables carry a ``_row_idx`` column that acts as the join key.
        Coefficients are ordered as [intercept, *x_sql].
        """
        fitted_col = f"fitted_{endog_sql}"

        # intercept term + Σ coef_i * col_i
        expr_parts = [str(float(coefs[0]))]
        for i, col in enumerate(x_sql):
            expr_parts.append(f"({float(coefs[i + 1])} * {quote_identifier(col)})")
        fitted_expr = " + ".join(expr_parts)

        self.conn.execute(
            f"ALTER TABLE {design_table} "
            f"ADD COLUMN {quote_identifier(fitted_col)} DOUBLE"
        )
        self.conn.execute(
            f"UPDATE {design_table} "
            f"SET {quote_identifier(fitted_col)} = {fitted_expr}"
        )

        # Recreate staging table with the new fitted column appended
        self.conn.execute(f"""
        CREATE OR REPLACE TABLE {self._STAGING_TABLE} AS
        SELECT s.*, d.{quote_identifier(fitted_col)}
        FROM {self._STAGING_TABLE} s
        JOIN {design_table} d ON s._row_idx = d._row_idx
        """)
        logger.debug(f"Added fitted column '{fitted_col}' to staging table.")

    # =========================================================================
    # Singleton removal
    # =========================================================================

    def _remove_singleton_fe_observations(self):
        if not self.remove_singletons or not self.fe_cols:
            return
        fe_sql_names = self._resolve_fe_sql_names()
        if fe_sql_names:
            self._remove_singleton_observations(self._STAGING_TABLE, fe_sql_names)

    # =========================================================================
    # Column name helpers
    # =========================================================================

    def _get_exog_sql(self) -> List[str]:
        """SQL names of exogenous (non-intercept, non-endogenous) covariates."""
        return [
            v.sql_name for v in self.formula.covariates
            if not v.is_intercept() and v.display_name not in self.endogenous_vars
        ]

    def _get_inst_sql(self) -> List[str]:
        """SQL names of instruments."""
        return [v.sql_name for v in self.formula.instruments]

    def _x_extra(self, extra: List[str]) -> List[str]:
        """Subset of *extra_regressors* for the second-stage X matrix.

        Keeps Mundlak means of exog and fitted endogenous; excludes means of
        instruments (those belong in Z only).  Fixed-FE dummies and dummy-means
        are kept in both X and Z.
        """
        inst_prefixes = {f"avg_{s}_fe" for s in (self._inst_sql or [])}
        return [
            col for col in extra
            if not any(col.startswith(p) for p in inst_prefixes)
        ]

    def _z_extra(self, extra: List[str]) -> List[str]:
        """Subset of *extra_regressors* for the instrument matrix Z.

        Keeps Mundlak means of exog and instruments; excludes means of fitted
        endogenous (those belong in X only).  Fixed-FE dummies and dummy-means
        are kept.
        """
        fitted_prefixes = {f"avg_{s}_fe" for s in (self._fitted_endog_sql or [])}
        return [
            col for col in extra
            if not any(col.startswith(p) for p in fitted_prefixes)
        ]

    def _build_fs_display_names(
        self,
        exog_sql: List[str],
        inst_sql: List[str],
        transformer: MundlakTransformer,
    ) -> List[str]:
        """Human-readable coefficient names for first-stage output."""
        sql_to_display = {v.sql_name: v.display_name for v in self.formula.covariates}
        sql_to_display.update(
            {v.sql_name: v.display_name for v in self.formula.instruments}
        )
        names = ["Intercept"]
        for sql in exog_sql:
            names.append(sql_to_display.get(sql, sql))
        for sql in inst_sql:
            names.append(sql_to_display.get(sql, sql))
        names.extend(transformer.extra_regressors)
        return names

    def _build_second_stage_coef_names(self, x_extra: List[str]) -> List[str]:
        """Human-readable coefficient names for second-stage output."""
        sql_to_display = {v.sql_name: v.display_name for v in self.formula.covariates}
        sql_to_display.update(
            {v.sql_name: v.display_name for v in self.formula.endogenous}
        )
        names = ["Intercept"]
        for sql in (self._exog_sql or []):
            names.append(sql_to_display.get(sql, sql))
        for sql in (self._fitted_endog_sql or []):
            base = sql[len("fitted_"):] if sql.startswith("fitted_") else sql
            names.append(sql_to_display.get(base, base))
        names.extend(x_extra)
        return names

    # =========================================================================
    # WHERE clause
    # =========================================================================

    def _build_where_clause(self, user_subset: Optional[str] = None) -> str:
        return self.formula.get_where_clause_sql(user_subset)

    # =========================================================================
    # Summary / output
    # =========================================================================

    def summary(self) -> Dict[str, Any]:
        from ..core.results import ModelSummary
        return ModelSummary.from_estimator(self).to_dict()

    def summary_df(self) -> pd.DataFrame:
        if self.results is None:
            return pd.DataFrame()
        return self.results.to_dataframe()

    def print_summary(self, precision: int = 4, include_diagnostics: bool = True):
        """Print formatted 2SLS results to console using unified formatter."""
        from ..utils.summary import print_summary as fmt_print
        fmt_print(
            self.summary(),
            precision=precision,
            include_diagnostics=include_diagnostics,
        )

    def to_tidy_df(self) -> pd.DataFrame:
        """Get results as a tidy DataFrame using unified formatter."""
        from ..utils.summary import to_tidy_df as fmt_tidy
        if self.results:
            return fmt_tidy(self.results)
        return pd.DataFrame()