"""
Generalized linear mediation estimator for large datasets.

``DuckMediation`` fits a system of equations representing parallel linear
mediation:

    For each mediator ``m_k`` in ``M``:
        m_k = alpha_k + A_k · X + G_k · C + u_k          (mediator equations)

    Outcome equation:
        Y   = beta   + C' · X  + B · M  + P · C + e

where X are the focal exposures, M the mediators, and C the controls.

Direct, specific indirect, total indirect, and total effects are derived
from the estimated equation-level coefficients using the delta method for
analytical standard errors.

Supported FE methods: ``'demean'`` (iterative MAP), ``'mundlak'``.
Supported SE types: anything accepted by :class:`~duckreg.core.vcov.VcovSpec`.
Supported backends: ``'numpy'`` (always) and ``'duckdb'`` (staged to numpy
after FE absorption — pure DuckDB aggregation is not yet implemented for
multi-equation mediation).
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import DuckEstimator, SEMethod
from ..core.fitters.numpy_fitter import NumpyFitter
from ..core.transformers import IterativeDemeanTransformer, MundlakTransformer
from ..core.vcov import VcovSpec
from ..core.results import RegressionResults, MediationEffects, MediationResults


# ============================================================================
# EquationSpec — internal per-equation layout descriptor
# ============================================================================

@dataclass
class EquationSpec:
    """Internal description of a single regression equation.

    Used by :class:`DuckMediation` to encapsulate per-equation regressor
    lists and metadata without duplicating logic.

    Parameters
    ----------
    name : str
        Identifier for the equation (mediator column name, or
        ``'__outcome__'``).
    dependent : str
        SQL column name of the dependent variable.
    regressors : List[str]
        SQL column names of the regressors, *excluding* the intercept.
    has_intercept : bool
        Whether an intercept is included.  ``False`` for iterative-demean
        path (already zero-mean).
    exposure_indices : List[int]
        Zero-based positions of the exposure variables within *regressors*.
    mediator_indices : List[int]
        Zero-based positions of the mediator variables within *regressors*
        (empty for mediator equations).
    display_map : Dict[str, str]
        Map from SQL column name to display name shown in summaries.
    fe_correction : int
        Number of absorbed FE levels for DOF adjustment (demean path).
    n_fe : int
        Number of FE dimensions (demean path).
    """

    name: str
    dependent: str
    regressors: List[str]
    has_intercept: bool = field(default=True)
    exposure_indices: List[int] = field(default_factory=list)
    mediator_indices: List[int] = field(default_factory=list)
    display_map: Dict[str, str] = field(default_factory=dict)
    fe_correction: int = 0
    n_fe: int = 0

    @property
    def coef_names(self) -> List[str]:
        """Human-readable coefficient names (intercept first if present)."""
        names = []
        if self.has_intercept:
            names.append("Intercept")
        for col in self.regressors:
            names.append(self.display_map.get(col, col))
        return names

    @property
    def n_params(self) -> int:
        """Total number of parameters including intercept."""
        return len(self.regressors) + int(self.has_intercept)

    def intercept_offset(self) -> int:
        """Offset to add to regressor index to get parameter index."""
        return 1 if self.has_intercept else 0

logger = logging.getLogger(__name__)


class DuckMediation(DuckEstimator):
    """System-of-equations linear mediation estimator.

    Fits a parallel mediation system (no mediator-to-mediator paths)
    for large datasets using DuckDB for data preparation and optional
    fixed-effect absorption.

    Parameters
    ----------
    db_name : str
        DuckDB database path (``':memory:'`` for in-memory operation).
    table_name : str
        Source table / view name inside DuckDB.
    outcome : str
        Column name of the outcome variable Y.
    exposures : List[str]
        Column names of the focal exposure variables X.
    mediators : List[str], optional
        Column names of mediator variables M.  Pass an empty list (or omit)
        for a plain outcome regression without mediation.
    controls : List[str], optional
        Column names of control variables C.
    fe_cols : List[str], optional
        Column names of the fixed-effect dimensions.
    fe_method : {'demean', 'mundlak'}
        FE absorption strategy.  Ignored when ``fe_cols`` is empty.
        ``'demean'`` (default) absorbs FEs once via iterative MAP and runs
        all equations without an intercept.  ``'mundlak'`` augments each
        equation with within-group means of its covariates.
    cluster_col : str, optional
        Column name for cluster-robust standard errors.  Can also be
        supplied via ``se_method={'CRV1': 'col'}`` at :meth:`fit` time.
    vcov_spec : VcovSpec, optional
        Pre-parsed variance-covariance specification.  Built automatically
        from ``se_method`` when not provided.
    fitter : str
        ``'numpy'`` (default) or ``'duckdb'``.  Both paths stage data into
        numpy arrays after FE absorption; the setting currently controls
        staging behaviour only.
    subset : str, optional
        SQL ``WHERE`` clause to filter the source table.
    seed : int
        Random seed.
    remove_singletons : bool
        Drop singleton FE groups (observations that are the sole member of
        a FE cell).
    max_iterations : int
        Maximum MAP iterations for the ``'demean'`` path.
    tolerance : float
        MAP convergence tolerance.
    fe_types : Dict[str, str], optional
        Manual FE type overrides for the Mundlak path
        (``{'fe_col': 'fixed' | 'asymptotic'}``).
    cardinality_threshold : int
        Cardinality at or below which a FE is classified as *fixed*
        (Mundlak path only).
    max_fixed_fe_levels : int
        Upper cardinality limit for *fixed* FEs (Mundlak path only).
    duckdb_kwargs : Dict, optional
        DuckDB resource settings (e.g. ``threads``, ``memory_limit``).
    """

    _STAGING_TABLE = "_mediation_staging"
    _DEMEAN_VIEW   = "_mediation_demeaned"
    _CLUSTER_ALIAS = "__cluster__"

    # ------------------------------------------------------------------
    # Expression-variable helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _col_alias(name: str) -> str:
        """Return the SQL-safe alias for a column name or parenthesised expression."""
        from ..utils.formula_parser import _make_sql_safe_name
        return _make_sql_safe_name(name) if name.startswith('(') else name

    @staticmethod
    def _col_select_expr(raw_name: str, bool_cols: set) -> str:
        """Return a SELECT fragment for one variable.

        * Expression ``(col == 190)``  →  ``CAST((col = 190) AS SMALLINT) AS alias``
        * Boolean column               →  ``CAST(col AS SMALLINT) AS col``
        * Plain column                 →  ``col``
        """
        import re as _re
        from ..utils.formula_parser import _make_sql_safe_name
        if raw_name.startswith('('):
            alias    = _make_sql_safe_name(raw_name)
            sql_expr = _re.sub(r'(?<![!<>=])={2}(?!=)', '=', raw_name)
            return f"CAST({sql_expr} AS SMALLINT) AS {alias}"
        if raw_name in bool_cols:
            return f"CAST({raw_name} AS SMALLINT) AS {raw_name}"
        return raw_name

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        db_name: str,
        table_name: str,
        outcome: str,
        exposures: List[str],
        mediators: Optional[List[str]] = None,
        controls: Optional[List[str]] = None,
        fe_cols: Optional[List[str]] = None,
        fe_method: str = "demean",
        cluster_col: Optional[str] = None,
        vcov_spec: Optional[VcovSpec] = None,
        fitter: str = "numpy",
        subset: Optional[str] = None,
        seed: int = 42,
        remove_singletons: bool = True,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        fe_types: Optional[Dict[str, str]] = None,
        cardinality_threshold: int = 50,
        max_fixed_fe_levels: int = 100,
        duckdb_kwargs: Optional[Dict] = None,
        formula=None,
        **kwargs,
    ):
        super().__init__(
            db_name=db_name,
            table_name=table_name,
            seed=seed,
            n_bootstraps=0,
            fitter=fitter,
            remove_singletons=remove_singletons,
            duckdb_kwargs=duckdb_kwargs,
        )

        # ── variable blocks ─────────────────────────────────────────────
        self.outcome_col    = outcome
        # Keep raw names (may be expressions like "(col == 1)") for SELECT
        self._exposures_raw = list(exposures)
        self._mediators_raw = list(mediators or [])
        self._controls_raw  = list(controls or [])
        # SQL-safe aliases used as column names everywhere after staging
        self.exposures_sql  = [self._col_alias(v) for v in self._exposures_raw]
        self.mediators_sql  = [self._col_alias(v) for v in self._mediators_raw]
        self.controls_sql   = [self._col_alias(v) for v in self._controls_raw]
        self.fe_cols        = list(fe_cols or [])
        self._formula       = formula  # FormulaParser result; needed for interaction FE SELECT
        self.cluster_col    = cluster_col
        self.fe_method      = fe_method if self.fe_cols else None
        self.subset         = subset
        self.vcov_spec      = vcov_spec

        # convenience aliases expected by DuckEstimator helpers
        self.outcome_vars   = [outcome]
        self.covariates     = self.exposures_sql + self.mediators_sql + self.controls_sql

        if self.fe_method is not None and self.fe_method not in ("demean", "mundlak"):
            raise ValueError(
                f"DuckMediation: fe_method must be 'demean' or 'mundlak', "
                f"got {self.fe_method!r}."
            )

        # ── FE transformer settings ──────────────────────────────────────
        self.max_iterations        = max_iterations
        self.tolerance             = tolerance
        self.fe_types              = fe_types or {}
        self.cardinality_threshold = cardinality_threshold
        self.max_fixed_fe_levels   = max_fixed_fe_levels

        # ── internal state ───────────────────────────────────────────────
        #  per-equation data: name -> {X, y, cluster_ids, eq_spec}
        self._eq_data: Dict[str, Dict[str, Any]] = {}
        #  per-equation raw fitter results (no vcov)
        self._eq_raw: Dict[str, Any] = {}
        #  demean transformer (set after compress_data on demean path)
        self._demean_transformer: Optional[IterativeDemeanTransformer] = None
        self._df_correction: int = 0
        #  Mundlak extra regressors list
        self._extra_regressors: List[str] = []

        self.mediation_results: Optional[MediationResults] = None
        self.n_rows_dropped_singletons: int = 0

        logger.debug(
            f"DuckMediation: outcome={outcome}, "
            f"exposures={self.exposures_sql}, "
            f"mediators={self.mediators_sql}, "
            f"fe={self.fe_cols}, fe_method={self.fe_method}"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _effective_cluster_col(self) -> Optional[str]:
        """Cluster variable from explicit arg or from ``vcov_spec``."""
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
    def _cluster_staging_col(self) -> Optional[str]:
        """Name of the cluster column in the staging / transformed table."""
        eff = self._effective_cluster_col
        if not eff:
            return None
        if eff in self.fe_cols:
            return eff
        return self._CLUSTER_ALIAS

    # ------------------------------------------------------------------
    # Pipeline (DuckEstimator interface)
    # ------------------------------------------------------------------

    def prepare_data(self):
        """Build a denormalised staging table with all needed columns.

        Adds a ``_row_idx`` column and handles boolean casting.  Applies
        the optional ``subset`` filter.  After running, removes singleton
        FE observations.
        """
        # All variable inputs in their raw form (expressions preserved)
        # Note: fe_cols are handled separately via formula.get_fe_select_sql
        # to correctly expand interaction terms (e.g. country*year → country_year).
        all_raw = (
            [self.outcome_col]
            + self._exposures_raw
            + self._mediators_raw
            + self._controls_raw
        )
        # SQL-safe aliases (used for singleton removal, NULL checks, etc.)
        all_cols = (
            [self.outcome_col]
            + self.exposures_sql
            + self.mediators_sql
            + self.controls_sql
            + self.fe_cols
        )

        # Detect boolean columns in source table
        try:
            desc = self.conn.execute(
                f"SELECT column_name, data_type FROM (DESCRIBE {self.table_name})"
            ).fetchdf()
            bool_cols: set = set(
                desc[desc["data_type"].str.upper() == "BOOLEAN"]["column_name"].tolist()
            )
        except Exception:
            bool_cols = set()
        self._boolean_cols = bool_cols

        eff_cluster = self._effective_cluster_col

        select_parts = []
        for raw in all_raw:
            select_parts.append(self._col_select_expr(raw, bool_cols))

        # Add FE columns.  When a parsed Formula is available, use its
        # get_fe_select_sql() so that interaction terms like country*year are
        # properly expanded to  country || '_' || CAST(year AS VARCHAR) AS country_year
        # rather than being referenced as a non-existent plain column.
        if self.fe_cols:
            if self._formula is not None:
                fe_sql = self._formula.get_fe_select_sql(bool_cols)
                if fe_sql:
                    select_parts.append(fe_sql)
            else:
                # Fallback: plain columns only (no interaction support)
                for fe_col in self.fe_cols:
                    select_parts.append(fe_col)

        # Pull in cluster column if it is not already in the column list
        if eff_cluster and eff_cluster not in all_cols:
            select_parts.append(f"{eff_cluster} AS {self._CLUSTER_ALIAS}")

        select_parts.append("ROW_NUMBER() OVER () AS _row_idx")

        where_clause = f"WHERE {self.subset}" if self.subset else ""

        self.conn.execute(f"""
        CREATE OR REPLACE TABLE {self._STAGING_TABLE} AS
        SELECT {', '.join(select_parts)}
        FROM {self.table_name}
        {where_clause}
        """)

        # Singleton removal
        if self.remove_singletons and self.fe_cols:
            self._remove_singleton_observations(self._STAGING_TABLE, self.fe_cols)

        self.n_obs = self.conn.execute(
            f"SELECT COUNT(*) FROM {self._STAGING_TABLE}"
        ).fetchone()[0]
        logger.debug(f"DuckMediation.prepare_data: n_obs={self.n_obs}")

    def compress_data(self):
        """Apply FE transforms and build per-equation numpy arrays."""
        if not self.fe_cols or self.fe_method is None:
            self._stage_no_fe()
        elif self.fe_method == "demean":
            self._stage_demean()
        else:
            self._stage_mundlak()

    def estimate(self) -> np.ndarray:
        """Run OLS for each equation; return outcome-equation coefficients."""
        fitter = NumpyFitter(alpha=1e-8, se_type="stata")

        for eq_name, eq in self._eq_data.items():
            X, y = eq["X"], eq["y"]
            raw = fitter.fit(X=X, y=y, weights=np.ones(len(y)))
            self._eq_raw[eq_name] = (fitter, raw, eq)

        if "__outcome__" in self._eq_data:
            out_eq = self._eq_data["__outcome__"]
            _, raw_out, _ = self._eq_raw["__outcome__"]
            self.coef_names_ = out_eq["eq_spec"].coef_names
            self.point_estimate = raw_out.coefficients
            return raw_out.coefficients
        return np.array([])

    def fit_vcov(self, se_method: str = SEMethod.HC1):
        """Compute per-equation vcovs and derive mediation effects.

        Analytical delta-method SEs are computed for all direct, specific
        indirect, total indirect, and total effects.
        """
        vcov_spec = self.vcov_spec or VcovSpec.build(
            se_method, has_fixef=bool(self.fe_cols)
        )

        k_fe  = self._df_correction if self.fe_method == "demean" else 0
        n_fe  = len(self.fe_cols)    if self.fe_method == "demean" else 0

        eq_results: Dict[str, RegressionResults] = {}

        for eq_name, (np_fitter, raw, eq) in self._eq_raw.items():
            spec: EquationSpec = eq["eq_spec"]
            X, y, ids = eq["X"], eq["y"], eq["cluster_ids"]
            w = np.ones(len(y))

            vcov, vcov_meta, _ = np_fitter.fit_vcov(
                X=X, y=y, weights=w,
                coefficients=raw.coefficients,
                cluster_ids=ids,
                vcov_spec=vcov_spec,
                k_fe=k_fe, n_fe=n_fe,
                existing_result=raw,
            )
            eq_results[eq_name] = RegressionResults(
                coefficients=raw.coefficients.reshape(-1, 1),
                coef_names=spec.coef_names,
                vcov=vcov,
                n_obs=self.n_obs,
                se_type=vcov_meta.get("vcov_type_detail", se_method),
            )

        self.se    = vcov_spec.vcov_detail
        self.vcov  = eq_results.get("__outcome__", RegressionResults(
            np.array([]), [], None, None
        )).vcov

        med_results     = {k: v for k, v in eq_results.items() if k != "__outcome__"}
        outcome_result  = eq_results.get("__outcome__")

        effects = (
            self._compute_effects(med_results, outcome_result)
            if outcome_result is not None else None
        )

        self.mediation_results = MediationResults(
            mediator_results=med_results,
            outcome_result=outcome_result,
            effects=effects,
            n_obs=self.n_obs,
            se_type=self.se,
        )
        # Reset cached single-equation result to prevent stale access
        self._results = None
        logger.debug("DuckMediation.fit_vcov: mediation effects computed.")

    def bootstrap(self) -> np.ndarray:
        """Bootstrap stub — analytical SEs are used instead."""
        logger.warning(
            "DuckMediation: bootstrap not yet implemented; using analytical SEs."
        )
        self.fit_vcov()
        return self.vcov if self.vcov is not None else np.array([])

    # ------------------------------------------------------------------
    # Staging helpers
    # ------------------------------------------------------------------

    def _stage_no_fe(self):
        """No FE: build per-equation numpy arrays from raw staging data."""
        all_vars = (
            [self.outcome_col]
            + self.exposures_sql
            + self.mediators_sql
            + self.controls_sql
        )
        cluster_col = self._cluster_staging_col

        df = (
            self.conn.execute(f"SELECT * FROM {self._STAGING_TABLE}")
            .fetchdf()
            .dropna(subset=all_vars)
        )
        self.n_obs = len(df)
        ids = (
            df[cluster_col].values
            if cluster_col and cluster_col in df.columns
            else None
        )

        # ── mediator equations ───────────────────────────────────────
        for med in self.mediators_sql:
            regressors    = self.exposures_sql + self.controls_sql
            exp_idx       = list(range(len(self.exposures_sql)))
            X             = self._build_X(df, regressors, intercept=True)
            y             = df[med].values.reshape(-1, 1)
            spec          = self._make_eq_spec(
                name=med, dependent=med,
                regressors=regressors, has_intercept=True,
                exposure_indices=exp_idx, mediator_indices=[],
                fe_correction=0, n_fe=0,
            )
            self._eq_data[med] = dict(X=X, y=y, cluster_ids=ids, eq_spec=spec)

        # ── outcome equation ─────────────────────────────────────────
        regressors_out = self.exposures_sql + self.mediators_sql + self.controls_sql
        exp_idx_out    = list(range(len(self.exposures_sql)))
        med_idx_out    = list(range(
            len(self.exposures_sql),
            len(self.exposures_sql) + len(self.mediators_sql),
        ))
        X_out = self._build_X(df, regressors_out, intercept=True)
        y_out = df[self.outcome_col].values.reshape(-1, 1)
        spec_out = self._make_eq_spec(
            name="__outcome__", dependent=self.outcome_col,
            regressors=regressors_out, has_intercept=True,
            exposure_indices=exp_idx_out, mediator_indices=med_idx_out,
            fe_correction=0, n_fe=0,
        )
        self._eq_data["__outcome__"] = dict(
            X=X_out, y=y_out, cluster_ids=ids, eq_spec=spec_out
        )

        logger.debug(
            f"_stage_no_fe: n_obs={self.n_obs}, "
            f"n_mediator_eqs={len(self.mediators_sql)}"
        )

    def _stage_demean(self):
        """Demean all variables once; build per-equation arrays (no intercept)."""
        all_vars     = (
            [self.outcome_col]
            + self.exposures_sql
            + self.mediators_sql
            + self.controls_sql
        )
        cluster_col  = self._cluster_staging_col

        transformer  = IterativeDemeanTransformer(
            conn=self.conn,
            table_name=self._STAGING_TABLE,
            fe_cols=self.fe_cols,
            cluster_col=cluster_col,
            remove_singletons=False,   # singletons already handled in prepare_data
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )
        result_table = transformer.fit_transform(all_vars, where_clause="")
        self._demean_transformer = transformer
        self._df_correction      = transformer.df_correction

        # Build renaming view: ``resid_v AS v``
        rename_frag = transformer.transform_query(all_vars)
        parts: List[str] = []
        if self.fe_cols:
            parts.append(", ".join(self.fe_cols))
        if cluster_col:
            parts.append(cluster_col)
        parts.append(rename_frag)

        self.conn.execute(f"""
        CREATE OR REPLACE VIEW {self._DEMEAN_VIEW} AS
        SELECT {', '.join(parts)}
        FROM {result_table}
        """)

        df = (
            self.conn.execute(f"SELECT * FROM {self._DEMEAN_VIEW}")
            .fetchdf()
            .dropna(subset=all_vars)
        )
        self.n_obs = len(df)
        ids = (
            df[cluster_col].values
            if cluster_col and cluster_col in df.columns
            else None
        )

        # ── mediator equations (no intercept) ────────────────────────
        for med in self.mediators_sql:
            regressors  = self.exposures_sql + self.controls_sql
            exp_idx     = list(range(len(self.exposures_sql)))
            X           = self._build_X(df, regressors, intercept=False)
            y           = df[med].values.reshape(-1, 1)
            spec        = self._make_eq_spec(
                name=med, dependent=med,
                regressors=regressors, has_intercept=False,
                exposure_indices=exp_idx, mediator_indices=[],
                fe_correction=self._df_correction, n_fe=len(self.fe_cols),
            )
            self._eq_data[med] = dict(X=X, y=y, cluster_ids=ids, eq_spec=spec)

        # ── outcome equation (no intercept) ───────────────────────────
        regressors_out = self.exposures_sql + self.mediators_sql + self.controls_sql
        exp_idx_out    = list(range(len(self.exposures_sql)))
        med_idx_out    = list(range(
            len(self.exposures_sql),
            len(self.exposures_sql) + len(self.mediators_sql),
        ))
        X_out  = self._build_X(df, regressors_out, intercept=False)
        y_out  = df[self.outcome_col].values.reshape(-1, 1)
        spec_out = self._make_eq_spec(
            name="__outcome__", dependent=self.outcome_col,
            regressors=regressors_out, has_intercept=False,
            exposure_indices=exp_idx_out, mediator_indices=med_idx_out,
            fe_correction=self._df_correction, n_fe=len(self.fe_cols),
        )
        self._eq_data["__outcome__"] = dict(
            X=X_out, y=y_out, cluster_ids=ids, eq_spec=spec_out
        )

        logger.debug(
            f"_stage_demean: n_obs={self.n_obs}, "
            f"df_correction={self._df_correction}"
        )

    def _stage_mundlak(self):
        """Mundlak FE absorption; one transformer run, subsetted per equation."""
        all_cov_cols = self.exposures_sql + self.mediators_sql + self.controls_sql
        cluster_col  = self._cluster_staging_col
        variables    = [self.outcome_col] + all_cov_cols

        transformer  = MundlakTransformer(
            conn=self.conn,
            table_name=self._STAGING_TABLE,
            fe_cols=self.fe_cols,
            cluster_col=cluster_col,
            covariate_cols=all_cov_cols,
            remove_singletons=False,
            fe_types=self.fe_types,
            cardinality_threshold=self.cardinality_threshold,
            max_fixed_fe_levels=self.max_fixed_fe_levels,
        )
        result_table         = transformer.fit_transform(variables, where_clause="")
        self._extra_regressors = transformer.extra_regressors

        df = (
            self.conn.execute(f"SELECT * FROM {result_table}")
            .fetchdf()
            .dropna(subset=[self.outcome_col] + all_cov_cols)
        )
        self.n_obs = len(df)
        ids = (
            df[cluster_col].values
            if cluster_col and cluster_col in df.columns
            else None
        )

        # Extras for mediator equations: exclude group means of mediators.
        med_mean_prefixes  = {f"avg_{m}_fe" for m in self.mediators_sql}
        extras_for_med  = [
            x for x in self._extra_regressors
            if not any(x.startswith(p) for p in med_mean_prefixes)
        ]
        extras_for_out  = self._extra_regressors  # all extras

        # ── mediator equations ────────────────────────────────────────
        for med in self.mediators_sql:
            regressors  = self.exposures_sql + self.controls_sql + extras_for_med
            valid_cols  = [c for c in regressors if c in df.columns]
            exp_idx     = list(range(len(self.exposures_sql)))
            X           = self._build_X(df, valid_cols, intercept=True)
            y           = df[med].values.reshape(-1, 1)
            spec        = self._make_eq_spec(
                name=med, dependent=med,
                regressors=valid_cols, has_intercept=True,
                exposure_indices=exp_idx, mediator_indices=[],
                fe_correction=0, n_fe=0,
            )
            self._eq_data[med] = dict(X=X, y=y, cluster_ids=ids, eq_spec=spec)

        # ── outcome equation ──────────────────────────────────────────
        regressors_out  = (
            self.exposures_sql + self.mediators_sql
            + self.controls_sql + extras_for_out
        )
        valid_cols_out  = [c for c in regressors_out if c in df.columns]
        exp_idx_out     = list(range(len(self.exposures_sql)))
        med_idx_out     = list(range(
            len(self.exposures_sql),
            len(self.exposures_sql) + len(self.mediators_sql),
        ))
        X_out  = self._build_X(df, valid_cols_out, intercept=True)
        y_out  = df[self.outcome_col].values.reshape(-1, 1)
        spec_out = self._make_eq_spec(
            name="__outcome__", dependent=self.outcome_col,
            regressors=valid_cols_out, has_intercept=True,
            exposure_indices=exp_idx_out, mediator_indices=med_idx_out,
            fe_correction=0, n_fe=0,
        )
        self._eq_data["__outcome__"] = dict(
            X=X_out, y=y_out, cluster_ids=ids, eq_spec=spec_out
        )

        logger.debug(
            f"_stage_mundlak: n_obs={self.n_obs}, "
            f"extra_regressors={self._extra_regressors}"
        )

    # ------------------------------------------------------------------
    # Effect decomposition (delta method)
    # ------------------------------------------------------------------

    def _compute_effects(
        self,
        med_results: Dict[str, RegressionResults],
        outcome_result: RegressionResults,
    ) -> MediationEffects:
        """Compute direct / indirect / total indirect / total effects.

        Uses the delta method with a block-diagonal assumption across separate
        equations (i.e. zero covariance between mediator-equation coefficients
        and outcome-equation coefficients).  Within the outcome equation the
        full vcov matrix is used, capturing covariance between direct-effect
        and path-B coefficients.

        Parameters
        ----------
        med_results
            Fitted :class:`RegressionResults` per mediator equation.
        outcome_result
            Fitted :class:`RegressionResults` for the outcome equation.

        Returns
        -------
        MediationEffects
        """
        ne = len(self.exposures_sql)
        nm = len(self.mediators_sql)

        # Outcome equation layout
        out_spec: EquationSpec = self._eq_data["__outcome__"]["eq_spec"]
        out_coefs = outcome_result.coefficients.flatten()
        out_vcov  = outcome_result.vcov         # (k_out, k_out) or None
        out_off   = out_spec.intercept_offset()

        # Exposure indices in outcome equation (absolute, incl. intercept offset)
        de_idx = [out_off + j for j in out_spec.exposure_indices]
        # Mediator (path-B) indices in outcome equation
        b_idx  = [out_off + k for k in out_spec.mediator_indices]

        # ── A matrix and its variance ─────────────────────────────────
        # A[j, k] = effect of exposure j on mediator k
        A     = np.zeros((ne, nm))
        A_var = np.zeros((ne, nm))   # Var(A[j,k]) diagonal only

        for col_k, med in enumerate(self.mediators_sql):
            if med not in med_results:
                continue
            res_k   = med_results[med]
            coefs_k = res_k.coefficients.flatten()
            vcov_k  = res_k.vcov
            med_spec: EquationSpec = self._eq_data[med]["eq_spec"]
            med_off  = med_spec.intercept_offset()

            for j_idx, j in enumerate(med_spec.exposure_indices):
                param_idx = med_off + j
                if param_idx < len(coefs_k):
                    A[j_idx, col_k] = coefs_k[param_idx]
                if vcov_k is not None and param_idx < vcov_k.shape[0]:
                    A_var[j_idx, col_k] = vcov_k[param_idx, param_idx]

        # ── B vector ─────────────────────────────────────────────────
        B = np.array([
            out_coefs[b_idx[k]] if k < len(b_idx) and b_idx[k] < len(out_coefs) else 0.0
            for k in range(nm)
        ])

        # ── direct effects (C') ───────────────────────────────────────
        DE = np.array([
            out_coefs[de_idx[j]] if j < len(de_idx) and de_idx[j] < len(out_coefs) else 0.0
            for j in range(ne)
        ])

        # ── specific indirect effects IE[j, k] = A[j,k] * B[k] ──────
        IE = A * B[np.newaxis, :]   # (ne, nm)

        # delta-method variance for IE[j,k]
        IE_var = np.zeros((ne, nm))
        if out_vcov is not None:
            for j in range(ne):
                for k in range(nm):
                    var_Ajk = A_var[j, k]
                    b_i     = b_idx[k] if k < len(b_idx) else -1
                    var_Bk  = (
                        float(out_vcov[b_i, b_i])
                        if b_i >= 0 and b_i < out_vcov.shape[0]
                        else 0.0
                    )
                    IE_var[j, k] = B[k] ** 2 * var_Ajk + A[j, k] ** 2 * var_Bk
        IE_se = np.sqrt(np.maximum(IE_var, 0.0))

        # ── total indirect effect TIE[j] = sum_k IE[j,k] ─────────────
        TIE = IE.sum(axis=1)  # (ne,)

        TIE_var = np.zeros(ne)
        if out_vcov is not None:
            for j in range(ne):
                # Contribution from mediator-equation vcovs (block-diagonal)
                med_part = float(sum(
                    B[k] ** 2 * A_var[j, k] for k in range(nm)
                ))
                # Contribution from outcome-equation vcov (B cross-terms)
                g = np.zeros(len(out_coefs))
                for k in range(nm):
                    b_i = b_idx[k] if k < len(b_idx) else -1
                    if 0 <= b_i < len(g):
                        g[b_i] = A[j, k]
                out_part = float(g @ out_vcov @ g)
                TIE_var[j] = med_part + out_part
        TIE_se = np.sqrt(np.maximum(TIE_var, 0.0))

        # ── total effects TE[j] = DE[j] + TIE[j] ────────────────────
        TE = DE + TIE

        TE_var = np.zeros(ne)
        if out_vcov is not None:
            for j in range(ne):
                med_part = float(sum(
                    B[k] ** 2 * A_var[j, k] for k in range(nm)
                ))
                # Gradient w.r.t. outcome coefs: 1 at DE_idx[j], A[j,k] at b_idx[k]
                g_te = np.zeros(len(out_coefs))
                de_i = de_idx[j] if j < len(de_idx) else -1
                if 0 <= de_i < len(g_te):
                    g_te[de_i] = 1.0
                for k in range(nm):
                    b_i = b_idx[k] if k < len(b_idx) else -1
                    if 0 <= b_i < len(g_te):
                        g_te[b_i] = A[j, k]
                out_part = float(g_te @ out_vcov @ g_te)
                TE_var[j] = med_part + out_part
        TE_se = np.sqrt(np.maximum(TE_var, 0.0))

        # ── SE for direct effects (from outcome vcov diagonal) ────────
        if out_vcov is not None:
            de_se = np.array([
                np.sqrt(max(0.0, float(out_vcov[de_idx[j], de_idx[j]])))
                if j < len(de_idx) and de_idx[j] < out_vcov.shape[0]
                else np.nan
                for j in range(ne)
            ])
        else:
            de_se = None

        return MediationEffects(
            exposure_names=list(self.exposures_sql),
            mediator_names=list(self.mediators_sql),
            direct=DE,
            direct_se=de_se,
            indirect=IE,
            indirect_se=IE_se if out_vcov is not None else None,
            total_indirect=TIE,
            total_indirect_se=TIE_se if out_vcov is not None else None,
            total=TE,
            total_se=TE_se if out_vcov is not None else None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_X(
        df: pd.DataFrame,
        regressors: List[str],
        intercept: bool,
    ) -> np.ndarray:
        """Build a design matrix from *df* columns.

        Parameters
        ----------
        df : DataFrame
            Source data.
        regressors : List[str]
            Column names (already validated to exist in *df*).
        intercept : bool
            Prepend a column of ones when ``True``.

        Returns
        -------
        np.ndarray  shape (n, k)
        """
        n = len(df)
        if regressors:
            cols = np.column_stack([df[c].values for c in regressors])
        else:
            cols = np.empty((n, 0))
        return np.c_[np.ones(n), cols] if intercept else cols

    @staticmethod
    def _make_eq_spec(
        name: str,
        dependent: str,
        regressors: List[str],
        has_intercept: bool,
        exposure_indices: List[int],
        mediator_indices: List[int],
        fe_correction: int,
        n_fe: int,
    ) -> EquationSpec:
        """Factory for :class:`~duckreg.utils.mediation_spec.EquationSpec`."""
        return EquationSpec(
            name=name,
            dependent=dependent,
            regressors=regressors,
            has_intercept=has_intercept,
            exposure_indices=exposure_indices,
            mediator_indices=mediator_indices,
            fe_correction=fe_correction,
            n_fe=n_fe,
        )

    # ------------------------------------------------------------------
    # Summary / output
    # ------------------------------------------------------------------

    @property
    def results(self):
        """Alias to maintain compatibility with single-equation estimators.

        Returns the outcome-equation :class:`RegressionResults` or ``None``.
        """
        if self.mediation_results is not None:
            return self.mediation_results.outcome_result
        return None

    def summary(self) -> Dict[str, Any]:
        """Return a structured summary dictionary."""
        from .._version import __version__
        from ..core.results import _get_timestamp

        out: Dict[str, Any] = {
            "version_info": {
                "duckreg_version": __version__,
                "computed_at": _get_timestamp(),
            },
            "model_spec": {
                "estimator_type": self.__class__.__name__,
                "outcome": self.outcome_col,
                "exposures": self.exposures_sql,
                "mediators": self.mediators_sql,
                "controls": self.controls_sql,
                "fe_cols": self.fe_cols,
                "fe_method": self.fe_method,
                "cluster_col": self._effective_cluster_col,
            },
            "sample_info": {
                "n_obs": self.n_obs,
                "n_rows_dropped_singletons": self.n_rows_dropped_singletons,
            },
        }

        if self.mediation_results is not None:
            # Equation-level coefficients
            eq_section = {}
            for med, res in self.mediation_results.mediator_results.items():
                eq_section[f"mediator:{med}"] = res.to_dict()
            if self.mediation_results.outcome_result is not None:
                eq_section["outcome"] = self.mediation_results.outcome_result.to_dict()
            out["equations"] = eq_section

            # Effect decomposition
            if self.mediation_results.effects is not None:
                out["effects"] = self.mediation_results.effects.to_tidy_df().to_dict(
                    orient="records"
                )

        return out

    def summary_df(self) -> pd.DataFrame:
        """Equation-level results as a tidy DataFrame."""
        if self.mediation_results is None:
            return pd.DataFrame()
        return self.mediation_results.equations_df()

    def to_tidy_df(self) -> pd.DataFrame:
        """Combined tidy DataFrame: equation results + effect decomposition."""
        if self.mediation_results is None:
            return pd.DataFrame()
        return self.mediation_results.to_tidy_df()

    def print_summary(self, precision: int = 4):
        """Print formatted mediation results to the console."""
        if self.mediation_results is None:
            print("DuckMediation: model not yet fitted.")
            return

        width = 80
        print("=" * width)
        print("MEDIATION ANALYSIS RESULTS")
        print("=" * width)
        spec = self.summary()["model_spec"]
        print(f"Outcome  : {spec['outcome']}")
        print(f"Exposures: {', '.join(spec['exposures'])}")
        print(f"Mediators: {', '.join(spec['mediators']) or '(none)'}")
        if spec['controls']:
            print(f"Controls : {', '.join(spec['controls'])}")
        if spec['fe_cols']:
            print(f"FE       : {', '.join(spec['fe_cols'])} ({spec['fe_method']})")
        if spec['cluster_col']:
            print(f"Cluster  : {spec['cluster_col']}")
        print(f"SE type  : {self.mediation_results.se_type or '?'}")
        n = self.n_obs or 0
        n_drop = self.n_rows_dropped_singletons or 0
        print(f"N        : {n:,}" + (f"  ({n_drop:,} singletons removed)" if n_drop else ""))
        print()

        # ── equation results ─────────────────────────────────────────
        fmt = f"{{:<32}} {{:>{precision+8}}} {{:>{precision+8}}} {{:>10}} {{:>10}} {{:>3}}"
        hdr = fmt.format("Variable", "Estimate", "Std. Error", "t-stat", "p-val", "")

        def _print_eq(title: str, res: RegressionResults):
            print(f"  {title}")
            print("  " + "-" * (width - 2))
            print("  " + hdr)
            print("  " + "-" * (width - 2))
            coefs = res.coefficients.flatten()
            ses   = res.std_errors
            for i, name in enumerate(res.coef_names):
                c = coefs[i]
                if ses is not None:
                    s = ses[i]
                    t = res.t_stats[i] if res.t_stats is not None else np.nan
                    p = res.p_values[i] if res.p_values is not None else np.nan
                    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("." if p < 0.10 else "")))
                    row = fmt.format(name[:31], f"{c:.{precision}f}", f"{s:.{precision}f}", f"{t:.{precision}f}", f"{p:.{precision}f}", sig)
                else:
                    row = fmt.format(name[:31], f"{c:.{precision}f}", "N/A", "N/A", "N/A", "")
                print("  " + row)
            print()

        print("EQUATION RESULTS")
        print("-" * width)
        for med, res in self.mediation_results.mediator_results.items():
            _print_eq(f"Mediator equation: {med}", res)
        if self.mediation_results.outcome_result is not None:
            _print_eq("Outcome equation", self.mediation_results.outcome_result)

        # ── effect decomposition ─────────────────────────────────────
        if self.mediation_results.effects is not None:
            print("MEDIATION EFFECTS")
            print("-" * width)
            eff = self.mediation_results.effects
            efmt = f"{{:<36}} {{:>{precision+8}}} {{:>{precision+8}}} {{:>10}} {{:>10}} {{:>3}}"
            ehdr = efmt.format("Effect", "Estimate", "Std. Error", "z-stat", "p-val", "")
            print("  " + ehdr)
            print("  " + "-" * (width - 2))
            tidy = eff.to_tidy_df()
            for _, row in tidy.iterrows():
                label = f"[{row['effect_type']}] {row['exposure']}"
                if row.get("mediator"):
                    label += f" → {row['mediator']}"
                se_v = row["std_error"]
                z_v  = row["z_stat"]
                p_v  = row["p_value"]
                sig  = "" if not np.isfinite(p_v) else ("***" if p_v < 0.001 else ("**" if p_v < 0.01 else ("*" if p_v < 0.05 else ("." if p_v < 0.10 else ""))))
                print("  " + efmt.format(
                    label[:35],
                    f"{row['estimate']:.{precision}f}",
                    f"{se_v:.{precision}f}" if np.isfinite(se_v) else "N/A",
                    f"{z_v:.{precision}f}"  if np.isfinite(z_v) else "N/A",
                    f"{p_v:.{precision}f}"  if np.isfinite(p_v) else "N/A",
                    sig,
                ))
            print()
        print("=" * width)
