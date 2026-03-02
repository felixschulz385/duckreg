"""Unified fixed-effects estimator backed by pluggable transformers.

``DuckFE`` is a single entry point for FE regression that delegates the
data transformation step to a :class:`~duckreg.core.transformers.FETransformer`
subclass.  The estimator itself only handles formula parsing, compression,
coefficient naming, and vcov dispatch — concerns that are independent of the
chosen FE method.

Supported methods
-----------------
``'iterative_demean'``
    Method of Alternating Projections (MAP).  Absorbs FEs by iterative
    demeaning.  Correct for any panel structure including unbalanced multi-way
    FE.  No intercept; robust DOF correction via :attr:`df_correction`.

``'mundlak'``
    Mundlak device.  Includes within-group means of covariates as explicit
    regressors (and binary dummies for low-cardinality FEs).  Uses the
    Wooldridge correction for unbalanced panels.  Keeps data in levels;
    estimates both within- and between-group effects.  Not recommended for
    unbalanced panels.

``'auto_fe'``
    Automatic routing: each FE dimension is independently routed based on a
    cheap cardinality estimate on a random row sample.  Very low and very
    high cardinality dimensions use Mundlak; intermediate cardinality
    dimensions use MAP.  Hybrid cases apply Mundlak augmentation first, then
    MAP demeaning on the augmented data.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .DuckLinearModel import DuckLinearModel
from ..core.transformers import AutoFETransformer, IterativeDemeanTransformer, MundlakTransformer

logger = logging.getLogger(__name__)


class DuckFE(DuckLinearModel):
    """Unified fixed-effects OLS estimator.

    Parameters
    ----------
    *args
        Positional arguments forwarded to :class:`DuckLinearModel`.
    method : {'iterative_demean', 'mundlak', 'auto_fe'}
        FE absorption strategy.  Default is ``'iterative_demean'``.
    max_iterations : int
        *(Iterative demean only)* Maximum MAP iterations.
    tolerance : float
        *(Iterative demean only)* Convergence criterion
        (max absolute residual change).
    fe_types : Dict[str, str], optional
        *(Mundlak only)* Override automatic FE classification:
        ``{fe_col: 'fixed' | 'asymptotic'}``.
    cardinality_threshold : int
        *(Mundlak only)* Max cardinality for automatic *fixed* classification.
    singleton_threshold : float
        *(Mundlak only)* Max singleton share for *asymptotic* classification.
    max_fixed_fe_levels : int
        *(Mundlak only)* Level limit for fixed FEs (prevents column explosion).
    auto_fe_kwargs : dict, optional
        *(auto_fe only)* Extra keyword arguments forwarded to
        :class:`~duckreg.core.transformers.AutoFETransformer` (e.g.
        ``cardinality_ratio``, ``sample_size``, ``mundlak_kwargs``,
        ``map_kwargs``).
    **kwargs
        Keyword arguments forwarded to :class:`DuckLinearModel`.  Includes
        ``round_strata`` (int or ``None``) from :func:`duckreg`: when set,
        demeaned covariate columns are rounded to that many decimal places
        before grouping, increasing strata deduplication at the cost of a
        small approximation error.
    """

    _CLUSTER_ALIAS = "__cluster__"
    _STAGING_TABLE = "_duckfe_staging"

    def __init__(
        self,
        *args,
        method: str = "iterative_demean",
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        fe_types: Optional[Dict[str, str]] = None,
        cardinality_threshold: int = 50,
        singleton_threshold: float = 0.1,
        max_fixed_fe_levels: int = 100,
        auto_fe_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if method == "auto_fe":
            raise NotImplementedError(
                "'auto_fe' is experimental and has been temporarily disabled. "
                "Use method='iterative_demean' instead."
            )
        if method not in ("iterative_demean", "mundlak"):
            raise ValueError(
                f"Unknown method {method!r}. "
                "Choose 'iterative_demean' or 'mundlak'."
            )
        if not self.fe_cols:
            raise ValueError("DuckFE requires at least one fixed effect.")

        self.method = method
        if self.method == "mundlak":
            logger.warning(
                "Mundlak regression is not recommended for unbalanced panels; "
                "consider using 'demean' instead."
            )
        # auto_fe may internally use Mundlak for high-cardinality FEs, but
        # the routing is deliberate — suppress the blanket warning.
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.fe_types = fe_types or {}
        self.cardinality_threshold = cardinality_threshold
        self.singleton_threshold = singleton_threshold
        self.max_fixed_fe_levels = max_fixed_fe_levels
        self.auto_fe_kwargs: Dict[str, Any] = auto_fe_kwargs or {}

        self._transformer: Optional[object] = None
        self._result_table: Optional[str] = None

    @property
    def fe_metadata(self) -> dict:
        """FE classification metadata from the transformer (mundlak only)."""
        if self._transformer is not None:
            return getattr(self._transformer, 'fe_metadata', {})
        return {}

    @property
    def _dummy_mean_cols(self) -> list:
        """Dummy-mean columns added by the Mundlak transformer (mundlak only)."""
        if self._transformer is not None:
            return getattr(self._transformer, '_dummy_mean_cols', [])
        return []

    # -------------------------------------------------------------------------
    # Transformer factory helpers
    # -------------------------------------------------------------------------

    def _resolve_fe_sql_names(self) -> List[str]:
        """Resolve FE logical names to SQL-safe names via formula metadata."""
        result = []
        for fe_name in self.fe_cols:
            fe_var = self.formula.get_fe_by_name(fe_name)
            mfe = self.formula.get_merged_fe_by_name(fe_name)
            result.append(
                fe_var.sql_name if fe_var
                else (mfe.sql_name if mfe else fe_name)
            )
        return result

    def _drop_staging_if_exists(self) -> None:
        """Drop _STAGING_TABLE unconditionally, regardless of whether it is a
        TABLE or a VIEW.  DuckDB's ``DROP VIEW IF EXISTS`` raises an error when
        the name belongs to a TABLE (and vice versa), so we must look up the
        actual object type in the catalog first."""
        row = self.conn.execute(
            "SELECT table_type FROM information_schema.tables "
            f"WHERE table_name = '{self._STAGING_TABLE}'"
        ).fetchone()
        if row is None:
            return
        drop_kw = "VIEW" if row[0] == "VIEW" else "TABLE"
        self.conn.execute(f"DROP {drop_kw} {self._STAGING_TABLE}")

    # _effective_cluster_col is inherited from DuckLinearModel

    def _make_transformer(
        self,
        fe_sql_names: List[str],
        cov_sql_names: List[str],
        cluster_col: Optional[str],
        source_table: str,
    ):
        """Instantiate the appropriate :class:`FETransformer`."""
        common = dict(
            conn=self.conn,
            table_name=source_table,
            fe_cols=fe_sql_names,
            cluster_col=cluster_col,
            remove_singletons=self.remove_singletons,
        )
        if self.method == "iterative_demean":
            return IterativeDemeanTransformer(
                **common,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
            )
        elif self.method == "auto_fe":
            return AutoFETransformer(
                **common,
                covariate_cols=cov_sql_names,
                **self.auto_fe_kwargs,
            )
        else:  # mundlak
            return MundlakTransformer(
                **common,
                covariate_cols=cov_sql_names,
                fe_types=self.fe_types,
                cardinality_threshold=self.cardinality_threshold,
                singleton_threshold=self.singleton_threshold,
                max_fixed_fe_levels=self.max_fixed_fe_levels,
            )

    # -------------------------------------------------------------------------
    # Estimation pipeline
    # -------------------------------------------------------------------------

    def prepare_data(self):
        """Build the transformed data table via the selected transformer.

        For ``'mundlak'``, a staging table is first created using the formula's
        SQL helpers (handles boolean casting, interactions, cluster aliasing)
        before handing off to :class:`MundlakTransformer`.

        For ``'iterative_demean'``, the transformer operates directly on the
        source table with the WHERE clause.
        """
        # Upgrade merged FE expressions to the numeric (BIGINT) merge path when
        # all component columns are integer types and no overflow risk exists.
        from ..utils.formula_parser import FormulaParser
        self.formula = FormulaParser.resolve_numeric_merge(
            self.formula, self.conn, self.table_name
        )

        boolean_cols = self._get_boolean_columns()
        unit_col = self._get_unit_col()
        fe_sql_names = self._resolve_fe_sql_names()

        outcome_sql = [v.sql_name for v in self.formula.outcomes]
        cov_sql = [
            v.sql_name for v in self.formula.covariates
            if not v.is_intercept()
        ]

        if self.method == "mundlak":
            # Build a staging table via formula helpers so that boolean
            # casting, interaction columns, and the cluster alias are all
            # handled correctly before the transformer sees the data.
            eff_cluster = self._effective_cluster_col
            cluster_alias = self._CLUSTER_ALIAS if eff_cluster else None

            select_parts = [
                self.formula.get_fe_select_sql(boolean_cols),
                self.formula.get_outcomes_select_sql(
                    unit_col, "year", boolean_cols
                ),
                self.formula.get_covariates_select_sql(
                    unit_col, "year", boolean_cols, include_interactions=True
                ),
            ]
            # Pass eff_cluster as fallback so clusters specified via
            # se_method={'CRV1': 'var'} (not in formula) are also included.
            cluster_sql = self.formula.get_cluster_select_sql(
                boolean_cols, self._CLUSTER_ALIAS, eff_cluster
            )
            if cluster_sql:
                select_parts.append(cluster_sql)

            # Drop whatever type of object currently holds this name;
            # _drop_staging_if_exists handles the TABLE vs VIEW type lookup.
            self._drop_staging_if_exists()
            # Use a VIEW (not TABLE) so MundlakTransformer immediately materialises
            # it into design_matrix — the ~320 GB staging copy is never stored.
            self.conn.execute(f"""
            CREATE OR REPLACE VIEW {self._STAGING_TABLE} AS
            SELECT {', '.join(p for p in select_parts if p)}
            FROM {self.table_name}
            {self._build_where_clause(self.subset)}
            """)

            self._transformer = self._make_transformer(
                fe_sql_names=fe_sql_names,
                cov_sql_names=cov_sql,
                cluster_col=cluster_alias,
                source_table=self._STAGING_TABLE,
            )
            # Staging view has all columns; transformer reads from it with no
            # extra WHERE clause (filtering applied via view definition).
            variables = outcome_sql + cov_sql
            self._result_table = self._transformer.fit_transform(
                variables, where_clause=""
            )
            # design_matrix has been materialised; drop the staging view now.
            self._drop_staging_if_exists()

        else:  # iterative_demean or auto_fe
            # Create a staging view whenever any computed expression is needed:
            # merged FEs (e.g. country*year) or transformed variables
            # (e.g. log(viirs_annual + 0.01)).  The view uses SELECT * so all
            # original columns remain accessible, plus extra aliases for each
            # computed expression so the transformer can reference them by
            # their sql_name.
            from ..utils.formula_parser import TransformType

            extra_select_exprs = []

            # Merged FE expressions (e.g. country_year from country*year)
            for mfe in self.formula.merged_fes:
                extra_select_exprs.append(mfe.get_select_sql())

            # Transformed outcome/covariate expressions
            # (only needed when a transform is applied; plain columns already
            # exist in the source table under their original name)
            for var in self.formula.outcomes:
                if var.transform != TransformType.NONE:
                    extra_select_exprs.append(
                        var.get_select_sql(unit_col, "year", boolean_cols)
                    )
            for var in self.formula.covariates:
                if not var.is_intercept() and var.transform != TransformType.NONE:
                    extra_select_exprs.append(
                        var.get_select_sql(unit_col, "year", boolean_cols)
                    )

            if extra_select_exprs:
                # Drop whatever type of object currently holds this name;
                # see _drop_staging_if_exists for the rationale.
                self._drop_staging_if_exists()
                self.conn.execute(f"""
                CREATE OR REPLACE VIEW {self._STAGING_TABLE} AS
                SELECT *, {', '.join(extra_select_exprs)}
                FROM {self.table_name}
                """)
                source_table = self._STAGING_TABLE
            else:
                source_table = self.table_name

            self._transformer = self._make_transformer(
                fe_sql_names=fe_sql_names,
                cov_sql_names=cov_sql,
                cluster_col=self._effective_cluster_col,
                source_table=source_table,
            )
            variables = outcome_sql + cov_sql
            self._result_table = self._transformer.fit_transform(
                variables,
                where_clause=self._build_where_clause(self.subset),
            )
            # demeaned_data has been materialised; drop the staging view now.
            self._drop_staging_if_exists()

        self.n_rows_dropped_singletons = getattr(
            self._transformer, "n_rows_dropped_singletons", 0
        )
        logger.debug(
            f"DuckFE.prepare_data complete: method={self.method}, "
            f"result_table={self._result_table}, "
            f"n_obs={self._transformer.n_obs}"
        )

    def compress_data(self):
        """Create a compressed view for the estimation step.

        Both methods group by unique covariate strata and aggregate outcomes
        as weighted sums.  For **mundlak** this typically yields large
        compression ratios on repeated covariate patterns.  For **iterative
        demean**, demeaned floating-point variables rarely produce exact
        duplicate rows; set ``round_strata`` to round covariate columns
        before grouping to improve deduplication at the cost of a small
        approximation.
        """
        outcome_sql = [v.sql_name for v in self.formula.outcomes]
        cov_sql = [
            v.sql_name for v in self.formula.covariates
            if not v.is_intercept()
        ]
        all_variables = outcome_sql + cov_sql
        extra = self._transformer.extra_regressors

        _uses_map_path = not self._is_pure_mundlak()

        if _uses_map_path:
            # Group the demeaned data by unique covariate patterns (strata).
            # transform_query aliases resid_v → v, so the subquery exposes
            # original column names that the GROUP BY can reference directly.
            # round_strata (inherited from DuckEstimator) is applied here to
            # improve deduplication of demeaned continuous covariates.
            from ..core.sql_builders import build_round_expr
            # Include extra_regressors in transform_query so that Mundlak-added
            # columns (hybrid auto_fe) appear as real columns inside the _t
            # subquery.  Without this, they are only referenced as outer-SELECT
            # aliases and DuckDB rejects ROUND(alias, n) in GROUP BY.
            transformed_select = self._transformer.transform_query(all_variables + extra)
            # Use the effective cluster col (formula or vcov_spec fallback) so
            # CRV1 requested via se_method={'CRV1': 'unit'} is preserved.
            eff_cluster = self._effective_cluster_col
            cluster_part_select = (
                f", {eff_cluster}" if eff_cluster else ""
            )
            rhs_cols = cov_sql + extra

            # Expose for tests / callers (mirrors mundlak convention)
            self.strata_cols = list(rhs_cols)
            self._rhs_cols = list(rhs_cols)

            # Build SELECT and GROUP BY with optional rounding for covariates.
            select_parts, group_parts = [], []
            for col in rhs_cols:
                sel, grp = build_round_expr(col, col, self.round_strata)
                select_parts.append(sel)
                group_parts.append(grp)
            if eff_cluster:
                select_parts.append(eff_cluster)
                group_parts.append(eff_cluster)

            outcome_aggs = []
            for v in self.formula.outcomes:
                outcome_aggs.append(f"SUM({v.sql_name}) AS sum_{v.sql_name}")
                outcome_aggs.append(f"SUM({v.sql_name} * {v.sql_name}) AS sum_{v.sql_name}_sq")

            group_select = ", ".join(select_parts) if select_parts else "1"
            group_by_clause = f"GROUP BY {', '.join(group_parts)}" if group_parts else ""
            self.agg_query = f"""
            SELECT {group_select}, COUNT(*) AS count,
                   {', '.join(outcome_aggs)}
            FROM (
                SELECT {transformed_select}{cluster_part_select}
                FROM {self._result_table}
            ) _t
            {group_by_clause}
            """

        else:  # mundlak or auto_fe pure-mundlak
            # Mundlak: group by all RHS columns + cluster col; aggregate outcomes.
            # For 'mundlak', the cluster column was aliased to __cluster__ during
            # staging; for 'auto_fe' pure-Mundlak we use the original column name.
            _eff = self._effective_cluster_col
            cluster_alias = (
                self._CLUSTER_ALIAS if self.method == "mundlak" and _eff
                else _eff if self.method == "auto_fe" and _eff
                else None
            )
            rhs_cols = cov_sql + extra
            group_cols = rhs_cols + ([cluster_alias] if cluster_alias else [])
            group_by = ", ".join(group_cols)

            # Expose for tests / callers
            self.strata_cols = list(rhs_cols)
            self._rhs_cols   = list(rhs_cols)

            outcome_aggs = []
            for v in self.formula.outcomes:
                outcome_aggs.append(f"SUM({v.sql_name}) AS sum_{v.sql_name}")
                outcome_aggs.append(f"SUM({v.sql_name} * {v.sql_name}) AS sum_{v.sql_name}_sq")

            self.agg_query = f"""
            SELECT {group_by}, COUNT(*) AS count,
                   {', '.join(outcome_aggs)}
            FROM {self._result_table}
            GROUP BY {group_by}
            """

        self._create_compressed_view()
        # Invalidate any cached sum_sq values from a previous run.
        self._sum_sq_cache: Dict[str, float] = {}
        logger.debug(
            f"Compressed view: {self.n_compressed_rows} strata, "
            f"{self.n_obs} observations"
        )

    def _compute_sum_sq(self, outcome_sql_name: str) -> float:
        """Return ``SUM(y * y)`` for *outcome_sql_name*, computed on demand.

        The result is computed from the compressed view and cached on
        ``self._sum_sq_cache`` so repeated calls within the same fit are free.

        The compressed view now always includes an exact ``sum_{v}_sq`` column
        (added during :meth:`compress_data`) so this method reads it directly::

            SELECT SUM(sum_{v}_sq) FROM _compressed_view

        A fallback to the approximation ``SUM(sum_v * sum_v / count)`` is
        kept for robustness (e.g. old pickled models without the column).

        Parameters
        ----------
        outcome_sql_name : str
            The SQL-safe name of the outcome column (e.g. ``"log_gdp"``).

        Returns
        -------
        float
            ``SUM(y²)`` — exact when the ``sum_{v}_sq`` column is present.
        """
        cache = getattr(self, "_sum_sq_cache", None)
        if cache is None:
            self._sum_sq_cache: Dict[str, float] = {}
            cache = self._sum_sq_cache
        if outcome_sql_name in cache:
            return cache[outcome_sql_name]
        # Prefer the exact pre-computed sum_sq column added during compression.
        sum_sq_col = f"sum_{outcome_sql_name}_sq"
        sum_col = f"sum_{outcome_sql_name}"
        try:
            val = self.conn.execute(
                f"SELECT SUM({sum_sq_col}) FROM {self._COMPRESSED_VIEW}"
            ).fetchone()[0]
        except Exception:
            # Fallback: approximate formula (exact only for singleton strata).
            val = self.conn.execute(
                f"SELECT SUM({sum_col} * {sum_col} / count) "
                f"FROM {self._COMPRESSED_VIEW}"
            ).fetchone()[0]
        result = float(val) if val is not None else 0.0
        cache[outcome_sql_name] = result
        return result

    def collect_data(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract *y*, *X*, and weights from the compressed data frame.

        Both methods produce a compressed view whose outcomes are stored as
        ``sum_{v}`` columns.  This method divides by ``count`` to recover
        per-stratum means used in WLS, with ``count`` as the weight vector.
        For ``'mundlak'`` an intercept column is prepended to *X*.
        """
        outcome_sql = [v.sql_name for v in self.formula.outcomes]
        cov_sql = [
            v.sql_name for v in self.formula.covariates
            if not v.is_intercept()
        ]
        extra = self._transformer.extra_regressors

        _uses_map_path = not self._is_pure_mundlak()

        rhs_cols = cov_sql + extra
        y_cols = [f"sum_{v}" for v in outcome_sql]
        n = data["count"].values
        y = data[y_cols].values / n.reshape(-1, 1)
        if not _uses_map_path:
            # pure-Mundlak: data is in levels, always needs intercept
            X = np.c_[np.ones(len(data)), data[rhs_cols].values]
        elif self._transformer.has_intercept:
            # Future path: any MAP-based transformer that explicitly keeps
            # an intercept would land here.  Currently unused.
            X = np.c_[np.ones(len(data)), data[rhs_cols].values]
        else:
            # pure-MAP, iterative_demean, or hybrid auto_fe:
            # MAP demeaning absorbs the constant — no intercept column.
            # For hybrid, rhs_cols = covariates + Mundlak mean cols (demeaned).
            X = data[rhs_cols].values

        y = y.reshape(-1, 1) if y.ndim == 1 else y
        return y, X, n

    # -------------------------------------------------------------------------
    # DuckLinearModel hook overrides
    # -------------------------------------------------------------------------

    def _is_pure_mundlak(self) -> bool:
        """True only when no MAP transformer is involved.

        - method='mundlak'                         → True
        - method='auto_fe', all FEs → Mundlak      → True
        - method='auto_fe', hybrid or pure-MAP     → False
        - method='iterative_demean'                → False
        """
        if self.method == "mundlak":
            return True
        if self.method == "auto_fe":
            return bool(self._transformer._mundlak_fe_cols) and \
                   not bool(self._transformer._map_fe_cols)
        return False

    def _is_demean_coef_path(self) -> bool:
        """True for pure-MAP and iterative_demean only.

        Hybrid auto_fe uses the Mundlak coefficient-name path because it
        includes an intercept and Mundlak mean column names.
        """
        if self.method == "iterative_demean":
            return True
        if self.method == "auto_fe":
            return (
                bool(self._transformer._map_fe_cols)
                and not bool(self._transformer._mundlak_fe_cols)
            )
        return False  # mundlak

    def _needs_intercept_for_duckdb(self) -> bool:
        return self._transformer.has_intercept

    def _get_vcov_fe_params(self) -> Tuple[int, int, int, int]:
        """Return (k_fe, n_fe, k_fe_nested, n_fe_fully_nested) for VcovContext."""
        uses_map = not self._is_pure_mundlak()
        if not uses_map:
            return 0, 0, 0, 0

        # Be robust to attribute naming (vcovspec vs vcov_spec, isclustered vs is_clustered)
        vc = getattr(self, "vcovspec", None)
        if vc is None:
            vc = getattr(self, "vcov_spec", None)

        is_clustered = False
        if vc is not None:
            is_clustered = bool(getattr(vc, "isclustered", getattr(vc, "is_clustered", False)))

        active = getattr(self._transformer, "_active_transformer", self._transformer)
        n_fe = len(active.fe_cols) if active is not None else len(self._transformer.fe_cols)

        # Key fix: always return k_fe and n_fe; only gate nesting on clustering
        k_fe_nested = 0
        n_fe_fully_nested = 0
        if is_clustered:
            k_fe_nested, n_fe_fully_nested = self._compute_fe_nesting()

        return (
            self._transformer.df_correction,  # k_fe
            n_fe,
            k_fe_nested,
            n_fe_fully_nested,
        )


    def _compute_fe_nesting(self) -> Tuple[int, int]:
        """Detect nested FE structure for the nonnested SSC path.

        A FE dimension B is fully nested in A when every B-group is contained
        within exactly one A-group.  This is detected by checking whether
        COUNT(DISTINCT a) == 1 for every level of b (i.e. each b value maps
        to a single a value).

        Returns
        -------
        k_fe_nested : int
            Total number of FE levels belonging to fully-nested dimensions
            (sum of distinct levels for each nested dimension).
        n_fe_fully_nested : int
            Number of FE dimensions that are fully nested within another.
        """
        # Return cached result (computed during fit, before connection closes)
        if hasattr(self, '_fe_nesting_cache'):
            return self._fe_nesting_cache

        if self._transformer is None or self._result_table is None:
            return 0, 0
        # For auto_fe, use the active (MAP) transformer's fe_cols, since those
        # are the dimensions whose group structure lives in demeaned_data.
        active = getattr(self._transformer, "_active_transformer", self._transformer)
        effective_transformer = active if active is not None else self._transformer

        fe_cols = effective_transformer.fe_cols
        result_table = self._result_table
        k_fe_nested = 0
        n_fe_fully_nested = 0

        # --- (A) FE-to-FE nesting: B nested in A (B levels map to one A level) ---
        if len(fe_cols) >= 2:
            for i, fe_b in enumerate(fe_cols):
                is_nested = False
                n_levels_b = 0
                for j, fe_a in enumerate(fe_cols):
                    if i == j:
                        continue
                    # B is nested in A iff every B-level maps to exactly one A-level
                    row = self.conn.execute(f"""
                        SELECT MAX(cnt) AS max_a_per_b,
                               COUNT(DISTINCT {fe_b}) AS n_levels_b
                        FROM (
                            SELECT {fe_b}, COUNT(DISTINCT {fe_a}) AS cnt
                            FROM {result_table}
                            GROUP BY {fe_b}
                        ) t
                    """).fetchone()
                    if row is not None and row[0] == 1:
                        is_nested = True
                        n_levels_b = int(row[1]) if row[1] is not None else 0
                        break
                if is_nested:
                    n_fe_fully_nested += 1
                    k_fe_nested += n_levels_b

        # --- (B) FE-to-cluster nesting: B nested in the CRV cluster column ---
        # pyfixest's kfixef='nonnested' also subtracts FE dimensions that are
        # fully nested *within the cluster*.  The unit FE is trivially nested
        # in a CRV1 cluster-by-unit because every unit level maps to exactly
        # one cluster level (they are the same identifier).  Only check this
        # when a cluster column is available and not already counted above.
        cluster_col = self._effective_cluster_col
        if cluster_col:
            already_nested = set()
            # Rebuild the set of FE dims detected as FE-to-FE nested
            if len(fe_cols) >= 2:
                for i, fe_b in enumerate(fe_cols):
                    for j, fe_a in enumerate(fe_cols):
                        if i == j:
                            continue
                        row = self.conn.execute(f"""
                            SELECT MAX(cnt) FROM (
                                SELECT {fe_b}, COUNT(DISTINCT {fe_a}) AS cnt
                                FROM {result_table} GROUP BY {fe_b}
                            ) t
                        """).fetchone()
                        if row is not None and row[0] == 1:
                            already_nested.add(i)
                            break

            for i, fe_b in enumerate(fe_cols):
                if i in already_nested:
                    continue
                row = self.conn.execute(f"""
                    SELECT MAX(cnt) AS max_c_per_b,
                           COUNT(DISTINCT {fe_b}) AS n_levels_b
                    FROM (
                        SELECT {fe_b}, COUNT(DISTINCT {cluster_col}) AS cnt
                        FROM {result_table}
                        GROUP BY {fe_b}
                    ) t
                """).fetchone()
                if row is not None and row[0] == 1:
                    n_fe_fully_nested += 1
                    k_fe_nested += int(row[1]) if row[1] is not None else 0

        result = (k_fe_nested, n_fe_fully_nested)
        self._fe_nesting_cache = result
        return result

    def _get_y_col_for_duckdb(self) -> str:
        main_outcome = self.formula.outcomes[0].sql_name
        return f"sum_{main_outcome}"  # both methods now store aggregated outcomes

    def _get_x_cols_for_duckdb(self) -> List[str]:
        cov_sql = [
            v.sql_name for v in self.formula.covariates
            if not v.is_intercept()
        ]
        return cov_sql + self._transformer.extra_regressors

    def _get_cluster_col_for_vcov(self) -> Optional[str]:
        if self.method == "mundlak":
            return self._CLUSTER_ALIAS if self._effective_cluster_col else None
        if self.method in ("iterative_demean", "auto_fe"):
            # Return the cluster col that the transformer was given; it honours
            # both formula-level clusters and vcov_spec.cluster_vars fallback.
            eff = self._effective_cluster_col
            if eff:
                return getattr(self._transformer, "cluster_col", eff)
        return None

    def _get_cluster_data_for_bootstrap(self) -> Tuple[pd.DataFrame, str]:
        if self.method in ("iterative_demean", "auto_fe"):
            # Use the SQL-safe FE name so merged FEs (e.g. country_year for
            # country*year) resolve correctly against the demeaned_data table.
            fe_sql_names = self._resolve_fe_sql_names()
            sampling_col = self.cluster_col or (fe_sql_names[0] if fe_sql_names else self.fe_cols[0])
            cluster_df = self.conn.execute(
                f"SELECT DISTINCT {sampling_col} FROM {self._result_table}"
            ).fetchdf()
            return cluster_df, sampling_col
        else:
            self._ensure_data_fetched()
            return self.df_compressed, self._CLUSTER_ALIAS

    def _build_coef_names_from_formula(self) -> List[str]:
        from ..utils.name_utils import build_coef_name_lists

        if self._is_demean_coef_path():
            display_names, sql_names = build_coef_name_lists(
                formula=self.formula,
                fe_method="demean",
                include_intercept=False,
                fe_cols=None,
                is_iv=False,
            )
        else:  # mundlak, pure-mundlak auto_fe, AND hybrid auto_fe
            # For auto_fe, fe_metadata lives on the active MundlakTransformer.
            active = getattr(self._transformer, "_active_transformer", self._transformer)
            fe_metadata = getattr(active, "fe_metadata", None) or getattr(
                self._transformer, "fe_metadata", None
            )
            fe_cols_for_names = self.fe_cols

            # Hybrid auto_fe: active is the MAP transformer (no fe_metadata).
            # Use the inner MundlakTransformer's metadata and only the
            # Mundlak-routed FE cols so that MAP-routed FEs (not in the table)
            # do not generate spurious avg_*_feN names — which would produce
            # more coef names than actual coefficients.
            is_hybrid = (
                self.method == "auto_fe"
                and bool(getattr(self._transformer, "_mundlak_fe_cols", []))
                and bool(getattr(self._transformer, "_map_fe_cols", []))
            )
            if is_hybrid:
                mundlak_ref = getattr(self._transformer, "_mundlak_transformer_ref", None)
                if mundlak_ref is not None:
                    fe_metadata = mundlak_ref.fe_metadata
                    fe_cols_for_names = self._transformer._mundlak_fe_cols

            # Hybrid: MAP demeaning absorbs the constant; no intercept term.
            # Pure Mundlak (auto_fe or method='mundlak'): data in levels, needs intercept.
            include_intercept = not is_hybrid

            display_names, sql_names = build_coef_name_lists(
                formula=self.formula,
                fe_method="mundlak",
                include_intercept=include_intercept,
                fe_cols=fe_cols_for_names,
                is_iv=False,
                fe_metadata=fe_metadata,
            )

        self._coef_sql_names = sql_names
        return display_names

    def _update_coef_names(self):
        self.coef_names_ = self._build_coef_names_from_formula()
        if len(self.outcome_vars) > 1:
            self.coef_names_ = [
                f"{name}:{outcome}"
                for outcome in self.outcome_vars
                for name in self.coef_names_
            ]

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        result = super().summary()
        result["estimator_type"] = "DuckFE"
        result["fe_method"] = self.method

        if self.method == "iterative_demean" and self._transformer is not None:
            result.update(
                {
                    "n_iterations": getattr(
                        self._transformer, "n_iterations", None
                    ),
                    "tolerance": self.tolerance,
                    "max_iterations": self.max_iterations,
                    "converged": (
                        self._transformer.n_iterations < self.max_iterations
                        if getattr(self._transformer, "n_iterations", None)
                        is not None
                        else None
                    ),
                }
            )
        if self.method == "auto_fe" and self._transformer is not None:
            result["routing"] = getattr(self._transformer, "routing_", {})
            result["cardinalities"] = getattr(self._transformer, "cardinalities_", {})
            active = getattr(self._transformer, "_active_transformer", None)
            if active is not None and not self._transformer.has_intercept:
                n_iter = getattr(active, "n_iterations", None)
                result["n_iterations"] = n_iter
                max_iter = getattr(active, "max_iterations", None)
                result["converged"] = (
                    n_iter < max_iter if n_iter is not None and max_iter is not None else None
                )
        return result
