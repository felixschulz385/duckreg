"""Auto-routing fixed-effects transformer.

Routes each FE dimension to either the Mundlak device or iterative demeaning
/ MAP based on a cheap cardinality estimate computed on a random row sample.
Very low and very high cardinality dimensions use Mundlak; intermediate
cardinality dimensions use MAP.
"""
import logging
from typing import Dict, List, Optional

import duckdb

from .base import FETransformer
from .iterative_demean import IterativeDemeanTransformer
from .mundlak import MundlakTransformer

logger = logging.getLogger(__name__)

_EXPERIMENTAL_ERROR = (
    "AutoFETransformer is experimental and has been temporarily disabled. "
    "Use fe_method='demean' (iterative demeaning) instead."
)


class AutoFETransformer(FETransformer):
    """Automatically routes FE dimensions to Mundlak or MAP.

        For each FE dimension the transformer estimates cardinality on a random
        sample of rows and compares it against two cutoffs:

        * **Low-cardinality cutoff** (fixed at ``30``): dimensions with
            ``c_est < 30`` are routed to :class:`MundlakTransformer`.
        * **High-cardinality cutoff** ``N / cardinality_ratio``: dimensions with
            ``c_est > N / cardinality_ratio`` are routed to
            :class:`MundlakTransformer`.
        * Dimensions in the intermediate band are absorbed via
            :class:`IterativeDemeanTransformer` (MAP demeaning).

    Three execution modes are supported:

    **Pure Mundlak** — all FEs are routed to Mundlak (all outside the
    intermediate band).
        Delegates entirely to :class:`MundlakTransformer`.  The result table
        is renamed to ``demeaned_data`` for a uniform downstream interface.

    **Pure MAP** — all FEs fall in the intermediate band.
        Delegates entirely to :class:`IterativeDemeanTransformer`.

    **Hybrid** — mixed FEs.
        Step 1: :class:`MundlakTransformer` augments the data with group-mean
        columns for the low-cardinality FEs.  Step 2:
        :class:`IterativeDemeanTransformer` demeaning is applied to all
        variables (including the Mundlak-added columns) using the
        high-cardinality FEs, producing the final ``demeaned_data`` table.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    table_name : str
        Source table/view name.
    fe_cols : List[str]
        SQL column names of the FE dimensions used for partitioning.
    cluster_col : str, optional
        SQL column name to carry through (for clustered SEs / bootstrap).
    covariate_cols : List[str], optional
        SQL column names of covariates for which Mundlak group means are
        computed (used only for Mundlak-routed dimensions).
    remove_singletons : bool
        Drop singleton FE groups before transformation.
    cardinality_ratio : float
        High-cardinality cutoff denominator.  A FE dimension with estimated
        cardinality ``c_est`` is routed to Mundlak when
        ``c_est > N / cardinality_ratio``.
    sample_size : int
        Number of rows sampled for the cardinality estimate.  The actual
        sample is ``min(N, sample_size)``.
    mundlak_kwargs : dict, optional
        Extra keyword arguments forwarded to :class:`MundlakTransformer`.
    map_kwargs : dict, optional
        Extra keyword arguments forwarded to
        :class:`IterativeDemeanTransformer`.
    **kwargs
        Additional keyword arguments forwarded to the base class.
    """

    _RESULT_TABLE = "demeaned_data"
    _LOW_CARDINALITY_CUTOFF = 30

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        table_name: str,
        fe_cols: List[str],
        cluster_col: Optional[str] = None,
        covariate_cols: Optional[List[str]] = None,
        remove_singletons: bool = True,
        cardinality_ratio: float = 10.0,
        sample_size: int = 50_000,
        mundlak_kwargs: Optional[dict] = None,
        map_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        raise NotImplementedError(_EXPERIMENTAL_ERROR)
        super().__init__(
            conn=conn,
            table_name=table_name,
            fe_cols=fe_cols,
            cluster_col=cluster_col,
            remove_singletons=remove_singletons,
            **kwargs,
        )
        self.covariate_cols: List[str] = list(covariate_cols or [])
        self.cardinality_ratio = cardinality_ratio
        self.sample_size = sample_size
        self.mundlak_kwargs: dict = mundlak_kwargs or {}
        self.map_kwargs: dict = map_kwargs or {}

        # Populated by fit_transform
        self.routing_: Dict[str, str] = {}
        self.cardinalities_: Dict[str, int] = {}
        self._mundlak_fe_cols: List[str] = []
        self._map_fe_cols: List[str] = []
        self._active_transformer: Optional[FETransformer] = None

        # Populated after routing is resolved
        self._has_intercept: bool = False
        self._extra_regressors: List[str] = []

    # -------------------------------------------------------------------------
    # FETransformer interface
    # -------------------------------------------------------------------------

    def fit_transform(self, variables: List[str], where_clause: str = "") -> str:
        """Route FE dimensions and apply the appropriate transformation strategy.

        Parameters
        ----------
        variables : List[str]
            SQL column names to transform (outcomes + non-intercept covariates,
            excluding FE columns and the cluster column).
        where_clause : str
            SQL ``WHERE`` clause (including the keyword) or empty string.

        Returns
        -------
        str
            ``"demeaned_data"`` — the result table after transformation.
        """
        self._route(where_clause)

        if self._mundlak_fe_cols and not self._map_fe_cols:
            self._fit_pure_mundlak(variables, where_clause)
        elif self._map_fe_cols and not self._mundlak_fe_cols:
            self._fit_pure_map(variables, where_clause)
        else:
            self._fit_hybrid(variables, where_clause)

        self._fitted = True
        return self._RESULT_TABLE

    def transform_query(self, variables: List[str]) -> str:
        """Delegate to the active sub-transformer's ``transform_query``.

        Parameters
        ----------
        variables : List[str]
            Original SQL column names (same list passed to
            :meth:`fit_transform`).

        Returns
        -------
        str
            Comma-separated SQL expressions suitable for embedding in a
            ``SELECT`` list.
        """
        if self._active_transformer is None:
            raise RuntimeError("fit_transform() has not been called yet.")
        return self._active_transformer.transform_query(variables)

    @property
    def n_obs(self) -> int:
        """Observations after filtering / singleton removal."""
        if self._active_transformer is None:
            raise RuntimeError("fit_transform() has not been called yet.")
        return self._active_transformer.n_obs

    @property
    def df_correction(self) -> int:
        """DOF correction delegated to the active sub-transformer.

        In pure-Mundlak mode this is ``0`` (FE parameters are explicit
        regressors).  In pure-MAP or hybrid mode this is the sum of distinct
        FE levels absorbed by MAP.
        """
        if self._active_transformer is None:
            return 0
        return self._active_transformer.df_correction

    @property
    def extra_regressors(self) -> List[str]:
        """Columns added by the transformation that must enter the RHS.

        * Pure Mundlak — Mundlak mean cols, dummy cols, dummy-mean cols.
        * Pure MAP — ``[]``.
        * Hybrid — Mundlak-added columns only (MAP demeaned them in-place;
          they still appear as regressors but carry no ``resid_`` prefix after
          :meth:`transform_query` remapping).
        """
        return list(self._extra_regressors)

    @property
    def has_intercept(self) -> bool:
        """``True`` iff the active transformation requires a constant term.

        Pure Mundlak keeps data in levels → intercept needed.
        Pure MAP absorbs the constant via demeaning → no intercept.
        Hybrid: MAP demeaning over the demean_fe absorbs the constant from
        the full augmented matrix (covariates + Mundlak means), so no
        intercept is needed — matching ``mixed_demean_mundlak_ols``.
        """
        return self._has_intercept

    # -------------------------------------------------------------------------
    # Routing
    # -------------------------------------------------------------------------

    def _route(self, where_clause: str) -> None:
        """Estimate cardinalities and assign each FE to Mundlak or MAP.

        Parameters
        ----------
        where_clause : str
            SQL ``WHERE`` clause applied to the cardinality sample query.
        """
        # Total row count for scaling
        count_sql = f"SELECT COUNT(*) FROM {self.table_name}"
        if where_clause:
            count_sql += f" {where_clause}"
        n_total: int = self.conn.execute(count_sql).fetchone()[0]

        actual_sample = min(n_total, self.sample_size)
        threshold = n_total / self.cardinality_ratio

        if actual_sample == 0 or not self.fe_cols:
            # Degenerate case: no rows or no FEs — route everything to MAP
            self._map_fe_cols = list(self.fe_cols)
            self._mundlak_fe_cols = []
            logger.info(
                "AutoFE routing: no rows or no FE columns — all FEs → MAP"
            )
            return

        # Single SQL pass: COUNT(DISTINCT fe) for every FE simultaneously
        count_exprs = ", ".join(
            f"COUNT(DISTINCT {fe}) AS _card_{i}"
            for i, fe in enumerate(self.fe_cols)
        )
        sample_where = f"{where_clause} USING SAMPLE {actual_sample} ROWS"
        sample_sql = (
            f"SELECT {count_exprs} "
            f"FROM {self.table_name} "
            f"{sample_where}"
        )
        row = self.conn.execute(sample_sql).fetchone()

        for i, fe in enumerate(self.fe_cols):
            c_sample: int = row[i] or 0
            # Scale sample cardinality back to full-table estimate, cap at N
            c_est = min(int(c_sample * n_total / actual_sample), n_total)
            self.cardinalities_[fe] = c_est

            if c_est < self._LOW_CARDINALITY_CUTOFF or c_est > threshold:
                # very low cardinality (e.g. year) or very high cardinality
                # (e.g. unit FE with ~1 obs/group) => Mundlak device.
                # Very-high-cardinality groups approach individual fixed
                # effects; the Mundlak group-mean control collapses to the
                # observation itself and provides negligible bias at far lower
                # computational cost than demeaning 200k groups.
                self.routing_[fe] = "mundlak"
                self._mundlak_fe_cols.append(fe)
            else:
                # intermediate cardinality => MAP (within-demeaning)
                self.routing_[fe] = "map"
                self._map_fe_cols.append(fe)

            logger.info(
                f"AutoFE routing: '{fe}' → "
                f"{'MUNDLAK' if self.routing_[fe] == 'mundlak' else 'MAP'} "
                f"(c_est={c_est}, threshold={threshold:.1f}, N={n_total})"
            )

    # -------------------------------------------------------------------------
    # Execution paths
    # -------------------------------------------------------------------------

    def _fit_pure_mundlak(self, variables: List[str], where_clause: str) -> None:
        """Absorb all FEs via Mundlak.  Renames output to ``demeaned_data``."""
        logger.debug("AutoFE: pure Mundlak path")
        mundlak = MundlakTransformer(
            conn=self.conn,
            table_name=self.table_name,
            fe_cols=self._mundlak_fe_cols,
            cluster_col=self.cluster_col,
            covariate_cols=self.covariate_cols,
            remove_singletons=self.remove_singletons,
            **self.mundlak_kwargs,
        )
        mundlak.fit_transform(variables, where_clause)

        # Rename 'design_matrix' → 'demeaned_data' for a uniform interface
        if MundlakTransformer._RESULT_TABLE != self._RESULT_TABLE:
            self.conn.execute(
                f"CREATE OR REPLACE TEMP TABLE {self._RESULT_TABLE} AS "
                f"SELECT * FROM {MundlakTransformer._RESULT_TABLE}"
            )
            self.conn.execute(
                f"DROP TABLE IF EXISTS {MundlakTransformer._RESULT_TABLE}"
            )

        self._active_transformer = mundlak
        self._has_intercept = True
        self._extra_regressors = list(mundlak.extra_regressors)
        self.n_rows_dropped_singletons = mundlak.n_rows_dropped_singletons

    def _fit_pure_map(self, variables: List[str], where_clause: str) -> None:
        """Absorb all FEs via MAP."""
        logger.debug("AutoFE: pure MAP path")
        mapper = IterativeDemeanTransformer(
            conn=self.conn,
            table_name=self.table_name,
            fe_cols=self._map_fe_cols,
            cluster_col=self.cluster_col,
            remove_singletons=self.remove_singletons,
            **self.map_kwargs,
        )
        mapper.fit_transform(variables, where_clause)

        self._active_transformer = mapper
        self._has_intercept = False
        self._extra_regressors = []
        self.n_rows_dropped_singletons = mapper.n_rows_dropped_singletons

    def _fit_hybrid(self, variables: List[str], where_clause: str) -> None:
        """Absorb FEs in two steps: Mundlak first, then MAP on augmented data.

        Step 1
        ------
        :class:`MundlakTransformer` adds group-mean regressors for the
        low-cardinality FEs.  Singleton removal is performed here.

        Step 2
        ------
        :class:`IterativeDemeanTransformer` demeaning is run on the
        ``design_matrix`` table produced by Step 1, using the high-cardinality
        FEs.  The variables to demean include both the original *variables* and
        all columns added by the Mundlak step (``mundlak_extra``), so that MAP
        partials out the Mundlak regressors as well.  ``remove_singletons`` is
        ``False`` because Step 1 already filtered singletons.
        """
        logger.debug(
            f"AutoFE: hybrid path — "
            f"{len(self._mundlak_fe_cols)} Mundlak FE(s), "
            f"{len(self._map_fe_cols)} MAP FE(s)"
        )

        # ── Step 1: Mundlak augmentation ─────────────────────────────────────
        # Pass MAP FE columns as extra variables so that design_matrix
        # contains them as plain columns.  The Mundlak transformer does not
        # compute group means for them (they are not in covariate_cols), so
        # they are simply passed through unchanged.  Without this, the MAP
        # step would fail because its null-filter and GROUP BY reference
        # columns that do not exist in design_matrix.
        variables_for_mundlak = list(variables) + self._map_fe_cols

        mundlak = MundlakTransformer(
            conn=self.conn,
            table_name=self.table_name,
            fe_cols=self._mundlak_fe_cols,
            cluster_col=self.cluster_col,
            covariate_cols=self.covariate_cols,
            remove_singletons=self.remove_singletons,
            **self.mundlak_kwargs,
        )
        mundlak.fit_transform(variables_for_mundlak, where_clause)
        mundlak_extra = list(mundlak.extra_regressors)
        self.n_rows_dropped_singletons = mundlak.n_rows_dropped_singletons

        logger.debug(
            f"AutoFE hybrid Step 1 complete: "
            f"{len(mundlak_extra)} Mundlak column(s) added"
        )

        # ── Step 2: MAP on augmented data ─────────────────────────────────────
        # Demean original variables + Mundlak-added columns.  The MAP FE
        # columns (e.g. pixel_id) are present in design_matrix as plain
        # pass-through columns and serve as fe_cols here, not as variables.
        # Pass empty where_clause: Mundlak already filtered rows.
        all_demean_vars = list(variables) + mundlak_extra

        mapper = IterativeDemeanTransformer(
            conn=self.conn,
            table_name=MundlakTransformer._RESULT_TABLE,  # 'design_matrix'
            fe_cols=self._map_fe_cols,
            cluster_col=self.cluster_col,
            remove_singletons=False,  # already handled by Mundlak step
            **self.map_kwargs,
        )
        mapper.fit_transform(all_demean_vars, where_clause="")

        # design_matrix is fully consumed by MAP (_init_demeaned_table copied it
        # into demeaned_data).  Drop it immediately to reclaim peak-temp storage.
        self.conn.execute(
            f"DROP TABLE IF EXISTS {MundlakTransformer._RESULT_TABLE}"
        )

        logger.debug("AutoFE hybrid Step 2 (MAP) complete")

        self._active_transformer = mapper
        self._mundlak_transformer_ref = mundlak  # for coef naming in DuckFE
        # MAP demeaning over demean_fe absorbs the constant term (every
        # column, including the Mundlak means, is zero-mean within each MAP
        # group after convergence).  No intercept is needed — matching the
        # reference mixed_demean_mundlak_ols which runs OLS without const.
        self._has_intercept = False
        self._extra_regressors = mundlak_extra
