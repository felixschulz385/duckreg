"""Iterative demeaning transformer (Method of Alternating Projections)."""
import logging
from typing import List, Optional

import duckdb

from .base import FETransformer

logger = logging.getLogger(__name__)


class IterativeDemeanTransformer(FETransformer):
    """Absorbs fixed effects via iterative demeaning (MAP).

    The Method of Alternating Projections demeaning alternates between
    subtracting the conditional mean for each FE dimension until convergence.
    This is provably correct for any panel structure, including unbalanced
    multi-way FE, where analytical within-transformation fails.

    After :meth:`fit_transform`, the result table ``demeaned_data`` contains:

    * The original FE columns (for downstream sampling / DOF computation).
    * The original cluster column, if supplied.
    * A ``resid_{v}`` column for each ``v`` in *variables* (zero-mean within
      every FE group at convergence).

    :meth:`transform_query` renaming aliases ``resid_{v} AS {v}`` so that
    downstream SQL can refer to transformed variables by their original names.

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
    remove_singletons : bool
        Drop singleton FE groups before demeaned.
    max_iterations : int
        Maximum MAP iterations before issuing a convergence warning.
    tolerance : float
        Convergence criterion: stop when the maximum absolute within-group
        mean across all FE dimensions and variables is below *tolerance*.
        Residuals are stored as float32; tolerances below 1e-7 may not be
        achievable and will cause the solver to exhaust *max_iterations*.
    check_interval : int
        Evaluate the convergence criterion only every *check_interval*
        iterations (always checked on the final iteration).  Higher values
        reduce overhead when many iterations are needed; ``1`` (every
        iteration) reproduces the original behaviour.
    convergence_sample : float
        Fraction of rows used to evaluate the convergence criterion
        (0 < convergence_sample <= 1.0).  Values below 1.0 reduce
        convergence-check I/O at the cost of a noisier stopping criterion;
        0.1 is usually sufficient for large datasets.  Default: 1.0 (exact).
    """

    _RESULT_TABLE = "demeaned_data"

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        table_name: str,
        fe_cols: List[str],
        cluster_col: Optional[str] = None,
        remove_singletons: bool = True,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        check_interval: int = 5,
        convergence_sample: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            conn=conn,
            table_name=table_name,
            fe_cols=fe_cols,
            cluster_col=cluster_col,
            remove_singletons=remove_singletons,
            **kwargs,
        )
        if not (0 < convergence_sample <= 1.0):
            raise ValueError(
                f"convergence_sample must be in (0, 1.0]; got {convergence_sample!r}"
            )
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.check_interval = check_interval
        self.convergence_sample = convergence_sample
        self.n_iterations: Optional[int] = None
        self._fe_total_levels: int = 0
        # Reduce temp-storage pressure on large datasets
        self.conn.execute("SET preserve_insertion_order = false")

    # -------------------------------------------------------------------------
    # FETransformer interface
    # -------------------------------------------------------------------------

    def fit_transform(self, variables: List[str], where_clause: str = "") -> str:
        """Demean *variables* via MAP and store the result in ``demeaned_data``.

        Parameters
        ----------
        variables : List[str]
            SQL column names to demean (typically outcomes + non-intercept
            covariates, **excluding** FE columns and the cluster column).
        where_clause : str
            SQL ``WHERE`` clause (including the keyword) or empty string.

        Returns
        -------
        str
            ``"demeaned_data"`` — the temp table holding ``resid_{v}`` columns.
        """
        self._init_demeaned_table(variables, where_clause)
        resid_cols = [f"resid_{v}" for v in variables]
        self._run_map(resid_cols)
        self._compute_fe_levels()
        self._fitted = True
        return self._RESULT_TABLE

    def transform_query(self, variables: List[str]) -> str:
        """Return ``resid_x AS x, resid_y AS y, ...`` for each variable.

        After applying this fragment in a ``SELECT`` against ``demeaned_data``,
        the resulting columns carry their original names, hiding the ``resid_``
        prefix from downstream code.
        """
        return ", ".join(f"resid_{v} AS {v}" for v in variables)

    @property
    def n_obs(self) -> int:
        """Observations in ``demeaned_data`` (after singleton removal)."""
        if self._n_obs is None:
            raise RuntimeError("fit_transform() has not been called yet.")
        return self._n_obs

    @property
    def df_correction(self) -> int:
        """Total distinct FE levels (used for DOF adjustment in vcov).

        Equal to :math:`\\sum_d |\\mathcal{G}_d|` where :math:`|\\mathcal{G}_d|`
        is the number of distinct groups in dimension *d*.
        """
        return self._fe_total_levels

    @property
    def extra_regressors(self) -> List[str]:
        """Empty list — iterative demeaning adds no new regressors."""
        return []

    @property
    def has_intercept(self) -> bool:
        """``False`` — demeaning absorbs the constant term."""
        return False

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _init_demeaned_table(self, variables: List[str], where_clause: str) -> None:
        """Create the initial ``demeaned_data`` table from the source.

        Only complete-case rows (no NULL in any FE, cluster, or model
        variable) are included, so that group means are computed over the
        exact regression sample and the resulting numpy arrays contain no NaN.
        Residuals are stored as FLOAT (float32) to halve on-disk size relative
        to DOUBLE.  MAP convergence is unaffected for tolerances >= 1e-7.  If
        sub-1e-7 tolerance is required, change the CAST to DOUBLE here.

        No synthetic row-id is written here.  ``_run_map`` instead materialises
        DuckDB’s zero-storage ``rowid`` pseudo-column into ``_fe_store`` and
        ``_resid_store`` as an explicit BIGINT join key.  Both tables are
        created from the same ``demeaned_data`` table before it is dropped, so
        their ``rowid`` values are guaranteed to correspond row-for-row.

        Handles singleton removal via hash-joins against lightweight
        ``GROUP BY … HAVING COUNT(*) > 1`` subqueries — one join per FE
        dimension — when ``remove_singletons=True``.  This avoids the
        full sort/buffer pass that ``QUALIFY count(*) OVER (PARTITION BY
        …)`` requires and prevents out-of-memory errors on large datasets.
        """
        self.n_rows_dropped_singletons = 0
        select_parts = list(self.fe_cols)
        if self.cluster_col:
            select_parts.append(self.cluster_col)
        for v in variables:
            select_parts.append(f"CAST({v} AS FLOAT) AS resid_{v}")

        # Complete-case filter: drop rows with NULL in any model column
        # (FE dims, cluster, outcomes, covariates) so demeaning group means
        # are computed over the same sample that enters the regression.
        all_model_cols = list(self.fe_cols) + list(variables)
        if self.cluster_col:
            all_model_cols.append(self.cluster_col)
        null_filter = " AND ".join(f"{c} IS NOT NULL" for c in all_model_cols)
        if where_clause:
            combined_where = f"{where_clause} AND {null_filter}"
        else:
            combined_where = f"WHERE {null_filter}"

        if self.remove_singletons and self.fe_cols:
            # inner_where: used inside singleton subqueries (always needs WHERE keyword)
            if where_clause:
                inner_where = f"{where_clause} AND {null_filter}"
            else:
                inner_where = f"WHERE {null_filter}"

            # rows_before counts only the NULL-filtered set so that
            # n_rows_dropped_singletons reflects singleton removal only,
            # not NULL filtering.
            rows_before = self.conn.execute(
                f"SELECT COUNT(*) FROM {self.table_name} {inner_where}"
            ).fetchone()[0]

            # outer_where: applied to the main SELECT (same content, explicit label)
            outer_where = inner_where

            # One hash-join per FE dimension: keep only rows whose group has
            # more than one member.  Uses GROUP BY … HAVING, which is a
            # lightweight aggregation with no sort pass, instead of the
            # window-function QUALIFY that buffers the full table.
            singleton_joins = "\n".join(
                f"JOIN (\n"
                f"    SELECT {fe} FROM {self.table_name} {inner_where}\n"
                f"    GROUP BY {fe} HAVING COUNT(*) > 1\n"
                f") _sg_{i} USING ({fe})"
                for i, fe in enumerate(self.fe_cols)
            )

            self.conn.execute(f"""
            CREATE OR REPLACE TABLE {self._RESULT_TABLE} AS
            SELECT {', '.join(select_parts)}
            FROM {self.table_name}
            {singleton_joins}
            {outer_where}
            """)

            rows_after = self.conn.execute(
                f"SELECT COUNT(*) FROM {self._RESULT_TABLE}"
            ).fetchone()[0]
            self.n_rows_dropped_singletons = rows_before - rows_after
            logger.debug(
                f"Singleton removal via HAVING join: {self.n_rows_dropped_singletons} rows "
                f"removed ({rows_before} → {rows_after})"
            )
        else:
            self.conn.execute(f"""
            CREATE OR REPLACE TABLE {self._RESULT_TABLE} AS
            SELECT {', '.join(select_parts)}
            FROM {self.table_name}
            {combined_where}
            """)

        self._n_obs = self.conn.execute(
            f"SELECT COUNT(*) FROM {self._RESULT_TABLE}"
        ).fetchone()[0]
        logger.debug(
            f"Initialising MAP for {len(self.fe_cols)} FE(s) × "
            f"{len(variables)} variable(s), n={self._n_obs}"
        )

    def _run_map(self, resid_cols: List[str]) -> None:
        """Execute the Method of Alternating Projections until convergence.

        Convergence is measured as the maximum absolute within-group mean
        across all (FE dimension, variable) pairs.  At convergence every
        group mean is zero, so this criterion is equivalent to the
        traditional row-level change check but requires only lightweight
        ``GROUP BY`` aggregations.

        The check is performed every :attr:`check_interval` iterations (and
        always on the final iteration) to amortise its cost when many
        iterations are needed.
        When :attr:`convergence_sample` is below 1.0, each convergence scan uses
        Bernoulli row sampling to reduce I/O at the cost of a noisier stopping
        criterion; the sample is drawn independently per FE dimension.

        **Split-table design** (no ``UPDATE``, no MVCC shadow copy):

        * ``_fe_store``    — materialised ``rowid`` + FE cols + cluster col.
          Written once, never mutated.  The ``rowid`` is DuckDB's zero-storage
          pseudo-column materialised once so we have a stable, unique per-row
          join key without writing an explicit BIGINT column into both tables
          (~128 GB saved at 8 B rows).
        * ``_dict_{fe}``  — compact bijection ``{fe_col} ↔ _code_{fe_col}``
          (UINTEGER, 0-based), built once from the distinct values in
          ``_fe_store``.  Replaces the per-row FE string/BIGINT copies that
          previously lived in ``_resid_store`` (~64 GB per BIGINT FE column).
        * ``_resid_store`` — materialised ``rowid`` + one UINTEGER code per FE
          + ``resid_*`` FLOAT cols.  Narrow; recreated on every demeaning step
          via a streaming hash-join against an inline ``(SELECT … GROUP BY …)``
          subquery fused directly into the ``CREATE … AS SELECT … JOIN``.
          Because each step is a ``CREATE … AS SELECT … JOIN``, DuckDB streams
          rows without buffering the full column set — eliminating the
          MVCC shadow-copy that ``UPDATE … FROM (GROUP BY …)`` triggers.
          The separate ``_means`` scratch table no longer exists; per-group
          averages are computed inline.

        After the loop (converged or exhausted), ``demeaned_data`` is
        reconstructed from ``_fe_store`` ⋈ ``_resid_store`` joined on
        ``rowid``, with FE and cluster columns sourced from ``_fe_store``
        only.  All scratch tables (``_fe_store``, ``_dict_*``, ``_resid_store``,
        ``_resid_new``) are dropped unconditionally via ``try/finally``.

        MAP *requires* sequential application — the group means for dimension
        *d+1* must be computed from residuals already centred on dimension *d*.
        Each FE dimension therefore remains a separate SQL statement.
        """
        fe_cols_sql = ", ".join(self.fe_cols)
        cluster_sql = f", {self.cluster_col}" if self.cluster_col else ""

        # _fe_store: materialised rowid + FE cols + cluster col — written once.
        # rowid is DuckDB's pseudo-column; materialising it avoids writing an
        # explicit BIGINT _row_id column into demeaned_data (~128 GB at 8 B rows).
        self.conn.execute(f"""
            CREATE OR REPLACE TABLE _fe_store AS
            SELECT rowid, {fe_cols_sql}{cluster_sql}
            FROM {self._RESULT_TABLE}
        """)

        # Build one compact dictionary per FE dimension.
        # _dict_{fe}: {fe_col} ↔ _code_{fe_col} UINTEGER (0-based).
        # UINTEGER (4 bytes) vs BIGINT FE key (8 bytes) saves ~64 GB per FE
        # column at 8 B rows over the lifetime of _resid_store.
        for fe_col in self.fe_cols:
            self.conn.execute(f"""
                CREATE OR REPLACE TABLE _dict_{fe_col} AS
                SELECT {fe_col},
                       CAST(ROW_NUMBER() OVER (ORDER BY {fe_col}) - 1 AS UINTEGER)
                           AS _code_{fe_col}
                FROM (SELECT DISTINCT {fe_col} FROM _fe_store) t
            """)

        # _resid_store: materialised rowid + one UINTEGER code per FE + resid_* cols.
        # The original FE string/integer values live only in _fe_store and the
        # _dict_* tables; _resid_store holds only compact codes and residuals.
        resid_select = ", ".join(resid_cols)
        code_joins = "\n".join(
            f"JOIN _dict_{fe_col} ON {self._RESULT_TABLE}.{fe_col} = _dict_{fe_col}.{fe_col}"
            for fe_col in self.fe_cols
        )
        code_select = ", ".join(
            f"_dict_{fe_col}._code_{fe_col}" for fe_col in self.fe_cols
        )
        self.conn.execute(f"""
            CREATE OR REPLACE TABLE _resid_store AS
            SELECT {self._RESULT_TABLE}.rowid, {code_select}, {resid_select}
            FROM {self._RESULT_TABLE}
            {code_joins}
        """)

        # Wide table no longer needed during iteration
        self.conn.execute(f"DROP TABLE IF EXISTS {self._RESULT_TABLE}")

        # Pre-build convergence SQL: max |AVG(resid)| over every
        # (FE dimension × variable) combination.
        # One scan per FE dimension: all variables are averaged in a single
        # GROUP BY, then UNNEST expands the array of averages into rows so
        # MAX(ABS(...)) works across all variables.  This reduces the number
        # of full scans from |fe_cols| × |resid_cols| to |fe_cols|.
        # Groups by integer code — avoids string/BIGINT comparison overhead.
        _sample_clause = (
            f"USING SAMPLE {self.convergence_sample * 100} PERCENT (bernoulli)"
            if self.convergence_sample < 1.0
            else ""
        )
        # When convergence_sample < 1.0 each FE dimension samples _resid_store
        # independently (different random draws).  This is intentional: correlated
        # draws are unnecessary for a convergence criterion and independent draws
        # give better coverage across dimensions at no extra cost.
        _dim_conv_parts = [
            (
                "SELECT UNNEST([" +
                ", ".join(f"avg_{rc}" for rc in resid_cols) +
                "]) AS avg_val "
                "FROM ("
                "SELECT " +
                ", ".join(f"AVG({rc}) AS avg_{rc}" for rc in resid_cols) +
                f" FROM _resid_store {_sample_clause} "
                f"GROUP BY _code_{fe}"
                ") _agg"
            )
            for fe in self.fe_cols
        ]
        conv_sql = (
            "SELECT MAX(ABS(avg_val)) AS max_group_mean "
            f"FROM ({' UNION ALL '.join(_dim_conv_parts)}) t"
        )

        avg_exprs = ", ".join(f"AVG({rc}) AS _avg_{rc}" for rc in resid_cols)

        # All code column expressions for SELECT — carried forward in every
        # _resid_new write so subsequent dimensions can GROUP BY their code.
        all_code_select = ", ".join(f"r._code_{fe}" for fe in self.fe_cols)

        # Compute scratch table names upfront for the finally block.
        dict_table_names = [f"_dict_{fe}" for fe in self.fe_cols]

        max_change = float("inf")
        try:
            for iteration in range(self.max_iterations):
                # One pass: demean sequentially by each FE dimension.
                for fe_col in self.fe_cols:
                    code_col = f"_code_{fe_col}"

                    # Fused step: compute per-group means inline and subtract
                    # in a single CREATE … AS SELECT … JOIN.  The inline
                    # subquery replaces the former _means scratch table,
                    # saving one materialisation + one table drop per iteration
                    # per FE dimension.  All code columns are carried forward
                    # so subsequent dimensions can GROUP BY their code without
                    # an extra join.
                    # Note: the inline subquery causes _resid_store to be scanned twice per
                    # step (once as the outer `r`, once inside the GROUP BY subquery).  For
                    # typical panel datasets where the number of groups G << N this is cheaper
                    # overall than a separate _means materialisation because the GROUP BY
                    # result (G × V rows) fits in L3 cache; for datasets where G is large
                    # relative to N, consider re-introducing an explicit _means table.
                    sub_exprs = ", ".join(
                        f"r.{rc} - m._avg_{rc} AS {rc}" for rc in resid_cols
                    )
                    self.conn.execute(f"""
                        CREATE OR REPLACE TABLE _resid_new AS
                        SELECT r.rowid, {all_code_select},
                               {sub_exprs}
                        FROM _resid_store r
                        JOIN (
                            SELECT {code_col}, {avg_exprs}
                            FROM _resid_store
                            GROUP BY {code_col}
                        ) m USING ({code_col})
                    """)

                    # Atomic swap — drop old, rename new.
                    self.conn.execute("DROP TABLE _resid_store")
                    self.conn.execute("ALTER TABLE _resid_new RENAME TO _resid_store")

                # Check convergence every check_interval iterations, and always
                # on the last iteration so the warning message is accurate.
                is_last = iteration == self.max_iterations - 1
                if (iteration + 1) % self.check_interval == 0 or is_last:
                    result = self.conn.execute(conv_sql).fetchone()
                    max_change = result[0] if result[0] is not None else 0.0
                    logger.debug(
                        f"MAP iteration {iteration + 1}: max_group_mean={max_change:.2e}"
                    )
                    if max_change < self.tolerance:
                        self.n_iterations = iteration + 1
                        logger.debug(f"MAP converged after {self.n_iterations} iterations")
                        break
            else:
                self.n_iterations = self.max_iterations
                logger.warning(
                    f"MAP did not converge after {self.max_iterations} iterations "
                    f"(max_group_mean={max_change:.2e}, tolerance={self.tolerance:.2e})"
                )
        finally:
            # Check which scratch tables survived an exception mid-swap.
            all_scratch = ["_fe_store", "_resid_store", "_resid_new"] + dict_table_names
            placeholders = ", ".join(f"'{t}'" for t in all_scratch)
            tables_present = {
                row[0]
                for row in self.conn.execute(
                    "SELECT table_name FROM information_schema.tables "
                    f"WHERE table_name IN ({placeholders})"
                ).fetchall()
            }
            if "_resid_store" in tables_present and "_fe_store" in tables_present:
                # Reconstruct demeaned_data: FE/cluster cols from _fe_store,
                # residuals from _resid_store.  rowid is the join key only —
                # it is NOT written to the result table.
                fe_exprs = ", ".join(f"f.{fe}" for fe in self.fe_cols)
                cluster_expr = f", f.{self.cluster_col}" if self.cluster_col else ""
                resid_exprs = ", ".join(f"r.{rc}" for rc in resid_cols)
                self.conn.execute(f"""
                    CREATE OR REPLACE TABLE {self._RESULT_TABLE} AS
                    SELECT {fe_exprs}{cluster_expr}, {resid_exprs}
                    FROM _fe_store f
                    JOIN _resid_store r ON f.rowid = r.rowid
                """)
            for t in all_scratch:
                if t in tables_present:
                    self.conn.execute(f"DROP TABLE IF EXISTS {t}")

    def _compute_fe_levels(self) -> None:
        """Count distinct levels across all FE dimensions in a single query.

        Replaces the previous per-dimension loop (N round-trips) with a
        single SQL statement that sums all ``COUNT(DISTINCT ...)`` expressions
        at once, reducing latency for high-FE-count models.
        """
        if not self.fe_cols:
            self._fe_total_levels = 0
            return
        # Build: SELECT COUNT(DISTINCT fe1) + COUNT(DISTINCT fe2) + ... AS total
        sum_expr = " + ".join(
            f"COUNT(DISTINCT {fe_col})" for fe_col in self.fe_cols
        )
        result = self.conn.execute(
            f"SELECT {sum_expr} AS total FROM {self._RESULT_TABLE}"
        ).fetchone()
        self._fe_total_levels = result[0] or 0
        logger.debug(
            f"FE levels: {self._fe_total_levels} total across {len(self.fe_cols)} dim(s)"
        )
