"""Iterative demeaning transformer (Method of Alternating Projections)."""

import logging
from typing import Dict, List, Literal, Optional

import duckdb

from .base import FETransformer

logger = logging.getLogger(__name__)


class IterativeDemeanTransformer(FETransformer):
    """Absorb fixed effects via iterative demeaning (MAP).

    The Method of Alternating Projections repeatedly subtracts FE-specific
    means until the residual columns reach a fixed point. Convergence is
    measured by the maximum absolute remaining FE-group mean across all FE
    dimensions and transformed variables. This fixed-point criterion is a
    robust global check, but it is not identical to the finite-tolerance
    row-level update norm used by fixest / PyFixest.

    After :meth:`fit_transform`, ``demeaned_data`` contains the original FE
    columns, the original cluster column if supplied, and one safe internal
    residual column per transformed variable. :meth:`transform_query`
    projects those internal residual columns back to the original variable
    names.

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
        Drop singleton FE groups before demeaning.
    max_iterations : int
        Maximum MAP iterations before issuing a convergence warning.
    tolerance : float
        Stop once the exact maximum absolute remaining FE-group mean falls
        below this threshold. Residuals are stored as DOUBLE, so the default
        ``1e-8`` tolerance is intentionally supported.
    check_interval : int
        Evaluate the convergence criterion every ``check_interval`` iterations
        and always on the final iteration.
    convergence_sample : float
        Fraction of rows used for a cheap sampled convergence screen
        ``(0 < convergence_sample <= 1.0)``. Sampled checks never stop the
        algorithm on their own; when a sampled check passes, an exact full-data
        check is run immediately and only the exact check can terminate MAP.
    min_iterations_before_check : int
        Minimum number of MAP iterations before a non-final convergence check
        is allowed.
    check_interval_growth : bool
        If ``True``, convergence checks become less frequent as iterations
        accumulate.
    max_check_interval : int
        Upper bound for the adaptive convergence-check interval.
    singleton_pruning : {"iterative", "one_pass"}
        Singleton pruning strategy. ``"iterative"`` removes cascading
        singletons; ``"one_pass"`` removes only groups detected in the first
        pass.
    fe_order : {"input", "ascending_groups", "descending_groups"}
        Order in which FE dimensions are applied inside each MAP sweep.
    drop_constant_variables : bool
        If ``True``, residual columns that are constant after filtering are
        skipped during MAP. With at least one FE dimension, those columns are
        set to zero.
    residual_type : {"DOUBLE", "FLOAT"}
        Storage type used for residual columns. ``DOUBLE`` supports the default
        ``1e-8`` tolerance; ``FLOAT`` reduces I/O but should be paired with
        a relaxed tolerance such as ``1e-6``.
    duckdb_memory_limit : str, optional
        If provided, forwarded to ``SET memory_limit``.
    duckdb_threads : int, optional
        If provided, forwarded to ``SET threads``.

    Notes
    -----
    For very large problems, practical settings are often
    ``convergence_sample=1.0``, ``check_interval=10``,
    ``min_iterations_before_check=5``, ``check_interval_growth=True``,
    ``singleton_pruning="one_pass"`` only when upstream pruning is trusted,
    and ``residual_type="FLOAT"`` only with a relaxed tolerance such as
    ``1e-6``.
    """

    _RESULT_TABLE = "demeaned_data"
    _ROW_ID_COL = "_row_id"

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
        min_iterations_before_check: int = 5,
        check_interval_growth: bool = True,
        max_check_interval: int = 25,
        singleton_pruning: Literal["iterative", "one_pass"] = "iterative",
        fe_order: Literal["input", "ascending_groups", "descending_groups"] = "input",
        drop_constant_variables: bool = False,
        residual_type: Literal["DOUBLE", "FLOAT"] = "DOUBLE",
        duckdb_memory_limit: Optional[str] = None,
        duckdb_threads: Optional[int] = None,
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
        if singleton_pruning not in {"iterative", "one_pass"}:
            raise ValueError(
                "singleton_pruning must be 'iterative' or 'one_pass'; "
                f"got {singleton_pruning!r}"
            )
        if fe_order not in {"input", "ascending_groups", "descending_groups"}:
            raise ValueError(
                "fe_order must be 'input', 'ascending_groups', or "
                f"'descending_groups'; got {fe_order!r}"
            )
        if residual_type not in {"DOUBLE", "FLOAT"}:
            raise ValueError(
                f"residual_type must be 'DOUBLE' or 'FLOAT'; got {residual_type!r}"
            )
        if residual_type == "FLOAT" and tolerance < 1e-7:
            raise ValueError(
                "residual_type='FLOAT' requires tolerance >= 1e-7; "
                f"got {tolerance:.2e}"
            )
        if min_iterations_before_check < 1:
            raise ValueError("min_iterations_before_check must be >= 1")
        if max_check_interval < 1:
            raise ValueError("max_check_interval must be >= 1")
        if duckdb_threads is not None and duckdb_threads < 1:
            raise ValueError("duckdb_threads must be >= 1")
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.check_interval = check_interval
        self.convergence_sample = convergence_sample
        self.min_iterations_before_check = min_iterations_before_check
        self.check_interval_growth = check_interval_growth
        self.max_check_interval = max_check_interval
        self.singleton_pruning = singleton_pruning
        self.fe_order = fe_order
        self.drop_constant_variables = drop_constant_variables
        self.residual_type = residual_type
        self.duckdb_memory_limit = duckdb_memory_limit
        self.duckdb_threads = duckdb_threads
        self.n_iterations: Optional[int] = None
        self._fe_total_levels: int = 0
        self._fe_level_counts: Dict[str, int] = {}
        self._resid_name_map: Dict[str, str] = {}
        self._fe_code_map: List[Dict[str, str]] = []
        self._map_fe_code_order: List[Dict[str, str]] = []
        self.conn.execute("SET preserve_insertion_order = false")
        if self.duckdb_memory_limit is not None:
            self.conn.execute(f"SET memory_limit = '{self.duckdb_memory_limit}'")
            logger.debug("Set DuckDB memory_limit=%s", self.duckdb_memory_limit)
        if self.duckdb_threads is not None:
            self.conn.execute(f"SET threads = {self.duckdb_threads}")
            logger.debug("Set DuckDB threads=%d", self.duckdb_threads)

    @staticmethod
    def qident(name: str) -> str:
        """Return *name* quoted as a SQL identifier."""
        return '"' + name.replace('"', '""') + '"'

    @classmethod
    def _source_relation_sql(cls, table_name: str) -> str:
        """Return SQL for a source table name or relation expression."""
        stripped = table_name.strip()
        if "(" in stripped or stripped.startswith('"'):
            return table_name
        if any(char.isspace() for char in stripped):
            return cls.qident(stripped)
        return table_name

    def _known_tables(self) -> List[str]:
        return [
            "_fe_store",
            "_resid_store",
            "_resid_new",
            "_singleton_work",
            "_singleton_next",
            "_result_final",
        ] + [spec["dict_table"] for spec in self._fe_code_map]

    def _table_exists(self, table_name: str) -> bool:
        return bool(
            self.conn.execute(
                "SELECT 1 FROM information_schema.tables "
                f"WHERE table_name = '{table_name}'"
            ).fetchone()
        )

    def _drop_tables_if_exist(self, table_names: List[str]) -> None:
        for table_name in table_names:
            self.conn.execute(
                f"DROP TABLE IF EXISTS {self.qident(table_name)}"
            )

    def _build_fe_code_map(self) -> None:
        self._fe_code_map = [
            {
                "fe_col": fe_col,
                "dict_table": f"_dict_fe_{idx}",
                "code_col": f"_code_fe_{idx}",
            }
            for idx, fe_col in enumerate(self.fe_cols)
        ]

    def _resid_cols(self, variables: List[str]) -> List[str]:
        return [self._resid_name_map[var] for var in variables]

    def _active_check_interval(self, iteration: int) -> int:
        """Return the active convergence-check interval for *iteration*."""
        if not self.check_interval_growth:
            return self.check_interval

        iter_num = iteration + 1
        if iter_num <= 20:
            return self.check_interval
        if iter_num <= 100:
            return min(self.max_check_interval, max(self.check_interval, 10))
        return self.max_check_interval

    def _should_check_convergence(self, iteration: int) -> bool:
        """Return whether convergence should be checked at *iteration*."""
        iter_num = iteration + 1
        if iteration == self.max_iterations - 1:
            return True
        if iter_num < self.min_iterations_before_check:
            return False
        return iter_num % self._active_check_interval(iteration) == 0

    # -------------------------------------------------------------------------
    # FETransformer interface
    # -------------------------------------------------------------------------

    def fit_transform(self, variables: List[str], where_clause: str = "") -> str:
        """Demean *variables* via MAP and store the result in ``demeaned_data``."""
        self._resid_name_map = {
            variable: f"_resid_{idx}" for idx, variable in enumerate(variables)
        }
        self._build_fe_code_map()
        self.n_iterations = None

        logger.debug(
            "Starting iterative demeaning for table=%s with %d variable(s), %d FE(s), "
            "remove_singletons=%s, tolerance=%.2e, max_iterations=%d, "
            "check_interval=%d, convergence_sample=%.3f",
            self.table_name,
            len(variables),
            len(self.fe_cols),
            self.remove_singletons,
            self.tolerance,
            self.max_iterations,
            self.check_interval,
            self.convergence_sample,
        )

        self._init_demeaned_table(variables, where_clause)

        if self.n_obs == 0:
            self.n_iterations = 0
            logger.debug(
                "No observations remain after filtering; skipping MAP and finalizing empty output"
            )
            self._finalize_result_from_source(self._RESULT_TABLE)
            self._compute_fe_levels()
            self._fitted = True
            logger.debug(
                "Finished iterative demeaning: result_table=%s, n_obs=%d, iterations=%d",
                self._RESULT_TABLE,
                self.n_obs,
                self.n_iterations,
            )
            return self._RESULT_TABLE

        resid_cols = self._resid_cols(variables)
        map_resid_cols = self._filter_map_resid_cols(resid_cols)

        if not self.fe_cols:
            self.n_iterations = 0
            logger.debug(
                "No FE dimensions provided; skipping MAP iterations and finalizing pass-through output"
            )
            self._finalize_result_from_source(self._RESULT_TABLE)
            self._compute_fe_levels()
            self._fitted = True
            logger.debug(
                "Finished iterative demeaning: result_table=%s, n_obs=%d, iterations=%d",
                self._RESULT_TABLE,
                self.n_obs,
                self.n_iterations,
            )
            return self._RESULT_TABLE

        self._configure_map_fe_order()
        logger.debug("Prepared MAP residual columns: %s", ", ".join(map_resid_cols))

        if not map_resid_cols:
            self.n_iterations = 0
            logger.debug("All residual columns are constant; skipping MAP iterations")
            self._finalize_result_from_source(self._RESULT_TABLE)
            self._compute_fe_levels()
            self._fitted = True
            logger.debug(
                "Finished iterative demeaning: result_table=%s, n_obs=%d, iterations=%d",
                self._RESULT_TABLE,
                self.n_obs,
                self.n_iterations,
            )
            return self._RESULT_TABLE

        if len(self.fe_cols) == 1:
            self._run_oneway_exact(map_resid_cols)
            self._compute_fe_levels()
            self._fitted = True
            logger.debug(
                "Finished iterative demeaning via one-way shortcut: "
                "result_table=%s, n_obs=%d, iterations=%d",
                self._RESULT_TABLE,
                self.n_obs,
                self.n_iterations,
            )
            return self._RESULT_TABLE

        if self._is_complete_cartesian_panel():
            self._run_balanced_cartesian_exact(map_resid_cols)
            self._compute_fe_levels()
            self._fitted = True
            logger.debug(
                "Finished iterative demeaning via balanced Cartesian shortcut: "
                "result_table=%s, n_obs=%d, iterations=%d",
                self._RESULT_TABLE,
                self.n_obs,
                self.n_iterations,
            )
            return self._RESULT_TABLE

        self._run_map(map_resid_cols)
        self._compute_fe_levels()
        self._fitted = True
        logger.debug(
            "Finished iterative demeaning: result_table=%s, n_obs=%d, iterations=%s",
            self._RESULT_TABLE,
            self.n_obs,
            self.n_iterations,
        )
        return self._RESULT_TABLE

    def transform_query(self, variables: List[str]) -> str:
        """Return transformed residual columns aliased back to original names."""
        if not self._resid_name_map:
            raise RuntimeError("fit_transform() has not been called yet.")

        missing = [variable for variable in variables if variable not in self._resid_name_map]
        if missing:
            raise ValueError(
                f"transform_query() received unknown variable(s): {', '.join(missing)}"
            )

        return ", ".join(
            f"{self.qident(self._resid_name_map[variable])} AS {self.qident(variable)}"
            for variable in variables
        )

    def residual_column_name(self, variable: str) -> str:
        """Return the internal residual-column name for *variable*."""
        if variable not in self._resid_name_map:
            raise ValueError(f"Unknown transformed variable: {variable}")
        return self._resid_name_map[variable]

    @property
    def n_obs(self) -> int:
        """Observations in ``demeaned_data`` after filtering."""
        if self._n_obs is None:
            raise RuntimeError("fit_transform() has not been called yet.")
        return self._n_obs

    @property
    def df_correction(self) -> int:
        """Total distinct FE levels used for DOF adjustment."""
        return self._fe_total_levels

    @property
    def extra_regressors(self) -> List[str]:
        """Iterative demeaning adds no new regressors."""
        return []

    @property
    def has_intercept(self) -> bool:
        """Demeaning absorbs the constant term."""
        return False

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _init_demeaned_table(self, variables: List[str], where_clause: str) -> None:
        """Build the working table after null filtering and singleton pruning."""
        result_table_sql = self.qident(self._RESULT_TABLE)
        # table_name may be a plain identifier or a DuckDB relation expression
        # such as read_parquet('...'), so it must not be identifier-quoted.
        source_table_sql = self._source_relation_sql(self.table_name)
        row_id_sql = self.qident(self._ROW_ID_COL)
        singleton_work_sql = self.qident("_singleton_work")
        singleton_next_sql = self.qident("_singleton_next")

        self._drop_tables_if_exist(["_singleton_work", "_singleton_next"])
        self.n_rows_dropped_singletons = 0

        select_parts = [f"ROW_NUMBER() OVER () AS {row_id_sql}"]
        select_parts.extend(self.qident(fe_col) for fe_col in self.fe_cols)
        if self.cluster_col:
            select_parts.append(self.qident(self.cluster_col))
        for variable in variables:
            select_parts.append(
                f"CAST({self.qident(variable)} AS {self.residual_type}) AS "
                f"{self.qident(self._resid_name_map[variable])}"
            )

        all_model_cols = list(self.fe_cols) + list(variables)
        if self.cluster_col:
            all_model_cols.append(self.cluster_col)
        null_filter = " AND ".join(
            f"{self.qident(col)} IS NOT NULL" for col in all_model_cols
        )
        combined_where = (
            f"{where_clause} AND {null_filter}" if where_clause else f"WHERE {null_filter}"
        )

        logger.debug(
            "Building initial demeaned table from %s with variables=%s and where_clause=%r",
            self.table_name,
            ", ".join(variables),
            where_clause,
        )

        try:
            self.conn.execute(
                f"""
                CREATE OR REPLACE TABLE {singleton_work_sql} AS
                SELECT {', '.join(select_parts)}
                FROM {source_table_sql}
                {combined_where}
                """
            )

            rows_before_pruning = self.conn.execute(
                f"SELECT COUNT(*) FROM {singleton_work_sql}"
            ).fetchone()[0]
            logger.debug(
                "Rows before singleton pruning after null filtering: %d",
                rows_before_pruning,
            )

            if self.remove_singletons and self.fe_cols:
                logger.debug(
                    "Applying singleton pruning with strategy=%s across %d FE dimension(s)",
                    self.singleton_pruning,
                    len(self.fe_cols),
                )
                rows_before_pass = rows_before_pruning
                pass_num = 0

                while True:
                    pass_num += 1
                    valid_group_joins = []
                    for idx, fe_col in enumerate(self.fe_cols):
                        fe_sql = self.qident(fe_col)
                        valid_group_joins.append(
                            f"""
                            JOIN (
                                SELECT {fe_sql}
                                FROM {singleton_work_sql}
                                GROUP BY {fe_sql}
                                HAVING COUNT(*) > 1
                            ) _valid_fe_{idx}
                            ON w.{fe_sql} = _valid_fe_{idx}.{fe_sql}
                            """
                        )

                    self.conn.execute(
                        f"""
                        CREATE OR REPLACE TABLE {singleton_next_sql} AS
                        SELECT w.*
                        FROM {singleton_work_sql} w
                        {' '.join(valid_group_joins)}
                        """
                    )

                    rows_after_pass = self.conn.execute(
                        f"SELECT COUNT(*) FROM {singleton_next_sql}"
                    ).fetchone()[0]
                    rows_removed = rows_before_pass - rows_after_pass
                    self.n_rows_dropped_singletons += rows_removed

                    logger.debug(
                        "Singleton pruning pass %d: removed=%d, remaining=%d",
                        pass_num,
                        rows_removed,
                        rows_after_pass,
                    )

                    if self.singleton_pruning == "one_pass":
                        if rows_removed > 0:
                            self.conn.execute(f"DROP TABLE {singleton_work_sql}")
                            self.conn.execute(
                                f"ALTER TABLE {singleton_next_sql} RENAME TO "
                                f"{self.qident('_singleton_work')}"
                            )
                        else:
                            self.conn.execute(
                                f"DROP TABLE IF EXISTS {singleton_next_sql}"
                            )
                        logger.debug(
                            "Singleton pruning strategy=one_pass: removed=%d, "
                            "remaining=%d; no cascading passes were performed",
                            self.n_rows_dropped_singletons,
                            rows_after_pass if rows_removed > 0 else rows_before_pass,
                        )
                        break

                    if rows_removed == 0:
                        self.conn.execute(f"DROP TABLE IF EXISTS {singleton_next_sql}")
                        break

                    self.conn.execute(f"DROP TABLE {singleton_work_sql}")
                    self.conn.execute(
                        f"ALTER TABLE {singleton_next_sql} RENAME TO "
                        f"{self.qident('_singleton_work')}"
                    )
                    rows_before_pass = rows_after_pass

                logger.debug(
                    "Total singleton rows removed with strategy=%s: %d",
                    self.singleton_pruning,
                    self.n_rows_dropped_singletons,
                )
            else:
                logger.debug(
                    "Skipping singleton pruning; using null-filtered sample directly"
                )

            self.conn.execute(
                f"""
                CREATE OR REPLACE TABLE {result_table_sql} AS
                SELECT *
                FROM {singleton_work_sql}
                """
            )
        finally:
            self._drop_tables_if_exist(["_singleton_work", "_singleton_next"])

        self._n_obs = self.conn.execute(
            f"SELECT COUNT(*) FROM {result_table_sql}"
        ).fetchone()[0]
        logger.debug(
            "Initial MAP working table ready: FE dims=%d, variables=%d, n_obs=%d",
            len(self.fe_cols),
            len(variables),
            self._n_obs,
        )

    def _configure_map_fe_order(self) -> None:
        """Compute FE distinct counts and choose the MAP sweep order."""
        if not self.fe_cols:
            self._map_fe_code_order = []
            return

        result_table_sql = self.qident(self._RESULT_TABLE)
        count_aliases = [f"_n_groups_{idx}" for idx, _ in enumerate(self.fe_cols)]
        count_query = ", ".join(
            f"COUNT(DISTINCT {self.qident(fe_col)}) AS {self.qident(alias)}"
            for fe_col, alias in zip(self.fe_cols, count_aliases)
        )
        counts_row = self.conn.execute(
            f"SELECT {count_query} FROM {result_table_sql}"
        ).fetchone()
        fe_counts = {
            spec["fe_col"]: int(counts_row[idx] or 0)
            for idx, spec in enumerate(self._fe_code_map)
        }
        self._fe_level_counts = fe_counts

        original_order = [spec["fe_col"] for spec in self._fe_code_map]
        if self.fe_order == "input":
            ordered_specs = list(self._fe_code_map)
        elif self.fe_order == "ascending_groups":
            ordered_specs = sorted(
                self._fe_code_map,
                key=lambda spec: (fe_counts[spec["fe_col"]], original_order.index(spec["fe_col"])),
            )
        else:
            ordered_specs = sorted(
                self._fe_code_map,
                key=lambda spec: (-fe_counts[spec["fe_col"]], original_order.index(spec["fe_col"])),
            )

        self._map_fe_code_order = ordered_specs
        logger.debug(
            "MAP FE ordering: original=%s, distinct_counts=%s, mode=%s, chosen=%s",
            original_order,
            fe_counts,
            self.fe_order,
            [spec["fe_col"] for spec in self._map_fe_code_order],
        )

    def _filter_map_resid_cols(self, resid_cols: List[str]) -> List[str]:
        """Drop constant residual columns from MAP and zero them when needed."""
        if not self.drop_constant_variables or not resid_cols:
            return resid_cols

        result_table_sql = self.qident(self._RESULT_TABLE)
        agg_parts = []
        for idx, resid_col in enumerate(resid_cols):
            agg_parts.append(
                f"MIN({self.qident(resid_col)}) AS {self.qident(f'_min_{idx}')}"
            )
            agg_parts.append(
                f"MAX({self.qident(resid_col)}) AS {self.qident(f'_max_{idx}')}"
            )
        stats = self.conn.execute(
            f"SELECT {', '.join(agg_parts)} FROM {result_table_sql}"
        ).fetchone()

        constant_cols = []
        active_cols = []
        for idx, resid_col in enumerate(resid_cols):
            col_min = stats[2 * idx]
            col_max = stats[2 * idx + 1]
            if col_min is None or col_min == col_max:
                constant_cols.append(resid_col)
            else:
                active_cols.append(resid_col)

        if not constant_cols:
            return resid_cols

        constant_vars = [
            variable
            for variable, resid_col in self._resid_name_map.items()
            if resid_col in constant_cols
        ]
        logger.debug(
            "Dropping constant residual columns from MAP: vars=%s, cols=%s",
            constant_vars,
            constant_cols,
        )

        if self.fe_cols and self.n_obs > 0:
            update_expr = ", ".join(
                f"{self.qident(resid_col)} = 0.0" for resid_col in constant_cols
            )
            self.conn.execute(
                f"UPDATE {result_table_sql} SET {update_expr}"
            )
            logger.debug(
                "Set constant residual columns to zero before MAP because FE dimensions are present"
            )

        return active_cols

    def _is_complete_cartesian_panel(self) -> bool:
        """Return whether every FE combination appears exactly once."""
        if len(self.fe_cols) < 2 or self.n_obs == 0 or not self._fe_level_counts:
            return False

        expected_cells = 1
        for fe_col in self.fe_cols:
            expected_cells *= self._fe_level_counts.get(fe_col, 0)
        if expected_cells != self.n_obs:
            return False

        result_table_sql = self.qident(self._RESULT_TABLE)
        fe_select = ", ".join(self.qident(fe_col) for fe_col in self.fe_cols)
        combo_count = self.conn.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT {fe_select}
                FROM {result_table_sql}
                GROUP BY {fe_select}
            ) _fe_combos
            """
        ).fetchone()[0]
        is_complete = combo_count == self.n_obs
        logger.debug(
            "Balanced Cartesian FE check: expected_cells=%d, observed_combos=%d, "
            "n_obs=%d, complete=%s",
            expected_cells,
            combo_count,
            self.n_obs,
            is_complete,
        )
        return is_complete

    def _replace_result_with_select(self, select_parts: List[str]) -> None:
        """Replace ``demeaned_data`` with a final projection."""
        result_table_sql = self.qident(self._RESULT_TABLE)
        final_table_sql = self.qident("_result_final")
        try:
            self.conn.execute(
                f"""
                CREATE OR REPLACE TABLE {final_table_sql} AS
                SELECT {', '.join(select_parts)}
                FROM {result_table_sql} src
                """
            )
            self.conn.execute(f"DROP TABLE {result_table_sql}")
            self.conn.execute(
                f"ALTER TABLE {final_table_sql} RENAME TO {self.qident(self._RESULT_TABLE)}"
            )
        finally:
            self.conn.execute(f"DROP TABLE IF EXISTS {final_table_sql}")

    def _run_oneway_exact(self, resid_cols: List[str]) -> None:
        """Demean exactly in one pass for a single FE dimension."""
        fe_sql = self.qident(self.fe_cols[0])
        active_resid_cols = set(resid_cols)
        select_parts = [f"src.{self.qident(fe_col)}" for fe_col in self.fe_cols]
        if self.cluster_col:
            select_parts.append(f"src.{self.qident(self.cluster_col)}")
        for resid_col in self._resid_name_map.values():
            resid_sql = self.qident(resid_col)
            if resid_col in active_resid_cols:
                select_parts.append(
                    f"src.{resid_sql} - AVG(src.{resid_sql}) OVER "
                    f"(PARTITION BY src.{fe_sql}) AS {resid_sql}"
                )
            else:
                select_parts.append(f"src.{resid_sql}")

        self._replace_result_with_select(select_parts)
        self.n_iterations = 1
        logger.debug("Applied exact one-way FE shortcut")

    def _run_balanced_cartesian_exact(self, resid_cols: List[str]) -> None:
        """Demean exactly for complete one-observation-per-cell FE grids."""
        active_resid_cols = set(resid_cols)
        n_fe = len(self.fe_cols)
        partition_clauses = [
            f"PARTITION BY src.{self.qident(fe_col)}" for fe_col in self.fe_cols
        ]

        select_parts = [f"src.{self.qident(fe_col)}" for fe_col in self.fe_cols]
        if self.cluster_col:
            select_parts.append(f"src.{self.qident(self.cluster_col)}")
        for resid_col in self._resid_name_map.values():
            resid_sql = self.qident(resid_col)
            if resid_col in active_resid_cols:
                fe_mean_terms = " + ".join(
                    f"AVG(src.{resid_sql}) OVER ({partition_clause})"
                    for partition_clause in partition_clauses
                )
                select_parts.append(
                    f"src.{resid_sql} - ({fe_mean_terms}) + "
                    f"{n_fe - 1} * AVG(src.{resid_sql}) OVER () AS {resid_sql}"
                )
            else:
                select_parts.append(f"src.{resid_sql}")

        self._replace_result_with_select(select_parts)
        self.n_iterations = 1
        logger.debug("Applied exact balanced Cartesian FE shortcut")

    def _build_convergence_sql(self, resid_cols: List[str], source_relation: str) -> str:
        """Build SQL for the maximum absolute remaining FE-group mean."""
        dim_parts = []
        for spec in self._fe_code_map:
            code_sql = self.qident(spec["code_col"])
            avg_aliases = [f"_avg_{idx}" for idx, _ in enumerate(resid_cols)]
            dim_parts.append(
                "SELECT UNNEST(["
                + ", ".join(self.qident(alias) for alias in avg_aliases)
                + "]) AS avg_val "
                "FROM (SELECT "
                + ", ".join(
                    f"AVG({self.qident(resid_col)}) AS {self.qident(avg_aliases[idx])}"
                    for idx, resid_col in enumerate(resid_cols)
                )
                + f" FROM {source_relation} GROUP BY {code_sql}) _agg"
            )

        return (
            "SELECT MAX(ABS(avg_val)) AS max_group_mean "
            f"FROM ({' UNION ALL '.join(dim_parts)}) t"
        )

    def _build_group_sampled_convergence_sql(
        self, resid_cols: List[str], source_relation: str
    ) -> str:
        """Build convergence SQL using complete deterministic FE-group samples."""
        modulo = 1_000_000
        threshold = max(1, min(modulo, int(self.convergence_sample * modulo)))
        ctes = []
        union_parts = []
        count_parts = []

        for dim_idx, spec in enumerate(self._fe_code_map):
            code_sql = self.qident(spec["code_col"])
            avg_aliases = [f"_avg_{idx}" for idx, _ in enumerate(resid_cols)]
            cte_name = f"_sample_dim_{dim_idx}"
            ctes.append(
                f"{cte_name} AS ("
                f"SELECT {code_sql}, "
                + ", ".join(
                    f"AVG({self.qident(resid_col)}) AS {self.qident(avg_aliases[idx])}"
                    for idx, resid_col in enumerate(resid_cols)
                )
                + f" FROM {source_relation} "
                f"WHERE hash({code_sql}) % {modulo} < {threshold} "
                f"GROUP BY {code_sql})"
            )
            union_parts.append(
                "SELECT UNNEST(["
                + ", ".join(self.qident(alias) for alias in avg_aliases)
                + f"]) AS avg_val FROM {cte_name}"
            )
            count_parts.append(f"SELECT COUNT(*) AS n_groups FROM {cte_name}")

        return (
            f"WITH {', '.join(ctes)}, "
            "_sample_counts AS ("
            f"SELECT MIN(n_groups) AS min_groups FROM ({' UNION ALL '.join(count_parts)}) _counts"
            ") "
            "SELECT CASE "
            "WHEN (SELECT min_groups FROM _sample_counts) = 0 THEN NULL "
            "ELSE ("
            "SELECT MAX(ABS(avg_val)) "
            f"FROM ({' UNION ALL '.join(union_parts)}) _sample_avgs"
            ") END AS max_group_mean"
        )

    def _measure_max_group_mean(
        self,
        exact: bool,
        conv_sql_exact: str,
        conv_sql_sampled: Optional[str],
    ) -> float:
        """Execute the requested convergence query and return ``inf`` on NULL."""
        sql = conv_sql_exact if exact else conv_sql_sampled
        if sql is None:
            raise ValueError("Sampled convergence SQL was requested but not built.")

        result = self.conn.execute(sql).fetchone()
        if result is None or result[0] is None:
            return float("inf")
        return float(result[0])

    def _finalize_result_from_source(self, source_table: str) -> None:
        """Rebuild ``demeaned_data`` without the internal row-id column."""
        result_table_sql = self.qident(self._RESULT_TABLE)
        source_table_sql = self.qident(source_table)
        final_table_sql = self.qident("_result_final")

        select_parts = [self.qident(fe_col) for fe_col in self.fe_cols]
        if self.cluster_col:
            select_parts.append(self.qident(self.cluster_col))
        select_parts.extend(
            self.qident(resid_col) for resid_col in self._resid_name_map.values()
        )

        try:
            self.conn.execute(
                f"""
                CREATE OR REPLACE TABLE {final_table_sql} AS
                SELECT {', '.join(select_parts)}
                FROM {source_table_sql}
                """
            )
            if source_table == self._RESULT_TABLE:
                self.conn.execute(f"DROP TABLE {result_table_sql}")
                self.conn.execute(
                    f"ALTER TABLE {final_table_sql} RENAME TO {self.qident(self._RESULT_TABLE)}"
                )
            else:
                self.conn.execute(
                    f"""
                    CREATE OR REPLACE TABLE {result_table_sql} AS
                    SELECT *
                    FROM {final_table_sql}
                    """
                )
        finally:
            self.conn.execute(f"DROP TABLE IF EXISTS {final_table_sql}")
        logger.debug(
            "Finalized %s from %s without exposing %s",
            self._RESULT_TABLE,
            source_table,
            self._ROW_ID_COL,
        )

    def _run_map(self, resid_cols: List[str]) -> None:
        """Execute MAP until the exact FE-group-mean criterion converges."""
        result_table_sql = self.qident(self._RESULT_TABLE)
        row_id_sql = self.qident(self._ROW_ID_COL)
        fe_store_sql = self.qident("_fe_store")
        resid_store_sql = self.qident("_resid_store")
        resid_new_sql = self.qident("_resid_new")

        logger.debug(
            "Preparing MAP scratch tables for %d FE dimension(s) and %d residual column(s)",
            len(self.fe_cols),
            len(resid_cols),
        )

        self._drop_tables_if_exist(self._known_tables())

        max_group_mean = float("inf")

        try:
            fe_select_parts = [row_id_sql]
            fe_select_parts.extend(self.qident(fe_col) for fe_col in self.fe_cols)
            if self.cluster_col:
                fe_select_parts.append(self.qident(self.cluster_col))
            self.conn.execute(
                f"""
                CREATE OR REPLACE TABLE {fe_store_sql} AS
                SELECT {', '.join(fe_select_parts)}
                FROM {result_table_sql}
                """
            )
            logger.debug("Created scratch table _fe_store")

            for spec in self._fe_code_map:
                fe_sql = self.qident(spec["fe_col"])
                dict_table_sql = self.qident(spec["dict_table"])
                code_sql = self.qident(spec["code_col"])
                logger.debug(
                    "Encoding FE dimension %s into compact integer codes using %s",
                    spec["fe_col"],
                    spec["dict_table"],
                )
                self.conn.execute(
                    f"""
                    CREATE OR REPLACE TABLE {dict_table_sql} AS
                    SELECT {fe_sql},
                           CAST(ROW_NUMBER() OVER (ORDER BY {fe_sql}) - 1 AS UINTEGER)
                               AS {code_sql}
                    FROM (SELECT DISTINCT {fe_sql} FROM {fe_store_sql}) t
                    """
                )

            code_select = ", ".join(
                f"d{idx}.{self.qident(spec['code_col'])}"
                for idx, spec in enumerate(self._fe_code_map)
            )
            resid_select = ", ".join(
                f"src.{self.qident(resid_col)}" for resid_col in resid_cols
            )
            code_joins = " ".join(
                f"JOIN {self.qident(spec['dict_table'])} d{idx} "
                f"ON src.{self.qident(spec['fe_col'])} = d{idx}.{self.qident(spec['fe_col'])}"
                for idx, spec in enumerate(self._fe_code_map)
            )
            self.conn.execute(
                f"""
                CREATE OR REPLACE TABLE {resid_store_sql} AS
                SELECT src.{row_id_sql}, {code_select}, {resid_select}
                FROM {result_table_sql} src
                {code_joins}
                """
            )
            logger.debug("Created residual scratch table _resid_store")

            conv_sql_exact = self._build_convergence_sql(resid_cols, "_resid_store")
            conv_sql_sampled = None
            if self.convergence_sample < 1.0:
                conv_sql_sampled = self._build_group_sampled_convergence_sql(
                    resid_cols, "_resid_store"
                )
                logger.debug(
                    "Prepared FE-group sampled and exact convergence queries "
                    "(sample=%.3f)",
                    self.convergence_sample,
                )
            else:
                logger.debug("Prepared exact convergence query")

            avg_exprs = ", ".join(
                f"AVG({self.qident(resid_col)}) AS {self.qident(f'_avg_{idx}')}"
                for idx, resid_col in enumerate(resid_cols)
            )
            carry_code_select = ", ".join(
                f"r.{self.qident(spec['code_col'])}" for spec in self._fe_code_map
            )

            for iteration in range(self.max_iterations):
                should_check = self._should_check_convergence(iteration)
                active_interval = self._active_check_interval(iteration)
                if should_check or (iteration + 1) % 10 == 0:
                    logger.debug(
                        "MAP iteration %d: should_check=%s, active_check_interval=%d",
                        iteration + 1,
                        should_check,
                        active_interval,
                    )

                for spec in self._map_fe_code_order:
                    code_sql = self.qident(spec["code_col"])
                    sub_exprs = ", ".join(
                        f"r.{self.qident(resid_col)} - m.{self.qident(f'_avg_{idx}')}"
                        f" AS {self.qident(resid_col)}"
                        for idx, resid_col in enumerate(resid_cols)
                    )
                    self.conn.execute(
                        f"""
                        CREATE OR REPLACE TABLE {resid_new_sql} AS
                        SELECT r.{row_id_sql}, {carry_code_select}, {sub_exprs}
                        FROM {resid_store_sql} r
                        JOIN (
                            SELECT {code_sql}, {avg_exprs}
                            FROM {resid_store_sql}
                            GROUP BY {code_sql}
                        ) m USING ({code_sql})
                        """
                    )

                    self.conn.execute(f"DROP TABLE {resid_store_sql}")
                    self.conn.execute(
                        f"ALTER TABLE {resid_new_sql} RENAME TO "
                        f"{self.qident('_resid_store')}"
                    )

                if not should_check:
                    continue

                logger.debug("Iteration %d: running convergence check", iteration + 1)
                if self.convergence_sample < 1.0:
                    sampled_group_mean = self._measure_max_group_mean(
                        exact=False,
                        conv_sql_exact=conv_sql_exact,
                        conv_sql_sampled=conv_sql_sampled,
                    )
                    logger.debug(
                        "Iteration %d: sampled max_group_mean=%s",
                        iteration + 1,
                        (
                            f"{sampled_group_mean:.2e}"
                            if sampled_group_mean != float("inf")
                            else "inf"
                        ),
                    )

                    if (
                        sampled_group_mean != float("inf")
                        and sampled_group_mean >= self.tolerance
                    ):
                        max_group_mean = sampled_group_mean
                        continue

                    if sampled_group_mean == float("inf"):
                        logger.debug(
                            "Iteration %d: sampled screen was empty; running exact "
                            "convergence check",
                            iteration + 1,
                        )
                    else:
                        logger.debug(
                            "Iteration %d: sampled screen passed; running exact "
                            "convergence check",
                            iteration + 1,
                        )

                max_group_mean = self._measure_max_group_mean(
                    exact=True,
                    conv_sql_exact=conv_sql_exact,
                    conv_sql_sampled=conv_sql_sampled,
                )
                logger.debug(
                    "Iteration %d: exact max_group_mean=%.2e",
                    iteration + 1,
                    max_group_mean,
                )
                if max_group_mean < self.tolerance:
                    self.n_iterations = iteration + 1
                    logger.debug(
                        "MAP converged after %d iteration(s) with exact max_group_mean=%.2e",
                        self.n_iterations,
                        max_group_mean,
                    )
                    break
            else:
                self.n_iterations = self.max_iterations
                logger.warning(
                    "MAP did not converge after %d iterations "
                    "(max_group_mean=%.2e, tolerance=%.2e)",
                    self.max_iterations,
                    max_group_mean,
                    self.tolerance,
                )
        finally:
            logger.debug("MAP cleanup starting")
            if self._table_exists("_fe_store") and self._table_exists("_resid_store"):
                logger.debug("Reconstructing %s from MAP scratch tables", self._RESULT_TABLE)
                fe_exprs = ", ".join(f"f.{self.qident(fe_col)}" for fe_col in self.fe_cols)
                cluster_expr = (
                    f", f.{self.qident(self.cluster_col)}" if self.cluster_col else ""
                )
                active_resid_cols = set(resid_cols)
                all_resid_cols = set(self._resid_name_map.values())
                if active_resid_cols == all_resid_cols:
                    resid_exprs = ", ".join(
                        f"r.{self.qident(resid_col)}"
                        for resid_col in self._resid_name_map.values()
                    )
                    self.conn.execute(
                        f"""
                        CREATE OR REPLACE TABLE {result_table_sql} AS
                        SELECT {fe_exprs}{cluster_expr}, {resid_exprs}
                        FROM {fe_store_sql} f
                        JOIN {resid_store_sql} r
                          ON f.{row_id_sql} = r.{row_id_sql}
                        """
                    )
                else:
                    resid_exprs = ", ".join(
                        (
                            f"r.{self.qident(resid_col)}"
                            if resid_col in active_resid_cols
                            else f"src.{self.qident(resid_col)}"
                        )
                        for resid_col in self._resid_name_map.values()
                    )
                    self.conn.execute(
                        f"""
                        CREATE OR REPLACE TABLE {result_table_sql} AS
                        SELECT {fe_exprs}{cluster_expr}, {resid_exprs}
                        FROM {fe_store_sql} f
                        JOIN {resid_store_sql} r
                          ON f.{row_id_sql} = r.{row_id_sql}
                        JOIN {result_table_sql} src
                          ON f.{row_id_sql} = src.{row_id_sql}
                        """
                    )
            self._drop_tables_if_exist(
                ["_fe_store", "_resid_store", "_resid_new"]
                + [spec["dict_table"] for spec in self._fe_code_map]
            )
            logger.debug("MAP cleanup finished")

    def _compute_fe_levels(self) -> None:
        """Count distinct levels across all FE dimensions in one query."""
        if not self.fe_cols:
            self._fe_total_levels = 0
            logger.debug("No FE dimensions provided; df_correction set to 0")
            return

        if self._fe_level_counts:
            self._fe_total_levels = sum(self._fe_level_counts.values())
        else:
            result_table_sql = self.qident(self._RESULT_TABLE)
            sum_expr = " + ".join(
                f"COUNT(DISTINCT {self.qident(fe_col)})" for fe_col in self.fe_cols
            )
            result = self.conn.execute(
                f"SELECT {sum_expr} AS total FROM {result_table_sql}"
            ).fetchone()
            self._fe_total_levels = result[0] or 0
        logger.debug(
            "FE levels: %d total across %d dim(s)",
            self._fe_total_levels,
            len(self.fe_cols),
        )
