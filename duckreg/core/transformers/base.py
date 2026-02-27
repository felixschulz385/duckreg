"""Abstract base class for fixed-effects transformers."""
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import duckdb

logger = logging.getLogger(__name__)


class FETransformer(ABC):
    """Abstract base class for fixed-effects transformers.

    A transformer encapsulates the data transformation needed to absorb fixed
    effects.  It operates on raw SQL column names and a DuckDB connection —
    it has no knowledge of formula parsing or coefficient estimation.

    The interface is intentionally minimal so that different estimators
    (``DuckFE``) can share the same
    transformation logic without coupling.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    table_name : str
        Name of the source table/view in DuckDB.
    fe_cols : List[str]
        SQL column names of the fixed-effect dimensions.
    cluster_col : str, optional
        SQL column name used for clustering (may be an alias).
    remove_singletons : bool
        Whether to drop observations that are the sole member of a FE group.
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        table_name: str,
        fe_cols: List[str],
        cluster_col: Optional[str] = None,
        remove_singletons: bool = True,
        **kwargs,
    ):
        self.conn = conn
        self.table_name = table_name
        self.fe_cols = list(fe_cols)
        self.cluster_col = cluster_col
        self.remove_singletons = remove_singletons
        self.n_rows_dropped_singletons: int = 0
        self._n_obs: Optional[int] = None
        self._fitted: bool = False

    # -------------------------------------------------------------------------
    # Abstract interface
    # -------------------------------------------------------------------------

    @abstractmethod
    def fit_transform(self, variables: List[str], where_clause: str = "") -> str:
        """Transform *variables* and return the name of the result table/view.

        Parameters
        ----------
        variables : List[str]
            SQL column names to include and/or transform.  These are the data
            columns (outcomes + covariates) **excluding** FE columns and the
            cluster column, which are handled separately via constructor args.
        where_clause : str
            SQL WHERE clause (including the ``WHERE`` keyword) used to filter
            the source table, or an empty string for no filtering.

        Returns
        -------
        str
            Name of the DuckDB table or view that holds the transformed data.
            Subsequent calls to :meth:`transform_query` produce ``SELECT``
            fragments that can be applied against this table.
        """

    @abstractmethod
    def transform_query(self, variables: List[str]) -> str:
        """Return a SQL ``SELECT`` fragment mapping original names to transformed names.

        The fragment can be embedded directly in a ``SELECT`` list:

        .. code-block:: sql

            SELECT {transformer.transform_query(variables)}, count
            FROM {result_table}

        Parameters
        ----------
        variables : List[str]
            Original SQL column names (same list passed to :meth:`fit_transform`).

        Returns
        -------
        str
            Comma-separated SQL expressions.  Examples:

            * Iterative demean: ``"resid_x AS x, resid_y AS y"``
            * Mundlak (identity): ``"x, y"``
        """

    @property
    @abstractmethod
    def n_obs(self) -> int:
        """Number of observations after filtering (e.g., singleton removal)."""

    @property
    @abstractmethod
    def df_correction(self) -> int:
        """Degrees of freedom consumed by the transformation.

        * **Iterative demean** — sum of distinct FE levels across all
          dimensions (FEs are absorbed and not counted in ``k``).
        * **Mundlak** — ``0`` (FE parameters appear as explicit regressors
          in the design matrix and are already counted in ``k``).
        """

    @property
    @abstractmethod
    def extra_regressors(self) -> List[str]:
        """New SQL column names added to the design matrix by this transformation.

        * **Iterative demean** — ``[]`` (variables are demeaned in-place;
          no new columns created).
        * **Mundlak** — Mundlak mean columns, fixed-FE dummy columns,
          and unbalanced-panel correction columns.

        The estimator adds these to the right-hand side of the regression.
        """

    @property
    @abstractmethod
    def has_intercept(self) -> bool:
        """Whether the transformed model requires an intercept term.

        * **Iterative demean** — ``False`` (demeaning absorbs the constant).
        * **Mundlak** — ``True`` (levels model with explicit FE controls).
        """

    # -------------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------------

    def _remove_singleton_observations(self, table_name: str) -> None:
        """Drop singleton FE groups from *table_name* (in-place replacement).

        Processes each FE dimension sequentially.  Removing singletons in one
        dimension can create new singletons in another, so multiple passes may
        be needed for strict singleton-free panels (not implemented here — one
        pass per dimension matches the behaviour expected by callers).

        Parameters
        ----------
        table_name : str
            Name of the DuckDB table to filter (modified in place).
        """
        if not self.remove_singletons or not self.fe_cols:
            return

        logger.debug(
            f"Removing singleton FE observations from {len(self.fe_cols)} FE groups"
        )

        rows_before = self.conn.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]

        for fe_sql in self.fe_cols:
            self.conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT *
            FROM {table_name}
            ANTI JOIN (
                SELECT {fe_sql}
                FROM {table_name}
                GROUP BY {fe_sql}
                HAVING COUNT(*) < 2
            ) singletons
            USING ({fe_sql})
            """)

        rows_after = self.conn.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]
        self.n_rows_dropped_singletons = rows_before - rows_after

        logger.debug(
            f"After singleton removal: {rows_after} observations "
            f"({self.n_rows_dropped_singletons} rows removed)"
        )
