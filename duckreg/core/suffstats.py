"""
Sufficient statistics computation for weighted least squares.

Extracted from DuckDBFitter for reuse and clarity.

This module also hosts all DuckDB *execution* helpers — functions that accept
a ``duckdb.DuckDBPyConnection`` and call ``.execute()``.  Keeping them here
(rather than in ``sql_builders``) ensures that ``sql_builders`` remains a
pure string-production module with no database dependencies.

Public API
----------
- ``SuffStats`` — dataclass returned by both compute_sufficient_stats_* functions.
  Supports tuple unpacking for backward compatibility.
- ``compute_sufficient_stats_numpy`` — NumPy backend.
- ``compute_sufficient_stats_sql``   — DuckDB SQL backend (DRY: delegates SQL
  string construction to ``build_xtx_query`` / ``build_xty_query``).
- ``execute_to_matrix``              — Execute a SQL query → numpy matrix.
- ``compute_cross_sufficient_stats_sql`` — IV cross-product stats (X'Z, Z'Z).
- ``profile_fe_column``              — Profile a FE column for classification.
- ``get_fe_unique_levels``           — Retrieve sorted unique FE levels.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple
import duckdb

from .sql_builders import (
    build_xtx_query,
    build_xty_query,
    build_cross_xtz_query,
)
from .linalg import DEFAULT_ALPHA

logger = logging.getLogger(__name__)


# ============================================================================
# SuffStats Dataclass
# ============================================================================

@dataclass
class SuffStats:
    """Container for sufficient statistics returned by compute_sufficient_stats_*.

    Supports tuple unpacking for backward compatibility::

        XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names = compute_sufficient_stats_numpy(...)

    Attributes
    ----------
    XtX : np.ndarray
        Weighted X'X matrix (k, k), with regularisation added.
    Xty : np.ndarray
        Weighted X'y vector (k,).
    n_obs : int
        Total observation count (sum of weights).
    sum_y : float
        Weighted sum of y.
    sum_y_sq : float
        Weighted sum of y².
    coef_names : List[str]
        Coefficient names (including "Intercept" if add_intercept=True).
    sum_y_sq_exact : bool
        True if ``sum_y_sq`` was computed from an exact per-stratum column,
        False if it uses the within-stratum-mean approximation.
    """

    XtX: np.ndarray
    Xty: np.ndarray
    n_obs: int
    sum_y: float
    sum_y_sq: float
    coef_names: List[str]
    sum_y_sq_exact: bool = False

    def __iter__(self) -> Iterator:
        """Iterate in (XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names) order.

        Enables transparent tuple unpacking so existing call-sites that do::

            XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names = compute_sufficient_stats_*(...)

        continue to work without modification.
        """
        yield self.XtX
        yield self.Xty
        yield self.n_obs
        yield self.sum_y
        yield self.sum_y_sq
        yield self.coef_names


# ============================================================================
# NumPy Backend
# ============================================================================

def compute_sufficient_stats_numpy(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    coef_names: Optional[List[str]] = None,
    alpha: float = DEFAULT_ALPHA
) -> SuffStats:
    """
    NumPy backend for computing sufficient statistics X'WX, X'Wy, and summary
    statistics.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n, k).
    y : np.ndarray
        Response vector (n,) or (n, 1).
    weights : np.ndarray
        Frequency weights (n,).
    coef_names : List[str], optional
        Coefficient names.  Defaults to ``["x0", "x1", ...]``.
    alpha : float
        Regularisation added to the X'X diagonal.

    Returns
    -------
    SuffStats
        Container with XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names.
        Also supports 6-tuple unpacking for backward compatibility.
    """
    # Validate required parameters
    if X is None:
        raise ValueError("X cannot be None")
    if y is None:
        raise ValueError("y cannot be None")
    if weights is None:
        raise ValueError("weights cannot be None")

    # Validate and reshape inputs
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    weights = weights.flatten()
    # Ensure numeric float64 (guards against Decimal inputs from DuckDB)
    X = X.astype(np.float64)
    y = y.astype(np.float64)

    n_rows, k = X.shape
    n_obs = int(weights.sum())

    # Compute weighted matrices
    sqrt_w = np.sqrt(weights).reshape(-1, 1)
    Xw = X * sqrt_w
    yw = y * sqrt_w

    XtX = Xw.T @ Xw + alpha * np.eye(k)
    Xty = (Xw.T @ yw).flatten()

    # Summary statistics
    sum_y = float((y.flatten() * weights).sum())
    sum_y_sq = float(((y.flatten() ** 2) * weights).sum())

    # Coefficient names
    if coef_names is None:
        coef_names = [f"x{i}" for i in range(k)]

    return SuffStats(
        XtX=XtX,
        Xty=Xty,
        n_obs=n_obs,
        sum_y=sum_y,
        sum_y_sq=sum_y_sq,
        coef_names=coef_names,
        sum_y_sq_exact=True,  # Exact for observation-level data
    )

# ============================================================================
# SQL Backend
# ============================================================================

def compute_sufficient_stats_sql(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    x_cols: List[str],
    y_col: str,
    weight_col: str,
    add_intercept: bool = True,
    alpha: float = DEFAULT_ALPHA,
    sum_y_sq_col: Optional[str] = None,
    where_clause: str = ""
) -> SuffStats:
    """
    SQL backend for computing sufficient statistics X'WX, X'Wy, and summary
    statistics.

    Delegates SQL string construction to ``build_xtx_query`` and
    ``build_xty_query`` from ``sql_builders``, eliminating the previously
    duplicated loop logic.  A third query fetches observation counts and the
    y summary statistics.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    table_name : str
        Source table name.
    x_cols : List[str]
        X variable column names.
    y_col : str
        Y column name (sum_y format for compressed data).
    weight_col : str
        Weight/count column name.
    add_intercept : bool
        Whether to include intercept (prepends ``"1"`` to x_cols).
    alpha : float
        Regularisation added to the X'X diagonal.
    sum_y_sq_col : str, optional
        If provided and the column exists, use it for exact sum(y²).
        Otherwise an approximation is computed from compressed data.
    where_clause : str, optional
        WHERE clause (including ``'WHERE'`` keyword).

    Returns
    -------
    SuffStats
        Container with XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names.
        Also supports 6-tuple unpacking for backward compatibility.

    Notes
    -----
    For compressed data without ``sum_y_sq_col``, computes::

        sum_y_sq ≈ sum((sum_y/count)^2 * count)

    This is exact for uncompressed data but approximate for compressed strata.
    A warning is logged when using the approximation.
    """
    # Validate required parameters
    if conn is None:
        raise ValueError("conn cannot be None")
    if not table_name:
        raise ValueError("table_name cannot be empty")
    if not x_cols:
        raise ValueError("x_cols cannot be empty")
    if not y_col:
        raise ValueError("y_col cannot be empty")
    if not weight_col:
        raise ValueError("weight_col cannot be empty")

    # Build coef_names
    if add_intercept:
        coef_names = ["Intercept"] + x_cols
    else:
        coef_names = x_cols.copy()

    k = len(coef_names)

    # ------------------------------------------------------------------
    # Step 1: X'WX  (delegates to build_xtx_query — no duplication)
    # ------------------------------------------------------------------
    xtx_query = build_xtx_query(
        table_name=table_name,
        x_cols=x_cols,
        weight_col=weight_col,
        where_clause=where_clause,
        add_intercept=add_intercept,
    )
    xtx_row = conn.execute(xtx_query).fetchone()

    XtX = np.zeros((k, k))
    idx = 0
    for i in range(k):
        for j in range(i, k):
            XtX[i, j] = xtx_row[idx]
            XtX[j, i] = xtx_row[idx]
            idx += 1

    # Add regularisation
    XtX += alpha * np.eye(k)

    # ------------------------------------------------------------------
    # Step 2: X'y  (delegates to build_xty_query — no duplication)
    # ------------------------------------------------------------------
    xty_query = build_xty_query(
        table_name=table_name,
        x_cols=x_cols,
        y_col=y_col,
        weight_col=weight_col,
        where_clause=where_clause,
        add_intercept=add_intercept,
    )
    xty_row = conn.execute(xty_query).fetchone()
    Xty = np.array([float(xty_row[i]) for i in range(k)], dtype=float)

    # ------------------------------------------------------------------
    # Step 3: Summary statistics (n_obs, sum_y, sum_y_sq)
    # ------------------------------------------------------------------
    using_exact_sum_y_sq = False
    if sum_y_sq_col is not None:
        try:
            col_check = conn.execute(f"""
                SELECT column_name
                FROM (DESCRIBE SELECT * FROM {table_name})
                WHERE column_name = '{sum_y_sq_col}'
            """).fetchone()
            using_exact_sum_y_sq = col_check is not None
        except Exception:
            using_exact_sum_y_sq = False

    sum_y_sq_expr = (
        f"SUM({sum_y_sq_col})"
        if using_exact_sum_y_sq
        else f"SUM(POW({y_col} / {weight_col}, 2) * {weight_col})"
    )

    stats_query = f"""
    SELECT
        SUM({weight_col}) AS n_obs,
        SUM({y_col})      AS sum_y,
        {sum_y_sq_expr}   AS sum_y_sq
    FROM {table_name}
    {where_clause}
    """
    stats_row = conn.execute(stats_query).fetchone()
    n_obs = int(stats_row[0])
    sum_y = float(stats_row[1])
    sum_y_sq = float(stats_row[2])

    if not using_exact_sum_y_sq:
        logger.info(
            f"Using approximate sum_y_sq for '{y_col}': "
            f"SUM(POW(sum_y/count, 2) * count). "
            "This is exact for uncompressed data but approximate for compressed "
            "strata. For exact computation, use build_exact_sum_y_sq_sql() to "
            "add sum_y_sq columns during compression."
        )

    return SuffStats(
        XtX=XtX,
        Xty=Xty,
        n_obs=n_obs,
        sum_y=sum_y,
        sum_y_sq=sum_y_sq,
        coef_names=coef_names,
        sum_y_sq_exact=using_exact_sum_y_sq,
    )


# ============================================================================
# Execution Helpers (moved from sql_builders to keep sql_builders conn-free)
# ============================================================================

def execute_to_matrix(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    shape: Tuple[int, int],
    upper_triangle: bool = False
) -> np.ndarray:
    """
    Execute a SQL query and parse the result into a numpy matrix.

    Handles symmetric matrices efficiently by only storing the upper triangle.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    query : str
        SQL query to execute.
    shape : Tuple[int, int]
        Expected shape of the output matrix (rows, cols).
    upper_triangle : bool, default False
        If True, assumes query returns upper triangle only (for symmetric matrices).
        Result columns must be in row-major order: (0,0), (0,1), …, (0,n-1), (1,1), …

    Returns
    -------
    np.ndarray
        Parsed matrix of requested shape.

    Raises
    ------
    ValueError
        If the query result does not match the expected shape or format.

    Examples
    --------
    >>> XtX = execute_to_matrix(conn, xtx_query, (2, 2), upper_triangle=True)
    >>> XtX.shape
    (2, 2)
    """
    if conn is None:
        raise ValueError("conn cannot be None")
    if not query or not query.strip():
        raise ValueError("query cannot be empty")
    if len(shape) != 2:
        raise ValueError("shape must be a 2-tuple")
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("shape dimensions must be positive")

    result = conn.execute(query).fetchone()
    if result is None or all(v is None for v in result):
        raise ValueError("Query returned no results")

    rows, cols = shape
    matrix = np.zeros(shape)

    if upper_triangle:
        if rows != cols:
            raise ValueError("upper_triangle=True requires a square matrix")

        expected_len = (rows * (rows + 1)) // 2
        if len(result) < expected_len:
            raise ValueError(
                f"Query returned {len(result)} values, expected at least "
                f"{expected_len} for {rows}x{rows} upper triangle"
            )

        idx = 0
        for i in range(rows):
            for j in range(i, cols):
                matrix[i, j] = result[idx]
                matrix[j, i] = result[idx]
                idx += 1
    else:
        expected_len = rows * cols
        if len(result) < expected_len:
            raise ValueError(
                f"Query returned {len(result)} values, expected at least "
                f"{expected_len} for {rows}x{cols} matrix"
            )

        idx = 0
        for i in range(rows):
            for j in range(cols):
                matrix[i, j] = result[idx]
                idx += 1

    return matrix


def compute_cross_sufficient_stats_sql(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    x_cols: List[str],
    z_cols: List[str],
    weight_col: str = 'count',
    add_intercept: bool = True,
    where_clause: str = ""
) -> Dict[str, Any]:
    """
    Compute cross-product sufficient statistics for IV estimation.

    High-level wrapper that:

    1. Builds the SQL query via ``build_cross_xtz_query``.
    2. Executes the query.
    3. Parses results into numpy arrays (``tXZ``, ``tZZ``).
    4. Returns metadata about column ordering.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    table_name : str
        Source table name.
    x_cols : List[str]
        X column names (excludes intercept).
    z_cols : List[str]
        Z column names (excludes intercept).
    weight_col : str, default 'count'
        Weight column name (frequency weights).
    add_intercept : bool, default True
        Whether to include intercept in both X and Z.
    where_clause : str, optional
        WHERE clause (must include ``'WHERE'`` keyword if non-empty).

    Returns
    -------
    Dict[str, Any]
        Keys:

        * ``'tXZ'``     — np.ndarray (k_x, k_z), X'WZ cross-product.
        * ``'tZZ'``     — np.ndarray (k_z, k_z), Z'WZ cross-product (symmetric).
        * ``'n_obs'``   — int, total observation count.
        * ``'x_order'`` — List[str], X column order.
        * ``'z_order'`` — List[str], Z column order.

    Raises
    ------
    ValueError
        If ``x_cols`` or ``z_cols`` is empty, or ``table_name``/``conn`` is
        invalid.
    """
    if conn is None:
        raise ValueError("conn cannot be None")
    if not table_name or not table_name.strip():
        raise ValueError("table_name cannot be empty")
    if not x_cols:
        raise ValueError("x_cols cannot be empty")
    if not z_cols:
        raise ValueError("z_cols cannot be empty")

    all_x_cols = ["1"] + x_cols if add_intercept else x_cols
    all_z_cols = ["1"] + z_cols if add_intercept else z_cols

    k_x = len(all_x_cols)
    k_z = len(all_z_cols)

    query = build_cross_xtz_query(
        table_name, x_cols, z_cols, weight_col, where_clause, add_intercept
    )

    try:
        result = conn.execute(query).fetchone()
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        logger.debug(f"Query: {query}")
        raise ValueError(f"Failed to execute cross-product query: {e}")

    if result is None:
        raise ValueError("Query returned no results")

    # Parse tXZ (k_x × k_z, row-major)
    tXZ = np.zeros((k_x, k_z))
    idx = 0
    for i in range(k_x):
        for j in range(k_z):
            tXZ[i, j] = result[idx]
            idx += 1

    # Parse tZZ (k_z × k_z, upper triangle, symmetric)
    tZZ = np.zeros((k_z, k_z))
    for i in range(k_z):
        for j in range(i, k_z):
            tZZ[i, j] = result[idx]
            tZZ[j, i] = result[idx]
            idx += 1

    n_obs = int(result[idx])

    return {
        'tXZ': tXZ,
        'tZZ': tZZ,
        'n_obs': n_obs,
        'x_order': all_x_cols,
        'z_order': all_z_cols,
    }


def profile_fe_column(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    fe_col_sql_name: str
) -> Dict[str, Any]:
    """Profile a fixed effect column to determine if it's fixed or asymptotic.

    Computes metadata used by the classification heuristic:

    * ``cardinality``          — number of unique levels.
    * ``singleton_share``      — proportion of levels with only 1 observation.
    * ``avg_obs_per_level``    — mean observations per level.
    * ``median_obs_per_level`` — median observations per level.
    * ``total_obs``            — total row count.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active database connection.
    table_name : str
        Name of the table (will be quoted if needed).
    fe_col_sql_name : str
        SQL-safe name of the FE column (will be quoted if needed).

    Returns
    -------
    Dict[str, Any]
        Profiling metadata.
    """
    from ..utils.formula_parser import quote_identifier

    quoted_table = quote_identifier(table_name)
    quoted_fe_col = quote_identifier(fe_col_sql_name)

    query = f"""
    WITH fe_counts AS (
        SELECT {quoted_fe_col}, COUNT(*) as n_obs
        FROM {quoted_table}
        GROUP BY {quoted_fe_col}
    )
    SELECT
        COUNT(DISTINCT {quoted_fe_col}) as cardinality,
        SUM(CASE WHEN n_obs = 1 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as singleton_share,
        AVG(n_obs) as avg_obs_per_level,
        MEDIAN(n_obs) as median_obs_per_level,
        SUM(n_obs) as total_obs
    FROM fe_counts
    """

    result = conn.execute(query).fetchone()
    return {
        'cardinality': int(result[0]),
        'singleton_share': float(result[1]),
        'avg_obs_per_level': float(result[2]),
        'median_obs_per_level': float(result[3]),
        'total_obs': int(result[4]),
    }


def get_fe_unique_levels(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    fe_col_sql_name: str,
    max_levels: int = 1000
) -> List[Any]:
    """Retrieve the sorted unique levels of a fixed effect column.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active database connection.
    table_name : str
        Name of the table.
    fe_col_sql_name : str
        SQL-safe name of the FE column.
    max_levels : int, default 1000
        Safety guard: raises ``ValueError`` if the column exceeds this many
        unique values.

    Returns
    -------
    List[Any]
        Sorted list of unique FE levels.

    Raises
    ------
    ValueError
        If the number of levels exceeds ``max_levels``.
    """
    from ..utils.formula_parser import quote_identifier

    quoted_table = quote_identifier(table_name)
    quoted_fe_col = quote_identifier(fe_col_sql_name)

    query = f"""
    SELECT DISTINCT {quoted_fe_col}
    FROM {quoted_table}
    ORDER BY {quoted_fe_col}
    LIMIT {max_levels + 1}
    """

    result = conn.execute(query).fetchdf()
    levels = result.iloc[:, 0].tolist()

    if len(levels) > max_levels:
        raise ValueError(
            f"FE column {fe_col_sql_name} has more than {max_levels} levels. "
            "This may cause column explosion. Consider reclassifying as "
            "asymptotic or increasing max_levels."
        )

    return levels


def get_boolean_columns(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    column_names: List[str],
) -> "Set[str]":
    """Get the subset of ``column_names`` that have BOOLEAN type in ``table_name``.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active database connection.
    table_name : str
        Table name to query.
    column_names : List[str]
        Column names to check.

    Returns
    -------
    Set[str]
        Names of columns that are BOOLEAN type.
    """
    from typing import Set  # local import to avoid top-level overhead
    cols_sql = ', '.join(f"'{c}'" for c in column_names)
    query = f"""
    SELECT column_name FROM (DESCRIBE SELECT * FROM {table_name})
    WHERE column_name IN ({cols_sql}) AND column_type = 'BOOLEAN'
    """
    return set(conn.execute(query).fetchdf()['column_name'].tolist())


def get_table_columns(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
) -> "Set[str]":
    """Get all column names from ``table_name``.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active database connection.
    table_name : str
        Table name to query.

    Returns
    -------
    Set[str]
        Set of column names.
    """
    return set(
        conn.execute(f"SELECT column_name FROM (DESCRIBE {table_name})")
        .fetchdf()['column_name'].tolist()
    )
