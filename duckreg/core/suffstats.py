"""
Sufficient statistics computation for weighted least squares.

Extracted from DuckDBFitter for reuse and clarity.
"""

import numpy as np
import logging
from typing import Tuple, List, Optional
import duckdb

from .sql_builders import build_xtx_query, build_xty_query
from .linalg import DEFAULT_ALPHA

logger = logging.getLogger(__name__)

def compute_sufficient_stats_numpy(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    coef_names: Optional[List[str]] = None,
    alpha: float = DEFAULT_ALPHA
) -> Tuple[np.ndarray, np.ndarray, int, float, float, List[str]]:
    """
    Numpy backend for computing sufficient statistics X'WX, X'y, and summary statistics.
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix (n, k)
    y : np.ndarray
        Response vector (n,) or (n, 1)
    weights : np.ndarray
        Frequency weights (n,)
    coef_names : List[str], optional
        Coefficient names. If None, defaults to ["x0", "x1", ...]
    alpha : float
        Regularization added to X'X diagonal
        
    Returns
    -------
    Tuple containing:
        - XtX : np.ndarray (k, k) - X'WX + alpha*I
        - Xty : np.ndarray (k,) - X'Wy
        - n_obs : int - Sum of weights
        - sum_y : float - Sum of y (weighted)
        - sum_y_sq : float - Sum of y^2 (weighted)
        - coef_names : List[str] - Coefficient names
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
    
    n_rows, k = X.shape
    n_obs = int(weights.sum())
    
    # Compute weighted matrices
    sqrt_w = np.sqrt(weights).reshape(-1, 1)
    Xw = X * sqrt_w
    yw = y * sqrt_w
    
    XtX = Xw.T @ Xw + alpha * np.eye(k)
    Xty = (Xw.T @ yw).flatten()
    
    # Summary statistics
    sum_y = (y.flatten() * weights).sum()
    sum_y_sq = ((y.flatten() ** 2) * weights).sum()
    
    # Coefficient names
    if coef_names is None:
        coef_names = [f"x{i}" for i in range(k)]
    
    return XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names

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
) -> Tuple[np.ndarray, np.ndarray, int, float, float, List[str]]:
    """
    SQL backend for computing sufficient statistics X'WX, X'y, and summary statistics.
    
    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection
    table_name : str
        Source table name
    x_cols : List[str]
        X variable column names
    y_col : str
        Y column name (sum_y format for compressed data)
    weight_col : str
        Weight/count column name
    add_intercept : bool
        Whether to include intercept (prepends "1" to x_cols)
    alpha : float
        Regularization added to X'X diagonal
    sum_y_sq_col : str, optional
        If provided, use this column for exact sum(y^2).
        Otherwise compute approximate value from compressed data.
    where_clause : str, optional
        WHERE clause (including 'WHERE')
        
    Returns
    -------
    Tuple containing:
        - XtX : np.ndarray (k, k) - X'WX + alpha*I
        - Xty : np.ndarray (k,) - X'y
        - n_obs : int - Sum of weights
        - sum_y : float - Sum of y
        - sum_y_sq : float - Sum of y^2 (exact or approximate)
        - coef_names : List[str] - Coefficient names
        
    Notes
    -----
    For compressed data without sum_y_sq_col, computes:
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
    
    # Build column lists
    if add_intercept:
        all_x_cols = ["1"] + x_cols
        coef_names = ["Intercept"] + x_cols
    else:
        all_x_cols = x_cols
        coef_names = x_cols.copy()
    
    k = len(all_x_cols)
    
    # Build X'WX computation (upper triangle)
    xtx_parts = []
    for i in range(k):
        for j in range(i, k):
            xtx_parts.append(
                f"SUM(({all_x_cols[i]}) * ({all_x_cols[j]}) * {weight_col}) AS xtx_{i}_{j}"
            )
    
    # Build X'y computation
    xty_parts = [
        f"SUM(({col}) * {y_col}) AS xty_{i}"
        for i, col in enumerate(all_x_cols)
    ]
    
    # Build summary statistics
    stats_parts = [
        f"SUM({weight_col}) AS n_obs",
        f"SUM({y_col}) AS sum_y",
    ]
    
    # Sum of y squared - check if exact column exists
    using_exact_sum_y_sq = False
    if sum_y_sq_col is not None:
        # Check if column exists in table
        try:
            col_check = conn.execute(f"""
                SELECT column_name 
                FROM (DESCRIBE SELECT * FROM {table_name})
                WHERE column_name = '{sum_y_sq_col}'
            """).fetchone()
            
            if col_check is not None:
                stats_parts.append(f"SUM({sum_y_sq_col}) AS sum_y_sq")
                using_exact_sum_y_sq = True
            else:
                # Column doesn't exist, use approximation
                stats_parts.append(
                    f"SUM(POW({y_col} / {weight_col}, 2) * {weight_col}) AS sum_y_sq"
                )
        except Exception:
            # Error checking column, fall back to approximation
            stats_parts.append(
                f"SUM(POW({y_col} / {weight_col}, 2) * {weight_col}) AS sum_y_sq"
            )
    else:
        # No sum_y_sq_col provided, use approximation
        stats_parts.append(
            f"SUM(POW({y_col} / {weight_col}, 2) * {weight_col}) AS sum_y_sq"
        )
    
    # Execute query
    query = f"""
    SELECT {', '.join(xtx_parts)}, {', '.join(xty_parts)}, {', '.join(stats_parts)}
    FROM {table_name}
    {where_clause}
    """
    
    logger.debug(f"Computing sufficient statistics for {k} parameters")
    result = conn.execute(query).fetchone()
    
    # Parse X'X (upper triangle, then symmetrize)
    XtX = np.zeros((k, k))
    idx = 0
    for i in range(k):
        for j in range(i, k):
            XtX[i, j] = result[idx]
            XtX[j, i] = result[idx]
            idx += 1
    
    # Add regularization
    XtX += alpha * np.eye(k)
    
    # Parse X'y
    Xty = np.array([result[idx + i] for i in range(k)])
    idx += k
    
    # Parse summary statistics
    n_obs = int(result[idx])
    sum_y = float(result[idx + 1])
    sum_y_sq = float(result[idx + 2])
    
    # Warn about approximation at INFO level (visible to users)
    if not using_exact_sum_y_sq:
        logger.info(
            f"Using approximate sum_y_sq for '{y_col}': SUM(POW(sum_y/count, 2) * count). "
            "This is exact for uncompressed data but approximate for compressed strata. "
            "For exact computation, use build_exact_sum_y_sq_sql() to add sum_y_sq columns "
            "during compression."
        )
    
    return XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names


def build_exact_sum_y_sq_sql(
    y_cols: List[str],
    weight_col: str = "count"
) -> List[str]:
    """Build SQL expressions for exact sum(y^2) computation during compression.
    
    This helper generates SQL expressions that can be included in GROUP BY queries
    to compute exact sum(y^2) for each outcome variable. These columns can then
    be passed to compute_sufficient_stats via the sum_y_sq_col parameter.
    
    Parameters
    ----------
    y_cols : List[str]
        Outcome variable column names (original, not sum_y format)
    weight_col : str
        Weight column name (default: 'count')
        
    Returns
    -------
    List[str]
        SQL expressions for each outcome: "SUM(y * y * weight) AS sum_y_sq"
        
    Example
    -------
    >>> # In compress_data method:
    >>> y_cols = ['temperature', 'rainfall']
    >>> sum_y_parts = [f"SUM({col}) AS sum_{col}" for col in y_cols]
    >>> sum_y_sq_parts = build_exact_sum_y_sq_sql(y_cols, 'count')
    >>> query = f'''
    ... SELECT {strata_cols}, COUNT(*) as count,
    ...        {', '.join(sum_y_parts)},
    ...        {', '.join(sum_y_sq_parts)}
    ... FROM {table}
    ... GROUP BY {strata_cols}
    ... '''
    
    Notes
    -----
    For unweighted data (weight=1), this computes: SUM(y * y)
    For weighted/compressed data, this computes: SUM(y * y * weight)
    
    This ensures exact variance computation even after aggregation.
    """
    sql_parts = []
    for col in y_cols:
        if weight_col == "1" or weight_col == "count":
            # Standard case: SUM(y^2 * weight)
            sql_parts.append(f"SUM(({col}) * ({col}) * {weight_col}) AS sum_{col}_sq")
        else:
            # Custom weight column
            sql_parts.append(f"SUM(({col}) * ({col}) * {weight_col}) AS sum_{col}_sq")
    return sql_parts
