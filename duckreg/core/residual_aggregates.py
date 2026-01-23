"""
Residual-based aggregate computation for variance estimation.

Provides unified interface for computing RSS, scores, meat matrices,
cluster scores, and leverages in both numpy and SQL backends.

Weight Convention (Frequency Weights):
--------------------------------------
- weights[i] = count of observations represented by row i
- For compressed/aggregated data, weights are strata counts
- Residuals: residual_i = (y_i / weight_i) - X_i @ theta
  (i.e., per-observation residual for stratum i)
- Scores: score_i = X_i * (residual_i * weight_i)
  (stratum-level score, sum of original observation scores)
- RSS: sum((residual_i * sqrt(weight_i))^2)
  (weighted sum of squared residuals)

This ensures that aggregated/compressed data and original observation-level
data produce identical statistical results (within numerical precision).
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List
import duckdb

from .sql_builders import (
    build_residual_expr,
    build_meat_query,
    build_cluster_scores_query,
    build_leverage_query
)

logger = logging.getLogger(__name__)


def compute_residual_aggregates_numpy(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    residuals: Optional[np.ndarray] = None,
    cluster_ids: Optional[np.ndarray] = None,
    XtX_inv: Optional[np.ndarray] = None,
    compute_rss: bool = True,
    compute_scores: bool = False,
    compute_meat: bool = False,
    compute_cluster_scores: bool = False,
    compute_leverages: bool = False,
    Z: Optional[np.ndarray] = None,
    is_iv: bool = False
) -> Dict[str, Any]:
    """
    Numpy backend for residual aggregates computation.
    
    Weight convention: weights[i] = count of observations in stratum i
    - residuals: (y[i]/weights[i]) - X[i] @ theta
    - scores: X[i] * (residuals[i] * weights[i])
    - RSS: sum((residuals[i] * sqrt(weights[i]))^2)
    
    Parameters
    ----------
    theta : np.ndarray
        Coefficient estimates (k,)
    X : np.ndarray
        Design matrix (n, k)
    y : np.ndarray
        Response vector (n,)
    weights : np.ndarray
        Frequency weights (n,)
    residuals : np.ndarray, optional
        Pre-computed residuals (n,). If None, computed as y - X @ theta
    cluster_ids : np.ndarray, optional
        Cluster identifiers (n,), required for compute_cluster_scores
    XtX_inv : np.ndarray, optional
        (X'X)^{-1} matrix (k, k), required for compute_leverages
    Z : np.ndarray, optional
        Instrument matrix (n, m) for IV regressions. When provided with is_iv=True,
        scores will be computed as Z * residuals instead of X * residuals.
    is_iv : bool
        Whether this is an IV regression. If True and Z is provided, uses Z for scores.
    compute_* : bool
        Flags for which quantities to compute
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with requested quantities
    """
    result = {}
    
    # Compute or use provided residuals
    if residuals is None:
        residuals = (y.flatten() - X @ theta)
    else:
        residuals = residuals.flatten()
    
    # RSS: sum((residuals * sqrt(weights))^2)
    if compute_rss:
        result['rss'] = np.sum((residuals * np.sqrt(weights)) ** 2)
    
    # Scores: Use Z (instruments) for IV, X (regressors) for OLS
    # For IV: scores = Z * residuals to compute Ω = Z' diag(e²) Z correctly
    # For OLS: scores = X * residuals as usual
    scores = None
    if compute_scores or compute_meat or compute_cluster_scores:
        # Use instruments Z for IV, regressors X for OLS
        score_matrix = Z if (is_iv and Z is not None) else X
        # For duckreg with compressed data, we use:
        # scores = matrix * (residuals * weights) to properly weight the stratum contributions
        scores = score_matrix * (residuals * weights).reshape(-1, 1)
        if compute_scores:
            result['scores'] = scores
    
    # Meat
    if compute_meat:
        result['meat'] = scores.T @ scores
    
    # Cluster scores
    if compute_cluster_scores:
        if cluster_ids is None:
            raise ValueError("cluster_ids required for compute_cluster_scores")
        # Import here to avoid circular dependency
        from .vcov import compute_cluster_scores as _compute_cluster_scores
        cluster_scores_arr, G = _compute_cluster_scores(scores, cluster_ids)
        result['cluster_scores'] = cluster_scores_arr
        result['n_clusters'] = G
    
    # Leverages
    if compute_leverages:
        if XtX_inv is None:
            raise ValueError("XtX_inv required for compute_leverages")
        # h_ii = x_i' (X'X)^{-1} x_i = sum_j sum_l x_ij * inv_jl * x_il
        result['leverages'] = np.sum((X @ XtX_inv) * X, axis=1)
    
    return result


def compute_residual_aggregates_sql(
    theta: np.ndarray,
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    x_cols: List[str],
    y_col: str,
    weight_col: str,
    cluster_col: Optional[str] = None,
    add_intercept: bool = True,
    residual_x_cols: Optional[List[str]] = None,
    XtX_inv: Optional[np.ndarray] = None,
    compute_rss: bool = True,
    compute_scores: bool = False,
    compute_meat: bool = False,
    compute_cluster_scores: bool = False,
    compute_leverages: bool = False,
    where_clause: str = "",
    z_cols: Optional[List[str]] = None,
    is_iv: bool = False
) -> Dict[str, Any]:
    """
    SQL backend for residual aggregates computation.
    
    Embeds theta and XtX_inv (when needed) into SQL queries with full precision.
    
    Weight convention: weights[i] = count of observations in stratum i
    - residuals: (y[i]/weights[i]) - X[i] @ theta
    - scores: X[i] * (residuals[i] * weights[i])
    - RSS: sum((residuals[i] * sqrt(weights[i]))^2)
    
    Parameters
    ----------
    theta : np.ndarray
        Coefficient estimates (k,)
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection
    table_name : str
        Source table name
    x_cols : List[str]
        X column names
    y_col : str
        Y column name (sum_y format)
    weight_col : str
        Weight column name
    cluster_col : str, optional
        Cluster column, required for compute_cluster_scores
    add_intercept : bool
        Whether to include intercept
    residual_x_cols : List[str], optional
        Alternative X columns for residual computation.
        If provided, must match length of x_cols (+1 if add_intercept).
    XtX_inv : np.ndarray, optional
        (X'X)^{-1} matrix, required for compute_leverages
    compute_* : bool
        Flags for which quantities to compute
    where_clause : str, optional
        WHERE clause (including 'WHERE')
    z_cols : List[str], optional
        Instrument column names (excludes intercept) for IV regressions.
        When provided with is_iv=True, scores/meat/cluster_scores will be
        computed using Z instead of X.
    is_iv : bool
        Whether this is an IV regression. If True and z_cols is provided,
        uses Z for scores/meat/cluster_scores.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with requested quantities
    """
    result = {}
    
    # Build column lists
    if add_intercept:
        all_x_cols = ["1"] + x_cols
    else:
        all_x_cols = x_cols
    
    # Determine X columns for residual computation
    if residual_x_cols is not None:
        expected_len = len(x_cols) + (1 if add_intercept else 0)
        residual_len = len(residual_x_cols) + (1 if add_intercept else 0)
        if residual_len != expected_len:
            logger.warning(
                f"residual_x_cols length mismatch: expected {expected_len}, "
                f"got {residual_len}. Using x_cols for residuals."
            )
            residual_x_for_expr = all_x_cols
        else:
            residual_x_for_expr = ["1"] + residual_x_cols if add_intercept else residual_x_cols
    else:
        residual_x_for_expr = all_x_cols
    
    # Build residual expression
    residual_expr = build_residual_expr(theta, residual_x_for_expr, y_col, weight_col)
    
    k = len(all_x_cols)
    
    # Determine which columns to use for scores/meat/cluster_scores
    # For IV: use Z (instruments), for OLS: use X (regressors)
    if is_iv and z_cols is not None:
        score_cols_list = ["1"] + z_cols if add_intercept else z_cols
        k_scores = len(score_cols_list)
    else:
        score_cols_list = all_x_cols
        k_scores = k
    
    # RSS
    if compute_rss:
        rss_query = f"""
        SELECT SUM(POW({residual_expr}, 2) * {weight_col}) AS rss
        FROM {table_name}
        {where_clause}
        """
        result['rss'] = float(conn.execute(rss_query).fetchone()[0])
    
    # Cluster scores (takes priority over meat/scores)
    if compute_cluster_scores:
        if cluster_col is None:
            raise ValueError("cluster_col required for compute_cluster_scores")
        
        # Use z_cols for IV, x_cols for OLS
        cols_for_scores = z_cols if (is_iv and z_cols is not None) else x_cols
        
        # Note: build_cluster_scores_query expects cols without intercept
        # and add_intercept parameter
        query = build_cluster_scores_query(
            table_name, cols_for_scores, residual_expr, weight_col, cluster_col, where_clause, add_intercept
        )
        
        df = conn.execute(query).fetchdf()
        cluster_scores_arr = df[[f"score_{i}" for i in range(k_scores)]].values
        result['cluster_scores'] = cluster_scores_arr
        result['n_clusters'] = len(df)
    
    # Meat (more efficient than scores -> multiply)
    elif compute_meat:
        # Use z_cols for IV, x_cols for OLS
        cols_for_scores = z_cols if (is_iv and z_cols is not None) else x_cols
        
        # Note: build_meat_query expects cols without intercept and add_intercept parameter
        query = build_meat_query(
            table_name, cols_for_scores, residual_expr, weight_col, where_clause, add_intercept
        )
        
        meat_result = conn.execute(query).fetchone()
        meat = np.zeros((k_scores, k_scores))
        idx = 0
        for i in range(k_scores):
            for j in range(i, k_scores):
                meat[i, j] = meat_result[idx]
                meat[j, i] = meat_result[idx]
                idx += 1
        result['meat'] = meat
    
    # Scores (individual rows, less efficient)
    elif compute_scores:
        score_cols = [
            f"({col}) * {residual_expr} * {weight_col} AS score_{i}"
            for i, col in enumerate(score_cols_list)
        ]
        
        query = f"""
        SELECT {', '.join(score_cols)}
        FROM {table_name}
        {where_clause}
        """
        
        df = conn.execute(query).fetchdf()
        result['scores'] = df.values
    
    # Leverages
    if compute_leverages:
        if XtX_inv is None:
            raise ValueError("XtX_inv required for compute_leverages")
        
        # Note: build_leverage_query now expects x_cols without intercept and add_intercept parameter
        query = build_leverage_query(table_name, x_cols, XtX_inv, where_clause, add_intercept)
        result['leverages'] = conn.execute(query).fetchdf()['leverage'].values
    
    return result


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
    Compute cross-product sufficient statistics for IV estimation via SQL.
    
    Delegates to sql_builders.compute_cross_sufficient_stats_sql.
    
    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection
    table_name : str
        Source table name
    x_cols : List[str]
        X column names (excludes intercept)
    z_cols : List[str]
        Z column names (excludes intercept)
    weight_col : str
        Weight column name
    add_intercept : bool
        Whether to include intercept in both X and Z
    where_clause : str, optional
        WHERE clause (including 'WHERE')
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with 'tXZ' (np.ndarray k_x x k_z), 
        'tZZ' (np.ndarray k_z x k_z), 'n_obs' (int)
    """
    from .sql_builders import compute_cross_sufficient_stats_sql as _compute
    return _compute(conn, table_name, x_cols, z_cols, weight_col, add_intercept, where_clause)
