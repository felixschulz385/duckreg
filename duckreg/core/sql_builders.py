"""
SQL expression builders for DuckDB queries.

Organization:
-------------
1. Expression Builders: Pure, small functions that build SQL expression strings
2. Query Builders: Functions that return complete SQL statements

This module is a *pure string-production* module: no function here accepts a
``duckdb.DuckDBPyConnection`` or calls ``.execute()``.  All execution helpers
that were previously in "Section 3" have been moved to ``suffstats.py``.

This is the single source of truth for all SQL string construction.

API Design Principles:
----------------------
- All query builders accept consistent signatures:
  (table_name, cols..., weight_col, where_clause, add_intercept)
- All functions have type annotations and input validation
- Column ordering is explicit and documented
"""

import re
import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Tuple, Set, Dict, Any, Union

logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1: Expression Builders (Pure, Atom-Level)
# ============================================================================
# These functions build SQL expression strings without side effects.
# They are small, composable, and have no database dependencies.

def build_round_expr(
    expr: str,
    alias: str,
    round_strata: Optional[int]
) -> Tuple[str, str]:
    """
    Build expression with optional rounding for data compression.
    
    Pure function: no side effects, no database access.
    
    Parameters
    ----------
    expr : str
        SQL expression to round
    alias : str
        Column alias for SELECT clause (will be quoted if needed)
    round_strata : int, optional
        Number of decimals to round to. If None, no rounding.
        
    Returns
    -------
    Tuple[str, str]
        (select_expr with quoted alias, group_by_expr without alias)
        
    Examples
    --------
    >>> build_round_expr("price * 1.1", "adj_price", 2)
    ('ROUND(price * 1.1, 2) AS "adj_price"', 'ROUND(price * 1.1, 2)')
    >>> build_round_expr("price", "price", None)
    ('price AS "price"', 'price')
        
    Raises
    ------
    ValueError
        If expr or alias is empty
    """
    from ..utils.formula_parser import quote_identifier
    
    if not expr or not expr.strip():
        raise ValueError("expr cannot be empty")
    if not alias or not alias.strip():
        raise ValueError("alias cannot be empty")
    
    # Quote alias to handle special characters and reserved words
    quoted_alias = quote_identifier(alias)
    
    if round_strata is not None:
        if not isinstance(round_strata, int) or round_strata < 0:
            raise ValueError(f"round_strata must be a non-negative integer, got: {round_strata}")
        round_expr = f"ROUND({expr}, {round_strata})"
        return f"{round_expr} AS {quoted_alias}", round_expr
    return f"{expr} AS {quoted_alias}", expr


def build_residual_expr(
    theta: np.ndarray,
    x_cols: List[str],
    y_col: str,
    weight_col: str
) -> str:
    """
    Build SQL expression for residuals: (y/weight) - X @ theta.
    
    Pure function: computes expression string from coefficients.
    Uses full precision (17 decimals) to avoid numerical errors.
    
    Parameters
    ----------
    theta : np.ndarray
        Coefficient vector (k,)
    x_cols : List[str]
        X column names (k,). Must match length of theta.
    y_col : str
        Y column name (typically in sum_y format)
    weight_col : str
        Weight column name
        
    Returns
    -------
    str
        SQL expression for residuals
        
    Raises
    ------
    ValueError
        If theta and x_cols have mismatched lengths or are empty
        
    Examples
    --------
    >>> theta = np.array([1.5, 2.0])
    >>> build_residual_expr(theta, ["x1", "x2"], "sum_y", "count")
    '((sum_y / count) - ((1.5e+00) * (x1) + (2.0e+00) * (x2)))'
    """
    if len(theta) == 0:
        raise ValueError("theta cannot be empty")
    if len(x_cols) == 0:
        raise ValueError("x_cols cannot be empty")
    if len(theta) != len(x_cols):
        raise ValueError(
            f"theta length ({len(theta)}) must match x_cols length ({len(x_cols)})"
        )
    if not y_col or not y_col.strip():
        raise ValueError("y_col cannot be empty")
    if not weight_col or not weight_col.strip():
        raise ValueError("weight_col cannot be empty")
    
    # Format theta with full precision, locale-neutral
    theta_terms = " + ".join([
        f"({theta[i]:.17e}) * ({col})"
        for i, col in enumerate(x_cols)
    ])
    return f"(({y_col} / {weight_col}) - ({theta_terms}))"


def build_fitted_expr(
    theta: np.ndarray,
    x_cols: List[str]
) -> str:
    """
    Build SQL expression for fitted values: X @ theta.
    
    Pure function: computes expression string from coefficients.
    Uses full precision (17 decimals) to avoid numerical errors.
    
    Parameters
    ----------
    theta : np.ndarray
        Coefficient vector (k,)
    x_cols : List[str]
        X column names (k,). Must match length of theta.
        
    Returns
    -------
    str
        SQL expression for fitted values
        
    Raises
    ------
    ValueError
        If theta and x_cols have mismatched lengths or are empty
        
    Examples
    --------
    >>> theta = np.array([1.5, 2.3, -0.5])
    >>> build_fitted_expr(theta, ["1", "age", "income"])
    '(1.5e+00) * (1) + (2.3e+00) * (age) + (-5.0e-01) * (income)'
    """
    if len(theta) == 0:
        raise ValueError("theta cannot be empty")
    if len(x_cols) == 0:
        raise ValueError("x_cols cannot be empty")
    if len(theta) != len(x_cols):
        raise ValueError(
            f"theta length ({len(theta)}) must match x_cols length ({len(x_cols)})"
        )
    
    theta_terms = " + ".join([
        f"({theta[i]:.17e}) * ({col})"
        for i, col in enumerate(x_cols)
    ])
    return theta_terms


# ============================================================================
# SECTION 2: Query Builders (Full SQL Statements)
# ============================================================================
# These functions return complete SQL queries with consistent signatures.
# Standard parameters: table_name, cols, weight_col, where_clause, add_intercept

def build_agg_columns(
    outcome_vars: List,
    boolean_cols: Set[str],
    unit_col: Optional[str]
) -> List[str]:
    """
    Build aggregation column expressions for outcomes.
    
    Creates COUNT(*), SUM(y), and SUM(y²) columns for each outcome variable.
    This is used for data compression and sufficient statistics computation.
    
    Parameters
    ----------
    outcome_vars : List
        List of Variable objects representing outcomes (must have sql_name attribute)
    boolean_cols : Set[str]
        Set of boolean column names (for casting to INT)
    unit_col : str, optional
        Unit column for panel data transformations
        
    Returns
    -------
    List[str]
        List of SQL aggregation expressions in order:
        ['COUNT(*) as count', 'SUM(y1) as sum_y1', 'SUM(y1²) as sum_y1_sq', ...]
        
    Raises
    ------
    ValueError
        If outcome_vars is empty
        
    Examples
    --------
    >>> # Assuming Variable objects with sql_name attributes
    >>> agg_cols = build_agg_columns(outcomes, {"employed"}, "country")
    >>> agg_cols[0]
    'COUNT(*) as count'
    """
    if not outcome_vars:
        raise ValueError("outcome_vars cannot be empty")
    
    from ..utils.formula_parser import cast_if_boolean
    
    agg_parts = ["COUNT(*) as count"]
    for var in outcome_vars:
        col_expr = var.get_sql_expression(unit_col, 'year')
        col_expr = cast_if_boolean(col_expr, var.name, boolean_cols)
        agg_parts.append(f"SUM({col_expr}) as sum_{var.sql_name}")
        agg_parts.append(f"SUM(({col_expr}) * ({col_expr})) as sum_{var.sql_name}_sq")
    return agg_parts


def build_strata_select_sql(
    formula,
    strata_cols: List[str],
    boolean_cols: Set[str],
    unit_col: Optional[str],
    round_strata: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    """
    Build SELECT and GROUP BY parts for strata columns.
    
    Handles interactions, merged FEs, and optional rounding for compression.
    This function looks up Variable objects from the formula to get proper
    SQL expressions and aliases.
    
    Parameters
    ----------
    formula : FormulaParser
        Parsed formula object with variable lookup methods
    strata_cols : List[str]
        Raw column names for strata (covariates + FEs)
    boolean_cols : Set[str]
        Boolean columns requiring casting to INT
    unit_col : str, optional
        Unit column for panel transformations
    round_strata : int, optional
        Decimals for rounding (compression)
        
    Returns
    -------
    Tuple[List[str], List[str]]
        (select_expressions, group_by_expressions)
        
        Column ordering matches strata_cols input order.
        
    Raises
    ------
    ValueError
        If strata_cols is empty or formula is None
        
    Examples
    --------
    >>> select_parts, group_parts = build_strata_select_sql(
    ...     formula, ["age", "region"], boolean_cols, None, 2
    ... )
    >>> select_parts[0]
    'ROUND(age, 2) AS age'
    >>> group_parts[0]
    'ROUND(age, 2)'
    """
    if formula is None:
        raise ValueError("formula cannot be None")
    if not strata_cols:
        raise ValueError("strata_cols cannot be empty")
    
    from ..utils.formula_parser import cast_if_boolean, _make_sql_safe_name, quote_identifier
    
    select_parts, group_by_parts = [], []
    
    for col_name in strata_cols:
        # Try to find Variable object
        var = (formula.get_covariate_by_name(col_name) or 
               formula.get_fe_by_name(col_name))
        
        if var:
            col_expr = var.get_sql_expression(unit_col, 'year')
            col_expr = cast_if_boolean(col_expr, var.name, boolean_cols)
            select_expr, group_expr = build_round_expr(col_expr, var.sql_name, round_strata)
        else:
            # Handle interactions or merged FEs
            interaction = formula.get_interaction_by_name(col_name)
            if interaction:
                col_expr = interaction.get_sql_expression(unit_col, 'year', boolean_cols)
                select_expr, group_expr = build_round_expr(col_expr, interaction.sql_name, round_strata)
            else:
                mfe = formula.get_merged_fe_by_name(col_name)
                if mfe:
                    col_expr = mfe.get_sql_expression(boolean_cols)
                    select_expr, group_expr = build_round_expr(col_expr, mfe.sql_name, round_strata)
                else:
                    # Fallback: raw column
                    col_expr = formula.get_covariate_expression(col_name, unit_col, 'year', boolean_cols)
                    if col_expr == quote_identifier(col_name) or col_expr == col_name:
                        col_expr = formula.get_fe_expression(col_name, boolean_cols)
                    safe_name = _make_sql_safe_name(col_name)
                    select_expr, group_expr = build_round_expr(col_expr, safe_name, round_strata)
        
        select_parts.append(select_expr)
        group_by_parts.append(group_expr)
    
    return select_parts, group_by_parts


def build_xtx_query(
    table_name: str,
    x_cols: List[str],
    weight_col: str = "count",
    where_clause: str = "",
    add_intercept: bool = False
) -> str:
    """
    Build SQL query for X'WX computation (upper triangle).
    
    Computes weighted cross-product matrix for OLS sufficient statistics.
    Only upper triangle is computed for efficiency (matrix is symmetric).
    
    Parameters
    ----------
    table_name : str
        Source table name (will be quoted if needed)
    x_cols : List[str]
        X column names (k,). If add_intercept=True, intercept is prepended.
        Column names should already be properly quoted if needed.
    weight_col : str, default "count"
        Weight column name (will be quoted if needed)
    where_clause : str, optional
        WHERE clause (must include 'WHERE' keyword if non-empty)
    add_intercept : bool, default False
        Whether to prepend "1" for intercept column
        
    Returns
    -------
    str
        SQL query returning columns: xtx_0_0, xtx_0_1, ..., xtx_{k-1}_{k-1}
        
        Column order: upper triangle in row-major order
        
    Raises
    ------
    ValueError
        If (x_cols is empty and add_intercept is False) or table_name is empty
        
    Examples
    --------
    >>> query = build_xtx_query("data", ["x1", "x2"], "count")
    >>> # Returns: SELECT SUM((x1)*(x1)*"count") AS xtx_0_0, 
    >>> #                SUM((x1)*(x2)*"count") AS xtx_0_1,
    >>> #                SUM((x2)*(x2)*"count") AS xtx_1_1
    >>> #         FROM "data"
    """
    from ..utils.formula_parser import quote_identifier
    
    if not table_name or not table_name.strip():
        raise ValueError("table_name cannot be empty")
    if not x_cols and not add_intercept:
        raise ValueError("x_cols cannot be empty if add_intercept is False")
    if not weight_col or not weight_col.strip():
        raise ValueError("weight_col cannot be empty")
    
    all_x_cols = ["1"] + x_cols if add_intercept else x_cols
    k = len(all_x_cols)
    
    # Quote table name and weight column
    quoted_table = quote_identifier(table_name)
    quoted_weight = quote_identifier(weight_col)
    
    xtx_parts = []
    for i in range(k):
        for j in range(i, k):
            xtx_parts.append(
                f"SUM(CAST(({all_x_cols[i]}) AS DOUBLE) * CAST(({all_x_cols[j]}) AS DOUBLE) * CAST({quoted_weight} AS DOUBLE)) AS xtx_{i}_{j}"
            )
    
    return f"""
    SELECT {', '.join(xtx_parts)}
    FROM {quoted_table}
    {where_clause}
    """


def build_xty_query(
    table_name: str,
    x_cols: List[str],
    y_col: str,
    weight_col: str = "count",
    where_clause: str = "",
    add_intercept: bool = False
) -> str:
    """
    Build SQL query for X'y computation.
    
    Computes weighted cross-product of design matrix and outcome for OLS.
    
    Parameters
    ----------
    table_name : str
        Source table name (will be quoted if needed)
    x_cols : List[str]
        X column names (k,). If add_intercept=True, intercept is prepended.
        Column names should already be properly quoted if needed.
    y_col : str
        Y column name (typically in sum_y format, will be quoted if needed)
    weight_col : str, default "count"
        Weight column name (will be quoted if needed)
    where_clause : str, optional
        WHERE clause (must include 'WHERE' keyword if non-empty)
    add_intercept : bool, default False
        Whether to prepend "1" for intercept column
        
    Returns
    -------
    str
        SQL query returning columns: xty_0, xty_1, ..., xty_{k-1}
        
        Column order matches x_cols (with intercept first if add_intercept=True)
        
    Raises
    ------
    ValueError
        If (x_cols is empty and add_intercept is False), y_col is empty, or table_name is empty
        
    Examples
    --------
    >>> query = build_xty_query("data", ["x1", "x2"], "sum_y")
    >>> # Returns: SELECT SUM((x1)*"sum_y") AS xty_0,
    >>> #                SUM((x2)*"sum_y") AS xty_1
    >>> #         FROM "data"
    """
    from ..utils.formula_parser import quote_identifier
    
    if not table_name or not table_name.strip():
        raise ValueError("table_name cannot be empty")
    if not x_cols and not add_intercept:
        raise ValueError("x_cols cannot be empty if add_intercept is False")
    if not y_col or not y_col.strip():
        raise ValueError("y_col cannot be empty")
    
    # Quote identifiers
    quoted_table = quote_identifier(table_name)
    quoted_y = quote_identifier(y_col)
    
    all_x_cols = ["1"] + x_cols if add_intercept else x_cols
    xty_parts = [
        f"SUM(CAST(({col}) AS DOUBLE) * CAST({quoted_y} AS DOUBLE)) AS xty_{i}"
        for i, col in enumerate(all_x_cols)
    ]
    
    return f"""
    SELECT {', '.join(xty_parts)}
    FROM {quoted_table}
    {where_clause}
    """


def build_cross_xtz_query(
    table_name: str,
    x_cols: List[str],
    z_cols: List[str],
    weight_col: str = "count",
    where_clause: str = "",
    add_intercept: bool = False
) -> str:
    """
    Build SQL query for X'Z and Z'Z computation (for IV estimation).
    
    Returns both tXZ (k_x x k_z) and tZZ (k_z x k_z) in one query for efficiency.
    Z'Z is symmetric, so only upper triangle is computed.
    
    Parameters
    ----------
    table_name : str
        Source table name (will be quoted if needed)
    x_cols : List[str]
        X column names (k_x,). If add_intercept=True, intercept is prepended.
        Column names should already be properly quoted if needed.
    z_cols : List[str]
        Z column names (k_z,). If add_intercept=True, intercept is prepended.
        Column names should already be properly quoted if needed.
    weight_col : str, default "count"
        Weight column name (will be quoted if needed)
    where_clause : str, optional
        WHERE clause (must include 'WHERE' keyword if non-empty)
    add_intercept : bool, default False
        Whether to prepend "1" for intercept in both X and Z
        
    Returns
    -------
    str
        SQL query returning columns in order:
        - txz_i_j for i in range(k_x), j in range(k_z) (row-major)
        - tzz_i_j for i <= j (upper triangle, row-major)
        - n_obs (observation count)
        
    Raises
    ------
    ValueError
        If x_cols or z_cols is empty, or table_name is empty
        
    Examples
    --------
    >>> query = build_cross_xtz_query("data", ["x1"], ["z1", "z2"], "count")
    >>> # Returns columns: txz_0_0, txz_0_1, tzz_0_0, tzz_0_1, tzz_1_1, n_obs
    """
    from ..utils.formula_parser import quote_identifier
    
    if not table_name or not table_name.strip():
        raise ValueError("table_name cannot be empty")
    if not x_cols:
        raise ValueError("x_cols cannot be empty")
    if not z_cols:
        raise ValueError("z_cols cannot be empty")
    if not weight_col or not weight_col.strip():
        raise ValueError("weight_col cannot be empty")
    
    # Quote identifiers
    quoted_table = quote_identifier(table_name)
    quoted_weight = quote_identifier(weight_col)
    
    all_x_cols = ["1"] + x_cols if add_intercept else x_cols
    all_z_cols = ["1"] + z_cols if add_intercept else z_cols
    
    k_x = len(all_x_cols)
    k_z = len(all_z_cols)
    
    parts = []
    
    # X'Z (all elements, row-major order)
    for i, x_col in enumerate(all_x_cols):
        for j, z_col in enumerate(all_z_cols):
            parts.append(
                f"SUM(({x_col}) * ({z_col}) * {quoted_weight}) AS txz_{i}_{j}"
            )
    
    # Z'Z (upper triangle only, symmetric)
    for i in range(k_z):
        for j in range(i, k_z):
            parts.append(
                f"SUM(({all_z_cols[i]}) * ({all_z_cols[j]}) * {quoted_weight}) AS tzz_{i}_{j}"
            )
    
    # Observation count
    parts.append(f"SUM({quoted_weight}) AS n_obs")
    
    return f"""
    SELECT {', '.join(parts)}
    FROM {table_name}
    {where_clause}
    """


def build_meat_query(
    table_name: str,
    x_cols: List[str],
    residual_expr: str,
    weight_col: str = "count",
    where_clause: str = "",
    add_intercept: bool = False
) -> str:
    """
    Build SQL query for meat matrix: sum_i weight_i * (X_i * resid_i)^2.
    
    Computes the "meat" part of the sandwich estimator (X'ΩX where Ω = diag(e²w)).
    Only upper triangle is returned (matrix is symmetric).
    
    The weight appears once (not squared): this correctly replicates the
    per-observation sum sum_j x_j^2 u_j^2 = sum_i n_i x_i^2 u_i^2
    for compressed data where every observation within stratum i shares the
    same (x_i, u_i).
    
    Parameters
    ----------
    table_name : str
        Source table name
    x_cols : List[str]
        X column names (k,). If add_intercept=True, intercept is prepended.
    residual_expr : str
        SQL expression for residuals (from build_residual_expr)
    weight_col : str, default "count"
        Weight column name
    where_clause : str, optional
        WHERE clause (must include 'WHERE' keyword if non-empty)
    add_intercept : bool, default False
        Whether to prepend "1" for intercept column
        
    Returns
    -------
    str
        SQL query returning columns: meat_0_0, meat_0_1, ..., meat_{k-1}_{k-1}
        
        Column order: upper triangle in row-major order
        
    Raises
    ------
    ValueError
        If (x_cols is empty and add_intercept is False), residual_expr is empty, or table_name is empty
    """
    if not table_name or not table_name.strip():
        raise ValueError("table_name cannot be empty")
    if not x_cols and not add_intercept:
        raise ValueError("x_cols cannot be empty if add_intercept is False")
    if not residual_expr or not residual_expr.strip():
        raise ValueError("residual_expr cannot be empty")
    if not weight_col or not weight_col.strip():
        raise ValueError("weight_col cannot be empty")
    
    all_x_cols = ["1"] + x_cols if add_intercept else x_cols
    k = len(all_x_cols)
    meat_parts = []
    
    for i in range(k):
        for j in range(i, k):
            # Cast X columns to DOUBLE to prevent DECIMAL(38) overflow on
            # columns that come from DuckDB aggregate / parquet sources.
            col_i = f"CAST(({all_x_cols[i]}) AS DOUBLE)"
            col_j = f"CAST(({all_x_cols[j]}) AS DOUBLE)"
            meat_parts.append(
                f"SUM({col_i} * {col_j} * "
                f"POW(CAST(({residual_expr}) AS DOUBLE), 2) * ({weight_col})) AS meat_{i}_{j}"
            )
    
    return f"""
    SELECT {', '.join(meat_parts)}
    FROM {table_name}
    {where_clause}
    """


def build_exact_meat_query(
    table_name: str,
    x_cols: List[str],
    theta: np.ndarray,
    sum_y_col: str,
    sum_y_sq_col: str,
    weight_col: str = "count",
    where_clause: str = "",
    add_intercept: bool = False
) -> str:
    """
    Build SQL query for exact meat matrix using stored per-stratum sum-of-squares.

    When data are compressed into strata (e.g. via ``round_strata``), each stratum
    may contain observations with different y values.  The standard approach of
    using the mean residual ``(mean_y - x@theta)`` underestimates the true meat
    because it ignores within-stratum y variation.

    This function uses the pre-aggregated ``sum_y_sq`` column (sum of individual
    y² within each stratum) to recover the exact per-stratum residual sum-of-squares::

        ss_i = sum_y_sq_i - 2 * fitted_i * sum_y_i + count_i * fitted_i^2

    and computes the exact meat::

        meat_{pq} = SUM_strata( x_p * x_q * ss_i )

    Parameters
    ----------
    table_name : str
        Source table / view name.
    x_cols : List[str]
        X column names (without intercept). Intercept prepended if ``add_intercept``.
    theta : np.ndarray
        Coefficient vector (k,).
    sum_y_col : str
        Column containing SUM(y) within each stratum (e.g. ``"sum_avhrr_median"``).
    sum_y_sq_col : str
        Column containing SUM(y²) within each stratum (e.g. ``"sum_avhrr_median_sq"``).
    weight_col : str, default "count"
        Column containing observation counts per stratum.
    where_clause : str, optional
        WHERE clause including the ``WHERE`` keyword.
    add_intercept : bool, default False
        If True, prepend ``"1"`` to x_cols and the first element of theta is
        the intercept coefficient.

    Returns
    -------
    str
        SQL query returning ``meat_{i}_{j}`` columns for the upper triangle.
    """
    if not table_name or not table_name.strip():
        raise ValueError("table_name cannot be empty")
    if not x_cols and not add_intercept:
        raise ValueError("x_cols cannot be empty if add_intercept is False")
    if theta is None or len(theta) == 0:
        raise ValueError("theta cannot be empty")
    if not sum_y_col or not sum_y_sq_col:
        raise ValueError("sum_y_col and sum_y_sq_col cannot be empty")

    all_x_cols = ["1"] + x_cols if add_intercept else x_cols
    k = len(all_x_cols)

    if len(theta) != k:
        raise ValueError(
            f"theta length ({len(theta)}) must match number of columns ({k})"
        )

    # Fitted value: x @ theta  (stratum-level prediction)
    fitted_terms = " + ".join(
        [f"({theta[i]:.17e}) * ({col})" for i, col in enumerate(all_x_cols)]
    )
    fitted_expr = f"({fitted_terms})"

    # Exact per-stratum residual SS:
    #   ss = sum_y_sq - 2 * fitted * sum_y + count * fitted^2
    exact_ss = (
        f"({sum_y_sq_col} - 2.0 * {fitted_expr} * {sum_y_col} "
        f"+ {weight_col} * POW({fitted_expr}, 2))"
    )

    meat_parts = []
    for i in range(k):
        for j in range(i, k):
            meat_parts.append(
                f"SUM(({all_x_cols[i]}) * ({all_x_cols[j]}) * {exact_ss}) AS meat_{i}_{j}"
            )

    return f"""
    SELECT {', '.join(meat_parts)}
    FROM {table_name}
    {where_clause}
    """


def build_cluster_scores_query(
    table_name: str,
    x_cols: List[str],
    residual_expr: str,
    weight_col: str,
    cluster_col: str,
    where_clause: str = "",
    add_intercept: bool = False
) -> str:
    """
    Build SQL query for cluster-aggregated scores.
    
    Computes scores (X * residuals * weights) aggregated by cluster.
    Used for cluster-robust standard errors (CRV1, CRV3).
    
    Parameters
    ----------
    table_name : str
        Source table name
    x_cols : List[str]
        X column names (k,). If add_intercept=True, intercept is prepended.
    residual_expr : str
        SQL expression for residuals (from build_residual_expr)
    weight_col : str
        Weight column name
    cluster_col : str
        Cluster identifier column
    where_clause : str, optional
        WHERE clause (must include 'WHERE' keyword if non-empty)
    add_intercept : bool, default False
        Whether to prepend "1" for intercept column
        
    Returns
    -------
    str
        SQL query returning columns: cluster_col, score_0, score_1, ..., score_{k-1}
        
        Each row represents one cluster's aggregated score.
        
    Raises
    ------
    ValueError
        If (x_cols is empty and add_intercept is False), cluster_col is empty, or residual_expr is empty
    """
    if not table_name or not table_name.strip():
        raise ValueError("table_name cannot be empty")
    if not x_cols and not add_intercept:
        raise ValueError("x_cols cannot be empty if add_intercept is False")
    if not residual_expr or not residual_expr.strip():
        raise ValueError("residual_expr cannot be empty")
    if not weight_col or not weight_col.strip():
        raise ValueError("weight_col cannot be empty")
    if not cluster_col or not cluster_col.strip():
        raise ValueError("cluster_col cannot be empty")
    
    all_x_cols = ["1"] + x_cols if add_intercept else x_cols
    score_cols = [
        f"SUM(({col}) * {residual_expr} * {weight_col}) AS score_{i}"
        for i, col in enumerate(all_x_cols)
    ]
    
    return f"""
    SELECT {cluster_col}, {', '.join(score_cols)}
    FROM {table_name}
    {where_clause}
    GROUP BY {cluster_col}
    """


def build_leverage_query(
    table_name: str,
    x_cols: List[str],
    XtX_inv: np.ndarray,
    where_clause: str = "",
    add_intercept: bool = False
) -> str:
    """
    Build SQL query for leverage values: h_ii = x_i' (X'X)^{-1} x_i.
    
    Computes diagonal of hat matrix for HC2/HC3 standard errors.
    Note: XtX_inv must be computed first.
    
    Parameters
    ----------
    table_name : str
        Source table name
    x_cols : List[str]
        X column names (k,). If add_intercept=True, intercept is prepended.
    XtX_inv : np.ndarray
        (X'X)^{-1} matrix (k, k). Must match length of x_cols (+ intercept).
    where_clause : str, optional
        WHERE clause (must include 'WHERE' keyword if non-empty)
    add_intercept : bool, default False
        Whether to prepend "1" for intercept column
        
    Returns
    -------
    str
        SQL query returning single column: leverage
        
    Raises
    ------
    ValueError
        If x_cols is empty, XtX_inv shape doesn't match, or table_name is empty
    """
    if not table_name or not table_name.strip():
        raise ValueError("table_name cannot be empty")
    if not x_cols:
        raise ValueError("x_cols cannot be empty")
    if XtX_inv is None:
        raise ValueError("XtX_inv cannot be None")
    
    all_x_cols = ["1"] + x_cols if add_intercept else x_cols
    k = len(all_x_cols)
    
    if XtX_inv.shape != (k, k):
        raise ValueError(
            f"XtX_inv shape {XtX_inv.shape} doesn't match x_cols length {k}"
        )
    
    h_terms = []
    for j in range(k):
        for l in range(k):
            h_terms.append(
                f"({all_x_cols[j]}) * ({XtX_inv[j, l]:.17e}) * ({all_x_cols[l]})"
            )
    
    h_expr = " + ".join(h_terms)
    
    return f"""
    SELECT {h_expr} AS leverage
    FROM {table_name}
    {where_clause}
    """


# ============================================================================
# SECTION 3: Pure Aggregation Helper
# ============================================================================
# build_exact_sum_y_sq_sql is a pure string builder that was previously in
# suffstats.py.  It lives here because it belongs with the other SQL builders.

def _execution_helpers_moved():
    """Marker: execute_to_matrix and compute_cross_sufficient_stats_sql have
    been moved to ``duckreg.core.suffstats`` to keep this module conn-free."""


def execute_to_matrix(
    conn,
    query: str,
    shape: Tuple[int, int],
    upper_triangle: bool = False
) -> np.ndarray:
    """
    Execute SQL query and parse result into a numpy matrix.
    
    Handles symmetric matrices efficiently by only storing upper triangle.
    
    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection
    query : str
        SQL query to execute
    shape : Tuple[int, int]
        Expected shape of output matrix (rows, cols)
    upper_triangle : bool, default False
        If True, assumes query returns upper triangle only (for symmetric matrices).
        Result columns should be in row-major order: (0,0), (0,1), ..., (0,n-1), (1,1), ...
        
    Returns
    -------
    np.ndarray
        Parsed matrix of given shape
        
    Raises
    ------
    ValueError
        If query result doesn't match expected shape or format
        
    Examples
    --------
    >>> # Query returns: xtx_0_0, xtx_0_1, xtx_1_1 (upper triangle)
    >>> XtX = execute_to_matrix(conn, xtx_query, (2, 2), upper_triangle=True)
    >>> XtX.shape
    (2, 2)
    >>> # Result is symmetric: XtX[0,1] == XtX[1,0]
    """
    if conn is None:
        raise ValueError("conn cannot be None")
    if not query or not query.strip():
        raise ValueError("query cannot be empty")
    if len(shape) != 2:
        raise ValueError("shape must be a 2-tuple")
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("shape dimensions must be positive")
    
    # Deprecated: this shim keeps the function importable from sql_builders
    # but the real implementation now lives in duckreg.core.suffstats.
    # It will be removed in a future version.
    from .suffstats import execute_to_matrix as _real
    return _real(conn, query, shape, upper_triangle)


def compute_cross_sufficient_stats_sql(
    conn,
    table_name: str,
    x_cols: List[str],
    z_cols: List[str],
    weight_col: str = 'count',
    add_intercept: bool = True,
    where_clause: str = ""
) -> Dict[str, Any]:
    """Deprecated shim — real implementation is in duckreg.core.suffstats."""
    from .suffstats import compute_cross_sufficient_stats_sql as _real
    return _real(conn, table_name, x_cols, z_cols, weight_col, add_intercept, where_clause)


def build_exact_sum_y_sq_sql(
    y_cols: List[str],
    weight_col: str = "count"
) -> List[str]:
    """Build SQL expressions for exact sum(y^2) computation during compression.

    Generates SQL expressions to include in GROUP BY queries so that exact
    sum(y^2) is preserved for each outcome variable.  These columns can then
    be passed to ``compute_sufficient_stats_sql`` via the ``sum_y_sq_col``
    parameter.

    Parameters
    ----------
    y_cols : List[str]
        Outcome variable column names (original, not sum_y format).
    weight_col : str
        Weight column name (default: ``'count'``).

    Returns
    -------
    List[str]
        SQL expressions for each outcome:
        ``"SUM(y * y * weight) AS sum_y_sq"``.

    Examples
    --------
    >>> sum_y_sq_parts = build_exact_sum_y_sq_sql(['temperature', 'rainfall'], 'count')
    >>> sum_y_sq_parts[0]
    'SUM((temperature) * (temperature) * count) AS sum_temperature_sq'

    Notes
    -----
    For unweighted data (weight=1), this computes: ``SUM(y * y)``.
    For weighted/compressed data: ``SUM(y * y * weight)``.
    This ensures exact variance computation even after aggregation.
    """
    return [
        f"SUM(({col}) * ({col}) * {weight_col}) AS sum_{col}_sq"
        for col in y_cols
    ]


# ============================================================================
# SECTION 4: Mundlak Device SQL Builders
# ============================================================================
# Functions for computing and managing Mundlak group means (fixed effect averages)

def build_mundlak_avg_col_names(
    var_sql_names: List[str],
    fe_index: int
) -> List[str]:
    """Build column names for Mundlak group means.
    
    Parameters
    ----------
    var_sql_names : List[str]
        SQL-safe names of variables to average
    fe_index : int
        Fixed effect index (0, 1, 2, ...)
        
    Returns
    -------
    List[str]
        List of avg column names like ['avg_x1_fe0', 'avg_x2_fe0', ...]
        
    Examples
    --------
    >>> build_mundlak_avg_col_names(['gdp', 'pop'], 0)
    ['avg_gdp_fe0', 'avg_pop_fe0']
    """
    return [f"avg_{var}_fe{fe_index}" for var in var_sql_names]


def build_add_mundlak_means_sql(
    table_name: str,
    var_sql_names: List[str],
    fe_col_sql_name: str,
    fe_index: int
) -> str:
    """Build SQL to add Mundlak group mean columns via self-join.
    
    This creates a new table with the same data plus additional columns
    containing FE-level averages of the specified variables.
    
    Parameters
    ----------
    table_name : str
        Name of table to augment with means (will be quoted if needed)
    var_sql_names : List[str]
        SQL-safe names of variables to average (will be quoted if needed)
    fe_col_sql_name : str
        SQL-safe name of the fixed effect column (will be quoted if needed)
    fe_index : int
        Fixed effect index (for naming: avg_{var}_fe{i})
        
    Returns
    -------
    str
        Complete SQL statement to create updated table
        
    Raises
    ------
    ValueError
        If var_sql_names is empty
        
    Examples
    --------
    >>> sql = build_add_mundlak_means_sql('data', ['gdp', 'pop'], 'country', 0)
    >>> print(sql)  # doctest: +NORMALIZE_WHITESPACE
    CREATE OR REPLACE TABLE "data" AS
    SELECT t.*, fe_means.avg_gdp_fe0, fe_means.avg_pop_fe0
    FROM "data" t
    JOIN (
        SELECT "country", AVG("gdp") AS avg_gdp_fe0, AVG("pop") AS avg_pop_fe0
        FROM "data"
        GROUP BY "country"
    ) fe_means ON t."country" = fe_means."country"
    """
    from ..utils.formula_parser import quote_identifier
    
    if not var_sql_names:
        raise ValueError("var_sql_names cannot be empty")
    
    # Quote identifiers
    quoted_table = quote_identifier(table_name)
    quoted_fe_col = quote_identifier(fe_col_sql_name)
    
    # Build AVG expressions with quoted column names
    avg_select = ", ".join([
        f"AVG({quote_identifier(var)}) AS avg_{var}_fe{fe_index}"
        for var in var_sql_names
    ])
    
    # Build column list for join (output columns don't need quoting, they're always SQL-safe)
    join_cols = ", ".join([
        f"fe_means.avg_{var}_fe{fe_index}"
        for var in var_sql_names
    ])
    
    return f"""
CREATE OR REPLACE TABLE {quoted_table} AS
SELECT t.*, {join_cols}
FROM {quoted_table} t
JOIN (
    SELECT {quoted_fe_col}, {avg_select}
    FROM {quoted_table}
    GROUP BY {quoted_fe_col}
) fe_means ON t.{quoted_fe_col} = fe_means.{quoted_fe_col}
""".strip()


def build_add_fitted_mundlak_means_sql(
    table_name: str,
    fitted_var_sql_names: List[str],
    fe_col_sql_name: str,
    fe_index: int
) -> str:
    """Build SQL to add Mundlak means of fitted values.
    
    Similar to build_add_mundlak_means_sql, but for fitted_* columns.
    Used in 2SLS to compute FE-level averages of first-stage predictions.
    
    Parameters
    ----------
    table_name : str
        Name of table containing fitted_* columns
    fitted_var_sql_names : List[str]
        SQL-safe names of variables (e.g., ['price'] for fitted_price)
    fe_col_sql_name : str
        SQL-safe name of the fixed effect column
    fe_index : int
        Fixed effect index
        
    Returns
    -------
    str
        Complete SQL statement
        
    Examples
    --------
    >>> sql = build_add_fitted_mundlak_means_sql('data', ['price'], 'country', 0)
    >>> 'avg_fitted_price_fe0' in sql
    True
    """
    if not fitted_var_sql_names:
        raise ValueError("fitted_var_sql_names cannot be empty")
    
    # Build AVG expressions for fitted_* columns
    avg_select = ", ".join([
        f"AVG(fitted_{var}) AS avg_fitted_{var}_fe{fe_index}"
        for var in fitted_var_sql_names
    ])
    
    # Build column list for join
    join_cols = ", ".join([
        f"fe_means.avg_fitted_{var}_fe{fe_index}"
        for var in fitted_var_sql_names
    ])
    
    return f"""
CREATE OR REPLACE TABLE {table_name} AS
SELECT t.*, {join_cols}
FROM {table_name} t
JOIN (
    SELECT {fe_col_sql_name}, {avg_select}
    FROM {table_name}
    GROUP BY {fe_col_sql_name}
) fe_means ON t.{fe_col_sql_name} = fe_means.{fe_col_sql_name}
""".strip()


# ============================================================================
# SECTION 6: FE Classification and Unbalanced Panel Support
# ============================================================================

def profile_fe_column(conn, table_name: str, fe_col_sql_name: str) -> Dict[str, Any]:
    """Deprecated shim — real implementation is in duckreg.core.suffstats."""
    from .suffstats import profile_fe_column as _real
    return _real(conn, table_name, fe_col_sql_name)


def classify_fe_type(
    profile: Dict[str, Any],
    cardinality_threshold: int = 50,
    singleton_threshold: float = 0.1,
    user_override: Optional[str] = None
) -> str:
    """Classify FE as 'fixed' or 'asymptotic' using heuristic.
    
    Classification logic:
    1. If user_override provided, use it
    2. Else if cardinality <= threshold, classify as 'fixed'
    3. Else if singleton_share > threshold, classify as 'fixed' (unstable)
    4. Else if median_obs_per_level == 1, classify as 'fixed' (no pooling)
    5. Else classify as 'asymptotic'
    
    Parameters
    ----------
    profile : Dict[str, Any]
        Output from profile_fe_column()
    cardinality_threshold : int, default=50
        Max cardinality for 'fixed' classification
    singleton_threshold : float, default=0.1
        Max singleton share for 'asymptotic' classification
    user_override : str, optional
        Explicit classification ('fixed' or 'asymptotic')
        
    Returns
    -------
    str
        Either 'fixed' or 'asymptotic'
        
    Examples
    --------
    >>> profile = {'cardinality': 10, 'singleton_share': 0.0, 
    ...            'avg_obs_per_level': 100, 'median_obs_per_level': 100}
    >>> classify_fe_type(profile)
    'fixed'
    
    >>> profile = {'cardinality': 1000, 'singleton_share': 0.01,
    ...            'avg_obs_per_level': 50, 'median_obs_per_level': 45}
    >>> classify_fe_type(profile)
    'asymptotic'
    """
    if user_override:
        if user_override not in ('fixed', 'asymptotic'):
            raise ValueError(f"user_override must be 'fixed' or 'asymptotic', got: {user_override}")
        return user_override
    
    # Heuristic classification
    if profile['cardinality'] <= cardinality_threshold:
        return 'fixed'
    
    if profile['singleton_share'] > singleton_threshold:
        return 'fixed'  # Too unstable for asymptotic treatment
    
    if profile['median_obs_per_level'] == 1:
        return 'fixed'  # No pooling possible
    
    return 'asymptotic'


def build_add_fixed_fe_dummy_means_sql(
    table_name: str,
    fixed_fe_col_sql_name: str,
    fixed_fe_levels: List[Any],
    asymptotic_fe_col_sql_name: str,
    asymptotic_fe_index: int,
    reference_level: Optional[Any] = None
) -> str:
    """Build SQL to add within-asymptotic-FE means of fixed-FE dummies.
    
    For unbalanced panels with mixed FE types, Wooldridge's correction requires
    adding within-asymptotic-FE means of fixed-FE dummies to preserve equivalence.
    
    For example, if year is fixed and firm is asymptotic:
    - For each year level (except reference), compute mean(year==level | firm)
    - Add columns: avg_year_2020_fe0, avg_year_2021_fe0, etc.
    
    Parameters
    ----------
    table_name : str
        Name of table to augment (will be quoted if needed)
    fixed_fe_col_sql_name : str
        SQL-safe name of the fixed FE column
    fixed_fe_levels : List[Any]
        All levels of the fixed FE (will exclude reference_level)
    asymptotic_fe_col_sql_name : str
        SQL-safe name of the asymptotic FE column
    asymptotic_fe_index : int
        Index of asymptotic FE dimension (for naming)
    reference_level : Any, optional
        Reference level to exclude (typically first level)
        
    Returns
    -------
    str
        Complete SQL statement to add dummy-mean columns
        
    Examples
    --------
    >>> sql = build_add_fixed_fe_dummy_means_sql(
    ...     'data', 'year', [2019, 2020, 2021], 'firm_id', 0, reference_level=2019
    ... )
    >>> 'avg_year_2020_fe0' in sql
    True
    """
    from ..utils.formula_parser import quote_identifier
    
    if not fixed_fe_levels:
        raise ValueError("fixed_fe_levels cannot be empty")
    
    quoted_table = quote_identifier(table_name)
    quoted_fixed_fe = quote_identifier(fixed_fe_col_sql_name)
    quoted_asymp_fe = quote_identifier(asymptotic_fe_col_sql_name)
    
    # Exclude reference level
    levels_to_include = [lvl for lvl in fixed_fe_levels if lvl != reference_level]
    
    if not levels_to_include:
        logger.warning(f"All levels of {fixed_fe_col_sql_name} excluded (only reference level)")
        return f"-- No dummy means needed, only reference level"
    
    # Build AVG expressions for dummy means
    # Use CASE to create dummy (0/1), then AVG within asymptotic FE
    avg_select_parts = []
    join_col_parts = []
    
    for level in levels_to_include:
        # SQL-safe column name for this dummy mean (sanitise level to avoid dots
        # or other characters that are invalid in unquoted SQL identifiers)
        safe_level = re.sub(r'[^a-zA-Z0-9_]', '_', str(level))
        col_name = f"avg_{fixed_fe_col_sql_name}_{safe_level}_fe{asymptotic_fe_index}"
        
        # Handle string vs numeric levels
        if isinstance(level, str):
            level_expr = f"'{level}'"
        else:
            level_expr = str(level)
        
        avg_expr = f"AVG(CASE WHEN {quoted_fixed_fe} = {level_expr} THEN 1.0 ELSE 0.0 END) AS {col_name}"
        avg_select_parts.append(avg_expr)
        join_col_parts.append(f"fe_means.{col_name}")
    
    avg_select = ", ".join(avg_select_parts)
    join_cols = ", ".join(join_col_parts)
    
    return f"""
CREATE OR REPLACE TABLE {quoted_table} AS
SELECT t.*, {join_cols}
FROM {quoted_table} t
JOIN (
    SELECT {quoted_asymp_fe}, {avg_select}
    FROM {quoted_table}
    GROUP BY {quoted_asymp_fe}
) fe_means ON t.{quoted_asymp_fe} = fe_means.{quoted_asymp_fe}
""".strip()


def get_fe_unique_levels(conn, table_name: str, fe_col_sql_name: str, max_levels: int = 1000) -> List[Any]:
    """Deprecated shim — real implementation is in duckreg.core.suffstats."""
    from .suffstats import get_fe_unique_levels as _real
    return _real(conn, table_name, fe_col_sql_name, max_levels)


# ============================================================================
# SECTION 5: Utility Functions (pure — no conn dependency)
# ============================================================================
# Helper utilities for working with formulas
# Note: conn-requiring helpers (get_boolean_columns, get_table_columns) have
# been moved to duckreg.core.suffstats.

def get_boolean_columns(conn, table_name: str, column_names: List[str]) -> Set[str]:
    """Deprecated shim — real implementation is in duckreg.core.suffstats."""
    from .suffstats import get_boolean_columns as _real
    return _real(conn, table_name, column_names)


def get_table_columns(conn, table_name: str) -> Set[str]:
    """Deprecated shim — real implementation is in duckreg.core.suffstats."""
    from .suffstats import get_table_columns as _real
    return _real(conn, table_name)


def build_x_cols_for_duckdb(
    formula,
    fe_method: str = "demean",
    fe_cols: Optional[List[str]] = None,
    is_iv: bool = False,
    endogenous_vars: Optional[List[str]] = None
) -> List[str]:
    """Build X column names for DuckDB fitter.
    
    Parameters
    ----------
    formula : Formula
        Parsed formula object
    fe_method : str
        Method for handling fixed effects ('demean' or 'mundlak')
    fe_cols : List[str], optional
        List of fixed effect column names
    is_iv : bool
        Whether this is IV/2SLS regression
    endogenous_vars : List[str], optional
        List of endogenous variable display names (for IV)
        
    Returns
    -------
    List[str]
        List of SQL column names for X matrix
    """
    x_cols = []
    
    if not is_iv:
        # Standard OLS case
        for var in formula.covariates:
            if not var.is_intercept():
                x_cols.append(var.sql_name)
        
        # Mundlak means if applicable
        if fe_method == "mundlak" and fe_cols:
            simple_covs = [var for var in formula.covariates if not var.is_intercept()]
            for i in range(len(fe_cols)):
                for var in simple_covs:
                    x_cols.append(f"avg_{var.sql_name}_fe{i}")
    else:
        # 2SLS case
        endogenous_set = set(endogenous_vars or [])
        
        # Exogenous covariates
        for var in formula.covariates:
            if not var.is_intercept() and var.display_name not in endogenous_set:
                x_cols.append(var.sql_name)
        
        # Mundlak means for exogenous
        if fe_method == "mundlak" and fe_cols:
            for var in formula.covariates:
                if not var.is_intercept() and var.display_name not in endogenous_set:
                    for i in range(len(fe_cols)):
                        x_cols.append(f"avg_{var.sql_name}_fe{i}")
            
            # Mundlak means for fitted endogenous
            for var in formula.endogenous:
                for i in range(len(fe_cols)):
                    x_cols.append(f"avg_fitted_{var.sql_name}_fe{i}")
        
        # Fitted endogenous
        for var in formula.endogenous:
            x_cols.append(f"fitted_{var.sql_name}")
    
    return x_cols


def build_z_cols_for_duckdb(
    formula,
    fe_method: str = "demean",
    fe_cols: Optional[List[str]] = None,
    endogenous_vars: Optional[List[str]] = None
) -> List[str]:
    """Build instrument (Z) column names for first-stage DuckDB fitter.
    
    Parameters
    ----------
    formula : Formula
        Parsed formula object
    fe_method : str
        Method for handling fixed effects
    fe_cols : List[str], optional
        List of fixed effect column names
    endogenous_vars : List[str], optional
        List of endogenous variable display names
        
    Returns
    -------
    List[str]
        List of SQL column names for Z matrix (exogenous + instruments + Mundlak means)
    """
    z_cols = []
    endogenous_set = set(endogenous_vars or [])
    
    # Exogenous covariates
    for var in formula.covariates:
        if not var.is_intercept() and var.display_name not in endogenous_set:
            z_cols.append(var.sql_name)
    
    # Instruments
    for var in formula.instruments:
        z_cols.append(var.sql_name)
    
    # Mundlak means if applicable
    if fe_method == "mundlak" and fe_cols:
        # Means of exogenous covariates
        for var in formula.covariates:
            if not var.is_intercept() and var.display_name not in endogenous_set:
                for i in range(len(fe_cols)):
                    z_cols.append(f"avg_{var.sql_name}_fe{i}")
        
        # Means of instruments
        for var in formula.instruments:
            for i in range(len(fe_cols)):
                z_cols.append(f"avg_{var.sql_name}_fe{i}")
    
    return z_cols


def build_residual_x_cols_for_duckdb(
    formula,
    fe_method: str = "demean",
    fe_cols: Optional[List[str]] = None,
    endogenous_vars: Optional[List[str]] = None
) -> List[str]:
    """Build X column names using ACTUAL endogenous for residual computation (2SLS).
    
    This is critical for correct 2SLS standard errors: residuals must use
    actual endogenous values, not fitted ones. The order matches
    build_x_cols_for_duckdb so coefficients align correctly.
    
    Parameters
    ----------
    formula : Formula
        Parsed formula object
    fe_method : str
        Method for handling fixed effects
    fe_cols : List[str], optional
        List of fixed effect column names
    endogenous_vars : List[str], optional
        List of endogenous variable display names
        
    Returns
    -------
    List[str]
        List of SQL column expressions for residual computation
    """
    actual_x_cols = []
    endogenous_set = set(endogenous_vars or [])
    
    # Exogenous covariates (same as fitted)
    for var in formula.covariates:
        if not var.is_intercept() and var.display_name not in endogenous_set:
            actual_x_cols.append(var.sql_name)
    
    # Mundlak means for exogenous (same as fitted)
    if fe_method == "mundlak" and fe_cols:
        for var in formula.covariates:
            if not var.is_intercept() and var.display_name not in endogenous_set:
                for i in range(len(fe_cols)):
                    actual_x_cols.append(f"avg_{var.sql_name}_fe{i}")
        
        # Mundlak means for fitted endogenous (keep fitted for consistency)
        for var in formula.endogenous:
            for i in range(len(fe_cols)):
                actual_x_cols.append(f"avg_fitted_{var.sql_name}_fe{i}")
    
    # ACTUAL endogenous (key difference: use actual, not fitted)
    # For compressed data, these are sum columns divided by count
    for var in formula.endogenous:
        actual_x_cols.append(f"sum_{var.sql_name} / count")
    
    return actual_x_cols


def compute_mundlak_means_numpy(
    df: pd.DataFrame,
    cov_cols: List[str],
    fe_cols: List[str],
    formula
) -> Tuple[np.ndarray, List[str]]:
    """Compute group means for Mundlak device (numpy/pandas version).
    
    The Mundlak device absorbs fixed effects by including group means
    of covariates as additional regressors.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with data
    cov_cols : List[str]
        Columns to compute means for
    fe_cols : List[str]
        Fixed effect column names
    formula : Formula
        Parsed formula object (for getting SQL names)
        
    Returns
    -------
    Tuple[np.ndarray, List[str]]
        (mean values array, column names)
    """
    mean_parts = []
    mean_names = []
    
    for i, fe_name in enumerate(fe_cols):
        # Get SQL-safe name for FE
        fe_var = formula.get_fe_by_name(fe_name)
        if fe_var:
            fe_col = fe_var.sql_name
        else:
            mfe = formula.get_merged_fe_by_name(fe_name)
            fe_col = mfe.sql_name if mfe else fe_name
        
        if fe_col not in df.columns:
            logger.debug(f"FE column {fe_col} not found in dataframe")
            continue
        
        for cov_col in cov_cols:
            if cov_col not in df.columns:
                continue
            group_means = df.groupby(fe_col)[cov_col].transform('mean')
            mean_parts.append(group_means.values.reshape(-1, 1))
            mean_names.append(f"avg_{cov_col}_fe{i}")
    
    if mean_parts:
        return np.hstack(mean_parts), mean_names
    return np.empty((len(df), 0)), []


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Section 1: Expression Builders (Pure Functions)
    'build_round_expr',
    'build_residual_expr',
    'build_fitted_expr',

    # Section 2: Query Builders (SQL Statement Generators)
    'build_agg_columns',
    'build_strata_select_sql',
    'build_xtx_query',
    'build_xty_query',
    'build_cross_xtz_query',
    'build_meat_query',
    'build_exact_meat_query',
    'build_cluster_scores_query',
    'build_leverage_query',

    # Section 3: Pure string helper (execution helpers moved to suffstats)
    'build_exact_sum_y_sq_sql',
    # Deprecated shims (real implementations in duckreg.core.suffstats)
    'execute_to_matrix',
    'compute_cross_sufficient_stats_sql',

    # Section 4: Mundlak Device SQL Builders
    'build_mundlak_avg_col_names',
    'build_add_mundlak_means_sql',
    'build_add_fitted_mundlak_means_sql',

    # Section 5: Utility Functions
    'get_boolean_columns',
    'get_table_columns',
    'build_x_cols_for_duckdb',
    'build_z_cols_for_duckdb',
    'build_residual_x_cols_for_duckdb',
    'compute_mundlak_means_numpy',

    # Section 6: FE Classification and Unbalanced Panel Support
    # Deprecated shims (real implementations in duckreg.core.suffstats)
    'profile_fe_column',
    'classify_fe_type',
    'build_add_fixed_fe_dummy_means_sql',
    'get_fe_unique_levels',
]
