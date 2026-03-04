"""
Core computational utilities for duckreg.

This package contains low-level helpers for:
- Linear algebra operations with numerical stability
- SQL query construction
- Sufficient statistics computation
- Residual-based aggregates for variance estimation
- Variance-covariance computation
"""

from .linalg import safe_solve, safe_inv, check_condition_number
from .sql_builders import (
    # Expression builders
    build_round_expr,
    build_residual_expr,
    build_fitted_expr,
    # Query builders
    build_agg_columns,
    build_strata_select_sql,
    build_xtx_query,
    build_xty_query,
    build_cross_xtz_query,
    build_meat_query,
    build_cluster_scores_query,
    build_leverage_query,
    # Pure string helper
    build_exact_sum_y_sq_sql,
)
from .suffstats import (
    SuffStats,
    compute_sufficient_stats_numpy,
    compute_sufficient_stats_sql,
    # Execution helpers (live here, shims remain in sql_builders for back-compat)
    execute_to_matrix,
    compute_cross_sufficient_stats_sql,
    # FE utilities
    profile_fe_column,
    get_fe_unique_levels,
)
from .residual_aggregates import (
    compute_residual_aggregates_numpy,
    compute_residual_aggregates_sql
)
from .vcov import (
    SSCConfig,
    VcovContext,
    VcovTypeNotSupportedError,
    parse_vcov_specification,
    parse_cluster_vars,
    compute_ssc,
    compute_bread,
    sandwich_from_meat,
    vcov_iid,
    compute_iid_vcov,
    vcov_hetero,
    compute_hetero_vcov,
    vcov_crv1,
    compute_cluster_vcov,
    compute_twoway_cluster_vcov,
    compute_cluster_scores,
)

__all__ = [
    # Linear algebra
    'safe_solve',
    'safe_inv',
    'check_condition_number',
    # SQL builders - expressions
    'build_round_expr',
    'build_residual_expr',
    'build_fitted_expr',
    # SQL builders - queries
    'build_agg_columns',
    'build_strata_select_sql',
    'build_xtx_query',
    'build_xty_query',
    'build_cross_xtz_query',
    'build_meat_query',
    'build_cluster_scores_query',
    'build_leverage_query',
    # SQL builders - execution helpers (real home: suffstats)
    'execute_to_matrix',
    'compute_cross_sufficient_stats_sql',
    # Sufficient statistics
    'SuffStats',
    'compute_sufficient_stats_numpy',
    'compute_sufficient_stats_sql',
    'build_exact_sum_y_sq_sql',
    # FE utilities
    'profile_fe_column',
    'get_fe_unique_levels',
    # Residual aggregates
    'compute_residual_aggregates',
    'compute_residual_aggregates_numpy',
    'compute_residual_aggregates_sql',
    # VCOV - Configuration
    'SSCConfig',
    'VcovContext',
    # VCOV - Exceptions
    'VcovTypeNotSupportedError',
    # VCOV - Parsing
    'parse_vcov_specification',
    'parse_cluster_vars',
    # VCOV - Core functions
    'compute_ssc',
    'compute_bread',
    'sandwich_from_meat',
    'vcov_iid',
    'compute_iid_vcov',
    'vcov_hetero',
    'compute_hetero_vcov',
    'vcov_crv1',
    'compute_cluster_vcov',
    'compute_twoway_cluster_vcov',
    # VCOV - Helpers
    'compute_cluster_scores',
]
