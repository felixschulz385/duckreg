"""
Comprehensive tests for residual aggregate computation.

Tests all residual aggregate computation functions across both NumPy and DuckDB backends
with simulated data including:
- Simple unweighted data
- Weighted data with frequency weights
- Clustered data
- IV regression data

Residual Aggregates Tested:
---------------------------
- RSS (residual sum of squares)
- Scores (X' * residuals)
- Meat matrix (scores' * scores)
- Cluster scores (aggregated by cluster)
- Leverages (diagonal of hat matrix)

Weight Convention:
------------------
- weights[i] = count of observations represented by row i
- For compressed/aggregated data, weights are strata counts
- Residuals: residual_i = (y_i / weight_i) - X_i @ theta
- Scores: score_i = X_i * (residual_i * weight_i)
- RSS: sum((residual_i * sqrt(weight_i))^2)
"""
import pytest
import numpy as np
import pandas as pd
import duckdb
import logging
from typing import Dict, Any, Tuple, Optional, Literal, List

from duckreg.core.residual_aggregates import (
    compute_residual_aggregates_numpy,
    compute_residual_aggregates_sql,
)
from duckreg.core.linalg import safe_solve

logger = logging.getLogger(__name__)


# ============================================================================
# Reference Implementation: pyfixest patterns
# ============================================================================

def compute_residuals_scores_meat_pyfixest_style(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    theta: np.ndarray,
    Z: Optional[np.ndarray] = None,
    is_iv: bool = False
):
    """
    Compute residuals, scores, and meat matrix using pyfixest patterns.
    
    Extracted from pyfixest.estimation.feols_ and vcov_utils.
    This serves as our reference implementation.
    
    Weight convention (frequency weights):
    - weights[i] = count of observations represented by row i
    - residuals: (y[i] / weights[i]) - X[i] @ theta (per-observation residual)
    - scores: X[i] * (residuals[i] * weights[i]) (stratum-level score)
    - RSS: sum((residuals[i] * sqrt(weights[i]))^2)
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix (n, k)
    y : np.ndarray
        Response vector (n,)
    weights : np.ndarray
        Frequency weights (n,)
    theta : np.ndarray
        Coefficient estimates (k,)
    Z : np.ndarray, optional
        Instrument matrix (n, m) for IV regressions
    is_iv : bool
        Whether to use Z (instruments) for scores instead of X
        
    Returns
    -------
    residuals : np.ndarray (n,)
    scores : np.ndarray (n, k or m)
    meat : np.ndarray (k x k or m x m)
    rss : float
    """
    # Ensure proper shapes
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    weights = weights.flatten()
    theta = theta.flatten()
    
    # Compute residuals: per-observation residuals
    # For compressed data: (y[i] / weights[i]) - X[i] @ theta
    residuals = y.flatten() - X @ theta
    
    # RSS: sum of weighted squared residuals
    # For frequency weights: sum((residuals * sqrt(weights))^2)
    sqrt_w = np.sqrt(weights)
    rss = np.sum((residuals * sqrt_w) ** 2)
    
    # Scores: Use Z for IV, X for OLS
    # Weighted by frequency weights to get stratum-level scores
    score_matrix = Z if (is_iv and Z is not None) else X
    if score_matrix.ndim == 1:
        score_matrix = score_matrix.reshape(-1, 1)
    
    # scores[i] = matrix[i] * (residuals[i] * weights[i])
    scores = score_matrix * (residuals * weights).reshape(-1, 1)
    
    # Meat matrix: scores' @ scores
    meat = scores.T @ scores
    
    return residuals, scores, meat, rss


def compute_cluster_scores_pyfixest_style(
    scores: np.ndarray,
    cluster_ids: np.ndarray
):
    """
    Aggregate scores by cluster using pyfixest patterns.
    
    Extracted from pyfixest vcov_utils cluster score aggregation logic.
    
    Parameters
    ----------
    scores : np.ndarray
        Individual observation scores (n, k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
        
    Returns
    -------
    cluster_scores : np.ndarray (G, k)
        Aggregated scores by cluster
    n_clusters : int
        Number of clusters
    """
    # Get unique clusters
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    k = scores.shape[1]
    
    # Aggregate scores by cluster
    cluster_scores = np.zeros((n_clusters, k))
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_ids == cluster_id
        cluster_scores[i] = scores[mask].sum(axis=0)
    
    return cluster_scores, n_clusters


# ============================================================================
# FIXTURES: Data Generation
# ============================================================================

@pytest.fixture
def simple_ols_data():
    """
    Simple OLS data: y = 2 + 3*X1 + 1.5*X2 + e
    
    No weights, no intercept in X (will be added).
    """
    np.random.seed(42)
    n = 200
    
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    e = np.random.randn(n) * 0.5
    
    y = 2.0 + 3.0 * X1 + 1.5 * X2 + e
    
    X = np.column_stack([X1, X2])
    weights = np.ones(n)
    
    # Fit coefficients
    X_with_intercept = np.column_stack([np.ones(n), X])
    theta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    
    return {
        'X': X,
        'y': y,
        'weights': weights,
        'X_with_intercept': X_with_intercept,
        'theta': theta,
        'true_beta': np.array([2.0, 3.0, 1.5]),
        'n': n,
        'k': 2,
    }


@pytest.fixture
def weighted_ols_data():
    """
    Weighted OLS data with varying frequency weights.
    
    y = 1 + 2*X1 - 0.5*X2 + 1.2*X3 + e
    """
    np.random.seed(123)
    n = 300
    
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    X3 = np.random.randn(n)
    e = np.random.randn(n) * 0.3
    
    y = 1.0 + 2.0 * X1 - 0.5 * X2 + 1.2 * X3 + e
    
    X = np.column_stack([X1, X2, X3])
    weights = np.random.randint(1, 10, n).astype(float)  # Frequency weights 1-9
    
    # Fit weighted coefficients
    X_with_intercept = np.column_stack([np.ones(n), X])
    sqrt_w = np.sqrt(weights).reshape(-1, 1)
    Xw = X_with_intercept * sqrt_w
    yw = (y * sqrt_w.flatten()).reshape(-1, 1)
    theta = np.linalg.lstsq(Xw, yw, rcond=None)[0].flatten()
    
    return {
        'X': X,
        'y': y,
        'weights': weights,
        'X_with_intercept': X_with_intercept,
        'theta': theta,
        'true_beta': np.array([1.0, 2.0, -0.5, 1.2]),
        'n': n,
        'k': 3,
        'n_obs': int(weights.sum()),
    }


@pytest.fixture
def clustered_data():
    """
    Clustered data with 20 clusters, cluster sizes 10-30.
    """
    np.random.seed(456)
    n_clusters = 20
    cluster_sizes = np.random.randint(10, 30, n_clusters)
    n = cluster_sizes.sum()
    
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    X3 = np.random.randn(n)
    e = np.random.randn(n) * 0.5
    y = 1.0 + 1.5 * X1 + 0.8 * X2 - 0.3 * X3 + e
    
    # Add cluster effect
    cluster_ids = np.repeat(np.arange(n_clusters), cluster_sizes)
    cluster_effect = np.repeat(np.random.randn(n_clusters) * 0.3, cluster_sizes)
    y = y + cluster_effect
    
    X = np.column_stack([X1, X2, X3])
    weights = np.ones(n)
    
    # Fit coefficients
    X_with_intercept = np.column_stack([np.ones(n), X])
    theta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    
    return {
        'X': X,
        'y': y,
        'weights': weights,
        'X_with_intercept': X_with_intercept,
        'theta': theta,
        'cluster_ids': cluster_ids,
        'true_beta': np.array([1.0, 1.5, 0.8, -0.3]),
        'n': n,
        'k': 3,
        'n_clusters': n_clusters,
    }


@pytest.fixture
def compressed_data():
    """
    Pre-compressed (aggregated) data simulating strata compression.
    
    This mimics data that has been grouped by strata with counts.
    """
    np.random.seed(789)
    n_strata = 100
    
    # Strata-level means
    X1_mean = np.random.randn(n_strata)
    X2_mean = np.random.randn(n_strata)
    
    # Strata counts (frequency weights)
    counts = np.random.randint(5, 20, n_strata).astype(float)
    
    # Generate y for each stratum
    e_mean = np.random.randn(n_strata) * 0.5
    y_mean = 0.5 + 1.5 * X1_mean + 0.8 * X2_mean + e_mean
    
    X = np.column_stack([X1_mean, X2_mean])
    y = y_mean
    weights = counts
    
    # Fit weighted coefficients
    X_with_intercept = np.column_stack([np.ones(n_strata), X])
    sqrt_w = np.sqrt(weights).reshape(-1, 1)
    Xw = X_with_intercept * sqrt_w
    yw = (y * sqrt_w.flatten()).reshape(-1, 1)
    theta = np.linalg.lstsq(Xw, yw, rcond=None)[0].flatten()
    
    return {
        'X': X,
        'y': y,
        'weights': weights,
        'X_with_intercept': X_with_intercept,
        'theta': theta,
        'true_beta': np.array([0.5, 1.5, 0.8]),
        'n_strata': n_strata,
        'n_obs': int(counts.sum()),
        'k': 2,
    }


@pytest.fixture
def iv_data():
    """
    IV data: Y = endogenous_X + X2 + X3 + e
    Instrument Z correlated with endogenous_X but not with e.
    """
    np.random.seed(999)
    n = 500
    
    # Instrument
    Z = np.random.randn(n)
    
    # Endogenous regressor correlated with Z and with error
    endogenous_X = 0.7 * Z + np.random.randn(n) * 0.3
    
    # Exogenous regressors
    X2 = np.random.randn(n)
    X3 = np.random.randn(n)
    
    # Error (correlated with endogenous_X)
    e = 0.3 * endogenous_X + np.random.randn(n) * 0.5
    
    # Generate y
    y = 1.0 * endogenous_X + 2.0 * X2 + 0.5 * X3 + e
    
    X = np.column_stack([endogenous_X, X2, X3])
    Z_mat = Z.reshape(-1, 1)
    weights = np.ones(n)
    
    # Fit coefficients (IV)
    X_with_intercept = np.column_stack([np.ones(n), X])
    Z_with_intercept = np.column_stack([np.ones(n), Z_mat, X2, X3])
    
    # Two-stage least squares
    # Stage 1: X = Z * pi + u
    pi = np.linalg.lstsq(Z_with_intercept, X_with_intercept[:, 1], rcond=None)[0]
    X_hat = Z_with_intercept @ pi
    
    # Stage 2: y = X_hat * theta + e
    X_stage2 = np.column_stack([np.ones(n), X_hat, X2, X3])
    theta = np.linalg.lstsq(X_stage2, y, rcond=None)[0]
    
    return {
        'X': X,
        'Z': Z_mat,
        'y': y,
        'weights': weights,
        'X_with_intercept': X_with_intercept,
        'Z_with_intercept': Z_with_intercept,
        'theta': theta,
        'true_beta': np.array([1.0, 2.0, 0.5]),
        'n': n,
        'k': 3,
    }


@pytest.fixture
def duckdb_simple_data(simple_ols_data):
    """Create DuckDB table from simple OLS data."""
    conn = duckdb.connect(":memory:")
    
    df = pd.DataFrame({
        'X1': simple_ols_data['X'][:, 0],
        'X2': simple_ols_data['X'][:, 1],
        'y': simple_ols_data['y'],
        'weight': simple_ols_data['weights'],
    })
    
    conn.execute("CREATE TABLE data AS SELECT * FROM df")
    
    return {'conn': conn, 'df': df, 'theta': simple_ols_data['theta']}


@pytest.fixture
def duckdb_weighted_data(weighted_ols_data):
    """Create DuckDB table from weighted OLS data in compressed format."""
    conn = duckdb.connect(":memory:")
    
    # Store in compressed format: sum_y = y * count
    # SQL backend expects y_col to be in sum format, divides by weight to get mean
    df = pd.DataFrame({
        'X1': weighted_ols_data['X'][:, 0],
        'X2': weighted_ols_data['X'][:, 1],
        'X3': weighted_ols_data['X'][:, 2],
        'sum_y': weighted_ols_data['y'] * weighted_ols_data['weights'],
        'count': weighted_ols_data['weights'],
    })
    
    # Add sum_y_sq for exact variance computation
    df['sum_y_sq'] = (weighted_ols_data['y'] ** 2) * weighted_ols_data['weights']
    
    conn.execute("CREATE TABLE weighted_data AS SELECT * FROM df")
    
    return {'conn': conn, 'df': df, 'theta': weighted_ols_data['theta']}


@pytest.fixture
def duckdb_clustered_data(clustered_data):
    """Create DuckDB table from clustered data."""
    conn = duckdb.connect(":memory:")
    
    df = pd.DataFrame({
        'X1': clustered_data['X'][:, 0],
        'X2': clustered_data['X'][:, 1],
        'X3': clustered_data['X'][:, 2],
        'y': clustered_data['y'],
        'weight': clustered_data['weights'],
        'cluster_id': clustered_data['cluster_ids'],
    })
    
    conn.execute("CREATE TABLE clustered_data AS SELECT * FROM df")
    
    return {'conn': conn, 'df': df, 'theta': clustered_data['theta']}


@pytest.fixture
def duckdb_compressed_data(compressed_data):
    """Create DuckDB table from compressed data (sum format)."""
    conn = duckdb.connect(":memory:")
    
    # For compressed data, store as sum_y = y * count
    # SQL backend expects y_col in sum format, divides by weight to get mean
    df = pd.DataFrame({
        'X1': compressed_data['X'][:, 0],
        'X2': compressed_data['X'][:, 1],
        'sum_y': compressed_data['y'] * compressed_data['weights'],
        'count': compressed_data['weights'],
    })
    
    # Add sum_y_sq for exact variance computation
    df['sum_y_sq'] = (compressed_data['y'] ** 2) * compressed_data['weights']
    
    conn.execute("CREATE TABLE compressed_data AS SELECT * FROM df")
    
    return {'conn': conn, 'df': df, 'theta': compressed_data['theta']}


# ============================================================================
# TESTS: NumPy Backend vs pyfixest patterns
# ============================================================================

class TestResidualAggregatesNumpyVsPyfixest:
    """Compare our NumPy implementation with pyfixest patterns."""
    
    def test_rss_simple_vs_pyfixest(self, simple_ols_data):
        """Test RSS computation matches pyfixest pattern."""
        # Reference: pyfixest pattern
        _, _, _, rss_ref = compute_residuals_scores_meat_pyfixest_style(
            X=simple_ols_data['X_with_intercept'],
            y=simple_ols_data['y'],
            weights=simple_ols_data['weights'],
            theta=simple_ols_data['theta']
        )
        
        # Our implementation
        result = compute_residual_aggregates_numpy(
            theta=simple_ols_data['theta'],
            X=simple_ols_data['X_with_intercept'],
            y=simple_ols_data['y'],
            weights=simple_ols_data['weights'],
            compute_rss=True,
        )
        
        np.testing.assert_allclose(result['rss'], rss_ref, rtol=1e-10, atol=1e-12)
    
    def test_scores_simple_vs_pyfixest(self, simple_ols_data):
        """Test score computation matches pyfixest pattern."""
        # Reference: pyfixest pattern
        _, scores_ref, _, _ = compute_residuals_scores_meat_pyfixest_style(
            X=simple_ols_data['X_with_intercept'],
            y=simple_ols_data['y'],
            weights=simple_ols_data['weights'],
            theta=simple_ols_data['theta']
        )
        
        # Our implementation
        result = compute_residual_aggregates_numpy(
            theta=simple_ols_data['theta'],
            X=simple_ols_data['X_with_intercept'],
            y=simple_ols_data['y'],
            weights=simple_ols_data['weights'],
            compute_scores=True,
        )
        
        np.testing.assert_allclose(result['scores'], scores_ref, rtol=1e-10, atol=1e-12)
    
    def test_meat_simple_vs_pyfixest(self, simple_ols_data):
        """Test meat matrix computation matches pyfixest pattern."""
        # Reference: pyfixest pattern
        _, _, meat_ref, _ = compute_residuals_scores_meat_pyfixest_style(
            X=simple_ols_data['X_with_intercept'],
            y=simple_ols_data['y'],
            weights=simple_ols_data['weights'],
            theta=simple_ols_data['theta']
        )
        
        # Our implementation
        result = compute_residual_aggregates_numpy(
            theta=simple_ols_data['theta'],
            X=simple_ols_data['X_with_intercept'],
            y=simple_ols_data['y'],
            weights=simple_ols_data['weights'],
            compute_meat=True,
        )
        
        np.testing.assert_allclose(result['meat'], meat_ref, rtol=1e-10, atol=1e-12)
    
    def test_weighted_ols_vs_pyfixest(self, weighted_ols_data):
        """Test weighted OLS with varying frequency weights matches pyfixest."""
        # Reference: pyfixest pattern
        residuals_ref, scores_ref, meat_ref, rss_ref = compute_residuals_scores_meat_pyfixest_style(
            X=weighted_ols_data['X_with_intercept'],
            y=weighted_ols_data['y'],
            weights=weighted_ols_data['weights'],
            theta=weighted_ols_data['theta']
        )
        
        # Our implementation
        result = compute_residual_aggregates_numpy(
            theta=weighted_ols_data['theta'],
            X=weighted_ols_data['X_with_intercept'],
            y=weighted_ols_data['y'],
            weights=weighted_ols_data['weights'],
            compute_rss=True,
            compute_scores=True,
            compute_meat=True,
        )
        
        # Compare all components
        np.testing.assert_allclose(result['rss'], rss_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(result['scores'], scores_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(result['meat'], meat_ref, rtol=1e-10, atol=1e-12)
    
    def test_cluster_scores_vs_pyfixest(self, clustered_data):
        """Test cluster score aggregation matches pyfixest pattern."""
        # First compute individual scores using pyfixest pattern
        _, scores_ref, _, _ = compute_residuals_scores_meat_pyfixest_style(
            X=clustered_data['X_with_intercept'],
            y=clustered_data['y'],
            weights=clustered_data['weights'],
            theta=clustered_data['theta']
        )
        
        # Aggregate by cluster using pyfixest pattern
        cluster_scores_ref, n_clusters_ref = compute_cluster_scores_pyfixest_style(
            scores=scores_ref,
            cluster_ids=clustered_data['cluster_ids']
        )
        
        # Our implementation
        result = compute_residual_aggregates_numpy(
            theta=clustered_data['theta'],
            X=clustered_data['X_with_intercept'],
            y=clustered_data['y'],
            weights=clustered_data['weights'],
            cluster_ids=clustered_data['cluster_ids'],
            compute_cluster_scores=True,
        )
        
        # Compare
        assert result['n_clusters'] == n_clusters_ref
        np.testing.assert_allclose(
            result['cluster_scores'], 
            cluster_scores_ref, 
            rtol=1e-10, 
            atol=1e-12
        )
    
    def test_iv_scores_vs_pyfixest(self, iv_data):
        """Test IV score computation (using Z not X) matches pyfixest pattern."""
        # Reference: pyfixest pattern with IV
        _, scores_ref, meat_ref, _ = compute_residuals_scores_meat_pyfixest_style(
            X=iv_data['X_with_intercept'],
            y=iv_data['y'],
            weights=iv_data['weights'],
            theta=iv_data['theta'],
            Z=iv_data['Z_with_intercept'],
            is_iv=True
        )
        
        # Our implementation
        result = compute_residual_aggregates_numpy(
            theta=iv_data['theta'],
            X=iv_data['X_with_intercept'],
            y=iv_data['y'],
            weights=iv_data['weights'],
            Z=iv_data['Z_with_intercept'],
            is_iv=True,
            compute_scores=True,
            compute_meat=True,
        )
        
        # Compare (scores should use Z, not X)
        np.testing.assert_allclose(result['scores'], scores_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(result['meat'], meat_ref, rtol=1e-10, atol=1e-12)
    
    def test_compressed_data_vs_pyfixest(self, compressed_data):
        """Test with pre-compressed (aggregated) data matches pyfixest pattern."""
        # Reference: pyfixest pattern
        _, scores_ref, meat_ref, rss_ref = compute_residuals_scores_meat_pyfixest_style(
            X=compressed_data['X_with_intercept'],
            y=compressed_data['y'],
            weights=compressed_data['weights'],
            theta=compressed_data['theta']
        )
        
        # Our implementation
        result = compute_residual_aggregates_numpy(
            theta=compressed_data['theta'],
            X=compressed_data['X_with_intercept'],
            y=compressed_data['y'],
            weights=compressed_data['weights'],
            compute_rss=True,
            compute_scores=True,
            compute_meat=True,
        )
        
        # Compare
        np.testing.assert_allclose(result['rss'], rss_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(result['scores'], scores_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(result['meat'], meat_ref, rtol=1e-10, atol=1e-12)


# ============================================================================
# TESTS: NumPy Backend - Basic Functionality
# ============================================================================

class TestResidualAggregatesNumpy:
    """Test NumPy backend for residual aggregate computation."""
    
    def test_rss_simple(self, simple_ols_data):
        """Test RSS computation with simple unweighted data."""
        result = compute_residual_aggregates_numpy(
            theta=simple_ols_data['theta'],
            X=simple_ols_data['X_with_intercept'],
            y=simple_ols_data['y'],
            weights=simple_ols_data['weights'],
            compute_rss=True,
        )
        
        # RSS should be positive
        assert result['rss'] > 0
        
        # Manually compute RSS
        residuals = simple_ols_data['y'] - simple_ols_data['X_with_intercept'] @ simple_ols_data['theta']
        expected_rss = np.sum(residuals ** 2)
        
        np.testing.assert_allclose(result['rss'], expected_rss, rtol=1e-10)
    
    def test_rss_weighted(self, weighted_ols_data):
        """Test RSS computation with frequency weights."""
        result = compute_residual_aggregates_numpy(
            theta=weighted_ols_data['theta'],
            X=weighted_ols_data['X_with_intercept'],
            y=weighted_ols_data['y'],
            weights=weighted_ols_data['weights'],
            compute_rss=True,
        )
        
        # Manually compute weighted RSS
        residuals = weighted_ols_data['y'] - weighted_ols_data['X_with_intercept'] @ weighted_ols_data['theta']
        expected_rss = np.sum((residuals * np.sqrt(weighted_ols_data['weights'])) ** 2)
        
        np.testing.assert_allclose(result['rss'], expected_rss, rtol=1e-10)
    
    def test_scores_simple(self, simple_ols_data):
        """Test score computation with simple data."""
        result = compute_residual_aggregates_numpy(
            theta=simple_ols_data['theta'],
            X=simple_ols_data['X_with_intercept'],
            y=simple_ols_data['y'],
            weights=simple_ols_data['weights'],
            compute_scores=True,
        )
        
        # Scores shape
        assert result['scores'].shape == (simple_ols_data['n'], 3)
        
        # Manually compute scores
        residuals = simple_ols_data['y'] - simple_ols_data['X_with_intercept'] @ simple_ols_data['theta']
        expected_scores = simple_ols_data['X_with_intercept'] * residuals.reshape(-1, 1)
        
        np.testing.assert_allclose(result['scores'], expected_scores, rtol=1e-10)
    
    def test_scores_weighted(self, weighted_ols_data):
        """Test score computation with weights."""
        result = compute_residual_aggregates_numpy(
            theta=weighted_ols_data['theta'],
            X=weighted_ols_data['X_with_intercept'],
            y=weighted_ols_data['y'],
            weights=weighted_ols_data['weights'],
            compute_scores=True,
        )
        
        # Manually compute weighted scores
        residuals = weighted_ols_data['y'] - weighted_ols_data['X_with_intercept'] @ weighted_ols_data['theta']
        expected_scores = weighted_ols_data['X_with_intercept'] * (residuals * weighted_ols_data['weights']).reshape(-1, 1)
        
        np.testing.assert_allclose(result['scores'], expected_scores, rtol=1e-10)
    
    def test_meat_matrix(self, simple_ols_data):
        """Test meat matrix computation."""
        result = compute_residual_aggregates_numpy(
            theta=simple_ols_data['theta'],
            X=simple_ols_data['X_with_intercept'],
            y=simple_ols_data['y'],
            weights=simple_ols_data['weights'],
            compute_meat=True,
        )
        
        # Meat should be symmetric
        np.testing.assert_allclose(result['meat'], result['meat'].T, rtol=1e-10)
        
        # Meat should be k x k
        assert result['meat'].shape == (3, 3)
        
        # Manually compute meat
        residuals = simple_ols_data['y'] - simple_ols_data['X_with_intercept'] @ simple_ols_data['theta']
        scores = simple_ols_data['X_with_intercept'] * residuals.reshape(-1, 1)
        expected_meat = scores.T @ scores
        
        np.testing.assert_allclose(result['meat'], expected_meat, rtol=1e-10)
    
    def test_cluster_scores(self, clustered_data):
        """Test cluster score computation."""
        result = compute_residual_aggregates_numpy(
            theta=clustered_data['theta'],
            X=clustered_data['X_with_intercept'],
            y=clustered_data['y'],
            weights=clustered_data['weights'],
            cluster_ids=clustered_data['cluster_ids'],
            compute_cluster_scores=True,
        )
        
        # Cluster scores shape: (n_clusters, k)
        assert result['cluster_scores'].shape == (clustered_data['n_clusters'], 4)
        assert result['n_clusters'] == clustered_data['n_clusters']
        
        # Verify cluster scores sum correctly
        residuals = clustered_data['y'] - clustered_data['X_with_intercept'] @ clustered_data['theta']
        scores = clustered_data['X_with_intercept'] * residuals.reshape(-1, 1)
        
        # Manually aggregate by cluster
        for cluster_idx in range(clustered_data['n_clusters']):
            mask = clustered_data['cluster_ids'] == cluster_idx
            expected_cluster_score = scores[mask].sum(axis=0)
            np.testing.assert_allclose(
                result['cluster_scores'][cluster_idx],
                expected_cluster_score,
                rtol=1e-10
            )
    
    def test_leverages(self, simple_ols_data):
        """Test leverage computation."""
        # Compute XtX_inv
        XtX = simple_ols_data['X_with_intercept'].T @ simple_ols_data['X_with_intercept']
        XtX_inv = np.linalg.inv(XtX)
        
        result = compute_residual_aggregates_numpy(
            theta=simple_ols_data['theta'],
            X=simple_ols_data['X_with_intercept'],
            y=simple_ols_data['y'],
            weights=simple_ols_data['weights'],
            XtX_inv=XtX_inv,
            compute_leverages=True,
        )
        
        # Leverages should be between 0 and 1
        assert np.all(result['leverages'] >= 0)
        assert np.all(result['leverages'] <= 1)
        
        # Manually compute leverages
        expected_leverages = np.sum((simple_ols_data['X_with_intercept'] @ XtX_inv) * simple_ols_data['X_with_intercept'], axis=1)
        
        np.testing.assert_allclose(result['leverages'], expected_leverages, rtol=1e-10)
    
    def test_precomputed_residuals(self, simple_ols_data):
        """Test using pre-computed residuals."""
        residuals = simple_ols_data['y'] - simple_ols_data['X_with_intercept'] @ simple_ols_data['theta']
        
        result1 = compute_residual_aggregates_numpy(
            theta=simple_ols_data['theta'],
            X=simple_ols_data['X_with_intercept'],
            y=simple_ols_data['y'],
            weights=simple_ols_data['weights'],
            residuals=residuals,
            compute_rss=True,
        )
        
        result2 = compute_residual_aggregates_numpy(
            theta=simple_ols_data['theta'],
            X=simple_ols_data['X_with_intercept'],
            y=simple_ols_data['y'],
            weights=simple_ols_data['weights'],
            compute_rss=True,
        )
        
        # Should give same result
        np.testing.assert_allclose(result1['rss'], result2['rss'], rtol=1e-10)
    
    def test_iv_scores(self, iv_data):
        """Test score computation with instruments for IV."""
        result = compute_residual_aggregates_numpy(
            theta=iv_data['theta'],
            X=iv_data['X_with_intercept'],
            y=iv_data['y'],
            weights=iv_data['weights'],
            Z=iv_data['Z_with_intercept'],
            is_iv=True,
            compute_scores=True,
        )
        
        # Scores should use Z (instruments), not X
        residuals = iv_data['y'] - iv_data['X_with_intercept'] @ iv_data['theta']
        expected_scores = iv_data['Z_with_intercept'] * residuals.reshape(-1, 1)
        
        np.testing.assert_allclose(result['scores'], expected_scores, rtol=1e-10)


# ============================================================================
# TESTS: SQL Backend - Basic Functionality
# ============================================================================

class TestResidualAggregatesSQL:
    """Test SQL backend for residual aggregate computation."""
    
    def test_rss_simple(self, duckdb_simple_data):
        """Test RSS computation via SQL with simple data."""
        result = compute_residual_aggregates_sql(
            theta=duckdb_simple_data['theta'],
            conn=duckdb_simple_data['conn'],
            table_name="data",
            x_cols=["X1", "X2"],
            y_col="y",
            weight_col="weight",
            add_intercept=True,
            compute_rss=True,
        )
        
        assert result['rss'] > 0
        
        # Compare with manual computation
        df = duckdb_simple_data['df']
        X = np.column_stack([np.ones(len(df)), df['X1'], df['X2']])
        residuals = df['y'].values - X @ duckdb_simple_data['theta']
        expected_rss = np.sum(residuals ** 2)
        
        np.testing.assert_allclose(result['rss'], expected_rss, rtol=1e-10)
    
    def test_rss_weighted(self, duckdb_weighted_data):
        """Test RSS computation via SQL with weights."""
        result = compute_residual_aggregates_sql(
            theta=duckdb_weighted_data['theta'],
            conn=duckdb_weighted_data['conn'],
            table_name="weighted_data",
            x_cols=["X1", "X2", "X3"],
            y_col="sum_y",
            weight_col="count",
            add_intercept=True,
            compute_rss=True,
        )
        
        # Compare with manual computation
        df = duckdb_weighted_data['df']
        X = np.column_stack([np.ones(len(df)), df['X1'], df['X2'], df['X3']])
        # y_mean = sum_y / count
        y_mean = df['sum_y'].values / df['count'].values
        residuals = y_mean - X @ duckdb_weighted_data['theta']
        expected_rss = np.sum((residuals * np.sqrt(df['count'].values)) ** 2)
        
        np.testing.assert_allclose(result['rss'], expected_rss, rtol=1e-10)
    
    def test_scores_simple(self, duckdb_simple_data):
        """Test score computation via SQL."""
        result = compute_residual_aggregates_sql(
            theta=duckdb_simple_data['theta'],
            conn=duckdb_simple_data['conn'],
            table_name="data",
            x_cols=["X1", "X2"],
            y_col="y",
            weight_col="weight",
            add_intercept=True,
            compute_scores=True,
        )
        
        # Scores shape
        assert result['scores'].shape == (len(duckdb_simple_data['df']), 3)
        
        # Compare with manual computation
        df = duckdb_simple_data['df']
        X = np.column_stack([np.ones(len(df)), df['X1'], df['X2']])
        residuals = df['y'].values - X @ duckdb_simple_data['theta']
        expected_scores = X * residuals.reshape(-1, 1)
        
        np.testing.assert_allclose(result['scores'], expected_scores, rtol=1e-9)
    
    def test_meat_matrix(self, duckdb_simple_data):
        """Test meat matrix computation via SQL."""
        result = compute_residual_aggregates_sql(
            theta=duckdb_simple_data['theta'],
            conn=duckdb_simple_data['conn'],
            table_name="data",
            x_cols=["X1", "X2"],
            y_col="y",
            weight_col="weight",
            add_intercept=True,
            compute_meat=True,
        )
        
        # Meat should be symmetric
        np.testing.assert_allclose(result['meat'], result['meat'].T, rtol=1e-10)
        
        # Compare with manual computation
        df = duckdb_simple_data['df']
        X = np.column_stack([np.ones(len(df)), df['X1'], df['X2']])
        residuals = df['y'].values - X @ duckdb_simple_data['theta']
        scores = X * residuals.reshape(-1, 1)
        expected_meat = scores.T @ scores
        
        np.testing.assert_allclose(result['meat'], expected_meat, rtol=1e-9)
    
    def test_cluster_scores(self, duckdb_clustered_data):
        """Test cluster score computation via SQL."""
        result = compute_residual_aggregates_sql(
            theta=duckdb_clustered_data['theta'],
            conn=duckdb_clustered_data['conn'],
            table_name="clustered_data",
            x_cols=["X1", "X2", "X3"],
            y_col="y",
            weight_col="weight",
            cluster_col="cluster_id",
            add_intercept=True,
            compute_cluster_scores=True,
        )
        
        # Verify shape and count
        assert 'cluster_scores' in result
        assert 'n_clusters' in result
        assert result['cluster_scores'].shape[1] == 4
        
        # Compare with manual computation
        df = duckdb_clustered_data['df']
        X = np.column_stack([np.ones(len(df)), df['X1'], df['X2'], df['X3']])
        residuals = df['y'].values - X @ duckdb_clustered_data['theta']
        scores = X * residuals.reshape(-1, 1)
        
        # Aggregate manually by cluster
        unique_clusters = df['cluster_id'].unique()
        for cluster_idx, cluster_id in enumerate(sorted(unique_clusters)):
            mask = df['cluster_id'] == cluster_id
            expected_cluster_score = scores[mask].sum(axis=0)
            np.testing.assert_allclose(
                result['cluster_scores'][cluster_idx],
                expected_cluster_score,
                rtol=1e-9
            )
    
    def test_leverages(self, duckdb_simple_data):
        """Test leverage computation via SQL."""
        # Compute XtX_inv
        df = duckdb_simple_data['df']
        X = np.column_stack([np.ones(len(df)), df['X1'], df['X2']])
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        
        result = compute_residual_aggregates_sql(
            theta=duckdb_simple_data['theta'],
            conn=duckdb_simple_data['conn'],
            table_name="data",
            x_cols=["X1", "X2"],
            y_col="y",
            weight_col="weight",
            add_intercept=True,
            XtX_inv=XtX_inv,
            compute_leverages=True,
        )
        
        # Leverages should be between 0 and 1
        assert np.all(result['leverages'] >= 0)
        assert np.all(result['leverages'] <= 1)
        
        # Compare with manual computation
        expected_leverages = np.sum((X @ XtX_inv) * X, axis=1)
        
        np.testing.assert_allclose(result['leverages'], expected_leverages, rtol=1e-9)


# ============================================================================
# TESTS: Cross-Backend Consistency
# ============================================================================

class TestCrossBackendConsistency:
    """Verify NumPy and SQL backends produce identical results."""
    
    def test_rss_consistency(self, simple_ols_data, duckdb_simple_data):
        """Test RSS consistency between backends."""
        # NumPy
        result_numpy = compute_residual_aggregates_numpy(
            theta=simple_ols_data['theta'],
            X=simple_ols_data['X_with_intercept'],
            y=simple_ols_data['y'],
            weights=simple_ols_data['weights'],
            compute_rss=True,
        )
        
        # SQL
        result_sql = compute_residual_aggregates_sql(
            theta=duckdb_simple_data['theta'],
            conn=duckdb_simple_data['conn'],
            table_name="data",
            x_cols=["X1", "X2"],
            y_col="y",
            weight_col="weight",
            add_intercept=True,
            compute_rss=True,
        )
        
        np.testing.assert_allclose(result_numpy['rss'], result_sql['rss'], rtol=1e-10)
    
    def test_scores_consistency(self, simple_ols_data, duckdb_simple_data):
        """Test score consistency between backends."""
        # NumPy
        result_numpy = compute_residual_aggregates_numpy(
            theta=simple_ols_data['theta'],
            X=simple_ols_data['X_with_intercept'],
            y=simple_ols_data['y'],
            weights=simple_ols_data['weights'],
            compute_scores=True,
        )
        
        # SQL
        result_sql = compute_residual_aggregates_sql(
            theta=duckdb_simple_data['theta'],
            conn=duckdb_simple_data['conn'],
            table_name="data",
            x_cols=["X1", "X2"],
            y_col="y",
            weight_col="weight",
            add_intercept=True,
            compute_scores=True,
        )
        
        np.testing.assert_allclose(result_numpy['scores'], result_sql['scores'], rtol=1e-9)
    
    def test_meat_consistency(self, simple_ols_data, duckdb_simple_data):
        """Test meat matrix consistency between backends."""
        # NumPy
        result_numpy = compute_residual_aggregates_numpy(
            theta=simple_ols_data['theta'],
            X=simple_ols_data['X_with_intercept'],
            y=simple_ols_data['y'],
            weights=simple_ols_data['weights'],
            compute_meat=True,
        )
        
        # SQL
        result_sql = compute_residual_aggregates_sql(
            theta=duckdb_simple_data['theta'],
            conn=duckdb_simple_data['conn'],
            table_name="data",
            x_cols=["X1", "X2"],
            y_col="y",
            weight_col="weight",
            add_intercept=True,
            compute_meat=True,
        )
        
        np.testing.assert_allclose(result_numpy['meat'], result_sql['meat'], rtol=1e-9)
    
    def test_weighted_consistency(self, weighted_ols_data, duckdb_weighted_data):
        """Test consistency with weighted data."""
        # NumPy
        result_numpy = compute_residual_aggregates_numpy(
            theta=weighted_ols_data['theta'],
            X=weighted_ols_data['X_with_intercept'],
            y=weighted_ols_data['y'],
            weights=weighted_ols_data['weights'],
            compute_rss=True,
            compute_meat=True,
        )
        
        # SQL
        result_sql = compute_residual_aggregates_sql(
            theta=duckdb_weighted_data['theta'],
            conn=duckdb_weighted_data['conn'],
            table_name="weighted_data",
            x_cols=["X1", "X2", "X3"],
            y_col="sum_y",
            weight_col="count",
            add_intercept=True,
            compute_rss=True,
            compute_meat=True,
        )
        
        np.testing.assert_allclose(result_numpy['rss'], result_sql['rss'], rtol=1e-10)
        np.testing.assert_allclose(result_numpy['meat'], result_sql['meat'], rtol=1e-9)


# ============================================================================
# TESTS: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_observation(self):
        """Test with single observation."""
        X = np.array([[1.0, 2.0]])
        y = np.array([5.0])
        weights = np.array([1.0])
        theta = np.array([1.0, 2.0])
        
        result = compute_residual_aggregates_numpy(
            theta=theta,
            X=X,
            y=y,
            weights=weights,
            compute_rss=True,
            compute_scores=True,
        )
        
        # Should work without errors
        assert result['rss'] >= 0
        assert result['scores'].shape == (1, 2)
    
    def test_zero_weights(self):
        """Test behavior with some zero weights."""
        X = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
        y = np.array([5.0, 7.0, 9.0])
        weights = np.array([1.0, 0.0, 1.0])  # Middle observation has zero weight
        theta = np.array([1.0, 2.0])
        
        result = compute_residual_aggregates_numpy(
            theta=theta,
            X=X,
            y=y,
            weights=weights,
            compute_rss=True,
        )
        
        # RSS should only include non-zero weighted observations
        residuals = y - X @ theta
        expected_rss = (residuals[0] ** 2) * weights[0] + (residuals[2] ** 2) * weights[2]
        
        np.testing.assert_allclose(result['rss'], expected_rss, rtol=1e-10)
    
    def test_missing_cluster_ids(self):
        """Test error when cluster_ids missing but compute_cluster_scores=True."""
        X = np.array([[1.0, 2.0], [1.0, 3.0]])
        y = np.array([5.0, 7.0])
        weights = np.array([1.0, 1.0])
        theta = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="cluster_ids required for compute_cluster_scores"):
            compute_residual_aggregates_numpy(
                theta=theta,
                X=X,
                y=y,
                weights=weights,
                compute_cluster_scores=True,
            )
    
    def test_missing_xtx_inv(self):
        """Test error when XtX_inv missing but compute_leverages=True."""
        X = np.array([[1.0, 2.0], [1.0, 3.0]])
        y = np.array([5.0, 7.0])
        weights = np.array([1.0, 1.0])
        theta = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="XtX_inv required for compute_leverages"):
            compute_residual_aggregates_numpy(
                theta=theta,
                X=X,
                y=y,
                weights=weights,
                compute_leverages=True,
            )
    
    def test_all_aggregates_at_once(self, simple_ols_data):
        """Test computing all aggregates in a single call."""
        # Compute XtX_inv for leverages
        XtX = simple_ols_data['X_with_intercept'].T @ simple_ols_data['X_with_intercept']
        XtX_inv = np.linalg.inv(XtX)
        
        result = compute_residual_aggregates_numpy(
            theta=simple_ols_data['theta'],
            X=simple_ols_data['X_with_intercept'],
            y=simple_ols_data['y'],
            weights=simple_ols_data['weights'],
            XtX_inv=XtX_inv,
            compute_rss=True,
            compute_scores=True,
            compute_meat=True,
            compute_leverages=True,
        )
        
        # All components should be present
        assert 'rss' in result
        assert 'scores' in result
        assert 'meat' in result
        assert 'leverages' in result
        
        # Verify relationships
        # meat = scores.T @ scores
        expected_meat = result['scores'].T @ result['scores']
        np.testing.assert_allclose(result['meat'], expected_meat, rtol=1e-10)


# ============================================================================
# TESTS: Numerical Precision
# ============================================================================

class TestNumericalPrecision:
    """Test numerical precision and stability."""
    
    def test_high_precision_rss(self, simple_ols_data):
        """Test that RSS is computed with high precision."""
        result1 = compute_residual_aggregates_numpy(
            theta=simple_ols_data['theta'],
            X=simple_ols_data['X_with_intercept'],
            y=simple_ols_data['y'],
            weights=simple_ols_data['weights'],
            compute_rss=True,
        )
        
        result2 = compute_residual_aggregates_numpy(
            theta=simple_ols_data['theta'],
            X=simple_ols_data['X_with_intercept'],
            y=simple_ols_data['y'],
            weights=simple_ols_data['weights'],
            compute_rss=True,
        )
        
        # Should be exactly identical (not just close)
        assert result1['rss'] == result2['rss']
    
    def test_meat_symmetry(self, weighted_ols_data):
        """Test that meat matrix is exactly symmetric."""
        result = compute_residual_aggregates_numpy(
            theta=weighted_ols_data['theta'],
            X=weighted_ols_data['X_with_intercept'],
            y=weighted_ols_data['y'],
            weights=weighted_ols_data['weights'],
            compute_meat=True,
        )
        
        # Check exact symmetry
        np.testing.assert_array_equal(result['meat'], result['meat'].T)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
