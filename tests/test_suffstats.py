"""
Test sufficient statistics computation against pyfixest reference implementation.

This module tests our compute_sufficient_stats implementation by comparing
results with manual computations using patterns from pyfixest.estimation.feols_.

The pyfixest code (feols_.py) is kept as a reference implementation and remains
unchanged. We extract the core computation patterns and compare against our
unified API.
"""

import numpy as np
import pandas as pd
import duckdb
import pytest

from duckreg.core.suffstats import (
    compute_sufficient_stats_numpy,
    compute_sufficient_stats_sql,
)
from duckreg.core.linalg import DEFAULT_ALPHA


# ============================================================================
# Reference Implementation: pyfixest patterns
# ============================================================================

def compute_XtX_Xty_pyfixest_style(X: np.ndarray, y: np.ndarray, weights: np.ndarray):
    """
    Compute X'WX and X'Wy using pyfixest patterns.
    
    Extracted from pyfixest.estimation.feols_ weighted regression logic.
    This serves as our reference implementation.
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix (n, k) 
    y : np.ndarray
        Response vector (n,)
    weights : np.ndarray
        Frequency weights (n,)
        
    Returns
    -------
    XtX : np.ndarray (k, k)
    Xty : np.ndarray (k,)
    n_obs : int
    sum_y : float
    sum_y_sq : float
    """
    # Ensure proper shapes
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    weights = weights.flatten()
    
    # Weighted computation: weight each observation
    # For frequency weights: sqrt(w) transforms preserve sum properties
    sqrt_w = np.sqrt(weights).reshape(-1, 1)
    Xw = X * sqrt_w
    yw = y * sqrt_w
    
    # Compute sufficient statistics
    XtX = Xw.T @ Xw
    Xty = (Xw.T @ yw).flatten()
    
    # Summary statistics
    n_obs = int(weights.sum())
    sum_y = (y.flatten() * weights).sum()
    sum_y_sq = ((y.flatten() ** 2) * weights).sum()
    
    return XtX, Xty, n_obs, sum_y, sum_y_sq


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
    
    return {
        'X': X,
        'y': y,
        'weights': weights,
        'true_beta': np.array([2.0, 3.0, 1.5]),  # [intercept, X1, X2]
        'n': n,
        'k': 2,  # excluding intercept
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
    
    return {
        'X': X,
        'y': y,
        'weights': weights,
        'true_beta': np.array([1.0, 2.0, -0.5, 1.2]),
        'n': n,
        'k': 3,
        'n_obs': int(weights.sum()),
    }


@pytest.fixture
def compressed_data():
    """
    Pre-compressed (aggregated) data simulating strata compression.
    
    This mimics data that has been grouped by strata with counts.
    """
    np.random.seed(456)
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
    
    return {
        'X': X,
        'y': y,
        'weights': weights,
        'true_beta': np.array([0.5, 1.5, 0.8]),
        'n_strata': n_strata,
        'n_obs': int(counts.sum()),
        'k': 2,
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
    
    return {'conn': conn, 'df': df}


@pytest.fixture
def duckdb_weighted_data(weighted_ols_data):
    """Create DuckDB table from weighted OLS data in compressed format."""
    conn = duckdb.connect(":memory:")
    
    # Store in compressed format: sum_y = y * count
    df = pd.DataFrame({
        'X1': weighted_ols_data['X'][:, 0],
        'X2': weighted_ols_data['X'][:, 1],
        'X3': weighted_ols_data['X'][:, 2],
        'sum_y': weighted_ols_data['y'] * weighted_ols_data['weights'],  # Compressed format
        'count': weighted_ols_data['weights'],
    })
    
    # Add sum_y_sq for exact variance computation
    df['sum_y_sq'] = (weighted_ols_data['y'] ** 2) * weighted_ols_data['weights']
    
    conn.execute("CREATE TABLE weighted_data AS SELECT * FROM df")
    
    return {'conn': conn, 'df': df}


@pytest.fixture
def duckdb_compressed_data(compressed_data):
    """Create DuckDB table from compressed data (sum format)."""
    conn = duckdb.connect(":memory:")
    
    # For compressed data, store as sum_y (mimics our compression output)
    df = pd.DataFrame({
        'X1': compressed_data['X'][:, 0],
        'X2': compressed_data['X'][:, 1],
        'sum_y': compressed_data['y'] * compressed_data['weights'],  # sum format
        'count': compressed_data['weights'],
    })
    
    # Add sum_y_sq for exact variance computation
    df['sum_y_sq'] = (compressed_data['y'] ** 2) * compressed_data['weights']
    
    conn.execute("CREATE TABLE compressed_data AS SELECT * FROM df")
    
    return {'conn': conn, 'df': df}


# ============================================================================
# TESTS: NumPy Backend vs pyfixest patterns
# ============================================================================

class TestSuffstatsNumpyVsPyfixest:
    """Compare our NumPy implementation with pyfixest patterns."""
    
    def test_simple_ols_no_weights(self, simple_ols_data):
        """Test simple OLS with unit weights."""
        X = simple_ols_data['X']
        y = simple_ols_data['y']
        weights = simple_ols_data['weights']
        
        # Add intercept for both methods
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # Reference: pyfixest pattern
        XtX_ref, Xty_ref, n_obs_ref, sum_y_ref, sum_y_sq_ref = \
            compute_XtX_Xty_pyfixest_style(X_with_intercept, y, weights)
        
        # Our implementation
        XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names = compute_sufficient_stats_numpy(
            X=X_with_intercept,
            y=y,
            weights=weights,
            alpha=0.0  # No regularization for comparison
        )
        
        # Compare
        np.testing.assert_allclose(XtX, XtX_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(Xty, Xty_ref, rtol=1e-10, atol=1e-12)
        assert n_obs == n_obs_ref
        np.testing.assert_allclose(sum_y, sum_y_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(sum_y_sq, sum_y_sq_ref, rtol=1e-10, atol=1e-12)
    
    def test_weighted_ols(self, weighted_ols_data):
        """Test weighted OLS with varying frequency weights."""
        X = weighted_ols_data['X']
        y = weighted_ols_data['y']
        weights = weighted_ols_data['weights']
        
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # Reference
        XtX_ref, Xty_ref, n_obs_ref, sum_y_ref, sum_y_sq_ref = \
            compute_XtX_Xty_pyfixest_style(X_with_intercept, y, weights)
        
        # Our implementation
        XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names = compute_sufficient_stats_numpy(
            X=X_with_intercept,
            y=y,
            weights=weights,
            alpha=0.0
        )
        
        # Compare
        np.testing.assert_allclose(XtX, XtX_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(Xty, Xty_ref, rtol=1e-10, atol=1e-12)
        assert n_obs == n_obs_ref
        np.testing.assert_allclose(sum_y, sum_y_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(sum_y_sq, sum_y_sq_ref, rtol=1e-10, atol=1e-12)
    
    def test_compressed_strata_data(self, compressed_data):
        """Test with pre-compressed (aggregated) data."""
        X = compressed_data['X']
        y = compressed_data['y']
        weights = compressed_data['weights']
        
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # Reference
        XtX_ref, Xty_ref, n_obs_ref, sum_y_ref, sum_y_sq_ref = \
            compute_XtX_Xty_pyfixest_style(X_with_intercept, y, weights)
        
        # Our implementation
        XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names = compute_sufficient_stats_numpy(
            X=X_with_intercept,
            y=y,
            weights=weights,
            alpha=0.0
        )
        
        # Compare
        np.testing.assert_allclose(XtX, XtX_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(Xty, Xty_ref, rtol=1e-10, atol=1e-12)
        assert n_obs == n_obs_ref
        assert n_obs == compressed_data['n_obs']




# ============================================================================
# TESTS: Cross-Backend Consistency
# ============================================================================

class TestCrossBackendConsistency:
    """Verify NumPy and SQL backends produce identical results."""
    
    def test_simple_ols_consistency(self, simple_ols_data, duckdb_simple_data):
        """Compare NumPy and SQL backends on simple OLS data."""
        # NumPy backend
        X = simple_ols_data['X']
        y = simple_ols_data['y']
        weights = simple_ols_data['weights']
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        XtX_np, Xty_np, n_obs_np, sum_y_np, sum_y_sq_np, coef_names_np = \
            compute_sufficient_stats_numpy(
                X=X_with_intercept,
                y=y,
                weights=weights,
                alpha=0.0
            )
        
        # SQL backend
        conn = duckdb_simple_data['conn']
        XtX_sql, Xty_sql, n_obs_sql, sum_y_sql, sum_y_sq_sql, coef_names_sql = \
            compute_sufficient_stats_sql(
                conn=conn,
                table_name='data',
                x_cols=['X1', 'X2'],
                y_col='y',
                weight_col='weight',
                add_intercept=True,
                alpha=0.0
            )
        
        # Compare (allow small numerical differences from SQL arithmetic)
        np.testing.assert_allclose(XtX_np, XtX_sql, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(Xty_np, Xty_sql, rtol=1e-10, atol=1e-12)
        assert n_obs_np == n_obs_sql
        np.testing.assert_allclose(sum_y_np, sum_y_sql, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(sum_y_sq_np, sum_y_sq_sql, rtol=1e-10, atol=1e-12)
    
    def test_weighted_ols_consistency(self, weighted_ols_data, duckdb_weighted_data):
        """Compare NumPy and SQL backends on weighted OLS data."""
        # NumPy backend
        X = weighted_ols_data['X']
        y = weighted_ols_data['y']
        weights = weighted_ols_data['weights']
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        XtX_np, Xty_np, n_obs_np, sum_y_np, sum_y_sq_np, _ = \
            compute_sufficient_stats_numpy(
                X=X_with_intercept,
                y=y,
                weights=weights,
                alpha=0.0
            )
        
        # SQL backend (expects compressed format: sum_y, not raw y)
        conn = duckdb_weighted_data['conn']
        XtX_sql, Xty_sql, n_obs_sql, sum_y_sql, sum_y_sq_sql, _ = \
            compute_sufficient_stats_sql(
                conn=conn,
                table_name='weighted_data',
                x_cols=['X1', 'X2', 'X3'],
                y_col='sum_y',  # Use sum_y for compressed data
                weight_col='count',
                add_intercept=True,
                alpha=0.0,
                sum_y_sq_col='sum_y_sq'  # Use exact sum_y_sq
            )
        
        # Compare
        np.testing.assert_allclose(XtX_np, XtX_sql, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(Xty_np, Xty_sql, rtol=1e-10, atol=1e-12)
        assert n_obs_np == n_obs_sql
        np.testing.assert_allclose(sum_y_np, sum_y_sql, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(sum_y_sq_np, sum_y_sq_sql, rtol=1e-10, atol=1e-12)
    
    def test_compressed_data_with_sum_format(self, duckdb_compressed_data):
        """Test SQL backend with sum_y format (compressed data)."""
        conn = duckdb_compressed_data['conn']
        
        # SQL backend with sum_y column
        XtX_sql, Xty_sql, n_obs_sql, sum_y_sql, sum_y_sq_sql, coef_names = \
            compute_sufficient_stats_sql(
                conn=conn,
                table_name='compressed_data',
                x_cols=['X1', 'X2'],
                y_col='sum_y',  # Already in sum format
                weight_col='count',
                add_intercept=True,
                alpha=0.0,
                sum_y_sq_col='sum_y_sq'  # Use exact sum_y_sq
            )
        
        # Verify shapes
        assert XtX_sql.shape == (3, 3)  # intercept + X1 + X2
        assert Xty_sql.shape == (3,)
        assert n_obs_sql > 0
        assert sum_y_sql > 0 or sum_y_sql < 0  # Non-zero
        assert sum_y_sq_sql > 0  # Always positive


# ============================================================================
# TESTS: Regularization
# ============================================================================

class TestRegularization:
    """Test regularization parameter (alpha) handling."""
    
    def test_regularization_effect(self, simple_ols_data):
        """Verify alpha adds to diagonal of XtX."""
        X = simple_ols_data['X']
        y = simple_ols_data['y']
        weights = simple_ols_data['weights']
        
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        k = X_with_intercept.shape[1]
        
        # Without regularization
        XtX_no_reg, _, _, _, _, _ = compute_sufficient_stats_numpy(
            X=X_with_intercept,
            y=y,
            weights=weights,
            alpha=0.0
        )
        
        # With regularization
        alpha = 1e-5
        XtX_reg, _, _, _, _, _ = compute_sufficient_stats_numpy(
            X=X_with_intercept,
            y=y,
            weights=weights,
            alpha=alpha
        )
        
        # Difference should be alpha * I
        diff = XtX_reg - XtX_no_reg
        expected = alpha * np.eye(k)
        
        np.testing.assert_allclose(diff, expected, rtol=1e-10, atol=1e-12)
    
    def test_default_alpha(self, simple_ols_data):
        """Test with default alpha value."""
        X = simple_ols_data['X']
        y = simple_ols_data['y']
        weights = simple_ols_data['weights']
        
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # Should use DEFAULT_ALPHA
        XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names = compute_sufficient_stats_numpy(
            X=X_with_intercept,
            y=y,
            weights=weights
        )
        
        # Verify it ran without error and produced sensible results
        assert XtX.shape == (3, 3)
        assert Xty.shape == (3,)
        assert n_obs == len(y)


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
        
        XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names = compute_sufficient_stats_numpy(
            X=X,
            y=y,
            weights=weights,
            alpha=0.0
        )
        
        assert XtX.shape == (2, 2)
        assert Xty.shape == (2,)
        assert n_obs == 1
        assert sum_y == 5.0
        assert sum_y_sq == 25.0
    
    def test_missing_required_params_numpy(self):
        """Test error handling for missing required parameters."""
        # compute_sufficient_stats_numpy expects non-None X, y, weights
        # Direct call will fail with appropriate error
        with pytest.raises((ValueError, TypeError)):
            compute_sufficient_stats_numpy(
                X=None,  # Missing required param
                y=np.array([1, 2, 3]),
                weights=np.ones(3)
            )
    
    def test_missing_required_params_sql(self):
        """Test error handling for missing SQL parameters."""
        conn = duckdb.connect(":memory:")
        
        with pytest.raises((ValueError, TypeError, duckdb.CatalogException)):
            compute_sufficient_stats_sql(
                conn=conn,
                table_name=None,  # Missing required param
                x_cols=['X1'],
                y_col='y',
                weight_col='weight'
            )



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
