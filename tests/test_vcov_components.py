"""
Comprehensive tests for VCOV (variance-covariance) components.

Tests all VCOV computation functions with simulated data and validates
against pyfixest reference implementations.

Test Coverage:
- SSC computation (all k_fixef options, cluster adjustments)
- IID variance
- Heteroskedastic-robust variance (HC1, HC2, HC3)
- Cluster-robust variance (CRV1)
- VCOV parsing and validation
- Configuration dataclasses
"""
import numpy as np
import pandas as pd
import pytest
import duckdb
import logging
from typing import Tuple

from duckreg.core.vcov import (
    SSCConfig,
    VcovContext,
    parse_vcov_specification,
    parse_cluster_vars,
    compute_bread,
    compute_ssc,
    vcov_iid,
    compute_iid_vcov,
    vcov_hetero,
    compute_hetero_vcov,
    vcov_crv1,
    compute_cluster_vcov,
    compute_cluster_scores,
    VcovTypeNotSupportedError,
)

from duckreg.core.linalg import safe_solve, safe_inv

# Import pyfixest references for validation
import sys
sys.path.insert(0, '/scicore/home/meiera/schulz0022/projects/duckreg/foreign')
from pyfixest.utils import get_ssc as pyfixest_get_ssc

logger = logging.getLogger(__name__)


# ============================================================================
# FIXTURES: Data Generation
# ============================================================================

@pytest.fixture
def simple_data():
    """Simple unweighted data: y = 1*X1 + 2*X2 + 0.5*X3 + e"""
    np.random.seed(42)
    n = 500
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    X3 = np.random.randn(n)
    e = np.random.randn(n) * 0.5
    y = 1.0 * X1 + 2.0 * X2 + 0.5 * X3 + e
    
    X = np.column_stack([X1, X2, X3])
    weights = np.ones(n)
    
    return {
        'X': X,
        'y': y,
        'weights': weights,
        'true_beta': np.array([1.0, 2.0, 0.5]),
        'n': n,
        'k': 3,
    }


@pytest.fixture
def weighted_data():
    """Weighted data with frequency weights"""
    np.random.seed(42)
    n = 500
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    X3 = np.random.randn(n)
    e = np.random.randn(n) * 0.5
    y = 1.0 * X1 + 2.0 * X2 + 0.5 * X3 + e
    
    X = np.column_stack([X1, X2, X3])
    weights = np.random.randint(1, 5, n)
    
    return {
        'X': X,
        'y': y,
        'weights': weights,
        'true_beta': np.array([1.0, 2.0, 0.5]),
        'n': n,
        'n_obs': int(weights.sum()),
        'k': 3,
    }


@pytest.fixture
def clustered_data():
    """Clustered data with 20 clusters"""
    np.random.seed(42)
    n_clusters = 20
    cluster_sizes = np.random.randint(10, 30, n_clusters)
    n = cluster_sizes.sum()
    
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    X3 = np.random.randn(n)
    e = np.random.randn(n) * 0.5
    y = 1.0 * X1 + 2.0 * X2 + 0.5 * X3 + e
    
    cluster_ids = np.repeat(np.arange(n_clusters), cluster_sizes)
    cluster_effect = np.repeat(np.random.randn(n_clusters) * 0.2, cluster_sizes)
    y = y + cluster_effect
    
    X = np.column_stack([X1, X2, X3])
    weights = np.ones(n)
    
    return {
        'X': X,
        'y': y,
        'weights': weights,
        'cluster_ids': cluster_ids,
        'true_beta': np.array([1.0, 2.0, 0.5]),
        'n': n,
        'k': 3,
        'n_clusters': n_clusters,
    }


@pytest.fixture
def heteroskedastic_data():
    """Heteroskedastic data: error variance increases with X1"""
    np.random.seed(42)
    n = 500
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    X3 = np.random.randn(n)
    
    error_sd = 0.5 * (1 + np.abs(X1))
    e = np.random.randn(n) * error_sd
    y = 1.0 * X1 + 2.0 * X2 + 0.5 * X3 + e
    
    X = np.column_stack([X1, X2, X3])
    weights = np.ones(n)
    
    return {
        'X': X,
        'y': y,
        'weights': weights,
        'true_beta': np.array([1.0, 2.0, 0.5]),
        'n': n,
        'k': 3,
        'error_sd': error_sd,
    }


# ============================================================================
# TESTS: Configuration Dataclasses
# ============================================================================

class TestConfigDataclasses:
    """Test SSCConfig and VcovContext dataclasses."""
    
    def test_ssc_config_defaults(self):
        """Test SSCConfig default values."""
        config = SSCConfig()
        assert config.k_adj == True
        assert config.k_fixef == "full"
        assert config.G_adj == True
        assert config.G_df == "conventional"
    
    def test_ssc_config_from_dict(self):
        """Test SSCConfig.from_dict() method."""
        ssc_dict = {'k_adj': False, 'k_fixef': 'none', 'G_adj': False, 'G_df': 'min'}
        config = SSCConfig.from_dict(ssc_dict)
        assert config.k_adj == False
        assert config.k_fixef == 'none'
        assert config.G_adj == False
        assert config.G_df == 'min'
    
    def test_vcov_context_creation(self):
        """Test VcovContext creation."""
        context = VcovContext(N=500, k=3, k_fe=10, n_fe=2)
        assert context.N == 500
        assert context.k == 3
        assert context.k_fe == 10
        assert context.n_fe == 2
        assert context.k_fe_nested == 0
        assert context.n_fe_fully_nested == 0


# ============================================================================
# TESTS: VCOV Parsing and Validation
# ============================================================================

class TestVcovParsing:
    """Tests for VCOV specification parsing and validation."""
    
    def test_parse_vcov_iid(self):
        """Parse IID vcov specification."""
        vcov_type, detail, is_clustered, clustervars = parse_vcov_specification("iid")
        assert vcov_type == "iid"
        assert detail == "iid"
        assert not is_clustered
        assert clustervars is None
    
    def test_parse_vcov_hetero_string(self):
        """Parse heteroskedastic vcov specifications."""
        for spec in ["hetero", "HC1", "HC2", "HC3"]:
            vcov_type, detail, is_clustered, clustervars = parse_vcov_specification(spec)
            assert vcov_type == "hetero"
            assert detail == spec
            assert not is_clustered
            assert clustervars is None
    
    def test_parse_vcov_crv1(self):
        """Parse CRV1 cluster specification."""
        vcov_type, detail, is_clustered, clustervars = parse_vcov_specification(
            {"CRV1": "cluster"}
        )
        assert vcov_type == "CRV"
        assert detail == "CRV1"
        assert is_clustered
        assert clustervars == ["cluster"]
    
    def test_parse_vcov_crv1_twoway(self):
        """Parse two-way CRV specification."""
        vcov_type, detail, is_clustered, clustervars = parse_vcov_specification(
            {"CRV1": "state+firm"}
        )
        assert vcov_type == "CRV"
        assert detail == "CRV1"
        assert is_clustered
        assert clustervars == ["state", "firm"]
    
    def test_parse_cluster_vars(self):
        """Extract cluster variables from vcov."""
        clustervars = parse_cluster_vars({"CRV1": "cluster"})
        assert clustervars == ["cluster"]
        
        clustervars = parse_cluster_vars("HC1")
        assert clustervars is None
    
    def test_parse_vcov_invalid_type(self):
        """Raise error for invalid vcov type."""
        with pytest.raises(Exception):
            parse_vcov_specification("INVALID_TYPE")


# ============================================================================
# TESTS: SSC Computation vs pyfixest
# ============================================================================

class TestSSCComputation:
    """Tests for small sample correction computation against pyfixest."""
    
    def test_ssc_iid_vs_pyfixest(self):
        """Compare SSC for IID with pyfixest."""
        N, k, k_fe, n_fe = 500, 3, 0, 0
        
        # Our implementation
        ssc_config = SSCConfig(k_adj=True, k_fixef='none', G_adj=True, G_df='conventional')
        context = VcovContext(N=N, k=k, k_fe=k_fe, n_fe=n_fe)
        ssc, df_k, df_t = compute_ssc(ssc_config, context, G=1, vcov_type="iid", vcov_sign=1)
        
        # pyfixest reference
        ssc_dict = {'k_adj': True, 'k_fixef': 'none', 'G_adj': True, 'G_df': 'conventional'}
        ssc_ref, df_k_ref, df_t_ref = pyfixest_get_ssc(
            ssc_dict=ssc_dict, N=N, k=k, k_fe=k_fe, k_fe_nested=0,
            n_fe=n_fe, n_fe_fully_nested=0, G=1, vcov_sign=1, vcov_type="iid"
        )
        
        assert np.isclose(ssc, ssc_ref), f"SSC mismatch: {ssc} vs {ssc_ref}"
        assert df_k == df_k_ref, f"df_k mismatch: {df_k} vs {df_k_ref}"
        assert df_t == df_t_ref, f"df_t mismatch: {df_t} vs {df_t_ref}"
    
    def test_ssc_hetero_vs_pyfixest(self):
        """Compare SSC for hetero with pyfixest."""
        N, k, k_fe, n_fe = 500, 3, 0, 0
        
        # Our implementation
        ssc_config = SSCConfig(k_adj=True, k_fixef='full', G_adj=True, G_df='conventional')
        context = VcovContext(N=N, k=k, k_fe=k_fe, n_fe=n_fe)
        ssc, df_k, df_t = compute_ssc(ssc_config, context, G=N, vcov_type="hetero", vcov_sign=1)
        
        # pyfixest reference
        ssc_dict = {'k_adj': True, 'k_fixef': 'full', 'G_adj': True, 'G_df': 'conventional'}
        ssc_ref, df_k_ref, df_t_ref = pyfixest_get_ssc(
            ssc_dict=ssc_dict, N=N, k=k, k_fe=k_fe, k_fe_nested=0,
            n_fe=n_fe, n_fe_fully_nested=0, G=N, vcov_sign=1, vcov_type="hetero"
        )
        
        assert np.isclose(ssc, ssc_ref), f"SSC mismatch: {ssc} vs {ssc_ref}"
        assert df_k == df_k_ref, f"df_k mismatch: {df_k} vs {df_k_ref}"
        assert df_t == df_t_ref, f"df_t mismatch: {df_t} vs {df_t_ref}"
    
    def test_ssc_cluster_vs_pyfixest(self):
        """Compare SSC for cluster with pyfixest."""
        N, k, k_fe, n_fe, G = 500, 3, 0, 0, 20
        
        # Our implementation
        ssc_config = SSCConfig(k_adj=True, k_fixef='full', G_adj=True, G_df='conventional')
        context = VcovContext(N=N, k=k, k_fe=k_fe, n_fe=n_fe)
        ssc, df_k, df_t = compute_ssc(ssc_config, context, G=G, vcov_type="CRV", vcov_sign=1)
        
        # pyfixest reference
        ssc_dict = {'k_adj': True, 'k_fixef': 'full', 'G_adj': True, 'G_df': 'conventional'}
        ssc_ref, df_k_ref, df_t_ref = pyfixest_get_ssc(
            ssc_dict=ssc_dict, N=N, k=k, k_fe=k_fe, k_fe_nested=0,
            n_fe=n_fe, n_fe_fully_nested=0, G=G, vcov_sign=1, vcov_type="CRV"
        )
        
        assert np.isclose(ssc, ssc_ref), f"SSC mismatch: {ssc} vs {ssc_ref}"
        assert df_k == df_k_ref, f"df_k mismatch: {df_k} vs {df_k_ref}"
        assert df_t == df_t_ref, f"df_t mismatch: {df_t} vs {df_t_ref}"
    
    def test_ssc_all_k_fixef_options(self):
        """Test all k_fixef options: none, nonnested, full."""
        N, k, k_fe, n_fe = 500, 3, 50, 2
        
        for k_fixef in ['none', 'nonnested', 'full']:
            ssc_config = SSCConfig(k_adj=True, k_fixef=k_fixef, G_adj=True, G_df='conventional')
            context = VcovContext(N=N, k=k, k_fe=k_fe, n_fe=n_fe)
            ssc, df_k, df_t = compute_ssc(ssc_config, context, G=1, vcov_type="iid", vcov_sign=1)
            
            # Compare with pyfixest
            ssc_dict = {'k_adj': True, 'k_fixef': k_fixef, 'G_adj': True, 'G_df': 'conventional'}
            ssc_ref, df_k_ref, df_t_ref = pyfixest_get_ssc(
                ssc_dict=ssc_dict, N=N, k=k, k_fe=k_fe, k_fe_nested=0,
                n_fe=n_fe, n_fe_fully_nested=0, G=1, vcov_sign=1, vcov_type="iid"
            )
            
            assert np.isclose(ssc, ssc_ref), f"k_fixef={k_fixef}: SSC {ssc} != {ssc_ref}"
            assert df_k == df_k_ref, f"k_fixef={k_fixef}: df_k {df_k} != {df_k_ref}"


# ============================================================================
# TESTS: Bread Matrix Computation
# ============================================================================

class TestBreadComputation:
    """Tests for bread matrix (Hessian inverse) computation."""
    
    def test_bread_ols(self, simple_data):
        """Compute bread matrix for OLS."""
        X = simple_data['X']
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        XtX = X_with_intercept.T @ X_with_intercept
        hessian = XtX
        bread = compute_bread(hessian, is_iv=False)
        
        # Verify bread is symmetric
        np.testing.assert_allclose(bread, bread.T, rtol=1e-10)
        
        # Verify bread is (XtX)^-1
        np.testing.assert_allclose(bread @ XtX, np.eye(4), rtol=1e-9, atol=1e-12)


# ============================================================================
# TESTS: IID Variance
# ============================================================================

class TestIIDVariance:
    """Tests for IID variance computation."""
    
    def test_compute_iid_vcov(self, simple_data):
        """Test compute_iid_vcov with new signature."""
        X = simple_data['X']
        y = simple_data['y']
        weights = simple_data['weights']
        n = simple_data['n']
        k = simple_data['k']
        
        X_with_intercept = np.column_stack([np.ones(n), X])
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y
        theta = safe_solve(XtX, Xty)
        residuals = y - X_with_intercept @ theta
        rss = float(residuals @ residuals)
        bread = safe_inv(XtX, use_pinv=True)
        
        # Compute vcov using new signature
        context = VcovContext(N=n, k=k+1, k_fe=0, n_fe=0)
        ssc_config = SSCConfig(k_adj=True, k_fixef='none', G_adj=True, G_df='conventional')
        vcov, vcov_meta = compute_iid_vcov(
            bread=bread, rss=rss, context=context, ssc_config=ssc_config, is_iv=False
        )
        
        assert vcov.shape == (k+1, k+1)
        np.testing.assert_allclose(vcov, vcov.T, rtol=1e-10)
        eigenvalues = np.linalg.eigvalsh(vcov)
        assert np.all(eigenvalues > 0), "VCOV should be positive definite"
    
    def test_vcov_iid_legacy(self, simple_data):
        """Test vcov_iid legacy function."""
        X = simple_data['X']
        y = simple_data['y']
        n = simple_data['n']
        
        X_with_intercept = np.column_stack([np.ones(n), X])
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y
        theta = safe_solve(XtX, Xty)
        residuals = y - X_with_intercept @ theta
        bread = safe_inv(XtX, use_pinv=True)
        
        vcov = vcov_iid(bread, residuals, n)
        
        assert vcov.shape == (4, 4)
        np.testing.assert_allclose(vcov, vcov.T, rtol=1e-10)


# ============================================================================
# TESTS: Heteroskedastic-Robust Variance
# ============================================================================

class TestHeteroVariance:
    """Tests for heteroskedastic-robust variance."""
    
    def test_compute_hetero_vcov_hc1(self, heteroskedastic_data):
        """Test compute_hetero_vcov with HC1."""
        X = heteroskedastic_data['X']
        y = heteroskedastic_data['y']
        n = heteroskedastic_data['n']
        k = heteroskedastic_data['k']
        
        X_with_intercept = np.column_stack([np.ones(n), X])
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y
        theta = safe_solve(XtX, Xty)
        residuals = y - X_with_intercept @ theta
        scores = X_with_intercept * residuals.reshape(-1, 1)
        bread = safe_inv(XtX, use_pinv=True)
        
        ssc_dict = {'k_adj': True, 'k_fixef': 'full', 'G_adj': True, 'G_df': 'conventional'}
        vcov, vcov_meta = compute_hetero_vcov(
            bread=bread, scores=scores, vcov_type_detail="HC1",
            ssc_dict=ssc_dict, N=n, k=k+1, k_fe=0, n_fe=0, is_iv=False
        )
        
        assert vcov.shape == (k+1, k+1)
        np.testing.assert_allclose(vcov, vcov.T, rtol=1e-10)
        assert vcov_meta['vcov_type'] == 'hetero'
        assert vcov_meta['vcov_type_detail'] == 'HC1'
    
    def test_compute_hetero_vcov_from_meat(self, heteroskedastic_data):
        """Test compute_hetero_vcov with pre-computed meat."""
        X = heteroskedastic_data['X']
        y = heteroskedastic_data['y']
        n = heteroskedastic_data['n']
        k = heteroskedastic_data['k']
        
        X_with_intercept = np.column_stack([np.ones(n), X])
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y
        theta = safe_solve(XtX, Xty)
        residuals = y - X_with_intercept @ theta
        scores = X_with_intercept * residuals.reshape(-1, 1)
        meat = scores.T @ scores
        bread = safe_inv(XtX, use_pinv=True)
        
        ssc_dict = {'k_adj': True, 'k_fixef': 'full', 'G_adj': True, 'G_df': 'conventional'}
        vcov, vcov_meta = compute_hetero_vcov(
            bread=bread, meat=meat, vcov_type_detail="HC1",
            ssc_dict=ssc_dict, N=n, k=k+1, k_fe=0, n_fe=0, is_iv=False
        )
        
        assert vcov.shape == (k+1, k+1)
        np.testing.assert_allclose(vcov, vcov.T, rtol=1e-10)
    
    def test_hetero_all_types(self, heteroskedastic_data):
        """Test all hetero types: HC1, hetero."""
        X = heteroskedastic_data['X']
        y = heteroskedastic_data['y']
        n = heteroskedastic_data['n']
        k = heteroskedastic_data['k']
        
        X_with_intercept = np.column_stack([np.ones(n), X])
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y
        theta = safe_solve(XtX, Xty)
        residuals = y - X_with_intercept @ theta
        scores = X_with_intercept * residuals.reshape(-1, 1)
        bread = safe_inv(XtX, use_pinv=True)
        
        ssc_dict = {'k_adj': True, 'k_fixef': 'full', 'G_adj': True, 'G_df': 'conventional'}
        
        for vcov_type in ["HC1", "hetero"]:
            vcov, meta = compute_hetero_vcov(
                bread=bread, scores=scores, vcov_type_detail=vcov_type,
                ssc_dict=ssc_dict, N=n, k=k+1
            )
            assert vcov.shape == (k+1, k+1)
            assert meta['vcov_type_detail'] == vcov_type


# ============================================================================
# TESTS: Cluster-Robust Variance
# ============================================================================

class TestClusterVariance:
    """Tests for cluster-robust variance."""
    
    def test_compute_cluster_scores(self, clustered_data):
        """Test cluster score aggregation."""
        X = clustered_data['X']
        y = clustered_data['y']
        cluster_ids = clustered_data['cluster_ids']
        n = clustered_data['n']
        
        X_with_intercept = np.column_stack([np.ones(n), X])
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y
        theta = safe_solve(XtX, Xty)
        residuals = y - X_with_intercept @ theta
        scores = X_with_intercept * residuals.reshape(-1, 1)
        
        cluster_scores, G = compute_cluster_scores(scores, cluster_ids)
        
        assert cluster_scores.shape == (G, X_with_intercept.shape[1])
        assert G == clustered_data['n_clusters']
        
        # Verify aggregation
        for g in range(G):
            mask = cluster_ids == g
            expected = scores[mask].sum(axis=0)
            np.testing.assert_allclose(cluster_scores[g], expected, rtol=1e-10)
    
    def test_compute_cluster_vcov(self, clustered_data):
        """Test compute_cluster_vcov with new signature."""
        X = clustered_data['X']
        y = clustered_data['y']
        cluster_ids = clustered_data['cluster_ids']
        n = clustered_data['n']
        k = clustered_data['k']
        G = clustered_data['n_clusters']
        
        X_with_intercept = np.column_stack([np.ones(n), X])
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y
        theta = safe_solve(XtX, Xty)
        residuals = y - X_with_intercept @ theta
        scores = X_with_intercept * residuals.reshape(-1, 1)
        cluster_scores, G_computed = compute_cluster_scores(scores, cluster_ids)
        bread = safe_inv(XtX, use_pinv=True)
        
        context = VcovContext(N=n, k=k+1, k_fe=0, n_fe=0)
        ssc_config = SSCConfig(k_adj=True, k_fixef='full', G_adj=True, G_df='conventional')
        vcov, vcov_meta = compute_cluster_vcov(
            bread=bread, cluster_scores=cluster_scores,
            context=context, G=G_computed, ssc_config=ssc_config, is_iv=False
        )
        
        assert vcov.shape == (k+1, k+1)
        np.testing.assert_allclose(vcov, vcov.T, rtol=1e-10)
        assert vcov_meta['vcov_type'] == 'cluster'
        assert vcov_meta['n_clusters'] == G
    
    def test_vcov_crv1_legacy(self, clustered_data):
        """Test vcov_crv1 legacy function."""
        X = clustered_data['X']
        y = clustered_data['y']
        cluster_ids = clustered_data['cluster_ids']
        n = clustered_data['n']
        k = clustered_data['k']
        
        X_with_intercept = np.column_stack([np.ones(n), X])
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y
        theta = safe_solve(XtX, Xty)
        residuals = y - X_with_intercept @ theta
        scores = X_with_intercept * residuals.reshape(-1, 1)
        bread = safe_inv(XtX, use_pinv=True)
        
        vcov = vcov_crv1(bread, scores, cluster_ids)
        
        assert vcov.shape == (k+1, k+1)
        np.testing.assert_allclose(vcov, vcov.T, rtol=1e-10)


# ============================================================================
# TESTS: Comprehensive All VCOV Types
# ============================================================================

class TestAllVcovTypes:
    """Comprehensive tests for all VCOV types."""
    
    def test_iid_all_ssc_variants(self, simple_data):
        """Test IID with all SSC variants."""
        X = simple_data['X']
        y = simple_data['y']
        n = simple_data['n']
        k = simple_data['k']
        
        X_with_intercept = np.column_stack([np.ones(n), X])
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y
        theta = safe_solve(XtX, Xty)
        residuals = y - X_with_intercept @ theta
        rss = float(residuals @ residuals)
        bread = safe_inv(XtX, use_pinv=True)
        
        for k_adj in [True, False]:
            for k_fixef in ['none', 'full']:
                context = VcovContext(N=n, k=k+1)
                config = SSCConfig(k_adj=k_adj, k_fixef=k_fixef)
                vcov, meta = compute_iid_vcov(bread, rss, context, config)
                
                assert vcov.shape == (k+1, k+1)
                np.testing.assert_allclose(vcov, vcov.T, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
