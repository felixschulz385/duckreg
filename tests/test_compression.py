"""
Test suite for data compression in DuckFE and DuckRegression.

Tests compression correctness, sufficient statistics preservation,
and consistency between compressed and uncompressed estimation.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from duckreg import compressed_ols


@pytest.fixture
def simple_data():
    """Create simple test data for compression tests"""
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'y': np.random.randn(n) + 5,
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'fe1': np.repeat(range(100), 10),
        'fe2': np.tile(range(10), 100),
        'cluster': np.repeat(range(20), 50)
    })
    
    return df


@pytest.fixture
def panel_data():
    """Create panel data for Mundlak tests"""
    np.random.seed(123)
    n_firms = 100  # More than threshold (50) to ensure asymptotic classification
    n_years = 10
    
    panel = pd.DataFrame([
        {'firm_id': firm, 'year': year}
        for firm in range(n_firms)
        for year in range(2010, 2010 + n_years)
    ])
    
    panel['x'] = np.random.randn(len(panel))
    panel['y'] = 2 + 1.5 * panel['x'] + np.random.randn(len(panel)) * 0.5
    
    return panel


class TestDuckRegressionCompression:
    """Test compression in DuckRegression estimator"""
    
    def test_basic_compression_no_fe(self, simple_data):
        """Test basic compression without fixed effects"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        simple_data.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x1 + x2",
                data=temp_path,
                round_strata=3,
                se_method="none"
            )
            
            # Check compression view was created
            assert hasattr(model, 'df_compressed')
            # Note: Random continuous data may not compress much
            assert len(model.df_compressed) <= len(simple_data)
            
            # Check compressed data has required columns
            assert 'count' in model.df_compressed.columns
            assert 'sum_y' in model.df_compressed.columns
            assert 'sum_y_sq' in model.df_compressed.columns
            
            # Check counts sum to original N
            assert model.df_compressed['count'].sum() == len(simple_data)
            
        finally:
            import os
            os.unlink(temp_path)
    
    def test_compression_with_fe(self, simple_data):
        """Test compression preserves FE information via Mundlak means (asymptotic FE)"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        simple_data.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x1 | fe1",
                data=temp_path,
                round_strata=3,
                se_method="none",
                fe_method="mundlak"
            )
            
            # fe1 has 100 levels (> cardinality_threshold=50), so it is classified
            # as asymptotic: original fe1 column is NOT in the compressed view,
            # but its Mundlak mean is.
            assert 'avg_x1_fe0' in model.df_compressed.columns
            assert 'sum_y' in model.df_compressed.columns
            assert model.df_compressed['count'].sum() == len(simple_data)
            
        finally:
            import os
            os.unlink(temp_path)
    
    def test_compression_rounding_levels(self, simple_data):
        """Test different rounding levels affect compression ratio"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        simple_data.to_parquet(temp_path)
        
        try:
            # More rounding = more compression
            model_round3 = compressed_ols(
                formula="y ~ x1 + x2",
                data=temp_path,
                round_strata=3,
                se_method="none"
            )
            
            model_round5 = compressed_ols(
                formula="y ~ x1 + x2",
                data=temp_path,
                round_strata=5,
                se_method="none"
            )
            
            # Higher precision (round_strata=5) should have more strata
            assert len(model_round5.df_compressed) >= len(model_round3.df_compressed)
            
        finally:
            import os
            os.unlink(temp_path)
    
    def test_compression_preserves_sufficient_stats(self, simple_data):
        """Test that sufficient statistics are preserved after compression"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        simple_data.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x1 + x2",
                data=temp_path,
                round_strata=3,
                se_method="none"
            )
            
            # Sum of y should be preserved (approximately due to rounding)
            compressed_sum_y = (model.df_compressed['sum_y']).sum()
            original_sum_y = simple_data['y'].sum()
            
            # Allow for rounding error
            assert abs(compressed_sum_y - original_sum_y) < 1e-1
            
            # Total count should match exactly
            assert model.df_compressed['count'].sum() == len(simple_data)
            
        finally:
            import os
            os.unlink(temp_path)
    
    def test_compression_with_cluster(self, simple_data):
        """Test compression includes cluster variable"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        simple_data.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x1 + x2 | 0 | 0 | cluster",
                data=temp_path,
                round_strata=3,
                se_method="none"
            )
            
            # Cluster variable should be in compressed data
            assert 'cluster' in model.df_compressed.columns
            
        finally:
            import os
            os.unlink(temp_path)


class TestDuckFECompression:
    """Test compression in DuckFE estimator (mundlak method)"""
    
    def test_compression_includes_mundlak_means(self, panel_data):
        """Test compression includes Mundlak mean columns"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        panel_data.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x | firm_id",
                data=temp_path,
                fe_method="mundlak",
                round_strata=3,
                se_method="none"
            )
            
            # Should have Mundlak mean columns
            assert 'avg_x_fe0' in model.df_compressed.columns
            
            # Should have count and sums
            assert 'count' in model.df_compressed.columns
            assert 'sum_y' in model.df_compressed.columns
            
        finally:
            import os
            os.unlink(temp_path)
    
    def test_compression_with_mixed_fe_types(self):
        """Test compression with fixed and asymptotic FEs"""
        np.random.seed(456)
        n_firms = 100
        n_years = 5
        years = list(range(2015, 2015 + n_years))
        
        panel = pd.DataFrame([
            {'firm_id': firm, 'year': year}
            for firm in range(n_firms)
            for year in years
        ])
        
        panel['x'] = np.random.randn(len(panel))
        panel['y'] = 2 + 1.5 * panel['x'] + np.random.randn(len(panel)) * 0.5
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        panel.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x | firm_id + year",
                data=temp_path,
                fe_method="mundlak",
                round_strata=3,
                se_method="none"
            )
            
            # Should have Mundlak means (for asymptotic FE)
            assert 'avg_x_fe0' in model.df_compressed.columns
            
            # Should have fixed FE dummies
            fixed_dummies = [col for col in model.df_compressed.columns if col.startswith('dummy_year_')]
            assert len(fixed_dummies) == 4  # 5 years - 1 reference
            
            # Should have dummy-means
            dummy_means = [col for col in model.df_compressed.columns if 'avg_year_' in col and '_fe0' in col]
            assert len(dummy_means) == 4
            
        finally:
            import os
            os.unlink(temp_path)
    
    def test_compression_strata_structure(self, panel_data):
        """Test that compression creates proper strata structure"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        panel_data.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x | firm_id",
                data=temp_path,
                fe_method="mundlak",
                round_strata=3,
                se_method="none"
            )
            
            # Check that strata_cols attribute exists
            assert hasattr(model, 'strata_cols')
            
            # Should include covariates and Mundlak means
            assert 'x' in model.strata_cols or any('x' in col for col in model.strata_cols)
            assert any('avg_x' in col for col in model.strata_cols)
            
        finally:
            import os
            os.unlink(temp_path)
    
    def test_rhs_cols_tracking(self, panel_data):
        """Test that _rhs_cols correctly tracks all RHS variables"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        panel_data.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x | firm_id",
                data=temp_path,
                fe_method="mundlak",
                round_strata=3,
                se_method="none",
                fitter="numpy"  # Force numpy to trigger collect_data
            )
            
            # Should have _rhs_cols attribute
            assert hasattr(model, '_rhs_cols')
            
            # Should include covariate and Mundlak mean
            assert 'x' in model._rhs_cols
            assert 'avg_x_fe0' in model._rhs_cols
            
        finally:
            import os
            os.unlink(temp_path)


class TestCompressionConsistency:
    """Test that compression doesn't affect estimation results"""
    
    def test_estimates_match_with_different_rounding(self, simple_data):
        """Test that different rounding levels give similar estimates"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        simple_data.to_parquet(temp_path)
        
        try:
            model_round2 = compressed_ols(
                formula="y ~ x1 + x2",
                data=temp_path,
                round_strata=2,
                se_method="none"
            )
            
            model_round4 = compressed_ols(
                formula="y ~ x1 + x2",
                data=temp_path,
                round_strata=4,
                se_method="none"
            )
            
            # Coefficients should be similar (allowing for rounding effects)
            coef_diff = np.abs(model_round2.point_estimate - model_round4.point_estimate)
            assert np.all(coef_diff < 0.01), "Estimates differ too much with different rounding"
            
        finally:
            import os
            os.unlink(temp_path)
    
    def test_mundlak_compression_preserves_estimates(self, panel_data):
        """Test Mundlak estimates are stable across compression levels"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        panel_data.to_parquet(temp_path)
        
        try:
            model_round3 = compressed_ols(
                formula="y ~ x | firm_id",
                data=temp_path,
                fe_method="mundlak",
                round_strata=3,
                se_method="none"
            )
            
            model_round5 = compressed_ols(
                formula="y ~ x | firm_id",
                data=temp_path,
                fe_method="mundlak",
                round_strata=5,
                se_method="none"
            )
            
            # Main coefficient should be very similar
            coef_diff = np.abs(model_round3.point_estimate[1] - model_round5.point_estimate[1])
            assert coef_diff < 0.05, "Mundlak estimates too sensitive to rounding"
            
        finally:
            import os
            os.unlink(temp_path)
    
    def test_both_fitters_give_same_result(self, panel_data):
        """Test numpy and duckdb fitters give same results after compression"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        panel_data.to_parquet(temp_path)
        
        try:
            model_numpy = compressed_ols(
                formula="y ~ x | firm_id",
                data=temp_path,
                fe_method="mundlak",
                round_strata=4,
                fitter="numpy",
                se_method="none"
            )
            
            model_duckdb = compressed_ols(
                formula="y ~ x | firm_id",
                data=temp_path,
                fe_method="mundlak",
                round_strata=4,
                fitter="duckdb",
                se_method="none"
            )
            
            # Should give identical results
            np.testing.assert_allclose(
                model_numpy.point_estimate,
                model_duckdb.point_estimate,
                rtol=1e-10,
                err_msg="Numpy and DuckDB fitters disagree after compression"
            )
            
        finally:
            import os
            os.unlink(temp_path)


class TestCompressionEdgeCases:
    """Test edge cases in compression"""
    
    def test_compression_with_no_variation(self):
        """Test compression when data has no variation in covariates"""
        df = pd.DataFrame({
            'y': np.random.randn(100),
            'x': np.ones(100),  # No variation
            'fe': np.repeat(range(10), 10)
        })
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        df.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x | fe",
                data=temp_path,
                round_strata=3,
                se_method="none",
                fe_method="mundlak"
            )
            
            # Should still work, just compress heavily
            assert len(model.df_compressed) < len(df)
            
        finally:
            import os
            os.unlink(temp_path)
    
    def test_compression_with_single_stratum(self):
        """Test when compression results in single stratum"""
        df = pd.DataFrame({
            'y': np.random.randn(100) + 5,
            'x': np.ones(100),
        })
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        df.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x",
                data=temp_path,
                round_strata=1,  # Very aggressive rounding
                se_method="none"
            )
            
            # Might compress to very few strata
            assert len(model.df_compressed) >= 1
            assert 'count' in model.df_compressed.columns
            
        finally:
            import os
            os.unlink(temp_path)
    
    def test_compression_preserves_all_fe_levels(self):
        """Test that all FE levels are represented after compression"""
        np.random.seed(789)
        n_obs = 500
        n_fe_levels = 50
        
        df = pd.DataFrame({
            'y': np.random.randn(n_obs),
            'x': np.random.randn(n_obs),
            'fe': np.random.choice(range(n_fe_levels), n_obs)
        })
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        df.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x | fe",
                data=temp_path,
                round_strata=3,
                se_method="none",
                fe_method="mundlak",
                remove_singletons=False  # Preserve all FE levels including singletons
            )
            
            # fe has 50 levels (≤ cardinality_threshold=50) → classified as FIXED.
            # The FE is encoded via binary dummy columns (49 non-reference dummies);
            # the original 'fe' column does not appear in the compressed view.
            dummy_cols = [c for c in model.df_compressed.columns if c.startswith('dummy_fe_')]
            # 49 non-reference levels should have a dummy column
            assert len(dummy_cols) == n_fe_levels - 1, (
                f"Expected {n_fe_levels - 1} dummy columns, got {len(dummy_cols)}"
            )
            assert model.df_compressed['count'].sum() == n_obs
            
        finally:
            import os
            os.unlink(temp_path)


class TestCompressionQuery:
    """Test the compression SQL query generation"""
    
    def test_agg_query_attribute_exists(self, simple_data):
        """Test that agg_query attribute is set"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        simple_data.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x1 + x2",
                data=temp_path,
                round_strata=3,
                se_method="none"
            )
            
            # Should have agg_query
            assert hasattr(model, 'agg_query')
            assert isinstance(model.agg_query, str)
            assert 'SELECT' in model.agg_query.upper()
            assert 'GROUP BY' in model.agg_query.upper()
            
        finally:
            import os
            os.unlink(temp_path)
    
    def test_compression_query_includes_sum_sq(self, simple_data):
        """Test that compression query includes sum_y_sq for variance"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        simple_data.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x1 + x2",
                data=temp_path,
                round_strata=3,
                se_method="none"
            )
            
            # Query should include sum_y_sq
            assert 'sum_y_sq' in model.agg_query or 'sum_y * sum_y' in model.agg_query.lower()
            
        finally:
            import os
            os.unlink(temp_path)
