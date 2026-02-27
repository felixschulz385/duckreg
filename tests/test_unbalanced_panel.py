"""
Tests for unbalanced panel support with FE classification.

This test suite focuses on the NEW functionality added for unbalanced panels:
1. FE classification (fixed vs asymptotic) based on data characteristics
2. Dummy-mean generation for unbalanced panel correction (Wooldridge)
3. Edge cases (all-fixed, all-asymptotic, column explosion guards)

Note: Coefficient accuracy against pyfixest is tested in test_compressed_ols.py.
This suite focuses on verifying the classification and feature generation mechanisms.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from pathlib import Path

from duckreg import compressed_ols


@pytest.fixture
def balanced_panel_data():
    """Create balanced panel data (all firms observed in all years)"""
    np.random.seed(42)
    
    n_firms = 100  # Increased to ensure asymptotic classification
    n_years = 5
    years = list(range(2015, 2015 + n_years))
    
    # Create balanced panel structure
    panel = pd.DataFrame([
        {'firm_id': firm, 'year': year}
        for firm in range(n_firms)
        for year in years
    ])
    
    # Add covariates and outcome
    panel['x1'] = np.random.randn(len(panel))
    panel['x2'] = np.random.randn(len(panel))
    
    # Add FE (firm and year effects)
    firm_fe = np.random.randn(n_firms)
    year_fe = np.random.randn(n_years)
    
    panel['y'] = (
        2.0 + 
        1.5 * panel['x1'] + 
        0.8 * panel['x2'] +
        firm_fe[panel['firm_id']] +
        year_fe[panel['year'] - 2015] +
        np.random.randn(len(panel)) * 0.5
    )
    
    return panel


@pytest.fixture
def unbalanced_panel_data():
    """Create unbalanced panel data (random missingness)"""
    np.random.seed(42)
    
    n_firms = 100  # Increased to ensure asymptotic classification
    n_years = 5
    years = list(range(2015, 2015 + n_years))
    
    # Create full panel structure
    panel = pd.DataFrame([
        {'firm_id': firm, 'year': year}
        for firm in range(n_firms)
        for year in years
    ])
    
    # Randomly drop 30% of observations to create unbalancedness
    drop_mask = np.random.rand(len(panel)) < 0.3
    panel = panel[~drop_mask].reset_index(drop=True)
    
    # Add covariates and outcome
    panel['x1'] = np.random.randn(len(panel))
    panel['x2'] = np.random.randn(len(panel))
    
    # Add FE (firm and year effects)
    firm_fe = np.random.randn(n_firms)
    year_fe = np.random.randn(n_years)
    
    panel['y'] = (
        2.0 + 
        1.5 * panel['x1'] + 
        0.8 * panel['x2'] +
        firm_fe[panel['firm_id']] +
        year_fe[panel['year'] - 2015] +
        np.random.randn(len(panel)) * 0.5
    )
    
    return panel


@pytest.fixture
def temp_parquet_file(request):
    """Create temporary parquet file from data fixture"""
    data = request.param
    
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
        temp_path = f.name
    
    data.to_parquet(temp_path)
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestFEClassificationAndFeatureGeneration:
    """Test FE classification and feature generation for mixed panel types"""
    
    @pytest.mark.parametrize("fitter", ["numpy", "duckdb"])
    def test_balanced_panel_classification(self, balanced_panel_data, fitter):
        """Test that balanced panels classify FEs correctly and add appropriate features"""
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        balanced_panel_data.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x1 + x2 | firm_id + year",
                data=temp_path,
                fe_method="mundlak",
                fitter=fitter,
                se_method="none",
                round_strata=5
            )
            
            # Check FE classification
            assert 'firm_id' in model.fe_metadata
            assert 'year' in model.fe_metadata
            
            # Year should be classified as fixed (small cardinality)
            assert model.fe_metadata['year']['type'] == 'fixed'
            assert model.fe_metadata['year']['profile']['cardinality'] == 5
            
            # Firm should be classified as asymptotic (large cardinality = 100 > threshold 50)
            assert model.fe_metadata['firm_id']['type'] == 'asymptotic'
            assert model.fe_metadata['firm_id']['profile']['cardinality'] == 100
            
            # Check that Mundlak means were added (for asymptotic FE)
            mundlak_means = [name for name in model.coef_names_ if 'avg_x' in name and '_fe0' in name]
            assert len(mundlak_means) > 0, "Expected Mundlak means for asymptotic FE"
            
            # Fixed FE dummies should be added (year dummies, excluding reference)
            fixed_dummies = [name for name in model.coef_names_ if name.startswith('dummy_year_')]
            assert len(fixed_dummies) == 4, "Expected 4 year dummies (5 years - 1 reference)"
            
            # Dummy-means should be added (year within firm)
            dummy_means = [name for name in model.coef_names_ if 'avg_year_' in name]
            assert len(dummy_means) > 0, "Expected dummy-means for unbalanced correction"
            assert len(dummy_means) == 4, "Expected 4 dummy-means (5 years - 1 reference)"
            
            # Verify estimation succeeds
            assert model.point_estimate is not None
            assert len(model.coef_names_) > 0
            
            # Check coefficient count: 
            # Intercept + 2 covariates + 2 Mundlak means + 4 fixed FE dummies + 4 dummy-means = 13
            expected_n_coefs = 1 + 2 + 2 + 4 + 4
            assert len(model.coef_names_) == expected_n_coefs
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @pytest.mark.parametrize("fitter", ["numpy", "duckdb"])
    def test_unbalanced_panel_adds_dummy_means(self, unbalanced_panel_data, fitter):
        """Test that unbalanced panels add dummy-mean columns"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        unbalanced_panel_data.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x1 + x2 | firm_id + year",
                data=temp_path,
                fe_method="mundlak",
                fitter=fitter,
                se_method="none",
                round_strata=5
            )
            
            # Check FE classification
            assert model.fe_metadata['year']['type'] == 'fixed'
            assert model.fe_metadata['firm_id']['type'] == 'asymptotic'
            
            # Check that dummy-mean columns were added
            # Format: avg_year_{level}_fe{asymp_idx}
            dummy_mean_cols = [name for name in model.coef_names_ if 'avg_year_' in name]
            assert len(dummy_mean_cols) > 0, "Expected dummy-mean columns for unbalanced panel"
            
            # Should have one column per non-reference year level
            n_years = len(model.fe_metadata['year']['levels'])
            expected_dummy_cols = n_years - 1  # Exclude reference level
            assert len(dummy_mean_cols) == expected_dummy_cols
            
            # Verify estimation succeeds
            assert model.point_estimate is not None
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

class TestFEClassificationHeuristic:
    """Test FE classification heuristic in isolation"""
    
    def test_small_cardinality_classified_as_fixed(self):
        """Test that FEs with small cardinality are classified as fixed"""
        np.random.seed(42)
        
        # Create data with small cardinality FE
        data = pd.DataFrame({
            'entity_id': np.repeat(range(100), 5),
            'time_id': np.tile(range(5), 100),  # Only 5 time periods
            'x': np.random.randn(500),
            'y': np.random.randn(500)
        })
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        data.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x | entity_id + time_id",
                data=temp_path,
                fe_method="mundlak",
                fitter="numpy",
                se_method="none"
            )
            
            # time_id should be classified as fixed (cardinality = 5)
            assert model.fe_metadata['time_id']['type'] == 'fixed'
            assert model.fe_metadata['time_id']['profile']['cardinality'] == 5
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_large_cardinality_classified_as_asymptotic(self):
        """Test that FEs with large cardinality are classified as asymptotic"""
        np.random.seed(42)
        
        # Create data with large cardinality FE
        data = pd.DataFrame({
            'entity_id': np.repeat(range(200), 5),  # 200 entities
            'time_id': np.tile(range(5), 200),
            'x': np.random.randn(1000),
            'y': np.random.randn(1000)
        })
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        data.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x | entity_id + time_id",
                data=temp_path,
                fe_method="mundlak",
                fitter="numpy",
                se_method="none"
            )
            
            # entity_id should be classified as asymptotic (cardinality = 200)
            assert model.fe_metadata['entity_id']['type'] == 'asymptotic'
            assert model.fe_metadata['entity_id']['profile']['cardinality'] == 200
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_user_override_classification(self):
        """Test that user can override FE classification"""
        np.random.seed(42)
        
        data = pd.DataFrame({
            'entity_id': np.repeat(range(100), 5),
            'time_id': np.tile(range(5), 100),
            'x': np.random.randn(500),
            'y': np.random.randn(500)
        })
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        data.to_parquet(temp_path)
        
        try:
            # Force time_id to be treated as asymptotic (even though it's small)
            model = compressed_ols(
                formula="y ~ x | entity_id + time_id",
                data=temp_path,
                fe_method="mundlak",
                fitter="numpy",
                se_method="none",
                fe_types={'time_id': 'asymptotic'}
            )
            
            # Should use user override
            assert model.fe_metadata['time_id']['type'] == 'asymptotic'
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestEdgeCases:
    """Test edge cases and guards"""
    
    def test_all_fes_asymptotic(self):
        """Test case where all FEs are asymptotic (no dummy-means needed)"""
        np.random.seed(42)
        
        data = pd.DataFrame({
            'entity1': np.repeat(range(200), 5),
            'entity2': np.tile(range(200), 5),
            'x': np.random.randn(1000),
            'y': np.random.randn(1000)
        })
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        data.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x | entity1 + entity2",
                data=temp_path,
                fe_method="mundlak",
                fitter="numpy",
                se_method="none"
            )
            
            # Both should be asymptotic
            assert model.fe_metadata['entity1']['type'] == 'asymptotic'
            assert model.fe_metadata['entity2']['type'] == 'asymptotic'
            
            # No dummy-mean columns should be added
            dummy_mean_cols = [name for name in model.coef_names_ 
                              if name.startswith('avg_entity')]
            # Should only have Mundlak means, not dummy-means
            assert all('_fe' in name and '=' not in name for name in dummy_mean_cols)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_all_fes_fixed(self):
        """Test case where all FEs are fixed (no Mundlak means, only dummy-means)"""
        np.random.seed(42)
        
        data = pd.DataFrame({
            'region': np.random.choice(['A', 'B', 'C'], size=500),
            'year': np.random.choice([2018, 2019, 2020], size=500),
            'x': np.random.randn(500),
            'y': np.random.randn(500)
        })
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        data.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x | region + year",
                data=temp_path,
                fe_method="mundlak",
                fitter="numpy",
                se_method="none"
            )
            
            # Both should be fixed
            assert model.fe_metadata['region']['type'] == 'fixed'
            assert model.fe_metadata['year']['type'] == 'fixed'
            
            # No Mundlak means should be added (no asymptotic FEs)
            mundlak_means = [name for name in model.coef_names_ if 'avg_x_fe' in name]
            assert len(mundlak_means) == 0
            
            # No dummy-means either (need at least one asymptotic FE)
            assert len(model._dummy_mean_cols) == 0
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_column_explosion_guard(self):
        """Test that column explosion guard prevents too many dummy-mean columns"""
        np.random.seed(42)
        
        # Create data with many levels in "fixed" FE
        data = pd.DataFrame({
            'entity_id': np.repeat(range(100), 5),
            'time_id': np.tile(range(150), 100)[:500],  # Many time periods
            'x': np.random.randn(500),
            'y': np.random.randn(500)
        })
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        data.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x | entity_id + time_id",
                data=temp_path,
                fe_method="mundlak",
                fitter="numpy",
                se_method="none",
                max_fixed_fe_levels=100  # Set guard threshold
            )
            
            # time_id should be reclassified as asymptotic due to too many levels
            assert model.fe_metadata['time_id']['type'] == 'asymptotic'
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
