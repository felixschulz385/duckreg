"""
Tests for data preparation and singleton FE removal.

Tests that singleton fixed effect groups (groups with only one observation)
are correctly identified and removed when remove_singletons=True.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

from duckreg import compressed_ols
from duckreg.estimators import DuckMundlak, Duck2SLS
from duckreg.estimators.DuckDoubleDemeaning import DuckDoubleDemeaning
from duckreg.utils.formula_parser import FormulaParser


@pytest.fixture
def data_with_singletons():
    """Create test data with singleton FE groups.
    
    Creates data with:
    - firm_id: groups 1,2,3,4,5 where 5 is a singleton
    - year: groups 2000,2001,2002 where some year-firm combos are singletons
    - Some firms appear in multiple years, some in only one
    """
    np.random.seed(42)
    
    data = []
    # Firm 1: appears in all years (not singleton)
    for year in [2000, 2001, 2002]:
        data.append({'firm_id': 1, 'year': year, 'x': np.random.randn(), 'z': np.random.randn()})
    
    # Firm 2: appears in 2 years (not singleton)
    for year in [2000, 2001]:
        data.append({'firm_id': 2, 'year': year, 'x': np.random.randn(), 'z': np.random.randn()})
    
    # Firm 3: appears in 2 years (not singleton)
    for year in [2001, 2002]:
        data.append({'firm_id': 3, 'year': year, 'x': np.random.randn(), 'z': np.random.randn()})
    
    # Firm 4: appears in 2 observations in same year (not singleton by firm)
    for _ in range(2):
        data.append({'firm_id': 4, 'year': 2000, 'x': np.random.randn(), 'z': np.random.randn()})
    
    # Firm 5: SINGLETON - only appears once
    data.append({'firm_id': 5, 'year': 2001, 'x': np.random.randn(), 'z': np.random.randn()})
    
    # Firm 6: appears in 2 years (not singleton)
    for year in [2000, 2002]:
        data.append({'firm_id': 6, 'year': year, 'x': np.random.randn(), 'z': np.random.randn()})
    
    df = pd.DataFrame(data)
    
    # Add outcome variable
    df['y'] = 2.0 + 1.5 * df['x'] + df['firm_id'] * 0.1 + df['year'] * 0.001 + np.random.randn(len(df)) * 0.5
    
    # Add endogenous variable for IV tests
    df['endog'] = df['x'] + 0.8 * df['z'] + np.random.randn(len(df)) * 0.3
    
    return df


@pytest.fixture
def temp_data_file(data_with_singletons):
    """Create temporary parquet file with test data."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
        temp_path = f.name
    
    data_with_singletons.to_parquet(temp_path)
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestSingletonRemovalMundlak:
    """Test singleton removal for DuckMundlak estimator."""
    
    def test_removes_singletons_by_default(self, temp_data_file, data_with_singletons):
        """Test that singletons are removed by default (remove_singletons=True)."""
        model = compressed_ols(
            formula="y ~ x | firm_id",
            data=temp_data_file,
            fe_method="mundlak",
            fitter="numpy",
            se_method="none"
        )
        
        # Firm 5 is a singleton, so n_obs should be 1 less than original
        expected_obs = len(data_with_singletons) - 1
        assert model.n_obs == expected_obs, f"Expected {expected_obs} obs after removing singleton, got {model.n_obs}"
    
    def test_keeps_singletons_when_disabled(self, temp_data_file, data_with_singletons):
        """Test that singletons are kept when remove_singletons=False."""
        model = compressed_ols(
            formula="y ~ x | firm_id",
            data=temp_data_file,
            fe_method="mundlak",
            fitter="numpy",
            se_method="none",
            remove_singletons=False
        )
        
        # Should keep all observations including singleton
        assert model.n_obs == len(data_with_singletons), \
            f"Expected {len(data_with_singletons)} obs when keeping singletons, got {model.n_obs}"
    
    def test_multiple_fe_singleton_removal(self, temp_data_file):
        """Test singleton removal with multiple FE (firm and year)."""
        # With firm_id + year FE, the QUALIFY will filter based on both
        model = compressed_ols(
            formula="y ~ x | firm_id + year",
            data=temp_data_file,
            fe_method="mundlak",
            fitter="numpy",
            se_method="none",
            remove_singletons=True
        )
        
        # Should remove observations that are singletons in the firm_id + year partition
        # This is more restrictive than just firm_id
        assert model.n_obs > 0, "Should have some observations remaining"
        assert hasattr(model, 'point_estimate'), "Should successfully estimate"
    
    def test_estimation_succeeds_after_singleton_removal(self, temp_data_file):
        """Test that estimation completes successfully after removing singletons."""
        model = compressed_ols(
            formula="y ~ x | firm_id",
            data=temp_data_file,
            fe_method="mundlak",
            fitter="numpy",
            se_method="HC1"
        )
        
        assert model.point_estimate is not None
        assert model.vcov is not None
        assert len(model.coef_names_) > 0
        
        # Check that coefficient estimates are reasonable
        # (intercept + x + mundlak mean of x)
        assert len(model.point_estimate) >= 2

class TestSingletonRemoval2SLS:
    """Test singleton removal for Duck2SLS estimator."""
    
    def test_removes_singletons_by_default(self, temp_data_file, data_with_singletons):
        """Test that singletons are removed by default in 2SLS."""
        model = compressed_ols(
            formula="y ~ x | firm_id | endog (z)",
            data=temp_data_file,
            fe_method="mundlak",
            fitter="numpy",
            se_method="none"
        )
        
        # Firm 5 is a singleton, so n_obs should be 1 less than original
        expected_obs = len(data_with_singletons) - 1
        assert model.n_obs == expected_obs, \
            f"Expected {expected_obs} obs after removing singleton, got {model.n_obs}"
    
    def test_keeps_singletons_when_disabled(self, temp_data_file, data_with_singletons):
        """Test that singletons are kept when remove_singletons=False."""
        model = compressed_ols(
            formula="y ~ x | firm_id | endog (z)",
            data=temp_data_file,
            fe_method="mundlak",
            fitter="numpy",
            se_method="none",
            remove_singletons=False
        )
        
        # Should keep all observations including singleton
        assert model.n_obs == len(data_with_singletons), \
            f"Expected {len(data_with_singletons)} obs when keeping singletons, got {model.n_obs}"
    
    def test_estimation_succeeds_after_singleton_removal(self, temp_data_file):
        """Test that 2SLS estimation completes successfully after removing singletons."""
        model = compressed_ols(
            formula="y ~ x | firm_id | endog (z)",
            data=temp_data_file,
            fe_method="mundlak",
            fitter="numpy",
            se_method="HC1"
        )
        
        assert model.point_estimate is not None
        assert model.vcov is not None
        
        # Check first stage results
        assert hasattr(model, '_first_stage_results')
        assert len(model._first_stage_results) > 0


class TestSingletonRemovalEdgeCases:
    """Test edge cases for singleton removal."""
    
    def test_no_singletons_in_data(self, temp_data_file):
        """Test that estimation works when there are no singletons."""
        # Using year as FE - no singletons since multiple obs per year
        model = compressed_ols(
            formula="y ~ x | year",
            data=temp_data_file,
            fe_method="mundlak",
            fitter="numpy",
            se_method="none"
        )
        
        assert model.n_obs > 0
        assert model.point_estimate is not None
    
    def test_all_observations_removed_scenario(self):
        """Test handling when all observations would be removed."""
        # Create data where every group is a singleton
        df = pd.DataFrame({
            'y': [1, 2, 3, 4],
            'x': [1, 2, 3, 4],
            'firm_id': [1, 2, 3, 4],  # Each firm appears only once
        })
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            temp_path = f.name
        df.to_parquet(temp_path)
        
        try:
            model = compressed_ols(
                formula="y ~ x | firm_id",
                data=temp_path,
                fe_method="mundlak",
                fitter="numpy",
                se_method="none",
                remove_singletons=True
            )
            
            # With all singletons removed, n_obs should be 0
            assert model.n_obs == 0, "All observations should be removed"
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_no_fe_columns(self, temp_data_file):
        """Test that remove_singletons has no effect when there are no FE."""
        model = compressed_ols(
            formula="y ~ x",
            data=temp_data_file,
            fitter="numpy",
            se_method="none",
            remove_singletons=True  # Should have no effect without FE
        )
        
        # Should include all observations since no FE to check for singletons
        assert model.n_obs > 0
        assert model.point_estimate is not None


class TestQualifyClauseGeneration:
    """Test the QUALIFY clause generation utility."""
    
    def test_qualify_single_fe(self, temp_data_file):
        """Test QUALIFY clause with single FE column."""
        formula = FormulaParser().parse("y ~ x | firm_id")
        model = DuckMundlak(
            db_name=":memory:",
            table_name=f"read_parquet('{temp_data_file}')",
            formula=formula,
            seed=42,
            remove_singletons=True
        )
        
        # Test QUALIFY clause generation
        qualify_clause = model._build_qualify_singleton_filter(['firm_id'])
        assert 'QUALIFY' in qualify_clause
        assert 'PARTITION BY firm_id' in qualify_clause
        assert '> 1' in qualify_clause
    
    def test_qualify_multiple_fe(self, temp_data_file):
        """Test QUALIFY clause with multiple FE columns."""
        formula = FormulaParser().parse("y ~ x | firm_id + year")
        model = DuckMundlak(
            db_name=":memory:",
            table_name=f"read_parquet('{temp_data_file}')",
            formula=formula,
            seed=42,
            remove_singletons=True
        )
        
        # Test QUALIFY clause generation with multiple FE
        qualify_clause = model._build_qualify_singleton_filter(['firm_id', 'year'])
        assert 'QUALIFY' in qualify_clause
        assert 'PARTITION BY' in qualify_clause
        assert 'firm_id' in qualify_clause
        assert 'year' in qualify_clause
        assert '> 1' in qualify_clause
    
    def test_qualify_disabled(self, temp_data_file):
        """Test that QUALIFY clause is empty when remove_singletons=False."""
        formula = FormulaParser().parse("y ~ x | firm_id")
        model = DuckMundlak(
            db_name=":memory:",
            table_name=f"read_parquet('{temp_data_file}')",
            formula=formula,
            seed=42,
            remove_singletons=False
        )
        
        # QUALIFY clause should be empty when disabled
        qualify_clause = model._build_qualify_singleton_filter(['firm_id'])
        assert qualify_clause == ""


class TestModelSummaryIntegration:
    """Test that n_rows_dropped_singletons flows through ModelSummary class."""
    
    def test_model_summary_tracks_dropped_rows_mundlak(self, data_with_singletons):
        """Test ModelSummary properly captures n_rows_dropped_singletons from Mundlak."""
        from duckreg.core.results import ModelSummary
        
        df = pd.DataFrame(data_with_singletons)
        y = np.random.randn(len(df))
        df['y'] = y
        
        # Write to temp parquet file
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_file = os.path.join(tmpdir, "test_data.parquet")
            df.to_parquet(temp_file)
            
            formula = FormulaParser().parse("y ~ x | firm_id")
            model = DuckMundlak(
                db_name=":memory:",
                table_name=f"read_parquet('{temp_file}')",
                formula=formula,
                seed=42,
                remove_singletons=True,
                fitter='numpy'
            )
            
            # Fit the model
            model.fit()
            
            # Create ModelSummary from estimator
            summary = ModelSummary.from_estimator(model)
            
            # Verify n_rows_dropped_singletons is captured
            assert summary.n_rows_dropped_singletons == 1, \
                f"Expected 1 singleton dropped, got {summary.n_rows_dropped_singletons}"
            assert summary.n_obs == len(df) - 1, \
                f"Expected n_obs={len(df)-1}, got {summary.n_obs}"
    
    def test_model_summary_to_dict_includes_dropped_rows(self, data_with_singletons):
        """Test that to_dict() includes n_rows_dropped_singletons in sample_info."""
        from duckreg.core.results import ModelSummary
        
        df = pd.DataFrame(data_with_singletons)
        y = np.random.randn(len(df))
        df['y'] = y
        
        # Write to temp parquet file
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_file = os.path.join(tmpdir, "test_data.parquet")
            df.to_parquet(temp_file)
            
            formula = FormulaParser().parse("y ~ x | firm_id")
            model = DuckMundlak(
                db_name=":memory:",
                table_name=f"read_parquet('{temp_file}')",
                formula=formula,
                seed=42,
                remove_singletons=True,
                fitter='numpy'
            )
            
            # Fit the model
            model.fit()
            
            # Create ModelSummary from estimator
            summary = ModelSummary.from_estimator(model)
            summary_dict = summary.to_dict()
            
            # Verify sample_info contains n_rows_dropped_singletons
            assert 'sample_info' in summary_dict
            assert 'n_rows_dropped_singletons' in summary_dict['sample_info']
            assert summary_dict['sample_info']['n_rows_dropped_singletons'] == 1
    
    def test_model_summary_no_dropped_rows_when_disabled(self, data_with_singletons):
        """Test that n_rows_dropped_singletons is 0 when remove_singletons=False."""
        from duckreg.core.results import ModelSummary
        
        df = pd.DataFrame(data_with_singletons)
        y = np.random.randn(len(df))
        df['y'] = y
        
        # Write to temp parquet file
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_file = os.path.join(tmpdir, "test_data.parquet")
            df.to_parquet(temp_file)
            
            formula = FormulaParser().parse("y ~ x | firm_id")
            model = DuckMundlak(
                db_name=":memory:",
                table_name=f"read_parquet('{temp_file}')",
                formula=formula,
                seed=42,
                remove_singletons=False,
                fitter='numpy'
            )
            
            # Fit the model
            model.fit()
            
            # Create ModelSummary from estimator
            summary = ModelSummary.from_estimator(model)
            
            # When remove_singletons=False, no rows should be dropped
            assert summary.n_rows_dropped_singletons == 0
            assert summary.n_obs == len(df)
