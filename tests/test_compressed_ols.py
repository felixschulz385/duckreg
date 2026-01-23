import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from pathlib import Path

import pyfixest as pf
from duckreg import compressed_ols


@pytest.fixture(scope="module")
def simulated_panel_data():
    """Generate simulated panel data similar to the notebook example"""
    np.random.seed(42)
    
    # Simulation parameters
    n_pixels = 1000
    n_years = 10
    n_countries = 20
    
    # Create panel structure
    pixels = np.arange(n_pixels)
    years = np.arange(2010, 2010 + n_years)
    panel = pd.MultiIndex.from_product([pixels, years], names=['pixel_id', 'year']).to_frame(index=False)
    
    # Assign countries and create fixed effects
    panel['country'] = (panel['pixel_id'] % n_countries).astype(int)
    pixel_fe = np.random.randn(n_pixels) * 2
    year_fe = np.random.randn(n_years) * 1.5
    country_fe = np.random.randn(n_countries) * 1
    
    # Create instrument (rainfall) - correlated with endogenous variable but not error
    panel['rainfall'] = np.random.randn(len(panel)) * 10 + 100
    
    # Create endogenous variable (ntl_harm) - correlated with instrument and has endogeneity
    panel['ntl_harm'] = (
        0.5 * panel['rainfall'] +  # Instrument effect
        pixel_fe[panel['pixel_id']] +  # Pixel fixed effect
        year_fe[panel['year'] - 2010] +  # Year fixed effect
        np.random.randn(len(panel)) * 5  # Random noise
    )
    
    # Create additional instrument
    panel['temperature'] = np.random.randn(len(panel)) * 2 + 15
    
    # Add second endogenous variable for testing multiple instruments
    panel['population'] = (
        0.3 * panel['rainfall'] +
        0.4 * panel['temperature'] +
        pixel_fe[panel['pixel_id']] * 0.5 +
        np.random.randn(len(panel)) * 3
    )
    
    # Create outcome with endogeneity issue
    omitted_variable = np.random.randn(len(panel)) * 2
    panel['modis_median'] = (
        0.8 * panel['ntl_harm'] +  # True causal effect
        0.5 * panel['population'] +  # Effect of population
        pixel_fe[panel['pixel_id']] * 1.2 +  # Pixel fixed effect
        year_fe[panel['year'] - 2010] * 0.8 +  # Year fixed effect
        country_fe[panel['country']] * 0.6 +  # Country fixed effect
        0.5 * omitted_variable +  # Omitted variable
        0.3 * panel['ntl_harm'] * omitted_variable / 10 +  # Endogeneity
        np.random.randn(len(panel)) * 1  # Random noise
    )
    
    # Add exogenous covariate
    panel['exog_control'] = np.random.randn(len(panel)) * 3
    
    return panel


@pytest.fixture(scope="module")
def test_parquet_file(simulated_panel_data):
    """Save simulated data to a temporary parquet file"""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        temp_path = f.name
    
    simulated_panel_data.to_parquet(temp_path)
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    # Clean up any duckreg cache files
    cache_dir = Path(temp_path).parent / ".duckreg"
    if cache_dir.exists():
        for file in cache_dir.glob("*"):
            file.unlink()
        cache_dir.rmdir()


@pytest.mark.parametrize("fitter", ["numpy", "duckdb"])
def test_compressed_ols_multivariate(simulated_panel_data, test_parquet_file, fitter):
    """Test OLS with multiple covariates"""
    
    # pyfixest
    fit_pf = pf.feols("modis_median ~ ntl_harm + exog_control + population", 
                       simulated_panel_data, vcov="HC1")
    
    # compressed_ols
    m_multi = compressed_ols(
        formula="modis_median ~ ntl_harm + exog_control + population",
        data=test_parquet_file,
        round_strata=5,
        seed=42,
        se_method="HC1",
        fitter=fitter
    )
    
    results = m_multi.summary_df()
    
    # Check each coefficient
    for var in ['ntl_harm', 'exog_control', 'population']:
        pf_coef = fit_pf.coef().loc[var]
        pf_se = fit_pf.se().loc[var]
        
        duckreg_coef = results.loc[var, 'coefficient']
        duckreg_se = results.loc[var, 'std_error']
        
        np.testing.assert_allclose(duckreg_coef, pf_coef, rtol=1e-4, atol=1e-6,
                                    err_msg=f"Multivariate coefficient for {var} doesn't match (fitter={fitter})")
        np.testing.assert_allclose(duckreg_se, pf_se, rtol=1e-3, atol=1e-6,
                                    err_msg=f"Multivariate SE for {var} doesn't match (fitter={fitter})")


# ============================================================================
# VCOV Variant Tests: Test all variance-covariance estimators
# ============================================================================

@pytest.mark.parametrize("fitter,vcov_type", [
    ("numpy", "iid"),
    ("numpy", "HC1"),
    ("numpy", "CRV1"),
    ("duckdb", "iid"),
    ("duckdb", "HC1"),
    ("duckdb", "CRV1"),
])
def test_vcov_variants_pooled(simulated_panel_data, test_parquet_file, fitter, vcov_type):
    """Test all VCOV types for pooled OLS"""
    
    # Map vcov_type to pyfixest format
    if vcov_type == "CRV1":
        pf_vcov = {"CRV1": "country"}
    else:
        pf_vcov = vcov_type
    
    # pyfixest
    fit_pf = pf.feols("modis_median ~ ntl_harm + exog_control", 
                       simulated_panel_data, vcov=pf_vcov)
    pf_coef = fit_pf.coef().loc['ntl_harm']
    pf_se = fit_pf.se().loc['ntl_harm']
    
    # compressed_ols
    formula = "modis_median ~ ntl_harm + exog_control"
    if vcov_type == "CRV1":
        formula += " | 0 | 0 | country"
    
    m = compressed_ols(
        formula=formula,
        data=test_parquet_file,
        round_strata=5,
        seed=42,
        se_method=vcov_type,
        fitter=fitter
    )
    
    results = m.summary_df()
    duckreg_coef = results.loc['ntl_harm', 'coefficient']
    duckreg_se = results.loc['ntl_harm', 'std_error']
    
    # Coefficients should always match exactly (independent of vcov)
    np.testing.assert_allclose(duckreg_coef, pf_coef, rtol=1e-4, atol=1e-6,
                                err_msg=f"Pooled coefficient doesn't match (fitter={fitter}, vcov={vcov_type})")
    
    # Standard errors comparison (tolerance depends on vcov type)
    se_rtol = 0.05 if vcov_type == "CRV1" else 1e-3
    np.testing.assert_allclose(duckreg_se, pf_se, rtol=se_rtol, atol=1e-6,
                                err_msg=f"Pooled SE doesn't match (fitter={fitter}, vcov={vcov_type})")


@pytest.mark.parametrize("fitter,vcov_type", [
    ("numpy", "iid"),
    ("numpy", "HC1"),
    ("numpy", "CRV1"),
    ("duckdb", "iid"),
    ("duckdb", "HC1"),
    ("duckdb", "CRV1"),
])
def test_vcov_variants_fixed_effects(simulated_panel_data, test_parquet_file, fitter, vcov_type):
    """Test all VCOV types for fixed effects OLS"""
    
    # Map vcov_type to pyfixest format
    if vcov_type == "CRV1":
        pf_vcov = {"CRV1": "country"}
    else:
        pf_vcov = vcov_type
    
    # pyfixest with FE
    panel_clean = simulated_panel_data.dropna(subset=["country"])
    fit_pf = pf.feols("modis_median ~ ntl_harm | pixel_id + year", 
                       panel_clean, vcov=pf_vcov)
    pf_coef = fit_pf.coef().values[0]
    pf_se = fit_pf.se().values[0]
    
    # compressed_ols with FE
    formula_cluster = " | country" if vcov_type == "CRV1" else ""
    m = compressed_ols(
        formula=f"modis_median ~ ntl_harm | pixel_id + year | 0{formula_cluster}",
        data=test_parquet_file,
        round_strata=5,
        seed=42,
        fe_method="mundlak",
        se_method=vcov_type,
        fitter=fitter
    )
    
    results = m.summary_df()
    duckreg_coef = results.loc['ntl_harm', 'coefficient']
    duckreg_se = results.loc['ntl_harm', 'std_error']
    
    # Coefficients (Mundlak may have slight differences)
    np.testing.assert_allclose(duckreg_coef, pf_coef, rtol=1e-2, atol=1e-3,
                                err_msg=f"FE coefficient doesn't match (fitter={fitter}, vcov={vcov_type})")
    
    # Standard errors (more relaxed tolerance for FE and clustering)
    se_rtol = 0.15 if vcov_type == "CRV1" else 0.05
    np.testing.assert_allclose(duckreg_se, pf_se, rtol=se_rtol, atol=1e-3,
                                err_msg=f"FE SE doesn't match (fitter={fitter}, vcov={vcov_type})")


@pytest.mark.parametrize("fitter,vcov_type", [
    ("numpy", "iid"),
    ("numpy", "HC1"),
    ("numpy", "CRV1"),
    ("duckdb", "iid"),
    ("duckdb", "HC1"),
    ("duckdb", "CRV1"),
])
def test_vcov_variants_iv(simulated_panel_data, test_parquet_file, fitter, vcov_type):
    """Test all VCOV types for IV/2SLS"""
    
    # Map vcov_type to pyfixest format
    if vcov_type == "CRV1":
        pf_vcov = {"CRV1": "country"}
    else:
        pf_vcov = vcov_type
    
    # pyfixest IV
    fit_pf = pf.feols("modis_median ~ exog_control | ntl_harm ~ rainfall", 
                       simulated_panel_data, vcov=pf_vcov)
    coef_df = fit_pf.tidy()
    pf_coef = coef_df.loc['ntl_harm', 'Estimate']
    pf_se = coef_df.loc['ntl_harm', 'Std. Error']
    
    # compressed_ols IV
    formula_cluster = " | country" if vcov_type == "CRV1" else ""
    m = compressed_ols(
        formula=f"modis_median ~ exog_control | 0 | ntl_harm (rainfall){formula_cluster}",
        data=test_parquet_file,
        round_strata=5,
        seed=42,
        se_method=vcov_type,
        fitter=fitter
    )
    
    results = m.summary_df()
    duckreg_coef = results.loc['ntl_harm', 'coefficient']
    duckreg_se = results.loc['ntl_harm', 'std_error']
    
    # Coefficients
    np.testing.assert_allclose(duckreg_coef, pf_coef, rtol=1e-2, atol=1e-3,
                                err_msg=f"IV coefficient doesn't match (fitter={fitter}, vcov={vcov_type})")
    
    # Standard errors (relaxed tolerance for IV)
    se_rtol = 0.15 if vcov_type == "CRV1" else 0.05
    np.testing.assert_allclose(duckreg_se, pf_se, rtol=se_rtol, atol=1e-3,
                                err_msg=f"IV SE doesn't match (fitter={fitter}, vcov={vcov_type})")


@pytest.mark.parametrize("fitter,vcov_type", [
    ("numpy", "iid"),
    ("numpy", "HC1"),
    ("numpy", "CRV1"),
    ("duckdb", "iid"),
    ("duckdb", "HC1"),
    ("duckdb", "CRV1"),
])
def test_vcov_variants_iv_with_fe(simulated_panel_data, test_parquet_file, fitter, vcov_type):
    """Test all VCOV types for IV/2SLS with fixed effects"""
    
    # Map vcov_type to pyfixest format
    if vcov_type == "CRV1":
        pf_vcov = {"CRV1": "country"}
    else:
        pf_vcov = vcov_type
    
    # pyfixest with FE and IV
    fit_pf = pf.feols("modis_median ~ exog_control | pixel_id + year | ntl_harm ~ rainfall",
                       simulated_panel_data, vcov=pf_vcov)
    coef_df = fit_pf.tidy()
    pf_coef = coef_df.loc['ntl_harm', 'Estimate']
    pf_se = coef_df.loc['ntl_harm', 'Std. Error']
    
    # compressed_ols with FE and IV
    formula_cluster = " | country" if vcov_type == "CRV1" else ""
    m = compressed_ols(
        formula=f"modis_median ~ exog_control | pixel_id + year | ntl_harm (rainfall){formula_cluster}",
        data=test_parquet_file,
        round_strata=5,
        seed=42,
        fe_method="mundlak",
        se_method=vcov_type,
        fitter=fitter
    )
    
    results = m.summary_df()
    duckreg_coef = results.loc['ntl_harm', 'coefficient']
    duckreg_se = results.loc['ntl_harm', 'std_error']
    
    # Coefficients
    np.testing.assert_allclose(duckreg_coef, pf_coef, rtol=1e-2, atol=1e-3,
                                err_msg=f"IV+FE coefficient doesn't match (fitter={fitter}, vcov={vcov_type})")
    
    # Standard errors (most relaxed tolerance for IV+FE+clustering)
    se_rtol = 0.2 if vcov_type == "CRV1" else 0.1
    np.testing.assert_allclose(duckreg_se, pf_se, rtol=se_rtol, atol=1e-3,
                                err_msg=f"IV+FE SE doesn't match (fitter={fitter}, vcov={vcov_type})")
