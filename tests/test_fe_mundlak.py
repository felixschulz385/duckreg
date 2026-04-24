"""
DuckFE (Mundlak device) coefficient accuracy and VCOV tests.

Coverage
--------
Coef / SE accuracy (vs pyfixest.feols oracle):
  - Mundlak warning on unbalanced panels               → 1 test
  - 2 balance × 2 IV × 2 vcov deep FE tests            → 8 tests
  - 2 FE depths × 1 method shallow smoke tests         → 2 tests
  - 2 merged-FE specs                                  → 2 tests

VCOV / inference infrastructure:
  - TestDuckFEVcovParamsMundlak  — get_vcov_fe_params() returns (0,0,0,0) for Mundlak
  - TestEndToEndSEParityMundlak  — SE ≈ pyfixest for HC1 / iid / CRV1 (two-way FE)
  - TestSSCAutoSelectionMundlak  — auto SSC from formula: kfixef, Gdf per vcov type
  - TestCompressionCorrectnessMundlak — nobs == len(data), round_strata stability

Note: Mundlak is an approximation of within-estimator FE absorption.
      Coefficient and SE tolerances are wider than for the exact demean method.
"""

import os
import tempfile
import logging
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import pyfixest as pf
from duckreg import duckreg
from duckreg.estimators import DuckFE
from duckreg.utils.formula_parser import FormulaParser
from duckreg.core.results import ModelSummary


# ── constants ─────────────────────────────────────────────────────────────────

_FE_METHOD = "mundlak"
_UNBALANCED_DROP_FRAC = 0.25
_UNBALANCED_SEED = 99


# ── data generation ───────────────────────────────────────────────────────────

def _make_balanced_panel() -> pd.DataFrame:
    np.random.seed(42)
    n_pixels, n_years, n_countries, n_soil = 200, 6, 20, 50

    panel = pd.MultiIndex.from_product(
        [np.arange(n_pixels), np.arange(2010, 2010 + n_years)],
        names=["pixel_id", "year"],
    ).to_frame(index=False)

    panel["country"]   = (panel["pixel_id"] % n_countries).astype(int)
    soil_by_pixel      = np.random.randint(0, n_soil, size=n_pixels)
    panel["soil_type"] = soil_by_pixel[panel["pixel_id"]]

    pixel_fe   = np.random.randn(n_pixels)    * 2
    year_fe    = np.random.randn(n_years)     * 1.5
    country_fe = np.random.randn(n_countries) * 1
    soil_fe    = np.random.randn(n_soil)      * 0.8

    panel["rainfall"] = np.random.randn(len(panel)) * 10 + 100
    panel["ntl_harm"] = (
        0.5 * panel["rainfall"]
        + pixel_fe[panel["pixel_id"]]
        + year_fe[panel["year"] - 2010]
        + np.random.randn(len(panel)) * 5
    )

    ov = np.random.randn(len(panel)) * 2
    panel["modis_median"] = (
        0.8 * panel["ntl_harm"]
        + pixel_fe[panel["pixel_id"]] * 1.2
        + year_fe[panel["year"] - 2010] * 0.8
        + country_fe[panel["country"]] * 0.6
        + soil_fe[panel["soil_type"]] * 0.5
        + 0.5 * ov
        + 0.3 * panel["ntl_harm"] * ov / 10
        + np.random.randn(len(panel)) * 1
    )
    panel["exog_control"] = np.random.randn(len(panel)) * 3
    return panel.reset_index(drop=True)


def _make_unbalanced_panel(balanced: pd.DataFrame) -> pd.DataFrame:
    rng   = np.random.default_rng(_UNBALANCED_SEED)
    panel = balanced.loc[rng.random(len(balanced)) >= _UNBALANCED_DROP_FRAC].copy()
    panel = panel.reset_index(drop=True)

    fe_dims = ["pixel_id", "year", "country", "soil_type"]
    changed = True
    while changed:
        changed = False
        for dim in fe_dims:
            counts = panel.groupby(dim)[dim].transform("count")
            before = len(panel)
            panel  = panel.loc[counts > 1].reset_index(drop=True)
            if len(panel) < before:
                changed = True
    return panel


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def balanced_df():
    return _make_balanced_panel()


@pytest.fixture(scope="module")
def unbalanced_df(balanced_df):
    return _make_unbalanced_panel(balanced_df)


def _write_parquet(df: pd.DataFrame) -> str:
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    df.to_parquet(path)
    return path


def _cleanup(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
    cache = Path(path).parent / ".duckreg"
    if cache.exists():
        for f in cache.glob("*"):
            f.unlink()
        cache.rmdir()


@pytest.fixture(scope="module")
def balanced_path(balanced_df):
    path = _write_parquet(balanced_df)
    yield path
    _cleanup(path)


@pytest.fixture(scope="module")
def unbalanced_path(unbalanced_df):
    path = _write_parquet(unbalanced_df)
    yield path
    _cleanup(path)


@pytest.fixture(scope="class")
def panel_data():
    rng = np.random.default_rng(42)
    n_units, n_periods = 30, 10
    n = n_units * n_periods
    unit = np.repeat(np.arange(n_units), n_periods)
    year = np.tile(np.arange(n_periods), n_units)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    unit_fe = np.repeat(rng.standard_normal(n_units), n_periods)
    year_fe = np.tile(rng.standard_normal(n_periods), n_units)
    y = 1.5*x1 + 0.8*x2 + unit_fe + year_fe + rng.standard_normal(n)*0.5
    return pd.DataFrame({'y': y, 'x1': x1, 'x2': x2,
                         'unit': unit, 'year': year})


# ── shared helpers ────────────────────────────────────────────────────────────

def _fe_cols(depth: int):
    return {1: ["year"], 2: ["country", "year"], 3: ["country", "year", "soil_type"]}[depth]


def _pf_vcov(vcov):
    return {"CRV1": "country"} if vcov == "CRV1" else vcov


def _pf_coef_se(fit, var: str, has_iv: bool):
    if has_iv:
        tidy = fit.tidy()
        return float(tidy.loc[var, "Estimate"]), float(tidy.loc[var, "Std. Error"])
    return float(fit.coef().loc[var]), float(fit.se().loc[var])


def _dr_coef_se(m, var: str = "ntl_harm"):
    res = m.summary_df()
    return float(res.loc[var, "coefficient"]), float(res.loc[var, "std_error"])


def _duckreg(formula, path, vcov, fitter="numpy", fe_method=_FE_METHOD):
    return duckreg(formula=formula, data=path, seed=42, round_strata=5,
                   se_method=vcov, fitter=fitter, fe_method=fe_method)


def _dr_se(vcov):
    """Convert a vcov label to a duckreg se_method (dict for CRV1)."""
    return {"CRV1": "country"} if vcov == "CRV1" else vcov


def _assert_close(dr_coef, pf_coef, dr_se, pf_se, coef_rtol, se_rtol, label):
    np.testing.assert_allclose(
        dr_coef, pf_coef, rtol=coef_rtol, atol=1e-6,
        err_msg=f"coef mismatch: {label}",
    )
    np.testing.assert_allclose(
        dr_se, pf_se, rtol=se_rtol, atol=1e-6,
        err_msg=f"SE mismatch: {label}",
    )


def check_fe(df, path, *, fitter, has_iv, fe_depth, vcov, panel_balance):
    fe_part = " + ".join(_fe_cols(fe_depth))
    f_pf = (f"modis_median ~ exog_control | {fe_part} | ntl_harm ~ rainfall"
            if has_iv else f"modis_median ~ ntl_harm + exog_control | {fe_part}")
    fit_pf = pf.feols(f_pf, df, vcov=_pf_vcov(vcov))

    if has_iv:
        f_dr = f"modis_median ~ exog_control | {fe_part} | (ntl_harm ~ rainfall)"
    else:
        f_dr = f"modis_median ~ ntl_harm + exog_control | {fe_part}"

    m = _duckreg(f_dr, path, _dr_se(vcov), fitter)
    pf_coef, pf_se = _pf_coef_se(fit_pf, "ntl_harm", has_iv)
    dr_coef, dr_se = _dr_coef_se(m)

    # Mundlak is an approximation — wider tolerances than demean
    coef_rtol = 2e-2 if has_iv else 1e-2
    se_rtol   = 1e-1 if has_iv else 5e-2
    if panel_balance == "unbalanced":
        coef_rtol *= 2
        se_rtol   *= 2

    _assert_close(
        dr_coef, pf_coef, dr_se, pf_se, coef_rtol, se_rtol,
        label=f"fe_depth={fe_depth} iv={has_iv} fitter={fitter} "
              f"fe_method={_FE_METHOD} vcov={vcov} balance={panel_balance}",
    )


def check_merged_fe(df, path, *, fitter, dr_fe_part, pf_fe_part, vcov="HC1"):
    f_pf   = f"modis_median ~ ntl_harm + exog_control | {pf_fe_part}"
    fit_pf = pf.feols(f_pf, df, vcov=_pf_vcov(vcov))

    f_dr = f"modis_median ~ ntl_harm + exog_control | {dr_fe_part}"
    m = _duckreg(f_dr, path, _dr_se(vcov), fitter)
    pf_coef = float(fit_pf.coef().loc["ntl_harm"])
    pf_se   = float(fit_pf.se().loc["ntl_harm"])
    dr_coef, dr_se = _dr_coef_se(m)

    # Mundlak approximation — 15% SE and 1% coef tolerance
    _assert_close(dr_coef, pf_coef, dr_se, pf_se,
                  coef_rtol=1e-2, se_rtol=0.15,
                  label=f"merged fe={dr_fe_part!r} fe_method={_FE_METHOD}")


# ============================================================================
# Mundlak warning
# ============================================================================

def test_mundlak_warning_logs(caplog, unbalanced_df, unbalanced_path):
    """Using the Mundlak method on an unbalanced panel should emit a warning."""
    caplog.set_level(logging.WARNING)
    duckreg("ntl_harm ~ rainfall | pixel_id + year",
            data=unbalanced_path, fe_method="mundlak")
    assert "not recommended for unbalanced panels" in caplog.text


def test_unbalanced_accuracy_vs_demean(unbalanced_df, unbalanced_path,
                                       balanced_df, balanced_path):
    """Mundlak coef on an unbalanced panel should stay within 30% of the demean
    estimate (both methods estimate the same within-group effect)."""
    formula = "modis_median ~ ntl_harm + exog_control | country + year"
    dr_demean  = duckreg(formula, data=unbalanced_path, fe_method="demean",
                         se_method="HC1", fitter="numpy")
    dr_mundlak = duckreg(formula, data=unbalanced_path, fe_method="mundlak",
                         se_method="HC1", fitter="numpy")

    coef_demean  = float(dr_demean.summary_df().loc["ntl_harm", "coefficient"])
    coef_mundlak = float(dr_mundlak.summary_df().loc["ntl_harm", "coefficient"])

    rtol = 0.30  # Mundlak is an approximation; allow 30% on unbalanced
    assert abs(coef_mundlak - coef_demean) / (abs(coef_demean) + 1e-8) < rtol, (
        f"Mundlak coef {coef_mundlak:.4f} deviates >{int(rtol * 100)}% "
        f"from demean {coef_demean:.4f} on unbalanced panel"
    )


# ============================================================================
# Coef / SE accuracy tests
# ============================================================================

# 8 deep-FE tests: 1 fitter × 2 balance × 2 IV × 2 vcov (mundlak only)
@pytest.mark.parametrize("panel_balance", ["balanced", "unbalanced"])
@pytest.mark.parametrize("has_iv", [False, True])
@pytest.mark.parametrize("vcov", ["HC1", "CRV1"])
@pytest.mark.parametrize("fitter", ["numpy"])
def test_fe_deep(
    balanced_df, balanced_path, unbalanced_df, unbalanced_path,
    fitter, has_iv, vcov, panel_balance,
):
    """3-way FE Mundlak: tests approximation, panel balance, and both fitters."""
    df   = balanced_df   if panel_balance == "balanced" else unbalanced_df
    path = balanced_path if panel_balance == "balanced" else unbalanced_path
    check_fe(df, path, fitter=fitter, has_iv=has_iv, fe_depth=3,
             vcov=vcov, panel_balance=panel_balance)


# 2 shallow smoke tests: depth 1 & 2
@pytest.mark.parametrize("fe_depth", [1, 2])
def test_fe_shallow_smoke(balanced_df, balanced_path, fe_depth):
    """Shallow FE smoke (Mundlak): ensures depth-1/2 paths aren't broken."""
    check_fe(balanced_df, balanced_path, fitter="numpy", has_iv=False,
             fe_depth=fe_depth, vcov="HC1", panel_balance="balanced")


# 2 merged-FE tests
@pytest.mark.parametrize(
    "dr_fe_part, pf_fe_part",
    [
        ("country*year",            "country^year"),
        ("pixel_id + country*year", "pixel_id + country^year"),
    ],
    ids=["pure_interaction", "additive_plus_interaction"],
)
def test_merged_fe(balanced_df, balanced_path, dr_fe_part, pf_fe_part):
    """Merged FE (Mundlak): pure interaction vs. additive TWFE."""
    check_merged_fe(balanced_df, balanced_path, fitter="numpy",
                    dr_fe_part=dr_fe_part, pf_fe_part=pf_fe_part)


# ============================================================================
# TestDuckFEVcovParamsMundlak
# ============================================================================

class TestDuckFEVcovParamsMundlak:
    """For Mundlak, FE parameters are explicit regressors → get_vcov_fe_params()==(0,0,0,0)."""

    def test_oneway_returns_zeros(self, panel_data):
        model = duckreg("y ~ x1 + x2 | unit", data=panel_data,
                        se_method="HC1", fe_method="mundlak", fitter="duckdb")
        kfe, nfe, kfenested, nfefullynested = model.get_vcov_fe_params()
        assert nfe == 0, (
            "Mundlak includes FE as explicit regressors; "
            "n_fe should be 0 (no absorbed FE dimensions)"
        )
        assert kfe == 0
        assert kfenested == 0
        assert nfefullynested == 0

    def test_twoway_returns_zeros(self, panel_data):
        model = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                        se_method="HC1", fe_method="mundlak", fitter="duckdb")
        kfe, nfe, kfenested, nfefullynested = model.get_vcov_fe_params()
        assert nfe == 0
        assert kfe == 0
        assert kfenested == 0
        assert nfefullynested == 0


# ============================================================================
# TestEndToEndSEParityMundlak — SE ≈ pyfixest for Mundlak, two-way FE
# ============================================================================

class TestEndToEndSEParityMundlak:
    """
    Mundlak SE parity vs pyfixest.feols (two-way FE).

    Mundlak is an approximation of the within-estimator, so we use looser
    tolerances (rtol=5e-2 for HC1, rtol=0.15 for CRV1) than the demean suite.
    """

    @pytest.fixture(autouse=True)
    def _pyfixest(self):
        try:
            import pyfixest  # noqa: F401
        except ImportError:
            pytest.skip("pyfixest not available")

    def test_coefs_in_range(self, panel_data):
        """Mundlak x1/x2 coefficients should be within 5% of pyfixest within-estimator."""
        pf_fit = pf.feols("y ~ x1 + x2 | unit + year",
                          data=panel_data, vcov="HC1")
        pf_coef = np.asarray(pf_fit.coef())  # shape (2,): [x1, x2]
        dr_fit = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                         se_method="HC1", fe_method="mundlak", fitter="duckdb")
        # Mundlak adds group-mean columns; extract only the original regressors
        summary = dr_fit.summary_df()
        dr_coef = summary.loc[['x1', 'x2'], 'coefficient'].values
        np.testing.assert_allclose(dr_coef, pf_coef, rtol=5e-2,
            err_msg="Mundlak coefficients deviate > 5% from pyfixest within-estimator")

    @pytest.mark.parametrize("pf_vcov,dr_vcov", [
        ("HC1",            "HC1"),
        ("iid",            "iid"),
        ({"CRV1": "unit"}, {"CRV1": "unit"}),
    ])
    def test_se_in_range(self, panel_data, pf_vcov, dr_vcov):
        """Mundlak x1/x2 SEs should be in a reasonable range vs. pyfixest (rtol=0.15)."""
        pf_fit = pf.feols("y ~ x1 + x2 | unit + year",
                          data=panel_data, vcov=pf_vcov)
        pf_se = np.asarray(pf_fit.se())  # shape (2,): [x1, x2]

        dr_fit = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                         se_method=dr_vcov, fe_method="mundlak", fitter="duckdb")
        # Mundlak adds group-mean columns; extract only the original regressors
        summary = dr_fit.summary_df()
        dr_se = summary.loc[['x1', 'x2'], 'std_error'].values

        np.testing.assert_allclose(
            dr_se, pf_se, rtol=0.15,
            err_msg=f"SE mismatch (Mundlak) for vcov={pf_vcov}: "
                    f"duckreg={dr_se} pyfixest={pf_se}",
        )


# ============================================================================
# TestSSCAutoSelectionMundlak — auto SSC from formula properties (Mundlak)
# ============================================================================

class TestSSCAutoSelectionMundlak:
    """Verify SSC is automatically determined from formula/se_method properties (Mundlak)."""

    def test_default_kfixef_is_nonnested(self, panel_data):
        """Non-clustered Mundlak model should auto-select kfixef='nonnested'."""
        m = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                    se_method="HC1", fe_method="mundlak", fitter="duckdb")
        assert m.vcov_spec.ssc.kfixef == 'nonnested'

    def test_default_kadj_is_true(self, panel_data):
        """Model should auto-select kadj=True."""
        m = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                    se_method="HC1", fe_method="mundlak", fitter="duckdb")
        assert m.vcov_spec.ssc.kadj is True

    def test_nonclustered_gdf_is_conventional(self, panel_data):
        """Non-clustered Mundlak model gets Gdf='conventional'."""
        m = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                    se_method="HC1", fe_method="mundlak", fitter="duckdb")
        assert m.vcov_spec.ssc.Gdf == 'conventional'

    def test_clustered_gdf_is_min(self, panel_data):
        """Clustered Mundlak model auto-selects Gdf='min' (matching pyfixest default)."""
        m = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                    se_method={"CRV1": "unit"}, fe_method="mundlak", fitter="duckdb")
        assert m.vcov_spec.ssc.Gdf == 'min'

    def test_ssc_dict_attr_reflects_auto_ssc(self, panel_data):
        """model.ssc_dict is derived from the auto-selected SSC."""
        m = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                    se_method="HC1", fe_method="mundlak", fitter="duckdb")
        assert m.ssc_dict == m.vcov_spec.ssc.to_dict()


# ============================================================================
# TestCompressionCorrectnessMundlak — nobs and round_strata (Mundlak)
# ============================================================================

class TestCompressionCorrectnessMundlak:
    """Verify that strata compression doesn't corrupt N or SE (Mundlak)."""

    def test_nobs_is_observation_count_not_strata(self, panel_data):
        model = duckreg(
            "y ~ x1 + x2 | unit + year", data=panel_data,
            se_method="iid", fe_method="mundlak", fitter="duckdb",
        )
        assert model.nobs == len(panel_data), (
            f"model.nobs={model.nobs} but len(panel_data)={len(panel_data)}; "
            "N in VcovContext is the compressed row count instead of nobs (Mundlak)"
        )
        assert model.vcov_meta.get('N', model.nobs) == len(panel_data)

    def test_round_strata_se_stability(self, panel_data):
        dr_exact = duckreg(
            "y ~ x1 + x2 | unit + year", data=panel_data,
            se_method="HC1", fe_method="mundlak", fitter="duckdb",
            round_strata=None,
        )
        dr_rounded = duckreg(
            "y ~ x1 + x2 | unit + year", data=panel_data,
            se_method="HC1", fe_method="mundlak", fitter="duckdb",
            round_strata=5,
        )
        se_exact   = np.sqrt(np.diag(dr_exact.vcov))
        se_rounded = np.sqrt(np.diag(dr_rounded.vcov))

        np.testing.assert_allclose(
            se_rounded, se_exact, rtol=0.01,
            err_msg=(
                "round_strata=5 caused >1% SE deviation (Mundlak). "
                "Check that rounding affects strata formation only, "
                "not the residual or score computation."
            )
        )


# ── Singleton fixtures (shared by the classes below) ───────────────────────────

@pytest.fixture(scope="module")
def singleton_panel():
    """Balanced panel (6 units × 5 years) plus one singleton unit (99)."""
    rng = np.random.default_rng(13)
    rows = [
        {"unit": u, "year": t,
         "x": float(rng.standard_normal()),
         "y": float(rng.standard_normal())}
        for u in range(6) for t in range(5)
    ]
    rows.append({"unit": 99, "year": 0,
                 "x": float(rng.standard_normal()),
                 "y": float(rng.standard_normal())})
    return pd.DataFrame(rows)  # 31 rows


@pytest.fixture(scope="module")
def singleton_panel_path(singleton_panel):
    path = _write_parquet(singleton_panel)
    yield path
    _cleanup(path)


# ============================================================================
# TestMundlakCompressionStructure — df_compressed properties
# ============================================================================

class TestMundlakCompressionStructure:
    """Verify that DuckFE (Mundlak) populates df_compressed correctly."""

    def test_mundlak_means_in_df_compressed(self, balanced_path):
        """Asymptotic FE (pixel_id, 1000 levels) must produce avg_*_feN columns."""
        m = duckreg("modis_median ~ ntl_harm + exog_control | pixel_id",
                    data=balanced_path, fe_method="mundlak",
                    round_strata=3, se_method="none")
        assert "avg_ntl_harm_fe0" in m.df_compressed.columns
        assert "avg_exog_control_fe0" in m.df_compressed.columns

    def test_rhs_cols_includes_means(self, balanced_path):
        """_rhs_cols must list both raw covariates and their Mundlak means."""
        m = duckreg("modis_median ~ ntl_harm + exog_control | pixel_id",
                    data=balanced_path, fe_method="mundlak",
                    round_strata=3, se_method="none", fitter="numpy")
        assert hasattr(m, "_rhs_cols")
        assert "ntl_harm" in m._rhs_cols
        assert "avg_ntl_harm_fe0" in m._rhs_cols

    def test_mixed_fe_types_count_preserved(self, balanced_df, balanced_path):
        """pixel_id (asymptotic) + year (fixed dummies): count must still sum to N."""
        m = duckreg("modis_median ~ ntl_harm | pixel_id + year",
                    data=balanced_path, fe_method="mundlak",
                    round_strata=3, se_method="none", fitter="numpy")
        assert m.df_compressed["count"].sum() == len(balanced_df)
        # Mundlak means for pixel_id and dummies for year must both be present
        assert any("avg_ntl_harm" in c for c in m.df_compressed.columns)
        assert any(c.startswith("dummy_year_") for c in m.df_compressed.columns)

    def test_rounding_more_strata_higher_precision(self, balanced_path):
        """round_strata=5 must yield at least as many Mundlak strata as round_strata=3."""
        m3 = duckreg("modis_median ~ ntl_harm | pixel_id",
                     data=balanced_path, fe_method="mundlak",
                     round_strata=3, se_method="none")
        m5 = duckreg("modis_median ~ ntl_harm | pixel_id",
                     data=balanced_path, fe_method="mundlak",
                     round_strata=5, se_method="none")
        assert len(m5.df_compressed) >= len(m3.df_compressed)


# ============================================================================
# TestSingletonRemovalMundlak — remove_singletons for Mundlak FE
# ============================================================================

class TestSingletonRemovalMundlak:
    """Verify singleton group removal for DuckFE (Mundlak method)."""

    def test_removes_singleton_by_default(self, singleton_panel, singleton_panel_path):
        m = duckreg("y ~ x | unit", data=singleton_panel_path, fe_method="mundlak",
                    fitter="numpy", se_method="none")
        assert m.n_obs == len(singleton_panel) - 1

    def test_keeps_singletons_when_disabled(self, singleton_panel, singleton_panel_path):
        m = duckreg("y ~ x | unit", data=singleton_panel_path, fe_method="mundlak",
                    fitter="numpy", se_method="none", remove_singletons=False)
        assert m.n_obs == len(singleton_panel)

    def test_nrows_dropped_attribute_is_one(self, singleton_panel_path):
        m = duckreg("y ~ x | unit", data=singleton_panel_path, fe_method="mundlak",
                    fitter="numpy", se_method="none")
        assert m.n_rows_dropped_singletons == 1

    def test_multiple_fe_removal(self, singleton_panel_path):
        """Singleton removal with two FE columns must still produce a valid estimate."""
        m = duckreg("y ~ x | unit + year", data=singleton_panel_path, fe_method="mundlak",
                    fitter="numpy", se_method="none", remove_singletons=True)
        assert m.n_obs > 0 and m.point_estimate is not None

    def test_estimation_succeeds_after_removal(self, singleton_panel_path):
        m = duckreg("y ~ x | unit", data=singleton_panel_path, fe_method="mundlak",
                    fitter="numpy", se_method="HC1")
        assert m.point_estimate is not None
        assert m.vcov is not None


# ============================================================================
# TestQualifyClauseGeneration — _build_qualify_singleton_filter on DuckFE
# ============================================================================

class TestQualifyClauseGeneration:
    """Unit tests for the QUALIFY clause builder on the base estimator."""

    def _make_fe(self, path, remove_singletons):
        formula = FormulaParser().parse("y ~ x | unit")
        return DuckFE(
            db_name=":memory:",
            table_name=f"read_parquet('{path}')",
            formula=formula,
            seed=42,
            remove_singletons=remove_singletons,
        )

    def test_qualify_single_fe(self, singleton_panel_path):
        model = self._make_fe(singleton_panel_path, remove_singletons=True)
        clause = model._build_qualify_singleton_filter(["unit"])
        assert "QUALIFY" in clause
        assert "PARTITION BY" in clause
        assert "unit" in clause
        assert "> 1" in clause

    def test_qualify_multiple_fe(self, singleton_panel_path):
        formula = FormulaParser().parse("y ~ x | unit + year")
        model = DuckFE(
            db_name=":memory:",
            table_name=f"read_parquet('{singleton_panel_path}')",
            formula=formula,
            seed=42,
            remove_singletons=True,
        )
        clause = model._build_qualify_singleton_filter(["unit", "year"])
        assert "PARTITION BY" in clause
        assert "unit" in clause
        assert "year" in clause

    def test_qualify_disabled_returns_empty(self, singleton_panel_path):
        model = self._make_fe(singleton_panel_path, remove_singletons=False)
        clause = model._build_qualify_singleton_filter(["unit"])
        assert clause == ""


# ============================================================================
# TestModelSummaryIntegration — n_rows_dropped_singletons in ModelSummary
# ============================================================================

class TestModelSummaryIntegration:
    """Verify that n_rows_dropped_singletons flows through ModelSummary."""

    def _fit_fe(self, singleton_panel_path, remove_singletons):
        formula = FormulaParser().parse("y ~ x | unit")
        model = DuckFE(
            db_name=":memory:",
            table_name=f"read_parquet('{singleton_panel_path}')",
            formula=formula,
            seed=42,
            remove_singletons=remove_singletons,
            fitter="numpy",
        )
        model.fit()
        return model

    def test_n_rows_dropped_tracked(self, singleton_panel, singleton_panel_path):
        model = self._fit_fe(singleton_panel_path, remove_singletons=True)
        summary = ModelSummary.from_estimator(model)
        assert summary.n_rows_dropped_singletons == 1
        assert summary.n_obs == len(singleton_panel) - 1

    def test_to_dict_includes_dropped_rows(self, singleton_panel_path):
        model = self._fit_fe(singleton_panel_path, remove_singletons=True)
        summary = ModelSummary.from_estimator(model)
        d = summary.to_dict()
        assert "sample_info" in d
        assert d["sample_info"]["n_rows_dropped_singletons"] == 1

    def test_no_dropped_when_singletons_kept(self, singleton_panel, singleton_panel_path):
        model = self._fit_fe(singleton_panel_path, remove_singletons=False)
        summary = ModelSummary.from_estimator(model)
        assert summary.n_rows_dropped_singletons == 0
        assert summary.n_obs == len(singleton_panel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
