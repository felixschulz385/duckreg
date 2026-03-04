"""
DuckFE (iterative-demean) coefficient accuracy and VCOV tests.

Coverage
--------
Coef / SE accuracy (vs pyfixest.feols oracle):
  - 2 balance × 2 IV × 2 vcov deep FE tests        → 8 tests
  - 2 FE depths × 1 method shallow smoke tests      → 2 tests
  - 2 merged-FE specs                               → 2 tests

VCOV / inference infrastructure:
  - TestDuckFEVcovParams   — get_vcov_fe_params() one-way, two-way, nested
  - TestEndToEndSEParity   — SE == pyfixest for HC1 / iid / CRV1 (two-way FE)
  - TestSSCAutoSelection   — auto SSC from formula: kfixef, Gdf per vcov type
  - TestCompressionCorrectness — nobs == len(data), round_strata stability
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
import duckdb
from duckreg.core.transformers import IterativeDemeanTransformer


# ── constants ─────────────────────────────────────────────────────────────────

_FE_METHOD = "demean"
_UNBALANCED_DROP_FRAC = 0.25
_UNBALANCED_SEED = 99


# ── data generation ───────────────────────────────────────────────────────────

def _make_balanced_panel() -> pd.DataFrame:
    np.random.seed(42)
    n_pixels, n_years, n_countries, n_soil = 1000, 10, 20, 50

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


# ── fixtures (module scope for heavy data, class scope for panel_data) ────────

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
    n_units, n_periods = 40, 15
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

    coef_rtol = 5e-3 if has_iv else 1e-3
    se_rtol   = 1e-2 if has_iv else 5e-3
    if panel_balance == "unbalanced":
        coef_rtol *= 1.5
        se_rtol   *= 1.5

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

    _assert_close(dr_coef, pf_coef, dr_se, pf_se,
                  coef_rtol=1e-3, se_rtol=1e-2,
                  label=f"merged fe={dr_fe_part!r} fe_method={_FE_METHOD}")


# ============================================================================
# Coef / SE accuracy tests
# ============================================================================

# 16 deep-FE tests: 2 fitters × 2 balance × 2 IV × 2 vcov (demean only)
@pytest.mark.parametrize("panel_balance", ["balanced", "unbalanced"])
@pytest.mark.parametrize("has_iv", [False, True])
@pytest.mark.parametrize("vcov", ["HC1", "CRV1"])
@pytest.mark.parametrize("fitter", ["numpy", "duckdb"])
def test_fe_deep(
    balanced_df, balanced_path, unbalanced_df, unbalanced_path,
    fitter, has_iv, vcov, panel_balance,
):
    """3-way FE demean: stresses MAP convergence, panel balance, and both fitters."""
    df   = balanced_df   if panel_balance == "balanced" else unbalanced_df
    path = balanced_path if panel_balance == "balanced" else unbalanced_path
    check_fe(df, path, fitter=fitter, has_iv=has_iv, fe_depth=3,
             vcov=vcov, panel_balance=panel_balance)


# 2 shallow smoke tests: depth 1 & 2 (demean only)
@pytest.mark.parametrize("fe_depth", [1, 2])
def test_fe_shallow_smoke(balanced_df, balanced_path, fe_depth):
    """Shallow FE smoke: ensures depth-1/2 demean paths aren't broken."""
    check_fe(balanced_df, balanced_path, fitter="numpy", has_iv=False,
             fe_depth=fe_depth, vcov="HC1", panel_balance="balanced")


# 2 merged-FE tests (demean only)
@pytest.mark.parametrize(
    "dr_fe_part, pf_fe_part",
    [
        ("country*year",            "country^year"),
        ("pixel_id + country*year", "pixel_id + country^year"),
    ],
    ids=["pure_interaction", "additive_plus_interaction"],
)
def test_merged_fe(balanced_df, balanced_path, dr_fe_part, pf_fe_part):
    """Merged FE (demean): pure interaction vs. additive TWFE."""
    check_merged_fe(balanced_df, balanced_path, fitter="numpy",
                    dr_fe_part=dr_fe_part, pf_fe_part=pf_fe_part)


# ============================================================================
# TestDuckFEVcovParams — get_vcov_fe_params() shape for demean models
# ============================================================================

class TestDuckFEVcovParams:
    """Verify get_vcov_fe_params() returns correct values for demean FE structures."""

    def test_oneway_fe_params(self, panel_data):
        model = duckreg("y ~ x1 + x2 | unit", data=panel_data,
                        se_method="HC1", fe_method="demean", fitter="duckdb")
        kfe, nfe, kfenested, nfefullynested = model.get_vcov_fe_params()
        assert nfe == 1
        assert kfe == panel_data['unit'].nunique()  # 40 levels, no subtraction
        assert kfenested == 0
        assert nfefullynested == 0

    def test_twoway_fe_params_crossed(self, panel_data):
        model = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                        se_method="HC1", fe_method="demean", fitter="duckdb")
        kfe, nfe, kfenested, nfefullynested = model.get_vcov_fe_params()
        n_units = panel_data['unit'].nunique()   # 40
        n_years = panel_data['year'].nunique()   # 15
        assert nfe == 2
        assert kfe == n_units + n_years
        assert kfenested == 0
        assert nfefullynested == 0

    def test_twoway_fe_params_nested(self):
        """Sub-unit nested within unit: subunit is fully nested."""
        rng = np.random.default_rng(7)
        n_units = 10
        subs_per_unit = 4
        obs_per_sub = 20
        n = n_units * subs_per_unit * obs_per_sub
        unit    = np.repeat(np.arange(n_units), subs_per_unit * obs_per_sub)
        subunit = np.repeat(np.arange(n_units * subs_per_unit), obs_per_sub)
        x1 = rng.standard_normal(n)
        y  = (x1
              + np.repeat(rng.standard_normal(n_units), subs_per_unit * obs_per_sub)
              + np.repeat(rng.standard_normal(n_units * subs_per_unit), obs_per_sub)
              + rng.standard_normal(n) * 0.3)
        df = pd.DataFrame({'y': y, 'x1': x1, 'unit': unit, 'subunit': subunit})

        model = duckreg("y ~ x1 | unit + subunit", data=df,
                        se_method="HC1", fe_method="demean", fitter="duckdb")
        kfe, nfe, kfenested, nfefullynested = model.get_vcov_fe_params()

        assert nfe == 2
        assert kfenested == n_units * subs_per_unit   # 40 nested sub-unit levels
        assert nfefullynested == 1                     # subunit fully nested in unit


# ============================================================================
# TestEndToEndSEParity — SE == pyfixest for demean, two-way FE
# ============================================================================

class TestEndToEndSEParity:
    """Numerical SE parity: duckreg (demean) vs pyfixest.feols, two-way FE."""

    @pytest.fixture(autouse=True)
    def _pyfixest(self):
        try:
            import pyfixest  # noqa: F401
        except ImportError:
            pytest.skip("pyfixest not available")

    def test_coefs_match_pyfixest(self, panel_data):
        pf_fit = pf.feols("y ~ x1 + x2 | unit + year",
                          data=panel_data, vcov="HC1")
        pf_coef = np.asarray(pf_fit.coef())
        dr_fit = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                         se_method="HC1", fe_method="demean", fitter="duckdb")
        dr_coef = dr_fit.point_estimate
        np.testing.assert_allclose(dr_coef, pf_coef, rtol=1e-4,
            err_msg="Coefficients diverge before SE comparison is meaningful")

    @pytest.mark.parametrize("pf_vcov,dr_vcov", [
        ("HC1",            "HC1"),
        ("iid",            "iid"),
        # CRV1 with auto SSC: both pyfixest and duckreg use
        # kfixef='nonnested', which subtracts FE levels nested in the cluster
        # (unit FE is nested in the unit cluster → dfk is reduced accordingly).
        ({"CRV1": "unit"}, {"CRV1": "unit"}),
    ])
    def test_se_parity(self, panel_data, pf_vcov, dr_vcov):
        pf_fit = pf.feols("y ~ x1 + x2 | unit + year",
                          data=panel_data, vcov=pf_vcov)
        pf_se = np.asarray(pf_fit.se())

        dr_fit = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                         se_method=dr_vcov, fe_method="demean", fitter="duckdb")
        dr_se = np.sqrt(np.diag(dr_fit.vcov))

        np.testing.assert_allclose(
            dr_se, pf_se, rtol=1e-3,
            err_msg=f"SE mismatch (demean) for vcov={pf_vcov}: "
                    f"duckreg={dr_se} pyfixest={pf_se}",
        )


# ============================================================================
# TestSSCAutoSelection — auto SSC determined from formula properties (demean)
# ============================================================================

class TestSSCAutoSelection:
    """Verify SSC is automatically determined from formula/se_method properties."""

    def test_default_kfixef_is_nonnested(self, panel_data):
        """Non-clustered model should auto-select kfixef='nonnested'."""
        m = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                    se_method="HC1", fe_method="demean", fitter="duckdb")
        assert m.vcov_spec.ssc.kfixef == 'nonnested'

    def test_default_kadj_is_true(self, panel_data):
        """Model should auto-select kadj=True."""
        m = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                    se_method="HC1", fe_method="demean", fitter="duckdb")
        assert m.vcov_spec.ssc.kadj is True

    def test_nonclustered_gdf_is_conventional(self, panel_data):
        """Non-clustered model gets Gdf='conventional'."""
        m = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                    se_method="HC1", fe_method="demean", fitter="duckdb")
        assert m.vcov_spec.ssc.Gdf == 'conventional'

    def test_clustered_gdf_is_min(self, panel_data):
        """Clustered model auto-selects Gdf='min' (matching pyfixest default)."""
        m = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                    se_method={"CRV1": "unit"}, fe_method="demean", fitter="duckdb")
        assert m.vcov_spec.ssc.Gdf == 'min'

    def test_ssc_dict_attr_reflects_auto_ssc(self, panel_data):
        """model.ssc_dict is derived from the auto-selected SSC (for introspection)."""
        m = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                    se_method="HC1", fe_method="demean", fitter="duckdb")
        assert m.ssc_dict == m.vcov_spec.ssc.to_dict()

    def test_ssc_kfixef_affects_se_numerically(self, panel_data):
        """kfixef='full' must give larger SE than kfixef='none' when kfe > 0."""
        from duckreg.core.vcov import SSCConfig, VcovContext, compute_iid_vcov
        from duckreg.core.linalg import safe_inv, safe_solve

        rng = np.random.default_rng(1)
        n, k, kfe, nfe = 600, 3, 55, 2
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        y = X @ np.array([1.0, 0.5, -0.3]) + rng.standard_normal(n)
        XtX = X.T @ X
        theta = safe_solve(XtX, X.T @ y)
        XtXinv = safe_inv(XtX, use_pinv=True)
        rss = float(((y - X @ theta) ** 2).sum())

        ctx = VcovContext(N=n, k=k, kfe=kfe, nfe=nfe)
        cfg_none = SSCConfig.from_dict(
            {'kadj': True, 'kfixef': 'none', 'Gadj': False, 'Gdf': 'conventional'})
        cfg_full = SSCConfig.from_dict(
            {'kadj': True, 'kfixef': 'full', 'Gadj': False, 'Gdf': 'conventional'})

        vcov_none, _ = compute_iid_vcov(XtXinv, rss, ctx, cfg_none)
        vcov_full, _ = compute_iid_vcov(XtXinv, rss, ctx, cfg_full)
        se_none = np.sqrt(np.diag(vcov_none))
        se_full = np.sqrt(np.diag(vcov_full))
        assert np.all(se_full > se_none), (
            "kfixef='full' must inflate SE relative to kfixef='none' when kfe > 0"
        )


# ============================================================================
# TestCompressionCorrectness — nobs and round_strata (demean)
# ============================================================================

class TestCompressionCorrectness:
    """Verify that strata compression doesn't corrupt N or SE (demean)."""

    def test_nobs_is_observation_count_not_strata(self, panel_data):
        model = duckreg(
            "y ~ x1 + x2 | unit + year", data=panel_data,
            se_method="iid", fe_method="demean", fitter="duckdb",
        )
        assert model.nobs == len(panel_data), (
            f"model.nobs={model.nobs} but len(panel_data)={len(panel_data)}; "
            "N in VcovContext is the compressed row count instead of nobs"
        )
        assert model.vcov_meta.get('N', model.nobs) == len(panel_data)

    def test_round_strata_se_stability(self, panel_data):
        dr_exact = duckreg(
            "y ~ x1 + x2 | unit + year", data=panel_data,
            se_method="HC1", fe_method="demean", fitter="duckdb",
            round_strata=None,
        )
        dr_rounded = duckreg(
            "y ~ x1 + x2 | unit + year", data=panel_data,
            se_method="HC1", fe_method="demean", fitter="duckdb",
            round_strata=5,
        )
        se_exact   = np.sqrt(np.diag(dr_exact.vcov))
        se_rounded = np.sqrt(np.diag(dr_rounded.vcov))

        np.testing.assert_allclose(
            se_rounded, se_exact, rtol=0.01,
            err_msg=(
                "round_strata=5 caused >1% SE deviation (demean). "
                "Check that rounding affects strata formation only, "
                "not the residual or score computation."
            )
        )

    def test_df_compressed_structure(self, panel_data):
        """After fitting, df_compressed must contain the standard aggregation columns."""
        m = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                    se_method="iid", fe_method="demean", fitter="duckdb")
        for col in ("count", "sum_y", "sum_y_sq"):
            assert col in m.df_compressed.columns, f"Missing column: {col!r}"
        assert m.df_compressed["count"].sum() == len(panel_data)


# ============================================================================
# TestDemeanConvergence — IterativeDemeanTransformer edge cases
# ============================================================================

class TestDemeanConvergence:
    """Direct tests for IterativeDemeanTransformer convergence behaviour."""

    def _make_conn(self) -> duckdb.DuckDBPyConnection:
        """Fresh DuckDB connection with a small **unbalanced** panel.

        An unbalanced panel is needed so that MAP (alternating projections)
        requires more than one iteration to converge.  The balanced case
        converges in a single pass (round-trip error < 1e-16).
        """
        rng = np.random.default_rng(5)
        n_units, n_periods = 20, 10
        n = n_units * n_periods
        df = pd.DataFrame({
            "x1":   rng.standard_normal(n),
            "x2":   rng.standard_normal(n),
            "y":    rng.standard_normal(n),
            "unit": np.repeat(np.arange(n_units), n_periods),
            "year": np.tile(np.arange(n_periods), n_units),
        })
        # Drop ~30 % of rows to make the panel unbalanced; MAP then needs
        # multiple sweeps before the within-group means stabilise.
        drop_mask = rng.random(len(df)) < 0.30
        df = df[~drop_mask].reset_index(drop=True)
        conn = duckdb.connect()
        conn.register("panel", df)
        return conn

    def _transformer(self, conn, max_iterations=1000, tolerance=1e-8):
        return IterativeDemeanTransformer(
            conn=conn,
            table_name="panel",
            fe_cols=["unit", "year"],
            remove_singletons=False,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

    def test_n_iterations_set_after_fit(self):
        """n_iterations must be a positive integer after fit_transform."""
        conn = self._make_conn()
        t = self._transformer(conn)
        t.fit_transform(["x1", "x2", "y"])
        assert t.n_iterations is not None
        assert t.n_iterations >= 1
        conn.close()

    def test_tight_tolerance_uses_more_iterations(self):
        """Tighter convergence tolerance requires ≥ iterations than a loose one."""
        conn_loose = self._make_conn()
        conn_tight = self._make_conn()
        t_loose = self._transformer(conn_loose, max_iterations=500, tolerance=1e-2)
        t_tight = self._transformer(conn_tight, max_iterations=500, tolerance=1e-7)
        t_loose.fit_transform(["x1", "x2", "y"])
        t_tight.fit_transform(["x1", "x2", "y"])
        assert t_tight.n_iterations >= t_loose.n_iterations, (
            f"Expected tight ({t_tight.n_iterations}) ≥ loose ({t_loose.n_iterations}) iterations"
        )
        conn_loose.close()
        conn_tight.close()

    def test_max_iterations_exceeded_warns(self, caplog):
        """Setting max_iterations=1 on a two-way panel must trigger a convergence warning."""
        caplog.set_level(logging.WARNING)
        conn = self._make_conn()
        t = self._transformer(conn, max_iterations=1, tolerance=1e-12)
        t.fit_transform(["x1", "x2", "y"])
        assert any("MAP did not converge" in r.message for r in caplog.records), (
            "Expected convergence warning when max_iterations is exhausted"
        )
        assert t.n_iterations == 1
        conn.close()


# ============================================================================
# TestSingletonRemovalDemean — remove_singletons for iterative-demean FE
# ============================================================================

class TestSingletonRemovalDemean:
    """Verify singleton group removal for DuckFE (demean method)."""

    @pytest.fixture(scope="class")
    def singleton_path(self, tmp_path_factory):
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
        df = pd.DataFrame(rows)
        path = str(tmp_path_factory.mktemp("demean_singleton") / "data.parquet")
        df.to_parquet(path)
        return path, len(df)  # 31 rows

    def test_removes_singleton_by_default(self, singleton_path):
        path, total = singleton_path
        m = duckreg("y ~ x | unit", data=path, fe_method="demean",
                    fitter="numpy", se_method="none")
        assert m.n_obs == total - 1

    def test_keeps_singletons_when_disabled(self, singleton_path):
        path, total = singleton_path
        m = duckreg("y ~ x | unit", data=path, fe_method="demean",
                    fitter="numpy", se_method="none", remove_singletons=False)
        assert m.n_obs == total

    def test_nrows_dropped_attribute_is_one(self, singleton_path):
        path, _ = singleton_path
        m = duckreg("y ~ x | unit", data=path, fe_method="demean",
                    fitter="numpy", se_method="none")
        assert m.n_rows_dropped_singletons == 1

    def test_estimation_succeeds_after_removal(self, singleton_path):
        path, _ = singleton_path
        m = duckreg("y ~ x | unit", data=path, fe_method="demean",
                    fitter="numpy", se_method="HC1")
        assert m.point_estimate is not None
        assert m.vcov is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
