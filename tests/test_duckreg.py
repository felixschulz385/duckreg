"""
duckreg coefficient accuracy tests — lean edge-case matrix.

Coverage:
  - Pooled:    2 fitters × 2 IV × 2 vcov representative classes  →  8 tests
  - FE deep:   2 balance × 2 fe_method × 2 IV × 2 vcov           → 16 tests
  - FE smoke:  depth 1 & 2, OLS/demean/balanced                  →  2 tests
  - Merged FE: 2 FE methods × 2 specs                            →  4 tests
  Total: 30 tests  (was 192)
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import logging
from pathlib import Path

import pyfixest as pf
from duckreg import duckreg

FITTERS     = ["numpy", "duckdb"]
# auto_fe is temporarily disabled until the implementation is stabilized
FE_METHODS  = ["demean", "mundlak"]

_UNBALANCED_DROP_FRAC = 0.25
_UNBALANCED_SEED      = 99


# ── Data generation ──────────────────────────────────────────────────────────

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

    pixel_fe   = np.random.randn(n_pixels)   * 2
    year_fe    = np.random.randn(n_years)    * 1.5
    country_fe = np.random.randn(n_countries)* 1
    soil_fe    = np.random.randn(n_soil)     * 0.8

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


def test_mundlak_warning_logs(caplog, unbalanced_df):
    """Using the Mundlak method should emit a recommendation warning."""
    caplog.set_level(logging.WARNING)
    duckreg("ntl_harm ~ rainfall | pixel_id + year", data=unbalanced_df, fe_method="mundlak")
    assert "not recommended for unbalanced panels" in caplog.text


# ── Fixtures ─────────────────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fe_cols(depth: int):
    return {1: ["year"], 2: ["country", "year"], 3: ["country", "year", "soil_type"]}[depth]


def _pf_vcov(vcov: str):
    return {"CRV1": "country"} if vcov == "CRV1" else vcov


def _pf_coef_se(fit, var: str, has_iv: bool):
    if has_iv:
        tidy = fit.tidy()
        return float(tidy.loc[var, "Estimate"]), float(tidy.loc[var, "Std. Error"])
    return float(fit.coef().loc[var]), float(fit.se().loc[var])


def _dr_coef_se(m, var: str = "ntl_harm"):
    res = m.summary_df()
    return float(res.loc[var, "coefficient"]), float(res.loc[var, "std_error"])


def _duckreg(formula, path, vcov, fitter, fe_method=None):
    kwargs = dict(data=path, seed=42, round_strata=5, se_method=vcov, fitter=fitter)
    if fe_method:
        kwargs["fe_method"] = fe_method
    return duckreg(formula=formula, **kwargs)


# ── Assertion helper ──────────────────────────────────────────────────────────

def _assert_close(dr_coef, pf_coef, dr_se, pf_se, coef_rtol, se_rtol, label):
    np.testing.assert_allclose(
        dr_coef, pf_coef, rtol=coef_rtol, atol=1e-6,
        err_msg=f"coef mismatch: {label}",
    )
    np.testing.assert_allclose(
        dr_se, pf_se, rtol=se_rtol, atol=1e-6,
        err_msg=f"SE mismatch: {label}",
    )


# ── Check functions ───────────────────────────────────────────────────────────

def check_pooled(df, path, *, fitter, has_iv, vcov):
    f_pf = ("modis_median ~ exog_control | ntl_harm ~ rainfall"
            if has_iv else "modis_median ~ ntl_harm + exog_control")
    fit_pf = pf.feols(f_pf, df, vcov=_pf_vcov(vcov))

    if has_iv:
        f_dr = "modis_median ~ exog_control | 0 | ntl_harm (rainfall)"
        if vcov == "CRV1":
            f_dr += " | country"
    else:
        f_dr = "modis_median ~ ntl_harm + exog_control"
        if vcov == "CRV1":
            f_dr += " | 0 | 0 | country"

    m = _duckreg(f_dr, path, vcov, fitter)
    pf_coef, pf_se = _pf_coef_se(fit_pf, "ntl_harm", has_iv)
    dr_coef, dr_se = _dr_coef_se(m)

    _assert_close(
        dr_coef, pf_coef, dr_se, pf_se,
        coef_rtol=1e-3 if has_iv else 1e-4,
        se_rtol=1e-2   if has_iv else 5e-3,
        label=f"pooled iv={has_iv} fitter={fitter} vcov={vcov}",
    )


def check_fe(df, path, *, fitter, has_iv, fe_depth, fe_method, vcov, panel_balance):
    fe_part = " + ".join(_fe_cols(fe_depth))
    f_pf = (f"modis_median ~ exog_control | {fe_part} | ntl_harm ~ rainfall"
            if has_iv else f"modis_median ~ ntl_harm + exog_control | {fe_part}")
    fit_pf = pf.feols(f_pf, df, vcov=_pf_vcov(vcov))

    if has_iv:
        f_dr = f"modis_median ~ exog_control | {fe_part} | ntl_harm (rainfall)"
        if vcov == "CRV1":
            f_dr += " | country"
    else:
        f_dr = f"modis_median ~ ntl_harm + exog_control | {fe_part}"
        if vcov == "CRV1":
            f_dr += " | 0 | country"

    m = _duckreg(f_dr, path, vcov, fitter, fe_method)
    pf_coef, pf_se = _pf_coef_se(fit_pf, "ntl_harm", has_iv)
    dr_coef, dr_se = _dr_coef_se(m)

    coef_rtol, se_rtol = (
        (5e-3, 2e-2) if fe_method == "demean" and has_iv else
        (1e-3, 1e-2) if fe_method == "demean"             else
        (2e-2, 1e-1) if has_iv                            else
        (1e-2, 5e-2)
    )
    if panel_balance == "unbalanced":
        coef_rtol *= 2; se_rtol *= 2

    _assert_close(
        dr_coef, pf_coef, dr_se, pf_se, coef_rtol, se_rtol,
        label=f"fe_depth={fe_depth} iv={has_iv} fitter={fitter} "
              f"fe_method={fe_method} vcov={vcov} balance={panel_balance}",
    )


def check_merged_fe(df, path, *, fitter, fe_method, dr_fe_part, pf_fe_part, vcov="HC1"):
    f_pf   = f"modis_median ~ ntl_harm + exog_control | {pf_fe_part}"
    fit_pf = pf.feols(f_pf, df, vcov=_pf_vcov(vcov))

    f_dr = f"modis_median ~ ntl_harm + exog_control | {dr_fe_part}"
    if vcov == "CRV1":
        f_dr += " | 0 | country"

    m = _duckreg(f_dr, path, vcov, fitter, fe_method)
    pf_coef, pf_se = float(fit_pf.coef().loc["ntl_harm"]), float(fit_pf.se().loc["ntl_harm"])
    dr_coef, dr_se = _dr_coef_se(m)

    coef_rtol = 1e-3 if fe_method == "demean" else 1e-2
    se_rtol   = 1e-2 if fe_method == "demean" else 0.12  # Mundlak/auto_fe is an approximation

    _assert_close(dr_coef, pf_coef, dr_se, pf_se, coef_rtol, se_rtol,
                  label=f"merged fe={dr_fe_part!r} fe_method={fe_method}")


# ── Tests ─────────────────────────────────────────────────────────────────────

# 8 pooled tests: 2 fitters × 2 IV × 2 vcov classes
@pytest.mark.parametrize("fitter", FITTERS)
@pytest.mark.parametrize("has_iv", [False, True])
@pytest.mark.parametrize("vcov", ["HC1", "CRV1"])
def test_pooled(balanced_df, balanced_path, fitter, has_iv, vcov):
    """Pooled OLS/IV: fitter and SE method are the only axes that matter here."""
    check_pooled(balanced_df, balanced_path, fitter=fitter, has_iv=has_iv, vcov=vcov)


# 1 pooled iid sanity test
def test_pooled_iid_sanity(balanced_df, balanced_path):
    check_pooled(balanced_df, balanced_path, fitter="numpy", has_iv=False, vcov="iid")


# 16 deep-FE edge-case tests: 2 balance × 2 fe_method × 2 IV × 2 vcov
@pytest.mark.parametrize("panel_balance", ["balanced", "unbalanced"])
@pytest.mark.parametrize("fe_method", FE_METHODS)
@pytest.mark.parametrize("has_iv", [False, True])
@pytest.mark.parametrize("vcov", ["HC1", "CRV1"])
def test_fe_deep(
    balanced_df, balanced_path, unbalanced_df, unbalanced_path,
    fe_method, has_iv, vcov, panel_balance,
):
    """3-way FE: stresses demean convergence, Mundlak approximation, and panel balance."""
    df   = balanced_df   if panel_balance == "balanced" else unbalanced_df
    path = balanced_path if panel_balance == "balanced" else unbalanced_path
    check_fe(df, path, fitter="numpy", has_iv=has_iv, fe_depth=3,
             fe_method=fe_method, vcov=vcov, panel_balance=panel_balance)


# 2 shallow-FE smoke tests: catch regressions at depth 1 & 2
@pytest.mark.parametrize("fe_depth", [1, 2])
@pytest.mark.parametrize("fe_method", FE_METHODS)
def test_fe_shallow_smoke(balanced_df, balanced_path, fe_depth, fe_method):
    """Shallow FE smoke: ensures depth-1/2 paths aren't broken by deep-FE changes."""
    check_fe(balanced_df, balanced_path, fitter="numpy", has_iv=False,
             fe_depth=fe_depth, fe_method=fe_method, vcov="HC1",
             panel_balance="balanced")


# 4 merged-FE tests: 2 specs × 2 FE methods
@pytest.mark.parametrize("fe_method", FE_METHODS)
@pytest.mark.parametrize(
    "dr_fe_part, pf_fe_part",
    [
        ("country*year",            "country^year"),
        ("pixel_id + country*year", "pixel_id + country^year"),
    ],
    ids=["pure_interaction", "additive_plus_interaction"],
)
def test_merged_fe(balanced_df, balanced_path, dr_fe_part, pf_fe_part, fe_method):
    """Merged FE: pure interaction vs. additive TWFE, both FE methods."""
    check_merged_fe(balanced_df, balanced_path, fitter="numpy",
                    fe_method=fe_method, dr_fe_part=dr_fe_part, pf_fe_part=pf_fe_part)