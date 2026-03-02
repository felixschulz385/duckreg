"""
Pooled (no FE) OLS and IV accuracy tests — including fitter-level VCOV tests.

Structure
---------
A.  Helpers             — ols(), scores() for building vcov matrices from scratch
B.  Low-level fixtures  — generic numpy arrays (simple, weighted, clustered, het, FE)
C.  Panel fixtures      — balanced panel parquet file for end-to-end accuracy
D.  TestPooledCoef      — coefficient / SE parity vs pyfixest.feols (8 + 1 tests)
E.  TestVcovDispatch    — compute_vcov_dispatch: iid / hetero / cluster branches
F.  TestNumpyFitterVcov — NumpyFitter.fit_vcov end-to-end
G.  TestDuckDBFitterVcov— DuckDBFitter.fit_vcov end-to-end
H.  TestVcovVsPyfixest  — numerical SE parity with pyfixest for all SE types

FE-method-specific tests live in:
  - tests/test_fe_demean.py   (iterative-demean method)
  - tests/test_fe_mundlak.py  (Mundlak device)
  - tests/test_duck2sls.py    (2SLS)
"""

import os
import sys
import tempfile
import duckdb
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import pyfixest as pf
from duckreg import duckreg

from duckreg.core.vcov import (
    SSCConfig,
    VcovContext,
    compute_iid_vcov,
    compute_hetero_vcov,
    compute_cluster_vcov,
    compute_cluster_scores,
)
from duckreg.core.linalg import safe_solve, safe_inv
from duckreg.core.fitters import (
    NumpyFitter,
    DuckDBFitter,
    compute_vcov_dispatch,
)

FITTERS = ["numpy", "duckdb"]


# ============================================================================
# A. Helpers
# ============================================================================

def ols(X, y, weights=None):
    """Return theta, XtXinv, residuals, rss for WLS."""
    if weights is None:
        weights = np.ones(len(y))
    w = weights.flatten()
    sqw = np.sqrt(w).reshape(-1, 1)
    Xw = X * sqw
    yw = (y * sqw.flatten())
    XtX = Xw.T @ Xw
    Xty = Xw.T @ yw
    theta = safe_solve(XtX, Xty)
    XtXinv = safe_inv(XtX, use_pinv=True)
    residuals = y.flatten() - X @ theta
    rss = float((residuals ** 2 * w).sum())
    return theta, XtXinv, residuals, rss


def scores(X, residuals, weights=None):
    """X * uhat (optionally weight-scaled)."""
    if weights is None:
        weights = np.ones(len(residuals))
    return X * (residuals * weights).reshape(-1, 1)


# ============================================================================
# B. Low-level fixtures
# ============================================================================

@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def simple(rng):
    n = 500
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    y = X @ np.array([0.5, 1.0, 2.0, 0.5]) + rng.standard_normal(n) * 0.5
    w = np.ones(n)
    return dict(X=X, y=y, w=w, n=n, k=4)


@pytest.fixture(scope="module")
def weighted(rng):
    n = 500
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    y = X @ np.array([0.5, 1.0, 2.0, 0.5]) + rng.standard_normal(n) * 0.5
    w = rng.integers(1, 6, n).astype(float)
    return dict(X=X, y=y, w=w, n=n, k=4, nobs=int(w.sum()))


@pytest.fixture(scope="module")
def clustered(rng):
    G = 25
    sizes = rng.integers(15, 35, G)
    n = int(sizes.sum())
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    cluster_ids = np.repeat(np.arange(G), sizes)
    fe = np.repeat(rng.standard_normal(G) * 0.3, sizes)
    y = X @ np.array([0.5, 1.0, 2.0, 0.5]) + fe + rng.standard_normal(n) * 0.5
    w = np.ones(n)
    return dict(X=X, y=y, w=w, n=n, k=4, G=G, cluster_ids=cluster_ids)


@pytest.fixture(scope="module")
def heteroskedastic(rng):
    n = 500
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    sd = 0.3 * (1 + np.abs(X[:, 1]))
    y = X @ np.array([0.5, 1.0, 2.0, 0.5]) + rng.standard_normal(n) * sd
    w = np.ones(n)
    return dict(X=X, y=y, w=w, n=n, k=4)


@pytest.fixture(scope="module")
def with_fe(rng):
    """Simulate panel with 2 absorbed FE dimensions (unit + time)."""
    n_units, n_periods = 30, 20
    n = n_units * n_periods
    unit = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)
    X = np.column_stack([
        np.ones(n),
        rng.standard_normal((n, 2)),
    ])
    unit_fe = np.repeat(rng.standard_normal(n_units), n_periods)
    time_fe = np.tile(rng.standard_normal(n_periods), n_units)
    y = X @ np.array([0.0, 1.0, 2.0]) + unit_fe + time_fe + rng.standard_normal(n) * 0.4
    w = np.ones(n)
    kfe = n_units + n_periods   # absorbed levels (approximate for test)
    nfe = 2
    return dict(X=X, y=y, w=w, n=n, k=3, kfe=kfe, nfe=nfe, unit=unit, time=time)


@pytest.fixture(scope="module")
def duckdb_conn():
    """In-memory DuckDB connection with a small regression table."""
    rng_ = np.random.default_rng(0)
    n = 300
    X1 = rng_.standard_normal(n)
    X2 = rng_.standard_normal(n)
    y = 1.0 + 1.5 * X1 + 0.8 * X2 + rng_.standard_normal(n) * 0.5
    cluster = (np.arange(n) % 15).astype(float)
    cnt = np.ones(n)
    df = pd.DataFrame({'x1': X1, 'x2': X2, 'y': y,
                       'cluster': cluster, 'count': cnt,
                       'sumy': y, 'sumysq': y**2})
    conn = duckdb.connect()
    conn.register('reg_table', df)
    yield conn
    conn.close()


# ============================================================================
# C. Panel fixtures
# ============================================================================

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
def balanced_df():
    return _make_balanced_panel()


@pytest.fixture(scope="module")
def balanced_path(balanced_df):
    path = _write_parquet(balanced_df)
    yield path
    _cleanup(path)


# ============================================================================
# D. TestPooledCoef — coefficient / SE parity vs pyfixest
# ============================================================================

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


def _duckreg(formula, path, vcov, fitter):
    return duckreg(formula=formula, data=path, seed=42,
                   round_strata=5, se_method=vcov, fitter=fitter)


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


def check_pooled(df, path, *, fitter, has_iv, vcov):
    f_pf = ("modis_median ~ exog_control | ntl_harm ~ rainfall"
            if has_iv else "modis_median ~ ntl_harm + exog_control")
    fit_pf = pf.feols(f_pf, df, vcov=_pf_vcov(vcov))

    if has_iv:
        f_dr = "modis_median ~ exog_control | 0 | (ntl_harm ~ rainfall)"
    else:
        f_dr = "modis_median ~ ntl_harm + exog_control"

    m = _duckreg(f_dr, path, _dr_se(vcov), fitter)
    pf_coef, pf_se = _pf_coef_se(fit_pf, "ntl_harm", has_iv)
    dr_coef, dr_se = _dr_coef_se(m)

    _assert_close(
        dr_coef, pf_coef, dr_se, pf_se,
        coef_rtol=1e-3 if has_iv else 1e-4,
        se_rtol=1e-2   if has_iv else 5e-3,
        label=f"pooled iv={has_iv} fitter={fitter} vcov={vcov}",
    )


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


# ============================================================================
# E. TestVcovDispatch
# ============================================================================

class TestVcovDispatch:

    def _run(self, data, vcov_type, cluster_ids=None, ssc_dict=None):
        X, y, w = data['X'], data['y'], data['w']
        theta, XtXinv, resid, rss = ols(X, y)
        ssc_dict = ssc_dict or {'kadj': True, 'kfixef': 'full', 'Gadj': True, 'Gdf': 'conventional'}
        vcov, meta, agg = compute_vcov_dispatch(
            X=X, y=y.reshape(-1, 1), weights=w,
            coefficients=theta, residuals=resid,
            XtXinv=XtXinv, vcov_type=vcov_type,
            cluster_ids=cluster_ids, ssc_dict=ssc_dict)
        return vcov, meta, agg

    def test_dispatch_iid(self, simple):
        vcov, meta, _ = self._run(simple, 'iid')
        assert meta['vcov_type'] == 'iid'
        assert vcov.shape == (4, 4)
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)

    def test_dispatch_hc1(self, heteroskedastic):
        vcov, meta, _ = self._run(heteroskedastic, 'HC1')
        assert meta['vcov_type'] == 'hetero'
        assert vcov.shape == (4, 4)

    def test_dispatch_hetero(self, heteroskedastic):
        vcov, meta, _ = self._run(heteroskedastic, 'hetero')
        assert meta['vcov_type'] == 'hetero'

    def test_dispatch_hc2(self, simple):
        vcov, meta, _ = self._run(simple, 'HC2')
        assert meta['vcov_type_detail'] == 'HC2'

    def test_dispatch_cluster(self, clustered):
        vcov, meta, _ = self._run(clustered, 'HC1',
                                   cluster_ids=clustered['cluster_ids'])
        assert meta['vcov_type'] == 'cluster'
        assert meta['n_clusters'] == clustered['G']

    def test_dispatch_iid_stores_rss(self, simple):
        _, _, agg = self._run(simple, 'iid')
        assert 'rss' in agg
        assert agg['rss'] > 0

    def test_dispatch_cluster_stores_cluster_scores(self, clustered):
        _, _, agg = self._run(clustered, 'HC1',
                               cluster_ids=clustered['cluster_ids'])
        assert 'cluster_scores' in agg
        assert agg['cluster_scores'].shape[0] == clustered['G']


# ============================================================================
# F. TestNumpyFitterVcov
# ============================================================================

class TestNumpyFitterVcov:

    def _fit_vcov(self, data, vcov_type, cluster_ids=None, ssc_dict=None):
        X, y, w = data['X'], data['y'], data['w']
        ssc_dict = ssc_dict or {'kadj': True, 'kfixef': 'full', 'Gadj': True, 'Gdf': 'conventional'}
        fitter = NumpyFitter()
        result = fitter.fit(X, y, w)
        vcov, meta, agg = fitter.fit_vcov(
            X=X, y=y, weights=w,
            coefficients=result.coefficients,
            cluster_ids=cluster_ids,
            vcov_type=vcov_type,
            ssc_dict=ssc_dict,
            existing_result=result,
        )
        return vcov, meta, result

    def test_iid(self, simple):
        vcov, meta, result = self._fit_vcov(simple, 'iid')
        assert vcov.shape == (4, 4)
        assert meta['vcov_type'] == 'iid'
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)

    def test_hc1(self, heteroskedastic):
        vcov, meta, result = self._fit_vcov(heteroskedastic, 'HC1')
        assert meta['vcov_type'] == 'hetero'

    def test_cluster(self, clustered):
        vcov, meta, result = self._fit_vcov(
            clustered, 'HC1', cluster_ids=clustered['cluster_ids'])
        assert meta['vcov_type'] == 'cluster'
        assert meta['n_clusters'] == clustered['G']

    def test_iid_with_fe_ssc(self, with_fe):
        X, y, w = with_fe['X'], with_fe['y'], with_fe['w']
        kfe, nfe = with_fe['kfe'], with_fe['nfe']
        ssc_dict = {'kadj': True, 'kfixef': 'nonnested', 'Gadj': True, 'Gdf': 'conventional'}
        fitter = NumpyFitter()
        result = fitter.fit(X, y, w)
        vcov, meta, _ = fitter.fit_vcov(
            X=X, y=y, weights=w, coefficients=result.coefficients,
            vcov_type='iid', ssc_dict=ssc_dict, kfe=kfe, nfe=nfe,
            existing_result=result,
        )
        assert vcov.shape == (with_fe['k'], with_fe['k'])
        assert meta['vcov_type'] == 'iid'

    def test_se_positive(self, simple):
        vcov, _, _ = self._fit_vcov(simple, 'HC1')
        assert np.all(np.diag(vcov) > 0)


# ============================================================================
# G. TestDuckDBFitterVcov
# ============================================================================

class TestDuckDBFitterVcov:

    def test_iid_duckdb(self, duckdb_conn):
        fitter = DuckDBFitter(conn=duckdb_conn)
        result = fitter.fit(
            table_name='reg_table', xcols=['x1', 'x2'],
            ycol='sumy', weightcol='count', add_intercept=True)
        vcov, meta, _ = fitter.fit_vcov(
            table_name='reg_table', xcols=['x1', 'x2'],
            ycol='sumy', weightcol='count', add_intercept=True,
            coefficients=result.coefficients,
            vcov_type='iid',
            ssc_dict={'kadj': True, 'kfixef': 'full', 'Gadj': True, 'Gdf': 'conventional'},
            existing_result=result,
        )
        assert vcov.shape == (3, 3)
        assert meta['vcov_type'] == 'iid'
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)

    def test_hc1_duckdb(self, duckdb_conn):
        fitter = DuckDBFitter(conn=duckdb_conn)
        result = fitter.fit(
            table_name='reg_table', xcols=['x1', 'x2'],
            ycol='sumy', weightcol='count', add_intercept=True)
        vcov, meta, _ = fitter.fit_vcov(
            table_name='reg_table', xcols=['x1', 'x2'],
            ycol='sumy', weightcol='count', add_intercept=True,
            coefficients=result.coefficients,
            vcov_type='HC1',
            ssc_dict={'kadj': True, 'kfixef': 'full', 'Gadj': True, 'Gdf': 'conventional'},
            existing_result=result,
        )
        assert meta['vcov_type'] == 'hetero'
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)

    def test_cluster_duckdb(self, duckdb_conn):
        fitter = DuckDBFitter(conn=duckdb_conn)
        result = fitter.fit(
            table_name='reg_table', xcols=['x1', 'x2'],
            ycol='sumy', weightcol='count', add_intercept=True,
            cluster_col='cluster')
        vcov, meta, _ = fitter.fit_vcov(
            table_name='reg_table', xcols=['x1', 'x2'],
            ycol='sumy', weightcol='count', add_intercept=True,
            coefficients=result.coefficients,
            cluster_col='cluster',
            vcov_type='CRV1',
            ssc_dict={'kadj': True, 'kfixef': 'full', 'Gadj': True, 'Gdf': 'conventional'},
            existing_result=result,
        )
        assert meta['vcov_type'] == 'cluster'
        assert meta['n_clusters'] == 15

    def test_iid_equals_numpy(self, duckdb_conn, rng):
        """DuckDB fitter and numpy fitter must agree on IID SE for the same data."""
        n = 300
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        y = X @ np.array([1.0, 1.5, 0.8]) + rng.standard_normal(n) * 0.5
        w = np.ones(n)
        ssc_dict = {'kadj': True, 'kfixef': 'none', 'Gadj': False, 'Gdf': 'conventional'}

        np_fitter = NumpyFitter()
        np_result = np_fitter.fit(X, y, w)
        np_vcov, _, _ = np_fitter.fit_vcov(
            X=X, y=y, weights=w,
            coefficients=np_result.coefficients,
            vcov_type='iid', ssc_dict=ssc_dict, existing_result=np_result)

        df = pd.DataFrame({'x1': X[:, 1], 'x2': X[:, 2],
                           'sumy': y, 'count': w, 'sumysq': y**2})
        conn2 = duckdb.connect()
        conn2.register('t', df)
        db_fitter = DuckDBFitter(conn=conn2)
        db_result = db_fitter.fit('t', ['x1', 'x2'], 'sumy', 'count', add_intercept=True)
        db_vcov, _, _ = db_fitter.fit_vcov(
            't', ['x1', 'x2'], 'sumy', 'count', add_intercept=True,
            coefficients=db_result.coefficients,
            vcov_type='iid', ssc_dict=ssc_dict, existing_result=db_result)
        conn2.close()

        np.testing.assert_allclose(
            np.sqrt(np.diag(np_vcov)),
            np.sqrt(np.diag(db_vcov)),
            rtol=1e-5,
            err_msg="DuckDB and numpy IID SEs diverge",
        )


# ============================================================================
# H. TestVcovVsPyfixest
# ============================================================================

class TestVcovVsPyfixest:
    """Numerical parity with pyfixest.feols for all SE types.

    Uses pyfixest as an oracle — runs feols() on the same data and compares
    standard errors returned by duckreg compute functions.
    """

    @pytest.fixture(autouse=True)
    def _pyfixest(self):
        try:
            from pyfixest.estimation import feols
            self._feols = feols
        except ImportError:
            pytest.skip("pyfixest not available")

    def _pyfixest_se(self, df, formula, vcov):
        fit = self._feols(formula, data=df, vcov=vcov)
        return fit.se()

    # ── IID ────────────────────────────────────────────────────────────────

    def test_iid_parity(self, simple, rng):
        X, y, n, k = simple['X'], simple['y'], simple['n'], simple['k']
        df = pd.DataFrame(X[:, 1:], columns=['x1', 'x2', 'x3'])
        df['y'] = y
        theta, XtXinv, resid, rss = ols(X, y)

        ctx = VcovContext(N=n, k=k)
        cfg = SSCConfig.from_dict({'kadj': True, 'kfixef': 'none', 'Gadj': True, 'Gdf': 'conventional'})
        vcov, _ = compute_iid_vcov(XtXinv, rss, ctx, cfg)
        our_se = np.sqrt(np.diag(vcov))[1:]   # drop intercept

        ref_se = self._pyfixest_se(df, 'y ~ x1 + x2 + x3', vcov='iid').values[1:]
        np.testing.assert_allclose(our_se, ref_se, rtol=1e-4,
                                   err_msg="IID SE mismatch vs pyfixest")

    # ── HC1 ────────────────────────────────────────────────────────────────

    def test_hc1_parity(self, heteroskedastic, rng):
        X, y, n, k = heteroskedastic['X'], heteroskedastic['y'], heteroskedastic['n'], heteroskedastic['k']
        df = pd.DataFrame(X[:, 1:], columns=['x1', 'x2', 'x3'])
        df['y'] = y
        theta, XtXinv, resid, _ = ols(X, y)
        sc = X * resid.reshape(-1, 1)

        cfg = SSCConfig.from_dict({'kadj': True, 'kfixef': 'full', 'Gadj': True, 'Gdf': 'conventional'})
        vcov, _ = compute_hetero_vcov(
            bread=XtXinv, scores=sc, vcov_type_detail='HC1',
            ssc_config=cfg, N=n, k=k)
        our_se = np.sqrt(np.diag(vcov))[1:]

        ref_se = self._pyfixest_se(df, 'y ~ x1 + x2 + x3', vcov='HC1').values[1:]
        np.testing.assert_allclose(our_se, ref_se, rtol=1e-4,
                                   err_msg="HC1 SE mismatch vs pyfixest")

    # ── CRV1 ───────────────────────────────────────────────────────────────

    def test_crv1_parity(self, clustered, rng):
        X, y, n, k = clustered['X'], clustered['y'], clustered['n'], clustered['k']
        cids = clustered['cluster_ids']
        df = pd.DataFrame(X[:, 1:], columns=['x1', 'x2', 'x3'])
        df['y'] = y
        df['cluster'] = cids.astype(str)

        theta, XtXinv, resid, _ = ols(X, y)
        sc = X * resid.reshape(-1, 1)
        cs, G = compute_cluster_scores(sc, cids)
        ctx = VcovContext(N=n, k=k)
        cfg = SSCConfig.from_dict({'kadj': True, 'kfixef': 'full', 'Gadj': True, 'Gdf': 'conventional'})
        vcov, _ = compute_cluster_vcov(XtXinv, cs, ctx, G, cfg)
        our_se = np.sqrt(np.diag(vcov))[1:]

        ref_se = self._pyfixest_se(df, 'y ~ x1 + x2 + x3',
                                   vcov={'CRV1': 'cluster'}).values[1:]
        np.testing.assert_allclose(our_se, ref_se, rtol=1e-3,
                                   err_msg="CRV1 SE mismatch vs pyfixest")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
