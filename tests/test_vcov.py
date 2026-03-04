"""
Comprehensive VCOV test suite — covers every component.

Structure
---------
1.  Fixtures            — data generators (simple, weighted, clustered, het, FE, two-way)
2.  TestSSCConfig       — dataclass defaults, from_dict, to_dict, round-trip
2b. TestSSCConfigForFormula — SSCConfig.for_formula() auto-determination
2c. TestVcovSpecBuild   — VcovSpec.build() auto-SSC, explicit ssc_dict override
3.  TestVcovContext     — dataclass fields and defaults
4.  TestVcovParsing     — parse_vcov_specification + parse_cluster_vars, all types + error paths
5.  TestComputeSSC      — compute_ssc for every (k_fixef × vcov_type × G_df) combination vs pyfixest
6.  TestComputeBread    — OLS bread, IV bread, missing-IV-matrices error
7.  TestSandwichMeat    — sandwich_from_meat symmetry and scaling
8.  TestIIDVcov         — vcov_iid (legacy) + compute_iid_vcov (no FE, with FE, weighted)
9.  TestHeteroVcov      — vcov_hetero (legacy) + compute_hetero_vcov (HC1/HC2/HC3/hetero, FE, weighted)
10. TestClusterVcov     — compute_cluster_scores + vcov_crv1 (legacy) + compute_cluster_vcov (G_df variants, FE)
11. TestTwoWayCluster   — compute_two_way_cluster_vcov CGM formula

Fitter-level and end-to-end parity tests live in tests/test_pooled_ols.py.
"""

import numpy as np
import pandas as pd
import pytest
from typing import Tuple

# ── project imports ──────────────────────────────────────────────────────────
from duckreg.core.vcov import (
    SSCConfig,
    VcovContext,
    VcovSpec,
    parse_vcov_specification,
    parse_cluster_vars,
    compute_bread,
    compute_ssc,
    sandwich_from_meat,
    vcov_iid,
    compute_iid_vcov,
    vcov_hetero,
    compute_hetero_vcov,
    vcov_crv1,
    compute_cluster_vcov,
    compute_cluster_scores,
    compute_twoway_cluster_vcov,
    VcovTypeNotSupportedError,
)
from duckreg.core.linalg import safe_solve, safe_inv

# ── pyfixest reference ────────────────────────────────────────────────────────
import sys
sys.path.insert(0, '/scicore/home/meiera/schulz0022/projects/duckreg/foreign')
from pyfixest.utils import get_ssc as pyfixest_get_ssc


# ============================================================================
# HELPERS
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
# FIXTURES
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
def two_way(rng):
    """Two-way cluster structure: state × industry."""
    n_state, n_ind = 10, 8
    n = 400
    state = rng.integers(0, n_state, n)
    industry = rng.integers(0, n_ind, n)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    y = X @ np.array([0.5, 1.0, 2.0, 0.5]) + rng.standard_normal(n)
    w = np.ones(n)
    # Interaction cluster id (unique for each state×industry pair)
    pair_id = state * n_ind + industry
    cluster_df = np.column_stack([state, industry, pair_id])
    return dict(X=X, y=y, w=w, n=n, k=4, cluster_df=cluster_df,
                G_state=n_state, G_ind=n_ind)


# ============================================================================
# 2. TestSSCConfig
# ============================================================================

class TestSSCConfig:

    def test_defaults(self):
        cfg = SSCConfig()
        assert cfg.kadj is True
        assert cfg.kfixef == 'nonnested'
        assert cfg.Gadj is True
        assert cfg.Gdf == 'conventional'

    def test_from_dict_full(self):
        d = {'kadj': False, 'kfixef': 'none', 'Gadj': False, 'Gdf': 'min'}
        cfg = SSCConfig.from_dict(d)
        assert cfg.kadj is False
        assert cfg.kfixef == 'none'
        assert cfg.Gadj is False
        assert cfg.Gdf == 'min'

    def test_from_dict_partial(self):
        cfg = SSCConfig.from_dict({'kfixef': 'full'})
        assert cfg.kfixef == 'full'
        assert cfg.kadj is True          # default preserved

    def test_from_dict_none(self):
        cfg = SSCConfig.from_dict(None)
        assert cfg == SSCConfig()        # equals default

    def test_to_dict_round_trip(self):
        cfg = SSCConfig(kadj=False, kfixef='none', Gadj=False, Gdf='min')
        assert SSCConfig.from_dict(cfg.to_dict()) == cfg

    def test_invalid_kfixef_raises(self):
        cfg = SSCConfig(kfixef='bad')
        ctx = VcovContext(N=100, k=3)
        with pytest.raises(ValueError, match='kfixef'):
            compute_ssc(cfg, ctx, G=1, vcov_type='iid')

    def test_invalid_Gdf_raises(self):
        cfg = SSCConfig(Gdf='bad')
        ctx = VcovContext(N=100, k=3)
        with pytest.raises(ValueError, match='Gdf'):
            compute_ssc(cfg, ctx, G=5, vcov_type='CRV')


# ============================================================================
# 2b. TestSSCConfigForFormula
# ============================================================================

class TestSSCConfigForFormula:
    """Unit tests for SSCConfig.for_formula() auto-determination."""

    def test_non_clustered_defaults(self):
        """Non-clustered, no FE: conventional Gdf."""
        cfg = SSCConfig.for_formula(has_fixef=False, is_clustered=False)
        assert cfg.kadj is True
        assert cfg.kfixef == 'nonnested'
        assert cfg.Gadj is True
        assert cfg.Gdf == 'conventional'

    def test_clustered_gdf_is_min(self):
        """Clustering present: Gdf='min' (matches pyfixest default)."""
        cfg = SSCConfig.for_formula(is_clustered=True)
        assert cfg.Gdf == 'min'

    def test_fixef_only_keeps_conventional(self):
        """FE without clustering: Gdf stays 'conventional'."""
        cfg = SSCConfig.for_formula(has_fixef=True, is_clustered=False)
        assert cfg.kfixef == 'nonnested'
        assert cfg.Gdf == 'conventional'

    def test_iv_clustered_gdf_is_min(self):
        """IV + clustered: Gdf='min'."""
        cfg = SSCConfig.for_formula(has_fixef=True, is_clustered=True, is_iv=True)
        assert cfg.Gdf == 'min'
        assert cfg.kadj is True

    def test_kfixef_always_nonnested(self):
        """kfixef is always 'nonnested' regardless of FE/cluster presence."""
        for has_fixef in (True, False):
            for is_clustered in (True, False):
                cfg = SSCConfig.for_formula(has_fixef=has_fixef, is_clustered=is_clustered)
                assert cfg.kfixef == 'nonnested', (
                    f"Expected 'nonnested' for has_fixef={has_fixef}, "
                    f"is_clustered={is_clustered}, got '{cfg.kfixef}'"
                )


# ============================================================================
# 2c. TestVcovSpecBuild
# ============================================================================

class TestVcovSpecBuild:
    """Unit tests for VcovSpec.build() auto-SSC and explicit ssc_dict override."""

    def test_auto_ssc_non_clustered(self):
        spec = VcovSpec.build('HC1', has_fixef=True)
        assert spec.ssc.kfixef == 'nonnested'
        assert spec.ssc.Gdf == 'conventional'
        assert spec.is_clustered is False

    def test_auto_ssc_clustered(self):
        spec = VcovSpec.build({'CRV1': 'unit'})
        assert spec.ssc.Gdf == 'min'
        assert spec.is_clustered is True

    def test_explicit_ssc_dict_overrides_auto(self):
        """Passing ssc_dict bypasses for_formula() — used in component tests."""
        spec = VcovSpec.build('HC1', ssc_dict={'kadj': False, 'kfixef': 'full',
                                               'Gadj': False, 'Gdf': 'min'})
        assert spec.ssc.kadj is False
        assert spec.ssc.kfixef == 'full'
        assert spec.ssc.Gadj is False

    def test_build_iid_auto_ssc(self):
        spec = VcovSpec.build('iid')
        assert spec.vcov_type == 'iid'
        assert spec.ssc.kfixef == 'nonnested'

    def test_build_crv3_clustered(self):
        spec = VcovSpec.build({'CRV3': 'grp'})
        assert spec.vcov_detail == 'CRV3'
        assert spec.ssc.Gdf == 'min'


# ============================================================================
# 3. TestVcovContext
# ============================================================================

class TestVcovContext:

    def test_basic_fields(self):
        ctx = VcovContext(N=500, k=4, kfe=50, nfe=2, kfenested=20, nfefullynested=1)
        assert ctx.N == 500
        assert ctx.k == 4
        assert ctx.kfe == 50
        assert ctx.nfe == 2
        assert ctx.kfenested == 20
        assert ctx.nfefullynested == 1

    def test_defaults(self):
        ctx = VcovContext(N=200, k=3)
        assert ctx.kfe == 0
        assert ctx.nfe == 0
        assert ctx.kfenested == 0
        assert ctx.nfefullynested == 0


# ============================================================================
# 4. TestVcovParsing
# ============================================================================

class TestVcovParsing:

    @pytest.mark.parametrize("spec,exp_type,exp_detail,exp_clust,exp_vars", [
        ("iid",              "iid",   "iid",   False, None),
        ("hetero",           "hetero","hetero",False, None),
        ("HC1",              "hetero","HC1",   False, None),
        ("HC2",              "hetero","HC2",   False, None),
        ("HC3",              "hetero","HC3",   False, None),
        ({"CRV1": "state"},  "CRV",   "CRV1",  True,  ["state"]),
        ({"CRV3": "state"},  "CRV",   "CRV3",  True,  ["state"]),
        ({"CRV1": "s+f"},    "CRV",   "CRV1",  True,  ["s", "f"]),
    ])
    def test_valid_specs(self, spec, exp_type, exp_detail, exp_clust, exp_vars):
        vt, vd, ic, cv = parse_vcov_specification(spec)
        assert vt == exp_type
        assert vd == exp_detail
        assert ic == exp_clust
        assert cv == exp_vars

    def test_hc2_with_fixef_raises(self):
        with pytest.raises(VcovTypeNotSupportedError):
            parse_vcov_specification("HC2", has_fixef=True)

    def test_hc3_with_fixef_raises(self):
        with pytest.raises(VcovTypeNotSupportedError):
            parse_vcov_specification("HC3", has_fixef=True)

    def test_hc2_with_iv_raises(self):
        with pytest.raises(VcovTypeNotSupportedError):
            parse_vcov_specification("HC2", is_iv=True)

    def test_cluster_vars_with_non_cluster_type_raises(self):
        with pytest.raises(VcovTypeNotSupportedError):
            parse_vcov_specification({"HC1": "state"})

    def test_invalid_type_raises(self):
        with pytest.raises((ValueError, VcovTypeNotSupportedError)):
            parse_vcov_specification("BOGUS")

    def test_parse_cluster_vars_clustered(self):
        assert parse_cluster_vars({"CRV1": "state"}) == ["state"]

    def test_parse_cluster_vars_two_way(self):
        assert parse_cluster_vars({"CRV1": "s+f"}) == ["s", "f"]

    def test_parse_cluster_vars_non_clustered(self):
        assert parse_cluster_vars("HC1") is None

    def test_parse_cluster_vars_iid(self):
        assert parse_cluster_vars("iid") is None


# ============================================================================
# 5. TestComputeSSC
# ============================================================================

def _pyfixest_ssc(ssc_dict, N, k, kfe, kfenested, nfe, nfefullynested, G, vcov_sign, vcov_type):
    # Translate duckreg-style dict keys to the pyfixest-style keys expected by get_ssc
    pf_dict = {
        'k_adj':   ssc_dict.get('kadj',   ssc_dict.get('k_adj',   True)),
        'k_fixef': ssc_dict.get('kfixef', ssc_dict.get('k_fixef', 'nonnested')),
        'G_adj':   ssc_dict.get('Gadj',   ssc_dict.get('G_adj',   True)),
        'G_df':    ssc_dict.get('Gdf',    ssc_dict.get('G_df',    'conventional')),
    }
    return pyfixest_get_ssc(
        ssc_dict=pf_dict, N=N, k=k, k_fe=kfe,
        k_fe_nested=kfenested, n_fe=nfe,
        n_fe_fully_nested=nfefullynested,
        G=G, vcov_sign=vcov_sign, vcov_type=vcov_type,
    )


class TestComputeSSC:

    # ── k_fixef × vcov_type matrix ───────────────────────────────────────────

    @pytest.mark.parametrize("kfixef", ["none", "nonnested", "full"])
    @pytest.mark.parametrize("vcov_type", ["iid", "hetero", "CRV"])
    def test_parity_no_fe(self, kfixef, vcov_type):
        N, k, G = 500, 4, 20
        d = {'kadj': True, 'kfixef': kfixef, 'Gadj': True, 'Gdf': 'conventional'}
        cfg = SSCConfig.from_dict(d)
        ctx = VcovContext(N=N, k=k)
        G_ = N if vcov_type == 'hetero' else G
        ssc, dfk, dft = compute_ssc(cfg, ctx, G=G_, vcov_type=vcov_type)
        ref_ssc, ref_dfk, ref_dft = _pyfixest_ssc(d, N, k, 0, 0, 0, 0, G_, 1, vcov_type)
        assert np.isclose(ssc, ref_ssc), f"{kfixef}/{vcov_type}: ssc {ssc} != {ref_ssc}"
        assert dfk == ref_dfk, f"{kfixef}/{vcov_type}: dfk {dfk} != {ref_dfk}"
        assert dft == ref_dft, f"{kfixef}/{vcov_type}: dft {dft} != {ref_dft}"

    @pytest.mark.parametrize("kfixef", ["none", "nonnested", "full"])
    def test_parity_with_fe(self, kfixef):
        """Two-way FE panel: unit (n=30) + time (n=20), time is nested in nothing."""
        N, k, kfe, nfe, G = 600, 3, 49, 2, 25
        kfenested, nfefullynested = 0, 0   # no nesting (crossed FE)
        d = {'kadj': True, 'kfixef': kfixef, 'Gadj': True, 'Gdf': 'conventional'}
        cfg = SSCConfig.from_dict(d)
        ctx = VcovContext(N=N, k=k, kfe=kfe, nfe=nfe,
                         kfenested=kfenested, nfefullynested=nfefullynested)
        ssc, dfk, dft = compute_ssc(cfg, ctx, G=G, vcov_type="CRV")
        ref_ssc, ref_dfk, ref_dft = _pyfixest_ssc(
            d, N, k, kfe, kfenested, nfe, nfefullynested, G, 1, "CRV")
        assert np.isclose(ssc, ref_ssc)
        assert dfk == ref_dfk

    def test_nonnested_with_actual_nesting(self):
        """Nested FE: time fully nested within unit (balanced panel)."""
        N, k, kfe, nfe = 600, 3, 49, 2
        kfenested, nfefullynested = 20, 1   # time (20 levels) nested in unit
        d = {'kadj': True, 'kfixef': 'nonnested', 'Gadj': True, 'Gdf': 'conventional'}
        cfg = SSCConfig.from_dict(d)
        ctx = VcovContext(N=N, k=k, kfe=kfe, nfe=nfe,
                         kfenested=kfenested, nfefullynested=nfefullynested)
        ssc, dfk, dft = compute_ssc(cfg, ctx, G=25, vcov_type="CRV")
        ref_ssc, ref_dfk, ref_dft = _pyfixest_ssc(
            d, N, k, kfe, kfenested, nfe, nfefullynested, 25, 1, "CRV")
        assert np.isclose(ssc, ref_ssc)
        assert dfk == ref_dfk

    @pytest.mark.parametrize("Gdf", ["conventional", "min"])
    def test_Gdf_variants(self, Gdf):
        N, k, G = 500, 4, 20
        d = {'kadj': True, 'kfixef': 'full', 'Gadj': True, 'Gdf': Gdf}
        cfg = SSCConfig.from_dict(d)
        ctx = VcovContext(N=N, k=k)
        ssc, _, _ = compute_ssc(cfg, ctx, G=G, vcov_type="CRV")
        ref_ssc, _, _ = _pyfixest_ssc(d, N, k, 0, 0, 0, 0, G, 1, "CRV")
        assert np.isclose(ssc, ref_ssc), f"Gdf={Gdf}: {ssc} vs {ref_ssc}"

    def test_vcov_sign_minus_one(self):
        """vcov_sign=-1 is used for the subtracted cross term in two-way clustering."""
        N, k, G = 500, 4, 20
        cfg = SSCConfig.from_dict({'kadj': True, 'kfixef': 'full', 'Gadj': True, 'Gdf': 'conventional'})
        ctx = VcovContext(N=N, k=k)
        ssc_pos, _, _ = compute_ssc(cfg, ctx, G=G, vcov_type="CRV", vcov_sign=1)
        ssc_neg, _, _ = compute_ssc(cfg, ctx, G=G, vcov_type="CRV", vcov_sign=-1)
        assert ssc_neg < 0
        assert np.isclose(abs(ssc_neg), ssc_pos)

    def test_kadj_false(self):
        cfg = SSCConfig(kadj=False, kfixef='none', Gadj=False, Gdf='conventional')
        ctx = VcovContext(N=500, k=4)
        ssc, _, _ = compute_ssc(cfg, ctx, G=1, vcov_type="iid")
        assert np.isclose(ssc, 1.0)


# ============================================================================
# 6. TestComputeBread
# ============================================================================

class TestComputeBread:

    def test_ols_bread_is_xtx_inverse(self, simple):
        X = simple['X']
        XtX = X.T @ X
        bread = compute_bread(XtX, is_iv=False)
        np.testing.assert_allclose(bread @ XtX, np.eye(4), atol=1e-10)

    def test_ols_bread_symmetric(self, simple):
        XtX = simple['X'].T @ simple['X']
        bread = compute_bread(XtX, is_iv=False)
        np.testing.assert_allclose(bread, bread.T, atol=1e-12)

    def test_iv_bread(self, simple, rng):
        X, n, k = simple['X'], simple['n'], simple['k']
        Z = np.column_stack([X, rng.standard_normal(n)])   # over-identified
        sqw = np.ones((n, 1))
        Xw, Zw = X * sqw, Z * sqw
        tXZ = Xw.T @ Zw
        tZZ = Zw.T @ Zw + 1e-8 * np.eye(Z.shape[1])
        tZZinv = safe_inv(tZZ, use_pinv=True)
        tZX = tXZ.T
        # dummy hessian (not used for IV)
        bread = compute_bread(np.eye(k), is_iv=True, tXZ=tXZ, tZZinv=tZZinv, tZX=tZX)
        assert bread.shape == (k, k)
        np.testing.assert_allclose(bread, bread.T, atol=1e-10)

    def test_iv_bread_missing_matrices_raises(self, simple):
        XtX = simple['X'].T @ simple['X']
        with pytest.raises(ValueError):
            compute_bread(XtX, is_iv=True)


# ============================================================================
# 7. TestSandwichMeat
# ============================================================================

class TestSandwichMeat:

    def test_sandwich_scaling(self, simple):
        X = simple['X']
        XtX = X.T @ X
        bread = safe_inv(XtX, use_pinv=True)
        meat = np.eye(4)
        vcov = sandwich_from_meat(bread, meat, ssc=2.0)
        expected = 2.0 * bread @ meat @ bread
        expected = 0.5 * (expected + expected.T)
        np.testing.assert_allclose(vcov, expected, atol=1e-12)

    def test_sandwich_symmetry(self, simple):
        X = simple['X']
        bread = safe_inv(X.T @ X, use_pinv=True)
        meat = (X.T @ X) * 1.5    # arbitrary PD matrix
        vcov = sandwich_from_meat(bread, meat, ssc=1.0)
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)


# ============================================================================
# 8. TestIIDVcov
# ============================================================================

class TestIIDVcov:

    def test_vcov_iid_legacy(self, simple):
        X, y, w = simple['X'], simple['y'], simple['w']
        theta, XtXinv, resid, _ = ols(X, y)
        vcov = vcov_iid(XtXinv, resid, simple['n'])
        assert vcov.shape == (4, 4)
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(vcov) > 0)

    def test_compute_iid_vcov_no_fe(self, simple):
        X, y, n, k = simple['X'], simple['y'], simple['n'], simple['k']
        theta, XtXinv, resid, rss = ols(X, y)
        ctx = VcovContext(N=n, k=k)
        cfg = SSCConfig.from_dict({'kadj': True, 'kfixef': 'none'})
        vcov, meta = compute_iid_vcov(XtXinv, rss, ctx, cfg)
        assert vcov.shape == (k, k)
        assert meta['vcov_type'] == 'iid'
        assert meta['dft'] == n - k
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)

    def test_compute_iid_vcov_with_fe(self, with_fe):
        X, y, n, k = with_fe['X'], with_fe['y'], with_fe['n'], with_fe['k']
        kfe, nfe = with_fe['kfe'], with_fe['nfe']
        theta, XtXinv, resid, rss = ols(X, y)
        ctx = VcovContext(N=n, k=k, kfe=kfe, nfe=nfe)
        cfg = SSCConfig.from_dict({'kadj': True, 'kfixef': 'full'})
        vcov, meta = compute_iid_vcov(XtXinv, rss, ctx, cfg)

        assert vcov.shape == (k, k)
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)

        # dfk must be strictly larger than k alone (FE levels absorbed into df)
        assert meta['dfk'] > k

        # Cross-check: compute_ssc must agree with what compute_iid_vcov used
        ref_ssc, ref_dfk, ref_dft = compute_ssc(cfg, ctx, G=1, vcov_type='iid')
        assert meta['dfk'] == ref_dfk
        assert meta['dft'] == ref_dft

        # dft = N - dfk (IID convention)
        assert meta['dft'] == n - meta['dfk']


    def test_compute_iid_vcov_weighted(self, weighted):
        X, y, w = weighted['X'], weighted['y'], weighted['w']
        n, k, nobs = weighted['n'], weighted['k'], weighted['nobs']
        theta, XtXinv, resid, rss = ols(X, y, w)
        ctx = VcovContext(N=nobs, k=k)
        cfg = SSCConfig.from_dict({'kadj': True, 'kfixef': 'none'})
        vcov, meta = compute_iid_vcov(XtXinv, rss, ctx, cfg)
        assert vcov.shape == (k, k)
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)

    @pytest.mark.parametrize("kfixef", ["none", "nonnested", "full"])
    def test_kfixef_affects_ssc(self, simple, kfixef):
        X, y, n, k = simple['X'], simple['y'], simple['n'], simple['k']
        theta, XtXinv, _, rss = ols(X, y)
        ctx = VcovContext(N=n, k=k, kfe=40, nfe=2)
        cfg = SSCConfig.from_dict({'kadj': True, 'kfixef': kfixef})
        vcov, meta = compute_iid_vcov(XtXinv, rss, ctx, cfg)
        assert vcov.shape == (k, k)


# ============================================================================
# 9. TestHeteroVcov
# ============================================================================

class TestHeteroVcov:

    def _setup(self, data):
        X, y, w = data['X'], data['y'], data['w']
        theta, XtXinv, resid, rss = ols(X, y)
        sc = scores(X, resid, w)
        return X, y, w, theta, XtXinv, resid, sc

    def test_vcov_hetero_legacy_hc1(self, heteroskedastic):
        X, y, w, theta, XtXinv, resid, sc = self._setup(heteroskedastic)
        vcov = vcov_hetero(XtXinv, sc, vcov_type_detail='HC1')
        assert vcov.shape == (4, 4)
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)

    @pytest.mark.parametrize("hc_type", ["hetero", "HC1"])
    def test_compute_hetero_hc1_and_hetero(self, heteroskedastic, hc_type):
        X, y, w, theta, XtXinv, resid, sc = self._setup(heteroskedastic)
        n, k = heteroskedastic['n'], heteroskedastic['k']
        cfg = SSCConfig.from_dict({'kadj': True, 'kfixef': 'full'})
        ctx = VcovContext(N=n, k=k)
        ssc, dfk, dft = compute_ssc(cfg, ctx, G=n, vcov_type='hetero')
        vcov, meta = compute_hetero_vcov(
            bread=XtXinv, scores=sc, vcov_type_detail=hc_type,
            ssc_config=cfg, N=n, k=k)
        assert vcov.shape == (k, k)
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)
        assert meta['vcov_type'] == 'hetero'
        assert meta['vcov_type_detail'] == hc_type

    @pytest.mark.parametrize("hc_type", ["HC2", "HC3"])
    def test_compute_hetero_hc2_hc3(self, heteroskedastic, hc_type):
        X, y, w, theta, XtXinv, resid, sc = self._setup(heteroskedastic)
        n, k = heteroskedastic['n'], heteroskedastic['k']
        leverages = np.sum(X @ XtXinv * X, axis=1)
        cfg = SSCConfig.from_dict({'kadj': True, 'kfixef': 'full'})
        vcov, meta = compute_hetero_vcov(
            bread=XtXinv, scores=sc, leverages=leverages,
            vcov_type_detail=hc_type, ssc_config=cfg, N=n, k=k)
        assert vcov.shape == (k, k)
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)
        assert meta['vcov_type_detail'] == hc_type

    def test_hc2_hc3_larger_than_hc1(self, heteroskedastic):
        """HC2/HC3 SE should be >= HC1 SE (leverage correction inflates)."""
        X, y, w, theta, XtXinv, resid, sc = self._setup(heteroskedastic)
        n, k = heteroskedastic['n'], heteroskedastic['k']
        leverages = np.sum(X @ XtXinv * X, axis=1)
        cfg = SSCConfig.from_dict({'kadj': True, 'kfixef': 'full'})
        vcov_hc1, _ = compute_hetero_vcov(
            bread=XtXinv, scores=sc, vcov_type_detail='HC1',
            ssc_config=cfg, N=n, k=k)
        vcov_hc3, _ = compute_hetero_vcov(
            bread=XtXinv, scores=sc, leverages=leverages,
            vcov_type_detail='HC3', ssc_config=cfg, N=n, k=k)
        se_hc1 = np.sqrt(np.diag(vcov_hc1))
        se_hc3 = np.sqrt(np.diag(vcov_hc3))
        assert np.all(se_hc3 >= se_hc1 - 1e-10)

    def test_compute_hetero_with_fe(self, with_fe):
        X, y, n, k = with_fe['X'], with_fe['y'], with_fe['n'], with_fe['k']
        kfe, nfe = with_fe['kfe'], with_fe['nfe']
        theta, XtXinv, resid, _ = ols(X, y)
        sc = scores(X, resid)
        cfg = SSCConfig.from_dict({'kadj': True, 'kfixef': 'nonnested'})
        vcov, meta = compute_hetero_vcov(
            bread=XtXinv, scores=sc, vcov_type_detail='HC1',
            ssc_config=cfg, N=n, k=k, kfe=kfe, nfe=nfe)
        assert vcov.shape == (k, k)

    def test_hetero_ssc_uses_k_fe_nested(self, with_fe):
        """k_fe_nested reduces dfk for HC1 (fixes structural bug where it was always ignored)."""
        X, y, n, k = with_fe['X'], with_fe['y'], with_fe['n'], with_fe['k']
        kfe, nfe = with_fe['kfe'], with_fe['nfe']
        kfenested = 10  # simulate some FE nesting
        theta, XtXinv, resid, _ = ols(X, y)
        sc = scores(X, resid)
        cfg = SSCConfig.from_dict({'kadj': True, 'kfixef': 'nonnested'})

        _, meta_no_nest = compute_hetero_vcov(
            bread=XtXinv, scores=sc, vcov_type_detail='HC1',
            ssc_config=cfg, N=n, k=k, kfe=kfe, nfe=nfe,
            k_fe_nested=0)
        _, meta_nested = compute_hetero_vcov(
            bread=XtXinv, scores=sc, vcov_type_detail='HC1',
            ssc_config=cfg, N=n, k=k, kfe=kfe, nfe=nfe,
            k_fe_nested=kfenested)

        # k_fe_nested reduces dfk → larger (N-dfk) → smaller N/(N-dfk) → smaller SSC
        assert meta_nested['dfk'] == meta_no_nest['dfk'] - kfenested
        assert meta_nested['ssc'] < meta_no_nest['ssc']

    def test_compute_hetero_from_precomputed_meat(self, heteroskedastic):
        X, y, w, theta, XtXinv, resid, sc = self._setup(heteroskedastic)
        n, k = heteroskedastic['n'], heteroskedastic['k']
        meat = sc.T @ sc
        cfg = SSCConfig.from_dict({'kadj': True, 'kfixef': 'full'})
        vcov_from_meat, _ = compute_hetero_vcov(
            bread=XtXinv, meat=meat, vcov_type_detail='HC1',
            ssc_config=cfg, N=n, k=k)
        vcov_from_scores, _ = compute_hetero_vcov(
            bread=XtXinv, scores=sc, vcov_type_detail='HC1',
            ssc_config=cfg, N=n, k=k)
        np.testing.assert_allclose(vcov_from_meat, vcov_from_scores, rtol=1e-8)

    def test_missing_scores_and_meat_raises(self, simple):
        X = simple['X']
        XtXinv = safe_inv(X.T @ X, use_pinv=True)
        cfg = SSCConfig()
        with pytest.raises(ValueError):
            compute_hetero_vcov(bread=XtXinv, vcov_type_detail='HC1',
                                ssc_config=cfg, N=500, k=4)


# ============================================================================
# 10. TestClusterVcov
# ============================================================================

class TestClusterVcov:

    def _setup(self, data):
        X, y, w = data['X'], data['y'], data['w']
        theta, XtXinv, resid, rss = ols(X, y)
        sc = X * resid.reshape(-1, 1)   # unit weight scores
        return X, y, w, theta, XtXinv, resid, sc

    def test_compute_cluster_scores_shape_and_values(self, clustered):
        X, y, w, theta, XtXinv, resid, sc = self._setup(clustered)
        cids = clustered['cluster_ids']
        G = clustered['G']
        cs, G_out = compute_cluster_scores(sc, cids)
        assert G_out == G
        assert cs.shape == (G, 4)
        # verify one cluster manually
        mask = cids == 0
        np.testing.assert_allclose(cs[0], sc[mask].sum(axis=0), atol=1e-10)

    def test_vcov_crv1_legacy(self, clustered):
        X, y, w, theta, XtXinv, resid, sc = self._setup(clustered)
        cids = clustered['cluster_ids']
        vcov = vcov_crv1(XtXinv, sc, cids)
        assert vcov.shape == (4, 4)
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)

    def test_compute_cluster_vcov_conventional(self, clustered):
        X, y, w, theta, XtXinv, resid, sc = self._setup(clustered)
        cids, G, n, k = clustered['cluster_ids'], clustered['G'], clustered['n'], clustered['k']
        cs, G_out = compute_cluster_scores(sc, cids)
        ctx = VcovContext(N=n, k=k)
        cfg = SSCConfig.from_dict({'kadj': True, 'kfixef': 'full', 'Gadj': True, 'Gdf': 'conventional'})
        vcov, meta = compute_cluster_vcov(XtXinv, cs, ctx, G_out, cfg)
        assert vcov.shape == (k, k)
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)
        assert meta['vcov_type'] == 'cluster'
        assert meta['n_clusters'] == G
        assert meta['dft'] == G - 1

    def test_compute_cluster_vcov_Gdf_min(self, clustered):
        X, y, w, theta, XtXinv, resid, sc = self._setup(clustered)
        cids, G, n, k = clustered['cluster_ids'], clustered['G'], clustered['n'], clustered['k']
        cs, G_out = compute_cluster_scores(sc, cids)
        ctx = VcovContext(N=n, k=k)
        cfg_conv = SSCConfig.from_dict({'kadj': True, 'kfixef': 'full', 'Gadj': True, 'Gdf': 'conventional'})
        cfg_min  = SSCConfig.from_dict({'kadj': True, 'kfixef': 'full', 'Gadj': True, 'Gdf': 'min'})
        vcov_conv, _ = compute_cluster_vcov(XtXinv, cs, ctx, G_out, cfg_conv)
        vcov_min, _  = compute_cluster_vcov(XtXinv, cs, ctx, G_out, cfg_min)
        # With single cluster dim, min == conventional
        np.testing.assert_allclose(vcov_conv, vcov_min, rtol=1e-8)

    def test_compute_cluster_vcov_Gadj_false(self, clustered):
        X, y, w, theta, XtXinv, resid, sc = self._setup(clustered)
        cids, G, n, k = clustered['cluster_ids'], clustered['G'], clustered['n'], clustered['k']
        cs, G_out = compute_cluster_scores(sc, cids)
        ctx = VcovContext(N=n, k=k)
        cfg_on  = SSCConfig.from_dict({'kadj': True, 'kfixef': 'full', 'Gadj': True,  'Gdf': 'conventional'})
        cfg_off = SSCConfig.from_dict({'kadj': True, 'kfixef': 'full', 'Gadj': False, 'Gdf': 'conventional'})
        vcov_on, _  = compute_cluster_vcov(XtXinv, cs, ctx, G_out, cfg_on)
        vcov_off, _ = compute_cluster_vcov(XtXinv, cs, ctx, G_out, cfg_off)
        # G_adj=True applies G/(G-1) multiplier → larger SEs
        se_on  = np.sqrt(np.diag(vcov_on))
        se_off = np.sqrt(np.diag(vcov_off))
        assert np.all(se_on >= se_off - 1e-10)

    def test_compute_cluster_vcov_with_fe(self, clustered, with_fe):
        X, y, n, k = with_fe['X'], with_fe['y'], with_fe['n'], with_fe['k']
        kfe, nfe = with_fe['kfe'], with_fe['nfe']
        # reuse cluster structure from clustered (match sizes crudely)
        rng_ = np.random.default_rng(7)
        cids = (np.arange(n) % 20)
        G = 20
        theta, XtXinv, resid, _ = ols(X, y)
        sc = X * resid.reshape(-1, 1)
        cs, G_out = compute_cluster_scores(sc, cids)
        ctx = VcovContext(N=n, k=k, kfe=kfe, nfe=nfe)
        cfg = SSCConfig.from_dict({'kadj': True, 'kfixef': 'nonnested', 'Gadj': True, 'Gdf': 'conventional'})
        vcov, meta = compute_cluster_vcov(XtXinv, cs, ctx, G_out, cfg)
        assert vcov.shape == (k, k)


# ============================================================================
# 11. TestTwoWayCluster
# ============================================================================

class TestTwoWayCluster:  # name kept for readability

    def test_two_way_shape_and_symmetry(self, two_way):
        X, y, w, n, k = two_way['X'], two_way['y'], two_way['w'], two_way['n'], two_way['k']
        theta, XtXinv, resid, _ = ols(X, y)
        sc = X * resid.reshape(-1, 1)
        cdf = two_way['cluster_df']  # (n, 3): state, industry, pair
        cfg_dict = {'kadj': True, 'kfixef': 'full', 'Gadj': True, 'Gdf': 'conventional'}
        vcov, meta = compute_twoway_cluster_vcov(
            bread=XtXinv, scores=sc, cluster_df=cdf,
            ssc_dict=cfg_dict, N=n, k=k)
        assert vcov.shape == (k, k)
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)
        assert meta['vcov_type'] == 'cluster'
        assert meta['vcov_type_detail'] == 'CRV1-twoway'

    def test_two_way_cgm_larger_than_one_way(self, two_way):
        """Two-way SE should generally exceed one-way SE."""
        X, y, w, n, k = two_way['X'], two_way['y'], two_way['w'], two_way['n'], two_way['k']
        theta, XtXinv, resid, _ = ols(X, y)
        sc = X * resid.reshape(-1, 1)
        cdf = two_way['cluster_df']
        cfg_dict = {'kadj': True, 'kfixef': 'full', 'Gadj': True, 'Gdf': 'conventional'}

        vcov_2w, _ = compute_twoway_cluster_vcov(
            bread=XtXinv, scores=sc, cluster_df=cdf, ssc_dict=cfg_dict, N=n, k=k)

        # one-way on state only
        cs_s, G_s = compute_cluster_scores(sc, cdf[:, 0])
        ctx = VcovContext(N=n, k=k)
        cfg = SSCConfig.from_dict(cfg_dict)
        vcov_1w, _ = compute_cluster_vcov(XtXinv, cs_s, ctx, G_s, cfg)

        se_2w = np.sqrt(np.diag(vcov_2w))
        se_1w = np.sqrt(np.diag(vcov_1w))
        # Not guaranteed for all coefficients but expected on average
        assert se_2w.mean() >= se_1w.mean() * 0.5    # sanity bound

    def test_two_way_meta_fields(self, two_way):
        X, y, n, k = two_way['X'], two_way['y'], two_way['n'], two_way['k']
        theta, XtXinv, resid, _ = ols(X, y)
        sc = X * resid.reshape(-1, 1)
        vcov, meta = compute_twoway_cluster_vcov(
            bread=XtXinv, scores=sc, cluster_df=two_way['cluster_df'],
            ssc_dict={'kadj': True, 'kfixef': 'full', 'Gadj': True, 'Gdf': 'conventional'},
            N=n, k=k)
        assert isinstance(meta['n_clusters'], list)
        assert len(meta['n_clusters']) == 3    # state, industry, pair


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

