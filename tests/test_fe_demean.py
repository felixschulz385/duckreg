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
  - TestCompressionCorrectness — nobs == len(data), compression stability
"""

import logging
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

import pyfixest as pf
from duckreg import duckreg
import duckdb
from duckreg.core.transformers import IterativeDemeanTransformer
from duckreg.estimators.DuckLinearModel import DuckLinearModel
from tests.helpers import (
    assert_coef_se_close,
    duckreg_coef_se,
    duckreg_se_method,
    make_fe_regression_panel,
    make_unbalanced_panel,
    pyfixest_coef_se,
    pyfixest_vcov,
)


# ── constants ─────────────────────────────────────────────────────────────────

_FE_METHOD = "demean"
_UNBALANCED_DROP_FRAC = 0.25
_UNBALANCED_SEED = 99


# ── data generation ───────────────────────────────────────────────────────────

def _make_balanced_panel() -> pd.DataFrame:
    return make_fe_regression_panel(n_pixels=1000, n_years=10)


def _make_unbalanced_panel(balanced: pd.DataFrame) -> pd.DataFrame:
    return make_unbalanced_panel(
        balanced,
        fe_dims=["pixel_id", "year", "country", "soil_type"],
        drop_frac=_UNBALANCED_DROP_FRAC,
        seed=_UNBALANCED_SEED,
    )


# ── fixtures (module scope for heavy data, class scope for panel_data) ────────

@pytest.fixture(scope="module")
def balanced_df():
    return _make_balanced_panel()


@pytest.fixture(scope="module")
def unbalanced_df(balanced_df):
    return _make_unbalanced_panel(balanced_df)


@pytest.fixture(scope="module")
def balanced_path(balanced_df, tmp_path_factory):
    path = tmp_path_factory.mktemp("fe_demean") / "balanced.parquet"
    balanced_df.to_parquet(path, index=False)
    return str(path)


@pytest.fixture(scope="module")
def unbalanced_path(unbalanced_df, tmp_path_factory):
    path = tmp_path_factory.mktemp("fe_demean") / "unbalanced.parquet"
    unbalanced_df.to_parquet(path, index=False)
    return str(path)


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


def _duckreg(formula, path, vcov, fitter="numpy", fe_method=_FE_METHOD):
    return duckreg(formula=formula, data=path, seed=42, compression=5,
                   se_method=vcov, fitter=fitter, fe_method=fe_method)


def check_fe(df, path, *, fitter, has_iv, fe_depth, vcov, panel_balance):
    fe_part = " + ".join(_fe_cols(fe_depth))
    f_pf = (f"modis_median ~ exog_control | {fe_part} | ntl_harm ~ rainfall"
            if has_iv else f"modis_median ~ ntl_harm + exog_control | {fe_part}")
    fit_pf = pf.feols(f_pf, df, vcov=pyfixest_vcov(vcov))

    if has_iv:
        f_dr = f"modis_median ~ exog_control | {fe_part} | (ntl_harm ~ rainfall)"
    else:
        f_dr = f"modis_median ~ ntl_harm + exog_control | {fe_part}"

    m = _duckreg(f_dr, path, duckreg_se_method(vcov), fitter)
    pf_coef, pf_se = pyfixest_coef_se(fit_pf, "ntl_harm", has_iv)
    dr_coef, dr_se = duckreg_coef_se(m)

    coef_rtol = 5e-3 if has_iv else 1e-3
    se_rtol   = 1e-2 if has_iv else 5e-3
    if panel_balance == "unbalanced":
        coef_rtol *= 1.5
        se_rtol   *= 1.5

    assert_coef_se_close(
        dr_coef,
        pf_coef,
        dr_se,
        pf_se,
        coef_rtol=coef_rtol,
        se_rtol=se_rtol,
        label=f"fe_depth={fe_depth} iv={has_iv} fitter={fitter} "
        f"fe_method={_FE_METHOD} vcov={vcov} balance={panel_balance}",
    )


def check_merged_fe(df, path, *, fitter, dr_fe_part, pf_fe_part, vcov="HC1"):
    f_pf   = f"modis_median ~ ntl_harm + exog_control | {pf_fe_part}"
    fit_pf = pf.feols(f_pf, df, vcov=pyfixest_vcov(vcov))

    f_dr = f"modis_median ~ ntl_harm + exog_control | {dr_fe_part}"
    m = _duckreg(f_dr, path, duckreg_se_method(vcov), fitter)
    pf_coef = float(fit_pf.coef().loc["ntl_harm"])
    pf_se   = float(fit_pf.se().loc["ntl_harm"])
    dr_coef, dr_se = duckreg_coef_se(m)

    assert_coef_se_close(
        dr_coef,
        pf_coef,
        dr_se,
        pf_se,
        coef_rtol=1e-3,
        se_rtol=1e-2,
        label=f"merged fe={dr_fe_part!r} fe_method={_FE_METHOD}",
    )


def _fetch_transformed_df(transformer, variables):
    select_parts = [transformer.qident(fe_col) for fe_col in transformer.fe_cols]
    select_parts.append(transformer.transform_query(variables))
    query = (
        f"SELECT {', '.join(select_parts)} "
        f"FROM {transformer.qident(transformer._RESULT_TABLE)}"
    )
    return transformer.conn.execute(query).fetchdf()


def _result_columns(transformer):
    return list(
        transformer.conn.execute(
            f"SELECT column_name FROM (DESCRIBE {transformer.qident(transformer._RESULT_TABLE)})"
        ).fetchdf()["column_name"]
    )


def _max_abs_group_mean(df, fe_cols, variables):
    maxima = {}
    for fe_col in fe_cols:
        grouped = df.groupby(fe_col, dropna=False)[variables].mean().abs()
        maxima[fe_col] = grouped.max().max() if not grouped.empty else 0.0
    return maxima


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
# TestCompressionCorrectness — nobs and compression (demean)
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

    def test_compression_se_stability(self, panel_data):
        dr_exact = duckreg(
            "y ~ x1 + x2 | unit + year", data=panel_data,
            se_method="HC1", fe_method="demean", fitter="duckdb",
            compression=None,
        )
        dr_rounded = duckreg(
            "y ~ x1 + x2 | unit + year", data=panel_data,
            se_method="HC1", fe_method="demean", fitter="duckdb",
            compression=5,
        )
        se_exact   = np.sqrt(np.diag(dr_exact.vcov))
        se_rounded = np.sqrt(np.diag(dr_rounded.vcov))

        np.testing.assert_allclose(
            se_rounded, se_exact, rtol=0.01,
            err_msg=(
                "compression=5 caused >1% SE deviation (demean). "
                "Check that rounding affects strata formation only, "
                "not the residual or score computation."
            )
        )

    def test_df_compressed_structure(self, panel_data):
        """After fitting, df_compressed must contain the standard aggregation columns."""
        m = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                    se_method="iid", fe_method="demean", fitter="numpy")
        for col in ("count", "sum_y", "sum_y_sq"):
            assert col in m.df_compressed.columns, f"Missing column: {col!r}"
        assert m.df_compressed["count"].sum() == len(panel_data)


class TestDuckDBMemoryBehavior:
    """Regression tests for the out-of-core FE path."""

    def test_fe_duckdb_does_not_eagerly_populate_df_compressed(self, panel_data):
        m = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                    se_method="HC1", fe_method="demean", fitter="duckdb")
        assert m.point_estimate is not None
        assert m.vcov is not None
        assert m._data_fetched is False
        assert m._df_compressed is None

    def test_fe_duckdb_fit_and_vcov_avoid_fetch_path(self, panel_data):
        with patch.object(
            DuckLinearModel, "_ensure_data_fetched",
            side_effect=AssertionError("DuckDB FE path should stay out-of-core"),
        ):
            m = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                        se_method="HC1", fe_method="demean", fitter="duckdb")
        assert m.point_estimate is not None
        assert m.vcov is not None
        assert m._data_fetched is False

    def test_large_fetch_guard_skips_automatic_materialization(self, panel_data):
        m = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                    se_method="HC1", fe_method="demean", fitter="duckdb")
        m._DUCKDB_AUTO_FETCH_MAX_ROWS = 1
        m._ensure_data_fetched(force=False)
        assert m._data_fetched is False
        assert m._df_compressed is None

    def test_df_compressed_property_fetches_on_demand(self, panel_data):
        m = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                    se_method="iid", fe_method="demean", fitter="duckdb")
        df = m.df_compressed
        assert df is not None
        for col in ("count", "sum_y", "sum_y_sq"):
            assert col in df.columns, f"Missing column: {col!r}"
        assert df["count"].sum() == len(panel_data)

    def test_low_compression_warns_when_compression_unset(self, panel_data, caplog):
        with caplog.at_level(logging.WARNING):
            duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                    se_method="none", fe_method="demean", fitter="duckdb",
                    compression=None)
        assert any("consider setting compression" in r.message for r in caplog.records)

    def test_fe_tuning_knobs_reach_transformer(self, panel_data):
        m = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                    se_method="none", fe_method="demean", fitter="duckdb",
                    check_interval=7, convergence_sample=0.25)
        assert m._transformer.check_interval == 7
        assert m._transformer.convergence_sample == pytest.approx(0.25)

    def test_compression_minus_one_disables_grouping(self, panel_data):
        m = duckreg("y ~ x1 + x2 | unit + year", data=panel_data,
                    se_method="none", fe_method="demean", fitter="duckdb",
                    compression=-1)
        df = m.df_compressed
        assert m.n_compressed_rows == len(panel_data)
        assert len(df) == len(panel_data)
        assert np.all(df["count"].values == 1)


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

    def _transformer(self, conn, max_iterations=1000, tolerance=1e-8, **kwargs):
        return IterativeDemeanTransformer(
            conn=conn,
            table_name="panel",
            fe_cols=["unit", "year"],
            remove_singletons=False,
            max_iterations=max_iterations,
            tolerance=tolerance,
            **kwargs,
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

    def test_min_iterations_before_check_defers_nonfinal_checks(self):
        conn = self._make_conn()
        t = self._transformer(
            conn,
            max_iterations=6,
            min_iterations_before_check=5,
            check_interval=1,
            check_interval_growth=False,
        )
        checks = [i + 1 for i in range(t.max_iterations) if t._should_check_convergence(i)]
        assert checks == [5, 6]
        conn.close()

    def test_final_iteration_checks_even_before_minimum(self):
        conn = self._make_conn()
        t = self._transformer(
            conn,
            max_iterations=3,
            min_iterations_before_check=10,
            check_interval=1,
            check_interval_growth=False,
        )
        checks = [i + 1 for i in range(t.max_iterations) if t._should_check_convergence(i)]
        assert checks == [3]
        conn.close()

    def test_adaptive_convergence_intervals_grow(self):
        conn = self._make_conn()
        t = self._transformer(
            conn,
            max_iterations=130,
            min_iterations_before_check=5,
            check_interval=5,
            check_interval_growth=True,
            max_check_interval=25,
        )
        checks = [i + 1 for i in range(t.max_iterations) if t._should_check_convergence(i)]
        assert checks == [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 130]
        conn.close()

    def test_oneway_group_means_are_zero(self):
        conn = duckdb.connect()
        df = pd.DataFrame(
            {
                "unit": np.repeat([0, 1, 2], 4),
                "x": np.array([1.0, 2.0, 4.0, 7.0, 3.0, 6.0, 9.0, 12.0, 5.0, 8.0, 11.0, 14.0]),
                "y": np.array([2.0, 4.0, 1.0, 3.0, 7.0, 5.0, 9.0, 8.0, 6.0, 10.0, 12.0, 11.0]),
            }
        )
        conn.register("oneway_panel", df)
        t = IterativeDemeanTransformer(
            conn=conn,
            table_name="oneway_panel",
            fe_cols=["unit"],
            remove_singletons=False,
            tolerance=1e-10,
            check_interval=1,
        )
        t.fit_transform(["x", "y"])
        transformed = _fetch_transformed_df(t, ["x", "y"])
        maxima = _max_abs_group_mean(transformed, ["unit"], ["x", "y"])
        assert t.n_iterations == 1
        assert maxima["unit"] < 1e-10
        conn.close()

    def test_balanced_cartesian_shortcut_matches_map(self):
        rng = np.random.default_rng(31)
        n_units, n_periods = 12, 5
        unit = np.repeat(np.arange(n_units), n_periods)
        year = np.tile(np.arange(n_periods), n_units)
        df = pd.DataFrame(
            {
                "unit": unit,
                "year": year,
                "x": rng.standard_normal(len(unit)) + unit * 0.2 - year * 0.3,
                "y": rng.standard_normal(len(unit)) - unit * 0.5 + year * 0.4,
            }
        )

        conn_shortcut = duckdb.connect()
        conn_map = duckdb.connect()
        conn_shortcut.register("balanced_panel", df)
        conn_map.register("balanced_panel", df)

        t_shortcut = IterativeDemeanTransformer(
            conn=conn_shortcut,
            table_name="balanced_panel",
            fe_cols=["unit", "year"],
            remove_singletons=False,
            tolerance=1e-10,
            check_interval=1,
        )
        t_map = IterativeDemeanTransformer(
            conn=conn_map,
            table_name="balanced_panel",
            fe_cols=["unit", "year"],
            remove_singletons=False,
            tolerance=1e-10,
            check_interval=1,
        )
        t_map._is_complete_cartesian_panel = lambda: False

        t_shortcut.fit_transform(["x", "y"])
        t_map.fit_transform(["x", "y"])
        shortcut_df = _fetch_transformed_df(t_shortcut, ["x", "y"]).sort_values(
            ["unit", "year"]
        ).reset_index(drop=True)
        map_df = _fetch_transformed_df(t_map, ["x", "y"]).sort_values(
            ["unit", "year"]
        ).reset_index(drop=True)
        maxima = _max_abs_group_mean(shortcut_df, ["unit", "year"], ["x", "y"])

        assert t_shortcut.n_iterations == 1
        assert maxima["unit"] < 1e-10
        assert maxima["year"] < 1e-10
        np.testing.assert_allclose(shortcut_df[["x", "y"]], map_df[["x", "y"]], atol=1e-10)
        conn_shortcut.close()
        conn_map.close()

    def test_twoway_group_means_are_zero(self):
        conn = self._make_conn()
        t = self._transformer(conn, tolerance=1e-10)
        t.check_interval = 1
        t.fit_transform(["x1", "x2", "y"])
        transformed = _fetch_transformed_df(t, ["x1", "x2", "y"])
        maxima = _max_abs_group_mean(transformed, ["unit", "year"], ["x1", "x2", "y"])
        assert maxima["unit"] < 1e-10
        assert maxima["year"] < 1e-10
        conn.close()

    def test_multiple_variables_global_convergence(self):
        rng = np.random.default_rng(123)
        n_units, n_periods = 12, 8
        unit = np.repeat(np.arange(n_units), n_periods)
        year = np.tile(np.arange(n_periods), n_units)
        unit_fe = np.repeat(rng.standard_normal(n_units), n_periods)
        year_fe = np.tile(rng.standard_normal(n_periods), n_units)
        x_fast = np.zeros(len(unit))
        x_slow = 1e6 * rng.standard_normal(len(unit)) + unit_fe + year_fe
        df = pd.DataFrame(
            {"unit": unit, "year": year, "x_fast": x_fast, "x_slow": x_slow}
        )
        df = df[rng.random(len(df)) > 0.25].reset_index(drop=True)

        conn_fast = duckdb.connect()
        conn_both = duckdb.connect()
        conn_fast.register("panel_fast", df)
        conn_both.register("panel_both", df)

        t_fast = IterativeDemeanTransformer(
            conn=conn_fast,
            table_name="panel_fast",
            fe_cols=["unit", "year"],
            remove_singletons=False,
            tolerance=1e-8,
            check_interval=1,
        )
        t_both = IterativeDemeanTransformer(
            conn=conn_both,
            table_name="panel_both",
            fe_cols=["unit", "year"],
            remove_singletons=False,
            tolerance=1e-8,
            check_interval=1,
        )

        t_fast.fit_transform(["x_fast"])
        t_both.fit_transform(["x_fast", "x_slow"])
        transformed = _fetch_transformed_df(t_both, ["x_fast", "x_slow"])
        maxima = _max_abs_group_mean(transformed, ["unit", "year"], ["x_fast", "x_slow"])

        assert t_both.n_iterations >= t_fast.n_iterations
        assert maxima["unit"] < 1e-8
        assert maxima["year"] < 1e-8
        conn_fast.close()
        conn_both.close()

    def test_sampled_convergence_requires_exact_confirmation(self):
        conn = self._make_conn()
        t = IterativeDemeanTransformer(
            conn=conn,
            table_name="panel",
            fe_cols=["unit", "year"],
            remove_singletons=False,
            max_iterations=4,
            tolerance=1e-8,
            check_interval=1,
            convergence_sample=0.5,
            min_iterations_before_check=1,
        )

        calls = []
        values = iter([1e-12, 1e-3, 1e-12, 1e-12])

        def fake_measure(*, exact, conv_sql_exact, conv_sql_sampled):
            calls.append(exact)
            return next(values)

        t._measure_max_group_mean = fake_measure
        t.fit_transform(["x1", "x2", "y"])

        assert calls == [False, True, False, True]
        assert t.n_iterations == 2
        conn.close()

    def test_empty_group_sample_falls_back_to_exact_check(self):
        conn = self._make_conn()
        t = IterativeDemeanTransformer(
            conn=conn,
            table_name="panel",
            fe_cols=["unit", "year"],
            remove_singletons=False,
            max_iterations=3,
            tolerance=1e-8,
            check_interval=1,
            convergence_sample=0.5,
            min_iterations_before_check=1,
        )

        calls = []
        values = iter([float("inf"), 1e-12])

        def fake_measure(*, exact, conv_sql_exact, conv_sql_sampled):
            calls.append(exact)
            return next(values)

        t._measure_max_group_mean = fake_measure
        t.fit_transform(["x1", "x2", "y"])

        assert calls == [False, True]
        assert t.n_iterations == 1
        conn.close()

    def test_group_sampled_convergence_sql_samples_fe_groups(self):
        conn = self._make_conn()
        t = IterativeDemeanTransformer(
            conn=conn,
            table_name="panel",
            fe_cols=["unit", "year"],
            remove_singletons=False,
            convergence_sample=0.25,
        )
        t._build_fe_code_map()
        sql = t._build_group_sampled_convergence_sql(["_resid_0"], "_resid_store")

        assert "USING SAMPLE" not in sql
        assert "hash(\"_code_fe_0\")" in sql
        assert "hash(\"_code_fe_1\")" in sql
        assert "GROUP BY \"_code_fe_0\"" in sql
        assert "GROUP BY \"_code_fe_1\"" in sql
        conn.close()

    def test_fe_ordering_modes_are_equivalent(self):
        rng = np.random.default_rng(77)
        n_units, n_periods = 9, 4
        unit = np.repeat(np.arange(n_units), n_periods)
        year = np.tile(np.arange(n_periods), n_units)
        x = rng.standard_normal(len(unit)) + unit * 0.2 + year * 0.1
        y = rng.standard_normal(len(unit)) - unit * 0.4 + year * 0.3
        df = pd.DataFrame({"unit": unit, "year": year, "x": x, "y": y})
        df = df[(df["unit"] + df["year"]) % 5 != 0].reset_index(drop=True)

        outputs = {}
        for mode in ["input", "ascending_groups", "descending_groups"]:
            conn = duckdb.connect()
            conn.register("ordered_panel", df)
            t = IterativeDemeanTransformer(
                conn=conn,
                table_name="ordered_panel",
                fe_cols=["unit", "year"],
                remove_singletons=False,
                tolerance=1e-8,
                check_interval=1,
                fe_order=mode,
            )
            t.fit_transform(["x", "y"])
            transformed = _fetch_transformed_df(t, ["x", "y"]).sort_values(
                ["unit", "year", "x", "y"]
            ).reset_index(drop=True)
            outputs[mode] = (transformed, t.df_correction, _result_columns(t))
            conn.close()

        base_df, base_df_corr, base_cols = outputs["input"]
        for mode in ["ascending_groups", "descending_groups"]:
            other_df, other_df_corr, other_cols = outputs[mode]
            np.testing.assert_allclose(other_df[["x", "y"]], base_df[["x", "y"]], atol=1e-8)
            assert other_df_corr == base_df_corr
            assert other_cols[:2] == ["unit", "year"]
            assert other_cols == base_cols

    def test_constant_variable_with_fe_becomes_zero_and_skips_map(self):
        conn = duckdb.connect()
        df = pd.DataFrame(
            {
                "unit": [0, 0, 1, 1],
                "year": [0, 1, 0, 1],
                "const_var": [5.0, 5.0, 5.0, 5.0],
                "vary": [1.0, 3.0, 2.0, 6.0],
            }
        )
        conn.register("const_panel", df)
        t = IterativeDemeanTransformer(
            conn=conn,
            table_name="const_panel",
            fe_cols=["unit", "year"],
            remove_singletons=False,
            check_interval=1,
            drop_constant_variables=True,
        )
        t.fit_transform(["const_var", "vary"])
        transformed = _fetch_transformed_df(t, ["const_var", "vary"])
        maxima = _max_abs_group_mean(transformed, ["unit", "year"], ["vary"])

        assert np.allclose(transformed["const_var"].values, 0.0)
        assert maxima["unit"] < 1e-8
        assert maxima["year"] < 1e-8
        conn.close()

    def test_constant_variable_without_fe_remains_unchanged(self):
        conn = duckdb.connect()
        df = pd.DataFrame({"const_var": [7.0, 7.0, 7.0], "vary": [1.0, 2.0, 4.0]})
        conn.register("plain_const", df)
        t = IterativeDemeanTransformer(
            conn=conn,
            table_name="plain_const",
            fe_cols=[],
            remove_singletons=False,
            drop_constant_variables=True,
        )
        t.fit_transform(["const_var", "vary"])
        transformed = conn.execute(
            f"SELECT {t.transform_query(['const_var', 'vary'])} FROM {t.qident(t._RESULT_TABLE)}"
        ).fetchdf()

        assert np.allclose(transformed["const_var"].values, 7.0)
        assert list(transformed.columns) == ["const_var", "vary"]
        conn.close()

    def test_all_constant_variables_skip_map(self):
        conn = duckdb.connect()
        df = pd.DataFrame({"unit": [0, 0, 1, 1], "const_a": [2.0] * 4, "const_b": [3.0] * 4})
        conn.register("all_const", df)
        t = IterativeDemeanTransformer(
            conn=conn,
            table_name="all_const",
            fe_cols=["unit"],
            remove_singletons=False,
            drop_constant_variables=True,
        )
        t._run_map = lambda cols: (_ for _ in ()).throw(AssertionError("_run_map should not be called"))
        t.fit_transform(["const_a", "const_b"])
        transformed = _fetch_transformed_df(t, ["const_a", "const_b"])

        assert t.n_iterations == 0
        assert np.allclose(transformed["const_a"].values, 0.0)
        assert np.allclose(transformed["const_b"].values, 0.0)
        conn.close()

    def test_float_residual_type_supports_relaxed_tolerance(self):
        conn = self._make_conn()
        t = self._transformer(
            conn,
            tolerance=1e-6,
            check_interval=1,
            residual_type="FLOAT",
        )
        t.fit_transform(["x1", "x2", "y"])
        transformed = _fetch_transformed_df(t, ["x1", "x2", "y"])
        maxima = _max_abs_group_mean(transformed, ["unit", "year"], ["x1", "x2", "y"])
        assert maxima["unit"] < 1e-5
        assert maxima["year"] < 1e-5
        conn.close()

    def test_float_residual_type_rejects_tight_tolerance(self):
        conn = self._make_conn()
        with pytest.raises(ValueError, match="residual_type='FLOAT' requires tolerance >= 1e-7"):
            self._transformer(conn, tolerance=1e-8, residual_type="FLOAT")
        conn.close()

    def test_no_resid_prev_table_is_created(self):
        conn = self._make_conn()
        t = self._transformer(conn, check_interval=1)
        t.fit_transform(["x1", "x2", "y"])
        tables = set(
            conn.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchdf()["table_name"]
        )
        assert "_resid_prev" not in tables
        conn.close()

    def test_duckdb_runtime_settings_do_not_break_initialization(self):
        conn = duckdb.connect()
        df = pd.DataFrame({"unit": [0, 0, 1, 1], "year": [0, 1, 0, 1], "x": [1.0, 2.0, 3.0, 4.0]})
        conn.register("settings_panel", df)
        t = IterativeDemeanTransformer(
            conn=conn,
            table_name="settings_panel",
            fe_cols=["unit", "year"],
            remove_singletons=False,
            duckdb_memory_limit="256MB",
            duckdb_threads=1,
        )
        t.fit_transform(["x"])
        assert t.n_obs == 4
        conn.close()

    def test_zero_fixed_effects_passthrough(self):
        conn = duckdb.connect()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        conn.register("plain_tbl", df)
        t = IterativeDemeanTransformer(
            conn=conn,
            table_name="plain_tbl",
            fe_cols=[],
            remove_singletons=False,
        )
        result = t.fit_transform(["x", "y"])
        transformed = conn.execute(
            f"SELECT {t.transform_query(['x', 'y'])} FROM {t.qident(result)}"
        ).fetchdf()

        assert result == "demeaned_data"
        assert t.n_iterations == 0
        assert t.df_correction == 0
        assert t.n_obs == len(df)
        pd.testing.assert_frame_equal(transformed.reset_index(drop=True), df)
        cols = set(
            conn.execute(
                f"SELECT column_name FROM (DESCRIBE {t.qident(result)})"
            ).fetchdf()["column_name"]
        )
        assert "_row_id" not in cols
        conn.close()

    def test_identifier_safety_with_reserved_words_and_spaces(self):
        conn = duckdb.connect()
        df = pd.DataFrame(
            {
                "group": [0, 0, 1, 1],
                "state id": [10, 11, 10, 11],
                "x value": [1.0, 3.0, 5.0, 7.0],
            }
        )
        conn.register("weird names", df)
        t = IterativeDemeanTransformer(
            conn=conn,
            table_name="weird names",
            fe_cols=["group", "state id"],
            remove_singletons=False,
            tolerance=1e-10,
            check_interval=1,
        )
        t.fit_transform(["x value"])
        query = t.transform_query(["x value"])
        transformed = conn.execute(
            f"SELECT {t.qident('group')}, {t.qident('state id')}, {query} "
            f"FROM {t.qident(t._RESULT_TABLE)}"
        ).fetchdf()
        cols = set(
            conn.execute(
                f"SELECT column_name FROM (DESCRIBE {t.qident(t._RESULT_TABLE)})"
            ).fetchdf()["column_name"]
        )

        assert query == '"_resid_0" AS "x value"'
        assert "x value" in transformed.columns
        assert "_resid_0" in cols
        conn.close()

    def test_double_precision_default_tolerance_is_achievable(self):
        conn = self._make_conn()
        t = self._transformer(conn, tolerance=1e-8)
        t.check_interval = 1
        t.fit_transform(["x1", "x2", "y"])
        transformed = _fetch_transformed_df(t, ["x1", "x2", "y"])
        maxima = _max_abs_group_mean(transformed, ["unit", "year"], ["x1", "x2", "y"])

        assert maxima["unit"] < 1e-8
        assert maxima["year"] < 1e-8
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

    def test_iterative_singleton_pruning_removes_cascading_singletons(self):
        conn = duckdb.connect()
        df = pd.DataFrame(
            [
                {"fe1": "a", "fe2": "x", "x": 1.0},
                {"fe1": "a", "fe2": "y", "x": 2.0},
                {"fe1": "b", "fe2": "y", "x": 3.0},
                {"fe1": "c", "fe2": "z", "x": 4.0},
                {"fe1": "c", "fe2": "z", "x": 5.0},
            ]
        )
        conn.register("cascade_tbl", df)
        t = IterativeDemeanTransformer(
            conn=conn,
            table_name="cascade_tbl",
            fe_cols=["fe1", "fe2"],
            remove_singletons=True,
            tolerance=1e-10,
            check_interval=1,
        )
        t.fit_transform(["x"])
        remaining = conn.execute(
            f"SELECT {t.qident('fe1')}, {t.qident('fe2')} "
            f"FROM {t.qident(t._RESULT_TABLE)}"
        ).fetchdf()

        assert t.n_rows_dropped_singletons == 3
        assert t.n_obs == 2
        assert set(map(tuple, remaining.to_records(index=False))) == {("c", "z")}
        conn.close()

    def test_one_pass_singleton_pruning_keeps_cascaded_singletons(self):
        conn = duckdb.connect()
        df = pd.DataFrame(
            [
                {"fe1": "a", "fe2": "x", "x": 1.0},
                {"fe1": "a", "fe2": "y", "x": 2.0},
                {"fe1": "b", "fe2": "y", "x": 3.0},
                {"fe1": "c", "fe2": "z", "x": 4.0},
                {"fe1": "c", "fe2": "z", "x": 5.0},
            ]
        )
        conn.register("cascade_tbl_one_pass", df)
        t = IterativeDemeanTransformer(
            conn=conn,
            table_name="cascade_tbl_one_pass",
            fe_cols=["fe1", "fe2"],
            remove_singletons=True,
            singleton_pruning="one_pass",
            tolerance=1e-10,
            check_interval=1,
        )
        t.fit_transform(["x"])
        remaining = conn.execute(
            f"SELECT {t.qident('fe1')}, {t.qident('fe2')} "
            f"FROM {t.qident(t._RESULT_TABLE)}"
        ).fetchdf()

        assert t.n_rows_dropped_singletons == 2
        assert t.n_obs == 3
        assert set(map(tuple, remaining.to_records(index=False))) == {("a", "y"), ("c", "z")}
        conn.close()

    def test_estimation_succeeds_after_removal(self, singleton_path):
        path, _ = singleton_path
        m = duckreg("y ~ x | unit", data=path, fe_method="demean",
                    fitter="numpy", se_method="HC1")
        assert m.point_estimate is not None
        assert m.vcov is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
