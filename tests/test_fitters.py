"""
tests/test_fitters.py

Comprehensive test suite for:
    - base.py   : FitterResult, _resolve_vcov_spec, _validate_and_prepare_data,
                  _compute_weighted_matrices, BaseFitter._normalize_fit_kwargs,
                  compute_vcov_dispatch
    - numpy_fitter.py : NumpyFitter.fit, NumpyFitter.fit_vcov
    - duckdb_fitter.py : DuckDBFitter.fit, DuckDBFitter.fit_vcov,
                         get_fitter, wls_duckdb

All SQL tests use an in-process DuckDB connection — no external fixtures needed.

Usage
-----
    pytest tests/test_fitters.py -v
"""

from __future__ import annotations

import math
from typing import List
from unittest.mock import MagicMock, patch

import duckdb
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Imports under test — adjust path to match your package layout
# ---------------------------------------------------------------------------
from duckreg.core.fitters.base import (
    DEFAULT_ALPHA,
    BaseFitter,
    FitterResult,
    _compute_weighted_matrices,
    _resolve_vcov_spec,
    _validate_and_prepare_data,
    compute_vcov_dispatch,
)
from duckreg.core.fitters.numpy_fitter import NumpyFitter
from duckreg.core.fitters.duckdb_fitter import (
    DuckDBFitter,
    get_fitter,
    wls_duckdb,
)
from duckreg.core.vcov import VcovSpec

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

ATOL = 1e-9
RTOL = 1e-7


def _make_conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(":memory:")


def _seed_table(conn, table_name, x1, x2, y, count=None):
    n = len(x1)
    if count is None:
        count = [1] * n
    sum_y_sq = [yi ** 2 for yi in y]
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(
        f"""
        CREATE TABLE {table_name} AS
        SELECT
            unnest({x1!r}) AS x1,
            unnest({x2!r}) AS x2,
            unnest({y!r})  AS sum_y,
            unnest({sum_y_sq!r}) AS sum_y_sq,
            unnest({count!r}) AS count
        """
    )


def _ols_reference(X, y, w):
    """Manual WLS via SVD for ground-truth comparison."""
    W = np.diag(np.sqrt(w))
    theta, *_ = np.linalg.lstsq(W @ X, W @ y, rcond=None)
    return theta


def _make_vcov_spec(vcov_type="HC1"):
    return VcovSpec.build(vcov_type, None)


# ---------------------------------------------------------------------------
# ── 1. FitterResult ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestFitterResult:

    def _make(self, k=3):
        return FitterResult(
            coefficients=np.ones(k),
            coef_names=[f"x{i}" for i in range(k)],
            n_obs=100,
        )

    def test_field_defaults(self):
        r = self._make()
        assert r.vcov is None
        assert r.se_type == "none"
        assert r.r_squared is None
        assert r.rss is None
        assert r.n_clusters is None

    def test_standard_errors_none_when_no_vcov(self):
        r = self._make()
        assert r.standard_errors is None

    def test_standard_errors_computed_from_vcov(self):
        r = self._make(k=2)
        r.vcov = np.array([[4.0, 0.0], [0.0, 9.0]])
        se = r.standard_errors
        assert se == pytest.approx([2.0, 3.0])

    def test_standard_errors_clamps_negative_diag(self):
        r = self._make(k=2)
        r.vcov = np.array([[1.0, 0.0], [0.0, -1.0]])
        se = r.standard_errors
        assert se[1] == pytest.approx(math.sqrt(1e-16))

    def test_to_dict_required_keys(self):
        r = self._make()
        d = r.to_dict()
        assert "coefficients" in d
        assert "coef_names" in d
        assert "n_obs" in d
        assert "se_type" in d

    def test_to_dict_optional_keys_absent_when_none(self):
        r = self._make()
        d = r.to_dict()
        assert "vcov" not in d
        assert "r_squared" not in d
        assert "rss" not in d

    def test_to_dict_includes_vcov_when_set(self):
        r = self._make(k=2)
        r.vcov = np.eye(2)
        d = r.to_dict()
        assert "vcov" in d
        assert "standard_errors" in d

    def test_to_dict_includes_residual_stats(self):
        r = self._make(k=2)
        r.XtX_inv = np.eye(2)
        r.meat = np.eye(2)
        d = r.to_dict()
        assert "XtX_inv" in d
        assert "meat" in d

    def test_to_dict_extra_merged(self):
        r = self._make()
        r.extra = {"custom_field": 42}
        d = r.to_dict()
        assert d["custom_field"] == 42


# ---------------------------------------------------------------------------
# ── 2. _resolve_vcov_spec ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestResolveVcovSpec:

    def test_returns_provided_spec_unchanged(self):
        spec = _make_vcov_spec("HC1")
        result = _resolve_vcov_spec(spec, None, None, False)
        assert result is spec

    def test_defaults_to_hc1_when_no_clusters(self):
        spec = _resolve_vcov_spec(None, None, None, False)
        assert spec.vcov_detail == "HC1"

    def test_uses_provided_vcov_type(self):
        spec = _resolve_vcov_spec(None, "HC2", None, False)
        assert spec.vcov_detail == "HC2"

    def test_iid_without_clusters(self):
        spec = _resolve_vcov_spec(None, "iid", None, False)
        assert spec.vcov_type == "iid"

    def test_cluster_overrides_to_crv1(self):
        """When has_clusters=True and vcov_type is not CRV, must switch to CRV1."""
        spec = _resolve_vcov_spec(None, "HC1", None, has_clusters=True)
        assert spec.vcov_detail == "CRV1"
        assert spec.is_clustered is True

    def test_explicit_crv1_with_clusters_unchanged(self):
        spec = _resolve_vcov_spec(None, "CRV1", None, has_clusters=True)
        assert spec.vcov_detail == "CRV1"

    def test_explicit_crv3_with_clusters_unchanged(self):
        spec = _resolve_vcov_spec(None, "CRV3", None, has_clusters=True)
        assert spec.vcov_detail == "CRV3"

    def test_none_vcov_type_with_clusters_defaults_crv1(self):
        spec = _resolve_vcov_spec(None, None, None, has_clusters=True)
        assert spec.vcov_detail == "CRV1"


# ---------------------------------------------------------------------------
# ── 3. _validate_and_prepare_data ──────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestValidateAndPrepareData:

    def test_returns_five_tuple(self):
        X = np.ones((5, 2))
        result = _validate_and_prepare_data(X, np.ones(5), np.ones(5))
        assert len(result) == 5

    def test_1d_X_becomes_2d(self):
        X, y, w, n_rows, n_obs = _validate_and_prepare_data(
            np.ones(5), np.ones(5), np.ones(5)
        )
        assert X.ndim == 2
        assert X.shape == (5, 1)

    def test_1d_y_becomes_column(self):
        X, y, w, n_rows, n_obs = _validate_and_prepare_data(
            np.ones((5, 1)), np.ones(5), np.ones(5)
        )
        assert y.ndim == 2

    def test_weights_flattened(self):
        w_col = np.ones((5, 1))
        _, _, w, _, _ = _validate_and_prepare_data(
            np.ones((5, 1)), np.ones(5), w_col
        )
        assert w.ndim == 1

    def test_n_rows_correct(self):
        _, _, _, n_rows, _ = _validate_and_prepare_data(
            np.ones((7, 2)), np.ones(7), np.ones(7)
        )
        assert n_rows == 7

    def test_n_obs_is_sum_of_weights(self):
        w = np.array([1.0, 2.0, 3.0])
        _, _, _, _, n_obs = _validate_and_prepare_data(
            np.ones((3, 1)), np.ones(3), w
        )
        assert n_obs == 6


# ---------------------------------------------------------------------------
# ── 4. _compute_weighted_matrices ──────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestComputeWeightedMatrices:

    def test_xtx_shape(self):
        X = np.random.randn(10, 3)
        XtX, _ = _compute_weighted_matrices(X, np.ones(10), np.ones(10), alpha=0.0)
        assert XtX.shape == (3, 3)

    def test_xty_shape(self):
        X = np.random.randn(10, 3)
        _, Xty = _compute_weighted_matrices(X, np.ones((10, 1)), np.ones(10), alpha=0.0)
        assert Xty.shape == (3, 1)

    def test_xtx_symmetry(self):
        np.random.seed(0)
        X = np.random.randn(20, 4)
        XtX, _ = _compute_weighted_matrices(X, np.ones(20), np.ones(20), alpha=0.0)
        assert np.allclose(XtX, XtX.T)

    def test_regularisation_on_diagonal(self):
        X = np.eye(3)
        alpha = 0.5
        XtX, _ = _compute_weighted_matrices(X, np.zeros(3), np.ones(3), alpha=alpha)
        assert np.allclose(np.diag(XtX), 1.0 + alpha)

    def test_agrees_with_manual_computation(self):
        np.random.seed(1)
        X = np.random.randn(15, 2)
        y = np.random.randn(15, 1)
        w = np.ones(15)
        XtX, Xty = _compute_weighted_matrices(X, y, w, alpha=0.0)
        assert np.allclose(XtX, X.T @ X, atol=ATOL)
        assert np.allclose(Xty, X.T @ y, atol=ATOL)


# ---------------------------------------------------------------------------
# ── 5. BaseFitter._normalize_fit_kwargs ─────────────────────────────────────
# ---------------------------------------------------------------------------


class TestNormalizeFitKwargs:

    def test_xcols_renamed(self):
        kw = BaseFitter._normalize_fit_kwargs(xcols=["a", "b"])
        assert "x_cols" in kw
        assert "xcols" not in kw

    def test_ycol_renamed(self):
        kw = BaseFitter._normalize_fit_kwargs(ycol="y")
        assert "y_col" in kw
        assert "ycol" not in kw

    def test_weightcol_renamed(self):
        kw = BaseFitter._normalize_fit_kwargs(weightcol="n")
        assert "weight_col" in kw
        assert "weightcol" not in kw

    def test_kfe_renamed(self):
        kw = BaseFitter._normalize_fit_kwargs(kfe=3)
        assert "k_fe" in kw
        assert "kfe" not in kw

    def test_nfe_renamed(self):
        kw = BaseFitter._normalize_fit_kwargs(nfe=2)
        assert "n_fe" in kw

    def test_XtXinv_renamed(self):
        mat = np.eye(2)
        kw = BaseFitter._normalize_fit_kwargs(XtXinv=mat)
        assert "XtX_inv" in kw

    def test_canonical_name_not_overwritten(self):
        """If both alias and canonical are present, canonical wins."""
        kw = BaseFitter._normalize_fit_kwargs(xcols=["old"], x_cols=["new"])
        assert kw["x_cols"] == ["new"]

    def test_unrelated_kwargs_preserved(self):
        kw = BaseFitter._normalize_fit_kwargs(alpha=0.1, some_flag=True)
        assert kw["alpha"] == 0.1
        assert kw["some_flag"] is True


# ---------------------------------------------------------------------------
# ── 6. NumpyFitter.fit ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestNumpyFitterFit:

    @pytest.fixture
    def fitter(self):
        return NumpyFitter(alpha=0.0)

    def test_returns_fitter_result(self, fitter):
        X = np.ones((10, 1))
        y = np.arange(10, dtype=float)
        result = fitter.fit(X=X, y=y, weights=np.ones(10))
        assert isinstance(result, FitterResult)

    def test_intercept_only_theta_equals_mean(self, fitter):
        n = 20
        X = np.ones((n, 1))
        y = np.full(n, 5.0)
        result = fitter.fit(X=X, y=y, weights=np.ones(n))
        assert result.coefficients == pytest.approx([5.0])

    def test_two_regressors_matches_lstsq(self, fitter):
        np.random.seed(0)
        n, k = 50, 2
        X = np.random.randn(n, k)
        y = X @ np.array([1.0, -2.0]) + 0.1 * np.random.randn(n)
        w = np.ones(n)
        result = fitter.fit(X=X, y=y, weights=w)
        theta_ref = _ols_reference(X, y, w)
        assert np.allclose(result.coefficients, theta_ref, atol=1e-6)

    def test_frequency_weights_scale_n_obs(self, fitter):
        X = np.ones((5, 1))
        y = np.ones(5)
        w = 3 * np.ones(5)
        result = fitter.fit(X=X, y=y, weights=w)
        assert result.n_obs == 15

    def test_r_squared_perfect_fit(self, fitter):
        X = np.column_stack([np.ones(5), np.array([1., 2., 3., 4., 5.])])
        y = 2.0 + 3.0 * X[:, 1]
        result = fitter.fit(X=X, y=y, weights=np.ones(5))
        assert result.r_squared == pytest.approx(1.0, abs=1e-10)

    def test_r_squared_zero_when_mean_only(self, fitter):
        """Intercept-only model should have R²=0 when y has variance."""
        X = np.ones((10, 1))
        y = np.arange(10, dtype=float)
        result = fitter.fit(X=X, y=y, weights=np.ones(10))
        assert result.r_squared == pytest.approx(0.0, abs=1e-10)

    def test_precomputed_coefficients_used(self, fitter):
        X = np.ones((5, 1))
        y = np.ones(5)
        theta_given = np.array([42.0])
        result = fitter.fit(X=X, y=y, weights=np.ones(5), coefficients=theta_given)
        assert result.coefficients == pytest.approx([42.0])

    def test_coef_names_stored(self, fitter):
        X = np.ones((5, 2))
        result = fitter.fit(X=X, y=np.ones(5), weights=np.ones(5),
                            coef_names=["alpha", "beta"])
        assert result.coef_names == ["alpha", "beta"]

    def test_coef_names_default_generated(self, fitter):
        X = np.ones((5, 2))
        result = fitter.fit(X=X, y=np.ones(5), weights=np.ones(5))
        assert result.coef_names == ["x0", "x1"]

    def test_xtx_stored(self, fitter):
        X = np.eye(3)
        result = fitter.fit(X=X, y=np.ones(3), weights=np.ones(3))
        assert result.XtX is not None
        assert result.XtX.shape == (3, 3)

    def test_last_result_cached(self, fitter):
        X = np.ones((5, 1))
        result = fitter.fit(X=X, y=np.ones(5), weights=np.ones(5))
        assert fitter._last_result is result

    def test_rss_nonnegative(self, fitter):
        np.random.seed(3)
        X = np.random.randn(30, 3)
        y = np.random.randn(30)
        result = fitter.fit(X=X, y=y, weights=np.ones(30))
        assert result.rss >= -1e-10  # allow tiny floating-point noise


# ---------------------------------------------------------------------------
# ── 7. NumpyFitter.fit_vcov ─────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestNumpyFitterFitVcov:

    @pytest.fixture
    def fitter_and_data(self):
        np.random.seed(7)
        n, k = 100, 3
        X = np.random.randn(n, k)
        y = X @ np.array([1.0, -1.0, 0.5]) + 0.3 * np.random.randn(n)
        w = np.ones(n)
        fitter = NumpyFitter(alpha=0.0)
        result = fitter.fit(X=X, y=y, weights=w)
        return fitter, X, y, w, result

    def test_returns_three_tuple(self, fitter_and_data):
        fitter, X, y, w, result = fitter_and_data
        out = fitter.fit_vcov(X=X, y=y, weights=w,
                              coefficients=result.coefficients,
                              vcov_type="HC1",
                              existing_result=result)
        assert len(out) == 3

    def test_vcov_is_symmetric(self, fitter_and_data):
        fitter, X, y, w, result = fitter_and_data
        vcov, _, _ = fitter.fit_vcov(X=X, y=y, weights=w,
                                     coefficients=result.coefficients,
                                     vcov_type="HC1",
                                     existing_result=result)
        assert np.allclose(vcov, vcov.T, atol=ATOL)

    def test_vcov_positive_semidefinite(self, fitter_and_data):
        fitter, X, y, w, result = fitter_and_data
        vcov, _, _ = fitter.fit_vcov(X=X, y=y, weights=w,
                                     coefficients=result.coefficients,
                                     vcov_type="HC1",
                                     existing_result=result)
        eigenvalues = np.linalg.eigvalsh(vcov)
        assert np.all(eigenvalues >= -1e-10)

    def test_vcov_shape(self, fitter_and_data):
        fitter, X, y, w, result = fitter_and_data
        k = X.shape[1]
        vcov, _, _ = fitter.fit_vcov(X=X, y=y, weights=w,
                                     coefficients=result.coefficients,
                                     vcov_type="HC1",
                                     existing_result=result)
        assert vcov.shape == (k, k)

    def test_iid_vcov(self, fitter_and_data):
        fitter, X, y, w, result = fitter_and_data
        vcov, meta, _ = fitter.fit_vcov(X=X, y=y, weights=w,
                                        coefficients=result.coefficients,
                                        vcov_type="iid",
                                        existing_result=result)
        assert vcov is not None

    def test_crv1_vcov(self, fitter_and_data):
        fitter, X, y, w, result = fitter_and_data
        n = X.shape[0]
        cluster_ids = np.repeat(np.arange(10), n // 10)
        vcov, meta, agg = fitter.fit_vcov(
            X=X, y=y, weights=w,
            coefficients=result.coefficients,
            vcov_type="CRV1",
            cluster_ids=cluster_ids,
            existing_result=result,
        )
        assert agg.get("n_clusters") == 10

    def test_alias_kfe_nfe(self, fitter_and_data):
        fitter, X, y, w, result = fitter_and_data
        # Should not raise
        fitter.fit_vcov(X=X, y=y, weights=w,
                        coefficients=result.coefficients,
                        vcov_type="HC1",
                        kfe=5, nfe=2,
                        existing_result=result)

    def test_missing_coefficients_raises(self, fitter_and_data):
        fitter, X, y, w, _ = fitter_and_data
        fitter._last_result = None
        with pytest.raises(ValueError, match="coefficients must be provided"):
            fitter.fit_vcov(X=X, y=y, weights=w, vcov_type="HC1")

    def test_residual_X_overrides_residual_computation(self, fitter_and_data):
        """2SLS path: residual_X must be used to compute residuals."""
        fitter, X, y, w, result = fitter_and_data
        residual_X = X.copy()
        residual_X[:, 0] *= 0.5  # perturb first column
        vcov_standard, _, _ = fitter.fit_vcov(
            X=X, y=y, weights=w, coefficients=result.coefficients,
            vcov_type="HC1", existing_result=result
        )
        vcov_alt, _, _ = fitter.fit_vcov(
            X=X, y=y, weights=w, coefficients=result.coefficients,
            vcov_type="HC1", existing_result=result, residual_X=residual_X
        )
        # Different residuals → different vcov
        assert not np.allclose(vcov_standard, vcov_alt)

    @pytest.mark.parametrize("vcov_type", ["HC1", "HC2", "HC3"])
    def test_heteroskedastic_types(self, fitter_and_data, vcov_type):
        fitter, X, y, w, result = fitter_and_data
        vcov, _, _ = fitter.fit_vcov(X=X, y=y, weights=w,
                                     coefficients=result.coefficients,
                                     vcov_type=vcov_type,
                                     existing_result=result)
        assert vcov.shape == (X.shape[1], X.shape[1])
        assert np.allclose(vcov, vcov.T, atol=ATOL)


# ---------------------------------------------------------------------------
# ── 8. DuckDBFitter.fit ─────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestDuckDBFitterFit:

    @pytest.fixture
    def conn(self):
        c = _make_conn()
        yield c
        c.close()

    @pytest.fixture
    def simple_table(self, conn):
        _seed_table(conn, "t",
                    x1=[1.0, 2.0, 3.0, 4.0, 5.0],
                    x2=[2.0, 4.0, 1.0, 3.0, 5.0],
                    y=[3.0, 8.0, 5.0, 9.0, 14.0])
        return "t"

    @pytest.fixture
    def fitter(self, conn):
        return DuckDBFitter(conn=conn, alpha=0.0)

    def test_returns_fitter_result(self, fitter, simple_table):
        result = fitter.fit(table_name=simple_table,
                            x_cols=["x1", "x2"], y_col="sum_y")
        assert isinstance(result, FitterResult)

    def test_n_obs_correct(self, fitter, simple_table):
        result = fitter.fit(table_name=simple_table,
                            x_cols=["x1"], y_col="sum_y")
        assert result.n_obs == 5

    def test_matches_numpy_fitter(self, conn, simple_table):
        """DuckDB and numpy backends must produce the same theta."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [2.0, 4.0, 1.0, 3.0, 5.0]
        y  = [3.0, 8.0, 5.0, 9.0, 14.0]

        # DuckDB
        db_fitter = DuckDBFitter(conn=conn, alpha=0.0)
        db_result = db_fitter.fit(table_name=simple_table,
                                  x_cols=["x1", "x2"], y_col="sum_y",
                                  add_intercept=True)

        # NumPy
        X = np.column_stack([np.ones(5), x1, x2])
        np_fitter = NumpyFitter(alpha=0.0)
        np_result = np_fitter.fit(X=X, y=np.array(y), weights=np.ones(5),
                                  coef_names=["Intercept", "x1", "x2"])

        assert np.allclose(db_result.coefficients, np_result.coefficients,
                           atol=1e-8, rtol=1e-6)

    def test_r_squared_stored(self, fitter, simple_table):
        result = fitter.fit(table_name=simple_table,
                            x_cols=["x1"], y_col="sum_y")
        assert result.r_squared is not None
        assert 0.0 <= result.r_squared <= 1.0 + 1e-10

    def test_precomputed_coefficients(self, fitter, simple_table):
        theta_given = np.array([0.0, 1.0, 1.0])
        result = fitter.fit(table_name=simple_table,
                            x_cols=["x1", "x2"], y_col="sum_y",
                            add_intercept=True,
                            coefficients=theta_given)
        assert np.allclose(result.coefficients, theta_given)

    def test_alias_xcols(self, fitter, simple_table):
        result = fitter.fit(table_name=simple_table,
                            xcols=["x1"], ycol="sum_y")
        assert isinstance(result, FitterResult)

    def test_alias_weightcol(self, conn, simple_table):
        fitter = DuckDBFitter(conn=conn, alpha=0.0)
        result = fitter.fit(table_name=simple_table,
                            x_cols=["x1"], y_col="sum_y",
                            weightcol="count")
        assert isinstance(result, FitterResult)

    def test_last_result_cached(self, fitter, simple_table):
        result = fitter.fit(table_name=simple_table,
                            x_cols=["x1"], y_col="sum_y")
        assert fitter._last_result is result

    def test_xtx_stored(self, fitter, simple_table):
        result = fitter.fit(table_name=simple_table,
                            x_cols=["x1"], y_col="sum_y",
                            add_intercept=True)
        assert result.XtX is not None

    def test_coef_names_with_intercept(self, fitter, simple_table):
        result = fitter.fit(table_name=simple_table,
                            x_cols=["x1", "x2"], y_col="sum_y",
                            add_intercept=True)
        assert result.coef_names[0] == "Intercept"

    def test_coef_names_without_intercept(self, fitter, simple_table):
        result = fitter.fit(table_name=simple_table,
                            x_cols=["x1", "x2"], y_col="sum_y",
                            add_intercept=False)
        assert result.coef_names == ["x1", "x2"]


# ---------------------------------------------------------------------------
# ── 9. DuckDBFitter.fit_vcov ────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestDuckDBFitterFitVcov:

    @pytest.fixture
    def conn(self):
        c = _make_conn()
        yield c
        c.close()

    @pytest.fixture
    def large_table(self, conn):
        """50-row table with genuine variance for SE estimation."""
        np.random.seed(42)
        n = 50
        x1 = np.random.randn(n).tolist()
        x2 = np.random.randn(n).tolist()
        y = (np.array(x1) + 2 * np.array(x2) + 0.5 * np.random.randn(n)).tolist()
        _seed_table(conn, "big", x1=x1, x2=x2, y=y)
        return "big"

    @pytest.fixture
    def fitter(self, conn):
        return DuckDBFitter(conn=conn, alpha=0.0)

    def test_returns_three_tuple(self, fitter, large_table):
        result = fitter.fit(table_name=large_table,
                            x_cols=["x1", "x2"], y_col="sum_y")
        out = fitter.fit_vcov(table_name=large_table,
                              x_cols=["x1", "x2"], y_col="sum_y",
                              vcov_type="HC1",
                              coefficients=result.coefficients,
                              existing_result=result)
        assert len(out) == 3

    def test_vcov_symmetric(self, fitter, large_table):
        result = fitter.fit(table_name=large_table,
                            x_cols=["x1", "x2"], y_col="sum_y")
        vcov, _, _ = fitter.fit_vcov(table_name=large_table,
                                     x_cols=["x1", "x2"], y_col="sum_y",
                                     vcov_type="HC1",
                                     coefficients=result.coefficients,
                                     existing_result=result)
        assert np.allclose(vcov, vcov.T, atol=ATOL)

    def test_vcov_positive_semidefinite(self, fitter, large_table):
        result = fitter.fit(table_name=large_table,
                            x_cols=["x1", "x2"], y_col="sum_y")
        vcov, _, _ = fitter.fit_vcov(table_name=large_table,
                                     x_cols=["x1", "x2"], y_col="sum_y",
                                     vcov_type="HC1",
                                     coefficients=result.coefficients,
                                     existing_result=result)
        eigenvalues = np.linalg.eigvalsh(vcov)
        assert np.all(eigenvalues >= -1e-10)

    def test_iid_vcov(self, fitter, large_table):
        result = fitter.fit(table_name=large_table,
                            x_cols=["x1", "x2"], y_col="sum_y")
        vcov, _, _ = fitter.fit_vcov(table_name=large_table,
                                     x_cols=["x1", "x2"], y_col="sum_y",
                                     vcov_type="iid",
                                     coefficients=result.coefficients,
                                     existing_result=result)
        assert vcov is not None

    def test_crv1_vcov(self, conn, large_table):
        """CRV1: cluster_scores and n_clusters must be in aggregates."""
        conn.execute("""
            ALTER TABLE big ADD COLUMN IF NOT EXISTS cluster_id INT
        """)
        conn.execute("""
            UPDATE big SET cluster_id = (rowid % 5)
        """)
        fitter = DuckDBFitter(conn=conn, alpha=0.0)
        result = fitter.fit(table_name=large_table,
                            x_cols=["x1", "x2"], y_col="sum_y")
        vcov, _, agg = fitter.fit_vcov(
            table_name=large_table,
            x_cols=["x1", "x2"], y_col="sum_y",
            cluster_col="cluster_id",
            coefficients=result.coefficients,
            existing_result=result,
        )
        assert "cluster_scores" in agg
        assert agg["n_clusters"] == 5

    def test_vcov_agrees_with_numpy_fitter(self, conn, large_table):
        """DuckDB and NumPy HC1 vcov must agree on the same (uncompressed) data."""
        rows = conn.execute("SELECT x1, x2, sum_y FROM big").fetchall()
        x1 = [r[0] for r in rows]
        x2 = [r[1] for r in rows]
        y  = [r[2] for r in rows]
        X  = np.column_stack([np.ones(len(rows)), x1, x2])
        y_arr = np.array(y)
        w = np.ones(len(rows))

        # NumPy
        np_fitter = NumpyFitter(alpha=0.0)
        np_result = np_fitter.fit(X=X, y=y_arr, weights=w,
                                  coef_names=["Intercept", "x1", "x2"])
        np_vcov, _, _ = np_fitter.fit_vcov(
            X=X, y=y_arr, weights=w,
            coefficients=np_result.coefficients,
            vcov_type="HC1", existing_result=np_result,
        )

        # DuckDB
        db_fitter = DuckDBFitter(conn=conn, alpha=0.0)
        db_result = db_fitter.fit(table_name=large_table,
                                  x_cols=["x1", "x2"], y_col="sum_y")
        db_vcov, _, _ = db_fitter.fit_vcov(
            table_name=large_table,
            x_cols=["x1", "x2"], y_col="sum_y",
            vcov_type="HC1",
            coefficients=db_result.coefficients,
            existing_result=db_result,
        )

        assert np.allclose(np_vcov, db_vcov, atol=1e-8, rtol=1e-5)

    def test_alias_xcols_ycol(self, fitter, large_table):
        result = fitter.fit(table_name=large_table,
                            x_cols=["x1", "x2"], y_col="sum_y")
        vcov, _, _ = fitter.fit_vcov(
            table_name=large_table,
            xcols=["x1", "x2"], ycol="sum_y",
            vcov_type="HC1",
            coefficients=result.coefficients,
            existing_result=result,
        )
        assert vcov is not None

    @pytest.mark.parametrize("vcov_type", ["HC1", "HC2", "HC3"])
    def test_heteroskedastic_types(self, fitter, large_table, vcov_type):
        result = fitter.fit(table_name=large_table,
                            x_cols=["x1", "x2"], y_col="sum_y")
        vcov, _, _ = fitter.fit_vcov(
            table_name=large_table,
            x_cols=["x1", "x2"], y_col="sum_y",
            vcov_type=vcov_type,
            coefficients=result.coefficients,
            existing_result=result,
        )
        assert vcov.shape == (3, 3)
        assert np.allclose(vcov, vcov.T, atol=ATOL)

    def test_reuses_xtx_from_existing_result(self, fitter, large_table, mocker):
        """fit_vcov must not recompute XtX when existing_result is provided."""
        result = fitter.fit(table_name=large_table,
                            x_cols=["x1", "x2"], y_col="sum_y")
        spy = mocker.spy(fitter, "_fetch_suffstats")
        fitter.fit_vcov(
            table_name=large_table,
            x_cols=["x1", "x2"], y_col="sum_y",
            vcov_type="HC1",
            coefficients=result.coefficients,
            existing_result=result,
        )
        # _fetch_suffstats should NOT have been called (XtX already cached)
        spy.assert_not_called()


# ---------------------------------------------------------------------------
# ── 10. Cross-backend coefficient consistency ───────────────────────────────
# ---------------------------------------------------------------------------


class TestCrossBackendCoefficients:
    """Parametrised check that NumPy and DuckDB give identical theta."""

    @pytest.fixture
    def conn(self):
        c = _make_conn()
        yield c
        c.close()

    @pytest.mark.parametrize("n,k,seed", [
        (20, 2, 10),
        (100, 4, 11),
        (500, 5, 12),
    ])
    def test_theta_agrees(self, conn, n, k, seed):
        rng = np.random.default_rng(seed)
        X_raw = rng.standard_normal((n, k))
        true_theta = rng.standard_normal(k + 1)
        X_full = np.column_stack([np.ones(n), X_raw])
        y = X_full @ true_theta + 0.2 * rng.standard_normal(n)

        # Build table
        col_vals = {f"x{i}": X_raw[:, i].tolist() for i in range(k)}
        col_vals["sum_y"] = y.tolist()
        col_vals["sum_y_sq"] = (y ** 2).tolist()
        col_vals["count"] = [1] * n
        conn.execute("DROP TABLE IF EXISTS cross_t")
        select_parts = ", ".join(
            f"unnest({v!r}) AS {c}" for c, v in col_vals.items()
        )
        conn.execute(f"CREATE TABLE cross_t AS SELECT {select_parts}")

        x_cols = [f"x{i}" for i in range(k)]

        db_fitter = DuckDBFitter(conn=conn, alpha=0.0)
        db_result = db_fitter.fit(table_name="cross_t", x_cols=x_cols,
                                  y_col="sum_y", add_intercept=True)

        np_fitter = NumpyFitter(alpha=0.0)
        np_result = np_fitter.fit(X=X_full, y=y, weights=np.ones(n),
                                  coef_names=["Intercept"] + x_cols)

        assert np.allclose(db_result.coefficients, np_result.coefficients,
                           atol=1e-8, rtol=1e-6), (
            f"Backends disagree: db={db_result.coefficients}, "
            f"np={np_result.coefficients}"
        )


# ---------------------------------------------------------------------------
# ── 11. get_fitter factory ──────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestGetFitter:

    def test_numpy_fitter_returned(self):
        f = get_fitter("numpy")
        assert isinstance(f, NumpyFitter)

    def test_duckdb_fitter_returned(self):
        conn = _make_conn()
        f = get_fitter("duckdb", conn=conn)
        assert isinstance(f, DuckDBFitter)
        conn.close()

    def test_duckdb_without_conn_raises(self):
        with pytest.raises(ValueError, match="conn must be provided"):
            get_fitter("duckdb")

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown fitter type"):
            get_fitter("spark")

    def test_alpha_passed_through(self):
        f = get_fitter("numpy", alpha=0.1)
        assert f.alpha == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# ── 12. wls_duckdb convenience function ─────────────────────────────────────
# ---------------------------------------------------------------------------


class TestWlsDuckdb:

    @pytest.fixture
    def conn(self):
        c = _make_conn()
        yield c
        c.close()

    @pytest.fixture
    def table(self, conn):
        _seed_table(conn, "wls_t",
                    x1=[1.0, 2.0, 3.0, 4.0, 5.0],
                    x2=[5.0, 4.0, 3.0, 2.0, 1.0],
                    y=[2.0, 4.0, 6.0, 8.0, 10.0])
        return "wls_t"

    def test_returns_dict(self, conn, table):
        result = wls_duckdb(conn, table, ["x1", "x2"], "sum_y")
        assert isinstance(result, dict)

    def test_has_coefficients(self, conn, table):
        result = wls_duckdb(conn, table, ["x1", "x2"], "sum_y")
        assert "coefficients" in result
        assert isinstance(result["coefficients"], np.ndarray)

    def test_has_vcov(self, conn, table):
        result = wls_duckdb(conn, table, ["x1", "x2"], "sum_y")
        assert "vcov" in result
        assert result["vcov"] is not None

    def test_has_standard_errors(self, conn, table):
        result = wls_duckdb(conn, table, ["x1", "x2"], "sum_y")
        assert "standard_errors" in result

    def test_has_n_obs(self, conn, table):
        result = wls_duckdb(conn, table, ["x1", "x2"], "sum_y")
        assert result["n_obs"] == 5

    def test_perfect_fit_high_r_squared(self, conn, table):
        result = wls_duckdb(conn, table, ["x1"], "sum_y",
                            add_intercept=True)
        assert result.get("r_squared", 0) > 0.9
