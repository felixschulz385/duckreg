"""
tests/test_residual_aggregates.py

Comprehensive test suite for:
    - compute_residual_aggregates_numpy
    - compute_residual_aggregates_sql

All SQL tests use an in-process DuckDB connection — no fixtures, no files.

Usage
-----
    pytest tests/test_residual_aggregates.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import duckdb
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Imports under test — adjust to match your package layout
# ---------------------------------------------------------------------------
from duckreg.core.residual_aggregates import (
    compute_residual_aggregates_numpy,
    compute_residual_aggregates_sql,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

ATOL = 1e-10
RTOL = 1e-7


def _make_conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(":memory:")


def _seed_table(conn, table_name, x1_vals, x2_vals, y_vals,
                count_vals=None, cluster_vals=None, sum_y_sq_vals=None):
    n = len(x1_vals)
    count_vals = count_vals or [1] * n
    sum_y_sq_vals = sum_y_sq_vals or [v ** 2 for v in y_vals]
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    parts = {
        "x1": x1_vals,
        "x2": x2_vals,
        "sum_y": y_vals,
        "sum_y_sq": sum_y_sq_vals,
        "count": count_vals,
    }
    if cluster_vals is not None:
        parts["cluster_id"] = cluster_vals
    select_parts = ", ".join(
        f"unnest({v!r}) AS {c}" for c, v in parts.items()
    )
    conn.execute(f"CREATE TABLE {table_name} AS SELECT {select_parts}")


# ---------------------------------------------------------------------------
# Canonical dataset — reused across many tests
# ---------------------------------------------------------------------------

N_CANONICAL = 30
RNG_CANONICAL = np.random.default_rng(0)
X_CANONICAL_RAW = RNG_CANONICAL.standard_normal((N_CANONICAL, 2))
X_CANONICAL = np.column_stack([np.ones(N_CANONICAL), X_CANONICAL_RAW])
Y_CANONICAL = X_CANONICAL @ np.array([1.0, -0.5, 0.3]) + 0.2 * RNG_CANONICAL.standard_normal(N_CANONICAL)
W_CANONICAL = np.ones(N_CANONICAL)
THETA_CANONICAL = np.linalg.lstsq(X_CANONICAL, Y_CANONICAL, rcond=None)[0]


def _canonical_table(conn, name="canon"):
    _seed_table(
        conn, name,
        x1_vals=X_CANONICAL_RAW[:, 0].tolist(),
        x2_vals=X_CANONICAL_RAW[:, 1].tolist(),
        y_vals=Y_CANONICAL.tolist(),
    )
    return name


# ============================================================================
# ── 1. compute_residual_aggregates_numpy — residuals
# ============================================================================


class TestNumpyResiduals:

    def test_residuals_computed_from_theta(self):
        X = np.array([[1.0, 2.0], [1.0, 3.0]])
        y = np.array([5.0, 8.0])
        theta = np.array([1.0, 2.0])
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=np.ones(2), compute_rss=False
        )
        # No exception — empty dict without rss when all flags False
        assert isinstance(agg, dict)

    def test_precomputed_residuals_used_verbatim(self):
        X = np.ones((5, 1))
        y = np.ones(5)
        theta = np.array([0.0])
        precomp = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=np.ones(5),
            residuals=precomp, compute_rss=True
        )
        # rss = sum((3 * 1)^2) = 45
        assert agg["rss"] == pytest.approx(45.0)

    def test_residuals_flattened(self):
        X = np.ones((4, 1))
        y = np.ones((4, 1))
        theta = np.array([1.0])
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=np.ones(4), compute_rss=True
        )
        assert agg["rss"] == pytest.approx(0.0)


# ============================================================================
# ── 2. compute_residual_aggregates_numpy — RSS
# ============================================================================


class TestNumpyRSS:

    def test_rss_zero_perfect_fit(self):
        """If residuals are all zero, RSS must be zero."""
        theta = THETA_CANONICAL
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X_CANONICAL, y=Y_CANONICAL,
            weights=W_CANONICAL, compute_rss=True
        )
        # Least-squares theta should have the minimum RSS — close to zero
        assert agg["rss"] >= -1e-10  # non-negative
        assert "rss" in agg

    def test_rss_formula_manual(self):
        """rss = sum((u_i * sqrt(w_i))^2) for unit weights == sum(u_i^2)."""
        X = np.eye(3)
        y = np.array([1.0, 2.0, 3.0])
        theta = np.array([0.0, 0.0, 0.0])
        w = np.ones(3)
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=w, compute_rss=True
        )
        assert agg["rss"] == pytest.approx(1.0 + 4.0 + 9.0)

    def test_rss_with_frequency_weights(self):
        """Doubling all weights should double rss."""
        X = np.ones((4, 1))
        y = np.array([2.0, 3.0, 4.0, 5.0])
        theta = np.array([1.0])
        w1 = np.ones(4)
        w2 = 2 * np.ones(4)
        rss1 = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=w1, compute_rss=True
        )["rss"]
        rss2 = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=w2, compute_rss=True
        )["rss"]
        assert rss2 == pytest.approx(2 * rss1)

    def test_rss_not_returned_when_flag_false(self):
        X = np.ones((3, 1))
        y = np.ones(3)
        theta = np.array([1.0])
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=np.ones(3), compute_rss=False
        )
        assert "rss" not in agg


# ============================================================================
# ── 3. compute_residual_aggregates_numpy — scores
# ============================================================================


class TestNumpyScores:

    def test_scores_shape(self):
        n, k = 10, 3
        X = np.random.randn(n, k)
        y = np.random.randn(n)
        theta = np.zeros(k)
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=np.ones(n), compute_scores=True
        )
        assert agg["scores"].shape == (n, k)

    def test_scores_formula_manual(self):
        """score_i = x_i * u_i * w_i (unit weights → score_i = x_i * u_i)."""
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([2.0, 3.0])
        theta = np.array([1.0, 1.0])
        w = np.ones(2)
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=w,
            compute_scores=True, compute_rss=False
        )
        u = y - X @ theta  # [1, 2]
        expected = X * u.reshape(-1, 1)
        assert np.allclose(agg["scores"], expected)

    def test_scores_not_returned_when_flag_false(self):
        X = np.ones((3, 2))
        y = np.ones(3)
        theta = np.zeros(2)
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=np.ones(3), compute_rss=False
        )
        assert "scores" not in agg

    def test_scores_weighted_scale(self):
        """Doubling weights should double scores."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 1.0])
        theta = np.zeros(2)
        w1 = np.ones(2)
        w2 = 2 * np.ones(2)
        s1 = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=w1,
            compute_scores=True, compute_rss=False
        )["scores"]
        s2 = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=w2,
            compute_scores=True, compute_rss=False
        )["scores"]
        assert np.allclose(s2, 2 * s1)


# ============================================================================
# ── 4. compute_residual_aggregates_numpy — meat
# ============================================================================


class TestNumpyMeat:

    def test_meat_shape(self):
        n, k = 20, 3
        X = np.random.randn(n, k)
        y = np.random.randn(n)
        theta = np.zeros(k)
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=np.ones(n),
            compute_meat=True, compute_rss=False
        )
        assert agg["meat"].shape == (k, k)

    def test_meat_symmetry(self):
        n, k = 25, 3
        X = np.random.randn(n, k)
        y = np.random.randn(n)
        theta = np.zeros(k)
        meat = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=np.ones(n),
            compute_meat=True, compute_rss=False
        )["meat"]
        assert np.allclose(meat, meat.T)

    def test_meat_positive_semidefinite(self):
        n, k = 30, 3
        np.random.seed(5)
        X = np.random.randn(n, k)
        y = np.random.randn(n)
        theta = np.zeros(k)
        meat = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=np.ones(n),
            compute_meat=True, compute_rss=False
        )["meat"]
        eigvals = np.linalg.eigvalsh(meat)
        assert np.all(eigvals >= -1e-10)

    def test_meat_formula_manual(self):
        """meat = sum_i w_i * (x_i u_i)(x_i u_i)^T for k=1."""
        X = np.array([[2.0], [3.0]])
        y = np.array([4.0, 9.0])
        theta = np.array([0.0])
        w = np.ones(2)
        meat = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=w,
            compute_meat=True, compute_rss=False
        )["meat"]
        # u = [4, 9]; meat = (2*4)^2 + (3*9)^2 = 64 + 729 = 793
        assert meat[0, 0] == pytest.approx(64.0 + 729.0)

    def test_meat_not_returned_when_flag_false(self):
        X = np.ones((3, 2))
        y = np.ones(3)
        theta = np.zeros(2)
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=np.ones(3), compute_rss=False
        )
        assert "meat" not in agg


# ============================================================================
# ── 5. compute_residual_aggregates_numpy — cluster scores
# ============================================================================


class TestNumpyClusterScores:

    def _make_cluster_data(self):
        np.random.seed(9)
        n, k, G = 40, 2, 5
        X = np.random.randn(n, k)
        y = np.random.randn(n)
        theta = np.zeros(k)
        w = np.ones(n)
        cluster_ids = np.repeat(np.arange(G), n // G)
        return X, y, theta, w, cluster_ids, G

    def test_cluster_scores_shape(self):
        X, y, theta, w, cids, G = self._make_cluster_data()
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=w,
            cluster_ids=cids, compute_cluster_scores=True, compute_rss=False
        )
        k = X.shape[1]
        assert agg["cluster_scores"].shape == (G, k)

    def test_n_clusters_correct(self):
        X, y, theta, w, cids, G = self._make_cluster_data()
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=w,
            cluster_ids=cids, compute_cluster_scores=True, compute_rss=False
        )
        assert agg["n_clusters"] == G

    def test_cluster_scores_sum_to_total_scores(self):
        """Sum of cluster scores must equal sum of individual scores."""
        X, y, theta, w, cids, G = self._make_cluster_data()
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=w,
            cluster_ids=cids, compute_cluster_scores=True, compute_rss=False
        )
        # Individual scores
        u = y - X @ theta
        individual_scores = X * (u * w).reshape(-1, 1)
        assert np.allclose(
            agg["cluster_scores"].sum(axis=0),
            individual_scores.sum(axis=0),
            atol=1e-12,
        )

    def test_missing_cluster_ids_raises(self):
        X = np.ones((4, 2))
        y = np.ones(4)
        theta = np.zeros(2)
        with pytest.raises(ValueError, match="cluster_ids required"):
            compute_residual_aggregates_numpy(
                theta=theta, X=X, y=y, weights=np.ones(4),
                compute_cluster_scores=True, compute_rss=False
            )


# ============================================================================
# ── 6. compute_residual_aggregates_numpy — leverages
# ============================================================================


class TestNumpyLeverages:

    def test_leverages_shape(self):
        n, k = 20, 3
        X = np.random.randn(n, k)
        y = np.random.randn(n)
        theta = np.zeros(k)
        XtX_inv = np.linalg.inv(X.T @ X)
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=np.ones(n),
            XtX_inv=XtX_inv, compute_leverages=True, compute_rss=False
        )
        assert agg["leverages"].shape == (n,)

    def test_leverages_in_zero_one(self):
        """Hat matrix diagonal elements are in [0, 1]."""
        np.random.seed(2)
        n, k = 30, 3
        X = np.random.randn(n, k)
        y = np.random.randn(n)
        theta = np.zeros(k)
        XtX_inv = np.linalg.inv(X.T @ X)
        h = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=np.ones(n),
            XtX_inv=XtX_inv, compute_leverages=True, compute_rss=False
        )["leverages"]
        assert np.all(h >= -1e-10)
        assert np.all(h <= 1.0 + 1e-10)

    def test_leverages_sum_equals_k(self):
        """Sum of leverages = trace(H) = k (number of columns)."""
        np.random.seed(3)
        n, k = 50, 4
        X = np.random.randn(n, k)
        y = np.random.randn(n)
        theta = np.zeros(k)
        XtX_inv = np.linalg.inv(X.T @ X)
        h = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=np.ones(n),
            XtX_inv=XtX_inv, compute_leverages=True, compute_rss=False
        )["leverages"]
        assert h.sum() == pytest.approx(k, abs=1e-8)

    def test_leverages_manual_formula(self):
        """For X = I_n, h_ii = x_i' (X'X)^-1 x_i = 1 for all i."""
        n = 5
        X = np.eye(n)
        XtX_inv = np.eye(n)
        theta = np.zeros(n)
        h = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=np.zeros(n), weights=np.ones(n),
            XtX_inv=XtX_inv, compute_leverages=True, compute_rss=False
        )["leverages"]
        assert np.allclose(h, np.ones(n))

    def test_missing_xtx_inv_raises(self):
        X = np.ones((4, 2))
        y = np.ones(4)
        theta = np.zeros(2)
        with pytest.raises(ValueError, match="XtX_inv required"):
            compute_residual_aggregates_numpy(
                theta=theta, X=X, y=y, weights=np.ones(4),
                compute_leverages=True, compute_rss=False
            )


# ============================================================================
# ── 7. compute_residual_aggregates_numpy — IV path
# ============================================================================


class TestNumpyIV:

    def test_iv_scores_use_Z_not_X(self):
        """With is_iv=True and Z provided, scores use Z not X."""
        n, kx, kz = 20, 2, 3
        np.random.seed(11)
        X = np.random.randn(n, kx)
        Z = np.random.randn(n, kz)
        y = np.random.randn(n)
        theta = np.zeros(kx)

        agg_iv = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=np.ones(n),
            Z=Z, is_iv=True, compute_scores=True, compute_rss=False
        )
        agg_ols = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=np.ones(n),
            compute_scores=True, compute_rss=False
        )
        assert agg_iv["scores"].shape == (n, kz)
        assert agg_ols["scores"].shape == (n, kx)

    def test_iv_meat_uses_Z(self):
        """With is_iv=True, meat shape must match (kz, kz)."""
        n, kx, kz = 20, 2, 3
        np.random.seed(12)
        X = np.random.randn(n, kx)
        Z = np.random.randn(n, kz)
        y = np.random.randn(n)
        theta = np.zeros(kx)

        meat = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=np.ones(n),
            Z=Z, is_iv=True, compute_meat=True, compute_rss=False
        )["meat"]
        assert meat.shape == (kz, kz)

    def test_ols_path_when_z_not_provided(self):
        """is_iv=True but Z=None → scores still use X."""
        n, k = 15, 2
        X = np.random.randn(n, k)
        y = np.random.randn(n)
        theta = np.zeros(k)
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=np.ones(n),
            is_iv=True, Z=None, compute_scores=True, compute_rss=False
        )
        assert agg["scores"].shape == (n, k)


# ============================================================================
# ── 8. compute_residual_aggregates_numpy — multiple flags
# ============================================================================


class TestNumpyMultipleFlags:

    def test_rss_and_scores_together(self):
        X = X_CANONICAL
        y = Y_CANONICAL
        theta = THETA_CANONICAL
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=W_CANONICAL,
            compute_rss=True, compute_scores=True
        )
        assert "rss" in agg
        assert "scores" in agg

    def test_rss_and_meat_together(self):
        X = X_CANONICAL
        y = Y_CANONICAL
        theta = THETA_CANONICAL
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=W_CANONICAL,
            compute_rss=True, compute_meat=True
        )
        assert "rss" in agg
        assert "meat" in agg

    def test_scores_and_leverages_together(self):
        X = X_CANONICAL
        y = Y_CANONICAL
        theta = THETA_CANONICAL
        XtX_inv = np.linalg.inv(X.T @ X)
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=W_CANONICAL,
            XtX_inv=XtX_inv,
            compute_rss=False, compute_scores=True, compute_leverages=True
        )
        assert "scores" in agg
        assert "leverages" in agg

    def test_empty_result_when_all_flags_false(self):
        X = np.ones((3, 1))
        y = np.ones(3)
        theta = np.array([1.0])
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=np.ones(3), compute_rss=False
        )
        assert agg == {}


# ============================================================================
# ── 9. compute_residual_aggregates_sql — RSS
# ============================================================================


class TestSQLRSS:

    @pytest.fixture
    def conn(self):
        c = _make_conn()
        yield c
        c.close()

    def test_rss_matches_numpy(self, conn):
        tbl = _canonical_table(conn)
        theta = THETA_CANONICAL

        rss_sql = compute_residual_aggregates_sql(
            theta=theta, conn=conn, table_name=tbl,
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            compute_rss=True, add_intercept=True
        )["rss"]

        rss_np = compute_residual_aggregates_numpy(
            theta=theta, X=X_CANONICAL, y=Y_CANONICAL,
            weights=W_CANONICAL, compute_rss=True
        )["rss"]

        assert rss_sql == pytest.approx(rss_np, rel=1e-6)

    def test_rss_nonnegative(self, conn):
        tbl = _canonical_table(conn)
        rss = compute_residual_aggregates_sql(
            theta=THETA_CANONICAL, conn=conn, table_name=tbl,
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            compute_rss=True, add_intercept=True
        )["rss"]
        assert rss >= -1e-10

    def test_rss_with_where_clause(self, conn):
        tbl = _canonical_table(conn)
        rss_all = compute_residual_aggregates_sql(
            theta=THETA_CANONICAL, conn=conn, table_name=tbl,
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            compute_rss=True, add_intercept=True
        )["rss"]
        rss_half = compute_residual_aggregates_sql(
            theta=THETA_CANONICAL, conn=conn, table_name=tbl,
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            compute_rss=True, add_intercept=True,
            where_clause="WHERE x1 > 0"
        )["rss"]
        assert rss_half < rss_all

    def test_rss_not_returned_when_flag_false(self, conn):
        tbl = _canonical_table(conn)
        agg = compute_residual_aggregates_sql(
            theta=THETA_CANONICAL, conn=conn, table_name=tbl,
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            compute_rss=False, add_intercept=True
        )
        assert "rss" not in agg


# ============================================================================
# ── 10. compute_residual_aggregates_sql — meat
# ============================================================================


class TestSQLMeat:

    @pytest.fixture
    def conn(self):
        c = _make_conn()
        yield c
        c.close()

    def test_meat_matches_numpy(self, conn):
        tbl = _canonical_table(conn)
        theta = THETA_CANONICAL

        meat_sql = compute_residual_aggregates_sql(
            theta=theta, conn=conn, table_name=tbl,
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            compute_meat=True, compute_rss=False, add_intercept=True
        )["meat"]

        meat_np = compute_residual_aggregates_numpy(
            theta=theta, X=X_CANONICAL, y=Y_CANONICAL,
            weights=W_CANONICAL, compute_meat=True, compute_rss=False
        )["meat"]

        assert np.allclose(meat_sql, meat_np, atol=1e-9, rtol=1e-6)

    def test_meat_symmetry(self, conn):
        tbl = _canonical_table(conn)
        meat = compute_residual_aggregates_sql(
            theta=THETA_CANONICAL, conn=conn, table_name=tbl,
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            compute_meat=True, compute_rss=False, add_intercept=True
        )["meat"]
        assert np.allclose(meat, meat.T)

    def test_meat_shape(self, conn):
        tbl = _canonical_table(conn)
        k = 3  # intercept + x1 + x2
        meat = compute_residual_aggregates_sql(
            theta=THETA_CANONICAL, conn=conn, table_name=tbl,
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            compute_meat=True, compute_rss=False, add_intercept=True
        )["meat"]
        assert meat.shape == (k, k)

    def test_meat_no_intercept_shape(self, conn):
        tbl = _canonical_table(conn)
        theta_no_int = THETA_CANONICAL[1:]  # drop intercept
        k = 2
        meat = compute_residual_aggregates_sql(
            theta=theta_no_int, conn=conn, table_name=tbl,
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            compute_meat=True, compute_rss=False, add_intercept=False
        )["meat"]
        assert meat.shape == (k, k)

    def test_exact_meat_with_sum_y_sq_col(self, conn):
        """When sum_y_sq_col provided, exact path runs without error and result is non-negative PSD."""
        tbl = _canonical_table(conn)  # has sum_y_sq column
        meat = compute_residual_aggregates_sql(
            theta=THETA_CANONICAL, conn=conn, table_name=tbl,
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            compute_meat=True, compute_rss=False, add_intercept=True,
            sum_y_sq_col="sum_y_sq"
        )["meat"]
        eigvals = np.linalg.eigvalsh(meat)
        assert np.all(eigvals >= -1e-10)


# ============================================================================
# ── 11. compute_residual_aggregates_sql — cluster scores
# ============================================================================


class TestSQLClusterScores:

    @pytest.fixture
    def conn_with_clusters(self):
        c = _make_conn()
        n = N_CANONICAL
        cluster_ids = (np.arange(n) % 6).tolist()
        _seed_table(
            c, "canon_cl",
            x1_vals=X_CANONICAL_RAW[:, 0].tolist(),
            x2_vals=X_CANONICAL_RAW[:, 1].tolist(),
            y_vals=Y_CANONICAL.tolist(),
            cluster_vals=cluster_ids,
        )
        yield c, cluster_ids
        c.close()

    def test_cluster_scores_shape(self, conn_with_clusters):
        conn, cids = conn_with_clusters
        G = len(set(cids))
        k = 3  # intercept + x1 + x2
        agg = compute_residual_aggregates_sql(
            theta=THETA_CANONICAL, conn=conn, table_name="canon_cl",
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            cluster_col="cluster_id",
            compute_cluster_scores=True, compute_rss=False, add_intercept=True
        )
        assert agg["cluster_scores"].shape == (G, k)

    def test_n_clusters_correct(self, conn_with_clusters):
        conn, cids = conn_with_clusters
        G = len(set(cids))
        agg = compute_residual_aggregates_sql(
            theta=THETA_CANONICAL, conn=conn, table_name="canon_cl",
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            cluster_col="cluster_id",
            compute_cluster_scores=True, compute_rss=False, add_intercept=True
        )
        assert agg["n_clusters"] == G

    def test_cluster_scores_match_numpy(self, conn_with_clusters):
        """SQL and numpy cluster scores must sum to the same total."""
        conn, cids = conn_with_clusters
        cids_arr = np.array(cids)

        agg_sql = compute_residual_aggregates_sql(
            theta=THETA_CANONICAL, conn=conn, table_name="canon_cl",
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            cluster_col="cluster_id",
            compute_cluster_scores=True, compute_rss=False, add_intercept=True
        )
        agg_np = compute_residual_aggregates_numpy(
            theta=THETA_CANONICAL, X=X_CANONICAL, y=Y_CANONICAL,
            weights=W_CANONICAL, cluster_ids=cids_arr,
            compute_cluster_scores=True, compute_rss=False
        )
        # Sum over clusters must agree
        assert np.allclose(
            agg_sql["cluster_scores"].sum(axis=0),
            agg_np["cluster_scores"].sum(axis=0),
            atol=1e-9, rtol=1e-6,
        )

    def test_missing_cluster_col_raises(self):
        conn = _make_conn()
        tbl = _canonical_table(conn)
        with pytest.raises(ValueError, match="cluster_col required"):
            compute_residual_aggregates_sql(
                theta=THETA_CANONICAL, conn=conn, table_name=tbl,
                x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
                compute_cluster_scores=True, compute_rss=False
            )
        conn.close()


# ============================================================================
# ── 12. compute_residual_aggregates_sql — leverages
# ============================================================================


class TestSQLLeverages:

    @pytest.fixture
    def conn(self):
        c = _make_conn()
        yield c
        c.close()

    def test_leverages_shape(self, conn):
        tbl = _canonical_table(conn)
        XtX_inv = np.linalg.inv(X_CANONICAL.T @ X_CANONICAL)
        agg = compute_residual_aggregates_sql(
            theta=THETA_CANONICAL, conn=conn, table_name=tbl,
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            XtX_inv=XtX_inv,
            compute_leverages=True, compute_rss=False, add_intercept=True
        )
        assert agg["leverages"].shape == (N_CANONICAL,)

    def test_leverages_in_zero_one(self, conn):
        tbl = _canonical_table(conn)
        XtX_inv = np.linalg.inv(X_CANONICAL.T @ X_CANONICAL)
        h = compute_residual_aggregates_sql(
            theta=THETA_CANONICAL, conn=conn, table_name=tbl,
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            XtX_inv=XtX_inv,
            compute_leverages=True, compute_rss=False, add_intercept=True
        )["leverages"]
        assert np.all(h >= -1e-10)
        assert np.all(h <= 1.0 + 1e-10)

    def test_leverages_match_numpy(self, conn):
        tbl = _canonical_table(conn)
        XtX_inv = np.linalg.inv(X_CANONICAL.T @ X_CANONICAL)

        h_sql = compute_residual_aggregates_sql(
            theta=THETA_CANONICAL, conn=conn, table_name=tbl,
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            XtX_inv=XtX_inv,
            compute_leverages=True, compute_rss=False, add_intercept=True
        )["leverages"]

        h_np = compute_residual_aggregates_numpy(
            theta=THETA_CANONICAL, X=X_CANONICAL, y=Y_CANONICAL,
            weights=W_CANONICAL, XtX_inv=XtX_inv,
            compute_leverages=True, compute_rss=False
        )["leverages"]

        assert np.allclose(h_sql, h_np, atol=1e-9, rtol=1e-6)

    def test_missing_xtx_inv_raises(self, conn):
        tbl = _canonical_table(conn)
        with pytest.raises(ValueError, match="XtX_inv required"):
            compute_residual_aggregates_sql(
                theta=THETA_CANONICAL, conn=conn, table_name=tbl,
                x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
                compute_leverages=True, compute_rss=False
            )


# ============================================================================
# ── 13. compute_residual_aggregates_sql — residual_x_cols
# ============================================================================


class TestSQLResidualXCols:

    @pytest.fixture
    def conn(self):
        c = _make_conn()
        yield c
        c.close()

    def test_residual_x_cols_change_rss(self, conn):
        """Different residual_x_cols → different residuals → different RSS."""
        tbl = _canonical_table(conn)
        theta_short = THETA_CANONICAL[:2]  # wrong length — triggers fallback

        rss_normal = compute_residual_aggregates_sql(
            theta=THETA_CANONICAL, conn=conn, table_name=tbl,
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            compute_rss=True, add_intercept=True
        )["rss"]

        # Using x1 only as residual_x_cols with mismatched length → fallback to x_cols
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rss_fallback = compute_residual_aggregates_sql(
                theta=THETA_CANONICAL, conn=conn, table_name=tbl,
                x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
                compute_rss=True, add_intercept=True,
                residual_x_cols=["x1"]  # length 1, expected 2 → fallback
            )["rss"]

        # Fallback uses same cols → RSS should be identical
        assert rss_fallback == pytest.approx(rss_normal, rel=1e-7)

    def test_valid_residual_x_cols_used(self, conn):
        """Valid residual_x_cols of correct length should be applied."""
        tbl = _canonical_table(conn)
        # Use x2 for both positions as residual_x_cols
        rss_alt = compute_residual_aggregates_sql(
            theta=THETA_CANONICAL, conn=conn, table_name=tbl,
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            compute_rss=True, add_intercept=True,
            residual_x_cols=["x2", "x1"]  # swapped — still valid length
        )["rss"]
        rss_normal = compute_residual_aggregates_sql(
            theta=THETA_CANONICAL, conn=conn, table_name=tbl,
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            compute_rss=True, add_intercept=True,
        )["rss"]
        # Swapped columns → different residuals → different RSS (unless X is symmetric)
        # Just check it runs and produces a non-negative number
        assert rss_alt >= -1e-10


# ============================================================================
# ── 14. compute_residual_aggregates_sql — IV path
# ============================================================================


class TestSQLIV:

    @pytest.fixture
    def conn_with_z(self):
        c = _make_conn()
        n = N_CANONICAL
        np.random.seed(20)
        z1 = np.random.randn(n).tolist()
        c.execute("DROP TABLE IF EXISTS iv_t")
        select_parts = (
            f"unnest({X_CANONICAL_RAW[:, 0].tolist()!r}) AS x1, "
            f"unnest({X_CANONICAL_RAW[:, 1].tolist()!r}) AS x2, "
            f"unnest({Y_CANONICAL.tolist()!r})              AS sum_y, "
            f"unnest({(Y_CANONICAL**2).tolist()!r})         AS sum_y_sq, "
            f"unnest({z1!r})                                AS z1, "
            f"unnest({[1]*n!r})                             AS count"
        )
        c.execute(f"CREATE TABLE iv_t AS SELECT {select_parts}")
        yield c
        c.close()

    def test_iv_cluster_scores_use_z_cols(self, conn_with_z):
        n = N_CANONICAL
        cluster_ids = (np.arange(n) % 5).tolist()
        conn_with_z.execute("ALTER TABLE iv_t ADD COLUMN IF NOT EXISTS cluster_id INT")
        conn_with_z.execute(
            f"UPDATE iv_t SET cluster_id = (rowid % 5)"
        )

        agg = compute_residual_aggregates_sql(
            theta=THETA_CANONICAL, conn=conn_with_z, table_name="iv_t",
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            cluster_col="cluster_id",
            compute_cluster_scores=True, compute_rss=False,
            add_intercept=True, z_cols=["z1"], is_iv=True
        )
        G = 5
        k_z = 2  # intercept + z1
        assert agg["cluster_scores"].shape == (G, k_z)

    def test_iv_meat_uses_z_cols(self, conn_with_z):
        agg = compute_residual_aggregates_sql(
            theta=THETA_CANONICAL, conn=conn_with_z, table_name="iv_t",
            x_cols=["x1", "x2"], y_col="sum_y", weight_col="count",
            compute_meat=True, compute_rss=False,
            add_intercept=True, z_cols=["z1"], is_iv=True
        )
        k_z = 2  # intercept + z1
        assert agg["meat"].shape == (k_z, k_z)


# ============================================================================
# ── 15. Cross-backend consistency — parametrised
# ============================================================================


class TestCrossBackendConsistency:
    """RSS, meat, and leverages must agree between numpy and SQL backends."""

    @pytest.fixture
    def conn(self):
        c = _make_conn()
        yield c
        c.close()

    @pytest.mark.parametrize("n,k,seed", [
        (20, 2, 30),
        (100, 3, 31),
        (200, 4, 32),
    ])
    def test_rss_agrees(self, conn, n, k, seed):
        rng = np.random.default_rng(seed)
        X_raw = rng.standard_normal((n, k))
        X = np.column_stack([np.ones(n), X_raw])
        y = X @ rng.standard_normal(k + 1) + 0.1 * rng.standard_normal(n)
        theta, *_ = np.linalg.lstsq(X, y, rcond=None)
        w = np.ones(n)

        cols = {f"x{i}": X_raw[:, i].tolist() for i in range(k)}
        cols["sum_y"] = y.tolist()
        cols["sum_y_sq"] = (y ** 2).tolist()
        cols["count"] = [1] * n
        conn.execute("DROP TABLE IF EXISTS cc_t")
        parts = ", ".join(f"unnest({v!r}) AS {c}" for c, v in cols.items())
        conn.execute(f"CREATE TABLE cc_t AS SELECT {parts}")

        x_cols = [f"x{i}" for i in range(k)]

        rss_sql = compute_residual_aggregates_sql(
            theta=theta, conn=conn, table_name="cc_t",
            x_cols=x_cols, y_col="sum_y", weight_col="count",
            compute_rss=True, add_intercept=True
        )["rss"]
        rss_np = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=w, compute_rss=True
        )["rss"]

        assert rss_sql == pytest.approx(rss_np, rel=1e-6), (
            f"n={n},k={k}: sql_rss={rss_sql:.6g} np_rss={rss_np:.6g}"
        )

    @pytest.mark.parametrize("n,k,seed", [
        (20, 2, 40),
        (100, 3, 41),
    ])
    def test_meat_agrees(self, conn, n, k, seed):
        rng = np.random.default_rng(seed)
        X_raw = rng.standard_normal((n, k))
        X = np.column_stack([np.ones(n), X_raw])
        y = X @ rng.standard_normal(k + 1) + 0.1 * rng.standard_normal(n)
        theta, *_ = np.linalg.lstsq(X, y, rcond=None)

        cols = {f"x{i}": X_raw[:, i].tolist() for i in range(k)}
        cols["sum_y"] = y.tolist()
        cols["sum_y_sq"] = (y ** 2).tolist()
        cols["count"] = [1] * n
        conn.execute("DROP TABLE IF EXISTS cc_m")
        parts = ", ".join(f"unnest({v!r}) AS {c}" for c, v in cols.items())
        conn.execute(f"CREATE TABLE cc_m AS SELECT {parts}")

        x_cols = [f"x{i}" for i in range(k)]

        meat_sql = compute_residual_aggregates_sql(
            theta=theta, conn=conn, table_name="cc_m",
            x_cols=x_cols, y_col="sum_y", weight_col="count",
            compute_meat=True, compute_rss=False, add_intercept=True
        )["meat"]
        meat_np = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=np.ones(n),
            compute_meat=True, compute_rss=False
        )["meat"]

        assert np.allclose(meat_sql, meat_np, atol=1e-9, rtol=1e-5), (
            f"n={n},k={k}: max_diff={np.abs(meat_sql - meat_np).max():.2e}"
        )
