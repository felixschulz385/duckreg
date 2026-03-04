"""
tests/test_suffstats.py

Comprehensive test suite for suffstats.py.

Covers:
    - SuffStats dataclass (construction, iteration/unpacking, sum_y_sq_exact flag)
    - compute_sufficient_stats_numpy  (correctness, edge cases, validation)
    - compute_sufficient_stats_sql    (correctness, DRY delegation, sum_y_sq_col
                                       branch, where_clause, validation)
    - execute_to_matrix               (full matrix, upper-triangle, validation)
    - compute_cross_sufficient_stats_sql (tXZ / tZZ shapes, symmetry, validation)
    - profile_fe_column               (cardinality, singleton share)
    - get_fe_unique_levels            (happy path, max_levels guard)

All SQL tests use an in-process DuckDB connection — no external fixtures needed.

Usage
-----
    pytest tests/test_suffstats.py -v
"""

from __future__ import annotations

import math
from typing import List

import duckdb
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# The module under test — adjust the import path to match your package layout
# ---------------------------------------------------------------------------
from duckreg.core.suffstats import (
    SuffStats,
    compute_cross_sufficient_stats_sql,
    compute_sufficient_stats_numpy,
    compute_sufficient_stats_sql,
    execute_to_matrix,
    get_fe_unique_levels,
    profile_fe_column,
)

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

DEFAULT_ALPHA = 1e-8
ATOL = 1e-9   # absolute tolerance for floating-point comparisons
RTOL = 1e-7   # relative tolerance


def _make_conn() -> duckdb.DuckDBPyConnection:
    """Return a fresh in-process DuckDB connection."""
    return duckdb.connect(":memory:")


def _seed_table(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    x1: List[float],
    x2: List[float],
    y: List[float],
    count: List[int] | None = None,
) -> None:
    """Create a simple table with columns x1, x2, sum_y, count (and sum_y_sq)."""
    n = len(x1)
    if count is None:
        count = [1] * n
    sum_y = y          # for unit-weight rows, sum_y == y
    sum_y_sq = [yi ** 2 for yi in y]

    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(
        f"""
        CREATE TABLE {table_name} AS
        SELECT
            unnest({x1!r}) AS x1,
            unnest({x2!r}) AS x2,
            unnest({sum_y!r}) AS sum_y,
            unnest({sum_y_sq!r}) AS sum_y_sq,
            unnest({count!r}) AS count
        """
    )


# ---------------------------------------------------------------------------
# ── 1. SuffStats dataclass ──────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestSuffStatsDataclass:
    """Tests for the SuffStats container itself."""

    def _make(self, k: int = 2) -> SuffStats:
        return SuffStats(
            XtX=np.eye(k),
            Xty=np.ones(k),
            n_obs=100,
            sum_y=10.0,
            sum_y_sq=20.0,
            coef_names=[f"x{i}" for i in range(k)],
            sum_y_sq_exact=True,
        )

    def test_field_access(self):
        s = self._make()
        assert s.n_obs == 100
        assert s.sum_y == pytest.approx(10.0)
        assert s.sum_y_sq_exact is True

    def test_iteration_order(self):
        """Tuple unpacking must yield (XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names)."""
        s = self._make()
        XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names = s
        assert np.array_equal(XtX, np.eye(2))
        assert np.array_equal(Xty, np.ones(2))
        assert n_obs == 100
        assert sum_y == pytest.approx(10.0)
        assert sum_y_sq == pytest.approx(20.0)
        assert coef_names == ["x0", "x1"]

    def test_iteration_does_not_yield_sum_y_sq_exact(self):
        """sum_y_sq_exact is NOT part of the backward-compat 6-tuple."""
        s = self._make()
        unpacked = list(s)
        assert len(unpacked) == 6

    def test_default_sum_y_sq_exact_is_false(self):
        s = SuffStats(
            XtX=np.eye(2),
            Xty=np.zeros(2),
            n_obs=10,
            sum_y=0.0,
            sum_y_sq=0.0,
            coef_names=["a", "b"],
        )
        assert s.sum_y_sq_exact is False

    def test_repr_contains_key_fields(self):
        s = self._make()
        r = repr(s)
        assert "SuffStats" in r
        assert "n_obs" in r


# ---------------------------------------------------------------------------
# ── 2. compute_sufficient_stats_numpy ──────────────────────────────────────
# ---------------------------------------------------------------------------


class TestComputeSufficientStatsNumpy:
    """Correctness and validation tests for the NumPy backend."""

    # ── basic correctness ─────────────────────────────────────────────────

    def test_simple_intercept_only(self):
        """OLS with X = column of 1s should give theta = mean(y)."""
        n = 10
        X = np.ones((n, 1))
        y = np.arange(1, n + 1, dtype=float)
        weights = np.ones(n)

        s = compute_sufficient_stats_numpy(X, y, weights, alpha=0.0)

        expected_XtX = np.array([[float(n)]])
        expected_Xty = np.array([y.sum()])
        assert np.allclose(s.XtX, expected_XtX, atol=ATOL)
        assert np.allclose(s.Xty, expected_Xty, atol=ATOL)
        assert s.n_obs == n
        assert s.sum_y == pytest.approx(y.sum())
        assert s.sum_y_sq == pytest.approx((y ** 2).sum())

    def test_two_regressors_no_intercept(self):
        """XtX computed manually must match."""
        np.random.seed(0)
        n, k = 50, 2
        X = np.random.randn(n, k)
        y = np.random.randn(n)
        w = np.ones(n)

        s = compute_sufficient_stats_numpy(X, y, w, alpha=0.0)

        expected_XtX = X.T @ X
        expected_Xty = X.T @ y
        assert np.allclose(s.XtX, expected_XtX, atol=ATOL)
        assert np.allclose(s.Xty, expected_Xty, atol=ATOL)

    def test_frequency_weights(self):
        """Doubling weights must double n_obs, sum_y, sum_y_sq."""
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0])
        w1 = np.ones(3)
        w2 = 2 * np.ones(3)

        s1 = compute_sufficient_stats_numpy(X, y, w1, alpha=0.0)
        s2 = compute_sufficient_stats_numpy(X, y, w2, alpha=0.0)

        assert s2.n_obs == 2 * s1.n_obs
        assert s2.sum_y == pytest.approx(2 * s1.sum_y)
        assert s2.sum_y_sq == pytest.approx(2 * s1.sum_y_sq)
        assert np.allclose(s2.XtX - np.eye(1) * DEFAULT_ALPHA,
                           2 * (s1.XtX - np.eye(1) * DEFAULT_ALPHA), atol=ATOL)

    def test_regularisation_added_to_diagonal(self):
        X = np.eye(3)
        y = np.zeros(3)
        w = np.ones(3)
        alpha = 0.5

        s = compute_sufficient_stats_numpy(X, y, w, alpha=alpha)
        diag = np.diag(s.XtX)
        assert np.allclose(diag, 1.0 + alpha, atol=ATOL)

    def test_coef_names_default(self):
        X = np.ones((5, 3))
        s = compute_sufficient_stats_numpy(X, np.ones(5), np.ones(5))
        assert s.coef_names == ["x0", "x1", "x2"]

    def test_coef_names_provided(self):
        X = np.ones((5, 2))
        s = compute_sufficient_stats_numpy(X, np.ones(5), np.ones(5),
                                           coef_names=["age", "income"])
        assert s.coef_names == ["age", "income"]

    def test_sum_y_sq_exact_is_true(self):
        """numpy backend always has exact sum_y_sq."""
        X = np.ones((4, 1))
        s = compute_sufficient_stats_numpy(X, np.array([1.0, 2.0, 3.0, 4.0]), np.ones(4))
        assert s.sum_y_sq_exact is True

    def test_returns_suffstats_type(self):
        X = np.ones((3, 1))
        result = compute_sufficient_stats_numpy(X, np.ones(3), np.ones(3))
        assert isinstance(result, SuffStats)

    # ── reshaping ─────────────────────────────────────────────────────────

    def test_1d_X_reshaped(self):
        X = np.ones(5)  # 1-D — should be treated as (5, 1)
        s = compute_sufficient_stats_numpy(X, np.ones(5), np.ones(5), alpha=0.0)
        assert s.XtX.shape == (1, 1)

    def test_1d_y_reshaped(self):
        X = np.ones((5, 1))
        s = compute_sufficient_stats_numpy(X, np.ones(5), np.ones(5))
        assert isinstance(s, SuffStats)

    # ── validation ────────────────────────────────────────────────────────

    def test_none_X_raises(self):
        with pytest.raises(ValueError, match="X cannot be None"):
            compute_sufficient_stats_numpy(None, np.ones(3), np.ones(3))

    def test_none_y_raises(self):
        with pytest.raises(ValueError, match="y cannot be None"):
            compute_sufficient_stats_numpy(np.ones((3, 1)), None, np.ones(3))

    def test_none_weights_raises(self):
        with pytest.raises(ValueError, match="weights cannot be None"):
            compute_sufficient_stats_numpy(np.ones((3, 1)), np.ones(3), None)

    # ── numerical cross-check against manual WLS ──────────────────────────

    def test_matches_manual_wls(self):
        """Solve via SuffStats and compare to numpy.linalg.lstsq."""
        np.random.seed(42)
        n, k = 100, 3
        X = np.random.randn(n, k)
        y = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * np.random.randn(n)
        w = np.random.uniform(0.5, 2.0, n)

        s = compute_sufficient_stats_numpy(X, y, w, alpha=0.0)
        theta_suffstats = np.linalg.solve(s.XtX, s.Xty)

        # Reference: weighted lstsq
        W_sqrt = np.diag(np.sqrt(w))
        theta_ref, *_ = np.linalg.lstsq(W_sqrt @ X, W_sqrt @ y, rcond=None)

        assert np.allclose(theta_suffstats, theta_ref, atol=1e-6)


# ---------------------------------------------------------------------------
# ── 3. compute_sufficient_stats_sql ────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestComputeSufficientStatsSql:
    """Correctness, delegation, and validation tests for the SQL backend."""

    @pytest.fixture
    def conn(self):
        c = _make_conn()
        yield c
        c.close()

    @pytest.fixture
    def simple_table(self, conn):
        """Three-row table: x1, x2, sum_y, sum_y_sq, count=1."""
        _seed_table(conn, "t",
                    x1=[1.0, 2.0, 3.0],
                    x2=[4.0, 5.0, 6.0],
                    y=[10.0, 20.0, 30.0])
        return "t"

    # ── correctness ───────────────────────────────────────────────────────

    def test_matches_numpy_backend(self, conn, simple_table):
        """SQL and NumPy backends must agree on XtX, Xty, n_obs, sum_y, sum_y_sq."""
        sql_s = compute_sufficient_stats_sql(
            conn, simple_table,
            x_cols=["x1", "x2"],
            y_col="sum_y",
            weight_col="count",
            add_intercept=True,
            alpha=0.0,
            sum_y_sq_col="sum_y_sq",
        )

        X = np.array([[1, 1.0, 4.0],
                      [1, 2.0, 5.0],
                      [1, 3.0, 6.0]])
        y = np.array([10.0, 20.0, 30.0])
        w = np.ones(3)
        np_s = compute_sufficient_stats_numpy(X, y, w, alpha=0.0,
                                              coef_names=["Intercept", "x1", "x2"])

        assert np.allclose(sql_s.XtX, np_s.XtX, atol=ATOL)
        assert np.allclose(sql_s.Xty, np_s.Xty, atol=ATOL)
        assert sql_s.n_obs == np_s.n_obs
        assert sql_s.sum_y == pytest.approx(np_s.sum_y)
        assert sql_s.sum_y_sq == pytest.approx(np_s.sum_y_sq)

    def test_add_intercept_false(self, conn, simple_table):
        s = compute_sufficient_stats_sql(
            conn, simple_table,
            x_cols=["x1", "x2"],
            y_col="sum_y",
            weight_col="count",
            add_intercept=False,
            alpha=0.0,
        )
        assert s.XtX.shape == (2, 2)
        assert s.coef_names == ["x1", "x2"]

    def test_add_intercept_true_coef_names(self, conn, simple_table):
        s = compute_sufficient_stats_sql(
            conn, simple_table,
            x_cols=["x1", "x2"],
            y_col="sum_y",
            weight_col="count",
            add_intercept=True,
        )
        assert s.coef_names[0] == "Intercept"
        assert s.coef_names[1:] == ["x1", "x2"]

    def test_regularisation_on_diagonal(self, conn, simple_table):
        alpha = 0.5
        s = compute_sufficient_stats_sql(
            conn, simple_table,
            x_cols=["x1"],
            y_col="sum_y",
            weight_col="count",
            add_intercept=False,
            alpha=alpha,
        )
        # XtX[0,0] = SUM(x1*x1*count) + alpha
        raw = float(conn.execute(
            "SELECT SUM(x1 * x1 * count) FROM t"
        ).fetchone()[0])
        assert s.XtX[0, 0] == pytest.approx(raw + alpha)

    def test_xtx_symmetry(self, conn, simple_table):
        s = compute_sufficient_stats_sql(
            conn, simple_table,
            x_cols=["x1", "x2"],
            y_col="sum_y",
            weight_col="count",
            add_intercept=True,
        )
        assert np.allclose(s.XtX, s.XtX.T, atol=ATOL)

    # ── sum_y_sq_col branch ───────────────────────────────────────────────

    def test_exact_sum_y_sq_when_col_exists(self, conn, simple_table):
        s = compute_sufficient_stats_sql(
            conn, simple_table,
            x_cols=["x1"],
            y_col="sum_y",
            weight_col="count",
            add_intercept=False,
            sum_y_sq_col="sum_y_sq",
        )
        assert s.sum_y_sq_exact is True
        expected = 10.0 ** 2 + 20.0 ** 2 + 30.0 ** 2
        assert s.sum_y_sq == pytest.approx(expected)

    def test_approximate_sum_y_sq_when_col_missing(self, conn, simple_table):
        # sum_y_sq column does not exist in the table when we pass a wrong name
        s = compute_sufficient_stats_sql(
            conn, simple_table,
            x_cols=["x1"],
            y_col="sum_y",
            weight_col="count",
            add_intercept=False,
            sum_y_sq_col="nonexistent_col",
        )
        assert s.sum_y_sq_exact is False

    def test_no_sum_y_sq_col_defaults_to_approximate(self, conn, simple_table):
        s = compute_sufficient_stats_sql(
            conn, simple_table,
            x_cols=["x1"],
            y_col="sum_y",
            weight_col="count",
            add_intercept=False,
        )
        assert s.sum_y_sq_exact is False

    def test_sum_y_sq_approximate_is_exact_for_unit_weights(self, conn, simple_table):
        """For count=1 rows, approx == exact."""
        s_exact = compute_sufficient_stats_sql(
            conn, simple_table,
            x_cols=["x1"],
            y_col="sum_y",
            weight_col="count",
            add_intercept=False,
            sum_y_sq_col="sum_y_sq",
        )
        s_approx = compute_sufficient_stats_sql(
            conn, simple_table,
            x_cols=["x1"],
            y_col="sum_y",
            weight_col="count",
            add_intercept=False,
        )
        assert s_exact.sum_y_sq == pytest.approx(s_approx.sum_y_sq, rel=1e-6)

    # ── where_clause ──────────────────────────────────────────────────────

    def test_where_clause_filters_rows(self, conn, simple_table):
        s_all = compute_sufficient_stats_sql(
            conn, simple_table,
            x_cols=["x1"],
            y_col="sum_y",
            weight_col="count",
            add_intercept=False,
            alpha=0.0,
        )
        s_filtered = compute_sufficient_stats_sql(
            conn, simple_table,
            x_cols=["x1"],
            y_col="sum_y",
            weight_col="count",
            add_intercept=False,
            alpha=0.0,
            where_clause="WHERE x1 <= 2",
        )
        assert s_filtered.n_obs < s_all.n_obs
        assert s_filtered.n_obs == 2

    def test_where_clause_affects_sum_y(self, conn, simple_table):
        s = compute_sufficient_stats_sql(
            conn, simple_table,
            x_cols=["x1"],
            y_col="sum_y",
            weight_col="count",
            add_intercept=False,
            where_clause="WHERE x1 = 1",
        )
        assert s.sum_y == pytest.approx(10.0)

    # ── return type ───────────────────────────────────────────────────────

    def test_returns_suffstats_type(self, conn, simple_table):
        result = compute_sufficient_stats_sql(
            conn, simple_table,
            x_cols=["x1"],
            y_col="sum_y",
            weight_col="count",
        )
        assert isinstance(result, SuffStats)

    def test_tuple_unpacking(self, conn, simple_table):
        XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names = compute_sufficient_stats_sql(
            conn, simple_table,
            x_cols=["x1"],
            y_col="sum_y",
            weight_col="count",
        )
        assert isinstance(XtX, np.ndarray)
        assert isinstance(coef_names, list)

    # ── validation ────────────────────────────────────────────────────────

    def test_none_conn_raises(self, simple_table):
        with pytest.raises(ValueError, match="conn cannot be None"):
            compute_sufficient_stats_sql(
                None, simple_table, ["x1"], "sum_y", "count"
            )

    def test_empty_table_name_raises(self, conn):
        with pytest.raises(ValueError, match="table_name cannot be empty"):
            compute_sufficient_stats_sql(conn, "", ["x1"], "sum_y", "count")

    def test_empty_x_cols_raises(self, conn, simple_table):
        with pytest.raises(ValueError, match="x_cols cannot be empty"):
            compute_sufficient_stats_sql(conn, simple_table, [], "sum_y", "count")

    def test_empty_y_col_raises(self, conn, simple_table):
        with pytest.raises(ValueError, match="y_col cannot be empty"):
            compute_sufficient_stats_sql(conn, simple_table, ["x1"], "", "count")

    def test_empty_weight_col_raises(self, conn, simple_table):
        with pytest.raises(ValueError, match="weight_col cannot be empty"):
            compute_sufficient_stats_sql(conn, simple_table, ["x1"], "sum_y", "")

    # ── weighted data (count > 1) ─────────────────────────────────────────

    def test_weighted_rows(self, conn):
        """Two rows with count=3 each: n_obs should be 6."""
        conn.execute("DROP TABLE IF EXISTS tw")
        conn.execute("""
            CREATE TABLE tw AS
            SELECT * FROM (VALUES
                (1.0, 5.0, 25.0, 3),
                (2.0, 8.0, 64.0, 3)
            ) t(x1, sum_y, sum_y_sq, count)
        """)
        s = compute_sufficient_stats_sql(
            conn, "tw",
            x_cols=["x1"],
            y_col="sum_y",
            weight_col="count",
            add_intercept=False,
            alpha=0.0,
        )
        assert s.n_obs == 6
        # XtX[0,0] = SUM(x1 * x1 * count) = 1*3 + 4*3 = 15
        assert s.XtX[0, 0] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# ── 4. execute_to_matrix ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestExecuteToMatrix:
    """Tests for the generic SQL → numpy matrix helper."""

    @pytest.fixture
    def conn(self):
        c = _make_conn()
        yield c
        c.close()

    def _const_query(self, values: list[float]) -> str:
        """Build a SELECT query returning fixed literal values."""
        cols = ", ".join(str(v) for v in values)
        return f"SELECT {cols}"

    # ── upper_triangle=False (full matrix) ───────────────────────────────

    def test_full_2x2(self, conn):
        query = self._const_query([1.0, 2.0, 3.0, 4.0])
        M = execute_to_matrix(conn, query, (2, 2), upper_triangle=False)
        assert M.shape == (2, 2)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert np.allclose(M, expected)

    def test_full_2x3(self, conn):
        query = self._const_query([1, 2, 3, 4, 5, 6])
        M = execute_to_matrix(conn, query, (2, 3), upper_triangle=False)
        assert M.shape == (2, 3)
        assert M[1, 2] == pytest.approx(6.0)

    # ── upper_triangle=True (symmetric matrix) ────────────────────────────

    def test_upper_triangle_2x2(self, conn):
        # Upper triangle of a 2x2: (0,0), (0,1), (1,1) → 3 values
        query = self._const_query([4.0, 2.0, 9.0])
        M = execute_to_matrix(conn, query, (2, 2), upper_triangle=True)
        assert M.shape == (2, 2)
        assert M[0, 0] == pytest.approx(4.0)
        assert M[0, 1] == pytest.approx(2.0)
        assert M[1, 0] == pytest.approx(2.0)  # symmetrised
        assert M[1, 1] == pytest.approx(9.0)

    def test_upper_triangle_3x3_symmetric(self, conn):
        # 6 values for 3x3 upper triangle
        query = self._const_query([1, 2, 3, 5, 6, 9])
        M = execute_to_matrix(conn, query, (3, 3), upper_triangle=True)
        assert np.allclose(M, M.T)

    def test_upper_triangle_requires_square(self, conn):
        query = self._const_query([1, 2, 3])
        with pytest.raises(ValueError, match="square"):
            execute_to_matrix(conn, query, (2, 3), upper_triangle=True)

    # ── validation ────────────────────────────────────────────────────────

    def test_none_conn_raises(self):
        with pytest.raises(ValueError, match="conn cannot be None"):
            execute_to_matrix(None, "SELECT 1", (1, 1))

    def test_empty_query_raises(self, conn):
        with pytest.raises(ValueError, match="query cannot be empty"):
            execute_to_matrix(conn, "   ", (1, 1))

    def test_bad_shape_raises(self, conn):
        with pytest.raises(ValueError, match="shape must be a 2-tuple"):
            execute_to_matrix(conn, "SELECT 1", (1,))

    def test_non_positive_shape_raises(self, conn):
        with pytest.raises(ValueError, match="positive"):
            execute_to_matrix(conn, "SELECT 1", (0, 2))

    def test_too_few_columns_raises(self, conn):
        query = self._const_query([1.0, 2.0])  # only 2 values
        with pytest.raises(ValueError, match="expected at least"):
            execute_to_matrix(conn, query, (2, 2), upper_triangle=False)

    def test_upper_triangle_too_few_columns_raises(self, conn):
        query = self._const_query([1.0, 2.0])  # need 3 for 2x2 upper triangle
        with pytest.raises(ValueError, match="expected at least"):
            execute_to_matrix(conn, query, (2, 2), upper_triangle=True)

    def test_no_results_raises(self, conn):
        conn.execute("CREATE TABLE empty_t (x INT)")
        with pytest.raises(ValueError, match="no results"):
            execute_to_matrix(conn, "SELECT SUM(x) FROM empty_t WHERE 1=0", (1, 1))


# ---------------------------------------------------------------------------
# ── 5. compute_cross_sufficient_stats_sql ──────────────────────────────────
# ---------------------------------------------------------------------------


class TestComputeCrossSufficientStatsSql:
    """Tests for IV cross-product sufficient statistics."""

    @pytest.fixture
    def conn(self):
        c = _make_conn()
        yield c
        c.close()

    @pytest.fixture
    def iv_table(self, conn):
        """Table with x1, z1, z2, sum_y, count columns."""
        conn.execute("DROP TABLE IF EXISTS iv")
        conn.execute("""
            CREATE TABLE iv AS
            SELECT * FROM (VALUES
                (1.0,  2.0,  3.0,  5.0, 1),
                (2.0,  4.0,  1.0,  8.0, 1),
                (3.0,  6.0,  2.0, 12.0, 1),
                (4.0,  8.0,  4.0, 15.0, 1)
            ) t(x1, z1, z2, sum_y, count)
        """)
        return "iv"

    # ── shapes and types ─────────────────────────────────────────────────

    def test_tXZ_shape_with_intercept(self, conn, iv_table):
        r = compute_cross_sufficient_stats_sql(
            conn, iv_table, x_cols=["x1"], z_cols=["z1", "z2"],
            weight_col="count", add_intercept=True,
        )
        assert r["tXZ"].shape == (2, 3)  # (1+1) x (1+2)

    def test_tZZ_shape_with_intercept(self, conn, iv_table):
        r = compute_cross_sufficient_stats_sql(
            conn, iv_table, x_cols=["x1"], z_cols=["z1", "z2"],
            weight_col="count", add_intercept=True,
        )
        assert r["tZZ"].shape == (3, 3)

    def test_tZZ_is_symmetric(self, conn, iv_table):
        r = compute_cross_sufficient_stats_sql(
            conn, iv_table, x_cols=["x1"], z_cols=["z1", "z2"],
            weight_col="count", add_intercept=True,
        )
        assert np.allclose(r["tZZ"], r["tZZ"].T, atol=ATOL)

    def test_n_obs(self, conn, iv_table):
        r = compute_cross_sufficient_stats_sql(
            conn, iv_table, x_cols=["x1"], z_cols=["z1"],
            weight_col="count", add_intercept=False,
        )
        assert r["n_obs"] == 4

    def test_column_ordering_with_intercept(self, conn, iv_table):
        r = compute_cross_sufficient_stats_sql(
            conn, iv_table, x_cols=["x1"], z_cols=["z1", "z2"],
            weight_col="count", add_intercept=True,
        )
        assert r["x_order"] == ["1", "x1"]
        assert r["z_order"] == ["1", "z1", "z2"]

    def test_column_ordering_without_intercept(self, conn, iv_table):
        r = compute_cross_sufficient_stats_sql(
            conn, iv_table, x_cols=["x1"], z_cols=["z1"],
            weight_col="count", add_intercept=False,
        )
        assert r["x_order"] == ["x1"]
        assert r["z_order"] == ["z1"]

    def test_tXZ_values_manually(self, conn, iv_table):
        """tXZ[0,0] (no intercept) = SUM(x1 * z1 * count)."""
        r = compute_cross_sufficient_stats_sql(
            conn, iv_table, x_cols=["x1"], z_cols=["z1"],
            weight_col="count", add_intercept=False,
        )
        expected = conn.execute(
            "SELECT SUM(x1 * z1 * count) FROM iv"
        ).fetchone()[0]
        assert r["tXZ"][0, 0] == pytest.approx(expected)

    def test_tZZ_diagonal_values(self, conn, iv_table):
        """tZZ[0,0] (no intercept, single z) = SUM(z1 * z1 * count)."""
        r = compute_cross_sufficient_stats_sql(
            conn, iv_table, x_cols=["x1"], z_cols=["z1"],
            weight_col="count", add_intercept=False,
        )
        expected = conn.execute(
            "SELECT SUM(z1 * z1 * count) FROM iv"
        ).fetchone()[0]
        assert r["tZZ"][0, 0] == pytest.approx(expected)

    # ── validation ────────────────────────────────────────────────────────

    def test_none_conn_raises(self):
        with pytest.raises(ValueError, match="conn cannot be None"):
            compute_cross_sufficient_stats_sql(None, "t", ["x1"], ["z1"])

    def test_empty_table_name_raises(self, conn):
        with pytest.raises(ValueError, match="table_name cannot be empty"):
            compute_cross_sufficient_stats_sql(conn, "   ", ["x1"], ["z1"])

    def test_empty_x_cols_raises(self, conn, iv_table):
        with pytest.raises(ValueError, match="x_cols cannot be empty"):
            compute_cross_sufficient_stats_sql(conn, iv_table, [], ["z1"])

    def test_empty_z_cols_raises(self, conn, iv_table):
        with pytest.raises(ValueError, match="z_cols cannot be empty"):
            compute_cross_sufficient_stats_sql(conn, iv_table, ["x1"], [])

    # ── result keys ───────────────────────────────────────────────────────

    def test_result_has_required_keys(self, conn, iv_table):
        r = compute_cross_sufficient_stats_sql(
            conn, iv_table, x_cols=["x1"], z_cols=["z1"],
            weight_col="count",
        )
        assert {"tXZ", "tZZ", "n_obs", "x_order", "z_order"} <= set(r.keys())

    def test_tXZ_is_ndarray(self, conn, iv_table):
        r = compute_cross_sufficient_stats_sql(
            conn, iv_table, x_cols=["x1"], z_cols=["z1"],
            weight_col="count",
        )
        assert isinstance(r["tXZ"], np.ndarray)
        assert isinstance(r["tZZ"], np.ndarray)


# ---------------------------------------------------------------------------
# ── 6. profile_fe_column ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestProfileFeColumn:
    """Tests for the FE column profiling helper."""

    @pytest.fixture
    def conn(self):
        c = _make_conn()
        yield c
        c.close()

    @pytest.fixture
    def fe_table(self, conn):
        """Table: country (A×10, B×5, C×1 → C is a singleton), outcome."""
        rows = (
            [("A", i) for i in range(10)]
            + [("B", i) for i in range(5)]
            + [("C", 0)]
        )
        conn.execute("DROP TABLE IF EXISTS fe_data")
        conn.execute("""
            CREATE TABLE fe_data AS
            SELECT unnest(['A','A','A','A','A','A','A','A','A','A',
                           'B','B','B','B','B',
                           'C']) AS country,
                   unnest([1,2,3,4,5,6,7,8,9,10,
                           1,2,3,4,5,
                           1]) AS outcome
        """)
        return "fe_data"

    def test_cardinality(self, conn, fe_table):
        p = profile_fe_column(conn, fe_table, "country")
        assert p["cardinality"] == 3

    def test_total_obs(self, conn, fe_table):
        p = profile_fe_column(conn, fe_table, "country")
        assert p["total_obs"] == 16

    def test_singleton_share(self, conn, fe_table):
        # Only C has 1 obs → 1 out of 3 groups
        p = profile_fe_column(conn, fe_table, "country")
        assert p["singleton_share"] == pytest.approx(1 / 3, rel=1e-5)

    def test_avg_obs_per_level(self, conn, fe_table):
        # (10 + 5 + 1) / 3 ≈ 5.333
        p = profile_fe_column(conn, fe_table, "country")
        assert p["avg_obs_per_level"] == pytest.approx(16 / 3, rel=1e-5)

    def test_result_keys(self, conn, fe_table):
        p = profile_fe_column(conn, fe_table, "country")
        expected_keys = {
            "cardinality", "singleton_share",
            "avg_obs_per_level", "median_obs_per_level", "total_obs",
        }
        assert expected_keys <= set(p.keys())

    def test_no_singletons(self, conn):
        """All groups have the same size → singleton_share == 0."""
        conn.execute("DROP TABLE IF EXISTS uniform_fe")
        conn.execute("""
            CREATE TABLE uniform_fe AS
            SELECT unnest(['A','A','B','B','C','C']) AS grp,
                   unnest([1,2,1,2,1,2]) AS val
        """)
        p = profile_fe_column(conn, "uniform_fe", "grp")
        assert p["singleton_share"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ── 7. get_fe_unique_levels ────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestGetFeUniqueLevels:
    """Tests for FE level retrieval with max_levels guard."""

    @pytest.fixture
    def conn(self):
        c = _make_conn()
        yield c
        c.close()

    @pytest.fixture
    def level_table(self, conn):
        conn.execute("DROP TABLE IF EXISTS levels_t")
        conn.execute("""
            CREATE TABLE levels_t AS
            SELECT unnest(['B','A','C','A','B']) AS fe_col
        """)
        return "levels_t"

    def test_returns_sorted_unique(self, conn, level_table):
        levels = get_fe_unique_levels(conn, level_table, "fe_col")
        assert levels == ["A", "B", "C"]

    def test_correct_count(self, conn, level_table):
        levels = get_fe_unique_levels(conn, level_table, "fe_col")
        assert len(levels) == 3

    def test_returns_list(self, conn, level_table):
        levels = get_fe_unique_levels(conn, level_table, "fe_col")
        assert isinstance(levels, list)

    def test_max_levels_exceeded_raises(self, conn):
        """If the table has more unique values than max_levels, raise ValueError."""
        conn.execute("DROP TABLE IF EXISTS big_fe")
        # Create 10 unique levels
        conn.execute("""
            CREATE TABLE big_fe AS
            SELECT range AS fe_col FROM range(10)
        """)
        with pytest.raises(ValueError, match="more than"):
            get_fe_unique_levels(conn, "big_fe", "fe_col", max_levels=5)

    def test_max_levels_not_exceeded(self, conn):
        conn.execute("DROP TABLE IF EXISTS small_fe")
        conn.execute("""
            CREATE TABLE small_fe AS
            SELECT range AS fe_col FROM range(3)
        """)
        levels = get_fe_unique_levels(conn, "small_fe", "fe_col", max_levels=5)
        assert len(levels) == 3

    def test_exact_max_levels_does_not_raise(self, conn):
        """Exactly max_levels unique values should NOT raise."""
        conn.execute("DROP TABLE IF EXISTS exact_fe")
        conn.execute("""
            CREATE TABLE exact_fe AS
            SELECT range AS fe_col FROM range(5)
        """)
        levels = get_fe_unique_levels(conn, "exact_fe", "fe_col", max_levels=5)
        assert len(levels) == 5

    def test_numeric_levels_sorted(self, conn):
        conn.execute("DROP TABLE IF EXISTS num_fe")
        conn.execute("""
            CREATE TABLE num_fe AS
            SELECT unnest([3, 1, 4, 1, 5, 9]) AS fe_col
        """)
        levels = get_fe_unique_levels(conn, "num_fe", "fe_col")
        assert levels == sorted(set([3, 1, 4, 5, 9]))


# ---------------------------------------------------------------------------
# ── 8. Cross-backend consistency (numpy vs sql) ────────────────────────────
# ---------------------------------------------------------------------------


class TestCrossBackendConsistency:
    """Ensure numpy and SQL backends produce numerically identical results
    across a range of random design matrices."""

    @pytest.fixture
    def conn(self):
        c = _make_conn()
        yield c
        c.close()

    @pytest.mark.parametrize("n,k,seed", [
        (20, 2, 0),
        (50, 3, 1),
        (100, 5, 2),
    ])
    def test_xtx_xty_agree(self, conn, n, k, seed):
        rng = np.random.default_rng(seed)
        X_raw = rng.standard_normal((n, k))
        y = rng.standard_normal(n)

        # Build table with add_intercept=False (X_raw already explicit)
        col_exprs_val = {f"x{i}": X_raw[:, i].tolist() for i in range(k)}
        col_exprs_val["sum_y"] = y.tolist()
        col_exprs_val["sum_y_sq"] = (y ** 2).tolist()
        col_exprs_val["count"] = [1] * n

        conn.execute("DROP TABLE IF EXISTS rand_t")
        select_parts = ", ".join(
            f"unnest({v!r}) AS {c}" for c, v in col_exprs_val.items()
        )
        conn.execute(f"CREATE TABLE rand_t AS SELECT {select_parts}")

        x_cols = [f"x{i}" for i in range(k)]
        sql_s = compute_sufficient_stats_sql(
            conn, "rand_t", x_cols=x_cols, y_col="sum_y",
            weight_col="count", add_intercept=True, alpha=0.0,
            sum_y_sq_col="sum_y_sq",
        )

        X_with_intercept = np.hstack([np.ones((n, 1)), X_raw])
        np_s = compute_sufficient_stats_numpy(
            X_with_intercept, y, np.ones(n),
            coef_names=["Intercept"] + x_cols, alpha=0.0,
        )

        assert np.allclose(sql_s.XtX, np_s.XtX, atol=1e-8, rtol=1e-6)
        assert np.allclose(sql_s.Xty, np_s.Xty, atol=1e-8, rtol=1e-6)
        assert sql_s.n_obs == np_s.n_obs
        assert sql_s.sum_y == pytest.approx(np_s.sum_y, rel=1e-8)
        assert sql_s.sum_y_sq == pytest.approx(np_s.sum_y_sq, rel=1e-8)
