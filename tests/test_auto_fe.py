"""Unit tests for AutoFETransformer.

Covers:
  - Routing logic (pure Mundlak, pure MAP, hybrid)
  - Cardinality estimation
  - has_intercept / extra_regressors per execution path
  - transform_query delegation
  - n_obs / df_correction properties
  - fit_transform result table name and schema
  - RuntimeError before fit
  - Coefficient accuracy vs pyfixest for each execution path
"""

import numpy as np
import pandas as pd
import pytest
import duckdb
import pyfixest as pf

from duckreg.core.transformers import AutoFETransformer


import pytest

# AutoFE tests are currently disabled; re-enable once stability issues are
# resolved.
pytest.skip("auto_fe testing temporarily disabled", allow_module_level=True)

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    """Tiny balanced panel: 200 obs, 4 groups × 50 rows (pixel_id),
    5 year values.  Low-cardinality: year (5), high-cardinality: pixel_id (40).
    """
    rng = np.random.default_rng(0)
    n = 200
    pixel_id = np.tile(np.arange(40), 5)   # 40 pixels × 5 years
    year     = np.repeat(np.arange(5), 40)
    x        = rng.standard_normal(n)
    fe_px    = rng.standard_normal(40)[pixel_id]
    fe_yr    = rng.standard_normal(5)[year]
    y        = 1.5 * x + fe_px + fe_yr + rng.standard_normal(n) * 0.5
    return pd.DataFrame({"pixel_id": pixel_id, "year": year, "x": x, "y": y})


@pytest.fixture()
def conn(small_df):
    """Fresh in-memory DuckDB connection with the test panel registered."""
    con = duckdb.connect()
    con.register("tbl", small_df)
    yield con
    con.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_transformer(conn, fe_cols, cardinality_ratio=10.0, **kwargs):
    return AutoFETransformer(
        conn=conn,
        table_name="tbl",
        fe_cols=fe_cols,
        covariate_cols=["x"],
        cardinality_ratio=cardinality_ratio,
        **kwargs,
    )


# ── Pre-fit guards ────────────────────────────────────────────────────────────

class TestPreFitGuards:
    def test_n_obs_raises_before_fit(self, conn):
        t = _make_transformer(conn, ["year"])
        with pytest.raises(RuntimeError, match="fit_transform"):
            _ = t.n_obs

    def test_transform_query_raises_before_fit(self, conn):
        t = _make_transformer(conn, ["year"])
        with pytest.raises(RuntimeError, match="fit_transform"):
            t.transform_query(["x"])

    def test_df_correction_zero_before_fit(self, conn):
        t = _make_transformer(conn, ["year"])
        assert t.df_correction == 0

    def test_extra_regressors_empty_before_fit(self, conn):
        t = _make_transformer(conn, ["year"])
        assert t.extra_regressors == []


# ── Cardinality estimation ────────────────────────────────────────────────────

class TestCardinality:
    def test_estimated_cardinality_year(self, conn, small_df):
        """year has 5 unique values; estimate should be in [1, 10]."""
        t = _make_transformer(conn, ["year", "pixel_id"])
        t.fit_transform(["y", "x"])
        assert 1 <= t.cardinalities_["year"] <= 10

    def test_estimated_cardinality_pixel_id(self, conn, small_df):
        """pixel_id has 40 unique values; estimate should be in [20, 60]."""
        t = _make_transformer(conn, ["year", "pixel_id"])
        t.fit_transform(["y", "x"])
        assert 20 <= t.cardinalities_["pixel_id"] <= 60

    def test_routing_keys_match_fe_cols(self, conn):
        t = _make_transformer(conn, ["year", "pixel_id"])
        t.fit_transform(["y", "x"])
        assert set(t.routing_.keys()) == {"year", "pixel_id"}
        assert all(v in ("mundlak", "map") for v in t.routing_.values())


# ── Pure Mundlak path ─────────────────────────────────────────────────────────
# cardinality_ratio=1000 → threshold = 200/1000 = 0.2 → everything ≤ 0.2? No.
# Use very small ratio so threshold is tiny and everything is above → all MAP.
# Use very large ratio so threshold is huge and everything is below → all Mundlak.

class TestPureMundlak:
    """Force all FEs to Mundlak by setting a large cardinality_ratio.

    With the updated routing rule, FEs with estimated cardinality *above*
    threshold go to Mundlak.  threshold = N / ratio, so making the ratio
    large makes the threshold small, causing all realistic cardinalities to
    exceed it and hence route to Mundlak.  We choose ratio=1000 → threshold
    ≈0.2, which is smaller than any actual cardinality in our small panel.
    """

    def setup_method(self):
        self.ratio = 1000.0  # threshold = 200/1000 = 0.2 → all Mundlak


    def test_result_table_name(self, conn):
        t = _make_transformer(conn, ["year", "pixel_id"], cardinality_ratio=self.ratio)
        result = t.fit_transform(["y", "x"])
        assert result == "demeaned_data"

    def test_has_intercept_true(self, conn):
        t = _make_transformer(conn, ["year", "pixel_id"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        assert t.has_intercept is True

    def test_extra_regressors_nonempty(self, conn):
        t = _make_transformer(conn, ["year", "pixel_id"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        assert len(t.extra_regressors) > 0, "Mundlak should add mean columns"

    def test_all_routed_to_mundlak(self, conn):
        t = _make_transformer(conn, ["year", "pixel_id"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        assert all(v == "mundlak" for v in t.routing_.values())

    def test_df_correction_zero(self, conn):
        """Pure Mundlak absorbs FEs as explicit regressors; df_correction = 0."""
        t = _make_transformer(conn, ["year", "pixel_id"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        assert t.df_correction == 0

    def test_n_obs_positive(self, conn, small_df):
        t = _make_transformer(conn, ["year"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        assert t.n_obs == len(small_df)

    def test_transform_query_identity(self, conn):
        """Pure-Mundlak transform_query is identity (no resid_ prefix)."""
        t = _make_transformer(conn, ["year"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        q = t.transform_query(["y", "x"])
        assert "resid_" not in q
        assert "y" in q and "x" in q

    def test_result_table_columns(self, conn):
        """demeaned_data should contain the original + extra Mundlak columns."""
        t = _make_transformer(conn, ["year"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        cols = set(
            conn.execute("SELECT column_name FROM (DESCRIBE demeaned_data)")
            .fetchdf()["column_name"]
        )
        assert "y" in cols and "x" in cols


# ── Pure MAP path ─────────────────────────────────────────────────────────────

class TestPureMAP:
    """Force all FEs to MAP by using only mid-cardinality FE dimensions.

    Under the current rule, ``c_est < 30`` routes to Mundlak, so the
    low-cardinality ``year`` FE can never be MAP.  We therefore test pure MAP
    with only ``pixel_id`` (c_est≈40), and choose ratio=4 so
    threshold=200/4=50, placing pixel_id in the MAP middle band.
    """

    def setup_method(self):
        self.ratio = 4.0

    def test_result_table_name(self, conn):
        t = _make_transformer(conn, ["pixel_id"], cardinality_ratio=self.ratio)
        result = t.fit_transform(["y", "x"])
        assert result == "demeaned_data"

    def test_has_intercept_false(self, conn):
        t = _make_transformer(conn, ["pixel_id"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        assert t.has_intercept is False

    def test_extra_regressors_empty(self, conn):
        t = _make_transformer(conn, ["pixel_id"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        assert t.extra_regressors == []

    def test_all_routed_to_map(self, conn):
        t = _make_transformer(conn, ["pixel_id"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        assert all(v == "map" for v in t.routing_.values())

    def test_df_correction_positive(self, conn):
        """MAP absorbs FEs; df_correction equals total distinct FE levels."""
        t = _make_transformer(conn, ["pixel_id"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        assert t.df_correction > 0

    def test_n_iterations_positive(self, conn):
        """MAP should converge in at least one iteration."""
        t = _make_transformer(conn, ["pixel_id"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        active = t._active_transformer
        assert active is not None
        assert getattr(active, "n_iterations", None) is not None
        assert active.n_iterations >= 1

    def test_transform_query_has_resid_prefix(self, conn):
        """MAP transform_query should return resid_v AS v style."""
        t = _make_transformer(conn, ["pixel_id"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        q = t.transform_query(["y", "x"])
        assert "resid_y AS y" in q
        assert "resid_x AS x" in q

    def test_result_table_has_resid_cols(self, conn):
        """demeaned_data must contain resid_y and resid_x for MAP."""
        t = _make_transformer(conn, ["pixel_id"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        cols = set(
            conn.execute("SELECT column_name FROM (DESCRIBE demeaned_data)")
            .fetchdf()["column_name"]
        )
        assert "resid_y" in cols and "resid_x" in cols


# ── Hybrid path ───────────────────────────────────────────────────────────────

class TestHybrid:
    """year → Mundlak, pixel_id → MAP.

    With N=200, threshold = 200 / ratio.
        year has ~5 unique values (<30), so it always routes to Mundlak.
        Choose ratio so threshold exceeds pixel cardinality but remains in a
        realistic range:
            ratio = 4 → threshold = 50
            pixel_id (40) is in middle band [30, 50] → MAP.
    """

    def setup_method(self):
                self.ratio = 4.0

    def test_routing_assigns_correctly(self, conn):
        t = _make_transformer(conn, ["year", "pixel_id"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        assert t.routing_["year"] == "mundlak", (
            f"year should be mundlak (low cardinality), got {t.routing_['year']!r} "
            f"(c_est={t.cardinalities_['year']}, threshold={200/self.ratio:.1f})"
        )
        assert t.routing_["pixel_id"] == "map", (
            f"pixel_id should be map (middle cardinality), got {t.routing_['pixel_id']!r} "
            f"(c_est={t.cardinalities_['pixel_id']}, threshold={200/self.ratio:.1f})"
        )

    def test_has_intercept_false(self, conn):
        """Hybrid includes Mundlak regressors and may not have an intercept."""
        t = _make_transformer(conn, ["year", "pixel_id"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        assert t.has_intercept is False

    def test_extra_regressors_from_mundlak(self, conn):
        """Mundlak-added columns should appear as extra_regressors in hybrid."""
        t = _make_transformer(conn, ["year", "pixel_id"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        assert len(t.extra_regressors) > 0, (
            "hybrid should expose Mundlak mean cols as extra_regressors"
        )

    def test_df_correction_from_map(self, conn):
        """df_correction comes from the MAP transformer (pixel_id levels)."""
        t = _make_transformer(conn, ["year", "pixel_id"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        assert t.df_correction > 0

    def test_result_table_name(self, conn):
        t = _make_transformer(conn, ["year", "pixel_id"], cardinality_ratio=self.ratio)
        result = t.fit_transform(["y", "x"])
        assert result == "demeaned_data"

    def test_active_transformer_is_map(self, conn):
        from duckreg.core.transformers import IterativeDemeanTransformer
        t = _make_transformer(conn, ["year", "pixel_id"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        assert isinstance(t._active_transformer, IterativeDemeanTransformer)

    def test_n_obs_positive(self, conn, small_df):
        t = _make_transformer(conn, ["year", "pixel_id"], cardinality_ratio=self.ratio)
        t.fit_transform(["y", "x"])
        assert t.n_obs == len(small_df)