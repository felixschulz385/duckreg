"""
Unit and integration tests for Duck2SLS.

Covers components *specific* to the 2SLS estimator.  The following aspects
are deliberately NOT re-tested here because they are covered by other suites:

  - test_compressed_ols.py    coefficient accuracy vs pyfixest for IV models
  - test_data_prep.py         singleton removal (including for Duck2SLS)
  - test_vcov_components.py   building blocks of the sandwich estimator
  - test_residual_aggregates.py residual aggregate logic
  - test_suffstats.py         sufficient-statistics computation
  - test_unbalanced_panel.py  FE classification / Wooldridge correction
"""

import logging
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

from duckreg.estimators.Duck2SLS import Duck2SLS
from duckreg.utils.formula_parser import FormulaParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_parser = FormulaParser()


def _parse(f: str):
    return _parser.parse(f)


def _make_parquet(df: pd.DataFrame) -> str:
    """Write *df* to a temp parquet file and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as fh:
        path = fh.name
    df.to_parquet(path, index=False)
    return path


def _build(parquet_path: str, formula: str, **kwargs) -> Duck2SLS:
    """Construct a Duck2SLS from a parquet path (not yet fitted)."""
    f = _parse(formula)
    return Duck2SLS(
        db_name=":memory:",
        table_name=f"read_parquet('{parquet_path}')",
        formula=f,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def iv_panel_data():
    """Panel with a strong instrument and known true 2SLS coefficient.

    DGP:
        z_i          ~ N(0, 1)   first-stage instrument
        endog = 0.8*z + u_endog  (strong first stage, F >> 10)
        y = 2.0*endog + 1.0*x + firm_fe + eps
    True *causal* coefficient on endog: 2.0
    """
    rng = np.random.default_rng(0)
    n_firms, n_years = 80, 8
    index = pd.MultiIndex.from_product(
        [np.arange(n_firms), np.arange(2010, 2010 + n_years)],
        names=["firm_id", "year"],
    ).to_frame(index=False)

    firm_fe = rng.standard_normal(n_firms)
    n = len(index)

    index["x"]     = rng.standard_normal(n)
    index["z"]     = rng.standard_normal(n)
    index["endog"] = 0.8 * index["z"] + rng.standard_normal(n) * 0.5
    index["y"]     = (
        2.0 * index["endog"]
        + 1.0 * index["x"]
        + firm_fe[index["firm_id"]]
        + rng.standard_normal(n) * 0.5
    )
    return index


@pytest.fixture(scope="module")
def iv_parquet(iv_panel_data):
    path = _make_parquet(iv_panel_data)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture(scope="module")
def fitted_numpy(iv_parquet):
    """Duck2SLS, numpy fitter, no FE, fitted."""
    m = _build(iv_parquet, "y ~ x | | (endog ~ z)", fitter="numpy", remove_singletons=False)
    m.fit(se_method="HC1")
    return m


@pytest.fixture(scope="module")
def fitted_numpy_fe(iv_parquet):
    """Duck2SLS, numpy fitter, one FE (firm_id), fitted."""
    m = _build(iv_parquet, "y ~ x | firm_id | (endog ~ z)", fitter="numpy", remove_singletons=False)
    m.fit(se_method="HC1")
    return m


@pytest.fixture(scope="module")
def fitted_duckdb(iv_parquet):
    """Duck2SLS, duckdb fitter, no FE, fitted."""
    m = _build(iv_parquet, "y ~ x | | (endog ~ z)", fitter="duckdb", remove_singletons=False)
    m.fit(se_method="HC1")
    return m


# ===========================================================================
# Construction and validation
# ===========================================================================

class TestConstruction:
    def test_defaults(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | endog (z)")
        assert m.method == "mundlak"
        assert m.fitter == "numpy"
        assert m.endogenous_vars == ["endog"]
        assert m.instrument_vars == ["z"]

    def test_fe_method_compat_alias(self, iv_parquet):
        """fe_method= should map to method= for backward compatibility."""
        m = _build(iv_parquet, "y ~ x | | endog (z)", fe_method="mundlak")
        assert m.method == "mundlak"

    def test_method_demean_accepted(self, iv_parquet):
        """method='demean' is now a supported FE absorption strategy."""
        m = _build(iv_parquet, "y ~ x | | endog (z)", method="demean")
        assert m.method == "demean"

    def test_invalid_method_raises(self, iv_parquet):
        """Unsupported method names must raise ValueError."""
        with pytest.raises(ValueError, match="mundlak"):
            _build(iv_parquet, "y ~ x | | endog (z)", method="invalid")

    def test_internal_state_initialized_to_none(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | endog (z)")
        assert m._ss_transformer is None
        assert m._exog_sql is None
        assert m._fitted_endog_sql is None
        assert m._X_fitted is None
        assert m._Z is None

    def test_formula_extractions(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | firm_id | endog (z)")
        assert "endog" in m.endogenous_vars
        assert "z" in m.instrument_vars
        assert "x" in m.exogenous_vars
        assert "firm_id" in m.fe_cols

    def test_first_stage_results_empty_before_fit(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | endog (z)")
        assert m.first_stage == {}

    def test_results_none_before_fit(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | endog (z)")
        assert m.results is None


# ===========================================================================
# Column-name helpers (pure logic, uses post-prepare_data state)
# ===========================================================================

class TestColumnNameHelpers:
    def test_get_exog_sql_excludes_endogenous(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | endog (z)", remove_singletons=False)
        m.prepare_data()
        exog = m._get_exog_sql()
        # "x" should be present; "endog" must NOT be in exog
        assert any("x" in s for s in exog)
        assert not any("endog" in s for s in exog)

    def test_get_exog_sql_excludes_intercept(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | endog (z)", remove_singletons=False)
        m.prepare_data()
        exog = m._get_exog_sql()
        assert "Intercept" not in exog
        assert "intercept" not in exog

    def test_get_inst_sql(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | endog (z)", remove_singletons=False)
        m.prepare_data()
        inst = m._get_inst_sql()
        assert any("z" in s for s in inst)

    def test_x_extra_excludes_instrument_mundlak_means(self):
        """_x_extra must drop avg_{inst}_fe* columns from extra_regressors."""
        # Simulate the state after _setup_second_stage
        class _Stub:
            _inst_sql         = ["rainfall"]
            _fitted_endog_sql = ["fitted_endog"]
        stub = _Stub()
        extra = [
            "avg_x_fe0",            # exog mean → keep in X
            "avg_rainfall_fe0",     # instrument mean → exclude from X
            "avg_fitted_endog_fe0", # fitted-endog mean → keep in X
            "dummy_firm_id_5",      # fixed-FE dummy → keep in both
        ]
        result = Duck2SLS._x_extra(stub, extra)
        assert "avg_x_fe0" in result
        assert "avg_rainfall_fe0" not in result
        assert "avg_fitted_endog_fe0" in result
        assert "dummy_firm_id_5" in result

    def test_z_extra_excludes_fitted_endog_mundlak_means(self):
        """_z_extra must drop avg_{fitted_endog}_fe* columns."""
        class _Stub:
            _inst_sql         = ["rainfall"]
            _fitted_endog_sql = ["fitted_endog"]
        stub = _Stub()
        extra = [
            "avg_x_fe0",
            "avg_rainfall_fe0",
            "avg_fitted_endog_fe0",
            "dummy_firm_id_5",
        ]
        result = Duck2SLS._z_extra(stub, extra)
        assert "avg_x_fe0" in result
        assert "avg_rainfall_fe0" in result
        assert "avg_fitted_endog_fe0" not in result
        assert "dummy_firm_id_5" in result

    def test_x_extra_z_extra_overlap_is_common_cols(self):
        """Columns that belong to neither set (FE dummies, exog means) appear in both."""
        class _Stub:
            _inst_sql         = ["z"]
            _fitted_endog_sql = ["fitted_endog"]
        stub = _Stub()
        extra = ["avg_x_fe0", "avg_z_fe0", "avg_fitted_endog_fe0"]
        x = set(Duck2SLS._x_extra(stub, extra))
        z = set(Duck2SLS._z_extra(stub, extra))
        # exog mean should appear in both
        assert "avg_x_fe0" in x and "avg_x_fe0" in z


# ===========================================================================
# Display-name builders
# ===========================================================================

class TestDisplayNameBuilders:
    def test_second_stage_coef_names_start_with_intercept(self, fitted_numpy):
        assert fitted_numpy.coef_names_[0] == "Intercept"

    def test_second_stage_coef_names_contains_endog_display(self, fitted_numpy):
        # The display name "endog" should appear (not the "fitted_endog" sql name)
        assert "endog" in fitted_numpy.coef_names_

    def test_second_stage_coef_names_contains_exog(self, fitted_numpy):
        assert "x" in fitted_numpy.coef_names_

    def test_first_stage_coef_names_contains_instrument(self, fitted_numpy):
        fs_res = fitted_numpy.first_stage["endog"]
        assert "z" in fs_res.coef_names

    def test_first_stage_coef_names_start_with_intercept(self, fitted_numpy):
        fs_res = fitted_numpy.first_stage["endog"]
        assert fs_res.coef_names[0] == "Intercept"

    def test_coef_names_length_matches_coefficients(self, fitted_numpy):
        n_coefs   = len(fitted_numpy.point_estimate.flatten())
        n_names   = len(fitted_numpy.coef_names_)
        assert n_coefs == n_names

    def test_fe_coef_names_no_fitted_prefix(self, fitted_numpy_fe):
        """Second-stage coefficient names should show display names, not 'fitted_*'."""
        for name in fitted_numpy_fe.coef_names_:
            assert not name.startswith("fitted_"), (
                f"Unexpected 'fitted_' prefix in coef name: {name!r}"
            )


# ===========================================================================
# prepare_data – staging table structure
# ===========================================================================

class TestPrepareData:
    def test_staging_table_exists(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | endog (z)", remove_singletons=False)
        m.prepare_data()
        count = m.conn.execute(
            f"SELECT COUNT(*) FROM {Duck2SLS._STAGING_TABLE}"
        ).fetchone()[0]
        assert count > 0

    def test_staging_table_has_row_idx(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | endog (z)", remove_singletons=False)
        m.prepare_data()
        cols = set(
            m.conn.execute(
                f"SELECT column_name FROM (DESCRIBE {Duck2SLS._STAGING_TABLE})"
            ).fetchdf()["column_name"].tolist()
        )
        assert "_row_idx" in cols

    def test_staging_table_row_idx_unique(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | endog (z)", remove_singletons=False)
        m.prepare_data()
        total, unique = m.conn.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT _row_idx) FROM {Duck2SLS._STAGING_TABLE}"
        ).fetchone()
        assert total == unique, "_row_idx values must be unique"

    def test_staging_table_has_outcome_and_endog(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | endog (z)", remove_singletons=False)
        m.prepare_data()
        cols = set(
            m.conn.execute(
                f"SELECT column_name FROM (DESCRIBE {Duck2SLS._STAGING_TABLE})"
            ).fetchdf()["column_name"].tolist()
        )
        assert "y" in cols or any("y" in c for c in cols)
        assert "endog" in cols or any("endog" in c for c in cols)
        assert "z" in cols or any("z" in c for c in cols)

    def test_n_obs_set_after_prepare_data(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | endog (z)", remove_singletons=False)
        m.prepare_data()
        assert m.n_obs == 80 * 8


# ===========================================================================
# _add_fitted_column – join-back mechanism
# ===========================================================================

class TestAddFittedColumn:
    def test_fitted_column_appears_in_staging_after_first_stage(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | endog (z)", remove_singletons=False)
        m.prepare_data()
        m._run_first_stages()
        cols = set(
            m.conn.execute(
                f"SELECT column_name FROM (DESCRIBE {Duck2SLS._STAGING_TABLE})"
            ).fetchdf()["column_name"].tolist()
        )
        # Expect a "fitted_<endog_sql_name>" column
        fitted_cols = [c for c in cols if c.startswith("fitted_")]
        assert len(fitted_cols) >= 1

    def test_fitted_column_has_no_nulls(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | endog (z)", remove_singletons=False)
        m.prepare_data()
        m._run_first_stages()
        # Get the fitted column name dynamically
        cols = m.conn.execute(
            f"SELECT column_name FROM (DESCRIBE {Duck2SLS._STAGING_TABLE})"
        ).fetchdf()["column_name"].tolist()
        fitted_col = next(c for c in cols if c.startswith("fitted_"))
        null_count = m.conn.execute(
            f'SELECT COUNT(*) FROM {Duck2SLS._STAGING_TABLE} WHERE "{fitted_col}" IS NULL'
        ).fetchone()[0]
        assert null_count == 0

    def test_row_idx_preserved_after_first_stage(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | endog (z)", remove_singletons=False)
        m.prepare_data()
        n_before = m.conn.execute(
            f"SELECT COUNT(*) FROM {Duck2SLS._STAGING_TABLE}"
        ).fetchone()[0]
        m._run_first_stages()
        n_after = m.conn.execute(
            f"SELECT COUNT(*) FROM {Duck2SLS._STAGING_TABLE}"
        ).fetchone()[0]
        assert n_before == n_after, "Row count must not change during first-stage join-back"


# ===========================================================================
# First-stage results
# ===========================================================================

class TestFirstStage:
    def test_first_stage_dict_keys_match_endogenous(self, fitted_numpy):
        assert set(fitted_numpy.first_stage.keys()) == {"endog"}

    def test_first_stage_has_f_statistic(self, fitted_numpy):
        fs = fitted_numpy.first_stage["endog"]
        assert fs.f_statistic is not None
        assert np.isfinite(fs.f_statistic)

    def test_first_stage_strong_instrument(self, fitted_numpy):
        """Our instrument has a first-stage F >> 10."""
        fs = fitted_numpy.first_stage["endog"]
        assert fs.f_statistic > 10, (
            f"Expected strong instrument (F > 10), got F={fs.f_statistic:.2f}"
        )

    def test_get_first_stage_f_stats(self, fitted_numpy):
        fstats = fitted_numpy.get_first_stage_f_stats()
        assert isinstance(fstats, dict)
        assert "endog" in fstats
        assert fstats["endog"] > 10

    def test_has_weak_instruments_false_for_strong(self, fitted_numpy):
        assert fitted_numpy.has_weak_instruments() is False

    def test_has_weak_instruments_true_for_weak(self, iv_parquet):
        """A near-zero instrument should trigger is_weak_instrument."""
        # Build data with a weak instrument
        df = pd.read_parquet(iv_parquet).copy()
        rng = np.random.default_rng(99)
        df["z_weak"] = rng.standard_normal(len(df)) * 0.001   # tiny
        path = _make_parquet(df)
        try:
            m = _build(path, "y ~ x | | endog (z_weak)", fitter="numpy",
                       remove_singletons=False)
            m.fit(se_method="HC1")
            assert m.has_weak_instruments() is True
        finally:
            os.unlink(path)

    def test_first_stage_coef_count_no_fe(self, fitted_numpy):
        fs = fitted_numpy.first_stage["endog"]
        coef_flat = fs.results.coefficients.flatten()
        n_coefs = len(coef_flat)
        # With no FE: intercept + x + z = 3
        assert n_coefs == 3

    def test_first_stage_coef_count_with_fe(self, fitted_numpy_fe):
        fs = fitted_numpy_fe.first_stage["endog"]
        coef_flat = fs.results.coefficients.flatten()
        # intercept + x + z + mundlak_means_per_FE  (>= 3)
        assert len(coef_flat) >= 3

    def test_first_stage_n_obs_matches_panel(self, fitted_numpy):
        fs = fitted_numpy.first_stage["endog"]
        assert fs.results.n_obs == 80 * 8


# ===========================================================================
# Properties
# ===========================================================================

class TestProperties:
    def test_results_none_without_vcov(self, iv_parquet):
        """results property requires at least point_estimate."""
        m = _build(iv_parquet, "y ~ x | | endog (z)", remove_singletons=False)
        assert m.results is None

    def test_results_not_none_after_fit(self, fitted_numpy):
        assert fitted_numpy.results is not None

    def test_results_has_correct_coef_names(self, fitted_numpy):
        res = fitted_numpy.results
        assert res.coef_names == fitted_numpy.coef_names_

    def test_results_has_vcov(self, fitted_numpy):
        res = fitted_numpy.results
        assert res.vcov is not None
        k = len(fitted_numpy.point_estimate.flatten())
        assert res.vcov.shape == (k, k)

    def test_results_vcov_is_symmetric(self, fitted_numpy):
        vcov = fitted_numpy.results.vcov
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)

    def test_results_vcov_is_positive_semidefinite(self, fitted_numpy):
        eigvals = np.linalg.eigvalsh(fitted_numpy.results.vcov)
        assert np.all(eigvals >= -1e-10), "VCov must be PSD"

    def test_results_n_obs(self, fitted_numpy):
        assert fitted_numpy.results.n_obs == 80 * 8


# ===========================================================================
# Numpy vs DuckDB consistency
# ===========================================================================

class TestNumpyDuckdbConsistency:
    def test_coefficients_close(self, fitted_numpy, fitted_duckdb):
        np.testing.assert_allclose(
            fitted_numpy.point_estimate.flatten(),
            fitted_duckdb.point_estimate.flatten(),
            rtol=1e-4, atol=1e-6,
            err_msg="Numpy and DuckDB fitters should give same second-stage coefs",
        )

    def test_se_close(self, fitted_numpy, fitted_duckdb):
        se_np = np.sqrt(np.diag(fitted_numpy.results.vcov))
        se_db = np.sqrt(np.diag(fitted_duckdb.results.vcov))
        np.testing.assert_allclose(
            se_np, se_db, rtol=1e-3, atol=1e-6,
            err_msg="Numpy and DuckDB SEs should be very close",
        )

    def test_n_obs_matches(self, fitted_numpy, fitted_duckdb):
        assert fitted_numpy.n_obs == fitted_duckdb.n_obs


# ===========================================================================
# Coefficient accuracy (sanity check – not vs pyfixest)
# ===========================================================================

class TestCoefficientSanity:
    def test_endog_coef_near_true_value(self, fitted_numpy):
        """Second-stage endog coefficient should be near 2.0 (true DGP value)."""
        idx   = fitted_numpy.coef_names_.index("endog")
        coef  = fitted_numpy.point_estimate.flatten()[idx]
        assert abs(coef - 2.0) < 0.5, (
            f"Expected endog coef ≈ 2.0, got {coef:.4f}"
        )

    def test_fe_model_endog_coef_near_true_value(self, fitted_numpy_fe):
        idx  = fitted_numpy_fe.coef_names_.index("endog")
        coef = fitted_numpy_fe.point_estimate.flatten()[idx]
        assert abs(coef - 2.0) < 0.5

    def test_duckdb_endog_coef_near_true_value(self, fitted_duckdb):
        idx  = fitted_duckdb.coef_names_.index("endog")
        coef = fitted_duckdb.point_estimate.flatten()[idx]
        assert abs(coef - 2.0) < 0.5


# ===========================================================================
# VCov
# ===========================================================================

class TestVCov:
    def test_vcov_shape(self, fitted_numpy):
        k = len(fitted_numpy.point_estimate.flatten())
        assert fitted_numpy.vcov.shape == (k, k)

    def test_vcov_symmetric(self, fitted_numpy_fe):
        vcov = fitted_numpy_fe.vcov
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)

    def test_fit_vcov_raises_before_compress_data(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | endog (z)", remove_singletons=False)
        m.prepare_data()
        # point_estimate not set → _X_actual is None
        with pytest.raises(RuntimeError):
            m._fit_vcov_numpy()

    def test_bootstrap_warns_and_returns_vcov(self, fitted_numpy, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="duckreg.estimators.Duck2SLS"):
            vcov = fitted_numpy.bootstrap()
        assert any("not yet implemented" in r.message.lower() for r in caplog.records)
        assert vcov is not None
        assert vcov.shape[0] == vcov.shape[1]

    def test_se_positive(self, fitted_numpy):
        se = np.sqrt(np.diag(fitted_numpy.vcov))
        assert np.all(se > 0)

    def test_se_positive_fe(self, fitted_numpy_fe):
        se = np.sqrt(np.diag(fitted_numpy_fe.vcov))
        assert np.all(se > 0)


# ===========================================================================
# Output methods
# ===========================================================================

class TestOutputMethods:
    def test_summary_df_returns_dataframe(self, fitted_numpy):
        df = fitted_numpy.summary_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_summary_df_has_endog_in_index(self, fitted_numpy):
        df = fitted_numpy.summary_df()
        assert "endog" in df.index

    def test_summary_df_has_coefficient_column(self, fitted_numpy):
        df = fitted_numpy.summary_df()
        assert "coefficient" in df.columns

    def test_to_tidy_df_returns_dataframe(self, fitted_numpy):
        df = fitted_numpy.to_tidy_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_to_tidy_df_has_variable_column(self, fitted_numpy):
        df = fitted_numpy.to_tidy_df()
        assert "variable" in df.columns

    def test_summary_df_empty_before_fit(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | endog (z)")
        df = m.summary_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_to_tidy_df_empty_before_fit(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | endog (z)")
        df = m.to_tidy_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_summary_df_matches_coef_names(self, fitted_numpy):
        df   = fitted_numpy.summary_df()
        names = set(fitted_numpy.coef_names_) - {"Intercept"}
        for n in names:
            assert n in df.index, f"Expected {n!r} in summary_df index"


# ===========================================================================
# Multiple endogenous variables
# ===========================================================================

class TestMultipleEndogenousVariables:
    @pytest.fixture(scope="class")
    def multi_endog_data(self):
        rng = np.random.default_rng(7)
        n = 500
        df = pd.DataFrame({
            "y":      np.zeros(n),
            "x":      rng.standard_normal(n),
            "endog1": np.zeros(n),
            "endog2": np.zeros(n),
            "z1":     rng.standard_normal(n),
            "z2":     rng.standard_normal(n),
        })
        df["endog1"] = 0.8 * df["z1"] + rng.standard_normal(n) * 0.3
        df["endog2"] = 0.7 * df["z2"] + rng.standard_normal(n) * 0.3
        df["y"]      = 1.5 * df["endog1"] + 0.5 * df["endog2"] + df["x"] + rng.standard_normal(n) * 0.5
        return df

    @pytest.fixture(scope="class")
    def multi_endog_parquet(self, multi_endog_data):
        path = _make_parquet(multi_endog_data)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_two_first_stage_results(self, multi_endog_parquet):
        m = _build(
            multi_endog_parquet,
            "y ~ x | | endog1 + endog2 (z1 + z2)",
            fitter="numpy", remove_singletons=False,
        )
        m.fit(se_method="HC1")
        assert set(m.first_stage.keys()) == {"endog1", "endog2"}

    def test_f_stats_for_both_endogenous(self, multi_endog_parquet):
        m = _build(
            multi_endog_parquet,
            "y ~ x | | endog1 + endog2 (z1 + z2)",
            fitter="numpy", remove_singletons=False,
        )
        m.fit(se_method="HC1")
        fstats = m.get_first_stage_f_stats()
        for k, v in fstats.items():
            assert v is not None and v > 0, f"Expected positive F-stat for {k}"

    def test_coef_names_include_both_endog(self, multi_endog_parquet):
        m = _build(
            multi_endog_parquet,
            "y ~ x | | endog1 + endog2 (z1 + z2)",
            fitter="numpy", remove_singletons=False,
        )
        m.fit(se_method="HC1")
        assert "endog1" in m.coef_names_
        assert "endog2" in m.coef_names_

    def test_two_fitted_columns_in_staging(self, multi_endog_parquet):
        m = _build(
            multi_endog_parquet,
            "y ~ x | | endog1 + endog2 (z1 + z2)",
            fitter="numpy", remove_singletons=False,
        )
        m.prepare_data()
        m._run_first_stages()
        cols = m.conn.execute(
            f"SELECT column_name FROM (DESCRIBE {Duck2SLS._STAGING_TABLE})"
        ).fetchdf()["column_name"].tolist()
        fitted = [c for c in cols if c.startswith("fitted_")]
        assert len(fitted) == 2


# ===========================================================================
# Subset / WHERE clause
# ===========================================================================

class TestSubset:
    def test_subset_reduces_n_obs(self, iv_parquet, iv_panel_data):
        full_n = len(iv_panel_data)
        m = _build(
            iv_parquet, "y ~ x | | endog (z)",
            subset="year >= 2014", remove_singletons=False,
        )
        m.prepare_data()
        assert m.n_obs < full_n
        assert m.n_obs > 0
