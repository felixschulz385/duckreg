"""
Unit and integration tests for Duck2SLS.

Covers components *specific* to the 2SLS estimator.  The following aspects
are deliberately NOT re-tested here because they are covered by other suites:

  - test_compression.py       coefficient accuracy vs pyfixest for IV models
  - test_data_prep.py         singleton removal (including for Duck2SLS)
  - test_vcov.py              building blocks of the sandwich estimator
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
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)")
        assert m.method == "mundlak"
        assert m.fitter == "numpy"
        assert m.endogenous_vars == ["endog"]
        assert m.instrument_vars == ["z"]

    def test_fe_method_compat_alias(self, iv_parquet):
        """fe_method= should map to method= for backward compatibility."""
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)", fe_method="mundlak")
        assert m.method == "mundlak"

    def test_method_demean_accepted(self, iv_parquet):
        """method='demean' is now a supported FE absorption strategy."""
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)", method="demean")
        assert m.method == "demean"

    def test_invalid_method_raises(self, iv_parquet):
        """Unsupported method names must raise ValueError."""
        with pytest.raises(ValueError, match="mundlak"):
            _build(iv_parquet, "y ~ x | | (endog ~ z)", method="invalid")

    def test_internal_state_initialized_to_none(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)")
        assert m._ss_transformer is None
        assert m._exog_sql is None
        assert m._fitted_endog_sql is None
        assert m._X_fitted is None
        assert m._Z is None

    def test_formula_extractions(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | firm_id | (endog ~ z)")
        assert "endog" in m.endogenous_vars
        assert "z" in m.instrument_vars
        assert "x" in m.exogenous_vars
        assert "firm_id" in m.fe_cols

    def test_first_stage_results_empty_before_fit(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)")
        assert m.first_stage == {}

    def test_results_none_before_fit(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)")
        assert m.results is None


# ===========================================================================
# Column-name helpers (pure logic, uses post-prepare_data state)
# ===========================================================================

class TestColumnNameHelpers:
    def test_get_exog_sql_excludes_endogenous(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)", remove_singletons=False)
        m.prepare_data()
        exog = m._get_exog_sql()
        # "x" should be present; "endog" must NOT be in exog
        assert any("x" in s for s in exog)
        assert not any("endog" in s for s in exog)

    def test_get_exog_sql_excludes_intercept(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)", remove_singletons=False)
        m.prepare_data()
        exog = m._get_exog_sql()
        assert "Intercept" not in exog
        assert "intercept" not in exog

    def test_get_inst_sql(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)", remove_singletons=False)
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
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)", remove_singletons=False)
        m.prepare_data()
        count = m.conn.execute(
            f"SELECT COUNT(*) FROM {Duck2SLS._STAGING_TABLE}"
        ).fetchone()[0]
        assert count > 0

    def test_staging_table_has_row_idx(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)", remove_singletons=False)
        m.prepare_data()
        cols = set(
            m.conn.execute(
                f"SELECT column_name FROM (DESCRIBE {Duck2SLS._STAGING_TABLE})"
            ).fetchdf()["column_name"].tolist()
        )
        assert "_row_idx" in cols

    def test_staging_table_row_idx_unique(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)", remove_singletons=False)
        m.prepare_data()
        total, unique = m.conn.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT _row_idx) FROM {Duck2SLS._STAGING_TABLE}"
        ).fetchone()
        assert total == unique, "_row_idx values must be unique"

    def test_staging_table_has_outcome_and_endog(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)", remove_singletons=False)
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
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)", remove_singletons=False)
        m.prepare_data()
        assert m.n_obs == 80 * 8


# ===========================================================================
# _add_fitted_column – join-back mechanism
# ===========================================================================

class TestAddFittedColumn:
    def test_fitted_column_appears_in_staging_after_first_stage(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)", remove_singletons=False)
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
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)", remove_singletons=False)
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
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)", remove_singletons=False)
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
            m = _build(path, "y ~ x | | (endog ~ z_weak)", fitter="numpy",
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
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)", remove_singletons=False)
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
# Demean FE absorption path
# ===========================================================================

class TestDemeanPath:
    """Tests for Duck2SLS with method='demean' (MAP-based FE absorption)."""

    @pytest.fixture(scope="class")
    def fitted_demean_fe(self, iv_parquet):
        """Duck2SLS, demean FE, one FE (firm_id), fitted."""
        m = _build(iv_parquet, "y ~ x | firm_id | (endog ~ z)",
                   method="demean", fitter="numpy", remove_singletons=False)
        m.fit(se_method="HC1")
        return m

    def test_demean_fit_produces_results(self, fitted_demean_fe):
        """demean path must produce a valid Results object."""
        assert fitted_demean_fe.results is not None
        assert fitted_demean_fe.vcov is not None

    def test_demean_coef_near_true_value(self, fitted_demean_fe):
        """endog coefficient should be near the DGP value of 2.0."""
        idx = fitted_demean_fe.coef_names_.index("endog")
        coef = fitted_demean_fe.point_estimate.flatten()[idx]
        assert abs(coef - 2.0) < 0.5, (
            f"Demean 2SLS endog coef {coef:.4f} deviates > 0.5 from true value 2.0"
        )

    def test_demean_first_stage_strong(self, fitted_demean_fe):
        """First-stage F-statistic should exceed 10 for the strong instrument."""
        assert fitted_demean_fe.has_weak_instruments() is False

    def test_demean_vcov_is_psd(self, fitted_demean_fe):
        """VCov from demean path must be positive semi-definite."""
        eigvals = np.linalg.eigvalsh(fitted_demean_fe.vcov)
        assert np.all(eigvals >= -1e-10), "Demean 2SLS VCov is not PSD"


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

    def test_endog_coef_vs_pyfixest(self, fitted_numpy, iv_panel_data):
        """Second-stage coefficient must match pyfixest IV oracle (no-FE model) within 5%."""
        pf = pytest.importorskip("pyfixest")
        fit_pf = pf.feols("y ~ x | endog ~ z", data=iv_panel_data, vcov="HC1")
        pf_coef = float(fit_pf.tidy().loc["endog", "Estimate"])
        idx = fitted_numpy.coef_names_.index("endog")
        dr_coef = float(fitted_numpy.point_estimate.flatten()[idx])
        np.testing.assert_allclose(
            dr_coef, pf_coef, rtol=0.05, atol=0.01,
            err_msg=(
                f"2SLS endog coef {dr_coef:.4f} deviates >5% "
                f"from pyfixest oracle {pf_coef:.4f}"
            ),
        )


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
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)", remove_singletons=False)
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

    def test_crv1_se_positive(self, iv_parquet):
        """Cluster-robust (CRV1) SEs must be positive for a clustered 2SLS model."""
        m = _build(iv_parquet, "y ~ x | firm_id | (endog ~ z)",
                   fitter="numpy", remove_singletons=False)
        m.fit(se_method={"CRV1": "firm_id"})
        se = np.sqrt(np.diag(m.vcov))
        assert np.all(se > 0), "CRV1 SEs contain non-positive values"


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
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)")
        df = m.summary_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_to_tidy_df_empty_before_fit(self, iv_parquet):
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)")
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
            "y ~ x | | (endog1 + endog2 ~ z1 + z2)",
            fitter="numpy", remove_singletons=False,
        )
        m.fit(se_method="HC1")
        assert set(m.first_stage.keys()) == {"endog1", "endog2"}

    def test_f_stats_for_both_endogenous(self, multi_endog_parquet):
        m = _build(
            multi_endog_parquet,
            "y ~ x | | (endog1 + endog2 ~ z1 + z2)",
            fitter="numpy", remove_singletons=False,
        )
        m.fit(se_method="HC1")
        fstats = m.get_first_stage_f_stats()
        for k, v in fstats.items():
            assert v is not None and v > 0, f"Expected positive F-stat for {k}"

    def test_coef_names_include_both_endog(self, multi_endog_parquet):
        m = _build(
            multi_endog_parquet,
            "y ~ x | | (endog1 + endog2 ~ z1 + z2)",
            fitter="numpy", remove_singletons=False,
        )
        m.fit(se_method="HC1")
        assert "endog1" in m.coef_names_
        assert "endog2" in m.coef_names_

    def test_two_fitted_columns_in_staging(self, multi_endog_parquet):
        m = _build(
            multi_endog_parquet,
            "y ~ x | | (endog1 + endog2 ~ z1 + z2)",
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
            iv_parquet, "y ~ x | | (endog ~ z)",
            subset="year >= 2014", remove_singletons=False,
        )
        m.prepare_data()
        assert m.n_obs < full_n
        assert m.n_obs > 0


# ===========================================================================
# Over-identification (more instruments than endogenous variables)
# ===========================================================================

class TestOveridentification:
    """Tests for the over-identified 2SLS case: 1 endogenous variable, 2 instruments."""

    @pytest.fixture(scope="class")
    def overid_parquet(self):
        rng = np.random.default_rng(77)
        n = 500
        x  = rng.standard_normal(n)
        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)
        endog = 0.7 * z1 + 0.5 * z2 + rng.standard_normal(n) * 0.3
        y = 2.0 * endog + x + rng.standard_normal(n) * 0.5
        df = pd.DataFrame({"y": y, "x": x, "endog": endog, "z1": z1, "z2": z2})
        path = _make_parquet(df)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_fit_succeeds(self, overid_parquet):
        """Over-identified 2SLS (1 endog, 2 instruments) must fit without error."""
        m = _build(overid_parquet, "y ~ x | | (endog ~ z1 + z2)",
                   fitter="numpy", remove_singletons=False)
        m.fit(se_method="HC1")
        assert m.results is not None

    def test_coef_near_true_value(self, overid_parquet):
        """endog coefficient should be near the DGP value of 2.0."""
        m = _build(overid_parquet, "y ~ x | | (endog ~ z1 + z2)",
                   fitter="numpy", remove_singletons=False)
        m.fit(se_method="HC1")
        idx = m.coef_names_.index("endog")
        coef = m.point_estimate.flatten()[idx]
        assert abs(coef - 2.0) < 0.5, (
            f"Over-identified 2SLS endog coef {coef:.4f} deviates > 0.5 from true value 2.0"
        )

    def test_first_stage_f_both_instruments(self, overid_parquet):
        """First-stage F-statistic using both instruments should exceed 10."""
        m = _build(overid_parquet, "y ~ x | | (endog ~ z1 + z2)",
                   fitter="numpy", remove_singletons=False)
        m.fit(se_method="HC1")
        assert m.has_weak_instruments() is False

    def test_vcov_shape_matches_second_stage(self, overid_parquet):
        """VCov must be square with side equal to the number of second-stage coefs."""
        m = _build(overid_parquet, "y ~ x | | (endog ~ z1 + z2)",
                   fitter="numpy", remove_singletons=False)
        m.fit(se_method="HC1")
        k = len(m.point_estimate.flatten())
        assert m.vcov.shape == (k, k)


# ===========================================================================
# End-to-end SE parity — Duck2SLS vs pyfixest (IV + one-way FE)
# ===========================================================================

class TestEndToEndSEParity2SLS:
    """Numerical SE parity: Duck2SLS (mundlak FE) vs pyfixest.feols, one-way FE.

    Uses the module-level *iv_panel_data* fixture (80 firms × 8 years, strong
    instrument).  For the Mundlak path the FE is firm_id; pyfixest absorbs the
    same FE via within-transformation so coefficients and SEs must agree within
    tight tolerances.
    """

    @pytest.fixture(autouse=True)
    def _pyfixest(self):
        try:
            import pyfixest  # noqa: F401
        except ImportError:
            pytest.skip("pyfixest not available")

    def test_coefs_match_pyfixest(self, iv_parquet, iv_panel_data):
        """Second-stage *endog* coefficient must match pyfixest IV+FE oracle."""
        import pyfixest as pf
        pf_fit = pf.feols(
            "y ~ x | firm_id | endog ~ z",
            data=iv_panel_data, vcov="HC1",
        )
        pf_coef = float(pf_fit.tidy().loc["endog", "Estimate"])

        m = _build(iv_parquet, "y ~ x | firm_id | (endog ~ z)",
                   method="demean", fitter="numpy", remove_singletons=False)
        m.fit(se_method="HC1")
        idx     = m.coef_names_.index("endog")
        dr_coef = float(m.point_estimate.flatten()[idx])

        np.testing.assert_allclose(
            dr_coef, pf_coef, rtol=1e-3,
            err_msg=(
                f"2SLS endog coef {dr_coef:.4f} diverges from "
                f"pyfixest oracle {pf_coef:.4f}"
            ),
        )

    @pytest.mark.parametrize("pf_vcov,dr_vcov,rtol", [
        ("HC1",              "HC1",               1e-2),
        # iid-IV: duckreg/pyfixest differ in σ² normalisation for homoskedastic IV
        ("iid",              "iid",               1e-2),
        ({"CRV1": "firm_id"}, {"CRV1": "firm_id"}, 1e-2),
    ])
    def test_se_parity(self, iv_parquet, iv_panel_data, pf_vcov, dr_vcov, rtol):
        """SE for *endog* must agree with pyfixest within tolerance.

        HC1 and CRV1 use 1 % (matching IV demean tolerance in test_fe_demean.py).
        iid-IV is allowed 10 % because duckreg and pyfixest differ in how σ²
        is normalised for the homoskedastic IV formula.
        """
        import pyfixest as pf
        pf_fit = pf.feols(
            "y ~ x | firm_id | endog ~ z",
            data=iv_panel_data, vcov=pf_vcov,
        )
        pf_se = float(pf_fit.tidy().loc["endog", "Std. Error"])

        m = _build(iv_parquet, "y ~ x | firm_id | (endog ~ z)",
                   method="demean", fitter="numpy", remove_singletons=False)
        m.fit(se_method=dr_vcov)
        idx   = m.coef_names_.index("endog")
        dr_se = float(np.sqrt(np.diag(m.vcov))[idx])

        np.testing.assert_allclose(
            dr_se, pf_se, rtol=rtol,
            err_msg=(
                f"SE mismatch (2SLS demean) for vcov={pf_vcov}: "
                f"duckreg={dr_se:.6f} pyfixest={pf_se:.6f}"
            ),
        )

    def test_duckdb_fitter_se_parity(self, iv_parquet, iv_panel_data):
        """DuckDB fitter SE must agree with pyfixest HC1 oracle within 1 %."""
        import pyfixest as pf
        pf_fit = pf.feols(
            "y ~ x | firm_id | endog ~ z",
            data=iv_panel_data, vcov="HC1",
        )
        pf_se = float(pf_fit.tidy().loc["endog", "Std. Error"])

        m = _build(iv_parquet, "y ~ x | firm_id | (endog ~ z)",
                   method="demean", fitter="duckdb", remove_singletons=False)
        m.fit(se_method="HC1")
        idx   = m.coef_names_.index("endog")
        dr_se = float(np.sqrt(np.diag(m.vcov))[idx])

        np.testing.assert_allclose(
            dr_se, pf_se, rtol=1e-2,
            err_msg=(
                f"DuckDB fitter SE {dr_se:.6f} diverges from "
                f"pyfixest HC1 {pf_se:.6f}"
            ),
        )


# ===========================================================================
# SSC auto-selection for Duck2SLS
# ===========================================================================

class TestSSCAutoSelection2SLS:
    """Verify SSC is automatically determined from formula/se_method for Duck2SLS.

    Mirrors :class:`TestSSCAutoSelection` in *test_fe_demean.py* but exercises
    the 2SLS code path (Duck2SLS with mundlak FE).
    """

    def _fit(self, iv_parquet, se_method):
        m = _build(iv_parquet, "y ~ x | firm_id | (endog ~ z)",
                   fitter="numpy", remove_singletons=False)
        m.fit(se_method=se_method)
        return m

    def test_default_kfixef_is_nonnested(self, iv_parquet):
        """Non-clustered 2SLS should auto-select kfixef='nonnested'."""
        m = self._fit(iv_parquet, "HC1")
        assert m.vcov_spec.ssc.kfixef == 'nonnested'

    def test_default_kadj_is_true(self, iv_parquet):
        """Auto-selected SSC must have kadj=True."""
        m = self._fit(iv_parquet, "HC1")
        assert m.vcov_spec.ssc.kadj is True

    def test_nonclustered_gdf_is_conventional(self, iv_parquet):
        """Non-clustered 2SLS gets Gdf='conventional'."""
        m = self._fit(iv_parquet, "HC1")
        assert m.vcov_spec.ssc.Gdf == 'conventional'

    def test_clustered_gdf_is_min(self, iv_parquet):
        """Clustered 2SLS auto-selects Gdf='min' (matching pyfixest default)."""
        m = self._fit(iv_parquet, {"CRV1": "firm_id"})
        assert m.vcov_spec.ssc.Gdf == 'min'

    def test_ssc_dict_attr_reflects_auto_ssc(self, iv_parquet):
        """ssc_dict must be consistent with vcov_spec.ssc for introspection."""
        m = self._fit(iv_parquet, "HC1")
        assert m.ssc_dict == m.vcov_spec.ssc.to_dict()

    def test_ssc_kfixef_affects_se_numerically(self):
        """kfixef='full' must give strictly larger SE than kfixef='none' when kfe > 0.

        This is a pure-math test using the vcov building-blocks directly,
        independent of the 2SLS estimator.
        """
        from duckreg.core.vcov import SSCConfig, VcovContext, compute_iid_vcov
        from duckreg.core.linalg import safe_inv, safe_solve

        rng = np.random.default_rng(2)
        n, k, kfe, nfe = 640, 3, 80, 1
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        y = X @ np.array([1.0, 0.5, 2.0]) + rng.standard_normal(n)
        XtX    = X.T @ X
        theta  = safe_solve(XtX, X.T @ y)
        XtXinv = safe_inv(XtX, use_pinv=True)
        rss    = float(((y - X @ theta) ** 2).sum())

        ctx      = VcovContext(N=n, k=k, kfe=kfe, nfe=nfe)
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


# ===========================================================================
# Compression correctness for Duck2SLS
# ===========================================================================

class TestCompressionCorrectness2SLS:
    """Verify nobs tracking and round_strata stability for the 2SLS path.

    Mirrors :class:`TestCompressionCorrectness` in *test_fe_demean.py*.
    """

    def test_nobs_is_observation_count_not_strata(self, iv_parquet, iv_panel_data):
        """m.n_obs must equal the number of observations, not compressed strata."""
        m = _build(iv_parquet, "y ~ x | firm_id | (endog ~ z)",
                   fitter="numpy", remove_singletons=False)
        m.fit(se_method="iid")
        assert m.n_obs == len(iv_panel_data), (
            f"m.n_obs={m.n_obs} but len(iv_panel_data)={len(iv_panel_data)}; "
            "N reflects compressed row count instead of original observations"
        )

    def test_vcov_meta_n_equals_nobs(self, iv_parquet, iv_panel_data):
        """vcov_meta['N'] must match the un-compressed observation count."""
        m = _build(iv_parquet, "y ~ x | firm_id | (endog ~ z)",
                   fitter="numpy", remove_singletons=False)
        m.fit(se_method="iid")
        assert m.vcov_meta.get('N', m.n_obs) == len(iv_panel_data)

    def test_round_strata_se_stability(self, iv_parquet):
        """round_strata=5 must not shift SEs by more than 1 %."""
        m_exact = _build(iv_parquet, "y ~ x | firm_id | (endog ~ z)",
                         fitter="numpy", remove_singletons=False)
        m_exact.fit(se_method="HC1")

        m_rounded = _build(iv_parquet, "y ~ x | firm_id | (endog ~ z)",
                           fitter="numpy", remove_singletons=False,
                           round_strata=5)
        m_rounded.fit(se_method="HC1")

        se_exact   = np.sqrt(np.diag(m_exact.vcov))
        se_rounded = np.sqrt(np.diag(m_rounded.vcov))

        np.testing.assert_allclose(
            se_rounded, se_exact, rtol=0.01,
            err_msg=(
                "round_strata=5 caused >1 % SE deviation for Duck2SLS. "
                "Check that rounding affects strata formation only, "
                "not the residual or score computation."
            ),
        )

    def test_df_compressed_structure(self, iv_parquet):
        """After fitting, df_compressed must contain the standard aggregation columns."""
        m = _build(iv_parquet, "y ~ x | firm_id | (endog ~ z)",
                   fitter="duckdb", remove_singletons=False)
        m.fit(se_method="iid")
        for col in ("count", "sum_y", "sum_y_sq"):
            assert col in m.df_compressed.columns, (
                f"Missing column {col!r} in df_compressed"
            )
        assert m.df_compressed["count"].sum() == m.n_obs

    def test_no_fe_nobs_is_full_dataset(self, iv_parquet, iv_panel_data):
        """Even without FE, n_obs must equal total observations."""
        m = _build(iv_parquet, "y ~ x | | (endog ~ z)",
                   fitter="numpy", remove_singletons=False)
        m.fit(se_method="HC1")
        assert m.n_obs == len(iv_panel_data)


# ===========================================================================
# Singleton removal for Duck2SLS
# ===========================================================================

def _make_singleton_iv_panel() -> pd.DataFrame:
    """Panel with 6 firms × 5 years plus one singleton firm (id 99)."""
    rng = np.random.default_rng(17)
    rows = [
        {
            "firm_id": f, "year": t,
            "x": float(rng.standard_normal()),
            "z": float(rng.standard_normal()),
        }
        for f in range(6) for t in range(5)
    ]
    rows.append({"firm_id": 99, "year": 0,
                 "x": float(rng.standard_normal()),
                 "z": float(rng.standard_normal())})
    df = pd.DataFrame(rows)
    firm_fe = {f: rng.standard_normal() for f in df["firm_id"].unique()}
    df["endog"] = 0.8 * df["z"] + df["firm_id"].map(firm_fe) + rng.standard_normal(len(df)) * 0.3
    df["y"] = 2.0 * df["endog"] + df["x"] + df["firm_id"].map(firm_fe) + rng.standard_normal(len(df)) * 0.5
    return df


@pytest.fixture(scope="module")
def singleton_iv_parquet():
    df = _make_singleton_iv_panel()
    path = _make_parquet(df)
    yield path, len(df)  # 31 rows
    if os.path.exists(path):
        os.unlink(path)


class TestSingletonRemoval2SLS:
    """Verify singleton group removal for Duck2SLS."""

    def test_removes_singleton_by_default(self, singleton_iv_parquet):
        path, total = singleton_iv_parquet
        m = _build(path, "y ~ x | firm_id | (endog ~ z)", remove_singletons=True)
        m.fit(se_method="HC1")
        assert m.n_obs == total - 1

    def test_keeps_singletons_when_disabled(self, singleton_iv_parquet):
        path, total = singleton_iv_parquet
        m = _build(path, "y ~ x | firm_id | (endog ~ z)", remove_singletons=False)
        m.fit(se_method="HC1")
        assert m.n_obs == total

    def test_estimation_succeeds_after_removal(self, singleton_iv_parquet):
        path, _ = singleton_iv_parquet
        m = _build(path, "y ~ x | firm_id | (endog ~ z)", remove_singletons=True)
        m.fit(se_method="HC1")
        assert m.results is not None
        assert m.vcov is not None
        se = np.sqrt(np.diag(m.vcov))
        assert np.all(se > 0), "SEs must be positive after singleton removal"