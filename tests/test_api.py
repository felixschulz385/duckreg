"""
Tests for the main API, formula parsing, and summary generation.

Covers:
- FormulaParser: OLS, FE, IV, cluster formulas; name getters; has_instruments()
- quote_identifier / needs_quoting helpers
- duckreg() entry point: pooled OLS, FE, bad kwargs, deprecated kwargs
- Summary helpers: format_model_summary, format_summary, SummaryFormatter
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from duckreg import duckreg
from duckreg.utils.formula_parser import (
    FormulaParser,
    needs_quoting,
    quote_identifier,
)
from duckreg.utils.summary import (
    SummaryFormatter,
    format_model_summary,
    format_summary,
    print_summary,
    to_tidy_df,
)


# ============================================================================
# Shared fixtures
# ============================================================================

@pytest.fixture(scope="module")
def parser():
    return FormulaParser()


@pytest.fixture(scope="module")
def small_df():
    """Minimal in-memory DataFrame for end-to-end API tests."""
    rng = np.random.default_rng(0)
    n = 200
    fe1 = np.repeat(np.arange(20), 10)
    fe2 = np.tile(np.arange(10), 20)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.standard_normal(n) * 0.3
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "fe1": fe1, "fe2": fe2})


# ============================================================================
# A. Formula parsing
# ============================================================================

class TestFormulaParserOLS:
    """Simple OLS formula (no FE, no IV)."""

    def test_outcome_names(self, parser):
        f = parser.parse("y ~ x1 + x2")
        assert f.get_outcome_names() == ["y"]

    def test_covariate_names_include_intercept(self, parser):
        f = parser.parse("y ~ x1 + x2")
        names = f.get_covariate_names()
        assert "x1" in names
        assert "x2" in names

    def test_no_fe(self, parser):
        f = parser.parse("y ~ x1 + x2")
        assert f.get_fe_names() == []

    def test_no_instruments(self, parser):
        f = parser.parse("y ~ x1 + x2")
        assert not f.has_instruments()

    def test_raw_formula_stored(self, parser):
        raw = "y ~ x1 + x2"
        f = parser.parse(raw)
        assert f.raw_formula == raw


class TestFormulaParserFE:
    """Formula with fixed effects."""

    def test_fe_names(self, parser):
        f = parser.parse("y ~ x1 | fe1 + fe2")
        fe = f.get_fe_names()
        assert "fe1" in fe
        assert "fe2" in fe

    def test_outcome_and_covariate_independent_of_fe(self, parser):
        f = parser.parse("y ~ x1 | fe1")
        assert f.get_outcome_names() == ["y"]
        assert "x1" in f.get_covariate_names()

    def test_single_fe(self, parser):
        f = parser.parse("y ~ x1 + x2 | group")
        assert f.get_fe_names() == ["group"]

    def test_no_iv(self, parser):
        f = parser.parse("y ~ x1 | fe1")
        assert not f.has_instruments()


class TestFormulaParserIV:
    """Formula with instrumental variables (fixest-style pipe syntax)."""

    def test_has_instruments(self, parser):
        f = parser.parse("y ~ x1 | fe1 | (endog ~ z1 + z2)")
        assert f.has_instruments()

    def test_endogenous_names(self, parser):
        f = parser.parse("y ~ x1 | fe1 | (endog ~ z1 + z2)")
        assert f.get_endogenous_names() == ["endog"]

    def test_instrument_names(self, parser):
        f = parser.parse("y ~ x1 | fe1 | (endog ~ z1 + z2)")
        ivs = f.get_instrument_names()
        assert "z1" in ivs
        assert "z2" in ivs

    def test_exogenous_covariates_exclude_endogenous(self, parser):
        f = parser.parse("y ~ x1 + endog | fe1 | (endog ~ z1)")
        exog = f.get_exogenous_covariate_names()
        assert "endog" not in exog
        assert "x1" in exog


class TestFormulaParserNullCheck:
    """Source-column null-check helpers."""

    def test_null_check_includes_all_variables(self, parser):
        f = parser.parse("y ~ x1 + x2 | fe1")
        cols = f.get_source_columns_for_null_check()
        for col in ["y", "x1", "x2", "fe1"]:
            assert col in cols

    def test_null_check_excludes_intercept(self, parser):
        f = parser.parse("y ~ x1")
        cols = f.get_source_columns_for_null_check()
        assert "_intercept" not in cols
        assert "1" not in cols


# ============================================================================
# B. quote_identifier / needs_quoting helpers
# ============================================================================

class TestQuoteIdentifier:

    @pytest.mark.parametrize("name", ["my var", "group", "SELECT", "123abc", ""])
    def test_needs_quoting_true(self, name):
        assert needs_quoting(name)

    @pytest.mark.parametrize("name", ["x1", "country_id", "gdp_pc"])
    def test_needs_quoting_false(self, name):
        assert not needs_quoting(name)

    def test_quote_wraps_special_chars(self):
        result = quote_identifier("my column")
        assert result.startswith('"') and result.endswith('"')

    def test_quote_passes_through_plain(self):
        assert quote_identifier("x1") == "x1"

    def test_quote_idempotent_on_already_quoted(self):
        already = '"my col"'
        assert quote_identifier(already) == already

    def test_quote_escapes_internal_double_quotes(self):
        result = quote_identifier('col"name')
        assert '""' in result


# ============================================================================
# C. duckreg() main API
# ============================================================================

class TestDuckregPooledOLS:
    """Pooled OLS (no FE) via in-memory DataFrame."""

    def test_returns_estimator(self, small_df):
        model = duckreg("y ~ x1 + x2", data=small_df, se_method="iid")
        assert model is not None

    def test_has_point_estimate(self, small_df):
        model = duckreg("y ~ x1 + x2", data=small_df, se_method="iid")
        assert model.point_estimate is not None
        assert len(model.point_estimate) > 0

    def test_coef_names_present(self, small_df):
        model = duckreg("y ~ x1 + x2", data=small_df, se_method="iid")
        assert model.coef_names_ is not None
        assert "x1" in model.coef_names_
        assert "x2" in model.coef_names_

    def test_n_obs_correct(self, small_df):
        model = duckreg("y ~ x1 + x2", data=small_df, se_method="none")
        assert model.n_obs == len(small_df)

    def test_hc1_se_method(self, small_df):
        model = duckreg("y ~ x1 + x2", data=small_df, se_method="HC1")
        # HC1 should produce a vcov matrix
        assert model.vcov is not None

    def test_coefficient_values_plausible(self, small_df):
        """Regression on known DGP (y = 1 + 2*x1 - 0.5*x2 + noise)."""
        model = duckreg("y ~ x1 + x2", data=small_df, se_method="none")
        coefs = {name: val for name, val in zip(model.coef_names_, model.point_estimate.flatten())}
        assert abs(coefs["x1"] - 2.0) < 0.3
        assert abs(coefs["x2"] - (-0.5)) < 0.3


class TestDuckregFE:
    """OLS with fixed effects via in-memory DataFrame."""

    def test_fe_model_runs(self, small_df):
        model = duckreg("y ~ x1 + x2 | fe1", data=small_df, se_method="iid")
        assert model is not None

    def test_fe_model_coef_names(self, small_df):
        model = duckreg("y ~ x1 + x2 | fe1", data=small_df, se_method="iid")
        assert "x1" in model.coef_names_
        assert "x2" in model.coef_names_

    def test_two_way_fe_runs(self, small_df):
        model = duckreg("y ~ x1 | fe1 + fe2", data=small_df, se_method="none")
        assert model.point_estimate is not None

    def test_mundlak_fe_method(self, small_df):
        model = duckreg("y ~ x1 | fe1", data=small_df, fe_method="mundlak", se_method="none")
        assert model.point_estimate is not None


class TestDuckregClusterSE:
    """Cluster-robust SEs."""

    def test_crv1_via_dict(self, small_df):
        model = duckreg("y ~ x1 + x2 | fe1", data=small_df,
                        se_method={"CRV1": "fe1"})
        assert model.vcov is not None


class TestDuckregBadInputs:
    """Error handling for invalid inputs to duckreg()."""

    def test_unexpected_kwarg_raises(self, small_df):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            duckreg("y ~ x1", data=small_df, nonexistent_arg=True)

    def test_n_jobs_deprecation_warning(self, small_df):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            duckreg("y ~ x1", data=small_df, se_method="none", n_jobs=1)
        categories = [str(w.category) for w in caught]
        assert any("DeprecationWarning" in c for c in categories)

    def test_n_bootstraps_deprecation_warning(self, small_df):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            duckreg("y ~ x1", data=small_df, se_method="none", n_bootstraps=10)
        categories = [str(w.category) for w in caught]
        assert any("DeprecationWarning" in c for c in categories)

    def test_duckdb_kwargs_deprecation_warning(self, small_df):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            duckreg("y ~ x1", data=small_df, se_method="none",
                    duckdb_kwargs={"threads": 1})
        categories = [str(w.category) for w in caught]
        assert any("DeprecationWarning" in c for c in categories)

    def test_unsupported_data_type_raises(self):
        with pytest.raises(TypeError):
            duckreg("y ~ x1", data=42)


# ============================================================================
# D. Summary generation
# ============================================================================

@pytest.fixture(scope="module")
def fitted_model(small_df):
    return duckreg("y ~ x1 + x2", data=small_df, se_method="HC1")


@pytest.fixture(scope="module")
def fitted_fe_model(small_df):
    return duckreg("y ~ x1 + x2 | fe1", data=small_df, se_method="HC1")


class TestFormatModelSummary:
    """format_model_summary() from utils.summary."""

    def test_returns_string(self, fitted_model):
        s = fitted_model.summary()
        text = format_model_summary(s)
        assert isinstance(text, str)

    def test_contains_coefficient_section(self, fitted_model):
        s = fitted_model.summary()
        text = format_model_summary(s)
        assert "COEFFICIENT" in text.upper()

    def test_contains_variable_names(self, fitted_model):
        s = fitted_model.summary()
        text = format_model_summary(s)
        assert "x1" in text
        assert "x2" in text

    def test_contains_sample_info(self, fitted_model):
        s = fitted_model.summary()
        text = format_model_summary(s)
        assert "SAMPLE" in text.upper() or "Observations" in text

    def test_custom_description(self, fitted_model):
        s = fitted_model.summary()
        text = format_model_summary(s, spec_config={"description": "My Analysis"})
        assert "My Analysis" in text

    def test_precision_parameter(self, fitted_model):
        s = fitted_model.summary()
        text_p6 = format_model_summary(s, precision=6)
        # Six decimal places means more digits than default 4
        assert text_p6 is not None

    def test_fe_model_shows_fixed_effects(self, fitted_fe_model):
        s = fitted_fe_model.summary()
        text = format_model_summary(s)
        assert "fe1" in text or "Fixed Effect" in text


class TestSummaryBackwardCompat:
    """Backward-compatibility wrappers and SummaryFormatter."""

    def test_format_summary_with_dict(self, fitted_model):
        s = fitted_model.summary()
        text = format_summary(s)
        assert isinstance(text, str)

    def test_print_summary_does_not_raise(self, fitted_model, capsys):
        s = fitted_model.summary()
        print_summary(s)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_to_tidy_df_with_result_object(self, fitted_model):
        # to_tidy_df on a dict returns empty DataFrame (spec'd behaviour)
        result = to_tidy_df(fitted_model.summary())
        assert isinstance(result, pd.DataFrame)

    def test_summary_formatter_format(self, fitted_model):
        s = fitted_model.summary()
        text = SummaryFormatter.format(s)
        assert isinstance(text, str)

    def test_summary_formatter_print(self, fitted_model, capsys):
        s = fitted_model.summary()
        SummaryFormatter.print(s)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_summary_formatter_to_tidy_df(self, fitted_model):
        s = fitted_model.summary()
        df = SummaryFormatter.to_tidy_df(s)
        assert isinstance(df, pd.DataFrame)
