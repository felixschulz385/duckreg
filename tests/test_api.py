"""Tests for the main API, formula parsing, and summary generation."""

import numpy as np
import pandas as pd
import pytest
import duckdb

from duckreg import duckreg
from duckreg.utils.formula_parser import (
    FormulaParser,
    Formula,
    MergedFixedEffect,
    Variable,
    VariableRole,
    needs_quoting,
    quote_identifier,
)
from duckreg.utils.summary import (
    format_model_summary,
    format_summary,
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

    def test_nested_fe_absorbs_only_child(self, parser):
        f = parser.parse("y ~ x1 | pixel_id %in% country + country^year")
        assert f.get_fe_names() == ["pixel_id", "country_year"]
        nesting = f.get_fe_nesting()
        assert len(nesting) == 1
        assert nesting[0].child_name == "pixel_id"
        assert nesting[0].parent_name == "country"

    def test_fe_segment_star_raises_clear_error(self, parser):
        with pytest.raises(ValueError, match="uses '\\^' instead of '\\*'"):
            parser.parse("y ~ x1 | country*year")

    def test_chained_nested_fe_is_rejected(self, parser):
        with pytest.raises(ValueError, match="exactly 'child %in% parent'"):
            parser.parse("y ~ x1 | pixel_id %in% adm2 %in% country")


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

    def test_legacy_iv_syntax_raises_clear_error(self, parser):
        with pytest.raises(ValueError, match="fixest-style syntax"):
            parser.parse("y ~ x1 | fe1 | endog(z1 + z2)")


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

    def test_expression_outcome_is_not_treated_as_source_column(self, parser):
        f = parser.parse("(y == 1) ~ x1 | fe1")
        assert f.outcomes[0].is_expression
        assert "(y == 1)" not in f.get_source_columns_for_null_check()
        assert "((y = 1)) IS NOT NULL" in f.get_where_clause_sql()

    def test_quadratic_expression_stays_numeric(self, parser):
        f = parser.parse("y ~ x + x^2")
        quad = f.get_covariate_by_name("x^2")
        assert quad is not None
        assert quad.is_expression
        assert not quad.expression_is_boolean
        assert quad.sql_name == "x_pow_2"
        assert f.get_covariates_select_sql() == "x AS x, POW(x, 2) AS x_pow_2"

    def test_identity_wrapped_quadratic_is_supported(self, parser):
        f = parser.parse("y ~ x + I((x + 1)^2)")
        quad = f.get_covariate_by_name("(x + 1)^2")
        assert quad is not None
        assert quad.is_expression
        assert f.get_covariates_select_sql() == "x AS x, POW((x + 1), 2) AS x_1_pow_2"

    def test_squared_transform_expression_uses_sql_transform(self, parser):
        f = parser.parse("y ~ log(ntl_harm + 0.01) + log(ntl_harm + 0.01)^2")
        assert f.get_covariate_display_names() == [
            "log(ntl_harm+0.01)",
            "log(ntl_harm + 0.01)^2",
        ]
        assert f.get_covariates_select_sql() == (
            "LN((ntl_harm + 0.01)) AS log_ntl_harm_0_01, "
            "POW(LN((ntl_harm + 0.01)), 2) AS log_ntl_harm_0_01_pow_2"
        )
        cols = f.get_source_columns_for_null_check()
        assert cols == ["y", "ntl_harm"] or cols == ["ntl_harm", "y"]

    def test_nested_fe_parent_is_included_in_null_check(self, parser):
        f = parser.parse("y ~ x1 | pixel_id %in% country + country^year")
        cols = f.get_source_columns_for_null_check()
        assert "country" in cols

    def test_resolve_numeric_merge_accepts_relation_expression(self):
        conn = duckdb.connect()
        formula = Formula(
            outcomes=(),
            covariates=(),
            interactions=(),
            fixed_effects=(),
            merged_fes=(
                MergedFixedEffect(
                    name="country_year",
                    sql_name="country_year",
                    components=(
                        Variable("country", VariableRole.FIXED_EFFECT),
                        Variable("year", VariableRole.FIXED_EFFECT),
                    ),
                ),
            ),
            cluster=None,
            raw_formula="",
        )

        resolved = FormulaParser.resolve_numeric_merge(
            formula,
            conn,
            "(SELECT 1 AS country, 2000 AS year)",
        )

        assert resolved.merged_fes[0].use_numeric_merge


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

    def test_quadratic_formula_runs(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(-2.0, 2.0, 400)
        y = 1.0 + 2.0 * x + 3.0 * (x ** 2) + rng.standard_normal(len(x)) * 0.1
        df = pd.DataFrame({"y": y, "x": x})

        model = duckreg("y ~ x + x^2", data=df, se_method="none")

        coefs = {name: val for name, val in zip(model.coef_names_, model.point_estimate.flatten())}
        assert abs(coefs["x"] - 2.0) < 0.2
        assert abs(coefs["x^2"] - 3.0) < 0.2

    def test_squared_log_formula_runs(self):
        rng = np.random.default_rng(2)
        ntl_harm = rng.uniform(0.05, 3.0, 400)
        log_term = np.log(ntl_harm + 0.01)
        y = 1.0 + 1.5 * log_term - 0.75 * (log_term ** 2) + rng.standard_normal(len(ntl_harm)) * 0.05
        df = pd.DataFrame({"y": y, "ntl_harm": ntl_harm})

        model = duckreg(
            "y ~ log(ntl_harm + 0.01) + log(ntl_harm + 0.01)^2",
            data=df,
            se_method="none",
        )

        coefs = {name: val for name, val in zip(model.coef_names_, model.point_estimate.flatten())}
        assert abs(coefs["log(ntl_harm+0.01)"] - 1.5) < 0.2
        assert abs(coefs["log(ntl_harm + 0.01)^2"] - (-0.75)) < 0.2


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

    def test_fe_accepts_expression_outcome(self, small_df):
        df = small_df.assign(y_class=(small_df["y"] > small_df["y"].median()).astype(int))
        model = duckreg(
            "(y_class == 1) ~ x1 | fe1 + fe2",
            data=df,
            se_method="none",
            fe_method="demean",
            fitter="duckdb",
        )
        assert model.n_obs == len(df)
        assert model.point_estimate is not None

    def test_mundlak_fe_method(self, small_df):
        with pytest.raises(NotImplementedError, match="temporarily disabled"):
            duckreg("y ~ x1 | fe1", data=small_df, fe_method="mundlak", se_method="none")


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

    @pytest.mark.parametrize(
        "kwarg, value",
        [
            ("n_jobs", 1),
            ("n_bootstraps", 10),
            ("duckdb_kwargs", {"threads": 1}),
            ("bootstrap", {"n": 50}),
            ("round_strata", 5),
        ],
    )
    def test_removed_legacy_kwargs_raise(self, small_df, kwarg, value):
        with pytest.raises(TypeError, match="unsupported legacy keyword argument"):
            duckreg("y ~ x1", data=small_df, se_method="none", **{kwarg: value})

    def test_bootstrap_se_method_raises(self, small_df):
        with pytest.raises(ValueError, match="Bootstrap standard errors are no longer supported"):
            duckreg("y ~ x1", data=small_df, se_method="BS")

    def test_fourth_pipe_cluster_segment_raises(self, small_df):
        with pytest.raises(ValueError, match="4th pipe segment"):
            duckreg("y ~ x1 | fe1 | | fe2", data=small_df, se_method={"CRV1": "fe2"})

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
        s = fitted_model.as_dict()
        text = format_model_summary(s)
        assert isinstance(text, str)

    def test_contains_coefficient_section(self, fitted_model):
        s = fitted_model.as_dict()
        text = format_model_summary(s)
        assert "COEFFICIENT" in text.upper()

    def test_contains_variable_names(self, fitted_model):
        s = fitted_model.as_dict()
        text = format_model_summary(s)
        assert "x1" in text
        assert "x2" in text

    def test_contains_sample_info(self, fitted_model):
        s = fitted_model.as_dict()
        text = format_model_summary(s)
        assert "SAMPLE" in text.upper() or "Observations" in text

    def test_custom_description(self, fitted_model):
        s = fitted_model.as_dict()
        text = format_model_summary(s, spec_config={"description": "My Analysis"})
        assert "My Analysis" in text

    def test_precision_parameter(self, fitted_model):
        s = fitted_model.as_dict()
        text_p6 = format_model_summary(s, precision=6)
        # Six decimal places means more digits than default 4
        assert text_p6 is not None

    def test_fe_model_shows_fixed_effects(self, fitted_fe_model):
        s = fitted_fe_model.as_dict()
        text = format_model_summary(s)
        assert "fe1" in text or "Fixed Effect" in text

    def test_compression_uses_compression_base_rows(self):
        summary = {
            "version_info": {"duckreg_version": "test", "computed_at": "now"},
            "model_spec": {
                "estimator_type": "Duck2SLS",
                "outcome_vars": ["y"],
            },
            "sample_info": {
                "n_obs": 95,
                "n_compressed": 100,
                "n_compression_base_rows": 120,
                "compression_ratio": 1 - 100 / 120,
            },
        }

        text = format_model_summary(summary)

        assert "Observations (final): 95" in text
        assert "Compression Base Rows: 120" in text
        assert "Compression: 16.7% reduction (120 \u2192 100 rows)" in text


class TestResultApiAlignment:
    def test_summary_returns_human_readable_string(self, fitted_model):
        text = fitted_model.summary()
        assert isinstance(text, str)
        assert "COEFFICIENT RESULTS" in text

    def test_as_dict_preserves_structured_export(self, fitted_model):
        data = fitted_model.as_dict()
        assert "model_spec" in data
        assert "coefficients" in data

    def test_format_summary_with_dict(self, fitted_model):
        text = format_summary(fitted_model.as_dict())
        assert isinstance(text, str)

    def test_tidy_returns_dataframe(self, fitted_model):
        df = fitted_model.tidy()
        assert isinstance(df, pd.DataFrame)
        assert "estimate" in df.columns

    def test_removed_aliases_are_not_present(self, fitted_model):
        assert not hasattr(fitted_model, "summary_df")
        assert not hasattr(fitted_model, "to_tidy_df")
        assert not hasattr(fitted_model, "print_summary")

    def test_coef_se_tstat_pvalue_confint(self, fitted_model):
        assert isinstance(fitted_model.coef(), pd.Series)
        assert isinstance(fitted_model.se(), pd.Series)
        assert isinstance(fitted_model.tstat(), pd.Series)
        assert isinstance(fitted_model.pvalue(), pd.Series)
        assert isinstance(fitted_model.confint(), pd.DataFrame)

    def test_accessors_raise_before_fit(self, fitted_model):
        model = fitted_model
        model._results = None
        model.point_estimate = None
        with pytest.raises(ValueError, match="Call fit\\(\\) first"):
            model.summary()
        with pytest.raises(ValueError, match="Call fit\\(\\) first"):
            model.tidy()
