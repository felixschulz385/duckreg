"""
Tests for DuckMediation — generalized linear mediation estimator.

Covers:
1.  No FE, NumPy backend
2.  No FE, DuckDB backend (falls back to numpy for equation OLS)
3.  Mundlak FE path
4.  Demean FE path
5.  Clustered SE path
6.  Zero mediators (plain outcome regression)
7.  One mediator
8.  Multiple mediators
9.  Multiple exposures
10. Controls included
11. Coefficient recovery for known DGP
12. Indirect effects = product of estimated paths
13. Total indirect = sum of specific indirect effects
14. Total effect = direct + total indirect
15. Tidy / summary output contains expected labels and rows
"""

import numpy as np
import pandas as pd
import pytest

from duckreg import DuckMediation, MediationResults, MediationEffects


# ============================================================================
# Fixtures — synthetic data
# ============================================================================

@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


def _make_df(
    n: int,
    n_units: int,
    rng: np.random.Generator,
    a1: float = 0.8,
    a2: float = -0.5,
    b1: float = 1.2,
    b2: float = 0.6,
    c_prime: float = 0.3,
    gamma: float = 0.5,   # control coefficient in outcome
    psi1: float = 0.4,    # control coefficient in mediator 1
    psi2: float = 0.2,    # control coefficient in mediator 2
    fe_sd: float = 0.5,
) -> pd.DataFrame:
    """Generate a synthetic mediation dataset.

    True coefficients
    -----------------
    M1 = a1 * X + psi1 * C + unit_fe + noise
    M2 = a2 * X + psi2 * C + unit_fe + noise
    Y  = c_prime * X + b1 * M1 + b2 * M2 + gamma * C + unit_fe + noise
    """
    unit = rng.integers(0, n_units, size=n)
    fe   = rng.normal(0, fe_sd, size=n_units)[unit]

    X = rng.normal(0, 1, n)
    C = rng.normal(0, 1, n)

    M1 = a1 * X + psi1 * C + fe + rng.normal(0, 0.5, n)
    M2 = a2 * X + psi2 * C + fe + rng.normal(0, 0.5, n)
    Y  = c_prime * X + b1 * M1 + b2 * M2 + gamma * C + fe + rng.normal(0, 0.5, n)

    return pd.DataFrame({
        "y": Y, "x": X, "m1": M1, "m2": M2, "c": C, "unit": unit
    })


@pytest.fixture(scope="module")
def df_basic(rng) -> pd.DataFrame:
    return _make_df(n=2000, n_units=50, rng=rng)


@pytest.fixture(scope="module")
def df_no_fe(rng) -> pd.DataFrame:
    """Dataset where FE are zero — simpler coefficient recovery."""
    n = 2000
    X  = rng.normal(0, 1, n)
    C  = rng.normal(0, 1, n)
    M1 = 0.8 * X + 0.4 * C + rng.normal(0, 0.3, n)
    M2 = -0.5 * X + 0.2 * C + rng.normal(0, 0.3, n)
    Y  = 0.3 * X + 1.2 * M1 + 0.6 * M2 + 0.5 * C + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": Y, "x": X, "m1": M1, "m2": M2, "c": C})


# ============================================================================
# Helper: fit DuckMediation from a DataFrame
# ============================================================================

def fit_mediation(
    df: pd.DataFrame,
    outcome: str = "y",
    exposures: list = None,
    mediators: list = None,
    controls: list = None,
    fe_cols: list = None,
    fe_method: str = "demean",
    cluster_col: str = None,
    se_method: str = "HC1",
    fitter: str = "numpy",
) -> DuckMediation:
    import duckdb
    conn = duckdb.connect(":memory:")
    conn.register("data", df)

    med = DuckMediation(
        db_name=":memory:",
        table_name="data",
        outcome=outcome,
        exposures=exposures or ["x"],
        mediators=mediators if mediators is not None else ["m1"],
        controls=controls or [],
        fe_cols=fe_cols or [],
        fe_method=fe_method,
        cluster_col=cluster_col,
        fitter=fitter,
        seed=42,
    )
    # Register the data in the estimator's own connection
    med.conn.register("data", df)
    med.fit(se_method=se_method)
    return med


# ============================================================================
# 1. No FE, NumPy backend — basic smoke test
# ============================================================================

class TestNoFENumpy:
    def test_fit_runs(self, df_no_fe):
        med = fit_mediation(df_no_fe, fe_cols=[], fe_method=None)
        assert med.mediation_results is not None

    def test_n_obs(self, df_no_fe):
        med = fit_mediation(df_no_fe, fe_cols=[], fe_method=None)
        assert med.n_obs == len(df_no_fe)

    def test_outcome_coef_names(self, df_no_fe):
        med = fit_mediation(df_no_fe, fe_cols=[], mediators=["m1"],
                            fe_method=None)
        names = med.mediation_results.outcome_result.coef_names
        assert "Intercept" in names
        assert "x" in names
        assert "m1" in names

    def test_mediator_coef_names(self, df_no_fe):
        med = fit_mediation(df_no_fe, fe_cols=[], mediators=["m1"],
                            fe_method=None)
        names = med.mediation_results.mediator_results["m1"].coef_names
        assert "Intercept" in names
        assert "x" in names

    def test_effects_not_none(self, df_no_fe):
        med = fit_mediation(df_no_fe, fe_cols=[], mediators=["m1"],
                            fe_method=None)
        assert med.mediation_results.effects is not None


# ============================================================================
# 2. No FE, DuckDB backend
# ============================================================================

class TestNoFEDuckDB:
    def test_fit_duckdb(self, df_no_fe):
        med = fit_mediation(df_no_fe, fe_cols=[], fe_method=None,
                            fitter="duckdb")
        assert med.mediation_results is not None
        assert med.n_obs == len(df_no_fe)


# ============================================================================
# 3. Mundlak FE path
# ============================================================================

class TestMundlakFE:
    def test_fit_runs(self, df_basic):
        med = fit_mediation(df_basic, fe_cols=["unit"], fe_method="mundlak",
                            mediators=["m1"])
        assert med.mediation_results is not None

    def test_n_obs(self, df_basic):
        med = fit_mediation(df_basic, fe_cols=["unit"], fe_method="mundlak",
                            mediators=["m1"])
        assert med.n_obs == len(df_basic)

    def test_effects_present(self, df_basic):
        med = fit_mediation(df_basic, fe_cols=["unit"], fe_method="mundlak",
                            mediators=["m1"])
        assert med.mediation_results.effects is not None


# ============================================================================
# 4. Demean FE path
# ============================================================================

class TestDemeanFE:
    def test_fit_runs(self, df_basic):
        med = fit_mediation(df_basic, fe_cols=["unit"], fe_method="demean",
                            mediators=["m1"])
        assert med.mediation_results is not None

    def test_no_intercept_in_demean(self, df_basic):
        med = fit_mediation(df_basic, fe_cols=["unit"], fe_method="demean",
                            mediators=["m1"])
        # Demean path: no 'Intercept' term in coefficient names
        names = med.mediation_results.outcome_result.coef_names
        assert "Intercept" not in names

    def test_effects_present(self, df_basic):
        med = fit_mediation(df_basic, fe_cols=["unit"], fe_method="demean",
                            mediators=["m1"])
        assert med.mediation_results.effects is not None

    def test_df_correction_positive(self, df_basic):
        med = fit_mediation(df_basic, fe_cols=["unit"], fe_method="demean",
                            mediators=["m1"])
        # df_correction should be > 0 (absorbed FE levels)
        assert med._df_correction > 0


# ============================================================================
# 5. Clustered SE path
# ============================================================================

class TestClusteredSE:
    def test_crv1_via_dict(self, df_basic):
        med = fit_mediation(
            df_basic, fe_cols=[], fe_method=None,
            mediators=["m1"], se_method={"CRV1": "unit"}
        )
        assert med.se == "CRV1"
        outcome_res = med.mediation_results.outcome_result
        assert outcome_res.vcov is not None

    def test_crv1_via_cluster_col(self, df_basic):
        med = fit_mediation(
            df_basic, fe_cols=[], fe_method=None,
            mediators=["m1"], cluster_col="unit", se_method="CRV1"
        )
        assert med.se == "CRV1"


# ============================================================================
# 6. Zero mediators
# ============================================================================

class TestZeroMediators:
    def test_no_mediator_eq(self, df_no_fe):
        med = fit_mediation(df_no_fe, mediators=[], fe_cols=[], fe_method=None)
        assert len(med.mediation_results.mediator_results) == 0

    def test_effects_direct_only(self, df_no_fe):
        med = fit_mediation(df_no_fe, mediators=[], fe_cols=[], fe_method=None)
        eff = med.mediation_results.effects
        assert eff is not None
        assert len(eff.exposure_names) == 1
        assert len(eff.mediator_names) == 0
        # Total indirect should be 0
        np.testing.assert_array_almost_equal(eff.total_indirect, [0.0])
        # Total = direct
        np.testing.assert_array_almost_equal(eff.total, eff.direct)


# ============================================================================
# 7. One mediator
# ============================================================================

class TestOneMediator:
    def test_indirect_equals_a_times_b(self, df_no_fe):
        """IE = A * B (product of estimated path coefficients)."""
        med = fit_mediation(df_no_fe, mediators=["m1"], fe_cols=[],
                            fe_method=None)
        eff = med.mediation_results.effects
        # A = coefficient of x in mediator equation
        m1_coefs = med.mediation_results.mediator_results["m1"].coefficients.flatten()
        m1_names = med.mediation_results.mediator_results["m1"].coef_names
        a_idx = m1_names.index("x")
        A = m1_coefs[a_idx]
        # B = coefficient of m1 in outcome equation
        out_coefs = med.mediation_results.outcome_result.coefficients.flatten()
        out_names = med.mediation_results.outcome_result.coef_names
        b_idx = out_names.index("m1")
        B = out_coefs[b_idx]

        expected_ie = A * B
        np.testing.assert_allclose(eff.indirect[0, 0], expected_ie, rtol=1e-10)

    def test_total_indirect_equals_indirect(self, df_no_fe):
        """With one mediator, TIE == IE[0, 0]."""
        med = fit_mediation(df_no_fe, mediators=["m1"], fe_cols=[],
                            fe_method=None)
        eff = med.mediation_results.effects
        np.testing.assert_allclose(eff.total_indirect[0], eff.indirect[0, 0], rtol=1e-12)

    def test_total_equals_direct_plus_total_indirect(self, df_no_fe):
        med = fit_mediation(df_no_fe, mediators=["m1"], fe_cols=[],
                            fe_method=None)
        eff = med.mediation_results.effects
        np.testing.assert_allclose(eff.total[0], eff.direct[0] + eff.total_indirect[0], rtol=1e-12)

    def test_direct_se_positive(self, df_no_fe):
        med = fit_mediation(df_no_fe, mediators=["m1"], fe_cols=[],
                            fe_method=None)
        eff = med.mediation_results.effects
        assert eff.direct_se is not None
        assert eff.direct_se[0] > 0

    def test_indirect_se_positive(self, df_no_fe):
        med = fit_mediation(df_no_fe, mediators=["m1"], fe_cols=[],
                            fe_method=None)
        eff = med.mediation_results.effects
        assert eff.indirect_se is not None
        assert eff.indirect_se[0, 0] > 0


# ============================================================================
# 8. Multiple mediators
# ============================================================================

class TestMultipleMediators:
    def test_two_mediator_equations(self, df_no_fe):
        med = fit_mediation(df_no_fe, mediators=["m1", "m2"], fe_cols=[],
                            fe_method=None)
        assert len(med.mediation_results.mediator_results) == 2
        assert "m1" in med.mediation_results.mediator_results
        assert "m2" in med.mediation_results.mediator_results

    def test_indirect_shape(self, df_no_fe):
        med = fit_mediation(df_no_fe, mediators=["m1", "m2"], fe_cols=[],
                            fe_method=None)
        eff = med.mediation_results.effects
        # 1 exposure × 2 mediators
        assert eff.indirect.shape == (1, 2)
        assert eff.indirect_se.shape == (1, 2)

    def test_each_indirect_equals_product(self, df_no_fe):
        """IE[0,k] = A_k * B_k for each mediator k."""
        med = fit_mediation(df_no_fe, mediators=["m1", "m2"], fe_cols=[],
                            fe_method=None)
        eff = med.mediation_results.effects
        out_coefs = med.mediation_results.outcome_result.coefficients.flatten()
        out_names = med.mediation_results.outcome_result.coef_names

        for k, med_name in enumerate(["m1", "m2"]):
            m_coefs = med.mediation_results.mediator_results[med_name].coefficients.flatten()
            m_names = med.mediation_results.mediator_results[med_name].coef_names
            A = m_coefs[m_names.index("x")]
            B = out_coefs[out_names.index(med_name)]
            expected = A * B
            np.testing.assert_allclose(eff.indirect[0, k], expected, rtol=1e-10)

    def test_tie_equals_sum_of_specific_ie(self, df_no_fe):
        med = fit_mediation(df_no_fe, mediators=["m1", "m2"], fe_cols=[],
                            fe_method=None)
        eff = med.mediation_results.effects
        expected_tie = eff.indirect[0, 0] + eff.indirect[0, 1]
        np.testing.assert_allclose(eff.total_indirect[0], expected_tie, rtol=1e-12)

    def test_total_equals_direct_plus_tie(self, df_no_fe):
        med = fit_mediation(df_no_fe, mediators=["m1", "m2"], fe_cols=[],
                            fe_method=None)
        eff = med.mediation_results.effects
        np.testing.assert_allclose(
            eff.total[0], eff.direct[0] + eff.total_indirect[0], rtol=1e-12
        )


# ============================================================================
# 9. Multiple exposures
# ============================================================================

class TestMultipleExposures:
    @pytest.fixture(scope="class")
    def df_multi_x(self, rng) -> pd.DataFrame:
        n = 2000
        X1 = rng.normal(0, 1, n)
        X2 = rng.normal(0, 1, n)
        M1 = 0.6 * X1 - 0.4 * X2 + rng.normal(0, 0.3, n)
        Y  = 0.2 * X1 + 0.5 * X2 + 0.8 * M1 + rng.normal(0, 0.3, n)
        return pd.DataFrame({"y": Y, "x1": X1, "x2": X2, "m1": M1})

    def test_indirect_shape_multi_x(self, df_multi_x):
        med = fit_mediation(
            df_multi_x,
            exposures=["x1", "x2"], mediators=["m1"],
            fe_cols=[], fe_method=None,
        )
        eff = med.mediation_results.effects
        assert eff.indirect.shape == (2, 1)

    def test_direct_shape_multi_x(self, df_multi_x):
        med = fit_mediation(
            df_multi_x,
            exposures=["x1", "x2"], mediators=["m1"],
            fe_cols=[], fe_method=None,
        )
        eff = med.mediation_results.effects
        assert eff.direct.shape == (2,)

    def test_total_indirect_shape_multi_x(self, df_multi_x):
        med = fit_mediation(
            df_multi_x,
            exposures=["x1", "x2"], mediators=["m1"],
            fe_cols=[], fe_method=None,
        )
        eff = med.mediation_results.effects
        assert eff.total_indirect.shape == (2,)

    def test_indirect_equals_product_multi_x(self, df_multi_x):
        """IE[j, 0] = A_0[j] * B[0] for both exposures."""
        med = fit_mediation(
            df_multi_x,
            exposures=["x1", "x2"], mediators=["m1"],
            fe_cols=[], fe_method=None,
        )
        eff = med.mediation_results.effects
        m1_coefs = med.mediation_results.mediator_results["m1"].coefficients.flatten()
        m1_names = med.mediation_results.mediator_results["m1"].coef_names
        out_coefs = med.mediation_results.outcome_result.coefficients.flatten()
        out_names = med.mediation_results.outcome_result.coef_names

        B = out_coefs[out_names.index("m1")]
        for j, xname in enumerate(["x1", "x2"]):
            A_j = m1_coefs[m1_names.index(xname)]
            np.testing.assert_allclose(eff.indirect[j, 0], A_j * B, rtol=1e-10)


# ============================================================================
# 10. Controls included
# ============================================================================

class TestWithControls:
    def test_control_in_coef_names(self, df_no_fe):
        med = fit_mediation(
            df_no_fe, mediators=["m1"], controls=["c"],
            fe_cols=[], fe_method=None
        )
        names = med.mediation_results.outcome_result.coef_names
        assert "c" in names
        med_names = med.mediation_results.mediator_results["m1"].coef_names
        assert "c" in med_names

    def test_exposure_coef_ignores_control_index(self, df_no_fe):
        """Exposure index in mediator equation must still be correct when controls present."""
        med = fit_mediation(
            df_no_fe, mediators=["m1"], controls=["c"],
            fe_cols=[], fe_method=None
        )
        eff = med.mediation_results.effects
        # Direct structural check: IE = A * B
        m1_coefs = med.mediation_results.mediator_results["m1"].coefficients.flatten()
        m1_names = med.mediation_results.mediator_results["m1"].coef_names
        out_coefs = med.mediation_results.outcome_result.coefficients.flatten()
        out_names = med.mediation_results.outcome_result.coef_names
        A = m1_coefs[m1_names.index("x")]
        B = out_coefs[out_names.index("m1")]
        np.testing.assert_allclose(eff.indirect[0, 0], A * B, rtol=1e-10)


# ============================================================================
# 11. Coefficient recovery (large-sample consistency check)
# ============================================================================

class TestCoefficientRecovery:
    TRUE_A = 0.8       # x → m1
    TRUE_B = 1.2       # m1 → y
    TRUE_CP = 0.3      # direct x → y

    @pytest.fixture(scope="class")
    def df_large(self, rng) -> pd.DataFrame:
        n = 10_000
        X  = rng.normal(0, 1, n)
        M1 = self.TRUE_A * X + rng.normal(0, 0.3, n)
        Y  = self.TRUE_CP * X + self.TRUE_B * M1 + rng.normal(0, 0.3, n)
        return pd.DataFrame({"y": Y, "x": X, "m1": M1})

    def test_a_recovered(self, df_large):
        med = fit_mediation(df_large, mediators=["m1"], fe_cols=[], fe_method=None)
        m1_coefs = med.mediation_results.mediator_results["m1"].coefficients.flatten()
        m1_names = med.mediation_results.mediator_results["m1"].coef_names
        A_hat = m1_coefs[m1_names.index("x")]
        assert abs(A_hat - self.TRUE_A) < 0.05

    def test_b_recovered(self, df_large):
        med = fit_mediation(df_large, mediators=["m1"], fe_cols=[], fe_method=None)
        out_coefs = med.mediation_results.outcome_result.coefficients.flatten()
        out_names = med.mediation_results.outcome_result.coef_names
        B_hat = out_coefs[out_names.index("m1")]
        assert abs(B_hat - self.TRUE_B) < 0.05

    def test_direct_recovered(self, df_large):
        med = fit_mediation(df_large, mediators=["m1"], fe_cols=[], fe_method=None)
        eff = med.mediation_results.effects
        assert abs(eff.direct[0] - self.TRUE_CP) < 0.05

    def test_indirect_recovered(self, df_large):
        TRUE_IE = self.TRUE_A * self.TRUE_B
        med = fit_mediation(df_large, mediators=["m1"], fe_cols=[], fe_method=None)
        eff = med.mediation_results.effects
        assert abs(eff.indirect[0, 0] - TRUE_IE) < 0.1

    def test_total_recovered(self, df_large):
        TRUE_TE = self.TRUE_CP + self.TRUE_A * self.TRUE_B
        med = fit_mediation(df_large, mediators=["m1"], fe_cols=[], fe_method=None)
        eff = med.mediation_results.effects
        assert abs(eff.total[0] - TRUE_TE) < 0.1


# ============================================================================
# 12. Output format tests
# ============================================================================

class TestOutputFormats:
    def test_to_tidy_df_returns_dataframe(self, df_no_fe):
        med = fit_mediation(df_no_fe, mediators=["m1"], fe_cols=[], fe_method=None)
        df = med.to_tidy_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_tidy_df_has_required_columns(self, df_no_fe):
        med = fit_mediation(df_no_fe, mediators=["m1"], fe_cols=[], fe_method=None)
        df = med.to_tidy_df()
        for col in ("estimate", "std_error"):
            assert col in df.columns

    def test_effects_tidy_has_effect_types(self, df_no_fe):
        med = fit_mediation(df_no_fe, mediators=["m1"], fe_cols=[], fe_method=None)
        eff_df = med.mediation_results.effects.to_tidy_df()
        effect_types = set(eff_df["effect_type"].tolist())
        assert "direct" in effect_types
        assert "indirect" in effect_types
        assert "total_indirect" in effect_types
        assert "total" in effect_types

    def test_summary_df_has_equation_column(self, df_no_fe):
        med = fit_mediation(df_no_fe, mediators=["m1"], fe_cols=[], fe_method=None)
        df = med.summary_df()
        assert "equation" in df.columns
        equations = set(df["equation"].tolist())
        assert "mediator:m1" in equations
        assert "outcome" in equations

    def test_summary_dict_keys(self, df_no_fe):
        med = fit_mediation(df_no_fe, mediators=["m1"], fe_cols=[], fe_method=None)
        s = med.summary()
        for key in ("model_spec", "sample_info", "equations", "effects"):
            assert key in s

    def test_print_summary_runs(self, df_no_fe, capsys):
        med = fit_mediation(df_no_fe, mediators=["m1"], fe_cols=[], fe_method=None)
        med.print_summary()
        out = capsys.readouterr().out
        assert "MEDIATION" in out


# ============================================================================
# 13. Standard error sanity checks
# ============================================================================

class TestSETypes:
    def test_hc1_se_positive(self, df_no_fe):
        med = fit_mediation(df_no_fe, mediators=["m1"], fe_cols=[],
                            fe_method=None, se_method="HC1")
        se = med.mediation_results.outcome_result.std_errors
        assert se is not None and np.all(se > 0)

    def test_iid_se_positive(self, df_no_fe):
        med = fit_mediation(df_no_fe, mediators=["m1"], fe_cols=[],
                            fe_method=None, se_method="iid")
        se = med.mediation_results.outcome_result.std_errors
        assert se is not None and np.all(se > 0)

    def test_se_type_stored(self, df_no_fe):
        med = fit_mediation(df_no_fe, mediators=["m1"], fe_cols=[],
                            fe_method=None, se_method="HC1")
        assert med.mediation_results.se_type is not None

    def test_indirect_se_delta_method_structure(self, df_no_fe):
        """IE_se^2 ≈ B^2 * Var(A) + A^2 * Var(B) for one mediator."""
        med = fit_mediation(df_no_fe, mediators=["m1"], fe_cols=[],
                            fe_method=None, se_method="HC1")
        eff  = med.mediation_results.effects

        m1_res  = med.mediation_results.mediator_results["m1"]
        out_res = med.mediation_results.outcome_result

        m1_coefs = m1_res.coefficients.flatten()
        m1_vcov  = m1_res.vcov
        m1_names = m1_res.coef_names
        out_coefs = out_res.coefficients.flatten()
        out_vcov  = out_res.vcov
        out_names = out_res.coef_names

        A     = m1_coefs[m1_names.index("x")]
        B     = out_coefs[out_names.index("m1")]
        x_in_m1   = m1_names.index("x")
        m1_in_out = out_names.index("m1")

        var_A = m1_vcov[x_in_m1, x_in_m1]
        var_B = out_vcov[m1_in_out, m1_in_out]

        expected_var = B ** 2 * var_A + A ** 2 * var_B
        computed_var = eff.indirect_se[0, 0] ** 2
        np.testing.assert_allclose(computed_var, expected_var, rtol=1e-10)
