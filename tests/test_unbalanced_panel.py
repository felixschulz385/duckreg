"""
Tests for unbalanced panel support with FE classification.

This suite focuses on FE classification heuristics and dummy-mean generation,
not coefficient parity.
"""

import numpy as np
import pandas as pd
import pytest

from duckreg import compressed_ols
from tests.helpers import make_fe_classification_panel


def _fit_model(parquet_path: str, formula: str, **kwargs):
    return compressed_ols(
        formula=formula,
        data=parquet_path,
        fe_method="mundlak",
        fitter="numpy",
        se_method="none",
        **kwargs,
    )


@pytest.fixture
def balanced_panel_data():
    return make_fe_classification_panel(n_firms=100, n_years=5)


@pytest.fixture
def unbalanced_panel_data():
    return make_fe_classification_panel(n_firms=100, n_years=5, drop_frac=0.3)


@pytest.fixture
def balanced_panel_path(balanced_panel_data, parquet_path_factory):
    return parquet_path_factory(balanced_panel_data, "balanced_panel.parquet")


@pytest.fixture
def unbalanced_panel_path(unbalanced_panel_data, parquet_path_factory):
    return parquet_path_factory(unbalanced_panel_data, "unbalanced_panel.parquet")


@pytest.fixture
def small_cardinality_data():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "entity_id": np.repeat(np.arange(100), 5),
            "time_id": np.tile(np.arange(5), 100),
            "x": rng.standard_normal(500),
            "y": rng.standard_normal(500),
        }
    )


@pytest.fixture
def large_cardinality_data():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "entity_id": np.repeat(np.arange(200), 5),
            "time_id": np.tile(np.arange(5), 200),
            "x": rng.standard_normal(1000),
            "y": rng.standard_normal(1000),
        }
    )


class TestFEClassificationAndFeatureGeneration:
    @pytest.mark.parametrize("fitter", ["numpy", "duckdb"])
    def test_balanced_panel_classification(self, balanced_panel_path, fitter):
        model = compressed_ols(
            formula="y ~ x1 + x2 | firm_id + year",
            data=balanced_panel_path,
            fe_method="mundlak",
            fitter=fitter,
            se_method="none",
            compression=5,
        )

        assert "firm_id" in model.fe_metadata
        assert "year" in model.fe_metadata
        assert model.fe_metadata["year"]["type"] == "fixed"
        assert model.fe_metadata["year"]["profile"]["cardinality"] == 5
        assert model.fe_metadata["firm_id"]["type"] == "asymptotic"
        assert model.fe_metadata["firm_id"]["profile"]["cardinality"] == 100

        mundlak_means = [name for name in model.coef_names_ if "avg_x" in name and "_fe0" in name]
        fixed_dummies = [name for name in model.coef_names_ if name.startswith("dummy_year_")]
        dummy_means = [name for name in model.coef_names_ if "avg_year_" in name]

        assert mundlak_means
        assert len(fixed_dummies) == 4
        assert len(dummy_means) == 4
        assert model.point_estimate is not None
        assert len(model.coef_names_) == 13

    @pytest.mark.parametrize("fitter", ["numpy", "duckdb"])
    def test_unbalanced_panel_adds_dummy_means(self, unbalanced_panel_path, fitter):
        model = compressed_ols(
            formula="y ~ x1 + x2 | firm_id + year",
            data=unbalanced_panel_path,
            fe_method="mundlak",
            fitter=fitter,
            se_method="none",
            compression=5,
        )

        assert model.fe_metadata["year"]["type"] == "fixed"
        assert model.fe_metadata["firm_id"]["type"] == "asymptotic"

        dummy_mean_cols = [name for name in model.coef_names_ if "avg_year_" in name]
        expected_dummy_cols = len(model.fe_metadata["year"]["levels"]) - 1

        assert dummy_mean_cols
        assert len(dummy_mean_cols) == expected_dummy_cols
        assert model.point_estimate is not None


class TestFEClassificationHeuristic:
    @pytest.mark.parametrize(
        ("fixture_name", "expected_col", "expected_type", "expected_cardinality"),
        [
            ("small_cardinality_data", "time_id", "fixed", 5),
            ("large_cardinality_data", "entity_id", "asymptotic", 200),
        ],
    )
    def test_cardinality_based_classification(
        self,
        request,
        parquet_path_factory,
        fixture_name,
        expected_col,
        expected_type,
        expected_cardinality,
    ):
        data = request.getfixturevalue(fixture_name)
        path = parquet_path_factory(data, f"{fixture_name}.parquet")
        model = _fit_model(path, "y ~ x | entity_id + time_id")

        assert model.fe_metadata[expected_col]["type"] == expected_type
        assert model.fe_metadata[expected_col]["profile"]["cardinality"] == expected_cardinality

    def test_user_override_classification(self, small_cardinality_data, parquet_path_factory):
        path = parquet_path_factory(small_cardinality_data, "override_classification.parquet")
        model = _fit_model(
            path,
            "y ~ x | entity_id + time_id",
            fe_types={"time_id": "asymptotic"},
        )

        assert model.fe_metadata["time_id"]["type"] == "asymptotic"


class TestEdgeCases:
    def test_all_fes_asymptotic(self, parquet_path_factory):
        rng = np.random.default_rng(42)
        data = pd.DataFrame(
            {
                "entity1": np.repeat(np.arange(200), 5),
                "entity2": np.tile(np.arange(200), 5),
                "x": rng.standard_normal(1000),
                "y": rng.standard_normal(1000),
            }
        )

        path = parquet_path_factory(data, "all_asymptotic.parquet")
        model = _fit_model(path, "y ~ x | entity1 + entity2")

        assert model.fe_metadata["entity1"]["type"] == "asymptotic"
        assert model.fe_metadata["entity2"]["type"] == "asymptotic"
        dummy_mean_cols = [name for name in model.coef_names_ if name.startswith("avg_entity")]
        assert all("_fe" in name and "=" not in name for name in dummy_mean_cols)

    def test_all_fes_fixed(self, parquet_path_factory):
        rng = np.random.default_rng(42)
        data = pd.DataFrame(
            {
                "region": rng.choice(["A", "B", "C"], size=500),
                "year": rng.choice([2018, 2019, 2020], size=500),
                "x": rng.standard_normal(500),
                "y": rng.standard_normal(500),
            }
        )

        path = parquet_path_factory(data, "all_fixed.parquet")
        model = _fit_model(path, "y ~ x | region + year")

        assert model.fe_metadata["region"]["type"] == "fixed"
        assert model.fe_metadata["year"]["type"] == "fixed"
        assert [name for name in model.coef_names_ if "avg_x_fe" in name] == []
        assert model._dummy_mean_cols == []

    def test_column_explosion_guard(self, parquet_path_factory):
        rng = np.random.default_rng(42)
        data = pd.DataFrame(
            {
                "entity_id": np.repeat(np.arange(100), 5),
                "time_id": np.tile(np.arange(150), 100)[:500],
                "x": rng.standard_normal(500),
                "y": rng.standard_normal(500),
            }
        )

        path = parquet_path_factory(data, "column_explosion_guard.parquet")
        model = _fit_model(
            path,
            "y ~ x | entity_id + time_id",
            max_fixed_fe_levels=100,
        )

        assert model.fe_metadata["time_id"]["type"] == "asymptotic"
