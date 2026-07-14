import numpy as np
import pandas as pd
import pytest

from duckreg import duckreg
from duckreg.core.fitters.duckdb_fitter import ridge_closed_form, ridge_closed_form_batch
from duckreg.core.fitters.numpy_fitter import NumpyFitter
from duckreg.core.vcov import compute_cluster_scores
from duckreg.estimators.DuckLinearModel import DuckLinearModel


@pytest.fixture
def upgrade_data():
    rng = np.random.default_rng(2026)
    n = 1200
    group = np.repeat(np.arange(120), 10)
    period = np.tile(np.arange(10), 120)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    endog = 0.8 * z + 0.2 * x + rng.normal(size=n)
    y = 1.5 * x + 2.0 * endog + group / 20 + period / 10 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "endog": endog,
                         "group": group, "period": period})


def test_none_is_strict_and_iv_diagnostics_are_unset(upgrade_data, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("VCOV work must not run")

    monkeypatch.setattr("duckreg.core.fitters.numpy_fitter.NumpyFitter.fit_vcov", forbidden)
    model = duckreg(
        "y ~ x | | (endog ~ z)", upgrade_data,
        se_method="none", fitter="numpy", compression=-1,
    )
    assert model.vcov is None
    assert model.results.vcov is None
    assert model.get_first_stage_f_stats() == {"endog": None}
    assert model.first_stage["endog"].f_pvalue is None


def test_resources_close_retention_and_thread_one(upgrade_data):
    model = duckreg("y ~ x", upgrade_data, se_method="none", threads=1)
    assert model._closed and model.conn is None
    assert int(model.metadata["threads"]) == 1
    model.close()
    with pytest.raises(RuntimeError, match="retain_compressed=True"):
        _ = model.df_compressed

    retained = duckreg(
        "y ~ x", upgrade_data, se_method="none", retain_compressed=True
    )
    assert retained._closed
    assert retained.df_compressed is not None


def test_automatic_compression_decisions(upgrade_data):
    categorical = duckreg("y ~ period", upgrade_data, se_method="none")
    continuous = duckreg("y ~ x", upgrade_data, se_method="none")
    assert categorical.resolved_compression is None
    assert continuous.resolved_compression == -1


def test_automatic_numpy_memory_fallback(upgrade_data, monkeypatch):
    expected = duckreg(
        "y ~ x", upgrade_data, se_method="none", fitter="duckdb", compression=-1
    ).point_estimate

    def out_of_memory(self):
        raise MemoryError

    monkeypatch.setattr(DuckLinearModel, "_estimate_numpy", out_of_memory)
    actual = duckreg(
        "y ~ x", upgrade_data, se_method="none", fitter="auto", compression=-1
    )
    np.testing.assert_allclose(actual.point_estimate, expected)
    assert actual.resolved_fitter == "duckdb"
    assert actual.metadata["numpy_memory_fallback"] is True


def test_numpy_map_matches_duckdb(upgrade_data):
    kwargs = dict(se_method="HC1", fitter="numpy", compression=-1)
    numpy_model = duckreg(
        "y ~ x | group + period", upgrade_data,
        demean_backend="numpy", **kwargs,
    )
    duckdb_model = duckreg(
        "y ~ x | group + period", upgrade_data,
        demean_backend="duckdb", **kwargs,
    )
    np.testing.assert_allclose(numpy_model.point_estimate, duckdb_model.point_estimate, rtol=1e-8)
    np.testing.assert_allclose(numpy_model.vcov, duckdb_model.vcov, rtol=1e-7)
    assert numpy_model.map_iterations is not None


def test_cluster_reduction_and_ridge_path_are_exact():
    rng = np.random.default_rng(9)
    scores = rng.normal(size=(3000, 5))
    ids = rng.integers(0, 91, size=3000)
    reduced, count = compute_cluster_scores(scores, ids)
    expected = np.vstack([scores[ids == value].sum(axis=0) for value in np.unique(ids)])
    np.testing.assert_allclose(reduced, expected)
    assert count == 91

    X = rng.normal(size=(500, 7))
    y = rng.normal(size=(500, 1))
    weights = rng.integers(1, 4, size=500)
    lambdas = np.array([1e-4, 0.2, 4.0])
    path = ridge_closed_form_batch(X, y, weights, lambdas)
    sequential = np.vstack([
        ridge_closed_form(X, y, weights, value).ravel() for value in lambdas
    ])
    np.testing.assert_allclose(path, sequential, rtol=1e-11, atol=1e-11)


def test_sufficient_statistics_accept_shared_multi_outcomes():
    rng = np.random.default_rng(81)
    X = np.c_[np.ones(400), rng.normal(size=(400, 3))]
    coefficients = rng.normal(size=(4, 3))
    Y = X @ coefficients + rng.normal(scale=0.1, size=(400, 3))
    result = NumpyFitter().fit(X, Y, np.ones(400))
    expected = np.linalg.solve(X.T @ X + 1e-8 * np.eye(4), X.T @ Y)
    np.testing.assert_allclose(result.coefficients, expected)
    assert result.Xty.shape == (4, 3)
