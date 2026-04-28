from __future__ import annotations

import numpy as np
import pandas as pd


def make_fe_regression_panel(
    *,
    n_pixels: int,
    n_years: int,
    n_countries: int = 20,
    n_soil: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    panel = pd.MultiIndex.from_product(
        [np.arange(n_pixels), np.arange(2010, 2010 + n_years)],
        names=["pixel_id", "year"],
    ).to_frame(index=False)

    panel["country"] = (panel["pixel_id"] % n_countries).astype(int)
    soil_by_pixel = rng.integers(0, n_soil, size=n_pixels)
    panel["soil_type"] = soil_by_pixel[panel["pixel_id"]]

    pixel_fe = rng.standard_normal(n_pixels) * 2.0
    year_fe = rng.standard_normal(n_years) * 1.5
    country_fe = rng.standard_normal(n_countries)
    soil_fe = rng.standard_normal(n_soil) * 0.8

    n_obs = len(panel)
    panel["rainfall"] = rng.standard_normal(n_obs) * 10 + 100
    panel["ntl_harm"] = (
        0.5 * panel["rainfall"]
        + pixel_fe[panel["pixel_id"]]
        + year_fe[panel["year"] - 2010]
        + rng.standard_normal(n_obs) * 5
    )

    ov = rng.standard_normal(n_obs) * 2
    panel["modis_median"] = (
        0.8 * panel["ntl_harm"]
        + pixel_fe[panel["pixel_id"]] * 1.2
        + year_fe[panel["year"] - 2010] * 0.8
        + country_fe[panel["country"]] * 0.6
        + soil_fe[panel["soil_type"]] * 0.5
        + 0.5 * ov
        + 0.3 * panel["ntl_harm"] * ov / 10
        + rng.standard_normal(n_obs)
    )
    panel["exog_control"] = rng.standard_normal(n_obs) * 3
    return panel.reset_index(drop=True)


def make_unbalanced_panel(
    balanced: pd.DataFrame,
    *,
    fe_dims: list[str],
    drop_frac: float = 0.25,
    seed: int = 99,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    panel = balanced.loc[rng.random(len(balanced)) >= drop_frac].copy()
    panel = panel.reset_index(drop=True)

    changed = True
    while changed:
        changed = False
        for dim in fe_dims:
            counts = panel.groupby(dim)[dim].transform("count")
            before = len(panel)
            panel = panel.loc[counts > 1].reset_index(drop=True)
            if len(panel) < before:
                changed = True
    return panel


def make_iv_panel_data(
    *,
    n_firms: int = 80,
    n_years: int = 8,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    panel = pd.MultiIndex.from_product(
        [np.arange(n_firms), np.arange(2010, 2010 + n_years)],
        names=["firm_id", "year"],
    ).to_frame(index=False)

    firm_fe = rng.standard_normal(n_firms)
    n_obs = len(panel)
    panel["x"] = rng.standard_normal(n_obs)
    panel["z"] = rng.standard_normal(n_obs)
    panel["endog"] = 0.8 * panel["z"] + rng.standard_normal(n_obs) * 0.5
    panel["y"] = (
        2.0 * panel["endog"]
        + panel["x"]
        + firm_fe[panel["firm_id"]]
        + rng.standard_normal(n_obs) * 0.5
    )
    return panel


def make_fe_classification_panel(
    *,
    n_firms: int = 100,
    n_years: int = 5,
    seed: int = 42,
    drop_frac: float | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    panel = pd.DataFrame(
        [
            {"firm_id": firm, "year": year}
            for firm in range(n_firms)
            for year in range(2015, 2015 + n_years)
        ]
    )

    if drop_frac is not None:
        keep_mask = rng.random(len(panel)) >= drop_frac
        panel = panel.loc[keep_mask].reset_index(drop=True)

    panel["x1"] = rng.standard_normal(len(panel))
    panel["x2"] = rng.standard_normal(len(panel))
    firm_fe = rng.standard_normal(n_firms)
    year_fe = rng.standard_normal(n_years)
    panel["y"] = (
        2.0
        + 1.5 * panel["x1"]
        + 0.8 * panel["x2"]
        + firm_fe[panel["firm_id"]]
        + year_fe[panel["year"] - 2015]
        + rng.standard_normal(len(panel)) * 0.5
    )
    return panel


def pyfixest_vcov(vcov: str):
    return {"CRV1": "country"} if vcov == "CRV1" else vcov


def duckreg_se_method(vcov: str):
    return {"CRV1": "country"} if vcov == "CRV1" else vcov


def pyfixest_coef_se(fit, var: str, has_iv: bool):
    if has_iv:
        tidy = fit.tidy()
        return float(tidy.loc[var, "Estimate"]), float(tidy.loc[var, "Std. Error"])
    return float(fit.coef().loc[var]), float(fit.se().loc[var])


def duckreg_coef_se(model, var: str = "ntl_harm"):
    res = model.summary_df()
    return float(res.loc[var, "coefficient"]), float(res.loc[var, "std_error"])


def assert_coef_se_close(
    duckreg_coef: float,
    pyfixest_coef: float,
    duckreg_se: float,
    pyfixest_se: float,
    *,
    coef_rtol: float,
    se_rtol: float,
    label: str,
) -> None:
    np.testing.assert_allclose(
        duckreg_coef,
        pyfixest_coef,
        rtol=coef_rtol,
        atol=1e-6,
        err_msg=f"coef mismatch: {label}",
    )
    np.testing.assert_allclose(
        duckreg_se,
        pyfixest_se,
        rtol=se_rtol,
        atol=1e-6,
        err_msg=f"SE mismatch: {label}",
    )
