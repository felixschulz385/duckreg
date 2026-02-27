# `duckreg` (fork) : very fast out-of-memory regressions with `duckdb`

> **This is a fork of [py-econometrics/duckreg](https://github.com/py-econometrics/duckreg)** with significant new features:
> a unified high-level API, true out-of-memory fitting in DuckDB, continuous variable rounding for compression,
> DuckDB-native iterative demeaning, 2SLS / instrumental-variables support, and full analytical standard error coverage.

A Python wrapper around `duckdb` that runs regressions on very large datasets that do not fit in memory.
Data are reduced to sufficient statistics (compressed strata), and weighted least squares is run on the
compressed representation. Robust standard errors are computed analytically from sufficient statistics;
clustered SEs are computed via the cluster bootstrap. Methodological details and benchmarks: [arXiv 2410.09952](https://arxiv.org/abs/2410.09952).

---

## What's new in this fork

| Feature | Upstream | This fork |
|---|---|---|
| High-level `duckreg()` API | ❌ | ✅ lfe-style formula interface |
| OOM fitting | partial (loads to memory after conversion) | ✅ stays in DuckDB throughout |
| Continuous variable rounding | ❌ | ✅ `round_strata` parameter |
| Iterative demeaning | ❌ | ✅ MAP algorithm in DuckDB (`DuckIterativeFE`) |
| 2SLS / IV regression | ❌ | ✅ `Duck2SLS` estimator |
| Analytical SE | partial (HC1 + bootstrap cluster) | ✅ iid, HC1, CRV1, BS, none |

---

## Install

```bash
pip install duckreg
```

Dev install (preferably in a `venv`):

```bash
(uv) pip install git+https://github.com/felixschulz385/duckreg
```

Or clone and install in editable mode.

---

## Quickstart

All models are accessible through the single `duckreg()` function using an **lfe-style formula**:

```
"y ~ x1 + x2 | fe1 + fe2 | endog (inst1 + inst2) | cluster_var"
  ^outcome  ^covariates  ^fixed effects  ^IV spec   ^cluster
```

```python
from duckreg import duckreg

# OLS, no fixed effects
result = duckreg("y ~ x1 + x2", data="mydata.parquet")

# Two-way FE via iterative demeaning (default for FE models)
result = duckreg("y ~ x1 + x2 | unit + time", data="mydata.parquet")

# Two-way FE via Mundlak device (not recommended for unbalanced panels)
result = duckreg("y ~ x1 + x2 | unit + time", data="mydata.parquet", fe_method="mundlak")

# 2SLS with an instrument
result = duckreg("y ~ x1 | fe1 | endog (instrument)", data="mydata.parquet")

# Cluster-robust SEs
result = duckreg("y ~ x1 + x2 | unit | | cluster_var", data="mydata.parquet", se_method="CRV1")

print(result.summary())
```

`data` accepts: `.parquet`, `.csv`, `.tsv`, `.json`, `.feather/.arrow` files, a directory of
`.parquet` files, or an in-memory pandas / Polars / PyArrow DataFrame or DuckDB relation.

---

## Standard errors

| `se_method` | Description |
|---|---|
| `"iid"` | Homoscedastic (classical) |
| `"HC1"` | Heteroscedasticity-robust (default) |
| `"CRV1"` | Cluster-robust (analytical) |
| `"BS"` | Cluster bootstrap |
| `"none"` | Skip SE computation |

The cluster variable is specified either in the formula's 4th slot (`| cluster`) or passed separately alongside `se_method="CRV1"`.

---

## Key parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `formula` | `str` | required | lfe-style formula (see above) |
| `data` | various | required | File path, DataFrame, or DuckDB relation |
| `fe_method` | `str` | `"auto"` | `"mundlak"` (not recommended for unbalanced panels), `"demean"`, or `"auto"` |
| `se_method` | `str` | `"HC1"` | Standard error method |
| `round_strata` | `int` | `None` | Round continuous columns to N decimals before stratification (improves compression) |
| `fitter` | `str` | `"numpy"` | `"numpy"` (in-memory WLS) or `"duckdb"` (fully out-of-core) |
| `subset` | `str` | `None` | SQL `WHERE` clause to filter rows |
| `n_bootstraps` | `int` | `100` | Bootstrap iterations (only for `se_method="BS"`) |
| `remove_singletons` | `bool` | `True` | Drop singleton FE groups before fitting |
| `seed` | `int` | `42` | Random seed |
| `n_jobs` | `int` | `1` | Parallel jobs for bootstrap |
| `cache_dir` | `str` | `None` | Directory for DuckDB cache files |
| `duckdb_kwargs` | `dict` | `None` | DuckDB configuration overrides |

---

## How it works

The core insight (from [Lal, Fischer & Wardrop 2024](https://arxiv.org/abs/2410.09952)) is that any OLS regression can be
rewritten as a *small* WLS regression on compressed sufficient statistics. This fork extends that to:

- **True OOM fitting**: compressed sufficient statistics are computed and (optionally) fitted entirely inside DuckDB,
  so the raw data never has to be loaded into Python memory. Set `fitter="duckdb"` to enable.
- **`round_strata`**: continuous covariates create an explosion of unique strata. Rounding to `N` decimals before
  compression dramatically reduces the number of strata and speeds up both aggregation and WLS.
- **Iterative demeaning in DuckDB**: the MAP demeaning loop runs as a sequence of DuckDB SQL queries, keeping
  intermediate demean tables out of Python memory even for very large panels.
- **2SLS**: the first stage is run as a compressed DuckDB aggregation, fitted values are substituted, and
  the second stage is run as standard compressed WLS with correct analytical SEs propagated.

---

## Citation

```bibtex
@misc{lal2024largescalelongitudinalexperiments,
      title={Large Scale Longitudinal Experiments: Estimation and Inference},
      author={Apoorva Lal and Alexander Fischer and Matthew Wardrop},
      year={2024},
      eprint={2410.09952},
      archivePrefix={arXiv},
      primaryClass={econ.EM},
      url={https://arxiv.org/abs/2410.09952},
}
```

---

## References

**Methods:**
- [Arkhangelsky and Imbens (2023)](https://arxiv.org/abs/1807.02099)
- [Wooldridge (2021) – Two-Way FE, Mundlak, and DiD](https://doi.org/10.1007/s00181-025-02807-z)
- [Wong et al. (2021)](https://arxiv.org/abs/2102.11297)

**Libraries / tutorials:**
- [py-econometrics/duckreg (upstream)](https://github.com/py-econometrics/duckreg)
