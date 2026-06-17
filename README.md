# duckreg

`duckreg` is a Python package for linear regression on large datasets using DuckDB-backed preprocessing and sufficient-statistics compression.

The project is based on [`py-econometrics/duckreg`](https://github.com/py-econometrics/duckreg) and extends it with a higher-level formula API, out-of-core fitting options, fixed-effects support via iterative demeaning, instrumental-variables estimation, and mediation tooling.

## What the package does

At a high level, `duckreg`:

- reads data from files or in-memory tabular objects,
- builds regression design matrices inside DuckDB,
- compresses repeated strata into sufficient statistics,
- fits weighted least squares on the compressed representation,
- computes analytical or bootstrap standard errors, depending on the specification.

This design is useful when the original dataset is larger than comfortable in-memory workflows, or when repeated covariate patterns allow substantial compression before estimation.

## Current scope

The repository currently exposes:

- pooled OLS,
- fixed-effects regression via iterative demeaning,
- 2SLS / instrumental-variables regression,
- mediation models,
- DuckDB-backed and NumPy-backed fitting paths,
- analytical and bootstrap inference utilities,
- notebooks, tests, generated API docs, and benchmark scripts.

Some code paths are present but intentionally disabled in the current branch:

- `fe_method="mundlak"` raises `NotImplementedError`,
- `fe_method="auto_fe"` is marked experimental and disabled.

## Installation

Install from the repository root:

```bash
pip install .
```

For development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install pytest
```

The project metadata currently targets Python 3.9 through 3.12 in CI.

## Dependencies

Runtime dependencies are kept light:

- `duckdb`
- `numpy`
- `pandas`
- `numba`
- `tqdm`

Documentation generation uses `pdoc`.

## Quick start

The main entry point is the high-level `duckreg()` function:

```python
from duckreg import duckreg

model = duckreg(
    "y ~ x1 + x2",
    data="data/example.parquet",
    se_method="HC1",
)

summary = model.summary()
tidy = model.to_tidy_df()
```

`data` can be:

- a path to `.parquet`, `.csv`, `.tsv`, `.json`, `.ndjson`, `.jsonl`, `.arrow`, or `.feather`,
- a directory of parquet files,
- a pandas `DataFrame`,
- a Polars `DataFrame`,
- a PyArrow table or record batch,
- a DuckDB relation.

File-backed inputs are resolved into a DuckDB database path and scan expression automatically. In-memory objects are registered as temporary DuckDB views.

## Formula syntax

The package uses a pipe-based formula syntax inspired by `lfe` / `fixest`.

Basic OLS:

```python
duckreg("y ~ x1 + x2", data=df)
```

Fixed effects:

```python
duckreg("y ~ x1 + x2 | unit + year", data=df, fe_method="demean")
```

IV / 2SLS:

```python
duckreg("y ~ x1 | unit + year | (endog ~ z1 + z2)", data=df)
```

Clustered standard errors using a fourth pipe segment:

```python
duckreg("y ~ x1 + x2 | unit + year | | firm_id", data=df, se_method={"CRV1": "firm_id"})
```

Mediation using `via(...)` in the covariate slot:

```python
duckreg("y ~ x + c + via(m1, m2) | unit", data=df, fe_method="demean")
```

Supported formula features in the parser include:

- multiple covariates separated by `+`,
- fixed effects in the second pipe segment,
- IV specifications in the third pipe segment with `(endog ~ instruments)`,
- clustering,
- simple transforms such as `log(...)`, `log1p(...)`, `exp(...)`, and `sqrt(...)`,
- powers such as `x^2`,
- interaction handling,
- boolean expressions inside parentheses where supported by the parser.

## Standard errors

The main standard error options are:

- `"iid"` for homoskedastic standard errors,
- `"HC1"` for heteroskedasticity-robust standard errors,
- `{"CRV1": "cluster_var"}` for cluster-robust standard errors,
- `"BS"` for bootstrap-based inference,
- `"none"` to skip variance estimation.

For bootstrap inference, pass settings through `bootstrap`, for example:

```python
model = duckreg(
    "y ~ x1 | unit",
    data=df,
    se_method="BS",
    bootstrap={"n": 200, "seed": 0},
    threads=4,
)
```

## Main options

The `duckreg()` API includes the following commonly used parameters:

- `formula`: model specification string.
- `data`: file path, directory, or supported in-memory table object.
- `se_method`: inference specification.
- `bootstrap`: bootstrap settings when `se_method="BS"`.
- `fe_method`: currently `None`, `"auto"`, or `"demean"` in practical use. `mundlak` is currently disabled.
- `remove_singletons`: drop singleton FE groups before estimation.
- `subset`: SQL `WHERE` filter applied before estimation.
- `cache_dir`: location for DuckDB cache files for file-backed inputs.
- `db_name`: explicit DuckDB database path.
- `fitter`: `"numpy"` or `"duckdb"`.
- `compression`: controls row compression before fitting.
- `seed`: random seed.
- `threads`: DuckDB thread count and bootstrap parallelism control.
- `memory_limit`: DuckDB memory limit such as `"8GB"`.
- `max_temp_directory_size`: DuckDB temporary storage limit.

For fixed-effects estimation, the API also exposes demeaning controls such as:

- `max_iterations`,
- `tolerance`,
- `check_interval`,
- `convergence_sample`,
- `singleton_pruning`,
- `fe_order`,
- `drop_constant_variables`,
- `residual_type`.

## Compression behavior

Compression is one of the core ideas in the project.

- `compression=None` keeps exact compression and still groups rows when the strata match exactly.
- `compression=5` rounds continuous strata-defining values to 5 decimals before grouping.
- `compression=-1` disables grouping entirely and leaves the compressed table at one row per observation.

Rounding can materially reduce the number of unique strata in continuous-data settings and improve memory use. The tradeoff is approximation in the compression stage, so it should be chosen deliberately and validated for the application.

## Backends

Two fitting paths are exposed:

- `fitter="numpy"` performs the final weighted least squares step in memory.
- `fitter="duckdb"` keeps more of the workflow inside DuckDB and is the more relevant option for out-of-core use cases.

The project also uses DuckDB throughout the preprocessing pipeline even when the final fit is done with NumPy.

## Returned objects

`duckreg()` returns an estimator object rather than a plain coefficient table.

Depending on the model type, useful methods and attributes include:

- `summary()`: structured model summary as a dictionary,
- `summary_df()`: summary table as a pandas DataFrame,
- `to_tidy_df()`: tidy coefficient output,
- `results`: underlying result container,
- `first_stage`: first-stage IV results for 2SLS models.

Result containers track metadata such as:

- coefficient names and estimates,
- variance-covariance matrices,
- number of observations,
- number of compressed rows,
- compression ratio inputs,
- package version and computation timestamp.

## Typical examples

Pooled OLS:

```python
model = duckreg("y ~ x1 + x2", data=df, se_method="iid")
```

Two-way fixed effects with iterative demeaning:

```python
model = duckreg(
    "y ~ treatment + control | unit + year",
    data="panel.parquet",
    fe_method="demean",
    se_method="HC1",
    fitter="duckdb",
)
```

2SLS with one endogenous regressor:

```python
model = duckreg(
    "y ~ x1 | firm_id + year | (price ~ cost_shifter)",
    data=df,
    se_method="HC1",
)
```

Cluster-robust specification:

```python
model = duckreg(
    "y ~ x1 + x2 | firm_id + year",
    data=df,
    se_method={"CRV1": "firm_id"},
)
```

Subset a larger file-backed dataset:

```python
model = duckreg(
    "y ~ x1 + x2 | region",
    data="data/full_sample.parquet",
    subset="year >= 2015 AND treated IS NOT NULL",
)
```

## Repository layout

The top-level structure is:

- [`duckreg`](/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Job/UNI/Basel/Research/duckreg/duckreg): package source.
- [`tests`](/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Job/UNI/Basel/Research/duckreg/tests): unit and integration tests.
- [`notebooks`](/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Job/UNI/Basel/Research/duckreg/notebooks): exploratory and usage notebooks.
- [`benchmarks`](/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Job/UNI/Basel/Research/duckreg/benchmarks): benchmarking notebook and orchestration scripts.
- [`docs`](/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Job/UNI/Basel/Research/duckreg/docs): generated HTML documentation.

Within the package:

- `duckreg/duckreg.py` contains the main high-level API.
- `duckreg/utils/formula_parser.py` implements formula parsing and SQL-safe expression handling.
- `duckreg/estimators/` contains estimator implementations such as pooled OLS, FE, 2SLS, ridge, and mediation.
- `duckreg/core/` contains result containers, sufficient-statistics logic, variance estimation, linear algebra helpers, and SQL builders.

## Development

Run the test suite from the repository root:

```bash
pytest
```

Generate API documentation:

```bash
pdoc duckreg -o docs/ --math
```

The GitHub Actions configuration currently:

- runs tests on Python 3.9, 3.10, 3.11, and 3.12,
- builds and deploys `pdoc` documentation from `master`.

## Benchmarks and notebooks

The repository includes:

- an introduction notebook,
- an event-study notebook,
- a regularization notebook,
- a benchmark notebook and SLURM-oriented benchmark scripts.

See [`benchmarks/README.md`](/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Job/UNI/Basel/Research/duckreg/benchmarks/README.md) for the benchmark workflow.

## Limitations and practical notes

- The current codebase is focused on linear-model workflows.
- Mundlak fixed-effects absorption is present in code history but disabled in this branch.
- `auto_fe` is also disabled.
- Some advanced features, especially compression tuning and out-of-core operation, are best validated against the specific data-generating setting rather than treated as drop-in defaults.
- Cluster and bootstrap inference can be materially more expensive than plain HC1 inference.

## References

The methodological motivation is tied to sufficient-statistics compression for large-scale regression workflows. The repository README and code reference:

- Lal, Fischer, and Wardrop, "Large Scale Longitudinal Experiments: Estimation and Inference", arXiv:2410.09952.
- the upstream `py-econometrics/duckreg` project.

## License

This repository is distributed under the MIT License. See [`LICENSE`](/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Job/UNI/Basel/Research/duckreg/LICENSE).
