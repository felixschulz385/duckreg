"""High-level API for compressed OLS regression.

This module provides the main entry point for users to run compressed OLS regressions
with support for fixed effects, instrumental variables, and various standard error methods.
It handles formula parsing, data source resolution, and estimator selection.
"""
import logging
import warnings
from typing import Any, Dict, Optional, Union

from .core.vcov import parse_vcov_specification, parse_cluster_vars, VcovTypeNotSupportedError, VcovSpec
from .estimators.base import DuckEstimator, SEMethod
from .utils.api import FEMethod, _resolve_data_source, _DUCKDB_VIEW_NAME

logger = logging.getLogger(__name__)

# Backward compatibility - re-export from base
DuckReg = DuckEstimator

# DuckDB settings that can be passed as top-level kwargs
_DUCKDB_RESOURCE_KWARGS = frozenset({"threads", "memory_limit", "max_temp_directory_size"})


# ============================================================================
# High-level API
# ============================================================================

def duckreg(
    formula: str,
    data: Any,
    # ── SE settings ──────────────────────────────────────────────────────────
    se_method: Union[str, Dict] = SEMethod.HC1,
    bootstrap: Optional[Dict] = None,
    # ── Fixed-effects settings ───────────────────────────────────────────────
    fe_method: str = FEMethod.AUTO,
    remove_singletons: bool = True,
    # ── Data filtering ────────────────────────────────────────────────────────
    subset: str = None,
    # ── Storage / cache settings ─────────────────────────────────────────────
    cache_dir: str = None,
    db_name: str = None,
    # ── Engine settings ──────────────────────────────────────────────────────
    fitter: str = "numpy",
    round_strata: int = None,
    seed: int = 42,
    # ── FE classification settings ────────────────────────────────────────────
    fe_types: Optional[Dict] = None,
    max_fixed_fe_levels: Optional[int] = None,
    # ── DuckDB resource kwargs: threads, memory_limit, max_temp_directory_size
    **kwargs,
) -> "DuckEstimator":
    """High-level API for DuckReg regression with lfe/fixest-style formula.

    Orchestrates the entire regression workflow:
    1. Parse the formula to extract outcomes, covariates, FE, IV, and clusters
    2. Resolve data source and database paths
    3. Select appropriate estimator based on model type (OLS, IV, or Mundlak)
    4. Fit the model and compute standard errors

    Args:
        formula: Regression formula.  Supports the lfe/fixest pipe syntax::

            "y ~ x1 + x2 | fe1 + fe2 | (endog ~ inst1 + inst2)"

            Fixed effects are the 2nd pipe segment.
            Instrumental variables use fixest-style ``(endog ~ instruments)``
            in the 3rd pipe segment.

        data: Data source.  Accepts a file path (str/Path) to a .parquet, .csv,
            .tsv, .json, .ndjson, .feather/.arrow file, a directory of .parquet
            files, an in-memory pandas/Polars/PyArrow DataFrame, or a DuckDB
            relation object.
        se_method: Standard error method.  Either a string (``'iid'``, ``'HC1'``,
            ``'BS'``, ``'none'``) or a dict specifying cluster-robust SEs::

                se_method = {"CRV1": "state"}          # single cluster
                se_method = {"CRV1": "state + firm"}   # two-way clustering

        bootstrap: Bootstrap settings, only used when ``se_method='BS'``.
            Accepted keys:

            * ``"n"`` – number of bootstrap replications (default 100)
            * ``"seed"`` – random seed for bootstrap draws (defaults to *seed*)

        fe_method: Method for handling fixed effects (``'demean'`` or
            ``'mundlak'``).  ``'demean'`` (iterative alternating projections) is
            the default for OLS; ``'mundlak'`` is the default for IV.  Mundlak
            is not recommended for unbalanced panels.
        remove_singletons: Remove observations from singleton FE groups
            (default ``True``).
        subset: SQL ``WHERE`` clause to filter data before estimation.
        cache_dir: Directory for DuckDB cache files.  Also accepts
            ``memory_limit`` and ``max_temp_directory_size`` as top-level
            keyword arguments.
        db_name: Full path to a persistent DuckDB database file.
        fitter: Estimation backend – ``'numpy'`` (default, in-memory) or
            ``'duckdb'`` (out-of-core).
        round_strata: Round demeaned covariate columns to this many decimal
            places before grouping (improves compression; slight approximation).
        seed: Global random seed for reproducibility.
        **kwargs: DuckDB resource settings passed directly as keyword arguments:

            * ``threads`` (int) – number of DuckDB threads; also controls
              bootstrap parallelism (replaces ``n_jobs``).
            * ``memory_limit`` (str) – e.g. ``"8GB"``.
            * ``max_temp_directory_size`` (str) – e.g. ``"20GB"``.

    Returns:
        Fitted estimator object with results.

    Examples:
        Basic OLS with two-way FE and heteroskedasticity-robust SEs::

            duckreg("y ~ x1 + x2 | unit + year", data=df)

        Cluster-robust SEs via *se_method* dict (no cluster in formula)::

            duckreg("y ~ x1 + x2 | unit + year", data=df,
                    se_method={"CRV1": "unit"})

        IV with fixest-style syntax::

            duckreg("y ~ x1 | unit + year | (endog ~ z1 + z2)", data=df)

        Bootstrap SEs with 200 replications and 4 threads::

            duckreg("y ~ x1 | unit", data=df,
                    se_method="BS", bootstrap={"n": 200, "seed": 0},
                    threads=4)
    """
    logger.debug("=== duckreg START ===")

    # ------------------------------------------------------------------
    # 1. Extract and validate DuckDB / resource kwargs
    # ------------------------------------------------------------------
    threads = int(kwargs.pop("threads", 1))
    memory_limit = kwargs.pop("memory_limit", None)
    max_temp_dir_size = kwargs.pop("max_temp_directory_size", None)

    # Backward-compat shims – warn but still work
    if "n_jobs" in kwargs:
        warnings.warn(
            "'n_jobs' is deprecated; use the 'threads' keyword argument instead.",
            DeprecationWarning, stacklevel=2,
        )
        threads = int(kwargs.pop("n_jobs"))
    if "duckdb_kwargs" in kwargs:
        warnings.warn(
            "'duckdb_kwargs' is deprecated; pass DuckDB settings as top-level "
            "keyword arguments (e.g. threads=4, memory_limit='8GB').",
            DeprecationWarning, stacklevel=2,
        )
        for k, v in kwargs.pop("duckdb_kwargs").items():
            kwargs.setdefault(k, v)
        # Re-extract resource kwargs that the shim may have put back into kwargs
        threads = int(kwargs.pop("threads", threads))
        memory_limit = kwargs.pop("memory_limit", memory_limit)
        max_temp_dir_size = kwargs.pop("max_temp_directory_size", max_temp_dir_size)
    if "n_bootstraps" in kwargs:
        warnings.warn(
            "'n_bootstraps' is deprecated; use bootstrap={'n': N} instead.",
            DeprecationWarning, stacklevel=2,
        )
        _nb = kwargs.pop("n_bootstraps")
        if bootstrap is None:
            bootstrap = {"n": _nb}

    if kwargs:
        raise TypeError(
            f"duckreg() got unexpected keyword argument(s): {sorted(kwargs)}"
        )

    # Build DuckDB config dict from extracted resource kwargs
    duckdb_kwargs: Dict[str, Any] = {}
    if threads != 1:
        duckdb_kwargs["threads"] = threads
    if memory_limit is not None:
        duckdb_kwargs["memory_limit"] = memory_limit
    if max_temp_dir_size is not None:
        duckdb_kwargs["max_temp_directory_size"] = max_temp_dir_size

    # ------------------------------------------------------------------
    # 2. Bootstrap settings
    # ------------------------------------------------------------------
    is_bs = isinstance(se_method, str) and se_method == SEMethod.BS
    if is_bs:
        _bs = bootstrap or {}
        n_bootstraps = int(_bs.get("n", 100))
        seed = int(_bs.get("seed", seed))  # bootstrap seed overrides global seed
        # Bootstrap benefits from multiple threads – propagate to DuckDB
        if threads > 1:
            duckdb_kwargs.setdefault("threads", threads)
    else:
        n_bootstraps = 0

    # ------------------------------------------------------------------
    # 3. Parse formula
    # ------------------------------------------------------------------
    from .estimators import DuckRegression, Duck2SLS, DuckFE
    from .utils.formula_parser import FormulaParser

    parsed_formula = FormulaParser().parse(formula)
    fe_cols = parsed_formula.get_fe_names()
    has_iv = parsed_formula.has_instruments()

    logger.debug(
        f"Parsed: outcomes={parsed_formula.get_outcome_names()}, "
        f"covariates={parsed_formula.get_covariate_names()}, "
        f"fe={fe_cols}, cluster={parsed_formula.cluster}, "
        f"has_iv={has_iv}, fitter={fitter}"
    )

    # ------------------------------------------------------------------
    # 4. Resolve data source
    # ------------------------------------------------------------------
    resolved_db, table_name, obj_to_register = _resolve_data_source(
        data, cache_dir, db_name
    )

    # ------------------------------------------------------------------
    # 5. Resolve FE method
    # ------------------------------------------------------------------
    resolved_fe_method = fe_method
    if fe_method == FEMethod.AUTO:
        resolved_fe_method = (
            FEMethod.MUNDLAK if has_iv
            else (FEMethod.DEMEAN if fe_cols else None)
        )

    # ------------------------------------------------------------------
    # 6. Build VcovSpec (once at the API boundary)
    # ------------------------------------------------------------------
    # Bootstrap/none are runtime-only flags; the analytical spec falls back to HC1.
    _spec_se = (
        se_method
        if not (isinstance(se_method, str) and se_method in (SEMethod.BS, SEMethod.NONE, "none"))
        else SEMethod.HC1
    )
    vcov_spec = VcovSpec.build(
        se_method=_spec_se,
        has_fixef=bool(fe_cols),
        is_iv=has_iv,
    )

    # ------------------------------------------------------------------
    # 7. Select and construct estimator
    # ------------------------------------------------------------------
    _common = dict(
        db_name=resolved_db,
        table_name=table_name,
        formula=parsed_formula,
        subset=subset,
        n_bootstraps=n_bootstraps,
        round_strata=round_strata,
        seed=seed,
        duckdb_kwargs=duckdb_kwargs or None,
        n_jobs=threads,   # threads controls bootstrap parallelism (n_jobs in estimators)
        fitter=fitter,
        remove_singletons=remove_singletons,
        vcov_spec=vcov_spec,
    )

    if has_iv:
        estimator = Duck2SLS(**_common, fe_method=resolved_fe_method)
    elif fe_cols:
        if resolved_fe_method == FEMethod.MUNDLAK:
            fe_method_str = "mundlak"
        elif resolved_fe_method == FEMethod.DEMEAN:
            fe_method_str = "iterative_demean"
        elif resolved_fe_method == FEMethod.AUTO_FE:
            raise NotImplementedError(
                "fe_method='auto_fe' is experimental and has been temporarily disabled. "
                "Use fe_method='demean' (iterative demeaning) instead."
            )
        else:
            raise ValueError(
                f"With fixed effects, fe_method must be '{FEMethod.MUNDLAK}' "
                f"or '{FEMethod.DEMEAN}', got '{resolved_fe_method}'"
            )
        _duckfe_extra = {}
        if fe_types is not None:
            _duckfe_extra["fe_types"] = fe_types
        if max_fixed_fe_levels is not None:
            _duckfe_extra["max_fixed_fe_levels"] = max_fixed_fe_levels
        estimator = DuckFE(**_common, method=fe_method_str, **_duckfe_extra)
    else:
        estimator = DuckRegression(**_common)

    # For in-memory data sources, register the object as a DuckDB view.
    if obj_to_register is not None:
        estimator.conn.register(_DUCKDB_VIEW_NAME, obj_to_register)

    # Fit the model
    estimator.fit(se_method=se_method)

    logger.debug("=== duckreg END ===")
    return estimator


# Backward compatibility alias
compressed_ols = duckreg


