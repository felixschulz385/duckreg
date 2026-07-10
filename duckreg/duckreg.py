"""High-level API for compressed OLS regression.

This module provides the main entry point for users to run compressed OLS regressions
with support for fixed effects, instrumental variables, and various standard error methods.
It handles formula parsing, data source resolution, and estimator selection.
"""
import logging
from typing import Any, Dict, Optional, Union

from .core.vcov import VcovSpec
from .estimators.base import SEMethod
from .utils.api import (
    FEMethod,
    MUNDLAK_DISABLED_MESSAGE,
    _resolve_data_source,
    _DUCKDB_VIEW_NAME,
)

logger = logging.getLogger(__name__)

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
    compression: Optional[int] = None,
    seed: int = 42,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
    check_interval: int = 10,
    convergence_sample: float = 1.0,
    min_iterations_before_check: int = 5,
    check_interval_growth: bool = True,
    max_check_interval: int = 25,
    singleton_pruning: str = "iterative",
    fe_order: str = "input",
    drop_constant_variables: bool = False,
    residual_type: str = "DOUBLE",
    # ── FE classification settings ────────────────────────────────────────────
    fe_types: Optional[Dict] = None,
    max_fixed_fe_levels: Optional[int] = None,
    # ── DuckDB resource kwargs: threads, memory_limit, max_temp_directory_size
    **kwargs,
) -> object:
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
            ``'none'``) or a dict specifying cluster-robust SEs::

                se_method = {"CRV1": "state"}          # single cluster
                se_method = {"CRV1": "state + firm"}   # two-way clustering

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
        compression: Compression setting for grouping duplicate strata.
            ``None`` means exact compression, non-negative integers round
            continuous strata columns before grouping, and ``-1`` disables
            compression entirely.
        seed: Global random seed for reproducibility.
        max_iterations: For FE demeaning, maximum MAP iterations.
        tolerance: For FE demeaning, exact maximum absolute remaining
            FE-group mean required for convergence.
        check_interval: For FE demeaning, evaluate convergence every
            ``check_interval`` MAP iterations. Higher values reduce
            convergence-check overhead on large jobs.
        convergence_sample: For FE demeaning, fraction of rows sampled when
            checking MAP convergence. Lower values reduce convergence-check
            I/O on large jobs.
        min_iterations_before_check: For FE demeaning, minimum number of MAP
            iterations before non-final convergence checks are allowed.
        check_interval_growth: For FE demeaning, whether convergence checks
            become less frequent after early iterations.
        max_check_interval: For FE demeaning, upper bound for adaptive
            convergence-check intervals.
        singleton_pruning: For FE demeaning, singleton pruning strategy:
            ``"iterative"`` for cascading pruning or ``"one_pass"`` for a
            single pruning pass.
        fe_order: For FE demeaning, MAP sweep order: ``"input"``,
            ``"ascending_groups"``, or ``"descending_groups"``.
        drop_constant_variables: For FE demeaning, skip variables that are
            constant after filtering and set them to zero when FEs are present.
        residual_type: For FE demeaning, storage type for residual columns:
            ``"DOUBLE"`` or ``"FLOAT"``. ``"FLOAT"`` requires a relaxed
            tolerance.
        **kwargs: DuckDB resource settings passed directly as keyword arguments:

            * ``threads`` (int) – number of DuckDB threads.
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

    """
    logger.debug("=== duckreg START ===")

    if isinstance(se_method, str) and se_method == "BS":
        raise ValueError(
            "Bootstrap standard errors are no longer supported. "
            "Use 'iid', 'HC1', 'HC2', 'HC3', or cluster-robust se_method values instead."
        )

    # ------------------------------------------------------------------
    # 1. Extract and validate DuckDB / resource kwargs
    # ------------------------------------------------------------------
    threads = int(kwargs.pop("threads", 1))
    memory_limit = kwargs.pop("memory_limit", None)
    max_temp_dir_size = kwargs.pop("max_temp_directory_size", None)

    removed_kwargs = []
    for name in ("bootstrap", "round_strata", "n_jobs", "duckdb_kwargs", "n_bootstraps"):
        if name in kwargs:
            removed_kwargs.append(name)
            kwargs.pop(name)

    if removed_kwargs:
        raise TypeError(
            "duckreg() got unsupported legacy keyword argument(s): "
            + ", ".join(sorted(removed_kwargs))
        )

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
    # 2. Compression settings
    # ------------------------------------------------------------------
    if compression is not None and not isinstance(compression, int):
        raise TypeError(
            f"compression must be an integer or None, got {type(compression)!r}"
        )
    if compression is not None and compression < -1:
        raise ValueError(
            f"compression must be >= -1 or None, got {compression!r}"
        )

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
            FEMethod.DEMEAN if has_iv and fe_cols
            else (FEMethod.DEMEAN if fe_cols else None)
        )

    if fe_cols and resolved_fe_method == FEMethod.MUNDLAK:
        raise NotImplementedError(MUNDLAK_DISABLED_MESSAGE)

    # ------------------------------------------------------------------
    # 6. Build VcovSpec (once at the API boundary)
    # ------------------------------------------------------------------
    vcov_spec = VcovSpec.build(
        se_method=se_method if se_method not in (SEMethod.NONE, "none") else SEMethod.HC1,
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
        compression=compression,
        seed=seed,
        duckdb_kwargs=duckdb_kwargs or None,
        fitter=fitter,
        remove_singletons=remove_singletons,
        vcov_spec=vcov_spec,
    )

    # ── Mediation model (via(...) syntax) ────────────────────────────
    if parsed_formula.has_mediators():
        from .estimators import DuckMediation

        # All non-intercept, non-mediator covariates are treated as exposures.
        med_names    = parsed_formula.get_mediator_names()
        exposure_names = [
            v for v in parsed_formula.get_non_intercept_simple_covariate_names()
            if v not in med_names
        ]

        # Resolve cluster column (first cluster var from VcovSpec, if present)
        _cluster_col: Optional[str] = None
        if vcov_spec.is_clustered and vcov_spec.cluster_vars:
            _cluster_col = vcov_spec.cluster_vars[0]

        # Mediation currently supports only the demean FE path.
        _med_fe_method = resolved_fe_method or "demean"
        if _med_fe_method not in ("demean", "mundlak"):
            _med_fe_method = "demean"
        if fe_cols and _med_fe_method == FEMethod.MUNDLAK:
            raise NotImplementedError(MUNDLAK_DISABLED_MESSAGE)

        estimator = DuckMediation(
            db_name=resolved_db,
            table_name=table_name,
            outcome=parsed_formula.get_outcome_names()[0],
            exposures=exposure_names,
            mediators=med_names,
            fe_cols=fe_cols,
            fe_method=_med_fe_method,
            cluster_col=_cluster_col,
            vcov_spec=vcov_spec,
            fitter=fitter,
            subset=subset,
            seed=seed,
            remove_singletons=remove_singletons,
            duckdb_kwargs=duckdb_kwargs or None,
            formula=parsed_formula,
        )

        if obj_to_register is not None:
            estimator.conn.register(_DUCKDB_VIEW_NAME, obj_to_register)

        estimator.fit(se_method=se_method)
        logger.debug("=== duckreg END (mediation) ===")
        return estimator

    if has_iv:
        estimator = Duck2SLS(
            **_common,
            method=resolved_fe_method or "mundlak",
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
    elif fe_cols:
        if resolved_fe_method == FEMethod.MUNDLAK:
            raise NotImplementedError(MUNDLAK_DISABLED_MESSAGE)
        elif resolved_fe_method == FEMethod.DEMEAN:
            fe_method_str = "iterative_demean"
        elif resolved_fe_method == FEMethod.AUTO_FE:
            raise NotImplementedError(
                "fe_method='auto_fe' is experimental and has been temporarily disabled. "
                "Use fe_method='demean' (iterative demeaning) instead."
            )
        else:
            raise ValueError(
                f"With fixed effects, fe_method must be '{FEMethod.DEMEAN}', "
                f"got '{resolved_fe_method}'"
            )
        _duckfe_extra = {}
        if fe_types is not None:
            _duckfe_extra["fe_types"] = fe_types
        if max_fixed_fe_levels is not None:
            _duckfe_extra["max_fixed_fe_levels"] = max_fixed_fe_levels
        _duckfe_extra["max_iterations"] = max_iterations
        _duckfe_extra["tolerance"] = tolerance
        _duckfe_extra["check_interval"] = check_interval
        _duckfe_extra["convergence_sample"] = convergence_sample
        _duckfe_extra["min_iterations_before_check"] = min_iterations_before_check
        _duckfe_extra["check_interval_growth"] = check_interval_growth
        _duckfe_extra["max_check_interval"] = max_check_interval
        _duckfe_extra["singleton_pruning"] = singleton_pruning
        _duckfe_extra["fe_order"] = fe_order
        _duckfe_extra["drop_constant_variables"] = drop_constant_variables
        _duckfe_extra["residual_type"] = residual_type
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
