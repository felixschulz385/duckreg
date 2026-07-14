"""Base estimator class for all DuckDB-based estimators"""
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

import duckdb
import numpy as np

logger = logging.getLogger(__name__)


class SEMethod:
    """Standard error computation methods"""
    IID = "iid"
    HC1 = "HC1"
    CRV1 = "CRV1"
    NONE = "none"


class _StandardErrorAccessor:
    """Callable accessor that preserves ``model.se()`` and string compatibility."""

    def __init__(self, estimator: "DuckEstimator"):
        self._estimator = estimator

    def _value(self) -> Optional[str]:
        return getattr(self._estimator, "_se_type", None)

    def __call__(self):
        results = getattr(self._estimator, "results", None)
        if results is None:
            raise ValueError("No results available. Call fit() first.")
        return results.se()

    def __str__(self) -> str:
        return self._value() or ""

    def __repr__(self) -> str:
        return repr(self._value())

    def __eq__(self, other: object) -> bool:
        return self._value() == other


class DuckEstimator(ABC):
    """Abstract base class for all DuckDB-based estimators.
    
    This provides the minimal interface that all estimators must implement,
    plus shared DuckDB connection management.
    """
    
    def __init__(
        self,
        db_name: str,
        table_name: str,
        seed: int,
        n_bootstraps: int = 0,
        fitter: str = "auto",
        keep_connection_open: bool = False,
        compression: Any = "auto",
        round_strata: int = None,
        duckdb_kwargs: dict = None,
        remove_singletons: bool = True,
        retain_compressed: Optional[bool] = None,
        demean_backend: str = "auto",
    ):
        logger.debug(f"DuckEstimator.__init__: db={db_name}, table={table_name}")
        
        self.db_name = db_name
        self.table_name = table_name
        self.n_bootstraps = n_bootstraps
        self.seed = seed
        self.fitter = fitter
        self.keep_connection_open = keep_connection_open
        if compression is not None and round_strata is not None and compression != round_strata:
            raise ValueError(
                "compression and round_strata specify different settings. "
                "Use only one, or make them equal."
            )
        resolved_compression = compression if compression != "auto" else "auto"
        if compression is None and round_strata is not None:
            resolved_compression = round_strata
        if resolved_compression not in (None, "auto"):
            if not isinstance(resolved_compression, int) or resolved_compression < -1:
                raise ValueError(
                    f"compression must be an integer >= -1 or None, got {resolved_compression!r}"
                )
        self.compression = resolved_compression
        self.round_strata = None if resolved_compression in (-1, "auto") else resolved_compression
        self.duckdb_kwargs = duckdb_kwargs
        self.remove_singletons = remove_singletons
        self._legacy_lazy_compressed = retain_compressed is None
        self.retain_compressed = bool(retain_compressed)
        self.demean_backend = demean_backend
        self.resolved_fitter = None if fitter == "auto" else fitter
        self.resolved_compression = None if resolved_compression == "auto" else resolved_compression
        self.resolved_demean_backend = None if demean_backend == "auto" else demean_backend
        self.estimated_memory = None
        self.compression_ratio = None
        self.map_iterations = None
        self.metadata: Dict[str, Any] = {}
        self._closed = False
        
        # State
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self.rng: Optional[np.random.Generator] = None
        self.point_estimate: Optional[np.ndarray] = None
        self.vcov: Optional[np.ndarray] = None
        self._se_type: Optional[str] = None
        self.se = _StandardErrorAccessor(self)
        self.coef_names_: Optional[List[str]] = None
        self.n_obs: Optional[int] = None
        self.n_compression_base_rows: Optional[int] = None
        self.n_rows_dropped_singletons: int = 0
        
        self._init_connection()

    @property
    def compression_disabled(self) -> bool:
        """Whether row-level compression has been disabled entirely."""
        return self.compression == -1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        """Release DuckDB resources. Safe to call more than once."""
        if self._closed:
            return
        conn = self.conn
        if conn is not None:
            try:
                self._cleanup_internal_objects()
            finally:
                try:
                    conn.close()
                finally:
                    self.conn = None
                    self._closed = True

    def _cleanup_internal_objects(self):
        """Drop scratch objects created by an estimator before closing."""
        if self.conn is None:
            return
        names = {
            getattr(self, "_COMPRESSED_VIEW", None),
            getattr(self, "_STAGING_TABLE", None),
            getattr(self, "_DEMEANED_STAGING", None),
            "_fs_view", "_fs_demean_view", "_fs_first_stage_norm",
            "demeaned_data", "design_matrix", "_numpy_demeaned_data",
            "_iv_staging", "_duckfe_staging", "_demeaned_staging",
            "iv_compressed", "_resid_store", "_singleton_work",
            "_singleton_next", "_numpy_iv_demeaned",
        }
        for name in filter(None, names):
            for kind in ("VIEW", "TABLE"):
                try:
                    self.conn.execute(f'DROP {kind} IF EXISTS "{name}"')
                except Exception:
                    pass

    def _init_connection(self):
        """Initialize DuckDB connection and RNG"""
        self.conn = duckdb.connect(self.db_name)
        self._apply_duckdb_config(self.duckdb_kwargs)
        self.rng = np.random.default_rng(self.seed)

    def _apply_duckdb_config(self, config: Optional[Dict[str, Any]]):
        """Apply DuckDB configuration settings"""
        if config:
            for key, value in config.items():
                self.conn.execute(f"SET {key} = '{value}'")

    def fit(self, se_method: str = SEMethod.IID):
        """Main fitting method - orchestrates the estimation pipeline.
        
        Subclasses should not override this; override the individual steps instead.
        """
        logger.debug(f"fit() START with se_method={se_method}")

        self._no_vcov = se_method in (SEMethod.NONE, "none")
        # If se_method is a dict (e.g. {"CRV1": "firm_id"}), build/update
        # vcov_spec from it BEFORE prepare_data() runs so the staging table
        # includes the cluster column via _effective_cluster_col.
        if isinstance(se_method, dict):
            from ..core.vcov import VcovSpec
            has_fixef = bool(getattr(self, 'fe_cols', None))
            is_iv = bool(getattr(self, 'endogenous_vars', None))
            self.vcov_spec = VcovSpec.build(
                se_method=se_method,
                has_fixef=has_fixef,
                is_iv=is_iv,
            )

        try:
            try:
                self.prepare_data()
            except MemoryError:
                if self.demean_backend != "auto" or self.resolved_demean_backend != "numpy":
                    raise
                self.demean_backend = self.resolved_demean_backend = "duckdb"
                self.metadata["numpy_demean_memory_fallback"] = True
                self.prepare_data()
            self._resolve_compression_if_needed()
            # IV and mediation staging know their analytic row count before
            # building equation data, so resolve early enough to choose the
            # correct setup path. Pooled/FE models resolve after compression.
            if self.fitter == "auto" and getattr(self, "n_obs", None) is not None:
                self._resolve_fitter_if_needed()
            try:
                self.compress_data()
            except MemoryError:
                if not getattr(self, "_fitter_was_auto", False) or self.resolved_fitter != "numpy":
                    raise
                self._release_numpy_state()
                self.fitter = self.resolved_fitter = "duckdb"
                self.metadata["numpy_memory_fallback"] = True
                self.compress_data()
            self._resolve_fitter_if_needed()
            try:
                self.point_estimate = self.estimate()
            except MemoryError:
                if self.fitter != "numpy" or self.resolved_fitter != "numpy" or not getattr(self, "_fitter_was_auto", False):
                    raise
                self._release_numpy_state()
                self.fitter = self.resolved_fitter = "duckdb"
                self.metadata["numpy_memory_fallback"] = True
                self.point_estimate = self.estimate()
            self._compute_standard_errors(se_method)
            transformer = getattr(self, "_transformer", None) or getattr(self, "_demean_transformer", None)
            self.map_iterations = getattr(transformer, "n_iterations", None)
            if self.n_obs:
                compressed_rows = getattr(self, "n_compressed_rows", None) or self.n_obs
                self.compression_ratio = compressed_rows / self.n_obs
            self.metadata.update({
                "resolved_fitter": self.resolved_fitter,
                "resolved_compression": self.resolved_compression,
                "resolved_demean_backend": self.resolved_demean_backend,
                "estimated_memory": self.estimated_memory,
                "compression_ratio": self.compression_ratio,
                "map_iterations": self.map_iterations,
            })
            for setting in ("threads", "memory_limit", "max_temp_directory_size"):
                try:
                    self.metadata[setting] = self.conn.execute(
                        f"SELECT current_setting('{setting}')"
                    ).fetchone()[0]
                except Exception:
                    pass
            if self.retain_compressed:
                try:
                    if hasattr(self, "_ensure_data_fetched"):
                        try:
                            self._ensure_data_fetched(force=True)
                        except TypeError:
                            self._ensure_data_fetched()
                    elif hasattr(type(self), "df_compressed"):
                        _ = self.df_compressed
                except Exception as exc:
                    logger.debug("No materialized compressed relation to retain: %s", exc)
            elif not self._legacy_lazy_compressed and hasattr(self, "_df_compressed"):
                self._df_compressed = None
                self._data_fetched = False
            # IV and mediation keep observation-sized working arrays only until
            # all requested covariance calculations have finished.
            for name in ("_y", "_X_fitted", "_X_actual", "_Z", "_weights", "_cluster_ids"):
                if hasattr(self, name):
                    setattr(self, name, None)
        finally:
            keep_legacy_lazy = self._legacy_lazy_compressed and self.fitter == "duckdb"
            if not self.keep_connection_open and not keep_legacy_lazy:
                self.close()
        
        logger.debug(f"fit() END")

    def _compute_standard_errors(self, se_method: str):
        """Dispatch standard error computation based on method"""
        # When vcov_spec is set (via duckreg API), derive the effective method from it.
        # This ensures the parsed VcovSpec is used rather than the string fallback.
        if getattr(self, "_no_vcov", False):
            self.vcov = None
            self._se_type = SEMethod.NONE
            return
        vcov_spec = getattr(self, 'vcov_spec', None)
        effective = vcov_spec.vcov_detail if vcov_spec is not None else se_method

        # For string se_method (not dict), ensure vcov_spec is built so all
        # subclasses (DuckLinearModel, Duck2SLS) can read self.vcov_spec in
        # their fit_vcov implementations.
        if vcov_spec is None and effective not in (SEMethod.NONE, None):
            from ..core.vcov import VcovSpec
            has_fixef = bool(getattr(self, 'fe_cols', None))
            is_iv     = bool(getattr(self, 'endogenous_vars', None))
            try:
                self.vcov_spec = VcovSpec.build(
                    effective, None, has_fixef=has_fixef, is_iv=is_iv
                )
            except Exception:
                pass

        if effective == SEMethod.NONE:
            logger.debug("Skipping standard error computation")
        elif effective in (SEMethod.IID, SEMethod.HC1, SEMethod.CRV1,
                           'HC2', 'HC3', 'CRV3', 'hetero', 'iid'):
            logger.debug(f"Computing {effective} standard errors")
            self.fit_vcov(effective)
        else:
            logger.warning(f"Unknown se_method '{effective}'")

    def _resolve_fitter_if_needed(self):
        if self.fitter != "auto":
            self.resolved_fitter = self.fitter
            return
        self._fitter_was_auto = True
        rows = int(getattr(self, "n_compressed_rows", None) or getattr(self, "n_obs", 0) or 0)
        k = max(1, int(getattr(self, "_get_n_coefs", lambda: 1)()))
        cluster_cols = len(getattr(getattr(self, "vcov_spec", None), "cluster_vars", []) or [])
        self.estimated_memory = 3 * rows * (8 * (k + 2) + 8 * cluster_cols)
        budget = min(2 * 1024**3, self._duckdb_memory_budget())
        self.fitter = self.resolved_fitter = "numpy" if self.estimated_memory < budget else "duckdb"

    def _resolve_compression_if_needed(self):
        """Resolve automatic compression from a bounded deterministic sample."""
        if self.compression != "auto":
            self.resolved_compression = self.compression
            return
        formula = getattr(self, "formula", None) or getattr(self, "_formula", None)
        keys = []
        if formula is not None:
            keys.extend(
                v.sql_name for v in formula.covariates if not v.is_intercept()
            )
        cluster = getattr(self, "_effective_cluster_col", None)
        if cluster:
            keys.append(cluster)
        # Intercept-only specifications have a single exact stratum.
        if not keys:
            resolved = None
            ratio = 0.0
        else:
            aliases = [f"__k{i}" for i in range(len(keys))]
            projection = ", ".join(
                f"{key} AS {alias}" for key, alias in zip(keys, aliases)
            )
            distinct = ", ".join(aliases)
            where = self._build_where_clause(getattr(self, "subset", None))
            query = f"""
                WITH sampled AS (
                    SELECT {projection} FROM {self.table_name}
                    {where}
                    USING SAMPLE reservoir(100000 ROWS) REPEATABLE ({int(self.seed)})
                )
                SELECT
                    (SELECT COUNT(*) FROM (SELECT DISTINCT {distinct} FROM sampled)),
                    (SELECT COUNT(*) FROM sampled)
            """
            try:
                distinct_n, sample_n = self.conn.execute(query).fetchone()
                ratio = float(distinct_n) / sample_n if sample_n else 0.0
            except Exception:
                # Expression-heavy formulas are resolved against transformed
                # columns later; disabling grouping remains exact and safe.
                ratio = 1.0
            resolved = None if ratio <= 0.75 else -1
        self.compression = self.resolved_compression = resolved
        self.round_strata = None
        self.metadata["compression_sample_ratio"] = ratio

    def _duckdb_memory_budget(self) -> int:
        """Return forty percent of DuckDB's effective memory limit."""
        try:
            raw = self.conn.execute("SELECT current_setting('memory_limit')").fetchone()[0]
            units = {"B": 1, "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4,
                     "KIB": 1024, "MIB": 1024**2, "GIB": 1024**3, "TIB": 1024**4}
            parts = str(raw).upper().replace(" ", "").rstrip("B")
            import re
            match = re.fullmatch(r"([0-9.]+)([KMGT]?I?)", parts)
            if match:
                suffix = (match.group(2) + "B") if match.group(2) else "B"
                return int(float(match.group(1)) * units[suffix] * 0.4)
        except Exception:
            pass
        return 2 * 1024**3

    def _release_numpy_state(self):
        for name in ("_numpy_X", "_numpy_y", "_numpy_weights", "_numpy_cluster_ids"):
            setattr(self, name, None)

    # -------------------------------------------------------------------------
    # Abstract methods - must be implemented by subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    def prepare_data(self):
        """Prepare data tables for estimation.
        
        This may include:
        - Creating design matrices
        - Running first-stage regressions (for IV)
        - Computing Mundlak means (for Mundlak approach)
        """
        pass

    @abstractmethod
    def compress_data(self):
        """Compress data for efficient estimation.
        
        Creates aggregated views/tables with sufficient statistics.
        """
        pass

    @abstractmethod
    def estimate(self) -> np.ndarray:
        """Estimate model coefficients.
        
        Returns:
            Array of coefficient estimates
        """
        pass

    @abstractmethod
    def fit_vcov(self, se_method: str = SEMethod.HC1):
        """Compute variance-covariance matrix."""
        pass

    # -------------------------------------------------------------------------
    # Common utility methods
    # -------------------------------------------------------------------------

    def _get_boolean_columns(self) -> set:
        """Get boolean columns from source table (cached)."""
        if hasattr(self, '_boolean_cols') and self._boolean_cols is not None:
            return self._boolean_cols
        
        all_cols = set(self.formula.get_source_columns_for_null_check())
        cols_sql = ', '.join(f"'{c}'" for c in all_cols)
        query = f"""
        SELECT column_name FROM (DESCRIBE SELECT * FROM {self.table_name})
        WHERE column_name IN ({cols_sql}) AND column_type = 'BOOLEAN'
        """
        self._boolean_cols = set(self.conn.execute(query).fetchdf()['column_name'].tolist())
        return self._boolean_cols

    def _get_table_columns(self, table_name: str) -> set:
        """Get column names from a table"""
        return set(
            self.conn.execute(f"SELECT column_name FROM (DESCRIBE {table_name})")
            .fetchdf()['column_name'].tolist()
        )

    def _build_where_clause(self, user_subset: Optional[str] = None) -> str:
        """Build WHERE clause with NULL checks and optional user subset"""
        if hasattr(self, 'formula'):
            return self.formula.get_where_clause_sql(user_subset)
        return f"WHERE ({user_subset})" if user_subset else ""

    def _build_qualify_singleton_filter(self, fe_col_sql_names: List[str]) -> str:
        """Build QUALIFY clause to exclude singleton groups from multiple FE columns.
        
        Uses window functions for efficiency - single pass filter vs multiple DELETE statements.
        A singleton group is one with exactly one observation.
        
        Args:
            fe_col_sql_names: List of SQL names of fixed effects columns
            
        Returns:
            QUALIFY clause excluding singletons (empty string if remove_singletons=False)
        """
        if not self.remove_singletons or not fe_col_sql_names:
            return ""
        
        # Build window function for all FE columns
        partition_clause = ', '.join(fe_col_sql_names)
        return f"QUALIFY count(*) OVER (PARTITION BY {partition_clause}) > 1"

    def _remove_singleton_observations(self, table_name: str, fe_col_sql_names: List[str]):
        """Remove observations from singleton FE groups if remove_singletons=True.
        
        Strategy:
        Uses ANTI JOIN to exclude observations from singleton groups (groups with < 2 obs).
        Processes each FE dimension sequentially since removing singletons in one dimension
        can create new singletons in another dimension.
        
        An observation is removed if it belongs to a singleton group in ANY FE dimension.
        
        Args:
            table_name: Name of the table to filter
            fe_col_sql_names: List of SQL names of fixed effects columns
        """
        if not self.remove_singletons or not fe_col_sql_names:
            return
        
        logger.debug(f"Removing singleton FE observations from {len(fe_col_sql_names)} FE groups")
        
        rows_before = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        # Process each FE dimension sequentially
        for fe_sql in fe_col_sql_names:
            # Use ANTI JOIN to exclude singleton groups directly
            self.conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT *
            FROM {table_name}
            ANTI JOIN (
                SELECT {fe_sql}
                FROM {table_name}
                GROUP BY {fe_sql}
                HAVING COUNT(*) < 2
            ) singletons
            USING ({fe_sql})
            """)
        
        rows_after = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        self.n_rows_dropped_singletons = rows_before - rows_after
        
        logger.debug(f"After singleton removal: {rows_after} observations "
                    f"({self.n_rows_dropped_singletons} rows removed)")

    def summary(self) -> Dict[str, Any]:
        """Provide results summary. Subclasses should override for richer output."""
        return {
            "point_estimate": self.point_estimate,
            "coef_names": self.coef_names_,
            "n_obs": self.n_obs,
            "se_type": self._se_type,
        }
