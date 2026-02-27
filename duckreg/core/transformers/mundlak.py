"""Mundlak device transformer for fixed effects."""
import logging
import re
from typing import Any, Dict, List, Optional

import duckdb

from .base import FETransformer
from ...core.sql_builders import (
    build_add_fixed_fe_dummy_means_sql,
    build_add_mundlak_means_sql,
    classify_fe_type,
    get_fe_unique_levels,
    profile_fe_column,
)

logger = logging.getLogger(__name__)


class MundlakTransformer(FETransformer):
    """Absorbs fixed effects via the Mundlak device.

    The Mundlak device controls for FE-level heterogeneity by augmenting
    the regression with within-group means of the covariates.  The data
    remain in levels so between-group effects are identified.

    For unbalanced panels with mixed FE types (e.g., firm + year), the
    transformer applies the Wooldridge correction:

    1. FEs are classified as *fixed* (low cardinality, e.g., year) or
       *asymptotic* (high cardinality, e.g., firm).
    2. Mundlak means are added for *asymptotic* FEs only.
    3. Binary dummy columns are added for *fixed* FEs (excluding the
       reference level).
    4. Within-asymptotic-FE means of the fixed-FE dummies are added to
       correct for unbalancedness.

    After :meth:`fit_transform`, the result table ``design_matrix`` contains
    the original data columns plus all the added columns.  All added column
    names are accessible via :attr:`extra_regressors`.

    :meth:`transform_query` returns an identity fragment ``x, y, ...``
    because the original variable names are unchanged.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    table_name : str
        Source table/view name.
    fe_cols : List[str]
        SQL column names of the FE dimensions (already resolved, SQL-safe).
    cluster_col : str, optional
        SQL column name for cluster / alias (e.g. ``"__cluster__"``).
    covariate_cols : List[str], optional
        SQL column names of the covariates for which to compute group means.
        Typically all non-intercept covariates.
    remove_singletons : bool
        Drop FE singletons before adding means.
    fe_types : Dict[str, str], optional
        Override automatic classification: ``{fe_col: 'fixed'|'asymptotic'}``.
    cardinality_threshold : int
        Cardinality at or below which a FE is classified as *fixed*.
    singleton_threshold : float
        Singleton share above which a FE is classified as *fixed*.
    max_fixed_fe_levels : int
        Upper bound on levels for a *fixed* FE; exceeding it forces
        reclassification as *asymptotic*.
    """

    _RESULT_TABLE = "design_matrix"

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        table_name: str,
        fe_cols: List[str],
        cluster_col: Optional[str] = None,
        covariate_cols: Optional[List[str]] = None,
        remove_singletons: bool = True,
        fe_types: Optional[Dict[str, str]] = None,
        cardinality_threshold: int = 50,
        singleton_threshold: float = 0.1,
        max_fixed_fe_levels: int = 100,
        **kwargs,
    ):
        super().__init__(
            conn=conn,
            table_name=table_name,
            fe_cols=fe_cols,
            cluster_col=cluster_col,
            remove_singletons=remove_singletons,
            **kwargs,
        )
        self.covariate_cols: List[str] = list(covariate_cols or [])
        self.fe_types: Dict[str, str] = fe_types or {}
        self.cardinality_threshold = cardinality_threshold
        self.singleton_threshold = singleton_threshold
        self.max_fixed_fe_levels = max_fixed_fe_levels

        # Populated during fit_transform
        self.fe_metadata: Dict[str, Dict[str, Any]] = {}
        self._mundlak_mean_cols: List[str] = []
        self._fixed_dummy_cols: List[str] = []
        self._dummy_mean_cols: List[str] = []

    # -------------------------------------------------------------------------
    # FETransformer interface
    # -------------------------------------------------------------------------

    def fit_transform(self, variables: List[str], where_clause: str = "") -> str:
        """Build the design matrix with Mundlak means and panel correction.

        Parameters
        ----------
        variables : List[str]
            SQL column names to include (outcomes + covariates, excluding FE
            columns and the cluster column which are handled separately).
        where_clause : str
            SQL ``WHERE`` clause (including the keyword) or empty string.

        Returns
        -------
        str
            ``"design_matrix"`` — name of the table with all augmented columns.
        """
        self._create_design_matrix(variables, where_clause)
        self._remove_singleton_observations(self._RESULT_TABLE)

        self._n_obs = self.conn.execute(
            f"SELECT COUNT(*) FROM {self._RESULT_TABLE}"
        ).fetchone()[0]

        self._profile_and_classify_fes()
        self._add_mundlak_means()
        self._add_fixed_fe_dummies()
        self._add_fixed_fe_dummy_means()

        self._fitted = True
        return self._RESULT_TABLE

    def transform_query(self, variables: List[str]) -> str:
        """Return identity ``SELECT`` fragment: ``x, y, ...`` (no renaming).

        The Mundlak transformer leaves original variable names untouched;
        only new columns are appended.
        """
        return ", ".join(variables)

    @property
    def n_obs(self) -> int:
        """Observations in ``design_matrix`` after singleton removal."""
        if self._n_obs is None:
            raise RuntimeError("fit_transform() has not been called yet.")
        return self._n_obs

    @property
    def df_correction(self) -> int:
        """Zero — FE parameters are explicit regressors (counted in *k*)."""
        return 0

    @property
    def extra_regressors(self) -> List[str]:
        """All columns added by the Mundlak device.

        In order: Mundlak mean columns, fixed-FE dummy columns,
        unbalanced-panel dummy-mean columns.
        """
        return (
            list(self._mundlak_mean_cols)
            + list(self._fixed_dummy_cols)
            + list(self._dummy_mean_cols)
        )

    @property
    def has_intercept(self) -> bool:
        """``True`` — the Mundlak device is a levels model requiring an intercept."""
        return True

    # -------------------------------------------------------------------------
    # Design-matrix construction
    # -------------------------------------------------------------------------

    def _create_design_matrix(self, variables: List[str], where_clause: str) -> None:
        """Create the initial ``design_matrix`` table from the source.

        Only complete-case rows (no NULL in any model column) are included.
        This ensures that the Mundlak group-mean columns are computed over
        exactly the same observations that will enter the regression,
        matching pyfixest's complete-case sample and preventing NaN
        contamination of group means from partially-observed rows.

        The filter covers FE columns, the cluster column (if any), and all
        outcome / covariate *variables*.
        """
        select_parts = list(self.fe_cols)
        if self.cluster_col:
            select_parts.append(self.cluster_col)
        select_parts.extend(variables)

        # Build a complete-case filter: every column that will be used
        # (FE dims, cluster, outcomes, covariates) must be non-null.
        all_model_cols = list(self.fe_cols) + variables
        if self.cluster_col:
            all_model_cols.append(self.cluster_col)
        null_conditions = " AND ".join(f"{v} IS NOT NULL" for v in all_model_cols)
        if where_clause:
            combined_where = f"{where_clause} AND {null_conditions}"
        else:
            combined_where = f"WHERE {null_conditions}"

        self.conn.execute(f"""
        CREATE OR REPLACE TABLE {self._RESULT_TABLE} AS
        SELECT {', '.join(select_parts)}
        FROM {self.table_name}
        {combined_where}
        """)

    # -------------------------------------------------------------------------
    # FE profiling and classification
    # -------------------------------------------------------------------------

    def _profile_and_classify_fes(self) -> None:
        """Profile each FE column and classify it as *fixed* or *asymptotic*."""
        logger.debug(f"Profiling {len(self.fe_cols)} FE dimension(s)")

        for i, fe_col in enumerate(self.fe_cols):
            try:
                profile = profile_fe_column(self.conn, self._RESULT_TABLE, fe_col)
            except Exception as exc:
                logger.warning(f"Failed to profile FE '{fe_col}': {exc}")
                continue

            user_override = self.fe_types.get(fe_col)
            fe_type = classify_fe_type(
                profile,
                cardinality_threshold=self.cardinality_threshold,
                singleton_threshold=self.singleton_threshold,
                user_override=user_override,
            )

            metadata: Dict[str, Any] = {
                "type": fe_type,
                "profile": profile,
                "sql_name": fe_col,
                "index": i,
            }

            if fe_type == "fixed":
                try:
                    levels = get_fe_unique_levels(
                        self.conn,
                        self._RESULT_TABLE,
                        fe_col,
                        max_levels=self.max_fixed_fe_levels,
                    )
                    metadata["levels"] = levels
                    metadata["reference_level"] = levels[0] if levels else None
                    logger.debug(
                        f"FE '{fe_col}' → FIXED "
                        f"({len(levels)} levels, reference={metadata['reference_level']})"
                    )
                except ValueError as exc:
                    logger.warning(
                        f"FE '{fe_col}' has too many levels; reclassifying as "
                        f"asymptotic. {exc}"
                    )
                    metadata["type"] = "asymptotic"
                    fe_type = "asymptotic"

            if fe_type == "asymptotic":
                logger.debug(
                    f"FE '{fe_col}' → ASYMPTOTIC "
                    f"(cardinality={profile['cardinality']}, "
                    f"avg_obs={profile['avg_obs_per_level']:.1f})"
                )

            self.fe_metadata[fe_col] = metadata

        n_fixed = sum(1 for m in self.fe_metadata.values() if m["type"] == "fixed")
        logger.debug(
            f"FE classification complete: {n_fixed} fixed, "
            f"{len(self.fe_metadata) - n_fixed} asymptotic"
        )

    # -------------------------------------------------------------------------
    # Column additions
    # -------------------------------------------------------------------------

    def _add_mundlak_means(self) -> None:
        """Add group-mean columns for each covariate × asymptotic FE pair."""
        available = set(
            self.conn.execute(
                f"SELECT column_name FROM (DESCRIBE {self._RESULT_TABLE})"
            ).fetchdf()["column_name"].tolist()
        )
        cov_cols = [c for c in self.covariate_cols if c in available]

        if not cov_cols:
            logger.debug("No covariate columns found for Mundlak means; skipping.")
            return

        for i, fe_col in enumerate(self.fe_cols):
            if self.fe_metadata.get(fe_col, {}).get("type") == "fixed":
                logger.debug(f"Skipping Mundlak means for fixed FE '{fe_col}'")
                continue

            sql = build_add_mundlak_means_sql(
                table_name=self._RESULT_TABLE,
                var_sql_names=cov_cols,
                fe_col_sql_name=fe_col,
                fe_index=i,
            )
            logger.debug(f"Adding Mundlak means for asymptotic FE '{fe_col}' (fe{i})")
            self.conn.execute(sql)

            for cov in cov_cols:
                self._mundlak_mean_cols.append(f"avg_{cov}_fe{i}")

        logger.debug(f"Added {len(self._mundlak_mean_cols)} Mundlak mean column(s)")

    def _add_fixed_fe_dummies(self) -> None:
        """Add binary dummy columns for each non-reference level of fixed FEs."""
        fixed_fes = [
            (name, meta)
            for name, meta in self.fe_metadata.items()
            if meta["type"] == "fixed"
        ]
        if not fixed_fes:
            logger.debug("No fixed FEs to add dummies for.")
            return

        logger.debug(f"Adding dummy variables for {len(fixed_fes)} fixed FE(s)")

        for fixed_name, fixed_meta in fixed_fes:
            levels = fixed_meta.get("levels", [])
            reference_level = fixed_meta.get("reference_level")
            fixed_sql = fixed_meta["sql_name"]
            non_ref = [lvl for lvl in levels if lvl != reference_level]

            if not non_ref:
                logger.debug(
                    f"No non-reference levels for '{fixed_name}'; skipping."
                )
                continue

            # Build all dummy columns for this FE in a single CTAS statement.
            # ALTER TABLE + UPDATE in two separate SQL statements passed to a
            # single conn.execute() is unreliable (DuckDB execute() handles
            # only one statement, so the UPDATE may be silently ignored,
            # leaving the dummy columns as NULL and corrupting coefficients).
            dummy_exprs = []
            new_col_names = []
            for level in non_ref:
                safe_level = re.sub(r'[^a-zA-Z0-9_]', '_', str(level))
                col_name = f"dummy_{fixed_sql}_{safe_level}"
                level_expr = f"'{level}'" if isinstance(level, str) else str(level)
                dummy_exprs.append(
                    f"CASE WHEN {fixed_sql} = {level_expr}"
                    f" THEN 1.0 ELSE 0.0 END AS {col_name}"
                )
                new_col_names.append(col_name)

            sql = f"""
            CREATE OR REPLACE TABLE {self._RESULT_TABLE} AS
            SELECT *, {', '.join(dummy_exprs)}
            FROM {self._RESULT_TABLE}
            """
            try:
                self.conn.execute(sql)
                self._fixed_dummy_cols.extend(new_col_names)
                for col_name in new_col_names:
                    logger.debug(f"Added dummy: {col_name}")
            except Exception as exc:
                logger.error(
                    f"Failed to add dummies for '{fixed_name}': {exc}"
                )
                raise

            logger.debug(
                f"Added {len(non_ref)} dummies for '{fixed_name}' "
                f"(reference: {reference_level})"
            )

    def _add_fixed_fe_dummy_means(self) -> None:
        """Add within-asymptotic-FE means of fixed-FE dummies (Wooldridge correction).

        Only applied when at least one fixed FE *and* at least one asymptotic
        FE are present.
        """
        fixed_fes = [
            (n, m) for n, m in self.fe_metadata.items() if m["type"] == "fixed"
        ]
        asymp_fes = [
            (n, m) for n, m in self.fe_metadata.items() if m["type"] == "asymptotic"
        ]

        if not fixed_fes or not asymp_fes:
            logger.debug(
                f"Skipping dummy-means: {len(fixed_fes)} fixed, "
                f"{len(asymp_fes)} asymptotic FEs"
            )
            return

        logger.debug(
            f"Adding dummy-means for unbalanced panel correction: "
            f"{len(fixed_fes)} fixed × {len(asymp_fes)} asymptotic FEs"
        )

        for fixed_name, fixed_meta in fixed_fes:
            levels = fixed_meta.get("levels", [])
            reference_level = fixed_meta.get("reference_level")
            fixed_sql = fixed_meta["sql_name"]

            if not levels:
                logger.warning(f"No levels for fixed FE '{fixed_name}'; skipping.")
                continue

            for asymp_name, asymp_meta in asymp_fes:
                asymp_sql = asymp_meta["sql_name"]
                asymp_index = asymp_meta["index"]

                try:
                    sql = build_add_fixed_fe_dummy_means_sql(
                        table_name=self._RESULT_TABLE,
                        fixed_fe_col_sql_name=fixed_sql,
                        fixed_fe_levels=levels,
                        asymptotic_fe_col_sql_name=asymp_sql,
                        asymptotic_fe_index=asymp_index,
                        reference_level=reference_level,
                    )
                    if sql.startswith("--"):
                        continue  # Only reference level — nothing to add

                    logger.debug(
                        f"Adding dummy-means: '{fixed_name}' within '{asymp_name}' "
                        f"(fe{asymp_index})"
                    )
                    self.conn.execute(sql)

                    for level in [lvl for lvl in levels if lvl != reference_level]:
                        safe_level = re.sub(r'[^a-zA-Z0-9_]', '_', str(level))
                        col_name = f"avg_{fixed_sql}_{safe_level}_fe{asymp_index}"
                        self._dummy_mean_cols.append(col_name)

                except Exception as exc:
                    logger.error(
                        f"Failed to add dummy-means for '{fixed_name}' × "
                        f"'{asymp_name}': {exc}"
                    )
                    raise

        logger.debug(f"Added {len(self._dummy_mean_cols)} dummy-mean column(s)")
