import numpy as np
import pandas as pd
from tqdm import tqdm

from ..duckreg import DuckReg
from ..fitters import wls


class DuckMundlakEventStudy(DuckReg):
    """Event study estimator using Mundlak device"""
    def __init__(
        self,
        db_name: str,
        table_name: str,
        outcome_var: str,
        treatment_col: str,
        unit_col: str,
        time_col: str,
        cluster_col: str,
        pre_treat_interactions: bool = True,
        n_bootstraps: int = 100,
        duckdb_kwargs: dict = None,
        variable_casts: dict = None,
        **kwargs,
    ):
        super().__init__(
            db_name=db_name,
            table_name=table_name,
            n_bootstraps=n_bootstraps,
            duckdb_kwargs=duckdb_kwargs,
            variable_casts=variable_casts,
            **kwargs,
        )
        self.table_name = table_name
        self.outcome_var = outcome_var
        self.treatment_col = treatment_col
        self.unit_col = unit_col
        self.time_col = time_col
        self.num_periods = None
        self.cohorts = None
        self.time_dummies = None
        self.post_treatment_dummies = None
        self.transformed_query = None
        self.compression_query = None
        self.cluster_col = cluster_col
        self.pre_treat_interactions = pre_treat_interactions

    def prepare_data(self):
        # Create cohort data using CTE instead of temp table
        self.cohort_cte = f"""
        WITH cohort_data AS (
            SELECT *,
                   CASE WHEN cohort_min = 2147483647 THEN NULL ELSE cohort_min END as cohort,
                   CASE WHEN cohort_min IS NOT NULL AND cohort_min != 2147483647 THEN 1 ELSE 0 END as ever_treated
            FROM (
                SELECT *,
                       (SELECT MIN({self._cast_col(self.time_col)})
                        FROM {self.table_name} AS p2
                        WHERE p2.{self.treatment_col} = 1 AND p2.{self.unit_col} = p1.{self.unit_col}
                       ) as cohort_min
                FROM {self.table_name} p1
            )
        )
        """
#  retrieve_num_periods_and_cohorts using CTE instead of temp table
        self.num_periods = self.conn.execute(
            f"{self.cohort_cte} SELECT MAX({self.time_col}) FROM cohort_data"
        ).fetchone()[0]
        cohorts = self.conn.execute(
            f"{self.cohort_cte} SELECT DISTINCT cohort FROM cohort_data WHERE cohort IS NOT NULL"
        ).fetchall()
        self.cohorts = [row[0] for row in cohorts]
        # generate_time_dummies
        self.time_dummies = ",\n".join(
            [
                f"CASE WHEN {self.time_col} = {i} THEN 1 ELSE 0 END AS time_{i}"
                for i in range(self.num_periods + 1)
            ]
        )
        # generate cohort dummies
        cohort_intercepts = []
        for cohort in self.cohorts:
            cohort_intercepts.append(
                f"CASE WHEN cohort = {cohort} THEN 1 ELSE 0 END AS cohort_{cohort}"
            )
        self.cohort_intercepts = ",\n".join(cohort_intercepts)

        # generate_treatment_dummies
        treatment_dummies = []
        for cohort in self.cohorts:
            for i in range(self.num_periods + 1):
                treatment_dummies.append(
                    f"""CASE WHEN cohort = {cohort} AND
                        {self.time_col} = {i}
                        {f"AND {self.treatment_col} == 1" if not self.pre_treat_interactions else ""}
                        THEN 1 ELSE 0 END AS treatment_time_{cohort}_{i}"""
                )
        self.treatment_dummies = ",\n".join(treatment_dummies)

        #  create_transformed_query using CTE instead of temp table
        self.design_matrix_cte = f"""
        {self.cohort_cte},
        transformed_panel_data AS (
            SELECT
                p.{self.unit_col},
                p.{self.time_col},
                p.{self.treatment_col},
                p.{self.outcome_var},
                -- Intercept (constant term)
                1 AS intercept,
                -- cohort intercepts
                {self.cohort_intercepts},
                -- Time dummies for each period
                {self.time_dummies},
                -- Treated group interacted with treatment time dummies
                {self.treatment_dummies}
            FROM cohort_data p
        )
        """

    def compress_data(self):
        # Pre-compute RHS columns to avoid repeated string operations
        cohort_cols = [f"cohort_{cohort}" for cohort in self.cohorts]
        time_cols = [f"time_{i}" for i in range(self.num_periods + 1)]
        treatment_cols = [f"treatment_time_{cohort}_{i}" for cohort in self.cohorts for i in range(self.num_periods + 1)]
        
        rhs_cols = ["intercept"] + cohort_cols + time_cols + treatment_cols
        rhs_clause = ", ".join(rhs_cols)
        
        # Use single query with CTE instead of temp table
        self.compression_query = f"""
        {self.design_matrix_cte}
        SELECT
            {rhs_clause},
            COUNT(*) AS count,
            SUM({self.outcome_var}) AS sum_{self.outcome_var}
        FROM transformed_panel_data
        GROUP BY {rhs_clause}
        """
        
        self.df_compressed = self.conn.execute(self.compression_query).fetchdf()
        self.df_compressed[f"mean_{self.outcome_var}"] = (
            self.df_compressed[f"sum_{self.outcome_var}"] / self.df_compressed["count"]
        )
        
        # Store for later use
        self.rhs_cols = rhs_cols

    def collect_data(self, data):
        self._rhs_list = self.rhs_cols
        X = data[self._rhs_list].values
        y = data[f"mean_{self.outcome_var}"].values
        n = data["count"].values

        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        return y, X, n

    def estimate(self):
        y, X, n = self.collect_data(data=self.df_compressed)
        res = wls(X, y, n)
        cohort_names = [x.split("_")[1] for x in self._rhs_list if "cohort_" in x]
        event_study_coefs = {}
        for c in cohort_names:
            offset = res.filter(regex=f"^cohort_{c}", axis=0).values
            event_study_coefs[c] = (
                res.filter(regex=f"treatment_time_{c}_", axis=0) + offset
            )

        return event_study_coefs

    def bootstrap(self):
        # list all clusters
        total_clusters = self.conn.execute(
            f"SELECT COUNT(DISTINCT {self.cluster_col}) FROM transformed_panel_data"
        ).fetchone()[0]
        boot_coefs = {str(cohort): [] for cohort in self.cohorts}
        # bootstrap loop
        for _ in tqdm(range(self.n_bootstraps)):
            resampled_clusters = (
                self.conn.execute(
                    f"SELECT UNNEST(ARRAY(SELECT {self.cluster_col} FROM transformed_panel_data ORDER BY RANDOM() LIMIT {total_clusters}))"
                )
                .fetchdf()
                .values.flatten()
                .tolist()
            )

            self.conn.execute(
                f"""
                CREATE TEMP TABLE resampled_transformed_panel_data AS
                SELECT * FROM transformed_panel_data
                WHERE {self.cluster_col} IN ({", ".join(map(str, resampled_clusters))})
            """
            )

            self.conn.execute(
                f"""
                CREATE TEMP TABLE resampled_compressed_panel_data AS
                SELECT
                    {self.rhs.replace(";", "")},
                    COUNT(*) AS count,
                    SUM({self.outcome_var}) AS sum_{self.outcome_var}
                FROM
                    resampled_transformed_panel_data
                GROUP BY
                    {self.rhs.replace(";", "")}
            """
            )

            df_boot = self.conn.execute(
                "SELECT * FROM resampled_compressed_panel_data"
            ).fetchdf()
            df_boot[f"mean_{self.outcome_var}"] = (
                df_boot[f"sum_{self.outcome_var}"] / df_boot["count"]
            )

            y, X, n = self.collect_data(data=df_boot)
            res = wls(X, y, n)
            cohort_names = [x.split("_")[1] for x in self._rhs_list if "cohort_" in x]
            for c in cohort_names:
                offset = res.filter(regex=f"^cohort_{c}", axis=0).values
                event_study_coefs = (
                    res.filter(regex=f"treatment_time_{c}_", axis=0) + offset
                )
                boot_coefs[c].append(event_study_coefs.values.flatten())

            self.conn.execute("DROP TABLE resampled_transformed_panel_data")
            self.conn.execute("DROP TABLE resampled_compressed_panel_data")
        # Calculate the covariance matrix for each cohort
        bootstrap_cov_matrix = {
            cohort: np.cov(np.array(coefs).T) for cohort, coefs in boot_coefs.items()
        }
        return bootstrap_cov_matrix

    def summary(self) -> dict:
        """Summary of event study regression (overrides the parent class method)

        Returns:
            dict of event study coefficients and their standard errors
        """
        if self.n_bootstraps > 0:
            summary_tables = {}
            for c in self.point_estimate.keys():
                point_estimate = self.point_estimate[c]
                se = np.sqrt(np.diag(self.vcov[c]))
                summary_tables[c] = pd.DataFrame(
                    np.c_[point_estimate, se],
                    columns=["point_estimate", "se"],
                    index=point_estimate.index,
                )
            return summary_tables
        return {"point_estimate": self.point_estimate}
