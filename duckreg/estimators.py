import numpy as np
import pandas as pd
from tqdm import tqdm
from .demean import demean, _convert_to_int
from .duckreg import DuckReg, wls

################################################################################


class DuckRegression(DuckReg):
    def __init__(
        self,
        db_name: str,
        table_name: str,
        outcome_vars: list,  # changed from formula to explicit list
        covariates: list,  # new argument
        seed: int,
        fe_cols: list = None,  # new argument (optional fixed effects)
        n_bootstraps: int = 100,
        cluster_col: str = None,  # changed from required to optional
        rowid_col: str = "rowid",
        fitter: str = "numpy",
        round_strata: int = None,
        duckdb_kwargs: dict = None,
        subset: str = None,
        **kwargs,
    ):
        super().__init__(
            db_name=db_name,
            table_name=table_name,
            seed=seed,
            n_bootstraps=n_bootstraps,
            fitter=fitter,
            round_strata=round_strata,
            duckdb_kwargs=duckdb_kwargs,
            **kwargs,
        )
        self.outcome_vars = outcome_vars if isinstance(outcome_vars, list) else [outcome_vars]
        self.covariates = covariates if isinstance(covariates, list) else [covariates]
        self.fe_cols = fe_cols if fe_cols else []
        self.cluster_col = cluster_col
        self.rowid_col = rowid_col
        self.round_strata = round_strata
        self.subset = subset
        
        # Build strata columns for grouping
        self.strata_cols = self.covariates + self.fe_cols

        if not self.outcome_vars:
            raise ValueError("No outcome variables provided")

    def prepare_data(self):
        # No preparation needed for simple regression
        pass

    def compress_data(self):
        # Build SELECT and GROUP BY clauses for strata, optionally rounding columns
        if self.round_strata is not None:
            select_strata = ", ".join(
                [f"ROUND({col}, {self.round_strata}) AS {col}" for col in self.strata_cols]
            )
            group_by_clause = ", ".join(
                [f"ROUND({col}, {self.round_strata})" for col in self.strata_cols]
            )
        else:
            select_strata = ", ".join(self.strata_cols)
            group_by_clause = ", ".join(self.strata_cols)

        # Build aggregation expressions
        agg_parts = ["COUNT(*) as count"]
        sum_expressions = []
        sum_sq_expressions = []
        
        for var in self.outcome_vars:
            sum_expr = f"SUM({var}) as sum_{var}"
            sum_sq_expr = f"SUM(POW({var}, 2)) as sum_{var}_sq"
            sum_expressions.append(sum_expr)
            sum_sq_expressions.append(sum_sq_expr)
        
        # Single join operation instead of multiple concatenations
        all_agg_expressions = ", ".join(agg_parts + sum_expressions + sum_sq_expressions)
        
        na_filter_cols = self.strata_cols + self.outcome_vars
        
        # Add WHERE clause if subset is provided
        where_clause = f"WHERE COLUMNS({na_filter_cols}) IS NOT NULL"
        where_clause += f" AND {self.subset}" if self.subset else ""
        
        self.agg_query = f"""
        SELECT {select_strata}, {all_agg_expressions}
        FROM {self.table_name}
        {where_clause}
        GROUP BY {group_by_clause}
        """
        
        self.df_compressed = self.conn.execute(self.agg_query).fetchdf()
        
        # Drops NAs. Unproblematic because NA either in strata or cases when all values per strata are NA
        self.df_compressed.dropna(inplace=True)
        
        # Pre-compute column lists
        sum_cols = [f"sum_{var}" for var in self.outcome_vars]
        sum_sq_cols = [f"sum_{var}_sq" for var in self.outcome_vars]
        
        self.df_compressed.columns = self.strata_cols + ["count"] + sum_cols + sum_sq_cols
        
        # Single eval operation for all means
        mean_expressions = [f"mean_{var} = sum_{var}/count" for var in self.outcome_vars]
        if mean_expressions:
            self.df_compressed.eval("\n".join(mean_expressions), inplace=True)

    def collect_data(self, data: pd.DataFrame) -> pd.DataFrame:
        y = data.filter(
            regex=f"mean_{'(' + '|'.join(self.outcome_vars) + ')'}", axis=1
        ).values
        X = data[self.covariates].values
        n = data["count"].values

        # y, X, w need to be two-dimensional for the demean function
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X

        if self.fe_cols:
            # fe needs to contain of only integers for
            # the demean function to work
            fe = _convert_to_int(data[self.fe_cols])
            fe = fe.reshape(-1, 1) if fe.ndim == 1 else fe

            y, _ = demean(x=y, flist=fe, weights=n)
            X, _ = demean(x=X, flist=fe, weights=n)
        else:
            X = np.c_[np.ones(X.shape[0]), X]

        return y, X, n

    def estimate(self):
        y, X, n = self.collect_data(data=self.df_compressed)
        betahat = wls(X, y, n).flatten()
        return betahat

    def fit_vcov(self):
        """compressed estimation of the heteroskedasticity-robust variance covariance matrix"""
        self.se = "hc1"
        y, X, n = self.collect_data(data=self.df_compressed)
        betahat = wls(X, y, n).flatten()
        # only works for single outcome for now
        self.n_bootstraps = 0  # disable bootstrap
        yprime = self.df_compressed[f"sum_{self.outcome_vars[0]}"].values.reshape(-1, 1)
        yprimeprime = self.df_compressed[
            f"sum_{self.outcome_vars[0]}_sq"
        ].values.reshape(-1, 1)
        yhat = (X @ betahat).reshape(-1, 1)
        rss_g = (yhat**2) * n.reshape(-1, 1) - 2 * yhat * yprime + yprimeprime
        bread = np.linalg.inv(X.T @ np.diag(n.flatten()) @ X)
        meat = X.T @ np.diag(rss_g.flatten()) @ X
        n_nk = n.sum() / (n.sum() - X.shape[1])
        self.vcov = n_nk * (bread @ meat @ bread)

    def bootstrap(self):
        self.se = "bootstrap"
        if self.fe_cols:
            boot_coefs = np.zeros(
                (self.n_bootstraps, len(self.covariates) * len(self.outcome_vars))
            )
        else:
            boot_coefs = np.zeros(
                (
                    self.n_bootstraps,
                    (len(self.covariates) + 1) * len(self.outcome_vars),
                )
            )
        
        boot_sizes = np.zeros(self.n_bootstraps)

        if not self.cluster_col:
            # IID bootstrap - resample compressed dataframe by integer index
            # Pre-compute y, X for all rows of compressed data
            y, X, n = self.collect_data(data=self.df_compressed)
            n_rows = len(self.df_compressed)
            
            print(f"Starting bootstrap with {self.n_bootstraps} iterations")
            for b in tqdm(range(self.n_bootstraps)):
                # Resample row indices with replacement
                resampled_indices = self.rng.choice(n_rows, size=n_rows, replace=True)
                
                # Count how many times each row was sampled
                row_counts = np.bincount(resampled_indices, minlength=n_rows)
                
                # Scale the weights for each row based on resampling
                n_boot = n * row_counts
                boot_sizes[b] = n_boot.sum()

                boot_coefs[b, :] = wls(X, y, n_boot).flatten()
        else:
            # Cluster bootstrap
            # prepare strata SELECT/GROUP BY with optional rounding
            if self.round_strata is not None:
                select_strata = ", ".join(
                    [f"ROUND({col}, {self.round_strata}) AS {col}" for col in self.strata_cols]
                )
                group_by_clause = ", ".join(
                    [f"ROUND({col}, {self.round_strata})" for col in self.strata_cols]
                )
            else:
                select_strata = ", ".join(self.strata_cols)
                group_by_clause = ", ".join(self.strata_cols)

            where_clause = f" AND {self.subset}" if self.subset else ""
            self.bootstrap_query = f"""
            SELECT {select_strata}, {self.cluster_col}, {", ".join(["COUNT(*) as count"] + [f"SUM({var}) as sum_{var}" for var in self.outcome_vars])}
            FROM {self.table_name}
            WHERE 1=1 {where_clause}
            GROUP BY {group_by_clause}, {self.cluster_col}
            """
            
            # Fetch data once with cluster information
            df_clusters = self.conn.execute(self.bootstrap_query).fetchdf()
            df_clusters.columns = (
                self.strata_cols
                + [self.cluster_col]
                + ["count"]
                + [f"sum_{var}" for var in self.outcome_vars]
            )
            
            # Create mean columns
            create_means = "\n".join(
                [f"mean_{var} = sum_{var}/count" for var in self.outcome_vars]
            )
            df_clusters.eval(create_means, inplace=True)
            
            # Get unique clusters and create mapping
            unique_groups = df_clusters[self.cluster_col].unique()
            group_to_idx = {int(x): i for i, x in enumerate(unique_groups)}
            n_unique_groups = unique_groups.shape[0]
            
            # Pre-compute y, X for all rows
            y, X, n = self.collect_data(data=df_clusters)
            
            # Create group index array
            group_idx = df_clusters[self.cluster_col].map(group_to_idx).to_numpy(dtype=int)

            print(f"Starting bootstrap with {self.n_bootstraps} iterations")
            for b in tqdm(range(self.n_bootstraps)):
                # Resample cluster IDs
                resampled_group_ids = self.rng.choice(
                    n_unique_groups, size=n_unique_groups, replace=True
                )
                
                # Count how many times each cluster was sampled
                bootstrap_scale = np.bincount(resampled_group_ids, minlength=n_unique_groups)
                
                # Scale the weights for each row based on cluster resampling
                row_scales = bootstrap_scale[group_idx]
                n_boot = n * row_scales
                boot_sizes[b] = n_boot.sum()

                boot_coefs[b, :] = wls(X, y, n_boot).flatten()

        # else np.diag() fails if input is not at least 1-dim
        vcov = np.cov(boot_coefs.T, aweights=boot_sizes.T)
        vcov = np.expand_dims(vcov, axis=0) if vcov.ndim == 0 else vcov

        return vcov

    def summary(
        self,
    ):  # ovveride the summary method to include the heteroskedasticity-robust variance covariance matrix when available
        if self.n_bootstraps > 0 or (hasattr(self, "se") and self.se == "hc1"):
            return {
                "point_estimate": self.point_estimate,
                "standard_error": np.sqrt(np.diag(self.vcov)),
            }
        return {"point_estimate": self.point_estimate}


################################################################################


class DuckMundlak(DuckReg):
    def __init__(
        self,
        db_name: str,
        table_name: str,
        outcome_vars: list,  # changed to support multiple outcomes
        covariates: list,
        seed: int,
        fe_cols: list,
        n_bootstraps: int = 100,
        cluster_col: str = None,
        duckdb_kwargs: dict = None,
        subset: str = None,
        **kwargs,
    ):
        super().__init__(
            db_name=db_name,
            table_name=table_name,
            seed=seed,
            n_bootstraps=n_bootstraps,
            duckdb_kwargs=duckdb_kwargs,
            **kwargs,
        )
        self.outcome_vars = outcome_vars if isinstance(outcome_vars, list) else [outcome_vars]
        self.covariates = covariates if isinstance(covariates, list) else [covariates]
        self.fe_cols = fe_cols if isinstance(fe_cols, list) else [fe_cols]
        self.cluster_col = cluster_col
        self.subset = subset

    def prepare_data(self):
        # Compute averages for each FE dimension and store as temp tables
        where_clause = f"WHERE {self.subset}" if self.subset else ""
        
        for i, fe_col in enumerate(self.fe_cols):
            print(f"Computing averages for fixed effect dimension {i+1}/{len(self.fe_cols)}: {fe_col}")
            avg_table_name = f"fe_{i}_avgs"
            avg_query = f"""
            CREATE OR REPLACE TABLE {avg_table_name} AS
            SELECT {fe_col},
                   {", ".join([f"AVG({cov}) AS avg_{cov}_fe{i}" for cov in self.covariates])}
            FROM {self.table_name}
            {where_clause}
            GROUP BY {fe_col}
            """
            self.conn.execute(avg_query)

        # Build list of columns to select
        fe_cols_select = ", ".join([f"{fe_col}" for fe_col in self.fe_cols])
        covariates_select = ", ".join([f"{cov}" for cov in self.covariates])
        outcome_vars_select = ", ".join([f"{var}" for var in self.outcome_vars])
        
        # Build nested SELECT with pairwise joins
        # Start with the base table, selecting only necessary columns
        nested_query = f"""
        SELECT {fe_cols_select}, {outcome_vars_select}, {covariates_select}, {self.cluster_col}
        FROM {self.table_name}
        {where_clause}
        """
        
        # Add each FE average table one at a time using nested SELECTs
        for i, fe_col in enumerate(self.fe_cols):
            nested_query = f"""
            SELECT *
            FROM ({nested_query})
            JOIN fe_{i}_avgs fe{i}
            USING ({fe_col})
            """
        
        # Create the final design matrix
        print("Creating design matrix with all fixed effect averages")
        self.design_matrix_query = f"""
        CREATE OR REPLACE TABLE design_matrix AS
        {nested_query}
        """
        self.conn.execute(self.design_matrix_query)

    def compress_data(self):
        # Pre-compute column lists
        cov_cols = [f"{cov}" for cov in self.covariates]
        avg_cols = []
        for i in range(len(self.fe_cols)):
            avg_cols.extend([f"avg_{cov}_fe{i}" for cov in self.covariates])
        cluster_cols = [self.cluster_col]
        
        # Use consistent naming: strata_cols
        self.strata_cols = cov_cols + avg_cols + cluster_cols
        
        # Build SELECT and GROUP BY columns
        # prepare strata SELECT/GROUP BY with optional rounding
        if self.round_strata is not None:
            select_clause = ", ".join(
                [f"ROUND({col}, {self.round_strata}) AS {col}" for col in self.strata_cols]
            )
            group_by_clause = ", ".join(
                [f"ROUND({col}, {self.round_strata})" for col in self.strata_cols]
            )
        else:
            select_clause = ", ".join(self.strata_cols)
            group_by_clause = ", ".join(self.strata_cols)
        
        ## Large DF split by clusters
        
        # Build aggregation for all outcome variables
        outcome_aggs = []
        for var in self.outcome_vars:
            outcome_aggs.append(f"SUM({var}) as sum_{var}")
        outcome_aggs_clause = ", ".join(outcome_aggs)
        
        na_filter_cols = self.strata_cols + self.outcome_vars
        where_clause = f"WHERE COLUMNS({', '.join(na_filter_cols)}) IS NOT NULL"
        where_clause += f" AND {self.subset}" if self.subset else ""
        
        # Use consistent naming: agg_query
        self.agg_query = f"""
        SELECT
            {select_clause},
            COUNT(*) as count,
            {outcome_aggs_clause}
        FROM design_matrix
        {where_clause}
        GROUP BY {group_by_clause}
        """
        print("Compressing data by computing group-level statistics")
        self.df_compressed_clusters = self.conn.execute(self.agg_query).fetchdf()
        
        ## Further compress, dropping cluster info for main regression
        agg_dict = {"count": "sum"}
        for var in self.outcome_vars:
            agg_dict[f"sum_{var}"] = "sum"
        
        self.df_compressed = self.df_compressed_clusters.\
            groupby(cov_cols + avg_cols, as_index=False).\
                agg(agg_dict)

        # Create mean columns for all outcome variables
        for var in self.outcome_vars:
            self.df_compressed_clusters[f"mean_{var}"] = (
                self.df_compressed_clusters[f"sum_{var}"] / self.df_compressed_clusters["count"]
            )
            self.df_compressed[f"mean_{var}"] = (
                self.df_compressed[f"sum_{var}"] / self.df_compressed["count"]
            )

    def collect_data(self, data: pd.DataFrame):
        # Build list of RHS variables (covariates + all FE averages)
        self.rhs = self.covariates.copy()
        for i in range(len(self.fe_cols)):
            self.rhs.extend([f"avg_{cov}_fe{i}" for cov in self.covariates])

        X = data[self.rhs].values
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Collect all outcome variables
        y_cols = [f"mean_{var}" for var in self.outcome_vars]
        y = data[y_cols].values
        n = data["count"].values

        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X

        return y, X, n

    def estimate(self):
        y, X, n = self.collect_data(data=self.df_compressed)
        return wls(X, y, n)

    def bootstrap(self):
        # Build list of RHS variables
        rhs = self.covariates.copy()
        for i in range(len(self.fe_cols)):
            rhs.extend([f"avg_{cov}_fe{i}" for cov in self.covariates])
        
        # Adjust for multiple outcomes
        n_coefs = (len(rhs) + 1) * len(self.outcome_vars)
        boot_coefs = np.zeros((self.n_bootstraps, n_coefs))
        boot_sizes = np.zeros(self.n_bootstraps)

        # Bootstrap in blocks of first FE if cluster_col not given
        if self.cluster_col is None:
            self.cluster_col = self.fe_cols[0]
            
        # Get the unique groups
        unique_groups = self.df_compressed_clusters[self.cluster_col].unique()
        group_to_idx = {int(x): i for i, x in enumerate(unique_groups)}
        n_unique_groups = unique_groups.shape[0]
        
        resampled_group_ids = self.rng.choice(
                n_unique_groups, size=n_unique_groups, replace=True
        )
        
        y, X, n = self.collect_data(data=self.df_compressed_clusters)
        
        group_idx = self.df_compressed_clusters[self.cluster_col].map(group_to_idx).to_numpy(dtype=int)

        print(f"Starting bootstrap with {self.n_bootstraps} iterations")
        for b in tqdm(range(self.n_bootstraps), desc="Bootstrap iterations"):
            
            resampled_group_ids = self.rng.choice(
                n_unique_groups, size=n_unique_groups, replace=True
            )
            bootstrap_scale = np.bincount(resampled_group_ids, minlength=n_unique_groups)
            row_scales = bootstrap_scale[group_idx]
            
            n_boot = n * row_scales
            boot_sizes[b] = n_boot.sum()

            boot_coefs[b, :] = wls(X, y, n_boot).flatten()
            
        return np.cov(boot_coefs.T, aweights = boot_sizes.T)


################################################################################
class DuckMundlakEventStudy(DuckReg):
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
        duckdb_kwargs: dict = None,  # renamed parameter
        **kwargs,
    ):
        super().__init__(
            db_name=db_name,
            table_name=table_name,
            n_bootstraps=n_bootstraps,
            duckdb_kwargs=duckdb_kwargs,  # pass to parent
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
                       (SELECT MIN({self.time_col})
                        FROM {self.table_name} AS p2
                        WHERE p2.{self.unit_col} = p1.{self.unit_col} AND p2.{self.treatment_col} = 1
                       ) as cohort_min
                FROM {self.table_name} p1
            )
        )
        """
        #  retrieve_num_periods_and_cohorts using CTE
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
        coef = wls(X, y, n)
        res = pd.DataFrame(
            {
                "est": coef.squeeze(),
            },
            index=self._rhs_list,
        )
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
            coef = wls(X, y, n)
            res = pd.DataFrame(
                {
                    "est": coef.squeeze(),
                },
                index=self._rhs_list,
            )
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


################################################################################
class DuckDoubleDemeaning(DuckReg):
    def __init__(
        self,
        db_name: str,
        table_name: str,
        outcome_var: str,
        treatment_var: str,
        fe_cols: list,  # changed from unit_col and time_col to list of FE columns
        seed: int,
        n_bootstraps: int = 100,
        cluster_col: str = None,
        duckdb_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(
            db_name=db_name,
            table_name=table_name,
            seed=seed,
            n_bootstraps=n_bootstraps,
            duckdb_kwargs=duckdb_kwargs,
            **kwargs,
        )
        self.outcome_var = outcome_var
        self.treatment_var = treatment_var
        self.fe_cols = fe_cols  # list of FE dimensions (e.g., ['unit_id', 'time_col'])
        self.cluster_col = cluster_col

    def prepare_data(self):
        # Compute overall mean
        self.overall_mean_query = f"""
        CREATE TEMP TABLE overall_mean AS
        SELECT AVG({self.treatment_var}) AS mean_{self.treatment_var}
        FROM {self.table_name}
        """
        self.conn.execute(self.overall_mean_query)

        # Compute means for each FE dimension
        for i, fe_col in enumerate(self.fe_cols):
            mean_table_name = f"fe_{i}_means"
            mean_query = f"""
            CREATE TEMP TABLE {mean_table_name} AS
            SELECT {fe_col}, AVG({self.treatment_var}) AS mean_{self.treatment_var}_fe{i}
            FROM {self.table_name}
            GROUP BY {fe_col}
            """
            self.conn.execute(mean_query)

        # Create multi-way demeaned variables
        join_clauses = []
        demean_terms = []
        
        for i, fe_col in enumerate(self.fe_cols):
            join_clauses.append(f"JOIN fe_{i}_means fe{i} ON t.{fe_col} = fe{i}.{fe_col}")
            demean_terms.append(f"fe{i}.mean_{self.treatment_var}_fe{i}")
        
        # Formula: X_ddot = X - sum(FE_means) + (k-1) * overall_mean
        # where k is the number of FE dimensions
        k = len(self.fe_cols)
        demean_formula = f"t.{self.treatment_var} - {' - '.join(demean_terms)} + {k-1} * om.mean_{self.treatment_var}"
        
        self.double_demean_query = f"""
        CREATE TEMP TABLE multi_demeaned AS
        SELECT
            {", ".join([f"t.{fe_col}" for fe_col in self.fe_cols])},
            t.{self.outcome_var},
            {demean_formula} AS ddot_{self.treatment_var}
        FROM {self.table_name} t
        {" ".join(join_clauses)}
        CROSS JOIN overall_mean om
        """
        self.conn.execute(self.double_demean_query)

    def compress_data(self):
        self.compress_query = f"""
        SELECT
            ddot_{self.treatment_var},
            COUNT(*) as count,
            SUM({self.outcome_var}) as sum_{self.outcome_var}
        FROM multi_demeaned
        GROUP BY ddot_{self.treatment_var}
        """
        self.df_compressed = self.conn.execute(self.compress_query).fetchdf()
        self.df_compressed[f"mean_{self.outcome_var}"] = (
            self.df_compressed[f"sum_{self.outcome_var}"] / self.df_compressed["count"]
        )

    def collect_data(self, data: pd.DataFrame):
        X = data[f"ddot_{self.treatment_var}"].values
        X = np.c_[np.ones(X.shape[0]), X]
        y = data[f"mean_{self.outcome_var}"].values
        n = data["count"].values
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X

        return y, X, n

    def estimate(self):
        y, X, n = self.collect_data(data=self.df_compressed)
        return wls(X, y, n)

    def bootstrap(self):
        boot_coefs = np.zeros((self.n_bootstraps, 2))  # Intercept and treatment effect

        if self.cluster_col is None:
            # Bootstrap on first FE dimension
            total_units = self.conn.execute(
                f"SELECT COUNT(DISTINCT {self.fe_cols[0]}) FROM {self.table_name}"
            ).fetchone()[0]
            self.bootstrap_query = f"""
            SELECT
                ddot_{self.treatment_var},
                COUNT(*) as count,
                SUM({self.outcome_var}) as sum_{self.outcome_var}
            FROM multi_demeaned
            WHERE {self.fe_cols[0]} IN (SELECT unnest((?)))
            GROUP BY ddot_{self.treatment_var}
            """
            total_samples = total_units
        else:
            # Cluster bootstrap
            total_clusters = self.conn.execute(
                f"SELECT COUNT(DISTINCT {self.cluster_col}) FROM {self.table_name}"
            ).fetchone()[0]
            self.bootstrap_query = f"""
            SELECT
                ddot_{self.treatment_var},
                COUNT(*) as count,
                SUM({self.outcome_var}) as sum_{self.outcome_var}
            FROM multi_demeaned
            WHERE {self.cluster_col} IN (SELECT unnest((?)))
            GROUP BY ddot_{self.treatment_var}
            """
            total_samples = total_clusters

        for b in tqdm(range(self.n_bootstraps)):
            resampled_samples = self.rng.choice(
                total_samples, size=total_samples, replace=True
            )

            df_boot = self.conn.execute(
                self.bootstrap_query, [resampled_samples.tolist()]
            ).fetchdf()
            df_boot[f"mean_{self.outcome_var}"] = (
                df_boot[f"sum_{self.outcome_var}"] / df_boot["count"]
            )

            y, X, n = self.collect_data(data=df_boot)

            boot_coefs[b, :] = wls(X, y, n).flatten()

        return np.cov(boot_coefs.T)


################################################################################
