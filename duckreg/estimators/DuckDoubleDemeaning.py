import numpy as np
import pandas as pd
from tqdm import tqdm

from ..duckreg import DuckReg
from ..fitters import wls


class DuckDoubleDemeaning(DuckReg):
    def __init__(
        self,

        db_name: str,
        table_name: str,
        outcome_var: str,
        treatment_var: str,
        fe_cols: list,
        seed: int,
        n_bootstraps: int = 100,
        cluster_col: str = None,
        duckdb_kwargs: dict = None,
        variable_casts: dict = None,
        **kwargs,
    ):
        super().__init__(
            db_name=db_name,
            table_name=table_name,
            seed=seed,
            n_bootstraps=n_bootstraps,
            duckdb_kwargs=duckdb_kwargs,
            variable_casts=variable_casts,
            **kwargs,
        )
        self.outcome_var = outcome_var
        self.treatment_var = treatment_var
        self.fe_cols = fe_cols
        self.cluster_col = cluster_col

    def prepare_data(self):
        self.conn.execute(f"""
        CREATE TEMP TABLE overall_mean AS
        SELECT AVG({self._cast_col(self.treatment_var)}) AS mean_{self.treatment_var}
        FROM {self.table_name}
        """)

        for i, fe_col in enumerate(self.fe_cols):
            self.conn.execute(f"""
            CREATE TEMP TABLE fe_{i}_means AS
            SELECT {self._cast_col(fe_col)} AS {fe_col}, AVG({self._cast_col(self.treatment_var)}) AS mean_{self.treatment_var}_fe{i}
            FROM {self.table_name}
            GROUP BY {self._cast_col(fe_col)}
            """)

        join_clauses = [f"JOIN fe_{i}_means fe{i} ON t.{fe_col} = fe{i}.{fe_col}" for i, fe_col in enumerate(self.fe_cols)]
        demean_terms = [f"fe{i}.mean_{self.treatment_var}_fe{i}" for i in range(len(self.fe_cols))]
        demean_formula = f"t.{self._cast_col(self.treatment_var)} - {' - '.join(demean_terms)} + {len(self.fe_cols)-1} * om.mean_{self.treatment_var}"
        
        self.conn.execute(f"""
        CREATE TEMP TABLE multi_demeaned AS
        SELECT {", ".join([f"t.{fe_col}" for fe_col in self.fe_cols])}, t.{self.outcome_var}, {demean_formula} AS ddot_{self.treatment_var}
        FROM {self.table_name} t {" ".join(join_clauses)} CROSS JOIN overall_mean om
        """)

    def compress_data(self):
        self.df_compressed = self.conn.execute(f"""
            SELECT ddot_{self.treatment_var}, COUNT(*) as count, SUM({self.outcome_var}) as sum_{self.outcome_var}
            FROM multi_demeaned GROUP BY ddot_{self.treatment_var}
        """).fetchdf()
        
        self.n_obs = int(self.df_compressed['count'].sum())
        self.df_compressed[f"mean_{self.outcome_var}"] = self.df_compressed[f"sum_{self.outcome_var}"] / self.df_compressed["count"]

    def collect_data(self, data: pd.DataFrame):
        X = np.c_[np.ones(len(data)), data[f"ddot_{self.treatment_var}"].values]
        y = data[f"mean_{self.outcome_var}"].values.reshape(-1, 1)
        n = data["count"].values
        self.coef_names_ = ['Intercept', f'ddot_{self.treatment_var}']
        return y, X, n

    def estimate(self):
        y, X, n = self.collect_data(data=self.df_compressed)
        return wls(X, y, n)

    def bootstrap(self):
        boot_coefs = np.zeros((self.n_bootstraps, 2))

        if self.cluster_col is None:
            total_samples = self.conn.execute(f"SELECT COUNT(DISTINCT {self.fe_cols[0]}) FROM {self.table_name}").fetchone()[0]
            self.bootstrap_query = f"""
            SELECT ddot_{self.treatment_var}, COUNT(*) as count, SUM({self.outcome_var}) as sum_{self.outcome_var}
            FROM multi_demeaned WHERE {self.fe_cols[0]} IN (SELECT unnest((?))) GROUP BY ddot_{self.treatment_var}
            """
        else:
            total_samples = self.conn.execute(f"SELECT COUNT(DISTINCT {self.cluster_col}) FROM {self.table_name}").fetchone()[0]
            self.bootstrap_query = f"""
            SELECT ddot_{self.treatment_var}, COUNT(*) as count, SUM({self.outcome_var}) AS sum_{self.outcome_var}
            FROM multi_demeaned WHERE {self.cluster_col} IN (SELECT unnest((?))) GROUP BY ddot_{self.treatment_var}
            """

        for b in tqdm(range(self.n_bootstraps)):
            resampled_samples = self.rng.choice(total_samples, size=total_samples, replace=True)
            df_boot = self.conn.execute(self.bootstrap_query, [resampled_samples.tolist()]).fetchdf()
            df_boot[f"mean_{self.outcome_var}"] = df_boot[f"sum_{self.outcome_var}"] / df_boot["count"]
            y, X, n = self.collect_data(data=df_boot)
            boot_coefs[b, :] = wls(X, y, n).flatten()

        return np.cov(boot_coefs.T)

    def summary(self) -> dict:
        """Summary of double-demeaning regression"""
        result = {
            "point_estimate": self.point_estimate,
            "coef_names": getattr(self, 'coef_names_', None),
            "n_obs": getattr(self, 'n_obs', None),
            "n_obs_compressed": len(self.df_compressed) if hasattr(self, 'df_compressed') else None,
            "estimator_type": "DuckDoubleDemeaning",
            "fe_method": "double_demean",
            "outcome_var": self.outcome_var,
            "treatment_var": self.treatment_var,
            "fe_cols": self.fe_cols,
            "cluster_col": self.cluster_col
        }
        
        if self.n_bootstraps > 0:
            result.update({
                "standard_error": np.sqrt(np.diag(self.vcov)),
                "vcov": self.vcov,
                "se_type": "bootstrap",
            })
        
        return result