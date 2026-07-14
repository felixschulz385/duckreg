"""Exact in-memory alternating-projection fixed-effect transformer."""

from typing import List, Optional

import numpy as np
import pandas as pd

from .base import FETransformer


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class NumpyDemeanTransformer(FETransformer):
    """Absorb fixed effects with compact codes and vectorized group reductions."""

    _RESULT_TABLE = "_numpy_demeaned_data"

    def __init__(
        self,
        *args,
        carry_cols: Optional[List[str]] = None,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        singleton_pruning: str = "iterative",
        residual_type: str = "DOUBLE",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.carry_cols = list(carry_cols or [])
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.singleton_pruning = singleton_pruning
        self.residual_type = residual_type
        self.n_iterations = 0
        self._resid_name_map = {}
        self._fe_total_levels = 0
        self._frame = None

    def fit_transform(self, variables: List[str], where_clause: str = "") -> str:
        self._resid_name_map = {v: f"_resid_{i}" for i, v in enumerate(variables)}
        selected = list(dict.fromkeys(self.fe_cols + self.carry_cols + variables))
        projection = ", ".join(_q(c) for c in selected)
        frame = self.conn.execute(
            f"SELECT {projection} FROM {self.table_name} {where_clause}"
        ).fetchdf()
        before = len(frame)

        if self.remove_singletons and self.fe_cols:
            while len(frame):
                keep = np.ones(len(frame), dtype=bool)
                for fe in self.fe_cols:
                    keep &= frame.groupby(fe, dropna=False)[fe].transform("size").to_numpy() > 1
                if keep.all():
                    break
                frame = frame.loc[keep].reset_index(drop=True)
                if self.singleton_pruning == "one_pass":
                    break
        self.n_rows_dropped_singletons = before - len(frame)
        self._n_obs = len(frame)

        codes = []
        counts = []
        for fe in self.fe_cols:
            code, levels = pd.factorize(frame[fe], sort=True)
            code = code.astype(np.int32, copy=False)
            codes.append(code)
            counts.append(np.bincount(code, minlength=len(levels)).astype(np.float64))
        self._fe_total_levels = sum(len(c) for c in counts)

        dtype = np.float32 if self.residual_type == "FLOAT" else np.float64
        values = frame[variables].to_numpy(dtype=dtype, copy=True)
        if len(frame) and codes:
            for iteration in range(self.max_iterations):
                for code, count in zip(codes, counts):
                    for j in range(values.shape[1]):
                        sums = np.bincount(code, weights=values[:, j], minlength=len(count))
                        values[:, j] -= (sums / count)[code]
                max_mean = 0.0
                for code, count in zip(codes, counts):
                    for j in range(values.shape[1]):
                        sums = np.bincount(code, weights=values[:, j], minlength=len(count))
                        max_mean = max(max_mean, float(np.max(np.abs(sums / count))))
                self.n_iterations = iteration + 1
                if max_mean <= self.tolerance:
                    break

        for j, variable in enumerate(variables):
            frame[self._resid_name_map[variable]] = values[:, j]
        keep_cols = list(dict.fromkeys(self.fe_cols + self.carry_cols))
        result = frame[keep_cols + list(self._resid_name_map.values())]
        self._frame = result
        try:
            self.conn.unregister(self._RESULT_TABLE)
        except Exception:
            pass
        self.conn.register(self._RESULT_TABLE, result)
        self._fitted = True
        return self._RESULT_TABLE

    def transform_query(self, variables: List[str]) -> str:
        missing = [v for v in variables if v not in self._resid_name_map]
        if missing:
            raise ValueError(f"Unknown transformed variable(s): {missing}")
        return ", ".join(
            f"{_q(self._resid_name_map[v])} AS {_q(v)}" for v in variables
        )

    def residual_column_name(self, variable: str) -> str:
        if variable not in self._resid_name_map:
            raise ValueError(f"Unknown transformed variable: {variable}")
        return self._resid_name_map[variable]

    @property
    def n_obs(self) -> int:
        if self._n_obs is None:
            raise RuntimeError("fit_transform() has not been called")
        return self._n_obs

    @property
    def df_correction(self) -> int:
        return self._fe_total_levels

    @property
    def extra_regressors(self) -> List[str]:
        return []

    @property
    def has_intercept(self) -> bool:
        return False
