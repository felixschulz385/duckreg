"""Small MAP convergence-sampling benchmark.

This script compares exact convergence checks with deterministic FE-group
sampled checks on a modest synthetic two-way FE problem. It is intentionally
small enough to run locally:

    python benchmarks/scripts/map_sampling_benchmark.py
"""

import time
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from duckreg.core.transformers import IterativeDemeanTransformer


def make_panel(n=80_000, k=5, n_fe1=2_000, n_fe2=500, seed=42):
    rng = np.random.default_rng(seed)
    fe1 = rng.integers(0, n_fe1, size=n)
    fe2 = rng.integers(0, n_fe2, size=n)
    x = rng.normal(size=(n, k))
    y = (
        x @ rng.normal(size=k)
        + rng.normal(size=n)
        + rng.normal(size=n_fe1)[fe1]
        + rng.normal(size=n_fe2)[fe2]
    )

    data = {"fe1": fe1, "fe2": fe2, "y": y}
    for idx in range(k):
        data[f"x{idx}"] = x[:, idx]
    return pd.DataFrame(data)


def run_once(df, convergence_sample):
    conn = duckdb.connect()
    conn.register("panel", df)
    variables = ["y"] + [col for col in df.columns if col.startswith("x")]
    transformer = IterativeDemeanTransformer(
        conn=conn,
        table_name="panel",
        fe_cols=["fe1", "fe2"],
        remove_singletons=False,
        tolerance=1e-6,
        check_interval=5,
        convergence_sample=convergence_sample,
        max_iterations=100,
    )
    started = time.perf_counter()
    transformer.fit_transform(variables)
    elapsed = time.perf_counter() - started
    conn.close()
    return elapsed, transformer.n_iterations


def main():
    df = make_panel()
    for convergence_sample in (1.0, 0.25):
        elapsed, iterations = run_once(df, convergence_sample)
        print(
            f"convergence_sample={convergence_sample:.2f} "
            f"elapsed={elapsed:.3f}s iterations={iterations}"
        )


if __name__ == "__main__":
    main()
