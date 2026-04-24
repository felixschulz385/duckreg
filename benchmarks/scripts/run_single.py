"""
Run a single benchmark combination and write a JSON result file.

This script is normally invoked from `scripts/run_single_benchmark.sh`, which
first changes working directory to the benchmarks root.  The ``--output-dir``
argument must be the **absolute** path to the per-run results directory
(e.g. ``benchmarks/results/<run_id>``).

Usage (called by run_single_benchmark.sh):
    python run_single.py \
        --package duckreg --model-type fe \
        --N 1000000 --K 10 --nFE1 1000 --nFE2 1000 \
        --vcov HC1 --threads 4 \
        --memory-limit 16GB \
        --output-dir /abs/path/to/benchmarks/results/<run_id>
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_regression_data(N, K, nFE1, nFE2, seed=42, has_instrument=False):
    rng = np.random.default_rng(seed)

    fe1 = rng.choice(nFE1, size=N)
    fe2 = rng.choice(nFE2, size=N)
    X = rng.normal(size=(N, K))

    if has_instrument:
        Z = rng.normal(size=N)
        D = 0.5 * Z + rng.normal(size=N)
    else:
        D = rng.normal(size=N)
        Z = None

    fe1_effects = rng.normal(0, 1, nFE1)
    fe2_effects = rng.normal(0, 1, nFE2)
    beta_D = 1.5
    beta_X = rng.uniform(0.5, 2.0, K)

    y = (
        beta_D * D
        + X @ beta_X
        + fe1_effects[fe1]
        + fe2_effects[fe2]
        + rng.normal(size=N)
    )

    data_dict = {"y": y, "D": D, "fe1": fe1, "fe2": fe2}
    for i in range(K):
        data_dict[f"x{i+1}"] = X[:, i]
    if has_instrument:
        data_dict["Z"] = Z

    return pd.DataFrame(data_dict)


# ---------------------------------------------------------------------------
# VCOV helpers
# ---------------------------------------------------------------------------

def duckreg_se_method(vcov: str):
    """Map a vcov label to the duckreg `se_method` argument.

    CRV1 clusters on ``fe1`` (the first fixed-effect dimension).
    """
    if vcov == "CRV1":
        return {"CRV1": "fe1"}
    return vcov  # "HC1" passed as-is


def pyfixest_vcov(vcov: str):
    """Map a vcov label to the pyfixest ``vcov`` argument."""
    if vcov == "HC1":
        return "hetero"
    if vcov == "CRV1":
        return {"CRV1": "fe1"}
    raise ValueError(f"Unknown vcov: {vcov!r}")


# ---------------------------------------------------------------------------
# Individual benchmark functions
# ---------------------------------------------------------------------------


def run_duckreg(df, K, model_type, vcov: str, memory_limit: str, threads: int):
    x_vars = " + ".join([f"x{i+1}" for i in range(K)])

    if model_type == "pooled":
        formula = f"y ~ D + {x_vars}"
    elif model_type == "fe":
        formula = f"y ~ D + {x_vars} | fe1 + fe2"
    else:  # iv
        formula = f"y ~ {x_vars} | fe1 + fe2 | (D ~ Z)"

    from duckreg import duckreg
    t0 = time.perf_counter()
    model = duckreg(
        formula=formula,
        data=df,
        se_method=duckreg_se_method(vcov),
        fe_method="demean",
        fitter="duckdb",
        threads=threads,
        memory_limit=memory_limit,
    )
    model.summary()
    return time.perf_counter() - t0


def run_pyfixest(df, K, model_type, vcov: str, threads: int):
    import numba
    import pyfixest as pf

    # pyfixest's demean / solve kernels use numba parallel=True; honour the
    # requested thread count so the comparison with duckreg is meaningful.
    numba.set_num_threads(threads)

    x_vars = " + ".join([f"x{i+1}" for i in range(K)])

    if model_type == "pooled":
        formula = f"y ~ D + {x_vars}"
    elif model_type == "fe":
        formula = f"y ~ D + {x_vars} | fe1 + fe2"
    else:  # iv
        formula = f"y ~ {x_vars} | fe1 + fe2 | D ~ Z"

    t0 = time.perf_counter()
    model = pf.feols(formula, df, vcov=pyfixest_vcov(vcov))
    model.tidy()
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def result_path(output_dir, package, model_type, N, K, nFE1, vcov, threads):
    return Path(output_dir) / f"{package}_{model_type}_N{N}_K{K}_nFE{nFE1}_{vcov}_t{threads}.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package",      required=True, choices=["duckreg", "pyfixest"])
    parser.add_argument("--model-type",   required=True, choices=["pooled", "fe", "iv"])
    parser.add_argument("--N",            type=int, required=True)
    parser.add_argument("--K",            type=int, required=True)
    parser.add_argument("--nFE1",         type=int, required=True)
    parser.add_argument("--nFE2",         type=int, required=True)
    parser.add_argument("--vcov",         default="HC1", choices=["HC1", "CRV1"])
    parser.add_argument("--threads",      type=int, default=1,
                        help="DuckDB thread count (duckreg only, default: 1)")
    parser.add_argument("--memory-limit", default="16GB",
                        help="DuckDB memory_limit (duckreg only, default: 16GB)")
    parser.add_argument("--output-dir",   default="results")
    args = parser.parse_args()

    job_id    = os.environ.get("SLURM_JOB_ID", "local")
    out_path  = result_path(args.output_dir, args.package, args.model_type,
                            args.N, args.K, args.nFE1, args.vcov, args.threads)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    result = {
        "package":       args.package,
        "model_type":    args.model_type,
        "N":             args.N,
        "K":             args.K,
        "nFE1":          args.nFE1,
        "nFE2":          args.nFE2,
        "vcov":          args.vcov,
        "threads":       args.threads,
        "memory_limit":  args.memory_limit,
        "job_id":        job_id,
        "status":        "failed",
        "time_seconds":  None,
        "error":         None,
    }

    try:
        has_iv = args.model_type == "iv"
        df = generate_regression_data(args.N, args.K, args.nFE1, args.nFE2,
                                      has_instrument=has_iv)
        print(f"Data generated: N={args.N}, K={args.K}, nFE={args.nFE1}, "
              f"model={args.model_type}, pkg={args.package}, vcov={args.vcov}, "
              f"threads={args.threads}")

        if args.package == "duckreg":
            elapsed = run_duckreg(df, args.K, args.model_type,
                                  args.vcov, args.memory_limit, args.threads)
        else:
            elapsed = run_pyfixest(df, args.K, args.model_type, args.vcov, args.threads)

        result["time_seconds"] = round(elapsed, 4)
        result["status"] = "success"
        print(f"Done in {elapsed:.2f}s")

    except Exception as e:
        result["error"] = str(e)
        print(f"ERROR: {e}")

    finally:
        gc.collect()

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result written to {out_path}")


if __name__ == "__main__":
    main()
