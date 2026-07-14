"""Non-default benchmark grid for the 0.4.6 efficiency upgrades."""

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from duckreg import duckreg
from duckreg.core.vcov import compute_cluster_scores


def peak_rss_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if value > 10_000_000 else 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=80_000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(2026)
    n = args.rows
    a = rng.integers(0, max(2, n // 20), n)
    b = rng.integers(0, 100, n)
    x = rng.normal(size=n)
    y = 1.8 * x + rng.normal(size=n) + rng.normal(size=a.max() + 1)[a]
    frame = pd.DataFrame({"y": y, "x": x, "a": a, "b": b})
    output = []

    scores = rng.normal(size=(max(n, 200_000), 8))
    ids = rng.integers(0, 2_000, len(scores))
    start = time.perf_counter()
    compute_cluster_scores(scores, ids)
    output.append({"phase": 1, "benchmark": "cluster_scores",
                   "seconds": time.perf_counter() - start, "peak_rss_mb": peak_rss_mb()})

    for backend in ("numpy", "duckdb"):
        start = time.perf_counter()
        model = duckreg(
            "y ~ x | a + b", frame, se_method="none", fitter="numpy",
            demean_backend=backend, compression=-1, threads=args.threads,
        )
        output.append({
            "phase": 2, "benchmark": "map", "backend": backend,
            "threads": args.threads, "compression": model.resolved_compression,
            "seconds": time.perf_counter() - start, "iterations": model.map_iterations,
            "peak_rss_mb": peak_rss_mb(),
        })

    for compression in ("auto", -1):
        start = time.perf_counter()
        model = duckreg(
            "y ~ x", frame, se_method="none", compression=compression,
            threads=args.threads,
        )
        output.append({
            "phase": 1, "benchmark": "automatic_compression",
            "backend": model.resolved_fitter, "threads": args.threads,
            "compression": model.resolved_compression,
            "seconds": time.perf_counter() - start, "peak_rss_mb": peak_rss_mb(),
        })

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
