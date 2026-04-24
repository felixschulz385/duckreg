"""
Orchestrator: submits one sbatch job per benchmark combination, throttling to
MAX_CONCURRENT jobs at a time.  The helper scripts live in the `scripts/`
directory; this program should be run from the benchmarks root (or via the
`scripts/slurm_benchmark.sh` wrapper).

Each invocation generates (or reuses) a *run ID* (YYYYMMDD_HHMMSS) that
organises all artefacts for that run into three sibling directories under the
benchmarks root:

    benchmarks/results/<run_id>/     – per-combination JSON result files
    benchmarks/manifests/<run_id>/   – manifest.csv tracking submitted jobs
    benchmarks/logs/<run_id>/        – SLURM .log / .err files

After all jobs finish, the orchestrator calls `scripts/collect_results.py`
to merge JSON results into a single CSV.

Usage:
    python orchestrate.py [--run-id ID] [--max-concurrent 4]
    python orchestrate.py --collect-only --run-id 20260305_143022
    python orchestrate.py --dry-run
"""
import argparse
import csv
import datetime
import subprocess
import time
from itertools import product
from pathlib import Path

# ---------------------------------------------------------------------------
# Benchmark grid
# ---------------------------------------------------------------------------

GRID = dict(
    # Three log-spaced N values; extend to 100 M for a larger stress test.
    N_values     = [100_000, 1_000_000, 10_000_000, 100_000_000],
    K_values     = [5, 10, 20],
    # Three nFE breakpoints including a large 100 K level.
    nFE_values   = [(100, 100), (10_000, 10_000), (100_000, 100_000)],
    model_types  = ["pooled", "fe", "iv"],
    packages     = ["duckreg", "pyfixest"],
    # Both HC1 (heteroskedasticity-robust) and CRV1 (cluster-robust on fe1).
    vcov_types   = ["HC1", "CRV1"],
    # Thread counts: multi-thread only meaningful for duckreg (DuckDB).
    threads_values = [1, 4],
)

# Memory limit forwarded to DuckDB for every duckreg call.
DUCKREG_MEMORY_LIMIT = "16GB"


def all_combinations(grid):
    """Yield one dict per benchmark run."""
    for N, K, (nFE1, nFE2), model_type, pkg, vcov, threads in product(
        grid["N_values"],
        grid["K_values"],
        grid["nFE_values"],
        grid["model_types"],
        grid["packages"],
        grid["vcov_types"],
        grid["threads_values"],
    ):
        yield dict(package=pkg, model_type=model_type,
                   N=N, K=K, nFE1=nFE1, nFE2=nFE2, vcov=vcov, threads=threads)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# `scripts` folder is not the user-facing root; define ROOT_DIR as
# the parent of this file so that outputs live in benchmarks/ not scripts/.
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
SLURM_SCRIPT = SCRIPT_DIR / "run_single_benchmark.sh"

MANIFEST_FIELDS = ["package", "model_type", "N", "K", "nFE1", "nFE2", "vcov", "threads", "job_id"]


def make_run_id() -> str:
    """Return a timestamp-based run ID like ``20260305_143022``."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def result_json(results_path: Path, pkg, model_type, N, K, nFE1, vcov, threads):
    return results_path / f"{pkg}_{model_type}_N{N}_K{K}_nFE{nFE1}_{vcov}_t{threads}.json"


def load_manifest(manifests_path: Path):
    """Return {(pkg, model_type, N, K, nFE1, vcov, threads): job_id} for submitted jobs."""
    manifest_csv = manifests_path / "manifest.csv"
    if not manifest_csv.exists():
        return {}
    with open(manifest_csv) as f:
        reader = csv.DictReader(f)
        return {
            (r["package"], r["model_type"],
             int(r["N"]), int(r["K"]), int(r["nFE1"]),
             r.get("vcov", "HC1"), int(r.get("threads", 1))): r["job_id"]
            for r in reader
        }


def append_manifest(manifests_path: Path, row: dict):
    manifest_csv = manifests_path / "manifest.csv"
    write_header = not manifest_csv.exists()
    with open(manifest_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def count_running_jobs(job_name="dr_bench", user="schulz0022"):
    """Count jobs currently queued/running for this user+name via squeue."""
    proc = subprocess.run(
        ["squeue", "-u", user, "-n", job_name, "-h"],
        capture_output=True, text=True,
    )
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    return len(lines)


def submit_job(combo: dict, results_path: Path, logs_path: Path) -> str:
    """Submit one sbatch job; return the SLURM job ID.

    The SLURM log/err paths and cpus-per-task are injected on the sbatch
    command line so they land under ``logs/<run_id>/`` and honour the thread
    count without editing the script header.
    """
    threads = combo.get("threads", 1)
    cmd = [
        "sbatch",
        f"--output={logs_path}/slurm-%j.log",
        f"--error={logs_path}/slurm-%j.err",
        f"--cpus-per-task={threads}",
        str(SLURM_SCRIPT),
        combo["package"],
        combo["model_type"],
        str(combo["N"]),
        str(combo["K"]),
        str(combo["nFE1"]),
        str(combo["nFE2"]),
        str(results_path),
        combo["vcov"],
        DUCKREG_MEMORY_LIMIT,
        str(threads),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT_DIR))
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed: {proc.stderr.strip()}")
    # "Submitted batch job 12345"
    job_id = proc.stdout.strip().split()[-1]
    return job_id


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def orchestrate(run_id: str, max_concurrent: int, dry_run: bool,
                results_base: str = "results",
                manifests_base: str = "manifests",
                logs_base: str = "logs"):
    results_path   = ROOT_DIR / results_base   / run_id
    manifests_path = ROOT_DIR / manifests_base / run_id
    logs_path      = ROOT_DIR / logs_base      / run_id

    results_path.mkdir(parents=True, exist_ok=True)
    manifests_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)

    print(f"Run ID       : {run_id}")
    print(f"Results      : {results_path}")
    print(f"Manifests    : {manifests_path}")
    print(f"Logs         : {logs_path}")

    manifest = load_manifest(manifests_path)
    combos   = list(all_combinations(GRID))

    pending = []
    skipped = 0
    for c in combos:
        key  = (c["package"], c["model_type"], c["N"], c["K"], c["nFE1"],
                c["vcov"], c.get("threads", 1))
        rjson = result_json(results_path, c["package"], c["model_type"],
                            c["N"], c["K"], c["nFE1"], c["vcov"], c.get("threads", 1))
        if rjson.exists():
            skipped += 1
            continue  # already completed
        if key in manifest:
            skipped += 1
            continue  # already submitted (may still be running)
        pending.append(c)

    print(f"Total combos : {len(combos)}")
    print(f"Already done : {skipped}")
    print(f"To submit    : {len(pending)}")

    if not pending:
        print("Nothing to submit.")
        return

    if dry_run:
        for c in pending:
            print(f"  [dry-run] would submit: {c}")
        return

    submitted = 0
    for c in pending:
        # Throttle
        while True:
            running = count_running_jobs()
            if running < max_concurrent:
                break
            print(f"  {running} jobs running (max {max_concurrent}), waiting 30s …")
            time.sleep(30)

        job_id = submit_job(c, results_path, logs_path)
        append_manifest(manifests_path, {**c, "job_id": job_id})
        submitted += 1
        print(f"  [{submitted}/{len(pending)}] submitted job {job_id}: "
              f"{c['package']} {c['model_type']} N={c['N']} K={c['K']} "
              f"nFE={c['nFE1']} vcov={c['vcov']} threads={c.get('threads', 1)}")

    print(f"\nAll {submitted} jobs submitted.  Waiting for completion …")

    # Poll until all dr_bench jobs finish
    while True:
        running = count_running_jobs()
        if running == 0:
            break
        print(f"  {running} jobs still running, checking again in 60s …")
        time.sleep(60)

    print("All jobs finished.  Collecting results …")
    collect_proc = subprocess.run(
        [
            "python", "collect_results.py",
            "--run-id",        run_id,
            "--results-base",  results_base,
            "--manifests-base", manifests_base,
            "--logs-base",     logs_base,
        ],
        cwd=str(SCRIPT_DIR),
    )
    if collect_proc.returncode != 0:
        print("collect_results.py exited with an error — check logs.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate benchmark SLURM jobs")
    parser.add_argument("--run-id",          default=None,
                        help="Run ID to use/resume (default: new timestamp ID)")
    parser.add_argument("--results-base",    default="results",
                        help="Base directory for result JSON files (default: results)")
    parser.add_argument("--manifests-base",  default="manifests",
                        help="Base directory for manifest CSVs (default: manifests)")
    parser.add_argument("--logs-base",       default="logs",
                        help="Base directory for SLURM logs (default: logs)")
    parser.add_argument("--max-concurrent",  type=int, default=4)
    parser.add_argument("--dry-run",         action="store_true",
                        help="Print what would be submitted without actually submitting")
    parser.add_argument("--collect-only",    action="store_true",
                        help="Skip submission; just collect existing results")
    args = parser.parse_args()

    run_id = args.run_id or make_run_id()
    print(f"Using run ID: {run_id}")

    if args.collect_only:
        subprocess.run(
            [
                "python", "collect_results.py",
                "--run-id",        run_id,
                "--results-base",  args.results_base,
                "--manifests-base", args.manifests_base,
                "--logs-base",     args.logs_base,
            ],
            cwd=str(SCRIPT_DIR),
        )
    else:
        orchestrate(
            run_id,
            max_concurrent=args.max_concurrent,
            dry_run=args.dry_run,
            results_base=args.results_base,
            manifests_base=args.manifests_base,
            logs_base=args.logs_base,
        )
