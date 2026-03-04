"""
Orchestrator: submits one sbatch job per benchmark combination, throttling to
MAX_CONCURRENT jobs at a time.  The helper scripts live in the `scripts/`
directory; this program should be run from the benchmarks root (or via the
`scripts/slurm_benchmark.sh` wrapper).  Submitted job metadata is stored in
`<results-dir>/manifest.csv` so that multiple output directories may coexist.
After all jobs finish, the orchestrator calls `scripts/collect_results.py`
to merge JSON results.

Usage:
    python orchestrate.py [--results-dir results] [--max-concurrent 4]
    python orchestrate.py --collect-only   # just collect, don't submit
"""
import argparse
import csv
import subprocess
import time
from itertools import product
from pathlib import Path

# ---------------------------------------------------------------------------
# Benchmark grid
# ---------------------------------------------------------------------------

GRID = dict(
    N_values    = [100_000, 1_000_000, 10_000_000, 50_000_000],
    K_values    = [5, 10, 20],
    nFE_values  = [(100, 100), (1_000, 1_000), (10_000, 10_000)],
    model_types = ["pooled", "fe", "iv"],
    packages    = ["duckreg", "pyfixest"],
)


def all_combinations(grid):
    """Yield one dict per benchmark run."""
    for N, K, (nFE1, nFE2), model_type, pkg in product(
        grid["N_values"],
        grid["K_values"],
        grid["nFE_values"],
        grid["model_types"],
        grid["packages"],
    ):
        yield dict(package=pkg, model_type=model_type,
                   N=N, K=K, nFE1=nFE1, nFE2=nFE2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# `scripts` folder is not the user-facing root; define ROOT_DIR as
# the parent of this file so that outputs (results, log) live in
# benchmarks/ rather than benchmarks/scripts.
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
SLURM_SCRIPT = SCRIPT_DIR / "run_single_benchmark.sh"

MANIFEST_FIELDS = ["package", "model_type", "N", "K", "nFE1", "nFE2", "job_id"]


def result_json(results_path: Path, pkg, model_type, N, K, nFE1):
    return results_path / f"{pkg}_{model_type}_N{N}_K{K}_nFE{nFE1}.json"


def load_manifest(results_path: Path):
    """Return {(pkg,model_type,N,K,nFE1): job_id} for already-submitted jobs.
    Manifest lives inside the results directory so that everything related
    to a particular run is kept together.
    """
    manifest_csv = results_path / "manifest.csv"
    if not manifest_csv.exists():
        return {}
    with open(manifest_csv) as f:
        reader = csv.DictReader(f)
        return {
            (r["package"], r["model_type"], int(r["N"]), int(r["K"]), int(r["nFE1"])): r["job_id"]
            for r in reader
        }


def append_manifest(results_path: Path, row: dict):
    manifest_csv = results_path / "manifest.csv"
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


def submit_job(combo: dict, results_dir: str) -> str:
    """Submit one sbatch job; return the SLURM job ID."""
    cmd = [
        "sbatch",
        str(SLURM_SCRIPT),
        combo["package"],
        combo["model_type"],
        str(combo["N"]),
        str(combo["K"]),
        str(combo["nFE1"]),
        str(combo["nFE2"]),
        results_dir,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPT_DIR))
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed: {proc.stderr.strip()}")
    # "Submitted batch job 12345"
    job_id = proc.stdout.strip().split()[-1]
    return job_id


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def orchestrate(results_dir: str, max_concurrent: int, dry_run: bool):
    # create results/log directories under the benchmarks root
    results_path = ROOT_DIR / results_dir
    results_path.mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "log").mkdir(exist_ok=True)

    manifest = load_manifest(results_path)
    combos = list(all_combinations(GRID))

    pending = []
    skipped = 0
    for c in combos:
        key = (c["package"], c["model_type"], c["N"], c["K"], c["nFE1"])
        rjson = result_json(results_path, *key)
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

        job_id = submit_job(c, results_dir)
        append_manifest(results_path, {**c, "job_id": job_id})
        submitted += 1
        print(f"  [{submitted}/{len(pending)}] submitted job {job_id}: "
              f"{c['package']} {c['model_type']} N={c['N']} K={c['K']} nFE={c['nFE1']}")

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
        ["python", "collect_results.py", "--results-dir", results_dir],
        cwd=str(SCRIPT_DIR),
    )
    if collect_proc.returncode != 0:
        print("collect_results.py exited with an error — check logs.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate benchmark SLURM jobs")
    parser.add_argument("--results-dir",    default="results")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--dry-run",        action="store_true",
                        help="Print what would be submitted without actually submitting")
    parser.add_argument("--collect-only",   action="store_true",
                        help="Skip submission; just collect existing results")
    args = parser.parse_args()

    if args.collect_only:
        subprocess.run(
            ["python", "collect_results.py", "--results-dir", args.results_dir],
            cwd=str(SCRIPT_DIR),
        )
    else:
        orchestrate(args.results_dir, args.max_concurrent, args.dry_run)
