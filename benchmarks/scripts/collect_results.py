"""
Collect individual JSON result files into a single CSV.  This helper
is kept under `scripts/`; it should be invoked from the benchmarks root or via
`scripts/orchestrate.py`.  Results and the optional manifest are expected to
live inside the supplied `results-dir`.

Usage:
    python collect_results.py [--results-dir results] [--output benchmark_results_large.csv]
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

# relocate to benchmarks root so that results and log dirs are siblings
SCRIPT_DIR = Path(__file__).parent
BENCH_DIR = SCRIPT_DIR.parent
OOM_PATTERN = re.compile(r"Detected \d+ oom_kill event", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_manifest(results_path: Path) -> pd.DataFrame:
    manifest_csv = results_path / "manifest.csv"
    if not manifest_csv.exists():
        return pd.DataFrame(columns=["package", "model_type", "N", "K", "nFE1", "nFE2", "job_id"])
    return pd.read_csv(manifest_csv, dtype={"job_id": str})


def check_err_for_oom(log_dir: Path, job_id: str) -> bool:
    """Return True if the .err file for job_id contains an OOM kill event."""
    err_file = log_dir / f"slurm-{job_id}.err"
    if not err_file.exists():
        return False
    try:
        content = err_file.read_text(errors="replace")
        return bool(OOM_PATTERN.search(content))
    except Exception:
        return False


def load_json_result(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"_load_error": str(e), "status": "parse_error"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect(results_dir: str, output_csv: str):
    results_path = BENCH_DIR / results_dir
    log_dir = BENCH_DIR / "log"
    manifest = load_manifest(results_path)

    records = []

    # --- 1. Load all JSON result files ---
    json_files = sorted(results_path.glob("*.json"))
    print(f"Found {len(json_files)} result JSON files in {results_path}")

    completed_keys = set()
    for jf in json_files:
        data = load_json_result(jf)
        if "_load_error" not in data:
            records.append(data)
            key = (data.get("package"), data.get("model_type"),
                   data.get("N"), data.get("K"), data.get("nFE1"))
            completed_keys.add(key)

    # --- 2. Find missing runs from manifest and check for OOM ---
    oom_count = 0
    missing_count = 0
    if not manifest.empty:
        for _, row in manifest.iterrows():
            key = (row["package"], row["model_type"],
                   int(row["N"]), int(row["K"]), int(row["nFE1"]))
            if key in completed_keys:
                continue  # already have result

            job_id = str(row.get("job_id", ""))
            is_oom = check_err_for_oom(log_dir, job_id) if job_id else False
            status = "oom_killed" if is_oom else "missing"
            if is_oom:
                oom_count += 1
            else:
                missing_count += 1

            records.append({
                "package":      row["package"],
                "model_type":   row["model_type"],
                "N":            int(row["N"]),
                "K":            int(row["K"]),
                "nFE1":         int(row["nFE1"]),
                "nFE2":         int(row.get("nFE2", row["nFE1"])),
                "job_id":       job_id,
                "status":       status,
                "time_seconds": None,
                "error":        "OOM killed" if is_oom else "No result file found",
            })

    # --- 3. Also scan ALL .err files for any OOM events (summary) ---
    all_err_files = sorted(log_dir.glob("slurm-*.err"))
    oom_job_ids = []
    for err_file in all_err_files:
        job_id = err_file.stem.replace("slurm-", "")
        if check_err_for_oom(log_dir, job_id):
            oom_job_ids.append(job_id)

    # --- 4. Build DataFrame and save ---
    df = pd.DataFrame(records)

    if df.empty:
        print("No results to collect.")
        return df

    # Ensure consistent column order
    cols = ["package", "model_type", "N", "K", "nFE1", "nFE2",
            "job_id", "status", "time_seconds", "error"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]

    out_path = BENCH_DIR / output_csv
    df.to_csv(out_path, index=False)

    # --- 5. Summary ---
    print(f"\nResults saved to {out_path}")
    print(f"  Total records  : {len(df)}")
    print(f"  Success        : {(df['status'] == 'success').sum()}")
    print(f"  OOM killed     : {(df['status'] == 'oom_killed').sum()}")
    print(f"  Failed (Python): {(df['status'] == 'failed').sum()}")
    print(f"  Missing        : {(df['status'] == 'missing').sum()}")

    if oom_job_ids:
        print(f"\n  OOM events detected in .err files (job IDs): {', '.join(oom_job_ids)}")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect benchmark JSON results into CSV")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output",      default="benchmark_results_large.csv")
    args = parser.parse_args()

    df = collect(args.results_dir, args.output)
    if df is not None and not df.empty:
        print("\nStatus breakdown:")
        print(df.groupby(["package", "status"]).size().to_string())
