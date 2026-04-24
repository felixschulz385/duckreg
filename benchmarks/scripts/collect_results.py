"""
Collect individual JSON result files into a single CSV.

Artefacts for a given run live under three sibling directories:

    benchmarks/results/<run_id>/     – per-combination JSON result files
    benchmarks/manifests/<run_id>/   – manifest.csv tracking submitted jobs
    benchmarks/logs/<run_id>/        – SLURM .log / .err files

Usage:
    python collect_results.py --run-id 20260305_143022
    python collect_results.py --run-id 20260305_143022 --output my_results.csv
    # Override base directories if needed:
    python collect_results.py --run-id ID --results-base results \
        --manifests-base manifests --logs-base logs
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
BENCH_DIR  = SCRIPT_DIR.parent
OOM_PATTERN = re.compile(r"Detected \d+ oom_kill event", re.IGNORECASE)
# Lines in .err files that indicate a memory-related failure.
_ERR_PATTERNS = re.compile(
    r"(oom_kill|Killed|MemoryError|Out of memory|Cannot allocate|bad_alloc"
    r"|std::bad_alloc|memory exhausted|Traceback|Error:|Exception:)",
    re.IGNORECASE,
)
_MAX_ERR_LINES = 10  # max lines to capture per job


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_manifest(manifests_path: Path) -> pd.DataFrame:
    manifest_csv = manifests_path / "manifest.csv"
    if not manifest_csv.exists():
        return pd.DataFrame(
            columns=["package", "model_type", "N", "K", "nFE1", "nFE2",
                     "vcov", "threads", "job_id"]
        )
    return pd.read_csv(manifest_csv, dtype={"job_id": str})


def read_err_info(logs_path: Path, job_id: str) -> tuple[bool, str | None]:
    """Read the SLURM .err file for *job_id*.

    Returns
    -------
    (is_oom, err_snippet)
        *is_oom*      – True when an OOM kill event was detected.
        *err_snippet* – Up to _MAX_ERR_LINES relevant lines joined by ``|``,
                        or None when no matching lines were found.
    """
    err_file = logs_path / f"slurm-{job_id}.err"
    if not err_file.exists():
        return False, None
    try:
        content = err_file.read_text(errors="replace")
    except Exception:
        return False, None
    is_oom   = bool(OOM_PATTERN.search(content))
    relevant = [
        line.strip()
        for line in content.splitlines()
        if _ERR_PATTERNS.search(line)
    ]
    snippet = " | ".join(relevant[:_MAX_ERR_LINES]) or None
    return is_oom, snippet


def load_json_result(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"_load_error": str(e), "status": "parse_error"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect(run_id: str, output_csv: str,
            results_base: str  = "results",
            manifests_base: str = "manifests",
            logs_base: str     = "logs"):
    results_path   = BENCH_DIR / results_base   / run_id
    manifests_path = BENCH_DIR / manifests_base / run_id
    logs_path      = BENCH_DIR / logs_base      / run_id

    manifest = load_manifest(manifests_path)

    records = []

    # --- 1. Load all JSON result files ---
    json_files = sorted(results_path.glob("*.json"))
    print(f"Found {len(json_files)} result JSON files in {results_path}")

    completed_keys = set()
    for jf in json_files:
        data = load_json_result(jf)
        if "_load_error" not in data:
            records.append(data)
            key = (
                data.get("package"),
                data.get("model_type"),
                data.get("N"),
                data.get("K"),
                data.get("nFE1"),
                data.get("vcov", "HC1"),
                data.get("threads", 1),
            )
            completed_keys.add(key)

    # --- 2. Find missing runs from manifest and check for OOM ---
    oom_count     = 0
    missing_count = 0
    if not manifest.empty:
        for _, row in manifest.iterrows():
            key = (
                row["package"],
                row["model_type"],
                int(row["N"]),
                int(row["K"]),
                int(row["nFE1"]),
                row.get("vcov", "HC1"),
                int(row.get("threads", 1)),
            )
            if key in completed_keys:
                continue  # already have result

            job_id = str(row.get("job_id", ""))
            is_oom, slurm_err_snippet = read_err_info(logs_path, job_id) if job_id else (False, None)
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
                "vcov":         row.get("vcov", "HC1"),
                "threads":      int(row.get("threads", 1)),
                "job_id":       job_id,
                "status":       status,
                "time_seconds": None,
                "error":        "OOM killed" if is_oom else "No result file found",
                "slurm_error":  slurm_err_snippet,
            })

    # --- 3. Scan ALL .err files in logs_path; attach slurm_error to existing records ---
    # Also collect a set of OOM job IDs for the summary print.
    oom_job_ids = []
    job_err_map: dict[str, str | None] = {}   # job_id -> snippet
    if logs_path.exists():
        for err_file in sorted(logs_path.glob("slurm-*.err")):
            jid = err_file.stem.replace("slurm-", "")
            is_oom, snippet = read_err_info(logs_path, jid)
            if is_oom:
                oom_job_ids.append(jid)
            if snippet:
                job_err_map[jid] = snippet

    # Back-fill slurm_error on records loaded from JSON (they don't have it).
    for rec in records:
        if "slurm_error" not in rec:
            rec["slurm_error"] = job_err_map.get(str(rec.get("job_id", "")), None)

    # --- 4. Build DataFrame and save ---
    df = pd.DataFrame(records)

    if df.empty:
        print("No results to collect.")
        return df

    # Ensure consistent column order
    cols = ["package", "model_type", "N", "K", "nFE1", "nFE2", "vcov", "threads",
            "job_id", "status", "time_seconds", "error", "slurm_error"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]

    out_path = BENCH_DIR / output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # --- 5. Summary ---
    print(f"\nResults saved to {out_path}")
    print(f"  Total records  : {len(df)}")
    print(f"  Success        : {(df['status'] == 'success').sum()}")
    print(f"  OOM killed     : {(df['status'] == 'oom_killed').sum()}")
    print(f"  Failed (Python): {(df['status'] == 'failed').sum()}")
    print(f"  Missing        : {(df['status'] == 'missing').sum()}")

    if oom_job_ids:
        print(f"\n  OOM events in .err files (job IDs): {', '.join(oom_job_ids)}")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect benchmark JSON results into CSV")
    parser.add_argument("--run-id",          required=True,
                        help="Run ID (e.g. 20260305_143022)")
    parser.add_argument("--results-base",    default="results")
    parser.add_argument("--manifests-base",  default="manifests")
    parser.add_argument("--logs-base",       default="logs")
    parser.add_argument("--output",          default=None,
                        help="Output CSV path relative to benchmarks root "
                             "(default: benchmark_results_<run_id>.csv)")
    args = parser.parse_args()

    output_csv = args.output or f"benchmark_results_{args.run_id}.csv"

    df = collect(
        run_id=args.run_id,
        output_csv=output_csv,
        results_base=args.results_base,
        manifests_base=args.manifests_base,
        logs_base=args.logs_base,
    )
    if df is not None and not df.empty:
        print("\nStatus breakdown:")
        print(df.groupby(["package", "threads", "vcov", "status"]).size().to_string())

