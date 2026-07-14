"""Core benchmark-suite functionality shared by the CLI and tests."""

from __future__ import annotations

import csv
import gc
import hashlib
import html
import importlib.metadata
import json
import math
import os
import platform
import re
import resource
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
RUNS_DIR = BENCH_DIR / "runs"
CONFIG_DIR = BENCH_DIR / "configs"
SCHEMA_VERSION = 1
PACKAGES = ("duckreg", "pyfixest")
IDENTITY_FIELDS = (
    "track", "model", "N", "K", "nFE1", "nFE2", "fe_structure",
    "data_regime", "input_mode", "vcov", "threads", "seed",
)


@dataclass(frozen=True)
class Resources:
    memory_gb: int
    time_minutes: int
    cpus: int

    @property
    def bucket(self) -> str:
        return f"c{self.cpus}_m{self.memory_gb}_t{self.time_minutes}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id(profile: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git_info()["commit"][:8]
    return f"{stamp}-{profile}-{sha}"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False,
                  default=_json_default)
        handle.write("\n")
    os.replace(tmp, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def stable_id(value: dict[str, Any], length: int = 64) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:length]


def git_info() -> dict[str, Any]:
    def command(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
        )
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"

    status = command("status", "--porcelain")
    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status and status != "unknown"),
    }


def package_versions() -> dict[str, str | None]:
    names = ("duckreg", "pyfixest", "numpy", "pandas", "duckdb", "numba")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_metadata() -> dict[str, Any]:
    processor = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if not processor and cpuinfo.exists():
        try:
            match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo.read_text(), re.MULTILINE)
            processor = match.group(1) if match else ""
        except OSError:
            pass
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except OSError:
            pass
    thread_environment = {
        key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
        )
    }
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": processor,
        "hostname": socket.gethostname(),
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": affinity,
        "thread_environment": thread_environment,
        "packages": package_versions(),
    }


def load_profile(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{name}.json"
    if not path.exists():
        available = ", ".join(p.stem for p in sorted(CONFIG_DIR.glob("*.json")))
        raise ValueError(f"Unknown profile {name!r}; available: {available}")
    profile = load_json(path)
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported profile schema in {path}")
    packages = profile.get("packages", [])
    if not packages or any(package not in PACKAGES for package in packages):
        raise ValueError(f"Invalid packages in {path}: {packages!r}")
    if int(profile.get("warmups", 0)) < 0 or int(profile.get("repetitions", 1)) < 1:
        raise ValueError(f"Invalid warmup/repetition counts in {path}")
    if not isinstance(profile.get("tracks"), list) or not profile["tracks"]:
        raise ValueError(f"Profile {path} must define at least one track")
    return profile


def _base_case(track: str, profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "track": track,
        "model": "pooled",
        "N": 100_000,
        "K": 5,
        "nFE1": 1_000,
        "nFE2": 200,
        "fe_structure": "balanced",
        "data_regime": "continuous",
        "input_mode": "memory",
        "vcov": "HC1",
        "threads": 1,
        "seed": int(profile.get("seed", 20260714)),
        "packages": list(profile.get("packages", PACKAGES)),
        "warmups": int(profile.get("warmups", 0)),
        "repetitions": int(profile.get("repetitions", 1)),
    }


def _values(item: dict[str, Any], key: str, default: Iterable[Any]) -> list[Any]:
    value = item.get(key, list(default))
    return value if isinstance(value, list) else [value]


def _expand_track(item: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
    kind = item["kind"]
    track = item.get("name", kind)
    base = _base_case(track, profile)
    base.update(item.get("base", {}))
    cases: list[dict[str, Any]] = []

    if kind in {"matrix", "comparison", "out_of_core"}:
        for n in _values(item, "N", [base["N"]]):
            for k in _values(item, "K", [base["K"]]):
                for model in _values(item, "models", [base["model"]]):
                    for vcov in _values(item, "vcovs", [base["vcov"]]):
                        for regime in _values(item, "data_regimes", [base["data_regime"]]):
                            for threads in _values(item, "threads", [base["threads"]]):
                                case = dict(base, N=int(n), K=int(k), model=model,
                                            vcov=vcov, data_regime=regime,
                                            threads=int(threads))
                                divisor1, divisor2 = item.get("fe_divisors", [100, 500])
                                case["nFE1"] = max(2, int(n) // int(divisor1))
                                case["nFE2"] = max(2, int(n) // int(divisor2))
                                if kind == "out_of_core":
                                    case["input_mode"] = "parquet"
                                    case["packages"] = ["duckreg"]
                                cases.append(case)
    elif kind == "fe_cardinality":
        for pair in item["levels"]:
            cases.append(dict(base, nFE1=int(pair[0]), nFE2=int(pair[1])))
    elif kind == "explicit":
        for values in item["scenarios"]:
            cases.append(dict(base, **values))
    elif kind == "diagnostics":
        for benchmark in item["benchmarks"]:
            cases.append(dict(base, model="diagnostic", diagnostic=benchmark,
                              packages=["duckreg"], vcov="none"))
    else:
        raise ValueError(f"Unknown track kind: {kind}")
    return cases


def resources_for(case: dict[str, Any]) -> Resources:
    n = int(case["N"])
    cpus = int(case["threads"])
    if n <= 1_000_000:
        return Resources(32, 60, cpus)
    if n <= 10_000_000:
        return Resources(64, 240, cpus)
    return Resources(192, 360, cpus)


def expand_profile(profile: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for track in profile["tracks"]:
        cases.extend(_expand_track(track, profile))

    unique: dict[str, dict[str, Any]] = {}
    for case in cases:
        _validate_case(case)
        identity = {field: case[field] for field in IDENTITY_FIELDS}
        if case.get("diagnostic"):
            identity["diagnostic"] = case["diagnostic"]
        scenario_id = stable_id(identity)
        resources = resources_for(case)
        case.update(
            schema_version=SCHEMA_VERSION,
            scenario_id=scenario_id,
            resources=asdict(resources),
            resource_bucket=resources.bucket,
        )
        if scenario_id in unique and unique[scenario_id] != case:
            raise ValueError(f"Scenario ID collision: {scenario_id}")
        unique[scenario_id] = case
    return list(unique.values())


def _validate_case(case: dict[str, Any]) -> None:
    for field in ("N", "K", "nFE1", "nFE2", "threads", "repetitions"):
        if int(case[field]) < 1:
            raise ValueError(f"{field} must be positive: {case!r}")
    if int(case["warmups"]) < 0:
        raise ValueError(f"warmups must be non-negative: {case!r}")
    if case["model"] not in {"pooled", "fe", "iv", "diagnostic"}:
        raise ValueError(f"Unknown model: {case['model']!r}")
    if case["vcov"] not in {"HC1", "CRV1", "none"}:
        raise ValueError(f"Unknown vcov: {case['vcov']!r}")
    if case["fe_structure"] not in {"balanced", "sparse"}:
        raise ValueError(f"Unknown FE structure: {case['fe_structure']!r}")
    if case["data_regime"] not in {"continuous", "compressible"}:
        raise ValueError(f"Unknown data regime: {case['data_regime']!r}")
    if case["input_mode"] not in {"memory", "parquet"}:
        raise ValueError(f"Unknown input mode: {case['input_mode']!r}")
    if (not case["packages"] or len(case["packages"]) != len(set(case["packages"])) or
            any(package not in PACKAGES for package in case["packages"])):
        raise ValueError(f"Invalid package set: {case['packages']!r}")
    if int(case["nFE1"]) > int(case["N"]) or int(case["nFE2"]) > int(case["N"]):
        raise ValueError("FE cardinalities cannot exceed N")
    if case["model"] == "diagnostic" and case["vcov"] != "none":
        raise ValueError("Diagnostic scenarios must use vcov='none'")
    if case["model"] != "diagnostic" and case["vcov"] == "none":
        raise ValueError("Comparison scenarios must use HC1 or CRV1")
    if case["input_mode"] == "parquet" and case["packages"] != ["duckreg"]:
        raise ValueError("Parquet scenarios must be DuckReg-only")


def run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run-id may contain only letters, digits, '.', '_' and '-'")
    return RUNS_DIR / run_id


def create_run(profile_name: str, requested_id: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    profile = load_profile(profile_name)
    scenarios = expand_profile(profile)
    run_id = requested_id or make_run_id(profile_name)
    root = run_dir(run_id)
    manifest_path = root / "tasks.jsonl"
    if manifest_path.exists():
        existing = read_manifest(run_id)
        if existing != scenarios:
            raise ValueError(f"Run {run_id!r} exists with a different manifest")
        return run_id, existing

    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "profile": profile_name,
        "created_at": utc_now(),
        "git": git_info(),
        "environment": environment_metadata(),
        "scenario_count": len(scenarios),
    }
    atomic_json(root / "run.json", metadata)
    tmp = manifest_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for scenario in scenarios:
            handle.write(json.dumps(scenario, sort_keys=True) + "\n")
    os.replace(tmp, manifest_path)
    return run_id, scenarios


def read_manifest(run_id: str) -> list[dict[str, Any]]:
    path = run_dir(run_id) / "tasks.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def find_scenario(run_id: str, scenario_id: str) -> dict[str, Any]:
    for scenario in read_manifest(run_id):
        if scenario["scenario_id"] == scenario_id:
            return scenario
    raise ValueError(f"Unknown scenario {scenario_id!r} in run {run_id!r}")


def result_path(run_id: str, scenario_id: str, package: str) -> Path:
    return run_dir(run_id) / "raw" / scenario_id / f"{package}.json"


def result_succeeded(path: Path) -> bool:
    try:
        value = load_json(path)
        validate_result_record(value)
        return value.get("status") == "success"
    except Exception:
        return False


def validate_result_record(value: dict[str, Any], *, scenario_id: str | None = None,
                           package: str | None = None) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported or missing result schema_version")
    if scenario_id is not None and value.get("scenario_id") != scenario_id:
        raise ValueError("Result scenario_id does not match its manifest entry")
    if package is not None and value.get("package") != package:
        raise ValueError("Result package does not match its manifest entry")
    status = value.get("status")
    allowed = {
        "success", "oom", "timeout", "convergence_failure", "exception",
        "process_failure", "scheduler_failure",
    }
    if status not in allowed:
        raise ValueError(f"Unknown or missing result status: {status!r}")
    if status == "success":
        required = ("case", "measurements", "coefficients", "standard_errors", "nobs")
        missing = [field for field in required if field not in value]
        if missing:
            raise ValueError(f"Successful result is missing fields: {missing}")
        if not isinstance(value["measurements"], list) or not value["measurements"]:
            raise ValueError("Successful result must contain measurements")
        if value["case"].get("model") != "diagnostic" and not value.get("data_digest"):
            raise ValueError("Successful comparison result is missing data_digest")


def pending_scenarios(run_id: str) -> list[dict[str, Any]]:
    pending = []
    for scenario in read_manifest(run_id):
        if any(not result_succeeded(result_path(run_id, scenario["scenario_id"], package))
               for package in scenario["packages"]):
            pending.append(scenario)
    return pending


def configure_threads(threads: int) -> None:
    value = str(threads)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = value
    os.environ.setdefault("PYTHONHASHSEED", "0")


def _balanced_ids(n: int, groups: int, rng: np.random.Generator) -> np.ndarray:
    values = np.arange(n, dtype=np.int64) % groups
    rng.shuffle(values)
    return values


def generate_data(case: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float], str, float]:
    started = time.perf_counter()
    n, k = int(case["N"]), int(case["K"])
    rng = np.random.default_rng(int(case["seed"]))
    if case["fe_structure"] == "balanced":
        fe1 = _balanced_ids(n, int(case["nFE1"]), rng)
        fe2 = _balanced_ids(n, int(case["nFE2"]), rng)
    elif case["fe_structure"] == "sparse":
        fe1 = rng.integers(0, int(case["nFE1"]), n, dtype=np.int64)
        fe2 = rng.integers(0, int(case["nFE2"]), n, dtype=np.int64)
    else:
        raise ValueError(f"Unknown FE structure: {case['fe_structure']}")

    if case["data_regime"] == "continuous":
        x = rng.normal(size=(n, k))
    elif case["data_regime"] == "compressible":
        x = rng.integers(0, 2, size=(n, k)).astype(np.float64)
    else:
        raise ValueError(f"Unknown data regime: {case['data_regime']}")

    instrument = rng.normal(size=n)
    structural_error = rng.normal(size=n)
    treatment = 0.8 * instrument + 0.6 * structural_error + rng.normal(size=n)
    beta = np.linspace(0.5, 1.5, k)
    fe1_effect = rng.normal(size=int(case["nFE1"]))
    fe2_effect = rng.normal(size=int(case["nFE2"]))
    outcome = (1.5 * treatment + x @ beta + fe1_effect[fe1] +
               fe2_effect[fe2] + structural_error)
    values: dict[str, Any] = {
        "y": outcome, "D": treatment, "Z": instrument, "fe1": fe1, "fe2": fe2,
    }
    values.update({f"x{i + 1}": x[:, i] for i in range(k)})
    frame = pd.DataFrame(values)
    sample = np.linspace(0, n - 1, min(n, 1024), dtype=np.int64)
    hashed = pd.util.hash_pandas_object(frame.iloc[sample], index=True).values.tobytes()
    digest = hashlib.sha256(hashed).hexdigest()
    truth = {"D": 1.5, **{f"x{i + 1}": float(beta[i]) for i in range(k)}}
    return frame, truth, digest, time.perf_counter() - started


def generate_parquet(case: dict[str, Any], path: Path, chunk_size: int = 250_000
                     ) -> tuple[dict[str, float], str, float]:
    """Generate a file-backed workload without materialising the full frame."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.perf_counter()
    n, k = int(case["N"]), int(case["K"])
    nfe1, nfe2 = int(case["nFE1"]), int(case["nFE2"])
    rng = np.random.default_rng(int(case["seed"]))
    beta = np.linspace(0.5, 1.5, k)
    fe1_effect = rng.normal(size=nfe1)
    fe2_effect = rng.normal(size=nfe2)
    digest = hashlib.sha256()
    writer = None
    try:
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            size = stop - start
            index = np.arange(start, stop, dtype=np.int64)
            if case["fe_structure"] == "balanced":
                fe1 = index % nfe1
                fe2 = (index * 31) % nfe2
            elif case["fe_structure"] == "sparse":
                fe1 = rng.integers(0, nfe1, size, dtype=np.int64)
                fe2 = rng.integers(0, nfe2, size, dtype=np.int64)
            else:
                raise ValueError(f"Unknown FE structure: {case['fe_structure']}")
            if case["data_regime"] == "continuous":
                x = rng.normal(size=(size, k))
            elif case["data_regime"] == "compressible":
                x = rng.integers(0, 2, size=(size, k)).astype(np.float64)
            else:
                raise ValueError(f"Unknown data regime: {case['data_regime']}")
            instrument = rng.normal(size=size)
            structural_error = rng.normal(size=size)
            treatment = 0.8 * instrument + 0.6 * structural_error + rng.normal(size=size)
            outcome = (1.5 * treatment + x @ beta + fe1_effect[fe1] +
                       fe2_effect[fe2] + structural_error)
            values: dict[str, Any] = {
                "y": outcome, "D": treatment, "Z": instrument,
                "fe1": fe1, "fe2": fe2,
            }
            values.update({f"x{i + 1}": x[:, i] for i in range(k)})
            frame = pd.DataFrame(values)
            digest.update(pd.util.hash_pandas_object(
                frame.iloc[:min(32, size)], index=True,
            ).values.tobytes())
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    truth = {"D": 1.5, **{f"x{i + 1}": float(beta[i]) for i in range(k)}}
    return truth, digest.hexdigest(), time.perf_counter() - started


def formulas(case: dict[str, Any]) -> tuple[str, str]:
    x_vars = " + ".join(f"x{i + 1}" for i in range(int(case["K"])))
    model = case["model"]
    if model == "pooled":
        formula = f"y ~ D + {x_vars}"
        return formula, formula
    if model == "fe":
        formula = f"y ~ D + {x_vars} | fe1 + fe2"
        return formula, formula
    if model == "iv":
        return (f"y ~ {x_vars} | fe1 + fe2 | (D ~ Z)",
                f"y ~ {x_vars} | fe1 + fe2 | D ~ Z")
    raise ValueError(f"Unknown model: {model}")


def _peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def _normalise_series(values: pd.Series) -> dict[str, float]:
    result = {}
    for name, value in values.items():
        normal = "Intercept" if str(name).lower() in {"intercept", "const"} else str(name)
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Non-finite estimate for {normal}: {number}")
        result[normal] = number
    return result


def _extract_result(package: str, model: Any) -> tuple[dict[str, float], dict[str, float], int, dict[str, Any]]:
    if package == "duckreg":
        coef = _normalise_series(model.coef())
        se = _normalise_series(model.se())
        nobs_value = getattr(model, "nobs", None)
        if callable(nobs_value):
            nobs_value = nobs_value()
        if nobs_value is None:
            nobs_value = getattr(getattr(model, "results", None), "n_obs", None)
        if nobs_value is None:
            nobs_value = getattr(model, "n_obs", 0)
        nobs = int(nobs_value)
        extra = {
            "resolved_fitter": getattr(model, "resolved_fitter", None),
            "resolved_compression": getattr(model, "resolved_compression", None),
            "map_iterations": getattr(model, "map_iterations", None),
            "n_compressed_rows": getattr(model, "n_compressed_rows", None),
        }
    else:
        tidy = model.tidy()
        coef = _normalise_series(tidy["Estimate"])
        se = _normalise_series(tidy["Std. Error"])
        nobs = int(getattr(model, "_N_rows", getattr(model, "_N", 0)))
        extra = {}
    return coef, se, nobs, extra


def prepare_fit(package: str, case: dict[str, Any]):
    """Import a package and build its formula outside the timed region."""
    duck_formula, fixest_formula = formulas(case)
    if package == "duckreg":
        from duckreg import duckreg

        se_method: str | dict[str, str] = case["vcov"]
        if case["vcov"] == "CRV1":
            se_method = {"CRV1": "fe1"}
        memory_gb = int(case["resources"]["memory_gb"] * 0.75)
        def fit(data: Any):
            return duckreg(
                duck_formula, data, se_method=se_method, fe_method="demean",
                fitter="duckdb", threads=int(case["threads"]),
                memory_limit=f"{memory_gb}GB",
                max_temp_directory_size=f"{case['resources']['memory_gb']}GB",
            )
        return fit

    import numba
    import pyfixest as pf

    numba.set_num_threads(int(case["threads"]))
    vcov: str | dict[str, str] = "hetero" if case["vcov"] == "HC1" else {"CRV1": "fe1"}
    def fit(data: Any):
        return pf.feols(fixest_formula, data, vcov=vcov)
    return fit


def _run_diagnostic(case: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    name = case["diagnostic"]
    started = time.perf_counter()
    if name == "cluster_scores":
        from duckreg.core.vcov import compute_cluster_scores
        scores = frame[[f"x{i + 1}" for i in range(min(case["K"], 5))]].to_numpy()
        compute_cluster_scores(scores, frame["fe1"].to_numpy())
        extra = {}
    elif name in {"map_numpy", "map_duckdb"}:
        from duckreg import duckreg
        model = duckreg(
            formulas(dict(case, model="fe"))[0], frame, se_method="none",
            fitter="numpy", compression=-1,
            demean_backend=name.removeprefix("map_"), threads=case["threads"],
        )
        extra = {"map_iterations": getattr(model, "map_iterations", None)}
    elif name == "automatic_compression":
        from duckreg import duckreg
        model = duckreg("y ~ x1", frame, se_method="none", threads=case["threads"])
        extra = {"resolved_compression": getattr(model, "resolved_compression", None)}
    else:
        raise ValueError(f"Unknown diagnostic: {name}")
    return {"phase": "cold", "iteration": 0,
            "wall_seconds": time.perf_counter() - started, "cpu_seconds": None,
            "extra": extra}


def classify_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, MemoryError) or any(word in text for word in ("out of memory", "oom", "bad_alloc")):
        return "oom"
    if isinstance(exc, TimeoutError) or "time limit" in text or "timeout" in text:
        return "timeout"
    if "converg" in text or "maximum iterations" in text:
        return "convergence_failure"
    return "exception"


def execute_worker(run_id: str, scenario_id: str, package: str, scratch: Path | None = None) -> int:
    case = find_scenario(run_id, scenario_id)
    if package not in case["packages"]:
        raise ValueError(f"Package {package!r} is not part of scenario {scenario_id}")
    configure_threads(int(case["threads"]))
    output = result_path(run_id, scenario_id, package)
    scratch_root = scratch or Path(tempfile.mkdtemp(prefix=f"duckreg-bench-{scenario_id}-"))
    scratch_root.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch_root)
    if scratch is not None and len(scratch_root.parents) >= 2:
        cache_root = scratch_root.parents[1] / "_cache" / package
    else:
        cache_root = scratch_root / "_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "scenario_id": scenario_id,
        "package": package,
        "status": "exception",
        "started_at": utc_now(),
        "case": case,
        "environment": environment_metadata(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
        },
        "measurements": [],
        "warnings": [],
        "error": None,
    }
    exit_code = 0
    try:
        if case["input_mode"] == "parquet":
            if package != "duckreg":
                raise ValueError("File-backed scenarios are DuckReg-only")
            parquet = scratch_root / "data.parquet"
            truth, digest, generation_seconds = generate_parquet(case, parquet)
            data = str(parquet)
        else:
            frame, truth, digest, generation_seconds = generate_data(case)
            data = frame
        record.update(data_digest=digest, truth=truth,
                      data_generation_seconds=generation_seconds)

        if case["model"] == "diagnostic":
            record["measurements"] = [_run_diagnostic(case, data)]
            record.update(status="success", coefficients={}, standard_errors={}, nobs=case["N"])
        else:
            total = int(case["warmups"]) + int(case["repetitions"])
            last_model = None
            fit = prepare_fit(package, case)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for iteration in range(total):
                    gc.collect()
                    cpu_start = time.process_time()
                    wall_start = time.perf_counter()
                    last_model = fit(data)
                    wall = time.perf_counter() - wall_start
                    cpu = time.process_time() - cpu_start
                    phase = "cold" if iteration < int(case["warmups"]) else "steady"
                    if not case["warmups"]:
                        phase = "cold"
                    record["measurements"].append({
                        "phase": phase,
                        "iteration": iteration,
                        "wall_seconds": wall,
                        "cpu_seconds": cpu,
                    })
                record["warnings"] = sorted({
                    f"{item.category.__name__}: {item.message}" for item in caught
                })[:20]
            coef, se, nobs, extra = _extract_result(package, last_model)
            record.update(status="success", coefficients=coef,
                          standard_errors=se, nobs=nobs, extra=extra)
        record["peak_rss_mb"] = _peak_rss_mb()
    except BaseException as exc:  # preserve diagnostics for worker-level failures
        record["status"] = classify_exception(exc)
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["peak_rss_mb"] = _peak_rss_mb()
        exit_code = 1
    finally:
        record["finished_at"] = utc_now()
        atomic_json(output, record)
        if scratch is None:
            shutil.rmtree(scratch_root, ignore_errors=True)
    return exit_code


def worker_command(run_id: str, scenario_id: str, package: str, scratch: Path | None = None) -> list[str]:
    command = [sys.executable, str(BENCH_DIR / "bench.py"), "worker",
               "--run-id", run_id, "--scenario-id", scenario_id, "--package", package]
    if scratch is not None:
        command.extend(["--scratch", str(scratch)])
    return command


def execute_scenario(run_id: str, scenario: dict[str, Any], use_srun: bool = False) -> int:
    failures = 0
    packages = list(scenario["packages"])
    if int(scenario["scenario_id"], 16) % 2:
        packages.reverse()
    for package in packages:
        path = result_path(run_id, scenario["scenario_id"], package)
        if result_succeeded(path):
            continue
        scratch_base = Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir()))
        scratch = scratch_base / run_id / scenario["scenario_id"] / package
        command = worker_command(run_id, scenario["scenario_id"], package, scratch)
        if use_srun and shutil.which("srun"):
            command = ["srun", "--exclusive", "--nodes=1", "--ntasks=1",
                       f"--cpus-per-task={scenario['threads']}", *command]
        proc = subprocess.run(command, cwd=REPO_ROOT)
        if proc.returncode and not path.exists():
            if proc.returncode in {-9, 137}:
                status = "oom"
            elif proc.returncode in {124, 143}:
                status = "timeout"
            else:
                status = "process_failure"
            atomic_json(path, {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "scenario_id": scenario["scenario_id"],
                "package": package,
                "status": status,
                "case": scenario,
                "measurements": [],
                "error": f"Worker exited with code {proc.returncode} before writing output",
                "slurm": {
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                    "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                    "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                    "node": os.environ.get("SLURMD_NODENAME"),
                },
            })
        failures += proc.returncode != 0
    return 1 if failures else 0


def _comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    model = left["case"]["model"]
    coef_rtol, se_rtol = ((1e-3, 1e-2) if model == "iv" else (1e-6, 1e-3))
    names_left, names_right = set(left["coefficients"]), set(right["coefficients"])
    se_left, se_right = set(left["standard_errors"]), set(right["standard_errors"])
    common = sorted(names_left & names_right)
    result = {
        "valid": True,
        "reason": None,
        "coef_rtol": coef_rtol,
        "se_rtol": se_rtol,
        "max_coef_relative_error": None,
        "max_se_relative_error": None,
    }
    if left.get("data_digest") != right.get("data_digest"):
        result.update(valid=False, reason="data_digest_mismatch")
        return result
    if left["nobs"] != right["nobs"]:
        result.update(valid=False, reason="observation_count_mismatch")
        return result
    if names_left != names_right or se_left != se_right or not common:
        result.update(valid=False, reason="coefficient_name_mismatch")
        return result

    def relative(a: float, b: float) -> float:
        return abs(a - b) / max(abs(a), abs(b), 1e-12)

    coef_errors = [relative(left["coefficients"][name], right["coefficients"][name])
                   for name in common]
    se_errors = [relative(left["standard_errors"][name], right["standard_errors"][name])
                 for name in common]
    result["max_coef_relative_error"] = max(coef_errors)
    result["max_se_relative_error"] = max(se_errors)
    coef_ok = all(np.isclose(left["coefficients"][name], right["coefficients"][name],
                             rtol=coef_rtol, atol=1e-6) for name in common)
    se_ok = all(np.isclose(left["standard_errors"][name], right["standard_errors"][name],
                           rtol=se_rtol, atol=1e-6) for name in common)
    if not coef_ok or not se_ok:
        result.update(valid=False, reason="numerical_mismatch")
    return result


RESULT_FIELDS = [
    "run_id", "scenario_id", "package", *IDENTITY_FIELDS, "diagnostic",
    "phase", "iteration", "status", "validation_status", "validation_reason",
    "wall_seconds", "cpu_seconds", "peak_rss_mb", "data_generation_seconds",
    "nobs", "max_coef_relative_error", "max_se_relative_error", "job_id",
    "node", "slurm_state", "slurm_max_rss", "error",
]


def collect_results(run_id: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    submissions_path = run_dir(run_id) / "submissions.json"
    submissions = load_json(submissions_path).get("scenarios", {}) if submissions_path.exists() else {}
    for case in read_manifest(run_id):
        loaded: dict[str, dict[str, Any]] = {}
        for package in case["packages"]:
            path = result_path(run_id, case["scenario_id"], package)
            try:
                value = load_json(path)
                validate_result_record(value, scenario_id=case["scenario_id"], package=package)
                loaded[package] = value
            except FileNotFoundError:
                loaded[package] = {"status": "missing", "error": "No result file"}
            except Exception as exc:
                loaded[package] = {"status": "parse_error", "error": str(exc)}

        validation: dict[str, Any] | None = None
        if set(case["packages"]) == set(PACKAGES) and all(
                loaded[p].get("status") == "success" for p in PACKAGES):
            validation = _comparison(loaded["duckreg"], loaded["pyfixest"])

        for package in case["packages"]:
            value = loaded[package]
            status = value.get("status", "parse_error")
            submission = submissions.get(case["scenario_id"], {})
            validation_status = "not_applicable"
            validation_reason = None
            if validation is not None:
                validation_status = "valid" if validation["valid"] else "numerical_mismatch"
                validation_reason = validation["reason"]
                if not validation["valid"] and status == "success":
                    status = "numerical_mismatch"
            measurements = value.get("measurements") or [{
                "phase": None, "iteration": None, "wall_seconds": None, "cpu_seconds": None,
            }]
            for measurement in measurements:
                row = {field: case.get(field) for field in IDENTITY_FIELDS}
                row.update(
                    run_id=run_id,
                    scenario_id=case["scenario_id"],
                    package=package,
                    diagnostic=case.get("diagnostic"),
                    phase=measurement.get("phase"),
                    iteration=measurement.get("iteration"),
                    status=status,
                    validation_status=validation_status,
                    validation_reason=validation_reason,
                    wall_seconds=measurement.get("wall_seconds"),
                    cpu_seconds=measurement.get("cpu_seconds"),
                    peak_rss_mb=value.get("peak_rss_mb"),
                    data_generation_seconds=value.get("data_generation_seconds"),
                    nobs=value.get("nobs"),
                    max_coef_relative_error=(validation or {}).get("max_coef_relative_error"),
                    max_se_relative_error=(validation or {}).get("max_se_relative_error"),
                    job_id=(value.get("slurm", {}).get("job_id") or
                            submission.get("accounting_job_id")),
                    node=value.get("slurm", {}).get("node") or value.get("environment", {}).get("hostname"),
                    slurm_state=None,
                    slurm_max_rss=None,
                    error=value.get("error"),
                )
                records.append(row)
    frame = pd.DataFrame(records, columns=RESULT_FIELDS)
    job_ids = sorted({str(value) for value in frame["job_id"].dropna().unique()})
    if job_ids and shutil.which("sacct"):
        proc = subprocess.run(
            ["sacct", "--jobs", ",".join(job_ids), "--parsable2",
             "--format=JobID,State,MaxRSS,ElapsedRaw,TotalCPU,NodeList"],
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            accounting = parse_sacct(proc.stdout)
            frame["slurm_state"] = frame["job_id"].map(
                lambda value: accounting.get(str(value), {}).get("State")
            )
            frame["slurm_max_rss"] = frame["job_id"].map(
                lambda value: accounting.get(str(value), {}).get("MaxRSS")
            )
            missing = frame["status"] == "missing"
            states = frame["slurm_state"].fillna("").str.upper()
            frame.loc[missing & states.str.contains("OUT_OF_MEMORY"), "status"] = "oom"
            frame.loc[missing & states.str.contains("TIMEOUT"), "status"] = "timeout"
            scheduler_failure = missing & states.str.contains("FAILED|CANCELLED|NODE_FAIL", regex=True)
            frame.loc[scheduler_failure, "status"] = "scheduler_failure"
    output = run_dir(run_id) / "results.csv"
    frame.to_csv(output, index=False)
    return frame


def _geometric_mean(values: pd.Series) -> float:
    positive = values.dropna()
    positive = positive[positive > 0]
    return float(math.exp(np.log(positive).mean())) if len(positive) else math.nan


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    good = frame[(frame["status"] == "success") &
                 (frame["validation_status"].isin(["valid", "not_applicable"]))]
    steady = good[good["phase"] == "steady"]
    if steady.empty:
        steady = good[good["phase"] == "cold"]
    keys = [*IDENTITY_FIELDS, "diagnostic", "package"]
    summary = steady.groupby(keys, dropna=False, as_index=False).agg(
        median_seconds=("wall_seconds", "median"),
        q25_seconds=("wall_seconds", lambda value: value.quantile(0.25)),
        q75_seconds=("wall_seconds", lambda value: value.quantile(0.75)),
        peak_rss_mb=("peak_rss_mb", "max"),
        repetitions=("wall_seconds", "count"),
    )
    return summary


def build_report(run_id: str) -> Path:
    csv_path = run_dir(run_id) / "results.csv"
    frame = pd.read_csv(csv_path) if csv_path.exists() else collect_results(run_id)
    summary = summary_table(frame)
    identity = [*IDENTITY_FIELDS, "diagnostic"]
    metrics = ["median_seconds", "q25_seconds", "q75_seconds",
               "peak_rss_mb", "repetitions"]
    duckreg_summary = summary[summary["package"] == "duckreg"][identity + metrics]
    pyfixest_summary = summary[summary["package"] == "pyfixest"][identity + metrics]
    compared = duckreg_summary.merge(
        pyfixest_summary, on=identity, how="outer",
        suffixes=("_duckreg", "_pyfixest"), validate="one_to_one",
    )
    if {"median_seconds_duckreg", "median_seconds_pyfixest"}.issubset(compared.columns):
        compared["speedup_pyfixest_over_duckreg"] = (
            compared["median_seconds_pyfixest"] / compared["median_seconds_duckreg"]
        )
    if {"peak_rss_mb_duckreg", "peak_rss_mb_pyfixest"}.issubset(compared.columns):
        compared["memory_ratio_pyfixest_over_duckreg"] = (
            compared["peak_rss_mb_pyfixest"] / compared["peak_rss_mb_duckreg"]
        )

    coverage = frame[["scenario_id", "package", "status", "validation_status"]].drop_duplicates()
    coverage = coverage.groupby(
        ["package", "status", "validation_status"], dropna=False,
    ).size().rename("scenarios").reset_index()
    failures = frame[frame["status"] != "success"][
        ["scenario_id", "package", "track", "model", "N", "K", "vcov",
         "threads", "status", "error"]
    ].drop_duplicates()
    track_summary = pd.DataFrame()
    if "speedup_pyfixest_over_duckreg" in compared:
        track_summary = compared.groupby("track", as_index=False).agg(
            scenarios=("speedup_pyfixest_over_duckreg", "count"),
            geometric_mean_speedup=("speedup_pyfixest_over_duckreg", _geometric_mean),
        )

    assets = run_dir(run_id) / "report_assets"
    assets.mkdir(exist_ok=True)
    cache = run_dir(run_id) / ".cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
    plot_html = ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_data = compared.dropna(subset=["speedup_pyfixest_over_duckreg"]) if (
            "speedup_pyfixest_over_duckreg" in compared) else pd.DataFrame()
        if not plot_data.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            for model, subset in plot_data.groupby("model"):
                subset = subset.sort_values("N")
                ax.plot(subset["N"], subset["speedup_pyfixest_over_duckreg"],
                        marker="o", linestyle="none", label=model, alpha=0.75)
            ax.axhline(1.0, color="black", linewidth=1)
            ax.set_xscale("log")
            ax.set_xlabel("Rows")
            ax.set_ylabel("Speedup (pyfixest / DuckReg)")
            ax.legend()
            fig.tight_layout()
            plot_path = assets / "speedup.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            plot_html = '<img src="report_assets/speedup.png" alt="Speedup by row count">'
        scaling = summary[summary["track"] == "scaling"]
        if not scaling.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            groups = scaling.groupby(["package", "model", "vcov"], dropna=False)
            for (package, model, vcov), subset in groups:
                subset = subset.sort_values("N")
                lower = subset["median_seconds"] - subset["q25_seconds"]
                upper = subset["q75_seconds"] - subset["median_seconds"]
                ax.errorbar(
                    subset["N"], subset["median_seconds"],
                    yerr=np.vstack([lower, upper]), marker="o", capsize=3,
                    label=f"{package} / {model} / {vcov}",
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Rows")
            ax.set_ylabel("Fit time (seconds; median and IQR)")
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            timing_path = assets / "scaling_iqr.png"
            fig.savefig(timing_path, dpi=150)
            plt.close(fig)
            plot_html += '<img src="report_assets/scaling_iqr.png" alt="Scaling medians and IQR">'
    except ImportError:
        plot_html = "<p>Matplotlib is unavailable; plots were skipped.</p>"

    metadata = load_json(run_dir(run_id) / "run.json")
    report = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>DuckReg benchmark {html.escape(run_id)}</title>
<style>body{{font:14px system-ui;margin:2rem;max-width:1400px}}table{{border-collapse:collapse}}
th,td{{border:1px solid #ddd;padding:.35rem;text-align:right}}th{{background:#f4f4f4}}
img{{max-width:900px}}code{{background:#eee;padding:.1rem .25rem}}</style></head><body>
<h1>DuckReg benchmark: {html.escape(run_id)}</h1>
<p>Profile: <code>{html.escape(str(metadata['profile']))}</code>; commit:
<code>{html.escape(str(metadata['git']['commit']))}</code>; dirty: {metadata['git']['dirty']}.</p>
<h2>Coverage and failures</h2>{coverage.to_html(index=False, escape=True)}
{failures.to_html(index=False, escape=True) if not failures.empty else '<p>No failures.</p>'}
<h2>Track summary</h2>{track_summary.to_html(index=False, escape=True)}
<h2>Scenario comparisons</h2>{compared.to_html(index=False, escape=True)}
<h2>Speedup</h2>{plot_html}
</body></html>"""
    output = run_dir(run_id) / "report.html"
    output.write_text(report, encoding="utf-8")
    summary.to_csv(run_dir(run_id) / "summary.csv", index=False)
    return output


def bucket_scenarios(scenarios: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for scenario in scenarios:
        buckets.setdefault(scenario["resource_bucket"], []).append(scenario)
    return buckets


def slurm_array_command(run_id: str, scenarios: list[dict[str, Any]], max_concurrent: int,
                        partition: str, qos: str | None = None) -> list[str]:
    first = scenarios[0]
    resource = first["resources"]
    indices = ",".join(str(read_manifest(run_id).index(case)) for case in scenarios)
    wrap = (f"{shutil.which('python') or sys.executable} {BENCH_DIR / 'bench.py'} "
            f"slurm-worker --run-id {run_id} --index $SLURM_ARRAY_TASK_ID")
    command = [
        "sbatch", "--parsable", f"--job-name=dr-{run_id[:20]}",
        f"--partition={partition}", f"--array={indices}%{max_concurrent}",
        f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory_gb']}G",
        f"--time={resource['time_minutes']}",
        f"--output={run_dir(run_id) / 'logs'}/%A_%a.log",
        f"--error={run_dir(run_id) / 'logs'}/%A_%a.err",
        f"--wrap={wrap}",
    ]
    if qos:
        command.insert(4, f"--qos={qos}")
    return command


def submit_run(run_id: str, max_concurrent: int, partition: str, qos: str | None,
               dry_run: bool = False) -> list[list[str]]:
    pending = pending_scenarios(run_id)
    (run_dir(run_id) / "logs").mkdir(exist_ok=True)
    buckets = list(bucket_scenarios(pending).values())
    if buckets and max_concurrent < len(buckets):
        raise ValueError(
            f"--max-concurrent must be at least the number of resource buckets "
            f"({len(buckets)})"
        )
    per_bucket, remainder = divmod(max_concurrent, len(buckets) or 1)
    limits = [per_bucket + (index < remainder) for index in range(len(buckets))]
    commands = [slurm_array_command(run_id, cases, limit, partition, qos)
                for cases, limit in zip(buckets, limits)]
    if dry_run:
        return commands
    job_ids = []
    submission_records: dict[str, Any] = {}
    manifest = read_manifest(run_id)
    for cases, command in zip(buckets, commands):
        proc = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT)
        if proc.returncode:
            raise RuntimeError(f"sbatch failed: {proc.stderr.strip()}")
        job_id = proc.stdout.strip().split(";")[0]
        job_ids.append(job_id)
        for case in cases:
            index = manifest.index(case)
            submission_records[case["scenario_id"]] = {
                "array_job_id": job_id,
                "array_task_id": index,
                "accounting_job_id": f"{job_id}_{index}",
            }
        atomic_json(run_dir(run_id) / "submissions.json", {
            "submitted_at": utc_now(),
            "scenarios": submission_records,
        })
    if job_ids:
        dependency = ":".join(job_ids)
        wrap = (f"{shutil.which('python') or sys.executable} {BENCH_DIR / 'bench.py'} "
                f"collect --run-id {run_id} --report")
        command = [
            "sbatch", "--parsable", f"--job-name=collect-{run_id[:16]}",
            f"--partition={partition}", f"--dependency=afterany:{dependency}",
            "--cpus-per-task=1", "--mem=4G", "--time=30",
            f"--output={run_dir(run_id) / 'logs'}/collect-%j.log",
            f"--error={run_dir(run_id) / 'logs'}/collect-%j.err",
            f"--wrap={wrap}",
        ]
        if qos:
            command.insert(4, f"--qos={qos}")
        proc = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT)
        if proc.returncode:
            raise RuntimeError(f"collector sbatch failed: {proc.stderr.strip()}")
        commands.append(command)
    return commands


def parse_sacct(text: str) -> dict[str, dict[str, str]]:
    """Parse ``sacct -P`` output for tests and optional diagnostics."""
    rows: dict[str, dict[str, str]] = {}
    reader = csv.DictReader(text.splitlines(), delimiter="|")
    for row in reader:
        job_id = row.get("JobID", "")
        if job_id and "." not in job_id:
            previous_rss = rows.get(job_id, {}).get("MaxRSS")
            rows[job_id] = row
            if previous_rss and not rows[job_id].get("MaxRSS"):
                rows[job_id]["MaxRSS"] = previous_rss
        elif job_id:
            parent = job_id.split(".", 1)[0]
            existing = rows.setdefault(parent, {key: "" for key in row})
            for field in ("State", "MaxRSS", "ElapsedRaw", "TotalCPU", "NodeList"):
                if row.get(field) and not existing.get(field):
                    existing[field] = row[field]
    return rows
