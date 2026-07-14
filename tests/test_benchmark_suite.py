import json
import sys

import numpy as np
import pandas as pd
import pytest

if sys.version_info < (3, 10):
    pytest.skip("The benchmark environment requires Python 3.11", allow_module_level=True)

from benchmarks import benchlib


def _case(**overrides):
    case = {
        "track": "test",
        "model": "pooled",
        "N": 1000,
        "K": 2,
        "nFE1": 20,
        "nFE2": 5,
        "fe_structure": "balanced",
        "data_regime": "continuous",
        "input_mode": "memory",
        "vcov": "HC1",
        "threads": 1,
        "seed": 7,
        "packages": ["duckreg", "pyfixest"],
        "warmups": 0,
        "repetitions": 1,
    }
    case.update(overrides)
    return case


def _success_result(case, package):
    return {
        "schema_version": 1,
        "scenario_id": case["scenario_id"],
        "package": package,
        "status": "success",
        "case": case,
        "measurements": [{"phase": "cold", "iteration": 0,
                          "wall_seconds": 1.0, "cpu_seconds": 0.9}],
        "coefficients": {"x1": 1.0},
        "standard_errors": {"x1": 0.1},
        "nobs": case["N"],
        "data_digest": "same-data",
    }


def test_shipped_profile_sizes_and_unique_ids():
    expected = {"ci": 3, "smoke": 6, "standard": 28, "stress": 13, "diagnostics": 4}
    for name, count in expected.items():
        scenarios = benchlib.expand_profile(benchlib.load_profile(name))
        assert len(scenarios) == count
        assert len({case["scenario_id"] for case in scenarios}) == count


def test_identity_includes_second_fe_dimension():
    profile = {
        "schema_version": 1,
        "seed": 7,
        "packages": ["duckreg", "pyfixest"],
        "tracks": [{
            "kind": "explicit",
            "scenarios": [_case(nFE2=5), _case(nFE2=6)],
        }],
    }
    scenarios = benchlib.expand_profile(profile)
    assert len(scenarios) == 2
    assert scenarios[0]["scenario_id"] != scenarios[1]["scenario_id"]


def test_generator_is_deterministic_and_iv_is_endogenous():
    first, truth, digest1, _ = benchlib.generate_data(_case(N=5000))
    second, _, digest2, _ = benchlib.generate_data(_case(N=5000))
    pd.testing.assert_frame_equal(first, second)
    assert digest1 == digest2
    assert truth["D"] == 1.5
    structural_component = first["D"] - 0.8 * first["Z"]
    ols_error_proxy = first["y"] - 1.5 * first["D"]
    assert np.corrcoef(structural_component, ols_error_proxy)[0, 1] > 0.1


def test_resource_buckets():
    assert benchlib.resources_for(_case(N=1_000_000)).memory_gb == 32
    assert benchlib.resources_for(_case(N=10_000_000)).memory_gb == 64
    stress = benchlib.resources_for(_case(N=100_000_000, threads=4))
    assert (stress.memory_gb, stress.time_minutes, stress.cpus) == (192, 360, 4)


def test_atomic_result_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(benchlib, "RUNS_DIR", tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    case = _case(scenario_id="abc", resources={"memory_gb": 32, "time_minutes": 60, "cpus": 1},
                 resource_bucket="c1_m32_t60", schema_version=1)
    (root / "tasks.jsonl").write_text(json.dumps(case) + "\n")
    path = benchlib.result_path("run", "abc", "duckreg")
    benchlib.atomic_json(path, _success_result(case, "duckreg"))
    pending = benchlib.pending_scenarios("run")
    assert pending == [case]
    benchlib.atomic_json(
        benchlib.result_path("run", "abc", "pyfixest"),
        _success_result(case, "pyfixest"),
    )
    assert benchlib.pending_scenarios("run") == []


def test_comparison_detects_mismatch():
    left = {"case": {"model": "fe"}, "nobs": 10,
            "coefficients": {"x1": 1.0}, "standard_errors": {"x1": 0.1}}
    right = {"case": {"model": "fe"}, "nobs": 10,
             "coefficients": {"x1": 1.1}, "standard_errors": {"x1": 0.1}}
    result = benchlib._comparison(left, right)
    assert not result["valid"]
    assert result["reason"] == "numerical_mismatch"


def test_duckreg_result_adapter_supports_iv_result_nobs():
    class FakeResults:
        n_obs = 123

    class FakeIV:
        results = FakeResults()
        resolved_fitter = "duckdb"

        def coef(self):
            return pd.Series({"D": 1.5})

        def se(self):
            return pd.Series({"D": 0.1})

    coefficients, standard_errors, nobs, _ = benchlib._extract_result("duckreg", FakeIV())
    assert coefficients == {"D": 1.5}
    assert standard_errors == {"D": 0.1}
    assert nobs == 123


def test_slurm_array_command_and_sacct_parser(tmp_path, monkeypatch):
    monkeypatch.setattr(benchlib, "RUNS_DIR", tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    cases = []
    for index in range(2):
        case = _case(
            scenario_id=f"case{index}",
            resources={"memory_gb": 32, "time_minutes": 60, "cpus": 1},
            resource_bucket="c1_m32_t60",
            schema_version=1,
        )
        cases.append(case)
    (root / "tasks.jsonl").write_text("".join(json.dumps(case) + "\n" for case in cases))
    command = benchlib.slurm_array_command("run", cases, 4, "scicore")
    assert "--array=0,1%4" in command
    assert "--mem=32G" in command
    rows = benchlib.parse_sacct("JobID|State|MaxRSS\n123|COMPLETED|\n123.batch|COMPLETED|10G\n")
    assert rows["123"]["MaxRSS"] == "10G"


def test_collection_reports_missing_and_malformed_results(tmp_path, monkeypatch):
    monkeypatch.setattr(benchlib, "RUNS_DIR", tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    case = _case(
        scenario_id="broken",
        resources={"memory_gb": 32, "time_minutes": 60, "cpus": 1},
        resource_bucket="c1_m32_t60",
        schema_version=1,
    )
    (root / "tasks.jsonl").write_text(json.dumps(case) + "\n")
    malformed = benchlib.result_path("run", "broken", "duckreg")
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{not-json")
    frame = benchlib.collect_results("run")
    statuses = dict(zip(frame["package"], frame["status"]))
    assert statuses == {"duckreg": "parse_error", "pyfixest": "missing"}


def test_collection_uses_sacct_to_classify_missing_oom(tmp_path, monkeypatch):
    monkeypatch.setattr(benchlib, "RUNS_DIR", tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    case = _case(
        packages=["duckreg"],
        scenario_id="oom-case",
        resources={"memory_gb": 32, "time_minutes": 60, "cpus": 1},
        resource_bucket="c1_m32_t60",
        schema_version=1,
    )
    (root / "tasks.jsonl").write_text(json.dumps(case) + "\n")
    benchlib.atomic_json(root / "submissions.json", {
        "scenarios": {"oom-case": {"accounting_job_id": "123_0"}},
    })

    class Completed:
        returncode = 0
        stdout = "JobID|State|MaxRSS\n123_0|OUT_OF_MEMORY|\n123_0.batch|OUT_OF_MEMORY|31G\n"

    monkeypatch.setattr(benchlib.shutil, "which", lambda name: "/usr/bin/sacct")
    monkeypatch.setattr(benchlib.subprocess, "run", lambda *args, **kwargs: Completed())
    frame = benchlib.collect_results("run")
    assert frame.loc[0, "status"] == "oom"
    assert frame.loc[0, "slurm_max_rss"] == "31G"
