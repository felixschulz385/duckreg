#!/usr/bin/env python3
"""Command-line interface for the DuckReg benchmark suite."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchlib import (
    build_report,
    collect_results,
    create_run,
    execute_scenario,
    execute_worker,
    pending_scenarios,
    read_manifest,
    run_dir,
    submit_run,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproducible DuckReg benchmarks")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run a profile locally")
    run.add_argument("--profile", default="smoke")
    run.add_argument("--run-id")

    submit = commands.add_parser("submit", help="Submit a profile through SLURM arrays")
    submit.add_argument("--profile", default="standard")
    submit.add_argument("--run-id")
    submit.add_argument("--max-concurrent", type=int, default=4)
    submit.add_argument("--partition", default="scicore")
    submit.add_argument("--qos", default=None)
    submit.add_argument("--dry-run", action="store_true")

    collect = commands.add_parser("collect", help="Collect raw task results")
    collect.add_argument("--run-id", required=True)
    collect.add_argument("--report", action="store_true")

    report = commands.add_parser("report", help="Create a deterministic HTML report")
    report.add_argument("--run-id", required=True)

    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--run-id", required=True)
    worker.add_argument("--scenario-id", required=True)
    worker.add_argument("--package", required=True, choices=["duckreg", "pyfixest"])
    worker.add_argument("--scratch", type=Path)

    slurm = commands.add_parser("slurm-worker", help=argparse.SUPPRESS)
    slurm.add_argument("--run-id", required=True)
    slurm.add_argument("--index", required=True, type=int)

    args = parser.parse_args(argv)
    if args.command == "run":
        run_id, scenarios = create_run(args.profile, args.run_id)
        failures = 0
        for index, scenario in enumerate(scenarios, 1):
            if scenario not in pending_scenarios(run_id):
                continue
            print(f"[{index}/{len(scenarios)}] {scenario['scenario_id']} {scenario['track']} ")
            failures += execute_scenario(run_id, scenario)
        frame = collect_results(run_id)
        report_path = build_report(run_id)
        print(f"Run: {run_id}\nResults: {run_dir(run_id) / 'results.csv'}\nReport: {report_path}")
        return 1 if failures or (frame["status"] != "success").any() else 0
    if args.command == "submit":
        if args.max_concurrent < 1:
            parser.error("--max-concurrent must be positive")
        run_id, _ = create_run(args.profile, args.run_id)
        commands_run = submit_run(run_id, args.max_concurrent, args.partition,
                                  args.qos, args.dry_run)
        print(f"Run: {run_id}")
        for command in commands_run:
            print(shlex.join(command))
        return 0
    if args.command == "collect":
        frame = collect_results(args.run_id)
        print(frame.groupby(["package", "status"], dropna=False).size().to_string())
        if args.report:
            print(build_report(args.run_id))
        return 0
    if args.command == "report":
        print(build_report(args.run_id))
        return 0
    if args.command == "worker":
        return execute_worker(args.run_id, args.scenario_id, args.package, args.scratch)
    if args.command == "slurm-worker":
        scenario = read_manifest(args.run_id)[args.index]
        return execute_scenario(args.run_id, scenario, use_srun=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
