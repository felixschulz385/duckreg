# DuckReg benchmark suite

This suite produces reproducible DuckReg-versus-pyfixest comparisons. It uses
deterministic workloads, isolated package processes, numerical parity checks,
and immutable run manifests. Benchmark outputs are written below `runs/` and
are ignored by Git.

Create the dedicated Python 3.11 benchmark environment with:

```bash
conda env create -f benchmarks/environment.yml
conda activate duckreg-bench
```

## Profiles

| Profile | Purpose | Measurements |
| --- | --- | --- |
| `smoke` | Fast local pooled, FE, and IV coverage for HC1 and CRV1 | One cold fit |
| `standard` | 28 controlled scaling, width, compression, FE-cardinality, and threading scenarios | One cold fit and five steady fits |
| `stress` | Opt-in 100M-row and DuckReg file-backed cases | One cold fit |
| `diagnostics` | DuckReg MAP, cluster-score, and compression components | One cold measurement |

The standard profile changes one performance dimension at a time. Its scaling
track covers 100K, 1M, and 10M rows. Additional scenarios isolate 20-covariate
width, compressible inputs, low/high FE cardinality, and four-thread execution.

## Commands

Run the local smoke profile:

```bash
python benchmarks/bench.py run --profile smoke
```

Submit resource-bucketed SLURM arrays:

```bash
python benchmarks/bench.py submit \
  --profile standard \
  --partition scicore \
  --max-concurrent 4
```

`submit` prints the run ID. Reusing it resumes a run and submits only scenarios
whose package result is absent or unsuccessful:

```bash
python benchmarks/bench.py submit --profile standard --run-id RUN_ID
```

Collect or regenerate a report independently:

```bash
python benchmarks/bench.py collect --run-id RUN_ID --report
python benchmarks/bench.py report --run-id RUN_ID
```

Use `--dry-run` with `submit` to inspect the generated `sbatch` commands.

## Measurement contract

- Package imports, formula construction, and deterministic data generation are
  outside the fit timer. Data-generation time is reported separately.
- Both packages receive the same pandas input in comparison tracks. File-backed
  scenarios are DuckReg-only and are reported separately.
- Package workers run in isolated processes with the same thread environment.
  On SLURM, they run as independent `srun` steps on the same allocation.
- The standard profile reports steady-state median and interquartile range.
  Cold-start time remains available in the raw results.
- Peak process RSS and SLURM accounting data are recorded when available.
- Pooled/FE coefficients and standard errors use relative tolerances of `1e-6`
  and `1e-3`; IV uses `1e-3` and `1e-2`. Observation counts must match exactly.
  Invalid comparisons retain their timings but are excluded from speedups.

The IV generator is genuinely endogenous: treatment contains part of the
outcome disturbance and is identified by an independent strong instrument.
Balanced and sparse FE structures and continuous and compressible covariates
are represented by the profiles.

## Run layout

Each `runs/<run_id>/` directory contains:

- `run.json`: Git, Python, package, OS, and hardware provenance.
- `tasks.jsonl`: immutable expanded scenarios and resource assignments.
- `raw/<scenario>/<package>.json`: atomic package-level measurements.
- `results.csv` and `summary.csv`: normalized raw and aggregated results.
- `report.html` and `report_assets/`: deterministic tables and plots.
- `logs/`: SLURM stdout/stderr.

Statuses distinguish missing or malformed output, Python exceptions, numerical
mismatches, convergence failures, timeouts, and OOM failures. Reports calculate
speedups only for complete numerically valid package pairs and use geometric
means for track-level summaries.

## SLURM behavior

Scenarios are grouped by CPU, memory, and time requirements. Defaults are 32 GB
and one hour through 1M rows, 64 GB and four hours at 10M rows, and 192 GB and
six hours at 100M rows. The collection job uses an `afterany` dependency, so no
coordinator job polls the queue. Run-specific node-local scratch prevents
DuckDB spill-file collisions.

The JSON files in `configs/` are the versioned workload definitions. Change or
add a profile rather than editing the runner for a new experiment.
