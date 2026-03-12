# DuckReg Benchmarks


## Key components

- `performance_benchmark.ipynb` – notebook that checks for
  `benchmark_results_large.csv` (in `results/`), triggers
  `scripts/orchestrate.py` if necessary, and visualises the results.

- `scripts/orchestrate.py` – builds the full grid of parameter combinations,
  submits one `scripts/run_single_benchmark.sh` job per combo (throttled to a
  configurable maximum), tracks submissions in `results/manifest.csv`, and upon
  completion calls `scripts/collect_results.py` to merge JSON outputs.

- `scripts/run_single_benchmark.sh` – SLURM wrapper invoked by the orchestrator.
  It activates the conda environment and runs `run_single.py` with the
  arguments supplied by `orchestrate.py`.

- `scripts/run_single.py` – generates synthetic data, runs either `duckreg` or
  `pyfixest` for a single configuration, times the execution, and writes a
  JSON record to the chosen results directory.

- `scripts/collect_results.py` – aggregates the individual JSON files from the
  results directory into one CSV, flags missing runs / OOM kills by
  scanning `log/`, and is used by both the orchestrator and the notebook.

- `scripts/slurm_benchmark.sh` – optional top‑level SLURM script that simply
  launches `orchestrate.py` in one job.  Pass an output directory name
  (default `results`) if you want to override the location of the JSON
  outputs.

- `benchmark_results_large.csv` – the merged results file produced by the
  orchestrator; this is what the notebook reads.  It is generated, not
  edited manually.

- `log/` – directory containing SLURM stdout/err logs for each per‑combo
  job.  Useful for diagnosing OOM kills.

- `results/` (or other directory you specify) – holds the per‑run JSON
  result files; these can be deleted once `benchmark_results_large.csv`
  exists.

## Typical usage

```bash
# submit orchestrated grid via SLURM
sbatch scripts/slurm_benchmark.sh
# or run the orchestrator directly on a login node
python scripts/orchestrate.py --results-dir results

# once finished, open the notebook to explore or rerun
jupyter notebook performance_benchmark.ipynb
```

## Cleaning up

After collecting results you can safely remove:

- `results/*.json` (intermediate files)
- `manifest.csv` if you want to restart from scratch
- `log/slurm-*.err` if you don’t need them

The remaining Python scripts and the notebook are kept for future
benchmark runs.

