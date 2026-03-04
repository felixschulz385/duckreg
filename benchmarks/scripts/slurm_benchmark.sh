#!/bin/bash
#SBATCH --job-name=duckreg_bench
#SBATCH --output=./log/slurm-%j.log
#SBATCH --error=./log/slurm-%j.err
#SBATCH --partition=scicore
#SBATCH --time=0-12:00:00
#SBATCH --qos=1day
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

# compute root based on script location and cd there
SCRIPT_DIR="$(dirname "$0")"
ROOT_DIR="$(realpath "$SCRIPT_DIR/..")"
cd "${ROOT_DIR}"

mkdir -p "${ROOT_DIR}/log"

# we accept an optional results directory argument; default is the standard CSV
OUTPUT_DIR="${1:-results}"

# orchestrate.py creates its own log directory and manifest; it writes JSON
# results under $OUTPUT_DIR and then merges into a CSV named
# benchmark_results_large.csv inside benchmarks/ when finished.

echo "Starting orchestration at $(date)"
echo "Results directory: ${OUTPUT_DIR}"

python scripts/orchestrate.py --results-dir "${OUTPUT_DIR}"

echo "Orchestration finished at $(date)"
