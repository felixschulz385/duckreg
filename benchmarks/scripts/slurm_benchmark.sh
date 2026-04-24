#!/bin/bash
#SBATCH --job-name=duckreg_bench
#SBATCH --output=./logs/slurm-%j.log
#SBATCH --error=./logs/slurm-%j.err
#SBATCH --partition=scicore
#SBATCH --time=0-12:00:00
#SBATCH --qos=1day
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

# Determine benchmarks root.  When SLURM copies the script to
# /var/spool/slurm/scripts the automatic `dirname $0` trick points there,
# which is not what we want.  Instead trust the submission directory if
# available, otherwise fall back to the hard-coded repository path.
ROOT_DIR="${SLURM_SUBMIT_DIR:-/scicore/home/meiera/schulz0022/projects/duckreg/benchmarks}"
cd "${ROOT_DIR}" || exit 1

mkdir -p "${ROOT_DIR}/logs"

# Optional: pass a run ID to resume or name a specific run.
#   sbatch slurm_benchmark.sh [run_id]
RUN_ID_ARG=""
if [[ -n "$1" ]]; then
    RUN_ID_ARG="--run-id $1"
fi

echo "Starting orchestration at $(date)"
[[ -n "${RUN_ID_ARG}" ]] && echo "Run ID arg: ${RUN_ID_ARG}"

python scripts/orchestrate.py ${RUN_ID_ARG}

echo "Orchestration finished at $(date)"
