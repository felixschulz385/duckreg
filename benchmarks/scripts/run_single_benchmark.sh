#!/bin/bash
#SBATCH --job-name=dr_bench
#SBATCH --output=./logs/slurm-%j.log
#SBATCH --error=./logs/slurm-%j.err
#SBATCH --partition=scicore
#SBATCH --time=0-02:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# Arguments passed from orchestrate.py:
#   $1  package        (duckreg | pyfixest)
#   $2  model_type     (pooled | fe | iv)
#   $3  N
#   $4  K
#   $5  nFE1
#   $6  nFE2
#   $7  output_dir     absolute path to benchmarks/results/<run_id>
#   $8  vcov           (HC1 | CRV1, default: HC1)
#   $9  memory_limit   DuckDB memory limit for duckreg (default: 16GB)
#   $10 threads        DuckDB thread count for duckreg (default: 1)
#   (cpus-per-task is set on the sbatch command line by orchestrate.py)

PACKAGE="$1"
MODEL_TYPE="$2"
N="$3"
K="$4"
NFE1="$5"
NFE2="$6"
OUTPUT_DIR="$7"
VCOV="${8:-HC1}"
MEMORY_LIMIT="${9:-16GB}"
THREADS="${10:-1}"

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

# Resolve benchmarks root.  When SLURM copies the script to
# /var/spool/slurm/scripts/, dirname $0 points there instead of the repo.
# Trust SLURM_SUBMIT_DIR when available; otherwise derive from the real path.
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
    WD="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(dirname "$(realpath "$0")")"
    WD="$(realpath "$SCRIPT_DIR/..")"
fi
cd "${WD}" || exit 1

mkdir -p "${OUTPUT_DIR}"

echo "Job ${SLURM_JOB_ID}: ${PACKAGE} ${MODEL_TYPE} N=${N} K=${K} nFE=${NFE1} vcov=${VCOV} threads=${THREADS}"
echo "  output_dir   : ${OUTPUT_DIR}"
echo "  memory_limit : ${MEMORY_LIMIT}"
echo "Started at $(date)"

python scripts/run_single.py \
    --package      "${PACKAGE}" \
    --model-type   "${MODEL_TYPE}" \
    --N            "${N}" \
    --K            "${K}" \
    --nFE1         "${NFE1}" \
    --nFE2         "${NFE2}" \
    --vcov         "${VCOV}" \
    --threads      "${THREADS}" \
    --memory-limit "${MEMORY_LIMIT}" \
    --output-dir   "${OUTPUT_DIR}"

EXIT_CODE=$?
echo "Finished at $(date) (exit code ${EXIT_CODE})"
exit ${EXIT_CODE}
