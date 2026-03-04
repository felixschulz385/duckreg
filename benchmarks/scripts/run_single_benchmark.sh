#!/bin/bash
#SBATCH --job-name=dr_bench
#SBATCH --output=./log/slurm-%j.log
#SBATCH --error=./log/slurm-%j.err
#SBATCH --partition=scicore
#SBATCH --time=0-00:30:00
#SBATCH --qos=30min
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# Arguments passed from orchestrate.py:
#   $1  package      (duckreg | pyfixest)
#   $2  model_type   (pooled | fe | iv)
#   $3  N
#   $4  K
#   $5  nFE1
#   $6  nFE2
#   $7  output_dir   (optional, default: results)

PACKAGE="$1"
MODEL_TYPE="$2"
N="$3"
K="$4"
NFE1="$5"
NFE2="$6"
OUTPUT_DIR="${7:-results}"

# Activate conda environment
eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate gnt

# compute path to benchmarks root (one level up from this script)
SCRIPT_DIR="$(dirname "$0")"
WD="$(realpath "$SCRIPT_DIR/..")"
cd "${WD}"

mkdir -p "${WD}/${OUTPUT_DIR}"
mkdir -p "${WD}/log"

echo "Job ${SLURM_JOB_ID}: ${PACKAGE} ${MODEL_TYPE} N=${N} K=${K} nFE=${NFE1}"
echo "Started at $(date)"

python run_single.py \
    --package      "${PACKAGE}" \
    --model-type   "${MODEL_TYPE}" \
    --N            "${N}" \
    --K            "${K}" \
    --nFE1         "${NFE1}" \
    --nFE2         "${NFE2}" \
    --output-dir   "${OUTPUT_DIR}"

EXIT_CODE=$?
echo "Finished at $(date) (exit code ${EXIT_CODE})"
exit ${EXIT_CODE}
