#!/bin/bash
#SBATCH --job-name=train_sasrec
#SBATCH --output=scripts/slurm/train_sasrec_%j.log
#SBATCH --error=scripts/slurm/train_sasrec_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate recsys

export PYTHONPATH=$PWD:$PYTHONPATH

# Define category to run (default to Beauty, but can be overridden)
CATEGORY="Beauty"
EXPERIMENT="sasrec"
TASKNAME="SASRec"

echo "========================================"
echo ">>> Training ${TASKNAME} on ${CATEGORY}"
echo "========================================"

python src/train.py experiment=${EXPERIMENT} \
    data.datasets.category=${CATEGORY}

echo "----------------------------------------"
echo ">>> Training complete. Resolving latest run for evaluation..."
echo "----------------------------------------"

# Resolve latest run directory under outputs/checkpoints/marius/
MARIUS_BASE_DIR="outputs/checkpoints/marius"
LATEST_RUN_DIR=$(ls -1d "${MARIUS_BASE_DIR}/${TASKNAME}"_*/ 2>/dev/null | sort | tail -n 1)

if [ -z "$LATEST_RUN_DIR" ]; then
    echo "Error: No checkpoint directory found for ${TASKNAME} in ${MARIUS_BASE_DIR}"
    exit 1
fi

RUN=$(basename "${LATEST_RUN_DIR%/}")
echo "Evaluating latest run: ${RUN}"

python src/test.py run_directory="${RUN}"

echo "All complete for category ${CATEGORY}!"
