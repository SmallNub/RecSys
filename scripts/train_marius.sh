#!/bin/bash
#SBATCH --job-name=train_marius
#SBATCH --output=scripts/slurm/train_marius%j.log
#SBATCH --error=scripts/slurm/train_marius%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate recsys

# export $(cat .env | xargs)
export PYTHONPATH=$PWD:$PYTHONPATH

# CATEGORIES=(
#     "Arts_Crafts_and_Sewing"
#     "Beauty"
#     "Toys_and_Games"
#     "Video_Games"
# )

CATEGORY="Beauty"
EXPERIMENT="MARIUS_small"

COSETTE_BASE_DIR="outputs/checkpoints/cosette"
LATEST_COSETTE_DIR=$(ls -1d ${COSETTE_BASE_DIR}/*/ 2>/dev/null | sort | tail -n 1)

if [ -z "$LATEST_COSETTE_DIR" ]; then
    echo "Error: No timestamp directory found inside $COSETTE_BASE_DIR"
    exit 1
fi

COSETTE_TIMESTAMP=$(basename ${LATEST_COSETTE_DIR})
echo "Using latest Cosette timestamp: ${COSETTE_TIMESTAMP}"

QUANT="COSETTE_${CATEGORY}_${COSETTE_TIMESTAMP}"

# python src/train.py experiment=${EXPERIMENT} \
#     data.datasets.category=${CATEGORY} data.datasets.quant_id=${QUANT}-col

TASK_NAME="marius_small"
MARIUS_BASE_DIR="outputs/checkpoints/marius"
LATEST_MARIUS_RUN=$(ls -1d ${MARIUS_BASE_DIR}/${TASK_NAME}_*/ 2>/dev/null | sort | tail -n 1)

if [ -z "$LATEST_MARIUS_RUN" ]; then
    echo "Error: No timestamp directory found for task '${TASK_NAME}' in $MARIUS_BASE_DIR"
    exit 1
fi

RUN=$(basename ${LATEST_MARIUS_RUN})
echo "Running evaluation on latest experiment directory: ${RUN}"

python src/test.py run_directory=${RUN}
