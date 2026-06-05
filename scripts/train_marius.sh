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

export $(cat .env | xargs)
export RAY_TRAIN_V2_ENABLED=1
export PYTHONPATH=$PWD:$PYTHONPATH

COSETTE_TIMESTAMP="20260605_193615"
CATEGORY="Beauty"
QUANT="COSETTE_${CATEGORY}_${COSETTE_TIMESTAMP}"
EXPERIMENT="marius_small"

python src/train.py experiment=${EXPERIMENT} \
    data.ray_datasets.category=${CATEGORY} data.ray_datasets.quant_id=${QUANT}-col

BASE_DIR="outputs/checkpoints/marius"

LATEST_RUN=$(ls -1d ${BASE_DIR}/${EXPERIMENT}_*/ 2>/dev/null | sort | tail -n 1)

if [ -z "$LATEST_RUN" ]; then
    echo "Error: No timestamp directory found for experiment '${EXPERIMENT}' in $BASE_DIR"
    exit 1
fi

RUN=${LATEST_RUN}
echo "Running evaluation on latest experiment directory: ${RUN}"

python src/test.py run_directory=${RUN}
