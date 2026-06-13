#!/bin/bash
#SBATCH --job-name=run_test
#SBATCH --output=scripts/slurm/run_test_%j.log
#SBATCH --error=scripts/slurm/run_test_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate recsys

export PYTHONPATH=$PWD:$PYTHONPATH
export WANDB_MODE=offline

MARIUS_BASE_DIR="outputs/checkpoints/marius"
DEBUG=false

for RUN_DIR in ${MARIUS_BASE_DIR}/*/; do
    RUN_DIRECTORY=$(basename ${RUN_DIR%/})
    echo "========================================"
    echo ">>> Testing: ${RUN_DIRECTORY}"
    echo "========================================"

    python src/run_test.py \
        run_directory=${RUN_DIRECTORY} \
        debug=${DEBUG}

    if [ $? -ne 0 ]; then
        echo "[WARNING] Failed for ${RUN_DIRECTORY}, continuing..."
    fi
done

echo "All runs complete."