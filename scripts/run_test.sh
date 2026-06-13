#!/bin/bash
#SBATCH --job-name=run_test
#SBATCH --output=scripts/slurm/run_test_%j.log
#SBATCH --error=scripts/slurm/run_test_%j.err
#SBATCH --time=5:00:00
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

# Set to a specific run directory to test just one, or leave empty to loop all
RUN_DIRECTORY=""

run_test() {
    local RUN_DIR=$1
    echo "========================================"
    echo ">>> Testing: ${RUN_DIR}"
    echo "========================================"

    python src/test.py run_directory=${RUN_DIR}

    if [ $? -ne 0 ]; then
        echo "[WARNING] Failed for ${RUN_DIR}, continuing..."
    fi
}

if [ -n "${RUN_DIRECTORY}" ]; then
    # Single checkpoint mode
    run_test ${RUN_DIRECTORY}
else
    # Loop through all checkpoints
    for RUN_DIR in ${MARIUS_BASE_DIR}/*/; do
        run_test $(basename ${RUN_DIR%/})
    done
fi

echo "All runs complete."