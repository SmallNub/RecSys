#!/bin/bash
#SBATCH --job-name=train_sasrec
#SBATCH --output=scripts/slurm/train_sasrec_%j.log
#SBATCH --error=scripts/slurm/train_sasrec_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate recsys

export PYTHONPATH=$PWD:$PYTHONPATH
export HYDRA_FULL_ERROR=1

# CATEGORIES=("Beauty" "Sports_and_Outdoors" "Toys_and_Games" "Movies_and_TV" "Video_Games")
CATEGORIES=("Arts_Crafts_and_Sewing")
N_RUNS=5
EXPERIMENT="sasrec"
TASKNAME="SASRec"
MARIUS_BASE_DIR="outputs/checkpoints/marius"

for CATEGORY in "${CATEGORIES[@]}"; do
    echo "========================================"
    echo ">>> CATEGORY: ${CATEGORY}"
    echo "========================================"

    for RUN_IDX in $(seq 1 ${N_RUNS}); do
        SEED=${RUN_IDX}
        echo "----------------------------------------"
        echo ">>> ${CATEGORY} - Run ${RUN_IDX}/${N_RUNS} (Seed: ${SEED})"
        echo "----------------------------------------"

        # Step 1: Train SASRec
        python src/train.py experiment=${EXPERIMENT} \
            data.datasets.category=${CATEGORY} \
            seed=${SEED}

        # Step 2: Resolve latest SASRec run directory
        LATEST_RUN=$(ls -1d "${MARIUS_BASE_DIR}/${TASKNAME}"_*/ 2>/dev/null | sort | tail -n 1)
        if [ -z "$LATEST_RUN" ]; then
            echo "Error: No SASRec run found in ${MARIUS_BASE_DIR}"
            exit 1
        fi
        RUN=$(basename "${LATEST_RUN%/}")
        echo "[${CATEGORY}] Run ${RUN_IDX}: Evaluating run_directory=${RUN}"

        # Step 3: Test and append to results
        python src/test.py run_directory="${RUN}"
    done
done

echo "All SASRec categories and runs complete."
