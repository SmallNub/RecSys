#!/bin/bash
#SBATCH --job-name=train_pipeline
#SBATCH --output=scripts/slurm/train_pipeline_%j.log
#SBATCH --error=scripts/slurm/train_pipeline_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

set -e

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate recsys

export PYTHONPATH=$PWD:$PYTHONPATH
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1

# CATEGORIES=("Beauty" "Sports_and_Outdoors" "Toys_and_Games" "Movies_and_TV" "Video_Games")
CATEGORIES=("Arts_Crafts_and_Sewing" "Cell_Phones_and_Accessories" "Health_and_Household")

N_RUNS=5
EXPERIMENT="marius"
TASKNAME="MARIUS"
COSETTE_BASE_DIR="outputs/checkpoints/cosette"
MARIUS_BASE_DIR="outputs/checkpoints/marius"

for CATEGORY in "${CATEGORIES[@]}"; do
    echo "========================================"
    echo ">>> CATEGORY: ${CATEGORY}"
    echo "========================================"

    for RUN_IDX in $(seq 1 ${N_RUNS}); do
        SEED=${RUN_IDX}
        echo "----------------------------------------"
        echo ">>> ${CATEGORY} - Run ${RUN_IDX}/${N_RUNS}"
        echo "----------------------------------------"

        # Step 1: Train COSETTE
        python data_scripts/2_train_cosette.py cosette=default \
            data.category=${CATEGORY} \
            marker=run${RUN_IDX}

        # Step 2: Remove collisions (reconstructs quant name internally via get_latest_timestamp_dir)
        python data_scripts/3_remove_colisions.py \
            data.category=${CATEGORY} \
            marker=run${RUN_IDX}

        # Step 3: Resolve quant_id from parquet output — name is COSETTE_SIGLIP_{category}_{timestamp}_run{N}-col
        QUANT=$(python -c "
from pathlib import Path
base = Path('datasets/data/embeddings/sentence-t5-xl/${CATEGORY}/')
runs = [f.stem for f in base.glob('COSETTE_SIGLIP_${CATEGORY}_*.parquet') if 'col' not in f.stem and '_model' not in f.stem]
print(max(runs))
")
        echo "[${CATEGORY}] Run ${RUN_IDX}: Using quant_id=${QUANT}-col"

        # Step 4: Train MARIUS
        python src/train.py experiment=${EXPERIMENT} \
            data.datasets.category=${CATEGORY} \
            data.datasets.quant_id=${QUANT}-col \
            seed=${SEED}

        # Step 5: Resolve latest MARIUS run directory
        LATEST_MARIUS_RUN=$(ls -1d "${MARIUS_BASE_DIR}/${TASKNAME}"_*/ 2>/dev/null | sort | tail -n 1)
        if [ -z "$LATEST_MARIUS_RUN" ]; then
            echo "Error: No MARIUS run found in ${MARIUS_BASE_DIR}"
            exit 1
        fi
        RUN=$(basename "${LATEST_MARIUS_RUN%/}")
        echo "[${CATEGORY}] Run ${RUN_IDX}: Evaluating run_directory=${RUN}"

        # Step 6: Test
        python src/test.py run_directory="${RUN}"

    done
done

echo "All categories and runs complete."