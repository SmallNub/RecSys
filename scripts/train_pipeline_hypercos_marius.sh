#!/bin/bash
#SBATCH --job-name=train_hyper_pipeline
#SBATCH --output=scripts/slurm/train_hyper_pipeline_%j.log
#SBATCH --error=scripts/slurm/train_hyper_pipeline_%j.err
#SBATCH --time=65:00:00
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
export HYDRA_FULL_ERROR=1

DATASET_YEAR=${1:-2014}
CATEGORIES=("Beauty" "Sports_and_Outdoors" "Video_Games")
C_VALUES=(1.0 2.0 3.0)

N_RUNS=5
EXPERIMENT="marius_small"
TASKNAME="MARIUS_small"

DATASET_ROOT="${PWD}/datasets/"
COSETTE_BASE_DIR="outputs/checkpoints/cosette-${DATASET_YEAR}-corrected-hyper"
MARIUS_BASE_DIR="outputs/checkpoints/marius-${DATASET_YEAR}-corrected-hyper"

echo "========================================"
echo ">>> DATASET YEAR: ${DATASET_YEAR}"
echo ">>> DATASET ROOT: ${DATASET_ROOT}"
echo ">>> COSETTE BASE: ${COSETTE_BASE_DIR}"
echo ">>> MARIUS BASE:  ${MARIUS_BASE_DIR}"
echo "========================================"

for CATEGORY in "${CATEGORIES[@]}"; do
    echo "========================================"
    echo ">>> CATEGORY: ${CATEGORY}"
    echo "========================================"

    for C_VAL in "${C_VALUES[@]}"; do
        echo "----------------------------------------"
        echo ">>> CURVATURE (c): ${C_VAL}"
        echo "----------------------------------------"

        for RUN_IDX in $(seq 1 ${N_RUNS}); do
            SEED=${RUN_IDX}
            echo "   >>> ${CATEGORY} | c=${C_VAL} | Run ${RUN_IDX}/${N_RUNS} (Seed: ${SEED})"

            python data_scripts/2_train_cosette.py cosette=hyper \
                data.category=${CATEGORY} \
                model.c=${C_VAL} \
                marker=run${RUN_IDX} \
                seed=${SEED} \
                paths.root=${DATASET_ROOT} \
                ckpt_dir=${COSETTE_BASE_DIR}

            python data_scripts/3_remove_colisions.py cosette=col_hyper \
                data.category=${CATEGORY} \
                marker=run${RUN_IDX} \
                paths.root=${DATASET_ROOT} \
                paths.ckpt_dir=${COSETTE_BASE_DIR}

            QUANT=$(python -c "
from pathlib import Path
base = Path('${DATASET_ROOT}/data/embeddings/sentence-t5-xl/${CATEGORY}/')
runs = [f.stem for f in base.glob('COSETTE_HYPER_${CATEGORY}_*.parquet') if 'col' not in f.stem and '_model' not in f.stem]
print(max(runs))
")
            echo "[${CATEGORY}] c=${C_VAL} Run ${RUN_IDX}: Using quant_id=${QUANT}-col"

            python src/train.py experiment=${EXPERIMENT} \
                data.datasets.category=${CATEGORY} \
                data.datasets.quant_id=${QUANT}-col \
                paths.root=${DATASET_ROOT} \
                paths.model_folder_tplt=${PWD}/${MARIUS_BASE_DIR} \
                seed=${SEED}

            LATEST_MARIUS_RUN=$(ls -1d "${MARIUS_BASE_DIR}/${TASKNAME}"_*/ 2>/dev/null | sort | tail -n 1)
            if [ -z "$LATEST_MARIUS_RUN" ]; then
                echo "Error: No MARIUS run found in ${MARIUS_BASE_DIR}"
                exit 1
            fi
            RUN=$(basename "${LATEST_MARIUS_RUN%/}")
            echo "[${CATEGORY}] c=${C_VAL} Run ${RUN_IDX}: Evaluating run_directory=${RUN}"

            python src/test.py \
                run_directory="${RUN}" \
                paths.root=${DATASET_ROOT} \
                paths.model_folder_tplt=${PWD}/${MARIUS_BASE_DIR}

        done
    done
done

