#!/bin/bash
#SBATCH --job-name=test_sasrec_latest
#SBATCH --output=scripts/slurm/test_sasrec_latest_%j.log
#SBATCH --error=scripts/slurm/test_sasrec_latest_%j.err
#SBATCH --time=00:30:00
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

TASKNAME="SASRec"
MARIUS_BASE_DIR="outputs/checkpoints/marius"

# Find the latest SASRec run directory
LATEST_RUN=$(ls -1d "${MARIUS_BASE_DIR}/${TASKNAME}"_*/ 2>/dev/null | sort | tail -n 1)

if [ -z "$LATEST_RUN" ]; then
    echo "Error: No SASRec run found in ${MARIUS_BASE_DIR}"
    exit 1
fi

RUN=$(basename "${LATEST_RUN%/}")
echo "========================================"
echo ">>> Evaluating latest run: ${RUN}"
echo "========================================"

# Run the evaluation script
python src/test.py run_directory="${RUN}"

echo "Evaluation complete."
