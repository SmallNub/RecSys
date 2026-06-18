#!/bin/bash
#SBATCH --job-name=process_data
#SBATCH --output=scripts/slurm/process_data_%j.log
#SBATCH --error=scripts/slurm/process_data_%j.err
#SBATCH --time=4:00:00
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

# CATEGORIES=("All_Beauty" "Sports_and_Outdoors" "Toys_and_Games" "Movies_and_TV" "Video_Games")
CATEGORIES=("Arts_Crafts_and_Sewing", "Sports_and_Outdoors", "Health_and_Household")

# Join array with commas for hydra list format
JOINED_CATEGORIES=$(IFS=, ; echo "${CATEGORIES[*]}")

python scripts/download_data.py --year 2023 --categories "${CATEGORIES[@]}"

python data_scripts/0_raw_to_parquet.py --config-name 0_raw_to_parquet \
    categories="[$JOINED_CATEGORIES]" \
    paths.skip_download=true

for CATEGORY in "${CATEGORIES[@]}"; do
    echo ">>> Embedding: ${CATEGORY}"
    python data_scripts/1_make_embeddings.py category=${CATEGORY} num_gpus=1
done