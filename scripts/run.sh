#!/bin/bash
#SBATCH --job-name=cosette_training
#SBATCH --output=slurm/cosette_training_%j.log
#SBATCH --error=slurm/cosette_training_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1


export RAY_TRAIN_V2_ENABLED=1

module purge
module load 2025
module load Anaconda3/2025.06-1
module load CUDA/12.4.0

source activate recsys

cd ..

python data_scripts/0_raw_to_parquet.py
python data_scripts/1_make_embeddings.py
python data_scripts/2_train_cosette.py


