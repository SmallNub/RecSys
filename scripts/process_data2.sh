#!/bin/bash
#SBATCH --job-name=process_data
#SBATCH --output=scripts/slurm/process_data_%j.log
#SBATCH --error=scripts/slurm/process_data_%j.err
#SBATCH --time=0:30:00
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

python scripts/download_data.py --year 2014 --categories Beauty

python data_scripts/0_raw_to_parquet.py --config-name 0_raw_to_parquet_2014 categories=[Beauty] paths.skip_download=true
python data_scripts/1_make_embeddings.py category=Beauty num_gpus=1

