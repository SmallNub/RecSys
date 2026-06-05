#!/bin/bash
#SBATCH --job-name=process_data
#SBATCH --output=scripts/slurm/process_data_%j.log
#SBATCH --error=scripts/slurm/process_data_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1


export RAY_TRAIN_V2_ENABLED=1

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate recsys

export PYTHONPATH=/gpfs/home4/scur1207/RecSys:$PYTHONPATH

python scripts/download_data.py --year 2014 --categories Beauty Video_Games

PYTHONPATH=. python data_scripts/0_raw_to_parquet.py --config-name 0_raw_to_parquet_2014 categories=[Beauty] paths.skip_download=true
PYTHONPATH=. python data_scripts/0_raw_to_parquet.py --config-name 0_raw_to_parquet_2014 categories=[Video_Games] paths.skip_download=true

