#!/bin/bash
#SBATCH --job-name=train_cosette
#SBATCH --output=scripts/slurm/train_cosette%j.log
#SBATCH --error=scripts/slurm/train_cosette%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

module purge
module load 2025
module load Anaconda3/2025.06-1

source activate recsys

# export $(cat .env | xargs)
export PYTHONPATH=$PWD:$PYTHONPATH
export HYDRA_FULL_ERROR=1

python data_scripts/2_train_cosette.py cosette=default \
  data.category=Beauty

python data_scripts/3_remove_colisions.py data.category=Beauty
