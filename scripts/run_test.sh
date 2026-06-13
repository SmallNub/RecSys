#!/bin/bash
#SBATCH --job-name=run_test
#SBATCH --output=scripts/slurm/run_test_%j.log
#SBATCH --error=scripts/slurm/run_test_%j.err
#SBATCH --time=2:00:00
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

# Set to the run directory you want to test
RUN_DIRECTORY="MARIUS_small_260611_144729"

# Set to true for debug mode (one forward pass, no full metrics)
DEBUG=false

python src/run_test.py \
    run_directory=${RUN_DIRECTORY} \
    debug=${DEBUG}