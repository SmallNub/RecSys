#!/bin/bash
#SBATCH --job-name=process_data
#SBATCH --output=scripts/slurm/process_data_%j.log
#SBATCH --error=scripts/slurm/process_data_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1


export RAY_TRAIN_V2_ENABLED=1

module purge
module load 2025
module load Anaconda3/2025.06-1
module load CUDA/12.4.0

source activate recsys

export PYTHONPATH=/gpfs/home4/scur1207/RecSys:$PYTHONPATH

python scripts/download_data.py --year 2014 --categories Beauty Video_Games

PYTHONPATH=. python data_scripts/0_raw_to_parquet.py --config-name 0_raw_to_parquet_2014 categories=[Beauty] paths.skip_download=true
PYTHONPATH=. python data_scripts/0_raw_to_parquet.py --config-name 0_raw_to_parquet_2014 categories=[Video_Games] paths.skip_download=true

CUDA_VISIBLE_DEVICES=0 python data_scripts/1_make_embeddings.py category=Beauty num_gpus=1
CUDA_VISIBLE_DEVICES=1 python data_scripts/1_make_embeddings.py category=Video_Games num_gpus=1

CUDA_VISIBLE_DEVICES=0 python data_scripts/2_train_cosette.py \
  data.category=Beauty optim.batch_size=256 optim.epochs=1000 optim.eval_step=100 optim.dropout_prob=0.1

CUDA_VISIBLE_DEVICES=1 python data_scripts/2_train_cosette.py \
  data.category=Video_Games optim.batch_size=256 optim.epochs=1000 optim.eval_step=100 optim.dropout_prob=0.1

PYTHONPATH=. python data_scripts/3_remove_colisions.py data.category=Beauty
PYTHONPATH=. python data_scripts/3_remove_colisions.py data.category=Video_Games

BEAUTY_QUANT=$(python -c "
from pathlib import Path
base = Path('/tmp/cosette')
runs = [d for d in base.iterdir() if d.is_dir() and 'Beauty' in d.name]
print(max(runs, key=lambda d: d.name).name)
")
VIDEO_GAMES_QUANT=$(python -c "
from pathlib import Path
base = Path('/tmp/cosette')
runs = [d for d in base.iterdir() if d.is_dir() and 'Video_Games' in d.name]
print(max(runs, key=lambda d: d.name).name)
")

RAY_TRAIN_V2_ENABLED=1 python src/train.py experiment=marius_small \
  data.ray_datasets.category=Beauty data.ray_datasets.quant_id=${BEAUTY_QUANT}-col ray.run_config.storage_path=

RAY_TRAIN_V2_ENABLED=1 python src/train.py experiment=marius_small \
  data.ray_datasets.category=Video_Games data.ray_datasets.quant_id=${VIDEO_GAMES_QUANT}-col

BEAUTY_RUN=$(python -c "
from pathlib import Path
base = Path('/gpfs/home4/scur1207/RecSys/models/')
runs = [d for d in base.iterdir() if d.is_dir() and 'Beauty' in d.name]
print(max(runs, key=lambda d: d.name).name)
")

VIDEO_GAMES_RUN=$(python -c "
from pathlib import Path
base = Path('/gpfs/home4/scur1207/RecSys/models/')
runs = [d for d in base.iterdir() if d.is_dir() and 'Video_Games' in d.name]
print(max(runs, key=lambda d: d.name).name)
")

python src/test.py run_directory=${BEAUTY_RUN}
python src/test.py run_directory=${VIDEO_GAMES_RUN}