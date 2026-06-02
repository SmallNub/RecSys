import json
import os
import pickle

import fsspec
import hydra
import pytorch_lightning as L
import ray
from omegaconf import OmegaConf

from src.utils.tools import patch_fsspec

FILENAMES = {
    "metrics": "metrics.pkl",
    "config": "config.yaml",
    "checkpoint": "checkpoint.ckpt",
    "progress": "checkpoint_manager_snapshot.json",
}


def get_top_cfg(fs, config):
    print(f"[STEP] Loading from {config.run_directory}")

    root = os.path.join(config.paths.model_folder_tplt, config.run_directory)

    # Get a config file to look for a best metric
    folders = [f for f in fs.ls(root) if "/checkpoint_" in f]
    with fs.open(os.path.join(folders[0], FILENAMES["config"])) as f:
        cfg = OmegaConf.load(f)
    print(cfg.data)

    metric_name = cfg.ray.run_config.checkpoint_config.checkpoint_score_attribute

    # Get the checkpoint manager
    with fs.open(os.path.join(root, FILENAMES["progress"])) as f:
        data = json.load(f)

    best_path = None
    best_metric = float("-inf")
    for ckpt in data["checkpoint_results"]:
        if ckpt["metrics"][metric_name] > best_metric:
            best_metric = ckpt["metrics"][metric_name]
            best_path = ckpt["checkpoint_dir_name"]

    best_checkpoint = os.path.join(root, best_path)
    with fs.open(os.path.join(best_checkpoint, FILENAMES["config"])) as f:
        cfg = OmegaConf.load(f)

    return cfg, best_checkpoint


@ray.remote(num_gpus=0.5, max_retries=0)
def get_metrics(cfg, ckpt_path, split, ray_datasets):
    patch_fsspec()

    # Load datamodule
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, ray_datasets=ray_datasets)
    datamodule.setup(stage="test")  # Load valid and test

    # Load lightning module
    model = hydra.utils.instantiate(cfg["model"])
    n_params = sum(p.numel() for p in model.parameters())
    model.full_hydra_config = cfg

    # Trainer
    trainer = L.Trainer(accelerator="gpu", precision="bf16-mixed")

    fn = trainer.validate if split == "valid" else trainer.test

    metrics = fn(
        model, datamodule=datamodule, ckpt_path=cfg.paths.protocol + "://" + ckpt_path
    )

    metrics[0]["n_params"] = n_params

    return metrics


def to_pickle(fs, path, data):
    with fs.open(path, "wb") as f:
        pickle.dump(data, f)


@hydra.main(version_base="1.3", config_path="../configs", config_name="test")
def main(test_config):
    ray.init()

    patch_fsspec()
    fs = fsspec.filesystem(test_config.paths.protocol)

    # 1. Get the config file of the previous run.
    cfg, best_checkpoint = get_top_cfg(fs, test_config)

    if test_config.enforce_filtering:
        cfg.model.net.filter_preds = True

    # 2. Instantiate ray datasets
    cfg.data.ray_datasets.which = ["valid", "test"]  # Not train.
    ray_datasets = hydra.utils.instantiate(cfg.data.ray_datasets, paths=cfg.paths)

    best_model = os.path.join(best_checkpoint, FILENAMES["checkpoint"])
    valid_metrics = ray.get(get_metrics.remote(cfg, best_model, "valid", ray_datasets))
    test_metrics = ray.get(get_metrics.remote(cfg, best_model, "test", ray_datasets))

    to_pickle(
        fs,
        os.path.join(best_checkpoint, FILENAMES["metrics"]),
        {
            "model_cfg": cfg.model,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        },
    )


if __name__ == "__main__":
    main()
