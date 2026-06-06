import json
import os
import pickle

import fsspec
import hydra
import pytorch_lightning as L
from omegaconf import OmegaConf

from src.utils.tools import patch_fsspec

FILENAMES = {
    "metrics": "metrics.pkl",
    "config": "config.yaml",
    "checkpoint": "checkpoint.ckpt",
    "snapshot": "checkpoint_manager_snapshot.json",
}


def get_top_cfg(fs, config):
    print(f"[STEP] Loading from {config.run_directory}")

    root = os.path.join(config.paths.model_folder_tplt, config.run_directory)

    # Get a config file to look for a best metric
    cfg_path = os.path.join(root, FILENAMES["config"])
    with fs.open(cfg_path) as f:
        cfg = OmegaConf.load(f)
    print(cfg.data)

    snapshot_path = os.path.join(root, FILENAMES["snapshot"])

    if fs.exists(snapshot_path):
        with fs.open(snapshot_path, "r") as f:
            snapshot_data = json.load(f)

        latest_dir_name = snapshot_data["latest_checkpoint_result"]["checkpoint_dir_name"]
        print(f"[INFO] Found latest sub-checkpoint from snapshot: {latest_dir_name}")

        best_checkpoint = os.path.join(root, latest_dir_name, FILENAMES["checkpoint"])
    else:
        print(f"[WARNING] {FILENAMES['snapshot']} not found in {root}. Falling back to root directory.")
        best_checkpoint = os.path.join(root, FILENAMES["checkpoint"])

    return cfg, best_checkpoint


def get_metrics(cfg, ckpt_path, split, datasets):
    patch_fsspec()

    # Load datamodule
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, datasets=datasets)
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
    patch_fsspec()
    fs = fsspec.filesystem(test_config.paths.protocol)

    # 1. Get the config file of the previous run.
    cfg, best_checkpoint = get_top_cfg(fs, test_config)

    if test_config.enforce_filtering:
        cfg.model.net.filter_preds = True

    # 2. Instantiate datasets
    cfg.data.datasets.which = ["valid", "test"]  # Not train.
    datasets = hydra.utils.instantiate(cfg.data.datasets, paths=cfg.paths)

    valid_metrics = get_metrics(cfg, best_checkpoint, "valid", datasets)
    test_metrics = get_metrics(cfg, best_checkpoint, "test", datasets)

    to_pickle(
        fs,
        os.path.join(os.path.dirname(best_checkpoint), FILENAMES["metrics"]),
        {
            "model_cfg": cfg.model,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        },
    )


if __name__ == "__main__":
    main()
