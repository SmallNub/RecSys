import json
import os
import pickle
import csv

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

    # This is: outputs/checkpoints/marius/MARIUS_small_2026-06-05_21-10-57
    root = os.path.join(config.paths.model_folder_tplt, config.run_directory)

    # 1. Look for the snapshot registry file directly in the root folder
    snapshot_path = os.path.join(root, FILENAMES["snapshot"])

    if fs.exists(snapshot_path):
        with fs.open(snapshot_path, "r") as f:
            snapshot_data = json.load(f)

        # Pull the exact folder name designated as the latest checkpoint
        latest_dir_name = snapshot_data["latest_checkpoint_result"]["checkpoint_dir_name"]
        print(f"[INFO] Found latest sub-checkpoint from snapshot: {latest_dir_name}")

        # Target directory becomes: root/checkpoint_2026-06-05_22-02-14.786737
        target_dir = os.path.join(root, latest_dir_name)
    else:
        # Fallback: If snapshot JSON doesn't exist, search for any "checkpoint_*" directory inside root
        print(f"[WARNING] {FILENAMES['snapshot']} not found at {root}. Searching for sub-directories...")
        try:
            subdirs = [os.path.basename(x.rstrip('/')) for x in fs.glob(os.path.join(root, "checkpoint_*"))]
            if subdirs:
                latest_dir_name = sorted(subdirs)[-1]
                print(f"[INFO] Fallback selected latest sub-directory: {latest_dir_name}")
                target_dir = os.path.join(root, latest_dir_name)
            else:
                target_dir = root
        except Exception as e:
            print(f"[WARNING] Could not list subdirectories: {e}")
            target_dir = root

    # 2. Now load config.yaml from inside the correct sub-directory
    cfg_path = os.path.join(target_dir, FILENAMES["config"])
    print(f"[INFO] Attempting to load config from: {cfg_path}")

    with fs.open(cfg_path) as f:
        cfg = OmegaConf.load(f)
    print(cfg.data)

    # 3. Define the path to checkpoint.ckpt inside the sub-directory
    best_checkpoint = os.path.join(target_dir, FILENAMES["checkpoint"])

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

    # 3. Append results to shared summary CSV
    summary_path = os.path.join(
        test_config.paths.model_folder_tplt, "results_summary.csv"
    )
    category = cfg.data.datasets.category
    run_directory = test_config.run_directory

    rows = []
    for split, metrics in [("valid", valid_metrics), ("test", test_metrics)]:
        for k, v in metrics[0].items():
            rows.append({
                "run_directory": run_directory,
                "category": category,
                "split": split,
                "metric": k,
                "value": v,
            })

    file_exists = os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_directory", "category", "split", "metric", "value"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Results appended to {summary_path}")


if __name__ == "__main__":
    main()
