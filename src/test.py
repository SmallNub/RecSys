import json
import os
import pickle
import csv

import fsspec
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as L
from omegaconf import OmegaConf

from src.utils.tools import patch_fsspec
from src.data.datasets import get_quantized

FILENAMES = {
    "metrics": "metrics.pkl",
    "config": "config.yaml",
    "checkpoint": "checkpoint.ckpt",
    "snapshot": "checkpoint_manager_snapshot.json",
}


def get_top_cfg(fs, config):
    print(f"[STEP] Loading from {config.run_directory}")

    root = os.path.join(config.paths.model_folder_tplt, config.run_directory)
    snapshot_path = os.path.join(root, FILENAMES["snapshot"])

    if fs.exists(snapshot_path):
        with fs.open(snapshot_path, "r") as f:
            snapshot_data = json.load(f)
        latest_dir_name = snapshot_data["latest_checkpoint_result"][
            "checkpoint_dir_name"
        ]
        print(f"[INFO] Found latest sub-checkpoint from snapshot: {latest_dir_name}")
        target_dir = os.path.join(root, latest_dir_name)
    else:
        print(
            f"[WARNING] {FILENAMES['snapshot']} not found at {root}. Searching for sub-directories..."
        )
        try:
            # Fallback: Manually scan for the chronologically latest checkpoint subdirectory if snapshot metadata is missing
            subdirs = [
                os.path.basename(x.rstrip("/"))
                for x in fs.glob(os.path.join(root, "checkpoint_*"))
            ]
            if subdirs:
                latest_dir_name = sorted(subdirs)[-1]
                print(
                    f"[INFO] Fallback selected latest sub-directory: {latest_dir_name}"
                )
                target_dir = os.path.join(root, latest_dir_name)
            else:
                target_dir = root
        except Exception as e:
            print(f"[WARNING] Could not list subdirectories: {e}")
            target_dir = root

    cfg_path = os.path.join(target_dir, FILENAMES["config"])
    print(f"[INFO] Attempting to load config from: {cfg_path}")

    with fs.open(cfg_path) as f:
        cfg = OmegaConf.load(f)
    print(cfg.data)

    best_checkpoint = os.path.join(target_dir, FILENAMES["checkpoint"])
    return cfg, best_checkpoint


def get_metrics(cfg, ckpt_path, split, datasets, diversity_context=None):
    patch_fsspec()

    datamodule = hydra.utils.instantiate(cfg.data.datamodule, datasets=datasets)
    datamodule.setup(stage="test")

    model = hydra.utils.instantiate(cfg["model"])
    n_params = sum(p.numel() for p in model.parameters())
    model.full_hydra_config = cfg

    if split == "test" and diversity_context is not None:
        model.code_to_item = diversity_context.get("code_to_item")
        model.id_to_item = diversity_context.get("id_to_item")
        model.item_embeddings = diversity_context.get("item_embeddings")
        model.popularity_counts = diversity_context.get("popularity_counts")
        model.n_centroids = diversity_context.get("n_centroids")

    trainer = L.Trainer(accelerator="gpu", precision="bf16-mixed")

    fn = trainer.validate if split == "valid" else trainer.test

    metrics = fn(
        model, datamodule=datamodule, ckpt_path=cfg.paths.protocol + "://" + ckpt_path
    )

    metrics[0]["n_params"] = n_params

    return metrics


def build_code_to_item(fs, cfg):
    """Build reverse mapping: tuple(centroid_code) -> item_id from the quantized parquet."""
    quant_path = cfg.paths.semantic_ids_tplt.format(
        emb_method=cfg.data.datasets.emb_id,
        category=cfg.data.datasets.category,
        quant_method=cfg.data.datasets.quant_id,
    )
    quant_df = get_quantized(fs, quant_path)
    code_cols = sorted([c for c in quant_df.columns if c.startswith("L")])
    code_to_item = {}
    for item_id, row in quant_df.iterrows():
        code = tuple(int(row[c]) for c in code_cols)
        code_to_item[code] = item_id
    return code_to_item


def get_n_centroids(cfg):
    """Extract n_centroids per level from the saved config."""
    try:
        n = cfg.data.datasets.get("n_centroids", None)
        if n is not None:
            return n
    except Exception:
        pass
    print("[INFO] n_centroids not found in config, defaulting to 256.")
    return 256


def load_item_embeddings(fs, cfg):
    """Load raw item embeddings as a dict: item_id -> np.array."""
    emb_method = cfg.data.datasets.get("emb_id")
    if emb_method is None:
        emb_method = "sentence-t5-xl"
    emb_path = cfg.paths.embeddings_tplt.format(
        emb_method=emb_method,
        category=cfg.data.datasets.category,
    )
    emb_df = pd.read_parquet(emb_path, filesystem=fs)
    return {
        row["product_id"]: np.array(row["embedding"]) for _, row in emb_df.iterrows()
    }


def get_popularity_counts(fs, cfg):
    """Count item frequency in training timelines."""
    train_path = cfg.paths.timelines_tplt.format(
        category=cfg.data.datasets.category, split="train"
    )
    train_df = pd.read_parquet(train_path, filesystem=fs)
    counts = {}
    for timeline in train_df["timeline"]:
        for item in timeline:
            counts[item] = counts.get(item, 0) + 1
    return counts


def to_pickle(fs, path, data):
    with fs.open(path, "wb") as f:
        pickle.dump(data, f)


@hydra.main(version_base="1.3", config_path="../configs", config_name="test")
def main(test_config):
    patch_fsspec()
    fs = fsspec.filesystem(test_config.paths.protocol)

    cfg, best_checkpoint = get_top_cfg(fs, test_config)

    L.seed_everything(cfg.seed, workers=True)

    if test_config.enforce_filtering:
        cfg.model.net.filter_preds = True

    cfg.data.datasets.which = ["valid", "test"]
    datasets = hydra.utils.instantiate(cfg.data.datasets, paths=cfg.paths)

    from src.models import SpecialTokens

    first_ds = next(iter(datasets.values())) if datasets else None
    if (
        first_ds is not None
        and hasattr(first_ds, "pp")
        and hasattr(first_ds.pp, "item_to_id")
    ):
        vocab_size = len(first_ds.pp.item_to_id) + len(SpecialTokens)
        if (
            "model" in cfg
            and "net" in cfg.model
            and cfg.model.net.get("_target_") == "src.models.sasrec.SASRec"
        ):
            cfg.model.net.vocab_size = vocab_size
            print(
                f"[INFO] Overrode model.net.vocab_size in test to {vocab_size} for SASRec"
            )

    diversity_context = {}

    first_ds = next(iter(datasets.values())) if datasets else None
    if (
        first_ds is not None
        and hasattr(first_ds, "pp")
        and hasattr(first_ds.pp, "id_to_item")
    ):
        diversity_context["id_to_item"] = first_ds.pp.id_to_item
    else:
        diversity_context["id_to_item"] = None

    uses_quantization = (
        cfg.data.datasets.get("quant_id") is not None
        and cfg.data.datasets.get("emb_id") is not None
    )
    if uses_quantization:
        print("[INFO] Building diversity context (with quantization)...")
        diversity_context["code_to_item"] = build_code_to_item(fs, cfg)
        diversity_context["n_centroids"] = get_n_centroids(cfg)
    else:
        print("[INFO] No quantization (SASRec++ mode).")
        diversity_context["code_to_item"] = None
        diversity_context["n_centroids"] = None

    try:
        diversity_context["item_embeddings"] = load_item_embeddings(fs, cfg)
    except Exception as e:
        print(
            f"[WARNING] Could not load item embeddings: {e}. ILD metric will be disabled."
        )
        diversity_context["item_embeddings"] = None

    try:
        diversity_context["popularity_counts"] = get_popularity_counts(fs, cfg)
    except Exception as e:
        print(f"[WARNING] Could not load popularity counts: {e}")
        diversity_context["popularity_counts"] = None

    valid_metrics = get_metrics(cfg, best_checkpoint, "valid", datasets)

    test_metrics = get_metrics(
        cfg, best_checkpoint, "test", datasets, diversity_context=diversity_context
    )

    to_pickle(
        fs,
        os.path.join(os.path.dirname(best_checkpoint), FILENAMES["metrics"]),
        {
            "model_cfg": cfg.model,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        },
    )

    summary_path = os.path.join(
        test_config.paths.model_folder_tplt, "results_summary.csv"
    )
    category = cfg.data.datasets.category
    run_directory = test_config.run_directory

    rows = []
    for split, metrics in [("valid", valid_metrics), ("test", test_metrics)]:
        for k, v in metrics[0].items():
            rows.append(
                {
                    "run_directory": run_directory,
                    "category": category,
                    "split": split,
                    "metric": k,
                    "value": v,
                }
            )

    file_exists = os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run_directory", "category", "split", "metric", "value"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Results appended to {summary_path}")


if __name__ == "__main__":
    main()
