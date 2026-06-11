import json
import os
import pickle
import csv

import fsspec
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.utils.tools import patch_fsspec
from src.data.datasets import get_items_map, get_quantized

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
        latest_dir_name = snapshot_data["latest_checkpoint_result"]["checkpoint_dir_name"]
        print(f"[INFO] Found latest sub-checkpoint from snapshot: {latest_dir_name}")
        target_dir = os.path.join(root, latest_dir_name)
    else:
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

    cfg_path = os.path.join(target_dir, FILENAMES["config"])
    print(f"[INFO] Attempting to load config from: {cfg_path}")

    with fs.open(cfg_path) as f:
        cfg = OmegaConf.load(f)
    print(cfg.data)

    best_checkpoint = os.path.join(target_dir, FILENAMES["checkpoint"])
    return cfg, best_checkpoint


def get_metrics(cfg, ckpt_path, split, datasets):
    patch_fsspec()

    datamodule = hydra.utils.instantiate(cfg.data.datamodule, datasets=datasets)
    datamodule.setup(stage="test")

    model = hydra.utils.instantiate(cfg["model"])
    n_params = sum(p.numel() for p in model.parameters())
    model.full_hydra_config = cfg

    trainer = L.Trainer(accelerator="gpu", precision="bf16-mixed")

    fn = trainer.validate if split == "valid" else trainer.test

    metrics = fn(
        model, datamodule=datamodule, ckpt_path=cfg.paths.protocol + "://" + ckpt_path
    )

    metrics[0]["n_params"] = n_params

    return metrics


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


def get_diversity_metrics(
    model,
    dataset,
    id_to_item,
    item_embeddings,  # dict: item_id (str) -> np.array
    popularity_counts,  # dict: item_id (str) -> int
    k=10,
    batch_size=256,
    device="cuda",
):
    """
    Compute ILD, Gini, and Entropy over top-K recommendations.

    - ILD: mean pairwise cosine distance within each user's top-K list
    - Gini: inequality of item exposure across all recommendations
    - Entropy: entropy of item frequency distribution across all recommendations
    """
    model.eval()
    model.to(device)

    all_recommended_items = []  # flat list of all recommended item IDs
    all_ild_scores = []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # indices: B x K x L (K candidates, L depth codes per item)
            indices = model.net.search(batch, n_results=k)  # B x K x L

            B = indices.shape[0]
            for b in range(B):
                # Map code sequences back to item IDs
                rec_codes = indices[b].cpu().tolist()  # K x L
                rec_items = []
                for code in rec_codes:
                    # Look up item by matching code tuple in quantizer
                    item_id = id_to_item.get(tuple(code))
                    if item_id is not None:
                        rec_items.append(item_id)

                all_recommended_items.extend(rec_items)

                # ILD: mean pairwise cosine distance
                embs = np.array([
                    item_embeddings[item] for item in rec_items
                    if item in item_embeddings
                ])
                if len(embs) >= 2:
                    # Normalize
                    norms = np.linalg.norm(embs, axis=1, keepdims=True)
                    embs = embs / (norms + 1e-8)
                    sim_matrix = embs @ embs.T  # cosine similarity
                    dist_matrix = 1 - sim_matrix
                    # Mean of upper triangle (pairwise distances)
                    n = len(embs)
                    ild = dist_matrix[np.triu_indices(n, k=1)].mean()
                    all_ild_scores.append(ild)

    # --- ILD ---
    mean_ild = float(np.mean(all_ild_scores)) if all_ild_scores else 0.0

    # --- Popularity bias metrics ---
    # Count how many times each item was recommended
    rec_counts = {}
    for item in all_recommended_items:
        rec_counts[item] = rec_counts.get(item, 0) + 1

    counts = np.array(list(rec_counts.values()), dtype=float)
    counts_sorted = np.sort(counts)
    n = len(counts_sorted)

    # Gini coefficient
    if n > 1:
        cumulative = np.cumsum(counts_sorted)
        gini = (2 * np.sum((np.arange(1, n + 1)) * counts_sorted) /
                (n * cumulative[-1])) - (n + 1) / n
    else:
        gini = 0.0

    # Entropy
    probs = counts / counts.sum()
    entropy = float(-np.sum(probs * np.log(probs + 1e-8)))

    return {
        "diversity/ILD": mean_ild,
        "popularity_bias/Gini": float(gini),
        "popularity_bias/Entropy": entropy,
    }


def build_code_to_item(fs, cfg):
    """Build reverse mapping: tuple(code) -> item_id from the quantized parquet."""
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


def load_item_embeddings(fs, cfg):
    """Load raw item embeddings as a dict: item_id -> np.array."""
    emb_path = cfg.paths.embeddings_tplt.format(
        emb_method=cfg.data.datasets.emb_id,
        category=cfg.data.datasets.category,
    )
    emb_df = pd.read_parquet(emb_path, filesystem=fs)
    return {row["product_id"]: np.array(row["embedding"]) for _, row in emb_df.iterrows()}


def to_pickle(fs, path, data):
    with fs.open(path, "wb") as f:
        pickle.dump(data, f)


@hydra.main(version_base="1.3", config_path="../configs", config_name="test")
def main(test_config):
    patch_fsspec()
    fs = fsspec.filesystem(test_config.paths.protocol)

    # 1. Get config and checkpoint from previous run
    cfg, best_checkpoint = get_top_cfg(fs, test_config)

    if test_config.enforce_filtering:
        cfg.model.net.filter_preds = True

    # 2. Instantiate datasets
    cfg.data.datasets.which = ["valid", "test"]
    datasets = hydra.utils.instantiate(cfg.data.datasets, paths=cfg.paths)

    # 3. Standard accuracy metrics
    valid_metrics = get_metrics(cfg, best_checkpoint, "valid", datasets)
    test_metrics = get_metrics(cfg, best_checkpoint, "test", datasets)

    # 4. Diversity and popularity bias metrics
    print("[INFO] Computing diversity and popularity bias metrics...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = hydra.utils.instantiate(cfg["model"])
    ckpt = torch.load(
        cfg.paths.protocol + "://" + best_checkpoint,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)

    code_to_item = build_code_to_item(fs, cfg)
    item_embeddings = load_item_embeddings(fs, cfg)
    popularity_counts = get_popularity_counts(fs, cfg)

    diversity_metrics = get_diversity_metrics(
        model=model,
        dataset=datasets["test"],
        id_to_item=code_to_item,
        item_embeddings=item_embeddings,
        popularity_counts=popularity_counts,
        k=10,
        batch_size=256,
        device=device,
    )
    print(f"[INFO] Diversity metrics: {diversity_metrics}")
    test_metrics[0].update(diversity_metrics)

    # 5. Save pickle
    to_pickle(
        fs,
        os.path.join(os.path.dirname(best_checkpoint), FILENAMES["metrics"]),
        {
            "model_cfg": cfg.model,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        },
    )

    # 6. Append to shared summary CSV
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
        writer = csv.DictWriter(
            f, fieldnames=["run_directory", "category", "split", "metric", "value"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Results appended to {summary_path}")


if __name__ == "__main__":
    main()