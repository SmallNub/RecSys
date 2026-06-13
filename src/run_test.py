"""
Standalone test script — loads specific checkpoints and runs metrics.
Usage:
    python src/run_test.py run_directory=MARIUS_small_2026-06-08_13-43-08
    python src/run_test.py run_directory=MARIUS_small_2026-06-08_13-43-08 debug=true
"""
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
from omegaconf import OmegaConf, DictConfig

from src.utils.tools import patch_fsspec
from src.data.datasets import get_quantized

FILENAMES = {
    "metrics": "metrics.pkl",
    "config": "config.yaml",
    "checkpoint": "checkpoint.ckpt",
    "snapshot": "checkpoint_manager_snapshot.json",
}


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def find_checkpoint_dir(fs, root):
    """Find the best checkpoint directory inside a run root."""
    snapshot_path = os.path.join(root, FILENAMES["snapshot"])

    if fs.exists(snapshot_path):
        with fs.open(snapshot_path, "r") as f:
            snapshot_data = json.load(f)
        latest_dir_name = snapshot_data["latest_checkpoint_result"]["checkpoint_dir_name"]
        print(f"[INFO] Found checkpoint from snapshot: {latest_dir_name}")
        return os.path.join(root, latest_dir_name)

    print(f"[WARNING] No snapshot found at {root}, searching for checkpoint_* dirs...")
    try:
        subdirs = [
            os.path.basename(x.rstrip('/'))
            for x in fs.glob(os.path.join(root, "checkpoint_*"))
        ]
        if subdirs:
            latest = sorted(subdirs)[-1]
            print(f"[INFO] Using latest sub-directory: {latest}")
            return os.path.join(root, latest)
    except Exception as e:
        print(f"[WARNING] Could not list subdirectories: {e}")

    return root


def load_checkpoint(run_directory, model_folder, fs):
    """
    Load config and checkpoint path for a given run directory name.
    Works for both MARIUS and SASRec++ since they share the same
    LitModule/checkpoint structure.

    Returns:
        cfg: OmegaConf config from the checkpoint
        ckpt_path: full path to checkpoint.ckpt
    """
    root = os.path.join(model_folder, run_directory)
    print(f"[INFO] Loading from: {root}")

    if not fs.exists(root):
        raise FileNotFoundError(f"Run directory not found: {root}")

    target_dir = find_checkpoint_dir(fs, root)

    cfg_path = os.path.join(target_dir, FILENAMES["config"])
    print(f"[INFO] Loading config from: {cfg_path}")
    with fs.open(cfg_path) as f:
        cfg = OmegaConf.load(f)

    ckpt_path = os.path.join(target_dir, FILENAMES["checkpoint"])
    print(f"[INFO] Checkpoint path: {ckpt_path}")

    return cfg, ckpt_path


# ---------------------------------------------------------------------------
# Diversity context builders
# ---------------------------------------------------------------------------

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
    print(f"[INFO] code_to_item size: {len(code_to_item)}")
    return code_to_item


def load_item_embeddings(fs, cfg):
    """Load raw item embeddings as a dict: item_id -> np.array."""
    emb_path = cfg.paths.embeddings_tplt.format(
        emb_method=cfg.data.datasets.emb_id,
        category=cfg.data.datasets.category,
    )
    emb_df = pd.read_parquet(emb_path, filesystem=fs)
    result = {row["product_id"]: np.array(row["embedding"]) for _, row in emb_df.iterrows()}
    print(f"[INFO] item_embeddings size: {len(result)}")
    return result


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
    print(f"[INFO] popularity_counts size: {len(counts)}")
    return counts


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def run_metrics(cfg, ckpt_path, split, datasets, diversity_context=None):
    """
    Run standard accuracy metrics and optionally diversity metrics.
    diversity_context is only used for split='test'.
    """
    patch_fsspec()

    datamodule = hydra.utils.instantiate(cfg.data.datamodule, datasets=datasets)
    datamodule.setup(stage="test")

    model = hydra.utils.instantiate(cfg["model"])
    n_params = sum(p.numel() for p in model.parameters())
    model.full_hydra_config = cfg

    # Inject diversity context for test split
    if split == "test" and diversity_context is not None:
        model.code_to_item = diversity_context["code_to_item"]
        model.item_embeddings = diversity_context["item_embeddings"]
        model.popularity_counts = diversity_context["popularity_counts"]

    trainer = L.Trainer(accelerator="gpu", precision="bf16-mixed")
    fn = trainer.validate if split == "valid" else trainer.test

    metrics = fn(
        model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )
    metrics[0]["n_params"] = n_params

    return metrics


def debug_code_lookup(model, batch, code_to_item, device):
    """
    Print debug info about code lookup using filter_preds=False
    to get clean codebook codes.
    """
    original_filter = model.net.filter_preds
    model.net.filter_preds = False

    with torch.no_grad():
        gen = model.net.search(batch, n_results=10)

    model.net.filter_preds = original_filter

    sample_code = tuple(gen[0][0].cpu().tolist())
    print(f"[DEBUG] gen shape: {gen.shape}")
    print(f"[DEBUG] gen dtype: {gen.dtype}")
    print(f"[DEBUG] gen min/max: {gen.min().item()}/{gen.max().item()}")
    print(f"[DEBUG] Sample gen code (user 0, item 0): {sample_code}")
    print(f"[DEBUG] code_to_item size: {len(code_to_item)}")
    sample_keys = list(code_to_item.keys())[:3]
    print(f"[DEBUG] Sample code_to_item keys: {sample_keys}")
    print(f"[DEBUG] Match found: {sample_code in code_to_item}")
    key_arr = np.array(list(code_to_item.keys()))
    print(f"[DEBUG] code_to_item key min/max per level: {key_arr.min(axis=0)}/{key_arr.max(axis=0)}")

    # Count how many of the 10 recommended codes map to real items
    hits = 0
    for code in gen[0].cpu().tolist():
        if tuple(code) in code_to_item:
            hits += 1
    print(f"[DEBUG] Codes mapping to real items (user 0): {hits}/10")


def to_pickle(fs, path, data):
    with fs.open(path, "wb") as f:
        pickle.dump(data, f)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="../configs", config_name="test")
def main(test_config: DictConfig):
    patch_fsspec()
    fs = fsspec.filesystem(test_config.paths.protocol)

    debug = test_config.get("debug", False)

    # 1. Load checkpoint
    cfg, ckpt_path = load_checkpoint(
        run_directory=test_config.run_directory,
        model_folder=test_config.paths.model_folder_tplt,
        fs=fs,
    )

    if test_config.enforce_filtering:
        cfg.model.net.filter_preds = True

    # 2. Instantiate datasets
    cfg.data.datasets.which = ["valid", "test"]
    datasets = hydra.utils.instantiate(cfg.data.datasets, paths=cfg.paths)

    # 3. Build diversity context (only if quantization is used i.e. MARIUS, not SASRec++)
    uses_quantization = (
        cfg.data.datasets.get("quant_id") is not None
        and cfg.data.datasets.get("emb_id") is not None
    )
    if uses_quantization:
        print("[INFO] Building diversity context...")
        diversity_context = {
            "code_to_item": build_code_to_item(fs, cfg),
            "item_embeddings": load_item_embeddings(fs, cfg),
            "popularity_counts": get_popularity_counts(fs, cfg),
        }
    else:
        print("[INFO] Skipping diversity context — no quantization (SASRec++ mode).")
        diversity_context = None

    # 4. Debug mode — run one forward pass and inspect code lookup, then exit
    if debug:
        if not uses_quantization:
            print("[DEBUG] No quantization — skipping code lookup debug.")
        else:
            print("[DEBUG] Running one forward pass to inspect code lookup...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = hydra.utils.instantiate(cfg["model"])
            model.full_hydra_config = cfg
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            model.to(device)

            from torch.utils.data import DataLoader
            loader = DataLoader(datasets["test"], batch_size=4, shuffle=False)
            batch = next(iter(loader))
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            debug_code_lookup(model, batch, diversity_context["code_to_item"], device)

        print("[DEBUG] Exiting after debug pass. Re-run without debug=true for full metrics.")
        return

    # 5. Run metrics
    valid_metrics = run_metrics(cfg, ckpt_path, "valid", datasets)
    test_metrics = run_metrics(cfg, ckpt_path, "test", datasets,
                               diversity_context=diversity_context)

    # 6. Save pickle
    to_pickle(
        fs,
        os.path.join(os.path.dirname(ckpt_path), FILENAMES["metrics"]),
        {
            "model_cfg": cfg.model,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        },
    )

    # 7. Append to shared summary CSV
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