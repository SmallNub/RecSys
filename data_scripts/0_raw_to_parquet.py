import os
import pickle
import subprocess
from pathlib import Path

import fsspec
import hydra
import pandas as pd
import ray

from src.utils.tools import patch_fsspec


@hydra.main(
    config_path="../configs", config_name="0_raw_to_parquet", version_base="1.2"
)
def main(config):
    # Get a trace of the entire repo
    if not os.path.exists(config.paths.tmp_lfs_folder):
        subprocess.run(
            [
                "git",
                "clone",
                "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023",
                config.paths.tmp_lfs_folder,
            ],
            env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"),
        )

    ray.init()
    print(ray.cluster_resources())
    wd = os.getcwd()
    refs = [
        process_category.remote(category, config, wd) for category in config.categories
    ]
    ray.get(refs)
    ray.shutdown()


def _sanitize(p):
    try:
        return float(p)
    except (ValueError, TypeError):
        return None


META_SELECTED_COLS = [
    "parent_asin",
    "title",
    "price",
    "average_rating",
    "rating_number",
    "features",
    "store",
    "categories",
]


@ray.remote(num_cpus=5)
def process_category(category, config, wd):
    os.chdir(config.paths.tmp_lfs_folder)

    for file in [
        f"raw/meta_categories/meta_{category}.jsonl",
        # f"raw/review_categories/{category}.jsonl",
        f"benchmark/5core/last_out/{category}.train.csv",
        f"benchmark/5core/last_out/{category}.valid.csv",
        f"benchmark/5core/last_out/{category}.test.csv",
    ]:
        subprocess.run(["git", "lfs", "pull", "-I", file])

    os.chdir(wd)
    print(f"{category} - Converting and moving files.")
    git_root = Path(config.paths.tmp_lfs_folder)

    data_root = config.paths.root

    patch_fsspec()
    filesystem = fsspec.filesystem(config.paths.protocol)
    if not filesystem.exists(data_root):
        filesystem.mkdir(data_root)
        filesystem.mkdir(data_root + "/timelines")
        filesystem.mkdir(data_root + "/meta")
        filesystem.mkdir(data_root + "/embeddings")

    # Aggregating timelines from individual interactions
    train_tl = pd.read_csv(git_root / f"benchmark/5core/last_out/{category}.train.csv")
    # Valid and Test only contain 1 interaction for each user
    valid_tl = pd.read_csv(git_root / f"benchmark/5core/last_out/{category}.valid.csv")
    test_tl = pd.read_csv(git_root / f"benchmark/5core/last_out/{category}.test.csv")

    assert set(valid_tl["user_id"]) == set(train_tl["user_id"])
    assert set(test_tl["user_id"]) == set(train_tl["user_id"])

    # Group and aggregate training reviews
    print(f"{category} - Training")
    train_df = (
        train_tl.sort_values("timestamp")
        .groupby("user_id")
        .agg(list)
        .rename({"parent_asin": "timeline"}, axis=1)
    )

    train_df.to_parquet(
        config.paths.timelines_tplt.format(category=category, split="train"),
        filesystem=filesystem,
    )

    # Group validation reviews by user and add them to training timelines
    print(f"{category} - Validation")
    valid_df = (
        valid_tl.groupby("user_id")
        .agg(list)
        .rename({"parent_asin": "timeline"}, axis=1)
    )

    train_val_df = train_df.add(valid_df)
    train_val_df.to_parquet(
        config.paths.timelines_tplt.format(category=category, split="valid"),
        filesystem=filesystem,
    )

    # Group and aggregate test reviews
    print(f"{category} - Test")
    test_df = (
        test_tl.groupby("user_id").agg(list).rename({"parent_asin": "timeline"}, axis=1)
    )

    train_val_test_df = train_val_df.add(test_df)
    train_val_test_df.to_parquet(
        config.paths.timelines_tplt.format(category=category, split="test"),
        filesystem=filesystem,
    )

    # Unique items
    print(f"{category} - Exporting unique items.")
    all_items = train_val_test_df.timeline.explode().unique().tolist()
    with filesystem.open(
        config.paths.unique_items_tplt.format(category=category), "wb"
    ) as f:
        pickle.dump(all_items, f)

    # Converting JSONl to parquet
    print(f"{category} - Formatting and filtering metadata items.")

    df = pd.read_json(
        git_root / f"raw/meta_categories/meta_{category}.jsonl", lines=True
    )
    df = df[META_SELECTED_COLS]
    df["price"] = df["price"].apply(_sanitize)
    df = df[df.parent_asin.isin(set(all_items))]
    print(
        f"{category} - {df.shape[0]} items left for {train_val_test_df.shape[0]} timelines."
    )
    df.to_parquet(
        config.paths.meta_tplt.format(category=category),
        index=False,
        filesystem=filesystem,
    )


if __name__ == "__main__":
    main()
