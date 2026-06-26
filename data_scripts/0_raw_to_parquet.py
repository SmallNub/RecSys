"""Converts raw Amazon datasets into Parquet format for faster processing and training."""

import os
import pickle
import subprocess
from pathlib import Path

import fsspec
import hydra
import pandas as pd

from src.utils.tools import patch_fsspec


@hydra.main(
    config_path="../configs", config_name="0_raw_to_parquet", version_base="1.2"
)
def main(config):
    if not config.paths.get("skip_download", False):
        subprocess.run(
            [
                "python",
                "scripts/download_data.py",
                "--year",
                str(config.get("dataset_year", 2023)),
                "--base_dir",
                str(Path(config.paths.hf_data_folder).parent),
                "--categories",
                *config.categories,
            ],
            check=True,
        )

    wd = os.getcwd()
    for category in config.categories:
        process_category(category, config, wd)


def _sanitize(p):
    """Sanitizes price strings into floats, returning None on failure."""
    try:
        return float(p)
    except (ValueError, TypeError):
        return None


# Columns selected from Amazon 2023 metadata
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


def process_category(category, config, wd):
    """Processes a dataset category, standardizing timelines and metadata into parquet files."""
    # Original LFS pull per file (replaced by snapshot_download in main).
    # os.chdir(config.paths.tmp_lfs_folder)
    # for file in _category_files(category):
    #     subprocess.run(["git", "lfs", "pull", "-I", file])
    # os.chdir(wd)

    patch_fsspec()
    filesystem = fsspec.filesystem(config.paths.protocol)
    data_root = config.paths.root
    for subdir in ["data/timelines", "data/meta", "data/embeddings"]:
        filesystem.makedirs(data_root + "/" + subdir, exist_ok=True)

    dataset_year = config.get("dataset_year", 2023)
    if dataset_year == 2014:
        _process_category_2014(category, config, filesystem)
    else:
        _process_category_2023(category, config, filesystem)


# ---------------------------------------------------------------------------
# Amazon 2023
# ---------------------------------------------------------------------------


def _process_category_2023(category, config, filesystem):
    print(f"{category} [2023] - Converting and moving files.")
    git_root = Path(config.paths.hf_data_folder)

    train_tl = pd.read_csv(git_root / f"benchmark/5core/last_out/{category}.train.csv")
    # Valid and Test only contain 1 interaction per user
    valid_tl = pd.read_csv(git_root / f"benchmark/5core/last_out/{category}.valid.csv")
    test_tl = pd.read_csv(git_root / f"benchmark/5core/last_out/{category}.test.csv")

    assert set(valid_tl["user_id"]) == set(train_tl["user_id"])
    assert set(test_tl["user_id"]) == set(train_tl["user_id"])

    print(f"{category} [2023] - Training")
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

    print(f"{category} [2023] - Validation")
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

    print(f"{category} [2023] - Test")
    test_df = (
        test_tl.groupby("user_id").agg(list).rename({"parent_asin": "timeline"}, axis=1)
    )
    train_val_test_df = train_val_df.add(test_df)
    train_val_test_df.to_parquet(
        config.paths.timelines_tplt.format(category=category, split="test"),
        filesystem=filesystem,
    )

    print(f"{category} [2023] - Exporting unique items.")
    all_items = train_val_test_df.timeline.explode().unique().tolist()
    with filesystem.open(
        config.paths.unique_items_tplt.format(category=category), "wb"
    ) as f:
        pickle.dump(all_items, f)

    print(f"{category} [2023] - Formatting and filtering metadata items.")
    df = pd.read_json(
        git_root / f"raw/meta_categories/meta_{category}.jsonl", lines=True
    )
    df = df[META_SELECTED_COLS]
    df["price"] = df["price"].apply(_sanitize)
    df = df[df.parent_asin.isin(set(all_items))]
    print(
        f"{category} [2023] - {df.shape[0]} items for {train_val_test_df.shape[0]} users."
    )
    df.to_parquet(
        config.paths.meta_tplt.format(category=category),
        index=False,
        filesystem=filesystem,
    )


# ---------------------------------------------------------------------------
# Amazon 2014
# ---------------------------------------------------------------------------


def _loocv_split(ratings_df):
    """Leave-one-out split: last interaction = test, second-to-last = valid, rest = train."""
    ratings_df = ratings_df.sort_values(["reviewerID", "unixReviewTime"])
    train_rows, valid_rows, test_rows = [], [], []
    for user_id, group in ratings_df.groupby("reviewerID"):
        items = group["asin"].tolist()
        timestamps = group["unixReviewTime"].tolist()
        ratings = group["rating"].tolist()
        n = len(items)
        for i, (item, ts, rating) in enumerate(zip(items, timestamps, ratings)):
            row = {
                "user_id": user_id,
                "parent_asin": item,
                "timestamp": ts,
                "rating": rating,
            }
            if i == n - 1:
                test_rows.append(row)
            elif i == n - 2:
                valid_rows.append(row)
            else:
                train_rows.append(row)
    return (
        pd.DataFrame(train_rows),
        pd.DataFrame(valid_rows),
        pd.DataFrame(test_rows),
    )


def _process_category_2014(category, config, filesystem):
    print(f"{category} [2014] - Converting and moving files.")
    data_root_2014 = Path(config.paths.hf_data_folder_2014)

    # Load 5-core ratings (reviewerID, asin, overall, unixReviewTime)
    # Expected file: datasets/amazon-2014/reviews_{category}_5.json
    ratings_path = data_root_2014 / f"reviews_{category}_5.json"
    print(f"{category} [2014] - Reading ratings from {ratings_path}")
    ratings = pd.read_json(ratings_path, lines=True)[
        ["reviewerID", "asin", "overall", "unixReviewTime"]
    ].rename(columns={"overall": "rating"})

    # Leave-one-out split
    print(f"{category} [2014] - Splitting train/valid/test (LOOCV).")
    train_tl, valid_tl, test_tl = _loocv_split(ratings)

    print(f"{category} [2014] - Training")
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

    print(f"{category} [2014] - Validation")
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

    print(f"{category} [2014] - Test")
    test_df = (
        test_tl.groupby("user_id").agg(list).rename({"parent_asin": "timeline"}, axis=1)
    )
    train_val_test_df = train_val_df.add(test_df)
    train_val_test_df.to_parquet(
        config.paths.timelines_tplt.format(category=category, split="test"),
        filesystem=filesystem,
    )

    print(f"{category} [2014] - Exporting unique items.")
    all_items = train_val_test_df.timeline.explode().unique().tolist()
    with filesystem.open(
        config.paths.unique_items_tplt.format(category=category), "wb"
    ) as f:
        pickle.dump(all_items, f)

    # Load metadata (asin, title, price, brand, categories, description)
    # Expected file: datasets/amazon-2014/meta_{category}.json
    print(f"{category} [2014] - Formatting and filtering metadata items.")
    meta_path = data_root_2014 / f"meta_{category}.json"
    # Amazon 2014 meta uses Python dict syntax (single quotes), not valid JSON
    import ast

    records = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(ast.literal_eval(line))
                except Exception:
                    pass
    meta = pd.DataFrame(records)

    meta = meta.rename(columns={"asin": "parent_asin", "brand": "store"})
    meta["price"] = meta["price"].apply(_sanitize)
    meta["average_rating"] = None
    meta["rating_number"] = None
    meta["features"] = meta.get("description", pd.Series(dtype=object)).apply(
        lambda x: [x] if isinstance(x, str) and x else []
    )
    meta = meta[META_SELECTED_COLS]
    meta = meta[meta.parent_asin.isin(set(all_items))]
    print(
        f"{category} [2014] - {meta.shape[0]} items for {train_val_test_df.shape[0]} users."
    )
    meta.to_parquet(
        config.paths.meta_tplt.format(category=category),
        index=False,
        filesystem=filesystem,
    )


if __name__ == "__main__":
    main()
