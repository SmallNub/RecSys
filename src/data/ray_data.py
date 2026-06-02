import pickle
import random

import fsspec
import hydra
import pandas as pd
import ray
import ray.data

from src.models import SpecialTokens
from src.utils.tools import patch_fsspec


def get_quantized(fs, path):
    df = pd.read_parquet(path, filesystem=fs)
    df = df.set_index("product_id")
    # Reorder : make sure to have L0, L1...
    sorted_cols = sorted([col for col in df.columns if col.startswith("L")])
    df = df[sorted_cols]

    return df


def get_items_map(fs, path):
    with fs.open(path, "rb") as f:
        items = pickle.load(f)

    m = {}
    m["item_to_id"] = {item: i + len(SpecialTokens) for i, item in enumerate(items)}
    m["id_to_item"] = {id: item for item, id in m["item_to_id"].items()}

    return m


COL_ORDER = ["user_id", "timeline", "rating", "timestamp"]


class ToRow:
    def __init__(self, timeline_ref):
        self.timelines_df = ray.get(timeline_ref)

    def __call__(self, _):
        idx = random.randint(0, len(self.timelines_df) - 1)
        row = self.timelines_df.values[idx]
        return {k: row[i] for i, k in enumerate(COL_ORDER)}


def _drop_len_1(row):
    return len(row["timeline"]) > 1


def make_pipeline(
    fs, split, path, quantizer_ref, items_ref, prepro_cfg, num_cpus, total_len=None
):
    pp_cfg = prepro_cfg.copy()
    pp_cls = hydra.utils.get_class(pp_cfg.pop("_cls_"))
    pp_cfg.update(
        {
            "quantizer_ref": quantizer_ref,
            "items_ref": items_ref,
            "split": split,
        }
    )

    if split == "train":
        assert (
            total_len is not None
        ), "total_len must be specified for the training dataset."
        df = pd.read_parquet(path, filesystem=fs).reset_index()[COL_ORDER]
        df = df[df.timeline.apply(len) > 1]

        timeline_ref = ray.put(df)

        ds = (
            ray.data.range(total_len)
            .map(
                ToRow,
                fn_constructor_kwargs={"timeline_ref": timeline_ref},
                concurrency=5,
            )
            .map(pp_cls, fn_constructor_kwargs=pp_cfg, concurrency=num_cpus)
        )

    else:
        # Just read the dataframe
        ds = (
            ray.data.read_parquet(path, filesystem=fs)
            .filter(fn=_drop_len_1)
            .map(pp_cls, fn_constructor_kwargs=pp_cfg, concurrency=num_cpus)
            .materialize()  # Compute once and store
        )

    return ds


def make_ray_dataset(
    emb_id,
    quant_id,
    category,
    prepro_cfg,
    total_len,
    num_cpus,
    paths,
    which=["train", "valid"],
):
    patch_fsspec()
    fs = fsspec.filesystem(paths.protocol)

    items_path = paths.unique_items_tplt.format(category=category)
    items_ref = ray.put(get_items_map(fs, items_path))

    if quant_id is not None:
        quantizer_path = paths.semantic_ids_tplt.format(
            emb_method=emb_id, category=category, quant_method=quant_id
        )
        quantizer_ref = ray.put(get_quantized(fs, quantizer_path))
    else:
        quantizer_ref = None

    datasets = {}

    if "train" in which:
        datasets["train"] = make_pipeline(
            fs=fs,
            split="train",
            path=paths.timelines_tplt.format(category=category, split="train"),
            quantizer_ref=quantizer_ref,
            items_ref=items_ref,
            prepro_cfg=prepro_cfg,
            total_len=total_len,
            num_cpus=num_cpus,
        )

    if "valid" in which:
        datasets["valid"] = make_pipeline(
            fs=fs,
            split="valid",
            path=paths.timelines_tplt.format(category=category, split="valid"),
            quantizer_ref=quantizer_ref,
            items_ref=items_ref,
            prepro_cfg=prepro_cfg,
            num_cpus=5,
        )

    if "test" in which:
        datasets["test"] = make_pipeline(
            fs=fs,
            split="test",
            path=paths.timelines_tplt.format(category=category, split="test"),
            quantizer_ref=quantizer_ref,
            items_ref=items_ref,
            prepro_cfg=prepro_cfg,
            num_cpus=5,
        )

    return datasets
